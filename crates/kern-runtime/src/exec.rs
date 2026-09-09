//! Execution: one program at one set of var values, onto the compute
//! stream.
//!
//! The names died at load (`compile`); what runs here is a flat launch
//! list whose slots are constants or var-indexed expressions, so a launch
//! is evaluation and `cuLaunchKernelEx`, no lookup and nothing left to
//! panic on. The caller's env is densified once per call into manifest
//! var order, the index space every compiled expression uses; vars the
//! program does not read densify to the minimum, so a graph is keyed by
//! the values that shaped it.
//!
//! A `graph` program is captured into a CUDA graph the first time it is
//! issued at a var assignment and replayed with one launch after. Grid
//! dims and scalar args are baked in at capture, input buffer contents
//! are read at replay: one program holds one graph per var assignment (a
//! batched decode keeps one per bucket) and per-step H2D writes stay
//! outside the graph.
//!
//! Ranks whose kernels wait on each other (an EP dispatch, a tray
//! collective) must all be issued before any is waited for: `enqueue` or
//! `issue` each, then [`Runtime::synchronize`] each.

use std::collections::BTreeMap;
use std::os::raw::c_void;

use cudarc::driver::sys;

use crate::compile::{CompiledProgram, Launch, LaunchKind, RVal, Slot};
use crate::device::{gemm_bf16_tn, gemm_bf16_tn_f32};
use crate::error::{bail, cuda_check};
use crate::{Error, Result, Runtime};

impl Runtime {
    /// Validate the caller's var values and densify them into manifest var
    /// order — the index space every compiled expression uses. A program
    /// needs the vars it reads (`used`); the rest are no part of it and
    /// densify to the minimum whatever the caller passed, so a graph is
    /// keyed by the values that shaped it.
    pub(crate) fn dense_env(&self, env: &BTreeMap<String, u64>, used: &[bool]) -> Result<Vec<u64>> {
        self.manifest
            .vars
            .iter()
            .zip(used)
            .map(|((var, decl), &used)| {
                if !used {
                    return Ok(kern_manifest::types::Var::MIN);
                }
                let Some(&v) = env.get(var) else {
                    bail!(Api, "var `{var}` not provided");
                };
                if v < kern_manifest::types::Var::MIN || v > decl.max {
                    bail!(Api, "var `{var}` = {v} outside declared [{}, {}]", kern_manifest::types::Var::MIN, decl.max);
                }
                Ok(v)
            })
            .collect()
    }

    /// `var=value` in manifest var order, for error messages.
    fn fmt_env(&self, env: &[u64]) -> String {
        self.manifest.vars.keys().zip(env).map(|(s, v)| format!("{s}={v}")).collect::<Vec<_>>().join(", ")
    }

    /// Execute one program with the given var values, then synchronize.
    pub fn run(&self, program: &str, env: &BTreeMap<String, u64>) -> Result<()> {
        self.enqueue(program, env)?;
        self.ctx.bind_to_thread()?;
        self.stream.synchronize()?;
        Ok(())
    }

    /// Launch every program eagerly from now on, ignoring the manifest's
    /// `graph`: a debug switch for telling a graph problem from a kernel one.
    pub fn set_eager(&mut self, eager: bool) {
        self.eager = eager;
    }

    /// Issue one program the way its manifest says, without waiting: a
    /// `graph` program through its CUDA graph, captured at these var
    /// values on first use; any other launch by launch (see
    /// [`Runtime::enqueue`] for the ordering ranks need).
    pub fn issue(&mut self, program: &str, env: &BTreeMap<String, u64>) -> Result<()> {
        let Some(prog) = self.programs.get(program) else {
            bail!(Api, "no program `{program}`");
        };
        if !prog.graph || self.eager {
            return self.enqueue(program, env);
        }
        if !self.is_captured(program, env) {
            let t = std::time::Instant::now();
            let calls = prog.call_ranges.len();
            self.capture(program, env)?;
            tracing::info!(program, env = ?env, calls, capture_ms = t.elapsed().as_millis() as u64, "graph captured");
        }
        self.enqueue_captured(program, env)
    }

    /// Issue one program's launches onto the stream and return without
    /// waiting. Ranks whose kernels wait on each other (an EP dispatch, a
    /// tray collective) must all be issued before any is waited for:
    /// `enqueue` each, then [`Runtime::synchronize`] each.
    pub fn enqueue(&self, program: &str, env: &BTreeMap<String, u64>) -> Result<()> {
        let Some(prog) = self.programs.get(program) else {
            bail!(Api, "no program `{program}`");
        };
        self.require_peers()?;
        let env = self.dense_env(env, &prog.vars)?;
        self.ctx.bind_to_thread()?;
        self.replay(prog, &env)
    }

    /// Capture one program into an instantiated CUDA graph. Grid dims and
    /// scalar args (var values included) are baked in at capture; input
    /// buffer *contents* are read at replay, so per-step H2D writes stay
    /// outside the graph and `run_captured` replays the whole call list
    /// with one launch.
    pub fn capture(&mut self, program: &str, env: &BTreeMap<String, u64>) -> Result<()> {
        let Some(prog) = self.programs.get(program) else {
            bail!(Api, "no program `{program}`");
        };
        self.require_peers()?;
        let env = self.dense_env(env, &prog.vars)?;
        self.ctx.bind_to_thread()?;
        cuda_check(
            unsafe {
                sys::cuStreamBeginCapture_v2(
                    self.stream.cu_stream(),
                    sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                )
            },
            "cuStreamBeginCapture",
        )?;
        let replayed = self.replay(prog, &env);
        // Always end the capture, even on error — a stream stuck in capture
        // mode poisons every later operation on it.
        let mut graph: sys::CUgraph = std::ptr::null_mut();
        let end = unsafe { sys::cuStreamEndCapture(self.stream.cu_stream(), &mut graph) };
        if let Err(e) = replayed {
            if !graph.is_null() {
                unsafe { sys::cuGraphDestroy(graph) };
            }
            return Err(e);
        }
        cuda_check(end, "cuStreamEndCapture")?;
        let mut exec: sys::CUgraphExec = std::ptr::null_mut();
        let r = unsafe { sys::cuGraphInstantiateWithFlags(&mut exec, graph, 0) };
        unsafe { sys::cuGraphDestroy(graph) };
        cuda_check(r, "cuGraphInstantiateWithFlags")?;
        if let Some(old) = self.graphs.insert((program.to_string(), env), exec) {
            unsafe { sys::cuGraphExecDestroy(old) };
        }
        Ok(())
    }

    /// Whether `capture(program, env)` has been done for exactly these var
    /// values.
    pub fn is_captured(&self, program: &str, env: &BTreeMap<String, u64>) -> bool {
        self.programs
            .get(program)
            .and_then(|prog| self.dense_env(env, &prog.vars).ok())
            .is_some_and(|env| self.graphs.contains_key(&(program.to_string(), env)))
    }

    /// The graph captured for (program, env), or an `Api` error naming the
    /// var values that were captured instead.
    pub(crate) fn graph(&self, program: &str, env: &BTreeMap<String, u64>) -> Result<sys::CUgraphExec> {
        let Some(prog) = self.programs.get(program) else {
            bail!(Api, "no program `{program}`");
        };
        let dense = self.dense_env(env, &prog.vars)?;
        if let Some(exec) = self.graphs.get(&(program.to_string(), dense.clone())) {
            return Ok(*exec);
        }
        let others: Vec<String> =
            self.graphs.keys().filter(|(p, _)| p == program).map(|(_, e)| format!("{{{}}}", self.fmt_env(e))).collect();
        if others.is_empty() {
            bail!(Api, "program `{program}` has not been captured");
        }
        bail!(Api, "program `{program}` called with {{{}}} but captured at {}", self.fmt_env(&dense), others.join(", "))
    }

    /// Replay a previously captured program, then synchronize.
    pub fn run_captured(&self, program: &str, env: &BTreeMap<String, u64>) -> Result<()> {
        self.enqueue_captured(program, env)?;
        self.stream.synchronize()?;
        Ok(())
    }

    /// Launch a previously captured program's graph without waiting (see
    /// [`Runtime::enqueue`]).
    pub fn enqueue_captured(&self, program: &str, env: &BTreeMap<String, u64>) -> Result<()> {
        let exec = self.graph(program, env)?;
        self.ctx.bind_to_thread()?;
        cuda_check(unsafe { sys::cuGraphLaunch(exec, self.stream.cu_stream()) }, "cuGraphLaunch")
    }

    /// Issue every launch of a compiled program onto the stream (no sync).
    pub(crate) fn replay(&self, prog: &CompiledProgram, env: &[u64]) -> Result<()> {
        for l in &prog.launches {
            self.launch(l, env).map_err(|e| Error::Call { context: l.ctx.clone(), source: Box::new(e) })?;
        }
        Ok(())
    }

    pub(crate) fn launch(&self, l: &Launch, env: &[u64]) -> Result<()> {
        // Materialize the slots; only var-dependent scalars are left to
        // compute, everything else was finished at load. Packs (and the
        // tensor maps inside them) ride along as pointers to their images.
        let mut vals = Vec::with_capacity(l.slots.len());
        let mut images: Vec<Option<Vec<u8>>> = Vec::with_capacity(l.slots.len());
        for s in &l.slots {
            let (v, m) = match s {
                Slot::Const(rv) => (*rv, None),
                Slot::Expr(e) => (RVal { val: e.eval(env)?, bytes: 0 }, None),
                Slot::Pack(p) => (RVal { val: 0, bytes: 0 }, Some(p.image(env)?)),
            };
            vals.push(v);
            images.push(m);
        }
        match &l.kind {
            LaunchKind::Gemm { beta } => gemm_bf16_tn(&self.blt, &self.stream, &vals, *beta),
            LaunchKind::GemmF32 => gemm_bf16_tn_f32(&self.blas, &vals),
            LaunchKind::Cubin { func, block, grid, shared_mem, cluster, pdl } => {
                let grid = [grid[0].eval(env)? as u32, grid[1].eval(env)? as u32, grid[2].eval(env)? as u32];
                let smem = match shared_mem {
                    Some(e) => e.eval(env)? as u32,
                    None => 0,
                };
                // Every scalar/pointer slot staged as a little-endian u64;
                // the launch ABI reads the low `size_bytes()` of each slot.
                // A pack slot points at its image instead.
                let raw: Vec<u64> = vals.iter().map(|r| r.val).collect();
                let mut params: Vec<*mut c_void> = raw
                    .iter()
                    .zip(&images)
                    .map(|(s, m)| match m {
                        Some(b) => b.as_ptr() as *mut c_void,
                        None => s as *const u64 as *mut c_void,
                    })
                    .collect();
                let blank = sys::CUlaunchAttribute {
                    id: sys::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION,
                    pad: [0; 4],
                    value: sys::CUlaunchAttributeValue { pad: [0; 64] },
                };
                let mut attrs = [blank; 2];
                let mut num_attrs = 0;
                if let Some(c) = cluster {
                    attrs[num_attrs].value.clusterDim =
                        sys::CUlaunchAttributeValue_union__bindgen_ty_1 { x: c[0], y: c[1], z: c[2] };
                    num_attrs += 1;
                }
                // Under stream capture this becomes a programmatic edge of
                // the graph: the kernel is launched as its predecessor
                // drains and does its own `griddepcontrol.wait`.
                if *pdl {
                    attrs[num_attrs].id =
                        sys::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
                    attrs[num_attrs].value.programmaticStreamSerializationAllowed = 1;
                    num_attrs += 1;
                }
                let cfg = sys::CUlaunchConfig {
                    gridDimX: grid[0],
                    gridDimY: grid[1],
                    gridDimZ: grid[2],
                    blockDimX: block[0],
                    blockDimY: block[1],
                    blockDimZ: block[2],
                    sharedMemBytes: smem,
                    hStream: self.stream.cu_stream(),
                    attrs: attrs.as_mut_ptr(),
                    numAttrs: num_attrs as u32,
                };
                cuda_check(
                    unsafe { sys::cuLaunchKernelEx(&cfg, *func, params.as_mut_ptr(), std::ptr::null_mut()) },
                    "cuLaunchKernelEx",
                )
            }
        }
    }
}
