//! Thin verifier-driven executor for kern manifests.
//!
//! The runtime knows nothing about models. It loads a verified manifest,
//! resolves each declared kernel against the cubins in a directory, allocates
//! every buffer/state, binds weight buffers by name from a safetensors blob,
//! and replays the program's dispatch list. The only kernels it understands
//! natively are `extern:` ops (currently `extern:cublaslt_bf16_tn`).
//!
//! Names stop at load time: device pointers are static once buffers, states
//! and scratch are allocated, so [`Runtime::load`] lowers every program into
//! a flat launch list (see `compile`) whose slots are finished values or
//! symbol-indexed expressions. The name-keyed maps that remain exist only on
//! the caller API surface (`write_input("token_ids")`, `run("decode")`);
//! the execution path performs no name lookups.
//!
//! Same-name Triton kernels ship multiple constexpr instances with different
//! ABIs across modules; resolution picks the instance whose
//! `cuFuncGetParamInfo` layout matches the manifest's declared params — the
//! phase-2 ABI check doubles as instance selection.

mod compile;
mod cubin;
mod device;
mod error;
pub mod values;

use std::collections::BTreeMap;
use std::os::raw::c_void;
use std::sync::Arc;

use cudarc::cublaslt::CudaBlasLT;
use cudarc::driver::{result as cu, sys, CudaContext, CudaStream, PinnedHostSlice};
use kern_manifest::types::{BufferClass, Manifest};

use compile::{CompiledProgram, Launch, LaunchKind, RVal, Slot};
use device::{alloc, gemm_bf16_tn, DeviceBuf};
use error::{bail, cuda_check};
pub use error::{Error, Result};

pub struct Runtime {
    pub manifest: Manifest,
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blt: CudaBlasLT,
    /// Token capacity every state was provisioned for (`index_into` a
    /// state resolves against it).
    capacity: u64,
    /// Name-keyed because names are the caller API (`write_input`,
    /// `read_output`, weight binding); execution never looks these up —
    /// their device pointers are baked into `programs`.
    buffers: BTreeMap<String, DeviceBuf>,
    states: BTreeMap<String, DeviceBuf>,
    /// Persistent pinned staging, one per input buffer: H2D from pageable
    /// memory degrades to a synchronous driver-staged copy (tens of µs per
    /// call); through page-locked staging it is a true async DMA. The pinned
    /// slice's event guards reuse across steps.
    staging: BTreeMap<String, PinnedHostSlice<u8>>,
    /// Programs lowered to flat launch lists at load.
    programs: BTreeMap<String, CompiledProgram>,
    /// Owners of the impl-private scratch allocations whose pointers are
    /// baked into `programs`.
    #[allow(dead_code)]
    scratch: Vec<DeviceBuf>,
    /// Per kernel: the module each impl step resolved to (introspection).
    resolution: Vec<(String, Vec<String>)>,
    n_modules: usize,
    /// Program name -> instantiated CUDA graph + the dense symbol values it
    /// was captured with (grid dims and scalar args are baked in at capture).
    graphs: BTreeMap<String, (sys::CUgraphExec, Vec<u64>)>,
}

impl Drop for Runtime {
    fn drop(&mut self) {
        for (exec, _) in self.graphs.values() {
            unsafe { sys::cuGraphExecDestroy(*exec) };
        }
    }
}

/// A pool of timing events (attestation only).
struct Events(Vec<sys::CUevent>);

impl Events {
    fn new(n: usize) -> Result<Events> {
        let mut evs = Vec::with_capacity(n);
        for _ in 0..n {
            let mut ev: sys::CUevent = std::ptr::null_mut();
            cuda_check(unsafe { sys::cuEventCreate(&mut ev, 0) }, "cuEventCreate")?;
            evs.push(ev);
        }
        Ok(Events(evs))
    }

    fn record(&self, i: usize, stream: &CudaStream) -> Result<()> {
        cuda_check(unsafe { sys::cuEventRecord(self.0[i], stream.cu_stream()) }, "cuEventRecord")
    }

    fn elapsed_ms(&self, a: usize, b: usize) -> Result<f32> {
        let mut ms = 0f32;
        cuda_check(
            unsafe { sys::cuEventElapsedTime_v2(&mut ms, self.0[a], self.0[b]) },
            "cuEventElapsedTime",
        )?;
        Ok(ms)
    }
}

impl Drop for Events {
    fn drop(&mut self) {
        for ev in &self.0 {
            unsafe { sys::cuEventDestroy_v2(*ev) };
        }
    }
}

impl Runtime {
    /// Verify the manifest, load every `*.cubin` under `kernels_dir`, resolve
    /// kernels, allocate all buffers and states, and lower every program.
    /// `state_capacity_tokens` scales each declared state by its
    /// `bytes_per_token` (a `bytes_fixed` state is allocated as declared).
    pub fn load(
        manifest_json: &str,
        kernels_dir: &std::path::Path,
        gpu: usize,
        state_capacity_tokens: u64,
    ) -> Result<Runtime> {
        let manifest = Manifest::from_json(manifest_json)?;
        kern_manifest::verify(&manifest)?;

        let ctx = CudaContext::new(gpu)?;
        // A created (non-legacy) stream: the NULL stream cannot be captured
        // into a CUDA graph.
        let stream = ctx.new_stream()?;
        let blt = CudaBlasLT::new(stream.clone())?;
        ctx.bind_to_thread()?;

        let remote = cubin::fetch_registry_cubins(&manifest)?;
        let modules = cubin::load_all_modules(kernels_dir, &remote)?;

        let max_env: BTreeMap<_, _> =
            manifest.symbols.iter().map(|(s, v)| (s.clone(), v.max)).collect();

        // Buffer sizes are static: shapes only reference symbols, sized at max.
        let mut buffers = BTreeMap::new();
        for (name, b) in &manifest.buffers {
            let bytes = compile::shaped_bytes(
                &format!("buffer `{name}`"),
                &b.shape,
                b.dtype.bytes(),
                &max_env,
            )?;
            buffers.insert(name.clone(), alloc(&stream, bytes)?);
        }
        let mut states = BTreeMap::new();
        for (name, s) in &manifest.states {
            let bytes = s
                .bytes_per_token
                .checked_mul(state_capacity_tokens)
                .and_then(|b| b.checked_add(s.bytes_fixed))
                .ok_or_else(|| Error::Manifest(format!("state `{name}`: size overflow")))?;
            states.insert(name.clone(), alloc(&stream, bytes)?);
        }
        let mut staging = BTreeMap::new();
        for (name, b) in &manifest.buffers {
            if b.class == BufferClass::Input {
                let mut pinned =
                    unsafe { ctx.alloc_pinned::<u8>(buffers[name].bytes.max(1) as usize)? };
                pinned.as_mut_slice()?.fill(0);
                staging.insert(name.clone(), pinned);
            }
        }

        let resolved =
            compile::resolve_kernels(&manifest, &modules, &remote, kernels_dir, &stream, &max_env)?;
        let resolution = resolved.iter().map(|(n, rk)| (n.clone(), rk.step_modules())).collect();
        let programs = compile::compile_programs(&manifest, &resolved, &buffers, &states)?;
        let scratch = resolved.into_values().flat_map(|rk| rk.scratch.into_values()).collect();

        Ok(Runtime {
            manifest,
            ctx,
            stream,
            blt,
            capacity: state_capacity_tokens,
            buffers,
            states,
            staging,
            programs,
            scratch,
            resolution,
            n_modules: modules.len(),
            graphs: BTreeMap::new(),
        })
    }

    pub fn module_count(&self) -> usize {
        self.n_modules
    }

    /// (name, class, allocated bytes) for every buffer.
    pub fn buffer_sizes(&self) -> Vec<(&str, BufferClass, u64)> {
        self.manifest
            .buffers
            .iter()
            .map(|(n, b)| (n.as_str(), b.class, self.buffers[n].bytes))
            .collect()
    }

    /// (name, bytes_per_token, allocated bytes) for every state.
    pub fn state_sizes(&self) -> Vec<(&str, u64, u64)> {
        self.manifest
            .states
            .iter()
            .map(|(n, s)| (n.as_str(), s.bytes_per_token, self.states[n].bytes))
            .collect()
    }

    /// Per kernel: the module each impl step resolved to, in step order.
    pub fn kernel_resolution(&self) -> Vec<(String, Vec<String>)> {
        self.resolution.clone()
    }

    /// Bind every `weight` buffer by name from one or more safetensors blobs
    /// (a target and a draft artifact, say); each weight must come from
    /// exactly one of them.
    pub fn load_weights(&mut self, blobs: &[&[u8]]) -> Result<()> {
        let sts = blobs
            .iter()
            .map(|b| safetensors::SafeTensors::deserialize(b))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| Error::WeightArtifact(format!("unparseable safetensors: {e}")))?;
        for (name, b) in &self.manifest.buffers {
            if b.class != BufferClass::Weight {
                continue;
            }
            let found: Vec<_> = sts.iter().filter_map(|st| st.tensor(name).ok()).collect();
            let t = match found.as_slice() {
                [t] => t,
                [] => bail!(WeightArtifact, "weight `{name}` missing from the artifact(s)"),
                _ => bail!(WeightArtifact, "weight `{name}` present in {} artifacts", found.len()),
            };
            let dst = self.buffers.get_mut(name).unwrap();
            if t.data().len() as u64 != dst.bytes {
                bail!(
                    WeightArtifact,
                    "weight `{name}`: artifact has {} bytes, manifest declares {}",
                    t.data().len(),
                    dst.bytes
                );
            }
            self.stream.memcpy_htod(t.data(), &mut dst.slice)?;
        }
        self.stream.synchronize()?;
        Ok(())
    }

    /// Check `data` (a prefix of buffer `name`) against the buffer's declared
    /// domain, if any, at the given symbol values. Symbol-dependent bounds
    /// need `env`; pass the values the next run will use.
    pub fn check_domain(&self, name: &str, data: &[u8], env: &BTreeMap<String, u64>) -> Result<()> {
        let Some(b) = self.manifest.buffers.get(name) else {
            bail!(Api, "no buffer `{name}`");
        };
        let Some(d) = &b.domain else { return Ok(()) };
        let r = d
            .resolve(&self.manifest, env, self.capacity)
            .map_err(|e| Error::Domain(format!("buffer `{name}`: {e}")))?;
        let vals = values::to_f64(b.dtype, data);
        let fmt_bound = |v: Option<f64>| v.map_or("∞".to_string(), |x| format!("{x}"));
        for (i, &v) in vals.iter().enumerate() {
            if !r.contains(v) {
                bail!(
                    Domain,
                    "buffer `{name}`[{i}] = {v} outside declared [{}, {}]",
                    fmt_bound(r.lo),
                    fmt_bound(r.hi)
                );
            }
            if r.monotone && i > 0 && v < vals[i - 1] {
                bail!(Domain, "buffer `{name}` is declared monotone but [{i}] = {v} < [{}] = {}", i - 1, vals[i - 1]);
            }
        }
        Ok(())
    }

    /// Write an input buffer. The domain check needs the symbol values the
    /// next run will use; `write_input` checks against symbol upper bounds
    /// (the loosest valid reading), `write_input_at` against exact values.
    pub fn write_input(&mut self, name: &str, data: &[u8]) -> Result<()> {
        let max_env: BTreeMap<_, _> =
            self.manifest.symbols.iter().map(|(s, v)| (s.clone(), v.max)).collect();
        self.write_input_at(name, data, &max_env)
    }

    pub fn write_input_at(&mut self, name: &str, data: &[u8], env: &BTreeMap<String, u64>) -> Result<()> {
        let Some(b) = self.manifest.buffers.get(name) else {
            bail!(Api, "no buffer `{name}`");
        };
        if b.class != BufferClass::Input {
            bail!(Api, "buffer `{name}` is {}, not input", b.class);
        }
        if data.len() as u64 > self.buffers[name].bytes {
            bail!(Api, "input `{name}`: got {} bytes, buffer is {}", data.len(), self.buffers[name].bytes);
        }
        self.check_domain(name, data, env)?;
        let dst = self.buffers.get_mut(name).unwrap();
        let pinned = self.staging.get_mut(name).unwrap();
        // Waits on the pinned slice's event: the previous step's DMA from
        // this staging must finish before we overwrite it. A prefix write
        // (variable-length inputs) still DMAs the whole buffer — the stale
        // tail is never read, grids are bounded by the symbols.
        pinned.as_mut_slice()?[..data.len()].copy_from_slice(data);
        self.stream.memcpy_htod(pinned, &mut dst.slice)?;
        Ok(())
    }

    pub fn read_output(&self, name: &str) -> Result<Vec<u8>> {
        let Some(b) = self.manifest.buffers.get(name) else {
            bail!(Api, "no buffer `{name}`");
        };
        if b.class != BufferClass::Output {
            bail!(Api, "buffer `{name}` is {}, not output", b.class);
        }
        Ok(self.stream.clone_dtoh(&self.buffers[name].slice)?)
    }

    // ---- attestation surface: whole-buffer access, partial replay, timing.
    // Nothing here is on the serving path; every call synchronizes.

    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    pub fn dispatch_count(&self, program: &str) -> Result<usize> {
        match self.programs.get(program) {
            Some(p) => Ok(p.dispatch_ranges.len()),
            None => bail!(Api, "no program `{program}`"),
        }
    }

    /// Whole allocation of any buffer, regardless of class.
    pub fn read_buffer(&self, name: &str) -> Result<Vec<u8>> {
        let Some(b) = self.buffers.get(name) else {
            bail!(Api, "no buffer `{name}`");
        };
        Ok(self.stream.clone_dtoh(&b.slice)?)
    }

    /// The first `bytes` of any buffer (the live prefix at a symbol value
    /// below the allocation bound).
    pub fn read_buffer_prefix(&self, name: &str, bytes: usize) -> Result<Vec<u8>> {
        let Some(b) = self.buffers.get(name) else {
            bail!(Api, "no buffer `{name}`");
        };
        if bytes as u64 > b.bytes {
            bail!(Api, "buffer `{name}`: prefix {bytes} exceeds allocation {}", b.bytes);
        }
        let view = b.slice.slice(0..bytes);
        Ok(self.stream.clone_dtoh(&view)?)
    }

    /// Overwrite a prefix of any buffer, regardless of class (synchronous).
    pub fn write_buffer(&mut self, name: &str, data: &[u8]) -> Result<()> {
        let Some(b) = self.buffers.get_mut(name) else {
            bail!(Api, "no buffer `{name}`");
        };
        if data.len() as u64 > b.bytes {
            bail!(Api, "buffer `{name}`: got {} bytes, buffer is {}", data.len(), b.bytes);
        }
        let mut view = b.slice.slice_mut(0..data.len());
        self.stream.memcpy_htod(data, &mut view)?;
        self.stream.synchronize()?;
        Ok(())
    }

    /// Whole allocation of a state.
    pub fn read_state(&self, name: &str) -> Result<Vec<u8>> {
        let Some(s) = self.states.get(name) else {
            bail!(Api, "no state `{name}`");
        };
        Ok(self.stream.clone_dtoh(&s.slice)?)
    }

    /// Execute dispatches `[lo, hi)` of a program eagerly, then synchronize.
    pub fn run_range(
        &self,
        program: &str,
        env: &BTreeMap<String, u64>,
        lo: usize,
        hi: usize,
    ) -> Result<()> {
        let Some(prog) = self.programs.get(program) else {
            bail!(Api, "no program `{program}`");
        };
        let n = prog.dispatch_ranges.len();
        if lo > hi || hi > n {
            bail!(Api, "program `{program}`: dispatch range [{lo}, {hi}) outside 0..{n}");
        }
        let env = self.dense_env(env)?;
        self.ctx.bind_to_thread()?;
        if lo < hi {
            let (l0, _) = prog.dispatch_ranges[lo];
            let (_, l1) = prog.dispatch_ranges[hi - 1];
            for l in &prog.launches[l0..l1] {
                self.launch(l, &env).map_err(|e| Error::Dispatch {
                    context: l.ctx.clone(),
                    source: Box::new(e),
                })?;
            }
        }
        self.stream.synchronize()?;
        Ok(())
    }

    /// Per-dispatch GPU time in ms (eager, event-bracketed), minimum over
    /// `iters` replays of the whole program. Note this attributes launch
    /// gaps to the dispatch that follows them.
    pub fn time_dispatches(
        &self,
        program: &str,
        env: &BTreeMap<String, u64>,
        iters: usize,
    ) -> Result<Vec<f32>> {
        let n = self.dispatch_count(program)?;
        self.time_range(program, env, 0, n, iters)
    }

    /// Same, for dispatches `[lo, hi)` only — replaying just that range, so
    /// a cut can be timed without the rest of the program.
    pub fn time_range(
        &self,
        program: &str,
        env: &BTreeMap<String, u64>,
        lo: usize,
        hi: usize,
        iters: usize,
    ) -> Result<Vec<f32>> {
        let Some(prog) = self.programs.get(program) else {
            bail!(Api, "no program `{program}`");
        };
        if lo > hi || hi > prog.dispatch_ranges.len() {
            bail!(Api, "program `{program}`: dispatch range [{lo}, {hi}) outside 0..{}", prog.dispatch_ranges.len());
        }
        let env = self.dense_env(env)?;
        self.ctx.bind_to_thread()?;
        let n = hi - lo;
        let events = Events::new(n + 1)?;
        let mut best = vec![f32::INFINITY; n];
        for _ in 0..iters.max(1) {
            events.record(0, &self.stream)?;
            for (di, &(l0, l1)) in prog.dispatch_ranges[lo..hi].iter().enumerate() {
                for l in &prog.launches[l0..l1] {
                    self.launch(l, &env).map_err(|e| Error::Dispatch {
                        context: l.ctx.clone(),
                        source: Box::new(e),
                    })?;
                }
                events.record(di + 1, &self.stream)?;
            }
            self.stream.synchronize()?;
            for (di, b) in best.iter_mut().enumerate() {
                *b = b.min(events.elapsed_ms(di, di + 1)?);
            }
        }
        Ok(best)
    }

    /// Median wall time per replay of a captured program, in ms, over
    /// `iters` back-to-back graph launches.
    pub fn time_captured(
        &self,
        program: &str,
        env: &BTreeMap<String, u64>,
        iters: usize,
    ) -> Result<f32> {
        let Some((exec, captured)) = self.graphs.get(program) else {
            bail!(Api, "program `{program}` has not been captured");
        };
        let env = self.dense_env(env)?;
        if *captured != env {
            bail!(Api, "graph for `{program}` was captured with different symbol values");
        }
        self.ctx.bind_to_thread()?;
        let iters = iters.max(1);
        let events = Events::new(iters + 1)?;
        events.record(0, &self.stream)?;
        for i in 0..iters {
            cuda_check(
                unsafe { sys::cuGraphLaunch(*exec, self.stream.cu_stream()) },
                "cuGraphLaunch",
            )?;
            events.record(i + 1, &self.stream)?;
        }
        self.stream.synchronize()?;
        let mut ts: Vec<f32> = (0..iters).map(|i| events.elapsed_ms(i, i + 1)).collect::<Result<_>>()?;
        ts.sort_by(|a, b| a.total_cmp(b));
        Ok(ts[iters / 2])
    }

    /// Validate the caller's symbol values and densify them into manifest
    /// symbol order — the index space every compiled expression uses.
    fn dense_env(&self, env: &BTreeMap<String, u64>) -> Result<Vec<u64>> {
        self.manifest
            .symbols
            .iter()
            .map(|(sym, decl)| {
                let Some(&v) = env.get(sym) else {
                    bail!(Api, "symbol `{sym}` not provided");
                };
                if v < decl.min || v > decl.max {
                    bail!(Api, "symbol `{sym}` = {v} outside declared [{}, {}]", decl.min, decl.max);
                }
                Ok(v)
            })
            .collect()
    }

    /// `sym=value` in manifest symbol order, for error messages.
    fn fmt_env(&self, env: &[u64]) -> String {
        self.manifest
            .symbols
            .keys()
            .zip(env)
            .map(|(s, v)| format!("{s}={v}"))
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Execute one program with the given symbol values, then synchronize.
    pub fn run(&self, program: &str, env: &BTreeMap<String, u64>) -> Result<()> {
        let Some(prog) = self.programs.get(program) else {
            bail!(Api, "no program `{program}`");
        };
        let env = self.dense_env(env)?;
        self.ctx.bind_to_thread()?;
        self.replay(prog, &env)?;
        self.stream.synchronize()?;
        Ok(())
    }

    /// Capture one program into an instantiated CUDA graph. Grid dims and
    /// scalar args (symbol values included) are baked in at capture; input
    /// buffer *contents* are read at replay, so per-step H2D writes stay
    /// outside the graph and `run_captured` replays the whole dispatch list
    /// with one launch.
    pub fn capture(&mut self, program: &str, env: &BTreeMap<String, u64>) -> Result<()> {
        let Some(prog) = self.programs.get(program) else {
            bail!(Api, "no program `{program}`");
        };
        let env = self.dense_env(env)?;
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
        if let Some((old, _)) = self.graphs.insert(program.to_string(), (exec, env)) {
            unsafe { sys::cuGraphExecDestroy(old) };
        }
        Ok(())
    }

    /// Replay a previously captured program, then synchronize.
    pub fn run_captured(&self, program: &str, env: &BTreeMap<String, u64>) -> Result<()> {
        let Some((exec, captured)) = self.graphs.get(program) else {
            bail!(Api, "program `{program}` has not been captured");
        };
        let env = self.dense_env(env)?;
        if *captured != env {
            bail!(
                Api,
                "graph for `{program}` was captured with {{{}}}, called with {{{}}}",
                self.fmt_env(captured),
                self.fmt_env(&env)
            );
        }
        self.ctx.bind_to_thread()?;
        cuda_check(
            unsafe { sys::cuGraphLaunch(*exec, self.stream.cu_stream()) },
            "cuGraphLaunch",
        )?;
        self.stream.synchronize()?;
        Ok(())
    }

    /// Issue every launch of a compiled program onto the stream (no sync).
    fn replay(&self, prog: &CompiledProgram, env: &[u64]) -> Result<()> {
        for l in &prog.launches {
            self.launch(l, env).map_err(|e| Error::Dispatch {
                context: l.ctx.clone(),
                source: Box::new(e),
            })?;
        }
        Ok(())
    }

    fn launch(&self, l: &Launch, env: &[u64]) -> Result<()> {
        // Materialize the slots; only symbol-dependent scalars are left to
        // compute, everything else was finished at load.
        let mut vals = Vec::with_capacity(l.slots.len());
        for s in &l.slots {
            vals.push(match s {
                Slot::Const(rv) => *rv,
                Slot::Expr(e) => RVal { val: e.eval(env)?, bytes: 0 },
            });
        }
        match &l.kind {
            LaunchKind::Gemm { beta } => gemm_bf16_tn(&self.blt, &self.stream, &vals, *beta),
            LaunchKind::Cubin { func, block, grid, shared_mem, pdl } => {
                let grid =
                    (grid[0].eval(env)? as u32, grid[1].eval(env)? as u32, grid[2].eval(env)? as u32);
                let smem = match shared_mem {
                    Some(e) => e.eval(env)? as u32,
                    None => 0,
                };
                // Every param slot staged as a little-endian u64; the launch
                // ABI reads the low `size_bytes()` of each slot.
                let raw: Vec<u64> = vals.iter().map(|r| r.val).collect();
                let mut params: Vec<*mut c_void> =
                    raw.iter().map(|s| s as *const u64 as *mut c_void).collect();
                if *pdl {
                    // Programmatic dependent launch: inside stream capture
                    // this becomes a programmatic graph edge, so the kernel
                    // may begin (and stream its own inputs) while the
                    // previous launch drains; its griddepcontrol.wait keeps
                    // the data dependency.
                    let mut attr = sys::CUlaunchAttribute {
                        id: sys::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION,
                        pad: [0; 4],
                        value: sys::CUlaunchAttributeValue { programmaticStreamSerializationAllowed: 1 },
                    };
                    let cfg = sys::CUlaunchConfig {
                        gridDimX: grid.0,
                        gridDimY: grid.1,
                        gridDimZ: grid.2,
                        blockDimX: block[0],
                        blockDimY: block[1],
                        blockDimZ: block[2],
                        sharedMemBytes: smem,
                        hStream: self.stream.cu_stream(),
                        attrs: &mut attr,
                        numAttrs: 1,
                    };
                    let r = unsafe {
                        sys::cuLaunchKernelEx(&cfg, *func, params.as_mut_ptr(), std::ptr::null_mut())
                    };
                    return cuda_check(r, "cuLaunchKernelEx");
                }
                unsafe {
                    cu::launch_kernel(
                        *func,
                        grid,
                        (block[0], block[1], block[2]),
                        smem,
                        self.stream.cu_stream(),
                        &mut params,
                    )
                    .map_err(|e| Error::Cuda(format!("cuLaunchKernel: {e:?}")))
                }
            }
        }
    }
}
