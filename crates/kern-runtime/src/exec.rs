//! Execution: one program at one set of var values, onto the compute
//! stream.
//!
//! The names died at load (`compile`); what runs here is a flat launch
//! list whose slots are constants or var-indexed expressions, so a launch
//! is evaluation and `cuLaunchKernelEx`, no lookup and nothing left to
//! panic on. The caller's vars are densified once per call into manifest
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
//!
//! A caller that wants to know when the device got somewhere without
//! stopping the thread that issues takes a [`Mark`] and waits on it from
//! another thread.

use std::collections::BTreeMap;
use std::ffi::CStr;
use std::os::raw::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use cudarc::driver::{sys, CudaContext};

use crate::compile::{CompiledProgram, Dense, Launch, LaunchKind, RVal, Slot};
use crate::cublas::{gemm_bf16_tn, gemm_bf16_tn_f32, gemm_bf16_tn_pinned, gemm_fp8_tn};
use crate::device::Pinned;
use crate::error::{bail, cuda_check};
use crate::{Error, Result, Runtime};
use kern_manifest::types::BufferKind;

/// A point on the compute stream: everything issued before
/// [`Runtime::mark`]. [`Mark::wait`] blocks until the device has passed
/// it, on any thread.
///
/// Passing is read off a word in page-locked memory that a kernel at the
/// mark writes, never off the driver: the thread that waits on a mark may
/// be one the issuing thread is waiting for (it reports items done to the
/// stage before, which pushes the next item, which opens the hold the
/// issuing thread's launches sit behind), and any driver call it made
/// could block on the context lock that thread holds inside a blocking
/// launch (lessons.md, 2026-10-08). So a mark carries no event and says
/// nothing about faults; the issuing thread asks [`Runtime::fault`].
pub struct Mark {
    seq: u64,
    passed: Arc<Pinned>,
}

/// How long a mark is waited for before the device is given up on.
const TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

impl Mark {
    /// Whether the device has passed the mark.
    pub fn passed(&self) -> bool {
        unsafe { &*(self.passed.ptr() as *const AtomicU64) }.load(Ordering::Acquire) >= self.seq
    }

    /// Spins for the first moments, then sleeps in short steps.
    pub fn wait(self) -> Result<()> {
        let t = std::time::Instant::now();
        while !self.passed() {
            if t.elapsed() > TIMEOUT {
                bail!(Cuda, "the device has not passed mark {} in {TIMEOUT:?}", self.seq);
            }
            if t.elapsed() > std::time::Duration::from_millis(2) {
                std::thread::sleep(std::time::Duration::from_micros(20));
            } else {
                std::hint::spin_loop();
            }
        }
        Ok(())
    }
}

/// A count in page-locked host memory the compute stream can wait on
/// ([`Runtime::hold`]) and any thread can raise ([`Gate::open`]): the
/// mirror of a [`Mark`]. With it a caller issues work before the data it
/// reads is there and the device, not the host, holds the launches back.
pub struct Gate {
    host: Pinned,
    device: u64,
}

impl Gate {
    /// Raise the count to `n`; every hold for at most `n` lets go.
    pub fn open(&self, n: u64) {
        unsafe { &*(self.host.ptr() as *const AtomicU64) }.store(n, Ordering::Release);
    }
}

/// The two kernels the runtime signals with, a few lines of PTX loaded
/// once: `hold` spins on a gate's count, `flag` writes a mark's number.
/// The hold is a kernel and not a stream memory operation because
/// `cuStreamWaitValue` stalls the hardware channel its stream is
/// multiplexed onto, and cuBLAS's own streams share those channels: a
/// GEMM issued behind one deadlocked against the next hold (tray08,
/// 2026-10-08). A spinning kernel holds one SM and nothing else.
pub(crate) struct Signals {
    module: sys::CUmodule,
    hold: sys::CUfunction,
    flag: sys::CUfunction,
    passed: Arc<Pinned>,
    passed_device: u64,
    seq: AtomicU64,
    fetches: Arc<Mutex<Vec<Pinned>>>,
}

unsafe impl Send for Signals {}
unsafe impl Sync for Signals {}

const SIGNALS_PTX: &CStr = c"
.version 8.0
.target sm_90
.address_size 64
.visible .entry hold(.param .u64 gate, .param .u64 count)
{
    .reg .b64 %g, %c, %v;
    .reg .pred %p;
    ld.param.u64 %g, [gate];
    ld.param.u64 %c, [count];
    cvta.to.global.u64 %g, %g;
L:
    ld.acquire.sys.global.u64 %v, [%g];
    setp.ge.u64 %p, %v, %c;
    @%p bra D;
    nanosleep.u32 256;
    bra L;
D:
    ret;
}
.visible .entry flag(.param .u64 word, .param .u64 value)
{
    .reg .b64 %w, %v;
    ld.param.u64 %w, [word];
    ld.param.u64 %v, [value];
    cvta.to.global.u64 %w, %w;
    st.release.sys.global.u64 [%w], %v;
    ret;
}
";

impl Signals {
    pub(crate) fn load(ctx: &Arc<CudaContext>, gpu: usize) -> Result<Signals> {
        ctx.bind_to_thread()?;
        let mut module: sys::CUmodule = std::ptr::null_mut();
        cuda_check(
            unsafe { sys::cuModuleLoadData(&mut module, SIGNALS_PTX.as_ptr() as *const c_void) },
            "cuModuleLoadData(signals)",
        )?;
        let function = |name: &CStr| {
            let mut f: sys::CUfunction = std::ptr::null_mut();
            cuda_check(
                unsafe { sys::cuModuleGetFunction(&mut f, module, name.as_ptr()) },
                "cuModuleGetFunction(signals)",
            )
            .map(|()| f)
        };
        let loaded = function(c"hold").and_then(|hold| Ok((hold, function(c"flag")?)));
        let (hold, flag) = match loaded {
            Ok(fs) => fs,
            Err(e) => {
                unsafe { sys::cuModuleUnload(module) };
                return Err(e);
            }
        };
        let passed = Arc::new(Pinned::alloc_mapped(8, gpu as i32)?);
        unsafe { &*(passed.ptr() as *const AtomicU64) }.store(0, Ordering::Release);
        let passed_device = passed.device()?;
        Ok(Signals { module, hold, flag, passed, passed_device, seq: AtomicU64::new(0), fetches: Arc::default() })
    }
}

impl Drop for Signals {
    fn drop(&mut self) {
        unsafe { sys::cuModuleUnload(self.module) };
    }
}

/// An output's bytes on their way to the host: a copy the compute stream
/// makes into page-locked memory, complete once a [`Mark`] taken after it
/// has passed. The thread that issues never waits for it. Dropped, the
/// memory goes back to the runtime for the next fetch rather than to the
/// driver: freeing page-locked memory synchronizes the device, and the
/// thread reading a fetch is one that must not (lessons.md, 2026-10-08).
pub struct Fetch {
    pinned: Option<Pinned>,
    len: usize,
    pool: Arc<Mutex<Vec<Pinned>>>,
}

impl Fetch {
    /// The bytes; meaningful only after a mark taken after the fetch passed.
    pub fn bytes(&self) -> &[u8] {
        let p = self.pinned.as_ref().expect("taken only on drop");
        unsafe { std::slice::from_raw_parts(p.ptr() as *const u8, self.len) }
    }
}

impl Drop for Fetch {
    fn drop(&mut self) {
        if let Some(p) = self.pinned.take() {
            self.pool.lock().unwrap_or_else(|e| e.into_inner()).push(p);
        }
    }
}

impl Runtime {
    fn signal(&self, f: sys::CUfunction, word: u64, value: u64, what: &str) -> Result<()> {
        self.ctx.bind_to_thread()?;
        let mut params = [&word as *const u64 as *mut c_void, &value as *const u64 as *mut c_void];
        let r = unsafe {
            sys::cuLaunchKernel(
                f,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                self.stream.cu_stream(),
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        cuda_check(r, what)
    }

    /// Start fetching `name`, an output buffer, as it is after everything
    /// issued so far.
    pub fn fetch(&self, name: &str) -> Result<Fetch> {
        self.ctx.bind_to_thread()?;
        let Some(b) = self.manifest.buffers.get(name) else {
            bail!(Api, "no buffer `{name}`");
        };
        if !matches!(b.kind, BufferKind::Output | BufferKind::Inout) {
            bail!(Api, "buffer `{name}` is {}, not output", b.kind);
        }
        let src = &self.buffers[name];
        let pool = &self.signals.fetches;
        let spare = {
            let mut v = pool.lock().unwrap_or_else(|e| e.into_inner());
            v.iter().position(|p| p.bytes() >= src.bytes).map(|i| v.swap_remove(i))
        };
        let pinned = match spare {
            Some(p) => p,
            None => Pinned::alloc(src.bytes, self.gpu as i32)?,
        };
        let r = unsafe {
            sys::cuMemcpyDtoHAsync_v2(pinned.ptr() as *mut c_void, src.ptr, src.bytes as usize, self.stream.cu_stream())
        };
        cuda_check(r, "cuMemcpyDtoHAsync")?;
        Ok(Fetch { pinned: Some(pinned), len: src.bytes as usize, pool: pool.clone() })
    }

    /// Mark the compute stream where it is now.
    pub fn mark(&self) -> Result<Mark> {
        let seq = self.signals.seq.fetch_add(1, Ordering::Relaxed) + 1;
        self.signal(self.signals.flag, self.signals.passed_device, seq, "cuLaunchKernel(flag)")?;
        Ok(Mark { seq, passed: self.signals.passed.clone() })
    }

    /// The fault of any launch so far, without waiting for the rest: the
    /// issuing thread's way of hearing what a mark it never waits on would
    /// have said.
    pub fn fault(&self) -> Result<()> {
        self.ctx.bind_to_thread()?;
        match unsafe { sys::cuStreamQuery(self.stream.cu_stream()) } {
            sys::CUresult::CUDA_SUCCESS | sys::CUresult::CUDA_ERROR_NOT_READY => Ok(()),
            r => cuda_check(r, "cuStreamQuery"),
        }
    }

    /// A gate at count zero.
    pub fn gate(&self) -> Result<Gate> {
        let host = Pinned::alloc_mapped(8, self.gpu as i32)?;
        let device = host.device()?;
        let gate = Gate { host, device };
        gate.open(0);
        Ok(gate)
    }

    /// Hold the compute stream where it is now until `gate` reaches `n`.
    pub fn hold(&self, gate: &Gate, n: u64) -> Result<()> {
        self.signal(self.signals.hold, gate.device, n, "cuLaunchKernel(hold)")
    }

    fn require_ready(&self) -> Result<()> {
        if self.awaits_host() {
            bail!(Api, "the manifest's host states are not bound; bind_host before executing programs");
        }
        if !self.host_weights_ready {
            bail!(Api, "host weights must be bound with load_weights before executing programs");
        }
        Ok(())
    }

    /// Execute one program with the given var values, then synchronize.
    pub fn run(&self, program: &str, vars: &BTreeMap<String, u64>) -> Result<()> {
        self.enqueue(program, vars)?;
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
    pub fn issue(&mut self, program: &str, vars: &BTreeMap<String, u64>) -> Result<()> {
        self.require_ready()?;
        let Some(prog) = self.programs.get(program) else {
            bail!(Api, "no program `{program}`");
        };
        if !prog.graph || self.eager {
            return self.enqueue(program, vars);
        }
        if !self.is_captured(program, vars) {
            // One capture + instantiate at a time across ranks: CUPTI's
            // graph-node tracing (nsys --cuda-graph-trace=node) dereferences
            // null inside cuGraphInstantiate when several contexts instantiate
            // concurrently. Steady-state replay stays parallel.
            static CAPTURE: std::sync::Mutex<()> = std::sync::Mutex::new(());
            let _serial = CAPTURE.lock().unwrap_or_else(|e| e.into_inner());
            let t = std::time::Instant::now();
            let calls = prog.call_ranges.len();
            self.capture(program, vars)?;
            tracing::info!(program, vars = ?vars, calls, capture_ms = t.elapsed().as_millis() as u64, "graph captured");
        }
        self.enqueue_captured(program, vars)
    }

    /// Issue one program's launches onto the stream and return without
    /// waiting. Ranks whose kernels wait on each other (an EP dispatch, a
    /// tray collective) must all be issued before any is waited for:
    /// `enqueue` each, then [`Runtime::synchronize`] each.
    pub(crate) fn enqueue(&self, program: &str, vars: &BTreeMap<String, u64>) -> Result<()> {
        self.require_ready()?;
        let Some(prog) = self.programs.get(program) else {
            bail!(Api, "no program `{program}`");
        };
        self.require_peers()?;
        self.require_nccl()?;
        let vars = Dense::check(&self.manifest, vars, &prog.vars)?;
        self.ctx.bind_to_thread()?;
        self.replay(prog, &vars)
    }

    /// Capture one program into an instantiated CUDA graph. Grid dims and
    /// scalar args (var values included) are baked in at capture; input
    /// buffer *contents* are read at replay, so per-step H2D writes stay
    /// outside the graph and `run_captured` replays the whole call list
    /// with one launch.
    pub fn capture(&mut self, program: &str, vars: &BTreeMap<String, u64>) -> Result<()> {
        self.require_ready()?;
        let Some(prog) = self.programs.get(program) else {
            bail!(Api, "no program `{program}`");
        };
        self.require_peers()?;
        self.require_nccl()?;
        let vars = Dense::check(&self.manifest, vars, &prog.vars)?;
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
        let replayed = self.replay(prog, &vars);
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
        if let Some(old) = self.graphs.insert((program.to_string(), vars), exec) {
            unsafe { sys::cuGraphExecDestroy(old) };
        }
        Ok(())
    }

    /// Whether `capture(program, vars)` has been done for exactly these var
    /// values.
    pub fn is_captured(&self, program: &str, vars: &BTreeMap<String, u64>) -> bool {
        self.programs
            .get(program)
            .and_then(|prog| Dense::check(&self.manifest, vars, &prog.vars).ok())
            .is_some_and(|vars| self.graphs.contains_key(&(program.to_string(), vars)))
    }

    /// The graph captured for (program, vars), or an `Api` error naming the
    /// var values that were captured instead.
    pub(crate) fn graph(&self, program: &str, vars: &BTreeMap<String, u64>) -> Result<sys::CUgraphExec> {
        let Some(prog) = self.programs.get(program) else {
            bail!(Api, "no program `{program}`");
        };
        let dense = Dense::check(&self.manifest, vars, &prog.vars)?;
        if let Some(exec) = self.graphs.get(&(program.to_string(), dense.clone())) {
            return Ok(*exec);
        }
        let others: Vec<String> = self
            .graphs
            .keys()
            .filter(|(p, _)| p == program)
            .map(|(_, e)| format!("{{{}}}", e.describe(&self.manifest)))
            .collect();
        if others.is_empty() {
            bail!(Api, "program `{program}` has not been captured");
        }
        bail!(
            Api,
            "program `{program}` called with {{{}}} but captured at {}",
            dense.describe(&self.manifest),
            others.join(", ")
        )
    }

    /// Replay a previously captured program, then synchronize.
    pub fn run_captured(&self, program: &str, vars: &BTreeMap<String, u64>) -> Result<()> {
        self.enqueue_captured(program, vars)?;
        self.stream.synchronize()?;
        Ok(())
    }

    /// Launch a previously captured program's graph without waiting (see
    /// [`Runtime::enqueue`]).
    fn enqueue_captured(&self, program: &str, vars: &BTreeMap<String, u64>) -> Result<()> {
        self.require_ready()?;
        let exec = self.graph(program, vars)?;
        self.ctx.bind_to_thread()?;
        cuda_check(unsafe { sys::cuGraphLaunch(exec, self.stream.cu_stream()) }, "cuGraphLaunch")
    }

    /// Issue every launch of a compiled program onto the stream (no sync).
    pub(crate) fn replay(&self, prog: &CompiledProgram, vars: &Dense) -> Result<()> {
        for l in &prog.launches {
            self.launch(l, vars).map_err(|e| Error::Call { context: l.ctx.clone(), source: Box::new(e) })?;
        }
        Ok(())
    }

    pub(crate) fn launch(&self, l: &Launch, vars: &Dense) -> Result<()> {
        if let Some((v, lo, hi)) = &l.when {
            if !(*lo..=*hi).contains(&v.eval(vars)?) {
                return Ok(());
            }
        }
        // Materialize the slots; only var-dependent scalars are left to
        // compute, everything else was finished at load. Packs (and the
        // tensor maps inside them) ride along as pointers to their images.
        let mut vals = Vec::with_capacity(l.slots.len());
        let mut images: Vec<Option<Vec<u8>>> = Vec::with_capacity(l.slots.len());
        for s in &l.slots {
            let (v, m) = match s {
                Slot::Const(rv) => (*rv, None),
                Slot::Expr(e) => (RVal { val: e.eval(vars)?, bytes: 0 }, None),
                Slot::Pack(p) => (RVal { val: 0, bytes: 0 }, Some(p.image(vars)?)),
            };
            vals.push(v);
            images.push(m);
        }
        match &l.kind {
            LaunchKind::Gemm { beta, algo: None } => gemm_bf16_tn(&self.blt, &self.stream, &vals, *beta),
            LaunchKind::Gemm { beta, algo: Some(algo) } => {
                gemm_bf16_tn_pinned(&self.blt, &self.blas, &self.stream, &vals, *beta, algo)
            }
            LaunchKind::GemmF32 => gemm_bf16_tn_f32(&self.blas, &vals),
            LaunchKind::GemmFp8 { f32_out } => gemm_fp8_tn(&self.blt, &self.blas, &self.stream, &vals, *f32_out),
            LaunchKind::Nccl { coll, elem, group } => self.collective(*coll, *elem, group, &vals),
            LaunchKind::Cubin { func, block, grid, shared_mem, cluster, pdl } => {
                let grid = [grid[0].eval(vars)? as u32, grid[1].eval(vars)? as u32, grid[2].eval(vars)? as u32];
                let smem = match shared_mem {
                    Some(e) => e.eval(vars)? as u32,
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
