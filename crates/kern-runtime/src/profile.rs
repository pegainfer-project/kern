//! Opt-in measurement for `kern bench`. A sample is one traced replay of a
//! program's graph, from the program's pre-image behind an L2 flush. The tracer names every kernel by the call
//! that launched it, so a call's time and the whole graph's come from the
//! same uninstrumented replay; nothing is inserted between kernels.
//! Profiling never changes the serving path. Like every runtime entry
//! point, these bind the runtime's context on entry: a rank is measured
//! from whichever thread holds it.
//!
//! This lives inside the runtime because a sample owns a graph of the
//! compiled launch list and copies pre-images on the runtime's own stream:
//! none of that is reachable through the public surface.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use cudarc::driver::sys;
use kern_manifest::types::{Arg, BufferKind, Dir};

use crate::compile::Dense;
use crate::device::alloc_uninit;
use crate::{alloc, bail, cuda_check, trace, DeviceBuf, Result, Runtime};

fn copy(rt: &Runtime, dst: u64, src: u64, bytes: u64) -> Result<()> {
    cuda_check(unsafe { sys::cuMemcpyDtoDAsync_v2(dst, src, bytes as usize, rt.stream.cu_stream()) }, "profile copy")
}

/// The device a probe measures on, and the buffers its L2 flush streams
/// through: a copy-engine copy of eight times the L2 evicts it as fully as
/// an SM copy does, whatever the bytes are.
pub struct Probe {
    pub device: String,
    pub sm_count: u32,
    pub l2_bytes: u64,
    pub driver_version: i32,
    /// Whether the tracer's timestamps are the GPU's own (HES) rather than
    /// CUPTI's software ones.
    pub hes: bool,
    flush: [DeviceBuf; 2],
}

impl Probe {
    pub fn new(rt: &Runtime) -> Result<Self> {
        if !trace::on() {
            bail!(Api, "a probe reads its samples from the tracer: `trace::start()` before any runtime loads");
        }
        rt.ctx.bind_to_thread()?;
        let l2_bytes = rt.ctx.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)? as u64;
        let sm_count = rt.ctx.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as u32;
        let bytes = (l2_bytes * 8).max(256 << 20);
        let mut driver_version = 0;
        cuda_check(unsafe { sys::cuDriverGetVersion(&mut driver_version) }, "driver version")?;
        Ok(Self {
            device: rt.ctx.name()?,
            sm_count,
            l2_bytes,
            driver_version,
            hes: trace::hes()?,
            flush: [alloc_uninit(&rt.stream, bytes)?, alloc_uninit(&rt.stream, bytes)?],
        })
    }

    /// Restores `pre`, flushes L2, replays `g` and waits for it; the
    /// issue's sequence number, under which the tracer returns its kernels.
    pub fn sample(&self, rt: &Runtime, pre: &PreImage, g: &Graph) -> Result<u64> {
        rt.ctx.bind_to_thread()?;
        pre.0.iter().try_for_each(|(dst, src)| copy(rt, *dst, src.ptr, src.bytes))?;
        let [a, b] = &self.flush;
        copy(rt, b.ptr, a.ptr, a.bytes)?;
        let seq = rt.launch_traced(g.exec, &g.program, &g.vars, g.kernels.clone())?;
        rt.stream.synchronize()?;
        Ok(seq)
    }
}

/// What a sample of a program restores before it runs, GPU resident.
pub struct PreImage(Vec<(u64, DeviceBuf)>);

impl PreImage {
    /// The program's declared carry and state writes as they are now.
    /// Workspace is written before it is read within a run, so it is left
    /// alone; snapshotting every carry and state of a large manifest would
    /// not fit beside them. An exported carry is not this rank's to
    /// restore either: peers write into it on their own clock, and the
    /// barrier epochs inside it must only advance. A paged state is only
    /// appended to: a run writes the rows it adds, as a function of what is
    /// restored, so a replay rewrites them with the same bytes, and
    /// snapshotting it would double the largest allocation, a long
    /// context's KV. Whole allocations are copied: the manifest does not
    /// declare which bytes of a state a kernel writes.
    pub fn new(rt: &Runtime, program: &str) -> Result<Self> {
        let m = &rt.manifest;
        let Some(p) = m.programs.get(program) else {
            bail!(Api, "no program `{program}`");
        };
        let (mut carries, mut states) = (BTreeSet::new(), BTreeSet::new());
        for call in &p.calls {
            for (a, param) in call.args.iter().zip(&m.ops[&call.op].params) {
                if !matches!(param.dir(), Some(Dir::Out | Dir::InOut)) {
                    continue;
                }
                match a {
                    Arg::Buf { buf, .. } if m.buffers[buf].kind == BufferKind::Carry && !m.buffers[buf].export => {
                        carries.insert(buf);
                    }
                    Arg::State { state, .. } if m.states[state].bytes_per_token == 0 => {
                        states.insert(state);
                    }
                    _ => {}
                }
            }
        }
        rt.ctx.bind_to_thread()?;
        let sources = carries.into_iter().map(|n| &rt.buffers[n]).chain(states.into_iter().map(|n| &rt.states[n]));
        let saved = sources
            .map(|src| {
                let dst = alloc(&rt.stream, src.bytes)?;
                copy(rt, dst.ptr, src.ptr, src.bytes)?;
                Ok((src.ptr, dst))
            })
            .collect::<Result<_>>()?;
        rt.stream.synchronize()?;
        Ok(Self(saved))
    }
}

/// A program's graph at one set of var values, and the call behind each
/// of its kernel nodes.
pub struct Graph {
    exec: sys::CUgraphExec,
    program: String,
    vars: Dense,
    kernels: Arc<BTreeMap<u32, u32>>,
}

impl Graph {
    pub fn new(rt: &Runtime, program: &str, vars: &BTreeMap<String, u64>) -> Result<Self> {
        let Some(prog) = rt.programs.get(program) else {
            bail!(Api, "no program `{program}`");
        };
        let vars = Dense::check(&rt.manifest, vars, &prog.vars)?;
        let (exec, kernels) = rt.record(prog, &vars)?;
        Ok(Self { exec, program: program.to_string(), vars, kernels: Arc::new(kernels) })
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        unsafe { sys::cuGraphExecDestroy(self.exec) };
    }
}
