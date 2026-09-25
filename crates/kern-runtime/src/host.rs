//! Hosted execution: the runtime inside another engine's process, on that
//! engine's memory and in that engine's stream order.
//!
//! A host state (`states.<name>.host` in the manifest) is memory the host
//! allocates, sizes and indexes — a serving engine's KV cache, laid out the
//! way that engine lays it out. The manifest declares the layout its kernels
//! were written against; [`Runtime::bind_host`] takes the host's address and
//! the layout the host actually has, admits it only if the two agree, and
//! compiles the programs then: every program bakes addresses in, so a
//! manifest with host states compiles at bind, not at load. Load still
//! allocates everything else (weights, buffers, scratch), which is what a
//! host that budgets its memory around the model needs to see first. A host
//! may reallocate (vLLM sizes a throwaway cache to profile its graphs, then
//! the real one), so binding again replaces every host state and compiles
//! again; the runtime's own captured graphs, which baked the old addresses,
//! are dropped.
//!
//! The host's work and the runtime's are ordered by the host's stream:
//! [`Runtime::enqueue_after`] makes the runtime's stream wait for what the
//! host enqueued, launches the program, and makes the host's stream wait for
//! the program. Both edges are events, so the same call records into a CUDA
//! graph the host is capturing.
//!
//! [`Runtime::region`] hands the host the address of a runtime buffer, so it
//! writes inputs and reads outputs in its own stream order instead of
//! through the runtime's staging.

use std::collections::BTreeMap;

use cudarc::driver::sys;
use kern_manifest::types::DType;

use crate::device::DeviceBuf;
use crate::error::{bail, cuda_check};
use crate::{compile, Error, Result, Runtime};

/// A host tensor as the host has it: its address, element type, extents and
/// element strides, outermost first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostRegion {
    pub ptr: u64,
    pub dtype: DType,
    pub shape: Vec<u64>,
    pub strides: Vec<u64>,
}

impl Runtime {
    /// Bind every host state to the host's memory and compile the programs.
    /// Each region must be the layout the manifest declares; nothing is
    /// bound unless every one is. Binding again replaces the previous binding.
    pub fn bind_host(&mut self, regions: &BTreeMap<String, HostRegion>) -> Result<()> {
        self.ctx.bind_to_thread()?;
        let declared: BTreeMap<&str, _> =
            self.manifest.states.iter().filter_map(|(n, s)| s.host.as_ref().map(|h| (n.as_str(), h))).collect();
        if declared.is_empty() {
            bail!(Api, "nothing to bind: the manifest has no host states");
        }
        if let Some(extra) = regions.keys().find(|n| !declared.contains_key(n.as_str())) {
            bail!(Api, "region `{extra}`: the manifest declares no host state of that name");
        }
        let bound = declared
            .iter()
            .map(|(&name, h)| {
                let Some(r) = regions.get(name) else {
                    bail!(Api, "host state `{name}`: no region given");
                };
                let bytes =
                    h.admit(r.dtype, &r.shape, &r.strides).map_err(|e| Error::Api(format!("state `{name}`: {e}")))?;
                if r.ptr == 0 {
                    bail!(Api, "state `{name}`: null address");
                }
                Ok((name.to_string(), DeviceBuf::borrowed(&self.stream, r.ptr, bytes)))
            })
            .collect::<Result<Vec<_>>>()?;
        self.states.extend(bound);
        for (_, exec) in std::mem::take(&mut self.graphs) {
            cuda_check(unsafe { sys::cuGraphExecDestroy(exec) }, "cuGraphExecDestroy")?;
        }
        match compile::compile(&self.manifest, &self.ops, &self.buffers, &self.states, &self.ranks, &self.peers) {
            Ok(programs) => {
                self.programs = programs;
                Ok(())
            }
            Err(e) => {
                self.states.retain(|n, _| !declared.contains_key(n.as_str()));
                self.programs.clear();
                Err(e)
            }
        }
    }

    /// Launch a program after everything the host enqueued on `stream` (a
    /// raw `CUstream` in this runtime's context) and before anything it
    /// enqueues there next. Launch by launch whatever the manifest's
    /// `graph`: under the host's capture the launches record into the
    /// host's graph.
    pub fn enqueue_after(&self, program: &str, vars: &BTreeMap<String, u64>, stream: u64) -> Result<()> {
        self.ctx.bind_to_thread()?;
        let host = stream as sys::CUstream;
        let [before, after] = self.joins;
        cuda_check(unsafe { sys::cuEventRecord(before, host) }, "cuEventRecord")?;
        cuda_check(unsafe { sys::cuStreamWaitEvent(self.stream.cu_stream(), before, 0) }, "cuStreamWaitEvent")?;
        self.enqueue(program, vars)?;
        cuda_check(unsafe { sys::cuEventRecord(after, self.stream.cu_stream()) }, "cuEventRecord")?;
        cuda_check(unsafe { sys::cuStreamWaitEvent(host, after, 0) }, "cuStreamWaitEvent")
    }

    /// The address and byte size of a runtime buffer, for a host that reads
    /// and writes it in its own stream order.
    pub fn region(&self, name: &str) -> Result<(u64, u64)> {
        match self.buffers.get(name) {
            Some(b) => Ok((b.ptr, b.bytes)),
            None => bail!(Api, "no buffer `{name}`"),
        }
    }

    /// Whether the manifest has host states still waiting for [`Runtime::bind_host`].
    pub fn awaits_host(&self) -> bool {
        self.manifest.states.keys().any(|n| !self.states.contains_key(n))
    }
}
