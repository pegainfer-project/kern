//! `import kern`: the runtime for a Python host.
//!
//! The binding speaks the runtime's words and nothing else — a manifest,
//! buffers, host states, programs, vars — so it is the same for every host.
//! What a particular engine calls its KV cache, its block table or its
//! stream lives in that engine's adapter, in Python, next to the engine it
//! tracks (`kern_vllm` for vLLM): an engine that changes its
//! internals changes the adapter, never this crate.
//!
//! The host reads the verified manifest as JSON ([`Runtime::manifest`]) to
//! learn every buffer's role and layout, writes inputs and reads outputs
//! through their addresses ([`Runtime::region`]) in its own stream order,
//! binds its memory to the host states ([`Runtime::bind_host`]) and orders
//! each program after its own work ([`Runtime::enqueue_after`]). The
//! runtime is not `Send` (it holds CUDA graph and event handles), so every
//! call stays on the thread that loaded it; only the load lets go of the
//! GIL, being minutes for a large checkpoint.
//!
//! A manifest with a topology loads as one rank of it, and its ranks meet
//! through [`Runtime::offer`] and [`Runtime::connect`]: the host carries the
//! offers' bytes over its own process group (`kern.group` does it over a
//! `torch.distributed` one). The pure-Python half of the package is that
//! and the manifest-contract readers every engine adapter shares
//! (`kern.host`).
//!
//! A host that wants its kernels traced subscribes the process before its
//! first CUDA context ([`trace_start`]), has the runtime keep them on disk
//! ([`Runtime::record`]), and tells the tracer when it launches a CUDA
//! graph it captured programs into ([`Runtime::captured`], [`HostGraph`]);
//! `kern.trace` does the last over torch's graphs.

mod record;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use kern_manifest::types::DType;
use kern_manifest::Verified;
use kern_runtime::{trace, GroupRank, HostRegion, Offer, Safetensors, Topology};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// A loaded runtime: every buffer allocated and every weight on the device.
/// A manifest with host states runs nothing until they are bound; one with
/// a topology runs nothing until it is connected.
#[pyclass(unsendable, module = "kern")]
struct Runtime {
    rt: kern_runtime::Runtime,
    /// What the files [`Runtime::record`] writes say of their writer.
    meta: Vec<(String, String)>,
}

/// The programs one CUDA graph of the host's holds: what a launch of that
/// graph issues.
#[pyclass(module = "kern")]
struct HostGraph {
    graph: kern_runtime::HostGraph,
}

#[pymethods]
impl HostGraph {
    /// Tag this thread, so the launch of the graph it makes next is traced
    /// as one step per program, until `launched`.
    fn launching(&self) -> PyResult<()> {
        self.graph.launching().map_err(runtime_err)
    }

    fn launched(&self) -> PyResult<()> {
        self.graph.launched().map_err(runtime_err)
    }
}

/// Whether this process already has a thread recording its kernels.
static RECORDING: AtomicBool = AtomicBool::new(false);

fn runtime_err(e: kern_runtime::Error) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// A runtime carried out of the load, which ran on this same thread with
/// the GIL released: it never crosses to another thread.
struct Loaded(kern_runtime::Runtime);
#[allow(unsafe_code)]
unsafe impl Send for Loaded {}

fn load(
    m: &Verified,
    kernels: Option<&Path>,
    weights: &[PathBuf],
    gpu: usize,
    topology: Option<&Topology>,
) -> kern_runtime::Result<Loaded> {
    let mut rt = kern_runtime::Runtime::load(m, kernels, gpu, None, topology)?;
    rt.load_weights(&Safetensors::open(weights)?)?;
    Ok(Loaded(rt))
}

#[pymethods]
impl Runtime {
    /// Verify the manifest, load its kernels onto `gpu` and its weights out
    /// of `weights` (safetensors files, or directories of shards). Kernels
    /// come from `kernels` or, without one, from the registry the manifest
    /// names (fetched into `$KERN_CACHE_DIR`, by default `~/.cache/kern`). A
    /// manifest with a topology takes this rank's place in every group,
    /// `{group: (index, size)}`.
    #[new]
    #[pyo3(signature = (manifest, kernels, weights, gpu, topology=None))]
    fn new(
        py: Python<'_>,
        manifest: PathBuf,
        kernels: Option<PathBuf>,
        weights: Vec<PathBuf>,
        gpu: usize,
        topology: Option<BTreeMap<String, (u64, u64)>>,
    ) -> PyResult<Self> {
        let text = std::fs::read_to_string(&manifest)
            .map_err(|e| PyValueError::new_err(format!("manifest {}: {e}", manifest.display())))?;
        let m = Verified::from_json(&text)
            .map_err(|e| PyValueError::new_err(format!("manifest {}:\n{e}", manifest.display())))?;
        let topology = topology.map(|t| Topology {
            groups: t.into_iter().map(|(g, (index, size))| (g, GroupRank { index, size })).collect(),
        });
        let rt = py.detach(|| load(&m, kernels.as_deref(), &weights, gpu, topology.as_ref())).map_err(runtime_err)?.0;
        Ok(Runtime { rt, meta: kern_trace::meta(&m, &text, env!("CARGO_PKG_VERSION")) })
    }

    /// The verified manifest as JSON.
    fn manifest(&self) -> String {
        self.rt.manifest.to_json()
    }

    /// This rank's index in topology group `group`; `None` outside one.
    fn rank(&self, group: &str) -> Option<u64> {
        self.rt.rank(group)
    }

    /// This rank's offer to its groups, as bytes for the host's channel:
    /// its exported allocations' fabric handles, and the communicator id
    /// of every NCCL group it is member 0 of.
    fn offer<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let b = self.rt.offer().and_then(|o| o.to_bytes()).map_err(runtime_err)?;
        Ok(PyBytes::new(py, &b))
    }

    /// Map every peer and join every communicator: `members[g]` is group
    /// `g`'s offers in member order, this rank's own among them. Blocks
    /// until every member of an NCCL group has joined.
    fn connect(&mut self, members: BTreeMap<String, Vec<Vec<u8>>>) -> PyResult<()> {
        let members = members
            .into_iter()
            .map(|(g, v)| Ok((g, v.iter().map(|b| Offer::from_bytes(b)).collect::<kern_runtime::Result<_>>()?)))
            .collect::<kern_runtime::Result<BTreeMap<_, _>>>()
            .map_err(runtime_err)?;
        self.rt.connect(&members).map_err(runtime_err)
    }

    /// Whether host states still wait for `bind_host`.
    #[getter]
    fn awaits_host(&self) -> bool {
        self.rt.awaits_host()
    }

    /// Bind every host state, `{name: (address, dtype, shape, strides)}`
    /// with element strides, then compile the programs.
    fn bind_host(&mut self, regions: BTreeMap<String, (u64, String, Vec<u64>, Vec<u64>)>) -> PyResult<()> {
        let regions = regions
            .into_iter()
            .map(|(name, (ptr, dtype, shape, strides))| {
                let dtype: DType =
                    dtype.parse().map_err(|e: String| PyValueError::new_err(format!("`{name}`: {e}")))?;
                Ok((name, HostRegion { ptr, dtype, shape, strides }))
            })
            .collect::<PyResult<BTreeMap<_, _>>>()?;
        self.rt.bind_host(&regions).map_err(runtime_err)
    }

    /// `(address, bytes)` of a buffer.
    fn region(&self, name: &str) -> PyResult<(u64, u64)> {
        self.rt.region(name).map_err(runtime_err)
    }

    /// Run a program on the runtime's stream and wait for it.
    fn run(&self, program: &str, vars: BTreeMap<String, u64>) -> PyResult<()> {
        self.rt.run(program, &vars).map_err(runtime_err)
    }

    /// Launch a program after the work already on the host's `stream` (a raw
    /// `CUstream`, e.g. `torch.cuda.current_stream().cuda_stream`) and
    /// before the work the host enqueues there next.
    fn enqueue_after(&self, program: &str, vars: BTreeMap<String, u64>, stream: u64) -> PyResult<()> {
        self.rt.enqueue_after(program, &vars, stream).map_err(runtime_err)
    }

    /// Wait for everything the runtime enqueued.
    fn synchronize(&self) -> PyResult<()> {
        self.rt.synchronize().map_err(runtime_err)
    }

    /// Keep every kernel this process runs under `dir` as `rank`, in the
    /// files kern-serve's `--trace-dir` writes, each kernel of a program
    /// named by its call in this runtime's manifest. Once per process,
    /// which `trace_start` subscribed before its first CUDA context.
    #[pyo3(signature = (dir, rank=0))]
    fn record(&self, dir: PathBuf, rank: usize) -> PyResult<()> {
        if !trace::on() {
            return Err(PyRuntimeError::new_err(
                "tracing is off: kern.trace.start() must run before the process's first CUDA context",
            ));
        }
        if RECORDING.swap(true, Ordering::AcqRel) {
            return Err(PyRuntimeError::new_err("this process already records its kernels"));
        }
        record::start(dir, rank, kern_trace::Sites::new(&self.rt.manifest), self.meta.clone())
            .map_err(|e| PyRuntimeError::new_err(format!("starting the recording thread: {e}")))
    }

    /// What the host's last capture took from this runtime, once it has
    /// ended: `None` when no program was enqueued into it, or tracing is
    /// off.
    fn captured(&self) -> Option<HostGraph> {
        self.rt.captured().map(|graph| HostGraph { graph })
    }
}

/// Subscribe this process to CUPTI's hardware event system, before it
/// makes its first CUDA context. Needs libcupti 13.3 or newer.
#[pyfunction]
fn trace_start() -> PyResult<()> {
    match trace::on() {
        true => Ok(()),
        false => trace::start().map_err(runtime_err),
    }
}

#[pymodule]
fn _kern(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Runtime>()?;
    m.add_class::<HostGraph>()?;
    m.add_function(wrap_pyfunction!(trace_start, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
