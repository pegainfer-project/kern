//! `import kern`: the runtime for a Python host.
//!
//! The binding speaks the runtime's words and nothing else — a manifest,
//! buffers, host states, programs, vars — so it is the same for every host.
//! What a particular engine calls its KV cache, its block table or its
//! stream lives in that engine's adapter, in Python, next to the engine it
//! tracks (`python/kern_vllm` for vLLM): an engine that changes its
//! internals changes the adapter, never this crate.
//!
//! The host reads the verified manifest as JSON ([`Runtime::manifest`]) to
//! learn every buffer's role and layout, writes inputs and reads outputs
//! through their addresses ([`Runtime::region`]) in its own stream order,
//! binds its memory to the host states ([`Runtime::bind_host`]) and orders
//! each program after its own work ([`Runtime::enqueue_after`]). The
//! runtime is not `Send` (it holds CUDA graph and event handles), so every
//! call keeps the GIL and stays on the thread that loaded it.

use std::collections::BTreeMap;
use std::path::PathBuf;

use kern_manifest::types::DType;
use kern_manifest::Verified;
use kern_runtime::{HostRegion, Safetensors};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// A loaded runtime: every buffer allocated and every weight on the device.
/// A manifest with host states runs nothing until they are bound.
#[pyclass(unsendable, module = "kern")]
struct Runtime {
    rt: kern_runtime::Runtime,
}

fn runtime_err(e: kern_runtime::Error) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

#[pymethods]
impl Runtime {
    /// Verify the manifest, load its kernels onto `gpu` and its weights out
    /// of `weights` (safetensors files, or directories of shards).
    #[new]
    #[pyo3(signature = (manifest, kernels, weights, gpu))]
    fn new(manifest: PathBuf, kernels: PathBuf, weights: Vec<PathBuf>, gpu: usize) -> PyResult<Self> {
        let text = std::fs::read_to_string(&manifest)
            .map_err(|e| PyValueError::new_err(format!("manifest {}: {e}", manifest.display())))?;
        let m = Verified::from_json(&text)
            .map_err(|e| PyValueError::new_err(format!("manifest {}:\n{e}", manifest.display())))?;
        let mut rt = kern_runtime::Runtime::load(&m, Some(&kernels), gpu, None, None).map_err(runtime_err)?;
        rt.load_weights(&Safetensors::open(&weights).map_err(runtime_err)?).map_err(runtime_err)?;
        Ok(Runtime { rt })
    }

    /// The verified manifest as JSON.
    fn manifest(&self) -> String {
        self.rt.manifest.to_json()
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
}

#[pymodule]
fn kern(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Runtime>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
