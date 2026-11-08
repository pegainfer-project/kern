# Manifest and kernel tools

Dense Qwen3 models live in [kern-qwen3](https://github.com/pegainfer-project/kern-qwen3).
The old Qwen3-4B and DSpark capture scripts and fixtures remain in Git history;
the standalone repository implements dense Qwen3, without the legacy DSpark path.
Qwen3.8 and its DFlash2 draft remain here.
Use [`kern bench`](../docs/bench.md) for per-kernel measurements.

| Tool | Contract |
|---|---|
| `capture_qwen38.sh`, `capture_qwen38_spec.sh` | Capture target and draft launches through `kernel-capture/`. |
| `gen_qwen35.py` | Generate the Qwen3.8 target and DFlash2 manifests from captured kernel ABIs. |
| `mine_capture.py` | Analyze launch records, pointer lifetimes, and launch geometry. |
| `kern_manifest.py` | Pin captured modules by content and normalize manifest interfaces. |
| `qwen_weights.py`, `qwen_constants.py` | Bind Qwen3.8 checkpoint tensors, derive missing tables, and name dimensions. |
| `once.py`, `kernels-src/weight_prep.cu` | Build load programs for derived tables and weight transforms. |
| `sampling.py`, `flashinfer-sampling/`, `kernels-src/sample_glue.cu` | Shared sampling chain and kernel sources: temperature, top-k, top-p, min-p, and seeded sampling. See `flashinfer-sampling/PROVENANCE.md` for the captured launch geometry and seed contract. |
| `kernels/`, `../kernel_index/` | Import, check, and select content-addressed kernel variants; see [registry.md](../docs/registry.md). |
| `extract_kernels.sh` | Resolve pinned modules from capture dumps and local builds. |
| `trtllm_fmha_abi.py`, `gen_trtllm_fmha_probe.py` | Pack the TensorRT-LLM ragged attention ABI and generate its probe manifest. |
| `capture_abi_probe.sh`, `capture_sglang.sh`, `capture_sglang.py` | Inspect kernel ABIs across attention backends and engines. |
| `toy/`, `e2e/` | Model-independent runtime and serving gates. |

Kimi-K3 tooling lives in [kern-k3-pp-prefill](https://github.com/pegainfer-project/kern-k3-pp-prefill).
The small cut fixtures under `crates/kern-manifest/tests/fixtures/k3/` retain
coverage of its runtime contracts. The synthetic `fixtures/paged.json` tests
pipeline cuts and workload planning without a model dependency; its opaque
externs are for host tests only.
