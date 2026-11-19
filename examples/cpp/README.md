# Pure-C++20 kern runner

A standalone, zero-dependency C++20 reference consumer for a kern v7 manifest:

- `json.hpp` — minimal RFC 8259 JSON parser
- `manifest.hpp` — manifest parser, constant expansion, scalar expressions, and `Protocol` projection (pure CPU)
- `cuda_api.hpp` — `dlopen` bindings for `libcuda.so.1` and `libcublasLt.so` (no compile-time CUDA toolkit required)
- `executor.hpp` — single-GPU executor with built-in SHA-256 cubin verification, `mmap` safetensors loading, TMA tensormap encoding, cuBLASLt BF16 GEMM, and CUDA Graphs
- `run.cpp` — token-ID CLI runner (`kern-run`) driving prefill + decode / speculative rounds through `Protocol`
- `test_manifest.cpp` — pure-CPU unit tests

## Build & test

```bash
make -C examples/cpp test
```

## Run

```bash
./examples/cpp/kern-run \
  --manifest /path/to/manifest.json \
  --kernels /path/to/cubins \
  --weights /path/to/checkpoint \
  --tokens 791,6864,315,9822,374 \
  --steps 32
```
