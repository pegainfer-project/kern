# Run without Rust: Python & C++20

A kern manifest is a typed, language-agnostic JSON contract of buffers, states,
ops, and programs. The Rust runtime is one consumer of that contract, not the
only one: anything that can parse the manifest, verify its kernel hashes, and
issue CUDA launches can run the same model artifact unchanged.

![Manifest consumers](/figures/manifest-consumers.svg)

Two standalone reference consumers live under `examples/`—one in pure Python
and one in zero-dependency C++20—showing how little code it takes to drive a
manifest without `crates/`.

## Pure Python (`examples/python`)

`examples/python/` pairs a zero-dependency v7 manifest parser and `Protocol`
projection (`manifest.py`) with a single-GPU executor over `cuda-python`,
PyTorch, and `safetensors` (`executor.py`).

Run the pure-CPU manifest unit tests:

```sh
python3 -m unittest discover -s examples/python -p test_manifest.py
```

Generate text from a `kern.toml` target:

```sh
python3 examples/python/run.py \
  --config /path/to/kern.toml \
  --prompt "The capital of France is"
```

Start the stdlib continuous-batching OpenAI HTTP/SSE server:

```sh
python3 examples/python/server.py \
  --config /path/to/kern.toml \
  --port 8000
```

## Pure C++20 (`examples/cpp`)

`examples/cpp/` is a self-contained C++20 runner with no external build-time
dependencies—not even the CUDA toolkit. It parses JSON and projects `Protocol`
in headers (`json.hpp`, `manifest.hpp`), loads `libcuda.so.1` and
`libcublasLt.so` via `dlopen` (`cuda_api.hpp`), and verifies cubin SHA-256
hashes, mmaps safetensors, encodes TMA tensormaps, and captures CUDA graphs
(`executor.hpp`).

Build `kern-run` and run its pure-CPU unit tests:

```sh
make -C examples/cpp test
```

Run greedy generation from token IDs:

```sh
./examples/cpp/kern-run \
  --manifest /path/to/manifest.json \
  --kernels /path/to/cubins \
  --weights /path/to/checkpoint \
  --tokens 791,6864,315,9822,374 \
  --steps 32
```

## Consumer comparison

| | Rust (`kern-serve` / `kern`) | `examples/python` | `examples/cpp` |
| --- | --- | --- | --- |
| Role | Production runtime & server | Reference Python runner & server | Zero-dependency C++20 runner |
| GPU scope | Single GPU, TP/EP, PP, P/D | Single GPU | Single GPU |
| Runtime dependencies | NVIDIA driver + `libcublas` (`dlopen`) | `cuda-python`, PyTorch, `safetensors` | NVIDIA driver + `libcublasLt` (`dlopen`) |
| Entry points | `kern-serve`, `kern run`, `kern test`, `kern bench` | `run.py`, `server.py` | `kern-run` (token IDs) |
| Serving features | OpenAI API, prefix cache, spec decode, GPU tracing | Minimal continuous-batching OpenAI HTTP/SSE | CLI prefill + decode / spec rounds |
| CUDA graphs | Yes | Yes | Yes |
