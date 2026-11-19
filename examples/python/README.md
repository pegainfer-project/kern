# Pure-Python kern runner and server

A standalone pure-Python reference consumer for a kern v7 manifest with no
Rust runtime dependency:

- `manifest.py` — zero-dependency v7 manifest parser, constant expansion, scalar expressions, and `Protocol` projection (pure CPU)
- `executor.py` — single-GPU executor over `cuda-python`, PyTorch, and `safetensors`
- `run.py` — minimal model-agnostic CLI runner over `Protocol`
- `server.py` — minimal stdlib continuous-batching OpenAI HTTP/SSE server over `Executor` and `Protocol`
- `test_manifest.py` — pure-CPU unit tests

## Test

```bash
python3 -m unittest discover -s examples/python -p test_manifest.py
```

## Run

```bash
python3 examples/python/run.py --config /path/to/kern.toml --prompt "The capital of France is"
python3 examples/python/server.py --config /path/to/kern.toml --port 8000
```
