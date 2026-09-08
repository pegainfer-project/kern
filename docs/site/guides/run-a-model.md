# Run a model

`kern run` performs greedy generation for one sequence. A complete artifact is
required: a verified manifest, every referenced kernel module, the weights, and
a Hugging Face `tokenizer.json`.

## Define a local target

Create `kern.toml` at the repository root or another directory above where you
run the command:

```toml
gpu = 0

[targets.demo]
manifest = "artifacts/demo/manifest.json"
kernels = "artifacts/demo/kernels"
weights = ["artifacts/demo/model.safetensors"]
tokenizer = "artifacts/demo/tokenizer.json"

[run]
prompt = "The capital of France is"
steps = 32
chunk = 512
```

Relative paths are resolved from the directory containing `kern.toml`.

## Generate

```sh
./target/release/kern run demo
```

Logs describing verification, allocation, and execution go to stderr. Generated
text goes to stdout, so it can be redirected without mixing it with diagnostics.

Override an individual setting with a flag:

```sh
./target/release/kern run demo \
  --prompt "Write a CUDA kernel for" \
  --steps 64
```

## Choose execution behavior

- `--chunk N` sets the chunked-prefill size, bounded by the manifest.
- `--eager` disables CUDA graph capture.
- `--rows N` selects a declared decode shape. A one-row program is ordinary
  decode; a wider declared program may represent a speculative round.
- `--capacity N` sets state capacity in tokens and is rounded down to the
  manifest's page unit.

Run `kern run --help` for the complete flag list.
