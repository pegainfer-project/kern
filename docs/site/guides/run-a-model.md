# Run a model

`kern run` performs greedy generation for one sequence. A complete artifact is
required: a verified manifest, every referenced kernel module, and the weights.
A checkpoint directory also supplies the tokenizer (`tokenizer.json`) and the
stop tokens (`generation_config.json`); a bare `.safetensors` file needs
`tokenizer` in the target and `--stop-tokens`.

## Define a local target

Create `kern.toml` at the repository root or another directory above where you
run the command:

```toml
gpu = 0

[targets."qwen3.8-27b"]
manifest = "kern-qwen38/manifests/qwen3.8-27b.json"   # hf download Pegainfer/kern-qwen38-sm103 --local-dir kern-qwen38
kernels = "kern-qwen38/cubins"
weights = ["<your_qwen38_checkpoint_path>"]           # HF checkpoint dir(s) or .safetensors files
tokenizer = "<your_qwen38_checkpoint_path>/tokenizer.json"   # optional: defaults to the checkpoint's

[run]
prompt = "The capital of France is"
steps = 32
chunk = 512
```

Relative paths are resolved from the directory containing `kern.toml`.

## Generate

```sh
kern run qwen3.8-27b
```

Logs describing verification, allocation, and execution go to stderr. Generated
text goes to stdout, so it can be redirected without mixing it with diagnostics.

Override an individual setting with a flag:

```sh
kern run qwen3.8-27b \
  --prompt "Write a CUDA kernel for" \
  --steps 64
```

## Choose execution behavior

- `--chunk N` sets the prefill chunk; the default is the manifest's `tokens`
  bound, and a larger value is an error.
- `--eager` launches every program eagerly, ignoring the manifest's `graph`
  flags (a debug switch).
- `--rows N` selects a declared decode shape. A one-row program is ordinary
  decode; a wider declared program is a speculative round (the DFlash2
  manifest's is 8 rows). Default: the widest the manifest declares.
- `--capacity N` sets state capacity in tokens and is rounded down to the
  manifest's page unit.

Run `kern run --help` for the complete flag list.
