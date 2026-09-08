# Quick start

Run **Qwen3.8-27B** on one Blackwell GPU with a released `kern` binary, the
kernels and manifest published for it, and the checkpoint you already have.
No Python, no PyTorch, no CUDA toolkit: the whole path below was walked in a
bare `nvidia/cuda:13.0.1-runtime-ubuntu24.04` container.

::: tip What you need
- Linux x86_64 or aarch64, an NVIDIA driver for CUDA 13 (r580 or newer)
- `libcublas.so.13` / `libcublasLt.so.13` on the loader path (a CUDA 13
  toolkit or runtime image; or `pip install nvidia-cublas-cu13` plus
  `LD_LIBRARY_PATH`)
- A GPU the published kernels were built for: the Qwen3.8 registry is
  `sm_103a` (GB300)
- The Qwen3.8-27B checkpoint (`hf download Qwen/Qwen3.8-27B --local-dir …`,
  or the snapshot already in your Hugging Face cache)
:::

## 1. Install kern

```sh
curl -fsSL https://kern-baa.pages.dev/install.sh | sh
kern --version          # kern 0.1.0 (<commit>, cuda 13.0)
```

One file lands in `~/.local/bin`. Prefer to build it yourself? See
[Install](/getting-started/install): `cargo build --release` needs Rust and
a C++ compiler, still no toolkit.

## 2. Get the model's kernels and manifest

```sh
hf download Pegainfer/kern-qwen38-sm103 --local-dir kern-qwen38
```

`kern-qwen38/manifests/` holds the two manifests (plain decode, and decode
with the DFlash2 draft); `kern-qwen38/cubins/` holds every kernel they pin,
content-addressed by SHA-256.

## 3. Generate

```sh
kern run --manifest kern-qwen38/manifests/qwen3.8-27b.json \
         --kernels  kern-qwen38/cubins \
         --weights  <your_qwen38_checkpoint_path> \
         --prompt "The capital of France is" --steps 64
```

`--weights` is the Qwen/Qwen3.8-27B checkpoint directory as downloaded: the
manifest says which tensor fills which buffer, and `tokenizer.json` and
`generation_config.json` beside the shards supply the tokenizer and the stop
tokens. Nothing is converted.

Diagnostics go to stderr: the manifest verified, every kernel resolved by
its pinned hash, the weights assembled straight out of the safetensors
shards, the tokenizer and stop tokens found, the decode step captured as
one CUDA graph. The generated text goes to stdout:

```text
The capital of France is Paris.
The capital of Germany is Berlin.
The capital of Italy is Rome.
…
```

`--gpu N` picks the device (default 0). `--prompt` is raw text, no chat
template; wrap it yourself if you want the instruct format.

## 4. Speculative decoding (optional)

Qwen3.8-27B has a DFlash2 draft model. Download it, add it as a second
`--weights`, switch to the speculative manifest and ask for 8-row rounds:

```sh
hf download incoai/Qwen3.8-27B-DFlash2 --local-dir dflash2

kern run --manifest kern-qwen38/manifests/qwen3.8-27b-dflash2.json \
         --kernels  kern-qwen38/cubins \
         --weights  <your_qwen38_checkpoint_path> --weights dflash2 \
         --rows 8 \
         --prompt "The capital of France is" --steps 64
```

Each step drafts 7 tokens and verifies them in one pass; greedy decoding
means the text is the same as plain decode. No extra runtime code is
involved: the draft, verify and accept logic is a program in the manifest.

Measured on one GB300 with the commands above (2026-09-08):

| | plain | DFlash2, `--rows 8` |
| --- | --- | --- |
| decode | 105 tok/s | 261 tok/s (4.0 tokens accepted per step) |
| weights | 51.7 GiB mapped, assembled in 0.9 s | plus the 3.7 GiB draft |

## 5. Save the flags as a target

Write a `kern.toml` next to where you run, and the commands shrink to a
name:

```toml
[targets."qwen3.8-27b"]
manifest = "kern-qwen38/manifests/qwen3.8-27b.json"
kernels  = "kern-qwen38/cubins"
weights  = ["<your_qwen38_checkpoint_path>"]

[targets."qwen3.8-27b-dflash2"]
manifest = "kern-qwen38/manifests/qwen3.8-27b-dflash2.json"
kernels  = "kern-qwen38/cubins"
weights  = ["<your_qwen38_checkpoint_path>", "dflash2"]
```

```sh
kern run qwen3.8-27b --steps 64
kern run qwen3.8-27b-dflash2 --steps 64      # --rows defaults to the widest declared round
```

Paths are resolved relative to the `kern.toml`. The full file format is in
the [`kern.toml` reference](/reference/config).

## 6. Serve over HTTP (optional)

`kern-serve` puts the same manifest behind an OpenAI-compatible endpoint
with continuous batching and speculative rounds under load. It is a separate
workspace built from source (`cd crates/kern-serve && cargo build --release`)
and takes the target above plus `--model-path <your_qwen38_checkpoint_path>`
for the chat template. See `docs/serve.md` in the repository.

## What you just ran

The binary contains no model. Everything specific to Qwen3.8 (the layer
structure, which kernel runs where, how the draft is verified, which
checkpoint tensor fills which buffer) is in the manifest, and the runtime
verified it before executing. Read [The model artifact](/concepts/artifact)
for what that contract is, and [Test a kernel change](/guides/test-a-kernel-change)
for how a swapped kernel earns its way into a manifest.
