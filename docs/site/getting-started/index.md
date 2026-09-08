# Quick start

Qwen3.8-27B on one Blackwell GPU, served over an OpenAI-compatible endpoint.
No Python, no PyTorch, no CUDA toolkit: the binary loads the driver and cuBLAS
at first use, and everything specific to the model is in the manifest it
verifies before running.

::: tip What you need
- Linux x86_64 or aarch64, an NVIDIA driver for CUDA 13 (r580 or newer),
  `libcublas.so.13` on the loader path (a CUDA 13 toolkit or runtime image)
- A GB300 (`sm_103a`): what the published Qwen3.8 kernels were built for
:::

## Install

```sh
curl -fsSL https://kern-baa.pages.dev/install.sh | sh      # kern: run, test, bench, verify
```

`kern-serve` is built from source for now: `cd crates/kern-serve && cargo build --release`
(details in [Install](/getting-started/install)).

## Get the model

```sh
hf download Pegainfer/kern-qwen38-sm103 --local-dir kern-qwen38     # kernels + manifests
hf download Qwen/Qwen3.8-27B                                        # prints the checkpoint path, if you don't have one yet
```

Name the pieces once in a `kern.toml`:

```toml
[targets."qwen3.8-27b"]
manifest = "kern-qwen38/manifests/qwen3.8-27b.json"
kernels  = "kern-qwen38/cubins"
weights  = ["<your_qwen38_checkpoint_path>"]
```

The checkpoint is used as published: the manifest says which tensor fills
which buffer, and `tokenizer.json` and `generation_config.json` beside the
shards supply the tokenizer and the stop tokens.

## Serve

Start the server on one GPU:

```sh
kern-serve qwen3.8-27b --model-path <your_qwen38_checkpoint_path> --gpus 0 --port 8000
```

Then, from another shell, a completion:

```sh
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.8-27b", "prompt": "The capital of France is", "max_tokens": 32}'
```

A chat turn, answered directly (Qwen3.8 thinks first unless told not to):

```sh
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.8-27b", "messages": [{"role": "user", "content": "Write a haiku about Rust."}],
       "max_tokens": 64, "chat_template_kwargs": {"enable_thinking": false}}'
```

The same, streamed:

```sh
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.8-27b", "messages": [{"role": "user", "content": "Write a haiku about Rust."}],
       "max_tokens": 64, "stream": true}'
```

With thinking on, the reasoning arrives as `reasoning` deltas and the answer
as `content`. `/v1/models` lists what is served, `/metrics` is Prometheus.

`--model-path` is the checkpoint directory again, read by the front end for
the chat template. `--gpus 0,1,2,3` drives several GPUs in lockstep when the
manifest declares a topology; this one is single-GPU.

## Generate without a server

```sh
kern run qwen3.8-27b --prompt "The capital of France is" --steps 64
```

One greedy sequence. Diagnostics go to stderr (the manifest verified, every
kernel resolved by its pinned hash, the weights assembled from the shards,
the decode step captured as one CUDA graph); the text goes to stdout.
Without a `kern.toml`, `--manifest`, `--kernels` and `--weights` name the
same three things on either command.

## Speculative decoding

Qwen3.8-27B has a DFlash2 draft. One more download and one more target:

```sh
hf download incoai/Qwen3.8-27B-DFlash2
```

```toml
[targets."qwen3.8-27b-dflash2"]
manifest = "kern-qwen38/manifests/qwen3.8-27b-dflash2.json"
kernels  = "kern-qwen38/cubins"
weights  = ["<your_qwen38_checkpoint_path>", "<your_dflash2_checkpoint_path>"]
```

```sh
kern-serve qwen3.8-27b-dflash2 --model-path <your_qwen38_checkpoint_path> --gpus 0 --port 8000
kern run   qwen3.8-27b-dflash2 --prompt "The capital of France is" --steps 64
```

Every step drafts 7 tokens and verifies them in one pass; the manifest
carries the draft, verify and accept programs, the runtime is unchanged.
Greedy output is the same text as plain decode. On one GB300:

| | plain | DFlash2 |
| --- | --- | --- |
| single sequence | 105 tok/s | 261 tok/s, 4.0 tokens accepted per step |

## What you just ran

The binary contains no model. Which kernel runs where, how the draft is
verified, which checkpoint tensor fills which buffer: all of it is in the
manifest, and the runtime verified it before executing. Read
[The model artifact](/concepts/artifact) for what that contract is, and
[Test a kernel change](/guides/test-a-kernel-change) for how a swapped kernel
earns its way into a manifest.
