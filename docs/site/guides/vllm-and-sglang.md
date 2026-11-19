# Host inside vLLM and SGLang

`kern-llm` runs a verified kern manifest as the model inside an unmodified vLLM
(MRv2) or SGLang server. The serving engine keeps scheduling, KV/state pools,
CUDA graphs, sampling, and the HTTP API; kern runs the forward pass directly on
the engine's allocated memory.

![Hosting kern inside vLLM and SGLang](/figures/host-engines.svg)

## Install and fetch a hosted bundle

Install `kern-llm` inside your vLLM or SGLang Python environment and download a
published artifact bundle:

```sh
pip install kern-llm
hf download Pegainfer/kern-qwen38-sm103 --local-dir kern-qwen38
```

The plugins stay dormant unless `KERN_MANIFEST` is set.

## Host in vLLM

```sh
KERN_MANIFEST=kern-qwen38/manifests/qwen3.8-27b-vllm.json KERN_KERNELS=kern-qwen38/cubins \
vllm serve Qwen/Qwen3.8-27B --hf-overrides '{"architectures": ["KernForCausalLM"]}' \
  --max-num-seqs 128 --max-num-batched-tokens 8192 --enable-prefix-caching \
  -cc '{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}'
```

`--max-num-seqs` and `--max-num-batched-tokens` must stay within the manifest's
declared `seqs` and `tokens` bounds.

**How it works:**

- **Plugin registration.** `kern_vllm` registers `KernForCausalLM` through
  `vllm.general_plugins`; `--hf-overrides` selects it while weights still come
  from the Hugging Face checkpoint.
- **Memory binding.** Stub layers declare the manifest's `host` states
  (`LBNHC`, 64-token kernel blocks). Once vLLM allocates the KV/Mamba cache,
  `bind_host` verifies every pointer, shape, and stride before compiling the
  manifest.
- **Stream & graph capture.** `enqueue_after` synchronizes kern's stream with
  vLLM's via CUDA events, so `FULL_DECODE_ONLY` captures kern's decode launches
  directly into vLLM's graphs without PyTorch integration, and `head` returns
  logits to vLLM's sampler.

## Host in SGLang

```sh
KERN_MANIFEST=kern-qwen38/manifests/qwen3.8-27b-rsi-sglang.json KERN_KERNELS=kern-qwen38/cubins \
sglang serve --model-path Qwen/Qwen3.8-27B \
  --page-size 64 --attention-backend trtllm_mha --mamba-radix-cache-strategy extra_buffer \
  --disable-prefill-cuda-graph --max-running-requests 128 --cuda-graph-max-bs-decode 128 \
  --chunked-prefill-size 8192
```

No architecture override is needed: `kern_sglang` replaces the checkpoint's own
model class in SGLang's registry so SGLang keeps its model-specific pool layout
and cache defaults.

**How it works:**

- **Architecture hook.** `kern_sglang` hooks `sglang.srt.plugins` and swaps the
  checkpoint's model class in `ModelRegistry` when `KERN_MANIFEST` is set.
- **Pool adaptation.** SGLang builds its `HybridLinearKVPool` and `MambaPool`
  from the HF config; the `-sglang` manifest declares `host` states matching
  those exact layouts, bound in `prepare_before_cuda_graph_capture`.
- **Radix cache & graphs.** `enqueue_after` joins SGLang's decode CUDA graphs,
  while `extra_buffer` prefix caching splits prefill at snapshot positions and
  runs `track_mamba_states_all_layers` inside the decode graph.

To record GPU-stamped per-kernel Parquet traces from a hosted SGLang server,
see [Trace an SGLang server](/guides/trace-sglang).

## vLLM vs SGLang hosting

| | vLLM (`kern_vllm`) | SGLang (`kern_sglang`) |
| --- | --- | --- |
| Activation | `--hf-overrides '{"architectures": ["KernForCausalLM"]}'` | Automatic when `KERN_MANIFEST` is set |
| Cache allocation | Stub layers declare layout; vLLM allocates to match | SGLang allocates pools from HF config; manifest adapts |
| Prefix caching | `--enable-prefix-caching` (align mode) | `--mamba-radix-cache-strategy extra_buffer` |
| CUDA graphs | `-cc '{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}'` | Decode graphs on; `--disable-prefill-cuda-graph` |
| Multi-rank | One kern rank per TP rank over `torch.distributed` gloo | One kern rank per TP rank over `torch.distributed` gloo |
| Kernel tracing | Step histograms (`KERN_TRACE`) | GPU-stamped Parquet traces (`KERN_TRACE_DIR`, [guide](/guides/trace-sglang)) |
