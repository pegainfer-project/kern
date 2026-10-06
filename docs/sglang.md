# Running a kern manifest as an SGLang model

## Summary

A kern manifest can now run as the model inside an unmodified SGLang
nightly, loaded through a plugin, the way [vllm.md](vllm.md) does it for
vLLM. SGLang keeps everything outside the forward pass: scheduling, the
radix cache, the KV and recurrent-state pools, CUDA graphs, sampling and the
API. kern runs the forward pass on the tensors SGLang allocates.

```bash
(cd crates/kern-py && maturin build --release) && pip install target/wheels/kern-*.whl
pip install python/kern_sglang
hf download Pegainfer/kern-qwen38-sm103 --local-dir kern-qwen38
KERN_MANIFEST=kern-qwen38/manifests/qwen3.8-27b-rsi-sglang.json KERN_KERNELS=kern-qwen38/cubins \
sglang serve --model-path Qwen/Qwen3.8-27B \
  --page-size 64 --attention-backend trtllm_mha --mamba-radix-cache-strategy extra_buffer \
  --disable-prefill-cuda-graph --max-running-requests 128 --cuda-graph-max-bs-decode 128 \
  --chunked-prefill-size 8192
```

There is no architecture override. The plugin registers kern's model class
under the checkpoint's own architecture name, `Qwen3_5ForConditionalGeneration`,
so SGLang still treats the model as what it is (pool layout, cache policy,
defaults) and only the class that runs the forward changes. It does nothing
unless `KERN_MANIFEST` is set. The page size and attention backend are set
explicitly because kern's attention kernels are TRT-LLM-gen's on 64-token
pages; SGLang's own defaults for this model are 1-token pages and Triton
attention.

The manifest above is the one the [vLLM optimization loop](vllm-rsi-example.md)
ended with, rewritten for SGLang's memory. Nothing the loop optimized was
specific to vLLM: the base manifest it produced is the input to both
transforms.

State as of 2026-10-05, Qwen3.8-27B on one GB300, SGLang nightly `f70e8c68`:

- **Correct, prefix cache on.** `kern test` of the SGLang manifest against
  its vLLM twin is bit-identical at every span. Greedy output inside SGLang
  equals `kern run` for 48 of 48 tokens, and a request answered from cached
  GDN snapshots matches the same request computed cold.
- **Ahead on agent traffic.** In an hour of AgentX replay, SGLang + kern
  served 10.5% more requests than SGLang running its own model with the same
  flags, and 10× more than SGLang out of the box. Numbers are in
  [Results](#results).
- **Scope.** Single GPU, one model, mixed chunks off. The adapter reads a
  small named contract from the manifest (see [Limits](#limits)).

The rest of this note explains what SGLang's model interface gives an
external runtime and where it differs from vLLM's, then what kern needed on
its side.

## How SGLang hosts it

SGLang's runner, like vLLM's, needs only a forward function from the model.
Three things differ from vLLM, and each decides part of the adapter. Source
excerpts are from SGLang `f70e8c68`, trimmed.

### 1. A plugin can replace a model class

`load_plugins()` runs every entry point in the group `sglang.srt.plugins` in
every process before the model loads (`srt/plugins/__init__.py`). A plugin
function can do anything; kern's puts its class into the model registry
under the checkpoint's architecture:

```python
# python/kern_sglang/kern_sglang/__init__.py
def register() -> None:
    if not os.environ.get("KERN_MANIFEST"):
        return
    ModelRegistry.models[ARCHITECTURE] = KernForCausalLM
```

SGLang then builds the class with the HF config and calls
`forward(input_ids, positions, forward_batch)` every step. The batch's
lengths are on the host as `*_cpu` lists and its ids on the device. The
overlap scheduler runs the forward one step ahead on its own stream, so the
forward may read only the `*_cpu` fields on the host, never a device tensor.

### 2. Memory comes from the config, not from the model

vLLM asks every layer of the model for a cache spec, so the vLLM plugin's
stub layers declare kern's layout and vLLM allocates it. SGLang never asks
the model. It recognizes a hybrid GDN model by its config class and builds
the pools itself:

```python
# srt/configs/hybrid_arch.py
def hybrid_gdn_config(model_config: ModelConfig):
    config = model_config.hf_config.get_text_config()
    if isinstance(config, Qwen3NextConfig | Qwen3_5Config | Qwen3_5MoeConfig | ...):
        return config
    return None
```

For Qwen3.8 that gives a `HybridLinearKVPool` (16 full-attention layers) and
a `MambaPool` (48 GDN layers). A stub model that computes nothing, run in the
nightly with `--page-size 64`, dumped every tensor's shape and strides:

| tensor | per layer | layout | indexed by |
|---|---|---|---|
| K, V | two separate arrays | `[tokens, 4, 256]` bf16 | token slot; page = slot / 64 |
| conv | `[slots, 10240, 3]` bf16 | dim-major: the transpose of kern's `[3][10240]` | mamba slot (0 is a dummy) |
| ssm | `[slots, 48, 128, 128]` f32 | as kern's | the same slot |

The model class cannot change any of this, and several of SGLang's defaults
(page size, the mamba cache policy) key on the architecture name. So the
plugin keeps the name, and kern adapts to SGLang's tensors instead.

### 3. Prefix caching asks the model for snapshots

vLLM's align mode keeps one state page per block and copies state across
blocks itself; the model always runs on one page. SGLang's `extra_buffer`
mode, the only one that works with 64-token pages, splits the job between
the runner and the model.

Restoring is the runner's. Before an extend step it copies a cached state
into the request's working slot, and zeroes fresh slots:

```python
# srt/model_executor/model_runner.py
def _maybe_execute_deferred_mamba_cow_and_clear(self, forward_batch):
    if forward_batch.mamba_clear_indices is not None and len(...) > 0:
        pool.mamba_pool.clear_slots(pool.translate_mamba_indices(forward_batch.mamba_clear_indices))
    if forward_batch.mamba_cow_src_indices is not None and len(...) > 0:
        pool.copy_mamba_state(src, dst)
```

Saving is the model's. In an extend, the scheduler names per request a
position (64-aligned from the cached prefix) and a track slot, and the state
after that position must end up in the track slot. In a decode, every
request whose length crosses the 256-token track interval must have its
slot copied to its track slot. SGLang's own GDN backend does the decode copy
in one masked launch over all layers, which also runs inside a CUDA graph:

```python
# srt/layers/attention/hybrid_linear_attn_backend.py
track_mamba_states_all_layers(
    conv_pool, ssm_pool, cache_indices,
    forward_batch.mamba_track_mask,                  # which rows crossed the interval
    self.forward_metadata.mamba_track_indices,       # their track slots
    forward_batch.batch_size, ...)
```

Nothing checks either snapshot. A missing one is only seen when a later
request restores garbage from it, which is why the prefix-cache gate below
compares cached requests with cold ones.

SGLang's decode CUDA graph captures the whole `model.forward`, logits
included, so kern's launches join it through `enqueue_after`'s events as
they join vLLM's. Its prefill graph captures only a submodule with
`.layers` and is turned off.

## What kern needed

No change to the runtime or the manifest schema. Everything is a manifest
transform, a plugin, and layout builds of existing kernels.

### The manifest transform: `tools/qwen38_sglang.py`

Like `tools/qwen38_vllm.py`, it is a pure function of a base manifest, and
it shares the vLLM transform's program changes (programs stop at the hidden
states, `decode_batch` runs at SGLang's width, logits come from a separate
`head` program). It takes either base: `examples/qwen3.8-27b.json`, or the
manifest the vLLM loop ended with. The memory changes:

- **States.** Four host states per layer instead of one: `k.l<L>` and
  `v.l<L>` as `[0, 64, 4, 256]` over SGLang's token arrays read as pages,
  `conv.l<L>` as `[0, 10240, 3]`, `ssm.l<L>` as `[0, 48, 128, 128]`.
- **Strides.** Rewritten launch by launch, keyed by the kernel it runs and
  checked against the base's value: the K and V tensormaps of both
  TRT-LLM-gen attention kernels, `attn_prep`'s KV strides, the GDN line
  strides (61,440 B for conv, 3,145,728 B for ssm, offset 0).
- **Tables.** Conv and ssm differ in layout, so each gets its own line table
  (`gdn.conv_index`, `gdn.ssm_index`) over the same slot ids.
- **Conv kernels.** Every kernel that touches the conv state reads it
  through a `CONV_AT` macro, which `-DCONV_DIM_MAJOR` switches to SGLang's
  layout. The fused decode kernel (conv and step in one launch) also takes
  the conv state as its own array then. Without the flag each builds to the
  same bytes as before; with it they are the `*_dm` modules, pinned by sha.
  The sources live with the cubins in the artifact repo, not here.

The generated manifests, the `*_dm` cubins and their sources are in the HF
repo [`Pegainfer/kern-qwen38-sm103`](https://huggingface.co/Pegainfer/kern-qwen38-sm103):
`qwen3.8-27b-sglang.json` from the examples base, `qwen3.8-27b-rsi-sglang.json`
from the loop's, next to its vLLM twin `qwen3.8-27b-rsi-vllm.json`.

### The plugin: `python/kern_sglang`

The model class does five things:

1. **Loads** the runtime in its constructor, inside SGLang's weight-loading
   window. SGLang measures free memory after loading to size its pools, so
   kern's weights (53.7 GB) are counted.
2. **Binds** on the first eager step and whenever the pools move: each host
   state gets its tensor from SGLang's pools, then the manifest's `once`
   programs run.
3. **Runs a decode step** as one `decode_batch` over SGLang's padded batch.
   The page table is `req_to_token[:, ::64] / 64`; the slot ids come from
   the GDN backend's metadata, with padded rows at -1, which the kernels
   skip. After it, `track_mamba_states_all_layers` saves the decode
   snapshots, inside the graph.
4. **Splits an extend step** on the host's lengths: one-token rows go
   through `decode_batch`, every longer row through its own `prefill`. A row
   with a snapshot position is two `prefill` calls cut there, with the copy
   to the track slot in between.
5. **Computes logits** through SGLang's `LogitsProcessor`, with kern's `head`
   program as the lm_head, so pruning, logprobs and hidden-state capture
   stay SGLang's.

SGLang still builds its attention and GDN metadata every step. The plugin
reads the slot ids from it and ignores the rest.

## Results

2026-10-05, tray04, one GB300 per server, SGLang nightly `f70e8c68`.
Three server configurations appear below:

- **SGLang + kern**: the plugin and the launch command at the top.
- **SGLang, kern's flags**: SGLang's own Qwen3.8 model with the same flags
  (`--page-size 64 --attention-backend trtllm_mha
  --mamba-radix-cache-strategy extra_buffer`). Attention runs the same
  TRT-LLM-gen kernels as kern's; GDN, gemms and the small ops are SGLang's.
- **SGLang out of the box**: SGLang's own model with no backend flags. For
  this model SGLang picks 1-token pages, Triton attention, Triton GDN and
  `extra_buffer`.

All three use `--max-running-requests 128 --chunked-prefill-size 8192
--mem-fraction-static 0.85`. Servers being compared ran side by side, on
two GPUs of the same tray.

### Correctness

On the loop's manifest (`qwen3.8-27b-rsi-sglang.json`):

- **Against its vLLM twin.** `kern test` replays the vLLM manifest's spans
  on the SGLang one: bit-identical at every span and in the logits. The
  layouts differ; the arithmetic does not.
- **Inside SGLang against `kern run`.** A 38-token prose prompt, 48 greedy
  tokens: identical.
- **Prefix cache.** Each case is compared with the same request after
  `/flush_cache`:
  - a snapshot taken in a prefill (192 of 345 tokens cached): 64 tokens identical;
  - snapshots taken in a 300-token decode (512 of 645 cached): 64 tokens identical;
  - 8 prompts at once, then 8 continuations (128 cached each): 6 identical,
    2 diverge at steps where the cold run's top two logprobs are equal.

The examples-base manifest passed the same gates with two differences. Its
greedy output follows `kern run` for 3 tokens: at the 4th, `kern run`'s own
top two are an exact tie, and the two runs take different sides. Its two
concurrent divergences are at top-1/top-2 margins of 0.125.

### AgentX, one hour

Each server was driven by the aiperf command of
[the vLLM RSI write-up](vllm-rsi-example.md#results) (AgentX replay,
concurrency 24, 3600 s, the same seed). Every server also ran with
`--served-model-name qwen38 --enable-cache-report --enable-metrics`.

| | SGLang out of the box | SGLang, kern's flags | SGLang + kern |
|---|---|---|---|
| requests in 1 h (0 errors) | 104 | 970 | **1072** |
| output tok/s | 23.1 | 217.8 | **233.0** |
| ITL p50 / p99 ms | 261.7 / 943.4 | 26.8 / **146.4** | **25.2** / 179.5 |
| TTFT p50 / p99 ms | 110552 / 483286 | **548** / 59053 | 649 / **48720** |
| request latency p50 ms | 315964 | 17418 | **16306** |
| prompt-cache hit | 22.0% | 74.0% | 72.1% |

The SGLang + kern column ran beside "kern's flags". A second SGLang + kern
hour, run beside "out of the box", served 1070 requests at 235.2 tok/s
(ITL p50 24.5 ms), so the column repeats within 1%.

- **Against SGLang with kern's flags**, kern served 10.5% more requests and
  7.0% more output tokens, with ITL p50 6% lower. Attention is the same
  kernel on both sides, so the difference is the rest of the step: kern's
  GDN, gemm and fused small ops against SGLang's. ITL p99 is worse (180 vs
  146 ms) for the reason it is under vLLM: an extend step runs one eager
  `prefill` call per request.
- **SGLang out of the box** does not keep up with this traffic. An
  8192-token prefill chunk over a 100k-token prefix runs at about 2300
  tokens/s on Triton attention, the queue never drains, the KV pool stays
  about 94% full, and only 22% of prompt tokens hit the cache.
- **Against vLLM.** The same manifest inside vLLM served 1528 requests in
  the loop's hour (on another tray, vLLM at 0.92 memory), with 93% of prompt
  tokens cached. Here both SGLang servers with 64-token pages cache 72–74%,
  so the gap is in how SGLang's cache holds up under this traffic, not in
  kern. Not yet profiled.

### Online A/B

These and the TTFT scan below are on the examples-base manifest, against
SGLang out of the box. vllm-bench, random prompts as token ids, ignore-eos,
prefix cache flushed before each run.

| shape | concurrency | output tok/s, kern / SGLang | TTFT p50 ms | TPOT p50 ms |
|---|---|---|---|---|
| 1024→256 | 1 | 101 / 96 | 58.5 / 66.3 | 9.70 / 9.95 |
| 1024→256 | 8 | 678 / 579 | 371 / 316 | 10.38 / 10.71 |
| 1024→256 | 32 | 1746 / 1814 | 1098 / 725 | 14.09 / 14.87 |
| 1024→256 | 64 | 2387 / 2498 | 1814 / 1334 | 19.80 / 20.47 |
| 1024→256 | 128 | 2764 / 2887 | 3254 / 2554 | 28.19 / 29.38 |
| 8192→256 | 1 | 90 / 75 | 336 / 365 | 9.81 / 11.90 |
| 8192→256 | 8 | 378 / 337 | 1512 / 1644 | 15.28 / 17.35 |
| 8192→256 | 32 | 576 / 524 | 5432 / 5903 | 34.48 / 38.19 |

### Single-request TTFT (one output token)

| input tokens | 256 | 512 | 1k | 2k | 4k | 8k | 16k | 32k |
|---|---|---|---|---|---|---|---|---|
| SGLang + kern, ms | 31.1 | 38.6 | 57.1 | 95.2 | 173 | 335 | 677 | 1402 |
| SGLang out of the box, ms | 68.8 | 66.6 | 66.6 | 95.3 | 176 | 363 | 892 | 2711 |

Reading these, not yet profiled:

- **Decode.** kern's step is 3–4% faster at 1k context and 11–18% at 8k,
  where SGLang's Triton attention reads a longer KV.
- **Long prefill.** Even from 2k to 8k, then ahead at 16k and 32k (1.3× and
  1.9×), where Triton prefill attention falls behind TRT-LLM-gen. This is
  the same effect that sinks SGLang out of the box on AgentX.
- **Concurrency ≥ 32 with 1k prompts.** SGLang leads by 4–5% throughput and
  on TTFT: an extend step with several new requests is one `prefill` call
  per request, reading every weight once per request, where SGLang runs the
  batch in one pass. This is the same limit as under vLLM.

## Limits

- **One program call per prefill request**, as under vLLM; a request with a
  snapshot position costs one more. A ragged program over a whole extend
  step is not written.
- **Mixed chunks off.** With `--enable-mixed-chunk`, decode rows ride in
  extend steps, and their decode snapshots are not taken.
- **Prefix-cache hit below vLLM's** on AgentX (72% vs 93%), for SGLang's own
  model as well.
- **Single GPU only**; TP is not wired.
- **SGLang internals.** The plugin reads the pools, the GDN backend's
  metadata and `track_mamba_states_all_layers` from SGLang's modules, so an
  SGLang bump is checked against the stub probe and the gates above.
- **Named contract.** As under vLLM, the adapter finds inputs and programs
  by name, documented at the top of `python/kern_sglang/kern_sglang/model.py`.
  Another model needs a transform that produces the same contract.
