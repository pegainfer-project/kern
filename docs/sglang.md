# Running a kern manifest as an SGLang model

## Summary

A kern manifest runs as the model inside an unmodified SGLang nightly, loaded
through a plugin, the way [vllm.md](vllm.md) does it for vLLM. SGLang keeps
everything outside the forward pass: scheduling, the radix cache, the KV and
recurrent-state pools, CUDA graphs, sampling, the API. kern runs the forward
pass on the tensors SGLang allocates.

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

The plugin registers itself under the checkpoint's own architecture
(`Qwen3_5ForConditionalGeneration`), so SGLang still treats the model as what
it is, and only the class that runs it changes. It does nothing unless
`KERN_MANIFEST` is set.

State as of 2026-10-05, Qwen3.8-27B on one GB300, SGLang nightly `f70e8c68`:

- **Correct, prefix cache on.** Greedy output inside SGLang follows `kern run`
  until a step where `kern run`'s own top-2 is an exact tie. A request
  answered from cached GDN snapshots, whether the snapshot came from a prefill
  or a decode, matches the same request computed cold. 8 concurrent requests
  diverge only at margins of 0.125.
- **Ahead on agent traffic.** An hour of AgentX replay at concurrency 24
  served 10% more requests and 7% more output tokens than native SGLang on
  the same backends, at 6–9% lower inter-token latency p50; against native
  SGLang's own defaults for this model it served 10× as many.
- **Faster at decode, at long prefills and at low concurrency, behind on
  throughput at concurrency ≥ 32 with 1k prompts.** Numbers are in
  [Results](#results).
- **Scope.** Single GPU, one model, mixed chunks off.

The rest explains where SGLang's model interface differs from vLLM's and what
that costs on kern's side.

## How SGLang differs from vLLM

Three differences shape the integration. SGLang excerpts are from `f70e8c68`.

### Memory comes from the config, not from the model

vLLM asks every layer of the model for a cache spec, so the plugin's stub
layers declare kern's layout and vLLM allocates it. SGLang never asks the
model. It recognizes a hybrid GDN model by its HF config class
(`configs/hybrid_arch.py`, `hybrid_gdn_config`) and builds the pools itself:

| tensor | per layer | layout | id |
|---|---|---|---|
| K, V (`MHATokenToKVPool`) | separate arrays | `[tokens, 4, 256]` bf16 | token slot; page = slot / 64 |
| conv (`MambaPool`) | `[slots, 10240, 3]` bf16 | dim-major | mamba slot (0 is a dummy) |
| ssm (`MambaPool`) | `[slots, 48, 128, 128]` f32 | as kern's | same slot |

The model class cannot change these, and several of SGLang's defaults (page
size, the mamba radix-cache policy) key on the architecture name. So the
plugin keeps the name and adapts kern to the pools instead. A stub model run
in the nightly (`--page-size 64`) confirmed every shape and stride above.

### Prefix caching asks the model for snapshots

vLLM's align mode keeps a state page per block and copies state across
blocks itself; the model only ever runs on one page. SGLang's
`extra_buffer` mode, the only one that works with 64-token KV pages, splits
the job:

- **Restoring** a cached state into a request's slot happens before the
  forward (`ModelRunner._maybe_execute_deferred_mamba_cow_and_clear`), and
  fresh slots are zeroed there too. The forward sees one slot per request.
- **Saving** is the model's job. In an extend the scheduler names one
  position per request (64-aligned from the prefix) and a track slot; the
  model must leave the state after that position there. In a decode it must
  copy the slot to its track slot whenever the length crosses the 256-token
  track interval. Nothing checks either; a missing snapshot is only seen when
  a later request restores garbage.

### The decode graph is the whole forward

SGLang's decode graph captures `model.forward`, logits included, so kern's
launches join it through `enqueue_after`'s events as they join vLLM's. Its
prefill graph captures only a submodule with `.layers` and is turned off.
The overlap scheduler runs the forward one step ahead on its own stream, so
the forward reads only the batch's `*_cpu` lists on the host.

## What kern needed

No runtime or manifest-schema change, and no new kernel: only layout builds
of existing ones.

### The manifest transform: `tools/qwen38_sglang.py`

Like `tools/qwen38_vllm.py`, a pure function of the base manifest, sharing
the vLLM transform's program changes (`headless`, `wide`, `fused_in_proj`,
`head`). The memory changes:

- **States.** Four families of host states per layer instead of one:
  `k.l<L>`, `v.l<L>` as `[0, 64, 4, 256]` over SGLang's token arrays;
  `conv.l<L>`, `ssm.l<L>` as `[0, 10240, 3]` and `[0, 48, 128, 128]`.
- **Strides.** Rewritten per op and by position, each against the base's
  value: the K and V tensormaps of both TRT-LLM-gen attention kernels (token
  2048 B, head 512 B, page 131072 B), `attn_prep`'s three strides, the GDN
  line strides (61440 B for conv, 3145728 B for ssm, offset 0).
- **Tables.** Conv and ssm differ in layout, so each gets a line table
  (`gdn.conv_index`, `gdn.ssm_index`) over the same slot ids; `kern bench`
  can provision both (`Verified::self_hosted`).
- **Conv kernels.** Every kernel that touches the conv state reads it
  through `CONV_AT`, which `-DCONV_DIM_MAJOR` switches to SGLang's layout;
  the fused decode kernel (conv and step in one launch) also takes the conv
  state as its own array then. Without the flag each builds to the same
  bytes as before; with it they are the `*_dm` modules, pinned by sha. Their
  sources live next to the cubins in the
  [kern-qwen38-sm103](https://huggingface.co/Pegainfer/kern-qwen38-sm103)
  artifact repo (`sources/`), not here.

The transform takes either base: `examples/qwen3.8-27b.json`, or the
manifest the [vLLM optimization loop](vllm-rsi-example.md) ended with. Both
results are on the artifact repo (`qwen3.8-27b-sglang.json`,
`qwen3.8-27b-rsi-sglang.json`). `kern test` of the RSI one against its vLLM
twin (`qwen3.8-27b-rsi-vllm.json`) is bit-identical at every span: the
layouts differ, the arithmetic does not.

### The plugin: `python/kern_sglang`

An `sglang.srt.plugins` entry point puts `KernForCausalLM` in the model
registry under the checkpoint's architecture. The model:

1. **Loads** the runtime in its constructor, inside SGLang's weight-loading
   window, so SGLang's free-memory reading after the load counts kern's
   weights (53.7 GB) when it sizes the pools.
2. **Binds** on the first eager step and whenever the pools move: each
   host state's tensor from SGLang's pools, then the `once` programs.
3. **Runs a decode step** as one `decode_batch` over SGLang's padded batch:
   the page table is `req_to_token[:, ::64] / 64`, the slot ids are the GDN
   backend's metadata (padded rows are -1, which the kernels skip). After it,
   SGLang's own masked all-layers copy (`track_mamba_states_all_layers`) saves
   the snapshots, inside the graph.
4. **Splits an extend step** on the host's lengths: one-token rows through
   `decode_batch`, every longer row through `prefill`; a tracked row is two
   `prefill` calls cut at its snapshot position, with the copy in between.
5. **Computes logits** through SGLang's `LogitsProcessor` with kern's `head`
   as the lm_head, so pruning, logprobs and hidden-state capture are
   SGLang's.

SGLang still builds its attention and GDN metadata every step; the plugin
reads the slot ids from it and ignores the rest.

## Results

2026-10-05, one GB300 per server (tray04), SGLang nightly `f70e8c68`. Native
SGLang runs its own defaults for this model (page 1, triton attention and
GDN, `extra_buffer`, breakable prefill graphs); both use
`--max-running-requests 128 --chunked-prefill-size 8192
--mem-fraction-static 0.85`. The two servers ran side by side.

### Correctness

The RSI manifest (`qwen3.8-27b-rsi-sglang.json`, what the AgentX runs serve):

- **Against its vLLM twin**, `kern test`: bit-identical at every span and in
  the logits.
- **Against `kern run`** (the loop's manifest, the 38-token prose prompt
  below): 48 of 48 greedy tokens identical.
- **Prefix cache**, the cases below: extend and decode snapshots identical;
  8 concurrent: 6 identical, 2 diverge where the cold run's top two are tied.

The base manifest (`qwen3.8-27b-sglang.json`, the A/B and TTFT tables):

- **Against `kern run`** (base manifest, a 38-token prose prompt, 48 greedy
  tokens): identical for 3 tokens. At the 4th, `kern run`'s top two are an
  exact tie (both −1.2923) and the two take different sides. Before that the
  logprobs agree to within one bf16 step of the logits.
- **Prefix cache** (each case against the same request after `/flush_cache`):
  - a 64-aligned snapshot from a prefill (192 of 345 tokens cached): 64 tokens identical;
  - snapshots from a 300-token decode (512 of 645 cached): 64 tokens identical;
  - 8 concurrent prompts, then 8 continuations (128 cached each): 6 identical,
    2 diverge at margins of 0.125.

### AgentX, one hour

The aiperf command of [the vLLM RSI write-up](vllm-rsi-example.md#results)
(AgentX replay, concurrency 24, 3600 s, seed 20260925), every server with
`--served-model-name qwen38 --enable-cache-report --enable-metrics`. kern
serves the RSI manifest. "Native, kern's backends" is native SGLang with
kern's flags (`--page-size 64 --attention-backend trtllm_mha
--mamba-radix-cache-strategy extra_buffer`). Two pairs ran side by side: kern
run 1 next to native defaults, kern run 2 next to native on kern's backends.

| | SGLang + kern, run 1 / run 2 | native, kern's backends | native, defaults |
|---|---|---|---|
| requests in 1 h (0 errors) | **1070 / 1072** | 970 | 104 |
| output tok/s | **235.2 / 233.0** | 217.8 | 23.1 |
| ITL p50 / p99 ms | **24.5** / 158.5, 25.2 / 179.5 | 26.8 / **146.4** | 261.7 / 943.4 |
| TTFT p50 / p99 ms | 583 / **42696**, 649 / 48720 | **548** / 59053 | 110552 / 483286 |
| request latency p50 ms | **15409** / 16306 | 17418 | 315964 |
| prompt-cache hit | 72.3% / 72.1% | 74.0% | 22.0% |

- **Against native on the same backends** (run 2, which ran beside it) kern
  serves 10.5% more requests and 7.0% more output tokens, ITL p50 6% lower.
  Attention is the same TRT-LLM-gen kernel on both sides, so the difference
  is the rest of the step: kern's GDN, gemm and fused small ops against
  SGLang's. ITL p99 is worse (159–180 vs 146 ms), as under vLLM: an extend
  step is one eager `prefill` call per request.
- **Native's defaults** (page 1, triton attention) do not survive this
  traffic: an 8192-token prefill chunk over a 100k+ prefix runs at ~2300
  tok/s, the queue never drains, the KV pool sits at ~94%, and only 22% of
  prompt tokens hit the cache.
- **Against vLLM.** The same manifest inside vLLM served 1528 requests in
  the loop's hour (another tray, vLLM at 0.92 memory, 93% cache hit). Here
  the hit rate is 72% on both kern and native, so the gap is SGLang's cache
  under this traffic (GDN snapshots only at 64-aligned prompt positions and
  every 256 decoded tokens, 0.85 memory), not kern; not yet profiled.

### Online A/B

vllm-bench, random prompts as token ids, ignore-eos, prefix cache flushed
before each run.

| shape | concurrency | output tok/s, kern / native | TTFT p50 ms | TPOT p50 ms |
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
| kern, ms | 31.1 | 38.6 | 57.1 | 95.2 | 173 | 335 | 677 | 1402 |
| native, ms | 68.8 | 66.6 | 66.6 | 95.3 | 176 | 363 | 892 | 2711 |

Reading these, not yet profiled:

- **Decode.** kern's step is 3–4% faster at 1k and 11–18% at 8k context,
  where native's triton attention reads a longer KV.
- **Long prefill.** Even at 2k–8k, then ahead at 16k and 32k (1.3× and 1.9×),
  where native's triton prefill attention falls behind TRT-LLM-gen.
- **Concurrency ≥ 32 with 1k prompts.** Native leads by 4–5% throughput and
  TTFT: an extend step with several new requests is one `prefill` call per
  request, reading every weight once per request, where native runs the
  batch in one pass. This is the same limit as under vLLM.

## Limits

- **One program call per prefill request**, as under vLLM; a tracked request
  costs one more call. A ragged program over a whole extend step is not
  written.
- **Mixed chunks off.** With `--enable-mixed-chunk`, decode rows ride in
  extend steps, and their decode snapshots are not taken.
- **Single GPU only**; TP is not wired.
- **Prefix-cache hit below vLLM's** on AgentX (72% vs 93%), for native
  SGLang as well; see [AgentX](#agentx-one-hour).
- **SGLang internals.** The plugin reads the pools, the GDN backend's
  metadata and `track_mamba_states_all_layers` from SGLang's modules, so an
  SGLang bump is checked against the stub probe and the gates above.
- **Named contract.** As under vLLM, the adapter finds inputs and programs by
  name; another model needs a transform that produces the same contract
  (`python/kern_sglang/kern_sglang/model.py`).
