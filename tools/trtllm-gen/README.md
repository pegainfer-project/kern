# TRTLLM-GEN full attention for Qwen3.8-27B

Uses unmodified NVIDIA kernels through kern's existing packed-parameter and
TMA descriptor support. Validated on **GB300 (SM103), BF16, Q24/KV4/D256**.
This replaces the model's 16 full-attention layers; its 48 GDN layers keep
their existing implementations. This is full attention, not MLA.

`artifacts.json` pins three cubins served by NVIDIA, their SHA256 checksums,
entry points and launch geometry. No NVIDIA source or binary is vendored.
FlashInfer 0.6.16.post3 is the reference launcher. The prefill artifact is
SM103-specific; other GPUs and attention geometries are not validated.

## Use in an existing kern export

Run from the kern checkout, with CUDA-enabled PyTorch, FlashInfer, vLLM,
Triton and the CUDA toolkit available. Select an idle GPU before compiling
the KV append specialization. `GPU` is a zero-based device index.

```bash
GPU=0
python tools/trtllm_attention.py --download kernels-qwen38
CUDA_VISIBLE_DEVICES=$GPU python tools/trtllm-gen/build_cache.py \
  --output kernels-qwen38
python tools/trtllm_attention.py \
  --input examples/qwen3.8-27b.json \
  --cache-metadata kernels-qwen38/cache.json \
  --output qwen38-trtllm.json
cargo build --release -p kern-run --bin kern --example program_io
target/release/kern run qwen3.8-27b --gpu $GPU \
  --manifest qwen38-trtllm.json --kernels kernels-qwen38
```

The base model's exported weights, tokenizer and other kernel artifacts must
already be configured. The same converter accepts
`examples/qwen3.8-27b-dflash2.json`; use that target's kernel directory and
configured weights. The draft's own attention is preserved.

The page size changes from 784 to **64**. Conversion updates the target page
table, layer offsets and KV append block stride together. Start a fresh
runtime/cache with the generated manifest. `build_cache.py` compiles vLLM's
unmodified Apache-2.0 KV append kernel for this layout and checks writes on
both sides of a page boundary. Copy the new artifacts into the selected
model's existing kernel directory, keeping its other kernels.

## Execution contract

`tools/trtllm_attention.py:op` builds one attention op with parameters
`out, q, k, v, page_table, seq_lens, cu_q, batch, rows`.

- Q/O are contiguous BF16 `[total_query_rows, 24, 256]`.
- KV storage is `[pages, layers, 64, 4, 2, 256]`; the call supplies the layer's
  K/V byte offsets. Tables are contiguous int32 `[batch, ceil(max_context/64)]`.
- Decode consumes one query per sequence. Prefill/verification uses cumulative
  query offsets and bottom-right causal masking against the supplied KV lengths.
- The ABI is one zero-initialized 1,280-byte parameter block with four TMA
  descriptors, buffer pointers and scalar launch metadata. The recipe contains
  the complete byte offsets. Unused bytes stay zero.
- Decode uses **38 fixed splits** by default. `--splits 1` chooses the persistent
  unsplit artifact; larger values select the split-KV artifact. This is a fixed
  profile, not FlashInfer's automatic dispatch policy. Scratch is bounded by
  the declared maximum batch. Repeated graph replay checks counter reuse.

## Validation

```bash
python3 -m unittest discover -s tools -p test_trtllm_manifest.py
CUDA_VISIBLE_DEVICES=$GPU python tools/test_trtllm_attention.py \
  --cubins kernels-qwen38 --work-dir results/trtllm-gates
```

The GPU gate compares the actual Rust runtime's final output after eager
execution and repeated CUDA Graph replay with unmodified FlashInfer. It covers
short decode, randomized disjoint pages, nonzero layer offsets, batch 1/2/16/32,
chunked prefill, 8-token verification, and average-200,000-token decode.
For batch > 1, KV lengths range from 150,000 to 250,000. `--quick` skips those
large cases. The largest case writes about 61 GiB of temporary fixtures;
they are removed after each case. It requires a large-memory GPU.

Require finite outputs, relative L2 < 0.02 and maximum absolute error < 0.02.
Different split/reduction orders need not be bit-identical. Reported latency
is a single attention op's CUDA Graph median; it excludes allocation, fixture
I/O, projections, GDN, KV append and model serving.

## Provenance

- [FlashInfer decode API](https://docs.flashinfer.ai/generated/flashinfer.decode.trtllm_batch_decode_with_kv_cache.html)
- [FlashInfer prefill API](https://docs.flashinfer.ai/generated/flashinfer.prefill.trtllm_batch_context_with_kv_cache.html)
- [vLLM KV append source](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_reshape_and_cache_flash.py)

NVIDIA artifacts are fetched from upstream and remain subject to upstream
terms. This recipe does not grant rights to those artifacts.
