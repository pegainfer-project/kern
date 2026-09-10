# DSV4.1 attention and indexer artifacts

These builders instantiate pinned upstream kernels; serving uses only extracted
cubins and schema ops. C++ shared libraries are oracle/capture bridges, never
runtime dependencies. Paths and model checkpoints are caller arguments.

Pinned sources:

- FlashMLA `4f38f29ef6793c228363e4af5be66d44e81167ba`, PR #221.
- DeepGEMM `ab69f76be5bb9ea3499bc755002b1a876cb0b3d9`, PR #432.
- DeepSelect `8e70df71d2a4b0c969ef96dc3b8998efa09a3315`; this revision removed
  `do_check_nan`, so older `TopkSelectArgs` offsets must not be reused.

All `*_ops.py` builders return `modules, ops`; interfaces and layouts are
specified in their function docstrings. The original `ops.py` does the same.
The caller supplies the cubin path and row-count expression. CUDA graph/model
state policy stays outside these implementations.

| File | Implementation | Verification completed |
|---|---|---|
| `ops.py` | FlashMLA BF16 sparse attention,64 heads/512dim | Official sparse-attention oracle; four kern replays byte-exact |
| `paged_ops.py` | FP8 window + FP4 compressed cache, no KV gather | Official quant+attention oracle; rows1/12/32 kern byte-exact with64/64,128/64 pages and window-only |
| `fused_ops.py` | Q RoPE + paged attention + inverse O RoPE + MXFP8 cast | Official RoPE/quant/attention oracle; output/live scales byte-exact, including window-only |
| `fused_prefill_ops.py` | Fused BF16-KV prefill counterpart | Official RoPE/attention oracle; three kern replays byte-exact, including live scales |
| `indexer_ops.py` | DeepGEMM dense MXFP4 index scores,32heads/128dim | Actual kern scores vs official quantization + model score expression |
| `sparse_ops.py` | DeepGEMM candidate-only MXFP4 index scores | Actual kern score+partial-block mask+DeepSelect+position mapping checked |
| `paged_indexer_ops.py` | Direct paged dense MXFP4 scores + BF16/F32 weight adapter | page64/128 real kern rows4/12/32 vs supplied oracle; relRMS≤0.00168 |
| `paged_sparse_ops.py` | Direct paged index-K sparse scoring,64/128 pages | Both pages actual kern vs supplied oracle; relRMS0.00448 |
| `select_ops.py` | DeepSelect f32/BF16 top512/2048, sorted indices | Empty/short/tie/latest-block tests; all four kern replays byte-exact |
| `candidate_ops.py` | Block max8 + pin latest + filter invalid selections | Whole chain with DeepSelect matches actual `model.select_candidate_blocks` |

FlashMLA Q RoPE is applied to the last64 logical head dimensions. The plain and
paged variants consume ordinary head-major Q and return BF16 needing inverse
RoPE. Fused variants consume dim16/head-permuted Q and return group/dim32/head
FP8 data with packed UE8M0 scales; they require the corresponding Q-B/O-A weight
permutation. Their final FP8 quantization accounts for roughly2.66% relative RMS
against the BF16 official attention result on test inputs. This is an operator
check, not an end-to-end model accuracy or speculation-acceptance result.

For direct paged attention, FP8 window pages must be multiples of32 tokens and
FP4 pages multiples of8. Prefer original page128 and ratio2 compressed page64.
Each page stores all token data, then all token scales (not interleaved).
Independent page sizes are supported;128/64 ordinary paged replay passed.
Fused128/64 replay also passed for rows1/12/32 (output byte-exact).
Non-split kernels are provided; low-batch split-KV optimization remains open.

DeepSelect uses an exclusive per-row end, fills short rows with-1, and does not
promise a tie order. Tests check cutoff membership and unique sorted indices,
not equality with torch's unspecified tie choices. top2048 uses upstream's
correctness-only tier; no performance claim is made for that tier. Scores
selected at-infinity need filtering before use as candidates.

DeepGEMM's dense source scorer returns F32; its sparse scorer returns BF16 in
candidate-slot coordinates. Sparse slots must be mapped through the selected
block array. The sparse kernel writes the full final8-token block, so an explicit
postmask must reject positions beyond the per-row end before selection.

Build and verification:

```
bash build.sh "$FLASHMLA_CHECKOUT" "$ARTIFACTS"
bash build_indexer.sh "$DEEPGEMM_CHECKOUT" "$ARTIFACTS"
bash build_sparse.sh "$DEEPGEMM_CHECKOUT" "$ARTIFACTS"
bash build_paged_indexer.sh "$DEEPGEMM_CHECKOUT" "$ARTIFACTS"
bash build_select.sh "$DEEPSELECT_CHECKOUT" "$ARTIFACTS"
python3 compare.py --library "$ARTIFACTS/libdsv41_sparse_prefill.so" \
    --inference "$MODEL/inference" --output "$ARTIFACTS/precision.json" \
    --dump "$ARTIFACTS/io"
bash replay.sh "$PROGRAM_IO" "$ARTIFACTS"
```

The corresponding `paged_*`, `fused_*`, `indexer_*`, `select_*`, and `candidate_*`
scripts cover their operators. Run program_io replays serially on a GPU: its
state allocator reserves the free device pool even for these small manifests.

Upstream source retains its original license and is not vendored here.
`patch_fused.py` broadens two static assertions inside discarded generic-lambda
arms for NVCC13.0; it changes no arithmetic, memory access or control flow.

Direct paged index scoring (`paged_indexer_ops.py`, dense, and
`paged_sparse_ops.py`, candidate-only) uses an index cache with data[P,64],
UE8M0 scales[P,4], then padding to a512-byte page stride:4608B for P64,8704B
for P128. The dense scorer accepts a second cache alias at offset P*64 for its
scale TensorMap. Neither path gathers the whole key cache. Dense per-query
page-table rows support multiple requests and dynamic prefill/draft rows.
`dsv41_index_weights` converts BF16 projection output into BF16 and F32
weights, both scaled by1/64. The dense public query scales use fp8e8m0; sparse
accepts `scale_dtype='fp8e8m0'` for the same raw UE8M0 layout.

When extra_topk=0, paged and fused attention explicitly null the optional extra
length pointer. Upstream gives a non-null per-row pointer precedence over the
zero width, so passing an aliased window-length buffer otherwise enables an
invalid extra-cache TMA load.

`checkpoint_indexer.py` exercises the actual layer2 checkpoint and original
`Indexer.forward`, then replays the complete lowering from `dsv41/indexer.py`
through query projection, RoPE, quantization, weight projection, paged scoring,
DeepSelect, and physical-page mapping. Five independent decode queries passed
with score relative RMS0.002872 and top512 set agreement100%,100%,100%,
99.8047%,99.8047%; the final two queries each differ at one cutoff position.
The index output is consequently not claimed bit-exact with the BF16 reference.
Physical-page mapping is exact. The deterministic harness supplies prepopulated
paged cache bytes as an input pointer; full state publishing is covered by the
coordinator's integration, not this probe. `program_io` does not automatically
execute once programs, so this probe explicitly prepends scale-pack calls.

State-cache TensorMaps infer the physical allocation from the runtime span
(last dimension0). Logical context capacity can exceed the currently allocated
physical pool. Raw-buffer probes retain explicit capacities. FlashMLA's
num_blocks fields are used only by upstream host descriptor construction; its
device kernels address pages through indices and the encoded TensorMap.

`checkpoint_dspark.py` checks all three checkpoint DSpark attention layers
against the supplied `DSparkAttention.forward`. Each26-call probe includes
weight-scale preparation, real context KV initialization, draft metadata,
Q/KV projections, RoPE, window-only attention, and both output projections.
Two requests at anchor positions7 and137 cover short and wrapping context.
All window indices and positions a..a+4 match exactly, and every query sees
all five draft rows through the padded192-column noncausal list. Actual kern
and graph replay completed for all three layers. Attention-branch output
relative RMS was4.5728%,4.4929%,4.7162% against the reference; the optimized
path quantizes O-A activations to FP8. These measurements do not establish
whole-model accuracy or speculative acceptance rates.
# Original BF16 WO_A

`woa.py` restores each original FP8/E8M0 WO_A matrix to a BF16 carry buffer
in the once program. Eight strided cuBLASLt calls then consume BF16 attention
output directly, preserving the supplied inference's grouped einsum semantics.
This replaces the earlier extra activation-FP8 quantization for target and
draft WO_A. The extra carry storage is 64 MiB per layer; no checkpoint export
is required.

`woa_compare.py` checked actual Runtime execution and graph replay against the
original checkpoint dequantization and `einsum('bsgd,grd->bsgr', ...)` for both
target layer 0 and draft layer 0, at 1, 5, 17 and 128 rows. All eight weight
restorations were byte-exact; output relative RMS was at most `9.25e-5` (zero
for 17 rows). These are operator checks, not a full-model precision result.
