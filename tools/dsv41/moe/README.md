# V4.1 MoE and mHC operators

Source pin: DeepGEMM PR #432, `ab69f76be5bb9ea3499bc755002b1a876cb0b3d9`.

The Mega kernels (MoE, mHC, the draft gate) are persistent: one CTA per SM,
and their dispatch and split reductions barrier over all of them, so the SM
count is a template parameter and the launch grid. An instance built for more
SMs than the device has waits for CTAs that never become resident, which the
driver ends as a grid sync timeout. Each cubin therefore carries one instance
per device kern serves — GB300 152, B300 148 — and `gen.py --sms` picks the
set the manifest pins, together with the matching `moe_layout_<E>_sm<N>.json`
(the MoE ring is sized from the pool the persistent grid drains).
Build with `DEEPGEMM_ROOT=<checkout-with-submodules> tools/dsv41/moe/build.sh`.
Generated cubins go under ignored `target/cubins/dsv41`, never into the repository.

* `ops.pieces(rows=...)`: small boundary operations. SwiGLU's final scalar is the element count, unlike HC operations' token count. Uses training clamp 10 (gate upper-only, up bilateral).
* `mhc.pieces(rows=..., max_tokens=..., fp8='gemm'|'moe')`: shifted upstream mHC, fused previous post + current mix + shifted collapse + RMSNorm, storing the normalized rows as BF16 and as MXFP8 for the consumer: column-major GEMM scales for the attention projections, or routed + shared-expert scale pages written straight into the MoE slab. The FP8 is byte-identical to `dsv41_dense_quant_x` / `dsv41_mega_quant_x` applied to the BF16 output (`test_mhc.py`), which the forward path therefore no longer launches. Public ABI documented in the module. Weight fn is F32 with TF32 TMA conversion. Requires kern TF32 tensor-map support. Partials scratch is implementation-private; the split barriers are the last public parameter, a caller-owned `u64[524288]` zeroed once at load (as upstream's per-stream allocation), so a call is one launch. `gate.pieces` takes its `u64[8192]` score barriers the same way; `ops.zero(count)` is the load-time clear.
* `gate.pieces(rows=..., experts=384|128, max_tokens=...)`: upstream fused BF16 router + sqrtsoftplus + bias selection + unbiased normalized top-k weights times 1.5. Output indices are I64. Gate weights are BF16 in the checkpoint. Text routing only; vision bias is not wired.

Numerical tests load the model's supplied `inference/model.py` using `--inference`. `test_boundary.py` calls actual `Block.hc_pre/hc_post` and `Expert.forward`; `test_mhc.py` calls actual Block methods and RMSNorm plus supplied TileLang Sinkhorn; `test_gate.py` calls actual Gate.forward. The AOT tests exercise the public op metadata's packed launch ABI, not a separate JIT kernel.

Validated on GB300: boundary tokens 1/5/33/128; mHC and target/draft gate 1/5/65/128. MHC maximum relative squared error across all outputs below 2e-9; gate indices exact, weights maximum absolute error below 9e-7. This is operator correctness evidence, not full model verification or performance certification.

MegaMoE integration is still in progress. PR #432 uses MXFP8/FP4 scales with K groups of 32 and can fuse one FP8 shared expert. The old K3 packing and activation ABI cannot be used unchanged.

The EP4 MegaMoE cubin now builds, and `moe.pieces(experts=...)` supplies
its full typed ABI and exact upstream slab layout. `test_moe.py` needs an
exclusive four-GPU allocation; it has not yet completed numerical validation.
Weight prep is byte-exact against upstream pure Torch transforms for both
routed FP4 and shared FP8, including W1/W2 scales. Raw routed bindings are
`i8`, shared weight bindings `fp8e4m3`, and raw scale bindings `fp8e8m0`;
prepared weight buffers are `u8`, packed scale buffers are `i32`. The grid's
y axis is the expert: one launch per transform prepares a layer's whole
expert slab, and the single-expert launch is a grid of height one. The
batched launches were checked byte-exact against the per-expert ones at
the real 96-expert size (`~/bench_results/2026-09-10-dsv41-batched-prep`).

EP4 validation completed for both 128/top3 and 384/top6, with counts
`[1,1,1,1]` then `[1,5,0,3]` in one process. Empty rank participation and
repeated slab reuse completed. Distinct per-rank random quantized weights
were compared through original `Expert.forward`, `model.linear`, and original
FP4/FP8 TileLang GEMMs. Maximum relative squared output error was 3.565e-4
for draft, 1.989e-4 for target (about 1.89% / 1.41% relative L2).
This does not establish full-model accuracy: MegaMoE fuses FP8 intermediates,
where the reference materializes BF16 before re-quantizing, and this harness
reuses each rank's synthetic weights across its local experts. Real checkpoint
validation and acceptance-length tests remain necessary. The initial kernel
uses BLOCK_M=16; large prefill needs additional tuned instances.

## Dense and O-A projection

`dense.pieces(rows, n, k, ...)` is one unmodified upstream MXFP8 GEMM
with dynamic M/N/K and configurable row strides. The public interface is
`out BF16, A FP8, B FP8, A SF I32, B SF I32, M, N, K`.
`dense.oa_pieces(rows)` is a single batched launch for
`[T,8,4096] × [8,1024,4096] -> [T,8,1024]` that writes MXFP8 directly:
DeepGEMM's dynamically-scaled epilogue casts D to E4M3 and emits per-32
UE8M0 scales as I32 words indexed by `(batch*N+n)/128` and the row, which
is exactly the `[K/128,align(capacity,4)]` A-scale layout `dsv41_dense`
reads over the flattened 8192-wide output. WO_B therefore consumes O-A's
result with no cast between them, and the ABI gains a trailing SF output:
`out FP8, A, B, A SF, B SF, M, N, K, out SF`.
Unlike MegaMoE, dense SF has **no UTCCP row permutation**: its kernel
transposes internally. Physical scale layout is `[groups,K/128,N]` for
weights and `[8*K/128,align(capacity,4)]` for O activations.
`dense.prep_pieces` provides load-time raw E8M0 weight scale packing and
per-call BF16 activation quantization (FP8 + packed E8M0 scales). Quant
clears padding rows explicitly. These op names must be given unique aliases
when different shape-specific definitions coexist in a manifest.

Dense numerics against original `model.linear`: tested M=1/5/65 at
N=1280,K=5120 and M=5,N=1024,K=4096, relative squared error <=2.1e-11.
O-A against its dequantized inputs: M=1/5/65/128 at fixed capacity 128,
relative squared error <=7.12e-4 from the MXFP8 output cast, and its FP8
bytes and UE8M0 exponents are identical both to quantizing the reference
BF16 O-A result and to the superseded BF16 kernel followed by
`dsv41_dense_quant_x`. Feeding that output straight into `dsv41_dense`
(N=5120, K=8192) agrees with the dequantized reference to <=1.9e-10.
This is not a certification of model-level accuracy. Weight scale packing
is byte-exact for one dense matrix and eight O groups.

`test_oa.py` quantizes its own activations rather than calling the
reference `act_quant`: that kernel races above 32 rows on GB300 and
sporadically writes E4M3 NaN bytes, which a GEMM then spreads over whole
rows. `check_reference` pins the harness quantizer to the reference at 32
rows, where it is stable.

`replay.py` fixtures exercise actual `program_io`, including a safetensors
load followed by a `once` program. mHC graph replay is byte-exact at
T=1/5/65/128; all eight Gate graph cases and four expert scale once cases
are also byte-exact. The separate `runtime_replay` Rust crate uses the actual
Runtime and peer-handle exchange for EP4 fixtures (no runtime modifications).

TMA descriptors are built once: all geometry uses fixed capacity, while
M controls active rows at launch. For symbolic `rows`, pass `max_tokens` to
dense/mHC/Gate pieces. Dense quant `sf_rows` is the fixed padded capacity.


The EP4 Runtime harness passes both `[1,1,1,1]` and `[1,5,0,3]`
local token counts for target 384 and draft 128 experts, including peer-handle
import, eager execution, capture, and replay. Outputs match the separately
validated AOT launch byte-for-byte. `test_checkpoint_moe.py` builds fixtures
for checkpoint-bound whole-layer MoE and uses the original Gate/Expert oracle.
The harness imports peers before invoking `once`, as required by Runtime.

`gate.pieces(raw_outputs=True)` declares both routing outputs as byte storage
so they can write directly to offsets in the exported MoE slab. The pointer
ABI and int64/float32 physical layouts remain unchanged. Both target and
draft variants pass actual graph replay with a single output slab. Routed
W2 uses public `i8`, shared W2 uses `fp8e4m3`, matching checkpoint bindings;
TMA interprets their existing bytes as FP4 and FP8 without a copy.

`dense.layout_pieces('query'|'output')` returns once operators that permute
FP8 weight bytes and pack E8M0 scales for fused attention. Query rows change
from `[head64,tile32,lane16]` to `[tile32,head64,lane16]`. Output projection
columns change from `[head8,tile16,lane32]` to `[tile16,head8,lane32]` within
each of eight groups. Use the resulting scale buffer in place of ordinary
dense scale packing; it already expands source scale rows and remaps blocks.

Actual checkpoint layer 0 passes the whole EP4 MoE chain (one independent
row per rank), including 291 once launches and graph replay. Against the
supplied Gate/Expert implementation, relative squared error is
`[0.000122506,0.000140288,0.000290526,0.000123945]`. This establishes the
loading/packing/communication integration on a real layer; the inputs are
synthetic activations, and full-model output agreement is still separate.

Both fused layout transforms pass actual safetensors `once` replay and four
real-weight GEMM graph cases. On checkpoint layer 0, query projection agrees
with original `model.linear` to relative squared error <5e-16 at T=1/5.
O-A, dequantized from its MXFP8 output, agrees with the original BF16
einsum to 1.40e-3/1.41e-3 at T=1/5, reflecting the FP8 activation cast and
the output cast. Weight and scale permutations themselves
are byte-exact. These fixed transforms apply to main-attention Wq_b and
WO_A, not the differently shaped indexer projection.

The complete checkpoint-bound Block 0 now passes 30-call EP4 prefill at
T=1 and T=5 per rank, including metadata, two mHC calls, window attention,
MoE, and final materialization, in both eager and graph execution. Oracle:
original `Block.forward`/`Attention.forward`/`MoE.forward`; the oracle follows
the official BF16 dequantization of WO_A. Across all eight rank cases, final
residual relative squared error is <=1.37e-5 and the shifted next-pre error
is <=9.29e-8. `test_checkpoint_block.py` creates these fixtures. The initial
illegal instruction was traced to the disabled compressed branch retaining
a non-null extra-length pointer; the attention provider now nulls it when
extra_topk is zero. This test uses real weights with synthetic embeddings;
it does not replace full-model generation or DSpark acceptance tests.

`test_compression_lowering.py` exercises source-2 publication through the real
Runtime: BF16 checkpoint projections with FP32 pooling, unrotated index-key
projection, normalization/RoPE/quantization, both cache formats, and commit.
For five input rows, the two completed groups match the original Compressor
exactly; both packed cache pages are byte-exact. Dequantized index keys agree
numerically, with only signed-zero representation differences.

`test_engram_lowering.py` uses one valid five-row request and one independent
padding request. `valid` means scheduler padding, not the original model's
image-span `token_mask`; padding hash scratch is unspecified, and residuals
must pass through unchanged. The single-GPU `runtime_replay` binary `probe`
loads complete host tables, runs once and forward programs, and dumps buffers
and state prefixes. `compare_probe.py` checks the oracle outputs and padding
contract. Host Engram tests release their mapped allocations when they exit.
The real host-bound 11-call eager/graph run passes: all five valid hashes
and embeddings are byte-exact; final residual relative squared error is
1.21e-10, and the entire independent padding request passes through byte-exact.

EP4 eager callers must enqueue ranks concurrently. A later cuBLAS call can
block on lazy initialization while an earlier MegaMoE waits for peer ranks;
issuing an entire multi-layer program on rank 0 before rank 1 can therefore
deadlock. The replay harness uses one worker per rank. Already-warm
captured graphs can use a separate sequential enqueue path.

`dspark_oracle.DraftOracle` constructs the original three DSpark blocks and
shares the original embedding/head, without constructing the target backbone.
Expert weights remain in their checkpoint FP4/FP8 formats and load lazily;
forward calls delegate to the supplied `Transformer.forward_spec`, `MoE`, and
`Expert`. `draft(anchor_ids, main_hidden, start_pos)` uses greedy sampling.
Seed once at position zero with the full prompt taps, then provide one real
AR tap row at every subsequent position to maintain the reference window.
Taps concatenate layers 37, 38, and 39 attention-input means. This measures
conditional draft agreement on the supplied target trajectory; it does not
measure an independent reference target model's generation distribution.

`runtime_replay` binary `teacher_verify` isolates target verification from draft
quality. Its config provides eight histories and six teacher IDs per history
on each of four ranks. It prefills each history, forks the actual Runtime lease
(including compressor state), then compares six sequential eight-row target
forwards against one 48-row verification with identical IDs. It bypasses draft
and acceptance calculation only in this diagnostic. The original context and
commit calls then commit a configured accepted prefix (default three), and a
plain target step replaces the rejected suffix. Full vocabulary logits are
saved for both forward and post-commit comparisons. `compare_teacher_verify.py`
reports top-five IDs, margins, winner-logit deltas, and full-vector errors;
`prepare_teacher_verify.py` derives eight histories from an AR trace config.

The eight-history autumn trace reproduces a c32 discrepancy only for prefix
52. Forked state and all boundaries through layer 19 are byte-exact. At layer
20, identical BF16 inputs produce one different BF16 compressor projection
entry under cuBLASLt M=8 versus M=48: element 296 is 0.046142578125 versus
0.0458984375. Normalization then crosses an FP4 quantization boundary at the
same channel; indexer scores and logical selections remain byte-exact.
Replacing only that single verification projection element with the plain
value restores every final logit in all 192 verification rows and all 32
post-commit rows exactly. This diagnostic establishes a shape-dependent
projection-rounding cause for this case; it does not imply arbitrary output
differences can be dismissed as rounding.

An experiment fixing just the ratio-1 compressor WKV GEMM's M to its allocated
capacity (128) in prefill, decode, and verification also makes all 224 rows
exact, without patching outputs. Both input and output workspaces cover those
128 rows; subsequent consumers still process only live rows. The fixed path
selects the former verification-side rounding, so the old plain result at
position 54 changes too. This is a consistent numerical path, not preservation
of the previous shape-specific bit pattern. Production lowering and serving
performance validation are handled separately.
