# GLM-5.3-Flash decode: MoE block, dense MLP, tail, TP comms — kernel-level spec

Scope: one sglang TP8 eager decode step (bs=1) on H100, GLM-5.3-Flash.
This document covers the dense MLP (layers 0-2), the MoE block (layers 3-44),
the vocab embedding, the model tail (hc_contract, final norm, lm_head, argmax),
and the TP all-reduce / all-gather path. KDA/mHC/norms are covered by the
sibling doc `kda_mhc.md`; DSA attention is out of scope.

## 0. Sources

Capture:
- `$GLM53_ARTIFACTS/recipe.json` — key `bs1`, 1351 launches. Layer L =
  sites B[L]..B[L+1)-1 (B from `hc_prenorm` sites). Dense layer 0 MLP = sites
  16..20, MoE layer 4 = sites 122..131, tail = 1345..1350, embedding = 0..1.
  Read via `tools/glm53/recipe.py`; `scalar()` decodes {"size","scalar"(hex)}.
- `$GLM53_ARTIFACTS/lifts/sm90_fp8_gemm_1d2d.json` — decoded tensormap
  ABI of the site-17 kernel instance (same entry symbol/grid/smem).

Code (under `<sglang site-packages>/`):
- `sglang/srt/models/glm5_next.py` → `glm5_next.py:N`; `sglang/srt/models/deepseek_v2.py` → `deepseek_v2.py:N`
- `sglang/kernels/ops/gemm/tiny_gemm.py` + `sglang/kernels/jit/csrc/gemm/tiny_gemm.cuh` → `tiny_gemm.py:N` / `tiny_gemm.cuh:N`
- `sglang/kernels/ops/moe/moe_fused_gate.py` → `moe_fused_gate.py:N`
- `sglang/kernels/ops/moe/moe_align_small_numel.py` → `align_small.py:N`
- `sglang/kernels/ops/moe/fused_moe_triton_kernels.py` → `fmoe_kernels.py:N`
- `sglang/srt/layers/moe/fused_moe_triton/layer.py` → `fmoe_layer.py:N`
- `sglang/srt/layers/moe/topk.py` → `topk.py:N`
- `sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py` → `triton_moe.py:N`
- `sglang/kernels/jit/csrc/gemm/per_token_group_quant.cuh` → `quant.cuh:N`
- `sglang/kernels/jit/csrc/deepseek_v4/silu_and_mul_masked_post_quant.cuh` → `silu_clamp.cuh:N`
- `sglang/kernels/jit/csrc/distributed/custom_all_reduce.cuh` → `allreduce.cuh:N`
- `sglang/kernels/jit/include/sgl_kernel/distributed/communicator.cuh` → `communicator.cuh:N`
- `sglang/kernels/ops/embeddings/vocab_parallel_embedding.py` → `embedding.py:N`
- `sglang/srt/distributed/device_communicators/triton_symm_mem_ag.py` → `symm_ag.py:N`
- `sglang/srt/layers/logits_processor.py` / `sglang/srt/layers/sampler.py` → `logits_processor.py:N` / `sampler.py:N`
- `deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh` → `fp8_1d2d.cuh:N`
- Checkpoint inventory `$GLM53_ARTIFACTS/inventory.json` (cited `inventory`),
  config `config.json` in the `$GLM53_CHECKPOINT` snapshot (cited `config.json`).

## 1. Constants for this path (config.json)

- Dense MLP (layers 0-2, `first_k_dense_replace=3`): intermediate 12288.
- MoE (layers 3-44): `n_routed_experts=288`, `num_experts_per_tok=8`,
  `n_shared_experts=1`, `moe_intermediate_size=2048`, `n_group=1`,
  `topk_group=1`, `norm_topk_prob=true`, `scoring_func="sigmoid"`,
  `routed_scaling_factor=2.5`, `swiglu_limit=10.0`.
- Quant: fp8 e4m3, block `[128,128]` weight scales, dynamic per-token-group-128
  activation scales (`quantization_config` in config.json).
- TP8 with NO expert parallelism: `DeepseekV2MoE` requires
  `tp_size <= n_routed_experts` (deepseek_v2.py:644-648); every rank holds ALL
  experts with the intermediate dim TP-sharded (2048/8 = 256 per rank). The
  fused-shared-expert weight remap is gated on `moe_ep_size == 1`
  (glm5_next.py:1360-1368). MoE runs in "moe_tp" mode; EP-style expert-count
  sharding (36/rank) is NOT what happens here.

## 2. Embedding (sites 0-1)

- Site 0: `_vocab_parallel_embedding_kernel` (Triton, embedding.py:9), grid
  [1,4,1], block [256,1,1]. Runtime params: `input_ptr`, `weight_ptr`,
  `out_ptr` (+2 zero i64 Triton scratch). Everything else is `tl.constexpr`
  (embedding.py:12-21, "one compile per layer"): `hidden_dim=4096`,
  `weight_stride0=4096`, shard window `[org_vocab_start, org_vocab_end)` =
  `[rank*19360, (rank+1)*19360)`, `added_vocab` window empty, `BLOCK_H=1024`
  (4096/1024 = 4 col blocks). Recipe: input = token ids (i64) in alloc
  0x7f0224600000; weight = `model.language_model.embed_tokens.weight` shard
  [19360,4096] BF16 (inventory; alloc 0x7f0de6600000, 159,383,552 B segment
  for 158,597,120 B); out = scratch 0x7eff99a03600 [1,4096] BF16. Tokens
  outside the local window produce zero rows (masked load, embedding.py:40-50).
- Site 1: `all_reduce_1shot_push_kernel` grid [132], block 128 — sums the 8
  partial embedding rows (§5). in 0x7eff99a03600 → out 0x7eff99a05600.
- The same kernel+config runs 91 times per forward: 1 embedding + 45 layers ×
  (attn post + ffn post) (recipe symbol count = 91).

## 3. Dense MLP, layers 0-2 (sites 16-20, layer 0)

`Glm5NextMLP` = `DeepseekV2MLP` (glm5_next.py:95, 751-760; deepseek_v2.py:255):
`MergedColumnParallelLinear(4096, [12288,12288])` → `swiglu_clamped(limit=10)`
→ `RowParallelLinear(12288, 4096, reduce_results=False)`; the all-reduce is
deferred to the communicator (site 21). fp8 block-quant GEMMs via DeepGEMM.

### 3.1 Weights (checkpoint → runtime TP8 shard)

| checkpoint tensor (inventory, `layers.0.mlp.*`) | shape | dtype | runtime per-rank shard |
|---|---|---|---|
| gate_proj.weight | [12288,4096] | F8_E4M3 | rows [r·1536,(r+1)·1536) merged into gate_up [3072,4096] |
| up_proj.weight | [12288,4096] | F8_E4M3 | rows appended after gate (merged order [gate 1536; up 1536]) |
| gate/up_proj.weight_scale_inv | [96,32] each | F32 | merged sfb [24,32] (N/128 × K/128) |
| down_proj.weight | [4096,12288] | F8_E4M3 | cols shard → [4096,1536] |
| down_proj.weight_scale_inv | [32,96] | F32 | sfb [32,12] |

N=3072 = 2×(12288/8) (merged gate+up TP shard); K=1536 = 12288/8 for down.

### 3.2 Op sequence and ABI (all pointers into the 2 MiB scratch arena 0x7eff99a00000)

1. Site 16 `per_token_group_quant_flat_kernel<QuantTrait<bf16,e4m3,128,ue8m0=F,rowmajor=F,aligned=T,fuse_silu=F>, PDL=T>`
   grid [1], block 256. 80 B `QuantKernelParams` (quant.cuh:195-203):
   in {0x0f600, tok_stride 4096} = layer_input [1,4096] BF16 → out {0x03600,
   4096} fp8 [1,4096]; scale {0x16c00, expert 0, token_stride 1, group_stride
   4, num_groups 32}; num_tokens 1, hidden 4096. Scale layout is COL-major
   (kRowMajor=false): f32 [32 k-groups × 4 (m padded)] row stride 16 B — the
   exact layout the DeepGEMM SFA tensormap wants (§3.3).
2. Site 17 `sm90_fp8_gemm_1d2d_impl<...BLOCK_M=16,BLOCK_N=128,BLOCK_K=128,...>`
   grid [132] (=kNumSMs), block 256 (128 TMA + 128 math), smem 101760, 248 regs
   (lift json). Signature (fp8_1d2d.cuh:50-55):
   `f(float* sfb, int* grouped_layout, uint M, uint N, uint K, tmap a, b, d, sfa)`.
   - p0 sfb = misc alloc 0x7f0e2de00000+163840: merged gate_up block scales
     [24,32] f32 (row-major n×k blocks; kernel indexes
     `sfb[n_block·shape_k_scales + k_block]`, fp8_1d2d.cuh:243-249, kMajorSFB=K).
   - p1 grouped_layout = nullptr (GemmType 0 = dense, not m-grouped).
   - p2/3/4 = M=1, N=3072, K=4096.
   - p5 tmap_a: u8 dims [4096,1] box [128,16] swizzle 128 → A fp8 at 0x03600.
   - p6 tmap_b: u8 dims [4096,3072] box [128,32] swizzle 128 → W at 0x7f0df1a00000.
   - p7 tmap_d: bf16 dims [3072,1] stride 6144 B box [32,16] swizzle 64 → D
     [1,3072] BF16 out at 0x11600.
   - p8 tmap_sfa: f32 dims [4,32] stride 16 B box [16,1] no swizzle → A scales
     at 0x16c00 (= site-16 scale out).
   - D-type is bf16 with EpilogueIdentity (symbol template) — no C accumulate.
3. Site 18 `silu_mul_clamp_kernel<bf16, PDL=T>` grid [1], block 192. 32 B
   `SiluAndMulClampParams` (silu_clamp.cuh:197-202): in 0x11600 (GEMM out),
   out 0x03600 (reuses the fp8-A slot), `swiglu_limit=10.0` (0x41200000),
   `out_vecs=192`, `blocks_per_row=1`. Vec = 8 bf16 → 192×8 = 1536 out elems;
   input row is 2·192 vecs = [gate 1536 | up 1536] (kernel reads gate at
   `row·2·out_vecs + vec`, up at `+ out_vecs`, silu_clamp.cuh:232-236).
   Math (silu_clamp.cuh:43-51, on bf16x2): `gate=min(gate,10)`,
   `up=clamp(up,-10,10)`, `out = silu(gate)·up` — matches the reference
   `swiglu_clamped` (glm5_next.py:144-149). GLM-5.3 has no swiglu alpha.
4. Site 19 `per_token_group_quant_flat_kernel` (same template as site 16):
   in 0x03600 [1,1536] BF16 (tok_stride 1536) → out 0x15e00 fp8 [1,1536];
   scale 0x16c00 col-major [12,4] (num_groups=12, group_stride 4); hidden 1536.
5. Site 20 `sm90_fp8_gemm_1d2d_impl` (same template): sfb = misc+166912
   (down block scales [32,12]; immediately after the 24×32 gate_up scales:
   163840+3072 = 166912), grouped_layout=null, M=1, N=4096, K=1536; tmap_a →
   0x15e00 (dims [1536,1]), tmap_b → down weight at 0x7f0df2600000, tmap_d →
   [1,4096] BF16 out at 0x12e00, tmap_sfa → 0x16c00. smem 101680.
6. Site 21 `all_reduce_1shot_push_kernel` — in 0x12e00 → out, sums the 8
   partial down-proj outputs.

`_act_quant_kernel` is NOT on this path: its only sites (74,195,...) are inside
DSA attention blocks (indexer quant). Dense MLP and MoE both use sglang's
`per_token_group_quant_flat_kernel`.

## 4. MoE, layers 3-44 (sites 122-131, layer 4; identical in every MoE layer)

`self.mlp = Glm5NextMoE` = `DeepseekV2MoE` (glm5_next.py:96, 738-744;
deepseek_v2.py:563). `num_fused_shared_experts=1` → expert space is 289
(288 routed + shared at index 288), topk = 9 (8 routed + 1 shared)
(deepseek_v2.py:596-615). The shared expert has NO separate GEMM launches.

### 4.1 Weights (per rank, all TP-sharded, all 289 experts resident)

| runtime tensor | shape | dtype | recipe evidence |
|---|---|---|---|
| w13_weight | [289, 512, 4096] | F8_E4M3 | site 127 b_ptr alloc 0x7f0b58000000, 606,076,928 B = 289×2,097,152; stride_be 2097152 |
| w13_scale_inv | [289, 4, 32] | F32 | site 127 b_scale_ptr, strides (128, 32, 1) |
| w2_weight | [289, 4096, 256] | F8_E4M3 | site 130 b_ptr alloc 0x7f0b44000000, 304,087,040 B = 290 slots (1 slot slack); stride_be 1048576 |
| w2_scale_inv | [289, 32, 2] | F32 | site 130 b_scale_ptr, strides (64, 2, 1) |
| gate.weight (router) | [288, 4096] | BF16 | inventory `layers.N.mlp.gate.weight`; site 122 w ptr |
| gate.e_score_correction_bias | [288] | F32 | inventory; site 123 bias ptr |

Checkpoint keeps per-expert tensors `layers.N.mlp.experts.E.{gate,up,down}_proj`
([2048,4096]/[2048,4096]/[4096,2048] F8_E4M3 + `weight_scale_inv`
[16,32]/[16,32]/[32,16] F32) plus `shared_experts.*` (same shapes). Load
mapping: `make_expert_params_mapping(gate,down,up, num_experts=289)`
(glm5_next.py:1535-1540; fmoe_layer.py:1650-1675); shared_experts → expert id
288 (deepseek_v2.py:588-592 comment); w13 stacking is [w1=gate rows 0..255 ;
w3=up rows 256..511] (`idx = 0 if shard_id == "w1" else 1`, fmoe_layer.py:661),
rows TP-narrowed (fmoe_layer.py:764-787). Gate/up are NOT interleaved
(gate_up_interleaved=False; see §4.4 silu addressing).

### 4.2 Site 122 — router GEMM `tiny_n_gemm_kernel<GEMMTraitN<288,4096,3,32>, M=1, f32, PDL>`

grid [96], block 256. 32 B `TinyGEMMParams` = {out, x, w, stride_x}
(tiny_gemm.cuh:64-69): out = 0x15e00 (scores [1,288] F32), x = 0x17000
(layer_input [1,4096] BF16, the mHC ffn-pre output), w = 0x7f0bb8f00000
(gate.weight), stride_x = 4096. N_SPLIT=3 → 288/3 = 96 blocks; block = K/16 =
256 threads (tiny_gemm.py:30-34, 96-106). Output dtype F32
(`tiny_gemm_bf16(..., out_dtype=torch.float32)`, deepseek_v2.py:539-543) —
router logits are f32, not bf16.

### 4.3 Site 123 — `_router_triton_kernel` (top-8 + shared, sigmoid scoring)

grid [1], block 32 (1 warp; BLOCK_N=512=next_pow2(288), BLOCK_M=1,
moe_fused_gate.py:466-479). Runtime args (moe_fused_gate.py:96-111; constexprs
and M=1/stride-1 specialized out by Triton): scores 0x15e00, bias =
e_score_correction_bias (0x7f0e2de00000+2057728), out_weights [1,9] F32 at
0x16c00, out_indices [1,9] i32 at 0x16400, packed-dummy at 0x15a00;
scalars: `routed_scaling_factor=2.5` (0x40200000), `moe_softcapping=0`,
stride_bias_alt 0, stride_input_ids 0, stride_sm 288, stride_wm 9, stride_im 9,
stride_pm/pk 0. Constexpr: K=9, K_ROUTED=8, N_GROUP=1, SCORING_FUNC=0
(sigmoid), RENORMALIZE=T, APPLY_SCALE=F (Triton fp8 runner does not fuse the
scale in topk — fmoe_layer.py:104-119, 560-562).
Math: `activated = sigmoid(logits)`; ranking on `activated + bias`, bias-free
weight (moe_fused_gate.py:196-199); 8 sequential argmax rounds; shared slot:
weight = routed_sum/2.5, id = 288 (moe_fused_gate.py:282-292); then
renormalize all 9 by routed_sum (moe_fused_gate.py:294-299) → routed weights
sum to 1, shared weight = 0.4 exactly.

### 4.4 Sites 124-125 — topk_ids rebuild + align

- Site 124: ATen `CatArrayBatchedCopy` grid [1,2], block 128 — `torch.cat(
  [routed_cols [1,8], shared_cols [1,1]], dim=-1)` (topk.py:2271) after
  `_biased_grouped_topk_postprocess` on the routed columns. Inputs are the two
  column views of the router's out_indices (0x16400, 0x16420 = +8 i32); output
  = fresh [1,9] i32 at 0x16600.
- Site 125: `_moe_align_small_numel_kernel` grid [1], block 128
  (align_small.py:23, host :114-148). Params: topk_ids 0x16600,
  sorted_token_ids 0x03600, expert_ids 0x16400 (reuses router indices slot),
  num_tokens_post_pad 0x04000; scalars: `num_experts=290` (E+1 bucket
  convention, E=289), `block_size=64`, `numel=9`. Single-CTA pairwise-rank
  sort; pads each expert bucket to 64; outputs sorted_token_ids [576] i32,
  expert_ids [9] i32, num_tokens_post_pad.

### 4.5 Sites 126-127 — quant + w13 fused_moe

- Site 126: `per_token_group_quant_flat_kernel<...rowmajor=T...>`: in 0x17000
  [1,4096] BF16 → out 0x04200 fp8 [1,4096]; scale 0x05200 ROW-major
  [1,32] (token_stride 32, group_stride 1). NOTE: A is quantized ONCE per
  token; fused_moe gathers rows through sorted_token_ids — there is no
  [9,4096] gathered activation buffer.
- Site 127: `fused_moe_kernel` grid [36], block 128, smem 73728
  (fmoe_kernels.py:325). Non-constexpr params: a_ptr 0x04200, b_ptr w13,
  c_ptr 0x0f600, a_scale 0x05200, b_scale, topk_weights 0x16c00,
  sorted_token_ids 0x03600, expert_ids 0x16400, num_tokens_post_padded
  0x04000; scalars N=512, K=4096, EM=576, num_valid=9; strides am=4096,
  be=2097152, bk=4096, cm=512, asm=32, bse=128, bsk=32 (bn=cn=ask=bsn=1
  specialized out; bias strides 0,0). Derived constexprs: BLOCK_M=64,
  BLOCK_N=128 (grid = cdiv(576,64)·cdiv(512,128) = 9·4 = 36), BLOCK_K=128
  with 3 stages (smem 73728 = 3·(64·128+128·128) fp8). MUL_ROUTED_WEIGHT=F
  on this GEMM. Output c1 [9,512] BF16 at 0x0f600.

### 4.6 Sites 128-129 — swiglu clamp + requant

- Site 128: `silu_mul_clamp_kernel<bf16>` grid [9], block 32: in 0x0f600
  [9,512] BF16 → out 0x04200 [9,256] BF16, limit=10.0, out_vecs=32
  (32·8=256), blocks_per_row=1. Same clamp math as §3.2; row layout
  [gate 256 | up 256] confirms non-interleaved w13.
- Site 129: `per_token_group_quant_flat_kernel<rowmajor=T>`: in 0x04200
  [9,256] BF16 (tok_stride 256) → out 0x11a00 fp8 [9,256]; scale 0x05400
  [9,2] (token_stride 2, num_groups 2); num_tokens=9, hidden=256.

### 4.7 Sites 130-132 — w2 fused_moe + sum-reduce + all-reduce

- Site 130: `fused_moe_kernel` grid [288], block 128, smem 73728: a_ptr
  0x11a00, b_ptr w2, c_ptr 0x27000, a_scale 0x05400, topk_weights 0x16c00,
  align buffers as site 127; scalars N=4096, K=256, EM=576, num_valid=9;
  strides am=256, be=1048576, bk=256, cm=4096, asm=2, bse=64, bsk=2.
  Grid = 9·cdiv(4096,128) = 288. MUL_ROUTED_WEIGHT=T (topk weights applied
  here). Output c2 [9,4096] BF16 at 0x27000.
- Site 131: `_moe_sum_reduce_kernel` grid [1,2], block 512
  (fmoe_kernels.py:1189, host :1245-1276: BLOCK_M=1, BLOCK_DIM=2048,
  num_warps=16 → grid (1, 4096/2048)). Params: in 0x27000, in strides
  (36864 = 9·4096, 4096, 1), out 0x17000 [1,4096] BF16 (out_stride 4096),
  token_num=1, topk_num=9, hidden=4096; `routed_scaling_factor=2.5` is a
  CONSTEXPR baked in (no scalar in recipe). Math: F32 accumulate over the 9
  expert rows, ×2.5, store bf16 (fmoe_kernels.py:1225-1241). Net effect:
  routed experts contribute 2.5·p_i (p_i = renormalized sigmoid), the shared
  expert contributes 0.4·2.5 = 1.0 — exactly the config semantics.
  The output lands at 0x17000 — the SAME buffer that held layer_input: the
  MoE output overwrites the ffn input in place.
- Site 132: `all_reduce_1shot_push_kernel`, in 0x17000 → partial sum over
  ranks; then the mHC ffn post (site 133, see kda_mhc.md §6.4).

## 5. `all_reduce_1shot_push_kernel` — 112 B param struct (91 sites)

`AllReducePushParams<8>` (allreduce.cuh:47-53): `{const void* input; void*
output; uint32 num_vecs; uint32 rank; PushWorkSpace<8> ws}`.
`PushWorkSpace<8>` (communicator.cuh:81-86): `{uint8_t* workspaces[8];
Counter* counter; uint8_t* mc_workspace; uint32 slot_bytes}` (88 B with pad).
Decoded from the site-1 blob (all 91 sites share every field except
input/output):

| off | field | value (site 1) |
|---|---|---|
| 0 | input | 0x7eff99a03600 (varies per site: the sublayer output) |
| 8 | output | 0x7eff99a05600 |
| 16 | num_vecs | 512 = 4096 elems / 8 (AlignedVector<bf16,8> = 16 B) |
| 20 | rank | 0 (capture was rank 0) |
| 24..87 | ws.workspaces[0..7] | 8 IPC peer push buffers, 0x7f0dd9400000 + i·0x1400000 (20 MiB stride, i=0..7) |
| 88 | ws.counter | 0x7f0e2de00800 (local Counter in the misc alloc) |
| 96 | ws.mc_workspace | 0x7f0de3400000 (multicast VA of the local workspace) |
| 104 | ws.slot_bytes | 131072 (0x20000) |

Kernel: grid [132], block 128, `LoadStoreImpl<AlignedVector<bf16,8>, 8, ...>`
1-shot push: each rank pushes its vec8-chunked input to every peer's
workspace, then polls and reduces locally (allreduce.cuh:1-12, 137). Inputs
and outputs are always [1,4096] BF16 on this path.

`_all_gather_kernel_inner` (site 1348, symm_ag.py:250): grid [4], block 1024.
Runtime params: input_ptr = lm_head out 0x7eff99a05600 [1,19360] BF16,
multicast_ptr = 0x31c4000000 (38 MB symmetric segment), signal_pad_ptr
(scratch+89088), hidden_offset = 0 (rank·19360; rank 0); `total_tokens=1` is
Triton-specialized out. Each block writes 128-bit chunks of the local shard
into the multicast buffer via `multimem.st` (symm_ag.py:271-290);
`_MIN_BLOCKS=4`, `_BLOCK_THREADS=1024` (symm_ag.py:354-356, 402-416).
`safe=True` → output `.clone()` of the [1,154880] gathered row.

## 6. Tail (sites 1345-1350)

1. Site 1345 `at::reduce_kernel<...MeanOps<BFloat16,float,float>...>` grid
   [8], block [32,4]: `hc_contract` = unweighted mean of the 4 mHC streams
   (mhc.py:1885-1886): in [1,4,4096] BF16 at 0x17000 → out [1,4096] BF16 at
   0x0f600, F32 accumulation. 8 CTAs × 128 threads × 4 elems = 4096.
2. Site 1346 flashinfer cutlass `RMSNormKernel` grid [1], block 128, smem
   16400: the final `model.norm` (glm5_next.py:1209). Params: {in 0x0f600,
   i64 1}, weight = `model.language_model.norm.weight` [4096] BF16
   (0x7f02f4200000+2076160), {out 0x03600, i64 1}, i64 n=1, eps f32
   0x3727C5AC = 1e-5.
3. Site 1347 `nvjet_sm90_tst_256x8_64x6_4x1_v_bz_TNT` grid [4,19], block 384,
   smem 219364: lm_head GEMM [1,4096] BF16 @ [4096,19360] → [1,19360] BF16
   (cublasLt; per-rank vocab shard 154880/8 = 19360 of
   `lm_head.weight` [154880,4096] BF16, inventory). No split-K reduce. Output
   at 0x05600.
4. Site 1348 `_all_gather_kernel_inner` (§5): [1,19360] → [1,154880] BF16
   via multicast + clone.
5. Site 1349 `at::unrolled_elementwise_kernel direct_copy` grid [303], block
   128: `logits_buffer.copy_(logits)` — cast BF16→F32 into the persistent
   f32 logits buffer (logits_processor.py:1123-1133; buffer dtype asserted
   torch.float). numel = 154880 scalar; in at 0x17000. 303·128·4 = 155,136.
6. Site 1350 `at::reduce_kernel<512,1,ReduceOp<float,ArgMaxOps<float>,uint,int,4,4>>`
   grid [1,19], block 512, smem 8192: `torch.argmax(logits, -1)`
   (sampler.py:192). Input f32 [1,154880]; 19 column blocks cooperate through
   global scratch; smem 8192 = 512 threads × 16 B (f32 value + int index);
   output = token index [1] i64.

## 7. Manifest-generator notes (buffer roles, dtypes)

Scratch arena (2 MiB, base 0x7eff99a00000) offsets for layer-4 MoE / layer-0
MLP; R = read, W = written:

| offset | size | role |
|---|---|---|
| 0x03600 | 8192 B | embed out / all-reduce in; dense a1 fp8 W then R; silu out W; sorted_token_ids [576] i32 W; RMSNorm out |
| 0x04000 | 4 B | num_tokens_post_pad W |
| 0x04200 | 4096 B | MoE a1 fp8 W→R; silu-MoE out [9,256] bf16 W→R |
| 0x05200 | 128 B | a1 scales [1,32] f32 row-major |
| 0x05400 | 72 B | a2 scales [9,2] f32 |
| 0x05600 | 8192 B | embed all-reduce out; lm_head out [1,19360] bf16 |
| 0x0f600 | 8192 B | dense layer_input R; MoE c1 [9,512] bf16 W→R; hc_contract out |
| 0x11600 | 6144 B | dense GEMM1 D [1,3072] bf16 W→R |
| 0x11a00 | 2304 B | MoE a2 fp8 [9,256] W→R |
| 0x12e00 | 8192 B | dense GEMM2 D [1,4096] bf16 W→R (all-reduce site 21 in) |
| 0x15a00 | 4 B | router packed-dummy (never read) |
| 0x15e00 | 1152 B | router scores [1,288] f32; dense a2 fp8 [1,1536] (different layers) |
| 0x16400 | 36 B | router out_indices [1,9] i32 W→R, then align expert_ids [9] W |
| 0x16600 | 36 B | cat out topk_ids [1,9] i32 W→R |
| 0x16c00 | 512 B | dense SFA [32,4]/[12,4] f32 col-major; MoE topk_weights [1,9] f32 (different layers) |
| 0x17000 | 8192 B | layer_input (mHC ffn-pre out) R by tiny_gemm + a1 quant; then MoE out [1,4096] bf16 W in place; all-reduce site 132 in |
| 0x27000 | 36864 B | MoE c2 [9,4096] bf16 W→R |

Rules: all GEMM D outputs are BF16 (DeepGEMM template `bfloat16_t +
EpilogueIdentity`; nvjet bf16 out). Router scores and topk_weights are F32.
fused_moe A/C are fp8/BF16. Logits are BF16 until the site-1349 cast; argmax
consumes F32. Every all-reduce moves exactly [1,4096] BF16. Weights are never
written. `expert_ids`/`sorted_token_ids`/`num_tokens_post_pad` are i32.
The MoE output buffer (0x17000) aliases the ffn input — a manifest must order
the a1-quant/tiny_gemm reads before the site-131 store.

## 8. Surprises vs the kda_mhc.md table

1. `routed_scaling_factor` is 2.5, not 2.4 (recipe site 123 scalar
   0x40200000; config.json). It is applied ONCE, as a constexpr in
   `_moe_sum_reduce_kernel`, not in the router and not per-expert.
2. The shared expert is FUSED into the MoE as expert 288 with topk 9
   (8 routed + 1 shared): no separate shared-expert GEMM/quant launches
   exist. Its router weight is routed_sum/2.5 → renormalized to exactly 0.4;
   the ×2.5 at sum-reduce restores its effective weight to 1.0.
3. This is TP-mode MoE, not EP: every rank holds all 289 experts with the
   intermediate dim sharded to 256; there is no 36-experts/rank split and no
   expert-parallel dispatch/combine.
4. MoE activations are quantized ONCE per token ([1,4096]); fused_moe gathers
   rows via sorted_token_ids (EM=576 = 9×64 padding, grid 36 = 9×4 tiles).
5. Router logits are F32 (`tiny_gemm_bf16(out_dtype=f32)`), and the router
   topk runs with N_GROUP=1 (grouped routing compiled out).
6. `_act_quant_kernel` belongs to the DSA indexer (sites 74,195,...), not to
   MoE/MLP; the whole MLP/MoE path uses `per_token_group_quant_flat_kernel`
   (col-major SFA for DeepGEMM, row-major for Triton MoE).
7. The site-124 `CatArrayBatchedCopy` is not a MoE kernel: it rebuilds
   topk_ids [1,9] from the router output after the EPLB-safe routed-column
   postprocess (topk.py:2271).
8. Argmax input is F32 [1,154880]: the gathered BF16 logits are first copied
   into the persistent f32 buffer (site 1349), then `ArgMaxOps<float>`
   reduces over 19 blocks with 8 KiB smem.
9. The MoE output overwrites the layer_input buffer in place (0x17000) right
   before the all-reduce — the only destructive aliasing on this path.
10. all-reduce param blob: rank is baked in as a scalar (0 in this capture),
    slot_bytes = 128 KiB, and the 8 IPC workspaces sit at a fixed 20 MiB VA
    stride with a separate multicast VA — the struct is shared verbatim by
    all 91 sites; only input/output pointers change.
