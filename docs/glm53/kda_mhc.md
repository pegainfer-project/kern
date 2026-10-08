# GLM-5.3-Flash decode: KDA + mHC + norms — kernel-level spec

Scope: one sglang TP8 eager decode step (bs=1) on H100, GLM-5.3-Flash.
This document covers the KDA (Kimi Delta Attention) layers, the mHC
(multi-stream Hyper-Connections) machinery, and every norm on that path.
DSA layers (3,7,11,...,43) and the MTP layer (45) are out of scope except
where they share the mHC path.

## 0. Sources

Capture:
- `$GLM53_ARTIFACTS/recipe.json` — per-launch symbol/grid/block/smem/params, key `bs1`, 1351 launches.
- `$GLM53_ARTIFACTS/decode_fwd25_seq.txt` — same forward, human-readable.
- Recipe block map: layer 0 = sites 4..22 (KDA + dense MLP), layer 3 = sites 61..109 (DSA + MoE), layer 4 = sites 110..133 (KDA + MoE), final contract = sites 1344..1350.

Code (all under `<sglang site-packages>/`):
- `sglang/srt/models/glm5_next.py` (model), cited as `glm5_next.py:N`
- `sglang/srt/configs/glm5_next.py` (config), cited as `cfg_glm5_next.py:N`
- `sglang/srt/configs/mamba_utils.py` (state shapes), cited as `mamba_utils.py:N`
- `sglang/srt/layers/attention/linear/kda_backend.py` → `kda_backend.py:N`
- `sglang/srt/layers/attention/linear/kernels/kda_triton.py` → `kda_triton.py:N`
- `sglang/kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py` → `fused_recurrent.py:N`
- `sglang/kernels/ops/mamba/causal_conv1d_triton.py` → `conv1d.py:N`
- `sglang/kernels/ops/attention/fla/fused_norm_gate.py` → `norm_gate.py:N`
- `sglang/kernels/ops/layernorm/mhc.py` → `mhc.py:N`
- `sglang/srt/layers/communicator_mhc.py` → `communicator_mhc.py:N`
- `sglang/srt/mem_cache/memory_pool.py` → `memory_pool.py:N`
- `sglang/srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py` → `fused_mhc.py:N`
- `deep_gemm/include/deep_gemm/impls/sm90_tf32_hc_prenorm_gemm.cuh` → `prenorm.cuh:N`
- Checkpoint tensor list: `$GLM53_ARTIFACTS/inventory.json`; model config: `config.json` in the `$GLM53_CHECKPOINT` snapshot.
- Serve command: our sglang launch script (`--tp 8 --mem-fraction-static 0.80`, no spec decode).

## 1. Model constants (config.json + cfg_glm5_next.py)

- 45 layers (0..44); KDA at the 34 layers with `layer_idx % 4 != 3` (cfg_glm5_next.py:199-226 builds `kda_layers` from `layer_types`). DSA at {3,7,11,15,19,23,27,31,35,39,43}.
- KDA: `num_heads=64`, `head_dim=128`, `short_conv_kernel_size=4`, `gate_lower_bound=-5.0` (config.json `linear_attn_config`).
- mHC: `mhc=True`, `hc_mult=4`, `hc_eps=1e-6`, `hc_sinkhorn_iters=20` (config.json). Post multiplier = 2.0 (`_MHC_POST_MULT_VALUE`, glm5_next.py:135).
- `rms_norm_eps=1e-5`, `swiglu_limit=10.0`, hidden 4096, vocab 154880.
- MLP: layers 0-2 dense (`first_k_dense_replace=3`, intermediate 12288), layers 3-44 MoE (288 routed + 1 shared expert, moe_intermediate 2048).
- TP8: attn heads are sharded 8 ways → 8 local KDA heads/rank; local q/k/v width = 1024; full projection = 8192.
- MTP layer 45: DSA indexer weights, `enorm`/`hnorm`, NO `hc_*` tensors in the checkpoint → no mHC (out of scope).

## 2. KDA weights: checkpoint vs runtime

Checkpoint (`model.language_model.layers.N.self_attn.*`, layers 0/4/44 verified identical in glm53_inventory.json):

| name | shape | dtype |
|---|---|---|
| q_proj / k_proj / v_proj.weight | [8192, 4096] each | BF16 |
| b_proj.weight | [64, 4096] | BF16 |
| f_a_proj.weight / g_a_proj.weight | [128, 4096] each | BF16 |
| f_b_proj.weight / g_b_proj.weight | [8192, 128] each | BF16 |
| q_conv1d / k_conv1d / v_conv1d.weight | [8192, 1, 4] each | BF16 |
| dt_bias | [8192] | F32 |
| A_log | [64] | F32 |
| o_norm.weight | [128] | BF16 |
| o_proj.weight | [4096, 8192] | BF16 |
| layers.N.hc_attn_fn / hc_ffn_fn | [24, 16384] | BF16 (loaded into F32 params) |
| layers.N.hc_attn_base / hc_ffn_base | [24] | F32 |
| layers.N.hc_attn_scale / hc_ffn_scale | [3] | F32 |
| layers.N.input_layernorm / post_attention_layernorm.weight | [4096] | BF16 |

Runtime fusions per rank (load mapping glm5_next.py:1516-1532):

1. `fused_qkvbfg_a_proj` (glm5_next.py:426, MergedColumnParallelRepeatedLinear): one weight [3336, 4096] BF16 = [q(1024) | k(1024) | v(1024) | b(8) | f_a(128) | g_a(128)]. q/k/v/b are TP-sharded; f_a/g_a are replicated (`_PACKED_MODULES_MAPPING`, glm5_next.py:329-341). Answer to "qkvz?": there is NO z tensor in KDA; the fused row is qkvb+f_a+g_a, 3336 wide. The output-norm gate is NOT in this GEMM (see §3.2).
2. `fused_fg_b_proj` (glm5_next.py:440, ColumnParallelBatchedLinear): batched-2 weight [2, 1024, 128] BF16 = [f_b ; g_b], output-TP-sharded.
3. `qkv_conv1d` (glm5_next.py:535-546, MergedColumnParallelLinear, `params_dtype=torch.float32`): ONE merged depthwise conv weight [3072, 1, 4] F32, squeezed to [3072, 4] at glm5_next.py:570. The checkpoint keeps three separate conv1d weights; sglang merges them at load. No bias (`bias=False`; checkpoint has no conv bias).
4. `dt_bias` [1024] F32 (glm5_next.py:526), `A_log` [1,1,8,1] F32 (glm5_next.py:548).
5. `o_norm` = FusedRMSNormGated(128, eps=1e-5, activation="sigmoid") (glm5_next.py:556-559).
6. `o_proj` RowParallelLinear [1024→4096], `reduce_results=False` (glm5_next.py:561-568) — the TP all-reduce is deferred to the layer communicator.
7. mHC params live on the decoder layer with checkpoint-verbatim names (glm5_next.py:773-787): `hc_attn_base` [24] F32, `hc_attn_scale` [3] F32, `hc_attn_fn` [24,16384] F32, same for `hc_ffn_*`. `mix_hc = (2+4)*4 = 24`, `hc_dim = 4*4096 = 16384` (glm5_next.py:770-772).

## 3. KDA decode, op by op (recipe sites 6-12 of layer 0; identical in every KDA layer)

Model forward (glm5_next.py:628-661): `fused_qkvbfg_a_proj` → split → `fused_fg_b_proj` → `RadixLinearAttention` (conv + recurrence, kda_backend.py:546 `forward_decode`) → `o_norm` → `o_proj`.

### 3.1 Site 6+7 — fused qkvbfg_a GEMM

- Kernel: `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT`, grid [4,21,1], block 384, smem 225636; then `cublasLt splitKreduce_kernel<32,16>`, grid [105,1,1].
- Op: `layer_input [1,4096] BF16 @ W^T [4096,3336] → [1,3336] BF16` (cublasLt with split-K; 105 = ceil(3360/32) reduce blocks, N padded to 3360).
- Output split (glm5_next.py:435-440, 615): qkv [1,3072] (view, row pitch 3336), beta [1,8] (row pitch 3336), fg_a [1,256].

### 3.2 Site 8 — fused fg_b batched GEMM

- Kernel: `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT`, grid [4,8,1], block 384, smem 164308.
- Op (glm5_next.py:617-620): fg_a [1,256] viewed [1,2,128] → transposed [2,1,128]; batched-2 GEMM `[2,1,128] @ W[2,128,1024] → [2,1,1024]`.
- Outputs: `forget_gate` (= `a`) [1,1024] contiguous (pitch 1024), `g_proj_states` [1,1024]. Both are low-rank up-projections (bottleneck 128) of the same hidden input. `g_proj_states` becomes the o_norm gate (§3.5); it is NOT the recurrence gate.

### 3.3 Site 9 — causal conv1d update

- Kernel: `_causal_conv1d_update_kernel`, grid [1,12,1], block 128 (conv1d.py:586; grid = (batch, ceil(dim/256)), BLOCK_N=256, conv1d.py:1101-1105, 1201). Scalar 1 = batch.
- Call (kda_backend.py:730-737): `causal_conv1d_update(mixed_qkv [1,3072] BF16, conv_states.transpose(-1,-2) [581,3072,3] BF16, conv_weights [3072,4] F32, bias=None, activation="silu", conv_state_indices=cache_indices)`.
- Math per channel d (conv1d.py:679-695, 716-764, 845-890): read the 3 cached tokens (s0,s1,s2); shift state left by 1 and append the new token x → new state (s1,s2,x); output `silu(s0*w0 + s1*w1 + s2*w2 + x*w3)` (no bias, F32 accumulate).
- Output: fresh contiguous tensor `qkv` [1,3072] BF16 (conv1d.py:1079, 1208-1210). State written in place at the sequence's slot.

### 3.4 Site 10 — fused sigmoid-gating delta-rule update

- Kernel: `fused_sigmoid_gating_delta_rule_update_kernel` (Triton, fused_recurrent.py:38), grid [4,1,8], block 32, smem 64.
- Dispatch: packed fast path is SKIPPED because `layer.lower_bound = -5.0` is not None (kda_backend.py:740-744); the model falls to the split path (kda_backend.py:771-787) → `TritonKDAKernel.decode` (kda_triton.py:125-157).
- Inputs (fused_recurrent.py:376-558):
  - q,k,v: [1,1,8,128] BF16 views of the conv output (split + unflatten + unsqueeze, kda_backend.py:771-774). At bs=1 the size-1 T dim normalizes the stride, so `stride_q/k/v = 1024`.
  - a (forget-gate logits) [1,1024] BF16, `stride_a = 1024`.
  - b (beta logits) [1,8] BF16 view into the fused GEMM row, `stride_b = 3336`.
  - A_log [8] F32, dt_bias [1024] F32, h0_source = SSM pool view [581,8,128,128] F32, h0_indices = cache_indices int32, cu_seqlens = query_start_loc (IS_VARLEN=True even at decode).
  - o: fresh [1,1,1,8,128] BF16.
- Grid (fused_recurrent.py:454-455): CUDA uses SPLIT_N_HV_GRID → (NV, N, HV) = (V/BV, num_seqs, 8). BV=32, 1 warp (fused_recurrent.py:14-35), BK=128. Grid formula: (4, S, 8).
- Scalar args, decoded (recipe values in parentheses):
  - `softplus_beta` = 1.0 (1065353216), `softplus_threshold` = 20.0 (1101004800) — kda_triton.py:153-154. Dead args on this path (only used when `lower_bound is None`).
  - `lower_bound` = -5.0 (3231711232 = 0xC0A00000) — the GLM-5.3 safe-gate bound, from `gate_lower_bound` (cfg_glm5_next.py:214-217; glm5_next.py:587).
  - `stride_h0_source` = 131072 — per-slot pitch of the per-layer SSM view in ELEMENTS (8*128*128).
  - `cache_steps` = 0 (no spec verify).
  - `scale` = 0.08838835 (1035273459 = 0x3DB504F3) = 128^-0.5 (fused_recurrent.py:438; kda_backend.py passes `head_k_dim**-0.5`).
  - `T` = 1, `stride_a/q/k/v` = 1024, `stride_b` = 3336 (0x0D08).
- Recurrence math: see §5.
- PDL (programmatic dependent launch) chains this kernel behind the conv update (fused_recurrent.py:486-490, 553).

### 3.5 Site 11 — gated output RMSNorm (`layer_norm_gated_fwd_kernel`)

- Kernel: `layer_norm_gated_fwd_kernel` (norm_gate.py:27), grid [1,1,1], block 128 (num_warps 4), smem 256. D=128 ≤ 512 → BT=32 rows/CTA (norm_gate.py:230-249).
- Call (glm5_next.py:657-659): `o_norm(core_attn_out, norm_gate)`; x = recurrence output [1,1,8,128] → [8,128]; g = `g_proj_states.unflatten(-1,(-1,128))` → [8,128] (glm5_next.py:657); w = o_norm.weight [128] BF16.
- Scalars: eps = 1e-5 (925353388 = 0x3727C5AC), T = 8 rows.
- Math per row (norm_gate.py:91-110, IS_RMS_NORM=True, ACTIVATION="sigmoid"): `rstd = 1/sqrt(mean(x^2)+1e-5)`; `y = (x*rstd) * w * sigmoid(g)`. Norm BEFORE gate; plain RMSNorm (no mean subtraction, no (1+w)).
- Output [1,1,8,128] → flattened [1,1024] (glm5_next.py:659).

### 3.6 Site 12 — o_proj

- Kernel: `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT`, grid [4,16,1].
- Op: `[1,1024] BF16 @ W^T [1024,4096] → [1,4096]` partial (reduce_results=False).
- Site 13: `sglang all_reduce_1shot_push_kernel` grid [132,1,1] — the communicator's TP all-reduce of the attention output (communicator_mhc.py:266 `attention_tensor_model_parallel_all_reduce`, via `MHCCommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual`).

## 4. KDA state layout (per sequence, per layer)

Two pools, both from `MambaPool.__init__` (memory_pool.py:616-648, non-envelope branch; envelope is off — the recipe's `stride_h0_source = 131072` equals HV*V*K, which only holds for per-layer-contiguous blocks).

Shapes (mamba_utils.py:347-354, `KimiLinearStateShape.create` with tp_world_size=8):
- conv: ONE tensor [(k-1), q_dim + 2*k_dim] = [3, 3072] per layer-slot, BF16.
- temporal (SSM/delta state): (8, 128, 128) F32 per layer-slot = 8 heads × [V=128 × K=128], K contiguous.

Pool tensors (per GPU rank):
- `temporal_state = torch.zeros((34, 581, 8, 128, 128), f32)` (memory_pool.py:643-648): 10,356,785,152 B. The recipe's 10,357,833,728 B arena is this tensor's allocator segment (2 MiB roundup; 34×582 slots would need 10,374,610,944 B — larger than the arena, and 581 slots rounds up to exactly 10,357,833,728 B; so pool size = 580, +1 guard slot = 581 rows/layer).
  - Per-layer byte stride: 581 × 131072 × 4 = 304,611,328 B.
  - Per-slot byte stride: 524,288 B (= `stride_h0_source` 131072 elements × 4 B).
  - Element address of state (layer L, slot s, head h, v, k): `base + L*304611328 + s*524288 + h*65536 + v*512 + k*4`.
  - The recipe `offset` fields for this pointer are unusable (the arena was misclassified as a weight alloc; offsets are negative). The layout above follows from the exact size factorization plus `stride_h0_source`, and matches the code path cited.
- `conv_state[0] = torch.zeros((34, 581, 3, 3072), bf16)` (memory_pool.py:616-623): 364,105,728 B; recipe segment 364,904,448 B (same 2 MiB roundup, 174 blocks).
  - Per-layer stride 10,716,672 B; per-slot 18,432 B.
  - The conv kernel sees the transposed view [581, 3072, 3] (kda_backend.py:731): window-major storage, channels contiguous.
- Indexing (layer, seq slot): `mamba2_layer_cache(layer_id)` → `State.at_layer_idx` (memory_pool.py:436-451) gives the per-layer block; `cache_indices = mamba_cache_indices` (kda_backend.py:559) gives the per-sequence slot; both kernels index `base + slot * slot_pitch`.
- Layer set: 34 blocks, in layer-id order of `linear_layer_ids` (cfg_glm5_next.py:251-253 → `mamba2_cache_params`, cfg_glm5_next.py:264-272). The MTP layer is not in the pool.
- State totals per sequence (all 34 KDA layers, one rank): conv 34 × 18,432 = 626,688 B; SSM 34 × 524,288 = 17,825,792 B; sum ≈ 18.44 MB.

## 5. `fused_sigmoid_gating_delta_rule_update_kernel`: exact math

Per (sequence n, head h = i_hv, one token). K=V=128, BV=32 tile (kernel loops NV=4 v-tiles per head).

1. Load initial state S ← pool[slot, h] (F32, [V=128 rows, K=128 cols], fused_recurrent.py:165-172); it is written back in place at the end (fused_recurrent.py:346-360; DISABLE_STATE_UPDATE=False at decode).
2. L2-normalize per 128-dim head vector (fused_recurrent.py:313-314): `q̂ = q / sqrt(Σq² + 1e-6)`, same for k. Then `q̂ *= 128^-0.5` (fused_recurrent.py:316).
3. Decay gate, per-K-dim vector (USE_LOWER_BOUND=True, fused_recurrent.py:231-233):
   `g[k] = -5.0 * sigmoid(exp(A_log[h]) * (a[h,k] + dt_bias[h,k]))` ∈ (-5, 0).
   (The `softplus_beta/threshold` args serve the unused `b_g = -exp(A_log)*softplus(a+dt_bias)` branch, fused_recurrent.py:236-243.)
4. Write gate: `β = sigmoid(b[h])` (fused_recurrent.py:246).
5. Decay: `S *= exp(g)` broadcast over v rows (per-K-column decay, fused_recurrent.py:320).
6. Delta rule (fused_recurrent.py:325, 328, 331): `v' = (v - Sᵀ·k̂) * β`; `S += k̂ ⊗ v'` (outer product, k varies over columns).
7. Output (fused_recurrent.py:334): `o = Sᵀ · q̂` → [128] BF16 store.
8. Output gating is NOT here; it is the sigmoid gate in `layer_norm_gated` (§3.5), driven by `g_proj_states`.

So: alpha = exp(g) (fine-grained per-K decay, "gated" delta rule), beta = sigmoid write gate, and the sigmoid output gate is a separate low-rank projection applied after the recurrence, inside o_norm.

## 6. mHC, op by op

Per layer the mHC skeleton is 6 launches: 2 pre-boundaries (attn, ffn) and 2 post-boundaries (attn, ffn), with the attn-post fused into the ffn-pre. Weight set per boundary: `hc_{attn,ffn}_fn` [24,16384] F32, `hc_{attn,ffn}_scale` [3], `hc_{attn,ffn}_base` [24]. The residual R is [T, 4, 4096] BF16; mix vector order in fn/base: rows 0-3 pre, 4-7 post, 8-23 comb (4×4).

Reference math (`_mhc_pre_torch`, mhc.py:1889-1928; `_mhc_post_torch`, mhc.py:1930-1941):
- `rms = rsqrt(mean(R²) + 1e-5)` over the full 16384; `mixes = (R_flat @ fnᵀ) * rms`.
- `pre[j]  = sigmoid(mixes[j]·scale[0] + base[j]) + 1e-6` (j<4)
- `post[j] = 2.0 * sigmoid(mixes[4+j]·scale[1] + base[4+j])`
- `comb[j,k] = mixes[8+j·4+k]·scale[2] + base[8+j·4+k]`, then Sinkhorn: row-softmax (max-subtracted), +1e-6, column-normalize (+1e-6), then 19 more row/col normalizations (20 iterations total, mhc.py:1005-1035 in the kernel; config `hc_sinkhorn_iters=20`).
- pre step output: `layer_input[h] = Σ_j pre[j]·R[j,h]`, then the sublayer's RMSNorm (fused).
- post step output: `R'[j,h] = post[j]·x[h] + Σ_k comb[k,j]·R[k,h]` (column-stochastic stream mixing + gated input injection).

### 6.1 `sm90_tf32_hc_prenorm_gemm` (sites 4, 23, 42, 61, 110, ...)

- DeepGEMM JIT kernel `sm90_tf32_hc_prenorm_gemm_impl<24,16384,64,32,64,64,...>` (prenorm.cuh:35-46), grid [64,1,1], block 256 (128 math + 128 TMA threads), smem 232448.
- Inputs: x = R_flat [1,16384] BF16; fn [24,16384] F32. N=24 mixing logits, K=16384, split-K over 64 CTAs (256 K each; `_compute_num_split_for_mhc_pre(1,16384) = min(132, 256/4) = 64`, mhc.py:853-860).
- It is NOT a mixing-matrix GEMM on the 4 streams; it is a plain GEMM that computes the 24 mixing LOGITS from the flattened 4-stream state, in TF32 (WGMMA, BF16 A converted to F32 in registers, prenorm.cuh:148-176), PLUS a fused per-token square-sum of x (`sqr_sum`, prenorm.cuh:163-164, 201-202, 233-241).
- Outputs: `gemm_out_mul` [64,1,24] F32 partials, `gemm_out_sqrsum` [64,1] F32 partials (mhc.py:1151-1170).
- The RMSNorm scales do NOT come from this kernel — it only produces the raw logits and the sum of squares; norm weights enter in the next kernel.

### 6.2 `mhc_pre_big_fuse_with_norm_tilelang_kernel` (sites 5, 15, 24, 34, ...)

- TileLang kernel (mhc.py:930-1093), grid [T,1,1] = [1,1,1], 96 threads, smem 37232.
- Warp split: threads 0-31 compute post/comb; threads 32-95 compute pre/layer_input (mhc.py:992-1088).
- Steps: reduce the 64 partials → `rms = rsqrt(Σsqrsum/16384 + 1e-5)`; `mixes = Σpartials · rms`; pre/post/sinkhorn exactly as in §6 reference; `layer_input` accumulated in F32 from the BF16 streams, rounded to BF16 in shared memory, then a second sweep applies `layer_input *= rsqrt(mean(·²)+norm_eps) · norm_weight` (mhc.py:1054-1085).
- `norm_weight` = the sublayer norm weight (input_layernorm for the attn pre, post_attention_layernorm for the ffn pre), `norm_eps = 1e-5` (resolved from the module, communicator_mhc.py:82-84; passed by glm5_next.py:818-834). With the norm fused, the communicator skips the standalone RMSNorm (`norm_fused=True`, communicator_mhc.py:92-93).
- Outputs: `post_mix` [1,4] F32, `comb_mix` [1,16] F32 (held in `MHCState.h_post/h_res`, communicator_mhc.py:67-78), `layer_input` [1,4096] BF16 (already normed).
- Scalar 1 in the recipe = num_tokens.

### 6.3 `mhc_fused_post_pre_fma_tilelang_kernel` (sites 14, 33, 52, 96, 120, ...)

- TileLang kernel (mhc.py:1385-1623), grid (T=1, 12, 8) = (num_tokens, num_mix_output_tiles, split_k), 256 threads, smem 96. tile_mix_outputs=2 → 24/2 = 12 column tiles; split_k=8 (T<8 and hidden≤4096, mhc.py:1710-1715) → 512 hidden per split. Scalars [1, 8] = (1, split_k=8).
- Inputs (glm5_next.py:856-894 → fused_mhc.py:353-423): prev comb (h_res) [1,4,4] F32, prev residual [1,4,4096] BF16, prev post (h_post) [1,4] F32, hidden_in = attention output [1,4096] BF16, pre_fn = `hc_ffn_fn` viewed [24,4,4096] F32.
- One launch does three things (mhc.py:1478-1560): (A) the DEFERRED attn post: `cur_residual[j,h] = bf16(post[j]·x[h] + Σ_k comb[k,j]·prev_residual[k,h])`; (B) the ffn-pre square-sum partials of cur_residual; (C) the ffn-pre GEMM partials `Σ pre_fn[o,r,h]·cur_residual[r,h]`.
- Outputs: `cur_residual_out` [1,4,4096] BF16 (the new residual — the post is DONE here), `mixes_partial_out` [8,1,24] F32, `sqrsum_partial_out` [8,1] F32.
- It is immediately followed by `mhc_pre_big_fuse_with_norm` (site 15; n_splits=8 partials, norm = post_attention_layernorm) which finishes the ffn pre (mhc.py:1793-1821).

### 6.4 `mhc_post_tilelang_kernel` (sites 22, 41, 60, 109, 133, 1344)

- TileLang kernel (mhc.py:1305-1352), grid [T,1,1], 128 threads, smem 28672 (= 2·(4·1024·2 + 1024·2) + 4·1024·2: double-buffered stream+input tiles plus one output tile, h_blk=1024).
- Inputs: a = comb [1,4,4] F32, b = residual [1,4,4096] BF16, c = post [1,4] F32, d = sublayer output x [1,4096] BF16.
- Math (mhc.py:1337-1347): `out[j,h] = c[j]·d[h] + Σ_i a[i,j]·b[i,h]` — the FFN post. Output [1,4,4096] BF16 = the next layer's residual.

### 6.5 "Deferred post/pre fusion": why 2 pre launches and 1 post per layer

- Every sublayer boundary in mHC is a post (mix sublayer output back into 4 streams) followed by a pre (mix streams down to 1 vector + norm). Naively 3 launches per boundary (post, pre-GEMM, big-fuse).
- The attn→FFN boundary is fused (2 launches): `mhc_fused_post_pre_fma` absorbs the attn post AND the ffn pre-GEMM in one launch; the big-fuse stage stays separate (comment at glm5_next.py:857-859: "two launches instead of three"). Enabled by default (`SGLANG_OPT_FUSE_MHC_POST_PRE=True`, environ.py:1540; gated to T≤16 by `_MHC_FUSED_BOUNDARY_MAX_TOKENS`, glm5_next.py:140, 861-862).
- The FFN→next-attn boundary is NOT fused in this build: `mhc_post` runs standalone (site 22), and the next layer's attn pre reads its output (sites 23-24). Only `hc_ffn_post_pre` is wired into the communicator's attn_to_mlp path (communicator_mhc.py:95-124); the layer-exit path (`mlp_combine`, communicator_mhc.py:125-127) calls plain `hc_post`.
- Hence per layer: prenorm GEMM ×2 (attn pre, and the ffn pre inside the fma), big-fuse ×2, fma ×1, post ×1 — "two mhc_pre launches, one post".

## 7. Final contract after layer 44 (recipe sites 1344-1350)

1. Site 1344: `mhc_post_tilelang` — layer 44's FFN post → [1,4,4096].
2. Site 1345: `at::reduce_kernel ... MeanOps<BFloat16, float, float>` grid [8,1,1] — `hc_contract` (mhc.py:1885-1886): `x.unflatten(-1,(4,-1)).mean(dim=-2)` — the UNWEIGHTED mean of the 4 streams, F32 accumulation, BF16 out. Runs inside the last layer's `postprocess_layer` (`is_last_layer`, communicator_mhc.py:318-321). After this, `residual = None` for the rest of the model.
3. Site 1346: `flashinfer cutlass RMSNormKernel`, grid [1,1,1], eps bits 925353388 = 1e-5 — the final `model.norm` (glm5_next.py:1209; `residual is None` → plain `self.norm(hidden_states)`), plain RMSNorm over 4096, weight `model.language_model.norm.weight` [4096] BF16.
4. Site 1347: `nvjet_sm90_tst_256x8_64x6_4x1_v_bz_TNT` grid [4,19,1] — lm_head GEMM `[1,4096] @ [4096, 19360]` (vocab 154880 / TP8), no split-K reduce kernel.
5. Site 1348: `_all_gather_kernel_inner` grid [4,1,1] — TP all-gather of logits; site 1349: copy of [1,154880]; site 1350: `ArgMaxOps` reduce — greedy sampling. Sites 1351-1354: bookkeeping (position increment, index_put, small copies).

## 8. Norms on the KDA/head path

| norm | where | shape | eps | style |
|---|---|---|---|---|
| input_layernorm | fused in big_fuse §6.2 (attn pre) | [4096] BF16 | 1e-5 | plain RMSNorm |
| post_attention_layernorm | fused in big_fuse §6.3 (ffn pre) | [4096] BF16 | 1e-5 | plain RMSNorm |
| mHC pre-mix RMS | big_fuse (no weight) | over 16384 | 1e-5 (`rms_eps` = config.rms_norm_eps, glm5_next.py:824) | plain, scales the 24 logits only |
| o_norm (gated) | §3.5 `layer_norm_gated` | [128] BF16 | 1e-5 | RMSNorm then ×sigmoid(g); gate = g_a→g_b projection |
| final norm | site 1346, flashinfer cutlass | [4096] BF16 | 1e-5 (bits 925353388) | plain RMSNorm |
| q/k L2-norm inside KDA | §5 step 2 | per 128-dim head | 1e-6 additive inside sqrt | `x/sqrt(Σx²+1e-6)` |
| (reference, DSA path) indexer k_norm | site 72 LayerNormKernel | [128] + bias | 1e-6 (bits 897988541) | LayerNorm, not on the KDA path |

No Gemma-style (1+w) anywhere on this path. hc_eps = 1e-6 is used twice in mHC: additive eps on pre-mix and Sinkhorn eps (glm5_next.py:826-827).

## 9. Layer-0 expand (1→4 streams)

- `MHCLayerCommunicator.prepare_attn` (communicator_mhc.py:477-490): `if self.is_first_layer: hidden_states = hc_expand(hidden_states, 4)`; `hc_expand = x.repeat(1, n)` (mhc.py:1881-1882).
- Recipe evidence: site 3 `at::elementwise_kernel<128,4> direct_copy` grid [32,1,1] — the repeat copy 4096 → 16384 BF16 (32 blocks × 128 threads × 4 elems = 16384). Site 4's prenorm GEMM then reads the expanded [1,16384] residual.
- Site 2 `FillFunctor<float>` (90 zeros) is the per-forward `zero_allocator` init (45 layers × 2 F32, glm5_next.py:1117-1119), not mHC.
- Sites 0-1: vocab embedding (154880×4096) + its TP all-reduce.
- Layer 0 needs NOTHING else vs steady layers: the capture's layer-0 block (sites 4-22) is launch-identical to layer 1 (sites 23-41). No missing launches; the only layer-0 extras are the expand copy (site 3) and the forward-prologue fills.
- Note: because the expand is a plain repeat, layer 0's first pre/post/comb are computed from 4 identical streams; thereafter the streams diverge through comb mixing.

## 10. Recipe site table

Per-layer offsets repeat exactly; KDA+dense layers (0,1,2) have 19 sites/layer, KDA+MoE layers 24, DSA+MoE layers 49. Abbreviations: W = weight, S = persistent state, T = transient scratch.

| sites (layer 0 / layer 4) | op | kernel | weights used | state touched |
|---|---|---|---|---|
| 2 / — | zero_allocator fill | `FillFunctor<float>` (90 F32) | — | T: zero buf |
| 3 / — | hc_expand 1→4 streams | `at::elementwise_kernel` repeat | — | T: residual [1,16384] |
| 4 / 110 | mHC attn-pre logit GEMM + Σx² | `sm90_tf32_hc_prenorm_gemm` grid [64] | W: hc_attn_fn [24,16384] F32 | T: mul [64,1,24], sqrsum [64,1] |
| 5 / 111 | mHC attn-pre mixes + sinkhorn + fused input_layernorm | `mhc_pre_big_fuse_with_norm` grid [1], 96 thr | W: hc_attn_scale [3], hc_attn_base [24], input_layernorm.weight | T: h_post [1,4], h_res [1,16], layer_input [1,4096] |
| 6-7 / 112-113 | fused qkvbfg_a GEMM [1,4096]→[1,3336] | `nvjet 128x8_64x12 splitK` grid [4,21] + `splitKreduce` [105] | W: fused_qkvbfg_a_proj [3336,4096] BF16 | T: qkv/beta/fg_a views |
| 8 / 114 | fg_b batched GEMM [2,1,128]→[2,1,1024] | `nvjet 64x8_64x16` grid [4,8] | W: fused_fg_b_proj [2,1024,128] BF16 | T: a (forget), g_proj (out gate) |
| 9 / 115 | causal conv1d update k=4 + silu | `_causal_conv1d_update_kernel` grid [1,12], 128 thr | W: qkv_conv1d [3072,4] F32 | S: conv pool slot [3,3072] BF16 (shift+append); T: qkv [1,3072] |
| 10 / 116 | delta-rule step (safe gate -5.0) | `fused_sigmoid_gating_delta_rule_update_kernel` grid [4,1,8], 32 thr | W: A_log [8], dt_bias [1024] F32 | S: SSM pool slot [8,128,128] F32 read+write; T: o [1,1,8,128] |
| 11 / 117 | gated RMSNorm (×sigmoid(g)) | `layer_norm_gated_fwd_kernel` grid [1], 128 thr, eps 1e-5 | W: o_norm.weight [128] BF16 | T: [8,128]→[1,1024] |
| 12 / 118 | o_proj [1,1024]→[1,4096] partial | `nvjet 64x8_64x16` grid [4,16] | W: o_proj [4096,1024] BF16 | T: partial out |
| 13 / 119 | TP all-reduce attn out | `all_reduce_1shot_push` grid [132] | — | T |
| 14 / 120 | deferred attn-post + ffn-pre GEMM partials | `mhc_fused_post_pre_fma` grid [1,12,8], 256 thr | W: hc_ffn_fn [24,16384] F32; T in: h_res/h_post, attn out | T: cur_residual [1,4,4096], partials [8,1,24]/[8,1] |
| 15 / 121 | ffn-pre mixes + sinkhorn + fused post_attention_layernorm | `mhc_pre_big_fuse_with_norm` grid [1] | W: hc_ffn_scale, hc_ffn_base, post_attention_layernorm.weight | T: new h_post/h_res, layer_input |
| 16-20 / 122-131 | dense: quant + FP8 GEMM [1,4096]→[1,3072] + silu(clamp 10) + quant + FP8 GEMM [1,1536]→[1,4096]; MoE: router tiny_n_gemm 288 + router_triton (topk, scale 2.4) + align + quant + fused_moe ×2 + sum_reduce | `per_token_group_quant`, `sm90_fp8_gemm_1d2d`, `silu_mul_clamp`; `tiny_n_gemm`, `_router_triton_kernel`, `fused_moe_kernel` ×2 | W: gate_up/down (FP8), expert weights (FP8) | T |
| 21 / 132 | TP all-reduce MLP/MoE out | `all_reduce_1shot_push` grid [132] | — | T |
| 22 / 133 | mHC FFN post | `mhc_post_tilelang` grid [1], 128 thr | T in: h_res/h_post; no W | T: residual [1,4,4096] |
| (DSA layer 3: 61-109) | same mHC skeleton: prenorm 61, big_fuse 62, DSA attn 63-95 (MLA norms eps 1e-5 sites 65-66, indexer LayerNorm eps 1e-6 site 72, fa3 decode 90-91), fma 96, big_fuse 97, MoE 98-108, mhc_post 109 | as above + DSA kernels | hc_attn/hc_ffn + MLA weights | DSA KV cache (out of scope) |
| 1344 | layer-44 FFN post | `mhc_post_tilelang` | T in: h_res/h_post | T |
| 1345 | hc_contract: mean of 4 streams | `at::reduce_kernel MeanOps` grid [8] | — | T: [1,4096] |
| 1346 | final RMSNorm | flashinfer cutlass `RMSNormKernel`, eps 1e-5 | W: model.norm.weight [4096] | T |
| 1347 | lm_head [1,4096]→[1,19360] | `nvjet 256x8_64x6` grid [4,19] | W: lm_head [154880/8, 4096] | T |
| 1348-1350 | all-gather logits, copy, argmax | `_all_gather_kernel_inner`, `direct_copy`, `ArgMaxOps` | — | T |

## 11. Notes and non-obvious findings

1. The recipe's `fused_sigmoid_gating_delta_rule_update` is the plain Triton chain, NOT the fused CUDA `kda_fused_decode` (that JIT kernel is gated to the Kimi-K3 handoff, kda_fused_decode.py:1-30, and the packed path rejects `lower_bound=-5.0`, kda_backend.py:740-744). GLM-5.3 therefore runs 3 kernels (conv, recurrence, gated norm) where K3 runs 1.
2. GLM-5.3's gate is the "safe gate": `g = -5·sigmoid(exp(A_log)·(a+dt_bias))`, a per-K-dim vector decay bounded to (-5,0). The `softplus_beta=1.0/threshold=20.0` scalar args are dead on this path.
3. KDA processes ONE token per sequence: the mHC pre contracts 4 streams to 1 vector before attention; the 4-stream state exists only as the residual. All KDA GEMMs are M=1; the 4 in the nvjet grids is split-K/batching, not tokens.
4. q/k/v conv1d are merged at load into ONE [3072,4] F32 depthwise conv over the packed qkv width (checkpoint has them separate, BF16; runtime F32). Conv state is one [3,3072] BF16 window per slot, not per-q/k/v tensors.
5. The SSM arena (10,357,833,728 B) = tensor [34, 581, 8, 128, 128] F32 + 2 MiB allocator rounding; per-layer stride 304,611,328 B, per-slot 524,288 B (matches `stride_h0_source=131072`). Pool size 580 slots (+1).
6. The o_norm sigmoid gate comes from the g_a→g_b low-rank (128) projection, produced by the second (batched-2) GEMM — it is not part of the big fused qkvbfg_a GEMM and not part of the recurrence.
7. The attn→FFN mHC boundary is fused (fma kernel), the FFN→attn boundary is not; per layer: 2 prenorm GEMMs, 2 big-fuses, 1 fma, 1 post.
8. Layer 0's only difference is the `repeat` expand copy (site 3); no launches are missing.
9. `hc_contract` at the end is an unweighted mean of the 4 streams (ATen MeanOps), then a plain 1e-5 RMSNorm before lm_head.
10. All mHC mixing weights (hc_*_fn [24,16384]) are F32 at runtime although BF16 in the checkpoint; the prenorm GEMM consumes them as the F32 B operand of a TF32 WGMMA GEMM.
