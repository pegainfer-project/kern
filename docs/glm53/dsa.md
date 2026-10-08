# GLM-5.3-Flash DSA decode path (sglang TP8, H100) - kernel-level spec

Status: engineering spec for the DeepSeek Sparse Attention (DSA) decode path.
Scope: one eager decode step, batch size 1, TP=8, one DSA layer (layer 3).
Recipe: `$GLM53_ARTIFACTS/recipe.json` (key `bs1`, 1351 launches).
Human-readable dump: `$GLM53_ARTIFACTS/decode_fwd25_seq.txt`.
DSA layer block: recipe call-site indices 61..109 inclusive.
File references live under $GLM53_ARTIFACTS unless noted.

## 0. Environment and provenance

- Server: `sglang.launch_server --model-path "$GLM53_CHECKPOINT" --tp 8` (our sglang launch script).
- sglang: `0.5.6.post3.dev11186+gcf5df8268` at `<sglang site-packages>/sglang/` (a fork of upstream sglang; this tree contains the kpool/DSA extensions used below).
- sgl-kernel: `0.3.21` (compiled ops in `sgl_kernel/flash_ops.abi3.so` and JIT `.cuh` files under `sglang/kernels/jit/csrc/`).
- deep_gemm package: `<sglang site-packages>/deep_gemm/` (headers under `include/deep_gemm/`).
- The fa3 (FlashAttention-3 Hopper) C++ sources are compiled into `flash_ops.abi3.so`. They are not shipped as source in the venv. Line numbers below for `flash.h`, `flash_api.cpp`, `mainloop_fwd_sm90_tma_gmma_ws.hpp`, `epilogue_fwd.hpp`, `tile_scheduler.hpp`, `flash_prepare_scheduler.cu`, `heuristics.h`, `flash_fwd_kernel_sm90.h`, `flash_fwd_combine_kernel.h` refer to the matching sgl-kernel 0.3.21 fa3 sources.
- `SP` = the sglang venv's site-packages dir.

## 1. Model facts used in this spec

From `config.json` in the `$GLM53_CHECKPOINT` snapshot:

- 45 layers (`num_hidden_layers`); `layer_types` puts `deepseek_sparse_attention` at layers {3,7,11,15,19,23,27,31,35,39,43}; all other layers are `linear_attention` (KDA).
- hidden 4096; `num_attention_heads` 64; `qk_nope_head_dim` 256; `qk_rope_head_dim` 0 (NoPE); `v_head_dim` 256; `q_lora_rank` 1536; `kv_lora_rank` 512; `mla_use_nope` true.
- Indexer: `index_n_heads` 32, `index_head_dim` 128, `index_topk` 2048, `index_kpool` 4, `index_kpool_compress` true, `index_kpool_always_select_tail` true.
- Hyper-connections: `mhc` true, `hc_mult` 4, `hc_sinkhorn_iters` 20, `hc_eps` 1e-6. So the residual stream is 4 x 4096 = 16384 wide and the mixing width is `mix_hc = (2 + hc_mult) * hc_mult = 24`.
- MoE: `n_routed_experts` 288, `n_shared_experts` 1, `num_experts_per_tok` 8, `moe_intermediate_size` 2048 (256 per rank at TP8), `moe_router_dtype` float32, `scoring_func` sigmoid, `topk_method` noaux_tc, `routed_scaling_factor` 2.5. sglang fuses the 1 shared expert into the routed expert table (id 288), so the effective topk in the kernels is 9 (`glm5_next.py:1370-1377`).
- Quantization (`quantization_config`): FP8 e4m3, dynamic activations, block 128x128 weights; `modules_to_not_convert` excludes embed/lm_head/hyper_connection/attention-op modules. The DSA linear layers q_a/kv_a/q_b/o_proj run as FP8 block-quant GEMMs; kv_b and the indexer run in bf16; `weights_proj` runs in fp32.
- `rms_norm_eps` 1e-5.

Per-rank shard sizes at TP8 (64 heads / 8 ranks = 8 heads per rank):

- q_b_proj out: 8 heads x 256 = 2048. kv_b_proj out: 8 heads x (256+256) = 4096. o_proj in: 8 x 256 = 2048.

Captured step shape (T = 1 token, S = 1 sequence):

- The true sequence length is 3 (2 past tokens + the current token), NOT 2051. Proof from recorded scalars: (a) site 85 p7 = page_table_1 row stride = 3, and page_table_1 is a contiguous [1, max_seqlen_k] copy (`dsa_backend.py:865-867`; for eager bs=1 decode max_seqlen_k = seq_lens.max(), `dsa_backend.py:855-863`); (b) site 84 p2 = pooled block-table row stride = 1, i.e. the table has shape [1, 1]: the whole pooled history fits one 64-slot page, so floor(seq/4) <= 64; (c) site 84 p1 = logits row stride = 256 = align(64, 256), i.e. pool_max_seq_len = 64 = one pooled page (DeepGEMM host wrapper, `attention.hpp:517-524`). (b) and (c) bound seq <= 256; (a) pins seq = 3.
- The value 2051 that appears in sites 85/88/89/90 is a constant, not a length: `index_topk (2048) + (index_kpool - 1) (3)` = the fixed topk output width (`kpool_topk_transform.py:53`: `out_cols = topk + pool_size - 1`).
- Per-row effective attention length: `dsa_cache_seqlens = min(4*floor(seq/4), 2048) + seq%4` (`dsa/utils.py:72-84`) = min(0, 2048) + 3 = 3. So the capture has 0 closed kpool groups and 3 tail tokens. The kpool indexer runs in degenerate mode: site 85 takes its identity path and never reads the indexer logits (section 5.12), and fa3 attends to 3 slots.

## 2. DSA layer call graph (decode, eager, bs=1)

Python path per DSA layer (`SP/sglang/srt/models/glm5_next.py`):

1. `Glm5NextDecoderLayer.forward` -> `MHCLayerCommunicator.prepare_attn` (`SP/sglang/srt/layers/communicator_mhc.py:477-519`) -> `attn_split` (`communicator_mhc.py:85-93`) -> `hc_attn_pre` with `input_layernorm` fused. Sites 61-62.
2. `DeepseekV2AttentionMLA` (`glm5_next.py:699-716`, `skip_rope=True`) -> `forward_absorb_prepare` (`SP/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py:286`):
   - fused q_a+kv_a FP8 GEMM (site 64) via `fused_qkv_a_proj_with_mqa` (`deepseek_v2.py:2013-2019`);
   - `q_a_layernorm` (site 65), `kv_a_layernorm` (site 66) (`forward_mla.py:344-350`);
   - `q_b_proj_forward` (sites 67-68) (`forward_mla.py:369/395`);
   - DSA indexer (`IndexerKPool.forward_cuda`, `SP/sglang/srt/layers/attention/dsa/dsa_indexer_kpool.py:1502`) -> `_forward_cuda_impl` (`dsa_indexer_kpool.py:1538`) (sites 69-85);
   - absorbed bmm `q_nope_out = bmm(q_nope^T, w_kc)` (site 86) (`forward_mla.py:568`);
   - KV cache write (site 87) inside the attention op;
   - `attn_mqa` = `DeepseekSparseAttnBackend.forward_decode` -> `_forward_fa3` (`SP/sglang/srt/layers/attention/dsa_backend.py:2428`) (sites 88-91);
   - `attn_bmm_output = bmm(attn_out^T, w_vc)` (site 92) (`forward_mla.py:890-905`);
   - `o_proj` FP8 GEMM (sites 93-94) + TP all-reduce (site 95).
3. `prepare_mlp` (`communicator_mhc.py:521-531`) -> fused `hc_ffn_post_pre` + `hc_ffn_pre` with `post_attention_layernorm` (sites 96-97).
4. MoE (`DeepseekV2MoE`, used as `Glm5NextMoE`, `glm5_next.py:96`): router GEMM (98), routing (99), gather/align (100-101), FP8 quant (102), fused MoE w13 (103), SwiGLU clamp (104), quant (105), fused MoE w2 (106), topk sum (107), all-reduce (108).
5. `ffn_exit.finish` -> `hc_post` (site 109) (`communicator_mhc.py:116-126`).

## 3. Weights and load-time transforms (DSA layer, per rank)

Checkpoint names are relative to `model.layers.3.`.

| Tensor (checkpoint) | Shape | Dtype in ckpt | Runtime tensor | Load transform |
|---|---|---|---|---|
| `self_attn.q_a_proj.weight` | [1536, 4096] | fp8 e4m3 | `fused_qkv_a_proj_with_mqa.weight` [2048, 4096] fp8 | cat with kv_a along dim 0 (`glm5_next.py:1680-1721`, mapping at `glm5_next.py:1229`) |
| `self_attn.q_a_proj.weight_scale_inv` | [12, 32] | fp32 | `fused_qkv_a_proj_with_mqa.weight_scale_inv` [16, 32] | cat along dim 0 (same code path; `weight_scale` -> `weight_scale_inv` rename at `glm5_next.py:1561-1566`) |
| `self_attn.kv_a_proj_with_mqa.weight` | [512, 4096] | fp8 e4m3 | (rows 1536..2047 of fused tensor) | as above |
| `self_attn.q_a_layernorm.weight` | [1536] | bf16 | same | - |
| `self_attn.kv_a_layernorm.weight` | [512] | bf16 | same | - |
| `self_attn.q_b_proj.weight` | [16384, 1536] full; [2048, 1536] per rank | fp8 e4m3 | `q_b_proj.weight` + `weight_scale_inv` [128, 12] full / [16, 12] per rank | ColumnParallelLinear shard over heads (`deepseek_v2.py:2021-2030`) |
| `self_attn.kv_b_proj.weight` | [32768, 512] full; [4096, 512] per rank | bf16 | split into `w_kc` [8, 256, 512] and `w_vc` [8, 512, 256] | `post_load_weights`: `w.unflatten(0, (-1, 512)).split([256, 256], dim=1)`; `w_kc` stored transposed-for-bmm, `w_vc = w_vc.contiguous().transpose(1,2)` (`deepseek_weight_loader.py:734-758`) |
| `self_attn.o_proj.weight` | [4096, 16384] full; [4096, 2048] per rank | fp8 e4m3 | `o_proj.weight` + `weight_scale_inv` [32, 128] full / [32, 16] per rank | RowParallelLinear shard over input dim (`deepseek_v2.py:2100-2109`) |
| `self_attn.indexer.wq_b.weight` | [4096, 1536] | bf16 | same | ReplicatedLinear; input is q_lora (1536) (`dsa_indexer_kpool.py:141-147`) |
| `self_attn.indexer.wk.weight` | [128, 4096] | bf16 | same | ReplicatedLinear (`dsa_indexer_kpool.py:149-155`) |
| `self_attn.indexer.weights_proj.weight` | [32, 4096] | bf16 in ckpt, cast to fp32 on load | fp32 | `params_dtype=torch.float32` (`dsa_indexer_kpool.py:157-163`) |
| `self_attn.indexer.k_norm.weight` / `.bias` | [128] | bf16 -> fp32 | fp32 | `LayerNorm(self.head_dim, dtype=torch.float32)` (`dsa_indexer_kpool.py:164`) |
| `self_attn.indexer.index_kpool_compress_ape` | [4, 128] | fp32 | same | additive positional encoding for pool slots (`dsa_indexer_kpool.py:129-131`) |
| `self_attn.indexer.index_kpool_compress_gate` | [128, 4096] | bf16 | same | pool-score projection (`dsa_indexer_kpool.py:133-135`) |
| `hc_attn_fn` | [24, 16384] | fp32 | same | hyper-connection mixing projection (`glm5_next.py:776-778`) |
| `hc_attn_base` | [24] | fp32 | same | `glm5_next.py:774` |
| `hc_attn_scale` | [3] | fp32 | same | `glm5_next.py:775` |
| `hc_ffn_fn`, `hc_ffn_base`, `hc_ffn_scale` | as above | fp32 | same | `glm5_next.py:780-783` |
| `input_layernorm.weight` | [4096] | bf16 | same | RMSNorm eps 1e-5 (`glm5_next.py:787-790`) |
| `post_attention_layernorm.weight` | [4096] | bf16 | same | |
| `mlp.gate.weight` | [288, 4096] | fp32 | same | MoE router (`moe_router_dtype=float32`) |
| `mlp.experts.{0..288}.gate_up_proj` / `down_proj` (+ scales) | per rank w13 [289, 512, 4096] fp8, w2 [289, 4096, 256] fp8 | fp8 e4m3 | FusedMoE weights `w13_weight`, `w2_weight` | expert 288 = fused shared expert (`glm5_next.py:1370-1377`, loader rename `glm5_next.py:1588-1592`) |

Notes on the FP8 GEMM path (sites 64/68/94): weights are e4m3 with per-128x128-block `weight_scale_inv` (fp32). Activations are quantized per token per 128-group by `per_token_group_quant_flat_kernel` (scale layout column-major for the DeepGEMM 1D2D path: `QuantTrait<bf16, fp8_e4m3, 128, kUe8m0=false, kRowMajor=false, kAligned=true, kFuseSiluAndMul=false>`, see `SP/sglang/kernels/jit/csrc/gemm/per_token_group_quant.cuh:222-238`). The GEMM is DeepGEMM `sm90_fp8_gemm_1d2d_impl` (1D activation scales x 2D weight block scales), persistent over 132 SMs (`SP/deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh:38-50`).

## 4. State and cache layouts

### 4.1 MLA KV pool (attention KV)

- Owner: `MLATokenToKVPool` (`SP/sglang/srt/mem_cache/memory_pool.py:4419`). One buffer per layer: `kv_buffer[layer]` shape `(size + page_size, 1, kv_cache_dim)` bf16 (`memory_pool.py:4475-4490`). `kv_cache_dim = kv_lora_rank + qk_rope_head_dim = 512 + 0 = 512`. So each slot is one 512-element bf16 row (1024 bytes): the compressed latent c_kv. There is no separate K and V; the same latent serves as both.
- Physical pages are 64 consecutive slots (`page_size = 64` for DSA on CUDA, `memory_pool.py:4964-4968`). The per-request `req_to_token` page table maps token position -> flat slot id.
- Slot 0 is reserved for padding (`set_mla_kv_buffer` docstring, `SP/sglang/kernels/ops/kvcache/mla_buffer.py:147-149`).
- Write per decode token: `set_mla_kv_buffer_kernel_norope` copies the 512-element bf16 `k_nope` (= kv_a_layernorm output) to `kv_buffer[loc]` (see site 87).

### 4.2 Index-K kpool compressed cache (indexer, FP8)

- Owner: `DSATokenToKVPool.index_key_cache` (`memory_pool.py:4893`, `SP/sglang/srt/mem_cache/index_key_cache.py:14`).
- Buffer: `buffer[layer]` uint8 `[num_pages, 8448]` where `num_pages = (index_buf_size + page_size + 1) // page_size` and `8448 = page_size * (index_head_dim + (index_head_dim // quant_block_size) * 4) = 64 * (128 + 4)` (`index_key_cache.py:34-40`).
- Per-page byte layout (write side: `_kpool_cache_k_offsets`, `SP/sglang/srt/layers/attention/dsa/kpool_fp8_index.py:25-49`, non-preshuffled path):
  - bytes [0, 8192): 64 slots x 128 bytes of FP8 e4m3 index-k. Slot `s` at `page * 8448 + s * 128`.
  - bytes [8192, 8448): 64 slots x 4 bytes fp32 dequant scale. Slot `s` scale at fp32 index `page * (8448/4) + 8192/4 + s` (`S_OFFSET_NBYTES_IN_PAGE = slots_per_page * index_head_dim = 8192`, `kpool_fp8_index.py:866-868`).
- One slot holds one *pooled* entry = the compressed key of 4 consecutive tokens (pool size 4). Pool `p` of a request lives in slot `p % 64` of the physical page found via the *token* page table at row `(p // 64) * 4` (`_kpool_decode_update_and_maybe_write_cache_kernel`, `kpool_fp8_index.py:1155-1168`; `SLOTS_PER_PAGE = 64`). So pooled entries reuse the same physical pages that the token-level layout would use, at 4x density. `build_pooled_page_table_64` subsamples the token page table with stride 4 for the DeepGEMM logits kernel (`kpool_fp8_index.py:56-72`).

### 4.3 Indexer tail cache (bf16 ring buffer)

- `_compress_tail_k[layer]` and `_compress_tail_score[layer]`: both bf16 `[req_pool_size, 4, 128]`, contiguous (`memory_pool.py:4996-5046`; `tail_width = index_kpool + tail_extra_slots = 4`, `tail_extra_slots = 0` for this model).
- Holds the raw (k_norm'ed, *not* Hadamard-rotated, *not* quantized) index-k of the last up-to-4 tokens of each request, at ring position `pos % 4`, plus the gate score per element.
- The last `seq_len % 4` tokens (the open pool) are always selected into the attention candidate set (`index_kpool_always_select_tail = true`), so their logits are never computed; see site 85.

### 4.4 Topk / fa3 workspace buffers

- Indexer output `topk_indices`: int32 `[S, 2051]`. 2051 is a constant width (2048 + 3), not the sequence length. Row layout from site 85: [4*floor(seq/4) pool-expanded slots][seq%4 tail slots][-1 pad to 2051]. This capture: 3 valid + 2048 padding. Written by site 85.
- `dsa_cache_seqlens_int32`: int32 `[S]`, the per-row valid count (= 3 here); used as `cache_seqlens` (seqused_k) for fa3. The -1 padding of the page table is replaced by slot 0 via `page_table.clamp(min=0)` at site 88 (`dsa_backend.py:2453-2454`).
- fa3 split workspace: `oaccum` fp32 `[num_splits=29, num_heads=8, total_q=1, 512]` and `lseaccum` fp32 `[29, 8, 1]`, sized by the STATIC split cap 29 (host heuristic from the 2051-wide page table, flash_api.cpp:1099-1100). The executed split count is dynamic (= 1 here, site 89). Merged by site 91 into `out` bf16 `[1, 8, 512]`.
- Scheduler metadata buffers (int32): `tile_count_semaphore`, `num_m_blocks_ptr`, `num_splits_dynamic_ptr`, `varlen_batch_idx_ptr` (site 89 outputs; read by sites 90-91).

## 5. Site-by-site documentation (61..109)

Conventions: `T` = tokens this step (=1 in the recipe), `S` = sequences (=1). `grid_b2` is the recorded grid for the bs=2 capture when present. Pointer classifications (`weight`/`state`/`rotating`) are from the recipe. Scalars are decoded from the recorded little-endian hex.

### 5.1 Attention pre-norm (hyper-connection pre): sites 61-62

**Site 61 - `deep_gemm::sm90_tf32_hc_prenorm_gemm_impl<24, 16384, 64, 32, 64, 64, 128, 12, 128, 128>`** grid [64,1,1], block 256, smem 232448.
- Op: hyper-connection mixing projection + RMSNorm sum-of-squares, fused. Computes `D [T, 24] = A [T, 16384] (bf16) @ B [16384, 24] (fp32 as tf32)` and `sqr_sum [T] = sum(A^2, dim=1)` (fp32). Kernel signature `(shape_m, tensor_map_a, tensor_map_b, tensor_map_d, float* sqr_sum)` (`SP/deep_gemm/include/deep_gemm/impls/sm90_tf32_hc_prenorm_gemm.cuh:35-46`).
- Template args: SHAPE_N=24, SHAPE_K=16384, BLOCK_M=64, BLOCK_N=32, BLOCK_K=64, kNumSplits=64, kSwizzleCDMode=128, kNumStages=12, kNumMathThreads=128, kNumTMAThreads=128.
- Inputs: A = residual stream `hidden_states` [T, 16384] bf16 (4 hyper-connection streams x 4096). B = `hc_attn_fn` [24, 16384] fp32 (p4 ptr[weight]). D and sqr_sum are small fp32 activations.
- Scalars: p0 = shape_m = 1 (= T). p1..p3 = 128-byte TMA descriptors for A, B, D.
- Grid: 64 = kNumSplits (K-split factor), independent of T. grid_b2 = [64] confirms.
- Called from `mhc_pre_gemm_sqrsum` path of `hc_pre` (`SP/sglang/kernels/ops/layernorm/mhc.py:635-716`; dispatch via `_mhc_pre_dispatch` at `mhc.py:1943`, host wrapper `tf32_hc_prenorm_gemm` in the deep_gemm package).

**Site 62 - `mhc_pre_big_fuse_with_norm_tilelang_kernel`** grid [T,1,1], block 96, smem 37232.
- Op: consumes D [T,24] + `hc_attn_base` [24] + `hc_attn_scale` [3], runs 20 sinkhorn iterations on the 4x4 residual-mix part (h_res, 16 values) with eps 1e-6, computes the combined layer input `sum_i h_pre[i] * stream_i` [T, 4096], applies RMSNorm with `input_layernorm.weight` (eps 1e-5) using the sqr_sum from site 61, and emits `h_post` [T, 4] and `h_res` [T, 16] (fp32) for the post-attention mix.
- TileLang JIT kernel defined in `mhc.py` (`mhc_pre_big_fuse_with_norm_tilelang`, `mhc.py:930`; wrapper `hc_pre` `mhc.py:1095`; math doc at `glm5_next.py:818-854`).
 - TileLang JIT kernel defined in `mhc.py` (`mhc_pre_big_fuse_with_norm_tilelang`, `mhc.py:930`; public wrapper `hc_pre`, `mhc.py:2029`; math doc at `glm5_next.py:818-854`).
- Params: p0-p8 = 9 pointers (D, sqr_sum, hc_base, hc_scale, residual A, norm weight, and 3 outputs), p9 = T (=1). Grid = [T] (grid_b2 = [2]).

### 5.2 Fused q_a + kv_a FP8 GEMM: sites 63-64

**Site 63 - `sglang::per_token_group_quant_flat_kernel<QuantTrait<bf16, fp8_e4m3, 128, false, false, true, false>, true>`** grid [T], block 256.
- Op: dynamic per-token-per-128-group FP8 quantization of the normed hidden [T, 4096] bf16 -> fp8 e4m3 [T, 4096] + fp32 scales [32 groups, T] in column-major (DeepGEMM 1D2D) layout. Trait flags: kUe8m0=false, kRowMajor=false, kAligned=true (`per_token_group_quant.cuh:222-238`).
- Params: one 80-byte `QuantKernelParams` blob (input ptr, output ptr, scale ptr + strides, num_tokens=1, hidden_size=4096; struct at `per_token_group_quant.cuh:195-201`).
- Grid = [T] (grid_b2 = [2]).

**Site 64 - `deep_gemm::sm90_fp8_gemm_1d2d_impl<MajorK, 0,0,0, 1, 16, 16, 128, 128, 32, 16, 128, 128, 1, false, 132, GemmType::Normal, bf16, EpilogueIdentity>`** grid [132,1,1], block 256, smem 68992.
- Op: fused q_a+kv_a projection: `out [T, 2048] bf16 = act_fp8 [T, 4096] @ fused_qkv_a_proj_with_mqa.weight^T [2048, 4096]` with 1D activation scales (site 63) x 2D block scales (`weight_scale_inv` [16, 32]).
- Template (decoded from the mangled name; parameter order per `sm90_fp8_gemm_1d2d.cuh:38-49`): kMajorSFB=Major::K, SHAPE_M/N/K=0 (runtime), kNumGroups=1, BLOCK_M=16, BLOCK_N=16, BLOCK_K=128, kSwizzleAMode=128, kSwizzleBMode=128, kSwizzleDMode=32, kNumStages=16, kNumTMAThreads=128, kNumMathThreads=128 (block = 256), kNumTMAMulticast=1 (no multicast), kIsTMAMulticastOnA=false, kNumSMs=132, GemmType::Normal, cd=bf16, EpilogueIdentity.
- Scalars (kernel signature `sm90_fp8_gemm_1d2d.cuh:50-56`): p0 = sfb ptr (activation scales), p1 = grouped_layout int* = null (GemmType::Normal), p2 = shape_m = 1 (T), p3 = shape_n = 2048, p4 = shape_k = 4096. p5-p8 = 128-byte TMA descriptors tensor_map_a (activations), tensor_map_b (weight), tensor_map_d (output), tensor_map_sfa (activation scales).
- Grid: 132 persistent CTAs (H100 SM count), independent of T (grid_b2 = [132]).
- Output rows: 0..1535 = q_a latent (q), 1536..2047 = kv_a latent (c_kv). Split at `forward_mla.py:331-339` via `fetch_qkv_latent().split([1536, 512])`.

### 5.3 q_a / kv_a layernorms: sites 65-66

**Site 65 - flashinfer cutlass `RMSNormKernel` (bf16, n=1536)** grid [1], block 128, smem 12304.
- Op: `q = q_a_layernorm(q_a)` over 1536 (`forward_mla.py:349`), weight = `q_a_layernorm.weight` [1536].
- Scalars: p0/p2 = 16-byte tensor-descriptor blobs (input [1,1536], output), p1 = norm weight ptr, p3 = 1 (rows), p4 = `0x3727c5ac` = 1e-5f (eps = `rms_norm_eps`).

**Site 66 - flashinfer cutlass `RMSNormKernel` (bf16, n=512)** grid [1], block 128, smem 4112.
- Op: `k_nope = kv_a_layernorm(kv_a)` over 512 (`forward_mla.py:350`), weight = `kv_a_layernorm.weight` [512]. Same scalar layout; eps 1e-5.
- The 512-element bf16 output is the latent c_kv for this token; it is later written into the MLA KV pool (site 87).

### 5.4 q_b FP8 GEMM: sites 67-68

**Site 67 - `per_token_group_quant_flat_kernel<...128, false, false, true, false>`** grid [T], block 256.
- Op: quantize q_a_layernorm output [T, 1536] bf16 -> fp8 + scales [12, T]. grid_b2 = [1] (the bs2 capture still fits one CTA).

**Site 68 - `sm90_fp8_gemm_1d2d_impl<MajorK, 0,0,0, 1, 16, 16, 128, 128, 32, 16, 128, 128, 1, false, 132, GemmType::Normal, bf16, EpilogueIdentity>`** grid [132], block 256, smem 68912.
- Op: q_b projection: `q [T, 2048] bf16 = q_a_fp8 [T, 1536] @ q_b_proj.weight^T [2048, 1536]`.
- Scalars: p2 = M = 1, p3 = N = 2048, p4 = K = 1536; p0 = activation scale ptr; p5-p8 TMA descriptors. Same template as site 64 (BLOCK_N=16); only N/K differ.
- Output viewed as [T, 8 heads, 256] (qk_nope; rope part is empty). Call: `q_b_proj_forward` (`forward_mla.py:369`, `deepseek_v2.py:2484-2495`).

### 5.5 Indexer q/k projections: sites 69-72

**Site 69 - `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT`** grid [4,16,1], block 384, smem 164308.
- Op: indexer query projection `query = wq_b(q_lora)`: bf16 GEMM [T, 1536] x [4096, 1536]^T -> [T, 4096], viewed [T, 32, 128] (`dsa_indexer_kpool.py:618-620`; `wq_b` is ReplicatedLinear 1536 -> 32*128, `dsa_indexer_kpool.py:141-147`).
- Input: `q_lora` = q_a_layernorm output [T, 1536] bf16 (NOT the raw hidden states; proven by the call `self.indexer(x=hidden_states, q_lora=q_lora, ...)` at `forward_mla.py:446` and the wq_b input size = `q_lora_rank` at `dsa_indexer_kpool.py:143`).
- cuBLASLt heuristic kernel (bf16 TN); grid is chosen by cuBLAS and is bs-invariant here (grid_b2 = [4,16]).

**Sites 70-71 - `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` + `cublasLt::splitKreduce_kernel<32,16,...>`** grids [2,3,1] and [4,1,1].
- Op: indexer key projection `key = wk(x)`: bf16 GEMM [T, 4096] x [128, 4096]^T -> [T, 128] (`dsa_indexer_kpool.py:625`).
- Input: `x` = hidden states [T, 4096] bf16; weight = `indexer.wk.weight` [128, 4096] bf16.
- Split-K (2 partials x 3 tiles) plus fp32 reduce; site 71 blob = cublas splitK workspace params; reduce grid [4] = ceil(128/32).

**Site 72 - flashinfer cutlass `LayerNormKernel` (bf16, n=128)** grid [T], block 32, smem 8.
- Op: `key = k_norm(key)` (`dsa_indexer_kpool.py:626`), LayerNorm over 128 with `indexer.k_norm.weight` and `.bias` (fp32 casts; `LayerNorm(128, dtype=float32)`, `dsa_indexer_kpool.py:164`).
- Scalars: p0/p1 = 24-byte tensor descriptors (in/out [1,128]), p2 = weight ptr, p3 = bias ptr, p4 = 1 (rows), p5 = `0x358637bd` = 1e-6f (eps).

### 5.6 Hadamard + act_quant of indexer q: sites 73-74

**Site 73 - `sglang::fast_hadamard_transform_kernel<FastHadamardKernelTraits<16, 7, bf16>>`** grid [32*T], block 16, smem 512.
- Op: `rotate_activation(query)`: orthonormal Hadamard H_128/sqrt(128) applied to every [128] row of query [T, 32, 128] (`dsa_indexer_kpool.py:623` for the dual-stream path, `dsa_indexer_kpool.py:651` for the eager path; `rotate_activation` in `SP/sglang/kernels/ops/quantization/hadamard.py`).
- Traits: 16 threads per row, 2^7 = 128 elements per row (`SP/sglang/kernels/jit/csrc/fast-hadamard-transform/hadamard_jit.cuh:36-52`, kernel at line 119).
- Params: one 56-byte `HadamardParamsBase` blob (in/out ptrs, dim=128, log=7, strides). Grid = one block per row = 32*T (grid_b2 = [64]).
- Note: the index-k of past tokens in the pooled cache was rotated by the same transform when it was written (see site 77), so q·k is preserved (H is orthogonal). The tail cache holds un-rotated k, and the tail logits are never computed.

**Site 74 - `_act_quant_kernel` (Triton)** grid [cdiv(32T,32)=T, cdiv(128,128)=1], block 128, smem 128.
- Op: `q_fp8, q_scale = act_quant(query, block_size=128, scale_fmt='ue8m0')` (`dsa_indexer_kpool.py:1660-1664`; Triton kernel `SP/sglang/kernels/ops/attention/dsa/triton_kernel.py:86-130`).
- Input [T*32, 128] bf16 -> output fp8 e4m3 [T, 32, 128] + scales [T, 32, 1] fp32 (one group of 128 per head row).
- Scalars: p0 = x ptr, p1 = y ptr, p2 = s ptr, p3 = M = 32, p4 = N = 128, p5/p6 = 0. `scale_fmt = "ue8m0"` is set at indexer construction (`deepseek_v2.py:2079`) -> `round_scale=True` (scale rounded up to a power of 2, `triton_kernel.py:121`).

### 5.7 kpool gate score: sites 75-76

**Sites 75-76 - `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT` + `splitKreduce_kernel`** grids [2,3,1] and [4,1,1].
- Op: `gate_score = F.linear(x, index_kpool_compress_gate)`: bf16 GEMM [T, 4096] x [128, 4096]^T -> [T, 128] (`dsa_indexer_kpool.py:637-639`; weight `indexer.index_kpool_compress_gate` [128, 4096] bf16, `dsa_indexer_kpool.py:133-135`).
- The gate score is the per-element pool membership score used by the softmax pool compressor (site 77). Identical GEMM shape as site 70, hence identical grid.

### 5.8 kpool cache update: site 77

**Site 77 - `_kpool_decode_update_and_maybe_write_cache_kernel` (Triton)** grid [T], block 128, smem 512.
- Op: `kpool_decode_update_index_cache` (`memory_pool.py:5078-5112` -> `kpool_fp8_index.py:791-868`; kernel `kpool_fp8_index.py:1027-1195`). One program per token. Two actions:
  1. Tail ring update (always): store current `key` (bf16, k_norm'ed, unrotated) and `gate_score` into `_compress_tail_k[req, pos % 4]` and `_compress_tail_score[req, pos % 4]` (`kpool_fp8_index.py:1188-1194`).
  2. Pool compression (only when `pos % 4 == 3`): softmax over the 4 pool slots of `(tail_score + ape)` per element (`ape` = `index_kpool_compress_ape` [4,128] fp32), weighted average of the 4 bf16 keys in fp32, round-trip through bf16, apply Hadamard (`_hadamard128`, includes the 1/sqrt(128) factor, `kpool_fp8_index.py:880-890`), FP8 e4m3 quantize with fp32 scale (absmax/448, optionally power-of-2 when `scale_fmt` set; here `round_scale=True` since scale_fmt="ue8m0"), and write the 128 FP8 bytes + fp32 scale into the pooled cache page (layout in section 4.2).
- Recorded scalars: the seven 4-byte values [512, 128, 512, 128, 128, 128, 128] = strides tail_k(512=4*128, 128), tail_score(512, 128), key(128), slot_score(128), ape(128). Pointers: buf (kpool uint8 cache), tail_k, tail_score, key, slot_score, ape, block_tables (token page table [S, cols] int32), req_pool_indices, positions, seq_lens, out_cache_loc.
- For this capture the decode position is pos = seq-1 = 2 (`write_start = seq_lens - 1` for decode, `dsa_backend_kpool.py:117-122`), so `pos % 4 = 2 != 3`: only the tail write happens; no pooled-cache write this step.
- `index_kpool_compress_ape` = p? ptr; `ape` is loaded with stride 128 (`ape_stride_0`).

### 5.9 Indexer head-gate weights: sites 78-81

**Site 78 - `triton_poi_fused__to_copy_0`** grid [16*T], block 128.
- Op: `x.float()` cast [T, 4096] bf16 -> fp32 (first stage of the `@torch.compile`'d `_get_logits_head_gate`, `dsa_indexer_kpool.py:189-195`). Scalar p2 = 4096 (numel).

**Site 79 - `cublas dot_kernel<float, 128, 0, cublasDotParams<cublasGemvTensorStridedBatched...>>`** grid [16,1,32], block 128.
- Op: `weights = weights_proj(x_fp32)`: fp32 GEMV [T, 4096] x [32, 4096]^T -> [T, 32] computed as 32 strided-batched dots, each split over 16 blocks (`weights_proj` is ReplicatedLinear 4096 -> 32 in fp32, `dsa_indexer_kpool.py:157-163`).
- Weight: `indexer.weights_proj.weight` [32, 4096] fp32. bs-invariant grid (batch = 32 output rows... for T>1 it is one batched GEMV per token).

**Site 80 - `cublas reduce_1Block_kernel<float, 128, 7, ...>`** grid [1,1,32], block 128.
- Op: split reduction of site 79 partials -> weights [T, 32] fp32. Scalars: alpha = 1.0f (`0x3f800000`), 16 splits, etc.

**Site 81 - `triton_poi_fused_mul_unsqueeze_1`** grid [S*T], block 32.
- Op: fused `weights = weights * 32^-0.5; weights.unsqueeze(-1) * q_scale * softmax_scale` (rest of `_get_logits_head_gate`, `dsa_indexer_kpool.py:191-194`). Output `weights` [T, 32, 1] fp32.
- Scalars: p2 = double `0x3fb6a09e667f3bcd` = 0.0883883 = 128^-0.5 (`self.softmax_scale = head_dim**-0.5`, `dsa_indexer_kpool.py:186`); p3 = 32 (numel); p0 = weights ptr, p1 = q_scale ptr (from site 74). The `32^-0.5` factor is `n_heads**-0.5` folded by torch.compile.

### 5.10 Logits scheduler metadata + clamp: sites 82-83

**Site 82 - `at::native::vectorized_elementwise_kernel<4, ... clamp ...>`** grid [1], block 128.
- Op: `pool_context_lens.clamp(min=1)` ([S,1] int32 -> [S,1]) before the metadata kernel (`dsa_indexer_kpool.py:792-793`, inside `_get_kpool_decode_metadata` at `dsa_indexer_kpool.py:735-797`). pool_context_lens = floor(seq_len/4) = 0 for this row, so the clamp rewrites it to 1 (the DeepGEMM metadata/logits kernels require context >= 1; site 85 later uses the unclamped pool count 0).

**Site 83 - `deep_gemm::sched::sm90_paged_mqa_logits_metadata<32, 256, 132, false>`** grid [1], block 32, smem 128.
- Op: `deep_gemm.get_paged_mqa_logits_metadata(context_lens, block_kv=64, num_sms=132)` (call at `dsa_indexer_kpool.py:792-794`; kernel `SP/deep_gemm/include/deep_gemm/scheduler/sm90_paged_mqa_logits.cuh:9-16`).
- Semantics: per q row, `num_segs = ceil(context_len / SPLIT_KV)` with SPLIT_KV=256 (= BLOCK_KV 64 x 4 math warpgroups); warp-wide prefix sum over the aligned batch (32); then writes `schedule_metadata [133, 2]` int32: per-SM (q_idx, seg_idx) work assignment, reversed allocation so empty SMs point at atom 0 (`sm90_paged_mqa_logits.cuh:70-90`).
- Template: kAlignedBatchSize=32, SPLIT_KV=256, kNumSMs=132, kIsVarlen=false.
- Scalars: p0 = batch_size = 1 (S), p1 = next_n = 1, p2 = is_context_lens_2d = true (0x01), p3 = context_lens ptr (the clamped tensor from site 82), p4 = indices = null, p5 = schedule_metadata out ptr.
- For this row: ceil(1/256) = 1 segment, total = 1 work item. Reversed allocation (q = 1/132 = 0, r = 1, pivot = 131) puts the single (q row 0, seg 0) item on SM 131; all other SMs land on atom 0 and exit immediately (`sm90_paged_mqa_logits.cuh:71-90`).

### 5.11 Paged MQA logits (indexer scores): site 84

**Site 84 - `deep_gemm::sm90_fp8_paged_mqa_logits<1, 32, 128, 64, true, false, 3, 3, 256, 128, 512, float>`** grid [132], block 640, smem 131140.
- Op: `deep_gemm.fp8_paged_mqa_logits(q_fp8.unsqueeze(1), kv_cache_fp8, weights, pool_context_lens, pool_block_tables, pool_schedule_metadata, pool_max_seq_len=64, clean_logits=False)` (call at `dsa_indexer_kpool.py:988-997`; kernel `SP/deep_gemm/include/deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh:29-39`).
- Computes, for each q row and each pooled KV position t: `logits[q, t] = kv_scale[t] * sum_h weights[q, h] * relu(q_fp8[q, h, :] . k_fp8[t, :])`, h = 0..31 (`relu` then head-weight then head-sum: `fmaxf(shifted_accum[j], 0) * weights[...]` at `sm90_fp8_paged_mqa_logits.cuh:300-306`).
- Template args (`sm90_fp8_paged_mqa_logits.cuh:20-24`): kNextN=1, kNumHeads=32, kHeadDim=128, BLOCK_KV=64, kIsContextLens2D=true, kIsVarlen=false, kNumQStages=3, kNumKVStages=3, SPLIT_KV=256, kNumTMAThreads=128, kNumMathThreads=512, logits_dtype=float. Block = 128 + 512 = 640. Grid = 132 (one CTA per SM; persistent scheduler loop).
- The template args `Lj1ELj32ELj128ELj64` in the mangled name fix exactly: 1 query token per row (decode), 32 indexer heads, 128-dim index head, 64 KV slots per page (= the kpool page size).
- Recorded scalars (kernel signature `sm90_fp8_paged_mqa_logits.cuh:29-39`): p0 = batch_size = 1, p1 = logits_stride = 256, p2 = block_table_stride = 1, p3 = context_lens ptr, p4 = logits out ptr, p5 = block_table ptr, p6 = indices = null, p7 = schedule_meta ptr, p8-p11 = 128-byte TMA descriptors (tensor_map_q, tensor_map_kv, tensor_map_kv_scales, tensor_map_weights).
- p1/p2 decode (this capture): the DeepGEMM host wrapper allocates `logits [batch, aligned_max_context_len]` and slices it to `[batch, max_context_len]`, keeping the aligned row stride (`attention.hpp:517-524`). Here `max_context_len = pool_max_seq_len = pool_block_tables.shape[1] * 64 = 1 * 64 = 64` (`dsa_indexer_kpool.py:953`) and `aligned = align(align(64, 256), 256) = 256` (split_kv = 256, 1024-byte alignment = 256 fp32). So p1 = 256 and the logits tensor is [1, 64] valid with row stride 256. p2 = pool_block_tables.stride(0) = 1 because the pooled table has shape [1, 1]: one 64-slot page covers the whole pooled history. Both strides multiply a row index that is always 0 at batch 1, so the small values are benign.
- Inputs:
  - q: fp8 e4m3 [S, 1, 32, 128] + per-head-row scales folded into `weights` (from sites 74/81). TMA descriptor p8 (tensor_map_q): base 0x7eff99a0a600, global dims (128, 32) = (head_dim, heads) decoded from the descriptor.
  - kv_cache: the uint8 kpool cache viewed as `[num_pages, 64, 1, 132]` (132 = 128 fp8 bytes + 4 scale bytes per slot; view at `dsa_indexer_kpool.py:925-933`). The DeepGEMM host splits it into two from_blob tensors (`attention.hpp:455-475`): kv fp8 [14666, 64, 128] strides (8448, 128, 1) and kv scales fp32 [14666, 64] strides (2112, 1), base + 8192. TMA descriptor p9 (tensor_map_kv): base 0x7f020c000000, global dims (128, 64, 14666) = (head_dim, slots, num_pages). TMA descriptor p10 (tensor_map_kv_scales): base = kv base + 8192, dims (64, 14666). The descriptors confirm the page layout of section 4.2 (8192 B keys + 256 B scales per 8448 B page) and that the pool holds 14666 pages.
  - weights: fp32 [S, 1, 32] (site 81 output) (tensor_map_weights p11).
  - context_lens: [S, 1] int32 = 1 (pool count 0, clamped to 1 at site 82). block_table: pooled page table [S, 1] int32 (`build_pooled_page_table_64`, `kpool_fp8_index.py:56-72`). schedule_meta: site 83 output.
- Output: fp32 logits [S, 64] (row stride 256). The kernel writes only column 0 (context_len = 1): the dot of q with pool slot 0 of the page. Pool slot 0 was never written by site 77 in this capture (no pool ever closed), so the value is a garbage dot product. It is never read: site 85 takes its identity path with length = 0 and does not touch the logits (section 5.12).

### 5.12 kpool topk transform: site 85

**Site 85 - `sglang::(anon)::kpool_topk_transform_kernel<512>`** grid [S], block 1024, smem 32768.
- Op: `topk_from_pooled_history_logits` -> `fast_kpool_topk_transform_fused` (`kpool_fp8_index.py:585-645`; wrapper `SP/sglang/kernels/ops/moe/kpool_topk_transform.py:29-71`; kernel `SP/sglang/kernels/jit/csrc/dsa/kpool_topk_transform.cuh:220-290`).
- Template K = 512 = pool-level topk = `index_topk / index_kpool = 2048 / 4` (`history_group_budget_for_topk`, `kpool_fp8_index.py:407-409`).
- Semantics per row (`kpool_topk_transform.cuh:236-290`):
  1. If `length <= K` skip selection and take all pools in order (identity path, `kpool_topk_transform.cuh:243-261`). Else radix-select the top-512 pool ids from the fp32 logits (32-bit, 4-round byte-wise radix on the fp32 bit pattern, `fast_topk_cuda_tl_impl`, `kpool_topk_transform.cuh:55-204`).
  2. Expand each selected pool g to its 4 token positions `4g, 4g+1, 4g+2, 4g+3` (pool x 4 expansion, `kpool_topk_transform.cuh:276-284`).
  3. Append the open tail: `seq_len % 4` raw token positions after the history (`kpool_topk_transform.cuh:285-288`; `append_tail` because `seq_lens` is passed).
  4. Map every raw token position through the page table: `dst[col] = page_table_entry[raw_token]` (`transform_kpool_token`, `kpool_topk_transform.cuh:206-216`). The page table here is `page_table_1` = the token-level MLA KV slot table (page size 1 view) from `_kpool_fused_topk_mapping` (`dsa_indexer_kpool.py:806-824`). So the output holds **physical MLA KV slots**, not logical positions.
  5. Pad remaining columns with -1.
- Inputs: score = site 84 logits [1, 64] fp32 (recorded input_stride = 256 = the aligned row stride of the sliced logits); lengths = pool_seqlens [1] = 0; seq_lens [1] = 3; page_table = page_table_1 [1, 3] int32 (one MLA KV slot per token position).
- Scalars (kernel signature `kpool_topk_transform.cuh:220-234`): p0 = 40-byte FastTopKParams blob {input = logits ptr, row_starts = null, indices = null, lengths = pool_seqlens ptr, input_stride = 256} (struct at `kpool_topk_transform.cuh:37-43`); p1 = dst ptr; p2 = dst_stride = 2051; p3 = pool_size = 4; p4 = token_topk = 2048; p5 = out_cols = 2051; p6 = page_table ptr; p7 = page_table_stride = 3; p8 = page_table_row_index = null; p9 = topk_indices_offset = null; p10 = seq_lens ptr.
- p7 = 3 is the stride of the contiguous [1, 3] page_table_1. It is the cleanest single proof that the true sequence length of this capture is 3 (section 1).
- Output: `dst_token_indices` int32 [S, 2051] = topk_indices. Grid = [S] (grid_b2 = [2]).
- With lengths = 0 <= K = 512 the kernel takes the identity path: history_len = 0, tail_count = seq % 4 = 3, so it writes dst[0..2] = page_table_1[0..2] (the MLA KV slots of the 3 tokens, in order) and dst[3..2050] = -1. The site-84 logits are never read. In the general case (pool count > 512) the radix path selects the top-512 pools, so the valid prefix holds 2048 pool-expanded slots plus the tail; the valid prefix length is always dsa_cache_seqlens.

### 5.13 Absorbed q bmm (W_kc): site 86

**Site 86 - `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT`** grid [4,16,1], block 384, smem 164308.
- Op: absorbed query: `q_nope_out = torch.bmm(q_nope.transpose(0,1), self.w_kc)` (`forward_mla.py:568`). q_nope [T, 8, 256] -> transposed [8, T, 256]; w_kc [8, 256, 512] bf16; output [8, T, 512] -> [T, 8, 512] bf16.
- This is the MLA "absorb": q is multiplied by the K-half of kv_b once, so attention can run against the 512-dim latent directly. There is **no runtime kv_b GEMM that materializes per-head K/V**; see section 8.1.
- Same cuBLAS kernel as site 69 (bf16 TN); total output elements 4096 -> same heuristic grid [4,16].

### 5.14 MLA KV cache write: site 87

**Site 87 - `set_mla_kv_buffer_kernel_norope` (Triton)** grid [T, 1], block 128.
- Op: write this token's latent c_kv (512 bf16, site 66 output) into the MLA pool at slot `loc` (kernel `mla_buffer.py:87-111`; dispatch `mla_buffer.py:150-176`). `kv_buffer[loc, 0, 0:512] = k_nope`.
- Params: p0 = kv_buffer ptr (layer 3 buffer; base alloc 0x7f017e000000, 962592768 bytes = the MLA pool), p1 = k_nope ptr, p2 = loc ptr ([T]). The other recorded slots p3/p4 are null Triton hidden scratch pointers. buffer_stride = kv_buffer.stride(0) = 512, nope_stride = 512, nope_dim = 512, BLOCK = 512 are `tl.constexpr`, so they are baked into the JIT binary and do not appear as launch args (`mla_buffer.py:88-96`, call at `mla_buffer.py:166-176`).
- rope part is absent (qk_rope_head_dim = 0), hence the `_norope` variant.
- Grid = (n_loc, 1) = [T, 1] (grid_b2 = [2, 1]).

### 5.15 Topk clamp + fa3 scheduler: sites 88-89

**Site 88 - `at::native::vectorized_elementwise_kernel<4, ... clamp ...>`** grid [3], block 128.
- Op: `page_table = page_table.clamp(min=0)` on the topk_indices [S, 2051] int32 because `dsa_index_kpool > 1` (`dsa_backend.py:2453-2454`). Replaces the -1 padding with slot 0 (the reserved/dummy slot, so invalid lanes read a defined value and are masked by `cache_seqlens`).
- Scalar p0 = numel = 2051 (0x803). Grid = cdiv(numel, 128*4) rounded: [3] for 2051, [5] for 4102 (grid_b2).

**Site 89 - `flash::prepare_varlen_num_blocks_kernel<1, true>`** grid [1], block 32.
- Op: fa3 varlen scheduler prep (`flash_prepare_scheduler.cu:43-...`; host wrapper `prepare_varlen_num_blocks`, `flash_prepare_scheduler.cu:224-258`).
- Full scalar decode (kernel signature order, `flash_prepare_scheduler.cu:43-60`):
  - p0 seqlen_q_static = 1; p1 seqlen_k_static = 2051 (0x803); p2 seqlen_k_new_static = 0.
  - p3 = cu_seqlens_q ptr ([0, 1]); p4 = cu_seqlens_k = null; p5 = cu_seqlens_k_new = null; p6 = seqused_q = null; p7 = seqused_k ptr (= dsa_cache_seqlens, [3]); p8 = leftpad_k = null.
  - p9 num_batch = 1; p10 num_head = 1 (kv heads; q heads are packed); p11 qhead_per_khead = 8; p12 num_sm = 132; p13 num_splits_static = 29 (0x1d).
  - p14 = blockm_divmod: FastDivmod {divisor=64, multiplier=0x80000000, shift=5} (kBlockM = 64); p15 = blockn_divmod: FastDivmod {divisor=64, ...} (kBlockN = 64).
  - p16 = tile_count_semaphore out; p17 = num_m_blocks_ptr out; p18 = num_splits_dynamic_ptr out; p19 = varlen_batch_idx_ptr out; p20 = num_nheads_in_l2_ptr = null.
  - p21 enable_pdl = true (0x01); p22 is_causal = false (0x00); p23 packgqa = true (0x01); p24 max_kvblocks_in_l2 = 113 (0x71), computed host-side as `size_l2 / size_one_kvblock` (`flash_prepare_scheduler.cu:226-229`; with the fork's constants the recorded value is 113).
- Semantics (`flash_prepare_scheduler.cu:75-164`): per batch it reads the DYNAMIC lengths from seqused_q/seqused_k when those pointers are non-null; the static lengths (p0/p1) are fallbacks. Here seqused_k = dsa_cache_seqlens = [3], so: num_m_blocks = ceil(1*8 / 64) = 1 (packgqa multiplies seqlen_q by qhead_per_khead = 8), num_n_blocks = ceil(3/64) = 1, total_blocks = 1, blocks_per_sm = ceil(1 * 1.1 * 1 / 132) = 1, num_splits_dynamic = max(min(ceil(1/1), 29), 1) = 1. So the fa3 kernel executes ONE split for this capture. With Sort=true it sorts batches by work (trivial for S=1) and writes the four metadata buffers (`flash_prepare_scheduler.cu:166-213`).
- p1 seqlen_k_static = 2051 is NOT the sequence length; it is the page-table width (the constant 2051-wide topk output row). The static cap p13 = 29 comes from the host: `get_num_splits` -> `num_splits_heuristic(total_mblocks=1, num_SMs=132, num_n_blocks=ceil(2051/64)=33, num_m_blocks=1, size_one_kv_head, causal=false, 128)` (`flash_api.cpp:426-462`, `heuristics.h:33-63`): the efficiency loop picks the smallest split count with eff >= 85% of max, which is 29. sglang passes num_splits = 0 (auto) (`dsa_backend.py:331-334`). The device kernel clamps this cap to the dynamic value 1.

### 5.16 FlashAttnFwdSm90 sparse decode: site 90

**Site 90 - `cutlass::device_kernel<flash::FlashAttnFwdSm90<CollectiveMainloopFwdSm90<2, (1,1,1), (64,64,64), 512, bf16, float, Sm90, F,F,F, T,T,F, T,T, F,F, T,T, F, bf16, F, 1>, CollectiveEpilogueFwd<(64,512,64), (1,1,1), bf16, Sm90, 256, T,T,T, F>, VarlenDynamicPersistentTileScheduler<64, 64, 256, 128, T,T,T, F, T,T>>>`** grid [132], block 384, smem 232448, params 2944 bytes.
- Op: the sparse attention itself. Called from `_forward_fa3` (`dsa_backend.py:2428-2474`) via `flash_attn_with_kvcache` (`SP/sgl_kernel/flash_attn.py:36-271` -> `torch.ops.sgl_kernel.fwd`, flash_attn.py:230; C++ `mha_fwd`, flash_api.cpp).
- sglang call args (`dsa_backend.py:2455-2472`): `q=None` (only_qv), `k_cache=None`, `v_cache = kv_buffer viewed [num_slots, 1, 1, 512]`, `qv = q_nope_out [T, 8, 512]`, `only_qv=True`, `page_table = topk_indices [S, 2051] int32` (post-clamp), `cache_seqlens = [3]` (= dsa_cache_seqlens), `cu_seqlens_q = [0, 1]`, `cu_seqlens_k_new = None`, `max_seqlen_q = 1`, `softmax_scale = layer.scaling = 256^-0.5 = 0.0625`, `causal=True` (converted to is_causal=false because max_seqlen_q == 1, flash_api.cpp:569-573), `softcap = logit_cap = 0.0`, `num_splits = 0` (auto -> static cap 29, dynamic 1).
- Template decode (mainloop signature at `mainloop_fwd_sm90_tma_gmma_ws.hpp:31-33`):
  - Stages=2, ClusterShape=(1,1,1), TileShape_MNK=(64,64,64), kHeadDimV=512, Element=bf16, ElementAccum=float, Sm90.
  - Is_causal=false, Is_local=false, Has_softcap=false, Varlen=true, PagedKVNonTMA=true, AppendKV=false, HasQv=true, **OnlyQv=true**, MmaPV_is_RS=false, IntraWGOverlap=false, PackGQA=true, Split=true, V_colmajor=false, ElementSink=bf16, Has_sparse_mask=false, kBlockH=1.
- OnlyQv semantics: K is never loaded as a separate tensor; the score matmul is `S = QV x V^T` using the *V* cache as the key matrix (legal because for NoPE MLA, K == V == latent). See the OnlyQv branch at `mainloop_fwd_sm90_tma_gmma_ws.hpp:1330-1340` (`flash::gemm<zero_init=OnlyQv>(tiled_mma_qv, tSrQv, tSrV, tSrS)`). `qv` (the absorbed q [T, 8, 512]) plays the role of Q.
- PackGQA: the 8 query heads are packed into the M tile (8 rows of the 64-row M tile; qhead_per_khead = 8, one kv "head").
- PagedKVNonTMA: KV rows are gathered through the page table with page_size = 1 (set by `_forward_fa3`, `dsa_backend.py:2403/2472`). The page table IS the topk index buffer; each "page" is one latent slot. Gather logic in `paged_kv.h` (fa3 sources).
- Split: the split path is active with the static cap 29; the executed count is num_splits_dynamic = 1 (site 89 output, read via the scheduler params), so one fp32 partial is written to oaccum/lseaccum.
- Params blob layout (2944 B = MainloopParams + EpilogueParams + TileSchedulerParams; struct definitions: mainloop `mainloop_fwd_sm90_tma_gmma_ws.hpp:447-507`, epilogue `epilogue_fwd.hpp`, scheduler `tile_scheduler.hpp:506-522`). Key decoded fields (offsets in bytes):
  - 0: ptr_Q = 0x7eff99a07a00 (placeholder 64-dim q tensor; sglang passes a dummy because q=None with only_qv, flash_attn.py:168-202).
  - 8..: shape_Q ints (1, 64, 8, 1) + strides; the placeholder q is [1, 8, 64].
  - ~104: ptr_K/ptr_V pair: ptr_K is unused (only_qv), ptr_V = the MLA pool base 0x7f017e000000 (same allocation as site 87's kv_buffer; also the base of the tma_load_V descriptor).
  - 272: ptr_Qv = 0x7eff99a0a600 (the absorbed q_nope_out [1, 8, 512] bf16).
  - 368: seqlen_k = 2051 (0x803) = the static page-table width (constant topk width), used only where a static length is required; the per-batch true length (3) arrives via seqused_k.
  - 416..432: page_table shape ints (1, 2051) + stride.
  - 512, 768, 1024, 1280, 1536, 1792: six 128-byte CUtensorMap (TMA) descriptors: tma_load_Q (placeholder), tma_load_K, tma_load_V (base 0x7f017e000000 = MLA pool), tma_load_K_new, tma_load_V_new, tma_load_Qv (base 0x7eff99a0a600). With PagedKVNonTMA=true the KV descriptors exist but the KV loads use cp.async gather instead (`mainloop_fwd_sm90_tma_gmma_ws.hpp:60-63` shows Use_TMA_KV=false).
  - 2056: softmax_scale_log2 = 0x3db8aa3b = 0.09017 = 0.0625 * log2(e) (the kernel works in base-2: `mainloop_fwd_sm90_tma_gmma_ws.hpp:487`).
  - 2160/2560/2768: num_splits = 29 (0x1d) static cap in the mainloop/epilogue/scheduler params; the scheduler overrides it with num_splits_dynamic (= 1) from site 89.
  - 2304: epilogue ptr_O partial region (oaccum); 2432: ptr 0x7eff99a1f000; 2512: ptr_O (final out [1, 8, 512]); 2688: cu_seqlens_q ptr = 0x7f026dfef800 (same pointer as site 89 p3); 2784..2824: scheduler params: tile_count_semaphore, cu_seqlens, seqused, num_splits_dynamic_ptr, num_m_blocks_ptr, varlen_batch_idx_ptr (0x7eff99a08000/10/20/30 region).
  - FastDivmod fields inside scheduler params: head_divmod (divisor = num_head = 1), nsplits_divmod (divisor = 29 = the static cap; the dynamic count 1 comes from num_splits_dynamic_ptr); blockM/blockN divmods live only in site 89 (they are *not* in the mainloop params; the mainloop uses compile-time 64/64 tiles).
  - Uninitialized tail/padding bytes contain host-stack garbage (0x7ffd... values); they are never read by the kernel (e.g. unused AppendKV TMA descriptors when AppendKV=false).
- Grid = 132 persistent CTAs (VarlenDynamicPersistentTileScheduler over all SMs, `tile_scheduler.hpp:493-522`), block 384 = 1 load warpgroup (128) + 2 MMA warpgroups (256).

### 5.17 FlashAttnFwdCombine: site 91

**Site 91 - `cutlass::device_kernel<flash::FlashAttnFwdCombine<(8,128), 5, 256, 1, false, true, bf16, float, Sm90>>`** grid [1,4,1], block 256, smem 17536, params 232 bytes.
- Op: merge the split-KV partials into the final output (`flash_fwd_combine_kernel.h:24-26` template, `:134-170` Arguments/Params; launcher `run_mha_fwd_combine`, flash_api.cpp:371-395, called at flash_api.cpp:1227). The merge count is read from num_splits_dynamic_ptr (= 1 for this capture); the partial buffers are sized for the static cap 29.
- Template: TileShape_MK = (8, 128), kLogMaxSplits = 5 (max 32 splits >= 29), kNThreads = 256, AlignmentLSE = 1, Is_even_K = false, Varlen = true, Element = bf16, ElementPartial = float.
- Inputs: oaccum fp32 [29, 8, 1, 512] (contiguous; split stride 4096, head stride 512, row stride 512, elem stride 1), lseaccum fp32 [29, 8, 1] (flash_api.cpp:1099-1100). Output `out` bf16 [1, 8, 512].
- Semantics: per (m-tile of 8 q/head rows, 128-wide dv tile): read the per-split LSE, compute the softmax renormalization weights across the dynamic splits (log-sum-exp merge; 1 split here, so the merge is a copy), accumulate the weighted fp32 partials, cast to bf16 (`flash_fwd_combine_kernel.h`, compute path ~line 200+). Also produces the final softmax_lse when requested (not used here).
- Grid: [ceil(8 packed q-rows / 8) = 1 per sequence, 512/128 = 4, S = 1] (grid_b2 = [1, 4, 2]).
- 232-byte params decode (Params struct, `flash_fwd_combine_kernel.h:153-170`): ptr_O_partial = 0x7eff99ab6400; shape ints (1, 512, 29) = (total_q, dv, splits = static allocation cap) with head mode folded into the m-tile walk; strides consistent with (split 4096, head 512, dv 1) elements; ptr_LSE_partial; ptr_O = 0x7eff99a0c600; ptr_LSE; then two FastDivmods: seqlen_divmod {divisor=1, multiplier=0x80000000, shift=0} (= seqlen_q per batch = 1) and head_divmod {divisor=8, ...} (= num heads); then cu_seqlens = 0x7f026dfef800, seqused = null, num_splits_dynamic_ptr = 0x7eff99a01e00 (= 1, the actual merge count), varlen_batch_idx_ptr, semaphore_to_reset. Raw hex is preserved in `$GLM53_ARTIFACTS/lifts/FlashAttnFwdCombine.json`.
- So the answer to "which fields are FastDivmod(seqlen) vs constants": in the combine kernel, seqlen_divmod and head_divmod are runtime FastDivmods (divisors 1 and 8 here); in site 89, blockm/blockn divmods are compile-time-constant FastDivmods (divisor 64); in the mainloop params, page_size_divmod (=1), blockN_per_page_size_divmod (=64) and qhead_per_khead_divmod (=8) are runtime FastDivmods built from the call args, while tile M/N/K (64/64/64) and dv (512) are compile-time constants baked into the template.

### 5.18 Output bmm (W_vc) + o_proj: sites 92-94

**Site 92 - `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT`** grid [4,8,1], block 384, smem 164308.
- Op: `attn_bmm_output = torch.bmm(attn_output.transpose(0,1), self.w_vc)` (`forward_mla.py:890-905`): [8, T, 512] x [8, 512, 256] -> [8, T, 256] -> flattened [T, 2048] bf16.
- This is the projection from the 512-dim attention output latent back to the 256-dim per-head value space (V-half of kv_b). Weight: `w_vc` [8, 512, 256] bf16 (from kv_b_proj, section 3).
- Output elements 2048 (half of site 86's 4096) -> same tile count per column, half the column tiles: grid [4,8].

**Site 93 - `per_token_group_quant_flat_kernel<...128, false, false, true, false>`** grid [T], block 256.
- Op: quantize [T, 2048] bf16 -> fp8 + scales [16, T] for the o_proj GEMM.

**Site 94 - `sm90_fp8_gemm_1d2d_impl<MajorK, 0,0,0, 1, 16, 32, 128, 128, 64, 16, 128, 128, 1, false, 132, GemmType::Normal, bf16, EpilogueIdentity>`** grid [132], block 256, smem 101696.
- Op: `out [T, 4096] bf16 = act_fp8 [T, 2048] @ o_proj.weight^T [4096, 2048]` (row-parallel; per-rank K = 2048).
- Template (from the mangled name): BLOCK_M = 16, BLOCK_N = 32 (vs 16 for sites 64/68), BLOCK_K = 128, kSwizzleAMode = 128, kSwizzleBMode = 128, kSwizzleDMode = 64, kNumStages = 16, kNumTMAThreads = 128, kNumMathThreads = 128, kNumTMAMulticast = 1, kIsTMAMulticastOnA = false, kNumSMs = 132. Scalars: M = 1, N = 4096, K = 2048 (recorded p2/p3/p4 = 1, 0x1000, 0x800).

**Site 95 - `sglang::all_reduce_1shot_push_kernel<...>`** grid [132], block 128.
- Op: TP8 one-shot all-reduce of the attention output [T, 4096] bf16 (custom sglang allreduce; o_proj has `reduce_results=False`, reduction happens here inside `prepare_mlp`'s communicate step).

### 5.19 Post-attention / pre-FFN hyper-connection: sites 96-97

**Site 96 - `mhc_fused_post_pre_fma_tilelang_kernel`** grid [T,12,8], block 256, smem 96.
- Op: fused `hc_ffn_post_pre` (`glm5_next.py:856-895`, via `apply_mhc_post_pre_boundary` imported from `deepseek_common/amd/deepseek_v4_fused_mhc`, `glm5_next.py:82-84`): mixes the attention output into the 4-stream residual using `h_post`/`h_res` from site 62 (the hc_post part), then applies the FFN-side mixing projection with `hc_ffn_fn` [24, 16384] (the pre part), producing the next residual streams, new h_post/h_res, and the sqr_sum for the FFN RMSNorm. Scalars p8 = 1 (T), p9 = 8 (0x08: stream-related constant); 8 weight/activation pointers.
- Enabled because `is_cross_layer_mhc_fusion_enabled()` (`glm5_next.py:805-811`); on decline it would fall back to separate hc_post + hc_pre (`communicator_mhc.py:99-127`).

**Site 97 - `mhc_pre_big_fuse_with_norm_tilelang_kernel`** grid [T], block 96, smem 37232.
- Op: same kernel as site 62, now for the FFN side: sinkhorn on h_res, combine streams, RMSNorm with `post_attention_layernorm.weight` (eps 1e-5) -> FFN input hidden [T, 4096] bf16 (`communicator_mhc.py:118-125`).

### 5.20 MoE: sites 98-107

**Site 98 - `sglang::tiny_n_gemm_kernel<GEMMTraitN<288, 4096, 3, 32>, 1, float, true>`** grid [96], block 256.
- Op: MoE router GEMM: `router_logits [T, 288] fp32 = hidden [T, 4096] @ mlp.gate.weight^T [288, 4096]` (fp32 weight, `moe_router_dtype = float32`). Trait: N=288, K=4096, 3 outputs per block x 32 lanes... grid = 288/3 = 96, independent of T.

**Site 99 - `_router_triton_kernel`** grid [S], block 32, smem 64.
- Op: sigmoid scoring + noaux_tc expert selection: top-8 of 288 routed experts + fused shared expert (id 288) -> topk_ids/topk_weights [T, 9], weights scaled by `routed_scaling_factor` 2.5 and normalized (`norm_topk_prob`).
- Scalars: p5 = 0x40200000 = 2.5f (scaling), p9 = 288 (0x120; n_routed), p10 = 9, p11 = 9 (topk incl. shared), p12/p13 = 0. p0-p4 = router_logits, gate weight? no: pointers to logits, topk_ids, topk_weights, etc.

**Site 100 - `at::native::CatArrayBatchedCopy_alignedK_contig<...>`** grid [1,2,1], block 128.
- Op: torch.cat of the per-token hidden into the MoE input workspace (batched copy of 2 tensors, 4-byte elements; part of the fused MoE input preparation).

**Site 101 - `_moe_align_small_numel_kernel`** grid [1], block 128, smem 64.
- Op: build expert-aligned token index arrays for the small-numel fused MoE path (moe_align_block_size): sorted_token_ids, expert_ids, num_tokens_post_pad. Scalars: p4 = 290 (0x122; padded expert count 289+1), p5 = 64 (block pad multiple), p6 = 9 (topk).

**Site 102 - `per_token_group_quant_flat_kernel<QuantTrait<bf16, fp8_e4m3, 128, false, true, true, false>, true>`** grid [1], block 256.
- Op: quantize the MoE input [T*9?, 4096]... precisely: the aligned MoE input rows (up to 9*T rows) bf16 -> fp8, this time with **row-major** scale layout (kRowMajor=true) as the sglang fused_moe_kernel consumes per-token-group scales in row-major. Grid scales with padded row count (grid_b2 = [2]).

**Site 103 - `fused_moe_kernel`** grid [36], block 128, smem 73728, 24 params.
- Op: first fused MoE GEMM (w13 = gate_up): for each of the 9 token-expert pairs, `a1 [rows, 512] = act_fp8 [rows, 4096] @ w13[e]^T` with fp8 block scales; N = 512 (2 x 256 per-rank gate/up), K = 4096, expert weight stride = 2097152 elements (2*256*4096). Scalars p9 = 512 (N), p10 = 4096 (K), p11 = 576 (EM = 9*64 padded), p12 = 9 (topk), p13/p15 = 4096 (strides), p14 = 2097152 (0x200000, expert stride), p18 = 512 (0x200), p19-p21 = block sizes (32, 128, 32 as 0x20/0x80/0x20).
- Grid: 36 = (EM/BLOCK_M) x (N/BLOCK_N) with BLOCK_M=16, BLOCK_N=512? Recorded: 36 for T=1, 72 for T=2 -> scales with T (EM = 576*T... for T=2, EM = 1152, grid 72).

**Site 104 - `sglang::silu_mul_clamp_kernel<bf16, true>`** grid [9*T], block 32.
- Op: `SwiGLU` with clamp: `silu(gate).clamp(max=10) * up` on [9T, 512] -> [9T, 256] bf16 (`swiglu_limit = 10.0`; `swiglu_clamped`, `glm5_next.py:144-148`). Params blob = SiluAndMulClampParams (ptrs, rows, cols, limit). Grid = 9*T (grid_b2 = [18]).

**Site 105 - `per_token_group_quant_flat_kernel<...row-major...>`** grid [1], block 256.
- Op: quantize the SwiGLU output [9T, 256]... note K = 256 per rank (moe_intermediate 2048/8) for the down projection.

**Site 106 - `fused_moe_kernel`** grid [288], block 128, smem 73728, 24 params.
- Op: second fused MoE GEMM (w2 = down): `a2 [rows, 4096] = a1_fp8 [rows, 256] @ w2[e]^T`; N = 4096, K = 256, expert stride 1048576 (256*4096). Scalars p9 = 4096 (0x1000), p10 = 256 (0x100), p11 = 576, p12 = 9, p14 = 1048576 (0x100000), p18 = 4096, p19-p21 = (32, 64, 32) (0x20/0x40/0x20).
- Grid: 288 = 36 x 8 for T=1 (grid_b2 = [576]).

**Site 107 - `_moe_sum_reduce_kernel`** grid [S*T, 2], block 512.
- Op: weighted sum of the 9 expert outputs: `out [T, 4096] = sum_i topk_weight[i] * a2[i]` (scalars p1 = 36864 = 9*4096 (input numel), p2 = 4096, p4 = 4096, p5 = 1 (T), p6 = 9 (topk), p7 = 4096). Grid [T, 2] (2 = 4096/2048 elements per CTA split; grid_b2 = [2, 2]).

**Site 108 - `all_reduce_1shot_push_kernel`** grid [132], block 128.
- Op: TP8 all-reduce of the MoE output [T, 4096] bf16.

### 5.21 FFN hyper-connection post: site 109

**Site 109 - `mhc_post_tilelang_kernel`** grid [T], block 128, smem 28672.
- Op: `hc_post` (tilelang def `mhc_post_tilelang` at `mhc.py:1305`, public wrapper `hc_post` at `mhc.py:2076`; call `communicator_mhc.py:116/126`): mix the FFN output back into the 4-stream residual: `residual'[i] = sum_j h_res[i,j] * residual[j] + h_post[i] * ffn_out`, producing [T, 16384] bf16 for the next layer. Params p0-p4 = ffn_out, residual, h_post, h_res, output; p5 = T.

## 6. FlashAttnFwdSm90 (fa3) deep dive

- sglang wrapper: `_forward_fa3` at `SP/sglang/srt/layers/attention/dsa_backend.py:2428-2474`; dispatch at `dsa_backend.py:2389-2403` (decode) for `dsa_decode_impl == "fa3"` (the active backend for this capture; option list at `dsa_backend.py:288-292`).
- Python binding: `SP/sgl_kernel/flash_attn.py:36-271` (`flash_attn_with_kvcache`); native entry `torch.ops.sgl_kernel.fwd` (flash_attn.py:230); C++ `mha_fwd` (flash_api.cpp; `set_params_fprop` at flash_api.cpp:30).
- Params struct (`flash.h:37-...`, Flash_fwd_params; base Qkv_params at `flash.h:14-33`): the fields that matter for this call:
  - q/k/v ptrs: q = placeholder, k = null (only_qv), v = MLA pool base; qv = absorbed q_nope_out.
  - dims: b = 1 (varlen batches), seqlen_q = 1, seqlen_k = 2051 (the static page-table width; the per-batch true length 3 arrives via seqused_k), d = 64 on the placeholder q only - for only_qv the kernel uses kHeadDimV = 512 for both matmuls; the placeholder q exists only to satisfy the API (flash_attn.py:179-196).
  - page_table = topk_indices, page_table_batch_stride = 2051, page_size = 1, num_pages = pool slots.
  - cu_seqlens_q = [0,1], seqused_k = [2051], is_causal = false (converted, flash_api.cpp:569-573), softcap = 0.
  - num_splits = 29 static cap (host heuristic from the 2051-wide page table, flash_api.cpp:601), pack_gqa = true (flash_api.cpp:603), tile scheduler metadata pointers from site 89 (which carry the dynamic split count 1), num_sm = 132.
  - use_sparse_mask = false, sparse_mask_fine = null (flash_attn.py:266): the sparsity enters exclusively through the paged page table (topk indices), NOT through the fa3 sparse-mask bitmap.
- The kernel (site 90) computes standard flash attention over the seqused_k = 3 gathered latent rows: S = softmax(qv . V^T * 0.0625) then O = P . V, in 1 KV split with dynamic persistent scheduling; Is_causal=false because every selected row is by construction <= the current position.
- `prepare_varlen_num_blocks` (site 89) outputs: tile_count_semaphore (reset to 0), num_m_blocks[batch] (=1), num_splits_dynamic[batch] (=1), varlen_batch_idx (identity here). These drive the persistent scheduler's work partition (`tile_scheduler.hpp:506-522`).
- Combine (site 91): see 5.17.

## 7. DeepGEMM paged MQA logits deep dive

- `sm90_paged_mqa_logits_metadata` (site 83): one warp. For each q row computes num_segs = ceil(context_len/256) (SPLIT_KV = 256 slots = 4 pages of 64), prefix-sums over the batch, then assigns each of the 132 SMs a contiguous run of (q_idx, seg) work items, written as `schedule_metadata[sm_idx] = {q_atom_idx, kv_seg_idx}` for sm_idx in 0..132 (buffer shape [133, 2], `scheduler/sm90_paged_mqa_logits.cuh:45-90`). The metadata kernel uses `cudaGridDependencySynchronize` (PDL) so it chains after site 82.
- `sm90_fp8_paged_mqa_logits` (site 84): persistent kernel, one CTA per SM. Each CTA walks its assigned segments. Per segment of 256 slots: 4 math warpgroups each take a 64-slot page (BLOCK_KV=64), TMA-loads q (32x128 fp8), the page's 64x128 fp8 keys and 64 fp32 scales, runs FP8 WGMMA `[32,128] x [64,128]^T -> [32,64]`... more precisely `[kNextN*kNumHeads, kHeadDim] @ [BLOCK_KV, kHeadDim]^T` (`impls/sm90_fp8_paged_mqa_logits.cuh:262-285`), then per output element applies `relu(acc) * weight[h]`, reduces over the 32 heads, multiplies by the per-slot kv scale, and writes 2 fp32 logits per thread (`impls/sm90_fp8_paged_mqa_logits.cuh:287-330`).
- Scores outside `context_lens` are left unwritten (`clean_logits=False`), which is safe because the topk kernel only reads the first `lengths` columns.
- Template args recap: `<kNextN=1, kNumHeads=32, kHeadDim=128, BLOCK_KV=64, kIsContextLens2D=true, kIsVarlen=false, kNumQStages=3, kNumKVStages=3, SPLIT_KV=256, kNumTMAThreads=128, kNumMathThreads=512, float>`. `Lj1ELj32ELj128ELj64` = (next_n, heads, head_dim, page slots) as recorded in the mangled name.

## 8. Answers to the specific questions

### 8.1 Absorbed MLA or materialized per-head K/V?

Absorbed. Evidence:
- The attention consumes the 512-dim latent pool directly (site 87 writes only 512 dims; site 90's v_cache is the latent pool; v_head_dim = 512 at `dsa_backend.py:2443-2452`).
- The only kv_b-related GEMMs at runtime are two bmm's against the *absorbed* weights: site 86 (`bmm(q_nope^T, w_kc)`, `forward_mla.py:568`) and site 92 (`bmm(attn_out^T, w_vc)`, `forward_mla.py:890-905`).
- No kernel in the block computes kv = kv_b(kv_latent) ([T, 8, 512] per-head K and V do not exist). The nvjet [4,16] at site 69 is the *indexer* wq_b (bf16, 1536 -> 4096), not kv_b: it runs before the indexer k/k_norm/hadamard chain (sites 70-74) and its input is the q_a latent.
- The fa3 kernel runs with OnlyQv=true (template bool 8 of the mainloop instantiation), i.e. it never loads a K tensor at all.

### 8.2 What is the nvjet [4,8] after FlashAttnFwdCombine (site 92)?

The W_vc bmm: attention output latent [8, T, 512] x w_vc [8, 512, 256] -> [T, 2048], feeding the o_proj (after FP8 quantization at site 93). It is the V-side of the absorbed MLA, not a new projection.

### 8.3 Indexer q input: hidden states or q_a output?

From the q_a output (the 1536-dim q lora, post q_a_layernorm). `self.indexer(x=hidden_states, q_lora=q_lora, ...)` (`forward_mla.py:446`), `wq_b = ReplicatedLinear(q_lora_rank=1536, 32*128=4096)` (`dsa_indexer_kpool.py:141-147`), called as `self.wq_b(q_lora)` (`dsa_indexer_kpool.py:620`). Dims: 1536 -> 4096 (32 heads x 128). The hidden states feed only wk, weights_proj, and the kpool gate.

### 8.4 kpool layouts and topk transform outputs

- Pooled cache: section 4.2 (per page: 64 x 128 B FP8 + 64 x 4 B fp32 scales at byte offset 8192; 4 tokens per slot; slot p lives in token-page-table row (p//64)*4, column p%64).
- Tail cache: section 4.3 (per layer per request slot: [4, 128] bf16 k + [4, 128] bf16 gate score, ring at pos%4; raw k, unrotated).
- `kpool_topk_transform<512>` outputs int32 [S, 2051]: up to 512 selected pools x 4 tokens + up to 3 tail tokens, all mapped through the token page table (page_table_1), so they are **physical MLA KV slots** usable directly as fa3's page table with page_size=1 (`_kpool_fused_topk_mapping`, `dsa_indexer_kpool.py:806-824`; `transform_kpool_token`, `kpool_topk_transform.cuh:206-216`). The 2051 width is a constant (2048 + 3); the valid prefix length is dsa_cache_seqlens. In this capture the valid prefix is 3 entries = the whole context in token order (0 closed pools + 3 tail tokens).
- The fa3 kernel reads the index row as one shared list for all 8 local heads (the sparse set is head-independent in DSA; PackGQA packs all heads into the same M tile). Padding (-1) is clamped to slot 0 at site 88 and masked by `cache_seqlens` (= 3 here).

### 8.5 `_act_quant_kernel` and `fast_hadamard_transform` roles/order for indexer k

For q: Hadamard first (site 73, on the bf16 [T,32,128] query), then act_quant (site 74, -> FP8 + per-head-row scale). For k of *new* tokens: no Hadamard, no FP8 at projection time - the raw bf16 k goes into the tail ring (site 77). The Hadamard + FP8 quantization of k happens only when a pool of 4 closes: inside site 77 (`_hadamard128` at `kpool_fp8_index.py:880-890`, quantize + scale at `:1130-1150`), after the softmax weighted average. So the transform order differs by design: q is rotated then quantized immediately; k is averaged (linear op, commutes with the orthogonal transform only up to the bf16 round-trip), then rotated, then quantized, at pool-close time. Past pooled k in the cache is already rotated+quantized, matching the rotated+quantized q in the logits kernel.

## 9. Site table

| Site | Logical op | Kernel | Weight tensors used |
|---|---|---|---|
| 61 | HC pre-mix GEMM + sqr_sum (attn side) | deep_gemm sm90_tf32_hc_prenorm_gemm_impl<24,16384,...> | hc_attn_fn |
| 62 | HC sinkhorn + combine + input RMSNorm | mhc_pre_big_fuse_with_norm_tilelang_kernel | hc_attn_base, hc_attn_scale, input_layernorm.weight |
| 63 | act quant (hidden) | sglang per_token_group_quant_flat_kernel | - |
| 64 | fused q_a+kv_a FP8 GEMM (4096->2048) | deep_gemm sm90_fp8_gemm_1d2d_impl | fused_qkv_a_proj_with_mqa.weight(+scale_inv) = q_a_proj.weight + kv_a_proj_with_mqa.weight |
| 65 | q_a_layernorm (1536) | flashinfer RMSNormKernel | q_a_layernorm.weight |
| 66 | kv_a_layernorm (512) | flashinfer RMSNormKernel | kv_a_layernorm.weight |
| 67 | act quant (q lora) | sglang per_token_group_quant_flat_kernel | - |
| 68 | q_b FP8 GEMM (1536->2048) | deep_gemm sm90_fp8_gemm_1d2d_impl | q_b_proj.weight(+scale_inv) |
| 69 | indexer wq_b (1536->4096) bf16 | cublas nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT | indexer.wq_b.weight |
| 70 | indexer wk (4096->128) bf16 splitK | cublas nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT | indexer.wk.weight |
| 71 | wk splitK reduce | cublasLt splitKreduce_kernel | - |
| 72 | indexer k_norm (128) | flashinfer LayerNormKernel | indexer.k_norm.weight/.bias |
| 73 | Hadamard on indexer q | sglang fast_hadamard_transform_kernel<16,7,bf16> | - |
| 74 | indexer q act_quant -> FP8 | triton _act_quant_kernel | - |
| 75 | kpool gate (4096->128) bf16 splitK | cublas nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT | indexer.index_kpool_compress_gate |
| 76 | gate splitK reduce | cublasLt splitKreduce_kernel | - |
| 77 | kpool tail write (+pool compress on close) | triton _kpool_decode_update_and_maybe_write_cache_kernel | indexer.index_kpool_compress_ape |
| 78 | x.float() for weights_proj | triton_poi_fused__to_copy_0 | - |
| 79 | weights_proj fp32 GEMV (4096->32) | cublas dot_kernel<float,128,0,...> | indexer.weights_proj.weight |
| 80 | weights_proj split reduce | cublas reduce_1Block_kernel | - |
| 81 | head-gate scale x q_scale x 128^-0.5 | triton_poi_fused_mul_unsqueeze_1 | - |
| 82 | pool_context_lens clamp(min=1) (0 -> 1 here) | at vectorized_elementwise (clamp) | - |
| 83 | logits scheduler metadata | deep_gemm sm90_paged_mqa_logits_metadata<32,256,132> | - |
| 84 | indexer pooled logits (1 pool scored; unread downstream) | deep_gemm sm90_fp8_paged_mqa_logits<1,32,128,64,...> | (reads kpool FP8 cache + q_fp8 + weights) |
| 85 | kpool topk-512 + expand x4 + tail (identity path here) | sglang kpool_topk_transform_kernel<512> | (reads page_table_1) |
| 86 | absorbed q bmm (w_kc) [8,256]x[8,256,512] | cublas nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT | w_kc (from kv_b_proj.weight) |
| 87 | MLA latent KV write | triton set_mla_kv_buffer_kernel_norope | - |
| 88 | topk clamp(min=0) | at vectorized_elementwise (clamp) | - |
| 89 | fa3 varlen scheduler prep | flash prepare_varlen_num_blocks_kernel<1,true> | - |
| 90 | sparse latent attention over 3 slots (1 dynamic split; static cap 29) | flash FlashAttnFwdSm90 (OnlyQv, PackGQA, Split, PagedKVNonTMA) | (reads MLA latent pool via topk page table) |
| 91 | split combine (LSE merge of 1 partial) | flash FlashAttnFwdCombine<(8,128),5,256,1> | - |
| 92 | output bmm (w_vc) [8,512]x[8,512,256] | cublas nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT | w_vc (from kv_b_proj.weight) |
| 93 | act quant (attn out) | sglang per_token_group_quant_flat_kernel | - |
| 94 | o_proj FP8 GEMM (2048->4096) | deep_gemm sm90_fp8_gemm_1d2d_impl | o_proj.weight(+scale_inv) |
| 95 | TP8 all-reduce (attn) | sglang all_reduce_1shot_push_kernel | - |
| 96 | HC post(attn) + FFN pre-mix GEMM fused | mhc_fused_post_pre_fma_tilelang_kernel | hc_ffn_fn (+ h_post/h_res state) |
| 97 | HC sinkhorn + combine + post-attn RMSNorm | mhc_pre_big_fuse_with_norm_tilelang_kernel | hc_ffn_base, hc_ffn_scale, post_attention_layernorm.weight |
| 98 | MoE router GEMM (4096->288) fp32 | sglang tiny_n_gemm_kernel<288,4096,3,32> | mlp.gate.weight |
| 99 | routing (sigmoid noaux_tc top-8+shared) | _router_triton_kernel | - |
| 100 | MoE input concat | at CatArrayBatchedCopy | - |
| 101 | expert align (block size 64) | _moe_align_small_numel_kernel | - |
| 102 | act quant (MoE in, row-major scales) | sglang per_token_group_quant_flat_kernel | - |
| 103 | fused MoE w13 (4096->512) | fused_moe_kernel | experts w13_weight(+scales) incl. shared expert 288 |
| 104 | SwiGLU clamp 10.0 | sglang silu_mul_clamp_kernel | - |
| 105 | act quant (SwiGLU out) | sglang per_token_group_quant_flat_kernel | - |
| 106 | fused MoE w2 (256->4096) | fused_moe_kernel | experts w2_weight(+scales) |
| 107 | topk-9 weighted sum | _moe_sum_reduce_kernel | - |
| 108 | TP8 all-reduce (MoE) | sglang all_reduce_1shot_push_kernel | - |
| 109 | HC post (FFN) | mhc_post_tilelang_kernel | (h_post/h_res state) |

## 10. Known limitations of this spec

- cuBLAS/cublasLt launches (nvjet, splitKreduce, dot/reduce) do not record pointer args in the recipe (nparams = 0 or workspace blobs); operand assignment for sites 69/75/79/86/92 is from the Python call graph, which is unambiguous in order and shapes.
- The fa3 mainloop params blob contains uninitialized padding (host-stack garbage at e.g. offsets 640-760, 896-1016); only fields listed in 5.16 are meaningful.
- The capture's true sequence length is 3 (section 1). Several values that look like lengths are static constants: 2051 (topk width), 29 splits (host cap from that width), 256-wide logits (one aligned pooled page). The dynamic values (context 1 pool, 3 attention slots, 1 split) follow from seqused_k.
- max_kvblocks_in_l2 = 113 at site 89 p24 is the recorded value; the host formula is `size_l2 / size_one_kvblock` (`flash_prepare_scheduler.cu:226-229`).
