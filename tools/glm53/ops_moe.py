"""MoE (layers 3-44) + dense MLP (layers 0-2) ops for the GLM-5.3 decode manifest.

SITE-124 CAT FINDING: the aten CatArrayBatchedCopy (topk.py:2271) is SKIPPED.
`_biased_grouped_topk_postprocess` with expert_location_dispatch_info=None
(TP-mode MoE, no EPLB) and num_token_non_padded=None (eager decode, no padded
rows) returns routed_cols unchanged (topk_ids_logical_to_physical is the
identity on info=None; _mask_topk_ids_padded_region returns early on None), so
the cat only re-copies the router's contiguous [S,9] i32 out_indices. No cat
op is emitted; moe_align reads the router's out_indices buffer directly
(gen.py wires the same buffer to moe_topk.out_indices and moe_align.topk_ids).

MoE block (recipe layer 4 = sites 122..131, identical every MoE layer;
289 experts = 288 routed + shared at id 288, topk 9, TP8-sharded inter 256):

  router   tiny_n_gemm GEMMTraitN<288,4096,3,32>,M=1,f32: 32 B TinyGEMMParams
           pack {out@0, x@8, w@16, stride_x i64@24} (tiny_gemm.cuh:64-69).
           w is BF16 [288,4096] (`const bf16_t* w`; doc sec.4.1) -- NOT f32;
           only the scores output is f32 [S,288]. grid 96 = N/N_SPLIT, static.
  topk     _router_triton_kernel (sigmoid + bias ranking, 8 argmax rounds +
           fused shared slot, renormalize; routed_scaling_factor NOT applied
           here). M=1 and the unit strides (bias/sn/wk/ik) are Triton-
           specialized out; bias_alt/input_ids/num_token_non_padded are None
           (compiled out) -> 5 ptrs + 2 f32 + 7 i32 + 2 zero i64 scratch.
  align    _moe_align_small_numel_kernel: single CTA, num_experts=290 (E+1
           bucket convention), block_size 64, numel = 9*S. NOTE: the pinned
           cubin has NP=next_pow2(numel) baked from the bs=1 capture (16);
           owner verify owns the multi-batch story (same for the M=1 router).
  quant_a  per_token_group_quant row-major (module_333): A quantized ONCE per
           token [S,4096] -> fp8 + row-major f32 scales [S,32]; fused_moe
           gathers rows via sorted_token_ids (no [9S,4096] gathered buffer).
  w13      fused_moe_kernel (module_335, MUL_ROUTED_WEIGHT=F, 196 regs):
           N=512 K=4096, c1 [9S,512] bf16.
  silu     silu_mul_clamp limit 10.0: [9S,512] -> [9S,256] (gate rows 0..255,
           up rows 256..511, non-interleaved), out_vecs 32.
  quant_b  row-major quant [9S,256] -> fp8, scales [9S,2].
  w2       fused_moe_kernel (module_337, MUL_ROUTED_WEIGHT=T, 189 regs):
           N=4096 K=256, c2 [9S,4096] bf16.
  reduce   _moe_sum_reduce_kernel: f32 acc over the 9 expert rows x2.5
           (routed_scaling_factor is a baked constexpr), store bf16 [S,4096],
           in place over the ffn input buffer.
  ar       TP8 nccl all-reduce, [S,4096] bf16 in place.

fused_moe ABI (recipe sites 127/130; a_desc/b_desc/bias/add_mask ptrs are
compiled out): 9 ptrs {a, b, c, a_scale, b_scale, topk_weights,
sorted_token_ids, expert_ids, num_tokens_post_padded}, then i32 N, K, EM,
num_valid=9, stride_am, stride_be, stride_bk, stride_bias_e=0,
stride_bias_n=0, stride_cm, stride_asm, stride_bse, stride_bsk
(ak/bn/cn/ask/bsn = 1 specialized out), 2 zero i64 Triton scratch.
EM is a HOST-side upper bound: min(9*S, 289)*64; for S<=8, EM = 9*S*64.
grid.x = 9*S*cdiv(N,128) (BLOCK_M=64, BLOCK_N=128), smem 73728 (3 stages).

QuantKernelParams (80 B, quant.cuh:195-203): TensorArgs input {ptr, i64
expert_stride=0, i64 token_stride}, TensorArgs output (same), ScaleStoreArgs
{base, u32 expert_stride=0, u32 token_stride, u32 group_stride, u32
num_groups}, u32 num_tokens, u32 hidden_size. Strides in ELEMENTS.
Flat kernel grid = cdiv(num_tokens*num_groups*kNumLanes, 256) with
kNumLanes = kGroupSize/kVecSize = 128/16 = 8, kBlockSize 256.

Dense MLP (sites 16..21, layers 0-2; intermediate 12288, TP8 shard 1536):
  quant    COL-major scales (module_233) for the DeepGEMM SFA: token_stride 1,
           STATIC group_stride = S_MAX = 16 (sglang pads m to ceil(m/4)*4 = 4
           at bs=1; the manifest fixes the [hidden/128, 16] f32 layout so the
           SFA tensormap is batch-independent: dims [16, groups], stride 64 B).
  gemm     sm90_fp8_gemm_1d2d: sites 17 and 20 are the SAME cubin (identical
           mangled symbol in the recipe, skeleton-pinned, 248 regs); only
           N/K/tensormaps/smem differ -> mlp_fp8_gemm_gate_up (N=3072 K=4096,
           smem 101760) and mlp_fp8_gemm_down (N=4096 K=1536, smem 101680).
           M dims are static S_MAX=16 in the tensormaps; the runtime M scalar
           is the seqs var. sfb = merged block scales [N/128, K/128] f32,
           kMajorSFB (sfb[n_block*K/128 + k_block]); grouped_layout null.
  silu     silu_mul_clamp limit 10.0: [S,3072] -> [S,1536], out_vecs 192.
  ar       TP8 nccl all-reduce (site 21), [S,4096] bf16 in place.
"""

from .ops_common import allreduce_bf16
from .ops_common import (S, a, cdiv, expr, extern, f32, i32, i64, mined, mined_file,  # noqa: F401
                         mul, pack, rank, tmap, var, handwritten)

# dump module variants (verified: sha + cuobjdump -res-usage, see docstring)
QUANT_COL = "module_233.cubin"      # QuantTrait<bf16,e4m3,128,ue8m0=F,rowmajor=F,aligned=T,fuse_silu=F>
QUANT_ROW = "module_333.cubin"      # QuantTrait<...,rowmajor=T,...>
FUSED_MOE_W13 = "module_335.cubin"  # fused_moe_kernel, MUL_ROUTED_WEIGHT=F, 196 regs
FUSED_MOE_W2 = "module_337.cubin"   # fused_moe_kernel, MUL_ROUTED_WEIGHT=T, 189 regs

# skeleton substrings (unique across the two quant / two 1d2d template variants)
QCOL_SUB = "per_token_group_quant_flat_kernelINS_10QuantTraitI13__nv_bfloat1613__nv_fp8_e4m3Lj128ELb0ELb0ELb1"
QROW_SUB = "per_token_group_quant_flat_kernelINS_10QuantTraitI13__nv_bfloat1613__nv_fp8_e4m3Lj128ELb0ELb1ELb1"
FP8_GEMM_SUB = "sm90_fp8_gemm_1d2d_implILN4cute4UMMA5MajorE0ELj0ELj0ELj0ELj1ELj16ELj32ELj128"

S_MAX = 16                # static M pitch of the DeepGEMM A/D/SFA tensormaps
HID = 4096
N_ROUTED = 288            # router experts (shared expert 288 is fused, topk 9)
TOPK9 = 9                 # 8 routed + 1 fused shared
ALIGN_EXPERTS = 290       # E+1 bucket convention (E = 289)
ALIGN_BLOCK = 64
MOE_INTER = 256           # 2048 / TP8
W13_N = 512               # [gate 256 | up 256]
DENSE_SHARD = 1536        # 12288 / TP8
GATE_UP_N = 3072          # 2 * 1536 (merged gate+up)
SWIGLU_LIMIT = 10.0
EM_TILE = 64              # fused_moe BLOCK_M: EM = 9*S*64

F32X = "in buffer<f32>"
F32O = "out buffer<f32>"
BF16X = "in buffer<bf16>"
BF16O = "out buffer<bf16>"
FP8X = "in buffer<fp8>"
FP8O = "out buffer<fp8>"
I32X = "in buffer<i32>"
I32O = "out buffer<i32>"


def _quant(mod, sub, hidden, groups, scale_tok, scale_grp, ntokens, grid):
    """per_token_group_quant_flat: bf16 [T,hidden] -> fp8 [T,hidden] + f32 scales.

    scale_tok/scale_grp: ScaleStoreArgs strides (elements). Row-major:
    [T, groups] -> tok=groups, grp=1. Col-major (DeepGEMM SFA): tok=1,
    grp=S_MAX (static [groups, S_MAX] f32)."""
    return {
        "params": [BF16X, FP8O, F32O],
        "impl": {"launches": [
            mined_file(mod, sub, ["bytes<80>"],
                       [256, 1, 1], grid,
                       [pack(80,
                             {"at": 0, "param": 0}, {"at": 8, "i64": 0}, {"at": 16, "i64": hidden},
                             {"at": 24, "param": 1}, {"at": 32, "i64": 0}, {"at": 40, "i64": hidden},
                             {"at": 48, "param": 2},
                             {"at": 56, "i32": 0}, {"at": 60, "i32": scale_tok},
                             {"at": 64, "i32": scale_grp}, {"at": 68, "i32": groups},
                             {"at": 72, **ntokens, "width": 4}, {"at": 76, "i32": hidden})]),
        ]},
    }


def _fused_moe(mod, n, k, am, be, bk, cm, asm, bse, bsk):
    """One fused_moe_kernel launch (w13 or w2). EM = 9*S*64 host bound;
    grid.x = 9*S*cdiv(N,128)."""
    return {
        "params": [FP8X, FP8X, BF16O, F32X, F32X, F32X, I32X, I32X, I32X],
        "impl": {"launches": [
            mined_file(mod, "fused_moe_kernel",
                       ["buffer"] * 9 + ["i32"] * 13 + ["i64"] * 2,
                       [128, 1, 1], [mul(mul(S, TOPK9), n // 128), 1, 1],
                       [a(i) for i in range(9)]
                       + [i32(n), i32(k), expr(mul(mul(S, TOPK9), EM_TILE)), i32(TOPK9),
                          i32(am), i32(be), i32(bk), i32(0), i32(0), i32(cm),
                          i32(asm), i32(bse), i32(bsk), i64(0), i64(0)],
                       smem=73728),
        ]},
    }


def _silu(out_vecs, block, grid):
    """silu_mul_clamp: 32 B SiluAndMulClampParams pack {in@0, out@8,
    limit f32@16, out_vecs@20, blocks_per_row@24, pad@28}."""
    return {
        "params": [BF16X, BF16O],
        "impl": {"launches": [
            mined("silu_mul_clamp", ["bytes<32>"], block, grid,
                  [pack(32, {"at": 0, "param": 0}, {"at": 8, "param": 1},
                        {"at": 16, "f32": SWIGLU_LIMIT}, {"at": 20, "i32": out_vecs},
                        {"at": 24, "i32": 1}, {"at": 28, "i32": 0})]),
        ]},
    }


def _fp8_gemm(n, k, smem):
    """sm90_fp8_gemm_1d2d (dense GemmType 0): fp8 [S,K] @ w [N,K]^T -> bf16 [S,N].

    kernel ABI (fp8_1d2d.cuh:50-55, lift sm90_fp8_gemm_1d2d.json):
    (float* sfb, int* grouped_layout=null, u32 M=seqs, u32 N, u32 K,
     tmap A u8 [K,S_MAX] box [128,16] swz128, tmap B u8 [K,N] box [128,32]
     swz128, tmap D bf16 [N,S_MAX] stride N*2 box [32,16] swz64,
     tmap SFA f32 [S_MAX,K/128] stride S_MAX*4 box [16,1] swz0) -- all
    l2_promotion 256 per the lift. Persistent kernel: grid = kNumSMs = 132."""
    return {
        "params": [FP8X, FP8X, F32X, F32X, BF16O],
        "impl": {"launches": [
            mined(FP8_GEMM_SUB,
                  ["buffer", "i64", "i32", "i32", "i32",
                   "bytes<128>", "bytes<128>", "bytes<128>", "bytes<128>"],
                  [256, 1, 1], [132, 1, 1],
                  [a(3), i64(0), var("seqs"), i32(n), i32(k),
                   pack(128, tmap(0, "u8", [k, S_MAX], [k], [128, 16])),
                   pack(128, tmap(1, "u8", [k, n], [k], [128, 32])),
                   pack(128, tmap(4, "bf16", [n, S_MAX], [n * 2], [32, 16], swizzle=64)),
                   pack(128, tmap(2, "f32", [S_MAX, k // 128], [S_MAX * 4], [16, 1], swizzle=0))],
                  smem=smem),
        ]},
    }


def _ar():
    """TP8 all-reduce of [S,4096], in place; gen.py selects the wire."""
    return {
        "params": ["inout buffer<bf16>"],
        "impl": {"launches": [
            allreduce_bf16(a(0), expr(mul(S, HID))),
        ]},
    }


def ops():
    out = {}

    # --- MoE router: scores [S,288] f32 = x [S,4096] bf16 @ gate.weight^T
    # D3: the template M is the row count; pin the M=16 entry of module_327
    # (per-row arithmetic identical to the M=1 entry, tiny_gemm.cuh:103-140).
    out["moe_router"] = {
        "params": [BF16X, BF16X, F32O],
        "impl": {"launches": [
            mined("Lj16Ef", ["bytes<32>"],
                  [256, 1, 1], [96, 1, 1],
                  [pack(32, {"at": 0, "param": 2}, {"at": 8, "param": 0},
                        {"at": 16, "param": 1}, {"at": 24, "i64": HID})]),
        ]},
    }

    # --- MoE topk: sigmoid + correction-bias ranking, 8 routed + shared slot,
    #     renormalized weights (shared = 0.4); recipe site 123 scalar order:
    #     scale 2.5, softcap 0, stride_bias_alt 0, stride_input_ids 0,
    #     stride_sm 288, stride_wm 9, stride_im 9, stride_pm 0, stride_pk 0.
    # D1: module_329 bakes M=1 (mask_m is a constant); module_411 takes M as a
    # runtime i32 at arg 5 (ELF KPARAM: 17 params = 329's 16 + M).
    out["moe_topk"] = {
        "params": [F32X, F32X, F32O, I32O, I32O],
        "impl": {"launches": [
            mined_file("module_411.cubin", "_router_triton_kernel",
                  ["buffer"] * 5 + ["i32"] + ["f32", "f32"] + ["i32"] * 7 + ["i64"] * 2,
                  [32, 1, 1], [S, 1, 1],
                  [a(0), a(1), a(2), a(3), a(4),
                   var("tokens"), f32(2.5), f32(0.0),
                   i32(0), i32(0), i32(N_ROUTED), i32(TOPK9), i32(TOPK9), i32(0), i32(0),
                   i64(0), i64(0)],
                  smem=64),
        ]},
    }

    # --- MoE align: small-numel sort + 64-pad (reads the router's out_indices
    #     directly -- see the site-124 cat note in the module docstring).
    # D2: the mined cubins bake NP = 16 / 32 and silently drop entries at
    # S >= 4; glm53_moe_align reproduces the contract for numel <= 144.
    out["moe_align"] = {
        "params": [I32X, I32O, I32O, I32O],
        "impl": {"launches": [
            handwritten("glm53_misc", "glm53_moe_align",
                  ["buffer"] * 4 + ["i32"] * 3,
                  [256, 1, 1], [1, 1, 1],
                  [a(0), a(1), a(2), a(3),
                   i32(ALIGN_EXPERTS), i32(ALIGN_BLOCK), expr(mul(S, TOPK9))]),
        ]},
    }

    # --- MoE a1 quant: [S,4096] bf16 -> fp8, row-major scales [S,32]
    out["moe_quant_a"] = _quant(QUANT_ROW, QROW_SUB, HID, HID // 128, 32, 1,
                                var("seqs"), [S, 1, 1])

    # --- MoE w13 GEMM: c1 [9S,512] bf16
    out["moe_w13"] = _fused_moe(FUSED_MOE_W13, W13_N, HID,
                                am=4096, be=2097152, bk=4096, cm=512,
                                asm=32, bse=128, bsk=32)

    # --- MoE swiglu clamp: [9S,512] -> [9S,256]
    out["moe_silu"] = _silu(MOE_INTER // 8, [32, 1, 1], [mul(S, TOPK9), 1, 1])

    # --- MoE a2 quant: [9S,256] bf16 -> fp8, row-major scales [9S,2]
    out["moe_quant_b"] = _quant(QUANT_ROW, QROW_SUB, MOE_INTER, MOE_INTER // 128, 2, 1,
                                expr(mul(S, TOPK9)), [cdiv(mul(S, TOPK9), 16), 1, 1])

    # --- MoE w2 GEMM (topk weights applied here): c2 [9S,4096] bf16
    out["moe_w2"] = _fused_moe(FUSED_MOE_W2, HID, MOE_INTER,
                               am=256, be=1048576, bk=256, cm=4096,
                               asm=2, bse=64, bsk=2)

    # --- MoE sum-reduce: sum over 9 rows x2.5 (constexpr) -> [S,4096] bf16
    out["moe_sum_reduce"] = {
        "params": [BF16X, BF16O],
        "impl": {"launches": [
            mined("_moe_sum_reduce_kernel",
                  ["buffer", "i32", "i32", "buffer", "i32", "i32", "i32", "i32", "i64", "i64"],
                  [512, 1, 1], [S, 2, 1],
                  [a(0), i32(TOPK9 * HID), i32(HID), a(1), i32(HID),
                   var("seqs"), i32(TOPK9), i32(HID), i64(0), i64(0)]),
        ]},
    }

    # --- MoE all-reduce (site 132)
    out["moe_ar"] = _ar()

    # --- dense MLP (layers 0-2): quant -> 1d2d GEMM -> silu -> quant -> GEMM -> ar
    out["mlp_quant_a"] = _quant(QUANT_COL, QCOL_SUB, HID, HID // 128, 1, S_MAX,
                                var("seqs"), [S, 1, 1])
    out["mlp_fp8_gemm_gate_up"] = _fp8_gemm(GATE_UP_N, HID, 101760)
    out["mlp_silu"] = _silu(DENSE_SHARD // 8, [192, 1, 1], [S, 1, 1])
    out["mlp_quant_b"] = _quant(QUANT_COL, QCOL_SUB, DENSE_SHARD, DENSE_SHARD // 128, 1, S_MAX,
                                var("seqs"), [cdiv(mul(S, 12), 32), 1, 1])
    out["mlp_fp8_gemm_down"] = _fp8_gemm(HID, DENSE_SHARD, 101680)
    out["mlp_ar"] = _ar()

    return out
