"""DSA (DeepSeek Sparse Attention) ops for the GLM-5.3 decode manifest.

One DSA layer per call, sites 61-95 of docs/glm53/dsa.md. Kernel wiring
is probe/track-verified; the fa3 2944-B pack follows the field map from
the ABI review (6 tensormaps at 512..1919 all unused: zero TMA
instructions in module_383 SASS; Q/K/V/Qv/O ride raw pointer+stride
fields; kv state is slot-linear base + slot*1024).

Per-step (gen.py calls once, not per layer):
  prep      glm53_dsa_prep: seq_lens -> pool_seqlens / pool_ctx_lens
            (clamped) / dsa_seqlens (fa3 seqused_k) / pooled page table
            (block_table[:, ::4], device-built; never a stride-256 input)
            / slot_table (token -> physical kv slot, page_size-1 view).

Per DSA layer:
  quant_qkv / qkv_a / q_a_norm / kv_a_norm / quant_q / q_b
  wq_b / wk / k_norm / hadamard / act_quant / kpool_gate / kpool_update
  logits_meta / logits / topk / clamp / fa3_prep / fa3 / fa3_combine
  w_kc / kv_store / w_vc / quant_o / o_proj / ar

Layout constants (bt_cols, st_cols) are gen.py capacity choices passed on
the interface; unit-256 pages make the block table the kpool page table
directly (logits row stride 64*bt_cols) and st_cols = 256*bt_cols.
Var-shaped activations use static S_MAX=16 pitch; the runtime M scalar
is the seqs var (fp8 GEMM kernels never write rows >= M).
"""

from .ops_common import allreduce_bf16
from .ops_common import (S, a, cdiv, expr, extern, f32, i32, i64, mined,
                         mined_file, mul, pack, rank, scr, tmap, var,  # noqa: F401
                         handwritten)

QUANT_COL = "module_233.cubin"     # per_token_group_quant_flat, col-major SFA
QCOL_SUB = ("per_token_group_quant_flat_kernelINS_10QuantTraitI13__nv_bfloat16"
            "13__nv_fp8_e4m3Lj128ELb0ELb0ELb1")
FP8_N16_SUB = ("sm90_fp8_gemm_1d2d_implILN4cute4UMMA5MajorE0ELj0ELj0ELj0ELj1"
               "ELj16ELj16ELj128")
FP8_N32_SUB = ("sm90_fp8_gemm_1d2d_implILN4cute4UMMA5MajorE0ELj0ELj0ELj0ELj1"
               "ELj16ELj32ELj128")

S_MAX = 16
HID = 4096
QLORA = 1536
KVLORA = 512
QKV_A_N = 2048             # fused q_a(1536) + kv_a(512)
IDX_HEADS = 32
IDX_DIM = 128
TOPK_W = 2051              # constant topk row width (2048 + 3)
SPLITS = 29                # fa3 static split cap
L2_BLOCKS = 113            # max_kvblocks_in_l2 (host heuristic, recorded)
SMS = 132

F32X = "in buffer<f32>"
F32O = "out buffer<f32>"
BF16X = "in buffer<bf16>"
BF16O = "out buffer<bf16>"
FP8X = "in buffer<fp8>"
FP8O = "out buffer<fp8>"
I32X = "in buffer<i32>"
I32O = "out buffer<i32>"
STATE = "inout state"


def _quant(hidden, groups, grid):
    """per_token_group_quant_flat (col-major SFA): bf16 [T,hidden] ->
    fp8 [T,hidden] + f32 scales [groups, S_MAX] (group_stride 16, tok 1)."""
    return {
        "params": [BF16X, FP8O, F32O],
        "impl": {"launches": [
            mined_file(QUANT_COL, QCOL_SUB, ["bytes<80>"],
                       [256, 1, 1], grid,
                       [pack(80,
                             {"at": 0, "param": 0}, {"at": 8, "i64": 0}, {"at": 16, "i64": hidden},
                             {"at": 24, "param": 1}, {"at": 32, "i64": 0}, {"at": 40, "i64": hidden},
                             {"at": 48, "param": 2},
                             {"at": 56, "i32": 0}, {"at": 60, "i32": 1},
                             {"at": 64, "i32": S_MAX}, {"at": 68, "i32": groups},
                             {"at": 72, "var": S, "width": 4}, {"at": 76, "i32": hidden})]),
        ]},
    }


def _fp8_gemm(sub, n, k, smem, block_n):
    """sm90_fp8_gemm_1d2d: fp8 [S,K] @ w [N,K]^T -> bf16 [S,N].
    block_n selects the template variant (16: swzD 32 box [16,16];
    32: swzD 64 box [32,16]). M dims static S_MAX, M scalar = seqs."""
    dbox = [block_n, 16]
    dswz = 32 if block_n == 16 else 64
    return {
        "params": [FP8X, FP8X, F32X, F32X, BF16O],
        "impl": {"launches": [
            mined(sub,
                  ["buffer", "i64", "i32", "i32", "i32",
                   "bytes<128>", "bytes<128>", "bytes<128>", "bytes<128>"],
                  [256, 1, 1], [SMS, 1, 1],
                  [a(3), i64(0), var("seqs"), i32(n), i32(k),
                   pack(128, tmap(0, "u8", [k, S_MAX], [k], [128, 16])),
                   pack(128, tmap(1, "u8", [k, n], [k], [128, block_n])),
                   pack(128, tmap(4, "bf16", [n, S_MAX], [n * 2], dbox, swizzle=dswz)),
                   pack(128, tmap(2, "f32", [S_MAX, k // 128], [S_MAX * 4], [16, 1], swizzle=0))],
                  smem=smem),
        ]},
    }


def _gemm_tn(n, k, a_off=0, w_off=0, c_off=0, ldc=None, a_stride=None):
    """One cublasLt bf16 TN extern: out[m,n] = a[m,k] @ w[n,k]^T.
    m rides op param 3 (i32); offsets in bytes."""
    args = [a(0, a_off), a(1, w_off), a(2, c_off), a(3), i32(n), i32(k)]
    if ldc is not None:
        args.append(i32(ldc))
    if a_stride is not None:
        args.append(i32(a_stride))
    types = [BF16X, BF16X, BF16O, "i32", "i32", "i32"] + ["i32"] * (len(args) - 6)
    return extern("cublaslt_bf16_tn", types, args)


def _bmm8(n, k, a_stride, ldc):
    """8 per-head extern GEMMs emulating one [8,T,k] x [8,n,k] bmm.
    a: [T, 8*k] token-major, w: [8, n, k] (n-major per head), c: [T, 8*n]."""
    return [_gemm_tn(n, k, a_off=h * k * 2, w_off=h * n * k * 2,
                     c_off=h * n * 2, ldc=ldc, a_stride=a_stride)
            for h in range(8)]


def ops(cfg):
    # Unit-256 pages (arch_decode.md section 1): block_table [S, bt_cols]
    # covers 256 tokens = 64 pools per entry; the logits kernel reads it
    # directly (block_table_stride = bt_cols, logits row stride 64*bt_cols).
    bt_cols = cfg["bt_cols"]           # shared token-page table cols (page 256)
    st_cols = cfg["st_cols"]           # slot table cols (= 256*bt_cols)
    assert st_cols == 256 * bt_cols
    logits_stride = bt_cols * 64
    out = {}

    # ---------- once per step ----------
    out["dsa_prep"] = {
        "params": [I32X, I32X, I32O, I32O, I32O, I32O, "i32", "i32"],
        "impl": {"launches": [
            handwritten("glm53_dsa", "glm53_dsa_prep",
                        ["buffer", "buffer", "i32",
                         "buffer", "buffer", "buffer", "buffer", "i32"],
                        [128, 1, 1], [S, 1, 1],
                        # kernel ABI interleaves the two i32 scalars:
                        # (seq_lens, block_table, bt_cols, pool, pool_ctx,
                        #  dsa_lens, slot_table, st_cols)
                        [a(0), a(1), a(6), a(2), a(3), a(4), a(5), a(7)]),
        ]},
    }

    # ---------- qkv_a path (sites 63-68) ----------
    out["dsa_quant_qkv"] = _quant(HID, 32, [S, 1, 1])
    out["dsa_qkv_a"] = _fp8_gemm(FP8_N16_SUB, QKV_A_N, HID, 68992, 16)

    # q_a / kv_a norms reuse head_rms_norm (d/eps on the interface)

    out["dsa_quant_q"] = _quant(QLORA, 12, [cdiv(mul(S, 3), 8), 1, 1])
    out["dsa_q_b"] = _fp8_gemm(FP8_N16_SUB, QKV_A_N, QLORA, 68912, 16)

    # ---------- indexer q/k (sites 69-76) ----------
    out["dsa_wq_b"] = {
        "params": [BF16X, BF16X, BF16O, "i32"],
        "impl": {"launches": [_gemm_tn(IDX_HEADS * IDX_DIM, QLORA)]},
    }
    out["dsa_wk"] = {
        "params": [BF16X, BF16X, BF16O, "i32"],
        "impl": {"launches": [_gemm_tn(IDX_DIM, HID)]},
    }
    out["dsa_kpool_gate"] = {
        "params": [BF16X, BF16X, BF16O, "i32"],
        "impl": {"launches": [_gemm_tn(IDX_DIM, HID)]},
    }

    out["dsa_k_norm"] = {
        "params": [BF16X, F32X, F32X, BF16O],
        "impl": {"launches": [
            handwritten("glm53_dsa", "glm53_layer_norm_128",
                        ["buffer", "buffer", "buffer", "buffer", "f32"],
                        [128, 1, 1], [S, 1, 1],
                        [a(0), a(1), a(2), a(3), f32(1e-6)]),
        ]},
    }

    # site 73: rotate_activation(query), H_128/sqrt(128) per [128] row
    out["dsa_hadamard"] = {
        "params": [BF16X, BF16O],
        "impl": {"launches": [
            mined("fast_hadamard_transform_kernel", ["bytes<56>"],
                  [16, 1, 1], [mul(S, IDX_HEADS), 1, 1],
                  [pack(56,
                        {"at": 0, "expr": mul(S, IDX_HEADS), "width": 4},
                        {"at": 4, "i32": IDX_DIM}, {"at": 8, "i32": 7}, {"at": 12, "i32": 0},
                        {"at": 16, "i64": IDX_DIM}, {"at": 24, "i64": IDX_DIM},
                        {"at": 32, "i32": 0x3db504f3}, {"at": 36, "i32": 0},
                        {"at": 40, "param": 0}, {"at": 48, "param": 1})],
                  smem=512),
        ]},
    }

    # site 74: q -> fp8 [S,32,128] + ue8m0 scales [S,32]
    out["dsa_act_quant"] = {
        "params": [BF16X, FP8O, F32O],
        "impl": {"launches": [
            mined("_act_quant_kernel",
                  ["buffer", "buffer", "buffer", "i32", "i32", "i64", "i64"],
                  [128, 1, 1], [S, 1, 1],
                  [a(0), a(1), a(2), expr(mul(S, IDX_HEADS)), i32(IDX_DIM), i64(0), i64(0)],
                  smem=128),
        ]},
    }

    # site 77: tail ring + pool compress (handwritten; mined cubins bake
    # BLOCK_TABLE_COLS / stride specializations from the seq=3 capture)
    out["dsa_kpool_update"] = {
        "params": [STATE, STATE, BF16X, BF16X, F32X, I32X, I32X, I32X, I32X, I32X, "i32"],
        "impl": {"launches": [
            handwritten("glm53_dsa", "glm53_kpool_update",
                        ["buffer"] * 10 + ["i32"],
                        [128, 1, 1], [S, 1, 1],
                        [a(i) for i in range(11)]),
        ]},
    }

    # sites 78-81 fused: per-head logits weights
    #   w[s,h] = (x_norm[s] . weights_proj[h]) * 32^-0.5 * q_scale[s,h] * 128^-0.5
    # block 1024 = 32 warps (warp h = head h); the fp32 cast + GEMV + scale
    # folds of _get_logits_head_gate are all inside (glm53_dsa.cu).
    out["dsa_weights_proj"] = {
        "params": [BF16X, F32X, F32X, F32O],
        "impl": {"launches": [
            handwritten("glm53_dsa", "glm53_weights_proj",
                        ["buffer", "buffer", "buffer", "buffer"],
                        [1024, 1, 1], [S, 1, 1],
                        [a(0), a(1), a(2), a(3)]),
        ]},
    }

    # ---------- indexer logits + topk (sites 82-85, 88) ----------
    out["dsa_logits_meta"] = {
        "params": [I32X, I32O],
        "impl": {"launches": [
            mined("sm90_paged_mqa_logits_metadata",
                  ["i32", "i32", "bytes<1>", "buffer", "i64", "buffer"],
                  [32, 1, 1], [1, 1, 1],
                  [var("seqs"), i32(1), pack(1, {"at": 0, "i32": 1, "width": 1}),
                   a(0), i64(0), a(1)],
                  smem=128),
        ]},
    }

    out["dsa_logits"] = {
        "params": [I32X, F32O, I32X, I32X, FP8X, "in state", "in state", F32X],
        "impl": {"launches": [
            mined("sm90_fp8_paged_mqa_logitsILj1ELj32ELj128",
                  ["i32", "i32", "i32", "buffer", "buffer", "buffer", "i64", "buffer",
                   "bytes<128>", "bytes<128>", "bytes<128>", "bytes<128>"],
                  [640, 1, 1], [SMS, 1, 1],
                  [var("seqs"), i32(logits_stride), i32(bt_cols),
                   a(0), a(1), a(2), i64(0), a(3),
                   # q is a rank-2 descriptor in sglang (rows = heads*seqs);
                   # rank-3 here made the TMA unit trap at use time.
                   pack(128, tmap(4, "u8", [IDX_DIM, 0],
                                  [IDX_DIM], [IDX_DIM, IDX_HEADS])),
                   # idx state: layered 92928-B page (11 x 8448); the layer's
                   # 8448-B block base rides the call-site state offset
                   pack(128, tmap(5, "u8", [IDX_DIM, 64, 0],
                                  [IDX_DIM, 92928], [IDX_DIM, 64, 1])),
                   pack(128, tmap(6, "f32", [64, 0],
                                  [92928], [64, 1], swizzle=0)),
                   # weights w is rank-2 [heads, seqs]; dim1 spans the buffer.
                   pack(128, tmap(7, "f32", [IDX_HEADS, 0],
                                  [IDX_HEADS * 4], [IDX_HEADS, 1],
                                  swizzle=0))],
                  smem=131140),
        ]},
    }

    out["dsa_topk"] = {
        "params": [F32X, I32X, I32X, I32X, I32O],
        "impl": {"launches": [
            mined("kpool_topk_transform_kernel",
                  ["bytes<40>", "buffer", "i64", "i32", "i32", "i32",
                   "buffer", "i64", "i64", "i64", "buffer"],
                  [1024, 1, 1], [S, 1, 1],
                  [pack(40,
                        {"at": 0, "param": 0}, {"at": 8, "i64": 0}, {"at": 16, "i64": 0},
                        {"at": 24, "param": 1}, {"at": 32, "i64": logits_stride}),
                   a(4), i64(TOPK_W), i32(4), i32(2048), i32(TOPK_W),
                   a(2), i64(st_cols), i64(0), i64(0), a(3)],
                  smem=32768),
        ]},
    }

    out["dsa_clamp"] = {
        "params": ["inout buffer<i32>"],
        "impl": {"launches": [
            handwritten("glm53_dsa", "glm53_clamp0_i32",
                        ["buffer", "i32"],
                        [256, 1, 1], [cdiv(mul(S, TOPK_W), 256), 1, 1],
                        [a(0), expr(mul(S, TOPK_W))]),
        ]},
    }

    # ---------- fa3 (sites 89-91) ----------
    fa3_scratch = {
        "oacc": {"dtype": "f32", "shape": [SPLITS * 8, S_MAX, 512]},
        "lseacc": {"dtype": "f32", "shape": [SPLITS * 8, S_MAX]},
        "lse": {"dtype": "f32", "shape": [8 * S_MAX]},
        "nsd": {"dtype": "i32", "shape": [8]},
        "nmb": {"dtype": "i32", "shape": [8]},
        "vbi": {"dtype": "i32", "shape": [8]},
        "sem": {"dtype": "i32", "shape": [8]},
    }

    # fa3 fwd: 2944-B pack. Only listed fields are nonzero; ptr/shape/stride
    # map per the ABI review. T dims are i32 var tokens; scratch strides use
    # the static S_MAX pitch (oacc [29*8,16,512], lseacc [29*8,16]).
    fa3_fields = [
        {"at": 0, "param": 0},                          # ptr_Q (alias qv)
        {"at": 8, "var": S, "width": 4},                # shape_Q (T,64,8,1)
        {"at": 12, "i32": 64}, {"at": 16, "i32": 8}, {"at": 20, "i32": 1},
        {"at": 24, "i64": 512}, {"at": 32, "i64": 64}, {"at": 40, "i64": 0},   # stride_Q
        {"at": 48, "i32": 8}, {"at": 52, "var": S, "width": 4},                 # shape_Q_packed
        {"at": 56, "i32": 64}, {"at": 60, "i32": 1}, {"at": 64, "i32": 1},
        {"at": 72, "i64": 64}, {"at": 80, "i64": 512}, {"at": 88, "i64": 512}, {"at": 96, "i64": 0},
        {"at": 104, "param": 1},                        # ptr_K (alias V)
        {"at": 112, "i32": 1}, {"at": 116, "i32": 64}, {"at": 120, "i32": 1},   # shape_K
        {"at": 124, "i32": 0x7fffffff},
        {"at": 128, "i64": 5632}, {"at": 136, "i64": 5632}, {"at": 144, "i64": 5632},  # stride_K
        {"at": 152, "param": 1},                        # ptr_V
        {"at": 160, "i32": 512},                        # headdim_v
        {"at": 168, "i64": 5632}, {"at": 176, "i64": 5632}, {"at": 184, "i64": 5632},  # stride_V
        {"at": 200, "i32": 0}, {"at": 204, "i32": 64}, {"at": 208, "i32": 1},   # shape_K_new
        {"at": 212, "var": S, "width": 4},
        {"at": 272, "param": 0},                        # ptr_Qv
        {"at": 280, "i64": 4096}, {"at": 288, "i64": 512}, {"at": 296, "i64": 0},   # stride_Qv
        {"at": 304, "i32": 8}, {"at": 308, "var": S, "width": 4},               # shape_Qv_packed
        {"at": 312, "i32": 512}, {"at": 316, "i32": 1}, {"at": 320, "i32": 1},
        {"at": 328, "i64": 512}, {"at": 336, "i64": 4096}, {"at": 344, "i64": 4096},
        {"at": 352, "i64": 0},
        {"at": 368, "i32": TOPK_W},                     # seqlen_ro (not read)
        {"at": 408, "param": 2},                        # ptr_pagetable
        {"at": 416, "var": S, "width": 4}, {"at": 420, "i32": TOPK_W},          # shape_pagetable
        {"at": 424, "i64": TOPK_W},                     # stride_pagetable
        {"at": 432, "i32": 1}, {"at": 436, "i32": 0}, {"at": 440, "i32": 0},    # page_size divmod
        {"at": 444, "i32": 1}, {"at": 448, "i32": 0}, {"at": 452, "i32": 0},    # blockN_per_page
        {"at": 456, "i32": 8}, {"at": 460, "i32": 0x80000000}, {"at": 464, "i32": 2},  # qhead divmod
        # Six rank-4 TMA descriptors (dtype u16, swizzle 128, l2 128), layout
        # decoded byte-exact from the sglang capture. All-zero descriptors
        # here made the mainloop's tma_load_Qv fault (CUDA_ERROR_ILLEGAL_ADDRESS).
        tmap(0, "u16", [64, 1, 8, 1], [1024, 128, 1024], [64, 64, 1, 1], l2=128, at=512),  # stride2 unused (dim3=1); 0 rejected by verify
        tmap(1, "u16", [64, 1, 1, 0], [128, 128, 11264], [64, 64, 1, 1], l2=128, at=768),
        tmap(1, "u16", [512, 1, 1, 0], [11264, 11264, 11264], [64, 64, 1, 1], l2=128, at=1024),
        tmap(1, "u16", [64, 1, 1, 0], [128, 128, 11264], [64, 64, 1, 1], l2=128, at=1280),
        tmap(1, "u16", [512, 1, 1, 0], [11264, 11264, 11264], [64, 64, 1, 1], l2=128, at=1536),
        tmap(0, "u16", [512, 1, 8, 0], [8192, 1024, 8192], [64, 64, 1, 1], l2=128, at=1792),
        {"at": 2056, "i32": 0x3db8aa3b},                # softmax_scale_log2
        {"at": 2140, "i32": TOPK_W - 1},                # window_size_left
        {"at": 2160, "i32": SPLITS},                    # num_splits static cap
        {"at": 2168, "i64": 0},                         # kv_batch_idx
        {"at": 2176, "param": 4},                       # cu_seqlens_q
        {"at": 2184, "i64": 0},                         # cu_seqlens_k
        {"at": 2200, "i64": 0},                         # seqused_q
        {"at": 2208, "param": 3},                       # seqused_k
        {"at": 2216, "i64": 0},                         # leftpad
        {"at": 2232, "i64": 0},                         # fork field
        {"at": 2304, "param": 5},                       # ptr_O
        {"at": 2312, "var": S, "width": 4},             # shape_O (T,512,8,1,29)
        {"at": 2316, "i32": 512}, {"at": 2320, "i32": 8}, {"at": 2324, "i32": 1},
        {"at": 2328, "i32": SPLITS},
        {"at": 2336, "i64": 4096}, {"at": 2344, "i64": 512},                    # stride_O
        {"at": 2352, "i64": 0}, {"at": 2360, "i64": 0},
        {"at": 2368, "i32": 8}, {"at": 2372, "var": S, "width": 4},             # shape_O_packed
        {"at": 2376, "i32": 512}, {"at": 2380, "i32": 1}, {"at": 2384, "i32": 1},
        {"at": 2388, "i32": SPLITS},
        {"at": 2392, "i64": 512}, {"at": 2400, "i64": 4096},                    # stride_O_packed
        {"at": 2408, "i64": 4096}, {"at": 2416, "i64": 0}, {"at": 2424, "i64": 0},
        {"at": 2432, "scratch": "oacc"},                # ptr_O_partial
        {"at": 2440, "i64": 512}, {"at": 2448, "i64": S_MAX * 512},             # stride_O_partial
        {"at": 2456, "i64": 0}, {"at": 2464, "i64": S_MAX * 4096},
        {"at": 2472, "i64": S_MAX * 512}, {"at": 2480, "i64": 512},             # _packed
        {"at": 2488, "i64": S_MAX * 4096}, {"at": 2496, "i64": 0},
        {"at": 2504, "i64": S_MAX * 4096},
        {"at": 2512, "scratch": "lse"},                 # ptr_LSE
        {"at": 2520, "i64": S_MAX}, {"at": 2528, "i64": 0}, {"at": 2536, "i64": 0},
        {"at": 2544, "i32": 8}, {"at": 2548, "var": S, "width": 4},             # shape_LSE_packed
        {"at": 2552, "i32": 1}, {"at": 2556, "i32": 1}, {"at": 2560, "i32": SPLITS},
        {"at": 2568, "i64": S_MAX}, {"at": 2576, "i64": 8 * S_MAX},
        {"at": 2584, "i64": 0}, {"at": 2592, "i64": 0},
        {"at": 2600, "scratch": "lseacc"},              # ptr_LSE_partial
        {"at": 2608, "i64": S_MAX}, {"at": 2616, "i64": 0}, {"at": 2624, "i64": 8 * S_MAX},
        {"at": 2632, "i64": S_MAX}, {"at": 2640, "i64": 8 * S_MAX},
        {"at": 2648, "i64": 0}, {"at": 2656, "i64": 8 * S_MAX},
        {"at": 2664, "i32": 8}, {"at": 2668, "i32": 0x80000000}, {"at": 2672, "i32": 2},
        {"at": 2688, "param": 4},                       # cu_seqlens (epilogue)
        {"at": 2704, "i32": 0}, {"at": 2708, "i32": SMS},                       # hw_info
        {"at": 2744, "i32": 1}, {"at": 2748, "var": S, "width": 4},             # num_head, num_batch
        {"at": 2752, "i32": 8}, {"at": 2756, "i32": 1},                         # qhead_per_khead, seqlen
        {"at": 2760, "i32": 1}, {"at": 2764, "i32": 0}, {"at": 2768, "i32": 0},  # head divmod
        {"at": 2772, "i32": SPLITS}, {"at": 2776, "i32": 0x8d3dcb09}, {"at": 2780, "i32": 4},
        {"at": 2784, "scratch": "sem"},
        {"at": 2792, "param": 4},
        {"at": 2808, "scratch": "nsd"},
        {"at": 2816, "scratch": "nmb"},
        {"at": 2824, "scratch": "vbi"},
    ]
    # combine: 232-B pack, replicate recorded fields (T dims var; scratch
    # strides static S_MAX pitch). grid [1, 4, S].
    comb_fields = [
        {"at": 0, "scratch": "oacc"},                   # ptr_O_partial
        {"at": 8, "var": S, "width": 4},                # shape (T,512,29,8)
        {"at": 12, "i32": 512}, {"at": 16, "i32": SPLITS}, {"at": 20, "i32": 8},
        {"at": 24, "i32": 1},                           # dv stride
        {"at": 32, "i64": 512},                         # q stride
        {"at": 40, "i64": S_MAX * 4096},                # split stride (8*S_MAX*512)
        {"at": 48, "i64": S_MAX * 512},                 # head stride
        {"at": 56, "i64": 0},
        {"at": 64, "scratch": "lseacc"},                # ptr_LSE_partial
        {"at": 72, "var": S, "width": 4},               # shape (T,29,8,1)
        {"at": 76, "i32": SPLITS}, {"at": 80, "i32": 8}, {"at": 84, "i32": 1},
        {"at": 88, "i64": 8 * S_MAX},                   # split stride
        {"at": 96, "i64": S_MAX},                       # q stride (pitch)
        {"at": 104, "i64": 0},
        {"at": 112, "param": 5},                        # ptr_O
        {"at": 120, "i64": 4096}, {"at": 128, "i64": 512}, {"at": 136, "i64": 0},
        {"at": 144, "scratch": "lse"},                  # ptr_LSE
        {"at": 152, "i64": 1}, {"at": 160, "i64": 0},
        {"at": 168, "i32": 1}, {"at": 172, "i32": 0}, {"at": 176, "i32": 0},    # seqlen divmod
        {"at": 180, "i32": 8}, {"at": 184, "i32": 0x80000000}, {"at": 188, "i32": 2},  # head divmod
        {"at": 192, "param": 4},                        # cu_seqlens
        {"at": 200, "i64": 0},                          # seqused
        {"at": 208, "scratch": "nsd"},
        {"at": 216, "scratch": "vbi"},
        {"at": 224, "scratch": "sem"},
    ]
    # D8: op scratch is op-private, so prepare, fa3 and combine are ONE op's
    # three launches (combine would otherwise read partials fa3 never wrote).
    out["dsa_attn"] = {
        "params": [BF16X, "in state", I32X, I32X, I32X, BF16O],
        "impl": {"scratch": fa3_scratch, "launches": [
            mined("prepare_varlen_num_blocks_kernel",
                  ["i32", "i32", "i32", "buffer", "i64", "i64", "i64", "buffer", "i64",
                   "i32", "i32", "i32", "i32", "i32", "bytes<12>", "bytes<12>",
                   "buffer", "buffer", "buffer", "buffer", "i64",
                   "bytes<1>", "bytes<1>", "bytes<1>", "i32"],
                  [32, 1, 1], [1, 1, 1],
                  # p3 = cu_seqlens_q, p7 = seqused_k (capture pointer identity).
                  # Swapping them made prepare compute garbage scheduler
                  # metadata and fa3 then gathered out of bounds.
                  [i32(1), i32(TOPK_W), i32(0), a(4), i64(0), i64(0), i64(0), a(3), i64(0),
                   var("seqs"), i32(1), i32(8), i32(SMS), i32(SPLITS),
                   pack(12, {"at": 0, "i32": 64}, {"at": 4, "i32": 0x80000000}, {"at": 8, "i32": 5}),
                   pack(12, {"at": 0, "i32": 64}, {"at": 4, "i32": 0x80000000}, {"at": 8, "i32": 5}),
                   scr("sem"), scr("nmb"), scr("nsd"), scr("vbi"), i64(0),
                   pack(1, {"at": 0, "i32": 1, "width": 1}),
                   pack(1, {"at": 0, "i32": 0, "width": 1}),
                   pack(1, {"at": 0, "i32": 1, "width": 1}),
                   i32(L2_BLOCKS)]),
            mined("FlashAttnFwdSm90", ["bytes<2944>"],
                  [384, 1, 1], [SMS, 1, 1],
                  [pack(2944, *fa3_fields)],
                  smem=232448, pdl=True),
            mined("FlashAttnFwdCombine", ["bytes<232>"],
                  [256, 1, 1], [1, 4, S],
                  [pack(232, *comb_fields)],
                  smem=17536),
        ]},
    }

    # ---------- absorbed bmms + kv write + o_proj (sites 86-87, 92-95) ----------
    out["dsa_w_kc"] = {   # [T,8,256] x w_kc[8,512,256] -> [T,8,512] token-major
        "params": [BF16X, BF16X, BF16O, "i32"],
        "impl": {"launches": _bmm8(512, 256, a_stride=2048, ldc=4096)},
    }

    out["dsa_kv_store"] = {
        "params": [STATE, BF16X, I32X],
        "impl": {"launches": [
            handwritten("glm53_dsa", "glm53_kv_store",
                        ["buffer", "buffer", "buffer", "i64", "i64"],
                        [128, 1, 1], [S, 1, 1],
                        [a(0), a(1), a(2), i64(11264), i64(0)]),
        ]},
    }

    out["dsa_w_vc"] = {   # [T,8,512] x w_vc[8,256,512] -> [T,8,256] -> [T,2048]
        "params": [BF16X, BF16X, BF16O, "i32"],
        "impl": {"launches": _bmm8(256, 512, a_stride=4096, ldc=2048)},
    }

    out["dsa_quant_o"] = _quant(2048, 16, [cdiv(S, 2), 1, 1])
    out["dsa_o_proj"] = _fp8_gemm(FP8_N32_SUB, HID, 2048, 101696, 32)

    out["dsa_ar"] = {
        "params": ["inout buffer<bf16>"],
        "impl": {"launches": [
            allreduce_bf16(a(0), expr(mul(S, HID))),
        ]},
    }

    return out
