"""TRT-LLM gen's ragged context FMHA (tools/kernels-bin/trtllm_fmha_ctx_h192_v128.cubin,
docs/k3-kernel-abi.md K13) as one manifest op: this rank's rows of a chunk against
the expanded K/V of their sequence, causal aligned to the end of the sequence.

The ABI below is data, not derivation: it is what tools/kernel-capture lifted
from FlashInfer's own launch (tools/trtllm-fmha/probe.py), field for field
against `KernelParams` (flashinfer/trtllm/fmha/kernelParams.h): four TMA
descriptors, then the pointers and scalars the kernel reads; everything the
launcher left zero stays zero. K and V are two views of one row: 192 k
elements then 128 v per head, 320 per head per token.
"""
MODULE = "trtllm_fmha_ctx_h192_v128"
ENTRY = "fmhaSm103aKernel_QkvBfloat16OBfloat16HQk192HV128SeparateQkvCausalVarSeqQ256Kv128PersistentContext"
HQK, HV = 192, 128
KV_ROW = HQK + HV
TILE_Q = 256
SMEM = 199296
PARAMS = 1344
# softmax scale 192^-0.5 * log2(e), rounded once to f32 as the launcher does
SCALE_LOG2 = 0.10411754995584488
# the multi-CTA partial buffers the persistent kernel is handed: stats first,
# partial O at the launcher's offset (sm count * tile * 8 B), unused at one CTA per KV
PARTIAL_O_OFFSET = 311296
SCRATCH_BYTES = 8 << 20


def tensormap(at, param, dims, strides, box):
    return {"at": at, "tensormap": {"param": param, "dtype": "bf16", "dims": dims, "strides": strides, "box": box,
                                    "swizzle": 128, "l2_promotion": 128}}


def op(heads, q_max, kv_max, module, q, scale_log2=SCALE_LOG2):
    """Interface: q (bf16 [rows, heads*192]) | k | v (the expanded rows, bf16 [kv_max, heads*320], v the same
    buffer 384 B in) | o (bf16 [rows, heads*128]) | seq_lens_kv | cum_q | cum_kv (i32 [1] / [2] / [2]) |
    stats | partial (u8 scratch). `q` is the rows' dimension (a var name or an expression), at most `q_max`;
    the unit batch and head-group dims keep a nonzero stride because the descriptor rank is the kernel's."""
    rows = {"var": q} if isinstance(q, str) else {"expr": q}
    ctas = {"expr": {"ceil_div": [q, TILE_Q]}}
    i32 = lambda at, v: {"at": at, "i32": v}
    kv = lambda at, param, d: tensormap(at, param, [d, kv_max, heads, 1], [heads * KV_ROW * 2, KV_ROW * 2, 16],
                                        [64, 128, 1, 1])
    fields = [
        tensormap(0, 0, [HQK, 1, heads, q_max], [HQK * 2, HQK * 2, heads * HQK * 2], [64, 1, 1, 128]),
        kv(128, 1, HQK),
        kv(384, 2, HV),
        tensormap(512, 3, [HV, q_max, heads, 1, 1], [heads * HV * 2, HV * 2, heads * HV * 2, 16], [64, 128, 1, 1, 1]),
        {"at": 912, "param": 3},   # ptrO
        {"at": 936, "param": 5},   # ptrCumSeqLensQ
        {"at": 944, "param": 6},   # ptrCumSeqLensKv
        {"at": 1024, "param": 8},  # ptrPartialO
        {"at": 1032, "param": 7},  # ptrPartialStats
        {"at": 1096, "param": 4},  # ptrSeqLensKv
        i32(1128, 0x7FFFFFFF),     # mAttentionWindowSize
        i32(1132, 1),              # mBatchSize
        {"at": 1172, **rows},      # mMaxSeqLenQ
        i32(1176, kv_max),         # mMaxSeqLenKv
        {"at": 1180, **ctas},      # mMaxNumCtasQ
        i32(1184, 1),              # mMaxNumCtasKv
        i32(1192, heads), i32(1196, heads), i32(1200, 1),
        i32(1204, 1), i32(1208, -(1 << 31)), i32(1216, -1),  # FastModDivInt32(1)
        {"at": 1224, "i64": heads * HQK},  # mNumHiddenEltsO
        i32(1236, TILE_Q),         # mNumTokensPerCtaQ
        i32(1240, -1),             # mNumTokensPerPageLog2: not paged
        i32(1244, 1),              # mReshapeFactorKv
        {"at": 1248, "f32": 1.0},  # mOutputScale
        {"at": 1252, "f32": scale_log2},
        {"at": 1260, "f32": 1.0},  # mScaleSfO
        {"at": 1276, **rows},      # mSumOfSeqLensQ
        i32(1280, kv_max),         # mSumOfSeqLensKv
    ]
    launch = {
        **module, "entry": ENTRY, "block": [512, 1, 1], "grid": [ctas["expr"], heads, 1], "shared_mem": SMEM,
        "params": [f"bytes<{PARAMS}>"], "args": [{"pack": {"size": PARAMS, "fields": fields}}],
    }
    return {
        "params": ["in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "in buffer<i32>",
                   "in buffer<i32>", "in buffer<i32>", "out buffer<u8>", "out buffer<u8>"],
        "impl": {"launches": [launch]},
    }
