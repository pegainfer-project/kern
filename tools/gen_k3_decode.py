#!/usr/bin/env python3
"""Generate kern's Kimi-K3 (pruned, 224 experts) decode superstep at EP<R>.

    python3 tools/gen_k3_decode.py --layers 4 --ranks 1 > examples/k3-4l-ep1.json
    python3 tools/gen_k3_decode.py --ranks 4 > examples/k3-ep4.json
    python3 tools/gen_k3_decode.py --ranks 4 --tp 4 > examples/k3-ep4-tp4.json

One SPMD manifest per world: every rank runs the whole dense trunk on its own
batch of sequences and serves its expert shard to the world through MegaMoE
(tools/gen_k3_moe.py). Program `decode`: one token per sequence (`tokens` ==
`seqs`, up to --seqs) through all layers — attention-residual mix, KDA (conv +
delta rule, state in a `bytes_per_seq` line) or absorbed paged MLA (latent
cache in `kv`), latent MoE (router → down-proj → MegaMoE → norm → up-proj,
plus the shared experts) or the dense MLP — then the output mix, final norm,
lm_head and argmax into `next_token`.

`--tp R` makes the tray one batch (docs/multi-gpu.md "最终形态"): the `tp`
group's R ranks each own `tokens` rows and run the trunk on all `rows` ==
R * `tokens` rows in the "own rows first" layout (rank r's rows are rows
0..tokens of every row buffer, rank q's follow at block (q - r) mod R). The
ops that only work on their owner's rows — the attention with its paged /
per-sequence state, the expert dispatch — run on rows 0..tokens; their
outputs are all-gathered (tools/kernels-src/peer_collective.cu) and the
rest runs on `rows`. The KDA layers are head-sharded: every rank holds
HEADS / R heads of every row (weights from tools/shard_k3_tp.py, kernels
built for that width, the state line that many heads long), runs them on
all `rows`, and the o_proj partial is all-reduced. The dense FFN and the
shared expert are column-sharded the same way (gate / up rows, down
columns, the down partial all-reduced). Replicated on every rank: the
norms and scoring, the MLA projections, lat_down / lat_up, the router,
the LM head. The caller sets `rows` = R * `tokens` every run (kern does
not relate vars) and leases every row's KDA line on every rank.

The kernels are kern's own (docs/k3-kernel-abi.md, tools/kernels-src/k3_*.cu):
B is a runtime argument, every launch takes one row per block.x, and the
landing / residual / norm / append work is fused into its neighbours, so a
layer is 8 cuBLAS GEMMs (`extern:cublas_bf16_tn_f32`, f32 partials) plus a
dozen kernels. The launch sequence still follows pegainfer's certified
`k3_step` operand for operand; only the kernel boundaries moved.
"""
import argparse
import json
import math
import pathlib
import struct
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import flash_kda_abi
import gen_k3_moe
import handwritten
import kern_manifest

H = 7168
V = 163840
HEADS, HEAD_DIM = 96, 128
INNER = HEADS * HEAD_DIM           # 12288
Q_LORA, KV_LORA, ROPE = 1536, 512, 64
NOPE_DIM = 128  # per-head q/k dim before the rope part
KV_A = KV_LORA + ROPE              # 576
Q_B = HEADS * 192                  # 18432
MLA_FUSED = Q_LORA + KV_A + INNER  # 14400
KDA_FUSED = 4 * INNER              # 49152
WSM = 256                          # b_proj 96 | f_a 128 | pad
EXPERTS, TOPK, LATENT, INTER = 224, 16, 3584, 3072
SHARED = 2 * INTER                 # 6144
DENSE_I = 33792
ATTN_RES_BLOCK = 12
NB_MAX = 8
LAYERS = 93
MLA_LAYERS = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 92}

# KDA state line: recurrent f32 [96,128,128] then the three conv windows
# bf16 [3][12288] (q, k, v).
KDA_REC_BYTES = HEADS * HEAD_DIM * HEAD_DIM * 4
KDA_WIN_BYTES = 3 * INNER * 2
KDA_LINE_BYTES = KDA_REC_BYTES + 3 * KDA_WIN_BYTES
# MLA latent cache: 64-token pages of [mla_layers][64][576] bf16.
PAGE = 64
LATENT_ROW = KV_A

# MLA decode attention: FlashInfer's CuTe-DSL Blackwell kernel, prebuilt
# (tools/kernels-bin/README.md), two entries in one module. It walks a row
# in 128-token tiles; each row runs as `mla_bsk[b]` splits of a 2-CTA
# cluster and a reduction merges them. The parameter ABI packed below is
# the DSL's flattened struct layout (docs/k3-kernel-abi.md K5).
MLA_MODULE = "mla_decode_h96_p64"
MLA_MAIN = ("kernel_cutlass_split_kv_kernel_flashinfercute_dslattentionmonolithicmla_decode_fp16"
            "BlackwellMultiHeadLatentAttentionForwardFP16_object_at__TiledMMA_ThrLayoutVMNK21111000_PermutationMNK____0")
MLA_REDUCE = ("kernel_cutlass_reduction_kernel_flashinfercute_dslattentionmonolithicmla_decode_fp16"
              "BlackwellMultiHeadLatentAttentionForwardFP16_object_at__tensorptrbf16gmemalign16odiv16i64div161i64div16_1")
MLA_M_TILE = 128          # the MMA's row tile: 96 heads pad to one
MLA_MAIN_SMEM = 232448
MLA_REDUCE_SMEM = 1024    # 256-split reducer scratch

T = "tokens"
R = "rows"
SP = "span"
TP_GRID = 256
TP_AR_GRID = 152  # the GB300's SM count, a multiple of the cluster of 8 and under the 256-row flag table
TP_TIMEOUT_NS = 2_000_000_000

# Launch geometry per entry, as the kernel headers document it
# (docs/k3-kernel-abi.md §1). grid.x is always the batch.
GEOM = {
    "kern_k3_attnres_rms": ([T, 1, 1], [1024, 1, 1], 0),
    "kern_k3_land_add_attnres_rms": ([T, 1, 1], [1024, 1, 1], 0),
    "kern_k3_land_add2": ([T, 1, 1], [1024, 1, 1], 0),
    "kern_k3_conv_silu": ([T, 3, INNER // 512], [128, 1, 1], 0),  # 4 columns per thread
    "kern_k3_kda_core": ([T, HEADS, 1], [128, 1, 1], 0),
    "kern_k3_mla_prep": ([T, 4, 1], [512, 1, 1], 0),  # 1 norm/append block + 3 gate segments
    "kern_k3_mla_absorb": ([{"ceil_div": [T, 32]}, HEADS, 8], [128, 1, 1], 0),  # 32 rows x 64 columns per block
    "kern_k3_mla_vup_gate": ([{"ceil_div": [T, 32]}, HEADS, 4], [256, 1, 1], 0),  # 32 rows x 32 dv per block
    "kern_k3_mla_split_plan": ([1, 1, 1], [1024, 1, 1], 0),
    "kern_k3_router_topk": ([T, 1, 1], [256, 1, 1], 0),
    "kern_k3_argmax_f32_partial": ([T, 64, 1], [1024, 1, 1], 0),
    "kern_k3_argmax_f32_final": ([T, 1, 1], [64, 1, 1], 0),
    "kern_k3_rms": ([T, 1, 1], [1024, 1, 1], 0),
}


def is_mla(i):
    return i in MLA_LAYERS


def launch(cubin, entry, grid=None, block=None, smem=None, var=T, defines=None, **extra):
    """A launch of `entry`; `var` is the batch var its grid.x runs over,
    `defines` selects a variant build of the source (handwritten.hw)."""
    g, b, s = GEOM.get(entry, (None, None, 0))
    g = [var if d == T else d for d in (grid or g)]
    l = {**handwritten.hw(cubin, **(defines or {})), "entry": entry, "block": block or b, "grid": g, **extra}
    s = s if smem is None else smem
    if s:
        l["shared_mem"] = s
    return l


def bf16(x):
    """x rounded to bf16 (nearest even), as a float."""
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    return struct.unpack("<f", struct.pack("<I", ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) << 16))[0]


def fast_divmod(d):
    """The DSL's 32-bit FastDivmod divisor image: {divisor, multiplier, shift1, shift2},
    q = ((x - mulhi(x, m)) >> s1 + mulhi(x, m)) >> s2."""
    if d == 1:
        return [{"at": 0, "i32": 1}, {"at": 4, "i32": 1}]
    l = (d - 1).bit_length()
    m = ((1 << (32 + l)) + d - 1) // d - (1 << 32)
    return [{"at": 0, "i32": d}, {"at": 4, "i32": m - (1 << 32) if m >= 1 << 31 else m}, {"at": 8, "u8": 1},
            {"at": 9, "u8": l - 1}]


def pack(size, *fields):
    return {"pack": {"size": size, "fields": list(fields)}}


def mla_attn_op(seqs_max, page_stride, split_max):
    """The DSL attention as one op: split kernel + reduction, structs packed from the interface.
    Interface: q_abs latent | q_abs rope (+1024 B) | kv latent | kv rope (+1024 B) | block_table |
    seq_lens | mla_bsk | o_lat | lse | acc_o | acc_lse | B | max_pages."""
    V = {"at": 0, "var": T}
    tmap = lambda param, d0, page, box1, stride1: pack(128, {"at": 0, "tensormap": {
        "param": param, "dtype": "bf16", "dims": [d0, page, 0 if page == PAGE else seqs_max],
        "strides": [LATENT_ROW * 2, stride1], "box": [64, box1, 1], "swizzle": 128, "l2_promotion": 128}})
    at = lambda off, f: {**f, "at": off}
    q_stride, kv_stride = HEADS * LATENT_ROW * 2, page_stride * 2
    acc_o = pack(48, {"at": 0, "param": 9}, {"at": 8, "i32": MLA_M_TILE}, {"at": 12, "i32": split_max}, {"at": 16, "i32": KV_LORA},
                 {"at": 20, "i32": 1}, at(24, V), {"at": 28, "i32": split_max * KV_LORA}, {"at": 32, "i32": KV_LORA},
                 {"at": 36, "i32": split_max * MLA_M_TILE * KV_LORA}, {"at": 40, "i32": MLA_M_TILE * split_max * KV_LORA})
    acc_lse = pack(40, {"at": 0, "param": 10}, {"at": 8, "i32": MLA_M_TILE}, {"at": 12, "i32": split_max}, {"at": 16, "i32": 1},
                   at(20, V), {"at": 24, "i32": split_max}, {"at": 28, "i32": MLA_M_TILE * split_max},
                   {"at": 32, "i32": MLA_M_TILE * split_max})
    seqs = pack(16, {"at": 0, "param": 5}, {"at": 8, "var": T, "width": 8})
    bsk = pack(16, {"at": 0, "param": 6}, {"at": 8, "var": T, "width": 8})
    # the tiled-MMA descriptors and the TMA coordinate shapes are not read by this build; zero
    main_params = ["bytes<64>", "bytes<64>", "bytes<128>", "bytes<8>", "bytes<128>", "bytes<8>", "bytes<128>", "bytes<12>",
                   "bytes<128>", "bytes<12>", "bytes<128>", "bytes<12>", "bytes<24>", "bytes<48>", "bytes<24>", "bytes<48>",
                   "bytes<40>", "i32", "bytes<16>", "bytes<16>", "f32", "f32", "i32", "i32", "i32", "bytes<12>", "bytes<12>",
                   "bytes<12>"]
    main_args = [
        pack(64), pack(64),
        tmap(0, KV_LORA, HEADS, 64, q_stride), pack(8, {"at": 0, "i32": KV_LORA}, at(4, V)),
        tmap(1, ROPE, HEADS, 64, q_stride), pack(8, {"at": 0, "i32": ROPE}, at(4, V)),
        tmap(2, KV_LORA, PAGE, 64, kv_stride), pack(12, {"at": 0, "i32": PAGE}, {"at": 4, "i32": KV_LORA}),
        tmap(3, ROPE, PAGE, 64, kv_stride), pack(12, {"at": 0, "i32": PAGE}, {"at": 4, "i32": ROPE}),
        tmap(2, KV_LORA, PAGE, 32, kv_stride), pack(12, {"at": 0, "i32": KV_LORA}, {"at": 4, "i32": PAGE}),
        # page table [max_pages, B] strides (1, max_pages)
        pack(24, {"at": 0, "param": 4}, {"at": 8, "param": 12}, at(12, V), {"at": 16, "param": 12, "width": 8}),
        # o as (128, 512, tiles=1, B); lse as (128, 1, B): the split path never stores through these
        pack(48, {"at": 0, "param": 7}, {"at": 8, "i32": KV_LORA}, {"at": 12, "i32": 1}, at(16, V), {"at": 24, "i64": KV_LORA},
             {"at": 32, "i64": MLA_M_TILE * KV_LORA}, {"at": 40, "i64": HEADS * KV_LORA}),
        pack(24, {"at": 0, "param": 8}, {"at": 8, "i32": 1}, at(12, V), {"at": 16, "i64": HEADS}),
        acc_o, acc_lse,
        {"i32": split_max}, seqs, bsk,
        {"f32": bf16((NOPE_DIM + ROPE) ** -0.5) * math.log2(math.e)}, {"f32": 1.0},
        {"param": 11}, {"i32": 1}, {"i32": split_max},
        pack(12), pack(12, *fast_divmod(1)), pack(12, *fast_divmod(split_max)),
    ]
    reduce_params = ["bytes<48>", "bytes<40>", "bytes<48>", "bytes<40>", "i32", "bytes<16>", "bytes<16>"]
    reduce_args = [
        # o as (H, 512, S=1, B), lse as (H, 1, B)
        pack(48, {"at": 0, "param": 7}, {"at": 8, "i32": HEADS}, {"at": 12, "i32": KV_LORA}, {"at": 16, "i32": 1}, at(20, V),
             {"at": 24, "i64": KV_LORA}, {"at": 32, "i64": HEADS * KV_LORA}, {"at": 40, "i64": HEADS * KV_LORA}),
        pack(40, {"at": 0, "param": 8}, {"at": 8, "i32": HEADS}, {"at": 12, "i32": 1}, at(16, V), {"at": 24, "i64": HEADS},
             {"at": 32, "i64": HEADS}),
        acc_o, acc_lse, {"i32": split_max}, seqs, bsk,
    ]
    module = handwritten.prebuilt(MLA_MODULE)
    return {
        "params": ["in buffer<bf16>", "in buffer<bf16>", "in state", "in state", "in buffer<i32>", "in buffer<i32>",
                   "in buffer<i32>", "out buffer<bf16>", "out buffer<f32>", "out buffer<f32>", "out buffer<f32>", "i32", "i32"],
        "impl": {"launches": [
            {**module, "entry": MLA_MAIN, "params": main_params, "args": main_args, "block": [384, 1, 1],
             "grid": [2, T, split_max], "cluster": [2, 1, 1], "shared_mem": MLA_MAIN_SMEM},
            {**module, "entry": MLA_REDUCE, "params": reduce_params, "args": reduce_args, "block": [128, 1, 1],
             "grid": [HEADS, 1, T], "shared_mem": MLA_REDUCE_SMEM},
        ]},
    }


def build(layers, ranks, max_ctx, seqs_max, tp=1, mla_split_max=32, span_max=0):
    assert 1 <= layers <= LAYERS
    assert tp == 1 or ranks % tp == 0, "the tp group is a subset of the ep world"
    n_kda = sum(1 for i in range(layers) if not is_mla(i))
    mla_index = {i: k for k, i in enumerate(i for i in range(layers) if is_mla(i))}
    n_mla = len(mla_index)
    assert n_mla > 0, "the decode program needs at least one MLA layer (layer 3)"
    max_pages = -(-max_ctx // PAGE)
    page_stride = n_mla * PAGE * LATENT_ROW  # elements
    blocks_total = -(-layers // ATTN_RES_BLOCK)
    assert blocks_total <= NB_MAX
    mp = gen_k3_moe.mega_pieces(ranks, seqs_max)
    rows_max = tp * seqs_max
    # This rank's KDA heads: whole, or the tray group's shard (heads, the
    # per-head weights and the state line all HEADS / tp wide; the kernels
    # are built for that width, tools/shard_k3_tp.py cuts the weights).
    hl = HEADS // tp
    inner_l, fused_l = hl * HEAD_DIM, 4 * hl * HEAD_DIM
    line_l = hl * HEAD_DIM * HEAD_DIM * 4 + 3 * (3 * inner_l * 2)
    kda_defs = {"HEADS": hl} if tp > 1 else None
    # This rank's columns of the shared expert and the dense FFN.
    sh_l, dn_l = SHARED // tp, DENSE_I // tp
    # In a tray batch the KDA layers run on every row (their state is
    # head-sharded, every rank holds a slice of every row's); alone, rows
    # and tokens are the same number.
    KV = R if tp > 1 else T
    # The span (docs/roadmap.md K5): rows 0..span of a tray batch are one
    # sequence's prefill chunk, run through the KDA layers by FlashKDA
    # instead of the per-row recurrence; the decode rows stay on K2/K3.
    span_max = min(span_max, rows_max)

    def per_row(n):
        return [T, -(-n // 1024), 1]

    # Ops on all `rows` of the tray batch (var R) and ops on their owner's
    # rows only (var T); with tp == 1 the two are the same number.
    ops = {
        "embedding": {
            "params": ["in buffer<i64>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32"],
            "impl": {"launches": [launch("embedding", "kern_embedding_i64_bf16", grid=[T, 1, 1], block=[256, 1, 1], var=R)]},
        },
        "gemm_f32": {
            "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<f32>", "i32", "i32", "i32", "i32"],
            "impl": {"launches": [{"entry": "extern:cublas_bf16_tn_f32"}]},
        },
        # K1 residual stream
        "attnres_rms": {
            "params": ["in buffer<bf16>", "inout buffer<bf16>", "in buffer<f32>", "in buffer<bf16>", "out buffer<bf16>",
                       "i32", "i32", "i32"],
            "impl": {"launches": [launch("k3_residual", "kern_k3_attnres_rms", var=R)]},
        },
        # Layer 0: nb == 0 reads no snapshot, so `blocks` is a pure output there
        # (the verifier wants the first touch of a workspace to be a write).
        "attnres_rms_first": {
            "params": ["in buffer<bf16>", "out buffer<bf16>", "in buffer<f32>", "in buffer<bf16>", "out buffer<bf16>",
                       "i32", "i32", "i32"],
            "impl": {"launches": [launch("k3_residual", "kern_k3_attnres_rms", var=R)]},
        },
        "land_add_attnres_rms": {
            "params": ["in buffer<f32>", "in buffer<bf16>", "in buffer<bf16>", "in buffer<f32>", "in buffer<bf16>",
                       "out buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i32"],
            "impl": {"launches": [launch("k3_residual", "kern_k3_land_add_attnres_rms", var=R)]},
        },
        "land_add2": {
            "params": ["in buffer<f32>", "in buffer<f32>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32"],
            "impl": {"launches": [launch("k3_residual", "kern_k3_land_add2", var=R)]},
        },
        # K2 / K3 KDA
        "conv_silu": {
            "params": ["in buffer<f32>", "in buffer<f32>", "inout state", "in buffer<i32>", "i64",
                       "out buffer<bf16>", "out buffer<bf16>", "out buffer<bf16>", "i32", "in buffer<i32>", "i32"],
            "impl": {"launches": [launch("k3_conv_silu", "kern_k3_conv_silu", grid=[T, 3, inner_l // 512],
                                         var=KV, defines=kda_defs)]},
        },
        "kda_core": {
            "params": ["in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "in buffer<f32>", "in buffer<f32>",
                       "in buffer<bf16>", "in buffer<f32>", "in buffer<f32>", "in buffer<f32>",
                       "inout state", "in buffer<i32>", "i64", "out buffer<bf16>", "i32", "in buffer<i32>", "i32"],
            "impl": {"launches": [launch("k3_kda_core", "kern_k3_kda_core", grid=[T, hl, 1], var=KV,
                                         defines=kda_defs)]},
        },
        # K9 / K10 / K11 + K8: the span's KDA path
        **({
            "span_gather": {
                "params": ["in buffer<f32>", "in buffer<f32>", "inout state", "in buffer<i32>", "i64", "in buffer<f32>",
                           "out buffer<bf16>", "out buffer<bf16>", "out buffer<bf16>", "out buffer<bf16>",
                           "out buffer<bf16>", "in buffer<i32>", "i32"],
                "impl": {"launches": [launch("k3_span_gather", "kern_k3_span_gather",
                                             grid=[inner_l // 512, 4, {"ceil_div": [SP, 8]}], block=[128, 1, 1],
                                             defines=kda_defs)]},
            },
            "span_state_load": {
                "params": ["in state", "in buffer<i32>", "i64", "in buffer<i32>", "out buffer<f32>", "i32"],
                "impl": {"launches": [launch("k3_span_state", "kern_k3_span_state", grid=[hl, 32, 1],
                                             block=[128, 1, 1], defines=kda_defs)]},
            },
            "span_state_store": {
                "params": ["inout state", "in buffer<i32>", "i64", "in buffer<i32>", "in buffer<f32>", "i32"],
                "impl": {"launches": [launch("k3_span_state", "kern_k3_span_state", grid=[hl, 32, 1],
                                             block=[128, 1, 1], defines=kda_defs)]},
            },
            "gemm_bf16": {
                "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i32", "i32"],
                "impl": {"launches": [{"entry": "extern:cublaslt_bf16_tn"}]},
            },
            "flash_kda": flash_kda_abi.op(hl, span_max, handwritten.prebuilt(flash_kda_abi.MODULE), span=SP),
            "kda_out_gate": {
                "params": ["in buffer<bf16>", "in buffer<f32>", "in buffer<f32>", "inout buffer<bf16>", "in buffer<i32>",
                           "i32"],
                "impl": {"launches": [launch("k3_kda_out_gate", "kern_k3_kda_out_gate", grid=[SP, hl, 1],
                                             block=[128, 1, 1], defines=kda_defs)]},
            },
        } if span_max else {}),
        # K4 / K5 MLA
        "mla_prep": {
            "params": ["in buffer<f32>", "in buffer<bf16>", "in buffer<bf16>", "in buffer<i64>", "inout state",
                       "i64", "i64", "out buffer<bf16>", "out buffer<bf16>", "i32"],
            "impl": {"launches": [launch("k3_mla_prep", "kern_k3_mla_prep")]},
        },
        "mla_absorb": {
            "params": ["in buffer<f32>", "in buffer<bf16>", "out buffer<bf16>", "i32"],
            "impl": {"launches": [launch("k3_mla_absorb", "kern_k3_mla_absorb")]},
        },
        "mla_split_plan": {
            "params": ["in buffer<i32>", "out buffer<i32>", "i32", "i32"],
            "impl": {"launches": [launch("k3_mla_split_plan", "kern_k3_mla_split_plan")]},
        },
        "mla_attn": mla_attn_op(seqs_max, page_stride, mla_split_max),
        "mla_vup_gate": {
            "params": ["in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32"],
            "impl": {"launches": [launch("k3_mla_vup_gate", "kern_k3_mla_vup_gate")]},
        },
        # K6 / K7
        "router_topk": {
            "params": ["in buffer<f32>", "in buffer<f32>", "in buffer<bf16>", "out buffer<i32>", "out buffer<f32>", "i32"],
            "impl": {"launches": [launch("k3_router_argmax", "kern_k3_router_topk")]},
        },
        "argmax_f32": {
            "params": ["in buffer<f32>", "out buffer<i64>", "i32"],
            "impl": {
                "scratch": {"pmax": {"dtype": "f32", "shape": [R, 64]}, "pidx": {"dtype": "i32", "shape": [R, 64]}},
                "launches": [
                    launch("k3_router_argmax", "kern_k3_argmax_f32_partial", var=R,
                           params=["in buffer<f32>", "out buffer<f32>", "out buffer<i32>", "i32"],
                           args=[{"param": 0}, {"scratch": "pmax"}, {"scratch": "pidx"}, {"param": 2}]),
                    launch("k3_router_argmax", "kern_k3_argmax_f32_final", var=R,
                           params=["in buffer<f32>", "in buffer<i32>", "out buffer<i64>", "i32"],
                           args=[{"scratch": "pmax"}, {"scratch": "pidx"}, {"param": 1}, {"i32": 64}]),
                ],
            },
        },
        "rms": {
            "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32"],
            "impl": {"launches": [launch("k3_land", "kern_k3_rms", var=R)]},
        },
    }
    if tp > 1:
        # The tray-local all-gather (peer_collective.cu): one op per row dtype,
        # both over the same symmetric buffer and epoch carry.
        coll = ["inout buffer<u8>", "in buffer<u64>", "inout buffer<u32>", "out buffer<i32>",
                "i32", "i32", "i32", "i32", "i32", "i64"]
        for dt in ["f32", "bf16"]:
            ops[f"tp_allgather_{dt}"] = {
                "params": [f"in buffer<{dt}>", f"out buffer<{dt}>"] + coll,
                "impl": {"launches": [launch("peer_collective", "kern_peer_allgather",
                                             grid=[TP_GRID, 1, 1], block=[256, 1, 1])]},
            }
        # The allreduce is TensorRT-LLM's protocol (peer_allreduce.cu): one
        # token per cluster of 8 CTAs, one float4 per thread; the Lamport
        # stages are poisoned once by `tp_init` after the peers are imported.
        ops["tp_allreduce_f32"] = {
            "params": ["in buffer<f32>", "out buffer<f32>", "inout buffer<u8>", "in buffer<u64>",
                       "inout buffer<i32>", "in buffer<u64>", "inout buffer<u8>", "in buffer<u64>",
                       "inout buffer<i32>", "out buffer<i32>", "i32", "i32", "i32", "i64", "i32", "i64"],
            "impl": {"launches": [launch("peer_allreduce", "kern_peer_allreduce_f32", defines={"NRANKS": tp},
                                         grid=[TP_AR_GRID, 1, 1], block=[H // 4 // 8, 1, 1], cluster=[8, 1, 1])]},
        }
        ops["tp_lamport_init"] = {
            "params": ["inout buffer<u8>", "i64"],
            "impl": {"launches": [launch("peer_allreduce", "kern_peer_lamport_init", defines={"NRANKS": tp},
                                         grid=[256, 1, 1], block=[256, 1, 1])]},
        }
    # land / land_situ: grid.y depends on the width, one op per width.
    land_ops = {}

    def land_op(n):
        name = f"land_n{n}"
        if name not in ops:
            ops[name] = {
                "params": ["in buffer<f32>", "out buffer<bf16>", "i32", "i32", "i32", "i32"],
                "impl": {"launches": [launch("k3_land", "kern_k3_land", grid=per_row(n), block=[1024, 1, 1])]},
            }
        return name

    def situ_op(n):
        name = f"land_situ_n{n}"
        if name not in ops:
            ops[name] = {
                "params": ["in buffer<f32>", "out buffer<bf16>", "i32", "i32"],
                "impl": {"launches": [launch("k3_land", "kern_k3_land_situ", grid=per_row(n), block=[1024, 1, 1], var=R)]},
            }
        return name

    ops.update(mp["ops"])

    # ---- buffers
    buffers = {
        "token_ids": {"dtype": "i64", "shape": [R], "kind": "input", "fill": "token",
                      "domain": {"index_into": "embed"}},
        "slot_mapping": {"dtype": "i64", "shape": [T], "kind": "input", "fill": "slot",
                         "domain": {"index_into": "kv"}},
        "block_table": {"dtype": "i32", "shape": ["seqs", max_pages], "kind": "input",
                        "domain": {"index_into": "kv", "stride": PAGE}},
        "seq_lens": {"dtype": "i32", "shape": ["seqs"], "kind": "input", "fill": "seq_len", "domain": {"min": 1}},
        "kda.line_index": {"dtype": "i32", "shape": [n_kda, R], "kind": "input",
                           "domain": {"index_into": "kda", "stride": line_l}},
        "next_token": {"dtype": "i64", "shape": [R], "kind": "output", "fill": "tokens",
                       "domain": {"index_into": "embed"}},
        **mp["buffers"],
    }
    ag_region = seqs_max * H * 4 // 8  # packs: the widest gathered row is the f32 attention landing
    ar_stage = tp * rows_max * H * 4  # bytes: one Lamport stage, `tp` slots of the tray batch's f32 [rows, H]
    if tp > 1:
        buffers.update({
            "tp_sym": {"dtype": "u8", "shape": [2 * tp * ag_region * 16], "kind": "carry", "export": True},
            "tp_peers": {"dtype": "u64", "shape": [tp], "kind": "peer", "of": "tp_sym", "group": "tp"},
            "tp_epochs": {"dtype": "u32", "shape": [TP_GRID], "kind": "carry"},
            # peer_allreduce.cu: the two-shot copy + sum, the barrier flag table,
            # three Lamport stages, and the phase / stage / clear-size words.
            "tp_ar_comm": {"dtype": "u8", "shape": [2 * rows_max * H * 4], "kind": "carry", "export": True},
            "tp_ar_comm_peers": {"dtype": "u64", "shape": [tp], "kind": "peer", "of": "tp_ar_comm", "group": "tp"},
            "tp_ar_flags": {"dtype": "i32", "shape": [tp * 256], "kind": "carry", "export": True},
            "tp_ar_flag_peers": {"dtype": "u64", "shape": [tp], "kind": "peer", "of": "tp_ar_flags", "group": "tp"},
            "tp_ar_lamport": {"dtype": "u8", "shape": [3 * ar_stage], "kind": "carry", "export": True},
            "tp_ar_lamport_peers": {"dtype": "u64", "shape": [tp], "kind": "peer", "of": "tp_ar_lamport", "group": "tp"},
            "tp_ar_state": {"dtype": "i32", "shape": [8], "kind": "carry"},
            "tp_err": {"dtype": "i32", "shape": [1], "kind": "output", "fill": "error"},
        })
    states = {
        "kv": {"bytes_per_token": n_mla * LATENT_ROW * 2},
        "kda": {"bytes_per_seq": n_kda * line_l},
    }

    def weight(name, shape, dtype="bf16"):
        buffers[name] = {"dtype": dtype, "shape": list(shape), "kind": "weight"}

    b = lambda name, off=0: {"buf": name, "offset": off} if off else {"buf": name}

    def work(name, width, dtype="bf16", var=T):
        buffers[name] = {"dtype": dtype, "shape": [var, width], "kind": "workspace"}

    weight("embed", [V, H])
    weight("gamma_final", [H])
    weight("sw_out", [H], "f32")
    weight("w_lm", [V, H])
    for n in ["hidden", "prefix2", "normed"]:
        work(n, H, var=R)
    buffers["blocks"] = {"dtype": "bf16", "shape": [R, NB_MAX, H], "kind": "workspace"}
    work("hidden_partial", H, "f32")
    if tp > 1:
        work("hidden_partial_all", H, "f32", var=R)
        work("routed_latent_all", LATENT, var=R)
        work("o_partial", H, "f32", var=R)
        work("gated_kda", inner_l, var=R)
    work("kda_partial", fused_l, "f32", var=KV)
    work("wsm_partial", WSM, "f32", var=KV)
    for n in ["conv_q", "conv_k", "conv_v"]:
        work(n, inner_l, var=KV)
    for n in ["gated", "mla_gate"]:
        work(n, INNER)
    gated_kda = b("gated_kda") if tp > 1 else b("gated")
    # The span's first batch row, an input the KDA kernels skip past even
    # when there is no span (then `span` is 0 and the row is never read).
    buffers["span_at"] = {"dtype": "i32", "shape": [1], "kind": "input", "fill": "span_at"}
    if span_max:
        buffers["span_beta"] = {"dtype": "bf16", "shape": [hl, SP], "kind": "workspace"}
        for n in ["span_q", "span_k", "span_v", "span_out"]:
            work(n, inner_l, var=SP)
        work("span_flow", HEAD_DIM, var=SP)
        work("span_g", inner_l, var=SP)
        for n in ["span_state_in", "span_state_out"]:
            buffers[n] = {"dtype": "f32", "shape": [hl, HEAD_DIM, HEAD_DIM], "kind": "workspace"}
        buffers.update(flash_kda_abi.workspace_buffers(hl, span_max))
    work("mla_fused_partial", MLA_FUSED, "f32")
    work("q_norm", Q_LORA)
    work("q_partial", Q_B, "f32")
    work("q_abs", HEADS * LATENT_ROW)
    work("o_lat", HEADS * KV_LORA)
    work("mla_lse", HEADS, "f32")
    work("mla_acc_o", mla_split_max * MLA_M_TILE * KV_LORA, "f32")
    work("mla_acc_lse", mla_split_max * MLA_M_TILE, "f32")
    buffers["mla_bsk"] = {"dtype": "i32", "shape": [T], "kind": "workspace"}
    work("router_partial", EXPERTS, "f32")
    work("topk_idx", TOPK, "i32")
    work("topk_weight", TOPK, "f32")
    work("latent_partial", LATENT, "f32")
    for n in ["latent", "routed_latent"]:
        work(n, LATENT)
    work("routed_latent_norm", LATENT, var=R)
    work("routed_partial", H, "f32", var=R)
    work("shared_partial", 2 * sh_l, "f32", var=R)
    work("shared_act", sh_l, var=R)
    work("shared_partial2", H, "f32", var=R)
    work("dense_partial", 2 * dn_l, "f32", var=R)
    work("dense_act", dn_l, var=R)
    if tp > 1:
        work("mlp_all", H, "f32", var=R)
    work("logit_partial", V, "f32", var=R)

    # ---- program
    prog = []
    i32 = lambda v: {"i32": v}
    i64 = lambda v: {"i64": v}
    B = {"var": T}
    RB = {"var": R}

    def step(label, op, *args):
        prog.append({"label": label, "op": op, "args": list(args)})

    def gemm(label, a, w, c, n, k, ldc=None, m=B):
        """c[m, ldc] (cols 0..n from c's offset) = a[m, k] @ w[n, k]^T, f32."""
        step(label, "gemm_f32", a, w, c, m, i32(n), i32(k), i32(ldc or n))

    def land(label, p, o, n, off, ldc):
        step(label, land_op(n), p, o, i32(n), i32(off), i32(ldc), B)

    def land_situ(label, p, act, n):
        step(label, situ_op(n), p, act, i32(n), RB)

    def gathered(label, own, whole, dt, row_bytes):
        """The tray's rows of `own` (this rank's `tokens` rows): `whole` after
        the all-gather with tp > 1, `own` itself otherwise."""
        if tp == 1:
            return own
        step(label, f"tp_allgather_{dt}", own, whole, b("tp_sym"), b("tp_peers"), b("tp_epochs"), b("tp_err"),
             {"rank": "tp"}, i32(tp), B, i32(row_bytes), i32(ag_region), i64(TP_TIMEOUT_NS))
        return whole

    def reduced(label, partial, whole):
        """The tray group's sum of a head-sharded f32 [rows, H] partial."""
        step(label, "tp_allreduce_f32", partial, whole, b("tp_ar_comm"), b("tp_ar_comm_peers"), b("tp_ar_flags"),
             b("tp_ar_flag_peers"), b("tp_ar_lamport"), b("tp_ar_lamport_peers"), b("tp_ar_state"), b("tp_err"),
             {"rank": "tp"}, B, i32(H), i64(ar_stage), i32(0), i64(TP_TIMEOUT_NS))
        return whole

    def span_kda(L, w, line, KB, S):
        """The span rows' KDA layer: conv taps + beta/flow gathered (K9), the
        rec state staged (K10), g by one GEMM, FlashKDA over the chunk, the
        state written back, then the output gate in place (K11)."""
        step(L + "span_gather", "span_gather", b("kda_partial"), w("cw"), {"state": "kda"}, line, i64(line_l),
             b("wsm_partial"), b("span_q"), b("span_k"), b("span_v"), b("span_beta"), b("span_flow"), b("span_at"), S)
        step(L + "span_state_in", "span_state_load", {"state": "kda"}, line, i64(line_l), b("span_at"),
             b("span_state_in"), i32(0))
        step(L + "span_g", "gemm_bf16", b("span_flow"), w("w_f_b"), b("span_g"), S, i32(inner_l), i32(HEAD_DIM),
             i32(inner_l))
        step(L + "span_kda", "flash_kda", b("span_q"), b("span_k"), b("span_v"), b("span_g"), b("span_beta"),
             w("dt_bias"), w("a_log"), b("span_state_in"), b("span_state_out"), b("span_out"),
             *(b(n) for n in ["span_ws_kd", "span_ws_qd", "span_ws_kr", "span_ws_gt", "span_ws_inv", "span_ws_mqk"]),
             S)
        step(L + "span_state_out", "span_state_store", {"state": "kda"}, line, i64(line_l), b("span_at"),
             b("span_state_out"), i32(1))

    def span_out_gate(L, w, S):
        """The span rows of the layer's output, finished after K3 wrote the decode rows."""
        step(L + "span_out_gate", "kda_out_gate", b("span_out"), b("kda_partial"), w("gamma_o"), gated_kda,
             b("span_at"), S)

    def emit(span):
        """The decode program; with `span`, rows 0..span take the span's KDA path."""
        nonlocal prog
        prog = []
        S = {"var": SP} if span else i32(0)
        step("embed", "embedding", b("token_ids"), b("embed"), b("hidden"), RB, i32(H))
        step("mla_plan", "mla_split_plan", b("seq_lens"), b("mla_bsk"), i32(mla_split_max), B)

        blocks = 0
        kda_k = 0
        for i in range(layers):
            L = f"l{i}."
            w = lambda n, off=0, i=i: b(f"layers.{i}.{n}", off)
            snapshot = i % ATTN_RES_BLOCK == 0
            nb_in = blocks
            if snapshot:
                blocks += 1
            nb_mlp = blocks

            # residual mix in + snapshot + norm → normed
            step(L + "res_in", "attnres_rms" if nb_in > 0 else "attnres_rms_first", b("hidden"), b("blocks"),
                 w("sw_attn") if nb_in > 0 else w("sw_mlp"),
                 w("gamma_in"), b("normed"), i32(nb_in), i32(int(snapshot)), RB)
            if is_mla(i):
                k = mla_index[i]
                layer_off = k * PAGE * LATENT_ROW  # elements
                gemm(L + "wfu", b("normed"), w("wfu"), b("mla_fused_partial"), MLA_FUSED, H)
                step(L + "mla_prep", "mla_prep", b("mla_fused_partial"), w("gamma_q_a"), w("gamma_kv_a"), b("slot_mapping"),
                     {"state": "kv"}, i64(layer_off), i64(page_stride), b("q_norm"), b("mla_gate"), B)
                gemm(L + "q_b", b("q_norm"), w("w_q_b"), b("q_partial"), Q_B, Q_LORA)
                step(L + "absorb", "mla_absorb", b("q_partial"), w("w_kv_b"), b("q_abs"), B)
                step(L + "attn", "mla_attn", b("q_abs"), b("q_abs", KV_LORA * 2), {"state": "kv", "offset": layer_off * 2},
                     {"state": "kv", "offset": layer_off * 2 + KV_LORA * 2}, b("block_table"), b("seq_lens"), b("mla_bsk"),
                     b("o_lat"), b("mla_lse"), b("mla_acc_o"), b("mla_acc_lse"), B, i32(max_pages))
                step(L + "vup", "mla_vup_gate", b("o_lat"), w("w_kv_b"), b("mla_gate"), b("gated"), B)
            else:
                line = b("kda.line_index", kda_k * rows_max * 4)
                kda_k += 1
                KB = {"var": KV}
                gemm(L + "qkvg", b("normed"), w("wbig"), b("kda_partial"), fused_l, H, m=KB)
                gemm(L + "wsm", b("normed"), w("wsm"), b("wsm_partial"), WSM, H, m=KB)
                step(L + "conv", "conv_silu", b("kda_partial"), w("cw"), {"state": "kda"}, line, i64(line_l),
                     b("conv_q"), b("conv_k"), b("conv_v"), KB, b("span_at"), S)
                if span:
                    span_kda(L, w, line, KB, S)
                step(L + "kda_core", "kda_core", b("conv_q"), b("conv_k"), b("conv_v"), b("wsm_partial"), b("kda_partial"),
                     w("w_f_b"), w("dt_bias"), w("a_log"), w("gamma_o"), {"state": "kda"}, line, i64(line_l),
                     gated_kda, KB, b("span_at"), S)
                if span:
                    span_out_gate(L, w, S)
            if is_mla(i) or tp == 1:
                gemm(L + "o_proj", b("gated"), w("w_o"), b("hidden_partial"), H, INNER)
                attn_out = gathered(L + "gather_attn", b("hidden_partial"), b("hidden_partial_all"), "f32", H * 4)
            else:
                # Head-sharded o_proj on every row: each rank's slice of the sum.
                gemm(L + "o_proj", gated_kda, w("w_o"), b("o_partial"), H, inner_l, m=RB)
                attn_out = reduced(L + "reduce_attn", b("o_partial"), b("hidden_partial_all"))
            # attn_out landing + residual (or snapshot replace) + mix + norm → prefix2, normed
            step(L + "res_mlp", "land_add_attnres_rms", attn_out, b("hidden"), b("blocks"), w("sw_mlp"),
                 w("gamma_post"), b("prefix2"), b("normed"), i32(nb_mlp), i32(int(snapshot)), RB)

            # The dense FFN and the shared expert are column-sharded across the
            # tray (gate/up rows, down columns): the down projection's output is
            # a partial the group sums. lat_up stays replicated: its input is
            # a row of `routed_latent_norm`, and a K-split would need a
            # rank-dependent offset into it.
            if i == 0:
                gemm(L + "wgu", b("normed"), w("wgu"), b("dense_partial"), 2 * dn_l, H, m=RB)
                land_situ(L + "situ", b("dense_partial"), b("dense_act"), dn_l)
                gemm(L + "w_dn", b("dense_act"), w("w_dn"), b("routed_partial"), H, dn_l, m=RB)
                mlp = reduced(L + "reduce_mlp", b("routed_partial"), b("mlp_all")) if tp > 1 else b("routed_partial")
                step(L + "hidden", "land_add2", mlp, mlp, b("prefix2"), b("hidden"), i32(0), RB)
            else:
                gemm(L + "router", b("normed"), w("w_router"), b("router_partial"), EXPERTS, H)
                step(L + "topk", "router_topk", b("router_partial"), w("bias"), w("rs"), b("topk_idx"), b("topk_weight"), B)
                gemm(L + "lat_down", b("normed"), w("w_lat_down"), b("latent_partial"), LATENT, H)
                land(L + "latent", b("latent_partial"), b("latent"), LATENT, 0, LATENT)
                prog.extend(gen_k3_moe.mega_pieces(ranks, seqs_max, wprefix=f"layers.{i}.")["steps"](
                    b("latent"), b("topk_idx"), b("topk_weight"), b("routed_latent"), label=L))
                routed = gathered(L + "gather_moe", b("routed_latent"), b("routed_latent_all"), "bf16", LATENT * 2)
                step(L + "lat_norm", "rms", routed, w("gamma_lat"), b("routed_latent_norm"), i32(LATENT), RB)
                gemm(L + "lat_up", b("routed_latent_norm"), w("w_lat_up"), b("routed_partial"), H, LATENT, m=RB)
                gemm(L + "wsh", b("normed"), w("wsh"), b("shared_partial"), 2 * sh_l, H, m=RB)
                land_situ(L + "shared_situ", b("shared_partial"), b("shared_act"), sh_l)
                gemm(L + "sh_down", b("shared_act"), w("sh_down"), b("shared_partial2"), H, sh_l, m=RB)
                shared = reduced(L + "reduce_mlp", b("shared_partial2"), b("mlp_all")) if tp > 1 else b("shared_partial2")
                step(L + "hidden", "land_add2", b("routed_partial"), shared, b("prefix2"), b("hidden"), i32(1), RB)

            # weights
            weight(f"layers.{i}.gamma_in", [H])
            weight(f"layers.{i}.gamma_post", [H])
            if nb_in > 0:
                weight(f"layers.{i}.sw_attn", [H], "f32")
            weight(f"layers.{i}.sw_mlp", [H], "f32")
            if is_mla(i):
                weight(f"layers.{i}.wfu", [MLA_FUSED, H])
                weight(f"layers.{i}.gamma_q_a", [Q_LORA])
                weight(f"layers.{i}.gamma_kv_a", [KV_LORA])
                weight(f"layers.{i}.w_q_b", [Q_B, Q_LORA])
                weight(f"layers.{i}.w_kv_b", [HEADS * 256, KV_LORA])
            else:
                weight(f"layers.{i}.wbig", [fused_l, H])
                weight(f"layers.{i}.wsm", [WSM, H])
                weight(f"layers.{i}.w_f_b", [inner_l, HEAD_DIM])
                weight(f"layers.{i}.cw", [3, 4, inner_l], "f32")
                weight(f"layers.{i}.dt_bias", [inner_l], "f32")
                weight(f"layers.{i}.a_log", [hl], "f32")
                weight(f"layers.{i}.gamma_o", [HEAD_DIM], "f32")
            weight(f"layers.{i}.w_o", [H, INNER if is_mla(i) else inner_l])
            if i == 0:
                weight(f"layers.{i}.wgu", [2 * dn_l, H])
                weight(f"layers.{i}.w_dn", [H, dn_l])
            else:
                weight(f"layers.{i}.w_router", [EXPERTS, H])
                weight(f"layers.{i}.bias", [EXPERTS], "f32")
                weight(f"layers.{i}.rs", [1])
                weight(f"layers.{i}.w_lat_down", [LATENT, H])
                weight(f"layers.{i}.w_lat_up", [H, LATENT])
                weight(f"layers.{i}.gamma_lat", [LATENT])
                weight(f"layers.{i}.wsh", [2 * sh_l, H])
                weight(f"layers.{i}.sh_down", [H, sh_l])
                for n, d in mp["weights"].items():
                    buffers[f"layers.{i}.{n}"] = dict(d)

        assert blocks == blocks_total
        step("out.res", "attnres_rms", b("hidden"), b("blocks"), b("sw_out"), b("gamma_final"), b("normed"),
             i32(blocks_total), i32(0), RB)
        gemm("out.lm_head", b("normed"), b("w_lm"), b("logit_partial"), V, H, m=RB)
        step("out.argmax", "argmax_f32", b("logit_partial"), b("next_token"), i32(V))
        return prog

    groups = {"ep": ranks, **({"tp": tp} if tp > 1 else {})}
    # A decode step over the batch; with a span, the same step in which
    # rows [span_at, span_at + span) are one sequence's prompt chunk.
    programs = {"decode": kern_manifest.program(emit(False), groups=seqs_max, rows=1)}
    if span_max:
        programs["decode_span"] = kern_manifest.program(emit(True), groups=seqs_max, rows=1, span=SP)
    # Run once after the peers are imported: the Lamport stages must read
    # -0.0 before the first allreduce, and a carry starts at zero.
    if tp > 1:
        programs["tp_init"] = kern_manifest.program(
            [{"label": "tp_init", "op": "tp_lamport_init", "args": [b("tp_ar_lamport"), i64(3 * ar_stage)]}],
            once=True)
    m = {
        "schema_version": kern_manifest.SCHEMA_VERSION,
        "model": f"kimi-k3-pruned-75pct/{layers}l/ep{ranks}" + (f"-tp{tp}" if tp > 1 else ""),
        "vars": {T: {"max": seqs_max}, "seqs": {"max": seqs_max}, R: {"max": tp * seqs_max},
                 **({SP: {"max": span_max}} if span_max else {})},
        "topology": {"groups": groups},
        "states": states,
        "buffers": buffers,
        "ops": ops,
        "programs": programs,
    }
    return kern_manifest.normalize(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=LAYERS)
    ap.add_argument("--ranks", type=int, default=4)
    ap.add_argument("--max-ctx", type=int, default=16384)
    ap.add_argument("--seqs", type=int, default=64, help="sequences per rank (the `tokens`/`seqs` bound)")
    ap.add_argument("--tp", type=int, default=1, help="tray-batch group size (a divisor of --ranks)")
    ap.add_argument("--mla-split-max", type=int, default=32,
                    help="KV splits a row's attention may run as; the workspace is tokens x this x 256 KiB")
    ap.add_argument("--span-max", type=int, default=0,
                    help="rows a `decode_span` program may fill with one sequence's prefill chunk (0: no span program)")
    a = ap.parse_args()
    json.dump(build(a.layers, a.ranks, a.max_ctx, a.seqs, a.tp, a.mla_split_max, a.span_max), sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
