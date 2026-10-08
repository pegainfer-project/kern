"""KDA (Kimi Delta Attention) ops for the GLM-5.3 decode manifest.

Probe-verified wiring (sglang wrapper callargs + recipe site 6..12):
qkvbfg: x_norm[S,4096] @ W[3336,4096]^T -> fused[S,3336] (cublasLt).
fg_b: f_a=fused[:,3080:3208] @ f_b[1024,128]^T -> forget[S,1024];
g_a=fused[:,3208:3336] @ g_b[1024,128]^T -> gproj[S,1024] (a_stride 3336).
conv: causal_conv1d_update(x=fused[:,:3072] (row stride 3336), w[3072,4] f32,
conv_state line via indices, out=conv_out[S,3072], silu).
delta: fused_sigmoid_gating_delta_rule(q|k|v views of conv_out, a=forget,
b=fused[:,3072:3080], A_log[8], dt_bias[1024], ssm line via indices,
cu_seqlens, lower_bound=-5, scale=128^-0.5) -> o[S,8,128].
o_norm: layer_norm_gated(x=o, g=gproj, y=o in place, w=o_norm[128], rstd,
eps=1e-5, T=S*8, RMS+sigmoid).
o_proj: o[S,1024] @ W[4096,1024]^T -> partial[S,4096]; TP all-reduce (selected by gen.py).

The state line tables (kda_conv_lines / kda_ssm_lines [34, S_MAX] i32) are
host-filled per state; each kernel gets its own state's table row for the
layer. All strides are layout constants (3336 / 3072 / 1024 / 131072), valid
at every batch size (the delta kernel's T-dim stride is never dereferenced
past row 0 for a size-1 extent).
"""

from .ops_common import (S, a, cdiv, extern, f32, i32, i64, mined, mined_file,  # noqa: F401
                         mul, var, handwritten, allreduce_bf16)

# D4: module_473 bakes num_cache_lines = 581 (any slot >= 17 silently loses
# its conv history). module_conv_generic is the same Triton source recompiled
# offline with num_cache_lines = 2**31-1; TTIR diff vs the captured variant is
# exactly the one constant (tools/glm53/remine_conv.py).
CONV_UPDATE = "module_conv_generic.cubin"
# D5: module_399's T carries tt.divisibility=16 (violated at odd S, T=8*S);
# module_227's TTIR body is identical with a plain T.
NORM_GATED = "module_227.cubin"

F32X = "in buffer<f32>"
F32O = "out buffer<f32>"
BF16X = "in buffer<bf16>"
BF16O = "out buffer<bf16>"
I32X = "in buffer<i32>"

QKVBFG_N = 3336
HID = 4096
QKV_W = 3072
B_OFF = 3072
FA_OFF = 3080
GA_OFF = 3208
FG_N = 1024
FG_K = 128
SSM_STRIDE = 131072
QSCALE = 0.08838834764831845


def ops():
    out = {}

    out["kda_qkvbfg"] = {
        "params": [BF16X, BF16X, BF16O, "i32", "i32", "i32"],
        "impl": {"launches": [
            extern("cublaslt_bf16_tn", [BF16X, BF16X, BF16O, "i32", "i32", "i32"],
                   [a(0), a(1), a(2), a(3), a(4), a(5)]),
        ]},
    }

    out["kda_fg_b"] = {
        "params": [BF16X, BF16X, BF16X, BF16O, BF16O, "i32"],
        "impl": {"launches": [
            extern("cublaslt_bf16_tn",
                   [BF16X, BF16X, BF16O, "i32", "i32", "i32", "i32", "i32"],
                   [a(0, FA_OFF * 2), a(1), a(3), a(5), i32(FG_N), i32(FG_K), i32(FG_N), i32(QKVBFG_N)]),
            extern("cublaslt_bf16_tn",
                   [BF16X, BF16X, BF16O, "i32", "i32", "i32", "i32", "i32"],
                   [a(0, GA_OFF * 2), a(2), a(4), a(5), i32(FG_N), i32(FG_K), i32(FG_N), i32(QKVBFG_N)]),
        ]},
    }

    out["kda_conv"] = {
        "params": [BF16X, F32X, "inout state", I32X, BF16O, "i32"],
        "impl": {"launches": [
            handwritten("module_conv_generic", "_causal_conv1d_update_kernel",
                       ["buffer", "buffer", "buffer", "buffer", "buffer", "buffer", "i32", "i64", "i64"],
                       [128, 1, 1], [S, 12, 1],
                       [a(0), a(1), a(2), a(3), a(0), a(4), a(5), i64(0), i64(0)]),
        ]},
    }

    out["kda_delta"] = {
        "params": [F32X, BF16X, F32X, BF16X, BF16X, BF16O, "inout state", I32X, I32X, "i32"],
        "impl": {"launches": [
            mined("fused_sigmoid_gating_delta_rule",
                  ["buffer", "buffer", "buffer", "f32", "f32", "f32",
                   "buffer", "buffer", "buffer", "buffer", "buffer",
                   "buffer", "buffer", "i32", "buffer", "i32", "f32", "i32",
                   "i32", "i32", "i32", "i32", "i32", "i64", "i64"],
                  [32, 1, 1], [4, S, 8],
                  [a(0), a(1), a(2), f32(1.0), f32(20.0), f32(-5.0),
                   a(3, 0), a(3, 2048), a(3, 4096), a(4, B_OFF * 2), a(5),
                   a(6), a(7), i32(SSM_STRIDE), a(8), i32(0), f32(QSCALE), a(9),
                   i32(FG_N), i32(QKV_W), i32(QKV_W), i32(QKV_W), i32(QKVBFG_N),
                   i64(0), i64(0)],
                  smem=64),
        ]},
    }

    out["kda_norm_gated"] = {
        "params": ["inout buffer<bf16>", BF16X, BF16X, F32O, "i32"],
        "impl": {"launches": [
            mined_file(NORM_GATED, "layer_norm_gated_fwd_kernel",
                       ["buffer", "buffer", "buffer", "buffer", "buffer", "f32", "i32", "i64", "i64"],
                       [128, 1, 1], [cdiv(mul(S, 8), 32), 1, 1],
                       [a(0), a(1), a(0), a(2), a(3), f32(1e-5), a(4), i64(0), i64(0)],
                       smem=256),
        ]},
    }

    # op-level param 2 is OUT (not inout): the cublasLt launch writes sub_out
    # before the fused all-reduce reads it; no prior contents are consumed.
    out["kda_o_proj_ar"] = {
        "params": [BF16X, BF16X, BF16O, "i32"],
        "impl": {"launches": [
            extern("cublaslt_bf16_tn", [BF16X, BF16X, BF16O, "i32", "i32", "i32"],
                   [a(0), a(1), a(2), a(3), i32(HID), i32(FG_N)]),
            allreduce_bf16(a(2), {"expr": mul(S, HID)}),
        ]},
    }

    return out
