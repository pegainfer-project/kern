"""mHC (multi-stream Hyper-Connection) ops for the GLM-5.3 decode manifest.

Wiring is probe-verified (sglang wrapper callargs):

  prenorm(residual[T,16384] bf16, fn[24,16384] f32) -> mul[64,T,24], sqr[64,T]
  big_fuse64(mul, sqr, hc_scale, hc_base, residual) -> post[T,4], comb[T,16], x_norm[T,4096]
  fma(comb, residual, post, hidden_in, ffn_fn[24,4,4096] f32)
      -> cur_residual[T,4,4096], mul8[8,T,24], sqr8[8,T]
  big_fuse8(mul8, sqr8, ...) -> post, comb, x_norm
  mhc_post(comb, residual, post, x) -> out[T,4,4096]

D6 (arch_decode.md): the prenorm writes sqr_sum compact at pitch shape_m, so
call sites pass M = S_MAX and big_fuse64 T = S_MAX -- both partial tensors
then have the static pitch 16 and no repack exists. big_fuse8 keeps the
dynamic T = tokens (the fma's own partials are compact). Rows [S, S_MAX) of
every output are junk and never read (T7).

D7 (arch_decode.md): the TileLang kernels take their buffer params in
ALPHABETICAL order, not wrapper order (device_kernel.cu signatures):
  big_fuse: (comb_mix, gemm_out_mul, gemm_out_sqrsum, hc_base, hc_scale,
             layer_input, norm_weight, post_mix, residual, num_tokens)
  fma:      (cur_residual_out, hidden_in, mixes_partial_out, pre_fn,
             prev_comb_mix, prev_post_mix, prev_residual,
             sqrsum_partial_out, num_tokens, split_k)
The op param lists below keep the semantic (wrapper) order; the launch args
are permuted to the kernel order.
"""

from .ops_common import S, a, cdiv, extern, f32, i32, i64, mined, mined_file, mul, pack, tmap, var, handwritten  # noqa: F401

S_MAX = 16
HC = 4
H = 4096
HC_DIM = HC * H          # 16384
MIX = 24                 # 4 pre + 4 post + 16 comb
SPLITS = 64              # prenorm split-K (n_splits at T<=128)
FSPLIT = 8               # fma split_k at T<=8

# dump module variants (see ops_common docstring)
BIG_FUSE64 = "module_127.cubin"
BIG_FUSE8 = "module_231.cubin"

F32X = "in buffer<f32>"
F32O = "out buffer<f32>"
BF16X = "in buffer<bf16>"
BF16O = "out buffer<bf16>"


def ops():
    out = {}

    # --- hc_prenorm: residual x fn^T TF32 GEMM + square-sum partials
    # kernel ABI: (u32 M, tmap A, tmap B, tmap C, f32* sqrsum)
    #   A = residual [HC_DIM, T] bf16 stride 32768; B = fn [HC_DIM, 24] tf32 stride 65536;
    #   C = mul [24, S_MAX, 64] f32 strides (96, 96*S_MAX); sqrsum [64, S_MAX] f32.
    out["hc_prenorm"] = {
        "params": [BF16X, F32X, F32O, F32O, "i32"],
        "impl": {"launches": [
            mined("hc_prenorm_gemm",
                  ["i32", "bytes<128>", "bytes<128>", "bytes<128>", "buffer"],
                  [256, 1, 1], [SPLITS, 1, 1],
                  [a(4),
                   pack(128, tmap(0, "bf16", [HC_DIM, S_MAX], [HC_DIM * 2], [64, 64])),
                   pack(128, tmap(1, "tf32", [HC_DIM, MIX], [HC_DIM * 4], [32, 32])),
                   pack(128, tmap(2, "f32", [MIX, S_MAX, SPLITS], [MIX * 4, MIX * 4 * S_MAX], [32, 64, 1])),
                   a(3)],
                  smem=232448),
        ]},
    }

    # --- big_fuse: partials -> post/comb/layer_input (fused RMSNorm)
    # kernel ABI: 9 ptrs + i32 T in ALPHABETICAL order (D7); n_splits is a
    # baked constexpr (64 / 8). Call sites: fuse64 T = S_MAX, fuse8 T = tokens.
    def big_fuse(name, mod):
        out[name] = {
            "params": [F32X, F32X, F32X, F32X, BF16X, F32O, F32O, BF16O, BF16X, "i32"],
            "impl": {"launches": [
                mined_file(mod, "mhc_pre_big_fuse",
                           ["buffer"] * 9 + ["i32"],
                           [96, 1, 1], [S, 1, 1],
                           # comb_mix, mul, sqr, hc_base, hc_scale, layer_input,
                           # norm_weight, post_mix, residual, num_tokens
                           [a(6), a(0), a(1), a(3), a(2), a(7), a(8), a(5), a(4), a(9)],
                           smem=37232),
            ]},
        }
    big_fuse("hc_big_fuse64", BIG_FUSE64)
    big_fuse("hc_big_fuse8", BIG_FUSE8)

    # --- fma: deferred attn-post + ffn-pre GEMM partials (one launch)
    # kernel ABI: 8 ptrs + i32 T + i32 split_k in ALPHABETICAL order (D7)
    out["hc_fma"] = {
        "params": [F32X, BF16X, F32X, BF16X, F32X, F32O, F32O, BF16O, "i32", "i32"],
        "impl": {"launches": [
            mined("mhc_fused_post_pre_fma",
                  ["buffer"] * 8 + ["i32", "i32"],
                  [256, 1, 1], [S, 12, FSPLIT],
                  # cur_residual, hidden_in, mul8, pre_fn, comb, post,
                  # residual, sqr8, num_tokens, split_k
                  [a(7), a(3), a(5), a(4), a(0), a(2), a(1), a(6), a(8), a(9)],
                  smem=96),
        ]},
    }

    # --- mhc_post: out[j,h] = post[j]*x[h] + sum_k comb[k,j]*residual[k,h]
    out["hc_post"] = {
        "params": [F32X, BF16X, F32X, BF16X, BF16O, "i32"],
        "impl": {"launches": [
            mined("mhc_post_tilelang",
                  ["buffer"] * 5 + ["i32"],
                  [128, 1, 1], [S, 1, 1],
                  [a(i) for i in range(6)],
                  smem=28672),
        ]},
    }

    return out
