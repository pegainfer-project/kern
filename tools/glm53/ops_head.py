"""Embedding + tail ops for the GLM-5.3 decode manifest.

All GPU work is the handwritten kernels in kernels-glm53-handwritten/
glm53_misc.cubin (signatures from tools/glm53/kernels/glm53_misc.cu); the
lm_head GEMM and the vocab all-gather are externs.

  embed     glm53_embedding(ids i64 [S], emb bf16 [V,4096], out bf16 [S,4096],
            d=4096): one CTA per row, plain gather (replaces the mined Triton
            shard-masked embedding + the site-1 all-reduce; gen.py binds the
            weight accordingly).
  expand    glm53_hc_expand(x [S,4096] -> streams [S,4,4096]): broadcast the
            embedding output into the 4 mHC residual streams (grid.y = stream).
  contract  glm53_hc_contract(streams [S,4,4096] -> [S,4096]): unweighted
            f32 mean over the n=4 streams (tail, replaces the site-1345 aten
            MeanOps reduce).
  rms_norm  glm53_rms_norm(x, w, out, eps=1e-5, d): one CTA per row, block
            1024. Used by the final model.norm (d=4096, site 1346) AND reused
            by the DSA q_a (d=1536) / kv_a (d=512) norms -- one op, the d and
            eps scalars ride the interface (fold_constants keeps the literals).
  lm_head   cublasLt bf16 TN [S,4096] @ [4096,19360] -> shard [S,19360]
            (replaces site-1347 nvjet), then nccl all-gather of the 8 vocab
            shards into logits_full [8,S,19360] bf16. Two launches, one op.
  cast      glm53_gathered_logits_f32: fuse [rank,S,shard] -> [S,rank,shard]
            permutation with BF16->FP32 conversion; no extra launch or buffer.
  argmax    glm53_argmax_f32(logits f32 [S,154880], out i64 [S], n=154880):
            one CTA per row (replaces the site-1350 aten ArgMax reduce).
"""

from .ops_common import S, a, expr, extern, f32, i32, i64, handwritten, mul, rank, var  # noqa: F401

HID = 4096
VOCAB = 154880
VOCAB_SHARD = VOCAB // 8       # 19360 per TP rank
HC = 4                         # mHC streams

F32X = "in buffer<f32>"
F32O = "out buffer<f32>"
BF16X = "in buffer<bf16>"
BF16O = "out buffer<bf16>"
I64X = "in buffer<i64>"
I64O = "out buffer<i64>"


def ops():
    out = {}

    out["head_embed"] = {
        "params": [I64X, BF16X, BF16O, "i32"],
        "impl": {"launches": [
            handwritten("glm53_misc", "glm53_embedding",
                        ["buffer", "buffer", "buffer", "i32"],
                        [1024, 1, 1], [S, 1, 1],
                        [a(0), a(1), a(2), a(3)]),
        ]},
    }

    out["head_expand"] = {
        "params": [BF16X, BF16O, "i32"],
        "impl": {"launches": [
            handwritten("glm53_misc", "glm53_hc_expand",
                        ["buffer", "buffer", "i32"],
                        [1024, 1, 1], [S, HC, 1],
                        [a(0), a(1), a(2)]),
        ]},
    }

    out["head_contract"] = {
        "params": [BF16X, BF16O, "i32", "i32"],
        "impl": {"launches": [
            handwritten("glm53_misc", "glm53_hc_contract",
                        ["buffer", "buffer", "i32", "i32"],
                        [1024, 1, 1], [S, 1, 1],
                        [a(0), a(1), a(2), a(3)]),
        ]},
    }

    # also serves the DSA q_a (d=1536) / kv_a (d=512) norms -- see docstring
    out["head_rms_norm"] = {
        "params": [BF16X, BF16X, BF16O, "f32", "i32", "i32", "i32"],
        "impl": {"launches": [
            handwritten("glm53_misc", "glm53_rms_norm",
                        ["buffer", "buffer", "buffer", "f32", "i32", "i32", "i32"],
                        [1024, 1, 1], [S, 1, 1],
                        [a(0), a(1), a(2), a(3), a(4), a(5), a(6)]),
        ]},
    }

    out["head_lm_head"] = {
        "params": [BF16X, BF16X, BF16O, BF16O, "i32"],
        "impl": {"launches": [
            extern("cublaslt_bf16_tn",
                   [BF16X, BF16X, BF16O, "i32", "i32", "i32"],
                   [a(0), a(1), a(2), a(4), i32(VOCAB_SHARD), i32(HID)]),
            extern("nccl_allgather_bf16",
                   [BF16X, BF16O, "i64", "i32"],
                   [a(2), a(3), expr(mul(S, VOCAB_SHARD)), rank()]),
        ]},
    }

    out["head_cast"] = {
        "params": [BF16X, F32O],
        "impl": {"launches": [
            handwritten("glm53_misc", "glm53_gathered_logits_f32",
                        ["buffer", "buffer", "i32", "i32", "i32"],
                        [256, 1, 1], [256, 1, 1],
                        [a(0), a(1), var(S), i32(VOCAB_SHARD), i32(8)]),
        ]},
    }

    out["head_argmax"] = {
        "params": [F32X, I64O, "i32"],
        "impl": {"launches": [
            handwritten("glm53_misc", "glm53_argmax_f32",
                        ["buffer", "buffer", "i32"],
                        [1024, 1, 1], [S, 1, 1],
                        [a(0), a(1), a(2)]),
        ]},
    }

    return out
