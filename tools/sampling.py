"""The `sample` op and the fills it reads, shared by the generators: a
step's token per sequence from the lm_head's bf16 logits, drawn by
FlashInfer's chain (tools/flashinfer-sampling/PROVENANCE.md) with the
glue of tools/kernels-src/sample_glue.cu around it, the argmax for a
sequence whose temperature is 0.

Per step, every parameter a `[seqs]` fill: argmax of the bf16 logits;
cast to f32; the chain's inputs prepared (temperature 0 -> 1, top-k out
of range -> vocab, the Philox offset from the sequence's length, the
radix row states zeroed); softmax / temperature, top-k renorm, top-p
renorm, min-p draw, all in place; then the pick. The radix top-k kernel is
multi-CTA with the geometry its host launcher derives for the GPU, so the
constants here are sm_103a's (152 SMs, 232448 B shared memory opt-in).
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from kernels.index import variant  # noqa: E402

BLOCK = 1024
VEC = 4
SMS = 152
# the largest chunk of a row one CTA's shared memory orders (probe.cu)
CHUNK_MAX = 50648
# sizeof(RadixRowState) in u32 words, the fixed shared memory of a radix
# CTA, and the other kernels' temp storage
STATE_WORDS = 776
RADIX_FIXED_SMEM = 2080
SMEM = {"softmax": 304, "topp": 324, "minp": 2080}


def fills():
    """The sampling fills, one entry per sequence of a call."""
    return {
        "temperature": {"dtype": "f32", "shape": ["seqs"], "kind": "input", "fill": "temperature"},
        "top_k": {"dtype": "i32", "shape": ["seqs"], "kind": "input", "fill": "top_k"},
        "top_p": {"dtype": "f32", "shape": ["seqs"], "kind": "input", "fill": "top_p"},
        "min_p": {"dtype": "f32", "shape": ["seqs"], "kind": "input", "fill": "min_p"},
        "seed": {"dtype": "i64", "shape": ["seqs"], "kind": "input", "fill": "seed"},
    }


def args(seq_lens="seq_lens", logits="logits", tokens="next_token"):
    """A call's args: the op's interface is (logits, seq_lens, the five fills, tokens, seqs)."""
    bufs = [logits, seq_lens, "temperature", "top_k", "top_p", "min_p", "seed", tokens]
    return [{"buf": b} for b in bufs] + [{"var": "seqs"}]


def ceil_div(a, b):
    return -(-a // b)


def radix(vocab):
    """CTAs per row, elements per CTA, row groups in flight and the dynamic shared memory of a CTA."""
    ctas = ceil_div(vocab, CHUNK_MAX)
    chunk = min(ceil_div(ceil_div(vocab, ctas), VEC) * VEC, CHUNK_MAX)
    return ctas, chunk, max(SMS // ctas, 1), RADIX_FIXED_SMEM + chunk * 4


def op(vocab, seqs_max):
    assert seqs_max <= BLOCK, f"the glue takes a call's sequences in one block of {BLOCK}"
    ctas, chunk, groups, radix_smem = radix(vocab)
    chain, glue, argmax = variant("flashinfer_sampling"), variant("sample_glue"), variant("argmax")

    def launch(v, name, params, block, grid, args, smem=None):
        entry = next(e for e in v.entries if name in e)
        s = {"entry": entry, "params": params, "block": [block, 1, 1], "grid": grid, "args": args, **v.module}
        if smem is not None:
            s["shared_mem"] = smem
        return s

    def p(i):
        return {"param": i}

    def scr(n):
        return {"scratch": n}

    def i32(v):
        return {"i32": v}

    def i64(v):
        return {"i64": v}

    def f32(v):
        return {"f32": v}

    S = "seqs"
    logits, seq_lens, temperature, top_k, top_p, min_p, seed, tokens, seqs = (p(i) for i in range(9))
    probs = scr("probs")
    return {
        "params": ["in buffer<bf16>", "in buffer<i32>", "in buffer<f32>", "in buffer<i32>", "in buffer<f32>",
                   "in buffer<f32>", "in buffer<i64>", "out buffer<i64>", "i32"],
        "impl": {
            "scratch": {
                "pmax": {"dtype": "f32", "shape": [S, 64]},
                "pidx": {"dtype": "i32", "shape": [S, 64]},
                "greedy": {"dtype": "i64", "shape": [S]},
                "probs": {"dtype": "f32", "shape": [S, vocab]},
                "chain_temperature": {"dtype": "f32", "shape": [S]},
                "chain_top_k": {"dtype": "i32", "shape": [S]},
                "offset": {"dtype": "u64", "shape": [S]},
                "states": {"dtype": "u32", "shape": [groups, STATE_WORDS]},
                "drawn": {"dtype": "i32", "shape": [S]},
                "valid": {"dtype": "u8", "shape": [S]},
            },
            "launches": [
                launch(argmax, "kern_argmax_partial_bf16",
                       ["in buffer<bf16>", "out buffer<f32>", "out buffer<i32>", "i32"],
                       1024, [S, 64, 1], [logits, scr("pmax"), scr("pidx"), i32(vocab)]),
                launch(argmax, "kern_argmax_final_i64",
                       ["in buffer<f32>", "in buffer<i32>", "out buffer<i64>", "i32"],
                       64, [S, 1, 1], [scr("pmax"), scr("pidx"), scr("greedy"), i32(64)]),
                launch(glue, "kern_sample_cast_bf16",
                       ["in buffer<bf16>", "out buffer<f32>", "i32"],
                       BLOCK, [S, ceil_div(vocab, BLOCK), 1], [logits, probs, i32(vocab)]),
                launch(glue, "kern_sample_prep",
                       ["in buffer<f32>", "in buffer<i32>", "in buffer<i32>", "out buffer<f32>", "out buffer<i32>",
                        "out buffer<u64>", "out buffer<u32>", "i32", "i32", "i32"],
                       BLOCK, [1, 1, 1],
                       [temperature, top_k, seq_lens, scr("chain_temperature"), scr("chain_top_k"), scr("offset"),
                        scr("states"), seqs, i32(vocab), i32(groups * STATE_WORDS)]),
                launch(chain, "OnlineSoftmaxFusedKernel",
                       ["inout buffer<f32>", "inout buffer<f32>", "in buffer<f32>", "f32", "i32"],
                       BLOCK, [S, 1, 1], [probs, probs, scr("chain_temperature"), f32(1.0), i32(vocab)],
                       SMEM["softmax"]),
                launch(chain, "RadixTopKRenormProbKernel_MultiCTA",
                       ["inout buffer<f32>", "inout buffer<f32>", "in buffer<i32>", "i32", "i32", "i32",
                        "inout buffer<u32>", "i32", "i32"],
                       BLOCK, [{"mul": [{"clamp": [S, 0, groups]}, ctas]}, 1, 1],
                       [probs, probs, scr("chain_top_k"), i32(0), i32(vocab), seqs, scr("states"), i32(chunk),
                        i32(ctas)],
                       radix_smem),
                launch(chain, "TopPRenormProbKernel",
                       ["inout buffer<f32>", "inout buffer<f32>", "in buffer<f32>", "f32", "i32"],
                       BLOCK, [S, 1, 1], [probs, probs, top_p, f32(0.0), i32(vocab)], SMEM["topp"]),
                # indices null (a row is its block), the scalar seed /
                # offset / min-p unread behind their arrays, strides 1
                launch(chain, "MinPSamplingFromProbKernel",
                       ["inout buffer<f32>", "in buffer<f32>", "out buffer<i32>", "out buffer<u8>", "i64", "f32",
                        "i32", "in buffer<i64>", "i64", "in buffer<u64>", "i64", "i32", "i32"],
                       BLOCK, [S, 1, 1],
                       [probs, min_p, scr("drawn"), scr("valid"), i64(0), f32(0.0), i32(vocab), seed, i64(0),
                        scr("offset"), i64(0), i32(1), i32(1)],
                       SMEM["minp"]),
                launch(glue, "kern_sample_select",
                       ["in buffer<f32>", "in buffer<i32>", "in buffer<i64>", "out buffer<i64>", "i32"],
                       BLOCK, [1, 1, 1], [temperature, scr("drawn"), scr("greedy"), tokens, seqs]),
            ],
        },
    }
