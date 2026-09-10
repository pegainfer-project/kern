"""FP32 greedy target head and sequential five-step DSpark Markov chain.

Both projections use the existing BF16-input / FP32-output cuBLAS built-in.
The bias is added in FP32 inside the argmax, matching forward_head's add_
without rounding logits to BF16. Confidence does not control greedy acceptance.
"""
from pathlib import Path
import hashlib
import copy

from .programs import call, buf, integer


def offset(name, byte_offset):
    return {"buf": name, "offset": byte_offset}


def definitions(cubin, *, seqs="seqs"):
    cubin = Path(cubin)
    modules = {"dsv41_head": {"source": str(cubin),
                             "sha256": hashlib.sha256(cubin.read_bytes()).hexdigest()}}
    def launch(entry, params, args, grid, block):
        return {"module": "dsv41_head", "entry": entry, "params": params,
                "args": args, "grid": grid, "block": [block, 1, 1]}
    p = lambda i: {"param": i}
    scratch = lambda name: {"scratch": name}
    ops = {
        "head_project": {
            "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<f32>",
                       "i32", "i32", "i32"],
            "impl": {"launches": [{"entry": "extern:cublas_bf16_tn_f32"}]},
        },
        "head_embedding": {
            "params": ["in buffer<i64>", "i32", "in buffer<bf16>",
                       "out buffer<bf16>", "i32"],
            "impl": {"launches": [{
                "module": "dsv41_head", "entry": "kern_embedding_rows_i64_bf16",
                "grid": [seqs, 1, 1], "block": [256, 1, 1],
            }]},
        },
        "head_greedy": {
            "params": ["in buffer<f32>", "i64", "in buffer<f32>", "i64",
                       "out buffer<i64>", "i32", "i32"],
            "impl": {
                "scratch": {name: {"dtype": dtype, "shape": [seqs, 64]}
                            for name, dtype in (("max", "f32"), ("idx", "i32"))},
                "launches": [
                    launch("kern_argmax_rows_partial_f32_bias",
                           ["in buffer<f32>", "i64", "in buffer<f32>", "i64",
                            "out buffer<f32>", "out buffer<i32>", "i32"],
                           [p(0), p(1), p(2), p(3), scratch("max"), scratch("idx"), p(6)],
                           [seqs, 64, 1], 1024),
                    launch("kern_argmax_rows_final_i64",
                           ["in buffer<f32>", "in buffer<i32>", "out buffer<i64>", "i32", "i32"],
                           [scratch("max"), scratch("idx"), p(4), p(5), integer(64)],
                           [seqs, 1, 1], 64),
                ],
            },
        },
    }
    plain = copy.deepcopy(ops["head_greedy"])
    plain["params"] = ["in buffer<f32>", "out buffer<i64>", "i64", "i32"]
    first, final = plain["impl"]["launches"]
    first["params"][2] = "i64"
    first["args"] = [p(0), p(2), {"i64":0}, {"i64":0}, scratch("max"), scratch("idx"), p(3)]
    final["args"] = [scratch("max"), scratch("idx"), p(1), integer(1), integer(64)]
    ops["head_plain"] = plain
    return modules, ops


def markov_calls(*, vocab, hidden=5120, rank=256, seqs="seqs",
                 normalized="draft.head_hidden", logits="draft.logits",
                 anchor="anchor_token", output="draft_tokens",
                 head="head.weight", embed="mtp.2.markov_head.embed.weight",
                 markov_head="mtp.2.markov_head.head.weight",
                 workspace="draft.markov_embed", bias="draft.markov_bias"):
    """Every prediction is conditioned on the previously sampled token.

    normalized/logits contain five contiguous rows per sequence. The Markov
    workspace only contains one row per sequence and is overwritten each step.
    """
    rows = {"mul": [seqs, 5]}
    calls = [call("head.project", "head_project", buf(normalized), buf(head),
                  buf(logits), {"expr": rows}, integer(vocab), integer(hidden))]
    for step in range(5):
        prev = buf(anchor) if step == 0 else offset(output, (step - 1) * 8)
        calls += [
            call(f"markov{step}.embed", "head_embedding", prev,
                 integer(1 if step == 0 else 5), buf(embed), buf(workspace), integer(rank)),
            call(f"markov{step}.project", "head_project", buf(workspace), buf(markov_head),
                 buf(bias), {"var": seqs}, integer(vocab), integer(rank)),
            call(f"markov{step}.sample", "head_greedy", offset(logits, step * vocab * 4),
                 {"i64": 5 * vocab}, buf(bias), {"i64": vocab},
                 offset(output, step * 8), integer(5), integer(vocab)),
        ]
    return calls
