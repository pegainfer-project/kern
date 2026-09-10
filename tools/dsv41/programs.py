"""Assemble serving programs from lowered V4.1 call lists.

Stage builders emit ordinary op calls; stages are never runtime programs or
control flow. The serving loop sees only a chunk forward, a one-row forward,
and a six-row speculative round. Five draft rows remain internal to the round.
"""

from dataclasses import dataclass


def buf(name):
    return {"buf": name}


def integer(value):
    return {"i32": value}


def call(label, op, *args):
    return {"label": label, "op": op, "args": list(args)}


def prefixed(prefix, calls):
    return [dict(c, label=f"{prefix}.{c['label']}") for c in calls]


@dataclass(frozen=True)
class Stages:
    """Lowered calls, with all stage-specific shape/ABI choices resolved.

    `prefill` and `decode` include the target head and produce next_token.
    `verify` consumes verify_ids and produces verify_tokens, plus all three
    target taps. `draft` consumes five rows per sequence from draft_ids and
    produces five predictions per sequence in draft_tokens. Context updates
    and accept-state operations must honor the selected valid sequence extent.
    """

    load: list
    prefill: list
    decode: list
    draft: list
    verify: list
    prefill_context: list
    decode_context: list
    verify_context: list
    commit: list


def assemble(stages, *, max_seqs, block_size=5, noise_token=128799):
    """Return schema programs; no modules/ops placeholders enter this layer.

    Verification includes all draft predictions plus the anchor. Counts are
    greedy accepted-prefix lengths plus one target token, in [1, block+1].
    The caller advances by count and future writes replace rejected slots.
    Stateful providers additionally implement `commit` (for example selecting
    a versioned compressor state). A plain forward updates draft context too,
    so switching from plain to speculative steps preserves the context.
    """
    if max_seqs < 1 or block_size != 5:
        raise ValueError("V4.1 requires positive sequence capacity and a five-row draft")
    verify_rows = block_size + 1
    draft_rows = [
        call("splice_draft", "splice_draft", buf("anchor_token"),
             buf("draft_ids"), integer(block_size), {"i64": noise_token}),
    ] + prefixed("draft", stages.draft)
    verification = [
        call("splice_verify", "splice_verify", buf("anchor_token"),
             buf("draft_tokens"), buf("verify_ids"), integer(verify_rows),
             integer(block_size)),
    ] + prefixed("verify", stages.verify)
    accept = [
        call("count", "spec_count", buf("draft_tokens"), buf("verify_tokens"),
             buf("nacc"), integer(verify_rows), integer(block_size)),
    ]
    return {
        "load": {"once": True, "calls": stages.load},
        "prefill": {
            "batch": {"groups": 1, "rows": "tokens"},
            "calls": stages.prefill + prefixed("context", stages.prefill_context),
        },
        "decode_batch": {
            "batch": {"groups": max_seqs, "rows": 1}, "graph": True,
            "calls": stages.decode + prefixed("context", stages.decode_context),
        },
        "round": {
            "batch": {"groups": max_seqs, "rows": verify_rows}, "graph": True,
            "calls": draft_rows + verification + accept
            + prefixed("context", stages.verify_context)
            + prefixed("commit", stages.commit),
        },
    }


def spec_ops(module):
    """Existing spec_round.cu kernels, with no V4.1-specific device code."""
    signatures = {
        "splice_draft": ["in buffer<i64>", "out buffer<i64>", "i32", "i64"],
        "splice_verify": ["in buffer<i64>", "in buffer<i64>", "out buffer<i64>", "i32", "i32"],
        "spec_count": ["in buffer<i64>", "in buffer<i64>", "out buffer<i32>", "i32", "i32"],
    }
    return {
        name: {"params": params, "impl": {"launches": [{
            "module": module, "entry": f"kern_{name}",
            "block": [32, 1, 1], "grid": ["seqs", 1, 1],
        }]}}
        for name, params in signatures.items()
    }
