"""Slab-3: absorb the post-MoE-verify spec_ar into the fused MoE (round_k2).

[moe_verify_v2 -> spec_ar] x 42 (verify layers 3..44) becomes ONE
moe_verify_v2_slab3 call per layer: route + w13 unchanged (champion cubins,
same sha256 as fused4), and the w2 launch becomes glm53_moe_v2_w2_ar
(fused4 w2 v6 body + per-row Lamport AR epilogue, ABI-compatible with
spec_ar_lamport). Call-count delta: -42 round_k2 calls.

The AR epilogue shares spec_ar_sym with the remaining 52 spec_ar calls:
one phase protocol (16 per-CTA replicas, flipped in lockstep), alternating
payload halves per call. fuse_slab3_manifest enlarges spec_ar_sym
4194368 -> 4194560 B; the +192 B tail holds the slab3 OutReady/ARDone
counters (self-resetting). No new buffers, no checkpoint changes, no
weight binds.

Numeric contract: bitwise identical to the [moe_verify_v2; spec_ar] chain
at every rows/validity shape (gate: 14 cases rows 1..32 x all-valid /
mixed-valid / all-invalid x 8 replays, 8-rank emulation; 2000-call
robustness; dead-rank -> qNaN + err=1+missing; shared-arena interop with
interleaved spec_ar calls). Decode/draft path (rows<=16) is untouched:
moe_decode_v2 keeps its separate spec_ar, so the champion decode path is
bitwise unchanged by construction.

Integration: run AFTER ops_moe_v2.fuse_round_manifest (needs moe_verify_v2
calls) and BEFORE lower_wire, same hook point. Escape hatch:
GLM53_SLAB3=0 makes fuse_slab3_manifest a no-op.

CPU build:  PYTHONPATH=tools python -m glm53.kernels.build_slab3
Manifest:   PYTHONPATH=tools python -c "from glm53.ops_slab3 import fuse_slab3_manifest as f; ..."
"""
import copy
import hashlib
import json
import os
from pathlib import Path

from .ops_common import a, scr, var, mul, i64, rank, handwritten
from . import ops_moe_v2

HERE = Path(__file__).resolve().parent

SLAB3_ARENA_BYTES = 4194560   # spec_ar_sym 4194368 + 192 B slab3 counter tail
SPEC_ARENA_BYTES = 4194368
TIMEOUT_NS = 1_000_000_000    # same watchdog as spec_ar_lamport


def _op_slab3():
    """moe_verify_v2_slab3 = champion moe_verify_v2 with w2 -> w2_ar.

    Built from ops_moe_v2.ops_verify() so route/w13 stay byte-identical to
    the fused4 champion (its build sha256 is re-asserted on every call).
    """
    base = copy.deepcopy(ops_moe_v2.ops_verify()["moe_verify_v2"])
    md = json.loads((HERE / "kernels/slab3_build.json").read_text())
    info = md["glm53_moe_v2_w2_ar"]
    # Kernel ABI: (H8, W, HS, WS, Weights, Sorted, Experts, NPost, Counts,
    # C2, Valid, Out, Ready, Rows, Sym, Peers, Err, Rank, Timeout).
    w2ar = handwritten(
        "glm53_moe_v2_w2_ar", info["entry"],
        ["buffer"] * 13 + ["i32"] + ["buffer"] * 3 + ["i32", "i64", "i64", "i64"],
        info["block"], [mul("tokens", 288), 1, 1],
        [scr("h8"), a(5), scr("hs"), a(6), scr("weights"),
         scr("sorted"), scr("experts"), scr("npost"), scr("w2_counts"), scr("c2"),
         a(7), a(8), scr("counts"), var("tokens"),
         a(9), a(10), a(11), rank("tp"), i64(TIMEOUT_NS), i64(0), i64(0)],
        smem=info["shared_mem"])
    assert w2ar["sha256"] == info["sha256"], "Rebuild slab3 artifacts (build_slab3.py)"
    # Programmatic edge from W13, same discipline as the fused4 w2 launch:
    # the per-tile ready-ticket spin is the synchronization; without a
    # PDL-aware launcher the path is serial-exact.
    w2ar["pdl"] = True
    base["params"] += ["inout buffer<u8>", "in buffer<u64>", "out buffer<i32>"]
    base["impl"]["launches"][2] = w2ar
    return base


def fuse_slab3_manifest(m, program="round_k2", expect=42):
    """Replace each [moe_verify_v2, spec_ar] pair with moe_verify_v2_slab3.

    Only pairs where spec_ar reduces the MoE output in place
    (spec_ar.args[0] == moe_verify_v2.args[8]) are fused; the other 52
    spec_ar calls (attention/draft ARs) keep the standalone op, which stays
    registered. spec_ar_sym is enlarged for the slab3 counter tail.
    Returns m for convenience; modifies only the caller-owned Python object.
    """
    if os.environ.get("GLM53_SLAB3", "1") == "0":
        return m
    assert "moe_verify_v2" in m["ops"], "run ops_moe_v2.fuse_round_manifest first"
    sym = m["buffers"]["spec_ar_sym"]
    assert sym["dtype"] == "u8" and sym["shape"] == [SPEC_ARENA_BYTES], \
        f"unexpected spec_ar_sym {sym}; slab3 measured against {SPEC_ARENA_BYTES}"
    sym["shape"] = [SLAB3_ARENA_BYTES]
    old = m["programs"][program]["calls"]
    new, i, fused = [], 0, 0
    while i < len(old):
        c = old[i]
        n = old[i + 1] if i + 1 < len(old) else None
        if c["op"] == "moe_verify_v2" and n is not None and n["op"] == "spec_ar" \
                and n["args"][0] == c["args"][8]:
            call = dict(c)
            call["op"] = "moe_verify_v2_slab3"
            if "label" in call:
                call["label"] = call["label"] + "_slab3"
            call["args"] = list(c["args"][:9]) + [n["args"][1], n["args"][2], n["args"][3]]
            new.append(call)
            fused += 1
            i += 2
            continue
        new.append(c)
        i += 1
    assert fused == expect, f"expected {expect} verify MoE+AR pairs, found {fused}"
    m["programs"][program]["calls"] = new
    m["ops"]["moe_verify_v2_slab3"] = _op_slab3()
    return m
