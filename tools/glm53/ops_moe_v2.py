"""Three-launch decode MoE (M<=16) and MTP-verify MoE (M<=32).

Keep ops_moe for dense MLP, AR and rollback.

v3 (fused4): same three launches, new cubins. Universal 32-pair align tiles
(rows<=16 tiling is unchanged, so the decode/draft path stays bitwise
identical; rows 17..32 no longer re-read expert shards), order-free
shared-ticket ordinals in the route kernel, transposed as/hs layouts so the
W13/W2 main-loop scale loads coalesce. Scratch shapes below are the v3 ones.

Decode integration: fuse_manifest(m) on the RAW gen.py manifest, before
wire_allreduce, resolve_modules and lower_wire.
Verify integration: fuse_round_manifest(m) right after
ops_spec.extend_manifest (which resolves its own modules), before
lower_wire. The same three cubins serve both; the route kernel sizes its
clear/fill loops from the runtime rows argument and gives each expert
ceil(count/16) consecutive 16-pair align tiles (one at rows<=16, at most
two at rows<=32), so W13/W2 tile-indexed kernels are unchanged.
No checkpoint conversion or new weight binds.
"""
import hashlib
import json
import os
from pathlib import Path
from .ops_common import a, scr, var, mul, i32, i64, rank, handwritten, HAND_DIR, DUMP_DIR

HERE = Path(__file__).resolve().parent


def _resolved(name, sha):
    """gen.resolve_modules for one handwritten artifact, after it already ran."""
    fname = f"{name}.cubin"
    matches = [p.resolve() for d in (DUMP_DIR, HAND_DIR)
               if (p := d / fname).is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == sha]
    assert len(matches) == 1, f"{fname}: expected one artifact with pinned sha256 {sha}"
    return str(matches[0])


def _op(rows_max, varname, resolve):
    md = json.loads((HERE / "kernels/moe_v2_build.json").read_text())

    def launch(name, grid, args, params):
        info = md[name]
        op = handwritten(name, info["entry"], params, info["block"], grid, args,
                         smem=info["shared_mem"])
        assert op["sha256"] == info["sha256"], "Rebuild all v2 artifacts together"
        if resolve:
            op["cubin"] = _resolved(name, info["sha256"])
        if name == "glm53_moe_v2_w2":
            # Programmatic edge from W13: W2 starts while W13 runs and waits
            # per tile on the ready count (8 tickets + 1 epilogue release).
            # The spin is the synchronization; without a PDL-aware launcher
            # the count is already 9 at start and the path is serial-exact.
            op["pdl"] = True
        return op

    pairs = rows_max * 9
    scratch = {
        "weights": ("f32", [rows_max, 9]), "ids": ("i32", [rows_max, 9]),
        "sorted": ("i32", [pairs, 32]), "experts": ("i32", [pairs]),
        "npost": ("i32", [1]), "a8": ("fp8e4m3", [rows_max, 4096]),
        "as": ("f32", [32, 32]), "counts": ("i32", [pairs]),
        "scores": ("f32", [rows_max, 288]), "route_counter": ("i32", [1]),
        "c1": ("bf16", [pairs, 512]), "h8": ("fp8e4m3", [pairs, 256]),
        "hs": ("f32", [2, 288]), "w2_counts": ("i32", [rows_max, 32]),
        "c2": ("bf16", [pairs, 4096]),
    }
    scratch = {k: {"dtype": dtype, "shape": shape} for k, (dtype, shape) in scratch.items()}
    launches = [
        launch("glm53_moe_v2_route", [96, 1, 1],
               [a(0), a(1), a(2), a(7)] +
               [scr(n) for n in ("weights", "ids", "sorted", "experts", "npost", "a8", "as", "counts", "scores", "route_counter", "w2_counts")] + [var(varname)],
               ["buffer"] * 15 + ["i32"]),
    ]
    if os.environ.get("GLM53_ROUTE_DUMP") == "1":
        # Diagnostic builds only: dump route ids/n_post from ep rank 0 as one
        # RDUMP stdout line per call. Never enable for timing manifests.
        launches.append(handwritten(
            "glm53_route_dump", "glm53_route_dump",
            ["buffer", "buffer", "i32", "i32", "i32"],
            [32, 1, 1], [1, 1, 1],
            [scr("ids"), scr("npost"), var(varname),
             i32(1 if rows_max == 32 else 0), rank("ep")]))
    launches += [
        launch("glm53_moe_v2_w13", [mul(varname, 72), 1, 1],
               [scr("a8"), a(3), scr("as"), a(4)] +
               [scr(n) for n in ("sorted", "experts", "npost", "counts", "c1", "h8", "hs")] +
               [var(varname), i64(0), i64(0)],
               ["buffer"] * 11 + ["i32", "i64", "i64"]),
        launch("glm53_moe_v2_w2", [mul(varname, 288), 1, 1],
               [scr("h8"), a(5), scr("hs"), a(6), scr("weights")] +
               [scr(n) for n in ("sorted", "experts", "npost", "w2_counts", "c2")] +
               [a(7), a(8), scr("counts"), var(varname), i64(0), i64(0)],
               ["buffer"] * 13 + ["i32", "i64", "i64"]),
    ]
    return {
        "params": ["in buffer<bf16>", "in buffer<bf16>", "in buffer<f32>",
                   "in buffer<fp8>", "in buffer<f32>", "in buffer<fp8>",
                   "in buffer<f32>", "in buffer<i32>", "out buffer<bf16>"],
        "impl": {
            "scratch": scratch,
            "launches": launches,
        },
    }


def ops():
    return {"moe_decode_v2": _op(16, "seqs", resolve=False)}


def ops_verify():
    """MTP verify-path MoE op; integrate after ops_spec.extend_manifest."""
    return {"moe_verify_v2": _op(32, "tokens", resolve=True)}


def fuse_manifest(m):
    """Replace exactly the nine baseline calls per MoE layer, not the AR.

    The manifest must still use gen.py shorthand. Refuse instrumented MoE
    cuts: a probe inside the old nine-call region has no v2 counterpart.
    Returns m for convenience; modifies only the caller-owned Python object.
    """
    order = ["moe_router", "moe_topk", "moe_align", "moe_quant_a", "moe_w13",
             "moe_silu", "moe_quant_b", "moe_w2", "moe_sum_reduce"]
    assert m["vars"]["seqs"]["max"] <= 16 and m["vars"]["tokens"]["max"] <= 16
    assert m["programs"]["decode"]["batch"]["rows"] == 1
    assert "moe_decode_v2" not in m["ops"], "Already fused"
    old = m["programs"]["decode"]["calls"]
    new = []
    i = 0
    while i < len(old):
        if old[i]["op"] != order[0]:
            new.append(old[i]); i += 1; continue
        chunk = old[i:i+9]
        assert [c["op"] for c in chunk] == order, "Unexpected MoE calls/probes; disable --probes"
        r, t, _, _, w13, _, _, w2, reduce = chunk
        call = dict(r)
        call["op"] = "moe_decode_v2"
        if "label" in call:
            call["label"] = call["label"].removesuffix("router") + "moe_v2"
        call["args"] = [r["args"][0], r["args"][1], t["args"][1],
                        w13["args"][1], w13["args"][4], w2["args"][1], w2["args"][4],
                        {"buf": "valid"}, reduce["args"][1]]
        new.append(call)
        i += 9
    m["programs"]["decode"]["calls"] = new
    used = {c["op"] for p in m["programs"].values() for c in p["calls"]}
    for name in order:
        if name not in used:
            m["ops"].pop(name, None)
    if "moe_decode_v2" in used:
        m["ops"].update(ops())
    used_buffers = {a["buf"] for p in m["programs"].values() for c in p["calls"]
                    for a in c["args"] if "buf" in a}
    for name in list(m["buffers"]):
        if name.startswith("moe_") and name not in used_buffers:
            del m["buffers"][name]
    return m


def fuse_draft_manifest(m, program="round_k2", expect=2, valid_buf="spec_ones"):
    """Replace the nine mtp16_moe_* calls per draft step with moe_decode_v2.

    Runs after fuse_round_manifest (same hook point, before lower_wire).
    Draft MoE is the unmasked M<=16 shape: the unfused mtp16 path computes
    all rows=seqs with no validity input, so the fused call reads spec_ones,
    the manifest's own per-seq flag (spec_uniform writes it from the tray
    valid before draft0 runs). draft_valid is a cache-write gate consumed by
    the draft DSA ops; no MoE call reads it, so it does not participate here.

    The op is the decode champion's moe_decode_v2 (rows<=16, var('seqs')):
    added resolved when the moe==v1 decode did not create it. The v1 moe_*
    workspace buffers are global manifest buffers shared by ops (safe via
    stream serialization); sweep the ones no remaining call uses.
    """
    order = ["moe_router", "moe_topk", "moe_align", "moe_quant_a", "moe_w13",
             "moe_silu", "moe_quant_b", "moe_w2", "moe_sum_reduce"]
    pref = ["mtp16_" + n for n in order]
    assert m["vars"]["seqs"]["max"] <= 16
    vb = m["buffers"][valid_buf]
    assert vb["dtype"] == "i32" and vb["shape"] == [16]
    old = m["programs"][program]["calls"]
    new = []
    i = 0
    fused = 0
    while i < len(old):
        if old[i]["op"] != pref[0]:
            new.append(old[i]); i += 1; continue
        chunk = old[i:i+9]
        assert [c["op"] for c in chunk] == pref, "Unexpected draft MoE calls/probes"
        r, t, _, _, w13, _, _, w2, reduce = chunk
        assert t["args"][0] == r["args"][2], "topk input must be the router scores"
        call = dict(r)
        call["op"] = "moe_decode_v2"
        if "label" in call:
            call["label"] = call["label"].removesuffix("router") + "moe_v2"
        call["args"] = [r["args"][0], r["args"][1], t["args"][1],
                        w13["args"][1], w13["args"][4], w2["args"][1], w2["args"][4],
                        {"buf": valid_buf}, reduce["args"][1]]
        new.append(call)
        fused += 1
        i += 9
    assert fused == expect, f"expected {expect} draft MoE blocks, found {fused}"
    m["programs"][program]["calls"] = new
    used = {c["op"] for p in m["programs"].values() for c in p["calls"]}
    for name in pref:
        if name not in used:
            m["ops"].pop(name, None)
    if "moe_decode_v2" in used and "moe_decode_v2" not in m["ops"]:
        m["ops"]["moe_decode_v2"] = _op(16, "seqs", resolve=True)
    used_buffers = {a["buf"] for p in m["programs"].values() for c in p["calls"]
                    for a in c["args"] if "buf" in a}
    for name in list(m["buffers"]):
        if name.startswith("moe_") and name not in used_buffers:
            del m["buffers"][name]
    return m


def fuse_round_manifest(m, program="round_k2", expect=42):
    """Replace the nine mtp32_moe_* calls per verify layer with moe_verify_v2.

    Runs after ops_spec.extend_manifest (which already resolved the mtp32
    modules) and before lower_wire. Draft passes keep their mtp16 ops; the
    verify all-reduce (spec_ar) and the baseline v1 MoE workspace buffers
    stay: draft still uses them, so no buffer sweep is needed here.
    Returns m for convenience; modifies only the caller-owned Python object.
    """
    order = ["moe_router", "moe_topk", "moe_align", "moe_quant_a", "moe_w13",
             "moe_silu", "moe_quant_b", "moe_w2", "moe_sum_reduce"]
    pref = ["mtp32_" + n for n in order]
    assert m["vars"]["tokens"]["max"] <= 32
    assert m["buffers"]["valid"]["shape"] == ["tokens"]
    assert "moe_verify_v2" not in m["ops"], "Already fused"
    old = m["programs"][program]["calls"]
    new = []
    i = 0
    fused = 0
    while i < len(old):
        if old[i]["op"] != pref[0]:
            new.append(old[i]); i += 1; continue
        chunk = old[i:i+9]
        assert [c["op"] for c in chunk] == pref, "Unexpected verify MoE calls/probes"
        r, t, _, _, w13, _, _, w2, reduce = chunk
        assert t["args"][0] == r["args"][2], "topk input must be the router scores"
        # lower_wire has not run: var/expr launch scalars are not lifted onto
        # calls yet, so only fixed-position buffer args are safe to read here.
        call = dict(r)
        call["op"] = "moe_verify_v2"
        if "label" in call:
            call["label"] = call["label"].removesuffix("router") + "moe_v2"
        call["args"] = [r["args"][0], r["args"][1], t["args"][1],
                        w13["args"][1], w13["args"][4], w2["args"][1], w2["args"][4],
                        {"buf": "valid"}, reduce["args"][1]]
        new.append(call)
        fused += 1
        i += 9
    assert fused == expect, f"expected {expect} verify MoE blocks, found {fused}"
    m["programs"][program]["calls"] = new
    used = {c["op"] for p in m["programs"].values() for c in p["calls"]}
    for name in pref:
        if name not in used:
            m["ops"].pop(name, None)
    if "moe_verify_v2" in used:
        m["ops"].update(ops_verify())
    return m
