"""Fused decode kernels for the Qwen3.8-27B manifest.

Rewrites `examples/qwen3.8-27b.json` in place: each pass below swaps a
run of captured vLLM ops in the decode programs for one handwritten kernel
(tools/kernels-src/*.cu, pinned by sha256 through `handwritten.hw`). The
prefill program is left as captured. Every pass is a pure function of the
manifest; the numerics it must reproduce are written at the top of the
kernel source it introduces.

    python tools/qwen38_fuse.py [--input examples/qwen3.8-27b.json] [--output ...]

The cubins land in `target/cubins/`; copy them into the model's kernel
directory with `tools/extract_kernels.sh <manifest> target/cubins kernels-qwen38`.
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import handwritten  # noqa: E402
from kern_manifest import normalize  # noqa: E402

GDN_LINE_BYTES = 3211264      # one layer's state page: conv [3][10240] bf16, then [48][128][128] f32
GDN_REC_OFFSET = 61440
GDN_DIM = 10240
GDN_HV = 48
QKVZ_WIDTH = 16384
BA_WIDTH = 96
QKV_K_OFFSET = 24576         # bytes: k heads start after 24 x [q | gate]
QKV_V_OFFSET = 26624         # bytes: v heads after the 4 k heads
KV_BLOCK_STRIDE = 2097152    # elements between pages (16 layers x 64 tokens x 4 heads x [k | v] x 256)
KV_PAGE_STRIDE = 2048        # elements between tokens of a page
KV_HEAD_STRIDE = 512         # elements between heads ([k | v] x 256)
QKV_WIDTH = 14336
MLP_WIDTH = 17408
ATTN_WIDTH = 6144
GDN_SCALE = 0.0883883461356163
GDN_EPS = 9.999999974752427e-07


def gdn_ops():
    cubin = handwritten.hw("gdn_decode")
    conv = dict(
        params=["inout buffer<bf16>", "in buffer<bf16>", "inout state", "in buffer<i32>"],
        impl=dict(launches=[dict(
            **cubin, entry="kern_gdn_conv_bf16",
            params=["inout buffer<bf16>", "in buffer<bf16>", "inout state", "in buffer<i32>", "i32", "i32", "i64"],
            block=[256, 1, 1], grid=["seqs", GDN_DIM // 256, 1],
            args=[{"param": 0}, {"param": 1}, {"param": 2}, {"param": 3},
                  {"i32": GDN_DIM}, {"i32": QKVZ_WIDTH}, {"i64": GDN_LINE_BYTES}])]))
    step = dict(
        params=["in buffer<bf16>", "in buffer<bf16>", "in buffer<f32>", "in buffer<bf16>", "in buffer<bf16>",
                "out buffer<bf16>", "inout state", "in buffer<i32>"],
        impl=dict(launches=[dict(
            **cubin, entry="kern_gdn_step_bf16",
            params=["in buffer<bf16>", "in buffer<bf16>", "in buffer<f32>", "in buffer<bf16>", "in buffer<bf16>",
                    "out buffer<bf16>", "inout state", "in buffer<i32>", "f32", "f32", "i32", "i32", "i64"],
            block=[256, 1, 1], grid=[GDN_HV, "seqs", 1], shared_mem=128 * 128 * 4,
            args=[{"param": i} for i in range(8)] +
                 [{"f32": GDN_SCALE}, {"f32": GDN_EPS}, {"i32": QKVZ_WIDTH}, {"i32": BA_WIDTH}, {"i64": GDN_LINE_BYTES}])]))
    return {"gdn_conv": conv, "gdn_step": step}


def fuse_gdn(m):
    """conv_update + recurrent (+ z copy) + gated_norm -> gdn_conv + gdn_step."""
    m["ops"].update(gdn_ops())
    for name in ("decode", "decode_batch"):
        calls = m["programs"][name]["calls"]
        out = []
        i = 0
        while i < len(calls):
            c = calls[i]
            if c["op"] != "conv_update":
                out.append(c)
                i += 1
                continue
            layer = c["label"].split(".")[0]
            run = {}
            while i < len(calls) and calls[i]["label"].startswith(layer + ".") and \
                    calls[i]["op"] in ("conv_update", "recurrent", "copy_rows", "gated_norm_decode"):
                run[calls[i]["op"]] = calls[i]
                i += 1
            assert {"conv_update", "recurrent", "gated_norm_decode"} <= run.keys(), (name, layer, run.keys())
            cu, rec, gn = run["conv_update"], run["recurrent"], run["gated_norm_decode"]
            qkvz, w, state, line = cu["args"][0], cu["args"][1], cu["args"][2], cu["args"][3]
            assert cu["args"][4] == qkvz, "conv update is in place"
            a, b, a_log, dt_bias, core = rec["args"][1], rec["args"][2], rec["args"][3], rec["args"][4], rec["args"][5]
            assert a["buf"] == b["buf"] == "ba" and a.get("offset", 0) == BA_WIDTH and not b.get("offset")
            assert rec["args"][6] == rec["args"][7] == {**state, "offset": GDN_REC_OFFSET}
            assert gn["args"][0] == gn["args"][1] == core
            out.append(dict(label=layer + ".gdn_conv", op="gdn_conv", args=[qkvz, w, state, line]))
            out.append(dict(label=layer + ".gdn_step", op="gdn_step",
                            args=[qkvz, {"buf": "ba"}, a_log, dt_bias, gn["args"][2], core,
                                  {**state, "offset": GDN_REC_OFFSET}, line]))
        m["programs"][name]["calls"] = out
    return m


# Handwritten kernels rewritten since the capture (same entry, same ABI,
# bit-exact with the pinned build: see the harness notes in NOTES.md).
REWRITTEN = ["gemma_rms_norm", "sigmoid_mul"]
# Elementwise rewrites take 8 elements per thread: grid.y covers the row.
ELEM_BLOCK = 256


def attn_ops():
    cubin = handwritten.hw("attn_prep")
    prep = dict(
        params=["in buffer<bf16>", "in buffer<f32>", "in buffer<f32>", "in buffer<bf16>", "in buffer<bf16>",
                "out buffer<bf16>", "out buffer<bf16>", "inout state", "inout state", "in buffer<i64>", "i32"],
        impl=dict(launches=[dict(
            **cubin, entry="kern_attn_prep_bf16",
            params=["in buffer<bf16>", "in buffer<f32>", "in buffer<f32>", "in buffer<bf16>", "in buffer<bf16>",
                    "out buffer<bf16>", "out buffer<bf16>", "inout state", "inout state", "in buffer<i64>",
                    "i32", "f32", "i32", "i64", "i64", "i64"],
            block=[256, 1, 1], grid=["tokens", 1, 1],
            args=[{"param": i} for i in range(11)] +
                 [{"f32": GDN_EPS}, {"i32": QKV_WIDTH},
                  {"i64": KV_BLOCK_STRIDE}, {"i64": KV_PAGE_STRIDE}, {"i64": KV_HEAD_STRIDE}])]))
    return {"attn_prep": prep}


# Programs whose q_norm + k_norm + rope + kv_write become attn_prep.
ATTN_PROGRAMS = ("prefill", "decode", "decode_batch")


def fuse_attn(m):
    """q_norm + k_norm + rope + kv_write -> attn_prep."""
    m["ops"].update(attn_ops())
    for name in ATTN_PROGRAMS:
        calls = m["programs"][name]["calls"]
        out = []
        i = 0
        while i < len(calls):
            c = calls[i]
            if c["op"] != "gemma_norm_qhead":
                out.append(c)
                i += 1
                continue
            layer = c["label"].split(".")[0]
            run = {}
            while i < len(calls) and calls[i]["label"].startswith(layer + ".") and \
                    calls[i]["op"] in ("gemma_norm_qhead", "gemma_norm_khead", "mrope", "reshape_and_cache"):
                run[calls[i]["op"]] = calls[i]
                i += 1
            assert len(run) == 4, (name, layer, run.keys())
            qn, kn, rope, kv = run["gemma_norm_qhead"], run["gemma_norm_khead"], run["mrope"], run["reshape_and_cache"]
            qkv = qn["args"][1]
            assert not qkv.get("offset") and kn["args"][1] == {**qkv, "offset": QKV_K_OFFSET}
            assert rope["args"][:2] == [qn["args"][0], kn["args"][0]] and kv["args"][0] == kn["args"][0]
            assert kv["args"][1] == {**qkv, "offset": QKV_V_OFFSET}
            out.append(dict(label=layer + ".attn_prep", op="attn_prep",
                            args=[qkv, qn["args"][2], kn["args"][2], rope["args"][2], rope["args"][3],
                                  qn["args"][0], kn["args"][0], kv["args"][2], kv["args"][3], kv["args"][4],
                                  {"var": "tokens"}]))
        m["programs"][name]["calls"] = out
    return m


def elementwise(m):
    """The mined vLLM silu-and-mul becomes the handwritten one; the rewritten
    elementwise kernels get a grid that covers a row with 8 elements per
    thread."""
    silu = m["ops"]["silu_mul"]["impl"]["launches"][0]
    silu.pop("module")
    silu.update(handwritten.hw("silu_mul"), entry="kern_silu_mul_bf16",
                params=["out buffer<bf16>", "in buffer<bf16>", "i32"],
                args=[{"param": 0}, {"param": 1}, {"i32": MLP_WIDTH}],
                block=[ELEM_BLOCK, 1, 1], grid=["tokens", -(-MLP_WIDTH // (8 * ELEM_BLOCK)), 1])
    sig = m["ops"]["sigmoid_mul"]["impl"]["launches"][0]
    sig.update(block=[ELEM_BLOCK, 1, 1], grid=["tokens", -(-ATTN_WIDTH // (8 * ELEM_BLOCK)), 1])
    return m


def repin_handwritten(m):
    """Point every launch of a rewritten module at the current build of
    its source. The head norms (N = 256, ATen width <= 64) run 64-thread
    blocks: the kernel idles threads past the row's chunks."""
    for op in m["ops"].values():
        for launch in op["impl"]["launches"]:
            mod = m["modules"].get(launch.get("module"))
            if mod and mod["source"].removesuffix(".cubin") in REWRITTEN:
                launch.pop("module")
                launch.update(handwritten.hw(mod["source"].removesuffix(".cubin")))
    for name in ("gemma_norm_qhead", "gemma_norm_khead"):
        if name in m["ops"]:
            m["ops"][name]["impl"]["launches"][0]["block"] = [64, 1, 1]
    return m


# Decode attention splits. The converted manifest fixed 38 (2432 CTAs, one
# per SM at 145 KB of smem); a sweep at batch 16 on 32k contexts found the
# step fastest at 12-16 and 4% slower at 38 (NOTES.md has the table).
ATTN_SPLITS = 16


def attn_splits(m):
    """Regenerate the decode attention ops with ATTN_SPLITS split-KV CTAs."""
    import trtllm_attention
    for name in ("attn", "attn_batch"):
        if name in m["ops"]:
            m["ops"][name] = trtllm_attention.op("decode", max_rows=m["vars"]["tokens"]["max"],
                                                 max_seqs=m["vars"]["seqs"]["max"], max_context=262144,
                                                 splits=ATTN_SPLITS)
    return m


# cublasLt algorithms pinned by (tile id, split-K) for decode_batch GEMMs
# whose default heuristic is not the fastest: every non-split algorithm
# accumulates in the same order, so a pinned non-split tile keeps the bits
# (checked in a harness: 0 of 557k / 3.97M values differ) and the runtime
# refuses a pin it cannot find. Standalone at M = 16: gate_up 56.8 -> 54.3
# µs (tile 487 -> 312), lm_head 364.7 -> 357.9 (tile 394 -> 312).
GEMM_TILES = {".gate_up": (312, 1), "lm_head": (312, 1)}


def gemm_tiles(m):
    """decode_batch GEMMs named in GEMM_TILES run the pinned algorithm."""
    for (suffix, (tile, splitk)) in GEMM_TILES.items():
        name = f"gemm_tile{tile}_{splitk}"
        m["ops"][name] = dict(
            params=list(m["ops"]["gemm"]["params"]),
            impl=dict(launches=[dict(
                entry="extern:cublaslt_bf16_tn_tile",
                params=list(m["ops"]["gemm"]["params"]) + ["i32", "i32"],
                args=[{"param": i} for i in range(6)] + [{"i32": tile}, {"i32": splitk}])]))
        for c in m["programs"]["decode_batch"]["calls"]:
            if c["op"] == "gemm" and c["label"].endswith(suffix):
                c["op"] = name
    return m


# decode_batch GEMMs on the handwritten tiled-layout kernel
# (tools/kernels-src/gemm16_tiled.cu). The weight gets a second copy on the
# device in the kernel's tile order (`layout` on a `weight` buffer bound to
# the same `tensor`); prefill and the single-sequence decode keep cuBLAS on
# the row-major original. Per shape: (N, K, split-K count at M <= 8 and at
# M > 8 — cuBLASLt's own choices, which the kernel reproduces bit for bit —
# and the pipeline depth measured fastest). down_proj (5120 x 17408, 13 or
# 17 splits) stays on cuBLAS: 37.3 vs 33.3 µs standalone.
TILED_GEMMS = {   # label suffix: (op name, N, K, splits at M <= 8, splits at M > 8, stages)
    ".in_proj_qkvz": ("gemm_tiled_qkvz", QKVZ_WIDTH, 5120, 1, 1, 8),
    ".qkv_proj": ("gemm_tiled_qkv", QKV_WIDTH, 5120, 2, 2, 6),
    ".out_proj": ("gemm_tiled_out", 5120, ATTN_WIDTH, 5, 5, 6),
    ".o_proj": ("gemm_tiled_o", 5120, ATTN_WIDTH, 5, 5, 6),
    ".gate_up": ("gemm_tiled_gate_up", 2 * MLP_WIDTH, 5120, 1, 1, 4),
    "lm_head": ("gemm_tiled_lm_head", 248320, 5120, 1, 1, 6),
}
TILE = {"tile": [64, 64], "swizzle": 8}


def gemm_tiled(m):
    """decode_batch GEMMs named in TILED_GEMMS run the tiled kernel on a
    tiled copy of their weight."""
    cubin = handwritten.hw("gemm16_tiled")
    for suffix, (name, n, k, lo, hi, stages) in TILED_GEMMS.items():
        smax = max(lo, hi)
        m["ops"][name] = dict(
            params=["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32"],
            impl=dict(
                scratch={"partials": {"dtype": "f32", "shape": [smax * 16, n]},
                         "arrivals": {"dtype": "i32", "shape": [n // 64]}},
                launches=[dict(
                    **cubin, entry=f"kern_gemm16_tiled_s{stages}_bf16",
                    params=["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "out buffer<f32>",
                            "out buffer<i32>", "i32", "i32", "i32", "i32", "i32"],
                    block=[128, 1, 1], grid=[n // 64, smax, 1], shared_mem=stages * 10240 + 1024,
                    args=[{"param": 0}, {"param": 1}, {"param": 2}, {"scratch": "partials"}, {"scratch": "arrivals"},
                          {"param": 3}, {"i32": n}, {"i32": k}, {"i32": lo}, {"i32": hi}])]))
        for c in m["programs"]["decode_batch"]["calls"]:
            if not c["op"].startswith("gemm") or not c["label"].endswith(suffix):
                continue
            w = c["args"][1]["buf"]
            assert m["buffers"][w]["shape"] == [n, k], (c["label"], m["buffers"][w]["shape"])
            m["buffers"][w + ".tiled"] = dict(dtype="bf16", shape=[n, k], kind="weight", tensor=w, layout=dict(TILE))
            c["op"] = name
            c["args"] = [c["args"][0], {"buf": w + ".tiled"}, c["args"][2], c["args"][3]]
    return m


def prune(m):
    """Drop ops, modules and workspace buffers no program refers to any more."""
    used_ops = {c["op"] for p in m["programs"].values() for c in p["calls"]}
    m["ops"] = {k: v for k, v in m["ops"].items() if k in used_ops}
    used_bufs = {a["buf"] for p in m["programs"].values() for c in p["calls"] for a in c["args"] if "buf" in a}
    m["buffers"] = {k: v for k, v in m["buffers"].items() if k in used_bufs or v["kind"] in ("input", "output")}
    used_mods = {l["module"] for o in m["ops"].values() for l in o["impl"]["launches"] if "module" in l}
    m["modules"] = {k: v for k, v in m["modules"].items() if k in used_mods}
    return m


PASSES = {"gdn": fuse_gdn, "repin": repin_handwritten, "attn": fuse_attn, "elem": elementwise, "splits": attn_splits,
          "tiles": gemm_tiles, "tiled": gemm_tiled}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=pathlib.Path, default=pathlib.Path("examples/qwen3.8-27b.json"))
    p.add_argument("--output", type=pathlib.Path)
    p.add_argument("--passes", default=",".join(PASSES), help="comma-separated subset, in this order")
    args = p.parse_args()
    m = json.loads(args.input.read_text())
    for name in args.passes.split(","):
        m = PASSES[name](m)
    m = prune(m)
    (args.output or args.input).write_text(json.dumps(normalize(m), indent=1) + "\n")


if __name__ == "__main__":
    main()
