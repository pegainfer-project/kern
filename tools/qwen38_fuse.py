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


def prune(m):
    """Drop ops, modules and workspace buffers no program refers to any more."""
    used_ops = {c["op"] for p in m["programs"].values() for c in p["calls"]}
    m["ops"] = {k: v for k, v in m["ops"].items() if k in used_ops}
    used_bufs = {a["buf"] for p in m["programs"].values() for c in p["calls"] for a in c["args"] if "buf" in a}
    m["buffers"] = {k: v for k, v in m["buffers"].items() if k in used_bufs or v["kind"] != "workspace"}
    used_mods = {l["module"] for o in m["ops"].values() for l in o["impl"]["launches"] if "module" in l}
    m["modules"] = {k: v for k, v in m["modules"].items() if k in used_mods}
    return m


PASSES = {"gdn": fuse_gdn, "repin": repin_handwritten, "elem": elementwise}


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
