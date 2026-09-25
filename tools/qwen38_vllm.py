"""The Qwen3.8-27B manifest on vLLM's memory.

Rewrites `examples/qwen3.8-27b.json` into a manifest a vLLM model runner
drives (`python/kern_vllm`): the KV cache and the GDN states become host
states, one per layer, laid out the way vLLM's hybrid allocator lays them
out, and the programs stop at the hidden states, since vLLM samples. The
tables index them as vLLM's ids do: 64-token blocks of every attention
layer, one page per sequence of every GDN layer, so a tool that is its own
host (`kern bench`) can provision them.

vLLM gives every layer of a hybrid model one page per block, the same size
for the attention and the GDN groups: a GDN line (conv [3][10240] bf16,
then [48][128][128] f32) rounded up to whole 64-token attention pages
(TRTLLM-GEN's P64), 832 tokens. An attention layer's view of it is 13
kernel blocks of [64 tokens][4 heads][k | v] (vLLM's LBNHC); a GDN layer's
is one line at the start of the page. Only strides change: the kernels
already take them as arguments or tensormap fields, except the captured
prefill conv, which baked the line stride in and is replaced by
`gdn_conv_fwd`, pinned by sha; its source and cubin are published with
the manifest in `Pegainfer/kern-qwen38-sm103` (`sources/`, `cubins/`).

    python tools/qwen38_vllm.py [--input examples/qwen3.8-27b.json] --output qwen3.8-27b-vllm.json [--sampled]

`--sampled` keeps the base's head and sampling after the final norm: vLLM
never runs it, `kern test` gates on it.
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from kern_manifest import normalize, resolve_constants  # noqa: E402
from qwen_constants import name_constants  # noqa: E402

LAYERS = 64
ATTN_LAYERS = [l for l in range(LAYERS) if l % 4 == 3]
GDN_LAYERS = [l for l in range(LAYERS) if l % 4 != 3]
KV_HEADS, HEAD_DIM, BLOCK = 4, 256, 64
KV_TOKEN_BYTES = KV_HEADS * 2 * HEAD_DIM * 2
KV_BLOCK_BYTES = BLOCK * KV_TOKEN_BYTES
GDN_DIM, GDN_WIDTH, QKVZ_WIDTH = 10240, 4, 16384
QKVZBA_WIDTH = QKVZ_WIDTH + 96
HIDDEN, MLP_WIDTH = 5120, 17408
GDN_LINE_BYTES = (GDN_WIDTH - 1) * GDN_DIM * 2 + 48 * 128 * 128 * 4
PAGE_BYTES = -(-GDN_LINE_BYTES // KV_BLOCK_BYTES) * KV_BLOCK_BYTES

OLD_KV_PAGE_BYTES = len(ATTN_LAYERS) * KV_BLOCK_BYTES
OLD_GDN_LINE_BYTES = 3211264
MAX_SEQS = 128
CONV_SHA256 = "98b86305a3d49a0b930c6bbbed3f981417b6a7eda584e8645ee4cd51136c9f59"


def layer_of(call):
    return int(call["label"].split(".")[0][1:])


def host_states():
    kv = {"dtype": "bf16", "shape": [0, KV_HEADS, BLOCK, 2 * HEAD_DIM],
          "strides": [KV_BLOCK_BYTES // 2, 2 * HEAD_DIM, KV_HEADS * 2 * HEAD_DIM, 1]}
    gdn = {"dtype": "u8", "shape": [0, GDN_LINE_BYTES], "strides": [PAGE_BYTES, 1]}
    return ({f"kv.l{l}": {"host": kv} for l in ATTN_LAYERS} |
            {f"gdn.l{l}": {"host": gdn} for l in GDN_LAYERS})


def rebind(call):
    """A call's `kv` / `gdn` args onto its layer's host state."""
    def arg(a):
        if a.get("state") == "kv":
            k, off = divmod(a.get("offset", 0), KV_BLOCK_BYTES)
            assert ATTN_LAYERS[k] == layer_of(call), call["label"]
            return {"state": f"kv.l{layer_of(call)}", "offset": off} if off else {"state": f"kv.l{layer_of(call)}"}
        if a.get("state") == "gdn":
            return {**a, "state": f"gdn.l{layer_of(call)}"}
        if a.get("buf") == "gdn.line_index":
            assert a.get("offset", 0) == GDN_LAYERS.index(layer_of(call)) * MAX_SEQS * 4, call["label"]
        return a
    return {**call, "args": [arg(a) for a in call["args"]]}


def restride(op):
    """One op's baked page strides, kern's pool to vLLM's pages."""
    swap = {OLD_KV_PAGE_BYTES: KV_BLOCK_BYTES, OLD_KV_PAGE_BYTES // 2: KV_BLOCK_BYTES // 2,
            OLD_GDN_LINE_BYTES: PAGE_BYTES}

    def field(f):
        if "tensormap" in f and len(f["tensormap"]["strides"]) == 3:
            t = f["tensormap"]
            return {**f, "tensormap": {**t, "strides": [swap.get(s, s) for s in t["strides"]]}}
        return f

    def arg(a):
        if "i64" in a:
            return {"i64": swap.get(a["i64"], a["i64"])}
        if "pack" in a:
            return {"pack": {**a["pack"], "fields": [field(f) for f in a["pack"]["fields"]]}}
        return a

    return {**op, "impl": {**op["impl"], "launches": [
        {**l, "args": [arg(a) for a in l["args"]]} if "args" in l else l for l in op["impl"]["launches"]]}}


def conv_fwd():
    common = ["in buffer<i32>", "in buffer<u8>", "in buffer<i32>"]
    return dict(
        params=["in buffer<bf16>", "in buffer<bf16>", "inout state", *common, "out buffer<bf16>", "i32"],
        impl=dict(launches=[
            dict(module="gdn_conv_fwd", entry="kern_gdn_conv_fwd_bf16",
                 params=["in buffer<bf16>", "in buffer<bf16>", "in state", *common, "out buffer<bf16>",
                         "i32", "i32", "i32", "i32", "i64"],
                 block=[256, 1, 1], grid=[GDN_DIM // 256, "tokens", 1],
                 args=[{"param": i} for i in range(8)] +
                      [{"i32": GDN_DIM}, {"i32": QKVZ_WIDTH}, {"i32": GDN_DIM}, {"i64": PAGE_BYTES}]),
            dict(module="gdn_conv_fwd", entry="kern_gdn_conv_fwd_state_bf16",
                 params=["in buffer<bf16>", "inout state", *common, "i32", "i32", "i32", "i64"],
                 block=[256, 1, 1], grid=[GDN_DIM // 256, "seqs", 1],
                 args=[{"param": i} for i in (0, 2, 3, 4, 5, 7)] +
                      [{"i32": GDN_DIM}, {"i32": QKVZ_WIDTH}, {"i64": PAGE_BYTES}]),
        ]))


def conv_call(call):
    """The captured conv's args without its Triton tiling tables, plus the cu_seqlens length."""
    return {**call, "args": [a for i, a in enumerate(call["args"]) if i not in (6, 7)] +
            [{"expr": {"add": ["seqs", 1]}}]}


def wide(call):
    """A gemm16 call (at most 16 rows) as the cuBLASLt gemm it is bit-identical
    to, so a decode step takes every sequence vLLM batches."""
    a, rows = call["args"], {"var": "tokens"}
    gemm = lambda label, x, w, y, n, k: {"label": label, "op": "gemm", "args": [x, w, y, rows, {"i32": n}, {"i32": k}]}
    layer = call["label"].rsplit(".", 1)[0]
    match call["op"]:
        case "gemm16_in_proj":
            return [gemm(f"{layer}.in_proj_qkvzba", a[0], {"buf": in_proj(layer_of(call))}, {"buf": "qkvzba"},
                         QKVZBA_WIDTH, HIDDEN)]
        case "gemm16_gate_up_silu":
            return [gemm(f"{layer}.gate_up", a[0], a[1], {"buf": "gate_up"}, 2 * MLP_WIDTH, HIDDEN),
                    {"label": f"{layer}.silu_mul", "op": "silu_mul", "args": [a[2], {"buf": "gate_up"}]}]
        case "gemm16_qkv":
            return [gemm(call["label"], a[0], a[1], a[2], 14336, HIDDEN)]
        case "gemm16_o" | "gemm16_out":
            return [gemm(call["label"], a[0], a[1], a[2], HIDDEN, 6144)]
        case _:
            return [call]


def in_proj(layer):
    return f"model.layers.{layer}.linear_attn.in_proj_qkvzba.weight"


def fused_in_proj(m):
    """One weight per GDN layer for qkvz and ba, the four checkpoint tensors
    stacked. Decode reads it in one gemm into `qkvzba` rows (`wide`), and
    the conv and step kernels take that row stride, `ba` at its column
    16384; a small separate ba gemm is a launch and a tail for 1 MB. Prefill
    keeps its two gemms, over the two row ranges of the same weight."""
    qkvz = lambda l: f"model.layers.{l}.linear_attn.in_proj_qkvz.weight"
    ba = lambda l: f"model.layers.{l}.linear_attn.in_proj_ba.weight"
    moved = {qkvz(l): {"buf": in_proj(l)} for l in GDN_LAYERS} | \
            {ba(l): {"buf": in_proj(l), "offset": QKVZ_WIDTH * HIDDEN * 2} for l in GDN_LAYERS}
    arg = lambda x: moved.get(x.get("buf"), x)
    step = {"gdn_conv": {0: {"buf": "qkvzba"}},
            "gdn_step": {0: {"buf": "qkvzba"}, 1: {"buf": "qkvzba", "offset": QKVZ_WIDTH * 2}}}
    call = lambda c: {**c, "args": [step.get(c["op"], {}).get(i, arg(x)) for i, x in enumerate(c["args"])]}
    strides = {"gdn_conv": {5: QKVZ_WIDTH}, "gdn_step": {10: QKVZ_WIDTH, 11: 96}}

    def restrided(name, op):
        at = strides.get(name)
        if not at:
            return op
        (launch,) = op["impl"]["launches"]
        assert all(launch["args"][i] == {"i32": old} for i, old in at.items()), f"{name}: row strides moved"
        args = [{"i32": QKVZBA_WIDTH} if i in at else x for i, x in enumerate(launch["args"])]
        return {**op, "impl": {**op["impl"], "launches": [{**launch, "args": args}]}}

    b = m["buffers"]
    return {**m,
            "ops": {k: restrided(k, v) for k, v in m["ops"].items()},
            "programs": {k: {**p, "calls": [call(c) for c in p["calls"]]} for k, p in m["programs"].items()},
            "buffers": b | {"qkvzba": {"dtype": "bf16", "shape": ["tokens", QKVZBA_WIDTH], "kind": "workspace"}} | {
                in_proj(l): {**b[qkvz(l)], "shape": [QKVZBA_WIDTH, HIDDEN], "bind": b[qkvz(l)]["bind"] + b[ba(l)]["bind"]}
                for l in GDN_LAYERS}}


def headless(calls):
    """The step up to the final norm, which writes `hidden`; vLLM samples."""
    end = next(i for i, c in enumerate(calls) if c["label"].endswith(".final_norm"))
    final = calls[end]
    return calls[:end] + [{**final, "args": [{"buf": "hidden"}, *final["args"][1:]]}]


def hosted(m):
    m = {**m, "model": f"{m['model']}-vllm", "states": host_states()}
    m["ops"] = {k: restride(v) for k, v in m["ops"].items()} | {"conv_fwd": conv_fwd()}
    m["modules"] = {**m["modules"], "gdn_conv_fwd": {
        "source": "gdn_conv_fwd.cubin", "sha256": CONV_SHA256}}

    def step(p):
        calls = [conv_call(c) if c["op"] == "conv_fwd" else c for c in map(rebind, headless(p["calls"]))]
        return {**p, "calls": calls}

    m["programs"] = {
        "load": m["programs"]["load"],
        "prefill": step(m["programs"]["prefill"]),
        "decode_batch": {**step(m["programs"]["decode_batch"]), "calls": [
            c for call in step(m["programs"]["decode_batch"])["calls"] for c in wide(call)]},
        "head": {"calls": [{"label": "lm_head", "op": "gemm", "args": [
            {"buf": "head_in"}, {"buf": "lm_head.weight"}, {"buf": "logits"},
            {"var": "seqs"}, {"i32": 248320}, {"i32": 5120}]}]},
    }
    b = m["buffers"]
    kv, gdn = f"kv.l{ATTN_LAYERS[0]}", f"gdn.l{GDN_LAYERS[0]}"
    ids = {"slot_mapping": {"index_into": kv},
           "block_table": {"index_into": kv, "stride": BLOCK},
           "gdn.line_index": {"index_into": gdn, "stride": PAGE_BYTES}}
    indexed = {k: {**b[k], "domain": d} for k, d in ids.items()}
    m["buffers"] = {k: v for k, v in b.items() if k != "next_token"} | indexed | {
        "hidden": {"dtype": "bf16", "shape": ["tokens", 5120], "kind": "output"},
        "head_in": {"dtype": "bf16", "shape": ["seqs", 5120], "kind": "input"},
        "logits": {**b["logits"], "kind": "output"},
    }
    return m


def sampled(m, base):
    """Hands back the base's head after the final norm, reading `hidden`, so
    `kern test` gates the hosted programs against the base manifest."""
    def tail(p):
        calls = base["programs"][p]["calls"]
        end = next(i for i, c in enumerate(calls) if c["label"].endswith(".final_norm"))
        return [{**c, "args": [{"buf": "hidden"} if a == {"buf": "x"} else a for a in c["args"]]}
                for c in calls[end + 1:]]
    programs = {p: {**m["programs"][p], "calls": m["programs"][p]["calls"] + tail(p)}
                for p in ("prefill", "decode_batch")}
    return {**m, "programs": {**m["programs"], **programs},
            "buffers": m["buffers"] | {k: base["buffers"][k] for k in ("next_token", "final_x")}}


def prune(m):
    """Drop what no program refers to any more: ops, modules, buffers, and
    the `load` calls filling only those buffers."""
    def bufs(calls):
        return {a["buf"] for c in calls for a in c["args"] if "buf" in a}
    steps = [c for n, p in m["programs"].items() if n != "load" for c in p["calls"]]
    load = [c for c in m["programs"]["load"]["calls"] if bufs([c]) & bufs(steps)]
    programs = {**m["programs"], "load": {**m["programs"]["load"],
                "calls": [{**c, "label": f"load.{c['op']}.{i}"} for i, c in enumerate(load)]}}
    calls = [c for p in programs.values() for c in p["calls"]]
    ops = {k: v for k, v in m["ops"].items() if k in {c["op"] for c in calls}}
    modules = {l["module"] for o in ops.values() for l in o["impl"]["launches"] if "module" in l}
    return {**m, "programs": programs, "ops": ops,
            "buffers": {k: v for k, v in m["buffers"].items() if k in bufs(calls)},
            "modules": {k: v for k, v in m["modules"].items() if k in modules}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=pathlib.Path, default=pathlib.Path("examples/qwen3.8-27b.json"))
    p.add_argument("--output", type=pathlib.Path, required=True)
    p.add_argument("--sampled", action="store_true", help="keep the base's head and sampling, for `kern test`")
    args = p.parse_args()
    base = resolve_constants(json.loads(args.input.read_text()))
    m = fused_in_proj(hosted(base))
    m = prune(sampled(m, base) if args.sampled else m)
    args.output.write_text(json.dumps(name_constants(normalize(m)), indent=1) + "\n")


if __name__ == "__main__":
    main()
