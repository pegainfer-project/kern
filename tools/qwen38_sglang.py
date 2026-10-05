"""The Qwen3.8-27B manifest on SGLang's memory.

Rewrites `examples/qwen3.8-27b.json`, or a manifest derived from it (the
vLLM optimization loop's), into a manifest an SGLang model runner drives (`python/kern_sglang`), as `qwen38_vllm.py` does for vLLM:
the same programs (prefill, decode_batch, head, stopping at the hidden
states), over SGLang's pools instead of vLLM's pages.

SGLang allocates every tensor of a layer on its own (`MHATokenToKVPool`,
`MambaPool`), so each layer has four kinds of host state, one per tensor:

- `k.l<L>`, `v.l<L>`: a full-attention layer's keys and values, separate
  `[tokens, 4 heads, 256]` bf16 arrays read as 64-token pages
  (`[page][token][head][dim]`). Page ids are token slot / 64.
- `conv.l<L>`: a GDN layer's conv state, `[slot][10240][3]` bf16, the
  transpose of vLLM's `[3][10240]`. The kernels that touch it are built
  with `-DCONV_DIM_MAJOR` (the `*_dm` modules, pinned by sha; sources in
  the kern-qwen38-sm103 artifact repo).
- `ssm.l<L>`: a GDN layer's recurrent state, `[slot][48][128][128]` f32,
  the layout kern's kernels already use.

Every other kernel takes these strides as arguments or tensormap fields.
Conv and ssm differ in layout, so each has its own line table
(`gdn.conv_index`, `gdn.ssm_index`) over the same slot ids; a tool that is
its own host (`kern bench`) provisions both from them.

    python tools/qwen38_sglang.py [--input examples/qwen3.8-27b.json] --output qwen3.8-27b-sglang.json [--sampled]
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from kern_manifest import normalize, resolve_constants  # noqa: E402
from qwen38_vllm import (  # noqa: E402
    ATTN_LAYERS, BLOCK, GDN_DIM, GDN_LAYERS, GDN_WIDTH, HEAD_DIM, KV_HEADS, MAX_SEQS, QKVZ_WIDTH,
    conv_call, fused_in_proj, headless, layer_of, prune, sampled, wide)
from qwen_constants import name_constants  # noqa: E402

KV_TOKEN_BYTES = KV_HEADS * HEAD_DIM * 2
KV_PAGE_BYTES = BLOCK * KV_TOKEN_BYTES
CONV_LINE_BYTES = GDN_DIM * (GDN_WIDTH - 1) * 2
SSM_LINE_BYTES = 48 * 128 * 128 * 4

BASE_KV_LAYER_BYTES = 2 * KV_PAGE_BYTES
BASE_KV_PAGE_BYTES = len(ATTN_LAYERS) * BASE_KV_LAYER_BYTES
BASE_GDN_LINE_BYTES = 3211264
BASE_SSM_OFFSET = CONV_LINE_BYTES
DIM_MAJOR = {
    "gdn_conv_fwd": "07c6ca52a71e870c65a6a86ab054f7eab967ffce8ca81308b0354db844854141",
    "gdn_decode": "9689d2ecc08727c66f577f4d6cab14fe8f0bf915c1853b52c1b50f35455d7ae5",
    "gdn_conv_rows": "9ed0d69e1b110603abe6daace4e63d13be5f7693c3790f6d3ead638a438120f1",
    "gdn_decode_fused": "c5af218f67ef2d64484e6a7e13973fa391dad0ed0ba42c3c4c25eb2a470407a0",
}
CONV_STATES = ("gdn_conv", "conv_fwd", "conv_fwd_rows")
SPLIT_STEP = "kern_gdn_conv_step_bf16"
TABLES = {"conv": "gdn.conv_index", "ssm": "gdn.ssm_index"}


def host_states():
    kv = {"dtype": "bf16", "shape": [0, BLOCK, KV_HEADS, HEAD_DIM],
          "strides": [KV_PAGE_BYTES // 2, KV_HEADS * HEAD_DIM, HEAD_DIM, 1]}
    conv = {"dtype": "bf16", "shape": [0, GDN_DIM, GDN_WIDTH - 1], "strides": [CONV_LINE_BYTES // 2, GDN_WIDTH - 1, 1]}
    ssm = {"dtype": "f32", "shape": [0, 48, 128, 128], "strides": [SSM_LINE_BYTES // 4, 128 * 128, 128, 1]}
    return ({f"{kind}.l{l}": {"host": kv} for l in ATTN_LAYERS for kind in ("k", "v")} |
            {f"conv.l{l}": {"host": conv} for l in GDN_LAYERS} |
            {f"ssm.l{l}": {"host": ssm} for l in GDN_LAYERS})


def rebind(call, split):
    """A call's `kv` / `gdn` args onto its layer's tensors, and its line
    table onto the one over the tensor it reads. A call of a `split` op
    (conv and step in one launch) also gets the conv state and its table."""
    part = "conv" if call["op"] in CONV_STATES else "ssm"

    def arg(a):
        l = layer_of(call) if "state" in a or a.get("buf") == "gdn.line_index" else None
        if a.get("state") == "kv":
            k, off = divmod(a.get("offset", 0), BASE_KV_LAYER_BYTES)
            assert ATTN_LAYERS[k] == l and off in (0, 2 * HEAD_DIM), call["label"]
            return {"state": f"{'v' if off else 'k'}.l{l}"}
        if a.get("state") == "gdn":
            assert a.get("offset", 0) in ((0,) if part == "conv" else (0, BASE_SSM_OFFSET)), call["label"]
            return {"state": f"{part}.l{l}"}
        if a.get("buf") == "gdn.line_index":
            assert a.get("offset", 0) == GDN_LAYERS.index(l) * MAX_SEQS * 4, call["label"]
            return {**a, "buf": TABLES[part]}
        return a
    args = [arg(a) for a in call["args"]]
    if call["op"] in split:
        (table,) = [a for a in args if a.get("buf") == TABLES["ssm"]]
        args += [{"state": f"conv.l{layer_of(call)}"}, {**table, "buf": TABLES["conv"]}]
    return {**call, "args": args}


def tail_args(launch, old, new):
    """`launch` with its trailing scalar args `old` replaced by `new`."""
    args = launch["args"]
    assert args[-len(old):] == old, f"{launch['entry']}: {args[-len(old):]} != {old}"
    return {**launch, "args": args[:-len(old)] + new}


def kv_tensormaps(launch):
    """K and V tensormaps (params 2 and 3) over separate `[page][token][head][dim]` arrays."""
    def field(f):
        t = f.get("tensormap")
        if not t or t["param"] not in (2, 3):
            return f
        assert t["strides"] == [4096, 1024, BASE_KV_PAGE_BYTES], t
        return {**f, "tensormap": {**t, "strides": [KV_TOKEN_BYTES, HEAD_DIM * 2, KV_PAGE_BYTES]}}
    (pack,) = launch["args"]
    return {**launch, "args": [{"pack": {**pack["pack"], "fields": [field(f) for f in pack["pack"]["fields"]]}}]}


def split_step(launch, n):
    """The fused conv + step over separate conv and ssm lines: the ssm line
    stride in place, the conv state, its table (op params `n`, `n + 1`) and
    its line stride appended."""
    args = launch["args"]
    i = args.index({"i64": BASE_GDN_LINE_BYTES})
    return {**launch, "module": "gdn_decode_fused_dm",
            "params": [*launch["params"], "inout state", "in buffer<i32>", "i64"],
            "args": [*args[:i], {"i64": SSM_LINE_BYTES}, *args[i + 1:],
                     {"param": n}, {"param": n + 1}, {"i64": CONV_LINE_BYTES}]}


def restride(op):
    """One op's baked strides, kern's pools to SGLang's tensors, launch by
    launch on the kernel it runs."""
    i64 = lambda *v: [{"i64": x} for x in v]
    line = i64(BASE_GDN_LINE_BYTES)
    kv = lambda l: tail_args(l, i64(BASE_KV_PAGE_BYTES // 2, 2048, 512),
                             i64(KV_PAGE_BYTES // 2, KV_HEADS * HEAD_DIM, HEAD_DIM))
    conv = lambda l: {**tail_args(l, line, i64(CONV_LINE_BYTES)), "module": f"{l['module']}_dm"}
    lines = lambda l: tail_args(l, i64(BASE_GDN_LINE_BYTES, BASE_SSM_OFFSET, SSM_LINE_BYTES),
                                i64(SSM_LINE_BYTES, 0, SSM_LINE_BYTES))
    per_entry = {
        "kern_attn_prep_bf16": kv,
        "kern_attn_prep_wide_bf16": kv,
        "kern_gdn_conv_bf16": conv,
        "kern_gdn_conv_rows_bf16": conv,
        "kern_gdn_conv_fwd_state_bf16": conv,
        "kern_gdn_step_bf16": lambda l: tail_args(l, line, i64(SSM_LINE_BYTES)),
        "kern_chunk_h": lambda l: tail_args(l, i64(BASE_GDN_LINE_BYTES, BASE_SSM_OFFSET), i64(SSM_LINE_BYTES, 0)),
        "kern_line_gather": lines,
        "kern_line_scatter": lines,
    }

    def launch(l):
        if l["entry"] == SPLIT_STEP:
            return split_step(l, len(op["params"]))
        if l["entry"].startswith("fmha"):
            return kv_tensormaps(l)
        return per_entry.get(l["entry"], lambda l: l)(l)
    if "launches" not in op["impl"]:
        return op
    split = any(l["entry"] == SPLIT_STEP for l in op["impl"]["launches"])
    params = [*op["params"], "inout state", "in buffer<i32>"] if split else op["params"]
    return {**op, "params": params, "impl": {**op["impl"], "launches": [launch(l) for l in op["impl"]["launches"]]}}


def conv_fwd():
    """The prefill conv over dim-major conv lines (vLLM's `conv_fwd`, other stride and module)."""
    common = ["in buffer<i32>", "in buffer<u8>", "in buffer<i32>"]
    return dict(
        params=["in buffer<bf16>", "in buffer<bf16>", "inout state", *common, "out buffer<bf16>", "i32"],
        impl=dict(launches=[
            dict(module="gdn_conv_fwd_dm", entry="kern_gdn_conv_fwd_bf16",
                 params=["in buffer<bf16>", "in buffer<bf16>", "in state", *common, "out buffer<bf16>",
                         "i32", "i32", "i32", "i32", "i64"],
                 block=[256, 1, 1], grid=[GDN_DIM // 256, "tokens", 1],
                 args=[{"param": i} for i in range(8)] +
                      [{"i32": GDN_DIM}, {"i32": QKVZ_WIDTH}, {"i32": GDN_DIM}, {"i64": CONV_LINE_BYTES}]),
            dict(module="gdn_conv_fwd_dm", entry="kern_gdn_conv_fwd_state_bf16",
                 params=["in buffer<bf16>", "inout state", *common, "i32", "i32", "i32", "i64"],
                 block=[256, 1, 1], grid=[GDN_DIM // 256, "seqs", 1],
                 args=[{"param": i} for i in (0, 2, 3, 4, 5, 7)] +
                      [{"i32": GDN_DIM}, {"i32": QKVZ_WIDTH}, {"i64": CONV_LINE_BYTES}]),
        ]))


def hosted(m):
    m = {**m, "model": f"{m['model']}-sglang", "states": host_states()}
    m["ops"] = {k: restride(v) for k, v in m["ops"].items()} | {"conv_fwd": conv_fwd()}
    m["modules"] = m["modules"] | {f"{k}_dm": {"source": f"{k}_dm.cubin", "sha256": v} for k, v in DIM_MAJOR.items()}
    split = {k for k, v in m["ops"].items() if any(l["entry"] == SPLIT_STEP for l in v["impl"].get("launches", []))}

    def step(p):
        return {**p, "calls": [rebind(conv_call(c) if c["op"] == "conv_fwd" else c, split) for c in headless(p["calls"])]}

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
    k, conv, ssm = f"k.l{ATTN_LAYERS[0]}", f"conv.l{GDN_LAYERS[0]}", f"ssm.l{GDN_LAYERS[0]}"
    lines = b["gdn.line_index"]
    m["buffers"] = {n: v for n, v in b.items() if n not in ("next_token", "gdn.line_index")} | {
        "slot_mapping": {**b["slot_mapping"], "domain": {"index_into": k}},
        "block_table": {**b["block_table"], "domain": {"index_into": k, "stride": BLOCK}},
        TABLES["conv"]: {**lines, "domain": {"index_into": conv, "stride": CONV_LINE_BYTES}},
        TABLES["ssm"]: {**lines, "domain": {"index_into": ssm, "stride": SSM_LINE_BYTES}},
        "hidden": {"dtype": "bf16", "shape": ["tokens", 5120], "kind": "output"},
        "head_in": {"dtype": "bf16", "shape": ["seqs", 5120], "kind": "input"},
        "logits": {**b["logits"], "kind": "output"},
    }
    return m


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
