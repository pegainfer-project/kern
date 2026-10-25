"""The Qwen3-4B manifest with its KV in SGLang's layout, for a kern prefill
whose pages an SGLang decode instance reads (`kern_sglang.prefill`).

SGLang's `MHATokenToKVPool` keeps a layer's keys and values as two arrays,
`[tokens, 8 heads, 128]` bf16 each, and its P/D transfer moves a page as one
run of `page * 2048` bytes per array. `examples/qwen3-4b.json` keeps a layer
as one pooled state `kv.<L>`, `[page][token][head][K 128 | V 128]`. Here each
layer becomes two pooled states, `k.l<L>` and `v.l<L>`, `[page][token][head]
[128]`, named by the prefill adapter's contract: a page of either is the bytes
SGLang moves for that page and layer, so a decode instance with the same page
size takes them as they are.

Nothing else changes. The cache write (`reshape_and_cache_kernel_flash`) and
both attention kernels take their cache strides as arguments; in the new
layout each of them (page, token, head, in elements) is half the base's, and
the V operand is its own state instead of the K state 256 bytes on.

    python tools/qwen3_sglang_kv.py [--input examples/qwen3-4b.json] --output qwen3-4b-sglang-kv.json
"""
import argparse
import json
import pathlib
import re

LAYERS = 36
TOKEN_BYTES = 8 * 128 * 2
# (page, token, head) strides in elements, base layout then this one.
BASE = (32768, 2048, 256)
STRIDES = {
    # entry: argument positions of the page, token and head strides, per cache operand
    "reshape_and_cache_kernel_flash": [(9, 13, 10)],
    "kernel_unified_attention": [(18, 19, 20), (21, 22, 23)],
}


def restride(op):
    """An op's cache strides halved, each checked against the base's."""
    launches = []
    for launch in op["impl"]["launches"]:
        args = [dict(a) for a in launch["args"]]
        for at in STRIDES.get(launch["entry"], []):
            for i, base in zip(at, BASE):
                assert args[i] == {"i64": base}, (launch["entry"], i, args[i])
                args[i] = {"i64": base // 2}
        launches.append({**launch, "args": args})
    return {**op, "impl": {**op["impl"], "launches": launches}}


def split(arg):
    """A `kv.<L>` state argument onto the layer's K or V state."""
    m = re.fullmatch(r"kv\.(\d+)", arg.get("state", ""))
    if not m:
        return arg
    kind = {0: "k", 256: "v"}[arg.get("offset", 0)]
    return {"state": f"{kind}.l{m[1]}"}


def sglang_kv(m):
    assert set(m["states"]) == {f"kv.{l}" for l in range(LAYERS)}, sorted(m["states"])
    constants = {k: v for k, v in m["constants"].items() if not re.fullmatch(r"KV\.\d+_BYTES_PER_TOKEN", k)}
    states = {f"{kind}.l{l}": {"bytes_per_token": "KV_TOKEN_BYTES"} for l in range(LAYERS) for kind in ("k", "v")}
    touched = {n for n, op in m["ops"].items()
               if any(l["entry"] in STRIDES for l in op["impl"]["launches"])}
    assert touched == {"reshape_and_cache", "attn_prefill", "attn", "attn_batch"}, touched
    b = m["buffers"]
    return {
        **m,
        "model": f"{m['model']}-sglang-kv",
        "constants": constants | {"KV_TOKEN_BYTES": TOKEN_BYTES},
        "states": states,
        "buffers": b | {
            "slot_mapping": {**b["slot_mapping"], "domain": {"index_into": "k.l0"}},
            "block_table": {**b["block_table"], "domain": {**b["block_table"]["domain"], "index_into": "k.l0"}},
        },
        "ops": {n: restride(op) if n in touched else op for n, op in m["ops"].items()},
        "programs": {n: {**p, "calls": [{**c, "args": [split(a) for a in c["args"]]} for c in p["calls"]]}
                     for n, p in m["programs"].items()},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=pathlib.Path, default=pathlib.Path("examples/qwen3-4b.json"))
    p.add_argument("--output", type=pathlib.Path, required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps(sglang_kv(json.loads(a.input.read_text())), indent=1) + "\n")


if __name__ == "__main__":
    main()
