#!/usr/bin/env python3
"""Replay tools/kernels/abi/trtllm_bmm.py against a capture of FlashInfer's
own launches (tools/kernel-capture over probe.py): for every `bmm_*` launch
in the capture, build the manifest op for the variant it names on the
probe's shape and compare field for field: grid, block, shared memory,
every descriptor (dtype, dims, strides, box, swizzle, L2, base address),
every pointer and scalar, and that no byte outside the fields is set
(the routing arrays inside `KernelParams` are uninitialised host memory in
the launcher and are skipped). The base addresses the probe printed name
the weights and activations; the workspace buffers (FC1's output and
scales, the routing tables) are checked for consistency across the two
launches instead.

    python3 check_abi.py <launches.jsonl> <probe log> [--tokens 300] [--local 56]
"""
import argparse
import json
import pathlib
import re
import struct
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from kernels import index  # noqa: E402
from kernels.abi import trtllm_bmm  # noqa: E402

H, I, TOPK = 3584, 3072, 16
TMA_DTYPE = {"u8": 0, "u16": 1, "u32": 2, "i32": 3, "u64": 4, "i64": 5, "f16": 6, "f32": 7, "bf16": 9, "tf32": 11,
             "u4packed": 13, "u4": 14}
FAMILIES = ["trtllm_bmm_mxe4m3_mxe2m1_mxe4m3", "trtllm_bmm_bf16_mxe2m1_mxe4m3"]
# probe pointer name -> interface param, per GEMM
NAMED = {"fc1": {"a": "w13s", "sf_a": "w13_sfs", "b": "x_fp8", "sf_b": "x_sf", "alpha": "alpha", "beta": "beta"},
         "fc2": {"a": "w2s", "sf_a": "w2_sfs"}}
# uninitialised host memory: descriptor slots the launcher did not build, the static routing arrays, the tail padding
UNINITIALISED = [(0, 768), (1008, 17392), (17448, 17472)]


def variant_of(entry):
    for fam in FAMILIES:
        for v in index.load(fam).get("variant", []):
            if v["name"] == entry:
                return index.variant_by_name(fam, entry)
    raise KeyError(entry)


def check(rec, tokens, local, ptrs, seen):
    v = variant_of(rec["symbol"])
    gated = bool(v.tags["fused_act"])
    which = "fc1" if gated else "fc2"
    m, k = (2 * I, H) if gated else (H, I)
    ctas = trtllm_bmm.max_ctas(tokens, TOPK, local, v.tags["tile"][1])
    o = trtllm_bmm.op(v, m, k, local, tokens, tokens, ctas, ctas)
    launch = o["impl"]["launches"][0]
    names = trtllm_bmm.params(v)
    errs = []
    fact = lambda what, got, want: errs.append(f"{what}: got {got}, want {want}") if got != want else None
    fact("grid", rec["grid"], launch["grid"])
    fact("block", rec["block"], launch["block"])
    fact("shared_mem", rec["dynamic_shared_mem_bytes"], launch["shared_mem"])
    p = rec["params"][0]
    fact("param size", p["size"], trtllm_bmm.SIZE)
    raw = bytes.fromhex(p["data"])
    maps = {t["at"]: t for t in p.get("tensormaps", [])}
    covered = bytearray(len(raw))

    def address(param, addr, what):
        """A pointer field's value: the probe named it, or it is a workspace buffer every launch must agree on."""
        name = names[param]
        key = NAMED[which].get(name)
        if key:
            want = ptrs.get(key)
        else:
            shared = (("fc1", {"b": "c", "sf_b": "sf_c"}[name]) if which == "fc2" and name in ("b", "sf_b")
                      else ("tables", name) if name.startswith(("num_", "total_", "cta_")) else (which, name))
            want = seen.setdefault(shared, addr)
        fact(f"{what} ({name})", hex(addr), hex(want) if want is not None else "?")

    for f in launch["args"][0]["pack"]["fields"]:
        at = f["at"]
        if "tensormap" in f:
            t, got = f["tensormap"], maps.get(at)
            covered[at:at + 128] = b"\1" * 128
            if got is None:
                errs.append(f"tensormap at {at}: not in the capture")
                continue
            for key, want in (("dtype_enum", TMA_DTYPE[t["dtype"]]), ("dims", t["dims"]), ("strides", t["strides"]),
                              ("box", t["box"]), ("swizzle", t["swizzle"]), ("l2_promotion", t["l2_promotion"])):
                fact(f"tensormap at {at} {key}", got[key], want)
            address(t["param"], int(got["address"], 16), f"tensormap at {at} address")
        elif "param" in f:
            covered[at:at + 8] = b"\1" * 8
            address(f["param"], struct.unpack_from("<Q", raw, at)[0], f"pointer at {at}")
        elif "i64" in f:
            covered[at:at + 8] = b"\1" * 8
            fact(f"i64 at {at}", struct.unpack_from("<q", raw, at)[0], f["i64"])
        elif "i32" in f:
            covered[at:at + 4] = b"\1" * 4
            fact(f"i32 at {at}", struct.unpack_from("<i", raw, at)[0], f["i32"])
        elif "var" in f:
            covered[at:at + 4] = b"\1" * 4
            fact(f"var at {at}", struct.unpack_from("<i", raw, at)[0], tokens)
        else:
            errs.append(f"field at {at}: unexpected source {f}")
    for at in maps:
        if not covered[at]:
            errs.append(f"tensormap at {at}: the launcher built one, the port does not")
    stray = [i for i, (b, c) in enumerate(zip(raw, covered)) if b and not c and not any(lo <= i < hi for lo, hi in UNINITIALISED)]
    if stray:
        errs.append(f"{len(stray)} set bytes outside the fields, first at {stray[0]} ({raw[stray[0]:stray[0] + 8].hex()})")
    return v.name, errs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("launches", type=pathlib.Path)
    ap.add_argument("log", type=pathlib.Path)
    ap.add_argument("--tokens", type=int, default=300)
    ap.add_argument("--local", type=int, default=56)
    a = ap.parse_args()
    ptrs = {n: int(h, 16) for n, h in re.findall(r"(\w+)=(0x[0-9a-f]+)", a.log.read_text())}
    recs = [json.loads(l) for l in a.launches.read_text().splitlines() if '"symbol":"bmm_' in l]
    seen, bad = {}, 0
    for rec in recs:
        name, errs = check(rec, a.tokens, a.local, ptrs, seen)
        print(f"{'FAIL' if errs else 'ok  '} {name}")
        for e in errs:
            print(f"     {e}")
        bad += bool(errs)
    print(f"{len(recs)} launches, {bad} failing")
    sys.exit(1 if bad or not recs else 0)


if __name__ == "__main__":
    main()
