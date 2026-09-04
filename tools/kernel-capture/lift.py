#!/usr/bin/env python3
"""Lift a captured kernel launch into a manifest launch skeleton.

Input is a kernel-capture `launches.jsonl` (see capture.c); output is the JSON
a generator under tools/ needs to declare the same launch as a manifest op:
`entry`, `block`/`grid`/`shared_mem`, a `params` type list and an `args` list
with every TMA descriptor spelled out as a `tensormap` (dtype, dims, strides,
box, swizzle, L2 promotion) and every struct parameter as a `pack`.

What the machine can decide, it decides: parameter widths come from
`cuFuncGetParamInfo`, descriptors from the intercepted `cuTensorMapEncodeTiled`
calls, the live bytes of a cute TiledCopy struct from its demangled type (one
`int` per `ScaledBasis<int, k>` in the AuxTmaParams tuple, packed after the
128-byte descriptor; everything else in the struct is padding the host never
initialised). What it cannot decide is left as a placeholder for the author:
each device pointer becomes `"@<letter>+<offset>"` naming an allocation the
launch touched, and the `allocations` table lists those allocations by size,
so the author maps letters to interface params, states or scratch. A 4-byte
scalar is emitted as `i32` with its `f32` reading alongside; pick one.

    lift.py <launches.jsonl> --symbol <substring> [--index N] [--all]

`--symbol` selects launches whose mangled name contains the substring,
`--index` picks the N-th of them (default 0); `--all` prints every distinct
(grid, params) variant of that kernel instead, which is how one learns which
fields move with the shape. Needs `cu++filt` (any CUDA toolkit) on PATH or in
$CUDA_HOME/bin for the TiledCopy decode; without it struct tails are emitted
as raw hex for the author to read.
"""
import argparse
import json
import os
import re
import shutil
import string
import struct
import subprocess
import sys

TMAP_BYTES = 128


def demangle(symbol):
    tool = shutil.which("cu++filt") or os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "bin", "cu++filt")
    if not os.path.exists(tool):
        return None
    out = subprocess.run([tool, symbol], capture_output=True, text=True).stdout.strip()
    return out if out and out != symbol else None


def split_template_args(text):
    """Top-level comma split of the text between a template's angle brackets."""
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        if ch in "<(":
            depth += 1
        elif ch in ">)":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    parts.append(text[start:].strip())
    return parts


def balanced_span(text, start, open_ch, close_ch):
    """End index (exclusive) of the bracketed span opening at `start`."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == open_ch:
            depth += 1
        elif text[i] == close_ch:
            depth -= 1
            if depth == 0:
                return i
    return None


def kernel_param_types(demangled):
    """The kernel's parameter types in order, from `void f<A, B, ...>(T1, T2, x, ...)`.

    cu++filt names a parameter whose type is the k-th template argument `Tk`
    instead of spelling it out; those are substituted back so a TiledCopy
    parameter can be decoded from its full type.
    """
    close = demangled.rfind(")")
    if close < 0:
        return None
    # Template arguments carry casts like `(int)16`, so the parameter list's
    # opening paren is the one that balances the final close, not the last `(`.
    depth, open_paren = 0, None
    for i in range(close, -1, -1):
        if demangled[i] == ")":
            depth += 1
        elif demangled[i] == "(":
            depth -= 1
            if depth == 0:
                open_paren = i
                break
    if open_paren is None:
        return None
    params = split_template_args(demangled[open_paren + 1 : close])
    lt = demangled.find("<")
    template_args = []
    if 0 <= lt < open_paren:
        gt = balanced_span(demangled, lt, "<", ">")
        if gt is not None:
            template_args = split_template_args(demangled[lt + 1 : gt])

    def resolve(t):
        m = re.fullmatch(r"T(\d+)", t)
        return template_args[int(m.group(1)) - 1] if m and int(m.group(1)) <= len(template_args) else t

    return [resolve(t) for t in params]


def tiled_copy_dynamic_ints(param_type):
    """Number of runtime `int`s a cute TiledCopy parameter carries after its descriptor."""
    if "cute::TiledCopy<" not in param_type:
        return None
    aux = param_type.find("cute::AuxTmaParams<")
    if aux < 0:
        return 0
    tuple_start = param_type.find("cute::tuple<", aux)
    depth, i = 0, tuple_start
    while i < len(param_type):
        if param_type[i] == "<":
            depth += 1
        elif param_type[i] == ">":
            depth -= 1
            if depth == 0:
                break
        i += 1
    return param_type[tuple_start:i].count("ScaledBasis<int")


def tensormap_arg(t, alloc_of):
    return {
        "param": alloc_of(t["pointer"]["range_start"], int(t["address"], 16) - int(t["pointer"]["range_start"], 16))
        if "pointer" in t
        else t["address"],
        "dtype": t["dtype"],
        "dims": t["dims"],
        "strides": t["strides"],
        "box": t["box"],
        "swizzle": t["swizzle"],
        "l2_promotion": t["l2_promotion"],
        **({"oob_nan": True} if t["oob_fill"] else {}),
        **({"elem_strides": t["elem_strides"]} if any(s != 1 for s in t["elem_strides"]) else {}),
    }


def scalar_arg(data):
    raw = bytes.fromhex(data)
    if len(raw) == 4:
        (i,) = struct.unpack("<i", raw)
        (f,) = struct.unpack("<f", raw)
        return {"i32": i, "f32?": f}
    if len(raw) == 8:
        (i,) = struct.unpack("<q", raw)
        return {"i64": i}
    return {"bytes": data}


def lift(rec, names=None):
    """The manifest skeleton of one launch. `names` maps an allocation's device
    address (as the launcher printed it) to the buffer's name; an allocation
    without one gets a letter, and those the author has to map by hand."""
    letters = iter(string.ascii_uppercase)
    allocs = {}

    def alloc_of(range_start, offset):
        if range_start not in allocs:
            allocs[range_start] = (names or {}).get(range_start) or next(letters)
        return f"@{allocs[range_start]}+{offset:#x}"

    demangled = demangle(rec["symbol"])
    types = kernel_param_types(demangled) if demangled else None
    params, args, notes = [], [], []
    for i, p in enumerate(rec["params"]):
        size, maps = p["size"], p.get("tensormaps", [])
        ptype = types[i] if types and i < len(types) else None
        if size == TMAP_BYTES and maps and maps[0]["at"] == 0:
            params.append("tensormap")
            args.append({"tensormap": tensormap_arg(maps[0], alloc_of)})
        elif maps:
            fields = [{"at": m["at"], "tensormap": tensormap_arg(m, alloc_of)} for m in maps]
            dyn = tiled_copy_dynamic_ints(ptype) if ptype else None
            raw = bytes.fromhex(p["data"])
            if dyn is not None and len(maps) == 1:
                for k in range(dyn):
                    at = TMAP_BYTES + 4 * k
                    fields.append({"at": at, "i32": struct.unpack_from("<i", raw, at)[0]})
            else:
                tail = raw[maps[-1]["at"] + TMAP_BYTES :]
                if any(tail):
                    notes.append(f"param {i}: bytes after the descriptor are not decoded ({tail.hex()}); check the kernel's struct")
            params.append(f"bytes<{size}>")
            args.append({"pack": {"size": size, "fields": fields}})
        elif size == 8 and "pointer" in p:
            ptr = p["pointer"]
            value = int.from_bytes(bytes.fromhex(p["data"]), "little")
            params.append("buffer")
            args.append({"param": alloc_of(ptr["range_start"], value - int(ptr["range_start"], 16))})
        elif size in (4, 8):
            params.append("i32" if size == 4 else "i64")
            args.append(scalar_arg(p["data"]))
        else:
            params.append(f"bytes<{size}>")
            args.append({"pack": {"size": size, "fields": []}, "raw": p["data"]})
    if types:
        for i, t in enumerate(types):
            if i < len(params):
                notes.append(f"param {i}: {t[:80]}{'…' if len(t) > 80 else ''}")
    allocations = {}
    for rec_p in rec["params"]:
        for m in rec_p.get("tensormaps", []):
            if "pointer" in m:
                allocations.setdefault(m["pointer"]["range_start"], m["pointer"]["range_size"])
        if "pointer" in rec_p:
            allocations.setdefault(rec_p["pointer"]["range_start"], rec_p["pointer"]["range_size"])
    return {
        "entry": rec["symbol"],
        "block": rec["block"],
        "grid": rec["grid"],
        "shared_mem": rec["dynamic_shared_mem_bytes"],
        "attributes": rec["attributes"],
        "params": params,
        "args": args,
        "allocations": {allocs[k]: {"range_start": k, "size": v} for k, v in allocations.items() if k in allocs},
        "notes": notes,
    }


def variant_key(rec):
    """Shape identity of a launch: geometry plus every descriptor's dims/box, addresses ignored."""
    shape_of = lambda m: (m["dtype"], m["dims"], m["strides"], m["box"], m["swizzle"])
    return json.dumps(
        [rec["grid"], rec["block"], [(p["size"], [shape_of(m) for m in p.get("tensormaps", [])]) for p in rec["params"]]],
        sort_keys=True,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("launches")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--names", help="file of `<name> <0xaddress>` lines the launcher printed, one per allocation")
    a = ap.parse_args()
    names = {}
    for line in open(a.names) if a.names else []:
        parts = line.split()
        if len(parts) == 2 and parts[1].startswith("0x"):
            names[f"{int(parts[1], 16):#x}"] = parts[0]
    recs = [json.loads(l) for l in open(a.launches)]
    hits = [r for r in recs if a.symbol in r["symbol"] and isinstance(r["params"], list)]
    if not hits:
        sys.exit(f"no launch of a kernel containing {a.symbol!r} with a known parameter layout")
    if a.all:
        seen = {}
        for r in hits:
            seen.setdefault(variant_key(r), r)
        out = [lift(r, names) for r in seen.values()]
    else:
        out = lift(hits[a.index], names)
    json.dump(out, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
