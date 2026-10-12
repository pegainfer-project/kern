"""A few layers of a kern-k3-pp-prefill stage manifest, everything else dead.

    python3 trim.py <stage.json> <out.json> <label prefix>...

head.json and tail.json are, from kern-k3-pp-prefill 3baa6ef's manifests/:

    python3 trim.py k3-pruned-pp4-l0-23.json head.json embed fmha_lens l0. l1. l3. l22.hidden
    python3 trim.py k3-pruned-pp4-l69-93.json tail.json in. fmha_lens l69. l71. out.

a dense KDA layer, a KDA and an MLA layer over MoE, and the stage's own
calls at either end (the hidden state it hands on, the head).

Keeps the `prefill` calls whose label starts with one of the prefixes (a
layer is `l<i>.`, the stage's own calls `embed`, `in.`, `out.` ...), then
slices `load` and `derive` back to the calls the kept ones read from and
drops every buffer, op and module nothing kept names.

Two migrations bring the stage, made against kern b43bb5a, to schema 7:
its vars say which axis of a call they count, and `span_at` (the
row a span started at, always 0 in a chunk program) becomes a carry `load`
fills with 0, since the fill it named is gone.
"""
import json
import pathlib
import sys


AXES = {"tokens": "rows", "seqs": "groups", "rows": "ranks"}


def migrate(m):
    m["schema_version"] = 7
    for v, axis in AXES.items():
        m["vars"][v]["axis"] = axis
    m["buffers"]["span_at"] = {"dtype": "i32", "shape": [1], "kind": "carry"}
    m["ops"]["fill_i32"] = {
        "params": ["out buffer<i32>", "i32"],
        "impl": {"launches": [{
            "module": "weight_prep", "entry": "kern_fill_i32",
            "params": ["out buffer<i32>", "i32", "i32"], "block": [256, 1, 1], "grid": [1, 1, 1],
            "args": [{"param": 0}, {"i32": 1}, {"param": 1}],
        }]},
    }
    m["programs"]["load"]["calls"].append(
        {"label": "load.span_at", "op": "fill_i32", "args": [{"buf": "span_at"}, {"i32": 0}]})
    return m


def bufs(m, call, dirs):
    return {a["buf"] for p, a in zip(m["ops"][call["op"]]["params"], call["args"])
            if isinstance(a, dict) and "buf" in a and p.split()[0] in dirs}


def slice_back(m, calls, need):
    """The calls of a once program that write something `need` names, last
    to first, each adding what it reads."""
    kept = []
    for c in reversed(calls):
        if bufs(m, c, {"out", "inout"}) & need:
            kept.append(c)
            need |= bufs(m, c, {"in", "inout"})
    return kept[::-1], need


def trim(m, prefixes):
    p = m["programs"]["prefill"]
    p["calls"] = [c for c in p["calls"] if c["label"].startswith(tuple(prefixes))]
    need = set().union(*(bufs(m, c, {"in", "out", "inout"}) for c in p["calls"]))
    need |= {n for n, b in m["buffers"].items() if b["kind"] in ("input", "output", "inout")}
    for name in ("load", "derive"):
        m["programs"][name]["calls"], need = slice_back(m, m["programs"][name]["calls"], need)
    need |= {b["domain"]["index_into"] for n, b in m["buffers"].items()
             if n in need and "index_into" in b.get("domain", {})}
    m["buffers"] = {n: b for n, b in m["buffers"].items() if n in need}
    ops = {c["op"] for prog in m["programs"].values() for c in prog["calls"]}
    m["ops"] = {n: o for n, o in m["ops"].items() if n in ops}
    mods = {la["module"] for o in m["ops"].values() for la in o["impl"].get("launches", []) if "module" in la}
    m["modules"] = {n: x for n, x in m["modules"].items() if n in mods}
    return m


def main(stage, out, *prefixes):
    m = trim(migrate(json.loads(pathlib.Path(stage).read_text())), prefixes)
    pathlib.Path(out).write_text(json.dumps(m, indent=1) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
