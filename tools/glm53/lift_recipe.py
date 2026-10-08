#!/usr/bin/env python3
"""Enrich the decode recipe with cubin register/param facts for pinning.

For every distinct symbol in the recipe, find the capture module(s) that
define it (DumpIndex over the sglang capture dump), read register count and
parameter sizes with cuobjdump, and emit one pinning candidate per call
site: (symbol, regs, param sizes) -> module sha256.

Output: $GLM53_ARTIFACTS/skeleton.json (default ./glm53-artifacts/skeleton.json)
    python3 tools/glm53/lift_recipe.py
"""

import os
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from kern_manifest import DumpIndex  # noqa: E402

DUMP = "dumped-kernels-glm53-sglang"
ART = pathlib.Path(os.environ.get("GLM53_ARTIFACTS", "glm53-artifacts"))
RECIPE = str(ART / "recipe.json")
OUT = str(ART / "skeleton.json")


def main():
    rec = json.loads(pathlib.Path(RECIPE).read_text())["bs1"]
    idx = DumpIndex(DUMP)
    print("dump modules:", len(idx.mods))
    # symbol -> [(regs, sizes, sha)]
    table = {}
    missing = []
    for sha, (mod, fns) in idx.mods.items():
        for fn, regs in fns.items():
            table.setdefault(fn, set()).add(regs)
    out = {}
    for cs in rec:
        sym = cs["symbol"]
        if sym in out or not cs["params"]:
            continue
        sizes = [p["size"] for p in cs["params"]]
        if sym not in table:
            missing.append(sym)
            continue
        regs_options = table[sym]
        pinned = None
        for regs in regs_options:
            try:
                sha = idx.pin(sym, regs=regs, sizes=sizes)
                pinned = (regs, sha)
                break
            except AssertionError:
                continue
        out[sym] = {
            "regs_options": sorted(regs_options),
            "pinned": pinned,
            "n_sites": sum(1 for c in rec if c["symbol"] == sym),
        }
    pathlib.Path(OUT).write_text(json.dumps(out, indent=1))
    print("symbols:", len(out), "pinned:", sum(1 for v in out.values() if v["pinned"]))
    print("missing (no params or not in dump):")
    for s in sorted(set(missing)):
        print("  ", s[:120])


if __name__ == "__main__":
    main()
