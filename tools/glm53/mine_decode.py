#!/usr/bin/env python3
"""Mine the GLM-5.3 sglang capture into a decode-step recipe.

Slices bs=1 decode forwards (prompt1) out of a capture's launches.jsonl, uses
forward 1 (second decode step) as the canonical sequence, classifies every
pointer parameter by comparing against the other steps of prompt1 and the
steps of prompt2 (a different request):

  weight    same allocation in prompt1 and prompt2 steps
  state     same within one prompt steps, different across prompts
  rotating  changes between consecutive steps of one prompt

The recipe stops at the vocab argmax; everything past it is sglang serving
bookkeeping that the kern runtime replaces.

Output: $GLM53_ARTIFACTS/recipe.json (default ./glm53-artifacts/recipe.json)

    python3 tools/glm53/mine_decode.py [launches.jsonl]
"""

import os
import json
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "dumped-kernels-glm53-sglang/launches.jsonl"

EMB = {"p1": 253516, "p2": 256315, "b2": 293057}
N_FORWARD = 6  # decode steps per prompt (7th slice touches the next prefill)


def load():
    recs = []
    with open(PATH) as f:
        for line in f:
            recs.append(json.loads(line))
    return recs


def forwards(recs, start, n):
    out, cur = [], []
    for r in recs[start:]:
        if r["symbol"].startswith("_vocab_parallel_embedding") and cur:
            out.append(cur)
            if len(out) == n:
                break
            cur = []
        cur.append(r)
    return out


def alloc_of(p):
    if "pointer" not in p:
        return None
    return (p["pointer"]["range_start"], p["pointer"]["range_size"])


def occ_list(fwd):
    seen, out = {}, []
    for r in fwd:
        k = seen.get(r["symbol"], 0)
        seen[r["symbol"]] = k + 1
        out.append((r["symbol"], k))
    return out


def cut_at_argmax(fwd):
    for i, r in enumerate(fwd):
        if "ArgMaxOps" in r["symbol"]:
            return fwd[: i + 1]
    raise SystemExit("no argmax in forward")


def main():
    recs = load()
    sets = {k: [cut_at_argmax(f) for f in forwards(recs, s, N_FORWARD)] for k, s in EMB.items()}
    for k, fs in sets.items():
        print(k, "forwards:", len(fs), "lengths:", sorted({len(f) for f in fs}))
    canon = [occ_list(f) for f in sets["p1"]]
    ref = canon[1]  # second step: steady state
    for j, o in enumerate(canon):
        if j == 1:
            continue
        # tolerate the tail-bookkeeping wobble: compare only the first 1345 sites
        assert o[:1340] == ref[:1340], f"p1 step {j} model body diverges"
    occs_b2 = occ_list(sets["b2"][1])

    recipe = []
    for i, (sym, k) in enumerate(ref):
        rows = [fs[i] for fs in sets["p1"]]
        rows2 = [fs[i] for fs in sets["p2"]]
        r0 = rows[1]
        params = []
        for pi, p in enumerate(r0["params"] or []):
            e = {"size": p["size"]}
            a = alloc_of(p)
            if a is None:
                if p["size"] <= 8:
                    e["scalar"] = p["data"]
                else:
                    e["blob"] = p["data"]
            else:
                others = [alloc_of((fr["params"] or [])[pi]) for fr in rows[:1] + rows[2:]]
                a2 = alloc_of((rows2[1]["params"] or [])[pi])
                if all(x == a for x in others) and a2 == a:
                    e["ptr"] = "weight"
                elif all(x == a for x in others):
                    e["ptr"] = "state"
                else:
                    e["ptr"] = "rotating"
                e["alloc"] = [a[0], a[1]]
                e["offset"] = int.from_bytes(bytes.fromhex(p["data"]), "little") - int(a[0], 16) if p.get("data") else 0
            params.append(e)
        entry = {
            "i": i, "symbol": sym, "grid": r0["grid"], "block": r0["block"],
            "smem": r0["dynamic_shared_mem_bytes"], "params": params,
        }
        if i < len(occs_b2) and occs_b2[i][0] == sym:
            entry["grid_b2"] = sets["b2"][1][i]["grid"]
        recipe.append(entry)
    out = pathlib.Path(os.environ.get("GLM53_ARTIFACTS", "glm53-artifacts")) / "recipe.json"
    with open(out, "w") as f:
        json.dump({"bs1": recipe}, f, indent=1)
    print(f"wrote {out}:", len(recipe), "call sites")
    allocs = {}
    for cs in recipe:
        for p in cs["params"]:
            if p.get("ptr") == "weight":
                allocs.setdefault(tuple(p["alloc"]), set()).add(cs["i"])
    print("distinct weight allocations:", len(allocs))
    for a, sites in sorted(allocs.items(), key=lambda kv: -kv[0][1])[:20]:
        print("  %16s size=%12d sites=%d first=%d" % (a[0], a[1], len(sites), min(sites)))


if __name__ == "__main__":
    main()
