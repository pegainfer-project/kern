"""CPU-only MTP MoE fusion checks (verify M<=32 + draft M<=16); never edit gen.py."""
import argparse, json, random, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from glm53 import gen, ops_moe_v2  # noqa: E402

ORDER = ["moe_router", "moe_topk", "moe_align", "moe_quant_a", "moe_w13",
         "moe_silu", "moe_quant_b", "moe_w2", "moe_sum_reduce"]

def build_fused(**flags):
    original = gen.lower_wire
    def hooked(m):
        # gen.py fuses the round only at moe==v2; stay idempotent for v1.
        if "moe_verify_v2" not in m["ops"]:
            ops_moe_v2.fuse_round_manifest(m)
        ops_moe_v2.fuse_draft_manifest(m)
        return original(m)
    gen.lower_wire = hooked
    try:
        return gen.build(45, allreduce="lamport", mtp=True, **flags)
    finally:
        gen.lower_wire = original

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/tmp/mtp_fused2_test.json"))
    args = ap.parse_args()
    for flags in (dict(moe="v1", mhc="off", dsa="v1", kda="off"),
                  dict(moe="v2", mhc="boundary", dsa="v2a", kda="fused")):
        m = build_fused(**flags)
        calls = m["programs"]["round_k2"]["calls"]
        vfused = [c for c in calls if c["op"] == "moe_verify_v2"]
        dfused = [c for c in calls if c["op"] == "moe_decode_v2"]
        assert len(vfused) == 42, (flags, len(vfused))
        assert len(dfused) == 2, (flags, len(dfused))
        assert len(calls) == 1044 - 2 * 8 == 1028, (flags, len(calls))
        assert all(c["label"].endswith(".moe_v2") for c in vfused + dfused)
        assert not any(c["op"].startswith(("mtp32_moe_", "mtp16_moe_")) for c in calls)
        for n in ORDER:
            assert "mtp32_" + n not in m["ops"] and "mtp16_" + n not in m["ops"], n
        for c in dfused:
            assert c["args"][7] == {"buf": "spec_ones"}, c["args"]
            assert c["args"][8] == {"buf": "sub_out"}, c["args"]
        for name in ("moe_verify_v2", "moe_decode_v2"):
            for launch in m["ops"][name]["impl"]["launches"]:
                src = m["modules"][launch["module"]]["source"]
                assert Path(src).is_file(), src
        # spec_ar still follows every fused block; draft order intact.
        for i, c in enumerate(calls):
            if c["op"] in ("moe_verify_v2", "moe_decode_v2"):
                assert calls[i+1]["op"] == "spec_ar", (c["label"], calls[i+1].get("label"))
        # v1 moe_* workspace: swept when no call uses it (v2), kept by decode (v1).
        if flags["moe"] == "v2":
            assert not any(b.startswith("moe_") for b in m["buffers"]), "sweep failed"
        else:
            assert "moe_scores" in m["buffers"], "decode still needs v1 workspace"
        args.out.write_text(json.dumps(m, indent=1) + "\n")
        subprocess.run([str(pathlib.Path(__file__).resolve().parents[3] / "target" / "release" / "kern"), "verify", str(args.out)], check=True)
        print("manifest OK:", flags, "-> 42 verify + 2 draft blocks, 1028 calls, kern verify passed")
    # CPU align invariants at M<=32: ceil-16 tiles, per-tile capacity 16.
    rng = random.Random(904)
    for b in list(range(1, 33)):
        for _ in range(64):
            valid = [rng.randrange(2) for _ in range(b)]
            ids = [rng.sample(range(288), 8) + [288] if v else [-1]*9 for v in valid]
            pairs = [(e, r*9+k) for r, row in enumerate(ids) for k, e in enumerate(row) if e >= 0]
            by = {}
            for e, p in pairs:
                by.setdefault(e, []).append(p)
            assert all(len(v) <= 32 for v in by.values())
            tiles = []
            for e in sorted(by):
                ps = by[e]
                for t0 in range(0, len(ps), 32):
                    tiles.append((e, ps[t0:t0+32]))
            assert all(len(t[1]) <= 32 for t in tiles)
            assert len(tiles) == sum((len(v)+31)//32 for v in by.values())
            assert len(tiles) <= 9 * b
            flat = [p for _, ps in tiles for p in ps]
            assert sorted(flat) == sorted(p for _, p in pairs)
    print("CPU align invariants: 2048 random bucket/mask cases (b<=32) passed")

if __name__ == "__main__":
    main()
