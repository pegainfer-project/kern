#!/usr/bin/env python3
"""Record a measured choice in the index: the variant `kern bench` found best for an op on a workload, with
the bench's report in the blob store as the evidence (docs/registry.md `[[pick]]`).

    tools/kernels/pick.py <family> <op> <shape> <variant> <bench.json> [--measured "2026-09-19 tray03"] [--upload]
"""
import argparse
import datetime
import pathlib
import platform
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import index  # noqa: E402
import store  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("family")
    ap.add_argument("op")
    ap.add_argument("shape")
    ap.add_argument("variant")
    ap.add_argument("report", type=pathlib.Path)
    ap.add_argument("--measured", default=f"{datetime.date.today().isoformat()} {platform.node()}")
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args()
    doc = index.load(a.family)
    index.variant_by_name(a.family, a.variant, doc)
    sha = store.put(a.report)
    picks = [p for p in doc.get("pick", []) if (p["op"], p["shape"]) != (a.op, a.shape)]
    doc["pick"] = picks + [{"op": a.op, "shape": a.shape, "variant": a.variant, "report": sha, "measured": a.measured}]
    index.save(doc)
    print(f"{a.family}: {a.op} on {a.shape} -> {a.variant} (report @{sha[:12]})", file=sys.stderr)
    if a.upload:
        store.upload([sha], f"{a.family}: bench report for {a.op} on {a.shape}")


if __name__ == "__main__":
    main()
