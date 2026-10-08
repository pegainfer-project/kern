#!/usr/bin/env python3
"""Report greedy top-1/top-2 margins from kern --probe-dir logits.

The probe manifest writes full FP32 logits (the normal head_argmax path). This
is intentionally stdlib-only so it runs on the CPU host. MTP spec_head writes
only token ids; use this reader after adding a diagnostic full-logit output to
that op, not on round_tokens.bin.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
import struct
from pathlib import Path


def _best_two(values):
    b1 = (-math.inf, 2**63 - 1)
    b2 = (-math.inf, 2**63 - 1)
    for i, v in enumerate(values):
        # Match spec_head.cu: larger value wins; ties use the lower id.
        x = (float(v), i)
        if x[0] > b1[0] or (x[0] == b1[0] and x[1] < b1[1]):
            b2, b1 = b1, x
        elif x[0] > b2[0] or (x[0] == b2[0] and x[1] < b2[1]):
            b2 = x
    return b1, b2


def read_rows(path: Path, vocab: int, dtype: str):
    data = path.read_bytes()
    width = 4 if dtype == "f32" else 2
    row_bytes = vocab * width
    if len(data) % row_bytes:
        raise ValueError(f"{path}: {len(data)} bytes is not a whole {vocab}-logit row")
    rows = len(data) // row_bytes
    out = []
    for r in range(rows):
        chunk = data[r * row_bytes : (r + 1) * row_bytes]
        if dtype == "f32":
            vals = struct.unpack(f"<{vocab}f", chunk)
        else:
            # BF16 is not a CPython struct type. Expand its bits to FP32.
            u = struct.unpack(f"<{vocab}H", chunk)
            vals = (struct.unpack("<f", struct.pack("<I", x << 16))[0] for x in u)
        top1, top2 = _best_two(vals)
        out.append({
            "row": r,
            "top1_id": top1[1],
            "top1": top1[0],
            "top2_id": top2[1],
            "top2": top2[0],
            "margin": top1[0] - top2[0],
        })
    return out


def step_number(path: Path):
    m = re.search(r"decode(\d+).*logits\.bin$", path.name)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("probe_dir", type=Path)
    ap.add_argument("--vocab", type=int, default=154880)
    ap.add_argument("--dtype", choices=("f32", "bf16"), default="f32")
    ap.add_argument("--glob", default="decode*.logits.bin")
    ap.add_argument("--json", type=Path, help="write machine-readable output")
    args = ap.parse_args()

    files = [Path(p) for p in glob.glob(str(args.probe_dir / args.glob))]
    files.sort(key=lambda p: (step_number(p) is None, step_number(p) or 0, p.name))
    if not files:
        ap.error(f"no files matching {args.glob!r} under {args.probe_dir}")
    result = []
    for p in files:
        rows = read_rows(p, args.vocab, args.dtype)
        result.append({"file": str(p), "step": step_number(p), "rows": rows})
        for x in rows:
            print(f"{p.name} row={x['row']} top1={x['top1_id']} top2={x['top2_id']} margin={x['margin']:.9g}")
    if args.json:
        args.json.write_text(json.dumps({"vocab": args.vocab, "dtype": args.dtype, "files": result}, indent=2) + "\n")


if __name__ == "__main__":
    main()
