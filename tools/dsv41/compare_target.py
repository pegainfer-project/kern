"""Compare complete target logits only after checking identical token histories."""
import argparse
import json
from pathlib import Path

import numpy as np


def load(directory):
    meta = json.loads((directory / "trace.json").read_text())
    tokens = json.loads((directory / "tokens.json").read_text())
    if tokens and isinstance(tokens[0], list):
        if len(tokens) != 1:
            raise ValueError("comparison currently requires batch one")
        tokens = tokens[0]
    shape = meta.get("logits_shape")
    if shape is not None:
        rows, batch, width = shape
        if batch != 1:
            raise ValueError("comparison currently requires batch one")
    else:
        rows, width = meta["logits_rows"], meta["logits_width"]
    path = directory / "target_logits.f32"
    if rows <= 0 or path.stat().st_size != rows * width * 4:
        raise ValueError(f"invalid complete logits artifact: {path}")
    logits = np.memmap(path, dtype="<f4", mode="r", shape=(rows, width))
    return meta["logits_first_input_position"], tokens, logits


def compare(actual, reference):
    first, tokens, logits = load(actual)
    ref_first, ref_tokens, ref_logits = load(reference)
    if first != ref_first or logits.shape[1] != ref_logits.shape[1]:
        raise ValueError("prompt lengths or vocabulary dimensions differ")
    rows = min(len(logits), len(ref_logits))
    # Each row predicts the token after the processed input position. Compare
    # the entire causal prefix, including prompt tokens, before comparing math.
    end = first + rows
    if len(tokens) < end or len(ref_tokens) < end or tokens[:end] != ref_tokens[:end]:
        raise ValueError("target histories differ; use teacher forcing for logits comparison")
    details = []
    for i in range(rows):
        x, y = logits[i].astype(np.float64), ref_logits[i].astype(np.float64)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"nonfinite logits at row {i}")
        a, b = int(x.argmax()), int(y.argmax())
        d = x - y
        top = np.argpartition(y, -2)[-2:]
        ordered = sorted(top, key=lambda t: y[t], reverse=True)
        lx = x - x.max(); lx -= np.log(np.exp(lx).sum())
        ly = y - y.max(); ly -= np.log(np.exp(ly).sum())
        details.append({
            "input_position": first + i, "kern_top1": a, "reference_top1": b,
            "top1_equal": a == b, "reference_margin": float(y[ordered[0]] - y[ordered[1]]),
            "max_abs": float(abs(d).max()), "rmse": float(np.sqrt(np.mean(d*d))),
            "relative_squared": float(np.dot(d, d) / max(np.dot(y, y), 1e-30)),
            "reference_to_kern_kl": float(np.dot(np.exp(ly), ly-lx)),
        })
    return {
        "scope": "Full-vocabulary target logits on identical causal token histories",
        "rows": rows, "kern_rows": len(logits), "reference_rows": len(ref_logits),
        "top1_matches": sum(r["top1_equal"] for r in details),
        "max_abs": max(r["max_abs"] for r in details),
        "max_relative_squared": max(r["relative_squared"] for r in details),
        "mean_reference_to_kern_kl": sum(r["reference_to_kern_kl"] for r in details) / rows,
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kern", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.kern, args.reference)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "details"}, indent=2))


if __name__ == "__main__":
    main()
