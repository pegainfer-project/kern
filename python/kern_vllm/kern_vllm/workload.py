"""A step trace (`trace.py`) as a weighted `kern bench` workload.

    python -m kern_vllm.workload <trace.jsonl> [--samples 16] [--seed 0] > workload.toml

A step is the program calls the model makes of it: its one-token rows
together, one sequence each at its previous length, and every longer row
alone. Calls are counted into buckets, and each bucket becomes one shape
weighted by its count, so `kern bench` prices the traced traffic.

A decode call's sequence count rounds up to vLLM's CUDA graph sizes, which
is what a pure decode step runs at; its context is the mean previous
length. Lengths and prefill rows round to the nearest half octave.

vLLM's warmup and graph-capture batches repeat one (query, seq) pair across
every row or decode from nothing; they are left out.
"""
import argparse
import collections
import json
import math

GRAPH_SIZES = [1, 2, 4] + list(range(8, 257, 8))


def dummy(step: dict) -> bool:
    rows = list(zip(step["q"], step["s"]))
    return (len(rows) > 1 and len(set(rows)) == 1) or any(q == s == 1 for q, s in rows)


def calls(step: dict) -> list[tuple[int, int, list[int]]]:
    """(sequences, rows per sequence, previous length per sequence)."""
    rows = list(zip(step["q"], step["s"]))
    decode = [s - 1 for q, s in rows if q == 1]
    return ([(len(decode), 1, decode)] if decode else []) + [(1, q, [s - q]) for q, s in rows if q > 1]


def half_octave(n: float) -> int:
    return 0 if n < 1 else round(2 ** (round(math.log2(n) * 2) / 2))


def bucket(call: tuple[int, int, list[int]]) -> tuple[int, int, int]:
    groups, rows, context = call
    if rows == 1:
        return next(g for g in GRAPH_SIZES if g >= groups), 1, half_octave(sum(context) / len(context))
    return 1, half_octave(rows), half_octave(context[0])


def workload(steps: list[dict], samples: int, seed: int) -> str:
    counts = collections.Counter(bucket(c) for s in steps if not dummy(s) for c in calls(s))
    body = [f"[[sweep]]\ngroups = [{g}]\nrows = [{r}]\ncontext = [{c}]\nweight = {n}\n"
            for (g, r, c), n in sorted(counts.items())]
    return "\n".join([f"# {sum(counts.values())} calls in {len(counts)} buckets",
                      f"samples = {samples}", f"seed = {seed}", ""] + body)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace")
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    print(workload([json.loads(l) for l in open(a.trace)], a.samples, a.seed), end="")


if __name__ == "__main__":
    main()
