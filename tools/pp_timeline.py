#!/usr/bin/env python3
"""A pipeline's timeline from its stages' `pp item` log lines (kern-serve --pp-*).

Every stage logs when it issued each item and when the item left its
device (`issued_us`, `done_us`); the item number is the same on every
stage. A stage starts item k once it has issued it, is done with k - 1 and
its upstream is done with k, so per stage and item:

  wait = start - done[k-1]   (idle: nothing issued yet, or the upstream had not delivered)
  held = done - start        (its own work, plus any wait for the downstream to take)

and done to done is the stage's period. Across trays `done_us` is two wall
clocks; `--skew S=us` shifts stage S and every later one.

  pp_timeline.py stage*.log [--skew 4=-1200] [--from K] [--to K] [--gantt]
"""
import argparse
import re
import statistics
from collections import defaultdict

LINE = re.compile(r"pp item stage=(\d+) item=(\d+) rows=(\d+) issued_us=(\d+) done_us=(\d+)")


def parse(paths):
    issued, done, rows = defaultdict(dict), defaultdict(dict), {}
    for p in paths:
        for m in map(LINE.search, open(p, errors="replace")):
            if m:
                s, k, r, i, t = map(int, m.groups())
                issued[s][k], done[s][k], rows[k] = i, t, r
    return issued, done, rows


def timeline(issued, done, skew):
    stages = sorted(done)
    shift = {s: sum(v for at, v in skew.items() if at <= s) for s in stages}
    d = {s: {k: t + shift[s] for k, t in done[s].items()} for s in stages}
    i = {s: {k: t + shift[s] for k, t in issued[s].items()} for s in stages}
    items = sorted(set.intersection(*(set(v) for v in d.values())))
    start = {s: {} for s in stages}
    for s in stages:
        for k in items:
            ready = [i[s][k], d[s].get(k - 1, 0)] + ([d[s - 1][k]] if s - 1 in d else [])
            start[s][k] = max(ready)
    return stages, items, d, start


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")


def report(stages, items, d, start, rows, lo, hi):
    ks = [k for k in items if lo <= k <= hi and k - 1 in d[stages[0]]]
    print(f"items {ks[0]}..{ks[-1]} ({len(ks)}), rows/item mean {statistics.mean(rows[k] for k in ks):.0f}")
    print(f"{'stage':>5} {'period p50':>11} {'p90':>8} {'held p50':>9} {'wait p50':>9} {'wait share':>11}   (ms)")
    for s in stages:
        period = [(d[s][k] - d[s][k - 1]) / 1e3 for k in ks]
        held = [(d[s][k] - start[s][k]) / 1e3 for k in ks]
        wait = [(start[s][k] - d[s][k - 1]) / 1e3 for k in ks]
        share = sum(wait) / sum(period) if sum(period) else 0
        print(f"{s:>5} {pct(period, .5):>11.2f} {pct(period, .9):>8.2f} {pct(held, .5):>9.2f} "
              f"{pct(wait, .5):>9.2f} {share:>10.1%}")


def gantt(stages, items, d, start, lo, hi, width=120):
    ks = [k for k in items if lo <= k <= hi]
    t0 = min(start[s][ks[0]] for s in stages)
    t1 = max(d[s][ks[-1]] for s in stages)
    scale = (t1 - t0) / width or 1
    for s in stages:
        row = ["."] * width
        for k in ks:
            a, b = int((start[s][k] - t0) / scale), int((d[s][k] - t0) / scale)
            for x in range(max(a, 0), min(max(b, a + 1), width)):
                row[x] = str(k % 10)
        print(f"S{s} |{''.join(row)}|")
    print(f"     {(t1 - t0) / 1e3:.1f} ms across, '.' idle, digit = item number mod 10")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--skew", action="append", default=[], help="S=us: shift stage S and later by us")
    ap.add_argument("--from", dest="lo", type=int, default=1)
    ap.add_argument("--to", dest="hi", type=int, default=1 << 62)
    ap.add_argument("--gantt", action="store_true")
    a = ap.parse_args()
    issued, done, rows = parse(a.logs)
    skew = {int(s): int(v) for s, v in (x.split("=") for x in a.skew)}
    stages, items, d, start = timeline(issued, done, skew)
    report(stages, items, d, start, rows, a.lo, a.hi)
    if a.gantt:
        gantt(stages, items, d, start, a.lo, min(a.hi, a.lo + 19))


if __name__ == "__main__":
    main()
