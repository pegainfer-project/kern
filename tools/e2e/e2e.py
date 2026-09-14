#!/usr/bin/env python3
"""kern's GPU gate, driven the way a client drives it.

Every target of a kern.toml goes through the same scenarios over the
binaries a user has (`kern`, `kern-serve`) and the interface a user has
(the OpenAI endpoint, the server's log line), so a target is a name in a
toml and a scenario reads like a client:

  kern_test         `kern test` PASSes (targets with a `reference`)
  conc1             one request at a time equals `kern run`, token for token
  repeat            the same prompt again reports the hit its checkpoints allow
  turn2_warm        prompt + answer + a question, right after: the hit and the
                    same tokens a cold server gives it
  concurrent        every prompt at once: all finish, acceptance holds up
  abort             a client that hangs up mid-stream leaves the server whole
  park_wake         a pool of a few pages and a host tier: the coldest
                    checkpoints park, a prompt hitting one wakes it, the answer
                    is the cold one
  slot_growth       a stateful manifest served with two slots grows them out
                    of free pages as requests finish, answers unchanged
  rows1             a speculative manifest served with `--rows 1` equals
                    `kern run --rows 1`

A target is skipped, not failed, when its artifacts are not on this
machine. `kern run` is the oracle for single-rank targets; a tray target
(EP4) has no `kern run`, its cold conc1 answers are the oracle for the
warm ones. Byte identity of greedy token ids is the gate everywhere; a
divergence is reported with its position so a near-tie can be argued
from the logits, never assumed.

Runs on the machine with the GPUs:
  python3 tools/e2e/e2e.py --gpus 0,1,2,3 --out results/ [--config kern.toml] [--targets a b]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

PROMPTS = [
    "The lighthouse keeper had not seen a ship in eleven days, and the silence of the bay was starting to feel like",
    "To make a proper risotto, begin by warming the stock in a separate pan, because cold stock",
    "In 1911 the expedition reached the plateau with only three sledges left. The diary for that week records",
    "A hash map trades memory for speed: instead of scanning every entry, it computes",
    "The village market opened before dawn. By the time the sun cleared the hills, the fish stalls",
    "Photosynthesis converts light into chemical energy in two stages. In the first stage, chlorophyll",
    "She unfolded the map on the hood of the car and traced the river with one finger until",
    "The difference between a symphony and a concerto is mostly a question of who",
    "When the bridge was finally closed for repairs, the town council discovered that nobody",
    "A good chess opening does three things: it develops pieces, controls the center, and",
    "Rain had turned the trail to mud, so the surveyors camped early and spent the evening",
    "Compilers optimize loops by hoisting invariant computations out of them, which means",
]
SUFFIX = "\n\nNow explain that again for a ten-year-old, in two sentences."
LOG_FILTER = "kern_serve=debug,kern_runtime=info"


@dataclass
class Target:
    name: str
    config: Path
    manifest: Path
    kernels: Path
    weights: list[Path]
    reference: Path | None
    ranks: int
    stateful: bool

    def missing(self) -> list[str]:
        """Artifacts not on this machine (a weights glob is checked by its fixed prefix)."""
        out = [str(p) for p in (self.manifest, self.kernels) if not p.exists()]
        for w in self.weights:
            fixed = Path(*[c for c in w.parts if not any(x in c for x in "{*")][: len(w.parts)])
            top = fixed
            while not top.exists() and top != top.parent:
                top = top.parent
            if not fixed.exists() and not (fixed.parent.exists() and any(x in w.name for x in "{*")):
                out.append(str(w))
        return out


def load_targets(config: Path) -> list[Target]:
    doc = tomllib.loads(config.read_text())
    base = config.parent

    def path(p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else (base / q)

    out = []
    for name, t in doc.get("targets", {}).items():
        manifest = path(t["manifest"])
        ranks, stateful = 1, False
        if manifest.exists():
            m = json.loads(manifest.read_text())
            for n in (m.get("topology") or {}).get("groups", {}).values():
                ranks *= int(n)
            # A value may be a named constant; a per-sequence state names one either way.
            stateful = any(s.get("bytes_per_seq") not in (None, 0) for s in m.get("states", {}).values())
        out.append(
            Target(
                name=name,
                config=config,
                manifest=manifest,
                kernels=path(t["kernels"]),
                weights=[path(w) for w in t.get("weights", [])],
                reference=path(t["reference"]) if t.get("reference") else None,
                ranks=ranks,
                stateful=stateful,
            )
        )
    return out


@dataclass
class Check:
    name: str
    ok: bool | None  # None: reported, not gated
    note: str = ""


@dataclass
class Report:
    target: str
    facts: dict = field(default_factory=dict)
    checks: list[Check] = field(default_factory=list)
    counters: dict = field(default_factory=dict)
    skipped: str = ""

    def add(self, name: str, ok: bool | None, note: str = "") -> None:
        self.checks.append(Check(name, ok, note))
        mark = {True: "ok  ", False: "FAIL", None: "info"}[ok]
        print(f"  [{mark}] {name}: {note}", flush=True)


KV = re.compile(r'(\w+)=("(?:[^"\\]|\\.)*"|\S+)')


def kv(line: str) -> dict:
    out = {}
    for k, v in KV.findall(line):
        if v.startswith('"'):
            out[k] = v[1:-1]
        else:
            try:
                out[k] = int(v)
            except ValueError:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
    return out


class Server:
    """One kern-serve over a target, started through `kern server` and read through its log."""

    def __init__(self, a, target: Target, gpus: list[int], port: int, flags: dict[str, str], log: Path):
        self.a, self.target, self.gpus, self.port, self.log = a, target, gpus, port, log
        self.flags = {**a.server_flags, **flags}
        self.proc = None
        self.url = f"http://127.0.0.1:{port}"
        self.facts = {}

    def __enter__(self):
        cmd = [str(self.a.kern), "--config", str(self.target.config), "server", self.target.name, "--port", str(self.port)]
        cmd += ["--gpus", ",".join(map(str, self.gpus))] + [x for k, v in self.flags.items() for x in (k, v)]
        env = dict(os.environ, KERN_SERVE_BIN=str(self.a.kern_serve), RUST_LOG=LOG_FILTER)
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(cmd, stdout=open(self.log, "wb"), stderr=subprocess.STDOUT, env=env)
        t0 = time.monotonic()
        while True:
            text = self.log.read_text(errors="replace")
            ready = next((l for l in text.splitlines() if " scheduler ready " in l), None)
            if ready and " serving " in text:
                self.facts = kv(ready)
                self.facts["load_s"] = round(time.monotonic() - t0, 1)
                return self
            if self.proc.poll() is not None:
                raise RuntimeError(f"kern-serve exited {self.proc.returncode} while loading; tail:\n" + "\n".join(text.splitlines()[-8:]))
            if time.monotonic() - t0 > self.a.load_timeout:
                self.proc.kill()
                raise RuntimeError("kern-serve did not come up in time")
            time.sleep(1)

    def __exit__(self, *_):
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def lines(self, tag: str) -> list[dict]:
        return [kv(l) for l in self.log.read_text(errors="replace").splitlines() if f" {tag} " in l]

    def counters(self) -> dict:
        """Sums of the per-window counters over every stats line, the gauges as last seen."""
        sums = ("parks", "wakes", "host_hits", "resident_hits", "evictions", "host_evictions", "prefill_tokens")
        out = {k: 0 for k in sums}
        for s in self.lines("stats"):
            for k in sums:
                out[k] += s.get(k, 0) or 0
            for k in ("remaps", "slots", "slots_used", "checkpoints", "parked", "host_gib", "accept_pct", "accepted"):
                if k in s:
                    out[k] = s[k]
        return out

    def flush_stats(self) -> None:
        """The stats line covers a 5 s window that ends at a step: wait one out and take a step."""
        time.sleep(5.2)
        complete(self.url, "Hi", 1)
        time.sleep(0.5)


def post(url: str, path: str, body: dict, timeout: float = 900) -> dict:
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def complete(url: str, prompt, max_tokens: int, timeout: float = 900) -> dict:
    body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "return_token_ids": True}
    r = post(url, "/v1/completions", body, timeout)
    c = r["choices"][0]
    details = r["usage"].get("prompt_tokens_details") or {}
    ids, finish = c["token_ids"], c["finish_reason"]
    # The frontend's ids end with the stop token that ended the answer;
    # `kern run` stops before it.
    stop = ids[-1] if finish == "stop" and ids else None
    return {
        "ids": ids[:-1] if stop is not None else ids,
        "stop": stop,
        "prompt_ids": c["prompt_token_ids"],
        "cached": details.get("cached_tokens", 0) or 0,
        "finish": finish,
    }


def tokenize(url: str, text: str) -> list[int]:
    return post(url, "/tokenize", {"model": MODEL, "prompt": text})["tokens"]


def stream_then_hang_up(url: str, prompt: str, chunks: int) -> int:
    """Read `chunks` SSE chunks of a streaming completion and close the socket."""
    import http.client

    host, port = url.removeprefix("http://").split(":")
    conn = http.client.HTTPConnection(host, int(port), timeout=120)
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": 4096, "temperature": 0, "stream": True})
    conn.request("POST", "/v1/completions", body=body, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    got = 0
    while got < chunks:
        line = resp.readline()
        if not line:
            break
        if line.startswith(b"data:"):
            got += 1
    conn.sock.close()
    return got


def divergence(a: list[int], b: list[int]) -> str:
    if a == b:
        return f"identical ({len(a)} tokens)"
    k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return f"diverge at token {k}/{len(a)} vs {len(b)}: {a[k:k+4]} vs {b[k:k+4]}"


class Same:
    """Byte identity over a set of answers, a divergence excused only by a near-tie
    the oracle can show at that very token (the prompt and the tokens up to it
    run through `kern run`), never assumed."""

    def __init__(self, oracle: Oracle, rows: int):
        self.oracle, self.rows, self.n, self.same, self.notes, self.excused = oracle, rows, 0, 0, [], 0

    def add(self, label: str, prompt: list[int], got: list[int], want: list[int]) -> None:
        self.n += 1
        if got == want:
            self.same += 1
            return
        k = next((i for i, (x, y) in enumerate(zip(got, want)) if x != y), None)
        note = f"{label}: {divergence(got, want)}"
        if k is not None:
            verdict = self.oracle.near_tie(prompt + want[:k], got[k], want[k], self.rows)
            if verdict:
                note += f" ({verdict})"
                self.excused += verdict.startswith("near-tie")
        self.notes.append(note)

    @property
    def ok(self) -> bool:
        return self.same + self.excused == self.n

    def __str__(self) -> str:
        s = f"{self.same}/{self.n} identical"
        if self.excused:
            s += f", {self.excused} near-tie"
        return s + ("; " + "; ".join(self.notes) if self.notes else "")


class Oracle:
    """`kern run` answers, cached by (prompt, steps, rows), and the logit margin
    behind any token of one; only single-rank targets have an oracle, and it
    runs on a GPU the server is not on."""

    def __init__(self, a, target: Target, gpu: int | None, path: Path):
        self.a, self.target, self.gpu, self.path = a, target, gpu, path
        self.cache = json.loads(path.read_text()) if path.exists() else {}
        m = json.loads(target.manifest.read_text()) if target.manifest.exists() else {}
        logits = m.get("buffers", {}).get("logits", {})
        vocab = logits.get("shape", [None, None])[-1]
        self.vocab = m.get("constants", {}).get(vocab, vocab) if isinstance(vocab, str) else vocab
        self.dtype = logits.get("dtype")

    @property
    def available(self) -> bool:
        return self.target.ranks == 1 and self.gpu is not None

    def command(self, prompt: str | list[int]) -> list[str]:
        cmd = [str(self.a.kern), "--config", str(self.target.config), "run", self.target.name, "--gpu", str(self.gpu)]
        return cmd + (["--prompt-ids", ",".join(map(str, prompt))] if isinstance(prompt, list) else ["--prompt", prompt])

    def near_tie(self, context: list[int], a: int, b: int, rows: int) -> str | None:
        """Whether the server's `a` and `kern run`'s `b`, the first tokens on
        which two answers to `context` differ, are a near-tie: in the logits of
        the step right after `context` (a plain step, its own run) both are
        within 4 bf16 ULPs of the top one, ULPs at the top logit's scale
        (docs/test.md's logit rule). The verdict, or None without evidence."""
        if self.target.ranks != 1 or self.gpu is None or self.dtype != "bf16" or not isinstance(self.vocab, int):
            return None
        d = self.path.parent / f"probe-{hashlib.sha1(json.dumps(context).encode()).hexdigest()[:12]}"
        cmd = self.command(context) + ["--steps", "2", "--probe-dir", str(d), "--probe-steps", "2", "--probe-labels", ""]
        if rows > 1:
            cmd += ["--rows", "1"]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if p.returncode != 0:
            return f"no evidence (kern run --probe-dir failed: {p.stderr[-200:]})"
        # The first generated token comes out of the chunk forward when it
        # emits, else out of the first decode step: the dump whose logits
        # pick the token it dumped.
        for name in ("chunk", "decode0"):
            f = d / f"{name}.logits.bin"
            if f.exists():
                vals = self.logits(f)
                order = sorted(range(len(vals)), key=lambda i: -vals[i])
                if order[0] in self.first_token(d / f"{name}.tokens.bin"):
                    break
        else:
            return "no evidence (no dumped step produced the first token)"
        x1 = vals[order[0]]
        ulp = 2.0 ** (int(abs(x1)).bit_length() - 8) if x1 != 0 else 2.0 ** -133
        gap = lambda t: (x1 - vals[t]) / ulp
        verdict = "near-tie" if max(gap(a), gap(b)) <= 4 else "confident flip"
        top = ", ".join(f"{i}:{vals[i]:g}" for i in order[:4])
        return f"{verdict}: top {top}; server's {a} #{order.index(a) + 1} {gap(a):.1f} ULPs down, kern run's {b} #{order.index(b) + 1} {gap(b):.1f}"

    def first_token(self, f: Path) -> set[int]:
        """The first token of a dumped tokens buffer, read as i32 and as i64."""
        import struct

        raw = f.read_bytes()
        return {struct.unpack("<i", raw[:4])[0]} | ({struct.unpack("<q", raw[:8])[0]} if len(raw) >= 8 else set())

    def logits(self, f: Path) -> list[float]:
        """Row 0 of a dumped bf16 logits buffer, as f32: the high half of the bits."""
        import struct

        raw = f.read_bytes()[: self.vocab * 2]
        return [struct.unpack("<f", b"\0\0" + raw[i : i + 2])[0] for i in range(0, len(raw), 2)]

    def run(self, prompt: str | list[int], steps: int, rows: int | None = None) -> list[int] | None:
        if self.target.ranks != 1 or self.gpu is None:
            return None
        key = json.dumps([prompt, steps, rows])
        if key not in self.cache:
            cmd = self.command(prompt) + ["--steps", str(steps)]
            if rows is not None:
                cmd += ["--rows", str(rows)]
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            m = re.search(r"generated ids: \[([^\]]*)\]", p.stderr)
            if p.returncode != 0 or not m:
                raise RuntimeError(f"kern run failed ({p.returncode}): {p.stderr[-600:]}")
            self.cache[key] = [int(x) for x in m.group(1).split(",") if x.strip()]
            self.path.write_text(json.dumps(self.cache))
        return self.cache[key]


def kern_test(a, t: Target, gpu: int, rep: Report, out: Path) -> None:
    if not t.reference:
        return
    cmd = [str(a.kern), "--config", str(t.config), "test", t.name, "--gpu", str(gpu)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    (out / "kern-test.log").write_text(p.stdout + p.stderr)
    verdict = {0: "PASS", 1: "FAIL", 2: "INCONCLUSIVE"}.get(p.returncode, f"exit {p.returncode}")
    last = next((l for l in reversed((p.stdout + p.stderr).splitlines()) if l.strip()), "")
    rep.add("kern_test", p.returncode == 0, f"{verdict}: {last[:160]}")


def session_default(a, t: Target, gpus: list[int], port: int, rep: Report, oracle: Oracle, out: Path) -> dict:
    """conc1, repeat, turn2 (warm half), concurrent, abort. Returns what later sessions compare against."""
    cold: dict[str, list[int]] = {}
    prompt_ids: dict[str, list[int]] = {}
    stop: dict[str, int | None] = {}
    turn2: dict[str, dict] = {}
    prompt_max = 0
    with Server(a, t, gpus, port, {}, out / "server-default.log") as s:
        rep.facts = s.facts
        page, rows = s.facts["page"], s.facts["rows"]
        t.stateful = s.facts.get("checkpoints") == "at request end"
        print(f"  facts: {', '.join(f'{k}={v}' for k, v in s.facts.items() if k in ('ranks', 'tray', 'pages', 'page', 'rows', 'checkpoints', 'seq_slots', 'load_s'))}")
        # conc1: one request at a time, against `kern run`.
        same = Same(oracle, rows)
        for i, p in enumerate(PROMPTS):
            r = complete(s.url, p, a.max_tokens)
            cold[p], prompt_ids[p], stop[p] = r["ids"], r["prompt_ids"], r["stop"]
            prompt_max = max(prompt_max, len(r["prompt_ids"]))
            want = oracle.run(p, a.max_tokens)
            if want is not None:
                same.add(f"prompt {i}", r["prompt_ids"], r["ids"], want)
        if oracle.available:
            rep.add("conc1_equals_run", same.ok, str(same))
        else:
            rep.add("conc1_cold", None, f"{len(PROMPTS)} cold answers recorded (no `kern run` oracle: {t.ranks} ranks or no spare GPU)")
        # repeat: the hit the checkpoints allow. Every page: the prompt's whole
        # pages short of its last token; at request end: a stateful checkpoint
        # is prompt + answer, longer than the prompt, so no hit.
        hits, same = [], Same(oracle, rows)
        for i, p in enumerate(PROMPTS[:4]):
            r = complete(s.url, p, a.max_tokens)
            n = len(r["prompt_ids"])
            want = (n - 1) // page * page if not t.stateful else 0
            hits.append((r["cached"], want))
            same.add(f"prompt {i}", r["prompt_ids"], r["ids"], cold[p])
        rep.add("repeat_hit", all(c == w for c, w in hits) and same.ok, f"cached (got, want) {hits}; vs the cold answer: {same}")
        # turn2, warm half: prompt + its answer (the stop token it ended
        # with, as a chat template's end of turn) + a question, right after.
        # As ids: the text of an answer need not tokenize back to its ids.
        suffix = tokenize(s.url, SUFFIX)
        cached, same = [], Same(oracle, rows)
        for i, p in enumerate(PROMPTS[:4]):
            t2 = prompt_ids[p] + cold[p] + ([stop[p]] if stop[p] is not None else []) + suffix
            r = complete(s.url, t2, a.max_tokens)
            n1 = len(prompt_ids[p])
            turn2[p] = {"ids": t2, "warm_ids": r["ids"], "cached": r["cached"], "prompt_len": n1}
            cached.append((r["cached"], n1))
            want = oracle.run(t2, a.max_tokens)
            if want is not None:
                same.add(f"turn2 {i}", t2, r["ids"], want)
        # A hit covers at least the first prompt's tokens, its last one aside,
        # in whole pages for a paged-only manifest. A speculative round can
        # take a stateful sequence past what the client got: the server
        # says so ("not kept") and no hit is right, for those.
        floor = lambda n: (n - 1) // page * page if not t.stateful else n
        unkept = len(s.lines("not kept"))
        misses = sum(1 for c, _ in cached if c == 0)
        hit_ok = all(c >= floor(n) or c == 0 for c, n in cached) and misses <= unkept and misses < len(cached)
        rep.add("turn2_hit", hit_ok, f"cached (got, first prompt) {cached}; {unkept} states past the answer not kept")
        if same.n:
            rep.add("turn2_warm_equals_run", same.ok, str(same))
        # concurrent: everything at once.
        results: dict[str, dict] = {}

        def one(p):
            results[p] = complete(s.url, p, a.max_tokens)

        windows = len(s.lines("stats"))
        threads = [threading.Thread(target=one, args=(p,)) for p in PROMPTS]
        t0 = time.monotonic()
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        dt = time.monotonic() - t0
        finished = sum(1 for p in PROMPTS if p in results and results[p]["finish"] in ("length", "stop"))
        same = sum(1 for p in PROMPTS if p in results and results[p]["ids"] == cold[p])
        rep.add("concurrent", finished == len(PROMPTS), f"{finished}/{len(PROMPTS)} finished in {dt:.1f}s, {same} identical to conc1 (batch composition is not a gate)")
        s.flush_stats()
        c = s.counters()
        if rows > 1:
            # The 5 s window that held the burst; a later one may hold only
            # the flush request.
            burst = max(s.lines("stats")[windows:], key=lambda w: w.get("steps", 0), default={})
            pct = burst.get("accept_pct")
            rep.add("spec_acceptance", pct is not None and pct >= 20, f"rows={rows} accept_pct={pct} accepted={burst.get('accepted')} (the burst's window)")
        # abort: hang up mid-stream, then a plain request.
        got = stream_then_hang_up(s.url, PROMPTS[5], 3)
        time.sleep(1)
        r = complete(s.url, PROMPTS[0], a.max_tokens)
        same = Same(oracle, rows)
        same.add("after the abort", r["prompt_ids"], r["ids"], cold[PROMPTS[0]])
        rep.add("abort", got == 3 and same.ok, f"{got} chunks read before hanging up; next answer vs cold: {same}")
        rep.counters["default"] = c
    return {"cold": cold, "turn2": turn2, "page": page, "rows": rows, "prompt_max": prompt_max + a.max_tokens + len(suffix) + 1}


def session_host(a, t: Target, gpus: list[int], port: int, rep: Report, oracle: Oracle, out: Path, prev: dict) -> None:
    """A pool of a few pages and a host tier: the cold half of turn2, then parks and wakes.
    Answers that took another numerical path (warm after cold, a hit on a parked
    state) are held to identity only where the oracle can excuse a near-tie."""
    cold, turn2, page, rows = prev["cold"], prev["turn2"], prev["page"], prev["rows"]
    # Per rank, room for its share of four of the longest requests' worst
    # cases and a page over; a stateful manifest's first slots come on top
    # of that (pages and slots trade chunks), so the fill goes on until the
    # server parks.
    per = -(-4 // t.ranks)
    capacity = (per * -(-(prev["prompt_max"] + a.max_tokens + 16) // page) + 1) * page
    flags = {"--capacity": str(capacity), "--host-gib": "2", "--max-seqs": "4"}
    gate = lambda same: same.ok if oracle.available else None
    with Server(a, t, gpus, port, flags, out / "server-host.log") as s:
        # Cold turn2 on a fresh server: warm == cold.
        same = Same(oracle, rows)
        for i, p in enumerate(PROMPTS[:4]):
            r = complete(s.url, turn2[p]["ids"], a.max_tokens)
            turn2[p]["cold_ids"] = r["ids"]
            same.add(f"turn2 {i}", turn2[p]["ids"], turn2[p]["warm_ids"], r["ids"])
        rep.add("turn2_warm_equals_cold", gate(same), f"cached_tokens warm {[turn2[p]['cached'] for p in PROMPTS[:4]]}; {same}")
        # Finished requests fill the pool until their checkpoints park: the
        # twelve prompts (their answers vs cold), then numbered variants.
        fill, n = Same(oracle, rows), 0
        while n < len(PROMPTS) or (n < 48 * t.ranks and len(s.lines("parked")) < 4):
            p = PROMPTS[n % len(PROMPTS)]
            r = complete(s.url, p if n < len(PROMPTS) else f"{n}. {p}", a.max_tokens)
            if n < len(PROMPTS):
                fill.add(f"prompt {n}", r["prompt_ids"], r["ids"], cold[p])
            n += 1
        # The turn2 prompts hit the parked checkpoints of their first turns.
        same = Same(oracle, rows)
        for i, p in enumerate(PROMPTS[:4]):
            r = complete(s.url, turn2[p]["ids"], a.max_tokens)
            same.add(f"turn2 {i}", turn2[p]["ids"], r["ids"], turn2[p]["cold_ids"])
        s.flush_stats()
        c = s.counters()
        woke = sum(1 for l in s.lines("admitted") if l.get("woken") == "true")
        rep.counters["host"] = c
        identity = gate(fill) is not False and gate(same) is not False
        rep.add(
            "park_wake",
            c["parks"] >= 1 and c["wakes"] >= 1 and c["host_hits"] >= 1 and identity,
            f"capacity={capacity} fills={n} parks={c['parks']} host_evictions={c['host_evictions']} evictions={c['evictions']} wakes={c['wakes']} host_hits={c['host_hits']} woken requests={woke}; under the small pool vs cold: {fill}; turn2 after the parks vs cold: {same}",
        )


def session_slots(a, t: Target, gpus: list[int], port: int, rep: Report, oracle: Oracle, out: Path, prev: dict) -> None:
    """A stateful manifest starts with a few slots and grows them out of free
    pages: finished requests beyond the first slots leave their checkpoints
    resident (the pool is the default, nothing parks), the answers are the
    cold ones and a turn2 hits after the growth as it did before it."""
    if not t.stateful:
        return
    cold, turn2, rows = prev["cold"], prev["turn2"], prev["rows"]
    with Server(a, t, gpus, port, {"--max-seqs": "2"}, out / "server-slots.log") as s:
        # `seq_slots` is the tray's sum: twice as many finished requests
        # outgrow every rank's share however they are placed.
        first = s.facts.get("seq_slots", 0)
        same = Same(oracle, rows)
        for n in range(max(2 * first + 4, len(PROMPTS))):
            p = PROMPTS[n % len(PROMPTS)]
            r = complete(s.url, p if n < len(PROMPTS) else f"{n}. {p}", a.max_tokens)
            if n < len(PROMPTS):
                same.add(f"prompt {n}", r["prompt_ids"], r["ids"], cold[p])
        hits, warm = [], Same(oracle, rows)
        for i, p in enumerate(PROMPTS[:4]):
            r = complete(s.url, turn2[p]["ids"], a.max_tokens)
            hits.append((r["cached"], turn2[p]["cached"]))
            warm.add(f"turn2 {i}", turn2[p]["ids"], r["ids"], turn2[p]["warm_ids"])
        s.flush_stats()
        c = s.counters()
        rep.counters["slots"] = c
        grown = c.get("remaps", 0) >= 1 and c.get("slots", 0) > first
        rep.add(
            "slot_growth",
            grown and same.ok and warm.ok and all(g == w for g, w in hits),
            f"remaps={c.get('remaps')} slots={c.get('slots')} ({first} to start) slots_used={c.get('slots_used')} after {max(2 * first + 4, len(PROMPTS))} requests; the {len(PROMPTS)} prompts vs cold: {same}; turn2 hits (got, before) {hits}, vs the warm answers: {warm}",
        )


def session_rows1(a, t: Target, gpus: list[int], port: int, rep: Report, oracle: Oracle, out: Path, prev: dict) -> None:
    if prev["rows"] <= 1:
        return
    with Server(a, t, gpus, port, {"--rows": "1"}, out / "server-rows1.log") as s:
        same, same_spec = Same(oracle, 1), 0
        for i, p in enumerate(PROMPTS[:4]):
            r = complete(s.url, p, a.max_tokens)
            same_spec += r["ids"] == prev["cold"][p]
            want = oracle.run(p, a.max_tokens, rows=1)
            if want is not None:
                same.add(f"prompt {i}", r["prompt_ids"], r["ids"], want)
        if same.n:
            rep.add("rows1_equals_run", same.ok, str(same))
        rep.add("rows1_vs_spec", None, f"{same_spec}/4 identical to the {prev['rows']}-row answers (a near-tie may flip under a draft)")


def run_target(a, t: Target, gpus: list[int], out: Path) -> Report:
    rep = Report(target=t.name)
    print(f"== {t.name} ({t.ranks} rank{'s' if t.ranks > 1 else ''}, {'stateful' if t.stateful else 'paged'})", flush=True)
    missing = t.missing()
    if missing:
        rep.skipped = "missing: " + ", ".join(missing)
        print(f"  skipped: {rep.skipped}")
        return rep
    if t.ranks > len(gpus):
        rep.skipped = f"needs {t.ranks} GPUs, {len(gpus)} given"
        print(f"  skipped: {rep.skipped}")
        return rep
    out.mkdir(parents=True, exist_ok=True)
    use = gpus[: t.ranks]
    spare = gpus[t.ranks] if len(gpus) > t.ranks else None
    oracle = Oracle(a, t, spare, out / "oracle.json")
    try:
        kern_test(a, t, use[0], rep, out)
        prev = session_default(a, t, use, a.port, rep, oracle, out)
        session_host(a, t, use, a.port + 1, rep, oracle, out, prev)
        session_slots(a, t, use, a.port + 2, rep, oracle, out, prev)
        session_rows1(a, t, use, a.port + 3, rep, oracle, out, prev)
    except Exception as e:  # noqa: BLE001
        rep.add("error", False, f"{type(e).__name__}: {e}")
    return rep


def main() -> int:
    global MODEL
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("kern.toml"))
    ap.add_argument("--targets", nargs="*", default=None, help="targets to run (default: every one in the config)")
    ap.add_argument("--gpus", default="0", help="device ordinals a target may use, in rank order")
    ap.add_argument("--out", type=Path, required=True, help="results directory (one subdir per target)")
    ap.add_argument("--kern", type=Path, default=Path("target/host/release/kern"))
    ap.add_argument("--kern-serve", type=Path, default=Path("target/release/kern-serve"))
    ap.add_argument("--port", type=int, default=18060)
    ap.add_argument("--load-timeout", type=float, default=2400)
    ap.add_argument("--max-tokens", type=int, default=64, help="tokens generated per request")
    ap.add_argument("--server-flags", default="", help="kern-serve flags for every session, e.g. '--chunk 128 --max-seqs 16'")
    a = ap.parse_args()
    words = a.server_flags.split()
    a.server_flags = dict(zip(words[::2], words[1::2]))
    a.kern, a.kern_serve = a.kern.resolve(), a.kern_serve.resolve()
    gpus = [int(g) for g in a.gpus.split(",")]
    targets = load_targets(a.config.resolve())
    if a.targets:
        targets = [t for t in targets if t.name in a.targets]
    reports = []
    for t in targets:
        MODEL = json.loads(t.manifest.read_text())["model"] if t.manifest.exists() else t.name
        rep = run_target(a, t, gpus, a.out / t.name)
        reports.append(rep)
        (a.out / t.name / "report.json").parent.mkdir(parents=True, exist_ok=True)
        (a.out / t.name / "report.json").write_text(
            json.dumps({"target": rep.target, "skipped": rep.skipped, "facts": rep.facts, "counters": rep.counters, "checks": [c.__dict__ for c in rep.checks]}, indent=1)
        )
    lines = ["| target | checks | result |", "|---|---|---|"]
    failed = 0
    for r in reports:
        if r.skipped:
            lines.append(f"| {r.target} | — | skipped: {r.skipped} |")
            continue
        gated = [c for c in r.checks if c.ok is not None]
        bad = [c.name for c in gated if not c.ok]
        failed += bool(bad)
        lines.append(f"| {r.target} | {len(gated)} gated, {len(r.checks) - len(gated)} reported | {'FAIL: ' + ', '.join(bad) if bad else 'all pass'} |")
    summary = "\n".join(lines)
    print("\n" + summary)
    (a.out / "summary.md").write_text(summary + "\n")
    return 1 if failed else 0


MODEL = ""

if __name__ == "__main__":
    sys.exit(main())
