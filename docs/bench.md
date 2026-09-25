# kern bench: where a program's time goes

`kern bench` measures a sweep of call shapes on one device and writes the
raw samples out. It answers two questions, and they cost very different
amounts:

- **Where does this program's time go?** Every call is bracketed in a
  second copy of the graph and attributed. Always done, seconds per
  scenario.
- **What would this call cost on its own, cold and warm?** Every distinct
  call is lifted out of the program and measured against its own
  pre-image. Only for `--isolate`, and it is most of the wall clock.

The first tells you what to work on. The second tells you what the work
would be worth. Neither starts a server, constructs traffic, or needs an
external profiler.

```bash
kern bench qwen3-4b --workload tools/profiles/quick.toml --out results/quick.json
kern bench qwen3-4b --workload tools/profiles/atlas.toml --out results/atlas.json --isolate
kern bench dsv41 --gpu 0 --workload decode.toml --out results/decode.json   # EP4: gpus 0–3
```

The target names the manifest, kernels, weights and tokenizer in
`kern.toml`, exactly as `kern run` and `kern test` do; `--manifest`,
`--kernels`, `--weights`, `--tokenizer` and `--gpu` override any of them,
which is how you A/B two manifests against one checkpoint. The workload
and the output path are never implied: a command that writes a file says
which file.

## The workload is a sweep of shapes, not a list of programs

A workload names **shapes**. `groups` sequences of `rows` rows each, on
top of a previous `context`. `Protocol` turns a shape into a program, so
a decode step, a prefill chunk and a speculative round are the same kind
of entry and the file never contains a program name.

```toml
samples = 32                  # per measurement, 12..=256; every sample is kept
seed = 24301                  # the prose the context is built from

[[sweep]]
groups = [1, 2, 4, 8, 16]
rows = [1]
context = [128, 512, 1024, 2048]

[[sweep]]                     # one batch whose sequences sit at different depths
groups = [4]
rows = [1]
context = [[128, 512, 1024, 2048]]

[[sweep]]                     # prefill chunks over an empty cache
rows = [1, 16, 64, 128, 256, 512, 1024, 2048]
context = [0]
```

Each `[[sweep]]` is a cross product; the sweeps are unioned and repeated
shapes collapse. `groups` defaults to `[1]`. A `context` entry may be a
list instead of a number, meaning one length per sequence; it pairs
only with the group count it has lengths for, which must be one of the
sweep's `groups`.

A sweep may also say `weight = <calls>`: how many calls of each of its
shapes the traffic being modeled makes (default 1; a shape repeated across
sweeps sums its weights). `python -m kern_vllm.workload <trace>` writes one
such sweep per bucket of a vLLM step trace. After the last scenario the
report prices the workload: every shape once, at the fastest program that
takes it, times its weight, and that time by op:

```
cost      <seconds> · <calls> calls[ · <n> dropped] · <op> <share> · … · +<n> more
```

It is the GPU time the modeled traffic costs on this manifest, the number
an optimization is judged by. Shapes the manifest drops are counted
beside it, not priced.

A cross product is the point. On Qwen3-4B the same 2,048-row chunk is
31% matrix multiply and 47% attention over an empty cache, and 2% matrix
multiply and 96% attention over 32k of it (the `mix` line, GB300,
2026-09-16). A profile taken at one shape does not transfer to another,
so the tool takes them all.

**Shapes with no program are dropped, not fatal.** Which shapes a
manifest serves is part of what a sweep asks, so an answer of "not this
one" is an answer:

```
plan      4 scenarios · 3 dropped · 2064 tokens of state · 4 seqs
drop      g4-r2048-kv0 · 8192 rows exceeds the manifest's 2048
drop      g512-r1-kv0 · no program takes 512 sequences × 1 rows
```

**One shape can become more than one scenario.** A manifest with both a
one-row step program and a rows-as-fed chunk takes a single row either
way, and which is faster is a fair question, so both run. That is why the
program is part of a scenario's id and not something the file chooses:
`decode-g1-r1-kv128` and `prefill-g1-r1-kv128` are two scenarios of one
shape.

## What the terminal says

One line per fact, in the house style: the section name first, ` · `
between facts, the section's time last.

```
calibrate NVIDIA GB300 · 152 SMs · L2 129 MiB   170.6ms
plan      6 scenarios · 0 dropped · 2128 tokens of state · 4 seqs
scenario  decode-g1-r1-kv128 · prefix 1 chunk
program   2.577 ms · p10–p90 2.575–2.580 · cv 0.1% · 436 calls   150.3ms
mix       gemm 49.9% · fused_add_rms_norm 11.3% · attn 8.9% · rotary_embedding 6.7% · silu_mul 6.0% · rms_norm_qhead 5.7% · +5 more
isolate   85 cases · outputs match   868.7ms
…
scenario  decode_batch-g4-r1-kv512 · prefix 4 chunks
program   3.324 ms · p10–p90 3.318–3.327 · cv 0.1% · 436 calls   152.7ms
mix       gemm 44.2% · attn_batch 19.6% · fused_add_rms_norm 9.9% · rotary_embedding 5.8% · silu_mul 5.5% · rms_norm_qhead 5.0% · +5 more
isolate   85 cases · outputs match   891.3ms
out       results/quick.json   6.9s
```

`mix` is the answer to "where does the time go", and it is on the
terminal because that is where the question is asked. The shares are of
the traced total, not of the graph: bracketing every call costs time that
belongs to no call, and dividing by the graph would quietly hand that
time to whichever op happened to be measured. A `program` line's time
covers the scenario so far, prefix included; the `out` line's is the
whole run.

## What is actually measured

- **Every input is real.** A sequence's prefix is built by running the
  chunk program over deterministic prose, chunk by chunk, into its own
  lease. The page table, slot mapping, sequence lengths and KV pages are
  what a server would hand the kernel. A context length is never
  simulated by raising a scalar over uninitialised cache. `span_at` is
  the one fill written at a constant.
- **An isolated call is measured where it occurs.** The walk executes
  each call for real after measuring it, so the next call sees the inputs
  the program would give it. Calls are grouped by op, resolved arguments,
  buffer shape, aliasing, offsets and vars; weight names are ignored,
  state offsets are not. Qwen3-4B's decode is 436 calls in 85 groups.
- **Cold and warm.** Cold streams over a seeded buffer at least eight
  times the reported L2 first; warm primes the op and restores its
  declared writes again. The order alternates between samples. This is an
  empirical cache-sensitivity check, not a claim to flush every cache.
- **Every sample starts from the program's pre-image**: the carries and
  states its calls declare they write are restored before each replay.
  Workspace is written before it is read within a run, so it is left
  alone, and an exported carry (one that peers write into) is never
  restored: its barrier epochs must only advance.
- **Nothing is filtered.** Every sample is in the report, slow tails
  included, with p10/p50/p90, cv, tail ratio and per-quarter medians.
  Hardware anchors run before and after the sweep so a machine that
  drifted mid-run says so.

## Ranks

A manifest with a `topology` runs as its ranks, on consecutive GPUs from
`--gpu`. Every rank is fed the same shape with its own sequences and its
own prose, and the ranks replay in lockstep, so a call that reads its
peers finds them issuing. Attribution is per rank: the report has one
record per scenario per rank, with a `rank` field, and the terminal shows
the rank a step waits for.

```
calibrate NVIDIA GB300 · 152 SMs · L2 129 MiB · rank 0 of 4   171.2ms
plan      9 scenarios · 0 dropped · 33920 tokens of state · 8 seqs
scenario  decode_batch-g8-r1-kv4096 · prefix 16 chunks
program   9.070 ms · p10–p90 9.062–9.095 · cv 0.2% · 672 calls · rank 2 slowest of 4   3.1s
mix       dsv41_mega_moe_e384.0799aa8f 33.6% · dsv41_dense.d9228a3b 5.7% · …
```

`hardware` and the calibration anchors are rank 0's. `--isolate` is
refused for a topology: a call that reads its peers cannot be lifted out
of one rank's program and replayed alone.

## The gate

With `--isolate`, the tokens the call-by-call walk emits must equal the
tokens the whole program emits, or the scenario fails. That is what
`outputs match` on the `isolate` line means, and it is the check that
snapshot and restore actually put the state back.

Without `--isolate` there is no walk and no such check, which is the
honest cost of the cheap mode.

## Limits

- Attribution only across ranks. Exposed communication, a rank waiting
  on a peer inside a kernel, is inside that call's bracket and only a
  trace tells it from compute.
- One plan, one allocation. Capacity is sized for the largest scenario in
  the workload, so a sweep's cheap scenarios carry its expensive one's
  state. Split a sweep in two if that matters.
- `--isolate` reports are the only input the operator explorer
  (`tools/profile_export.py`) accepts: an attribution-only report has no
  operator evidence in it.
