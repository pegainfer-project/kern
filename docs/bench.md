# kern bench: where a program's time goes

> Qwen3-4B commands and measurements below are historical. The old fixtures
> and capture scripts are available at [the last core revision](https://github.com/pegainfer-project/kern/tree/8b25c07).
> Current dense Qwen3 development lives in [kern-qwen3](https://github.com/pegainfer-project/kern-qwen3);
> the legacy DSpark path is not part of that repository.


`kern bench` measures a sweep of call shapes on one device and writes the
raw samples out. Every sample is one replay of the program's real CUDA
graph, from its pre-image behind an L2 flush, and CUPTI's kernel records
(`kern_runtime::trace`) say when each kernel ran and which call launched
it. The graph's time and every call's come from the same replay, with
nothing inserted between kernels. It starts no server, constructs no
traffic and needs no external profiler; it cannot run under nsys, whose
CUPTI subscription excludes kern's.

```bash
kern bench qwen3-4b --workload tools/profiles/quick.toml --out results/quick.json
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
of entry and the file never contains a program name. What a sweep does
say is which of the protocol's two kinds of call its rows are: under
`rows` they are a step's, laid out for exactly that many a sequence (1 for
a decode step, 16 for a sixteen-row speculative round); under `chunk`
they are fed as they come (a prefill chunk).

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
chunk = [1, 16, 64, 128, 256, 512, 1024, 2048]
context = [0]
```

Each `[[sweep]]` is a cross product; the sweeps are unioned and repeated
shapes collapse. A sweep gives `rows` or `chunk`, never both. `groups`
defaults to `[1]`. A `context` entry may be a list instead of a number,
meaning one length per sequence; it pairs only with the group count it
has lengths for, which must be one of the sweep's `groups`.

A sweep may also say `weight = <calls>`: how many calls of each of its
shapes the traffic being modeled makes (default 1; a shape repeated across
sweeps sums its weights). `python -m kern_vllm.workload <trace>` writes one
such sweep per bucket of a vLLM step trace. After the last scenario the
report prices the workload: every shape's program time, times its weight,
and that time by op:

```
cost      <seconds> · <calls> calls[ · <n> dropped] · <op> <share> · … · +<n> more
```

It is the GPU time the modeled traffic costs on this manifest, the number
an optimization is judged by. Shapes the manifest drops are counted
beside it, not priced.

A cross product is the point. On Qwen3-4B a 2,048-row chunk over an
empty cache is 31% matrix multiply and 51% attention, and 512 rows over
32k of it 2% matrix multiply and 97% attention (the `mix` line, GB300,
2026-11-08). A profile taken at one shape does not transfer to another,
so the tool takes them all.

**Shapes with no program are dropped, not fatal.** Which shapes a
manifest serves is part of what a sweep asks, so an answer of "not this
one" is an answer:

```
plan      4 scenarios · 3 dropped · 2064 tokens of state · 4 seqs
drop      g4-r2048-kv0 · 8192 rows exceeds the manifest's 2048
drop      g512-r1-kv0 · no program takes 512 sequences × 1 rows
```

**A step and a chunk of the same rows are two shapes.** A sixteen-row
speculative round drafts, verifies and samples; a sixteen-row chunk only
advances state. Which of them a call was is part of its shape, so each
has one program and its own id: `round-g1-r16-kv128` and
`prefill-g1-c16-kv128`. To ask whether a one-row step or a one-row chunk
is faster, sweep both.

## What the terminal says

One line per fact, in the house style: the section name first, ` · `
between facts, the section's time last.

```
source    512 tokens of real prefix per rank   55.5ms
device    NVIDIA GB300 · 152 SMs · L2 129 MiB · HES
plan      6 scenarios · 0 dropped · 2640 tokens of state · 5 seqs
scenario  decode-g1-r1-kv128
program   2.657 ms · p10–p90 2.656–2.659 · cv 0.1% · 436 calls   905.5ms
mix       gemm 65.6% · attn 8.7% · fused_add_rms_norm 6.5% · silu_mul 4.1% · rotary_embedding 3.9% · sample 3.8% · +5 more
…
cost      0.011 s · 4 calls · gemm 60.2% · attn_batch 10.9% · fused_add_rms_norm 5.9% · rotary_embedding 3.9% · silu_mul 3.7% · attn 2.9% · +7 more
out       results/quick.json   5.6s
```

`device` says whether the kernel timestamps are the GPU's own (HES, on
Blackwell) or CUPTI's software ones. A `program` line's time is the p50
from the first kernel's start to the last kernel's end; it covers the
scenario so far, context copies included, and the `out` line's is the
whole run. `mix` is the answer to "where does the time go": each call's
kernel time, summed by op, as a share of all kernel time. Time between
kernels belongs to no call, and kernels that overlap (programmatic
dependent launch, lanes) are each charged in full.

`mix` is also what removing an op would buy. Before kern bench read
kernel records it bracketed every call with CUDA events, which cost a few
microseconds each and broke programmatic dependent launch, and it was
checked against the program captured again with the op's calls left out:
on the Qwen3-4B decode above the events gave gemm 49.7% and the fused
add-norm 11.6%, against 67.5% and 6.6% with the op gone (tray03,
2026-11-08). Read off the kernels, `mix` agreed with that to a point or
two, so the second capture was removed.

## What is actually measured

- **Every input is real.** Each rank builds one prefix, as long as the
  plan's longest context, by running the chunk program over deterministic
  prose. A scenario's sequence copies its first n positions from it into
  its own lease (`Runtime::replicate`): its own pages and state slot,
  nothing shared, so a batch reads as much memory as distinct sequences
  would, and a 185k context costs a copy, not its prefill. The page table,
  slot mapping, sequence lengths and KV pages are what a server would hand
  the kernel. A context length is never simulated by raising a scalar over
  uninitialised cache. `span_at` is the one fill written at a constant.
- **Every sample starts cold, from the program's pre-image.** A
  copy-engine copy of eight times the L2 flushes it (as fully as an SM
  copy: a 32 MiB read after either takes 19.1 µs against 16.3 µs warm,
  GB300, `bench_results/2026-11-08-l2-evict`), and the carries and states
  the program declares it writes are restored. Workspace is written
  before it is read within a run, so it is left alone; an exported carry
  (one that peers write into) is never restored, since its barrier epochs
  must only advance; a paged state is only appended to, so a replay
  rewrites its rows with the same bytes.
- **Nothing is filtered.** Every sample is in the report, slow tails
  included, with p10/p50/p90, cv, tail ratio and per-quarter medians,
  beside `nvidia-smi`'s clocks, power and temperature before and after
  each scenario, so a machine that throttled mid-run says so.

## Ranks

A manifest with a `topology` runs as its ranks, on consecutive GPUs from
`--gpu`. Every rank is fed the same shape with its own sequences and its
own prose, and the ranks replay in lockstep, so a call that reads its
peers finds them issuing. Attribution is per rank: the report has one
record per scenario per rank, with a `rank` field, and the terminal shows
the rank a step waits for.

```
device    NVIDIA GB300 · 152 SMs · L2 129 MiB · HES · rank 0 of 4
plan      9 scenarios · 0 dropped · 33920 tokens of state · 8 seqs
scenario  decode_batch-g8-r1-kv4096
program   9.070 ms · p10–p90 9.062–9.095 · cv 0.2% · 672 calls · rank 2 slowest of 4   3.1s
```

`hardware` is rank 0's.

## The gate

After the samples, one more replay from the restored pre-image must emit
the tokens the last sample did, or the scenario fails: it is the check
that restoring actually puts the state back. The tracer must return every
kernel node of every sampled graph, each named by its call, or the run
fails naming the program and how many records CUPTI dropped.

## Limits

- Exposed communication, a rank waiting on a peer inside a kernel, is
  inside that kernel's time; only the peers' records tell it from
  compute.
- One plan, one allocation. Capacity is sized for the largest scenario in
  the workload, so a sweep's cheap scenarios carry its expensive one's
  state. Split a sweep in two if that matters.
