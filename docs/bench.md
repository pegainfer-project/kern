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
sweeps sums its weights). `kern workload` writes one such sweep per bucket
of a kern-serve trace ([below](#a-workload-from-a-served-trace)), and
`python -m kern_vllm.workload <trace>` per bucket of a vLLM step trace.
After the last scenario the report prices the workload: every shape's
program time, times its weight, and that time by op:

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

## A workload from a served trace

`kern-serve --trace-dir` records every step it issues
([serve.md](serve.md), "Tracing"). `kern workload` reads those steps back
into the calls they were and writes the workload that stands for them:

```bash
kern workload qwen3.8-27b --trace traces/date=2026-11-10 --out served.toml
kern workload --manifest served.json --trace steps.parquet --out served.toml
```

`--trace` takes one rank's steps files or directories of them (bench
issues a call on every rank of the manifest, so a group's directory is
given as `…/rank=0`); the manifest is the target's, or the one
`--manifest` names in its place, and must be the one the trace was served
on (the files' `kern.manifest_sha256` is checked when they carry one, and
a step of a program the manifest does not have is refused either way). The
manifest is what reads a step back: its program's `batch` says whether the
call was a step or a chunk, the vars over the groups and rows axes say how
many sequences and rows, and the step's summed sequence lengths less its
own rows are the context. The trace keeps only the sums, so the context is
taken as equal across the call's sequences, and so are a chunk's rows, to
the nearest row. A manifest that deals a sequence's positions over a rank
group is refused: a member's trace holds the lengths of its own share.

On 3.3 hours of a public Qwen3.8-27B service (2026-11-10; one sm120 GPU,
fp8 weights, speculative rounds over up to 16 sequences, prefill in
1024-row chunks; the snapshot is one merged steps file, which carries no
sha256):

```
trace     85284 steps · 1 file · 8621.7 s of step time · 1 from programs no caller drives · no manifest sha256 in the files to check
fidelity  +0.48% priced at the trace's own nearest steps · worst prefill seqs<=1,tokens<=1024 +0.85%
plan      141 scenarios · 0 dropped · 718272 tokens of state · 17 seqs
out       served.toml
```

- **A bucket is a metric label set.** Steps are grouped as
  `kern_step_compute_seconds_total` labels them: program, shape, context
  rounded up to a power of two. A scenario is then one series, and what
  bench measures of it can be held against what the server reports for
  the same labels.
- **A bucket's sweep keeps its totals.** It is the call in the middle of
  the bucket's steps (so a bucket that is mostly full chunks is a full
  chunk, a shape that was served, and not an average row count that never
  was), at the bucket's mean context, weighted by their count. The mean
  keeps the bucket's total context, and a cost linear in context is
  priced by the total.
- **`fidelity` is the trace pricing itself.** Each bucket is priced at
  the median of the traced steps nearest its sweep (the 25 nearest in
  context, and every step as near as the last of them), and the sum is
  held against the time the bucket's steps took. It is the error of
  standing a bucket for its steps, known before a GPU is involved;
  `worst` is the program and shape furthest off in seconds.
- **`plan`** is what `kern bench` will make of the file on this manifest;
  a `drop` line names a sweep it will not take.

That is all it does: every bucket becomes a sweep, most expensive first,
so the file is the whole trace and its `cost` is held against the trace's
own step time. Each sweep carries a comment saying what it stands for:

```toml
# prefill · seqs<=1,tokens<=1024 · kv<=64k · 3314 steps · 1274.0 s
[[sweep]]
groups = [1]
chunk = [1024]
context = [46502]
weight = 3314
```

A smaller workload for a faster loop is an edit to the file, not a flag:
drop the sweeps of the tail, or merge a shape's context buckets into one
sweep (weights summed, context their weighted mean). The same goes for
`samples`, `seed`, and a sweep the plan dropped.

## A workload from the manifest alone

Before there is a trace, the manifest still says which shapes can be made
at all: its protocol bounds a chunk's rows and a step's sequences, its
page tables bound a context. Without `--trace`, `kern workload` writes the
sweep that covers those bounds, for whoever needs a call's time as a
function of its shape (a capacity estimate, a simulated server) and has no
traffic to read:

```bash
kern workload qwen3.8-27b --chunk 4096 --groups 1,2,4,8,16,24,32,48,64,96,128 --state-gib 128 --out envelope.toml
```

```
envelope  chunks of up to 4096 rows · steps of 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128 sequences · 2097152 tokens of state in 128 GiB · a sequence holds 262144
plan      194 scenarios · 0 dropped · 2363392 tokens of state · 129 seqs
out       envelope.toml
```

The three flags are what no manifest says. `--chunk` is the largest chunk
to price, the server's prefill chunk. `--groups` are the batch sizes the
server pads its steps to. `--state-gib` is the paged state a scenario may
hold on a rank: bench sizes one allocation for its largest scenario, and
with the page-table row it decides the deepest context a sequence is
measured at. A size or a chunk the manifest has no program for is a `drop`
line, as in any workload.

What it sweeps:

- **Every step size over two contexts**: one page, and the deepest the
  row and its share of the state leave it.
- **Chunks over an empty sequence, every 32 rows** up to `--chunk` (powers
  of two below 32, and `--chunk` itself whatever the spacing).
- **Chunks over the deepest context, every 128 rows**: what a context adds.

A call's price is then two numbers a size, its time over the shallowest
context and the slope to the deepest. A step of `g` sequences is its
`g`'s time plus the slope for every token its sequences sit past a page;
a chunk of `c` rows over `x` tokens is the empty-sequence time at `c`
(linear between the measured rows) plus `x` times the slope of the next
measured chunk at or above `c`.

Why this sweep and not a denser, a sparser or an adaptive one was
measured on Qwen3.8-27B (bf16, one GB300, 2026-11-18,
`bench_results/2026-11-18-kern-price-collect`, on the manifest that still
carried a one-sequence `decode` beside `decode_batch`); every error below
is predicted over measured on shapes the fit never saw:

- **Two contexts are enough.** A context adds to a call linearly. With
  three more contexts between the ends the errors did not move: chunks
  p90 6.59% with 28 scenarios and with 70, steps at most 1.93% with 22 and
  2.17% with 127.
- **A formula in rows and context does not price a chunk.**
  `a + b·rows + d·rows·context`, fitted to 130 measured chunks, is off by
  15% at the median and 62% at worst. What a context adds does not grow
  with the rows: over 262k tokens it is 39 ms for any chunk of 64 to 498
  rows, 76 ms from 777 and 117 ms from 1,552, a step each time the
  attention's blocks outnumber the GPU's 152 SMs. Hence the slope of the
  measured chunk above, not an interpolated one: at a 128-row spacing it
  takes the worst error on chunks near those steps from 19.5% to 6.1%.
- **A chunk over an empty sequence is not smooth in its rows.** 512 rows
  take 23.4 ms and 513 take 25.1; 640 rows are 6% faster than 622 or 645.
  On 96 random chunks of up to 4,096 rows over 1k to 262k tokens:

  | empty-sequence spacing | scenarios | p50 | p90 | worst |
  | --- | ---: | ---: | ---: | ---: |
  | every 32 rows | 172 | 0.57% | 2.23% | 5.32% |
  | every 64 rows | 108 | 0.52% | 3.11% | 6.22% |
  | every 128 rows | 76 | 1.04% | 3.97% | 7.60% |
  | every 256 rows | 60 | 1.50% | 5.87% | 10.53% |

  Halving the spacing buys about a point of the worst error, and the
  errors that remain are chunks of 256 to 1,024 rows (p50 2.06%).
- **Sampling where the points mispredict each other was not worth its
  rounds.** An adaptive sampler bisecting to a 1% tolerance priced steps
  worse with 30 scenarios (5.02% at worst) than every size at two contexts
  does with 22. On chunks it reached 2.49% at worst with 127 scenarios,
  in six bench runs, chasing a tolerance below the 0.8% two runs of the
  same chunk differ by.
- **Twelve samples after four warmups are what a chunk needs.** A step's
  first replay is 0.4 to 1.8% slow and every later one within 0.1%. A
  chunk's replays each scatter by 2 to 5%, and the median of twelve is
  within 0.56% of the settled value, of eight within 1.03%; two runs of
  the same workload differ by up to 0.81%.

The file above, benched as written (194 scenarios, six minutes on one
GB300, 2026-11-18), prices 48 random steps whose sequences sit at mixed
depths to within 1.39% (p90 1.08%, their total +0.21%), and 96 random
chunks of up to 4,096 rows over 1k to 262k tokens to within 4.47% (p90
2.22%, their total −0.54%).

What it does not say: every row of a measured step is real, so a batch
padded up to its size is priced as if full; host time between steps and
the pool's copies are no program's; and a large chunk's time moves with
the GPU's clock, which a 4,096-row chunk over 131k tokens left at 1,582 MHz
from 2,070, so a server under sustained load may run it slower than its
price. One manifest on one GPU is all that was measured.

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
scenario  decode-g8-r1-kv4096
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
