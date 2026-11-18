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

## Candidates: choosing an op's implementation

An op is an interface and an implementation, and a call binds only the
interface. So "which kernel, at which launch geometry, for which batch
size" is a question about one op at a time, and every answer to it is a
manifest that differs from the served one in that op's `impl`. A
candidate is such an answer as a file: the `modules` and `ops` sections of
a manifest and nothing else, each op under the name and with the `params`
of the op it stands in for.

```json
{
 "modules": {
  "vl_64": {"source": "vl_64.cubin", "sha256": "5b0c…"}
 },
 "ops": {
  "verify_linear_16384_5120": {
   "params": ["in buffer<bf16>", "in buffer<fp8e4m3>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i32"],
   "impl": {"launches": [
    {"module": "vl_64", "entry": "gemm", "block": [448, 1, 1],
     "grid": [{"ceil_div": ["tokens", 64]}, 152, 1], "shared_mem": 220160}
   ]}
  }
 }
}
```

A kernel written in another language, the same cubin at another grid, and
the same grid over another cubin are all candidates, and nothing tells
them apart. Tuning launch geometry needs no syntax of its own: the kernel's
author writes one file per geometry from the script that already knows the
kernel's parameters (a loop over tile sizes that emits the `grid`,
`block` and `shared_mem` each one needs), and a directory of them is the
sweep. A candidate carries no `when` unless there are shapes it does not
implement: where each one runs is what is being asked.

```bash
kern bench qwen38 --workload served.toml --out pick.json --candidate candidates/
```

`--candidate` takes a file or a directory of `.json` files, and may be
given more than once; a candidate is named after its file. Its cubins are
found in the target's kernels directory by hash, like any module's.

```
ops       5 with candidates · verify_linear_14336_5120 · verify_linear_16384_5120 · …
device    NVIDIA GB300 · 152 SMs · L2 129 MiB · software timestamps
plan      57 scenarios · 0 dropped · 718272 tokens of state · 17 seqs
baseline  2039.7 s · 73227 calls   96.6s
candidate dsl-016 · 3475.9 s · +70.4% · its ops 704.0 → 2138.1 s · identical · 57 scenarios · 9610 logits rows   126.2s
candidate dsl-032 · 2558.5 s · +25.4% · its ops 704.0 → 1222.5 s · identical · 57 scenarios · 9610 logits rows   105.3s
…
candidate dsl-256 · 2729.7 s · +33.8% · its ops 704.0 → 1394.0 s · identical · 57 scenarios · 9610 logits rows   104.1s
baseline  2039.6 s · -0.0% · identical · 57 scenarios · 9610 logits rows   93.9s
pick      verify_linear_14336_5120 · `seqs` ..1 dsl-032 · 2.. baseline · 28.0 → 27.8 s
pick      verify_linear_16384_5120 · `seqs` ..2 dsl-032 · 3.. baseline · 91.3 → 90.8 s
…
pick      verify_linear_5120_6144 · `seqs` baseline · 97.6 → 97.6 s
confirm   2038.2 s measured · 2038.5 s predicted · identical · 57 scenarios · 9610 logits rows   94.4s
cost      2039.7 → 2038.2 s · -0.1% · 24 launches of 17 modules over 5 ops
out       results/round-dsl-tiles.json   813.9s
```

A `candidate` line is the whole workload under that candidate against
the baseline, then what its own ops cost before and after, then what the
gate found. `pick` is one op's tiling of its var and what the op costs
under it; `confirm` and `cost` are the picked manifest measured.

**Every run is the whole manifest over the whole workload.** Each
candidate is spliced in (`kern_manifest::candidate::splice`: the interface
is checked, its modules come along, the manifest is verified again) and
benched exactly as above, in the same process, the weights left on the
device from one manifest to the next. The manifest itself runs first and
last. An op's time is read off the kernel records of the real graph, so a
candidate is priced where it will run: behind the kernel that precedes
it, on the cache that kernel left.

**A candidate is chosen from only where it holds.** Time alone would
choose a fast kernel that computes something else. After every scenario
the run's outputs are held against the first run's: the tokens and counts
the program hands back, and every row of every distribution over tokens
it writes (a `logits*` buffer, or any float buffer as wide as the token
domain). A scenario is `identical`; or `within`, every row's KL(manifest ‖
candidate) at most 0.01 nat, `kern test`'s default, so that a token that
moved under it moved on a tie; or it `differs`. A program that hands
nothing back leaves a scenario `unjudged`. A candidate is chosen from at
a value of the var only if every scenario there is identical or within.
A candidate that differs is still measured and reported, never chosen:
whether a different distribution is acceptable is `kern test
e2e-logits`'s question, and a candidate that passes it becomes the
manifest, against which its own variants are then identical.
A candidate whose replay does not reproduce its own tokens fails the gate
below and is reported as `failed`; the run goes on without it.

**The choice is a tiling of one var, with as few ranges as the noise
allows.** A manifest chooses with `when`, a range of one var, so an op's
choice is a candidate per range of the var its launches turn on: the one
they are already conditioned on, else the one their grids read. The
workload's scenarios give the values of that var and, per candidate, what
the op cost at each (weight × the graph's p50 × the op's share). Choosing
per value is the cheapest tiling and the one with the most ranges, and
every range is a launch the manifest carries and a cubin the runtime
loads. So the pick is the tiling of fewest ranges whose cost is within
the noise of that cheapest one, where the noise is the largest relative
difference the manifest's own two runs showed for the op at any value. A
boundary that buys less than the measurement can resolve is not drawn. A
range ends at the last value measured in it, the first and last are open,
so every value has an implementation.

**The pick is run before it is written.** Ops are chosen separately and
may come from different candidates, which makes the result a manifest
nobody measured. `confirm` is that manifest over the same workload, held
to the same outputs, its cost beside the sum the choice predicted.
Identical implementations combine into an identical manifest; ones merely
within the limit need not. If their combination does not hold, the choice
is made once more from the identical alone and confirmed again; if that
does not hold either the command fails, the report still written.

**What the gate cannot see.** A scenario is one call of a program, and a
program that drafts its own tokens feeds each row what the last one
chose. One tie that flips changes the input of every row after it, and
those rows then differ by far more than the kernel moved anything. A
candidate that is not bit-identical therefore seldom holds in every
scenario of such a program, and where it holds is partly where no tie
happened to flip. That is the gate working as intended: it proves a
choice or declines to make it, and what it declines is `kern test
e2e-logits`'s to judge, along forced tokens and against a measured
noise.

Measured on one GB300 (tray06, 2026-11-18, software timestamps), the
served Qwen3.8-27B FP8 manifest over the 57 `round` scenarios of its
trace, five W8A16 GEMM ops (`bench_results/2026-11-18-bench-candidates`):

- Against the manifest with a CuTe DSL GEMM in place at one tile (64
  rows) for every batch: the kernel's five row tiles as five candidates
  with no `when`. All are bit-identical to it in all 57 scenarios, and
  the pick tiles `tokens` in four ranges on three ops and five on two
  (`..32 dsl-032 · 33..64 baseline · 65..128 dsl-128 · 129.. dsl-256`),
  2220.7 → 2037.6 s measured against 2039.9 predicted.
- Against the same kernel as its author had tiled it by hand, five ranges
  of `seqs` per op (2039.7 s): the pick keeps the hand tiling but for a
  32-row tile where it had 16, on three ops, 2038.2 s measured against
  2038.5 predicted.
- Against the served manifest (Triton GEMM): the same DSL kernel costs
  2042.3 s to its 2753.0 (-25.8%) and is within the limit in 53 of 57
  scenarios; a hand-written tcgen05 kernel costs 3098.7 s and holds in
  52. The pick takes the DSL kernel at the two ranges of `seqs` where it
  held everywhere, 2753.0 → 2391.1 s measured against 2392.7 predicted,
  within the limit in every scenario (KL at most 7e-3, 11 ties).
- Two controls: a candidate whose launches run at `seqs` 3..4 only holds
  in exactly the 12 scenarios there and is not chosen from elsewhere; a
  build of the DSL kernel missing a fence between its TMA writes and its
  reads does not reproduce its own tokens, is reported `failed`, and the
  run goes on.
- The manifest's two runs are bit-identical in every scenario and under
  0.01% apart in cost; the noise a pick is held to, the worst value of an
  op, came out between 0.05% and 0.4%.

The report has the header of a plain run, then `baseline`, `candidates`
and `baseline_again` (each scenario's vars, program time, per-op times
and gate evidence: verdict, max and mean KL, flips, flips beyond the
limit, the widest margin that flipped), `choices` (per op the table the
pick was made from, its noise, the ranges and why they cost what they
do, and the confirming run) and `picked`: the choice as a candidate file,
tiled with `when`, ready to be spliced into the generator's output. Fed
back as the only candidate it is chosen again.

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
- `--candidate` chooses along one var. An op whose candidates are
  conditioned on two vars, or whose grids read two and are conditioned on
  none, gets one implementation for every shape.
- A candidate's value between two measured ones runs what the larger one
  runs. A workload from a served trace measures the values the service
  sees; a hand-written one should name every value that matters.
- The manifest's distributions of every scenario stay on the device for
  the whole run, to be compared there: a few hundred MiB a scenario for a
  250k vocabulary at 256 rows.
