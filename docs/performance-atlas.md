# Single-GPU Performance Atlas

The `/perf/` page connects a manifest's programs, calls and implementations to
measured latency. The runner is opt-in (`kern bench`); serving behavior and the
manifest schema are unchanged. It uses `Protocol` to discover programs and
stage their declared inputs, without model-specific ABI names in the runner.

## Scope and data provenance

This is a measurement and calibrated-composition explorer, not yet a
standalone kernel-only simulator. Collecting a report currently loads model
weights, executes real prefixes, walks the program to obtain op inputs, and
measures whole programs. It does not start a serving endpoint. Once exported,
the browser can inspect the evidence and compute what-if estimates without
running the model or accessing a GPU.

| Display | Source |
| --- | --- |
| Measured program and batch/context heatmap | Whole-program GPU measurements |
| Cold L2 / warm replay | Isolated op measurements at inputs from the program |
| In-program timing, contribution shares and program bar widths | Actual program GPU activity trace; shares are normalized to graph time |
| Calibrated op-cost estimate | Cold/warm microbench sums with parameters fitted to training program times |
| What-if speedup | Hypothetical savings from measured shares, not achieved performance |
| Hardware anchors | Device-local calibration measurements before and after the sweep |

The estimate is `scale * (alpha * cold_sum + (1 - alpha) * warm_sum)`.
Each sum counts every program call, reusing its measured case where applicable.
Both `alpha` and `scale` are fitted per program; `alpha` is not an L2 hit rate.
Training points participate in that fit, so their displayed errors are not
independent validation. Holdout program times are excluded from fitting, but
holdout microbench costs are still measured. Trace attribution does not enter
the cost prediction.

The current data stops at 2,048 cached context tokens. Long-context microbench
coverage, prediction for unmeasured configurations, and kernel-only input/state
construction without loading the model remain future work. The 2,048-token
prefill chunk bound is separate from a model's total context capacity.

## Measured snapshot · 2026-09-05

Each model ran independently on one GB300. Both use 31 workloads and 32
timed samples per mode. All call sequences and trajectory token checks passed.

| Model | Call observations | Configurations | Holdout median / max absolute error | Max difference in untraced repeat |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-4B | 13,477 | 2,609 | 0.52% / 2.56% | 0.79% |
| Qwen3.8-27B | 29,351 | 5,282 | 0.54% / 2.76% | 0.73% |

These are composition holdouts with measured local op costs, not unseen-shape
or serving predictions. The GPU activity/event timer discrepancy stayed below
0.23% and 0.06%, respectively. A separate application-replay counter check of
the eviction protocol saw only 1.25–3.75 KiB of DRAM reads for the resident
32.3 MiB probe, versus approximately the full payload after eviction.

## Reproduce

Configure a target in `kern.toml` with its manifest, kernels, weights and
tokenizer. Use an idle GPU; neither the runner nor the commands below stop
other jobs or change device clocks. Put experiments in a dedicated directory.

```sh
cargo build --release -p kern-run --bin kern
nsys profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none \
  --output results/sweep \
  target/release/kern bench qwen3-4b --gpu 0 \
    --workload tools/profiles/single-gpu.json --out results/raw.json
nsys export --type sqlite --output results/sweep.sqlite results/sweep.nsys-rep
python3 tools/profile_trace.py results/raw.json results/sweep.sqlite \
  --out results/activity.json
python3 tools/profile_export.py results/activity.json --out website/public/perf/data
```

Pass multiple activity reports to the export command to populate the model
selector. `qwen3.8-27b` uses the same runner and workload. Full trace archives
are experiment artifacts, not website assets. The checked-in workload has 31
scenarios with 32 timed samples per mode: batches 1–16, contexts 128–2048,
mixed-context batches, prompt lengths 1–2048, and extensions of existing KV.
Four whole-program target times are held out from composition calibration.

```sh
cd website
npm ci
npm run dev
# Open /perf/ on the local development server.
```

For a portable demo, after installing website dependencies:

```sh
node tools/profile_standalone.mjs --out results/performance-atlas.html
```

The resulting HTML embeds both models' data, JavaScript and styles. Open it
directly in a browser; it does not need a server, web fonts or a network
connection. JSON downloads work offline too. This does not publish the page
to production.

`kern bench ... --program-only` skips op microbenchmarks and records whole
programs for a quick repeat without an activity tracer. Those reports are
diagnostics, not valid inputs to the operator explorer exporter.
Supply these repeats with `profile_export.py --controls results/control.json`
to attach a same-workload cross-check. Output tokens must agree; the page
reports differences without interpreting them as pure tracing overhead.

## What is measured

- Each sequence has a private state lease and a deterministic, varied prose
  prefix executed by the actual model. Context lengths are not simulated by
  merely changing a scalar over uninitialized KV.
- Every call is visited in program order. Configurations are deduplicated
  conservatively by op, resolved arguments, shape, dtype, alias relationships,
  offsets and environment. Weight names are omitted; state offsets are not.
- An isolated call snapshots its declared writes, including whole opaque
  state allocations. Before every sample these writes are restored. Cold
  mode then streams over a seeded buffer at least eight times the reported
  L2 size. Warm mode primes the op, then restores its writes again. All this
  preparation is outside the timing span. Warm replay is not a guarantee
  that every input fits in L2.
- Cold/warm order alternates. Each mode retains every timed sample in order,
  including tails. Median, p10, p90, maximum, CV and block medians are exported.
  With 32 samples the page deliberately does not claim a reliable p99.
- Whole-program samples start with cold L2 and preserve natural reuse between
  calls. They repeat the same state pre-image, not successive decode steps.
  Token outputs from the whole program and the profiled call walk must match.
  This checks profiling restoration, **not independent model correctness**.

### Timing and observer overhead

CUDA events bracket the entire graph. The trace postprocessor uses GPU
activity start/end timestamps for op timing and in-program attribution.
Empty named kernels delimit samples; their durations, event records,
restoration and eviction are excluded. An op containing multiple launches
includes its internal gaps. Every program call's observed activity sequence
must match the isolated implementation sequence, with no unmapped activity.

Event-per-call measurements are also recorded as a diagnostic, but are not
used for the contribution chart. For short kernels, those event nodes can
substantially inflate the graph. The evidence retains the independent graph
event timing and the activity/event discrepancy. Tracing itself is not
assumed free; the demo is GPU execution evidence, not a serving benchmark.

### Hardware anchors

Before and after the sweep, measure local D2D copy, SM copy, streaming read,
L2-resident read, evicted read, an empty kernel, and a 4096³ BF16 GEMM.
Read/write copies count `2 × payload bytes`; a read counts payload bytes.
These are empirical references, not universal rooflines. Cache eviction is
a cache-sensitivity protocol, not an architectural flush instruction.

For independent counter validation use Nsight Compute application replay
with `--cache-control none --clock-control none`. Kernel replay can destroy
the warm precondition during profiling. `kern bench --calibrate-only` runs
only the hardware probes. Under the 12-sample smoke workload, filtering
`stream_read` with `--launch-skip 190 --launch-count 4` selects two resident
reads followed by two evicted reads; changing the sample count changes these
indices. Compare `dram__bytes_read.sum` with the probe payload, not just
latency. Counter-profiler timings are separate from the main samples. NVIDIA's
[profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#application-replay)
also describes application replay for cache states prepared by preceding GPU
work, and the requirement for deterministic launch matching across passes.

## Interpreting the page and AI quick view

The optimization map ranks observed program contribution. The inspector
shows cold/warm samples and the selected call's separate in-program samples.
A CV above 10% or p90/median above 1.15 marks observed variability in any
configuration or call. These are triage thresholds, not statistical tests.
Variability alone cannot establish an intrinsic kernel issue: cache state,
clocks, host interference and input data also matter. Stable and noisy ops
both remain visible; the quick view includes a separate variation watchlist.

The hypothetical speedup slider applies a fixed-workload contribution model.
It does not prove the kernel can attain the chosen speedup, or that cache
interactions stay unchanged after an optimization. Program widths use
activity duration; contribution shares are normalized to the independent
graph time, so inter-call gaps are not attributed as a separate operator.

The composition predictor fits a nonnegative mixture of cold and warm op
medians plus one scale per actual program. Holdout whole-program times never
enter fitting, but their local op costs **are measured**. Thus this validates
composition, not prediction on unseen shapes. All holdout errors are shown;
the displayed training residual envelope is neither p90 nor a confidence
interval. Summing per-op p90 values would not produce a valid program p90.

This first demo does not claim serving TTFT/TPOT: queueing, scheduling, host
launch loops, inter-step locality, prefix-cache policy, speculative acceptance
and multi-GPU communication remain outside scope. Batch and token count are
not the complete performance key; query/context split, shapes, dtype, strides,
layout, state and implementation choices also matter.
External library builds must also be pinned by the experiment environment
when comparing runs; a manifest hash alone is not a full performance-cache key.

## Checks

```sh
python3 -m unittest discover -s tools -p test_profile.py -v
cargo test -p kern-run --lib bench::tests
cd website && npm run build
```

`tools/profile_browser_check.py` exercises every exported workload, operator
filtering/sorting, call selection, speedup projection, downloads and desktop/
mobile overflow. Give it `--browser` and `--out`; `--url` accepts the hosted
page or an offline `file:` URL. Offline checks explicitly disable networking.

Evidence assets are losslessly gzip-compressed; the browser decompresses
them with `DecompressionStream`. The small AI quick-view JSON remains plain
text. Standalone HTML uses the same compressed evidence, with no CDN runtime.
