# GLM-5.3 TP8 decode MoE v2

Date: 2026-09-26. Machine: **single-node TP8**. Integration owner retains control of serving.

## Status and scope

Three final cubins are built. The schema-v5 manifest passes `kern verify`.
The final FP8-WGMMA implementation passed GPU A/B at buckets 1/2/4/8/16:
**all tested stages and outputs are bit-exact**, including partial buckets,
all-pad, changing non-prefix validity, and 20 graph replays per case.
Real-model/TP8 serving parity and architecture latency targets are not yet
validated. **Do not infer model parity from synthetic tests.**

No shared source, existing manifest, existing cubin, checkpoint, allreduce,
dense MLP, DSA or KDA implementation was changed. Only new MoE files/artifacts
were created. `ops_moe.py` remains the rollback implementation.

Important corrections to the initial design:

* The live MoE GEMMs are mined Triton `fused_moe_kernel`, modules 335/337.
  DeepGEMM 1d2d is the **dense MLP** path. Its raw weight-scale pointer and
  activation-scale TMA contract is untouched.
* The live source emits **nine launches** excluding AR: router, topk, align,
  quant-A, W13, activation, quant-B, W2, sum. The seven attribution labels omit
  the two quantizers. V2 emits **three launches**, in one `moe_decode_v2` op:
  252 fewer kernel launches and 336 fewer op calls across 42 MoE layers.
* N=32 cannot independently quantize the model's 128-element activation
  groups. A cross-CTA completion epilogue is required unless the layout/tile
  changes or quantization is moved to another launch.

## 1. Final composition

Let `b=seqs=tokens`, the static decode bucket in 1..16, and let `v` be the
number of nonzero device `valid` values, at any positions. Each valid token
has eight distinct routed experts and shared expert 288. Let `U` be the
number of distinct selected experts. For v>0, `U <= 8v+1`; for v=0, U=0.
Weights remain `[289,512,4096]` and `[289,4096,256]`, with current FP32 block
scales. **No interleave or checkpoint preparation is needed.**

| entry | block | grid (b) | launched CTAs b1 / b8 | CTAs with weight work b1 / b8 |
|---|---|---|---|---|
| `glm53_moe_v2_route` | 256,1,1 | 96,1,1 | 96 / 96 | 96 / 96 |
| `glm53_moe_v2_w13` | 128,1,1 | 144b,1,1 | 144 / 1152 | 16U: 144 / at most 1040 |
| `glm53_moe_v2_w2` | 128,1,1 | 288b,1,1 | 288 / 2304 | 32U: 288 / at most 2080 |

These are finite grids, not persistent/cooperative kernels. Empty W13 CTAs
return before any expert weight load. The first b W2 CTAs also own invalid-row output zeroing; absent expert tiles
return without a weight load.
There is still empty-CTA scheduling cost: “drop padding” does not mean zero
launched CTAs for a fixed bucket.

### 1.1 Router + topk + align + input quantization

Each of 96 CTAs computes three router columns for all b rows. The kernel
keeps the tiny GEMM's 256-thread partition, two vectors of eight BF16 values
per thread, FP32 FMA sequence, xor warp reduction and ascending eight-warp
sum. The router has 288 columns, not 289; shared expert 288 is appended by
selection, not evaluated by a router dot product.

The first b CTAs also quantize one input row each. Eight lanes cover a
128-element group. Scale = `max(amax,1e-10)/448`, with the same rounded
reciprocal constant and fast division as module 333. Scales are row-major
`[16,32]`; no TMA is used by these MoE kernels.

Each producer thread fences its global stores. After a CTA barrier, thread
0 issues a device-scope acq_rel increment of the route counter. The CTA that
receives ticket 95 does selection and align. The other CTAs return: **no
CTA spins and no grid co-residency assumption is needed**.

The last CTA gives one warp to each row (two iterations at b>8): sigmoid,
bias-for-ranking only, eight max rounds with lower expert ID winning ties,
shared slot, and renormalization. The reduction order for the eight selected
weights follows module 411's PTX: sum four sequential values, sum the other
four sequential values, then add. Shared weight is `(routed_sum/2.5)/norm`,
not a substituted constant 0.4. Invalid rows get all IDs -1 and weights zero.

### 1.2 Compact align: no -1 expert bucket

1. Count selected IDs 0..288 in shared memory. Do not count invalid pairs.
2. Prefix-scan **presence flags**, not padded token counts. This gives one
   compact tile per present expert, in ascending expert-ID order.
3. For each pair `p=row*9+slot`, count earlier pairs with the same expert.
   Place it in `sorted[compact_expert*16 + ordinal]`.
4. Fill every unused tile lane with sentinel `9b`, not -1. Write
   `n_post=16U`; unused expert entries are -1.

Why 16 lanes suffice: a token never selects an expert twice; b<=16. Thus
one expert receives at most 16 pairs. Pair order inside an expert tile is
ascending `row*9+slot`. Output remains pair-major, so align order never
changes the subsequent token/top-k reduction.

`sorted` has 144x16 entries, versus 144x64 before. Worst-case active capacity
is only 129x16 at b16. No host readback, shape transfer or per-step allocation
is used. Non-prefix validity masks are supported.

### 1.3 W13 + activation + quantization

Sixteen N=32 CTAs compute each present expert's 512 projected values. Keep
weights non-interleaved: N-tiles 0..7 are gate, 8..15 are up. Quant scales
index the original 128-row N groups; this avoids breaking the checkpoint's
2-D scaling when interleaving rows smaller than a quantization block.

Logical align M is 16. The **WGMMA computation tile M is 64**, with the last
48 rows masked. This distinction matters: Triton lowered an actual M=16
`tl.dot` to FP16 MMA with explicit FP8 conversion. The final implementation
emits `wgmma.mma_async...m64n32k32.f32.e4m3.e4m3`, verified in PTX/SASS.
K is processed in 32 groups of 128, like module 335.

CTAs store BF16 C1 into op-private scratch. Every thread executes
`membar.gl`, then a CTA barrier, then a scalar device acq_rel ticket. Ticket
15 owns the activation epilogue for all real rows of that expert. Loads use
`.cg`; the compiler broadcasts the atomic return through shared memory and
`bar.sync`. This orders the producers' stores before epilogue reads.

The epilogue clamps gate above +10 only, clamps up to [-10,+10], evaluates
SiLU using the captured fast exp/div lowering, rounds the product to BF16,
and quantizes **full 128-element groups** into E4M3 plus row-major `[144,2]`
FP32 scales. No separate SiLU or quant launch is needed.

This keeps C1 scratch (144 KiB). Fusion here removes launch boundaries, not
all intermediate traffic. For decode that small traffic is preferable to
repacking all 42 layers' expert weights or changing quantization semantics.

### 1.4 W2 + sum: parallel experts, last-CTA ordered reduction

The originally proposed token-local loop over all nine experts was built and
tested. It was bit-exact but too slow: about 64/100/136 us for the complete
block at bs1/4/8, versus 60/79/98 us for the baseline. See
`moe_v2_serial_w2_test.json`. **That design was rejected.**

The final kernel uses expert-major W2 CTAs, matching baseline M=64, N=128,
K=128 WGMMA and the same compact 16-row align. For each real pair, a CTA
computes two K groups, applies the router weight, and stores BF16 C2 into
private scratch. It fences and increments one completion counter for each
`(token,128-column tile)` that it wrote. There are 16x32 counters, reset by
the router on every call. Each counter receives exactly nine tickets.

Ticket 8 owns the corresponding output tile. It loads the nine BF16 partials
with `.cg`, converts them to FP32, adds them in slot order 0..8, multiplies
by 2.5, and rounds once to BF16. Completion order selects the CTA, **not the
floating-point summation order**. The output uses no floating-point atomic.
This preserves both W2 and reduction numerics, and retains expert-parallel
weight reuse at bs4/8.

Keeping the per-expert BF16 cut is essential: module 337 writes BF16 C2 and
module 487 reads that buffer before summation. Accumulating unrounded expert
outputs would not match the current sglang-order oracle. V2 removes the
sum launch, **not C2 scratch/traffic or that rounding operation**. C2 is
1.125 MiB at the maximum 144 pairs. Router+W13+W2 still total three launches.

Only existing CTAs are used to clear invalid output rows: CTA `pid<b` clears
row pid if its valid flag is zero, before processing any expert tile. This
works even when there are no selected experts. No valid row is cleared.

## 2. Scratch, graph replay, and ABI

All intermediates belong to **one** op, so no scratch pointer crosses op
boundaries. Kern shares that op's scratch across layer calls; the stream
serializes them. The route counter starts at zero through
`crates/kern-runtime/src/device.rs:alloc` (`alloc_zeros`). The last router
CTA resets it on every call. Router also clears all 144 W13 and 512 W2 counters on
every call. Do not depend on per-call scratch zeroing: there is none.

The route counter is an implementation-private completed-call invariant:
it is zero at entry/exit. The first launch declares this scratch slot as an
output, as required by kern's scratch dataflow checker. No nonzero state is
carried between calls. This op is **not reentrant on concurrent streams
sharing one scratch allocation**. The production serialized decode graph
satisfies that condition. A failed/aborted kernel requires runtime reload.

The current manifest uses ordinary graph edges (PDL is **not enabled**).
All three entries contain `griddepcontrol.wait` / ACQBULK. No early dependent
launch trigger is used. No cooperative launch or cross-rank buffer is used.

ELF `EIATTR_KPARAM_INFO` was checked, not inferred from source signatures:

| entry | compiled ABI | offsets |
|---|---|---|
| route | 15 pointers, i32 rows | pointers 0..112 step 8; rows at 120 |
| W13 | 11 pointers, i32 rows, 2 hidden i64 | pointers 0..80; rows 88; hidden 96,104 |
| W2 | 12 pointers, i32 rows, 2 hidden i64 | pointers 0..88; rows 96; hidden 104,112 |

Both hidden Triton pointers are null. No source argument has disappeared
from these particular compiled kernels. `moe_v2_build.json` records full
SHA256, dynamic shared memory, resource use, and ELF parameter offsets.
`check_moe_v2.py` checks pins, ABI, WGMMA lowering and schema verification.
The build also writes TTIR/TTGIR/PTX, ELF and SASS evidence next to the source.
Final resource use (per thread/CTA): router 64 registers, 8,708 B static shared;
W13 56 registers, 36,864 B dynamic plus 1,024 B static shared; W2 165 registers,
73,728 B dynamic plus 1,024 B static shared. No W13/W2 stack or local spill.
Router reports a 64-B stack; its dynamic selection arrays remain a tuning target.
Do not reuse a JSON pin after rebuilding only one artifact by hand.

## 3. Why not DeepGEMM MegaMoE

Inspected the installed `deep_gemm/impls/sm90_fp8_mega_moe.cuh` and
`layout/mega_moe.cuh` in uv archive `Dw0446rV0s3nvADy`.

* The entry takes a by-value `SymBuffer<kNumRanks>`, dispatches token indices
  to destination ranks and uses NVLink/grid barriers and receive counters.
* Source asserts `kNumExperts % kNumRanks == 0`. 289 experts at eight ranks
  does not satisfy this. More importantly, these are TP intermediate
  shards, not EP-owned whole experts.
* Its N tile is constrained to 128/256/512, not the proposed decode N=32.
* A rank-local kNumRanks=1 adaptation might be possible. It would require a
  new symmetric-buffer workspace/layout and scheduler audit, plus tests of
  intermediate=256 and reduction/rounding semantics. It is **not validated**.

Thus this patch uses the current Triton FP8-WGMMA computation pattern and
small handwritten router glue. It does not pretend the EP entry is a TP
replacement. Existing 1d2d cubins cannot, without changing their code,
execute a SiLU/quant/sum epilogue in the same launch.

## 4. Build and integration

From the kern repo root:

```sh
PYTHONDONTWRITEBYTECODE=1 <sglang-python> \
  tools/glm53/kernels/build_moe_v2.py
PYTHONDONTWRITEBYTECODE=1 <sglang-python> \
  tools/glm53/kernels/check_moe_v2.py
```

Build needs no GPU. The CUDA portion uses the same nvcc/sm_90a flow as
`build_kernels.sh`, but builds **only the new source**. Running the existing
wildcard script would rebuild another agent's sources; it was not run.
The two Triton kernels use offline `ASTSource`/`GPUTarget(cuda,90,32)`.
Three cubins land in `kernels-glm53-handwritten/glm53_moe_v2_{route,w13,w2}.cubin`.

**Exact integration point in `gen.py:build`:** keep importing and using
`ops_moe` (dense MLP, AR, and the existing pre-fusion overrides need it).
Import `ops_moe_v2`, and insert this one call after constructing raw `m`,
immediately before `wire_allreduce(m, allreduce, pdl=ar_pdl)`:

```python
ops_moe_v2.fuse_manifest(m)
```

This is the replacement function; **do not simply swap the ops_moe import**.
It adds `ops_moe_v2.ops()['moe_decode_v2']`, replaces the nine-call MoE region
with one call, keeps its output `sub_out` and the next `moe_ar` call, and
removes unused old MoE ops/workspaces. Dense layers 0..2 are unchanged.
`resolve_modules`, `lower_wire`, and `normalize` then run unchanged. Existing
`--bundle` copies all three new pinned modules. No new weights or binds.

In-op scratch is fixed at decode maximum 16; rows/grid use bucket `seqs`.
Do not use this op for prefill or a rows>1 speculative program.
The helper rejects interleaved old MoE probe calls: use `--probes` off for
this integration and use the standalone stage A/B harness instead.
`check_moe_v2.py` exercises this exact hook through an in-memory wrapper,
without modifying gen.py, and writes a four-layer verification manifest. `--layers 45` additionally
checks all 42 MoE replacements; that full-manifest check passed too.

## 5. Validation and performance status

CPU checks passed: all three ABI/pin checks, WGMMA PTX checks, four-layer and full 45-layer
schema-v5 `kern verify`, and 1,024 randomized CPU align invariant cases for
buckets 1..16 with non-prefix validity masks.

The initial M=16/MMA smoke test (`moe_v2_smoke.json`, **not the final WGMMA
cubins**) passed bs1 full-valid and all-pad, plus five graph replays:

* Router logits, selected IDs/weights, input FP8 values/scales: bit-exact.
* Align: exact pair bijection, correct expert ownership, no invalid pairs.
* Output relative RMS error 0.002228, max absolute error 0.03125; finite.
* All-pad output zero and IDs -1; route counter returns to zero.
* Warm-cache graph block latency: baseline 60.06 us, initial v2 43.05 us.

These measurements cover random tensors, one GPU/rank, one layer. They are
not the 42-layer model, not cold-HBM traffic, and not serving throughput.
The final build uses FP8 WGMMA and **parallel expert-major W2**. Its report
is `moe_v2_test.json`, with exact artifact pins:

* route `a32452292cef7fc1e99d7846e8aeb41cd7b220a9b4ebbcb1201191be840bb32d`
* W13 `860cafdc01a8d2d3fa958ce5f97bad429287b18e2171c2e6eb2fea8e957cf6f1`
* W2 `d28dfeb85e455230d727d85519fa373f537c530574c3ca11e31b50765723aea0`

Across 17 cases at buckets 1/2/4/8/16 (including all-pad and 3/5/6/7 live
rows in b8): router scores, weights, input FP8/scales, C1, hidden FP8/scales,
final output and isolated W2 all had **zero differing values**. Align checks,
zero-counter checks, invalid-ID/output checks, and 20 exact graph replays
passed. A b16 capture changed to non-prefix holes and back to full-valid;
both valid outputs were bit-exact and invalid outputs were zero.

A new `sgl` tmux session was starting around the final run (GPU utilization
was zero at the initial check). Thus use that run's timings as **indicative**,
not certified uncontended benchmarks. The test guard was strengthened to
reject sglang/integration panes and processes, and an occupied selected GPU.
No further GPU test was started while that server was active.

### Required short GPU A/B

The harness checks `tmux ls`, pane commands, `nvidia-smi`, active kern processes
and GPU utilization **before** importing/initializing CUDA. It also rejects
active sglang/integration panes and >1 GiB usage on the selected GPU. It exits rather
than compete with an active integration/server run. Check again before each
invocation. Do not stop another process to make room.

```sh
export LD_LIBRARY_PATH=<nccl + cuda-compat libs for your node>
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 timeout 90 \
 <sglang-python> tools/glm53/kernels/test_moe_v2.py \
 --batches 1,2,4,8,16 --report tools/glm53/kernels/moe_v2_test.json
```

Additional short cases: `--batches 3,5,6,7,9,10,11,12,13,14,15
--replays 5 --timing-repeats 5`; second seed; and `--zero-input` for exact
routing ties, zero quantization amax and all-zero expert output.

The test calls the **actual current mined cubins**, including gen.py's
`num_valid_tokens=9b` fix, and the **actual built v2 cubins** through the
CUDA driver. It checks stage deltas, isolated W2, align, all-pad cases,
partial buckets (including 3/5/6 live rows in b8), repeated graph replay,
and changing a non-prefix validity mask within the same graph. Reports
include artifact hashes, so evidence cannot silently refer to an old build.

Final synthetic gate: IDs and every recorded stage must be exact; no
NaN/Inf; pad output exactly zero; align must be a bijection; replay and
dynamic-valid outputs must be exact. This is still **not approval for model
deployment**. Any remaining deviation
needs cut-level investigation against recorded real layer activations.
After stage tests, the integration owner should run the existing sglang
oracle/greedy suite at bs1/2/4/8, including bucket padding, and the architecture
criterion of >=99.5% greedy agreement on 1k tokens at long context.

### Performance model, not a promise

Final single-layer graph diagnostic, microseconds, full-valid random input:

| bucket | baseline | final v2 | reduction | x42 diagnostic delta |
|---|---:|---:|---:|---:|
| 1 | 59.76 | 40.53 | 32.2% | 0.81 ms |
| 2 | 68.22 | 52.54 | 23.0% | 0.66 ms |
| 4 | 79.52 | 66.96 | 15.8% | 0.53 ms |
| 8 | 98.64 | 89.71 | 9.1% | 0.38 ms |
| 16 | 126.71 | 139.02 | **9.7% slower** | **-0.52 ms** |

These are not serving measurements. Input/weight locality differs from a
42-layer decode step; bs1 fits more selected weights in L2 than larger
batches. The possible sglang startup overlap above further limits timing
claims. The x42 column is a diagnostic scaling, **not a measured saving**.
The objective's latency range is bs1..8; b16 is correct but currently slower.

Attribution is inflated by about 1.7x: use its MoE **share**, not absolute
attributed milliseconds. The supplied 10.16-ms step and 25–30% share imply
2.54–3.05 ms for the current bs1 MoE region. Applying the final diagnostic
ratio gives about **0.82–0.98 ms bs1 saving**. At bs8, 12.22 ms x25–30% x9.1%
gives **0.28–0.33 ms**. With no supplied bs4 whole-step share, use the 0.53-ms
x42 diagnostic delta only as an initial hypothesis.

Conservative integration planning estimates: **bs1 ~0.8 ms, bs4 ~0.5 ms,
bs8 ~0.3 ms saved/step**. Confirm them with uncontended whole-graph timing.
Pad suppression can add savings at non-full buckets. At b8, measured v2
block times for 3/5/6/7 live rows were 68.25/75.61/84.08/85.48 us, versus
89.71 us full-valid. These depend on the synthetic expert overlap.

Architecture budgets are 0.76/1.60/3.00 ms for bs1/4/8, or
18.10/38.10/71.43 us per layer. The final diagnostic values
40.53/66.96/89.71 us are **above all three budgets**. Thus the three-launch
objective is met, but the P2 latency budget is **not demonstrated or met by
this microbenchmark**. Claiming <5.68-ms whole-step latency from this patch
alone would be incorrect. Further GEMM/scheduler tuning and the other
architecture work remain necessary.

## 6. Open risks / owner acceptance gates

1. Final-build synthetic multi-batch A/B passed; real-model/TP8 oracle and
   clean whole-step latency remain required. Other b values, second seeds,
   zero-input ties and sanitizer tests are scripted but not yet run.
2. Three last-CTA completion chains passed repeated replay and changing
   validity. Longer changing-activation stress and sanitizer memcheck remain
   recommended when GPUs are free.
3. W2 parallel expert scheduling retains C2 scratch and adds completion
   counters. Measure b4/b8 with cold model weights and real routing skew.
   The slower serial-nine-expert W2 variant was rejected.
4. M=64 WGMMA computes masked lanes. Measure its register/shared
   memory/occupancy tradeoff versus the earlier M=16 MMA variant.
5. Warm-cache microseconds cannot establish the architecture MoE budget.
   Owner should use whole-graph wall time and attributed shares.
6. No PDL optimization is claimed. Add programmatic edges only after the
   existing kern PDL source/ABI checks and graph replay tests.
7. Allreduce is unchanged and remains a separate launch. This patch does
   not claim or rely on the other agent's allreduce changes.
