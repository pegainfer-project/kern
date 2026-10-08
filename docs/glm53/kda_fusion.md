# KDA decode fusion v2 — TP8, H100 sm_90a

Date: 2026-09-26. Host: **the bring-up node only**. Integration is owner-controlled.
**Status:** built and synthetic-cut tested; not integrated into serving.
Source scope: `tools/glm53/ops_kda_v2.py`,
`tools/glm53/kernels/glm53_kda_v2.cu`, this document.
No existing source, manifest, kernel bundle, or serving process was changed.
Build products and test records are in `/tmp/glm53-kda-v2/`, not in the
read-only `kernels-glm53*` directories.

## 1. Decision and scope

There are two independently selectable paths:

* **core** retains both original cuBLAS fg_b GEMMs. It replaces conv, delta,
  and gated norm with one kernel. This is the conservative numerical boundary.
* **fused** also computes f_b and g_b inside that kernel with BF16 tensor-core MMA and
  FP32 accumulation. It explicitly rounds both projection outputs to BF16.
  Projection parity must be checked independently from recurrence parity.

The qkvbfg direct GEMV is a separate, **opt-in experiment**. Do not infer that
removing split-K is a speed win. The first bs1 test was slower than cuBLAS.
**Selected geometry: 512 threads.** The Python default is conservative core
mode; full mode is the faster candidate for owner acceptance.
The default keeps cuBLAS. o_proj and its Lamport allreduce are untouched.

Current kern middle-chain launch counts are 5 -> 1 in full mode (two fg
GEMMs plus conv/delta/norm), saving 136 launches across 34 layers. Core mode
saves 68. A successful direct qkv replacement would save another 34 when
cuBLAS selects split-K. The roadmap uses a single batched fg GEMM in sglang,
so its launch-count arithmetic is not the current kern arithmetic.

This code is for **groups 1..16, rows=1**, not prefill, spans, MTP verification,
or a no-store recurrence. All variants always store the final state.

### Why the grid differs from the roadmap sketch

Gated norm needs all 128 output values in a head. Four independent V-tile CTAs
cannot complete that reduction safely without a cluster, another launch, or a
publication protocol. Also, duplicate conv reads plus one early state writer
would race across those CTAs.

V2 uses **one CTA per (sequence, head)**. Real-lease conv channels have single writers.
Warps own disjoint V ranges. A CTA barrier makes all rounded recurrence outputs
available to the norm. There is no global barrier, counter, spin loop, scratch
initialization, or state migration.

The first 128/256-thread versions left too much work on eight SMs at bs1.
512/1024-thread entries increase intra-head parallelism without changing any
per-element reduction order. Keep the selected geometry fixed in each graph.

## 2. Kernel table and ABI

All production core/fused entries take these **14 arguments in this order**:

```
F, f_b_or_forget, g_b_or_gproj, conv_weight_f32,
A_log_f32, dt_bias_f32, o_norm_bf16,
conv_state, ssm_state, conv_lines, ssm_lines, cu_seqlens,
o_bf16, S_i32
```

In `core`, arguments 1/2 are the original `[S,1024]` BF16 projection outputs.
In `fused`, they are `[1024,128]` BF16 weight matrices. Do not interchange them.
These are CUDA C++ exports, **not TileLang alphabetical arguments**.
The mined ABI in the test includes Triton's two trailing i64 scratch pointers.

| entry | block | grid | CTAs bs1 / bs8 / bs16 | static shared B | registers |
|---|---:|---|---:|---:|---:|
| `glm53_kda_core128` | 128 | `(S,8,1)` | 8 / 64 / 128 | 1540 | 32 |
| `glm53_kda_core256` | 256 | same | same | 1540 | 32 |
| `glm53_kda_core512` | 512 | same | same | 3080 | 32 |
| `glm53_kda_core1024` | 1024 | same | same | 3080 | 32 |
| `glm53_kda_fused128` | 128 | same | same | 1540 | 32 |
| `glm53_kda_fused256` | 256 | same | same | 1540 | 32 |
| `glm53_kda_fused512` | 512 | same | same | 3080 | 32 |
| `glm53_kda_fused1024` | 1024 | same | same | 3080 | 32 |
| `glm53_kda_fg` (diagnostic) | 128 | same | same | 512 | 30 |
| `glm53_kda_qkv_direct` (experimental) | 128 | `(834,1,1)` | 834 / 834 / 834 | 0 | 128 |

`*_debug` versions of the eight core/fused entries append three pointers:
raw BF16 delta output, BF16 conv output, FP32 rstd. They are for cut checks,
not performance. Production entries do not write these diagnostics.

No dynamic shared memory is required. There are no spill loads/stores in the
production fusion builds. The final cubin SHA256 is
`8a080e2b6179025e41c57257fa847885753a44921d736f96340b74e347d44eea`.
`build.json` pins both source and cubin; the test refuses a stale build. `build.log` is the authoritative resource record.

The qkv entry retains the six-argument original op ABI `(x,w,y,S,N,K)`.
It is specialized to the actual `N=3336, K=4096` contract, with runtime S.
Four warps produce four N rows per CTA. The batch-specialized internal bodies
reuse a loaded weight across live rows. There is no partial buffer or reducer.
Its summation order differs from cuBLAS and is not claimed bit-identical.

### PDL and graph safety

Every new entry executes `griddepcontrol.wait` before any dependent read, and
`griddepcontrol.launch_dependents` only after its output/state stores. SASS was
checked for `ACQBULK` and `PREEXIT`. There is no early producer publication.

No device allocation, host readback, stream change, or epoch counter occurs in
a kernel. All pointers and grids are graph-stable. The diagnostic test captures
ordinary CUDA graph edges; **kern's programmatic-edge integration still needs
a graph replay test**. Opcode presence alone is not an end-to-end PDL test.

## 3. State and lease invariants

Unchanged storage:

* Conv line: `18432 B = [3,3072] BF16`.
* SSM line: `524288 B = [8,128 V,128 K] FP32`, K contiguous.
* Conv sequence slot: `34*18432 B`; SSM slot: `34*524288 B`.
* Each line table is `[34,seqs]` with **static max pitch 16**. The call passes
  the layer's table row at **byte offset `kda_layer*64`**.
* Conv and SSM indices are loaded separately. They need not have equal values.
* Addressing uses 64-bit multiplication. There is no captured 581-line limit.
* No change to page unit 256, page IDs, leases, checkpoint, fork, or restore.

The CUDA entry checks `cu[s]==s` and `cu[s+1]-cu[s]==1`. Negative lines or an
invalid decode extent cause an early return, not a state access. This is a
safety guard, not support for empty/variable-length groups. Production padding
must keep the existing valid pad lease and one row per group. No new `valid`
mask is used: that would change the existing pad-state semantics.

**Padding caveat:** `tray.rs` maps all padding rows on a rank to one reserved
pad lease. Those rows can race with each other in the old chain and in this
fusion. The pad state is not a real sequence. Bitwise recurrence comparisons
require distinct active leases; do not claim deterministic bytes for the
shared pad slot. No new skip-pad policy or lease allocation is introduced.

## 4. Numerical construction and proof boundary

The source was derived from the **actual cached PTX**, not just the high-level
formula in `kda_mhc.md`. Cache binaries were hashed against the pinned oracles:

```
module_363 (delta):
d4cab2beb31f8b7478c0e795565a79eb26a6c0e18cc19c01d298effce6a4e24f
module_227 (norm):
6b7c156dce8428b4a344a77848a31b9fd9ae2fe049e545ab8228667b1266312e
```

Delta PTX cache directory:
`$HOME/.cache/sglang/triton/II6BKLOL4OPLTWHLAYZ4GXTMKXJ5JALEZ7AWS7XJ4RJT2OQCQQRA/`.
Norm PTX directory:
`$HOME/.cache/sglang/triton/HQZKUMJZJ2AKVK6EWSOS3AUW2LRU4U33CTLM3DTCO75J3NAZPFRA/`.
Conv oracle is the existing `module_conv_generic.cubin`, not the defective
581-line specialization. We do not rebuild or overwrite any oracle.

### Conv

Each channel computes four `fma.rn.f32` operations, in tap order 0,1,2,3,
starting from +0. SiLU uses `ex2.approx.f32(x*0x3fb8aa3b)` and
`div.full.f32`, not `expf`, approximate reciprocal, or `--use_fast_math`.
The output rounds to BF16 before q/k normalization. The state writes the
original BF16 inputs `(old1,old2,new)`, not the activated values.

### State recurrence: exact per-coordinate operation sequence

For each V row, a warp maps lane L to `K=4L+[0,1,2,3]`, just as module_363.
Define `R4(a,b)` as:

```
t = mul.rn(a1,b1)
t = fma.rn(a0,b0,t)
t = fma.rn(a2,b2,t)
t = fma.rn(a3,b3,t)
t = XOR-sum(t, offsets 16,8,4,2,1; add.rn at each step)
```

This is **not** an arbitrary associative sum. The slightly unusual 1,0,2,3
local order is the mined PTX order. The same operation tree is used for q/k
square sums, the state prediction, and the final retrieval.

1. q and k are loaded from BF16 conv outputs. Normalize with
   `sqrt.approx.ftz(sum+1e-6)` followed by `div.full`; q then multiplies the
   FP32 constant `128^-0.5`. Do not replace sqrt/div with rsqrt.
2. `A=ex2.approx(A_log*log2e)`;
   `x=add.rn(dt_bias, BF16(forget))`;
   `neg=fma.rn(-A,x,+0)`;
   `g=mul.rn(-5, div.full(1,1+exp(neg)))`;
   `decay=exp(g)` with the same exp lowering.
3. Initial state load retains the oracle's `add.rn(S,+0)`.
   **First** materialize `D[k]=mul.rn(S[k],decay[k])`.
4. `pred=R4(k,D)`, `dv=mul.rn(sub.rn(v,pred),beta)`.
5. **Then** `U[k]=fma.rn(k[k],dv,D[k])`. Never contract decay with this FMA;
   never predict from undecayed S; never retrieve from D.
6. Store U as FP32 and compute `R4(q,U)`, then round output to BF16.

Different V rows do not interact. Assigning fewer V rows to each warp, or
placing the four original V tiles in one CTA, changes scheduling but not any
operand or reduction order. Thus equal inputs and initial state imply equal
updated state and raw output, subject to the same device instructions. The
A/B test checks integer views of the values, including signed-zero bits.
For 512/1024 threads, warp 0 computes q/k normalization, decay, and beta
once. A shared-memory broadcast replaces identical calculations in the other
warps; no arithmetic operation or rounding point changes.

### Gated norm

The raw delta output **must** round to BF16 even though it now stays in shared
memory. Norm uses eight consecutive values per lane, 16 lanes per head:
local square order 1,0,2,3,4,5,6,7, then XOR offsets 8,4,2,1. Divide by 128,
add eps, use sqrt.approx/div.full, multiply x*rstd, then weight, then sigmoid.
No reassociation, norm-before-round shortcut, or rsqrt substitution is used.

### Projection boundary

The fused fg path issues eight increasing-K `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`
operations per 16-output tile, accumulates in FP32, and rounds to BF16. The recurrence proof is
conditional on these rounded values: **it does not prove cuBLAS's hidden GEMM
algorithm has the same summation order for every input**.

The diagnostic fg entry allows two separate tests:

1. cuBLAS fg -> mined core vs cuBLAS fg -> new core (strict bit test).
2. new fg -> mined core vs fully fused kernel (strict bit test).

Only then compare cuBLAS fg against new fg and the full trajectories. A
projection difference must not be blamed on the recurrence or silently
accepted. The qkv direct path needs the same independent treatment.

A stability observation is not a tolerance waiver: with fixed q/k/beta,
`S -> (S D)(I-beta*k*k^T)` is non-expansive for normalized k and beta in [0,1],
but changing gates adds fresh errors and decay can approach one. Errors can
accumulate. Long-context state checks and end-to-end token agreement are still
required before approving a non-bitwise projection.

## 5. Measurement record and latency model

### Initial GPU test (before widening)

GPU 0 was idle; kbench was sleeping. One random decode step, bs1, seed 5300,
non-zero conv/SSM state, line indices 600/601. The original 128/256 core and
full paths matched the mined chain's state and output. New fg matched cuBLAS
in this test. This is a smoke test, not a general projection proof.

CUDA graph timings, microseconds per layer, median of five replays, 10 calls
per replay, warm weights. These are **not attributed kern op times**:

| path | bs1 us |
|---|---:|
| original fg_b + conv + delta + norm | 14.12 |
| original conv + delta + norm only | 9.44 |
| core128 | 14.65 |
| core256 | 9.08 |
| fused128 | 22.70 |
| fused256 | 14.23 |
| qkv cuBLAS | 9.30 |
| first direct qkv candidate | 9.71 |

These measurements **reject 128 threads** and do not justify shipping full
fusion at 256 threads. Do not claim the roadmap's 0.35 ms saving from a launch
count alone. Larger CTA measurements are tracked in the artifact report.

The actual cuBLASLt heuristic selected **split-K=3**, 40336 B workspace, for
qkv at bs1 in this test. fg_b selected split-K=1 and no workspace. The captured
sglang recipe instead had split-K=4, grid `[4,21,1]`, plus 105 reducer CTAs.
The serving manifest names an extern, not a fixed projection cubin: inspecting
only the old capture does not establish the current heuristic.

### Final MMA build: measured geometry selection

`mma.json` records the final hash-pinned build. All integer-view cut checks
passed at **bs1/2/4/8/16**, four changing-input steps per core/full geometry
(256/512/1024). Checks cover complete state arenas, conv output, raw delta
output, rstd, and gated output. Full fusion versus cuBLAS fg also had zero
state/output bit differences in these short synthetic trajectories.

The final fg implementation loads MMA operands directly into registers.
It removes zero-padded shared WMMA input matrices and shared output staging.
Eight identical activation columns feed the tensor instruction; only column
zero is retained. This avoids the staging cost and bank-conflict risk of the
first implementation. It does not change the increasing-K accumulation order.

Median graph timings (40 calls/replay, five replays, warm weights), us/layer:

| batch | original middle chain | mined core | new core512 | full512 | full1024 |
|---:|---:|---:|---:|---:|---:|
| 1 | 13.762 | 9.141 | 7.031 | **10.039** | 11.441 |
| 2 | 13.916 | 9.286 | 7.087 | **10.046** | 11.479 |
| 4 | 14.315 | 9.726 | 7.153 | **10.197** | 11.642 |
| 8 | 15.124 | 10.630 | 7.226 | **10.357** | 11.941 |
| 16 | 16.847 | 12.238 | 8.336 | **12.479** | 13.262 |

Select **512**, not 1024. The launch-count target works, but the measured
speed gain is smaller than the roadmap estimate. The direct-qkv candidate
is rejected: it is slower and not bitwise equal (about 10/41/86 us at
bs1/8/16 versus about 9/10/8 us for cuBLAS in this run).

### Model for integration

Use `34*(baseline_chain_us - replacement_chain_us)/1000` ms per step. For
core mode, add the retained two fg GEMMs to the replacement chain. Compare
whole graphs under the same clocks and load; do not add attributed op times.
The measured warm-cache deltas give these **expected, not full-server
measured**, savings across 34 layers:

| batch | conservative core512 | full512 candidate |
|---:|---:|---:|
| 1 | 0.072 ms | **0.127 ms** |
| 4 | 0.087 ms | **0.140 ms** |
| 8 | 0.116 ms | **0.162 ms** |

At bs1, 0.127 ms is about 1.7% of the supplied 7.43 ms graph. It is not a
claim that KDA alone gets kern below sglang. Do not multiply these graph
microbenchmark deltas by the attribution inflation factor.

The supplied five-op attribution sums to 1.773 ms. Dividing by the stated
~1.6 inflation gives ~1.11 ms, about 15% of 7.43 ms. The middle chain alone is
1.137 attributed ms / 1.6 ~= 0.71 ms. This is a **share model**, not a promise
that a microbenchmark gain transfers one-for-one to the full TP8 graph.

The P2 KDA budget (including o_proj but not AR) is 0.62/0.66/0.72 ms at bs1/4/8.
We do not claim that budget is met. qkv still streams ~27.3 MB per rank/layer;
o_proj/AR are outside this change. Warm-cache qkv tests are optimistic for
serving's 34 different weight matrices.

## 6. Build and A/B commands

Run on the bring-up node, in `the kern repo`:

```bash
export PYTHONPATH=tools PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=<nccl + cuda-compat libs for your node>
PY=<sglang-python>  # the sglang venv's python
$PY -m glm53.ops_kda_v2 build
```

Compiler: CUDA 13 nvcc, `-cubin -arch=sm_90a -std=c++17 -O3
-ccbin /usr/bin/g++-14`. No fast-math flag. The build does not initialize a GPU.
`GLM53_KDA_ARTIFACTS` can select another new artifact directory.

Before **each** GPU invocation check `tmux ls`, panes, and `nvidia-smi`. Wait
for kbench, kserve, and integration. The script repeats those checks, rejects
active kern processes, and rejects memory/use on the chosen GPU. It never
stops another job. The check is not a distributed GPU reservation.

```bash
# Short geometry and cut checks.
$PY -m glm53.ops_kda_v2 test --batches 1,2,4,8,16 --threads 512,1024 \
    --steps 8 --repeats 40 --report buckets.json
# Odd groups and every allowed shape.
$PY -m glm53.ops_kda_v2 test --batches 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \
    --threads 512 --steps 4 --report all_groups.json
# Real checkpoint weights; repeat for layers 4 and 44, rank 7.
$PY -m glm53.ops_kda_v2 test --batches 1,4,8 --threads 512 \
    --layer 0 --rank 0 --steps 256 --report real_l0.json
# Zero q/k, saturated beta, and extreme but finite decay logits.
$PY -m glm53.ops_kda_v2 test --batches 1,8 --threads 512 \
    --stress --steps 64 --report stress.json
```

The test compares **all state bytes**, including untouched lines, plus conv
output, raw delta output, rstd, and gated output. It uses the actual mined
cubin ABIs. No `torch` reference reduction substitutes for the oracle.

Executed records:

* `mma.json`: final binary, bs1/2/4/8/16, 256/512/1024 threads, strict cuts pass.
* `real_l0.json`: final binary, layer-0/rank-0 checkpoint weights, bs1/4/8,
  512 threads, 256 changing-input updates per comparison. Conditional core
  cuts passed; final full-fusion state/output also equal the cuBLAS baseline.
  Activations were synthetic, not captured model activations.
* `wide.json`: earlier staged-WMMA build; superseded for performance.
* `ab.json` / `first.log`: initial bs1 smoke test.
* `build.json`, `build.log`, `pdl.txt`: final source/binary hashes, resources,
  and PDL opcode audit.

Several attempts were deferred before CUDA initialization because other
bench/serve jobs held the GPUs. A later free window allowed `real_l0.json`
to complete. The all-groups sweep was then deferred by another active job.
No claim is made for captured activation trajectories, other layers/ranks,
saturated-gate stress, sanitizer runs, or a full serving graph. The supplied
commands cover these remaining checks. The full-versus-cuBLAS result reports
the final state/output after 256 updates; it is not an all-token model oracle.

Remaining owner acceptance gates:

1. Repeat at real probe activations, zero/fresh leases, reused/forked/restored
   slots, line-table layers 0/17/33, and maximum practical slot indices.
2. Compute Sanitizer memcheck/racecheck on short isolated tests. Do not run
   long sanitizer workloads on the shared box without a free window.
3. Kern graph replay with PDL on/off, bs1/2/4/8/16, repeated calls to every
   KDA layer. Compare state snapshots and token sequences to the baseline.
4. Original oracle prompts plus long context (including branch/restore), all
   eight ranks. Do not approve projection drift solely from one-step RMS error.
5. Measure full decode graphs under the same AR/MoE/mHC configuration. Report
   shares and end-to-end step time, not the sum of attributed op durations.

## 7. Exact owner integration hook

Do **not** replace the original `ops_kda.ops()` import in the collection loop:
that would disturb the existing op-count assertion. Add this import:

```python
from glm53 import ops_kda_v2
```

Then change only this part of `gen.build`:

```python
    resolve_modules(allops)
    # New ops already contain absolute paths and pinned hashes. Insert AFTER
    # resolve_modules: its two-root search counts an absolute path twice.
    ops_kda_v2.fuse_manifest(m, mode="core", threads=512, direct=False)
    lower_wire(m)
    return normalize(m)
```

Select `mode="fused", threads=512` for the full candidate after its
acceptance gate. Core512 is the conservative first integration step.
Keep `direct=False` unless the qkv candidate wins both the speed and numerics
gates. The helper rewrites only the contiguous fg_b/conv/delta/norm call group
and validates the buffer identities. It retains existing o_proj and AR calls.
In core mode it leaves fg_b in place. Existing workspace buffers can remain.
If internal probes interrupt the four calls, or a surviving call consumes a
removed intermediate, the helper raises instead of silently deleting it.

One-layer core/fused builds at 512/1024 threads and a **45-layer** fused
build were checked **in memory**, without writing a manifest. All 34 table
offsets were checked to equal `k*64` bytes. The new op, original fg_b, o_proj, and separate Lamport AR
were wired as intended. Full source integration remains the owner's task.

Use new outputs when integrating, for example:

```bash
$PY tools/glm53/gen.py --allreduce lamport \
  --out examples/glm53-flash-kda-v2.json --bundle kernels-glm53-kda-v2
```

The bundle command copies the hash-pinned cubin out of `/tmp`; do this before
removing the build directory or deploying elsewhere. Do not overwrite the
existing working manifest or shared kernel bundles.

## 8. Risks and explicit non-claims

* Too few CTAs at bs1; fusion can lose despite fewer launches. Measure geometry.
* These timings reuse warm state/weights. Full-model cold state/weight traffic
  can reduce the saving or cause regression. Confirm with the owner
  integration benchmark; do not deploy from the warm-cache estimate alone.
  A useful next experiment is to prefetch multiple independent V rows per
  warp before prediction, to raise memory-level parallelism without changing
  the per-V reduction tree.
* MMA/cuBLAS fg rounding and direct-qkv reduction order are separate risks.
* The direct projection is not yet a qualified split-K replacement.
* Same-input recurrence parity does not establish full-model token parity.
* Graph-safe code and PDL opcodes do not prove the complete runtime edge plan.
* Invalid varlen/negative-line guards return without outputs; only the stated
  decode contract is supported. This is not a general varlen kernel.
* Scratch is not assumed zero per call; there is **no** persistent scratch.
* No MTP no-store mode, state migration, o_proj fusion, or allreduce changes.
* `/tmp` build products are not a durable deployment bundle.
EOF'