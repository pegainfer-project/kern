# GLM-5.3 decode mHC fusion

Date: 2026-09-26. Scope: TP8, H100/H800 sm_90a, decode groups 1..16,
rows=1, hidden=4096, four residual streams. Existing sources and manifests
are unchanged. Integration is opt-in.

## Status

- Built `kernels-glm53-handwritten/glm53_mhc_v2.cubin` with CUDA 13.0,
  GCC 14, C++17, `-O3`, no fast math. Four entries, no spills.
- Cubin SHA256: `0eebff95d0038dea8624e9351446ba80d552fe5c1e16991906414c691aef8098`.
- Source/ELF ABI checks pass for all four entries and the mined prenorm,
  fma, big_fuse64/8, post, row/column quant references.
- All four 45-layer raw-manifest transforms pass `kern verify`: quant-only,
  boundary, cross-layer, cross-layer with BF16 fn storage.
- CPU checkpoint audit: all **90** original BF16 `hc_{attn,ffn}_fn` tensors
  exactly widen to the existing derived FP32 tensors. Both shapes are
  `[24,16384]`. No approximate weight conversion is needed.
- GPU validation: **NOT RUN**. The guard declined active kbench and kserve
  jobs; a bounded wait found no test window. Sglang then occupied the GPUs
  for integration work. No CUDA context was created by this task. Do not
  promote these candidates until GPU A/B and full-model checks pass.

## 1. Design

The actual graph is:

```
prenorm64 -> fuse64+norm -> attention+AR -> fma8 -> fuse8+norm
          -> dense/MoE+AR -> post -> next-layer prenorm64 ...
```

`big_fuse64` and `big_fuse8` are **different boundaries**, not consecutive
stages at the same boundary. Attention and the MLP cannot be crossed.

### Candidates

1. **Quant-only:** retain current prenorm/FMA, add FP8+SFA to the generated
   big-fuse norm epilogue. No reduction changes.
2. **Boundary:** run the current FMA decomposition, then let the last CTA
   for each row run big_fuse8, including norm and optional quant. This
   replaces the existing fma+fuse8 pair with one kernel.
3. **Cross-layer (experimental):** use the same boundary entry for the
   previous layer's FFN post and the **next layer's attention fn**. It
   replaces post+prenorm64+fuse64. The first prenorm+fuse64 and final post
   remain. This changes the prenorm reduction tree and needs a separate
   tolerance/model-quality gate.
4. **BF16 fn storage:** a separate boundary entry widens each fn element
   to FP32 before the unchanged multiply/accumulate. Only boundary-used
   binds change; the first DeepGEMM prenorm retains its FP32 bind.

A persistent one-CTA-per-row dot product was not selected. At bs1 it
would read a 1.57 MB FP32 fn on one SM. The 96-CTA decomposition has the
same weight-parallelism as the current FMA.

### Last-CTA protocol

The grid is `(T,12,8)`: two mix outputs per CTA, eight hidden splits. Each
CTA writes compact partials. Y=0 also writes the rounded BF16 residual
and square sum. Every thread fences its writes; a CTA barrier precedes
one leader `atom.acq_rel.gpu.global.add.u32` to a per-row counter. The
96th arrival continues; other CTAs leave. The acquire/release RMW chain
publishes all partials and residual writes to the winner.

In the winning CTA only threads 0..95 execute the original big-fuse
body. Barrier 4 synchronizes these 96 threads. The norm's existing
64-thread barriers retain IDs 1/2/3. CTA-wide barriers use ID 0. The other
160 threads wait at the full-CTA barrier, not at a partial barrier.

The winner resets its row counter to zero after all outputs are complete.
Thus scratch is zeroed at allocation only; replay needs no host read or
memset. Counters are separate for each row and op scratch instance.
Previous and next post/comb buffers may alias because all 96 CTAs have
consumed the previous row's mixes before the winner replaces them.
Residual input/output **must not alias**. Concurrent invocations must
not share this scratch (the normal kern same-stream decode does not).

There is no spinning, cooperative launch, or grid-residency assumption.
This avoids the deadlock risk of a resident-grid barrier. The cost is
counter contention, fences, and larger shared memory per FMA CTA.

### Quant epilogue

The generated norm epilogue owns eight adjacent BF16 outputs per thread.
Sixteen lanes cover one 128-value quant group. Max reduction is exact for
finite inputs. Quantization consumes the **rounded BF16 output**, not the
pre-rounding FP32 norm value:

```
amax = max(max(abs(bf16_output[128])), 1e-10)
SFA = amax * (1.f / 448.f)
quant_multiplier = 448.f / amax
FP8 = e4m3_satfinite(min(float(bf16_output) * quant_multiplier, 448.f))
```

This is the non-UE8M0 path used by module_233/333. Do not replace it with
power-of-two scales. Runtime `layout` is 0=no quant, 1=column, 2=row:

- Q: byte offset `row*4096+h` in both cases.
- DSA/dense: FP32 SFA offset `group*16+row`; static pitch is **16**, not T.
- MoE: FP32 SFA offset `row*32+group`.
- BF16 x_norm remains for KDA, DSA side projections and the MoE router.

**Count correction:** the current generator has 11 DSA x-input quants,
3 dense x-input quants and 42 MoE x-input quants: **56**, not 87. The 34
KDA attention input projections use BF16 and have no input quant to remove.
Other DSA qa/o/indexer quants and MLP intermediate quants stay unchanged.
Do not claim savings for those, or for the separate head RMS norms.

## 2. Kernel table and ABI

All cubin pointers are 64-bit; scalar args are 32-bit. The generator uses
semantic parameter order; `ops_mhc_v2.py` explicitly permutes launch args.
The new CUDA entries are not subject to TileLang alphabetical sorting.

| Entry | Block | Grid | CTAs bs1 / bs8 / bs16 | Dynamic shared | Registers |
|---|---|---|---|---|---|
| `glm53_mhc_fuse64_q` | 96,1,1 | T,1,1 | 1 / 8 / 16 | 37,232 B | 72 |
| `glm53_mhc_fuse8_q` | 96,1,1 | T,1,1 | 1 / 8 / 16 | 37,232 B | 70 |
| `glm53_mhc_boundary_f32` | 256,1,1 | T,12,8 | 96 / 768 / 1536 | 37,232 B | 64 |
| `glm53_mhc_boundary_bf16` | 256,1,1 | T,12,8 | 96 / 768 / 1536 | 37,232 B | 64 |
| retained mined first prenorm | 256,1,1 | 64,1,1 | 64 / 64 / 64 | 232,448 B | unchanged |
| retained mined final post | 128,1,1 | T,1,1 | 1 / 8 / 16 | 28,672 B | unchanged |

The boundary has an additional small static shared allocation, padded by
ptxas to 1,024 B. Register counts are from this build, not an estimate.
FMA partials remain `[8,T,24]` / `[8,T]` compact in max-sized op scratch.
For a standalone fuse64, the scalar partial pitch stays **16**, even
though grid.x is the active bucket T. For fuse8 it is T.

**Fuse entries: 13 args**

```
comb*, mul*, sqr*, base*, scale*, x_norm*, norm_weight*, post*, residual*,
int partial_pitch, q*, sf*, int layout
```

First nine pointer offsets: 0,8,...,64; pitch@72; q@80; sf@88; layout@96.

**Boundary entries: 19 args**

```
residual_out*, hidden*, mul_scratch*, fn*, prev_comb*, prev_post*,
prev_residual*, sqr_scratch*, int T,
base*, scale*, norm_weight*, comb_out*, post_out*, x_norm*, q*, sf*,
count_scratch*, int layout
```

First eight pointer offsets: 0,8,...,56; T@64; next nine pointers@72..136;
layout@144. Split count is fixed at 8; **do not launch a different Z grid**.

Every entry starts with `cudaGridDependencySynchronize()` and triggers
programmatic launch completion only after its own required work. The
boundary winner includes the counter reset before its trigger. SASS has
`ACQBULK` and tail `PREEXIT`. Op launch flags are conservatively non-PDL
until integration A/B; no early-release performance benefit is claimed.

## 3. Numerics

### Preserved operations

Generated TileLang sources are SHA-pinned by `glm53_mhc_build.py`. The
64- and 8-split big-fuse sources differ only in split count. The build
copies the scalar arithmetic, reductions, norm BF16 cuts, and Sinkhorn
body. It uses a small C++17-compatible copy of the required scalar
TileLang helpers, not a new mathematical implementation.

- FMA hidden iteration and route order are unchanged; warp sum uses XOR
  offsets 16,8,4,2,1, then warp partials accumulate serially 0..7.
- Big-fuse sums splits serially in increasing split index.
- RMS epsilon is 1e-5, pre-mix epsilon and Sinkhorn epsilon are 1e-6.
- Post multiplier is 2.
- Residual after post is rounded to BF16 **before** both fn and square sum.
- Pre-mix square sum uses its FP32 values; the mixed activation is rounded
  to BF16 before the final norm multiplication, exactly as in the oracle.
- Default nvcc `expf`, division, FMA contraction and rsqrt are retained;
  **no `--use_fast_math`**.

### Sinkhorn: fixed 20 passes, no stopping test

1. Row max, then `expf(cm-row_max)`.
2. Initial row normalization is `cm/row_sum + 1e-6` (epsilon **outside**
   the division, and no epsilon in this initial denominator).
3. Initial column normalization is `cm/(col_sum+1e-6)`.
4. Exactly **19** more row-then-column normalizations, each denominator
   has `+1e-6`. There is no convergence test or early exit.

The 4-value row and column sum butterfly orders are kept from TileLang.
Changing the epsilon placement or using 20 iterations after the initial
pass would implement a different model.

### BF16 fn argument and limits

A BF16 finite normal value has at most 8 significand bits. The product of
two such values needs at most 16 bits, within FP32's 24-bit significand,
provided the product does not overflow/underflow. The checkpoint fn is
already BF16; derived FP32 is its exact widening, confirmed for all 90
tensors. The BF16 boundary widens before multiplication and preserves
FP32 accumulation order. Thus it does not introduce weight approximation.
This is not a claim that BF16 accumulation, a changed reduction tree,
or arbitrary extreme/subnormal inputs are exact.

With only FFN boundaries changed, fn bytes saved are 35,389,440/step/rank.
With cross-layer fusion, 89 of 90 fn reads can use BF16: save 69,992,448 B
from the 141,557,760 B FP32 stream. The first fn remains FP32.

### Cross-layer deviation

Post+DeepGEMM64 and FMA8 use different dot/square-sum reductions, even
though the BF16 operands and products are exact. FP32 sum rounding is not
associative. Do not infer bit parity from the BF16 storage argument.
A/B must compare the **mined post+prenorm+fuse64** on the same inputs, not
only an eager PyTorch matmul. The supplied `--cross-layer` mode does this.

## 4. Build, validation, and rollout

Build (CPU only):

```sh
cd <kern-repo>
PYTHONDONTWRITEBYTECODE=1 python3 tools/glm53/kernels/glm53_mhc_build.py
PYTHONDONTWRITEBYTECODE=1 python3 tools/glm53/kernels/glm53_mhc_test.py --cpu
```

The generated CU is standalone. If the TileLang cache is gone,
build it directly:

```sh
/usr/local/cuda-13.0/bin/nvcc -cubin -arch=sm_90a -std=c++17 -O3 \
  -ccbin /usr/bin/g++-14 --ptxas-options=-v,--register-usage-level=10 \
  tools/glm53/kernels/glm53_mhc_v2.cu \
  -o kernels-glm53-handwritten/glm53_mhc_v2.cubin
```

Do **not** source `kserve_prod.sh` (it starts a server). Copy only its library
path. Check tmux and nvidia-smi first; the script also checks before CUDA:

```sh
export LD_LIBRARY_PATH=<nccl + cuda-compat libs for your node>
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \
  <sglang-python> \
  tools/glm53/kernels/glm53_mhc_test.py --cross-layer --real-layer 0
# For same-order strict checks, omit --cross-layer and add --strict.
# Also run --zero, --seed 124, and --batches 3,5,6,7,9,10,11,12,13,14,15.
# --summary omits exact-match tensor metrics, but retains all deviations.
```

The harness loads the actual cubins with the CUDA driver. It checks:

- residual, post, comb, x_norm, e4m3 values, FP32 SFA; finite values,
  bitwise element mismatch count, value mismatch count, max absolute error
  and RMS-relative error;
- no-quant, row and column SFA layouts;
- FP32/BF16 fn; compact T pitches; standalone fuse64 and fuse8;
- aliased previous/next mix buffers; zero counters after each call;
- repeated graph replay, including changed hidden input at the same pointer;
  no per-call counter zeroing;
- same-input mined cross-layer path including all three TMA descriptors;
- graph microtiming (not host launch/event-per-op attribution).

The script always fails on non-finite outputs or nonzero counters.
`--strict` also fails on any bit mismatch outside the cross-layer experiment.
Without `--strict`, a zero exit code is not an exact-parity claim.

Rollout gates:

1. Existing-boundary and standalone epilogue: seek exact equality. Any
   difference must be reported; no tolerance is silently substituted.
2. Cross-layer: report all BF16/FP32 cuts, then assess small-error candidates
   against 45-layer hidden/logits and greedy tokens. A suggested screening
   limit is x_norm RMS-relative <=1e-3 and mix absolute error <=1e-4;
   this is **not** a model-parity approval.
3. Full TP8 AR-manifest A/B at bs1/2/4/8 (plus bucket16), same inputs,
   with short GPU runs and no competing jobs. Verify greedy agreement
   >=99.5% over a controlled 1k-token trace before promotion.
4. Repeat with PDL launch flags; retain non-PDL fallback. Use memory/race
   checking in a separate quiet window. The last-CTA alias and replay
   tests do not replace a race checker.

## 5. Exact integration hook (owner applies)

In `tools/glm53/gen.py`, add `ops_mhc_v2` to the existing `from glm53 import`
line. Do **not** add it to the op-module loop (it would break the fixed
op-count assertion and leave unused ops). In `build()`, use exactly:

```python
    wire_allreduce(m, allreduce, pdl=ar_pdl)
    ops_mhc_v2.fuse_manifest(m, boundary=True, cross_layer=False, bf16=False)
    check_invariants(m, layers)
    resolve_modules(allops)
    lower_wire(m)
```

The function edits the raw in-memory call plan, inserts only used ops,
removes unused part8 workspace, and leaves allreduce untouched. `allops`
and `m['ops']` are the same dict. No manifest file is changed by import.

Selectable, separately gated paths:

```python
# Lowest arithmetic risk: just absorb the 56 input quants.
fuse_manifest(m, boundary=False)
# Existing attention-post -> FFN-pre boundary; same reduction order.
fuse_manifest(m, boundary=True)
# Add cross-layer boundaries; changed DeepGEMM reduction order.
fuse_manifest(m, boundary=True, cross_layer=True)
# Same cross-layer candidate, exact original BF16 weight binds.
fuse_manifest(m, boundary=True, cross_layer=True, bf16=True)
```

The exact semantic FFN call replacing `fma` and `fuse8` is:

```python
step(label + 'ffn_boundary', 'hc_boundary_f32_v2',
     b('mix_comb'), b('R_a'), b('mix_post'), b('sub_out'),
     w('hc_ffn_fn'), w('hc_ffn_base'), w('hc_ffn_scale'), w('post_attn_ln'),
     b('R_b'), b('mix_comb'), b('mix_post'), b('x_norm'), b('a8'),
     b('sfa' if layer < 3 else 'moe_as'), TV, i32(1 if layer < 3 else 2))
```

Remove that layer's `mlp_quant_a`/`moe_quant_a` call, but not `quant_b`.
The adapter also replaces DSA `fuse64` with the quant epilogue and removes
`dsa_quant_qkv`; it does not remove `dsa_quant_q` or `dsa_quant_o`.

Cross-layer mode passes **next-layer** `hc_attn_fn/base/scale/input_ln`
and writes R_a from R_b. It uses layout1 only before DSA, layout0 before
KDA. Final post remains. Do not accidentally use the current FFN fn twice.

Apply this before independent MoE/DSA transforms. A MoE v2 that already
quantizes internally needs an explicit prequantized-input interface before
it can consume these q/SFA outputs; do not double-count quant savings.
Use `quant=False` when testing boundary fusion against such an integration.
The adapter rejects probes inside fused regions and already-v2 call plans
rather than silently deleting probe cuts or incompatible quant work.

## 6. Cost model and risks

Current mHC: 225 launches. Existing-boundary candidate: 180. Cross-layer:
92 = first prenorm+fuse (2), 45 FFN boundaries, 44 inter-layer boundaries,
and final post (1). Each candidate can also remove 56 input quant launches.
The checked full manifest has 1184 original logical calls; quant-only
1128, boundary 1083, cross-layer 995. Logical calls are not always launches.

Planning targets, **not measured end-to-end savings**:

| Candidate | bs1 | bs4 | bs8 |
|---|---:|---:|---:|
| quant-only | 0.05–0.11 ms | 0.05–0.11 ms | 0.05–0.11 ms |
| plus existing-boundary | 0.10–0.22 ms | 0.10–0.22 ms | 0.08–0.20 ms |
| plus cross-layer and BF16 fn | 0.25–0.45 ms | 0.23–0.43 ms | 0.20–0.40 ms |

These assume about 1–2 us net saved per removed boundary after the
last-CTA overhead, and at most ~0.05 ms for fn bandwidth. They can be
invalidated by counter contention or shared-memory occupancy. Use the
supplied graph A/B to replace them, then measure the whole TP8 step.
Do not subtract 1.6x-inflated per-op attribution directly from 7.43 ms.
This optimization alone cannot reach <5.68 ms from 7.43 ms.

Main risks: last-CTA contention and shared-memory resource cost, cross-layer
sum-order drift accumulating through 45 layers, norm/quant compiler
lowering, global q/SFA lifetimes when other agents fuse consumers,
scratch alias/concurrent invocation, untested PDL overlap, and measurement
bias from hot fn data versus the full model's streaming weight working set.
