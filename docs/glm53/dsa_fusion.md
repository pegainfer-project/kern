# DSA decode fusion v2

Date: 2026-09-26. Host: **the bring-up node only**. Model: GLM-5.3-Flash,
TP8, H100 sm_90a. Decode: groups 1..16, one row per group.

## Status and release gate

**Direct-cubin GPU numerics passed: 730 checks, zero failures at
S=1/2/3/4/5/8/15/16.** Offline ABI and manifest checks also pass. This clears
the direct-kernel numerics gate, not the full-model gate. Model cuts, TP8
graph/greedy parity, and per-bucket performance remain owner integration
steps. No measured speedup is claimed by this unit-test work.

Created:

- `tools/glm53/ops_dsa_v2.py` — drop-in absorb ops, optional manifest fusion,
  ABI audit, in-memory CPU manifest checks, and direct-cubin GPU A/B script.
- `tools/glm53/kernels/glm53_dsa_v2.cu` — seven CUDA entries.
- `tools/glm53/kernels/glm53_dsa_topk_v2.cu` — deterministic top-k and slot mapping.
- `kernels-glm53-handwritten/glm53_dsa_v2.cubin`
- `kernels-glm53-handwritten/glm53_dsa_topk_v2.cubin`
- This document.

No existing source, manifest, checkpoint, service, or agent file was changed.
`gen.py` integration remains the owner's task.

Completed checks:

1. Both cubins compile with the requested CUDA-13 / g++-14 command.
2. `cuobjdump -elf`: all eight new entries match exact parameter offsets and
   sizes. All 28 cubin launches in the inherited plus new DSA op set match
   ordinal/size lists, including FA3's 2944/232-byte packs and Triton pointers.
   Remaining extern interfaces are inherited, unchanged.
3. `cuobjdump -sass`: every new entry has `ACQBULK` and `PREEXIT`.
4. All four `(xside, topk)` combinations build in memory and pass the Rust
   `kern verify /dev/stdin` check: schema v5, `seqs <= 16`, `tokens <= 16`,
   decode one row/group, page table `[seqs,1024]`.
5. All new entries have **zero spill loads/stores and zero stack bytes**.
6. The full manifest has 1,074 calls / 1,142 declared launch entries. The
   inspected baseline has 1,184 / 1,395. Thus 253 launch entries are removed.
   Extern calls can launch more than one CUDA kernel, so this is not a profiler
   count and does not include other agents' pending fusions.

GPU test history (short runs on GPU 0 only):

- Tests waited for TP8 services/benchmarks. GPU 7's unrelated allocation was
  never touched. No service was stopped.
- First run found a **harness alias**: at S=1, a narrowed kg row is already
  contiguous, so `.contiguous()` did not copy it before the baseline
  in-place norm. The harness now uses `.clone()`.
- Next free-window run passed all glue/state checks at S=1 and absorb plus
  graph replay at S=2. Absorb relative L2 errors: K/V at S=1 = 0 / 4.76e-7;
  S=2 = 2.62e-9 / 1.68e-7 (random unit inputs, Torch mm reference).
- That run found four QA8 code mismatches at S=2, despite exact BF16 QA.
  SASS inspection proved that pinned `module_233` quantization uses
  **MUFU.RCP + FMUL.FTZ**, whereas ordinary CUDA division adds reciprocal
  refinement. The final source now mirrors the approximate reciprocal and
  FTZ multiplies with explicit PTX, without enabling global fast math.
  The rebuilt SASS was checked against `module_233`.
- The owner reran the final cubins: the Q reciprocal/FTZ fix passed, but
  Hadamard FP8 failed at S=2 with 4096 mismatches. This was reproduced on
  GPU 0 before the owner's next TP8 benchmark started.
- Source/TTIR inspection found a **reference launch extent error**, not a
  proposed Hadamard arithmetic change. `module_367` quantizes flattened
  `[S*32,128]` head rows. Its output masks are
  `pid_x*32 + arange(32) < M`. The old test passed `M=32` with grid S, so
  all token rows after the first were unwritten in `torch.empty` reference
  buffers. The fixed test passes **`M=S*32`**.
- The repaired harness initializes FP8/scales with reserved sentinels,
  checks every active output and a guard row, compares a batched reference
  against S rebased one-token reference launches, and includes the old
  M=32 launch as a negative-control coverage test at S=2. It adds impulse,
  zero, constant, alternating-sign and small/large-amplitude Hadamard inputs.
  Numerical failures are collected through the full matrix and give a
  nonzero final exit; CUDA faults still abort immediately.
- Both cubins were rebuilt with the documented nvcc command during the GPU
  pause. Arithmetic and artifact hashes are unchanged. All eight new ABI
  checks, 28 inherited/new launch checks, PDL checks and four CPU manifest
  builds and four Rust `kern verify /dev/stdin` checks pass again.
- **Owner action:** the inherited `ops_dsa.py:dsa_act_quant` launch also
  passes M=32. It has not been changed here, nor overridden in absorb-only
  `ops()`. Stage A is intentionally unchanged during integration. For an
  unfused multi-sequence baseline, its fourth launch argument must be
  `expr(mul(S, IDX_HEADS))`, not `i32(IDX_HEADS)`. Both helpers are already
  imported there. This is separate from Stage A performance work. Above
  512 pools, invalid query bytes/scales can affect real selection for rows
  after the first; identity selection at <=512 pools can hide the error.
  Stage B replaces this launch and already dispatches all S*32 heads.
- An independent CPU mirror of the radix cutoff and packed scan passed 18
  signed/tied/all-equal cases at 513..65535 pools. This is not a GPU test.

### Follow-up completion: 2026-09-26, GPU 0

The run started only after the armd bench log contained
`EXIT=0`, tmux `kbench`/`kbuild` were sleeping, and GPU 0 showed 0 MiB / 0%.
GPU 7's unrelated allocation remained untouched. The test returned
`DSA_TEST_EXIT=0`; no GPU test process was left running.

Command (same venv and CUDA-13 compatibility library path as the owner):

```bash
cd <kern-repo>
export LD_LIBRARY_PATH=<nccl + cuda-compat libs for your node>
PYTHONPATH=tools <sglang-python> -u -B \
  -m glm53.ops_dsa_v2 --test --gpu 0
```

Seed 5303. `exact` below means zero differing values/FP8 codes/state bytes
against the pinned or same-input reference; it does not mean full-model
bitwise parity. Absorb uses the original unchanged 1e-3 relative-L2 gate.

| S | absorb KC rel L2 | absorb VC rel L2 | norm / QA8 / SFA / KV | Hadamard FP8 + scales | step + MQA | wk/gate + head | state epilogue | top-k |
|---:|---:|---:|---|---|---|---|---|---|
| 1 | 0 | 4.7612e-7 | exact | exact | exact | PASS | exact | exact |
| 2 | 2.6233e-9 | 1.6848e-7 | exact | exact | exact | PASS | exact | exact |
| 3 | 2.7402e-7 | 5.4888e-7 | exact | exact | exact | PASS | exact | exact |
| 4 | 1.8768e-6 | 9.6314e-7 | exact | exact | exact | PASS | exact | exact |
| 5 | 3.4122e-9 | 2.7369e-5 | exact | exact | exact | PASS | exact | exact |
| 8 | 5.5858e-6 | 6.1399e-5 | exact | exact | exact | PASS | exact | exact |
| 15 | 1.1198e-5 | 4.9060e-7 | exact | exact | exact | PASS | exact | exact |
| 16 | 2.4195e-5 | 6.0993e-5 | exact | exact | exact | PASS | exact | exact |

- **730 checks / 64 component-batch cases / zero failures.** Each batch has
  3 KC, 3 VC, 4 norm/quant/KV, 12 Hadamard, 4 step/MQA, 3 xproj, 15 state,
  and 47 top-k checks; S=2 has two additional negative-control checks.
  Numerical failures now accumulate to the final failing exit, rather than
  hiding later cases. CUDA errors still stop execution.
- Absorb guard rows and captured replay after input mutation pass at every
  S. Worst relative L2 is **6.14e-5**; minimum exact fraction is **99.9512%**.
  Eight Torch mm calls are the reference, not a claim of bitwise matching
  kern's chosen cuBLASLt algorithm on model activations.
- Hadamard random and structured/amplitude-varied inputs pass exactly,
  including guards, all-active-output sentinel checks, and batched versus
  per-token references. The old M=32 negative control leaves exactly
  **4096 active FP8 bytes and 32 active scales unwritten at S=2**. This
  confirms the reported failure was an invalid reference extent, not FTZ,
  scale layout, or a sign/permutation error in the fused kernel.
- wk/gate pass the unchanged tolerance at every S, **not bitwise at every
  S**. Worst wk/gate relative L2 is 3.4984e-4 (wk, S=15); worst absolute
  difference is 1.0 on these random dot products. Head projection/scaling
  and the isolated same-input state epilogue are exact. Changed wk/gate
  rounding can still affect near-tie indexer scores in full-model tests.
- Top-k passes all seven pool lengths **0,1,511,512,513,35280,65535** at all
  eight S values, for random/tied scores, +/-infinity, identity with NaN
  logits, nonidentity page maps, tail lengths, all-equal ties, output guards
  and zero padding. Three repeats are identical. Captured replay after
  in-place changes passes identity at 512 and **real selection at 513**,
  selecting high-score pool 512 rather than silently retaining identity.
- **Verdicts:** Stage A direct absorb gate PASS. Stage B direct norm/store,
  Hadamard and metadata gate PASS. Optional x-side and mapped top-k direct
  gates PASS. The previously blocked direct-numerics step is complete.
  Owner integration still must fix the inherited M extent for the unfused
  comparison, then run model-cut / greedy / clean full-graph A/B gates.
  No tolerance was relaxed and no selector threshold was changed.

### Pinned Hadamard reference ABI evidence

`module_365.cubin` (FHT): one 56-byte by-value pack, block 16, grid 32*S,
512 dynamic shared bytes. Fields are batch/dim/log_N at 0/4/8, int64 input
and output row strides at 16/24, float normalization at 32, and pointers
at 40/48. Pack format `<iiiiqqfIQQ` includes the padding at 12 and 36.

`module_367.cubin` (`_act_quant_kernel`): three pointers at 0/8/16,
i32 M/N at 24/28, two hidden i64 slots at 32/40, block 128, grid S,
128 dynamic shared bytes. **M=32*S and N=128.** The pinned TTIR uses
`rows = pid_x*32 + arange(32)` and `rows < M` for both Y and scale stores.
The cache directory is
`$HOME/.cache/sglang/triton/O3XFLJ3FJBRUHZRTWO3ULNVWHXG5SB7LTGGMOYOPXQ2A5ZK235RQ/`.

```text
module_365.cubin SHA256
6196060c189bda8ef99534825a4a367ef5cd73bc7a2ba83c4a4eb5d618a68c05
module_367.cubin SHA256
dcbcc1cc151d5e98946d06ed62cb63c5f29aa61f2d38b16fa8ce306eb1106f78
```

## 1. Design

### 1.1 Grouped absorb projections — first integration stage

Replace eight per-head cuBLASLt extern GEMMs by one CUDA entry for each BMM.
Keep the existing op signatures and weight storage:

- K absorb: A `[S,8,256]`, W `[8,512,256]`, output `[S,8,512]`.
- V absorb: A `[S,8,512]`, W `[8,256,512]`, output `[S,8,256]`.

A CTA has four warps. A warp computes four output columns and up to four
sequence rows. Each loaded weight is reused across the four rows. FP32 FMA
partials are reduced in a fixed lane tree, then rounded once to BF16. There
is no split-K, output atomic, transpose, weight conversion, or scratch.
Rows beyond S in the last tile are neither read nor written. This removes
14 extern calls/layer, or 154 across 11 DSA layers.

This is a SIMT grouped GEMV, not a tensor-core GEMM. Its speed at S=8/16 is a
measurement gate, not an assumption. Keep the old extern path for rollback.

### 1.2 RMSNorm + Q quant + latent KV store

One 1024-thread CTA per row reads QKV at **2048 BF16 elements/row**.
It uses the baseline 1024-thread reduction tree for both Q and KV norms.
The Q result is rounded to BF16, written to `qa`, and then quantized from
that rounded value. KV is rounded to BF16 and stored directly to its slot.

- Q norm: 1536 elements, epsilon 1e-5.
- KV norm: 512 elements at QKV column 1536, epsilon 1e-5.
- Q quant: non-ue8m0, groups of 128, floor absmax 1e-10,
  dequant scale `amax*(1/448)`, quant multiplier
  `rcp.approx.ftz(amax)*448`, followed by FTZ input multiplication.
  This is the pinned cubin's fast-math lowering; plain rounded division
  changed four FP8 midpoint ties in the first S=2 test.
- SFA: FP32 `[12,16]`, address `group*16 + row`.
- QA8 is compact `[S,1536]` in the existing `a8` allocation.
- KV byte address: layer-adjusted base + `int64(slot)*11264`.

Explicit rounded division/add operations keep a compile-time divisor from
introducing a fused multiply-add absent in the baseline dynamic-d kernel.
No `knope` workspace is needed. The helper removes its declaration only if
no program still uses it. Padding-row KV behavior remains the baseline
behavior; this kernel does not use slot zero as a validity test.

### 1.3 X-side projection and epilogue

The proposed physical BF16 concat N=288 needs one refinement: the last 32
head outputs must stay **FP32**, not be rounded as BF16 GEMM output. Also,
the existing weight ABI already stores their weights as FP32.

The implementation uses a **logical concatenation** with the same three
weight pointers: BF16 wk (128), BF16 gate (128), FP32 head weights (32).
One grouped projection entry computes all 288 outputs from the same x,
reusing weights across up to four rows. It emits BF16 `[16,256]` kg scratch
and FP32 `[16,32]` head scratch. This avoids a loader change, preserves full
FP32 head weights even if they are not BF16-representable, and avoids an
extra head-output rounding. It is not a claimed tensor-core N=288 GEMM.

A second entry fuses:

- 128-element k LayerNorm, epsilon 1e-6, FP32 norm weight and bias;
- BF16 key rounding before tail storage/compression;
- head-weight scaling in the original multiply order;
- tail-ring update and optional kpool close/compress/rotate/quant/store.

The two entries belong to **one op**, so their scratch is shared correctly.
The helper places this op after the fused Hadamard/quant so q scales exist.
The first entry's head-dot reduction matches the existing warp-per-head
`glm53_weights_proj`; wk and gate use a different reduction from cuBLASLt.

State semantics are copied from the fixed `glm53_dsa.cu`:

- `pos & 3 == 3` closes a pool. The ring is still `pos & 7`.
- `valid[row]`, not slot-zero, controls idx/tail writes.
- Tail line stride 4096 bytes, score array at +2048.
- Layer-adjusted idx base, page stride 92928, layer stride 8448,
  scale array at +8192.
- Page mapping is `bt[pool >> 6]`; each page covers 256 tokens.
- Pool softmax uses the existing `ex2.approx` / `div.full` lowering and BF16
  boundaries. No rewrite of the accepted prefill attention path is made.

### 1.4 Hadamard + activation quant

One 128-thread CTA per indexer head executes seven increasing-stride
butterflies. It preserves the BF16 rounding after normalized Hadamard and
before absmax. Quantization uses the baseline 1e-4 absmax floor and ue8m0
power-of-two scale. Output remains E4M3 `[S,32,128]` and FP32 `[S,32]`.
The kernel removes the `iqh` write/read and one launch/layer. Its grid is
32*S; the equivalent unfused Triton quantizer must receive M=32*S. A constant
M=32 masks all tokens after the first and is not a valid multi-token oracle.

### 1.5 Per-step preparation and metadata

One warp computes pool lengths, clamped MQA lengths, FA3 attention lengths,
and the `[133,2]` DeepGEMM schedule. The schedule is a direct specialization
of `sm90_paged_mqa_logits_metadata<32,256,132,false>` for next_n=1. It keeps
reversed SM allocation and the final `{S,0}` sentinel. S must be <=16 here.

This removes ten repeated MQA metadata calls. When the mapped top-k is
selected it also removes the slot-table build. The baseline FA3 prepare /
forward / combine op remains intact; its scheduler is **not** hoisted.
Per-sequence FA3 `nsd`, `nmb`, `vbi` scratch arrays are expanded from 8 to 16
entries. Semaphore storage, partial strides, and both parameter packs stay
unchanged.

### 1.6 Deterministic top-k with direct block-table mapping

This is a new single-CTA radix implementation rather than an atomic-output
copy of the sglang source. The input/output op shape stays small and explicit.

**Load-bearing threshold:**

- `pool_lens <= 512`: identity, all closed pools and the open tail, in order;
  no logits are read. This includes sequence lengths through 2051.
- `pool_lens > 512`: real top-512 selection from the supplied logits.
  Never replace this branch by identity or dense attention.

Four byte-radix passes find the cutoff. Integer shared-memory histogram
atomics only count bins; they never allocate output positions. A stable
scan selects all scores above the cutoff and the lowest pool ids among
cutoff ties. Selected pools are emitted in ascending logical pool order.
Each expands to four token positions. The open tail is appended. Mapping
is `bt[token >> 8]*256 + (token & 255)`. All remaining output columns are
written as zero; there is no clamp call or materialized token slot table.

The scan packs per-1024-item tie and greater counts into 11-bit fields of a
32-bit value. Counts cannot carry into each other. Warp-scan CUB lowering
has zero spills. Signed finite FP32 scores are ordered correctly; +/-0 tie.
For diagnostic non-finite input, NaN ranks as -infinity. Finite logits remain
the production contract. No host read, data-dependent graph choice, or
multi-CTA barrier is used.

Capacity is determined by `bt_cols` and the caller's length bounds. The
current layout permits 65536 pools; 35280 is covered. The script checks
larger contexts as well as both sides of the 512-pool threshold.

## 2. Kernel and ABI table

All grids use S=`seqs`, R=`ceil(S/4)`. All shared memory below is **static**;
launch dynamic shared memory is zero. P=64-bit pointer, I=32-bit integer.
Pointer order is the C CUDA order, not TileLang alphabetical order.

| entry (`glm53_dsa_` prefix) | ABI | block | grid | CTAs S=1 / 8 / 16 | registers | shared B |
|---|---|---:|---|---|---:|---:|
| `w_kc_v2` | P,P,P,I | 128 | (32,8,R) | 256 / 512 / 1024 | 40 | 0 |
| `w_vc_v2` | P,P,P,I | 128 | (16,8,R) | 128 / 256 / 512 | 40 | 0 |
| `norm_store_v2` | 8P | 1024 | (S,1,1) | 1 / 8 / 16 | 31 | 3328 |
| `had_quant_v2` | 3P | 128 | (32S,1,1) | 32 / 256 / 512 | 20 | 528 |
| `xproj_v2` | 6P,I | 128 | (72,R,1) | 72 / 144 / 288 | 36 | 0 |
| `xepilogue_v2` | 14P,I | 128 | (S,1,1) | 1 / 8 / 16 | 32 | 560 |
| `step_v2` | 5P,I | 32 | (1,1,1) | 1 / 1 / 1 | 20 | 128 |
| `topk_v2` | 5P,2I | 1024 | (S,1,1) | 1 / 8 / 16 | 31 | 32976 |

Exact pointer/scalar order:

```text
w_kc/vc:       a, weight, out, rows
norm_store:   QKV, q_norm_w, kv_norm_w, qa, qa8, sfa, kv_layer_base, slots
had_quant:    iq, iq8, q_scales
xproj:        x, wk, gate_w, head_w_f32, kg_scratch, head_scratch, rows
xepilogue:    idx_layer_base, tail_base, kg, head, norm_w, norm_b, ape,
              block_table, tail_lines, positions, seq_lens, valid,
              q_scales, head_weights_out, bt_cols
step:         seq_lens, pool_lens, pool_ctx, dsa_lens, mqa_sched, rows
topk:         logits, pool_lens, block_table, seq_lens, topk_out,
              logits_stride, bt_cols
```

The new pointer slots are at byte offsets `8*i`; trailing integers start
after the pointers and advance by four. `--audit` checks these against ELF
KPARAM metadata, not just the Python declarations.

**FP8 1d2d ABI is unchanged:** raw first pointer = weight scales, final TMA =
activation SFA. `ops()` asserts the correct parameter indices (3 and 2).
Do not alias rows [S,16) with another live buffer: inherited FP8 GEMMs can
TMA-store the whole 16-row tile. New row-indexed kernels only use rows <S.

**PDL:** every new entry waits before input access and releases after its
accesses. The manifest default is conservative (`pdl` off), compatible with
the current gen.py PDL-count guard. No PDL saving is included in the model.
Do not enable additional PDL edges until the owner updates the entry
allowlist and verifies mixed old/new graph edges. There are no counters
that depend on op scratch being freshly zeroed each invocation.

## 3. Numerics argument and risks

| part | expected relation to baseline | release gate |
|---|---|---|
| absorb GEMV | Same exact-real operation; FP32 reduction tree differs from tensor cores/split-K | relative L2 <1e-3 and max error <1% of baseline max on the direct test; recorded model cuts and greedy gate still required |
| norm/quant/store | Same reduction tree, BF16 boundary, scale layout and slot bytes | Exact QA, QA8, SFA and entire guarded KV allocation |
| Hadamard/quant | Same butterfly order and BF16 boundary | Exact FP8 codes and scales against the pinned cubins |
| x projection | wk/gate reduction differs; head projection retains original per-lane accumulation | Tolerance for wk/gate; exact head scaling in isolated test |
| x epilogue | Same state and pool arithmetic for identical kg/head inputs | Exact idx bytes, scale bytes, tail bytes and head output |
| step metadata | Integer arithmetic copied from the fixed specialization | Exact `[133,2]` versus pinned metadata cubin |
| top-k | Same largest-512 set on unique finite scores; lowest-id cutoff ties; fixed ascending output order | Exact reference selection and slot mapping, including ties and repeated replay |

Main risks:

1. **Full-model validation is outstanding.** Direct GPU tests now pass, but
   they do not cover model-weight cuts or TP8 greedy parity. The unfused
   baseline's M=32 extent defect must be fixed by the owner for multi-row
   Hadamard comparisons. Strict tolerances were not relaxed.
2. At S=8/16 the SIMT projection may lose to tensor-core throughput. Compare
   in-graph, and keep the old path if any target bucket regresses.
3. Near-tie logits can change selected pools after wk/gate rounding changes.
   Deterministic selection does not imply bitwise agreement with sglang's
   nondeterministic output order. Compare same-input top-k sets separately
   from full-model tolerance/greedy checks.
4. Single-CTA radix throughput at 35280 pools is unmeasured. No multi-CTA P3
   speedup is claimed. Histogram contention and long-context latency are
   explicit gates, including an all-equal-score case.
5. `valid` guards idx/tail writes. KV still writes the baseline's pad slot.
   The runtime must provide nonnegative leased slots and in-range page ids.
6. S=16 FA3 scratch is enlarged, but the inherited FA3 path has not been run
   at S=16 in this work. All shapes are decode-only, not prefill/speculation.
7. TP8 and 11 DSA layers are fixed. `fuse_manifest` rejects a changed KV/idx/
   tail state geometry; do not use these offsets for a 12-layer MTP layout.
8. The accepted dense-expanded versus absorbed prefill numerical difference
   is outside this change. No prefill kernel is replaced.

## 4. Expected time model — NOT measured

The supplied AR-manifest attribution is 0.343 ms (`dsa_w_vc`) + 0.312 ms
(`dsa_w_kc`). Do **not** subtract 0.655 ms directly from the 7.43 ms graph.
At the supplied ~1.6 attribution inflation, this is about **0.409 ms** of
untraced absorb cost. `crates/kern-run/src/bench/mod.rs` explains that op
shares use the *traced total*, not graph time; `profile.rs` inserts explicit
external timing events around calls. Those boundaries add cost and alter
execution. Use attribution to rank shares; use clean graph A/B for savings.
The referenced SHARES note was not found on this host; the factor here is
from the task and the mechanism is confirmed in the runtime source.

Planning assumptions, no PDL credit:

- A new absorb launch at S=1/4: approximately 4–8 us; two × 11 = 0.088–0.176
  ms/step, versus the calibrated 0.409 ms reference.
- Larger row tiles at S=8 can add work; no extrapolated measured baseline is
  available for S=4/8. Budget a wider uncertainty there.
- Norm/store and Hadamard fusion: roughly 0.05–0.09 ms combined.
- X-side and metadata/slot-map fusion: roughly 0.03–0.07 ms if neither the
  projection nor radix regresses. No radix-compute improvement is assumed.

| batch | absorb-only expected saving | full-candidate expected saving |
|---:|---:|---:|
| 1 | 0.23–0.32 ms | **0.31–0.48 ms/step** |
| 4 | 0.20–0.31 ms | **0.28–0.47 ms/step** |
| 8 | 0.14–0.27 ms | **0.22–0.43 ms/step** |

These are conditional engineering estimates, not confidence intervals. An
unmeasured top-k or GEMV regression can erase them. DSA alone does not close
the complete 7.43-versus-5.68 ms gap. The other agents' work remains needed.

## 5. Exact owner integration

First build **only these sources**, not the wildcard build script (other
agents may have kernels in progress):

```bash
cd <kern-repo>
for name in glm53_dsa_v2 glm53_dsa_topk_v2; do
  /usr/local/cuda-13.0/bin/nvcc -cubin -arch=sm_90a -std=c++17 -O3 \
    -ccbin /usr/bin/g++-14 -Xptxas=-v \
    tools/glm53/kernels/$name.cu \
    -o kernels-glm53-handwritten/$name.cubin
done
PYTHONPATH=tools python3 -B -m glm53.ops_dsa_v2 --audit --cpu-check
```

### Stage A: absorb only

In `gen.py`, add:

```python
from glm53 import ops_dsa_v2
```

Replace only the DSA op-builder line in `build()`:

```python
# old
allops = ops_dsa.ops({"bt_cols": BT_COLS, "st_cols": ST_COLS})
# new
allops = ops_dsa_v2.ops({"bt_cols": BT_COLS, "st_cols": ST_COLS})
```

No decode calls, weight binds, layouts, or count assertions change. This is
also the smallest rollback boundary.

### Stage B: norm/store, Hadamard and metadata hoist

Immediately after the raw `m = {...}` is constructed, and **before**
`wire_allreduce(m, ...)`, insert:

```python
ops_dsa_v2.fuse_manifest(m, xside=False, topk=False)
```

The helper mutates the same `allops` dictionary held by `m['ops']`. It runs
after the existing 58-op construction check but before call-set invariant
checks, module resolution and wire lowering. Do not put it after `lower_wire`.
The baseline top-k and its slot table stay in place at this stage.

### Stages C/D: gated optional x-side and mapped top-k

After each independent numerics/performance gate, use:

```python
ops_dsa_v2.fuse_manifest(m, xside=True, topk=False)  # X-side only
# or
ops_dsa_v2.fuse_manifest(m, xside=False, topk=True)  # mapped top-k only
# finally
ops_dsa_v2.fuse_manifest(m, xside=True, topk=True)   # full candidate
```

Call the helper once, not once per stage. It removes only the old DSA calls
and unused DSA buffers. It makes no MoE/mHC/KDA or AR substitution. It can
coexist with the MoE helper, each run once on the raw manifest. No packed
N=288 checkpoint tensor is needed.

Owner commands after editing, with a **new** manifest filename:

```bash
python3 -B tools/glm53/gen.py --allreduce lamport \
  --out examples/glm53-flash-dsa-v2.json --bundle kernels-glm53
target/release/kern verify examples/glm53-flash-dsa-v2.json
```

The owner has since added `--dsa v2a` to gen.py for Stage A; with that
interface use `--dsa v2a` in the command above. This work did not edit that
shared file, create the integrated manifest, or change the running service. Keep `examples/glm53-flash-ar.json` as the A reference. To roll back,
remove the helper call and restore `ops_dsa.ops(...)`.

## 6. A/B script and execution plan

The test script is embedded in `ops_dsa_v2.py` to stay inside the permitted
write scope. It does not JIT, alter a service, or write test artifacts.
Before CUDA init it checks `tmux ls`, all GPU utilization/memory, and serving
processes. It refuses an occupied selected GPU or an active shared serving/
benchmark process. An isolated allocation on a different GPU is left alone. Ask the owner to release it;
do not kill another agent's job.

```bash
# Do NOT source kserve_prod.sh: it would launch a server.
export LD_LIBRARY_PATH=<nccl + cuda-compat libs for your node>
cd <kern-repo>
PYTHONPATH=tools <sglang-python> -B \
  -m glm53.ops_dsa_v2 --audit --test --gpu 0
# Short event-bracketed CUDA-graph proxy timings, only after numerics pass:
PYTHONPATH=tools <sglang-python> -B \
  -m glm53.ops_dsa_v2 --bench --gpu 0
```

Implemented coverage:

- S=1/2/3/4/5/8/15/16 absorb projections; guard rows; captured graph replay
  after input mutation. Eight Torch mm calls are the microbench proxy, not a
  claim to reproduce kern's cuBLASLt algorithm/workspace exactly.
- Pitched QKV, exact QA and real mined Q quant, SFA pitch 16, whole guarded
  KV allocation at nonzero layer offset 10*1024.
- Hadamard and quant against the actual two pinned baseline cubins, with
  corrected flattened M=S*32, per-token reference cross-checks, sentinels,
  guard rows, random and structured/amplitude-varied inputs.
- Mixed scheduler lengths and exact comparison to pinned MQA metadata.
- X-side projection with BF16 wk/gate and arbitrary FP32 head weights.
- Isolated epilogue: exact ring and idx state bytes, scales, neighboring
  layers/pages, invalid rows; positions 254/255/256/259/263 cover pool close,
  page transition, both ring halves and wrap.
- Top-k at S=1/2/3/4/5/8/15/16, for 0/1/511/512/513/35280/65535 pools,
  random and tied signed scores,
  nonidentity physical page tables, tail lengths 0..3, all-equal ties,
  identity with NaN logits (must not read them), zero padding, guard rows,
  and repeated identical output. A captured top-k graph crosses 512 -> 513
  pools after in-place length/logit changes; the latter must select pool 512.

Required after direct tests pass:

1. Run short clean **full-graph** AR-manifest A/B at bs1/2/4/8, then bs16 for
   support validation. Use the same context and snapshots. Price each stage
   separately. Do not sum attributed milliseconds as graph savings.
2. Use actual layer weights and captured activations at all 11 DSA layers.
   Compare qv, ao, av, pool bytes, selected sets and final logits. Keep
   same-input selector correctness separate from upstream rounding changes.
3. Check prefixes around 2047/2048/2051/2052 tokens, page/ring boundaries,
   mixed lengths, and 141k context. Confirm no future/tail omission or
   repeated pool ids. Repeat long-context selection to verify determinism.
4. Run the model-level greedy gate from arch_decode.md: >=99.5% agreement
   over 1k tokens at long context, with the accepted prefill delta unchanged.
5. Run short sanitizer checks for state/guard access on a free GPU. Full
   stateful CUDA-graph/PDL validation and sanitizer results are outstanding.
6. Publish untraced graph medians and per-bucket regressions. Reject a stage
   that regresses any required bucket; do not hide it with another fusion.

## 7. Built artifact identity

Both artifacts were produced on the bring-up node with CUDA-13, sm_90a, C++17, O3,
g++-14. No `--use_fast_math` flag was used.

```text
glm53_dsa_v2.cubin
16efcdb822dc07b74fd76f995b3c754b296fede28bf8a9c4443786a52b7b427c

glm53_dsa_topk_v2.cubin
3bc73eed3c8e4246828b2641d277fc3059e331bff1a904a45ecf705960c4a5a1
```

`handwritten()` computes the current artifact SHA for the generated manifest;
the owner must rebuild, audit, regenerate and rebundle together after any
source edit. No silent model-family, weight-layout, or scale-ABI substitution
is involved.
