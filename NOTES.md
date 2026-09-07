# Notes: making the AgentX-median request cheaper on Qwen3.8-27B (one GB300)

score = T_prefill(1536 new on 32k) + 444 × T_decode_batch(16 seqs at 32k).
Baseline at 29dbc9c: extend 72.4 ms, step 18.49 ms, score 8283 (my rerun;
the task sheet says 8261). The step is 99% of the score; extend is noise.
Session 1 ended at 7203 (982e596); session 2 is at 6784 (d25ab02).

Measure: `target/release/kern bench qwen3.8-27b --gpu 0 --program-only
--workload /work/exp/workload.json --out /work/exp/results/<name>.json`
then `python3 /work/exp/score.py …`. Step cv must be < 1%; rerun otherwise.
Correctness: `kern test qwen3.8-27b --reference /tmp/an/ref.json --manifest
examples/qwen3.8-27b.json --gpu 0` where ref.json = `git show
29dbc9c:examples/qwen3.8-27b.json`. Cargo is at ~/.cargo/bin (not on PATH
in background shells).

## Where the step's 18.5 ms goes (in-graph, measured by ablation)

Ablation = replace an op's impl by a 1-block no-op launch (keeps the node),
`/tmp/an/ablate.py`; the `--program-only` bench then gives the step without
that op. The full bench's per-call "warm" numbers carry ~6-10 µs of
event/launch overhead per call and overstate small ops; the "attributed"
column is inflated ~26% by instrumentation. Trust ablation, not those.

| what | in-graph ms | notes |
|---|---|---|
| gemm (305 calls, cuBLASLt) | ~9.3 | 51 GB of bf16 weights; standalone cuBLASLt reaches 4.3-6.3 TB/s per shape (lm_head 7.0). Floor at 7 TB/s ≈ 7.3 ms |
| attn_batch (16, TRTLLM-GEN, 38 splits) | ~5.4 | 34 GB of KV; 6.4 TB/s already. Floor ≈ 4.8 ms |
| conv_update + recurrent (96) | ~1.5 | 100 MB of f32 state per layer read+written at ~3.5 TB/s |
| gemma_fused_norm (128) | ~1.4 | ~11 µs per call for 16 rows: the kernel serializes 10 dependent load round trips per thread |
| other small ops (~290: silu_mul, copy_rows, sigmoid_mul, gated_norm, q/k norm, rope, kv write) | ~1.5 | 5-8 µs each in-graph |

Graph node overhead itself is small on this GPU: 0.65 µs for an empty
node, ~1.6-2 µs for a 16-block trivial kernel (`/tmp/an/gap.cu`).

Read+write ceiling: a 50 MB D2D copy takes 17.7 µs (5.9 TB/s counting both
directions), so state-streaming kernels bottom out near there.

## Done

1. **GDN decode fused** (`tools/kernels-src/gdn_decode.cu`, wired by
   `tools/qwen38_fuse.py`): conv_update + recurrent + z copy + gated norm →
   `gdn_conv` + `gdn_step` per layer. Step kernel: one 64 KB bulk async
   copy of the head's state into smem, warps own rows, 20.9 µs/layer for
   16 seqs vs 21.8 for the Triton recurrent alone; the pair is 23.5 µs vs
   36.1 µs for the four captured kernels (standalone harness
   `/tmp/an/gdn_check.cu`, rotating 2.4 GB of state pages). Numerics: conv
   path bit-exact with the Triton cubin; recurrent state within 2.4e-7
   (f32 ulp), output 2 of 98k values off by 1 bf16 ulp. Matching Triton's
   lowering mattered: exp = ex2.approx(x·log2e), `/` = div.full,
   sqrt/rsqrt = approx.ftz; before that, beta = bf16(sigmoid(b)) flipped a
   rounding tie in a few heads and moved whole heads' states by 3e-4.
   Chunking the copy into 2 or 4 barriers was slower (21.9 / 22.1 µs).

2. **gemma_rms_norm rewritten** (same entry/ABI, bit-exact incl. ATen's
   reduction order): every thread issues its 16-byte loads up front instead
   of ten dependent scalar round trips. Standalone: 16 rows 7.2 → 4.4 µs,
   1 row 6.4 → 3.4, 1536 rows 22.0 → 19.5 (`/tmp/an/norm_check.cu`). The
   head norms (N = 256) get 64-thread blocks in the manifest (512 idle
   threads made the prefill-size case slower: 115 → 174 µs). Bench: step
   17.90 → 16.64 ms, extend 72.7 → 69.4, score 8020 → 7458.
3. **attn_prep** (`tools/kernels-src/attn_prep.cu`): q_norm + k_norm +
   mrope + reshape_and_cache → one launch per token, bit-exact. 16 tokens
   10.1 → 4.75 µs, 1536 tokens 144 → 13 µs (`/tmp/an/prep_check.cu`).
   Triton's mrope does the rotation in native bf16 instructions and the two
   halves fuse differently: x1' = fma.rn.bf16(x1, c, -mul.bf16(x2, s)),
   x2' = fma.rn.bf16(x1, s, mul.bf16(x2, c)). Any f32 emulation is off by
   one bf16 ulp on ~5% of the rope dims.
4. **silu_mul / sigmoid_mul rewritten** (vectorized, 8 elements per
   thread, grid.y over the row), bit-exact: 16 tokens 5.3 → 2.2 µs and
   13.0 → 2.3 µs; 1536 tokens 52 → 27 and 28 → 11. silu semantics (from
   the mined vLLM kernel): bf16(silu_f32(g)) then a bf16 multiply.

5. **attn_batch splits**: the manifest fixed 38 splits (2432 CTAs of 1
   per SM). Sweep at batch 16 / 32k (step ms, all cv ≤ 0.0014):
   8 → 16.223, 16 → 16.207, 24 → 16.282, 32 → 16.330, 38 → 16.39,
   48 → 16.502, 64 → 16.871. Extend is unaffected (prefill uses the
   context kernel). Splits 1 (persistent artifact), 12, 20: see
   `/tmp/an/extra_results.txt` / the commit.
6. **in_proj_ba folded into gdn_conv — reverted.** Computing the N = 96
   projection inside gdn_conv (one output per block, fixed-order f32 sum)
   saved 0.086 ms per step (16.207 → 16.121) but its accumulation order
   differs from cuBLAS, and once gdn_step was bit-exact it was the only
   op left moving the logits (2-3.5 ulp at scale, argmax always agreeing).
   Bit-identical logits are worth more than 0.5% of score, so the ba GEMM
   is back (commit 4e68579 has the folded version for reference).
7. **gdn_step made bit-exact**: Triton's PTX for the recurrent kernel
   (compiled here from vLLM's source with the model's constexprs) shows
   the reduction of a thread's four elements as mul(e1) then fma(e0),
   fma(e2), fma(e3), a butterfly xor 16..1, the update as fma(k, u, h·eg),
   q pre-scaled as scale·(q/sqrt). Mirrored with _rn intrinsics: state and
   output 0 of 12.6M / 98k values differ. Cost: none once the rows went
   back to two passes (20.5 µs; the per-row dependent version was 23.6).
   The gated norm tail is mirrored the same way (a lane owns o[4l..4l+3],
   var = sum / N, rstd = rsqrt(eps + var), y = (o·rstd)·w, y = y·(z·σ(z)));
   `__launch_bounds__(256, 3)` keeps it at 80 registers (103 unbounded cost
   a CTA per SM). Attention splits 38 → 16 were already bit-identical end to
   end, and the ba fold is reverted, so every decode op is now bit-exact.
   Before the norm tail was mirrored, a handful of cuts had one value of
   core_attn_out off by 1 ulp and the whole-state compare reported 22% of
   the GDN bytes differing (layers 1..47 of the driven slot: the ulp
   propagates through out_proj into every later layer's state). With it
   mirrored the attestation is bit-identical at every cut, in both states
   and in the logits, on the random and the prose prompt. `KERN_TEST_DUMP=
   <dir>` (added to kern test) writes both sides' final state images.

## Correctness evidence on the final tree

`kern test` vs 29dbc9c: random-token prompt (300 tokens, 16 steps) and the
prose prompt (docs/runtime.md, 431 tokens, 64 steps) are both
"bit-identical at every cut, real and perturbed inputs": every buffer,
both states and all logits. Earlier trees (GDN with its own reduction
order, or the folded ba GEMM) passed at 2-3.5 ulp at the row's scale with
all argmax agreeing; the attention split change alone was already
bit-identical.

## Hazards met

- A rewritten elementwise kernel with a new thread→element mapping needs
  its manifest grid changed in the same pass; the `repin` pass alone
  re-pins the sha and keeps the old grid. That produced a manifest whose
  sigmoid gate covered a third of each row (the first "m-norm" bench was
  on it). Keep every kernel's grid next to its `hw()` in the converter.
- `kernels-qwen38/` is gitignored (derived artifacts); handwritten cubins
  are reproduced from source by `tools/build_kernels.sh` and pinned by sha,
  so only the sources and the manifest are committed.

- `tools/test_trtllm_manifest.py` fails to import at 29dbc9c already
  (`resolve_constants` is not in kern_manifest.py; `trtllm_attention.convert`
  has the same dangling import). Untouched; qwen38_fuse.py uses only
  `trtllm_attention.op`.

## Tools

`tools/qwen38_fuse.py --input /tmp/an/ref.json --output
examples/qwen3.8-27b.json` (all passes: gdn,repin,attn,elem,splits,tiles,
tiled) regenerates the manifest from the 29dbc9c one; then
`tools/extract_kernels.sh <manifest> target/cubins kernels-qwen38` lands the
new cubins (it prints MISSING for the captured ones already there; harmless).
Validation harnesses for each kernel against the pinned cubins live in
`/tmp/an/*_check.cu` (driver API, random inputs, bit comparison + timing);
they are not in the repo. The GEMM harness that includes the production
source and checks every shape at M = 16 and M = 1 against cuBLASLt is
`/tmp/bn/tgf.cu` (`nvcc -O3 -arch=sm_103a -o tgf tgf.cu -lcublasLt -lcuda`).
`kern bench qwen3.8-27b --manifest <alt.json> ...` benches a variant
manifest (the target name is still required).

`kern test` with defaults panics on this target ("position past the lease",
`Caller::stage`, from the prefill sweep past the 4096-token lease). Working
invocation (71 s, after the attest fix that leases one sequence's slots):
`kern test qwen3.8-27b --reference /tmp/an/ref.json --manifest <cand> --gpu 0
--no-perf --capacity 4096 --prefill 300 --decode-steps 16 --no-sweep --fuzz 0`.
Before the fix a run grew past 700 GB of host RSS (whole-state reads of 128
GDN slots per cut).

## Tried, did not work

- 4-way / 2-way chunked bulk copies in gdn_step (see above).
- Register-resident state (64 floats/thread, 256 threads): 170 regs, one
  CTA per SM, 24.8 µs; capped at 128 regs it spilled and was still slower
  than the smem version.

8. **cuBLASLt tile pinned for gate_up and lm_head** (decode_batch only):
   `extern:cublaslt_bf16_tn_tile` takes (tile, splitk) and picks that
   algorithm from the heuristic list (fallback to the default when a shape
   is not offered it, e.g. M ≠ 16; debug-logged). Harness at M = 16: every
   splitk=1 candidate is bit-identical to the default; tile 312 is 54.3 µs
   vs 56.8 for gate_up and 357.9 vs 364.7 for lm_head. Split-K candidates
   differ in bits, so o_proj / down_proj / qkv keep their defaults. Pitfall
   met: cudarc's `device_ptr(stream)` on a `CudaSlice` makes the stream wait
   on the allocation's write event; inside a graph capture that invalidates
   the capture and every later launch fails with EXECUTION_FAILED (the
   harness replica of the same calls ran fine). Take the address once.

## State at the end of session 1 (commit 982e596)

score 8283 → 7203 ms (step 18.49 → 16.07 ms, extend 72.4 → 67.4 ms), every
op bit-exact with the reference (kern test: bit-identical at every cut on
a random and a prose prompt). decode_batch has 646 nodes: gemm
305, gemma_fused_norm 128, silu_mul 64, gdn_conv 48, gdn_step 48,
attn_prep 16, attn_batch 16, sigmoid_mul 16, embedding 3, gemma_norm 1,
argmax 1. Where the 16.2 ms goes now: gemm ~9.3, attention ~5.2,
gdn ~1.1, the rest ~0.6.

## Session 2: the GEMMs (commits 229cd91.. )

9. **Tiled weight layout + handwritten decode GEMM** (`tools/kernels-src/
   gemm16_tiled.cu`, manifest `layout` + `tensor` on weight buffers, pass
   `tiled` in qwen38_fuse.py). decode_batch's qkvz, qkv, out/o_proj,
   gate_up and lm_head read a second copy of their weight stored as
   contiguous 64x64 tiles (swizzled so ldmatrix is conflict-free); the
   runtime permutes the file's bytes at load (`layout.rs`). One CTA per
   (tile row, split-K chunk), cp.async ring, 4 warps of mma.sync. Score
   7203 → 7153 (step 16.071 → 15.958). kern test: bit-identical at every
   cut, decode_batch driven at M = 1.

### What the GEMM study established (harnesses in /tmp/bn)

- **cuBLASLt's bits are reproducible.** Non-split algorithms = one f32
  chain in ascending k16 steps, any tile. Split-K = kc = ceil(K/S) rounded
  up to 64 (last chunk shorter), f32 partials summed ascending, one bf16
  rounding. Its split choice depends on M: down 17 at M ≤ 8 / 13 above,
  qkv 2 / 1 at M = 9..15 / 2 at 16, out_proj 5, qkvz/gate_up/lm_head 1,
  ba 5/16/5/4/5 (M = 1, 2-4, 5-8, 9-15, 16). `kern test` drives
  decode_batch at M = 1, the bench at M = 16: an op carries both counts.
- **The bandwidth ceiling is ~7.1 TB/s** (pure streaming, any load path:
  cp.async, 1-D bulk, TMA boxes, plain loads; `/tmp/bn/stream.cu`). One
  CTA with a 16-deep 8 KB ring pulls at most 68 GB/s (`persm.cu`), so
  ~105 of 152 SMs busy already saturate HBM: tile imbalance is cheap.
- **The previous session's "contig 7.66 TB/s" was an L2 artifact**: its
  contiguous-mode address arithmetic made the split CTAs of an n-tile read
  overlapping regions. Tile order in memory (n-major vs k-major) makes no
  difference; DRAM page locality is not the story on this GPU.
- **Where the split-K shapes lose** (out_proj 63 MB in 14 µs = 4.4 TB/s;
  down 178 MB in 33 µs): per-item fixed latency — pipeline fill ~2 µs,
  then partial store + fence + atomic + reduce ~2-3 µs per item — idles
  the SM slot. Tried: TMA/bulk producer warp with 8-20 stages (5 TB/s: not
  the copy engine), persistent CTAs walking balanced item ranges with the
  ascending running sum in registers (only tile-straddling ranges touch
  the workspace; parity at best), 2 CTAs/SM, cp.async persistent. The
  non-persistent cp.async kernel (one CTA per item, 4-8 stages) is the
  best or equal everywhere; down stays on cuBLAS (37.3 vs 33.3 µs).
- Standalone µs cuBLASLt → tiled: gate_up 56.8 → 53.1, qkvz 27.6 → 27.0,
  qkv 27.5 → 26.0, out/o_proj 14.6 → 14.3, lm_head 364.6 → 348.6.
  In-graph (full bench, p50 minus ~6.4 µs of instrumentation per call)
  they match, except gate_up at ~61 µs: 544 CTAs is 3.6 per SM, wave
  quantization; the depth sweep is below.
- A reduce loop with a runtime trip count serializes its loads (one L2
  round trip per partial, ~10 µs for 13); unroll with a compile-time
  bound, but keep it to 4 partials per round — 32 float4 arrays spilled
  280 B of stack and cost 15% on every shape.

10. **gate_up at 8 stages** (f77ebc3): in-graph the 544-CTA shape ran 61 µs
    at 4 stages; 8 stages → step 15.958 → 15.901, score 7127.6.
11. **PDL — programmatic dependent launch** (ab96158, 3c7ec92): a kernel
    launch marked `pdl` gets CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_
    SERIALIZATION (a programmatic graph edge under capture). Every
    handwritten kernel does `griddepcontrol.wait` before its first dependent
    access, then `launch_dependents`; the tiled GEMM issues its first
    STAGES-1 weight tiles before the wait. The TRTLLM-GEN decode attention
    is marked too (its SASS has ACQBULK/PREEXIT; the sm_103a prefill cubin
    cannot be disassembled with the nvdisasm on this box, so it stays plain
    — prefill is 1% of the score anyway). cuBLAS externs stay plain. Step
    15.901 → 15.355 ms, score 7127.6 → 6885.4. kern test bit-identical at
    every cut (10973). Rule: a PDL kernel may touch only weights before its
    wait; the trigger fires when every CTA of the primary has passed it (or
    exited), so an early trigger in a multi-wave kernel cannot starve the
    primary of SM slots.

12. **gate_up + silu_mul fused** (2f7da28): 256-thread CTA, gate warps and
    up warps, up rounded to bf16 through smem, silu_mul's exact ops on the
    GEMM's own bf16 output. Standalone 54.2 vs 59.6 µs for the pair; in the
    step only 15.355 → 15.334 (PDL had hidden most of the silu launch).
13. **in_proj_qkvz + in_proj_ba one launch, qkv whole-tile** (d25ab02): the
    GEMM body owns a chunk range; with grid.y = 1 a CTA folds every chunk of
    its n-tile into the ascending running sum in registers (no workspace).
    qkv runs so (25.6 vs 26.5 µs). The dual entry streams the 96-row ba
    weight (32-row tiles, `layout {"tile": [32, 64]}`) as two extra CTAs
    running its 5 chunks in place. Step 15.334 → 15.126, score 6783.9.
    Bit-identical at every cut (9701).

## Next

- down_proj (64 × ~31 µs in-graph, cuBLAS, 13/17 chunks) is the last plain
  launch inside the layer and the biggest single GEMM cost (1.99 ms by
  ablation). Every handwritten form tried is slower (37 chunked, 34
  persistent, 36.6 cluster); whole-tile is 53 (80 CTAs); two CTAs per
  n-tile in the persistent range walk (G = 160, one publish and one
  reduce per n-tile) is 34.7 at 8 stages and 51 at 12+ (the 8 doubly
  loaded SMs fall into a second wave). A bit-identical 25 µs version would
  be worth 0.4 ms; every form tried is limited by bytes in flight per SM
  against per-item fill and epilogue idle time, not by the reduce.
- Remaining small fusions: sigmoid_mul into o_proj's A load (16 calls,
  ~0.05 ms). gdn_conv + gdn_step cannot fuse per (head, seq) CTA: a q/k
  head's conv state is shared by three value heads, so the in-place shift
  would race.
- Attention splits re-swept under PDL at d25ab02: 12 → 15.208, 16 → 15.126,
  20 → 15.165, 24 → 15.227 ms. 16 stays.
- Tried, no gain (session 2): a thread-block-cluster split-K (the CTAs of
  one n-tile in a cluster along grid.y, rank 0 holding the ascending
  running sum, the others single partials in smem, rank 0 reducing over
  DSMEM after a cluster barrier) is bit-exact but no faster than the
  global protocol — out_proj 15.4 vs 14.5 µs, down 36.6 vs 37.5 (cuBLAS
  33.3) — so the split-K shapes' cost is a ~10 µs kernel's fill and tail,
  not the reduce (code in the commit history of this note, dropped from the
  source). cuBLASLt offers no other bit-identical candidate for down (one
  with 13 splits) and every ba candidate is ~6.3 µs (/tmp/bn/algos.cu).
- down_proj on the tiled kernel with PDL: step 15.669 vs 15.355 — stays
  on cuBLAS. Ablation (op → zero-row copy_rows): down costs 1.99 ms of the
  15.36 ms step (31 µs a call, 5.7 TB/s), in_proj_ba 0.17 ms (3.5 µs).
- Tried, no gain: 128-row CTAs (8 warps, two tiles per stage; down 37.9,
  out_proj 16.1, qkv 27.6, gate_up 54.0 µs — never better than 64 rows);
  an L2 prefetch (`cp.async.bulk.prefetch.L2.global`) of the rest of the
  CTA's item before `griddepcontrol.wait` (/tmp/bn/pdlpair.cu: every cap
  from 64 KB up made the primary+GEMM pair slower, 15.1 → 15.7-16.7 µs for
  out_proj after a norm-sized primary — L2 does not keep the lines under
  the concurrent stream). PDL itself in a plain stream is worth 1.5-2 µs
  per pair.
- gdn_step 20.9 µs vs 17.7 copy ceiling (0.15 ms/step at most).
- The graders' criterion allows bf16 noise; if that ever matters more
  than exactness, 4e68579 (ba fold) is the measured 0.5%.
