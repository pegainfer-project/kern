# Notes: making the AgentX-median request cheaper on Qwen3.8-27B (one GB300)

score = T_prefill(1536 new on 32k) + 444 × T_decode_batch(16 seqs at 32k).
Baseline at 29dbc9c: extend 72.4 ms, step 18.49 ms, score 8283 (my rerun;
the task sheet says 8261). The step is 99% of the score; extend is noise.

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
6. **in_proj_ba folded into gdn_conv**: the N = 96 GEMM (1 MB, a 6 µs
   cuBLAS node per GDN layer) is computed by gdn_conv, one output per
   block over a 96-wide grid.y, each warp an eighth of `hidden`, partials
   summed in a fixed order, bf16 out (0 of 1536 values differ from the
   exactly rounded dot product on the harness). Standalone the conv launch
   grows 2.6 → 5.3 µs (L2 traffic: 16 seqs × 1 MB + 96 × 10 KB rows);
   two other layouts (4 outputs per block, 2 warps each; batched loads)
   cost the same. In-graph: step 16.207 → 16.121 ms, score 7264 → 7226.

## Hazards met

- A rewritten elementwise kernel with a new thread→element mapping needs
  its manifest grid changed in the same pass; the `repin` pass alone
  re-pins the sha and keeps the old grid. That produced a manifest whose
  sigmoid gate covered a third of each row (the first "m-norm" bench was
  on it). Keep every kernel's grid next to its `hw()` in the converter.
- `kernels-qwen38/` is gitignored (derived artifacts); handwritten cubins
  are reproduced from source by `tools/build_kernels.sh` and pinned by sha,
  so only the sources and the manifest are committed.

## Tools

`tools/qwen38_fuse.py --passes gdn,repin,attn,elem` regenerates
`examples/qwen3.8-27b.json` from the 29dbc9c manifest (`--input`); then
`tools/extract_kernels.sh <manifest> target/cubins kernels-qwen38` lands the
new cubins (it prints MISSING for the captured ones already there; harmless).
Validation harnesses for each kernel against the pinned cubins live in
`/tmp/an/*_check.cu` (driver API, random inputs, bit comparison + timing);
they are not in the repo.

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

## Next

- gemma_fused_norm rewrite (vectorized, all loads issued up front, same
  ATen reduction order so it stays bit-exact): 1.4 → ~0.4 ms.
- silu_mul, sigmoid_mul rewrites; fuse q_norm + k_norm + rope + kv_write.
- Attention: sweep `--splits` (38 fixed today; 2432 CTAs at 1 CTA/SM).
- GEMM at M=16: custom mma.sync kernel or cuBLASLt algo choice; the
  weak shapes are o_proj (4.3 TB/s), down_proj (5.3), qkv_proj (5.4).
  in_proj_ba (N=96) is a 6 µs node for 1 MB; fold into something.
