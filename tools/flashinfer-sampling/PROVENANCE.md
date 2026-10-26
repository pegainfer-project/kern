# flashinfer_sampling

The `flashinfer_sampling` registry family is FlashInfer's sampling kernels,
built from its headers with nothing of FlashInfer's changed: `instantiate.cu`
names the template instances FlashInfer's own host launchers pick on sm_100+
and `build.sh` compiles them to one cubin.

- Upstream: [FlashInfer](https://github.com/flashinfer-ai/flashinfer) v0.7.2
  (`d84335d8c54e0c8f9f7e22faa15c7e34a58abf94`), `include/flashinfer/sampling.cuh`
  and what it includes (`topk.cuh`, `topk_common.cuh`, `utils.cuh`,
  `vec_dtypes.cuh`, `math.cuh`, `allocator.h`, `exception.h`), plus CUB from
  the CUDA toolkit. Apache-2.0; the licence is the family's `license_blob`.
- Instances (`DType` f32, `IdType` i32, 1024 threads, 4-wide vectors, the
  deterministic scan and the fixed-point multi-CTA top-k sum):
  `OnlineSoftmaxFusedKernel`, `RadixTopKRenormProbKernel_MultiCTA`,
  `TopPRenormProbKernel`, `MinPSamplingFromProbKernel`. Entries are the
  mangled names `cuobjdump` lists; `build.sh` prints them.
- Rebuild: `FLASHINFER_INCLUDE=<flashinfer v0.7.2>/include tools/flashinfer-sampling/build.sh <out.cubin>`,
  then `tools/kernels/import_cubin.py --family flashinfer_sampling --cubin <out.cubin> …`.

## The chain a manifest runs

Per step, every parameter a per-row array, every kernel in place on the
`[rows, vocab]` f32 probs buffer the lm_head writes:

1. `OnlineSoftmaxFusedKernel(probs, probs, temperature[rows], 1.0, vocab)`,
   grid `rows`, dynamic smem `sizeof(OnlineSoftmaxTempStorage<1024>)`.
2. `RadixTopKRenormProbKernel_MultiCTA(probs, probs, top_k[rows], 0, vocab, rows, states, chunk, ctas)`,
   grid `groups × ctas`. `chunk`, `ctas` and the smem are what
   `RadixTopKRenormProbMultiCTA` derives from the vocab and the GPU's smem
   opt-in (`probe` prints them); `groups` is at most SMs / `ctas`, and the
   kernel loops rows past it. `states` is a scratch of `groups`
   `RadixRowState`, zeroed at load; the kernel leaves it reusable. A
   `top_k` of the vocab keeps every token.
3. `TopPRenormProbKernel(probs, probs, top_p[rows], 0.0, vocab)`, grid `rows`,
   smem `sizeof(RenormTempStorage<1024, REDUCE_ALGO>)`. A `top_p` of 1 keeps
   every token.
4. `MinPSamplingFromProbKernel(probs, min_p[rows], tokens, valid, null, 0.0, vocab, seed[rows], 0, offset[rows], 0, 1, 1)`,
   grid `rows`, smem `sizeof(SamplingTempStorage<1024, SCAN_ALGO, REDUCE_ALGO>)`.
   A `min_p` of 0 samples from what is left. `tokens` is i32 per row.

Temperature 0 is not the chain's to serve: the softmax divides by it. A
greedy row goes through the manifest's `argmax` and a select picks it over
the chain's token.

## What the seed means

The kernels draw from Philox with `curand_init(seed[row], blockIdx.x, offset[row])`:
the row's place in the batch is the subsequence. Two runs of the same batch
give the same tokens; the same request at another row of another batch does
not. kern-serve stages `seed` from the request (one drawn at admission
without) and `offset` from the row's position, and promises replay at the
same batch layout, nothing across layouts.

`probe.cu` drives the cubin as the runtime does (driver API, entries by name)
and checks the kept set, replay, the distribution of one row over many
offsets, and the time per step.
