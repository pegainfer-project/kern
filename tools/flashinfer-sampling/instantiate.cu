// FlashInfer's sampling kernels as kern launches them: the template instances
// FlashInfer's own host launchers pick on sm_100+ (1024 threads, 4-wide f32
// vectors, the deterministic scan, the fixed-point multi-CTA top-k sum), no
// source of FlashInfer's changed. The chain a manifest runs per step, every
// parameter per row:
//   OnlineSoftmaxFusedKernel      logits / temperature -> probs, in place
//   RadixTopKRenormProbKernel     keep the top-k, renormalise, in place
//   TopPRenormProbKernel          keep the top-p mass, renormalise, in place
//   MinPSamplingFromProbKernel    sample from probs >= min_p * max, seeded per row
//
//   FLASHINFER_INCLUDE=<flashinfer 0.7.2>/include tools/flashinfer-sampling/build.sh <out.cubin>
#include <flashinfer/sampling.cuh>

namespace fs = flashinfer::sampling;

constexpr uint32_t BLOCK = 1024;
constexpr uint32_t VEC = 4;

template __global__ void fs::OnlineSoftmaxFusedKernel<BLOCK, VEC, float, false>(float*, float*, float*, float,
                                                                               uint32_t);

template __global__ void fs::RadixTopKRenormProbKernel_MultiCTA<BLOCK, VEC, false, true, float, int>(
    float*, float*, int*, uint32_t, uint32_t, uint32_t, fs::RadixRowState*, uint32_t, uint32_t);

template __global__ void fs::TopPRenormProbKernel<BLOCK, fs::REDUCE_ALGO, VEC, float>(float*, float*, float*, float,
                                                                                     uint32_t);

template __global__ void fs::MinPSamplingFromProbKernel<BLOCK, fs::SCAN_ALGO, fs::REDUCE_ALGO, VEC, true, float, int>(
    float*, float*, int*, bool*, int*, float, uint32_t, uint64_t*, uint64_t, uint64_t*, uint64_t, uint32_t, uint32_t);
