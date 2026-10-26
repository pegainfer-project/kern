// The glue around FlashInfer's sampling chain (tools/flashinfer-sampling/PROVENANCE.md):
// the chain's per-sequence inputs from the manifest's sampling fills, and
// the pick between its draw and the argmax for a sequence that asked for
// the most likely token (temperature 0), which the chain cannot serve.
//   nvcc -cubin -arch=sm_103a -o kernels/sample_glue.cubin tools/kernels-src/sample_glue.cu
#include <cuda_bf16.h>

// grid [seqs, ceil(vocab / 1024)], block 1024: the chain works in place on
// f32, the lm_head writes bf16.
extern "C" __global__ void kern_sample_cast_bf16(const __nv_bfloat16* __restrict__ logits,
                                                 float* __restrict__ probs, int vocab) {
  int i = blockIdx.y * blockDim.x + threadIdx.x;
  if (i < vocab) {
    size_t at = (size_t)blockIdx.x * vocab + i;
    probs[at] = __bfloat162float(logits[at]);
  }
}

// grid [1], block 1024: what the chain reads per sequence. Temperature 0
// is argmax's, so the chain gets 1 and its draw is discarded; a top-k under
// 1 or past the vocab keeps every token, which the radix kernel takes as
// k = vocab; the Philox offset is the sequence's length, so every step
// draws fresh. The radix kernel's row states start each step at zero.
extern "C" __global__ void kern_sample_prep(const float* __restrict__ temperature, const int* __restrict__ top_k,
                                            const int* __restrict__ seq_len, float* __restrict__ chain_temperature,
                                            int* __restrict__ chain_top_k, unsigned long long* __restrict__ offset,
                                            unsigned* __restrict__ states, int seqs, int vocab, int state_words) {
  for (int i = threadIdx.x; i < seqs; i += blockDim.x) {
    float t = temperature[i];
    chain_temperature[i] = t > 0.f ? t : 1.f;
    int k = top_k[i];
    chain_top_k[i] = (k >= 1 && k < vocab) ? k : vocab;
    offset[i] = (unsigned long long)seq_len[i];
  }
  for (int i = threadIdx.x; i < state_words; i += blockDim.x) states[i] = 0;
}

// grid [1], block 1024: the chain's draw for a sequence that samples, the
// argmax for one that does not.
extern "C" __global__ void kern_sample_select(const float* __restrict__ temperature, const int* __restrict__ drawn,
                                              const long long* __restrict__ argmax, long long* __restrict__ tokens,
                                              int seqs) {
  for (int i = threadIdx.x; i < seqs; i += blockDim.x) {
    tokens[i] = temperature[i] > 0.f ? (long long)drawn[i] : argmax[i];
  }
}
