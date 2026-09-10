// FP32 target / Markov-biased greedy head. Embedding and tie-breaking reuse
// the existing row-strided kernels. Set bias_stride=0 with a zero vocabulary
// vector for a target-only head.
// Row-strided variants of the embedding gather and the two-stage argmax, for
// a per-sequence chain over a batch: step t of a block-of-7 draft touches
// rows {s*7 + t | s < seqs} of a [seqs*7, V] logits buffer and element
// s*7 + t of a [seqs, 7] token buffer. The caller bakes the step's byte
// offset into the buffer argument and passes the row stride in elements;
// grid.x is the sequence index. Stride 1 / n reproduces the unstrided
// kernels exactly (same reduction order), so seqs = 1 is the old chain.
//   nvcc -cubin -arch=sm_103a -o target/cubins/markov_rows.cubin tools/kernels-src/markov_rows.cu
#include <cuda_bf16.h>
#include <limits.h>

struct Pair {
  float v;
  int i;
};

__device__ static inline Pair pick(Pair a, Pair b) {
  return (b.v > a.v || (b.v == a.v && b.i < a.i)) ? b : a;
}

template <int BLOCK>
__device__ static inline Pair block_reduce(Pair best) {
  __shared__ float sv[BLOCK];
  __shared__ int si[BLOCK];
  sv[threadIdx.x] = best.v;
  si[threadIdx.x] = best.i;
  __syncthreads();
  for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      Pair o = {sv[threadIdx.x + stride], si[threadIdx.x + stride]};
      Pair m = pick({sv[threadIdx.x], si[threadIdx.x]}, o);
      sv[threadIdx.x] = m.v;
      si[threadIdx.x] = m.i;
    }
    __syncthreads();
  }
  return {sv[0], si[0]};
}

// grid [rows], block 256: out[r] = table[ids[r * id_stride]], rows of
// `hidden` elements, `out` contiguous.
extern "C" __global__ void kern_embedding_rows_i64_bf16(
    const long long* __restrict__ ids, int id_stride,
    const __nv_bfloat16* __restrict__ table, __nv_bfloat16* __restrict__ out,
    int hidden) {
  long long row = ids[(long long)blockIdx.x * id_stride];
  const __nv_bfloat16* src = table + row * hidden;
  __nv_bfloat16* dst = out + (long long)blockIdx.x * hidden;
  for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
    dst[j] = src[j];
  }
}

// grid [rows, NB], block 1024: block (r, b) scans chunk b of row r, which
// starts at logits + r * row_stride; writes pmax/pidx[r * NB + b].
extern "C" __global__ void kern_argmax_rows_partial_f32_bias(
    const float* __restrict__ logits, long long row_stride,
    const float* __restrict__ bias, long long bias_stride,
    float* __restrict__ pmax, int* __restrict__ pidx, int n) {
  int nb = gridDim.y;
  int chunk = (n + nb - 1) / nb;
  int lo = blockIdx.y * chunk;
  int hi = min(lo + chunk, n);
  const float* row = logits + (long long)blockIdx.x * row_stride;
  Pair best = {-__builtin_inff(), INT_MAX};
  for (int j = lo + threadIdx.x; j < hi; j += blockDim.x) {
    float value = row[j];
    if (bias) value += bias[(long long)blockIdx.x * bias_stride + j];
    best = pick(best, {value, j});
  }
  best = block_reduce<1024>(best);
  if (threadIdx.x == 0) {
    pmax[blockIdx.x * nb + blockIdx.y] = best.v;
    pidx[blockIdx.x * nb + blockIdx.y] = best.i;
  }
}

// grid [rows], block 64 (= NB): merge row r's partials, out[r * out_stride].
extern "C" __global__ void kern_argmax_rows_final_i64(
    const float* __restrict__ pmax, const int* __restrict__ pidx,
    long long* __restrict__ out, int out_stride, int nb) {
  Pair best = {-__builtin_inff(), INT_MAX};
  if (threadIdx.x < nb) {
    best = {pmax[blockIdx.x * nb + threadIdx.x],
            pidx[blockIdx.x * nb + threadIdx.x]};
  }
  best = block_reduce<64>(best);
  if (threadIdx.x == 0) out[(long long)blockIdx.x * out_stride] = best.i;
}
