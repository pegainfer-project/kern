// K3 prefill chunk, the tray form's row bookkeeping (docs/k3-kernel-abi.md
// K12): a chunk's rows are dealt to the tray's ranks in equal blocks of
// `own` rows, rank q holding rows [q * own, (q + 1) * own) — the block
// order NCCL's all-gather and reduce-scatter lay rows in — and the ranks
// past the chunk's end hold copies of its last row, so every row a rank
// computes on is a real one.
//
//   extern "C" __global__ void kern_k3_rank_rows(
//       void* dst, const void* src, int rank, int own, int row_bytes, int rows);
//   grid (own, 1, 1)   block (256, 1, 1)
//   dst[j] = src[min(rank * own + j, rows - 1)], rows of `row_bytes` bytes.
//
//   extern "C" __global__ void kern_k3_zero_rows(void* buf, int row_bytes, int from, int to);
//   grid (1, 1, 1)   block (1024, 1, 1)
//   Rows [from, to) of `buf` zeroed: the tail of a chunk-wide buffer past
//   the chunk, rows a reduce-scatter sums but nobody computes.
//
//   extern "C" __global__ void kern_k3_fmha_lens(const int* seq_lens, int* lens, int T);
//   grid (1, 1, 1)   block (32, 1, 1)
//   lens = {kv_len, 0, 0, T, 0, kv_len, 0, 0}: seq_lens_kv[1] at byte 0,
//   cum_seq_lens_q[2] at 8, cum_seq_lens_kv[2] at 16, for a chunk of T rows
//   ending at token seq_lens[0] of its sequence; the FMHA's causal mask is
//   end-aligned, so row i sees tokens up to kv_len - T + i.
//
//   nvcc -cubin -arch=sm_103a -O3 tools/kernels-src/k3_prefill.cu
#include <cstdint>

extern "C" __global__ void kern_k3_rank_rows(void* __restrict__ dst, const void* __restrict__ src, int rank, int own,
                                             int row_bytes, int rows) {
  const int j = blockIdx.x;
  const int r = min(rank * own + j, rows - 1);
  const char* s = static_cast<const char*>(src) + (size_t)r * row_bytes;
  char* d = static_cast<char*>(dst) + (size_t)j * row_bytes;
  if (row_bytes % 16 == 0) {
    for (int i = threadIdx.x; i < row_bytes / 16; i += blockDim.x)
      reinterpret_cast<uint4*>(d)[i] = reinterpret_cast<const uint4*>(s)[i];
  } else {
    for (int i = threadIdx.x; i < row_bytes; i += blockDim.x) d[i] = s[i];
  }
}

extern "C" __global__ void kern_k3_zero_rows(void* __restrict__ buf, int row_bytes, int from, int to) {
  if (to <= from) return;
  uint4* p = reinterpret_cast<uint4*>(static_cast<char*>(buf) + (size_t)from * row_bytes);
  const size_t n = (size_t)(to - from) * row_bytes / 16;
  for (size_t i = threadIdx.x; i < n; i += blockDim.x) p[i] = make_uint4(0, 0, 0, 0);
}

extern "C" __global__ void kern_k3_fmha_lens(const int* __restrict__ seq_lens, int* __restrict__ lens, int T) {
  if (threadIdx.x != 0) return;
  const int kv_len = seq_lens[0];
  lens[0] = kv_len;
  lens[1] = 0;
  lens[2] = 0;
  lens[3] = T;
  lens[4] = 0;
  lens[5] = kv_len;
  lens[6] = 0;
  lens[7] = 0;
}
