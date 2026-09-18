// K3 MLA prefill v2 (docs/k3-kernel-abi.md K13): the sequence's cached
// latent rows laid contiguous for the kv_b expansion GEMM, and the FMHA's
// length table for this rank's rows of the chunk.
//
//   extern "C" __global__ void kern_k3_latent_gather(
//       const bf16* slab,          // the layer's latent rows in the kv state (state base + layer offset)
//       const int*  block_table,   // [max_pages]  the sequence's pages
//       long long   page_stride,   // elements between pages
//       const int*  lens,          // lens[0] = tokens to gather
//       bf16*       out,           // [n, 576]
//       int n);                    // rows written: the first lens[0] from the pages, the rest zero
//   grid (ceil(n / 8), 1, 1)   block (576, 1, 1): a row is 72 x 16 B, eight rows per block
//
//   extern "C" __global__ void kern_k3_fmha_plan(
//       const int* seq_lens,       // [1]  the chunk's last row's length
//       const int* blocks,         // [nranks + 1]  kern_k3_mla_chunk_plan's deal of the chunk
//       int*       lens,           // [8]  seq_lens_kv[1] | pad | cum_seq_lens_q[2] | cum_seq_lens_kv[2]
//       int T, int rank);
//   grid (1, 1, 1)   block (32, 1, 1)
//   This rank's rows are chunk rows [blocks[rank], blocks[rank + 1]); the last of them is token
//   seq_lens[0] - T + blocks[rank + 1] - 1 of the sequence, so its KV length is that + 1 and the
//   kernel's end-aligned causal mask gives row i the tokens up to kv_len - q_len + i.
//
//   nvcc -cubin -arch=sm_103a -O3 tools/kernels-src/k3_mla_v2.cu
#include <cuda_bf16.h>

#define LATENT_ROW 576
#define PAGE 64
#define LANES (LATENT_ROW * 2 / 16)

extern "C" __global__ void __launch_bounds__(8 * LANES) kern_k3_latent_gather(
    const __nv_bfloat16* __restrict__ slab, const int* __restrict__ block_table, long long page_stride,
    const int* __restrict__ lens, __nv_bfloat16* __restrict__ out, int n) {
  const int t = blockIdx.x * 8 + threadIdx.x / LANES;
  const int lane = threadIdx.x % LANES;
  if (t >= n) return;
  uint4 v = make_uint4(0, 0, 0, 0);
  if (t < lens[0]) {
    const __nv_bfloat16* row = slab + block_table[t / PAGE] * page_stride + (long long)(t % PAGE) * LATENT_ROW;
    v = *reinterpret_cast<const uint4*>(row + lane * 8);
  }
  *reinterpret_cast<uint4*>(out + (long long)t * LATENT_ROW + lane * 8) = v;
}

extern "C" __global__ void kern_k3_fmha_plan(const int* __restrict__ seq_lens, const int* __restrict__ blocks,
                                             int* __restrict__ lens, int T, int rank) {
  if (threadIdx.x != 0) return;
  const int kv_len = seq_lens[0] - T + blocks[rank + 1];
  lens[0] = kv_len;
  lens[1] = 0;
  lens[2] = 0;
  lens[3] = blocks[rank + 1] - blocks[rank];
  lens[4] = 0;
  lens[5] = kv_len;
  lens[6] = 0;
  lens[7] = 0;
}
