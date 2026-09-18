// K3 MLA decode, split plan: how many KV splits each row's attention runs
// as (docs/k3-kernel-abi.md K5b). Decided on the GPU from the sequence
// lengths, so the step never waits on the host.
//
//   extern "C" __global__ void kern_k3_mla_split_plan(
//       const i32* seq_lens,         // [B]  includes the current token
//       i32*       block_split_kvs,  // [B]  splits for row b, 1 .. split_max
//       int split_max, int B);
//
//   grid (1, 1, 1)   block (1024, 1, 1)   smem 0 dynamic
//
// The attention kernel walks a row in 128-token tiles and runs each split as
// one 2-CTA cluster; a GPU holds nsm/2 clusters at once. Rows get tiles in
// proportion: `per` tiles per cluster such that the batch fills one wave of
// clusters (fewer, longer splits beat a second wave), then row b takes
// ceil(tiles_b / per) splits, clamped to 1 .. split_max.
//
//   nvcc -cubin -arch=sm_103a -O3 tools/kernels-src/k3_mla_split_plan.cu
#define TILE 128

__device__ __forceinline__ unsigned nsm() {
  unsigned n;
  asm("mov.u32 %0, %%nsmid;" : "=r"(n));
  return n;
}

extern "C" __global__ void __launch_bounds__(1024) kern_k3_mla_split_plan(const int* __restrict__ seq_lens,
                                                                         int* __restrict__ block_split_kvs,
                                                                         int split_max, int B) {
  __shared__ long long warp_sum[32];
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
  long long mine = 0;
  for (int b = t; b < B; b += 1024) mine += (seq_lens[b] + TILE - 1) / TILE;
#pragma unroll
  for (int o = 16; o; o >>= 1) mine += __shfl_xor_sync(0xffffffffu, mine, o);
  if (lane == 0) warp_sum[warp] = mine;
  __syncthreads();
  long long total = 0;
#pragma unroll
  for (int w = 0; w < 32; ++w) total += warp_sum[w];
  const long long clusters = nsm() / 2;
  long long budget = clusters - B;
  if (budget < 1) budget = 1;
  long long per = (total + budget - 1) / budget;
  if (per < 1) per = 1;
  for (int b = t; b < B; b += 1024) {
    const long long tiles = (seq_lens[b] + TILE - 1) / TILE;
    long long s = (tiles + per - 1) / per;
    if (s < 1) s = 1;
    if (s > split_max) s = split_max;
    block_split_kvs[b] = (int)s;
  }
}
