// K3's MoE around TRT-LLM gen's batched GEMMs (tools/kernels/abi/trtllm_bmm.py,
// docs/k3-kernel-abi.md K14): the routed activation in mxfp8, the routing
// tables the GEMMs' dynamic batch reads, the top-k combine, and the expert
// weights in the GEMMs' shuffled layout. `E` local experts of `tile`-row CTA
// tiles; ids are global expert indices, this rank's in [rank * E, rank * E + E).
//
//   kern_k3_moe_quant(const bf16* x, u8* q, u8* sf, int rows, int cols)
//   grid (ceil(rows * cols / 32 / 256), 1, 1)   block (256, 1, 1)
//   Per 32 elements: e = ceil(log2(amax / 448)) (0 when amax is 0), sf =
//   e + 127 (UE8M0), q = e4m3(x / 2^e) round-to-nearest; sf linear [rows, cols/32].
//
//   kern_k3_moe_route_count(const i32* ids, int T, int rank, int E, i32* blockcount)
//   grid (ceil(T / 256), 1, 1)   block (256, 1, 1)
//   blockcount[b][e] = the local expert e's ids among tokens [256 b, 256 b + 256).
//
//   kern_k3_moe_route_tables(const i32* blockcount, int blocks, int E, int tile,
//       i32* blockoff, i32* cta_batch, i32* cta_limit, i32* num_non_exiting,
//       i32* total_padded, i32* route_map)
//   grid (1, 1, 1)   block (1024, 1, 1)
//   count[e] = sum over blocks; ctas[e] = ceil(count[e] / tile); cta c of expert
//   e is CTA index off[e] + c with off the exclusive scan of ctas; its rows are
//   [tile * (off[e] + c), tile * (off[e] + c + 1)) of the permuted space, real
//   ones up to cta_limit = min(that end, tile * off[e] + count[e]); the padding
//   rows route token 0. blockoff[b][e] = tile * off[e] + the count of e in the
//   blocks before b: where block b's first id of e lands. num_non_exiting =
//   sum of ctas, total_padded = tile times that.
//
//   kern_k3_moe_route_scatter(const i32* ids, const i32* blockoff, int T, int rank, int E,
//       i32* route_map, i32* exp2perm)
//   grid (ceil(T / 256), 1, 1)   block (256, 1, 1)
//   The permuted row of expanded id (t, k) is blockoff[b][e] plus its rank
//   among the block's ids of e in expanded order; route_map[row] = t,
//   exp2perm[16 t + k] = row, or -1 when the expert is not local. Ranks come
//   from warp matches and a scan over the block's warp iterations, so the
//   tables are the same bytes every run.
//
//   kern_k3_moe_finalize(const bf16* fc2, const i32* exp2perm, const f32* wts, bf16* out,
//       int T, int cols)
//   grid (rows, 1, 1)   block (256, 1, 1)
//   out[t] = sum over k of wts[t][k] * fc2[exp2perm[16 t + k]] in f32, the
//   local experts only; rows t >= T (the gathered chunk's tail) are zero.
//
//   kern_k3_moe_w_shuffle(const u8* src, u8* dst, int experts, int n, int row_bytes, int gated)
//   grid (ceil(experts * n * row_bytes / 16 / 256), 1, 1)   block (256, 1, 1)
//   dst row r of every expert = src row shuffle(r): the GEMM's 32-row epilogue
//   permutation (trtllm-gen shuffleMatrixA, epilogue tile 128), after, when
//   `gated`, the gated-activation interleave that pairs src row j with row
//   n/2 + j (row 2j and 2j + 1 of the pair layout): src = [up; gate].
//
//   kern_k3_moe_sf_shuffle(const u8* src, u8* dst, int experts, int n, int kg, int gated)
//   grid (ceil(experts * n * kg / 256), 1, 1)   block (256, 1, 1)
//   The same row shuffle on the [n, kg] UE8M0 scales, re-laid in the 128x4
//   block layout (R128c4: 512 B per 128 rows x 4 columns, row r column c at
//   ((r/128) * (kg/4) + c/4) * 512 + (r%32) * 16 + ((r%128)/32) * 4 + c%4).
//
//   nvcc -cubin -arch=sm_103a -O3 tools/kernels-src/k3_moe_prefill.cu
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

#define TOPK 16
#define MAX_E 64
#define ROUTE_BLOCK 256
#define ROUTE_GROUPS (ROUTE_BLOCK / 32 * TOPK)

extern "C" __global__ void kern_k3_moe_quant(const __nv_bfloat16* __restrict__ x, uint8_t* __restrict__ q,
                                             uint8_t* __restrict__ sf, int rows, int cols) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)rows * cols / 32) return;
  const uint4* src = reinterpret_cast<const uint4*>(x + i * 32);
  float v[32];
  float amax = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint4 u = src[j];
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
    for (int l = 0; l < 4; ++l) {
      const float2 f = __bfloat1622float2(p[l]);
      v[j * 8 + l * 2] = f.x;
      v[j * 8 + l * 2 + 1] = f.y;
      amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
    }
  }
  int e = amax > 0.f ? (int)ceilf(log2f(amax / 448.f)) : 0;
  e = min(max(e, -127), 127);
  const float s = exp2f((float)-e);
  sf[i] = (uint8_t)(e + 127);
  uint8_t b[32];
#pragma unroll
  for (int j = 0; j < 32; ++j) b[j] = __nv_cvt_float_to_fp8(v[j] * s, __NV_SATFINITE, __NV_E4M3);
  uint4* dst = reinterpret_cast<uint4*>(q + i * 32);
  dst[0] = *reinterpret_cast<const uint4*>(b);
  dst[1] = *reinterpret_cast<const uint4*>(b + 16);
}

extern "C" __global__ void kern_k3_moe_route_count(const int* __restrict__ ids, int T, int rank, int E,
                                                   int* __restrict__ blockcount) {
  __shared__ int cnt[MAX_E];
  const int first = rank * E;
  if (threadIdx.x < E) cnt[threadIdx.x] = 0;
  __syncthreads();
  const int t = blockIdx.x * ROUTE_BLOCK + threadIdx.x;
  if (t < T) {
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      const int e = ids[t * TOPK + k] - first;
      if (e >= 0 && e < E) atomicAdd(&cnt[e], 1);
    }
  }
  __syncthreads();
  if (threadIdx.x < E) blockcount[blockIdx.x * E + threadIdx.x] = cnt[threadIdx.x];
}

extern "C" __global__ void kern_k3_moe_route_tables(const int* __restrict__ blockcount, int blocks, int E, int tile,
                                                    int* __restrict__ blockoff, int* __restrict__ cta_batch,
                                                    int* __restrict__ cta_limit, int* __restrict__ num_non_exiting,
                                                    int* __restrict__ total_padded, int* __restrict__ route_map) {
  __shared__ int count[MAX_E], off[MAX_E + 1];
  const int e = threadIdx.x;
  if (e < E) {
    int run = 0;
    for (int b = 0; b < blocks; ++b) {
      const int c = blockcount[b * E + e];
      blockoff[b * E + e] = run;
      run += c;
    }
    count[e] = run;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int run = 0;
    for (int j = 0; j < E; ++j) {
      off[j] = run;
      run += (count[j] + tile - 1) / tile;
    }
    off[E] = run;
    num_non_exiting[0] = run;
    total_padded[0] = run * tile;
  }
  __syncthreads();
  if (e < E) {
    for (int b = 0; b < blocks; ++b) blockoff[b * E + e] += off[e] * tile;
    // padding rows of the expert's last tile route token 0
    for (int r = off[e] * tile + count[e]; r < off[e + 1] * tile; ++r) route_map[r] = 0;
  }
  for (int i = threadIdx.x; i < off[E]; i += blockDim.x) {
    int j = 0;
    while (off[j + 1] <= i) ++j;
    cta_batch[i] = j;
    cta_limit[i] = min((i + 1) * tile, off[j] * tile + count[j]);
  }
}

extern "C" __global__ void kern_k3_moe_route_scatter(const int* __restrict__ ids, const int* __restrict__ blockoff,
                                                     int T, int rank, int E, int* __restrict__ route_map,
                                                     int* __restrict__ exp2perm) {
  __shared__ int hist[ROUTE_GROUPS][MAX_E];
  const int first = rank * E;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const unsigned below = (1u << lane) - 1;
  const long long base = (long long)blockIdx.x * ROUTE_BLOCK * TOPK;
  for (int i = threadIdx.x; i < ROUTE_GROUPS * E; i += blockDim.x) hist[i / E][i % E] = 0;
  __syncthreads();
  int ex[TOPK];
  unsigned peers[TOPK];
#pragma unroll
  for (int r = 0; r < TOPK; ++r) {
    const int g = r * (ROUTE_BLOCK / 32) + warp;
    const long long idx = base + g * 32 + lane;
    int e = -1;
    if (idx < (long long)T * TOPK) {
      e = ids[idx] - first;
      if (e < 0 || e >= E) e = -1;
    }
    ex[r] = e;
    peers[r] = __match_any_sync(0xffffffffu, e);
    if (e >= 0 && (peers[r] & below) == 0) hist[g][e] = __popc(peers[r]);
  }
  __syncthreads();
  if (threadIdx.x < E) {
    int run = 0;
    for (int g = 0; g < ROUTE_GROUPS; ++g) {
      const int c = hist[g][threadIdx.x];
      hist[g][threadIdx.x] = run;
      run += c;
    }
  }
  __syncthreads();
#pragma unroll
  for (int r = 0; r < TOPK; ++r) {
    const int g = r * (ROUTE_BLOCK / 32) + warp;
    const long long idx = base + g * 32 + lane;
    if (idx >= (long long)T * TOPK) continue;
    const int e = ex[r];
    if (e < 0) {
      exp2perm[idx] = -1;
      continue;
    }
    const int row = blockoff[blockIdx.x * E + e] + hist[g][e] + __popc(peers[r] & below);
    route_map[row] = (int)(idx / TOPK);
    exp2perm[idx] = row;
  }
}

extern "C" __global__ void kern_k3_moe_finalize(const __nv_bfloat16* __restrict__ fc2, const int* __restrict__ exp2perm,
                                                const float* __restrict__ wts, __nv_bfloat16* __restrict__ out, int T,
                                                int cols) {
  const int t = blockIdx.x;
  for (int h = threadIdx.x; h < cols / 8; h += blockDim.x) {
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    if (t < T) {
#pragma unroll
      for (int k = 0; k < TOPK; ++k) {
        const int p = exp2perm[t * TOPK + k];
        if (p < 0) continue;
        const float w = wts[t * TOPK + k];
        const uint4 u = reinterpret_cast<const uint4*>(fc2 + (long long)p * cols)[h];
        const __nv_bfloat162* v = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
          const float2 f = __bfloat1622float2(v[l]);
          acc[2 * l] += w * f.x;
          acc[2 * l + 1] += w * f.y;
        }
      }
    }
    uint4 o;
    __nv_bfloat162* ov = reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
    for (int l = 0; l < 4; ++l) ov[l] = __floats2bfloat162_rn(acc[2 * l], acc[2 * l + 1]);
    reinterpret_cast<uint4*>(out + (long long)t * cols)[h] = o;
  }
}

// dst row r of the GEMM layout takes src row `shuffle(r)`
__device__ __forceinline__ int shuffle(int r, int n, int gated) {
  const int u = (r / 32) * 32 + (r % 8) * 4 + (r % 32) / 8;
  return gated ? (u % 2 ? n / 2 : 0) + u / 2 : u;
}

extern "C" __global__ void kern_k3_moe_w_shuffle(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, int experts,
                                                 int n, int row_bytes, int gated) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const int per_row = row_bytes / 16;
  if (i >= (long long)experts * n * per_row) return;
  const int c = (int)(i % per_row), r = (int)(i / per_row % n), e = (int)(i / per_row / n);
  reinterpret_cast<uint4*>(dst)[i] =
      reinterpret_cast<const uint4*>(src + ((long long)e * n + shuffle(r, n, gated)) * row_bytes)[c];
}

extern "C" __global__ void kern_k3_moe_sf_shuffle(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, int experts,
                                                  int n, int kg, int gated) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)experts * n * kg) return;
  const int c = (int)(i % kg), r = (int)(i / kg % n), e = (int)(i / kg / n);
  const long long at = ((long long)(r / 128) * (kg / 4) + c / 4) * 512 + (r % 32) * 16 + ((r % 128) / 32) * 4 + c % 4;
  dst[(long long)e * n * kg + at] = src[((long long)e * n + shuffle(r, n, gated)) * kg + c];
}
