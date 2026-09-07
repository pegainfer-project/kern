// Full-attention prologue for one token per block: the per-head q/k Gemma
// norms, the partial rotary embedding and the KV append, which vLLM runs as
// four kernels (two gemma_rms_norm launches, _triton_mrope_forward,
// reshape_and_cache_kernel_flash).
//
//   qkv row  [Q heads][256 q | 256 gate] [KV heads][256 k] [KV heads][256 v]
//   q_n[t, h]  = rope(norm(q_h))        k_n[t, h] = rope(norm(k_h))
//   k_cache[slot] = k_n[t]              v_cache[slot] = v (as is)
//
// Numerics, per row of 256 (bit-exact with the pinned kernels, checked in
// a standalone harness):
//   * norm: ATen's mean order, see gemma_rms_norm.cu. ATen's block width
//     for N = 256 is 32 lanes when the launch has >= 16 rows, else 64
//     virtual lanes; lane l holds elements [4l, 4l+4) and [128+4l, 128+4l+4)
//     so both widths reduce from the same registers.
//   * rope on the first `rd` = 64 dims, cos/sin gathered per token as
//     bf16 [32]; Triton lowers the bf16 arithmetic to native instructions,
//     one product rounded and the other fused:
//       x1' = fma.rn.bf16(x1, c, -mul.bf16(x2, s))
//       x2' = fma.rn.bf16(x1, s,  mul.bf16(x2, c))
//   * the cache is bf16 with unit scales; the append is a copy.
//
// One warp per head row: rows w, w + 8, .. over the 24 + 4 heads; the
// warps holding k heads also copy that head's v.
//
//   nvcc -cubin -arch=sm_103a -o kernels/attn_prep.cubin tools/kernels-src/attn_prep.cu
#include <cuda_bf16.h>

#define QH 24
#define KH 4
#define HD 256
#define RD 64
#define NW 8

__device__ static inline int last_pow2(int n) {
  n |= (n >> 1); n |= (n >> 2); n |= (n >> 4); n |= (n >> 8); n |= (n >> 16);
  return (n - (n >> 1)) > 0 ? (n - (n >> 1)) : 1;
}

// ATen ReduceConfig::set_block_dimension(dim0 = N/4, dim1 = rows), mnt=512.
__device__ static inline int aten_block_width(int dim0, int dim1) {
  const int max_threads = 512;
  int dim0_pow2 = dim0 < max_threads ? last_pow2(dim0) : max_threads;
  int dim1_pow2 = dim1 < max_threads ? last_pow2(dim1) : max_threads;
  int block_width = min(dim0_pow2, 32);
  int block_height = min(dim1_pow2, max_threads / block_width);
  block_width = min(dim0_pow2, max_threads / block_height);
  return block_width;
}

__device__ static inline void unpack4(const uint2 v, float f[4]) {
  const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
  f[0] = __bfloat162float(p[0].x);
  f[1] = __bfloat162float(p[0].y);
  f[2] = __bfloat162float(p[1].x);
  f[3] = __bfloat162float(p[1].y);
}

__device__ static inline uint2 pack4(const float f[4]) {
  uint2 v;
  __nv_bfloat162* p = reinterpret_cast<__nv_bfloat162*>(&v);
  p[0] = __floats2bfloat162_rn(f[0], f[1]);
  p[1] = __floats2bfloat162_rn(f[2], f[3]);
  return v;
}

__device__ static inline float rbf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// The bf16 instructions Triton emits for the rope; operands are bf16 bit
// patterns.
__device__ static inline unsigned short bf_bits(float v) {
  return __bfloat16_as_ushort(__float2bfloat16_rn(v));
}
__device__ static inline float bf_val(unsigned short b) { return __bfloat162float(__ushort_as_bfloat16(b)); }
__device__ static inline unsigned short bf_mul(unsigned short a, unsigned short b) {
  unsigned short r;
  asm("mul.bf16 %0, %1, %2;" : "=h"(r) : "h"(a), "h"(b));
  return r;
}
__device__ static inline unsigned short bf_neg(unsigned short a) {
  unsigned short r;
  asm("neg.bf16 %0, %1;" : "=h"(r) : "h"(a));
  return r;
}
__device__ static inline unsigned short bf_fma(unsigned short a, unsigned short b, unsigned short c) {
  unsigned short r;
  asm("fma.rn.bf16 %0, %1, %2, %3;" : "=h"(r) : "h"(a), "h"(b), "h"(c));
  return r;
}

// Sum of squares of a 256-row held as chunks a (elements 4l..) and b
// (128+4l..) per lane, in ATen's order for W = 32 or 64; lane 0's result
// broadcast to the warp.
__device__ static inline float aten_sum256(const float a[4], const float b[4], int W) {
  float value;
  if (W == 32) {
    float acc[4];
#pragma unroll
    for (int i = 0; i < 4; i++) acc[i] = (0.f + __fmul_rn(a[i], a[i])) + __fmul_rn(b[i], b[i]);
    value = ((acc[0] + acc[1]) + acc[2]) + acc[3];
  } else {
    float va[4], vb[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
      va[i] = 0.f + __fmul_rn(a[i], a[i]);
      vb[i] = 0.f + __fmul_rn(b[i], b[i]);
    }
    value = (((va[0] + va[1]) + va[2]) + va[3]) + (((vb[0] + vb[1]) + vb[2]) + vb[3]);
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) value += __shfl_down_sync(0xffffffffu, value, off);
  return __shfl_sync(0xffffffffu, value, 0);
}

// Norm one head row (256) then rope its first 64 dims; a and b are the
// lane's chunks, updated in place.
__device__ static inline void norm_rope(float a[4], float b[4], const float* __restrict__ w1,
                                        const __nv_bfloat16* __restrict__ cos_t,
                                        const __nv_bfloat16* __restrict__ sin_t, int rows, int W,
                                        float eps) {
  const int lane = threadIdx.x & 31;
  const float sum = aten_sum256(a, b, W);
  const float factor = (float)rows / (float)((long long)rows * (long long)HD);
  const float var = __fmul_rn(sum, factor);
  const float r = rsqrtf(__fadd_rn(var, eps));
  const float4 wa = *reinterpret_cast<const float4*>(w1 + 4 * lane);
  const float4 wb = *reinterpret_cast<const float4*>(w1 + 128 + 4 * lane);
  const float wav[4] = {wa.x, wa.y, wa.z, wa.w}, wbv[4] = {wb.x, wb.y, wb.z, wb.w};
#pragma unroll
  for (int i = 0; i < 4; i++) {
    a[i] = rbf(__fmul_rn(__fmul_rn(a[i], r), wav[i]));
    b[i] = rbf(__fmul_rn(__fmul_rn(b[i], r), wbv[i]));
  }
  // rope: dims [0, 32) live in chunk a of lanes 0..7, dims [32, 64) in
  // chunk a of lanes 8..15; each side fetches the other's chunk.
  float o[4];
#pragma unroll
  for (int i = 0; i < 4; i++) o[i] = __shfl_xor_sync(0xffffffffu, a[i], 8);
  if (lane < 16) {
    float c[4], s[4];
    unpack4(*reinterpret_cast<const uint2*>(cos_t + 4 * (lane & 7)), c);
    unpack4(*reinterpret_cast<const uint2*>(sin_t + 4 * (lane & 7)), s);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const unsigned short x = bf_bits(a[i]), y = bf_bits(o[i]), cb = bf_bits(c[i]), sb = bf_bits(s[i]);
      // lanes < 8 hold x1 (o = x2); lanes 8..15 hold x2 (o = x1)
      a[i] = bf_val(lane < 8 ? bf_fma(x, cb, bf_neg(bf_mul(y, sb))) : bf_fma(y, sb, bf_mul(x, cb)));
    }
  }
}

extern "C" __global__ void __launch_bounds__(NW * 32) kern_attn_prep_bf16(
    const __nv_bfloat16* __restrict__ qkv, const float* __restrict__ q_w1,
    const float* __restrict__ k_w1, const __nv_bfloat16* __restrict__ cos_g,
    const __nv_bfloat16* __restrict__ sin_g, __nv_bfloat16* __restrict__ q_n,
    __nv_bfloat16* __restrict__ k_n, __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache, const long long* __restrict__ slot_mapping,
    int tokens, float eps, int qkv_stride, long long block_stride, long long page_stride,
    long long head_stride) {
  const int t = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const __nv_bfloat16* row = qkv + (long long)t * qkv_stride;
  const __nv_bfloat16* cos_t = cos_g + t * (RD / 2);
  const __nv_bfloat16* sin_t = sin_g + t * (RD / 2);
  const int Wq = aten_block_width(HD / 4, tokens * QH), Wk = aten_block_width(HD / 4, tokens * KH);
  const long long slot = slot_mapping[t];
  const long long cache_at =
      slot < 0 ? -1 : (slot / 64) * block_stride + (slot % 64) * page_stride;
  for (int r = warp; r < QH + KH; r += NW) {
    const bool isq = r < QH;
    const int h = isq ? r : r - QH;
    const __nv_bfloat16* src = isq ? row + h * (2 * HD) : row + QH * 2 * HD + h * HD;
    float a[4], b[4];
    unpack4(*reinterpret_cast<const uint2*>(src + 4 * lane), a);
    unpack4(*reinterpret_cast<const uint2*>(src + 128 + 4 * lane), b);
    norm_rope(a, b, isq ? q_w1 : k_w1, cos_t, sin_t, tokens * (isq ? QH : KH), isq ? Wq : Wk, eps);
    const uint2 pa = pack4(a), pb = pack4(b);
    __nv_bfloat16* dst = isq ? q_n + (long long)t * (QH * HD) + h * HD : k_n + (long long)t * (KH * HD) + h * HD;
    *reinterpret_cast<uint2*>(dst + 4 * lane) = pa;
    *reinterpret_cast<uint2*>(dst + 128 + 4 * lane) = pb;
    if (!isq && cache_at >= 0) {
      __nv_bfloat16* kc = k_cache + cache_at + h * head_stride;
      *reinterpret_cast<uint2*>(kc + 4 * lane) = pa;
      *reinterpret_cast<uint2*>(kc + 128 + 4 * lane) = pb;
      const __nv_bfloat16* v = row + QH * 2 * HD + KH * HD + h * HD;
      *reinterpret_cast<uint4*>(v_cache + cache_at + h * head_stride + 8 * lane) =
          *reinterpret_cast<const uint4*>(v + 8 * lane);
    }
  }
}
