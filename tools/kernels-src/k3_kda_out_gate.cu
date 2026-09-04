// K11 `k3_kda_out_gate` — the K3 epilogue on its own, for span rows: the
// FlashKDA span kernel (K8) writes the raw attention of the span's rows
// 0..span into `attn`, and this finishes them into the batch rows
// [*span_at, +span) of `gated` with the output rms, gamma_o and the sigmoid
// gate, with K3's landing points. Contract: docs/k3-kernel-abi.md section K11.
//
//   extern "C" __global__ void kern_k3_kda_out_gate(
//       const bf16* attn,          // [span, INNER]
//       const float* gate_partial, // [rows, KDA_FUSED]  band 3 only, rows at..at+span
//       const float* gamma_o,      // [128]
//       bf16* gated,               // [rows, INNER]  rows at..at+span written
//       const int* span_at,        // [1]  the span's first batch row
//       int span);
//
//   grid (span, HEADS, 1)   block 128   (i = blockIdx.x, h = blockIdx.y, dv = thread; b = at + i)
//
//   a       = f32(attn[i, h*128 + dv])
//   r       = rsqrt(mean_dv(a^2) + 1e-5)
//   o       = bf16(a * r * gamma_o[dv])
//   gt      = bf16(sigmoid(f32(bf16(gate_partial[b, 3*INNER + h*128 + dv]))))
//   gated[b, h*128 + dv] = bf16(f32(o) * f32(gt))
#include <cuda_bf16.h>

#ifndef HEADS
#define HEADS 96
#endif
#define K11_INNER (HEADS * 128)
#define K11_KDA_FUSED (4 * K11_INNER)
#define K11_RMS_EPS 1e-5f

typedef __nv_bfloat16 bf16;

extern "C" __global__ __launch_bounds__(128) void kern_k3_kda_out_gate(
    const bf16* __restrict__ attn, const float* __restrict__ gate_partial, const float* __restrict__ gamma_o,
    bf16* __restrict__ gated, const int* __restrict__ span_at, int span) {
  const int i = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
  const int b = span_at[0] + i;
  const float a = __bfloat162float(attn[(size_t)i * K11_INNER + (size_t)h * 128 + d]);
  __shared__ float red[4];
  float ss = a * a;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  if ((d & 31) == 0) red[d >> 5] = ss;
  __syncthreads();
  const float tot = red[0] + red[1] + red[2] + red[3];
  const float r = rsqrtf(tot * (1.0f / 128.0f) + K11_RMS_EPS);
  const bf16 o = __float2bfloat16(a * r * gamma_o[d]);
  const float g = __bfloat162float(__float2bfloat16(gate_partial[(size_t)b * K11_KDA_FUSED + 3 * K11_INNER + (size_t)h * 128 + d]));
  const bf16 gt = __float2bfloat16(1.0f / (1.0f + __expf(-g)));
  gated[(size_t)b * K11_INNER + (size_t)h * 128 + d] = __float2bfloat16(__bfloat162float(o) * __bfloat162float(gt));
}
