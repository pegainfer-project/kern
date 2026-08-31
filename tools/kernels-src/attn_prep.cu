// The attention prep chain of one layer in one launch:
//
//   per-head Gemma RMSNorm of q and k   kern_gemma_rms_norm_bf16 x2
//   partial rotary (dim 64 of 256)      _triton_mrope_forward
//   KV cache write                      reshape_and_cache_kernel_flash
//
// One CTA per (token, head): 24 q heads norm + rope into q_n, 4 k heads
// norm + rope into k_n and copy the k/v pair into the paged KV cache.
// No cross-CTA traffic at all -- the fusion removes three graph nodes
// per layer and the q_n/k_n round trips between norm, rope and the
// cache write.
//
// Rounding is bit-exact against the chain: the norm reproduces ATen's
// reduction order (same code as gemma_rms_norm.cu, W from the original
// per-kernel `rows`), the rope operates on the bf16-rounded norm output
// with the Triton kernel's instruction shapes (product of the far
// element rounded to bf16, then a single-rounding bf16 fma with the
// near element -- HMUL2/HFMA2 in the pinned cubin), and the cache write
// is a plain bf16 copy (FP8 off, scales unused).
//
//   nvcc -cubin -arch=sm_103a -o kernels-qwen38/attn_prep.cubin tools/kernels-src/attn_prep.cu
#include <cuda_bf16.h>
#include <cstdint>

namespace {
constexpr int NT = 512;
constexpr int HEADS = 24, KV_HEADS = 4, HD = 256, ROT = 32;  // rotary_dim 64
constexpr int QKV_ROW = 14336;               // q|gate 24*512 | k 4*256 | v 4*256
constexpr int K_OFF = HEADS * 2 * HD;        // 12288
constexpr int V_OFF = K_OFF + KV_HEADS * HD; // 13312
constexpr int GROUPS = HEADS + KV_HEADS;     // CTAs per token

__device__ inline int last_pow2(int n) {
  n |= (n >> 1); n |= (n >> 2); n |= (n >> 4); n |= (n >> 8); n |= (n >> 16);
  return (n - (n >> 1)) > 0 ? (n - (n >> 1)) : 1;
}
// ATen ReduceConfig::set_block_dimension(dim0 = N/vec, dim1 = rows), mnt=512.
__device__ inline int aten_block_width(int dim0, int dim1) {
  int dim0_pow2 = dim0 < NT ? last_pow2(dim0) : NT;
  int dim1_pow2 = dim1 < NT ? last_pow2(dim1) : NT;
  int block_width = min(dim0_pow2, 32);
  int block_height = min(dim1_pow2, NT / block_width);
  block_width = min(dim0_pow2, NT / block_height);
  return block_width;
}
// Sum of sq[0..N) in ATen's order for W lanes (see gemma_rms_norm.cu).
__device__ float aten_row_sum(const float* __restrict__ sq, int N, int W,
                              float* __restrict__ s) {
  const int t = threadIdx.x;
  float value = 0.f;
  if (t < W) {
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int idx = t; idx * 4 + 3 < N; idx += W) {
      const float4 v = *reinterpret_cast<const float4*>(sq + idx * 4);
      acc[0] += v.x; acc[1] += v.y; acc[2] += v.z; acc[3] += v.w;
    }
    value = ((acc[0] + acc[1]) + acc[2]) + acc[3];
  }
  if (W > 32) {
    s[t] = value;
    for (int off = W / 2; off >= 32; off >>= 1) {
      __syncthreads();
      if (t < off && t + off < W) { value += s[t + off]; s[t] = value; }
    }
  }
  __syncthreads();
  if (t < 32) {
    for (int off = 16; off > 0; off >>= 1)
      value += __shfl_down_sync(0xffffffffu, value, off);
    if (t == 0) s[0] = value;
  }
  __syncthreads();
  return s[0];
}
}  // namespace

extern "C" __global__ void __launch_bounds__(NT) kern_attn_prep_bf16(
    __nv_bfloat16* __restrict__ q_n,          // [tokens, 6144] out (post rope)
    __nv_bfloat16* __restrict__ k_n,          // [tokens, 1024] out (post rope)
    const __nv_bfloat16* __restrict__ qkv,    // [tokens, 14336]
    const float* __restrict__ qw1,            // q_norm weight + 1
    const float* __restrict__ kw1,            // k_norm weight + 1
    const __nv_bfloat16* __restrict__ cos_g,  // [tokens, 32]
    const __nv_bfloat16* __restrict__ sin_g,  // [tokens, 32]
    __nv_bfloat16* __restrict__ kv_k,         // paged cache, k plane of this layer
    __nv_bfloat16* __restrict__ kv_v,         // paged cache, v plane (k + 256)
    const int64_t* __restrict__ slot_mapping, // [tokens]
    int rows_q, int rows_k,                   // T*24 / T*4 (ATen width inputs)
    int64_t block_stride, int64_t page_stride, int64_t head_stride,
    int block_size, float eps) {
  __shared__ float z[HD], s[NT], sq[HD];
  __shared__ __nv_bfloat16 nb[HD];
  const int tok = blockIdx.x / GROUPS, h = blockIdx.x % GROUPS;
  const bool isq = h < HEADS;
  const int kh = h - HEADS;
  const __nv_bfloat16* in =
      qkv + (int64_t)tok * QKV_ROW + (isq ? h * 2 * HD : K_OFF + kh * HD);
  const float* w1 = isq ? qw1 : kw1;

  // ---- Gemma RMSNorm, ATen reduction order, rounded to bf16 ----
  for (int j = threadIdx.x; j < HD; j += NT) {
    const float v = __bfloat162float(in[j]);
    z[j] = v;
    sq[j] = __fmul_rn(v, v);
  }
  __syncthreads();
  const int rows = isq ? rows_q : rows_k;
  const int W = aten_block_width(HD / 4, rows);
  const float sum = aten_row_sum(sq, HD, W, s);
  const float factor = (float)rows / (float)((long long)rows * (long long)HD);
  const float var = __fmul_rn(sum, factor);
  const float r = rsqrtf(__fadd_rn(var, eps));
  for (int j = threadIdx.x; j < HD; j += NT)
    nb[j] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(z[j], r), w1[j]));
  __syncthreads();

  // ---- partial neox rotary on dims [0, 64): pair (d, d+32) ----
  // Triton emits (per SASS): y1 = bf16fma(x1, cos, -bf16(x2 * sin))
  //                           y2 = bf16fma(x1, sin,  bf16(x2 * cos))
  // -- the x1 product is HFMA2-fused in both outputs, the x2 product
  // pre-rounded.
  if (threadIdx.x < ROT) {
    const int j = threadIdx.x;
    const __nv_bfloat16 c = cos_g[(int64_t)tok * ROT + j];
    const __nv_bfloat16 sn = sin_g[(int64_t)tok * ROT + j];
    const __nv_bfloat16 x1 = nb[j], x2 = nb[j + ROT];
    nb[j] = __hfma(x1, c, __hneg(__hmul(x2, sn)));
    nb[j + ROT] = __hfma(x1, sn, __hmul(x2, c));
  }
  __syncthreads();

  // ---- stores: q_n / k_n, and the paged KV cache for k heads ----
  if (isq) {
    for (int j = threadIdx.x; j < HD; j += NT)
      q_n[(int64_t)tok * (HEADS * HD) + h * HD + j] = nb[j];
    return;
  }
  for (int j = threadIdx.x; j < HD; j += NT)
    k_n[(int64_t)tok * (KV_HEADS * HD) + kh * HD + j] = nb[j];
  const int64_t slot = slot_mapping[tok];
  if (slot < 0) return;  // padding token
  const int64_t base = (slot / block_size) * block_stride +
                       (slot % block_size) * page_stride + kh * head_stride;
  const __nv_bfloat16* v_src = qkv + (int64_t)tok * QKV_ROW + V_OFF + kh * HD;
  for (int j = threadIdx.x; j < HD; j += NT) {
    kv_k[base + j] = nb[j];
    kv_v[base + j] = v_src[j];
  }
}
