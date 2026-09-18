// K3's derived weight tables, computed once after load from the HF
// checkpoint's own tensors (docs/manifest.md "权重": what the checkpoint
// does not hold is a carry the `load` program fills). Elementwise; one
// thread per output element, grid.x = ceil(n / 256), block 256.
//
//   kern_fill_bf16       p[i] = v (the router's unit scale)
//   kern_k3_conv_taps    cw[s][t][j] = w_s[j][t]: the three conv1d weights
//                        [inner, 1, 4] f32 transposed into the taps block
//                        kern_k3_conv_silu / kern_k3_span_gather read
//   kern_k3_scoring      sw[j] = f32(norm[j]) * f32(proj[j]): the folded
//                        attention-residual scoring vector
//   kern_k3_wsm          the KDA small-projection block [256, h]: rows
//                        0..heads = b_proj (this rank's heads), rows 96..224
//                        = f_a_proj, the rest zero (the kernels' layout,
//                        docs/k3-kernel-abi.md)
//   kern_k3_mega_sf_pack MegaMoE's scale-factor tensor from the checkpoint's
//                        [n, k/32] UE8M0 bytes per expert: rows permuted
//                        (UTCCP transpose, then gate/up interleaved by 8 for
//                        L1), four bytes packed LSB-first into one i32 per
//                        128 elements, laid out [k/128, n] (pegainfer's
//                        transform_weights_for_mega_moe, tools/gen_k3_moe.py)
//   kern_k3_kvb_aug      the MLA prefill v2 expansion weight [96 * 320, 576]
//                        from kv_b_proj [96 * 256, 512]: per head 128 rows of
//                        W_UK, 64 rows selecting the latent row's rope part
//                        (an identity onto columns 512..576) and 128 rows of
//                        W_UV, each padded to 576 with zeros, so one GEMM of
//                        the latent rows gives k (192) | v (128) per head
//
//   nvcc -cubin -arch=sm_103a -O3 tools/kernels-src/k3_weight_prep.cu
#include <cuda_bf16.h>

extern "C" __global__ void kern_fill_bf16(__nv_bfloat16* __restrict__ p, int n, float v) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = __float2bfloat16_rn(v);
}

extern "C" __global__ void kern_k3_conv_taps(const float* __restrict__ q, const float* __restrict__ k,
                                             const float* __restrict__ v, float* __restrict__ cw, int inner) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= inner) return;
  const float* w[3] = {q, k, v};
#pragma unroll
  for (int s = 0; s < 3; ++s)
#pragma unroll
    for (int t = 0; t < 4; ++t) cw[(s * 4 + t) * inner + j] = w[s][j * 4 + t];
}

extern "C" __global__ void kern_k3_scoring(const __nv_bfloat16* __restrict__ norm,
                                           const __nv_bfloat16* __restrict__ proj, float* __restrict__ sw, int n) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < n) sw[j] = __bfloat162float(norm[j]) * __bfloat162float(proj[j]);
}

extern "C" __global__ void kern_k3_wsm(const __nv_bfloat16* __restrict__ b, const __nv_bfloat16* __restrict__ f_a,
                                       __nv_bfloat16* __restrict__ wsm, int heads, int h) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 256 * h) return;
  const int r = i / h, c = i % h;
  wsm[i] = r < heads ? b[r * h + c] : (r >= 96 && r < 224) ? f_a[(r - 96) * h + c] : __float2bfloat16_rn(0.f);
}

// Output row r of the packed table reads source row utccp(r), or for L1
// utccp then the gate/up interleave (row r of [gate; up] interleaved by 8
// is row (r / 16) * 8 + r % 8 of gate, or the same of up when bit 3 is set).
__device__ __forceinline__ int sf_src_row(int r, int n, int interleave) {
  const int u = (r / 128) * 128 + (r % 4) * 32 + (r % 128) / 4;
  if (!interleave) return u;
  return ((u >> 3) & 1 ? n / 2 : 0) + (u >> 4) * 8 + (u & 7);
}

extern "C" __global__ void kern_k3_mega_sf_pack(const unsigned char* __restrict__ raw, int* __restrict__ out,
                                                int experts, int n, int kg, int interleave) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const int words = kg / 4;
  if (i >= (long long)experts * words * n) return;
  const int r = (int)(i % n), w = (int)(i / n % words), e = (int)(i / n / words);
  const unsigned char* s = raw + ((long long)e * n + sf_src_row(r, n, interleave)) * kg + w * 4;
  out[i] = (int)((unsigned)s[0] | ((unsigned)s[1] << 8) | ((unsigned)s[2] << 16) | ((unsigned)s[3] << 24));
}

extern "C" __global__ void kern_k3_kvb_aug(const __nv_bfloat16* __restrict__ w_kv_b, __nv_bfloat16* __restrict__ out,
                                           int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int r = i / 576, c = i % 576, h = r / 320, d = r % 320;
  float v = 0.f;
  if (d < 128 || d >= 192) {
    if (c < 512) v = __bfloat162float(w_kv_b[(h * 256 + (d < 128 ? d : d - 64)) * 512 + c]);
  } else if (c == 512 + (d - 128)) {
    v = 1.f;
  }
  out[i] = __float2bfloat16_rn(v);
}
