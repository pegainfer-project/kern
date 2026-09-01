// The whole GDN decode chain of one layer in one launch (tokens == 1):
//
//   causal_conv1d_update  (width 4, silu)       _causal_conv1d_update_kernel
//   gated delta rule step (l2norm q/k)          fused_recurrent_gated_delta_rule_packed_decode_kernel
//   gated RMSNorm         (norm, then *silu(z)) layer_norm_fwd_kernel
//
// The real work is one 48-head recurrent state update (48 x 128 x 128 f32,
// 3 MB each way); everything else is bandwidth-free.  One thread-block
// cluster per q/k head: GDN_SPLIT CTAs per v head, 3 v heads, so the q/k
// row is convolved and l2-normalized once (cluster rank 0) and picked up
// by the siblings over distributed shared memory after one cluster.sync()
// -- no global scratch, no flags, no spinning, nothing to reset.  Every
// CTA owns its v channels and state rows exclusively: it convolves them,
// shifts their conv state, runs its rows of the delta rule, and applies
// the gated norm (GDN_SPLIT == 2 exchanges the two variance partials over
// the same cluster).  The sibling CTAs' state loads are issued before the
// sync, so rank 0's extra conv work hides under them.
//
// Rounding chains reproduce the Triton kernels bit-for-bit where the
// order is defined: bf16 x*w products rounded before the f32 conv
// accumulate; division is rcp.approx + mul and sqrt is sqrt.approx, the
// instructions the pinned cubins actually contain (MUFU.RCP / MUFU.SQRT
// -- Triton never emits IEEE division); silu -> bf16 store; the recurrent
// re-loads that bf16 value; beta goes through bf16; the norm input is the
// bf16-rounded recurrent output; rstd is rsqrtf (MUFU.RSQ); libdevice
// expf/logf match tl.exp / tl.log.  Only the sum orders of the wide
// reductions (l2norm, variance) differ.
//
// State layout: a cache line holds the conv state [3, CONV_DIM] bf16
// (token-major: tap j of channel ch at j*cds + ch, cds = CONV_DIM) at
// offset 0 and the SSM state [HV, V, K] f32 at the ssm offset; line
// strides are passed in elements of the respective dtype.
//
//   nvcc -cubin -arch=sm_103a -o kernels-qwen38/gdn_decode.cubin tools/kernels-src/gdn_decode.cu
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace cg = cooperative_groups;

#ifndef GDN_SPLIT
#define GDN_SPLIT 2
#endif

namespace {
constexpr int HV = 48, H = 16, K = 128, V = 128;
constexpr int CONV_DIM = 10240;          // q | k | v channels
constexpr int Z_OFF = CONV_DIM;          // z columns after q|k|v in the qkvz row
constexpr int QOFF = 0, KOFF = 2048, VOFF = 4096;
constexpr int WIDTH = 4;                 // conv kernel width; state_len = 3
constexpr int NT = 256;
constexpr int SPLIT = GDN_SPLIT, ROWS = V / SPLIT;  // CTAs per v head, state rows per CTA
constexpr int CL = 3 * SPLIT;            // CTAs per cluster (= per q/k head)
constexpr float SOFTPLUS_THRESHOLD = 20.0f;

__device__ __forceinline__ float bf16r(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ float rcpa(float x) {
  float r;
  asm("rcp.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ float sqrta(float x) {
  float r;
  asm("sqrt.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ float sigmoidf_(float x) { return rcpa(1.0f + expf(-x)); }

struct SmemT {
  float qk[2 * K];      // rank 0: normalized q|k of the head; others: local copy
  float v[ROWS], o[ROWS];
  float red[8];
};
}  // namespace

extern "C" __global__ void __launch_bounds__(NT) __cluster_dims__(CL, 1, 1)
kern_gdn_decode_bf16(__nv_bfloat16* __restrict__ qkvz, const __nv_bfloat16* __restrict__ w,
                     __nv_bfloat16* __restrict__ conv_state, float* __restrict__ ssm_state,
                     const int* __restrict__ idx, const __nv_bfloat16* __restrict__ ba,
                     const float* __restrict__ A_log, const __nv_bfloat16* __restrict__ dt_bias,
                     __nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ nw,
                     float scale, float eps, int nlines, int cls, int cds, int sls) {
  __shared__ SmemT s;
  const int hv = blockIdx.x / SPLIT, sp = blockIdx.x % SPLIT;
  const int ih = hv / (HV / H), t = threadIdx.x;
  const int r0 = sp * ROWS;                       // this CTA's first state row
  const int coord = idx[0];
  cg::cluster_group cluster = cg::this_cluster();

  if (coord <= 0) {  // null cache line: conv skips, the recurrent zeroes o,
                     // and the norm output is (0 * w) * (z * sigmoid(z)) --
                     // signed zeros (or NaN from an inf z) exactly as A emits
    if (t < ROWS) {
      const int j = r0 + t;
      const float z = __bfloat162float(qkvz[Z_OFF + hv * V + j]);
      const float y = __fmul_rn(__fmul_rn(0.0f, __bfloat162float(nw[j])), __fmul_rn(z, sigmoidf_(z)));
      out[hv * V + j] = __float2bfloat16_rn(y);
    }
    return;  // uniform across the cluster: no CTA reaches a cluster.sync

  }

  // ---- issue the SSM tile loads of the first row group now: thread t owns
  // state row r0 + t/8 (+32 per pass), k columns [(t&7)*16, +16)
  const int rl = t >> 3, c0 = (t & 7) * 16;
  float* hb0 = ssm_state + (size_t)coord * sls + ((size_t)hv * V + r0 + rl) * K + c0;
  float hh[ROWS / 32][16], h[16];
#pragma unroll
  for (int pass = 0; pass < ROWS / 32; pass++)
#pragma unroll
    for (int j = 0; j < 16; j += 4)
      *reinterpret_cast<float4*>(&hh[pass][j]) =
          *reinterpret_cast<const float4*>(hb0 + (size_t)pass * 32 * K + j);

  // ---- conv: silu(w0 s0 + w1 s1 + w2 s2 + w3 x) per channel, every value
  // rounded to bf16 exactly as the in-place Triton store does
  __nv_bfloat16* cbase = conv_state + (size_t)coord * cls;
  const bool valid_line = coord < nlines;   // Triton load mask (reads 0 beyond the pool)
  auto conv1 = [&](int ch) {
    const __nv_bfloat16* cs = cbase + ch;
    const float s0 = valid_line ? __bfloat162float(cs[0]) : 0.f;
    const float s1 = valid_line ? __bfloat162float(cs[cds]) : 0.f;
    const float s2 = valid_line ? __bfloat162float(cs[2 * cds]) : 0.f;
    const float x = __bfloat162float(qkvz[ch]);
    // Triton's `acc += x * w` multiplies two bf16 tiles: the product is
    // rounded to bf16 before the f32 accumulate
    float acc = 0.f;
    acc = __fadd_rn(acc, bf16r(__fmul_rn(s0, __bfloat162float(w[ch * WIDTH + 0]))));
    acc = __fadd_rn(acc, bf16r(__fmul_rn(s1, __bfloat162float(w[ch * WIDTH + 1]))));
    acc = __fadd_rn(acc, bf16r(__fmul_rn(s2, __bfloat162float(w[ch * WIDTH + 2]))));
    acc = __fadd_rn(acc, bf16r(__fmul_rn(x, __bfloat162float(w[ch * WIDTH + 3]))));
    const float y = bf16r(__fmul_rn(acc, sigmoidf_(acc)));  // silu, division as rcp.approx
    // the channel is exclusive to its convolving CTA: shift the conv state
    // and store the conv output over the qkvz column right away
    __nv_bfloat16* csw = cbase + ch;
    csw[0] = __float2bfloat16_rn(s1); csw[cds] = __float2bfloat16_rn(s2); csw[2 * cds] = __float2bfloat16_rn(x);
    qkvz[ch] = __float2bfloat16_rn(y);
    return y;
  };
  // own v channels
  for (int j = t; j < ROWS; j += NT) s.v[j] = conv1(VOFF + hv * V + r0 + j);

  if (cluster.block_rank() == 0) {
    // q/k conv + l2norm.  q in threads 0..127, k in 128..255.
    const float y = conv1(t < K ? QOFF + ih * K + t : KOFF + ih * K + (t - K));
    s.qk[t] = y;
    __syncthreads();
    if (t < 64) {  // warps 0/1 reduce q, k
      const float* src = s.qk + (t < 32 ? 0 : K);
      const int lane = t & 31;
      float p = 0.f;
      for (int j = lane; j < K; j += 32) p += __fmul_rn(src[j], src[j]);
      for (int off = 16; off > 0; off >>= 1) p += __shfl_down_sync(0xffffffffu, p, off);
      if (lane == 0) s.red[t >> 5] = p;
    }
    __syncthreads();
    // Triton: b_q / tl.sqrt(sum + 1e-6) lowers to sqrt.approx + rcp.approx + mul
    const float iq = rcpa(sqrta(s.red[0] + 1e-6f)), ik = rcpa(sqrta(s.red[1] + 1e-6f));
    s.qk[t] = t < K ? __fmul_rn(__fmul_rn(s.qk[t], iq), scale) : __fmul_rn(s.qk[t], ik);
  }
  cluster.sync();
  if (cluster.block_rank() != 0) {  // copy the head's q/k rows to local smem
    const float* remote = cluster.map_shared_rank(s.qk, 0);
    s.qk[t] = remote[t];
    __syncthreads();
  }

  // ---- delta rule step over this CTA's ROWS state rows, 32 per pass
  const float a_val = __bfloat162float(ba[HV + hv]);
  const float b_val = __bfloat162float(ba[hv]);
  const float xg = a_val + __bfloat162float(dt_bias[hv]);
  const float sp_ = xg <= SOFTPLUS_THRESHOLD ? logf(1.0f + expf(xg)) : xg;
  const float g_e = expf(-expf(A_log[hv]) * sp_);
  const float beta = bf16r(sigmoidf_(b_val));  // Triton rounds beta through bf16

  const float* kk = s.qk + K;
  // all row groups' state tiles are already prefetched into hh (issued
  // before the conv/cluster.sync so the loads run under them)
#pragma unroll
  for (int pass = 0; pass < ROWS / 32; pass++) {
    const int row = pass * 32 + rl;
    float* hb = hb0 + (size_t)pass * 32 * K;
#pragma unroll
    for (int j = 0; j < 16; j++) h[j] = hh[pass][j];
    float dot = 0.f;
#pragma unroll
    for (int j = 0; j < 16; j++) {
      h[j] *= g_e;
      dot = __fadd_rn(dot, __fmul_rn(h[j], kk[c0 + j]));
    }
    dot += __shfl_xor_sync(0xffffffffu, dot, 1);
    dot += __shfl_xor_sync(0xffffffffu, dot, 2);
    dot += __shfl_xor_sync(0xffffffffu, dot, 4);
    const float vn = (s.v[pass * 32 + rl] - dot) * beta;
    float op = 0.f;
#pragma unroll
    for (int j = 0; j < 16; j++) {
      h[j] = __fmaf_rn(vn, kk[c0 + j], h[j]);
      op = __fadd_rn(op, __fmul_rn(h[j], s.qk[c0 + j]));
    }
    op += __shfl_xor_sync(0xffffffffu, op, 1);
    op += __shfl_xor_sync(0xffffffffu, op, 2);
    op += __shfl_xor_sync(0xffffffffu, op, 4);
#pragma unroll
    for (int j = 0; j < 16; j += 4) *reinterpret_cast<float4*>(hb + j) = *reinterpret_cast<const float4*>(&h[j]);
    if ((t & 7) == 0) s.o[row] = bf16r(op);  // Triton stores o as bf16; the norm re-loads it
  }
  __syncthreads();

  // ---- gated RMSNorm: y = (x * rstd) * w, then y *= z * sigmoid(z)
  if (t < 32) {
    float p = 0.f;
    for (int j = t; j < ROWS; j += 32) p += __fmul_rn(s.o[j], s.o[j]);
    for (int off = 16; off > 0; off >>= 1) p += __shfl_down_sync(0xffffffffu, p, off);
    if (t == 0) s.red[0] = p;
  }
#if GDN_SPLIT == 2
  cluster.sync();  // exchange the two variance partials of this v head
  if (t == 0) {
    const float* other = cluster.map_shared_rank(s.red, cluster.block_rank() ^ 1);
    const float p0 = (sp == 0) ? s.red[0] : other[0];
    const float p1 = (sp == 0) ? other[0] : s.red[0];
    s.red[1] = rsqrtf(__fadd_rn(p0, p1) / (float)V + eps);
  }
  __syncthreads();
  const float rstd = s.red[1];
#else
  __syncthreads();
  const float rstd = rsqrtf(s.red[0] / (float)V + eps);
#endif
  for (int j = t; j < ROWS; j += NT) {
    const int col = r0 + j;
    const float z = __bfloat162float(qkvz[Z_OFF + hv * V + col]);
    float y = __fmul_rn(__fmul_rn(s.o[j], rstd), __bfloat162float(nw[col]));
    y = __fmul_rn(y, __fmul_rn(z, sigmoidf_(z)));
    out[hv * V + col] = __float2bfloat16_rn(y);
  }
}
