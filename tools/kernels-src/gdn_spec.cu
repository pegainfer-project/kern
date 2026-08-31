// The speculative-path GDN chain of one layer in one launch (tokens <= 8):
//
//   causal_conv1d_update      (IS_SPEC_DECODING,    _causal_conv1d_update_kernel
//                              width 4, silu, state
//                              roll from slot nacc-1)
//   fused post-conv prep      (split + l2norm +     _fused_post_conv_kernel
//                              gating)
//   a/b contiguous copies                           kern_copy_rows_bf16 x2
//   gated delta rule, T rows  (l2norm again, per-   fused_sigmoid_gating_delta_rule_update_kernel
//                              token SSM checkpoint
//                              to spec_slots[t])
//   z copy                                          kern_copy_rows_bf16
//   gated RMSNorm             (norm, then *silu(z)) layer_norm_fwd_kernel
//
// The pipeline the manifest pins normalizes q/k twice: the post_conv
// instance is the prefill one (APPLY_L2NORM, 1/sqrt form) and the
// recurrent kernel was captured with USE_QK_L2NORM_IN_KERNEL (rsqrt
// form); the fused kernel reproduces both, feeding the bf16-rounded
// first-pass values into the second, like the buffers did.
//
// Four CTAs per v head (192 total, co-resident) stream 32 state rows
// each and redundantly convolve + normalize the 256 q/k channels of
// their k head from registers/smem -- no producer/consumer chain.  The
// only synchronization: a per-k-head staging counter before the conv
// writeback may overwrite qkvz (twelve sibling CTAs read those raw
// columns), and the fixed-order variance merge of the gated norm.
//
// Rounding chains reproduce the Triton kernels bit-for-bit where the
// order is defined (bf16 x*w conv products rounded before the f32
// accumulate; silu -> bf16 store; the post_conv l2norm divides by
// tl.sqrt and rounds to bf16; the recurrent reloads that value,
// multiplies by tl.rsqrt of its own sum, then by scale; its softplus is
// the one-sided log(1+exp) form while the g/beta buffers keep the
// two-sided post_conv form; state updates go through fmaf like the
// Triton fma; the norm input is the bf16-rounded recurrent output).
// Only the sum orders of the wide reductions differ.
//
// Buffers past the cut carry every write of the chain: qkvz conv
// writeback, the conv state roll, the per-token SSM checkpoints,
// gdn_q/k/v, g/beta, a_c/b_c/z_c and the normed core_attn_out.
//
// All counter scratch is zero at rest: alloc_zeros at load, the last
// CTA resets (hpart holds data, write-before-read behind the counters).
//
//   nvcc -cubin -arch=sm_103a -o kernels-qwen38-dflash2/gdn_spec.cubin tools/kernels-src/gdn_spec.cu
#include <cuda_bf16.h>
#include <cstdint>

namespace {
constexpr int HV = 48, K = 128, V = 128;
constexpr int CONV_DIM = 10240;             // q | k | v channels
constexpr int Z_OFF = CONV_DIM;             // z columns after q|k|v
constexpr int QKVZ = 16384, BA = 96;
constexpr int KOFF = 2048, VOFF = 4096;
constexpr int NT = 256;
constexpr int SPLIT = 4, ROWS = V / SPLIT;  // CTAs per v head, state rows per CTA
constexpr int NB = SPLIT * HV;
constexpr int TMAX = 8;                     // SPEC_BLOCK
constexpr float L2EPS = 1e-6f;

__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ float bf16r(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
// Triton lowers float division to rcp.approx + mul and tl.sqrt to
// sqrt.approx (see the pinned cubins' SASS); replicate the instructions.
__device__ __forceinline__ float rcpa(float x) {
  float y;
  asm("rcp.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float sqrta(float x) {
  float y;
  asm("sqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float sigmoidf_(float x) { return rcpa(1.0f + expf(-x)); }
__device__ __forceinline__ float ldcg(const float* p) { return __ldcg(p); }

struct SmemT {
  __nv_bfloat16 cq[TMAX][K], ck[TMAX][K];   // conv outputs (raw x rows past T)
  __nv_bfloat16 nq[TMAX][K], nk[TMAX][K];   // post_conv l2norm'd values
  __nv_bfloat16 cv[TMAX][ROWS];             // v conv outputs (this CTA's rows)
  __nv_bfloat16 oarr[TMAX][ROWS];           // recurrent output rows
  float s1q[TMAX], s1k[TMAX];               // post_conv sums of squares
  float s2q[TMAX], s2k[TMAX];               // recurrent sums of the rounded values
  float red[TMAX][8];                       // warp partials for the sums
  float gg[TMAX], bb[TMAX];                 // exp(g) and beta, recurrent form
};
}  // namespace

// scratch: xcnt i32 [16] per-k-head staging counters, hpart f32
// [TMAX*HV*SPLIT] variance partials, hcnt i32 [HV] arrive counters,
// gcnt i32 [1] exit counter for the reset.
extern "C" __global__ void __launch_bounds__(NT) kern_gdn_spec_bf16(
    __nv_bfloat16* __restrict__ qkvz,          // [tokens, 16384] inout
    const __nv_bfloat16* __restrict__ w,       // [10240, 4]
    __nv_bfloat16* __restrict__ conv_state,    // state pool, conv area
    float* __restrict__ ssm_state,             // state pool, ssm area
    const int* __restrict__ line_p,            // conv line page id
    const int* __restrict__ nacc_p,            // num accepted (resume slot + 1)
    const int* __restrict__ cu,                // cu_seqlens_q [2]
    const int* __restrict__ slots,             // 8 checkpoint page ids
    const __nv_bfloat16* __restrict__ ba,      // [tokens, 96] b | a
    const float* __restrict__ A_log,           // [48]
    const __nv_bfloat16* __restrict__ dt_bias, // [48]
    __nv_bfloat16* __restrict__ out,           // core_attn_out [tokens, 6144]
    const __nv_bfloat16* __restrict__ nw,      // norm.weight [128]
    __nv_bfloat16* __restrict__ gdn_q,         // [tokens, 2048]
    __nv_bfloat16* __restrict__ gdn_k,         // [tokens, 2048]
    __nv_bfloat16* __restrict__ gdn_v,         // [tokens, 6144]
    float* __restrict__ g_buf,                 // [tokens, 48]
    float* __restrict__ beta_buf,              // [tokens, 48]
    __nv_bfloat16* __restrict__ a_c,           // [tokens, 48]
    __nv_bfloat16* __restrict__ b_c,           // [tokens, 48]
    __nv_bfloat16* __restrict__ z_c,           // [tokens, 6144]
    int* __restrict__ xcnt, float* __restrict__ hpart,
    int* __restrict__ hcnt, int* __restrict__ gcnt,
    float scale, float eps, int tokens,
    int nlines, int cls, int cds, int sls) {
  __shared__ SmemT s;
  const int tid = threadIdx.x;
  const int hv = blockIdx.x >> 2, sp = blockIdx.x & 3;
  const int ih = hv / 3;
  const bool qkwriter = (hv % 3) == 0;
  const int line = line_p[0], nacc = nacc_p[0];
  const int cu0 = cu[0];
  const int T = cu[1] - cu0;              // conv/recurrent extent
  const int tt = tokens;                  // post_conv/copies/norm extent
  const bool live = line > 0;

  // ---- stage: conv history + raw x for this thread's channel(s) ----
  // threads 0..127: q channel c of head ih; 128..255: k channel c.
  const int c = tid & 127;
  const int qk_ch = (tid < 128 ? ih * K : KOFF + ih * K) + c;
  const int v_ch = VOFF + hv * V + sp * ROWS + (tid < ROWS ? tid : 0);
  float wq[4], wv[4], hq[3], hvv[3];
  __nv_bfloat16 xq[TMAX], xv[TMAX];
  for (int j = 0; j < 4; j++) {
    wq[j] = __bfloat162float(w[qk_ch * 4 + j]);
    wv[j] = __bfloat162float(w[v_ch * 4 + j]);
  }
  // spec-pool conv state layout is token-major: [conv_len rows][CONV_DIM]
  const int64_t cbase_qk = (int64_t)line * cls + qk_ch;
  const int64_t cbase_v = (int64_t)line * cls + v_ch;
  for (int j = 0; j < 3; j++) {
    hq[j] = live ? __bfloat162float(conv_state[cbase_qk + (int64_t)(nacc - 1 + j) * CONV_DIM]) : 0.f;
    hvv[j] = live ? __bfloat162float(conv_state[cbase_v + (int64_t)(nacc - 1 + j) * CONV_DIM]) : 0.f;
  }
#pragma unroll
  for (int t = 0; t < TMAX; t++) {
    if (t >= tt) break;
    xq[t] = qkvz[(int64_t)(cu0 + t) * QKVZ + qk_ch];
    xv[t] = qkvz[(int64_t)(cu0 + t) * QKVZ + v_ch];
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) atomicAdd(&xcnt[ih], 1);

  // initial SSM state, prefetched early so the cold load overlaps the conv
  const int row = tid >> 3, lane8 = tid & 7, c0 = lane8 * 16;
  const int page0 = slots[nacc - 1];
  const bool rec = page0 > 0;
  const int64_t hb = (int64_t)hv * V * K + (int64_t)(sp * ROWS + row) * K + c0;
  float bh[16];
  if (rec) {
    const float4* h0 = (const float4*)(ssm_state + (int64_t)page0 * sls + hb);
#pragma unroll
    for (int j = 0; j < 4; j++) *(float4*)&bh[j * 4] = h0[j];
  }

  // ---- conv (Triton: f32 acc of bf16-rounded products, then silu) ----
  auto conv1 = [&](const float* h, const float* wt, const __nv_bfloat16* x, int t) {
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int i = t + j;  // window over [h0 h1 h2 x0 x1 ...]
      const float xf = i < 3 ? h[i] : __bfloat162float(x[i - 3]);
      acc = __fadd_rn(acc, bf16r(__fmul_rn(xf, wt[j])));
    }
    return __float2bfloat16_rn(__fmul_rn(acc, rcpa(1.0f + expf(-acc))));
  };
  {
    __nv_bfloat16* dst = tid < 128 ? &s.cq[0][0] : &s.ck[0][0];
#pragma unroll
    for (int t = 0; t < TMAX; t++) {
      if (t >= tt) break;
      dst[t * K + c] = (live && t < T) ? conv1(hq, wq, xq, t) : xq[t];
      if (tid < ROWS) s.cv[t][tid] = (live && t < T) ? conv1(hvv, wv, xv, t) : xv[t];
    }
  }
  // gating scalars for this v head (both softplus forms)
  if (tid < tt) {
    const int t = tid;
    const float bv = __bfloat162float(ba[(int64_t)t * BA + hv]);
    const float av = __bfloat162float(ba[(int64_t)t * BA + 48 + hv]);
    const float x = av + __bfloat162float(dt_bias[hv]);
    const float ea = expf(A_log[hv]);
    // recurrent form: (1/beta) * log(1 + exp(beta*x)), beta = 1
    const float spr = x <= 20.f ? 1.0f * logf(1.0f + expf(x)) : x;
    s.gg[t] = expf(-ea * spr);
    s.bb[t] = sigmoidf_(bv);
    if (sp == 0) {
      // post_conv form: two-sided stable softplus
      float spp = x > 0.f ? x + logf(1.0f + expf(-x)) : logf(1.0f + expf(x));
      if (x > 20.f) spp = x;
      g_buf[(int64_t)t * HV + hv] = -ea * spp;
      beta_buf[(int64_t)t * HV + hv] = sigmoidf_(bv);
      a_c[(int64_t)t * HV + hv] = ba[(int64_t)t * BA + 48 + hv];
      b_c[(int64_t)t * HV + hv] = ba[(int64_t)t * BA + hv];
    }
  }
  __syncthreads();

  // ---- sums of squares of the conv outputs (post_conv l2norm) ----
  auto sumsq = [&](const __nv_bfloat16* v128, int t, int part) {
    // 128 threads (4 warps): warp-tree then fixed-order partial merge
    const float x = __bfloat162float(v128[c]);
    float p = __fmul_rn(x, x);
    for (int d = 16; d > 0; d >>= 1) p += __shfl_xor_sync(0xffffffffu, p, d);
    if ((tid & 31) == 0) s.red[t][part * 4 + ((tid >> 5) & 3)] = p;
  };
  for (int t = 0; t < tt; t++) sumsq(tid < 128 ? s.cq[t] : s.ck[t], t, tid < 128 ? 0 : 1);
  __syncthreads();
  if (tid < 2 * tt) {
    const int t = tid >> 1;
    const float* r = &s.red[t][(tid & 1) * 4];
    (tid & 1 ? s.s1k : s.s1q)[t] = (r[0] + r[1]) + (r[2] + r[3]);
  }
  __syncthreads();
  {
    __nv_bfloat16* dst = tid < 128 ? &s.nq[0][0] : &s.nk[0][0];
    const __nv_bfloat16* src = tid < 128 ? &s.cq[0][0] : &s.ck[0][0];
    const float* ss = tid < 128 ? s.s1q : s.s1k;
    for (int t = 0; t < tt; t++) {
      const float inv = rcpa(sqrta(ss[t] + L2EPS));
      dst[t * K + c] = __float2bfloat16_rn(__bfloat162float(src[t * K + c]) * inv);
    }
  }
  __syncthreads();
  // sums of squares of the rounded values (recurrent l2norm)
  for (int t = 0; t < tt; t++) sumsq(tid < 128 ? s.nq[t] : s.nk[t], t, tid < 128 ? 0 : 1);
  __syncthreads();
  if (tid < 2 * tt) {
    const int t = tid >> 1;
    const float* r = &s.red[t][(tid & 1) * 4];
    (tid & 1 ? s.s2k : s.s2q)[t] = (r[0] + r[1]) + (r[2] + r[3]);
  }
  __syncthreads();

  // ---- writebacks that only this CTA touches ----
  for (int t = 0; t < tt; t++) {
    if (tid < ROWS) {
      const int col = hv * V + sp * ROWS + tid;
      gdn_v[(int64_t)t * 6144 + col] = s.cv[t][tid];
      z_c[(int64_t)t * 6144 + col] = qkvz[(int64_t)t * QKVZ + Z_OFF + col];
    }
    if (tid < 128) {
      gdn_q[(int64_t)t * KOFF + ih * K + c] = s.nq[t][c];
    } else {
      gdn_k[(int64_t)t * KOFF + ih * K + c] = s.nk[t][c];
    }
  }
  // conv state roll + qkvz writeback (v channels: this CTA is the only
  // reader; q/k channels: wait for the twelve sibling CTAs to stage)
  auto roll = [&](int64_t cbase, const float* h, const __nv_bfloat16* x) {
    const bool ok = line < nlines;  // A masks the shifted loads by this
    for (int i = 0; i < T + 2 && i < cds; i++) {
      __nv_bfloat16 val;
      if (i < 2)
        val = ok ? __float2bfloat16_rn(h[1 + i]) : __float2bfloat16_rn(0.f);
      else
        val = x[i - 2];
      conv_state[cbase + (int64_t)i * CONV_DIM] = val;
    }
  };
  if (live && tid < ROWS) {
    roll(cbase_v, hvv, xv);
    for (int t = 0; t < T; t++) qkvz[(int64_t)(cu0 + t) * QKVZ + v_ch] = s.cv[t][tid];
  }
  if (live && qkwriter && (c >> 5) == sp) {
    if (tid == (sp << 5)) while (ld_acquire(&xcnt[ih]) < 3 * SPLIT) {}
    __syncwarp();
    roll(cbase_qk, hq, xq);
    for (int t = 0; t < T; t++)
      qkvz[(int64_t)(cu0 + t) * QKVZ + qk_ch] = (tid < 128 ? s.cq : s.ck)[t][c];
  }

  // ---- gated delta rule over T rows, checkpoint each row's state ----
  if (rec) {
    for (int t = 0; t < T; t++) {
      const float i2q = rsqrtf(s.s2q[t] + L2EPS), i2k = rsqrtf(s.s2k[t] + L2EPS);
      float qv[16], kv[16];
      uint4 qraw[2], kraw[2];
      qraw[0] = ((const uint4*)&s.nq[t][c0])[0]; qraw[1] = ((const uint4*)&s.nq[t][c0])[1];
      kraw[0] = ((const uint4*)&s.nk[t][c0])[0]; kraw[1] = ((const uint4*)&s.nk[t][c0])[1];
#pragma unroll
      for (int j = 0; j < 16; j++) {
        qv[j] = __bfloat162float(((const __nv_bfloat16*)qraw)[j]) * i2q * scale;
        kv[j] = __bfloat162float(((const __nv_bfloat16*)kraw)[j]) * i2k;
      }
      float dot = 0.f;
      for (int j = 0; j < 16; j++) {
        bh[j] *= s.gg[t];
        dot += __fmul_rn(bh[j], kv[j]);
      }
      for (int d = 1; d < 8; d <<= 1) dot += __shfl_xor_sync(0xffffffffu, dot, d);
      float bv = __bfloat162float(s.cv[t][row]) - dot;
      bv *= s.bb[t];
      float o = 0.f;
      for (int j = 0; j < 16; j++) {
        bh[j] = fmaf(bv, kv[j], bh[j]);
        o += __fmul_rn(bh[j], qv[j]);
      }
      for (int d = 1; d < 8; d <<= 1) o += __shfl_xor_sync(0xffffffffu, o, d);
      if (lane8 == 0) s.oarr[t][row] = __float2bfloat16_rn(o);
      const int st = slots[t];
      if (st > 0) {
        float4* ht = (float4*)(ssm_state + (int64_t)st * sls + hb);
        for (int j = 0; j < 4; j++) ht[j] = *(float4*)&bh[j * 4];
      }
    }
  }
  __syncthreads();

  // ---- gated RMSNorm, variance merged across the four sibling CTAs ----
  if (tid < ROWS) {
    for (int t = 0; t < tt; t++) {
      float x;
      if (rec && t < T)
        x = __bfloat162float(s.oarr[t][tid]);
      else
        x = __bfloat162float(out[(int64_t)t * 6144 + hv * V + sp * ROWS + tid]);
      s.oarr[t][tid] = __float2bfloat16_rn(x);  // keep the norm input
      float p = __fmul_rn(x, x);
      for (int d = 16; d > 0; d >>= 1) p += __shfl_xor_sync(0xffffffffu, p, d);
      if (tid == 0) hpart[((int64_t)t * HV + hv) * SPLIT + sp] = p;
    }
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) atomicAdd(&hcnt[hv], 1);
  if (tid == 0) while (ld_acquire(&hcnt[hv]) < SPLIT) {}
  __syncthreads();
  if (tid < ROWS) {
    for (int t = 0; t < tt; t++) {
      const float* p = hpart + ((int64_t)t * HV + hv) * SPLIT;
      const float var = ((ldcg(p) + ldcg(p + 1)) + (ldcg(p + 2) + ldcg(p + 3))) / (float)V;
      const float rstd = rsqrtf(var + eps);
      const int col = hv * V + sp * ROWS + tid;
      float y = __bfloat162float(s.oarr[t][tid]) * rstd;
      y *= __bfloat162float(nw[sp * ROWS + tid]);
      const float z = __bfloat162float(qkvz[(int64_t)t * QKVZ + Z_OFF + col]);
      y *= z * sigmoidf_(z);
      out[(int64_t)t * 6144 + col] = __float2bfloat16_rn(y);
    }
  }

  // ---- last CTA resets the counter scratch ----
  __syncthreads();
  if (tid == 0 && atomicAdd(gcnt, 1) == NB - 1) {
    for (int i = 0; i < 16; i++) xcnt[i] = 0;
    for (int i = 0; i < HV; i++) hcnt[i] = 0;
    *gcnt = 0;
    __threadfence();
  }
}
