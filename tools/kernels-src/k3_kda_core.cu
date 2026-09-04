// K3 `k3_kda_core` — Kimi-K3 KDA (Kimi Delta Attention) decode step: the
// delta-rule recurrence for one row `b` and one head `h` per block, with the
// small per-head pieces (beta, the f_b projection of the forget path, the
// bf16 l2norm chain, the output rms + gate) fused in.  Contract:
// docs/k3-kernel-abi.md section K3; certified math source: pegainfer's
// TileLang `kda_core_batched` (kernels/tilelang_defs.py).
//
//   extern "C" __global__ void kern_k3_kda_core(
//       const bf16* conv_q, const bf16* conv_k, const bf16* conv_v,  // [B, INNER]
//       const f32*  wsm_partial,   // [B, WSM=256]   col h = b_proj, cols 96..223 = f_a
//       const f32*  gate_partial,  // [B, KDA_FUSED=49152], band 3 only
//       const bf16* w_f_b,         // [INNER, 128]
//       const f32*  dt_bias,       // [INNER]
//       const f32*  a_log,         // [HEADS]
//       const f32*  gamma_o,       // [128]
//       void* kda_base, const int* line_index, long long line_bytes,  // rec at offset 0
//       bf16* out,                 // [B, INNER]
//       int B, const int* span_at, int span);  // rows [*span_at, +span) are a span (K8 + K11 do them): the block returns
//
//   grid  (B, HEADS=96, 1)      (grid.x == B exactly, per the ABI; the `B`
//                               argument itself is therefore never read)
//   block 128            (one thread per output v-dim in the prologue/epilogue)
//   smem  static, 3872 B/block (no dynamic smem)
//
// Math (per block, b = blockIdx.x, h = blockIdx.y, base = h*128):
//   q,k,v   = conv_q/k/v[b, base + d]                                   (bf16)
//   qtot    = sum_d f32(bf16(q[d]*q[d]))          -- square lands in bf16, sum f32
//   qr      = bf16(rsqrt(f32(bf16(qtot)) + 1e-6))                       (same for k)
//   qs[d]   = f32(bf16(q[d]*qr)) * 128^-0.5   ;   kn[d] = f32(bf16(k[d]*kr))
//   beta    = sigmoid(f32(bf16(wsm_partial[b, h])))
//   flow[j] = bf16(wsm_partial[b, 96 + j])                              (128 values)
//   ga[d]   = sum_j f32(flow[j]) * f32(w_f_b[base + d, j])              (f32 accum)
//   raw[d]  = f32(bf16(ga[d])) + dt_bias[base + d]
//   dec[d]  = exp(LB * sigmoid(exp(a_log[h]) * raw[d])),  LB = -5
//   m[dv]   = sum_k S[dv,k]*dec[k]*kn[k]  ;  dlt[dv] = (f32(v[dv]) - m[dv]) * beta
//   S'[dv,k]= S[dv,k]*dec[k] + dlt[dv]*kn[k]                            (in place)
//   attn[dv]= bf16(sum_k S'[dv,k]*qs[k])                                (single landing)
//   out[dv] = bf16(f32(attn[dv]) * rsqrt(mean(attn^2) + 1e-5) * gamma_o[dv])
//             * bf16(sigmoid(f32(bf16(gate_partial[b, 3*INNER + base + dv]))))
//
// Landing points: q/k squares -> bf16 per term; the l2 sum -> bf16 before
// +1e-6; rsqrt -> bf16; q*qr and k*kr -> bf16; ga -> bf16 before dt_bias;
// attn -> bf16 once; the rms product -> bf16 once; the gate -> bf16 twice
// (partial and sigmoid); the state stays f32 throughout.
//
// State traffic is the whole point: rec is 96*128*128 f32 = 6.29 MB per row
// per layer, read once and written once.  The kernel therefore
//   * splits k across 8 lanes (kl = tid & 7) and dv across 16 lane groups
//     (dvl = tid >> 3), so one warp instruction covers 4 dv rows x 128 B of
//     fully contiguous, 16-B-aligned float4 -- 100% sector utilisation on
//     both the load and the store;
//   * holds one dv row's 16 k-values per lane in registers between the
//     m-reduction and the write-back, so rec is touched exactly once for
//     reading and once for writing (no second pass, no L2 re-read);
//   * hoists dec/kn/qs for the lane's fixed 16 k-values into registers ahead
//     of the dv loop (they are dv-invariant), so the inner loop issues no
//     shared-memory traffic at all;
//   * uses the identity  attn[dv] = sum_k (S*dec)*qs + dlt[dv] * (kn.qs)
//     so the second reduction rides along with the first and the row does not
//     have to be re-read after dlt is known.
//   * ROWS_PER_ITER dv rows are in flight per lane (16 B * 4 * RPI bytes of
//     outstanding loads per thread) to keep enough memory parallelism;
//   * streams rec both ways (__ldcs/__stcs): nothing re-reads it, so it must
//     not evict the w_f_b tile from L2  (+9% at B=64);
//   * remaps (blockIdx.x, blockIdx.y) to (b, h) head-fastest, so blocks issued
//     close in time walk contiguous rec instead of B lines 6.5 MB apart
//     (+8% at B=64).
// The w_f_b tile is read with the same lane split, and summed as a pairwise
// tree -- see the comment on the ga block.
//
//   nvcc -cubin -arch=sm_103a -O3 -Xptxas -v \
//        -o target/cubins/k3_kda_core.cubin tools/kernels-src/k3_kda_core.cu
#include <cuda_bf16.h>

// HEADS is the heads this rank holds: 96 whole, or a tray-group shard
// (`-DHEADS=24` for TP4, docs/multi-gpu.md "最终形态"); the per-head weights
// and the state line are then that rank's slice, the wsm partial keeps its
// whole-model layout (beta at column h, f_a at WSM_FA).
#ifndef HEADS
#define HEADS 96
#endif
#define KD 128
#define INNER (HEADS * KD)
#define KDA_FUSED (4 * INNER)
#define WSM 256
#define WSM_FA 96
#define LB (-5.0f)
#define RMS_EPS 1e-5f
#define L2_EPS 1e-6f
#define QSCALE 0.088388347648318447f /* 128^-0.5 */

#ifndef ROWS_PER_ITER
#define ROWS_PER_ITER 4
#endif
#ifndef KDA_SWIZZLE
#define KDA_SWIZZLE 1
#endif
#ifndef KDA_STREAM
#define KDA_STREAM 1
#endif

typedef __nv_bfloat16 bf16;

#define BFLO(u) __bfloat162float(__ushort_as_bfloat16((unsigned short)((u) & 0xffffu)))
#define BFHI(u) __bfloat162float(__ushort_as_bfloat16((unsigned short)((u) >> 16)))

// 8 k-lanes per dv row, 16 k-values per lane, 16 dv rows per pass.
#define KLANES 8
#define KPER (KD / KLANES) /* 16 */
#define KVEC (KPER / 4)    /* 4 float4 per lane */
#define DVGRP (128 / KLANES) /* 16 dv rows covered per pass */

#if KDA_STREAM
// rec has no reuse inside the kernel: stream both directions so the lines do
// not sit in L2 evicting the w_f_b tile.
#define LDST(p) __ldcs(p)
#define STST(p, v) __stcs(p, v)
#else
#define LDST(p) (*(p))
#define STST(p, v) (*(p) = (v))
#endif

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + __expf(-x)); }

// 128-thread (4 warp) sum.  `sm` is a 4-float scratch owned by the caller.
__device__ __forceinline__ float block_sum(float v, float* sm) {
#pragma unroll
  for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
  if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = v;
  __syncthreads();
  return (sm[0] + sm[1]) + (sm[2] + sm[3]);
}

extern "C" __global__ __launch_bounds__(128) void kern_k3_kda_core(
    const bf16* __restrict__ conv_q, const bf16* __restrict__ conv_k,
    const bf16* __restrict__ conv_v,
    const float* __restrict__ wsm_partial,
    const float* __restrict__ gate_partial,
    const bf16* __restrict__ w_f_b,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a_log,
    const float* __restrict__ gamma_o,
    void* kda_base, const int* __restrict__ line_index, long long line_bytes,
    bf16* __restrict__ out,
    int B, const int* __restrict__ span_at, int span) {
#if KDA_SWIZZLE
  // Linearise (blockIdx.x, blockIdx.y) -> (b, h) head-fastest so that blocks
  // issued close in time walk contiguous rec, instead of 64 lines 6.5 MB apart.
  const int lin = blockIdx.x + gridDim.x * blockIdx.y;
  const int b = lin / HEADS;
  const int h = lin - b * HEADS;
#else
  const int b = blockIdx.x;
  const int h = blockIdx.y;
#endif
  if ((unsigned)(b - span_at[0]) < (unsigned)span) return;
  const int d = threadIdx.x;
  const int base = h * KD;

  __shared__ float sh_qs[KD];
  __shared__ float sh_kn[KD];
  __shared__ float sh_dec[KD];
  __shared__ float sh_v[KD];
  __shared__ float sh_attn[KD];
  __shared__ float sh_flow[192];   // 8-float groups padded to 12: conflict-free 16-B LDS
  __shared__ float sh_ga[KD];
  __shared__ float sh_red[8];

  // ---- l2norm chain for q and k (all-bf16 chain, f32 sums) ----
  const bf16 qv = conv_q[(size_t)b * INNER + base + d];
  const bf16 kv = conv_k[(size_t)b * INNER + base + d];
  sh_v[d] = __bfloat162float(conv_v[(size_t)b * INNER + base + d]);

  float qsq = __bfloat162float(__hmul(qv, qv));
  float ksq = __bfloat162float(__hmul(kv, kv));
  const float qtot = block_sum(qsq, sh_red);
  __syncthreads();
  const float ktot = block_sum(ksq, sh_red + 4);

  const bf16 qr = __float2bfloat16(rsqrtf(__bfloat162float(__float2bfloat16(qtot)) + L2_EPS));
  const bf16 kr = __float2bfloat16(rsqrtf(__bfloat162float(__float2bfloat16(ktot)) + L2_EPS));
  const float qsd = __bfloat162float(__hmul(qv, qr)) * QSCALE;
  const float knd = __bfloat162float(__hmul(kv, kr));
  sh_qs[d] = qsd;
  sh_kn[d] = knd;

  // ---- beta, f_a -> f_b projection, the decay gate ----
  const float beta = sigmoidf_(__bfloat162float(__float2bfloat16(wsm_partial[(size_t)b * WSM + h])));
  sh_flow[d + ((d >> 3) << 2)] = __bfloat162float(__float2bfloat16(wsm_partial[(size_t)b * WSM + WSM_FA + d]));
  __syncthreads();

  // ga[d] = sum_j flow[j] * w_f_b[base+d, j].  This head's 128x128 bf16 tile is
  // 32 KB of *contiguous* w_f_b, shared by all B rows of the head, so it comes
  // out of L2.  Read it with the same 8-lanes-per-row split as rec: one warp
  // instruction then covers 4 rows x 128 B, where a thread-per-row walk touches
  // 32 distinct 128-B lines per instruction (8x the L1 wavefronts for the same
  // bytes) and cost 17 us of 140 at B=64.
  // Accuracy: every flow[j]*w[j] is exact in f32 (8+8 mantissa bits), so the
  // only error is the summation -- this is a full pairwise tree (4 levels in
  // the lane over fma-fused pairs, 3 by shuffle), ~7*u instead of ~128*u.  It
  // matters because ga lands in bf16 before dt_bias, and a flipped landing
  // moves dec, hence a whole dv row of the state.
  {
    const int rl = d & (KLANES - 1);       // 8 lanes per w_f_b row
    const int rg = d >> 3;                 // 16 rows per pass, 8 passes
    const uint4* tile = reinterpret_cast<const uint4*>(w_f_b + (size_t)base * KD);
    const float4 fa = *reinterpret_cast<const float4*>(sh_flow + 12 * rl);
    const float4 fb = *reinterpret_cast<const float4*>(sh_flow + 12 * rl + 4);
    const float4 fc = *reinterpret_cast<const float4*>(sh_flow + 96 + 12 * rl);
    const float4 fd = *reinterpret_cast<const float4*>(sh_flow + 96 + 12 * rl + 4);
#pragma unroll
    for (int it = 0; it < KD / DVGRP; ++it) {
      const int row = it * DVGRP + rg;
      const uint4 a = tile[row * 16 + rl];        // cols 8*rl .. +8
      const uint4 c = tile[row * 16 + 8 + rl];    // cols 64 + 8*rl .. +8
      float p[8];
      p[0] = fmaf(fa.x, BFLO(a.x), fa.y * BFHI(a.x));
      p[1] = fmaf(fa.z, BFLO(a.y), fa.w * BFHI(a.y));
      p[2] = fmaf(fb.x, BFLO(a.z), fb.y * BFHI(a.z));
      p[3] = fmaf(fb.z, BFLO(a.w), fb.w * BFHI(a.w));
      p[4] = fmaf(fc.x, BFLO(c.x), fc.y * BFHI(c.x));
      p[5] = fmaf(fc.z, BFLO(c.y), fc.w * BFHI(c.y));
      p[6] = fmaf(fd.x, BFLO(c.z), fd.y * BFHI(c.z));
      p[7] = fmaf(fd.z, BFLO(c.w), fd.w * BFHI(c.w));
#pragma unroll
      for (int st = 1; st < 8; st <<= 1)
#pragma unroll
        for (int i = 0; i < 8; i += 2 * st) p[i] += p[i + st];
      float g = p[0];
#pragma unroll
      for (int msk = 1; msk < KLANES; msk <<= 1) g += __shfl_xor_sync(0xffffffffu, g, msk);
      if (rl == 0) sh_ga[row] = g;
    }
  }
  __syncthreads();

  const float raw = __bfloat162float(__float2bfloat16(sh_ga[d])) + dt_bias[base + d];
  sh_dec[d] = __expf(LB * sigmoidf_(__expf(a_log[h]) * raw));

  // c = sum_k kn[k]*qs[k]; lets attn ride along with the m reduction.  Its
  // internal barrier is also what publishes sh_dec/sh_qs/sh_kn to the block.
  const float ckq = block_sum(knd * qsd, sh_red);

  // ---- delta rule over rec ----
  const int kl = d & (KLANES - 1);   // 8 lanes over k
  const int dvl = d >> 3;            // 16 dv rows per pass
  float* rec = reinterpret_cast<float*>(reinterpret_cast<char*>(kda_base) +
                                        (long long)line_index[b] * line_bytes) +
               (size_t)h * KD * KD;

  // dv-invariant coefficients for this lane's 16 k values: k = 32*j + 4*kl + i
  float cdec[KPER], ckn[KPER], cqs[KPER];
#pragma unroll
  for (int j = 0; j < KVEC; ++j) {
    const float4 a = *reinterpret_cast<const float4*>(sh_dec + 32 * j + 4 * kl);
    const float4 c = *reinterpret_cast<const float4*>(sh_kn + 32 * j + 4 * kl);
    const float4 e = *reinterpret_cast<const float4*>(sh_qs + 32 * j + 4 * kl);
    cdec[4 * j + 0] = a.x; cdec[4 * j + 1] = a.y; cdec[4 * j + 2] = a.z; cdec[4 * j + 3] = a.w;
    ckn[4 * j + 0] = c.x;  ckn[4 * j + 1] = c.y;  ckn[4 * j + 2] = c.z;  ckn[4 * j + 3] = c.w;
    cqs[4 * j + 0] = e.x;  cqs[4 * j + 1] = e.y;  cqs[4 * j + 2] = e.z;  cqs[4 * j + 3] = e.w;
  }

  float4* recv = reinterpret_cast<float4*>(rec) + kl;   // + dv*32 + 8*j

#pragma unroll 1
  for (int it = 0; it < KD / (DVGRP * ROWS_PER_ITER); ++it) {
    float4 s[ROWS_PER_ITER][KVEC];
    int dv[ROWS_PER_ITER];
#pragma unroll
    for (int r = 0; r < ROWS_PER_ITER; ++r) {
      dv[r] = (it * ROWS_PER_ITER + r) * DVGRP + dvl;
      const float4* p = recv + (size_t)dv[r] * 32;
#pragma unroll
      for (int j = 0; j < KVEC; ++j) s[r][j] = LDST(p + 8 * j);
    }
#pragma unroll
    for (int r = 0; r < ROWS_PER_ITER; ++r) {
      float m = 0.0f, aq = 0.0f;
#pragma unroll
      for (int j = 0; j < KVEC; ++j) {
        float4 t = s[r][j];
        t.x *= cdec[4 * j + 0];  t.y *= cdec[4 * j + 1];   // S*dec, kept for write-back
        t.z *= cdec[4 * j + 2];  t.w *= cdec[4 * j + 3];
        m = fmaf(t.x, ckn[4 * j + 0], m);   aq = fmaf(t.x, cqs[4 * j + 0], aq);
        m = fmaf(t.y, ckn[4 * j + 1], m);   aq = fmaf(t.y, cqs[4 * j + 1], aq);
        m = fmaf(t.z, ckn[4 * j + 2], m);   aq = fmaf(t.z, cqs[4 * j + 2], aq);
        m = fmaf(t.w, ckn[4 * j + 3], m);   aq = fmaf(t.w, cqs[4 * j + 3], aq);
        s[r][j] = t;
      }
#pragma unroll
      for (int msk = 1; msk < KLANES; msk <<= 1) {
        m += __shfl_xor_sync(0xffffffffu, m, msk);
        aq += __shfl_xor_sync(0xffffffffu, aq, msk);
      }
      const float dlt = (sh_v[dv[r]] - m) * beta;
      if (kl == 0) sh_attn[dv[r]] = fmaf(dlt, ckq, aq);
      float4* p = recv + (size_t)dv[r] * 32;
#pragma unroll
      for (int j = 0; j < KVEC; ++j) {
        float4 o;
        o.x = fmaf(dlt, ckn[4 * j + 0], s[r][j].x);
        o.y = fmaf(dlt, ckn[4 * j + 1], s[r][j].y);
        o.z = fmaf(dlt, ckn[4 * j + 2], s[r][j].z);
        o.w = fmaf(dlt, ckn[4 * j + 3], s[r][j].w);
        STST(p + 8 * j, o);
      }
    }
  }
  __syncthreads();

  // ---- attn landing, rms(gamma_o), sigmoid gate ----
  const bf16 attnb = __float2bfloat16(sh_attn[d]);
  const float af = __bfloat162float(attnb);
  const float atot = block_sum(af * af, sh_red);
  const bf16 o = __float2bfloat16(af * rsqrtf(atot / (float)KD + RMS_EPS) * gamma_o[d]);
  const bf16 g = __float2bfloat16(
      sigmoidf_(__bfloat162float(__float2bfloat16(gate_partial[(size_t)b * KDA_FUSED + 3 * INNER + base + d]))));
  out[(size_t)b * INNER + base + d] = __hmul(o, g);
}
