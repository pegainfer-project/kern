// Kimi-K3 MLA fused-projection prep [K4]: one launch replaces the seven that
// used to sit between the fused `normed * wfu` GEMM and `w_q_b`.
//
// From the f32 partial row P = mla_fused_partial[b, 0 .. 14400):
//
//   q_norm[b]   = rms(bf16(P[0    .. 1536]), gamma_q_a)   // 1536, round-before-scale
//   kv_norm     = rms(bf16(P[1536 .. 2048]), gamma_kv_a)  // 512
//   rope        = bf16(P[2048 .. 2112])                   // 64
//   slab[slot]  = kv_norm | rope                          // the 576-wide latent row
//   mla_gate[b] = bf16(P[2112 .. 14400])                  // 12288
//
//   extern "C" __global__ void kern_k3_mla_prep(
//       const f32*  partial,       // [B, MLA_FUSED = 14400]
//       const bf16* gamma_q_a,     // [Q_LORA  = 1536]
//       const bf16* gamma_kv_a,    // [KV_LORA = 512]
//       const i64*  slot_mapping,  // [B]
//       bf16*       slab,          // state base
//       long long layer_off, long long page_stride,   // elements
//       bf16*       q_norm,        // [B, Q_LORA]
//       bf16*       mla_gate,      // [B, INNER = 12288]
//       int B);
//
// ---- grid / block / smem (this is what the manifest should copy) --------
//
//   grid (B, 4, 1)      block (512, 1, 1)      dynamic smem 0
//   static smem: 2112 f32 staged head columns + 512 staged gamma pairs
//                + 64 f32 warp partials = 12800 B/block
//
// Row b is split over gridDim.y blocks:
//   blockIdx.y == 0  : the norm/append head -- columns 0 .. 2112 of the row,
//                      i.e. q_norm and the latent row (kv_norm | rope).
//   blockIdx.y >= 1  : gate segment blockIdx.y - 1, an equal slice of the
//                      12288 gate columns: a pure f32 -> bf16 landing copy.
// grid.x is exactly B, per the ABI.
//
// The kernel has two paths.  The documented (512, gridDim.y == 4) shape takes
// a fully specialised path with compile-time trip counts -- at B = 1 this
// kernel is far too small to be bandwidth bound, so its cost is instruction
// issue and one memory round trip, and every predicate removed from the inner
// loops shows up in the measurement.  Any other launch shape falls through to
// a geometry-agnostic path (any gridDim.y >= 1, any blockDim.x that is a
// multiple of 32 in [32, 1024]), ~25% slower but correct, so that the kernel
// still does the right thing under the document's default grid (B, 1, 1)
// block 1024.  Both paths write identical bits.
//
// Fast head (512 threads, 4 columns each, float4 loads, 8-byte bf16 stores):
//   t <  384        -> q_norm  column 4*t                (warps 0..11)
//   384 <= t < 512  -> kv_norm column 1536 + 4*(t-384)   (warps 12..15)
//   t <  16         -> also rope column 2048 + 4*t
// so the two block reductions split on a warp boundary: one warp shuffle
// tree, one __syncthreads, and each thread re-sums the 12 (q) or 4 (kv) warp
// partials it needs.  Fast gate block: 4096 columns, 4 per thread per pass,
// two unrolled passes -> two float4 loads in flight, then two 8-byte stores.
// Generic path: plain strided float4 loops, with the head columns and the
// gamma words staged through shared memory in the same pass so that the gamma
// round trip is not stranded behind the block reduction.
//
// Slab addressing (elements, `slab` is the state base):
//   row = (slot / 64) * page_stride + layer_off + (slot % 64) * 576
// with kv_norm at row + 0 .. 512 and rope at row + 512 .. 576.  A negative
// slot is treated as "no slot" and the append is skipped; q_norm and
// mla_gate are written in full regardless.
//
// ---- landing points (pegainfer's chain: land_rms_norm_rbs + rms_norm_rbs)
//   * every f32 partial column lands to bf16 *first*, x = f32(bf16(P[i])),
//     including the copies that feed the sums of squares;
//   * rms is round-before-scale: y = bf16(x * rsqrt(mean(x^2) + 1e-5)),
//     then y * gamma as a bf16 x bf16 -> bf16 product (__hmul2);
//   * rope and mla_gate are a single bf16 landing, no arithmetic.
// f32 accumulation for the two sums of squares; the reduction order is a
// per-thread serial sum then a warp shuffle tree (the CPU reference sums
// serially in double -- inside tolerance, and in practice 0 or 1 bf16 ULP).
//
//   /usr/local/cuda-13.1/bin/nvcc -cubin -arch=sm_103a -O3 -Xptxas -v \
//       -o target/cubins/k3_mla_prep.cubin tools/kernels-src/k3_mla_prep.cu
#include <cuda_bf16.h>

#define Q_LORA    1536
#define KV_LORA   512
#define ROPE      64
#define KV_A      576
// -DINNER=<heads * 128> -DMLA_FUSED=<2112 + INNER>: the gate columns of this rank's heads (a tray form).
#ifndef INNER
#define INNER     12288
#endif
#ifndef MLA_FUSED
#define MLA_FUSED 14400
#endif
#define HEADC     (Q_LORA + KV_LORA + ROPE)   // 2112 head columns
#define EPSV      1e-5f

#define QU   (Q_LORA / 4)               // 384  float4 units of q
#define KU   ((Q_LORA + KV_LORA) / 4)   // 512  end of the kv units
#define HU   (HEADC / 4)                // 528  end of the rope units
#define GU   (INNER / 4)                // 3072 float4 units of gate

#define NTF   512                       // fast-path block
#ifndef GYF
#define GYF   4                         // fast-path gridDim.y
#endif
#define GSEGU (GU / (GYF - 1))          // 1024 gate units per fast segment
#define QWF   (QU / 32)                 // 12 warps of q on the fast head

#define MAXNT 1024
#define MAXW  (MAXNT / 32)

// bf16 pairs are carried as raw 32-bit words so that ptxas keeps them in
// consecutive registers and folds the accesses into LDG/STG.E.64 / .128;
// building them out of __nv_bfloat162 struct members makes it emit one
// 32-bit STG per pair instead.
__device__ __forceinline__ unsigned pack2(float lo, float hi) {
  __nv_bfloat162 p = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<unsigned*>(&p);
}
__device__ __forceinline__ unsigned mul2(unsigned a, unsigned g) {
  __nv_bfloat162 r = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&a),
                             *reinterpret_cast<const __nv_bfloat162*>(&g));
  return *reinterpret_cast<unsigned*>(&r);
}
__device__ __forceinline__ float landf(float x) {
  return __bfloat162float(__float2bfloat16(x));
}

extern "C" __global__ __launch_bounds__(MAXNT) void kern_k3_mla_prep(
    const float* __restrict__ partial,             // [B, MLA_FUSED]
    const __nv_bfloat16* __restrict__ gamma_q_a,   // [Q_LORA]
    const __nv_bfloat16* __restrict__ gamma_kv_a,  // [KV_LORA]
    const long long* __restrict__ slot_mapping,    // [B]
    __nv_bfloat16* __restrict__ slab,              // state base
    long long layer_off, long long page_stride,    // elements
    __nv_bfloat16* __restrict__ q_norm,            // [B, Q_LORA]
    __nv_bfloat16* __restrict__ mla_gate,          // [B, INNER]
    int B) {
  const int b = blockIdx.x;
  if (b >= B) return;
  const int t = threadIdx.x;
  const int nt = blockDim.x;
  const float* __restrict__ P = partial + (long long)b * MLA_FUSED;

  __shared__ float xs[HEADC];     // generic path: landed head columns
  __shared__ uint2 gs[KU];        // generic path: gamma_q_a | gamma_kv_a
  __shared__ float red[2][MAXW];  // warp partials: [0] q, [1] kv

  // =====================================================================
  // Fast path: the documented grid (B, 4, 1) block (512, 1, 1).
  // =====================================================================
  if (nt == NTF && gridDim.y == GYF) {
    if (blockIdx.y != 0) {  // ---- gate segment, 4096 columns -------------
      const float* __restrict__ src = P + HEADC;
      __nv_bfloat16* __restrict__ dst = mla_gate + (long long)b * INNER;
      const int base = (int)(blockIdx.y - 1) * GSEGU + t;
#pragma unroll
      for (int k = 0; k < GSEGU / NTF; ++k) {  // 2 passes, both loads in flight
        const int u = base + k * NTF;
        const float4 v = *reinterpret_cast<const float4*>(src + 4 * u);
        *reinterpret_cast<uint2*>(dst + 4 * u) =
            make_uint2(pack2(v.x, v.y), pack2(v.z, v.w));
      }
      return;
    }
    // ---- head: q_norm, kv_norm, rope, latent row ------------------------
    const bool isq = (t < QU);
    const int col = isq ? (4 * t) : (Q_LORA + 4 * (t - QU));
    const float4 v = *reinterpret_cast<const float4*>(P + col);
    const float x0 = landf(v.x), x1 = landf(v.y);
    const float x2 = landf(v.z), x3 = landf(v.w);

    float ss = x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      ss += __shfl_down_sync(0xffffffffu, ss, off);
    if ((t & 31) == 0) red[0][t >> 5] = ss;

    // independent of the reduction, so issued before the barrier
    const long long slot = slot_mapping[b];
    const bool append = (slot >= 0);
    __nv_bfloat16* const row = slab + (slot / 64) * page_stride + layer_off +
                               (slot % 64) * KV_A;
    const __nv_bfloat16* const gsrc =
        isq ? (gamma_q_a + col) : (gamma_kv_a + (col - Q_LORA));
    const uint2 g = *reinterpret_cast<const uint2*>(gsrc);
    uint2 rope4 = make_uint2(0u, 0u);
    if (t < ROPE / 4) {
      const float4 r =
          *reinterpret_cast<const float4*>(P + Q_LORA + KV_LORA + 4 * t);
      rope4 = make_uint2(pack2(r.x, r.y), pack2(r.z, r.w));
    }

    __syncthreads();

    float tot = 0.f;
    if (isq) {
#pragma unroll
      for (int w = 0; w < QWF; ++w) tot += red[0][w];
    } else {
#pragma unroll
      for (int w = QWF; w < NTF / 32; ++w) tot += red[0][w];
    }
    const float sc = isq ? rsqrtf(tot * (1.f / Q_LORA) + EPSV)
                         : rsqrtf(tot * (1.f / KV_LORA) + EPSV);
    const uint2 o = make_uint2(mul2(pack2(x0 * sc, x1 * sc), g.x),
                               mul2(pack2(x2 * sc, x3 * sc), g.y));
    if (isq) {
      *reinterpret_cast<uint2*>(q_norm + (long long)b * Q_LORA + col) = o;
      if (append && t < ROPE / 4)
        *reinterpret_cast<uint2*>(row + KV_LORA + 4 * t) = rope4;
    } else if (append) {
      *reinterpret_cast<uint2*>(row + (col - Q_LORA)) = o;
    }
    return;
  }

  // =====================================================================
  // Generic path: any gridDim.y >= 1, any blockDim.x = 32k in [32, 1024].
  // =====================================================================
  if (blockIdx.y == 0) {
    const long long slot = slot_mapping[b];
    const bool append = (slot >= 0);
    __nv_bfloat16* const row = slab + (slot / 64) * page_stride + layer_off +
                               (slot % 64) * KV_A;
    float sq = 0.f, sk = 0.f;
    // One pass over the 528 head units.  This unit's gamma word rides along
    // so that its round trip is not left sitting behind the block reduction.
    for (int u = t; u < HU; u += nt) {
      const float4 v = *reinterpret_cast<const float4*>(P + 4 * u);
      const float a0 = landf(v.x), a1 = landf(v.y);
      const float a2 = landf(v.z), a3 = landf(v.w);
      *reinterpret_cast<float4*>(xs + 4 * u) = make_float4(a0, a1, a2, a3);
      const float s = a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3;
      if (u < QU) {
        sq += s;
        gs[u] = *reinterpret_cast<const uint2*>(gamma_q_a + 4 * u);
      } else if (u < KU) {
        sk += s;
        gs[u] = *reinterpret_cast<const uint2*>(gamma_kv_a + 4 * (u - QU));
      }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      sq += __shfl_down_sync(0xffffffffu, sq, off);
      sk += __shfl_down_sync(0xffffffffu, sk, off);
    }
    if ((t & 31) == 0) { red[0][t >> 5] = sq; red[1][t >> 5] = sk; }
    __syncthreads();
    const int nw = nt >> 5;
    float tq = 0.f, tk = 0.f;
#pragma unroll
    for (int w = 0; w < MAXW; ++w) {
      if (w >= nw) break;
      tq += red[0][w];
      tk += red[1][w];
    }
    const float scq = rsqrtf(tq * (1.f / Q_LORA) + EPSV);
    const float sck = rsqrtf(tk * (1.f / KV_LORA) + EPSV);

    // same iteration space as the staging pass
    for (int u = t; u < HU; u += nt) {
      const float4 x = *reinterpret_cast<const float4*>(xs + 4 * u);
      if (u < QU) {
        const uint2 gq = gs[u];
        *reinterpret_cast<uint2*>(q_norm + (long long)b * Q_LORA + 4 * u) =
            make_uint2(mul2(pack2(x.x * scq, x.y * scq), gq.x),
                       mul2(pack2(x.z * scq, x.w * scq), gq.y));
      } else if (!append) {
        // no slot this step: the latent row is not touched
      } else if (u < KU) {
        const uint2 gk = gs[u];
        *reinterpret_cast<uint2*>(row + 4 * (u - QU)) =
            make_uint2(mul2(pack2(x.x * sck, x.y * sck), gk.x),
                       mul2(pack2(x.z * sck, x.w * sck), gk.y));
      } else {
        *reinterpret_cast<uint2*>(row + KV_LORA + 4 * (u - KU)) =
            make_uint2(pack2(x.x, x.y), pack2(x.z, x.w));
      }
    }
    if (gridDim.y > 1) return;  // the gate belongs to the other blocks
  }

  int lo = 0, hi = GU;
  if (gridDim.y > 1) {
    const int nseg = gridDim.y - 1;
    const int per = (GU + nseg - 1) / nseg;
    lo = min((int)(blockIdx.y - 1) * per, GU);
    hi = min(lo + per, GU);
  }
  const float* __restrict__ src = P + HEADC;
  __nv_bfloat16* __restrict__ dst = mla_gate + (long long)b * INNER;
  for (int u = lo + t; u < hi; u += nt) {
    const float4 v = *reinterpret_cast<const float4*>(src + 4 * u);
    *reinterpret_cast<uint2*>(dst + 4 * u) =
        make_uint2(pack2(v.x, v.y), pack2(v.z, v.w));
  }
}
