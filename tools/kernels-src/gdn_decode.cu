// Gated DeltaNet decode step for one token per sequence, in two launches
// that replace vLLM's four (causal_conv1d_update, fused_recurrent_gated_
// delta_rule_packed_decode, the z copy, RMSNormGated):
//
//   kern_gdn_conv_bf16   grid [seqs, max(ceil(dim/256), nba)], block 256
//     per channel c of sequence n: state[0..2] are the three previous
//     inputs, x = qkvz[n, c] the new one;
//       y = silu(sum_i bf16(s_i * w_i))    (products rounded to bf16 like
//                                           Triton's bf16 * bf16, sum in f32)
//       qkvz[n, c] := bf16(y)  (in place)   state := [s_1, s_2, x]
//     Block y < nba also computes output y of the layer's small projection
//     ba[n, y] = h[n] . Wba[y]^T (bf16 in, f32 accumulation, bf16 out),
//     which vLLM runs as its own 1 MB GEMM: each warp an eighth of
//     `hidden`, the eight partials summed in a fixed order.
//
//   kern_gdn_step_bf16   grid [HV, seqs], block 256, dynamic smem V*K*4
//     per (sequence n, value head hv), q/k head h = hv / (HV/H), state
//     S[V][K] f32 for that head:
//       q = l2norm(q) * scale, k = l2norm(k)
//       g = -exp(A_log) * softplus(a + dt_bias), beta = bf16(sigmoid(b))
//       S *= exp(g); u = (v - S k) * beta; S += u k^T; o = S q
//       o = bf16(o)                          (vLLM stores o as bf16 here)
//       y = (o * rsqrt(mean(o^2) + eps)) * w * silu(z)      -> out bf16
//     The head's 64 KB state is one contiguous block: one bulk async copy
//     brings it into shared memory while the block prepares q, k and the
//     scalars; each warp then owns rows w, w+8, .. and reads/writes them as
//     512-byte lines, so shared reads are conflict free and global stores
//     coalesce.
//
// Numerics follow the vLLM kernels' rounding points (inputs bf16, beta and
// o rounded to bf16, everything else f32) and Triton's lowering of the
// transcendentals (ex2.approx, div.full, sqrt/rsqrt.approx.ftz, libdevice
// logf). Reduction order differs, so the f32 state agrees to a few ulp,
// not bit for bit.
//
// The state page for sequence n is `line[n]` lines of `line_stride_bytes`
// from the state base; line 0 is the null line (skipped, output zero),
// mirroring the vLLM kernels. The conv state occupies the first 3 * dim
// bf16 of a line as [3][dim]; the recurrent state is at the byte offset the
// call passes, as [HV][V][K] f32.
//
//   nvcc -cubin -arch=sm_103a -o kernels/gdn_decode.cubin tools/kernels-src/gdn_decode.cu
#include <cuda_bf16.h>

#ifndef GDN_H
#define GDN_H 16      // q/k heads
#endif
#ifndef GDN_HV
#define GDN_HV 48     // value heads
#endif
#define GDN_K 128
#define GDN_V 128
#define CONV_W 4
#define BA_BATCH 3    // 16-byte chunks per lane: hidden / 8 / 256 rounded up (5120 -> 2.5)
#define NT 256
// Pieces the tile arrives in (each behind its own barrier). Measured on
// GB300: 1 is fastest (20.9 us/layer at batch 16, against a 17.7 us copy
// ceiling); 2 and 4 cost about a microsecond.
#ifndef NCHUNK
#define NCHUNK 1
#endif

__device__ static inline float bf(const __nv_bfloat16 v) { return __bfloat162float(v); }
__device__ static inline __nv_bfloat16 tobf(float v) { return __float2bfloat16_rn(v); }
__device__ static inline float round_bf(float v) { return bf(tobf(v)); }

// Triton's f32 lowering on NVIDIA: exp(x) = ex2.approx(x * log2e), `/` is
// div.full, sqrt/rsqrt are the approx.ftz forms.
__device__ static inline float tri_exp(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x * __int_as_float(0x3FB8AA3B)));
  return y;
}
__device__ static inline float tri_div(float a, float b) {
  float y;
  asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
  return y;
}
__device__ static inline float tri_sqrt(float x) {
  float y;
  asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ static inline float tri_rsqrt(float x) {
  float y;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ static inline float tri_sigmoid(float x) { return tri_div(1.f, 1.f + tri_exp(0.f - x)); }

extern "C" __global__ void __launch_bounds__(256) kern_gdn_conv_bf16(
    __nv_bfloat16* __restrict__ qkvz, const __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__ state, const int* __restrict__ line,
    const __nv_bfloat16* __restrict__ h, const __nv_bfloat16* __restrict__ wba,
    __nv_bfloat16* __restrict__ ba, int dim, int row_stride, long long line_stride_bytes,
    int hidden, int nba) {
  __shared__ float part[8];
  const int n = blockIdx.x;
  if (blockIdx.y < nba) {
    // Output o = blockIdx.y: warp w takes hidden/8 elements starting at
    // w * hidden / 8, a lane three 16-byte chunks; then a fixed-order sum.
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int o = blockIdx.y, span = hidden / 8;
    const __nv_bfloat16* hp = h + (long long)n * hidden + warp * span;
    const __nv_bfloat16* wp = wba + (long long)o * hidden + warp * span;
    uint4 hv[BA_BATCH], wv[BA_BATCH];
#pragma unroll
    for (int b = 0; b < BA_BATCH; b++) {
      const int i = lane * 8 + b * 256;
      if (i < span) {
        hv[b] = *reinterpret_cast<const uint4*>(hp + i);
        wv[b] = *reinterpret_cast<const uint4*>(wp + i);
      }
    }
    float acc = 0.f;
#pragma unroll
    for (int b = 0; b < BA_BATCH; b++) {
      if (lane * 8 + b * 256 < span) {
        const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&hv[b]);
        const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wv[b]);
#pragma unroll
        for (int j = 0; j < 4; j++) {
          acc = fmaf(bf(h2[j].x), bf(w2[j].x), acc);
          acc = fmaf(bf(h2[j].y), bf(w2[j].y), acc);
        }
      }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (lane == 0) part[warp] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
      float t = 0.f;
#pragma unroll
      for (int w = 0; w < 8; w++) t += part[w];
      ba[(long long)n * nba + o] = tobf(t);
    }
  }
  const int c = blockIdx.y * blockDim.x + threadIdx.x;
  if (c >= dim) return;
  const long long idx = line[n];
  if (idx <= 0) return;
  __nv_bfloat16* s = reinterpret_cast<__nv_bfloat16*>(
      reinterpret_cast<char*>(state) + idx * line_stride_bytes);
  __nv_bfloat16* xp = qkvz + (long long)n * row_stride + c;
  const __nv_bfloat16 s0 = s[c], s1 = s[dim + c], s2 = s[2 * dim + c], x = *xp;
  const uint2 wv = *reinterpret_cast<const uint2*>(w + (long long)c * CONV_W);
  const __nv_bfloat162 w01 = *reinterpret_cast<const __nv_bfloat162*>(&wv.x);
  const __nv_bfloat162 w23 = *reinterpret_cast<const __nv_bfloat162*>(&wv.y);
  float acc = 0.f;
  acc += round_bf(bf(s0) * bf(w01.x));
  acc += round_bf(bf(s1) * bf(w01.y));
  acc += round_bf(bf(s2) * bf(w23.x));
  acc += round_bf(bf(x) * bf(w23.y));
  acc = tri_div(acc, 1.f + tri_exp(0.f - acc));
  *xp = tobf(acc);
  s[c] = s1;
  s[dim + c] = s2;
  s[2 * dim + c] = x;
}

__device__ static inline float warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

// Block-wide sum over the first 128 threads' values (4 warps); every thread
// of the block gets the result.
__device__ static inline float sum128(float v, float* red) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  v = warp_sum(v);
  if (lane == 0 && warp < 4) red[warp] = v;
  __syncthreads();
  const float r = ((red[0] + red[1]) + red[2]) + red[3];
  __syncthreads();
  return r;
}

extern "C" __global__ void __launch_bounds__(NT) kern_gdn_step_bf16(
    const __nv_bfloat16* __restrict__ qkvz, const __nv_bfloat16* __restrict__ ba,
    const float* __restrict__ A_log, const __nv_bfloat16* __restrict__ dt_bias,
    const __nv_bfloat16* __restrict__ norm_w, __nv_bfloat16* __restrict__ out,
    float* __restrict__ state, const int* __restrict__ line, float scale,
    float eps, int row_stride, int ba_stride, long long line_stride_bytes) {
  extern __shared__ __align__(128) float tile[];  // [V][K]
  __shared__ float sq[GDN_K], sk[GDN_K], so[GDN_V], red[4];
  __shared__ __align__(8) unsigned long long mbar[NCHUNK];
  const int hv = blockIdx.x, n = blockIdx.y;
  const int h = hv / (GDN_HV / GDN_H);
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const long long idx = line[n];
  __nv_bfloat16* op = out + (long long)n * (GDN_HV * GDN_V) + hv * GDN_V;
  if (idx <= 0) {
    if (t < GDN_V) op[t] = tobf(0.f);
    return;
  }
  float* S = reinterpret_cast<float*>(reinterpret_cast<char*>(state) + idx * line_stride_bytes) +
             (long long)hv * GDN_V * GDN_K;
  // The tile arrives in NCHUNK pieces, each behind its own barrier, so a
  // warp's early rows are processed and stored while later rows are still
  // landing.
  const unsigned bytes = GDN_V * GDN_K * 4 / NCHUNK;
  if (t == 0) {
#pragma unroll
    for (int c = 0; c < NCHUNK; c++) {
      const unsigned mb = (unsigned)__cvta_generic_to_shared(&mbar[c]);
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(mb));
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#pragma unroll
    for (int c = 0; c < NCHUNK; c++) {
      const unsigned mb = (unsigned)__cvta_generic_to_shared(&mbar[c]);
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(mb), "r"(bytes) : "memory");
      asm volatile(
          "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
              (unsigned)__cvta_generic_to_shared(tile + c * (GDN_V * GDN_K / NCHUNK))),
          "l"(S + c * (GDN_V * GDN_K / NCHUNK)), "r"(bytes), "r"(mb)
          : "memory");
    }
  }

  const __nv_bfloat16* row = qkvz + (long long)n * row_stride;
  float qv = 0.f, kv = 0.f;
  if (t < GDN_K) {
    qv = bf(row[h * GDN_K + t]);
    kv = bf(row[GDN_H * GDN_K + h * GDN_K + t]);
  }
  const float qs = sum128(qv * qv, red);
  const float ks = sum128(kv * kv, red);
  if (t < GDN_K) {
    sq[t] = tri_div(qv, tri_sqrt(qs + 1e-6f)) * scale;
    sk[t] = tri_div(kv, tri_sqrt(ks + 1e-6f));
  }
  const float a = bf(ba[(long long)n * ba_stride + GDN_HV + hv]);
  const float b = bf(ba[(long long)n * ba_stride + hv]);
  const float x = a + bf(dt_bias[hv]);
  const float sp = x <= 20.f ? logf(1.f + tri_exp(x)) : x;
  const float eg = tri_exp(-tri_exp(A_log[hv]) * sp);
  const float beta = round_bf(tri_sigmoid(b));
  // Own columns: lane*4 .. +4 of rows warp + 8*i.
  float vv[16];
#pragma unroll
  for (int i = 0; i < 16; i++) vv[i] = bf(row[2 * GDN_H * GDN_K + hv * GDN_V + warp + 8 * i]);
  __syncthreads();
  const float4 kk = *reinterpret_cast<const float4*>(sk + lane * 4);
  const float4 qq = *reinterpret_cast<const float4*>(sq + lane * 4);
#pragma unroll
  for (int c = 0; c < NCHUNK; c++) {
    const unsigned mb = (unsigned)__cvta_generic_to_shared(&mbar[c]);
    asm volatile(
        "{\n .reg .pred p;\n"
        "W%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n"
        " @!p bra W%=;\n}" ::"r"(mb)
        : "memory");
    float u[16 / NCHUNK];
#pragma unroll
    for (int j = 0; j < 16 / NCHUNK; j++) {
      const int i = c * (16 / NCHUNK) + j;
      const float4 s4 = *reinterpret_cast<const float4*>(tile + (warp + 8 * i) * GDN_K + lane * 4);
      float d = (s4.x * eg) * kk.x + (s4.y * eg) * kk.y + (s4.z * eg) * kk.z + (s4.w * eg) * kk.w;
      d = warp_sum(d);
      u[j] = (vv[i] - d) * beta;
    }
#pragma unroll
    for (int j = 0; j < 16 / NCHUNK; j++) {
      const int i = c * (16 / NCHUNK) + j;
      const int r = warp + 8 * i;
      float4 s4 = *reinterpret_cast<const float4*>(tile + r * GDN_K + lane * 4);
      s4.x = s4.x * eg + u[j] * kk.x;
      s4.y = s4.y * eg + u[j] * kk.y;
      s4.z = s4.z * eg + u[j] * kk.z;
      s4.w = s4.w * eg + u[j] * kk.w;
      *reinterpret_cast<float4*>(S + r * GDN_K + lane * 4) = s4;
      float o = s4.x * qq.x + s4.y * qq.y + s4.z * qq.z + s4.w * qq.w;
      o = warp_sum(o);
      if (lane == 0) so[r] = round_bf(o);
    }
  }
  __syncthreads();
  const float ov = t < GDN_V ? so[t] : 0.f;
  const float ss = sum128(ov * ov, red);
  if (t < GDN_V) {
    const float rstd = tri_rsqrt(tri_div(ss, (float)GDN_V) + eps);
    const float z = bf(row[2 * GDN_H * GDN_K + GDN_HV * GDN_V + hv * GDN_V + t]);
    float y = (ov * rstd) * bf(norm_w[t]);
    y *= z * tri_sigmoid(z);
    op[t] = tobf(y);
  }
}
