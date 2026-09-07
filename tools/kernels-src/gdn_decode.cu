// Gated DeltaNet decode step for one token per sequence, in two launches
// that replace vLLM's four (causal_conv1d_update, fused_recurrent_gated_
// delta_rule_packed_decode, the z copy, RMSNormGated):
//
//   kern_gdn_conv_bf16   grid [seqs, ceil(dim/256)], block 256
//     per channel c of sequence n: state[0..2] are the three previous
//     inputs, x = qkvz[n, c] the new one;
//       y = silu(sum_i bf16(s_i * w_i))    (products rounded to bf16 like
//                                           Triton's bf16 * bf16, sum in f32)
//       qkvz[n, c] := bf16(y)  (in place)   state := [s_1, s_2, x]
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
//     brings it into shared memory while every warp normalizes q and k (a
//     lane owns elements 4l..4l+3, as Triton's layout does) and the
//     scalars; each warp then owns rows w, w+8, .. and reads/writes them as
//     512-byte lines, so shared reads are conflict free and global stores
//     coalesce. A chunk's rows go in two passes (all S k, then all updates
//     and S q) so the rows' warp reductions overlap.
//
// Numerics are bit for bit those of the vLLM kernels (checked on a
// standalone harness against the pinned cubins): the same rounding points
// (inputs bf16, beta and o rounded to bf16, everything else f32), Triton's
// lowering of the transcendentals (ex2.approx, div.full,
// sqrt/rsqrt.approx.ftz, libdevice logf) and its reduction order (see
// tri_dot4 / tri_warp_sum). The intrinsics with _rn keep nvcc from
// contracting or reassociating.
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

extern "C" __global__ void kern_gdn_conv_bf16(
    __nv_bfloat16* __restrict__ qkvz, const __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__ state, const int* __restrict__ line,
    int dim, int row_stride, long long line_stride_bytes) {
  // Programmatic dependent launch: wait for the previous kernel's writes
  // before touching anything, then let the next launch stage itself.
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const int n = blockIdx.x;
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

// Triton reduces a thread's four consecutive elements as
// ((e1*f1 + e0*f0) + e2*f2) + e3*f3 with fma from the first product on,
// then a butterfly over the warp (xor 16, 8, 4, 2, 1). Mirrored exactly.
__device__ static inline float tri_dot4(const float e[4], const float f[4]) {
  float d = __fmul_rn(e[1], f[1]);
  d = __fmaf_rn(e[0], f[0], d);
  d = __fmaf_rn(e[2], f[2], d);
  return __fmaf_rn(e[3], f[3], d);
}
__device__ static inline float tri_warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, off));
  return v;
}

extern "C" __global__ void __launch_bounds__(NT, 3) kern_gdn_step_bf16(
    const __nv_bfloat16* __restrict__ qkvz, const __nv_bfloat16* __restrict__ ba,
    const float* __restrict__ A_log, const __nv_bfloat16* __restrict__ dt_bias,
    const __nv_bfloat16* __restrict__ norm_w, __nv_bfloat16* __restrict__ out,
    float* __restrict__ state, const int* __restrict__ line, float scale,
    float eps, int row_stride, int ba_stride, long long line_stride_bytes) {
  // Programmatic dependent launch: wait for the previous kernel's writes
  // before touching anything, then let the next launch stage itself.
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  extern __shared__ __align__(128) float tile[];  // [V][K]
  __shared__ float so[GDN_V];
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
  // The tile arrives in NCHUNK pieces, each behind its own barrier.
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

  // Every warp normalizes q and k itself: a lane owns elements 4l..4l+3.
  const __nv_bfloat16* row = qkvz + (long long)n * row_stride;
  float q[4], k[4];
  unpack4(*reinterpret_cast<const uint2*>(row + h * GDN_K + lane * 4), q);
  unpack4(*reinterpret_cast<const uint2*>(row + GDN_H * GDN_K + h * GDN_K + lane * 4), k);
  const float qs = tri_warp_sum(tri_dot4(q, q)), ks = tri_warp_sum(tri_dot4(k, k));
  const float qr = tri_sqrt(__fadd_rn(qs, 1e-6f)), kr = tri_sqrt(__fadd_rn(ks, 1e-6f));
#pragma unroll
  for (int i = 0; i < 4; i++) {
    q[i] = __fmul_rn(scale, tri_div(q[i], qr));
    k[i] = tri_div(k[i], kr);
  }
  const float a = bf(ba[(long long)n * ba_stride + GDN_HV + hv]);
  const float b = bf(ba[(long long)n * ba_stride + hv]);
  const float x = __fadd_rn(a, bf(dt_bias[hv]));
  const float sp = x <= 20.f ? logf(__fadd_rn(1.f, tri_exp(x))) : x;
  const float eg = tri_exp(__fmul_rn(-tri_exp(A_log[hv]), sp));
  const float beta = round_bf(tri_sigmoid(b));
  float vv[16];
#pragma unroll
  for (int i = 0; i < 16; i++) vv[i] = bf(row[2 * GDN_H * GDN_K + hv * GDN_V + warp + 8 * i]);
  __syncthreads();
#pragma unroll
  for (int c = 0; c < NCHUNK; c++) {
    const unsigned mb = (unsigned)__cvta_generic_to_shared(&mbar[c]);
    asm volatile(
        "{\n .reg .pred p;\n"
        "W%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n"
        " @!p bra W%=;\n}" ::"r"(mb)
        : "memory");
    // Two passes over the chunk's rows so the warp reductions of different
    // rows overlap: first every row's S k, then every row's update and S q.
    float u[16 / NCHUNK];
#pragma unroll
    for (int j = 0; j < 16 / NCHUNK; j++) {
      const int i = c * (16 / NCHUNK) + j;
      const float4 s4 = *reinterpret_cast<const float4*>(tile + (warp + 8 * i) * GDN_K + lane * 4);
      const float hd[4] = {__fmul_rn(s4.x, eg), __fmul_rn(s4.y, eg), __fmul_rn(s4.z, eg), __fmul_rn(s4.w, eg)};
      u[j] = __fmul_rn(__fsub_rn(vv[i], tri_warp_sum(tri_dot4(k, hd))), beta);
    }
#pragma unroll
    for (int j = 0; j < 16 / NCHUNK; j++) {
      const int i = c * (16 / NCHUNK) + j;
      const int r = warp + 8 * i;
      const float4 s4 = *reinterpret_cast<const float4*>(tile + r * GDN_K + lane * 4);
      const float hd[4] = {__fmul_rn(s4.x, eg), __fmul_rn(s4.y, eg), __fmul_rn(s4.z, eg), __fmul_rn(s4.w, eg)};
      float hn[4];
#pragma unroll
      for (int e = 0; e < 4; e++) hn[e] = __fmaf_rn(k[e], u[j], hd[e]);
      *reinterpret_cast<float4*>(S + r * GDN_K + lane * 4) = make_float4(hn[0], hn[1], hn[2], hn[3]);
      const float o = tri_warp_sum(tri_dot4(q, hn));
      if (lane == 0) so[r] = round_bf(o);
    }
  }
  __syncthreads();
  // The gated RMS norm, as Triton's layer_norm_fwd_kernel does it with one
  // warp: a lane owns o[4l..4l+3]; var = sum / N, rstd = rsqrt(eps + var),
  // y = (o * rstd) * w, y = y * (z * sigmoid(z)).
  if (warp == 0) {
    float o[4], wv[4], z[4];
#pragma unroll
    for (int i = 0; i < 4; i++) o[i] = so[lane * 4 + i];
    unpack4(*reinterpret_cast<const uint2*>(norm_w + lane * 4), wv);
    unpack4(*reinterpret_cast<const uint2*>(row + 2 * GDN_H * GDN_K + GDN_HV * GDN_V + hv * GDN_V + lane * 4), z);
    const float var = tri_div(tri_warp_sum(tri_dot4(o, o)), (float)GDN_V);
    const float rstd = tri_rsqrt(__fadd_rn(eps, var));
    float y[4];
#pragma unroll
    for (int i = 0; i < 4; i++)
      y[i] = __fmul_rn(__fmul_rn(__fmul_rn(o[i], rstd), wv[i]), __fmul_rn(z[i], tri_sigmoid(z[i])));
    *reinterpret_cast<uint2*>(op + lane * 4) = pack4(y);
  }
}
