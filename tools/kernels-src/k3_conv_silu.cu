// Kimi-K3 KDA short conv + SiLU, three streams (q|k|v) in one launch.
//
//   extern "C" __global__ void kern_k3_conv_silu(
//       const float* partial,      // [B, KDA_FUSED=49152]  stream s at columns s*INNER..
//       const float* cw,           // [3 stream][4 tap][INNER]
//       void* kda_base, const int* line_index, long long line_bytes,
//       __nv_bfloat16* conv_q, __nv_bfloat16* conv_k, __nv_bfloat16* conv_v,  // [B, INNER]
//       int B, const int* span_at, int span);  // rows [*span_at, +span) are a span (K9 does them): the block returns
//
// For s = 0,1,2 and every column c < INNER = 12288:
//     x   = bf16(partial[b, s*INNER + c])
//     y   = f32(win_s[0][c])*cw[s][0][c] + f32(win_s[1][c])*cw[s][1][c]
//         + f32(win_s[2][c])*cw[s][2][c] + f32(x)*cw[s][3][c]        (f32 accumulate)
//     sb  = f32(bf16(y))
//     out_s[b, c] = bf16(sb * (1 / (1 + exp(-sb))))
//     win_s[0][c] = win_s[1][c];  win_s[1][c] = win_s[2][c];  win_s[2][c] = x
//
// Landing points (pegainfer `conv_silu_batched`, tilelang_defs.py): the merged
// f32 partial lands in bf16 once (that bf16 is both the 4th conv tap and the
// value pushed into the window), the conv sum lands in bf16 once before the
// SiLU, and the SiLU result lands in bf16 once.  Everything between is f32.
//
// Window state lives in the row's KDA line, which is *not* one of the flat
// buffers:  line  = (char*)kda_base + (long long)line_index[b] * line_bytes
//           win_s = line + REC_BYTES(6291456) + s*73728,  bf16 [3 tap][INNER],
//           tap 0 oldest.  The shift is in place: a thread owns a fixed set of
//           columns for all three taps, and reads all three taps into registers
//           before storing any of them, so the in-place update is safe with no
//           cross-thread ordering.
//
// Tiling / launch (the manifest formula):
//     grid  (B, 3, INNER / (BLOCK*VEC)) = (B, 3, 24)      block 128      smem 0
//   BLOCK = 128 threads, VEC = 4 columns per thread -> 512 columns per block.
//   blockIdx.y is the stream, blockIdx.z the column segment.  grid.x is exactly
//   B (one block row per sequence), so no bounds test is needed anywhere.
//   Per thread: 1 x 16 B f32 load (partial), 4 x 16 B f32 loads (cw), 3 x 8 B
//   bf16 loads + 3 x 8 B bf16 stores (window), 1 x 8 B bf16 store (out); a warp
//   therefore moves 512 B / 512 B / 256 B contiguous per access, fully
//   coalesced with no bounds predication.  BLOCK x VEC was swept on tray07:
//   128x4 and 96x4 tie for best at B=64 (harness median 10.4 us), 256x8 is the
//   worst (12.3 us), and the ABI document's default 256x1 scalar tiling is
//   14.1 us -- the 4-wide accesses are worth 1.35x.  Because the tiling is not
//   the document default, the harness needs the explicit launch:
//       --grid B,3,24 --block 128,1,1
//
// Bandwidth (B=64): per row 3*(12288*4) partial + 2*3*(3*12288*2) window r/w
//   + 3*12288*2 out = 648 KB; cw 576 KB is read once from DRAM and then served
//   from L2 to the other 63 rows.  Total 43.06 MB, of which 24.19 MB is read --
//   and ncu measures dram__bytes_read.sum = 24.19 MB exactly, i.e. zero
//   over-fetch.  At B=64 that is only ~6 us of DRAM work, so the launch is
//   latency/ramp bound rather than bandwidth bound; the same kernel at B=1024
//   (650 MB, past L2) runs at 6.62 TB/s = 83% of the 8 TB/s spec and 96% of
//   what a plain float4 copy reaches on this part.  See
//   tools/k3-harness/notes/k3_conv_silu.md.
//
//   nvcc -cubin -arch=sm_103a -O3 -o kernels/k3_conv_silu.cubin tools/kernels-src/k3_conv_silu.cu
#include <cuda_bf16.h>

#ifndef K2_BLOCK
#define K2_BLOCK 128
#endif
#ifndef K2_VEC
#define K2_VEC 4
#endif

// HEADS: this rank's heads (96 whole, `-DHEADS=24` for a TP4 shard); the
// partial, the taps and the window are that many heads wide.
#ifndef HEADS
#define HEADS 96
#endif
#define K2_INNER (HEADS * 128)
#define K2_KDA_FUSED (4 * K2_INNER)
#define K2_REC_BYTES ((long long)HEADS * 128 * 128 * 4)
#define K2_WIN_BYTES ((long long)3 * K2_INNER * 2)

#if K2_VEC == 8
typedef uint4 k2_hvec;
#elif K2_VEC == 4
typedef uint2 k2_hvec;
#elif K2_VEC == 2
typedef unsigned int k2_hvec;
#else
#error "K2_VEC must be 2, 4 or 8"
#endif

union k2_h {
  k2_hvec v;
  __nv_bfloat16 h[K2_VEC];
};

extern "C" __global__ __launch_bounds__(K2_BLOCK) void kern_k3_conv_silu(
    const float* __restrict__ partial,          // [B, 49152]
    const float* __restrict__ cw,               // [3][4][12288]
    void* __restrict__ kda_base,
    const int* __restrict__ line_index,         // [B]
    long long line_bytes,
    __nv_bfloat16* __restrict__ conv_q,         // [B, 12288]
    __nv_bfloat16* __restrict__ conv_k,
    __nv_bfloat16* __restrict__ conv_v,
    int B, const int* __restrict__ span_at, int span) {
  const int b = blockIdx.x;
  if ((unsigned)(b - span_at[0]) < (unsigned)span) return;
  const int s = blockIdx.y;
  const int c = (int)(blockIdx.z * K2_BLOCK + threadIdx.x) * K2_VEC;

  const float* __restrict__ pin =
      partial + (long long)b * K2_KDA_FUSED + (long long)s * K2_INNER + c;
  const float* __restrict__ cwp = cw + (long long)(s * 4) * K2_INNER + c;
  __nv_bfloat16* __restrict__ win =
      (__nv_bfloat16*)((char*)kda_base + (long long)line_index[b] * line_bytes +
                       K2_REC_BYTES + (long long)s * K2_WIN_BYTES) +
      c;
  __nv_bfloat16* __restrict__ out =
      (s == 0 ? conv_q : (s == 1 ? conv_k : conv_v)) + (long long)b * K2_INNER + c;

  // --- loads: window first (it is also the store target), then partial / cw ---
  k2_h w0, w1, w2;
  w0.v = *(const k2_hvec*)(win + 0 * K2_INNER);
  w1.v = *(const k2_hvec*)(win + 1 * K2_INNER);
  w2.v = *(const k2_hvec*)(win + 2 * K2_INNER);

  float xin[K2_VEC], k0[K2_VEC], k1[K2_VEC], k2[K2_VEC], k3[K2_VEC];
#pragma unroll
  for (int i = 0; i < K2_VEC; i += 4) {
    *(float4*)&xin[i] = *(const float4*)(pin + i);
    *(float4*)&k0[i] = *(const float4*)(cwp + 0 * K2_INNER + i);
    *(float4*)&k1[i] = *(const float4*)(cwp + 1 * K2_INNER + i);
    *(float4*)&k2[i] = *(const float4*)(cwp + 2 * K2_INNER + i);
    *(float4*)&k3[i] = *(const float4*)(cwp + 3 * K2_INNER + i);
  }

  // --- x lands in bf16 once; it is tap 3 and the new window entry ---
  k2_h xb;
#pragma unroll
  for (int i = 0; i < K2_VEC; ++i) xb.h[i] = __float2bfloat16_rn(xin[i]);

  // --- window shift, in place; every tap was read above ---
  *(k2_hvec*)(win + 0 * K2_INNER) = w1.v;
  *(k2_hvec*)(win + 1 * K2_INNER) = w2.v;
  *(k2_hvec*)(win + 2 * K2_INNER) = xb.v;

  // --- conv (f32) -> bf16 landing -> SiLU -> bf16 landing ---
  k2_h o;
#pragma unroll
  for (int i = 0; i < K2_VEC; ++i) {
    float y = __bfloat162float(w0.h[i]) * k0[i];
    y = fmaf(__bfloat162float(w1.h[i]), k1[i], y);
    y = fmaf(__bfloat162float(w2.h[i]), k2[i], y);
    y = fmaf(__bfloat162float(xb.h[i]), k3[i], y);
    const float sb = __bfloat162float(__float2bfloat16_rn(y));
    o.h[i] = __float2bfloat16_rn(sb * (1.0f / (1.0f + expf(-sb))));
  }
  *(k2_hvec*)out = o.v;
}
