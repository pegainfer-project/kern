// K9 `k3_span_gather` — the K2 conv + SiLU for a span: batch rows
// [*span_at, *span_at + span) are consecutive tokens of one sequence, so the
// conv taps of span row i are the sequence's window (the line of the span's
// first row) for i < 3 and the span's own earlier rows after that; the window
// leaves holding the span's last three inputs. The results land in the span's
// own buffers, rows 0..span, where the FlashKDA span kernel's descriptors
// (fixed at load) find them, together with what it wants transposed: beta as
// [HEADS][span] and the f_a flow as bf16 [span][128] (the f_b GEMM's input).
// Contract: docs/k3-kernel-abi.md section K9.
//
//   extern "C" __global__ void kern_k3_span_gather(
//       const float* partial,      // [rows, KDA_FUSED]  rows at..at+span read
//       const float* cw,           // [3 stream][4 tap][INNER]
//       void* kda_base, const int* line_index, long long line_bytes,  // line_index[at]'s line
//       const float* wsm_partial,  // [rows, WSM=256]  col h = b_proj, 96.. = f_a
//       bf16* span_q, bf16* span_k, bf16* span_v,   // [span, INNER]
//       bf16* span_beta,           // [HEADS * span]   h*span + i
//       bf16* span_flow,           // [span, 128]
//       const int* span_at,        // [1]  the span's first batch row
//       int span);
//
//   grid  (INNER/512, 4, ceil(span/8))   block 128   smem 0
//   blockIdx.y < 3: stream y, 4 consecutive columns per thread, rows
//   8*blockIdx.z .. +8;  blockIdx.y == 3: beta / flow for the same 8 rows.
//
// Per stream s, column c, with x_{-3..-1} = win_s[0..2][c] and
// x_i = bf16(partial[i, s*INNER + c]) for 0 <= i < span:
//   y_i        = sum_{t<3} f32(x_{i-3+t}) * cw[s][t][c] + f32(x_i) * cw[s][3][c]
//   sb         = bf16(y_i);  out_s[i, c] = bf16(sb * sigmoid(sb))
//   win_s[t][c] = x_{span-3+t}                                   (t < 3)
// which is exactly K2 applied to the rows one after another (same landing
// points), so a span of n tokens leaves the window as n decode steps would.
// Rows are independent given the inputs, so they run in parallel; only the
// z == 0 block reads the old window (rows 0..2 need it) and only it writes
// the new one, after its reads, so no two blocks touch the window.
#include <cuda_bf16.h>

#ifndef HEADS
#define HEADS 96
#endif
#define K9_INNER (HEADS * 128)
#define K9_KDA_FUSED (4 * K9_INNER)
#define K9_REC_BYTES ((long long)HEADS * 128 * 128 * 4)
#define K9_WIN_BYTES ((long long)3 * K9_INNER * 2)
#define K9_WSM 256
#define K9_WSM_FA 96
#define K9_BLOCK 128
#define K9_VEC 4
#define K9_ROWS 8

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ float k9_silu_bf16(float y) {
  float sb = __bfloat162float(__float2bfloat16(y));
  return sb / (1.0f + __expf(-sb));
}

__device__ __forceinline__ void k9_unpack(uint2 raw, float* x) {
  x[0] = __bfloat162float(__ushort_as_bfloat16((unsigned short)(raw.x & 0xffffu)));
  x[1] = __bfloat162float(__ushort_as_bfloat16((unsigned short)(raw.x >> 16)));
  x[2] = __bfloat162float(__ushort_as_bfloat16((unsigned short)(raw.y & 0xffffu)));
  x[3] = __bfloat162float(__ushort_as_bfloat16((unsigned short)(raw.y >> 16)));
}

__device__ __forceinline__ void k9_pack(const float* x, bf16* o) {
#pragma unroll
  for (int k = 0; k < K9_VEC; ++k) o[k] = __float2bfloat16(x[k]);
}

// x_i for i in [-3, span): the window for i < 0 (z == 0 blocks only), the
// bf16 landing of the partial otherwise.
__device__ __forceinline__ void k9_input(const float* __restrict__ partial, const bf16* win, int s, int c, int i,
                                         float* x) {
  if (i < 0) {
    k9_unpack(*(const uint2*)(win + (size_t)(i + 3) * K9_INNER + c), x);
  } else {
    const float4 p = *(const float4*)(partial + (size_t)i * K9_KDA_FUSED + (size_t)s * K9_INNER + c);
    x[0] = __bfloat162float(__float2bfloat16(p.x));
    x[1] = __bfloat162float(__float2bfloat16(p.y));
    x[2] = __bfloat162float(__float2bfloat16(p.z));
    x[3] = __bfloat162float(__float2bfloat16(p.w));
  }
}

extern "C" __global__ __launch_bounds__(K9_BLOCK) void kern_k3_span_gather(
    const float* __restrict__ partial,
    const float* __restrict__ cw,
    void* __restrict__ kda_base,
    const int* __restrict__ line_index,
    long long line_bytes,
    const float* __restrict__ wsm_partial,
    bf16* __restrict__ span_q, bf16* __restrict__ span_k, bf16* __restrict__ span_v,
    bf16* __restrict__ span_beta,
    bf16* __restrict__ span_flow,
    const int* __restrict__ span_at,
    int span) {
  const int at = span_at[0];
  const int row0 = blockIdx.z * K9_ROWS;
  partial += (size_t)at * K9_KDA_FUSED;
  wsm_partial += (size_t)at * K9_WSM;
  if (blockIdx.y == 3) {
    // 16 threads per row: thread k writes flow[8k..8k+8) and beta for heads k, k+16, ...
    const int i = row0 + (threadIdx.x >> 4), k = threadIdx.x & 15;
    if (i >= span) return;
    const float* row = wsm_partial + (size_t)i * K9_WSM;
    for (int h = k; h < HEADS; h += 16) span_beta[(size_t)h * span + i] = __float2bfloat16(row[h]);
    bf16 f[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) f[j] = __float2bfloat16(row[K9_WSM_FA + k * 8 + j]);
    *(uint4*)(span_flow + (size_t)i * 128 + k * 8) = *(const uint4*)f;
    return;
  }
  const int s = blockIdx.y;
  const int c = (int)(blockIdx.x * K9_BLOCK + threadIdx.x) * K9_VEC;
  bf16* __restrict__ out = s == 0 ? span_q : s == 1 ? span_k : span_v;
  bf16* win = (bf16*)((char*)kda_base + (long long)line_index[at] * line_bytes + K9_REC_BYTES +
                      (long long)s * K9_WIN_BYTES);
  const float* __restrict__ w = cw + (size_t)s * 4 * K9_INNER + c;
  const float4 w0 = *(const float4*)(w), w1 = *(const float4*)(w + K9_INNER),
               w2 = *(const float4*)(w + 2 * K9_INNER), w3 = *(const float4*)(w + 3 * K9_INNER);
  const float wt[4][K9_VEC] = {{w0.x, w0.y, w0.z, w0.w}, {w1.x, w1.y, w1.z, w1.w},
                               {w2.x, w2.y, w2.z, w2.w}, {w3.x, w3.y, w3.z, w3.w}};

  // The block's rows plus the three before them, in a sliding register set.
  float t0[K9_VEC], t1[K9_VEC], t2[K9_VEC], x[K9_VEC];
  k9_input(partial, win, s, c, row0 - 3, t0);
  k9_input(partial, win, s, c, row0 - 2, t1);
  k9_input(partial, win, s, c, row0 - 1, t2);
  const int rows = min(K9_ROWS, span - row0);
  for (int r = 0; r < rows; ++r) {
    k9_input(partial, win, s, c, row0 + r, x);
    bf16 o[K9_VEC];
#pragma unroll
    for (int k = 0; k < K9_VEC; ++k) {
      const float y = t0[k] * wt[0][k] + t1[k] * wt[1][k] + t2[k] * wt[2][k] + x[k] * wt[3][k];
      o[k] = __float2bfloat16(k9_silu_bf16(y));
      t0[k] = t1[k];
      t1[k] = t2[k];
      t2[k] = x[k];
    }
    *(uint2*)(out + (size_t)(row0 + r) * K9_INNER + c) = *(const uint2*)o;
  }
  if (blockIdx.z == 0) {
    // New window = x_{span-3..span-1}; every old tap this thread needs it
    // already read above, so the writes race with nothing.
    float nt[3][K9_VEC];
    for (int t = 0; t < 3; ++t) k9_input(partial, win, s, c, span - 3 + t, nt[t]);
    for (int t = 0; t < 3; ++t) {
      bf16 o[K9_VEC];
      k9_pack(nt[t], o);
      *(uint2*)(win + (size_t)t * K9_INNER + c) = *(const uint2*)o;
    }
  }
}
