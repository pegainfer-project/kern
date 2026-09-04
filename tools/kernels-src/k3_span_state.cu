// K10 `k3_span_state` — copies a row's KDA recurrent state between its line
// and a flat buffer, either way. The span kernel (FlashKDA, K8) reads and
// writes its state through TMA descriptors whose base is fixed at load, so
// the state of whichever sequence is extending is staged into a fixed
// buffer before the span and back into its line after.
//
//   extern "C" __global__ void kern_k3_span_state(
//       void* kda_base, const int* line_index, long long line_bytes,   // line_index[*span_at]'s line, rec at offset 0
//       const int* span_at,   // [1]  the span's first batch row
//       float* buf,        // [HEADS][128][128]
//       int to_line);      // 0: buf = rec;  1: rec = buf
//
//   grid (HEADS, 32, 1)   block 128   (one float4 per thread: 4096 per head)
#ifndef HEADS
#define HEADS 96
#endif

extern "C" __global__ __launch_bounds__(128) void kern_k3_span_state(
    void* __restrict__ kda_base, const int* __restrict__ line_index, long long line_bytes,
    const int* __restrict__ span_at, float* __restrict__ buf, int to_line) {
  float4* rec = (float4*)((char*)kda_base + (long long)line_index[span_at[0]] * line_bytes) + (size_t)blockIdx.x * 4096;
  float4* b = (float4*)buf + (size_t)blockIdx.x * 4096;
  const int j = blockIdx.y * 128 + threadIdx.x;
  if (to_line) {
    rec[j] = b[j];
  } else {
    b[j] = rec[j];
  }
}
