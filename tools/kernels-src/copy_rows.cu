// Plain bf16 row copies (the ATen `contiguous()` / row-select copies that
// vLLM does between kernels and that the manifest cannot express with a
// byte offset alone).
//
//   kern_copy_rows_bf16: dst[r, :width] = src[r, :width], grid.x = rows
//     (strided view -> contiguous, e.g. the GDN z gate out of the fused
//     qkvz projection)
//   kern_last_row_bf16:  dst[0, :width] = src[rows-1, :width], grid 1
//     (the final-norm / lm_head row of a prefill chunk; `rows-1` is not in
//     the manifest expression set, so the kernel takes `rows`)
//   kern_own_rows:       dst[j, :] = src[min(blocks[rank] + j, rows - 1), :],
//     grid.x = this rank's share of the rows, row_bytes a multiple of 16
//     (a tray's slice of a chunk in natural order, the blocks from
//     kern_k3_mla_chunk_plan; the base is a runtime value no manifest
//     expression reaches, so the kernel reads the plan). One entry for
//     every dtype: rows are bytes.
//
//   nvcc -cubin -arch=sm_103a -o kernels/copy_rows.cubin tools/kernels-src/copy_rows.cu
#include <cuda_bf16.h>

extern "C" __global__ void kern_copy_rows_bf16(
    __nv_bfloat16* __restrict__ dst, const __nv_bfloat16* __restrict__ src,
    int dst_stride, int src_stride, int width) {
  const long long r = blockIdx.x;
  const __nv_bfloat16* s = src + r * src_stride;
  __nv_bfloat16* d = dst + r * dst_stride;
  for (int j = threadIdx.x; j < width; j += blockDim.x) d[j] = s[j];
}

extern "C" __global__ void kern_last_row_bf16(
    __nv_bfloat16* __restrict__ dst, const __nv_bfloat16* __restrict__ src,
    int src_stride, int width, int rows) {
  const __nv_bfloat16* s = src + (long long)(rows - 1) * src_stride;
  for (int j = threadIdx.x; j < width; j += blockDim.x) dst[j] = s[j];
}

extern "C" __global__ void kern_own_rows(
    void* __restrict__ dst, const void* __restrict__ src, const int* __restrict__ blocks,
    int rank, int row_bytes, int rows) {
  const int j = blockIdx.x;
  const int r = min(blocks[rank] + j, rows - 1);
  const uint4* s = reinterpret_cast<const uint4*>(static_cast<const char*>(src) + (size_t)r * row_bytes);
  uint4* d = reinterpret_cast<uint4*>(static_cast<char*>(dst) + (size_t)j * row_bytes);
  for (int p = threadIdx.x; p < row_bytes / 16; p += blockDim.x) d[p] = s[p];
}
