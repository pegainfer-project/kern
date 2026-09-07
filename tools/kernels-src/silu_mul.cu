// SiLU-and-mul, bit-exact with the mined vLLM act_and_mul kernel (silu in
// f32, rounded to bf16, then a bf16 multiply):
//   out[t, i] = bf16(bf16(g / (1 + expf(-g))) * u),  g = in[t, i], u = in[t, d + i]
// in is [tokens, 2d], out [tokens, d]. Grid [tokens, ceil(d / 2048)], block
// 256: each thread owns 8 consecutive elements (one 16-byte load per operand).
//
//   nvcc -cubin -arch=sm_103a -o kernels/silu_mul.cubin tools/kernels-src/silu_mul.cu
#include <cuda_bf16.h>

extern "C" __global__ void kern_silu_mul_bf16(__nv_bfloat16* __restrict__ out,
                                              const __nv_bfloat16* __restrict__ in, int d) {
    // Programmatic dependent launch: everything before this line may run
    // while the previous kernel is still finishing; nothing produced by it
    // is read or overwritten until the wait returns. The trigger right after
    // lets the next launch start its own prologue (a GEMM's weight stream).
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

  const long long t = blockIdx.x;
  const int i = (blockIdx.y * blockDim.x + threadIdx.x) * 8;
  if (i >= d) return;
  const __nv_bfloat16* g = in + t * 2LL * d + i;
  const uint4 gv = *reinterpret_cast<const uint4*>(g);
  const uint4 uv = *reinterpret_cast<const uint4*>(g + d);
  const __nv_bfloat162* gp = reinterpret_cast<const __nv_bfloat162*>(&gv);
  const __nv_bfloat162* up = reinterpret_cast<const __nv_bfloat162*>(&uv);
  uint4 ov;
  __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(&ov);
#pragma unroll
  for (int j = 0; j < 4; j++) {
    const float gx = __bfloat162float(gp[j].x), gy = __bfloat162float(gp[j].y);
    const __nv_bfloat162 s = __floats2bfloat162_rn(gx / (1.0f + expf(-gx)), gy / (1.0f + expf(-gy)));
    op[j] = __hmul2(s, up[j]);
  }
  *reinterpret_cast<uint4*>(out + t * (long long)d + i) = ov;
}
