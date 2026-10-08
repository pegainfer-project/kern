// MTP-only norms. Source arithmetic: sglang fused_eh_norm.cuh and the
// flashinfer FusedAddRMSNormKernel used by sgl_kernel.elementwise on CUDA.
// 512 threads, eight contiguous BF16 elements per thread, XOR reductions.
// In add+RMS the variance AND normalized input use the FP32 residual sum.
// Only the persistent residual is rounded to BF16 before normalization.
#include <cuda_bf16.h>
#include <stdint.h>
using bf16=__nv_bfloat16;
__device__ __forceinline__ float norm_factor(float square, float* smem) {
    for (int off=16;off;off>>=1) square+=__shfl_xor_sync(~0u,square,off);
    if (!(threadIdx.x&31)) smem[threadIdx.x>>5]=square;
    __syncthreads();
    if (threadIdx.x<32) {
        square=threadIdx.x<16 ? smem[threadIdx.x] : 0.f;
        for (int off=16;off;off>>=1) square+=__shfl_xor_sync(~0u,square,off);
        if (!threadIdx.x) smem[16]=rsqrtf(square/4096.f+1e-5f);
    }
    __syncthreads();
    float factor=smem[16];
    __syncthreads(); // allow the next EH half to reuse smem
    return factor;
}
extern "C" __global__ void spec_eh_norm(
    const bf16* embed,const bf16* previous,const bf16* ew,const bf16* hw,bf16* out) {
    __shared__ float smem[17];
    int c=threadIdx.x*8;
    for (int half=0;half<2;++half) {
        const bf16* x=(half ? previous : embed)+(int64_t)blockIdx.x*4096+c;
        const bf16* w=(half ? hw : ew)+c;
        float v[8], square=0.f;
        #pragma unroll
        for (int j=0;j<8;++j) { v[j]=__bfloat162float(x[j]); square+=v[j]*v[j]; }
        float f=norm_factor(square,smem);
        #pragma unroll
        for (int j=0;j<8;++j)
            out[(int64_t)blockIdx.x*8192+half*4096+c+j]=__float2bfloat16(v[j]*f*__bfloat162float(w[j]));
    }
}
extern "C" __global__ void spec_add_norm(
    bf16* residual,const bf16* sub,const bf16* weight,bf16* out) {
    __shared__ float smem[17];
    int c=threadIdx.x*8;
    int64_t off=(int64_t)blockIdx.x*4096+c;
    float v[8],square=0.f;
    #pragma unroll
    for (int j=0;j<8;++j) {
        v[j]=__bfloat162float(residual[off+j])+__bfloat162float(sub[off+j]);
        square+=v[j]*v[j];
        residual[off+j]=__float2bfloat16(v[j]);
    }
    float f=norm_factor(square,smem);
    #pragma unroll
    for (int j=0;j<8;++j)
        out[off+j]=__float2bfloat16(v[j]*f*__bfloat162float(weight[c+j]));
}
