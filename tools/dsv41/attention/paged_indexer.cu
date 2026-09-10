#include <deep_gemm/impls/sm100_mqa_logits.cuh>
#include <deep_gemm/scheduler/sm100_paged_mqa_logits.cuh>
#include <cuda_bf16.h>
using namespace deep_gemm;
static void __instantiate_kernel() {
    auto meta=reinterpret_cast<void*>(&sched::sm100_paged_mqa_logits_metadata<1,false,false,256,148>);
    auto p64=reinterpret_cast<void*>(&sm100_paged_mqa_logits<1,32,128,64,true,false,false,3,10,256,16,128,256,cutlass::float_e2m1_t,float,float>);
    auto p128=reinterpret_cast<void*>(&sm100_paged_mqa_logits<1,32,128,128,true,false,false,3,10,256,16,128,256,cutlass::float_e2m1_t,float,float>);
}
extern "C" __global__ void dsv41_dense_initialize(float* scores,int rows,int width){
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<(long long)rows*width)scores[i]=-__int_as_float(0x7f800000);
}
extern "C" __global__ void dsv41_dense_mask(float* scores,const int* ends,int rows,int width){
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<(long long)rows*width&&i%width>=ends[i/width])scores[i]=-__int_as_float(0x7f800000);
}
extern "C" __global__ void dsv41_index_weights(const __nv_bfloat16* input,__nv_bfloat16* sparse,float* dense,int rows){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<rows*32){float v=__bfloat162float(input[i])*(1.0f/64.0f);sparse[i]=__float2bfloat16_rn(v);dense[i]=v;}
}
