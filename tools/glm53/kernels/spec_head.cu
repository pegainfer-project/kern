// Greedy head: gather only each rank's (FP32 max, i32 global token id).
// NCCL transports eight bytes as four BF16 lanes; it does NOT do arithmetic.
#include <cuda_bf16.h>
#include <stdint.h>
using bf16=__nv_bfloat16;
struct Choice { float value; int id; };
__device__ __forceinline__ Choice better(Choice a,Choice b) {
    return b.value>a.value || (b.value==a.value && b.id<a.id) ? b : a;
}
extern "C" __global__ void spec_head_local(const bf16* logits, Choice* out, int rank) {
    int s=blockIdx.x,t=threadIdx.x;
    Choice x={-__int_as_float(0x7f800000),0x7fffffff};
    for (int i=t;i<19360;i+=blockDim.x)
        x=better(x,{__bfloat162float(logits[(int64_t)s*19360+i]),rank*19360+i});
    for (int d=16;d;d>>=1) x=better(x,{__shfl_down_sync(~0u,x.value,d),__shfl_down_sync(~0u,x.id,d)});
    __shared__ Choice partial[32];
    if ((t&31)==0) partial[t>>5]=x;
    __syncthreads();
    if (t<32) {
        x=partial[t];
        for (int d=16;d;d>>=1) x=better(x,{__shfl_down_sync(~0u,x.value,d),__shfl_down_sync(~0u,x.id,d)});
        if (!t) out[s]=x;
    }
}
extern "C" __global__ void spec_head_global(const Choice* all, int64_t* tokens, int rows) {
    if (threadIdx.x) return;
    Choice x=all[blockIdx.x];
    for (int r=1;r<8;++r) x=better(x,all[(int64_t)r*rows+blockIdx.x]);
    tokens[blockIdx.x]=x.id;
}
