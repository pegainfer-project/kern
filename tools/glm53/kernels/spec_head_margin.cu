// Diagnostic-only MTP head. It preserves spec_head's BF16 logits and tie rule,
// but emits the global top-1 token and top-1 minus top-2 margin.
#include <cuda_bf16.h>
#include <stdint.h>
using bf16=__nv_bfloat16;
struct Choice { float value; int id; };
struct Pair { Choice a; Choice b; };
__device__ __forceinline__ Choice invalid() {
    return {-__int_as_float(0x7f800000), 0x7fffffff};
}
__device__ __forceinline__ Choice better(Choice a, Choice b) {
    return b.value>a.value || (b.value==a.value && b.id<a.id) ? b : a;
}
__device__ __forceinline__ Pair insert(Pair p, Choice x) {
    Choice a=p.a, b=p.b;
    if (x.value>a.value || (x.value==a.value && x.id<a.id)) {
        b=a; a=x;
    } else if (x.value>b.value || (x.value==b.value && x.id<b.id)) {
        b=x;
    }
    return {a,b};
}
extern "C" __global__ void spec_head_margin_local(const bf16* logits, Pair* out, int rank) {
    int s=blockIdx.x, t=threadIdx.x;
    Pair p={invalid(),invalid()};
    for (int i=t;i<19360;i+=blockDim.x)
        p=insert(p,{__bfloat162float(logits[(int64_t)s*19360+i]),rank*19360+i});
    __shared__ Pair partial[1024];
    partial[t]=p;
    __syncthreads();
    if (!t) {
        p={invalid(),invalid()};
        for (int i=0;i<blockDim.x;++i) {
            p=insert(p,partial[i].a);
            p=insert(p,partial[i].b);
        }
        out[s]=p;
    }
}
extern "C" __global__ void spec_head_margin_global(const Pair* all, int64_t* tokens, float* margin, int rows) {
    if (threadIdx.x) return;
    int row=blockIdx.x;
    Pair p={invalid(),invalid()};
    for (int r=0;r<8;++r) {
        const Pair x=all[(int64_t)r*rows+row];
        p=insert(p,x.a);
        p=insert(p,x.b);
    }
    tokens[row]=p.a.id;
    margin[row]=p.a.value-p.b.value;
}