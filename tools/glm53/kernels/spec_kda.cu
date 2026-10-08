// KDA speculative conv reads the COMMITTED three-tap history without stores.
// Unlike gdn_advance.cu, the anchor is NOT already committed: advance by nacc,
// not nacc-1. Existing conv state layout stays [line,3,3072] bf16 (18432 B).
// Two distinct constants share the value 3 at k=2: the round width (tokens
// per sequence, now the rows arg) and the conv window k-1=3 (fixed by the
// kernel size; the x0/x1/x2 tap shift and t<3 window test below keep it).
#include <cuda_bf16.h>
#include <stdint.h>
using bf16=__nv_bfloat16;
__device__ __forceinline__ float spec_exp(float x) {
    float y;
    asm("mul.f32 %0, %1, 0f3FB8AA3B;" : "=f"(y) : "f"(x));
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(y));
    return y;
}
__device__ __forceinline__ float spec_div(float x,float y) {
    float z; asm("div.full.f32 %0, %1, %2;" : "=f"(z) : "f"(x),"f"(y)); return z;
}
extern "C" __global__ void spec_conv_verify(
    const bf16* fused, const float* weight, const bf16* state,
    const int* lines, bf16* out, const int* valid, int rows) {
    int s=blockIdx.x, c=blockIdx.y*blockDim.x+threadIdx.x;
    if (c>=3072) return;
    const bf16* st=state+(int64_t)lines[s]*9216+c;
    float x0=__bfloat162float(st[0]), x1=__bfloat162float(st[3072]), x2=__bfloat162float(st[6144]);
    #pragma unroll
    for (int j=0;j<rows;++j) {
        float x=__bfloat162float(fused[(int64_t)(rows*s+j)*3336+c]);
        // Triton enables FMA on the additions but the first multiply is plain.
        float a=x0*weight[c*4];
        a=fmaf(x1,weight[c*4+1],a);
        a=fmaf(x2,weight[c*4+2],a);
        a=fmaf(x,weight[c*4+3],a);
        out[(int64_t)(rows*s+j)*3072+c]=__float2bfloat16(spec_div(a,1.f+spec_exp(-a)));
        x0=x1; x1=x2; x2=x;
    }
}
extern "C" __global__ void spec_conv_advance(
    const bf16* fused, bf16* state, const int* lines,
    const int* nacc, const int* valid, int rows) {
    int s=blockIdx.x, c=blockIdx.y*blockDim.x+threadIdx.x;
    int n=nacc[s];
    if (c>=3072 || n<1 || n>rows || !valid[rows*s] || lines[s]<=0) return;
    bf16* st=state+(int64_t)lines[s]*9216+c;
    // Read all overlapping source taps before any destination write.
    bf16 x[3];
    #pragma unroll
    for (int j=0;j<3;++j) {
        int t=j+n;
        x[j]=t<3 ? st[t*3072] : fused[(int64_t)(rows*s+t-3)*3336+c];
    }
    #pragma unroll
    for (int j=0;j<3;++j) st[j*3072]=x[j];
}

// Diagnostic only: deterministic two-word FNV hash of each post-conv row.
// The production manifest does not call this entry.
extern "C" __global__ void spec_conv_hash(const bf16* q, float* out, int rows) {
    int s = blockIdx.x;
    if (s >= rows || threadIdx.x != 0) return;
    uint32_t h = 2166136261u;
    const bf16* row = q + (int64_t)s * 3072;
    for (int c = 0; c < 3072; ++c) {
        h ^= (uint32_t)(*reinterpret_cast<const uint16_t *>(&row[c]));
        h *= 16777619u;
    }
    out[s] = (float)(h % 16777213u);
}


// Diagnostic only: hash a row with an explicit pitch and width.
extern "C" __global__ void spec_row_hash(const bf16* x, float* out, int rows, int cols, int stride) {
    int s = blockIdx.x;
    if (s >= rows || threadIdx.x != 0) return;
    uint32_t h = 2166136261u;
    const bf16* row = x + (int64_t)s * stride;
    for (int c = 0; c < cols; ++c) {
        h ^= (uint32_t)(*reinterpret_cast<const uint16_t *>(&row[c]));
        h *= 16777619u;
    }
    out[s] = (float)(h % 16777213u);
}


// Diagnostic only: hash integer token rows.
extern "C" __global__ void spec_i64_hash(const int64_t* x, float* out, int rows, int stride) {
    int s = blockIdx.x;
    if (s >= rows || threadIdx.x != 0) return;
    uint32_t h = 2166136261u;
    const int64_t* row = x + (int64_t)s * stride;
    for (int c = 0; c < stride; ++c) {
        uint64_t u = (uint64_t)row[c];
        h ^= (uint32_t)u; h *= 16777619u;
        h ^= (uint32_t)(u >> 32); h *= 16777619u;
    }
    out[s] = (float)(h % 16777213u);
}
