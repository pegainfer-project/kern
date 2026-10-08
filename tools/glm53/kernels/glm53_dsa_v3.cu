// DSA absorb rows-scaling fix (2026-09-28): v2 re-reads the weight
// slice per 4-row group and is latency-bound at low occupancy. v3 covers R
// rows per CTA (same per-output math: same k-loop, same fmaf order, same
// warp_sum tree, same rn conversion -> bitwise identical outputs).
#include <cuda_bf16.h>
using bf16 = __nv_bfloat16;
__device__ __forceinline__ void wait_dep() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void release_dep() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ float warp_sum(float v) {
    for (int d=16; d; d>>=1) v += __shfl_down_sync(~0u,v,d);
    return v;
}
// 16 output columns x R rows/CTA. Identical per-element math to absorb<>.
template<int N,int K,int R> __device__ void absorb_v3(const bf16* a,const bf16* w,bf16* out,int rows) {
    wait_dep();
    int lane=threadIdx.x&31, warp=threadIdx.x>>5;
    int n0=blockIdx.x*16+warp*4, h=blockIdx.y, r0=blockIdx.z*R;
    float sum[R][4];
    #pragma unroll
    for(int r=0;r<R;++r)
        #pragma unroll
        for(int j=0;j<4;++j) sum[r][j]=0.f;
    #pragma unroll
    for(int k=lane;k<K;k+=32) {
        float av[R];
        #pragma unroll
        for(int r=0;r<R;++r) av[r]=r0+r<rows?__bfloat162float(a[(size_t)(r0+r)*8*K+h*K+k]):0.f;
        #pragma unroll
        for(int j=0;j<4;++j) {
            float v=__bfloat162float(w[((size_t)h*N+n0+j)*K+k]);
            #pragma unroll
            for(int r=0;r<R;++r) sum[r][j]=fmaf(av[r],v,sum[r][j]);
        }
    }
    #pragma unroll
    for(int r=0;r<R;++r) {
        #pragma unroll
        for(int j=0;j<4;++j) {
            float v=warp_sum(sum[r][j]);
            if(lane==0 && r0+r<rows) out[(size_t)(r0+r)*8*N+h*N+n0+j]=__float2bfloat16_rn(v);
        }
    }
    release_dep();
}
extern "C" __global__ __launch_bounds__(128) void glm53_dsa_w_kc_v3_r1 (const bf16* a,const bf16* w,bf16* c,int rows) { absorb_v3<512,256, 1>(a,w,c,rows); }
extern "C" __global__ __launch_bounds__(128) void glm53_dsa_w_kc_v3_r2 (const bf16* a,const bf16* w,bf16* c,int rows) { absorb_v3<512,256, 2>(a,w,c,rows); }
extern "C" __global__ __launch_bounds__(128) void glm53_dsa_w_kc_v3_r8 (const bf16* a,const bf16* w,bf16* c,int rows) { absorb_v3<512,256, 8>(a,w,c,rows); }
extern "C" __global__ __launch_bounds__(128) void glm53_dsa_w_kc_v3_r16(const bf16* a,const bf16* w,bf16* c,int rows) { absorb_v3<512,256,16>(a,w,c,rows); }
extern "C" __global__ __launch_bounds__(128) void glm53_dsa_w_vc_v3_r1 (const bf16* a,const bf16* w,bf16* c,int rows) { absorb_v3<256,512, 1>(a,w,c,rows); }
extern "C" __global__ __launch_bounds__(128) void glm53_dsa_w_vc_v3_r2 (const bf16* a,const bf16* w,bf16* c,int rows) { absorb_v3<256,512, 2>(a,w,c,rows); }
extern "C" __global__ __launch_bounds__(128) void glm53_dsa_w_vc_v3_r8 (const bf16* a,const bf16* w,bf16* c,int rows) { absorb_v3<256,512, 8>(a,w,c,rows); }
extern "C" __global__ __launch_bounds__(128) void glm53_dsa_w_vc_v3_r16(const bf16* a,const bf16* w,bf16* c,int rows) { absorb_v3<256,512,16>(a,w,c,rows); }
// reference copy of the v2 production kernel for the bitwise A/B
template<int N,int K> __device__ void absorb_ref(const bf16* a,const bf16* w,bf16* out,int rows) {
    int lane=threadIdx.x&31, warp=threadIdx.x>>5;
    int n0=blockIdx.x*16+warp*4, h=blockIdx.y, r0=blockIdx.z*4;
    float sum[4][4]={};
    #pragma unroll
    for(int k=lane;k<K;k+=32) {
        float av[4];
        #pragma unroll
        for(int r=0;r<4;++r) av[r]=r0+r<rows?__bfloat162float(a[(size_t)(r0+r)*8*K+h*K+k]):0.f;
        #pragma unroll
        for(int j=0;j<4;++j) {
            float v=__bfloat162float(w[((size_t)h*N+n0+j)*K+k]);
            #pragma unroll
            for(int r=0;r<4;++r) sum[r][j]=fmaf(av[r],v,sum[r][j]);
        }
    }
    #pragma unroll
    for(int r=0;r<4;++r) {
        #pragma unroll
        for(int j=0;j<4;++j) {
            float v=warp_sum(sum[r][j]);
            if(lane==0 && r0+r<rows) out[(size_t)(r0+r)*8*N+h*N+n0+j]=__float2bfloat16_rn(v);
        }
    }
}
extern "C" __global__ __launch_bounds__(128) void glm53_dsa_w_kc_v2_ref(const bf16* a,const bf16* w,bf16* c,int rows) { absorb_ref<512,256>(a,w,c,rows); }
extern "C" __global__ __launch_bounds__(128) void glm53_dsa_w_vc_v2_ref(const bf16* a,const bf16* w,bf16* c,int rows) { absorb_ref<256,512>(a,w,c,rows); }
// a-preload variant: hoist the 4-row x K/32 a-values into registers before the
// k-loop (identical fmaf order per accumulator; only load scheduling differs).
template<int N,int K> __device__ void absorb_v4(const bf16* a,const bf16* w,bf16* out,int rows) {
    wait_dep();
    int lane=threadIdx.x&31, warp=threadIdx.x>>5;
    int n0=blockIdx.x*16+warp*4, h=blockIdx.y, r0=blockIdx.z*4;
    constexpr int IT=K/32;
    float av[4][IT];
    #pragma unroll
    for(int i=0;i<IT;++i) {
        int k=lane+i*32;
        #pragma unroll
        for(int r=0;r<4;++r) av[r][i]=r0+r<rows?__bfloat162float(a[(size_t)(r0+r)*8*K+h*K+k]):0.f;
    }
    float sum[4][4]={};
    #pragma unroll
    for(int i=0;i<IT;++i) {
        int k=lane+i*32;
        #pragma unroll
        for(int j=0;j<4;++j) {
            float v=__bfloat162float(w[((size_t)h*N+n0+j)*K+k]);
            #pragma unroll
            for(int r=0;r<4;++r) sum[r][j]=fmaf(av[r][i],v,sum[r][j]);
        }
    }
    #pragma unroll
    for(int r=0;r<4;++r) {
        #pragma unroll
        for(int j=0;j<4;++j) {
            float v=warp_sum(sum[r][j]);
            if(lane==0 && r0+r<rows) out[(size_t)(r0+r)*8*N+h*N+n0+j]=__float2bfloat16_rn(v);
        }
    }
    release_dep();
}
extern "C" __global__ __launch_bounds__(128) void glm53_dsa_w_kc_v4(const bf16* a,const bf16* w,bf16* c,int rows) { absorb_v4<512,256>(a,w,c,rows); }
extern "C" __global__ __launch_bounds__(128) void glm53_dsa_w_vc_v4(const bf16* a,const bf16* w,bf16* c,int rows) { absorb_v4<256,512>(a,w,c,rows); }
