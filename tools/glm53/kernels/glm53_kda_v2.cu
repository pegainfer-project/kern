// GLM-5.3 TP8 decode only. See docs/glm53/kda_fusion.md.
// Build without --use_fast_math. PTX operations below mirror module_363/227.
#include <cuda_bf16.h>
using bf16 = __nv_bfloat16;
namespace {
__device__ __forceinline__ float f(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float add(float x,float y) { return __fadd_rn(x,y); }
__device__ __forceinline__ float mul(float x,float y) { return __fmul_rn(x,y); }
__device__ __forceinline__ float ex(float x) {
    float z=mul(x,__int_as_float(0x3fb8aa3b)), r;
    asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(z)); return r;
}
__device__ __forceinline__ float divfull(float x,float y) {
    float r; asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(x),"f"(y)); return r;
}
__device__ __forceinline__ float sqrtapprox(float x) {
    float r; asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); return r;
}
__device__ __forceinline__ float sig(float x) {
    return divfull(1.f,add(ex(__fsub_rn(0.f,x)),1.f));
}
__device__ __forceinline__ float butterfly(float x) {
    #pragma unroll
    for(int d=16;d;d>>=1) x=add(x,__shfl_xor_sync(0xffffffff,x,d));
    return x;
}
// Triton first multiplies element 1, then FMA elements 0, 2, 3.
__device__ __forceinline__ float dot4(const float *a,const float *b) {
    float x=mul(a[1],b[1]);
    x=__fmaf_rn(a[0],b[0],x); x=__fmaf_rn(a[2],b[2],x);
    return __fmaf_rn(a[3],b[3],x);
}
__device__ __forceinline__ void wait() {
    asm volatile("griddepcontrol.wait;" ::: "memory");
}
__device__ __forceinline__ void done() {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
// 16 independent 16x8x128 products: f/g x 8 N tiles. All eight
// B columns replicate the same activation. Only output column 0 is retained.
// Direct MMA register loads avoid shared-memory bank conflicts and staging.
// Each increasing-K MMA is the same m16n8k16 instruction used by WMMA.
template<int NT>
__device__ void fg(const bf16 *F,const bf16 *wf,const bf16 *wg,int s,int h,bf16 *fgout) {
    const int t=threadIdx.x, warp=t/32, lane=t%32;
    const int row=lane/4, pair=(lane%4)*2;
    #pragma unroll
    for(int tile=warp;tile<16;tile+=NT/32) {
        int g=tile/8, n=(tile%8)*16;
        const bf16 *w=(g?wg:wf)+(h*128+n)*128;
        const bf16 *x=F+(size_t)s*3336+3080+128*g;
        float c0=0.f,c1=0.f,c2=0.f,c3=0.f;
        #pragma unroll
        for(int k=0;k<128;k+=16) {
            unsigned a0=*reinterpret_cast<const unsigned*>(w+row*128+k+pair);
            unsigned a1=*reinterpret_cast<const unsigned*>(w+(row+8)*128+k+pair);
            unsigned a2=*reinterpret_cast<const unsigned*>(w+row*128+k+pair+8);
            unsigned a3=*reinterpret_cast<const unsigned*>(w+(row+8)*128+k+pair+8);
            unsigned b0=*reinterpret_cast<const unsigned*>(x+k+pair);
            unsigned b1=*reinterpret_cast<const unsigned*>(x+k+pair+8);
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
                : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
        }
        if(pair==0) {
            fgout[g*128+n+row]=__float2bfloat16_rn(c0);
            fgout[g*128+n+row+8]=__float2bfloat16_rn(c2);
        }
    }
    __syncthreads();
}

template<int NT,bool FUSED>
__device__ void body(const bf16 *F,const bf16 *wf,const bf16 *wg,
 const float *cw,const float *al,const float *dt,const bf16 *nw,
 bf16 *cs,float *ss,const int *cl,const int *sl,const int *cu,bf16 *out,int S,bf16 *debug_raw=nullptr,
 bf16 *debug_cv=nullptr,float *debug_rs=nullptr) {
    wait();
    int s=blockIdx.x,h=blockIdx.y,t=threadIdx.x,lane=t%32,warp=t/32;
    if(s>=S) { done(); return; }
    int cidx=cl[s],sidx=sl[s],row=cu[s];
    // Decode contract: one token per group, in group order; no span/MTP.
    if(cidx<0 || sidx<0 || row!=s || cu[s+1]-row!=1) { done(); return; }
    __shared__ bf16 gates[256],qkv[384],raw[128];
    __shared__ float rstd;
    __shared__ float qkd[NT>=512?384:1], beta_shared;
    if constexpr(FUSED) fg<NT>(F,wf,wg,s,h,gates);
    else {
        for(int j=t;j<128;j+=NT) { gates[j]=wf[s*1024+h*128+j]; gates[128+j]=wg[s*1024+h*128+j]; }
    }
    bf16 *conv=cs+(size_t)cidx*9216;
    // Exactly one owner per conv channel. No duplicate q/k writers across V tiles.
    for(int j=t;j<384;j+=NT) {
        int channel=(j/128)*1024+h*128+j%128;
        bf16 x=F[(size_t)s*3336+channel];
        bf16 c0=conv[channel],c1=conv[3072+channel],c2=conv[6144+channel];
        float z=__fmaf_rn(f(c0),cw[4*channel],0.f);
        z=__fmaf_rn(f(c1),cw[4*channel+1],z);
        z=__fmaf_rn(f(c2),cw[4*channel+2],z);
        z=__fmaf_rn(f(x),cw[4*channel+3],z);
        qkv[j]=__float2bfloat16_rn(divfull(z,add(ex(-z),1.f)));
        conv[channel]=c1; conv[3072+channel]=c2; conv[6144+channel]=x;
    }
    __syncthreads();
    float q[4],k[4],decay[4];
    float beta;
    if(NT<512 || warp==0) {
    const float A=ex(al[h]);
    #pragma unroll
    for(int j=0;j<4;++j) { q[j]=f(qkv[lane*4+j]); k[j]=f(qkv[128+lane*4+j]); }
    float qden=sqrtapprox(add(butterfly(dot4(q,q)),1e-6f));
    float kden=sqrtapprox(add(butterfly(dot4(k,k)),1e-6f));
    #pragma unroll
    for(int j=0;j<4;++j) {
        q[j]=mul(divfull(q[j],qden),0.08838834764831845f);
        k[j]=divfull(k[j],kden);
        float x=add(dt[h*128+lane*4+j],f(gates[lane*4+j]));
        float neg=__fmaf_rn(-A,x,0.f);
        decay[j]=ex(mul(-5.f,divfull(1.f,add(ex(neg),1.f))));
    }
    beta=sig(f(F[(size_t)s*3336+3072+h]));
        if constexpr(NT>=512) {
            #pragma unroll
            for(int j=0;j<4;++j) {
                qkd[lane*4+j]=q[j]; qkd[128+lane*4+j]=k[j]; qkd[256+lane*4+j]=decay[j];
            }
            if(lane==0) beta_shared=beta;
        }
    }
    if constexpr(NT>=512) {
        __syncthreads();
        #pragma unroll
        for(int j=0;j<4;++j) { q[j]=qkd[lane*4+j]; k[j]=qkd[128+lane*4+j]; decay[j]=qkd[256+lane*4+j]; }
        beta=beta_shared;
    }
    float *state=ss+(size_t)sidx*131072+h*16384;
    // Per warp contiguous V range; each lane owns K=4*lane+[0,1,2,3].
    // All V rows are independent. Never fold decay into the update FMA.
    #pragma unroll
    for(int vi=0;vi<128/(NT/32);++vi) {
        int v=warp*(128/(NT/32))+vi;
        float z[4];
        #pragma unroll
        for(int j=0;j<4;++j) z[j]=mul(add(state[v*128+lane*4+j],0.f),decay[j]);
        float delta=mul(__fsub_rn(f(qkv[256+v]),butterfly(dot4(k,z))),beta);
        #pragma unroll
        for(int j=0;j<4;++j) {
            z[j]=__fmaf_rn(k[j],delta,z[j]);
            state[v*128+lane*4+j]=z[j];
        }
        float o=butterfly(dot4(q,z));
        if(lane==0) raw[v]=__float2bfloat16_rn(o);
    }
    __syncthreads();
    // module_227: 8 contiguous values/lane, 16 lanes/head; 1,0,2,...7
    // local square accumulation, then XOR 8,4,2,1. Use BF16-rounded raw.
    if(warp==0) {
        float x[8];
        #pragma unroll
        for(int j=0;j<8;++j) x[j]=f(raw[(lane%16)*8+j]);
        float z=mul(x[1],x[1]); z=__fmaf_rn(x[0],x[0],z);
        #pragma unroll
        for(int j=2;j<8;++j) z=__fmaf_rn(x[j],x[j],z);
        #pragma unroll
        for(int d=8;d;d>>=1) z=add(z,__shfl_xor_sync(0xffffffff,z,d));
        float r=divfull(1.f,sqrtapprox(add(1e-5f,divfull(z,128.f))));
        if(lane==0) rstd=r;
    }
    __syncthreads();
    for(int j=t;j<128;j+=NT) {
        float y=mul(mul(f(raw[j]),rstd),f(nw[j]));
        y=mul(y,sig(f(gates[128+j])));
        out[(size_t)s*1024+h*128+j]=__float2bfloat16_rn(y);
    }
    if(debug_raw) for(int j=t;j<128;j+=NT) debug_raw[s*1024+h*128+j]=raw[j];
    if(debug_cv) for(int j=t;j<384;j+=NT) debug_cv[s*3072+(j/128)*1024+h*128+j%128]=qkv[j];
    if(debug_rs && t==0) debug_rs[s*8+h]=rstd;
    // Trigger only after all output/state stores; no epoch counter or polling.
    done();
}
// Stream each weight vector once; keep up to 16 independent row accumulators.
// Each row retains the original direct candidate reduction order.
template<int M>
__device__ void direct(const bf16 *x,const bf16 *w,bf16 *y,int S,int N,int K) {
    int lane=threadIdx.x%32,n=blockIdx.x*4+threadIdx.x/32;
    if(n>=N) return;
    float p[M][4]={};
    for(int k=lane*4;k<K;k+=128) {
        float ww[4];
        #pragma unroll
        for(int j=0;j<4;++j) ww[j]=f(w[(size_t)n*K+k+j]);
        #pragma unroll
        for(int s=0;s<M;++s) if(s<S) {
            #pragma unroll
            for(int j=0;j<4;++j) p[s][j]=__fmaf_rn(f(x[(size_t)s*K+k+j]),ww[j],p[s][j]);
        }
    }
    #pragma unroll
    for(int s=0;s<M;++s) if(s<S) {
        float z=butterfly(add(add(p[s][0],p[s][1]),add(p[s][2],p[s][3])));
        if(lane==0) y[(size_t)s*N+n]=__float2bfloat16_rn(z);
    }
}
} // namespace
#define ARGS const bf16 *F,const bf16 *wf,const bf16 *wg,const float *cw, \
 const float *al,const float *dt,const bf16 *nw,bf16 *cs,float *ss, \
 const int *cl,const int *sl,const int *cu,bf16 *out,int S
#define CALL F,wf,wg,cw,al,dt,nw,cs,ss,cl,sl,cu,out,S
extern "C" {
__global__ void glm53_kda_fused128(ARGS) { body<128,true>(CALL); }
__global__ void glm53_kda_fused256(ARGS) { body<256,true>(CALL); }
__global__ void glm53_kda_fused512(ARGS) { body<512,true>(CALL); }
__global__ void glm53_kda_fused1024(ARGS) { body<1024,true>(CALL); }
__global__ void glm53_kda_core512(ARGS) { body<512,false>(CALL); }
__global__ void glm53_kda_core1024(ARGS) { body<1024,false>(CALL); }
__global__ void glm53_kda_core128(ARGS) { body<128,false>(CALL); }
__global__ void glm53_kda_core256(ARGS) { body<256,false>(CALL); }
__global__ void glm53_kda_core128_debug(ARGS,bf16 *r,bf16 *c,float *rs) { body<128,false>(CALL,r,c,rs); }
__global__ void glm53_kda_core256_debug(ARGS,bf16 *r,bf16 *c,float *rs) { body<256,false>(CALL,r,c,rs); }
__global__ void glm53_kda_core512_debug(ARGS,bf16 *r,bf16 *c,float *rs) { body<512,false>(CALL,r,c,rs); }
__global__ void glm53_kda_core1024_debug(ARGS,bf16 *r,bf16 *c,float *rs) { body<1024,false>(CALL,r,c,rs); }
__global__ void glm53_kda_fused128_debug(ARGS,bf16 *r,bf16 *c,float *rs) { body<128,true>(CALL,r,c,rs); }
__global__ void glm53_kda_fused256_debug(ARGS,bf16 *r,bf16 *c,float *rs) { body<256,true>(CALL,r,c,rs); }
__global__ void glm53_kda_fused512_debug(ARGS,bf16 *r,bf16 *c,float *rs) { body<512,true>(CALL,r,c,rs); }
__global__ void glm53_kda_fused1024_debug(ARGS,bf16 *r,bf16 *c,float *rs) { body<1024,true>(CALL,r,c,rs); }
// Standalone diagnostic of the exact fg_b arithmetic used in the fused CTA.
__global__ void glm53_kda_fg(const bf16 *F,const bf16 *wf,const bf16 *wg,
 bf16 *fo,bf16 *go,int S) {
    wait(); int s=blockIdx.x,h=blockIdx.y,t=threadIdx.x;
    __shared__ bf16 g[256];
    if(s<S) {
        fg<128>(F,wf,wg,s,h,g);
        fo[s*1024+h*128+t]=g[t]; go[s*1024+h*128+t]=g[128+t];
    }
    done();
}
// Direct no-split-K candidate. 4 warps, 4 output rows/CTA, all S rows reused.
// This deliberately has a separate tolerance gate; its reduction is not cuBLAS.
__global__ void glm53_kda_qkv_direct(const bf16 *x,const bf16 *w,bf16 *y,int S,int N,int K) {
    wait();
    if(S==1) direct<1>(x,w,y,S,N,K);
    else if(S<=2) direct<2>(x,w,y,S,N,K);
    else if(S<=4) direct<4>(x,w,y,S,N,K);
    else if(S<=8) direct<8>(x,w,y,S,N,K);
    else direct<16>(x,w,y,S,N,K);
    done();
}
}
