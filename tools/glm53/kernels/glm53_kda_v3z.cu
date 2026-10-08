// GLM-5.3 MTP verify KDA v3 Z-SPLIT (bitwise-by-construction exec speedup).
// Derived from glm53_kda_v3.cu (pinned cubin glm53_kda_v3-52d674fccb6d).
//
// Motivation: the pinned kernel runs grid [S,8] (one CTA per (seq,head));
// each CTA moves ~266KB (64KB committed-state read + 192KB f32 stash writes)
// and is per-SM bandwidth-capped (~24.5 GB/s/CTA): ~19.7us at bs1 (8 CTAs).
// The recurrence is elementwise over the 128 v-vectors of a head (warp-local
// k-dim butterflies only), so splitting the v-range across Z CTAs preserves
// every per-element FMA chain EXACTLY. The ONLY cross-v dependency is the
// module_227 gated-norm rstd over the 128 BF16-rounded raw values of a row:
// z-CTAs exchange raw slices through global scratch behind a per-(s,h,row)
// ticket barrier (slab1 fence+ticket idiom) and each recomputes rstd with
// the IDENTICAL butterfly order on IDENTICAL bf16 inputs -> bit-identical.
// Duplicated work (fg3 gates, q/k conv scan, warp0 q/k/decay/beta) is
// recomputed per z-CTA from the same inputs with the same instructions.
//
// Hazard discipline (wedge + livelock lessons from multi-rank bring-up):
// - every cross-CTA wait is BOUNDED (globaltimer watchdog 500ms), with a
//   sticky err word + TRIP COUNTER in scratch for observability; a trip
//   breaks the spin (corrupt but bounded; err/trip are checked by gates).
// - co-residency: grid [S,8,2] <= 128 CTAs for all S<=8 at 1 CTA/SM, so all
//   CTAs of every spin group are always resident. Z=4 would reach 256 CTAs
//   at S=8 (wave-split deadlock class) and is NOT shipped.
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
__device__ __forceinline__ unsigned long long gtimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
// Per-(s,h,row) MONOTONIC ticket barrier (slab1 idiom): arrive returns the
// multiple of Z this launch's arrivals complete, so back-to-back launches on
// a non-zeroed ticket line can never pass a wait on stale arrivals.
// errtrip[0] sticky err, errtrip[1] trip counter (zero-init op scratch).
__device__ __forceinline__ unsigned z_arrive(unsigned *tk, unsigned Z) {
    __threadfence();
    unsigned a = atomicAdd(tk, 1u);
    return (a / Z + 1u) * Z;
}
__device__ __forceinline__ void z_wait(unsigned *tk, unsigned target, unsigned *errtrip) {
    unsigned long long t0 = gtimer();
    while (atomicAdd(tk, 0u) < target) {
        if (gtimer() - t0 > 500000000ull) {
            atomicExch(errtrip, 1u);      // sticky err
            atomicAdd(errtrip + 1, 1u);   // trip counter (observability)
            break;
        }
    }
    __threadfence();
}
// 3-row fg: VERBATIM copy of glm53_kda_v3.cu fg3 (same mma.m16n8k16 sequence,
// same increasing-K order). Duplicated per z-CTA: bit-identical recompute.
template<int NT>
__device__ void fg3(const bf16 *F,const bf16 *wf,const bf16 *wg,int s,int h,bf16 *gates) {
    const int t=threadIdx.x, warp=t/32, lane=t%32;
    const int row=lane/4, pair=(lane%4)*2;
    const bf16 *x=F+(size_t)(3*s+(row<3?row:2))*3336+3080;
    #pragma unroll
    for(int tile=warp;tile<16;tile+=NT/32) {
        int g=tile/8, n=(tile%8)*16;
        const bf16 *w=(g?wg:wf)+(size_t)(h*128+n)*128;
        float c0=0.f,c1=0.f,c2=0.f,c3=0.f;
        #pragma unroll
        for(int k=0;k<128;k+=16) {
            unsigned a0=*reinterpret_cast<const unsigned*>(w+row*128+k+pair);
            unsigned a1=*reinterpret_cast<const unsigned*>(w+(row+8)*128+k+pair);
            unsigned a2=*reinterpret_cast<const unsigned*>(w+row*128+k+pair+8);
            unsigned a3=*reinterpret_cast<const unsigned*>(w+(row+8)*128+k+pair+8);
            unsigned b0=*reinterpret_cast<const unsigned*>(x+128*g+k+pair);
            unsigned b1=*reinterpret_cast<const unsigned*>(x+128*g+k+pair+8);
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
                : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
        }
        int m0=g*128+n+row;
        if(pair==0) {
            gates[0*256+m0]=__float2bfloat16_rn(c0);
            gates[1*256+m0]=__float2bfloat16_rn(c1);
            gates[0*256+m0+8]=__float2bfloat16_rn(c2);
            gates[1*256+m0+8]=__float2bfloat16_rn(c3);
        } else if(pair==2) {
            gates[2*256+m0]=__float2bfloat16_rn(c0);
            gates[2*256+m0+8]=__float2bfloat16_rn(c2);
        }
    }
    __syncthreads();
}

// scratch xchg layout (u32 view, zero-initialized by the runner):
//   [0] err sticky, [1] trip counter, [2..) tickets[S_MAX*8*3] u32,
//   raw_x at byte 1024: bf16[S_MAX*8*3*128] (per (s,h,row): 128 raw values).
#define Z_SMAX 8
#define Z_TK_OFF 2
#define Z_RAW_BYTE 1024
#define Z_SCRATCH_U32 ((Z_RAW_BYTE + Z_SMAX*8*3*128*2 + 3) / 4)

template<int NT,int Z>
__device__ void zbody(const bf16 *F,const bf16 *wf,const bf16 *wg,
 const float *cw,const float *al,const float *dt,const bf16 *nw,
 const bf16 *cs,const float *ss,const int *cl,const int *sl,const int *cu,
 bf16 *out,float *stash_ssm,bf16 *stash_conv,int S,unsigned *xchg,
 bf16 *dbg_raw=nullptr,bf16 *dbg_cv=nullptr,float *dbg_rs=nullptr) {
    wait();
    int s=blockIdx.x,h=blockIdx.y,z=blockIdx.z,t=threadIdx.x,lane=t%32,warp=t/32;
    if(s>=S) { done(); return; }
    int cidx=cl[s],sidx=sl[s],row=cu[s];
    if(cidx<0 || sidx<0 || row!=3*s || cu[s+1]-row!=3) { done(); return; }
    constexpr int VS=128/Z;          // v-vectors handled by this z-CTA
    constexpr int VPR=VS/(NT/32);    // per warp
    const int v0=z*VS;
    unsigned *tickets=xchg+Z_TK_OFF;
    unsigned *errtrip=xchg;
    bf16 *raw_x=reinterpret_cast<bf16*>(reinterpret_cast<char*>(xchg)+Z_RAW_BYTE);
    __shared__ bf16 gates[3][256],qkv[3][384],raw[3][128];
    __shared__ float qkd[384],beta_shared,rstd_v[3];
    fg3<NT>(F,wf,wg,s,h,&gates[0][0]);
    // Conv scan: every z-CTA computes i in [0,256) (q,k; needed for qkd) and
    // its own v-slice i in [256+v0, 256+v0+VS). Stash-tap writes partition:
    // z==0 writes the q,k channel taps, each z writes its own v-slice taps.
    const bf16 *conv=cs+(size_t)cidx*9216;
    bf16 *cstash=stash_conv+(size_t)s*3*3*3072;
    for(int i=t;i<384;i+=NT) {
        bool own = (i<256) || (i-256>=v0 && i-256<v0+VS);
        if(!own) continue;
        bool tapwr = (i>=256) || (z==0);
        int channel=(i/128)*1024+h*128+i%128;
        bf16 r0=conv[channel],r1=conv[3072+channel],r2=conv[6144+channel];
        float x0=f(r0),x1=f(r1),x2=f(r2);
        #pragma unroll
        for(int j=0;j<3;++j) {
            bf16 xb=F[(size_t)(3*s+j)*3336+channel];
            float a=mul(x0,cw[4*channel]);
            a=__fmaf_rn(x1,cw[4*channel+1],a);
            a=__fmaf_rn(x2,cw[4*channel+2],a);
            a=__fmaf_rn(f(xb),cw[4*channel+3],a);
            qkv[j][i]=__float2bfloat16_rn(divfull(a,add(ex(-a),1.f)));
            if(tapwr) {
                cstash[(size_t)j*9216+0*3072+channel]=r1;
                cstash[(size_t)j*9216+1*3072+channel]=r2;
                cstash[(size_t)j*9216+2*3072+channel]=xb;
            }
            x0=x1; x1=x2; x2=f(xb); r1=r2; r2=xb;
        }
    }
    __syncthreads();
    // Recurrence on the z-CTA's v-slice: the state fragment stays in
    // registers across all three rows; committed state read once.
    const float *state=ss+(size_t)sidx*131072+(size_t)h*16384;
    float *stash=stash_ssm+(size_t)s*3*8*16384+(size_t)h*16384;
    float frag[VPR][4];
    unsigned tgt[3];
    #pragma unroll
    for(int vi=0;vi<VPR;++vi) {
        int v=v0+warp*VPR+vi;
        #pragma unroll
        for(int j=0;j<4;++j) frag[vi][j]=state[v*128+lane*4+j];
    }
    #pragma unroll
    for(int j=0;j<3;++j) {
        float q[4],k[4],decay[4],beta;
        if(warp==0) {
            const float A=ex(al[h]);
            #pragma unroll
            for(int i=0;i<4;++i) { q[i]=f(qkv[j][lane*4+i]); k[i]=f(qkv[j][128+lane*4+i]); }
            float qden=sqrtapprox(add(butterfly(dot4(q,q)),1e-6f));
            float kden=sqrtapprox(add(butterfly(dot4(k,k)),1e-6f));
            #pragma unroll
            for(int i=0;i<4;++i) {
                q[i]=mul(divfull(q[i],qden),0.08838834764831845f);
                k[i]=divfull(k[i],kden);
                float x=add(dt[h*128+lane*4+i],f(gates[j][lane*4+i]));
                float neg=__fmaf_rn(-A,x,0.f);
                decay[i]=ex(mul(-5.f,divfull(1.f,add(ex(neg),1.f))));
            }
            beta=sig(f(F[(size_t)(3*s+j)*3336+3072+h]));
            #pragma unroll
            for(int i=0;i<4;++i) {
                qkd[lane*4+i]=q[i]; qkd[128+lane*4+i]=k[i]; qkd[256+lane*4+i]=decay[i];
            }
            if(lane==0) beta_shared=beta;
        }
        __syncthreads();
        #pragma unroll
        for(int i=0;i<4;++i) { q[i]=qkd[lane*4+i]; k[i]=qkd[128+lane*4+i]; decay[i]=qkd[256+lane*4+i]; }
        beta=beta_shared;
        #pragma unroll
        for(int vi=0;vi<VPR;++vi) {
            int v=v0+warp*VPR+vi;
            float zz[4];
            // add(frag,+0) replicates the sequential kernel's per-row reload.
            #pragma unroll
            for(int i=0;i<4;++i) zz[i]=mul(add(frag[vi][i],0.f),decay[i]);
            float delta=mul(__fsub_rn(f(qkv[j][256+v]),butterfly(dot4(k,zz))),beta);
            #pragma unroll
            for(int i=0;i<4;++i) {
                zz[i]=__fmaf_rn(k[i],delta,zz[i]);
                frag[vi][i]=zz[i];
            }
            *reinterpret_cast<float4*>(stash+(size_t)j*8*16384+v*128+lane*4)=
                make_float4(zz[0],zz[1],zz[2],zz[3]);
            float o=butterfly(dot4(q,zz));
            if(lane==0) raw[j][v]=__float2bfloat16_rn(o);
        }
        __syncthreads();
        // Ship this z-CTA's raw slice of row j to the exchange buffer and
        // arrive on the (s,h,row) ticket. No spin here: the wait happens
        // after all three rows' recurrence (row j+1 does not need rstd[j]).
        if(t<VS) raw_x[((size_t)(s*8+h)*3+j)*128+v0+t]=raw[j][v0+t];
        __syncthreads();
        if(t==0) tgt[j]=z_arrive(&tickets[(s*8+h)*3+j], (unsigned)Z);
    }
    // Epilogue: bounded-wait each row's ticket, complete the smem raw row
    // from the exchange buffer, recompute rstd with the IDENTICAL butterfly
    // order on the IDENTICAL bf16 values, write the out slice.
    #pragma unroll
    for(int j=0;j<3;++j) {
        if(t==0) z_wait(&tickets[(s*8+h)*3+j], tgt[j], errtrip);
        __syncthreads();
        if(t<128 && (t<v0 || t>=v0+VS))
            raw[j][t]=raw_x[((size_t)(s*8+h)*3+j)*128+t];
        __syncthreads();
        if(warp==0) {
            float x[8];
            #pragma unroll
            for(int i=0;i<8;++i) x[i]=f(raw[j][(lane%16)*8+i]);
            float zz=mul(x[1],x[1]); zz=__fmaf_rn(x[0],x[0],zz);
            #pragma unroll
            for(int i=2;i<8;++i) zz=__fmaf_rn(x[i],x[i],zz);
            #pragma unroll
            for(int d=8;d;d>>=1) zz=add(zz,__shfl_xor_sync(0xffffffff,zz,d));
            float r=divfull(1.f,sqrtapprox(add(1e-5f,divfull(zz,128.f))));
            if(lane==0) rstd_v[j]=r;
        }
        __syncthreads();
        for(int i=t;i<VS;i+=NT) {
            int v=v0+i;
            float y=mul(mul(f(raw[j][v]),rstd_v[j]),f(nw[v]));
            y=mul(y,sig(f(gates[j][128+v])));
            out[(size_t)(3*s+j)*1024+h*128+v]=__float2bfloat16_rn(y);
        }
        __syncthreads();
    }
    if(dbg_raw) for(int j=0;j<3;++j)
        for(int i=t;i<VS;i+=NT) dbg_raw[(size_t)(3*s+j)*1024+h*128+v0+i]=raw[j][v0+i];
    if(dbg_cv) for(int j=0;j<3;++j)
        for(int i=t;i<384;i+=NT) {
            bool own=(i<256&&z==0)||(i-256>=v0&&i-256<v0+VS);
            if(own) dbg_cv[(size_t)(3*s+j)*3072+(i/128)*1024+h*128+i%128]=qkv[j][i];
        }
    if(dbg_rs && t==0 && z==0) { dbg_rs[s*24+h*3+0]=rstd_v[0]; dbg_rs[s*24+h*3+1]=rstd_v[1]; dbg_rs[s*24+h*3+2]=rstd_v[2]; }
    done();
}
} // namespace
#define ZARGS const bf16 *F,const bf16 *wf,const bf16 *wg,const float *cw, \
 const float *al,const float *dt,const bf16 *nw,const bf16 *cs,const float *ss, \
 const int *cl,const int *sl,const int *cu,bf16 *out,float *stash_ssm, \
 bf16 *stash_conv,int S,unsigned *xchg
#define ZCALL F,wf,wg,cw,al,dt,nw,cs,ss,cl,sl,cu,out,stash_ssm,stash_conv,S,xchg
extern "C" {
__global__ void glm53_kda_verify_fused512z1(ZARGS) { zbody<512,1>(ZCALL); }
__global__ void glm53_kda_verify_fused512z1_debug(ZARGS,bf16 *r,bf16 *c,float *rs) {
    zbody<512,1>(ZCALL,r,c,rs); }
__global__ void glm53_kda_verify_fused512z2(ZARGS) { zbody<512,2>(ZCALL); }
__global__ void glm53_kda_verify_fused512z2_debug(ZARGS,bf16 *r,bf16 *c,float *rs) {
    zbody<512,2>(ZCALL,r,c,rs); }
}
