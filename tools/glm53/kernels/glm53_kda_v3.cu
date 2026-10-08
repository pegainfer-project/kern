// GLM-5.3 MTP verify (rows=N) fused KDA + per-row state stash; advance select.
// Extends glm53_kda_v2.cu (champion decode fusion, docs/glm53/kda_fusion.md)
// to the rows-per-sequence verify shape with DISABLE_STATE_UPDATE semantics:
// the recurrence runs all rows from the COMMITTED state with no stores,
// the per-row states are stashed to global scratch inside the same kernel, and
// the advance commits stash[nacc-1] by plain copy (no recompute, no pack).
// rows is a runtime argument (1..8: one fg mma tile holds up to 8 B columns);
// at rows=3 every arithmetic sequence is identical to the original rows=3
// kernel (same mma instructions, same loop trip counts, same clamps), so the
// rows=3 bitwise contract is preserved.
//
// Numerics contract (highest-risk fusion in the program):
// - conv taps: spec_conv_verify order (PLAIN mul on tap 0, then FMA 1,2,3),
//   silu = div.full(a, 1 + ex2.approx(-a)), BF16 round per row.
// - recurrence: the exact per-coordinate sequence proven bitwise against
//   module_363 in kda_fusion.md sec.4 (1,0,2,3 dot order; XOR 16..1 trees;
//   sqrt.approx.ftz + div.full q/k norm; never fold decay into the update;
//   add(S,+0) at every row's decay step to replicate the sequential kernel's
//   state reload, including signed-zero behaviour).
// - raw delta output rounds to BF16 before the gated norm (module_227 order).
// Consequence: stash[s][j] is BITWISE the state a sequential 1-row chain
// stores after row j, so the select commit introduces zero new arithmetic.
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
// N-row fg: the same mma.m16n8k16 instructions in the same increasing-K order
// as the v2 decode fg, with activation rows 0..2 in B columns 0..2 (columns
// 3..7 duplicate row 2 and are discarded at rows=3). One tile = 16 output
// channels; B column c carries activation row min(c, rows-1), so up to 8
// rows fit one tile.
template<int NT>
__device__ void fgN(const bf16 *F,const bf16 *wf,const bf16 *wg,int s,int h,bf16 *gates,int rows) {
    const int t=threadIdx.x, warp=t/32, lane=t%32;
    const int row=lane/4, pair=(lane%4)*2;
    // B column = activation row; clamp keeps padding columns in bounds.
    const bf16 *x=F+(size_t)(rows*s+(row<rows?row:rows-1))*3336+3080;
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
        // c0=C[m][pair], c1=C[m][pair+1], c2=C[m+8][pair], c3=C[m+8][pair+1].
        int m0=g*128+n+row;
        // Column pair holds activation row `pair`, pair+1 row `pair+1`.
        if(pair<rows) {
            gates[pair*256+m0]=__float2bfloat16_rn(c0);
            gates[pair*256+m0+8]=__float2bfloat16_rn(c2);
        }
        if(pair+1<rows) {
            gates[(pair+1)*256+m0]=__float2bfloat16_rn(c1);
            gates[(pair+1)*256+m0+8]=__float2bfloat16_rn(c3);
        }
    }
    __syncthreads();
}

template<int NT,bool FUSED>
__device__ void vbody(const bf16 *F,const bf16 *wf,const bf16 *wg,
 const float *cw,const float *al,const float *dt,const bf16 *nw,
 const bf16 *cs,const float *ss,const int *cl,const int *sl,const int *cu,
 bf16 *out,float *stash_ssm,bf16 *stash_conv,int S,int rows,
 bf16 *dbg_raw=nullptr,bf16 *dbg_cv=nullptr,float *dbg_rs=nullptr) {
    wait();
    int s=blockIdx.x,h=blockIdx.y,t=threadIdx.x,lane=t%32,warp=t/32;
    if(s>=S) { done(); return; }
    int cidx=cl[s],sidx=sl[s],row=cu[s];
    // Verify contract: exactly `rows` sequence-major rows per group.
    if(cidx<0 || sidx<0 || rows<1 || rows>8 || row!=rows*s || cu[s+1]-row!=rows) { done(); return; }
    __shared__ bf16 gates[8][256],qkv[8][384],raw[8][128];
    __shared__ float qkd[384],beta_shared,rstd_v[8];
    if constexpr(FUSED) fgN<NT>(F,wf,wg,s,h,&gates[0][0],rows);
    else {
        // wf = forget [T,1024], wg = gproj [T,1024] (original cuBLAS outputs).
        for(int i=t;i<128;i+=NT) {
            for(int j=0;j<rows;++j) {
                gates[j][i]=wf[(size_t)(rows*s+j)*1024+h*128+i];
                gates[j][128+i]=wg[(size_t)(rows*s+j)*1024+h*128+i];
            }
        }
    }
    // Conv scan: spec_conv_verify arithmetic. Reads committed taps; no store.
    // Per row j the raw input taps {x_{j-2},x_{j-1},x_j} go to the stash.
    const bf16 *conv=cs+(size_t)cidx*9216;
    bf16 *cstash=stash_conv+(size_t)s*rows*9216;
    for(int i=t;i<384;i+=NT) {
        int channel=(i/128)*1024+h*128+i%128;
        bf16 r0=conv[channel],r1=conv[3072+channel],r2=conv[6144+channel];
        float x0=f(r0),x1=f(r1),x2=f(r2);
        for(int j=0;j<rows;++j) {
            bf16 xb=F[(size_t)(rows*s+j)*3336+channel];
            float a=mul(x0,cw[4*channel]);
            a=__fmaf_rn(x1,cw[4*channel+1],a);
            a=__fmaf_rn(x2,cw[4*channel+2],a);
            a=__fmaf_rn(f(xb),cw[4*channel+3],a);
            qkv[j][i]=__float2bfloat16_rn(divfull(a,add(ex(-a),1.f)));
            cstash[(size_t)j*9216+0*3072+channel]=r1;
            cstash[(size_t)j*9216+1*3072+channel]=r2;
            cstash[(size_t)j*9216+2*3072+channel]=xb;
            x0=x1; x1=x2; x2=f(xb); r1=r2; r2=xb;
        }
    }
    __syncthreads();
    // Recurrence: the state fragment stays in registers across all three rows;
    // the committed state is read once, never written here.
    const float *state=ss+(size_t)sidx*131072+(size_t)h*16384;
    float *stash=stash_ssm+(size_t)s*rows*8*16384+(size_t)h*16384;
    constexpr int VPR=128/(NT/32);
    float frag[VPR][4];
    #pragma unroll
    for(int vi=0;vi<VPR;++vi) {
        int v=warp*VPR+vi;
        #pragma unroll
        for(int j=0;j<4;++j) frag[vi][j]=state[v*128+lane*4+j];
    }
    for(int j=0;j<rows;++j) {
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
            beta=sig(f(F[(size_t)(rows*s+j)*3336+3072+h]));
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
            int v=warp*VPR+vi;
            float z[4];
            // add(frag,+0) replicates the sequential kernel's per-row reload.
            #pragma unroll
            for(int i=0;i<4;++i) z[i]=mul(add(frag[vi][i],0.f),decay[i]);
            float delta=mul(__fsub_rn(f(qkv[j][256+v]),butterfly(dot4(k,z))),beta);
            #pragma unroll
            for(int i=0;i<4;++i) {
                z[i]=__fmaf_rn(k[i],delta,z[i]);
                frag[vi][i]=z[i];
            }
            *reinterpret_cast<float4*>(stash+(size_t)j*8*16384+v*128+lane*4)=
                make_float4(z[0],z[1],z[2],z[3]);
            float o=butterfly(dot4(q,z));
            if(lane==0) raw[j][v]=__float2bfloat16_rn(o);
        }
        __syncthreads();
        // module_227 gated norm on the BF16-rounded raw output of row j.
        if(warp==0) {
            float x[8];
            #pragma unroll
            for(int i=0;i<8;++i) x[i]=f(raw[j][(lane%16)*8+i]);
            float z=mul(x[1],x[1]); z=__fmaf_rn(x[0],x[0],z);
            #pragma unroll
            for(int i=2;i<8;++i) z=__fmaf_rn(x[i],x[i],z);
            #pragma unroll
            for(int d=8;d;d>>=1) z=add(z,__shfl_xor_sync(0xffffffff,z,d));
            float r=divfull(1.f,sqrtapprox(add(1e-5f,divfull(z,128.f))));
            if(lane==0) rstd_v[j]=r;
        }
        __syncthreads();
        for(int i=t;i<128;i+=NT) {
            float y=mul(mul(f(raw[j][i]),rstd_v[j]),f(nw[i]));
            y=mul(y,sig(f(gates[j][128+i])));
            out[(size_t)(rows*s+j)*1024+h*128+i]=__float2bfloat16_rn(y);
        }
        __syncthreads();
    }
    if(dbg_raw) for(int j=0;j<rows;++j)
        for(int i=t;i<128;i+=NT) dbg_raw[(size_t)(rows*s+j)*1024+h*128+i]=raw[j][i];
    if(dbg_cv) for(int j=0;j<rows;++j)
        for(int i=t;i<384;i+=NT) dbg_cv[(size_t)(rows*s+j)*3072+(i/128)*1024+h*128+i%128]=qkv[j][i];
    if(dbg_rs && t==0) for(int j=0;j<rows;++j) dbg_rs[(s*8+h)*rows+j]=rstd_v[j];
    done();
}
} // namespace
#define VARGS const bf16 *F,const bf16 *wf,const bf16 *wg,const float *cw, \
 const float *al,const float *dt,const bf16 *nw,const bf16 *cs,const float *ss, \
 const int *cl,const int *sl,const int *cu,bf16 *out,float *stash_ssm, \
 bf16 *stash_conv,int S,int rows
#define VCALL F,wf,wg,cw,al,dt,nw,cs,ss,cl,sl,cu,out,stash_ssm,stash_conv,S,rows
#define VK(mode,nt,fused) \
__global__ void glm53_kda_verify_##mode##nt(VARGS) { vbody<nt,fused>(VCALL); } \
__global__ void glm53_kda_verify_##mode##nt##_debug(VARGS,bf16 *r,bf16 *c,float *rs) { vbody<nt,fused>(VCALL,r,c,rs); }
extern "C" {
VK(core,256,false)
VK(core,512,false)
VK(core,1024,false)
VK(fused,256,true)
VK(fused,512,true)
VK(fused,1024,true)
// Advance: commit the stashed per-row states selected by nacc. Guards mirror
// spec_conv_advance: invalid/pad sequences and non-positive lines are skipped.
__global__ void glm53_kda_select(const float *stash_ssm,const bf16 *stash_conv,
 float *ss,bf16 *cs,const int *sl,const int *cl,const int *nacc,const int *valid,int rows) {
    wait();
    int s=blockIdx.x,h=blockIdx.y,t=threadIdx.x;
    int n=nacc[s];
    if(rows<1 || rows>8 || !valid[rows*s] || n<1 || n>rows) { done(); return; }
    int si=sl[s],ci=cl[s];
    if(si<=0 || ci<=0) { done(); return; }
    const float4 *src=reinterpret_cast<const float4*>(stash_ssm+((size_t)(s*rows+n-1)*8+h)*16384);
    float4 *dst=reinterpret_cast<float4*>(ss+(size_t)si*131072+(size_t)h*16384);
    for(int i=t;i<16384/4;i+=blockDim.x) dst[i]=src[i];
    const bf16 *csrc=stash_conv+(size_t)(s*rows+n-1)*9216;
    bf16 *cdst=cs+(size_t)ci*9216;
    for(int i=t;i<3*384;i+=blockDim.x) {
        int tap=i/384,c=h*384+i%384;
        cdst[tap*3072+c]=csrc[tap*3072+c];
    }
    done();
}
}
