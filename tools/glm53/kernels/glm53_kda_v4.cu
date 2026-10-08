// GLM-5.3 MTP verify KDA v4 STAGE-A: warp-specialized software pipeline.
// Derived from glm53_kda_v3.cu (pinned cubin glm53_kda_v3-52d674fccb6d);
// fg3 + conv scan + all per-element arithmetic copied VERBATIM (mechanical
// section diff in the gate script). Grid [seqs,8] UNCHANGED.
//
// v3's serial per-row chain [warp0 qkd prep -> 16-warp recurrence -> warp0
// rstd -> out] leaves 15/16 warps idle for ~1.9us x 3 rows. v4 splits the
// CTA after the joint fg3+conv into a prep crew (warps 0-7) and a
// recurrence crew (warps 8-15): qkd/beta for row j+1 is computed DURING
// the recurrence of row j into a double buffer, and rstd[j]+out[j] run on
// the prep crew DURING recurrence of row j+1. Handoff via named barriers
// (arrive/sync pairs, count 512). The recurrence is BW-bound (stash
// writes), so 8 warps sustain it at the per-SM cap as well as 16.
//
// Bitwise by construction: every value is produced by the SAME instruction
// stream as v3 (same butterflies, exp/div chain, FMA order, bf16 rounding);
// only warp assignment, barrier structure, and buffer duplication change.
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
// Named-barrier handoff (bar 0 reserved for __syncthreads). bar.arrive is
// non-blocking, bar.sync blocks; crew-crossing barriers count NT (prep crew
// 256 arrivals + recurrence crew NT-256 syncs, or vice versa).
template<int NT>
__device__ __forceinline__ void bar_arrive(int id) {
    asm volatile("bar.arrive %0, %1;" :: "r"(id), "n"(NT) : "memory");
}
template<int NT>
__device__ __forceinline__ void bar_sync(int id) {
    asm volatile("bar.sync %0, %1;" :: "r"(id), "n"(NT) : "memory");
}
enum { B_QKD0=1, B_QKD1, B_FREE0, B_RAW0, B_RAW1, B_RAW2 };

// Tile range [tb, tb+tc): tb=-1 = all 16 tiles (v4p). z-split: z owns tiles
// [z*(16/ZW), +16/ZW) (ZW=2: g-halves; ZW=4: g-quarters, z0/z1 -> wf, z2/z3
// -> wg). Per-tile mma stream unchanged, so any subset produces values
// identical to the full run (bitwise).
template<int NT>
__device__ void fg3(const bf16 *F,const bf16 *wf,const bf16 *wg,int s,int h,
                    bf16 *gates,int tb,int tc) {
    const int t=threadIdx.x, warp=t/32, lane=t%32;
    const int row=lane/4, pair=(lane%4)*2;
    // B column = activation row; clamp keeps padding columns in bounds.
    const bf16 *x=F+(size_t)(3*s+(row<3?row:2))*3336+3080;
    const int t0=(tb<0?0:tb), te=(tb<0?16:t0+tc);
    #pragma unroll
    for(int tile=t0+warp;tile<te;tile+=NT/32) {
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

// qkd prep for row j into buffer b: VERBATIM v3 warp0 block (same
// butterflies, exp/div chain, bf16 reads). Executed by CTA warp 0.
template<int NT>
__device__ void qkd_prep(const bf16 *F,const bf16 *gates,const bf16 *qkv,
 const float *al,const float *dt,int s,int h,int j,float *qkd,float *beta_v) {
    const int t=threadIdx.x, warp=t/32, lane=t%32;
    if(warp==0) {
        const bf16 (*gates3)[256]=reinterpret_cast<const bf16 (*)[256]>(gates);
        const bf16 (*qkv3)[384]=reinterpret_cast<const bf16 (*)[384]>(qkv);
        float q[4],k[4],decay[4],beta;
        const float A=ex(al[h]);
        #pragma unroll
        for(int i=0;i<4;++i) { q[i]=f(qkv3[j][lane*4+i]); k[i]=f(qkv3[j][128+lane*4+i]); }
        float qden=sqrtapprox(add(butterfly(dot4(q,q)),1e-6f));
        float kden=sqrtapprox(add(butterfly(dot4(k,k)),1e-6f));
        #pragma unroll
        for(int i=0;i<4;++i) {
            q[i]=mul(divfull(q[i],qden),0.08838834764831845f);
            k[i]=divfull(k[i],kden);
            float x=add(dt[h*128+lane*4+i],f(gates3[j][lane*4+i]));
            float neg=__fmaf_rn(-A,x,0.f);
            decay[i]=ex(mul(-5.f,divfull(1.f,add(ex(neg),1.f))));
        }
        beta=sig(f(F[(size_t)(3*s+j)*3336+3072+h]));
        #pragma unroll
        for(int i=0;i<4;++i) {
            qkd[lane*4+i]=q[i]; qkd[128+lane*4+i]=k[i]; qkd[256+lane*4+i]=decay[i];
        }
        if(lane==0) *beta_v=beta;
    }
}

template<int NT>
__device__ void pbody(const bf16 *F,const bf16 *wf,const bf16 *wg,
 const float *cw,const float *al,const float *dt,const bf16 *nw,
 const bf16 *cs,const float *ss,const int *cl,const int *sl,const int *cu,
 bf16 *out,float *stash_ssm,bf16 *stash_conv,int S,
 bf16 *dbg_raw=nullptr,bf16 *dbg_cv=nullptr,float *dbg_rs=nullptr) {
    wait();
    int s=blockIdx.x,h=blockIdx.y,t=threadIdx.x,lane=t%32,warp=t/32;
    if(s>=S) { done(); return; }
    int cidx=cl[s],sidx=sl[s],row=cu[s];
    // Verify contract: exactly three sequence-major rows per group.
    if(cidx<0 || sidx<0 || row!=3*s || cu[s+1]-row!=3) { done(); return; }
    __shared__ bf16 gates[3][256],qkv[3][384],raw[3][128];
    __shared__ float qkd[2][384],beta_v[2],rstd_v[3];
    fg3<NT>(F,wf,wg,s,h,&gates[0][0],-1,0);
    // Conv scan: VERBATIM v3 (all 512 threads). Reads committed taps; no
    // store. Per row j the raw input taps {x_{j-2},x_{j-1},x_j} go to stash.
    const bf16 *conv=cs+(size_t)cidx*9216;
    bf16 *cstash=stash_conv+(size_t)s*3*3*3072;
    for(int i=t;i<384;i+=NT) {
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
            cstash[(size_t)j*9216+0*3072+channel]=r1;
            cstash[(size_t)j*9216+1*3072+channel]=r2;
            cstash[(size_t)j*9216+2*3072+channel]=xb;
            x0=x1; x1=x2; x2=f(xb); r1=r2; r2=xb;
        }
    }
    __syncthreads();
    const float *state=ss+(size_t)sidx*131072+(size_t)h*16384;
    float *stash=stash_ssm+(size_t)s*3*8*16384+(size_t)h*16384;
    if(dbg_cv) for(int j=0;j<3;++j)
        for(int i=t;i<384;i+=NT) dbg_cv[(size_t)(3*s+j)*3072+(i/128)*1024+h*128+i%128]=qkv[j][i];
    if(warp<8) {
        // ===== prep crew (warps 0-7) =====
        qkd_prep<NT>(F,&gates[0][0],&qkv[0][0],al,dt,s,h,0,qkd[0],&beta_v[0]);
        __threadfence_block();
        bar_arrive<NT>(B_QKD0);                 // qkd[0] ready
        qkd_prep<NT>(F,&gates[0][0],&qkv[0][0],al,dt,s,h,1,qkd[1],&beta_v[1]);
        __threadfence_block();
        bar_arrive<NT>(B_QKD1);                 // qkd[1] ready
        bar_sync<NT>(B_FREE0);                  // recurrence crew consumed qkd[0]
        qkd_prep<NT>(F,&gates[0][0],&qkv[0][0],al,dt,s,h,2,qkd[0],&beta_v[0]);
        __threadfence_block();
        bar_arrive<NT>(B_QKD0);                 // qkd[0] reused for row 2
        // rstd + out per row, overlapped with the next row's recurrence.
        #pragma unroll
        for(int j=0;j<3;++j) {
            bar_sync<NT>(B_RAW0+j);             // raw[j] complete
            __threadfence_block();
            // module_227 gated norm on the BF16-rounded raw output of row j:
            // VERBATIM v3 warp0 block.
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
            // sync the prep crew only (bar 3, 256 threads) so rstd_v[j] is
            // visible to the out writers below.
            asm volatile("bar.sync 7, 256;" ::: "memory");
            // out row j: VERBATIM v3 formula; 256 prep threads (elementwise).
            for(int i=t;i<128;i+=NT/2) {
                float y=mul(mul(f(raw[j][i]),rstd_v[j]),f(nw[i]));
                y=mul(y,sig(f(gates[j][128+i])));
                out[(size_t)(3*s+j)*1024+h*128+i]=__float2bfloat16_rn(y);
            }
            if(dbg_raw) for(int i=t;i<128;i+=NT/2)
                dbg_raw[(size_t)(3*s+j)*1024+h*128+i]=raw[j][i];
        }
        if(dbg_rs && t==0) { dbg_rs[s*24+h*3+0]=rstd_v[0]; dbg_rs[s*24+h*3+1]=rstd_v[1]; dbg_rs[s*24+h*3+2]=rstd_v[2]; }
        done();
        return;
    }
    // ===== recurrence crew (warps 8-15) =====
    {
        constexpr int VPR=128/(NT/32/2);    // 16 v-vectors per warp at NT=512
        const int w=warp-8;
        float frag[VPR][4];
        #pragma unroll
        for(int vi=0;vi<VPR;++vi) {
            int v=w*VPR+vi;
            #pragma unroll
            for(int j=0;j<4;++j) frag[vi][j]=state[v*128+lane*4+j];
        }
        #pragma unroll
        for(int j=0;j<3;++j) {
            bar_sync<NT>(j==1?B_QKD1:B_QKD0);   // qkd buffer for row j ready
            __threadfence_block();
            const int b=j&1;
            float q[4],k[4],decay[4],beta;
            #pragma unroll
            for(int i=0;i<4;++i) { q[i]=qkd[b][lane*4+i]; k[i]=qkd[b][128+lane*4+i]; decay[i]=qkd[b][256+lane*4+i]; }
            beta=beta_v[b];
            if(j==0) bar_arrive<NT>(B_FREE0);   // qkd[0] consumed into registers
            // VERBATIM v3 recurrence (same instruction stream per element).
            #pragma unroll
            for(int vi=0;vi<VPR;++vi) {
                int v=w*VPR+vi;
                float z[4];
                // add(frag,+0) replicates the sequential kernel's reload.
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
            __threadfence_block();
            bar_arrive<NT>(B_RAW0+j);           // raw[j] complete
        }
        done();
    }
}
// ==========================================================================
// STAGE B (z2s): prep-SHARING z-split, grid [seqs,8,2]. z0 computes fg3+conv
// ONCE and ships a 3.0KB package (qkv q/k, z1's v-slice, forget gates, z1's
// gproj half) through global scratch; z1 skips fg3/conv entirely (the v1z
// killer: no duplicated 64KB weight stream, no duplicated mma chain). Both
// z-CTAs recompute qkd locally (compute-only, identical instructions ->
// identical values), run the v4 pipeline on their OWN v-half, and exchange
// raw halves per row exactly like v1z (redundant rstd recompute, identical
// butterfly order over identical bf16 values -> bit-identical).
//
// Ticket discipline (bring-up lesson): PKG and RAW are ZW-arrival phase-
// monotonic tickets; every z arrives AFTER shipping its own regions and
// waits its own arrival's phase target (deadlock-free by construction: the
// arrive precedes the wait in program order; bounded by a 500ms globaltimer
// watchdog with sticky err + trip counter). Requires CTA CO-RESIDENCY: all
// ZW CTAs of an (s,h) group must be in the same scheduling wave (zsbody uses
// 8 named barriers so 2 CTAs/SM fit -> capacity 264 >= any S<=8 grid).
// xchg scratch layout (u32 view, zero-init op scratch):
//   [0] err sticky, [1] trip counter
//   PKG tickets: u32[2 .. 2+64)        per (s,h): s*8+h
//   RAW tickets: u32[66 .. 66+192)     per (s,h,row): (s*8+h)*3+j
//   package: byte 1040 (>= 258*4 AND 16B-aligned), per (s,h) 3072B:
//            [g0 @0 | g1 @768 | q @1536 | k @2304], region = bf16[j][128]
//   NB: the old byte-1024 base ALIASED raw-ticket slots 256/257 (latent
//   corruption: survived pre-rewrite only because the clobbered phase base
//   happened to be even). Never place the package below 1032+alignment pad.
//   raw_x: byte V4_RAW_BYTE: bf16[(s*8+h)*3+j][128]
#define V4_PKG_TK 2
#define V4_RAW_TK 66
#define V4_PKG_BYTE 1040
#define V4_PKG_STRIDE 3072
#define V4_RAW_BYTE (V4_PKG_BYTE + 64 * V4_PKG_STRIDE)
#define V4_SCRATCH_U32 ((V4_RAW_BYTE + 64 * 3 * 128 * 2) / 4)

__device__ __forceinline__ unsigned long long gtimer4() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
// Ticket release/acquire (PTX memory model): the release RMW orders ALL
// happens-before-prior stores (incl. other threads' via the CTA barrier
// edge) — no per-thread fence storm. The acquire poll is morally strong and
// nanosleep-backed-off so spin RMWs never queue ahead of an arriver's RMW
// (the +2us/row convoy found by the phase profiler).
__device__ __forceinline__ unsigned v4_arrive(unsigned *tk, unsigned Z) {
    unsigned a;
    asm volatile("atom.add.release.gpu.global.u32 %0, [%1], 1;"
                 : "=r"(a) : "l"(tk) : "memory");
    return (a / Z + 1u) * Z;
}
__device__ __forceinline__ void v4_wait(unsigned *tk, unsigned target, unsigned *errtrip) {
    unsigned long long t0 = gtimer4();
    unsigned a;
    for (;;) {
        asm volatile("atom.add.acquire.gpu.global.u32 %0, [%1], 0;"
                     : "=r"(a) : "l"(tk) : "memory");
        if (a >= target) break;
        if (gtimer4() - t0 > 500000000ull) {
            atomicExch(errtrip, 1u);
            atomicAdd(errtrip + 1, 1u);
            break;
        }
        __nanosleep(128);
    }
}

// ZW-way z-split (grid [seqs,8,ZW]): the prefix is DISTRIBUTED — every z
// computes a 16/ZW-tile fg3 share (ZW=2: g-halves; ZW=4: quarters) plus its
// conv share (ZW=2: q or k block + v half; ZW=4: balanced 64-ch q/k slice +
// v quarter). All per-element arithmetic is VERBATIM v3 (pure work
// partition; mma tiles and conv channels are independent). The package is 4
// regions [g0|g1|q|k] x 768B (fits the existing 3072B stride); owners ship,
// everyone blind-unpacks ALL regions (shipped bytes are bit-identical to
// locally computed ones by construction).
// Tickets: PKG and RAW are ZW-arrival phase-monotonic (atom.add release/
// acquire, nanosleep backoff). ZW=4 halves the per-SM stash-store bytes per
// row (16KB) — the store drain is the recurrence wall (2.05us at ZW=2).
template<int NT,int ZW>
__device__ void zsbody(const bf16 *F,const bf16 *wf,const bf16 *wg,
 const float *cw,const float *al,const float *dt,const bf16 *nw,
 const bf16 *cs,const float *ss,const int *cl,const int *sl,const int *cu,
 bf16 *out,float *stash_ssm,bf16 *stash_conv,int S,unsigned *xchg,
 bf16 *dbg_raw=nullptr,bf16 *dbg_cv=nullptr,float *dbg_rs=nullptr,
 unsigned long long *prof=nullptr) {
    wait();
    int s=blockIdx.x,h=blockIdx.y,z=blockIdx.z,t=threadIdx.x,lane=t%32,warp=t/32;
    // Phase profiler stamps: globaltimer (ns, cross-SM comparable). PROF from
    // prep-crew thread 0, PROFR from recurrence-crew thread 256. Slots:
    //  0 entry | 1 post-PKG-wait | 2 prep post-qkd0
    //  3/5/7 prep post-RAW-wait rows 0/1/2 | 4/6/8 rec post-compute rows 0/1/2
    //  9 prep end | 10 rec end | 11 post-unpack (z<2: post-fg3 is slot 1-0)
    #define PROF(i) if(prof && t==0) prof[((size_t)(s*8+h)*gridDim.z+z)*12+(i)]=gtimer4();
    #define PROFR(i) if(prof && t==256) prof[((size_t)(s*8+h)*gridDim.z+z)*12+(i)]=gtimer4();
    PROF(0)
    if(s>=S) { done(); return; }
    int cidx=cl[s],sidx=sl[s],row=cu[s];
    if(cidx<0 || sidx<0 || row!=3*s || cu[s+1]-row!=3) { done(); return; }
    __shared__ bf16 gates[3][256],qkv[3][384],raw[3][128];
    __shared__ float qkd[2][384],beta_v[2],rstd_v[3];
    __shared__ unsigned tgt_s[3], tgt_pkg;
    __shared__ unsigned qkd0_free;          // replaces named barrier B_FREE0
    if(t==0) qkd0_free=0u;                  // (covered by the prefix bar 0)
    unsigned *pkg_tk=xchg+V4_PKG_TK, *raw_tk=xchg+V4_RAW_TK, *errtrip=xchg;
    char *pkg=reinterpret_cast<char*>(xchg)+V4_PKG_BYTE+(size_t)(s*8+h)*V4_PKG_STRIDE;
    bf16 *raw_x=reinterpret_cast<bf16*>(reinterpret_cast<char*>(xchg)+V4_RAW_BYTE);
    constexpr int VW=128/ZW;                // own v-slice width (64 or 32)
    const int v0=z*VW;

    // ---- distributed prefix -------------------------------------------------
    fg3<NT>(F,wf,wg,s,h,&gates[0][0],z*(16/ZW),16/ZW);  // z's fg3 share
    {
        // Owned conv channels (VERBATIM v3 body per channel). ZW=2: z0 owns
        // the q block, z1 the k block, each z its v half. ZW=4 (balanced):
        // z0 q[0:64), z2 q[64:128), z1 k[0:64), z3 k[64:128), + own v
        // quarter — 96 channels per z.
        int oc, qkb;
        if constexpr(ZW==2) { oc=128+VW; qkb=(z==0?0:128); }
        else                { oc=64+VW;  qkb=(z==0?0:(z==1?128:(z==2?64:192))); }
        const bf16 *conv=cs+(size_t)cidx*9216;
        bf16 *cstash=stash_conv+(size_t)s*3*3*3072;
        for(int idx=t;idx<oc;idx+=NT) {
            int i=(idx<oc-VW)?qkb+idx:256+v0+(idx-(oc-VW));
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
                cstash[(size_t)j*9216+0*3072+channel]=r1;
                cstash[(size_t)j*9216+1*3072+channel]=r2;
                cstash[(size_t)j*9216+2*3072+channel]=xb;
                x0=x1; x1=x2; x2=f(xb); r1=r2; r2=xb;
            }
        }
        __syncthreads();
        if(dbg_cv) for(int j=0;j<3;++j)
            for(int idx=t;idx<oc;idx+=NT) {
                int i=(idx<oc-VW)?qkb+idx:256+v0+(idx-(oc-VW));
                dbg_cv[(size_t)(3*s+j)*3072+(i/128)*1024+h*128+i%128]=qkv[j][i];
            }
    }
    // ---- package ship: gate slice + q/k slice per owner z -------------------
    // pkg layout [g0 @0 | g1 @768 | q @1536 | k @2304], region = [j][128].
    // ZW=2: z0 ships g0+q (whole), z1 g1+k. ZW=4: two 64-elem slices per z:
    // gate (region z>>1, slice z&1) + qk (region 2+(z&1), slice z>>1).
    if constexpr(ZW==2) {
        if(t<96) {
            int reg=t/48, u=t%48, j=u/16, e=(u%16)*8;
            const bf16 *src=(reg==0?(z==0?&gates[j][e]:&gates[j][128+e]):
                                     (z==0?&qkv[j][e]:&qkv[j][128+e]));
            *reinterpret_cast<uint4*>(pkg+(z==0?reg*1536:768+reg*1536)+u*16)=
                *reinterpret_cast<const uint4*>(src);
        }
    } else {
        if(t<48) {
            int sl=t/24, u=t%24, j=u/8, e=(u%8)*8;
            int reg, sofs;
            if(sl==0) { reg=z>>1;   sofs=(z&1)*64; }
            else      { reg=2+(z&1); sofs=(z>>1)*64; }
            const bf16 *src=(reg<2?&gates[j][reg*128+sofs+e]:
                                    &qkv[j][(reg-2)*128+sofs+e]);
            *reinterpret_cast<uint4*>(pkg+reg*768+(sofs+j*128+e)*2)=
                *reinterpret_cast<const uint4*>(src);
        }
    }
    // bar 0 orders all region stores before t==0's release-ticket RMW
    // (morally strong chain: bar.sync -> atom.release -> atom.acquire).
    __syncthreads();
    if(t==0) tgt_pkg=v4_arrive(&pkg_tk[s*8+h],ZW);
    if(t==0) v4_wait(&pkg_tk[s*8+h],tgt_pkg,errtrip);
    __syncthreads();
    PROF(1)
    // Blind unpack of all 4 regions (identical bytes for own regions).
    if(t<192) {
        int reg=t/48, u=t%48, j=u/16, e=(u%16)*8;
        uint4 v=*reinterpret_cast<const uint4*>(pkg+reg*768+u*16);
        bf16 *dst=(reg==0?&gates[j][e]:(reg==1?&gates[j][128+e]:
                   (reg==2?&qkv[j][e]:&qkv[j][128+e])));
        *reinterpret_cast<uint4*>(dst)=v;
    }
    __syncthreads();
    PROF(11)
    // From here both z-CTAs run the v4 pipeline on their own v-half.
    // Z2S_STASH_H: stash h-stride (f32 elems). Power-of-2 default aliases L2
    // slices across the 16 co-resident CTAs; padded variant is an experiment
    // (integration would pad the tensor + select op in lockstep).
#ifndef Z2S_STASH_H
#define Z2S_STASH_H 16384
#endif
    const float *state=ss+(size_t)sidx*131072+(size_t)h*16384;
    float *stash=stash_ssm+(size_t)s*3*8*Z2S_STASH_H+(size_t)h*Z2S_STASH_H;
    if(warp<8) {
        // ===== prep crew (warps 0-7) =====
        qkd_prep<NT>(F,&gates[0][0],&qkv[0][0],al,dt,s,h,0,qkd[0],&beta_v[0]);
        PROF(2)
        __threadfence_block();
        bar_arrive<NT>(B_QKD0);
        qkd_prep<NT>(F,&gates[0][0],&qkv[0][0],al,dt,s,h,1,qkd[1],&beta_v[1]);
        __threadfence_block();
        bar_arrive<NT>(B_QKD1);
        // qkd[0] reuse for row 2: wait until the rec crew consumed row 0's
        // qkd (smem flag set AFTER the row-0 ship bar 7, so ALL rec threads'
        // qkd[0] reads happen-before; drops named barrier B_FREE0 so zsbody
        // fits 8 barriers -> 2 CTAs/SM -> ZW=4 co-resident at S=8).
        if(t==0) { while(atomicAdd(&qkd0_free,0u)==0u) __nanosleep(32); }
        asm volatile("bar.sync 3, 256;" ::: "memory");
        qkd_prep<NT>(F,&gates[0][0],&qkv[0][0],al,dt,s,h,2,qkd[0],&beta_v[0]);
        __threadfence_block();
        bar_arrive<NT>(B_QKD0);
        #pragma unroll
        for(int j=0;j<3;++j) {
            bar_sync<NT>(B_RAW0+j);             // raw[j] own-half + ticket target up
            if(t==0) v4_wait(&raw_tk[(s*8+h)*3+j],tgt_s[j],errtrip);
            PROF(3+j*2)
            // prep-crew internal sync (bar 3, 256)
            asm volatile("bar.sync 3, 256;" ::: "memory");
            __threadfence_block();
            // complete smem raw[j] from the exchange buffer (other z slices)
            if(t<128-VW) {
                int blk=t/VW; if(blk>=z) blk+=1;
                int e=blk*VW+t%VW;
                raw[j][e]=raw_x[((size_t)(s*8+h)*3+j)*128+e];
            }
            asm volatile("bar.sync 3, 256;" ::: "memory");
            // module_227 gated norm: VERBATIM v3 warp0 block.
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
            asm volatile("bar.sync 3, 256;" ::: "memory");
            // out row j, own v-slice: VERBATIM v3 formula (elementwise).
            for(int i=t;i<VW;i+=NT/2) {
                int v=v0+i;
                float y=mul(mul(f(raw[j][v]),rstd_v[j]),f(nw[v]));
                y=mul(y,sig(f(gates[j][128+v])));
                out[(size_t)(3*s+j)*1024+h*128+v]=__float2bfloat16_rn(y);
            }
            if(dbg_raw) for(int i=t;i<VW;i+=NT/2)
                dbg_raw[(size_t)(3*s+j)*1024+h*128+v0+i]=raw[j][v0+i];
        }
        if(dbg_rs && t==0 && z==0) { dbg_rs[s*24+h*3+0]=rstd_v[0]; dbg_rs[s*24+h*3+1]=rstd_v[1]; dbg_rs[s*24+h*3+2]=rstd_v[2]; }
        PROF(9)
        done();
        return;
    }
    // ===== recurrence crew (warps 8..NT/32), own v-slice =====
    // Rec warps = NT/32-8 (8 at NT=512); VPR = VW/8 (8 at ZW=2, 4 at ZW=4).
    // The butterfly reductions are warp-local over the k dim (128 elems =
    // 4/lane x 32 lanes) and independent of the v->warp assignment, so any
    // VPR keeps the EXACT per-element instruction stream (bitwise).
    {
        constexpr int VPR=VW/(NT/32-8);
        constexpr int RCNT=NT-256;          // rec crew size for bar 7
        const int w=warp-8;
        float frag[VPR][4];
        #pragma unroll
        for(int vi=0;vi<VPR;++vi) {
            int v=v0+w*VPR+vi;
            #pragma unroll
            for(int j=0;j<4;++j) frag[vi][j]=state[v*128+lane*4+j];
        }
        #pragma unroll
        for(int j=0;j<3;++j) {
            bar_sync<NT>(j==1?B_QKD1:B_QKD0);
            __threadfence_block();
            const int b=j&1;
            float q[4],k[4],decay[4],beta;
            #pragma unroll
            for(int i=0;i<4;++i) { q[i]=qkd[b][lane*4+i]; k[i]=qkd[b][128+lane*4+i]; decay[i]=qkd[b][256+lane*4+i]; }
            beta=beta_v[b];
            unsigned opk[VPR/2];            // lane0-packed o for the direct ship
            #pragma unroll
            for(int vi=0;vi<VPR;++vi) {
                int v=v0+w*VPR+vi;
                float zz[4];
                #pragma unroll
                for(int i=0;i<4;++i) zz[i]=mul(add(frag[vi][i],0.f),decay[i]);
                float delta=mul(__fsub_rn(f(qkv[j][256+v]),butterfly(dot4(k,zz))),beta);
                #pragma unroll
                for(int i=0;i<4;++i) {
                    zz[i]=__fmaf_rn(k[i],delta,zz[i]);
                    frag[vi][i]=zz[i];
                }
                *reinterpret_cast<float4*>(stash+(size_t)j*8*Z2S_STASH_H+v*128+lane*4)=
                    make_float4(zz[0],zz[1],zz[2],zz[3]);
                float o=butterfly(dot4(q,zz));
                if(lane==0) {
                    bf16 ob=__float2bfloat16_rn(o);
                    raw[j][v]=ob;
                    unsigned pk=(unsigned)*reinterpret_cast<unsigned short*>(&ob);
                    if(vi&1) opk[vi/2]|=pk<<16; else opk[vi/2]=pk;
                }
            }
            PROFR(4+j*2)
            // Packed direct ship: lane0 of each rec warp stores the SAME bf16
            // o pattern it wrote to smem as ONE wide global store; bar 7 then
            // orders all ships before t256's release-ticket RMW. (Replaces the
            // 64-thread smem->global copy + 64-fence storm: 0.8-1.3us/row.)
            if(lane==0) {
                bf16 *dst=&raw_x[((size_t)(s*8+h)*3+j)*128+v0+w*VPR];
                if constexpr(VPR==8)
                    *reinterpret_cast<uint4*>(dst)=make_uint4(opk[0],opk[1],opk[2],opk[3]);
                else
                    *reinterpret_cast<uint2*>(dst)=make_uint2(opk[0],opk[1]);
            }
            asm volatile("bar.sync 7, %0;" :: "n"(RCNT) : "memory");
            if(t==256) tgt_s[j]=v4_arrive(&raw_tk[(s*8+h)*3+j],ZW);
            bar_arrive<NT>(B_RAW0+j);
            // row-0 qkd consumed by the whole crew (reads precede the ship
            // bar 7 above): release qkd[0] for the prep crew's row-2 reuse.
            if(j==0 && t==256) atomicExch(&qkd0_free,1u);
        }
        PROFR(10)
        done();
    }
}
} // namespace
#define PARGS const bf16 *F,const bf16 *wf,const bf16 *wg,const float *cw, \
 const float *al,const float *dt,const bf16 *nw,const bf16 *cs,const float *ss, \
 const int *cl,const int *sl,const int *cu,bf16 *out,float *stash_ssm, \
 bf16 *stash_conv,int S
#define PCALL F,wf,wg,cw,al,dt,nw,cs,ss,cl,sl,cu,out,stash_ssm,stash_conv,S
#define ZSARGS PARGS,unsigned *xchg
#define ZSCALL PCALL,xchg
extern "C" {
__global__ void glm53_kda_verify_fused512p(PARGS) { pbody<512>(PCALL); }
__global__ void glm53_kda_verify_fused512p_debug(PARGS,bf16 *r,bf16 *c,float *rs) {
    pbody<512>(PCALL,r,c,rs); }
__global__ void glm53_kda_verify_fused512s2(ZSARGS) {
    zsbody<512,2>(ZSCALL); }
__global__ void glm53_kda_verify_fused512s2_debug(
    ZSARGS,bf16 *r,bf16 *c,float *rs) {
    zsbody<512,2>(ZSCALL,r,c,rs); }
__global__ void glm53_kda_verify_fused512s2_prof(
    ZSARGS,unsigned long long *prof) {
    zsbody<512,2>(ZSCALL,nullptr,nullptr,nullptr,prof); }
__global__ void glm53_kda_verify_fused512s4(ZSARGS) {
    zsbody<512,4>(ZSCALL); }
__global__ void glm53_kda_verify_fused512s4_debug(
    ZSARGS,bf16 *r,bf16 *c,float *rs) {
    zsbody<512,4>(ZSCALL,r,c,rs); }
__global__ void glm53_kda_verify_fused512s4_prof(
    ZSARGS,unsigned long long *prof) {
    zsbody<512,4>(ZSCALL,nullptr,nullptr,nullptr,prof); }
}
