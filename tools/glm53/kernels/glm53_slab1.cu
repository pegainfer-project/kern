// glm53_slab1.cu -- slab-1 (KDA-layer attention half) for the GLM-5.3-Flash MTP
// verify round: [qkvbfg GEMM -> KDA v3 verify+stash core -> o_proj GEMM] as ONE
// launch with in-kernel phase barriers, plus glm53_kda_select_all (34-layer
// advance select gather, 34 calls -> 1).
//
// Numerics contract:
//  - Phase B (KDA core): FROZEN VERBATIM copy of glm53_kda_v3.cu device code
//    (helpers, fg3, vbody) with vbody's (s,h) blockIdx reads parameterized.
//    Bitwise-gated vs spec_kda_fused_v3 (production v3 cubin) on identical F.
//  - Phases A/C (handwritten weight-streaming GEMMs, bf16 mma m16n8k16, fp32
//    accum, interleaved-but-fixed k order, single-writer deterministic):
//    KL+sentinel gate vs cublasLt bf16 TN. NOT bitwise. Deterministic per run.
//  - select_all: pure copies with the same guards as glm53_kda_select. Bitwise.
//
// Phase barriers: cross-CTA, NO cooperative launch. Correctness relies on full
// residency: grid is fixed 132 CTAs, 195KB dynamic smem + ~6.5KB static =>
// 1 CTA/SM on 132 SMs, all CTAs co-resident. Monotonic u64 ticket barriers
// (zero-initialized op scratch, self-incrementing across launches, no reset),
// 500ms watchdog per barrier -> sticky err (mirrors spec_ar timeout discipline;
// a fired err is fatal to the round, runtime polls the buffer).
#include <cuda_bf16.h>
#include <stdint.h>

using bf16 = __nv_bfloat16;

namespace {

// ---------------------------------------------------------------------------
// Part 0: helpers -- VERBATIM from glm53_kda_v3.cu (bitwise-critical).
// ---------------------------------------------------------------------------
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
    float r;
    asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
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

// ---------------------------------------------------------------------------
// Part 1: cross-CTA phase barrier (monotonic u64 ticket, watchdog, sticky err)
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned long long gtimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
// ctr is zero-initialized op scratch, monotonically increasing across launches
// (never reset; every CTA of every launch arrives exactly once per barrier).
__device__ void gbar(unsigned long long *ctr, unsigned *err) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        unsigned long long a = atomicAdd(ctr, 1ull);
        unsigned long long target = (a / gridDim.x + 1ull) * gridDim.x;
        if (a + 1ull != target) {
            unsigned long long t0 = gtimer();
            while (atomicAdd(ctr, 0ull) < target) {
                if (gtimer() - t0 > 500000000ull) { atomicExch(err, 1u); break; }
            }
        }
        __threadfence();
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------
// Part 2: weight-streaming skinny GEMM.  out[t,n] = sum_k x[t,k]*w[n,k]
// x [T,K] bf16 row-major, w [N,K] bf16 row-major, out [T,N] bf16 (fp32 accum).
// One CTA per 32-wide n tile; warps 0-3 x 8 n-channels; K in 256-wide chunks,
// cp.async 4-stage pipelined. Per-SM HBM throughput is hard-capped ~24.5 GB/s
// (bwtest: independent of piece size 512B-2KB, cp.async vs TMA bulk, pipeline
// depth); aggregate scales linearly with participating CTAs to ~3.25 TB/s at
// 132 CTAs. So the GEMM phases tile N to engage as many slab CTAs as possible
// (A: 105 tiles, C: 128). K order: chunk-major, k16-step-minor, all warps
// walk the same order; every output element has exactly one writer warp, so
// the result is deterministic across runs (but NOT cublas-bitwise).
// ---------------------------------------------------------------------------
#define GEMM_NTILE 32
// Per-phase tile shape: gemmA (K=4096) walks K in 512-wide chunks, 3-stage
// cp.async pipeline (3*(32+32)*(512+8)*2B = 195KB smem); gemmC (K=1024)
// loads the whole K extent as ONE chunk, no pipeline (2*32*(1024+8)*2B =
// 128.8KB). Row stride is KC+8 bf16: the +16B pad rotates ldmatrix row
// addresses across smem banks (an unpadded stride put every row on the same
// bank: 8-16-way conflicts, ~20x slowdown).

__device__ __forceinline__ void cp16(void *smem, const void *gmem, int bytes) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 :: "r"(s), "l"(gmem), "r"(bytes));
}
__device__ __forceinline__ void cp_commit() {
    asm volatile("cp.async.commit_group;" ::: "memory");
}
template<int N> __device__ __forceinline__ void cp_wait() {
    if constexpr(N == 0) asm volatile("cp.async.wait_all;" ::: "memory");
    else asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory");
}

// Warp-specialized skinny GEMM: producer warps 4-15 (384 threads) stream
// w/x chunks through STAGES smem buffers; consumer warps 0-3 run mma.
// full_bar[s]: 384 arrivals via cp.async.mbarrier.arrive (trips when every
// producer thread's async copies into stage s complete). empty_bar[s]: 4
// arrivals (one per consumer warp after its last read of stage s). This
// removes the bulk-synchronous copy/compute serialization (~25-30% of phase
// time at small T). mbarriers live at the tail of dynamic smem.
// Accumulation order per output element is unchanged (k ascending), so the
// result is bitwise identical to the bulk-synchronous version.
#define GEMM_PRODUCERS 384
__device__ __forceinline__ void mbar_init(void *mbar, unsigned count) {
    unsigned s = (unsigned)__cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(s), "r"(count));
}
__device__ __forceinline__ void mbar_arrive(void *mbar) {
    unsigned s = (unsigned)__cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(s));
}
// .noinc is mandatory: the default form increments the expected count at
// issue and arrives on completion (net zero) -> the barrier never trips.
__device__ __forceinline__ void mbar_cp_arrive(void *mbar) {
    unsigned s = (unsigned)__cvta_generic_to_shared(mbar);
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" :: "r"(s));
}
__device__ __forceinline__ void mbar_wait(void *mbar, unsigned parity) {
    unsigned s = (unsigned)__cvta_generic_to_shared(mbar);
    unsigned done = 0;
    // bounded spin: a hung in-kernel wait must NOT wedge the GPU (a deadlocked
    // kernel pins the context even after the host process is SIGKILLed;
    // a GPU was lost that way once in bring-up). ~2^30 spins ~ 0.5-1s, then the
    // wait is treated as complete and downstream checks catch the corruption.
    for (long guard = 0; !done && guard < (1l << 30); ++guard) {
        asm volatile("{.reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;"
                     " selp.u32 %0, 1, 0, p;}"
                     : "=r"(done) : "r"(s), "r"(parity));
    }
}

template<int N, int K, int KC, int STAGES>
__device__ void gemm_phase(const bf16 *x, const bf16 *w, bf16 *out, int T, int bid,
                           char *smem) {
    constexpr int TILES = (N + GEMM_NTILE - 1) / GEMM_NTILE;
    constexpr int CHUNKS = K / KC;
    constexpr int LD = KC + 8;
    static_assert(K % KC == 0, "K must be a multiple of KC");
    static_assert(STAGES >= 2 && STAGES <= 4, "2..4 stages");
    bf16 *ws[STAGES], *xs[STAGES];
    #pragma unroll
    for (int b = 0; b < STAGES; ++b) {
        ws[b] = reinterpret_cast<bf16*>(smem) + b * GEMM_NTILE * LD;
        xs[b] = reinterpret_cast<bf16*>(smem) +
                (STAGES * GEMM_NTILE + b * 32) * LD;
    }
    unsigned long long *full_bar = reinterpret_cast<unsigned long long*>(
        smem + STAGES * (GEMM_NTILE + 32) * LD * 2);
    unsigned long long *empty_bar = full_bar + STAGES;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const int n0 = bid * GEMM_NTILE;
    if (t == 0) {
        #pragma unroll
        for (int s = 0; s < STAGES; ++s) {
            mbar_init(&full_bar[s], GEMM_PRODUCERS);
            mbar_init(&empty_bar[s], 4);
        }
    }
    __syncthreads();

    if (warp >= 4) {
        // ---- producer: stream all chunks ----------------------------------
        const int pt = t - 128;
        constexpr int WOPS = GEMM_NTILE * (KC / 8), XOPS = 32 * (KC / 8);
        for (int j = 0; j < CHUNKS; ++j) {
            const int s = j % STAGES;
            if (j >= STAGES) mbar_wait(&empty_bar[s], ((j / STAGES) - 1) & 1);
            bf16 *wd = ws[s], *xd = xs[s];
            #pragma unroll
            for (int o = pt; o < WOPS; o += GEMM_PRODUCERS) {
                int row = o / (KC / 8), col8 = (o % (KC / 8)) * 8;
                const bf16 *src = w + (size_t)(n0 + row) * K + j * KC + col8;
                cp16(wd + row * LD + col8, src, (n0 + row) < N ? 16 : 0);
            }
            #pragma unroll
            for (int o = pt; o < XOPS; o += GEMM_PRODUCERS) {
                int row = o / (KC / 8), col8 = (o % (KC / 8)) * 8;
                const bf16 *src = x + (size_t)row * K + j * KC + col8;
                cp16(xd + row * LD + col8, src, row < T ? 16 : 0);
            }
            mbar_cp_arrive(&full_bar[s]);
        }
        return;  // producers are done; consumers finish the epilogue
    }

    // ---- consumer: mma per chunk as it lands ------------------------------
    float acc[2][4] = {};
    for (int j = 0; j < CHUNKS; ++j) {
        const int s = j % STAGES;
        mbar_wait(&full_bar[s], (j / STAGES) & 1);
        const bf16 *wd = ws[s], *xd = xs[s];
        #pragma unroll
        for (int k16 = 0; k16 < KC / 16; ++k16) {
            // B frag: [8 n][16 k] from w tile, plain ldmatrix.x2.
            unsigned b0, b1;
            {
                const bf16 *addr = wd + (warp * 8 + (lane & 7)) * LD +
                                   k16 * 16 + ((lane >> 3) & 1) * 8;
                unsigned sa = (unsigned)__cvta_generic_to_shared(addr);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                             : "=r"(b0), "=r"(b1) : "r"(sa));
            }
            #pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                // A frag: [16 t][16 k] from x tile, plain ldmatrix.x4.
                unsigned a0, a1, a2, a3;
                const bf16 *addr = xd + (mt * 16 + (lane & 15)) * LD +
                                   k16 * 16 + (lane >> 4) * 8;
                unsigned sa = (unsigned)__cvta_generic_to_shared(addr);
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                             "{%0,%1,%2,%3}, [%4];"
                             : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(sa));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                             : "+f"(acc[mt][0]), "+f"(acc[mt][1]),
                               "+f"(acc[mt][2]), "+f"(acc[mt][3])
                             : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(&empty_bar[s]);
    }
    // Epilogue: C frag m16n8 -> out[t, n0 + warp*8 + col], predicated t < T.
    // c0=(r, c), c1=(r, c+1), c2=(r+8, c), c3=(r+8, c+1) with r=lane/4, c=2*(lane%4).
    const int r = lane >> 2, c = (lane & 3) * 2, ncol = n0 + warp * 8 + c;
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
        int t0 = mt * 16 + r, t1 = t0 + 8;
        if (ncol < N) {
            if (t0 < T) out[(size_t)t0 * N + ncol] = __float2bfloat16_rn(acc[mt][0]);
            if (t1 < T) out[(size_t)t1 * N + ncol] = __float2bfloat16_rn(acc[mt][2]);
        }
        if (ncol + 1 < N) {
            if (t0 < T) out[(size_t)t0 * N + ncol + 1] = __float2bfloat16_rn(acc[mt][1]);
            if (t1 < T) out[(size_t)t1 * N + ncol + 1] = __float2bfloat16_rn(acc[mt][3]);
        }
    }
    // no trailing __syncthreads: producers returned early; the phase-end
    // gbar's own __syncthreads reconverges the block.
}

// ---------------------------------------------------------------------------
// Part 3: KDA v3 core -- VERBATIM from glm53_kda_v3.cu (fg3 + vbody), with
// vbody's (s,h) blockIdx reads parameterized (vbody_slab). Instruction-level
// copy; do not touch arithmetic.
// ---------------------------------------------------------------------------
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

template<int NT,bool FUSED>
__device__ void vbody_slab(const bf16 *F,const bf16 *wf,const bf16 *wg,
 const float *cw,const float *al,const float *dt,const bf16 *nw,
 const bf16 *cs,const float *ss,const int *cl,const int *sl,const int *cu,
 bf16 *out,float *stash_ssm,bf16 *stash_conv,int S,int s,int h,
 bf16 *dbg_raw=nullptr,bf16 *dbg_cv=nullptr,float *dbg_rs=nullptr) {
    int t=threadIdx.x,lane=t%32,warp=t/32;
    if(s>=S) { return; }
    int cidx=cl[s],sidx=sl[s],row=cu[s];
    if(cidx<0 || sidx<0 || row!=3*s || cu[s+1]-row!=3) { return; }
    __shared__ bf16 gates[3][256],qkv[3][384],raw[3][128];
    __shared__ float qkd[384],beta_shared,rstd_v[3];
    if constexpr(FUSED) fg3<NT>(F,wf,wg,s,h,&gates[0][0]);
    else {
        for(int i=t;i<128;i+=NT) {
            #pragma unroll
            for(int j=0;j<3;++j) {
                gates[j][i]=wf[(size_t)(3*s+j)*1024+h*128+i];
                gates[j][128+i]=wg[(size_t)(3*s+j)*1024+h*128+i];
            }
        }
    }
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
    constexpr int VPR=128/(NT/32);
    float frag[VPR][4];
    #pragma unroll
    for(int vi=0;vi<VPR;++vi) {
        int v=warp*VPR+vi;
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
            int v=warp*VPR+vi;
            float z[4];
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
            out[(size_t)(3*s+j)*1024+h*128+i]=__float2bfloat16_rn(y);
        }
        __syncthreads();
    }
    if(dbg_raw) for(int j=0;j<3;++j)
        for(int i=t;i<128;i+=NT) dbg_raw[(size_t)(3*s+j)*1024+h*128+i]=raw[j][i];
    if(dbg_cv) for(int j=0;j<3;++j)
        for(int i=t;i<384;i+=NT) dbg_cv[(size_t)(3*s+j)*3072+(i/128)*1024+h*128+i%128]=qkv[j][i];
    if(dbg_rs && t==0) { dbg_rs[s*24+h*3+0]=rstd_v[0]; dbg_rs[s*24+h*3+1]=rstd_v[1]; dbg_rs[s*24+h*3+2]=rstd_v[2]; }
}

} // namespace

// ---------------------------------------------------------------------------
// Part 4: slab kernel. Phases: A qkvbfg GEMM (105 CTAs) | barrier | B KDA core
// (seqs x 8 CTAs) | barrier | C o_proj GEMM (128 CTAs, single K chunk). Fixed
// grid 132 CTAs, block 512, dynamic smem 195KB. Idle CTAs prefetch the NEXT
// phase/layer weights into L2 (fire-and-forget prefetch.global.L2; per-SM DRAM
// streams cap ~24.5 GB/s but L2 reads run 2-3x that, so streaming wq_next
// under phase B roughly halves the next layer's phase-A time):
//   phase A: CTAs 105.. prefetch wo (8.4MB) for phase C
//   phase B: CTAs max(seqs*8,105).. prefetch wq_next (27.3MB) for layer L+1
// <PHASES> bitmask: 1=A, 2=B, 4=C (debug isolation entries; production = 7).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void prefetch_l2(const void *p) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}
// Sweep [base, base+bytes) in 128B lines, strided across CTAs [lo, hi).
// Prefetch is an architectural nop: no effect on numerics or gates.
__device__ void prefetch_range(const char *base, long bytes, int lo, int hi) {
    const int bid = blockIdx.x;
    if (bid < lo || bid >= hi || !base) return;
    const int ncta = hi - lo, me = bid - lo;
    const long lines = bytes >> 7;
    for (long i = (long)me * blockDim.x + threadIdx.x; i < lines;
         i += (long)ncta * blockDim.x)
        prefetch_l2(base + (i << 7));
}

extern "C" {

// production entry: all phases
__global__ void __launch_bounds__(512, 1)
glm53_kda_slab1_full(const bf16 *x,const bf16 *wq,bf16 *F,
 const bf16 *wf,const bf16 *wg,const float *cw,const float *al,const float *dt,
 const bf16 *nw,const bf16 *cs,const float *ss,const int *cl,const int *sl,
 const int *cu,bf16 *kda_o,float *stash_ssm,bf16 *stash_conv,
 const bf16 *wo,bf16 *sub_out,unsigned *sync,int T,int S,
 const bf16 *wq_next,int do_prefetch) {
    extern __shared__ __align__(16) char smem[];
    unsigned long long *bar = reinterpret_cast<unsigned long long*>(sync);
    unsigned *err = sync + 4;
    wait();
    const int bid = blockIdx.x;
    if (bid < 105) gemm_phase<3336, 4096, 512, 3>(x, wq, F, T, bid, smem);
    else if (do_prefetch)
        prefetch_range(reinterpret_cast<const char*>(wo), 4096l * 1024 * 2, 105, 132);
    gbar(bar + 0, err);
    if (*(volatile unsigned*)err) { done(); return; }
    const int s = bid >> 3, h = bid & 7;
    if (s < S) vbody_slab<512, true>(F, wf, wg, cw, al, dt, nw, cs, ss, cl, sl,
                                     cu, kda_o, stash_ssm, stash_conv, S, s, h);
    else if (do_prefetch)
        prefetch_range(reinterpret_cast<const char*>(wq_next), 3336l * 4096 * 2,
                       S * 8 > 105 ? S * 8 : 105, 132);
    gbar(bar + 1, err);
    if (*(volatile unsigned*)err) { done(); return; }
    if (bid < 128) gemm_phase<4096, 1024, 512, 2>(kda_o, wo, sub_out, T, bid, smem);
    done();
}

// debug isolation entries (test harness only; same ABI)
__global__ void __launch_bounds__(512, 1)
glm53_kda_slab1_gemmA(const bf16 *x,const bf16 *wq,bf16 *F,
 const bf16 *wf,const bf16 *wg,const float *cw,const float *al,const float *dt,
 const bf16 *nw,const bf16 *cs,const float *ss,const int *cl,const int *sl,
 const int *cu,bf16 *kda_o,float *stash_ssm,bf16 *stash_conv,
 const bf16 *wo,bf16 *sub_out,unsigned *sync,int T,int S,
 const bf16 *wq_next,int do_prefetch) {
    extern __shared__ __align__(16) char smem[];
    unsigned long long *bar = reinterpret_cast<unsigned long long*>(sync);
    unsigned *err = sync + 4;
    wait();
    const int bid = blockIdx.x;
    if (bid < 105) gemm_phase<3336, 4096, 512, 3>(x, wq, F, T, bid, smem);
    gbar(bar + 0, err);
    done();
}
__global__ void __launch_bounds__(512, 1)
glm53_kda_slab1_core(const bf16 *x,const bf16 *wq,bf16 *F,
 const bf16 *wf,const bf16 *wg,const float *cw,const float *al,const float *dt,
 const bf16 *nw,const bf16 *cs,const float *ss,const int *cl,const int *sl,
 const int *cu,bf16 *kda_o,float *stash_ssm,bf16 *stash_conv,
 const bf16 *wo,bf16 *sub_out,unsigned *sync,int T,int S,
 const bf16 *wq_next,int do_prefetch) {
    extern __shared__ __align__(16) char smem[];
    (void)smem;
    wait();
    const int bid = blockIdx.x;
    const int s = bid >> 3, h = bid & 7;
    if (s < S) vbody_slab<512, true>(F, wf, wg, cw, al, dt, nw, cs, ss, cl, sl,
                                     cu, kda_o, stash_ssm, stash_conv, S, s, h);
    done();
}
__global__ void __launch_bounds__(512, 1)
glm53_kda_slab1_gemmC(const bf16 *x,const bf16 *wq,bf16 *F,
const bf16 *wf,const bf16 *wg,const float *cw,const float *al,const float *dt,
const bf16 *nw,const bf16 *cs,const float *ss,const int *cl,const int *sl,
const int *cu,bf16 *kda_o,float *stash_ssm,bf16 *stash_conv,
const bf16 *wo,bf16 *sub_out,unsigned *sync,int T,int S,
const bf16 *wq_next,int do_prefetch) {
    extern __shared__ __align__(16) char smem[];
    unsigned long long *bar = reinterpret_cast<unsigned long long*>(sync);
    unsigned *err = sync + 4;
    wait();
    const int bid = blockIdx.x;
    if (bid < 128) gemm_phase<4096, 1024, 512, 2>(kda_o, wo, sub_out, T, bid, smem);
    gbar(bar + 0, err);
    done();
}
// debug twin of _core with vbody's raw/cv/rs inspection buffers appended
__global__ void __launch_bounds__(512, 1)
glm53_kda_slab1_core_debug(const bf16 *F,
 const bf16 *wf,const bf16 *wg,const float *cw,const float *al,const float *dt,
 const bf16 *nw,const bf16 *cs,const float *ss,const int *cl,const int *sl,
 const int *cu,bf16 *kda_o,float *stash_ssm,bf16 *stash_conv,int S,
 bf16 *dbg_raw,bf16 *dbg_cv,float *dbg_rs) {
    wait();
    const int bid = blockIdx.x;
    const int s = bid >> 3, h = bid & 7;
    if (s < S) vbody_slab<512, true>(F, wf, wg, cw, al, dt, nw, cs, ss, cl, sl,
                                     cu, kda_o, stash_ssm, stash_conv, S, s, h,
                                     dbg_raw, dbg_cv, dbg_rs);
    done();
}

// Advance select, all 34 layers in one launch (34 calls -> 1). Copy semantics
// and guards VERBATIM from glm53_kda_select; layer index from blockIdx.x / S.
// stash strides fixed by the v3 stash geometry (stash_seqs=8): ssm layer =
// 8*3*8*16384 f32, conv layer = 8*3*9216 bf16. lstride = lines-table i32
// stride per layer (16). grid [34*S, 8, 1], block 256.
__global__ void glm53_kda_select_all(const float *stash_ssm,const bf16 *stash_conv,
 float *ss,bf16 *cs,const int *sl,const int *cl,const int *nacc,const int *valid,
 int S,int lstride,int rows) {
    wait();
    int x=blockIdx.x,h=blockIdx.y,t=threadIdx.x;
    int k=x/S,s=x-k*S;
    int n=nacc[s];
    if(rows<1 || rows>8 || !valid[rows*s] || n<1 || n>rows) { done(); return; }
    int si=sl[k*lstride+s],ci=cl[k*lstride+s];
    if(si<=0 || ci<=0) { done(); return; }
    // Layer strides: 8 seqs * rows * [8,128,128] f32 / 8 seqs * rows * [3,3072] bf16.
    const float4 *src=reinterpret_cast<const float4*>(stash_ssm+(size_t)k*rows*1048576+((size_t)(s*rows+n-1)*8+h)*16384);
    float4 *dst=reinterpret_cast<float4*>(ss+(size_t)si*131072+(size_t)h*16384);
    for(int i=t;i<16384/4;i+=blockDim.x) dst[i]=src[i];
    const bf16 *csrc=stash_conv+(size_t)k*rows*73728+(size_t)(s*rows+n-1)*9216;
    bf16 *cdst=cs+(size_t)ci*9216;
    for(int i=t;i<3*384;i+=blockDim.x) {
        int tap=i/384,c=h*384+i%384;
        cdst[tap*3072+c]=csrc[tap*3072+c];
    }
    done();
}

} // extern "C"
