// Decode only: sm_90a, groups=1..16, one row/group. See dsa_fusion.md.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
using bf16 = __nv_bfloat16;
__device__ __forceinline__ void wait_dep() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void release_dep() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ float warp_sum(float v) {
    for (int d=16; d; d>>=1) v += __shfl_down_sync(~0u,v,d);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int d=16; d; d>>=1) v=fmaxf(v,__shfl_xor_sync(~0u,v,d));
    return v;
}
__device__ __forceinline__ float tl_exp(float x) {
    float y; asm("mul.f32 %0,%1,0f3FB8AA3B;":"=f"(y):"f"(x));
    asm("ex2.approx.f32 %0,%1;":"=f"(y):"f"(y)); return y;
}
__device__ __forceinline__ float tl_div(float a,float b) {
    float y; asm("div.full.f32 %0,%1,%2;":"=f"(y):"f"(a),"f"(b)); return y;
}
// 16 output columns x 4 rows/CTA. Weights reused across rows; no split-K,
// atomics, padded-row reads, or intermediate BF16 rounding.
template<int N,int K> __device__ void absorb(const bf16* a,const bf16* w,bf16* out,int rows) {
    wait_dep();
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
    release_dep();
}
extern "C" __global__ __launch_bounds__(128) void glm53_dsa_w_kc_v2(const bf16* a,const bf16* w,bf16* c,int rows) { absorb<512,256>(a,w,c,rows); }
extern "C" __global__ __launch_bounds__(128) void glm53_dsa_w_vc_v2(const bf16* a,const bf16* w,bf16* c,int rows) { absorb<256,512>(a,w,c,rows); }

// EXACT baseline RMS reduction tree (1024 threads), BF16 norm boundary,
// non-ue8m0 quant, SFA[group*16+row]. KV base includes layer offset.
extern "C" __global__ __launch_bounds__(1024) void glm53_dsa_norm_store_v2(
    const bf16* qkv,const bf16* qw,const bf16* kw,bf16* qa,unsigned char* qa8,
    float* sfa,char* kv,const int* slots) {
    wait_dep(); int t=threadIdx.x, row=blockIdx.x;
    const bf16* x=qkv+(size_t)row*2048;
    __shared__ float redq[32],redk[32]; __shared__ bf16 q[1536];
    float aq=0,ak=0;
    for(int j=t;j<1536;j+=1024) {float v=__bfloat162float(x[j]); aq=fmaf(v,v,aq);}
    if(t<512) {float v=__bfloat162float(x[1536+t]); ak=v*v;}
    aq=warp_sum(aq); ak=warp_sum(ak);
    if(!(t&31)) {redq[t>>5]=aq;redk[t>>5]=ak;} __syncthreads();
    if(t<32) {
        float vq=warp_sum(redq[t]),vk=warp_sum(redk[t]);
        if(t==0) {redq[0]=rsqrtf(__fadd_rn(__fdiv_rn(vq,1536.f),1e-5f));redk[0]=rsqrtf(__fadd_rn(__fdiv_rn(vk,512.f),1e-5f));}
    } __syncthreads();
    for(int j=t;j<1536;j+=1024) {
        bf16 v=__float2bfloat16_rn(__bfloat162float(x[j])*redq[0]*__bfloat162float(qw[j]));
        qa[(size_t)row*1536+j]=v; q[j]=v;
    }
    if(t<512) reinterpret_cast<bf16*>(kv+(long long)slots[row]*11264)[t]=
        __float2bfloat16_rn(__bfloat162float(x[1536+t])*redk[0]*__bfloat162float(kw[t]));
    __syncthreads();
    if(t<384) {
        int g=t>>5,l=t&31; float v[4],am=1e-10f;
        #pragma unroll
        for(int j=0;j<4;++j) {v[j]=__bfloat162float(q[g*128+j*32+l]);am=fmaxf(am,fabsf(v[j]));}
        am=warp_max(am); float scale,qs;
        // module_233 was built with fast-math: MUFU.RCP then FMUL.FTZ.
        // RN division changes FP8 midpoint ties. Mirror only this quant math,
        // not RMS or the kpool softmax, which use different lowering.
        asm("rcp.approx.ftz.f32 %0,%1;":"=f"(qs):"f"(am));
        asm("mul.ftz.f32 %0,%1,0f43E00000;":"=f"(qs):"f"(qs));
        asm("mul.ftz.f32 %0,%1,0f3B124925;":"=f"(scale):"f"(am));
        if(l==0) sfa[g*16+row]=scale;
        #pragma unroll
        for(int j=0;j<4;++j) {
            float z; asm("mul.ftz.f32 %0,%1,%2;":"=f"(z):"f"(v[j]),"f"(qs));
            qa8[(size_t)row*1536+g*128+j*32+l]=
                __nv_cvt_float_to_fp8(fminf(z,448.f),__NV_SATFINITE,__NV_E4M3);
        }
    }
    release_dep();
}

// Grid.x is S*32 flattened head rows. The two-launch reference must pass
// M=S*32 to _act_quant_kernel (its CTA handles 32 heads), not a constant 32.
extern "C" __global__ __launch_bounds__(128) void glm53_dsa_had_quant_v2(
    const bf16* x,unsigned char* out,float* scales) {
    wait_dep(); int h=blockIdx.x,i=threadIdx.x; __shared__ float s[128],red[4];
    s[i]=__bfloat162float(x[(size_t)h*128+i]); __syncthreads();
    #pragma unroll
    for(int d=1;d<128;d<<=1) {
        if(i<64) {int a=(i/d)*2*d+i%d,b=a+d;float u=s[a],v=s[b];s[a]=u+v;s[b]=u-v;}
        __syncthreads();
    }
    // Preserve the materialized BF16 hadamard output before absmax/quant.
    float v=__bfloat162float(__float2bfloat16_rn(s[i]*0.08838834764831844f));
    float am=warp_max(fabsf(v)); if(!(i&31)) red[i>>5]=am; __syncthreads();
    am=fmaxf(fmaxf(red[0],red[1]),fmaxf(red[2],red[3]));
    float scale=exp2f(ceilf(log2f(fmaxf(am,1e-4f)*(1.f/448.f))));
    if(i==0) scales[h]=scale;
    out[(size_t)h*128+i]=__nv_cvt_float_to_fp8(fminf(fmaxf(tl_div(v,scale),-448.f),448.f),__NV_SATFINITE,__NV_E4M3);
    release_dep();
}

// Logical concat N=288, separate weight pointers keep existing checkpoint ABI.
// wk/gate: BF16 output; head projection: FP32 output, never BF16-rounded.
extern "C" __global__ __launch_bounds__(128) void glm53_dsa_xproj_v2(
    const bf16* x,const bf16* wk,const bf16* gate,const float* hw,
    bf16* kg,float* head,int rows) {
    wait_dep(); int lane=threadIdx.x&31,n=blockIdx.x*4+(threadIdx.x>>5),r0=blockIdx.y*4;
    float sums[4]={};
    for(int k=lane;k<4096;k+=32) {
        float w=n<128?__bfloat162float(wk[(size_t)n*4096+k]):
            n<256?__bfloat162float(gate[(size_t)(n-128)*4096+k]):hw[(size_t)(n-256)*4096+k];
        #pragma unroll
        for(int r=0;r<4;++r) if(r0+r<rows) sums[r]=fmaf(__bfloat162float(x[(size_t)(r0+r)*4096+k]),w,sums[r]);
    }
    #pragma unroll
    for(int r=0;r<4;++r) {
        float s=warp_sum(sums[r]);
        if(lane==0 && r0+r<rows) {
            if(n<256) kg[(size_t)(r0+r)*256+n]=__float2bfloat16_rn(s);
            else head[(size_t)(r0+r)*32+n-256]=s;
        }
    }
    release_dep();
}

// Fused dsa_prep and DeepGEMM <32,256,132,false> schedule. One warp;
// same reversed segment allocation and final {batch,0} sentinel as source.
extern "C" __global__ __launch_bounds__(32) void glm53_dsa_step_v2(
    const int* seq,int* pools,int* ctx,int* lens,int* sched,int rows) {
    wait_dep();int t=threadIdx.x; __shared__ int prefix[32];
    int p=t<rows?seq[t]/4:0,c=t<rows?max(p,1):0;
    if(t<rows) {pools[t]=p;ctx[t]=c;lens[t]=min(p*4,2048)+(seq[t]&3);}
    int v=(c+255)/256;
    #pragma unroll
    for(int d=1;d<32;d<<=1) {int a=__shfl_up_sync(~0u,v,d);if(t>=d)v+=a;}
    prefix[t]=v; __syncwarp(); int total=__shfl_sync(~0u,v,31),q=total/132,pivot=132-total%132;
    for(int sm=t;sm<132;sm+=32) {
        int start=sm*q+max(sm-pivot,0),lo=0,hi=rows;
        while(lo<hi) {int mid=(lo+hi)/2;if(prefix[mid]<=start)lo=mid+1;else hi=mid;}
        sched[sm*2]=lo;sched[sm*2+1]=start-(lo?prefix[lo-1]:0);
    }
    if(t==0) {sched[264]=rows;sched[265]=0;} release_dep();
}

// kpool arithmetic/state addressing transcribed unchanged from glm53_dsa.cu.
extern "C" __global__ __launch_bounds__(128) void glm53_dsa_xepilogue_v2(
    char* idx_base,char* tail_base,const bf16* kg,const float* head,
    const float* nw,const float* nb,const float* ape,const int* bt,
    const int* lines,const int* positions,const int* seq_lens,const int* valid,
    const float* qs,float* out,int bt_cols) {
    wait_dep();
    unsigned int row = blockIdx.x;
    unsigned int i = threadIdx.x;              // one element per thread, dim = 128
    int pos = positions[row];
    int seq = seq_lens[row];
    bool pos_valid = (valid[row] != 0) && (pos >= 0) && (pos < seq);


    float v=__bfloat162float(kg[(size_t)row*256+i]);
    float v2=v*v,s1=warp_sum(v),s2=warp_sum(v2);
    __shared__ float norm[8];
    if(!(i&31)) {norm[i>>5]=s1;norm[4+(i>>5)]=s2;}
    __syncthreads();
    if(i<4) {
        float a=norm[i],c=norm[4+i];
        for(int d=2;d;d>>=1) {a+=__shfl_down_sync(0xfu,a,d);c+=__shfl_down_sync(0xfu,c,d);}
        if(i==0) {float mean=a*(1.f/128.f);norm[0]=mean;norm[1]=rsqrtf(c*(1.f/128.f)-mean*mean+1e-6f);}
    }
    __syncthreads();
    float k_cur=__bfloat162float(__float2bfloat16_rn((v-norm[0])*norm[1]*nw[i]+nb[i]));
    float s_cur=__bfloat162float(kg[(size_t)row*256+128+i]);
    if(i<32) {float h=head[(size_t)row*32+i]*0.17677669529663687f;
        h=h*qs[(size_t)row*32+i];out[(size_t)row*32+i]=h*0.08838834764831844f;}


    char* line = tail_base + (size_t)lines[row] * 4096;
    bf16* tail_k = reinterpret_cast<bf16*>(line);
    bf16* tail_score = reinterpret_cast<bf16*>(line + 2048);
    int phys = pos & 7;

    int slot = pos & 3;  // pool size is 4; the physical tail ring has 8 slots
    if (pos_valid && slot == 3) {
        // softmax over the 4 pool slots of (score + ape), per element
        float sc[4], kv[4];
        float mx = -__int_as_float(0x7f7fffff);
        int pool_start = pos - 3;
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            int ph = (pool_start + p) & 7;
            float sbuf = __bfloat162float(tail_score[ph * 128 + i]);
            float kbuf = __bfloat162float(tail_k[ph * 128 + i]);
            sc[p] = (p == 3 ? s_cur : sbuf) + ape[p * 128 + i];
            kv[p] = p == 3 ? k_cur : kbuf;
            mx = fmaxf(mx, sc[p]);
        }
        float denom = 0.f, acc = 0.f;
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            float pr = tl_exp(sc[p] - mx);
            denom += pr;
            acc += kv[p] * pr;
        }
        float x = __bfloat162float(__float2bfloat16(tl_div(acc, denom)));
        // hadamard128: 7 butterfly stages (g,s) = (64,1)..(1,64), then x * 128^-0.5
        __shared__ float h[128];
        h[i] = x;
        __syncthreads();
        #pragma unroll
        for (int s = 1; s < 128; s <<= 1) {
            if (i < 64) {
                int gi = i / s, si = i % s;
                int i0 = gi * 2 * s + si, i1 = i0 + s;
                float a = h[i0], b = h[i1];
                h[i0] = a + b;   // race-free within a stage: 64 disjoint pairs
                h[i1] = a - b;
            }
            __syncthreads();
        }
        x = h[i] * 0.08838834764831845f;
        x = __bfloat162float(__float2bfloat16(x));
        // block absmax
        __shared__ float red[4];
        float a = fabsf(x);
        for (int off = 16; off > 0; off >>= 1) a = fmaxf(a, __shfl_down_sync(~0u, a, off));
        if ((i & 31) == 0) red[i >> 5] = a;
        __syncthreads();
        if (i < 4) {
            float v = red[i];
            for (int off = 2; off > 0; off >>= 1) v = fmaxf(v, __shfl_down_sync(0xfu, v, off));
            if (i == 0) red[0] = v;
        }
        __syncthreads();
        float absmax = fmaxf(red[0], 1e-4f);
        float scale = exp2f(ceilf(log2f(absmax * (1.f / 448.f))));
        float q = fminf(fmaxf(x / scale, -448.f), 448.f);
        // pool write (unit 256): pool g lands in page bt[g>>6], slot g%64 --
        // the page holding its own 4 tokens, so kern-serve retire/restore
        // of a shared prefix is safe (arch_decode.md section 1.5a).
        int pool = pos >> 2;
        int page_row = min(max(pool >> 6, 0), bt_cols - 1);
        long long page = bt[(size_t)row * bt_cols + page_row];
        int slot64 = pool & 63;
        constexpr long long IDX_PAGE_BYTES = 11 * 8448;  // all DSA layers in one physical page
        char* dst = idx_base + page * IDX_PAGE_BYTES + slot64 * 128;
        dst[i] = (char)__nv_cvt_float_to_fp8(q, __NV_SATFINITE, __NV_E4M3);
        if (i == 0)
            *reinterpret_cast<float*>(idx_base + page * IDX_PAGE_BYTES + 8192 + slot64 * 4) = scale;
    }
    if (pos_valid) {
        tail_k[phys * 128 + i] = __float2bfloat16(k_cur);
        tail_score[phys * 128 + i] = __float2bfloat16(s_cur);
    }

    release_dep();
}
