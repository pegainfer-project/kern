// Decode/verify TP8 route + topk + compact align + input FP8 quantization.
// No cooperative launch, no spinning, and no host read of the live row count.
// Capacity follows the runtime rows argument: decode rows<=16 (M16 champion)
// and MTP verify rows<=32 (tokens=3S). Clear/fill loops cover exactly the live
// region; never-launched tiles/counters keep stale values and are never read.
// v3 (fused4): universal 32-pair align tiles. At rows<=16 counts<=16, so the
// tiling is identical to the old 16-pair scheme (champion layout preserved);
// at rows 17..32 each expert still gets exactly one tile, so W13/W2 read each
// touched expert shard once (no 2x re-read). Within-expert pair order comes
// from shared-memory atomic tickets: order-free, and every downstream value is
// per-pair independent, so outputs stay bitwise identical. as/hs are stored
// transposed (kb-major) so the W13/W2 main-loop scale loads coalesce.
// v4: phase-1 software pipeline (PF-deep x-row prefetch into register
// buffers with compile-time-constant indices; same loads, same math, same
// per-score accumulation order) + the global clears (sorted/experts/
// w13_counts/w2_counts) move ahead of phase 1 and are distributed over all 96
// CTAs. Every CTA clears before its release ticket RMW, so the last CTA still
// observes cleared memory before its fill phase. Values are unchanged.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda/atomic>
#include <stdint.h>
#include <math.h>

#ifndef ROUTE_PF
#define ROUTE_PF 4
#endif

__device__ __forceinline__ float div_full(float x, float y) {
  float z; asm("div.full.f32 %0, %1, %2;" : "=f"(z) : "f"(x), "f"(y)); return z;
}
__device__ __forceinline__ float sigmoid(float x) {
  float e; x *= -1.4426950408889634f;
  asm("ex2.approx.f32 %0, %1;" : "=f"(e) : "f"(x));
  return div_full(1.f, 1.f + e);
}
__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int d=16; d; d>>=1) x += __shfl_xor_sync(0xffffffff, x, d);
  return x;
}

// counter is op-private, initially zero (kern alloc contract). Last CTA resets
// it on EVERY invocation. Use the same op only on a serial stream/graph.
extern "C" __global__ __launch_bounds__(256)
void glm53_moe_v2_route(const __nv_bfloat16* x, const __nv_bfloat16* w,
    const float* bias, const int* valid, float* weights, int* ids,
    int* sorted, int* experts, int* n_post, __nv_fp8_e4m3* a8, float* as,
    int* w13_counts, float* scores, unsigned* counter, int* w2_counts, int rows) {
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const int t=threadIdx.x, lane=t&31, warp=t>>5, tile=blockIdx.x;
  __shared__ float sums[8][96];
  __shared__ int last;
  __shared__ int hist[512], prefix[512], tmp[512], ordcnt[512];
  constexpr int PF=ROUTE_PF;
  const uint4* xs0=reinterpret_cast<const uint4*>(x+t*8);
  const uint4* xs1=reinterpret_cast<const uint4*>(x+(t+256)*8);
  // row stride: 4096 bf16 = 512 uint4
  uint4 wq[3][2];
#pragma unroll
  for(int n=0;n<3;n++)
#pragma unroll
    for(int u=0;u<2;u++) wq[n][u]=*reinterpret_cast<const uint4*>(w+(tile*3+n)*4096+(t+u*256)*8);
  uint4 xr[PF][2];
#pragma unroll
  for(int d=0;d<PF-1;d++) if(d<rows) { xr[d][0]=xs0[d*512]; xr[d][1]=xs1[d*512]; }
  float wv[3][16];
#pragma unroll
  for(int n=0;n<3;n++) {
#pragma unroll
    for(int u=0;u<2;u++) {
      const __nv_bfloat16* p=reinterpret_cast<const __nv_bfloat16*>(&wq[n][u]);
#pragma unroll
      for(int j=0;j<8;j++) wv[n][u*8+j]=__bfloat162float(p[j]);
    }
  }
  // Distributed global clears: each CTA clears its slice before the release
  // ticket; the last CTA's acquire observes all clears before its fills.
  {
    const int gt=tile*256+t, NT=96*256;
    for(int i=gt;i<rows*32;i+=NT) w2_counts[i]=0;
    for(int i=gt;i<rows*9*32;i+=NT) sorted[i]=rows*9;
    for(int i=gt;i<rows*9;i+=NT) { experts[i]=-1; w13_counts[i]=0; }
  }
  const int iters=(rows+PF-1)/PF;
  for(int it=0;it<iters;it++) {
    const int m=it*PF;
#pragma unroll
    for(int d=0;d<PF;d++) {
      const int pm=m+d+PF-1;
      if(pm<rows) { xr[(d+PF-1)%PF][0]=xs0[pm*512]; xr[(d+PF-1)%PF][1]=xs1[pm*512]; }
      float xv[16];
      {
        const __nv_bfloat16* p0=reinterpret_cast<const __nv_bfloat16*>(&xr[d][0]);
        const __nv_bfloat16* p1=reinterpret_cast<const __nv_bfloat16*>(&xr[d][1]);
#pragma unroll
        for(int j=0;j<8;j++) xv[j]=__bfloat162float(p0[j]);
#pragma unroll
        for(int j=0;j<8;j++) xv[8+j]=__bfloat162float(p1[j]);
      }
#pragma unroll
      for(int n=0;n<3;n++) {
        float acc=0;
#pragma unroll
        for(int j=0;j<16;j++) acc=fmaf(xv[j],wv[n][j],acc);
        acc=warp_sum(acc);
        if(lane==0) sums[warp][(m+d)*3+n]=acc;
      }
    }
  }
  __syncthreads();
  if(t<rows*3) {
    float acc=sums[0][t];
#pragma unroll
    for(int i=1;i<8;i++) acc+=sums[i][t];
    scores[(t/3)*288+tile*3+t%3]=acc;
  }
  // v3: distributed quantization. 128-element groups round-robin over all
  // CTAs, lanes t<8 per group. Same per-element math and same in-group amax
  // reduction sequence (d=4,2,1 within 8 lanes) as the one-CTA-per-token
  // scheme; only the CTA<->group assignment changed. as stays [kb][row].
  {
    // 4 groups per warp pass: lanes split into 4 sub-groups of 8, one
    // 128-element group each. Full-warp shfl (width 8) keeps the exact
    // in-group amax reduction sequence; loop iterations drop 4x.
    const int ng=rows*32;
    for(int gg=tile*4;gg<ng;gg+=96*4) {
      if(warp==0) {
        const int gi=gg+(lane>>3);
        if(gi<ng) {
          const int r=gi>>5, g=gi&31, off=r*4096+g*128+(lane&7)*16;
          float v[16], mx=1e-10f;
#pragma unroll
          for(int j=0;j<16;j++) { v[j]=valid[r]?__bfloat162float(x[off+j]):0.f; mx=fmaxf(mx,fabsf(v[j])); }
#pragma unroll
          for(int d=4;d;d>>=1) mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d,8));
          const float q=__fdividef(448.f,mx);
#pragma unroll
          for(int j=0;j<16;j++) a8[off+j]=__nv_fp8_e4m3(v[j]*q);
          if((lane&7)==0) as[g*32+r]=mx*(1.f/448.f);
        }
      }
    }
  }
  // v3: prefetch topk inputs (bias/valid are kernel inputs, not produced
  // by phase 1) so the selection rounds start with warm registers.
  float pb[9]; int pv=0;
  if(tile<rows && warp==0) {
    pv=valid[tile];
#pragma unroll
    for(int j=0;j<9;j++) pb[j]=bias[j*32+lane];
  }
  // v3: quant redistribution + small ticket-2 + prefetch + 320-scan. Same values, same math, same order per row.
  // Barrier-1: all CTAs bump, then ALL spin until 96 (nobody exits early).
  __threadfence(); __syncthreads();
  if(t==0) {
    cuda::atomic_ref<unsigned,cuda::thread_scope_device> done(*counter);
    done.fetch_add(1,cuda::memory_order_acq_rel);
    while(done.load(cuda::memory_order_acquire)<96) __nanosleep(64);
  }
  __syncthreads();
  // Distributed topk: CTA c computes row c (rows<=96 always), warp 0 only.
  // Instruction sequence identical to the v4 warp loop body for that row.
  if(tile<rows && warp==0) {
    const int row=tile;
    float act[9], cur[9], vals[8]; int chosen[8];
#pragma unroll
    for(int j=0;j<9;j++) {
      int e=j*32+lane;
      act[j]=sigmoid(scores[row*288+e]);
      cur[j]=act[j]+pb[j];
      if(isnan(cur[j])) cur[j]=-1e30f;
    }
#pragma unroll
    for(int k=0;k<8;k++) {
      float best=-INFINITY;
#pragma unroll
      for(int j=0;j<9;j++) best=fmaxf(best,cur[j]);
#pragma unroll
      for(int d=16;d;d>>=1) best=fmaxf(best,__shfl_xor_sync(0xffffffff,best,d));
      int win=289;
#pragma unroll
      for(int j=0;j<9;j++) if(cur[j]==best) win=min(win,j*32+lane);
      win=__reduce_min_sync(0xffffffff,win);
      float val=0.f;
#pragma unroll
      for(int j=0;j<9;j++) if(j*32+lane==win) { val=act[j]; cur[j]=-INFINITY; }
      vals[k]=warp_sum(val); chosen[k]=win;
    }
    // Captured router PTX: 4 sequential values per lane, then xor(2), xor(1).
    float sum=(((vals[0]+vals[1])+vals[2])+vals[3])+(((vals[4]+vals[5])+vals[6])+vals[7]);
    if(lane<9) {
      float val=lane<8?vals[lane]:div_full(sum,2.5f);
      weights[row*9+lane]=pv?div_full(val,sum>0.f?sum:1.f):0.f;
      ids[row*9+lane]=pv?(lane<8?chosen[lane]:288):-1;
    }
  }
  // v3 ticket-2: only the topk CTAs arrive; the last of the `rows`
  // arrivers runs the tail. Non-topk CTAs exit (their stores were consumed
  // before barrier-1). Counter reaches 96+rows, tail resets it to zero.
  if(tile>=rows) return;
  __threadfence(); __syncthreads();
  if(t==0) {
    cuda::atomic_ref<unsigned,cuda::thread_scope_device> done(*counter);
    last=(done.fetch_add(1,cuda::memory_order_acq_rel)==96+rows-1);
  }
  __syncthreads();
  if(!last) return;
  for(int i=t;i<320;i+=256) { hist[i]=0; ordcnt[i]=0; }
  __syncthreads();
  for(int q=t;q<rows*9;q+=256) if(ids[q]>=0) atomicAdd(hist+ids[q],1);
  __syncthreads();
  // Warp-0 scan: 16 sequential per-lane + shfl exclusive scan + carries.
  // Integer-exact; replaces the 9-round Hillis-Steele (same prefix values).
  for(int i=t;i<320;i+=256) prefix[i]=(hist[i]+31)>>5;
  __syncthreads();
  if(warp==0) {
    int loc[10];
#pragma unroll
    for(int u=0;u<10;u++) loc[u]=prefix[lane*10+u];
    int sum=0;
#pragma unroll
    for(int u=0;u<10;u++) { int v=loc[u]; loc[u]=sum; sum+=v; }
    int tot=sum;
#pragma unroll
    for(int d=1;d<32;d<<=1) { int o=__shfl_up_sync(0xffffffff,sum,d); if(lane>=d) sum+=o; }
    int carry=sum-tot;
#pragma unroll
    for(int u=0;u<10;u++) prefix[lane*10+u]=loc[u]+carry;
    // prefix here is EXCLUSIVE; convert to inclusive layout used by fills:
  }
  __syncthreads();
  // fills expect inclusive prefix: inclusive[e] = exclusive[e] + tiles[e]
  for(int e=t;e<289;e+=256) if(hist[e]) {
    int tiles=(hist[e]+31)>>5;
    int incl=prefix[e]+tiles;             // re-add own tiles -> inclusive
    int base=incl-tiles;                  // == exclusive prefix
    for(int k=0;k<tiles;k++) experts[base+k]=e;
  }
  __syncthreads();
  for(int q=t;q<rows*9;q+=256) if(ids[q]>=0) {
    int e=ids[q], ordinal=atomicAdd(ordcnt+e,1);
    sorted[(prefix[e]+(ordinal>>5))*32+(ordinal&31)]=q;
  }
  if(t==0) { *n_post=(prefix[288]+((hist[288]+31)>>5))*32; *counter=0; }
}
