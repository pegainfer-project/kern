// DSV4.1 boundary operations. Match inference/model.py Expert and Block.
#include <cuda_bf16.h>
#include <math.h>
extern "C" __global__ void dsv41_swiglu(__nv_bfloat16* y, const __nv_bfloat16* gate, const __nv_bfloat16* up, int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) { float g=fminf(__bfloat162float(gate[i]),10.f); float u=fminf(fmaxf(__bfloat162float(up[i]),-10.f),10.f); y[i]=__float2bfloat16_rn((g/(1.f+expf(-g)))*u); }
}
extern "C" __global__ void dsv41_hc_init(__nv_bfloat16* y, const __nv_bfloat16* x, int tokens) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<tokens*4*5120) y[i]=x[(i/(4*5120))*5120+i%5120];
}
extern "C" __global__ void dsv41_hc_pre(__nv_bfloat16* y,const __nv_bfloat16* x,const float* pre,int tokens) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<tokens*5120) { int t=i/5120,d=i%5120; float sum=0; for(int r=0;r<4;r++) sum+=pre[t*4+r]*__bfloat162float(x[(t*4+r)*5120+d]); y[i]=__float2bfloat16_rn(sum); }
}
extern "C" __global__ void dsv41_hc_post(__nv_bfloat16* y,const __nv_bfloat16* x,const __nv_bfloat16* residual,const float* post,const float* comb,int tokens) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<tokens*4*5120) { int t=i/(4*5120),r=(i/5120)%4,d=i%5120; float sum=0;
    // Reference reduces source-route dimension: comb[source, destination].
    for(int s=0;s<4;s++) sum+=comb[t*16+s*4+r]*__bfloat162float(residual[(t*4+s)*5120+d]);
    y[i]=__float2bfloat16_rn(post[t*4+r]*__bfloat162float(x[t*5120+d])+sum);
  }
}
extern "C" __global__ void dsv41_zero_u64(unsigned long long* y,int n) {int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=0;}
extern "C" __global__ void dsv41_hc_mean(__nv_bfloat16* y,const __nv_bfloat16* x,int tokens){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=tokens*5120)return;int t=i/5120,d=i%5120;float sum=0;for(int r=0;r<4;r++)sum+=__bfloat162float(x[(t*4+r)*5120+d]);y[i]=__float2bfloat16_rn(sum*.25f);
}
// Identity boundary coefficients; initial_pre is used ONLY at stack entry.
// Engram/tap materialization must preserve the actual previous pre mix.
extern "C" __global__ void dsv41_hc_identity(float* zero_post,float* identity_comb,float* initial_pre,int tokens){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<tokens*16)identity_comb[i]=(i%16)/4==i%4?1.f:0.f;if(i<tokens*4){zero_post[i]=0;initial_pre[i]=i%4==0?1.f:0.f;}
}
