#pragma once
#include <cuda.h>
#include <cuda_bf16.h>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>
namespace dsv41 {
using namespace deep_gemm;
constexpr int H=5120,I=2304,MaxT=8192,Sms=152;
constexpr int align(int n,int a){return (n+a-1)/a*a;}
constexpr int divup(int n,int a){return (n+a-1)/a;}
constexpr int ring(int E,int K,int R){
 int result=0;for(int b:layout::kCandidateBlockM){int pool=divup(MaxT*R*K,b)+E/R;int n=sched::get_num_max_live_pool_blocks(pool,Sms,H,I)*b;result=result>n?result:n;}return align(result,layout::kLCMCandidateBlockM);
}
constexpr int sf_ring(int n){int result=0;for(int b:layout::kCandidateBlockM){int x=layout::get_num_sf_ring_tokens(n,b);result=result>x?result:x;}return result;}
template<int E,int K,int R,int B=16>struct Config{
 static constexpr int BlockM=B,StoreM=B<=16?8:B<=64?16:B<=192?32:40,SfM=align(B,128),Ring=ring(E,K,R),SfRing=sf_ring(Ring);
 static constexpr int Dispatch=align(E*4,1024)+align(H*4,1024);
 static constexpr int Fixed=Dispatch+align(2*StoreM*128*2,1024)+StoreM*8*4+(4+4+16+4)*8+2*sizeof(sched::TaskInfo<true>)+4;
 static constexpr int PerStage=B/2*128+128*128+SfM*4+128*4+16;
 static constexpr int Stages=(232448-Fixed)/PerStage,Smem=Fixed+Stages*PerStage;
};
}
