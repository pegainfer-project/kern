#include "mega_config.cuh"
#include "sm100_fp8_fp4_mega_moe.cuh"
#define DSV41_MEGA_TM_PARAMS                                                                    \
  const __grid_constant__ CUtensorMap t0, const __grid_constant__ CUtensorMap t1,            \
      const __grid_constant__ CUtensorMap t2, const __grid_constant__ CUtensorMap t3,        \
      const __grid_constant__ CUtensorMap t4, const __grid_constant__ CUtensorMap t5,        \
      const __grid_constant__ CUtensorMap t6, const __grid_constant__ CUtensorMap t7,        \
      const __grid_constant__ CUtensorMap t8, const __grid_constant__ CUtensorMap s0,        \
      const __grid_constant__ CUtensorMap s1, const __grid_constant__ CUtensorMap s2,        \
      const __grid_constant__ CUtensorMap s3, const __grid_constant__ CUtensorMap s4,        \
      const __grid_constant__ CUtensorMap s5, const __grid_constant__ CUtensorMap s6,        \
      const __grid_constant__ CUtensorMap s7, const __grid_constant__ CUtensorMap s8
#define DSV41_MEGA_TM_ARGS t0, t1, t2, t3, t4, t5, t6, t7, t8, s0, s1, s2, s3, s4, s5, s6, s7, s8


#define WORLD(NAME,E,K,R) \
extern "C" __global__ __launch_bounds__(512,1) void NAME(void* y,int* stats,uint32_t tokens,const int64_t* peers,uint32_t rank,DSV41_MEGA_TM_PARAMS) { \
 using C=dsv41::Config<E,K,R>; \
 deep_gemm::sm100_fp8_fp4_mega_moe_body<8192,5120,2304,E,1,K,C::BlockM,128,128,C::StoreM,C::SfM,128,C::Ring,C::SfRing,C::Stages,5120,128,128,256,152,R,10.f,false,cutlass::detail::float_e2m1_unpacksmem_t>(y,stats,tokens,peers,rank,DSV41_MEGA_TM_ARGS); \
}
WORLD(dsv41_mega_moe_e384_r4,384,6,4)
WORLD(dsv41_mega_moe_e128_r4,128,3,4)
