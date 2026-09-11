// Unmodified DeepGEMM MXFP8 GEMMs; dynamic M/N/K, row-major operands. The dense
// instance returns BF16; O-A returns MXFP8 for WO_B to consume unchanged.
// One instance per tile config in dense.py CONFIGS (block_m, block_n, stages);
// the generator names the instance it wants by its mangled template arguments.
// The C/D swizzle is one store-block row: min(128, block_n * sizeof(out)).
#include <deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh>
using namespace deep_gemm;
#define DSV41_DENSE(BM,BN,CD,STAGES) {auto p=reinterpret_cast<void*>(&sm100_fp8_fp4_gemm_1d1d_impl<cute::UMMA::Major::K,cute::UMMA::Major::K,32,32,32,0,0,0,BM,BN,128,1,128,128,CD,STAGES,2,128,128,1,true,152,false,false,GemmType::Normal,false,cutlass::float_e4m3_t,cutlass::float_e4m3_t,cutlass::bfloat16_t,epilogue::transform::EpilogueIdentity>);asm volatile(""::"g"(p));}
#define DSV41_OA(BM,BN,CD,STAGES) {auto p=reinterpret_cast<void*>(&sm100_fp8_fp4_gemm_1d1d_impl<cute::UMMA::Major::K,cute::UMMA::Major::K,32,32,32,0,0,0,BM,BN,128,8,128,128,CD,STAGES,2,128,128,1,true,152,false,false,GemmType::Batched,false,cutlass::float_e4m3_t,cutlass::float_e4m3_t,cutlass::float_e4m3_t,epilogue::transform::EpilogueDynamicScaledFP8>);asm volatile(""::"g"(p));}
void instantiate_dsv41_dense(){
  DSV41_DENSE(64,128,128,4)
  DSV41_DENSE(32,16,32,32)
  DSV41_DENSE(32,32,64,24)
  DSV41_DENSE(32,64,128,16)
  DSV41_DENSE(32,128,128,10)
  DSV41_DENSE(32,256,128,5)
  DSV41_DENSE(64,16,32,20)
  DSV41_DENSE(64,32,64,16)
  DSV41_DENSE(64,64,128,12)
  DSV41_DENSE(64,128,128,8)
  DSV41_DENSE(64,256,128,5)
}
void instantiate_dsv41_oa(){
  DSV41_OA(64,128,128,4)
  DSV41_OA(32,32,32,24)
  DSV41_OA(32,64,64,16)
  DSV41_OA(32,128,128,10)
  DSV41_OA(32,256,128,5)
  DSV41_OA(64,32,32,16)
  DSV41_OA(64,64,64,12)
  DSV41_OA(64,128,128,8)
  DSV41_OA(64,256,128,5)
}
