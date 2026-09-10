// Unmodified DeepGEMM MXFP8 dense GEMM; dynamic M/N/K, row-major operands.
#include <deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh>
using namespace deep_gemm;
void instantiate_dsv41_dense(){auto p=reinterpret_cast<void*>(&sm100_fp8_fp4_gemm_1d1d_impl<cute::UMMA::Major::K,cute::UMMA::Major::K,32,32,32,0,0,0,64,128,128,1,128,128,128,4,2,128,128,1,true,152,false,false,GemmType::Normal,false,cutlass::float_e4m3_t,cutlass::float_e4m3_t,cutlass::bfloat16_t,epilogue::transform::EpilogueIdentity>);asm volatile(""::"g"(p));}

void instantiate_dsv41_oa(){auto p=reinterpret_cast<void*>(&sm100_fp8_fp4_gemm_1d1d_impl<cute::UMMA::Major::K,cute::UMMA::Major::K,32,32,32,0,0,0,64,128,128,8,128,128,128,4,2,128,128,1,true,152,false,false,GemmType::Batched,false,cutlass::float_e4m3_t,cutlass::float_e4m3_t,cutlass::bfloat16_t,epilogue::transform::EpilogueIdentity>);asm volatile(""::"g"(p));}
