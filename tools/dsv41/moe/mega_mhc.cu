// Instantiate the unmodified DeepGEMM PR #432 shifted mHC kernel, with the
// normalized output stored both as BF16 and as MXFP8: column-major GEMM scales
// for the attention projections (SF_BLOCK_M 0), row-major routed scales plus
// the Mega MoE shared-expert scale pages for the FFN (SF_BLOCK_M 16).
#include <deep_gemm/impls/sm100_mega_mhc.cuh>
using namespace deep_gemm;
void instantiate_dsv41_mhc() {
  auto gemm = reinterpret_cast<void*>(&sm100_mega_mhc_impl<5120,16,152,true,true,true,0>);
  auto moe = reinterpret_cast<void*>(&sm100_mega_mhc_impl<5120,16,152,true,true,true,16>);
  asm volatile("" : : "g"(gemm), "g"(moe));
}
