// Instantiate the unmodified DeepGEMM PR #432 shifted mHC kernel, with the
// normalized output stored both as BF16 and as MXFP8: column-major GEMM scales
// for the attention projections (SF_BLOCK_M 0), row-major routed scales plus
// the Mega MoE shared-expert scale pages for the FFN (SF_BLOCK_M 16).
#include <deep_gemm/impls/sm100_mega_mhc.cuh>
using namespace deep_gemm;
void instantiate_dsv41_mhc() {
  // 16 splits is upstream's deterministic-mode default; 40 is what its
  // heuristic picks for up to three 64-row m-blocks on 152 SMs (two K-blocks
  // per task), the serving case at capacity 128. The SM count is a template
  // parameter the split reduction's barriers count on: an instance launched
  // on a device with fewer SMs waits for CTAs that never become resident, so
  // every device the manifest targets gets its own (GB300 152, B300 148).
  auto gemm16 = reinterpret_cast<void*>(&sm100_mega_mhc_impl<5120,16,152,true,true,true,0>);
  auto moe16 = reinterpret_cast<void*>(&sm100_mega_mhc_impl<5120,16,152,true,true,true,16>);
  auto gemm40 = reinterpret_cast<void*>(&sm100_mega_mhc_impl<5120,40,152,true,true,true,0>);
  auto moe40 = reinterpret_cast<void*>(&sm100_mega_mhc_impl<5120,40,152,true,true,true,16>);
  auto gemm16_148 = reinterpret_cast<void*>(&sm100_mega_mhc_impl<5120,16,148,true,true,true,0>);
  auto moe16_148 = reinterpret_cast<void*>(&sm100_mega_mhc_impl<5120,16,148,true,true,true,16>);
  auto gemm40_148 = reinterpret_cast<void*>(&sm100_mega_mhc_impl<5120,40,148,true,true,true,0>);
  auto moe40_148 = reinterpret_cast<void*>(&sm100_mega_mhc_impl<5120,40,148,true,true,true,16>);
  asm volatile("" : : "g"(gemm16), "g"(moe16), "g"(gemm40), "g"(moe40),
                     "g"(gemm16_148), "g"(moe16_148), "g"(gemm40_148), "g"(moe40_148));
}
