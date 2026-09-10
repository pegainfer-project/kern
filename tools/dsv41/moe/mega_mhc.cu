// Instantiate the unmodified DeepGEMM PR #432 shifted mHC kernel.
#include <deep_gemm/impls/sm100_mega_mhc.cuh>
using namespace deep_gemm;
void instantiate_dsv41_mhc() {
  auto ptr = reinterpret_cast<void*>(&sm100_mega_mhc_impl<5120,16,152,true,true,false,0>);
  asm volatile("" : : "g"(ptr));
}
