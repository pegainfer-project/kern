#include <deep_gemm/impls/sm100_mqa_logits.cuh>
using namespace deep_gemm;
// Unmodified upstream MXFP4 contiguous scorer. Dedicated prefill source layer.
static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&sm100_mqa_logits<32,128,true,false,true,false,
        4,256,3,10,148,128,256,cutlass::float_e2m1_t,float,float>);
}
