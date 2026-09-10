// Conservative single-CTA GEMM groups, three expert groups for Flash target.
#include <deep_gemm/impls/sm100_bf16_mega_gate.cuh>
using namespace deep_gemm;
void instantiate_dsv41_gate() {
 auto target=reinterpret_cast<void*>(&sm100_bf16_mega_gate_impl<5120,384,16,4,128,1,1,3,150,6,1,true,false,false,false,true,false,false,false>);
 auto draft=reinterpret_cast<void*>(&sm100_bf16_mega_gate_impl<5120,128,16,4,128,1,1,1,152,3,1,true,false,false,false,true,false,false,false>);
 asm volatile(""::"g"(target),"g"(draft));
}
