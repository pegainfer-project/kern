// Flash target: three expert groups of 128. The 4-stage split-K 1 instance
// streams each group's 3.9 MB weight through one CTA; split-K 8 with the
// smem-maximal 12 stages is upstream's heuristic pick at small token counts
// (24 logical CTAs, 144 SMs = six worker groups).
#include <deep_gemm/impls/sm100_bf16_mega_gate.cuh>
using namespace deep_gemm;
void instantiate_dsv41_gate() {
 auto target=reinterpret_cast<void*>(&sm100_bf16_mega_gate_impl<5120,384,16,4,128,1,1,3,150,6,1,true,false,false,false,true,false,false,false>);
 auto target_split=reinterpret_cast<void*>(&sm100_bf16_mega_gate_impl<5120,384,16,12,128,1,8,3,144,6,1,true,false,false,false,true,false,false,false>);
 auto draft=reinterpret_cast<void*>(&sm100_bf16_mega_gate_impl<5120,128,16,4,128,1,1,1,152,3,1,true,false,false,false,true,false,false,false>);
 asm volatile(""::"g"(target),"g"(target_split),"g"(draft));
}
