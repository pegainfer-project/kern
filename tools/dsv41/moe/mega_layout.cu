#include <cuda_fp8.h>
#include <deep_gemm/layout/mega_mhc.cuh>
#include <deep_gemm/layout/mega_gate.cuh>
#include <deep_gemm/layout/gemm.cuh>
#include <cstdio>
#include <cstddef>
int main(){
 printf("dense=%zu\n",sizeof(deep_gemm::layout::SM100FP8FP4GemmSharedStorage<4,2,2,64,128,128,64,64,128,128,1,cutlass::float_e4m3_t,cutlass::float_e4m3_t,cutlass::bfloat16_t>));
 printf("gate384_split8_stages12=%zu\n",sizeof(deep_gemm::layout::mega_gate::SharedStorage<12,2,384,16,128,64,true,false,false>));
 printf("mhc_scratch_per_task=%zu\n",(size_t)deep_gemm::layout::mega_mhc::Workspace<40>::get_num_scratch_bytes(1,1));
 printf("gate384=%zu gate128=%zu\n",sizeof(deep_gemm::layout::mega_gate::SharedStorage<4,2,384,16,128,64,true,false,false>),sizeof(deep_gemm::layout::mega_gate::SharedStorage<4,2,128,16,128,64,true,false,false>));using namespace deep_gemm::layout::mega_mhc;
 printf("{\"mhc_shared_bytes\":%zu,\"mix_bytes\":%zu,\"norm_bytes\":%zu}\n",sizeof(SharedStorage),sizeof(MixArgs),sizeof(NormArgs));
}
