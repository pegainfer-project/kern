#include <cstdio>
#include <deep_gemm/layout/mqa_logits.cuh>
int main(){printf("%zu\n",sizeof(deep_gemm::layout::MQALogitsSharedStorage<32,128,true,4,256,3,10,3,cutlass::float_e2m1_t,float>));}
