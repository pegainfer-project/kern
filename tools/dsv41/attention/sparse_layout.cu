#include <cstdio>
#include <deep_gemm/layout/sparse_mqa_logits.cuh>
#include <deep_gemm/scheduler/sm100_sparse_mqa_logits_metadata.cuh>
int main(){
    printf("{\"score\":%zu,\"metadata\":%zu}\n",
        sizeof(deep_gemm::layout::sparse_mqa_logits::SharedStorage<2,8,640,2,5,5,cutlass::float_e2m1_t>),
        (sizeof(deep_gemm::sched::sparse_mqa_logits::SharedStorage<2,2048,80,8,256>)+127)/128*128);
}
