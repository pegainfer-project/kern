// Pinned DeepSelect upstream launch bridge. Runtime executes extracted cubins.
#include <cuda_runtime.h>
#include "cuda_kernels/config.h"
#include "cuda_kernels/v3/topk_select.cuh"
#include "cuda_kernels/v3_fp32/topk_select.cuh"
extern "C" void dsv41_select(void* input,void* end,void* output,int rows,int width,int topk,int bf16,void* stream){
    TopkSelectArgs p{};
    p.batch_size=rows;p.vocab_size=width;p.topk=topk;p.input=input;p.output_index=output;
    p.end_ptr=(int*)end;p.stride_input_batch=width;p.stride_output_index_batch=topk;
    p.sorted_index=true;p.return_value=false;p.idx_oob_fill_value=-1;p.value_oob_fill_value=-INFINITY;
    p.abort_when_nan_found=true;p.stream=(cudaStream_t)stream;
    int smem;cudaDeviceGetAttribute(&smem,cudaDevAttrMaxSharedMemoryPerMultiprocessor,0);p.shared_memory_size_per_sm=smem;
    if(bf16){
        if(topk<=512)topk_select_bf16_normal::run_topk_select_kernel<TopkSelectConfig<nv_bfloat16,int,false,true,false,512,256,2,4096,4096,4>>(p);
        else topk_select_bf16_normal::run_topk_select_kernel<TopkSelectConfig<nv_bfloat16,int,false,true,false,4096,512,1,8192,4096,3>>(p);
    }else{
        if(topk<=512)topk_select_fp32::run_topk_select_kernel<TopkSelectConfig<float,int,false,true,false,512,512,1,8192,4096,3>>(p);
        else topk_select_fp32::run_topk_select_kernel<TopkSelectConfig<float,int,false,true,false,4096,256,1,4096,4096,3>>(p);
    }
}
