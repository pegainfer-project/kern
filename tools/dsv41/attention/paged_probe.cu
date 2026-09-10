// Upstream oracle and launch-ABI capture bridge, not a serving dependency.
#include <cuda_runtime.h>
#include "kernels/sm100/decode/sparse/head64/kernel.cuh"
extern "C" void dsv41_paged_decode(void* q, void* kv, void* ids, void* lengths,
    void* sink, void* out, void* lse, void* extra, void* extra_ids, void* extra_lengths,
    int rows, int pages, int extra_pages, int page_size, int topk, int extra_topk, int extra_page_size,
    float scale, void* stream) {
    SparseAttnDecodeParams p{};
    p.b=rows;p.s_q=1;p.h_q=64;p.h_kv=1;p.d_qk=512;p.d_v=512;
    p.sm_scale=scale;p.sm_scale_div_log2=scale*1.4426950408889634f;
    p.num_blocks=pages;p.page_block_size=page_size;p.topk=topk;
    p.model_type=ModelType::V41;p.extra_model_type=ModelType::V41_FP4;
    p.q=(cutlass::bfloat16_t*)q;p.kv=(cutlass::bfloat16_t*)kv;p.indices=(int*)ids;
    p.topk_length=(int*)lengths;p.attn_sink=(float*)sink;p.out=(cutlass::bfloat16_t*)out;p.lse=(float*)lse;
    p.extra_num_blocks=extra_pages;p.extra_page_block_size=extra_page_size;p.extra_topk=extra_topk;
    p.extra_kv=(cutlass::bfloat16_t*)extra;p.extra_indices=(int*)extra_ids;p.extra_topk_length=(int*)extra_lengths;
    p.stride_q_b=32768;p.stride_q_s_q=32768;p.stride_q_h_q=512;
    p.stride_kv_block=page_size*528;p.stride_kv_row=528;
    p.stride_indices_b=topk;p.stride_indices_s_q=topk;
    p.stride_lse_b=64;p.stride_lse_s_q=64;
    p.stride_o_b=32768;p.stride_o_s_q=32768;p.stride_o_h_q=512;
    p.stride_extra_kv_block=extra_page_size*288;p.stride_extra_kv_row=288;
    p.stride_extra_indices_b=extra_topk;p.stride_extra_indices_s_q=extra_topk;
    p.stream=(cudaStream_t)stream;p.enable_split_kv=false;
    using namespace sm100::decode::sparse::head64;
    run_flash_splitkv_mla_fp8_sparse_kernel<Config{ModelType::V41,ModelType::V41_FP4,false}>(p);
}
