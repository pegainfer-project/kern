// Upstream oracle and launch-ABI capture bridge, not a serving dependency.
#include <cuda_runtime.h>
#include "fused_kernel.cuh"
extern "C" void dsv41_fused_decode(void* q, void* kv, void* ids, void* lengths,
    void* sink, void* out, void* lse, void* extra, void* extra_ids, void* extra_lengths,
    int rows, int pages, int extra_pages, int page_size, int topk, int extra_topk, int extra_page_size,
    float scale, void* stream, void* positions, void* rope, void* out_scales, int scale_rows) {
    using namespace sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn;
    ParamT<SparseAttnFwdMode::Decode> p{};
    p.b=1;p.s_q=rows;p.h_q=64;p.h_kv=1;p.d_qk=512;p.d_v=512;
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
    p.enable_q_norm=false;p.rms_norm_eps=1e-6f;
    p.token_positions=(uint32_t*)positions;p.is_rope_neox_style=false;p.rope_dim=64;p.cos_sin_cache=(float*)rope;
    p.n_wv_group=8;p.wv_group_size=8;p.num_per_channels=32;
    p.use_tma_aligned_col_major_sf=true;p.round_sf=true;p.use_packed_ue8m0=true;
    p.out_fp8=(fp8_e4m3*)out;p.out_sf=(uint32_t*)out_scales;
    p.stride_out_sf_head_dim=scale_rows;p.stride_out_sf_wv_group=scale_rows*32;
    run_fused_norm_rope_attn_rope_cast_fwd_kernel<Config{SparseAttnFwdMode::Decode,ModelType::V41,ModelType::V41_FP4,64,false}>(p);
}
