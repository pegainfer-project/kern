// Split-KV fused decode: the translation unit that instantiates DecodeWithSplitKV plus its
// combine kernel; build_fused.sh compiles it and extracts the `dsv41_fused_split` module the
// serving manifests pin. Needs FlashMLA 4f38f29 with ../patches/flashmla-split-kv.patch.
#include <cuda_runtime.h>
#include "fused_kernel.cuh"
#include "combine.cuh"
namespace sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn {
template __global__ void fused_split_combine_kernel<32>(const float*, const float*, const float*, const uint32_t*, const float*,
    cutlass::float_e4m3_t*, uint32_t*, int, int, int, int, int, int, int);
}
extern "C" void dsv41_fused_decode_split(void* q, void* kv, void* ids, void* lengths,
    void* sink, void* out, void* lse, void* extra, void* extra_ids, void* extra_lengths,
    int rows, int pages, int extra_pages, int page_size, int topk, int extra_topk, int extra_page_size,
    float scale, void* stream, void* positions, void* rope, void* out_scales, int scale_rows,
    void* o_accum, void* lse_accum, int parts) {
    using namespace sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn;
    ParamT<SparseAttnFwdMode::DecodeWithSplitKV> p{};
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
    p.stream=(cudaStream_t)stream;
    // Partials laid out [s_q][parts][64][512] / [s_q][parts][64] so that no stride depends on s_q
    p.enable_split_kv=true;p.num_sm_parts=parts;p.o_accum=(float*)o_accum;p.lse_accum=(float*)lse_accum;
    p.stride_o_accum_split=32768;p.stride_o_accum_s_q=parts*32768;p.stride_o_accum_h_q=512;
    p.stride_lse_accum_split=64;p.stride_lse_accum_s_q=parts*64;
    p.enable_q_norm=false;p.rms_norm_eps=1e-6f;
    p.token_positions=(uint32_t*)positions;p.is_rope_neox_style=false;p.rope_dim=64;p.cos_sin_cache=(float*)rope;
    p.n_wv_group=8;p.wv_group_size=8;p.num_per_channels=32;
    p.use_tma_aligned_col_major_sf=true;p.round_sf=true;p.use_packed_ue8m0=true;
    p.out_fp8=(fp8_e4m3*)out;p.out_sf=(uint32_t*)out_scales;
    p.stride_out_sf_head_dim=scale_rows;p.stride_out_sf_wv_group=scale_rows*32;
    static_assert(offsetof(ParamT<SparseAttnFwdMode::DecodeWithSplitKV>, enable_split_kv) == 224);
    static_assert(offsetof(ParamT<SparseAttnFwdMode::DecodeWithSplitKV>, lse_accum) == 232);
    static_assert(offsetof(ParamT<SparseAttnFwdMode::DecodeWithSplitKV>, o_accum) == 240);
    static_assert(offsetof(ParamT<SparseAttnFwdMode::DecodeWithSplitKV>, stride_lse_accum_split) == 248);
    static_assert(offsetof(ParamT<SparseAttnFwdMode::DecodeWithSplitKV>, num_sm_parts) == 288);
    static_assert(offsetof(ParamT<SparseAttnFwdMode::DecodeWithSplitKV>, out_sf) == 352);
    run_fused_norm_rope_attn_rope_cast_fwd_kernel<Config{SparseAttnFwdMode::DecodeWithSplitKV,ModelType::V41,ModelType::V41_FP4,64,false}>(p);
    auto combine = &fused_split_combine_kernel<32>;
    combine<<<dim3(rows,1,8),256,0,(cudaStream_t)stream>>>((const float*)o_accum,(const float*)lse_accum,(const float*)sink,
        (const uint32_t*)positions,(const float*)rope,(cutlass::float_e4m3_t*)out,(uint32_t*)out_scales,parts,
        32768,parts*32768,64,parts*64,scale_rows*32,scale_rows);
}
