// Host-only oracle bridge. The serving artifact uses the extracted cubin.
#include <cuda_runtime.h>
#include "fused_kernel.cuh"
extern "C" void dsv41_fused_prefill(void* q, void* kv, void* indices, void* sink,
    void* lengths, void* out, void* max_logits, void* lse,
    int rows, int kv_rows, int topk, float scale, void* stream, void* positions, void* rope, void* out_scales, int scale_rows) {
    using namespace sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn;
    ParamT<SparseAttnFwdMode::Prefill> p{};
    p.s_q=rows; p.s_kv=kv_rows; p.h_q=64; p.h_kv=1; p.d_qk=512; p.d_v=512;
    p.topk=topk; p.sm_scale=scale; p.sm_scale_div_log2=scale*1.4426950408889634f;
    p.q=(cutlass::bfloat16_t*)q; p.kv=(cutlass::bfloat16_t*)kv;
    p.indices=(int*)indices; p.attn_sink=(float*)sink; p.topk_length=(int*)lengths;
    p.stride_q_s_q=64*512; p.stride_q_h_q=512;
    p.stride_kv_s_kv=512; p.stride_kv_h_kv=512;
    p.stride_indices_s_q=topk; p.stride_indices_h_kv=topk;
    p.out=(cutlass::bfloat16_t*)out; p.max_logits=(float*)max_logits; p.lse=(float*)lse;
    cudaDeviceGetAttribute(&p.num_sm,cudaDevAttrMultiProcessorCount,0);
    p.stream=(cudaStream_t)stream;
    p.enable_q_norm=false;p.rms_norm_eps=1e-6f;
    p.token_positions=(uint32_t*)positions;p.is_rope_neox_style=false;p.rope_dim=64;p.cos_sin_cache=(float*)rope;
    p.n_wv_group=8;p.wv_group_size=8;p.num_per_channels=32;
    p.use_tma_aligned_col_major_sf=true;p.round_sf=true;p.use_packed_ue8m0=true;
    p.out_fp8=(fp8_e4m3*)out;p.out_sf=(uint32_t*)out_scales;
    p.stride_out_sf_head_dim=scale_rows;p.stride_out_sf_wv_group=scale_rows*32;
    run_fused_norm_rope_attn_rope_cast_fwd_kernel<Config{SparseAttnFwdMode::Prefill,ModelType::V4,ModelType::V4,64,false}>(p);
}
