// Host-only oracle bridge. The serving artifact uses the extracted cubin.
#include <cuda_runtime.h>
#include "kernels/sm100/prefill/sparse/fwd/head64/phase1.cuh"
extern "C" void dsv41_sparse_prefill(void* q, void* kv, void* indices, void* sink,
    void* lengths, void* out, void* max_logits, void* lse,
    int rows, int kv_rows, int topk, float scale, void* stream) {
    SparseAttnFwdParams p{};
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
    sm100::prefill::sparse_fwd::head64::run_sparse_fwd_phase1_kernel<SparseAttnFwdMode::Prefill,512>(p);
}
