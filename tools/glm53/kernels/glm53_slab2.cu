// GLM-5.3 MTP-round DSA indexer glue (slab 2): k_norm + fast_hadamard +
// _act_quant + weights_proj as ONE launch per DSA layer. Bitwise vs the
// four-call reference by construction:
//  - k_norm: verbatim glm53_layer_norm_128 (glm53_dsa.cu, the round's own
//    glm53_dsa_prep module), executed by CTA (t,0) with the ORIGINAL
//    128-thread block shape, so not one instruction changes.
//  - hadamard+quant: verbatim glm53_dsa_had_quant_v2 math (decode), already
//    bitwise-gated against the pinned sglang fast_hadamard<16,7,bf16> and
//    _act_quant_kernel cubins that the round launches. Warp-per-vec: lane l
//    replays virtual threads {l, l+32, l+64, l+96} of the 128-thread
//    reference. The butterfly network and every rounding point (BF16
//    materialize, exp2f/ceilf/log2f scale, div.full.f32 quant) are
//    unchanged; fmaxf is exact, so the absmax reduction tree is
//    schedule-free.
//  - weights_proj: verbatim glm53_weights_proj (glm53_dsa.cu), warp-per-head.
//
// Launch shape: grid [rows, 8], block 128. CTA (t,c) covers head vecs/heads
// g = 4c..4c+3, one warp each; the SAME warp produces qs[g] (phase B) and
// consumes it (phase C lane 0), so no CTA barrier is needed between phases.
// CTA (t,0) additionally runs the k_norm first (its four warps then do
// vecs 0..3 like the others). The one-CTA-per-token alternative starves:
// 32 warps of weights_proj traffic serialize through one SM's L2 port
// (measured +3us vs the 4-launch chain); 8 CTAs/token spread it.
//
// iq8/qs/w are written at k_norm's call slot and the intermediate hadamard
// buffer (iqh) disappears. spec_kpool_update reads ik/gs, not iq8/qs/w, so
// relocating weights_proj ahead of it is dependency-legal (asserted in
// ops_slab2.fuse_round_manifest; docs/glm53/round_megakernel.md section 4).
#include <cuda_bf16.h>
#include <cuda_fp8.h>

using bf16 = __nv_bfloat16;

__device__ __forceinline__ float slab2_warp_max(float v) {
    for (int d = 16; d; d >>= 1) v = fmaxf(v, __shfl_xor_sync(~0u, v, d));
    return v;
}
__device__ __forceinline__ float slab2_tl_div(float a, float b) {
    float y;
    asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
    return y;
}

extern "C" __global__ __launch_bounds__(128, 1) void glm53_slab2_dsa_index_glue(
    bf16* __restrict__ ik, const float* __restrict__ kn_w, const float* __restrict__ kn_b,
    float eps, const bf16* __restrict__ iq, unsigned char* __restrict__ iq8,
    float* __restrict__ qs, const bf16* __restrict__ x_norm,
    const float* __restrict__ wp_w, float* __restrict__ w) {
    const unsigned int t = blockIdx.x;
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int g = blockIdx.y * 4 + warp;  // head vec / head of this warp

    // ---- Phase A (CTA (t,0) only): k_norm, verbatim 128-thread body. ----
    if (blockIdx.y == 0) {
        const bf16* xr = ik + (size_t)t * 128;
        bf16* yr = ik + (size_t)t * 128;
        float v = 0.f, v2 = 0.f;
        if (threadIdx.x < 128) {
            v = __bfloat162float(xr[threadIdx.x]);
            v2 = v * v;
        }
        float s1 = v, s2 = v2;
        for (int off = 16; off > 0; off >>= 1) {
            s1 += __shfl_down_sync(~0u, s1, off);
            s2 += __shfl_down_sync(~0u, s2, off);
        }
        __shared__ float red[8];
        if ((threadIdx.x & 31) == 0) { red[threadIdx.x >> 5] = s1; red[4 + (threadIdx.x >> 5)] = s2; }
        __syncthreads();
        if (threadIdx.x < 4) {
            float a = red[threadIdx.x], c = red[4 + threadIdx.x];
            for (int off = 2; off > 0; off >>= 1) {
                a += __shfl_down_sync(0xfu, a, off);
                c += __shfl_down_sync(0xfu, c, off);
            }
            if (threadIdx.x == 0) {
                float mean = a * (1.f / 128.f);
                float var = c * (1.f / 128.f) - mean * mean;
                red[0] = mean;
                red[1] = rsqrtf(var + eps);
            }
        }
        __syncthreads();
        if (threadIdx.x < 128) {
            unsigned int i = threadIdx.x;
            yr[i] = __float2bfloat16((v - red[0]) * red[1] * kn_w[i] + kn_b[i]);
        }
    }

    // ---- Phase B: hadamard128 + per-vec fp8 quant for vec g (warp-local). ----
    __shared__ float s[4][128];
    {
        const bf16* xr = iq + (size_t)t * 4096 + (size_t)g * 128;
        s[warp][lane] = __bfloat162float(xr[lane]);
        s[warp][lane + 32] = __bfloat162float(xr[lane + 32]);
        s[warp][lane + 64] = __bfloat162float(xr[lane + 64]);
        s[warp][lane + 96] = __bfloat162float(xr[lane + 96]);
        __syncwarp();
        #pragma unroll
        for (int d = 1; d < 128; d <<= 1) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                int i = lane + 32 * j;  // virtual reference thread, always < 64
                int a = (i / d) * 2 * d + i % d, b = a + d;
                float u = s[warp][a], v = s[warp][b];
                s[warp][a] = u + v;
                s[warp][b] = u - v;
            }
            __syncwarp();
        }
        float v0 = __bfloat162float(__float2bfloat16_rn(s[warp][lane] * 0.08838834764831844f));
        float v1 = __bfloat162float(__float2bfloat16_rn(s[warp][lane + 32] * 0.08838834764831844f));
        float v2 = __bfloat162float(__float2bfloat16_rn(s[warp][lane + 64] * 0.08838834764831844f));
        float v3 = __bfloat162float(__float2bfloat16_rn(s[warp][lane + 96] * 0.08838834764831844f));
        float am = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
        am = slab2_warp_max(am);
        float scale = exp2f(ceilf(log2f(fmaxf(am, 1e-4f) * (1.f / 448.f))));
        if (lane == 0) qs[(size_t)t * 32 + g] = scale;
        unsigned char* orow = iq8 + (size_t)t * 4096 + (size_t)g * 128;
        orow[lane] = __nv_cvt_float_to_fp8(fminf(fmaxf(slab2_tl_div(v0, scale), -448.f), 448.f),
                                           __NV_SATFINITE, __NV_E4M3);
        orow[lane + 32] = __nv_cvt_float_to_fp8(fminf(fmaxf(slab2_tl_div(v1, scale), -448.f), 448.f),
                                                __NV_SATFINITE, __NV_E4M3);
        orow[lane + 64] = __nv_cvt_float_to_fp8(fminf(fmaxf(slab2_tl_div(v2, scale), -448.f), 448.f),
                                                __NV_SATFINITE, __NV_E4M3);
        orow[lane + 96] = __nv_cvt_float_to_fp8(fminf(fmaxf(slab2_tl_div(v3, scale), -448.f), 448.f),
                                                __NV_SATFINITE, __NV_E4M3);
    }

    // ---- Phase C: weights_proj for head g (verbatim warp body). ----
    {
        const bf16* xr = x_norm + (size_t)t * 4096;
        const float* wr = wp_w + (size_t)g * 4096;
        float acc = 0.f;
        for (unsigned int i = lane; i < 4096; i += 32)
            acc += __bfloat162float(xr[i]) * wr[i];
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(~0u, acc, off);
        if (lane == 0) {
            float tt = acc * 0.17677669529663687f;  // 32^-0.5
            tt = tt * qs[(size_t)t * 32 + g];
            tt = tt * 0.08838834764831844f;         // 128^-0.5 (softmax_scale)
            w[(size_t)t * 32 + g] = tt;
        }
    }
}
