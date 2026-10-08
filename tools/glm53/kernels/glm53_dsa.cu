// GLM-5.3 DSA glue kernels: indexer k_norm, fused head-gate weights_proj,
// MLA latent KV store, kpool decode prep, topk clamp.
// Replaces sglang sites 72, 78-81, 82(part), 87, 88 (docs/glm53/dsa.md).
// Build: nvcc -cubin -arch=sm_90a (tools/glm53/build_kernels.sh).
#include <cuda_bf16.h>
#include <cuda_fp8.h>

using bf16 = __nv_bfloat16;

extern "C" {

// Site 72: LayerNorm over 128, fp32 weight/bias, eps inside rsqrt.
// y[t] = (x[t] - mean) * rsqrt(var + eps) * w + b; x/y bf16 [T,128].
__global__ void glm53_layer_norm_128(const bf16* __restrict__ x, const float* __restrict__ w,
                                     const float* __restrict__ b, bf16* __restrict__ y,
                                     float eps) {
    unsigned int t = blockIdx.x;
    const bf16* xr = x + (size_t)t * 128;
    bf16* yr = y + (size_t)t * 128;
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
        yr[i] = __float2bfloat16((v - red[0]) * red[1] * w[i] + b[i]);
    }
}

// Sites 78-81 fused: weights[s,h] = dot(f32(x[s]), w[h]) * 32^-0.5 * q_scale[s,h] * 128^-0.5.
// x bf16 [S,4096], w fp32 [32,4096], q_scale fp32 [S,32], out fp32 [S,32].
// Multiply order mirrors sglang: ((dot * RSQRT32) * q_scale) * RSQRT128.
__global__ void glm53_weights_proj(const bf16* __restrict__ x, const float* __restrict__ w,
                                   const float* __restrict__ q_scale, float* __restrict__ out) {
    unsigned int s = blockIdx.x;
    unsigned int h = threadIdx.x >> 5;   // warp = head (block must be 32 warps)
    unsigned int lane = threadIdx.x & 31;
    const bf16* xr = x + (size_t)s * 4096;
    const float* wr = w + (size_t)h * 4096;
    float acc = 0.f;
    for (unsigned int i = lane; i < 4096; i += 32)
        acc += __bfloat162float(xr[i]) * wr[i];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(~0u, acc, off);
    if (lane == 0) {
        float t = acc * 0.17677669529663687f;              // 32^-0.5
        t = t * q_scale[(size_t)s * 32 + h];
        t = t * 0.08838834764831844f;                      // 128^-0.5 (softmax_scale)
        out[(size_t)s * 32 + h] = t;
    }
}

// Site 87: kv[loc[s]] = k_nope[s]; 512 bf16 latent per token.
// dst = kv_base + loc[s] * slot_pitch + layer_off (bytes); grid [S].
__global__ void glm53_kv_store(char* __restrict__ kv_base, const bf16* __restrict__ k_nope,
                               const int* __restrict__ loc, long long slot_pitch,
                               long long layer_off) {
    unsigned int s = blockIdx.x;
    const uint4* src = reinterpret_cast<const uint4*>(k_nope + (size_t)s * 512);
    uint4* dst = reinterpret_cast<uint4*>(kv_base + (long long)loc[s] * slot_pitch + layer_off);
    for (unsigned int i = threadIdx.x; i < 64; i += blockDim.x) dst[i] = src[i];  // 64 x 16B = 1024B
}

// Sites 82 prep + fa3 seqused_k, from seq_lens (decode). Unit-256 pages
// (arch_decode.md section 1): one block-table entry covers 256 tokens =
// 64 pools, so the logits kernel reads block_table DIRECTLY (no pooled
// ::4 table build).
// pool_seqlens[s]   = seq / 4                       (unclamped; site 85 lengths)
// pool_ctx_lens[s]  = max(seq / 4, 1)               (clamped; sites 83/84)
// dsa_seqlens[s]    = min(4 * (seq / 4), 2048) + seq % 4   (site 89 seqused_k, topk valid cols)
// slot_table[s, t]  = block_table[s, t/256]*256 + t%256 for t < seq
//                     (token -> physical kv slot; the page_table_1 site 85 maps through)
__global__ void glm53_dsa_prep(const int* __restrict__ seq_lens, const int* __restrict__ block_table,
                               int bt_cols, int* __restrict__ pool_seqlens,
                               int* __restrict__ pool_ctx_lens, int* __restrict__ dsa_seqlens,
                               int* __restrict__ slot_table, int st_cols) {
    unsigned int s = blockIdx.x;
    int seq = seq_lens[s];
    if (threadIdx.x == 0) {
        int pools = seq >> 2;
        pool_seqlens[s] = pools;
        pool_ctx_lens[s] = pools > 0 ? pools : 1;
        int hist = pools << 2;
        dsa_seqlens[s] = (hist > 2048 ? 2048 : hist) + (seq & 3);
    }
    const int* bt = block_table + (size_t)s * bt_cols;
    int* st = slot_table + (size_t)s * st_cols;
    for (int t = threadIdx.x; t < seq && t < st_cols; t += blockDim.x)
        st[t] = bt[t >> 8] * 256 + (t & 255);
}

// Triton lowering mirrors (arch_decode.md section 0.2): tl.exp lowers to
// ex2.approx.f32 of x*log2e, and fdiv to div.full.f32. Plain CUDA gives
// expf (a libdevice polynomial) and div.rn.f32 (IEEE), which drifts the
// pooled key by 1 ulp and can flip a topk near-tie.
__device__ __forceinline__ float tl_exp(float x) {
    float y;
    asm("mul.f32 %0, %1, 0f3FB8AA3B;" : "=f"(y) : "f"(x));
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(y));
    return y;
}
__device__ __forceinline__ float tl_div(float a, float b) {
    float y;
    asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b));
    return y;
}

// Site 88: x[i] = max(x[i], 0) on topk_indices (replaces -1 padding with dummy slot 0).
__global__ void glm53_clamp0_i32(int* __restrict__ x, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride) x[i] = x[i] < 0 ? 0 : x[i];
}

// Site 77: kpool decode update (handwritten; the mined triton cubins bake
// BLOCK_TABLE_COLS + stride specializations from the seq=3 capture).
// Transcribed from _kpool_decode_update_and_maybe_write_cache_kernel
// (kpool_fp8_index.py:1027-1195). One block per token, 128 threads.
//   idx_base:  this layer's kpool state base (state page = 11 x 8448 B, one
//              per 256-token block-table entry; layer offset at the call site)
//   tail_base: idx_tail state base; line = 4096 B, an 8-entry ring:
//              [8][128] bf16 k at +0, [8][128] bf16 score at +2048
//   key/score: this token's k_norm'ed index-k / gate score, bf16 [S,128]
//   ape:       compress positional encoding, f32 [4,128]
//   bt:        shared token-page table [S, bt_cols] (page = 256 tokens)
//   positions/seq_lens/valid: pos, seq, Fill::Valid per seq (pad guard;
//              kern has no reserved token slot 0, so slots==0 is NOT one)
__global__ void glm53_kpool_update(char* __restrict__ idx_base, char* __restrict__ tail_base,
                                   const bf16* __restrict__ key, const bf16* __restrict__ score,
                                   const float* __restrict__ ape, const int* __restrict__ bt,
                                   const int* __restrict__ lines, const int* __restrict__ positions,
                                   const int* __restrict__ seq_lens, const int* __restrict__ valid,
                                   int bt_cols) {
    unsigned int row = blockIdx.x;
    unsigned int i = threadIdx.x;              // one element per thread, dim = 128
    int pos = positions[row];
    int seq = seq_lens[row];
    bool pos_valid = (valid[row] != 0) && (pos >= 0) && (pos < seq);

    float k_cur = __bfloat162float(key[(size_t)row * 128 + i]);
    float s_cur = __bfloat162float(score[(size_t)row * 128 + i]);

    char* line = tail_base + (size_t)lines[row] * 4096;
    bf16* tail_k = reinterpret_cast<bf16*>(line);
    bf16* tail_score = reinterpret_cast<bf16*>(line + 2048);
    int phys = pos & 7;

    int slot = pos & 3;  // pool size is 4; the physical tail ring has 8 slots
    if (pos_valid && slot == 3) {
        // softmax over the 4 pool slots of (score + ape), per element
        float sc[4], kv[4];
        float mx = -__int_as_float(0x7f7fffff);
        int pool_start = pos - 3;
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            int ph = (pool_start + p) & 7;
            float sbuf = __bfloat162float(tail_score[ph * 128 + i]);
            float kbuf = __bfloat162float(tail_k[ph * 128 + i]);
            sc[p] = (p == 3 ? s_cur : sbuf) + ape[p * 128 + i];
            kv[p] = p == 3 ? k_cur : kbuf;
            mx = fmaxf(mx, sc[p]);
        }
        float denom = 0.f, acc = 0.f;
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            float pr = tl_exp(sc[p] - mx);
            denom += pr;
            acc += kv[p] * pr;
        }
        float x = __bfloat162float(__float2bfloat16(tl_div(acc, denom)));
        // hadamard128: 7 butterfly stages (g,s) = (64,1)..(1,64), then x * 128^-0.5
        __shared__ float h[128];
        h[i] = x;
        __syncthreads();
        #pragma unroll
        for (int s = 1; s < 128; s <<= 1) {
            if (i < 64) {
                int gi = i / s, si = i % s;
                int i0 = gi * 2 * s + si, i1 = i0 + s;
                float a = h[i0], b = h[i1];
                h[i0] = a + b;   // race-free within a stage: 64 disjoint pairs
                h[i1] = a - b;
            }
            __syncthreads();
        }
        x = h[i] * 0.08838834764831845f;
        x = __bfloat162float(__float2bfloat16(x));
        // block absmax
        __shared__ float red[4];
        float a = fabsf(x);
        for (int off = 16; off > 0; off >>= 1) a = fmaxf(a, __shfl_down_sync(~0u, a, off));
        if ((i & 31) == 0) red[i >> 5] = a;
        __syncthreads();
        if (i < 4) {
            float v = red[i];
            for (int off = 2; off > 0; off >>= 1) v = fmaxf(v, __shfl_down_sync(0xfu, v, off));
            if (i == 0) red[0] = v;
        }
        __syncthreads();
        float absmax = fmaxf(red[0], 1e-4f);
        float scale = exp2f(ceilf(log2f(absmax * (1.f / 448.f))));
        float q = fminf(fmaxf(x / scale, -448.f), 448.f);
        // pool write (unit 256): pool g lands in page bt[g>>6], slot g%64 --
        // the page holding its own 4 tokens, so kern-serve retire/restore
        // of a shared prefix is safe (arch_decode.md section 1.5a).
        int pool = pos >> 2;
        int page_row = min(max(pool >> 6, 0), bt_cols - 1);
        long long page = bt[(size_t)row * bt_cols + page_row];
        int slot64 = pool & 63;
        constexpr long long IDX_PAGE_BYTES = 11 * 8448;  // all DSA layers in one physical page
        char* dst = idx_base + page * IDX_PAGE_BYTES + slot64 * 128;
        dst[i] = (char)__nv_cvt_float_to_fp8(q, __NV_SATFINITE, __NV_E4M3);
        if (i == 0)
            *reinterpret_cast<float*>(idx_base + page * IDX_PAGE_BYTES + 8192 + slot64 * 4) = scale;
    }
    if (pos_valid) {
        tail_k[phys * 128 + i] = __float2bfloat16(k_cur);
        tail_score[phys * 128 + i] = __float2bfloat16(s_cur);
    }
}

}  // extern "C"
