// GLM-5.3 bring-up kernels: small glue ops the mined recipe does not cover.
// Build: nvcc -cubin -arch=sm_90a (tools/glm53/build_kernels.sh).
#include <cuda_bf16.h>

using bf16 = __nv_bfloat16;

// ---- debug probes (value parity) -------------------------------------------
// One CTA reduces sum / sumsq / min / max over n elems and prints one line
// from rank 0; the same stats land in out[tag*8..tag*8+7] = {sum, sumsq,
// min, max, x0..x3} so a host can read them back. `t` is a per-process
// monotonic launch counter so lines can be ordered into steps.
__device__ unsigned int glm53_probe_ctr = 0;

namespace {

__device__ inline float to_f32(float v) { return v; }
__device__ inline float to_f32(bf16 v) { return __bfloat162float(v); }

template <typename T>
__device__ void probe_body(const T* __restrict__ x, float* __restrict__ out,
                           unsigned int n, unsigned int tag, int rank_) {
    double s = 0.0, ss = 0.0;
    float mn = __int_as_float(0x7f7fffff), mx = -__int_as_float(0x7f7fffff);
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = to_f32(x[i]);
        s += (double)v;
        ss += (double)v * (double)v;
        mn = fminf(mn, v);
        mx = fmaxf(mx, v);
    }
    for (int off = 16; off > 0; off >>= 1) {
        s += __shfl_down_sync(~0u, s, off);
        ss += __shfl_down_sync(~0u, ss, off);
        mn = fminf(mn, __shfl_down_sync(~0u, mn, off));
        mx = fmaxf(mx, __shfl_down_sync(~0u, mx, off));
    }
    __shared__ double rs[32], rq[32];
    __shared__ float rn[32], rx[32];
    if ((threadIdx.x & 31) == 0) {
        unsigned int w = threadIdx.x >> 5;
        rs[w] = s; rq[w] = ss; rn[w] = mn; rx[w] = mx;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned int nw = blockDim.x >> 5;
        for (unsigned int r = 1; r < nw; ++r) {
            rs[0] += rs[r]; rq[0] += rq[r];
            rn[0] = fminf(rn[0], rn[r]); rx[0] = fmaxf(rx[0], rx[r]);
        }
        float f0 = n > 0 ? to_f32(x[0]) : 0.f;
        float f1 = n > 1 ? to_f32(x[1]) : 0.f;
        float f2 = n > 2 ? to_f32(x[2]) : 0.f;
        float f3 = n > 3 ? to_f32(x[3]) : 0.f;
        float* o = out + (size_t)tag * 8;
        o[0] = (float)rs[0]; o[1] = (float)rq[0]; o[2] = rn[0]; o[3] = rx[0];
        o[4] = f0; o[5] = f1; o[6] = f2; o[7] = f3;
        if (rank_ == 0) {
            unsigned int t = atomicAdd(&glm53_probe_ctr, 1u);
            printf("PROBE t=%u tag=%u n=%u sum=%.9e ssq=%.9e min=%.6e max=%.6e f4=%.6e,%.6e,%.6e,%.6e\n",
                   t, tag, n, rs[0], rq[0], (double)rn[0], (double)rx[0],
                   (double)f0, (double)f1, (double)f2, (double)f3);
        }
    }
}

}  // namespace

extern "C" {

__global__ void glm53_probe_bf16(const bf16* __restrict__ x, float* __restrict__ out,
                                 unsigned int n, unsigned int tag, int rank_) {
    probe_body(x, out, n, tag, rank_);
}

__global__ void glm53_probe_f32(const float* __restrict__ x, float* __restrict__ out,
                                unsigned int n, unsigned int tag, int rank_) {
    probe_body(x, out, n, tag, rank_);
}

// dst[i] = (float)src[i]; grid-stride, any n.
__global__ void glm53_cast_bf16_f32(const bf16* __restrict__ src, float* __restrict__ dst, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride) dst[i] = __bfloat162float(src[i]);
}

// NCCL concatenates [rank, batch, vocab_shard]. Convert directly to
// FP32 [batch, rank, vocab_shard] for head_argmax, without a BF16 temporary.
__global__ void glm53_gathered_logits_f32(const bf16* __restrict__ src, float* __restrict__ dst,
                                         unsigned int batch, unsigned int shard, unsigned int ranks) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t vocab = (size_t)ranks * shard;
    for (; i < (size_t)batch * vocab; i += stride) {
        size_t row = i / vocab, col = i % vocab;
        size_t rank = col / shard, lane = col % shard;
        dst[i] = __bfloat162float(src[(rank * batch + row) * shard + lane]);
    }
}

// dst[s] = emb[ids[s]]; one CTA per row.
__global__ void glm53_embedding(const long long* __restrict__ ids, const bf16* __restrict__ emb,
                                bf16* __restrict__ out, unsigned int d) {
    unsigned int s = blockIdx.x;
    const bf16* src = emb + (size_t)ids[s] * d;
    bf16* dst = out + (size_t)s * d;
    for (unsigned int i = threadIdx.x; i < d; i += blockDim.x) dst[i] = src[i];
}

// dst[s, j, :] = src[s, :] for j in 0..gridDim.y (the mHC 1->4 stream expand).
__global__ void glm53_hc_expand(const bf16* __restrict__ src, bf16* __restrict__ dst, unsigned int hdim) {
    unsigned int s = blockIdx.x, j = blockIdx.y;
    const bf16* in = src + (size_t)s * hdim;
    bf16* out = dst + ((size_t)s * gridDim.y + j) * hdim;
    for (unsigned int h = threadIdx.x; h < hdim; h += blockDim.x) out[h] = in[h];
}

// out[s, h] = mean_j res[s, j, h]; f32 accumulation (the final mHC contract).
__global__ void glm53_hc_contract(const bf16* __restrict__ res, bf16* __restrict__ out,
                                  unsigned int hdim, unsigned int n) {
    unsigned int s = blockIdx.x;
    const bf16* in = res + (size_t)s * n * hdim;
    bf16* o = out + (size_t)s * hdim;
    for (unsigned int h = threadIdx.x; h < hdim; h += blockDim.x) {
        float acc = 0.f;
        for (unsigned int j = 0; j < n; ++j) acc += __bfloat162float(in[(size_t)j * hdim + h]);
        o[h] = __float2bfloat16(acc / (float)n);
    }
}

// mul partials [64, SMAX, 24] -> [64, S, 24] and sqrsum [64, SMAX] -> [64, S].
// grid [64, 2], block 384: y=0 copies mul rows (t < S*24), y=1 sqrsum (t < S).
__global__ void glm53_hc_repack(const float* __restrict__ sm, float* __restrict__ dm,
                                const float* __restrict__ ss, float* __restrict__ ds,
                                unsigned int s_, unsigned int smax) {
    unsigned int split = blockIdx.x, t = threadIdx.x;
    if (blockIdx.y == 0) {
        if (t < s_ * 24) {
            unsigned int i = t / 24, j = t % 24;
            dm[((size_t)split * s_ + i) * 24 + j] = sm[((size_t)split * smax + i) * 24 + j];
        }
    } else {
        if (t < s_) ds[(size_t)split * s_ + t] = ss[(size_t)split * smax + t];
    }
}

// out[s] = x[s] * rsqrt(mean(x[s]^2) + eps) * w; one CTA per row, 1024 threads.
// D9: row pitches are explicit (q_a/kv_a are column slices of the fused
// [S,2048] qkv_a output: d < in_pitch there).
__global__ void glm53_rms_norm(const bf16* __restrict__ x, const bf16* __restrict__ w,
                               bf16* __restrict__ out, float eps, unsigned int d,
                               unsigned int in_pitch, unsigned int out_pitch) {
    unsigned int s = blockIdx.x;
    const bf16* xr = x + (size_t)s * in_pitch;
    bf16* o = out + (size_t)s * out_pitch;
    __shared__ float red[32];
    float acc = 0.f;
    for (unsigned int i = threadIdx.x; i < d; i += blockDim.x) {
        float v = __bfloat162float(xr[i]);
        acc += v * v;
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(~0u, acc, off);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < (blockDim.x >> 5)) ? red[threadIdx.x] : 0.f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(~0u, v, off);
        if (threadIdx.x == 0) red[0] = rsqrtf(v / (float)d + eps);
    }
    __syncthreads();
    float scale = red[0];
    for (unsigned int i = threadIdx.x; i < d; i += blockDim.x)
        o[i] = __float2bfloat16(__bfloat162float(xr[i]) * scale * __bfloat162float(w[i]));
}

// out[s] = argmax_j x[s, j]; first index on ties (torch ArgMaxOps semantics).
__global__ void glm53_argmax_f32(const float* __restrict__ x, long long* __restrict__ out, unsigned int n) {
    unsigned int s = blockIdx.x;
    const float* xr = x + (size_t)s * n;
    float best = -__int_as_float(0x7f7fffff);
    unsigned int bidx = 0;
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = xr[i];
        if (v > best || (v == best && i < bidx)) { best = v; bidx = i; }
    }
    __shared__ float sm[32];
    __shared__ unsigned int si[32];
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_down_sync(~0u, best, off);
        unsigned int oi = __shfl_down_sync(~0u, bidx, off);
        if (ov > best || (ov == best && oi < bidx)) { best = ov; bidx = oi; }
    }
    if ((threadIdx.x & 31) == 0) { sm[threadIdx.x >> 5] = best; si[threadIdx.x >> 5] = bidx; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float bv = -__int_as_float(0x7f7fffff);
        unsigned int bi = 0;
        unsigned int nw = blockDim.x >> 5;
        for (unsigned int r = 0; r < nw; ++r) {
            if (sm[r] > bv || (sm[r] == bv && si[r] < bi)) { bv = sm[r]; bi = si[r]; }
        }
        out[s] = (long long)bi;
    }
}

// D2: moe_align for numel <= 144 (the mined _moe_align_small_numel cubins
// bake NP = 16 / 32, silently dropping entries at S >= 4). One CTA.
// Reproduces the reference contract (moe_align_small_numel.py):
//   "+1 offset" buckets: expert id -1..288 -> bucket 0..289 (num_experts = 290);
//   every bucket padded to block_size, offsets in bucket order;
//   pad slots inside [0, total) hold `numel`; expert_ids per 64-block =
//   bucket - 1; intra-bucket pair order is free (fused_moe rows independent,
//   sum_reduce scatters by pair index token*9 + k).
__global__ void glm53_moe_align(const int* __restrict__ topk_ids,
                                int* __restrict__ sorted_token_ids,
                                int* __restrict__ expert_ids,
                                int* __restrict__ num_tokens_post_pad,
                                int num_experts, int block_size, int numel) {
    __shared__ int cnt[296];
    __shared__ int excl[296];
    __shared__ int cursor[296];
    __shared__ int total_s;
    const int tid = threadIdx.x;
    for (int b = tid; b < num_experts; b += blockDim.x) cnt[b] = 0;
    __syncthreads();
    for (int p = tid; p < numel; p += blockDim.x)
        atomicAdd(&cnt[topk_ids[p] + 1], 1);
    __syncthreads();
    if (tid == 0) {
        int running = 0;
        for (int b = 0; b < num_experts; ++b) {
            excl[b] = running;
            running += (cnt[b] + block_size - 1) / block_size * block_size;
        }
        total_s = running;
        *num_tokens_post_pad = running;
    }
    __syncthreads();
    const int total = total_s;
    for (int b = tid; b < num_experts; b += blockDim.x) {
        cursor[b] = excl[b];
        int lo = excl[b] / block_size;
        int hi = (excl[b] + (cnt[b] + block_size - 1) / block_size * block_size) / block_size;
        for (int m = lo; m < hi; ++m) expert_ids[m] = b - 1;
    }
    for (int t = tid; t < total; t += blockDim.x) sorted_token_ids[t] = numel;
    __syncthreads();
    for (int p = tid; p < numel; p += blockDim.x) {
        int dst = atomicAdd(&cursor[topk_ids[p] + 1], 1);
        sorted_token_ids[dst] = p;
    }
}

}  // extern "C"
