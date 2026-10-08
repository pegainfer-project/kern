// Sequence-ordered speculative DSA pool update. Arithmetic copied from the
// checked decode kernel; rows 1/3 and physical page pitch are explicit ABI.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
using bf16=__nv_bfloat16;
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


extern "C" __global__ void spec_kpool_update(char* __restrict__ idx_base, char* __restrict__ tail_base,
                                   const bf16* __restrict__ key, const bf16* __restrict__ score,
                                   const float* __restrict__ ape, const int* __restrict__ bt,
                                   const int* __restrict__ lines, const int* __restrict__ positions,
                                   const int* __restrict__ seq_lens, const int* __restrict__ valid,
                                   int bt_cols, int rows, long long page_bytes) {
    const int seq_id = blockIdx.x;
    for (int j=0; j<rows; ++j) {
    const int row = seq_id*rows+j;
    unsigned int i = threadIdx.x;              // one element per thread, dim = 128
    int pos = positions[row];
    int seq = seq_lens[row];
    bool pos_valid = (valid[row] != 0) && (pos >= 0) && (pos < seq);

    float k_cur = __bfloat162float(key[(size_t)row * 128 + i]);
    float s_cur = __bfloat162float(score[(size_t)row * 128 + i]);

    char* line = tail_base + (size_t)lines[seq_id] * 4096;
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
        long long page = bt[(size_t)seq_id * bt_cols + page_row];
        int slot64 = pool & 63;
        const long long IDX_PAGE_BYTES = page_bytes;  // all DSA layers in one physical page
        char* dst = idx_base + page * IDX_PAGE_BYTES + slot64 * 128;
        dst[i] = (char)__nv_cvt_float_to_fp8(q, __NV_SATFINITE, __NV_E4M3);
        if (i == 0)
            *reinterpret_cast<float*>(idx_base + page * IDX_PAGE_BYTES + 8192 + slot64 * 4) = scale;
    }
    if (pos_valid) {
        tail_k[phys * 128 + i] = __float2bfloat16(k_cur);
        tail_score[phys * 128 + i] = __float2bfloat16(s_cur);
    }
    __syncthreads(); // next row must see all prior ring writes
    }
}

// One CTA per QUERY row, not per sequence. Replicate the block table for the
// existing DeepGEMM ABI. FA3 cu_seqlens is [0,1,...,3S], not KDA's [0,3,...].
extern "C" __global__ void spec_dsa_prep(
    const int* positions, const int* bt, int* row_bt, int* lengths,
    int* pool, int* ctx, int* lens, int* slots, int* cu,
    int rows, int bt_cols, int st_cols) {
    int r=blockIdx.x, s=r/rows, n=positions[r]+1;
    const int* b=bt+(long long)s*bt_cols;
    for (int c=threadIdx.x;c<bt_cols;c+=blockDim.x) row_bt[(long long)r*bt_cols+c]=b[c];
    for (int p=threadIdx.x;p<n && p<st_cols;p+=blockDim.x) slots[(long long)r*st_cols+p]=b[p>>8]*256+(p&255);
    if (threadIdx.x==0) {
        lengths[r]=n; pool[r]=n/4; ctx[r]=max(n/4,1);
        lens[r]=min(n/4*4,2048)+(n&3); cu[r]=r;
        if (r==gridDim.x-1) cu[r+1]=r+1;
    }
}
extern "C" __global__ void spec_kv_store(
    char* kv, const bf16* key, const int* slot, const int* valid,
    long long pitch, long long layer_offset) {
    int r=blockIdx.x;
    if (!valid[r]) return;
    uint4* dst=reinterpret_cast<uint4*>(kv+(long long)slot[r]*pitch+layer_offset);
    const uint4* src=reinterpret_cast<const uint4*>(key+(long long)r*512);
    for (int c=threadIdx.x;c<64;c+=blockDim.x) dst[c]=src[c];
}
