// Decode GEMM on a tiled weight layout: C[M,N] = A[M,K] @ W[N,K]^T, M <= 16,
// bf16 in, f32 accumulate, bf16 out. Bit-identical to cuBLASLt's default
// algorithm for the shape (see the numerics contract below).
//
// W is the manifest `layout {"tile": [64, 64], "swizzle": 8}` image: each
// 64x64 tile is one contiguous 8 KB block, row-major inside with the
// 16-byte chunk c of tile row r at chunk c ^ (r & 7). Streaming whole tiles
// keeps DRAM reads sequential; the swizzle makes the tile, dropped into
// shared memory as is, conflict-free for ldmatrix. Row-major cuBLAS reads the
// same bytes as 64 strided 128-byte segments per tile and pays for it on
// this GPU (5.4 vs 6.4-7.3 TB/s on the big shapes).
//
// Numerics. cuBLASLt's non-split algorithms accumulate each output as one
// f32 chain over k in ascending k16 steps (tensor-core order); every tile
// choice gives the same bits. Its split-K algorithms cut K into `splits`
// chunks of kc = ceil(K/splits) rounded up to 64 (the last chunk shorter),
// accumulate each chunk the same way, then sum the f32 partials in
// ascending chunk order and round once to bf16. This kernel reproduces
// exactly that: one CTA per (64-row n-tile, chunk); with splits > 1 the
// partials go to `ws[split][16][N]` and the last CTA of an n-tile to arrive
// (a self-resetting counter, zero between launches) sums them 0..splits-1
// and writes C. The split count cuBLASLt picks depends on M, so the op
// carries two: `splits_lo` for M <= 8 and `splits_hi` for M > 8 (the shapes
// here change only at that boundary, except qkv at M = 9..15, which the
// decode_batch program never runs at in the measured workload). Rows of A
// past M are read (the buffer is sized at the var's max) and their outputs
// written, as cuBLAS does; nothing reads them.
//
// grid = [N/64, max(splits_lo, splits_hi)]: CTAs past the shape's split
// count for this M exit at once. block = 128 (4 warps x 16 rows).
// dynamic smem = STAGES * 10 KB + 1 KB (alignment).
#include <cuda_bf16.h>
#include <cstdint>

typedef unsigned short bf16;
constexpr int BN = 64, BK = 64, TILE_BYTES = BN * BK * 2, A_BYTES = 16 * BK * 2, STAGE_BYTES = TILE_BYTES + A_BYTES;
constexpr int NT = 128;

__device__ __forceinline__ void cp16(uint32_t s, const void* g) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(g));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;"); }
template <int N> __device__ __forceinline__ void wait_group() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }
__device__ __forceinline__ void ldsm4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(a));
}
__device__ __forceinline__ void mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <int STAGES>
__device__ __forceinline__ void gemm16_tiled(const bf16* __restrict__ A, const bf16* __restrict__ W,
                                             bf16* __restrict__ C, float* __restrict__ ws, int* __restrict__ counters,
                                             int M, int N, int K, int splits_lo, int splits_hi) {
    extern __shared__ uint8_t smem_raw[];
    const uint32_t sbase = (((uint32_t)__cvta_generic_to_shared(smem_raw)) + 1023) & ~1023u;
    __shared__ int s_last;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int KT = K / BK;
    const int S = M <= 8 ? splits_lo : splits_hi;
    const int kc = S == 1 ? K : ((K / S + 63) / 64) * 64;
    const int splits = (K + kc - 1) / kc;
    const int nt = blockIdx.x, sp = blockIdx.y;
    if (sp >= splits) return;
    const int k0 = sp * kc, k1 = min(K, k0 + kc), kt0 = k0 / BK, ktn = (k1 - k0) / BK;
    auto load = [&](int i, int slot) {
        const uint32_t sw = sbase + slot * STAGE_BYTES, sa = sw + TILE_BYTES;
        const bf16* wt = W + ((size_t)nt * KT + kt0 + i) * BN * BK;
#pragma unroll
        for (int c = 0; c < 4; c++) {
            const int q = tid + c * NT;
            cp16(sw + q * 16, wt + q * 8);
        }
        const int row = tid >> 3, ch = tid & 7;
        cp16(sa + row * 128 + ((ch ^ (row & 7)) << 4), A + (size_t)row * K + (kt0 + i) * BK + ch * 8);
    };
    float acc[2][4];
#pragma unroll
    for (int j = 0; j < 2; j++)
        for (int i = 0; i < 4; i++) acc[j][i] = 0.f;
#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < ktn) load(s, s);
        commit();
    }
    for (int i = 0; i < ktn; i++) {
        wait_group<STAGES - 2>();
        __syncthreads();
        {
            const int nk = i + STAGES - 1;
            if (nk < ktn) load(nk, nk % STAGES);
            commit();
        }
        const int slot = i % STAGES;
        const uint32_t sw = sbase + slot * STAGE_BYTES, sa = sw + TILE_BYTES;
#pragma unroll
        for (int kk = 0; kk < BK / 16; kk++) {
            uint32_t a[4];
            {
                const int row = lane & 15, ch = kk * 2 + (lane >> 4);
                ldsm4(a[0], a[1], a[2], a[3], sa + row * 128 + ((ch ^ (row & 7)) << 4));
            }
            uint32_t b[4];
            {
                const int mi = lane >> 3, i8 = lane & 7;
                const int row = warp * 16 + ((mi >> 1) << 3) + i8, ch = kk * 2 + (mi & 1);
                ldsm4(b[0], b[1], b[2], b[3], sw + row * 128 + ((ch ^ (row & 7)) << 4));
            }
            mma(acc[0], a, b[0], b[1]);
            mma(acc[1], a, b[2], b[3]);
        }
    }
    wait_group<0>();
    const int g = lane >> 2, c2 = (lane & 3) * 2;
    if (splits == 1) {
#pragma unroll
        for (int j = 0; j < 2; j++) {
            const int n = nt * BN + warp * 16 + j * 8 + c2;
            *reinterpret_cast<__nv_bfloat162*>(C + (size_t)g * N + n) = __floats2bfloat162_rn(acc[j][0], acc[j][1]);
            *reinterpret_cast<__nv_bfloat162*>(C + (size_t)(g + 8) * N + n) =
                __floats2bfloat162_rn(acc[j][2], acc[j][3]);
        }
        return;
    }
    float* P = ws + (size_t)sp * 16 * N;
#pragma unroll
    for (int j = 0; j < 2; j++) {
        const int n = nt * BN + warp * 16 + j * 8 + c2;
        *reinterpret_cast<float2*>(P + (size_t)g * N + n) = make_float2(acc[j][0], acc[j][1]);
        *reinterpret_cast<float2*>(P + (size_t)(g + 8) * N + n) = make_float2(acc[j][2], acc[j][3]);
    }
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        const int old = atomicAdd(counters + nt, 1);
        s_last = (old + 1 == splits);
    }
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    // 16 x 64 outputs: a thread owns 8 consecutive columns of one row; the
    // partials are read four chunks at a time so every load is in flight
    // before its add.
    const int m = tid >> 3, n = nt * BN + (tid & 7) * 8;
    const float* base = ws + (size_t)m * N + n;
    float4 v0 = __ldcg(reinterpret_cast<const float4*>(base)), v1 = __ldcg(reinterpret_cast<const float4*>(base + 4));
    for (int s0 = 1; s0 < splits; s0 += 4) {
        float4 x0[4], x1[4];
#pragma unroll
        for (int u = 0; u < 4; u++)
            if (s0 + u < splits) {
                x0[u] = __ldcg(reinterpret_cast<const float4*>(base + (size_t)(s0 + u) * 16 * N));
                x1[u] = __ldcg(reinterpret_cast<const float4*>(base + (size_t)(s0 + u) * 16 * N + 4));
            }
#pragma unroll
        for (int u = 0; u < 4; u++)
            if (s0 + u < splits) {
                v0.x += x0[u].x; v0.y += x0[u].y; v0.z += x0[u].z; v0.w += x0[u].w;
                v1.x += x1[u].x; v1.y += x1[u].y; v1.z += x1[u].z; v1.w += x1[u].w;
            }
    }
    bf16* c = C + (size_t)m * N + n;
    *reinterpret_cast<__nv_bfloat162*>(c) = __floats2bfloat162_rn(v0.x, v0.y);
    *reinterpret_cast<__nv_bfloat162*>(c + 2) = __floats2bfloat162_rn(v0.z, v0.w);
    *reinterpret_cast<__nv_bfloat162*>(c + 4) = __floats2bfloat162_rn(v1.x, v1.y);
    *reinterpret_cast<__nv_bfloat162*>(c + 6) = __floats2bfloat162_rn(v1.z, v1.w);
    if (tid == 0) counters[nt] = 0;
}

// One entry per pipeline depth; the manifest picks by shape (4 for the
// 544-CTA gate_up, 8 for qkvz, 6 elsewhere: measured in NOTES.md).
#define ENTRY(ST)                                                                                                   \
    extern "C" __global__ void __launch_bounds__(NT) kern_gemm16_tiled_s##ST##_bf16(                              \
        const bf16* A, const bf16* W, bf16* C, float* ws, int* counters, int M, int N, int K, int splits_lo,        \
        int splits_hi) {                                                                                            \
        gemm16_tiled<ST>(A, W, C, ws, counters, M, N, K, splits_lo, splits_hi);                                     \
    }
ENTRY(4)
ENTRY(6)
ENTRY(8)
