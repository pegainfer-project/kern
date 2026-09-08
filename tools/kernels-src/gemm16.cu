// Decode GEMM: C[M,N] = A[M,K] @ W[N,K]^T, M <= 16, bf16 in, f32
// accumulate, bf16 out, W row-major as the weights file has it.
// Bit-identical to cuBLASLt's default algorithm for the shape (see the
// numerics contract below).
//
// A CTA streams W as 64x64 tiles, each 64 segments of 128 bytes from
// consecutive rows, through a cp.async ring; the 16-byte chunk c of tile
// row r lands in shared memory at chunk c ^ (r & 7), the pattern a
// 128-byte-pitch ldmatrix reads without bank conflicts. (A copy of W
// stored as contiguous pre-swizzled tiles streams ~10% faster on this GPU
// but doubles the weight memory; measured in NOTES.md.)
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

// The body. A CTA owns 64 output columns (n-tile `nt`, rows of W) and the
// chunks [sp0, sp1): one chunk when the launch has a grid.y per chunk (the
// partials protocol below), every chunk when grid.y is 1 (the ascending
// running sum stays in registers, no workspace: the fastest form when the
// n-tile count alone fills the GPU). TR is the row-block height: 64, or 32
// for a narrow weight whose 64 rows are two 32-row blocks (the second may
// lie past N; its outputs are never written).
template <int STAGES, int TR>
__device__ __forceinline__ void gemm16(const bf16* __restrict__ A, const bf16* __restrict__ W,
                                             bf16* __restrict__ C, float* __restrict__ ws, int* __restrict__ counters,
                                             int M, int N, int K, int splits_lo, int splits_hi, int nt) {
    extern __shared__ uint8_t smem_raw[];
    const uint32_t sbase = (((uint32_t)__cvta_generic_to_shared(smem_raw)) + 1023) & ~1023u;
    __shared__ int s_last;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int S = M <= 8 ? splits_lo : splits_hi;
    const int kc = S == 1 ? K : ((K / S + 63) / 64) * 64;
    const int splits = (K + kc - 1) / kc;
    const bool whole = gridDim.y == 1;
    const int sp0 = whole ? 0 : (int)blockIdx.y, sp1 = whole ? splits : sp0 + 1;
    if (sp0 >= splits) return;
    const int kt0 = sp0 * kc / BK, ktn = min(K, sp1 * kc) / BK - kt0;
    // Chunk q of the tile: row q / 8 of W's 64 rows, 16 bytes at column
    // chunk q % 8; eight consecutive threads read one 128-byte row segment.
    auto load_w = [&](int i, int slot) {
        const uint32_t sw = sbase + slot * STAGE_BYTES;
        const bf16* wk = W + (size_t)(kt0 + i) * BK;
        if (TR == 64) {
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const int q = tid + c * NT, r = q >> 3, ch = q & 7;
                cp16(sw + r * 128 + ((ch ^ (r & 7)) << 4), wk + (size_t)(nt * BN + r) * K + ch * 8);
            }
        } else {
            // two 32-row blocks, row r of block h at smem row 32h + r
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int st = 2 * nt + h;
                if (st * TR >= N) break;
#pragma unroll
                for (int c = 0; c < 2; c++) {
                    const int q = tid + c * NT, r = q >> 3, ch = q & 7;
                    cp16(sw + h * (TR * BK * 2) + r * 128 + ((ch ^ (r & 7)) << 4),
                         wk + (size_t)(st * TR + r) * K + ch * 8);
                }
            }
        }
    };
    auto load_a = [&](int i, int slot) {
        const uint32_t sa = sbase + slot * STAGE_BYTES + TILE_BYTES;
        const int row = tid >> 3, ch = tid & 7;
        cp16(sa + row * 128 + ((ch ^ (row & 7)) << 4), A + (size_t)row * K + (kt0 + i) * BK + ch * 8);
    };
    const int g = lane >> 2, c2 = (lane & 3) * 2;
    float acc[2][4], tot[2][4];
#pragma unroll
    for (int j = 0; j < 2; j++)
        for (int i = 0; i < 4; i++) acc[j][i] = tot[j][i] = 0.f;
    // Programmatic dependent launch: the weights depend on no earlier
    // kernel, so the first stages' tiles are in flight before the wait;
    // A (the previous kernel's output) is fetched only after it. The
    // trigger lets the next launch stage itself while this one streams.
    // Groups: W0..W_{S-2}, then A0..A_{S-2}, then one W+A group per stage
    // of the loop, so before stage i at most S-2 groups may be pending.
#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < ktn) load_w(s, s);
        commit();
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < ktn) load_a(s, s);
        commit();
    }
    int sp = sp0, boundary = min(K, (sp + 1) * kc) / BK - kt0;  // stages of this CTA before chunk sp ends
    for (int i = 0; i < ktn; i++) {
        wait_group<STAGES - 2>();
        __syncthreads();
        {
            const int nk = i + STAGES - 1;
            if (nk < ktn) {
                load_w(nk, nk % STAGES);
                load_a(nk, nk % STAGES);
            }
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
        if (whole && i + 1 == boundary) {
            // chunk sp is complete: fold it into the running sum, ascending
#pragma unroll
            for (int j = 0; j < 2; j++)
                for (int q = 0; q < 4; q++) {
                    tot[j][q] = (sp == 0) ? acc[j][q] : tot[j][q] + acc[j][q];
                    acc[j][q] = 0.f;
                }
            sp++;
            boundary = min(K, (sp + 1) * kc) / BK - kt0;
        }
    }
    wait_group<0>();
    if (whole) {
#pragma unroll
        for (int j = 0; j < 2; j++) {
            const int n = nt * BN + warp * 16 + j * 8 + c2;
            if (n >= N) continue;
            *reinterpret_cast<__nv_bfloat162*>(C + (size_t)g * N + n) = __floats2bfloat162_rn(tot[j][0], tot[j][1]);
            *reinterpret_cast<__nv_bfloat162*>(C + (size_t)(g + 8) * N + n) =
                __floats2bfloat162_rn(tot[j][2], tot[j][3]);
        }
        return;
    }
    // one chunk per CTA: the partials protocol, cuBLASLt's order
    float* P = ws + (size_t)sp0 * 16 * N;
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

// One entry per pipeline depth; the manifest picks by shape (8 for qkvz,
// 6 elsewhere: measured in NOTES.md). grid = [N/64, chunks] with the
// partials protocol, or [N/64, 1] for every chunk in the CTA.
#define ENTRY(ST)                                                                                                   \
    extern "C" __global__ void __launch_bounds__(NT) kern_gemm16_s##ST##_bf16(                                    \
        const bf16* A, const bf16* W, bf16* C, float* ws, int* counters, int M, int N, int K, int splits_lo,        \
        int splits_hi) {                                                                                            \
        gemm16<ST, 64>(A, W, C, ws, counters, M, N, K, splits_lo, splits_hi, blockIdx.x);                          \
    }
ENTRY(4)
ENTRY(6)
ENTRY(8)

// Two projections of the same A in one launch: W1 (N1 rows, 64-row blocks,
// non-split) on CTAs [0, N1/64), then W2 (N2 rows, 32-row blocks, `splits2`
// chunks all in the CTA) on the next ceil(N2/64). grid = [N1/64 +
// ceil(N2/64), 1]; no workspace.
#define ENTRY_DUAL(ST)                                                                                              \
    extern "C" __global__ void __launch_bounds__(NT) kern_gemm16_dual_s##ST##_bf16(                               \
        const bf16* A, const bf16* W1, bf16* C1, const bf16* W2, bf16* C2, int M, int N1, int N2, int K,            \
        int splits2_lo, int splits2_hi) {                                                                           \
        const int nt1 = N1 / BN;                                                                                    \
        if ((int)blockIdx.x < nt1)                                                                                  \
            gemm16<ST, 64>(A, W1, C1, nullptr, nullptr, M, N1, K, 1, 1, blockIdx.x);                                \
        else                                                                                                        \
            gemm16<ST, 32>(A, W2, C2, nullptr, nullptr, M, N2, K, splits2_lo, splits2_hi, blockIdx.x - nt1);        \
    }
ENTRY_DUAL(8)

// ---- gate_up + silu_mul ------------------------------------------------
//
// act[m, i] = bf16(bf16(g / (1 + expf(-g))) * u) with g = C[m, i], u = C[m,
// d + i], C = A @ W^T the gate_up projection (N = 2d) and the same bf16
// rounding of C the plain GEMM writes: silu_mul.cu's numerics on the
// GEMM's own output. One CTA per 64 columns of act: warps 0-3 accumulate
// the gate tile (n-tile j), warps 4-7 the matching up tile (n-tile j + d/64);
// the up warps round to bf16 into shared memory and the gate warps finish.
// Non-split shapes only (gate_up is splitk 1 at every M). grid = [d/64, 1],
// block = 256, dynamic smem = STAGES * 18 KB + 1 KB.
constexpr int NT2 = 256, STAGE2 = 2 * TILE_BYTES + A_BYTES;

template <int STAGES>
__device__ __forceinline__ void gemm16_silu(const bf16* __restrict__ A, const bf16* __restrict__ W,
                                            bf16* __restrict__ act, int M, int N, int K) {
    extern __shared__ uint8_t smem_raw[];
    const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
    const uint32_t sbase = (sraw + 1023) & ~1023u;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int KT = K / BK, d = N / 2, ktn = KT;
    const int nt = blockIdx.x;                       // act columns [64 nt, 64 nt + 64)
    const int half = warp >> 2, wr = warp & 3;       // 0: gate rows, 1: up rows
    auto load_w = [&](int i, int slot) {
        const uint32_t sw = sbase + slot * STAGE2;
        const bf16* wk = W + (size_t)i * BK;
#pragma unroll
        for (int t = 0; t < 2; t++) {
            const int n0 = (nt + t * (d / BN)) * BN;
#pragma unroll
            for (int c = 0; c < 2; c++) {
                const int q = tid + c * NT2, r = q >> 3, ch = q & 7;
                cp16(sw + t * TILE_BYTES + r * 128 + ((ch ^ (r & 7)) << 4), wk + (size_t)(n0 + r) * K + ch * 8);
            }
        }
    };
    auto load_a = [&](int i, int slot) {
        if (tid < 128) {
            const uint32_t sa = sbase + slot * STAGE2 + 2 * TILE_BYTES;
            const int row = tid >> 3, ch = tid & 7;
            cp16(sa + row * 128 + ((ch ^ (row & 7)) << 4), A + (size_t)row * K + i * BK + ch * 8);
        }
    };
    float acc[2][4];
#pragma unroll
    for (int j = 0; j < 2; j++)
        for (int i = 0; i < 4; i++) acc[j][i] = 0.f;
#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < ktn) load_w(s, s);
        commit();
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < ktn) load_a(s, s);
        commit();
    }
    for (int i = 0; i < ktn; i++) {
        wait_group<STAGES - 2>();
        __syncthreads();
        {
            const int nk = i + STAGES - 1;
            if (nk < ktn) {
                load_w(nk, nk % STAGES);
                load_a(nk, nk % STAGES);
            }
            commit();
        }
        const int slot = i % STAGES;
        const uint32_t sw = sbase + slot * STAGE2 + half * TILE_BYTES, sa = sbase + slot * STAGE2 + 2 * TILE_BYTES;
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
                const int row = wr * 16 + ((mi >> 1) << 3) + i8, ch = kk * 2 + (mi & 1);
                ldsm4(b[0], b[1], b[2], b[3], sw + row * 128 + ((ch ^ (row & 7)) << 4));
            }
            mma(acc[0], a, b[0], b[1]);
            mma(acc[1], a, b[2], b[3]);
        }
    }
    wait_group<0>();
    __syncthreads();  // every warp is done with the ring; slot 0 becomes the exchange
    const int g = lane >> 2, c2 = (lane & 3) * 2;
    __nv_bfloat162* xch = reinterpret_cast<__nv_bfloat162*>(smem_raw + (sbase - sraw));  // [16][64] bf16
    if (half == 1) {
#pragma unroll
        for (int j = 0; j < 2; j++) {
            const int n = wr * 16 + j * 8 + c2;
            xch[(g * BN + n) / 2] = __floats2bfloat162_rn(acc[j][0], acc[j][1]);
            xch[((g + 8) * BN + n) / 2] = __floats2bfloat162_rn(acc[j][2], acc[j][3]);
        }
    }
    __syncthreads();
    if (half == 0) {
#pragma unroll
        for (int j = 0; j < 2; j++) {
            const int n = wr * 16 + j * 8 + c2;
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int row = g + 8 * h;
                const __nv_bfloat162 gg = __floats2bfloat162_rn(acc[j][2 * h], acc[j][2 * h + 1]);
                const float gx = __bfloat162float(gg.x), gy = __bfloat162float(gg.y);
                const __nv_bfloat162 s = __floats2bfloat162_rn(gx / (1.0f + expf(-gx)), gy / (1.0f + expf(-gy)));
                const __nv_bfloat162 u = xch[(row * BN + n) / 2];
                *reinterpret_cast<__nv_bfloat162*>(act + (size_t)row * d + nt * BN + n) = __hmul2(s, u);
            }
        }
    }
}

#define ENTRY_SILU(ST)                                                                                             \
    extern "C" __global__ void __launch_bounds__(NT2) kern_gemm16_silu_s##ST##_bf16(                              \
        const bf16* A, const bf16* W, bf16* act, int M, int N, int K) {                                            \
        gemm16_silu<ST>(A, W, act, M, N, K);                                                                       \
    }
ENTRY_SILU(4)
