// K3 decode, family K1: the residual stream (attnres + rms + landing adds).
//
// Entries (exact ABI of docs/k3-kernel-abi.md section K1):
//
//   [K1a] mixed = attnres(blocks, prefix, nb); if (snapshot) blocks[b, nb] = prefix;
//         normed = rms(mixed, gamma)
//     extern "C" __global__ void kern_k3_attnres_rms(
//         const bf16* prefix, bf16* blocks, const f32* sw, const bf16* gamma,
//         bf16* normed, int nb, int snapshot, int B);
//
//   [K1b] p = bf16(partial[b, :H]); prefix2 = snapshot ? p : bf16(prefix + p);
//         mixed = attnres(blocks, prefix2, nb); normed = rms(mixed, gamma)   (no snapshot write)
//     extern "C" __global__ void kern_k3_land_add_attnres_rms(
//         const f32* partial, const bf16* prefix, const bf16* blocks, const f32* sw,
//         const bf16* gamma, bf16* prefix2, bf16* normed, int nb, int snapshot, int B);
//
//   [K1c] hidden = bf16( prefix2 + bf16(p1[b, :H]) + (two ? bf16(p2[b, :H]) : 0) )
//     extern "C" __global__ void kern_k3_land_add2(
//         const f32* p1, const f32* p2, const bf16* prefix2, bf16* hidden, int two, int B);
//
// Launch geometry (all three entries -- this is what the manifest should copy):
//     grid  = (B, 1, 1)                     one row per block, B is a runtime variable
//     block = (1024, 1, 1)
//     smem  = 0                             static shared memory only (2504 B)
//
// Tiling.  H = 7168 bf16 = 896 sixteen-byte vectors; thread t owns vector t (elements
// 8t .. 8t+8).  Threads 0..895 -- warps 0..27 exactly, so `act` is warp-uniform and
// becomes a branch rather than per-instruction predication -- do every load and store as
// a fully coalesced 16 B access, and warps 28..31 contribute zeros to the reductions.
// Every reduction is a fixed-order __shfl_xor butterfly over a fixed-order 32-slot warp
// layout, so a result never depends on the schedule.
//
// Algorithm per row (K1a/K1b, nb > 0):
//   pass 1  one load per candidate (nb snapshot rows + the prefix, which is already in
//           registers), accumulating both sum(x^2) and sum(x*sw) from that single load;
//           sw stays in 8 registers and is reused by every candidate.  The score is
//           factored as  score_c = rsqrt(mean(x^2) + eps) * sum(x*sw)  instead of scaling
//           each element first -- the same value up to f32 rounding, and it removes a
//           whole pass over the row.  The candidate loop has a compile-time trip count
//           (NB_MAX) and runs two loads deep.
//   combine one barrier, then warp k reduces the 32 warp partials of value k (2*(nb+1)
//           values, so up to 18 warps work in parallel); a second barrier publishes them.
//   softmax every warp recomputes it from the 18 sums, so lane c ends up holding p_c and
//           pass 2 picks p_c up with a shuffle broadcast (no smem round trip).
//   pass 2  re-read the candidates (they are in L1/L2 from pass 1) and mix in f32.
//   rms     one more block reduction over the landed mixed row, then * gamma.
//   nb == 0 skips all of it: mixed = prefix, since bf16(1.0f * f32(prefix)) == prefix.
//
// Caching all nb+1 candidate rows in dynamic shared memory (up to 129024 B) to make
// pass 2 read from smem was implemented and measured: it is worth nothing, because the
// kernel is instruction-issue bound rather than L1 bound, and LDS.128 costs the same
// issue slot as the LDG.128 it replaces.  It was removed; smem = 0 is the ABI.
//
// Landing points (f32 -> bf16):
//   K1a/K1b: mixed = bf16(sum_c p_c * f32(cand_c));  rms: bf16(f32(mixed) * rsqrt) then
//            * gamma as bf16 x bf16 -> bf16.  Scores/softmax stay f32.
//   K1b:     p = bf16(partial) first, then prefix2 = bf16(f32(prefix) + f32(p)).
//   K1c:     bf16(p1), bf16(p2) first, sum with f32(prefix2) in f32, one final round.

#include <cuda_bf16.h>

typedef __nv_bfloat16  bf16_t;
typedef __nv_bfloat162 bf162_t;

#define KH        7168
#define KNB_MAX   8
#define KVEC      (KH / 8)        /* 896 sixteen-byte vectors per row */
#define KEPS      1e-5f
#define KTHREADS  1024

// 8 bf16 = one 16-byte vector, held as four packed-pair words (trivially copyable,
// so it stays in registers).
struct V8 { unsigned w[4]; };

__device__ __forceinline__ V8 ldv(const void* p) {
    const uint4 t = *(const uint4*)p;
    V8 v; v.w[0] = t.x; v.w[1] = t.y; v.w[2] = t.z; v.w[3] = t.w;
    return v;
}
__device__ __forceinline__ void stv(void* p, const V8& v) {
    *(uint4*)p = make_uint4(v.w[0], v.w[1], v.w[2], v.w[3]);
}
__device__ __forceinline__ bf162_t as_bf162(unsigned u) {
    return __halves2bfloat162(__ushort_as_bfloat16((unsigned short)(u & 0xffffu)),
                              __ushort_as_bfloat16((unsigned short)(u >> 16)));
}
__device__ __forceinline__ unsigned from_bf162(bf162_t h) {
    return (unsigned)(unsigned short)__bfloat16_as_ushort(__low2bfloat16(h)) |
           ((unsigned)(unsigned short)__bfloat16_as_ushort(__high2bfloat16(h)) << 16);
}
__device__ __forceinline__ float2 bf2f(unsigned u)  { return __bfloat1622float2(as_bf162(u)); }
__device__ __forceinline__ unsigned f2bf(float2 f)  { return from_bf162(__float22bfloat162_rn(f)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// mixed = attnres(candidates 0..nb-1 from blk_row, candidate nb = pv held in registers)
// then normed_row = rms(mixed, gamma).  Called with the whole block; t = threadIdx.x.
__device__ __forceinline__ void attnres_rms_row(
    const bf16_t* __restrict__ blk_row,   // &blocks[b * NB_MAX * H]  (unread when nb == 0)
    const V8                   pv,        // this thread's 8 elements of the prefix candidate
    const float*  __restrict__ sw,        // [H]   (unread when nb == 0)
    const bf16_t* __restrict__ gamma,     // [H]
    bf16_t*       __restrict__ normed_row,// &normed[b * H]
    int nb, int t)
{
    const int  lane = t & 31, warp = t >> 5;
    const bool act  = (t < KVEC);          // warp-uniform: KVEC = 896 = 28 whole warps

    __shared__ float s_red[(KNB_MAX + 1) * 2 * 32];   // [value][warp] partials
    __shared__ float s_val[(KNB_MAX + 1) * 2];        // [value] block sums
    __shared__ float s_red2[32];

    const int ncand = nb + 1;
    V8 mixed;

    if (nb == 0) {
        mixed = pv;                       // bf16(1.0f * f32(prefix)) == prefix
    } else {
        float swv[8];
        if (act) {
            const float4* sp4 = (const float4*)(sw + t * 8);
            const float4 a = sp4[0], b = sp4[1];
            swv[0] = a.x; swv[1] = a.y; swv[2] = a.z; swv[3] = a.w;
            swv[4] = b.x; swv[5] = b.y; swv[6] = b.z; swv[7] = b.w;
        } else {
#pragma unroll
            for (int j = 0; j < 8; ++j) swv[j] = 0.f;
        }

        // pass 1: (sum x^2, sum x*sw) per candidate.  The snapshot candidates run in a
        // loop with a compile-time trip count (NB_MAX), two loads deep so a candidate's
        // LDG is in flight while the previous one reduces; the prefix candidate -- already
        // in registers -- is done separately, so no iteration carries a select between
        // "load" and "use pv".  `act` is warp-uniform, so it is a branch, not
        // per-instruction predication.
#pragma unroll
        for (int c0 = 0; c0 < KNB_MAX; c0 += 2) {
            V8 xx[2];
#pragma unroll
            for (int g = 0; g < 2; ++g) if (act && c0 + g < nb) xx[g] = ldv(blk_row + (size_t)(c0 + g) * KH + t * 8);
#pragma unroll
            for (int g = 0; g < 2; ++g) {
                const int c = c0 + g;
                if (c >= nb) continue;
                float sq = 0.f, dp = 0.f;
                if (act) {
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float2 f = bf2f(xx[g].w[j]);
                        sq += f.x * f.x;  sq += f.y * f.y;
                        dp += f.x * swv[2 * j];  dp += f.y * swv[2 * j + 1];
                    }
                }
                sq = warp_sum(sq);
                dp = warp_sum(dp);
                if (lane == 0) { s_red[(c * 2) * 32 + warp] = sq; s_red[(c * 2 + 1) * 32 + warp] = dp; }
            }
        }
        {   // candidate nb = the prefix, straight out of registers
            float sq = 0.f, dp = 0.f;
            if (act) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 f = bf2f(pv.w[j]);
                    sq += f.x * f.x;  sq += f.y * f.y;
                    dp += f.x * swv[2 * j];  dp += f.y * swv[2 * j + 1];
                }
            }
            sq = warp_sum(sq);
            dp = warp_sum(dp);
            if (lane == 0) { s_red[(nb * 2) * 32 + warp] = sq; s_red[(nb * 2 + 1) * 32 + warp] = dp; }
        }
        __syncthreads();

        // Combine the 32 warp partials of all 2*ncand values in parallel: warp k owns
        // value k (one 32-lane butterfly), rather than one warp walking 18 x 32 slots.
        if (warp < 2 * ncand) {
            const float v = warp_sum(s_red[warp * 32 + lane]);
            if (lane == 0) s_val[warp] = v;
        }
        __syncthreads();

        // scores + softmax, recomputed identically by every warp: lane c ends up holding
        // p_c, which pass 2 picks up with a shuffle broadcast (no smem round trip).
        // score_c = rsqrt(mean(x^2) + eps) * sum(x*sw): the rms_nw scalar is factored out
        // of the dot product, which is the same value up to f32 rounding.
        float sc = -3.0e38f;
        if (lane < ncand) {
            const float sqv = s_val[2 * lane], dpv = s_val[2 * lane + 1];
            sc = dpv * rsqrtf(sqv * (1.0f / (float)KH) + KEPS);
        }
        float mx = sc;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
        const float ex = (lane < ncand) ? __expf(sc - mx) : 0.f;
        float den = ex;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) den += __shfl_xor_sync(0xffffffffu, den, o);
        const float pmine = ex / den;                 // lane c: p_c

        // pass 2: mix (f32 accumulate, one bf16 landing).  Same split as pass 1.
        float acc[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[j] = 0.f;
        if (act) {
#pragma unroll
            for (int c = 0; c < KNB_MAX; ++c) {
                if (c >= nb) continue;
                const V8 x = ldv(blk_row + (size_t)c * KH + t * 8);
                const float p = __shfl_sync(0xffffffffu, pmine, c);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 f = bf2f(x.w[j]);
                    acc[2 * j]     += p * f.x;
                    acc[2 * j + 1] += p * f.y;
                }
            }
            const float p = __shfl_sync(0xffffffffu, pmine, nb);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 f = bf2f(pv.w[j]);
                acc[2 * j]     += p * f.x;
                acc[2 * j + 1] += p * f.y;
            }
        }
#pragma unroll
        for (int j = 0; j < 4; ++j)
            mixed.w[j] = f2bf(make_float2(acc[2 * j], acc[2 * j + 1]));
    }

    // rms(mixed, gamma)
    float sq = 0.f;
    if (act) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = bf2f(mixed.w[j]);
            sq += f.x * f.x;  sq += f.y * f.y;
        }
    }
    sq = warp_sum(sq);
    if (lane == 0) s_red2[warp] = sq;
    __syncthreads();
    const float tot = warp_sum(s_red2[lane]);   // every warp reduces the same 32 partials
    const float r = rsqrtf(tot * (1.0f / (float)KH) + KEPS);

    if (act) {
        const V8 g = ldv(gamma + t * 8);
        V8 o;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 f = bf2f(mixed.w[j]);
            f.x *= r;  f.y *= r;
            o.w[j] = from_bf162(__hmul2(__float22bfloat162_rn(f), as_bf162(g.w[j])));
        }
        stv(normed_row + t * 8, o);
    }
}

// ---------------------------------------------------------------- K1a
extern "C" __global__ void __launch_bounds__(KTHREADS, 1) kern_k3_attnres_rms(
    const bf16_t* __restrict__ prefix,   // [B, H]
    bf16_t*       __restrict__ blocks,   // [B, NB_MAX, H]
    const float*  __restrict__ sw,       // [H]
    const bf16_t* __restrict__ gamma,    // [H]
    bf16_t*       __restrict__ normed,   // [B, H]
    int nb, int snapshot, int B)
{
    const int b = blockIdx.x;
    if (b >= B) return;
    const int t = threadIdx.x;
    const bool act = (t < KVEC);

    V8 pv;
    pv.w[0] = 0u; pv.w[1] = 0u; pv.w[2] = 0u; pv.w[3] = 0u;
    if (act) pv = ldv(prefix + (size_t)b * KH + t * 8);

    if (snapshot && nb < KNB_MAX && act)
        stv(blocks + ((size_t)b * KNB_MAX + nb) * KH + t * 8, pv);

    attnres_rms_row(blocks + (size_t)b * KNB_MAX * KH, pv, sw, gamma,
                    normed + (size_t)b * KH, nb, t);
}

// ---------------------------------------------------------------- K1b
// -DLAND_BF16: the landing arrives as bf16 (a reduce-scatter's sum of the
// tray's o_proj partials), which is the value the f32 form rounds to first.
#ifdef LAND_BF16
typedef bf16_t land_t;
#else
typedef float land_t;
#endif

extern "C" __global__ void __launch_bounds__(KTHREADS, 1) kern_k3_land_add_attnres_rms(
    const land_t* __restrict__ partial,  // [B, H]  partial of o_proj
    const bf16_t* __restrict__ prefix,   // [B, H]
    const bf16_t* __restrict__ blocks,   // [B, NB_MAX, H]
    const float*  __restrict__ sw,       // [H]
    const bf16_t* __restrict__ gamma,    // [H]
    bf16_t*       __restrict__ prefix2,  // [B, H]
    bf16_t*       __restrict__ normed,   // [B, H]
    int nb, int snapshot, int B)
{
    const int b = blockIdx.x;
    if (b >= B) return;
    const int t = threadIdx.x;
    const bool act = (t < KVEC);

    V8 pv;
    pv.w[0] = 0u; pv.w[1] = 0u; pv.w[2] = 0u; pv.w[3] = 0u;
    if (act) {
#ifdef LAND_BF16
        const V8 lp = ldv(partial + (size_t)b * KH + t * 8);
        float pf[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = bf2f(lp.w[j]);
            pf[2 * j] = f.x; pf[2 * j + 1] = f.y;
        }
#else
        const float4* q = (const float4*)(partial + (size_t)b * KH + t * 8);
        const float4 a = q[0], c = q[1];
        float pf[8] = { a.x, a.y, a.z, a.w, c.x, c.y, c.z, c.w };
#endif
        if (snapshot) {
#pragma unroll
            for (int j = 0; j < 4; ++j)
                pv.w[j] = f2bf(make_float2(pf[2 * j], pf[2 * j + 1]));
        } else {
            const V8 pr = ldv(prefix + (size_t)b * KH + t * 8);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 rf = bf2f(pr.w[j]);
                const float2 lf = bf2f(f2bf(make_float2(pf[2 * j], pf[2 * j + 1])));
                pv.w[j] = f2bf(make_float2(rf.x + lf.x, rf.y + lf.y));
            }
        }
        stv(prefix2 + (size_t)b * KH + t * 8, pv);
    }

    attnres_rms_row(blocks + (size_t)b * KNB_MAX * KH, pv, sw, gamma,
                    normed + (size_t)b * KH, nb, t);
}

// ---------------------------------------------------------------- K1c
// grid (B,1,1), block 1024, smem 0.  hidden is written in full.
extern "C" __global__ void __launch_bounds__(KTHREADS, 1) kern_k3_land_add2(
    const float*  __restrict__ p1,       // [B, H]
    const float*  __restrict__ p2,       // [B, H]  read only when two != 0
    const bf16_t* __restrict__ prefix2,  // [B, H]
    bf16_t*       __restrict__ hidden,   // [B, H]
    int two, int B)
{
    const int b = blockIdx.x;
    if (b >= B) return;
    const int t = threadIdx.x;
    if (t >= KVEC) return;

    const size_t off = (size_t)b * KH + t * 8;
    const float4* q1 = (const float4*)(p1 + off);
    const float4 a1 = q1[0], b1 = q1[1];
    const float f1[8] = { a1.x, a1.y, a1.z, a1.w, b1.x, b1.y, b1.z, b1.w };
    float f2[8];
    if (two) {
        const float4* q2 = (const float4*)(p2 + off);
        const float4 a2 = q2[0], b2 = q2[1];
        f2[0] = a2.x; f2[1] = a2.y; f2[2] = a2.z; f2[3] = a2.w;
        f2[4] = b2.x; f2[5] = b2.y; f2[6] = b2.z; f2[7] = b2.w;
    } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) f2[j] = 0.f;
    }
    const V8 pr = ldv(prefix2 + off);

    V8 o;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 rf = bf2f(pr.w[j]);
        const float2 l1 = bf2f(f2bf(make_float2(f1[2 * j], f1[2 * j + 1])));
        float x = rf.x + l1.x, y = rf.y + l1.y;
        if (two) {
            const float2 l2 = bf2f(f2bf(make_float2(f2[2 * j], f2[2 * j + 1])));
            x += l2.x;  y += l2.y;
        }
        o.w[j] = f2bf(make_float2(x, y));
    }
    stv(hidden + off, o);
}
