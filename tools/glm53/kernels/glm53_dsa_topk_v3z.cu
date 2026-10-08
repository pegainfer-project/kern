// glm53_dsa_topk_v3z.cu — cluster z-split deterministic top-512 (DSA kpool).
//
// BYTE-EXACT vs glm53_dsa_topk_v2 (pinned cubin sha 3bc73eed...). The output
// is pure int32; it depends on the fp32 logits only through the integer-exact
// cutoff (prefix,kleft) and the stable ascending-pool-id tie ranking. Every
// cross-CTA value in this kernel is a u32 integer sum (order-free exact) or a
// fixed-order integer prefix. Same score_key, same radix cutoff rule, same
// 11-bit packed tie accounting, same ascending-pool emission as v2. All scans
// are hand-rolled warp shuffles over packed u32 (eq in bits 0..10, gt at bit
// 11+; per-1024-block sums cannot carry: eq<=1024, gt<=1024). No CUB.
//
// Design (memory glm53-zsplit-dsmem-design-20260928.md, Tier 1):
//   grid (rows, 8), cluster (1,8,1), block 256. Row r's 8 CTAs own contiguous
//   1024-pool blocks. Radix pass p: local 256-bin histogram -> remote st into
//   rank0's all_hist[p&1][z] -> cluster barrier -> every CTA reduces the 8
//   histograms and redundantly runs the identical digit scan. Parity double
//   buffering removes the WAR hazard between passes with one barrier/pass.
//   Selection: per-block packed (gt|eq) totals -> rank0 gather -> global
//   block-order prefix -> per-CTA local scans emit at computed global ranks.
//   No atomic position allocation (same rule as v2).
// Identity path (len<=512) and tail append are per-column pure functions.
// PDL discipline identical to v2 (single edge). Co-residency: intra-cluster
// sync only; clusters drain in waves (no grid-wide co-residency, no global
// barrier). Cluster launches are CUDA-graph-capturable.
//
// Build: nvcc -cubin -arch=sm_90a -std=c++17
//   -O3 -ccbin /usr/bin/g++-14 glm53_dsa_topk_v3z.cu -o glm53_dsa_topk_v3z.cubin
#include <cuda_runtime.h>

#define V3Z_Z 8

__device__ __forceinline__ unsigned v3z_score_key(float v) {
    // Finite logits are the model contract. NaNs rank as -inf; +/-0 tie.
    if (isnan(v)) v = -__int_as_float(0x7f800000);
    if (v == 0.f) v = 0.f;
    unsigned u = __float_as_uint(v);
    return u ^ ((u & 0x80000000u) ? 0xffffffffu : 0x80000000u);
}
__device__ __forceinline__ int v3z_slot_of(const int* bt, int token) {
    return bt[token >> 8] * 256 + (token & 255);
}
__device__ __forceinline__ unsigned v3z_cluster_rank() {
    unsigned r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}
__device__ __forceinline__ void v3z_cluster_sync() {
    asm volatile("barrier.cluster.arrive.release;\n\tbarrier.cluster.wait.acquire;" ::: "memory");
}
__device__ __forceinline__ void v3z_st_remote_u32(const void* saddr, unsigned rank, unsigned v) {
    unsigned a = (unsigned)__cvta_generic_to_shared(saddr);
    unsigned m;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(m) : "r"(a), "r"(rank));
    asm volatile("st.shared::cluster.u32 [%0], %1;" :: "r"(m), "r"(v) : "memory");
}
__device__ __forceinline__ unsigned v3z_ld_remote_u32(const void* saddr, unsigned rank) {
    unsigned a = (unsigned)__cvta_generic_to_shared(saddr);
    unsigned m, v;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(m) : "r"(a), "r"(rank));
    asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(v) : "r"(m) : "memory");
    return v;
}
// Packed u32 warp inclusive scan (deterministic shuffle chain, ascending lane).
__device__ __forceinline__ unsigned v3z_wscan_incl(unsigned v, unsigned lane) {
    unsigned inc = v;
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        unsigned y = __shfl_up_sync(0xffffffffu, inc, o);
        if (lane >= o) inc += y;
    }
    return inc;
}

extern "C" __global__ __launch_bounds__(256) __cluster_dims__(1, V3Z_Z, 1) void
glm53_dsa_topk_v3z(const float* logits, const int* pools, const int* bt,
                   const int* seq, int* dst, int logits_stride, int bt_cols) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    const int t = threadIdx.x;
    const int row = blockIdx.x;
    const unsigned z = v3z_cluster_rank();
    const unsigned lane = t & 31, wid = t >> 5;
    const int len = pools[row], tail = seq[row] & 3;
    const int* table = bt + (size_t)row * bt_cols;
    int* out = dst + (size_t)row * 2051;
    const float* x = logits + (size_t)row * logits_stride;

    if (len <= 512) {
        // Identity: all closed pools and the open tail, in order. Same pure
        // per-column function as v2, split across the cluster. Uniform branch
        // (len is per-row), so the early return is cluster-safe.
        const int count = len * 4 + tail;
        for (int j = z * 256 + t; j < 2051; j += V3Z_Z * 256)
            out[j] = j < count ? v3z_slot_of(table, j) : 0;
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        return;
    }

    __shared__ unsigned hist[256];                    // this CTA's pass histogram
    __shared__ unsigned all_hist[2][V3Z_Z][256];      // gather area (rank 0's used)
    __shared__ unsigned merged[256];                  // merged pass histogram
    __shared__ unsigned blkcnt_all[64];               // per-1024-block packed totals (rank 0)
    __shared__ unsigned wsum[8];                      // warp partials (packed scans)
    __shared__ unsigned s_prefix, s_mask, s_kleft;

    const int nblocks = (len + 1023) >> 10;           // <= 64 (len <= 65535)
    const int b0 = (int)(((long long)z * nblocks) / V3Z_Z);
    const int b1 = (int)(((long long)(z + 1) * nblocks) / V3Z_Z);
    const int j0 = b0 * 1024;
    const int j1 = min(b1 * 1024, len);

    if (t == 0) { s_prefix = 0; s_mask = 0; s_kleft = 512; }
    __syncthreads();

    // ---- 4 byte-radix passes: exact global histogram via cluster merge ----
    int parity = 0;
    for (int shift = 24; shift >= 0; shift -= 8, parity ^= 1) {
        const unsigned prefix = s_prefix, mask = s_mask;
        hist[t] = 0;
        __syncthreads();
        for (int j = j0 + t; j < j1; j += 256) {
            unsigned key = v3z_score_key(x[j]);
            if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 255], 1u);
        }
        __syncthreads();
        v3z_st_remote_u32(&all_hist[parity][z][t], 0, hist[t]);
        v3z_cluster_sync();
        unsigned sum = 0;
        #pragma unroll
        for (int c = 0; c < V3Z_Z; ++c) sum += v3z_ld_remote_u32(&all_hist[parity][c][t], 0);
        merged[t] = sum;
        __syncthreads();
        if (t == 0) {
            // Identical integer scan in every CTA: smallest digit from the
            // top whose cumulative count reaches `remain` (same rule as v2).
            unsigned remain = s_kleft;
            int digit = 255;
            for (; digit > 0; --digit) {
                unsigned c = merged[digit];
                if (remain <= c) break;
                remain -= c;
            }
            s_prefix = prefix | ((unsigned)digit << shift);
            s_mask = mask | (255u << shift);
            s_kleft = remain;
        }
        __syncthreads();
    }
    const unsigned cutoff = s_prefix, kleft = s_kleft;

    // ---- selection: publish per-block packed totals, global prefix, emit ----
    for (int b = b0; b < b1; ++b) {
        const int base = b * 1024;
        unsigned sv = 0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int j = base + t * 4 + i;
            const bool in = j < len;
            const unsigned key = in ? v3z_score_key(x[j]) : 0;
            sv += (unsigned)(in && key == cutoff) + ((unsigned)(in && key > cutoff) << 11);
        }
        const unsigned inc = v3z_wscan_incl(sv, lane);
        if (lane == 31) wsum[wid] = inc;
        __syncthreads();
        if (t == 0) {
            unsigned tot = 0;
            #pragma unroll
            for (int w = 0; w < 8; ++w) tot += wsum[w];
            v3z_st_remote_u32(&blkcnt_all[b], 0, tot);
        }
        __syncthreads();
    }
    v3z_cluster_sync();

    // Global exclusive prefix over all blocks in ascending block order.
    // Redundant identical integer scan; prefix stops at my first block.
    unsigned eqbase = 0, gtbase = 0;
    for (int b = 0; b < b0; ++b) {
        const unsigned tot = v3z_ld_remote_u32(&blkcnt_all[b], 0);
        eqbase += tot & 2047u;
        gtbase += tot >> 11;
    }

    for (int b = b0; b < b1; ++b) {
        const int base = b * 1024;
        unsigned v[4], sv = 0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int j = base + t * 4 + i;
            const bool in = j < len;
            const unsigned key = in ? v3z_score_key(x[j]) : 0;
            v[i] = (unsigned)(in && key == cutoff) + ((unsigned)(in && key > cutoff) << 11);
            sv += v[i];
        }
        const unsigned inc = v3z_wscan_incl(sv, lane);
        if (lane == 31) wsum[wid] = inc;
        __syncthreads();
        unsigned wpre = 0, wtot = 0;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            wpre += (w < wid) ? wsum[w] : 0;
            wtot += wsum[w];
        }
        __syncthreads();
        unsigned before = wpre + inc - sv;   // this thread's packed exclusive prefix
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int j = base + t * 4 + i;
            if (j < len) {
                const unsigned key = v3z_score_key(x[j]);
                const bool eq = key == cutoff, gt = key > cutoff;
                const unsigned eqrank = eqbase + (before & 2047u);
                const unsigned gtrank = gtbase + (before >> 11);
                if (gt || (eq && eqrank < kleft)) {
                    const unsigned rank = gtrank + min(eqrank, kleft);
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) out[rank * 4 + k] = v3z_slot_of(table, j * 4 + k);
                }
            }
            before += v[i];
        }
        eqbase += wtot & 2047u;
        gtbase += wtot >> 11;
    }

    if (z == 0 && t < 3) out[2048 + t] = t < tail ? v3z_slot_of(table, len * 4 + t) : 0;

    // Keep every CTA resident until all DSMEM reads are done.
    v3z_cluster_sync();
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
