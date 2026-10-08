// TP8 bf16 one-shot push on kern's export/peer ABI, natural row order.
// Design lineage: tools/kernels-src/peer_allreduce.cu (TRT-LLM port).
// Numeric/protocol oracle: sglang custom_all_reduce.cuh 1shot_push,
// communicator.cuh LamportTrait<bf16,8,4>, vec.cuh reduce_vec.
// See docs/glm53/ar_lamport.md. No CUDA runtime or host synchronization.
//
// ABI (identical for the optional _pdl entry):
//   x, y: bf16 [rows,4096], 16-byte aligned; x == y is allowed.
//   sym: exported zero-initialized carry, SYM_BYTES bytes:
//        uint4 payload[2][8][8192], then u32 phase[16].
//   peers: u64[8], rank-ordered bases of sym (including self).
//   err: sticky i32[1], zero at allocation; timeout => 1+missing rank.
//   rank: i32 0..7; rows: i32 1..16; timeout_ns: i64 > 0.
// Launch MUST be grid [16,1,1], block [128,1,1], on all ranks.
// No concurrent use of sym. Never reset phases independently of payload.
// Errors are fatal to the collective group; poll err between steps, then
// discard/recreate the group. A timed-out result is deliberately qNaN.
#include <cuda_bf16.h>
#include <stdint.h>

#ifndef NRANKS
#define NRANKS 8
#endif
static_assert(NRANKS == 8, "GLM-5.3 manifest is TP8");
static constexpr int CTAS = 16;
static constexpr int THREADS = 128;
static constexpr int VECS_PER_ROW = 4096 / 8;
static constexpr int SLOT_VECS = 16 * VECS_PER_ROW;
static constexpr int PAYLOAD_VECS = 2 * NRANKS * SLOT_VECS;
static constexpr int SYM_BYTES = PAYLOAD_VECS * 16 + CTAS * 4;
static_assert(SYM_BYTES == 2097216, "keep ops_common.py ABI in sync");

__device__ __forceinline__ unsigned long long now_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ uint4 load_sys(const uint4* p) {
    uint4 v;
    asm volatile("ld.relaxed.sys.global.v4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void store_sys(uint4* p, uint4 v) {
    asm volatile("st.relaxed.sys.global.v4.b32 [%0], {%1,%2,%3,%4};"
                 :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

__device__ __forceinline__ uint4 encode(uint4 v) {
    // kAtom=4, NOT one marker per bf16. Preserve the upper +0 in (0,0).
    if (v.x == 0) v.x = 0x8000u;
    if (v.y == 0) v.y = 0x8000u;
    if (v.z == 0) v.z = 0x8000u;
    if (v.w == 0) v.w = 0x8000u;
    return v;
}

__device__ __forceinline__ bool empty(uint4 v) {
    return v.x == 0 || v.y == 0 || v.z == 0 || v.w == 0;
}

__device__ __forceinline__ float2 unpack(unsigned v) {
    // Exact bf16 -> fp32 widening, including signed zero and subnormals.
    return make_float2(__uint_as_float(v << 16), __uint_as_float(v & 0xffff0000u));
}

__device__ __forceinline__ unsigned pack_rn(float2 v) {
    const __nv_bfloat162 b = __float22bfloat162_rn(v);
    return *reinterpret_cast<const unsigned*>(&b);
}

__device__ __forceinline__ uint4 reduce_rank_order(const uint4 (&v)[NRANKS]) {
    float2 acc[4] = {unpack(v[0].x), unpack(v[0].y), unpack(v[0].z), unpack(v[0].w)};
#pragma unroll
    for (int r = 1; r < NRANKS; ++r) {
        const float2 f[4] = {unpack(v[r].x), unpack(v[r].y), unpack(v[r].z), unpack(v[r].w)};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[j].x = __fadd_rn(acc[j].x, f[j].x);
            acc[j].y = __fadd_rn(acc[j].y, f[j].y);
        }
    }
    return make_uint4(pack_rn(acc[0]), pack_rn(acc[1]), pack_rn(acc[2]), pack_rn(acc[3]));
}

template <bool PDL>
__device__ __forceinline__ void allreduce(const uint4* x, uint4* y, uint4* sym,
                                         const unsigned long long* peers, int* err,
                                         int rank, int rows, long long timeout_ns) {
    const int tid = (blockIdx.x + CTAS * (threadIdx.x / 32)) * 32 + threadIdx.x % 32;
    const int n = rows * VECS_PER_ROW;
    if constexpr (PDL) {
        // A hint only; loading the partial before wait would race its producer.
        if (tid < n) asm volatile("prefetch.global.L2 [%0];" :: "l"(x + tid));
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    unsigned* phases = reinterpret_cast<unsigned*>(sym + PAYLOAD_VECS);
    __shared__ unsigned phase;
    if (threadIdx.x == 0) phase = phases[blockIdx.x];
    __syncthreads();
    const unsigned half = (phase & 1) * NRANKS * SLOT_VECS;
    uint4* dest[NRANKS];
#pragma unroll
    for (int r = 0; r < NRANKS; ++r)
        dest[r] = reinterpret_cast<uint4*>(peers[r]) + half + rank * SLOT_VECS;

    // Send every owned vector before polling. No global/barrier dependency.
    for (int i = tid; i < n; i += CTAS * THREADS) {
        const uint4 v = encode(x[i]);
#pragma unroll
        for (int r = 0; r < NRANKS; ++r) store_sys(dest[r] + i, v);
    }

    unsigned long long first_wait = 0;
    int failure = 0;
    for (int i = tid; i < n; i += CTAS * THREADS) {
        uint4 v[NRANKS];
        while (!failure) {
            int missing = 0;
#pragma unroll
            for (int r = 0; r < NRANKS; ++r) {
                v[r] = load_sys(sym + half + r * SLOT_VECS + i);
                if (empty(v[r]) && missing == 0) missing = r + 1;
            }
            if (!missing) break;
            const unsigned long long t = now_ns();
            if (!first_wait) first_wait = t;
            if (t - first_wait >= static_cast<unsigned long long>(timeout_ns)) {
                failure = missing;
                atomicMax(err, failure);
            }
        }
        y[i] = failure ? make_uint4(0x7fc07fc0u, 0x7fc07fc0u, 0x7fc07fc0u, 0x7fc07fc0u)
                       : reduce_rank_order(v);
        // Retire exactly the vectors consumed this time. Tail rows stay zero.
#pragma unroll
        for (int r = 0; r < NRANKS; ++r)
            store_sys(sym + half + r * SLOT_VECS + i, make_uint4(0, 0, 0, 0));
    }
    __syncthreads();
    if (threadIdx.x == 0) phases[blockIdx.x] = phase ^ 1;
    if constexpr (PDL) {
        __syncthreads();
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
}

extern "C" __global__ __launch_bounds__(THREADS)
void glm53_ar_lamport(const uint4* x, uint4* y, uint4* sym,
                      const unsigned long long* peers, int* err,
                      int rank, int rows, long long timeout_ns) {
    allreduce<false>(x, y, sym, peers, err, rank, rows, timeout_ns);
}

extern "C" __global__ __launch_bounds__(THREADS)
void glm53_ar_lamport_pdl(const uint4* x, uint4* y, uint4* sym,
                          const unsigned long long* peers, int* err,
                          int rank, int rows, long long timeout_ns) {
    allreduce<true>(x, y, sym, peers, err, rank, rows, timeout_ns);
}
