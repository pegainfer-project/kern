// Tray-local f32 allreduce over exported buffers, the TensorRT-LLM protocol on
// kern's peer ABI. Ported from tensorrt_llm/kernels/communicationKernels/
// allReduceFusionKernels.cu (NVIDIA, Apache-2.0): one-shot Lamport below
// ONESHOT_MAX_ROWS tray rows, two-shot (reduce-scatter, all-gather) with two
// flag barriers above. What changed against the original is listed at the end.
//
//   one-shot   every rank pushes its whole partial into a slot of every peer's
//              Lamport buffer; a slot word that still reads -0.0 has not
//              arrived (the poison is the flag, so flags cost no bytes);
//              every rank sums the N slots in rank order. Three staged buffers
//              rotate: the stage two behind is cleared to -0.0 while this one
//              is in flight, so no barrier surrounds the exchange.
//   two-shot   copy the partial into the local comm buffer; barrier; reduce
//              own rows by pulling every peer's copy, push the sum into every
//              peer's second half; barrier; read the second half. Plain 16 B
//              loads and stores, no poison, so it works for any row count.
//
// Row layout is kern's "own rows first" (peer_collective.cu): rank q's block
// is tray rows [blocks[q], blocks[q+1]) and rank r's local row j is tray row
// (blocks[r] + j) mod rows, so the partials are rotated differently on every
// rank. Every remote write lands at the *receiver's* local row for the same
// tray row (`row_on`, `local_row`), and every rank reads its own buffers in
// its own order; sums are taken in rank order, so all ranks produce
// identical bytes. The two-shot deals clusters tray rows, not local rows:
// its barrier pairs block b with block b of each peer, so the block that
// wrote a row on one rank must be the block that reads it on another.
//
// Measured (tray03 GB300 x4, captured burst, grid 152 x 224 threads,
// cluster 8; `LL` is peer_collective.cu's epoch-flag one-shot):
//
//   tray rows      4     16     24     32     64     96    128    256
//   LL           5.2    8.3   11.9   14.1   24.9   35.7   46.0   90.3 us
//   one-shot     5.1    6.4    8.2    9.3   15.0   21.3   27.9   54.0 us
//   two-shot    23.3     --     --     --   30.5     --     --   49.1 us
//
// One-shot wins by 1.6-1.7x from 16 rows on (the flag rides in the poison,
// so the wire carries payload only). The two-shot's floor is its two
// barriers, 8.5 us each against the 3.8 us a poison exchange costs, so the
// crossover sits near 200 rows (ONESHOT_MAX_ROWS), not at TensorRT-LLM's 128.
//
//   kern_peer_allreduce_f32(in f32 x[rows, hidden], out f32 y[same],
//                           inout u8 comm, in u64 comm_peers[N],
//                           inout u8 flags, in u64 flag_peers[N],
//                           inout u8 lamport, in u64 lamport_peers[N],
//                           inout i32 state[8], out i32 err[1],
//                           in i32 blocks[N + 1], i32 rank, i32 rows, i32 hidden, i64 stage_bytes,
//                           i32 mode, i64 timeout_ns)
//     N is the compile-time NRANKS (default 4, `-DNRANKS=`); `rows` is the
//     tray's, blocks[N]. `comm` holds
//     2 * N*rows_max*hidden f32 (partial copy, then the sum); `flags` holds
//     N * 256 i32; `lamport` holds 3 stages of `stage_bytes` >= N*N*rows_max*
//     hidden*4 bytes, filled with -0.0 by kern_peer_lamport_init once after
//     the peers are imported. `state` starts zeroed: [0] block counter,
//     [1] two-shot barrier phase, [2] Lamport stage, [4..5] i64 float4 count
//     to clear next time. `mode` 0 picks by row count, 1 forces one-shot,
//     2 forces two-shot. Launch: cluster [C,1,1], block [hidden/4/C,1,1]
//     (one token per cluster, one float4 per thread), grid a multiple of C
//     and at most 256 CTAs (the barrier flag table).
//     `err` is sticky: 1 + the rank whose data (one-shot) or barrier arrival
//     (two-shot) did not show within `timeout_ns`; untouched otherwise.
//
//   kern_peer_lamport_init(inout u8 lamport, i64 bytes)
//     fills the three Lamport stages with -0.0. Grid-stride, any launch shape.
//
// Changes against TensorRT-LLM: the `void** workspace` table became separate
// peer arrays (kern fills one array per exported buffer); rows are remapped
// for own-rows-first; the two-shot slice is "own rows" instead of a host-
// computed token split; spins time out into `err` instead of hanging; the
// PDL calls are dropped (kern launches plain); the stage size is a launch
// argument instead of a word in the workspace; only f32 and the plain
// all-reduce pattern are kept.
//
// Follow-ups (not done here): the two-shot's barrier (grid-wide arrive on a
// counter, then one sys-scope signal per peer, as DeepGEMM's nvlink_barrier
// does, or a poison exchange for its scatter and broadcast stages as
// TensorRT-LLM's MNNVL two-shot does) would take the 200-row crossover down
// to ~64; the kARResidualRMSNorm fused tail, once k3's landing/snapshot
// semantics are checked against it; bf16 partials with f32 accumulation to
// halve the bytes again; the one-shot rewrites an input -0.0 as +0.0 (poison
// collision), harmless for GEMM partial sums.

#include <cooperative_groups.h>
#include <cstdint>

#ifndef NRANKS
#define NRANKS 4
#endif
#ifndef ONESHOT_MAX_ROWS
#define ONESHOT_MAX_ROWS 192
#endif
#define FLAG_COUNT 256

namespace cg = cooperative_groups;

__device__ __forceinline__ unsigned long long gtimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ bool is_neg_zero(float v) {
    return __float_as_uint(v) == 0x80000000u;
}

__device__ __forceinline__ float4 neg_zero4() {
    const float z = __uint_as_float(0x80000000u);
    return make_float4(z, z, z, z);
}

__device__ __forceinline__ float4 ld_volatile4(const float4* p) {
    float4 v;
    asm volatile("ld.volatile.global.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ void st_flag(int* p, int f) {
    asm volatile("st.global.release.sys.b32 [%1], %0;" ::"r"(f), "l"(p));
}

__device__ __forceinline__ int ld_flag(int* p) {
    int f;
    asm volatile("ld.global.acquire.sys.b32 %0, [%1];" : "=r"(f) : "l"(p));
    return f;
}

__device__ __forceinline__ float4 add4(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

// Rank `to`'s local row holding the tray row that is local row `j` on `from`.
__device__ __forceinline__ int row_on(int j, const int* off, int from, int to) {
    const int rows = off[NRANKS];
    return ((j + off[from]) % rows - off[to] + rows) % rows;
}

// Rank `q`'s local row holding tray row `t`.
__device__ __forceinline__ int local_row(int t, const int* off, int q) {
    return (t - off[q] + off[NRANKS]) % off[NRANKS];
}

// Timed spin: true if `ready` came up before `timeout_ns` passed.
template <typename F>
__device__ __forceinline__ bool spin(long long timeout_ns, F ready) {
    unsigned long long t0 = 0;
    while (!ready()) {
        const unsigned long long now = gtimer();
        if (t0 == 0) {
            t0 = now;
        } else if ((long long)(now - t0) > timeout_ns) {
            return false;
        }
    }
    return true;
}

struct Index {
    int token, in_token, token_stride, per_row;
    __device__ __forceinline__ Index(int hidden) {
        cg::cluster_group cluster = cg::this_cluster();
        cg::grid_group grid = cg::this_grid();
        token = grid.cluster_rank();
        in_token = cluster.thread_rank();
        token_stride = grid.num_clusters();
        per_row = hidden >> 2;
    }
};

// Every block counts itself in; block 0 publishes the new phase once all have.
__device__ __forceinline__ void arrive(int* counter) {
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(counter, 1);
}

__device__ __forceinline__ void publish(int* counter, int* slot, int value, long long* clear, long long clear_value) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        while (*reinterpret_cast<volatile int*>(counter) != (int)gridDim.x) {
        }
        *slot = value;
        if (clear) *clear = clear_value;
        *counter = 0;
    }
}

__device__ __forceinline__ void oneshot(const float* x, float* y, uint8_t* lamport, const unsigned long long* lamport_peers,
                                        int* state, int* err, const int* blocks, int rank, int rows, int hidden,
                                        long long stage_bytes, long long timeout_ns) {
    const Index ix(hidden);
    int off[NRANKS + 1];
#pragma unroll
    for (int q = 0; q <= NRANKS; ++q) off[q] = blocks[q];
    const int rows_all = rows;
    const int tot = rows_all * ix.per_row;
    const int flag = state[2];
    long long* clear_ptr = reinterpret_cast<long long*>(state + 4);
    const long long clear_count = *clear_ptr;
    const float4* stage[NRANKS];
    float4* slot[NRANKS];
#pragma unroll
    for (int q = 0; q < NRANKS; ++q) {
        uint8_t* base = reinterpret_cast<uint8_t*>(lamport_peers[q]) + (flag % 3) * stage_bytes;
        slot[q] = reinterpret_cast<float4*>(base) + (long long)rank * tot;
        stage[q] = reinterpret_cast<const float4*>(base);
    }
    float4* clear_buf = reinterpret_cast<float4*>(lamport + ((flag + 2) % 3) * stage_bytes);
    const float4* mine = stage[rank];
    arrive(state);

    // Push own rows into slot `rank` of every peer, at the peer's row order.
    for (int j = ix.token; j < rows_all; j += ix.token_stride) {
        float4 v = reinterpret_cast<const float4*>(x)[j * ix.per_row + ix.in_token];
        if (is_neg_zero(v.x)) v.x = 0.f;
        if (is_neg_zero(v.y)) v.y = 0.f;
        if (is_neg_zero(v.z)) v.z = 0.f;
        if (is_neg_zero(v.w)) v.w = 0.f;
#pragma unroll
        for (int q = 0; q < NRANKS; ++q) slot[q][row_on(j, off, rank, q) * ix.per_row + ix.in_token] = v;
    }
    // Poison the stage two behind for its next use.
    const float4 poison = neg_zero4();
    for (long long i = (long long)ix.token * ix.per_row + ix.in_token; i < clear_count; i += (long long)ix.token_stride * ix.per_row) {
        clear_buf[i] = poison;
    }
    // Collect: N slots per element, summed in rank order.
    int fail = 0;
    for (int j = ix.token; j < rows_all; j += ix.token_stride) {
        const int idx = j * ix.per_row + ix.in_token;
        float4 vals[NRANKS];
        int missing = 0;
        const bool ok = spin(timeout_ns, [&]() {
            missing = 0;
#pragma unroll
            for (int r = 0; r < NRANKS; ++r) {
                vals[r] = ld_volatile4(mine + (long long)r * tot + idx);
                const bool got = !(is_neg_zero(vals[r].x) || is_neg_zero(vals[r].y) || is_neg_zero(vals[r].z) || is_neg_zero(vals[r].w));
                if (!got && missing == 0) missing = 1 + r;
            }
            return missing == 0;
        });
        if (!ok) fail = missing;
        float4 acc = vals[0];
#pragma unroll
        for (int r = 1; r < NRANKS; ++r) acc = add4(acc, vals[r]);
        reinterpret_cast<float4*>(y)[idx] = acc;
    }
    if (fail) atomicMax(err, fail);
    publish(state, state + 2, (flag + 1) % 3, clear_ptr, (long long)NRANKS * tot);
}

struct Barrier {
    int phase;
    int* target;
    int* current;
    __device__ __forceinline__ Barrier(int rank, int phase0, int* flags, const unsigned long long* flag_peers)
        : phase(phase0), target(nullptr), current(nullptr) {
        if (threadIdx.x < NRANKS) {
            target = reinterpret_cast<int*>(flag_peers[threadIdx.x]) + rank;
            current = flags + blockIdx.x * NRANKS + threadIdx.x;
        }
    }
    // Block b tells block b of every rank its new phase, then waits for each
    // rank's word to leave the old one: a pairwise block barrier, which is
    // enough because every block handles the same tray rows on every rank.
    // (TensorRT-LLM writes all FLAG_COUNT rows so a grid that shrinks between
    // calls cannot read a stale word; kern's grid is a manifest constant, and
    // the one-row store took 5 us off the barrier at B = 1.)
    __device__ __forceinline__ void sync(int* err, long long timeout_ns) {
        __syncthreads();
        if (threadIdx.x < NRANKS) {
            phase = phase == 2 ? 0 : phase + 1;
            const int prev = phase == 0 ? 2 : phase - 1;
            st_flag(target + blockIdx.x * NRANKS, phase);
            if (!spin(timeout_ns, [&]() { return ld_flag(current) != prev; })) atomicMax(err, 1 + (int)threadIdx.x);
        }
        __syncthreads();
    }
};

__device__ __forceinline__ void twoshot(const float* x, float* y, uint8_t* comm, const unsigned long long* comm_peers,
                                        int* flags, const unsigned long long* flag_peers, int* state, int* err, const int* blocks,
                                        int rank, int rows, int hidden, long long timeout_ns) {
    const Index ix(hidden);
    int off[NRANKS + 1];
#pragma unroll
    for (int q = 0; q <= NRANKS; ++q) off[q] = blocks[q];
    const int rows_all = rows;
    const int tot = rows_all * ix.per_row;
    float4* bufs[NRANKS];
#pragma unroll
    for (int q = 0; q < NRANKS; ++q) bufs[q] = reinterpret_cast<float4*>(comm_peers[q]);
    float4* local = reinterpret_cast<float4*>(comm);
    Barrier barrier(rank, state[1], flags, flag_peers);
    arrive(state);

    // The barrier pairs block b with block b of every peer, not with the
    // whole peer grid, so a row must be handled by the same cluster on every
    // rank: clusters are dealt tray rows, and the rotation only moves the
    // address (`local_row`). Own rows first means rank r's slice is tray
    // rows [blocks[r], blocks[r+1]).
    for (int t = ix.token; t < rows_all; t += ix.token_stride) {
        const int idx = local_row(t, off, rank) * ix.per_row + ix.in_token;
        local[idx] = reinterpret_cast<const float4*>(x)[idx];
    }
    barrier.sync(err, timeout_ns);
    const int first = off[rank], last = off[rank + 1];
    for (int t = first + ((ix.token - first) % ix.token_stride + ix.token_stride) % ix.token_stride; t < last;
         t += ix.token_stride) {
        float4 vals[NRANKS];
#pragma unroll
        for (int q = 0; q < NRANKS; ++q) vals[q] = bufs[q][local_row(t, off, q) * ix.per_row + ix.in_token];
        float4 acc = vals[0];
#pragma unroll
        for (int q = 1; q < NRANKS; ++q) acc = add4(acc, vals[q]);
#pragma unroll
        for (int q = 0; q < NRANKS; ++q) bufs[q][tot + local_row(t, off, q) * ix.per_row + ix.in_token] = acc;
    }
    barrier.sync(err, timeout_ns);
    for (int t = ix.token; t < rows_all; t += ix.token_stride) {
        const int idx = local_row(t, off, rank) * ix.per_row + ix.in_token;
        reinterpret_cast<float4*>(y)[idx] = local[tot + idx];
    }
    publish(state, state + 1, barrier.phase, nullptr, 0);
}

extern "C" __global__ void __launch_bounds__(1024)
kern_peer_allreduce_f32(const float* x, float* y, uint8_t* comm, const unsigned long long* comm_peers, int* flags,
                        const unsigned long long* flag_peers, uint8_t* lamport, const unsigned long long* lamport_peers,
                        int* state, int* err, const int* blocks, int rank, int rows, int hidden, long long stage_bytes,
                        int mode, long long timeout_ns) {
    const bool one = mode == 1 || (mode == 0 && rows <= ONESHOT_MAX_ROWS);
    if (one) {
        oneshot(x, y, lamport, lamport_peers, state, err, blocks, rank, rows, hidden, stage_bytes, timeout_ns);
    } else {
        twoshot(x, y, comm, comm_peers, flags, flag_peers, state, err, blocks, rank, rows, hidden, timeout_ns);
    }
}

extern "C" __global__ void kern_peer_lamport_init(uint8_t* lamport, long long bytes) {
    float4* p = reinterpret_cast<float4*>(lamport);
    const float4 poison = neg_zero4();
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < bytes / 16; i += (long long)gridDim.x * blockDim.x) {
        p[i] = poison;
    }
}
