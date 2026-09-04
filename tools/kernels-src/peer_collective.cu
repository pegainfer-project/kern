// Tray-local all-gather over exported buffers, Lamport style: every 16-byte
// store carries its own flag ({data0, epoch, data1, epoch}), so a receiver
// spins on the slot it is waiting for and nothing surrounds the exchange —
// no barrier before (the slot is fresh because the epoch is) and none after
// (a slot holding the epoch is the arrival). Plain st/ld only: no TMA, no
// bulk copy, no multicast. The shape is NCCL's LL protocol on kern's peer
// ABI: a `u64[nranks]` of the group's copies of the symmetric buffer, a
// carried epoch, a timeout that reports instead of hanging. The allreduce
// that used to live beside it is `peer_allreduce.cu` (TensorRT-LLM's
// protocol): the epoch-in-payload flag doubles the bytes on the link, which
// is fine for a gather whose output is the data itself and was the ceiling
// for the reduction.
//
// Row layout, "own rows first": a group of R ranks runs one batch of
// `blocks[R]` tray rows, rank q's block being tray rows
// [blocks[q], blocks[q+1]) — blocks need not be equal (one rank may carry a
// prefill run, the others a row each). Rank r's local row j is tray row
// (blocks[r] + j) mod blocks[R]: a rank's own rows are always rows 0..own of
// every gathered buffer and the ops that only work on their owner's rows
// (paged attention, the expert dispatch) need no rank-dependent offset.
// Source q's rows land at local rows (blocks[q] - blocks[r]) mod blocks[R]
// onwards, in source order around the group, so every rank produces
// bit-identical results.
//
// Symmetric buffer: `sym` holds 2 * nranks regions of `region_packs`
// 16-byte slots — two sub-buffers by epoch parity (a peer may be one
// exchange ahead, never two: it cannot finish the next one without this
// rank's contribution), one region per source. Epochs live in
// `epochs[gridDim.x]`, one carry per CTA so no CTA has to wait for another;
// every rank runs the same launch sequence with the same grid, so the
// epochs agree. A stale slot holds an older epoch and never matches.
//
//   kern_peer_allgather(in u8 x[own * row_bytes], out u8 y[blocks[nranks] * row_bytes],
//                       inout u8 sym, in u64 peers[nranks], inout u32 epochs[grid],
//                       in i32 blocks[nranks + 1], i32 rank, i32 nranks, i32 row_bytes,
//                       i32 region_packs, i64 timeout_ns)
//     own rows are copied to local rows 0.., source q's to local row
//     (blocks[q] - blocks[rank]) mod blocks[nranks] onwards; a region holds
//     one source's rows, so region_packs >= rows_max * row_bytes / 16.
//     row_bytes a multiple of 8. block [256,1,1], grid [G,1,1] with G fixed
//     per op (the epochs carry is per CTA), nranks <= 8; `err` is sticky:
//     1 + the source rank a slot never arrived from within `timeout_ns`,
//     left untouched otherwise.
//
// Measured (tray03 GB300 x4, captured burst, grid 256): allgather [B,7168]
// bf16 4.5-5.4 us, [B,28672] f32 8.6 us at B=16.

__device__ __forceinline__ unsigned long long gtimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// {d0, e, d1, e}: each 8-byte half is one data word and its flag, so a
// receiver that sees both flags has both words.
// `asm volatile` without a memory clobber: the flagged accesses stay in
// program order among themselves, and the compiler is free to hoist the
// plain loads of the next pack over them (with the clobber every pack
// serialised load -> 3 stores -> load, a third of the link).
__device__ __forceinline__ void st_ll(uint4* p, unsigned d0, unsigned d1, unsigned e) {
    asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(d0), "r"(e), "r"(d1), "r"(e));
}

__device__ __forceinline__ uint4 ld_ll(const uint4* p) {
    uint4 v;
    // System scope: the writer is another GPU. (A gpu-scope load also
    // observes fabric writes on GB300 and reads back faster at B=64, but the
    // memory model does not promise it; a weak .cg load never sees them.)
    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

#define MAX_RANKS 8

// Spin until every one of the `n` slots carries epoch `e`, polling them all
// in one sweep; returns 0, or 1 + the first slot still missing once
// `timeout_ns` pass. Slot i's data lands in d[i] as {x, z}.
__device__ __forceinline__ int recv_ll(const uint4* const* slots, int n, unsigned e, long long timeout_ns,
                                       uint2* d) {
    unsigned long long t0 = 0;
    for (;;) {
        uint4 v[MAX_RANKS];
#pragma unroll
        for (int i = 0; i < MAX_RANKS; ++i) {
            if (i < n) v[i] = ld_ll(slots[i]);
        }
        int missing = 0;
#pragma unroll
        for (int i = 0; i < MAX_RANKS; ++i) {
            if (i < n) {
                d[i] = make_uint2(v[i].x, v[i].z);
                if (missing == 0 && (v[i].y != e || v[i].w != e)) missing = 1 + i;
            }
        }
        if (missing == 0) return 0;
        const unsigned long long now = gtimer();
        if (t0 == 0) {
            t0 = now;
        } else if ((long long)(now - t0) > timeout_ns) {
            return missing;
        }
    }
}

// Per-CTA epoch: read once, published to the block, written back at the end.
__device__ __forceinline__ unsigned epoch_begin(unsigned* epochs, unsigned* s_e, int* s_fail) {
    if (threadIdx.x == 0) {
        *s_e = epochs[blockIdx.x] + 1;
        *s_fail = 0;
    }
    __syncthreads();
    return *s_e;
}

__device__ __forceinline__ void epoch_end(unsigned* epochs, int* err, unsigned e, int* s_fail) {
    __syncthreads();
    if (threadIdx.x == 0) {
        epochs[blockIdx.x] = e;
        if (*s_fail) atomicMax(err, *s_fail);
    }
}

extern "C" __global__ void kern_peer_allgather(const uint2* x, uint2* y, uint4* sym, const unsigned long long* peers,
                                               unsigned* epochs, int* err, const int* blocks, int rank, int nranks,
                                               int row_bytes, int region_packs, long long timeout_ns) {
    __shared__ unsigned s_e;
    __shared__ int s_fail;
    const unsigned e = epoch_begin(epochs, &s_e, &s_fail);
    const int per = row_bytes >> 3;
    const int total = blocks[nranks];
    const int nown = (blocks[rank + 1] - blocks[rank]) * per;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int sub = (e & 1) * nranks;

    for (int p = tid; p < nown; p += stride) {
        const uint2 d = x[p];
        for (int q = 0; q < nranks; ++q) {
            if (q == rank) continue;
            st_ll(reinterpret_cast<uint4*>(peers[q]) + (long long)(sub + rank) * region_packs + p, d.x, d.y, e);
        }
    }
    for (int p = tid; p < nown; p += stride) y[p] = x[p];
    for (int q = 0; q < nranks; ++q) {
        if (q == rank) continue;
        const int n_q = (blocks[q + 1] - blocks[q]) * per;
        uint2* dst = y + (long long)((blocks[q] - blocks[rank] + total) % total) * per;
        const uint4* region = sym + (long long)(sub + q) * region_packs;
        for (int p = tid; p < n_q; p += stride) {
            const uint4* slot = region + p;
            uint2 d;
            if (recv_ll(&slot, 1, e, timeout_ns, &d)) atomicMax(&s_fail, q + 1);
            dst[p] = d;
        }
    }
    epoch_end(epochs, err, e, &s_fail);
}
