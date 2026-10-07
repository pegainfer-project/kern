// The pipeline edge between two stages of a cut manifest (`kern cut`): one
// mailbox on the downstream GPU, written by the upstream stage through a
// peer pointer.
//
// The box is `u8[256 + data]`. Its header holds two item counters, `full`
// at byte 0 (items the upstream has posted) and `empty` at byte 64 (items
// the downstream has taken out). Both live on the downstream GPU, so the
// downstream spins locally and the upstream reads and writes remote
// memory; neither side keeps a counter of its own. An upstream item: wait
// until `empty == full` (the box is free), put each boundary buffer at its
// offset, post `full + 1`. A downstream item: wait until `full > empty`,
// take each buffer out, post `empty + 1`. Each step is its own launch, so
// stream order puts every copy of an item between its two flags; a post
// fences system-wide before its release store.
//
// A wait gives up after `timeout_ns` of globaltimer and traps: the process
// dies with a launch failure instead of hanging on a dead stage.
//
// Every stage also keeps a clock: `clock` is `i64[RING * 5]`, entry `item %
// RING` = {item + 1, recv wait begin, recv wait end, send wait begin, send
// wait end} in globaltimer ns. The item is read off the counters, never
// passed: a downstream stage's is its box's `empty` before the take, an
// upstream stage's the downstream box's `full` before the post. The waits
// stamp their own spin; a first stage stamps its start as "recv end" and a
// last stage its end as "send begin", so busy = send begin - recv end on
// every stage. One thread writes each field.
//
//   kern_pp_wait_empty (in u64 peers[2], out i64 clock[], i64 timeout_ns)
//   kern_pp_put        (in u64 peers[2], in u8 src[], i64 bytes, i64 offset)
//   kern_pp_post_full  (in u64 peers[2])
//   kern_pp_wait_full  (in u8 box[], out i64 clock[], i64 timeout_ns)
//   kern_pp_take       (in u8 box[], out u8 dst[], i64 bytes, i64 offset)
//   kern_pp_post_empty (inout u8 box[])
//   kern_pp_stamp_first(in u64 peers[2], out i64 clock[])
//   kern_pp_stamp_last (in u8 box[], out i64 clock[])
//   waits, posts, stamps: block [1,1,1], grid [1,1,1]; put / take: any grid.
// peers[1] is the downstream box; the upstream's own box (peers[0]) is unused.

#define PP_HEADER 256
#define PP_RING 256

typedef unsigned long long u64;

__device__ __forceinline__ u64 gtimer() {
    u64 t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ u64 ld_acquire_sys(const u64* p) {
    u64 v;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void st_release_sys(u64* p, u64 v) {
    asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

__device__ __forceinline__ u64* full_of(const unsigned char* box) {
    return reinterpret_cast<u64*>(const_cast<unsigned char*>(box));
}

__device__ __forceinline__ u64* empty_of(const unsigned char* box) {
    return reinterpret_cast<u64*>(const_cast<unsigned char*>(box) + 64);
}

__device__ __forceinline__ unsigned char* downstream(const u64* peers) {
    return reinterpret_cast<unsigned char*>(peers[1]);
}

__device__ __forceinline__ long long* entry(long long* clock, u64 item) {
    long long* e = clock + (item % PP_RING) * 5;
    e[0] = (long long)item + 1;
    return e;
}

__device__ void copy(unsigned char* dst, const unsigned char* src, long long bytes) {
    const long long stride = (long long)gridDim.x * blockDim.x;
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long vecs = bytes / 16;
    const uint4* s = reinterpret_cast<const uint4*>(src);
    uint4* d = reinterpret_cast<uint4*>(dst);
    for (long long i = t; i < vecs; i += stride) {
        d[i] = s[i];
    }
    for (long long i = vecs * 16 + t; i < bytes; i += stride) {
        dst[i] = src[i];
    }
}

extern "C" __global__ void kern_pp_wait_empty(const u64* peers, long long* clock, long long timeout_ns) {
    const unsigned char* box = downstream(peers);
    const u64 item = ld_acquire_sys(full_of(box));
    long long* e = entry(clock, item);
    const u64 t0 = gtimer();
    e[3] = (long long)t0;
    while (ld_acquire_sys(empty_of(box)) != item) {
        if ((long long)(gtimer() - t0) > timeout_ns) {
            printf("kern pp: item %llu waited %lld ns for the downstream stage to take the last one\n", item,
                   timeout_ns);
            __trap();
        }
    }
    e[4] = (long long)gtimer();
}

extern "C" __global__ void kern_pp_put(const u64* peers, const unsigned char* src, long long bytes,
                                       long long offset) {
    copy(downstream(peers) + PP_HEADER + offset, src, bytes);
}

extern "C" __global__ void kern_pp_post_full(const u64* peers) {
    const unsigned char* box = downstream(peers);
    __threadfence_system();
    st_release_sys(full_of(box), ld_acquire_sys(full_of(box)) + 1);
}

extern "C" __global__ void kern_pp_wait_full(const unsigned char* box, long long* clock, long long timeout_ns) {
    const u64 item = ld_acquire_sys(empty_of(box));
    long long* e = entry(clock, item);
    const u64 t0 = gtimer();
    e[1] = (long long)t0;
    while (ld_acquire_sys(full_of(box)) <= item) {
        if ((long long)(gtimer() - t0) > timeout_ns) {
            printf("kern pp: item %llu waited %lld ns for the upstream stage\n", item, timeout_ns);
            __trap();
        }
    }
    __threadfence_system();
    e[2] = (long long)gtimer();
}

extern "C" __global__ void kern_pp_take(const unsigned char* box, unsigned char* dst, long long bytes,
                                        long long offset) {
    copy(dst, box + PP_HEADER + offset, bytes);
}

extern "C" __global__ void kern_pp_post_empty(unsigned char* box) {
    __threadfence_system();
    st_release_sys(empty_of(box), *empty_of(box) + 1);
}

extern "C" __global__ void kern_pp_stamp_first(const u64* peers, long long* clock) {
    entry(clock, ld_acquire_sys(full_of(downstream(peers))))[2] = (long long)gtimer();
}

extern "C" __global__ void kern_pp_stamp_last(const unsigned char* box, long long* clock) {
    entry(clock, ld_acquire_sys(empty_of(box)) - 1)[3] = (long long)gtimer();
}
