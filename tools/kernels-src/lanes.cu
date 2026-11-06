// The kernels of the lanes gate (crates/kern-runtime/examples/lanes.rs):
// a step that takes a known time, and a join that reads two branches.
// Both wait on their predecessors first (`griddepcontrol.wait`), so they
// may be launched with programmatic dependent launch; the step lets its
// dependents launch at once, so a dependent that started early reads the
// step's output only through its own wait.
//
//   kern_lanes_step(in f32 x[n], out f32 y[n], i32 n, i64 ns)
//     y = 2 x + 1 after `ns` of globaltimer; block [256,1,1], grid [1,1,1].
//   kern_lanes_join(in f32 a[n], in f32 b[n], out f32 y[n], i32 n)
//     y = a + b; block [256,1,1], grid [1,1,1].

__device__ __forceinline__ unsigned long long gtimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

extern "C" __global__ void kern_lanes_step(const float* x, float* y, int n, long long ns) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;");
    const unsigned long long t0 = gtimer();
    while (gtimer() - t0 < (unsigned long long)ns) {
    }
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = 2.0f * x[i] + 1.0f;
}

extern "C" __global__ void kern_lanes_join(const float* a, const float* b, float* y, int n) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = a[i] + b[i];
}
