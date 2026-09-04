// kern's translation unit for the vendored FlashKDA kernels: pull in the
// upstream launch layer so nvcc instantiates kernel 1 and kernel 2 for the
// one configuration kern launches, and emit them into a cubin. The host-side
// launch code compiles too but is never called; kern launches the kernels
// from the manifest (tensormaps and packed structs, docs/k3-kernel-abi.md K7).
#include "csrc/smxx/fwd_launch.cu"
