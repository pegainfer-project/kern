# `tools/kernel-capture` — lifting a kernel's launch ABI into a manifest

kern launches third-party kernels (CUTLASS, CuTe DSL, Triton, cuBLAS-free
hand-written CUDA) from a manifest, with no host code of theirs: the manifest
declares the entry, the geometry, the parameter types and, for every TMA
descriptor and struct parameter, exactly which bytes go where (`tensormap`,
`bytes<n>` + `pack` in `docs/manifest.md`). Getting that declaration right by
reading a template library is slow and error-prone. This directory gets it
from the running kernel instead.

```
capture.c   CUPTI injection library: every module load, every launch, every
            cuTensorMapEncodeTiled — cubin bytes, staged parameter bytes,
            descriptor encode arguments
lift.py     one captured launch -> manifest launch skeleton
build.sh    builds libkernelcapture.so (needs CUPTI from a CUDA toolkit)
```

## The workflow

1. **Vendor the kernel** under `tools/<name>/` with its license, a
   `PROVENANCE.md` (upstream URL, commit, what was changed and why) and a
   `build.sh` that produces the cubin into `tools/kernels-bin/`; add the
   recipe to `tools/kernels-bin/README.md`. Trim template instantiations to
   the ones kern launches, keep everything else verbatim.
2. **Write a reference launcher** — a small host program that calls the
   kernel through upstream's own launch code on deterministic inputs and
   dumps inputs and outputs (`tools/flash-kda/probe.cu` is the model). This
   is both the ABI source and the bit-exact oracle for the manifest op.
3. **Capture** it:

       CUDA_INJECTION64_PATH=$PWD/tools/kernel-capture/libkernelcapture.so \
       KERNEL_CAPTURE_DIR=/some/dir ./probe 64 24

   `/some/dir/pid<N>/launches.jsonl` then holds one record per launch:
   symbol, grid/block/smem, function attributes, and per parameter its
   offset, size, staged bytes, whether it is a live device pointer (with the
   owning allocation), and any `CUtensorMap` found in it (at which byte,
   with the dtype/dims/strides/box/swizzle/L2-promotion it was encoded
   from). Descriptors are matched by their bytes against the encode calls
   the process made, so a descriptor buried in a 256-byte cute `TiledCopy`
   is found the same way as a bare 128-byte one.
4. **Lift** the launch:

       PATH=$PATH:/usr/local/cuda/bin \
       tools/kernel-capture/lift.py /some/dir/pid*/launches.jsonl --symbol fwd_prepare

   prints the manifest skeleton: `params` (`tensormap` / `bytes<n>` /
   `buffer` / `i32` …), `args` with every descriptor spelled out and every
   struct as a `pack`, and an `allocations` table. Pointers are
   `"@A+0x60000"` placeholders: allocation A, byte offset. Have the
   launcher print `<name> <pointer>` per allocation and pass the lines as
   `--names`, and the placeholders read `"@out+0x0"` instead: a letter is
   an address, and mapping letters by what a param "should" be is how the
   FlashKDA recurrence's `q`/`v`/`out` got swapped (the kernel takes `out`
   twice, as a TMA store descriptor and a pointer, and no `q` at all). The
   author turns shape-bound numbers into vars — `--all` prints one skeleton
   per distinct shape variant, which shows which numbers move. For cute `TiledCopy` parameters
   the live bytes after the descriptor (one `int` per dynamic
   `ScaledBasis<int, k>`) are decoded from the demangled type (`cu++filt`);
   the rest of the struct is host stack garbage that upstream never
   initialises and the pack leaves at zero.
5. **Declare** the op in the generator (`tools/gen_k3_decode.py` and
   friends) and record the ABI in `docs/k3-kernel-abi.md`. The runtime
   compares the declared params against `cuFuncGetParamInfo` at load, so a
   wrong width fails before a launch.
6. **Gate**: run the op through kern on the reference launcher's inputs and
   diff its outputs against the dump. Bit-exact is the expectation — same
   cubin, same bytes in, same bytes out; a difference is a wrong field.
   `crates/kern-run/examples/program_io.rs` runs any program of any manifest
   from raw input files and writes raw outputs, so the gate is a manifest
   with just the op (e.g. `tools/gen_flash_kda_probe.py`) plus `cmp`:

       program_io --manifest probe.json --cubins target/cubins --program span \
           --env span=64 --in q=dump/q.bin --in k=dump/k.bin ... --out out=got/out.bin
       cmp got/out.bin dump/out.bin

## Notes

- Driver-API and runtime `<<<>>>` launches are both seen. Kernels registered
  through the runtime cannot answer `cuFuncGetParamInfo` on their launch
  handle, so the layout comes from a private reload of the module at
  module-load time (`cache_module_abi`).
- The library is process-wide and provider-agnostic: vLLM/sglang/pegainfer
  engines work the same way (`pid<N>/` per rank). Loading cubins out of a
  running engine (`module_<id>.cubin`) is how the K3 kernels were originally
  extracted; `docs/k3-kernel-abi.md` ("ABI 是怎么拿到的") has that history.
- `cuTensorMapEncodeTiled` is intercepted through CUPTI's driver-API
  callback, which also catches callers that fetch the entry point with
  `cudaGetDriverEntryPoint` (cute does). Im2col descriptors are not
  recorded.
