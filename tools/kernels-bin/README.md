# Prebuilt kernels

Cubins built by a toolchain the repo's `nvcc` build does not run, checked in
as artifacts and pinned by sha256 like every other module. Each one has a
build recipe here; regenerate with it and commit the new bytes together.

## `mla_decode_h96_p64.cubin`

NVIDIA's Blackwell MLA decode kernel, written in the CuTe DSL and shipped as
Python source in FlashInfer (`flashinfer/cute_dsl/attention/monolithic/
mla_decode_fp16.py`, BSD-3), compiled once for K3's geometry:

| parameter          | value                                          |
|--------------------|------------------------------------------------|
| heads / q tokens   | 96 / 1                                         |
| latent / rope dims | 512 / 64, bf16                                 |
| page size          | 64 tokens                                      |
| split KV           | variable per row (`is_var_split_kv`), 256-split reducer |
| target             | sm_103a (GB300)                                |
| toolchain          | nvidia-cutlass-dsl 4.6.0, CUDA 13, FlashInfer 0.6.x |

Two entries: the split-KV attention (`kernel_cutlass_split_kv_kernel_…_0`,
384 threads, cluster 2×1×1, 232448 B dynamic smem) and its reduction
(`kernel_cutlass_reduction_kernel_…_1`, 128 threads, 1024 B smem). The
parameter ABI the manifest packs is documented in `docs/k3-kernel-abi.md`
K5; `tools/gen_k3_decode.py` writes it.

Rebuild (inside an image with FlashInfer and the CuTe DSL, a free GPU):

    python3 tools/build_mla_dsl.py tools/kernels-bin

The DSL JIT is deterministic for a fixed toolchain; a different toolchain is
a different cubin and the manifests pin whichever one is checked in.

## `flash_kda_d128.cubin`

MoonshotAI's FlashKDA chunked KDA prefill forward (CUTLASS/CuTe, MIT), the
K3 span kernel: kernel 1 `_flash_kda_fwd_prepare` (intra-chunk, grid
(tiles, H), 256 threads, 21248 B smem) and kernel 2
`_flash_kda_fwd_recurrence` (state recurrence + output, grid (1, H), 192
threads, 98432 B smem), one instantiation `<D=128, has_state_in,
has_state_out, output_state, varlen=false>`. Sources, license and the trim are
under `tools/flash-kda/` (see its `PROVENANCE.md`); the parameter ABI (11
cute TiledCopy structs of 256 B each: a `CUtensorMap` at 0 plus one runtime
stride `int` at 128 for the q/k/g and q/v copies, then scalars) is documented
in `docs/k3-kernel-abi.md` and was lifted with `tools/kernel-capture`.

| parameter | value |
|---|---|
| head dim | 128 (q/k/v/g `[T, H, 128]` bf16, beta `[H, T]` bf16) |
| chunk | 16 tokens |
| state | f32 `[H][128 v][128 k]` in and out |
| target | sm_103a (GB300) |
| toolchain | CUDA 13.1 nvcc, CUTLASS 4.x headers (FlashInfer 0.6's `3rdparty/cutlass`) |

Rebuild (host nvcc, no GPU needed):

    CUTLASS_INCLUDE=<cutlass>/include NVCC=/usr/local/cuda-13.1/bin/nvcc tools/flash-kda/build.sh
