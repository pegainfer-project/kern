# FlashKDA (vendored)

Upstream: https://github.com/MoonshotAI/FlashKDA
Commit: `7afb9f4` (2026-09-01, "replace fp16 Neumann inverse with 8x8 fp32
forward substitution + 16x16 bf16 merge")
License: MIT (Copyright (c) 2026 MoonshotAI) — `LICENSE` in this directory.

FlashKDA is MoonshotAI's chunkwise Kimi Delta Attention prefill forward
(CUTLASS/CuTe, SM90 TMA + mma.sync, one code path for sm_90a..sm_121a). kern
uses it as the K3 span (extend / prefill) KDA time-axis kernel: kernel 1 is the
intra-chunk (16-token tile) preprocessing, kernel 2 the inter-chunk state
recurrence and output, f32 `[H][128 v][128 k]` state in and out — the same
layout kern's decode kernel `k3_kda_core` keeps in the KDA line.

## What is vendored

- `csrc/fwd.h` — `launch_fwd` declaration (unmodified)
- `csrc/smxx/utils.cuh`, `fwd_kernel1.cuh`, `fwd_kernel2.cuh` (unmodified)
- `csrc/smxx/fwd_launch.cu` — **modified**: the explicit-instantiation list at
  the bottom is trimmed from 14 variants to the one kern launches
  (`<128, true, true, true, false>`). Everything above the marker is upstream
  verbatim.

Not vendored: the PyTorch binding (`csrc/flash_kda.cpp`), the Python package,
tests, benchmarks, the CUTLASS submodule. `kern_flash_kda.cu` is kern's own
translation unit (it only includes the launch layer so nvcc emits the two
kernels into one cubin); `build.sh` is the cubin recipe. The workspace-size
arithmetic (`tools/gen_k3_decode.py`) is reproduced from the upstream binding.

pegainfer vendors the same sources at commit `1ce47ea` with the same trim
(`pegainfer-kernels/third_party/flash-kda`); its C shim
`pegainfer-kernels/csrc/k3/k3_flash_kda.cu` documents the operand contract kern
follows (q/k/v/g `[T, H, 128]` bf16, beta `[H, T]` bf16, `gate_scale =
lower_bound * log2(e)`).
