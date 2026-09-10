#!/usr/bin/env bash
set -euo pipefail
upstream=${1:?FlashMLA checkout}
out=${2:?output directory}
src=$(cd "$(dirname "$0")" && pwd)
core="$upstream/csrc/kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn"
python3 "$src/patch_fused.py" "$core/kernel.cuh" "$out/fused_kernel.cuh"
nvcc -std=c++20 -O3 -shared -Xcompiler=-fPIC -gencode arch=compute_103a,code=sm_103a \
  --expt-relaxed-constexpr --expt-extended-lambda -DNDEBUG \
  -I"$out" -I"$core" -I"$upstream/csrc" -I"$upstream/csrc/kerutils/include" -I"$upstream/csrc/cutlass/include" \
  "$src/fused_probe.cu" -lcuda -o "$out/libdsv41_fused_decode.so"
(cd "$out" && cuobjdump -xelf all libdsv41_fused_decode.so)

nvcc -std=c++20 -O3 -shared -Xcompiler=-fPIC -gencode arch=compute_103a,code=sm_103a \
  --expt-relaxed-constexpr --expt-extended-lambda -DNDEBUG \
  -I"$out" -I"$core" -I"$upstream/csrc" -I"$upstream/csrc/kerutils/include" -I"$upstream/csrc/cutlass/include" \
  "$src/fused_prefill_probe.cu" -lcuda -o "$out/libdsv41_fused_prefill.so"
(cd "$out" && cuobjdump -xelf all libdsv41_fused_prefill.so)
