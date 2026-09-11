#!/usr/bin/env bash
set -euo pipefail
src="$(cd "$(dirname "$0")" && pwd)"
out="${1:-$src/../../../target/cubins/dsv41}"
mkdir -p "$out"
nvcc -std=c++20 -O3 --fmad=false -arch="${KERN_SM:-sm_103a}" -cubin "$src/boundary.cu" -o "$out/boundary.cubin"
if [[ -n "${DEEPGEMM_ROOT:-}" ]]; then
  nvcc -std=c++20 -O3 --expt-relaxed-constexpr -DDG_NO_TORCH -arch="${KERN_SM:-sm_103a}" -I "$DEEPGEMM_ROOT/deep_gemm/include" -I "$DEEPGEMM_ROOT/third-party/cutlass/include" -cubin "$src/mega_mhc.cu" -o "$out/mega_mhc.cubin"
  nvcc -std=c++20 -O3 --expt-relaxed-constexpr -DDG_NO_TORCH -arch="${KERN_SM:-sm_103a}" -I "$DEEPGEMM_ROOT/deep_gemm/include" -I "$DEEPGEMM_ROOT/third-party/cutlass/include" -cubin "$src/mega_gate.cu" -o "$out/mega_gate.cubin"
  nvcc -std=c++20 -O3 --expt-relaxed-constexpr -DDG_NO_TORCH -arch="${KERN_SM:-sm_103a}" -I "$DEEPGEMM_ROOT/deep_gemm/include" -I "$DEEPGEMM_ROOT/third-party/cutlass/include" -cubin "$src/dense.cu" -o "$out/dense.cubin"
  mkdir -p "$out/include/deep_gemm/layout"
  cp "$src/../../k3-mega/include/deep_gemm/layout/sym_buffer.cuh" "$out/include/deep_gemm/layout/"
  python3 "$src/fork_moe.py" "$DEEPGEMM_ROOT" "$out/sm100_fp8_fp4_mega_moe.cuh"
  nvcc -std=c++20 -O3 --expt-relaxed-constexpr -DDG_NO_TORCH -arch="${KERN_SM:-sm_103a}" -I "$out/include" -I "$out" -I "$DEEPGEMM_ROOT/deep_gemm/include" -I "$DEEPGEMM_ROOT/third-party/cutlass/include" -cubin "$src/mega_moe.cu" -o "$out/mega_moe.cubin"
  nvcc -std=c++20 -O3 --expt-relaxed-constexpr -DDG_NO_TORCH -I "$DEEPGEMM_ROOT/deep_gemm/include" -I "$DEEPGEMM_ROOT/third-party/cutlass/include" "$src/moe_layout.cu" -o "$out/moe_layout"
  # One layout per expert count and per SM count the kernels are built for:
  # the MegaMoE ring is sized from the pool the persistent grid drains.
  for sms in 152 148; do
    "$out/moe_layout" 128 "$sms" > "$out/moe_layout_128_sm$sms.json"
    "$out/moe_layout" 384 "$sms" > "$out/moe_layout_384_sm$sms.json"
  done
fi
nvcc -std=c++20 -O3 -arch="${KERN_SM:-sm_103a}" -cubin "$src/weight_prep.cu" -o "$out/weight_prep.cubin"
nvcc -std=c++20 -O3 -arch="${KERN_SM:-sm_103a}" -cubin "$src/stage.cu" -o "$out/stage.cubin"
nvcc -std=c++20 -O3 -arch="${KERN_SM:-sm_103a}" -cubin "$src/fused_prep.cu" -o "$out/fused_prep.cubin"
