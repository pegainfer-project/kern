#!/usr/bin/env bash
# Build kern's K3 MegaMoE cubin and the layout dump tool.
#
#   tools/build_k3_mega.sh [out_dir=target/cubins]
#     DEEPGEMM_ROOT=<pegainfer's vendored DeepGEMM>   KERN_SM=sm_103a   NVCC_EXTRA="-DK3_MEGA_CFG=<i>"
#
# The kernel is DeepGEMM's sm100 fp8×fp4 MegaMoE with kern's peer-table
# SymBuffer (tools/k3-mega/). Needs nvcc ≥ 13 and the DeepGEMM + cutlass
# headers; the cubin is sha-pinned by the manifest generator like every other
# handwritten kernel.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="${1:-$repo/target/cubins}"
arch="${KERN_SM:-sm_103a}"
dg="${DEEPGEMM_ROOT:-$HOME/pegainfer/pegainfer-kernels/third_party/DeepGEMM}"
src="$repo/tools/k3-mega"
mkdir -p "$out"
inc=(-I "$src/include" -I "$dg/deep_gemm/include" -I "$dg/third-party/cutlass/include" -I "$dg/third-party/fmt/include")
flags=(-std=c++20 -O3 --expt-relaxed-constexpr -DDG_NO_TORCH ${NVCC_EXTRA:-})
nvcc -cubin "-arch=$arch" "${flags[@]}" "${inc[@]}" -o "$out/k3_mega_moe.cubin" "$src/kern_k3_mega_moe.cu"
nvcc "${flags[@]}" "${inc[@]}" -o "$out/k3_mega_layout_dump" "$src/layout_dump.cu"
echo "built k3_mega_moe.cubin ($arch) and k3_mega_layout_dump -> $out" >&2
