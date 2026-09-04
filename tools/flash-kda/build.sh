#!/usr/bin/env bash
# Build the vendored FlashKDA kernels into tools/kernels-bin/flash_kda_d128.cubin.
#   CUTLASS_INCLUDE=<cutlass/include> tools/flash-kda/build.sh
# The kernels need CUTLASS/CuTe headers (upstream pins CUTLASS 4.x); any 4.3+
# include tree works, e.g. FlashInfer's 3rdparty/cutlass/include.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="${1:-$here/../kernels-bin/flash_kda_d128.cubin}"
nvcc="${NVCC:-nvcc}"
arch="${KERN_SM:-sm_103a}"
: "${CUTLASS_INCLUDE:?set CUTLASS_INCLUDE to a CUTLASS include directory}"
"$nvcc" -cubin -O3 -std=c++17 -arch="$arch" \
  --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math \
  -I"$here" -I"$here/csrc" -I"$here/csrc/smxx" -I"$CUTLASS_INCLUDE" \
  -o "$out" "$here/kern_flash_kda.cu"
echo "built $out ($arch)" >&2
