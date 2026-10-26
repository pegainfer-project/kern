#!/usr/bin/env bash
# The flashinfer_sampling cubin from FlashInfer's headers (PROVENANCE.md):
#   FLASHINFER_INCLUDE=<flashinfer 0.7.2>/include [NVCC=nvcc] [KERN_SM=sm_103a] tools/flashinfer-sampling/build.sh <out.cubin>
# then `tools/kernels/import_cubin.py --family flashinfer_sampling --cubin <out.cubin> ...`.
set -euo pipefail
: "${FLASHINFER_INCLUDE:?FLASHINFER_INCLUDE=<flashinfer>/include}"
out=${1:?out.cubin}
here=$(cd "$(dirname "$0")" && pwd)
"${NVCC:-nvcc}" -cubin -arch="${KERN_SM:-sm_103a}" -std=c++17 -O3 -I"$FLASHINFER_INCLUDE" -o "$out" "$here/instantiate.cu"
cuobjdump -res-usage "$out" | sed -n 's/^ *Function \(.*\):$/\1/p'
