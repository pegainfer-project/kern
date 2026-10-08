#!/bin/bash
# Build the GLM-5.3 handwritten kernels into kernels-glm53-handwritten/.
# Uses nvcc from PATH (or $CUDA_HOME/bin/nvcc); override CXX with CUDAHOSTCXX.
set -e
NVCC=${NVCC:-${CUDA_HOME:+/usr/local/cuda/bin}/nvcc}
NVCC=$(command -v nvcc || echo "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc")
DIR=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$DIR/kernels-glm53-handwritten
mkdir -p "$OUT"
for src in "$DIR"/tools/glm53/kernels/*.cu; do
  name=$(basename "$src" .cu)
  "$NVCC" -cubin -arch=sm_90a -O3 -o "$OUT/$name.cubin" "$src"
  sha256sum "$OUT/$name.cubin"
done
