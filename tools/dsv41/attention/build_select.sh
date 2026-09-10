#!/usr/bin/env bash
set -euo pipefail
upstream=${1:?DeepSelect checkout}
out=${2:?output directory}
[[ $(cat "$upstream/.git/HEAD") == 8e70df71d2a4b0c969ef96dc3b8998efa09a3315 ]]
src=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"
nvcc -std=c++20 -O3 -shared -Xcompiler=-fPIC -gencode arch=compute_103a,code=sm_103a \
  --expt-relaxed-constexpr --expt-extended-lambda -DNDEBUG \
  -I"$upstream/csrc" -I"$upstream/csrc/3rdparty/cutlass/include" -I"$upstream/csrc/3rdparty/kerutils/include" \
  "$src/select_probe.cu" -lcuda -o "$out/libdsv41_select.so"
(cd "$out" && cuobjdump -xelf all libdsv41_select.so)
nvcc -std=c++20 -O3 -gencode arch=compute_103a,code=sm_103a -cubin "$src/candidate.cu" -o "$out/candidate.cubin"
