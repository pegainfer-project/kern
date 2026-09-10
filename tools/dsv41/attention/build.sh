#!/usr/bin/env bash
set -euo pipefail
# Pinned upstream checkout with initialized csrc/cutlass; no model weights needed.
upstream=${1:?FlashMLA checkout}
out=${2:?artifact output directory}
expected=4f38f29ef6793c228363e4af5be66d44e81167ba
[[ $(cat "$upstream/.git/HEAD") == "$expected" ]] || { echo 'Wrong FlashMLA revision' >&2; exit 1; }
mkdir -p "$out"
src=$(cd "$(dirname "$0")" && pwd)
nvcc -std=c++20 -O3 -shared -Xcompiler=-fPIC -gencode arch=compute_103a,code=sm_103a \
  --expt-relaxed-constexpr --expt-extended-lambda -DNDEBUG \
  -I"$upstream/csrc" -I"$upstream/csrc/kerutils/include" -I"$upstream/csrc/cutlass/include" \
  "$src/probe.cu" -lcuda -o "$out/libdsv41_sparse_prefill.so"
# The second ELF contains this template; the first is CUDA's support module.
(cd "$out" && cuobjdump -xelf all libdsv41_sparse_prefill.so)

nvcc -std=c++20 -O3 -shared -Xcompiler=-fPIC -gencode arch=compute_103a,code=sm_103a \
  --expt-relaxed-constexpr --expt-extended-lambda -DNDEBUG \
  -I"$upstream/csrc" -I"$upstream/csrc/kerutils/include" -I"$upstream/csrc/cutlass/include" \
  "$src/paged_probe.cu" -lcuda -o "$out/libdsv41_paged_decode.so"
(cd "$out" && cuobjdump -xelf all libdsv41_paged_decode.so)

bash "$src/build_fused.sh" "$upstream" "$out"
