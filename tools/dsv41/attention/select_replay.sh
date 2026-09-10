#!/usr/bin/env bash
set -euo pipefail
runner=${1:?program_io}
artifacts=${2:?artifacts}
for kind in f32 bf16; do
  for k in 512 2048; do
    dir="$artifacts/select-$kind-$k"
    "$runner" --manifest "$dir/manifest.json" --cubins "$artifacts" --vars rows=8 \
      --in scores="$dir/scores.bin" --in end="$dir/end.bin" --out indices="$dir/kern.bin"
    cmp "$dir/indices.bin" "$dir/kern.bin"
  done
done
