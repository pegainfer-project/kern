#!/usr/bin/env bash
set -euo pipefail
runner=${1:?program_io}
artifacts=${2:?artifacts}
page=${3:-64}
for rows in 4 12 32; do
  dir="$artifacts/paged-indexer-io-$page-$rows"
  "$runner" --manifest "$dir/manifest.json" --cubins "$artifacts" --vars rows="$rows" \
    --in q="$dir/q.bin" --in qs="$dir/qs.bin" --in weights="$dir/weights.bin" \
    --in end="$dir/end.bin" --in table="$dir/table.bin" --in cache="$dir/cache.bin" \
    --in projection="$dir/projection.bin" --out weight_bf16="$dir/weight_bf16.bin" \
    --out weight_f32="$dir/weight_f32.bin" --out out="$dir/out.bin"
done
