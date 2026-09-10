#!/usr/bin/env bash
set -euo pipefail
runner=${1:?program_io}
artifacts=${2:?artifacts}
for page in 64 128; do
  dir="$artifacts/paged-sparse-io-$page"
  "$runner" --manifest "$dir/manifest.json" --cubins "$artifacts" --vars rows=4 \
    --in q="$dir/q.bin" --in qs="$dir/qs.bin" --in weights="$dir/weights.bin" \
    --in end="$dir/end.bin" --in table="$dir/table.bin" --in requests="$dir/requests.bin" \
    --in candidates="$dir/candidates.bin" --in cache="$dir/cache.bin" --out out="$dir/out.bin"
done
