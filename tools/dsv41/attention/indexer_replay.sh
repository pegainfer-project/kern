#!/usr/bin/env bash
set -euo pipefail
runner=${1:?program_io}
artifacts=${2:?artifacts}
for rows in 4 12 32; do
  dir="$artifacts/indexer-io-$rows"
  "$runner" --manifest "$dir/manifest.json" --cubins "$artifacts" --vars rows="$rows" \
    --in q="$dir/q.bin" --in qs="$dir/qs.bin" --in k="$dir/k.bin" --in ks="$dir/ks.bin" \
    --in weights="$dir/weights.bin" --in start="$dir/start.bin" --in end="$dir/end.bin" \
    --out out="$dir/out.bin"
done
