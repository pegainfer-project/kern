#!/usr/bin/env bash
set -euo pipefail
runner=${1:?program_io}
artifacts=${2:?artifacts}
for rows in 1 12 32; do
  dir="$artifacts/paged-io-$rows"
  "$runner" --manifest "$dir/manifest.json" --cubins "$artifacts" --vars rows="$rows" \
    --in q="$dir/q.bin" --in cache="$dir/cache.bin" --in ids="$dir/ids.bin" \
    --in lengths="$dir/lengths.bin" --in sink="$dir/sink.bin" \
    --in extra="$dir/extra.bin" --in extra_ids="$dir/extra_ids.bin" --in extra_lengths="$dir/extra_lengths.bin" \
    --out out="$dir/kern.bin"
  cmp "$dir/out.bin" "$dir/kern.bin"
done
