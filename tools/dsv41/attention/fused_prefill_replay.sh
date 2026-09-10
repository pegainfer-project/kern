#!/usr/bin/env bash
set -euo pipefail
runner=${1:?program_io}
artifacts=${2:?artifacts}
for rows in 1 17 32; do
  dir="$artifacts/fused-prefill-io-$rows"
  "$runner" --manifest "$dir/manifest.json" --cubins "$artifacts" --vars rows="$rows" \
    --in q="$dir/q.bin" --in kv="$dir/kv.bin" --in ids="$dir/ids.bin" --in lengths="$dir/lengths.bin" \
    --in sink="$dir/sink.bin" --in positions="$dir/positions.bin" --in rope="$dir/rope.bin" \
    --out out="$dir/kern.bin" --out scales="$dir/kern-scales.bin"
  cmp "$dir/out.bin" "$dir/kern.bin"
done
