#!/usr/bin/env bash
set -euo pipefail
runner=${1:?program_io binary}
artifacts=${2:?artifact directory from build.sh and compare.py --dump}
for mode in decode prefill dspark_noncausal verify; do
  case "$mode" in decode) rows=1;; prefill) rows=17;; dspark_noncausal) rows=10;; verify) rows=12;; esac
  dir="$artifacts/io/$mode"
  "$runner" --manifest "$dir/manifest.json" --cubins "$artifacts" --vars rows="$rows" \
    --in q="$dir/q.bin" --in kv="$dir/kv.bin" --in indices="$dir/indices.bin" \
    --in sink="$dir/sink.bin" --in lengths="$dir/lengths.bin" --out out="$dir/kern.bin"
  cmp "$dir/out.bin" "$dir/kern.bin"
done
