#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 ]]; then
  echo 'usage: build_woa.sh OUTPUT_CUBIN' >&2
  exit 2
fi
src=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p -- "$(dirname -- "$1")"
nvcc -std=c++17 -O3 -arch=sm_103a -cubin "$src/woa.cu" -o "$1"
