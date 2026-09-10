#!/usr/bin/env bash
set -euo pipefail
upstream=${1:?DeepGEMM}
out=${2:?artifacts}
python3 - "$upstream" <<'PIN'
from pathlib import Path
import sys
root=Path(sys.argv[1])/'.git'
if root.is_file():root=Path(root.read_text().strip().removeprefix('gitdir: '))
head=(root/'HEAD').read_text().strip()
if head.startswith('ref: '):head=(root/head[5:]).read_text().strip()
assert head=='ab69f76be5bb9ea3499bc755002b1a876cb0b3d9',head
PIN
src=$(cd "$(dirname "$0")" && pwd)
common=(-std=c++20 -O3 --expt-relaxed-constexpr -DDG_NO_TORCH -gencode arch=compute_103a,code=sm_103a -I"$upstream/deep_gemm/include" -I"$upstream/third-party/cutlass/include")
nvcc "${common[@]}" -cubin "$src/sparse_indexer.cu" -o "$out/sparse_indexer.cubin"
nvcc "${common[@]}" "$src/sparse_layout.cu" -o "$out/sparse_layout"
"$out/sparse_layout" > "$out/sparse_layout.json"
