#!/usr/bin/env bash
# Capture SGLang kernels through CUPTI. MODEL must name a local checkpoint.
# A real Python file is required because the scheduler uses spawn.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model="$(realpath "${MODEL:?set MODEL to a checkpoint directory}")"
out="${KERNEL_CAPTURE_DIR:-$repo/dumped-kernels/sglang}"
image="${SGLANG_IMAGE:-lmsysorg/sglang:latest}"

[ -f "$repo/tools/kernel-capture/libkernelcapture.so" ] || "$repo/tools/kernel-capture/build.sh"
mkdir -p "$out"

docker run --rm --gpus all --ipc=host --shm-size=32g \
  -v "$repo:$repo" -v "$model:$model:ro" -v "$out:$out" \
  -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  -e CUDA_INJECTION64_PATH="$repo/tools/kernel-capture/libkernelcapture.so" \
  -e KERNEL_CAPTURE_DIR="$out" \
  -e HF_HUB_OFFLINE=1 \
  "$image" python3 "$repo/tools/capture_sglang.py" "$model"

echo "--- capture summary ---"
for d in "$out"/pid*/; do
  echo "$d: $(ls "$d" | grep -c cubin) cubins, $(wc -l < "$d/launches.jsonl") launches"
done
