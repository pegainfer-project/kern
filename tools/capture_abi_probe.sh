#!/usr/bin/env bash
# Capture a short attention-backend run to inspect kernel parameter layouts.
set -euo pipefail
[[ $# -eq 2 ]] || { echo "usage: $0 BACKEND CHECKPOINT" >&2; exit 2; }
backend="$1"
repo="${KERN_REPO:-$HOME/kern}"
export PATH="$repo/.venv/bin:${CUDA_HOME:-/usr/local/cuda}/bin:$PATH"
export CUDA_INJECTION64_PATH="$repo/tools/kernel-capture/libkernelcapture.so"
export KERNEL_CAPTURE_DIR="$repo/dumped-kernels/abi-$backend"
export HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
# vLLM 0.28 selects the backend through AttentionConfig.
export ABI_PROBE_BACKEND="$backend"
mkdir -p "$KERNEL_CAPTURE_DIR"
"$repo/.venv/bin/python" - "$2" <<'PY'
import os
import sys
from vllm import LLM, SamplingParams
from vllm.config import AttentionConfig

llm = LLM(model=sys.argv[1], enforce_eager=True,
          gpu_memory_utilization=0.45, max_model_len=4096,
          attention_config=AttentionConfig(backend=os.environ["ABI_PROBE_BACKEND"]))
out = llm.generate(["The lighthouse keeper logged every passing ship in a worn ledger."],
                   SamplingParams(max_tokens=4))
print("OK", repr(out[0].outputs[0].text))
PY
