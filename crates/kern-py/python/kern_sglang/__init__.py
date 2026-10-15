"""A kern manifest as an SGLang model.

SGLang keeps everything it owns: the scheduler, the radix cache, the KV and
recurrent-state pools, the CUDA graphs, sampling. kern replaces the model:
its forward is one or more programs of a manifest whose states are host
states, bound to the tensors SGLang allocated for each layer.

    pip install kern-llm
    KERN_MANIFEST=<manifest.json> KERN_KERNELS=<cubin dir> \\
      sglang serve --model-path <checkpoint> --page-size 64 --attention-backend trtllm_mha \\
        --mamba-radix-cache-strategy extra_buffer --disable-prefill-cuda-graph ...

The plugin takes the checkpoint's own architecture name (`KERN_ARCHITECTURE`,
e.g. `KimiK3ForConditionalGeneration`), so SGLang still recognizes the model
as what it is (pools, cache policy, defaults) and only the class that runs it
changes. It does nothing unless `KERN_MANIFEST` is set. The manifest
(`tools/qwen38_sglang.py` writes one; K3's comes from kern-k3-pp-prefill's
`k3_sglang.py`) follows a small contract, see `model.py`.
"""
import os

ARCHITECTURE = os.environ.get("KERN_ARCHITECTURE", "Qwen3_5ForConditionalGeneration")


def register() -> None:
    if not os.environ.get("KERN_MANIFEST"):
        return
    from sglang.srt.models.registry import ModelRegistry

    from kern_sglang.model import KernForCausalLM

    ModelRegistry.models[ARCHITECTURE] = KernForCausalLM
