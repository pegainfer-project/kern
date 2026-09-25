"""A kern manifest as a vLLM model.

vLLM keeps everything it owns: the scheduler, the block pool and the KV
cache memory, the input buffers, the CUDA graphs, sampling. kern replaces
the model: its forward is one or more programs of a manifest whose states
are host states, bound to the pages vLLM allocated for each layer.

    pip install kern kern-vllm
    KERN_MANIFEST=<manifest.json> KERN_KERNELS=<cubin dir> \\
      vllm serve <checkpoint> --hf-overrides '{"architectures": ["KernForCausalLM"]}' ...

The manifest (`tools/qwen38_vllm.py` writes one) follows a small contract,
see `model.py`: host states `kv.l<i>` / `gdn.l<i>` per layer, programs
`prefill` (one sequence), `decode_batch` (one row per sequence) and `head`.
"""


def register() -> None:
    from vllm import ModelRegistry

    if "KernForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("KernForCausalLM", "kern_vllm.model:KernForCausalLM")
