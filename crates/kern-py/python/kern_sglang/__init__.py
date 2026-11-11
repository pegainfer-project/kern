"""A kern manifest as an SGLang model.

SGLang keeps everything it owns: the scheduler, the radix cache, the KV and
recurrent-state pools, the CUDA graphs, sampling. kern replaces the model:
its forward is one or more programs of a manifest whose states are host
states, bound to the tensors SGLang allocated for each layer.

    pip install kern-llm
    KERN_MANIFEST=<manifest.json> [KERN_KERNELS=<cubin dir>] \\
      sglang serve --model-path <checkpoint> --page-size 64 --attention-backend trtllm_mha \\
        --mamba-radix-cache-strategy extra_buffer --disable-prefill-cuda-graph ...

With `KERN_MANIFEST` set, kern's class stands under every architecture name
SGLang knows, so whatever checkpoint SGLang loads runs as the manifest while
SGLang still treats the model as what its config names (pools, cache policy,
defaults). Without it the plugin does nothing. Kernels come from
`KERN_KERNELS` or, unset, from the registry the manifest's modules name. A
flag that gives SGLang pools of another layout than the manifest's (page
size, KV backend, state dtype) fails when kern binds them, naming the state.
The manifest
(`tools/qwen38_sglang.py` writes one; K3's comes from kern-k3-pp-prefill's
`k3_sglang.py`) follows a small contract, see `model.py`.

With `KERN_TRACE_DIR=<dir>` every kernel the model's process runs is kept
under that directory (a replica of `--dp-size` under its own `dp=<i>`),
named by the manifest call that launched it, in the files kern-serve's
`--trace-dir` writes (`kern.trace`). SGLang calls
`register` in every process before it loads the model, which is before the
process's first CUDA context, where the tracer has to start. SGLang logs a
plugin's error and serves on, so the class is registered first and asks for
the tracer again before it loads: one that cannot start stops the server,
with its reason.
"""
import os


def register() -> None:
    if not os.environ.get("KERN_MANIFEST"):
        return
    from sglang.srt.models.registry import ModelRegistry

    from kern_sglang.model import KernForCausalLM

    ModelRegistry.models.update(dict.fromkeys(ModelRegistry.get_supported_archs(), KernForCausalLM))
    if os.environ.get("KERN_TRACE_DIR"):
        from kern import trace

        trace.start()
