# kern-llm

[kern](https://github.com/pegainfer-project/kern) runs a model as a manifest: a typed declaration of
buffers, states, ops and programs, compiled kernels, and weights. This package is kern for Python:

- `kern`: the runtime binding (`kern.Runtime`), the readers of a manifest's host contract
  (`kern.host`) and the rank group over `torch.distributed` (`kern.group`);
- `kern_vllm`: a vLLM plugin that runs a manifest as the model;
- `kern_sglang`: the same for SGLang.

```bash
pip install kern-llm
hf download Pegainfer/kern-qwen38-sm103 --local-dir kern-qwen38
KERN_MANIFEST=kern-qwen38/manifests/qwen3.8-27b-vllm.json KERN_KERNELS=kern-qwen38/cubins \
vllm serve Qwen/Qwen3.8-27B --hf-overrides '{"architectures": ["KernForCausalLM"]}' \
  --max-num-seqs 128 --max-num-batched-tokens 8192 --enable-prefix-caching \
  -cc '{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}'
```

Linux x86_64 and aarch64, Python 3.10 or newer. The runtime loads the NVIDIA driver (CUDA 13, r580 or
newer) and cuBLAS 13 when it first runs; torch comes from the engine. The plugins do nothing unless
`KERN_MANIFEST` is set. Guides: [vLLM](https://github.com/pegainfer-project/kern/blob/master/docs/vllm.md),
[SGLang](https://github.com/pegainfer-project/kern/blob/master/docs/sglang.md).
