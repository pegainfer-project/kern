"""Checkpoint-bound original DSpark, with lazy quantized experts.

The implementation delegates all forward math to the supplied inference Python.
Only construction/loading is specialized to avoid allocating the target model.
Instantiate one oracle per process: the reference model uses module globals.
"""
import contextlib
import importlib
import json
import pathlib
import sys
import types

import torch
from safetensors import safe_open


class DraftOracle:
    def __init__(self, checkpoint, *, max_batch_size=1, max_seq_len=32768, device='cuda'):
        self.checkpoint = pathlib.Path(checkpoint)
        self.device = torch.device(device)
        sys.path.insert(0, str(self.checkpoint / 'inference'))
        self.model = model = importlib.import_module('model')
        if torch.distributed.is_initialized():
            raise ValueError('The conditional oracle requires one unsharded process')
        model.world_size, model.rank = 1, 0
        model.default_dtype = torch.float8_e4m3fn
        config = json.loads((self.checkpoint / 'inference/config.json').read_text())
        config.update(max_batch_size=max_batch_size, max_seq_len=max_seq_len, temperature=0)
        self.args = args = model.ModelArgs(**config)
        self.index = json.loads((self.checkpoint / 'model.safetensors.index.json').read_text())['weight_map']
        self.files = {}
        self.stack = contextlib.ExitStack()
        original_moe = model.MoE
        owner = self

        class LazyMoE(torch.nn.Module):
            def __init__(self, layer_id, args):
                super().__init__()
                prefix = f'mtp.{layer_id - args.n_layers}.ffn.'
                count, topk = args.get_moe_config(layer_id)
                gate = types.SimpleNamespace(
                    weight=owner.get(prefix + 'gate.weight'),
                    bias=owner.get(prefix + 'gate.bias'), bias_vl=None,
                    gate_temp=args.gate_temp, score_func=args.score_func,
                    topk=topk, norm_topk_prob=args.norm_topk_prob, route_scale=args.route_scale)
                cache = {}

                def expert(eid=None):
                    if eid in cache:
                        return cache[eid]
                    path = prefix + ('shared_experts' if eid is None else f'experts.{eid}')
                    funcs = {}
                    for name in ('w1', 'w2', 'w3'):
                        dtype = torch.float8_e4m3fn if eid is None else torch.float4_e2m1fn_x2
                        weight = owner.get(path + '.' + name + '.weight').view(dtype)
                        weight.scale = owner.get(path + '.' + name + '.scale').view(torch.float8_e8m0fnu)
                        funcs[name] = lambda x, w=weight: model.linear(x, w)
                    context = types.SimpleNamespace(swiglu_limit=args.swiglu_limit, **funcs)
                    cache[eid] = lambda x, weights=None: model.Expert.forward(context, x, weights)
                    return cache[eid]

                class Experts:
                    def __getitem__(self, eid):
                        return expert(eid)

                self.context = types.SimpleNamespace(
                    dim=args.dim, n_routed_experts=count, experts_start_idx=0,
                    experts_end_idx=count, experts=Experts(), shared_experts=expert(),
                    gate=lambda x, mask: model.Gate.forward(gate, x, mask))
                self.expert_cache = cache

            def forward(self, x, image_mask=None):
                return original_moe.forward(self.context, x, image_mask)

        # Reference constructors create all experts eagerly. Substitute construction
        # only; LazyMoE.forward still invokes the original MoE.forward.
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            model.MoE = LazyMoE
            with torch.device(self.device):
                blocks = [model.DSparkBlock(args.n_layers + i, args) for i in range(args.n_mtp_layers)]
                embed = model.ParallelEmbedding(args.vocab_size, args.dim)
                head = model.ParallelHead(args.vocab_size, args.dim, args.norm_eps, args.hc_eps)
            for i, block in enumerate(blocks):
                self.load_module(block, f'mtp.{i}.')
            self.load_module(embed, 'embed.')
            self.load_module(head, 'head.')
            for block in blocks:
                block.embed, block.head = embed, head
            self.context = types.SimpleNamespace(mtp=blocks, hc_mult=args.hc_mult)
        finally:
            model.MoE = original_moe
            torch.set_default_dtype(previous_dtype)

    def get(self, name):
        shard = self.index[name]
        if shard not in self.files:
            self.files[shard] = self.stack.enter_context(
                safe_open(self.checkpoint / shard, framework='pt', device='cpu'))
        # Explicit CPU device prevents a caller's CUDA default from materializing
        # entire mapped checkpoint tensors on the wrong GPU.
        with torch.device('cpu'):
            source = self.files[shard].get_tensor(name)
        return source.to(self.device)

    @torch.inference_mode()
    def load_module(self, module, prefix):
        for name, param in module.named_parameters():
            source = self.get(prefix + name)
            if name.endswith('wo_a.weight'):
                scale = self.get(prefix + name.removesuffix('weight') + 'scale')
                source = (source.float() * scale.float().repeat_interleave(32, 0).repeat_interleave(32, 1)).bfloat16()
            if source.shape != param.shape:
                raise ValueError(f'{prefix + name}: checkpoint {source.shape}, reference {param.shape}')
            param.copy_(source)

    @torch.inference_mode()
    def draft(self, anchor_ids, main_hidden, start_pos=0):
        anchor_ids = anchor_ids.to(device=self.device, dtype=torch.int64).reshape(-1)
        main_hidden = main_hidden.to(device=self.device, dtype=torch.bfloat16)
        if main_hidden.ndim != 3 or main_hidden.shape[0] != anchor_ids.numel() or main_hidden.shape[2] != self.args.dim * len(self.args.dspark_target_layer_ids):
            raise ValueError('main_hidden must be [batch, sequence, 15360] in tap order 37,38,39')
        if start_pos > 0 and main_hidden.shape[1] != 1:
            raise ValueError('Reference incremental DSpark requires exactly one main token')
        # Reference GEMM output dtype and top-k index allocation use defaults.
        with torch.device(self.device), self.model.set_dtype(torch.bfloat16):
            return self.model.Transformer.forward_spec(self.context, anchor_ids, main_hidden, start_pos)

    def close(self):
        self.context = None
        self.files.clear()
        self.stack.close()
