"""`KernForCausalLM`: vLLM's model slot, filled by a kern runtime.

The manifest contract, read from the manifest and nothing else:

- a host state per cache layer, `<kind>.l<i>` for model layer i: a rank-4
  `[B, H, N, C]` view is an attention layer (C packs k | v), a rank-2
  `[B, bytes]` view is a recurrent state page. The layer stubs below turn
  each into the vLLM spec that allocates exactly that view.
- inputs `token_ids`, `positions`, `slot_mapping`, `seq_lens`,
  `cu_seqlens_q`, `block_table` (`[seqs, W]` kernel blocks) and
  `gdn.line_index` (`[state layers, seqs]` state pages, rows in layer order);
  output `hidden`; `head_in` -> `head` -> `logits`.
- programs `prefill` (one sequence, `tokens` rows), `decode_batch` (`seqs`
  sequences of one row), `head` (`seqs` rows of logits), and `once`
  programs to run after binding.

A step is split into runs the programs accept: consecutive one-token rows go
through `decode_batch` together, every longer row through `prefill` alone.
A pure decode step is one `decode_batch` over vLLM's padded batch, so vLLM
can capture it; padded rows have no slot and the null state page, which
the kernels skip. A sequence's first chunk finds its state pages zeroed
here: vLLM hands out pages without clearing them.

Prefix caching runs in vLLM's `align` mode: a sequence holds a state page
per block, and the page a step runs on is its last token's block. vLLM
copies the state there from the previous block before the step (a whole
page, `linear_attention_state_copy_func`), so the programs see one page per
sequence, as without caching.
"""
import functools
import json
import os
import re

import numpy as np
import torch
from torch import nn

import kern
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateCopyFuncCalculator
from vllm.model_executor.models.interfaces import IsHybrid, SupportsMRoPE
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import FullAttentionSpec

from kern_vllm.backend import KernAttentionBackend, KernStateBackend

DTYPES = {"bf16": torch.bfloat16, "u8": torch.uint8, "f32": torch.float32, "i32": torch.int32, "i64": torch.int64}


@functools.cache
def manifest() -> dict:
    return json.load(open(os.environ["KERN_MANIFEST"]))


def resolve(m: dict, dim):
    return m.get("constants", {}).get(dim, dim) if isinstance(dim, str) else dim


def host_states(m: dict) -> list[tuple[str, int, dict]]:
    """(state, layer, host tensor), in layer order."""
    hosts = [(n, int(re.fullmatch(r".*\.l(\d+)", n)[1]), s["host"]) for n, s in m["states"].items() if "host" in s]
    return sorted(hosts, key=lambda h: h[1])


def state_pages(m: dict) -> list[tuple[str, int, dict]]:
    return [h for h in host_states(m) if len(h[2]["shape"]) == 2]


class KernKV(nn.Module, AttentionLayerBase):
    def __init__(self, prefix: str, host: dict):
        super().__init__()
        _, self.num_kv_heads, self.block, c = host["shape"]
        self.head_size = c // 2
        self.dtype = DTYPES[host["dtype"]]
        self.kv_cache = torch.tensor([])
        self.impl = KernAttentionBackend.get_impl_cls()()
        register(prefix, self)

    def get_attn_backend(self):
        return KernAttentionBackend

    @property
    def view(self) -> torch.Tensor:
        return self.kv_cache

    def get_kv_cache_spec(self, vllm_config: VllmConfig):
        return FullAttentionSpec(block_size=vllm_config.cache_config.block_size, num_kv_heads=self.num_kv_heads,
                                 head_size=self.head_size, dtype=self.dtype)


class KernState(nn.Module, MambaBase):
    def __init__(self, prefix: str, host: dict):
        super().__init__()
        self.bytes = host["shape"][1]
        self.kv_cache = (torch.tensor([]),)
        register(prefix, self)

    @property
    def view(self) -> torch.Tensor:
        return self.kv_cache[0]

    def get_state_shape(self):
        return ((self.bytes,),)

    def get_state_dtype(self):
        return (torch.uint8,)

    @property
    def mamba_type(self):
        return MambaAttentionBackendEnum.GDN_ATTN

    def get_attn_backend(self):
        return KernStateBackend


def register(prefix: str, layer) -> None:
    context = get_current_vllm_config().compilation_config.static_forward_context
    if prefix in context:
        raise ValueError(f"duplicate layer {prefix}")
    context[prefix] = layer


def region(rt, m: dict, name: str) -> torch.Tensor:
    ptr, nbytes = rt.region(name)
    b = m["buffers"][name]
    dtype = DTYPES[b["dtype"]]
    shape = [resolve(m, m["vars"][d]["max"] if d in m["vars"] else d) for d in b["shape"]]
    raw = torch.as_tensor(Bytes(ptr, nbytes), device="cuda")
    return raw.view(dtype).view(*shape)


class Bytes:
    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (ptr, False), "version": 3}


def host_region(t: torch.Tensor) -> tuple:
    name = {v: k for k, v in DTYPES.items()}[t.dtype]
    return t.data_ptr(), name, list(t.shape), list(t.stride())


class KernForCausalLM(nn.Module, IsHybrid, SupportsMRoPE):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        m = manifest()
        self.m = m
        self.rt = kern.Runtime(os.environ["KERN_MANIFEST"], os.environ["KERN_KERNELS"],
                               [vllm_config.model_config.model], torch.cuda.current_device())
        self.layers = nn.ModuleDict({
            f"l{i}": (KernKV(f"model.layers.{i}.self_attn.attn", h) if len(h["shape"]) == 4
                      else KernState(f"model.layers.{i}.linear_attn", h))
            for _, i, h in host_states(m)})
        self.b = {n: region(self.rt, m, n) for n in (
            "token_ids", "positions", "slot_mapping", "seq_lens", "cu_seqlens_q", "block_table",
            "gdn.line_index", "hidden", "head_in", "logits")}
        tokens, width = self.b["hidden"].shape
        self.out = torch.zeros(tokens, width, dtype=torch.bfloat16, device="cuda")
        self.vocab = self.b["logits"].shape[1]
        self.decode_rows = resolve(m, m["programs"]["decode_batch"]["batch"]["groups"])
        self.cache_config = vllm_config.cache_config
        self.bound = None
        self.rope_dims = vllm_config.model_config.mrope_num_dims if vllm_config.model_config.uses_mrope else 1

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        return ((state_pages(manifest())[0][2]["shape"][1],),)

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config):
        return (torch.uint8,)

    @classmethod
    def get_mamba_state_copy_func(cls):
        return MambaStateCopyFuncCalculator.linear_attention_state_copy_func()

    def get_mrope_input_positions(self, input_tokens, mm_features):
        """Text only: every M-RoPE channel is the token's position."""
        return torch.arange(len(input_tokens)).expand(self.rope_dims, -1), 0

    def load_weights(self, weights) -> set[str]:
        return set()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("kern embeds inside its programs")

    def bind(self) -> None:
        """Bind (again, when vLLM reallocated) the layers' caches."""
        assert not torch.cuda.is_current_stream_capturing(), "kern binds on an eager step, before capture"
        first = self.bound is None
        # vLLM settles the state block size after the model is built.
        self.state_block = self.cache_config.mamba_block_size
        self.rt.bind_host({n: host_region(self.layers[f"l{i}"].view) for n, i, _ in host_states(self.m)})
        for name, p in sorted(self.m["programs"].items()):
            if first and p.get("once"):
                self.rt.run(name, {v: 1 for v in self.m["vars"]})
        self.state_names = [f"model.layers.{i}.linear_attn" for _, i, _ in state_pages(self.m)]
        self.state_layers = [self.layers[f"l{i}"] for _, i, _ in state_pages(self.m)]
        self.bound = self.layers[next(iter(self.layers))].view

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **_):
        n = input_ids.shape[0]
        positions = positions[0] if positions.dim() == 2 else positions
        meta = get_forward_context().attn_metadata
        cache = self.layers[next(iter(self.layers))].view
        if not meta or cache.numel() == 0:
            return self.out[:n]
        if self.bound is not cache:
            self.bind()
        if any(k not in meta for k in self.state_names):
            return self.out[:n]
        attn = next(meta[k] for k in meta if k.endswith(".attn"))
        states = [meta[k] for k in self.state_names]
        lines = self.lines(states, attn)
        stream = torch.cuda.current_stream().cuda_stream
        if attn.max_query_len == 1:
            for r0, r1 in chunks(0, attn.num_reqs, self.decode_rows):
                self.run(stream, "decode_batch", input_ids, positions, attn, lines, r0, r1, r0, r1)
            return self.out[:n]
        self.zero_fresh(attn, states)
        qsl = attn.query_start_loc_cpu.numpy()
        for r0, r1 in runs(np.diff(qsl[: attn.num_reqs + 1]), self.decode_rows):
            program = "prefill" if qsl[r1] - qsl[r0] > r1 - r0 else "decode_batch"
            self.run(stream, program, input_ids, positions, attn, lines, r0, r1, int(qsl[r0]), int(qsl[r1]))
        return self.out[:n]

    def pages(self, s, r: int) -> torch.Tensor:
        """The state page each row runs on: its last token's block."""
        block = ((s.seq_lens[:r] - 1) // self.state_block).clamp(min=0)
        return s.block_table_tensor[:r].gather(1, block[:, None].long())[:, 0]

    def lines(self, states, attn) -> torch.Tensor:
        """Each state layer's page per row; padded rows (no sequence) get the null page."""
        r = attn.num_reqs
        live = attn.seq_lens[:r] > 0
        return torch.stack([torch.where(live, self.pages(s, r), 0) for s in states])

    def zero_fresh(self, attn, states) -> None:
        r = attn.num_reqs
        fresh = attn.seq_lens[:r] == attn.query_start_loc[1 : r + 1] - attn.query_start_loc[:r]
        for layer, s in zip(self.state_layers, states):
            layer.view.index_fill_(0, torch.where(fresh, self.pages(s, r), 0).long(), 0)

    def run(self, stream, program, input_ids, positions, attn, lines, r0, r1, t0, t1) -> None:
        rows, k = r1 - r0, t1 - t0
        b = self.b
        b["token_ids"][:k].copy_(input_ids[t0:t1])
        b["positions"][:k].copy_(positions[t0:t1])
        b["slot_mapping"][:k].copy_(attn.slot_mapping[t0:t1])
        b["seq_lens"][:rows].copy_(attn.seq_lens[r0:r1])
        b["cu_seqlens_q"][: rows + 1].copy_(attn.query_start_loc[r0 : r1 + 1] - attn.query_start_loc[r0])
        b["cu_seqlens_q"][rows + 1 :].copy_(b["cu_seqlens_q"][rows].expand(b["cu_seqlens_q"].shape[0] - rows - 1))
        table = attn.block_table_tensor[r0:r1]
        b["block_table"][:rows, : table.shape[1]].copy_(table)
        b["gdn.line_index"][:, :rows].copy_(lines[:, r0:r1])
        self.rt.enqueue_after(program, {"tokens": k, "seqs": rows}, stream)
        self.out[t0:t1].copy_(b["hidden"][:k])

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        n = hidden_states.shape[0]
        out = torch.zeros(n, self.vocab, dtype=torch.bfloat16, device=hidden_states.device)
        if self.bound is None:
            return out
        stream = torch.cuda.current_stream().cuda_stream
        step = self.b["head_in"].shape[0]
        for i in range(0, n, step):
            k = min(step, n - i)
            self.b["head_in"][:k].copy_(hidden_states[i : i + k])
            self.rt.enqueue_after("head", {"seqs": k}, stream)
            out[i : i + k].copy_(self.b["logits"][:k])
        return out


def chunks(r0: int, r1: int, size: int) -> list[tuple[int, int]]:
    return [(r, min(r + size, r1)) for r in range(r0, r1, size)]


def runs(qlens: np.ndarray, decode_rows: int) -> list[tuple[int, int]]:
    """Row ranges one program call each: runs of up to `decode_rows`
    one-token rows, and every longer row alone."""
    out: list[tuple[int, int]] = []
    for r, q in enumerate(qlens):
        if q == 1 and out and out[-1][1] == r and qlens[out[-1][0]] == 1 and r - out[-1][0] < decode_rows:
            out[-1] = (out[-1][0], r + 1)
        elif q > 0:
            out.append((r, r + 1))
    return out
