"""`KernForCausalLM`: vLLM's model slot, filled by a kern runtime.

The manifest contract, read from the manifest and nothing else:

- a host state per cache layer, `<kind>.l<i>` for model layer i: a rank-4
  `[B, H, N, C]` view is an attention layer, `kv` packing k | v in C, `mla`
  one head of a latent (vLLM's `MLAAttentionSpec`); a rank-2 `[B, bytes]`
  view is a recurrent state page. The layer stubs below turn each into the
  vLLM spec that allocates exactly that view.
- inputs, each copied from vLLM's metadata if the manifest declares it:
  `token_ids`, `positions`, `slot_mapping`, `seq_lens` (the positions this
  rank holds: vLLM's DCP-local lengths under decode context parallelism),
  `cu_seqlens_q`, `block_table` (`[seqs, W]` kernel blocks), and the line
  table indexing the state pages (`[state layers, seqs]`, rows in layer
  order); output `hidden`; `head_in` -> `head` -> `logits`.
- programs `decode_batch` (`seqs` sequences of one row), `head` (`seqs`
  rows of logits), and `once` programs to run after binding; optionally
  `prefill` (`seqs` sequences of any lengths, `tokens` rows in all, split
  by `cu_seqlens_q`). Without it the manifest only decodes, and a prompt
  reaches the cache another way (a KV connector).

A step is split into runs the programs accept: consecutive one-token rows go
through `decode_batch` together, consecutive longer rows through one
`prefill`, so the step's prefill rows share every weight read.
A pure decode step is one `decode_batch` over vLLM's padded batch, so vLLM
can capture it; padded rows have no slot and the null state page, which
the kernels skip. A sequence's first chunk finds its state pages zeroed
here, a one-token prompt's too: vLLM hands out pages without clearing them
and reuses a freed one first.

A manifest with a topology runs one rank per vLLM tensor-parallel rank:
member `i` of every topology group is TP rank `i`. The ranks trade offers
over vLLM's gloo TP group when the model is built; on the first bind they
run the `once` programs and meet at a barrier. `logits` is then this
rank's contiguous slice of the vocab, gathered over the TP group.

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
from kern import group
from kern.host import DTYPES, host_region, region, resolve
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_tp_group, tensor_model_parallel_all_gather
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateCopyFuncCalculator
from vllm.model_executor.models.interfaces import IsHybrid, SupportsMRoPE
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import FullAttentionSpec, MLAAttentionSpec

from kern_vllm.backend import KernAttentionBackend, KernMLABackend, KernStateBackend

logger = init_logger(f"vllm.{__name__}")
INPUTS = ("token_ids", "positions", "slot_mapping", "seq_lens", "cu_seqlens_q", "block_table")


@functools.cache
def manifest() -> dict:
    return json.load(open(os.environ["KERN_MANIFEST"]))


def host_states(m: dict) -> list[tuple[str, int, dict]]:
    """(state, layer, host tensor), in layer order."""
    hosts = [(n, int(re.fullmatch(r".*\.l(\d+)", n)[1]), s["host"]) for n, s in m["states"].items() if "host" in s]
    return sorted(hosts, key=lambda h: h[1])


def state_pages(m: dict) -> list[tuple[str, int, dict]]:
    return [h for h in host_states(m) if len(h[2]["shape"]) == 2]


def line_table(m: dict) -> str | None:
    """The input indexing the state pages, if any."""
    pages = {n for n, _, _ in state_pages(m)}
    return next((n for n, b in m["buffers"].items()
                 if b["kind"] == "input" and b.get("domain", {}).get("index_into") in pages), None)


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


class KernMLA(nn.Module, AttentionLayerBase):
    def __init__(self, prefix: str, host: dict):
        super().__init__()
        _, heads, self.block, self.head_size = host["shape"]
        if heads != 1:
            raise ValueError(f"MLA state `{prefix}` of {heads} heads; a latent cache has one")
        self.dtype = DTYPES[host["dtype"]]
        self.kv_cache = torch.tensor([])
        self.impl = KernMLABackend.get_impl_cls()()
        register(prefix, self)

    def get_attn_backend(self):
        return KernMLABackend

    @property
    def view(self) -> torch.Tensor:
        return self.kv_cache

    def get_kv_cache_spec(self, vllm_config: VllmConfig):
        c = vllm_config.cache_config
        return MLAAttentionSpec(block_size=c.block_size, num_kv_heads=1, max_tp_shards=1, head_size=self.head_size,
                                dtype=self.dtype, cache_dtype_str=c.cache_dtype)


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


def stub(name: str, i: int, host: dict) -> nn.Module:
    if len(host["shape"]) == 2:
        return KernState(f"model.layers.{i}.linear_attn", host)
    attention = {"kv": KernKV, "mla": KernMLA}[name.split(".")[0]]
    return attention(f"model.layers.{i}.self_attn.attn", host)


def register(prefix: str, layer) -> None:
    context = get_current_vllm_config().compilation_config.static_forward_context
    if prefix in context:
        raise ValueError(f"duplicate layer {prefix}")
    context[prefix] = layer


class KernForCausalLM(nn.Module, IsHybrid, SupportsMRoPE):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        m = manifest()
        self.m = m
        self.group = get_tp_group().cpu_group if m.get("topology") else None
        self.rt = kern.Runtime(os.environ["KERN_MANIFEST"], os.environ["KERN_KERNELS"],
                               [vllm_config.model_config.model], torch.cuda.current_device(),
                               None if self.group is None else group.topology(m, self.group))
        if self.group is not None:
            group.connect(self.rt, m, self.group)
        self.layers = nn.ModuleDict({f"l{i}": stub(n, i, h) for n, i, h in host_states(m)})
        self.line_table = line_table(m)
        inputs = [n for n in (*INPUTS, self.line_table) if n in m["buffers"]]
        self.b = {n: region(self.rt, m, n) for n in inputs + ["hidden", "head_in", "logits"]}
        self.errors = {n: region(self.rt, m, n) for n, b in m["buffers"].items() if b.get("fill") == "error"}
        self.check = bool(os.environ.get("KERN_CHECK_ERRORS"))
        tokens, width = self.b["hidden"].shape
        self.tokens = tokens
        s = vllm_config.scheduler_config
        if s.max_num_batched_tokens > tokens or s.max_num_seqs > self.b["head_in"].shape[0]:
            raise ValueError(f"`{m['model']}` runs at most {tokens} tokens of {self.b['head_in'].shape[0]} sequences "
                             f"a step; vLLM schedules {s.max_num_batched_tokens} of {s.max_num_seqs}: lower "
                             "--max-num-batched-tokens / --max-num-seqs")
        self.out = torch.zeros(tokens, width, dtype=torch.bfloat16, device="cuda")
        self.vocab = vllm_config.model_config.get_vocab_size()
        self.vocab_slice = self.b["logits"].shape[1]
        self.decode_rows = resolve(m, m["programs"]["decode_batch"]["batch"]["groups"])
        self.prefill_seqs = resolve(m, m["vars"]["seqs"]["max"])
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
        regions = {n: host_region(self.layers[f"l{i}"].view) for n, i, _ in host_states(self.m)}
        self.rt.bind_host(regions)
        logger.info("kern bound %d host states at %#x", len(regions), min(r[0] for r in regions.values()))
        if first:
            group.start(self.rt, self.m, self.group)
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
            # A one-token prompt's first step is a decode row. vLLM runs a step holding a prompt
            # eagerly, never from a decode graph, so a graph never meets a fresh row.
            if not torch.cuda.is_current_stream_capturing():
                self.zero_fresh(attn, states)
            for r0, r1 in chunks(0, attn.num_reqs, self.decode_rows):
                self.run(stream, "decode_batch", input_ids, positions, attn, lines, r0, r1, r0, r1)
            return self.out[:n]
        if "prefill" not in self.m["programs"]:
            raise NotImplementedError(
                f"`{self.m['model']}` only decodes (no `prefill` program), but a step has rows of up to "
                f"{attn.max_query_len} tokens: fill the prompt's cache with a KV connector")
        self.zero_fresh(attn, states)
        qsl = attn.query_start_loc_cpu.numpy()
        for r0, r1 in runs(np.diff(qsl[: attn.num_reqs + 1]), self.decode_rows, self.prefill_seqs, self.tokens):
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
        """Zero the state pages of the rows that start a sequence (a host sync)."""
        r = attn.num_reqs
        fresh = (attn.seq_lens[:r] == attn.query_start_loc[1 : r + 1] - attn.query_start_loc[:r]).nonzero()[:, 0]
        if fresh.numel():
            for layer, s in zip(self.state_layers, states):
                layer.view.index_fill_(0, self.pages(s, r)[fresh].long(), 0)

    def run(self, stream, program, input_ids, positions, attn, lines, r0, r1, t0, t1) -> None:
        rows, k = r1 - r0, t1 - t0
        b = self.b
        held = attn.seq_lens if attn.dcp_local_seq_lens is None else attn.dcp_local_seq_lens
        sources = {"token_ids": input_ids[t0:t1], "positions": positions[t0:t1],
                   "slot_mapping": attn.slot_mapping[t0:t1], "seq_lens": held[r0:r1]}
        for name, src in sources.items():
            if name in b:
                b[name][: src.shape[0]].copy_(src)
        if "cu_seqlens_q" in b:
            cu = b["cu_seqlens_q"]
            cu[: rows + 1].copy_(attn.query_start_loc[r0 : r1 + 1] - attn.query_start_loc[r0])
            cu[rows + 1 :].copy_(cu[rows].expand(cu.shape[0] - rows - 1))
        if "block_table" in b:
            # vLLM pads its table past max_model_len; columns beyond the manifest's are never reached.
            width = min(attn.block_table_tensor.shape[1], b["block_table"].shape[1])
            b["block_table"][:rows, :width].copy_(attn.block_table_tensor[r0:r1, :width])
        if self.line_table is not None:
            b[self.line_table][:, :rows].copy_(lines[:, r0:r1])
        self.rt.enqueue_after(program, {"tokens": k, "seqs": rows}, stream)
        self.out[t0:t1].copy_(b["hidden"][:k])

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """`head` over the rows vLLM samples; a sliced vocab is gathered over the TP group."""
        n = hidden_states.shape[0]
        out = torch.zeros(n, self.vocab_slice, dtype=self.b["logits"].dtype, device=hidden_states.device)
        if self.bound is not None:
            stream = torch.cuda.current_stream().cuda_stream
            step = self.b["head_in"].shape[0]
            for i in range(0, n, step):
                k = min(step, n - i)
                self.b["head_in"][:k].copy_(hidden_states[i : i + k])
                self.rt.enqueue_after("head", {"seqs": k}, stream)
                out[i : i + k].copy_(self.b["logits"][:k])
            if self.check:
                self.raise_errors()
        if self.vocab_slice < self.vocab:
            out = tensor_model_parallel_all_gather(out, dim=-1)
        if self.check and self.bound is not None:
            logger.info_once("kern logits %s %s", tuple(out[:, : self.vocab].shape), str(out.dtype), scope="local")
        return out[:, : self.vocab]

    def raise_errors(self) -> None:
        """Every error output's word after the step (`KERN_CHECK_ERRORS`: a sync a step)."""
        words = {n: int(t.view(-1)[0]) for n, t in self.errors.items()}
        logger.info_once("kern error words %s", str(words), scope="local")
        if any(words.values()):
            raise RuntimeError(f"kern error words {words}")


def chunks(r0: int, r1: int, size: int) -> list[tuple[int, int]]:
    return [(r, min(r + size, r1)) for r in range(r0, r1, size)]


def runs(qlens: np.ndarray, decode_rows: int, prefill_seqs: int, tokens: int) -> list[tuple[int, int]]:
    """Row ranges one program call each: runs of up to `decode_rows`
    one-token rows, and runs of longer rows up to `prefill_seqs` rows and
    `tokens` tokens."""
    def joins(r0: int, r: int) -> bool:
        if (qlens[r] == 1) != (qlens[r0] == 1):
            return False
        if qlens[r] == 1:
            return r - r0 < decode_rows
        return r - r0 < prefill_seqs and qlens[r0 : r + 1].sum() <= tokens

    out: list[tuple[int, int]] = []
    for r, q in enumerate(qlens):
        if q > 0 and out and out[-1][1] == r and joins(out[-1][0], r):
            out[-1] = (out[-1][0], r + 1)
        elif q > 0:
            out.append((r, r + 1))
    return out
