"""`KernForCausalLM`: SGLang's model slot, filled by a kern runtime.

SGLang decides the memory from the checkpoint's config, not from the model:
for a hybrid GDN model it allocates per full-attention layer a K and a V
array (`MHATokenToKVPool`) and per GDN layer a conv and an ssm array
(`MambaPool`), all indexed by its own ids. The manifest declares the same
tensors as host states and the adapter binds them, so the contract is:

- host states `k.l<i>`, `v.l<i>` (`[pages, 64, heads, dim]` over SGLang's
  token arrays), `conv.l<i>` and `ssm.l<i>` (`[slots, ...]`), for model
  layer i.
- inputs `token_ids`, `positions`, `slot_mapping`, `seq_lens`,
  `cu_seqlens_q`, `block_table` (`[seqs, W]` page ids) and
  `gdn.conv_index`, `gdn.ssm_index` (`[state layers, seqs]` slot ids);
  output `hidden`; `head_in` -> `head` -> `logits`.
- programs `prefill` (one sequence, `tokens` rows), `decode_batch` (`seqs`
  sequences of one row), `head` (`seqs` rows of logits), and `once`
  programs to run after binding.

A decode step is `decode_batch` over SGLang's padded batch, which SGLang
captures; everything it reads is on the device, padded rows map to slot -1,
which the kernels skip. An extend step is split on the host's lengths: runs
of one-token rows through `decode_batch`, every longer row through
`prefill` alone.

Prefix caching (`extra_buffer`) needs snapshots of the recurrent state at
positions the scheduler picks; restoring one into a request's slot happens
before the forward and is SGLang's. In an extend the snapshot is at a
64-aligned offset of the row, so the row runs as two `prefill` calls cut
there and the slot is copied to the track slot in between. In a decode the
whole slot is copied after the step for rows whose length crossed the track
interval, with SGLang's own masked copy, so it stays inside the graph.
"""
import functools
import itertools
import json
import os
import re

import torch
from torch import nn

import kern
from sglang.kernels.ops.mamba.mamba_state_scatter_triton import track_mamba_states_all_layers
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_req_to_token_pool,
    get_token_to_kv_pool,
)
from sglang.srt.runtime_context import mamba_cache_chunk_size

DTYPES = {"bf16": torch.bfloat16, "u8": torch.uint8, "f32": torch.float32, "i32": torch.int32, "i64": torch.int64}
INPUTS = ("token_ids", "positions", "slot_mapping", "seq_lens", "cu_seqlens_q", "block_table",
          "gdn.conv_index", "gdn.ssm_index", "hidden", "head_in", "logits")


@functools.cache
def manifest() -> dict:
    return json.load(open(os.environ["KERN_MANIFEST"]))


def resolve(m: dict, dim):
    return m.get("constants", {}).get(dim, dim) if isinstance(dim, str) else dim


def host_states(m: dict) -> list[tuple[str, str, int, dict]]:
    """(state, kind, layer, host tensor), in layer order."""
    def parse(n):
        kind, layer = re.fullmatch(r"(\w+)\.l(\d+)", n).groups()
        return kind, int(layer)
    return sorted(((n, *parse(n), s["host"]) for n, s in m["states"].items() if "host" in s), key=lambda h: h[2])


class Bytes:
    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (ptr, False), "version": 3}


def region(rt, m: dict, name: str) -> torch.Tensor:
    ptr, nbytes = rt.region(name)
    b = m["buffers"][name]
    shape = [resolve(m, m["vars"][d]["max"] if d in m["vars"] else d) for d in b["shape"]]
    return torch.as_tensor(Bytes(ptr, nbytes), device="cuda").view(DTYPES[b["dtype"]]).view(*shape)


def host_region(t: torch.Tensor) -> tuple:
    name = {v: k for k, v in DTYPES.items()}[t.dtype]
    return t.data_ptr(), name, list(t.shape), list(t.stride())


def runs(qlens: list[int], decode_rows: int) -> list[tuple[int, int]]:
    """Row ranges one program call each: runs of up to `decode_rows`
    one-token rows, and every longer row alone."""
    out: list[tuple[int, int]] = []
    for r, q in enumerate(qlens):
        if q == 1 and out and out[-1][1] == r and qlens[out[-1][0]] == 1 and r - out[-1][0] < decode_rows:
            out[-1] = (out[-1][0], r + 1)
        elif q > 0:
            out.append((r, r + 1))
    return out


def cuts(n: int, prefix: int, track: int, chunk: int) -> tuple[list[int], int | None]:
    """A row's prefill call boundaries and the one after which its state is
    snapshotted: SGLang's tracked length, floored to its chunk grid
    relative to the prefix (its `mamba_track_aligned_lens`); no snapshot
    when the row is not tracked (`track` < 0)."""
    if track < 0:
        return [n], None
    at = (track - prefix) // chunk * chunk
    return ([at, n] if 0 < at < n else [n]), at


class Head(LogitsProcessor):
    """SGLang's logits path (pruning, logprobs, fp32) with kern's `head` as the lm_head."""

    def __init__(self, config, head):
        super().__init__(config)
        self.head = head

    def _compute_lm_head(self, hidden_states, lm_head, embedding_bias=None):
        return self.head(hidden_states)


class KernForCausalLM(nn.Module):
    def __init__(self, config, quant_config=None, prefix: str = "", **_):
        super().__init__()
        m = manifest()
        self.m = m
        self.config = config.get_text_config()
        self.rt = kern.Runtime(os.environ["KERN_MANIFEST"], os.environ["KERN_KERNELS"],
                               [config._name_or_path], torch.cuda.current_device())
        self.states = host_states(m)
        self.b = {n: region(self.rt, m, n) for n in INPUTS}
        self.decode_rows = resolve(m, m["programs"]["decode_batch"]["batch"]["groups"])
        self.width = self.b["block_table"].shape[1]
        self.page = resolve(m, m["buffers"]["block_table"]["domain"]["stride"])
        self.offsets = torch.arange(self.b["cu_seqlens_q"].shape[0], dtype=torch.int32, device="cuda")
        self.logits_processor = Head(self.config, self.head)
        self.lm_head = nn.Identity()
        self.start_layer, self.end_layer = 0, self.config.num_hidden_layers
        self.bound = None

    def load_weights(self, weights) -> set[str]:
        return set()

    def tensors(self) -> dict[str, torch.Tensor]:
        """Each host state's tensor in SGLang's pools."""
        kv = get_token_to_kv_pool()
        full, at = kv.full_kv_pool, kv.full_attention_layer_id_mapping
        mamba = get_req_to_token_pool().mamba_pool
        cache, slot = mamba.mamba_cache, {l: i for i, l in enumerate(mamba.mamba_layer_ids)}
        view = {
            "k": lambda l, h: full.k_buffer[at[l]].view(-1, *h["shape"][1:]),
            "v": lambda l, h: full.v_buffer[at[l]].view(-1, *h["shape"][1:]),
            "conv": lambda l, h: cache.conv[0][slot[l]],
            "ssm": lambda l, h: cache.temporal[slot[l]],
        }
        return {n: view[kind](l, h) for n, kind, l, h in self.states}

    def pools(self) -> tuple[int, int]:
        """Where SGLang's pools start: a change means it reallocated them."""
        return (get_token_to_kv_pool().full_kv_pool.k_buffer[0].data_ptr(),
                get_req_to_token_pool().mamba_pool.mamba_cache.temporal.data_ptr())

    def bind(self) -> None:
        """Bind (again, when SGLang reallocated) the pools' tensors."""
        first = self.bound is None
        self.rt.bind_host({n: host_region(x) for n, x in self.tensors().items()})
        for name, p in sorted(self.m["programs"].items()):
            if first and p.get("once"):
                self.rt.run(name, {v: 1 for v in self.m["vars"]})
        self.bound = self.pools()

    def forward(self, input_ids, positions, forward_batch, **_):
        fb = forward_batch
        if not torch.cuda.is_current_stream_capturing() and self.bound != self.pools():
            self.bind()
        meta = get_attn_backend().linear_attn_backend.forward_metadata
        lines, hidden = meta.mamba_cache_indices, torch.empty(input_ids.shape[0], self.b["hidden"].shape[1],
                                                               dtype=torch.bfloat16, device=input_ids.device)
        pages = get_req_to_token_pool().req_to_token
        step = functools.partial(self.run, fb, input_ids, positions, lines, pages, hidden)
        if fb.forward_mode.is_decode():
            n = input_ids.shape[0]
            for r0 in range(0, n, self.decode_rows):
                r1 = min(r0 + self.decode_rows, n)
                step("decode_batch", r0, r1, r0, r1, 0)
            if fb.mamba_track_mask is not None:
                self.track(lines, fb.mamba_track_mask, meta.mamba_track_indices, n)
            return self.logits_processor(input_ids, hidden, self.lm_head, fb)
        if not fb.forward_mode.is_extend():
            raise NotImplementedError(f"kern runs decode and extend steps, not {fb.forward_mode}")
        lens, prefix = fb.extend_seq_lens_cpu, fb.extend_prefix_lens_cpu
        tracks = fb.mamba_track_seqlens_cpu or [-1] * len(lens)
        starts = [0, *itertools.accumulate(lens)]
        for r0, r1 in runs(lens, self.decode_rows):
            if lens[r0] == 1:
                step("decode_batch", r0, r1, starts[r0], starts[r1], 0)
                continue
            ends, snap = cuts(lens[r0], prefix[r0], tracks[r0], mamba_cache_chunk_size())
            for a, e in zip([0, *ends], ends):
                step("prefill", r0, r1, starts[r0] + a, starts[r0] + e, lens[r0] - e)
                if e == snap:
                    self.track(lines[r0:r1], fb.mamba_track_mask[r0:r1], fb.mamba_track_indices[r0:r1], 1)
        return self.logits_processor(input_ids, hidden, self.lm_head, fb)

    def run(self, fb, input_ids, positions, lines, pages, hidden, program, r0, r1, t0, t1, ahead) -> None:
        """One program call over rows [r0, r1) and tokens [t0, t1); `ahead`
        tokens of the rows' extends come after this call."""
        rows, k, b = r1 - r0, t1 - t0, self.b
        b["token_ids"][:k].copy_(input_ids[t0:t1])
        b["positions"][:k].copy_(positions[t0:t1])
        b["slot_mapping"][:k].copy_(fb.out_cache_loc[t0:t1])
        b["seq_lens"][:rows].copy_(fb.seq_lens[r0:r1] - ahead)
        b["cu_seqlens_q"].copy_(torch.clamp(self.offsets * (k // rows), max=k))
        first_slots = pages[fb.req_pool_indices[r0:r1], : self.width * self.page : self.page]
        b["block_table"][:rows].copy_(first_slots // self.page)
        b["gdn.conv_index"][:, :rows].copy_(lines[r0:r1].expand(b["gdn.conv_index"].shape[0], -1))
        b["gdn.ssm_index"][:, :rows].copy_(lines[r0:r1].expand(b["gdn.ssm_index"].shape[0], -1))
        self.rt.enqueue_after(program, {"tokens": k, "seqs": rows}, torch.cuda.current_stream().cuda_stream)
        hidden[t0:t1].copy_(b["hidden"][:k])

    def track(self, lines, mask, dest, n) -> None:
        """Copies every GDN layer's state of the masked rows to their track slots."""
        cache = get_req_to_token_pool().mamba_pool.mamba_cache
        track_mamba_states_all_layers(cache.conv[0], cache.temporal, lines, mask, dest, n, check_freed_slots=True)

    def head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        n = hidden_states.shape[0]
        out = torch.empty(n, self.b["logits"].shape[1], dtype=torch.bfloat16, device=hidden_states.device)
        stream = torch.cuda.current_stream().cuda_stream
        step = self.b["head_in"].shape[0]
        for i in range(0, n, step):
            k = min(step, n - i)
            self.b["head_in"][:k].copy_(hidden_states[i : i + k])
            self.rt.enqueue_after("head", {"seqs": k}, stream)
            out[i : i + k].copy_(self.b["logits"][:k])
        return out
