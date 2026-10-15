"""`KernForCausalLM`: SGLang's model slot, filled by a kern runtime.

SGLang decides the memory from the checkpoint's config, not from the model:
for a hybrid model it allocates per full-attention layer its KV arrays
(`MHATokenToKVPool`'s K and V, or `MLATokenToKVPool`'s latent) and per
linear-attention layer a conv and an ssm array (`MambaPool`), all indexed by
its own ids. The manifest declares the same tensors as host states and the
adapter binds them, so the contract is:

- host states `<kind>.l<i>` for model layer i, by kind: `k`, `v`
  (`[pages, 64, heads, dim]` over SGLang's token arrays), `mla`
  (`[pages, 1, 64, dim]` over the latent's rows), `conv` and `ssm`
  (`[slots, ...]`).
- inputs, each copied if the manifest declares it: `token_ids`,
  `positions`, `slot_mapping`, `seq_lens`, `cu_seqlens_q`, `block_table`
  (`[seqs, W]` page ids), and the line tables (`[state layers, seqs]` slot
  ids, any input whose domain indexes a `conv` or `ssm` state); output
  `hidden`; `head_in` -> `head` -> `logits`.
- programs `decode_batch` (`seqs` sequences of one row), `head` (`seqs`
  rows of logits), `once` programs to run after binding, and optionally
  `prefill` (one sequence, `tokens` rows). Without `prefill` the manifest
  only decodes: an extend step may then hold one-token rows only, and a
  longer prompt reaches the cache through SGLang's P/D.

A decode step is `decode_batch` over SGLang's padded batch, which SGLang
captures; everything it reads is on the device, padded rows map to slot -1,
which the kernels skip. An extend step is split on the host's lengths: runs
of one-token rows through `decode_batch`, every longer row through
`prefill` alone.

A manifest with a topology runs one rank per SGLang tensor-parallel rank:
member `i` of every topology group is TP rank `i`. The ranks trade offers
over SGLang's gloo TP group when the model is built, and run the `once`
programs and meet at a barrier when the pools are first bound, before
SGLang's warmup. `logits` is then this rank's contiguous slice of the vocab,
which SGLang's logits processor gathers over the TP group. Under decode
context parallelism SGLang interleaves a sequence's positions over the DCP
group: its slot `s` is rank `s % w`'s row `s // w`, and a page of `w * 64`
slots is every rank's page of 64 rows. `slot_mapping`, `seq_lens` and
`block_table` are this rank's: the rows it writes, the positions it holds,
its pages.

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
import logging
import os
import re

import torch
from torch import nn

import kern
from kern import group
from kern.host import checkpoint, host_region, region, resolve
from sglang.kernels.ops.mamba.mamba_state_scatter_triton import track_mamba_states_all_layers
from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.logits_processor import LogitsProcessor, get_in_autotune_dummy_run
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.runtime_context import get_parallel, mamba_cache_chunk_size

logger = logging.getLogger(__name__)
INPUTS = ("token_ids", "positions", "slot_mapping", "seq_lens", "cu_seqlens_q", "block_table")
RECURRENT = ("conv", "ssm")


@functools.cache
def manifest() -> dict:
    return json.load(open(os.environ["KERN_MANIFEST"]))


def host_states(m: dict) -> list[tuple[str, str, int, dict]]:
    """(state, kind, layer, host tensor), in layer order."""
    def parse(n):
        kind, layer = re.fullmatch(r"(\w+)\.l(\d+)", n).groups()
        return kind, int(layer)
    return sorted(((n, *parse(n), s["host"]) for n, s in m["states"].items() if "host" in s), key=lambda h: h[2])


def line_tables(m: dict) -> list[str]:
    """The inputs holding a slot id per state layer and sequence."""
    recurrent = {n for n, kind, _, _ in host_states(m) if kind in RECURRENT}
    return sorted(n for n, b in m["buffers"].items()
                  if b["kind"] == "input" and b.get("domain", {}).get("index_into") in recurrent)


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


def local_slots(slots: torch.Tensor, w: int, r: int) -> torch.Tensor:
    """The rows DCP rank `r` of `w` writes: its own slots, -1 for the
    others' and for slot 0, the padded rows' scratch."""
    return slots if w == 1 else torch.where((slots % w == r) & (slots != 0), slots // w, -1)


def local_lens(lens: torch.Tensor, w: int, r: int) -> torch.Tensor:
    """The positions of each sequence DCP rank `r` of `w` holds."""
    return lens // w + (lens % w > r).to(lens.dtype)


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
        self.group = get_tp_group().cpu_group if m.get("topology") else None
        self.rt = kern.Runtime(os.environ["KERN_MANIFEST"], os.environ["KERN_KERNELS"],
                               [checkpoint(config._name_or_path, getattr(config, "_commit_hash", None))],
                               torch.cuda.current_device(),
                               None if self.group is None else group.topology(m, self.group))
        if self.group is not None:
            group.connect(self.rt, m, self.group)
        self.states = host_states(m)
        self.tables = line_tables(m)
        inputs = [n for n in (*INPUTS, *self.tables) if n in m["buffers"]]
        self.b = {n: region(self.rt, m, n) for n in (*inputs, "hidden", "head_in", "logits")}
        self.errors = {n: region(self.rt, m, n) for n, b in m["buffers"].items() if b.get("fill") == "error"}
        self.check = bool(os.environ.get("KERN_CHECK_ERRORS"))
        self.decode_rows = resolve(m, m["programs"]["decode_batch"]["batch"]["groups"])
        if "block_table" in self.b:
            self.width = self.b["block_table"].shape[1]
            self.page = resolve(m, m["buffers"]["block_table"]["domain"]["stride"])
        if "cu_seqlens_q" in self.b:
            self.offsets = torch.arange(self.b["cu_seqlens_q"].shape[0], dtype=torch.int32, device="cuda")
        p = get_parallel()
        self.dcp = (p.dcp_size, p.dcp_rank) if p.dcp_enabled else (1, 0)
        self.logits_processor = Head(self.config, self.head)
        self.lm_head = nn.Identity()
        self.start_layer, self.end_layer = 0, self.config.num_hidden_layers
        self.runner = None
        self.bound = None

    def load_weights(self, weights) -> set[str]:
        return set()

    def tensors(self) -> dict[str, torch.Tensor]:
        """Each host state's tensor in SGLang's pools."""
        kv = self.runner.token_to_kv_pool
        full, at = kv.full_kv_pool, kv.full_attention_layer_id_mapping
        mamba = self.runner.req_to_token_pool.mamba_pool
        cache, slot = mamba.mamba_cache, {l: i for i, l in enumerate(mamba.mamba_layer_ids)}
        view = {
            "k": lambda l, h: full.k_buffer[at[l]].view(-1, *h["shape"][1:]),
            "v": lambda l, h: full.v_buffer[at[l]].view(-1, *h["shape"][1:]),
            "mla": lambda l, h: full.kv_buffer[at[l]].view(-1, h["shape"][2], 1, h["shape"][3]).transpose(1, 2),
            "conv": lambda l, h: cache.conv[0][slot[l]],
            "ssm": lambda l, h: cache.temporal[slot[l]],
        }
        return {n: view[kind](l, h) for n, kind, l, h in self.states}

    def pools(self) -> tuple[int, int]:
        """Where SGLang's pools start: a change means it reallocated them."""
        full = self.runner.token_to_kv_pool.full_kv_pool
        kv = full.kv_buffer if hasattr(full, "kv_buffer") else full.k_buffer
        return kv[0].data_ptr(), self.runner.req_to_token_pool.mamba_pool.mamba_cache.temporal.data_ptr()

    def bind(self) -> None:
        """Bind (again, when SGLang reallocated) the pools' tensors."""
        first = self.bound is None
        regions = {n: host_region(x) for n, x in self.tensors().items()}
        self.rt.bind_host(regions)
        logger.info("kern bound %d host states at %#x", len(regions), min(r[0] for r in regions.values()))
        if first:
            group.start(self.rt, self.m, self.group)
        self.bound = self.pools()

    def prepare_before_cuda_graph_capture(self, model_runner) -> None:
        """SGLang's hook once the pools exist, on every rank, before any warmup
        forward and outside any forward context: the pools are read off the runner."""
        self.runner = model_runner
        self.bind()

    def forward(self, input_ids, positions, forward_batch, **_):
        fb = forward_batch
        n = input_ids.shape[0]
        hidden = torch.empty(n, self.b["hidden"].shape[1], dtype=torch.bfloat16, device=input_ids.device)
        if get_in_autotune_dummy_run():
            return self.logits_processor(input_ids, hidden, self.lm_head, fb)
        if not torch.cuda.is_current_stream_capturing() and self.bound != self.pools():
            self.bind()
        meta = get_attn_backend().linear_attn_backend.forward_metadata
        lines = meta.mamba_cache_indices
        pages = self.runner.req_to_token_pool.req_to_token
        step = functools.partial(self.run, fb, input_ids, positions, lines, pages, hidden)
        if fb.forward_mode.is_decode():
            for r0 in range(0, n, self.decode_rows):
                r1 = min(r0 + self.decode_rows, n)
                step("decode_batch", r0, r1, r0, r1, 0)
            if fb.mamba_track_mask is not None:
                self.track(lines, fb.mamba_track_mask, meta.mamba_track_indices, n)
            return self.logits_processor(input_ids, hidden, self.lm_head, fb)
        if not fb.forward_mode.is_extend():
            raise NotImplementedError(f"kern runs decode and extend steps, not {fb.forward_mode}")
        lens, prefix = fb.extend_seq_lens_cpu, fb.extend_prefix_lens_cpu
        if "prefill" not in self.m["programs"] and max(lens) > 1:
            raise NotImplementedError(
                f"`{self.m['model']}` only decodes (no `prefill` program), but an extend step has rows of up to "
                f"{max(lens)} tokens: run it as SGLang's decode instance (--disaggregation-mode decode)")
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
        w, r = self.dcp
        sources = {"token_ids": input_ids[t0:t1], "positions": positions[t0:t1],
                   "slot_mapping": local_slots(fb.out_cache_loc[t0:t1], w, r),
                   "seq_lens": local_lens(fb.seq_lens[r0:r1] - ahead, w, r)}
        for name, src in sources.items():
            if name in b:
                b[name][: src.shape[0]].copy_(src)
        if "cu_seqlens_q" in b:
            b["cu_seqlens_q"].copy_(torch.clamp(self.offsets * (k // rows), max=k))
        if "block_table" in b:
            span = self.page * w
            first = pages[fb.req_pool_indices[r0:r1], : self.width * span : span]
            b["block_table"][:rows, : first.shape[1]].copy_(first // span)
        for t in self.tables:
            b[t][:, :rows].copy_(lines[r0:r1].expand(b[t].shape[0], -1))
        self.rt.enqueue_after(program, {"tokens": k, "seqs": rows}, torch.cuda.current_stream().cuda_stream)
        hidden[t0:t1].copy_(b["hidden"][:k])

    def track(self, lines, mask, dest, n) -> None:
        """Copies every recurrent layer's state of the masked rows to their track slots."""
        cache = self.runner.req_to_token_pool.mamba_pool.mamba_cache
        track_mamba_states_all_layers(cache.conv[0], cache.temporal, lines, mask, dest, n, check_freed_slots=True)

    def head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """`head` over the rows SGLang samples: this rank's slice of the vocab."""
        n = hidden_states.shape[0]
        out = torch.empty(n, self.b["logits"].shape[1], dtype=self.b["logits"].dtype, device=hidden_states.device)
        stream = torch.cuda.current_stream().cuda_stream
        step = self.b["head_in"].shape[0]
        for i in range(0, n, step):
            k = min(step, n - i)
            self.b["head_in"][:k].copy_(hidden_states[i : i + k])
            self.rt.enqueue_after("head", {"seqs": k}, stream)
            out[i : i + k].copy_(self.b["logits"][:k])
        if self.check and not torch.cuda.is_current_stream_capturing():
            self.raise_errors()
        return out

    def raise_errors(self) -> None:
        """Every error output's word after the step (`KERN_CHECK_ERRORS`: a sync a step)."""
        words = {n: int(t.view(-1)[0]) for n, t in self.errors.items()}
        if any(words.values()):
            raise RuntimeError(f"kern error words {words}")
