"""One kern pipeline stage as one of SGLang's prefill pp ranks.

kern-serve (`--sglang-ip`, built with `sglang-pd`) embeds Python in every
stage process and drives a `Stage`: the stage computes its layers into its
own pooled states, and this module moves them to an unmodified SGLang
decode instance through SGLang's own P/D, unchanged: its bootstrap server
(on rank 0), its `MooncakeKVManager` in prefill mode and a
`MooncakeKVSender` per request. Nothing of SGLang's server or scheduler
runs; what they would have set up is set up here:

- the runtime context SGLang's disaggregation code reads (`ServerArgs`
  published as a scheduler, the parallel ranks, the pp group): the pp group
  is the one thing ranks share, and only once, when rank 0's consensus
  store tells the others where it listens (`broadcast_object`). Rank 0
  records what it broadcast (`Stage.consensus`), the head carries it to the
  other stages in their tables, and they replay it.
- the transfer engine over the stage's memory. A pooled state is one
  reserved address range over 64 MiB chunks, each its own allocation, and
  RDMA registers memory by allocation, so every mapped chunk is registered
  on its own (`chunks`). The engine cuts a transfer into slices that must
  each lie in one registration, and neither a run of pages nor a token's
  row nor a sequence's slot stops at a chunk's end, so every transfer is
  cut where its source crosses one (`within`).
- `KVArgs` from the manifest's naming contract, the one the decode plugin
  reads (`model.py`): pooled states `k.l<L>` and `v.l<L>` are a layer's keys
  and values in SGLang's MHA layout, a page of either the bytes SGLang moves
  for that page and layer; `mla.l<L>` a layer's latent (MLA pools). The
  stage's layers are the ones its states name, in order. Recurrent states
  make the decode pool hybrid: `conv.l<L>` is a layer's conv window and
  `ssm.l<L>` its recurrent state, a slot of either the bytes of SGLang's
  per-slot tensor at this stage's TP of 1. A hybrid pool pairs entries by
  layer id, so both the KV and the state entries carry theirs, and the
  state goes as SGLang's mamba component, resliced for the decode side's
  TP by SGLang. Head counts, the state's shapes and the metadata layout
  come from SGLang's own model config, as its scheduler derives them.

Every rank sends the request's metadata on its last chunk and the decode
side reads it field by field from the one slot every rank writes into, so
the bytes must be the same whoever writes them; only the last stage knows
the first token, so only it writes the metadata, and the others' `send_aux`
is a no-op returning success (SGLang's decode waits for every rank's
success and the slot's bytes, never for who wrote them).

Calls come from one Rust thread per stage process, holding the GIL only for
the call: `open` when the head admits a request, then `advance` until it
answers, once the item that wrote the request's last rows has landed on
the device. SGLang's own threads (bootstrap, transfer) end only by raising,
and SGLang counts a prefill whose thread raised as dead, so a stage that
loses one exits, and the head fails the group (`die`).
"""
import bisect
import json
import math
import os
import re
import sys
import threading
import traceback
from pathlib import Path

import numpy as np
import torch
from sglang.srt import runtime_context as rc
from sglang.srt.configs.hybrid_arch import mambaish_config
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll, StateType
from sglang.srt.disaggregation.common.staging_buffer import compute_grid_segments
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVBootstrapServer, MooncakeKVManager, MooncakeKVSender
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    MetadataBuffers,
    append_state_component,
    get_dsa_seed_metadata_dim,
)
from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import init_mooncake_transfer_engine
from sglang.srt.runtime_context import get_disagg
from sglang.srt.server_args import ServerArgs

KV = ("k", "v", "mla")
RECURRENT = ("conv", "ssm")


def die(e: threading.ExceptHookArgs) -> None:
    traceback.print_exception(e.exc_type, e.exc_value, e.exc_traceback, file=sys.stderr)
    print(f"kern_sglang: thread `{e.thread.name if e.thread else '?'}` died, the stage exits", file=sys.stderr, flush=True)
    os._exit(1)


def layers(states: dict) -> dict[str, list[tuple[int, str]]]:
    """The stage's pooled states by kind, each `(layer, state)` in layer order."""
    out: dict[str, list[tuple[int, str]]] = {}
    for name in states:
        m = re.fullmatch(r"(\w+)\.l(\d+)", name)
        if m is None or m[1] not in KV + RECURRENT:
            raise ValueError(f"pooled state `{name}` is not one of the prefill contract's {KV + RECURRENT}")
        out.setdefault(m[1], []).append((int(m[2]), name))
    return {k: sorted(v) for k, v in out.items()}


class Group:
    """The pp group as SGLang's consensus store reads it: its size, this
    rank, and `broadcast_object`, which rank 0 records and the others
    replay from what rank 0 recorded."""

    def __init__(self, size: int, rank: int, replay: list | None):
        self.world_size, self.rank_in_group, self.rank = size, rank, rank
        self.sent: list = []
        self.replay = list(replay or [])

    def broadcast_object(self, obj, src=0):
        if self.rank_in_group == src:
            self.sent.append(obj)
            return obj
        if not self.replay:
            raise RuntimeError("pp rank %d: a broadcast rank 0 never made" % self.rank)
        v = self.replay.pop(0)
        return tuple(v) if isinstance(v, list) else v


def within(chunks: list):
    """Transfer blocks `(source, destination, bytes)` cut wherever the source
    crosses from one of `chunks` into the next."""
    bounds = sorted({int(p) for p, _ in chunks} | {int(p) + int(n) for p, n in chunks})

    def cut(blocks):
        out = []
        for s, d, n in blocks:
            s, d, n = int(s), int(d), int(n)
            i = bisect.bisect_right(bounds, s)
            while i < len(bounds) and bounds[i] < s + n:
                m = bounds[i] - s
                out.append((s, d, m))
                s, d, n, i = s + m, d + m, n - m, i + 1
            out.append((s, d, n))
        return out

    return cut


def mamba(kinds: dict, states: dict, cfg: ModelConfig) -> dict:
    """The stage's recurrent states as SGLang's mamba component: per entry,
    conv windows then recurrent states, each in layer order, its address,
    bytes, slot bytes, the axis it is sliced along for the decode side's TP
    (its length, the rows before it, the conv window's sub-blocks) and its
    layer, from the shapes SGLang's model config gives a TP of 1."""
    shape = mambaish_config(cfg).mamba2_cache_params.shape
    if len(shape.conv) != 1:
        raise NotImplementedError(f"{len(shape.conv)} conv windows a layer")
    axis = getattr(shape, "conv_slice_axis", 0)
    per = {
        "conv": (shape.conv[0], axis, getattr(shape, "conv_shard_groups", None)),
        "ssm": (shape.temporal, 0, None),
    }
    if [l for l, _ in kinds.get("conv", [])] != [l for l, _ in kinds.get("ssm", [])]:
        raise ValueError(f"conv layers {kinds.get('conv')} and ssm layers {kinds.get('ssm')} differ")
    out = {k: [] for k in ("ptrs", "lens", "items", "dims", "outer", "groups", "ids")}
    for kind, (dims, ax, groups) in per.items():
        for l, name in kinds[kind]:
            ptr, item, reserved = states[name]
            if item % math.prod(dims):
                raise ValueError(f"`{name}`: a slot of {item} bytes is not SGLang's {kind} shape {tuple(dims)}")
            for k, v in zip(out, (ptr, reserved, item, dims[ax], math.prod(dims[:ax]), groups, l)):
                out[k].append(v)
    return out


def manager(chunks: list, last: bool):
    """`MooncakeKVManager` registering `chunks` (every mapped chunk of the
    stage's states, `(address, bytes)`) instead of whole states, cutting
    every transfer at their boundaries, and whose `send_aux` on any stage
    but the last writes nothing."""
    cut = within(chunks)

    class Manager(MooncakeKVManager):
        def _registerable_regions(self):
            return [(int(p), int(n)) for p, n in chunks] + list(zip(self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens))

        def register_buffer_to_engine(self):
            # The engine's batch registration only logs a failure at debug level.
            ptrs, lens = zip(*self._registerable_regions())
            if (ret := self.engine.batch_register(list(ptrs), list(lens))) != 0:
                raise RuntimeError(f"registering {len(ptrs)} regions with the transfer engine failed ({ret})")

        def _transfer_data(self, session, blocks):
            return super()._transfer_data(session, cut(blocks))

        if not last:
            def send_aux(self, req, prefill_aux_index, dst_aux_ptrs):
                return 0

            def send_aux_tcp(self, req, prefill_aux_index, dst_aux_ptrs):
                return 0

    return Manager


class Stage:
    def __init__(self, model_path: str, ip: str, bootstrap_host: str, bootstrap_port: int, ib_device: str,
                 args: str, rank: int, size: int, page: int, states: dict, chunks: list, slots: int,
                 consensus: list | None):
        """`ip` is this stage's address, `bootstrap_host` rank 0's, where the bootstrap server
        runs and every rank registers (SGLang's server host); `states` is every pooled state
        of the stage, `{name: (address, page or slot bytes, bytes reserved)}`; `chunks` every
        mapped chunk under them; `slots` the requests that may be sending at once.
        `args` is a JSON object of the decode side's server arguments that must hold here
        too: `kv_cache_dtype`, which it checks against its own, and what shapes a prompt's
        tokens (`kern_sglang.prompt`)."""
        self.last = rank == size - 1
        threading.excepthook = die
        # The address a rank registers for the decode side to reach it.
        os.environ["SGLANG_HOST_IP"] = ip
        sa = ServerArgs(**json.loads(args) | dict(
            model_path=model_path, host=bootstrap_host, pp_size=size, page_size=page,
            disaggregation_mode="prefill", disaggregation_transfer_backend="mooncake",
            disaggregation_ib_device=ib_device, disaggregation_bootstrap_port=bootstrap_port))
        rc.publish(sa, role="scheduler", ranks=rc.SpawnRanks(world_rank=rank, gpu_id=0))
        self.group = Group(size, rank, consensus)
        rc.get_parallel().override_permanently(pp_group=self.group)
        cfg = ModelConfig.from_server_args(sa)
        self.bootstrap = MooncakeKVBootstrapServer(host=bootstrap_host, port=bootstrap_port) if rank == 0 else None
        init_mooncake_transfer_engine(hostname=ip, gpu_id=0, ib_device=ib_device)

        kinds = layers(states)
        hybrid = any(k in kinds for k in RECURRENT)
        kv = [s for kind in KV for _, s in kinds.get(kind, [])]
        ids = [l for kind in KV for l, _ in kinds.get(kind, [])]
        first = min(l for ls in kinds.values() for l, _ in ls)
        self.meta = MetadataBuffers(
            slots, hidden_size=16, hidden_states_dtype=torch.float32,
            max_sampling_mask_tokens=sa.sampling_mask_max_tokens,
            output_dsa_topk_indices_dim=get_dsa_seed_metadata_dim(cfg.hf_config),
            kv_checksum_enabled=get_disagg().disaggregation_enable_kv_checksum)
        a = KVArgs()
        a.engine_rank, a.pp_rank, a.system_dp_rank, a.rust_http_port = 0, rank, 0, None
        a.kv_cache_dtype_str = rc.get_model().kv_cache_dtype
        # A hybrid pool counts its KV layers among the full-attention ones.
        full = cfg.hf_text_config.full_attention_layer_ids if hybrid else range(first)
        a.prefill_start_layer, a.prefill_end_layer = sum(l < first for l in full), max(ids) + 1
        a.kv_data_ptrs = [states[s][0] for s in kv]
        a.kv_item_lens = [states[s][1] for s in kv]
        a.kv_data_lens = [states[s][2] for s in kv]
        a.kv_layer_ids = ids if hybrid else []
        a.num_draft_entries = 0
        a.kv_head_num = a.total_kv_head_num = cfg.get_total_num_kv_heads()
        a.page_size = page
        a.aux_data_ptrs, a.aux_data_lens, a.aux_item_lens = self.meta.get_buf_infos()
        a.ib_device, a.gpu_id = ib_device, 0
        a.state_types, a.state_data_ptrs, a.state_data_lens, a.state_item_lens = [], [], [], []
        a.state_dim_per_tensor, a.state_slice_outer_counts, a.state_layer_ids, a.state_conv_shard_groups = [], [], [], []
        a.is_hybrid_mla_backend, a.mla_compression_ratios = hybrid and "mla" in kinds, None
        if hybrid:
            m = mamba(kinds, states, cfg)
            append_state_component(a, StateType.MAMBA, m["ptrs"], m["lens"], m["items"], m["dims"], m["groups"],
                                   m["outer"], m["ids"])
        self.hybrid = hybrid
        self.mgr = manager(chunks, self.last)(a, DisaggregationMode.PREFILL, sa, "mla" in kinds and not hybrid)
        self.consensus = json.loads(json.dumps(self.group.sent))
        self.addr = f"{bootstrap_host}:{bootstrap_port}"
        self.senders: dict[int, MooncakeKVSender] = {}
        self.sending: dict[int, int] = {}
        self.free = list(range(slots))

    def open(self, room: int) -> None:
        """Meet the decode side in `room`: SGLang's sender bootstraps while the prompt runs."""
        self.senders[room] = MooncakeKVSender(self.mgr, self.addr, room)

    def advance(self, room: int, pages: list[int], slot: int | None, length: int, token: int | None) -> str | None:
        """Move the send of `room` on: `pages` hold `length` positions and
        `slot` the request's recurrent state, `token` is the first token on
        the last stage. None while it goes on; once it ended, "" when sent,
        else why it failed, and the room is let go."""
        s = self.senders[room]
        poll = s.poll()
        if room not in self.sending and poll == KVPoll.WaitingForInput:
            if not self.free:
                return self.end(room, f"no metadata slot free of {self.meta.output_ids.shape[0]}")
            aux = self.free.pop()
            self.sending[room] = aux
            if token is not None:
                self.meta.output_ids[aux].zero_()
                self.meta.output_ids[aux][0] = token
                self.meta.cached_tokens[aux].zero_()
                self.meta.bootstrap_room[aux, 0] = room
            # The decode side holds the prefix it already has; the rest goes, whole pages,
            # in pieces its relayout buffer holds, past which each token is an RDMA of its own.
            page = self.mgr.kv_args.page_size
            base = s.pop_decode_prefix_len() // page * page
            if self.hybrid and slot is None:
                return self.end(room, "a hybrid pool's request with no state slot")
            n = s.get_max_transfer_tokens()
            s.init(-(-length // page) - base // page, aux)
            for a, b in compute_grid_segments(base, length, base, n) if n else [(base, length)]:
                s.send(np.asarray(pages[a // page:-(-b // page)], dtype=np.int32),
                       [[slot]] if self.hybrid and b == length else None, num_kv_tokens=b - a)
            return None
        if poll == KVPoll.Success:
            return self.end(room, "")
        if poll == KVPoll.Failed:
            why = self.mgr.failure_records.get(room) or "failed"
            return self.end(room, str(why))
        return None

    def end(self, room: int, outcome: str) -> str:
        s = self.senders.pop(room)
        if room in self.sending:
            self.free.append(self.sending.pop(room))
        s.clear()
        return outcome
