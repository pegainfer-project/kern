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
  on its own (`chunks`). SGLang sends a run of consecutive pages as one
  transfer, which the engine cuts into slices that must each lie in one
  registration, so a run is also cut where a chunk starts (`within`).
- `KVArgs` from the manifest's naming contract, the one the decode plugin
  reads (`model.py`): pooled states `k.l<L>` and `v.l<L>` are a layer's keys
  and values in SGLang's MHA layout, a page of either the bytes SGLang moves
  for that page and layer; `mla.l<L>` a layer's latent (MLA pools). The
  stage's layers are the ones its states name, in order. Recurrent states
  (`conv.l<L>`, `ssm.l<L>`) make the decode pool hybrid, which pairs layers
  by id, so their presence publishes the KV layer ids; sending them is not
  written yet. Head counts and the metadata layout come from SGLang's own
  model config, as its scheduler derives them.

Every rank sends the request's metadata on its last chunk and the decode
side reads it field by field from the one slot every rank writes into, so
the bytes must be the same whoever writes them; only the last stage knows
the first token, so only it writes the metadata, and the others' `send_aux`
is a no-op returning success (SGLang's decode waits for every rank's
success and the slot's bytes, never for who wrote them).

Calls come from one Rust thread per stage process, holding the GIL only for
the call: `open` when the head admits a request, then `advance` until it
answers, once the item that wrote the request's last rows has landed on
the device.
"""
import json
import re
from pathlib import Path

import numpy as np
import torch
from sglang.srt import runtime_context as rc
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll
from sglang.srt.disaggregation.mooncake import conn as mooncake_conn
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVBootstrapServer, MooncakeKVManager, MooncakeKVSender
from sglang.srt.disaggregation.utils import DisaggregationMode, MetadataBuffers, get_dsa_seed_metadata_dim
from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import init_mooncake_transfer_engine
from sglang.srt.runtime_context import get_disagg
from sglang.srt.server_args import ServerArgs

KV = ("k", "v", "mla")
RECURRENT = ("conv", "ssm")


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


def within(per: int):
    """SGLang's grouping of consecutive pages into one transfer each, a group
    also ending where a chunk of `per` pages ends."""

    def group(src: np.ndarray, dst: np.ndarray):
        if src.size == 0 or dst.size == 0:
            return [], []
        if src.size != dst.size:
            raise ValueError(f"{src.size} source pages for {dst.size} destination pages")
        brk = np.where((np.diff(src) != 1) | (np.diff(dst) != 1) | (src[1:] % per == 0))[0] + 1
        return [g.tolist() for g in np.split(src, brk)], [g.tolist() for g in np.split(dst, brk)]

    return group


def manager(chunks: list, last: bool):
    """`MooncakeKVManager` registering `chunks` (every mapped chunk of the
    stage's states, `(address, bytes)`) instead of whole states, and whose
    `send_aux` on any stage but the last writes nothing."""

    class Manager(MooncakeKVManager):
        def _registerable_regions(self):
            return [(int(p), int(n)) for p, n in chunks] + list(zip(self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens))

        def register_buffer_to_engine(self):
            # The engine's batch registration only logs a failure at debug level.
            ptrs, lens = zip(*self._registerable_regions())
            if (ret := self.engine.batch_register(list(ptrs), list(lens))) != 0:
                raise RuntimeError(f"registering {len(ptrs)} regions with the transfer engine failed ({ret})")

        if not last:
            def send_aux(self, req, prefill_aux_index, dst_aux_ptrs):
                return 0

            def send_aux_tcp(self, req, prefill_aux_index, dst_aux_ptrs):
                return 0

    return Manager


class Stage:
    def __init__(self, model_path: str, ip: str, bootstrap_port: int, ib_device: str, rank: int, size: int,
                 page: int, states: dict, chunks: list, slots: int, consensus: list | None):
        """`states` is every pooled state of the stage, `{name: (address, page or slot bytes,
        bytes reserved)}`; `chunks` every mapped chunk under them; `slots` the requests
        that may be sending at once."""
        self.last = rank == size - 1
        sa = ServerArgs(model_path=model_path, host=ip, pp_size=size, page_size=page,
                        disaggregation_mode="prefill", disaggregation_transfer_backend="mooncake",
                        disaggregation_ib_device=ib_device, disaggregation_bootstrap_port=bootstrap_port)
        rc.publish(sa, role="scheduler", ranks=rc.SpawnRanks(world_rank=rank, gpu_id=0))
        self.group = Group(size, rank, consensus)
        rc.get_parallel().override_permanently(pp_group=self.group)
        cfg = ModelConfig.from_server_args(sa)
        self.bootstrap = MooncakeKVBootstrapServer(host=ip, port=bootstrap_port) if rank == 0 else None
        init_mooncake_transfer_engine(hostname=ip, gpu_id=0, ib_device=ib_device)

        kinds = layers(states)
        if any(k in kinds for k in RECURRENT):
            raise NotImplementedError("sending recurrent state (`conv` / `ssm`) to SGLang is not written")
        kv = [s for kind in KV for _, s in kinds.get(kind, [])]
        ids = [l for kind in KV for l, _ in kinds.get(kind, [])]
        chunk = {int(n) for _, n in chunks}
        item = {int(states[s][1]) for s in kv}
        if len(chunk) != 1 or len(item) != 1 or next(iter(chunk)) % next(iter(item)):
            raise NotImplementedError(f"pages of {sorted(item)} bytes over chunks of {sorted(chunk)} bytes")
        mooncake_conn.group_concurrent_contiguous = within(next(iter(chunk)) // next(iter(item)))
        first = min(ids)
        self.meta = MetadataBuffers(
            slots, hidden_size=16, hidden_states_dtype=torch.float32,
            max_sampling_mask_tokens=sa.sampling_mask_max_tokens,
            output_dsa_topk_indices_dim=get_dsa_seed_metadata_dim(cfg.hf_config),
            kv_checksum_enabled=get_disagg().disaggregation_enable_kv_checksum)
        a = KVArgs()
        a.engine_rank, a.pp_rank, a.system_dp_rank, a.rust_http_port = 0, rank, 0, None
        a.kv_cache_dtype_str = rc.get_model().kv_cache_dtype
        a.prefill_start_layer, a.prefill_end_layer = first, max(ids) + 1
        a.kv_data_ptrs = [states[s][0] for s in kv]
        a.kv_item_lens = [states[s][1] for s in kv]
        a.kv_data_lens = [states[s][2] for s in kv]
        a.kv_layer_ids = []
        a.num_draft_entries = 0
        a.kv_head_num = a.total_kv_head_num = cfg.get_total_num_kv_heads()
        a.page_size = page
        a.aux_data_ptrs, a.aux_data_lens, a.aux_item_lens = self.meta.get_buf_infos()
        a.ib_device, a.gpu_id = ib_device, 0
        a.state_types, a.state_data_ptrs, a.state_data_lens, a.state_item_lens = [], [], [], []
        a.state_dim_per_tensor, a.state_slice_outer_counts, a.state_layer_ids, a.state_conv_shard_groups = [], [], [], []
        a.is_hybrid_mla_backend, a.mla_compression_ratios = False, None
        self.mgr = manager(chunks, self.last)(a, DisaggregationMode.PREFILL, sa, "mla" in kinds)
        self.consensus = json.loads(json.dumps(self.group.sent))
        self.addr = f"{ip}:{bootstrap_port}"
        self.senders: dict[int, MooncakeKVSender] = {}
        self.sending: dict[int, int] = {}
        self.free = list(range(slots))

    def open(self, room: int) -> None:
        """Meet the decode side in `room`: SGLang's sender bootstraps while the prompt runs."""
        self.senders[room] = MooncakeKVSender(self.mgr, self.addr, room)

    def advance(self, room: int, pages: list[int], length: int, token: int | None) -> str | None:
        """Move the send of `room` on: `pages` hold `length` positions, `token`
        is the first token on the last stage. None while it goes on; once it
        ended, "" when sent, else why it failed, and the room is let go."""
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
            # The decode side holds the prefix it already has; the rest goes, whole pages.
            skip = s.pop_decode_prefix_len() // self.mgr.kv_args.page_size
            left = np.asarray(pages[skip:-(-length // self.mgr.kv_args.page_size)], dtype=np.int32)
            s.init(len(left), aux)
            s.send(left)
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
