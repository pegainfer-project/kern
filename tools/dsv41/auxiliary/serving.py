"""Serving metadata and state lowering for V4.1 DP/EP programs.

The caller merges vars/buffers/states/modules/ops into its manifest and appends
these ordinary call lists to its model stages. No program calls another program.
The host stages six rows for a speculative round; draft selects rows0..4, while
verify consumes all six. `fill: valid` is required: padding leases are real
addresses and token0 cannot be treated as a padding sentinel.
"""
from dataclasses import dataclass
from pathlib import Path

from .ops import definitions


def index_page_stride(page_size):
    """DeepGEMM TMA requires whole index pages aligned to512 bytes."""
    return (page_size * 68 + 511) // 512 * 512


def buf(name):
    return {"buf": name}


def state(name):
    return {"state": name}


def scalar(value):
    return {"expr": value} if isinstance(value, dict) else {"var": value} if isinstance(value, str) else {"i32": value}


def call(label, op, *args):
    return {"label": label, "op": op, "args": list(args)}


@dataclass(frozen=True)
class Layout:
    max_seqs: int = 128
    max_tokens: int = 8192
    max_context: int = 1048576
    page_size: int = 128
    target_layers: int = 40
    draft_layers: int = 3
    sources: tuple = ((2, 2), (8, 2), (14, 2), (20, 1))

    def __post_init__(self):
        if self.max_seqs < 1 or self.max_tokens < 6 * self.max_seqs:
            raise ValueError("tokens bound must cover six verification rows per sequence")
        if self.page_size < 32 or self.page_size & (self.page_size - 1):
            raise ValueError("FlashMLA requires power-of-two pages, at least32 window tokens")
        if self.max_context % self.page_size:
            raise ValueError("context bound must be a whole number of pages")
        if any(r not in (1, 2) for _, r in self.sources):
            raise ValueError("only ratio1 and ratio2 compressors are supported")

    @property
    def pages(self):
        return self.max_context // self.page_size

    @property
    def ring(self):
        """Window tokens kept per sequence: the128-token window plus every row one
        step can write ahead of the oldest read, rounded to whole pages. Draft
        rounds read133 back and write six, well inside the same bound."""
        return -(-(128 + self.max_tokens) // self.page_size) * self.page_size


@dataclass
class Serving:
    layout: Layout
    vars: dict
    buffers: dict
    states: dict
    modules: dict
    ops: dict

    @staticmethod
    def rows(mode):
        if mode not in ("prefill", "decode", "verify", "draft"):
            raise ValueError(mode)
        return {"mul": ["seqs", 5]} if mode == "draft" else "tokens"

    def _call(self, mode, label, op, *args):
        return call(f"{mode}.{label}", f"{mode}.{op}", *args)

    def pieces(self, programs):
        """Keep only helpers reached by the final call lists (schema forbids unused definitions).

        Weight definitions/ops from other providers are merged by the caller.
        Domain targets are retained as well as direct state operands.
        """
        calls = [c for p in programs.values() for c in p.get("calls", [])]
        op_names = {c["op"] for c in calls}
        buffer_names = {a["buf"] for c in calls for a in c["args"] if "buf" in a}
        state_names = {a["state"] for c in calls for a in c["args"] if "state" in a}
        buffers = {n: b for n, b in self.buffers.items() if n in buffer_names}
        for b in buffers.values():
            target = b.get("domain", {}).get("index_into")
            if target in self.states:
                state_names.add(target)
        ops = {n: o for n, o in self.ops.items() if n in op_names}
        module_names = {l["module"] for o in ops.values() for l in o["impl"]["launches"] if "module" in l}
        return {"vars": self.vars, "buffers": buffers,
            "states": {n: s for n, s in self.states.items() if n in state_names}, "ops": ops,
            "modules": {n: m for n, m in self.modules.items() if n in module_names}}

    def prepare(self, mode, ids):
        """After draft/verify token splice; before any model layer reads metadata.

        Prefill is groups1/rows=tokens, decode groups=seqs/rows1, verify rows6.
        Draft is an interior five-row stage, never a separately staged program.
        ids is staged input_ids for plain, verify_ids or draft_ids for the round.
        """
        rows = scalar(self.rows(mode))
        names = ("request", "position", "slot", "window_slot", "mask", "starts", "seq_valid", "block_end", "ids32", "all_count", "window_length", "compressed_length")
        calls = [self._call(mode, "metadata", "metadata", *(buf(f"{mode}.{n}") for n in names),
            buf("positions"), buf("slot_mapping"), buf("valid"), buf(ids), buf("cu_seqlens"), buf("seq_lens"),
            buf("window.lines"), rows, scalar("seqs"), scalar(int(mode == "draft")), scalar(self.layout.ring))]
        if mode != "draft":
            calls += [self._call(mode, f"compressed{ratio}", "compressed_metadata",
                buf(f"{mode}.c{ratio}_slot"), buf(f"{mode}.c{ratio}_position"),
                buf(f"{mode}.slot"), buf(f"{mode}.position"), buf(f"{mode}.mask"), rows, scalar(ratio))
                for ratio in (1, 2)]
        width = 192 if mode == "draft" else 128
        calls += [self._call(mode, "window_indices", "window_indices", buf(f"{mode}.window_indices"),
            buf(f"{mode}.request"), buf(f"{mode}.position"), buf(f"{mode}.block_end"), buf("window.lines"),
            rows, scalar(128), scalar(width), scalar(self.layout.ring), scalar(int(mode == "draft")), scalar(5)),
            self._call(mode, "window_mask", "indices_mask", buf(f"{mode}.window_indices"),
                buf(f"{mode}.mask"), rows, scalar(width))]
        return calls

    def engram_hash(self, mode, *, token_map, multipliers, primes, offsets, compressed_pad_id):
        """Constants are supplied by the caller from the official tokenizer/layout.

        Token map int64[vocab], multipliers int64[2,4], primes/offsets int64[2,24].
        Do not substitute raw vocabulary size for compressed vocabulary size.
        """
        if mode == "draft":
            raise ValueError("draft layers have no Engram")
        rows = scalar(self.rows(mode))
        return [self._call(mode, "history", "engram_history", state("engram_history"),
            buf(f"{mode}.ids32"), buf(f"{mode}.slot"), buf(token_map), buf(f"{mode}.mask"), rows),
            self._call(mode, "hash", "engram_hash", buf(f"{mode}.hashes"), state("engram_history"),
                buf(f"{mode}.request"), buf(f"{mode}.position"), buf("page_table"),
                buf(multipliers), buf(primes), buf(offsets), rows, scalar(self.layout.page_size),
                scalar(self.layout.pages), scalar(2), scalar(8), scalar(4), {"i64": compressed_pad_id})]

    def engram_lookup(self, mode, layer, table, scales, output, shard=None):
        """`shard` = (ranks, rows per rank, total rows) when the table is sharded
        into HBM across the EP group and read through `<table>.peers`; None
        when it is one shared host-memory table."""
        if layer not in (1, 14):
            raise ValueError("Engram only appears in target layers1 and14")
        geometry = [scalar(self.rows(mode)), scalar(24), scalar(256), scalar(48), scalar(0 if layer == 1 else 24)]
        if shard is None:
            return [self._call(mode, f"engram{layer}.lookup", "engram_lookup", buf(output),
                buf(f"{mode}.hashes"), buf(table), buf(scales), *geometry)]
        ranks, per_rank, total = shard
        return [self._call(mode, f"engram{layer}.lookup", "engram_lookup_peers", buf(output),
            buf(f"{mode}.hashes"), buf(table + ".peers"), buf(scales + ".peers"), *geometry,
            scalar(ranks), {"i64": per_rank}, {"i64": total})]

    def window_write(self, mode, layer, post_rope_kv):
        """Ring writes: window_slot is the row's position modulo the ring inside its sequence's slot."""
        prefix = "draft" if mode == "draft" else "target"
        return [self._call(mode, f"window{layer}.write", "cache_fp8", state(f"{prefix}.window.{layer}"),
            buf(post_rope_kv), buf(f"{mode}.window_slot"), scalar(self.rows(mode)), scalar(self.layout.page_size))]

    def compressor(self, mode, source, projected_kv, projected_score, norm_weight, output):
        """Projected f32 KV/score must survive until the accepted commit.

        ratio2 emits rows aligned to input rows; c2_slot=-1 masks incomplete
        groups and padding. Index-key projection uses unrotated output first.
        """
        ratio = dict(self.layout.sources)[source]
        rows = scalar(self.rows(mode))
        if ratio == 1:
            return [self._call(mode, f"compress{source}.norm", "compress1", buf(output),
                buf(projected_kv), buf(norm_weight), rows, scalar(512), {"f32": 1e-20})]
        return [self._call(mode, f"compress{source}.gather", "compressor_short_gather",
            buf(f"{mode}.pool_kv"), buf(f"{mode}.pool_score"), buf(f"{mode}.pool_valid"),
            state(f"compressor.{source}"), buf(projected_kv), buf(projected_score),
            buf(f"compressor.{source}.lines"), buf(f"{mode}.starts"), buf(f"{mode}.request"),
            buf(f"{mode}.position"), rows, scalar(512), scalar(1)),
            self._call(mode, f"compress{source}.pool", "compress2", buf(output),
                buf(f"{mode}.pool_kv"), buf(f"{mode}.pool_score"), buf(norm_weight), rows,
                scalar(512), {"f32": 1e-20})]

    def compressed_write(self, mode, source, post_rope_kv):
        ratio = dict(self.layout.sources)[source]
        return [self._call(mode, f"compressed{source}.write", "cache_fp4", state(f"compressed.{source}"),
            buf(post_rope_kv), buf(f"{mode}.c{ratio}_slot"), scalar(self.rows(mode)),
            scalar(self.layout.page_size // ratio))]

    def index_metadata(self, mode, source):
        """Query-local page table and causal end for paged dense scoring.

        Reuse these buffers among sources with the same compression ratio.
        Call after prepare; valid0 rows have end0 and an all-zero page table.
        """
        if mode == "draft":
            raise ValueError("draft layers have no compressed indexer")
        ratio = dict(self.layout.sources)[source]
        return [self._call(mode, f"index{source}.metadata", "index_metadata",
            buf(f"{mode}.c{ratio}_page_table"), buf(f"{mode}.c{ratio}_end"),
            buf(f"{mode}.index_request_ids"), buf(f"{mode}.sparse_ends"),
            buf(f"{mode}.request"), buf(f"{mode}.position"), buf(f"{mode}.mask"),
            buf("page_table"), scalar(self.rows(mode)), scalar(self.layout.pages),
            scalar(self.layout.page_size // ratio), scalar(ratio))]

    def index_write(self, mode, source, packed_key, key_scales):
        """Packed[page,64], E8M0[page,4], then pad page stride to512 bytes."""
        ratio = dict(self.layout.sources)[source]
        return [self._call(mode, f"index{source}.write", "cache_index", state(f"index_k.{source}"),
            buf(packed_key), buf(key_scales), buf(f"{mode}.c{ratio}_slot"), scalar(self.rows(mode)),
            scalar(self.layout.page_size // ratio))]

    def compressed_indices(self, mode, ratio, logical_indices, output):
        return [self._call(mode, "compressed.indices", "map_indices", buf(output), buf(logical_indices),
            buf(f"{mode}.request"), buf(f"{mode}.position"), buf("page_table"), scalar(self.rows(mode)),
            scalar(512), scalar(ratio), scalar(self.layout.page_size // ratio), scalar(self.layout.pages)),
            self._call(mode, "compressed.mask", "indices_mask", buf(output), buf(f"{mode}.mask"),
                scalar(self.rows(mode)), scalar(512))]

    def commit(self, mode, source, projected_kv, projected_score, accepted=None):
        """After acceptance: width1 line table, never null speculative columns.

        For plain modes omitted accepted means all active input rows. Verify
        requires accepted='nacc' after spec_count; zero is a true no-op.
        """
        if dict(self.layout.sources)[source] != 2:
            return []
        if mode == "verify" and accepted is None:
            raise ValueError("verification commit requires accepted prefix counts")
        counts = accepted or f"{mode}.all_count"
        return [self._call(mode, f"compress{source}.mask_accept", "accepted_mask", buf(f"{mode}.commit_count"),
            buf(counts), buf(f"{mode}.seq_valid"), buf(f"{mode}.starts"), scalar("seqs")),
            self._call(mode, f"compress{source}.commit", "compressor_commit", state(f"compressor.{source}"),
                buf(projected_kv), buf(projected_score), buf(f"compressor.{source}.lines"),
                buf(f"{mode}.starts"), buf(f"{mode}.commit_count"), scalar("seqs"), scalar(512), scalar(1))]


def rope_name(compress_ratio, *, split=False):
    """Use the layer's ratio for ALL its Q/K/index/O rotations, not cache type."""
    family = "compressed" if compress_ratio else "window"
    return f"rope.{family}.{'split' if split else 'interleaved'}"


def build(cubin: Path, layout=Layout(), peers=None):
    """Return concrete definitions and lowering helpers (weights supplied outside)."""
    variables = {"tokens": {"max": layout.max_tokens}, "seqs": {"max": layout.max_seqs}}
    buffers = {
        "input_ids": {"kind": "input", "dtype": "i64", "shape": ["tokens"], "fill": "token"},
        "anchor_token": {"kind": "input", "dtype": "i64", "shape": ["seqs"], "fill": "token"},
        "positions": {"kind": "input", "dtype": "i32", "shape": ["tokens"], "fill": "position"},
        "valid": {"kind": "input", "dtype": "i32", "shape": ["tokens"], "fill": "valid"},
        "slot_mapping": {"kind": "input", "dtype": "i64", "shape": ["tokens"], "fill": "slot",
                         "domain": {"index_into": "engram_history"}},
        "seq_lens": {"kind": "input", "dtype": "i32", "shape": ["seqs"], "fill": "seq_len"},
        "cu_seqlens": {"kind": "input", "dtype": "i32", "shape": [layout.max_seqs + 1], "fill": "cu_seqlens"},
        "page_table": {"kind": "input", "dtype": "i32", "shape": ["seqs", layout.pages],
                       "domain": {"index_into": "engram_history", "stride": layout.page_size}},
        # One line per sequence, the whole ring: the value is the sequence's
        # slot, shared by every window state of the lease.
        "window.lines": {"kind": "input", "dtype": "i32", "shape": [1, "seqs"],
                         "domain": {"index_into": "target.window.0", "stride": layout.ring * 528}},
    }
    states = {"engram_history": {"bytes_per_token": 8}}
    for prefix, count in (("target", layout.target_layers), ("draft", layout.draft_layers)):
        states.update({f"{prefix}.window.{layer}": {"bytes_per_seq": layout.ring * 528} for layer in range(count)})
    for source, ratio in layout.sources:
        # State allocation is measured per ORIGINAL token; kernels address
        # compressed tokens with page_size/ratio and slot_mapping/ratio.
        states[f"compressed.{source}"] = {"bytes_per_token": 288 // ratio}
        states[f"index_k.{source}"] = {"bytes_per_token": index_page_stride(layout.page_size // ratio) // layout.page_size}
        if ratio == 2:
            states[f"compressor.{source}"] = {"bytes_per_seq": 4096}
            buffers[f"compressor.{source}.lines"] = {"kind": "input", "dtype": "i32", "shape": [1, "seqs"],
                "domain": {"index_into": f"compressor.{source}", "stride": 4096}}
    modules, all_ops = {}, {}
    for mode in ("prefill", "decode", "verify", "draft"):
        rows = Serving.rows(mode)
        mods, ops = definitions(cubin, rows=rows, groups=rows, seqs="seqs", peers=peers)
        modules.update(mods)
        all_ops.update({f"{mode}.{name}": op for name, op in ops.items()})
        shapes = {"request": ("i32", ["tokens"]), "position": ("i32", ["tokens"]),
            "slot": ("i64", ["tokens"]), "window_slot": ("i64", ["tokens"]), "mask": ("u8", ["tokens"]), "starts": ("i32", [layout.max_seqs + 1]),
            "seq_valid": ("i32", ["seqs"]), "block_end": ("i32", ["tokens"]), "ids32": ("i32", ["tokens"]),
            "window_length": ("i32", ["tokens"]), "compressed_length": ("i32", ["tokens"]),
            "all_count": ("i32", ["seqs"]), "commit_count": ("i32", ["seqs"]),
            "window_indices": ("i32", ["tokens", 192 if mode == "draft" else 128])}
        if mode != "draft":
            shapes.update({"hashes": ("i64", ["tokens", 2, 24]), "pool_kv": ("f32", ["tokens", 2, 512]),
                "pool_score": ("f32", ["tokens", 2, 512]), "pool_valid": ("i32", ["tokens"]),
                "index_request_ids": ("i32", ["tokens"]), "sparse_ends": ("i32", ["tokens"])})
            for ratio in (1, 2):
                shapes[f"c{ratio}_page_table"] = ("i32", ["tokens", layout.pages])
                shapes[f"c{ratio}_end"] = ("i32", ["tokens"])
                shapes[f"c{ratio}_slot"] = ("i64", ["tokens"])
                shapes[f"c{ratio}_position"] = ("i32", ["tokens"])
        buffers.update({f"{mode}.{name}": {"kind": "workspace", "dtype": dtype, "shape": shape}
                        for name, (dtype, shape) in shapes.items()})
    return Serving(layout, variables, buffers, states, modules, all_ops)
