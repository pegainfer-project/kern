"""Typed auxiliary ops. Buffer dimensions and state contract live in README.md."""
from pathlib import Path
import hashlib


def align4(rows):
    return (rows + 3) // 4 * 4 if isinstance(rows, int) else {"mul": [{"ceil_div": [rows, 4]}, 4]}


def definitions(cubin: Path, rows="rows", groups="groups", copies=4, hash_cols=24, heads=64, seqs="seqs"):
    module = {"source": cubin.name, "sha256": hashlib.sha256(cubin.read_bytes()).hexdigest()}
    def op(entry, params, grid):
        return {"params": params.split(";"), "impl": {"launches": [{
            "module": "dsv41_auxiliary", "entry": "dsv41_" + entry,
            "block": [256, 1, 1], "grid": grid}]}}
    ops = {
        "engram_history": op("history", "inout state;in buffer<i32>;in buffer<i64>;in buffer<i64>;in buffer<u8>;i32", [{"ceil_div": [rows, 256]}, 1, 1]),
        "engram_hash": op("hash", "out buffer<i64>;in state;in buffer<i32>;in buffer<i32>;in buffer<i32>;in buffer<i64>;in buffer<i64>;in buffer<i64>;i32;i32;i32;i32;i32;i32;i64", [rows, 1, 1]),
        "engram_lookup": op("lookup", "out buffer<bf16>;in buffer<i64>;in buffer<fp8e4m3>;in buffer<fp8e8m0>;i32;i32;i32;i32;i32", [rows, hash_cols, 1]),
        "engram_inject": op("engram_inject", "out buffer<bf16>;in buffer<bf16>;in buffer<bf16>;in buffer<bf16>;in buffer<bf16>;in buffer<u8>;i32;i32;i32;f32", [rows, copies, 1]),
        "compress2": op("compress2", "out buffer<bf16>;in buffer<f32>;in buffer<f32>;in buffer<bf16>;i32;i32;f32", [groups, 1, 1]),
        "compress1": op("norm", "out buffer<bf16>;in buffer<bf16>;in buffer<bf16>;i32;i32;f32", [rows, 1, 1]),
        "norm_quant": op("norm_quant", "out buffer<bf16>;out buffer<fp8e4m3>;out buffer<i32>;in buffer<bf16>;in buffer<bf16>;i32;i32;i32;i32;f32", [align4(rows), 1, 1]),
        "norm_rope": op("norm_rope", "out buffer<bf16>;in buffer<bf16>;in buffer<bf16>;in buffer<f32>;in buffer<i32>;i32;i32;i32;i32;i32;f32", [rows, 1, 1]),
    }
    ops.update({
        "context_tap_init": op("context_tap", "out buffer<bf16>;in buffer<bf16>;i32;i32", [{"ceil_div": [{"mul": [rows, 5120]}, 256]}, 1, 1]),
        "context_tap": op("context_tap", "inout buffer<bf16>;in buffer<bf16>;i32;i32", [{"ceil_div": [{"mul": [rows, 5120]}, 256]}, 1, 1]),
        "context_slots": op("context_slots", "out buffer<i64>;in buffer<i64>;in buffer<i32>;in buffer<i32>;in buffer<i32>;i32", [{"ceil_div": [rows, 256]}, 1, 1]),
        "index_metadata": op("index_metadata", "out buffer<i32>;out buffer<i32>;out buffer<i32>;out buffer<i32>;in buffer<i32>;in buffer<i32>;in buffer<u8>;in buffer<i32>;i32;i32;i32;i32", [rows, 1, 1]),
        "cache_index": op("cache_index", "inout state;in buffer<u8>;in buffer<fp8e8m0>;in buffer<i64>;i32;i32", [rows, 1, 1]),
        "cache_bf16": op("cache_bf16", "inout state;in buffer<bf16>;in buffer<i64>;i32;i32", [rows, 1, 1]),
        "gather_bf16": op("gather_bf16", "out buffer<bf16>;in state;in buffer<i64>;i32;i32", [rows, 1, 1]),
        "metadata": op("metadata", "out buffer<i32>;out buffer<i32>;out buffer<i64>;out buffer<u8>;out buffer<i32>;out buffer<i32>;out buffer<i32>;out buffer<i32>;out buffer<i32>;out buffer<i32>;out buffer<i32>;in buffer<i32>;in buffer<i64>;in buffer<i32>;in buffer<i64>;in buffer<i32>;in buffer<i32>;i32;i32;i32", [{"ceil_div": [rows, 256]}, 1, 1]),
        "compressed_metadata": op("compressed_metadata", "out buffer<i64>;out buffer<i32>;in buffer<i64>;in buffer<i32>;in buffer<u8>;i32;i32", [{"ceil_div": [rows, 256]}, 1, 1]),
        "indices_mask": op("indices_mask", "inout buffer<i32>;in buffer<u8>;i32;i32", [rows, 1, 1]),
        "accepted_mask": op("accepted_mask", "out buffer<i32>;in buffer<i32>;in buffer<i32>;in buffer<i32>;i32", [{"ceil_div": [seqs, 256]}, 1, 1]),
        "compressor_short_gather": op("compressor_short_gather", "out buffer<f32>;out buffer<f32>;out buffer<i32>;in state;in buffer<f32>;in buffer<f32>;in buffer<i32>;in buffer<i32>;in buffer<i32>;in buffer<i32>;i32;i32;i32", [rows, 1, 1]),
        "compressor_commit": op("compressor_commit", "inout state;in buffer<f32>;in buffer<f32>;in buffer<i32>;in buffer<i32>;in buffer<i32>;i32;i32;i32", [seqs, 1, 1]),
        "index_quant": op("index_quant", "out buffer<u8>;out buffer<fp8e8m0>;out buffer<bf16>;in buffer<bf16>;i32;i32", [rows, 1, 1]),
        "compressor_stage": op("compressor_stage", "inout state;inout state;in buffer<f32>;in buffer<f32>;in buffer<i64>;i32;i32", [rows, 1, 1]),
        "compressor_gather": op("compressor_gather", "out buffer<f32>;out buffer<f32>;out buffer<i32>;in state;in state;in buffer<i32>;in buffer<i32>;in buffer<i32>;i32;i32;i32;i32", [rows, 1, 1]),
        "window_indices": op("window_indices", "out buffer<i32>;in buffer<i32>;in buffer<i32>;in buffer<i32>;in buffer<i32>;i32;i32;i32;i32;i32;i32;i32", [rows, 1, 1]),
        "map_indices": op("map_indices", "out buffer<i32>;in buffer<i32>;in buffer<i32>;in buffer<i32>;in buffer<i32>;i32;i32;i32;i32;i32", [rows, 1, 1]),
        "cache_fp8": op("cache_fp8", "inout state;in buffer<bf16>;in buffer<i64>;i32;i32", [rows, 1, 1]),
        "cache_fp4": op("cache_fp4", "inout state;in buffer<bf16>;in buffer<i64>;i32;i32", [rows, 1, 1]),
        "cache_gather": op("cache_gather", "out buffer<bf16>;in state;in buffer<i64>;i32;i32;i32", [rows, 1, 1]),
        "rope": op("rope", "out buffer<bf16>;in buffer<bf16>;in buffer<f32>;in buffer<i32>;i32;i32;i32;i32;i32", [rows, heads, 1]),
    })
    return {"dsv41_auxiliary": module}, ops
