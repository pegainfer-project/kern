#!/usr/bin/env python3
"""Generate the toy targets: manifests over tools/kernels-src/toy.cu in the
shapes kern-serve serves (paged, per-sequence state, speculative rounds,
a big state), a byte-level tokenizer, the salt weight, and a kern.toml
naming them all. Nothing here is a model; it is the states' shapes with
exact kernels over them (tools/toy/model.py is the oracle).

  python3 tools/toy/gen.py --out target/toy [--sm sm_103a]
  python3 tools/e2e/e2e.py --config target/toy/kern.toml --reference tools/toy/model.py --gpus 0 --out results/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from model import EOS, SALT  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "tools" / "kernels-src" / "toy.cu"


def build(out: Path, sm: str, block: int) -> tuple[str, str]:
    """nvcc the kernel at one block size into `out`, named by its sha; the module's file name and sha."""
    tmp = out / f"toy-b{block}.cubin.tmp"
    subprocess.run(["nvcc", "-cubin", f"-arch={sm}", f"-DBLOCK={block}", "-o", str(tmp), str(SRC)], check=True)
    sha = hashlib.sha256(tmp.read_bytes()).hexdigest()
    name = f"toy-{sha[:12]}.cubin"
    tmp.rename(out / name)
    return name, sha


def manifest(
    name: str,
    module: tuple[str, str],
    block: int,
    *,
    bytes_per_token: int,
    page: int,
    width: int,
    tokens_max: int,
    seqs_max: int,
    bytes_per_seq: int = 0,
    rows: int = 1,
) -> dict:
    ctx = width * page
    src, sha = module
    lined = bytes_per_seq > 0
    buffers = {
        "token_ids": {"dtype": "i64", "shape": ["tokens"], "kind": "input", "fill": "token", "domain": {"min": 0, "max": EOS}},
        "positions": {"dtype": "i64", "shape": ["tokens"], "kind": "input", "fill": "position", "domain": {"min": 0, "max": ctx - 1}},
        "slot_mapping": {"dtype": "i64", "shape": ["tokens"], "kind": "input", "fill": "slot", "domain": {"index_into": "kv"}},
        "block_table": {"dtype": "i32", "shape": ["seqs", width], "kind": "input", "domain": {"index_into": "kv", "stride": page}},
        "seq_lens": {"dtype": "i32", "shape": ["seqs"], "kind": "input", "fill": "seq_len", "domain": {"min": 1, "max": ctx}},
        "next_token": {"dtype": "i64", "shape": ["seqs"], "kind": "output", "fill": "tokens", "domain": {"min": 0, "max": EOS}},
        "salt": {"dtype": "i64", "shape": [64], "kind": "weight", "bind": [{"tensor": "salt"}]},
    }
    states = {"kv": {"bytes_per_token": bytes_per_token}}
    if lined:
        states["mem"] = {"bytes_per_seq": bytes_per_seq}
        buffers["mem.line"] = {"dtype": "i32", "shape": [1, "seqs"], "kind": "input", "domain": {"index_into": "mem", "stride": bytes_per_seq}}
    if rows > 1:
        buffers["round_tokens"] = {"dtype": "i64", "shape": ["seqs", rows], "kind": "output", "fill": "tokens", "domain": {"min": 0, "max": EOS}}
        buffers["count"] = {"dtype": "i32", "shape": ["seqs"], "kind": "output", "fill": "count", "domain": {"min": 1, "max": rows}}

    def op(entry: str, grid: str, iface: list[str], extra: list[dict]) -> dict:
        """An op whose one launch is the interface plus impl-private scalars."""
        args = [{"param": i} for i in range(len(iface))] + extra
        params = iface + [next(iter(e)) for e in extra]
        return {"params": iface, "impl": {"launches": [{"module": "toy", "entry": entry, "params": params, "block": [block, 1, 1], "grid": [grid, 1, 1], "args": args}]}}

    i32, i64 = (lambda v: {"i32": v}), (lambda v: {"i64": v})
    geometry = [i64(bytes_per_token), i32(page), i32(width)]
    line = ["in buffer<i32>", "in state"]
    ops = {
        "write": op("toy_write", "tokens", ["in buffer<i64>", "in buffer<i64>", "in buffer<i64>", "inout state", "in buffer<i64>"], [i64(bytes_per_token)]),
        "predict": op("toy_predict", "seqs", ["in buffer<i32>", "in buffer<i32>", "in state", "in buffer<i64>", "out buffer<i64>"], geometry),
    }
    if lined:
        ops["fold"] = op("toy_fold", "seqs", ["in buffer<i64>", "in buffer<i64>", "in buffer<i32>", "inout state", "in buffer<i64>", "i32"], [i64(bytes_per_seq)])
        ops["predict"] = op("toy_predict_mem", "seqs", ["in buffer<i32>", "in buffer<i32>", "in state"] + line + ["in buffer<i64>", "out buffer<i64>"], geometry + [i64(bytes_per_seq)])
    if rows > 1:
        rows_iface = ["in buffer<i64>", "in buffer<i64>", "in buffer<i64>", "in buffer<i32>", "inout state"]
        tail = ["in buffer<i64>", "out buffer<i64>", "out buffer<i32>"]
        if lined:
            ops["round"] = op("toy_round_mem", "seqs", rows_iface + ["in buffer<i32>", "inout state"] + tail, geometry + [i32(rows), i64(bytes_per_seq)])
        else:
            ops["round"] = op("toy_round", "seqs", rows_iface + tail, geometry + [i32(rows)])

    buf, state, var = (lambda n: {"buf": n}), (lambda n: {"state": n}), (lambda n: {"var": n})
    write = {"op": "write", "args": [buf("token_ids"), buf("positions"), buf("slot_mapping"), state("kv"), buf("salt")]}
    # A fold takes the group's rows: the whole prompt in prefill, one in a step.
    fold = lambda rows: {"op": "fold", "args": [buf("token_ids"), buf("positions"), buf("mem.line"), state("mem"), buf("salt"), rows]}
    predict = {
        "op": "predict",
        "args": [buf("seq_lens"), buf("block_table"), state("kv")] + ([buf("mem.line"), state("mem")] if lined else []) + [buf("salt"), buf("next_token")],
    }
    round_ = {
        "op": "round",
        "args": [buf("token_ids"), buf("positions"), buf("slot_mapping"), buf("block_table"), state("kv")]
        + ([buf("mem.line"), state("mem")] if lined else [])
        + [buf("salt"), buf("round_tokens"), buf("count")],
    }
    # A per-sequence state's prefill hands the token back (every prompt
    # token folds in order); a paged one leaves the last prompt token to
    # the first step, as an attention model's prefill does.
    programs = {
        "prefill": {"batch": {"groups": 1, "rows": "tokens"}, "calls": [write, fold(var("tokens")), predict] if lined else [write]},
        "decode": {"batch": {"groups": seqs_max, "rows": 1}, "graph": True, "calls": [write, fold(i32(1)), predict] if lined else [write, predict]},
    }
    if rows > 1:
        programs["round"] = {"batch": {"groups": seqs_max, "rows": rows}, "graph": True, "calls": [round_]}
    return {
        "schema_version": 5,
        "model": name,
        "vars": {"tokens": {"max": tokens_max}, "seqs": {"max": seqs_max}},
        "states": states,
        "buffers": buffers,
        "modules": {"toy": {"source": src, "sha256": sha}},
        "ops": ops,
        "programs": programs,
    }


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's byte-to-char table, the one HF's ByteLevel uses."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def tokenizer() -> dict:
    """Byte-level tokens whose id is the byte, and eos."""
    vocab = {c: b for b, c in bytes_to_unicode().items()}
    vocab["<|eos|>"] = EOS
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [{"id": EOS, "content": "<|eos|>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True}],
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None, "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": False, "vocab": vocab, "merges": []},
    }


def safetensors(tensors: dict[str, tuple[str, list[int], bytes]]) -> bytes:
    header, data, at = {}, b"", 0
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [at, at + len(raw)]}
        data += raw
        at += len(raw)
    h = json.dumps(header).encode()
    h += b" " * (-len(h) % 8)
    return struct.pack("<Q", len(h)) + h + data


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sm", default="sm_103a")
    a = ap.parse_args()
    out = a.out.resolve()
    (out / "kernels").mkdir(parents=True, exist_ok=True)
    (out / "model").mkdir(exist_ok=True)
    b256 = build(out / "kernels", a.sm, 256)
    b128 = build(out / "kernels", a.sm, 128)

    small = dict(bytes_per_token=4096, page=16, width=512, tokens_max=2048, seqs_max=256)
    shapes = {
        "toy-paged": small,
        "toy-stateful": dict(bytes_per_seq=1 << 20, **small),
        "toy-spec": dict(rows=4, **small),
        "toy-stateful-spec": dict(bytes_per_seq=1 << 20, rows=4, **small),
        "toy-big": dict(bytes_per_token=65536, page=64, width=1024, tokens_max=8192, seqs_max=256),
    }
    # Every shape twice, at two block sizes: `kern test` holds one to the other.
    targets = {f"{name}{suffix}": manifest(name, module, block, **shape) for name, shape in shapes.items() for suffix, module, block in (("", b256, 256), ("-ref", b128, 128))}
    for name, m in targets.items():
        (out / f"{name}.json").write_text(json.dumps(m, indent=1) + "\n")
    (out / "model" / "tokenizer.json").write_text(json.dumps(tokenizer()))
    (out / "model" / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{% for m in messages %}{{ m['content'] }}{% endfor %}", "eos_token": "<|eos|>"}))
    (out / "model" / "config.json").write_text(json.dumps({"model_type": "toy", "vocab_size": EOS + 1, "eos_token_id": EOS}))
    (out / "model" / "generation_config.json").write_text(json.dumps({"eos_token_id": EOS}))
    (out / "model" / "toy.safetensors").write_bytes(safetensors({"salt": ("I64", [64], struct.pack("<64q", *SALT))}))
    toml = []
    for name in shapes:
        toml.append(f"[targets.{name}]\nmanifest = \"{name}.json\"\nreference = \"{name}-ref.json\"")
        toml.append('kernels = "kernels"\nweights = ["model/toy.safetensors"]\ntokenizer = "model/tokenizer.json"\n')
    (out / "kern.toml").write_text("\n".join(toml))
    print(f"{len(targets)} manifests, {b256[0]} and {b128[0]} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
