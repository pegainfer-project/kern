"""The `load` program under construction: what the checkpoint does not hold
is a `carry` buffer that a once-after-load program computes from the
tensors it does (tools/kernels-src/weight_prep.cu, k3_weight_prep.cu). No
exported artifact, no torch: `kern run --weights <HF snapshot dir>`.

A generator declares its weight buffers with `bind` segments straight
from the checkpoint (`seg` / `bind`), turns a buffer into a carry with
`derive`, queues the calls that fill it (`call` and the named helpers), and
`finish` turns the queue into the ops and the program. Ops are shared; each
op's grid covers its widest call, and every kernel bounds by its length
argument so a wider grid is harmless.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from handwritten import hw  # noqa: E402
from kern_manifest import program  # noqa: E402

BLOCK = 256

# op -> (cubin, entry, params); every launch is one thread per element in
# blocks of BLOCK, except rope_table (one block of 64 per table row).
KERNELS = {
    "weight_p1": ("weight_prep", "kern_weight_p1_bf16_f32", ["in buffer<bf16>", "out buffer<f32>", "i32"]),
    "cast_f32": ("weight_prep", "kern_cast_bf16_f32", ["in buffer<bf16>", "out buffer<f32>", "i32"]),
    "fill_f32": ("weight_prep", "kern_fill_f32", ["out buffer<f32>", "i32", "f32"]),
    "fill_i32": ("weight_prep", "kern_fill_i32", ["out buffer<i32>", "i32", "i32"]),
    "fill_i64": ("weight_prep", "kern_fill_i64", ["out buffer<i64>", "i32", "i64"]),
    "fill_u8": ("weight_prep", "kern_fill_u8", ["out buffer<u8>", "i32", "i32"]),
    "iota_i32": ("weight_prep", "kern_iota_i32", ["out buffer<i32>", "i32", "i32", "i32", "i32"]),
    "rope_table": ("weight_prep", "kern_rope_table_bf16",
                   ["out buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i32", "f32"]),
    "fill_bf16": ("k3_weight_prep", "kern_fill_bf16", ["out buffer<bf16>", "i32", "f32"]),
    "k3_conv_taps": ("k3_weight_prep", "kern_k3_conv_taps",
                     ["in buffer<f32>", "in buffer<f32>", "in buffer<f32>", "out buffer<f32>", "i32"]),
    "k3_scoring": ("k3_weight_prep", "kern_k3_scoring", ["in buffer<bf16>", "in buffer<bf16>", "out buffer<f32>", "i32"]),
    "k3_wsm": ("k3_weight_prep", "kern_k3_wsm", ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32"]),
    "k3_mega_sf_pack": ("k3_weight_prep", "kern_k3_mega_sf_pack",
                        ["in buffer<u8>", "out buffer<i32>", "i32", "i32", "i32", "i32"]),
    "k3_kvb_aug": ("k3_weight_prep", "kern_k3_kvb_aug", ["in buffer<bf16>", "out buffer<bf16>", "i32"]),
}
FILL = {"f32": ("fill_f32", "f32"), "bf16": ("fill_bf16", "f32"), "i32": ("fill_i32", "i32"),
        "i64": ("fill_i64", "i64"), "u8": ("fill_u8", "i32")}


def seg(tensor, rows=None, cols=None, interleave=None):
    """One bind segment: a checkpoint tensor (a name, or a per-rank table),
    optionally a row / column range (a `[from, to)` or a per-rank table),
    or interleaved block-wise with a second tensor of the same shape."""
    s = {"tensor": tensor}
    if rows is not None:
        s["rows"] = list(rows) if not isinstance(rows, dict) else rows
    if cols is not None:
        s["cols"] = list(cols) if not isinstance(cols, dict) else cols
    if interleave is not None:
        s["interleave"] = interleave
    return s


def bind(*tensors):
    """Whole tensors laid end to end (a merged projection)."""
    return [seg(t) for t in tensors]


def buf(n, off=0):
    return {"buf": n, "offset": off} if off else {"buf": n}


class Once:
    def __init__(self, m):
        self.m = m
        self.calls = []
        self.widest = {}

    def call(self, op, args, n):
        self.widest[op] = max(self.widest.get(op, 0), n)
        self.calls.append({"label": f"load.{op}.{len(self.calls)}", "op": op, "args": args})

    def derive(self, name, raw=None, dtype="bf16"):
        """Turn weight buffer `name` into a carry; when the value comes from
        a checkpoint tensor, add `raw` bound to it (same shape, `dtype`)."""
        b = self.m["buffers"][name]
        b["kind"] = "carry"
        b.pop("bind", None)
        if raw is not None:
            rname, tensor = raw
            self.m["buffers"][rname] = {"dtype": dtype, "shape": list(b["shape"]), "kind": "weight", "bind": bind(tensor)}
        return b

    def elems(self, name):
        n = 1
        for d in self.m["buffers"][name]["shape"]:
            n *= d
        return n

    def p1(self, name, tensor):
        """`name` (f32, `<x>.weight_p1`) = float(<x>.weight) + 1."""
        raw = name[: -len("_p1")]
        self.derive(name, (raw, tensor))
        self.call("weight_p1", [buf(raw), buf(name), {"i32": self.elems(name)}], self.elems(name))

    def cast(self, name, tensor):
        raw = name + ".bf16"
        self.derive(name, (raw, tensor))
        self.call("cast_f32", [buf(raw), buf(name), {"i32": self.elems(name)}], self.elems(name))

    def fill(self, name, value):
        b = self.derive(name)
        n = self.elems(name)
        op, lit = FILL[b["dtype"]]
        self.call(op, [buf(name), {"i32": n}, {lit: value}], n)

    def iota(self, name, n, start=0, step=1, stride=1, off=0):
        self.call("iota_i32", [buf(name, off), {"i32": n}, {"i32": start}, {"i32": step}, {"i32": stride}], n)

    def rope(self, cos, sin, rows, half, stride, base):
        """vLLM's cos/sin cache: `cos`/`sin` are (buffer, byte offset)."""
        for name, _ in (cos, sin):
            if self.m["buffers"][name]["kind"] == "weight":
                self.derive(name)
        self.call("rope_table", [buf(*cos), buf(*sin), {"i32": rows}, {"i32": half}, {"i32": stride}, {"f32": base}], rows)

    def finish(self):
        for op, n in self.widest.items():
            cubin, entry, params = KERNELS[op]
            grid, block = ([n, 1, 1], [64, 1, 1]) if op == "rope_table" else ([(n + BLOCK - 1) // BLOCK, 1, 1], [BLOCK, 1, 1])
            assert op not in self.m["ops"], op
            self.m["ops"][op] = {"params": params, "impl": {"launches": [
                {"entry": entry, "params": params, "block": block, "grid": grid,
                 "args": [{"param": i} for i in range(len(params))], **hw(cubin)}]}}
        assert "load" not in self.m["programs"]
        self.m["programs"]["load"] = program(self.calls, once=True)
        return self.m
