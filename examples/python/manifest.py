"""Pure-Python reader for a kern manifest and its serving protocol.

Reads a manifest JSON (schema v7), expands any `constants` table, evaluates
the closed scalar expression set (`ceil_div`, `mul`, `add`, `clamp`), and
projects the serving protocol (`Protocol`) so a driver or server never names
a model's buffers, vars or programs. Zero CUDA or Rust dependencies.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DTYPE_BYTES: dict[str, int] = {
    "u8": 1,
    "i8": 1,
    "fp8e4m3": 1,
    "bf16": 2,
    "f16": 2,
    "u16": 2,
    "i16": 2,
    "f32": 4,
    "i32": 4,
    "u32": 4,
    "f64": 8,
    "i64": 8,
    "u64": 8,
}

# Dict keys whose values are identifier/string references, never numeric constants.
_NAME_KEYS = frozenset({
    "model", "dtype", "kind", "fill", "axis", "index_into", "tensor",
    "source", "sha256", "module", "entry", "params", "op", "buf",
    "state", "var", "scratch", "context", "label", "lane", "after",
})


def _expand_constants(obj: Any, consts: dict[str, Any]) -> Any:
    if not consts:
        return obj
    if isinstance(obj, str):
        return consts.get(obj, obj)
    if isinstance(obj, list):
        return [_expand_constants(x, consts) for x in obj]
    if isinstance(obj, dict):
        return {k: v if k in _NAME_KEYS else _expand_constants(v, consts) for k, v in obj.items()}
    return obj


def eval_expr(expr: Any, vals: dict[str, int]) -> int:
    """Evaluate a manifest scalar expression at `vals`."""
    if isinstance(expr, int):
        return expr
    if isinstance(expr, str):
        return vals[expr]
    if isinstance(expr, dict):
        if "ceil_div" in expr:
            e, c = expr["ceil_div"]
            return (eval_expr(e, vals) + c - 1) // c
        if "mul" in expr:
            e, c = expr["mul"]
            return eval_expr(e, vals) * c
        if "add" in expr:
            a, b = expr["add"]
            return eval_expr(a, vals) + eval_expr(b, vals)
        if "clamp" in expr:
            e, start, length = expr["clamp"]
            return min(max(eval_expr(e, vals) - start, 0), length)
    raise ValueError(f"unsupported manifest expression: {expr!r}")


class Manifest:
    """A parsed kern manifest with its `constants` expanded."""

    def __init__(self, raw: dict[str, Any]):
        consts = raw.get("constants") or {}
        if consts:
            expanded = {k: _expand_constants(v, consts) for k, v in raw.items() if k != "constants"}
        else:
            expanded = raw
        if expanded.get("schema_version") != 7:
            raise ValueError(f"unsupported schema_version {expanded.get('schema_version')!r}; expected 7")
        self.raw: dict[str, Any] = expanded
        self.model: str = expanded["model"]
        self.vars: dict[str, dict[str, Any]] = expanded.get("vars", {})
        self.states: dict[str, dict[str, Any]] = expanded.get("states", {})
        self.buffers: dict[str, dict[str, Any]] = expanded["buffers"]
        self.modules: dict[str, dict[str, Any]] = expanded.get("modules", {})
        self.ops: dict[str, dict[str, Any]] = expanded["ops"]
        self.programs: dict[str, dict[str, Any]] = expanded["programs"]
        self.vars_max: dict[str, int] = {k: v["max"] for k, v in self.vars.items()}

    @classmethod
    def load(cls, path: str | Path) -> Manifest:
        return cls(json.loads(Path(path).read_text()))

    def shape_at(self, shape: list[int | str], vals: dict[str, int] | None = None) -> tuple[int, ...]:
        v = self.vars_max if vals is None else vals
        return tuple(v[d] if isinstance(d, str) else d for d in shape)

    def buffer_bytes(self, name: str) -> int:
        buf = self.buffers[name]
        return math.prod(self.shape_at(buf["shape"])) * DTYPE_BYTES[buf["dtype"]]

    def state_bytes(self, name: str, capacity_tokens: int, seq_slots: int) -> int:
        st = self.states[name]
        if "bytes_per_token" in st:
            return st["bytes_per_token"] * capacity_tokens
        if "bytes_per_seq" in st:
            return st["bytes_per_seq"] * seq_slots
        return st["bytes"]

    def once_programs(self) -> list[str]:
        return sorted(n for n, p in self.programs.items() if p.get("once"))

    def protocol(self) -> Protocol:
        return Protocol.from_manifest(self)


@dataclass(frozen=True)
class Filled:
    name: str
    fill: str
    dtype: str
    shape: tuple[int | str, ...]
    axis: str  # "rows" | "groups" | "ranks" | "fixed"


@dataclass(frozen=True)
class Table:
    name: str
    state: str
    dtype: str
    stride: int
    shape: tuple[int | str, ...]
    lines_per_slot: int  # 0 for page tables, >0 for per-seq line tables


@dataclass(frozen=True)
class Forward:
    name: str
    groups: int
    rows: int | str  # int for fixed-row decode/round, var name for chunk prefill
    context_var: str | None
    graph: bool
    emits: str | None  # output buffer with fill == "tokens" written by this program
    count: str | None  # output buffer with fill == "count" written by this program


@dataclass(frozen=True)
class Sampling:
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    seed: int = 0


def _written_buffers(m: Manifest, prog: dict[str, Any]) -> set[str]:
    written: set[str] = set()
    for call in prog["calls"]:
        op = m.ops[call["op"]]
        for param, arg in zip(op["params"], call["args"]):
            if (param.startswith("out ") or param.startswith("inout ")) and isinstance(arg, dict) and "buf" in arg:
                written.add(arg["buf"])
    return written


class Protocol:
    """Serving protocol projected from a manifest's `vars`, `buffers` and `programs`.

    A driver or server queries this object instead of hardcoding buffer,
    var or program names.
    """

    def __init__(
        self,
        rows_var: str | None,
        groups_var: str | None,
        rows_max: int,
        groups_max: int,
        inputs: dict[str, Filled],
        outputs: dict[str, list[Filled]],
        page_tables: list[Table],
        line_tables: list[Table],
        forwards: dict[str, Forward],
    ):
        self.rows_var = rows_var
        self.groups_var = groups_var
        self.rows_max = rows_max
        self.groups_max = groups_max
        self.inputs = inputs
        self.outputs = outputs
        self.page_tables = page_tables
        self.line_tables = line_tables
        self.forwards = forwards

    @classmethod
    def from_manifest(cls, m: Manifest) -> Protocol:
        rows_var = next((k for k, v in m.vars.items() if v.get("axis") == "rows"), None)
        groups_var = next((k for k, v in m.vars.items() if v.get("axis") == "groups"), None)
        ranks_var = next((k for k, v in m.vars.items() if v.get("axis") == "ranks"), None)

        def axis_of(shape: list[int | str]) -> str:
            if not shape:
                return "fixed"
            d0 = shape[0]
            if d0 == rows_var:
                return "rows"
            if d0 == groups_var:
                return "groups"
            if d0 == ranks_var:
                return "ranks"
            return "fixed"

        inputs: dict[str, Filled] = {}
        outputs: dict[str, list[Filled]] = {}
        page_tables: list[Table] = []
        line_tables: list[Table] = []

        for name, buf in m.buffers.items():
            kind = buf["kind"]
            fill = buf.get("fill")
            if fill is not None:
                f = Filled(name=name, fill=fill, dtype=buf["dtype"], shape=tuple(buf["shape"]), axis=axis_of(buf["shape"]))
                if kind in ("input", "inout"):
                    inputs[fill] = f
                elif kind == "output":
                    outputs.setdefault(fill, []).append(f)
                continue

            dom = buf.get("domain")
            if kind in ("input", "inout") and isinstance(dom, dict) and "index_into" in dom:
                target = dom["index_into"]
                if target in m.states:
                    st = m.states[target]
                    stride = dom.get("stride", 1)
                    if "bytes_per_token" in st:
                        page_tables.append(Table(name, target, buf["dtype"], stride, tuple(buf["shape"]), 0))
                    elif "bytes_per_seq" in st:
                        lines_per_slot = st["bytes_per_seq"] // stride
                        line_tables.append(Table(name, target, buf["dtype"], stride, tuple(buf["shape"]), lines_per_slot))

        tok_outs = {f.name for f in outputs.get("tokens", [])}
        cnt_outs = {f.name for f in outputs.get("count", [])}
        forwards: dict[str, Forward] = {}
        for pname, prog in m.programs.items():
            batch = prog.get("batch")
            if not batch:
                continue
            written = _written_buffers(m, prog)
            emits = next((n for n in written if n in tok_outs), None)
            count = next((n for n in written if n in cnt_outs), None)
            forwards[pname] = Forward(
                name=pname,
                groups=batch["groups"],
                rows=batch["rows"],
                context_var=batch.get("context"),
                graph=bool(prog.get("graph", False)),
                emits=emits,
                count=count,
            )

        return cls(
            rows_var=rows_var,
            groups_var=groups_var,
            rows_max=m.vars_max[rows_var] if rows_var else 1,
            groups_max=m.vars_max[groups_var] if groups_var else 1,
            inputs=inputs,
            outputs=outputs,
            page_tables=page_tables,
            line_tables=line_tables,
            forwards=forwards,
        )

    @property
    def page_unit(self) -> int:
        return max((t.stride for t in self.page_tables), default=1)

    def chunk_forward(self) -> Forward:
        """The variable-rows forward used for prompt prefill."""
        for f in self.forwards.values():
            if isinstance(f.rows, str):
                return f
        raise ValueError("manifest declares no variable-rows prefill program")

    def step_forward(self, rows: int | None = None, groups: int = 1) -> Forward:
        """A fixed-row forward matching `rows` (or the widest declared when `rows` is None)."""
        cands = [f for f in self.forwards.values() if isinstance(f.rows, int) and f.groups >= groups]
        if rows is not None:
            cands = [f for f in cands if f.rows == rows]
        if not cands:
            raise ValueError(f"no forward for rows={rows}, groups>={groups}")
        # Prefer the widest rows when unspecified, and the smallest groups bound that fits.
        cands.sort(key=lambda f: (-f.rows, f.groups))
        return cands[0]

    def vars_for(self, fwd: Forward, groups: int, rows: int, context: int | None = None) -> dict[str, int]:
        v: dict[str, int] = {}
        if self.groups_var:
            v[self.groups_var] = groups
        if self.rows_var:
            v[self.rows_var] = rows
        if fwd.context_var and context is not None:
            v[fwd.context_var] = context
        return v
