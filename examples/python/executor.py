"""Pure-Python single-GPU executor for a kern manifest (cubin + JSON + safetensors).

Runs a manifest directly through the CUDA Driver API (`cuda-python`), PyTorch
(device allocations, CUDA graphs, BF16 GEMM) and `safetensors`, with no Rust
runtime dependency. Built as a thin reference consumer so custom Python runners
and servers can drive any compliant manifest through `manifest.Protocol`.
"""

from __future__ import annotations

import ctypes
import hashlib
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch
from cuda.bindings import driver as cu
from safetensors import safe_open

from manifest import DTYPE_BYTES, Forward, Manifest, Sampling, eval_expr

NP_DTYPES: dict[str, Any] = {
    "u8": np.uint8,
    "i8": np.int8,
    "f32": np.float32,
    "i32": np.int32,
    "u32": np.uint32,
    "f64": np.float64,
    "i64": np.int64,
    "u64": np.uint64,
}

TORCH_DTYPES: dict[str, torch.dtype] = {
    "u8": torch.uint8,
    "i8": torch.int8,
    "fp8e4m3": torch.float8_e4m3fn,
    "bf16": torch.bfloat16,
    "f16": torch.float16,
    "f32": torch.float32,
    "i32": torch.int32,
    "i64": torch.int64,
}

TMA_DTYPES = {
    "u8": cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    "u16": cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT16,
    "u32": cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT32,
    "i32": cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_INT32,
    "u64": cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT64,
    "i64": cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_INT64,
    "f16": cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
    "bf16": cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    "f32": cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
    "tf32": cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_TFLOAT32,
}

TMA_SWIZZLE = {
    0: cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
    32: cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_32B,
    64: cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B,
    128: cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
}

TMA_L2 = {
    0: cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
    64: cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_64B,
    128: cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
    256: cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
}

_SCALAR_CTYPES = {
    "i32": ctypes.c_int32,
    "u32": ctypes.c_uint32,
    "i64": ctypes.c_int64,
    "u64": ctypes.c_uint64,
    "f32": ctypes.c_float,
    "u8": ctypes.c_uint8,
}


def _cu_check(res: Any) -> Any:
    err = res[0] if isinstance(res, tuple) else res
    if err != cu.CUresult.CUDA_SUCCESS:
        _, msg = cu.cuGetErrorString(err)
        raise RuntimeError(f"CUDA error {err}: {msg}")
    return res[1:] if isinstance(res, tuple) else None


def _encode_tensormap(tm: dict[str, Any], ptr: int, total_bytes: int) -> bytes:
    dims = list(tm["dims"])
    strides = list(tm["strides"])
    box = list(tm["box"])
    if dims[-1] == 0:
        dims[-1] = total_bytes // strides[-1]
    rank = len(dims)
    oob = (
        cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
        if tm.get("oob_nan", False)
        else cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    )
    (tmap,) = _cu_check(
        cu.cuTensorMapEncodeTiled(
            TMA_DTYPES[tm["dtype"]],
            rank,
            ptr,
            [cu.cuuint64_t(x) for x in dims],
            [cu.cuuint64_t(x) for x in strides],
            [cu.cuuint32_t(x) for x in box],
            [cu.cuuint32_t(1)] * rank,
            cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
            TMA_SWIZZLE[tm.get("swizzle", 0)],
            TMA_L2[tm.get("l2_promotion", 0)],
            oob,
        )
    )
    return struct.pack("<16Q", *(int(x) for x in tmap.opaque))


def _bf16_view(ptr: int, rows: int, cols: int, ld: int | None = None) -> torch.Tensor:
    stride_row = cols if ld is None else ld
    nbytes = ((rows - 1) * stride_row + cols) * 2 if rows > 0 else 0

    class _Buf:
        __cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }

    flat = torch.as_tensor(_Buf(), device="cuda").view(torch.bfloat16)
    return flat.as_strided((rows, cols), (stride_row, 1))


class Executor:
    """Single-GPU pure-Python executor for a `Manifest`."""

    def __init__(
        self,
        manifest: Manifest | str | Path,
        kernels_dir: str | Path,
        weights: list[str | Path],
        capacity_tokens: int = 4096,
        seq_slots: int = 2,
        gpu: int = 0,
    ):
        torch.cuda.set_device(gpu)
        torch.cuda.init()
        # Initialize cuBLAS once before any CUDA graph capture.
        _w = torch.empty((16, 16), device="cuda", dtype=torch.bfloat16)
        torch.mm(_w, _w)
        _cu_check(cu.cuInit(0))

        self.gpu = gpu
        self.manifest = manifest if isinstance(manifest, Manifest) else Manifest.load(manifest)
        self.protocol = self.manifest.protocol()
        self.capacity_tokens = capacity_tokens
        self.seq_slots = seq_slots

        # 1. Load sha256-pinned cubin modules.
        by_sha: dict[str, Path] = {}
        for p in Path(kernels_dir).iterdir():
            if p.is_file() and p.suffix == ".cubin":
                by_sha[hashlib.sha256(p.read_bytes()).hexdigest()] = p
        self._modules: dict[str, Any] = {}
        for name, mod in self.manifest.modules.items():
            sha = mod["sha256"]
            if sha not in by_sha:
                raise FileNotFoundError(f"module `{name}` sha256 {sha} not found in {kernels_dir}")
            (cmod,) = _cu_check(cu.cuModuleLoadData(by_sha[sha].read_bytes()))
            self._modules[name] = cmod
        self._funcs: dict[tuple[str, str], Any] = {}

        # 2. Allocate states, buffers and op-private scratch (zero-initialized).
        self.states: dict[str, torch.Tensor] = {
            name: torch.zeros(self.manifest.state_bytes(name, capacity_tokens, seq_slots), dtype=torch.uint8, device="cuda")
            for name in self.manifest.states
        }
        self.buffers: dict[str, torch.Tensor] = {
            name: torch.zeros(self.manifest.buffer_bytes(name), dtype=torch.uint8, device="cuda")
            for name in self.manifest.buffers
        }
        self._scratch: dict[tuple[str, str], torch.Tensor] = {}
        for op_name, op in self.manifest.ops.items():
            for sname, sbuf in op["impl"].get("scratch", {}).items():
                numel = 1
                for d in self.manifest.shape_at(sbuf["shape"]):
                    numel *= d
                self._scratch[(op_name, sname)] = torch.zeros(
                    numel * DTYPE_BYTES[sbuf["dtype"]], dtype=torch.uint8, device="cuda"
                )

        # 3. Load checkpoint weights via `bind`.
        self._load_weights([Path(p) for p in weights])

        # 4. Lower programs and run `once` setup programs.
        self._compiled: dict[str, list[dict[str, Any]]] = {
            pname: self._compile_program(prog) for pname, prog in self.manifest.programs.items()
        }
        self._graphs: dict[tuple[str, tuple[tuple[str, int], ...]], torch.cuda.CUDAGraph] = {}
        for pname in self.manifest.once_programs():
            self.run(pname, {}, graph=False)
        torch.cuda.synchronize()

    def _get_func(self, mod_name: str, entry: str, max_shmem: int) -> Any:
        key = (mod_name, entry)
        fn = self._funcs.get(key)
        if fn is None:
            (fn,) = _cu_check(cu.cuModuleGetFunction(self._modules[mod_name], entry.encode()))
            self._funcs[key] = fn
        if max_shmem > 48 * 1024:
            _cu_check(
                cu.cuFuncSetAttribute(
                    fn,
                    cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    max_shmem,
                )
            )
        return fn

    def _load_weights(self, weight_paths: list[Path]) -> None:
        files: list[Path] = []
        for p in weight_paths:
            files.extend(sorted(p.glob("*.safetensors")) if p.is_dir() else [p])
        index: dict[str, Path] = {}
        handles: dict[Path, Any] = {}
        for f in files:
            h = safe_open(str(f), framework="pt", device="cpu")
            handles[f] = h
            for k in h.keys():
                index[k] = f
        for name, buf in self.manifest.buffers.items():
            if buf["kind"] != "weight":
                continue
            dst = self.buffers[name]
            off = 0
            for seg in buf["bind"]:
                tname = seg["tensor"]
                t = handles[index[tname]].get_tensor(tname)
                if "rows" in seg:
                    r0, r1 = seg["rows"]
                    t = t[r0:r1]
                if "cols" in seg:
                    c0, c1 = seg["cols"]
                    t = t[:, c0:c1]
                raw = t.contiguous().view(torch.uint8).reshape(-1)
                n = raw.numel()
                dst[off : off + n].copy_(raw, non_blocking=True)
                off += n
            if off != dst.numel():
                raise ValueError(f"weight `{name}`: bound {off} bytes, expected {dst.numel()}")
        torch.cuda.synchronize()
        handles.clear()

    def _resolve_call_arg(self, arg: dict[str, Any]) -> tuple[str, Any, int]:
        if "buf" in arg:
            t = self.buffers[arg["buf"]]
            off = arg.get("offset", 0)
            return ("ptr", t.data_ptr() + off, t.numel() - off)
        if "state" in arg:
            t = self.states[arg["state"]]
            off = arg.get("offset", 0)
            return ("ptr", t.data_ptr() + off, t.numel() - off)
        if "var" in arg:
            return ("var", arg["var"], 0)
        if "expr" in arg:
            return ("expr", arg["expr"], 0)
        for k in ("i32", "i64", "u32", "u64", "f32", "u8"):
            if k in arg:
                return (k, arg[k], 0)
        raise ValueError(f"unsupported call arg: {arg!r}")

    def _compile_slot(self, larg: dict[str, Any], ptype: str, op_name: str, call_args: list[Any]) -> dict[str, Any]:
        if "param" in larg:
            ckind, cval, _ = call_args[larg["param"]]
            if ckind == "ptr":
                return {"type": "const_u64", "val": cval}
            if ckind in ("var", "expr"):
                return {"type": "dyn", "ptype": ptype, "expr": cval}
            return {"type": "const_scalar", "ptype": ptype, "val": cval}
        if "scratch" in larg:
            return {"type": "const_u64", "val": self._scratch[(op_name, larg["scratch"])].data_ptr()}
        if "pack" in larg:
            pack = larg["pack"]
            base = bytearray(pack["size"])
            dyn_fields: list[tuple[int, int, Any]] = []
            for fld in pack["fields"]:
                at = fld["at"]
                if "tensormap" in fld:
                    tm = fld["tensormap"]
                    _, ptr, nbytes = call_args[tm["param"]]
                    base[at : at + 128] = _encode_tensormap(tm, ptr, nbytes)
                elif "param" in fld:
                    ckind, cval, _ = call_args[fld["param"]]
                    w = fld.get("width")
                    if ckind == "ptr":
                        struct.pack_into("<Q", base, at, cval)
                    elif ckind in ("var", "expr"):
                        dyn_fields.append((at, w or 4, cval))
                    elif ckind == "f32":
                        struct.pack_into("<f", base, at, cval)
                    else:
                        fmt = {1: "<B", 4: "<i", 8: "<q"}[w or (8 if ckind in ("i64", "u64") else 4)]
                        struct.pack_into(fmt, base, at, cval)
                elif "scratch" in fld:
                    struct.pack_into("<Q", base, at, self._scratch[(op_name, fld["scratch"])].data_ptr())
                elif "var" in fld:
                    dyn_fields.append((at, fld.get("width", 4), fld["var"]))
                elif "expr" in fld:
                    dyn_fields.append((at, fld.get("width", 4), fld["expr"]))
                elif "i32" in fld:
                    struct.pack_into("<i", base, at, fld["i32"])
                elif "i64" in fld:
                    struct.pack_into("<q", base, at, fld["i64"])
                elif "u8" in fld:
                    struct.pack_into("<B", base, at, fld["u8"])
                elif "f32" in fld:
                    struct.pack_into("<f", base, at, fld["f32"])
                else:
                    raise ValueError(f"unsupported pack field: {fld!r}")
            return {"type": "pack", "base": bytes(base), "dyn": dyn_fields}
        for k in ("i32", "i64", "u32", "u64", "f32", "u8"):
            if k in larg:
                return {"type": "const_scalar", "ptype": ptype, "val": larg[k]}
        raise ValueError(f"unsupported launch arg: {larg!r}")

    def _compile_program(self, prog: dict[str, Any]) -> list[dict[str, Any]]:
        steps: list[dict[str, Any]] = []
        for call in prog["calls"]:
            op_name = call["op"]
            op = self.manifest.ops[op_name]
            call_args = [self._resolve_call_arg(a) for a in call["args"]]
            for launch in op["impl"]["launches"]:
                lparams = launch.get("params", op["params"])
                largs = launch.get("args") or [{"param": i} for i in range(len(lparams))]
                entry = launch["entry"]
                when = launch.get("when")
                slots = [self._compile_slot(a, pt, op_name, call_args) for a, pt in zip(largs, lparams)]
                if entry.startswith("extern:"):
                    steps.append({"kind": "extern", "entry": entry, "when": when, "args": slots})
                else:
                    shmem = launch.get("shared_mem", 0)
                    max_shmem = eval_expr(shmem, self.manifest.vars_max) if shmem else 0
                    fn = self._get_func(launch["module"], entry, max_shmem)
                    steps.append({
                        "kind": "kernel",
                        "fn": fn,
                        "when": when,
                        "block": launch["block"],
                        "grid": launch["grid"],
                        "shmem": shmem,
                        "args": slots,
                    })
        return steps

    def _slot_int(self, slot: dict[str, Any], vals: dict[str, int]) -> int:
        if slot["type"] == "const_u64":
            return slot["val"]
        if slot["type"] == "const_scalar":
            return int(slot["val"])
        if slot["type"] == "dyn":
            return eval_expr(slot["expr"], vals)
        raise ValueError(f"expected scalar/ptr slot, got {slot['type']}")

    def _issue(self, pname: str, vals: dict[str, int]) -> None:
        stream_ptr = torch.cuda.current_stream().cuda_stream
        for step in self._compiled[pname]:
            when = step["when"]
            if when is not None:
                v = vals[when["var"]]
                if ("min" in when and v < when["min"]) or ("max" in when and v > when["max"]):
                    continue

            if step["kind"] == "extern":
                entry = step["entry"]
                if entry in ("extern:cublaslt_bf16_tn", "extern:cublaslt_bf16_tn_acc"):
                    a_ptr, w_ptr, c_ptr, m, n, k = (self._slot_int(s, vals) for s in step["args"][:6])
                    ldc = self._slot_int(step["args"][6], vals) if len(step["args"]) > 6 else n
                    lda = self._slot_int(step["args"][7], vals) if len(step["args"]) > 7 else k
                    a = _bf16_view(a_ptr, m, k, lda)
                    w = _bf16_view(w_ptr, n, k)
                    c = _bf16_view(c_ptr, m, n, ldc)
                    if entry.endswith("_acc"):
                        c.addmm_(a, w.t())
                    else:
                        torch.mm(a, w.t(), out=c)
                    continue
                raise NotImplementedError(f"unsupported extern: {entry}")

            gx, gy, gz = (eval_expr(g, vals) for g in step["grid"])
            if gx == 0 or gy == 0 or gz == 0:
                continue
            bx, by, bz = step["block"]
            shmem = eval_expr(step["shmem"], vals) if step["shmem"] else 0

            keepers = []
            ptrs = (ctypes.c_void_p * len(step["args"]))()
            for i, slot in enumerate(step["args"]):
                st = slot["type"]
                if st == "const_u64":
                    cval: Any = ctypes.c_uint64(slot["val"])
                elif st == "const_scalar":
                    cval = _SCALAR_CTYPES[slot["ptype"]](slot["val"])
                elif st == "dyn":
                    cval = _SCALAR_CTYPES[slot["ptype"]](eval_expr(slot["expr"], vals))
                elif st == "pack":
                    buf = bytearray(slot["base"])
                    for at, w, expr in slot["dyn"]:
                        struct.pack_into({1: "<B", 4: "<i", 8: "<q"}[w], buf, at, eval_expr(expr, vals))
                    cval = (ctypes.c_ubyte * len(buf)).from_buffer(buf)
                else:
                    raise ValueError(st)
                keepers.append(cval)
                ptrs[i] = ctypes.addressof(cval)

            _cu_check(
                cu.cuLaunchKernel(
                    step["fn"], gx, gy, gz, bx, by, bz, shmem, stream_ptr, ctypes.addressof(ptrs), 0
                )
            )

    def run(self, program: str, vals: dict[str, int], graph: bool = True) -> None:
        """Issue `program` at `vals`, capturing a CUDA graph when declared and `graph` is True."""
        prog = self.manifest.programs[program]
        if graph and prog.get("graph", False):
            key = (program, tuple(sorted(vals.items())))
            if key not in self._graphs:
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self._issue(program, vals)
                self._graphs[key] = g
            self._graphs[key].replay()
        else:
            self._issue(program, vals)

    def buffer(self, name: str) -> torch.Tensor:
        """Zero-copy torch tensor view over buffer `name` at its max shape."""
        spec = self.manifest.buffers[name]
        shape = self.manifest.shape_at(spec["shape"])
        return self.buffers[name].view(TORCH_DTYPES[spec["dtype"]]).view(*shape)

    def write(self, name: str, arr: np.ndarray) -> None:
        raw = np.ascontiguousarray(arr).view(np.uint8).reshape(-1)
        self.buffers[name][: raw.size].copy_(torch.from_numpy(raw), non_blocking=True)

    def read(self, name: str, count: int) -> np.ndarray:
        dtype = NP_DTYPES[self.manifest.buffers[name]["dtype"]]
        nbytes = count * np.dtype(dtype).itemsize
        return self.buffers[name][:nbytes].cpu().numpy().view(dtype).copy()

    def zero_seq_slot(self, seq_slot: int) -> None:
        """Zero `seq_slot` in every per-sequence state (e.g. GDN / draft KV)."""
        for name, st in self.manifest.states.items():
            if "bytes_per_seq" in st:
                stride = st["bytes_per_seq"]
                self.states[name][seq_slot * stride : (seq_slot + 1) * stride].zero_()

    def write_tables(self, pages_per_seq: list[list[int]], seq_slots: list[int]) -> None:
        """Populate page tables and per-sequence line tables for a batch of sequences."""
        p = self.protocol
        for tbl in p.page_tables:
            shape = self.manifest.shape_at(list(tbl.shape))
            cols = shape[1] if len(shape) > 1 else shape[0]
            dt = NP_DTYPES[tbl.dtype]
            if len(shape) == 2:
                arr = np.zeros((len(pages_per_seq), cols), dtype=dt)
                for i, pages in enumerate(pages_per_seq):
                    n = min(len(pages), cols)
                    arr[i, :n] = pages[:n]
            else:
                arr = np.zeros(cols, dtype=dt)
                pages = pages_per_seq[0] if pages_per_seq else []
                n = min(len(pages), cols)
                arr[:n] = pages[:n]
            self.write(tbl.name, arr)

        for tbl in p.line_tables:
            shape = self.manifest.shape_at(list(tbl.shape))
            lines = np.zeros(shape, dtype=NP_DTYPES[tbl.dtype])
            for i, slot in enumerate(seq_slots):
                base = slot * tbl.lines_per_slot
                if lines.ndim == 2:
                    lines[:, i] = np.arange(base, base + shape[0], dtype=lines.dtype)
                else:
                    lines[:, i, 0] = np.arange(base, base + shape[0], dtype=lines.dtype)
            self.write(tbl.name, lines)

    def init_single_sequence(self, seq_slot: int = 1) -> None:
        """Populate page and per-sequence line tables for a single active sequence."""
        n_pages = self.capacity_tokens // self.protocol.page_unit
        self.write_tables([list(range(n_pages))], [seq_slot])

    def stage(
        self,
        fwd: Forward,
        positions: list[int],
        lens: list[int],
        token_ids: list[int],
        slots: list[int],
        sampling: Sampling | list[Sampling] = Sampling(),
    ) -> dict[str, int]:
        """Stage all protocol input fills for one call of `fwd`; returns its `vars`."""
        p = self.protocol
        groups = len(lens)
        rows = sum(lens)
        context = sum(pos + n for pos, n in zip(positions, lens))
        starts = []
        acc = 0
        for n in lens:
            starts.append(acc)
            acc += n

        samplings = [sampling] * groups if isinstance(sampling, Sampling) else sampling
        sampling_vals = {
            "temperature": [s.temperature for s in samplings],
            "top_k": [s.top_k for s in samplings],
            "top_p": [s.top_p for s in samplings],
            "min_p": [s.min_p for s in samplings],
            "seed": [s.seed for s in samplings],
        }

        for role, filled in p.inputs.items():
            dt = NP_DTYPES[filled.dtype]
            if role in sampling_vals:
                self.write(filled.name, np.array(sampling_vals[role], dtype=dt))
            elif role == "token":
                vals = [token_ids[s] for s in starts] if filled.axis == "groups" else token_ids
                self.write(filled.name, np.array(vals, dtype=dt))
            elif role == "valid":
                self.write(filled.name, np.ones(rows, dtype=dt))
            elif role == "position":
                vals = [pos + i for pos, n in zip(positions, lens) for i in range(n)]
                self.write(filled.name, np.array(vals, dtype=dt))
            elif role == "slot":
                self.write(filled.name, np.array(slots, dtype=dt))
            elif role == "seq_len":
                self.write(filled.name, np.array([pos + n for pos, n in zip(positions, lens)], dtype=dt))
            elif role == "cu_seqlens":
                self.write(filled.name, np.array(starts + [rows], dtype=dt))

        return p.vars_for(fwd, groups=groups, rows=rows, context=context)

    def emitted_batch(self, fwd: Forward, groups: int) -> list[list[int]]:
        """Read the tokens emitted by each of `groups` sequences on the last call of `fwd`."""
        if fwd.emits is None:
            return [[] for _ in range(groups)]
        counts = self.read(fwd.count, groups).tolist() if fwd.count else [1] * groups
        shape = self.manifest.buffers[fwd.emits]["shape"]
        width = int(shape[1]) if len(shape) > 1 else 1
        raw = self.read(fwd.emits, groups * width).tolist()
        return [raw[i * width : i * width + int(counts[i])] for i in range(groups)]

    def emitted(self, fwd: Forward) -> list[int]:
        """Read the tokens emitted by sequence 0 on the last call of `fwd`."""
        return self.emitted_batch(fwd, 1)[0]
