#!/usr/bin/env python3
"""Shard kern's exported K3 dense layers for a tray group of R ranks.

    python3 tools/shard_k3_tp.py --weights /data/kern-k3/full --tp 4 [--layers 93]

Reads `dense/l{i}.safetensors` (tools/export_k3.py) and writes
`dense-tp{R}/r{r}/l{i}.safetensors` per rank, the layout the `--tp R`
manifests of tools/gen_k3.py load in place of `dense/l{i}`:

  KDA layers — the head-sliced tensors carry rank r's heads
  [r*HEADS/R, (r+1)*HEADS/R):
    wbig    [4*INNER, H]  each of the q|k|v|g segments row-sliced by head
    wsm     [256, H]      rows 0..HEADS/R = b_proj of the rank's heads, rows
                          96..224 = f_a (whole), the rest zero — the kernel's
                          whole-model layout (beta at column h, f_a at 96)
    w_f_b   [INNER, 128]  rows;   cw [3, 4, INNER], dt_bias [INNER]  last axis
    a_log   [HEADS]       slice;  w_o [H, INNER]  columns
  every layer — the shared expert (or layer 0's dense FFN), gate / up rows
  and down columns [r*I/R, (r+1)*I/R) of each:
    wsh  [2*SHARED, H]  -> [2*SHARED/R, H]   sh_down [H, SHARED]  -> [H, SHARED/R]
    wgu  [2*DENSE_I, H] -> [2*DENSE_I/R, H]  w_dn    [H, DENSE_I] -> [H, DENSE_I/R]
  everything else (norms, scoring weights, gamma_o, the MLA projections,
  the router, lat_down / lat_up, the routed experts' own files) is copied
  whole. Existing output files are rewritten.

Pure Python on the safetensors bytes (no torch / numpy on the trays):
bf16 rows are copied as bytes, the strided `w_o` columns row by row.
"""
import argparse
import json
import os
import pathlib
import struct
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gen_k3 import HEADS, HEAD_DIM, INNER, WSM, SHARED, DENSE_I, LAYERS, H, is_mla  # noqa: E402

ITEM = {"BF16": 2, "F32": 4, "I32": 4, "U8": 1}


def read(path):
    """(header dict, data bytes) of a safetensors file."""
    raw = path.read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    hdr = json.loads(raw[8 : 8 + n])
    hdr.pop("__metadata__", None)
    return hdr, memoryview(raw)[8 + n :]


def write(path, tensors):
    """`tensors`: name -> (dtype, shape, bytes)."""
    hdr, off, blobs = {}, 0, []
    for name in sorted(tensors):
        dt, shape, b = tensors[name]
        assert len(b) == ITEM[dt] * prod(shape), (name, dt, shape, len(b))
        hdr[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    h = json.dumps(hdr, separators=(",", ":")).encode()
    h += b" " * (-(8 + len(h)) % 8)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)
    os.replace(tmp, path)


def prod(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def rows(t, lo, hi):
    """Rows lo..hi of a row-major tensor: contiguous bytes."""
    dt, shape, b = t
    row = ITEM[dt] * prod(shape[1:])
    return dt, [hi - lo] + list(shape[1:]), bytes(b[lo * row : hi * row])


def cols(t, lo, hi):
    """Columns lo..hi of a 2-D row-major tensor: one slice per row."""
    dt, shape, b = t
    n, w = shape
    it = ITEM[dt]
    out = bytearray()
    for r in range(n):
        base = (r * w + lo) * it
        out += b[base : base + (hi - lo) * it]
    return dt, [n, hi - lo], bytes(out)


def last_axis(t, lo, hi):
    """[.., W] -> [.., hi-lo]: the leading axes flattened are rows of `cols`."""
    dt, shape, b = t
    lead = prod(shape[:-1])
    dt2, _, out = cols((dt, [lead, shape[-1]], b), lo, hi)
    return dt2, list(shape[:-1]) + [hi - lo], out


def gate_up(t, inter, lo, hi):
    """Rows lo..hi of the gate half and of the up half of a [2*inter, H]."""
    dt, shape, _ = t
    assert shape[0] == 2 * inter, (shape, inter)
    g, u = rows(t, lo, hi), rows(t, inter + lo, inter + hi)
    return dt, [2 * (hi - lo)] + list(shape[1:]), g[2] + u[2]


def shard_layer(tensors, i, r, tp):
    """Rank r's copy of layer i (name -> (dtype, shape, bytes))."""
    p = f"layers.{i}."
    out = dict(tensors)
    t = lambda n: tensors[p + n]
    if i == 0:
        dn = DENSE_I // tp
        out[p + "wgu"] = gate_up(t("wgu"), DENSE_I, r * dn, (r + 1) * dn)
        out[p + "w_dn"] = cols(t("w_dn"), r * dn, (r + 1) * dn)
    else:
        sh = SHARED // tp
        out[p + "wsh"] = gate_up(t("wsh"), SHARED, r * sh, (r + 1) * sh)
        out[p + "sh_down"] = cols(t("sh_down"), r * sh, (r + 1) * sh)
    if is_mla(i):
        return out
    hl = HEADS // tp
    h0, h1 = r * hl, (r + 1) * hl
    d0, d1 = h0 * HEAD_DIM, h1 * HEAD_DIM
    wbig = t("wbig")
    assert wbig[1] == [4 * INNER, H]
    segs = [rows(wbig, s * INNER + d0, s * INNER + d1) for s in range(4)]
    out[p + "wbig"] = ("BF16", [4 * hl * HEAD_DIM, H], b"".join(s[2] for s in segs))
    wsm = t("wsm")
    assert wsm[1] == [WSM, H]
    row = 2 * H
    z = bytes(row)
    body = bytes(wsm[2][h0 * row : h1 * row]) + z * (HEADS - hl) + bytes(wsm[2][HEADS * row : (HEADS + HEAD_DIM) * row])
    body += z * (WSM - HEADS - HEAD_DIM)
    out[p + "wsm"] = ("BF16", [WSM, H], body)
    out[p + "w_f_b"] = rows(t("w_f_b"), d0, d1)
    out[p + "cw"] = last_axis(t("cw"), d0, d1)
    out[p + "dt_bias"] = rows(t("dt_bias"), d0, d1)
    out[p + "a_log"] = rows(t("a_log"), h0, h1)
    out[p + "w_o"] = cols(t("w_o"), d0, d1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--layers", type=int, default=LAYERS)
    a = ap.parse_args()
    assert HEADS % a.tp == 0 and SHARED % a.tp == 0 and DENSE_I % a.tp == 0
    root = pathlib.Path(a.weights)
    t0 = time.time()
    for r in range(a.tp):
        (root / f"dense-tp{a.tp}" / f"r{r}").mkdir(parents=True, exist_ok=True)
    for i in range(a.layers):
        src = root / "dense" / f"l{i}.safetensors"
        hdr, data = read(src)
        tensors = {n: (m["dtype"], m["shape"], data[m["data_offsets"][0] : m["data_offsets"][1]]) for n, m in hdr.items()}
        for r in range(a.tp):
            dst = root / f"dense-tp{a.tp}" / f"r{r}" / f"l{i}.safetensors"
            if dst.is_symlink():
                dst.unlink()
            write(dst, shard_layer(tensors, i, r, a.tp))
        print(f"layer {i} sharded ({time.time() - t0:.0f}s)", flush=True)
    print(f"done: {a.layers} layers, tp{a.tp}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
