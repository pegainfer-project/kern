"""Shared launch-building helpers for the GLM-5.3 generator.

Every mined kernel is declared from the skeleton ($GLM53_ARTIFACTS/
skeleton.json: symbol -> pinned module sha256) plus a hand-written
wiring table in the ops_*.py files. The launch's `cubin` is the dump module
file (resolved to a `<name>-<sha12>.cubin` bundle artifact by gen.py);
identity is the sha256. Grid expressions use the manifest's closed set:
"tokens" / "seqs" vars, {"ceil_div": [e, c]}, {"mul": [e, c]}.

Module variants: some symbols JIT to several cubins (triton specialization,
tilelang constexpr). `mined_file` pins an explicit dump module file; the
rule is to pick the LEAST specialized variant (no divisibility/value
assumptions) so one module serves every batch size:
  big_fuse n_splits=64 -> module_127 (attn pre), n_splits=8 -> module_231 (ffn pre)
  conv update          -> module_473 (batch as plain runtime arg)
  norm_gated           -> module_399 (T with no div-16 assumption)
"""

import os
import hashlib
import json
import pathlib

SKEL = json.loads((pathlib.Path(os.environ.get("GLM53_ARTIFACTS", "glm53-artifacts")) / "skeleton.json").read_text())
DUMP_DIR = pathlib.Path(__file__).resolve().parents[2] / "dumped-kernels-glm53-sglang"
HAND_DIR = pathlib.Path(__file__).resolve().parents[2] / "kernels-glm53-handwritten"

_sha_path = None


def sha_path(sha):
    global _sha_path
    if _sha_path is None:
        _sha_path = {}
        for mod in DUMP_DIR.glob("module_*.cubin"):
            _sha_path[hashlib.sha256(mod.read_bytes()).hexdigest()] = mod.name
    return _sha_path[sha]


T = "tokens"
S = "seqs"


def cdiv(e, c):
    return {"ceil_div": [e, c]}


def mul(e, c):
    return {"mul": [e, c]}


def a(i, offset=0):
    out = {"param": i}
    if offset:
        out["offset"] = offset
    return out


def scr(name):
    return {"scratch": name}


def i32(v):
    return {"i32": v}


def i64(v):
    return {"i64": v}


def f32(v):
    return {"f32": v}


def var(name):
    return {"var": name}


def rank(g="ep"):
    return {"rank": g}


def expr(e):
    return {"expr": e}


def pack(size, *fields):
    return {"pack": {"size": size, "fields": list(fields)}}


def tmap(param, dtype, dims, strides, box, swizzle=128, l2=256, at=0):
    """One tensormap field of a 128-byte pack (must sit at a 64-byte multiple)."""
    return {"at": at, "tensormap": {"param": param, "dtype": dtype, "dims": dims,
                                    "strides": strides, "box": box, "swizzle": swizzle,
                                    "l2_promotion": l2}}


def sym(sub):
    """The unique skeleton symbol containing `sub`."""
    hits = [k for k in SKEL if sub in k]
    assert len(hits) == 1, f"{sub}: {len(hits)} skeleton matches"
    return hits[0]


def module_of(sub):
    """(label, cubin_file, sha256) pinned for a mined symbol (substring match)."""
    entry = SKEL[sym(sub)]
    assert entry["pinned"], f"{sub}: no pinned module"
    regs, sha = entry["pinned"]
    return sym(sub)[:60], sha_path(sha), sha


def module_file(fname, entry_sub):
    """(label, cubin_file, sha256) for an explicit dump module file."""
    mod = DUMP_DIR / fname
    sha = hashlib.sha256(mod.read_bytes()).hexdigest()
    return entry_sub[:60], fname, sha


def hand(name, entry):
    """(label, cubin_file, sha256) for a handwritten kernel built into HAND_DIR."""
    mod = HAND_DIR / f"{name}.cubin"
    sha = hashlib.sha256(mod.read_bytes()).hexdigest()
    return entry[:60], f"{name}.cubin", sha


def mined(sub, params, block, grid, args, smem=0, pdl=False):
    """One launch of a skeleton-pinned mined kernel."""
    label, cubin, sha = module_of(sub)
    launch = {
        "cubin": cubin, "sha256": sha, "label": label, "entry": sym(sub),
        "params": params, "block": block, "grid": grid, "args": args,
    }
    if smem:
        launch["shared_mem"] = smem
    if pdl:
        launch["pdl"] = True
    return launch


def mined_file(fname, entry_sub, params, block, grid, args, smem=0, pdl=False):
    """One launch of an explicit dump module (variant override)."""
    label, cubin, sha = module_file(fname, entry_sub)
    launch = {
        "cubin": cubin, "sha256": sha, "label": label, "entry": sym(entry_sub),
        "params": params, "block": block, "grid": grid, "args": args,
    }
    if smem:
        launch["shared_mem"] = smem
    if pdl:
        launch["pdl"] = True
    return launch


def handwritten(name, entry, params, block, grid, args, smem=0):
    """One launch of a handwritten kernel (kernels-glm53-handwritten/<name>.cubin)."""
    label, cubin, sha = hand(name, entry)
    launch = {
        "cubin": cubin, "sha256": sha, "label": label, "entry": entry,
        "params": params, "block": block, "grid": grid, "args": args,
    }
    if smem:
        launch["shared_mem"] = smem
    return launch


def extern(name, params, args):
    return {"entry": f"extern:{name}", "params": params, "args": args}


# Fixed-grid TP8 Lamport ABI; phases are in the exported carry so bench
# snapshots cannot rewind them separately from peer-owned payload writes.
AR_CTAS = 16
AR_SLOT_BYTES = 16 * 4096 * 2
AR_SYM_BYTES = 2 * 8 * AR_SLOT_BYTES + 4 * AR_CTAS


def allreduce_bf16(x, count):
    """Baseline wire; gen.py replaces only these launches in Lamport mode."""
    return extern("nccl_allreduce_bf16",
                  ["inout buffer<bf16>", "inout buffer<bf16>", "i64", "i32"],
                  [dict(x), dict(x), count, rank()])


def lamport_buffers():
    return {
        "ar_sym": {"dtype": "u8", "shape": [AR_SYM_BYTES], "kind": "carry", "export": True},
        "ar_peers": {"dtype": "u64", "shape": [8], "kind": "peer", "of": "ar_sym", "group": "tp"},
        # Sticky status, host-readable with Runtime::read_output between steps.
        # Output buffers are allocated zeroed and not rewound by bench snapshots.
        "ar_error": {"dtype": "i32", "shape": [1], "kind": "output"},
    }


def lamport_allreduce_bf16(x, sym, peers, err, pdl=False):
    entry = "glm53_ar_lamport" + ("_pdl" if pdl else "")
    launch = handwritten("glm53_ar_lamport", entry,
                         ["inout buffer<bf16>", "inout buffer<bf16>",
                          "inout buffer<u8>", "in buffer<u64>", "out buffer<i32>",
                          "i32", "i32", "i64"],
                         [128, 1, 1], [AR_CTAS, 1, 1],
                         [dict(x), dict(x), sym, peers, err, rank("tp"), var(S), i64(1_000_000_000)])
    if pdl:
        launch["pdl"] = True
    return launch
