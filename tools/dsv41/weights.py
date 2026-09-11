"""Direct checkpoint bindings for replicated dense and EP-local experts.

No tensor bytes are converted or written here. The manifest selects expert
names by rank and assembles their original storage into contiguous buffers.
Kernel-specific transforms consume these buffers in the once program.
"""

import re


DTYPES = {
    "BF16": "bf16", "F16": "f16", "F32": "f32",
    "F8_E4M3": "fp8e4m3", "F8_E8M0": "fp8e8m0", "I8": "i8",
    "I32": "i32", "U32": "u32", "I64": "i64", "U64": "u64", "U8": "u8",
}
EXPERT = re.compile(r"^(.*\.(?:ffn|mlp)\.experts)\.(\d+)\.(.*)$")


def raw(tensor, source):
    return {"dtype": DTYPES[tensor["dtype"]], "shape": tensor["shape"],
            "kind": "weight", "bind": [{"tensor": source}]}


def sharded(tensor, source, ep):
    """One table split into `ep` equal row slices, one per rank, exported so
    the group reads the whole table through a peer array. The last slice
    starts early enough to end at the table's end, so all slices are the
    buffer's one shape (rows overlap, never run past the table)."""
    total = tensor["shape"][0]
    per_rank = -(-total // ep)
    starts = [min(rank * per_rank, total - per_rank) for rank in range(ep)]
    return {"dtype": DTYPES[tensor["dtype"]], "shape": [per_rank, *tensor["shape"][1:]],
            "kind": "weight", "export": True,
            "bind": [{"tensor": source, "rows": {"group": "ep", "ranges": [[s, s + per_rank] for s in starts]}}]}


def peers(of, ep):
    return {"kind": "peer", "dtype": "u64", "shape": [ep], "of": of, "group": "ep"}


def bindings(tensors, config, ep=4, engram="host"):
    """Return (GPU buffers, shared host tensors) for the text and draft model.

    Grouped raw expert buffers have shape [local_experts, *checkpoint_shape].
    Their scales retain E8M0 bytes; FP4 weights retain packed signed bytes.
    Host descriptors declare mapped host placement. The serving loader shares
    one checkpoint scope across ranks; these never become HBM weights.
    The Engram tables are host descriptors (`engram="host"`: one copy in the
    tray's host memory, read through ATS; a GB300) or sharded into HBM across
    the EP group with a `<name>.peers` array each (`engram="device"`: a
    device that cannot read host memory at speed; an HGX B300).
    """
    if ep < 1:
        raise ValueError("EP must be positive")
    if engram not in ("host", "device"):
        raise ValueError("engram tables live in host memory or on the devices")
    gpu, host, groups = {}, {}, {}
    for name, tensor in sorted(tensors.items()):
        if name.startswith(("vision.", "aligner.")):
            continue
        if ".engram.embed." in name:
            if engram == "host":
                host[name] = dict(raw(tensor, name), placement="host")
            else:
                gpu[name] = sharded(tensor, name, ep)
                gpu[name + ".peers"] = peers(name, ep)
            continue
        match = EXPERT.fullmatch(name)
        if match is None:
            gpu[name] = raw(tensor, name)
        else:
            prefix, expert, suffix = match.groups()
            groups.setdefault((prefix, suffix), {})[int(expert)] = (name, tensor)
    for (prefix, suffix), entries in sorted(groups.items()):
        count = config["dspark_n_routed_experts" if prefix.startswith("mtp.") else "n_routed_experts"]
        if count % ep or set(entries) != set(range(count)):
            raise ValueError(f"incomplete or indivisible expert group {prefix}.{suffix}")
        local = count // ep
        first = entries[0][1]
        if any(t["shape"] != first["shape"] or t["dtype"] != first["dtype"]
               for _, t in entries.values()):
            raise ValueError(f"nonuniform expert tensor {prefix}.{suffix}")
        gpu[f"{prefix}.{suffix}"] = {
            "dtype": DTYPES[first["dtype"]], "shape": [local, *first["shape"]],
            "kind": "weight",
            "bind": [{"tensor": {"group": "ep", "tensors": [
                entries[rank * local + i][0] for rank in range(ep)
            ]}} for i in range(local)],
        }
    return gpu, host
