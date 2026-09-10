#!/usr/bin/env python3
"""Inspect and plan a DP/EP checkpoint without loading tensor payloads.

The returned tensor locations are also the input to the artifact exporter.
Dense tensors are replicated, routed experts have one EP owner, and Engram
tables have one shared host allocation. Sizes exclude repacking/workspaces.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import struct


def inventory(root):
    index = json.loads((root / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    tensors = {}
    headers = {}
    for filename in sorted(set(weight_map.values())):
        path = root / filename
        with path.open("rb") as stream:
            raw = stream.read(8)
            if len(raw) != 8:
                raise ValueError(f"truncated header: {filename}")
            length = struct.unpack("<Q", raw)[0]
            if length > path.stat().st_size - 8:
                raise ValueError(f"invalid header length: {filename}")
            raw_header = stream.read(length)
        header = json.loads(raw_header)
        headers[filename] = hashlib.sha256(raw_header).hexdigest()
        extent = path.stat().st_size - 8 - length
        end = 0
        entries = [(name, value) for name, value in header.items()
                   if name != "__metadata__"]
        for name, value in sorted(entries, key=lambda item: item[1]["data_offsets"]):
            start, stop = value["data_offsets"]
            if start != end or stop < start or stop > extent:
                raise ValueError(f"invalid tensor extent: {filename}: {name}")
            if name in tensors or weight_map.get(name) != filename:
                raise ValueError(f"index/header mismatch: {name}")
            tensors[name] = {
                "file": filename, "offset": 8 + length + start,
                "bytes": stop - start, "shape": value["shape"],
                "dtype": value["dtype"],
            }
            end = stop
        if end != extent:
            raise ValueError(f"unexpected trailing payload: {filename}")
    if tensors.keys() != weight_map.keys():
        raise ValueError("checkpoint index has missing tensor headers")
    return tensors, headers


def placement(name, config, ep):
    match = re.search(r"(?:mlp|ffn)\.experts\.(\d+)\.", name)
    if match:
        draft = name.startswith("mtp.")
        count = config["dspark_n_routed_experts" if draft else "n_routed_experts"]
        expert = int(match[1])
        if count % ep or expert >= count:
            raise ValueError(f"invalid EP partition: {name}")
        return ("draft_experts" if draft else "experts"), expert // (count // ep)
    if ".engram.embed." in name:
        return "host_engram", None
    if name.startswith(("vision.", "aligner.")):
        return "vision", None
    return ("draft_dense" if name.startswith("mtp.") else "dense"), None


def plan(tensors, config, ep):
    categories = Counter()
    ranks = [Counter() for _ in range(ep)]
    entries = {}
    for name, tensor in sorted(tensors.items()):
        category, owner = placement(name, config, ep)
        categories[category] += tensor["bytes"]
        entries[name] = dict(tensor, category=category, owner=owner)
        if category in ("host_engram", "vision"):
            continue
        for rank in range(ep) if owner is None else (owner,):
            ranks[rank][category] += tensor["bytes"]
    return {
        "ep": ep, "tensor_count": len(entries),
        "stored_bytes": dict(categories),
        "text_with_dspark_gpu_weight_bytes": [dict(rank) for rank in ranks],
        "text_with_dspark_gpu_weight_total_bytes": [sum(rank.values()) for rank in ranks],
        "shared_host_table_bytes": categories["host_engram"],
        "tensors": entries,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--ep", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.ep < 1:
        parser.error("--ep must be positive")
    config_bytes = (args.checkpoint / "config.json").read_bytes()
    config = json.loads(config_bytes)
    if config.get("model_type") != "deepseek_v41":
        parser.error("expected a deepseek_v41 checkpoint")
    tensors, headers = inventory(args.checkpoint)
    result = plan(tensors, config["text_config"], args.ep)
    result["config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
    result["header_sha256"] = headers
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("tensors", "header_sha256")}, indent=2))


if __name__ == "__main__":
    main()
