#!/usr/bin/env python3
"""Opt-in FA3 preparation experiment for the audited GLM-5.3-Flash ABI.

This is not a generator default or a performance policy. It changes only
three preparation launches in an existing 32-row-capacity MTP manifest.
See docs/glm53/fa3_schedule_control.md for the evidence and open gates.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path


OPS = ("dsa_attn", "mtp16_dsa_attn", "mtp32_dsa_attn")
MAX_ROWS = 32
SM_HINT = 132 * MAX_ROWS
MODULE_SHA256 = (
    "f691748aa14be4a6b1572b276b9d81a49729f34f6b7d87210c4b9f0ae7ef28d9",
    "6e1d657d3fa2bf11f71777e737e8cc00edcdc4e50f0a7ddd4c26582201c585bd",
    "0e1e3dbcaf559715e150d3c386f1ad8255bd091915cd87936d7e4dc7479575b6",
)
# Canonical JSON of the complete original preparation launch, except its
# arbitrary module alias. This pins all 25 arguments, types, entry and grid.
PREPARE_ABI_SHA256 = "16d495fe47a58d41a66bdf7eaa6c865701bbb21a3d32cf2ab139fce83f0e81a0"
PARAMS = [
    "in buffer<bf16>", "in state", "in buffer<i32>", "in buffer<i32>",
    "in buffer<i32>", "out buffer<bf16>", "i32",
]


def prepare_abi_sha256(launch):
    descriptor = {k: v for k, v in launch.items() if k != "module"}
    data = json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def _validate(source):
    if source["model"] != "GLM-5.3-Flash":
        raise ValueError("this experiment requires the exact GLM-5.3-Flash model")
    tokens, seqs = (source["vars"][v]["max"] for v in ("tokens", "seqs"))
    if type(tokens) is not int or tokens != MAX_ROWS:
        raise ValueError("this experiment requires tokens.max = 32")
    if type(seqs) is not int or not 1 <= seqs <= MAX_ROWS:
        raise ValueError("this experiment requires 1 <= seqs.max <= 32")
    ops, modules = source["ops"], source["modules"]
    if {n for n in ops if n.endswith("dsa_attn")} != set(OPS):
        raise ValueError("expected exactly the decode, mtp16 and mtp32 attention ops")
    for name, op in ops.items():
        if name in OPS:
            continue
        for launch in op.get("impl", {}).get("launches", []):
            if (modules.get(launch.get("module"), {}).get("sha256") == MODULE_SHA256[0]
                    or "prepare_varlen_num_blocks_kernel" in launch.get("entry", "")):
                raise ValueError("an additional preparation op needs a separate audit")
    for name in OPS:
        op = ops[name]
        launches = op["impl"]["launches"]
        if op["params"] != PARAMS or len(launches) != 3:
            raise ValueError(f"{name}: unsupported attention ABI")
        hashes = tuple(modules[l["module"]]["sha256"] for l in launches)
        if hashes != MODULE_SHA256:
            raise ValueError(f"{name}: unsupported FA3 module hashes")
        for key in ("sem", "nmb", "nsd", "vbi"):
            scratch = op["impl"]["scratch"][key]
            if scratch["dtype"] != "i32" or scratch["shape"] != [MAX_ROWS]:
                raise ValueError(f"{name}: {key} must already have 32 i32 entries")
        if prepare_abi_sha256(launches[0]) != PREPARE_ABI_SHA256:
            raise ValueError(f"{name}: unsupported preparation ABI or policy already applied")


def transform(source):
    """Return a separate candidate; do not alter the source or resize scratch.

    Validation is deliberately narrow and remains active under python -O.
    It does not replace `kern verify` or full-model correctness testing.
    """
    try:
        _validate(source)
    except (KeyError, TypeError, AttributeError, IndexError):
        raise ValueError("unsupported or incomplete FA3 manifest contract") from None
    out = copy.deepcopy(source)
    for name in OPS:
        prep = out["ops"][name]["impl"]["launches"][0]
        # The exact same cubin exports the two-warp specialization. Merely
        # increasing the block for the one-warp entry is not a repair.
        prep["entry"] = prep["entry"].replace("kernelILi1ELb1E", "kernelILi2ELb1E")
        prep["block"] = [64, 1, 1]
        prep["args"][12] = {"i32": SM_HINT}
    return out


def verify_modules(source, manifest_dir):
    """Hash the three actual FA3 cubins before the CLI writes a candidate."""
    names = {launch["module"] for name in OPS
             for launch in source["ops"][name]["impl"]["launches"]}
    for name in sorted(names):
        module = source["modules"][name]
        if not isinstance(module.get("source"), str) or not module["source"]:
            raise ValueError("an FA3 module needs a local cubin source path")
        path = Path(manifest_dir) / module["source"]
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != module["sha256"]:
            raise ValueError("an FA3 cubin does not match its pinned sha256")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="existing audited MTP manifest")
    parser.add_argument("--out", type=Path, required=True,
                        help="new file in the source directory; never overwrites")
    args = parser.parse_args(argv)
    try:
        # Keep all relative module sources unchanged and valid. Relocation
        # would be a second transformation, outside this controlled experiment.
        if args.source.parent.resolve() != args.out.parent.resolve():
            raise ValueError("--out must be in the source manifest directory")
        if args.out.exists() or args.out.is_symlink():
            raise ValueError("destination already exists; use a new experiment filename")
        source = json.loads(args.source.read_text())
        candidate = transform(source)
        verify_modules(source, args.source.parent)
        encoded = json.dumps(candidate, indent=2) + "\n"
        with args.out.open("x") as stream:
            stream.write(encoded)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print("Experimental FA3 candidate written. Native correctness and performance gates remain open.")


if __name__ == "__main__":
    main()
