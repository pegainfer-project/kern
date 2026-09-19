#!/usr/bin/env python3
"""Put one artifact into the registry: the bytes into the cache (and, with
--upload, the blob store), the facts into $KERN_INDEX_DIR/<family>.toml.

    tools/kernels/import_cubin.py --family trtllm_fmha_ctx_h192_v128 --cubin path/to.cubin \\
        --upstream "flashinfer-cubin 0.6.18, bundle 2d6a5a02…, fmha/trtllm-gen/…" --license Apache-2.0 \\
        --license-file LICENSE.txt --sm sm103a [--name <variant>] [--define K=V ...] \\
        [--launch block=512,shared_mem=199296,cluster=1,1,1] [--tag k=v ...] [--abi …] [--toolchain …] [--upload]

The variant's name defaults to the family's (plus `+K=V` per define); its
entries are read from the cubin with cuobjdump. Family fields given on the
command line replace the recorded ones; the rest stay. The import date is
today's.
"""
import argparse
import datetime
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import index  # noqa: E402
import store  # noqa: E402


def entries(cubin):
    """Kernel entry names as the manifest spells them: every function the cubin defines."""
    out = subprocess.run(["cuobjdump", "-res-usage", str(cubin)], capture_output=True, text=True, check=True).stdout
    return sorted(line.strip().split()[1].rstrip(":") for line in out.splitlines() if line.strip().startswith("Function "))


def scalar(s):
    try:
        return int(s)
    except ValueError:
        return s


def parse_launch(s):
    if not s:
        return None
    launch = {}
    for kv in s.split(","):
        k, v = kv.split("=", 1)
        launch.setdefault(k, []).append(int(v))
    return {k: (v[0] if len(v) == 1 else v) for k, v in launch.items()}


def family_doc(family, args):
    try:
        doc = index.load(family)
    except KeyError:
        doc = {"family": {"name": family, "kind": "cubin"}}
    fam = doc["family"]
    for field in ("kind", "sm", "abi", "upstream", "license", "toolchain", "rebuild", "abi_source", "abi_capture"):
        v = getattr(args, field, None)
        if v is not None:
            fam[field] = v
    if args.license_file:
        fam["license_blob"] = store.put(args.license_file)
    fam["imported"] = datetime.date.today().isoformat()
    return doc


def import_cubin(family, cubin, name=None, defines=None, launch=None, tags=None, args=None):
    defines = dict(defines or {})
    doc = family_doc(family, args or argparse.Namespace())
    sha = store.put(cubin)
    v = {"name": name or index.variant_name(family, defines), "sha256": sha, "entries": entries(cubin)}
    if defines:
        v["defines"] = defines
    if launch:
        v["launch"] = launch
    if tags:
        v["tags"] = tags
    old = next((x["sha256"] for x in doc.get("variant", []) if x["name"] == v["name"]), None)
    index.upsert_variant(doc, v)
    p = index.save(doc)
    state = "unchanged" if old == sha else ("new" if old is None else f"was {old[:12]}")
    print(f"{p.name}: {v['name']} @{sha[:12]} ({state})", file=sys.stderr)
    return sha, old


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True)
    ap.add_argument("--cubin", required=True, type=pathlib.Path)
    ap.add_argument("--name")
    ap.add_argument("--define", action="append", default=[], metavar="K=V")
    ap.add_argument("--launch", help="block=..,shared_mem=..,cluster=x,y,z")
    ap.add_argument("--tag", action="append", default=[], metavar="k=v")
    for f in ("kind", "sm", "abi", "upstream", "license", "toolchain", "rebuild", "abi_source", "abi_capture"):
        ap.add_argument(f"--{f.replace('_', '-')}", dest=f)
    ap.add_argument("--license-file", type=pathlib.Path)
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args()
    defines = {k: scalar(v) for k, v in (d.split("=", 1) for d in a.define)}
    tags = {k: scalar(v) for k, v in (t.split("=", 1) for t in a.tag)}
    sha, _ = import_cubin(a.family, a.cubin, a.name, defines, parse_launch(a.launch), tags, a)
    if a.upload:
        store.upload([sha], f"{a.family}: {a.name or index.variant_name(a.family, defines)} @{sha[:12]}")


if __name__ == "__main__":
    main()
