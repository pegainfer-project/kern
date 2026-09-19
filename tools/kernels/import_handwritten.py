#!/usr/bin/env python3
"""Build handwritten kernels (tools/kernels-src/*.cu) and put them in the
registry: nvcc once, the cubin into the cache, the sha into the index. A
generator then pins the index's sha and never runs nvcc.

    tools/kernels/import_handwritten.py k3_residual k3_residual+LAND_BF16=1 [--upload]
    tools/kernels/import_handwritten.py --all [--upload]        # every variant the index lists, plus every .cu it does not

Run it where the toolchain is the project's (the kernel-lab container's
nvcc; docs/lessons.md): a different nvcc is a different sha, and every
manifest pinning the old one must be regenerated — the tool prints each
sha that moved. `KERN_SM` picks the target (sm_103a).
"""
import argparse
import os
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import import_cubin  # noqa: E402
import index  # noqa: E402
import store  # noqa: E402

SRC = index.REPO / "tools" / "kernels-src"
UPSTREAM = "tools/kernels-src/"


def toolchain():
    out = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=True).stdout
    build = next(line.strip() for line in out.splitlines() if line.startswith("Build "))
    return f"nvcc {build}, -arch={arch()}"


def arch():
    return os.environ.get("KERN_SM", "sm_103a")


def parse(spec):
    """`name+K=V+K2=V2` -> (name, {K: V})."""
    name, *defs = spec.split("+")
    return name, {k: import_cubin.scalar(v) for k, v in (d.split("=", 1) for d in defs)}


def build(name, defines, out):
    src = SRC / f"{name}.cu"
    if not src.exists():
        sys.exit(f"no {src.relative_to(index.REPO)}")
    flags = [f"-D{k}={v}" for k, v in sorted(defines.items())]
    subprocess.run(["nvcc", "-cubin", f"-arch={arch()}", *flags, "-o", str(out), str(src)], check=True)


def handwritten_variants():
    """Every (name, defines) the index records for a handwritten family, and the plain build of every .cu without one."""
    specs = []
    for fam in index.families():
        doc = index.load(fam)
        if not doc["family"].get("upstream", "").startswith(UPSTREAM):
            continue
        specs += [parse(v["name"]) for v in doc.get("variant", [])]
    have = {n for n, _ in specs}
    specs += [(p.stem, {}) for p in sorted(SRC.glob("*.cu")) if p.stem not in have]
    return specs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("variants", nargs="*", help="name[+K=V...]")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args()
    specs = handwritten_variants() if a.all else [parse(s) for s in a.variants]
    if not specs:
        ap.error("name a variant or pass --all")
    tc = toolchain()
    shas, moved = [], []
    with tempfile.TemporaryDirectory() as d:
        for name, defines in specs:
            out = pathlib.Path(d) / f"{index.variant_name(name, defines)}.cubin"
            build(name, defines, out)
            args = import_cubin.argparse.Namespace(kind="cubin", sm=arch(), upstream=f"{UPSTREAM}{name}.cu",
                                                   license="kern (LICENSE)", toolchain=tc, license_file=None)
            sha, old = import_cubin.import_cubin(name, out, defines=defines, args=args)
            shas.append(sha)
            if old not in (None, sha):
                moved.append((index.variant_name(name, defines), old, sha))
    for name, old, new in moved:
        print(f"MOVED {name}: {old[:12]} -> {new[:12]}; regenerate the manifests that pin it", file=sys.stderr)
    if a.upload:
        store.upload(shas, f"handwritten: {len(shas)} builds, {tc}")


if __name__ == "__main__":
    main()
