#!/usr/bin/env python3
"""Check the kernel index (docs/registry.md), offline, in CI:

- every tools/kernels/index/*.toml parses, its file name is its family's;
- a sha256 is 64 hex chars and names one variant across all families;
- a pick names a variant that exists, has launch geometry, and carries a report;
- the document is normalized (regenerating it from its own facts is a no-op);
- every `hf:Pegainfer/kern-kernels` module an example manifest pins is in the index.

`--online` also asks the blob store for every sha (HEAD), which a release
does before it ships and an import does after it uploads.

    python3 tools/kernels/check.py [--online]
"""
import argparse
import json
import pathlib
import re
import sys
import tomllib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import index  # noqa: E402

HEX64 = re.compile(r"^[0-9a-f]{64}$")


def check(online=False):
    errs, seen = [], {}
    for fam in index.families():
        p = index.path(fam)
        try:
            doc = tomllib.loads(p.read_text())
        except tomllib.TOMLDecodeError as e:
            errs.append(f"{p.name}: {e}")
            continue
        ctx = p.name
        f = doc.get("family", {})
        if f.get("name") != fam:
            errs.append(f"{ctx}: family name `{f.get('name')}` is not the file's")
        for k in ("kind", "sm", "upstream", "license"):
            if not f.get(k):
                errs.append(f"{ctx}: family has no `{k}`")
        if f.get("license_blob") and not HEX64.match(f["license_blob"]):
            errs.append(f"{ctx}: license_blob is not a sha256")
        names = {}
        for v in doc.get("variant", []):
            n, sha = v.get("name"), v.get("sha256", "")
            if n in names:
                errs.append(f"{ctx}: variant `{n}` twice")
            names[n] = v
            if not HEX64.match(sha):
                errs.append(f"{ctx}: variant `{n}`: sha256 `{sha}` is not 64 hex chars")
            elif sha in seen and not seen[sha].startswith(f"{fam}/"):
                # two builds of one family may be the same bytes (a define the source ignores); two families may not
                errs.append(f"{ctx}: variant `{n}`: sha {sha[:12]} is already {seen[sha]}")
            seen.setdefault(sha, f"{fam}/{n}")
            if not v.get("entries"):
                errs.append(f"{ctx}: variant `{n}` lists no entries")
        for pk in doc.get("pick", []):
            who = f"{ctx}: pick ({pk.get('op')}, {pk.get('shape')})"
            v = names.get(pk.get("variant"))
            if v is None:
                errs.append(f"{who}: variant `{pk.get('variant')}` does not exist")
            elif not v.get("launch"):
                errs.append(f"{who}: variant `{v['name']}` has no launch geometry")
            if not HEX64.match(pk.get("report", "")):
                errs.append(f"{who}: no bench report (a pick is a measurement)")
        if index.dumps(doc) != p.read_text():
            errs.append(f"{ctx}: not normalized; rewrite it with tools/kernels/index.py save()")
    prefix = f"hf:{index.BLOB_REPO}/blobs/"
    for ex in sorted((index.REPO / "examples").glob("*.json")):
        m = json.loads(ex.read_text())
        for name, md in m.get("modules", {}).items():
            if md["source"].startswith(prefix) and md["sha256"] not in seen:
                errs.append(f"{ex.name}: module `{name}` pins {md['sha256'][:12]} from the blob store, not in the index")
    if online:
        import store
        for sha, who in sorted(seen.items(), key=lambda kv: kv[1]):
            if not store.exists_remote(sha):
                errs.append(f"{who}: {sha[:12]} is not in the blob store")
    return errs, len(seen)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--online", action="store_true")
    a = ap.parse_args()
    errs, n = check(a.online)
    for e in errs:
        print(e, file=sys.stderr)
    print(f"{len(index.families())} families, {n} variants, {len(errs)} problems", file=sys.stderr)
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
