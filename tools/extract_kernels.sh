#!/usr/bin/env bash
# Put every module a manifest pins into one kernel directory, by content.
#
#   tools/extract_kernels.sh <manifest.json> <dump_dir>[:<dump_dir>...] [out_dir=kernels]
#
# Every module the manifest's `modules` table names is pinned by sha256 (the
# verifier insists). This script finds each local-source sha among the capture
# dumps, the registry cache (`$KERN_CACHE_DIR` or ~/.cache/kern, where
# tools/kernels/import_*.py land every build) and the repo's kernels/, and
# lands it as `<module name>-<sha12>.cubin` — readable in `ls`, unique per
# version. Registry refs (`hf:` / `https://`) are the runtime's to fetch. The runtime resolves by hash, never by name, so the
# directory only ever grows: extract A, extract B, and `kern test` loads both
# from it — the runtime loads only what the manifest pins, so the rest of
# the directory is inert. Dump dirs are searched recursively (a capture
# root holding `pid*/module_*.cubin`).
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="${1:?manifest.json}"
IFS=: read -r -a dumps <<<"${2:?dump dir(s)}"
out="${3:-$repo/kernels}"
cache="${KERN_CACHE_DIR:-$HOME/.cache/kern}/blobs"
mkdir -p "$out"

# (module source, sha, one entry) per local module the manifest pins
mapfile -t wanted < <(python3 - "$manifest" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
entry_of = {}
for op in m["ops"].values():
    for l in op["impl"]["launches"]:
        if "module" in l:
            entry_of.setdefault(l["module"], l["entry"])
for name, md in m["modules"].items():
    if not (md["source"].startswith("hf:") or md["source"].startswith("https://")):
        print(md["source"], md["sha256"], entry_of.get(name, "-"))
PY
)

# content index over every candidate cubin
declare -A by_sha
while read -r sha path; do
  by_sha[$sha]="${by_sha[$sha]:-$path}"
done < <(for d in "${dumps[@]}" "$repo/kernels" "$out"; do
           find "$d" -name '*.cubin' -print0 2>/dev/null | xargs -0 -r sha256sum; done
         find "$cache" -maxdepth 1 -type f -print0 2>/dev/null | xargs -0 -r sha256sum)

# what the out dir already holds, by content (the same bytes never land twice)
declare -A landed
while read -r sha path; do landed[$sha]=$path; done < <(ls "$out"/*.cubin >/dev/null 2>&1 && sha256sum "$out"/*.cubin)

land() {  # land <src> <display name> <sha>
  local src="$1" name="$2" sha="$3"
  [ -n "${landed[$sha]:-}" ] && return
  local dst="$out/${name%.cubin}-${sha:0:12}.cubin"
  cp "$src" "$dst"; landed[$sha]=$dst; echo "  + $(basename "$dst")   ($sym)" >&2
}

missing=0
for entry in "${wanted[@]}"; do
  read -r cubin sha sym <<<"$entry"
  src="${by_sha[$sha]:-}"
  if [ -z "$src" ]; then
    echo "MISSING $cubin @${sha:0:12} ($sym): no file with that sha256 in ${dumps[*]}, $cache, $repo/kernels" >&2
    missing=$((missing+1)); continue
  fi
  land "$src" "$cubin" "$sha"
done
[ "$missing" -eq 0 ] || { echo "$missing pinned cubin(s) missing" >&2; exit 1; }
echo "$out: $(ls "$out"/*.cubin | wc -l) cubins" >&2
