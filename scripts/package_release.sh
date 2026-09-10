#!/usr/bin/env bash
# Package built `kern` and `kern-serve` for a GitHub release and prove they
# start without a CUDA toolkit: no CUDA library linked (the runtime dlopens
# them), the version the tag says.
#
#   scripts/package_release.sh VERSION TARGET OUT_DIR
#
# Reads target/release/kern (or $KERN_BIN) and
# crates/kern-serve/target/release/kern-serve (or $KERN_SERVE_BIN); writes
# OUT_DIR/kern-TARGET.tar.gz and OUT_DIR/kern-TARGET.tar.gz.sha256. Run it
# where the binaries were built; the checks run them.
set -euo pipefail

[[ $# -eq 3 ]] || { echo "usage: $0 VERSION TARGET OUT_DIR" >&2; exit 2; }
version=${1#v}
target=$2
out=$3
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bin=${KERN_BIN:-$root/target/release/kern}
serve=${KERN_SERVE_BIN:-$root/crates/kern-serve/target/release/kern-serve}
asset=kern-$target

fail() { echo "package: $*" >&2; exit 1; }

# The version a binary reports is the version the tag names, and nothing
# CUDA is a link-time dependency: the loader must not need a toolkit.
check() {
  local exe=$1 reported needed
  [[ -x $exe ]] || fail "no binary at $exe"
  reported=$("$exe" --version | awk '{print $2}')
  [[ $reported == "$version" ]] || fail "$exe reports $reported, release is $version"
  needed=$(objdump -p "$exe" | awk '$1 == "NEEDED" {print $2}')
  if grep -Eq 'libcu|libnv' <<<"$needed"; then
    fail "$exe links a CUDA library:"$'\n'"$needed"
  fi
}
check "$bin"
check "$serve"

# A GPU-free command runs end to end: the loader is satisfied here.
"$bin" verify "$root/examples/qwen3-4b.json" >/dev/null
"$serve" --help >/dev/null

stage=$(mktemp -d)
trap 'rm -rf -- "$stage"' EXIT
install -m 0755 "$bin" "$stage/kern"
install -m 0755 "$serve" "$stage/kern-serve"
install -m 0644 "$root/LICENSE" "$stage/LICENSE"
mkdir -p "$out"
tar -C "$stage" -czf "$out/$asset.tar.gz" kern kern-serve LICENSE
(cd "$out" && sha256sum "$asset.tar.gz" >"$asset.tar.gz.sha256")
echo "$out/$asset.tar.gz"
