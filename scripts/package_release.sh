#!/usr/bin/env bash
# Package a built `kern` for a GitHub release and prove it will start on
# the machines the release promises: glibc 2.28, no CUDA library linked
# (the runtime dlopens them), the version the tag says.
#
#   scripts/package_release.sh VERSION TARGET OUT_DIR
#
# Reads target/release/kern (or $KERN_BIN); writes
# OUT_DIR/kern-TARGET.tar.gz and OUT_DIR/kern-TARGET.tar.gz.sha256. Run it
# where the binary was built; the checks run the binary.
set -euo pipefail

[[ $# -eq 3 ]] || { echo "usage: $0 VERSION TARGET OUT_DIR" >&2; exit 2; }
version=${1#v}
target=$2
out=$3
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bin=${KERN_BIN:-$root/target/release/kern}
glibc_max=2.28
asset=kern-$target

fail() { echo "package: $*" >&2; exit 1; }

[[ -x $bin ]] || fail "no binary at $bin"

# The version the binary reports is the version the tag names.
reported=$("$bin" --version | awk '{print $2}')
[[ $reported == "$version" ]] || fail "binary reports $reported, release is $version"

# Nothing CUDA is a link-time dependency; the loader must not need a toolkit.
needed=$(objdump -p "$bin" | awk '$1 == "NEEDED" {print $2}')
if grep -Eq 'libcu|libnv' <<<"$needed"; then
  fail "binary links a CUDA library:"$'\n'"$needed"
fi

# Every glibc symbol version the binary asks for is at most $glibc_max.
asked=$(objdump -T "$bin" | grep -o 'GLIBC_[0-9.]*' | sed 's/GLIBC_//' | sort -Vu | tail -1)
[[ $(printf '%s\n%s\n' "$asked" "$glibc_max" | sort -V | tail -1) == "$glibc_max" ]] ||
  fail "binary needs glibc $asked, release promises $glibc_max"

# A GPU-free command runs end to end: the loader is satisfied here.
"$bin" verify "$root/examples/qwen3-4b.json" >/dev/null

stage=$(mktemp -d)
trap 'rm -rf -- "$stage"' EXIT
install -m 0755 "$bin" "$stage/kern"
install -m 0644 "$root/LICENSE" "$stage/LICENSE"
mkdir -p "$out"
tar -C "$stage" -czf "$out/$asset.tar.gz" kern LICENSE
(cd "$out" && sha256sum "$asset.tar.gz" >"$asset.tar.gz.sha256")
echo "$out/$asset.tar.gz"
