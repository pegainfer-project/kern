#!/bin/sh
# Install the kern binary from a GitHub release.
#
#   curl -fsSL https://kern-baa.pages.dev/install.sh | sh
#
#   KERN_VERSION      release tag to install (default: latest)
#   KERN_INSTALL_DIR  where the binary goes (default: ~/.local/bin)
#   KERN_BASE_URL     where the release assets are (default: GitHub releases)
#
# Downloads kern-<arch>-unknown-linux-gnu.tar.gz and its checksum, verifies,
# installs one file. Nothing else is written. Then it looks at the machine
# and says what `kern run` will need that it cannot see, without failing.
set -eu

repo=pegainfer-project/kern
version=${KERN_VERSION:-latest}
dir=${KERN_INSTALL_DIR:-$HOME/.local/bin}

say() { printf 'kern install: %s\n' "$*"; }
warn() { printf 'kern install: warning: %s\n' "$*" >&2; }
die() { printf 'kern install: %s\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = Linux ] || die "kern runs on Linux only (this is $(uname -s))"
case $(uname -m) in
  x86_64) target=x86_64-unknown-linux-gnu ;;
  aarch64 | arm64) target=aarch64-unknown-linux-gnu ;;
  *) die "no release for $(uname -m); build from source: cargo install --git https://github.com/$repo kern-run" ;;
esac

if [ -n "${KERN_BASE_URL:-}" ]; then
  base=$KERN_BASE_URL
elif [ "$version" = latest ]; then
  base=https://github.com/$repo/releases/latest/download
else
  base=https://github.com/$repo/releases/download/$version
fi

if command -v curl >/dev/null; then
  fetch() { curl -fsSL -o "$2" "$1"; }
elif command -v wget >/dev/null; then
  fetch() { wget -qO "$2" "$1"; }
else
  die "need curl or wget"
fi
if command -v sha256sum >/dev/null; then
  digest() { sha256sum "$1" | cut -d' ' -f1; }
elif command -v shasum >/dev/null; then
  digest() { shasum -a 256 "$1" | cut -d' ' -f1; }
else
  die "need sha256sum or shasum"
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
asset=kern-$target.tar.gz

say "downloading $base/$asset"
fetch "$base/$asset" "$tmp/$asset" || die "download failed; is $version a release with a $target build?"
fetch "$base/SHA256SUMS" "$tmp/SHA256SUMS" || die "download of SHA256SUMS failed"
want=$(awk -v a="$asset" '$2 == a {print $1}' "$tmp/SHA256SUMS")
[ -n "$want" ] || die "SHA256SUMS has no entry for $asset"
have=$(digest "$tmp/$asset")
[ "$have" = "$want" ] || die "checksum mismatch for $asset: expected $want, got $have"

tar -xzf "$tmp/$asset" -C "$tmp"
mkdir -p "$dir"
install -m 0755 "$tmp/kern" "$dir/kern"
say "installed $("$dir/kern" --version) to $dir/kern"

case ":$PATH:" in
  *":$dir:"*) ;;
  *) say "add it to PATH:  export PATH=\"$dir:\$PATH\"" ;;
esac

# What the runtime will dlopen at first use; say now what is missing.
if ! command -v nvidia-smi >/dev/null; then
  warn "no nvidia-smi on PATH; kern needs an NVIDIA driver that supports CUDA 13 (r580 or newer)"
else
  driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
  case ${driver%%.*} in
    '' | *[!0-9]*) warn "could not read the driver version from nvidia-smi" ;;
    *) [ "${driver%%.*}" -ge 580 ] || warn "driver $driver is older than r580; kern binds CUDA 13" ;;
  esac
fi

on_loader_path() {
  (ldconfig -p 2>/dev/null || true) | grep -q "$1" && return 0
  IFS=:
  for d in ${LD_LIBRARY_PATH:-}; do
    [ -n "$d" ] && [ -e "$d/$1" ] && return 0
  done
  return 1
}
if ! on_loader_path libcublasLt.so.13; then
  hint=
  for d in /usr/local/cuda/lib64 /usr/local/cuda/targets/*/lib /usr/local/cuda-13*/lib64; do
    [ -e "$d/libcublasLt.so.13" ] && { hint=$d; break; }
  done
  if [ -n "$hint" ]; then
    warn "cuBLAS 13 is in $hint but not on the loader path:  export LD_LIBRARY_PATH=\"$hint:\${LD_LIBRARY_PATH:-}\""
  else
    warn "no cuBLAS 13 on the loader path; install the CUDA 13 toolkit, or pip install nvidia-cublas-cu13 and point LD_LIBRARY_PATH at its lib/"
  fi
fi
