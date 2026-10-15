#!/usr/bin/env bash
# Prove a built kern-llm wheel installs into a clean Python without a GPU:
# the version the tag says, both engine plugins and their entry points, no
# CUDA library linked (the runtime dlopens them).
#
#   scripts/check_wheel.sh VERSION WHEEL
set -euo pipefail

[[ $# -eq 2 ]] || { echo "usage: $0 VERSION WHEEL" >&2; exit 2; }
version=${1#v}
wheel=$(realpath "$2")

fail() { echo "wheel: $*" >&2; exit 1; }

venv=$(mktemp -d)
trap 'rm -rf -- "$venv"' EXIT
python3 -m venv "$venv"
"$venv/bin/pip" install --quiet "$wheel"

cd "$venv"
reported=$("$venv/bin/python" -c 'import kern; print(kern.__version__)')
[[ $reported == "$version" ]] || fail "kern reports $reported, release is $version"

points=$("$venv/bin/python" -c '
import importlib.metadata as m
print(" ".join(sorted(f"{e.group}={e.value}" for e in m.distribution("kern-llm").entry_points)))')
[[ $points == "sglang.srt.plugins=kern_sglang:register vllm.general_plugins=kern_vllm:register" ]] ||
  fail "entry points are: $points"

for plugin in kern_vllm kern_sglang; do
  "$venv/bin/python" -c "import importlib.util, sys; sys.exit(importlib.util.find_spec('$plugin') is None)" ||
    fail "the wheel has no $plugin"
done

ext=$(find "$venv" -name '_kern*.so')
needed=$(objdump -p "$ext" | awk '$1 == "NEEDED" {print $2}')
if grep -Eq 'libcu|libnv' <<<"$needed"; then
  fail "$ext links a CUDA library:"$'\n'"$needed"
fi
echo "$wheel"
