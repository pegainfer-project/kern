"""Replay all dense scale transforms from original checkpoint bindings."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kern_manifest import SCHEMA_VERSION
from dsv41.loading import Pieces, dense_scales


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bindings", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--runner", type=Path, required=True)
    p.add_argument("--cubins", type=Path, required=True)
    p.add_argument("--gpu", default="3")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    raw = json.loads(a.bindings.read_text())["gpu"]
    pieces = Pieces()
    buffers, calls, layouts = dense_scales(raw, pieces, cubin_dir=a.cubins)
    needed = {arg["buf"] for c in calls for arg in c["args"] if "buf" in arg} - buffers.keys()
    buffers.update({n: raw[n] for n in needed})
    manifest = {
        "schema_version": SCHEMA_VERSION, "model": "dsv41-dense-checkpoint-load",
        "buffers": buffers, "modules": pieces.modules, "ops": pieces.ops,
        "programs": {"load": {"once": True, "calls": calls}},
    }
    path = a.out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    cmd = [str(a.runner), "--manifest", str(path), "--cubins", str(a.cubins),
           "--weights", str(a.weights), "--gpu", a.gpu]
    for layout in layouts.values():
        name = layout["scale"]
        cmd += ["--dump", f"{name}={a.out/(name+'.bin')}"]
    subprocess.run(cmd, check=True)
    from safetensors import safe_open
    import numpy as np
    index = json.loads((a.weights/"model.safetensors.index.json").read_text())["weight_map"]
    for prefix, layout in layouts.items():
        name = prefix + ".scale"
        with safe_open(a.weights/index[name], framework="pt", device="cpu") as st:
            src = st.get_tensor(name).view(__import__("torch").uint8).numpy()
        g, n, k = (layout[key] for key in ("groups", "n", "k"))
        # Original [g*N/32,K/32] blocks broadcast over 32 output rows,
        # grouped by four K scales per word and laid out [g,K/128,N].
        expanded = np.repeat(src, 32, axis=0).reshape(g,n,k//128,4)
        expected = expanded.transpose(0,2,1,3).copy().tobytes()
        actual = (a.out/(layout["scale"]+".bin")).read_bytes()
        assert actual == expected, prefix
    print(f"{len(layouts)} real checkpoint dense scale bindings and once transforms byte-exact", flush=True)


if __name__ == "__main__":
    main()
