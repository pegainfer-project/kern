"""Real checkpoint binding -> scale pack -> activation quant -> dense projection."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kern_manifest import SCHEMA_VERSION
from dsv41.loading import Pieces, dense_scales
from dsv41.forward import projection


def main():
    p = argparse.ArgumentParser()
    for name in ("bindings", "weights", "inference", "out", "runner", "cubins"):
        p.add_argument("--"+name, type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(43)
    sys.path.insert(0, str(a.inference))
    import model
    torch.set_default_dtype(torch.bfloat16)
    raw = json.loads(a.bindings.read_text())["gpu"]
    index = json.loads((a.weights/"model.safetensors.index.json").read_text())["weight_map"]
    for prefix in ("layers.0.attn.wq_a", "layers.0.attn.wkv"):
        for rows in (1, 5, 65):
            case = a.out / (prefix+"."+str(rows))
            case.mkdir(parents=True, exist_ok=True)
            sources = {n: raw[n] for n in (prefix+".weight", prefix+".scale")}
            pieces = Pieces()
            packed, load, layouts = dense_scales(sources, pieces, cubin_dir=a.cubins)
            layout = layouts[prefix]
            lowered = projection(pieces, "project", layout, "x", "y", rows=rows,
                                 workspace="quant", cubin_dir=a.cubins)
            buffers = dict(sources, **packed, **lowered.buffers)
            buffers["x"] = {"dtype":"bf16", "shape":[rows,layout["k"]], "kind":"input"}
            buffers["y"]["kind"] = "output"
            manifest = {"schema_version":SCHEMA_VERSION, "model":"dsv41-projection",
                        "buffers":buffers, "modules":pieces.modules, "ops":pieces.ops,
                        "programs":{"probe":{"calls":load+lowered.calls}}}
            path = case/"manifest.json"
            path.write_text(json.dumps(manifest, indent=2))
            x = torch.randn(rows,layout["k"],dtype=torch.bfloat16)
            (case/"x.bin").write_bytes(x.view(torch.uint8).numpy().tobytes())
            subprocess.run([str(a.runner),"--manifest",str(path),"--cubins",str(a.cubins),
                            "--gpu","3","--weights",str(a.weights),"--in",f"x={case/'x.bin'}",
                            "--out",f"y={case/'y.bin'}","--graph","--iters","3"],check=True)
            with safe_open(a.weights/index[prefix+".weight"],framework="pt",device="cpu") as f:
                w=f.get_tensor(prefix+".weight").cuda(3)
            with safe_open(a.weights/index[prefix+".scale"],framework="pt",device="cpu") as f:
                w.scale=f.get_tensor(prefix+".scale").cuda(3)
            with torch.cuda.device(3):
                expected=model.linear(x.cuda(3),w).float().cpu()
            actual=torch.frombuffer(bytearray((case/"y.bin").read_bytes()),dtype=torch.bfloat16).reshape(expected.shape).float()
            error=((actual-expected).square().sum()/expected.square().sum()).item()
            assert error<1e-5,(prefix,rows,error)
            print(prefix,rows,"real checkpoint projection relative squared error",error,flush=True)


if __name__=="__main__":
    main()
