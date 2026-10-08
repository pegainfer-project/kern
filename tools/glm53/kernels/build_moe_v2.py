#!/usr/bin/env python3
"""Build only new MoE artifacts: nvcc build_kernels.sh flow + offline Triton."""
import hashlib
import json
import pathlib
import subprocess
import re
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "kernels-glm53-handwritten"
import shutil
import triton
from triton.compiler.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from moe_v2_kernels import glm53_moe_v2_w13, glm53_moe_v2_w2

def _cuda_bin(tool):
    for d in ("/usr/local/cuda-13.0/bin", "/usr/local/cuda/bin", "/usr/local/cuda-13.3/bin"):
        p = pathlib.Path(d) / tool
        if p.is_file():
            return str(p)
    found = shutil.which(tool)
    assert found, f"{tool}: no CUDA toolchain found"
    return found

NVCC = _cuda_bin("nvcc")
CUOBJDUMP = _cuda_bin("cuobjdump")
for _cc in ("/usr/bin/g++-14", "/usr/bin/g++-13", "/usr/bin/g++"):
    if pathlib.Path(_cc).is_file():
        GXX = _cc
        break

def main():
    OUT.mkdir(exist_ok=True)
    subprocess.run([NVCC, "-ccbin", GXX,
                    "-cubin", "-arch=sm_90a", "-std=c++17", "-O3",
                    "-o", str(OUT / "glm53_moe_v2_route.cubin"), str(HERE / "glm53_moe_v2_route.cu")], check=True)
    metadata = {}
    specs = [
        (glm53_moe_v2_w13, ["*fp8e4nv", "*fp8e4nv", "*fp32", "*fp32", "*i32", "*i32", "*i32", "*i32", "*bf16", "*fp8e4nv", "*fp32", "i32"]),
        (glm53_moe_v2_w2, ["*fp8e4nv", "*fp8e4nv", "*fp32", "*fp32", "*fp32", "*i32", "*i32", "*i32", "*i32", "*bf16", "*i32", "*bf16", "*i32", "i32"]),
    ]
    for fn, types in specs:
        sig = dict(zip(fn.arg_names, types))
        attrs = {(i,): [["tt.divisibility", 16]] for i, ty in enumerate(types) if ty.startswith("*")}
        cc = triton.compile(ASTSource(fn, sig, constexprs={}, attrs=attrs),
                            target=GPUTarget("cuda", 90, 32),
                            options={"num_warps": 4, "num_stages": 3, "enable_fp_fusion": True})
        name = fn.__name__
        (OUT / (name + ".cubin")).write_bytes(cc.asm["cubin"])
        for ext in ("ttir", "ttgir", "ptx"):
            (HERE / (name + "." + ext)).write_text(cc.asm[ext])
        metadata[name] = {"entry": cc.name, "shared_mem": cc.metadata.shared,
                          "block": [128, 1, 1], "source_signature": sig}
    metadata["glm53_moe_v2_route"] = {"entry": "glm53_moe_v2_route", "shared_mem": 0, "block": [256, 1, 1]}
    for name, md in metadata.items():
        path = OUT / (name + ".cubin")
        md["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        elf = subprocess.check_output([CUOBJDUMP, "-elf", str(path)], text=True)
        (HERE / (name + ".elf.txt")).write_text(elf)
        fields = re.findall(r"Ordinal : (0x[\da-f]+)\s+Offset\s+: (0x[\da-f]+)\s+Size\s+: (0x[\da-f]+)", elf)
        fields = sorted((int(i, 16), int(o, 16), int(z, 16)) for i, o, z in fields)
        expected = {"glm53_moe_v2_route": [8]*15+[4], "glm53_moe_v2_w13": [8]*11+[4,8,8], "glm53_moe_v2_w2": [8]*13+[4,8,8]}[name]
        assert [z for _, _, z in fields] == expected, (name, fields)
        md["elf_params"] = [{"ordinal": i, "offset": o, "size": z} for i, o, z in fields]
        sass = subprocess.check_output([CUOBJDUMP, "-sass", str(path)], text=True)
        (HERE / (name + ".sass.txt")).write_text(sass)
        if name != "glm53_moe_v2_w2":
            assert "ACQBULK" in sass, name
        md["resource_usage"] = subprocess.check_output([CUOBJDUMP, "-res-usage", str(path)], text=True)
        print(name, md["sha256"], md["shared_mem"])
    (HERE / "moe_v2_build.json").write_text(json.dumps(metadata, indent=2) + "\n")
if __name__ == "__main__":
    main()
