"""Build and pin the DSV4.1 auxiliary modules (no GPU required).

`dsv41_auxiliary` is every serving-side kernel; `dsv41_engram_peers` is the
one lookup over Engram tables sharded into HBM across the EP group, its own
module so a manifest that keeps the tables in host memory pins nothing new.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


SOURCES = {"dsv41_auxiliary": "auxiliary.cu", "dsv41_engram_peers": "engram_peers.cu"}


def build(output: Path, nvcc: str = "nvcc", arch: str = "sm_103a") -> dict:
    """Compile every module into `output`; the pins, by module name."""
    output.mkdir(parents=True, exist_ok=True)
    pins = {}
    for module, source in SOURCES.items():
        artifact = output / f"{module}.cubin"
        subprocess.run([nvcc, "-cubin", f"-arch={arch}", "--fmad=false",
                        "-o", str(artifact), str(Path(__file__).with_name(source))], check=True)
        pins[module] = {"source": artifact.name, "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}
    return pins


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    p.add_argument("--nvcc", default=os.environ.get("NVCC", "nvcc"))
    p.add_argument("--arch", default=os.environ.get("KERN_SM", "sm_103a"))
    args = p.parse_args()
    print(json.dumps(build(args.output, args.nvcc, args.arch)))
