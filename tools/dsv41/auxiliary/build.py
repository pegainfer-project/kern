"""Build and pin the DSV4.1 auxiliary module (no GPU required)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def build(output: Path, nvcc: str = "nvcc", arch: str = "sm_103a") -> dict:
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / "dsv41_auxiliary.cubin"
    subprocess.run([nvcc, "-cubin", f"-arch={arch}", "--fmad=false",
                    "-o", str(artifact), str(Path(__file__).with_name("auxiliary.cu"))], check=True)
    return {"source": artifact.name, "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    p.add_argument("--nvcc", default=os.environ.get("NVCC", "nvcc"))
    p.add_argument("--arch", default=os.environ.get("KERN_SM", "sm_103a"))
    args = p.parse_args()
    print(json.dumps(build(args.output, args.nvcc, args.arch)))
