"""Derive Engram tokenizer/hash constants and compile their once initializer.

Only metadata is generated; model weights bind the original checkpoint directly.
The tokenizer normalization, prime layout and RNG are executed from the supplied
inference/engram.py. Constants become small carry buffers, filled by one op in
load's once program; no checkpoint export or runtime Python launcher is needed.
"""
import argparse
import ast
import copy
import math
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


def derive(model: Path, inference: Path):
    from tokenizers import Tokenizer

    spec = importlib.util.spec_from_file_location("dsv41_official_engram", inference / "engram.py")
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    args = SimpleNamespace(**json.loads((inference / "config.json").read_text()))
    backend = Tokenizer.from_file(str(model / "tokenizer.json"))

    class TokenizerView:
        backend_tokenizer = backend

        def __len__(self):
            return backend.get_vocab_size(with_added_tokens=True)

    token_map, vocab = reference.build_compressed_token_map(TokenizerView())
    if vocab != args.engram_compressed_vocab_size:
        raise ValueError(f"compressed vocabulary mismatch: {vocab} != {args.engram_compressed_vocab_size}")
    layout = reference.EngramLayout.from_args(args)
    primes = [p for layer in layout.primes for gram in layer for p in gram]
    offsets = []
    for layer in layout.primes:
        offset = 0
        for gram in layer:
            for prime in gram:
                offsets.append(offset)
                offset += prime
    multipliers = reference.compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, vocab).flatten().tolist()
    return {"token_map": token_map, "multipliers": multipliers, "primes": primes, "offsets": offsets}, {
        "compressed_vocab_size": vocab, "compressed_pad_id": token_map[args.engram_pad_id],
        "layer_ids": list(layout.layer_ids), "hash_columns": (layout.max_ngram_size - 1) * layout.n_heads,
        "reference_sha256": hashlib.sha256((inference / "engram.py").read_bytes()).hexdigest(),
        "tokenizer_sha256": hashlib.sha256((model / "tokenizer.json").read_bytes()).hexdigest(),
    }


def rope_frequencies(inference, device="cuda"):
    """Execute the original frequency calculation, before its outer product."""
    import torch
    tree = ast.parse((inference / "model.py").read_text())
    original = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "precompute_freqs_cis")
    function = copy.deepcopy(original)
    cutoff = next(i for i,n in enumerate(function.body) if isinstance(n, ast.Assign)
                  and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                  and n.value.func.attr == "outer")
    function.body = function.body[:cutoff] + [ast.Return(ast.Name(id="freqs", ctx=ast.Load()))]
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    ns = {"torch": torch, "math": math, "lru_cache": __import__("functools").lru_cache}
    exec(compile(module, "official_rope_frequencies", "exec"), ns)
    cfg = json.loads((inference / "config.json").read_text())
    results = {}
    with torch.device(device):
        for name, original_length, theta in (("window", 0, cfg["rope_theta"]),
                ("compressed", cfg["original_seq_len"], cfg["compress_rope_theta"])):
            results[name] = ns["precompute_freqs_cis"](cfg["rope_head_dim"], 1, original_length,
                theta, cfg["rope_factor"], cfg["beta_fast"], cfg["beta_slow"]).cpu().tolist()
    return results


def build(model, inference, output, nvcc="nvcc", arch="sm_103a", max_positions=1048576, reference_device="cuda"):

    arrays, metadata = derive(model, inference)
    output.mkdir(parents=True, exist_ok=True)
    declarations = []
    for name, values in arrays.items():
        dtype = "unsigned int" if name == "token_map" else "long long"
        declarations.append(f"__device__ const {dtype} values_{name}[{len(values)}] = {{" +
            ",".join(str(v) + ("LL" if dtype == "long long" else "U") for v in values) + "};")
    frequencies = rope_frequencies(inference, reference_device)
    for name, values in frequencies.items():
        declarations.append(f"__device__ const float rope_{name}[32] = {{" +
            ",".join(float(v).hex() + "f" for v in values) + "};")
    params = ",".join("long long* " + name for name in arrays)
    body = "\n".join(f"if(i<{len(values)}) {name}[i]=values_{name}[i];" for name, values in arrays.items())
    source = output / "engram_constants.cu"
    source.write_text("\n".join(declarations) + f'\nextern "C" __global__ void dsv41_engram_constants({params})' +
        "{int i=blockIdx.x*blockDim.x+threadIdx.x;\n" + body + "\n}\n")
    with source.open("a") as f:
        f.write(r'''
extern "C" __global__ void dsv41_rope_constants(float* wi,float* ws,float* ci,float* cs,int positions){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=positions*32)return;int p=i/32,d=i%32;
 float s,c;sincosf((float)p*rope_window[d],&s,&c);wi[p*64+2*d]=c;wi[p*64+2*d+1]=s;ws[p*64+d]=c;ws[p*64+32+d]=s;
 sincosf((float)p*rope_compressed[d],&s,&c);ci[p*64+2*d]=c;ci[p*64+2*d+1]=s;cs[p*64+d]=c;cs[p*64+32+d]=s;
}
''')
    cubin = output / "dsv41_engram_constants.cubin"
    subprocess.run([nvcc, "-cubin", f"-arch={arch}", "-o", str(cubin), str(source)], check=True)
    names = {name: "engram." + name for name in arrays}
    pieces = {
        "modules": {"dsv41_engram_constants": {"source": cubin.name, "sha256": hashlib.sha256(cubin.read_bytes()).hexdigest()}},
        "buffers": {names[name]: {"kind": "carry", "dtype": "i64", "shape": [len(values)]} for name, values in arrays.items()},
        "ops": {"engram.constants": {"params": ["out buffer<i64>"] * 4, "impl": {"launches": [{
            "module": "dsv41_engram_constants", "entry": "dsv41_engram_constants", "block": [256, 1, 1],
            "grid": [(len(arrays["token_map"]) + 255) // 256, 1, 1]}]}}},
        "calls": [{"label": "engram.constants", "op": "engram.constants", "args": [{"buf": n} for n in names.values()]}],
        "names": names, "metadata": metadata,
    }
    rope_names = {family: {layout: f"rope.{family}.{layout}" for layout in ("interleaved", "split")}
                  for family in ("window", "compressed")}
    for family in rope_names.values():
        for name in family.values():
            pieces["buffers"][name] = {"kind": "carry", "dtype": "f32", "shape": [max_positions, 64]}
    pieces["ops"]["rope.constants"] = {"params": ["out buffer<f32>"] * 4 + ["i32"],
        "impl": {"launches": [{"module": "dsv41_engram_constants", "entry": "dsv41_rope_constants",
            "block": [256, 1, 1], "grid": [(max_positions * 32 + 255) // 256, 1, 1]}]}}
    pieces["calls"].append({"label": "rope.constants", "op": "rope.constants", "args":
        [{"buf": name} for family in rope_names.values() for name in family.values()] + [{"i32": max_positions}]})
    pieces["rope"] = rope_names
    pieces["metadata"].update(max_positions=max_positions, frequency_device=reference_device)
    (output / "rope_frequencies.reference.json").write_text(json.dumps(frequencies) + "\n")
    (output / "engram_constants.json").write_text(json.dumps(pieces, indent=2) + "\n")
    (output / "engram_constants.reference.json").write_text(json.dumps(arrays) + "\n")
    return pieces


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model", type=Path)
    p.add_argument("inference", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--nvcc", default=os.environ.get("NVCC", "nvcc"))
    p.add_argument("--arch", default=os.environ.get("KERN_SM", "sm_103a"))
    p.add_argument("--max-positions", type=int, default=1048576)
    p.add_argument("--reference-device", choices=("cpu", "cuda"), default="cuda")
    a = p.parse_args()
    result = build(a.model, a.inference, a.output, a.nvcc, a.arch, a.max_positions, a.reference_device)
    print(json.dumps(result["metadata"]))
