"""Run the complete five-step Markov head through kern, including CUDA graphs."""
import argparse
import ast
from types import SimpleNamespace
import json
from pathlib import Path
import subprocess
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kern_manifest import SCHEMA_VERSION
from dsv41.head import definitions, markov_calls


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--runner", type=Path, required=True)
    p.add_argument("--cubin", type=Path, required=True)
    p.add_argument("--gpu", default="3")
    p.add_argument("--inference", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    tree = ast.parse((a.inference / "model.py").read_text())
    draft = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DSparkBlock")
    forward = next(n for n in draft.body if isinstance(n, ast.FunctionDef) and n.name == "forward_head")
    sample = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "sample")
    reference = {"torch": torch}
    exec(compile(ast.Module(body=[forward, sample], type_ignores=[]), "inference_head", "exec"), reference)
    torch.manual_seed(102)
    torch.backends.cuda.matmul.allow_tf32 = False
    a.out.mkdir(parents=True, exist_ok=True)
    vocab, hidden, rank = 2048, 128, 256
    for batch in (1, 17):
        case = a.out / str(batch)
        case.mkdir(exist_ok=True)
        tensors = {
            "draft.head_hidden": torch.randn(batch*5, hidden, dtype=torch.bfloat16),
            "head.weight": torch.randn(vocab, hidden, dtype=torch.bfloat16),
            "mtp.2.markov_head.embed.weight": torch.randn(vocab, rank, dtype=torch.bfloat16),
            "mtp.2.markov_head.head.weight": torch.randn(vocab, rank, dtype=torch.bfloat16),
            "anchor_token": torch.randint(vocab, (batch,), dtype=torch.int64),
        }
        buffers = {name: {"kind": "input", "dtype": "i64" if t.dtype == torch.int64 else "bf16",
                          "shape": list(t.shape)} for name, t in tensors.items()}
        for name, shape, dtype, kind in (
            ("draft.logits", [batch*5,vocab], "f32", "workspace"),
            ("draft.markov_embed", [batch,rank], "bf16", "workspace"),
            ("draft.markov_bias", [batch,vocab], "f32", "workspace"),
            ("draft_tokens", [batch,5], "i64", "output"),
        ):
            buffers[name] = {"kind": kind, "dtype": dtype, "shape": shape}
        modules, ops = definitions(a.cubin)
        calls = markov_calls(vocab=vocab, hidden=hidden, rank=rank)
        manifest = {
            "schema_version": SCHEMA_VERSION, "model": "dsv41-markov-chain",
            "vars": {"seqs": {"max": batch}}, "buffers": buffers,
            "modules": modules, "ops": ops, "programs": {"head": {"calls": calls}},
        }
        path = case / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2))
        cmd = [str(a.runner), "--manifest", str(path), "--cubins", str(a.cubin.parent),
               "--gpu", a.gpu, "--vars", f"seqs={batch}", "--graph", "--iters", "3",
               "--out", f"draft_tokens={case/'actual.bin'}"]
        for name, t in tensors.items():
            f = case / (name + ".bin")
            f.write_bytes(t.view(torch.uint8).numpy().tobytes())
            cmd += ["--in", f"{name}={f}"]
        subprocess.run(cmd, check=True)
        # Execute the model's unmodified forward_head; only its linear modules
        # are supplied with the synthetic weights used by the actual manifest.
        def markov(ids):
            embedding = tensors["mtp.2.markov_head.embed.weight"][ids]
            logits = embedding.float() @ tensors["mtp.2.markov_head.head.weight"].float().T
            return logits, embedding
        oracle = SimpleNamespace(
            head=lambda x, full_logits: x.float() @ tensors["head.weight"].float().T,
            norm=lambda x: x,
            hc_pre=lambda x, pre: x,
            markov_head=markov, confidence_head=lambda x, e: torch.zeros(x.shape[:2]),
            block_size=5, temperature=0,
        )
        expected, _, _ = reference["forward_head"](
            oracle, tensors["draft.head_hidden"].view(batch,5,hidden),
            None, tensors["anchor_token"])
        expected = expected[:, 1:]
        actual = torch.frombuffer(bytearray((case/"actual.bin").read_bytes()), dtype=torch.int64).reshape(batch,5)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        print(f"batch={batch}: complete five-step Markov kern graph matches sequential FP32 reference", flush=True)


if __name__ == "__main__":
    main()
