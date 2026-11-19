"""Minimal model-agnostic CLI runner over `manifest` and `executor`.

Drives prefill and decode or speculative rounds solely through `Protocol` —
zero model names or buffer names appear here.

    python3 examples/python/run.py --config /path/to/kern.toml --prompt "The capital of France is"
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

import torch
from tokenizers import Tokenizer

from executor import Executor
from manifest import Sampling


def generate(
    rt: Executor,
    prompt_ids: list[int],
    steps: int = 32,
    rows: int | None = None,
    sampling: Sampling = Sampling(),
    use_graph: bool = True,
) -> list[int]:
    """Run chunked prefill followed by fixed-row decode or speculative rounds."""
    p = rt.protocol
    chunk_fwd = p.chunk_forward()
    step_fwd = p.step_forward(rows=rows, groups=1)
    step_rows = int(step_fwd.rows)

    # 1. Chunked prefill
    pos = 0
    generated: list[int] = []
    while pos < len(prompt_ids):
        chunk = prompt_ids[pos : pos + p.rows_max]
        slots = list(range(pos, pos + len(chunk)))
        vals = rt.stage(chunk_fwd, [pos], [len(chunk)], chunk, slots, sampling)
        rt.run(chunk_fwd.name, vals, graph=use_graph)
        torch.cuda.synchronize()
        pos += len(chunk)
        if pos == len(prompt_ids):
            generated.extend(rt.emitted(chunk_fwd))

    # 2. Decode or speculative rounds (CUDA Graph)
    while len(generated) < steps:
        anchor = generated[-1] if generated else prompt_ids[-1]
        ids = [anchor] + [0] * (step_rows - 1)
        slots = list(range(pos, pos + step_rows))
        vals = rt.stage(step_fwd, [pos], [step_rows], ids, slots, sampling)
        rt.run(step_fwd.name, vals, graph=use_graph)
        torch.cuda.synchronize()
        out = rt.emitted(step_fwd)
        pos += len(out)
        generated.extend(out)

    return generated[:steps]


def resolve_target(args: argparse.Namespace) -> tuple[Path, Path, list[Path], Path]:
    if args.config is not None:
        cfg = tomllib.loads(args.config.read_text())
        targets = cfg["targets"]
        target = targets[args.target] if args.target else next(iter(targets.values()))
        base = args.config.parent
        manifest = (base / target["manifest"]).resolve()
        kernels = (base / target["kernels"]).resolve()
        weights = [Path(w) for w in target["weights"]]
    else:
        if not (args.manifest and args.kernels and args.weights):
            raise SystemExit("pass --config or all of --manifest, --kernels, --weights")
        manifest = args.manifest.resolve()
        kernels = args.kernels.resolve()
        weights = [Path(w) for w in args.weights]
    return manifest, kernels, weights, args.tokenizer or (weights[0] / "tokenizer.json")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Run a kern manifest in pure Python.")
    ap.add_argument("--config", type=Path)
    ap.add_argument("--target")
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--kernels", type=Path)
    ap.add_argument("--weights", nargs="+", type=Path)
    ap.add_argument("--tokenizer", type=Path)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--rows", type=int)
    ap.add_argument("--capacity", type=int, default=4096)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args(argv)

    manifest, kernels, weights, tok_path = resolve_target(args)
    rt = Executor(manifest, kernels, weights, capacity_tokens=args.capacity, gpu=args.gpu)
    rt.init_single_sequence(seq_slot=1)

    tok = Tokenizer.from_file(str(tok_path))
    prompt_ids = tok.encode(args.prompt, add_special_tokens=False).ids
    out = generate(rt, prompt_ids, args.steps, args.rows, use_graph=not args.eager)
    print(f"{args.prompt}{tok.decode(out, skip_special_tokens=False)}")


if __name__ == "__main__":
    main()
