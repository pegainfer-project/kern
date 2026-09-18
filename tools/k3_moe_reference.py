#!/usr/bin/env python3
"""A test vector for one pruned-K3 MoE layer, with a host reference output
(the oracle of crates/kern-runtime/examples/k3_moe_ep.rs; the weights
themselves bind from the checkpoint, tools/gen_k3_moe.py).

    python3 tools/k3_moe_reference.py --out weights/k3-moe-l1 [--layer 1] [--ranks 4]
        [--tokens-per-rank 64] [--seed 0] [--device cpu]

Writes --out/inputs.safetensors: x bf16 [R*T, 3584], topk_idx i32 [R*T, 16],
topk_weight f32 [R*T, 16], y_ref f32 [R*T, 3584].

The reference follows the kernel's arithmetic: x -> e4m3 with per-32 UE8M0
scales (ceil, exponent 1..254); L1 in f32 over dequantised fp4; gate/up
rounded to bf16; situ activation 4·tanh(g/4)·σ(g)·25·tanh(u/25) times the
routing weight in f32; requantised per 32; L2 in f32; each expert's output
rounded to bf16 and the 16 summed in f32. Needs torch + safetensors (the
kernel-lab container has them).
"""
import argparse
import json
import os
import pathlib

import numpy as np
import torch
from safetensors import safe_open
from safetensors.numpy import save_file

HIDDEN, INTER, TOPK, EXPERTS = 3584, 3072, 16, 224
CKPT = "/mnt/shared/weights/kimi-k3-pruned-75pct"
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def dequant_fp4(packed_u8, sf_u8, dev):
    """[n, k/2] u8 + [n, k/32] u8 -> [n, k] f32 (low nibble = even element)."""
    p = torch.from_numpy(packed_u8).to(dev)
    lo, hi = (p & 0xF).long(), (p >> 4).long()
    vals = torch.stack([E2M1.to(dev)[lo], E2M1.to(dev)[hi]], dim=-1).reshape(p.shape[0], -1)
    scale = torch.exp2(torch.from_numpy(sf_u8).to(dev).float() - 127.0)
    return vals * scale.repeat_interleave(32, dim=1)


def quant_e4m3(v):
    """Per-32 UE8M0 quantisation as the kernel does it; returns the dequantised f32."""
    n, k = v.shape
    g = v.reshape(n, k // 32, 32)
    amax = g.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    raw = amax / 448.0
    bits = raw.view(torch.int32) & 0x7FFFFFFF
    exp = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).int()
    exp = exp.clamp(1, 254)
    sf = (exp << 23).view(torch.float32)
    q = (g / sf).to(torch.float8_e4m3fn).float()
    return (q * sf).reshape(n, k)


def situ(gate, up):
    return 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate) * (25.0 * torch.tanh(up / 25.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--ranks", type=int, default=4)
    ap.add_argument("--tokens-per-rank", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ckpt", default=CKPT)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    R, T = args.ranks, args.tokens_per_rank

    index = json.load(open(os.path.join(args.ckpt, "model.safetensors.index.json")))["weight_map"]
    prefix = f"language_model.model.layers.{args.layer}.block_sparse_moe.experts."
    shards = {}

    def tensor(name):
        f = index[prefix + name]
        if f not in shards:
            shards[f] = safe_open(os.path.join(args.ckpt, f), framework="np")
        return shards[f].get_tensor(prefix + name)

    # ---- inputs
    g = torch.Generator().manual_seed(args.seed)
    x = torch.randn(R * T, HIDDEN, generator=g).to(torch.bfloat16)
    topk = torch.stack([torch.randperm(EXPERTS, generator=g)[:TOPK] for _ in range(R * T)]).int()
    w = torch.rand(R * T, TOPK, generator=g) + 0.05
    w = (w / w.sum(dim=1, keepdim=True)).float()

    # ---- reference
    xq = quant_e4m3(x.float().to(dev))
    y = torch.zeros(R * T, HIDDEN, device=dev)
    topk_d, w_d = topk.to(dev), w.to(dev)
    for e in range(EXPERTS):
        w1, w3, w2 = (tensor(f"{e}.{n}.weight_packed") for n in ("w1", "w3", "w2"))
        s1, s3, s2 = (tensor(f"{e}.{n}.weight_scale") for n in ("w1", "w3", "w2"))
        w13 = np.concatenate([w1, w3], axis=0)
        s13 = np.concatenate([s1, s3], axis=0)
        rows, slots = torch.nonzero(topk_d == e, as_tuple=True)
        if rows.numel() == 0:
            continue
        W13 = dequant_fp4(w13, s13, dev)  # [6144, 3584]
        W2 = dequant_fp4(w2, s2, dev)      # [3584, 3072]
        h = xq[rows] @ W13.T
        gate = h[:, :INTER].to(torch.bfloat16).float()
        up = h[:, INTER:].to(torch.bfloat16).float()
        act = situ(gate, up) * w_d[rows, slots][:, None]
        mq = quant_e4m3(act)
        ye = (mq @ W2.T).to(torch.bfloat16).float()
        y.index_add_(0, rows, ye)
        if e % 32 == 0:
            print(f"expert {e}/{EXPERTS}", flush=True)

    save_file({
        "x": x.view(torch.int16).numpy().view(np.uint16).reshape(-1),
        "topk_idx": topk.numpy().reshape(-1),
        "topk_weight": w.numpy().reshape(-1),
        "y_ref": y.cpu().numpy().reshape(-1),
    }, str(out / "inputs.safetensors"), metadata={"ranks": str(R), "tokens_per_rank": str(T),
                                                   "layer": str(args.layer), "seed": str(args.seed)})
    print(f"wrote {out / 'inputs.safetensors'}: {R}x{T} tokens")


if __name__ == "__main__":
    main()
