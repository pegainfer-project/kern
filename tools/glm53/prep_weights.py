#!/usr/bin/env python3
"""Build the derived GLM-5.3-Flash weight tensors the manifest cannot bind
directly (transposes / dtype casts / small merges). One-off, CPU-only.

Output: $GLM53_ARTIFACTS/derived.safetensors

Derived tensors (all verified against sglang's loaders):
  per DSA layer i in [3,7,...,43] (prefix P = model.language_model.layers.i):
    P.self_attn.kv_b.w_kc_t   bf16 [64,512,256]  per-head k-half TRANSPOSED
                              (sglang: w_kc.transpose(1,2).contiguous())
    P.self_attn.kv_b.w_vc     bf16 [64,256,512]  per-head v-half raw slices
    P.self_attn.indexer.k_norm_w_f32   f32 [128]      (k_norm.weight)
    P.self_attn.indexer.k_norm_b_f32   f32 [128]      (k_norm.bias)
    P.self_attn.indexer.weights_proj_f32 f32 [32,4096]
    P.self_attn.indexer.ape_f32        f32 [4,128]    (index_kpool_compress_ape)
  per KDA layer i:
    P.self_attn.conv1d_merged_f32  f32 [24576,4]  q|k|v conv1d rows concat
                                  ([8192,1,4] each -> [24576,4], f32)
  every layer i in 0..44:
    P.hc_attn_fn_f32  f32 [24,16384]
    P.hc_ffn_fn_f32   f32 [24,16384]
"""
import json
import os
import pathlib

import torch
from safetensors import safe_open
from safetensors.torch import save_file

CKPT = os.environ.get("GLM53_CHECKPOINT", "weights/GLM-5.3-Flash")
OUT = str(pathlib.Path(os.environ.get("GLM53_ARTIFACTS", "glm53-artifacts")) / "derived.safetensors")
LM = "model.language_model."
DSA = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]
KDA = [i for i in range(45) if i not in DSA]


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    idx = json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]
    handles = {}

    def get(name):
        f = idx[name]
        if f not in handles:
            handles[f] = safe_open(f"{CKPT}/{f}", framework="pt", device="cpu")
        return handles[f].get_tensor(name)

    out = {}
    for i in DSA:
        p = f"{LM}layers.{i}.self_attn."
        w = get(p + "kv_b_proj.weight")                       # [32768, 512]
        w = w.unflatten(0, (-1, 512))                          # [64, 512, 512]
        w_kc, w_vc = w.split([256, 256], dim=1)                # [64,256,512] each
        out[p + "kv_b.w_kc_t"] = w_kc.transpose(1, 2).contiguous()   # [64,512,256]
        out[p + "kv_b.w_vc"] = w_vc.contiguous()                     # [64,256,512]
        ip = p + "indexer."
        out[ip + "k_norm_w_f32"] = get(ip + "k_norm.weight").to(torch.float32)
        out[ip + "k_norm_b_f32"] = get(ip + "k_norm.bias").to(torch.float32)
        out[ip + "weights_proj_f32"] = get(ip + "weights_proj.weight").to(torch.float32)
        out[ip + "ape_f32"] = get(ip + "index_kpool_compress_ape").to(torch.float32)
        print("dsa", i, "done")

    for i in KDA:
        p = f"{LM}layers.{i}.self_attn."
        parts = [get(p + f"{t}_conv1d.weight").squeeze(1) for t in ("q", "k", "v")]
        out[p + "conv1d_merged_f32"] = torch.cat(parts, 0).to(torch.float32)  # [24576,4]

    for i in range(45):
        p = f"{LM}layers.{i}."
        out[p + "hc_attn_fn_f32"] = get(p + "hc_attn_fn").to(torch.float32)
        out[p + "hc_ffn_fn_f32"] = get(p + "hc_ffn_fn").to(torch.float32)

    save_file(out, OUT, metadata={"format": "pt"})
    total = sum(v.numel() * v.element_size() for v in out.values())
    print("wrote", OUT, len(out), "tensors", total / 2**30, "GiB")


if __name__ == "__main__":
    main()
