#!/usr/bin/env python3
"""Track per-step raw param bytes of shape-sensitive decode kernels.

For each tracked call site (symbol + occurrence in the forward), dump the
raw param bytes of the 6 bs=1 decode steps of prompt1. Fields that vary
step-over-step are position/seqlen-dependent; constants are ABI literals.

Output: $GLM53_ARTIFACTS/tracks.json
"""
import os
import json
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from mine_decode import load, forwards, EMB, occ_list  # noqa: E402

TRACK = [
    "FlashAttnFwdSm90", "FlashAttnFwdCombine", "prepare_varlen_num_blocks",
    "sm90_fp8_paged_mqa_logits", "sm90_paged_mqa_logits_metadata",
    "kpool_topk_transform", "_kpool_decode_update_and_maybe_write_cache",
    "set_mla_kv_buffer_kernel_norope", "_act_quant_kernel",
    "fast_hadamard_transform", "sm90_tf32_hc_prenorm_gemm",
    "mhc_pre_big_fuse_with_norm", "mhc_fused_post_pre_fma", "mhc_post_tilelang",
    "fused_moe_kernel", "_router_triton_kernel", "_moe_align_small_numel",
    "_moe_sum_reduce_kernel", "fused_sigmoid_gating_delta_rule_update",
    "_causal_conv1d_update_kernel", "layer_norm_gated_fwd_kernel",
    "per_token_group_quant_flat", "tiny_n_gemm", "silu_mul_clamp",
    "all_reduce_1shot_push",
]


def main():
    recs = load()
    fs = forwards(recs, EMB["p1"], 6)
    occs = [occ_list(f) for f in fs]
    n = len(occs[1])
    tracks = {}
    for i in range(n):
        sym, k = occs[1][i]
        if not any(t in sym for t in TRACK):
            continue
        steps = []
        for f in fs:
            r = f[i]
            steps.append({"grid": r["grid"],
                          "params": [{"size": p["size"], "data": p["data"],
                                      "ptr": p.get("pointer", {}).get("range_start")}
                                     for p in (r["params"] or [])]})
        tracks.setdefault(sym, []).append({"site": i, "steps": steps})
    out = pathlib.Path(os.environ.get("GLM53_ARTIFACTS", "glm53-artifacts")) / "tracks.json"
    with open(out, "w") as f:
        json.dump(tracks, f)
    print("tracked symbols:", len(tracks),
          "sites:", sum(len(v) for v in tracks.values()))


if __name__ == "__main__":
    main()
