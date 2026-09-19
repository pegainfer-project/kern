#!/usr/bin/env python3
"""A manifest of one rank's MoE over TRT-LLM gen's batched GEMMs, shaped like
tools/trtllm-bmm/probe.py's dumps (inputs x bf16 [T, 3584], ids i32 [T, 16],
wts f32 [T, 16], the raw expert tensors of `local` experts, alpha / beta f32
[local]; outputs the shuffled weights, the quantised activation and out bf16
[T, 3584]): the K14 gate is `program_io` running program `moe` on the probe's
dumps, the shuffled weights and the quantisation matching its dumps bit for
bit and `out` matching `out.bin`. `rank` is the probe's (its experts start at
rank * local).

    python3 tools/gen_trtllm_bmm_probe.py <T> <local> <rank> > probe.json
"""
import json
import sys

import k3_moe_bmm as moe
import kern_manifest
from once import KERNELS

# the variants FlashInfer's own launcher picked for the probe's shape (tools/trtllm-bmm/probe.py capture)
CAPTURED = ("bmm_MxE4m3_MxE2m1MxE4m3_Fp32_Ab32_Bb32_Cb32_t128x64x256u2_s4_et128x64_m256x64x32_c2x1x1_rM_TN_transOut_"
            "schPd2x1x2x3_biasFp32M_bN_tma_ldgstsSf_rgTma_clmp_siTuGlu_lbW1_lsfbW1_dynB_sm100f",
            "bmm_Bfloat16_MxE2m1MxE4m3_Fp32_Ab32_Bb32_t128x64x256u2_s4_et128x64_m256x64x32_c2x1x1_rM_TN_transOut_"
            "schPd2x1x2x3_biasFp32M_bN_rgTma_clmp_dynB_sm100f")


def build(T, local, rank):
    H, I, K = moe.H, moe.I, moe.TOPK
    buffers = {
        "x": {"dtype": "bf16", "shape": ["tokens", H], "kind": "input"},
        "ids": {"dtype": "i32", "shape": ["tokens", K], "kind": "input"},
        "wts": {"dtype": "f32", "shape": ["tokens", K], "kind": "input"},
        "w13": {"dtype": "u8", "shape": [local, 2 * I, H // 2], "kind": "input"},
        "w13_sf": {"dtype": "u8", "shape": [local, 2 * I, H // 32], "kind": "input"},
        "w2": {"dtype": "u8", "shape": [local, H, I // 2], "kind": "input"},
        "w2_sf": {"dtype": "u8", "shape": [local, H, I // 32], "kind": "input"},
        "alpha": {"dtype": "f32", "shape": [local], "kind": "input"},
        "beta": {"dtype": "f32", "shape": [local], "kind": "input"},
        "w13s": {"dtype": "u8", "shape": [local, 2 * I, H // 2], "kind": "output"},
        "w13_sfs": {"dtype": "u8", "shape": [local, 2 * I, H // 32], "kind": "output"},
        "w2s": {"dtype": "u8", "shape": [local, H, I // 2], "kind": "output"},
        "w2_sfs": {"dtype": "u8", "shape": [local, H, I // 32], "kind": "output"},
        "x_fp8": {"dtype": "u8", "shape": ["tokens", H], "kind": "output"},
        "x_sf": {"dtype": "u8", "shape": ["tokens", H // 32], "kind": "output"},
        "out": {"dtype": "bf16", "shape": ["tokens", H], "kind": "output"},
    }
    p = moe.pieces(local, "tokens", T, T, "tokens", "tokens", names=CAPTURED)
    buffers.update(p["buffers"])
    ops = dict(p["ops"])
    b = lambda s: {"buf": s}
    i32 = lambda v: {"i32": v}
    prog = []
    for op, raw, scalars, threads in moe.shuffles(local):
        cubin, entry, params = KERNELS[op]
        ops.setdefault(op, {"params": params, "impl": {"launches": [moe.glue(entry, [-(-threads // 256), 1, 1])]}})
        prog.append({"label": raw, "op": op, "args": [b(raw), b(raw + "s"), *map(i32, scalars)]})
    prog.append(p["quant_step"](b("x"), b("x_fp8"), b("x_sf"), {"var": "tokens"}))
    prog += p["steps"](b("x_fp8"), b("x_sf"), b("ids"), b("wts"), b("w13s"), b("w13_sfs"), b("w2s"), b("w2_sfs"),
                       b("alpha"), b("beta"), b("out"), i32(rank))
    m = {
        "schema_version": kern_manifest.SCHEMA_VERSION,
        "model": f"trtllm-bmm-probe/t{T}-e{local}-r{rank}",
        "vars": {"tokens": {"max": T}},
        "buffers": buffers,
        "ops": ops,
        "programs": {"moe": kern_manifest.program(prog)},
    }
    return kern_manifest.normalize(m)


if __name__ == "__main__":
    json.dump(build(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])), sys.stdout, indent=1)
    print()
