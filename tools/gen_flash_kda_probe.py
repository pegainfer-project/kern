"""A manifest of just the FlashKDA op, shaped like tools/flash-kda/probe.cu's dumps
(inputs q k v g [T, H*128] bf16, beta_ht [H, T] bf16, dt_bias [H*128] f32, a_log [H] f32,
state_in [H, 128, 128] f32; outputs out [T, H*128] bf16, state_out): the C2 gate is
`program_io` running it on the probe's dumps and the outputs matching bit for bit.

    python3 tools/gen_flash_kda_probe.py <T> <H> > probe.json
"""
import json
import sys

import flash_kda_abi
import handwritten
import kern_manifest


def build(T, H):
    D = flash_kda_abi.HEAD_DIM
    inner = H * D
    buffers = {
        **{n: {"dtype": "bf16", "shape": ["span", inner], "kind": "input"} for n in ["q", "k", "v", "g"]},
        "beta_ht": {"dtype": "bf16", "shape": [H, "span"], "kind": "input"},
        "dt_bias": {"dtype": "f32", "shape": [inner], "kind": "input"},
        "a_log": {"dtype": "f32", "shape": [H], "kind": "input"},
        "state_in": {"dtype": "f32", "shape": [H, D, D], "kind": "input"},
        "state_out": {"dtype": "f32", "shape": [H, D, D], "kind": "output"},
        "out": {"dtype": "bf16", "shape": ["span", inner], "kind": "output"},
        **flash_kda_abi.workspace_buffers(H, T),
    }
    ws = ["span_ws_kd", "span_ws_qd", "span_ws_kr", "span_ws_gt", "span_ws_inv", "span_ws_mqk"]
    args = [{"buf": n} for n in ["q", "k", "v", "g", "beta_ht", "dt_bias", "a_log", "state_in", "state_out", "out"] + ws]
    m = {
        "schema_version": kern_manifest.SCHEMA_VERSION,
        "model": f"flash-kda-probe/t{T}-h{H}",
        "vars": {"span": {"max": T}},
        "buffers": buffers,
        # probe.cu passes scale = 1/128
        "ops": {"flash_kda": flash_kda_abi.op(H, T, handwritten.prebuilt(flash_kda_abi.MODULE), scale=1.0 / 128)},
        "programs": {"span": [{"label": "kda", "op": "flash_kda", "args": args + [{"var": "span"}]}]},
    }
    return kern_manifest.normalize(m)


if __name__ == "__main__":
    json.dump(build(int(sys.argv[1]), int(sys.argv[2])), sys.stdout, indent=1)
    print()
