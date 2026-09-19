"""A manifest of just the FMHA op, shaped like tools/trtllm-fmha/probe.py's dumps
(inputs q [T, 96*192] bf16, kv [n, 96*320] bf16, lens i32 [8] = seq_lens | pad | cum_q | cum_kv;
output o [T, 96*128] bf16): the K13 gate is `program_io` running it on the probe's dumps and
`o` matching `out.bin` bit for bit.

    python3 tools/gen_trtllm_fmha_probe.py <T> <n> > probe.json
"""
import json
import sys

from kernels import index
import kern_manifest
import trtllm_fmha_abi as fmha

HEADS = 96


def build(T, n):
    buffers = {
        "q": {"dtype": "bf16", "shape": ["tokens", HEADS * fmha.HQK], "kind": "input"},
        "kv": {"dtype": "bf16", "shape": [n, HEADS * fmha.KV_ROW], "kind": "input"},
        "lens": {"dtype": "i32", "shape": [8], "kind": "input"},
        "o": {"dtype": "bf16", "shape": ["tokens", HEADS * fmha.HV], "kind": "output"},
        "scratch": {"dtype": "u8", "shape": [fmha.SCRATCH_BYTES], "kind": "workspace"},
    }
    args = [{"buf": "q"}, {"buf": "kv"}, {"buf": "kv", "offset": fmha.HQK * 2}, {"buf": "o"}, {"buf": "lens"},
            {"buf": "lens", "offset": 8}, {"buf": "lens", "offset": 16}, {"buf": "scratch"},
            {"buf": "scratch", "offset": fmha.PARTIAL_O_OFFSET}]
    m = {
        "schema_version": kern_manifest.SCHEMA_VERSION,
        "model": f"trtllm-fmha-probe/t{T}-n{n}",
        "vars": {"tokens": {"max": T}},
        "buffers": buffers,
        "ops": {"fmha": fmha.op(HEADS, T, n, index.variant(fmha.MODULE).module, "tokens")},
        "programs": {"attn": kern_manifest.program([{"label": "attn", "op": "fmha", "args": args}])},
    }
    return kern_manifest.normalize(m)


if __name__ == "__main__":
    json.dump(build(int(sys.argv[1]), int(sys.argv[2])), sys.stdout, indent=1)
    print()
