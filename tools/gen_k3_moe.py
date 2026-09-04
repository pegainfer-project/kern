#!/usr/bin/env python3
"""Generate kern's MegaMoE program for one pruned-K3 MoE layer at EP<R>.

    python3 tools/gen_k3_moe.py --ranks 4 > examples/k3-moe-l1-ep4.json
    python3 tools/gen_k3_moe.py --ranks 1 > examples/k3-moe-l1-ep1.json

One SPMD manifest per world: every rank runs the same program with its own
expert shard, its slab exported to the `ep` group, and the peers' slab bases
in `slab_peers`. Program `moe`: quantise x into the slab, widen the routing,
then the fused DeepGEMM MegaMoE kernel (dispatch → L1 → situ → L2 → combine)
writes `y`. Geometry and slab offsets come from tools/k3-mega/layout_dump;
the cubins are pinned by sha256 (tools/build_k3_mega.sh, build_kernels.sh).
"""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import handwritten
import kern_manifest

REPO = pathlib.Path(__file__).resolve().parent.parent
EXPERTS = 224


def mega_build():
    out = REPO / "target" / "cubins"
    if not (out / "k3_mega_moe.cubin").exists() or not (out / "k3_mega_layout_dump").exists():
        subprocess.run([str(REPO / "tools" / "build_k3_mega.sh"), str(out)], check=True)
    return out


def alignup(a, b):
    return (a + b - 1) // b * b


def tmap(param, dtype, dims, strides, box, swizzle=0, l2=256):
    t = {"param": param, "dtype": dtype, "dims": dims, "strides": strides, "box": box,
         "l2_promotion": l2}
    if swizzle:
        t["swizzle"] = swizzle
    return {"pack": {"size": 128, "fields": [{"at": 0, "tensormap": t}]}}


def mega_pieces(ranks, tokens_max, wprefix=""):
    """The MegaMoE program pieces for one EP<ranks> world: the symmetric slab
    buffers, the routed-expert weight shapes (`wprefix` + name), the three ops,
    and `steps(x, topk_idx, topk_weight, y, label)` — the program steps that
    run one layer's routed experts from latent `x` into `y`."""
    cub = mega_build()
    lay = json.loads(subprocess.check_output([str(cub / "k3_mega_layout_dump"), str(EXPERTS), str(ranks)]))
    mega = {"cubin": "k3_mega_moe.cubin",
            "sha256": hashlib.sha256((cub / "k3_mega_moe.cubin").read_bytes()).hexdigest()}
    stage = handwritten.hw("k3_mega_stage")
    H, I, K = lay["hidden"], lay["intermediate"], lay["topk"]
    epr = lay["experts_per_rank"]
    ring, sf_ring = lay["ring_tokens"], lay["sf_ring_tokens"]
    off = lay["offsets"]
    assert tokens_max <= lay["max_tokens_per_rank"]
    assert lay["block_n"] == 128 and lay["block_k"] % 128 == 0
    sf_outer = lay["block_k"] // 128
    load_m = lay["block_m"] // 2

    buffers = {
        # The symmetric slab: workspace counters (self-restoring, start at
        # zero), this rank's staged inputs, the dispatch/L1/L2 rings that
        # peers write into over the fabric.
        "slab": {"dtype": "u8", "shape": [lay["slab_bytes"]], "kind": "carry", "export": True},
        "slab_peers": {"dtype": "u64", "shape": [ranks], "kind": "peer", "of": "slab", "group": "ep"},
        "stats": {"dtype": "i32", "shape": [epr], "kind": "workspace"},
    }
    weights = {
        "l1_weights": {"dtype": "u8", "shape": [epr * 2 * I * (H // 2)], "kind": "weight"},
        "l1_weights_sf": {"dtype": "i32", "shape": [epr * (H // 128) * 2 * I], "kind": "weight"},
        "l2_weights": {"dtype": "u8", "shape": [epr * H * (I // 2)], "kind": "weight"},
        "l2_weights_sf": {"dtype": "i32", "shape": [epr * (I // 128) * H], "kind": "weight"},
    }

    slab = lambda name: {"buf": "slab", "offset": off[name]}
    quant = {
        "params": ["in buffer<bf16>", "out buffer<u8>", "out buffer<u8>", "i32", "i32", "i32", "i32"],
        "impl": {"launches": [{
            **stage, "entry": "kern_k3_mega_quant_x", "block": [256, 1, 1],
            "grid": [{"ceil_div": [{"mul": ["tokens", H // 128]}, 8]}, 1, 1],
        }]},
    }
    routing = {
        "params": ["in buffer<i32>", "in buffer<f32>", "out buffer<u8>", "out buffer<u8>", "i32"],
        "impl": {"launches": [{
            **stage, "entry": "kern_k3_mega_write_routing", "block": [256, 1, 1],
            "grid": [{"ceil_div": [{"mul": ["tokens", K]}, 256]}, 1, 1],
        }]},
    }
    # Interface: y, stats, tokens, peers, rank, then the seven tensors the
    # 18 tensor maps describe (slab rings by region, weights whole).
    P_L1_ACTS, P_L1_SF, P_L1_W, P_L1_WSF, P_L2_ACTS, P_L2_SF, P_L2_W, P_L2_WSF = range(5, 13)
    sf_mn_l1, sf_mn_l2 = alignup(sf_ring, 4), alignup(sf_ring, 4)
    maps = [
        tmap(P_L1_ACTS, "u8", [H, ring], [H], [128, load_m], swizzle=128),
        tmap(P_L1_SF, "i32", [sf_mn_l1, H // 128], [sf_mn_l1 * 4], [lay["sf_block_m"], sf_outer]),
        tmap(P_L1_W, "u4", [H, epr * 2 * I], [H // 2], [128, 128], swizzle=128),
        tmap(P_L1_WSF, "i32", [2 * I, (H // 128) * epr], [2 * I * 4], [128, sf_outer]),
        # L1 output = L2 activations: the post-activation tile is BLOCK_N/2 wide.
        tmap(P_L2_ACTS, "u8", [I, ring], [I], [64, lay["store_block_m"]], swizzle=64),
        tmap(P_L2_ACTS, "u8", [I, ring], [I], [128, load_m], swizzle=128),
        tmap(P_L2_SF, "i32", [sf_mn_l2, I // 128], [sf_mn_l2 * 4], [lay["sf_block_m"], sf_outer]),
        tmap(P_L2_W, "u4", [I, epr * H], [I // 2], [128, 128], swizzle=128),
        tmap(P_L2_WSF, "i32", [H, (I // 128) * epr], [H * 4], [128, sf_outer]),
    ]
    mega_op = {
        "params": ["out buffer<bf16>", "out buffer<i32>", "i32", "in buffer<u64>", "i32",
                   "inout buffer<u8>", "inout buffer<u8>", "in buffer<u8>", "in buffer<i32>",
                   "inout buffer<u8>", "inout buffer<u8>", "in buffer<u8>", "in buffer<i32>"],
        "impl": {"launches": [{
            **mega, "entry": f"kern_k3_mega_moe_e{EXPERTS}_r{ranks}_situ",
            "params": ["out buffer<bf16>", "out buffer<i32>", "i32", "in buffer<u64>", "i32"] + ["bytes<128>"] * 18,
            "block": [lay["num_threads"], 1, 1], "grid": [lay["num_sms"], 1, 1], "cluster": [2, 1, 1],
            "shared_mem": lay["smem_size"],
            "args": [{"param": i} for i in range(5)] + maps + maps,
        }]},
    }
    ops = {"quant_x": quant, "write_routing": routing, "mega_moe": mega_op}

    def steps(x, topk_idx, topk_weight, y, label=""):
        w = lambda n: {"buf": wprefix + n}
        return [
            {"label": label + "quant_x", "op": "quant_x", "args": [
                x, slab("x"), slab("x_sf"), {"var": "tokens"}, {"i32": H}, {"i32": H}, {"i32": H // 128}]},
            {"label": label + "routing", "op": "write_routing", "args": [
                topk_idx, topk_weight, slab("topk_idx"), slab("topk_weights"),
                {"expr": {"mul": ["tokens", K]}}]},
            {"label": label + "mega_moe", "op": "mega_moe", "args": [
                y, {"buf": "stats"}, {"var": "tokens"}, {"buf": "slab_peers"}, {"rank": "ep"},
                slab("l1_acts"), slab("l1_acts_sf"), w("l1_weights"), w("l1_weights_sf"),
                slab("l2_acts"), slab("l2_acts_sf"), w("l2_weights"), w("l2_weights_sf")]},
        ]

    return {"buffers": buffers, "weights": weights, "ops": ops, "steps": steps, "layout": lay}


def build(ranks, tokens_max):
    mp = mega_pieces(ranks, tokens_max)
    H, K = mp["layout"]["hidden"], mp["layout"]["topk"]
    buffers = {
        "x": {"dtype": "bf16", "shape": ["tokens", H], "kind": "input"},
        "topk_idx": {"dtype": "i32", "shape": ["tokens", K], "kind": "input",
                     "domain": {"min": 0, "max": EXPERTS - 1}},
        "topk_weight": {"dtype": "f32", "shape": ["tokens", K], "kind": "input"},
        **mp["buffers"],
        **mp["weights"],
        "y": {"dtype": "bf16", "shape": ["tokens", H], "kind": "output"},
    }
    program = mp["steps"]({"buf": "x"}, {"buf": "topk_idx"}, {"buf": "topk_weight"}, {"buf": "y"})
    m = {
        "schema_version": kern_manifest.SCHEMA_VERSION,
        "model": f"kimi-k3-pruned-75pct/moe-l1/ep{ranks}",
        "vars": {"tokens": {"max": tokens_max}},
        "topology": {"groups": {"ep": ranks}},
        "buffers": buffers,
        "ops": mp["ops"],
        "programs": {"moe": kern_manifest.program(program)},
    }
    return kern_manifest.normalize(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", type=int, default=4)
    ap.add_argument("--tokens-max", type=int, default=16896)
    args = ap.parse_args()
    json.dump(build(args.ranks, args.tokens_max), sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
