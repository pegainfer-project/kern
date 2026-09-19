"""K3's routed MoE as TRT-LLM gen batched GEMMs (kernel index families
`trtllm_bmm_mxe4m3_mxe2m1_mxe4m3` for FC1 with the fused SiTU gate and
`trtllm_bmm_bf16_mxe2m1_mxe4m3` for FC2, tools/kernels/abi/trtllm_bmm.py) around
the glue of tools/kernels-src/k3_moe_prefill.cu: the pieces the prefill
generator (tools/gen_k3.py) and the probe gate (tools/gen_trtllm_bmm_probe.py)
share.

One rank holds `local` experts. Its part of a chunk: quantise the latent rows
to mxfp8, then over every token of the chunk (gathered from the tray) build
the routing tables of its experts, FC1 (gate | up, SiTU, mxfp8 out), FC2
(bf16 out per permuted row), and the top-k combine into a chunk-wide bf16
partial whose rows past the chunk are zero, ready for the reduce-scatter.
The expert weights are the checkpoint's mxfp4 tensors row-shuffled once
after load (`k3_moe_w_shuffle` / `k3_moe_sf_shuffle`, once.py): FC1's rows
are [up; gate] before the shuffle, so the kernel's pair (x0, x1) is (up, gate)
and its activation beta·tanh(x0/beta)·alpha·tanh(x1/alpha)·sigmoid(x1) is
K3's situ with alpha 4, beta 25.
"""
from kernels import index
from kernels.abi import trtllm_bmm

H, I, TOPK, EXPERTS = 3584, 3072, 16, 224
ALPHA, BETA = 4.0, 25.0
GLUE = "k3_moe_prefill"
FC1, FC2 = "trtllm_bmm_mxe4m3_mxe2m1_mxe4m3", "trtllm_bmm_bf16_mxe2m1_mxe4m3"
SHAPE = "k3-prefill-16k-ep4"
ROUTE_BLOCK = 256


def ceil_div(a, b):
    return -(-a // b)


def variants(names=None):
    """The two GEMMs: the index's picks for SHAPE, or the (fc1, fc2) variant names given (a sweep, the probe's capture)."""
    v1, v2 = ((index.variant_by_name(FC1, names[0]), index.variant_by_name(FC2, names[1])) if names
              else (index.pick(FC1, "moe_fc1", SHAPE), index.pick(FC2, "moe_fc2", SHAPE)))
    assert v1.tags["sf_layout"][2] == v2.tags["sf_layout"][1], "FC1 writes its scales in the layout FC2 reads"
    assert v1.tags["act"] == "siTuGlu" and v1.tags["fused_act"] and v1.tags["route"] != "none"
    return v1, v2


def glue(entry, grid, block=256, params=None, args=None):
    return {**index.variant(GLUE).module, "entry": entry, "grid": grid, "block": [block, 1, 1],
            **({"params": params, "args": args} if params else {})}


def pieces(local, tokens, tokens_max, rows_max, quant_rows, out_rows, prefix="", names=None):
    """The ops and workspace buffers for `local` experts over `tokens` (the chunk's token count: a var
    name or an expression, at most `tokens_max`) routed from an activation of `rows_max` rows; the quant
    runs over `quant_rows` rows and the combine writes `out_rows` rows (expressions). `steps` lists the
    calls of one layer."""
    v1, v2 = variants(names)
    tile = v1.tags["tile"][1]
    assert v2.tags["tile"][1] == tile, "both GEMMs read one set of CTA tables"
    ctas = trtllm_bmm.ctas_bound(tokens, TOPK, local, tile)
    ctas_max = local + ceil_div(TOPK * tokens_max, tile)
    padded_max = ctas_max * tile
    blocks_max = ceil_div(tokens_max, ROUTE_BLOCK)
    blocks = {"ceil_div": [tokens, ROUTE_BLOCK]}
    n = lambda s: prefix + s
    i32 = lambda v: {"i32": v}
    dim = lambda x: {"var": x} if isinstance(x, str) else {"expr": x}
    buffers = {
        n("blockcount"): {"dtype": "i32", "shape": [blocks_max, local], "kind": "workspace"},
        n("blockoff"): {"dtype": "i32", "shape": [blocks_max, local], "kind": "workspace"},
        n("cta_batch"): {"dtype": "i32", "shape": [ctas_max], "kind": "workspace"},
        n("cta_limit"): {"dtype": "i32", "shape": [ctas_max], "kind": "workspace"},
        n("num_non_exiting"): {"dtype": "i32", "shape": [1], "kind": "workspace"},
        n("total_padded"): {"dtype": "i32", "shape": [1], "kind": "workspace"},
        n("route_map"): {"dtype": "i32", "shape": [padded_max], "kind": "workspace"},
        n("exp2perm"): {"dtype": "i32", "shape": [tokens_max, TOPK], "kind": "workspace"},
        n("fc1_out"): {"dtype": "u8", "shape": [padded_max, I], "kind": "workspace"},
        n("fc1_sf"): {"dtype": "u8", "shape": [ceil_div(padded_max, 128) * 128, I // 32], "kind": "workspace"},
        n("fc2_out"): {"dtype": "bf16", "shape": [padded_max, H], "kind": "workspace"},
    }
    ops = {
        "moe_quant": {
            "params": ["in buffer<bf16>", "out buffer<u8>", "out buffer<u8>", "i32", "i32"],
            "impl": {"launches": [glue("kern_k3_moe_quant", [{"ceil_div": [{"mul": [quant_rows, H // 32]}, 256]}, 1, 1])]},
        },
        "moe_route_count": {
            "params": ["in buffer<i32>", "i32", "i32", "i32", "out buffer<i32>"],
            "impl": {"launches": [glue("kern_k3_moe_route_count", [blocks, 1, 1])]},
        },
        "moe_route_tables": {
            "params": ["in buffer<i32>", "i32", "i32", "i32", "out buffer<i32>", "out buffer<i32>", "out buffer<i32>",
                       "out buffer<i32>", "out buffer<i32>", "out buffer<i32>"],
            "impl": {"launches": [glue("kern_k3_moe_route_tables", [1, 1, 1], block=1024)]},
        },
        "moe_route_scatter": {
            "params": ["in buffer<i32>", "in buffer<i32>", "i32", "i32", "i32", "inout buffer<i32>", "out buffer<i32>"],
            "impl": {"launches": [glue("kern_k3_moe_route_scatter", [blocks, 1, 1])]},
        },
        "moe_fc1": trtllm_bmm.op(v1, 2 * I, H, local, tokens, rows_max, ctas, ctas_max),
        "moe_fc2": trtllm_bmm.op(v2, H, I, local, tokens, rows_max, ctas, ctas_max),
        "moe_finalize": {
            "params": ["in buffer<bf16>", "in buffer<i32>", "in buffer<f32>", "out buffer<bf16>", "i32", "i32"],
            "impl": {"launches": [glue("kern_k3_moe_finalize", [out_rows, 1, 1])]},
        },
    }
    assert trtllm_bmm.params(v1) == ["a", "sf_a", "b", "sf_b", "c", "sf_c", "route_map", "alpha", "beta",
                                     "num_non_exiting", "total_padded", "cta_batch", "cta_limit"]
    assert trtllm_bmm.params(v2) == ["a", "sf_a", "b", "sf_b", "c", "num_non_exiting", "total_padded", "cta_batch",
                                     "cta_limit"]
    b = lambda s: {"buf": n(s)}
    tables = [b("num_non_exiting"), b("total_padded"), b("cta_batch"), b("cta_limit")]

    def quant_step(x, q, sf, rows, label=""):
        return {"label": label + "moe_quant", "op": "moe_quant", "args": [x, q, sf, rows, i32(H)]}

    def steps(q, sf, ids, wts, w13s, w13_sfs, w2s, w2_sfs, alpha, beta, out, rank, label=""):
        T = dim(tokens)
        return [
            {"label": label + "route_count", "op": "moe_route_count",
             "args": [ids, T, rank, i32(local), b("blockcount")]},
            {"label": label + "route_tables", "op": "moe_route_tables",
             "args": [b("blockcount"), dim(blocks), i32(local), i32(tile), b("blockoff"), b("cta_batch"),
                      b("cta_limit"), b("num_non_exiting"), b("total_padded"), b("route_map")]},
            {"label": label + "route_scatter", "op": "moe_route_scatter",
             "args": [ids, b("blockoff"), T, rank, i32(local), b("route_map"), b("exp2perm")]},
            {"label": label + "fc1", "op": "moe_fc1",
             "args": [w13s, w13_sfs, q, sf, b("fc1_out"), b("fc1_sf"), b("route_map"), alpha, beta, *tables]},
            {"label": label + "fc2", "op": "moe_fc2", "args": [w2s, w2_sfs, b("fc1_out"), b("fc1_sf"), b("fc2_out"), *tables]},
            {"label": label + "finalize", "op": "moe_finalize",
             "args": [b("fc2_out"), b("exp2perm"), wts, out, T, i32(H)]},
        ]

    return {"buffers": buffers, "ops": ops, "quant_step": quant_step, "steps": steps, "tile": tile,
            "padded_max": padded_max}


def shuffles(local):
    """The four weight-prep calls of one layer as (op, args after the buffers, threads): FC1's [up; gate] rows
    and scales with the gated interleave, FC2's plain."""
    return [
        ("k3_moe_w_shuffle", "w13", [local, 2 * I, H // 2, 1], local * 2 * I * (H // 2) // 16),
        ("k3_moe_sf_shuffle", "w13_sf", [local, 2 * I, H // 32, 1], local * 2 * I * (H // 32)),
        ("k3_moe_w_shuffle", "w2", [local, H, I // 2, 0], local * H * (I // 2) // 16),
        ("k3_moe_sf_shuffle", "w2_sf", [local, H, I // 32, 0], local * H * (I // 32)),
    ]
