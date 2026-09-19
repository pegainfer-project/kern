"""TRT-LLM gen's batched GEMM (the `trtllm_bmm_*` kernel index families) as one
manifest op: one rank's local experts in batch-N mode (A the expert weights,
B the routed activation, C the output), the batch's extent decided on
device by routing tables (early exit), so one launch covers any token count.

The pack is `KernelParams` (KernelParamsDecl.h; 17472 bytes as the cubin
declares it). Everything in it is derived, not copied: `abc` and `sf` port
`makeTmaShapeStrideAbc` / `makeTmaShapeStrideSfAb`, `nd` and `sf_map` port
the two descriptor builders (TmaDescriptor.h), `op` ports the batch-N branch
of `setKernelParams` and `getGridDim` (BatchedGemmInterface.h) from the
variant's tags and the problem shape. tools/trtllm-bmm/check_abi.py replays
the port against the capture the family's `abi_capture` names, field for
field. The descriptor for a non-routed activation or for C uses the
launcher's out-of-bounds trick (dims of 2^31 whose strides wrap the address
space, so a coordinate past the tile's row limit reads zero or drops the
store); such a descriptor is unbounded, and the buffers behind it are sized
here by the padded token count instead.

Interface (`params(v)` names it in order): `a` u8 [batches, m, k/2] mxfp4
weights in trtllm's shuffled layout | `sf_a` u8 [batches*m*k/32] UE8M0 in
R128c4 | `b` u8 [rows, k] fp8 activation | `sf_b` u8 its UE8M0 scales (linear
[rows, k/32] when routed, the layout `sf_layout[1]` says otherwise) | `c` the
output ([padded, m/2] u8 with a gated activation, [padded, m] otherwise) |
`sf_c` u8 when the variant writes scales | `route_map` i32 [padded] permuted
row -> token when it routes | `alpha`, `beta` f32 [batches] when gated |
`num_non_exiting`, `total_padded` i32 [1] | `cta_batch`, `cta_limit` i32
[ctas]: the routing tables (docs/k3-kernel-abi.md, routing).
"""
SIZE = 17472
TMA_DIM_MAX, XLARGE_N = 1 << 31, 1 << 35
OFF = {
    "tmaA": 0, "tmaB": 128, "tmaC": 256, "tmaSfA": 384, "tmaSfB": 512, "ptrA": 768, "strideInBytesA": 776,
    "ptrB": 784, "strideInBytesB": 792, "ptrC": 800, "ptrGatedActAlpha": 856, "ptrGatedActBeta": 864, "k": 872,
    "nm": 876, "tileStridePerBatch": 880, "ptrSfA": 896, "ptrSfB": 904, "ptrSfC": 944, "ptrRouteMap": 952,
    "numTokens": 960, "numBatches": 964, "ptrNumNonExitingCtas": 968, "ptrTotalNumPaddedTokens": 976,
    "ptrCtaIdxXyToBatchIdx": 984, "ptrCtaIdxXyToMnLimit": 992, "tpGrpSize": 17404,
}
BITS = {"mxe2m1": 4, "e2m1": 4, "mxe4m3": 8, "e4m3": 8, "ue8m0": 8, "u8": 8, "bf16": 16, "f16": 16, "f32": 32}
# trtllm dtype -> (kern tensormap dtype, kern buffer dtype); an mxfp4 matrix the MMA pads is `u4`
# (CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B), an unpadded one `u4packed`
TMA = {"mxe4m3": "u8", "e4m3": "u8", "ue8m0": "u8", "u8": "u8", "bf16": "bf16", "f16": "f16", "f32": "f32"}
BUF = {"mxe4m3": "u8", "e4m3": "u8", "ue8m0": "u8", "u8": "u8", "mxe2m1": "u8", "e2m1": "u8", "bf16": "bf16",
       "f16": "f16", "f32": "f32"}


def ceil_div(a, b):
    return -(-a // b)


def max_ctas(tokens, topk, experts, tile_n):
    """The most CTAs the batch dim can need (flashinfer's `getMaxNumCtasInBatchDim`): one per expert
    that gets a token, then one per full tile of what is left."""
    expanded = tokens * topk
    filled = min(experts, expanded)
    return filled + (expanded - filled) // tile_n


def ctas_bound(tokens, topk, experts, tile_n):
    """`max_ctas` as a manifest expression over the token var, rounded up: experts + ceil(tokens*topk/tile_n)."""
    return {"add": [{"ceil_div": [{"mul": [tokens, topk]}, tile_n]}, experts]}


def abc(t, m, n, k, tile_m, tile_n, tile_k, matrix, batches, valid_n=None, valid_k=None):
    """`makeTmaShapeStrideAbc` for batch N: (shape, stride, box) in elements, innermost first."""
    valid_n, valid_k = n if valid_n is None else valid_n, k if valid_k is None else valid_k
    transposed = bool(t["transpose_out"])
    weights = (matrix == "A" and transposed) or (matrix == "B" and not transposed)
    oob = bool(t["tma_oob"]) and {"A": False, "B": t["route"] == "none",
                                   "C": bool(t["tma_store"]) and not t["c_multicast"]}[matrix]
    tokens, tokens_valid, cta_tokens = (m, m, tile_m) if matrix in "AC" else (n, valid_n, tile_n)
    tile_tokens = t["epilogue_tile"][0] if matrix == "C" else cta_tokens
    hidden, hidden_valid, cta_hidden = (n, valid_n, tile_n) if matrix == "C" else (k, valid_k, tile_k)
    tile_hidden = t["epilogue_tile"][1] if matrix == "C" else cta_hidden
    if matrix == "C" and transposed:
        tokens, hidden, tokens_valid, hidden_valid = hidden, tokens, hidden_valid, tokens_valid
        cta_tokens, cta_hidden, tile_tokens, tile_hidden = cta_hidden, cta_tokens, tile_hidden, tile_tokens
    if matrix == "C" and t["fused_act"]:
        hidden, hidden_valid, tile_hidden, cta_hidden = hidden // 2, hidden_valid // 2, tile_hidden // 2, cta_hidden // 2
    if oob:
        shape, stride = [hidden_valid, cta_tokens, TMA_DIM_MAX, TMA_DIM_MAX], [1, hidden, XLARGE_N - hidden, hidden]
    elif weights:
        shape, stride = [hidden_valid, tokens_valid, batches], [1, hidden, hidden * tokens]
    else:
        shape, stride = [hidden_valid, tokens_valid], [1, hidden]
    box = [tile_hidden, tile_tokens]
    if matrix == "B" and box[1] > 1 and t["cluster"][0] >= 2:
        box[1] //= 2
    return shape, stride, box


def sf(rows, k, tile_rows, tile_k, layout, reshape, per_sf=32):
    """`makeTmaShapeStrideSfAb`: the scale tensor of `rows` x `k` elements in R128c4 or R8c4 blocks."""
    if layout == "r128c4":
        shape = [256, 2, ceil_div(k, per_sf * 4), ceil_div(rows, 128)]
        box = [256, 2, ceil_div(tile_k, per_sf * 4), ceil_div(tile_rows, 128)]
    elif layout == "r8c4":
        r = min(ceil_div(tile_k, per_sf * 4), reshape)
        assert ceil_div(k, per_sf * 4) % r == 0, f"sf hidden {k} is not a multiple of {r} repeats"
        shape = [r * 32, ceil_div(k, per_sf * 4 * r), ceil_div(rows, 8)]
        box = [r * 32, ceil_div(tile_k, per_sf * 4 * r), ceil_div(tile_rows, 8)]
    else:
        raise ValueError(f"sf layout {layout}")
    stride = [1]
    for s in shape[:-1]:
        stride.append(stride[-1] * s)
    return shape, stride, box


def nd(param, dtype, shape, stride, box, pad, swizzle=True):
    """`buildNdTmaDescriptor` as a kern tensormap: swizzle by the fastest box's bytes, that box clamped to 128 B."""
    bits, mult = BITS[dtype], 1
    kind = TMA.get(dtype)
    if dtype in ("mxe2m1", "e2m1"):
        kind, mult = ("u4", 2) if pad else ("u4packed", 1)
    fastest = box[0] * bits * mult // 8
    sw = 0
    if swizzle:
        sw = next((s for s in (128, 64, 32) if fastest % s == 0), None)
        if sw is None:
            assert fastest % 16 == 0 and bits == 8, f"fastest box of {fastest} bytes"
            sw = 0
    per_u32 = 32 // (bits * mult)
    boxes = [min(per_u32 * 32, box[0])] + [1] * (len(shape) - 1)
    for i, b in enumerate(box[1:], 1):
        assert b <= 256, f"box[{i}] = {b}"
        boxes[i] = b
    return {"param": param, "dtype": kind, "dims": list(shape), "strides": [s * bits // 8 for s in stride[1:]],
            "box": boxes, "swizzle": sw, "l2_promotion": 128, **({"wrap": True} if TMA_DIM_MAX in shape else {})}


def sf_map(param, shape, stride, box):
    """`buildSfTmaDescriptor`: UE8M0 bytes, no swizzle."""
    return {"param": param, "dtype": "u8", "dims": list(shape), "strides": list(stride[1:]), "box": list(box),
            "swizzle": 0, "l2_promotion": 128}


def params(v):
    """The op's interface param names in order for variant `v` (see the module doc)."""
    t = v.tags
    return (["a", "sf_a", "b", "sf_b", "c"] + (["sf_c"] if t["dtype_sf_c"] != "void" else [])
            + (["route_map"] if t["route"] != "none" else []) + (["alpha", "beta"] if t["fused_act"] else [])
            + ["num_non_exiting", "total_padded", "cta_batch", "cta_limit"])


def op(v, m, k, batches, tokens, tokens_max, ctas, ctas_max):
    """The manifest op for variant `v` over `batches` local experts of `m` output rows and `k` inputs.
    `tokens` (a var name or an expression) is the token count the routing saw, `tokens_max` its bound
    (the routed activation's rows); `ctas` is the batch-dim grid (an expression, see `ctas_bound`),
    `ctas_max` its bound, which sizes the padded rows every descriptor over them declares."""
    t = v.tags
    tile_m, tile_n, tile_k = t["tile"]
    assert t["batch_mode"] == "batch_n" and not t["static_batch"] and t["early_exit"], "not a dynamic batch-N kernel"
    assert t["layout"] == ["major_k", "major_k"] and not t["deepseek_fp8"] and not t["c_multicast"]
    assert t["sched"] in ("static", "persistent"), f"scheduler {t['sched']} needs a fixed grid"
    assert t["cluster"][1] == 1, "a cluster along the batch dim changes the tile of the routing tables"
    assert m % tile_m == 0, f"m {m} is not a multiple of the tile's {tile_m}"
    P = {n: i for i, n in enumerate(params(v))}
    padded = ctas_max * tile_n
    pad_a = t["mma_kind"] == "mxfp8fp6fp4" and t["dtype_a"] == "mxe2m1"
    pad_b = t["mma_kind"] == "mxfp8fp6fp4" and t["dtype_b"] == "mxe2m1"
    routed, route_sf = t["route"] != "none", t["route_sf"]
    per_sf = t["sf_block"][0]
    tmap = lambda name, d: {"at": OFF[name], "tensormap": d}
    fields = [tmap("tmaA", nd(P["a"], t["dtype_a"], *abc(t, m, tokens_max, k, tile_m, tile_n, tile_k, "A", batches),
                              pad_a))]
    if t["route"] != "ldgsts":
        shape = (abc(t, m, tokens_max, k, tile_m, 1, tile_k, "B", batches) if routed
                 else abc(t, m, padded, k, tile_m, tile_n, tile_k, "B", batches))
        fields.append(tmap("tmaB", nd(P["b"], t["dtype_b"], *shape, pad_b)))
    fields.append(tmap("tmaSfA", sf_map(P["sf_a"], *sf(m * batches, k, tile_m, tile_k, t["sf_layout"][0],
                                                          t["sf_reshape"], per_sf))))
    if route_sf == "tma":
        sfs = ceil_div(k // per_sf, 16) * 16
        shape = abc(t, m, tokens_max, sfs, tile_m, 1, tile_k // per_sf, "B", batches, valid_k=sfs)
        fields.append(tmap("tmaSfB", nd(P["sf_b"], "ue8m0", *shape, False)))
    elif route_sf == "none":
        fields.append(tmap("tmaSfB", sf_map(P["sf_b"], *sf(padded, k, tile_n, tile_k, t["sf_layout"][1],
                                                              t["sf_reshape"], per_sf))))
    if t["tma_store"]:
        fields.append(tmap("tmaC", nd(P["c"], t["dtype_c"], *abc(t, m, padded, k, tile_m, tile_n, tile_k, "C", batches),
                                      False)))
    else:
        fields.append({"at": OFF["ptrC"], "param": P["c"]})
    ptr = lambda name, p: {"at": OFF[name], "param": P[p]}
    i32 = lambda name, val: {"at": OFF[name], "i32": val}
    fields += [
        ptr("ptrA", "a"), {"at": OFF["strideInBytesA"], "i64": k * BITS[t["dtype_a"]] // 8},
        ptr("ptrB", "b"), {"at": OFF["strideInBytesB"], "i64": k * BITS[t["dtype_b"]] // 8},
        i32("k", k), i32("nm", m), i32("tileStridePerBatch", m // tile_m),
        ptr("ptrSfA", "sf_a"), ptr("ptrSfB", "sf_b"),
        {"at": OFF["numTokens"], **{{int: "i32", str: "var"}.get(type(tokens), "expr"): tokens}},
        i32("numBatches", batches), i32("tpGrpSize", 1),
        ptr("ptrNumNonExitingCtas", "num_non_exiting"), ptr("ptrTotalNumPaddedTokens", "total_padded"),
        ptr("ptrCtaIdxXyToBatchIdx", "cta_batch"), ptr("ptrCtaIdxXyToMnLimit", "cta_limit"),
    ]
    if "sf_c" in P:
        fields.append(ptr("ptrSfC", "sf_c"))
    if routed:
        fields.append(ptr("ptrRouteMap", "route_map"))
    if t["fused_act"]:
        fields += [ptr("ptrGatedActAlpha", "alpha"), ptr("ptrGatedActBeta", "beta")]
    fields.sort(key=lambda f: f["at"])
    tiles = ceil_div(ceil_div(m, tile_m), t["cluster"][0]) * t["cluster"][0]
    launch = {
        **v.module, "entry": v.name, "block": [v.launch["block"], 1, 1], "grid": [tiles, ctas, t["split_k"]],
        "shared_mem": v.launch["shared_mem"], "params": [f"bytes<{SIZE}>"],
        "args": [{"pack": {"size": SIZE, "fields": fields}}],
        **({"cluster": v.launch["cluster"]} if v.launch["cluster"] != [1, 1, 1] else {}),
    }
    kind = {"a": "in buffer<u8>", "sf_a": "in buffer<u8>", "b": f"in buffer<{BUF[t['dtype_b']]}>", "sf_b": "in buffer<u8>",
            "c": f"out buffer<{BUF[t['dtype_c']]}>", "sf_c": "out buffer<u8>", "route_map": "in buffer<i32>",
            "alpha": "in buffer<f32>", "beta": "in buffer<f32>", "num_non_exiting": "in buffer<i32>",
            "total_padded": "in buffer<i32>", "cta_batch": "in buffer<i32>", "cta_limit": "in buffer<i32>"}
    return {"params": [kind[n] for n in params(v)], "impl": {"launches": [launch]}}
