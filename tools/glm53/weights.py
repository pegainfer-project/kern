"""GLM-5.3-Flash weight buffers + TP8 bind plan (mirrors sglang's TP8 shards).

sglang serves GLM-5.3-Flash as pure TP8 (verified against the capture):
- replicated per rank: embed, lm_head (we replicate; sglang shards+allgathers),
  norms, mHC hc_* tables, router gate + bias, DSA q_a+kv_a (fused GEMM N=2048),
  indexer wq_b/wk/weights_proj/kpool tables, KDA small projections
  (f_a/f_b/g_a/g_b/b), KDA conv weights
- sharded 1/8 per rank (bind rows/cols ranges by rank):
  - dense MLP (layers 0..2): gate/up rows (12288/8=1536), down cols (12288/8)
  - MoE experts: gate/up rows (2048/8=256), down cols (2048/8); scale_inv
    rows/cols (2048/128/8=2 blocks)
  - DSA: q_b rows (16384/8=2048), kv_b rows (32768/8=4096), o_proj cols
    (16384/8=2048); fp8 scale_inv sharded on the matching dim
  - KDA: q/k/v_proj rows (8192/8=1024 each), o_proj cols (8192/8=1024);
    conv1d weights rows (8192/8=1024: the conv state and delta-rule heads are
    per-rank, 8 heads of 64); f_b/g_b rows (8192/8=1024), dt_bias rows

The `bind` entries use the manifest's per-rank selection:
{"group": "ep", "tensors": [...]} picks a tensor per rank;
{"rows": {"group": "ep", "ranges": [[a, b], ...]}} picks a row range per rank.
"""

from . import constants as C

TP = C.EP  # 8; the group is named "ep" in the manifest either way


def rows_ranges(total, group=TP):
    step = total // group
    assert total % group == 0
    return [[r * step, (r + 1) * step] for r in range(group)]


def cols_ranges(total, group=TP):
    return rows_ranges(total, group)


def shard_rows(tensor, total):
    return {"tensor": tensor, "rows": {"group": "ep", "ranges": rows_ranges(total)}}


def shard_cols(tensor, total):
    return {"tensor": tensor, "cols": {"group": "ep", "ranges": cols_ranges(total)}}


def full(tensor):
    return {"tensor": tensor}


LM = "model.language_model."


def kda_layer_binds(i):
    """weight buffers of KDA layer i -> (name, dtype, shape, bind)"""
    p = f"{LM}layers.{i}.self_attn."
    q8 = C.KDA_QKV            # 8192
    r8 = q8 // TP             # 1024
    out = []
    # fused q|k|v|b|f_a|g_a projection [3336, 4096] per rank:
    # q/k/v rows-sharded (1024 each), b rows-sharded (64->8), f_a/g_a full (128)
    out.append((f"{p}qkvbfg_fused.weight", "bf16", [C.KDA_FUSED_QKVBFG_A, C.HIDDEN],
                [{"tensor": p + "q_proj.weight", "rows": {"group": "ep", "ranges": rows_ranges(q8)}},
                 {"tensor": p + "k_proj.weight", "rows": {"group": "ep", "ranges": rows_ranges(q8)}},
                 {"tensor": p + "v_proj.weight", "rows": {"group": "ep", "ranges": rows_ranges(q8)}},
                 {"tensor": p + "b_proj.weight", "rows": {"group": "ep", "ranges": rows_ranges(C.KDA_HEADS)}},
                 full(p + "f_a_proj.weight"),
                 full(p + "g_a_proj.weight")]))
    # merged q|k|v conv1d, f32, rows-sharded [1024, 4] per rank (derived)
    out.append((f"{p}conv1d_merged_f32", "f32", [3 * r8, C.KDA_CONV_K],
                [shard_rows(f"{p}conv1d_merged_f32", 3 * q8)]))
    # f_b/g_b sharded rows (8192 -> 1024)
    for t in ("f_b_proj.weight", "g_b_proj.weight"):
        out.append((f"{p}{t}", "bf16", [r8, 128], [shard_rows(p + t, q8)]))
    out.append((f"{p}A_log", "f32", [C.KDA_HEADS // TP], [shard_rows(p + "A_log", C.KDA_HEADS)]))
    out.append((f"{p}dt_bias", "f32", [r8], [shard_rows(p + "dt_bias", q8)]))
    out.append((f"{p}o_norm.weight", "bf16", [C.KDA_DIM], [full(p + "o_norm.weight")]))
    out.append((f"{p}o_proj.weight", "bf16", [C.HIDDEN, r8], [shard_cols(p + "o_proj.weight", q8)]))
    return out


def dsa_layer_binds(i):
    p = f"{LM}layers.{i}.self_attn."
    out = []
    # fused q_a|kv_a (replicated): [1536+512, 4096] fp8 + scales [12+4, 32]
    out.append((f"{p}qkv_a_proj.weight", "fp8e4m3", [C.Q_LORA + C.KV_LORA, C.HIDDEN],
                [full(p + "q_a_proj.weight"), full(p + "kv_a_proj_with_mqa.weight")]))
    out.append((f"{p}qkv_a_proj.scale_inv", "f32", [(C.Q_LORA + C.KV_LORA) // C.FP8_BLOCK, C.HIDDEN // C.FP8_BLOCK],
                [full(p + "q_a_proj.weight_scale_inv"), full(p + "kv_a_proj_with_mqa.weight_scale_inv")]))
    out.append((f"{p}q_a_layernorm.weight", "bf16", [C.Q_LORA], [full(p + "q_a_layernorm.weight")]))
    out.append((f"{p}kv_a_layernorm.weight", "bf16", [C.KV_LORA], [full(p + "kv_a_layernorm.weight")]))
    # q_b sharded rows 16384/8=2048 fp8 + scale rows 128/8=16
    out.append((f"{p}q_b_proj.weight", "fp8e4m3", [C.Q_B_DIM // TP, C.Q_LORA],
                [shard_rows(p + "q_b_proj.weight", C.Q_B_DIM)]))
    out.append((f"{p}q_b_proj.scale_inv", "f32", [C.Q_B_DIM // TP // C.FP8_BLOCK, C.Q_LORA // C.FP8_BLOCK],
                [shard_rows(p + "q_b_proj.weight_scale_inv", C.Q_B_DIM // C.FP8_BLOCK)]))
    # kv_b split for the absorbed bmms (derived.safetensors):
    # w_kc_t [64,512,256] (per-head k-half transposed, sglang layout),
    # w_vc [64,256,512] (v-half raw slices); 8 heads per rank.
    out.append((f"{p}kv_b.w_kc_t", "bf16", [C.DSA_HEADS // TP, C.KV_LORA, C.QK_HEAD],
                [shard_rows(f"{p}kv_b.w_kc_t", C.DSA_HEADS)]))
    out.append((f"{p}kv_b.w_vc", "bf16", [C.DSA_HEADS // TP, C.V_HEAD, C.KV_LORA],
                [shard_rows(f"{p}kv_b.w_vc", C.DSA_HEADS)]))
    # o_proj sharded cols 16384/8=2048 fp8 + scale cols 128/8=16
    out.append((f"{p}o_proj.weight", "fp8e4m3", [C.HIDDEN, C.Q_B_DIM // TP],
                [shard_cols(p + "o_proj.weight", C.Q_B_DIM)]))
    out.append((f"{p}o_proj.scale_inv", "f32", [C.HIDDEN // C.FP8_BLOCK, C.Q_B_DIM // TP // C.FP8_BLOCK],
                [shard_cols(p + "o_proj.weight_scale_inv", C.Q_B_DIM // C.FP8_BLOCK)]))
    ip = p + "indexer."
    out.append((f"{ip}wq_b.weight", "bf16", [C.IDX_QB_DIM, C.Q_LORA], [full(ip + "wq_b.weight")]))
    out.append((f"{ip}wk.weight", "bf16", [C.IDX_DIM, C.HIDDEN], [full(ip + "wk.weight")]))
    # indexer fp32 tables (derived.safetensors casts)
    out.append((f"{ip}k_norm_w_f32", "f32", [C.IDX_DIM], [full(ip + "k_norm_w_f32")]))
    out.append((f"{ip}k_norm_b_f32", "f32", [C.IDX_DIM], [full(ip + "k_norm_b_f32")]))
    out.append((f"{ip}weights_proj_f32", "f32", [C.IDX_HEADS, C.HIDDEN], [full(ip + "weights_proj_f32")]))
    out.append((f"{ip}ape_f32", "f32", [C.IDX_KPOOL, C.IDX_DIM], [full(ip + "ape_f32")]))
    out.append((f"{ip}kpool_gate.weight", "bf16", [C.IDX_DIM, C.HIDDEN], [full(ip + "index_kpool_compress_gate")]))
    return out


def mhc_binds(i):
    p = f"{LM}layers.{i}."
    out = []
    for kind in ("attn", "ffn"):
        # hc_*_fn cast to f32 at prep (derived.safetensors); base/scale are f32 in ckpt
        out.append((f"{p}hc_{kind}_fn_f32", "f32", [C.HC_ROWS, C.HC_STATE], [full(p + f"hc_{kind}_fn_f32")]))
        out.append((f"{p}hc_{kind}_base", "f32", [C.HC_ROWS], [full(p + f"hc_{kind}_base")]))
        out.append((f"{p}hc_{kind}_scale", "f32", [3], [full(p + f"hc_{kind}_scale")]))
    return out


def mlp_dense_binds(i):
    p = f"{LM}layers.{i}.mlp."
    r = C.FFN // TP   # 1536
    rb = r // C.FP8_BLOCK  # 12 blocks
    out = []
    # fused gate|up rows-sharded: [2*1536, 4096]
    out.append((f"{p}gate_up_proj.weight", "fp8e4m3", [2 * r, C.HIDDEN],
                [{"tensor": p + "gate_proj.weight", "rows": {"group": "ep", "ranges": rows_ranges(C.FFN)}},
                 {"tensor": p + "up_proj.weight", "rows": {"group": "ep", "ranges": rows_ranges(C.FFN)}}]))
    out.append((f"{p}gate_up_proj.scale_inv", "f32", [2 * rb, C.HIDDEN // C.FP8_BLOCK],
                [{"tensor": p + "gate_proj.weight_scale_inv", "rows": {"group": "ep", "ranges": rows_ranges(C.FFN // C.FP8_BLOCK)}},
                 {"tensor": p + "up_proj.weight_scale_inv", "rows": {"group": "ep", "ranges": rows_ranges(C.FFN // C.FP8_BLOCK)}}]))
    out.append((f"{p}down_proj.weight", "fp8e4m3", [C.HIDDEN, r],
                [shard_cols(p + "down_proj.weight", C.FFN)]))
    out.append((f"{p}down_proj.scale_inv", "f32", [C.HIDDEN // C.FP8_BLOCK, rb],
                [shard_cols(p + "down_proj.weight_scale_inv", C.FFN // C.FP8_BLOCK)]))
    return out


def moe_layer_binds(i):
    """one MoE layer: router (replicated) + per-rank expert slices.

    TP-mode fused layout (probe-verified): the shared expert is expert 288
    of the same fused_moe weight tensors, sharded like the routed ones.
    Per rank w13: [289, 512, 4096] fp8 = for each expert e, gate rows
    [256r,+256) then up rows [256r,+256); w2: [289, 4096, 256] fp8 = down
    cols [256r,+256). scale_inv blocks sharded on the matching dim
    (2048/128/8 = 2 blocks). Emitted as 289 two-segment binds (row ranges
    land sequentially).
    """
    p = f"{LM}layers.{i}.mlp."
    out = [(f"{p}gate.weight", "bf16", [C.N_EXPERTS, C.HIDDEN], [full(p + "gate.weight")]),
           (f"{p}gate.e_score_correction_bias", "f32", [C.N_EXPERTS], [full(p + "gate.e_score_correction_bias")])]
    r = C.MOE_INTER // TP            # 256
    rb = r // C.FP8_BLOCK            # 2 scale blocks
    w13, w13s, w2, w2s = [], [], [], []
    prefixes = [f"{p}experts.{e}." for e in range(C.N_EXPERTS)] + [p + "shared_experts."]
    for ep_ in prefixes:
        w13.append({"tensor": ep_ + "gate_proj.weight", "rows": {"group": "ep", "ranges": rows_ranges(C.MOE_INTER)}})
        w13.append({"tensor": ep_ + "up_proj.weight", "rows": {"group": "ep", "ranges": rows_ranges(C.MOE_INTER)}})
        w13s.append({"tensor": ep_ + "gate_proj.weight_scale_inv", "rows": {"group": "ep", "ranges": rows_ranges(C.MOE_INTER // C.FP8_BLOCK)}})
        w13s.append({"tensor": ep_ + "up_proj.weight_scale_inv", "rows": {"group": "ep", "ranges": rows_ranges(C.MOE_INTER // C.FP8_BLOCK)}})
        w2.append({"tensor": ep_ + "down_proj.weight", "cols": {"group": "ep", "ranges": cols_ranges(C.MOE_INTER)}})
        w2s.append({"tensor": ep_ + "down_proj.weight_scale_inv", "cols": {"group": "ep", "ranges": cols_ranges(C.MOE_INTER // C.FP8_BLOCK)}})
    n_fused = C.N_EXPERTS + 1        # 289 routed + shared
    out.append((f"{p}experts.w13", "fp8e4m3", [n_fused, 2 * r, C.HIDDEN], w13))
    out.append((f"{p}experts.w13_scale_inv", "f32", [n_fused, 2 * rb, C.HIDDEN // C.FP8_BLOCK], w13s))
    out.append((f"{p}experts.w2", "fp8e4m3", [n_fused, C.HIDDEN, r], w2))
    out.append((f"{p}experts.w2_scale_inv", "f32", [n_fused, C.HIDDEN // C.FP8_BLOCK, rb], w2s))
    return out


def global_binds():
    return [
        (f"{LM}embed_tokens.weight", "bf16", [C.VOCAB, C.HIDDEN], [full(f"{LM}embed_tokens.weight")]),
        # lm_head row-sharded [19360, 4096] per rank + nccl all-gather of logits
        ("lm_head.weight", "bf16", [C.VOCAB // TP, C.HIDDEN], [shard_rows("lm_head.weight", C.VOCAB)]),
        (f"{LM}norm.weight", "bf16", [C.HIDDEN], [full(f"{LM}norm.weight")]),
    ]


def all_binds(layers=None):
    """[(buffer_name, dtype, shape, bind)] for the given layers (default all 45)."""
    layers = C.DENSE_LAYERS + C.MOE_LAYERS if layers is None else layers
    out = list(global_binds())
    for i in layers:
        out += mhc_binds(i)
        out.append((f"{LM}layers.{i}.input_layernorm.weight", "bf16", [C.HIDDEN],
                    [full(f"{LM}layers.{i}.input_layernorm.weight")]))
        out.append((f"{LM}layers.{i}.post_attention_layernorm.weight", "bf16", [C.HIDDEN],
                    [full(f"{LM}layers.{i}.post_attention_layernorm.weight")]))
        out += (kda_layer_binds(i) if i in C.KDA_LAYERS else dsa_layer_binds(i))
        out += (mlp_dense_binds(i) if i in C.DENSE_LAYERS else moe_layer_binds(i))
    return out
