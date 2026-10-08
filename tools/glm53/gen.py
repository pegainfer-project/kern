#!/usr/bin/env python3
"""Generate the GLM-5.3-Flash TP8 decode manifest (GEN_SPEC.md).

    python3 tools/glm53/gen.py --layers 4 --out /tmp/glm53-4l.json
    python3 tools/glm53/gen.py --allreduce lamport --bundle kernels-glm53
    python3 tools/glm53/gen.py --allreduce nccl --out /tmp/glm53-45l-nccl.json

All derived weights already exist in the merged checkpoint. Generation is
CPU-only: it reads pinned cubins, but never loads weights or runs kernels.
Without --bundle, module sources name the existing cubins by absolute path.

Serving uses replicated TP rows and rank-local paged state, selected by
`topology.replicated_rows`. It requires a kern-serve build with that protocol.
"""

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import sys

# Keep generation read-only outside the explicitly requested outputs.
sys.dont_write_bytecode = True
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from kern_manifest import SCHEMA_VERSION, normalize, program  # noqa: E402
from glm53 import ops_common, ops_debug, ops_dsa, ops_dsa_v2, ops_head, ops_kda, ops_kda_v2, ops_mhc, ops_mhc_v2, ops_moe, ops_moe_v2, ops_slab1, ops_slab2, ops_slab3  # noqa: E402

LAYERS = 45
S_MAX = 16
EP = 8
HIDDEN = 4096
VOCAB = 154880
DSA_LAYERS = 11
KDA_LAYERS = 34
PAGE = 256
BT_COLS = 1024
ST_COLS = BT_COLS * PAGE
KV_LAYER_BYTES = 1024
IDX_LAYER_BYTES = 8448
IDX_PAGE_BYTES = 92928
CONV_LINE_BYTES = 18432
SSM_LINE_BYTES = 524288
TAIL_LINE_BYTES = 4096
LINE_TABLE_PITCH = S_MAX * 4
TV = {"var": "tokens"}
SV = {"var": "seqs"}


def b(name, off=0):
    return {"buf": name, "offset": off} if off else {"buf": name}


def st(name, off=0):
    assert off % 16 == 0
    return {"state": name, "offset": off} if off else {"state": name}


def i32(value):
    return {"i32": value}


def f32(value):
    return {"f32": value}


def wb(layer, name):
    return b(f"layers.{layer}.{name}")


def seg(tensor, axis=None, width=None, base=0):
    """A whole tensor, or this rank's contiguous row/column shard."""
    out = {"tensor": tensor}
    if axis is not None:
        assert axis in ("rows", "cols") and width > 0
        out[axis] = {"group": "ep", "ranges": [
            [base + r * width, base + (r + 1) * width] for r in range(EP)
        ]}
    return out


def io_buffers():
    return {
        "token_ids": {"dtype": "i64", "shape": ["tokens"], "kind": "input", "fill": "token",
                      "domain": {"index_into": "embed"}},
        "positions": {"dtype": "i32", "shape": ["tokens"], "kind": "input", "fill": "position",
                      "domain": {"min": 0}},
        "slot": {"dtype": "i32", "shape": ["tokens"], "kind": "input", "fill": "slot",
                 "domain": {"index_into": "kv", "stride": 1}},
        "seq_lens": {"dtype": "i32", "shape": ["seqs"], "kind": "input", "fill": "seq_len",
                     "domain": {"min": 1}},
        "cu_seqlens": {"dtype": "i32", "shape": [S_MAX + 1], "kind": "input", "fill": "cu_seqlens",
                       "domain": {"monotone": True}},
        "valid": {"dtype": "i32", "shape": ["tokens"], "kind": "input", "fill": "valid"},
        "block_table": {"dtype": "i32", "shape": ["seqs", BT_COLS], "kind": "input",
                        "domain": {"index_into": "kv", "stride": PAGE}},
        "kda_conv_lines": {"dtype": "i32", "shape": [KDA_LAYERS, "seqs"], "kind": "input",
                           "domain": {"index_into": "kda_conv", "stride": CONV_LINE_BYTES}},
        "kda_ssm_lines": {"dtype": "i32", "shape": [KDA_LAYERS, "seqs"], "kind": "input",
                          "domain": {"index_into": "kda_ssm", "stride": SSM_LINE_BYTES}},
        "idx_tail_lines": {"dtype": "i32", "shape": [DSA_LAYERS, "seqs"], "kind": "input",
                           "domain": {"index_into": "idx_tail", "stride": TAIL_LINE_BYTES}},
        "next_token": {"dtype": "i64", "shape": ["seqs"], "kind": "output", "fill": "tokens",
                       "domain": {"index_into": "embed"}},
    }


def workspace_buffers(probes=False):
    # Shared buffers are consumed before the next stream-ordered writer.
    # In particular, a8/sfa use the static 16-row DeepGEMM pitch, not S.
    layouts = [
        ("x", "bf16", ["tokens", HIDDEN]),
        ("R_a R_b", "bf16", ["tokens", 4, HIDDEN]),
        ("mix_post", "f32", ["tokens", 4]),
        ("mix_comb", "f32", ["tokens", 16]),
        ("part64_mul", "f32", [64, S_MAX, 24]),
        ("part64_sqr", "f32", [64, S_MAX]),
        ("part8_mul", "f32", [8, S_MAX, 24]),
        ("part8_sqr", "f32", [8, S_MAX]),
        ("x_norm", "bf16", ["tokens", HIDDEN]),
        ("a8", "fp8", ["tokens", HIDDEN]),
        ("sfa", "f32", [32, S_MAX]),
        ("sub_out", "bf16", ["tokens", HIDDEN]),
        ("kda_F", "bf16", ["tokens", 3336]),
        ("kda_forget kda_gproj", "bf16", ["tokens", 1024]),
        ("kda_qkvc", "bf16", ["tokens", 3072]),
        ("kda_o", "bf16", ["tokens", 1024]),
        ("kda_rstd", "f32", ["tokens", 8]),
        ("QKV", "bf16", ["tokens", 2048]),
        ("qa", "bf16", ["tokens", 1536]),
        ("knope", "bf16", ["tokens", 512]),
        ("q", "bf16", ["tokens", 2048]),
        ("iq iqh", "bf16", ["tokens", 4096]),
        ("iq8", "fp8", ["tokens", 4096]),
        ("qs", "f32", ["tokens", 32]),
        ("ik gs", "bf16", ["tokens", 128]),
        ("w", "f32", ["tokens", 32]),
        ("lg", "f32", ["tokens", BT_COLS * 64]),
        ("tk", "i32", ["tokens", 2051]),
        ("qv ao", "bf16", ["tokens", 4096]),
        ("av", "bf16", ["tokens", 2048]),
        ("moe_scores", "f32", ["tokens", 288]),
        ("moe_wts", "f32", ["tokens", 9]),
        ("moe_ids moe_packed", "i32", ["tokens", 9]),
        ("moe_sorted", "i32", [9216]),
        ("moe_expert_ids", "i32", [144]),
        ("moe_n_post", "i32", [1]),
        ("moe_as", "f32", ["tokens", 32]),
        ("moe_c1", "bf16", [144, 512]),
        ("moe_h1", "bf16", [144, 256]),
        ("moe_h1_8", "fp8", [144, 256]),
        ("moe_bs", "f32", [144, 2]),
        ("moe_c2", "bf16", [144, HIDDEN]),
        ("mlp_d1", "bf16", ["tokens", 3072]),
        ("mlp_h1", "bf16", ["tokens", 1536]),
        ("pool_lens pool_ctx dsa_lens", "i32", [S_MAX]),
        ("slot_table", "i32", [S_MAX, ST_COLS]),
        ("mqa_sched", "i32", [133, 2]),
        ("h h_norm", "bf16", ["tokens", HIDDEN]),
        ("lg_shard", "bf16", ["tokens", VOCAB // EP]),
        ("lg_full", "bf16", [EP, "tokens", VOCAB // EP]),
        ("lg_f32", "f32", ["tokens", VOCAB]),
    ]
    if probes:
        layouts.append(("dbg", "f32", [1024, 8]))
    return {name: {"dtype": dtype, "shape": list(shape), "kind": "workspace"}
            for names, dtype, shape in layouts for name in names.split()}


def weight_buffers(layers):
    buffers = {}

    def weight(name, shape, dtype, *binds):
        assert name not in buffers
        buffers[name] = {"dtype": dtype, "shape": shape, "kind": "weight", "bind": list(binds)}

    weight("embed", [VOCAB, HIDDEN], "bf16", seg("model.language_model.embed_tokens.weight"))
    weight("final_norm", [HIDDEN], "bf16", seg("model.language_model.norm.weight"))
    weight("lm_head", [VOCAB // EP, HIDDEN], "bf16", seg("lm_head.weight", "rows", VOCAB // EP))

    for layer in range(layers):
        prefix = f"model.language_model.layers.{layer}."
        sa, mlp = prefix + "self_attn.", prefix + "mlp."

        def w(name, shape, dtype, *binds):
            weight(f"layers.{layer}.{name}", shape, dtype, *binds)

        for stage in ("attn", "ffn"):
            w(f"hc_{stage}_fn", [24, 16384], "f32", seg(prefix + f"hc_{stage}_fn_f32"))
            w(f"hc_{stage}_scale", [3], "f32", seg(prefix + f"hc_{stage}_scale"))
            w(f"hc_{stage}_base", [24], "f32", seg(prefix + f"hc_{stage}_base"))
        w("input_ln", [HIDDEN], "bf16", seg(prefix + "input_layernorm.weight"))
        w("post_attn_ln", [HIDDEN], "bf16", seg(prefix + "post_attention_layernorm.weight"))

        if layer % 4 != 3:
            w("qkvbfg", [3336, HIDDEN], "bf16",
              *(seg(sa + f"{p}_proj.weight", "rows", 1024) for p in ("q", "k", "v")),
              seg(sa + "b_proj.weight", "rows", 8),
              seg(sa + "f_a_proj.weight"), seg(sa + "g_a_proj.weight"))
            for p in ("f", "g"):
                w(f"{p}_b", [1024, 128], "bf16", seg(sa + f"{p}_b_proj.weight", "rows", 1024))
            # The derived conv tensor is [q_all; k_all; v_all], not rank-major.
            w("conv", [3072, 4], "f32", *(seg(sa + "conv1d_merged_f32", "rows", 1024, base)
                                          for base in (0, 8192, 16384)))
            w("A_log", [8], "f32", seg(sa + "A_log", "cols", 8))
            w("dt_bias", [1024], "f32", seg(sa + "dt_bias", "cols", 1024))
            w("o_norm", [128], "bf16", seg(sa + "o_norm.weight"))
            w("o_proj", [HIDDEN, 1024], "bf16", seg(sa + "o_proj.weight", "cols", 1024))
        else:
            w("qkv_a", [2048, HIDDEN], "fp8",
              seg(sa + "q_a_proj.weight"), seg(sa + "kv_a_proj_with_mqa.weight"))
            w("qkv_a_sfb", [16, 32], "f32",
              seg(sa + "q_a_proj.weight_scale_inv"), seg(sa + "kv_a_proj_with_mqa.weight_scale_inv"))
            w("qa_norm", [1536], "bf16", seg(sa + "q_a_layernorm.weight"))
            w("kva_norm", [512], "bf16", seg(sa + "kv_a_layernorm.weight"))
            w("q_b", [2048, 1536], "fp8", seg(sa + "q_b_proj.weight", "rows", 2048))
            w("q_b_sfb", [16, 12], "f32", seg(sa + "q_b_proj.weight_scale_inv", "rows", 16))
            w("w_kc_t", [8, 512, 256], "bf16", seg(sa + "kv_b.w_kc_t", "rows", 8))
            w("w_vc", [8, 256, 512], "bf16", seg(sa + "kv_b.w_vc", "rows", 8))
            w("o_proj", [HIDDEN, 2048], "fp8", seg(sa + "o_proj.weight", "cols", 2048))
            w("o_proj_sfb", [32, 16], "f32", seg(sa + "o_proj.weight_scale_inv", "cols", 16))
            w("wq_b", [4096, 1536], "bf16", seg(sa + "indexer.wq_b.weight"))
            w("wk", [128, HIDDEN], "bf16", seg(sa + "indexer.wk.weight"))
            w("kpool_gate", [128, HIDDEN], "bf16", seg(sa + "indexer.index_kpool_compress_gate"))
            w("k_norm_w", [128], "f32", seg(sa + "indexer.k_norm_w_f32"))
            w("k_norm_b", [128], "f32", seg(sa + "indexer.k_norm_b_f32"))
            w("weights_proj", [32, HIDDEN], "f32", seg(sa + "indexer.weights_proj_f32"))
            w("ape", [4, 128], "f32", seg(sa + "indexer.ape_f32"))

        if layer < 3:
            w("gate_up", [3072, HIDDEN], "fp8",
              *(seg(mlp + f"{p}_proj.weight", "rows", 1536) for p in ("gate", "up")))
            w("gate_up_sfb", [24, 32], "f32",
              *(seg(mlp + f"{p}_proj.weight_scale_inv", "rows", 12) for p in ("gate", "up")))
            w("down", [HIDDEN, 1536], "fp8", seg(mlp + "down_proj.weight", "cols", 1536))
            w("down_sfb", [32, 12], "f32", seg(mlp + "down_proj.weight_scale_inv", "cols", 12))
        else:
            w("router", [288, HIDDEN], "bf16", seg(mlp + "gate.weight"))
            w("router_bias", [288], "f32", seg(mlp + "gate.e_score_correction_bias"))
            # TP mode: all 289 experts on every rank, intermediate sharded.
            experts = [mlp + f"experts.{e}." for e in range(288)] + [mlp + "shared_experts."]
            w("w13", [289, 512, HIDDEN], "fp8",
              *(seg(e + p + "_proj.weight", "rows", 256) for e in experts for p in ("gate", "up")))
            w("w13_sfb", [289, 4, 32], "f32",
              *(seg(e + p + "_proj.weight_scale_inv", "rows", 2) for e in experts for p in ("gate", "up")))
            w("w2", [289, HIDDEN, 256], "fp8",
              *(seg(e + "down_proj.weight", "cols", 256) for e in experts))
            w("w2_sfb", [289, 32, 2], "f32",
              *(seg(e + "down_proj.weight_scale_inv", "cols", 2) for e in experts))
    return buffers


def decode_calls(layers, allops, probes=False):
    calls, labels = [], set()

    def step(label, op, *args):
        assert label not in labels, f"duplicate call label: {label}"
        assert len(args) == len(allops[op]["params"]), (label, op)
        labels.add(label)
        calls.append({"label": label, "op": op, "args": list(args)})

    def ex(c):
        return {"expr": {"mul": ["tokens", c]}}

    def probe(tag, bufname, n, f32_=False):
        if not probes:
            return
        step(f"probe.{tag}", "debug_probe_f32" if f32_ else "debug_probe_bf16",
             b(bufname), b("dbg"), n, i32(tag), {"rank": "ep"})

    def prenorm(layer):
        step(f"l{layer}.prenorm", "hc_prenorm", b("R_a"), wb(layer, "hc_attn_fn"),
             b("part64_mul"), b("part64_sqr"), i32(S_MAX))
        step(f"l{layer}.fuse64", "hc_big_fuse64", b("part64_mul"), b("part64_sqr"),
             wb(layer, "hc_attn_scale"), wb(layer, "hc_attn_base"), b("R_a"),
             b("mix_post"), b("mix_comb"), b("x_norm"), wb(layer, "input_ln"), i32(S_MAX))
        probe(100 + layer, "x_norm", ex(HIDDEN))

    step("embed", "head_embed", b("token_ids"), b("embed"), b("x"), i32(HIDDEN))
    probe(1, "x", ex(HIDDEN))
    step("expand", "head_expand", b("x"), b("R_a"), i32(HIDDEN))
    probe(2, "R_a", ex(4 * HIDDEN))
    step("dsa_prep", "dsa_prep", b("seq_lens"), b("block_table"),
         b("pool_lens"), b("pool_ctx"), b("dsa_lens"), b("slot_table"), i32(BT_COLS), i32(ST_COLS))
    prenorm(0)

    dsa_ids = [layer for layer in range(layers) if layer % 4 == 3]
    for layer in range(layers):
        label = f"l{layer}."

        def w(name):
            return wb(layer, name)

        if layer % 4 != 3:
            k = layer - sum(d <= layer for d in dsa_ids)
            assert 0 <= k < KDA_LAYERS
            row = k * LINE_TABLE_PITCH
            step(label + "qkvbfg", "kda_qkvbfg", b("x_norm"), w("qkvbfg"), b("kda_F"),
                 SV, i32(3336), i32(HIDDEN))
            step(label + "fg_b", "kda_fg_b", b("kda_F"), w("f_b"), w("g_b"),
                 b("kda_forget"), b("kda_gproj"), SV)
            step(label + "conv", "kda_conv", b("kda_F"), w("conv"), st("kda_conv"),
                 b("kda_conv_lines", row), b("kda_qkvc"), SV)
            step(label + "delta", "kda_delta", w("A_log"), b("kda_forget"), w("dt_bias"),
                 b("kda_qkvc"), b("kda_F"), b("kda_o"), st("kda_ssm"), b("kda_ssm_lines", row),
                 b("cu_seqlens"), SV)
            step(label + "onorm", "kda_norm_gated", b("kda_o"), b("kda_gproj"), w("o_norm"),
                 b("kda_rstd"), {"expr": {"mul": ["seqs", 8]}})
            step(label + "o_proj", "kda_o_proj_ar", b("kda_o"), w("o_proj"), b("sub_out"), SV)
        else:
            i = dsa_ids.index(layer)
            assert 0 <= i < DSA_LAYERS
            kv_off, idx_off = i * KV_LAYER_BYTES, i * IDX_LAYER_BYTES
            assert idx_off + IDX_LAYER_BYTES <= IDX_PAGE_BYTES
            step(label + "quant_qkv", "dsa_quant_qkv", b("x_norm"), b("a8"), b("sfa"))
            step(label + "qkv_a", "dsa_qkv_a", b("a8"), w("qkv_a"), b("sfa"), w("qkv_a_sfb"), b("QKV"))
            step(label + "qa_norm", "head_rms_norm", b("QKV"), w("qa_norm"), b("qa"),
                 f32(1e-5), i32(1536), i32(2048), i32(1536))
            step(label + "kva_norm", "head_rms_norm", b("QKV", 1536 * 2), w("kva_norm"), b("knope"),
                 f32(1e-5), i32(512), i32(2048), i32(512))
            step(label + "quant_q", "dsa_quant_q", b("qa"), b("a8"), b("sfa"))
            step(label + "q_b", "dsa_q_b", b("a8"), w("q_b"), b("sfa"), w("q_b_sfb"), b("q"))
            step(label + "wq_b", "dsa_wq_b", b("qa"), w("wq_b"), b("iq"), SV)
            step(label + "wk", "dsa_wk", b("x_norm"), w("wk"), b("ik"), SV)
            step(label + "kgate", "dsa_kpool_gate", b("x_norm"), w("kpool_gate"), b("gs"), SV)
            step(label + "knorm", "dsa_k_norm", b("ik"), w("k_norm_w"), w("k_norm_b"), b("ik"))
            step(label + "had", "dsa_hadamard", b("iq"), b("iqh"))
            step(label + "aquant", "dsa_act_quant", b("iqh"), b("iq8"), b("qs"))
            step(label + "kpool", "dsa_kpool_update", st("idx", idx_off), st("idx_tail"),
                 b("ik"), b("gs"), w("ape"), b("block_table"), b("idx_tail_lines", i * LINE_TABLE_PITCH),
                 b("positions"), b("seq_lens"), b("valid"), i32(BT_COLS))
            step(label + "wproj", "dsa_weights_proj", b("x_norm"), w("weights_proj"), b("qs"), b("w"))
            step(label + "lmeta", "dsa_logits_meta", b("pool_ctx"), b("mqa_sched"))
            step(label + "logits", "dsa_logits", b("pool_ctx"), b("lg"), b("block_table"), b("mqa_sched"),
                 b("iq8"), st("idx", idx_off), st("idx", idx_off + 8192), b("w"))
            step(label + "topk", "dsa_topk", b("lg"), b("pool_lens"), b("slot_table"), b("seq_lens"), b("tk"))
            step(label + "clamp", "dsa_clamp", b("tk"))
            step(label + "w_kc", "dsa_w_kc", b("q"), w("w_kc_t"), b("qv"), SV)
            step(label + "kv_store", "dsa_kv_store", st("kv", kv_off), b("knope"), b("slot"))
            step(label + "fa3", "dsa_attn", b("qv"), st("kv", kv_off), b("tk"), b("dsa_lens"),
                 b("cu_seqlens"), b("ao"))
            step(label + "w_vc", "dsa_w_vc", b("ao"), w("w_vc"), b("av"), SV)
            step(label + "quant_o", "dsa_quant_o", b("av"), b("a8"), b("sfa"))
            step(label + "o_proj", "dsa_o_proj", b("a8"), w("o_proj"), b("sfa"), w("o_proj_sfb"), b("sub_out"))
            # GEN_SPEC uses lL.ar twice in DSA/MoE layers; labels must be unique.
            step(label + "attn_ar", "dsa_ar", b("sub_out"))

        probe(200 + layer, "sub_out", ex(HIDDEN))
        step(label + "fma", "hc_fma", b("mix_comb"), b("R_a"), b("mix_post"), b("sub_out"),
             w("hc_ffn_fn"), b("part8_mul"), b("part8_sqr"), b("R_b"), TV, i32(8))
        step(label + "fuse8", "hc_big_fuse8", b("part8_mul"), b("part8_sqr"),
             w("hc_ffn_scale"), w("hc_ffn_base"), b("R_b"), b("mix_post"), b("mix_comb"),
             b("x_norm"), w("post_attn_ln"), TV)
        probe(300 + layer, "x_norm", ex(HIDDEN))

        if layer < 3:
            step(label + "quant_a", "mlp_quant_a", b("x_norm"), b("a8"), b("sfa"))
            step(label + "gate_up", "mlp_fp8_gemm_gate_up", b("a8"), w("gate_up"), b("sfa"),
                 w("gate_up_sfb"), b("mlp_d1"))
            step(label + "silu", "mlp_silu", b("mlp_d1"), b("mlp_h1"))
            step(label + "quant_b", "mlp_quant_b", b("mlp_h1"), b("a8"), b("sfa"))
            step(label + "down", "mlp_fp8_gemm_down", b("a8"), w("down"), b("sfa"), w("down_sfb"), b("sub_out"))
            step(label + "ar", "mlp_ar", b("sub_out"))
        else:
            step(label + "router", "moe_router", b("x_norm"), w("router"), b("moe_scores"))
            probe(500 + layer, "moe_scores", ex(288), f32_=True)
            step(label + "rtopk", "moe_topk", b("moe_scores"), w("router_bias"),
                 b("moe_wts"), b("moe_ids"), b("moe_packed"))
            step(label + "align", "moe_align", b("moe_ids"), b("moe_sorted"), b("moe_expert_ids"), b("moe_n_post"))
            step(label + "quant_a", "moe_quant_a", b("x_norm"), b("a8"), b("moe_as"))
            step(label + "w13", "moe_w13", b("a8"), w("w13"), b("moe_c1"), b("moe_as"),
                 w("w13_sfb"), b("moe_wts"), b("moe_sorted"), b("moe_expert_ids"), b("moe_n_post"))
            step(label + "silu", "moe_silu", b("moe_c1"), b("moe_h1"))
            step(label + "quant_b", "moe_quant_b", b("moe_h1"), b("moe_h1_8"), b("moe_bs"))
            step(label + "w2", "moe_w2", b("moe_h1_8"), w("w2"), b("moe_c2"), b("moe_bs"),
                 w("w2_sfb"), b("moe_wts"), b("moe_sorted"), b("moe_expert_ids"), b("moe_n_post"))
            step(label + "reduce", "moe_sum_reduce", b("moe_c2"), b("sub_out"))
            step(label + "ar", "moe_ar", b("sub_out"))

        probe(400 + layer, "sub_out", ex(HIDDEN))
        step(label + "post", "hc_post", b("mix_comb"), b("R_b"), b("mix_post"), b("sub_out"), b("R_a"), TV)
        probe(10 + layer, "R_a", ex(4 * HIDDEN))
        # P0 keeps sglang's post -> prenorm -> fuse64 order, including truncation.
        if layer + 1 < layers:
            prenorm(layer + 1)

    step("contract", "head_contract", b("R_a"), b("h"), i32(HIDDEN), i32(4))
    probe(56, "h", ex(HIDDEN))
    step("final_norm", "head_rms_norm", b("h"), b("final_norm"), b("h_norm"),
         f32(1e-5), i32(HIDDEN), i32(HIDDEN), i32(HIDDEN))
    probe(57, "h_norm", ex(HIDDEN))
    step("lm_head", "head_lm_head", b("h_norm"), b("lm_head"), b("lg_shard"), b("lg_full"), SV)
    step("cast", "head_cast", b("lg_full"), b("lg_f32"))
    probe(58, "lg_f32", ex(VOCAB), f32_=True)
    step("argmax", "head_argmax", b("lg_f32"), b("next_token"), i32(VOCAB))
    return calls


def wire_allreduce(m, backend, pdl=False):
    """Append only communication params; keep producer/consumer sub_out bindings.

    This runs after decode_calls (whose params describe the existing model ABI)
    and before lower_wire (which lifts runtime rows into each op's calls).
    """
    if backend == "nccl":
        if pdl:
            raise ValueError("AR PDL requires --allreduce lamport")
        return
    if backend != "lamport":
        raise ValueError(f"unknown allreduce backend: {backend}")
    m["buffers"].update(ops_common.lamport_buffers())
    # Peer args are forbidden on any op with an extern, even if that extern
    # does not consume them. Split the existing two-launch KDA op into two
    # calls, not two extra launches. Preserve the cuBLASLt BF16O interface.
    kda = m["ops"]["kda_o_proj_ar"]
    assert len(kda["impl"]["launches"]) == 2
    assert kda["impl"]["launches"][-1]["entry"] == "extern:nccl_allreduce_bf16"
    kda["impl"]["launches"].pop()
    m["ops"]["kda_ar"] = {
        "params": ["inout buffer<bf16>"],
        "impl": {"launches": [ops_common.allreduce_bf16(
            ops_common.a(0), ops_common.expr(ops_common.mul("seqs", HIDDEN)))]},
    }
    for prog in m["programs"].values():
        expanded = []
        for call in prog["calls"]:
            expanded.append(call)
            if call["op"] == "kda_o_proj_ar":
                expanded.append({"label": call["label"] + ".ar", "op": "kda_ar",
                                 "args": [dict(call["args"][2])]})
        prog["calls"] = expanded
    replaced = set()
    for name, op in m["ops"].items():
        for index, launch in enumerate(op["impl"]["launches"]):
            if launch["entry"] != "extern:nccl_allreduce_bf16":
                continue
            assert name not in replaced
            assert launch["args"][0] == launch["args"][1]
            first = len(op["params"])
            op["params"].extend(["inout buffer<u8>", "in buffer<u64>", "out buffer<i32>"])
            op["impl"]["launches"][index] = ops_common.lamport_allreduce_bf16(
                launch["args"][0], ops_common.a(first), ops_common.a(first + 1),
                ops_common.a(first + 2), pdl=pdl)
            for prog in m["programs"].values():
                for call in prog["calls"]:
                    if call["op"] == name:
                        call["args"].extend([b("ar_sym"), b("ar_peers"), b("ar_error")])
            replaced.add(name)
    assert replaced == {"kda_ar", "dsa_ar", "moe_ar", "mlp_ar"}


def resolve_modules(allops):
    """Use existing artifacts, keeping each op's pinned bytes and launch ABI."""
    checked = {}
    for op in allops.values():
        for launch in op["impl"]["launches"]:
            if "cubin" not in launch:
                continue
            name, sha = launch["cubin"], launch["sha256"]
            if (name, sha) not in checked:
                path = pathlib.Path(name)
                candidates = ([path] if path.is_absolute() else
                              [d / path for d in (ops_common.DUMP_DIR, ops_common.HAND_DIR)])
                matches = {p.resolve() for p in candidates
                           if p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == sha}
                assert len(matches) == 1, f"{name}: expected one artifact with pinned sha256 {sha}"
                checked[name, sha] = str(next(iter(matches)))
            launch["cubin"] = checked[name, sha]


def lower_wire(m):
    """Lower the op modules' shorthand to schema 5, without changing ABI bytes.

    The modules use fp8 for e4m3, untyped launch pointers, launch-local
    var/expr scalars, and pointer offsets. Schema 5 requires fp8e4m3,
    directional pointers, and scalar/offset bindings on the call interface.
    Keep the semantic call plan above intact and expand only its wire form.
    """
    for buf in m["buffers"].values():
        if buf["dtype"] == "fp8":
            buf["dtype"] = "fp8e4m3"
    calls_by_op = {}
    for prog in m["programs"].values():
        for call in prog["calls"]:
            calls_by_op.setdefault(call["op"], []).append(call)

    for name, op in m["ops"].items():
        op["params"] = [p.replace("<fp8>", "<fp8e4m3>") for p in op["params"]]
        calls = calls_by_op.get(name, [])
        lifted = {}
        scratch_written = set()

        def lift(dtype, values):
            key = (dtype, json.dumps(values, sort_keys=True))
            if key not in lifted:
                index = len(op["params"])
                op["params"].append(dtype)
                for call, value in zip(calls, values):
                    call["args"].append(value)
                lifted[key] = index
            return {"param": lifted[key]}

        for launch in op["impl"]["launches"]:
            params = launch["params"]
            for j, arg in enumerate(launch["args"]):
                dtype = params[j].replace("<fp8>", "<fp8e4m3>")
                if dtype == "buffer":
                    if "param" in arg:
                        dtype = op["params"][arg["param"]]
                    elif "scratch" in arg:
                        scratch = arg["scratch"]
                        sdtype = op["impl"]["scratch"][scratch]["dtype"]
                        direction = "inout" if scratch in scratch_written else "out"
                        dtype = f"{direction} buffer<{sdtype}>"
                        scratch_written.add(scratch)
                    else:
                        raise ValueError(f"{name}: cannot type pointer {arg}")
                params[j] = dtype
                if "var" in arg or "expr" in arg:
                    launch["args"][j] = lift(dtype, [dict(arg) for _ in calls])
                elif arg.get("offset", 0):
                    index, offset = arg["param"], arg["offset"]
                    values = []
                    for call in calls:
                        value = dict(call["args"][index])
                        assert "buf" in value or "state" in value
                        value["offset"] = value.get("offset", 0) + offset
                        values.append(value)
                    launch["args"][j] = lift(op["params"][index], values)
                else:
                    arg.pop("offset", None)

    def signed_i32(value):
        # Preserve the recorded uint32 bit patterns in Rust's signed i32 JSON.
        if isinstance(value, dict):
            if "i32" in value and 2**31 <= value["i32"] < 2**32:
                value["i32"] -= 2**32
            for item in value.values():
                signed_i32(item)
        elif isinstance(value, list):
            for item in value:
                signed_i32(item)

    signed_i32(m)


def check_invariants(m, layers):
    assert S_MAX == m["vars"]["tokens"]["max"] == m["vars"]["seqs"]["max"] == 16
    assert m["programs"]["decode"]["batch"] == {"groups": S_MAX, "rows": 1}
    assert BT_COLS * PAGE >= 200_000
    assert IDX_LAYER_BYTES * DSA_LAYERS == IDX_PAGE_BYTES == 92928
    assert m["states"]["idx"]["bytes_per_token"] * PAGE == IDX_PAGE_BYTES
    for i in range(DSA_LAYERS):
        assert IDX_LAYER_BYTES * i + IDX_LAYER_BYTES <= IDX_PAGE_BYTES
    for name, buf in m["buffers"].items():
        domain = buf.get("domain", {})
        state = m["states"].get(domain.get("index_into"), {})
        if buf["kind"] == "input" and "bytes_per_token" in state:
            assert domain.get("stride", 1) in (1, PAGE), name
        if buf["kind"] == "input" and "bytes_per_seq" in state:
            assert state["bytes_per_seq"] % domain["stride"] == 0, name
    calls = m["programs"]["decode"]["calls"]
    for call in calls:
        for arg in call["args"]:
            if "state" in arg or "buf" in arg:
                assert arg.get("offset", 0) % 16 == 0, call["label"]
    assert len({c["label"] for c in calls}) == len(calls)
    if layers >= 4:
        assert {c["op"] for c in calls} == set(m["ops"]), "uncalled op"
    # Do not add PDL to other launches, even when they share a module.
    # glm53_moe_v2_w2 is allowed: its per-tile ready-flag protocol replaces
    # the whole-grid gdc_wait (kernels/moe_v2_kernels.py v5 note).
    pdl = [(name, launch["entry"]) for name, op in m["ops"].items()
           for launch in op["impl"]["launches"] if launch.get("pdl")
           and not launch["entry"].startswith("glm53_ar_lamport")
           and launch["entry"] != "glm53_moe_v2_w2"]
    import os as _os
    _pdl_max = int(_os.environ.get("GLM53_PDL_MAX", "1"))
    assert len(pdl) <= _pdl_max, f"pdl launches {len(pdl)} > GLM53_PDL_MAX={_pdl_max}"


def build(layers=LAYERS, probes=False, allreduce="lamport", ar_pdl=False, moe="v1", mhc="off", dsa="v1", kda="off", mtp=False, slab2="off", slab1="off", arpdl=False, slab3=False, mtp_steps=2, kda_round=None):
    if not 1 <= layers <= LAYERS:
        raise ValueError(f"layers must be in 1..{LAYERS}")
    if dsa == "v2a":
        allops = ops_dsa_v2.ops({"bt_cols": BT_COLS, "st_cols": ST_COLS})
    elif dsa == "v1":
        allops = ops_dsa.ops({"bt_cols": BT_COLS, "st_cols": ST_COLS})
    else:
        raise ValueError(f"unknown dsa version: {dsa}")
    for mod in (ops_kda, ops_mhc, ops_moe, ops_head):
        added = mod.ops()
        assert not (allops.keys() & added.keys())
        allops.update(added)
    # Generator-owned overrides: ops_moe.py and ops_dsa.py are also edited
    # independently during bring-up. This field is runtime num_valid_tokens,
    # not EM (the padded dispatch capacity); both GEMMs address 9 pairs/row.
    for name in ("moe_w13", "moe_w2"):
        launch = allops[name]["impl"]["launches"][0]
        assert launch["params"][12] == "i32"
        launch["args"][12] = {"expr": {"mul": ["seqs", ops_moe.TOPK9]}}
    if os.environ.get("GLM53_FA3_PDL", "1") == "0":
        for launch in allops["dsa_attn"]["impl"]["launches"]:
            launch["pdl"] = False
    if probes:
        added = ops_debug.ops()
        assert not (allops.keys() & added.keys())
        allops.update(added)
    assert len(allops) == 58 + (2 if probes else 0)
    buffers = io_buffers()
    buffers.update(workspace_buffers(probes))
    buffers.update(weight_buffers(layers))
    m = {
        "schema_version": SCHEMA_VERSION,
        "model": "GLM-5.3-Flash",
        "vars": {"tokens": {"max": S_MAX}, "seqs": {"max": S_MAX}},
        # Both groups span the same eight ranks. Keep ep for existing
        # shard binds and NCCL calls; tp selects the serving batch group.
        "topology": {"groups": {"ep": EP, "tp": EP}, "replicated_rows": True},
        "states": {
            "kv": {"bytes_per_token": DSA_LAYERS * KV_LAYER_BYTES},
            "idx": {"bytes_per_token": IDX_PAGE_BYTES // PAGE},
            "kda_conv": {"bytes_per_seq": KDA_LAYERS * CONV_LINE_BYTES},
            "kda_ssm": {"bytes_per_seq": KDA_LAYERS * SSM_LINE_BYTES},
            "idx_tail": {"bytes_per_seq": DSA_LAYERS * TAIL_LINE_BYTES},
        },
        "buffers": buffers,
        "modules": {},
        "ops": allops,
        "programs": {
            "load": program([], once=True),
            "decode": program(decode_calls(layers, allops, probes), groups=S_MAX, rows=1, graph=True),
        },
    }
    if mhc in ("boundary", "boundary-h3"):
        if probes:
            raise ValueError("--probes has no mHC-v2 counterpart; use --mhc off")
        ops_mhc_v2.fuse_manifest(m, boundary=True, cross_layer=False, bf16=False,
                                 quant=True, skip_moe_quant=(moe == "v2"))
    elif mhc != "off":
        raise ValueError(f"unknown mhc version: {mhc}")
    if moe == "v2":
        if probes:
            raise ValueError("--probes has no v2 MoE counterpart; use --moe v1")
        ops_moe_v2.fuse_manifest(m)
    elif moe != "v1":
        raise ValueError(f"unknown moe version: {moe}")
    wire_allreduce(m, allreduce, pdl=ar_pdl)
    check_invariants(m, layers)
    resolve_modules(allops)
    # After resolve_modules: absolute paths would be double-counted (kda_fusion.md sect.7).
    if kda in ("core", "fused"):
        if probes:
            raise ValueError("--probes has no KDA-v2 counterpart; use --kda off")
        ops_kda_v2.fuse_manifest(m, mode=kda, threads=512, direct=False)
        # Fused mode absorbs these into glm53_kda_v2; drop what no call uses.
        used_ops = {c["op"] for p_ in m["programs"].values() for c in p_["calls"]}
        for dead in ("kda_conv", "kda_delta", "kda_fg_b", "kda_norm_gated"):
            if dead not in used_ops:
                m["ops"].pop(dead, None)
        used_bufs = {a_["buf"] for p_ in m["programs"].values() for c in p_["calls"]
                     for a_ in c["args"] if "buf" in a_}
        for dead in ("kda_forget", "kda_gproj", "kda_qkvc", "kda_rstd"):
            if dead not in used_bufs:
                m["buffers"].pop(dead, None)
    elif kda != "off":
        raise ValueError(f"unknown kda version: {kda}")
    # --- MTP opt-in boundary (no changes to the default decode) ---
    if mtp:
        if probes:
            raise ValueError("--mtp requires no --probes")
        from glm53 import ops_spec
        ops_spec.extend_manifest(m, layers, allreduce=allreduce, dsa=dsa, steps=mtp_steps)
        rprog = f"round_k{mtp_steps}"
        if mhc == "boundary":
            ops_mhc_v2.fuse_round_manifest(m, program=rprog)
        elif mhc == "boundary-h3":
            ops_mhc_v2.fuse_round_manifest(m, boundary_kernel="h3", program=rprog)
        if moe == "v2":
            ops_moe_v2.fuse_round_manifest(m, program=rprog)
            ops_moe_v2.fuse_draft_manifest(m, program=rprog, expect=mtp_steps)
        if slab3:
            ops_slab3.fuse_slab3_manifest(m, program=rprog)
        # Round KDA v3 fusion is rows=3 ABI (glm53_kda_v3.cu); only steps=2 fuses.
        kr = kda_round if kda_round is not None else (kda if mtp_steps == 2 else "off")
        if mtp_steps != 2 and kda_round is None and kda in ("core", "fused") and kr == "off":
            print(f"gen: mtp_steps={mtp_steps}: round KDA fusion off "
                  f"(v3 kernels rows=3 ABI); decode fusion unchanged")
        if kr in ("core", "fused"):
            ops_kda_v2.fuse_round_manifest(m, mode=kr, threads=512, program=rprog)
        elif kr != "off":
            raise ValueError(f"unknown kda-round mode: {kr}")
        if slab2 == "glue":
            ops_slab2.fuse_round_manifest(m, program=rprog)
        # slab-1 round modes ride on the v3 fusion; arpdl is op-def only.
        if kr != "off":
            slab1_v3 = slab1
        elif mtp_steps != 2:
            slab1_v3 = "off"
            if slab1 != "off":
                print(f"gen: slab1 mode {slab1} needs the rows=3 v3 round fusion; "
                      f"skipped at mtp_steps={mtp_steps} (arpdl unaffected)")
        else:
            slab1_v3 = slab1  # ops_slab1 raises its v3 precondition, as before
        if slab1_v3 == "select":
            ops_slab1.fuse_round_manifest(m, mode="select", program=rprog)
        elif slab1_v3 in ("s2", "s4"):
            ops_slab1.fuse_round_manifest(m, mode="select", program=rprog)
            ops_slab1.fuse_round_manifest(m, mode=slab1_v3, program=rprog)
        elif slab1_v3 != "off":
            raise ValueError(f"unknown slab1 mode: {slab1_v3}")
        if arpdl:
            ops_slab1.fuse_round_manifest(m, mode="arpdl", program=rprog)
    # --- end MTP opt-in boundary ---
    if arpdl:
        import os as _os
        _os.environ.setdefault("GLM53_PDL_MAX", "64")
    # Round fusers can add new handwritten launches after the initial pass.
    # Resolve those too, before normalize records their portable module sources.
    resolve_modules(m["ops"])
    lower_wire(m)
    return normalize(m)


def bundle_modules(m, bundle, out):
    """Optional portable cubin bundle, with content-addressed artifact names."""
    bundle.mkdir(parents=True, exist_ok=True)
    for module in m["modules"].values():
        source = pathlib.Path(module["source"])
        target = bundle / f"{source.stem}-{module['sha256'][:12]}.cubin"
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != module["sha256"]:
                raise ValueError(f"refusing to overwrite a different artifact: {target}")
        else:
            shutil.copyfile(source, target)
        module["source"] = os.path.relpath(target.resolve(), out.resolve().parent)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", type=int, default=LAYERS, help="emit the first N layers (1..45)")
    ap.add_argument("--out", type=pathlib.Path,
                    help="default: examples/glm53-flash-ar.json (Lamport) or -nccl.json")
    ap.add_argument("--allreduce", choices=("lamport", "nccl"), default="lamport")
    ap.add_argument("--ar-pdl", action="store_true", help="opt-in Lamport PDL; baseline is off")
    ap.add_argument("--bundle", type=pathlib.Path, help="copy pinned cubins into a serving artifact directory")
    ap.add_argument("--moe", choices=("v1", "v2"), default="v1",
                    help="v2: three-launch fused MoE block (ops_moe_v2.fuse_manifest)")
    ap.add_argument("--mhc", choices=("off", "boundary", "boundary-h3"), default="off",
                    help="boundary: fuse hc_fma+hc_big_fuse8 pairs and absorb input quants (ops_mhc_v2); boundary-h3: same + h3 hybrid boundary kernel (bitwise, faster rows>3)")
    ap.add_argument("--dsa", choices=("v1", "v2a"), default="v1",
                    help="v2a: grouped absorb GEMV for DSA K/V projections (ops_dsa_v2)")
    ap.add_argument("--kda", choices=("off", "core", "fused"), default="off",
                    help="fused: one kernel for fg_b+conv+delta+norm (ops_kda_v2, 512 threads)")
    ap.add_argument("--probes", action="store_true",
                    help="insert debug value probes (embed, per-layer streams, head) printing from rank 0")
    # --- MTP opt-in CLI; never writes a champion manifest ---
    ap.add_argument("--mtp", action="store_true", help="experimental k=2 MTP graph; see docs/glm53/mtp.md")
    ap.add_argument("--mtp-steps", type=int, choices=(2,4,6,7,8), default=2,
                    help="draft steps for --mtp (round_k{steps}, rows=steps+1; ship: 6)")
    ap.add_argument("--kda-round", choices=("off", "core", "fused"), default=None,
                    help="round KDA fusion override; default follows --kda at steps=2, off at steps>2 (v3 rows=3 ABI)")
    ap.add_argument("--slab2", choices=("off", "glue"), default="off",
                    help="glue: fuse DSA indexer knorm+had+quant+wproj chain (ops_slab2)")
    ap.add_argument("--slab1", choices=("off", "select", "s2", "s4"), default="off",
                    help="select: v0.5 select_all fold, bitwise (ops_slab1); s2/s4: select + z2s stage-B fused_v3 swap (grid [seqs,8,2/4], ops_slab1)")
    ap.add_argument("--arpdl", action="store_true", help="multi-edge PDL for lamport ARs (ops_slab1 arpdl)")
    ap.add_argument("--slab3", action="store_true", help="absorb post-MoE-verify spec_ar into fused MoE (ops_slab3)")
    args = ap.parse_args()
    if args.mtp:
        if args.out is None:
            args.out = pathlib.Path("examples/glm53-flash-mtp.json")
        if (args.out.parent.resolve() == pathlib.Path("examples").resolve()
                and not args.out.name.startswith("glm53-flash-mtp")):
            ap.error("--mtp may write only examples/glm53-flash-mtp*.json in examples/")
    # --- end MTP opt-in CLI ---
    if not 1 <= args.layers <= LAYERS:
        ap.error(f"--layers must be in 1..{LAYERS}")
    if args.out is None:
        suffix = "ar" if args.allreduce == "lamport" else "nccl"
        args.out = pathlib.Path(f"examples/glm53-flash-{suffix}.json")
    if args.allreduce == "lamport" and args.out.resolve() == pathlib.Path(
            __file__).resolve().parents[2] / "examples/glm53-flash.json":
        ap.error("Lamport must not overwrite examples/glm53-flash.json")
    if args.ar_pdl and args.allreduce != "lamport":
        ap.error("--ar-pdl requires --allreduce lamport")
    m = build(args.layers, probes=args.probes, allreduce=args.allreduce, ar_pdl=args.ar_pdl, moe=args.moe, mhc=args.mhc, dsa=args.dsa, kda=args.kda, mtp=args.mtp, slab2=args.slab2, slab1=args.slab1, arpdl=args.arpdl, slab3=args.slab3, mtp_steps=args.mtp_steps, kda_round=args.kda_round)
    if args.bundle is not None:
        bundle_modules(m, args.bundle, args.out)
    args.out.write_text(json.dumps(m, indent=1) + "\n")
    print(f"wrote {args.out}: {args.layers} layers, {len(m['buffers'])} buffers, "
          f"{len(m['modules'])} modules, {len(m['ops'])} ops, "
          f"load={len(m['programs']['load']['calls'])} calls, "
          f"decode={len(m['programs']['decode']['calls'])} calls")


if __name__ == "__main__":
    main()
