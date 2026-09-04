"""FlashKDA's two span kernels (tools/flash-kda, docs/k3-kernel-abi.md K8) as one manifest op.

The ABI below is data, not derivation: it is what tools/kernel-capture lifted
from the vendored build's own launch (probe.cu), field for field — every cute
`TiledCopy` parameter is a 256-byte pack with the `CUtensorMap` at 0 and, for
the copies whose gmem layout has a runtime stride, one `int` (= 1) at 128.
Descriptor extents are the span's upper bound; the kernels bound the rows
they touch by the `T`/`tiles` scalars and zero the tail tile themselves.
"""
MODULE = "flash_kda_d128"
PREPARE = "_Z22_flash_kda_fwd_prepareIN4cute9TiledCopyINS0_9Copy_AtomIJNS0_11Copy_TraitsINS0_13SM90_TMA_LOADEJNS0_1CILi32768EEENS0_12AuxTmaParamsINS0_5tupleIJNS0_11ScaledBasisINS5_ILi1EEEJLi2EEEENS9_ISA_JLi1EEEENS9_IiJLi0EEEEEEERKNS0_6LayoutINS8_IJNS5_ILi128EEENS5_ILi16EEESA_EEENS8_IJSB_SC_NS9_ISA_JLi0EEEEEEEEERKNS0_7SwizzleILi0ELi4ELi3EEEEEEEEN7cutlass10bfloat16_tEEEENSF_INS8_IJSA_NS8_IJNS8_IJNS8_IJSG_SH_EEESA_EEEEEEEEENS8_IJNS5_ILi0EEENS8_IJNS8_IJNS8_IJSH_SA_EEES11_EEEEEEEEEEENS8_IJSA_SH_SG_EEEEES18_NS1_INS2_IJNS3_IS4_JNS5_ILi512EEENS7_INS8_IJSJ_EEERKNSF_INS8_IJNS5_ILi32EEEEEES1A_EESR_EEEEESV_EEENSF_INS8_IJSA_NS8_IJNS8_IJS1B_SA_EEEEEEEEENS8_IJS11_NS8_IJNS8_IJSA_S11_EEEEEEEEEEES1C_EES18_NS1_INS2_IJNS3_IS4_JNS5_ILi4096EEENS7_INS8_IJSC_SJ_EEERKNSF_INS8_IJSG_SA_EEES1S_EESR_EEEEEfEEENSF_INS8_IJSA_NS8_IJS1T_EEEEEES1O_EENS8_IJSA_SG_EEEEENS1_INS2_IJNS3_INS0_14SM90_TMA_STOREEJNS5_ILi2048EEENS7_ISK_RKNSF_INS8_IJNS5_ILi8EEESH_SA_EEESK_EESR_EEEEESV_EEENSF_INS8_IJSA_NS8_IJNS8_IJNS8_IJS27_SH_EEESH_EEEEEEEEENS8_IJS11_NS8_IJNS8_IJS12_SG_EEEEEEEEEEES17_EES2N_S2N_NS1_INS2_IJNS3_IS25_JS1R_S1X_EEEfEEES22_S23_EENS1_IS2E_NSF_INS8_IJSA_NS8_IJNS8_IJS2F_NS5_ILi2EEEEEEEEEEEES2L_EENS8_IJSA_SH_SH_EEEEES2X_Li16ELi128ELi256ELb0EEvT_T0_T1_T2_T3_T4_T5_T6_T7_T8_T9_fiiiPKliPKffPKi"
RECURRENCE = "_Z25_flash_kda_fwd_recurrenceIN4cute9TiledCopyINS0_9Copy_AtomIJNS0_11Copy_TraitsINS0_13SM90_TMA_LOADEJNS0_1CILi2048EEENS0_12AuxTmaParamsINS0_5tupleIJNS0_11ScaledBasisINS5_ILi1EEEJLi2EEEENS9_ISA_JLi1EEEENS9_IiJLi0EEEEEEERKNS0_6LayoutINS8_IJNS5_ILi8EEENS5_ILi16EEESA_EEENS8_IJSB_SC_NS9_ISA_JLi0EEEEEEEEERKNS0_7SwizzleILi0ELi4ELi3EEEEEEEEN7cutlass10bfloat16_tEEEENSF_INS8_IJSA_NS8_IJNS8_IJNS8_IJSG_SH_EEESH_EEEEEEEEENS8_IJNS5_ILi0EEENS8_IJNS8_IJNS8_IJSH_SA_EEENS5_ILi128EEEEEEEEEEEEEENS8_IJSA_SH_S13_EEEEENS1_INS2_IJNS3_IS4_JNS5_ILi512EEENS7_INS8_IJSJ_EEERKNSF_INS8_IJNS5_ILi32EEEEEES1B_EESR_EEEEESV_EEENSF_INS8_IJSA_NS8_IJNS8_IJS1C_SA_EEEEEEEEENS8_IJS11_NS8_IJNS8_IJSA_S11_EEEEEEEEEEES1D_EENS1_INS2_IJNS3_IS4_JS6_NS7_ISK_SN_SR_EEEEESV_EEES17_S18_EES1V_S1V_NS1_INS2_IJNS3_IS4_JNS5_ILi4096EEENS7_INS8_IJSC_SJ_EEERKNSF_INS8_IJS13_SA_EEES1X_EESR_EEEEEfEEENSF_INS8_IJSA_NS8_IJS1Y_EEEEEES1P_EENS8_IJSA_S13_EEEEENS1_IS1U_NSF_INS8_IJSA_NS8_IJNS8_IJSX_NS5_ILi2EEEEEEEEEEEES16_EENS8_IJSA_SH_SH_EEEEES2G_NS1_INS2_IJNS3_IS4_JNS5_ILi32768EEENS7_ISK_RKNSF_INS8_IJSG_S13_SA_EEESK_EERKNSO_ILi1ELi4ELi3EEEEEEEEfEEENSF_INS8_IJSA_NS8_IJNS8_IJNS8_IJSG_S13_EEESH_EEEEEEEEENS8_IJS11_NS8_IJNS8_IJS1Y_NS5_ILi1024EEEEEEEEEEEEEENS8_IJSA_S13_S13_EEEEENS1_INS2_IJNS3_INS0_14SM90_TMA_STOREEJS2H_S2P_EEEfEEES30_S31_EENS1_INS2_IJNS3_IS33_JS6_SS_EEESV_EEES17_S18_EELi16ELi128ELi3ELi2ELi192ELb1ELb1ELb1ELb0EEvT_T0_T1_T2_T3_T4_T5_T6_T7_T8_T9_PSV_iiiPKli"
CHUNK = 16
HEAD_DIM = 128
# scale multiplies q inside the kernel (after its own L2 norm), as K3's 128^-0.5
QSCALE = 0.08838834764831845
# gate_scale = lower_bound * log2(e), lower_bound = -5 (K3's LB)
GATE_SCALE = -5.0 * 1.4426950408889634


def tiled_copy(param, dtype, dims, strides, box, swizzle=0, dynamic=False):
    fields = [{"at": 0, "tensormap": {"param": param, "dtype": dtype, "dims": dims, "strides": strides, "box": box,
                                      "swizzle": swizzle, "l2_promotion": 128}}]
    if dynamic:
        fields.append({"at": 128, "i32": 1})
    return {"pack": {"size": 256, "fields": fields}}


def workspace_buffers(hl, span_max):
    """The six separated workspace arrays, sized for the span bound (upstream WS::k*)."""
    n = -(-span_max // CHUNK) * hl
    return {
        "span_ws_kd": {"dtype": "bf16", "shape": [n, CHUNK, HEAD_DIM], "kind": "workspace"},
        "span_ws_qd": {"dtype": "bf16", "shape": [n, CHUNK, HEAD_DIM], "kind": "workspace"},
        "span_ws_kr": {"dtype": "bf16", "shape": [n, CHUNK, HEAD_DIM], "kind": "workspace"},
        "span_ws_gt": {"dtype": "f32", "shape": [n, HEAD_DIM], "kind": "workspace"},
        "span_ws_inv": {"dtype": "bf16", "shape": [n, CHUNK, CHUNK], "kind": "workspace"},
        "span_ws_mqk": {"dtype": "bf16", "shape": [n, CHUNK, CHUNK], "kind": "workspace"},
    }


def op(hl, span_max, module, span="span", scale=QSCALE):
    """Interface: q | k | v | g (bf16 [rows, hl*128], rows 0..span) | beta (bf16 [hl, span]) | dt_bias (f32
    [hl*128]) | a_log (f32 [hl]) | state_in | state_out (f32 [hl][128][128]) | out (bf16 [rows, hl*128],
    rows 0..span) | ws_kd | ws_qd | ws_kr | ws_gt | ws_inv | ws_mqk | span."""
    n = -(-span_max // CHUNK) * hl
    T = {"param": 16}
    # `tiles` is a derived scalar: a 4-byte pack is the launch-arg form that can hold an expression.
    tiles_expr = {"ceil_div": [span, CHUNK]}
    tiles = {"pack": {"size": 4, "fields": [{"at": 0, "expr": tiles_expr}]}}
    rows = lambda param, box_rows, dynamic: tiled_copy(param, "bf16", [HEAD_DIM, span_max, hl], [hl * 256, 256],
                                                       [box_rows, CHUNK, 1], dynamic=dynamic)
    beta = tiled_copy(4, "bf16", [hl * span_max], [], [32])
    dt_bias = tiled_copy(5, "f32", [HEAD_DIM, hl], [512], [HEAD_DIM, 1])
    ws_rows = lambda param: tiled_copy(param, "bf16", [HEAD_DIM, CHUNK, n], [256, 4096], [8, CHUNK, 1])
    ws_gt = tiled_copy(13, "f32", [HEAD_DIM, n], [512], [HEAD_DIM, 1])
    ws_sq = lambda param: tiled_copy(param, "bf16", [CHUNK, CHUNK, n], [32, 512], [8, CHUNK, 1])
    state = lambda param: tiled_copy(param, "f32", [HEAD_DIM, HEAD_DIM, hl], [512, 65536], [8, HEAD_DIM, 1], swizzle=32)
    copies = ["bytes<256>"] * 11
    prepare = {
        **module, "entry": PREPARE, "block": [256, 1, 1], "grid": [tiles_expr, hl, 1], "shared_mem": 21248,
        "params": copies + ["f32", "i32", "i32", "i32", "i64", "bytes<4>", "in buffer<f32>", "f32", "i64"],
        "args": [rows(0, HEAD_DIM, True), rows(1, HEAD_DIM, True), beta, rows(3, HEAD_DIM, True), dt_bias,
                 ws_rows(10), ws_rows(11), ws_rows(12), ws_gt, ws_sq(14), ws_sq(15),
                 {"f32": scale}, T, {"i32": hl}, {"i32": 1}, {"i64": 0}, tiles, {"param": 6}, {"f32": GATE_SCALE},
                 {"i64": 0}],
    }
    recurrence = {
        **module, "entry": RECURRENCE, "block": [192, 1, 1], "grid": [1, hl, 1], "shared_mem": 98432,
        "params": copies + ["out buffer<bf16>", "i32", "i32", "i32", "i64", "bytes<4>"],
        # v in, the workspace, the state in and out, then `out` twice: a TMA store
        # descriptor and the plain pointer. q is consumed by prepare (ws_qd).
        "args": [rows(2, 8, True), beta, ws_rows(10), ws_rows(11), ws_rows(12), ws_gt, ws_sq(14), ws_sq(15),
                 state(7), state(8), rows(9, 8, True),
                 {"param": 9}, T, {"i32": hl}, {"i32": 1}, {"i64": 0}, tiles],
    }
    return {
        "params": ["in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>",
                   "in buffer<f32>", "in buffer<f32>", "in buffer<f32>", "out buffer<f32>", "out buffer<bf16>",
                   "out buffer<bf16>", "out buffer<bf16>", "out buffer<bf16>", "out buffer<f32>", "out buffer<bf16>",
                   "out buffer<bf16>", "i32"],
        "impl": {"launches": [prepare, recurrence]},
    }
