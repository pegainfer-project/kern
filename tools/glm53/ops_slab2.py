"""MTP-round DSA indexer glue fusion (slab 2): k_norm+hadamard+act_quant+weights_proj.

Joins [dsa_k_norm, dsa_hadamard, dsa_act_quant, spec_kpool_update,
dsa_weights_proj] into ONE glm53_slab2 launch plus the untouched kpool call,
for every round chain that carries the full pattern (draft0 mtp16 + 11
verify mtp32 = 12 joins, -36 calls). The repair tail's k_norm (no
hadamard/act_quant/weights_proj siblings) and draft1 (no indexer query)
pass through. extern cuBLASLt calls (dsa_wq_b/dsa_wk/dsa_kpool_gate),
dsa_logits and dsa_topk are NOT touched: cuBLASLt BF16 GEMMs are
M-dependent (root-caused) and topk fusion is policy-frozen.

Bitwise by construction; see the glm53_slab2.cu header for the per-phase
provenance (k_norm/weights_proj verbatim glm53_dsa.cu; hadamard+quant
verbatim decode glm53_dsa_had_quant_v2, already gated against the pinned
sglang cubins the round launches). weights_proj relocates ahead of
spec_kpool_update: kpool reads ik/gs, never iq8/qs/w/x_norm (asserted per
join below).

Integration: gen.py calls fuse_round_manifest(m) in the MTP section (after
extend_manifest; order-tolerant w.r.t. the mHC/MoE/KDA round passes).
phases=('glue',) is the shipped transform. phases=('glue','gemm') is the
extension point for the cuBLASLt segment (wq_b/wk/kgate + logits), pending
the algo-lock making those GEMMs M-invariant bitwise; it raises
NotImplementedError until that lands.
"""
import copy
from .ops_common import S, T, a, handwritten

MODULE = 'glm53_slab2'
INOUT_BF16 = 'inout buffer<bf16>'
BF16X = 'in buffer<bf16>'
F32X = 'in buffer<f32>'
F32O = 'out buffer<f32>'
FP8O = 'out buffer<fp8e4m3>'


def ops():
    out = {}
    # grid [rows, 8]: CTA (t,c) covers heads 4c..4c+3, one warp each;
    # CTA (t,0) also runs the k_norm (glm53_slab2.cu header).
    for pfx, grid in (('mtp16', [S, 8, 1]), ('mtp32', [T, 8, 1])):
        out[f'{pfx}_slab2_index_glue'] = {
            # ik (inout), k_norm_w, k_norm_b, iq, iq8, qs, x_norm,
            # weights_proj, w. eps arrives as an f32 launch literal (1e-6),
            # exactly like the reference k_norm launch.
            'params': [INOUT_BF16, F32X, F32X, BF16X, FP8O, F32O, BF16X, F32X, F32O],
            'impl': {'launches': [handwritten(
                MODULE, 'glm53_slab2_dsa_index_glue',
                ['buffer'] * 3 + ['f32'] + ['buffer'] * 6,
                [128, 1, 1], grid,
                [a(0), a(1), a(2), {'f32': 1e-06}, a(3), a(4), a(5), a(6), a(7), a(8)])]},
        }
    return out


def fuse_round_manifest(m, *, phases=('glue',), program='round_k2',
                        expect16=1, expect32=11):
    """Transform the MTP round program of a gen manifest, in place. No file I/O."""
    if 'gemm' in phases:
        raise NotImplementedError(
            'cuBLASLt segment fold (wq_b/wk/kgate + logits) waits on '
            "the algo-lock M-invariance result")
    assert m['vars']['tokens']['max'] == 32 and m['vars']['seqs']['max'] == 16
    calls = m['programs'][program]['calls']
    if any(c['label'].startswith('probe.') for c in calls):
        raise ValueError('Use unprobed manifest; internal probe cuts change under fusion')
    if any(c['op'].endswith('_slab2_index_glue') for c in calls):
        raise ValueError('slab2 transform already applied')
    newops = ops()
    result, i, n16, n32 = [], 0, 0, 0
    while i < len(calls):
        c = calls[i]
        matched = False
        if 'glue' in phases and i + 4 < len(calls):
            for pfx in ('mtp16', 'mtp32'):
                if c['op'] != f'{pfx}_dsa_k_norm':
                    continue
                had, aq, kp, wp = calls[i + 1:i + 5]
                if (had['op'] != f'{pfx}_dsa_hadamard'
                        or aq['op'] != f'{pfx}_dsa_act_quant'
                        or kp['op'] != 'spec_kpool_update'
                        or wp['op'] != f'{pfx}_dsa_weights_proj'):
                    continue
                # --- bitwise premise + wproj-relocation legality, all asserted ---
                assert c['args'][0] == c['args'][3], 'k_norm not in-place on ik: ' + c['label']
                assert c['args'][0].get('buf') == 'ik', 'unexpected k_norm buffer: ' + c['label']
                assert had['args'][0].get('buf') == 'iq', 'unexpected hadamard input: ' + c['label']
                assert aq['args'][0] == had['args'][1], 'hadamard->act_quant iqh handoff: ' + c['label']
                assert wp['args'][2] == aq['args'][2], 'weights_proj scales != act_quant qs: ' + c['label']
                assert wp['args'][0].get('buf') == 'x_norm', 'unexpected weights_proj input: ' + c['label']
                kp_bufs = {ar.get('buf') for ar in kp['args']}
                for b in (had['args'][0]['buf'], aq['args'][1]['buf'], aq['args'][2]['buf'],
                          wp['args'][0]['buf'], wp['args'][3]['buf']):
                    assert b not in kp_bufs, f'kpool reads {b}; weights_proj relocation illegal'
                fused = {'label': c['label'] + '.glue2', 'op': f'{pfx}_slab2_index_glue',
                         'args': copy.deepcopy(
                             [c['args'][0], c['args'][1], c['args'][2],
                              had['args'][0], aq['args'][1], aq['args'][2],
                              wp['args'][0], wp['args'][1], wp['args'][3]])}
                result.append(fused)
                result.append(kp)
                if pfx == 'mtp16':
                    n16 += 1
                else:
                    n32 += 1
                i += 5
                matched = True
                break
        if not matched:
            result.append(c)
            i += 1
    assert n16 == expect16, f'{n16} mtp16 glue joins, want {expect16}'
    assert n32 == expect32, f'{n32} mtp32 glue joins, want {expect32}'
    m['programs'][program]['calls'] = result
    # iqh had exactly one producer (hadamard) and one consumer (act_quant),
    # both now inside the fused op; drop the dead buffer and ops.
    used_buffers = {ar['buf'] for p in m['programs'].values() for c in p['calls']
                    for ar in c['args'] if 'buf' in ar}
    if 'iqh' not in used_buffers:
        m['buffers'].pop('iqh', None)
    used = {c['op'] for p in m['programs'].values() for c in p['calls']}
    m['ops'].update({n: v for n, v in newops.items() if n in used})
    for n in ('mtp16_dsa_hadamard', 'mtp32_dsa_hadamard',
              'mtp16_dsa_act_quant', 'mtp32_dsa_act_quant',
              'mtp16_dsa_weights_proj', 'mtp32_dsa_weights_proj'):
        if n not in used:
            m['ops'].pop(n, None)
    return m
