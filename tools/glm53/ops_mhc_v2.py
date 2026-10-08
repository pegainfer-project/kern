"""Decode + MTP-round mHC candidates; existing ops_mhc.py is the rollback/oracle.

Raw-manifest integration: after wire_allreduce, before check_invariants,
resolve_modules and lower_wire, call fuse_manifest(m). See mhc_fusion.md.
For the MTP round call fuse_round_manifest(m) after ops_spec.extend_manifest
and before ops_moe_v2.fuse_round_manifest (moe_quant_a stays for the MoE pass).
No existing source or manifest is modified by importing this module.
"""
from .ops_common import S, T, a, scr, handwritten
from .ops_mhc import BF16X, BF16O, F32X, F32O

MODULE = 'glm53_mhc_v2'
SMEM = 37232
# Warp-specialized round boundary (glm53_mhc_boundary_f32_r32): non-aliased
# smem plan xs 32KB + w 8KB + out 8KB + header 368B (see glm53_mhc_build.py
# WS_BODY). >48KB: the runner opts in via CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED.
SMEM_WS = 49536
COL, ROW, NONE = 1, 2, 0


def ops():
    out = {}
    for ns in (64, 8):
        name = f'hc_big_fuse{ns}_q_v2'
        out[name] = {
            # Same ten semantic params as big_fuse, then FP8, SFA and layout.
            'params': [F32X, F32X, F32X, F32X, BF16X, F32O, F32O, BF16O, BF16X,
                       'i32', 'out buffer<fp8>', F32O, 'i32'],
            'impl': {'launches': [handwritten(
                MODULE, f'glm53_mhc_fuse{ns}_q',
                ['buffer'] * 9 + ['i32', 'buffer', 'buffer', 'i32'],
                [96, 1, 1], [S, 1, 1],
                [a(6), a(0), a(1), a(3), a(2), a(7), a(8), a(5), a(4), a(9),
                 a(10), a(11), a(12)], smem=SMEM)]},
        }
    for dtype in ('f32', 'bf16'):
        out[f'hc_boundary_{dtype}_v2'] = {
            # prev_comb, prev_residual, prev_post, hidden, fn, base, scale,
            # norm_weight, residual_out, comb_out, post_out, x_norm, q, sf, T, layout
            'params': [F32X, BF16X, F32X, BF16X, f'in buffer<{dtype}>',
                       F32X, F32X, BF16X, BF16O, F32O, F32O, BF16O,
                       'out buffer<fp8>', F32O, 'i32', 'i32'],
            'impl': {
                'scratch': {
                    'mul': {'dtype': 'f32', 'shape': [8, 16, 24]},
                    'sqr': {'dtype': 'f32', 'shape': [8, 16]},
                    'count': {'dtype': 'i32', 'shape': [16]},
                },
                'launches': [handwritten(
                    MODULE, f'glm53_mhc_boundary_{dtype}',
                    ['buffer'] * 8 + ['i32'] + ['buffer'] * 9 + ['i32'],
                    [256, 1, 1], [S, 12, 8],
                    [a(8), a(3), scr('mul'), a(4), a(0), a(2), a(1), scr('sqr'),
                     a(14), a(5), a(6), a(7), a(9), a(10), a(11), a(12), a(13),
                     scr('count'), a(15)], smem=SMEM)],
            },
        }
    # MTP round variants (rows<=32): pitch-32 SFA and 32-row scratch. The
    # standalone fuse8 is never needed (every round fuse8 joins a boundary).
    out['hc_big_fuse64_q_v2r'] = {
        # Same ten semantic params as the raw mtp32 big_fuse64 (the raw call
        # carries the {i32:32} partial pitch), then FP8, SFA and layout.
        'params': [F32X, F32X, F32X, F32X, BF16X, F32O, F32O, BF16O, BF16X,
                   'i32', 'out buffer<fp8>', F32O, 'i32'],
        'impl': {'launches': [handwritten(
            MODULE, 'glm53_mhc_fuse64_q_r32',
            ['buffer'] * 9 + ['i32', 'buffer', 'buffer', 'i32'],
            [96, 1, 1], [T, 1, 1],
            [a(6), a(0), a(1), a(3), a(2), a(7), a(8), a(5), a(4),
             a(9), a(10), a(11), a(12)], smem=SMEM)]},
    }
    out['hc_boundary_f32_v2r'] = {
        # prev_comb, prev_residual, prev_post, hidden, fn, base, scale,
        # norm_weight, residual_out, comb_out, post_out, x_norm, q, sf, T, layout
        'params': [F32X, BF16X, F32X, BF16X, 'in buffer<f32>',
                   F32X, F32X, BF16X, BF16O, F32O, F32O, BF16O,
                   'out buffer<fp8>', F32O, 'i32', 'i32'],
        'impl': {
            'scratch': {
                'mul': {'dtype': 'f32', 'shape': [8, 32, 24]},
                'sqr': {'dtype': 'f32', 'shape': [8, 32]},
                'count': {'dtype': 'i32', 'shape': [32]},
            },
            'launches': [handwritten(
                MODULE, 'glm53_mhc_boundary_f32_r32',
                ['buffer'] * 8 + ['i32'] + ['buffer'] * 9 + ['i32'],
                [256, 1, 1], [T, 12, 8],
                [a(8), a(3), scr('mul'), a(4), a(0), a(2), a(1), scr('sqr'),
                 a(14), a(5), a(6), a(7), a(9), a(10), a(11), a(12), a(13),
                 scr('count'), a(15)], smem=SMEM_WS)],
        },
    }
    # h3 hybrid (glm53_mhc_boundary_f32_r32_h3): identical grid/smem/args to
    # the r32 boundary; at num_tokens<=3 it runs the verbatim r32 body, at
    # num_tokens>3 each CTA caches its pre_fn z-slice once and covers 3 rows
    # (fn L2 traffic /3). Bitwise-equal to r32 at every M in 1..32; ~2.1us
    # faster at M=12, ~3.5us at M=24 (see glm53-hcboundary-20260928.md).
    out['hc_boundary_f32_v2r_h3'] = {
        'params': [F32X, BF16X, F32X, BF16X, 'in buffer<f32>',
                   F32X, F32X, BF16X, BF16O, F32O, F32O, BF16O,
                   'out buffer<fp8>', F32O, 'i32', 'i32'],
        'impl': {
            'scratch': {
                'mul': {'dtype': 'f32', 'shape': [8, 32, 24]},
                'sqr': {'dtype': 'f32', 'shape': [8, 32]},
                'count': {'dtype': 'i32', 'shape': [32]},
            },
            'launches': [handwritten(
                MODULE, 'glm53_mhc_boundary_f32_r32_h3',
                ['buffer'] * 8 + ['i32'] + ['buffer'] * 9 + ['i32'],
                [256, 1, 1], [T, 12, 8],
                [a(8), a(3), scr('mul'), a(4), a(0), a(2), a(1), scr('sqr'),
                 a(14), a(5), a(6), a(7), a(9), a(10), a(11), a(12), a(13),
                 scr('count'), a(15)], smem=SMEM_WS)],
        },
    }
    return out


def fuse_manifest(m, *, boundary=True, cross_layer=False, bf16=False, quant=True, skip_moe_quant=False):
    """Transform a RAW gen manifest, in place. No file I/O and no GPU work.

    Default joins 45 existing fma/fuse8 pairs and absorbs 56 input quants.
    cross_layer additionally joins 44 post/prenorm/fuse64 triples; it is
    tolerance-gated because DeepGEMM and FMA have different reduction trees.
    bf16 switches only boundary-used fn binds; the first prenorm stays FP32.
    Probes inside fused regions are rejected rather than silently dropped.
    Apply before other agents' MoE/DSA transforms, or use quant=False.
    """
    import copy
    if cross_layer and not boundary:
        raise ValueError('cross_layer requires boundary=True')
    if bf16 and not boundary:
        raise ValueError('bf16 storage requires boundary=True')
    assert m['vars']['tokens']['max'] == m['vars']['seqs']['max'] == 16
    assert m['programs']['decode']['batch'] == {'groups':16,'rows':1}
    calls = m['programs']['decode']['calls']
    if any(c['label'].startswith('probe.') for c in calls):
        raise ValueError('Use unprobed manifest; internal probe cuts change under fusion')
    if any(c['op'].endswith('_v2') for c in calls):
        raise ValueError('Apply mHC transform first; then MoE/DSA transforms')
    newops = ops()
    # Associate quant consumers with the latest x_norm producer. Stop if a8
    # or its scales would be overwritten before the intended consumer.
    qmap, drop = {}, set()
    producer = None
    # With MoE v2 the router kernel quantizes internally; leave moe_quant_a
    # calls in place for ops_moe_v2.fuse_manifest to consume.
    quant_ops = {'dsa_quant_qkv': COL, 'mlp_quant_a': COL}
    if not skip_moe_quant:
        quant_ops['moe_quant_a'] = ROW
    for i, c in enumerate(calls):
        if c['op'] in ('hc_big_fuse64', 'hc_big_fuse8'):
            producer = i
        if quant and c['op'] in quant_ops:
            if producer is None or producer in qmap:
                raise ValueError('Unexpected quant/x_norm producer order')
            destinations={a['buf'] for a in c['args'][1:3]}
            for between in calls[producer+1:i]:
                for ty,arg in zip(m['ops'][between['op']]['params'],between['args']):
                    if ty.startswith(('out ', 'inout ')) and arg.get('buf') in destinations:
                        raise ValueError('Quant relocation crosses output write: '+between['label'])
            qmap[producer] = [copy.deepcopy(c['args'][1]), copy.deepcopy(c['args'][2]),
                              {'i32': quant_ops[c['op']]}]
            drop.add(i)
    noquant = [{'buf': 'a8'}, {'buf': 'sfa'}, {'i32': NONE}]
    def qargs(i): return copy.deepcopy(qmap.get(i, noquant))
    used_bf16 = set()
    def boundary_call(label, prev, fn, fuse, i):
        # prev has semantic hc_post args: comb,R,post,hidden,Rout,T.
        f = fuse['args']
        if bf16:
            used_bf16.add(fn['buf'])
        return {'label': label, 'op': f'hc_boundary_{"bf16" if bf16 else "f32"}_v2',
                'args': copy.deepcopy(prev[:4] + [fn, f[3], f[2], f[8], prev[4], f[6],
                                                  f[5], f[7]] + qargs(i)[:2] +
                                      [prev[5], qargs(i)[2]])}
    result, i = [], 0
    while i < len(calls):
        c = calls[i]
        if i in drop:
            i += 1; continue
        if boundary and c['op'] == 'hc_fma':
            f = calls[i+1]
            assert f['op'] == 'hc_big_fuse8'
            p = c['args']
            assert p[9] == {'i32': 8}
            assert p[7] == f['args'][4]
            result.append(boundary_call(c['label']+'.boundary_v2',p[:4]+[p[7],p[8]],p[4],f,i+1))
            i += 2; continue
        if cross_layer and c['op'] == 'hc_post' and i+2 < len(calls) and calls[i+1]['op']=='hc_prenorm':
            pn, f = calls[i+1:i+3]
            assert f['op'] == 'hc_big_fuse64'
            assert c['args'][4] == pn['args'][0] == f['args'][4]
            result.append(boundary_call(c['label']+'.next_pre_v2',c['args'],pn['args'][1],f,i+2))
            i += 3; continue
        if c['op'] in ('hc_big_fuse64', 'hc_big_fuse8') and i in qmap:
            c = copy.deepcopy(c)
            c['op'] = c['op'] + '_q_v2'
            c['args'] += qargs(i)
        result.append(c)
        i += 1
    # A checkpoint's original fn tensor is BF16; no downcast of non-BF16 data
    # is hidden here. Test script separately checks exact widening equivalence.
    for name in used_bf16:
        buf = m['buffers'][name]
        assert buf['dtype']=='f32' and len(buf['bind'])==1
        binding = buf['bind'][0]
        assert binding['tensor'].endswith('_fn_f32'), binding
        binding['tensor'] = binding['tensor'].removesuffix('_f32')
        buf['dtype'] = 'bf16'
    m['programs']['decode']['calls'] = result
    used_buffers={a['buf'] for p in m['programs'].values() for c in p['calls'] for a in c['args'] if 'buf' in a}
    for n in ('part8_mul','part8_sqr'):
        if n not in used_buffers: m['buffers'].pop(n,None)
    used = {c['op'] for p in m['programs'].values() for c in p['calls']}
    m['ops'].update({n: v for n,v in newops.items() if n in used})
    for n in list(m['ops']):
        if n not in used:
            del m['ops'][n]
    return m


def fuse_round_manifest(m, *, boundary=True, quant=True, program='round_k2',
                        expect_boundary=45, expect_quant=14,
                        boundary_kernel='r32'):
    """Transform the MTP round program of a gen manifest, in place.

    Joins the 45 mtp32_hc_fma + mtp32_hc_big_fuse8 pairs into
    hc_boundary_f32_v2r (rows<=32, pitch-32 SFA) and absorbs the 14 adjacent
    x_norm input quants (11 mtp32_dsa_quant_qkv into standalone
    hc_big_fuse64_q_v2r, 3 mtp32_mlp_quant_a into the boundary). The repair
    tail's dsa_quant_qkv is NOT an mHC consumer (its x_norm comes from
    head_rms_norm 48 calls later); it passes through. mtp32_moe_quant_a is
    left for ops_moe_v2.fuse_round_manifest (same skip rule as decode).
    Order-tolerant w.r.t. the MoE/KDA round transforms, but production order
    is mHC first. Run after ops_spec.extend_manifest, before lower_wire.
    boundary_kernel='h3' selects the hybrid h3 entry (identical ABI, bitwise,
    faster at rows>3; see hc_boundary_f32_v2r_h3 in ops()).
    """
    import copy
    if quant and not boundary:
        raise ValueError('round quant absorption requires boundary=True '
                         '(no standalone fuse8 r32 entry)')
    if boundary_kernel not in ('r32', 'h3'):
        raise ValueError(f'unknown boundary_kernel: {boundary_kernel}')
    boundary_op = ('hc_boundary_f32_v2r_h3' if boundary_kernel == 'h3'
                   else 'hc_boundary_f32_v2r')
    assert m['vars']['tokens']['max'] == 32 and m['vars']['seqs']['max'] == 16
    calls = m['programs'][program]['calls']
    if any(c['label'].startswith('probe.') for c in calls):
        raise ValueError('Use unprobed manifest; internal probe cuts change under fusion')
    if any(c['op'].endswith('_v2r') for c in calls):
        raise ValueError('mHC round transform already applied')
    newops = ops()
    # Associate a quant consumer with the immediately preceding big_fuse.
    # Anything farther away (repair.quant_qkv) is not an mHC consumer.
    qmap, drop = {}, set()
    producer = None
    for i, c in enumerate(calls):
        if c['op'] in ('mtp32_hc_big_fuse64', 'mtp32_hc_big_fuse8'):
            producer = i
        if quant and c['op'] in ('mtp32_dsa_quant_qkv', 'mtp32_mlp_quant_a'):
            if producer is None or i != producer + 1:
                continue  # e.g. repair.quant_qkv: x_norm from head_rms_norm
            if c['args'][0] != calls[producer]['args'][7]:
                raise ValueError('Quant input is not the producer x_norm: ' + c['label'])
            destinations = {a['buf'] for a in c['args'][1:3]}
            for between in calls[producer + 1:i]:
                for ty, arg in zip(m['ops'][between['op']]['params'], between['args']):
                    if ty.startswith(('out ', 'inout ')) and arg.get('buf') in destinations:
                        raise ValueError('Quant relocation crosses output write: ' + between['label'])
            qmap[producer] = [copy.deepcopy(c['args'][1]), copy.deepcopy(c['args'][2]),
                              {'i32': COL}]
            drop.add(i)
    noquant = [{'buf': 'a8'}, {'buf': 'sfa'}, {'i32': NONE}]
    def qargs(i): return copy.deepcopy(qmap.get(i, noquant))

    def boundary_call(label, prev, fn, fuse, i):
        # prev has semantic mtp32 post args: comb,R,post,hidden,Rout,T.
        f = fuse['args']
        return {'label': label, 'op': boundary_op,
                'args': copy.deepcopy(prev[:4] + [fn, f[3], f[2], f[8], prev[4], f[6],
                                                  f[5], f[7]] + qargs(i)[:2] +
                                      [prev[5], qargs(i)[2]])}
    result, i, nboundary = [], 0, 0
    consumed = set()
    while i < len(calls):
        c = calls[i]
        if i in drop:
            i += 1; continue
        if boundary and c['op'] == 'mtp32_hc_fma':
            f = calls[i + 1]
            assert f['op'] == 'mtp32_hc_big_fuse8', 'round fma must pair with fuse8'
            p = c['args']
            assert len(p) == 10 and p[8] == {'var': 'tokens'} and p[9] == {'i32': 8}
            assert f['args'][9] == {'var': 'tokens'}
            assert p[7] == f['args'][4], 'R_b handoff fma->fuse8'
            result.append(boundary_call(c['label'] + '.boundary_v2r',
                                        p[:4] + [p[7], p[8]], p[4], f, i + 1))
            nboundary += 1
            consumed.add(i + 1)
            i += 2; continue
        if c['op'] == 'mtp32_hc_big_fuse64' and i in qmap:
            c = copy.deepcopy(c)
            c['op'] = 'hc_big_fuse64_q_v2r'
            c['args'] += qargs(i)
            consumed.add(i)
        result.append(c)
        i += 1
    if boundary:
        assert nboundary == expect_boundary, f'{nboundary} boundary joins, want {expect_boundary}'
    if quant:
        assert len(drop) == expect_quant, f'{len(drop)} absorbed quants, want {expect_quant}'
    assert set(qmap) <= consumed, 'absorbed quant not re-emitted by any fused op'
    m['programs'][program]['calls'] = result
    used_buffers = {a['buf'] for p in m['programs'].values() for c in p['calls']
                    for a in c['args'] if 'buf' in a}
    for n in ('part8_mul', 'part8_sqr'):
        if n not in used_buffers:
            m['buffers'].pop(n, None)
    used = {c['op'] for p in m['programs'].values() for c in p['calls']}
    m['ops'].update({n: v for n, v in newops.items() if n in used})
    for n in ('mtp32_hc_fma', 'mtp32_hc_big_fuse8',
              'mtp32_mlp_quant_a', 'mtp32_dsa_quant_qkv'):
        if n not in used:
            m['ops'].pop(n, None)
    return m
