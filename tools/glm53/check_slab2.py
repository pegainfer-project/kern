#!/usr/bin/env python3
"""CPU gate for DSA-indexer slab 2: production manifest transform + kern verify.
gen.py does not know ops_slab2 yet (owner integrates); chain the pass onto
the ops_mhc_v2 hook, which runs at the right point (after extend_manifest,
before MoE/KDA round passes and lower_wire)."""
import collections, json, subprocess, sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from glm53 import gen, ops_mhc_v2, ops_slab2

orig_round = ops_mhc_v2.fuse_round_manifest


def chained(mm, *a, **k):
    orig_round(mm, *a, **k)
    return ops_slab2.fuse_round_manifest(mm)


ops_mhc_v2.fuse_round_manifest = chained
try:
    m = gen.build(layers=45, allreduce='lamport', moe='v2', mhc='boundary',
                  dsa='v2a', kda='fused', mtp=True)
finally:
    ops_mhc_v2.fuse_round_manifest = orig_round

ref = json.load(open('/tmp/mtp_fused45.json'))
rk = m['programs']['round_k2']['calls']
rk_ref = ref['programs']['round_k2']['calls']
c = collections.Counter(x['op'] for x in rk)
print('round_k2 calls:', len(rk), ' reference:', len(rk_ref))
assert len(rk) == len(rk_ref) - 36, (len(rk), len(rk_ref))
assert c['mtp16_slab2_index_glue'] == 1, c['mtp16_slab2_index_glue']
assert c['mtp32_slab2_index_glue'] == 11, c['mtp32_slab2_index_glue']
for gone in ('mtp16_dsa_hadamard', 'mtp32_dsa_hadamard',
             'mtp16_dsa_act_quant', 'mtp32_dsa_act_quant',
             'mtp16_dsa_weights_proj', 'mtp32_dsa_weights_proj'):
    assert c[gone] == 0 and gone not in m['ops'], gone
assert c['mtp32_dsa_k_norm'] == 1
assert c['spec_kpool_update'] == 13
# iqh stays: decode (dsa v2a) still runs the unfused hadamard/act_quant chain.
assert 'iqh' in m['buffers']
g = next(x for x in rk if x['label'] == 'draft0.knorm.glue2')
assert g['op'] == 'mtp16_slab2_index_glue'
assert g['args'][0] == {'buf': 'ik'} and g['args'][3] == {'buf': 'iq'}
assert g['args'][4] == {'buf': 'iq8'} and g['args'][5] == {'buf': 'qs'}
assert g['args'][6] == {'buf': 'x_norm'} and g['args'][8] == {'buf': 'w'}
i_g = rk.index(g)
assert rk[i_g + 1]['op'] == 'spec_kpool_update' and rk[i_g + 1]['label'] == 'draft0.kpool'
v3 = next(x for x in rk if x['label'] == 'verify.l3.knorm.glue2')
assert v3['op'] == 'mtp32_slab2_index_glue'
assert v3['args'][1] == {'buf': 'layers.3.k_norm_w'}
assert v3['args'][7] == {'buf': 'layers.3.weights_proj'}
assert rk[rk.index(v3) + 1]['label'] == 'verify.l3.kpool'
l16 = m['ops']['mtp16_slab2_index_glue']['impl']['launches'][0]
l32 = m['ops']['mtp32_slab2_index_glue']['impl']['launches'][0]
assert l16['grid'] == ['seqs', 8, 1] and l32['grid'] == ['tokens', 8, 1]
assert l16['block'] == [128, 1, 1] and l32['args'][3] == {'f32': 1e-06}
d1 = collections.Counter(x['op'] for x in m['programs']['decode']['calls'])
d2 = collections.Counter(x['op'] for x in ref['programs']['decode']['calls'])
assert d1 == d2, 'decode program changed!'
r = subprocess.run([str(pathlib.Path(__file__).resolve().parents[2] / 'target' / 'release' / 'kern'), 'verify', '/dev/stdin'],
                   input=json.dumps(m), text=True, capture_output=True)
print('kern verify rc:', r.returncode)
if r.returncode:
    print(r.stderr[-4000:]); sys.exit(1)
json.dump(m, open('/tmp/mtp_fused45_slab2.json', 'w'))
print('PASS; slab2 manifest at /tmp/mtp_fused45_slab2.json')
