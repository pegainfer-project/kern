#!/usr/bin/env python3
"""CPU gate for mHC round slab 1a: production manifest transform + kern verify.
gen.py applies ops_mhc_v2.fuse_round_manifest itself (mhc=='boundary'); the
reference build no-ops the pass to isolate exactly the round transform."""
import collections, json, subprocess, sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from glm53 import gen, ops_mhc_v2

m = gen.build(layers=45, allreduce='lamport', moe='v2', mhc='boundary',
              dsa='v2a', kda='fused', mtp=True)

orig_round = ops_mhc_v2.fuse_round_manifest
ops_mhc_v2.fuse_round_manifest = lambda *a, **k: None
try:
    m2 = gen.build(layers=45, allreduce='lamport', moe='v2', mhc='boundary',
                   dsa='v2a', kda='fused', mtp=True)
finally:
    ops_mhc_v2.fuse_round_manifest = orig_round
rk = m['programs']['round_k2']['calls']
c = collections.Counter(x['op'] for x in rk)
print('round_k2 calls:', len(rk))
print({k: v for k, v in sorted(c.items()) if 'hc' in k or 'quant' in k})
assert c['hc_boundary_f32_v2r'] == 45
assert c['hc_big_fuse64_q_v2r'] == 11
assert c['mtp32_hc_big_fuse64'] == 34
assert 'part8_mul' not in m['buffers'] and 'part8_sqr' not in m['buffers']
assert any(x['label'] == 'repair.quant_qkv' and x['op'] == 'mtp32_dsa_quant_qkv' for x in rk)
b0 = next(x for x in rk if x['label'].startswith('verify.l0.fma'))
assert b0['op'] == 'hc_boundary_f32_v2r'
assert b0['args'][12] == {'buf': 'a8'} and b0['args'][15] == {'i32': 1}, b0['args']
b3 = next(x for x in rk if x['label'].startswith('verify.l3.fma'))
assert b3['args'][15] == {'i32': 0}, b3['args']
f3 = next(x for x in rk if x['label'] == 'verify.l3.fuse64')
assert f3['op'] == 'hc_big_fuse64_q_v2r' and f3['args'][9] == {'buf': 'a8'} and f3['args'][10] == {'buf': 'sfa'}
# decode untouched: same op multiset as the reference build without the hook
d1 = collections.Counter(x['op'] for x in m['programs']['decode']['calls'])
d2 = collections.Counter(x['op'] for x in m2['programs']['decode']['calls'])
assert d1 == d2, 'decode program changed!'
r2k = m2['programs']['round_k2']['calls']
print('reference round_k2 calls:', len(r2k))
assert len(rk) == len(r2k) - 59, (len(rk), len(r2k))  # -45 boundary -14 quant
r = subprocess.run([str(pathlib.Path(__file__).resolve().parents[2] / 'target' / 'release' / 'kern'), 'verify', '/dev/stdin'],
                   input=json.dumps(m), text=True, capture_output=True)
print('kern verify rc:', r.returncode)
if r.returncode:
    print(r.stderr[-4000:]); sys.exit(1)
json.dump(m, open('/tmp/mtp_fused5_mhc.json', 'w'))
print('PASS; fused5-mhc manifest at /tmp/mtp_fused5_mhc.json')
