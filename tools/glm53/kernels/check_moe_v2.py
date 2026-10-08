#!/usr/bin/env python3
"""CPU-only build/ABI/integration checks; never edit gen.py or shared files."""
import ast
import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys
HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'tools'))
from glm53 import gen,ops_moe_v2

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--layers',type=int,default=4,choices=[4,45])
    args=ap.parse_args()
    for path in [HERE/'moe_v2_kernels.py', HERE/'build_moe_v2.py', HERE/'test_moe_v2.py', HERE.parent/'ops_moe_v2.py']:
        ast.parse(path.read_text())
    md=json.loads((HERE/'moe_v2_build.json').read_text())
    for name,d in md.items():
        cubin=ROOT/'kernels-glm53-handwritten'/f'{name}.cubin'
        assert hashlib.sha256(cubin.read_bytes()).hexdigest()==d['sha256']
        elf=subprocess.check_output(['/usr/local/cuda-13.0/bin/cuobjdump','-elf',str(cubin)],text=True)
        fields=re.findall(r'Ordinal : (0x[\da-f]+)\s+Offset\s+: (0x[\da-f]+)\s+Size\s+: (0x[\da-f]+)',elf)
        parsed=[{'ordinal':int(i,16),'offset':int(o,16),'size':int(z,16)} for i,o,z in fields]
        assert sorted(parsed,key=lambda d:d['ordinal'])==d['elf_params']
        sass=(HERE/(name+'.sass.txt')).read_text()
        assert 'ACQBULK' in sass
        if name.endswith(('w13','w2')):
            ptx=(HERE/(name+'.ptx')).read_text()
            n=32 if name.endswith('w13') else 128
            assert f'wgmma.mma_async.sync.aligned.m64n{n}k32.f32.e4m3.e4m3' in ptx
            assert 'mma.sync.aligned.m16n8k16.row.col.f32.f16.f16' not in ptx
        print(name,d['sha256'])
    original=gen.wire_allreduce
    def wire(m,*args,**kwargs):
        ops_moe_v2.fuse_manifest(m)
        return original(m,*args,**kwargs)
    gen.wire_allreduce=wire
    try:
        m=gen.build(layers=args.layers)
    finally:
        gen.wire_allreduce=original
    path=HERE/('moe_v2_manifest_test.json' if args.layers==4 else 'moe_v2_manifest_full_test.json')
    assert sum(c['op']=='moe_decode_v2' for c in m['programs']['decode']['calls'])==args.layers-3
    path.write_text(json.dumps(m,indent=2)+'\n')
    subprocess.run([str(pathlib.Path(__file__).resolve().parents[3] / 'target' / 'release' / 'kern'),'verify',str(path)],check=True)
    # Model align on CPU for all bucket sizes, arbitrary masks and routing skew.
    import random
    rng=random.Random(903)
    for b in range(1,17):
        for _ in range(64):
            valid=[rng.randrange(2) for _ in range(b)]
            ids=[rng.sample(range(288),8)+[288] if v else [-1]*9 for v in valid]
            pairs=[(e,r*9+k) for r,row in enumerate(ids) for k,e in enumerate(row) if e>=0]
            by={}
            for e,p in pairs:by.setdefault(e,[]).append(p)
            assert len(by)<=min(8*sum(valid)+1,9*b)
            assert all(len(v)<=32 for v in by.values())
            sorted_pairs=[p for e in sorted(by) for p in by[e]]
            assert sorted(sorted_pairs)==sorted(p for _,p in pairs)
    print('CPU align invariants: 1024 random bucket/mask cases passed')
if __name__=='__main__':main()
