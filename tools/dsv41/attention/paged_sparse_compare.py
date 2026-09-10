#!/usr/bin/env python3
"""Page-permute existing official sparse oracle inputs without GPU work."""
import argparse
import json
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--page',type=int,default=64);p.add_argument('--check',action='store_true');a=p.parse_args()
    import torch
    rows,keys=4,32768;page=a.page;pages=keys//page
    source=a.artifacts/'sparse-io';root=a.artifacts/f'paged-sparse-io-{page}'
    if a.check:
        ref=torch.frombuffer(bytearray((source/'expected.bin').read_bytes()),dtype=torch.bfloat16).float()
        got=torch.frombuffer(bytearray((root/'out.bin').read_bytes()),dtype=torch.bfloat16).float()
        valid=torch.isfinite(ref);assert torch.equal(torch.isneginf(ref),torch.isneginf(got))
        diff=(got[valid]-ref[valid]).abs();rel=(diff.square().mean()/ref[valid].square().mean()).sqrt().item()
        print(json.dumps({'page':page,'relative_rms':rel,'max_abs':diff.max().item()}));assert rel<0.025
        return
    root.mkdir(exist_ok=True)
    kp=torch.frombuffer(bytearray((source/'k.bin').read_bytes()),dtype=torch.uint8).reshape(pages,page*64)
    ks=torch.frombuffer(bytearray((source/'ks.bin').read_bytes()),dtype=torch.uint8).reshape(pages,page*4)
    torch.manual_seed(page);permutation=torch.randperm(pages)
    stride=((page*68+511)//512)*512
    cache=torch.zeros((pages,stride),dtype=torch.uint8)
    cache[:,:page*68]=torch.cat([kp[permutation],ks[permutation]],dim=1)
    table=permutation.argsort().int()[None].repeat(rows,1)
    requests=torch.tensor([0,1,2,2],dtype=torch.int32)
    for name,t in [('cache',cache),('table',table),('requests',requests)]:
        (root/f'{name}.bin').write_bytes(t.contiguous().view(torch.uint8).numpy().tobytes())
    for n in ('q','qs','weights','end','candidates'):(root/f'{n}.bin').write_bytes((source/f'{n}.bin').read_bytes())
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]));from kern_manifest import SCHEMA_VERSION
    from paged_sparse_ops import definitions
    modules,ops=definitions(a.artifacts/'sparse_indexer.cubin',rows='rows',rows_max=rows,page_size=page,page_cols=pages,cache_state=False)
    ops.pop('dsv41_sparse_positions')
    specs=[('q','u8',[rows,32,64]),('qs','u8',[rows,32,4]),('weights','bf16',[rows,32]),('end','i32',[rows]),('table','i32',[rows,pages]),('requests','i32',[rows]),('candidates','i32',[rows,2048]),('cache','u8',[pages,stride]),('out','bf16',[rows,16384])]
    buffers={n:{'kind':'output' if n=='out' else 'input','dtype':d,'shape':sh} for n,d,sh in specs}
    call={'op':'dsv41_paged_sparse_scores','args':[{'buf':n} for n,_,_ in specs]+[{'var':'rows'}]}
    m={'schema_version':SCHEMA_VERSION,'model':'paged-sparse-indexer-probe','vars':{'rows':{'max':rows}},'states':{},'buffers':buffers,'modules':modules,'ops':ops,'programs':{'probe':{'calls':[call]}}}
    (root/'manifest.json').write_text(json.dumps(m,indent=2))
if __name__=='__main__':main()
