#!/usr/bin/env python3
"""Permute model-oracle MXFP4 key pages and replay upstream paged dense scoring."""
import argparse
import json
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--artifacts',type=Path,required=True)
    p.add_argument('--page',type=int,default=64);p.add_argument('--check',action='store_true');a=p.parse_args()
    import torch
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]));from kern_manifest import SCHEMA_VERSION
    from paged_indexer_ops import definitions
    reports=[]
    for rows in (4,12,32):
        keys=512;page=a.page;pages=keys//page;stride=((page*68+511)//512)*512
        source=a.artifacts/f'indexer-io-{rows}';root=a.artifacts/f'paged-indexer-io-{page}-{rows}'
        if a.check:
            ref=torch.frombuffer(bytearray((source/'expected.bin').read_bytes()),dtype=torch.float32)
            got=torch.frombuffer(bytearray((root/'out.bin').read_bytes()),dtype=torch.float32)
            valid=torch.isfinite(ref);assert torch.equal(torch.isneginf(ref),torch.isneginf(got))
            diff=(got[valid]-ref[valid]).abs();rel=(diff.square().mean()/ref[valid].square().mean()).sqrt().item()
            r={'page':page,'rows':rows,'relative_rms':rel,'max_abs':diff.max().item()};print(json.dumps(r));reports.append(r);assert rel<0.01
            w=torch.frombuffer(bytearray((root/'projection.bin').read_bytes()),dtype=torch.bfloat16).float()/64
            dense=torch.frombuffer(bytearray((root/'weight_f32.bin').read_bytes()),dtype=torch.float32)
            sparse=torch.frombuffer(bytearray((root/'weight_bf16.bin').read_bytes()),dtype=torch.bfloat16)
            assert torch.equal(dense,w) and torch.equal(sparse,w.bfloat16())
            continue
        root.mkdir(exist_ok=True)
        def read(n,d):return torch.frombuffer(bytearray((source/f'{n}.bin').read_bytes()),dtype=d)
        kp=read('k',torch.uint8).reshape(pages,page*64);ks=read('ks',torch.uint8).reshape(pages,page*4)
        torch.manual_seed(page+rows);perm=torch.randperm(pages)
        cache=torch.zeros((pages,stride),dtype=torch.uint8);cache[:,:page*68]=torch.cat([kp[perm],ks[perm]],dim=1)
        table=perm.argsort().int()[None].repeat(rows,1)
        projection=(read('weights',torch.float32)*64).bfloat16().reshape(rows,32)
        for n,t in [('cache',cache),('table',table),('projection',projection)]:
            (root/f'{n}.bin').write_bytes(t.contiguous().view(torch.uint8).numpy().tobytes())
        for n in ('q','qs','weights','end'):(root/f'{n}.bin').write_bytes((source/f'{n}.bin').read_bytes())
        modules,ops=definitions(a.artifacts/'paged_indexer.cubin',rows='rows',rows_max=rows,kv_rows_max=keys,page_size=page,pages=pages,page_cols=pages,cache_state=False)
        specs=[('q','u8',[rows,32,64]),('qs','fp8e8m0',[rows,32,4]),('weights','f32',[rows,32]),('end','i32',[rows]),('table','i32',[rows,pages]),('cache','u8',[pages,stride]),('out','f32',[rows,keys]),('projection','bf16',[rows,32]),('weight_bf16','bf16',[rows,32]),('weight_f32','f32',[rows,32])]
        buffers={n:{'kind':'output' if n in ('out','weight_bf16','weight_f32') else 'input','dtype':d,'shape':sh} for n,d,sh in specs}
        calls=[{'op':'dsv41_paged_index_scores','args':[{'buf':n} for n in ('q','qs','weights','end','table','cache')]+[{'buf':'cache','offset':page*64},{'buf':'out'},{'var':'rows'}]},
               {'op':'dsv41_index_weights','args':[{'buf':n} for n in ('projection','weight_bf16','weight_f32')]+[{'var':'rows'}]}]
        m={'schema_version':SCHEMA_VERSION,'model':'paged-dense-indexer-probe','vars':{'rows':{'max':rows}},'states':{},'buffers':buffers,'modules':modules,'ops':ops,'programs':{'probe':{'calls':calls}}}
        (root/'manifest.json').write_text(json.dumps(m,indent=2))
    if a.check:(a.artifacts/f'paged-indexer-precision-{a.page}.json').write_text(json.dumps(reports,indent=2)+'\n')
if __name__=='__main__':main()
