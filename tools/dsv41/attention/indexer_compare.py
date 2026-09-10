#!/usr/bin/env python3
"""Prepare model-inference MXFP4 oracle + manifest, or check kern score output."""
import argparse
import json
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--inference',type=Path,required=True)
    p.add_argument('--artifacts',type=Path,required=True)
    p.add_argument('--check',action='store_true')
    a=p.parse_args()
    import torch
    sys.path.insert(0,str(a.inference))
    from kernel import fp4_act_quant
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from kern_manifest import SCHEMA_VERSION
    from indexer_ops import definitions
    results=[]
    for rows in (4,12,32):
        root=a.artifacts/f'indexer-io-{rows}'
        keys=512
        if a.check:
            expected=torch.frombuffer(bytearray((root/'expected.bin').read_bytes()),dtype=torch.float32)
            actual=torch.frombuffer(bytearray((root/'out.bin').read_bytes()),dtype=torch.float32)
            valid=torch.isfinite(expected)
            assert torch.equal(torch.isneginf(expected),torch.isneginf(actual))
            diff=(actual[valid]-expected[valid]).abs()
            rel=(diff.square().mean()/expected[valid].square().mean()).sqrt().item()
            result={'rows':rows,'relative_rms':rel,'max_abs':diff.max().item()}
            print(json.dumps(result),flush=True);results.append(result)
            assert rel<0.01,result
            continue
        root.mkdir(exist_ok=True)
        torch.manual_seed(432+rows)
        q=torch.randn(rows,32,128,device='cuda',dtype=torch.bfloat16)
        k=torch.randn(keys,128,device='cuda',dtype=torch.bfloat16)
        w=torch.randn(rows,32,device='cuda',dtype=torch.float32)*(128**-0.5*32**-0.5)
        qp,qs=fp4_act_quant(q);kp,ks=fp4_act_quant(k)
        qref=q.clone();kref=k.clone();fp4_act_quant(qref,inplace=True);fp4_act_quant(kref,inplace=True)
        # inference/model.py Indexer.forward: einsum -> ReLU -> head weights -> head sum.
        expected=(torch.einsum('shd,td->sht',qref,kref).relu()*w.unsqueeze(-1)).sum(dim=1)
        start=torch.zeros(rows,device='cuda',dtype=torch.int32)
        end=torch.arange(rows,device='cuda',dtype=torch.int32)*13+1
        expected.masked_fill_(torch.arange(keys,device='cuda')[None]>=end[:,None],-torch.inf)
        tensors={'q':qp.view(torch.uint8),'qs':qs.view(torch.uint8),'k':kp.view(torch.uint8),
                 'ks':ks.view(torch.uint8),'weights':w,'start':start,'end':end}
        for name,t in tensors.items():
            (root/f'{name}.bin').write_bytes(t.cpu().contiguous().view(torch.uint8).numpy().tobytes())
        (root/'expected.bin').write_bytes(expected.cpu().contiguous().view(torch.uint8).numpy().tobytes())
        modules,ops=definitions(a.artifacts/'indexer.cubin',rows='rows',rows_max=rows,kv_rows_max=keys)
        dtype={'torch.uint8':'u8','torch.float32':'f32','torch.int32':'i32'}
        buffers={name:{'kind':'input','dtype':dtype[str(t.dtype)],'shape':list(t.shape)} for name,t in tensors.items()}
        buffers['out']={'kind':'output','dtype':'f32','shape':[rows,keys]}
        call={'op':'dsv41_index_scores','args':[{'buf':n} for n in tensors]+[{'buf':'out'},{'var':'rows'},{'i32':keys}]}
        m={'schema_version':SCHEMA_VERSION,'model':'indexer-probe','vars':{'rows':{'max':rows}},'states':{},'buffers':buffers,'modules':modules,'ops':ops,'programs':{'probe':{'calls':[call]}}}
        (root/'manifest.json').write_text(json.dumps(m,indent=2))
    if a.check:(a.artifacts/'indexer-precision.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
