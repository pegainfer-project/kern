#!/usr/bin/env python3
"""Prepare/check sparse-indexer packed score oracle from supplied MXFP4 quant."""
import argparse
import json
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--inference',type=Path,required=True);p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--check',action='store_true');p.add_argument('--manifest-only',action='store_true');a=p.parse_args()
    import torch
    root=a.artifacts/'sparse-io';rows,keys=4,32768
    if a.manifest_only:
        from sparse_ops import definitions
        from select_ops import definitions as select_definitions
        m=json.loads((root/'manifest.json').read_text())
        m['modules'],m['ops']=definitions(a.artifacts/'sparse_indexer.cubin',rows='rows',rows_max=rows,kv_rows_max=keys)
        ms,os=select_definitions(a.artifacts/'libdsv41_select.2.sm_103a.cubin',rows='rows',rows_max=rows,width=16384,topk=512,dtype='bf16')
        m['modules'].update(ms);m['ops'].update(os)
        m['buffers']['select_end']={'kind':'input','dtype':'i32','shape':[rows]}
        m['buffers']['selected']={'kind':'workspace','dtype':'i32','shape':[rows,512]}
        m['buffers']['positions']={'kind':'output','dtype':'i32','shape':[rows,512]}
        import struct
        (root/'select_end.bin').write_bytes(struct.pack('4i',*[16384]*rows))
        m['programs']['probe']['calls']=m['programs']['probe']['calls'][:1]+[
            {'op':'dsv41_select','args':[{'buf':'out'},{'buf':'select_end'},{'buf':'selected'},{'var':'rows'}]},
            {'op':'dsv41_sparse_positions','args':[{'buf':'out'},{'buf':'selected'},{'buf':'candidates'},{'buf':'end'},{'buf':'positions'},{'var':'rows'}]}]
        (root/'manifest.json').write_text(json.dumps(m,indent=2));return
    if a.check:
        ref=torch.frombuffer(bytearray((root/'expected.bin').read_bytes()),dtype=torch.bfloat16).float()
        got=torch.frombuffer(bytearray((root/'out.bin').read_bytes()),dtype=torch.bfloat16).float()
        valid=torch.isfinite(ref)
        assert torch.equal(torch.isneginf(ref),torch.isneginf(got))
        diff=(got[valid]-ref[valid]).abs();rel=(diff.square().mean()/ref[valid].square().mean()).sqrt().item()
        print(json.dumps({'relative_rms':rel,'max_abs':diff.max().item()}));assert rel<0.025
        positions=torch.frombuffer(bytearray((root/'positions.bin').read_bytes()),dtype=torch.int32).reshape(rows,512)
        candidates=torch.frombuffer(bytearray((root/'candidates.bin').read_bytes()),dtype=torch.int32).reshape(rows,2048)
        scores=got.reshape(rows,16384)
        for row in range(rows):
            slots=scores[row].topk(512).indices.sort().values
            expected_pos=candidates[row,slots//8]*8+slots%8
            expected_pos[scores[row,slots]==-torch.inf]=-1
            # BF16 ties allow different but equally scoring token choices.
            actual=positions[row];valid_pos=actual[actual>=0]
            assert valid_pos.unique().numel()==valid_pos.numel()
            assert valid_pos.numel()==min(torch.isfinite(scores[row]).sum().item(),512)
            if valid_pos.numel():
                reverse=(candidates[row,:,None]*8+torch.arange(8)).flatten()
                reverse_map={int(v):i for i,v in enumerate(reverse.tolist()) if v>=0}
                actual_slots=torch.tensor([reverse_map[int(v)] for v in valid_pos])
                threshold=scores[row].topk(min(int(torch.isfinite(scores[row]).sum()),512)).values[-1]
                assert (scores[row,actual_slots]>=threshold).all()
        print('sparse scores + DeepSelect + absolute-position mapping passed')
        return
    root.mkdir(exist_ok=True)
    sys.path.insert(0,str(a.inference));from kernel import fp4_act_quant
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]));from kern_manifest import SCHEMA_VERSION
    from sparse_ops import definitions
    torch.manual_seed(4322048)
    q=torch.randn(rows,32,128,device='cuda',dtype=torch.bfloat16)
    key=torch.randn(keys,128,device='cuda',dtype=torch.bfloat16)
    qp,qs=fp4_act_quant(q);kp,ks=fp4_act_quant(key)
    qref=q.clone();kref=key.clone();fp4_act_quant(qref,inplace=True);fp4_act_quant(kref,inplace=True)
    weights=(torch.randn(rows,32,device='cuda')*(128**-0.5*32**-0.5)).bfloat16()
    starts=torch.zeros(rows,device='cuda',dtype=torch.int32)
    ends=torch.tensor([0,7,2049,keys],device='cuda',dtype=torch.int32)
    candidates=torch.full((rows,2048),-1,device='cuda',dtype=torch.int32)
    expected=torch.full((rows,16384),-torch.inf,device='cuda',dtype=torch.bfloat16)
    for row,n in enumerate(ends.tolist()):
        count=min((n+7)//8,2048)
        blocks=torch.randperm((n+7)//8,device='cuda')[:count].sort().values
        candidates[row,:count]=blocks.int()
        positions=(blocks[:,None]*8+torch.arange(8,device='cuda')).flatten()
        valid=positions<n
        selected=kref[positions[valid]]
        score=(torch.einsum('hd,td->ht',qref[row],selected).relu().float()*weights[row].float()[:,None]).sum(dim=0).bfloat16()
        expected[row,:count*8][valid]=score
    tensors={'q':qp.view(torch.uint8),'qs':qs.view(torch.uint8),'k':kp.view(torch.uint8),'ks':ks.view(torch.uint8),'weights':weights,'start':starts,'end':ends,'candidates':candidates}
    for name,t in list(tensors.items())+[('expected',expected)]:
        (root/f'{name}.bin').write_bytes(t.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    modules,ops=definitions(a.artifacts/'sparse_indexer.cubin',rows='rows',rows_max=rows,kv_rows_max=keys)
    dtype={'torch.uint8':'u8','torch.bfloat16':'bf16','torch.int32':'i32'}
    buffers={n:{'kind':'input','dtype':dtype[str(t.dtype)],'shape':list(t.shape)} for n,t in tensors.items()};buffers['out']={'kind':'output','dtype':'bf16','shape':[rows,16384]}
    call={'op':'dsv41_sparse_scores','args':[{'buf':n} for n in tensors]+[{'buf':'out'},{'var':'rows'},{'i32':keys}]}
    m={'schema_version':SCHEMA_VERSION,'model':'sparse-indexer-probe','vars':{'rows':{'max':rows}},'states':{},'buffers':buffers,'modules':modules,'ops':ops,'programs':{'probe':{'calls':[call]}}}
    (root/'manifest.json').write_text(json.dumps(m,indent=2))
if __name__=='__main__':main()
