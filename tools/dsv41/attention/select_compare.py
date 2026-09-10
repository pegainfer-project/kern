#!/usr/bin/env python3
"""DeepSelect vs inference topk semantics, allowing arbitrary valid tie choices."""
import argparse
import ctypes
import json
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--artifacts',type=Path,required=True);a=p.parse_args()
    import torch
    fn=ctypes.CDLL(str(a.artifacts/'libdsv41_select.so')).dsv41_select
    fn.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*4+[ctypes.c_void_p];fn.restype=None
    results=[]
    for bf16 in (False,True):
        for topk in (512,2048):
            rows=8;width=8192
            torch.manual_seed(topk)
            x=torch.randn(rows,width,device='cuda',dtype=torch.bfloat16 if bf16 else torch.float32)
            end=torch.tensor([0,13,topk-1,topk,topk+1,width,4097,5000],device='cuda',dtype=torch.int32)
            x[6].fill_(1)  # tied cutoff; model torch.topk does not define tie order either
            x[7,4999]=float('inf')  # forced most recent candidate block
            out=torch.empty(rows,topk,device='cuda',dtype=torch.int32)
            fn(x.data_ptr(),end.data_ptr(),out.data_ptr(),rows,width,topk,int(bf16),torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()
            for row,n in enumerate(end.tolist()):
                count=min(n,topk);selected=out[row,:count].long()
                assert (out[row,count:]==-1).all(),(bf16,topk,row,'padding')
                assert selected.unique().numel()==count
                assert ((selected>=0)&(selected<n)).all()
                assert torch.equal(selected,selected.sort().values)
                if count:
                    threshold=x[row,:n].float().topk(count).values[-1]
                    assert (x[row,selected].float()>=threshold).all(),(bf16,topk,row,'membership')
                    # Every strictly higher score must be included, not only tied cutoff values.
                    must=(x[row,:n].float()>threshold).nonzero()[:,0]
                    assert torch.isin(must,selected).all()
            kind='bf16' if bf16 else 'f32'
            result={'dtype':kind,'topk':topk,'rows':rows,'valid_membership':True,'sorted_indices':True,'short_fill':-1}
            print(json.dumps(result),flush=True);results.append(result)
            root=a.artifacts/f'select-{kind}-{topk}';root.mkdir(exist_ok=True)
            from select_ops import definitions
            sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
            from kern_manifest import SCHEMA_VERSION
            modules,ops=definitions(a.artifacts/'libdsv41_select.2.sm_103a.cubin',rows='rows',rows_max=rows,width=width,topk=topk,dtype=kind)
            buffers={'scores':{'kind':'input','dtype':kind,'shape':[rows,width]},'end':{'kind':'input','dtype':'i32','shape':[rows]},'indices':{'kind':'output','dtype':'i32','shape':[rows,topk]}}
            call={'op':'dsv41_select','args':[{'buf':n} for n in ['scores','end','indices']]+[{'var':'rows'}]}
            m={'schema_version':SCHEMA_VERSION,'model':'select-probe','vars':{'rows':{'max':rows}},'states':{},'buffers':buffers,'modules':modules,'ops':ops,'programs':{'probe':{'calls':[call]}}}
            (root/'manifest.json').write_text(json.dumps(m,indent=2))
            for name,t in [('scores',x),('end',end),('indices',out)]:
                (root/f'{name}.bin').write_bytes(t.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    (a.artifacts/'select-precision.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
