#!/usr/bin/env python3
"""End-to-end candidate scores -> DeepSelect -> filter vs model function."""
import argparse
import json
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--inference',type=Path,required=True);p.add_argument('--check',action='store_true');a=p.parse_args()
    import torch
    root=a.artifacts/'candidate-io'
    if a.check:
        expected=(root/'expected.bin').read_bytes();actual=(root/'out.bin').read_bytes()
        assert expected==actual,'candidate block membership differs from model inference'
        print('candidate block selection matches model inference exactly');return
    sys.path.insert(0,str(a.inference))
    from model import select_candidate_blocks
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from kern_manifest import SCHEMA_VERSION
    import candidate_ops,select_ops
    rows,width,blocks,k=4,32768,4096,2048
    torch.manual_seed(2048)
    logits=torch.randn(rows,width,device='cuda',dtype=torch.float32)
    ends=torch.tensor([0,7,511,width],device='cuda',dtype=torch.int32)
    logits.masked_fill_(torch.arange(width,device='cuda')[None]>=ends[:,None],-torch.inf)
    keep=select_candidate_blocks(logits,ends[:,None],k,8).reshape(rows,blocks,8).any(dim=-1)
    expected=torch.full((rows,k),-1,device='cuda',dtype=torch.int32)
    for row in range(rows):
        ix=keep[row].nonzero()[:,0];expected[row,:ix.numel()]=ix.int()
    root.mkdir(exist_ok=True)
    for name,t in [('logits',logits),('ends',ends),('expected',expected)]:
        (root/f'{name}.bin').write_bytes(t.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    modules,ops=candidate_ops.definitions(a.artifacts/'candidate.cubin',rows='rows',width=width,block_stride=blocks,topk=k)
    ms,os=select_ops.definitions(a.artifacts/'libdsv41_select.2.sm_103a.cubin',rows='rows',rows_max=rows,width=blocks,topk=k)
    modules.update(ms);ops.update(os)
    buffers={}
    for name,dtype,shape,kind in [('logits','f32',[rows,width],'input'),('ends','i32',[rows],'input'),('scores','f32',[rows,blocks],'workspace'),('block_ends','i32',[rows],'workspace'),('selected','i32',[rows,k],'workspace'),('out','i32',[rows,k],'output')]:
        buffers[name]={'dtype':dtype,'shape':shape,'kind':kind}
    def call(op,names):return {'op':op,'args':[{'buf':n} for n in names]+[{'var':'rows'}]}
    calls=[call('dsv41_candidate_scores',['logits','ends','scores','block_ends']),call('dsv41_select',['scores','block_ends','selected']),call('dsv41_selection_filter',['scores','selected','out'])]
    m={'schema_version':SCHEMA_VERSION,'model':'candidate-probe','vars':{'rows':{'max':rows}},'states':{},'buffers':buffers,'modules':modules,'ops':ops,'programs':{'probe':{'calls':calls}}}
    (root/'manifest.json').write_text(json.dumps(m,indent=2))
if __name__=='__main__':main()
