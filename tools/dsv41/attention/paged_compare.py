#!/usr/bin/env python3
"""Quantized paged attention vs the supplied model's quantization and attention."""
import argparse
import ctypes
import json
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--inference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--page',type=int,default=64)
    p.add_argument('--extra-page',type=int,default=64)
    p.add_argument('--window-only',action='store_true')
    a=p.parse_args()
    import torch
    sys.path.insert(0,str(a.inference))
    from kernel import act_quant,fp4_act_quant,sparse_attn
    fn=ctypes.CDLL(str(a.library)).dsv41_paged_decode
    fn.argtypes=[ctypes.c_void_p]*10+[ctypes.c_int]*7+[ctypes.c_float,ctypes.c_void_p]
    fn.restype=None
    torch.manual_seed(22141)
    results=[]
    for rows in (1,12,32):
        page=a.page;extra_page=a.extra_page;pages=5;extra_pages=3;topk=192 if a.window_only else 128;extra_topk=0 if a.window_only else 128
        q=torch.randn(rows,64,512,device='cuda',dtype=torch.bfloat16)
        original=torch.randn(pages*page,512,device='cuda',dtype=torch.bfloat16)
        compressed=torch.randn(extra_pages*extra_page,512,device='cuda',dtype=torch.bfloat16)
        q8,s8=act_quant(original,32,'ue8m0',torch.float8_e8m0fnu)
        q4,s4=fp4_act_quant(compressed,16,False,torch.float8_e4m3fn)
        cache=torch.cat([q8.view(torch.uint8).reshape(pages,page*512),s8.view(torch.uint8).reshape(pages,page*16)],dim=1)
        extra=torch.cat([q4.view(torch.uint8).reshape(extra_pages,extra_page*256),s4.view(torch.uint8).reshape(extra_pages,extra_page*32)],dim=1)
        dq=original.clone();act_quant(dq,32,'ue8m0',torch.float8_e8m0fnu,True)
        de=compressed.clone();fp4_act_quant(de,16,True,torch.float8_e4m3fn)
        ids=torch.stack([torch.randperm(pages*page,device='cuda')[:topk] for _ in range(rows)]).int()
        exids=torch.stack([torch.randperm(extra_pages*extra_page,device='cuda')[:extra_topk] for _ in range(rows)]).int()
        lengths=torch.full((rows,),topk,device='cuda',dtype=torch.int32)
        if a.window_only:exids=torch.full((rows,128),-1,device='cuda',dtype=torch.int32)
        exlengths=torch.full((rows,),extra_topk,device='cuda',dtype=torch.int32)
        if rows>1:
            ids[-1].fill_(-1);exids[-1].fill_(-1)
            ids[0,13:]=-1;exids[0,17:]=-1
        sink=torch.randn(64,device='cuda',dtype=torch.float32)
        out=torch.empty_like(q);lse=torch.empty((rows,64),device='cuda',dtype=torch.float32)
        tensors=[q,cache,ids,lengths,sink,out,lse,extra,exids,exlengths]
        fn(*[t.data_ptr() for t in tensors],rows,pages,extra_pages,page,topk,extra_topk,extra_page,512**-0.5,torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        allids=torch.cat([ids,torch.where(exids>=0,exids+pages*page,exids)],dim=1)
        ref=sparse_attn(q.unsqueeze(0),torch.cat([dq,de]).unsqueeze(0),sink,allids.unsqueeze(0),512**-0.5)[0]
        delta=(out.float()-ref.float()).abs()
        relative=(delta.square().mean()/ref.float().square().mean()).sqrt().item()
        result={'rows':rows,'max_abs':delta.max().item(),'relative_rms':relative}
        print(json.dumps(result),flush=True);results.append(result)
        assert torch.isfinite(out).all() and relative<0.01 and delta.max()<0.0625,result
        # Disabled compressed branch must ignore aliased nonzero length input.
        if a.window_only:exlengths.fill_(topk)
        case=a.output.parent/f'paged-io-{rows}';case.mkdir(exist_ok=True)
        names=['q','cache','ids','lengths','sink','out','lse','extra','extra_ids','extra_lengths']
        from paged_ops import definitions
        sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
        from kern_manifest import SCHEMA_VERSION
        cubin=a.library.with_name('libdsv41_paged_decode.2.sm_103a.cubin')
        modules,ops=definitions(cubin,rows='rows',rows_max=rows,pages=pages,extra_pages=extra_pages,
                               page_size=page,extra_page_size=extra_page,topk=topk,extra_topk=extra_topk,cache_state=False)
        buffers={name:{'dtype':{'torch.bfloat16':'bf16','torch.int32':'i32','torch.float32':'f32','torch.uint8':'u8'}[str(t.dtype)],'shape':list(t.shape),'kind':'output' if name=='out' else 'input'} for name,t in zip(names,tensors) if name!='lse'}
        order=['q','cache','ids','lengths','sink','out','extra','extra_ids','extra_lengths']
        call={'op':'dsv41_paged_attention','args':[{'buf':n} for n in order]+[{'var':'rows'}]}
        m={'schema_version':SCHEMA_VERSION,'model':'paged-attention-probe','vars':{'rows':{'max':rows}},'states':{},'buffers':buffers,'modules':modules,'ops':ops,'programs':{'probe':{'calls':[call]}}}
        (case/'manifest.json').write_text(json.dumps(m,indent=2))
        for name,t in zip(names,tensors):
            (case/f'{name}.bin').write_bytes(t.cpu().contiguous().view(torch.uint8).numpy().tobytes())
            if rows==1:print(f'{name} {t.data_ptr():#x}',flush=True)
    a.output.write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
