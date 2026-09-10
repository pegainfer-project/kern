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
    from model import apply_rotary_emb
    fn=ctypes.CDLL(str(a.library)).dsv41_fused_decode
    fn.argtypes=[ctypes.c_void_p]*10+[ctypes.c_int]*7+[ctypes.c_float,ctypes.c_void_p]+[ctypes.c_void_p]*3+[ctypes.c_int]
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
        q_original=q.clone()
        q=q.reshape(rows,64,32,16).transpose(1,2).contiguous().reshape(rows,64,512)
        positions=torch.arange(rows,device='cuda',dtype=torch.int32)
        angles=torch.randn(rows,32,device='cuda',dtype=torch.float32)
        rope=torch.cat([angles.cos(),angles.sin()],dim=1).contiguous()
        freqs=torch.polar(torch.ones_like(angles),angles)
        scale_rows=(rows+3)//4*4
        scales=torch.zeros((8,32,scale_rows),device='cuda',dtype=torch.int32)
        out=torch.empty(q.shape,device='cuda',dtype=torch.float8_e4m3fn);lse=torch.empty((rows,64),device='cuda',dtype=torch.float32)
        tensors=[q,cache,ids,lengths,sink,out,lse,extra,exids,exlengths]
        fn(*[t.data_ptr() for t in tensors],rows,pages,extra_pages,page,topk,extra_topk,extra_page,512**-0.5,torch.cuda.current_stream().cuda_stream,positions.data_ptr(),rope.data_ptr(),scales.data_ptr(),scale_rows)
        torch.cuda.synchronize()
        allids=torch.cat([ids,torch.where(exids>=0,exids+pages*page,exids)],dim=1)
        apply_rotary_emb(q_original.unsqueeze(0)[...,-64:],freqs)
        ref=sparse_attn(q_original.unsqueeze(0),torch.cat([dq,de]).unsqueeze(0),sink,allids.unsqueeze(0),512**-0.5)[0]
        apply_rotary_emb(ref.unsqueeze(0)[...,-64:],freqs,inverse=True)
        sf=scales.contiguous().view(torch.uint8).reshape(8,32,scale_rows,4).permute(2,0,1,3).reshape(scale_rows,1024)[:rows]
        scale=torch.pow(2.,sf.float()-127).repeat_interleave(32,dim=1)
        dequant=(out.float().reshape(rows,32768)*scale).reshape(rows,8,16,8,32).transpose(2,3).reshape(rows,64,512)
        delta=(dequant-ref.float()).abs()
        relative=(delta.square().mean()/ref.float().square().mean()).sqrt().item()
        result={'rows':rows,'max_abs':delta.max().item(),'relative_rms':relative}
        print(json.dumps(result),flush=True);results.append(result)
        assert torch.isfinite(out.float()).all() and relative<0.06 and delta.max()<0.25,result
        from fused_ops import definitions
        sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
        from kern_manifest import SCHEMA_VERSION
        if a.window_only:exlengths.fill_(topk)
        root=a.output.parent/f'fused-io-{rows}';root.mkdir(exist_ok=True)
        tensors=dict(zip(['q','cache','ids','lengths','sink','out','extra','extra_ids','extra_lengths','positions','rope','scales'],
                         [q,cache,ids,lengths,sink,out,extra,exids,exlengths,positions,rope,scales.view(torch.uint8)]))
        for name,t in tensors.items():
            (root/f'{name}.bin').write_bytes(t.cpu().contiguous().view(torch.uint8).numpy().tobytes())
        cubin=a.library.with_name('libdsv41_fused_decode.2.sm_103a.cubin')
        modules,ops=definitions(cubin,rows='rows',rows_max=rows,pages=pages,extra_pages=extra_pages,page_size=page,extra_page_size=extra_page,topk=topk,extra_topk=extra_topk,cache_state=False)
        dtype={'torch.bfloat16':'bf16','torch.float8_e4m3fn':'fp8e4m3','torch.int32':'i32','torch.float32':'f32','torch.uint8':'u8'}
        buffers={name:{'dtype':dtype[str(t.dtype)],'shape':list(t.shape),'kind':'output' if name in ('out','scales') else 'input'} for name,t in tensors.items()}
        order=['q','cache','ids','lengths','sink','out','extra','extra_ids','extra_lengths']
        args=[{'buf':n} for n in order]+[{'var':'rows'}]+[{'buf':n} for n in ['positions','rope','scales']]
        m={'schema_version':SCHEMA_VERSION,'model':'fused-attention-probe','vars':{'rows':{'max':rows}},'states':{},'buffers':buffers,'modules':modules,'ops':ops,'programs':{'probe':{'calls':[{'op':'dsv41_fused_attention','args':args}]}}}
        (root/'manifest.json').write_text(json.dumps(m,indent=2))
    a.output.write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
