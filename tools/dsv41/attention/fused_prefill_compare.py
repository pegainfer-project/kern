#!/usr/bin/env python3
"""Fused prefill vs supplied sparse_attn and RoPE functions, including FP8 output."""
import argparse
import ctypes
import json
import sys
from pathlib import Path


def write_manifest(root,artifacts,rows,keys=257,topk=256):
    from fused_prefill_ops import definitions
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from kern_manifest import SCHEMA_VERSION
    modules,ops=definitions(artifacts/'libdsv41_fused_prefill.2.sm_103a.cubin',rows='rows',rows_max=rows,kv_rows_max=keys,topk=topk)
    buffers={}
    specs=[('q','bf16',[rows,64,512]),('kv','bf16',[keys,512]),('ids','i32',[rows,topk]),('sink','f32',[64]),('lengths','i32',[rows]),('out','fp8e4m3',[rows,64,512]),('positions','i32',[rows]),('rope','f32',[rows,64]),('scales','u8',[8,32,(rows+3)//4*4,4])]
    for name,dtype,shape in specs:buffers[name]={'dtype':dtype,'shape':shape,'kind':'output' if name in ('out','scales') else 'input'}
    args=[{'buf':n} for n in ['q','kv','ids','sink','lengths','out']]+[{'var':'rows'},{'i32':keys}]+[{'buf':n} for n in ['positions','rope','scales']]
    m={'schema_version':SCHEMA_VERSION,'model':'fused-prefill-probe','vars':{'rows':{'max':rows}},'states':{},'buffers':buffers,'modules':modules,'ops':ops,'programs':{'probe':{'calls':[{'op':'dsv41_fused_prefill','args':args}]}}}
    (root/'manifest.json').write_text(json.dumps(m,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--inference',type=Path,required=True);p.add_argument('--manifest-only',action='store_true');a=p.parse_args()
    if a.manifest_only:
        for rows in (1,17,32):write_manifest(a.artifacts/f'fused-prefill-io-{rows}',a.artifacts,rows)
        return
    import torch
    sys.path.insert(0,str(a.inference));from kernel import sparse_attn;from model import apply_rotary_emb
    fn=ctypes.CDLL(str(a.artifacts/'libdsv41_fused_prefill.so')).dsv41_fused_prefill
    fn.argtypes=[ctypes.c_void_p]*8+[ctypes.c_int]*3+[ctypes.c_float,ctypes.c_void_p]+[ctypes.c_void_p]*3+[ctypes.c_int]
    fn.restype=None
    results=[]
    for rows in (1,17,32):
        keys,topk=257,256
        torch.manual_seed(221+rows)
        q=torch.randn(rows,64,512,device='cuda',dtype=torch.bfloat16)
        kv=torch.randn(keys,512,device='cuda',dtype=torch.bfloat16)
        qp=q.reshape(rows,64,32,16).transpose(1,2).contiguous().reshape(rows,64,512)
        sink=torch.randn(64,device='cuda',dtype=torch.float32)
        ids=torch.stack([torch.randperm(keys,device='cuda')[:topk] for _ in range(rows)]).int()
        if rows>1:ids[-1].fill_(-1)
        lengths=torch.full((rows,),topk,device='cuda',dtype=torch.int32)
        pos=torch.arange(rows,device='cuda',dtype=torch.int32)
        angle=torch.randn(rows,32,device='cuda');rope=torch.cat([angle.cos(),angle.sin()],dim=1).contiguous()
        freqs=torch.polar(torch.ones_like(angle),angle)
        out=torch.empty_like(q,dtype=torch.float8_e4m3fn)
        scale_rows=(rows+3)//4*4
        scales=torch.zeros(8,32,scale_rows,device='cuda',dtype=torch.int32)
        maxima=torch.empty(rows,64,device='cuda');lse=torch.empty_like(maxima)
        fn(*[t.data_ptr() for t in [qp,kv,ids,sink,lengths,out,maxima,lse]],rows,keys,topk,512**-0.5,torch.cuda.current_stream().cuda_stream,pos.data_ptr(),rope.data_ptr(),scales.data_ptr(),scale_rows)
        torch.cuda.synchronize()
        apply_rotary_emb(q.unsqueeze(0)[...,-64:],freqs)
        ref=sparse_attn(q.unsqueeze(0),kv.unsqueeze(0),sink,ids.unsqueeze(0),512**-0.5)[0]
        apply_rotary_emb(ref.unsqueeze(0)[...,-64:],freqs,inverse=True)
        sf=scales.view(torch.uint8).reshape(8,32,scale_rows,4).permute(2,0,1,3).reshape(scale_rows,1024)[:rows]
        scale=torch.pow(2.,sf.float()-127).repeat_interleave(32,dim=1)
        dequant=(out.float().reshape(rows,32768)*scale).reshape(rows,8,16,8,32).transpose(2,3).reshape(rows,64,512)
        diff=(dequant-ref.float()).abs();rel=(diff.square().mean()/ref.float().square().mean()).sqrt().item()
        result={'rows':rows,'relative_rms':rel,'max_abs':diff.max().item()};print(json.dumps(result),flush=True);results.append(result)
        assert rel<0.06 and diff.max()<0.25,result
        root=a.artifacts/f'fused-prefill-io-{rows}';root.mkdir(exist_ok=True)
        write_manifest(root,a.artifacts,rows,keys,topk)
        tensors={'q':qp,'kv':kv,'ids':ids,'sink':sink,'lengths':lengths,'out':out,'positions':pos,'rope':rope,'scales':scales.view(torch.uint8)}
        for name,t in tensors.items():(root/f'{name}.bin').write_bytes(t.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    (a.artifacts/'fused-prefill-precision.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
