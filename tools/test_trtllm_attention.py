#!/usr/bin/env python3
"""Run generated TRTLLM-GEN ops through kern, against unmodified FlashInfer.

Tests page boundaries, ragged batches, chunked prefill, verify and 200k decode.
The program_io runner executes eager and repeated CUDA graphs; comparison
uses the final graph output, including after split-KV counters are reused.
Temporary tensor fixtures are removed after each case.
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile

import torch
import flashinfer
from kern_manifest import normalize, program
from trtllm_attention import op, fetch, LAYER_PAGE_BYTES


def probe_manifest(batch, qlen, lengths, pages, table_width, mode, splits):
    rows=batch*qlen
    buffers={
        'q':dict(dtype='bf16',shape=[rows,24,256],kind='input'),
        'out':dict(dtype='bf16',shape=[rows,24,256],kind='output'),
        'slab':dict(dtype='bf16',shape=[pages,2,64,4,2,256],kind='input'),
        'table':dict(dtype='i32',shape=[batch,table_width],kind='input'),
        'seq':dict(dtype='i32',shape=[batch],kind='input'),
        'cuq':dict(dtype='i32',shape=[batch+1],kind='input'),
    }
    # Extra Q/O capacity checks descriptor bounds are independent of live rows.
    bound=max(64,rows)
    buffers['q']['shape'][0]=bound;buffers['out']['shape'][0]=bound
    args=[{'buf':'out'},{'buf':'q'},{'buf':'slab','offset':LAYER_PAGE_BYTES},
          {'buf':'slab','offset':LAYER_PAGE_BYTES+512},{'buf':'table'},
          {'buf':'seq'},{'buf':'cuq'},{'var':'seqs'},{'var':'tokens'}]
    return normalize(dict(schema_version=5,model='trtllm-gen-qwen38-probe',
        vars={'tokens':{'max':bound},'seqs':{'max':batch}},buffers=buffers,
        ops={'attn':op(mode,max_rows=bound,max_seqs=batch,max_context=table_width*64,
                       layers=2,splits=splits,kv_type='in buffer<bf16>')},
        programs={'probe':program([dict(label='attn',op='attn',args=args)],groups=batch,rows=qlen)}))


def run_case(args,batch,qlen,average,mode):
    torch.manual_seed(1701+batch+qlen+average)
    lengths=([round(average*(.75+.5*i/(batch-1))) for i in range(batch)] if batch>1 else [average])
    assert min(lengths)>=qlen
    width=math.ceil(max(lengths)/64);pages=width*batch
    slab=torch.randn(pages,2,64,4,2,256,device='cuda',dtype=torch.bfloat16)
    k,v=slab[:,1,:,:,0,:],slab[:,1,:,:,1,:]
    q=torch.randn(batch*qlen,24,256,device='cuda',dtype=torch.bfloat16)
    table=torch.randperm(pages,device='cuda',dtype=torch.int32).view(batch,width)
    seq=torch.tensor(lengths,device='cuda',dtype=torch.int32)
    cuq=torch.arange(batch+1,device='cuda',dtype=torch.int32)*qlen
    cuk=torch.cat((torch.zeros(1,device='cuda',dtype=torch.int32),seq.cumsum(0,dtype=torch.int32)))
    workspace=torch.zeros(256*1024*1024,device='cuda',dtype=torch.uint8)
    common=dict(query=q,kv_cache=(k,v),workspace_buffer=workspace,block_tables=table,seq_lens=seq,
                bmm1_scale=1/16,bmm2_scale=1.,kv_layout='NHD',enable_pdl=False)
    if mode=='decode':
        expected=flashinfer.decode.trtllm_batch_decode_with_kv_cache(**common,max_seq_len=max(lengths),backend='trtllm-gen')
    else:
        expected=flashinfer.prefill.trtllm_batch_context_with_kv_cache(**common,max_q_len=qlen,
            max_kv_len=max(lengths),batch_size=batch,cum_seq_lens_q=cuq,cum_seq_lens_kv=cuk,causal=True)
    torch.cuda.synchronize();expected=expected.cpu().float()
    manifest=probe_manifest(batch,qlen,lengths,pages,width,mode,args.splits)
    with tempfile.TemporaryDirectory(dir=args.work_dir,prefix='probe-') as tmp:
        tmp=Path(tmp);(tmp/'manifest.json').write_text(json.dumps(manifest))
        cmd=[str(args.runner.resolve()),'--gpu',str(args.gpu),'--manifest',str(tmp/'manifest.json'),
             '--cubins',str(args.cubins.resolve()),'--env',f'seqs={batch}','--env',f'tokens={batch*qlen}',
             '--graph','--iters','9','--out',f'out={tmp}/out.bin']
        for name,tensor in dict(q=q,slab=slab,seq=seq,table=table,cuq=cuq).items():
            # Stream slabs page-by-page, bounding CPU RAM and Python bytes copies.
            with (tmp/f'{name}.bin').open('wb') as f:
                if name=='slab':
                    for chunk in tensor.split(32):
                        f.write(chunk.contiguous().view(torch.uint8).cpu().numpy().tobytes())
                else:f.write(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes())
            cmd+=['--in',f'{name}={tmp}/{name}.bin']
        result=subprocess.run(cmd,text=True,capture_output=True,timeout=300)
        if result.returncode:
            raise RuntimeError(result.stderr)
        actual=torch.frombuffer(bytearray((tmp/'out.bin').read_bytes()),dtype=torch.bfloat16)[:batch*qlen*24*256].reshape_as(expected).float()
        delta=actual-expected
        rel=(delta.norm()/expected.norm().clamp_min(1e-12)).item()
        absmax=delta.abs().max().item()
        passed=bool(torch.isfinite(actual).all()) and rel<.02 and absmax<.02
        row=dict(mode=mode,batch=batch,query_tokens=qlen,mean_kv=average,min_kv=min(lengths),max_kv=max(lengths),
                 splits=args.splits if mode=='decode' else 1,passed=passed,bit_exact=torch.equal(actual,expected),
                 relative_l2=rel,max_abs=absmax,graph_median_ms=float(re.search(r'graph_median_ms=([\d.eE+-]+)',result.stdout)[1]))
        print(json.dumps(row),flush=True)
        if not passed:raise AssertionError(row)
        return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runner',type=Path,default=Path('target/release/examples/program_io'))
    p.add_argument('--cubins',type=Path,required=True)
    p.add_argument('--work-dir',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--splits',type=int,default=38)
    p.add_argument('--quick',action='store_true')
    args=p.parse_args();args.work_dir.mkdir(parents=True,exist_ok=True)
    torch.cuda.set_device(args.gpu);torch.set_grad_enabled(False)
    fetch(args.cubins)
    cases=[(1,1,1,'decode'),(2,1,63,'decode'),(16,1,65,'decode'),(32,1,513,'decode'),
           (1,1,32768,'decode'),(2,8,513,'prefill'),(16,8,513,'prefill'),
           (1,127,257,'prefill'),(2,129,513,'prefill'),(1,2048,8192,'prefill')]
    if not args.quick:cases += [(b,1,200000,'decode') for b in [1,2,16,32]]
    with (args.work_dir/'validation.jsonl').open('w',buffering=1) as f:
        for case in cases:
            row=run_case(args,*case);f.write(json.dumps(row)+'\n')
            gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
