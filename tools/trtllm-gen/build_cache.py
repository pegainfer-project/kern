#!/usr/bin/env python3
"""Compile vLLM's unmodified Apache-2.0 KV append kernel for page size 64."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

# Debug line tables otherwise embed the build machine's absolute source path.
os.environ['TRITON_DISABLE_LINE_INFO'] = '1'

import torch
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import reshape_and_cache_kernel_flash


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    key=torch.randn(2,4,256,device='cuda',dtype=torch.bfloat16)
    packed=torch.randn(2,14336,device='cuda',dtype=torch.bfloat16)
    value=packed[:,13312:].view(2,4,256)
    slab=torch.zeros(2,16,64,4,2,256,device='cuda',dtype=torch.bfloat16)
    k,v=slab[:,3,:,:,0,:],slab[:,3,:,:,1,:]
    slots=torch.tensor([63,64],device='cuda',dtype=torch.int64)
    scale=torch.ones(2,device='cuda',dtype=torch.float32)
    compiled=reshape_and_cache_kernel_flash[(2,1)](
        key,value,k,v,slots,scale[:1],scale[1:],
        1024,14336,2097152,512,0,0,2048,
        num_heads=4,head_size=256,block_size=64,x=1,USE_HEAD_MAJOR_LAYOUT=False,
        FP8_KV_CACHE=False,TILE_SIZE=1024,num_warps=16,num_stages=10)
    torch.cuda.synchronize()
    assert torch.equal(k[0,63],key[0]) and torch.equal(v[1,0],value[1])
    data=compiled.asm['cubin'];name='qwen38_cache_p64.cubin'
    (args.output/name).write_bytes(data)
    meta=dict(source=name,sha256=hashlib.sha256(data).hexdigest(),entry=compiled.name,
              upstream='vllm/v1/attention/ops/triton_reshape_and_cache_flash.py',license='Apache-2.0',
              versions={name:importlib.metadata.version(name) for name in ('vllm','torch','triton')},
              disable_line_info=True,
              specialization=dict(page_size=64,num_heads=4,head_dim=256,layers=16,
                                  num_warps=16,num_stages=10,tile_size=1024))
    (args.output/'cache.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta))

if __name__=='__main__':main()
