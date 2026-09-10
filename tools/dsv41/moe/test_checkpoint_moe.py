"""Build EP4 Runtime fixtures against the supplied Gate/Expert implementation.

Each rank has independent normalized synthetic activations; all weights are read
from the original checkpoint. Runtime fixtures load and repack the checkpoint
through its weight binder and once program. No checkpoint export is produced.
"""
import argparse, contextlib, json, pathlib, sys
import torch
from safetensors import safe_open


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=pathlib.Path,required=True)
    p.add_argument('--manifest',type=pathlib.Path,required=True)
    p.add_argument('--dump',type=pathlib.Path,required=True)
    p.add_argument('--tokens',type=int,default=1)
    a=p.parse_args();sys.path.insert(0,str(a.checkpoint/'inference'));import model
    manifest=json.loads(a.manifest.read_text())
    index=json.loads((a.checkpoint/'model.safetensors.index.json').read_text())['weight_map']
    def names(value):
        if isinstance(value,str):
            if value in index:yield value
        elif isinstance(value,list):
            for v in value:yield from names(v)
        elif isinstance(value,dict):
            for v in value.values():yield from names(v)
    shards=sorted({str(a.checkpoint/index[k]) for k in names(manifest['buffers'])})
    with contextlib.ExitStack() as stack:
        files={}
        def get(name):
            shard=index[name]
            if shard not in files:files[shard]=stack.enter_context(safe_open(a.checkpoint/shard,framework='pt',device='cpu'))
            return files[shard].get_tensor(name).cuda()
        prefix='layers.0.ffn.'
        gate=type('Gate',(),dict(weight=get(prefix+'gate.weight'),bias=get(prefix+'gate.bias'),bias_vl=None,
            gate_temp=1.,score_func='sqrtsoftplus',topk=6,norm_topk_prob=True,route_scale=1.5))()
        def expert(prefix,shared=False):
            funcs={}
            for name in ('w1','w2','w3'):
                weight=get(prefix+'.'+name+'.weight')
                weight=weight.view(torch.float8_e4m3fn if shared else torch.float4_e2m1fn_x2)
                weight.scale=get(prefix+'.'+name+'.scale').view(torch.float8_e8m0fnu)
                funcs[name]=staticmethod(lambda x,w=weight:model.linear(x,w))
            return type('Expert',(),dict(swiglu_limit=10.,**funcs))()
        shared=expert(prefix+'shared_experts',True)
        torch.set_default_dtype(torch.bfloat16)
        for rank in range(4):
            torch.manual_seed(620+rank)
            x=torch.randn(a.tokens,5120,device='cuda',dtype=torch.bfloat16)
            rw,idx=model.Gate.forward(gate,x)
            ref=torch.zeros_like(x,dtype=torch.float32)
            for eid in sorted(set(idx.flatten().tolist())):
                ex=expert(prefix+f'experts.{eid}')
                for j in range(6):
                    mask=idx[:,j]==eid
                    if mask.any():ref[mask]+=model.Expert.forward(ex,x[mask],rw[mask,j,None]).float()
            ref+=model.Expert.forward(shared,x).float();ref=ref.bfloat16()
            path=a.dump/f'rank{rank}';path.mkdir(parents=True,exist_ok=True)
            save=lambda name,t:(path/name).write_bytes(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
            save('x.bin',x);save('y.bin',ref);save('idx.bin',idx);save('routing_weights.bin',rw)
            io={'inputs':{'x':'x.bin'},'outputs':{'y':'y.bin'},'weights':shards,'vars':{'tokens':a.tokens},'bf16_relative_squared_limit':0.003}
            (path/'manifest.json').write_text(json.dumps(manifest));(path/'io.json').write_text(json.dumps(io))
            print('oracle rank',rank,'experts',idx.tolist(),'norm',float(ref.float().norm()),flush=True)
    torch.set_default_dtype(torch.float32)

if __name__=='__main__':main()
