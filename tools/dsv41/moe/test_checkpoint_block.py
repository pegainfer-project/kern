"""Original Block.forward oracle for a checkpoint-bound layer integration test.

Loads only selected experts lazily; attention, mHC, routing and expert math all
use the supplied inference Python. WO_A is dequantized as in official loading.
"""
import argparse,contextlib,json,pathlib,sys,types
import torch
from safetensors import safe_open


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=pathlib.Path,required=True);p.add_argument('--manifest',type=pathlib.Path,required=True);p.add_argument('--dump',type=pathlib.Path,required=True);p.add_argument('--tokens',type=int,default=1);a=p.parse_args()
    sys.path.insert(0,str(a.checkpoint/'inference'));import model
    config=json.loads((a.checkpoint/'inference/config.json').read_text());config.update(max_batch_size=1,max_seq_len=32768)
    args=model.ModelArgs(**config);manifest=json.loads(a.manifest.read_text());index=json.loads((a.checkpoint/'model.safetensors.index.json').read_text())['weight_map']
    def names(value):
        if isinstance(value,str):
            if value in index:yield value
        elif isinstance(value,list):
            for v in value:yield from names(v)
        elif isinstance(value,dict):
            for v in value.values():yield from names(v)
    shards=sorted({str(a.checkpoint/index[k]) for k in names(manifest['buffers'])})
    torch.set_default_dtype(torch.bfloat16);torch.set_default_device("cuda")
    with contextlib.ExitStack() as stack:
        files={}
        def get(name):
            shard=index[name]
            if shard not in files:files[shard]=stack.enter_context(safe_open(a.checkpoint/shard,framework='pt'))
            return files[shard].get_tensor(name).cuda()
        with torch.device('cuda'):attn=model.Attention(0,args)
        with torch.no_grad():
            for name,param in attn.named_parameters():
                source=get('layers.0.attn.'+name)
                if name=='wo_a.weight':source=(source.float()*get('layers.0.attn.wo_a.scale').float().repeat_interleave(32,0).repeat_interleave(32,1)).bfloat16()
                param.copy_(source)
        prefix='layers.0.ffn.'
        gate_ctx=types.SimpleNamespace(weight=get(prefix+'gate.weight'),bias=get(prefix+'gate.bias'),bias_vl=None,gate_temp=args.gate_temp,score_func=args.score_func,topk=6,norm_topk_prob=True,route_scale=args.route_scale)
        def expert(eid=None):
            path=prefix+('shared_experts' if eid is None else f'experts.{eid}');funcs={}
            for name in ('w1','w2','w3'):
                weight=get(path+'.'+name+'.weight').view(torch.float8_e4m3fn if eid is None else torch.float4_e2m1fn_x2)
                weight.scale=get(path+'.'+name+'.scale').view(torch.float8_e8m0fnu)
                funcs[name]=lambda x,w=weight:model.linear(x,w)
            ctx=types.SimpleNamespace(swiglu_limit=args.swiglu_limit,**funcs)
            return lambda x,weights=None:model.Expert.forward(ctx,x,weights)
        class Experts:
            def __getitem__(self,eid):return expert(eid)
        moe_ctx=types.SimpleNamespace(dim=5120,n_routed_experts=384,experts_start_idx=0,experts_end_idx=384,experts=Experts(),shared_experts=expert(),gate=lambda x,mask:model.Gate.forward(gate_ctx,x,mask))
        block=types.SimpleNamespace(norm_eps=args.norm_eps,hc_mult=4,hc_sinkhorn_iters=args.hc_sinkhorn_iters,hc_eps=args.hc_eps,attn=attn,ffn=lambda x,mask:model.MoE.forward(moe_ctx,x,mask))
        for name in ('hc_attn_fn','hc_attn_scale','hc_attn_base','hc_ffn_fn','hc_ffn_scale','hc_ffn_base'):setattr(block,name,get('layers.0.'+name))
        for name in ('hc_mixes','hc_pre','hc_post'):setattr(block,name,types.MethodType(getattr(model.Block,name),block))
        for name in ('attn_norm','ffn_norm'):
            weight=get('layers.0.'+name+'.weight');ctx=types.SimpleNamespace(weight=weight,eps=args.norm_eps)
            setattr(block,name,lambda x,c=ctx:model.RMSNorm.forward(c,x))
        for rank in range(4):
            torch.manual_seed(920+rank);embedding=torch.randn(a.tokens,5120,device='cuda',dtype=torch.bfloat16)
            residual=embedding.view(1,a.tokens,1,5120).repeat(1,1,4,1);pre=torch.zeros(1,a.tokens,4,device='cuda',dtype=torch.float32);pre[...,0]=1
            with torch.no_grad():y,newpre=model.Block.forward(block,residual,0,pre,None)
            path=a.dump/f'rank{rank}';path.mkdir(parents=True,exist_ok=True);path.chmod(0o777)
            def save(name,t):
                out=path/name;out.write_bytes(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes());out.chmod(0o666)
            inputs={'embedding':embedding,'input_ids':torch.arange(a.tokens,dtype=torch.int64),'positions':torch.arange(a.tokens,dtype=torch.int32),'valid':torch.ones(a.tokens,dtype=torch.int32),'slot_mapping':torch.arange(a.tokens,dtype=torch.int64),'seq_lens':torch.tensor([a.tokens],dtype=torch.int32),'cu_seqlens':torch.tensor([0,a.tokens],dtype=torch.int32),'page_table':torch.arange(256,dtype=torch.int32).unsqueeze(0),'cos_sin':torch.view_as_real(attn.freqs_cis).flatten(-2)}
            for name,t in inputs.items():save(name+'.bin',t)
            outputs={'prefill.materialized':y.flatten(2),'prefill.pre2':newpre.flatten(0,1)}
            for name,t in outputs.items():save(name+'.bin',t)
            io={'inputs':{k:k+'.bin' for k in inputs},'outputs':{k:k+'.bin' for k in outputs},'output_dtypes':{'prefill.pre2':'f32'},'weights':shards,'vars':{'tokens':a.tokens,'seqs':1},'capacity_tokens':32768,'program':'prefill','bf16_relative_squared_limit':0.003}
            for name,obj in [('manifest.json',manifest),('io.json',io)]:
                (path/name).write_text(json.dumps(obj));(path/name).chmod(0o666)
            print('block oracle rank',rank,'tokens',a.tokens,'residualnorm',float(y.float().norm()),'pre',newpre.flatten().tolist(),flush=True)
    torch.set_default_dtype(torch.float32)

if __name__=='__main__':main()
