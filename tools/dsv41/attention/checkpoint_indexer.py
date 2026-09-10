#!/usr/bin/env python3
"""Checkpoint Indexer.forward versus complete query-to-physical-index lowering.

The prepopulated paged state is fed as an input buffer for deterministic replay;
only public pointer types change, while the scorer TMA and call sequence remain.
Checkpoint weights bind directly; this single-program harness prepends scale packing.
"""
import argparse
import contextlib
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser()
    for n in ('checkpoint','bindings','artifacts','cubins','auxiliary','out'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--check',action='store_true');a=p.parse_args()
    import torch
    torch.set_num_threads(4)
    rows,capacity,context,page,ratio=5,8,4096,64,2
    if a.check:
        def read(name,dtype,shape):return torch.frombuffer(bytearray((a.out/name).read_bytes()),dtype=dtype).reshape(shape)
        expected=read('expected_scores.bin',torch.bfloat16,(rows,2048)).float()
        got=read('scores.bin',torch.float32,(capacity,2048))[:rows]
        valid=torch.isfinite(expected);assert torch.equal(torch.isneginf(expected),torch.isneginf(got))
        relative=((got[valid]-expected[valid]).square().mean()/expected[valid].square().mean()).sqrt().item()
        actual=read('logical.bin',torch.int32,(capacity,512))[:rows]
        reference=read('expected_indices.bin',torch.int32,(rows,512))
        overlap=[len(set(x.tolist())&set(y.tolist()))/512 for x,y in zip(actual,reference)]
        table=read('page_table.bin',torch.int32,(rows,context//128))
        physical=read('physical.bin',torch.int32,(capacity,512))[:rows]
        mapped=torch.gather(table,1,actual.long()//page)*page+actual%page
        assert torch.equal(physical,mapped)
        result={'rows':rows,'score_relative_rms':relative,'top512_overlap':overlap,'physical_mapping_exact':True}
        print(json.dumps(result),flush=True);(a.out/'precision.json').write_text(json.dumps(result,indent=2))
        assert relative<0.035 and min(overlap)>0.97,result
        return
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from kern_manifest import SCHEMA_VERSION
    from dsv41.loading import Pieces,dense_scales
    from dsv41.indexer import forward
    from dsv41.auxiliary.serving import build,Layout
    raw=json.loads(a.bindings.read_text())['gpu']
    prefix='layers.2.attn.indexer'
    weights={n:raw[n] for n in (prefix+'.wq_b.weight',prefix+'.wq_b.scale',prefix+'.weights_proj.weight')}
    pieces=Pieces();packed,load,layouts=dense_scales(weights,pieces,cubin_dir=a.cubins)
    serving=build(a.auxiliary,Layout(max_seqs=5,max_tokens=32,max_context=context))
    names={'ends':'decode.c2_end','pages':'decode.c2_page_table','request_ids':'decode.index_request_ids','sparse_ends':'decode.sparse_ends'}
    stage=forward(pieces,serving,2,2,layouts,'hidden','qr',mode='decode',capacity=capacity,
        pool_tokens=context,cos_sin='cos_sin',metadata=names,cubin_dir=a.cubins,
        auxiliary_cubin=a.auxiliary,dense_cubin=a.artifacts/'paged_indexer.cubin',
        sparse_cubin=a.artifacts/'sparse_indexer.cubin',select_cubin=a.artifacts/'libdsv41_select.2.sm_103a.cubin',candidate_cubin=a.artifacts/'candidate.cubin')
    programs={'probe':{'calls':load+serving.index_metadata('decode',2)+stage.lowered.calls}}
    aux=serving.pieces(programs)
    buffers={**weights,**packed,**stage.lowered.buffers,**aux['buffers']}
    for name,dtype,shape in [('hidden','bf16',[capacity,5120]),('qr','bf16',[capacity,1280]),('cos_sin','f32',[context,64]),('decode.position','i32',[32]),('decode.request','i32',[32]),('decode.mask','u8',[32]),('page_table','i32',[5,context//128]),('index_cache','u8',[context//128,4608])]:
        buffers[name]={'kind':'input','dtype':dtype,'shape':shape}
    for spec in buffers.values():spec.pop('domain',None)
    for name in (stage.logical,stage.physical,'decode.index2.scores'):buffers[name]['kind']='output'
    ops={**pieces.ops,**aux['ops']}
    for op in ops.values():op['params']=[x.replace('in state','in buffer<u8>') for x in op['params']]
    for prog in programs.values():
        for call in prog['calls']:
            for arg in call['args']:
                if 'state' in arg:arg.pop('state');arg['buf']='index_cache'
    modules={**pieces.modules,**aux['modules']}
    a.out.mkdir(parents=True,exist_ok=True)
    module_dir=a.out/'cubins';module_dir.mkdir(exist_ok=True)
    for module in modules.values():
        source=Path(module['source'])
        if not source.exists():
            source=next(root/source.name for root in (a.artifacts,a.auxiliary.parent,a.cubins) if (root/source.name).exists())
        target=module_dir/source.name
        if not target.exists():target.symlink_to(source.resolve())
        module['source']=source.name
    manifest={'schema_version':SCHEMA_VERSION,'model':'checkpoint-indexer-lowering','vars':{'tokens':aux['vars']['tokens']},
              'states':{},'buffers':buffers,'modules':modules,'ops':ops,'programs':programs}
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    sys.path.insert(0,str(a.checkpoint/'inference'));import model
    from kernel import fp4_act_quant
    from safetensors import safe_open
    config=json.loads((a.checkpoint/'inference/config.json').read_text());config.update(max_batch_size=1,max_seq_len=context)
    args=model.ModelArgs(**config);index=json.loads((a.checkpoint/'model.safetensors.index.json').read_text())['weight_map']
    torch.set_default_dtype(torch.bfloat16)
    with torch.device('cuda'),torch.no_grad(),contextlib.ExitStack() as stack:
        files={}
        def get(name):
            shard=index[name]
            if shard not in files:files[shard]=stack.enter_context(safe_open(a.checkpoint/shard,framework='pt'))
            return files[shard].get_tensor(name).cuda()
        ref=model.Indexer(args,2)
        for name,param in ref.named_parameters():param.copy_(get(prefix+'.'+name))
        ref.freqs_cis=model.precompute_freqs_cis(64,context,args.original_seq_len,args.compress_rope_theta,args.rope_factor,args.beta_fast,args.beta_slow)
        torch.manual_seed(221432)
        latent=torch.randn(1,context//ratio,512,dtype=torch.bfloat16)
        keys=ref.k_norm(ref.wk(latent));model.apply_rotary_emb(keys[...,-64:],ref.freqs_cis[::ratio])
        kp,ks=fp4_act_quant(keys);fp4_act_quant(keys,inplace=True)
        ref.k_cache.copy_(keys);model.shared_attn.index_k=ref.k_cache
        hidden=torch.randn(rows,5120,dtype=torch.bfloat16);qr=torch.randn(rows,1280,dtype=torch.bfloat16)
        positions=torch.arange(rows,dtype=torch.int32)*2+2048
        scores=[];indices=[]
        for row,pos in enumerate(positions.tolist()):
            x=hidden[row:row+1,None];z=qr[row:row+1,None]
            indices.append(ref(x,z,None,pos,0).flatten())
            q=ref.wq_b(z).reshape(1,1,32,128);model.apply_rotary_emb(q[...,-64:],ref.freqs_cis[pos:pos+1]);fp4_act_quant(q,inplace=True)
            w=ref.weights_proj(x)*(1/64)
            score=(torch.einsum('bshd,btd->bsht',q,keys).relu()*w.unsqueeze(-1)).sum(dim=2).flatten()
            score[(pos+1)//ratio:]=-torch.inf;scores.append(score)
        perm=torch.randperm(context//128);table=perm.argsort().int()[None].repeat(rows,1)
        cache=torch.zeros((context//128,4608),dtype=torch.uint8)
        cache[:,:page*68]=torch.cat([kp.view(torch.uint8).reshape(-1,page*64)[perm],ks.view(torch.uint8).reshape(-1,page*4)[perm]],dim=1)
        tensors={'hidden':hidden,'qr':qr,'cos_sin':torch.view_as_real(ref.freqs_cis).flatten(-2),
                 'decode.position':positions,'decode.request':torch.arange(rows,dtype=torch.int32),'decode.mask':torch.ones(rows,dtype=torch.uint8),
                 'page_table':table,'index_cache':cache,'expected_scores':torch.stack(scores),'expected_indices':torch.stack(indices)}
        for name,t in tensors.items():(a.out/(name+'.bin')).write_bytes(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    command=['--manifest',str(a.out/'manifest.json'),'--cubins',str(module_dir),'--program','probe','--vars','tokens=5','--weights',str(a.checkpoint),'--graph','--iters','3']
    for name in tensors:
        if not name.startswith('expected'):command+=['--in',f'{name}={a.out/(name+".bin")}']
    for name,file in ((stage.logical,'logical'),(stage.physical,'physical'),('decode.index2.scores','scores')):command+=['--out',f'{name}={a.out/(file+".bin")}']
    (a.out/'runner-args.json').write_text(json.dumps(command))
    print('Prepared checkpoint Indexer.forward oracle and',len(programs['probe']['calls']),'lowered calls',flush=True)
if __name__=='__main__':main()
