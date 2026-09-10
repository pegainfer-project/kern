#!/usr/bin/env python3
"""Original DSparkAttention versus three checkpoint-bound window-only lowerings."""
import argparse
import contextlib
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser()
    for n in ('checkpoint','bindings','artifacts','cubins','auxiliary','out'):
        parser.add_argument('--'+n,type=Path,required=True)
    parser.add_argument('--check',action='store_true');a=parser.parse_args()
    import torch
    torch.set_num_threads(4)
    anchors=[7,137];rows=10;capacity=12;context=256;page=128
    if a.check:
        report=[]
        for layer in range(3):
            root=a.out/f'layer{layer}'
            def read(n,d,s):return torch.frombuffer(bytearray((root/n).read_bytes()),dtype=d).reshape(s)
            ref=read('expected.bin',torch.bfloat16,(rows,5120)).float()
            got=read('output.bin',torch.bfloat16,(capacity,5120))[:rows].float()
            error=((got-ref).square().sum()/ref.square().sum()).sqrt().item()
            actual=read('indices.bin',torch.int32,(capacity,192))[:rows]
            expected=read('expected_indices.bin',torch.int32,(rows,192))
            assert torch.equal(actual,expected)
            pos=read('draft_position.bin',torch.int32,(capacity,))[:rows]
            assert pos.tolist()==[v for anchor in anchors for v in range(anchor,anchor+5)]
            item={'layer':layer,'relative_rms':error,'window_and_positions_exact':True}
            print(json.dumps(item),flush=True);report.append(item);assert error<0.07,item
        (a.out/'precision.json').write_text(json.dumps(report,indent=2));return
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from kern_manifest import SCHEMA_VERSION
    from dsv41.loading import Pieces,dense_scales
    from dsv41.attention_forward import forward
    from dsv41.auxiliary.serving import build,Layout
    from dsv41.auxiliary.ops import definitions
    from dsv41.forward import selected
    raw=json.loads(a.bindings.read_text())['gpu']
    sys.path.insert(0,str(a.checkpoint/'inference'));import model
    from safetensors import safe_open
    cfg=json.loads((a.checkpoint/'inference/config.json').read_text());cfg.update(max_batch_size=1,max_seq_len=context)
    args=model.ModelArgs(**cfg);index=json.loads((a.checkpoint/'model.safetensors.index.json').read_text())['weight_map']
    torch.set_default_dtype(torch.bfloat16)
    a.out.mkdir(parents=True,exist_ok=True)
    with torch.device('cuda'),torch.no_grad(),contextlib.ExitStack() as stack:
        files={}
        def get(n):
            shard=index[n]
            if shard not in files:files[shard]=stack.enter_context(safe_open(a.checkpoint/shard,framework='pt'))
            return files[shard].get_tensor(n).cuda()
        torch.manual_seed(137221)
        histories=[torch.randn(1,n,5120,dtype=torch.bfloat16) for n in anchors]
        x=torch.randn(2,5,5120,dtype=torch.bfloat16)
        table=torch.tensor([[2,0],[3,1]],dtype=torch.int32)
        positions=torch.tensor([v for anchor in anchors for v in range(anchor,anchor+6)],dtype=torch.int32)
        slots=torch.stack([table[s,v//page]*page+v%page for s,anchor in enumerate(anchors) for v in range(anchor,anchor+6)]).long()
        expected_ids=torch.full((rows,192),-1,dtype=torch.int32)
        for seq,anchor in enumerate(anchors):
            ids=torch.arange(max(0,anchor-128),anchor+5,dtype=torch.int32)
            expected_ids[seq*5:seq*5+5,:len(ids)]=table[seq,ids//page]*page+ids%page
        for layer in range(3):
            root=a.out/f'layer{layer}';root.mkdir(exist_ok=True)
            prefix=f'mtp.{layer}.attn';weights={n:b for n,b in raw.items() if n.startswith(prefix+'.')}
            ref=model.DSparkAttention(40+layer,args)
            for n,param in ref.named_parameters():
                source=get(prefix+'.'+n)
                if n=='wo_a.weight':source=(source.float()*get(prefix+'.wo_a.scale').float().repeat_interleave(32,0).repeat_interleave(32,1)).bfloat16()
                param.copy_(source)
            outputs=[];kv=[];kvslots=[]
            for seq,anchor in enumerate(anchors):
                history=histories[seq]
                ref(None,0,history[:,:anchor-1])
                outputs.append(ref(x[seq:seq+1],anchor-1,history[:,anchor-1:]))
                raw_kv=ref.kv_norm(ref.wkv(history));model.apply_rotary_emb(raw_kv[...,-64:],ref.freqs_cis[:anchor])
                kv.append(raw_kv.flatten(0,1))
                ix=torch.arange(anchor,dtype=torch.int32);kvslots.append((table[seq,ix//page]*page+ix%page).long())
            pieces=Pieces();packed,load,layouts=dense_scales(weights,pieces,cubin_dir=a.cubins)
            serving=build(a.auxiliary,Layout(max_seqs=2,max_tokens=capacity,max_context=context))
            stage=forward(pieces,serving,f'mtp.{layer}',layouts,'hidden','output',mode='draft',prefix='draft.attention',capacity=capacity,
                          pool_tokens=context*2,cos_sin='cos_sin',cubin_dir=a.cubins,auxiliary_cubin=a.auxiliary,
                          attention_cubin=a.artifacts/'libdsv41_paged_decode.2.sm_103a.cubin')
            init=selected(definitions(a.auxiliary,rows=sum(anchors)),'cache_fp8')
            init[1]['cache_fp8']['params'][0]='out buffer<u8>'
            name=pieces.add(init)['cache_fp8']
            calls=load+[{'op':name,'args':[{'buf':'cache'},{'buf':'context_kv'},{'buf':'context_slots'},{'i32':sum(anchors)},{'i32':page}]}]+serving.prepare('draft','draft_ids')+stage.calls
            programs={'probe':{'calls':calls}};aux=serving.pieces(programs)
            buffers={**weights,**packed,**stage.buffers,**aux['buffers']}
            specs=[('hidden','bf16',[capacity,5120]),('context_kv','bf16',[sum(anchors),512]),('context_slots','i64',[sum(anchors)]),('cos_sin','f32',[context,64]),('positions','i32',[capacity]),('slot_mapping','i64',[capacity]),('valid','i32',[capacity]),('draft_ids','i64',[rows]),('cu_seqlens','i32',[3]),('seq_lens','i32',[2]),('page_table','i32',[2,2])]
            for n,d,s in specs:buffers[n]={'kind':'input','dtype':d,'shape':s}
            buffers['cache']={'kind':'workspace','dtype':'u8','shape':[4,page*528]}
            for n in ('output','draft.window_indices','draft.position'):buffers[n]['kind']='output'
            for spec in buffers.values():spec.pop('domain',None)
            ops={**pieces.ops,**aux['ops']}
            for op in ops.values():
                op['params']=[t.replace('inout state','inout buffer<u8>').replace('in state','in buffer<u8>') for t in op['params']]
            for call in calls:
                for arg in call['args']:
                    if 'state' in arg:arg.pop('state');arg['buf']='cache'
            modules={**pieces.modules,**aux['modules']};mdir=root/'cubins';mdir.mkdir(exist_ok=True)
            for mod in modules.values():
                source=Path(mod['source'])
                if not source.exists():source=next(d/source.name for d in (a.artifacts,a.auxiliary.parent,a.cubins) if (d/source.name).exists())
                dst=mdir/source.name
                if not dst.exists():dst.symlink_to(source.resolve())
                mod['source']=source.name
            manifest={'schema_version':SCHEMA_VERSION,'model':'checkpoint-dspark-attention','vars':{'seqs':{'max':2}},'buffers':buffers,'modules':modules,'ops':ops,'programs':programs}
            # Serving buffers use token-capacity expressions even though this
            # probe's draft row count is exclusively five times seqs.
            def literal(v):
                if v=='tokens':return capacity
                if isinstance(v,list):return [literal(x) for x in v]
                if isinstance(v,dict):return {k:literal(x) for k,x in v.items()}
                return v
            manifest=literal(manifest);(root/'manifest.json').write_text(json.dumps(manifest,indent=2))
            tensors={'hidden':x.flatten(0,1),'context_kv':torch.cat(kv),'context_slots':torch.cat(kvslots),'cos_sin':torch.view_as_real(ref.freqs_cis).flatten(-2),'positions':positions,'slot_mapping':slots,'valid':torch.ones(capacity,dtype=torch.int32),'draft_ids':torch.zeros(rows,dtype=torch.int64),'cu_seqlens':torch.tensor([0,6,12],dtype=torch.int32),'seq_lens':torch.tensor([n+6 for n in anchors],dtype=torch.int32),'page_table':table,'expected':torch.cat(outputs).flatten(0,1),'expected_indices':expected_ids}
            for n,t in tensors.items():(root/(n+'.bin')).write_bytes(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
            cmd=['--manifest',str(root/'manifest.json'),'--cubins',str(mdir),'--weights',str(a.checkpoint),'--vars','seqs=2','--graph','--iters','3']
            for n in tensors:
                if not n.startswith('expected'):cmd+=['--in',f'{n}={root/(n+".bin")}']
            for n,f in [('output','output'),('draft.window_indices','indices'),('draft.position','draft_position')]:cmd+=['--out',f'{n}={root/(f+".bin")}']
            (root/'runner-args.json').write_text(json.dumps(cmd));print('prepared DSpark layer',layer,flush=True)
if __name__=='__main__':main()
