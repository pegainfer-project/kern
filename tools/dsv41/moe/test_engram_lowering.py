"""Real layer-1 host lookup, dense projection and Engram injection integration.

The oracle fetches only selected checkpoint rows. Runtime binds the complete
immutable host table via its ordinary loader, without exporting a checkpoint.
Two independent five-row sequences: one valid, one scheduler padding. Hash and
lookup contents of padding rows are unspecified; their residual must pass through.
"""
import argparse,json,pathlib,sys,types
import torch
from safetensors import safe_open
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]))
from dsv41.loading import Pieces,dense_scales
from dsv41.engram import inject
from dsv41.auxiliary.serving import build,Layout
from kern_manifest import SCHEMA_VERSION


def main():
 p=argparse.ArgumentParser()
 for name in ('checkpoint','bindings','auxiliary-cubin','constants','cubin-dir','dump'):p.add_argument('--'+name,type=pathlib.Path,required=True)
 a=p.parse_args();a.dump.mkdir(parents=True,exist_ok=True);a.dump.chmod(0o777)
 sys.path.insert(0,str(a.checkpoint/'inference'));import model,engram
 raw=json.loads(a.bindings.read_text());weights={n:v for n,v in raw['gpu'].items() if n.startswith('layers.1.engram.')}
 for n,v in raw['host'].items():
  if n.startswith('layers.1.engram.'):weights[n]={**v,'placement':'host'}
 const=json.loads((a.constants/'engram_constants.json').read_text());pieces=Pieces();serving=build(a.auxiliary_cubin,Layout(max_seqs=2,max_tokens=128,max_context=32768))
 packed,load,layouts=dense_scales({n:v for n,v in weights.items() if v.get("placement")!="host"},pieces,cubin_dir=a.cubin_dir)
 stage=inject(pieces,serving,1,layouts,'residual','output',mode='prefill',capacity=128,cubin_dir=a.cubin_dir)
 programs={'load':{'once':True,'calls':[const['calls'][0]]+load},'probe':{'batch':{'groups':2,'rows':5},'calls':serving.prepare('prefill','input_ids')+serving.engram_hash('prefill',**const['names'],compressed_pad_id=const['metadata']['compressed_pad_id'])+stage.calls}}
 aux=serving.pieces(programs);buffers={**weights,**packed,**const['buffers'],**aux['buffers'],**stage.buffers,'residual':{'kind':'input','dtype':'bf16','shape':[128,20480]}}
 used={v['buf'] for p in programs.values() for c in p['calls'] for v in c['args'] if 'buf'in v};buffers={n:b for n,b in buffers.items() if n in used}
 for name in ('output','prefill.hashes','prefill.engram.embedding'):buffers[name]['kind']='output'
 ops={**pieces.ops,**const['ops'],**aux['ops']};wanted={c['op'] for p in programs.values() for c in p['calls']};ops={n:o for n,o in ops.items() if n in wanted}
 modules={**pieces.modules,**const['modules'],**aux['modules']};wanted={l['module'] for o in ops.values() for l in o['impl']['launches'] if 'module'in l};modules={n:m for n,m in modules.items() if n in wanted}
 manifest={'schema_version':SCHEMA_VERSION,'model':'engram-host-integration','vars':aux['vars'],'states':aux['states'],'buffers':buffers,'modules':modules,'ops':ops,'programs':programs}
 index=json.loads((a.checkpoint/'model.safetensors.index.json').read_text())['weight_map']
 def get(name):
  with safe_open(a.checkpoint/index[name],framework='pt') as f:return f.get_tensor(name).cuda()
 torch.set_default_dtype(torch.bfloat16);torch.set_default_device('cuda');torch.manual_seed(713)
 constants=json.loads((a.constants/'engram_constants.reference.json').read_text())
 ctx=types.SimpleNamespace(layout=types.SimpleNamespace(max_ngram_size=4),pad_id=const['metadata']['compressed_pad_id'],DEAD=-1,cache=torch.zeros(2,32768,dtype=torch.int64),token_map=torch.tensor(constants['token_map'],dtype=torch.int64),multipliers=torch.tensor(constants['multipliers'],dtype=torch.int64).view(2,4),primes=torch.tensor(constants['primes'],dtype=torch.int64).view(2,3,8),offsets=torch.tensor(constants['offsets'],dtype=torch.int64).view(2,24))
 ids=torch.tensor([[42,1057,11,850,321],[0,0,0,0,0]],dtype=torch.int64);mask=torch.tensor([[1,1,1,1,1],[0,0,0,0,0]],dtype=torch.bool)
 hashes=engram.NgramHashState.forward(ctx,ids,0,mask);selected,inverse=hashes[:,:,0].flatten().unique(return_inverse=True)
 def rows(name):
  with torch.device('cpu'),safe_open(a.checkpoint/index[name],framework='pt',device='cpu') as f:
   view=f.get_slice(name);raw=torch.cat([view[int(i):int(i)+1].view(torch.uint8) for i in selected.cpu().tolist()])
  return raw.cuda()
 table=rows('layers.1.engram.embed.weight').view(torch.float8_e4m3fn);scale=rows('layers.1.engram.embed.scale').view(torch.float8_e8m0fnu)
 embed=types.SimpleNamespace(weight=table,scale=scale,vocab_start_idx=0,vocab_end_idx=len(selected),block_size=32)
 embedding=model.ParallelEngramEmbedding.forward(embed,inverse.view(2,5,24))
 weight=get('layers.1.engram.wkv.weight');weight.scale=get('layers.1.engram.wkv.scale')
 ctx=types.SimpleNamespace(hc_mult=4,dim=5120,eps=1e-20,clamp_value=1e-6,q_weight=get('layers.1.engram.q_weight'),k_weight=get('layers.1.engram.k_weight'),embed=lambda ids:embedding,wkv=lambda x:model.linear(x,weight))
 residual=torch.randn(2,5,4,5120);output=model.Engram.forward(ctx,residual,hashes[:,:,0],mask)
 inputs={'residual':residual.flatten(0,1),'input_ids':ids.flatten(),'positions':torch.arange(5,dtype=torch.int32).repeat(2),'valid':mask.flatten().int(),'slot_mapping':torch.cat([torch.arange(5,dtype=torch.int64),torch.arange(128,133,dtype=torch.int64)]),'seq_lens':torch.tensor([5,5],dtype=torch.int32),'cu_seqlens':torch.tensor([0,5,10],dtype=torch.int32),'page_table':torch.stack([torch.zeros(256,dtype=torch.int32),torch.ones(256,dtype=torch.int32)])}
 outputs={'output':output.flatten(0,1),'prefill.hashes':hashes.flatten(0,1),'prefill.engram.embedding':embedding.flatten(0,1)}
 def save(name,t):
  p=a.dump/(name+'.bin');p.write_bytes(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes());p.chmod(0o666)
 for group in (inputs,outputs):
  for name,t in group.items():save(name,t)
 shards=sorted({str(a.checkpoint/index[v['bind'][0]['tensor']]) for v in buffers.values() if v['kind']=='weight'})
 io={'vars':{'tokens':10,'seqs':2},'weights':shards,'inputs':{n:n+'.bin' for n in inputs},'outputs':{n:n+'.bin' for n in outputs},'compare_rows':{'prefill.hashes':5,'prefill.engram.embedding':5,'output':10},'padding_passthrough':{'output':{'input':'residual','start':5}}}
 for name,obj in [('manifest.json',manifest),('io.json',io)]:
  p=a.dump/name;p.write_text(json.dumps(obj));p.chmod(0o666)
 print('Engram fixture ready: real host table bytes',sum(torch.tensor(v['shape']).prod().item() for v in weights.values() if v.get('placement')=='host'),'calls',len(programs['probe']['calls']))

if __name__=='__main__':main()
