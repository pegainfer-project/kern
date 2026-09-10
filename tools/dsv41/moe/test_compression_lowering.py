"""Source-2 compressor and index-key publication against supplied inference."""
import argparse,json,pathlib,sys
import torch
from safetensors import safe_open
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]))
from dsv41.loading import Pieces
from dsv41.compression import publish
from dsv41.auxiliary.serving import build,Layout,index_page_stride
from kern_manifest import SCHEMA_VERSION


def main():
 p=argparse.ArgumentParser()
 for name in ('checkpoint','bindings','auxiliary-cubin','dump'):p.add_argument('--'+name,type=pathlib.Path,required=True)
 a=p.parse_args();a.dump.mkdir(parents=True,exist_ok=True);a.dump.chmod(0o777)
 sys.path.insert(0,str(a.checkpoint/'inference'));import model
 raw=json.loads(a.bindings.read_text())['gpu'];pieces=Pieces();serving=build(a.auxiliary_cubin,Layout(max_seqs=1,max_tokens=128,max_context=32768))
 stage=publish(pieces,serving,2,'x',mode='prefill',capacity=128,cos_sin='cos_sin',auxiliary_cubin=a.auxiliary_cubin)
 programs={'probe':{'batch':{'groups':1,'rows':'tokens'},'calls':serving.prepare('prefill','input_ids')+stage.lowered.calls+stage.commit}}
 aux=serving.pieces(programs);buffers={**aux['buffers'],**stage.lowered.buffers,'x':{'dtype':'bf16','shape':[128,5120],'kind':'input'},'cos_sin':{'dtype':'f32','shape':[32768,64],'kind':'input'}}
 used={arg['buf'] for call in programs['probe']['calls'] for arg in call['args'] if 'buf'in arg}
 buffers.update({n:raw[n] for n in used-buffers.keys()})
 for name in (stage.latent,'prefill.compress2.key_dequant'):buffers[name]['kind']='output'
 for value in buffers.values():
  if value.get('domain',{}).get('index_into')=='engram_history':value['domain']['index_into']='compressed.2'
 aux['states'].pop('engram_history',None)
 modules={**pieces.modules,**aux['modules']};ops={**pieces.ops,**aux['ops']}
 manifest={'schema_version':SCHEMA_VERSION,'model':'compressor-publish-integration','vars':aux['vars'],'states':aux['states'],'buffers':buffers,'modules':modules,'ops':ops,'programs':programs}
 index=json.loads((a.checkpoint/'model.safetensors.index.json').read_text())['weight_map']
 def get(name):
  with safe_open(a.checkpoint/index[name],framework='pt') as f:return f.get_tensor(name).cuda()
 torch.set_default_dtype(torch.bfloat16);torch.set_default_device('cuda');torch.manual_seed(477)
 config=json.loads((a.checkpoint/'inference/config.json').read_text());config.update(max_batch_size=1,max_seq_len=32768);args=model.ModelArgs(**config)
 compressor=model.Compressor(args,2)
 with torch.no_grad():
  for name,param in compressor.named_parameters():param.copy_(get('layers.2.attn.compressor.'+name))
 x=torch.randn(1,5,5120);latent=compressor(x,0);wk=get('layers.2.attn.indexer.wk.weight');norm=get('layers.2.attn.indexer.k_norm.weight')
 from types import SimpleNamespace
 key=model.RMSNorm.forward(SimpleNamespace(eps=args.norm_eps,weight=norm),model.linear(latent,wk))
 freqs=model.precompute_freqs_cis(64,32768,args.original_seq_len,args.compress_rope_theta,args.rope_factor,args.beta_fast,args.beta_slow)
 model.apply_rotary_emb(key[..., -64:],freqs[:4:2]);ik,isk=model.fp4_act_quant(key,32,False);dequant=model.fp4_act_quant(key.clone(),32,True)
 rotated=latent.clone();model.apply_rotary_emb(rotated[..., -64:],freqs[:4:2]);kv,ks=model.fp4_act_quant(rotated,16,False,scale_dtype=torch.float8_e4m3fn)
 # Publication emits input-aligned rows; incomplete groups are all-zero.
 aligned=torch.zeros(5,512);aligned[1:4:2]=latent[0];alignedkey=torch.zeros(5,128);alignedkey[1:4:2]=dequant[0]
 compressed=torch.zeros(64*288,dtype=torch.uint8);compressed[:2*256]=kv.view(torch.uint8).flatten();compressed[64*256:64*256+2*32]=ks.view(torch.uint8).flatten()
 index_cache=torch.zeros(index_page_stride(64),dtype=torch.uint8);index_cache[:2*64]=ik.view(torch.uint8).flatten();index_cache[64*64:64*64+2*4]=isk.view(torch.uint8).flatten()
 inputs={'x':x.flatten(0,1),'input_ids':torch.arange(5,dtype=torch.int64),'positions':torch.arange(5,dtype=torch.int32),'valid':torch.ones(5,dtype=torch.int32),'slot_mapping':torch.arange(5,dtype=torch.int64),'seq_lens':torch.tensor([5],dtype=torch.int32),'cu_seqlens':torch.tensor([0,5],dtype=torch.int32),'page_table':torch.arange(256,dtype=torch.int32).unsqueeze(0),'compressor.2.lines':torch.zeros(1,1,dtype=torch.int32),'cos_sin':torch.view_as_real(freqs).flatten(-2)}
 outputs={stage.latent:aligned,'prefill.compress2.key_dequant':alignedkey};states={'compressed.2':compressed,'index_k.2':index_cache}
 def save(name,t):
  file=a.dump/(name+'.bin');file.write_bytes(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes());file.chmod(0o666)
 for group in (inputs,outputs,states):
  for name,t in group.items():save(name,t)
 names=[v['bind'][0]['tensor'] for v in buffers.values() if v['kind']=='weight'];shards=sorted({str(a.checkpoint/index[n]) for n in names})
 io={'vars':{'tokens':5,'seqs':1},'weights':shards,'inputs':{n:n+'.bin' for n in inputs},'outputs':{n:n+'.bin' for n in outputs},'state_outputs':{n:n+'.bin' for n in states}}
 for name,obj in [('manifest.json',manifest),('io.json',io)]:
  file=a.dump/name;file.write_text(json.dumps(obj));file.chmod(0o666)
 print('compressor/index publication fixture ready; calls',len(programs['probe']['calls']))

if __name__=='__main__':main()
