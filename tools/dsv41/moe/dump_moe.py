"""Materialize synthetic four-rank fixtures for actual Runtime peer replay."""
import json,pathlib,sys
import torch

def dump_before(path,states,counts,lay,modules,op):
 sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]));from kern_manifest import SCHEMA_VERSION
 for rank,s in enumerate(states):
  case=pathlib.Path(path)/f'rank{rank}';case.mkdir(parents=True,exist_ok=True);n=counts[rank];io={'inputs':{},'outputs':{'y':'y.bin'}}
  buffers={'y':{'dtype':'bf16','shape':[max(n,1),5120],'kind':'output'},'stats':{'dtype':'i32','shape':[lay['experts']//4],'kind':'carry'},'slab':{'dtype':'u8','shape':[lay['slab_bytes']],'kind':'carry','export':True},'slab_peers':{'dtype':'u64','shape':[4],'kind':'peer','of':'slab','group':'ep'}}
  def save(name,t):
   (case/(name+'.bin')).write_bytes(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes());io['inputs'][name]=name+'.bin'
  save('slab',s['slab'][:lay['offsets']['shared_l2']]);save('stats',s['stats'])
  names=['w1','sf1','w2','sf2','shared_w1','shared_sf1','shared_w2','shared_sf2']
  for name,t in zip(names,s['weights']):buffers[name]={'dtype':'i32' if 'sf' in name else 'i8' if name=='w2' else 'fp8e4m3' if name=='shared_w2' else 'u8','shape':list(t.shape),'kind':'input'};save(name,t)
  slab=lambda key:{'buf':'slab','offset':lay['offsets'][key]};b=lambda key:{'buf':key}
  args=[b('y'),b('stats'),{'i32':n},b('slab_peers'),{'rank':'ep'},slab('l1'),slab('l1_sf'),b('w1'),b('sf1'),slab('l2'),slab('l2_sf'),b('w2'),b('sf2'),slab('x'),slab('shared_x_sf'),b('shared_w1'),b('shared_sf1'),slab('shared_l2'),slab('shared_l2_sf'),b('shared_w2'),b('shared_sf2')]
  manifest={'schema_version':SCHEMA_VERSION,'model':'dsv41-ep4-replay','topology':{'groups':{'ep':4}},'vars':{},'states':{},'buffers':buffers,'modules':{k:v for k,v in modules.items() if k in {l['module'] for l in op['impl']['launches']}},'ops':{'moe':op},'programs':{'probe':{'calls':[{'op':'moe','args':args}]}}};(case/'manifest.json').write_text(json.dumps(manifest));(case/'io.json').write_text(json.dumps(io))

def dump_after(path,states):
 for rank,s in enumerate(states):(pathlib.Path(path)/f'rank{rank}'/'y.bin').write_bytes(s['y'].detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
