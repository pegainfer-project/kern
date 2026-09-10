"""Dump op cases and replay using the actual kern program_io binary."""
import argparse,json,pathlib,subprocess,sys

def dump_case(path,modules,name,op,params):
 import torch
 sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]));from kern_manifest import SCHEMA_VERSION
 path=pathlib.Path(path);path.mkdir(parents=True,exist_ok=True);buffers={};args=[];io={'inputs':{},'outputs':{}}
 for i,(value,typ) in enumerate(zip(params,op['params'])):
  if isinstance(value,torch.Tensor):
   key=f'p{i}';dtype=typ.split('<')[1].split('>')[0];kind='output' if typ.startswith('out ') else 'input'
   buffers[key]={'dtype':dtype,'kind':kind,'shape':list(value.shape)};args.append({'buf':key});file=key+'.bin';(path/file).write_bytes(value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes());io['outputs' if kind=='output' else 'inputs'][key]=file
  else:args.append({typ:value})
 manifest={'schema_version':SCHEMA_VERSION,'model':'dsv41-op-replay','vars':{},'states':{},'buffers':buffers,'modules':{k:v for k,v in modules.items() if k in {l['module'] for l in op['impl']['launches']}},'ops':{name:op},'programs':{'probe':{'calls':[{'op':name,'args':args}]}}}
 (path/'manifest.json').write_text(json.dumps(manifest,indent=2));(path/'io.json').write_text(json.dumps(io))

def main():
 p=argparse.ArgumentParser();p.add_argument('--runner',type=pathlib.Path,required=True);p.add_argument('--cubins',type=pathlib.Path,required=True);p.add_argument('--cases',type=pathlib.Path,required=True);p.add_argument('--gpu',default='1');p.add_argument('--graph',action='store_true');a=p.parse_args()
 for case in sorted(a.cases.iterdir()):
  io=json.loads((case/'io.json').read_text());cmd=[str(a.runner),'--manifest',str(case/'manifest.json'),'--cubins',str(a.cubins),'--gpu',a.gpu]
  for key,file in io['inputs'].items():cmd+=['--in',f'{key}={case/file}']
  if io.get('once'):cmd+=['--program','load','--weights',str(case/'weights.safetensors')]
  for key in io['outputs']:cmd+=['--dump' if io.get('once') else '--out',f'{key}={case/(key+".kern.bin")}']
  if a.graph and not io.get('once'):cmd+=['--graph','--iters','3']
  subprocess.run(cmd,check=True)
  for key,file in io['outputs'].items():assert (case/file).read_bytes()==(case/(key+'.kern.bin')).read_bytes(),(case,key)
  print(case.name,'byte-exact kern replay',flush=True)
if __name__=='__main__':main()

def dump_once(path,modules,name,op,params):
 """Synthetic safetensors binding -> once transform -> carry dump."""
 import torch,struct
 sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]));from kern_manifest import SCHEMA_VERSION
 path=pathlib.Path(path);path.mkdir(parents=True,exist_ok=True);buffers={};args=[];header={};raw=bytearray();io={'inputs':{},'outputs':{}}
 dtypes={'i8':'I8','fp8e8m0':'F8_E8M0','fp8e4m3':'F8_E4M3','u8':'U8'}
 for i,(value,typ) in enumerate(zip(params,op['params'])):
  if not isinstance(value,torch.Tensor):args.append({typ:value});continue
  key=f'p{i}';dtype=typ.split('<')[1].split('>')[0];data=value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes();args.append({'buf':key})
  if typ.startswith('out '):
   buffers[key]={'dtype':dtype,'shape':list(value.shape),'kind':'carry'};(path/(key+'.bin')).write_bytes(data);io['outputs'][key]=key+'.bin'
  else:
   buffers[key]={'dtype':dtype,'shape':list(value.shape),'kind':'weight','bind':[{'tensor':key}]};header[key]={'dtype':dtypes[dtype],'shape':list(value.shape),'data_offsets':[len(raw),len(raw)+len(data)]};raw.extend(data)
 h=json.dumps(header,separators=(',',':')).encode();h+=b' '*((-len(h))%8);(path/'weights.safetensors').write_bytes(struct.pack('<Q',len(h))+h+raw)
 manifest={'schema_version':SCHEMA_VERSION,'model':'dsv41-once-replay','vars':{},'states':{},'buffers':buffers,'modules':{k:v for k,v in modules.items() if k in {l['module'] for l in op['impl']['launches']}},'ops':{name:op},'programs':{'load':{'once':True,'calls':[{'op':name,'args':args}]}}};(path/'manifest.json').write_text(json.dumps(manifest,indent=2));io['once']=True;(path/'io.json').write_text(json.dumps(io))


def gate_to_slab(path):
 """Collapse raw Gate output fixtures into one byte slab with call offsets."""
 path=pathlib.Path(path);manifest=json.loads((path/'manifest.json').read_text());io=json.loads((path/'io.json').read_text())
 idx=(path/'p3.bin').read_bytes();weights=(path/'p4.bin').read_bytes();data=idx+weights
 for name in ('p3','p4'):del manifest['buffers'][name]
 manifest['buffers']['slab']={'dtype':'u8','shape':[len(data)],'kind':'output'}
 args=manifest['programs']['probe']['calls'][0]['args'];args[3]={'buf':'slab','offset':0};args[4]={'buf':'slab','offset':len(idx)}
 io['outputs']={'slab':'slab.bin'};(path/'slab.bin').write_bytes(data)
 (path/'manifest.json').write_text(json.dumps(manifest));(path/'io.json').write_text(json.dumps(io))
