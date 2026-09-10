"""Real checkpoint context lowering through program_io against supplied DSpark methods."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import torch
from safetensors import safe_open

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kern_manifest import SCHEMA_VERSION
from dsv41.loading import Pieces,dense_scales
from dsv41.auxiliary.serving import build,Layout
from dsv41.draft_context import capture,publish


def main():
    p=argparse.ArgumentParser()
    for name in ('bindings','weights','inference','out','runner','cubins','auxiliary'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--modes',nargs='+',default=['prefill','decode','verify'])
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    artifacts=a.out/'cubins';artifacts.mkdir(exist_ok=True)
    for artifact in [*a.cubins.glob('*.cubin'),a.auxiliary]:
        dest=artifacts/artifact.name
        if not dest.exists():dest.symlink_to(artifact.resolve())
    sys.path.insert(0,str(a.inference));import model
    torch.set_num_threads(4);torch.manual_seed(57);torch.set_default_dtype(torch.bfloat16)
    raw=json.loads(a.bindings.read_text())['gpu']
    index=json.loads((a.weights/'model.safetensors.index.json').read_text())['weight_map']
    names=['mtp.0.main_proj.weight','mtp.0.main_proj.scale','mtp.0.main_norm.weight']
    for layer in range(3):
        names += [f'mtp.{layer}.attn.{suffix}' for suffix in ('wkv.weight','wkv.scale','kv_norm.weight')]
    sources={name:raw[name] for name in names}
    weights={}
    for name in names:
        with safe_open(a.weights/index[name],framework='pt',device='cpu') as f:weights[name]=f.get_tensor(name).cuda()
    for name,w in weights.items():
        if name.endswith('.weight') and name.removesuffix('.weight')+'.scale' in weights:
            w.scale=weights[name.removesuffix('.weight')+'.scale']
    def linear(prefix):return lambda x:model.linear(x,weights[prefix+'.weight'])
    def norm(prefix,width):
        n=model.RMSNorm(width,1e-20).cuda().bfloat16()
        n.weight=torch.nn.Parameter(weights[prefix+'.weight']);return n
    with torch.device('cuda'):
        freqs=model.precompute_freqs_cis(64,128,0,10000.,1.,32,1)
    for mode,rows,seqs in [('prefill',5,1),('decode',1,1),('verify',18,3)]:
        if mode not in a.modes:continue
        case=a.out/mode;case.mkdir(exist_ok=True)
        pieces=Pieces();packed,once,layouts=dense_scales(sources,pieces,cubin_dir=a.cubins)
        serving=build(a.auxiliary,Layout(max_seqs=seqs,max_tokens=max(rows,seqs*6),max_context=128))
        calls=[];buffers=dict(sources,**packed);inputs={}
        taps=[]
        for layer in (37,38,39):
            name=f'hc{layer}';x=torch.randn(rows,4,5120,device='cuda',dtype=torch.bfloat16)
            inputs[name]=x;taps.append(x.mean(1))
            buffers[name]={'dtype':'bf16','shape':[rows,4,5120],'kind':'input'}
            stage=capture(pieces,name,layer,mode=mode,rows=rows,capacity=rows,auxiliary_cubin=a.auxiliary)
            buffers.update(stage.buffers);calls+=stage.calls
        # This probe uses literal row count and observes state bytes as ordinary outputs.
        serving.rows=lambda _:rows
        stage=publish(pieces,serving,layouts,mode=mode,capacity=rows,
                      auxiliary_cubin=a.auxiliary,cubin_dir=a.cubins)
        buffers.update(stage.buffers);calls+=stage.calls
        positions=torch.arange(rows,dtype=torch.int32,device='cuda') if seqs==1 else torch.arange(6,device='cuda',dtype=torch.int32).repeat(seqs)
        slots=torch.arange(rows,device='cuda',dtype=torch.int64)
        if rows>1:slots[-1]=-1 # padded row must preserve state
        inputs[f'{mode}.position']=positions;inputs[f'{mode}.slot']=slots
        inputs['rope.window.interleaved']=torch.view_as_real(freqs).flatten(1)
        if mode=='verify':
            inputs[f'{mode}.request']=torch.arange(seqs,device='cuda',dtype=torch.int32).repeat_interleave(6)
            inputs[f'{mode}.starts']=torch.arange(0,rows+1,6,device='cuda',dtype=torch.int32)
            inputs['nacc']=torch.tensor([0,1,5],device='cuda',dtype=torch.int32)
        dtype_map={torch.int32:'i32',torch.int64:'i64',torch.float32:'f32',torch.bfloat16:'bf16',torch.uint8:'u8'}
        for name,x in inputs.items():buffers[name]={'dtype':dtype_map[x.dtype],'shape':list(x.shape),'kind':'input'}
        for op in pieces.ops.values():
            op['params']=['out buffer<u8>' if t=='inout state' else t for t in op['params']]
        for c in calls:
            for arg in c['args']:
                if 'state' in arg:
                    name=arg.pop('state');arg['buf']=name
                    buffers[name]={'dtype':'u8','shape':[128*528],'kind':'output'}
        for name in list(buffers):
            if name.endswith(('.taps','.main','.kv')):buffers[name]['kind']='output'
        manifest={'schema_version':SCHEMA_VERSION,'model':'dspark-context-probe','buffers':buffers,
                  'modules':pieces.modules,'ops':pieces.ops,'programs':{'probe':{'calls':once+calls}}}
        (case/'manifest.json').write_text(json.dumps(manifest,indent=2))
        cmd=[str(a.runner),'--manifest',str(case/'manifest.json'),'--cubins',str(artifacts),'--gpu','0','--weights',str(a.weights),'--graph','--iters','2']
        for name,x in inputs.items():
            path=case/(name+'.bin');path.write_bytes(x.cpu().contiguous().view(torch.uint8).numpy().tobytes());cmd+=['--in',f'{name}={path}']
        for name,spec in buffers.items():
            if spec['kind']=='output' or name.startswith('draft.window.'):cmd+=['--out',f'{name}={case/(name+".out")}']
        subprocess.run(cmd,check=True)
        def read(name,dtype,shape):return torch.frombuffer(bytearray((case/(name+'.out')).read_bytes()),dtype=dtype).reshape(shape).cuda()
        prefix=f'{mode}.draft_context'
        actual_taps=read(prefix+'.taps',torch.bfloat16,(rows,15360))
        torch.testing.assert_close(actual_taps,torch.cat(taps,-1),atol=0,rtol=0)
        stub=SimpleNamespace(main_proj=linear('mtp.0.main_proj'),main_norm=norm('mtp.0.main_norm',5120),
                             embed=lambda ids:torch.zeros(*ids.shape,5120,device='cuda',dtype=torch.bfloat16),
                             block_size=5,noise_token_id=0,hc_mult=4)
        _,expected_main=model.DSparkBlock.forward_embed(stub,torch.cat(taps,-1),torch.zeros(rows,device='cuda',dtype=torch.int64))
        actual_main=read(prefix+'.main',torch.bfloat16,(rows,5120))
        relative=lambda x,y:((x.float()-y.float()).square().sum()/y.float().square().sum()).item()
        error=relative(actual_main,expected_main);assert error<1e-4,error
        for layer in range(3):
            attention=SimpleNamespace(compress_ratio=0,window_size=128,rope_head_dim=64,freqs_cis=freqs,
                                      wkv=linear(f'mtp.{layer}.attn.wkv'),kv_norm=norm(f'mtp.{layer}.attn.kv_norm',512),
                                      window_kv_cache=torch.zeros(1,128,512,device='cuda',dtype=torch.bfloat16))
            # Official prefill path seeds exactly the context KV; verify expected
            # trajectories reuse its per-position values then select accepted rows.
            expected=[]
            for row in range(rows):
                attention.freqs_cis=freqs[positions[row]:]
                model.DSparkAttention.forward(attention,None,0,expected_main[row:row+1,None,:])
                expected.append(attention.window_kv_cache[0,0].clone())
            expected=torch.stack(expected)
            kv=read(f'{prefix}.{layer}.kv',torch.bfloat16,(rows,512))
            model.act_quant(kv,model.fp8_block_size,model.scale_fmt,model.scale_dtype,True)
            error=relative(kv,expected);assert error<3e-3,(layer,error)
            cache=read(f'draft.window.{layer}',torch.uint8,(128*528,))
            data=cache[:128*512].view(torch.float8_e4m3fn).float().reshape(128,512)
            scales=torch.exp2(cache[128*512:].float()-127).repeat_interleave(32,dim=0).reshape(128,512)
            dequant=(data*scales).bfloat16()
            keep=slots>=0
            if mode=='verify':keep &= torch.arange(rows,device='cuda')%6<inputs['nacc'].repeat_interleave(6)
            torch.testing.assert_close(dequant[slots[keep]],kv[keep],atol=0,rtol=0)
            untouched=torch.ones(128,device='cuda',dtype=torch.bool);untouched[slots[keep]]=False
            assert (cache[:128*512].reshape(128,512)[untouched]==0).all()
            assert (cache[128*512:].reshape(128,16)[untouched]==0).all()
            print(mode,layer,'official context relative squared error',error,'accepted cache exact',flush=True)


if __name__=='__main__':main()
