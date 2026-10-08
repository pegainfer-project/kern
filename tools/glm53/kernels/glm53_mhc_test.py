#!/usr/bin/env python3
"""CPU ABI check and short direct-cubin A/B. Does NOT run serving or edit gen.
Use --cpu for no GPU work. GPU mode refuses active shared runs. JSON to stdout.
"""
import argparse
import ast
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import struct
import subprocess
import sys
sys.dont_write_bytecode = True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'tools'))
from glm53 import ops_common, ops_mhc, ops_mhc_v2, ops_moe


def abi_check():
    new=ops_mhc_v2.ops()
    all_launches=[l for op in (new|ops_mhc.ops()).values() for l in op['impl']['launches']]
    all_launches += [ops_moe.ops()[k]['impl']['launches'][0] for k in ('mlp_quant_a','moe_quant_a')]
    result={}
    for l in all_launches:
        path=ops_common.HAND_DIR/l['cubin']
        if not path.exists():path=ops_common.DUMP_DIR/l['cubin']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==l['sha256']
        elf=subprocess.check_output([next(c for c in ('/usr/local/cuda-13.0/bin/cuobjdump','/usr/local/cuda-13.3/bin/cuobjdump','/usr/local/cuda/bin/cuobjdump') if Path(c).exists()),'-elf',str(path)],text=True)
        marker='\n.nv.info.'+l['entry']+'\n'
        assert marker in elf,(path,marker)
        section=elf.split(marker,1)[1].split('\n.nv.',1)[0]
        fields=re.findall(r'Ordinal : (0x[\da-f]+)\s+Offset\s+: (0x[\da-f]+)\s+Size\s+: (0x[\da-f]+)',section)
        got=sorted([(int(i,16),int(o,16),int(s,16)) for i,o,s in fields])
        expected=[]; off=0
        for i,t in enumerate(l['params']):
            size=4 if t in ('i32','f32') else (int(t[6:-1]) if t.startswith('bytes<') else 8)
            align=64 if size==128 else (8 if size>4 else 4)
            off=(off+align-1)//align*align
            expected.append((i,off,size));off+=size
        if 'hc_prenorm_gemm_impl' in l['entry']:
            expected=[(0,0,4),(1,112,128),(2,240,128),(3,368,128),(4,496,8)]
        assert got==expected,(l['entry'],got,expected)
        result[l['entry']]={'sha256':l['sha256'],'abi':got,'block':l['block'],'smem':l.get('shared_mem',0)}
    for p in (HERE/'glm53_mhc_build.py',HERE/'glm53_mhc_test.py',HERE.parent/'ops_mhc_v2.py'):
        ast.parse(p.read_text())
    return result


def manifest_check():
    from glm53 import gen
    original=gen.wire_allreduce
    results=[]
    try:
        for mode in ({'boundary':False},{},{'cross_layer':True},{'cross_layer':True,'bf16':True}):
            def hook(m,*args,**kw):
                original(m,*args,**kw);ops_mhc_v2.fuse_manifest(m,**mode)
            gen.wire_allreduce=hook
            m=gen.build(layers=45)
            r=subprocess.run([str(pathlib.Path(__file__).resolve().parents[3] / 'target' / 'release' / 'kern'),'verify','/dev/stdin'],
                             input=json.dumps(m),text=True,capture_output=True)
            if r.returncode:raise RuntimeError(r.stderr)
            results.append({'mode':mode,'calls':len(m['programs']['decode']['calls']),'verified':True})
    finally:
        gen.wire_allreduce=original
    return results


def guard():
    try:
        subprocess.run(['tmux','ls'],check=False,stdout=sys.stderr)
        panes=subprocess.check_output(['tmux','list-panes','-a','-F','#{session_name} #{pane_current_command}'],text=True)
        print(panes,file=sys.stderr)
    except FileNotFoundError:
        pass
    subprocess.run(['nvidia-smi'],check=True,stdout=sys.stderr)
    for line in subprocess.check_output(['ps','-eo','comm,args'],text=True).splitlines():
        parts=line.split()
        if not parts:continue
        if parts[0] in ('kern','kern-serve','kserve','kbench') or parts[0].startswith('sglang::') or (parts[0].startswith('python') and re.search(r'[/ ](integration|bench_decode|serve_sglang|test_moe|test_dsa)|sglang[.]launch_server',line)):
            raise SystemExit('GPU run deferred: '+line)
    util=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True)
    vis=os.environ.get('CUDA_VISIBLE_DEVICES')
    keep=set(range(len(util.split()))) if not vis else {int(x) for x in vis.split(',') if x.strip()}
    busy=[i for i,v in enumerate(util.split()) if i in keep and int(v)>3]
    if busy:raise SystemExit('GPU run deferred: active GPU utilization on visible '+str(busy))


class Driver:
    def __init__(self,torch):
        self.torch=torch; self.cuda=C.CDLL('libcuda.so.1');self.cache={};self.maps={}
    def check(self,e):
        if e:
            msg=C.c_char_p();self.cuda.cuGetErrorString(e,C.byref(msg));raise RuntimeError((e,msg.value))
    def fn(self,l):
        key=(l['cubin'],l['entry'])
        if key not in self.cache:
            path=ops_common.HAND_DIR/l['cubin']
            if not path.exists():path=ops_common.DUMP_DIR/l['cubin']
            m=C.c_void_p();self.check(self.cuda.cuModuleLoad(C.byref(m),str(path).encode()))
            f=C.c_void_p();self.check(self.cuda.cuModuleGetFunction(C.byref(f),m,l['entry'].encode()))
            if l.get('shared_mem',0)>48000:self.check(self.cuda.cuFuncSetAttribute(f,8,l['shared_mem']))
            self.cache[key]=(m,f)
        return self.cache[key][1]
    def ev(self,e,b):
        if isinstance(e,int):return e
        if isinstance(e,str):return b
        if 'mul' in e:return self.ev(e['mul'][0],b)*self.ev(e['mul'][1],b)
        if 'ceil_div' in e:return (self.ev(e['ceil_div'][0],b)+e['ceil_div'][1]-1)//e['ceil_div'][1]
        raise ValueError(e)
    def launch(self,l,p,b,sc=None):
        def val(a):
            if 'param' in a:
                v=p[a['param']]
                return (v.data_ptr() if hasattr(v,'data_ptr') else v)+a.get('offset',0)
            if 'scratch' in a:return sc[a['scratch']].data_ptr()
            for k in ('i32','i64','f32'):
                if k in a:return a[k]
            if 'var' in a:return b
            if 'expr' in a:return self.ev(a['expr'],b)
            raise ValueError(a)
        argv=[]
        for ty,a in zip(l['params'],l['args']):
            if 'pack' in a:
                data=bytearray(a['pack']['size'])
                for field in a['pack']['fields']:
                    if 'tensormap' in field:
                        tm=field['tensormap'];ptr=p[tm['param']].data_ptr()
                        key=(ptr,json.dumps(tm,sort_keys=True))
                        if key not in self.maps:
                            # CUtensorMap requires 64-byte alignment, size128.
                            raw=C.create_string_buffer(191);addr=(C.addressof(raw)+63)&~63
                            dims=(C.c_uint64*len(tm['dims']))(*tm['dims'])
                            strides=(C.c_uint64*len(tm['strides']))(*tm['strides'])
                            box=(C.c_uint32*len(tm['box']))(*tm['box'])
                            step=(C.c_uint32*len(tm['dims']))(*([1]*len(tm['dims'])))
                            types={'f32':7,'bf16':9,'tf32':11}
                            self.check(self.cuda.cuTensorMapEncodeTiled(C.c_void_p(addr),types[tm['dtype']],len(tm['dims']),C.c_void_p(ptr),dims,strides,box,step,0,{0:0,32:1,64:2,128:3}[tm['swizzle']],3,0))
                            self.maps[key]=C.string_at(addr,128)
                        data[field['at']:field['at']+128]=self.maps[key]
                    else:
                        fmt='f' if 'f32' in field else ('i' if 'i32' in field or field.get('width')==4 else 'q')
                        struct.pack_into('<'+fmt,data,field['at'],val(field))
                argv.append(C.create_string_buffer(bytes(data)))
            else:argv.append((C.c_int32 if ty=='i32' else C.c_float if ty=='f32' else C.c_uint64)(val(a)))
        args=(C.c_void_p*len(argv))(*[C.addressof(v) for v in argv])
        self.check(self.cuda.cuLaunchKernel(self.fn(l),*[self.ev(v,b) for v in l['grid']],*l['block'],l.get('shared_mem',0),C.c_void_p(self.torch.cuda.current_stream().cuda_stream),args,None))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cpu',action='store_true')
    ap.add_argument('--summary',action='store_true')
    ap.add_argument('--strict',action='store_true',help='Fail on any non-cross-layer bit mismatch')
    ap.add_argument('--batches',default='1,2,4,8,16')
    ap.add_argument('--seed',type=int,default=123)
    ap.add_argument('--replays',type=int,default=20)
    ap.add_argument('--real-layer',type=int,default=None)
    ap.add_argument('--zero',action='store_true')
    ap.add_argument('--cross-layer',action='store_true')
    args=ap.parse_args()
    abi=abi_check()
    if args.cpu:
        print(json.dumps({'abi':abi,'manifests':manifest_check()},indent=2));return
    guard()
    import torch
    torch.cuda.set_device(0);torch.manual_seed(args.seed)
    d=Driver(torch);old=ops_mhc.ops();new=ops_mhc_v2.ops();moe=ops_moe.ops()
    def z(shape,dtype=torch.float32):return torch.zeros(shape,dtype=dtype,device='cuda')
    def rand(shape,scale=1.):return (torch.randn(shape,device='cuda')*scale).to(torch.bfloat16)
    residual=rand((16,4,4096));hidden=rand((16,4096));fn=rand((24,16384),.02).float()
    hc_base=torch.randn(24,device='cuda')*.2;hc_scale=torch.tensor([.1,.1,.1],device='cuda')
    nw=rand((4096,),.03)+1
    if args.real_layer is not None:
        from safetensors import safe_open
        weights=Path(os.environ.get('GLM53_CHECKPOINT', 'weights/GLM-5.3-Flash'))
        ix=json.loads((weights/'model.safetensors.index.json').read_text())['weight_map']
        prefix=f'model.language_model.layers.{args.real_layer}.'
        def weight(n):
            with safe_open(weights/ix[prefix+n],framework='pt',device='cpu') as f:return f.get_tensor(prefix+n).cuda()
        fn=weight('hc_ffn_fn').reshape(24,16384).float()
        hc_base=weight('hc_ffn_base').float();hc_scale=weight('hc_ffn_scale').float();nw=weight('post_attention_layernorm.weight')
    fn16=fn.bfloat16();assert torch.equal(fn,fn16.float()),'BF16 widening precondition failed'
    prevcomb=torch.softmax(torch.randn((16,4,4),device='cuda'),dim=-1).reshape(16,16)
    prevpost=torch.sigmoid(torch.randn((16,4),device='cuda'))*2
    if args.zero:residual.zero_();hidden.zero_()
    def output():return {'r':z((16,4,4096),torch.bfloat16),'mul':z((8,16,24)),'sqr':z((8,16)),
                         'comb':z((16,16)),'post':z((16,4)),'x':z((16,4096),torch.bfloat16),
                         'q':z((16,4096),torch.float8_e4m3fn),'sf':z((512,)),'count':z((16,),torch.int32)}
    ref=output();test=output()
    def runop(ops,n,p,b,sc=None):
        for l in ops[n]['impl']['launches']:d.launch(l,p,b,sc)
    def baseline(b,layout=2):
        runop(old,'hc_fma',[prevcomb,residual,prevpost,hidden,fn,ref['mul'],ref['sqr'],ref['r'],b,8],b)
        runop(old,'hc_big_fuse8',[ref['mul'],ref['sqr'],hc_scale,hc_base,ref['r'],ref['post'],ref['comb'],ref['x'],nw,b],b)
        if layout:runop(moe,'mlp_quant_a' if layout==1 else 'moe_quant_a',[ref['x'],ref['q'],ref['sf']],b)
    def fused(b,layout=2,bf16=False,alias=False):
        pc=test['comb'] if alias else prevcomb;pp=test['post'] if alias else prevpost
        p=[pc,residual,pp,hidden,fn16 if bf16 else fn,hc_base,hc_scale,nw,test['r'],test['comb'],test['post'],test['x'],test['q'],test['sf'],b,layout]
        runop(new,'hc_boundary_bf16_v2' if bf16 else 'hc_boundary_f32_v2',p,b,test)
    for op in (old|new|{k:moe[k] for k in ('mlp_quant_a','moe_quant_a')}).values():
        for l in op['impl']['launches']:d.fn(l)
    def metric(a,b):
        af=a.float();bf=b.float();dif=(af-bf).abs()
        bytes_a=a.contiguous().view(torch.uint8).reshape(-1,a.element_size())
        bytes_b=b.contiguous().view(torch.uint8).reshape(-1,b.element_size())
        return {'neq':int((bytes_a!=bytes_b).any(dim=1).sum()),'value_neq':int((af!=bf).sum()),'n':a.numel(),'max_abs':float(dif.max()),
                'rms_rel':float(dif.square().mean().sqrt()/(bf.square().mean().sqrt()+1e-30)),
                'finite':bool(torch.isfinite(af).all())}
    def metrics(b,layout):
        ret={k:metric(test[k][:b],ref[k][:b]) for k in ('r','comb','post','x')}
        if layout:
            ret['q']=metric(test['q'][:b],ref['q'][:b])
            ret['sf']=metric(test['sf'].view(32,16)[:,:b],ref['sf'].view(32,16)[:,:b]) if layout==1 else metric(test['sf'][:b*32],ref['sf'][:b*32])
        ret['counter_zero']=bool((test['count']==0).all());return ret
    def graph(fn):
        fn();torch.cuda.synchronize()
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(args.replays):fn()
        return g
    def timing(fn):
        g=graph(fn);g.replay();torch.cuda.synchronize();times=[]
        for _ in range(7):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record();g.replay();end.record();end.synchronize();times.append(start.elapsed_time(end)*1000/args.replays)
        return statistics.median(times)
    result={'seed':args.seed,'zero':args.zero,'real_layer':args.real_layer,'abi':abi,'cases':[]}
    for b in map(int,args.batches.split(',')):
        for layout in (0,1,2):
            baseline(b,layout)
            for bf16 in (False,True):
                fused(b,layout,bf16);torch.cuda.synchronize()
                case={'b':b,'layout':layout,'bf16':bf16,'metrics':metrics(b,layout)}
                # Replays validate reset and lifetime with changed outputs.
                g=graph(lambda:fused(b,layout,bf16));g.replay();g.replay();torch.cuda.synchronize()
                case['replay']=metrics(b,layout)
                hidden.neg_();baseline(b,layout);g.replay();torch.cuda.synchronize()
                case['changed_input_replay']=metrics(b,layout)
                hidden.neg_();baseline(b,layout)
                # Prev/next mixes alias in gen.py. Test this separately.
                test['comb'].copy_(prevcomb);test['post'].copy_(prevpost)
                fused(b,layout,bf16,alias=True);torch.cuda.synchronize();case['alias']=metrics(b,layout)
                result['cases'].append(case)
        baseline(b,2)
        result['cases'].append({'timing_b':b,'old_fma_fuse_quant_us':timing(lambda:baseline(b,2)),
                                'boundary_f32_us':timing(lambda:fused(b,2,False)),
                                'boundary_bf16_us':timing(lambda:fused(b,2,True))})
        # Standalone epilogue preserves 64- and 8-split reducers. Synthetic
        # positive sqrsums and BF16 residual cover the independent fuse64 path.
        for ns in (8,64):
            pm=torch.randn((ns,16,24),device='cuda')*.1
            ps=torch.rand((ns,16),device='cuda')*(16384/ns)
            pitch=16 if ns==64 else b
            for layout in (1,2):
                runop(old,f'hc_big_fuse{ns}',[pm,ps,hc_scale,hc_base,residual,ref['post'],ref['comb'],ref['x'],nw,pitch],b)
                runop(moe,'mlp_quant_a' if layout==1 else 'moe_quant_a',[ref['x'],ref['q'],ref['sf']],b)
                runop(new,f'hc_big_fuse{ns}_q_v2',[pm,ps,hc_scale,hc_base,residual,test['post'],test['comb'],test['x'],nw,pitch,test['q'],test['sf'],layout],b)
                torch.cuda.synchronize()
                met=metrics(b,layout);met.pop('r')
                result['cases'].append({'b':b,'standalone_ns':ns,'layout':layout,'metrics':met})
        if args.cross_layer:
            # Current post -> DeepGEMM prenorm -> fuse64, not a torch surrogate.
            pm=z((64,16,24));ps=z((64,16))
            def cross_ref():
                runop(old,'hc_post',[prevcomb,residual,prevpost,hidden,ref['r'],b],b)
                runop(old,'hc_prenorm',[ref['r'],fn,pm,ps,16],b)
                runop(old,'hc_big_fuse64',[pm,ps,hc_scale,hc_base,ref['r'],ref['post'],ref['comb'],ref['x'],nw,16],b)
                runop(moe,'mlp_quant_a',[ref['x'],ref['q'],ref['sf']],b)
            cross_ref();fused(b,1,False);torch.cuda.synchronize()
            case={'b':b,'cross_layer':True,'metrics':metrics(b,1)}
            case['old_post_prenorm_fuse_quant_us']=timing(cross_ref)
            case['boundary_f32_us']=timing(lambda:fused(b,1,False))
            result['cases'].append(case)
        print('completed batch '+str(b),file=sys.stderr,flush=True)
    failures=[]
    for idx,case in enumerate(result['cases']):
        for stage in ('metrics','replay','changed_input_replay','alias'):
            for key,v in case.get(stage,{}).items():
                if isinstance(v,dict):
                    if not v['finite'] or (args.strict and not case.get('cross_layer') and v['neq']):
                        failures.append([idx,stage,key])
                elif key=='counter_zero' and not v:
                    failures.append([idx,stage,key])
    result['failures']=failures
    if args.summary:
        result.pop('abi')
        for case in result['cases']:
            for stage in ('metrics','replay','changed_input_replay','alias'):
                if stage in case:
                    case[stage]={k:v for k,v in case[stage].items() if not isinstance(v,dict) or v['neq'] or not v['finite']}
    print(json.dumps(result,indent=2),flush=True)
    if failures:raise SystemExit(2)
if __name__=='__main__':main()
