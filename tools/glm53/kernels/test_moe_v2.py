#!/usr/bin/env python3
"""Short single-GPU cubin A/B. No serving; no checkpoint or shared-file writes.
Run from the sglang venv with kserve_prod.sh LD_LIBRARY_PATH and CUDA_VISIBLE_DEVICES=0.
"""
import argparse
import ctypes as C
import json
import os
import pathlib
import re
import struct
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'tools'))

def guard():
    import shutil
    if shutil.which('tmux'):
        subprocess.run(['tmux', 'ls'], check=False)
        subprocess.run(['tmux', 'list-panes', '-a', '-F', '#{session_name} #{pane_current_command}'], check=False)
    subprocess.run(['nvidia-smi'], check=True)
    ps = subprocess.check_output(['ps', '-eo', 'comm,args'], text=True)
    for line in ps.splitlines():
        name = line.split()[0] if line.split() else ''
        if name in ('kern', 'kern-serve', 'kserve', 'kbench') or (name.startswith(('python','sglang','EngineCore')) and re.search(r'(integration|bench_decode|serve_sglang|sglang.launch_server|sglang::|sglang.srt)', line)):
            raise SystemExit('GPU test deferred: active integration/serving/bench: ' + line)
    panes=subprocess.check_output(['tmux','list-panes','-a','-F','#{session_name} #{pane_current_command}'],text=True) if shutil.which('tmux') else ''
    for pane in panes.splitlines():
        parts=pane.split()
        if len(parts)==2 and parts[0] in ('sgl','sglang','kserve','integration') and parts[1] not in ('sleep',):
            raise SystemExit('GPU test deferred: integration/server pane '+pane)
    used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    gpu=int(os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')[0])
    if int(used.splitlines()[gpu])>1024:
        raise SystemExit('GPU test deferred: selected GPU is occupied')
    util = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    if any(int(x) > 3 for x in util.split()):
        raise SystemExit('GPU test deferred: active GPU work')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batches', default='1,2,4,8,16')
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--report', type=pathlib.Path, default=HERE/'moe_v2_test.json')
    ap.add_argument('--replays', type=int, default=20)
    ap.add_argument('--timing-repeats', type=int, default=30)
    ap.add_argument('--zero-input', action='store_true', help='Tie/clamp/zero-amax test')
    args = ap.parse_args()
    guard()
    import torch
    from glm53 import ops_moe, ops_moe_v2, ops_common
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    cuda = C.CDLL('libcuda.so.1')
    def check(err):
        if err:
            msg=C.c_char_p(); cuda.cuGetErrorString(err,C.byref(msg))
            raise RuntimeError((err,msg.value))
    cache = {}
    def function(launch):
        path=ops_common.HAND_DIR/launch['cubin']
        if not path.exists(): path=ops_common.DUMP_DIR/launch['cubin']
        key=(str(path),launch['entry'])
        if key not in cache:
            mod=C.c_void_p(); check(cuda.cuModuleLoad(C.byref(mod),str(path).encode()))
            fn=C.c_void_p(); check(cuda.cuModuleGetFunction(C.byref(fn),mod,launch['entry'].encode()))
            smem=launch.get('shared_mem',0)
            if smem>48000: check(cuda.cuFuncSetAttribute(fn,8,smem))
            cache[key]=(mod,fn)
        return cache[key][1]
    def ev(e,b):
        if isinstance(e,int): return e
        if isinstance(e,str): return b
        if 'mul' in e: return ev(e['mul'][0],b)*ev(e['mul'][1],b)
        if 'ceil_div' in e: return (ev(e['ceil_div'][0],b)+e['ceil_div'][1]-1)//e['ceil_div'][1]
        raise ValueError(e)
    def run(launch,params,b,scratch=None):
        def argval(a):
            if 'param' in a: return params[a['param']].data_ptr()+a.get('offset',0)
            if 'scratch' in a: return scratch[a['scratch']].data_ptr()
            if 'i32' in a: return a['i32']
            if 'i64' in a: return a['i64']
            if 'f32' in a: return a['f32']
            if 'var' in a: return b
            if 'expr' in a: return ev(a['expr'],b)
            raise ValueError(a)
        vals=[]
        for ty,a in zip(launch['params'],launch['args']):
            if 'pack' in a:
                data=bytearray(a['pack']['size'])
                for f in a['pack']['fields']:
                    kind='f' if 'f32' in f else ('i' if 'i32' in f or f.get('width')==4 else 'q')
                    struct.pack_into('<'+kind,data,f['at'],argval(f))
                vals.append(C.create_string_buffer(bytes(data)))
            elif ty=='i32': vals.append(C.c_int32(argval(a)))
            elif ty=='f32': vals.append(C.c_float(argval(a)))
            else: vals.append(C.c_uint64(argval(a)))
        argv=(C.c_void_p*len(vals))(*[C.addressof(v) for v in vals])
        grid=[ev(v,b) for v in launch['grid']]
        check(cuda.cuLaunchKernel(function(launch),*grid,*launch['block'],launch.get('shared_mem',0),
                                 C.c_void_p(torch.cuda.current_stream().cuda_stream),argv,None))
    dt={'f32':torch.float32,'i32':torch.int32,'bf16':torch.bfloat16,'fp8e4m3':torch.float8_e4m3fn}
    newop=ops_moe_v2.ops()['moe_decode_v2']
    sc={n:torch.zeros(d['shape'],dtype=dt[d['dtype']],device='cuda') for n,d in newop['impl']['scratch'].items()}
    oldops=ops_moe.ops()
    for name in ('moe_w13','moe_w2'):
        oldops[name]['impl']['launches'][0]['args'][12]={'expr':{'mul':['seqs',9]}}
    def zeros(shape,dtype):return torch.zeros(shape,dtype=dtype,device='cuda')
    x=torch.randn((16,4096),device='cuda',dtype=torch.bfloat16)
    rw=torch.randn((288,4096),device='cuda',dtype=torch.bfloat16)*.03
    bias=torch.randn((288,),device='cuda')*.1
    # Shapes and scales match every rank; random expert tensors are not a model-quality test.
    w13=torch.randn((289,512,4096),device='cuda',dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    s13=torch.rand((289,4,32),device='cuda')*.03+.01
    w2=torch.randn((289,4096,256),device='cuda',dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    s2=torch.rand((289,32,2),device='cuda')*.06+.01
    valid=torch.ones((16,),device='cuda',dtype=torch.int32)
    out=zeros((16,4096),torch.bfloat16)
    if args.zero_input:
        x.zero_(); bias.zero_()
    ref={n:zeros(shape,dtype) for n,shape,dtype in [
        ('scores',(16,288),torch.float32),('weights',(16,9),torch.float32),('ids',(16,9),torch.int32),
        ('packed',(16,9),torch.int32),('sorted',(9216,),torch.int32),('experts',(144,),torch.int32),('npost',(1,),torch.int32),
        ('a8',(16,4096),torch.float8_e4m3fn),('as',(16,32),torch.float32),
        ('c1',(144,512),torch.bfloat16),('h',(144,256),torch.bfloat16),('h8',(144,256),torch.float8_e4m3fn),
        ('hs',(144,2),torch.float32),('c2',(144,4096),torch.bfloat16),('out',(16,4096),torch.bfloat16)]}
    stages=[('moe_router',[x,rw,ref['scores']]),('moe_topk',[ref['scores'],bias,ref['weights'],ref['ids'],ref['packed']]),
            ('moe_align',[ref['ids'],ref['sorted'],ref['experts'],ref['npost']]),('moe_quant_a',[x,ref['a8'],ref['as']]),
            ('moe_w13',[ref['a8'],w13,ref['c1'],ref['as'],s13,ref['weights'],ref['sorted'],ref['experts'],ref['npost']]),
            ('moe_silu',[ref['c1'],ref['h']]),('moe_quant_b',[ref['h'],ref['h8'],ref['hs']]),
            ('moe_w2',[ref['h8'],w2,ref['c2'],ref['hs'],s2,ref['weights'],ref['sorted'],ref['experts'],ref['npost']]),
            ('moe_sum_reduce',[ref['c2'],ref['out']])]
    params=[x,rw,bias,w13,s13,w2,s2,valid,out]
    def baseline(b):
        for name,ps in stages:
            for launch in oldops[name]['impl']['launches']:run(launch,ps,b)
    def fused(b):
        for launch in newop['impl']['launches']:run(launch,params,b,sc)
    # Load all modules before any CUDA graph capture.
    for name,_ in stages:
        for launch in oldops[name]['impl']['launches']:function(launch)
    for launch in newop['impl']['launches']:function(launch)
    def metric(a,b):
        af=a.float();bf=b.float(); dif=(af-bf).abs()
        return {'neq':int((af!=bf).sum()),'numel':a.numel(),'max_abs':float(dif.max()),
                'rms_rel':float(dif.square().mean().sqrt()/(bf.square().mean().sqrt()+1e-20)),
                'finite':bool(torch.isfinite(af).all())}
    result={'seed':args.seed,'build':json.loads((HERE/'moe_v2_build.json').read_text()),'cases':[]}
    for b in map(int,args.batches.split(',')):
        for live in sorted(set([b,max(1,b-1),0] + ([3,5,6] if b==8 else []))):
            valid.zero_(); valid[:live]=1
            baseline(b); fused(b); torch.cuda.synchronize()
            ids=sc['ids'][:live]; rid=ref['ids'][:live]
            case={'bucket':b,'live':live,'ids_equal':bool(torch.equal(ids,rid)),
                  'counter_zero':int(sc['route_counter'][0])==0,
                  'n_experts':int(sc['npost'][0])//32,
                  'pad_zero':bool((out[live:b]==0).all()),
                  'pad_ids_minus_one':bool((sc['ids'][live:b]==-1).all())}
            if live:
                for name in ['scores','weights','a8','as','c1','h8','hs']:
                    nr=live*9 if name in ('c1','h8','hs') else live
                    # v3 keeps as ([32 kb][32 rows]) and hs ([2][288]) transposed.
                    lhs=sc[name].T[:nr] if name in ('as','hs') else sc[name][:nr]
                    case[name]=metric(lhs,ref[name][:nr])
                case['out']=metric(out[:live],ref['out'][:live])
                isolated=dict(sc)
                for name in ('h8','weights'):isolated[name]=ref[name]
                hst=torch.zeros((2,288),dtype=torch.float32,device='cuda')
                hst[:,:144]=ref['hs'].T
                isolated['hs']=hst
                sc['w2_counts'].zero_()
                run(newop['impl']['launches'][2],params,b,isolated)
                torch.cuda.synchronize()
                case['w2_isolated']=metric(out[:live],ref['out'][:live])
                fused(b)
            # Assert alignment is a bijection over live pairs in expert-major order.
            nn=int(sc['npost'][0])//32
            aligned=sc['sorted'].reshape(-1,32)[:nn].cpu()
            experts=sc['experts'][:nn].cpu()
            pairs=aligned[aligned<b*9]
            case['align_bijection']=sorted(pairs.tolist())==list(range(live*9))
            flatids=sc['ids'].flatten().cpu()
            case['align_experts']=all(int(flatids[p])==int(experts[e]) for e in range(nn) for p in aligned[e] if p<b*9)
            # Capture actual raw cubin calls. Counters must reset over replay.
            expected=out.clone()
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):fused(b)
            for _ in range(args.replays):g.replay()
            torch.cuda.synchronize()
            case['replay_exact']=bool(torch.equal(out[:b],expected[:b]))
            # Warm-L2 diagnostic only. Full-model latency must be measured by owner.
            gb=torch.cuda.CUDAGraph()
            with torch.cuda.graph(gb):baseline(b)
            def timing(graph):
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.timing_repeats):graph.replay()
                end.record();end.synchronize()
                return start.elapsed_time(end)*1000/args.timing_repeats
            case['baseline_warm_us']=timing(gb)
            case['v2_warm_us']=timing(g)
            print(json.dumps(case),flush=True)
            result['cases'].append(case)
    # Change valid IN THE SAME CAPTURE: non-prefix holes, then full validity.
    b=max(map(int,args.batches.split(',')))
    valid.fill_(1); fused(b); torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):fused(b)
    valid[:b]=torch.arange(b,device='cuda')%2
    g.replay(); torch.cuda.synchronize()
    selected=valid[:b].bool()
    baseline(b); torch.cuda.synchronize()
    result['dynamic_valid']={'hole_pad_zero':bool((out[:b][~selected]==0).all()),
                             'hole_ids_minus_one':bool((sc['ids'][:b][~selected]==-1).all())}
    if selected.any():result['dynamic_valid']['out']=metric(out[:b][selected],ref['out'][:b][selected])
    valid.fill_(1); g.replay(); torch.cuda.synchronize()
    result['dynamic_valid']['restore_out']=metric(out[:b],ref['out'][:b])
    args.report.write_text(json.dumps(result,indent=2)+'\n')
    ok=all(c['ids_equal'] and c['counter_zero'] and c['pad_zero'] and c['pad_ids_minus_one'] and c['align_bijection'] and c['align_experts'] and c['replay_exact'] and (not c['live'] or (c['out']['finite'] and all(c[k]['neq']==0 for k in ('scores','weights','a8','as','c1','h8','hs','out','w2_isolated')))) for c in result['cases'])
    ok=ok and result['dynamic_valid']['hole_pad_zero'] and result['dynamic_valid']['hole_ids_minus_one'] and result['dynamic_valid']['restore_out']['neq']==0
    if 'out' in result['dynamic_valid']:ok=ok and result['dynamic_valid']['out']['neq']==0
    if not ok:raise SystemExit('A/B gate failed; see report')
if __name__=='__main__':main()
