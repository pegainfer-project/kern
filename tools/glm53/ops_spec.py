"""GLM-5.3 MTP primitives and offline mining, isolated from decode/v2 ops.

CPU build: PYTHONPATH=tools <sglang-python> -m glm53.ops_spec build
GPU tests: PYTHONPATH=tools CUDA_VISIBLE_DEVICES=0 <sglang-python> -m glm53.ops_spec test
The production round is gated until all model-level integration checks pass.
See docs/glm53/mtp.md. No import initializes CUDA.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
ART = Path(os.environ.get('GLM53_SPEC_ARTIFACTS', 'glm53-artifacts/mtp')).resolve()
SRC = Path(__file__).with_name('kernels')
def sg():
    # The sglang source checkout's python/ dir (offline Triton mining only).
    return Path(os.environ['GLM53_SGLANG_SRC'])
DUMP = ROOT / 'dumped-kernels-glm53-sglang'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def remine():
    """No CUDA context. Real fused KDA flag is DISABLE_STATE_UPDATE, not
    STORE_FINAL_STATE (the latter belongs to a different recurrent kernel).
    Emit both flags from the same source/options; diff TTIR and pin ABI.
    """
    import ast
    import triton
    import triton.language as tl
    from triton.compiler.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    # Importing the sglang module calls device-availability probes at import.
    # Compile only its JIT function AST, retaining source locations for Triton.
    source_file = sg()/'sglang/kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py'
    tree=ast.parse(source_file.read_text(), filename=str(source_file))
    node=next(n for n in tree.body if isinstance(n, ast.FunctionDef) and
              n.name=='fused_sigmoid_gating_delta_rule_update_kernel')
    scope={'triton':triton,'tl':tl,'__name__':__name__}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source_file),'exec'),scope)
    fn=scope[node.name]
    live = dict(A_log='*fp32', a='*bf16', dt_bias='*fp32', q='*bf16',
                k='*bf16', v='*bf16', b='*bf16', o='*bf16', h0_source='*fp32',
                h0_indices='*i32', cu_seqlens='*i32')
    signature = dict(live)
    for n in ('softplus_beta', 'softplus_threshold', 'lower_bound', 'scale'):
        signature[n] = 'fp32'
    for n in ('stride_h0_source', 'cache_steps', 'T', 'stride_a', 'stride_q',
              'stride_k', 'stride_v', 'stride_b'):
        signature[n] = 'i32'
    constants = dict(
        intermediate_states_buffer=None, intermediate_state_indices=None,
        retrieve_parent_token_ptr=None, stride_retrieve_parent_token_seq=0,
        stride_retrieve_parent_token_token=0, NP2_T=4, B=1, H=8, HV=8,
        K=128, V=128, BK=128, BV=32, USE_INITIAL_STATE=True,
        USE_QK_L2NORM_IN_KERNEL=True, IS_VARLEN=True, IS_KDA=True,
        USE_LOWER_BOUND=True, DISABLE_STATE_UPDATE=True,
        CACHE_INTERMEDIATE_STATES=False, HAS_EAGLE_TREE_CUSTOM_ATTN_MASK=False,
        replayssm_rawv=None, replayssm_rawk=None, replayssm_g=None, replayssm_beta=None,
        stride_rawv_slot=0, stride_rawk_slot=0, stride_g_slot=0, stride_beta_slot=0,
        MAX_CACHE_LEN=0, CACHE_RING=False, SPLIT_N_HV_GRID=True, USE_GDC=True)
    assert set(signature) | set(constants) == set(fn.arg_names)
    for n in constants:
        signature[n] = 'constexpr'
    attrs = {(fn.arg_names.index(n),): [['tt.divisibility', 16]] for n in live}
    for n in ('stride_h0_source','cache_steps','stride_a','stride_q','stride_k','stride_v'):
        attrs[(fn.arg_names.index(n),)] = [['tt.divisibility',16]]
    result = {}
    for name, disable in (('spec_delta_verify', True), ('spec_delta_advance', False)):
        constants['DISABLE_STATE_UPDATE'] = disable
        source = ASTSource(fn, signature=signature, constexprs=constants, attrs=attrs)
        cc = triton.compile(source, target=GPUTarget('cuda', 90, 32),
                            options={'num_warps': 1, 'num_stages': 3, 'launch_pdl': True})
        for ext in ('cubin', 'ttir', 'ptx'):
            p = ART / f'{name}.{ext}'
            value = cc.asm[ext]
            p.write_bytes(value) if isinstance(value, bytes) else p.write_text(value)
        # Triton appends two scratch pointers to its live argument ABI.
        result[name] = {'sha256': sha(ART/f'{name}.cubin'), 'entry': cc.name,
                        'shared': cc.metadata.shared,
                        'runtime_args': [n for n in fn.arg_names if n not in constants]}
        print(name, result[name])
    source_file = sg()/'sglang/kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py'
    result['source_sha256'] = sha(source_file)
    result['source'] = str(source_file)
    (ART/'delta.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def build():
    ART.mkdir(parents=True, exist_ok=True)
    stamps = {}
    nvcc = os.environ.get('NVCC') or str(Path(os.environ.get('CUDA_HOME', '/usr/local/cuda')) / 'bin' / 'nvcc')
    cxx = os.environ.get('CUDAHOSTCXX', '/usr/bin/g++')
    for name in ('spec_round', 'spec_accept_carry_repair', 'spec_kda', 'spec_dsa', 'spec_ar', 'spec_head', 'spec_norm'):
        source = SRC/f'{name}.cu'; out = ART/f'{name}.cubin'
        cmd = [nvcc, '-cubin', '-arch=sm_90a',
               '-std=c++17', '-O3', '-ccbin', cxx, str(source), '-o', str(out)]
        subprocess.run(cmd, check=True)
        stamps[name] = {'source_sha256': sha(source), 'cubin_sha256': sha(out), 'command': cmd}
    stamps['delta'] = remine()
    (ART/'build.json').write_text(json.dumps(stamps, indent=2)+'\n')
    print('built', ART)


def launch(module, entry, params, args, grid, block=256):
    p = ART/f'{module}.cubin'
    return dict(cubin=str(p), sha256=sha(p), label=entry, entry=entry,
                params=params, args=args, grid=grid, block=[block, 1, 1])


def ops():
    """Raw generator ops; integrate after resolve_modules, before lower_wire.
    None of these ops modifies ops_kda*.py or any producer-owned op dictionary.
    """
    from .ops_common import a, var, i32, i64, f32
    bx, bo, ix, io = 'in buffer<bf16>', 'out buffer<bf16>', 'in buffer<i32>', 'out buffer<i32>'
    lx, lo, fx = 'in buffer<i64>', 'out buffer<i64>', 'in buffer<f32>'
    out = {}
    def add(name, module, params, grid, block=256):
        out[name] = {'params': params, 'impl': {'launches': [
            launch(module, name, params.copy(), [a(i) for i in range(len(params))], grid, block)]}}
    add('spec_uniform', 'spec_round', [ix,io,io,'i32'], ['seqs',1,1],32)
    add('spec_record_draft', 'spec_round', [lx,lo,'i32','i32'], ['seqs',1,1],32)
    add('spec_eh_norm', 'spec_norm', [bx,bx,bx,bx,bo], ['seqs',1,1],512)
    add('spec_add_norm', 'spec_norm', ['inout buffer<bf16>',bx,bx,bo], ['seqs',1,1],512)
    add('spec_pack_kda', 'spec_round', [bx,bx,bx,bo,bo,bo,ix,ix,'i32'], ['seqs',3,1])
    add('spec_splice', 'spec_round', [lx, lx, lo, 'i32'], ['seqs', 1, 1], 32)
    add('spec_accept', 'spec_round', [lx, lx, ix, lo, io, io, 'i32', 'i32'], [1,1,1], 32)
    add('spec_accept_carry_repair', 'spec_accept_carry_repair',
        [lx, lx, ix, bx, 'inout state', ix, lo, io, io, lo, io, 'i32', 'i32'],
        [1,1,1], 256)
    add('spec_pack', 'spec_round', [bx, bo, ix, ix, 'i32', 'i32', 'i32'], ['seqs',3,1])
    add('spec_carry_store', 'spec_round', [bx, 'inout state', ix, ix, ix, 'i32'], ['seqs',1,1])
    add('spec_carry_load', 'spec_round', ['in state', ix, bo], ['seqs',1,1])
    add('spec_draft_meta', 'spec_round', [ix, ix, ix, io, io, io, io, 'i32','i32','i32'], ['seqs',1,1],32)
    add('spec_repair_meta', 'spec_round', [lx, ix, ix, lo, io, 'i32'], ['seqs',1,1],32)
    add('spec_conv_verify', 'spec_kda', [bx, fx, 'in state', ix, bo, ix, 'i32'], ['seqs',12,1])
    add('spec_conv_advance', 'spec_kda', [bx, 'inout state', ix, ix, ix, 'i32'], ['seqs',12,1])
    add('spec_dsa_prep', 'spec_dsa', [ix,ix]+[io]*7+['i32']*3, ['tokens',1,1],128)
    add('spec_kv_store', 'spec_dsa', ['inout state', bx, ix, ix, 'i64','i64'], ['tokens',1,1],128)
    add('spec_kpool_update', 'spec_dsa', ['inout state','inout state',bx,bx,fx]+[ix]*5+['i32','i32','i64'], ['seqs',1,1],128)
    meta=json.loads((ART/'delta.json').read_text())
    for name, store in (('spec_delta_verify',False),('spec_delta_advance',True)):
        params=[fx,bx,fx,bx,bx,bo,'inout state' if store else 'in state',ix,ix,'i32']
        wire=['buffer']*3+['f32']*3+['buffer']*7+['i32','buffer','i32','f32']+['i32']*6+['i64']*2
        args=[a(0),a(1),a(2),f32(1),f32(20),f32(-5),a(3),a(3,2048),a(3,4096),a(4,6144),a(5),
              a(6),a(7),i32(131072),a(8),i32(0),f32(128**-.5),a(9),i32(1024),i32(3072),i32(3072),i32(3072),i32(3336),i64(0),i64(0)]
        kernel=launch(name,meta[name]['entry'],wire,args,[4,'seqs',8],32)
        kernel['shared_mem']=meta[name]['shared']
        out[name]={'params':params,'impl':{'launches':[kernel]}}
    return out


def gpu_check():
    """Check before CUDA init. A live bench/serve pane or process defers tests."""
    subprocess.run(['tmux','ls'],check=False)
    panes=subprocess.run(['tmux','list-panes','-a','-F','#S #{pane_current_command}'],
                         text=True,stdout=subprocess.PIPE,check=False).stdout
    print(panes,end='')
    subprocess.run(['nvidia-smi'],check=False)
    commands=subprocess.check_output(['ps','-eo','comm=,args='],text=True)
    for line in commands.splitlines():
        words=line.split()
        if not words: continue
        comm=words[0]
        if comm in ('kern','kern-run','kern-serve','kbench','kserve'):
            raise RuntimeError('defer GPU test: active bench/serve process')
        if comm.startswith('python') and ('sglang.launch_server' in line or 'integration' in line):
            raise RuntimeError('defer GPU test: active serving/integration process')
    for line in panes.splitlines():
        name, _, cmd=line.partition(' ')
        if name in ('kbench','kserve') and cmd not in ('bash','zsh','sh','sleep'):
            raise RuntimeError('defer GPU test: active bench/serve tmux pane')
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES','0')
    if not gpu.isdigit(): raise ValueError('tests need one numeric CUDA_VISIBLE_DEVICES')
    stats=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.used,utilization.gpu',
                                    '--format=csv,noheader,nounits'],text=True)
    mem,util=map(int,stats.strip().split(','))
    if mem>100 or util: raise RuntimeError(f'GPU {gpu} busy: {stats.strip()}')


class Driver:
    def __init__(self,t):
        self.t=t; self.lib=C.CDLL('libcuda.so.1'); self.modules=[]
        self.lib.cuModuleLoad.argtypes=[C.POINTER(C.c_void_p),C.c_char_p]
        self.lib.cuModuleGetFunction.argtypes=[C.POINTER(C.c_void_p),C.c_void_p,C.c_char_p]
        self.lib.cuLaunchKernel.argtypes=[C.c_void_p]+[C.c_uint]*7+[C.c_void_p]*3
    @staticmethod
    def ck(rc):
        if rc: raise RuntimeError(f'CUDA error {rc}')
    def module(self,p):
        m=C.c_void_p(); self.ck(self.lib.cuModuleLoad(C.byref(m),str(p).encode())); self.modules.append(m); return m
    def run(self,m,name,args,grid,block=256,shared=0):
        f=C.c_void_p(); self.ck(self.lib.cuModuleGetFunction(C.byref(f),m,name.encode()))
        vals=[C.c_uint64(v.data_ptr()) if hasattr(v,'data_ptr') else v for v in args]
        argv=(C.c_void_p*len(vals))(*[C.addressof(v) for v in vals])
        self.ck(self.lib.cuLaunchKernel(f,*grid,block,1,1,shared,
                                      C.c_void_p(self.t.cuda.current_stream().cuda_stream),argv,None))


def test():
    stamp=json.loads((ART/'build.json').read_text())
    for name in ('spec_round','spec_accept_carry_repair','spec_kda','spec_dsa','spec_ar','spec_head','spec_norm'):
        assert sha(SRC/f'{name}.cu')==stamp[name]['source_sha256'], 'rebuild stale source'
        assert sha(ART/f'{name}.cubin')==stamp[name]['cubin_sha256'], 'rebuild stale cubin'
    gpu_check()
    import torch as t
    t.cuda.init(); t.cuda.set_device(0); t.empty(1, device="cuda"); t.manual_seed(5302)
    D=Driver(t); I,J,F=C.c_int,C.c_int64,C.c_float
    dev='cuda'; bf=t.bfloat16
    mods={name:D.module(ART/f'{name}.cubin') for name in ('spec_round','spec_accept_carry_repair','spec_kda','spec_dsa','spec_delta_verify','spec_delta_advance','spec_head','spec_norm')}
    def tensor(v,dtype=t.int32): return t.tensor(v,device=dev,dtype=dtype)
    def rnd(shape,dtype=bf): return (t.randn(shape,device=dev)*.2).to(dtype)
    def bits(a,b): return t.equal(a.view(t.int16 if a.element_size()==2 else t.int32),b.view(t.int16 if b.element_size()==2 else t.int32))
    logs=[]
    for S in (1,2,4,8):
        # All prefix match cases, including first mismatch with a matching suffix.
        for pattern in range(4):
            draft=tensor([[11,12]]*S,t.int64); verify=tensor([[11 if pattern&1 else 21,12 if pattern&2 else 22,99]]*S,t.int64)
            anchor=tensor(list(range(S)),t.int64); ids=t.empty((S,3),device=dev,dtype=t.int64)
            valid=t.ones(3*S,device=dev,dtype=t.int32)
            D.run(mods['spec_round'],'spec_splice',[anchor,draft,ids,I(3)],(S,1,1),32)
            assert t.equal(ids,t.cat([anchor[:,None],draft],1))
            out=t.empty_like(verify); n=t.empty(S,device=dev,dtype=t.int32); cu=t.empty(S+1,device=dev,dtype=t.int32)
            D.run(mods['spec_round'],'spec_accept',[draft,verify,valid,out,n,cu,I(S),I(3)],(1,1,1),32)
            expect=1+(1 if pattern&1 else 0)+(1 if pattern==3 else 0)
            assert n.tolist()==[expect]*S and cu.tolist()==[expect*s for s in range(S+1)]
            assert t.equal(out,verify)
        # Heterogeneous nacc exposes wrong (uncompacted) cu_seqlens updates.
        n=tensor([(s%3)+1 for s in range(S)])
        cu=t.cat([tensor([0]),n.cumsum(0,dtype=t.int32)])
        lines=tensor([s*2+1 for s in range(S)])
        valid=t.ones(3*S,device=dev,dtype=t.int32)
        fused=rnd((3*S,3336)); weight=rnd((3072,4),t.float32)
        state=rnd((2*S+1,3,3072)); initial=state.clone(); conv=t.empty((3*S,3072),device=dev,dtype=bf)
        D.run(mods['spec_kda'],'spec_conv_verify',[fused,weight,state,lines,conv,valid,I(3)],(S,12,1))
        assert bits(state,initial), 'verify changed committed conv'
        # Compare with three sequential ORIGINAL Triton decode conv calls.
        oracle_mod=D.module(ROOT/'kernels-glm53-handwritten/module_conv_generic.cubin')
        oracle=initial.clone(); expected=t.empty_like(conv)
        for j in range(3):
            x=fused.reshape(S,3,3336)[:,j].contiguous(); y=t.empty((S,3072),device=dev,dtype=bf)
            D.run(oracle_mod,'_causal_conv1d_update_kernel',[x,weight,oracle,lines,x,y,I(S),J(0),J(0)],(S,12,1),128)
            expected.reshape(S,3,3072)[:,j].copy_(y)
        assert bits(conv,expected), f'conv differs S={S}: {(conv!=expected).sum().item()}'
        D.run(mods['spec_kda'],'spec_conv_advance',[fused,state,lines,n,valid,I(3)],(S,12,1))
        for s in range(S):
            ref=t.cat([initial[lines[s]],fused.reshape(S,3,3336)[s,:,:3072]],0)[n[s]:n[s]+3]
            assert bits(state[lines[s]],ref)
        # Verify/store recurrent kernels with identical inputs and state.
        al=rnd((8,),t.float32); dt=rnd((1024,),t.float32); forget=rnd((3*S,1024))
        ssm=rnd((2*S+1,8,128,128),t.float32); start=ssm.clone()
        cu3=t.arange(S+1,device=dev,dtype=t.int32)*3
        output=t.empty((3*S,1024),device=dev,dtype=bf); refout=t.empty_like(output)
        def delta(mod,a,q,ff,o,h,cc):
            # Strided q/k/v and b views. Pointers preserve 16-byte alignment.
            args=[al,a,dt,F(1),F(20),F(-5),q,q[:,1024:],q[:,2048:],ff[:,3072:],o,h,lines,I(131072),cc,I(0),F(128**-.5),I(3*S),I(1024),I(3072),I(3072),I(3072),I(3336),J(0),J(0)]
            D.run(mod,'fused_sigmoid_gating_delta_rule_update_kernel',args,(4,S,8),32,stamp['delta']['spec_delta_verify']['shared'])
        delta(mods['spec_delta_verify'],forget,conv,fused,output,ssm,cu3)
        assert bits(ssm,start), 'no-store delta changed committed SSM'
        reference=start.clone()
        delta(mods['spec_delta_advance'],forget,conv,fused,refout,reference,cu3)
        assert bits(output,refout), 'verify/store output mismatch'
        # A/B against the pinned original delta kernel, not only new twins.
        original=D.module(DUMP/'module_363.cubin'); oldstate=start.clone(); oldout=t.empty_like(output)
        delta(original,forget,conv,fused,oldout,oldstate,cu3)
        assert bits(output,oldout), 'new delta output differs from pinned original'
        assert bits(reference,oldstate), 'new store state differs from pinned original'
        packed=[]
        for src,cols,pitch in ((fused,3336,3336),(conv,3072,3072),(forget,1024,1024)):
            dst=t.empty_like(src)
            D.run(mods['spec_round'],'spec_pack',[src,dst,n,cu,I(cols),I(pitch),I(3)],(S,3,1))
            ref=t.cat([src[s*3:s*3+int(n[s])] for s in range(S)])
            assert bits(dst[:int(cu[-1])],ref)
            packed.append(dst)
        ff,qq,aa=packed
        advanced=start.clone(); unused=t.empty_like(output)
        delta(mods['spec_delta_advance'],aa,qq,ff,unused,advanced,cu)
        # Independent per-sequence prefix oracle, including untouched spare lines.
        gold=start.clone()
        for s in range(S):
            cc=tensor([0]*S+[0])
            counts=tensor([int(n[s]) if i==s else 0 for i in range(S)])
            cc=t.cat([tensor([0]),counts.cumsum(0,dtype=t.int32)])
            # Put s's inputs at the compact start, zero-length peers touch no rows.
            temp=t.empty_like(output)
            delta(original,forget[s*3:].contiguous(),conv[s*3:].contiguous(),fused[s*3:].contiguous(),temp,gold,cc)
        assert bits(advanced,gold), 'advance prefix state mismatch'
        # Carry follows pool lines, independent of batch slot changes.
        h=rnd((3*S,4096)); carry=t.zeros((2*S+1,4096),device=dev,dtype=bf)
        D.run(mods['spec_round'],'spec_carry_store',[h,carry,lines,n,valid,I(3)],(S,1,1))
        restored=t.empty((S,4096),device=dev,dtype=bf)
        D.run(mods['spec_round'],'spec_carry_load',[carry,lines,restored],(S,1,1))
        assert bits(restored,t.stack([h[3*s+int(n[s])-1] for s in range(S)]))
        logs.append({'seqs':S,'splice_accept_pack_conv_delta_carry':'PASS'})
        t.cuda.synchronize()
    accept_carry_repair_test(t,D,mods)
    dsa_test(t,D,mods)
    head_norm_test(t,D,mods)
    (ART/'test.json').write_text(json.dumps({'build_sha256':sha(ART/'build.json'),'results':logs,'head_eh_metadata_graph':'PASS','add_norm':'see norm-test.json'},indent=2)+'\n')
    print(json.dumps(logs,indent=2))



def accept_carry_repair_test(t,D,mods):
    """Forced nacc 1/2/3 A/B for the merged accept/carry/repair control CTA."""
    I=C.c_int
    S=8
    # Match each prefix exactly, then force the requested first mismatch.
    want=t.tensor([1,2,3,1,2,3,1,2],device='cuda',dtype=t.int32)
    drafts=t.tensor([[100+s,200+s] for s in range(S)],device='cuda',dtype=t.int64)
    verify=t.empty((S,3),device='cuda',dtype=t.int64)
    for s in range(S):
        a=int(want[s])
        vals=[int(drafts[s,0]),int(drafts[s,1]),900+s]
        if a < 2: vals[0] += 17
        if a < 3: vals[1] += 19
        verify[s]=t.tensor(vals,device='cuda',dtype=t.int64)
    valid=t.ones(S*3,device='cuda',dtype=t.int32)
    lines=t.arange(1,S+1,device='cuda',dtype=t.int32)
    h=(t.randn((S*3,4096),device='cuda')*.2).to(t.bfloat16)
    state0=(t.randn((S+1,4096),device='cuda')*.2).to(t.bfloat16)
    fused_state=state0.clone()
    tokens=t.empty((S*3,),device='cuda',dtype=t.int64)
    nacc=t.empty(S,device='cuda',dtype=t.int32)
    cu=t.empty(S+1,device='cuda',dtype=t.int32)
    repair_ids=t.empty((S*3,),device='cuda',dtype=t.int64)
    repair_valid=t.empty((S*3,),device='cuda',dtype=t.int32)
    D.run(mods['spec_accept_carry_repair'],'spec_accept_carry_repair',
          [drafts,verify,valid,h,fused_state,lines,tokens,nacc,cu,repair_ids,repair_valid,I(S),I(3)],
          (1,1,1),256)
    t.cuda.synchronize()
    assert t.equal(nacc,want), f'forced nacc mismatch: {nacc.tolist()}'
    expected_cu=t.cat([t.zeros(1,device='cuda',dtype=t.int32),want.cumsum(0)])
    assert t.equal(cu,expected_cu), f'compact cu mismatch: {cu.tolist()} vs {expected_cu.tolist()}'
    assert t.equal(tokens,verify.reshape(-1)), 'round token copy mismatch'
    expected_mask=t.stack([t.arange(3,device='cuda') < (want[s]-1) for s in range(S)]).reshape(-1).to(t.int32)
    assert t.equal(repair_ids,verify.reshape(-1)) and t.equal(repair_valid,expected_mask), 'repair metadata mismatch'

    # Run the old one-row carry store sequentially. Compare in FP32 so the
    # acceptance gate is explicit even if the input generator changes dtype.
    ref_state=state0.clone()
    for s in range(S):
        one_h=h[s*3:(s+1)*3].contiguous()
        one_n=want[s:s+1].contiguous()
        one_v=valid[s*3:s*3+3].contiguous()
        one_line=lines[s:s+1].contiguous()
        D.run(mods['spec_round'],'spec_carry_store',
              [one_h,ref_state,one_line,one_n,one_v,I(3)],(1,1,1))
    t.cuda.synchronize()
    err=(fused_state.float()-ref_state.float()).abs().max().item()
    den=ref_state.float().abs().max().item()
    rel=err/max(den,1e-12)
    assert rel <= 1e-3, f'carry fused vs sequential rel err {rel}'
    for s in range(S):
        assert t.equal(fused_state[lines[s]],h[s*3+int(want[s])-1]), f'off-by-one carry s={s}'
    print(f'accept/carry/repair forced nacc={{1,2,3}}: PASS (fp32 rel_err={rel:.3g})')


def dsa_test(t,D,mods):
    I,J=C.c_int,C.c_int64
    def ten(v): return t.tensor(v,device='cuda',dtype=t.int32)
    # All close-boundary residues, page crossing, ring wrap, padding and S=8.
    for S in (1,2,4,8):
        for start in (0,1,2,3,6,7,254,255,256,2047,2048):
            COLUMNS=16; ST=COLUMNS*256; PAGE=12*8448
            positions=ten([[start+j for j in range(3)] for s in range(S)]).flatten()
            bt=t.arange(S*COLUMNS,device='cuda',dtype=t.int32).reshape(S,COLUMNS)
            rb=t.empty((3*S,COLUMNS),device='cuda',dtype=t.int32)
            outs=[t.empty(3*S,device='cuda',dtype=t.int32) for _ in range(4)]
            sl=t.full((3*S,ST),-1,device='cuda',dtype=t.int32); cu=t.empty(3*S+1,device='cuda',dtype=t.int32)
            D.run(mods['spec_dsa'],'spec_dsa_prep',[positions,bt,rb,*outs,sl,cu,I(3),I(COLUMNS),I(ST)],(3*S,1,1),128)
            lengths,pool,ctx,lens=outs
            assert t.equal(lengths,positions+1)
            assert t.equal(pool,lengths//4) and t.equal(ctx,t.maximum(pool,t.ones_like(pool)))
            assert t.equal(lens,t.minimum(pool*4,t.full_like(pool,2048))+lengths%4)
            assert cu.tolist()==list(range(3*S+1)) and t.equal(rb,bt.repeat_interleave(3,0))
            for r in range(3*S):
                p=t.arange(start+r%3+1,device='cuda')
                assert t.equal(sl[r,:len(p)],bt[r//3,p//256]*256+p%256)
            lines=ten(list(range(1,S+1))); valid=t.ones(3*S,device='cuda',dtype=t.int32)
            if S>1: valid[-3:]=0
            key=t.randn(3*S,128,device='cuda').bfloat16(); score=t.randn_like(key)
            ape=t.randn(4,128,device='cuda')
            idx=t.zeros((S*COLUMNS,PAGE),device='cuda',dtype=t.uint8); ref=idx.clone()
            tail=t.randn((S+1,2048),device='cuda').bfloat16(); initial=tail.clone(); tailref=tail.clone()
            # Test last layer offset. Sentinel bytes check the physical page pitch.
            off=11*8448; ib=idx[:,off:]; ir=ref[:,off:]
            D.run(mods['spec_dsa'],'spec_kpool_update',[ib,tail,key,score,ape,bt,lines,positions,lengths,valid,I(COLUMNS),I(3),J(PAGE)],(S,1,1),128)
            for j in range(3):
                kj=key.reshape(S,3,128)[:,j].contiguous(); sj=score.reshape(S,3,128)[:,j].contiguous()
                pj=positions.reshape(S,3)[:,j].contiguous(); vj=valid.reshape(S,3)[:,j].contiguous(); lj=pj+1
                D.run(mods['spec_dsa'],'spec_kpool_update',[ir,tailref,kj,sj,ape,bt,lines,pj,lj,vj,I(COLUMNS),I(1),J(PAGE)],(S,1,1),128)
            assert t.equal(idx,ref) and t.equal(tail,tailref), f'ordered kpool S={S} start={start}'
            assert t.count_nonzero(idx[:,:off]).item()==0, 'wrote another layer'
            if S>1: assert t.equal(tail[-1],initial[-1]), 'padding modified tail'
    t.cuda.synchronize()
    print('DSA prep + sequential kpool equivalence: PASS')


def head_norm_test(t,D,mods):
    """Queued device tests. Oracle imports happen only AFTER gpu_check()."""
    import importlib.util
    from flashinfer.norm import fused_add_rmsnorm
    source=sg()/'sglang/kernels/ops/layernorm/fused_eh_norm.py'
    module_spec=importlib.util.spec_from_file_location('mtp_eh_oracle',source)
    oracle=importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(oracle)
    norm_errors=[]
    for rows in (1,2,3,4,8,12,16,24,32):
        logits=t.randn((8,rows,19360),device='cuda',dtype=t.bfloat16)
        # Equal maxima across ranks and inside a shard must select lowest ID.
        logits[:,0,:]=-2
        logits[0,0,5]=7; logits[0,0,37]=7; logits[7,0,0]=7
        if rows>1: logits[:,1,:]=-float('inf')
        choices=t.empty((8,rows,4),device='cuda',dtype=t.bfloat16)
        for rank in range(8):
            D.run(mods['spec_head'],'spec_head_local',[logits[rank],choices[rank],C.c_int(rank)],(rows,1,1),1024)
        tokens=t.empty(rows,device='cuda',dtype=t.int64)
        D.run(mods['spec_head'],'spec_head_global',[choices,tokens,C.c_int(rows)],(rows,1,1),32)
        gold=logits.permute(1,0,2).reshape(rows,-1).float().argmax(-1)
        assert t.equal(tokens,gold), f'head/ties M={rows}'
        x=t.randn((rows,4096),device='cuda',dtype=t.bfloat16)
        prev=t.randn_like(x); ew=t.randn(4096,device='cuda',dtype=t.bfloat16); hw=t.randn_like(ew)
        eh=t.empty((rows,8192),device='cuda',dtype=t.bfloat16)
        D.run(mods['spec_norm'],'spec_eh_norm',[x,prev,ew,hw,eh],(rows,1,1),512)
        ref=oracle.fused_eh_norm(x,prev,ew,hw,1e-5)
        assert t.equal(eh,ref), f'EH oracle M={rows}, unequal={(eh!=ref).sum().item()}'
        residual=prev.clone(); out=t.empty_like(x); gx=x.clone(); gr=prev.clone()
        D.run(mods['spec_norm'],'spec_add_norm',[residual,x,ew,out],(rows,1,1),512)
        fused_add_rmsnorm(gx,gr,ew,1e-5,enable_pdl=False)
        assert t.equal(residual,gr), f'add residual M={rows}'
        unequal=(out!=gx).sum().item()
        ulp=(out.view(t.int16).to(t.int32)-gx.view(t.int16).to(t.int32)).abs().max().item()
        norm_errors.append({'rows':rows,'unequal':unequal,'elements':out.numel(),'max_bf16_ulp':ulp})
        # Draft norms are numerical A/B, not target exactness. Record even
        # one-ULP differences; full draft/head/acceptance remains a release gate.
        assert ulp<=1 and unequal<=out.numel()*.001, f'add norm exceeds tolerance: {norm_errors[-1]}'
    (ART/'norm-test.json').write_text(json.dumps(norm_errors,indent=2)+'\n')
    print('head/ties + EH + residual: bitwise PASS; add RMS: <=1 BF16 ULP, <=0.1% mismatch')
    print(json.dumps(norm_errors))
    # One captured data-flow DAG; counts vary between replays, including pad=0.
    S=8
    draft=t.tensor([[11,12]]*S,device='cuda',dtype=t.int64)
    verify=t.empty((S,3),device='cuda',dtype=t.int64)
    valid=t.ones((S,3),device='cuda',dtype=t.int32); valid[-1]=0
    out=t.empty_like(verify); n=t.empty(S,device='cuda',dtype=t.int32)
    cu=t.empty(S+1,device='cuda',dtype=t.int32)
    ff=t.randn((S*3,3336),device='cuda',dtype=t.bfloat16)
    qq=t.randn((S*3,3072),device='cuda',dtype=t.bfloat16)
    aa=t.randn((S*3,1024),device='cuda',dtype=t.bfloat16)
    packed=[t.empty_like(x) for x in (ff,qq,aa)]
    def run():
        D.run(mods['spec_round'],'spec_accept',[draft,verify,valid,out,n,cu,C.c_int(S),C.c_int(3)],(1,1,1),32)
        D.run(mods['spec_round'],'spec_pack_kda',[ff,qq,aa,*packed,n,cu,C.c_int(3)],(S,3,1))
    stream=t.cuda.Stream(); stream.wait_stream(t.cuda.current_stream())
    with t.cuda.stream(stream): run()
    t.cuda.current_stream().wait_stream(stream); t.cuda.synchronize()
    graph=t.cuda.CUDAGraph()
    with t.cuda.graph(graph,stream=stream): run()
    for counts in ([1]*S,[2]*S,[3]*S,[1,3,2,1,3,2,1,0]):
        wanted=list(counts); wanted[-1]=0
        verify.copy_(t.tensor([[11 if a>=2 else 21,12 if a==3 else 22,99] for a in wanted],device='cuda'))
        graph.replay()
        assert n.tolist()==wanted
        assert cu.tolist()==[sum(wanted[:j]) for j in range(S+1)]
        for x,y in zip((ff,qq,aa),packed):
            gold=t.cat([x[3*j:3*j+a] for j,a in enumerate(wanted)])
            assert t.equal(y[:gold.shape[0]],gold)
    # Shifted metadata must use sequence-major anchors and local page IDs.
    bt=t.arange(S*16,device='cuda',dtype=t.int32).reshape(S,16)+1
    positions=t.tensor([[p,p+1,p+2] for p in (0,1,3,4,255,256,2048,4090)],device='cuda',dtype=t.int32)
    pos,lens,slot,ok=[t.empty(S,device='cuda',dtype=t.int32) for _ in range(4)]
    for step in (0,1):
        D.run(mods['spec_round'],'spec_draft_meta',[positions,valid,bt,pos,lens,slot,ok,C.c_int(3),C.c_int(step),C.c_int(16)],(S,1,1),32)
        for j,p in enumerate(positions[:,0].tolist()):
            pp=max(p-1+step,0); active=j!=S-1 and p-1+step>=0
            assert pos[j].item()==pp and lens[j].item()==pp+1 and ok[j].item()==active
            assert slot[j].item()==(bt[j,pp>>8].item()*256+(pp&255) if active else 0)
    print('static graph accept/pack heterogeneous counts + shifted draft metadata: PASS')


def kn_test(R, D6):
    """GPU value validation of the rows-parameterized round ABI at width R.

    Same oracles as test() (k=2): the generic conv cubin, the pinned
    module_363 delta kernel, and per-sequence prefix oracles. Proves the ABI
    change generalizes past rows=3; k=2 bitwise equivalence is covered by
    test() with the same cubins. R=round width (tokens/seq), D6=draft count.
    """
    stamp=json.loads((ART/'build.json').read_text())
    for name in ('spec_round','spec_accept_carry_repair','spec_kda','spec_dsa','spec_ar','spec_head','spec_norm'):
        assert sha(SRC/f'{name}.cu')==stamp[name]['source_sha256'], 'rebuild stale source'
        assert sha(ART/f'{name}.cubin')==stamp[name]['cubin_sha256'], 'rebuild stale cubin'
    gpu_check()
    import torch as t
    t.cuda.init(); t.cuda.set_device(0); t.empty(1, device="cuda"); t.manual_seed(9021)
    D=Driver(t); I,J,F=C.c_int,C.c_int64,C.c_float
    dev='cuda'; bf=t.bfloat16
    mods={name:D.module(ART/f'{name}.cubin') for name in ('spec_round','spec_accept_carry_repair','spec_kda','spec_dsa','spec_delta_verify','spec_delta_advance','spec_head','spec_norm')}
    def tensor(v,dtype=t.int32): return t.tensor(v,device=dev,dtype=dtype)
    def rnd(shape,dtype=bf): return (t.randn(shape,device=dev)*.2).to(dtype)
    def bits(a,b): return t.equal(a.view(t.int16 if a.element_size()==2 else t.int32),b.view(t.int16 if b.element_size()==2 else t.int32))
    logs=[]
    for S in (1,2,4):
        # splice: ids = [anchor, d0..d5]
        draft=tensor([[11,12,13,14,15,16]]*S,t.int64)
        anchor=tensor(list(range(S)),t.int64)
        ids=t.empty((S,R),device=dev,dtype=t.int64)
        D.run(mods['spec_round'],'spec_splice',[anchor,draft,ids,I(R)],(S,1,1),32)
        assert t.equal(ids,t.cat([anchor[:,None],draft],1)), f'splice S={S}'
        # accept: first mismatch at each draft column, full match, invalid seq.
        for miss in range(-1,D6):
            row=[[(11+j if j!=miss else 900+j) for j in range(D6)]+[99] for s in range(S)]
            verify=tensor(row,t.int64)
            out=t.empty_like(verify); n=t.empty(S,device=dev,dtype=t.int32); cu=t.empty(S+1,device=dev,dtype=t.int32)
            valid=t.ones(R*S,device=dev,dtype=t.int32)
            D.run(mods['spec_round'],'spec_accept',[draft,verify,valid,out,n,cu,I(S),I(R)],(1,1,1),32)
            exp=R if miss<0 else miss+1
            assert n.tolist()==[exp]*S, f'accept nacc miss={miss} S={S}: {n.tolist()}'
            assert cu.tolist()==[exp*s for s in range(S+1)]
            assert t.equal(out,verify)
        valid=t.zeros(R*S,device=dev,dtype=t.int32)
        D.run(mods['spec_round'],'spec_accept',[draft,verify,valid,out,n,cu,I(S),I(R)],(1,1,1),32)
        assert n.tolist()==[0]*S and cu.tolist()==[0]*(S+1), 'invalid seq nacc'
        # Heterogeneous nacc exposes wrong (uncompacted) cu_seqlens updates.
        n=tensor([(s%R)+1 for s in range(S)])
        cu=t.cat([tensor([0]),n.cumsum(0,dtype=t.int32)])
        lines=tensor([s*2+1 for s in range(S)])
        valid=t.ones(R*S,device=dev,dtype=t.int32)
        fused=rnd((R*S,3336)); weight=rnd((3072,4),t.float32)
        state=rnd((2*S+1,3,3072)); initial=state.clone(); conv=t.empty((R*S,3072),device=dev,dtype=bf)
        D.run(mods['spec_kda'],'spec_conv_verify',[fused,weight,state,lines,conv,valid,I(R)],(S,12,1))
        assert bits(state,initial), 'verify changed committed conv'
        # Compare with seven sequential ORIGINAL Triton decode conv calls.
        oracle_mod=D.module(ROOT/'kernels-glm53-handwritten/module_conv_generic.cubin')
        oracle=initial.clone(); expected=t.empty_like(conv)
        for j in range(R):
            x=fused.reshape(S,R,3336)[:,j].contiguous(); y=t.empty((S,3072),device=dev,dtype=bf)
            D.run(oracle_mod,'_causal_conv1d_update_kernel',[x,weight,oracle,lines,x,y,I(S),J(0),J(0)],(S,12,1),128)
            expected.reshape(S,R,3072)[:,j].copy_(y)
        assert bits(conv,expected), f'conv differs S={S}: {(conv!=expected).sum().item()}'
        D.run(mods['spec_kda'],'spec_conv_advance',[fused,state,lines,n,valid,I(R)],(S,12,1))
        for s in range(S):
            ref=t.cat([initial[lines[s]],fused.reshape(S,R,3336)[s,:,:3072]],0)[n[s]:n[s]+3]
            assert bits(state[lines[s]],ref), f'conv advance s={s}'
        # Verify/store recurrent kernels with identical inputs and state.
        al=rnd((8,),t.float32); dt=rnd((1024,),t.float32); forget=rnd((R*S,1024))
        ssm=rnd((2*S+1,8,128,128),t.float32); start=ssm.clone()
        cuR=t.arange(S+1,device=dev,dtype=t.int32)*R
        output=t.empty((R*S,1024),device=dev,dtype=bf); refout=t.empty_like(output)
        def delta(mod,a,q,ff,o,h,cc):
            # Strided q/k/v and b views. Pointers preserve 16-byte alignment.
            args=[al,a,dt,F(1),F(20),F(-5),q,q[:,1024:],q[:,2048:],ff[:,3072:],o,h,lines,I(131072),cc,I(0),F(128**-.5),I(R*S),I(1024),I(3072),I(3072),I(3072),I(3336),J(0),J(0)]
            D.run(mod,'fused_sigmoid_gating_delta_rule_update_kernel',args,(4,S,8),32,stamp['delta']['spec_delta_verify']['shared'])
        delta(mods['spec_delta_verify'],forget,conv,fused,output,ssm,cuR)
        assert bits(ssm,start), 'no-store delta changed committed SSM'
        reference=start.clone()
        delta(mods['spec_delta_advance'],forget,conv,fused,refout,reference,cuR)
        assert bits(output,refout), 'verify/store output mismatch'
        # A/B against the pinned original delta kernel, not only new twins.
        original=D.module(DUMP/'module_363.cubin'); oldstate=start.clone(); oldout=t.empty_like(output)
        delta(original,forget,conv,fused,oldout,oldstate,cuR)
        assert bits(output,oldout), 'new delta output differs from pinned original'
        assert bits(reference,oldstate), 'new store state differs from pinned original'
        packed=[]
        for src,cols,pitch in ((fused,3336,3336),(conv,3072,3072),(forget,1024,1024)):
            dst=t.empty_like(src)
            D.run(mods['spec_round'],'spec_pack',[src,dst,n,cu,I(cols),I(pitch),I(R)],(S,R,1))
            ref=t.cat([src[s*R:s*R+int(n[s])] for s in range(S)])
            assert bits(dst[:int(cu[-1])],ref), f'pack cols={cols}'
            packed.append(dst)
        ff,qq,aa=packed
        advanced=start.clone(); unused=t.empty_like(output)
        delta(mods['spec_delta_advance'],aa,qq,ff,unused,advanced,cu)
        # Independent per-sequence prefix oracle, including untouched spare lines.
        gold=start.clone()
        for s in range(S):
            counts=tensor([int(n[s]) if i==s else 0 for i in range(S)])
            cc=t.cat([tensor([0]),counts.cumsum(0,dtype=t.int32)])
            # Put s's inputs at the compact start, zero-length peers touch no rows.
            temp=t.empty_like(output)
            delta(original,forget[s*R:].contiguous(),conv[s*R:].contiguous(),fused[s*R:].contiguous(),temp,gold,cc)
        assert bits(advanced,gold), 'advance prefix state mismatch'
        # Carry follows pool lines, independent of batch slot changes.
        h=rnd((R*S,4096)); carry=t.zeros((2*S+1,4096),device=dev,dtype=bf)
        D.run(mods['spec_round'],'spec_carry_store',[h,carry,lines,n,valid,I(R)],(S,1,1))
        restored=t.empty((S,4096),device=dev,dtype=bf)
        D.run(mods['spec_round'],'spec_carry_load',[carry,lines,restored],(S,1,1))
        assert bits(restored,t.stack([h[R*s+int(n[s])-1] for s in range(S)])), 'carry round trip'
        logs.append({'seqs':S,f'k{D6} splice/accept/conv/delta/pack/carry':'PASS'})
        t.cuda.synchronize()
    # Merged accept/carry/repair control CTA, forced nacc 1..R over width R.
    S=4
    want=t.tensor([1,2,R,R//2+1],device='cuda',dtype=t.int32)
    drafts=t.tensor([[100+s,200+s,300+s,400+s,500+s,600+s] for s in range(S)],device='cuda',dtype=t.int64)
    verify=t.empty((S,R),device='cuda',dtype=t.int64)
    for s in range(S):
        a=int(want[s])
        vals=[int(drafts[s,j]) for j in range(D6)]+[900+s]
        if a<R: vals[a-1]+=17+2*(a-1)
        verify[s]=t.tensor(vals,device='cuda',dtype=t.int64)
    valid=t.ones(S*R,device=dev,dtype=t.int32)
    lines=t.arange(1,S+1,device='cuda',dtype=t.int32)
    h=(t.randn((S*R,4096),device='cuda')*.2).to(t.bfloat16)
    state0=(t.randn((S+1,4096),device='cuda')*.2).to(t.bfloat16)
    fused_state=state0.clone()
    tokens=t.empty((S*R,),device='cuda',dtype=t.int64)
    nacc=t.empty(S,device='cuda',dtype=t.int32)
    cu=t.empty(S+1,device='cuda',dtype=t.int32)
    repair_ids=t.empty((S*R,),device='cuda',dtype=t.int64)
    repair_valid=t.empty((S*R,),device='cuda',dtype=t.int32)
    D.run(mods['spec_accept_carry_repair'],'spec_accept_carry_repair',
          [drafts,verify,valid,h,fused_state,lines,tokens,nacc,cu,repair_ids,repair_valid,I(S),I(R)],
          (1,1,1),256)
    t.cuda.synchronize()
    assert t.equal(nacc,want), f'forced nacc mismatch: {nacc.tolist()}'
    expected_cu=t.cat([t.zeros(1,device='cuda',dtype=t.int32),want.cumsum(0)])
    assert t.equal(cu,expected_cu), f'compact cu mismatch: {cu.tolist()}'
    assert t.equal(tokens,verify.reshape(-1)), 'round token copy mismatch'
    expected_mask=t.stack([t.arange(R,device='cuda') < (want[s]-1) for s in range(S)]).reshape(-1).to(t.int32)
    assert t.equal(repair_ids,verify.reshape(-1)) and t.equal(repair_valid,expected_mask), 'repair metadata mismatch'
    # Run the old one-row carry store sequentially. FP32 compare, as in k=2.
    ref_state=state0.clone()
    for s in range(S):
        one_h=h[s*R:(s+1)*R].contiguous()
        one_n=want[s:s+1].contiguous()
        one_v=valid[s*R:s*R+R].contiguous()
        one_line=lines[s:s+1].contiguous()
        D.run(mods['spec_round'],'spec_carry_store',
              [one_h,ref_state,one_line,one_n,one_v,I(R)],(1,1,1))
    t.cuda.synchronize()
    err=(fused_state.float()-ref_state.float()).abs().max().item()
    den=ref_state.float().abs().max().item()
    rel=err/max(den,1e-12)
    assert rel <= 1e-3, f'carry fused vs sequential rel err {rel}'
    for s in range(S):
        assert t.equal(fused_state[lines[s]],h[s*R+int(want[s])-1]), f'off-by-one carry s={s}'
    print(f'k{D6} accept/carry/repair forced nacc={want.tolist()}: PASS (fp32 rel_err={rel:.3g})')
    # Draft metadata at per=R, steps 0..5; one invalid sequence covered.
    bt=t.arange(S*16,device='cuda',dtype=t.int32).reshape(S,16)+1
    positions=t.tensor([[p]*R for p in (0,1,3,255)],device='cuda',dtype=t.int32)
    dvalid=t.ones(S*R,device=dev,dtype=t.int32); dvalid[(S-1)*R]=0
    pos,lens,slot,ok=[t.empty(S,device='cuda',dtype=t.int32) for _ in range(4)]
    for step in range(D6):
        D.run(mods['spec_round'],'spec_draft_meta',[positions,dvalid,bt,pos,lens,slot,ok,I(R),I(step),I(16)],(S,1,1),32)
        for j,p in enumerate(positions[:,0].tolist()):
            pp=max(p-1+step,0); active=j!=S-1 and p-1+step>=0
            assert pos[j].item()==pp and lens[j].item()==pp+1 and ok[j].item()==active, (j,step)
            assert slot[j].item()==(bt[j,pp>>8].item()*256+(pp&255) if active else 0), (j,step)
    # record_draft writes column `step` of the per=6 drafts cache.
    nxt=tensor([1000+s for s in range(S)],t.int64)
    drafts_buf=t.zeros((S,D6),device=dev,dtype=t.int64)
    for step in range(D6):
        D.run(mods['spec_round'],'spec_record_draft',[nxt,drafts_buf,I(step),I(D6)],(S,1,1),32)
        assert drafts_buf[:,step].tolist()==[1000+s for s in range(S)], step
    # uniform: ones from the first column validity, cu is the identity.
    ones=t.empty(S,device=dev,dtype=t.int32); ucu=t.empty(S+1,device=dev,dtype=t.int32)
    D.run(mods['spec_round'],'spec_uniform',[dvalid,ones,ucu,I(R)],(S,1,1),32)
    assert ones.tolist()==[1]*(S-1)+[0] and ucu.tolist()==list(range(S+1))
    print(f'k{D6} draft_meta steps 0..{D6-1} + record_draft + uniform: PASS')
    # One captured data-flow DAG at rows=7; counts vary between replays.
    draft=tensor([[11,12,13,14,15,16]]*S,t.int64)
    verify=t.empty((S,R),device='cuda',dtype=t.int64)
    valid=t.ones((S,R),device=dev,dtype=t.int32); valid[-1]=0
    out=t.empty_like(verify); n=t.empty(S,device='cuda',dtype=t.int32)
    cu=t.empty(S+1,device='cuda',dtype=t.int32)
    ff=t.randn((S*R,3336),device='cuda',dtype=t.bfloat16)
    qq=t.randn((S*R,3072),device='cuda',dtype=t.bfloat16)
    aa=t.randn((S*R,1024),device='cuda',dtype=t.bfloat16)
    packed=[t.empty_like(x) for x in (ff,qq,aa)]
    def run():
        D.run(mods['spec_round'],'spec_accept',[draft,verify,valid,out,n,cu,C.c_int(S),I(R)],(1,1,1),32)
        D.run(mods['spec_round'],'spec_pack_kda',[ff,qq,aa,*packed,n,cu,I(R)],(S,R,1))
    stream=t.cuda.Stream(); stream.wait_stream(t.cuda.current_stream())
    with t.cuda.stream(stream): run()
    t.cuda.current_stream().wait_stream(stream); t.cuda.synchronize()
    graph=t.cuda.CUDAGraph()
    with t.cuda.graph(graph,stream=stream): run()
    for counts in ([1]*S,[R]*S,[1,R,3,5]):
        wanted=list(counts); wanted[-1]=0
        verify.copy_(t.tensor([[11+j if j<a-1 else 21+j for j in range(D6)]+[99] for a in wanted],device='cuda'))
        graph.replay()
        assert n.tolist()==wanted, (counts,n.tolist())
        assert cu.tolist()==[sum(wanted[:j]) for j in range(S+1)]
        for x,y in zip((ff,qq,aa),packed):
            gold=t.cat([x[R*j:R*j+a] for j,a in enumerate(wanted)])
            assert t.equal(y[:gold.shape[0]],gold)
    print(f'k{D6} static graph accept/pack heterogeneous counts: PASS')
    (ART/f'test-k{D6}.json').write_text(json.dumps({'build_sha256':sha(ART/'build.json'),'results':logs},indent=2)+'\n')
    print(json.dumps(logs,indent=2))


def k6_test():
    kn_test(7, 6)


def k7_test():
    kn_test(8, 7)


def cpu_check():
    """Check graph wiring and that private M32 imports do not change decode."""
    from . import gen as G
    before=G.build(45,allreduce='lamport')
    m=G.build(45,allreduce='lamport',mtp=True)
    after=G.build(45,allreduce='lamport')
    assert before==after, 'MTP changed producer module globals or default decode'
    cc=m['programs']['round_k2']['calls']
    draft1=[c for c in cc if c['label'].startswith('draft1.')]
    assert any(c['label']=='draft1.prep' for c in draft1)
    assert not any('dsa_topk' in c['op'] or 'kpool' in c['op'] for c in draft1)
    assert any(c['op']=='spec_eh_norm' for c in draft1)
    assert sum(c['op']=='spec_add_norm' for c in draft1)==2
    assert any(c['op']=='mtp16_dsa_attn' for c in draft1)
    assert len([c for c in cc if c['op']=='spec_delta_verify'])==34
    assert len([c for c in cc if c['op']=='spec_delta_advance'])==34
    assert len([c for c in cc if c['op']=='spec_pack_kda'])==34
    assert len([c for c in cc if c['op']=='spec_conv_advance'])==34
    assert len(m['ops']['mtp32_moe_router']['impl']['launches'])==2
    assert m['buffers']['spec_F']['shape']==[34,32,3336]
    assert m['buffers']['round_tokens']['fill']=='tokens' and m['buffers']['nacc']['fill']=='count'
    assert m['programs']['round_k2']['batch']=={'groups':8,'rows':3}
    assert m['states']['kv']['bytes_per_token']==12288
    assert m['states']['idx']['bytes_per_token']==396
    for name,op in m['ops'].items():
        if name.endswith('dsa_attn'):
            assert all(op['impl']['scratch'][n]['shape']==[32] for n in ('nsd','nmb','vbi','sem'))
    check_weights(m)
    print('MTP CPU invariants and unchanged default generation: PASS')

def cpu_check_k6():
    """k=6 structure: draft0..draft5 sections, rows=7 batch, round-width ABI args.
    No CUDA context. k2 default generation is checked separately in cpu_check."""
    from . import gen as G
    m6=G.build(45,allreduce='lamport',mtp=True,mtp_steps=6)
    assert 'round_k2' not in m6['programs']
    cc6=m6['programs']['round_k6']['calls']
    for step in range(6):
        assert any(c['label']==f'draft{step}.meta' for c in cc6), step
        assert any(c['label']==f'draft{step}.record' for c in cc6), step
    assert not any(c['label'].startswith('draft6.') for c in cc6)
    assert m6['programs']['round_k6']['batch']=={'groups':4,'rows':7}
    assert m6['buffers']['drafts']['shape']==[16,6]
    assert m6['buffers']['round_tokens']['shape']==['seqs',7]
    # draft0 runs the full indexer; drafts 1..5 reuse its expanded indices (R1)
    assert any('dsa_topk' in c['op'] or 'kpool' in c['op'] for c in cc6
               if c['label'].startswith('draft0.'))
    for step in range(1,6):
        sec=[c for c in cc6 if c['label'].startswith(f'draft{step}.')]
        assert not any('dsa_topk' in c['op'] or 'kpool' in c['op'] for c in sec), step
    # round-width ABI: rows/steps scalars are call-identical, so normalize folds
    # them into the launch wire args as literals (fold_constants).
    def folded_i32(op):
        out=[]
        for L in m6['ops'][op]['impl']['launches']:
            out += [a['i32'] for a in L.get('args',[]) if 'i32' in a]
        return out
    for op,val in (('spec_splice',7),('spec_pack_kda',7),('spec_conv_verify',7),
                   ('spec_conv_advance',7),('spec_accept_carry_repair',7),('spec_record_draft',6)):
        assert val in folded_i32(op), (op,folded_i32(op))
    assert m6['ops']['spec_pack_kda']['impl']['launches'][0]['grid'][1]==7
    assert len([c for c in cc6 if c['op']=='spec_delta_advance'])==34
    # Full ship-recipe composition at steps=6: fusers must accept round_k6.
    ship = dict(moe='v2', mhc='boundary', dsa='v2a', kda='fused',
                slab2='glue', slab1='select', arpdl=True)
    ms6 = G.build(45, allreduce='lamport', mtp=True, mtp_steps=6, **ship)
    ms2 = G.build(45, allreduce='lamport', mtp=True, mtp_steps=2, **ship)
    assert ms6['programs']['decode'] == ms2['programs']['decode'], 'k6 changed decode'
    rc = ms6['programs']['round_k6']['calls']
    from collections import Counter
    cnt = Counter(c['op'] for c in rc)
    assert cnt['moe_verify_v2'] == 42 and cnt['moe_decode_v2'] == 6, dict(cnt)
    # v3 round fusion is rows=3 ABI: auto-off at steps=6, unfused path present.
    assert cnt['spec_kda_fused_v3'] == 0 and cnt['spec_kda_select'] == 0
    for op in ('spec_conv_verify', 'spec_delta_verify', 'spec_conv_advance',
               'spec_delta_advance', 'spec_pack_kda', 'mtp32_kda_fg_b'):
        assert cnt[op] == 34, (op, cnt[op])
    # k2 ship round keeps its fused form (regression lock on composition).
    cnt2 = Counter(c['op'] for c in ms2['programs']['round_k2']['calls'])
    assert cnt2['spec_kda_fused_v3'] == 34 and cnt2['spec_kda_select_all'] == 1
    assert cnt2['moe_verify_v2'] == 42 and cnt2['moe_decode_v2'] == 2
    # arpdl edges survive in both rounds.
    for mm in (ms2, ms6):
        pdl_ar = [l for o in mm['ops'].values() for l in o['impl']['launches']
                  if l['entry'].endswith('_lamport_pdl') and l.get('pdl')]
        assert pdl_ar, 'arpdl edges missing'
    print('MTP k6 structure: PASS')


def cpu_check_k7():
    """k=7 structure: rows=8 batch (32 tokens = cap), drafts [16,7]. No CUDA."""
    from . import gen as G
    m7 = G.build(45, allreduce='lamport', mtp=True, mtp_steps=7)
    cc7 = m7['programs']['round_k7']['calls']
    assert m7['programs']['round_k7']['batch'] == {'groups': 4, 'rows': 8}
    assert m7['buffers']['drafts']['shape'] == [16, 7]
    assert m7['buffers']['round_tokens']['shape'] == ['seqs', 8]
    for step in range(7):
        assert any(c['label'] == f'draft{step}.meta' for c in cc7), step
    assert not any(c['label'].startswith('draft7.') for c in cc7)
    def folded_i32(op):
        out = []
        for L in m7['ops'][op]['impl']['launches']:
            out += [a['i32'] for a in L.get('args', []) if 'i32' in a]
        return out
    for op, val in (('spec_splice', 8), ('spec_pack_kda', 8), ('spec_conv_verify', 8),
                    ('spec_conv_advance', 8), ('spec_accept_carry_repair', 8),
                    ('spec_record_draft', 7)):
        assert val in folded_i32(op), (op, folded_i32(op))
    assert m7['ops']['spec_pack_kda']['impl']['launches'][0]['grid'][1] == 8
    ship = dict(moe='v2', mhc='boundary', dsa='v2a', kda='fused',
                slab2='glue', slab1='select', arpdl=True)
    ms7 = G.build(45, allreduce='lamport', mtp=True, mtp_steps=7, **ship)
    ms2 = G.build(45, allreduce='lamport', mtp=True, mtp_steps=2, **ship)
    assert ms7['programs']['decode'] == ms2['programs']['decode'], 'k7 changed decode'
    from collections import Counter
    cnt = Counter(c['op'] for c in ms7['programs']['round_k7']['calls'])
    assert cnt['moe_verify_v2'] == 42 and cnt['moe_decode_v2'] == 7, dict(cnt)
    assert cnt['spec_kda_fused_v3'] == 0 and cnt['spec_kda_select'] == 0
    print('MTP k7 structure: PASS')


def audit_v2a(baseline=None, candidate=None):
    """CPU-only audit of emitted MTP manifests and their exact bundled cubins.

    Unlike the inherited DSA audit, this follows lower_wire's lifted call
    arguments and checks the M32 copies actually served. No CUDA context.
    """
    import re
    baseline = Path(baseline or ROOT/'examples/glm53-flash-mtp.json').resolve()
    candidate = Path(candidate or ART/'mtp-v2a.json').resolve()
    old = json.loads(baseline.read_text())
    new = json.loads(candidate.read_text())
    changed = {prefix+'dsa_w_'+suffix for prefix in ('', 'mtp16_', 'mtp32_')
               for suffix in ('kc', 'vc')}
    assert old.keys() == new.keys()
    for key in old.keys() - {'ops', 'modules', 'programs'}:
        assert old[key] == new[key], ('unexpected non-absorb change', key)
    assert old['ops'].keys() == new['ops'].keys()
    assert {n for n in old['ops'] if old['ops'][n] != new['ops'][n]} == changed
    common = old['modules'].keys() & new['modules'].keys()
    assert all(old['modules'][n]['sha256'] == new['modules'][n]['sha256'] for n in common)
    assert not old['modules'].keys() - new['modules'].keys()
    assert new['modules'].keys() - old['modules'].keys() == {'glm53_dsa_w_kc_v2'}
    hashes = {}
    for name, desc in new['modules'].items():
        path = (candidate.parent/desc['source']).resolve()
        assert sha(path) == desc['sha256'], ('bundle hash', name, path)
        hashes[name] = path
    assert new['vars'] == {'tokens': {'max': 32}, 'seqs': {'max': 16}}
    assert new['states']['kv']['bytes_per_token'] == 12288
    assert new['states']['idx']['bytes_per_token'] == 396
    assert new['states']['idx_tail']['bytes_per_seq'] == 49152
    assert new['buffers']['idx_tail_lines']['shape'][0] == 12
    assert new['programs']['round_k2']['batch'] == {'groups': 8, 'rows': 3}
    assert new['programs'].keys() == old['programs'].keys()
    counts = {}; quant_calls = 0
    for name, prog in new['programs'].items():
        ref = old['programs'][name]
        assert {k:v for k,v in prog.items() if k != 'calls'} == {k:v for k,v in ref.items() if k != 'calls'}
        assert len(prog['calls']) == len(ref['calls'])
        for a, b in zip(ref['calls'], prog['calls']):
            op = b['op']
            if op in changed:
                assert {k:v for k,v in a.items() if k != 'args'} == {k:v for k,v in b.items() if k != 'args'}
                assert a['args'][:4] == b['args'] and len(b['args']) == 4
                axis = 'tokens' if op.startswith('mtp32_') else 'seqs'
                assert b['args'][3] == {'var': axis}
            else:
                assert a == b, ('non-absorb call changed', name, b.get('label'))
            if op.endswith('dsa_act_quant'):
                launch = new['ops'][op]['impl']['launches'][0]
                bound = launch['args'][3]
                if 'param' in bound:
                    bound = b['args'][bound['param']]
                axis = 'tokens' if op.startswith('mtp32_') else 'seqs'
                assert bound == {'expr': {'mul': [axis, 32]}}, (name, b['label'], bound)
                assert launch['grid'] == [axis, 1, 1]
                quant_calls += 1
        def launches(m, calls):
            return sum(len(m['ops'][c['op']]['impl']['launches']) for c in calls)
        counts[name] = {'manifest_calls': len(prog['calls']),
                        'v1_launches': launches(old, ref['calls']),
                        'v2a_launches': launches(new, prog['calls'])}
    for name in changed:
        op = new['ops'][name]; axis = 'tokens' if name.startswith('mtp32_') else 'seqs'
        assert op['params'] == ['in buffer<bf16>', 'in buffer<bf16>', 'out buffer<bf16>', 'i32']
        assert set(op['impl']) == {'launches'}
        assert len(op['impl']['launches']) == 1
        launch = op['impl']['launches'][0]
        n = 512 if name.endswith('kc') else 256
        assert launch['entry'] == 'glm53_dsa_w_'+name[-2:]+'_v2'
        assert launch['block'] == [128, 1, 1]
        assert launch['grid'] == [n//16, 8, {'ceil_div': [axis, 4]}]
    # Retain the MTP metadata, cache writers, and static M32 storage. The
    # decode-only full-v2 cache entries have baked eleven-layer pitches.
    dsa_ops = {n:o for n,o in new['ops'].items()
               if 'dsa_' in n or n.startswith(('spec_kpool', 'spec_kv_store'))}
    for name, op in dsa_ops.items():
        for launch in op['impl']['launches']:
            entry = launch['entry']
            if entry.startswith('glm53_dsa_') and entry.endswith('_v2'):
                assert entry in ('glm53_dsa_w_kc_v2', 'glm53_dsa_w_vc_v2')
        if name.endswith('dsa_attn'):
            scratch = op['impl']['scratch']
            assert all(scratch[k]['shape'] == [32] for k in ('nsd','nmb','vbi','sem'))
            fields = op['impl']['launches'][1]['args'][0]['pack']['fields']
            by_at = {f['at']: f for f in fields}
            assert all(by_at[k]['i64'] == 6144 for k in (128,136,144,168,176,184))
            if name.startswith('mtp32_'):
                assert scratch['oacc']['shape'] == [232,32,512]
                assert scratch['lseacc']['shape'] == [232,32]
    assert new['ops']['spec_kpool_update']['impl']['launches'][0]['args'][-1] == {'i64': 101376}
    assert new['ops']['spec_kv_store']['impl']['launches'][0]['args'][-2] == {'i64': 12288}
    # ABI follows the emitted call (implicit op params or explicit launch
    # params), not the Python producer's pre-lowering argument list.
    elf_cache = {}; abi_count = 0
    for name, op in dsa_ops.items():
        for launch in op['impl']['launches']:
            if launch['entry'].startswith('extern:'):
                continue
            path = hashes[launch['module']]
            if path not in elf_cache:
                text = subprocess.check_output(['/usr/local/cuda-13.0/bin/cuobjdump', '-elf', str(path)], text=True)
                parts = re.split(r'\.nv\.info\.(\w+)', text)
                elf_cache[path] = {}
                for entry, section in zip(parts[1::2], parts[2::2]):
                    vals = re.findall(r'Ordinal\s*:\s*0x([0-9a-f]+)\s+Offset\s*:\s*0x([0-9a-f]+)\s+Size\s*:\s*0x([0-9a-f]+)', section)
                    if vals:
                        elf_cache[path][entry] = {int(i,16):(int(o,16),int(z,16)) for i,o,z in vals}
            params = launch.get('params', op['params'])
            def size(t):
                if t.startswith('bytes<'): return int(t[6:-1])
                if 'buffer<' in t or t.endswith('state') or t in ('i64','u64','f64'): return 8
                assert t in ('i32','u32','f32'), (name, t)
                return 4
            actual = elf_cache[path].get(launch['entry'], {})
            expected = {i:size(t) for i,t in enumerate(params)}
            assert {i:v[1] for i,v in actual.items()} == expected, ('ABI sizes', name)
            if name in changed:
                assert actual == {0:(0,8),1:(8,8),2:(16,8),3:(24,4)}, ('absorb offsets', name)
            abi_count += 1
    report = {'scope': 'CPU-only emitted manifest + bundle audit; no device execution',
              'baseline': {'path': str(baseline), 'sha256': sha(baseline)},
              'candidate': {'path': str(candidate), 'sha256': sha(candidate)},
              'changed_ops': sorted(changed), 'bundle_modules_hashed': len(hashes),
              'dsa_launch_abi_checks': abi_count, 'act_quant_call_checks': quant_calls,
              'program_counts': counts, 'status': 'PASS'}
    ART.mkdir(parents=True, exist_ok=True)
    (ART/'v2a-static-audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return report


def check_weights(m):
    """Header-only shape/dtype audit, including all eight rank slices."""
    import math
    import struct
    headers={}
    for path in Path(os.environ.get('GLM53_CHECKPOINT', 'weights/GLM-5.3-Flash')).glob('*.safetensors'):
        with path.open('rb') as f:
            size=struct.unpack('<Q',f.read(8))[0]
            header=json.loads(f.read(size))
        for name,desc in header.items():
            if name=='__metadata__': continue
            assert name not in headers, f'duplicate tensor {name}'
            headers[name]=desc
    count=0
    for name,buf in m['buffers'].items():
        if buf['kind']!='weight': continue
        count+=1
        dtype={'bf16':'BF16','f32':'F32','fp8e4m3':'F8_E4M3'}[buf['dtype']]
        for rank in range(8):
            elements=0
            for bind in buf['bind']:
                meta=headers[bind['tensor']]
                assert meta['dtype']==dtype, (name,meta['dtype'],dtype)
                shape=list(meta['shape'])
                if len(shape)<2: shape=[1]+shape
                for field,dim in (('rows',0),('cols',1)):
                    if field not in bind: continue
                    lo,hi=bind[field]['ranges'][rank]
                    assert 0<=lo<=hi<=shape[dim], (name,rank,field,shape,lo,hi)
                    shape[dim]=hi-lo
                elements+=math.prod(shape)
            assert elements==math.prod(buf['shape']), (name,rank,elements,buf['shape'])
    print(f'weight headers: {count} buffers x8 ranks, shapes/dtypes/ranges/duplicates PASS')


def bench_advance():
    """Short single-rank KDA commit microbenchmark, not a model/TP8 timing.
    Distinct state for all 34 layers; capture a static 102-launch graph.
    """
    gpu_check()
    import torch as t
    t.cuda.init(); t.cuda.set_device(0); t.empty(1,device='cuda'); t.manual_seed(5303)
    D=Driver(t); I,J,F=C.c_int,C.c_int64,C.c_float
    mods={n:D.module(ART/f'{n}.cubin') for n in ('spec_round','spec_kda','spec_delta_advance')}
    results=[]
    for S in (1,2,4,8):
        n=t.full((S,),3,device='cuda',dtype=t.int32)
        cu=t.arange(S+1,device='cuda',dtype=t.int32)*3
        valid=t.ones(3*S,device='cuda',dtype=t.int32)
        lines=t.arange(1,S+1,device='cuda',dtype=t.int32)
        ff=t.randn((34,32,3336),device='cuda',dtype=t.bfloat16)*.02
        qq=t.randn((34,32,3072),device='cuda',dtype=t.bfloat16)*.02
        aa=t.randn((34,32,1024),device='cuda',dtype=t.bfloat16)*.02
        state=t.zeros((34,S+1,8,128,128),device='cuda',dtype=t.float32)
        conv=t.zeros((34,S+1,3,3072),device='cuda',dtype=t.bfloat16)
        al=t.zeros((34,8),device='cuda',dtype=t.float32)
        dt=t.zeros((34,1024),device='cuda',dtype=t.float32)
        pf=t.empty_like(ff[0]); pq=t.empty_like(qq[0]); pa=t.empty_like(aa[0]); out=t.empty_like(aa[0])
        def run():
            for k in range(34):
                D.run(mods['spec_round'],'spec_pack_kda',[ff[k],qq[k],aa[k],pf,pq,pa,n,cu,I(3)],(S,3,1))
                args=[al[k],pa,dt[k],F(1),F(20),F(-5),pq,pq[:,1024:],pq[:,2048:],pf[:,3072:],out,state[k],lines,
                      I(131072),cu,I(0),F(128**-.5),I(3*S),I(1024),I(3072),I(3072),I(3072),I(3336),J(0),J(0)]
                D.run(mods['spec_delta_advance'],'fused_sigmoid_gating_delta_rule_update_kernel',args,(4,S,8),32,64)
                D.run(mods['spec_kda'],'spec_conv_advance',[ff[k],conv[k],lines,n,valid,I(3)],(S,12,1))
        stream=t.cuda.Stream(); stream.wait_stream(t.cuda.current_stream())
        with t.cuda.stream(stream): run()
        t.cuda.current_stream().wait_stream(stream); t.cuda.synchronize()
        graph=t.cuda.CUDAGraph()
        with t.cuda.graph(graph,stream=stream): run()
        for acc in (1,3):
            n.fill_(acc); cu.copy_(t.arange(S+1,device='cuda',dtype=t.int32)*acc)
            for _ in range(3): graph.replay()
            start=t.cuda.Event(enable_timing=True); stop=t.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(20): graph.replay()
            stop.record(); stop.synchronize()
            results.append({'seqs':S,'nacc':acc,'layers':34,'graph_launches':102,'ms':start.elapsed_time(stop)/20})
        del graph,state,conv
    report={'scope':'single H100, 34 distinct layer states, warm graph, no TP or target/draft',
            'build_sha256':sha(ART/'build.json'),'results':results}
    (ART/'advance-bench.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('build','test','test-k6','test-k7','prepare','cpu','bench-advance','audit-v2a'))
    p.add_argument('--baseline', type=Path, help='audit-v2a baseline emitted manifest')
    p.add_argument('--candidate', type=Path, help='audit-v2a candidate emitted manifest')
    args=p.parse_args()
    if args.command == 'audit-v2a':
        audit_v2a(args.baseline, args.candidate)
        return
    if args.baseline or args.candidate:
        p.error('--baseline/--candidate require audit-v2a')
    if args.command == 'cpu':
        cpu_check(); cpu_check_k6(); cpu_check_k7()
        return
    {'build':build,'test':test,'test-k6':k6_test,'test-k7':k7_test,'prepare':prepare_weights,'bench-advance':bench_advance}[args.command]()



# ---------------- isolated --mtp generator integration ----------------
def _map(x, fn):
    if isinstance(x, dict): return {k:_map(v,fn) for k,v in x.items()}
    if isinstance(x, list): return [_map(v,fn) for v in x]
    return fn(x)


def _copy_module(name, rows):
    """Private Python globals, not monkey-patching the in-flight op modules."""
    import importlib.util
    p=Path(__file__).with_name(name+'.py')
    spec=importlib.util.spec_from_file_location('glm53._mtp_'+name,p)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    if hasattr(mod,'S_MAX'): mod.S_MAX=rows
    return mod


def _base_ops(rows, dsa="v1"):
    from copy import deepcopy
    from . import gen as G
    out={}
    for name in ('ops_dsa','ops_kda','ops_mhc','ops_moe','ops_head'):
        mod=_copy_module(name,rows)
        out.update(mod.ops({'bt_cols':G.BT_COLS,'st_cols':G.ST_COLS}) if name=='ops_dsa' else mod.ops())
    # Stage A only: absorb has no state/page pitch or static-M scratch.
    # Keep the private baseline's M=rows TMA descriptors and metadata. Using
    # ops_dsa_v2.ops() wholesale would restore its baseline's M=16 globals.
    if dsa == 'v2a':
        from . import ops_dsa_v2
        absorb = ops_dsa_v2.ops({'bt_cols':G.BT_COLS,'st_cols':G.ST_COLS})
        for name in ('dsa_w_kc','dsa_w_vc'):
            out[name] = deepcopy(absorb[name])
    elif dsa != 'v1':
        raise ValueError(f'unsupported MTP DSA mode: {dsa}')
    for name in ('moe_w13','moe_w2'):
        out[name]['impl']['launches'][0]['args'][12]={'expr':{'mul':['seqs',9]}}
    # FA3 scheduler metadata is per query. The old eight-entry allocation
    # is not enough for either sixteen-row decode or 24-query verification.
    for name in ('nsd','nmb','vbi','sem'):
        out['dsa_attn']['impl']['scratch'][name]['shape']=[32]
    if os.environ.get('GLM53_FA3_PDL','1')=='0':
        for kernel in out['dsa_attn']['impl']['launches']: kernel['pdl']=False
    # Convert all twelve-layer state strides, including unused FA3 TMA fields.
    out=_map(out,lambda v: {11264:12288,5632:6144,92928:101376}.get(v,v) if isinstance(v,int) else v)
    G.resolve_modules(out)
    if rows==16:
        out=_map(out,lambda v:'seqs' if v=='tokens' else v)
    if rows==32:
        out=_map(out,lambda v:'tokens' if v=='seqs' else v)
        router=out['moe_router']; second=deepcopy(router['impl']['launches'][0])
        router['params'] += ['in buffer<bf16>','out buffer<f32>']
        for field in second['args'][0]['pack']['fields']:
            if field.get('param')==0: field['param']=3
            elif field.get('param')==2: field['param']=4
        router['impl']['launches'].append(second)
    return out


def prepare_weights():
    """New small derived file only; no checkpoint/index or prep_weights edits."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    ckpt=Path(os.environ.get('GLM53_CHECKPOINT', 'weights/GLM-5.3-Flash'))
    idx=json.loads((ckpt/'model.safetensors.index.json').read_text())['weight_map']
    p='model.language_model.layers.45.self_attn.'
    def get(name):
        with safe_open(str(ckpt/idx[name]),framework='pt',device='cpu') as f:
            return f.get_tensor(name)
    w=get(p+'kv_b_proj.weight').reshape(64,512,512)
    out={p+'kv_b.w_kc_t':w[:,:256].transpose(1,2).contiguous(),
         p+'kv_b.w_vc':w[:,256:].contiguous()}
    for dst,src in [('k_norm_w_f32','k_norm.weight'),('k_norm_b_f32','k_norm.bias'),
                    ('weights_proj_f32','weights_proj.weight'),('ape_f32','index_kpool_compress_ape')]:
        out[p+'indexer.'+dst]=get(p+'indexer.'+src).float()
    ART.mkdir(parents=True,exist_ok=True)
    target=ART/'mtp-derived.safetensors'
    save_file(out,str(target),metadata={'format':'pt'})
    link=ckpt/'mtp-derived.safetensors'
    if link.exists() or link.is_symlink():
        if link.resolve()!=target: raise RuntimeError(f'refuse to replace {link}')
    else: link.symlink_to(target)
    print('derived',target, 'sha256',sha(target))


def extend_manifest(m, layers, *, allreduce='lamport', dsa='v1', steps=2):
    """Correctness-first MTP graph, parameterized by draft steps (k=steps).
    rows = steps+1 tokens per sequence; the round program is round_k{steps}
    and REPLACES round_k2 in kN builds. Verify uses unfused M32 variants;
    decode may keep the caller's mHC/MoE optimizations. Future v2 verify cuts
    must satisfy this no-store/carry interface, not mutate their decode ops.
    steps=2 emits the historical k=2 graph (values bitwise vs pre-parameterization).
    """
    rows=steps+1
    if steps not in (2,4,6,7,8): raise ValueError(f'unsupported --mtp-steps {steps}')
    groups=min(8,32//rows)  # kern verify: round tokens (groups*rows) <= 32
    from copy import deepcopy
    from types import FunctionType
    from . import gen as G
    from .ops_common import a, i32, i64, var, rank
    config=json.loads((Path(os.environ.get('GLM53_CHECKPOINT', 'weights/GLM-5.3-Flash')) / 'config.json').read_text())
    config=config.get('text_config',config)
    if not config.get('index_share_for_mtp_iteration',False):
        raise ValueError('--mtp requires this checkpoint index_share_for_mtp_iteration=true')
    if layers!=45: raise ValueError('--mtp requires all 45 target layers')
    if allreduce!='lamport': raise ValueError('--mtp currently requires --allreduce lamport')
    if any(c['op'].endswith('_v2') and c['op'].startswith('dsa_')
           for c in m['programs']['decode']['calls']):
        raise ValueError('--mtp supports DSA absorb-only v2a, not full DSA-v2 cache fusions')
    specops=ops()
    # Round-width grid dims follow the step count (pack grids iterate rows).
    for name in ('spec_pack_kda','spec_pack'):
        specops[name]['impl']['launches'][0]['grid'][1]=rows
    raw16=_base_ops(16,dsa=dsa)
    semantic=G.decode_calls(45,raw16)
    raw32=_base_ops(32,dsa=dsa)
    # Correctness-first M32 router: two existing M=16 entries. Do not rebuild
    # the tiny GEMM with a new reduction tree until its numerical gate passes.
    m['ops'].update({'mtp16_'+n:o for n,o in raw16.items()})
    m['ops'].update({'mtp32_'+n:o for n,o in raw32.items()})
    m['ops'].update(specops)
    for name in ('spec_eh_norm','spec_add_norm'):
        m['ops'][name+'32']=deepcopy(specops[name])
        m['ops'][name+'32']['impl']['launches'][0]['grid']=['tokens',1,1]
    arparams=['inout buffer<bf16>','inout buffer<u8>','in buffer<u64>','out buffer<i32>','i32']
    arwire=['inout buffer<bf16>','out buffer<bf16>','inout buffer<u8>','in buffer<u64>','out buffer<i32>','i32','i32','i64']
    m['ops']['spec_ar']={'params':arparams,'impl':{'launches':[
        launch('spec_ar','spec_ar_lamport',arwire,[a(0),a(0),a(1),a(2),a(3),rank('tp'),a(4),i64(1_000_000_000)],[16,1,1],128)]}}
    # Greedy argmax needs eight-byte choices, not a full-vocabulary gather.
    from .ops_common import extern, scr, expr, mul
    for width, axis in ((16,'seqs'),(32,'tokens')):
        bx,bo='in buffer<bf16>','out buffer<bf16>'
        m['ops'][f'spec_head{width}']={'params':[bx,bx,'out buffer<i64>','i32'], 'impl':{
            'scratch':{'logits':{'dtype':'bf16','shape':[32,19360]},
                       'choice':{'dtype':'bf16','shape':[32,4]},
                       'gather':{'dtype':'bf16','shape':[8,32,4]}},
            'launches':[
                extern('cublaslt_bf16_tn',[bx,bx,bo,'i32','i32','i32'],[a(0),a(1),scr('logits'),a(3),i32(19360),i32(4096)]),
                launch('spec_head','spec_head_local',[bx,bo,'i32'],[scr('logits'),scr('choice'),rank('tp')],[axis,1,1],1024),
                extern('nccl_allgather_bf16',[bx,bo,'i64','i32'],[scr('choice'),scr('gather'),expr(mul(axis,4)),rank('tp')]),
                launch('spec_head','spec_head_global',[bx,'out buffer<i64>','i32'],[scr('gather'),a(2),a(3)],[axis,1,1],32)]}}
    # KDA projection must be separate from peer-pointer collective ops.
    for prefix in ('mtp16_','mtp32_'):
        m['ops'][prefix+'kda_o_proj_ar']['impl']['launches'].pop()
    m['vars']['tokens']['max']=32
    # Keep groups.max=16 to preserve every existing line-table byte offset.
    m['states']['kv']['bytes_per_token']=12288
    m['states']['idx']['bytes_per_token']=396
    m['states']['idx_tail']['bytes_per_seq']=12*4096
    m['states']['mtp_hidden']={'bytes_per_seq':8192}
    m['buffers']['idx_tail_lines']['shape'][0]=12
    work=FunctionType(G.workspace_buffers.__code__,{**G.workspace_buffers.__globals__,'S_MAX':32})(False)
    for name,buf in work.items(): m['buffers'][name]=buf
    for name in ('moe_c1','moe_h1','moe_h1_8','moe_bs','moe_c2','moe_expert_ids'):
        m['buffers'][name]['shape'][0]=288
    m['buffers']['moe_sorted']['shape'][0]=288*64
    m['buffers'].update({
        'spec_ar_sym':{'dtype':'u8','shape':[4194368],'kind':'carry','export':True},
        'spec_ar_peers':{'dtype':'u64','shape':[8],'kind':'peer','of':'spec_ar_sym','group':'tp'},
        'spec_ar_error':{'dtype':'i32','shape':[1],'kind':'output'},
        'mtp_hidden_lines':{'dtype':'i32','shape':[1,'seqs'],'kind':'input',
                            'domain':{'index_into':'mtp_hidden','stride':8192}},
        'anchor_token':{'dtype':'i64','shape':['seqs'],'kind':'input','fill':'token',
                        'domain':{'index_into':'embed'}},
        'round_tokens':{'dtype':'i64','shape':['seqs',rows],'kind':'output','fill':'tokens',
                        'domain':{'index_into':'embed'}},
        'nacc':{'dtype':'i32','shape':['seqs'],'kind':'output','fill':'count'},
    })
    def buf(name,dtype,shape): m['buffers'][name]={'dtype':dtype,'shape':shape,'kind':'workspace'}
    for name,dtype,shape in [
        ('spec_ids','i64',[32]),('drafts','i64',[16,steps]),('draft_next','i64',[16]),
        ('draft_pos','i32',[16]),('draft_len','i32',[16]),('draft_slot','i32',[16]),('draft_valid','i32',[16]),
        ('draft_cu','i32',[17]),('spec_ones','i32',[16]),('advance_cu','i32',[17]),
        ('row_lens','i32',[32]),('row_bt','i32',[32,G.BT_COLS]),('fa3_cu','i32',[33]),
        ('draft_prev','bf16',[32,4096]),('draft_eh','bf16',[32,8192]),('draft_res','bf16',[32,4096]),
        ('spec_target_h','bf16',[32,4096]),('verify_tokens','i64',[32]),
        ('repair_ids','i64',[32]),('repair_valid','i32',[32]),
        ('spec_F','bf16',[34,32,3336]),('spec_Q','bf16',[34,32,3072]),('spec_A','bf16',[34,32,1024]),
        ('pack_F','bf16',[32,3336]),('pack_Q','bf16',[32,3072]),('pack_A','bf16',[32,1024]),
        ('advance_o','bf16',[32,1024])]: buf(name,dtype,shape)
    # Inventory verifies layer45 uses the same mixed FP8 DSA+MoE shapes as 43,
    # but has NO mHC parameters. Six derived DSA tensors are prepared separately.
    for name,desc in list(m['buffers'].items()):
        if name.startswith('layers.43.') and not name.startswith('layers.43.hc_'):
            m['buffers'][name.replace('layers.43.','layers.45.')]=_map(deepcopy(desc),
                lambda x:x.replace('layers.43.','layers.45.') if isinstance(x,str) else x)
    for name,shape,tensor in [('enorm',[4096],'enorm.weight'),('hnorm',[4096],'hnorm.weight'),
                              ('eh_proj',[4096,8192],'eh_proj.weight'),('shared_norm',[4096],'shared_head.norm.weight')]:
        m['buffers']['layers.45.'+name]={'dtype':'bf16','shape':shape,'kind':'weight',
            'bind':[{'tensor':'model.language_model.layers.45.'+tensor}]}
    B=G.b; ST=G.st; SV={'var':'seqs'}; TV={'var':'tokens'}
    def call(label,op,*args): return {'label':label,'op':op,'args':list(args)}
    def ar(label,x,rows): return call(label,'spec_ar',x,B('spec_ar_sym'),B('spec_ar_peers'),B('ar_error'),rows)
    def transform(c,rows):
        c=deepcopy(c); op=c['op']; c['op']=f'mtp{rows}_'+op
        if rows==32:
            c['args']=_map(c['args'],lambda x:'tokens' if x=='seqs' else x)
            if op=='moe_router':
                c['args'] += [B(c['args'][0]['buf'],c['args'][0].get('offset',0)+16*4096*2),
                              B(c['args'][2]['buf'],c['args'][2].get('offset',0)+16*288*4)]
            if op in ('hc_prenorm','hc_big_fuse64'): c['args'][-1]={'i32':32}
        return c
    def dsa_change(c,rows,valid,positions,seq_len,slot,cu):
        """Semantic DSA calls -> explicit 12-layer, per-query/sequence ABI."""
        op=c['op'].removeprefix('mtp16_').removeprefix('mtp32_'); aa=c['args']
        if op=='dsa_prep':
            return call(c['label'],'spec_dsa_prep',positions,B('block_table'),B('row_bt'),B('row_lens'),
                        B('pool_lens'),B('pool_ctx'),B('dsa_lens'),B('slot_table'),B('fa3_cu'),i32(rows),i32(G.BT_COLS),i32(G.ST_COLS))
        if op=='dsa_kpool_update':
            c['op']='spec_kpool_update'; aa[7]=positions; aa[8]=seq_len; aa[9]=valid
            aa += [i32(rows),i64(101376)]
        elif op=='dsa_kv_store':
            c['op']='spec_kv_store'; aa[2]=slot; aa += [valid,i64(12288),i64(0)]
        elif op=='dsa_logits': aa[2]=B('row_bt')
        elif op=='dsa_topk': aa[3]=seq_len
        elif op=='dsa_attn': aa[4]=cu
        return c
    # Draft one-row DSA uses seqs launch rows, unlike verify/repair tokens.
    for name in ('spec_dsa_prep','spec_kv_store'):
        m['ops'][name+'1']=deepcopy(m['ops'][name]); m['ops'][name+'1']['impl']['launches'][0]['grid'][0]='seqs'
    # Port decode's old fixed page pitches without touching its arithmetic.
    for name in ('dsa_logits','dsa_attn','dsa_kv_store'):
        if name in m['ops']:
            m['ops'][name]=_map(m['ops'][name],lambda x:{11264:12288,5632:6144,92928:101376}.get(x,x) if isinstance(x,int) else x)
    if 'dsa_attn' in m['ops']:
        for name in ('nsd','nmb','vbi','sem'): m['ops']['dsa_attn']['impl']['scratch'][name]['shape']=[32]
    target_decode=m['programs']['decode']['calls']
    for c in target_decode:
        if c['op']=='dsa_kpool_update':
            c['op']='spec_kpool_update'; c['args'] += [i32(1),i64(101376)]
    # Stem only is sufficient for draft cache repair: KV and pooled index keys
    # are functions of EH/input_ln, before attention/MoE. No extra lm_head.
    layer_calls=[c for c in semantic if c['label'].startswith('l43.')]
    attn=[c for c in layer_calls if c['op'].startswith('dsa_') or c['label'].endswith(('qa_norm','kva_norm'))]
    moe=[c for c in layer_calls if c['op'].startswith('moe_')]
    indexer_ops={'dsa_wq_b','dsa_wk','dsa_kpool_gate','dsa_k_norm','dsa_hadamard',
                 'dsa_act_quant','dsa_kpool_update','dsa_weights_proj',
                 'dsa_logits_meta','dsa_logits','dsa_topk','dsa_clamp'}
    cache_ops={'dsa_quant_qkv','dsa_qkv_a','head_rms_norm','dsa_wk','dsa_kpool_gate','dsa_k_norm','dsa_kpool_update','dsa_kv_store'}
    def draft(label,ids,hidden,rows,positions,lengths,slots,valid,full,reuse_topk=False):
        width=16 if rows==1 else 32; nrows=SV if rows==1 else TV; pre=f'mtp{width}_'
        def norm(tag,x,w,y,outpitch=4096):
            return call(label+tag,pre+'head_rms_norm',x,B('layers.45.'+w),y,G.f32(1e-5),i32(4096),i32(4096),i32(outpitch))
        cc=[call(label+'embed',pre+'head_embed',ids,B('embed'),B('x'),i32(4096)),
            call(label+'eh_norm','spec_eh_norm' if rows==1 else 'spec_eh_norm32',
                 B('x'),hidden,B('layers.45.enorm'),B('layers.45.hnorm'),B('draft_eh')),
            call(label+'eh',pre+'kda_qkvbfg',B('draft_eh'),B('layers.45.eh_proj'),B('draft_res'),nrows,i32(4096),i32(8192)),
            norm('input_norm',B('draft_res'),'input_ln',B('x_norm'))]
        if full:
            prep=call(label+'prep','dsa_prep')
            cc.append(dsa_change(prep,rows,valid,positions,lengths,slots,B('draft_cu')))
            if rows==1: cc[-1]['op']='spec_dsa_prep1'
        for original in attn:
            op=original['op']
            # Oracle index_share_for_mtp_iteration=true: reuse the seed
            # expanded physical indices AND its valid count, including tail.
            # Do not append the new tail token or run the skipped indexer.
            if reuse_topk and op in indexer_ops: continue
            if not full and op not in cache_ops: continue
            if not full and original['label'].endswith('qa_norm'): continue
            c=transform(original,width); c['label']=label+original['label'].split('.',1)[1]
            c['args']=_map(c['args'],lambda x:x.replace('layers.43.','layers.45.') if isinstance(x,str) else x)
            for arg in c['args']:
                if arg.get('state')=='kv': arg['offset']=11*1024
                if arg.get('state')=='idx': arg['offset']=11*8448+(8192 if arg.get('offset',0)%8448==8192 else 0)
                if arg.get('buf')=='idx_tail_lines': arg['offset']=11*64
            c=dsa_change(c,rows,valid,positions,lengths,slots,B('fa3_cu'))
            if c['op']=='spec_kv_store' and rows==1: c['op']='spec_kv_store1'
            if op=='dsa_ar': c=ar(c['label'],B('sub_out'),nrows)
            cc.append(c)
        if full:
            res='spec_add_norm' if rows==1 else 'spec_add_norm32'
            cc += [call(label+'ffn_norm',res,B('draft_res'),B('sub_out'),
                        B('layers.45.post_attn_ln'),B('x_norm'))]
            for original in moe:
                c=transform(original,width); c['label']=label+original['label'].split('.',1)[1]
                c['args']=_map(c['args'],lambda x:x.replace('layers.43.','layers.45.') if isinstance(x,str) else x)
                if original['op']=='moe_ar': c=ar(c['label'],B('sub_out'),nrows)
                cc.append(c)
            cc += [call(label+'shared_norm',res,B('draft_res'),B('sub_out'),
                        B('layers.45.shared_norm'),B('draft_prev')),
                   call(label+'head',f'spec_head{width}',B('draft_prev'),B('lm_head'),B('draft_next'),nrows)]
        return cc
    def meta(label,per,step):
        return call(label,'spec_draft_meta',B('positions'),B('valid'),B('block_table'),B('draft_pos'),B('draft_len'),
                    B('draft_slot'),B('draft_valid'),i32(per),i32(step),i32(G.BT_COLS))
    # One-row prefill/decode repairs the preceding draft slot using the ACTUAL
    # next prompt token (not the target's prediction for that prompt token).
    bootstrap=[call('mtp.load','spec_carry_load',ST('mtp_hidden'),B('mtp_hidden_lines'),B('draft_prev')),meta('mtp.meta',1,0)]
    bootstrap += draft('mtp.cache.',B('token_ids'),B('draft_prev'),1,B('draft_pos'),B('draft_len'),B('draft_slot'),B('draft_valid'),False)
    target_decode[:0]=bootstrap
    target_decode += [call('mtp.ones','spec_uniform',B('valid'),B('spec_ones'),B('draft_cu'),i32(1)),
                      call('mtp.carry','spec_carry_store',B('h_norm'),ST('mtp_hidden'),B('mtp_hidden_lines'),B('spec_ones'),B('valid'),i32(1))]
    # Fixed DAG. Each draft pass writes at the shifted absolute position.
    cc=[call('load_hidden','spec_carry_load',ST('mtp_hidden'),B('mtp_hidden_lines'),B('draft_prev')),
        call('ones','spec_uniform',B('valid'),B('spec_ones'),B('draft_cu'),i32(rows))]
    # R1 (scope note glm53-mtpk4-scope-20260928.md): steps>=1 all reuse draft0's
    # expanded physical indices per index_share_for_mtp_iteration=true. If the
    # checkpoint shares indices only per iteration pair, drafts 2..N-1 would be
    # stale; the serving nacc tripwire (>=N-0.1) arbitrates. Fallback is a full
    # indexer per step (reuse_topk=False for step>=2).
    for step in range(steps):
        cc.append(meta(f'draft{step}.meta',rows,step))
        cc += draft(f'draft{step}.',B('anchor_token') if step==0 else B('draft_next'),B('draft_prev'),1,
                    B('draft_pos'),B('draft_len'),B('draft_slot'),B('draft_valid'),True,reuse_topk=(step>=1))
        cc.append(call(f'draft{step}.record','spec_record_draft',B('draft_next'),B('drafts'),i32(step),i32(steps)))
    cc.append(call('splice','spec_splice',B('anchor_token'),B('drafts'),B('spec_ids'),i32(rows)))
    # Main model verification. Three different cu arrays have distinct jobs:
    # host [0,3,..] (KDA), device [0,1,..] (FA3), prefix sums(nacc) (advance).
    kda=[]
    for original in semantic:
        c=transform(original,32); c['label']='verify.'+c['label']; op=original['op']
        if op in ('head_cast','head_argmax'): continue
        if op=='head_lm_head':
            cc.append(call('verify.head','spec_head32',B('spec_target_h'),B('lm_head'),B('verify_tokens'),TV))
            continue
        for arg in c['args']:
            if arg.get('buf')=='token_ids': arg['buf']='spec_ids'
            if arg.get('buf')=='next_token': arg['buf']='verify_tokens'
            if arg.get('buf')=='h_norm': arg['buf']='spec_target_h'
        c=dsa_change(c,rows,B('valid'),B('positions'),B('row_lens'),B('slot'),B('fa3_cu'))
        if op.startswith('kda_'):
            layer=int(original['label'].split('.')[0][1:]); k=layer-layer//4
            for arg in c['args']:
                name=arg.get('buf')
                if name in ('kda_F','kda_qkvc','kda_forget'):
                    name,cols={'kda_F':('spec_F',3336),'kda_qkvc':('spec_Q',3072),'kda_forget':('spec_A',1024)}[name]
                    arg['buf']=name; arg['offset']=k*32*cols*2
            if op=='kda_conv':
                c['op']='spec_conv_verify'; c['args']=c['args'][:5]+[B('valid'),i32(rows)]
            if op=='kda_delta':
                c['op']='spec_delta_verify'; c['args'][-1]=TV
                kda.append((layer,k))
        if op in ('moe_ar','mlp_ar','dsa_ar'): c=ar(c['label'],B('sub_out'),TV)
        cc.append(c)
        if op=='kda_o_proj_ar': cc.append(ar(c['label']+'.ar',B('sub_out'),TV))
    cc.append(call('accept_commit','spec_accept_carry_repair',
                   B('drafts'),B('verify_tokens'),B('valid'),B('spec_target_h'),
                   ST('mtp_hidden'),B('mtp_hidden_lines'),B('round_tokens'),
                   B('nacc'),B('advance_cu'),B('repair_ids'),B('repair_valid'),
                   SV,i32(rows)))
    for layer,k in kda:
        ff=B('spec_F',k*32*3336*2); qq=B('spec_Q',k*32*3072*2); aa=B('spec_A',k*32*1024*2)
        cc += [call(f'advance.{layer}.pack','spec_pack_kda',ff,qq,aa,B('pack_F'),B('pack_Q'),B('pack_A'),B('nacc'),B('advance_cu'),i32(rows)),
               call(f'advance.{layer}.delta','spec_delta_advance',B(f'layers.{layer}.A_log'),B('pack_A'),B(f'layers.{layer}.dt_bias'),
                    B('pack_Q'),B('pack_F'),B('advance_o'),ST('kda_ssm'),B('kda_ssm_lines',k*64),B('advance_cu'),TV),
               call(f'advance.{layer}.conv','spec_conv_advance',ff,ST('kda_conv'),B('kda_conv_lines',k*64),B('nacc'),B('valid'),i32(rows))]
    cc += draft('repair.',B('repair_ids'),B('spec_target_h'),rows,B('positions'),B('row_lens'),B('slot'),B('repair_valid'),False)
    m['programs'][f'round_k{steps}']={'batch':{'groups':groups,'rows':rows},'graph':True,'calls':cc}
    # Both AR allocations share one sticky error word; serving polls it.
    m['buffers']['ar_error']['fill']='error'
    used={c['op'] for p in m['programs'].values() for c in p['calls']}
    m['ops']={n:o for n,o in m['ops'].items() if n in used}
    # Manifest verifier checks all buffers are used. Remove unused new copies,
    # but keep the caller's once/peer allocations and all state-domain inputs.
    used_buf={a['buf'] for p in m['programs'].values() for c in p['calls'] for a in c['args'] if 'buf' in a}
    peer_of={m['buffers'][n]['of'] for n in used_buf if m['buffers'][n]['kind']=='peer'}
    used_buf |= peer_of
    m['buffers']={n:v for n,v in m['buffers'].items() if n in used_buf or v['kind']=='input'}
    assert len(kda)==34
    assert m['vars']['tokens']['max']==32 and m['vars']['seqs']['max']==16
    assert groups*rows<=32, f'round_k{steps}: {groups}x{rows} exceeds the 32-token cap'
    assert all(len(c['args'])==len(m['ops'][c['op']]['params']) for p in m['programs'].values() for c in p['calls'])
    return m


if __name__=='__main__': main()
