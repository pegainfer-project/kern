"""DSA decode v2; no integration side effects and no checkpoint conversion.

ops(cfg) is an absorb-only, same-interface replacement for ops_dsa.ops(cfg).
fuse_manifest(m) optionally fuses the remaining glue in a RAW gen.py manifest,
before wire_allreduce/check_invariants/resolve_modules/lower_wire.
Run with -B -m glm53.ops_dsa_v2 --audit or --test. See dsa_fusion.md.
"""
import os
from copy import deepcopy
from pathlib import Path
from . import ops_dsa as baseline
from .ops_common import S, a, scr, var, cdiv, mul, i32, handwritten, HAND_DIR

BF = 'in buffer<bf16>'
BO = 'out buffer<bf16>'
FX = 'in buffer<f32>'
FO = 'out buffer<f32>'
IX = 'in buffer<i32>'
IO = 'out buffer<i32>'
STATE = 'inout state'

# Source/cubin contract: name -> (pointer count, trailing i32 count, threads).
ABI = {
    'glm53_dsa_w_kc_v2': (3, 1, 128),
    'glm53_dsa_w_vc_v2': (3, 1, 128),
    'glm53_dsa_norm_store_v2': (8, 0, 1024),
    'glm53_dsa_had_quant_v2': (3, 0, 128),
    'glm53_dsa_xproj_v2': (6, 1, 128),
    'glm53_dsa_xepilogue_v2': (14, 1, 128),
    'glm53_dsa_step_v2': (5, 1, 32),
    'glm53_dsa_topk_v2': (5, 2, 1024),
}


def _launch(entry, grid, args, pdl=False):
    ptrs, ints, threads = ABI[entry]
    assert len(args) == ptrs + ints
    module = 'glm53_dsa_topk_v2' if entry.endswith('topk_v2') else 'glm53_dsa_v2'
    launch = handwritten(module, entry, ['buffer'] * ptrs + ['i32'] * ints,
                         [threads, 1, 1], grid, args)
    # Conservative default, compatible with gen.py's current PDL allowlist.
    # Every new entry has a wait at entry and release after all data accesses.
    if pdl:
        launch['pdl'] = True
    return launch


def ops(cfg):
    """Replace only 16 extern launches with two grouped launches per layer."""
    out = baseline.ops(cfg)
    for suffix, n in [('kc', 512), ('vc', 256)]:
        out[f'dsa_w_{suffix}'] = {
            'params': [BF, BF, BO, 'i32'],
            'impl': {'launches': [_launch(f'glm53_dsa_w_{suffix}_v2',
                [n // 16, 8, cdiv(S, 4)], [a(i) for i in range(4)])]},
        }
    # Baseline allocates eight entries for per-sequence scheduler arrays.
    # prepare writes one entry/group. Keep 16 here for the requested bs16.
    for key in ('nsd', 'nmb', 'vbi'):
        out['dsa_attn']['impl']['scratch'][key]['shape'] = [16]
    # Explicit guard for the load-bearing, already-fixed 1d2d scale ABI.
    for key in ('dsa_qkv_a', 'dsa_q_b', 'dsa_o_proj'):
        launch = out[key]['impl']['launches'][0]
        assert launch['args'][0] == a(3), 'raw ptr must be WEIGHT scales'
        assert launch['args'][8]['pack']['fields'][0]['tensormap']['param'] == 2
    return out


def _fused_ops(bt_cols):
    return {
        'dsa_norm_store_v2': {
            'params': [BF, BF, BF, BO, 'out buffer<fp8>', FO, STATE, IX],
            'impl': {'launches': [_launch('glm53_dsa_norm_store_v2', [S, 1, 1],
                                          [a(i) for i in range(8)])]},
        },
        'dsa_had_quant_v2': {
            'params': [BF, 'out buffer<fp8>', FO],
            'impl': {'launches': [_launch('glm53_dsa_had_quant_v2', [mul(S, 32), 1, 1],
                                          [a(i) for i in range(3)])]},
        },
        'dsa_xside_v2': {
            # x,wk,gate,head_w,norm_w,norm_b,ape,idx,tail,bt,lines,pos,seq,
            # valid,q_scales,head_weights. Scratch is op-private, not per call.
            'params': [BF, BF, BF, FX, FX, FX, FX, STATE, STATE, IX, IX, IX, IX, IX, FX, FO],
            'impl': {
                'scratch': {'kg': {'dtype': 'bf16', 'shape': [16, 256]},
                            'head': {'dtype': 'f32', 'shape': [16, 32]}},
                'launches': [
                    _launch('glm53_dsa_xproj_v2', [72, cdiv(S, 4), 1],
                            [a(0), a(1), a(2), a(3), scr('kg'), scr('head'), var(S)]),
                    _launch('glm53_dsa_xepilogue_v2', [S, 1, 1],
                            [a(7), a(8), scr('kg'), scr('head'), a(4), a(5), a(6),
                             a(9), a(10), a(11), a(12), a(13), a(14), a(15), i32(bt_cols)]),
                ],
            },
        },
        'dsa_step_v2': {
            'params': [IX, IO, IO, IO, IO],
            'impl': {'launches': [_launch('glm53_dsa_step_v2', [1, 1, 1],
                                          [a(i) for i in range(5)] + [var(S)])]},
        },
        'dsa_topk_v2': {
            'params': [FX, IX, IX, IX, IO],
            'impl': {'launches': [_launch('glm53_dsa_topk_v2', [S, 1, 1],
                                          [a(i) for i in range(5)] + [i32(bt_cols * 64), i32(bt_cols)])]},
        },
    }


def fuse_manifest(m, *, xside=True, topk=True):
    """Fuse only decode calls; no files changed. Defaults are candidates, not
    performance-certified. Set xside=False/topk=False for staged A/B gates.
    Refuse unexpected layouts, partial graphs, or probes inside DSA cuts.
    """
    assert 'dsa_step_v2' not in m['ops'], 'Already fused'
    assert m['vars']['seqs']['max'] <= 16 and m['vars']['tokens']['max'] <= 16
    assert m['programs']['decode']['batch']['rows'] == 1
    assert m['states']['kv']['bytes_per_token'] == 11264
    assert m['states']['idx']['bytes_per_token'] == 363
    assert m['states']['idx_tail']['bytes_per_seq'] == 11 * 4096
    old = m['programs']['decode']['calls']
    prep = next(c for c in old if c['op'] == 'dsa_prep')
    bt_cols = prep['args'][6]['i32']
    extras = _fused_ops(bt_cols)
    starts = [i for i, c in enumerate(old) if c['op'] == 'dsa_qkv_a']
    if not starts:
        return m
    remove = set()
    replace = {}
    def call(label, op, args):
        return {'label': label, 'op': op, 'args': deepcopy(args)}
    for start in starts:
        end = next(i for i in range(start, len(old)) if old[i]['op'] == 'dsa_ar')
        chunk = old[start:end+1]
        assert not any(c['op'].startswith('debug_') for c in chunk), 'Probe inside DSA cut'
        by = {c['op']: (start+j, c) for j, c in enumerate(chunk)
              if c['op'] != 'head_rms_norm'}
        norms = [(start+j, c) for j, c in enumerate(chunk) if c['op'] == 'head_rms_norm']
        assert len(norms) == 2
        qi, qn = norms[0]; ki, kn = norms[1]
        assert qn['args'][4]['i32'] == 1536 and kn['args'][4]['i32'] == 512
        assert qn['args'][5]['i32'] == kn['args'][5]['i32'] == 2048
        assert kn['args'][0] == dict(qn['args'][0], offset=3072)
        quant_i, quant = by['dsa_quant_q']; store_i, store = by['dsa_kv_store']
        assert quant['args'][0] == qn['args'][2] and store['args'][1] == kn['args'][2]
        replace[qi] = call(qn['label'].rsplit('.', 1)[0]+'.norm_store_v2', 'dsa_norm_store_v2',
                          [qn['args'][0], qn['args'][1], kn['args'][1], qn['args'][2],
                           quant['args'][1], quant['args'][2], store['args'][0], store['args'][2]])
        remove.update((ki, quant_i, store_i))
        hi, had = by['dsa_hadamard']; ai, aq = by['dsa_act_quant']
        assert had['args'][1] == aq['args'][0]
        replace[hi] = call(had['label']+'q_v2', 'dsa_had_quant_v2',
                           [had['args'][0], aq['args'][1], aq['args'][2]])
        remove.add(ai)
        if xside:
            wi, wk = by['dsa_wk']; gi, gate = by['dsa_kpool_gate']
            ni, norm = by['dsa_k_norm']; pi, pool = by['dsa_kpool_update']
            oi, head = by['dsa_weights_proj']
            assert wk['args'][0] == gate['args'][0] == head['args'][0]
            assert hi < pi < oi, 'q scales must precede fused xside'
            assert norm['args'][0] == norm['args'][3] == wk['args'][2] == pool['args'][2]
            assert gate['args'][2] == pool['args'][3] and head['args'][2] == aq['args'][2]
            replace[pi] = call(pool['label']+'.xside_v2', 'dsa_xside_v2',
                [wk['args'][0], wk['args'][1], gate['args'][1], head['args'][1],
                 norm['args'][1], norm['args'][2], pool['args'][4],
                 pool['args'][0], pool['args'][1], pool['args'][5], pool['args'][6],
                 pool['args'][7], pool['args'][8], pool['args'][9], head['args'][2], head['args'][3]])
            remove.update((wi, gi, ni, oi))
        if topk:
            ti, tk = by['dsa_topk']; ci, clamp = by['dsa_clamp']
            assert clamp['args'][0] == tk['args'][4]
            args = deepcopy(tk['args']);args[2] = deepcopy(prep['args'][1])
            replace[ti] = call(tk['label']+'.v2', 'dsa_topk_v2', args)
            remove.add(ci)
        remove.add(by['dsa_logits_meta'][0])
    meta = next(c for c in old if c['op'] == 'dsa_logits_meta')
    prep_i = old.index(prep)
    if topk:
        replace[prep_i] = call('dsa_step_v2', 'dsa_step_v2',
                               [prep['args'][0], *prep['args'][2:5], meta['args'][1]])
    else:
        # Baseline topk still needs the slot table. Hoist metadata only.
        replace[prep_i] = deepcopy(prep)
    new = []
    for i, c in enumerate(old):
        if i in remove:
            continue
        new.append(replace.get(i, c))
        if not topk and i == prep_i:
            new.append(call('dsa_meta_once_v2', 'dsa_logits_meta', meta['args']))
    m['programs']['decode']['calls'] = new
    used = {c['op'] for p in m['programs'].values() for c in p['calls']}
    # Mutate this SAME dictionary: gen.py holds it as both m['ops'] and allops.
    m['ops'].update({k:v for k,v in extras.items() if k in used})
    for name in list(m['ops']):
        if name not in used:
            del m['ops'][name]
    used_bufs = {a['buf'] for p in m['programs'].values() for c in p['calls']
                 for a in c['args'] if 'buf' in a}
    for name in ('knope', 'iqh', 'ik', 'gs', 'slot_table'):
        if name not in used_bufs:
            m['buffers'].pop(name, None)
    return m


def audit():
    """No GPU: verify every new cubin parameter offset/size and PDL SASS."""
    import re, subprocess, hashlib
    for module in ('glm53_dsa_v2', 'glm53_dsa_topk_v2'):
        path = HAND_DIR / (module+'.cubin')
        elf = subprocess.check_output(['/usr/local/cuda-13.0/bin/cuobjdump', '-elf', str(path)], text=True)
        sass = subprocess.check_output(['/usr/local/cuda-13.0/bin/cuobjdump', '-sass', str(path)], text=True)
        sections = re.split(r'\.nv\.info\.(\w+)', elf)
        found = {}
        for name, text in zip(sections[1::2], sections[2::2]):
            if name not in ABI:
                continue
            params = {int(n,16):(int(o,16),int(z,16)) for n,o,z in
                      re.findall(r'Ordinal\s*:\s*0x([0-9a-f]+)\s+Offset\s*:\s*0x([0-9a-f]+)\s+Size\s*:\s*0x([0-9a-f]+)', text)}
            if params:
                found[name] = params
        entries = {n:v for n,v in ABI.items() if (n.endswith('topk_v2')) == (module.endswith('topk_v2'))}
        for name, (ptrs, ints, _) in entries.items():
            sizes = [8]*ptrs + [4]*ints
            offset = 0; expected = {}
            for i,size in enumerate(sizes):
                offset = (offset+size-1)//size*size;expected[i]=(offset,size);offset+=size
            assert found.get(name) == expected, (name, found.get(name), expected)
            code = sass.split('Function : '+name)[1].split('Function :')[0]
            assert 'ACQBULK' in code and 'PREEXIT' in code, name
            print(name, 'ABI OK', sizes, 'PDL wait/release OK')
        print(module, hashlib.sha256(path.read_bytes()).hexdigest())


def audit_inherited():
    # All manifest-wired DSA params, including the 2944/232-byte FA3 packs.
    # Existing extern ABIs remain baseline._gemm_tn/allreduce_bf16 unchanged.
    import re, subprocess
    from .ops_common import DUMP_DIR
    cache = {}
    allops = ops({'bt_cols':1024, 'st_cols':262144})
    allops.update(_fused_ops(1024))
    count = 0
    for op in allops.values():
        for launch in op['impl']['launches']:
            if launch['entry'].startswith('extern:'):
                continue
            path = (HAND_DIR if (HAND_DIR/launch['cubin']).exists() else DUMP_DIR)/launch['cubin']
            if path not in cache:
                elf = subprocess.check_output(['/usr/local/cuda-13.0/bin/cuobjdump','-elf',str(path)],text=True)
                parts = re.split(r'\.nv\.info\.(\w+)', elf)
                cache[path] = {}
                for name,text in zip(parts[1::2],parts[2::2]):
                    params = {int(n,16):int(z,16) for n,z in re.findall(
                        r'Ordinal\s*:\s*0x([0-9a-f]+)\s+Offset\s*:\s*0x[0-9a-f]+\s+Size\s*:\s*0x([0-9a-f]+)',text)}
                    if params:cache[path][name] = params
            expected = {}
            for i,t in enumerate(launch['params']):
                expected[i] = int(t[6:-1]) if t.startswith('bytes<') else (8 if t in ('buffer','i64','u64') else 4)
            actual = cache[path].get(launch['entry'])
            assert expected == actual, (launch['entry'], expected, actual)
            count += 1
    print('All DSA manifest cubin ABI sizes OK:', count, 'launches')
    aq = allops['dsa_act_quant']['impl']['launches'][0]
    if aq['args'][3] == i32(32):
        print('WARNING: inherited dsa_act_quant M=32 masks S>1; owner fix is '
              'expr(mul(S, IDX_HEADS)). Stage A ops() remains absorb-only.')


def cpu_check():
    """Build all four staged manifests in memory; no files or CUDA context."""
    from . import gen
    import sys
    saved_ops, saved_wire = gen.ops_dsa, gen.wire_allreduce
    try:
        gen.ops_dsa = sys.modules[__name__]
        for xside, topk in ((False, False), (True, False), (False, True), (True, True)):
            def wire(m, *args, **kwargs):
                fuse_manifest(m, xside=xside, topk=topk)
                return saved_wire(m, *args, **kwargs)
            gen.wire_allreduce = wire
            m = gen.build()
            print('CPU manifest OK', {'xside':xside, 'topk':topk,
                'ops':len(m['ops']), 'calls':len(m['programs']['decode']['calls'])})
    finally:
        gen.ops_dsa, gen.wire_allreduce = saved_ops, saved_wire


def _gpu_idle_check(gpu):
    """Fail closed: never run on top of another job on the selected GPU or a shared serving process."""
    import subprocess, os
    if os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise SystemExit('Unset CUDA_VISIBLE_DEVICES; --gpu must name the physical device index.')
    tmux = subprocess.run(['tmux', 'ls'], capture_output=True, text=True)
    print(tmux.stdout or tmux.stderr, end='')
    query = subprocess.check_output(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used',
                                     '--format=csv,noheader,nounits'], text=True)
    print(query, end='')
    procs = subprocess.check_output(['ps', '-eo', 'args'], text=True)
    busy = [line for line in procs.splitlines() if
            ('sglang.launch_server' in line or '/kern-serve ' in line or '/kern-run bench ' in line or '/kern bench ' in line or '/kern test ' in line)
            and not any(s in line for s in ('tmux ', 'sh -c', 'bash -', 'grep '))]
    used = any(int(line.split(',')[1]) > 0 or int(line.split(',')[2]) > 64
               for line in query.strip().splitlines() if int(line.split(',')[0]) == gpu)
    if busy or used:
        raise SystemExit('GPU tests deferred: another GPU job/server is active. Retry after its owner stops it.')


def gpu_test(gpu=0, bench=False):
    """Short direct-cubin tests; no JIT or filesystem outputs. Assertions are
    strict on glue/state, tolerance-based on changed GEMV reduction trees.
    Numerical failures are collected; CUDA errors abort immediately.
    Torch eight-mm timing is a proxy, NOT the kern cuBLASLt baseline.
    """
    _gpu_idle_check(gpu)
    import ctypes as ct
    import struct
    import torch
    from .ops_common import DUMP_DIR
    torch.set_num_threads(1)  # short CPU top-k references; do not occupy all host cores
    torch.cuda.set_device(gpu)
    torch.manual_seed(5303)
    device = torch.device('cuda', gpu)
    torch.empty(1, device=device)  # use Torch's current primary context
    cu = ct.CDLL('libcuda.so.1')
    def check(rc):
        if rc:
            text=ct.c_char_p();cu.cuGetErrorString(rc,ct.byref(text))
            raise RuntimeError(f'CUDA {rc}: {text.value}')
    check(cu.cuInit(0))
    modules, functions = {}, {}
    cu.cuModuleLoad.argtypes = [ct.POINTER(ct.c_void_p), ct.c_char_p]
    cu.cuModuleGetFunction.argtypes = [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_char_p]
    cu.cuLaunchKernel.argtypes = [ct.c_void_p] + [ct.c_uint]*7 + [ct.c_void_p,
                                 ct.POINTER(ct.c_void_p), ct.c_void_p]
    def launch(name, args, grid, block, module=None, smem=0):
        if module is None:
            module=HAND_DIR/('glm53_dsa_topk_v2.cubin' if name.endswith('topk_v2') else 'glm53_dsa_v2.cubin')
        module=str(module)
        if module not in modules:
            handle=ct.c_void_p();check(cu.cuModuleLoad(ct.byref(handle),module.encode()));modules[module]=handle
        key=(module,name)
        if key not in functions:
            f=ct.c_void_p();check(cu.cuModuleGetFunction(ct.byref(f),modules[module],name.encode()));functions[key]=f
        holders=[]
        for arg in args:
            if isinstance(arg, torch.Tensor):
                holders.append(ct.c_uint64(arg.data_ptr()))
            elif isinstance(arg, int):
                holders.append(ct.c_int32(arg))
            elif isinstance(arg, float):
                holders.append(ct.c_float(arg))
            elif isinstance(arg, (bytes, bytearray)):
                holders.append(ct.create_string_buffer(bytes(arg)))
            else:
                holders.append(arg)
        argv=(ct.c_void_p*len(holders))(*(ct.addressof(v) for v in holders))
        stream=ct.c_void_p(torch.cuda.current_stream().cuda_stream)
        check(cu.cuLaunchKernel(functions[key],*grid,block,1,1,smem,stream,argv,None))
    def rand(*shape):
        return torch.randn(shape,device=device,dtype=torch.bfloat16)
    # Record numerical failures and continue through the full matrix. CUDA
    # faults still abort: continuing after an illegal access is not safe.
    failures, results = [], {}
    case = 'setup'
    def require(ok, label, details=None):
        counts = results.setdefault(case, [0, 0])
        counts[0] += 1
        if not bool(ok):
            counts[1] += 1
            failures.append((case, label, details))
            print('FAIL', failures[-1], flush=True)
    def exact(x,y,label):
        torch.cuda.synchronize()
        same = torch.equal(x,y)
        require(same, label, None if same else {'mismatches':int((x!=y).sum())})
    def close(x,y,label):
        torch.cuda.synchronize()
        x,y=x.float(),y.float();delta=x-y
        rel=(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(y).clamp_min(1e-20)).item()
        maximum=delta.abs().max().item();scale=y.abs().max().item()
        print(label, {'rel_l2':rel,'max_abs':maximum,'exact_fraction':(x==y).float().mean().item()})
        require(rel<1e-3 and maximum<max(1e-5,scale*0.01), label, {'rel_l2':rel,'max_abs':maximum})
    def graph_us(fn):
        for _ in range(3):fn()
        torch.cuda.synchronize()
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(20):fn()
        a0,b0=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a0.record()
        for _ in range(10):g.replay()
        b0.record();b0.synchronize()
        return a0.elapsed_time(b0)*1000/200
    cfg={'bt_cols':1024,'st_cols':262144}
    refops=baseline.ops(cfg)
    def mined_launch(op,args):
        spec=refops[op]['impl']['launches'][0]
        return launch(spec['entry'],args,(1,1,1),spec['block'][0],DUMP_DIR/spec['cubin'],spec.get('shared_mem',0))
    def had_quant_check(x, tag):
        rows = x.shape[0]
        xh = torch.empty_like(x)
        # 0x7f is E4M3 NaN: finite test inputs must overwrite every active
        # byte. Extra token row checks both reference and fused write bounds.
        y = torch.full((rows+1,32,128),127,dtype=torch.uint8,device=device)
        yr, ys = y.clone(), y.clone()
        scale = torch.full((rows+1,32),-1.,device=device)
        sr, ss = scale.clone(), scale.clone()
        hp = struct.pack('<iiiiqqfIQQ',rows*32,128,7,0,128,128,
                         0.08838834764831844,0,x.data_ptr(),xh.data_ptr())
        spec = refops['dsa_hadamard']['impl']['launches'][0]
        launch(spec['entry'],[hp],(rows*32,1,1),16,DUMP_DIR/spec['cubin'],512)
        spec = refops['dsa_act_quant']['impl']['launches'][0]
        def quant(inp, out, scales, m, grid):
            launch(spec['entry'],[inp,out,scales,m,128,ct.c_int64(0),ct.c_int64(0)],
                   (grid,1,1),128,DUMP_DIR/spec['cubin'],128)
        if rows == 2 and tag == 'random':
            # Regression proof for the original whole-row mismatch. The
            # inherited M=32 launch writes token 0 and masks every later token.
            quant(xh,yr,sr,32,rows)
            require(bool((yr[0]!=127).all()) and bool((sr[0]>0).all()),
                    'legacy M=32 writes token 0')
            require(bool((yr[1:]==127).all()) and bool((sr[1:]==-1).all()),
                    'legacy M=32 leaves later tokens unwritten')
            print('PROOF: M=32 leaves 4096 active FP8 bytes and 32 scales unwritten at S=2')
            yr.fill_(127);sr.fill_(-1)
        # M is flattened head-row count, NOT heads per token. The pinned
        # Triton TTIR uses (pid_x*32 + arange(32)) < M for both output masks.
        quant(xh,yr,sr,rows*32,rows)
        # Independent addressing oracle: rebase pointers for each token and
        # use one CTA with M=32. This must equal the single batched launch.
        for r in range(rows):
            quant(xh[r],ys[r],ss[r],32,1)
        exact(yr,ys,'quant batched/per-token '+tag)
        exact(sr,ss,'scales batched/per-token '+tag)
        require(bool((yr[:rows]!=127).all()) and bool((sr[:rows]>0).all()),
                'reference writes all active outputs '+tag)
        launch('glm53_dsa_had_quant_v2',[x,y,scale],(rows*32,1,1),128)
        exact(y,yr,'Hadamard FP8 + guards '+tag)
        exact(scale,sr,'Hadamard scale + guards '+tag)
        require(bool((y[rows]==127).all()) and bool((scale[rows]==-1).all()),
                'Hadamard output guards '+tag)

    for rows in (1,2,3,4,5,8,15,16):
        for suffix,n,k in (('kc',512,256),('vc',256,512)):
            case = f'absorb {suffix} S={rows}'
            x,w=rand(rows,8,k),rand(8,n,k)
            out=torch.full((rows+1,8,n),-77.,dtype=torch.bfloat16,device=device)
            expected=torch.empty((rows,8,n),dtype=torch.bfloat16,device=device)
            def old_bmm():
                for h in range(8):torch.mm(x[:,h,:],w[h].t(),out=expected[:,h,:])
            def new_bmm():launch('glm53_dsa_w_'+suffix+'_v2',[x,w,out,rows],(n//16,8,(rows+3)//4),128)
            old_bmm();new_bmm();close(out[:rows],expected,f'absorb {suffix} bs{rows}')
            require((out[rows]==-77).all(), 'absorb guard row')
            # Replay after in-place input changes: graph must not bake rows' values.
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):new_bmm()
            x.mul_(0.5);graph.replay();old_bmm();close(out[:rows],expected,f'graph {suffix} bs{rows}')
            if bench and rows in (1,4,8,16):
                print('timing absorb proxy us',rows,suffix,{'8mm':graph_us(old_bmm),'v2':graph_us(new_bmm)})
        case = f'norm/quant/KV S={rows}'
        # Baseline RMS + real mined quant + baseline store versus one fused CTA.
        qkv,qw,kw=rand(16,2048),rand(1536),rand(512)
        qa=torch.full((16,1536),-13.,dtype=torch.bfloat16,device=device);qar=qa.clone()
        qa8=torch.full((16,1536),17,dtype=torch.uint8,device=device);q8r=qa8.clone()
        sf=torch.full((12,16),-11.,device=device);sfr=sf.clone()
        kn=torch.empty((rows,512),dtype=torch.bfloat16,device=device)
        slots=torch.arange(1,rows+1,device=device,dtype=torch.int32)
        kv=torch.full(((rows+2)*11264,),19,dtype=torch.uint8,device=device);kvr=kv.clone()
        # Layer 10 offset proves nonzero per-layer addressing and guard bytes.
        off=10*1024
        launch('glm53_rms_norm',[qkv,qw,qar,1e-5,1536,2048,1536],(rows,1,1),1024,HAND_DIR/'glm53_misc.cubin')
        launch('glm53_rms_norm',[qkv.flatten()[1536:],kw,kn,1e-5,512,2048,512],(rows,1,1),1024,HAND_DIR/'glm53_misc.cubin')
        qp=struct.pack('<QqqQqqQIIIIII',qar.data_ptr(),0,1536,q8r.data_ptr(),0,1536,sfr.data_ptr(),0,1,16,12,rows,1536)
        spec=refops['dsa_quant_q']['impl']['launches'][0]
        launch(spec['entry'],[qp],((rows*3+7)//8,1,1),256,DUMP_DIR/spec['cubin'])
        launch('glm53_kv_store',[kvr[off:],kn,slots,ct.c_int64(11264),ct.c_int64(0)],(rows,1,1),128,HAND_DIR/'glm53_dsa.cubin')
        launch('glm53_dsa_norm_store_v2',[qkv,qw,kw,qa,qa8,sf,kv[off:],slots],(rows,1,1),1024)
        exact(qa,qar,'qa RMS');exact(qa8,q8r,'q fp8');exact(sf,sfr,'SFA');exact(kv,kvr,'KV with guard bytes')
        case = f'Hadamard/quant S={rows}'
        had_quant_check(rand(rows,32,128), 'random')
        # Impulse locations across columns expose sign/permutation mistakes.
        # Include zero, constant, alternating-sign and widely scaled heads.
        pattern = torch.zeros((rows*32,128),dtype=torch.bfloat16,device=device)
        heads = torch.arange(rows*32,device=device)
        pattern[heads,heads%128] = 1
        pattern[0].zero_()
        pattern[1].fill_(1)
        pattern[2] = (1-2*(torch.arange(128,device=device)%2)).to(torch.bfloat16)
        amplitudes = torch.tensor([1e-8,1e-4,1.,128.],device=device)
        pattern = (pattern.float()*amplitudes[heads%4,None]).to(torch.bfloat16)
        had_quant_check(pattern.reshape(rows,32,128), 'patterns/scales')
        case = f'step/MQA S={rows}'
        # Scheduler: mixed lengths, identity boundary and 35k-pool context.
        values=[1,3,4,255,256,2048,2051,2052,141123,141120,5,8,512,513,4096,262143]
        seq=torch.tensor(values[:rows],dtype=torch.int32,device=device)
        pools=torch.empty_like(seq);ctx=torch.empty_like(seq);lens=torch.empty_like(seq)
        sched=torch.empty((133,2),dtype=torch.int32,device=device);schedr=torch.empty_like(sched)
        launch('glm53_dsa_step_v2',[seq,pools,ctx,lens,sched,rows],(1,1,1),32)
        exact(pools,seq//4,'pool lengths');exact(ctx,(seq//4).clamp_min(1),'pool ctx')
        exact(lens,((seq//4)*4).clamp_max(2048)+seq%4,'attention lengths')
        mined_launch('dsa_logits_meta',[rows,1,ct.c_uint8(1),ctx,ct.c_int64(0),schedr])
        exact(sched,schedr,'MQA metadata')
        case = f'xproj S={rows}'
        # X-side projection. Head weights deliberately arbitrary f32: no lossy
        # assumption that they are BF16-representable is needed by v2.
        xx,wk,gate=rand(rows,4096),rand(128,4096),rand(128,4096)
        hw=torch.randn((32,4096),device=device)
        kg=torch.empty((rows,256),dtype=torch.bfloat16,device=device)
        head=torch.empty((rows,32),device=device)
        launch('glm53_dsa_xproj_v2',[xx,wk,gate,hw,kg,head,rows],(72,(rows+3)//4,1),128)
        close(kg[:,:128],xx@wk.t(),'wk');close(kg[:,128:],xx@gate.t(),'gate')
        qs=torch.rand((rows,32),device=device)+0.01
        hpref=torch.empty_like(head)
        launch('glm53_weights_proj',[xx,hw,qs,hpref],(rows,1,1),1024,HAND_DIR/'glm53_dsa.cubin')
        hwant=((head*0.17677669529663687)*qs)*0.08838834764831844
        exact(hwant,hpref,'head projection/reduction')
        case = f'xepilogue/state S={rows}'
        # Isolate the epilogue using identical materialized kg: exact state
        # comparisons must not conflate the preceding changed GEMV reductions.
        nw,nb=torch.randn(128,device=device),torch.randn(128,device=device)
        ape=torch.randn((4,128),device=device)
        bt=torch.arange(rows*2,device=device,dtype=torch.int32).reshape(rows,2)
        lines=torch.arange(1,rows+1,device=device,dtype=torch.int32)
        tail=rand((rows+1)*2048).view(torch.uint8);tailr=tail.clone()
        idx=torch.full((rows*2*92928,),23,dtype=torch.uint8,device=device);idxr=idx.clone()
        out=torch.empty_like(head)
        # At bs1 a narrow row is already contiguous; clone to prevent the
        # in-place baseline norm from also changing v2 projection scratch.
        kval=kg[:,:128].clone(memory_format=torch.contiguous_format);score=kg[:,128:].contiguous()
        launch('glm53_layer_norm_128',[kval,nw,nb,kval,1e-6],(rows,1,1),128,HAND_DIR/'glm53_dsa.cubin')
        # 255 and 259 close pools across token-page and ring-of-eight boundaries;
        # 254/256 remain open. Invalid row must not write idx/tail.
        for posvalue in (254,255,256,259,263):
            pos=torch.full((rows,),posvalue,dtype=torch.int32,device=device);sq=pos+1
            valid=torch.ones_like(pos)
            if rows>1:valid[-1]=0
            launch('glm53_kpool_update',[idxr[10*8448:],tailr,kval,score,ape,bt,lines,pos,sq,valid,2],(rows,1,1),128,HAND_DIR/'glm53_dsa.cubin')
            launch('glm53_dsa_xepilogue_v2',[idx[10*8448:],tail,kg,head,nw,nb,ape,bt,lines,pos,sq,valid,qs,out,2],(rows,1,1),128)
            exact(tail,tailr,'tail ring');exact(idx,idxr,'idx + scale + layer/page guards');exact(out,hwant,'head epilogue')
        print('Completed absorb/glue S=',rows,flush=True)
    # Top-k reference on CPU: deterministic score-descending/id-ascending
    # selection, then ascending-id output. Includes ties, negatives, +inf,
    # -inf, zero pools, all four tail lengths, and no logits read on identity.
    for rows in (1,2,3,4,5,8,15,16):
        case = f'topk S={rows}'
        bt=torch.stack([torch.randperm(1024,device=device,dtype=torch.int32) for _ in range(rows)])
        logits=torch.randn((rows,65536),device=device)
        dst=torch.full((rows+1,2051),-777,dtype=torch.int32,device=device)
        for length in (0,1,511,512,513,35280,65535):
            for mode in ('random','ties'):
                scores=logits if mode=='random' else logits.round()
                if length<=512:scores=torch.full_like(scores,float('nan'))
                else:
                    scores[0,0]=float('inf');scores[0,1]=-float('inf')
                seq=torch.tensor([length*4+(r%4) for r in range(rows)],device=device,dtype=torch.int32)
                pools=torch.full((rows,),length,device=device,dtype=torch.int32)
                def top():launch('glm53_dsa_topk_v2',[scores,pools,bt,seq,dst,65536,1024],(rows,1,1),1024)
                top();snapshot=dst.clone()
                host_scores=scores.cpu();host_bt=bt.cpu()
                want=torch.zeros((rows,2051),dtype=torch.int32)
                for r in range(rows):
                    ids=torch.arange(length) if length<=512 else torch.argsort(host_scores[r,:length],descending=True,stable=True)[:512].sort().values
                    tokens=(ids[:,None]*4+torch.arange(4)).flatten()
                    tokens=torch.cat([tokens,torch.arange(length*4,length*4+r%4)])
                    want[r,:len(tokens)]=host_bt[r,tokens//256]*256+tokens%256
                exact(dst[:rows],want.to(device),f'topk selection/map/pad pools={length} {mode}')
                for _ in range(3):top()
                exact(dst,snapshot,'topk deterministic replay/guard')
                require((dst[rows]==-777).all(), 'topk guard row')
                if bench and length==35280 and mode=='random':print('topk us',rows,graph_us(top))
        # All ties must select pools 0..511, never arbitrary atomic order.
        scores=torch.zeros_like(logits);pools.fill_(35280);seq.fill_(141120)
        top()
        tokens=torch.arange(2048,device=device)
        want=bt[:,tokens//256]*256+tokens%256
        exact(dst[:rows,:2048],want,'topk all equal')
        require((dst[:rows,2048:]==0).all(), 'topk all-equal zero padding')
        # A captured graph must re-read lengths across the identity boundary.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):top()
        pools.fill_(512);seq.fill_(2051);graph.replay()
        tokens=torch.arange(2051,device=device)
        exact(dst[:rows],bt[:,tokens//256]*256+tokens%256,'topk graph identity 512')
        scores.zero_();scores[:,512]=1
        pools.fill_(513);seq.fill_(2052);graph.replay()
        tokens=torch.cat([torch.arange(2044,device=device),torch.arange(2048,2052,device=device)])
        want=torch.zeros((rows,2051),dtype=torch.int32,device=device)
        want[:,:2048]=bt[:,tokens//256]*256+tokens%256
        exact(dst[:rows],want,'topk graph real selection 513')
        require((dst[rows]==-777).all(), 'topk graph guard row')
        print('Completed topk S=',rows,flush=True)
    torch.cuda.synchronize()
    print('\nTEST MATRIX: case | checks | failures')
    for name,(checks,failed) in results.items():
        print(f'{name} | {checks} | {failed}')
    print('TOTAL',sum(n for n,_ in results.values()),'checks;',len(failures),'failures')
    if failures:
        raise AssertionError(f'{len(failures)} numerical checks failed: {failures}')
    print('PASS: direct-cubin A/B, Hadamard output coverage, guards, states, determinism and graph replay')


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',action='store_true')
    parser.add_argument('--cpu-check',action='store_true')
    parser.add_argument('--test',action='store_true')
    parser.add_argument('--bench',action='store_true',help='short CUDA-graph proxy timings; implies --test')
    parser.add_argument('--gpu',type=int,default=0)
    args=parser.parse_args()
    if args.audit:
        audit()
        audit_inherited()
    if args.cpu_check:cpu_check()
    if args.test or args.bench:gpu_test(args.gpu,args.bench)
    if not any((args.audit,args.cpu_check,args.test,args.bench)):parser.print_help()


if __name__ == '__main__':
    main()
