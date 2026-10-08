#!/usr/bin/env python3
"""GPU A/B gate for DSA-indexer slab 2 (knorm+hadamard+act_quant+weights_proj).

REF = the four-call chain exactly as the round launches it (pinned cubins +
launch dicts lifted from the fused45 manifest). FUSED = glm53_slab2 one-call
op. Bitwise byte equality on ik/iq8/qs/w for rows 1..32, edge rows
(zero-amax, hot row), graph capture + changed-input replay, and a timing
table. Does NOT run serving or edit gen. JSON to stdout.
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
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from glm53 import ops_common, ops_slab2

BUNDLE = ROOT / 'kernels-glm53'


def ref_launches(manifest):
    """The four reference launches, normalized for Driver (cubin + full args)."""
    m = json.loads(Path(manifest).read_text())
    out = {}
    for opn in ('k_norm', 'hadamard', 'act_quant', 'weights_proj'):
        op = m['ops']['mtp32_dsa_' + opn]
        l = dict(op['impl']['launches'][0])
        l['cubin'] = Path(m['modules'][l.pop('module')]['source']).name
        if 'args' not in l:
            l['params'] = list(op['params'])
            l['args'] = [{'param': i} for i in range(len(op['params']))]
        out[opn] = l
    return out


def abi_check(fused_op):
    l = fused_op['impl']['launches'][0]
    path = ops_common.HAND_DIR / l['cubin']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == l['sha256']
    cuobj = next(c for c in ('/usr/local/cuda-13.0/bin/cuobjdump',
                             '/usr/local/cuda-13.3/bin/cuobjdump',
                             '/usr/local/cuda/bin/cuobjdump') if Path(c).exists())
    elf = subprocess.check_output([cuobj, '-elf', str(path)], text=True)
    marker = '\n.nv.info.' + l['entry'] + '\n'
    assert marker in elf, (path, marker)
    section = elf.split(marker, 1)[1].split('\n.nv.', 1)[0]
    fields = re.findall(r'Ordinal : (0x[\da-f]+)\s+Offset\s+: (0x[\da-f]+)\s+Size\s+: (0x[\da-f]+)',
                        section)
    got = sorted((int(i, 16), int(o, 16), int(s, 16)) for i, o, s in fields)
    expected, off = [], 0
    for i, t in enumerate(l['params']):
        size = 4 if t in ('i32', 'f32') else 8
        off = (off + (7 if size > 4 else 3)) // (8 if size > 4 else 4) * (8 if size > 4 else 4)
        expected.append((i, off, size))
        off += size
    assert got == expected, (l['entry'], got, expected)
    for p in (HERE / 'glm53_slab2.cu', HERE.parent / 'ops_slab2.py'):
        ast.parse(p.read_text()) if p.suffix == '.py' else None
    return {'sha256': l['sha256'], 'abi': got, 'block': l['block']}


def guard():
    subprocess.run(['nvidia-smi'], check=True, stdout=sys.stderr)
    for line in subprocess.check_output(['ps', '-eo', 'comm,args'], text=True).splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] in ('kern', 'kern-serve', 'kserve', 'kbench') or parts[0].startswith('sglang::'):
            raise SystemExit('GPU run deferred: ' + line)
    util = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
        text=True)
    vis = os.environ.get('CUDA_VISIBLE_DEVICES')
    keep = set(range(len(util.split()))) if not vis else {int(x) for x in vis.split(',') if x.strip()}
    busy = [i for i, v in enumerate(util.split()) if i in keep and int(v) > 3]
    if busy:
        raise SystemExit('GPU run deferred: active GPU utilization on visible ' + str(busy))


class Driver:
    def __init__(self, torch):
        self.torch = torch
        self.cuda = C.CDLL('libcuda.so.1')
        self.cache = {}

    def check(self, e):
        if e:
            msg = C.c_char_p()
            self.cuda.cuGetErrorString(e, C.byref(msg))
            raise RuntimeError((e, msg.value))

    def fn(self, l):
        key = (l['cubin'], l['entry'])
        if key not in self.cache:
            path = None
            for d in (ops_common.HAND_DIR, ops_common.DUMP_DIR, BUNDLE):
                if (d / l['cubin']).exists():
                    path = d / l['cubin']
                    break
            assert path, l['cubin']
            m = C.c_void_p()
            self.check(self.cuda.cuModuleLoad(C.byref(m), str(path).encode()))
            f = C.c_void_p()
            self.check(self.cuda.cuModuleGetFunction(C.byref(f), m, l['entry'].encode()))
            if l.get('shared_mem', 0) > 48000:
                self.check(self.cuda.cuFuncSetAttribute(f, 8, l['shared_mem']))
            self.cache[key] = (m, f)
        return self.cache[key][1]

    def ev(self, e, b):
        if isinstance(e, int):
            return e
        if isinstance(e, str):
            return b
        if 'mul' in e:
            return self.ev(e['mul'][0], b) * self.ev(e['mul'][1], b)
        if 'ceil_div' in e:
            return (self.ev(e['ceil_div'][0], b) + e['ceil_div'][1] - 1) // e['ceil_div'][1]
        raise ValueError(e)

    def launch(self, l, p, b):
        def val(a):
            if 'param' in a:
                v = p[a['param']]
                return (v.data_ptr() if hasattr(v, 'data_ptr') else v) + a.get('offset', 0)
            for k in ('i32', 'i64', 'f32'):
                if k in a:
                    return a[k]
            if 'var' in a:
                return b
            if 'expr' in a:
                return self.ev(a['expr'], b)
            raise ValueError(a)
        argv = []
        for ty, a in zip(l['params'], l['args']):
            if 'pack' in a:
                data = bytearray(a['pack']['size'])
                for field in a['pack']['fields']:
                    fmt = 'f' if 'f32' in field else ('i' if 'i32' in field or field.get('width') == 4 else 'q')
                    struct.pack_into('<' + fmt, data, field['at'], val(field))
                argv.append(C.create_string_buffer(bytes(data)))
            else:
                argv.append((C.c_int32 if ty == 'i32' else C.c_float if ty == 'f32' else C.c_uint64)(val(a)))
        args = (C.c_void_p * len(argv))(*[C.addressof(v) for v in argv])
        self.check(self.cuda.cuLaunchKernel(self.fn(l), *[self.ev(v, b) for v in l['grid']],
                                            *l['block'], l.get('shared_mem', 0),
                                            C.c_void_p(self.torch.cuda.current_stream().cuda_stream),
                                            args, None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='/tmp/mtp_fused45.json')
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--replays', type=int, default=20)
    ap.add_argument('--timing-rows', default='3,12,24,32')
    args = ap.parse_args()
    abi = abi_check(ops_slab2.ops()['mtp32_slab2_index_glue'])
    ref = ref_launches(args.manifest)
    guard()
    import torch
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    d = Driver(torch)

    MAXR = 32

    def rand_bf(shape, scale=1.):
        return (torch.randn(shape, device='cuda') * scale).to(torch.bfloat16)

    def inputs():
        return {
            'ik': rand_bf((MAXR, 128)),
            'kn_w': (torch.randn(128, device='cuda') * 0.03 + 1).float(),
            'kn_b': (torch.randn(128, device='cuda') * 0.03).float(),
            'iq': rand_bf((MAXR, 4096)),
            'x_norm': rand_bf((MAXR, 4096)),
            'wp_w': (torch.randn((32, 4096), device='cuda') * 0.02).float(),
        }

    def outputs():
        return {
            'ik': torch.zeros((MAXR, 128), device='cuda', dtype=torch.bfloat16),
            'iqh': torch.zeros((MAXR, 4096), device='cuda', dtype=torch.bfloat16),
            'iq8': torch.zeros((MAXR, 4096), device='cuda', dtype=torch.uint8),
            'qs': torch.zeros((MAXR, 32), device='cuda', dtype=torch.float32),
            'w': torch.zeros((MAXR, 32), device='cuda', dtype=torch.float32),
        }

    def bind(o, inp):
        """Outputs get their own ik (knorm is in-place); other inputs are
        read-only and shared between the REF and FUSED runs."""
        o['ik'].copy_(inp['ik'])
        o['kn_w'], o['kn_b'], o['wp_w'] = inp['kn_w'], inp['kn_b'], inp['wp_w']
        o['iq'], o['x_norm'] = inp['iq'], inp['x_norm']

    def run_ref(o, b):
        d.launch(ref['k_norm'], [o['ik'], o['kn_w'], o['kn_b'], o['ik']], b)
        d.launch(ref['hadamard'], [o['iq'], o['iqh']], b)
        d.launch(ref['act_quant'], [o['iqh'], o['iq8'], o['qs'], b * 32], b)
        d.launch(ref['weights_proj'], [o['x_norm'], o['wp_w'], o['qs'], o['w']], b)

    fused_l = ops_slab2.ops()['mtp32_slab2_index_glue']['impl']['launches'][0]

    def run_fused(o, b):
        d.launch(fused_l, [o['ik'], o['kn_w'], o['kn_b'], o['iq'], o['iq8'], o['qs'],
                           o['x_norm'], o['wp_w'], o['w']], b)

    def metric(a, b):
        neq = int((a.contiguous().view(torch.uint8) != b.contiguous().view(torch.uint8)).sum())
        return {'byte_neq': neq, 'n_bytes': a.numel() * a.element_size(),
                'finite': bool(torch.isfinite(a.float()).all())}

    def compare(r, f, b, tag):
        out = {}
        for k in ('ik', 'iq8', 'qs', 'w'):
            out[k] = metric(f[k][:b], r[k][:b])
            assert out[k]['byte_neq'] == 0, (tag, k, out[k])
            assert out[k]['finite'], (tag, k)
        return out

    results = {'abi': abi, 'bitwise': {}, 'timing': {}}

    # --- rows 1..32 bitwise, fresh inputs each row count ---
    for b in range(1, MAXR + 1):
        inp = inputs()
        r, f = outputs(), outputs()
        bind(r, inp)
        bind(f, inp)
        run_ref(r, b)
        run_fused(f, b)
        torch.cuda.synchronize()
        results['bitwise'][f'rows{b}'] = compare(r, f, b, f'rows{b}')

    # --- edge: zero-amax row (scale clamp path) + zero ik row + hot row ---
    inp = inputs()
    b = MAXR
    inp['iq'][0].zero_()
    inp['iq'][b - 1].zero_()
    inp['ik'][1].zero_()
    inp['iq'][2] *= 1000.
    r, f = outputs(), outputs()
    bind(r, inp)
    bind(f, inp)
    run_ref(r, b)
    run_fused(f, b)
    torch.cuda.synchronize()
    results['bitwise']['edge'] = compare(r, f, b, 'edge')

    # --- graph capture + changed-input replay at b=24 (pad path) ---
    b = 24
    inp = inputs()
    r, f = outputs(), outputs()
    bind(r, inp)
    bind(f, inp)
    run_fused(f, b)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run_fused(f, b)  # single iteration: knorm is in-place, not idempotent
    f['ik'].copy_(inp['ik'])  # restore pre-knorm input before the timed replay
    g.replay()
    torch.cuda.synchronize()
    run_ref(r, b)
    torch.cuda.synchronize()
    results['bitwise']['graph'] = compare(r, f, b, 'graph')
    # Mutate inputs in place; the captured graph must track new contents.
    inp2 = inputs()
    r['ik'].copy_(inp2['ik'])
    f['ik'].copy_(inp2['ik'])
    inp['iq'].copy_(inp2['iq'])
    inp['x_norm'].copy_(inp2['x_norm'])
    inp['kn_w'].copy_(inp2['kn_w'])
    g.replay()
    torch.cuda.synchronize()
    run_ref(r, b)
    torch.cuda.synchronize()
    results['bitwise']['graph_mutated'] = compare(r, f, b, 'graph_mutated')

    # --- timing: 4-launch chain vs 1-launch fused ---
    def timing(fn, b):
        fn()
        torch.cuda.synchronize()
        gg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gg):
            for _ in range(args.replays):
                fn()
        gg.replay()
        torch.cuda.synchronize()
        times = []
        for _ in range(7):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            gg.replay()
            e.record()
            e.synchronize()
            times.append(s.elapsed_time(e) * 1000 / args.replays)
        return statistics.median(times)

    b = MAXR
    inp = inputs()
    r, f = outputs(), outputs()
    bind(r, inp)
    bind(f, inp)
    for rows in (int(x) for x in args.timing_rows.split(',')):
        t_ref = timing(lambda: run_ref(r, rows), rows)
        t_fused = timing(lambda: run_fused(f, rows), rows)
        results['timing'][f'rows{rows}'] = {'ref_us': t_ref, 'fused_us': t_fused,
                                            'saved_us': t_ref - t_fused}
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
