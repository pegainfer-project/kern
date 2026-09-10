#!/usr/bin/env python3
"""Checkpoint WO_A once-dequant + Runtime grouped BF16 versus original einsum."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'cubin', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--check', action='store_true')
    a = p.parse_args()
    import torch
    torch.set_num_threads(4)
    if a.check:
        report = []
        for prefix in ('layers.0.attn.wo_a', 'mtp.0.attn.wo_a'):
            for rows in (1, 5, 17, 128):
                d = a.out / prefix / str(rows)
                read = lambda name: torch.frombuffer(bytearray((d / name).read_bytes()), dtype=torch.bfloat16).float()
                ref, got = read('expected.bin'), read('actual.bin')
                err = ((got-ref).square().sum()/ref.square().sum()).sqrt().item()
                assert torch.isfinite(got).all() and err < .004, (prefix, rows, err)
                assert (d/'expected_weight.bin').read_bytes() == (d/'actual_weight.bin').read_bytes()
                report.append({'prefix': prefix, 'rows': rows, 'relative_rms': err, 'dequant_exact': True})
        (a.out/'precision.json').write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2)); return
    from safetensors import safe_open
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from kern_manifest import SCHEMA_VERSION
    from dsv41.loading import Pieces
    from dsv41.attention.woa import load, forward
    index = json.loads((a.checkpoint/'model.safetensors.index.json').read_text())['weight_map']
    def get(name):
        with safe_open(a.checkpoint/index[name], framework='pt', device='cpu') as f:
            return f.get_tensor(name).cuda()
    torch.manual_seed(432221)
    with torch.inference_mode():
        for prefix in ('layers.0.attn.wo_a', 'mtp.0.attn.wo_a'):
            w = get(prefix+'.weight').view(torch.float8_e4m3fn)
            s = get(prefix+'.scale').view(torch.float8_e8m0fnu)
            # Identical to the supplied loader's WO_A weight restoration.
            weight = (w.float()*s.float().repeat_interleave(32,0).repeat_interleave(32,1)).bfloat16()
            for rows in (1, 5, 17, 128):
                d = a.out/prefix/str(rows); d.mkdir(parents=True, exist_ok=True)
                x = torch.randn(1, rows, 8, 4096, device='cuda', dtype=torch.bfloat16)
                expected = torch.einsum('bsgd,grd->bsgr', x, weight.reshape(8,1024,4096))
                pieces = Pieces(); buffers, once, layout = load(pieces, prefix, cubin=a.cubin)
                stage = forward(pieces, 'woa', layout, 'x', 'y', rows=rows, capacity=rows)
                buffers.update(stage.buffers)
                buffers.update({prefix+'.weight': {'dtype':'fp8e4m3','kind':'input','shape':[8192,4096]},
                                prefix+'.scale': {'dtype':'fp8e8m0','kind':'input','shape':[256,128]},
                                'x': {'dtype':'bf16','kind':'input','shape':[rows,32768]}})
                buffers['y']['kind']='output'; buffers[layout['weight']]['kind']='output'
                m={'schema_version':SCHEMA_VERSION,'model':'original-bf16-woa','buffers':buffers,
                   'modules':pieces.modules,'ops':pieces.ops,'programs':{'probe':{'calls':once+stage.calls}}}
                (d/'manifest.json').write_text(json.dumps(m,indent=2))
                for name,t in [('weight',w),('scale',s),('x',x),('expected',expected),('expected_weight',weight)]:
                    (d/(name+'.bin')).write_bytes(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
                command=['--manifest',str(d/'manifest.json'),'--cubins',str(a.cubin.parent),'--program','probe',
                         '--in',f'{prefix}.weight={d}/weight.bin','--in',f'{prefix}.scale={d}/scale.bin',
                         '--in',f'x={d}/x.bin','--out',f'y={d}/actual.bin',
                         '--out',f'{layout["weight"]}={d}/actual_weight.bin','--graph','--iters','2']
                (d/'runner-args.json').write_text(json.dumps(command))


if __name__ == '__main__':
    main()
