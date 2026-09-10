#!/usr/bin/env python3
"""Compare the pinned upstream attention against the model's inference/kernel.py.

The bridge exists only to obtain upstream launch ABI and oracle outputs. It is
not a serving dependency. Pass --inference to the unmodified model directory.
"""
import argparse
import ctypes
import json
import math
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--library', required=True, type=Path)
    parser.add_argument('--inference', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--dump', type=Path)
    args = parser.parse_args()
    import torch
    sys.path.insert(0, str(args.inference))
    from kernel import sparse_attn
    bridge = ctypes.CDLL(str(args.library)).dsv41_sparse_prefill
    bridge.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int] * 3 + [ctypes.c_float, ctypes.c_void_p]
    bridge.restype = None
    torch.manual_seed(432221)
    results = []
    for rows, kv_rows, topk, mode in [(1, 128, 128, 'decode'), (17, 257, 256, 'prefill'),
                                     (10, 257, 256, 'dspark_noncausal'), (12, 257, 256, 'verify')]:
        q = torch.randn(rows, 64, 512, device='cuda', dtype=torch.bfloat16)
        kv = torch.randn(kv_rows, 1, 512, device='cuda', dtype=torch.bfloat16)
        sink = torch.randn(64, device='cuda', dtype=torch.float32)
        indices = torch.full((rows, 1, topk), -1, device='cuda', dtype=torch.int32)
        for row in range(rows):
            length = min(topk, kv_rows if mode == 'dspark_noncausal' else 1 + row * 13)
            indices[row, 0, :length] = torch.randperm(kv_rows, device='cuda')[:length].int()
        # Explicit all-invalid row: sink-only attention must return zero.
        if rows > 1:
            indices[-1].fill_(-1)
        lengths = torch.full((rows,), topk, device='cuda', dtype=torch.int32)
        out = torch.empty_like(q)
        maxima = torch.empty((rows,64), device='cuda', dtype=torch.float32)
        lse = torch.empty_like(maxima)
        pointers = [t.data_ptr() for t in [q,kv,indices,sink,lengths,out,maxima,lse]]
        bridge(*pointers, rows, kv_rows, topk, 1/math.sqrt(512), torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        expected = sparse_attn(q.unsqueeze(0),kv[:,0].unsqueeze(0),sink,indices[:,0].unsqueeze(0),1/math.sqrt(512))[0]
        torch.cuda.synchronize()
        delta = (out.float()-expected.float()).abs()
        rel = (delta.square().mean()/expected.float().square().mean().clamp_min(1e-20)).sqrt().item()
        entry = {'rows':rows,'kv_rows':kv_rows,'topk':topk,'mode':mode,'max_abs':delta.max().item(),'relative_rms':rel}
        print(json.dumps(entry), flush=True)
        results.append(entry)
        assert torch.isfinite(out).all(), entry
        assert rel < 0.01 and delta.max().item() < 0.0625, entry
        if rows > 1:
            assert out[-1].count_nonzero().item() == 0, entry
        if args.dump:
            from ops import definitions
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
            from kern_manifest import SCHEMA_VERSION
            case = args.dump / mode
            case.mkdir(parents=True,exist_ok=True)
            tensors = dict(zip(['q','kv','indices','sink','lengths','out'],[q,kv,indices,sink,lengths,out]))
            for name,tensor in tensors.items():
                (case/f'{name}.bin').write_bytes(tensor.cpu().contiguous().view(torch.uint8).numpy().tobytes())
            cubin = args.library.with_name('libdsv41_sparse_prefill.2.sm_103a.cubin')
            modules,ops = definitions(cubin,rows='rows',rows_max=rows,kv_rows_max=kv_rows,topk=topk)
            buffers = {name:{'dtype':{'torch.bfloat16':'bf16','torch.int32':'i32','torch.float32':'f32'}[str(t.dtype)],'shape':list(t.shape),'kind':'output' if name=='out' else 'input'} for name,t in tensors.items()}
            call = {'op':'dsv41_sparse_attention','args':[{'buf':name} for name in tensors]+[{'var':'rows'},{'i32':kv_rows}]}
            manifest={'schema_version':SCHEMA_VERSION,'model':'attention-probe','vars':{'rows':{'max':rows}},'states':{},'buffers':buffers,'modules':modules,'ops':ops,'programs':{'probe':{'calls':[call]}}}
            (case/'manifest.json').write_text(json.dumps(manifest,indent=2))
        if mode == 'decode':
            for name, tensor in zip(['q','kv','indices','sink','lengths','out','maxima','lse'],[q,kv,indices,sink,lengths,out,maxima,lse]):
                print(f'{name} {tensor.data_ptr():#x}', flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'oracle':'model inference/kernel.py:sparse_attn','results':results},indent=2)+'\n')

if __name__ == '__main__':
    main()
