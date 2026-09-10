"""Isolate supplied FP8 linear/quantization on fixed captured reference inputs.

Runs only when a GPU window is assigned. The reference source/cache is untouched.
No CPU readback occurs between launches unless --sync is explicitly selected.
"""
import argparse,json,pathlib,sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=pathlib.Path,required=True);p.add_argument('--input',type=pathlib.Path,required=True)
    p.add_argument('--weight',default='layers.3.attn.wq_a');p.add_argument('--output',type=pathlib.Path,required=True)
    p.add_argument('--iterations',type=int,default=20);p.add_argument('--sync',action='store_true');p.add_argument('--perturb-bytes',type=int,default=0)
    a=p.parse_args()
    import torch
    from safetensors import safe_open
    torch.set_num_threads(4);torch.set_default_dtype(torch.bfloat16);torch.set_default_device('cuda')
    sys.path.insert(0,str(a.checkpoint/'inference'));import model
    index=json.loads((a.checkpoint/'model.safetensors.index.json').read_text())['weight_map']
    def get(name):
        with torch.device('cpu'),safe_open(a.checkpoint/index[name],framework='pt',device='cpu') as f:return f.get_tensor(name).cuda()
    weight=get(a.weight+'.weight');weight.scale=get(a.weight+'.scale')
    assert weight.dtype==torch.float8_e4m3fn
    with torch.device('cpu'):x=torch.frombuffer(bytearray(a.input.read_bytes()),dtype=torch.bfloat16).reshape(-1,weight.shape[1]).cuda()
    with torch.inference_mode():
        qa,qs=model.act_quant(x,32,model.scale_fmt,model.scale_dtype)
        # Warm compilation without changing the captured data or checkpoint.
        model.linear(x,weight);model.fp8_gemm(qa,qs,weight,weight.scale,model.scale_dtype,32);torch.cuda.synchronize()
        dynamic=[];fixed=[];trash=[]
        for i in range(a.iterations):
            if a.perturb_bytes:trash.append(torch.empty(a.perturb_bytes,dtype=torch.uint8).fill_(i%256))
            dynamic.append(model.linear(x,weight))
            fixed.append(model.fp8_gemm(qa,qs,weight,weight.scale,model.scale_dtype,32))
            if a.sync:torch.cuda.synchronize()
        torch.cuda.synchronize()
    a.output.mkdir(parents=True,exist_ok=True);report={}
    for name,items in [('linear',dynamic),('fixed_quant_gemm',fixed)]:
        baseline=items[0].cpu();records=[]
        (a.output/(name+'.first.bf16')).write_bytes(baseline.contiguous().view(torch.uint8).numpy().tobytes())
        for i,item in enumerate(items):
            value=item.cpu();different=(value!=baseline).reshape(value.shape[0],-1).any(1)
            row=dict(iteration=i,changed_rows=torch.where(different)[0].tolist(),max_abs=float((value.float()-baseline.float()).abs().max()))
            records.append(row)
            if row['changed_rows']:(a.output/f'{name}.{i}.bf16').write_bytes(value.contiguous().view(torch.uint8).numpy().tobytes())
        report[name]=records
    report.update(shape=list(x.shape),weight=a.weight,iterations=a.iterations,sync=a.sync,perturb_bytes=a.perturb_bytes)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
