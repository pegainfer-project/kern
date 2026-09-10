"""Compare independent original and kern prefill hooks on identical token IDs."""
import argparse,json,pathlib
import numpy as np


def load(path,dtype):
    if dtype in ('bf16','bfloat16'):
        bits=np.fromfile(path,dtype='<u2').astype(np.uint32)<<16
        return bits.view(np.float32)
    return np.fromfile(path,dtype='<f4')


def metric(a,b):
    if a.size!=b.size:raise ValueError(f'size mismatch {a.size}/{b.size}')
    a,b=a.astype(np.float64),b.astype(np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('nonfinite boundary')
    return dict(equal=bool(np.array_equal(a,b)),elements=a.size,changed=int(np.count_nonzero(a!=b)),
                max_abs=float(np.max(np.abs(a-b))),relative_squared=float(np.sum((a-b)**2)/(np.sum(b*b)+1e-30)))


def main():
    p=argparse.ArgumentParser();p.add_argument('kern',type=pathlib.Path);p.add_argument('original',type=pathlib.Path);p.add_argument('--trajectory',type=pathlib.Path);a=p.parse_args()
    trajectory=a.trajectory or a.original.parent/'ar'
    ids=json.loads((a.kern/'input_ids.json').read_text());original_ids=json.loads((trajectory/'tokens.json').read_text())[0][:len(ids)]
    if ids!=original_ids:raise ValueError('Original and kern prompt token IDs differ')
    index={x['name']:x for x in json.loads((a.kern/'index.json').read_text())};reference=json.loads((a.original/'tensors.json').read_text());report=[]
    def original(name):
        entry=reference[name];return load(a.original/entry['file'],entry['dtype'])
    def kern(name):
        entry=index[name];return load(a.kern/entry['file'],entry['dtype'])
    for name in ['embedding']+[f'layer.{l:02d}.{stage}' for l in range(40) for stage in ('attn_norm','attn','ffn_norm','ffn')]:
        row=dict(name=name,**metric(kern(name),original(name)));report.append(row);print(json.dumps(row),flush=True)
    for l in range(40):
        prefix=f'layer.{l:02d}';row=dict(name=prefix+'.pre_mix',**metric(kern(prefix+'.pre2'),original(prefix+'.pre_mix')));report.append(row)
        # Derived only: kern carries delayed HC post into the next mHC kernel.
        residual=kern(prefix+'.residual2').reshape(-1,4,5120)
        update=kern(prefix+'.ffn').reshape(-1,5120)
        post=kern(prefix+'.post2').reshape(-1,4)
        comb=kern(prefix+'.comb2').reshape(-1,4,4)
        hidden=post[:,:,None]*update[:,None,:]+np.sum(comb[:,:,:,None]*residual[:,:,None,:],axis=1)
        # Round float32 to BF16, ties to even (finite values checked by metric).
        bits=hidden.astype(np.float32).view(np.uint32);bits=(bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)
        report.append(dict(name=prefix+'.h',derived=True,**metric(bits.view(np.float32).ravel(),original(prefix+'.h'))))
    kl=load(a.kern/'logits.f32','f32');ol=load(trajectory/'target_logits.f32','f32')[:kl.size]
    result={'independent_inputs':'both start from identical prompt token IDs and embedding','logits':dict(**metric(kl,ol),kern_top1=int(np.argmax(kl)),original_top1=int(np.argmax(ol))),'boundaries':report}
    (a.kern/'original-comparison.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
