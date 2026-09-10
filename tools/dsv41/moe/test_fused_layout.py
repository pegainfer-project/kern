"""Once permutations plus GEMM, checked against original model.linear/einsum."""
import argparse,ctypes,json,pathlib,sys
import torch
from safetensors import safe_open
from cuda.bindings import driver as cu
from test_boundary import check
from dense import layout_pieces,pieces,oa_pieces
from replay import dump_once,dump_case
from test_oa import dequantize,run


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=pathlib.Path,required=True);p.add_argument('--dump',type=pathlib.Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(a.checkpoint/'inference'));import model
    torch.manual_seed(772);torch.empty(1,device='cuda');torch.set_default_dtype(torch.bfloat16);drv=ctypes.CDLL('libcuda.so.1')
    index=json.loads((a.checkpoint/'model.safetensors.index.json').read_text())['weight_map']
    def get(name):
        with safe_open(a.checkpoint/index[name],framework='pt') as f:return f.get_tensor(name).cuda()
    root=pathlib.Path(__file__).resolve().parents[3]
    for kind,which,prefix,n,k,groups in [('query',0,'layers.0.attn.wq_b',32768,1280,1),('output',1,'layers.0.attn.wo_a',1024,4096,8)]:
        weight=get(prefix+'.weight');rawsf=get(prefix+'.scale');outw=torch.empty_like(weight);outsf=torch.empty(groups,k//128,n,device='cuda',dtype=torch.int32)
        mods,ops=layout_pieces(kind);mod=check(cu.cuModuleLoad(str(root/mods['dsv41_fused_prep']['source']).encode()))
        for suffix,output,source in [('weight',outw,weight),('scale',outsf,rawsf)]:
            op=ops[f'dsv41_{kind}_{suffix}_layout'];fn=check(cu.cuModuleGetFunction(mod,op['impl']['launches'][0]['entry'].encode()));count=output.numel()
            check(cu.cuLaunchKernel(fn,(count+255)//256,1,1,256,1,1,0,cu.CUstream(torch.cuda.current_stream().cuda_stream),((output.data_ptr(),source.data_ptr(),which),(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int)),0))
            torch.cuda.synchronize();dump_once(a.dump/f'{kind}_{suffix}',mods,f'dsv41_{kind}_{suffix}_layout',op,[output,source])
        if which:
            rw=weight.view(torch.uint8).view(8,n,8,16,32).transpose(2,3).contiguous().view_as(outw.view(torch.uint8))
            rs=rawsf.view(torch.uint8).view(8,n//32,8,16).transpose(2,3).contiguous().repeat_interleave(32,1).view(8,n,k//32).view(torch.int32).transpose(1,2).contiguous()
        else:
            rw=weight.view(torch.uint8).view(64,32,16,k).transpose(0,1).contiguous().view_as(outw.view(torch.uint8))
            rs=rawsf.view(torch.uint8).repeat_interleave(32,0).view(64,32,16,k//32).transpose(0,1).contiguous().view(n,k//32).view(torch.int32).t().contiguous().unsqueeze(0)
        torch.testing.assert_close(outw.view(torch.uint8),rw,rtol=0,atol=0);torch.testing.assert_close(outsf,rs,rtol=0,atol=0)
        for m in (1,5):
            x=torch.randn(m,groups*k,device='cuda',dtype=torch.bfloat16)
            transformed=x.view(m,8,8,16,32).transpose(2,3).contiguous().view_as(x) if which else x
            aq,ascale=model.act_quant(transformed,32,'ue8m0',torch.float8_e8m0fnu);sfrows=(m+3)//4*4;sfa=torch.zeros(groups*k//128,sfrows,device='cuda',dtype=torch.int32);sfa[:,:m]=ascale.view(torch.uint8).contiguous().view(torch.int32).t()
            if which:
                # O-A returns MXFP8; dequantize before comparing with the BF16 einsum.
                y=torch.empty(m,groups*n,device='cuda',dtype=torch.float8_e4m3fn);sfd=torch.empty(groups*n//128,sfrows,device='cuda',dtype=torch.int32)
                dm,do=oa_pieces(m);params=[y,aq,outw,sfa,outsf,m,n,k,sfd]
            else:
                y=torch.empty(m,n,device='cuda',dtype=torch.bfloat16);dm,do=pieces(m,n,k);params=[y,aq,outw,sfa,outsf,m,n,k]
            op=next(iter(do.values()));run(drv,root,dm,op,params)
            if which:
                wbf=(weight.float()*rawsf.float().repeat_interleave(32,0).repeat_interleave(32,1)).bfloat16().view(8,n,k)
                ref=torch.einsum('tgd,gnd->tgn',x.view(m,8,k),wbf).flatten(1)
                y=dequantize(y,sfd[:,:m].t().contiguous().view(torch.uint8))
            else:
                weight.scale=rawsf;ref=model.linear(x,weight).view(m,64,32,16).transpose(1,2).contiguous().view_as(y)
            diff=float((y.float()-ref.float()).square().sum()/ref.float().square().sum());assert diff<(0.003 if which else 1e-5),diff
            dump_case(a.dump/f'{kind}_gemm_{m}',dm,next(iter(do)),op,params);print(kind,m,'relative squared error',diff,flush=True)
    torch.set_default_dtype(torch.float32)

if __name__=='__main__':main()
