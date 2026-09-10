"""Numerical tests against methods loaded from the supplied inference/model.py."""
import argparse, ctypes, importlib.util, pathlib, sys
import torch
from cuda.bindings import driver as cu

def check(r):
    assert r[0] == cu.CUresult.CUDA_SUCCESS, r[0]
    return r[1] if len(r)==2 else r[1:]

def main():
    p=argparse.ArgumentParser();p.add_argument('--inference',type=pathlib.Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(a.inference)); import model
    torch.manual_seed(432); torch.empty(1,device="cuda")
    mod=check(cu.cuModuleLoad(str(pathlib.Path(__file__).resolve().parents[3] / 'target/cubins/dsv41/boundary.cubin').encode()))
    def run(name,out,*args):
        fn=check(cu.cuModuleGetFunction(mod,name.encode())); vals=tuple(v.data_ptr() if isinstance(v,torch.Tensor) else v for v in (out,*args)); types=tuple(ctypes.c_void_p if isinstance(v,torch.Tensor) else ctypes.c_int for v in (out,*args))
        check(cu.cuLaunchKernel(fn,(out.numel()+255)//256,1,1,256,1,1,0,cu.CUstream(torch.cuda.current_stream().cuda_stream),(vals,types),0));torch.cuda.synchronize();return out
    for n in (1,5,33,128):
        x=torch.randn(n,5120,device='cuda',dtype=torch.bfloat16)
        residual=torch.randn(n,4,5120,device='cuda',dtype=torch.bfloat16)
        pre=torch.randn(n,4,device='cuda');post=torch.randn_like(pre);comb=torch.randn(n,4,4,device='cuda')
        y=run('dsv41_hc_pre',torch.empty_like(x),residual,pre,n)
        ref=model.Block.hc_pre(None,residual.unsqueeze(0),pre.unsqueeze(0)).squeeze(0)
        torch.testing.assert_close(y,ref,rtol=0,atol=0)
        y=run('dsv41_hc_post',torch.empty_like(residual),x,residual,post,comb,n)
        ref=model.Block.hc_post(None,x.unsqueeze(0),residual.unsqueeze(0),post.unsqueeze(0),comb.unsqueeze(0)).squeeze(0)
        torch.testing.assert_close(y,ref,rtol=0,atol=0)
        # Use actual Expert.forward, replacing its projections with supplied branch tensors.
        gate=torch.randn(n,2304,device='cuda',dtype=torch.bfloat16)*20;up=torch.randn_like(gate)*20
        from types import SimpleNamespace
        expert=SimpleNamespace(w1=lambda _:gate,w3=lambda _:up,w2=lambda v:v,swiglu_limit=10.)
        ref=model.Expert.forward(expert,x)
        y=run('dsv41_swiglu',torch.empty_like(gate),gate,up,gate.numel())
        torch.testing.assert_close(y,ref,rtol=0.008,atol=0.0001)
        y=run('dsv41_hc_init',torch.empty_like(residual),x,n)
        torch.testing.assert_close(y,x[:,None,:].expand_as(y),rtol=0,atol=0)
        y=run('dsv41_hc_mean',torch.empty_like(x),residual,n)
        torch.testing.assert_close(y,residual.mean(dim=1),rtol=0,atol=0)
        zero=torch.empty(n,4,device='cuda');identity=torch.empty(n,4,4,device='cuda');initial=torch.empty(n,4,device='cuda')
        # run helper grids from first output size; identity needs full 16 entries/token.
        fn=check(cu.cuModuleGetFunction(mod,b'dsv41_hc_identity'));check(cu.cuLaunchKernel(fn,(n*16+255)//256,1,1,256,1,1,0,cu.CUstream(torch.cuda.current_stream().cuda_stream),((zero.data_ptr(),identity.data_ptr(),initial.data_ptr(),n),(ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int)),0));torch.cuda.synchronize()
        torch.testing.assert_close(zero,torch.zeros_like(zero),rtol=0,atol=0)
        torch.testing.assert_close(identity,torch.eye(4,device='cuda').expand(n,4,4),rtol=0,atol=0)
        torch.testing.assert_close(initial,model.make_identity_pre_mix(x[None],4).squeeze(0),rtol=0,atol=0)
        print(f'tokens={n}: hc_pre/post/init exact; Expert clamped SwiGLU passes')
if __name__=='__main__':main()
