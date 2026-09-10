"""Dense quant vs original act_quant; raw scale once packing through kern fixtures."""
import argparse,ctypes,pathlib,sys
import torch
from cuda.bindings import driver as cu
from test_boundary import check
from dense import prep_pieces
from replay import dump_case,dump_once

def main():
 p=argparse.ArgumentParser();p.add_argument('--inference',type=pathlib.Path,required=True);p.add_argument('--dump',type=pathlib.Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.inference));import model
 torch.manual_seed(432);torch.empty(1,device='cuda');root=pathlib.Path(__file__).resolve().parents[3]
 for m,k in [(1,5120),(5,5120),(17,32768)]:
  mods,ops=prep_pieces(m,1024,k);op=ops['dsv41_dense_quant'];x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16);y=torch.empty(m,k,device='cuda',dtype=torch.float8_e4m3fn);sfrows=(m+3)//4*4;sf=torch.empty(k//128,sfrows,device='cuda',dtype=torch.int32)
  mod=check(cu.cuModuleLoad(str(root/mods['dsv41_moe_stage']['source']).encode()));fn=check(cu.cuModuleGetFunction(mod,b'dsv41_dense_quant_x'));params=[x,y,sf,m,k,k,sfrows];vals=tuple(v.data_ptr() if isinstance(v,torch.Tensor) else v for v in params);types=(ctypes.c_void_p,)*3+(ctypes.c_int,)*4;check(cu.cuLaunchKernel(fn,(sfrows*(k//128)+7)//8,1,1,256,1,1,0,cu.CUstream(torch.cuda.current_stream().cuda_stream),(vals,types),0));torch.cuda.synchronize();ry,rs=model.act_quant(x,32,'ue8m0',torch.float8_e8m0fnu);torch.testing.assert_close(y.view(torch.uint8),ry.view(torch.uint8),rtol=0,atol=0);ref=torch.zeros_like(sf);ref[:,:m]=rs.view(torch.uint8).contiguous().view(torch.int32).t();torch.testing.assert_close(sf,ref,rtol=0,atol=0);dump_case(a.dump/f'quant_{m}_{k}',mods,'dsv41_dense_quant',op,params);print('quant exact',m,k,flush=True)
 for n,k,groups in [(1280,5120,1),(1024,4096,8)]:
  mods,ops=prep_pieces(1,n,k,groups=groups);op=ops['dsv41_dense_sf_pack'];source=torch.randint(1,254,(groups*n//32,k//32),device='cuda',dtype=torch.uint8);out=torch.empty(groups,k//128,n,device='cuda',dtype=torch.int32);mod=check(cu.cuModuleLoad(str(root/mods['dsv41_weight_prep']['source']).encode()));fn=check(cu.cuModuleGetFunction(mod,b'dsv41_pack_dense_sf'));params=[out,source,n,k,groups,32];vals=tuple(v.data_ptr() if isinstance(v,torch.Tensor) else v for v in params);types=(ctypes.c_void_p,)*2+(ctypes.c_int,)*4;check(cu.cuLaunchKernel(fn,(out.numel()+255)//256,1,1,256,1,1,0,cu.CUstream(torch.cuda.current_stream().cuda_stream),(vals,types),0));torch.cuda.synchronize();ref=source.repeat_interleave(32,0).view(torch.int32).view(groups,n,k//128).transpose(1,2).contiguous();torch.testing.assert_close(out,ref,rtol=0,atol=0);dump_once(a.dump/f'pack_{n}_{k}_{groups}',mods,'dsv41_dense_sf_pack',op,params);print('pack exact',n,k,groups,flush=True)
if __name__=='__main__':main()
