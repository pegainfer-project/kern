"""Byte-exact once packing vs pinned upstream pure Torch transforms."""
import argparse,ast,pathlib,ctypes
import torch
from cuda.bindings import driver as cu
from test_boundary import check

def main():
 p=argparse.ArgumentParser();p.add_argument('--deepgemm',type=pathlib.Path,required=True);p.add_argument('--dump',type=pathlib.Path);a=p.parse_args();torch.manual_seed(432);torch.empty(1,device='cuda')
 tree=ast.parse((a.deepgemm/'deep_gemm/mega/__init__.py').read_text());nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in ('_interleave_weights','_transpose_sf_for_utccp')];env={'torch':torch};exec(compile(ast.Module(body=nodes,type_ignores=[]),'upstream','exec'),env)
 mod=check(cu.cuModuleLoad(str(pathlib.Path(__file__).resolve().parents[3]/'target/cubins/dsv41/weight_prep.cubin').encode()))
 def run(entry,out,*args):
  vals=(out,*args);types=tuple(ctypes.c_void_p if isinstance(x,torch.Tensor) else ctypes.c_int for x in vals);values=tuple(x.data_ptr() if isinstance(x,torch.Tensor) else x for x in vals);fn=check(cu.cuModuleGetFunction(mod,entry.encode()));check(cu.cuLaunchKernel(fn,(out.numel()+255)//256,1,1,256,1,1,0,cu.CUstream(torch.cuda.current_stream().cuda_stream),(values,types),0));torch.cuda.synchronize()
 for shared in (False,True):
  n,k=2304,5120;row_bytes=k if shared else k//2;g=torch.randint(0,256,(n,row_bytes),device='cuda',dtype=torch.uint8);u=torch.randint_like(g,0,256);out=torch.empty(2*n,row_bytes,device='cuda',dtype=torch.uint8);run('dsv41_interleave_gate_up',out,g,u,n,row_bytes);torch.testing.assert_close(out,env['_interleave_weights'](torch.cat([g,u])),rtol=0,atol=0)
  for gate in (False,True):
   nr=n*2 if gate else 5120;kr=k if gate else 2304;group=32 if shared else 1;rows=nr//2 if gate else nr
   sf=torch.randint(1,254,(rows//group,kr//32),device='cuda',dtype=torch.uint8);su=torch.randint_like(sf,1,254)
   source=torch.cat([sf.repeat_interleave(group,0),su.repeat_interleave(group,0)]) if gate else sf.repeat_interleave(group,0)
   words=source.contiguous().view(torch.int32);words=env['_interleave_weights'](words) if gate else words;ref=env['_transpose_sf_for_utccp'](words).t().contiguous()
   packed=torch.empty(kr//128,nr,device='cuda',dtype=torch.int32);run('dsv41_pack_expert_sf',packed,sf,su,nr,kr,group,int(gate));torch.testing.assert_close(packed,ref,rtol=0,atol=0)
   if a.dump:
    from replay import dump_once
    from prep import pieces
    modules,ops=pieces(n=rows,k=kr,row_group=group,gate_up=gate);key=f'dsv41_pack_expert_sf_{"shared" if shared else "routed"}_{"gate_up" if gate else "down"}';dump_once(a.dump/f'prep_{shared}_{gate}',modules,key,ops[key],[packed,sf,su,nr,kr,group,int(gate)])
   print('shared',shared,'gate',gate,'packing exact',flush=True)
if __name__=='__main__':main()
