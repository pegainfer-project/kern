"""Single-process EP4 AOT launch with supplied Expert.forward numerical oracle.

Requires exclusive access to all four GPUs. Operates on synthetic quantized
weights, distinguishes each rank, and covers unequal/empty local batches.
"""
import argparse,ctypes,pathlib,struct,sys
import torch
from cuda.bindings import driver as cu
from test_boundary import check
from moe import pieces

class Attr(ctypes.Structure):_fields_=[('id',ctypes.c_int),('pad',ctypes.c_int),('value',ctypes.c_uint*16)]
class Config(ctypes.Structure):_fields_=[('gx',ctypes.c_uint),('gy',ctypes.c_uint),('gz',ctypes.c_uint),('bx',ctypes.c_uint),('by',ctypes.c_uint),('bz',ctypes.c_uint),('smem',ctypes.c_uint),('stream',ctypes.c_void_p),('attrs',ctypes.POINTER(Attr)),('count',ctypes.c_uint)]

def interleave(x):return torch.stack([x[:2304].reshape(-1,8,x.shape[-1]),x[2304:].reshape(-1,8,x.shape[-1])],1).reshape_as(x).contiguous()
def pack_scale(x):
 n,k=x.shape;return x.view(torch.int32).reshape(n//128,4,32,k//4).transpose(1,2).reshape(n,k//4).t().contiguous()
def call_simple(module,entry,out,*args):
 fn=check(cu.cuModuleGetFunction(module,entry.encode()));items=(out,*args);vals=tuple(x.data_ptr() if isinstance(x,torch.Tensor) else x for x in items);types=tuple(ctypes.c_void_p if isinstance(x,torch.Tensor) else ctypes.c_int for x in items);count=out.numel();check(cu.cuLaunchKernel(fn,(count+255)//256,1,1,256,1,1,0,cu.CUstream(torch.cuda.current_stream().cuda_stream),(vals,types),0))

def main():
 p=argparse.ArgumentParser();p.add_argument('--inference',type=pathlib.Path,required=True);p.add_argument('--experts',type=int,default=128);p.add_argument('--dump',type=pathlib.Path);a=p.parse_args();sys.path.insert(0,str(a.inference));import model
 assert torch.cuda.device_count()==4
 torch.manual_seed(432);drv=ctypes.CDLL('libcuda.so.1');modules,ops,lay=pieces(a.experts);op=next(iter(ops.values()));launch=op['impl']['launches'][0];local=a.experts//4;topk=lay['topk'];states=[]
 root=pathlib.Path(__file__).resolve().parents[3]/'target/cubins/dsv41'
 for rank in range(4):
  with torch.cuda.device(rank):
   torch.empty(1,device='cuda')
   torch.cuda.manual_seed(432+rank)
   for other in range(4):
    if rank!=other:
     err=cu.cuCtxEnablePeerAccess(check(cu.cuDevicePrimaryCtxRetain(other)),0)[0];assert err in (cu.CUresult.CUDA_SUCCESS,cu.CUresult.CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED),err
   slab=torch.zeros(lay['slab_bytes'],device='cuda',dtype=torch.uint8)
   # Every expert in a rank has identical weights, different ranks differ.
   gw=torch.randint(0,256,(2304,2560),device='cuda',dtype=torch.uint8);uw=torch.randint_like(gw,0,256);dw=torch.randint(0,256,(5120,1152),device='cuda',dtype=torch.uint8)
   sf1=torch.full((2304,160),120,device='cuda',dtype=torch.uint8);sf2=torch.full((5120,72),120,device='cuda',dtype=torch.uint8)
   w1=interleave(torch.cat([gw,uw])).repeat(local,1,1);w2=dw.repeat(local,1,1);s1=pack_scale(interleave(torch.cat([sf1,sf1]))).repeat(local,1,1);s2=pack_scale(sf2).repeat(local,1,1)
   sg=torch.randn(2304,5120,device='cuda').to(torch.float8_e4m3fn);su=torch.randn(2304,5120,device='cuda').to(torch.float8_e4m3fn);sd=torch.randn(5120,2304,device='cuda').to(torch.float8_e4m3fn)
   ss1=torch.full((72,160),120,device='cuda',dtype=torch.uint8);ss2=torch.full((160,72),120,device='cuda',dtype=torch.uint8)
   sw1=interleave(torch.cat([sg.view(torch.uint8),su.view(torch.uint8)]));sw2=sd.view(torch.uint8);sw1sf=pack_scale(interleave(torch.cat([ss1,ss1]).repeat_interleave(32,0)));sw2sf=pack_scale(ss2.repeat_interleave(32,0))
   mod=check(cu.cuModuleLoad(str(pathlib.Path(__file__).resolve().parents[3]/modules['dsv41_mega_moe']['source']).encode()));fn=check(cu.cuModuleGetFunction(mod,launch['entry'].encode()));check(cu.cuFuncSetAttribute(fn,cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,launch['shared_mem']));stage=check(cu.cuModuleLoad(str(root/'stage.cubin').encode()))
   states.append(dict(slab=slab,weights=[w1,s1,w2,s2,sw1,sw1sf,sw2,sw2sf],raw=[gw,uw,dw,sf1,sf2,sg,su,sd,ss1,ss2],fn=fn,stage=stage))
   torch.cuda.synchronize();print('prepared rank',rank,flush=True)
 peers=[s['slab'].data_ptr() for s in states]
 for counts in ([1,1,1,1],[1,5,0,3]):
  for rank,s in enumerate(states):
   with torch.cuda.device(rank):
    n=counts[rank];s['x']=torch.randn(n,5120,device='cuda',dtype=torch.bfloat16);s['y']=torch.empty_like(s['x']);s['stats']=torch.zeros(local,device='cuda',dtype=torch.int32);s['peers']=torch.tensor(peers,device='cuda',dtype=torch.int64)
    idx=(torch.arange(n*topk,device='cuda').reshape(n,topk)*local+rank)%a.experts;rw=torch.rand(n,topk,device='cuda');rw=rw/rw.sum(-1,keepdim=True)*1.5;s['idx']=idx;s['rw']=rw
    off=lay['offsets'];view=lambda key:s['slab'][off[key]:]
    if n:
     view('idx')[:n*topk*8].view(torch.int64).copy_(idx.flatten());view('weights')[:n*topk*4].view(torch.float32).copy_(rw.flatten())
     fn=check(cu.cuModuleGetFunction(s['stage'],b'dsv41_mega_quant_x'));vals=(s['x'].data_ptr(),view('x').data_ptr(),view('x_sf').data_ptr(),n,5120,5120,40,view('shared_x_sf').data_ptr(),lay['shared_sf_rows']);ty=(ctypes.c_void_p,)*3+(ctypes.c_int,)*4+(ctypes.c_void_p,ctypes.c_int);check(cu.cuLaunchKernel(fn,(n*40+7)//8,1,1,256,1,1,0,cu.CUstream(torch.cuda.current_stream().cuda_stream),(vals,ty),0))
    torch.cuda.synchronize()
    w1,sf1,w2,sf2,sw1,ss1,sw2,ss2=s['weights'];params=[s['y'],s['stats'],n,s['peers'],rank,view('l1'),view('l1_sf'),w1,sf1,view('l2'),view('l2_sf'),w2,sf2,view('x'),view('shared_x_sf'),sw1,ss1,view('shared_l2'),view('shared_l2_sf'),sw2,ss2]
    def arg(v):
     if 'param' in v:
      q=params[v['param']];return struct.pack('Q',q.data_ptr()) if isinstance(q,torch.Tensor) else struct.pack('i',q)
     t=v['pack']['fields'][0]['tensormap'];dt={'u8':cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,'i32':cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_INT32,'u4':cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B};sw={0:cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,64:cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B,128:cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B};tm=check(cu.cuTensorMapEncodeTiled(dt[t['dtype']],2,params[t['param']].data_ptr(),[cu.cuuint64_t(v) for v in t['dims']],[cu.cuuint64_t(v) for v in t['strides']],[cu.cuuint32_t(v) for v in t['box']],[cu.cuuint32_t(1)]*2,cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,sw[t['swizzle']],cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));return ctypes.string_at(tm.getPtr(),128)
    s['keep']=[ctypes.create_string_buffer(arg(v)) for v in launch['args']];s['argv']=(ctypes.c_void_p*len(s['keep']))(*[ctypes.addressof(v) for v in s['keep']])
  if a.dump:
   from dump_moe import dump_before
   dump_before(a.dump/('_'.join(map(str,counts))),states,counts,lay,modules,op)
  for rank,s in enumerate(states):
   with torch.cuda.device(rank):
    attr=Attr();attr.id=4;attr.value[0]=2;attr.value[1]=1;attr.value[2]=1;cfg=Config(152,1,1,512,1,1,launch['shared_mem'],torch.cuda.current_stream().cuda_stream,ctypes.pointer(attr),1);err=drv.cuLaunchKernelEx(ctypes.byref(cfg),ctypes.c_void_p(int(s['fn'])),s['argv'],None);assert err==0,err
  for rank,s in enumerate(states):
   with torch.cuda.device(rank):torch.cuda.synchronize();print('finished',counts,rank,flush=True)
  if a.dump:
   from dump_moe import dump_after
   dump_after(a.dump/('_'.join(map(str,counts))),states)
  # Reference uses actual supplied Expert.forward and model.linear kernels.
  torch.set_default_dtype(torch.bfloat16)
  for rank,s in enumerate(states):
   with torch.cuda.device(rank):
    n=counts[rank]
    if not n:continue
    def expert(raw,shared=False):
     if shared:g,u,d,sg,sd=raw[5:];dtype=torch.float8_e4m3fn
     else:g,u,d,sg,sd=raw[:5];dtype=torch.float4_e2m1fn_x2
     def linear(w,sf):
      w=w.to(rank).view(dtype);w.scale=sf.to(rank).view(torch.float8_e8m0fnu);return lambda x:model.linear(x,w)
     return type('Expert',(),{'w1':staticmethod(linear(g,sg)),'w3':staticmethod(linear(u,sg)),'w2':staticmethod(linear(d,sd)),'swiglu_limit':10.})()
    ref=torch.zeros_like(s['x'],dtype=torch.float32)
    for owner in range(4):
     ex=expert(states[owner]['raw'])
     for j in range(topk):
      mask=s['idx'][:,j]//local==owner
      if mask.any():ref[mask]+=model.Expert.forward(ex,s['x'][mask],s['rw'][mask,j,None]).float()
    ref+=model.Expert.forward(expert(s['raw'],True),s['x']).float();ref=ref.bfloat16();diff=(s['y'].float()-ref.float()).square().sum()/(ref.float().square().sum()+1e-20);print('oracle',counts,rank,float(diff),flush=True);assert diff<0.003,(counts,rank,diff)
  torch.set_default_dtype(torch.float32)
if __name__=='__main__':main()
