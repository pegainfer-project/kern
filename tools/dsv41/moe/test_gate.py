"""Numerical oracle is the supplied model.Gate.forward, including bias handling."""
import ctypes,pathlib,argparse,struct,sys
import torch
from cuda.bindings import driver as cu
from gate import pieces
from test_boundary import check

def main():
 p=argparse.ArgumentParser();p.add_argument('--inference',type=pathlib.Path,required=True);p.add_argument('--dump',type=pathlib.Path);p.add_argument('--raw-slab',action='store_true');a=p.parse_args();sys.path.insert(0,str(a.inference));import model
 torch.manual_seed(432);torch.empty(1,device='cuda');drv=ctypes.CDLL('libcuda.so.1')
 for experts in (128,384):
  topk=3 if experts==128 else 6
  for n in (1,5,65,128):
   modules,ops=pieces(rows=n,experts=experts,max_tokens=n,raw_outputs=a.raw_slab);op=next(iter(ops.values()));launch=op['impl']['launches'][-1]
   x=torch.randn(n,5120,device='cuda',dtype=torch.bfloat16);weight=torch.randn(experts,5120,device='cuda',dtype=torch.bfloat16)*.03;bias=torch.randn(experts,device='cuda')*.1
   idx=torch.empty(n,topk,device='cuda',dtype=torch.int64);weights=torch.empty(n,topk,device='cuda');params=[x,weight,bias,idx,weights,n,torch.zeros(8192,dtype=torch.int64,device='cuda')]
   scratch={k:torch.zeros(v['shape'],dtype=torch.uint8 if v['dtype']=='u8' else torch.int64,device='cuda') for k,v in op['impl']['scratch'].items()}
   def arg(v):
    if 'param' in v:
     q=params[v['param']];return struct.pack('Q',q.data_ptr()) if isinstance(q,torch.Tensor) else struct.pack('i',q)
    if 'scratch' in v:return struct.pack('Q',scratch[v['scratch']].data_ptr())
    if 'i64' in v:return struct.pack('q',v['i64'])
    if 'i32' in v:return struct.pack('i',v['i32'])
    if 'f32' in v:return struct.pack('f',v['f32'])
    t=v['pack']['fields'][0]['tensormap'];tm=check(cu.cuTensorMapEncodeTiled(cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,params[t['param']].data_ptr(),[cu.cuuint64_t(i) for i in t['dims']],[cu.cuuint64_t(i) for i in t['strides']],[cu.cuuint32_t(i) for i in t['box']],[cu.cuuint32_t(1)]*2,cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));return ctypes.string_at(tm.getPtr(),128)
   mod=check(cu.cuModuleLoad(str(pathlib.Path(__file__).resolve().parents[3]/modules['dsv41_gate']['source']).encode()));fn=check(cu.cuModuleGetFunction(mod,launch['entry'].encode()));check(cu.cuFuncSetAttribute(fn,cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,launch['shared_mem']))
   keep=[ctypes.create_string_buffer(arg(v)) for v in launch['args']];argv=(ctypes.c_void_p*len(keep))(*[ctypes.addressof(v) for v in keep]);err=drv.cuLaunchKernel(ctypes.c_void_p(int(fn)),launch['grid'][0],1,1,256,1,1,launch['shared_mem'],ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),argv,None);assert err==0;torch.cuda.synchronize()
   if a.dump:
    from replay import dump_case
    path=a.dump/f'gate_{experts}_{n}'
    dump_case(path,modules,f'dsv41_gate_e{experts}',op,params)
    if a.raw_slab:
     from replay import gate_to_slab
     gate_to_slab(path)
   ctx=type('Gate',(),dict(weight=weight,bias=bias,bias_vl=None,gate_temp=1.,score_func='sqrtsoftplus',topk=topk,norm_topk_prob=True,route_scale=1.5))();rw,ri=model.Gate.forward(ctx,x)
   torch.testing.assert_close(idx,ri,rtol=0,atol=0);torch.testing.assert_close(weights,rw,rtol=1e-4,atol=1e-5);print(experts,n,'indices exact, weights max abs',float((weights-rw).abs().max()),flush=True)
if __name__=='__main__':main()
