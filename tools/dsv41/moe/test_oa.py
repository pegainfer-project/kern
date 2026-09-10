"""Dense MXFP8 AOT vs original model.linear, then dump actual kern replay."""
import argparse,ctypes,pathlib,struct,sys
import torch
from cuda.bindings import driver as cu
from test_boundary import check
from dense import oa_pieces as pieces

def main():
 p=argparse.ArgumentParser();p.add_argument('--inference',type=pathlib.Path,required=True);p.add_argument('--dump',type=pathlib.Path);a=p.parse_args();sys.path.insert(0,str(a.inference));import model
 torch.manual_seed(432);torch.empty(1,device='cuda');torch.set_default_dtype(torch.bfloat16);drv=ctypes.CDLL('libcuda.so.1')
 for m,n,k in ((1,1024,4096),(5,1024,4096),(17,1024,4096)):
  x=torch.randn(m,8,k,device='cuda',dtype=torch.bfloat16);w=torch.randn(8,n,k,device='cuda',dtype=torch.float32).to(torch.float8_e4m3fn);scale=torch.full((8,n//32,k//32),120,device='cuda',dtype=torch.uint8).view(torch.float8_e8m0fnu)
  aq,ascale=model.act_quant(x.flatten(1),32,'ue8m0',torch.float8_e8m0fnu);aq=aq.view(m,8,k);sfrows=(m+3)//4*4;sfa=torch.zeros(8*k//128,sfrows,device='cuda',dtype=torch.int32);sfa[:,:m]=ascale.view(torch.uint8).contiguous().view(torch.int32).t();sfb=scale.view(torch.uint8).repeat_interleave(32,1).contiguous().view(torch.int32).transpose(1,2).contiguous();y=torch.empty(m,8,n,device='cuda',dtype=torch.bfloat16)
  modules,ops=pieces(m,sfa_rows=sfrows);op=ops['dsv41_oa'];launch=op['impl']['launches'][0];params=[y,aq,w,sfa,sfb,m,n,k]
  def val(x):return x if isinstance(x,int) else val(x['mul'][0])*val(x['mul'][1])
  def arg(v):
   if 'param'in v:return struct.pack('i',params[v['param']])
   if 'i64'in v:return struct.pack('q',v['i64'])
   pack=v['pack'];fields=pack['fields']
   if 'tensormap' not in fields[0]:return bytes(20)+struct.pack('f',1.)
   t=fields[0]['tensormap'];dtype={'u8':cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,'i32':cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_INT32,'bf16':cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16}[t['dtype']];sw=cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B if t['swizzle'] else cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE;tm=check(cu.cuTensorMapEncodeTiled(dtype,len(t['dims']),params[t['param']].data_ptr(),[cu.cuuint64_t(val(i)) for i in t['dims']],[cu.cuuint64_t(val(i)) for i in t['strides']],[cu.cuuint32_t(i) for i in t['box']],[cu.cuuint32_t(1)]*len(t['dims']),cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,sw,cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));return ctypes.string_at(tm.getPtr(),128)
  root=pathlib.Path(__file__).resolve().parents[3];mod=check(cu.cuModuleLoad(str(root/modules['dsv41_dense']['source']).encode()));fn=check(cu.cuModuleGetFunction(mod,launch['entry'].encode()));check(cu.cuFuncSetAttribute(fn,cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,launch['shared_mem']));keep=[ctypes.create_string_buffer(arg(v)) for v in launch['args']];argv=(ctypes.c_void_p*len(keep))(*[ctypes.addressof(v) for v in keep]);err=drv.cuLaunchKernel(ctypes.c_void_p(int(fn)),152,1,1,256,1,1,launch['shared_mem'],ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),argv,None);assert err==0;torch.cuda.synchronize()
  wbf=(w.float()*scale.float().repeat_interleave(32,1).repeat_interleave(32,2)).bfloat16();ref=torch.einsum('tgd,gnd->tgn',x,wbf);diff=((y.float()-ref.float()).square().sum()/ref.float().square().sum()).item();assert diff<1e-3,diff;print(m,n,k,'relative squared error',diff,flush=True)
  if a.dump:
   from replay import dump_case
   dump_case(a.dump/f'oa_{m}_{n}_{k}',modules,'dsv41_oa',op,params)
if __name__=='__main__':main()
