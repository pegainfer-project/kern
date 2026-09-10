"""O-A MXFP8 GEMM: its own dequantized output, then WO_B fed straight from it.

Buffers are allocated at the manifest's fixed row capacity and the live M is
passed as the shape, so the padding rows exercise the same clipping the
runtime relies on.
"""
import argparse,ctypes,pathlib,struct,sys
import torch
from cuda.bindings import driver as cu
from test_boundary import check
from dense import oa_pieces,pieces

DTYPE={'u8':cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
       'i32':cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_INT32,
       'bf16':cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16}

def val(x):return x if isinstance(x,int) else val(x['mul'][0])*val(x['mul'][1])

def tensormap(t,params):
 sw=cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B if t['swizzle'] else cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE
 rank=len(t['dims'])
 tm=check(cu.cuTensorMapEncodeTiled(DTYPE[t['dtype']],rank,params[t['param']].data_ptr(),
   [cu.cuuint64_t(val(i)) for i in t['dims']],[cu.cuuint64_t(val(i)) for i in t['strides']],
   [cu.cuuint32_t(i) for i in t['box']],[cu.cuuint32_t(1)]*rank,
   cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,sw,
   cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
   cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE))
 return ctypes.string_at(tm.getPtr(),128)

def field(f,params):
 if 'tensormap' in f:return tensormap(f['tensormap'],params)
 if 'param' in f:
  x=params[f['param']]
  return struct.pack('q',x.data_ptr()) if torch.is_tensor(x) else struct.pack('i',x)
 for key,fmt in (('i32','i'),('f32','f'),('i64','q'),('u8','B')):
  if key in f:return struct.pack(fmt,f[key])
 raise ValueError(f)

def arg(v,params):
 if 'param' in v:return struct.pack('i',params[v['param']])
 if 'i64' in v:return struct.pack('q',v['i64'])
 image=bytearray(v['pack']['size'])
 for f in v['pack']['fields']:
  raw=field(f,params);image[f['at']:f['at']+len(raw)]=raw
 return bytes(image)

def run(drv,root,modules,op,params):
 launch=op['impl']['launches'][0]
 mod=check(cu.cuModuleLoad(str(root/modules[launch['module']]['source']).encode()))
 fn=check(cu.cuModuleGetFunction(mod,launch['entry'].encode()))
 check(cu.cuFuncSetAttribute(fn,cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,launch['shared_mem']))
 keep=[ctypes.create_string_buffer(arg(v,params)) for v in launch['args']]
 argv=(ctypes.c_void_p*len(keep))(*[ctypes.addressof(v) for v in keep])
 err=drv.cuLaunchKernel(ctypes.c_void_p(int(fn)),*launch['grid'],*launch['block'],launch['shared_mem'],
                        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),argv,None)
 assert err==0,err
 torch.cuda.synchronize()

def mxfp8(x):
 """Per-32 UE8M0 quantization, values and raw exponents, matching `model.act_quant`.

 The reference kernel races above 32 rows and writes E4M3 NaN bytes, so the
 GEMM inputs come from here instead; `check_reference` ties the two together.
 """
 g=x.float().unflatten(-1,(-1,32))
 bits=(g.abs().amax(-1,keepdim=True).clamp_min(1e-4)/448.).view(torch.int32)
 exponent=(((bits>>23)&0xff)+((bits&0x7fffff)!=0).int()).clamp(1,254)
 return (g/torch.exp2((exponent-127).float())).to(torch.float8_e4m3fn).flatten(-2),exponent.squeeze(-1).to(torch.uint8)

def dequantize(values,exponent):
 return values.float()*torch.exp2(exponent.float()-127.).repeat_interleave(32,-1)

def error(got,ref):
 return ((got.float()-ref.float()).square().sum()/ref.float().square().sum()).item()

def check_reference(model):
 """32 rows is below the size at which the reference quantizer starts racing."""
 x=torch.randn(32,4096,device='cuda',dtype=torch.bfloat16)
 q,s=model.act_quant(x,32,'ue8m0',torch.float8_e8m0fnu);mine,exponent=mxfp8(x)
 assert (q.view(torch.uint8)==mine.view(torch.uint8)).all() and (s.view(torch.uint8)==exponent).all()

def main():
 p=argparse.ArgumentParser();p.add_argument('--inference',type=pathlib.Path,required=True);p.add_argument('--dump',type=pathlib.Path);a=p.parse_args()
 sys.path.insert(0,str(a.inference));import model
 torch.manual_seed(432);torch.empty(1,device='cuda');torch.set_default_dtype(torch.bfloat16);drv=ctypes.CDLL('libcuda.so.1')
 root=pathlib.Path(__file__).resolve().parents[3];check_reference(model)
 n,k,hidden,cap=1024,4096,5120,128
 w=torch.randn(8,n,k,device='cuda',dtype=torch.float32).to(torch.float8_e4m3fn)
 scale=torch.full((8,n//32,k//32),120,device='cuda',dtype=torch.uint8).view(torch.float8_e8m0fnu)
 wbf=(w.float()*scale.float().repeat_interleave(32,1).repeat_interleave(32,2)).bfloat16()
 sfb=scale.view(torch.uint8).repeat_interleave(32,1).contiguous().view(torch.int32).transpose(1,2).contiguous()
 wb=torch.randn(hidden,8*n,device='cuda',dtype=torch.float32).to(torch.float8_e4m3fn)
 wb_scale=torch.full((hidden//32,8*n//32),120,device='cuda',dtype=torch.uint8).view(torch.float8_e8m0fnu)
 wb_bf=(wb.float()*wb_scale.float().repeat_interleave(32,0).repeat_interleave(32,1)).bfloat16()
 sfb2=wb_scale.view(torch.uint8).repeat_interleave(32,0).contiguous().view(torch.int32).t().contiguous()
 modules,ops=oa_pieces(cap,sfa_rows=cap);dmodules,dops=pieces(cap,hidden,8*n,sfa_rows=cap)
 for m in (1,5,65,128):
  x=torch.randn(cap,8,k,device='cuda',dtype=torch.bfloat16)
  aq,ascale=mxfp8(x.flatten(1));aq=aq.view(cap,8,k)
  sfa=ascale.view(torch.int32).t().contiguous()
  y=torch.empty(cap,8,n,device='cuda',dtype=torch.float8_e4m3fn)
  sfd=torch.empty(8*n//128,cap,device='cuda',dtype=torch.int32)
  run(drv,root,modules,ops['dsv41_oa'],[y,aq,w,sfa,sfb,m,n,k,sfd])
  ref=torch.einsum('tgd,gnd->tgn',dequantize(aq[:m],ascale.view(cap,8,k//32)[:m]).bfloat16(),wbf).flatten(1)
  want,want_exponent=mxfp8(ref)
  exponent=sfd[:,:m].t().contiguous().view(torch.uint8)
  print(f'm={m} o-a dequantized {error(dequantize(y[:m].view(m,8*n),exponent),ref):.3e}'
        f' value bytes differing {(y[:m].view(torch.uint8).view(m,8*n)!=want.view(torch.uint8)).float().mean().item():.3e}'
        f' exponents differing {(exponent!=want_exponent).float().mean().item():.3e}',flush=True)
  out=torch.empty(cap,hidden,device='cuda',dtype=torch.bfloat16)
  run(drv,root,dmodules,dops['dsv41_dense'],[out,y.view(cap,8*n),wb,sfd,sfb2,m,hidden,8*n])
  chained=dequantize(y[:m].view(m,8*n),exponent).bfloat16()@wb_bf.t()
  print(f'm={m} wo_b from o-a {error(out[:m],chained):.3e}',flush=True)
  if a.dump:
   from replay import dump_case
   dump_case(a.dump/f'oa_{m}_{n}_{k}',modules,'dsv41_oa',ops['dsv41_oa'],[y,aq,w,sfa,sfb,m,n,k,sfd])
if __name__=='__main__':main()
