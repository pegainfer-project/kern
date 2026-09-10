"""Once permutations plus GEMM, checked against original model.linear/einsum."""
import argparse,ctypes,json,pathlib,struct,sys
import torch
from safetensors import safe_open
from cuda.bindings import driver as cu
from test_boundary import check
from dense import layout_pieces,pieces,oa_pieces
from replay import dump_once,dump_case


def dense_launch(modules,op,params):
    root=pathlib.Path(__file__).resolve().parents[3];launch=op['impl']['launches'][0]
    def arg(v):
        if 'param' in v:return struct.pack('i',params[v['param']])
        if 'i64' in v:return struct.pack('q',v['i64'])
        fields=v['pack']['fields']
        if 'tensormap' not in fields[0]:return bytes(20)+struct.pack('f',1.)
        t=fields[0]['tensormap'];rank=len(t['dims'])
        dtype={'u8':cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,'i32':cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_INT32,'bf16':cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16}[t['dtype']]
        sw=cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B if t['swizzle'] else cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE
        tm=check(cu.cuTensorMapEncodeTiled(dtype,rank,params[t['param']].data_ptr(),[cu.cuuint64_t(i) for i in t['dims']],[cu.cuuint64_t(i) for i in t['strides']],[cu.cuuint32_t(i) for i in t['box']],[cu.cuuint32_t(1)]*rank,cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,sw,cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE))
        return ctypes.string_at(tm.getPtr(),128)
    mod=check(cu.cuModuleLoad(str(root/modules['dsv41_dense']['source']).encode()));fn=check(cu.cuModuleGetFunction(mod,launch['entry'].encode()))
    check(cu.cuFuncSetAttribute(fn,cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,launch['shared_mem']))
    keep=[ctypes.create_string_buffer(arg(v)) for v in launch['args']];argv=(ctypes.c_void_p*len(keep))(*[ctypes.addressof(v) for v in keep])
    drv=ctypes.CDLL('libcuda.so.1');err=drv.cuLaunchKernel(ctypes.c_void_p(int(fn)),152,1,1,256,1,1,launch['shared_mem'],ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),argv,None);assert err==0
    torch.cuda.synchronize()


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=pathlib.Path,required=True);p.add_argument('--dump',type=pathlib.Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(a.checkpoint/'inference'));import model
    torch.manual_seed(772);torch.empty(1,device='cuda');torch.set_default_dtype(torch.bfloat16)
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
            y=torch.empty(m,groups*n,device='cuda',dtype=torch.bfloat16)
            dm,do=oa_pieces(m) if which else pieces(m,n,k);op=next(iter(do.values()));params=[y,aq,outw,sfa,outsf,m,n,k];dense_launch(dm,op,params)
            if which:
                wbf=(weight.float()*rawsf.float().repeat_interleave(32,0).repeat_interleave(32,1)).bfloat16().view(8,n,k)
                ref=torch.einsum('tgd,gnd->tgn',x.view(m,8,k),wbf).flatten(1)
            else:
                weight.scale=rawsf;ref=model.linear(x,weight).view(m,64,32,16).transpose(1,2).contiguous().view_as(y)
            diff=float((y.float()-ref.float()).square().sum()/ref.float().square().sum());assert diff<(0.003 if which else 1e-5),diff
            dump_case(a.dump/f'{kind}_gemm_{m}',dm,next(iter(do)),op,params);print(kind,m,'relative squared error',diff,flush=True)
    torch.set_default_dtype(torch.float32)

if __name__=='__main__':main()
