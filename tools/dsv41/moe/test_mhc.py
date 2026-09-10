"""Launch the exact AOT op ABI and compare to supplied inference Block methods.

The MXFP8 copy of the normalized output must be byte-identical to the
standalone quantization kernels applied to the BF16 output."""
import ctypes, pathlib, argparse, struct, sys
import torch
from cuda.bindings import driver as cu
from mhc import pieces
from test_boundary import check

def main():
    p=argparse.ArgumentParser();p.add_argument('--inference',type=pathlib.Path,required=True);p.add_argument('--dump',type=pathlib.Path);a=p.parse_args();sys.path.insert(0,str(a.inference));import model
    torch.manual_seed(432);torch.empty(1,device='cuda');drv=ctypes.CDLL('libcuda.so.1')
    stage=check(cu.cuModuleLoad(str(pathlib.Path(__file__).resolve().parents[3]/'target/cubins/dsv41/stage.cubin').encode()))
    for n,fp8 in [(n,fp8) for n in (1,5,65,128) for fp8 in ('gemm','moe')]:
        shared_rows=131072;modules,ops=pieces(rows=n,max_tokens=n,fp8=fp8,shared_rows=shared_rows if fp8=='moe' else None);launch=ops['dsv41_mhc']['impl']['launches'][-1]
        x=torch.randn(n,5120,device='cuda',dtype=torch.bfloat16);r=torch.randn(n,4,5120,device='cuda',dtype=torch.bfloat16)
        post=torch.randn(n,4,device='cuda').sigmoid();comb=torch.randn(n,4,4,device='cuda').softmax(-1);pre=torch.randn_like(post).sigmoid()
        fn=torch.randn(24,20480,device='cuda')*.01;scales=torch.randn(3,device='cuda')*.1;bases=torch.randn(24,device='cuda')*.1;norm=torch.randn(5120,device='cuda',dtype=torch.bfloat16)*.1+1
        sf_rows=(n+3)//4*4;y_fp8=torch.zeros(n,5120,dtype=torch.uint8,device='cuda');y_sf=torch.zeros(40,sf_rows,dtype=torch.int32,device='cuda') if fp8=='gemm' else torch.zeros(n,40,dtype=torch.int32,device='cuda');y_shared=torch.zeros(40,shared_rows,dtype=torch.int32,device='cuda')
        params=[x,r,post,comb,pre,fn,scales,bases,norm,torch.empty_like(r),torch.empty_like(pre),torch.empty_like(post),torch.empty_like(comb),torch.empty_like(x),n,torch.zeros(524288,dtype=torch.int64,device='cuda'),y_fp8,y_sf]+([y_shared] if fp8=='moe' else [])
        scratch={k:torch.zeros(v['shape'],dtype=torch.uint8 if v['dtype']=='u8' else torch.int64,device='cuda') for k,v in ops['dsv41_mhc']['impl']['scratch'].items()}
        def addr(i):return params[i].data_ptr() if isinstance(params[i],torch.Tensor) else params[i]
        keep=[]
        def arg(v):
            if 'scratch' in v:return struct.pack('Q',scratch[v['scratch']].data_ptr())
            if 'param' in v:return struct.pack('Q',addr(v['param']))
            blob=bytearray(v['pack']['size'])
            for f in v['pack']['fields']:
                if 'tensormap' in f:
                    t=f['tensormap'];dtypes={'bf16':cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,'f32':cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT32,'tf32':cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_TFLOAT32};sw={0:cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,64:cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B,128:cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B}
                    tm=check(cu.cuTensorMapEncodeTiled(dtypes[t['dtype']],len(t['dims']),addr(t['param']),[cu.cuuint64_t(v) for v in t['dims']],[cu.cuuint64_t(v) for v in t['strides']],[cu.cuuint32_t(v) for v in t['box']],[cu.cuuint32_t(1)]*len(t['dims']),cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,sw[t['swizzle']],cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));data=ctypes.string_at(tm.getPtr(),128)
                elif 'param' in f:data=struct.pack('Q' if isinstance(params[f['param']],torch.Tensor) else 'i',addr(f['param']))
                elif 'f32' in f:data=struct.pack('f',f['f32'])
                elif 'i64' in f:data=struct.pack('q',f['i64'])
                else:data=struct.pack('i',f['i32'])
                blob[f['at']:f['at']+len(data)]=data
            return bytes(blob)
        mod=check(cu.cuModuleLoad(str(pathlib.Path(__file__).resolve().parents[3]/modules['dsv41_mhc']['source']).encode()));fun=check(cu.cuModuleGetFunction(mod,launch['entry'].encode()));check(cu.cuFuncSetAttribute(fun,cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,launch['shared_mem']))
        keep=[ctypes.create_string_buffer(arg(v)) for v in launch['args']];argv=(ctypes.c_void_p*len(keep))(*[ctypes.addressof(v) for v in keep]);res=drv.cuLaunchKernel(ctypes.c_void_p(int(fun)),152,1,1,768,1,1,227328,ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),argv,None);assert res==0,res;torch.cuda.synchronize()
        ref_r=model.Block.hc_post(None,x[None],r[None],post[None],comb[None]);ctx=type('Ctx',(),{'norm_eps':1e-20,'hc_mult':4,'hc_sinkhorn_iters':20,'hc_eps':1e-6})();pr,po,co=model.Block.hc_mixes(ctx,ref_r,fn,scales,bases);collapse=model.Block.hc_pre(None,ref_r,pre[None]);ref_y=model.RMSNorm.forward(type('Norm',(),{'weight':norm,'eps':1e-20})(),collapse)
        if a.dump:
            from replay import dump_case
            dump_case(a.dump/f'mhc_{n}',modules,'dsv41_mhc',ops['dsv41_mhc'],params)
        for label,actual,expected in zip(['residual','pre','post','comb','y'],params[9:14],[ref_r.squeeze(0),pr.squeeze(0),po.squeeze(0),co.squeeze(0),ref_y.squeeze(0)]):
            diff=(actual.float()-expected.float()).square().sum()/(expected.float().square().sum()+1e-20);print(n,label,float(diff),flush=True);assert diff<5e-5,(n,label,diff)
        # The standalone kernels quantize the kernel's own BF16 output.
        y=params[13];q_fp8=torch.zeros_like(y_fp8);q_sf=torch.zeros_like(y_sf);q_shared=torch.zeros_like(y_shared)
        if fp8=='gemm':entry,ints,warps=b'dsv41_dense_quant_x',[n,5120,5120,sf_rows],sf_rows*40
        else:entry,ints,warps=b'dsv41_mega_quant_x',[n,5120,5120,40],n*40
        ptrs=[y.data_ptr(),q_fp8.data_ptr(),q_sf.data_ptr()];tail=[q_shared.data_ptr(),shared_rows] if fp8=='moe' else []
        values=[ctypes.c_void_p(v) for v in ptrs]+[ctypes.c_int(v) for v in ints]+([ctypes.c_void_p(tail[0]),ctypes.c_int(tail[1])] if tail else [])
        argv=(ctypes.c_void_p*len(values))(*[ctypes.addressof(v) for v in values]);qfun=check(cu.cuModuleGetFunction(stage,entry))
        res=drv.cuLaunchKernel(ctypes.c_void_p(int(qfun)),(warps+7)//8,1,1,256,1,1,0,ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),argv,None);assert res==0,res;torch.cuda.synchronize()
        same=(torch.equal(y_fp8,q_fp8),torch.equal(y_sf,q_sf),torch.equal(y_shared,q_shared));print(n,fp8,'fp8/sf/shared byte-equal',same,'nonzero sf words',int((y_sf!=0).sum()),flush=True);assert all(same),(n,fp8,same)
if __name__=='__main__':main()
