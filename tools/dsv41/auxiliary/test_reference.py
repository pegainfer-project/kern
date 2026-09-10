"""Compare cubin execution with methods extracted verbatim from supplied inference.

Only tensor allocation/linear projections are replaced; Engram.forward,
NgramHashState.forward and RMSNorm.forward execute their original source.
"""
import argparse
import ast
import ctypes
from pathlib import Path
import torch
from torch import nn
from cuda.bindings import driver as cu


def load_classes(path, names, namespace):
    tree = ast.parse(path.read_text())
    selected = ast.Module(body=[n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names], type_ignores=[])
    exec(compile(selected, str(path), "exec"), namespace)
    return namespace


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inference", type=Path)
    p.add_argument("cubin", type=Path)
    p.add_argument("--dump", type=Path)
    args = p.parse_args()
    torch.manual_seed(11)
    torch.zeros(1, device="cuda")
    def check(result):
        assert result[0] == cu.CUresult.CUDA_SUCCESS, result
        return result[1] if len(result) == 2 else result[1:]
    module = check(cu.cuModuleLoad(str(args.cubin).encode()))
    dump_counts = {}
    def launch(name, grid, *values):
        vals = [v.data_ptr() if isinstance(v, torch.Tensor) else v for v in values]
        types = [ctypes.c_void_p if isinstance(v, torch.Tensor) else ctypes.c_float if isinstance(v, float) else ctypes.c_int for v in values]
        if name == "hash": types[-1] = ctypes.c_int64
        fn = check(cu.cuModuleGetFunction(module, ("dsv41_" + name).encode()))
        check(cu.cuLaunchKernel(fn, *grid, 256, 1, 1, 0, cu.CUstream(torch.cuda.current_stream().cuda_stream), (tuple(vals), tuple(types)), 0))
        torch.cuda.synchronize()
        if args.dump and name in {"engram_inject", "norm", "compress2", "rope"} and not any(isinstance(v, torch.Tensor) and v.data_ptr()==values[0].data_ptr() for v in values[1:]):
            import json, sys
            from ops import definitions
            sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
            from kern_manifest import SCHEMA_VERSION
            key={"norm":"compress1"}.get(name,name)
            modules,all_ops=definitions(args.cubin,rows=grid[0],groups=grid[0],heads=grid[1],copies=grid[1])
            op=all_ops[key];buffers={};call_args=[];io={"inputs":{},"outputs":{}}
            count=dump_counts.get(name,0);dump_counts[name]=count+1
            case=args.dump/f"{name}_{count}";case.mkdir(parents=True,exist_ok=True)
            for index,(value,typ) in enumerate(zip(values,op["params"])):
                if isinstance(value,torch.Tensor):
                    pname=f"p{index}";kind="output" if typ.startswith("out ") else "input"
                    dtype=typ.split("<")[1].split(">")[0]
                    buffers[pname]={"dtype":dtype,"kind":kind,"shape":list(value.shape)}
                    path=case/f"{pname}.bin";path.write_bytes(value.cpu().contiguous().view(torch.uint8).numpy().tobytes())
                    io["outputs" if kind=="output" else "inputs"][pname]=path.name
                    call_args.append({"buf":pname})
                else:call_args.append({typ:value})
            manifest={"schema_version":SCHEMA_VERSION,"model":"auxiliary-probe","vars":{},"states":{},"buffers":buffers,"modules":modules,"ops":{key:op},"programs":{"probe":{"calls":[{"op":key,"args":call_args}]}}}
            (case/"manifest.json").write_text(json.dumps(manifest,indent=2))
            (case/"io.json").write_text(json.dumps(io))
    ns = load_classes(args.inference / "model.py", {"RMSNorm", "Engram", "Compressor"}, {"torch": torch, "nn": nn, "ModelArgs": object, "EngramLayout": object})
    for rows in [1, 5, 17]:
        dim,copies=5120,4
        x = torch.randn(rows,1,copies,dim,device="cuda",dtype=torch.bfloat16)
        kv = torch.randn(rows,1,(copies+1)*dim,device="cuda",dtype=torch.bfloat16)
        qw = torch.randn(copies,dim,device="cuda",dtype=torch.bfloat16)
        kw = torch.randn_like(qw)
        mask = (torch.arange(rows,device="cuda")%3 != 1).to(torch.uint8)
        obj = ns["Engram"].__new__(ns["Engram"]); nn.Module.__init__(obj)
        obj.dim=dim;obj.hc_mult=copies;obj.eps=1e-20;obj.clamp_value=1e-6
        obj.q_weight=nn.Parameter(qw);obj.k_weight=nn.Parameter(kw)
        obj.embed=lambda ids: ids;obj.wkv=lambda ignored:kv
        ref=obj(x,torch.zeros(rows,1,1,1,device="cuda"),mask[:,None].bool())
        out=torch.empty_like(x)
        launch("engram_inject",(rows,copies,1),out,x,kv,qw,kw,mask,rows,copies,dim,1e-20)
        torch.testing.assert_close(out,ref,atol=.03125,rtol=.008)
        print("engram_inject", rows, "max_abs",(out.float()-ref.float()).abs().max().item())
    dim,groups=512,17
    norm=ns["RMSNorm"](dim,1e-20).cuda().bfloat16()
    for ratio in [1,2]:
        if ratio==1:
            x=torch.randn(groups,dim,device="cuda",dtype=torch.bfloat16);ref=norm(x);out=torch.empty_like(ref)
            launch("norm",(groups,1,1),out,x,norm.weight,groups,dim,1e-20)
        else:
            kv=torch.randn(groups,2,dim,device="cuda");score=torch.randn_like(kv)*3
            ref=norm((kv*score.softmax(dim=1)).sum(dim=1).bfloat16());out=torch.empty_like(ref)
            launch("compress2",(groups,1,1),out,kv,score,norm.weight,groups,dim,1e-20)
        torch.testing.assert_close(out,ref,atol=.015625,rtol=.008)
        print("compress",ratio,"max_abs",(out.float()-ref.float()).abs().max().item())
    # Paged projected-history stage/gather versus official stateful compressor.
    obj=ns["Compressor"].__new__(ns["Compressor"]);nn.Module.__init__(obj)
    obj.compress_ratio=2;obj.head_dim=512;obj.norm=norm
    obj.wkv=lambda x:x[...,:512];obj.wgate=lambda x:x[...,512:]
    obj.kv_state=torch.zeros(1,2,512,device="cuda");obj.score_state=torch.full_like(obj.kv_state,-torch.inf)
    kh=torch.zeros(32,512,device="cuda");sh=torch.zeros_like(kh)
    pages2=torch.tensor([[2,0,3,1]],device="cuda",dtype=torch.int32)
    start=0
    for length in [5,1,1,1,1]:
        x=torch.randn(1,length,1024,device="cuda",dtype=torch.bfloat16)
        ref=obj(x,start)
        req=torch.zeros(length,device="cuda",dtype=torch.int32)
        pos=torch.arange(start,start+length,device="cuda",dtype=torch.int32)
        slots=pages2[0,(pos//8).long()].long()*8+pos%8
        kv=x[0,:,:512].float().contiguous();score=x[0,:,512:].float().contiguous()
        launch("compressor_stage",(length,1,1),kh,sh,kv,score,slots,length,512)
        gathered=torch.empty(length,2,512,device="cuda");gs=torch.empty_like(gathered);valid=torch.empty(length,device="cuda",dtype=torch.int32)
        launch("compressor_gather",(length,1,1),gathered,gs,valid,kh,sh,req,pos,pages2,length,512,8,4)
        out=torch.empty(length,512,device="cuda",dtype=torch.bfloat16)
        launch("compress2",(length,1,1),out,gathered,gs,norm.weight,length,512,1e-20)
        if ref is None: assert valid.sum()==0
        else: torch.testing.assert_close(out[valid.bool()],ref[0],atol=.015625,rtol=.008)
        start+=length
    print("paged compressor matches official prefill and partial decode groups")
    # Bounded state: rejected rows are evaluated but never committed. Oracle
    # state is snapshotted at acceptance and restored before the next round.
    obj.kv_state.zero_();obj.score_state.fill_(-torch.inf)
    state=torch.zeros(64,2,512,device="cuda")
    first=torch.randn(1,3,1024,device="cuda",dtype=torch.bfloat16)
    obj(first,0);state[1,0]=first[0,-1,:512].float();state[1,1]=first[0,-1,512:].float()
    start,committed=3,1
    for round_no,naccept in enumerate([1,2,5,6,3,4]):
        x=torch.randn(1,6,1024,device="cuda",dtype=torch.bfloat16)
        kv=x[0,:,:512].float().contiguous();sc=x[0,:,512:].float().contiguous()
        lines=torch.tensor([[committed]+list(range(2+round_no*6,7+round_no*6))],device="cuda",dtype=torch.int32)
        starts=torch.tensor([0,6],device="cuda",dtype=torch.int32)
        req=torch.zeros(6,device="cuda",dtype=torch.int32);pos=torch.arange(start,start+6,device="cuda",dtype=torch.int32)
        gkv=torch.empty(6,2,512,device="cuda");gsc=torch.empty_like(gkv);valid=torch.empty(6,device="cuda",dtype=torch.int32)
        before=state.clone()
        launch("compressor_short_gather",(6,1,1),gkv,gsc,valid,state,kv,sc,lines,starts,req,pos,6,512,6)
        assert torch.equal(state,before)
        out=torch.empty(6,512,device="cuda",dtype=torch.bfloat16)
        launch("compress2",(6,1,1),out,gkv,gsc,norm.weight,6,512,1e-20)
        for row in range(6):
            ref=obj(x[:,row:row+1],start+row)
            if ref is not None: torch.testing.assert_close(out[row],ref[0,0],atol=.015625,rtol=.008)
            assert bool(valid[row])==(ref is not None)
            if row+1==naccept: saved=(obj.kv_state.clone(),obj.score_state.clone())
        accepted=torch.tensor([naccept],device="cuda",dtype=torch.int32)
        launch("compressor_commit",(1,1,1),state,kv,sc,lines,starts,accepted,1,512,6)
        committed=lines[0,naccept-1].item()
        assert torch.equal(state[committed,0],kv[naccept-1])
        assert torch.equal(state[committed,1],sc[naccept-1])
        obj.kv_state.copy_(saved[0]);obj.score_state.copy_(saved[1]);start+=naccept
    print("bounded compressor rejection trajectories accepted1/2/5/6/3/4 match official")
    # FP8 embedding reference method, real E8M0 scales; exact cast expected.
    ns2=load_classes(args.inference/"model.py",{"ParallelEngramEmbedding"},{"torch":torch,"nn":nn,"F":torch.nn.functional,"world_size":1})
    obj=ns2["ParallelEngramEmbedding"].__new__(ns2["ParallelEngramEmbedding"]);nn.Module.__init__(obj)
    obj.vocab_start_idx=0;obj.vocab_end_idx=100;obj.block_size=32
    obj.weight=nn.Parameter(torch.randn(100,256,device="cuda").to(torch.float8_e4m3fn),requires_grad=False)
    obj.scale=nn.Parameter(torch.randint(120,135,(100,8),device="cuda",dtype=torch.uint8).view(torch.float8_e8m0fnu),requires_grad=False)
    ids=torch.randint(0,100,(5,24),device="cuda");ref=obj(ids);out=torch.empty_like(ref)
    launch("lookup",(5,24,1),out,ids,obj.weight,obj.scale,5,24,256,24,0)
    assert torch.equal(out,ref)
    print("lookup bit exact")
    host_table=obj.weight.detach().view(torch.uint8).cpu().pin_memory()
    host_scale=obj.scale.detach().view(torch.uint8).cpu().pin_memory()
    # UVA host pointer is valid for cudaHostAlloc-mapped PyTorch pinned storage.
    launch("lookup",(5,24,1),out,ids,host_table,host_scale,5,24,256,24,0)
    assert torch.equal(out,ref)
    print("pinned-host UVA lookup bit exact")
    # Paged hash: compare all rows and chunk/decode to official cached method.
    ns3=load_classes(args.inference/"engram.py",{"NgramHashState"},{"torch":torch,"nn":nn,"EngramLayout":object})
    obj=ns3["NgramHashState"].__new__(ns3["NgramHashState"]);nn.Module.__init__(obj)
    from types import SimpleNamespace
    obj.layout=SimpleNamespace(max_ngram_size=4);obj.pad_id=2
    obj.token_map=torch.arange(64,device="cuda",dtype=torch.int64)//2
    obj.cache=torch.zeros(2,32,device="cuda",dtype=torch.int64)
    obj.multipliers=torch.tensor([[11,31,63,97],[107,111,333,555]],device="cuda",dtype=torch.int64)
    obj.primes=torch.tensor([[[101,103],[107,109],[113,127]],[[131,137],[139,149],[151,157]]],device="cuda",dtype=torch.int64)
    sizes=obj.primes.flatten(1);obj.offsets=torch.cat([torch.zeros(2,1,device="cuda",dtype=torch.int64),sizes.cumsum(1)[:,:-1]],1)
    history=torch.full((64,),-9,device="cuda",dtype=torch.int64)
    pages=torch.tensor([[3,0,5,7],[1,6,2,4]],device="cuda",dtype=torch.int32)
    start=0
    for length in [5,1,9,1]:
        ids=torch.randint(0,64,(2,length),device="cuda",dtype=torch.int32)
        mask=torch.rand(2,length,device="cuda")>.2
        ref=obj(ids,start,mask)
        req=torch.arange(2,device="cuda",dtype=torch.int32).repeat_interleave(length)
        pos=torch.arange(start,start+length,device="cuda",dtype=torch.int32).repeat(2)
        slots=pages[req.long(),(pos//8).long()].long()*8+pos%8
        launch("history",((2*length+255)//256,1,1),history,ids,slots,obj.token_map,mask.to(torch.uint8),2*length)
        out=torch.empty_like(ref)
        launch("hash",(2*length,1,1),out,history,req,pos,pages,obj.multipliers,obj.primes,obj.offsets,2*length,8,4,2,2,4,2)
        assert torch.equal(out,ref),(out,ref)
        start+=length
    print("paged hash bit exact across chunk boundaries and dead tokens")
    import importlib.util
    spec=importlib.util.spec_from_file_location("dsv41_reference_kernel",args.inference/"kernel.py")
    kernel=importlib.util.module_from_spec(spec);spec.loader.exec_module(kernel)
    rows,page_size=17,16
    x=torch.randn(rows,512,device="cuda",dtype=torch.bfloat16)*4
    x[0]=0
    slots=torch.tensor([31,0,17,2,19,4,21,6,23,8,25,10,27,12,29,14,16],device="cuda",dtype=torch.int64)
    for fp4 in [0,1]:
        cache=torch.zeros(2*page_size*(288 if fp4 else 528),device="cuda",dtype=torch.uint8)
        launch("cache_fp4" if fp4 else "cache_fp8",(rows,1,1),cache,x,slots,rows,page_size)
        out=torch.empty_like(x)
        launch("cache_gather",(rows,1,1),out,cache,slots,rows,page_size,fp4)
        ref=x.clone()
        if fp4: kernel.fp4_act_quant(ref,16,True,scale_dtype=torch.float8_e4m3fn)
        else: kernel.act_quant(ref,32,"ue8m0",torch.float8_e8m0fnu,True)
        assert torch.equal(out,ref),(fp4,(out.float()-ref.float()).abs().max().item(),(out!=ref).sum().item())
        print("cache quant/gather bit exact", "FP4" if fp4 else "FP8")
    iq=torch.randn(17,128,device="cuda",dtype=torch.bfloat16);iq[0]=0
    packed=torch.empty(17,64,device="cuda",dtype=torch.uint8);scales=torch.empty(17,4,device="cuda",dtype=torch.uint8);deq=torch.empty_like(iq)
    launch("index_quant",(17,1,1),packed,scales,deq,iq,17,128)
    ref=iq.clone();kernel.fp4_act_quant(ref,32,True)
    assert torch.equal(deq,ref)
    print("index MXFP4 quant/dequant bit exact")
    # Actual reference rotation on interleaved pairs, including inverse.
    tree=ast.parse((args.inference/"model.py").read_text())
    ns={"torch":torch,"lru_cache":__import__("functools").lru_cache}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {"apply_rotary_emb", "get_window_topk_idxs", "get_dspark_topk_idxs"}],type_ignores=[]),"reference_model","exec"),ns)
    # Window order agrees with official prefill; long causal/DSpark sets
    # agree after converting official ring slots to logical positions.
    pages3=torch.arange(32,device="cuda",dtype=torch.int32).view(1,32)
    for noncausal in [0,1]:
        count,width=5,192 if noncausal else 128
        req=torch.zeros(count,device="cuda",dtype=torch.int32)
        pos=torch.arange(151,156,device="cuda",dtype=torch.int32) if noncausal else torch.arange(count,device="cuda",dtype=torch.int32)
        end=torch.full((count,),156,device="cuda",dtype=torch.int32)
        out=torch.empty(count,width,device="cuda",dtype=torch.int32)
        launch("window_indices",(count,1,1),out,req,pos,end,pages3,count,128,width,16,32,noncausal,5)
        if not noncausal:
            ref=ns["get_window_topk_idxs"](128,1,count,0)[0].cuda()
            assert torch.equal(out[:,:count],ref)
            assert (out[:,count:]==-1).all()
        else:
            ring=ns["get_dspark_topk_idxs"](128,1,5,150)[0].cuda()
            # accepted context positions23..150 occupy ring positions0..127.
            logical=torch.where(ring<128,torch.where(ring<23,ring+128,ring),ring-128+151)
            assert torch.equal(out[:,:133].sort(-1).values,logical.sort(-1).values)
            assert (out[:,133:]==-1).all()
    print("causal/DSpark indices match reference, padding invalid")
    x=torch.randn(1,5,64,512,device="cuda",dtype=torch.bfloat16)
    angle=torch.randn(32,32,device="cuda");freq=torch.polar(torch.ones_like(angle),angle)
    pos=torch.arange(3,8,device="cuda",dtype=torch.int32)
    for inverse in [0,1]:
        ref=x.clone();ns["apply_rotary_emb"](ref[...,-64:],freq[3:8],bool(inverse))
        out=torch.empty_like(x)
        launch("rope",(5,64,1),out,x,torch.view_as_real(freq),pos,5,64,512,64,inverse)
        torch.testing.assert_close(out,ref,atol=.015625,rtol=.008)
        print("rope",inverse,"max_abs",(out.float()-ref.float()).abs().max().item())
        alias=x.clone()
        launch("rope",(5,64,1),alias,alias,torch.view_as_real(freq),pos,5,64,512,64,inverse)
        assert torch.equal(alias,out)
        print("rope in-place matches out-of-place",inverse)

if __name__ == "__main__": main()
