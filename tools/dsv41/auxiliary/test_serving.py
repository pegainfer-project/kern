"""GPU metadata regression: chunk/verify boundaries, padding and null commits."""
import argparse
import ast
from functools import lru_cache
import ctypes
import json
from pathlib import Path
import torch
from cuda.bindings import driver as cu


def check(result):
    assert result[0] == cu.CUresult.CUDA_SUCCESS, result
    return result[1] if len(result) == 2 else result[1:]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cubin", type=Path)
    p.add_argument("--constants", type=Path)
    p.add_argument("--inference", type=Path)
    a = p.parse_args()
    reference_topk=None
    if a.inference:
        tree=ast.parse((a.inference/"model.py").read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="get_dspark_topk_idxs")
        scope={"torch":torch,"lru_cache":lru_cache}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),"official_dspark_indices","exec"),scope)
        reference_topk=scope["get_dspark_topk_idxs"]
    torch.manual_seed(31)
    torch.zeros(1, device="cuda")
    mod = check(cu.cuModuleLoad(str(a.cubin).encode()))

    def launch(name, grid, *args, module=mod):
        fn = check(cu.cuModuleGetFunction(module, name.encode()))
        vals = tuple(v.data_ptr() if isinstance(v, torch.Tensor) else v for v in args)
        types = tuple(ctypes.c_void_p if isinstance(v, torch.Tensor) else ctypes.c_int for v in args)
        check(cu.cuLaunchKernel(fn, grid, 1, 1, 256, 1, 1, 0,
            cu.CUstream(torch.cuda.current_stream().cuda_stream), (vals, types), 0))
        torch.cuda.synchronize()

    alloc = lambda n, dt: torch.empty(n, device="cuda", dtype=dt)
    tensor = lambda values, dt=torch.int32: torch.tensor(values, device="cuda", dtype=dt)
    for mode, starts_in, bases, active in [
        ("prefill", [0,17], [127], [1]),
        ("decode", [0,1,2,3], [0,129,500], [1,0,1]),
        ("verify", [0,6,12,18], [127,0,255], [1,0,1]),
        ("draft", [0,6,12,18], [127,0,255], [1,0,1]),
    ]:
        seqs = len(bases)
        positions = sum([list(range(base,base+starts_in[i+1]-starts_in[i])) for i,base in enumerate(bases)],[])
        valid = sum([[active[i]]*(starts_in[i+1]-starts_in[i]) for i in range(seqs)],[])
        # Noncontiguous physical pages prove positions are not physical slots.
        slots = [(p//128+3)*128+p%128 for p in positions]
        draft = int(mode=="draft")
        rows = seqs*5 if draft else starts_in[-1]
        ids = tensor(list(range(rows)),torch.int64)
        src_pos,src_slots,src_valid,cu_in = tensor(positions),tensor(slots,torch.int64),tensor(valid),tensor(starts_in)
        lens=tensor([base+starts_in[i+1]-starts_in[i] for i,base in enumerate(bases)])
        req,pos,slot,mask = alloc(rows,torch.int32),alloc(rows,torch.int32),alloc(rows,torch.int64),alloc(rows,torch.uint8)
        starts,live,ends,ids32,counts = alloc(seqs+1,torch.int32),alloc(seqs,torch.int32),alloc(rows,torch.int32),alloc(rows,torch.int32),alloc(seqs,torch.int32)
        winlen,complen=alloc(rows,torch.int32),alloc(rows,torch.int32)
        launch("dsv41_metadata",(rows+255)//256,req,pos,slot,mask,starts,live,ends,ids32,counts,winlen,complen,
            src_pos,src_slots,src_valid,ids,cu_in,lens,rows,seqs,draft)
        selection = [starts_in[s]+j for s in range(seqs) for j in range(5)] if draft else list(range(rows))
        expected_pos = tensor([positions[i] for i in selection])
        expected_valid = tensor([valid[i] for i in selection],torch.uint8)
        expected_slot = tensor([slots[i] if valid[i] else -1 for i in selection],torch.int64)
        assert torch.equal(pos,expected_pos) and torch.equal(slot,expected_slot) and torch.equal(mask,expected_valid)
        assert torch.equal(ids32,ids.int())
        assert torch.equal(winlen,torch.where(mask.bool(),torch.minimum(ends if draft else pos+1,torch.full_like(pos,133 if draft else 128)),0))
        assert torch.equal(complen,torch.where(mask.bool(),512,0).int())
        assert torch.equal(live,tensor(active))
        assert torch.equal(starts,tensor([i*5 for i in range(seqs+1)] if draft else starts_in))
        assert torch.equal(counts,tensor([active[i]*(5 if draft else starts_in[i+1]-starts_in[i]) for i in range(seqs)]))
        # Query-local pages are copied by request, never inferred from slots.
        # End lengths match sequential official Indexer calls, including verify.
        page_cols=8
        page_table=torch.arange(seqs*page_cols,device="cuda",dtype=torch.int32).reshape(seqs,page_cols)+37
        if draft and reference_topk:
            window=alloc((rows,192),torch.int32)
            launch("dsv41_window_indices",rows,window,req,pos,ends,page_table,rows,128,192,128,page_cols,1,5)
            for row in range(rows):
                seq=int(req[row]);anchor=bases[seq]
                if not active[seq] or anchor<=1:continue
                original=reference_topk(128,1,5,anchor-1)[0,0].tolist()
                logical=[]
                for index in original:
                    if index>=128:logical.append(anchor+index-128)
                    else:
                        absolute=(anchor-1)//128*128+index
                        logical.append(absolute if absolute<anchor else absolute-128)
                physical=[int(page_table[seq,p//128])*128+p%128 for p in logical]
                assert sorted(window[row][window[row]>=0].tolist())==sorted(physical)
                assert int(winlen[row])==len(original)
            print("draft RoPE positions and window visibility match official DSpark indices")
        for ratio in (1,2):
            query_pages=torch.full((rows,page_cols),-999,device="cuda",dtype=torch.int32)
            compressed_end=alloc(rows,torch.int32)
            request_ids,sparse_ends=alloc(rows,torch.int32),alloc(rows,torch.int32)
            launch("dsv41_index_metadata",rows,query_pages,compressed_end,request_ids,sparse_ends,req,pos,mask,page_table,
                   rows,page_cols,128//ratio,ratio)
            assert torch.equal(request_ids,torch.arange(rows,device="cuda",dtype=torch.int32))
            assert torch.equal(sparse_ends,torch.where(mask.bool(),16384,0).int())
            expected_end=torch.where(mask.bool(),(pos+1)//ratio,0)
            assert torch.equal(compressed_end,expected_end)
            used=(expected_end+128//ratio-1)//(128//ratio)
            expected_pages=torch.where(torch.arange(page_cols,device="cuda")[None,:]<used[:,None],
                                       page_table[req.long()],0)
            assert torch.equal(query_pages,expected_pages)
        cslot,cpos=alloc(rows,torch.int64),alloc(rows,torch.int32)
        launch("dsv41_compressed_metadata",(rows+255)//256,cslot,cpos,slot,pos,mask,rows,2)
        complete=mask.bool() & ((pos+1)%2==0)
        assert torch.equal(cslot,torch.where(complete,slot//2,-1))
        assert torch.equal(cpos,torch.where(complete,pos-1,0))
        history=torch.full((2048,),-777,device="cuda",dtype=torch.int64);expected=history.clone()
        remap=torch.arange(64,device="cuda",dtype=torch.int64)
        launch("dsv41_history",(rows+255)//256,history,ids32,slot,remap,mask,rows)
        expected[slot[mask.bool()]]=ids32[mask.bool()].long()
        assert torch.equal(history,expected)
        indices=torch.full((rows,192),17,device="cuda",dtype=torch.int32)
        launch("dsv41_indices_mask",rows,indices,mask,rows,192)
        assert (indices[~mask.bool()]==-1).all() and (indices[mask.bool()]==17).all()
        keys=torch.randn(rows,128,device="cuda",dtype=torch.bfloat16)
        cache=torch.full((2048,128),9,device="cuda",dtype=torch.bfloat16);expected=cache.clone()
        launch("dsv41_cache_bf16",rows,cache,keys,cslot,rows,128)
        expected[cslot[complete]]=keys[complete]
        assert torch.equal(cache,expected)
        packed=torch.randint(0,256,(rows,64),device="cuda",dtype=torch.uint8)
        scales=torch.randint(0,255,(rows,4),device="cuda",dtype=torch.uint8)
        for page_size in (32,64,128):
            stride=(page_size*68+511)//512*512
            icache=torch.full((2048//page_size*stride,),93,device="cuda",dtype=torch.uint8);expected=icache.clone()
            launch("dsv41_cache_index",rows,icache,packed,scales,cslot,rows,page_size)
            for row in range(rows):
                s=int(cslot[row])
                if s>=0:
                    base=(s//page_size)*stride;token=s%page_size
                    expected[base+token*64:base+token*64+64]=packed[row]
                    expected[base+page_size*64+token*4:base+page_size*64+token*4+4]=scales[row]
            assert torch.equal(icache,expected), page_size
        sanitized=alloc(seqs,torch.int32)
        raw=tensor([999]*seqs)
        launch("dsv41_accepted_mask",1,sanitized,raw,live,starts,seqs)
        assert torch.equal(sanitized,counts)
        print(mode,"metadata/padding/ratio2/history/index cache checks passed")
    # Null line must remain unchanged even if a caller supplies a positive count.
    state=torch.full((3,2,512),7,device="cuda",dtype=torch.float32);before=state.clone()
    kv=torch.randn(12,512,device="cuda");score=torch.randn_like(kv)
    launch("dsv41_compressor_commit",2,state,kv,score,tensor([0,2]),tensor([0,6,12]),tensor([6,0]),2,512,1)
    assert torch.equal(state,before)
    print("line0 and accepted0 commit are exact no-ops")
    if a.constants:
        constants=json.loads((a.constants/"engram_constants.reference.json").read_text())
        module=check(cu.cuModuleLoad(str(a.constants/"dsv41_engram_constants.cubin").encode()))
        outputs=[alloc(len(v),torch.int64) for v in constants.values()]
        launch("dsv41_engram_constants",(len(constants["token_map"])+255)//256,*outputs,module=module)
        for out,values in zip(outputs,constants.values()):assert torch.equal(out,tensor(values,torch.int64))
        print("once Engram constants bit-exact with supplied inference tokenizer/hash outputs")
        if a.inference:
            import math
            tree=ast.parse((a.inference/"model.py").read_text())
            function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="precompute_freqs_cis")
            ns={"torch":torch,"math":math,"lru_cache":__import__("functools").lru_cache}
            exec(compile(ast.Module(body=[function],type_ignores=[]),"official_rope","exec"),ns)
            config=json.loads((a.inference/"config.json").read_text())
            metadata=json.loads((a.constants/"engram_constants.json").read_text())["metadata"]
            n=metadata["max_positions"]
            tables=[alloc((n,64),torch.float32) for _ in range(4)]
            launch("dsv41_rope_constants",(n*32+255)//256,*tables,n,module=module)
            with torch.device("cuda"):
                for family,original,theta,interleaved,split in [
                    ("window",0,config["rope_theta"],tables[0],tables[1]),
                    ("compressed",config["original_seq_len"],config["compress_rope_theta"],tables[2],tables[3])]:
                    ref=ns["precompute_freqs_cis"](64,n,original,theta,config["rope_factor"],config["beta_fast"],config["beta_slow"])
                    expected=torch.view_as_real(ref).flatten(-2)
                    error=(interleaved-expected).abs().max().item()
                    print("rope once",family,"max_abs",error,flush=True)
                    torch.testing.assert_close(interleaved,expected,atol=2e-7,rtol=1e-6)
                    assert torch.equal(split,torch.cat([interleaved[:,::2],interleaved[:,1::2]],dim=-1))



if __name__=="__main__":main()
