"""EP4 fused routed FP4 + shared FP8 MegaMoE, private TMA ABI.

Public first five params: y, cumulative_stats, token_count, slab_peers, rank.
Then routed l1, l1_sf, w1, w1_sf, l2, l2_sf, w2, w2_sf;
then shared x, x_sf, w1, w1_sf, l2, l2_sf, w2, w2_sf.
Slab region pointers are supplied with call offsets from layout.offsets.
"""
from pathlib import Path
import hashlib,json,subprocess

def pieces(experts=384,cubin_dir=None,rows="tokens"):
 assert experts in (384,128)
 root=Path(cubin_dir) if cubin_dir else Path(__file__).resolve().parents[3]/'target/cubins/dsv41'
 lay=json.loads((root/f'moe_layout_{experts}.json').read_text());local=experts//4
 modules={'dsv41_mega_moe':{'source':str((Path(cubin_dir) if cubin_dir else Path('target/cubins/dsv41'))/'mega_moe.cubin'),'sha256':hashlib.sha256((root/'mega_moe.cubin').read_bytes()).hexdigest()}}
 def tm(p,d,dims,strides,box,swizzle=0):return {'pack':{'size':128,'fields':[{'at':0,'tensormap':{'param':p,'dtype':d,'dims':dims,'strides':strides,'box':box,'swizzle':swizzle,'l2_promotion':256}}]}}
 def maps(base,rows,sf_rows,groups,weight_dtype):
  h,i=5120,2304;stride=lambda k:k//2 if weight_dtype=='u4' else k
  return [tm(base,'u8',[h,rows],[h],[128,8],128),tm(base+1,'i32',[sf_rows,h//128],[sf_rows*4],[128,1]),tm(base+2,weight_dtype,[h,groups*2*i],[stride(h)],[128,128],128),tm(base+3,'i32',[2*i,groups*h//128],[2*i*4],[128,1]),tm(base+4,'u8',[i,rows],[i],[64,8],64),tm(base+4,'u8',[i,rows],[i],[128,8],128),tm(base+5,'i32',[sf_rows,i//128],[sf_rows*4],[128,1]),tm(base+6,weight_dtype,[i,groups*h],[stride(i)],[128,128],128),tm(base+7,'i32',[h,groups*i//128],[h*4],[128,1])]
 params=['out buffer<bf16>','inout buffer<i32>','i32','in buffer<u64>','i32']+['inout buffer<u8>','inout buffer<u8>','in buffer<u8>','in buffer<i32>','inout buffer<u8>','inout buffer<u8>','in buffer<u8>','in buffer<i32>']*2
 params[11]='in buffer<i8>';params[19]='in buffer<fp8e4m3>'
 op={'params':params,'impl':{'launches':[{'module':'dsv41_mega_moe','entry':f'dsv41_mega_moe_e{experts}_r4','params':params[:5]+['bytes<128>']*18,'args':[{'param':i} for i in range(5)]+maps(5,lay['ring'],lay['sf_ring'],local,'u4')+maps(13,8192,lay['shared_sf_rows'],1,'u8'),'block':[512,1,1],'grid':[152,1,1],'cluster':[2,1,1],'shared_mem':lay['smem'],'pdl':True}]}}
 modules['dsv41_moe_stage']={'source':str((Path(cubin_dir) if cubin_dir else Path('target/cubins/dsv41'))/'stage.cubin'),'sha256':hashlib.sha256((root/'stage.cubin').read_bytes()).hexdigest()}
 stage={'params':['in buffer<bf16>','out buffer<u8>','out buffer<u8>','i32','i32','i32','i32','out buffer<u8>','i32'],'impl':{'launches':[{'module':'dsv41_moe_stage','entry':'dsv41_mega_quant_x','block':[256,1,1],'grid':[{'ceil_div':[{'mul':[rows,40]},8]},1,1]}]}}
 return modules,{f'dsv41_mega_moe_e{experts}':op,'dsv41_moe_quant_x':stage},lay
