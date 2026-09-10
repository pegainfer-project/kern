"""MXFP8 dense GEMM; SF words are packed E8M0, column-major without UTCCP permutation.
Public ABI: out BF16, A FP8, B FP8, A scales I32, B scales I32, M, N, K.
Optional element row strides allow O projection group views without copies.
"""
from pathlib import Path
import hashlib

def tm(p,d,dims,strides,box,sw=0):
 return {'pack':{'size':128,'fields':[{'at':0,'tensormap':{'param':p,'dtype':d,'dims':dims,'strides':strides,'box':box,'swizzle':sw}}]}}

def pieces(rows,n,k,*,a_stride=None,b_stride=None,out_stride=None,sfa_rows=None,sfb_rows=None,cubin_dir=None,max_tokens=8192):
 root=Path(cubin_dir) if cubin_dir else Path(__file__).resolve().parents[3]/'target/cubins/dsv41'
 modules={'dsv41_dense':{'source':str((Path(cubin_dir) if cubin_dir else Path('target/cubins/dsv41'))/'dense.cubin'),'sha256':hashlib.sha256((root/'dense.cubin').read_bytes()).hexdigest()}}
 capacity=rows if isinstance(rows,int) else max_tokens
 a_stride=a_stride or k;b_stride=b_stride or k;out_stride=out_stride or n
 sfa_rows=sfa_rows or ((capacity+3)//4*4);sfb_rows=sfb_rows or n
 entry='_ZN9deep_gemm28sm100_fp8_fp4_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj32ELj32ELj32ELj0ELj0ELj0ELj0ELj64ELj128ELj128ELj1ELj128ELj128ELj128ELj4ELj2ELj128ELj128ELj1ELb1ELj152ELb0ELb0ELNS_8GemmTypeE0ELb0EN7cutlass12float_e4m3_tES6_NS5_10bfloat16_tENS_8epilogue9transform16EpilogueIdentityEEEvPijjjT28_14CUtensorMap_stSD_SD_SD_SD_'
 # SHAPE_M/N/K are the three zeros following K alignment.
 entry=entry.replace('ELj0ELj0ELj0ELj0ELj64','ELj0ELj0ELj0ELj64')
 maps=[tm(1,'u8',[k,capacity],[a_stride],[128,64],128),tm(2,'u8',[k,n],[b_stride],[128,128],128),tm(3,'i32',[sfa_rows,k//128],[sfa_rows*4],[128,1]),tm(4,'i32',[n,k//128],[sfb_rows*4],[128,1]),tm(0,'bf16',[n,capacity],[out_stride*2],[64,64],128)]
 params=['out buffer<bf16>','in buffer<fp8e4m3>','in buffer<fp8e4m3>','in buffer<i32>','in buffer<i32>','i32','i32','i32']
 op={'params':params,'impl':{'launches':[{'module':'dsv41_dense','entry':entry,'params':['i64','i32','i32','i32','bytes<24>']+['bytes<128>']*5,'args':[{'i64':0},{'param':5},{'param':6},{'param':7},{'pack':{'size':24,'fields':[{'at':20,'f32':1.}]}}]+maps,'block':[256,1,1],'grid':[152,1,1],'shared_mem':119808}]}}
 return modules,{'dsv41_dense':op}

def oa_pieces(rows,*,sfa_rows=None,cubin_dir=None,max_tokens=8192):
 """One-launch O-A einsum [T,8,4096] x [8,1024,4096] -> MXFP8 [T,8,1024].

 DeepGEMM's dynamically-scaled epilogue casts D to E4M3 and writes per-32 UE8M0
 scales as I32 words indexed by (batch*N+n)/128 and the row, i.e. exactly the
 [K/128,sf_rows] column-major A-scale layout dsv41_dense reads over the flattened
 8192-wide output. WO_B therefore consumes this output with no cast in between.
 ABI extends the dense one with the scale output: out FP8, A, B, sfA, sfB, M, N, K, outSF.
 """
 modules,ops=pieces(rows,1024,4096,sfa_rows=sfa_rows,cubin_dir=cubin_dir,max_tokens=max_tokens)
 op=ops.pop('dsv41_dense');launch=op['impl']['launches'][0]
 capacity=rows if isinstance(rows,int) else max_tokens;sf_rows=sfa_rows or ((capacity+3)//4*4)
 launch['entry']='_ZN9deep_gemm28sm100_fp8_fp4_gemm_1d1d_implILN4cute4UMMA5MajorE0ELS3_0ELj32ELj32ELj32ELj0ELj0ELj0ELj64ELj128ELj128ELj8ELj128ELj128ELj128ELj4ELj2ELj128ELj128ELj1ELb1ELj152ELb0ELb0ELNS_8GemmTypeE4ELb0EN7cutlass12float_e4m3_tES6_S6_NS_8epilogue9transform24EpilogueDynamicScaledFP8EEEvPijjjT28_14CUtensorMap_stSC_SC_SC_SC_'
 op['params'][0]='out buffer<fp8e4m3>';op['params'].append('out buffer<i32>')
 # EpilogueArgs {sfd, sfd_stride, shape_m, shape_n, alpha}: the epilogue clamps SF
 # writes to the live M and flattens SF columns over (batch, N), so N stays per-group.
 launch['args'][4]={'pack':{'size':24,'fields':[{'at':0,'param':8},{'at':8,'i32':sf_rows},{'at':12,'param':5},{'at':16,'i32':1024},{'at':20,'f32':1.}]}}
 maps=launch['args'][5:]
 for index,strides in [(0,[32768,4096]),(1,[4096,1024*4096])]:
  t=maps[index]['pack']['fields'][0]['tensormap'];t['dims'].append(8);t['box'].append(1);t['strides']=strides
 for index in (2,3):maps[index]['pack']['fields'][0]['tensormap']['dims'][1]*=8
 # FP8 D halves the store element, so the store block widens to the 128-byte swizzle span.
 maps[4]=tm(0,'u8',[1024,capacity,8],[8*1024,1024],[128,64,1],128)
 launch['args']=launch['args'][:5]+maps
 return modules,{'dsv41_oa':op}

def prep_pieces(rows,n,k,*,groups=1,row_group=32,cubin_dir=None):
 """Quant ABI xBF16,outFP8,outSFi32,M,K,x_stride,sf_rows;
 SF output [K/128,align(capacity,4)], column-major logical [M,K/128].
 Pass fixed sf_rows=align(capacity,4), not the live M.
 O-A quant flattens all 8 groups so pass k=32768. Pack ABI outSFi32,
 rawE8M0,N,K,groups,row_group; output [groups,K/128,N].
 """
 root=Path(cubin_dir) if cubin_dir else Path(__file__).resolve().parents[3]/'target/cubins/dsv41';source=Path(cubin_dir) if cubin_dir else Path('target/cubins/dsv41')
 modules={name:{'source':str(source/file),'sha256':hashlib.sha256((root/file).read_bytes()).hexdigest()} for name,file in [('dsv41_moe_stage','stage.cubin'),('dsv41_weight_prep','weight_prep.cubin')]}
 quant={'params':['in buffer<bf16>','out buffer<fp8e4m3>','out buffer<i32>','i32','i32','i32','i32'],'impl':{'launches':[{'module':'dsv41_moe_stage','entry':'dsv41_dense_quant_x','block':[256,1,1],'grid':[{'ceil_div':[{'mul':[{'mul':[{'ceil_div':[rows,4]},4]},k//128]},8]},1,1]}]}}
 pack={'params':['out buffer<i32>','in buffer<fp8e8m0>','i32','i32','i32','i32'],'impl':{'launches':[{'module':'dsv41_weight_prep','entry':'dsv41_pack_dense_sf','block':[256,1,1],'grid':[(groups*n*(k//128)+255)//256,1,1]}]}}
 return modules,{'dsv41_dense_quant':quant,'dsv41_dense_sf_pack':pack}

def layout_pieces(kind,*,cubin_dir=None):
 """Once permutations for fused MLA, preserving the checkpoint FP8 bytes.
 kind='query': Wq_b rows [64,32,16] -> [32,64,16], N32768 K1280.
 kind='output': WO_A K [8,16,32] -> [16,8,32], groups8 N1024 K4096.
 Weight op ABI outFP8,inFP8; SF op outI32,rawE8M0. SF output has the
 regular dense [groups,K/128,N] layout and includes both permutation and
 source row-group expansion. Use instead of the ordinary dense SF pack.
 """
 if kind not in ('query','output'):raise ValueError(kind)
 which=int(kind=='output');n,k,groups=(1024,4096,8) if which else (32768,1280,1)
 root=Path(cubin_dir) if cubin_dir else Path(__file__).resolve().parents[3]/'target/cubins/dsv41'
 source=Path(cubin_dir) if cubin_dir else Path('target/cubins/dsv41');file='fused_prep.cubin'
 modules={'dsv41_fused_prep':{'source':str(source/file),'sha256':hashlib.sha256((root/file).read_bytes()).hexdigest()}}
 def op(entry,types,count):return {'params':types,'impl':{'launches':[{'module':'dsv41_fused_prep','entry':entry,'params':types+['i32'],'args':[{'param':0},{'param':1},{'i32':which}],'block':[256,1,1],'grid':[(count+255)//256,1,1]}]}}
 return modules,{f'dsv41_{kind}_weight_layout':op('dsv41_fused_weight_layout',['out buffer<fp8e4m3>','in buffer<fp8e4m3>'],groups*n*k),f'dsv41_{kind}_scale_layout':op('dsv41_fused_scale_layout',['out buffer<i32>','in buffer<fp8e8m0>'],groups*n*(k//128))}
