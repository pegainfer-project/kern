"""Shifted Mega mHC interface using unmodified PR #432 cubin ABI.

Public parameters: x, residual, previous_post, previous_comb, shifted_pre,
fn, scales, bases, rms_weight, new_residual, new_pre, new_post, new_comb,
y_bf16, token_count. All token dimensions are flattened.
"""
from pathlib import Path
import hashlib

def pieces(rows='tokens', cubin_dir=None, max_tokens=8192):
    root=Path(cubin_dir) if cubin_dir else Path(__file__).resolve().parents[3]/'target/cubins/dsv41'
    modules={name:{'source':str((Path(cubin_dir) if cubin_dir else Path('target/cubins/dsv41'))/file),'sha256':hashlib.sha256((root/file).read_bytes()).hexdigest()} for name,file in [('dsv41_mhc','mega_mhc.cubin'),('dsv41_moe_boundary','boundary.cubin')]}
    capacity=rows if isinstance(rows,int) else max_tokens
    params=['in buffer<bf16>','in buffer<bf16>','in buffer<f32>','in buffer<f32>','in buffer<f32>','in buffer<f32>','in buffer<f32>','in buffer<f32>','in buffer<bf16>','out buffer<bf16>','out buffer<f32>','out buffer<f32>','out buffer<f32>','out buffer<bf16>','i32']
    def tm(p,dtype,dims,strides,box,swizzle=128):
        t={'param':p,'dtype':dtype,'dims':dims,'strides':strides,'box':box,'swizzle':swizzle,'l2_promotion':0}
        return {'pack':{'size':128,'fields':[{'at':0,'tensormap':t}]}}
    hidden=lambda p:tm(p,'bf16',[5120,capacity],[10240],[64,64])
    residual=lambda p:tm(p,'bf16',[5120,capacity,4],[40960,10240],[64,64,4])
    coeff=lambda p,n,s:tm(p,'f32',[n,capacity],[n*4],[n,64],s)
    def pack(size,fields): return {'pack':{'size':size,'fields':fields}}
    pointer=lambda at,p:{'at':at,'param':p}
    mix=pack(64,[pointer(0,6),pointer(8,7),pointer(16,10),pointer(24,11),pointer(32,12),{'at':40,'f32':1e-20},{'at':44,'f32':1e-6},{'at':48,'f32':2.},{'at':52,'f32':1e-6},{'at':56,'i32':20}])
    norm=pack(88,[pointer(0,14),pointer(8,8),pointer(16,9),{'at':24,'f32':1e-20},{'at':28,'f32':1.},pointer(32,13)])
    args=[residual(1),hidden(0),tm(5,'tf32',[5120,24,4],[81920,20480],[32,24,4]),coeff(2,4,0),coeff(3,16,64),coeff(4,4,0),residual(9),hidden(13),mix,norm,{'scratch':'partials'},{'scratch':'barriers'}]
    entry='_ZN9deep_gemm19sm100_mega_mhc_implILj5120ELj16ELj152ELb1ELb1ELb0ELj0EEEv14CUtensorMap_stS1_S1_S1_S1_S1_S1_S1_NS_6layout8mega_mhc7MixArgsENS3_8NormArgsEPvPm'
    op={'params':params,'impl':{'scratch':{'partials':{'dtype':'u8','shape':[((max_tokens+63)//64)*106496]},'barriers':{'dtype':'u64','shape':[524288]}},'launches':[{'module':'dsv41_moe_boundary','entry':'dsv41_zero_u64','params':['out buffer<u64>','i32'],'args':[{'scratch':'barriers'},{'i32':524288}],'block':[256,1,1],'grid':[2048,1,1]},{'module':'dsv41_mhc','entry':entry,'params':['bytes<128>']*8+['bytes<64>','bytes<88>','out buffer<u8>','inout buffer<u64>'],'args':args,'block':[768,1,1],'grid':[152,1,1],'shared_mem':227328}]}}
    return modules,{'dsv41_mhc':op}
