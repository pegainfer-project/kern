"""Shifted Mega mHC interface using unmodified PR #432 cubin ABI.

Public parameters: x, residual, previous_post, previous_comb, shifted_pre,
fn, scales, bases, rms_weight, new_residual, new_pre, new_post, new_comb,
y_bf16, token_count, split_barriers, then the MXFP8 copy of the normalized
output: `fp8="gemm"` adds y_fp8 and column-major GEMM scales
[hidden/128, align4(capacity)]; `fp8="moe"` adds y_fp8, row-major routed
scales [tokens, hidden/128] and the Mega MoE shared-expert scale pages whose
word stride is `shared_rows`, all three as raw slab regions (u8). FP8 is derived from the rounded BF16, the same
boundary as the standalone quantization kernels. All token dimensions are
flattened. The barriers are a caller-owned buffer zeroed once at load; the
kernel initializes and leaves them consistent per call, as upstream does
with one per-stream allocation.
"""
from pathlib import Path
import hashlib

SPLITS=40  # upstream's pick for <=3 m-blocks on 152 SMs; 16 is its deterministic-mode default
SCRATCH_PER_TASK=6656  # layout::mega_mhc::Workspace::get_num_scratch_bytes(1,1)

def pieces(rows='tokens', cubin_dir=None, max_tokens=8192, *, fp8, shared_rows=None, splits=SPLITS):
    if fp8 not in ('gemm','moe') or (fp8 == 'moe') != (shared_rows is not None):
        raise ValueError('fp8 output is gemm, or moe with the shared scale rows')
    root=Path(cubin_dir) if cubin_dir else Path(__file__).resolve().parents[3]/'target/cubins/dsv41'
    modules={name:{'source':str((Path(cubin_dir) if cubin_dir else Path('target/cubins/dsv41'))/file),'sha256':hashlib.sha256((root/file).read_bytes()).hexdigest()} for name,file in [('dsv41_mhc','mega_mhc.cubin'),('dsv41_moe_boundary','boundary.cubin')]}
    capacity=rows if isinstance(rows,int) else max_tokens
    params=['in buffer<bf16>','in buffer<bf16>','in buffer<f32>','in buffer<f32>','in buffer<f32>','in buffer<f32>','in buffer<f32>','in buffer<f32>','in buffer<bf16>','out buffer<bf16>','out buffer<f32>','out buffer<f32>','out buffer<f32>','out buffer<bf16>','i32','inout buffer<u64>']+(['out buffer<fp8e4m3>','out buffer<i32>'] if fp8 == 'gemm' else ['out buffer<u8>']*3)
    def tm(p,dtype,dims,strides,box,swizzle=128):
        t={'param':p,'dtype':dtype,'dims':dims,'strides':strides,'box':box,'swizzle':swizzle,'l2_promotion':0}
        return {'pack':{'size':128,'fields':[{'at':0,'tensormap':t}]}}
    hidden=lambda p:tm(p,'bf16',[5120,capacity],[10240],[64,64])
    residual=lambda p:tm(p,'bf16',[5120,capacity,4],[40960,10240],[64,64,4])
    coeff=lambda p,n,s:tm(p,'f32',[n,capacity],[n*4],[n,64],s)
    def pack(size,fields): return {'pack':{'size':size,'fields':fields}}
    pointer=lambda at,p:{'at':at,'param':p}
    mix=pack(64,[pointer(0,6),pointer(8,7),pointer(16,10),pointer(24,11),pointer(32,12),{'at':40,'f32':1e-20},{'at':44,'f32':1e-6},{'at':48,'f32':2.},{'at':52,'f32':1e-6},{'at':56,'i32':20}])
    strides=[1,(capacity+3)//4*4,0] if fp8 == 'gemm' else [5120//128,1,shared_rows]
    norm=pack(88,[pointer(0,14),pointer(8,8),pointer(16,9),{'at':24,'f32':1e-20},{'at':28,'f32':1.},pointer(32,13),pointer(40,16),pointer(48,17),{'at':56,'i64':strides[0]},{'at':64,'i64':strides[1]}]+([pointer(72,18),{'at':80,'i64':strides[2]}] if fp8 == 'moe' else []))
    args=[residual(1),hidden(0),tm(5,'tf32',[5120,24,4],[81920,20480],[32,24,4]),coeff(2,4,0),coeff(3,16,64),coeff(4,4,0),residual(9),hidden(13),mix,norm,{'scratch':'partials'},{'param':15}]
    entry=f"_ZN9deep_gemm19sm100_mega_mhc_implILj5120ELj{splits}ELj152ELb1ELb1ELb1ELj{0 if fp8 == 'gemm' else 16}EEEv14CUtensorMap_stS1_S1_S1_S1_S1_S1_S1_NS_6layout8mega_mhc7MixArgsENS3_8NormArgsEPvPm"
    op={'params':params,'impl':{'scratch':{'partials':{'dtype':'u8','shape':[((max_tokens+63)//64)*splits*SCRATCH_PER_TASK]}},'launches':[{'module':'dsv41_mhc','entry':entry,'params':['bytes<128>']*8+['bytes<64>','bytes<88>','out buffer<u8>','inout buffer<u64>'],'args':args,'block':[768,1,1],'grid':[152,1,1],'shared_mem':227328,'pdl':True}]}}
    return modules,{'dsv41_mhc':op}
