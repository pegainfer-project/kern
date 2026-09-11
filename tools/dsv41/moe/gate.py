"""Unmodified MegaGate: x, BF16 gate weight, correction bias, indices, weights, n."""
from pathlib import Path
import hashlib

# (split_k, stages, launch SMs, shared memory) per expert count. 384: upstream's heuristic
# pick at small token counts, 24 logical CTAs over six worker groups, stages filling smem
# (`mega_layout.cu`); the 4-stage split-K 1 instance is still in the cubin (LEGACY).
CONFIGS={384:(8,12,144,223232),128:(1,4,152,74752)}
LEGACY={384:(1,4,150,75776),128:(1,4,152,74752)}

def pieces(rows='tokens', experts=384, cubin_dir=None, max_tokens=8192, raw_outputs=False, config=None):
    assert experts in (128,384)
    capacity=rows if isinstance(rows,int) else max_tokens
    split_k,stages,sms,smem=config or CONFIGS[experts]
    topk=6 if experts==384 else 3;groups=3 if experts==384 else 1
    root=Path(cubin_dir) if cubin_dir else Path(__file__).resolve().parents[3]/'target/cubins/dsv41'
    modules={name:{'source':str((Path(cubin_dir) if cubin_dir else Path('target/cubins/dsv41'))/file),'sha256':hashlib.sha256((root/file).read_bytes()).hexdigest()} for name,file in [('dsv41_gate','mega_gate.cubin'),('dsv41_moe_boundary','boundary.cubin')]}
    def tm(p,n,m):return {'pack':{'size':128,'fields':[{'at':0,'tensormap':{'param':p,'dtype':'bf16','dims':[5120,n],'strides':[10240],'box':[64,m],'swizzle':128}}]}}
    ptr=lambda i:{'param':i};nil={'i64':0}
    args=[tm(0,capacity,16),tm(1,experts,experts//groups),ptr(2),nil,nil,nil,nil,nil,ptr(3),nil,ptr(4),{'scratch':'scores'},{'param':6},ptr(5),{'i32':experts},{'i32':0},{'i32':0},{'f32':1.5},{'i32':0},{'i64':0},nil,nil]
    entry=f'_ZN9deep_gemm25sm100_bf16_mega_gate_implILj5120ELj{experts}ELj16ELj{stages}ELj128ELj1ELj{split_k}ELj{groups}ELj{sms}ELj{topk}ELj1ELb1ELb0ELb0ELb0ELb1ELb0ELb0ELb0EEEv14CUtensorMap_stS1_PKfS3_PKbS5_PKiS7_PlS8_PfPvSA_jjjjfilS5_S5_'
    abi=['bytes<128>']*2+['in buffer<f32>','i64','i64','i64','i64','i64','out buffer<i64>','i64','out buffer<f32>','out buffer<u8>','inout buffer<u64>']+['i32']*4+['f32','i32','i64','i64','i64']
    op={'params':['in buffer<bf16>','in buffer<bf16>','in buffer<f32>','out buffer<i64>','out buffer<f32>','i32','inout buffer<u64>'],'impl':{'scratch':{'scores':{'dtype':'u8','shape':[((max_tokens+15)//16)*split_k*16*experts*4]}},'launches':[{'module':'dsv41_gate','entry':entry,'params':abi,'args':args,'block':[256,1,1],'grid':[sms,1,1],'shared_mem':smem,'pdl':True}]}}
    if raw_outputs:
        # Direct writes to exported byte slab offsets; pointer ABI is unchanged.
        op['params'][3:5]=['out buffer<u8>']*2
        abi[8]=abi[10]='out buffer<u8>'
    return modules,{f'dsv41_gate_e{experts}':op}
