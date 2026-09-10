"""Unmodified MegaGate: x, BF16 gate weight, correction bias, indices, weights, n."""
from pathlib import Path
import hashlib

def pieces(rows='tokens', experts=384, cubin_dir=None, max_tokens=8192, raw_outputs=False):
    assert experts in (128,384)
    capacity=rows if isinstance(rows,int) else max_tokens
    topk=6 if experts==384 else 3;groups=3 if experts==384 else 1;sms=150 if experts==384 else 152
    root=Path(cubin_dir) if cubin_dir else Path(__file__).resolve().parents[3]/'target/cubins/dsv41'
    modules={name:{'source':str((Path(cubin_dir) if cubin_dir else Path('target/cubins/dsv41'))/file),'sha256':hashlib.sha256((root/file).read_bytes()).hexdigest()} for name,file in [('dsv41_gate','mega_gate.cubin'),('dsv41_moe_boundary','boundary.cubin')]}
    def tm(p,n,m):return {'pack':{'size':128,'fields':[{'at':0,'tensormap':{'param':p,'dtype':'bf16','dims':[5120,n],'strides':[10240],'box':[64,m],'swizzle':128}}]}}
    ptr=lambda i:{'param':i};nil={'i64':0}
    args=[tm(0,capacity,16),tm(1,experts,experts//groups),ptr(2),nil,nil,nil,nil,nil,ptr(3),nil,ptr(4),{'scratch':'scores'},{'scratch':'barriers'},ptr(5),{'i32':experts},{'i32':0},{'i32':0},{'f32':1.5},{'i32':0},{'i64':0},nil,nil]
    entry=f'_ZN9deep_gemm25sm100_bf16_mega_gate_implILj5120ELj{experts}ELj16ELj4ELj128ELj1ELj1ELj{groups}ELj{sms}ELj{topk}ELj1ELb1ELb0ELb0ELb0ELb1ELb0ELb0ELb0EEEv14CUtensorMap_stS1_PKfS3_PKbS5_PKiS7_PlS8_PfPvSA_jjjjfilS5_S5_'
    abi=['bytes<128>']*2+['in buffer<f32>','i64','i64','i64','i64','i64','out buffer<i64>','i64','out buffer<f32>','out buffer<u8>','inout buffer<u64>']+['i32']*4+['f32','i32','i64','i64','i64']
    op={'params':['in buffer<bf16>','in buffer<bf16>','in buffer<f32>','out buffer<i64>','out buffer<f32>','i32'],'impl':{'scratch':{'scores':{'dtype':'u8','shape':[((max_tokens+15)//16)*16*experts*4]},'barriers':{'dtype':'u64','shape':[((max_tokens+15)//16)*16]}},'launches':[{'module':'dsv41_moe_boundary','entry':'dsv41_zero_u64','params':['out buffer<u64>','i32'],'args':[{'scratch':'barriers'},{'i32':((max_tokens+15)//16)*16}],'block':[256,1,1],'grid':[((max_tokens+15)//16+15)//16,1,1]},{'module':'dsv41_gate','entry':entry,'params':abi,'args':args,'block':[256,1,1],'grid':[sms,1,1],'shared_mem':75776 if experts==384 else 74752}]}}
    if raw_outputs:
        # Direct writes to exported byte slab offsets; pointer ABI is unchanged.
        op['params'][3:5]=['out buffer<u8>']*2
        abi[8]=abi[10]='out buffer<u8>'
    return modules,{f'dsv41_gate_e{experts}':op}
