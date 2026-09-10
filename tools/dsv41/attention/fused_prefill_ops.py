"""Fused prefill counterpart, same Q/output permutation as fused paged decode."""
import hashlib
from pathlib import Path
try:
    from .ops import definitions as plain_definitions
except ImportError:
    from ops import definitions as plain_definitions
ENTRY='_ZN5sm1007prefill34fused_norm_rope_attn_rope_cast_fwd9core_attn10fwd_kernelINS2_6KernelIXtlNS2_6ConfigEL17SparseAttnFwdMode0EL9ModelType1ELS7_1ELj64ELb0EEEEEEEvNT_6ParamsENS9_9TMAParamsENS9_9AuxParamsE'


def definitions(cubin,*,rows,rows_max,kv_rows_max,topk=640,scale=512**-0.5):
    """Plain8 params plus position i32,cos_sin f32,output_scales u8 (11 total).

    Q in dim16/head permuted layout; O FP8 in group/dim32/head layout.
    Scale buffer u8[8,32,align4(rows_max),4], same as fused paged.
    BF16 KV must reconstruct the quantized cache, not pre-quantization values.
    """
    cubin=Path(cubin)
    _,base=plain_definitions(cubin,rows=rows,rows_max=rows_max,kv_rows_max=kv_rows_max,topk=topk,scale=scale)
    op=base['dsv41_sparse_attention'];op['params'][5]='out buffer<fp8e4m3>'
    op['params']+=['in buffer<i32>','in buffer<f32>','out buffer<u8>']
    fields=op['impl']['launches'][0]['args'][0]['pack']['fields']
    scale_rows=(rows_max+3)//4*4
    fields += [{'at':148,'f32':1e-6},{'at':152,'param':8},{'at':164,'i32':64},
               {'at':168,'param':9},{'at':176,'i32':8},{'at':180,'i32':8},{'at':184,'i32':32},
               {'at':188,'u8':1},{'at':189,'u8':1},{'at':190,'u8':1},
               {'at':192,'param':5},{'at':200,'param':10},{'at':208,'i32':scale_rows*32},{'at':212,'i32':scale_rows}]
    tma={'at':0,'tensormap':{'param':1,'dtype':'bf16','dims':[512,kv_rows_max],'strides':[1024],'box':[64,1],'swizzle':128,'l2_promotion':256}}
    op['impl']['launches']=[{'module':'dsv41_fused_prefill','entry':ENTRY,'grid':[rows,1,1],'block':[512,1,1],
        'shared_mem':223232,'params':['bytes<216>','bytes<1152>','bytes<24>'],
        'args':[{'pack':{'size':216,'fields':fields}},{'pack':{'size':1152,'fields':[tma]}},
                {'pack':{'size':24,'fields':[{'at':0,'i32':1},{'at':12,'i32':1}]}}]}]
    modules={'dsv41_fused_prefill':{'source':cubin.name,'sha256':hashlib.sha256(cubin.read_bytes()).hexdigest()}}
    return modules,{'dsv41_fused_prefill':op}
