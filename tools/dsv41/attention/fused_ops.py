"""FlashMLA fused Q RoPE + paged attention + inverse O RoPE + MXFP8 cast."""
import hashlib
from pathlib import Path
try:
    from .paged_ops import definitions as paged_definitions
except ImportError:
    from paged_ops import definitions as paged_definitions

ENTRY='_ZN5sm1007prefill34fused_norm_rope_attn_rope_cast_fwd9core_attn10fwd_kernelINS2_6KernelIXtlNS2_6ConfigEL17SparseAttnFwdMode1EL9ModelType2ELS7_3ELj64ELb0EEEEEEEvNT_6ParamsENS9_9TMAParamsENS9_9AuxParamsE'


def definitions(cubin,*,rows,rows_max,pages,extra_pages,topk=128,extra_topk=512,
                scale=512**-0.5,cache_state=True,page_size=64,extra_page_size=None):
    """Paged ABI plus positions i32,cos_sin f32,output_scales u8 (13 params).

    Q is permutation [rows,32,64,16] of logical[rows,64,32,16], flattened.
    Output FP8[rows,8,16,8,32] groups+dimtiles+heads+dim32. Output scales are
    packed UE8M0 as uint32[8,32,align4(rows_max)] (row dimension contiguous).
    Independent power-of-two page sizes; token_positions index cos_sin [positions,64] (cos32 then sin32).
    Both RoPEs cover the last64 logical head dimensions. Qnorm disabled.
    """
    extra_page_size=page_size if extra_page_size is None else extra_page_size
    if page_size&(page_size-1) or extra_page_size&(extra_page_size-1):
        raise ValueError('fused page sizes must be powers of2')
    cubin=Path(cubin)
    _,base=paged_definitions(cubin,rows=rows,rows_max=rows_max,pages=pages,extra_pages=extra_pages,
        topk=topk,extra_topk=extra_topk,page_size=page_size,extra_page_size=extra_page_size,scale=scale,cache_state=cache_state)
    op=base['dsv41_paged_attention']
    op['params'][5]='out buffer<fp8e4m3>'
    op['params'] += ['in buffer<i32>','in buffer<f32>','out buffer<u8>']
    f=op['impl']['launches'][0]['args'][0]['pack']['fields']
    f=[x for x in f if x['at'] not in (0,4,104)]
    f += [{'at':0,'i32':1},{'at':4,'param':9},{'at':300,'f32':1e-6},
          {'at':304,'param':10},{'at':316,'i32':64},{'at':320,'param':11},
          {'at':328,'i32':8},{'at':332,'i32':8},{'at':336,'i32':32},
          {'at':340,'u8':1},{'at':341,'u8':1},{'at':342,'u8':1},
          {'at':344,'param':5},{'at':352,'param':12}]
    scale_rows=(rows_max+3)//4*4
    f += [{'at':360,'i32':scale_rows*32},{'at':364,'i32':scale_rows}]
    def tm(at,param,cols,length,stride,box):
        return {'at':at,'tensormap':{'param':param,'dtype':'u32','dims':[cols,length],
            'strides':[stride],'box':[box,1],'swizzle':0,'l2_promotion':256}}
    t=[tm(128,1,128,0 if cache_state else pages*page_size*528//512,512,132)]
    if extra_topk:t += [tm(896,6,64,0 if cache_state else extra_pages*extra_page_size*288//256,256,68)]
    aux=[]
    for base,page in ((0,page_size),(12,extra_page_size)):
        aux += [{'at':base,'i32':page},{'at':base+4,'i32':-2147483648},{'at':base+8,'i32':page.bit_length()-2}]
    op['impl']['launches']=[{'module':'dsv41_fused_decode','entry':ENTRY,'grid':[rows,1,1],
        'block':[512,1,1],'shared_mem':230400,'params':['bytes<368>','bytes<1152>','bytes<24>'],
        'args':[{'pack':{'size':368,'fields':f}},{'pack':{'size':1152,'fields':t}},{'pack':{'size':24,'fields':aux}}]}]
    modules={'dsv41_fused_decode':{'source':cubin.name,'sha256':hashlib.sha256(cubin.read_bytes()).hexdigest()}}
    return modules,{'dsv41_fused_attention':op}
