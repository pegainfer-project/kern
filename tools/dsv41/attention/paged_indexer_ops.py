"""Upstream direct paged dense MXFP4 index scoring and projection-weight adapter."""
import hashlib
from pathlib import Path


def definitions(cubin,*,rows,rows_max,kv_rows_max,page_size=64,pages,page_cols,
                cache_state=True,page_stride=None,scale_dtype='fp8e8m0'):
    """Qpacked,QS,weights F32,end,page_table,cache,cache_scales,out F32,rows.

    Q[rows,32,64], QS[rows,32,4] E8M0, weights[rows,32] already scaled1/64.
    Each query has an independent page-table row[page_cols] and end length.
    Cache pages: data[P,64], scales[P,4], trailing padding to512-byte stride.
    cache_scales aliases SAME state at byte offset P*64, not a separate allocation.
    Scores are logical key columns[rows,kv_rows_max], tail -infinity.
    pages is physical allocation capacity, kv_rows_max padded256, rows_max padded4.
    """
    if rows_max%4 or kv_rows_max%256 or page_size not in (64,128):
        raise ValueError('rows capacity4, key capacity256, page64/128 required')
    stride=page_stride or ((page_size*68+511)//512)*512
    if stride<page_size*68 or stride%512:raise ValueError('cache stride must align512')
    def tm(param,dtype,dims,strides,box,swizzle=0):
        return {'pack':{'size':128,'fields':[{'at':0,'tensormap':{'param':param,'dtype':dtype,
            'dims':dims,'strides':strides,'box':box,'swizzle':swizzle,'l2_promotion':256}}]}}
    cache='in state' if cache_state else 'in buffer<u8>'
    params=['in buffer<u8>',f'in buffer<{scale_dtype}>','in buffer<f32>','in buffer<i32>',
            'in buffer<i32>',cache,cache,'out buffer<f32>','i32']
    init={'module':'dsv41_paged_indexer','entry':'dsv41_dense_initialize',
          'grid':[{'mul':[rows,kv_rows_max//256]},1,1],'block':[256,1,1],
          'params':['out buffer<f32>','i32','i32'],
          'args':[{'param':7},{'param':8},{'i32':kv_rows_max}]}
    meta={'module':'dsv41_paged_indexer','entry':'_ZN9deep_gemm5sched31sm100_paged_mqa_logits_metadataILj1ELb0ELb0ELj256ELj148EEEvjjPKjS3_Pj','grid':[1,1,1],
          'block':[1024,1,1],'shared_mem':(rows_max+33)*4,'pdl':True,
          'params':['i32','i32','in buffer<i32>','i64','out buffer<u8>'],
          'args':[{'param':8},{'param':8},{'param':3},{'i64':0},{'scratch':'metadata'}]}
    score={'module':'dsv41_paged_indexer','entry':f'_ZN9deep_gemm22sm100_paged_mqa_logitsILj1ELj32ELj128ELj{page_size}ELb1ELb0ELb0ELj3ELj10ELj256ELj16ELj128ELj256EN7cutlass12float_e2m1_tEffLj2EEEvjjjPKjPT13_S4_S4_S4_14CUtensorMap_stS7_S7_S7_S7_',
           'grid':[148,1,1],'block':[384,1,1],'shared_mem':202240,'pdl':True,
           'params':['i32','i32','i32','in buffer<i32>','out buffer<f32>','in buffer<i32>','i64','in buffer<u8>']+['bytes<128>']*5,
           'args':[{'param':8},{'i32':kv_rows_max},{'i32':page_cols},{'param':3},{'param':7},{'param':4},{'i64':0},{'scratch':'metadata'},
                   tm(0,'u4packed',[128,rows_max*32],[64],[128,128],64),
                   tm(1,'u32',[32,rows_max],[128],[32,4]),
                   tm(5,'u4packed',[128,page_size,0 if cache_state else pages],[64,stride],[128,page_size,1],64),
                   tm(6,'u32',[page_size,0 if cache_state else pages],[stride],[page_size,1]),
                   tm(2,'f32',[32,rows_max],[128],[32,4])]}
    mask={'module':'dsv41_paged_indexer','entry':'dsv41_dense_mask',
          'grid':init['grid'],'block':[256,1,1],
          'params':['inout buffer<f32>','in buffer<i32>','i32','i32'],
          'args':[{'param':7},{'param':3},{'param':8},{'i32':kv_rows_max}]}
    cubin=Path(cubin)
    modules={'dsv41_paged_indexer':{'source':cubin.name,'sha256':hashlib.sha256(cubin.read_bytes()).hexdigest()}}
    op={'params':params,'impl':{'scratch':{'metadata':{'dtype':'u8','shape':[149*8]}},'launches':[init,meta,score,mask]}}
    weights={'params':['in buffer<bf16>','out buffer<bf16>','out buffer<f32>','i32'],
             'impl':{'launches':[{'module':'dsv41_paged_indexer','entry':'dsv41_index_weights',
                    'grid':[{'ceil_div':[{'mul':[rows,32]},256]},1,1],'block':[256,1,1]}]}}
    return modules,{'dsv41_paged_index_scores':op,'dsv41_index_weights':weights}
