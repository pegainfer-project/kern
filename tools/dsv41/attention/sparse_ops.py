"""DeepGEMM sparse MXFP4 indexer with private metadata/workspace initialization."""
import hashlib
from pathlib import Path
META='_ZN9deep_gemm5sched17sparse_mqa_logits32sm100_sparse_mqa_logits_metadataILb0ELb0ELj2ELj640ELj8ELj2048ELj0ELj8ELj148ELj256EEEvjjPKjS4_S4_S4_jS4_S4_PhS5_'
SCORE='_ZN9deep_gemm23sm100_sparse_mqa_logitsILj8ELj2ELj5ELj5ELj5ELj148ELj2ELb0ELb1EEEvjP13__nv_bfloat16PKhPKjS4_14CUtensorMap_stS7_S7_S7_S7_'


def definitions(cubin,*,rows,rows_max,kv_rows_max):
    """Params Q,QS,K,KS u8; weights BF16; starts,ends,candidates i32; output BF16; rows,kv_rows i32.

    Q[rows,32,64],QS[rows,32,4],K[keys,64],KS[keys,4]. Candidate rows hold2048
    sorted unique absolute block indices of8 keys, then-1. Valid candidate count
    must be min(ceil((end-start)/8),2048); start is8-aligned. Output[rows,16384]
    uses candidate-slot coordinates: absolute key=candidate[col//8]*8+col%8.
    Output tail is-infinity. Input row capacity even and key capacity multiple128.
    """
    if rows_max%2 or kv_rows_max%128:raise ValueError('pad capacities to2/128')
    splits=((rows_max+1)//2)*52
    metadata_bytes=16+splits*656+((splits+147)//148)*148*16
    params=['in buffer<u8>']*4+['in buffer<bf16>']+['in buffer<i32>']*3+['out buffer<bf16>','i32','i32']
    def tm(param,dtype,dims,strides,box,swizzle=0):
        return {'pack':{'size':128,'fields':[{'at':0,'tensormap':{'param':param,'dtype':dtype,'dims':dims,'strides':strides,'box':box,'swizzle':swizzle,'l2_promotion':256}}]}}
    init={'module':'dsv41_sparse_indexer','entry':'dsv41_sparse_initialize','grid':[{'mul':[rows,64]},1,1],'block':[256,1,1],
          'params':['out buffer<bf16>','out buffer<u8>','i32'],'args':[{'param':8},{'scratch':'workspace'},{'param':9}]}
    meta={'module':'dsv41_sparse_indexer','entry':META,'grid':[min((rows_max+1)//2,592),1,1],'block':[256,1,1],'shared_mem':33408,
          'params':['i32','i32','in buffer<i32>','in buffer<i32>','i64','i64','i32','i64','in buffer<i32>','out buffer<u8>','inout buffer<u8>'],
          'args':[{'param':9},{'param':10},{'param':5},{'param':6},{'i64':0},{'i64':0},{'i32':0},{'i64':0},{'param':7},{'scratch':'metadata'},{'scratch':'workspace'}]}
    score={'module':'dsv41_sparse_indexer','entry':SCORE,'grid':[148,1,1],'block':[896,1,1],'shared_mem':230912,
           'params':['i32','out buffer<bf16>','in buffer<u8>','in buffer<u8>','in buffer<u8>']+['bytes<128>']*5,
           'args':[{'i32':16384},{'param':8},{'param':2},{'param':3},{'scratch':'metadata'},
                   tm(0,'u4packed',[128,rows_max*32],[64],[128,64],64),
                   tm(1,'u32',[32,rows_max],[128],[32,2]),
                   tm(4,'bf16',[32,rows_max],[64],[32,2]),
                   tm(2,'u4packed',[128,kv_rows_max],[64],[128,128],64),
                   tm(3,'u32',[kv_rows_max,1],[kv_rows_max*4],[128,1])]}
    mask={'module':'dsv41_sparse_indexer','entry':'dsv41_sparse_mask','grid':[{'mul':[rows,64]},1,1],
          'block':[256,1,1],'params':['inout buffer<bf16>','in buffer<i32>','in buffer<i32>','i32'],
          'args':[{'param':8},{'param':7},{'param':6},{'param':9}]}
    cubin=Path(cubin)
    modules={'dsv41_sparse_indexer':{'source':cubin.name,'sha256':hashlib.sha256(cubin.read_bytes()).hexdigest()}}
    op={'params':params,'impl':{'scratch':{'metadata':{'dtype':'u8','shape':[metadata_bytes]},'workspace':{'dtype':'u8','shape':[384+rows_max*8]}},'launches':[init,meta,score,mask]}}
    positions={'params':['in buffer<bf16>','in buffer<i32>','in buffer<i32>','in buffer<i32>','out buffer<i32>','i32'],
               'impl':{'launches':[{'module':'dsv41_sparse_indexer','entry':'dsv41_sparse_positions',
                    'grid':[{'mul':[rows,2]},1,1],'block':[256,1,1]}]}}
    return modules,{'dsv41_sparse_scores':op,'dsv41_sparse_positions':positions}
