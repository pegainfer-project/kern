"""DeepGEMM MXFP4 MQA scoring, 32 heads of128, original source candidate layer."""
import hashlib
from pathlib import Path

ENTRY='_ZN9deep_gemm16sm100_mqa_logitsILj32ELj128ELb1ELb0ELb1ELb0ELj4ELj256ELj3ELj10ELj148ELj128ELj256EN7cutlass12float_e2m1_tEffLj2EEEvjjjPKjS4_S4_PT13_14CUtensorMap_stS7_S7_S7_S7_'


def definitions(cubin,*,rows,rows_max,kv_rows_max):
    """Return modules,ops. Qpacked,Qscales,Kpacked,Kscales,weights,start,end,scores,rows,kv_rows.

    Q u8[rows,32,64], Qscales u8[rows,32,4] (packed UE8M0), K u8[kv_rows,64],
    Kscales u8[kv_rows,4]; weights f32[rows,32] include attention/head scale;
    start,end i32[rows] hold half-open valid key bounds, scores f32[rows,kv_rows_max].
    rows_max must be padded to4 and kv_rows_max to256 for TMA scale rows.
    This computes dense candidate-source scores and masks invalid positions -inf.
    """
    if rows_max%4 or kv_rows_max%256:
        raise ValueError('pad row capacity to4 and key capacity to256')
    params=['in buffer<u8>']*4+['in buffer<f32>','in buffer<i32>','in buffer<i32>',
                             'out buffer<f32>','i32','i32']
    def desc(param,dtype,dims,strides,box,swizzle=0):
        return {'pack':{'size':128,'fields':[{'at':0,'tensormap':{'param':param,'dtype':dtype,
            'dims':dims,'strides':strides,'box':box,'swizzle':swizzle,'l2_promotion':256}}]}}
    args=[{'param':8},{'param':9},{'i32':kv_rows_max},{'param':5},{'param':6},{'i64':0},{'param':7},
          desc(0,'u4packed',[128,rows_max*32],[64],[128,128],64),
          desc(1,'u32',[32,rows_max],[128],[32,4]),
          desc(2,'u4packed',[128,kv_rows_max],[64],[128,256],64),
          desc(3,'u32',[kv_rows_max,1],[kv_rows_max*4],[256,1]),
          desc(4,'f32',[32,rows_max],[128],[32,4])]
    cubin=Path(cubin)
    modules={'dsv41_indexer':{'source':cubin.name,'sha256':hashlib.sha256(cubin.read_bytes()).hexdigest()}}
    op={'params':params,'impl':{'launches':[{'module':'dsv41_indexer','entry':ENTRY,
        'grid':[148,1,1],'block':[384,1,1],'shared_mem':202240,
        'params':['i32','i32','i32','in buffer<i32>','in buffer<i32>','i64','out buffer<f32>']+['bytes<128>']*5,
        'args':args}]}}
    return modules,{'dsv41_index_scores':op}
