"""Direct paged MXFP4 index-K sparse scorer, avoiding full index-cache gather."""
try:
    from .sparse_ops import definitions as contiguous_definitions
except ImportError:
    from sparse_ops import definitions as contiguous_definitions


def definitions(cubin,*,rows,rows_max,page_size=64,page_cols,cache_state=True,page_stride=None,scale_dtype="u8"):
    """Qpacked,QS,weights BF16,context i32,pages i32,request_ids i32,candidates i32,cache,out BF16,rows i32.

    Index-K cache stores data[page_size,64] then scales[page_size,4] per page.
    Page table is[rows,page_cols]; request_ids are consecutive and paired queries
    of one request must have equal page tables. Candidates are logical block8
    indices; context/end lengths are logical compressed-token counts.
    Output uses the same16384 candidate-slot columns as contiguous sparse_scores.
    """
    if page_size not in (64,128):raise ValueError('built index-cache pages are64/128')
    page_stride=page_stride or ((page_size*68+511)//512)*512
    if page_stride<page_size*68 or page_stride%512:raise ValueError('page stride must align512')
    modules,base=contiguous_definitions(cubin,rows=rows,rows_max=rows_max,kv_rows_max=128)
    op=base['dsv41_sparse_scores']
    op['params']=['in buffer<u8>',f'in buffer<{scale_dtype}>','in buffer<bf16>','in buffer<i32>',
        'in buffer<i32>','in buffer<i32>','in buffer<i32>',
        'in state' if cache_state else 'in buffer<u8>','out buffer<bf16>','i32']
    splits=rows_max*26
    op['impl']['scratch']['metadata']['shape']=[16+splits*656+((splits+147)//148)*148*16]
    init,meta,score,mask=op['impl']['launches']
    meta['entry']=f'_ZN9deep_gemm5sched17sparse_mqa_logits32sm100_sparse_mqa_logits_metadataILb1ELb0ELj2ELj640ELj8ELj2048ELj{page_size}ELj8ELj148ELj256EEEvjjPKjS4_S4_S4_jS4_S4_PhS5_'
    meta['grid']=[min(rows_max,592),1,1]
    meta['params']=['i32','i32','i64','i64','in buffer<i32>','in buffer<i32>','i32','in buffer<i32>','in buffer<i32>','out buffer<u8>','inout buffer<u8>']
    meta['args']=[{'param':9},{'i32':0},{'i64':0},{'i64':0},{'param':3},{'param':4},{'i32':page_cols},{'param':5},{'param':6},{'scratch':'metadata'},{'scratch':'workspace'}]
    tmas=score['args'][5:8]
    tmas[2]['pack']['fields'][0]['tensormap']['param']=2
    score['entry']=f'_ZN9deep_gemm29sm100_paged_sparse_mqa_logitsILj{page_size}ELj8ELj2ELj5ELj5ELj5ELj148ELj2ELb1EEEvjjP13__nv_bfloat16PKhS4_14CUtensorMap_stS5_S5_'
    score['params']=['i32','i32','out buffer<bf16>',op['params'][7],'in buffer<u8>']+['bytes<128>']*3
    score['args']=[{'i32':16384},{'i32':page_stride},{'param':8},{'param':7},{'scratch':'metadata'}]+tmas
    mask['args']=[{'param':8},{'param':6},{'param':3},{'param':9}]
    return modules,{'dsv41_paged_sparse_scores':op,'dsv41_sparse_positions':base['dsv41_sparse_positions']}
