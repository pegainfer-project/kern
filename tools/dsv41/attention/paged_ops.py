"""Direct FP8 window + FP4 compressed-cache FlashMLA decode, no KV gather.

Pinned upstream non-split head64 kernel. Cache positions encode physical
page*page_size+offset. All row-specific policy is expressed in indices.
"""
import hashlib
from pathlib import Path

ENTRY = '_ZN5sm1006decode6sparse6head6439flash_fwd_splitkv_mla_fp8_sparse_kernelINS2_14KernelTemplateIXtlNS2_6ConfigEL9ModelType2ELS6_3ELb0EEEEENS7_9TmaParamsIN4cute5tupleIJiiiiEEENS9_9TiledCopyINS9_9Copy_AtomIJNS9_11Copy_TraitsINS9_13SM90_TMA_LOADEJNS9_1CILi65536EEENS9_12AuxTmaParamsINSA_IJNS9_11ScaledBasisINSG_ILi1EEEJLi1EEEENSJ_ISK_JLi0EEEENSJ_ISK_JLi2EEEENSJ_ISK_JLi3EEEEEEERKNS9_6LayoutINSA_IJNSG_ILi64EEESR_SK_SK_EEESP_EERKNS9_7SwizzleILi3ELi4ELi3EEEEEEEEN7cutlass10bfloat16_tEEEENSQ_INSA_IJSK_NSA_IJNSA_IJNSA_IJSR_SR_EEENSG_ILi8EEEEEEEEEEEENSA_IJNSG_ILi0EEENSA_IJNSA_IJNSA_IJSR_SK_EEENSG_ILi4096EEEEEEEEEEEEEENSA_IJSR_NSG_ILi512EEEEEEEESB_NSC_INSD_IJNSE_INS9_14SM90_TMA_STOREEJSH_S10_EEES13_EEENSQ_INSA_IJSK_NSA_IJNSA_IJS15_SK_EEEEEEEEENSA_IJS1A_NSA_IJNSA_IJS1B_S1A_EEEEEEEEEEES15_EEEEEEv22SparseAttnDecodeParamsT0_'


def definitions(cubin, *, rows, rows_max, pages, extra_pages, page_size=64,
                topk=128, extra_topk=512, extra_page_size=None, scale=512**-0.5, cache_state=True):
    """Return modules,ops. Params Q,cache,ids,lengths,sink,O,extra,extra_ids,extra_lengths,rows.

    Q/O BF16[rows,64,512], ids i32[rows,topk], lengths i32[rows], sink f32[64].
    Extra indices/lengths follow the same contract. Cache is opaque state:
    per page FP8 data[page_size,512] then E8M0 scale[page_size,16]; compressed
    FP4 data[page_size,256] then E4M3 scale[page_size,32].
    Both widths must be multiples of64; extra_topk may be0. Tests use
    cache_state=False to feed raw bytes as ordinary input buffers.
    No split-KV: suitable for batch decode/verify; low-batch split variant pending.
    """
    if topk<64 or topk%64 or extra_topk%64 or extra_topk<0:
        raise ValueError('index widths must be multiples of64')
    extra_page_size=page_size if extra_page_size is None else extra_page_size
    if page_size%32 or extra_page_size%8:
        raise ValueError('FP8 page size multiple32; FP4 page size multiple8')
    cache='in state' if cache_state else 'in buffer<u8>'
    params=['in buffer<bf16>',cache,'in buffer<i32>','in buffer<i32>','in buffer<f32>',
            'out buffer<bf16>',cache,'in buffer<i32>','in buffer<i32>','i32']
    f=[{'at':0,'param':9}]
    values={4:1,8:64,12:1,16:512,20:512,32:pages,36:page_size,40:topk,44:2,48:3,
            112:extra_pages,116:extra_page_size,120:extra_topk,152:32768,156:32768,160:512,
            164:page_size*528,168:528,172:topk,176:topk,180:64,184:64,
            188:32768,192:32768,196:512,200:extra_page_size*288,204:288,
            208:extra_topk,212:extra_topk}
    f += [{'at':off,'i32':v} for off,v in values.items()]
    f += [{'at':24,'f32':scale},{'at':28,'f32':scale*1.4426950408889634}]
    f += [{'at':off,'param':i} for off,i in [(56,0),(64,1),(72,2),(80,3),(88,4),(104,5),(128,6),(136,7),(144,8)]]
    # Upstream length pointer overrides extra_topk; disabled extra must be null.
    if not extra_topk:
        f=[field for field in f if field['at']!=144]
        f.append({'at':144,'i64':0})
    f += [{'at':96,'scratch':'lse'}]
    t=[]
    for base in (0,384):
        t += [{'at':base,'i32':64},{'at':base+4,'i32':512},{'at':base+8,'i32':1},{'at':base+12,'param':9}]
    def tm(at,param,dtype,dims,strides,box,swizzle):
        return {'at':at,'tensormap':{'param':param,'dtype':dtype,'dims':dims,'strides':strides,'box':box,'swizzle':swizzle,'l2_promotion':128}}
    t += [tm(128,0,'bf16',[512,64,1,rows_max],[1024,65536,65536],[64,64,1,1],128),
          tm(512,5,'bf16',[512,64,1,rows_max],[1024,65536,65536],[64,64,1,1],128),
          tm(896,1,'u32',[128,0 if cache_state else pages*page_size*528//512],[512],[128,1],0)]
    if extra_topk:
        t += [tm(1152,6,'u32',[64,0 if cache_state else extra_pages*extra_page_size*288//256],[256],[72,1],0)]
    cubin=Path(cubin)
    modules={'dsv41_paged_h64':{'source':cubin.name,'sha256':hashlib.sha256(cubin.read_bytes()).hexdigest()}}
    op={'params':params,'impl':{'scratch':{'lse':{'dtype':'f32','shape':[rows_max,64]}},'launches':[{
        'module':'dsv41_paged_h64','entry':ENTRY,'grid':[1,rows,1],'block':[384,1,1],'shared_mem':225280,
        'params':['bytes<296>','bytes<1408>'],'args':[{'pack':{'size':296,'fields':f}},{'pack':{'size':1408,'fields':t}}]}]}}
    return modules,{'dsv41_paged_attention':op}
