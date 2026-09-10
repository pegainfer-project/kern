"""FlashMLA head64/512 ABI captured from pinned upstream on SM103a.

BF16 baseline; quantized paged/fused decode is a separate pending op.
"""
import hashlib
from pathlib import Path

REVISION = "4f38f29ef6793c228363e4af5be66d44e81167ba"
ENTRY = '_ZN5sm1007prefill10sparse_fwd6head6422sparse_attn_fwd_kernelINS2_14KernelTemplateIL17SparseAttnFwdMode0ELi512EEENS6_9TmaParamsIN4cute5tupleIJiiiEEENS8_9TiledCopyINS8_9Copy_AtomIJNS8_11Copy_TraitsINS8_13SM90_TMA_LOADEJNS8_1CILi65536EEENS8_12AuxTmaParamsINS9_IJNS8_11ScaledBasisINSF_ILi1EEEJLi1EEEENSI_ISJ_JLi0EEEENSI_ISJ_JLi2EEEEEEERKNS8_6LayoutINS9_IJNSF_ILi64EEESP_SJ_EEESN_EERKNS8_7SwizzleILi3ELi4ELi3EEEEEEEEN7cutlass10bfloat16_tEEEENSO_INS9_IJSJ_NS9_IJNS9_IJNS9_IJSP_SP_EEENSF_ILi8EEEEEEEEEEEENS9_IJNSF_ILi0EEENS9_IJNS9_IJNS9_IJSP_SJ_EEENSF_ILi4096EEEEEEEEEEEEEENS9_IJSP_NSF_ILi512EEEEEEEESA_NSB_INSC_IJNSD_ISE_JNSF_ILi32768EEENSH_ISN_RKNSO_INS9_IJNSF_ILi32EEESP_SJ_EEESN_EERKNSU_ILi2ELi4ELi3EEEEEEEES11_EEENSO_INS9_IJSJ_NS9_IJNS9_IJNS9_IJS1J_SP_EEES18_EEEEEEEEENS9_IJS18_NS9_IJNS9_IJS19_NSF_ILi2048EEEEEEEEEEEEEENS9_IJSP_S18_EEEEESA_NSB_INSC_IJNSD_INS8_14SM90_TMA_STOREEJSG_SY_EEES11_EEENSO_INS9_IJSJ_NS9_IJNS9_IJS13_SJ_EEEEEEEEENS9_IJS18_NS9_IJNS9_IJS19_S18_EEEEEEEEEEES13_EEEEEEv19SparseAttnFwdParamsT0_'


def definitions(cubin, *, rows, rows_max, kv_rows_max, topk=640, scale=512**-0.5):
    """Return modules, ops. Params: Q,KV,indices,sink,lengths,O,rows,kv_rows.

    Q/O bf16[rows,64,512], KV bf16[kv_rows_max,512]; indices i32[rows,topk]
    with -1 invalid; lengths i32[rows] <= topk. topk must be >=128, multiple64.
    Q has already received RoPE; output still needs inverse RoPE.
    rows is a grid Expr (including {mul:[seqs,5]} for DSpark).
    Caller must pad indices to topk; an all-invalid row returns zero.
    """
    if topk < 128 or topk % 64:
        raise ValueError("FlashMLA head64 requires topk>=128 and multiple of64")
    cubin=Path(cubin)
    modules={"dsv41_sparse_h64": {"source":cubin.name,"sha256":hashlib.sha256(cubin.read_bytes()).hexdigest()}}
    params=["in buffer<bf16>","in buffer<bf16>","in buffer<i32>","in buffer<f32>",
            "in buffer<i32>","out buffer<bf16>","i32","i32"]
    f=[{"at":0,"param":6},{"at":4,"param":7}]
    f += [{"at":off,"i32":v} for off,v in [(8,64),(12,1),(16,512),(20,512),(24,topk),
          (80,32768),(84,512),(88,512),(92,512),(96,topk),(100,topk)]]
    f += [{"at":28,"f32":scale},{"at":32,"f32":scale*1.4426950408889634}]
    f += [{"at":off,"param":i} for off,i in [(40,0),(48,1),(56,2),(64,3),(72,4),(104,5)]]
    f += [{"at":112,"scratch":"maxima"},{"at":120,"scratch":"lse"}]
    def tma(off,param,dims,strides,box,l2=128):
        return {"at":off,"tensormap":{"param":param,"dtype":"bf16","dims":dims,
                "strides":strides,"box":box,"swizzle":128,"l2_promotion":l2}}
    t=[]
    for base in (0,768):
        t += [{"at":base,"i32":64},{"at":base+4,"i32":512},{"at":base+8,"param":6}]
    # The unused RoPE descriptor stays zero: D_QK=512 makes HAVE_ROPE=false.
    t += [tma(128,0,[512,64,rows_max],[1024,65536],[64,64,1]),
          tma(896,5,[512,64,rows_max],[1024,65536],[64,64,1]),
          tma(1152,1,[512,kv_rows_max],[1024],[64,1],256)]
    op={"params":params,"impl":{"scratch":{
            "maxima":{"dtype":"f32","shape":[rows_max,64]},
            "lse":{"dtype":"f32","shape":[rows_max,64]}},"launches":[{
            "module":"dsv41_sparse_h64","entry":ENTRY,"grid":[rows,1,1],"block":[384,1,1],
            "shared_mem":222512,"params":["bytes<144>","bytes<1280>"],
            "args":[{"pack":{"size":144,"fields":f}},{"pack":{"size":1280,"fields":t}}]}]}}
    return modules,{"dsv41_sparse_attention":op}
