"""DeepSelect v1.0.0 pinned ABI, sorted indices and explicit -1 short-row fill."""
import hashlib
from pathlib import Path
REVISION='8e70df71d2a4b0c969ef96dc3b8998efa09a3315'
VARIANTS={('f32', 512): ('_ZN16topk_select_fp3211topk_kernelINS_20TopkSelectKernelFP32I16TopkSelectConfigIfiLb0ELb1ELb0ELj512ELj512ELj1ELj8192ELj4096ELj3ELj512ELj1EEEEEEv14TopkSelectArgsNT_9TmaParamsE', 512, 207872), ('f32', 2048): ('_ZN16topk_select_fp3211topk_kernelINS_20TopkSelectKernelFP32I16TopkSelectConfigIfiLb0ELb1ELb0ELj4096ELj256ELj1ELj4096ELj4096ELj3ELj512ELj1EEEEEEv14TopkSelectArgsNT_9TmaParamsE', 256, 183296), ('bf16', 512): ('_ZN23topk_select_bf16_normal11topk_kernelINS_26TopkSelectKernelBF16NormalI16TopkSelectConfigI13__nv_bfloat16iLb0ELb1ELb0ELj512ELj256ELj2ELj4096ELj4096ELj4ELj512ELj1EEEEEEv14TopkSelectArgsNT_9TmaParamsE', 256, 109568), ('bf16', 2048): ('_ZN23topk_select_bf16_normal11topk_kernelINS_26TopkSelectKernelBF16NormalI16TopkSelectConfigI13__nv_bfloat16iLb0ELb1ELb0ELj4096ELj512ELj1ELj8192ELj4096ELj3ELj512ELj1EEEEEEv14TopkSelectArgsNT_9TmaParamsE', 512, 216064)}


def definitions(cubin,*,rows,rows_max,width,topk=512,dtype='f32'):
    """Params scores[rows,width],end i32[rows],indices i32[rows,topk],rows i32.

    end is a per-row exclusive prefix bound. Width is physical stride and must
    be1024-byte aligned. k512/2048 supported, sorted ascending indices. Short
    rows emit valid indices followed by-1, including empty rows. Ties may select
    any cutoff-equal entries. Does not filter selected -inf: caller must mask
    such candidates after selection. k2048 is upstream correctness-only tier.
    """
    if (dtype,topk) not in VARIANTS:raise ValueError('supported dtype f32/bf16 and k512/2048')
    size=4 if dtype=='f32' else 2
    if width*size%1024 or width>=2**23:raise ValueError('invalid input stride or width')
    symbol,threads,smem=VARIANTS[dtype,topk]
    f=[{'at':0,'param':3},{'at':4,'i32':width},{'at':8,'i32':topk},
       {'at':16,'param':0},{'at':32,'param':2},{'at':48,'param':1},
       {'at':64,'i64':width},{'at':80,'i64':topk},{'at':89,'u8':1},
       {'at':92,'i32':-1},{'at':96,'i32':-8388608},{'at':100,'u8':1},
       {'at':104,'i64':233472}]
    inner=128//size
    t={'param':0,'dtype':dtype,'dims':[inner,width//inner,rows_max],
       'strides':[128,width*size],'box':[inner,512//inner,1],'swizzle':128,'l2_promotion':256}
    cubin=Path(cubin);module='dsv41_deepselect'
    modules={module:{'source':cubin.name,'sha256':hashlib.sha256(cubin.read_bytes()).hexdigest()}}
    op={'params':[f'in buffer<{dtype}>','in buffer<i32>','out buffer<i32>','i32'],
        'impl':{'launches':[{'module':module,'entry':symbol,'grid':[rows,1,1],'block':[threads,1,1],
        'shared_mem':smem,'params':['bytes<120>','bytes<128>'],
        'args':[{'pack':{'size':120,'fields':f}},{'pack':{'size':128,'fields':[{'at':0,'tensormap':t}]}}]}]}}
    return modules,{'dsv41_select':op}
