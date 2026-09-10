"""Small score/invalid-index adapters around upstream DeepSelect."""
import hashlib
from pathlib import Path


def definitions(cubin,*,rows,width,block_stride,topk=2048):
    if block_stride<(width+7)//8:raise ValueError('candidate row capacity too small')
    cubin=Path(cubin);module='dsv41_candidate'
    modules={module:{'source':cubin.name,'sha256':hashlib.sha256(cubin.read_bytes()).hexdigest()}}
    score={'params':['in buffer<f32>','in buffer<i32>','out buffer<f32>','out buffer<i32>','i32'],
        'impl':{'launches':[{'module':module,'entry':'dsv41_candidate_scores','grid':[(block_stride+255)//256,rows,1],
            'block':[256,1,1],'params':['in buffer<f32>','in buffer<i32>','out buffer<f32>','out buffer<i32>','i32','i32','i32'],
            'args':[{'param':i} for i in range(5)]+[{'i32':width},{'i32':block_stride}]}]}}
    filt={'params':['in buffer<f32>','in buffer<i32>','out buffer<i32>','i32'],
        'impl':{'launches':[{'module':module,'entry':'dsv41_selection_filter','grid':[(topk+255)//256,rows,1],
            'block':[256,1,1],'params':['in buffer<f32>','in buffer<i32>','out buffer<i32>','i32','i32','i32'],
            'args':[{'param':i} for i in range(4)]+[{'i32':block_stride},{'i32':topk}]}]}}
    return modules,{'dsv41_candidate_scores':score,'dsv41_selection_filter':filt}
