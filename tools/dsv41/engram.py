"""Hash-table lookup and gated residual injection.

The tables are one shared host-memory copy (`placement: host`, read through
ATS) or sharded into HBM across the EP group and read through peer arrays;
`shard` says which, see `weights.bindings`.
"""
from .forward import Lowered,projection,scalar
from .programs import buf,call,integer


def inject(pieces,serving,layer,layouts,residual,output,*,mode,capacity,cubin_dir,shard=None):
    prefix=f"{mode}.engram"
    weight=f"layers.{layer}.engram"
    embedded=prefix+".embedding"
    projected=prefix+".kv"
    buffers={embedded:{"dtype":"bf16","kind":"workspace","shape":[capacity,6144]},
             output:{"dtype":"bf16","kind":"workspace","shape":[capacity,20480]}}
    calls=serving.engram_lookup(mode,layer,weight+".embed.weight",weight+".embed.scale",embedded,shard=shard)
    stage=projection(pieces,prefix+f".layer{layer}",layouts[weight+".wkv"],embedded,projected,
                     rows=serving.rows(mode),row_capacity=capacity,workspace=prefix+".quant",
                     cubin_dir=cubin_dir)
    buffers.update(stage.buffers);calls+=stage.calls
    calls.append(call(prefix+f".layer{layer}.inject",f"{mode}.engram_inject",
                      buf(output),buf(residual),buf(projected),buf(weight+".q_weight"),
                      buf(weight+".k_weight"),buf(mode+".mask"),scalar(serving.rows(mode)),
                      integer(4),integer(5120),{"f32":1e-20}))
    return Lowered(buffers,calls)
