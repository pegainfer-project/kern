"""Compressor and index-key publication, including deferred speculative commit."""
from dataclasses import dataclass

from .forward import Lowered,bf16_projection,normalize,scalar,selected
from .auxiliary.ops import definitions
from .programs import buf,call,integer


@dataclass(frozen=True)
class Published:
    lowered: Lowered
    commit: list
    latent: str


def publish(pieces,serving,source,input_hidden,*,mode,capacity,cos_sin,auxiliary_cubin):
    rows=serving.rows(mode)
    ratio=dict(serving.layout.sources)[source]
    prefix=f"{mode}.compress{source}"
    weight=f"layers.{source}.attn.compressor"
    buffers,calls={},[]
    for projection in ("wkv","wgate") if ratio==2 else ("wkv",):
        # Fix the small ratio-one GEMM geometry across prefill/decode/verify.
        # cuBLASLt's M8/M48 choices can differ by one BF16 ULP, cross an FP4
        # rounding threshold, and change greedy verification. A/C are capacity
        # workspaces; only live rows are normalized or published below.
        projection_rows = capacity if ratio == 1 else rows
        stage=bf16_projection(pieces,prefix+"."+projection,input_hidden,
                              weight+"."+projection+".weight",prefix+"."+projection,
                              rows=projection_rows,capacity=capacity,n=512,k=5120,fp32=ratio==2)
        buffers.update(stage.buffers);calls+=stage.calls
    latent=prefix+".latent"
    buffers[latent]={"dtype":"bf16","kind":"workspace","shape":[capacity,512]}
    calls+=serving.compressor(mode,source,prefix+".wkv",prefix+".wgate",weight+".norm.weight",latent)

    # K is projected from the unrotated compressor latent, before KV RoPE.
    index=f"layers.{source}.attn.indexer"
    stage=bf16_projection(pieces,prefix+".index_k",latent,index+".wk.weight",prefix+".key_raw",
                          rows=rows,capacity=capacity,n=128,k=512)
    buffers.update(stage.buffers);calls+=stage.calls
    stage=normalize(pieces,prefix+".index_norm",prefix+".key_raw",index+".k_norm.weight",
                    prefix+".key",rows=rows,width=128,capacity=capacity,cubin=auxiliary_cubin)
    buffers.update(stage.buffers);calls+=stage.calls
    def rope(name,src,width):
        key=pieces.add(selected(definitions(auxiliary_cubin,rows=rows,heads=1),"rope"))["rope"]
        dst=prefix+"."+name
        buffers[dst]={"dtype":"bf16","kind":"workspace","shape":[capacity,width]}
        calls.append(call(prefix+"."+name,key,buf(dst),buf(src),buf(cos_sin),
                          buf(f"{mode}.c{ratio}_position"),scalar(rows),integer(1),
                          integer(width),integer(64),integer(0)))
        return dst
    key=rope("key_rope",prefix+".key",128)
    quant=pieces.add(selected(definitions(auxiliary_cubin,rows=rows),"index_quant"))["index_quant"]
    packed,scales,dequant=(prefix+"."+n for n in ("key_packed","key_scales","key_dequant"))
    for name,dtype,width in ((packed,"u8",64),(scales,"fp8e8m0",4),(dequant,"bf16",128)):
        buffers[name]={"dtype":dtype,"kind":"workspace","shape":[capacity,width]}
    calls.append(call(prefix+".key_quant",quant,buf(packed),buf(scales),buf(dequant),buf(key),
                      scalar(rows),integer(128)))
    calls+=serving.index_write(mode,source,packed,scales)
    rotated=rope("kv_rope",latent,512)
    calls+=serving.compressed_write(mode,source,rotated)
    commit=serving.commit(mode,source,prefix+".wkv",prefix+".wgate","nacc" if mode=="verify" else None)
    return Published(Lowered(buffers,calls),commit,latent)
