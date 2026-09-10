"""Reference-layout paged attention lowering for target layers.

Q/KV remain in ordinary head order. The fused variant needs corresponding
load-time Q-B/O-A weight permutations and is selected separately.
"""
from .forward import Lowered, attention_inputs, projection, quantized_projection, align4, scalar, selected
from .programs import buf, call, integer
from .auxiliary.ops import definitions as auxiliary_definitions
from .attention.paged_ops import definitions as paged_definitions
from .attention.fused_ops import definitions as fused_definitions


def forward(pieces, serving, layer, layouts, source, output, *,
            mode, prefix, capacity, pool_tokens, cos_sin,
            cubin_dir, auxiliary_cubin, attention_cubin, compressed=None, fused_cubin=None, fused_cos_sin=None,
            prepare_compressed=None):
    fused = fused_cubin is not None
    permuted = [bool(layouts[layer+".attn."+n].get("permutation")) for n in ("wq_b","wo_a")]
    if permuted != [fused, fused]:
        raise ValueError("Q-B/O-A weight layouts must match the attention kernel")
    if fused and fused_cos_sin is None:
        raise ValueError("fused attention requires split cos/sin constants")
    rows = serving.rows(mode)
    page = serving.layout.page_size
    if pool_tokens % page:
        raise ValueError("physical pool must contain complete pages")
    pre = attention_inputs(pieces,layer,layouts,source,prefix=prefix,
                           rows=rows,capacity=capacity,cubin_dir=cubin_dir,
                           auxiliary_cubin=auxiliary_cubin)
    buffers, calls = dict(pre.buffers), list(pre.calls)
    # Compressor and indexer depend on normalized hidden and Q-LoRA. Their
    # publication/selection must precede attention, while compressor commit is
    # retained by the caller until the speculative acceptance count is known.
    if prepare_compressed is not None:
        if compressed is not None:
            raise ValueError("choose precomputed compressed inputs or a preparation callback")
        lowered, compressed = prepare_compressed(source, prefix+".qr")
        buffers.update(lowered.buffers)
        calls.extend(lowered.calls)
    def rope(name, source, target, heads, inverse):
        names = pieces.add(prefix+"."+name,selected(
            auxiliary_definitions(auxiliary_cubin,rows=rows,heads=heads),"rope"))
        buffers[target] = {"dtype":"bf16","kind":"workspace","shape":[capacity,heads*512]}
        calls.append(call(prefix+"."+layer+"."+name,names["rope"],
                          buf(target),buf(source),buf(cos_sin),buf(mode+".position"),
                          scalar(rows),integer(heads),integer(512),integer(64),integer(inverse)))
    if not fused:
        rope("q_rope",prefix+".q",prefix+".q_rotated",64,0)
    rope("kv_rope",prefix+".kv",prefix+".kv_rotated",1,0)
    layer_id = int(layer.rsplit(".",1)[1])
    calls += serving.window_write(mode,layer_id,prefix+".kv_rotated")
    window = f"{'draft' if mode == 'draft' else 'target'}.window.{layer_id}"
    # Empty extra branch still has ABI arguments, never dereferenced with k=0.
    extra = compressed or {
        "state":window,"indices":mode+".window_indices","lengths":mode+".window_length",
        "ratio":1,"width":0,
    }
    builder = fused_definitions if fused else paged_definitions
    modules,ops = builder(
        fused_cubin if fused else attention_cubin,rows=rows,rows_max=capacity,pages=pool_tokens//page,
        extra_pages=pool_tokens//page,page_size=page,extra_page_size=page//extra["ratio"],
        topk=192 if mode == "draft" else 128,extra_topk=extra["width"])
    entry = "dsv41_fused_attention" if fused else "dsv41_paged_attention"
    if fused:
        # The byte ABI holds packed E8M0 words; expose I32 storage so O-A
        # consumes the same allocation without copying or reinterpreting buffers.
        ops[entry]["params"][-1] = "out buffer<i32>"
    names = pieces.add(prefix+"."+layer,(modules,ops))
    raw = prefix+".attention_raw"
    buffers[raw] = {"dtype":"fp8e4m3" if fused else "bf16","kind":"workspace","shape":[capacity,32768]}
    args = [
        buf(prefix+".q" if fused else prefix+".q_rotated"),{"state":window},
        buf(mode+".window_indices"),buf(mode+".window_length"),
        buf(layer+".attn.attn_sink"),buf(raw),{"state":extra["state"]},
        buf(extra["indices"]),buf(extra["lengths"]),scalar(rows),
    ]
    if fused:
        sf = prefix+".attention_sf"
        buffers[sf] = {"dtype":"i32","kind":"workspace","shape":[8,32,align4(capacity)]}
        args += [buf(mode+".position"),buf(fused_cos_sin),buf(sf)]
    calls.append(call(prefix+"."+layer+".attention",names[entry],*args))
    if fused:
        lowered = quantized_projection(
            pieces,prefix+"."+layer+".wo_a",layouts[layer+".attn.wo_a"],
            raw,sf,prefix+".o_lowrank",rows=rows,capacity=capacity,cubin_dir=cubin_dir)
        buffers.update(lowered.buffers);calls += lowered.calls
    else:
        rope("o_rope",raw,prefix+".o",64,1)
        oa_layout = layouts[layer+".attn.wo_a"]
        if oa_layout.get("dtype") == "bf16":
            from .attention.woa import forward as bf16_oa
            lowered = bf16_oa(pieces,prefix+"."+layer+".wo_a",oa_layout,
                              prefix+".o",prefix+".o_lowrank",rows=rows,capacity=capacity)
        else:
            lowered = projection(pieces,prefix+"."+layer+".wo_a",oa_layout,
                                 prefix+".o",prefix+".o_lowrank",rows=rows,row_capacity=capacity,
                                 workspace=prefix+".wo_a.quant",cubin_dir=cubin_dir)
        buffers.update(lowered.buffers);calls += lowered.calls
    lowered = projection(pieces,prefix+"."+layer+".wo_b",layouts[layer+".attn.wo_b"],
                         prefix+".o_lowrank",output,rows=rows,row_capacity=capacity,
                         workspace=prefix+".wo_b.quant",cubin_dir=cubin_dir)
    buffers.update(lowered.buffers);calls += lowered.calls
    return Lowered(buffers,calls)
