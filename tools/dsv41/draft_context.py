"""DSpark target taps and accepted-prefix context KV publication.

Capture the materialized HC residual after Engram and before attention. Call
publish after acceptance for verify, or after target forward for plain modes.
All three draft layers consume the same main_norm(main_proj(concat(taps))).
"""
from .auxiliary.ops import definitions
from .forward import Lowered, normalize, projection, scalar, selected
from .programs import buf, call, integer


TARGET_LAYERS = (37, 38, 39)


def hidden_name(mode):
    return f"{mode}.draft_context.taps"


def capture(pieces, materialized_hc, layer, *, mode, rows, capacity, auxiliary_cubin):
    if layer not in TARGET_LAYERS:
        raise ValueError("DSpark taps are target attention inputs37/38/39")
    entry = "context_tap_init" if layer == TARGET_LAYERS[0] else "context_tap"
    op = pieces.add(selected(definitions(auxiliary_cubin, rows=rows), entry))[entry]
    output = hidden_name(mode)
    return Lowered({output: {"dtype": "bf16", "shape": [capacity,15360], "kind": "workspace"}},
                   [call(f"{mode}.context.tap{layer}", op, buf(output), buf(materialized_hc),
                         scalar(rows), integer(TARGET_LAYERS.index(layer)))])


def publish(pieces, serving, layouts, *, mode, capacity, auxiliary_cubin, cubin_dir,
            cos_sin="rope.window.interleaved", accepted="nacc"):
    if mode not in ("prefill", "decode", "verify"):
        raise ValueError("context publication consumes target forward rows")
    rows=serving.rows(mode)
    prefix=f"{mode}.draft_context"
    buffers,calls={},[]
    def append(stage):
        buffers.update(stage.buffers);calls.extend(stage.calls)
    def project(label, source, weight, output):
        append(projection(pieces, label, layouts[weight], source, output, rows=rows,
                          row_capacity=capacity, workspace=label+".quant", cubin_dir=cubin_dir))
    def norm(label, source, weight, output, width):
        append(normalize(pieces,label,source,weight,output,rows=rows,width=width,
                         capacity=capacity,cubin=auxiliary_cubin))
    project(prefix+".main_proj",hidden_name(mode),"mtp.0.main_proj",prefix+".raw")
    norm(prefix+".main_norm",prefix+".raw","mtp.0.main_norm.weight",prefix+".main",5120)
    slots=f"{mode}.window_slot"
    if mode=="verify":
        slots=prefix+".slots"
        buffers[slots]={"dtype":"i64","shape":[capacity],"kind":"workspace"}
        op=pieces.add(selected(definitions(auxiliary_cubin,rows=rows),"context_slots"))["context_slots"]
        calls.append(call(prefix+".accepted_slots",op,buf(slots),buf(f"{mode}.window_slot"),
                          buf(f"{mode}.request"),buf(f"{mode}.starts"),buf(accepted),scalar(rows)))
    ops={name:pieces.add(selected(definitions(auxiliary_cubin,rows=rows,heads=1),name))[name]
         for name in ("rope","cache_fp8")}
    for layer in range(serving.layout.draft_layers):
        stage=f"{prefix}.{layer}"
        project(stage+".wkv",prefix+".main",f"mtp.{layer}.attn.wkv",stage+".raw")
        norm(stage+".norm",stage+".raw",f"mtp.{layer}.attn.kv_norm.weight",stage+".kv",512)
        calls.append(call(stage+".rope",ops["rope"],buf(stage+".kv"),buf(stage+".kv"),
                          buf(cos_sin),buf(f"{mode}.position"),scalar(rows),integer(1),
                          integer(512),integer(64),integer(0)))
        calls.append(call(stage+".publish",ops["cache_fp8"],{"state":f"draft.window.{layer}"},
                          buf(stage+".kv"),buf(slots),scalar(rows),integer(serving.layout.page_size)))
    return Lowered(buffers,calls)
