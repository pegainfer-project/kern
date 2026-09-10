"""Pure call-list lowering for V4.1 projections and normalization.

Token geometry belongs to each prefill/decode/draft/verify lowering instance,
not to runtime conditionals. Intermediate workspaces can be reused after their
last consumer; weight scale carries are produced by loading.dense_scales.
"""
from dataclasses import dataclass

from .moe import dense
from .programs import buf, call, integer


def view(name, offset=0):
    """A buffer argument, or a byte offset into one whose rows are wider."""
    return {"buf": name, "offset": offset} if offset else {"buf": name}


def scalar(value):
    if isinstance(value, int):
        return integer(value)
    if isinstance(value, str):
        return {"var": value}
    return {"expr": value}


def align4(rows):
    return (rows + 3) // 4 * 4 if isinstance(rows, int) else {"mul": [{"ceil_div": [rows, 4]}, 4]}


@dataclass(frozen=True)
class Lowered:
    buffers: dict
    calls: list


def selected(modules_ops, name):
    modules, ops = modules_ops
    op = ops[name]
    used = {x["module"] for x in op["impl"]["launches"] if "module" in x}
    return {n: modules[n] for n in used}, {name: op}


def quantize(pieces, label, source, workspace, *, rows, width, capacity, cubin_dir=None):
    """BF16 rows -> dynamic MXFP8 values plus column-major packed E8M0 scales, four-row padded.

    One quantization serves every projection that reads the same rows: the
    op depends on the row capacity and width only.
    """
    q, sf = workspace + ".fp8", workspace + ".sf"
    padded = align4(capacity)
    buffers = {
        q: {"dtype": "fp8e4m3", "shape": [capacity, width], "kind": "workspace"},
        sf: {"dtype": "i32", "shape": [width // 128, padded], "kind": "workspace"},
    }
    quant = pieces.add(selected(
        dense.prep_pieces(rows, width, width, cubin_dir=cubin_dir), "dsv41_dense_quant"))
    calls = [call(label, quant["dsv41_dense_quant"],
                  buf(source), buf(q), buf(sf), scalar(rows), integer(width), integer(width), scalar(padded))]
    return Lowered(buffers, calls), (q, sf)


def projection(pieces, label, layout, source, output, *, rows, workspace,
               cubin_dir=None, row_capacity=None):
    """BF16 input -> dynamic MXFP8 -> dense GEMM -> BF16."""
    n, k, groups = (layout[key] for key in ("n", "k", "groups"))
    if groups != 1:
        raise ValueError(f"unsupported dense projection groups: {groups}")
    capacity = rows if isinstance(rows, int) else row_capacity
    if not isinstance(capacity, int) or capacity < 1:
        raise ValueError("dynamic projections require a static row_capacity for TMA")
    quant, (q, sf) = quantize(pieces, label + ".quant", source, workspace,
                              rows=rows, width=k, capacity=capacity, cubin_dir=cubin_dir)
    gemm = quantized_projection(pieces, label + ".gemm", layout, q, sf, output,
                                rows=rows, capacity=capacity, cubin_dir=cubin_dir)
    return Lowered({**quant.buffers, **gemm.buffers}, quant.calls + gemm.calls)

def normalize(pieces, label, source, weight, output, *, rows, width,
              capacity, cubin, epsilon=1e-20):
    """The auxiliary norm kernel implements the reference BF16 RMSNorm."""
    from .auxiliary.ops import definitions
    op = pieces.add(selected(definitions(cubin, rows=rows), "compress1"))["compress1"]
    return Lowered(
        {output: {"dtype": "bf16", "shape": [capacity, width], "kind": "workspace"}},
        [call(label, op, buf(output), buf(source), buf(weight),
              scalar(rows), integer(width), {"f32": epsilon})],
    )


def fp8_input(prefix, capacity):
    """Where Mega mHC stores the attention input: FP8 rows and column-major GEMM scales."""
    q, sf = prefix + ".input.fp8", prefix + ".input.sf"
    return {q: {"dtype": "fp8e4m3", "shape": [capacity, 5120], "kind": "workspace"},
            sf: {"dtype": "i32", "shape": [5120 // 128, align4(capacity)], "kind": "workspace"}}, (q, sf)


def norm_quant(pieces, label, source, weight, output, workspace, *, rows, width,
               capacity, cubin, stride=None, epsilon=1e-20):
    """One launch for `normalize` and the `quantize` of its result.

    The normalized rows stay live because the compressor reads them; `source`
    may be the leading columns of a wider projection output, hence the stride.
    """
    from .auxiliary.ops import definitions
    op = pieces.add(selected(definitions(cubin, rows=rows), "norm_quant"))["norm_quant"]
    q, sf = workspace + ".fp8", workspace + ".sf"
    padded = align4(capacity)
    buffers = {
        output: {"dtype": "bf16", "shape": [capacity, width], "kind": "workspace"},
        q: {"dtype": "fp8e4m3", "shape": [capacity, width], "kind": "workspace"},
        sf: {"dtype": "i32", "shape": [width // 128, padded], "kind": "workspace"},
    }
    calls = [call(label, op, buf(output), buf(q), buf(sf), buf(source), buf(weight),
                  scalar(rows), integer(width), integer(stride or width), scalar(padded),
                  {"f32": epsilon})]
    return Lowered(buffers, calls), (q, sf)


def norm_rope(pieces, label, source, weight, output, cos_sin, positions, *, rows, width,
              rope_width, capacity, cubin, stride=None, offset=0, inverse=0, epsilon=1e-20):
    """One launch for `normalize` and the single-head RoPE of its result.

    Nothing reads the normalized rows on their own, so only the rotated ones
    reach a buffer.
    """
    from .auxiliary.ops import definitions
    op = pieces.add(selected(definitions(cubin, rows=rows), "norm_rope"))["norm_rope"]
    return Lowered(
        {output: {"dtype": "bf16", "shape": [capacity, width], "kind": "workspace"}},
        [call(label, op, buf(output), view(source, offset), buf(weight), buf(cos_sin),
              buf(positions), scalar(rows), integer(width), integer(rope_width),
              integer(stride or width), integer(inverse), {"f32": epsilon})],
    )


def attention_inputs(pieces, layer, layouts, *, prefix, rows, capacity,
                     cubin_dir, auxiliary_cubin, cos_sin, positions):
    """Lower Q-LoRA and shared KV projections up to their RoPE boundary.

    The input is already attention RMS-normalized and MXFP8-quantized by Mega
    mHC (`fp8_input`). Keep full DP-local head dimensions; no TP slicing or
    communication is inserted.
    """
    buffers, calls = {}, []
    def extend(stage):
        buffers.update(stage.buffers)
        calls.extend(stage.calls)
    def label(name):
        return prefix+"."+layer+"."+name
    # wq_a and wkv read the rows Mega mHC already quantized and are one
    # matrix at load: project once, then split the result by column.
    fp8, (q, sf) = fp8_input(prefix, capacity)
    buffers.update(fp8)
    wide = layouts[layer+".attn.wqkv"]
    raw, stride = prefix+".qkv_raw", wide["n"]
    extend(quantized_projection(pieces, label("wqkv.gemm"), wide, q, sf, raw,
                                rows=rows, capacity=capacity, cubin_dir=cubin_dir))
    q_lora = layouts[layer+".attn.wq_b"]["k"]
    stage, (qr, qr_sf) = norm_quant(pieces, label("q_norm"), raw,
                                    layer+".attn.q_norm.weight", prefix+".qr", prefix+".wq_b.quant",
                                    rows=rows, width=q_lora, stride=stride,
                                    capacity=capacity, cubin=auxiliary_cubin)
    extend(stage)
    extend(quantized_projection(pieces, label("wq_b.gemm"), layouts[layer+".attn.wq_b"],
                                qr, qr_sf, prefix+".q", rows=rows, capacity=capacity,
                                cubin_dir=cubin_dir))
    # The shared KV latent carries its interleaved 64-wide RoPE tail inline.
    extend(norm_rope(pieces, label("kv_rope"), raw, layer+".attn.kv_norm.weight",
                     prefix+".kv_rotated", cos_sin, positions, rows=rows,
                     width=stride-q_lora, rope_width=64, stride=stride, offset=q_lora*2,
                     capacity=capacity, cubin=auxiliary_cubin))
    return Lowered(buffers,calls)


def bf16_projection(pieces, label, source, weight, output, *, rows, capacity, n, k, fp32=False):
    """BF16 stored weights, with FP32 output for compressor softmax inputs."""
    dtype = "f32" if fp32 else "bf16"
    name = "bf16_project_f32" if fp32 else "bf16_project"
    entry = "extern:cublas_bf16_tn_f32" if fp32 else "extern:cublaslt_bf16_tn"
    op = {"params":["in buffer<bf16>","in buffer<bf16>",f"out buffer<{dtype}>","i32","i32","i32"],
          "impl":{"launches":[{"entry":entry}]}}
    key = pieces.add(({},{name:op}))[name]
    return Lowered({output:{"dtype":dtype,"kind":"workspace","shape":[capacity,n]}},
                   [call(label,key,buf(source),buf(weight),buf(output),scalar(rows),integer(n),integer(k))])


def quantized_projection(pieces,label,layout,source,scales,output,*,rows,capacity,cubin_dir=None):
    """Consume MXFP8 values and their packed I32 scales directly, emitting BF16.

    A layout carrying `columns` is a view of a wider matrix: its weight and
    scale offsets pick the slice and `columns` is that matrix's N, which the
    packed scales are strided by.
    """
    n,k,groups=(layout[key] for key in ("n","k","groups"))
    if groups!=1:
        raise ValueError("unsupported grouped projection")
    name=pieces.add(dense.pieces(capacity,n,k,sfa_rows=align4(capacity),
                                 sfb_rows=layout.get("columns"),cubin_dir=cubin_dir))["dsv41_dense"]
    return Lowered({output:{"dtype":"bf16","kind":"workspace","shape":[capacity,n]}},
                   [call(label,name,buf(output),buf(source),
                         view(layout["weight"],layout.get("weight_offset",0)),buf(scales),
                         view(layout["scale"],layout.get("scale_offset",0)),
                         scalar(rows),integer(n),integer(k))])


def output_projection(pieces,label,layout,source,scales,output,*,rows,capacity,cubin_dir=None):
    """O-A: the eight-group GEMM whose epilogue already writes MXFP8.

    Returns the lowering and the (values, scales) pair, in the very layout
    `quantized_projection` expects, so WO_B needs no quantization of its own.
    """
    n,k,groups=(layout[key] for key in ("n","k","groups"))
    if (n,k,groups)!=(1024,4096,8):
        raise ValueError("the O-A instance requires eight 1024x4096 matrices")
    name=pieces.add(dense.oa_pieces(capacity,sfa_rows=align4(capacity),cubin_dir=cubin_dir))["dsv41_oa"]
    sf=output+".sf"
    buffers={output:{"dtype":"fp8e4m3","kind":"workspace","shape":[capacity,n*groups]},
             sf:{"dtype":"i32","kind":"workspace","shape":[n*groups//128,align4(capacity)]}}
    return Lowered(buffers,
                   [call(label,name,buf(output),buf(source),buf(layout["weight"]),buf(scales),
                         buf(layout["scale"]),scalar(rows),integer(n),integer(k),buf(sf))]),(output,sf)
