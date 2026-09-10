"""Pure call-list lowering for V4.1 projections and normalization.

Token geometry belongs to each prefill/decode/draft/verify lowering instance,
not to runtime conditionals. Intermediate workspaces can be reused after their
last consumer; weight scale carries are produced by loading.dense_scales.
"""
from dataclasses import dataclass

from .moe import dense
from .programs import buf, call, integer


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


def projection(pieces, label, layout, source, output, *, rows, workspace,
               cubin_dir=None, row_capacity=None):
    """BF16 input -> dynamic MXFP8 -> dense / eight-group O-A -> BF16.

    All activation scales use column-major packed E8M0, with four-row padding.
    O-A quantizes the contiguous flattened eight-group input, then the grouped
    kernel consumes matching group slices without a separate transpose.
    """
    n, k, groups = (layout[key] for key in ("n", "k", "groups"))
    if groups not in (1, 8):
        raise ValueError(f"unsupported dense projection groups: {groups}")
    if groups == 8 and (n, k) != (1024, 4096):
        raise ValueError("the O-A instance requires eight 1024x4096 matrices")
    capacity = rows if isinstance(rows, int) else row_capacity
    if not isinstance(capacity, int) or capacity < 1:
        raise ValueError("dynamic projections require a static row_capacity for TMA")
    width = k * groups
    q, sf = workspace + ".fp8", workspace + ".sf"
    padded = align4(capacity)
    buffers = {
        q: {"dtype": "fp8e4m3", "shape": [capacity, width], "kind": "workspace"},
        sf: {"dtype": "i32", "shape": [width // 128, padded], "kind": "workspace"},
        output: {"dtype": "bf16", "shape": [capacity, groups*n], "kind": "workspace"},
    }
    quant = pieces.add(label, selected(
        dense.prep_pieces(rows, n, width, cubin_dir=cubin_dir), "dsv41_dense_quant"))
    implementation = (dense.pieces(capacity, n, k, sfa_rows=padded, cubin_dir=cubin_dir) if groups == 1
                      else dense.oa_pieces(capacity, sfa_rows=padded, cubin_dir=cubin_dir))
    entry = "dsv41_dense" if groups == 1 else "dsv41_oa"
    gemm = pieces.add(label, implementation)
    calls = [
        call(label + ".quant", quant["dsv41_dense_quant"],
             buf(source), buf(q), buf(sf), scalar(rows), integer(width), integer(width), scalar(padded)),
        call(label + ".gemm", gemm[entry],
             buf(output), buf(q), buf(layout["weight"]), buf(sf), buf(layout["scale"]),
             scalar(rows), integer(n), integer(k)),
    ]
    return Lowered(buffers, calls)

def normalize(pieces, label, source, weight, output, *, rows, width,
              capacity, cubin, epsilon=1e-20):
    """The auxiliary norm kernel implements the reference BF16 RMSNorm."""
    from .auxiliary.ops import definitions
    op = pieces.add(label, selected(definitions(cubin, rows=rows), "compress1"))["compress1"]
    return Lowered(
        {output: {"dtype": "bf16", "shape": [capacity, width], "kind": "workspace"}},
        [call(label, op, buf(output), buf(source), buf(weight),
              scalar(rows), integer(width), {"f32": epsilon})],
    )


def attention_inputs(pieces, layer, layouts, source, *, prefix, rows,
                     capacity, cubin_dir, auxiliary_cubin):
    """Lower Q-LoRA and shared KV projections up to their RoPE boundary.

    The input is already attention RMS-normalized by Mega mHC. Keep full
    DP-local head dimensions; no TP slicing or communication is inserted.
    """
    buffers, calls = {}, []
    def project(name, src, dst):
        stage = projection(pieces, prefix+"."+layer+"."+name, layouts[layer+".attn."+name],
                           src, dst, rows=rows, row_capacity=capacity,
                           workspace=prefix+"."+name+".quant", cubin_dir=cubin_dir)
        buffers.update(stage.buffers)
        calls.extend(stage.calls)
    def norm(name, src, dst, width):
        stage = normalize(pieces,prefix+"."+layer+"."+name,src,layer+".attn."+name+".weight",
                          dst,rows=rows,width=width,capacity=capacity,cubin=auxiliary_cubin)
        buffers.update(stage.buffers)
        calls.extend(stage.calls)
    project("wq_a", source, prefix+".qr_raw")
    norm("q_norm", prefix+".qr_raw", prefix+".qr", 1280)
    project("wq_b", prefix+".qr", prefix+".q")
    project("wkv", source, prefix+".kv_raw")
    norm("kv_norm", prefix+".kv_raw", prefix+".kv", 512)
    return Lowered(buffers,calls)


def bf16_projection(pieces, label, source, weight, output, *, rows, capacity, n, k, fp32=False):
    """BF16 stored weights, with FP32 output for compressor softmax inputs."""
    dtype = "f32" if fp32 else "bf16"
    name = "bf16_project_f32" if fp32 else "bf16_project"
    entry = "extern:cublas_bf16_tn_f32" if fp32 else "extern:cublaslt_bf16_tn"
    op = {"params":["in buffer<bf16>","in buffer<bf16>",f"out buffer<{dtype}>","i32","i32","i32"],
          "impl":{"launches":[{"entry":entry}]}}
    key = pieces.add("linear", ({},{name:op}))[name]
    return Lowered({output:{"dtype":dtype,"kind":"workspace","shape":[capacity,n]}},
                   [call(label,key,buf(source),buf(weight),buf(output),scalar(rows),integer(n),integer(k))])


def quantized_projection(pieces,label,layout,source,scales,output,*,rows,capacity,cubin_dir=None):
    """Consume fused attention's FP8 values and packed I32 scales directly."""
    n,k,groups=(layout[key] for key in ("n","k","groups"))
    if groups==8:
        definitions=dense.oa_pieces(capacity,sfa_rows=align4(capacity),cubin_dir=cubin_dir)
        entry="dsv41_oa"
    elif groups==1:
        definitions=dense.pieces(capacity,n,k,sfa_rows=align4(capacity),cubin_dir=cubin_dir)
        entry="dsv41_dense"
    else:
        raise ValueError("unsupported grouped projection")
    name=pieces.add(label,definitions)[entry]
    return Lowered({output:{"dtype":"bf16","kind":"workspace","shape":[capacity,n*groups]}},
                   [call(label,name,buf(output),buf(source),buf(layout["weight"]),buf(scales),
                         buf(layout["scale"]),scalar(rows),integer(n),integer(k))])
