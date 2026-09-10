"""Forty-layer target forward, shared by prefill, decode and verification."""
from dataclasses import dataclass, replace
from pathlib import Path
import hashlib

from .attention_forward import forward as attention
from .blocks import Blocks
from .compressed_attention import CompressedAttention
from .engram import inject
from .forward import Lowered, normalize, scalar, selected
from .head import definitions as head_ops
from .moe_forward import forward as moe
from .programs import buf, call, integer


@dataclass(frozen=True)
class Target:
    lowered: Lowered
    normalized: str
    commit: list


def forward(pieces, serving, dense_layouts, expert_layouts, *, mode, ids,
            capacity, pool_tokens, constants, cubin_dir, auxiliary_cubin,
            attention_cubin, head_cubin, dense_cubin, sparse_cubin, select_cubin,
            candidate_cubin, capture_tap, fused_cubin=None):
    """Produce all token hidden rows; the caller chooses serving head rows.

    capture_tap(layer,materialized_hc) returns a Lowered stage storing the
    attention-input mean in DSpark's concatenated target context allocation.
    Plain modes commit compressor state here; verify returns deferred commits.
    """
    if mode not in ("prefill", "decode", "verify"):
        raise ValueError("target mode must be prefill, decode or verify")
    rows = serving.rows(mode)
    prefix = mode + ".target"
    buffers, calls = {}, serving.prepare(mode,ids)
    calls += serving.engram_hash(mode,**constants["names"],
                                  compressed_pad_id=constants["metadata"]["compressed_pad_id"])
    embedding = prefix + ".embedding"
    buffers[embedding] = {"dtype":"bf16","kind":"workspace","shape":[capacity,5120]}
    embedding_op = pieces.add(prefix,selected(head_ops(head_cubin,seqs=rows),"head_embedding"))["head_embedding"]
    calls.append(call(prefix+".embed",embedding_op,buf(ids),integer(1),buf("embed.weight"),buf(embedding),integer(5120)))
    blocks = Blocks(pieces,prefix=prefix,rows=rows,capacity=capacity,cubin_dir=cubin_dir)
    state, initial = blocks.initialize(embedding)
    buffers.update(blocks.buffers())
    calls.extend(initial)
    compressed = CompressedAttention(pieces,serving,dense_layouts,mode=mode,capacity=capacity,
                                     pool_tokens=pool_tokens,cos_sin=constants["rope"]["compressed"]["interleaved"],
                                     cubin_dir=cubin_dir,auxiliary_cubin=auxiliary_cubin,
                                     dense_cubin=dense_cubin,sparse_cubin=sparse_cubin,
                                     select_cubin=select_cubin,candidate_cubin=candidate_cubin)

    def extend(stage):
        for name, spec in stage.buffers.items():
            if name in buffers and buffers[name] != spec:
                raise ValueError(f"target workspace shape collision: {name}")
            buffers[name] = spec
        calls.extend(stage.calls)

    for layer_id in range(40):
        layer = f"layers.{layer_id}"
        if layer_id in (1,14):
            state, flush = blocks.materialize(state,prefix+f".{layer}.engram_materialize")
            calls.extend(flush)
            output = prefix + ".engram_residual"
            extend(inject(pieces,serving,layer_id,dense_layouts,state.residual,output,
                          mode=mode,capacity=capacity,cubin_dir=cubin_dir))
            state = replace(state,residual=output)
        if layer_id in (37,38,39):
            state, flush = blocks.materialize(state,prefix+f".{layer}.tap_materialize")
            calls.extend(flush)
            extend(capture_tap(layer_id,state.residual))
        rope = constants["rope"]["window" if layer_id < 2 else "compressed"]

        def attn(source, output):
            return attention(pieces,serving,layer,dense_layouts,source,output,mode=mode,
                             prefix=prefix+".attention",capacity=capacity,pool_tokens=pool_tokens,
                             cos_sin=rope["interleaved"],cubin_dir=cubin_dir,
                             auxiliary_cubin=auxiliary_cubin,attention_cubin=attention_cubin,
                             fused_cubin=fused_cubin,fused_cos_sin=rope["split"] if fused_cubin else None,
                             prepare_compressed=(lambda hidden,qr: compressed.prepare(layer_id,hidden,qr))
                             if layer_id >= 2 else None)

        def ffn(source, output):
            return moe(pieces,layer,expert_layouts,source,output,rows=rows,capacity=capacity,
                       workspace=prefix+".moe",cubin_dir=cubin_dir)

        state, stage = blocks.layer(state,layer,attn,ffn)
        extend(stage)

    collapsed, normalized = prefix+".collapsed", prefix+".head_hidden"
    buffers[collapsed] = {"dtype":"bf16","kind":"workspace","shape":[capacity,5120]}
    calls.extend(blocks.head_input(state,collapsed))
    extend(normalize(pieces,prefix+".head_norm",collapsed,"norm.weight",normalized,
                     rows=rows,width=5120,capacity=capacity,cubin=auxiliary_cubin))
    if mode != "verify":
        calls.extend(compressed.commits)
    return Target(Lowered(buffers,calls),normalized,compressed.commits if mode == "verify" else [])


def head(pieces, hidden, output, *, mode, capacity, vocab, head_cubin, copy_cubin):
    """Prefill projects its last row; decode and verify project every row."""
    prefix = mode + ".target_head"
    rows = 1 if mode == "prefill" else "tokens"
    head_capacity = 1 if mode == "prefill" else capacity
    buffers, calls = {}, []
    if mode == "prefill":
        copy_cubin = Path(copy_cubin)
        modules = {"copy_rows":{"source":copy_cubin.name,
                                 "sha256":hashlib.sha256(copy_cubin.read_bytes()).hexdigest()}}
        ops = {"last_row":{"params":["out buffer<bf16>","in buffer<bf16>","i32","i32","i32"],
                           "impl":{"launches":[{"module":"copy_rows","entry":"kern_last_row_bf16",
                                                  "grid":[1,1,1],"block":[256,1,1]}]}}}
        name = pieces.add(prefix,(modules,ops))["last_row"]
        last = prefix + ".last"
        buffers[last] = {"dtype":"bf16","kind":"workspace","shape":[1,5120]}
        calls.append(call(prefix+".last",name,buf(last),buf(hidden),integer(5120),integer(5120),scalar("tokens")))
        hidden = last
    names = pieces.add(prefix,head_ops(head_cubin,seqs=rows))
    logits = prefix + ".logits"
    buffers[logits] = {"dtype":"f32","kind":"workspace","shape":[head_capacity,vocab]}
    # The serving generator declares output geometry/fill; verification has
    # six predictions per request whereas the plain programs return one.
    calls.append(call(prefix+".project",names["head_project"],buf(hidden),buf("head.weight"),buf(logits),
                      scalar(rows),integer(vocab),integer(5120)))
    calls.append(call(prefix+".sample",names["head_plain"],buf(logits),buf(output),{"i64":vocab},integer(vocab)))
    return Lowered(buffers,calls)
