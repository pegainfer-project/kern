"""Three DSpark blocks and sequential Markov-conditioned greedy predictions."""
from .attention_forward import forward as attention
from .blocks import Blocks
from .forward import Lowered, align4, fp8_input, normalize, scalar, selected
from .head import definitions, markov_calls
from .moe_forward import forward as moe, fp8_input as moe_fp8_input
from .programs import buf, call, integer


def forward(pieces, serving, dense_layouts, expert_layouts, *, max_seqs, pool_tokens,
            constants, cubin_dir, auxiliary_cubin, attention_cubin, fused_cubin, head_cubin, vocab):
    rows = serving.rows("draft")
    capacity = align4(max_seqs * 5)
    prefix = "draft.blocks"
    buffers, calls = {}, serving.prepare("draft","draft_ids")
    embedding = prefix + ".embedding"
    buffers[embedding] = {"kind":"workspace","dtype":"bf16","shape":[capacity,5120]}
    embedding_op = pieces.add(selected(definitions(head_cubin,seqs=rows),"head_embedding"))["head_embedding"]
    calls.append(call(prefix+".embed",embedding_op,buf("draft_ids"),integer(1),buf("embed.weight"),buf(embedding),integer(5120)))
    blocks = Blocks(pieces,prefix=prefix,rows=rows,capacity=capacity,cubin_dir=cubin_dir)
    state, initial = blocks.initialize(embedding)
    buffers.update(blocks.buffers())
    calls.extend(initial)
    for index in range(3):
        layer = f"mtp.{index}"
        def attn(source, output):
            return attention(pieces,serving,layer,dense_layouts,source,output,mode="draft",
                             prefix=prefix+".attention",capacity=capacity,pool_tokens=pool_tokens,
                             cos_sin=constants["rope"]["window"]["interleaved"],cubin_dir=cubin_dir,
                             auxiliary_cubin=auxiliary_cubin,attention_cubin=attention_cubin,
                             fused_cubin=fused_cubin,fused_cos_sin=constants["rope"]["window"]["split"])
        def ffn(source, output):
            return moe(pieces,layer,expert_layouts,source,output,rows=rows,capacity=capacity,
                       workspace=prefix+".moe",experts=128,cubin_dir=cubin_dir)
        state, stage = blocks.layer(state,layer,attn,ffn,
                                    attention_fp8=fp8_input(prefix+".attention",capacity)[1],
                                    ffn_fp8=moe_fp8_input(prefix+".moe",experts=128,cubin_dir=cubin_dir,rows=rows))
        buffers.update(stage.buffers)
        calls.extend(stage.calls)
    collapsed, normalized = prefix+".collapsed", "draft.head_hidden"
    buffers[collapsed] = {"kind":"workspace","dtype":"bf16","shape":[capacity,5120]}
    calls.extend(blocks.head_input(state,collapsed))
    stage = normalize(pieces,prefix+".norm",collapsed,"mtp.2.norm.weight",normalized,
                      rows=rows,width=5120,capacity=capacity,cubin=auxiliary_cubin)
    buffers.update(stage.buffers)
    calls.extend(stage.calls)
    pieces.fixed(definitions(head_cubin))
    for name,dtype,shape in (("draft.logits","f32",[capacity,vocab]),
                             ("draft.markov_embed","bf16",[max_seqs,256]),
                             ("draft.markov_bias","f32",[max_seqs,vocab])):
        buffers[name] = {"kind":"workspace","dtype":dtype,"shape":shape}
    calls.extend(markov_calls(vocab=vocab))
    return Lowered(buffers,calls)
