"""Paged two-level Indexer call lowering, with shared candidate block indices."""
from dataclasses import dataclass

from .forward import Lowered, projection, bf16_projection, scalar, selected, align4
from .programs import buf, call, integer
from .auxiliary.ops import definitions as auxiliary_ops
from .attention.paged_indexer_ops import definitions as dense_ops
from .attention.paged_sparse_ops import definitions as sparse_ops
from .attention.select_ops import definitions as select_ops
from .attention.candidate_ops import definitions as candidate_ops


@dataclass(frozen=True)
class Indexed:
    lowered: Lowered
    logical: str
    physical: str
    candidates: str | None


def forward(pieces, serving, layer, source, layouts, hidden, qr, *, mode,
            capacity, pool_tokens, cos_sin, metadata, cubin_dir,
            auxiliary_cubin, dense_cubin, sparse_cubin, select_cubin,
            candidate_cubin, candidates=None):
    """metadata names: ends, pages, request_ids (one virtual request per row).

    Source denotes the shared index-K owner, not the current indexer layer.
    Layer20 publishes candidate blocks; later indexers consume that allocation.
    """
    rows = serving.rows(mode)
    capacity = align4(capacity)
    ratio = dict(serving.layout.sources)[source]
    page = serving.layout.page_size // ratio
    width = ((serving.layout.max_context // ratio + 255) // 256) * 256
    prefix = f"{mode}.index{layer}"
    weight = f"layers.{layer}.attn.indexer"
    buffers, calls = {}, []

    def allocate(name, dtype, columns):
        # The full-context score rows are the one workspace that scales with
        # the context; prefill, decode and verify never run concurrently, so
        # the three modes share it (a mode-less name) instead of each holding
        # [capacity, context] of their own.
        shared = name in ("scores", "block_scores")
        name = (f"index{layer}" if shared else prefix) + "." + name
        buffers[name] = {"dtype":dtype,"kind":"workspace","shape":[capacity,columns]}
        return name

    def extend(stage):
        buffers.update(stage.buffers)
        calls.extend(stage.calls)

    query = prefix + ".query"
    extend(projection(pieces,prefix+".query",layouts[weight+".wq_b"],qr,query,
                      rows=rows,row_capacity=capacity,workspace=prefix+".query_quant",cubin_dir=cubin_dir))
    rotated = allocate("rotated","bf16",4096)
    rope = pieces.add(selected(auxiliary_ops(auxiliary_cubin,rows=rows,heads=32),"rope"))["rope"]
    calls.append(call(prefix+".rope",rope,buf(rotated),buf(query),buf(cos_sin),
                      buf(mode+".position"),scalar(rows),integer(32),integer(128),integer(64),integer(0)))
    packed = allocate("packed","u8",2048)
    scales = allocate("scales","fp8e8m0",128)
    dequant = allocate("dequant","bf16",4096)
    quant_rows = {"mul":[rows,32]}
    quant = pieces.add(selected(auxiliary_ops(auxiliary_cubin,rows=quant_rows),"index_quant"))["index_quant"]
    calls.append(call(prefix+".quant",quant,buf(packed),buf(scales),buf(dequant),buf(rotated),scalar(quant_rows),integer(128)))
    raw_weights = prefix + ".weights_raw"
    extend(bf16_projection(pieces,prefix+".weights",hidden,weight+".weights_proj.weight",
                           raw_weights,rows=rows,capacity=capacity,n=32,k=5120))
    dense = dense_ops(dense_cubin,rows=rows,rows_max=capacity,kv_rows_max=width,
                      page_size=page,pages=pool_tokens//serving.layout.page_size,
                      page_cols=serving.layout.pages)
    names = pieces.add(selected(dense,"dsv41_index_weights") if layer > 20 else dense)
    weights_bf16 = allocate("weights_bf16","bf16",32)
    weights_f32 = allocate("weights_f32","f32",32)
    calls.append(call(prefix+".scale_weights",names["dsv41_index_weights"],
                      buf(raw_weights),buf(weights_bf16),buf(weights_f32),scalar(rows)))
    logical = allocate("logical","i32",512)

    def select(scores, ends, output, stride, topk, dtype, label):
        name = pieces.add(select_ops(select_cubin,rows=rows,rows_max=capacity,
                          width=stride,topk=topk,dtype=dtype))["dsv41_select"]
        calls.append(call(prefix+"."+label,name,buf(scores),buf(ends),buf(output),scalar(rows)))

    if layer > 20:
        if candidates is None:
            raise ValueError("later indexers require layer20 candidate blocks")
        sparse = pieces.add(sparse_ops(sparse_cubin,rows=rows,rows_max=capacity,
                            page_size=page,page_cols=serving.layout.pages,scale_dtype="fp8e8m0"))
        scores = allocate("scores","bf16",16384)
        calls.append(call(prefix+".score",sparse["dsv41_paged_sparse_scores"],buf(packed),buf(scales),
                          buf(weights_bf16),buf(metadata["ends"]),buf(metadata["pages"]),
                          buf(metadata["request_ids"]),buf(candidates),{"state":f"index_k.{source}"},buf(scores),scalar(rows)))
        # Sparse scores use candidate-slot coordinates, so selection spans all
        # 16384 columns; sparse_positions removes masked and missing candidates.
        slots = allocate("selected_slots","i32",512)
        select(scores,metadata["sparse_ends"],slots,16384,512,"bf16","select_sparse")
        calls.append(call(prefix+".positions",sparse["dsv41_sparse_positions"],buf(scores),buf(slots),
                          buf(candidates),buf(metadata["ends"]),buf(logical),scalar(rows)))
    else:
        scores = allocate("scores","f32",width)
        cache = {"state":f"index_k.{source}"}
        calls.append(call(prefix+".score",names["dsv41_paged_index_scores"],buf(packed),buf(scales),
                          buf(weights_f32),buf(metadata["ends"]),buf(metadata["pages"]),cache,
                          {"state":f"index_k.{source}","offset":page*64},buf(scores),scalar(rows)))
        select(scores,metadata["ends"],logical,width,512,"f32","select_tokens")
        if layer == 20:
            block_stride = ((width//8+255)//256)*256
            block_scores = allocate("block_scores","f32",block_stride)
            block_ends = allocate("block_ends","i32",1)
            selected_blocks = allocate("selected_blocks","i32",2048)
            candidates = allocate("candidates","i32",2048)
            adapters = pieces.add(candidate_ops(candidate_cubin,rows=rows,width=width,block_stride=block_stride))
            calls.append(call(prefix+".block_scores",adapters["dsv41_candidate_scores"],buf(scores),
                              buf(metadata["ends"]),buf(block_scores),buf(block_ends),scalar(rows)))
            select(block_scores,block_ends,selected_blocks,block_stride,2048,"f32","select_blocks")
            calls.append(call(prefix+".filter_blocks",adapters["dsv41_selection_filter"],buf(block_scores),
                              buf(selected_blocks),buf(candidates),scalar(rows)))
    physical = allocate("physical","i32",512)
    calls += serving.compressed_indices(mode,ratio,logical,physical)
    return Indexed(Lowered(buffers,calls),logical,physical,candidates)
