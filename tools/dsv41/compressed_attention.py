"""Track shared KV, index results and deferred commits while lowering layers."""
from .compression import publish
from .indexer import forward as index
from .forward import Lowered


class CompressedAttention:
    def __init__(self, pieces, serving, layouts, *, mode, capacity, pool_tokens,
                 cos_sin, cubin_dir, auxiliary_cubin, dense_cubin, sparse_cubin,
                 select_cubin, candidate_cubin):
        self.pieces, self.serving, self.layouts = pieces, serving, layouts
        self.mode, self.capacity = mode, capacity
        self.pool_tokens, self.cos_sin = pool_tokens, cos_sin
        self.cubin_dir, self.auxiliary_cubin = cubin_dir, auxiliary_cubin
        self.index_cubins = dict(dense_cubin=dense_cubin,sparse_cubin=sparse_cubin,
                                 select_cubin=select_cubin,candidate_cubin=candidate_cubin)
        self.source = None
        self.indices = None
        self.candidates = None
        self.commits = []

    def prepare(self, layer, hidden, qr):
        """Call exactly once per compressed target layer, in model layer order."""
        if layer < 2:
            raise ValueError("window-only layers have no compressed branch")
        buffers, calls = {}, []
        if layer in dict(self.serving.layout.sources):
            self.source = layer
            published = publish(self.pieces,self.serving,layer,hidden,mode=self.mode,
                                capacity=self.capacity,cos_sin=self.cos_sin,
                                auxiliary_cubin=self.auxiliary_cubin)
            buffers.update(published.lowered.buffers)
            calls.extend(published.lowered.calls)
            self.commits.extend(published.commit)
        if self.source is None:
            raise ValueError("compressed KV source must be lowered before consumers")
        ratio = dict(self.serving.layout.sources)[self.source]
        if layer in (2,8,14,20,24,28,32,36):
            calls.extend(self.serving.index_metadata(self.mode,self.source))
            metadata = dict(ends=f"{self.mode}.c{ratio}_end",
                            pages=f"{self.mode}.c{ratio}_page_table",
                            request_ids=f"{self.mode}.index_request_ids",
                            sparse_ends=f"{self.mode}.sparse_ends")
            indexed = index(self.pieces,self.serving,layer,self.source,self.layouts,hidden,qr,
                            mode=self.mode,capacity=self.capacity,pool_tokens=self.pool_tokens,
                            cos_sin=self.cos_sin,metadata=metadata,cubin_dir=self.cubin_dir,
                            auxiliary_cubin=self.auxiliary_cubin,candidates=self.candidates,
                            **self.index_cubins)
            buffers.update(indexed.lowered.buffers)
            calls.extend(indexed.lowered.calls)
            self.indices, self.candidates = indexed.physical, indexed.candidates
        if self.indices is None:
            raise ValueError("index source must be lowered before consumers")
        extra = dict(state=f"compressed.{self.source}",indices=self.indices,
                     lengths=f"{self.mode}.compressed_length",ratio=ratio,width=512)
        return Lowered(buffers,calls), extra
