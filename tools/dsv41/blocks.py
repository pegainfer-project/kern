"""Shifted mHC block lowering; no model logic enters the runtime."""
from dataclasses import dataclass

from .forward import Lowered, scalar
from .loading import MHC_BARRIERS
from .moe import mhc, ops as boundary
from .programs import buf, call


@dataclass(frozen=True)
class Stream:
    """The final hc_post may be fused into the next sublayer's Mega mHC."""
    residual: str
    update: str
    post: str
    comb: str
    pre: str


class Blocks:
    def __init__(self, pieces, *, prefix, rows, capacity, cubin_dir=None, device_sms=152):
        self.prefix, self.rows, self.capacity = prefix, rows, capacity
        self.boundary = pieces.add(boundary.pieces(rows, cubin_dir))
        self.pieces, self.cubin_dir, self.device_sms = pieces, cubin_dir, device_sms

    def mhc(self, fp8, shared_rows=None):
        """The mHC op storing its normalized output for one consumer kind.

        TMA describes allocated capacity; token_count controls live rows.
        """
        return self.pieces.add(mhc.pieces(self.capacity, self.cubin_dir, max_tokens=self.capacity,
                                          fp8=fp8, shared_rows=shared_rows,
                                          device_sms=self.device_sms))["dsv41_mhc"]

    def name(self, suffix):
        return self.prefix + "." + suffix

    def buffers(self):
        shapes = {"post0": ("f32", 4), "comb0": ("f32", 16), "pre0": ("f32", 4),
                  "materialized": ("bf16", 20480), "normalized": ("bf16", 5120)}
        for bank in range(2):
            for name, dtype, width in (("residual","bf16",20480),("pre","f32",4),
                                       ("post","f32",4),("comb","f32",16)):
                shapes[f"{name}{bank+1}"] = (dtype, width)
        return {self.name(n): {"dtype": dtype, "shape": [self.capacity,width], "kind":"workspace"}
                for n,(dtype,width) in shapes.items()}

    def initialize(self, embedding):
        post, comb, pre = (self.name(n) for n in ("post0","comb0","pre0"))
        residual = self.name("materialized")
        calls = [
            call(self.name("identity"), self.boundary["dsv41_hc_identity"],
                 buf(post),buf(comb),buf(pre),scalar(self.rows)),
            call(self.name("expand"), self.boundary["dsv41_hc_init"],
                 buf(residual),buf(embedding),scalar(self.rows)),
        ]
        return Stream(residual,embedding,post,comb,pre), calls

    def materialize(self, state, label, output=None):
        """Flush only at Engram, target tap or final head boundaries.

        Reset only previous post/comb. The shifted pre from the preceding FFN
        remains live, including when Engram modifies the residual stream.
        """
        output = output or self.name("materialized")
        if state.residual == output:
            # Initial stream already is materialized and uses identity mixing.
            if state.post == self.name("post0"):
                return state, []
            raise ValueError("hc_post output must not alias its residual input")
        calls = [call(label, self.boundary["dsv41_hc_post"],buf(output),buf(state.update),
                      buf(state.residual),buf(state.post),buf(state.comb),scalar(self.rows))]
        return Stream(output,state.update,self.name("post0"),self.name("comb0"),state.pre), calls

    def sublayer(self, state, *, layer, kind, bank, update, fp8):
        """Collapse with previous pre, derive current mixes, apply RMSNorm.

        The normalized rows go out as BF16 and as the consumer's MXFP8 input:
        `fp8` is (q, sf) buffer names for attention, or (slab regions,
        shared_rows) for the MoE. Return pending state whose update is
        produced by the following attention or MoE calls. The caller appends
        those calls before consuming this state.
        """
        if kind not in ("attn","ffn") or bank not in (1,2):
            raise ValueError("sublayer kind/bank")
        residual, pre, post, comb = (self.name(f"{n}{bank}") for n in ("residual","pre","post","comb"))
        if state.residual == residual or state.pre == pre:
            raise ValueError("mHC output bank aliases live input")
        norm = self.name("normalized")
        if kind == "attn":
            op, outputs = self.mhc("gemm"), [buf(name) for name in fp8]
        else:
            regions, shared_rows = fp8
            op, outputs = self.mhc("moe", shared_rows), list(regions)
        calls = [call(f"{self.prefix}.{layer}.{kind}.mhc",op,
                      buf(state.update),buf(state.residual),buf(state.post),buf(state.comb),buf(state.pre),
                      buf(f"{layer}.hc_{kind}_fn"),buf(f"{layer}.hc_{kind}_scale"),
                      buf(f"{layer}.hc_{kind}_base"),buf(f"{layer}.{kind}_norm.weight"),
                      buf(residual),buf(pre),buf(post),buf(comb),buf(norm),scalar(self.rows),buf(MHC_BARRIERS),*outputs)]
        return Stream(residual,update,post,comb,pre), norm, calls

    def head_input(self, state, output):
        state, calls = self.materialize(state,self.name("head.materialize"))
        calls.append(call(self.name("head.collapse"),self.boundary["dsv41_hc_pre"],
                          buf(output),buf(state.residual),buf(state.pre),scalar(self.rows)))
        return calls

    def layer(self, state, layer, attention, feed_forward, *, attention_fp8, ffn_fp8):
        """Compose one full block from concrete sublayer lowering functions.

        Providers receive (normalized_input, output_buffer) and return Lowered;
        their MXFP8 inputs are written by the preceding mHC (see `sublayer`).
        Their calls are inlined in this program. There are no runtime subprograms.
        """
        next_state, normalized, before_attention = self.sublayer(
            state,layer=layer,kind="attn",bank=1,update=self.name("attention_result"),fp8=attention_fp8)
        attn = attention(normalized,next_state.update)
        final_state, normalized, before_ffn = self.sublayer(
            next_state,layer=layer,kind="ffn",bank=2,update=self.name("ffn_result"),fp8=ffn_fp8)
        ffn = feed_forward(normalized,final_state.update)
        buffers = self.buffers()
        for part in (attn.buffers,ffn.buffers):
            for name, spec in part.items():
                if name in buffers and buffers[name] != spec:
                    raise ValueError(f"block workspace shape collision: {name}")
                buffers[name] = spec
        return final_state, Lowered(buffers,before_attention+attn.calls+before_ffn+ffn.calls)
