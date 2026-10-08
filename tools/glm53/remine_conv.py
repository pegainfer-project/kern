#!/usr/bin/env python3
"""Re-mine _causal_conv1d_update_kernel with num_cache_lines = 2**31-1 (D4).
ASTSource offline compile for sm_90a; dead pointer args folded as constexpr
None (matches the captured 7-arg TTIR signature); stride_o_seq = 3072 (kern's
compact conv_out, vs the capture's 3336)."""
import os
import sys
import hashlib
import pathlib

SP = os.environ["GLM53_SGLANG_SP"]  # sglang venv site-packages
sys.path.insert(0, SP)

import triton
from triton.compiler.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from sglang.kernels.ops.mamba.causal_conv1d_triton import _causal_conv1d_update_kernel

CAP_TTIR = pathlib.Path(os.environ["GLM53_CONV_TTIR"])  # captured _causal_conv1d_update_kernel ttir
OUT = pathlib.Path(__file__).resolve().parents[2] / "kernels-glm53-handwritten" / "module_conv_generic.cubin"

fn = _causal_conv1d_update_kernel
live_ptrs = {"x_ptr": "*bf16", "w_ptr": "*fp32", "conv_state_ptr": "*bf16",
             "conv_state_indices_ptr": "*i32", "intermediate_conv_window_ptr": "*bf16",
             "o_ptr": "*bf16"}
dead_ptrs = ["bias_ptr", "cache_seqlens_ptr", "num_accept_tokens_ptr",
             "intermediate_state_indices_ptr", "retrieve_next_token_ptr",
             "retrieve_next_sibling_ptr", "retrieve_parent_token_ptr"]

signature = dict(live_ptrs)
signature["batch"] = "i32"
constexprs = {}
for name in dead_ptrs:
    signature[name] = "constexpr"
    constexprs[name] = None
constexprs.update(dict(
    dim=3072, seqlen=1, state_len=3, num_cache_lines=2**31 - 1,
    stride_x_seq=3336, stride_x_dim=1, stride_x_token=1,
    stride_w_dim=4, stride_w_width=1,
    stride_conv_state_seq=9216, stride_conv_state_dim=1, stride_conv_state_tok=3072,
    stride_state_indices=1,
    stride_inter_seq=0, stride_inter_step=0, stride_inter_dim=0, stride_inter_win=0,
    stride_intermediate_state_indices=0,
    stride_retrieve_next_token_seq=0, stride_retrieve_next_token_token=0,
    stride_retrieve_next_sibling_seq=0, stride_retrieve_next_sibling_token=0,
    stride_retrieve_parent_token_seq=0, stride_retrieve_parent_token_token=0,
    stride_o_seq=3072, stride_o_dim=1, stride_o_token=1,
    pad_slot_id=-1,
    HAS_BIAS=False, KERNEL_WIDTH=4, SILU_ACTIVATION=True,
    IS_CONTINUOUS_BATCHING=True, IS_SPEC_DECODING=False,
    NP2_STATELEN=4, NP2_SEQLEN=1, USE_PAD_SLOT=True,
    BLOCK_N=256, SAVE_INTERMEDIATE=False,
    HAS_EAGLE_TREE_CUSTOM_ATTN_MASK=False, USE_GDC=True,
))
for name in constexprs:
    signature[name] = "constexpr"

names = list(fn.arg_names)
attrs = {}
for name in live_ptrs:
    attrs[(names.index(name),)] = [["tt.divisibility", 16]]

src = ASTSource(fn=fn, signature=signature, constexprs=constexprs, attrs=attrs)
cc = triton.compile(src, target=GPUTarget("cuda", 90, 32),
                    options={"num_warps": 4, "num_stages": 3})
OUT.write_bytes(cc.asm["cubin"])
print("cubin:", OUT, "sha256:", hashlib.sha256(OUT.read_bytes()).hexdigest()[:16])

ttir_new = cc.asm["ttir"]
pathlib.Path("/tmp/conv_generic.ttir").write_text(ttir_new)
ttir_old = CAP_TTIR.read_text()
import difflib
diff = [l for l in difflib.unified_diff(ttir_old.splitlines(), ttir_new.splitlines(), lineterm="")
        if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))]
print("ttir diff lines:", len(diff))
for l in diff[:30]:
    print(" ", l[:170])
