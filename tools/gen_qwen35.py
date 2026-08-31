#!/usr/bin/env python3
"""Generate examples/qwen3.8-27b.json from the vLLM capture of Qwen3.8-27B.

Qwen3.8-27B is a Qwen3.5-family hybrid: 64 layers, 48 gated-delta-net
(GDN, "linear attention") layers and 16 full-attention layers (every 4th),
gated attention output, Gemma-style norms, mrope.  bs=1 manifest, two
programs sized by the `tokens` symbol:

- `prefill` (tokens ∈ [1, CHUNK_MAX]): the vLLM chunked-prefill forward —
  KV writes for the attention layers, conv/SSM state carry for the GDN
  layers — *plus* the final norm / lm_head / argmax on the last row.  Unlike
  the qwen3-4b manifests, prefill emits `next_token`: the GDN prefill path
  (FLA chunk kernels) and the decode path (recurrent kernel) are different
  arithmetic, and vLLM runs the last prompt token through the chunk path.
  Putting it through `decode` instead would change bits.  The driver
  (kern-run) sees the output buffer in the program and prefills all prompt
  tokens.
- `decode` (tokens = 1): one token, recurrent GDN kernels, split-KV
  attention, logits + argmax.

Data source: the TRITON_ATTN + `gdn_prefill_backend=triton` capture (the
only flat-ABI kernel set: every kernel is a plain CUDA/Triton launch).  The
generator takes launch geometry, scalar literals and instance identity from
the capture, and asserts the wiring it hand-writes against the captured
pointers (buffer identities between consecutive kernels) and the grid
formulas against four prefill forwards of different lengths.

What the runtime cannot express and had to be handwritten (all bit-exact
against vLLM's own ops, tools/test_kernels_qwen35.py):

- `gemma_rms_norm.cu`: GemmaRMSNorm is a chain of ATen ops in vLLM (pow /
  mean / rsqrt / mul); the kernel reproduces ATen's reduction order.
- `sigmoid_mul.cu`: `attn_out * sigmoid(gate)` (two ATen ops).
- `copy_rows.cu`: the strided z-gate copy (`layer_norm_fwd_kernel` bakes
  `%16` assumptions into its strides, so the [T,16384] view cannot be fed
  directly) and the last-row select for the prefill tail (`tokens-1` is not
  in the manifest expression set).

Triton instances: same-symbol kernels compiled for different
specializations (an int arg == 1, or % 16) have identical param layouts, so
the runtime's layout matching cannot pick one.  Every Triton step is pinned
to a dump module by sha256; the module is chosen by matching the dump
against the Triton cache and reading the `.ttir` signature — the least
specialized instance (most runtime args, fewest divisibility attributes)
among those with the launch's register count.  A generic instance computes
the same values as a specialized one; it just makes fewer assumptions.

State layout (baked into the kernels by vLLM's constexprs):

- GDN: one 3211264-B line per layer = [conv state 10240×3 bf16 | SSM state
  48×128×128 f32 | pad]; line 0 is the null line (kernels skip index <= 0),
  layer i uses line i+1; indices come from a constant table via byte
  offsets.  `bytes_fixed` state (per sequence, not per token) — the one
  schema extension this bring-up needed.
- KV: BLOCK_SIZE = 784 (vLLM aligned it to the mamba page), k/v interleaved
  per head, layers interleaved per page: [page][16][784][4][K|V][256],
  bytes_per_token = 65536.

Usage: python3 tools/gen_qwen35.py [dump_dir] [out.json]
"""

import hashlib
import json
import os
import pathlib
import re
import struct
import subprocess
import sys

# --- model geometry (config.json; asserted against the capture below)
HIDDEN = 5120
LAYERS = 64
FFN = 17408
VOCAB = 248320
HEADS = 24
KV_HEADS = 4
HEAD_DIM = 256
Q_DIM = HEADS * HEAD_DIM                 # 6144
KV_DIM = KV_HEADS * HEAD_DIM             # 1024
QKV_DIM = HEADS * 2 * HEAD_DIM + 2 * KV_DIM   # 14336: per head [q | gate], then k, v
GATE_OFF = HEAD_DIM                      # gate half inside a [q | gate] head pair
ROT_HALF = 32                            # rotary_dim 64
MAX_POS = 8192                           # exported rope table rows
# GDN
GDN_V_HEADS = 48
GDN_K_HEADS = 16
GDN_D = 128
GDN_Q = GDN_K_HEADS * GDN_D              # 2048
GDN_V = GDN_V_HEADS * GDN_D              # 6144
CONV_DIM = 2 * GDN_Q + GDN_V             # 10240 (q, k, v go through the conv)
QKVZ_DIM = CONV_DIM + GDN_V              # 16384 (+ z)
Z_OFF = CONV_DIM                         # z columns start after q|k|v
BA_DIM = 2 * GDN_V_HEADS                 # 96: [b | a]
FLA_CHUNK = 64
CONV_BLOCK_M = 8
POST_CONV_BLOCK = 16
LN_ROWS_PER_BLOCK = 4                    # prefill layer_norm instance
BF16 = 2
ATTN_LAYERS = [i for i in range(LAYERS) if (i + 1) % 4 == 0]
GDN_LAYERS = [i for i in range(LAYERS) if (i + 1) % 4 != 0]
CHUNK_MAX = 2048
NT_MAX = CHUNK_MAX // FLA_CHUNK          # 32 chunks
# attention
BLOCK_SIZE = 784                         # vLLM block_size (constexpr in the kernels)
GEMM8_GRID = 152                         # one persistent CTA per GB300 SM
GEMM8_THREADS = 288                      # 1 producer + 8 compute warps
GEMM8_SMEM = 224128                      # sizeof(Smem) + alignment (gemm8.cu)
BLOCK_Q = 5                              # unified 2D: query rows per block
NUM_SEGMENTS = 16                        # unified 3D: grid.z
BLOCK_TABLE_LEN = 8                      # vLLM block_table_stride
BLOCK_ELEMS_PER_LAYER = BLOCK_SIZE * KV_HEADS * 2 * HEAD_DIM   # 1605632
LAYER_KV_BYTES = BLOCK_ELEMS_PER_LAYER * BF16                  # 3211264
BLOCK_STRIDE = len(ATTN_LAYERS) * BLOCK_ELEMS_PER_LAYER        # elems
KV_BYTES_PER_TOKEN = len(ATTN_LAYERS) * KV_HEADS * 2 * HEAD_DIM * BF16  # 65536
V_BYTE_OFF = HEAD_DIM * BF16             # v = k + 256 elems inside a head
# GDN state lines
CONV_STATE_BYTES = CONV_DIM * 3 * BF16   # 61440
SSM_STATE_BYTES = GDN_V_HEADS * GDN_D * GDN_D * 4   # 3145728
GDN_LINE_BYTES = 3211264                 # conv + ssm + 4096 pad (vLLM page)
GDN_LINES = len(GDN_LAYERS) + 1          # line 0 = null
GDN_STATE_BYTES = GDN_LINES * GDN_LINE_BYTES

# --- DFlash2 speculative decoding (examples/qwen3.8-27b-dflash2.json, --spec)
SPEC_BLOCK = 8                           # draft block: anchor + 7 masks = verify rows
DRAFT_TOKENS = SPEC_BLOCK - 1
MASK_TOKEN = 248070                      # dflash_config.mask_token_id
TAPS = {5: 0, 19: 1, 33: 2, 47: 3, 61: 4}   # target_layer_ids -> fc column block
# GDN state under speculation (vLLM: mamba page aligned to an 832-token
# attention block): per layer 8 pages, one SSM checkpoint per verify row,
# the conv line of page 0 is 10 wide (3 history + 7 drafts).  Kernels read
# the SSM slot / conv offset `num_accepted - 1` of the previous round.
SPEC_PAGE_BYTES = 3407872
SPEC_CONV_BYTES = CONV_DIM * (3 + DRAFT_TOKENS) * BF16   # 204800
SPEC_SSM_OFF = SPEC_CONV_BYTES
SPEC_PAGES = len(GDN_LAYERS) * SPEC_BLOCK + 1            # page 0 = null
SPEC_STATE_BYTES = SPEC_PAGES * SPEC_PAGE_BYTES          # 1.31 GB
# draft geometry (5 Qwen3 layers, non-causal over the block, sliding window 2048)
D_LAYERS = 5
D_HEADS = 32
D_KV_HEADS = 8
D_HEAD_DIM = 128
D_Q = D_HEADS * D_HEAD_DIM               # 4096
D_KV = D_KV_HEADS * D_HEAD_DIM           # 1024
D_QKV = D_Q + 2 * D_KV                   # 6144
D_GROUPS = 320                           # grouped conv: 5120 / conv_group_size 16
D_CONV_PROJ = 2 * 2 * D_GROUPS           # kernel_projection out: (side, tap, group)
SEL_RANK = 256
SEL_K = 16
D_BLOCK = 16                             # draft KV page: constexpr of the borrowed non-causal instance
D_BLOCK_ELEMS_PER_LAYER = D_BLOCK * D_KV_HEADS * 2 * D_HEAD_DIM   # 32768
D_LAYER_KV_BYTES = D_BLOCK_ELEMS_PER_LAYER * BF16                # 65536
D_BLOCK_STRIDE = D_LAYERS * D_BLOCK_ELEMS_PER_LAYER              # elems per page
D_KV_BYTES_PER_TOKEN = D_LAYERS * D_KV_HEADS * 2 * D_HEAD_DIM * BF16   # 20480
D_V_BYTE_OFF = D_HEAD_DIM * BF16
D_TABLE_LEN = MAX_POS // D_BLOCK
D_KV_FLAT = D_LAYERS * 2 * D_KV          # fused context-KV GEMM width 10240
D_ATTN_SCALE = D_HEAD_DIM ** -0.5
SPEC_SYMS = {
    "conv_update_spec": "_causal_conv1d_update_kernel",
    "recurrent_spec": "fused_sigmoid_gating_delta_rule_update_kernel",
    "d_rms_norm": "_ZN4vllm15rms_norm_kernelIN3c108BFloat16ELi8ELi2ELb1EEEvPT_PKS3_lllllS6_lfii",
    "d_head_norm": "_ZN4vllm15rms_norm_kernelIN3c108BFloat16ELi8ELi3ELb1EEEvPT_PKS3_lllllS6_lfii",
    "d_fused_norm": "_ZN4vllm25fused_add_rms_norm_kernelIN3c108BFloat16ELi8ELb1EEENSt9enable_ifIXaagtT0_Li0Esr4vllm12_typeConvertIT_EE6existsEvE4typeEPS4_lS7_PKS4_fiil",
    "d_rope": "_ZN4vllm23rotary_embedding_kernelIN3c108BFloat16ES2_Lb1EEEvPKlPT_S6_PKT0_illliiilb",
    "d_cache": "_ZN4vllm30reshape_and_cache_flash_kernelI13__nv_bfloat16S1_LNS_18Fp8KVCacheDataTypeE0EEEvPKT_S5_PT0_S7_PKlllllliiiPKfSB_i",
}

TRITON = {
    "conv_fwd": "_causal_conv1d_fwd_kernel",
    "conv_update": "_causal_conv1d_update_kernel",
    "post_conv": "_fused_post_conv_kernel",
    "cumsum": "chunk_local_cumsum_scalar_kernel",
    "kkt": "chunk_scaled_dot_kkt_fwd_kernel",
    "solve_tril": "merge_16x16_to_64x64_inverse_kernel",
    "recompute": "recompute_w_u_fwd_kernel",
    "chunk_h": "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
    "chunk_o": "chunk_fwd_kernel_o",
    "recurrent": "fused_recurrent_gated_delta_rule_packed_decode_kernel",
    "layer_norm": "layer_norm_fwd_kernel",
    "mrope": "_triton_mrope_forward",
    "cache": "reshape_and_cache_kernel_flash",
    "unified": "kernel_unified_attention",
    "reduce": "reduce_segments",
}


# ----------------------------------------------------------------- capture
def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def forwards(recs):
    """Forward = launches up to and including the sampler's ArgMax reduce."""
    out, start = [], 0
    for i, r in enumerate(recs):
        if "ArgMaxOps" in r["symbol"]:
            out.append(recs[start:i + 1])
            start = i + 1
    return out


def tokens_of(fwd):
    for r in fwd:
        if r["symbol"] == TRITON["mrope"]:
            return r["grid"][0]
    return None


def group(fwd):
    by = {k: [] for k in TRITON}
    by["silu"] = []
    for r in fwd:
        s = r["symbol"]
        if not isinstance(r.get("params"), list):
            continue
        if "act_and_mul" in s:
            by["silu"].append(r)
            continue
        for tag, sym in TRITON.items():
            if s == sym:
                by[tag].append(r)
                break
    return by


def spec_launches(recs):
    """One bs=1 launch record per speculative-path kernel (tools/capture_qwen38_spec.sh
    dump): the IS_SPEC_DECODING conv update (13 params, batch 1), the T=8
    recurrent kernel (N=1), and the draft's vLLM CUDA kernels."""
    want = {
        "conv_update_spec": lambda r: len(r["params"]) == 13 and pv(r, 7) == 1,
        "recurrent_spec": lambda r: r["grid"][2] == GDN_V_HEADS and pv(r, 16) == 1 and pv(r, 17) == SPEC_BLOCK,
        "d_rms_norm": lambda r: r["block"][0] == HIDDEN // 8,
        "d_head_norm": lambda r: pv(r, 2) == D_HEAD_DIM,
        "d_fused_norm": lambda r: pv(r, 6) == HIDDEN,
        "d_rope": lambda r: pv(r, 4) == D_HEAD_DIM,
        "d_cache": lambda r: pv(r, 10) == D_KV_HEADS,
    }
    out = {}
    for r in recs:
        for tag, sym in SPEC_SYMS.items():
            if tag not in out and r["symbol"] == sym and isinstance(r.get("params"), list) and want[tag](r):
                out[tag] = r
    missing = [t for t in SPEC_SYMS if t not in out]
    assert not missing, f"spec capture lacks launches for {missing}"
    return out


def pv(rec, i):
    return int.from_bytes(bytes.fromhex(rec["params"][i]["data"]), "little")


def pf(rec, i):
    return struct.unpack("<f", bytes.fromhex(rec["params"][i]["data"]))[0]


def cdiv_i(a, b):
    return -(-a // b)


# ------------------------------------------------------ instance selection
def cuobjdump():
    return str(pathlib.Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin" / "cuobjdump")


def module_functions(mod):
    """{function: regs} for a cubin."""
    out = subprocess.run([cuobjdump(), "-res-usage", str(mod)],
                         capture_output=True, text=True).stdout
    fns, cur = {}, None
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("Function "):
            cur = s.split()[1].rstrip(":")
        elif cur and "REG:" in s:
            fns[cur] = int(s.split("REG:")[1].split()[0])
            cur = None
    return fns


def triton_cache_by_sha():
    root = pathlib.Path(os.environ.get("TRITON_CACHE_DIR", pathlib.Path.home() / ".triton" / "cache"))
    idx = {}
    for cb in root.glob("*/*.cubin"):
        idx[hashlib.sha256(cb.read_bytes()).hexdigest()] = cb
    return idx


def ttir_signature(ttir_path, symbol):
    """(runtime int args, int args carrying specialization attributes)."""
    text = ttir_path.read_text()
    m = re.search(r"tt\.func public @%s\((.*?)\)\s*attributes" % re.escape(symbol), text, re.S)
    assert m, f"no tt.func signature in {ttir_path}"
    sig = re.sub(r'loc\("[^"]*"\([^)]*\)\)', "", m.group(1))
    sig = re.sub(r"loc\([^)]*\)", "", sig)
    ints, attrs, ptrs = 0, 0, 0
    for arg in re.split(r",\s*(?=%)", sig):
        arg = arg.strip()
        if not arg:
            continue
        if "!tt.ptr" in arg:
            ptrs += 1
            continue
        ints += 1
        if "{" in arg:
            attrs += 1
    return ints, attrs, ptrs


class Pinner:
    """Pick the dump module for a Triton symbol: register count from the
    launch narrows to the constexpr instance, the Triton cache's .ttir picks
    the least specialized runtime-arg instance among the survivors."""

    def __init__(self, dump_dir):
        self.dump = pathlib.Path(dump_dir)
        self.mods = {}
        for mod in sorted(self.dump.glob("module_*.cubin")):
            fns = module_functions(mod)
            if fns:
                self.mods[mod] = fns
        self.shas = {m: hashlib.sha256(m.read_bytes()).hexdigest() for m in self.mods}
        self.cache = triton_cache_by_sha()

    def pin(self, symbol, regs, nparams=None):
        """`nparams` (the launch's param count) separates instances whose
        constexpr choice changes the pointer list (IS_SPEC_DECODING adds two
        pointers to the conv update) but not the register count."""
        cands = {}
        for mod, fns in self.mods.items():
            if fns.get(symbol) == regs:
                cands.setdefault(self.shas[mod], mod)  # dedupe repeated loads
        assert cands, f"{symbol} REG={regs}: no dump module"
        if len(cands) == 1:
            sha, mod = next(iter(cands.items()))
            return mod.name, sha
        scored = []
        for sha, mod in cands.items():
            cb = self.cache.get(sha)
            assert cb, f"{symbol}: dump module {mod.name} not in the Triton cache, cannot read its signature"
            ints, attrs, ptrs = ttir_signature(cb.with_suffix(".ttir"), symbol)
            if nparams is not None and ints + ptrs != nparams - 2:   # minus Triton's 2 scratch pointers
                continue
            scored.append(((-ints, attrs), sha, mod))
        scored.sort()
        assert scored, f"{symbol}: no instance with {nparams} launch params"
        assert len(scored) == 1 or scored[0][0] != scored[1][0], f"{symbol}: instances {scored} not separable"
        return scored[0][2].name, scored[0][1]

    def ttir(self, sha):
        cb = self.cache.get(sha)
        assert cb, f"{sha[:12]}: not in the Triton cache"
        return cb.with_suffix(".ttir").read_text().splitlines()

    def pin_nearest(self, symbol, regs, ref_sha):
        """Among the instances with `regs`, the one whose .ttir differs least
        from a reference instance (same specialization, different baked
        constants — e.g. the conv_fwd of the speculative state layout, whose
        only change is the page stride)."""
        import difflib
        ref = self.ttir(ref_sha)
        best = None
        for mod, fns in self.mods.items():
            if fns.get(symbol) != regs:
                continue
            sha = self.shas[mod]
            n = sum(1 for l in difflib.unified_diff(ref, self.ttir(sha), lineterm="", n=0)
                    if not l.startswith(("---", "+++", "@@")))
            if best is None or n < best[0]:
                best = (n, mod.name, sha)
        assert best, f"{symbol} REG={regs}: no dump module"
        return best[1], best[2]


# ------------------------------------------------------------ verification
def check_prefill(by, T):
    """Falsify the hand-written GDN/attention wiring against one prefill
    forward's pointers and scalars."""
    n_gdn, n_attn = len(GDN_LAYERS), len(ATTN_LAYERS)
    for tag in ("conv_fwd", "post_conv", "cumsum", "kkt", "solve_tril", "recompute",
                "chunk_h", "chunk_o", "layer_norm"):
        assert len(by[tag]) == n_gdn, (tag, len(by[tag]))
    for tag in ("mrope", "cache", "unified"):
        assert len(by[tag]) == n_attn, (tag, len(by[tag]))
    assert len(by["silu"]) == LAYERS and len(by["reduce"]) == 0
    assert len(by["unified"][0]["params"]) == 28, "not the 2D prefill instance"
    for i in range(n_gdn):
        conv, pc, cs, kkt = by["conv_fwd"][i], by["post_conv"][i], by["cumsum"][i], by["kkt"][i]
        st, rc, ch, co, ln = by["solve_tril"][i], by["recompute"][i], by["chunk_h"][i], by["chunk_o"][i], by["layer_norm"][i]
        assert conv["grid"] == [cdiv_i(T, CONV_BLOCK_M), CONV_DIM // 256, 1], conv["grid"]
        assert [pv(conv, 10), pv(conv, 11)] == [QKVZ_DIM, CONV_DIM], "conv strides"
        assert pv(pc, 0) == pv(conv, 8), "post_conv input is not the conv output"
        assert pc["grid"] == [cdiv_i(T, POST_CONV_BLOCK), GDN_V_HEADS + GDN_K_HEADS, 1], pc["grid"]
        assert [pv(pc, j) for j in range(10, 17)] == \
            [CONV_DIM, GDN_V_HEADS, GDN_V_HEADS, GDN_Q, GDN_Q, GDN_V, T], "post_conv strides/L"
        nt = cdiv_i(T, FLA_CHUNK)
        for r in (cs, kkt, st, rc):
            assert r["grid"] == [nt, GDN_V_HEADS, 1], (r["symbol"], r["grid"])
        assert ch["grid"] == [4, GDN_V_HEADS, 1] and co["grid"] == [2, nt, GDN_V_HEADS]
        assert pv(cs, 0) == pv(pc, 8) and pv(kkt, 1) == pv(pc, 9), "g/beta not from post_conv"
        assert pv(kkt, 0) == pv(pc, 6) == pv(rc, 0) == pv(ch, 0) == pv(co, 1), "k drift"
        assert pv(kkt, 2) == pv(cs, 1) == pv(rc, 6) == pv(ch, 4) == pv(co, 4), "cumsum g drift"
        assert pv(st, 0) == pv(kkt, 3) and pv(rc, 5) == pv(st, 1), "A / Ai chain"
        assert pv(rc, 4) == pv(kkt, 3), "vLLM aliases u onto A (same size); we give u its own buffer"
        assert pv(rc, 1) == pv(pc, 7), "recompute v is not post_conv v"
        assert pv(ch, 1) == pv(rc, 4) and pv(ch, 2) == pv(rc, 3), "chunk_h v/w"
        assert pv(co, 0) == pv(pc, 5) and pv(co, 2) == pv(ch, 3) and pv(co, 3) == pv(ch, 5), "chunk_o q/v_new/h"
        assert pv(ch, 6) != pv(ch, 7), "vLLM h0/ht are temporaries; kern aliases both onto the state line"
        for r, j in ((cs, 2), (kkt, 4), (st, 2), (rc, 7), (ch, 8), (co, 6), (conv, 5)):
            assert pv(r, j) == pv(cs, 2), "cu_seqlens drift"
        for r, j in ((cs, 3), (kkt, 5), (st, 3), (rc, 8), (co, 7)):
            assert pv(r, j) == pv(cs, 3), "chunk_indices drift"
        for r, j in ((cs, 4), (kkt, 6), (st, 4), (rc, 9), (ch, 10), (co, 9)):
            assert pv(r, j) == T, "T arg"
        assert pv(ln, 0) == pv(ln, 1), "layer_norm not in place"
        assert [pv(ln, j) for j in (5, 6, 7, 8)] == [GDN_D, GDN_D, GDN_D, T * GDN_V_HEADS]
        assert ln["grid"] == [cdiv_i(T * GDN_V_HEADS, LN_ROWS_PER_BLOCK), 1, 1]
    for i in range(n_attn):
        mr, ca, un = by["mrope"][i], by["cache"][i], by["unified"][i]
        assert mr["grid"] == [T, 1, 1] and ca["grid"] == [T, 1, 1]
        assert pv(mr, 4) == T, "mrope num_tokens"
        assert pv(ca, 0) == pv(mr, 1), "cache key is not the roped k"
        assert pv(un, 1) == pv(mr, 0), "attention query is not the roped q"
        assert pv(ca, 3) - pv(ca, 2) == V_BYTE_OFF and pv(un, 2) == pv(ca, 2) and pv(un, 3) == pv(ca, 3)
        assert [pv(ca, j) for j in range(7, 16)] == \
            [KV_DIM, QKV_DIM, BLOCK_ELEMS_PER_LAYER, 2 * HEAD_DIM, 0, 0, KV_HEADS * 2 * HEAD_DIM, 0, 0]
        assert un["grid"] == [T // BLOCK_Q + 1, KV_HEADS, 1], un["grid"]
        assert pv(un, 17) == pv(un, 5), "rswa_prefix_lens is not the seq_lens dup"
        assert [pv(un, j) for j in range(11, 17)] == [BLOCK_TABLE_LEN, Q_DIM, HEAD_DIM, Q_DIM, HEAD_DIM, 0]
        assert [pv(un, j) for j in range(18, 24)] == [BLOCK_ELEMS_PER_LAYER, KV_HEADS * 2 * HEAD_DIM, 2 * HEAD_DIM] * 2
        assert pv(un, 25) == 1 and pv(un, 7) == pv(ca, 5) and pv(un, 8) == pv(ca, 6)
    assert len(by["silu"][0]["params"]) == 6 and pv(by["silu"][0], 2) == FFN


def check_decode(by):
    n_gdn, n_attn = len(GDN_LAYERS), len(ATTN_LAYERS)
    for tag in ("conv_update", "recurrent", "layer_norm"):
        assert len(by[tag]) == n_gdn, (tag, len(by[tag]))
    for tag in ("mrope", "cache", "unified", "reduce"):
        assert len(by[tag]) == n_attn, (tag, len(by[tag]))
    assert len(by["unified"][0]["params"]) == 31 and len(by["reduce"][0]["params"]) == 12
    for i in range(n_gdn):
        cu, rec, ln = by["conv_update"][i], by["recurrent"][i], by["layer_norm"][i]
        assert pv(cu, 0) == pv(cu, 4) == pv(rec, 0), "conv update not in place on the qkvz row"
        assert [pv(cu, j) for j in (5, 7, 8)] == [1, 1, 1]
        assert pv(cu, 3) == pv(rec, 8), "conv/ssm index tables differ"
        assert pv(rec, 6) == pv(rec, 7), "decode h0 != ht"
        assert pv(rec, 1) - pv(rec, 2) == GDN_V_HEADS * BF16, "ba layout is not [b | a]"
        assert pv(ln, 0) == pv(ln, 1) == pv(rec, 5), "layer_norm not in place on the recurrent output"
        assert pv(ln, 3) == pv(rec, 0) + Z_OFF * BF16, "z is not the qkvz row tail"
        assert [pv(ln, j) for j in (5, 6, 7, 8)] == [GDN_D, GDN_D, GDN_D, GDN_V_HEADS]
        assert ln["grid"] == [GDN_V_HEADS, 1, 1]
    for i in range(n_attn):
        un, rd = by["unified"][i], by["reduce"][i]
        assert un["grid"] == [1, KV_HEADS, NUM_SEGMENTS]
        assert [pv(rd, j) for j in (1, 2, 3)] == [pv(un, j) for j in (26, 27, 28)]
        assert pv(rd, 0) == pv(un, 0) and pv(rd, 4) == pv(un, 5) and pv(rd, 9) == pv(un, 24)
        assert [pv(rd, j) for j in (6, 7, 8)] == [Q_DIM, HEAD_DIM, BLOCK_TABLE_LEN]
    assert len(by["mrope"][0]["params"]) == 6, "decode mrope should be the num_tokens==1 instance"


# ---------------------------------------------------------------- builders
def sym(s):
    return {"sym": s}


def mul(e, c):
    return {"mul": [e, c]}


def cdiv(e, c):
    return {"ceil_div": [e, c]}


def expr(e):
    return {"expr": e}


def buf(n, off=0):
    return {"buf": n, "offset": off} if off else {"buf": n}


def state(n, off=0):
    return {"state": n, "offset": off} if off else {"state": n}


def i32(v):
    return {"i32": v}


def i64(v):
    return {"i64": v}


def f32(v):
    return {"f32": v}


def u8(v):
    return {"u8": v}


def d(label, kernel, args):
    return {"label": label, "kernel": kernel, "args": args}


def a(i):
    return {"arg": i}


def scr(name, off=0):
    return {"scratch": name, "offset": off} if off else {"scratch": name}


def step(symbol, params, block, grid, args, shared_mem=None, cubin=None, sha256=None):
    s = {"symbol": symbol, "params": params, "block": block, "grid": grid, "args": args}
    if shared_mem is not None:
        s["shared_mem"] = shared_mem
    if cubin is not None:
        s["cubin"] = cubin
    if sha256 is not None:
        s["sha256"] = sha256
    return s


def single(symbol, params, block, grid, shared_mem=None, cubin=None, sha256=None):
    return {"params": params,
            "impl": {"steps": [step(symbol, params, block, grid, [a(i) for i in range(len(params))],
                                    shared_mem, cubin, sha256)]}}


TOKEN_DOMAIN = {"index_into": "model.embed_tokens.weight"}
DOMAINS = {
    "token_ids": TOKEN_DOMAIN,
    "positions": {"index_into": "rope.cos"},
    "slot_mapping": {"index_into": "kv"},
    "block_table": {"index_into": "kv", "unit": BLOCK_SIZE},
    "seq_lens": {"min": 1},
    "cu_seqlens_q": {"min": 0, "max": {"sym": "tokens"}, "monotone": True},
    "next_token": TOKEN_DOMAIN,
    "kv_scales": {"min": 0.0},
}

I2 = ["i64", "i64"]   # Triton's trailing global/profile scratch pointers (always 0)


def build(pre, dec, pins, eps, attn_scale, gdn_scale, silu_sym, spec=None):
    """`spec` (from --spec): {"src": bs=1 launch records of the speculative
    path, "pins": {tag: (cubin, sha)}, "attn_draft": DSpark's pinned
    non-causal unified-attention step} — adds the DFlash2 programs."""
    T = sym("tokens")
    # GDN state layout: one page per layer (Stage 1) or 8 checkpoint pages
    # per layer under speculation; the kernels' page stride is a constexpr
    if spec:
        page_bytes, ssm_off, line_table, gdn_state_bytes = SPEC_PAGE_BYTES, SPEC_SSM_OFF, "gdn.spec_line_index", SPEC_STATE_BYTES
        n_pages = SPEC_PAGES
    else:
        page_bytes, ssm_off, line_table, gdn_state_bytes = GDN_LINE_BYTES, CONV_STATE_BYTES, "gdn.line_index", GDN_STATE_BYTES
        n_pages = GDN_LINES

    def gdn_page(i):
        """First page of GDN layer i (its conv line and SSM checkpoint 0)."""
        g = GDN_LAYERS.index(i)
        return SPEC_BLOCK * g + 1 if spec else g + 1
    buffers = {
        "token_ids": {"dtype": "i64", "shape": ["tokens"], "class": "input"},
        "positions": {"dtype": "i64", "shape": ["tokens"], "class": "input"},
        "slot_mapping": {"dtype": "i64", "shape": ["tokens"], "class": "input"},
        "block_table": {"dtype": "i32", "shape": [BLOCK_TABLE_LEN], "class": "input"},
        "seq_lens": {"dtype": "i32", "shape": [1], "class": "input"},
        "cu_seqlens_q": {"dtype": "i32", "shape": [2], "class": "input"},
        "logits": {"dtype": "bf16", "shape": [1, VOCAB], "class": "workspace"},
        "next_token": {"dtype": "i64", "shape": [1], "class": "output"},
    }
    ws = {
        "residual": ["tokens", HIDDEN], "x": ["tokens", HIDDEN], "y": ["tokens", HIDDEN],
        "final_x": [1, HIDDEN],
        # GDN
        "qkvz": ["tokens", QKVZ_DIM], "ba": ["tokens", BA_DIM], "conv_out": ["tokens", CONV_DIM],
        "gdn_q": ["tokens", GDN_Q], "gdn_k": ["tokens", GDN_Q], "gdn_v": ["tokens", GDN_V],
        "Ai": ["tokens", GDN_V_HEADS, FLA_CHUNK],
        "w": ["tokens", GDN_V], "u": ["tokens", GDN_V], "v_new": ["tokens", GDN_V],
        "h": [NT_MAX, GDN_V_HEADS, GDN_D, GDN_D],
        "core_attn_out": ["tokens", GDN_V], "z_c": ["tokens", GDN_V],
        # attention
        "qkv": ["tokens", QKV_DIM], "q_n": ["tokens", Q_DIM], "k_n": ["tokens", KV_DIM],
        "cos_g": ["tokens", ROT_HALF], "sin_g": ["tokens", ROT_HALF],
        "attn_out": ["tokens", Q_DIM], "gated": ["tokens", Q_DIM],
        # MLP
        "gate_up": ["tokens", 2 * FFN], "act": ["tokens", FFN],
    }
    for name, shape in ws.items():
        buffers[name] = {"dtype": "bf16", "shape": shape, "class": "workspace"}
    for name, shape in {"g": ["tokens", GDN_V_HEADS], "beta": ["tokens", GDN_V_HEADS],
                        "g_cum": ["tokens", GDN_V_HEADS], "A": ["tokens", GDN_V_HEADS, FLA_CHUNK]}.items():
        buffers[name] = {"dtype": "f32", "shape": shape, "class": "workspace"}

    def weight(name, shape, dtype="bf16"):
        buffers[name] = {"dtype": dtype, "shape": shape, "class": "weight"}

    weight("model.embed_tokens.weight", [VOCAB, HIDDEN])
    weight("lm_head.weight", [VOCAB, HIDDEN])
    weight("model.norm.weight_p1", [HIDDEN], "f32")
    weight("rope.cos", [MAX_POS, ROT_HALF])
    weight("rope.sin", [MAX_POS, ROT_HALF])
    weight("kv_scales", [2], "f32")
    weight("fla.chunk_indices", [NT_MAX, 2], "i32")
    weight("fla.chunk_offsets", [2], "i64")
    weight("conv.batch_ptr", [CHUNK_MAX // CONV_BLOCK_M], "i32")
    weight("conv.token_chunk_offset", [CHUNK_MAX // CONV_BLOCK_M], "i32")
    weight("gdn.line_index", [64], "i32")
    weight("gdn.has_initial", [16], "u8")
    for i in range(LAYERS):
        p = f"model.layers.{i}."
        weight(p + "input_layernorm.weight_p1", [HIDDEN], "f32")
        weight(p + "post_attention_layernorm.weight_p1", [HIDDEN], "f32")
        weight(p + "mlp.gate_up_proj.weight", [2 * FFN, HIDDEN])
        weight(p + "mlp.down_proj.weight", [HIDDEN, FFN])
        if i in ATTN_LAYERS:
            weight(p + "self_attn.qkv_proj.weight", [QKV_DIM, HIDDEN])
            weight(p + "self_attn.q_norm.weight_p1", [HEAD_DIM], "f32")
            weight(p + "self_attn.k_norm.weight_p1", [HEAD_DIM], "f32")
            weight(p + "self_attn.o_proj.weight", [HIDDEN, Q_DIM])
        else:
            weight(p + "linear_attn.in_proj_qkvz.weight", [QKVZ_DIM, HIDDEN])
            weight(p + "linear_attn.in_proj_ba.weight", [BA_DIM, HIDDEN])
            weight(p + "linear_attn.conv1d.weight", [CONV_DIM, 4])
            weight(p + "linear_attn.A_log", [GDN_V_HEADS], "f32")
            weight(p + "linear_attn.dt_bias", [GDN_V_HEADS])
            weight(p + "linear_attn.norm.weight", [GDN_D])
            weight(p + "linear_attn.out_proj.weight", [HIDDEN, GDN_V])

    if spec:
        buffers.update({
            # caller contract: accepted tokens of the previous round (anchor
            # + accepted drafts), selects the SSM slot / conv offset to resume from
            "num_accepted_tokens": {"dtype": "i32", "shape": [1], "class": "input"},
            "draft_block_table": {"dtype": "i32", "shape": [D_TABLE_LEN], "class": "input"},
            "anchor_token": {"dtype": "i64", "shape": [1], "class": "input"},
            # target hidden states at the 5 taps, projected by fc: written by
            # prefill / decode_spec / verify, read by draft_precompute
            "fc_out": {"dtype": "bf16", "shape": ["tokens", HIDDEN], "class": "carry"},
            "verify_tokens": {"dtype": "i64", "shape": [SPEC_BLOCK], "class": "output"},
            "draft_tokens": {"dtype": "i64", "shape": [DRAFT_TOKENS], "class": "output"},
            "logits_blk": {"dtype": "bf16", "shape": [SPEC_BLOCK, VOCAB], "class": "workspace"},
            "cand_ids": {"dtype": "i64", "shape": [DRAFT_TOKENS, SEL_K], "class": "workspace"},
            "cand_vals": {"dtype": "f32", "shape": [DRAFT_TOKENS, SEL_K], "class": "workspace"},
        })
        for name, shape in {
            "a_c": ["tokens", GDN_V_HEADS], "b_c": ["tokens", GDN_V_HEADS],   # contiguous a/b for the T=8 recurrent kernel
            "kv_flat": ["tokens", D_KV_FLAT],
            "d_qkv": [SPEC_BLOCK, D_QKV], "d_q": [SPEC_BLOCK, D_Q], "d_coef": [SPEC_BLOCK, D_CONV_PROJ],
            "d_attn": [SPEC_BLOCK, D_Q],
            "hidden_r": [DRAFT_TOKENS, SEL_RANK], "succ_g": [DRAFT_TOKENS * SEL_K, SEL_RANK],
            "pred_g": [DRAFT_TOKENS * SEL_K, SEL_RANK], "pred_anchor": [1, SEL_RANK],
        }.items():
            buffers[name] = {"dtype": "bf16", "shape": shape, "class": "workspace"}
        weight("gdn.spec_line_index", [64], "i32")
        weight("gdn.spec_slots", [len(GDN_LAYERS), SPEC_BLOCK], "i32")
        weight("gdn.one", [1], "i32")
        for j in range(5):
            weight(f"draft.fc.{j}.weight", [HIDDEN, HIDDEN])
        weight("draft.hidden_norm.weight", [HIDDEN])
        weight("draft.norm.weight", [HIDDEN])
        weight("draft.fused_kv.weight", [D_KV_FLAT, HIDDEN])
        weight("draft.rope.cos_sin_cache", [MAX_POS, D_HEAD_DIM])
        weight("draft.kv_scales", [2 * D_LAYERS], "f32")
        weight("draft.selector.hidden_projection.weight", [SEL_RANK, HIDDEN])
        weight("draft.selector.predecessor", [VOCAB, SEL_RANK])
        weight("draft.selector.successor", [VOCAB, SEL_RANK])
        for l in range(D_LAYERS):
            p = f"draft.layers.{l}."
            weight(p + "input_layernorm.weight", [HIDDEN])
            weight(p + "post_attention_layernorm.weight", [HIDDEN])
            weight(p + "self_attn.qkv_proj.weight", [D_QKV, HIDDEN])
            weight(p + "self_attn.q_norm.weight", [D_HEAD_DIM])
            weight(p + "self_attn.k_norm.weight", [D_HEAD_DIM])
            weight(p + "self_attn.o_proj.weight", [HIDDEN, D_Q])
            weight(p + "mlp.gate_up_proj.weight", [2 * FFN, HIDDEN])
            weight(p + "mlp.down_proj.weight", [HIDDEN, FFN])
            for c in ("attention_conv", "mlp_conv"):
                weight(p + c + ".base_kernel", [4, HIDDEN])
                weight(p + c + ".kernel_projection.weight", [D_CONV_PROJ, HIDDEN])

    def blk(tag, src=pre):
        return src[tag][0]["block"]

    def smem(tag, src=pre):
        return src[tag][0]["dynamic_shared_mem_bytes"]

    def spec_kernel(tag, params, grid, **kw):
        """Single-step kernel from the speculative capture, pinned."""
        r = spec["src"][tag]
        cubin, sha = spec["pins"][tag]
        return single(SPEC_SYMS[tag], params, r["block"], grid,
                      shared_mem=r["dynamic_shared_mem_bytes"] or None, cubin=cubin, sha256=sha, **kw)

    def tri(tag, params, grid, src=pre, **kw):
        """Single-step Triton kernel pinned to its dump module."""
        cubin, sha = pins[(tag, src is pre)]
        return single(TRITON[tag], params, blk(tag, src), grid,
                      shared_mem=smem(tag, src) or None, cubin=cubin, sha256=sha, **kw)

    # gcnt is declared `out`: the kernel does read it (atomic task counter),
    # but its contents are zero at rest -- alloc_zeros at load, and the tail
    # resets it before the dispatch ends
    GEMM8_AN_PARAMS = ["out buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "inout buffer<bf16>",
                       "in buffer<f32>", "out buffer<bf16>", "out buffer<f32>", "out buffer<i32>",
                       "i32", "i32", "i32", "f32"]
    GEMM8_SG_PARAMS = ["out buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>",
                       "inout buffer<bf16>", "in buffer<f32>", "out buffer<bf16>", "out buffer<f32>",
                       "out buffer<i32>", "i32", "i32", "i32", "i32", "i32", "i32", "f32"]
    GEMMA_PARAMS = ["out buffer<bf16>", "in buffer<bf16>", "in buffer<f32>",
                    "i32", "i32", "i32", "i32", "i32", "i32", "i32", "f32"]
    GEMMA_FUSED_PARAMS = ["out buffer<bf16>", "in buffer<bf16>", "inout buffer<bf16>",
                          "in buffer<f32>", "i32", "i32", "i32", "i32", "f32"]
    GEMMA_SMEM = (2 * HIDDEN + 512) * 4
    GEMMA_HEAD_SMEM = (2 * HEAD_DIM + 512) * 4
    LN_PARAMS = ["inout buffer<bf16>", "out buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>",
                 "out buffer<f32>", "i32", "i32", "i32", "i32", "f32"] + I2
    LN_IFACE = LN_PARAMS[:4] + LN_PARAMS[5:]
    UNIFIED_PARAMS = [
        "out buffer<bf16>", "in buffer<bf16>", "inout ptr", "inout ptr",
        "in buffer<i32>", "in buffer<i32>", "f32",
        "in buffer<f32>", "in buffer<f32>", "f32", "f32",
        "i64", "i64", "i64", "i64", "i64", "i64",
        "in buffer<i32>", "i64", "i64", "i64", "i64", "i64", "i64",
        "in buffer<i32>", "i32", "out buffer<f32>", "out buffer<f32>", "out buffer<f32>"] + I2
    ATTN_IFACE = UNIFIED_PARAMS[:26] + UNIFIED_PARAMS[29:]
    REDUCE_PARAMS = ["out buffer<bf16>", "in buffer<f32>", "in buffer<f32>", "in buffer<f32>",
                     "in buffer<i32>", "f32", "i64", "i64", "i64", "in buffer<i32>"] + I2

    def layer_norm_kernel(src, grid, rows_shape):
        """Gated RMSNorm (FLA layer_norm_fwd) with its Rstd side output as
        impl scratch."""
        cubin, sha = pins[("layer_norm", src is pre)]
        return {
            "params": LN_IFACE,
            "impl": {
                "scratch": {"rstd": {"dtype": "f32", "shape": rows_shape}},
                "steps": [step(TRITON["layer_norm"], LN_PARAMS, blk("layer_norm", src), grid,
                               [a(0), a(1), a(2), a(3), scr("rstd")] + [a(i) for i in range(4, 11)],
                               cubin=cubin, sha256=sha)],
            },
        }

    kernels = {
        "embedding": single("kern_embedding_i64_bf16",
                            ["in buffer<i64>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32"],
                            [256, 1, 1], [T, 1, 1], cubin="embedding.cubin"),
        "gemm": single("extern:cublaslt_bf16_tn",
                       ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i32"],
                       [1, 1, 1], [1, 1, 1]),
        # Gemma norms: one block per row; rows/grid differ per use (ATen's
        # reduction width depends on the row count, so `rows` is an arg).
        "gemma_norm": single("kern_gemma_rms_norm_bf16", GEMMA_PARAMS, [512, 1, 1], [T, 1, 1],
                             shared_mem=GEMMA_SMEM, cubin="gemma_rms_norm.cubin"),
        # per-head q/k gemma norm + partial rope + kv cache write of one
        # attention layer (tools/kernels-src/attn_prep.cu): CTA = (token, head)
        "attn_prep": single("kern_attn_prep_bf16",
                            ["out buffer<bf16>", "out buffer<bf16>", "in buffer<bf16>",
                             "in buffer<f32>", "in buffer<f32>", "in buffer<bf16>",
                             "in buffer<bf16>", "inout ptr", "inout ptr", "in buffer<i64>",
                             "i32", "i32", "i64", "i64", "i64", "i32", "f32"],
                            [512, 1, 1], [mul(T, HEADS + KV_HEADS), 1, 1],
                            cubin="attn_prep.cubin"),
        "gemma_fused_norm": single("kern_gemma_fused_add_rms_norm_bf16", GEMMA_FUSED_PARAMS,
                                   [512, 1, 1], [T, 1, 1], shared_mem=GEMMA_SMEM,
                                   cubin="gemma_rms_norm.cubin"),
        "silu_mul": single(silu_sym, ["out buffer<bf16>", "in buffer<bf16>", "i32", "i32", "f32", "i32"],
                           blk("silu"), [T, 1, 1]),
        # streaming M<=8 GEMM (tools/kernels-src/gemm8.cu): one persistent CTA
        # per GB300 SM, weight rows via TMA ring + mma; grid/block/smem are
        # compile-time constants of that kernel
        "gemm8_gateup_silu": single("kern_gemm8_gateup_silu_bf16",
                                    ["out buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>",
                                     "i32", "i32", "i32"],
                                    [GEMM8_THREADS, 1, 1], [GEMM8_GRID, 1, 1],
                                    shared_mem=GEMM8_SMEM, cubin="gemm8.cubin"),
        # down GEMM + residual add + Gemma fused_add_rms_norm of the next
        # layer input (gemm8's split-K partials, y and the task counters are
        # impl scratch; the norm tail reproduces kern_gemma_rms_norm.cu)
        # two weights sharing one x: in_proj_qkvz + in_proj_ba per GDN layer
        "gemm8_dual": single("kern_gemm8_dual_bf16",
                             ["out buffer<bf16>", "out buffer<bf16>", "in buffer<bf16>",
                              "in buffer<bf16>", "in buffer<bf16>", "i32", "i32", "i32", "i32"],
                             [GEMM8_THREADS, 1, 1], [GEMM8_GRID, 1, 1],
                             shared_mem=GEMM8_SMEM, cubin="gemm8.cubin"),
        # o_proj + the sigmoid gate on its input + residual add + post-attention
        # norm: x = attn * sigmoid(gate) is built inside the GEMM
        "gemm8_sgate_add_norm": {
            "params": GEMM8_SG_PARAMS[:6] + GEMM8_SG_PARAMS[9:],
            "impl": {
                "scratch": {"y": {"dtype": "bf16", "shape": [8, HIDDEN]},
                            "partial": {"dtype": "f32", "shape": [8, 8, HIDDEN]},
                            "gcnt": {"dtype": "i32", "shape": [2]}},
                "steps": [step("kern_gemm8_sgate_add_norm_bf16", GEMM8_SG_PARAMS,
                               [GEMM8_THREADS, 1, 1], [GEMM8_GRID, 1, 1],
                               [a(0), a(1), a(2), a(3), a(4), a(5), scr("y"), scr("partial"), scr("gcnt"),
                                a(6), a(7), a(8), a(9), a(10), a(11), a(12)],
                               shared_mem=GEMM8_SMEM, cubin="gemm8.cubin")],
            },
        },
        "gemm8_add_norm": {
            "params": GEMM8_AN_PARAMS[:5] + GEMM8_AN_PARAMS[8:],
            "impl": {
                "scratch": {"y": {"dtype": "bf16", "shape": [8, HIDDEN]},
                            "partial": {"dtype": "f32", "shape": [8, 8, HIDDEN]},
                            "gcnt": {"dtype": "i32", "shape": [2]}},
                "steps": [step("kern_gemm8_add_norm_bf16", GEMM8_AN_PARAMS,
                               [GEMM8_THREADS, 1, 1], [GEMM8_GRID, 1, 1],
                               [a(0), a(1), a(2), a(3), a(4), scr("y"), scr("partial"), scr("gcnt"),
                                a(5), a(6), a(7), a(8)],
                               shared_mem=GEMM8_SMEM, cubin="gemm8.cubin")],
            },
        },
        "copy_rows": single("kern_copy_rows_bf16",
                            ["out buffer<bf16>", "in buffer<bf16>", "i32", "i32", "i32"],
                            [256, 1, 1], [T, 1, 1], cubin="copy_rows.cubin"),
        "last_row": single("kern_last_row_bf16",
                           ["out buffer<bf16>", "in buffer<bf16>", "i32", "i32", "i32"],
                           [256, 1, 1], [1, 1, 1], cubin="copy_rows.cubin"),
        "sigmoid_mul": single("kern_sigmoid_mul_bf16",
                              ["out buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>",
                               "i32", "i32", "i32", "i32"],
                              [256, 1, 1], [T, 1, 1], cubin="sigmoid_mul.cubin"),
        "argmax_row": {
            "params": ["in buffer<bf16>", "out buffer<i64>", "i32"],
            "impl": {
                "scratch": {"pmax": {"dtype": "f32", "shape": [1, 64]},
                            "pidx": {"dtype": "i32", "shape": [1, 64]}},
                "steps": [
                    step("kern_argmax_partial_bf16",
                         ["in buffer<bf16>", "out buffer<f32>", "out buffer<i32>", "i32"],
                         [1024, 1, 1], [1, 64, 1], [a(0), scr("pmax"), scr("pidx"), a(2)],
                         cubin="argmax.cubin"),
                    step("kern_argmax_final_i64",
                         ["in buffer<f32>", "in buffer<i32>", "out buffer<i64>", "i32"],
                         [64, 1, 1], [1, 1, 1], [scr("pmax"), scr("pidx"), a(1), i32(64)],
                         cubin="argmax.cubin"),
                ],
            },
        },
        # --- GDN prefill chain (vLLM triton backend, FLA chunk kernels)
        "conv_fwd": tri("conv_fwd",
                        ["in buffer<bf16>", "in buffer<bf16>", "inout ptr", "in buffer<i32>",
                         "in buffer<u8>", "in buffer<i32>", "in buffer<i32>", "in buffer<i32>",
                         "out buffer<bf16>", "i32", "i64", "i64"] + I2,
                        [cdiv(T, CONV_BLOCK_M), CONV_DIM // 256, 1]),
        "post_conv": tri("post_conv",
                         ["in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "in buffer<f32>",
                          "in buffer<bf16>", "out buffer<bf16>", "out buffer<bf16>", "out buffer<bf16>",
                          "out buffer<f32>", "out buffer<f32>",
                          "i32", "i32", "i32", "i32", "i32", "i32", "i32"] + I2,
                         [cdiv(T, POST_CONV_BLOCK), GDN_V_HEADS + GDN_K_HEADS, 1]),
        "cumsum": tri("cumsum", ["in buffer<f32>", "out buffer<f32>", "in buffer<i32>",
                                 "in buffer<i32>", "i32"] + I2,
                      [cdiv(T, FLA_CHUNK), GDN_V_HEADS, 1]),
        "kkt": tri("kkt", ["in buffer<bf16>", "in buffer<f32>", "in buffer<f32>", "out buffer<f32>",
                           "in buffer<i32>", "in buffer<i32>", "i32"] + I2,
                   [cdiv(T, FLA_CHUNK), GDN_V_HEADS, 1]),
        # writes only the lower/diagonal 16x16 tiles of each 64x64 block; the
        # upper tiles are the zeros the buffer was allocated with
        "solve_tril": tri("solve_tril", ["in buffer<f32>", "out buffer<bf16>", "in buffer<i32>",
                                         "in buffer<i32>", "i32"] + I2,
                          [cdiv(T, FLA_CHUNK), GDN_V_HEADS, 1]),
        "recompute": tri("recompute",
                         ["in buffer<bf16>", "in buffer<bf16>", "in buffer<f32>", "out buffer<bf16>",
                          "out buffer<bf16>", "in buffer<bf16>", "in buffer<f32>", "in buffer<i32>",
                          "in buffer<i32>", "i32"] + I2,
                         [cdiv(T, FLA_CHUNK), GDN_V_HEADS, 1]),
        # h0/ht are the SSM state line itself: each program loads its h0
        # tile first and stores ht last, so in-place is race-free
        "chunk_h": tri("chunk_h",
                       ["in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>",
                        "in buffer<f32>", "out buffer<bf16>", "inout ptr", "inout ptr",
                        "in buffer<i32>", "in buffer<i64>", "i32"] + I2,
                       [GDN_D // 32, GDN_V_HEADS, 1]),
        "chunk_o": tri("chunk_o",
                       ["in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>",
                        "in buffer<f32>", "out buffer<bf16>", "in buffer<i32>", "in buffer<i32>",
                        "f32", "i32"] + I2,
                       [GDN_D // 64, cdiv(T, FLA_CHUNK), GDN_V_HEADS]),
        "gated_norm": layer_norm_kernel(pre, [mul(T, GDN_V_HEADS // LN_ROWS_PER_BLOCK), 1, 1],
                                        ["tokens", GDN_V_HEADS]),
        # --- GDN decode: the whole chain in one launch, one thread-block
        # cluster per q/k head (tools/kernels-src/gdn_decode.cu)
        "gdn_decode": single("kern_gdn_decode_bf16",
                             ["inout buffer<bf16>", "in buffer<bf16>", "inout ptr", "inout ptr",
                              "in buffer<i32>", "in buffer<bf16>", "in buffer<f32>",
                              "in buffer<bf16>", "out buffer<bf16>", "in buffer<bf16>",
                              "f32", "f32", "i32", "i32", "i32", "i32"],
                             [256, 1, 1], [2 * GDN_V_HEADS, 1, 1], cubin="gdn_decode.cubin"),
        # --- attention
        "attn_prefill": tri("unified", ATTN_IFACE, [cdiv(T, BLOCK_Q), KV_HEADS, 1]),
        "attn": {
            "params": ATTN_IFACE,
            "impl": {
                "scratch": {
                    "segm_out": {"dtype": "f32", "shape": [1, HEADS, NUM_SEGMENTS, HEAD_DIM]},
                    "segm_max": {"dtype": "f32", "shape": [1, HEADS, NUM_SEGMENTS]},
                    "segm_expsum": {"dtype": "f32", "shape": [1, HEADS, NUM_SEGMENTS]},
                },
                "steps": [
                    step(TRITON["unified"], UNIFIED_PARAMS, blk("unified", dec),
                         [1, KV_HEADS, NUM_SEGMENTS],
                         [a(i) for i in range(26)]
                         + [scr("segm_out"), scr("segm_max"), scr("segm_expsum"), a(26), a(27)],
                         shared_mem=smem("unified", dec), cubin=pins[("unified", False)][0],
                         sha256=pins[("unified", False)][1]),
                    step(TRITON["reduce"], REDUCE_PARAMS, blk("reduce", dec), [1, HEADS, 1],
                         [a(0), scr("segm_out"), scr("segm_max"), scr("segm_expsum"), a(5),
                          f32(1.0), i64(Q_DIM), i64(HEAD_DIM), i64(BLOCK_TABLE_LEN), a(24),
                          i64(0), i64(0)],
                         shared_mem=smem("reduce", dec), cubin=pins[("reduce", False)][0],
                         sha256=pins[("reduce", False)][1]),
                ],
            },
        },
    }

    Z2 = [i64(0), i64(0)]

    if spec:
        # the whole spec-path GDN chain of a layer in one launch
        # (tools/kernels-src/gdn_spec.cu): conv update + post_conv split/
        # l2norm/gating + a/b/z copies + T-row delta rule with per-token
        # SSM checkpoints + gated RMSNorm.  Counter scratch is zero at
        # rest (the last CTA resets); hpart holds data, write-before-read.
        GDN_SPEC_PARAMS = [
            "inout buffer<bf16>",                                    # qkvz (conv writeback)
            "in buffer<bf16>", "inout ptr", "inout ptr",             # conv weight, conv state, ssm state
            "in buffer<i32>", "in buffer<i32>", "in buffer<i32>",    # line, num_accepted, cu_seqlens_q
            "in buffer<i32>",                                        # spec_slots row
            "in buffer<bf16>", "in buffer<f32>", "in buffer<bf16>",  # ba, A_log, dt_bias
            "out buffer<bf16>", "in buffer<bf16>",                   # core_attn_out, norm.weight
            "out buffer<bf16>", "out buffer<bf16>", "out buffer<bf16>",  # gdn_q/k/v
            "out buffer<f32>", "out buffer<f32>",                    # g, beta
            "out buffer<bf16>", "out buffer<bf16>", "out buffer<bf16>",  # a_c, b_c, z_c
            "out buffer<i32>", "out buffer<f32>", "out buffer<i32>", "out buffer<i32>",  # scratch
            "f32", "f32", "i32", "i32", "i32", "i32", "i32"]         # scale, eps, tokens, n_pages, cls, cds, sls
        RMS_PARAMS = ["out buffer<bf16>", "in buffer<bf16>", "i64", "i64", "i64", "i64", "i64",
                      "in buffer<bf16>", "i64", "f32", "i32", "i32"]
        ad = spec["attn_draft"]
        kernels.update({
            "gemm_acc": single("extern:cublaslt_bf16_tn_acc",
                               ["in buffer<bf16>", "in buffer<bf16>", "inout buffer<bf16>", "i32", "i32", "i32"],
                               [1, 1, 1], [1, 1, 1]),
            # --- target GDN under speculation: the whole chain is fused
            # into gdn_spec below (kern_gdn_spec_bf16)
            "gdn_spec": {
                "params": GDN_SPEC_PARAMS[:21] + GDN_SPEC_PARAMS[25:],
                "impl": {
                    "scratch": {"xcnt": {"dtype": "i32", "shape": [GDN_Q // GDN_D]},
                                "hpart": {"dtype": "f32", "shape": [SPEC_BLOCK, GDN_V_HEADS, 4]},
                                "hcnt": {"dtype": "i32", "shape": [GDN_V_HEADS]},
                                "gcnt": {"dtype": "i32", "shape": [1]}},
                    "steps": [step("kern_gdn_spec_bf16", GDN_SPEC_PARAMS, [256, 1, 1], [192, 1, 1],
                                   [a(i) for i in range(21)]
                                   + [scr("xcnt"), scr("hpart"), scr("hcnt"), scr("gcnt")]
                                   + [a(i) for i in range(21, 28)],
                                   cubin="gdn_spec.cubin")],
                },
            },
            # recurrent delta rule over T rows with the sigmoid gating fused:
            # initial state from SSM slot num_accepted-1, every row's state
            # checkpointed to its own slot (ssm_state_indices)
            "argmax": {
                "params": ["in buffer<bf16>", "out buffer<i64>", "i32"],
                "impl": {
                    "scratch": {"pmax": {"dtype": "f32", "shape": [SPEC_BLOCK, 64]},
                                "pidx": {"dtype": "i32", "shape": [SPEC_BLOCK, 64]}},
                    "steps": [
                        step("kern_argmax_partial_bf16",
                             ["in buffer<bf16>", "out buffer<f32>", "out buffer<i32>", "i32"],
                             [1024, 1, 1], [T, 64, 1], [a(0), scr("pmax"), scr("pidx"), a(2)],
                             cubin="argmax.cubin"),
                        step("kern_argmax_final_i64",
                             ["in buffer<f32>", "in buffer<i32>", "out buffer<i64>", "i32"],
                             [64, 1, 1], [T, 1, 1], [scr("pmax"), scr("pidx"), a(1), i32(64)],
                             cubin="argmax.cubin"),
                    ],
                },
            },
            # --- draft (vLLM CUDA kernels + the borrowed non-causal attention instance)
            "d_rms_norm": spec_kernel("d_rms_norm", RMS_PARAMS, [T, 1, 1]),
            "d_head_norm": spec_kernel("d_head_norm", RMS_PARAMS, [mul(T, D_HEADS), 1, 1]),
            "d_kv_norm": spec_kernel("d_head_norm", RMS_PARAMS, [mul(T, D_KV_HEADS), 1, 1]),
            "d_fused_norm": spec_kernel("d_fused_norm",
                                        ["inout buffer<bf16>", "i64", "inout buffer<bf16>", "in buffer<bf16>",
                                         "f32", "i32", "i32", "i64"], [T, 1, 1]),
            "d_rope": spec_kernel("d_rope",
                                  ["in buffer<i64>", "inout buffer<bf16>", "inout buffer<bf16>", "in buffer<bf16>",
                                   "i32", "i64", "i64", "i64", "i32", "i32", "i32", "i64", "u8"], [T, 1, 1]),
            "d_cache": spec_kernel("d_cache",
                                   ["in buffer<bf16>", "in buffer<bf16>", "inout ptr", "inout ptr", "in buffer<i64>",
                                    "i64", "i64", "i64", "i64", "i64", "i32", "i32", "i32",
                                    "in buffer<f32>", "in buffer<f32>", "i32"], [T, 1, 1]),
            "attn_draft": single("kernel_unified_attention", ATTN_IFACE, ad["block"],
                                 [cdiv(T, ad["grid"][0]["ceil_div"][1]), D_KV_HEADS, 1],
                                 shared_mem=ad.get("shared_mem"), cubin=ad["cubin"], sha256=ad["sha256"]),
            "dflash_conv": single("kern_dflash_conv_bf16",
                                  ["inout buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>",
                                   "i32", "i32", "i32", "i32", "i32"],
                                  [256, 1, 1], [HIDDEN // 256, 1, 1], cubin="dflash_conv.cubin"),
            "topk16": single("kern_topk16_bf16",
                             ["in buffer<bf16>", "out buffer<i64>", "out buffer<f32>", "i32", "i32"],
                             [1024, 1, 1], [DRAFT_TOKENS, 1, 1], cubin="topk_row.cubin"),
            "dflash_select": single("kern_dflash_select",
                                    ["in buffer<i64>", "in buffer<f32>", "in buffer<bf16>", "in buffer<bf16>",
                                     "in buffer<bf16>", "in buffer<bf16>", "out buffer<i64>", "i32", "i32", "i32"],
                                    [SEL_RANK, 1, 1], [1, 1, 1], shared_mem=SEL_K * SEL_RANK * 4,
                                    cubin="dflash_select.cubin"),
            "gather_cands": single("kern_embedding_i64_bf16",
                                   ["in buffer<i64>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32"],
                                   [256, 1, 1], [DRAFT_TOKENS * SEL_K, 1, 1], cubin="embedding.cubin"),
            "gather_row": single("kern_embedding_i64_bf16",
                                 ["in buffer<i64>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32"],
                                 [256, 1, 1], [1, 1, 1], cubin="embedding.cubin"),
        })

    def gemm(label, ab, w, c, m, n, k):
        return d(label, "gemm", [buf(ab), buf(w), buf(c), m, i32(n), i32(k)])

    def fused(label, x_in, w):
        return d(label, "gemma_fused_norm",
                 [buf("x"), buf(x_in), buf("residual"), buf(w), i32(HIDDEN), T,
                  i32(HIDDEN), i32(HIDDEN), f32(eps)])

    def gdn_layer(i, decode, nacc=None, small=False):
        """decode=False: the chunked FLA prefill chain.  decode=True: the
        recurrent chain — Stage 1's packed kernels, or, when `nacc` names
        the num_accepted_tokens buffer, vLLM's speculative kernels over the
        `tokens` rows (SSM checkpoint per row, resume from slot nacc-1)."""
        p = f"model.layers.{i}.linear_attn."
        l = f"l{i}."
        g = GDN_LAYERS.index(i)
        line = g + 1
        idx = buf(line_table, 4 * line)
        page = gdn_page(i)
        if small:
            # one weight stream for both input projections
            ds = [d(l + "in_proj", "gemm8_dual",
                    [buf("qkvz"), buf("ba"), buf("x"), buf(p + "in_proj_qkvz.weight"),
                     buf(p + "in_proj_ba.weight"), T, i32(QKVZ_DIM), i32(BA_DIM), i32(HIDDEN)])]
        else:
            ds = [
                gemm(l + "in_proj_qkvz", "x", p + "in_proj_qkvz.weight", "qkvz", T, QKVZ_DIM, HIDDEN),
                gemm(l + "in_proj_ba", "x", p + "in_proj_ba.weight", "ba", T, BA_DIM, HIDDEN),
            ]
        a_ = buf("ba", GDN_V_HEADS * BF16)   # ba row = [b | a]
        b_ = buf("ba")
        if not decode and nacc is None:
            ds += [
                d(l + "conv", "conv_fwd",
                  [buf("qkvz"), buf(p + "conv1d.weight"), state("gdn"), idx, buf("gdn.has_initial"),
                   buf("cu_seqlens_q"), buf("conv.batch_ptr"), buf("conv.token_chunk_offset"),
                   buf("conv_out"), i32(GDN_LINES), i64(QKVZ_DIM), i64(CONV_DIM)] + Z2),
                # a/b are strided views into ba (vLLM copies them contiguous;
                # the kernel takes row strides, so no copy here)
                d(l + "post_conv", "post_conv",
                  [buf("conv_out"), a_, b_, buf(p + "A_log"), buf(p + "dt_bias"),
                   buf("gdn_q"), buf("gdn_k"), buf("gdn_v"), buf("g"), buf("beta"),
                   i32(CONV_DIM), i32(BA_DIM), i32(BA_DIM), i32(GDN_Q), i32(GDN_Q), i32(GDN_V), T] + Z2),
                d(l + "cumsum", "cumsum",
                  [buf("g"), buf("g_cum"), buf("cu_seqlens_q"), buf("fla.chunk_indices"), T] + Z2),
                d(l + "kkt", "kkt",
                  [buf("gdn_k"), buf("beta"), buf("g_cum"), buf("A"), buf("cu_seqlens_q"),
                   buf("fla.chunk_indices"), T] + Z2),
                d(l + "solve_tril", "solve_tril",
                  [buf("A"), buf("Ai"), buf("cu_seqlens_q"), buf("fla.chunk_indices"), T] + Z2),
                d(l + "recompute_wu", "recompute",
                  [buf("gdn_k"), buf("gdn_v"), buf("beta"), buf("w"), buf("u"), buf("Ai"),
                   buf("g_cum"), buf("cu_seqlens_q"), buf("fla.chunk_indices"), T] + Z2),
                d(l + "chunk_h", "chunk_h",
                  [buf("gdn_k"), buf("u"), buf("w"), buf("v_new"), buf("g_cum"), buf("h"),
                   state("gdn", page * page_bytes + ssm_off),
                   state("gdn", page * page_bytes + ssm_off),
                   buf("cu_seqlens_q"), buf("fla.chunk_offsets"), T] + Z2),
                d(l + "chunk_o", "chunk_o",
                  [buf("gdn_q"), buf("gdn_k"), buf("v_new"), buf("h"), buf("g_cum"),
                   buf("core_attn_out"), buf("cu_seqlens_q"), buf("fla.chunk_indices"),
                   f32(gdn_scale), T] + Z2),
                d(l + "z_copy", "copy_rows",
                  [buf("z_c"), buf("qkvz", Z_OFF * BF16), i32(GDN_V), i32(QKVZ_DIM), i32(GDN_V)]),
                d(l + "gated_norm", "gated_norm",
                  [buf("core_attn_out"), buf("core_attn_out"), buf(p + "norm.weight"), buf("z_c"),
                   i32(GDN_D), i32(GDN_D), i32(GDN_D), expr(mul(T, GDN_V_HEADS)), f32(eps)] + Z2),
            ]
        elif nacc is None:
            # the decode chain (conv update + delta rule step + gated norm),
            # fused into one launch -- see tools/kernels-src/gdn_decode.cu
            ds += [
                d(l + "gdn", "gdn_decode",
                  [buf("qkvz"), buf(p + "conv1d.weight"), state("gdn"), state("gdn", ssm_off),
                   idx, buf("ba"), buf(p + "A_log"), buf(p + "dt_bias"), buf("core_attn_out"),
                   buf(p + "norm.weight"), f32(gdn_scale), f32(eps), i32(n_pages),
                   i32(page_bytes // BF16), i32(CONV_DIM), i32(page_bytes // 4)]),
            ]
        else:
            # vLLM's spec-decode GDN path, fused into one launch (the chain
            # was: conv update with the accepted-token offset, post_conv
            # split + l2norm + gating, contiguous a/b copies, the recurrent
            # kernel over T rows checkpointing every row's state, z copy,
            # gated RMSNorm).  The fused kernel reproduces every buffer
            # write of that chain, including the qkvz conv writeback and
            # the per-token SSM checkpoints.
            ds += [
                d(l + "gdn", "gdn_spec",
                  [buf("qkvz"), buf(p + "conv1d.weight"), state("gdn"), state("gdn", ssm_off),
                   idx, nacc, buf("cu_seqlens_q"), buf("gdn.spec_slots", 4 * SPEC_BLOCK * g),
                   buf("ba"), buf(p + "A_log"), buf(p + "dt_bias"), buf("core_attn_out"),
                   buf(p + "norm.weight"), buf("gdn_q"), buf("gdn_k"), buf("gdn_v"),
                   buf("g"), buf("beta"), buf("a_c"), buf("b_c"), buf("z_c"),
                   f32(gdn_scale), f32(eps), T, i32(n_pages),
                   i32(page_bytes // BF16), i32(ssm_off // (CONV_DIM * BF16)), i32(page_bytes // 4)]),
            ]
        ds.append(gemm(l + "out_proj", "core_attn_out", p + "out_proj.weight", "y", T, HIDDEN, GDN_V))
        return ds

    def attn_layer(i, decode, small=False):
        p = f"model.layers.{i}.self_attn."
        l = f"l{i}."
        koff = ATTN_LAYERS.index(i) * LAYER_KV_BYTES
        ks, vs = buf("kv_scales"), buf("kv_scales", 4)
        kv_k, kv_v = state("kv", koff), state("kv", koff + V_BYTE_OFF)
        ds = [
            gemm(l + "qkv_proj", "x", p + "qkv_proj.weight", "qkv", T, QKV_DIM, HIDDEN),
            # q/k gemma norm + partial rope + kv cache write, one CTA per
            # (token, head) -- see tools/kernels-src/attn_prep.cu
            d(l + "prep", "attn_prep",
              [buf("q_n"), buf("k_n"), buf("qkv"), buf(p + "q_norm.weight_p1"),
               buf(p + "k_norm.weight_p1"), buf("cos_g"), buf("sin_g"), kv_k, kv_v,
               buf("slot_mapping"), expr(mul(T, HEADS)), expr(mul(T, KV_HEADS)),
               i64(BLOCK_STRIDE), i64(KV_HEADS * 2 * HEAD_DIM), i64(2 * HEAD_DIM),
               i32(BLOCK_SIZE), f32(eps)]),
            d(l + "attn", "attn" if decode else "attn_prefill",
              [buf("attn_out"), buf("q_n"), kv_k, kv_v, buf("block_table"), buf("seq_lens"),
               f32(attn_scale), ks, vs, f32(1.0), f32(0.0),
               i64(BLOCK_TABLE_LEN), i64(Q_DIM), i64(HEAD_DIM), i64(Q_DIM), i64(HEAD_DIM), i64(0),
               buf("seq_lens"),
               i64(BLOCK_STRIDE), i64(KV_HEADS * 2 * HEAD_DIM), i64(2 * HEAD_DIM),
               i64(BLOCK_STRIDE), i64(KV_HEADS * 2 * HEAD_DIM), i64(2 * HEAD_DIM),
               buf("cu_seqlens_q"), i32(1)] + Z2),
            d(l + "gate", "sigmoid_mul",
              [buf("gated"), buf("attn_out"), buf("qkv", GATE_OFF * BF16), i32(HEADS), i32(HEAD_DIM),
               i32(QKV_DIM), i32(2 * HEAD_DIM)]),
            gemm(l + "o_proj", "gated", p + "o_proj.weight", "y", T, HIDDEN, Q_DIM),
        ]
        if small:
            # sigmoid gate + o_proj + residual add + post-attention norm
            ds[-2:] = [d(l + "o_norm", "gemm8_sgate_add_norm",
                         [buf("x"), buf("attn_out"), buf("qkv", GATE_OFF * BF16), buf(p + "o_proj.weight"),
                          buf("residual"), buf(f"model.layers.{i}.post_attention_layernorm.weight_p1"),
                          T, i32(HIDDEN), i32(Q_DIM), i32(HEAD_DIM), i32(QKV_DIM), i32(2 * HEAD_DIM),
                          f32(eps)])]
        return ds

    def forward(decode, taps=False, nacc=None, tail=None):
        """Target forward.  decode: recurrent GDN + split-KV attention (else
        chunked FLA + 2D attention over the `tokens` rows).  taps: the five
        fc GEMMs into `fc_out` at the DFlash taps (after layers 5/19/33/47/61:
        residual = hidden + residual there).  tail: "prefill" (last row ->
        next_token), "decode" (row 0 -> next_token), "verify" (all rows ->
        verify_tokens)."""
        tail = tail or ("decode" if decode else "prefill")
        # bs<=8 forwards (decode / decode_spec / verify) take the streaming
        # gemm8 kernels; chunked prefill keeps cuBLAS
        small = decode or tail == "verify"
        ds = [
            d("embed", "embedding",
              [buf("token_ids"), buf("model.embed_tokens.weight"), buf("residual"), T, i32(HIDDEN)]),
            # rope tables gathered by position; mrope gets num_tokens=0 so its
            # three (t/h/w) planes alias this one table — text-only positions
            d("rope_cos", "embedding", [buf("positions"), buf("rope.cos"), buf("cos_g"), T, i32(ROT_HALF)]),
            d("rope_sin", "embedding", [buf("positions"), buf("rope.sin"), buf("sin_g"), T, i32(ROT_HALF)]),
            d("l0.input_norm", "gemma_norm",
              [buf("x"), buf("residual"), buf("model.layers.0.input_layernorm.weight_p1"),
               i32(HIDDEN), T, i32(1), i32(HIDDEN), i32(0), i32(HIDDEN), i32(0), f32(eps)]),
        ]
        for i in range(LAYERS):
            p = f"model.layers.{i}."
            l = f"l{i}."
            lay = attn_layer(i, decode, small) if i in ATTN_LAYERS else gdn_layer(i, decode, nacc, small)
            if small and i not in ATTN_LAYERS:
                # out_proj GEMM + residual add + post-attention norm (the
                # attention layers fold their sigmoid gate in too: attn_layer)
                lay[-1] = d(l + "out_norm", "gemm8_add_norm",
                            [buf("x"), buf("core_attn_out"), buf(p + "linear_attn.out_proj.weight"),
                             buf("residual"), buf(p + "post_attention_layernorm.weight_p1"),
                             T, i32(HIDDEN), i32(GDN_V), f32(eps)])
            ds += lay
            if small:
                mlp = [d(l + "gate_up_silu", "gemm8_gateup_silu",
                         [buf("act"), buf("x"), buf(p + "mlp.gate_up_proj.weight"),
                          T, i32(FFN), i32(HIDDEN)])]
            else:
                mlp = [gemm(l + "gate_up", "x", p + "mlp.gate_up_proj.weight", "gate_up", T, 2 * FFN, HIDDEN),
                       d(l + "silu_mul", "silu_mul",
                         [buf("act"), buf("gate_up"), i32(FFN), i32(0), f32(1.0), i32(0)])]
            last = i + 1 == LAYERS
            wnorm = "model.norm.weight_p1" if last else f"model.layers.{i + 1}.input_layernorm.weight_p1"
            nlabel = l + ("final_norm" if last else "next_input_norm")
            if not small:
                ds.append(fused(l + "post_attn_norm", "y", p + "post_attention_layernorm.weight_p1"))
            ds += mlp
            if small:
                # down GEMM + residual add + next input norm in one dispatch
                ds.append(d(l + "down_norm", "gemm8_add_norm",
                            [buf("x"), buf("act"), buf(p + "mlp.down_proj.weight"), buf("residual"),
                             buf(wnorm), T, i32(HIDDEN), i32(FFN), f32(eps)]))
            else:
                ds.append(gemm(l + "down_proj", "act", p + "mlp.down_proj.weight", "y", T, HIDDEN, FFN))
                ds.append(fused(nlabel, "y", wnorm))
            if taps and i in TAPS:
                # residual now holds hidden + residual = the input of layer
                # i+1, vLLM's aux hidden state; fc's column block j, β=1
                # accumulating after the first (no concat)
                j = TAPS[i]
                ds.append(d(l + "fc_tap", "gemm" if j == 0 else "gemm_acc",
                            [buf("residual"), buf(f"draft.fc.{j}.weight"), buf("fc_out"),
                             T, i32(HIDDEN), i32(HIDDEN)]))
        # The final norm runs over all rows like vLLM (ATen's reduction width
        # depends on the row count); lm_head only needs the last one.
        if tail == "decode":
            ds.append(gemm("lm_head", "x", "lm_head.weight", "logits", i32(1), VOCAB, HIDDEN))
            ds.append(d("sample", "argmax_row", [buf("logits"), buf("next_token"), i32(VOCAB)]))
        elif tail == "verify":
            ds.append(gemm("lm_head", "x", "lm_head.weight", "logits_blk", T, VOCAB, HIDDEN))
            ds.append(d("sample", "argmax", [buf("logits_blk"), buf("verify_tokens"), i32(VOCAB)]))
        else:
            ds.append(d("last_row", "last_row", [buf("final_x"), buf("x"), i32(HIDDEN), i32(HIDDEN), T]))
            ds.append(gemm("lm_head", "final_x", "lm_head.weight", "logits", i32(1), VOCAB, HIDDEN))
            ds.append(d("sample", "argmax_row", [buf("logits"), buf("next_token"), i32(VOCAB)]))
        return ds

    def d_fused(label, x, w):
        return d(label, "d_fused_norm",
                 [buf(x), i64(HIDDEN), buf("residual"), buf(w), f32(eps), T, i32(HIDDEN), i64(HIDDEN)])

    def d_conv(label, x, c, side):
        return d(label, "dflash_conv",
                 [buf(x), buf("d_coef"), buf(c + ".base_kernel"), T, i32(HIDDEN), i32(D_GROUPS),
                  i32(D_CONV_PROJ), i32(side)])

    def draft():
        """One non-causal pass over the 8-row block [anchor, mask x7] at
        positions pos..pos+7 (env tokens=8), then top-16 + selector walk ->
        7 draft tokens.  Draft hidden size = target's, so the target's
        activation buffers are reused; the two grouped convs of every layer
        wrap attention and MLP (prepare before, finish after, coefficients
        from one GEMM of the pre-attention / pre-MLP normed state)."""
        mp = "draft."
        ds = [
            d("embed", "embedding",
              [buf("token_ids"), buf("model.embed_tokens.weight"), buf("residual"), T, i32(HIDDEN)]),
            d("l0.input_norm", "d_rms_norm",
              [buf("x"), buf("residual"), i64(HIDDEN), i64(0), i64(0), i64(0), i64(0),
               buf(mp + "layers.0.input_layernorm.weight"), i64(0), f32(eps), T, i32(HIDDEN)]),
        ]
        for j in range(D_LAYERS):
            p = f"{mp}layers.{j}."
            l = f"l{j}."
            koff = j * D_LAYER_KV_BYTES
            ks, vs = buf("draft.kv_scales", j * 8), buf("draft.kv_scales", j * 8 + 4)
            last = j + 1 == D_LAYERS
            ds += [
                gemm(l + "attn_conv_proj", "x", p + "attention_conv.kernel_projection.weight", "d_coef",
                     T, D_CONV_PROJ, HIDDEN),
                d_conv(l + "attn_conv_pre", "x", p + "attention_conv", 0),
                gemm(l + "qkv_proj", "x", p + "self_attn.qkv_proj.weight", "d_qkv", T, D_QKV, HIDDEN),
                d(l + "q_norm", "d_head_norm",
                  [buf("d_q"), buf("d_qkv"), i64(D_HEAD_DIM), i64(D_QKV), i64(0), i64(D_HEADS), i64(0),
                   buf(p + "self_attn.q_norm.weight"), i64(0), f32(eps), expr(mul(T, D_HEADS)), i32(D_HEAD_DIM)]),
                d(l + "k_norm", "d_kv_norm",
                  [buf("k_n"), buf("d_qkv", D_Q * BF16), i64(D_HEAD_DIM), i64(D_QKV), i64(0), i64(D_KV_HEADS),
                   i64(0), buf(p + "self_attn.k_norm.weight"), i64(0), f32(eps), expr(mul(T, D_KV_HEADS)),
                   i32(D_HEAD_DIM)]),
                d(l + "rope", "d_rope",
                  [buf("positions"), buf("d_q"), buf("k_n"), buf("draft.rope.cos_sin_cache"), i32(D_HEAD_DIM),
                   i64(D_Q), i64(D_KV), i64(D_HEAD_DIM), i32(D_HEADS), i32(D_KV_HEADS), i32(D_HEAD_DIM),
                   i64(0), u8(0)]),
                d(l + "kv_write", "d_cache",
                  [buf("k_n"), buf("d_qkv", (D_Q + D_KV) * BF16),
                   state("draft_kv", koff), state("draft_kv", koff + D_V_BYTE_OFF), buf("slot_mapping"),
                   i64(D_BLOCK_STRIDE), i64(D_KV_HEADS * 2 * D_HEAD_DIM), i64(2 * D_HEAD_DIM),
                   i64(D_KV), i64(D_QKV), i32(D_KV_HEADS), i32(D_HEAD_DIM), i32(D_BLOCK), ks, vs, i32(0)]),
                d(l + "attn", "attn_draft",
                  [buf("d_attn"), buf("d_q"), state("draft_kv", koff), state("draft_kv", koff + D_V_BYTE_OFF),
                   buf("draft_block_table"), buf("seq_lens"), f32(D_ATTN_SCALE), ks, vs, f32(1.0), f32(0.0),
                   i64(D_TABLE_LEN), i64(D_Q), i64(D_HEAD_DIM), i64(D_Q), i64(D_HEAD_DIM), i64(0),
                   buf("seq_lens"),
                   i64(D_BLOCK_STRIDE), i64(D_KV_HEADS * 2 * D_HEAD_DIM), i64(2 * D_HEAD_DIM),
                   i64(D_BLOCK_STRIDE), i64(D_KV_HEADS * 2 * D_HEAD_DIM), i64(2 * D_HEAD_DIM),
                   buf("cu_seqlens_q"), i32(1)] + Z2),
                gemm(l + "o_proj", "d_attn", p + "self_attn.o_proj.weight", "y", T, HIDDEN, D_Q),
                d_conv(l + "attn_conv_post", "y", p + "attention_conv", 1),
                d_fused(l + "post_attn_norm", "y", p + "post_attention_layernorm.weight"),
                gemm(l + "mlp_conv_proj", "y", p + "mlp_conv.kernel_projection.weight", "d_coef",
                     T, D_CONV_PROJ, HIDDEN),
                d_conv(l + "mlp_conv_pre", "y", p + "mlp_conv", 0),
                d(l + "gate_up_silu", "gemm8_gateup_silu",
                  [buf("act"), buf("y"), buf(p + "mlp.gate_up_proj.weight"), T, i32(FFN), i32(HIDDEN)]),
                gemm(l + "down_proj", "act", p + "mlp.down_proj.weight", "x", T, HIDDEN, FFN),
                d_conv(l + "mlp_conv_post", "x", p + "mlp_conv", 1),
                d_fused(l + ("final_norm" if last else "next_input_norm"), "x",
                        mp + "norm.weight" if last else f"{mp}layers.{j + 1}.input_layernorm.weight"),
            ]
        # rows 1..7 (the masks) -> shared lm_head -> top-16 candidates; the
        # rank-256 selector scores predecessor/successor codebook rows of the
        # candidates (gathered with the embedding kernel) and walks greedily
        row1 = HIDDEN * BF16
        ds += [
            d("lm_head", "gemm", [buf("x", row1), buf("lm_head.weight"), buf("logits_blk"),
                                  i32(DRAFT_TOKENS), i32(VOCAB), i32(HIDDEN)]),
            d("sel.topk", "topk16", [buf("logits_blk"), buf("cand_ids"), buf("cand_vals"), i32(VOCAB), i32(VOCAB)]),
            d("sel.hidden_proj", "gemm", [buf("x", row1), buf("draft.selector.hidden_projection.weight"),
                                          buf("hidden_r"), i32(DRAFT_TOKENS), i32(SEL_RANK), i32(HIDDEN)]),
            d("sel.succ", "gather_cands", [buf("cand_ids"), buf("draft.selector.successor"), buf("succ_g"),
                                           i32(DRAFT_TOKENS * SEL_K), i32(SEL_RANK)]),
            d("sel.pred", "gather_cands", [buf("cand_ids"), buf("draft.selector.predecessor"), buf("pred_g"),
                                           i32(DRAFT_TOKENS * SEL_K), i32(SEL_RANK)]),
            d("sel.pred_anchor", "gather_row", [buf("anchor_token"), buf("draft.selector.predecessor"),
                                                buf("pred_anchor"), i32(1), i32(SEL_RANK)]),
            d("sel.walk", "dflash_select", [buf("cand_ids"), buf("cand_vals"), buf("hidden_r"), buf("succ_g"),
                                            buf("pred_g"), buf("pred_anchor"), buf("draft_tokens"),
                                            i32(DRAFT_TOKENS), i32(SEL_RANK), i32(SEL_RANK)]),
        ]
        return ds

    def draft_precompute():
        """Target taps -> draft context KV (DSpark's shape, docs/spec-decode.md):
        hidden_norm(fc_out) -> fused KV GEMM [n, 10240] -> per layer k_norm,
        K-only rope (key = the same buffer with 0 kv heads), cache write.
        Runs at env tokens = valid rows of fc_out; positions / slot_mapping
        are still those of the forward that produced them."""
        ds = [
            d("hidden_norm", "d_rms_norm",
              [buf("x"), buf("fc_out"), i64(HIDDEN), i64(0), i64(0), i64(0), i64(0),
               buf("draft.hidden_norm.weight"), i64(0), f32(eps), T, i32(HIDDEN)]),
            gemm("fused_kv", "x", "draft.fused_kv.weight", "kv_flat", T, D_KV_FLAT, HIDDEN),
        ]
        for j in range(D_LAYERS):
            koff = j * D_LAYER_KV_BYTES
            ks, vs = buf("draft.kv_scales", j * 8), buf("draft.kv_scales", j * 8 + 4)
            ds += [
                d(f"ctx{j}.k_norm", "d_kv_norm",
                  [buf("k_n"), buf("kv_flat", j * 2 * D_KV * BF16), i64(D_HEAD_DIM), i64(D_KV_FLAT), i64(0),
                   i64(D_KV_HEADS), i64(0), buf(f"draft.layers.{j}.self_attn.k_norm.weight"), i64(0), f32(eps),
                   expr(mul(T, D_KV_HEADS)), i32(D_HEAD_DIM)]),
                d(f"ctx{j}.rope", "d_rope",
                  [buf("positions"), buf("k_n"), buf("k_n"), buf("draft.rope.cos_sin_cache"), i32(D_HEAD_DIM),
                   i64(D_KV), i64(0), i64(D_HEAD_DIM), i32(D_KV_HEADS), i32(0), i32(D_HEAD_DIM), i64(0), u8(0)]),
                d(f"ctx{j}.kv_write", "d_cache",
                  [buf("k_n"), buf("kv_flat", (j * 2 * D_KV + D_KV) * BF16),
                   state("draft_kv", koff), state("draft_kv", koff + D_V_BYTE_OFF), buf("slot_mapping"),
                   i64(D_BLOCK_STRIDE), i64(D_KV_HEADS * 2 * D_HEAD_DIM), i64(2 * D_HEAD_DIM),
                   i64(D_KV), i64(D_KV_FLAT), i32(D_KV_HEADS), i32(D_HEAD_DIM), i32(D_BLOCK), ks, vs, i32(0)]),
            ]
        return ds

    states = {"kv": {"bytes_per_token": KV_BYTES_PER_TOKEN}, "gdn": {"bytes_fixed": gdn_state_bytes}}
    meta = {"version": 2, "model": "qwen3.8-27b"}
    if spec:
        one = buf("gdn.one")
        programs = {
            "prefill": {"dispatches": forward(False, taps=True)},
            # non-speculative decode on the speculative state layout: the
            # spec kernels at tokens=1, resuming from SSM slot 0 / conv
            # offset 0 (where prefill and every previous tokens=1 step leave
            # the state)
            "decode": {"dispatches": forward(True, nacc=one)},
            "decode_spec": {"dispatches": forward(True, taps=True, nacc=one)},
            # verify = the target over [anchor, d0..d6] (tokens=8): 2D
            # attention, spec GDN resuming from num_accepted_tokens-1
            "verify": {"dispatches": forward(False, taps=True, nacc=buf("num_accepted_tokens"), tail="verify")},
            "draft": {"dispatches": draft()},
            "draft_precompute": {"dispatches": draft_precompute()},
        }
        states["draft_kv"] = {"bytes_per_token": D_KV_BYTES_PER_TOKEN}
        meta = {"version": 2, "model": "qwen3.8-27b-dflash2",
                # caller contract: draft rows = [anchor] + [mask] * (block-1) at
                # tokens=block; verify at tokens=block; num_accepted_tokens =
                # 1 + accepted drafts of the previous round (1 after prefill)
                "spec": {"block": SPEC_BLOCK, "mask_token": MASK_TOKEN}}
        DOMAINS.update({
            "num_accepted_tokens": {"min": 1, "max": SPEC_BLOCK},
            "draft_block_table": {"index_into": "draft_kv", "unit": D_BLOCK},
            "anchor_token": TOKEN_DOMAIN, "verify_tokens": TOKEN_DOMAIN, "draft_tokens": TOKEN_DOMAIN,
            "cand_ids": {"index_into": "draft.selector.successor"},
            "draft.kv_scales": {"min": 0.0},
        })
    else:
        programs = {
            "prefill": {"dispatches": forward(False)},
            "decode": {"dispatches": forward(True)},
        }
    for name, dom in DOMAINS.items():
        if name in buffers:
            buffers[name]["domain"] = dom
    # the runtime rejects dead kernels / buffers (a manifest describes only
    # what runs): the spec layout leaves Stage 1's packed decode chain unused
    used_k = {dd["kernel"] for pr in programs.values() for dd in pr["dispatches"]}
    used_b = {a["buf"] for pr in programs.values() for dd in pr["dispatches"] for a in dd["args"] if "buf" in a}
    kernels = {k: v for k, v in kernels.items() if k in used_k}
    buffers = {k: v for k, v in buffers.items() if k in used_b}
    return {
        "meta": meta,
        # bs=1; prefill per chunk (tokens <= CHUNK_MAX) over *all* prompt
        # tokens (it emits next_token), decode at tokens=1
        "symbols": {"tokens": {"max": CHUNK_MAX}},
        "states": states,
        "buffers": buffers,
        "kernels": kernels,
        "programs": programs,
    }


def main():
    repo = pathlib.Path(__file__).resolve().parent.parent
    args = sys.argv[1:]
    spec_dump = None
    if "--spec" in args:
        i = args.index("--spec")
        spec_dump = pathlib.Path(args[i + 1])
        del args[i:i + 2]
    dump = pathlib.Path(args[0] if args else repo / "dumped-kernels" / "pid1898802")
    out = pathlib.Path(args[1] if len(args) > 1 else
                       repo / "examples" / ("qwen3.8-27b-dflash2.json" if spec_dump else "qwen3.8-27b.json"))

    recs = load(dump / "launches.jsonl")
    fwds = forwards(recs)
    ts = [tokens_of(f) for f in fwds]
    # real prefill forwards (multi-token, after the profiling passes) + the
    # decode forward that follows the longest one
    prefills = [(t, i) for i, t in enumerate(ts) if t and 1 < t <= CHUNK_MAX and i > ts.index(1)]
    assert len(prefills) >= 3, f"need several prefill forwards to fit grids, got {prefills}"
    t_ref, i_ref = max(prefills)
    pre = group(fwds[i_ref])
    dec = group(fwds[i_ref + 1])
    assert tokens_of(fwds[i_ref + 1]) == 1
    for t, i in prefills:
        check_prefill(group(fwds[i]), t)
    check_decode(dec)
    print(f"capture: prefill forwards T={sorted(t for t, _ in prefills)} verified, "
          f"decode after T={t_ref} verified", file=sys.stderr)

    eps = pf(pre["layer_norm"][0], 9)
    attn_scale = pf(pre["unified"][0], 6)
    gdn_scale = pf(pre["chunk_o"][0], 8)
    assert pf(dec["recurrent"][0], 9) == gdn_scale and pf(dec["unified"][0], 6) == attn_scale
    assert pf(dec["layer_norm"][0], 9) == eps
    assert abs(eps - 1e-6) < 1e-12 and attn_scale == 0.0625
    silu_sym = pre["silu"][0]["symbol"]

    pinner = Pinner(dump)
    pins = {}
    for tag in TRITON:
        for src, is_pre in ((pre, True), (dec, False)):
            if src[tag]:
                pins[(tag, is_pre)] = pinner.pin(TRITON[tag], src[tag][0]["attributes"]["num_regs"])
    for (tag, is_pre), (mod, sha) in sorted(pins.items()):
        print(f"  pin {TRITON[tag]:<52} {'prefill' if is_pre else 'decode '} -> {mod} {sha[:12]}",
              file=sys.stderr)

    spec = None
    if spec_dump:
        # Two dumps feed one manifest (Stage 1's target instances + the
        # speculative path's), so pinned cubins are named by content
        # (sha prefix); extract_kernels.sh resolves them across dumps.
        by_sha = lambda mod_sha: (f"{mod_sha[1][:12]}.cubin", mod_sha[1])  # noqa: E731
        pins = {k: by_sha(v) for k, v in pins.items()}
        src = spec_launches(load(spec_dump / "launches.jsonl"))
        sp = Pinner(spec_dump)
        spins = {}
        for tag, r in src.items():
            spins[tag] = by_sha(sp.pin(SPEC_SYMS[tag], r["attributes"]["num_regs"], nparams=len(r["params"])))
        # conv_fwd of the spec state layout: Stage 1's instance with the page stride rebaked
        pins[("conv_fwd", True)] = by_sha(sp.pin_nearest(TRITON["conv_fwd"], pre["conv_fwd"][0]["attributes"]["num_regs"],
                                                         pinner.pin(TRITON["conv_fwd"], pre["conv_fwd"][0]["attributes"]["num_regs"])[1]))
        assert pf(src["recurrent_spec"], 15) == gdn_scale and pf(src["recurrent_spec"], 4) == 1.0
        assert pf(src["recurrent_spec"], 5) == 20.0, "softplus threshold"
        assert pv(src["conv_update_spec"], 9) == pv(src["conv_update_spec"], 10), "conv update not in place"
        assert pf(src["d_rms_norm"], 9) == eps and pf(src["d_fused_norm"], 4) == eps
        dspark = json.loads((repo / "examples" / "qwen3-4b-dspark.json").read_text())
        ad = dspark["kernels"]["attn_draft"]["impl"]["steps"][0]
        assert ad["symbol"] == TRITON["unified"] and ad["cubin"] == "unified_noncausal.cubin"
        for tag, (mod, sha) in sorted(spins.items()):
            print(f"  spec pin {SPEC_SYMS[tag][:52]:<52} -> {mod} {sha[:12]}", file=sys.stderr)
        print(f"  spec pin conv_fwd (page stride rebaked) -> {pins[('conv_fwd', True)][0]}", file=sys.stderr)
        spec = {"src": src, "pins": spins, "attn_draft": ad}

    m = build(pre, dec, pins, eps, attn_scale, gdn_scale, silu_sym, spec)
    out.write_text(json.dumps(m, indent=1) + "\n")
    n_disp = {k: len(v["dispatches"]) for k, v in m["programs"].items()}
    print(f"wrote {out}: {len(m['buffers'])} buffers, {len(m['kernels'])} kernels, "
          f"dispatches {n_disp}", file=sys.stderr)


if __name__ == "__main__":
    main()
