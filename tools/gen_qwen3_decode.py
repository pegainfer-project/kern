#!/usr/bin/env python3
"""Generate examples/qwen3-4b.json from mined vLLM data.

bs=1 manifest，两个 program：`prefill`（chunk 级，tokens ∈ [1, CHUNK_MAX]，
只落 KV 不出 logits）+ `decode`（tokens=1，出 next_token）。chunked prefill
= caller 连调 prefill 若干次 + 最后一个 token 走 decode（decode 就是
"prefill_last"，免掉 symbol 依赖的 offset）。数据源是 TRITON_ATTN backend
的 capture（唯一 flat-ABI 的 attention backend——FA4/trtllm-gen 都是
packed struct / TMA descriptor，不可 rebind）：

- 四个 CUDA flat 核（rms/rms_head/rope/fused）+ Triton 版 reshape_and_cache
  + `kernel_unified_attention`(3D decode 实例) + `reduce_segments`，全部用
  真实挖到的 symbol、逐参数类型/方向、标量字面量（取自代表性 decode
  forward）。注意 Triton 同名核不同 constexpr 实例 ABI 不同（unified 的
  2D prefill 实例 28 参数、3D decode 实例 31 参数；reduce 的 num_seqs=1
  被 Triton 特化进 binary 不再传参）——这里 pin 的是 decode 实例。
- GEMM 是 runtime 特判（symbol 前缀 `extern:`，cublasLt）；embedding 是
  待写的 Triton 占位。
- KV state 布局从 vLLM 的逐层池改为层交织 `[page][layer][16][8][2][128]`
  （同一批 kernel，靠 stride 参数 ×LAYERS 和 state offset 字面量适配），
  bytes_per_token = 36*2*8*128*2 = 147456。
- 发射前对挖矿数据做结构断言：q/k/v 在 qkv 中的视图偏移、residual 全程
  同址、逐层权重互异、KV 池 k/v 相距 256B、cache/attention 共享同一
  KV 池与 scale 指针、unified 与 reduce 共享 segm 缓冲——连线是手写的，
  挖矿数据负责证伪它。

另外发射 examples/qwen3-4b-dspark.json：同一 manifest 里带上 DSpark 投机解码
（deepseek-ai/dspark_qwen3_4b_block7 draft）。要点：

- draft 与 target 几何完全同构（hidden/heads/FFN/head_dim 全同），5 层 forward
  直接复用全部既有 kernel 条目——grid 是 tokens 的表达式，draft 以 env
  tokens=7、verify 以 tokens=8 跑同一批核。新增 kernel 条目全是布线/常量差异
  （gemm_acc、argmax_row、embedding_row、attn_draft），手写核数量为零。
- 两个 28 参 unified 实例（causal=prefill/verify、non-causal=draft）symbol、
  参数布局、block、smem 全部相同，ABI 无法消歧——从 launch 的 num_regs 区分
  （primary dump 只有 causal；spec dump 里 grid=[2,8,1] 的 draft 实例是另一个
  reg 数），再用 cuobjdump 定位 module 文件，强制 per-step cubin 钉定 + sha256。
- draft 的 context KV 不来自 draft forward，而是 target 隐状态投影：5 个 tap
  点（layer 0/8/16/24/32 的 next_input_norm 之后，residual 恰是 hidden+residual
  的 aux）各做一个 β=1 累加 GEMM（fc 权重按列切 5 块），免 concat 免拷贝；
  fc_out 是新 buffer class `carry`（verify/prefill 写、draft_precompute 读，
  跨 program 交接，顺序是 caller 契约）。
- draft_precompute：hidden_norm → 融合 KV GEMM [n,10240] → 逐层 k_norm（打包
  写进 k_n）→ K-only rope（num_kv=0 跳过 key，等效 vLLM 的 key=NULL）→
  reshape_and_cache 进 draft_kv（5 层交织 state）。
- markov 头展开成 7 步链：embedding_row(markov_w1[prev]) → gemm_acc 把
  markov_w2 偏置累进该行 base logits → argmax_row 出 draft token。
- 无损 oracle：greedy 投机解码逐 token 等于普通 decode，输出 byte-match 即
  全链路正确；接错 tap/头只会掉接受率。

跑法：python3 tools/gen_qwen3_decode.py \
    [primary launches.jsonl] [spec dump dir]
"""

import hashlib
import json
import pathlib
import struct
import subprocess
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from kern_manifest import DumpIndex, normalize, program, SCHEMA_VERSION  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mine_capture as mc
from handwritten import hw  # tools/handwritten.py: build + pin handwritten cubins

HIDDEN = 2560
LAYERS = 36
HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
FFN = 9728
VOCAB = 151936
MAX_POS = 40960
Q_DIM = HEADS * HEAD_DIM      # 4096
KV_DIM = KV_HEADS * HEAD_DIM  # 1024
QKV_DIM = Q_DIM + 2 * KV_DIM  # 6144
BF16 = 2
NUM_SEGMENTS = 16             # unified 3D 实例的 NUM_SEGMENTS_PER_SEQ（grid.z）
CHUNK_MAX = 2048              # prefill 单 chunk 上限（tokens symbol 的 max）
BLOCK_Q = 4                   # unified 2D 实例每 block 的 query 行数（拟合证实）

# DSpark draft（dspark_qwen3_4b_block7）：几何与 target 同构，只有 5 层
DRAFT_LAYERS = 5
BLOCK_TOKENS = 7              # draft 每轮 query 数 = num_speculative_tokens
# A round stages one row group per sequence for draft and verify alike, so
# both take BLOCK_TOKENS rows: verify = anchor + the first 6 drafts (d6 is
# drafted, never verified). The verify pass is causal, so dropping the last
# row changes nothing the other rows predict.
VERIFY_TOKENS = BLOCK_TOKENS
MASK_TOKEN = 151669           # the draft config's mask_token_id, the undrafted rows of draft's block
MARKOV_RANK = 256
TAPS = {0: 0, 8: 1, 16: 2, 24: 3, 32: 4}  # target_layer_ids [1,9,17,25,33]-1
DRAFT_KV_DIM = DRAFT_LAYERS * 2 * KV_DIM  # 融合 KV GEMM 输出行宽 10240

# 层交织 KV 布局（vLLM 逐层池 -> 我们的单 state）：
# 一个 block(16 token) 在一层里占 16*8*2*128 = 32768 elems；层间连续。
BLOCK_ELEMS_PER_LAYER = 16 * KV_HEADS * 2 * HEAD_DIM
BLOCK_STRIDE = LAYERS * BLOCK_ELEMS_PER_LAYER          # 传给 kernel 的 elems
LAYER_KV_BYTES = BLOCK_ELEMS_PER_LAYER * BF16          # state offset 步长
KV_BYTES_PER_TOKEN = LAYERS * 2 * KV_DIM * BF16        # 147456
DRAFT_BLOCK_STRIDE = DRAFT_LAYERS * BLOCK_ELEMS_PER_LAYER
DRAFT_KV_BYTES_PER_TOKEN = DRAFT_LAYERS * 2 * KV_DIM * BF16  # 20480
V_BYTE_OFF = 2 * HEAD_DIM                              # 挖矿实测 k/v 相距 256B
MAX_BLOCKS = MAX_POS // 16                             # block_table_stride=256 实测吻合
MAX_SEQS = 256                # `seqs` var 上限：decode_batch 一步的最大序列数

SYMS = {
    "rms": "rms_norm_kernelIN3c108BFloat16ELi8ELi2",
    "rms_head": "rms_norm_kernelIN3c108BFloat16ELi8ELi3",
    "rope": "rotary_embedding",
    "silu": "act_and_mul",
    "fused": "fused_add_rms",
    "cache": "reshape_and_cache_kernel_flash",
    "unified": "kernel_unified_attention",
    "reduce": "reduce_segments",
}


def pv(rec, i):
    return int.from_bytes(bytes.fromhex(rec["params"][i]["data"]), "little")


def pick_forwards(jsonl):
    """代表性 decode forward + 最后两个真实 prefill forward（多 token、
    位于 profiling dummy pass 之后，用两个不同长度拟合/证伪 2D grid）。"""
    recs = mc.load(jsonl)
    windows = mc.slice_windows(recs, mc.GAP_MS_DEFAULT)
    _, _, forwards = mc.slice_forwards(windows)
    _, tokens, forwards, _ = mc.pick_tokens_reference(forwards)
    di = max(i for i, t in enumerate(tokens) if t == 1)
    prefills = [(t, forwards[i][1]) for i, t in enumerate(tokens)
                if 1 < t <= CHUNK_MAX and i > tokens.index(1)][-2:]
    assert len(prefills) == 2, "需要两个真实 prefill forward 拟合 grid"
    return forwards[di][1], prefills


def extract(fwd):
    """代表性 decode forward -> 各 flat 核按序分组。"""
    by = {k: [] for k in SYMS}
    for r in fwd:
        for tag, pat in SYMS.items():
            if pat in r["symbol"] and isinstance(r.get("params"), list):
                by[tag].append(r)
                break
    assert len(by["rms"]) == 1, by["rms"]
    assert len(by["rms_head"]) == 2 * LAYERS
    assert len(by["rope"]) == LAYERS
    assert len(by["silu"]) == LAYERS
    assert len(by["fused"]) == 2 * LAYERS
    assert len(by["cache"]) == LAYERS
    assert len(by["unified"]) == LAYERS
    assert len(by["reduce"]) == LAYERS
    # Triton 同名不同实例：decode 3D 实例 31 参数 / reduce 12 参数
    assert len(by["unified"][0]["params"]) == 31, "不是 3D decode 实例"
    assert len(by["reduce"][0]["params"]) == 12, "num_seqs 未被特化，不是 bs=1 实例"
    return by


def check_topology(by):
    """挖矿地址证伪手写连线。"""
    residual = pv(by["rms"][0], 1)
    eps = pv(by["rms"][0], 9)
    weights = set()
    for i in range(LAYERS):
        q, k = by["rms_head"][2 * i], by["rms_head"][2 * i + 1]
        rope, cache = by["rope"][i], by["cache"][i]
        uni, red = by["unified"][i], by["reduce"][i]
        post, nxt = by["fused"][2 * i], by["fused"][2 * i + 1]
        qkv_base = pv(q, 1)
        assert pv(k, 1) - qkv_base == Q_DIM * BF16, "k 视图不在 qkv+8192"
        assert pv(rope, 1) == pv(q, 0) and pv(rope, 2) == pv(k, 0), \
            "rope 读的不是 normed q/k"
        assert pv(cache, 0) == pv(k, 0), "cache 的 key 不是 normed k"
        assert pv(cache, 1) == qkv_base + (Q_DIM + KV_DIM) * BF16, \
            "v 视图不在 qkv+10240"
        assert pv(cache, 3) - pv(cache, 2) == V_BYTE_OFF, "KV 池 k/v 间距非 256B"
        assert pv(uni, 1) == pv(q, 0), "attention 的 query 不是 normed q"
        assert pv(uni, 2) == pv(cache, 2) and pv(uni, 3) == pv(cache, 3), \
            "attention 与 cache 的 KV 池不一致"
        assert pv(uni, 7) == pv(cache, 5) and pv(uni, 8) == pv(cache, 6), \
            "attention 与 cache 的 k/v scale 不一致"
        assert pv(uni, 17) == pv(uni, 5), "unified [17] 不是 seq_lens 复用"
        assert [pv(red, j) for j in (1, 2, 3)] == [pv(uni, j) for j in (26, 27, 28)], \
            "reduce 读的不是 unified 写的 segm 缓冲"
        assert pv(red, 0) == pv(uni, 0), "reduce 输出与 unified 占位输出不同址"
        assert pv(red, 4) == pv(uni, 5) and pv(red, 9) == pv(uni, 24), \
            "reduce 的 seq_lens/cu_seqlens 与 unified 不一致"
        assert pv(post, 2) == residual and pv(nxt, 2) == residual, "residual 漂移"
        for r, wi in [(q, 7), (k, 7), (post, 3), (nxt, 3), (cache, 5), (cache, 6)]:
            weights.add(pv(r, wi))
        for r, ei in [(q, 9), (k, 9), (post, 4), (nxt, 4)]:
            assert pv(r, ei) == eps, "eps 不一致"
    assert len(weights) == 6 * LAYERS, "逐层权重指针有重合"
    scale = struct.unpack("<f", struct.pack("<I", pv(by["unified"][0], 6)))[0]
    return struct.unpack("<f", struct.pack("<I", eps))[0], scale


def check_prefill(prefills, by_dec):
    """真实 prefill forward 证伪 prefill program 的连线与 2D 实例几何。
    返回 (symbol, block, smem)。decode 时 tokens=1 掩盖的三个字面量在这里
    现形：head-norm 输入 stride=QKV_DIM（融合 qkv 行距）、head-norm [10]
    = tokens*heads（需要表达式标量）、cache value stride=QKV_DIM。"""
    uni_dec = by_dec["unified"][0]
    ref = None
    for t, fwd in prefills:
        by = {k: [] for k in SYMS}
        for r in fwd:
            for tag, pat in SYMS.items():
                if pat in r["symbol"] and isinstance(r.get("params"), list):
                    by[tag].append(r)
                    break
        assert len(by["unified"]) == LAYERS and len(by["reduce"]) == 0, \
            "prefill forward 不该出现 reduce_segments"
        u = by["unified"][0]
        assert len(u["params"]) == 28, "不是 2D prefill 实例"
        # 2D grid = [ceil_div(tokens, BLOCK_Q), kv_heads, 1]
        assert u["grid"] == [-(-t // BLOCK_Q), KV_HEADS, 1], (t, u["grid"])
        # 接口即 2D launch ABI：26 个前缀参数 + 两个尾部 i64 与 decode 3D
        # 实例的标量逐位一致（segm 三参恰好是被裁掉的实现细节）
        for j in list(range(9, 17)) + list(range(18, 24)) + [25]:
            assert pv(u, j) == pv(uni_dec, j), (j, pv(u, j), pv(uni_dec, j))
        assert pv(u, 6) == pv(uni_dec, 6), "softmax scale 不一致"
        assert pv(u, 17) == pv(u, 5), "unified [17] 不是 seq_lens 复用"
        q, k = by["rms_head"][0], by["rms_head"][1]
        assert pv(q, 3) == QKV_DIM and pv(k, 3) == QKV_DIM, \
            "head-norm 输入 stride 不是融合 qkv 行距"
        assert pv(q, 10) == t * HEADS and pv(k, 10) == t * KV_HEADS, \
            "head-norm [10] 不是 tokens*heads"
        assert pv(u, 1) == pv(by["rope"][0], 1), "attention query 不是 roped q"
        assert pv(by["cache"][0], 8) == QKV_DIM, \
            "cache value stride 不是融合 qkv 行距"
        ref = u
    return ref["symbol"], ref["block"], ref["dynamic_shared_mem_bytes"]


def unified_regs(recs, want):
    """28 参 unified launch 的 num_regs 集合，按 grid 谓词过滤。"""
    regs = {r["attributes"]["num_regs"] for r in recs
            if "kernel_unified_attention" in r["symbol"]
            and isinstance(r.get("params"), list) and len(r["params"]) == 28
            and want(r["grid"])}
    return regs


def check_spec(spec_dir, primary_jsonl, pf):
    """spec capture 证伪 dspark 布线，并消歧两个同 ABI 的 unified 实例。

    28 参 2D unified 有两份编译（causal / non-causal），symbol、参数布局、
    block、smem 逐位相同——静态 ABI 无法区分，唯一可见差异是 num_regs。
    primary dump（无投机）只含 causal；spec dump 里 grid=[2,8,1]（7 query）
    的 draft launch 是 non-causal。返回 (causal_regs, draft_regs)。"""
    _, pf_block, pf_smem = pf
    recs = mc.load(str(pathlib.Path(spec_dir) / "launches.jsonl"))
    causal = unified_regs(mc.load(primary_jsonl), lambda g: True)
    assert len(causal) == 1, f"primary dump 的 2D 实例 regs 不唯一: {causal}"
    causal_regs = causal.pop()

    # 一轮投机窗口内（rejection kernel 之间）：grid=[2,8,1]（ceil(7/4)）是
    # draft 的 5 层，grid=[3,8,1] 是 verify 的 36 层——短 prompt 的 prefill
    # 也会出 grid 2，必须限定在轮内看
    marks0 = [i for i, r in enumerate(recs) if r["symbol"] == "_rejection_kernel"]
    assert len(marks0) >= 2, "spec dump 里投机轮数不足"
    rnd0 = recs[marks0[-2] + 1:marks0[-1] + 1]
    draft = unified_regs(rnd0, lambda g: g[0] == 2)
    assert len(draft) == 1 and causal_regs not in draft, \
        f"draft 实例 regs 与 causal 未分离: causal={causal_regs} draft={draft}"
    draft_regs = draft.pop()
    # verify（8 token，vLLM grid [3,8,1] 是超配 padding）走的是 causal 实例——
    # 这是 verify 复用 attn_prefill 的实证
    vregs = unified_regs(rnd0, lambda g: g[0] == 3)
    assert vregs == {causal_regs}, f"verify 实例不是 causal: {vregs}"
    for r in recs:
        if "kernel_unified_attention" in r["symbol"] \
                and isinstance(r.get("params"), list) and len(r["params"]) == 28:
            assert r["block"] == pf_block \
                and r["dynamic_shared_mem_bytes"] == pf_smem, \
                "两实例 block/smem 与 prefill 不一致——ABI 消歧假设被推翻"

    # 一轮投机（最后两个 _rejection_kernel 之间）的 precompute 结构：
    # 融合 KV GEMM 之后逐层 5 次 reshape_and_cache；rope 的 key 指针为 NULL
    # （证实 K-only rope；我们用 num_kv=0 达成同一语义）；grouped k_norm 的
    # [3]=KV_DIM 打包行距、[8]=HEAD_DIM 逐层权重步长（证实逐层权重选择）
    marks = [i for i, r in enumerate(recs) if r["symbol"] == "_rejection_kernel"]
    assert len(marks) >= 2, "spec dump 里投机轮数不足"
    rnd = recs[marks[-2] + 1:marks[-1] + 1]
    caches = [r for r in rnd if "reshape_and_cache" in r["symbol"]
              and isinstance(r.get("params"), list)]
    # 一轮 = draft forward 5 层各一次 + verify 36 层各一次 + precompute 5 次
    assert len(caches) == DRAFT_LAYERS + LAYERS + DRAFT_LAYERS, len(caches)
    ropes = [r for r in rnd if "rotary_embedding" in r["symbol"]
             and isinstance(r.get("params"), list)]
    null_key = [r for r in ropes if pv(r, 2) == 0]
    assert len(null_key) == 1, "precompute 的 K-only rope（key=NULL）没找到"
    assert pv(null_key[0], 9) == KV_HEADS and pv(null_key[0], 6) == 0
    knorms = [r for r in rnd if "rms_norm_kernel" in r["symbol"]
              and isinstance(r.get("params"), list) and len(r["params"]) == 12
              and pv(r, 3) == KV_DIM]
    assert len(knorms) == 1, "grouped k_norm（打包 [L,n,8,128] 布局）没找到"
    assert pv(knorms[0], 8) == HEAD_DIM, "k_norm 逐层权重步长不是 head_dim"
    return causal_regs, draft_regs


def find_unified_module(dump_dir, want_regs):
    """在 dump 的 module cubin 里找 kernel_unified_attention REG==want_regs
    的唯一文件，返回 (basename, sha256)。"""
    cuobjdump = pathlib.Path(
        __import__("os").environ.get("CUDA_HOME", "/usr/local/cuda")) \
        / "bin" / "cuobjdump"
    hits = []
    for mod in sorted(pathlib.Path(dump_dir).glob("module_*.cubin")):
        out = subprocess.run([str(cuobjdump), "-res-usage", str(mod)],
                             capture_output=True, text=True).stdout
        take = False
        for line in out.splitlines():
            if line.strip().startswith("Function "):
                take = line.split()[1].rstrip(":") == "kernel_unified_attention"
            elif take and "REG:" in line:
                if int(line.split("REG:")[1].split()[0]) == want_regs:
                    hits.append(mod)
                take = False
    # 同一 module 可能被记录多次（重复 load）——按内容去重
    shas = {hashlib.sha256(m.read_bytes()).hexdigest(): m for m in hits}
    assert len(shas) == 1, \
        f"{dump_dir} 里 REG={want_regs} 的 unified module 内容不唯一: {hits}"
    sha, mod = next(iter(shas.items()))
    return mod.name, sha, mod


def var(s):
    return {"var": s}


def _e(e):
    """Expression form of a var arg: shapes/grids name a var bare."""
    return e["var"] if isinstance(e, dict) and "var" in e else e


def mul(e, c):
    return {"mul": [_e(e), c]}


def cdiv(e, c):
    return {"ceil_div": [_e(e), c]}


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


def d(label, op, args):
    return {"label": label, "op": op, "args": args}


def a(i):
    return {"param": i}


def scr(name):
    return {"scratch": name}


def step(symbol, params, block, grid, args, shared_mem=None, cubin=None,
         sha256=None):
    s = {"entry": symbol, "params": params, "block": block,
         "grid": [_e(g) for g in grid], "args": args}
    if shared_mem is not None:
        s["shared_mem"] = shared_mem
    if cubin is not None:
        s["cubin"] = cubin
    if sha256 is not None:
        s["sha256"] = sha256
    return s


def single(symbol, params, block, grid, shared_mem=None, cubin=None,
           sha256=None):
    """单步实现，恒等布线：接口即该核的 launch ABI。"""
    return {"params": params,
            "impl": {"launches": [step(symbol, params, block, grid,
                                    [a(i) for i in range(len(params))],
                                    shared_mem, cubin, sha256)]}}


# 结构输入的先验（domain）：接模型的人才知道 buffer<i32> 是页表不是激活。
# runtime 写入时校验；kern test 据此合成合法值 + 检查产出。激活不声明。
TOKEN_DOMAIN = {"index_into": "model.embed_tokens.weight"}
DOMAINS = {
    "token_ids": TOKEN_DOMAIN,
    "positions": {"index_into": "rope.cos_sin_cache"},
    "slot_mapping": {"index_into": "kv"},
    "block_table": {"index_into": "kv", "stride": 16},  # vLLM block_size
    "seq_lens": {"min": 1},
    "cu_seqlens_q": {"min": 0, "max": "tokens", "monotone": True},
    "next_token": TOKEN_DOMAIN,
    "kv_scales": {"min": 0.0},
    "anchor_token": TOKEN_DOMAIN,
    "draft_tokens": TOKEN_DOMAIN,
    "verify_tokens": TOKEN_DOMAIN,
    "draft_ids": TOKEN_DOMAIN,
    "verify_ids": TOKEN_DOMAIN,
    "nacc": {"min": 1, "max": VERIFY_TOKENS},
}


# silu_mul from the HF kernel hub (kernels-community/activation, packed
# bf16x2 variant): the shipped qwen3-4b.json uses it; the *-silu-mined.json
# fixture keeps the mined vLLM instance so the two form kern-test's A/B pair.
HUB_SILU = {
    "cubin": "hf:kernels-community/activation/build/torch29-cxx11-cu130-aarch64-linux/"
             "activation/_activation_320b408.abi3.so",
    "sha256": "73748b54059552f5983322f7dedc36ed349b38ad6fb9318301bb4965b1fe49aa",
    "entry": "_ZN4vllm18act_and_mul_kernelIN3c108BFloat16EXadL_ZNS_11silu_kernelIS2_EET_RKS4_EELb1EEEvPS4_PS5_i",
    "params": ["out buffer<bf16>", "in buffer<bf16>", "i32"],
}


def build(by, eps, scale, pf, pins, spec=False, silu="mined"):
    pf_sym, pf_block, pf_smem = pf
    causal_cubin, causal_sha = pins["causal"]

    def mp(tag):
        """`cubin` + `sha256` of the dump module a mined launch pins."""
        cubin, sha = pins[tag]
        return {"cubin": cubin, "sha256": sha}

    buffers = {
        "token_ids": {"dtype": "i64", "shape": ["tokens"], "kind": "input", "fill": "token"},
        "positions": {"dtype": "i64", "shape": ["tokens"], "kind": "input", "fill": "position"},
        "slot_mapping": {"dtype": "i64", "shape": ["tokens"], "kind": "input", "fill": "slot"},
        # attention 元数据按序列：block_table 一行一个序列（行距 MAX_BLOCKS
        # 就是 kernel 收到的 block_table_stride），seq_lens[i] = 序列 i 已见
        # token 数（含本次），cu_seqlens_q = 各序列 query 行数的前缀和
        # （seqs+1 项；shape 维度不能是表达式，按上界声明，尾部不被读）。
        # prefill 恒 seqs=1：只填第 0 行 / 前两项
        "block_table": {"dtype": "i32", "shape": ["seqs", MAX_BLOCKS], "kind": "input"},
        "seq_lens": {"dtype": "i32", "shape": ["seqs"], "kind": "input", "fill": "seq_len"},
        "cu_seqlens_q": {"dtype": "i32", "shape": [MAX_SEQS + 1], "kind": "input", "fill": "cu_seqlens"},
        # logits/next_token 一序列一行（decode 的 tokens = seqs），按 seqs
        # 上界分配（256 × 151936 × 2 B ≈ 74 MB），不挂 tokens（CHUNK_MAX
        # 上界要多付 ~600 MB）
        "logits": {"dtype": "bf16", "shape": ["seqs", VOCAB], "kind": "workspace"},
        "next_token": {"dtype": "i64", "shape": ["seqs"], "kind": "output", "fill": "tokens"},
    }
    for name, shape in {
        "residual": ["tokens", HIDDEN],
        "x": ["tokens", HIDDEN],
        "y": ["tokens", HIDDEN],
        "qkv": ["tokens", QKV_DIM],
        "q_n": ["tokens", Q_DIM],
        "k_n": ["tokens", KV_DIM],
        "attn_out": ["tokens", Q_DIM],
        "gate_up": ["tokens", 2 * FFN],
        "ffn_act": ["tokens", FFN],
    }.items():
        buffers[name] = {"dtype": "bf16", "shape": shape, "kind": "workspace"}

    def weight(name, shape, dtype="bf16"):
        buffers[name] = {"dtype": dtype, "shape": shape, "kind": "weight"}

    weight("model.embed_tokens.weight", [VOCAB, HIDDEN])
    weight("model.norm.weight", [HIDDEN])
    weight("lm_head.weight", [VOCAB, HIDDEN])
    weight("rope.cos_sin_cache", [MAX_POS, HEAD_DIM])
    weight("kv_scales", [2 * LAYERS], "f32")
    for i in range(LAYERS):
        p = f"model.layers.{i}."
        weight(p + "input_layernorm.weight", [HIDDEN])
        weight(p + "post_attention_layernorm.weight", [HIDDEN])
        weight(p + "self_attn.qkv_proj.weight", [QKV_DIM, HIDDEN])
        weight(p + "self_attn.q_norm.weight", [HEAD_DIM])
        weight(p + "self_attn.k_norm.weight", [HEAD_DIM])
        weight(p + "self_attn.o_proj.weight", [HIDDEN, Q_DIM])
        weight(p + "mlp.gate_up_proj.weight", [2 * FFN, HIDDEN])
        weight(p + "mlp.down_proj.weight", [HIDDEN, FFN])

    if spec:
        # DSpark：draft 与 target 几何同构，激活 buffer 全部复用；只加投机
        # 专属的交换面。fc_out 是 carry：verify/prefill/decode_spec 的 tap 写，
        # draft_precompute 读——跨 program 的交接棒，顺序是 caller 契约。
        # 一序列一行：draft/verify 都是 seqs 段 varlen（每段 7/8 行），
        # tokens = 7·seqs / 8·seqs；seqs=1 就是单序列的老契约
        buffers["anchor_token"] = {"dtype": "i64", "shape": ["seqs"],
                                   "kind": "input", "fill": "token"}
        buffers["draft_tokens"] = {"dtype": "i64", "shape": ["seqs", BLOCK_TOKENS],
                                   "kind": "output"}
        buffers["verify_tokens"] = {"dtype": "i64", "shape": ["seqs", VERIFY_TOKENS],
                                    "kind": "output", "fill": "tokens"}
        # how many of verify_tokens' rows a sequence takes: accepted + 1
        buffers["nacc"] = {"dtype": "i32", "shape": ["seqs"], "kind": "output", "fill": "count"}
        # the round's device-written rows: draft's ids ([anchor, mask x6]
        # from the anchor the caller staged) and verify's ([anchor, d0..d5])
        buffers["draft_ids"] = {"dtype": "i64", "shape": ["tokens"], "kind": "carry"}
        buffers["verify_ids"] = {"dtype": "i64", "shape": ["tokens"], "kind": "carry"}
        # the last prompt row of a prefill chunk, for its head
        buffers["final_x"] = {"dtype": "bf16", "shape": [1, HIDDEN], "kind": "workspace"}
        buffers["logits_blk"] = {"dtype": "bf16",
                                 "shape": ["tokens", VOCAB],
                                 "kind": "workspace"}
        buffers["fc_out"] = {"dtype": "bf16", "shape": ["tokens", HIDDEN],
                             "kind": "carry"}
        buffers["kv_flat"] = {"dtype": "bf16", "shape": ["tokens", DRAFT_KV_DIM],
                              "kind": "workspace"}
        buffers["membed"] = {"dtype": "bf16", "shape": ["seqs", MARKOV_RANK],
                             "kind": "workspace"}
        weight("draft.embed_tokens.weight", [VOCAB, HIDDEN])
        weight("draft.lm_head.weight", [VOCAB, HIDDEN])
        weight("draft.norm.weight", [HIDDEN])
        weight("draft.hidden_norm.weight", [HIDDEN])
        weight("draft.fused_kv.weight", [DRAFT_KV_DIM, HIDDEN])
        weight("draft.markov_w1", [VOCAB, MARKOV_RANK])
        weight("draft.markov_w2.weight", [VOCAB, MARKOV_RANK])
        weight("draft.kv_scales", [2 * DRAFT_LAYERS], "f32")
        for j in range(DRAFT_LAYERS):
            weight(f"draft.fc.{j}.weight", [HIDDEN, HIDDEN])
            p = f"draft.layers.{j}."
            weight(p + "input_layernorm.weight", [HIDDEN])
            weight(p + "post_attention_layernorm.weight", [HIDDEN])
            weight(p + "self_attn.qkv_proj.weight", [QKV_DIM, HIDDEN])
            weight(p + "self_attn.q_norm.weight", [HEAD_DIM])
            weight(p + "self_attn.k_norm.weight", [HEAD_DIM])
            weight(p + "self_attn.o_proj.weight", [HIDDEN, Q_DIM])
            weight(p + "mlp.gate_up_proj.weight", [2 * FFN, HIDDEN])
            weight(p + "mlp.down_proj.weight", [HIDDEN, FFN])

    def blk(tag):
        return by[tag][0]["block"]

    def smem(tag):
        return by[tag][0]["dynamic_shared_mem_bytes"]

    T = var("tokens")
    S = var("seqs")
    RMS_PARAMS = ["out buffer<bf16>", "in buffer<bf16>", "i64", "i64", "i64",
                  "i64", "i64", "in buffer<bf16>", "i64", "f32", "i32", "i32"]
    # unified 的完整 launch ABI（31 参数）；接口砍掉三个 segm 分部和缓冲
    # （26/27/28），它们是实现细节，降为 impl scratch
    UNIFIED_PARAMS = [
        "out buffer<bf16>",  # 3D 实例不写它，ABI 要求非空指针；reduce 写
        "in buffer<bf16>", "inout state", "inout state",
        "in buffer<i32>", "in buffer<i32>", "f32",
        "in buffer<f32>", "in buffer<f32>", "f32", "f32",
        "i64", "i64", "i64", "i64", "i64", "i64",
        "in buffer<i32>", "i64", "i64", "i64", "i64", "i64",
        "i64", "in buffer<i32>", "i32", "out buffer<f32>",
        "out buffer<f32>", "out buffer<f32>", "i64", "i64"]
    ATTN_IFACE = UNIFIED_PARAMS[:26] + UNIFIED_PARAMS[29:]
    REDUCE_PARAMS = ["out buffer<bf16>", "in buffer<f32>", "in buffer<f32>",
                     "in buffer<f32>", "in buffer<i32>", "f32", "i64", "i64",
                     "i64", "in buffer<i32>", "i64", "i64"]

    # 真实挖到的 kernel ABI。kernel = 接口 + impl（可整体替换的实现）：
    # 多数是单步恒等布线；attn / argmax 是"微程序 + 自报 scratch"的两步实现，
    # 分部和缓冲不再泄漏进调用方的 buffer 表。
    kernels = {
        "rms_norm": single(by["rms"][0]["symbol"], RMS_PARAMS, blk("rms"),
                           [T, 1, 1], **mp("rms")),
        # 同一 head-norm 核的两个实例化：grid 随 head 数烘焙在实现里
        "rms_norm_qhead": single(by["rms_head"][0]["symbol"], RMS_PARAMS,
                                 blk("rms_head"), [mul(T, HEADS), 1, 1], **mp("rms_head")),
        "rms_norm_khead": single(by["rms_head"][0]["symbol"], RMS_PARAMS,
                                 blk("rms_head"), [mul(T, KV_HEADS), 1, 1], **mp("rms_head")),
        "rotary_embedding": single(
            by["rope"][0]["symbol"],
            ["in buffer<i64>", "inout buffer<bf16>", "inout buffer<bf16>",
             "in buffer<bf16>", "i32", "i64", "i64", "i64", "i32", "i32",
             "i32", "i64", "u8"],
            blk("rope"), [T, 1, 1], **mp("rope")),
        "reshape_and_cache": single(
            by["cache"][0]["symbol"],
            ["in buffer<bf16>", "in buffer<bf16>", "inout state",
             "inout state", "in buffer<i64>", "in buffer<f32>",
             "in buffer<f32>", "i64", "i64", "i64", "i64", "i64", "i64",
             "i64", "i64", "i64"],
            blk("cache"), [T, 1, 1], **mp("cache")),
        # decode attention：3D split-KV 微程序。decode 恒 tokens=1，grid 与
        # scratch 定常（scratch 若挂 tokens 会按 CHUNK_MAX 上界多付 ~500MB）
        "attn": {
            "params": ATTN_IFACE,
            "impl": {
                "scratch": {
                    "segm_out": {"dtype": "f32",
                                 "shape": [1, HEADS, NUM_SEGMENTS, HEAD_DIM]},
                    "segm_max": {"dtype": "f32",
                                 "shape": [1, HEADS, NUM_SEGMENTS]},
                    "segm_expsum": {"dtype": "f32",
                                    "shape": [1, HEADS, NUM_SEGMENTS]},
                },
                "launches": [
                    step(by["unified"][0]["symbol"], UNIFIED_PARAMS,
                         blk("unified"), [1, KV_HEADS, NUM_SEGMENTS],
                         [a(i) for i in range(26)]
                         + [scr("segm_out"), scr("segm_max"), scr("segm_expsum"),
                            a(26), a(27)],
                         shared_mem=smem("unified"), **mp("unified")),
                    step(by["reduce"][0]["symbol"], REDUCE_PARAMS,
                         blk("reduce"), [1, HEADS, 1],
                         [a(0), scr("segm_out"), scr("segm_max"),
                          scr("segm_expsum"), a(5), f32(1.0), i64(Q_DIM),
                          i64(HEAD_DIM), i64(MAX_BLOCKS), a(24), i64(0), i64(0)],
                         shared_mem=smem("reduce"), **mp("reduce")),
                ],
            },
        },
        # prefill attention：同一接口的另一份实现——2D 实例单步无 scratch
        # （28 参 launch ABI 恰好就是接口本身，这是接口切分正确的实证）。
        # cubin 必须钉定：non-causal 的 draft 实例与它 symbol/ABI/block/smem
        # 逐位相同，仅 num_regs 可分——按参数布局解析会二义
        "attn_prefill": single(pf_sym, ATTN_IFACE, pf_block,
                               [cdiv(T, BLOCK_Q), KV_HEADS, 1],
                               shared_mem=pf_smem, cubin=causal_cubin,
                               sha256=causal_sha),
        # batched decode attention：还是这份 2D causal 实例（vLLM 自己在
        # num_seqs 超过 3D 阈值时 decode 就走它；挖到的 reduce_segments 是
        # num_seqs=1 特化实例，3D 路只能 bs=1）。grid.x 要盖住 vLLM 的
        # q-block 索引空间 tokens//BLOCK_Q + num_seqs（seq i 的首 block 在
        # cu_seqlens_q[i]//BLOCK_Q + i）；表达式集合没有两个 var 相加，用
        # ceil(5·tokens/4) ≥ tokens//4 + seqs（seqs ≤ tokens 的契约下恒成立），
        # 多出的 block 在核内按 query 行数提前返回
        "attn_batch": single(pf_sym, ATTN_IFACE, pf_block,
                             [cdiv(mul(T, 5), BLOCK_Q), KV_HEADS, 1],
                             shared_mem=pf_smem, cubin=causal_cubin,
                             sha256=causal_sha),
        "silu_mul": single(
            by["silu"][0]["symbol"],
            ["out buffer<bf16>", "in buffer<bf16>", "i32", "i32", "f32", "i32"],
            blk("silu"), [T, 1, 1], **mp("silu")) if silu == "mined" else {
            # same interface, hub impl: (out, in, d) — the caller's extra
            # scalars (stride/offset/scale) are dropped on the floor
            "params": ["out buffer<bf16>", "in buffer<bf16>", "i32", "i32", "f32", "i32"],
            "impl": {"launches": [dict(HUB_SILU, block=[1024, 1, 1], grid=[_e(T), 1, 1],
                                    args=[a(0), a(1), a(2)])]}},
        "fused_add_rms_norm": single(
            by["fused"][0]["symbol"],
            ["inout buffer<bf16>", "i64", "inout buffer<bf16>",
             "in buffer<bf16>", "f32", "i32", "i32", "i64"],
            blk("fused"), [T, 1, 1], **mp("fused")),
        "embedding": single(
            "kern_embedding_i64_bf16",
            ["in buffer<i64>", "in buffer<bf16>", "out buffer<bf16>",
             "i32", "i32"],
            [256, 1, 1], [T, 1, 1], **hw("embedding")),
        # c[m,n] = a[m,k] @ w[n,k]^T；runtime 按 extern: 前缀特判走 cublasLt
        "gemm": single(
            "extern:cublaslt_bf16_tn",
            ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>",
             "i32", "i32", "i32"],
            [1, 1, 1], [1, 1, 1]),
        # greedy 采样下沉：手写两段式（tools/kernels-src/argmax.cu）。分部
        # 缓冲与两次 launch 全在 impl 内，接口只有 (logits, tokens_out, n)。
        # 行数 = tokens（核天然多行：grid.x 是行号）：decode 以 env 1 跑一行，
        # verify 以 env 8 一次出 8 行
        "argmax": {
            "params": ["in buffer<bf16>", "out buffer<i64>", "i32"],
            "impl": {
                "scratch": {
                    "pmax": {"dtype": "f32", "shape": ["tokens", 64]},
                    "pidx": {"dtype": "i32", "shape": ["tokens", 64]},
                },
                "launches": [
                    step("kern_argmax_partial_bf16",
                         ["in buffer<bf16>", "out buffer<f32>",
                          "out buffer<i32>", "i32"],
                         [1024, 1, 1], [T, 64, 1],
                         [a(0), scr("pmax"), scr("pidx"), a(2)],
                         **hw("argmax")),
                    step("kern_argmax_final_i64",
                         ["in buffer<f32>", "in buffer<i32>",
                          "out buffer<i64>", "i32"],
                         [64, 1, 1], [T, 1, 1],
                         [scr("pmax"), scr("pidx"), a(1), i32(64)],
                         **hw("argmax")),
                ],
            },
        },
    }

    if spec:
        draft_cubin, draft_sha = pins["draft"]
        # draft attention：又一份同接口实现——non-causal 编译（DFlash 的
        # 7 个 query 全员互见），钉定到自己的 cubin
        # grid.x 盖住 vLLM 的 q-block 索引空间（seq i 的首 block 在
        # cu_seqlens_q[i]//BLOCK_Q + i）：7 行/段 → 7·seqs//4 + seqs ≤ ⌈11·seqs/4⌉
        kernels["attn_draft"] = single(
            pf_sym, ATTN_IFACE, pf_block, [cdiv(mul(S, 11), BLOCK_Q), KV_HEADS, 1],
            shared_mem=pf_smem, cubin=draft_cubin, sha256=draft_sha)
        # verify：causal 实例，7 行/段 → 7·seqs//4 + seqs ≤ 3·seqs
        kernels["attn_verify"] = single(
            pf_sym, ATTN_IFACE, pf_block, [mul(S, 3), KV_HEADS, 1],
            shared_mem=pf_smem, cubin=causal_cubin, sha256=causal_sha)
        # the round's glue (tools/kernels-src/spec_round.cu): draft's ids
        # from the anchor and the mask, verify's from draft's output, and the
        # count of verify rows taken (the matched prefix + 1)
        kernels["splice_draft"] = single(
            "kern_splice_draft", ["in buffer<i64>", "out buffer<i64>", "i32", "i64"],
            [32, 1, 1], [S, 1, 1], **hw("spec_round"))
        kernels["splice_verify"] = single(
            "kern_splice_verify", ["in buffer<i64>", "in buffer<i64>", "out buffer<i64>", "i32", "i32"],
            [32, 1, 1], [S, 1, 1], **hw("spec_round"))
        kernels["spec_count"] = single(
            "kern_spec_count", ["in buffer<i64>", "in buffer<i64>", "out buffer<i32>", "i32", "i32"],
            [32, 1, 1], [S, 1, 1], **hw("spec_round"))
        # prefill's head over its last row (tools/kernels-src/copy_rows.cu;
        # `rows-1` is not in the expression set, so the kernel takes `rows`)
        # and the one-row argmax it feeds
        kernels["last_row"] = single(
            "kern_last_row_bf16", ["out buffer<bf16>", "in buffer<bf16>", "i32", "i32", "i32"],
            [256, 1, 1], [1, 1, 1], **hw("copy_rows"))
        kernels["argmax_row"] = {
            "params": ["in buffer<bf16>", "out buffer<i64>", "i32"],
            "impl": {
                "scratch": {
                    "pmax": {"dtype": "f32", "shape": [1, 64]},
                    "pidx": {"dtype": "i32", "shape": [1, 64]},
                },
                "launches": [
                    step("kern_argmax_partial_bf16",
                         ["in buffer<bf16>", "out buffer<f32>",
                          "out buffer<i32>", "i32"],
                         [1024, 1, 1], [1, 64, 1],
                         [a(0), scr("pmax"), scr("pidx"), a(2)],
                         **hw("argmax")),
                    step("kern_argmax_final_i64",
                         ["in buffer<f32>", "in buffer<i32>",
                          "out buffer<i64>", "i32"],
                         [64, 1, 1], [1, 1, 1],
                         [scr("pmax"), scr("pidx"), a(1), i32(64)],
                         **hw("argmax")),
                ],
            },
        }
        # markov 链第 t 步作用在每序列的第 t 行——行集合 {s·7+t}：行距 7·V
        # 的 argmax、下标距 7 的 gather，grid.x = seqs，步的字节偏移烘在
        # buffer 参数里
        kernels["argmax_rows"] = {
            "params": ["in buffer<bf16>", "i64", "out buffer<i64>", "i32", "i32"],
            "impl": {
                "scratch": {
                    "pmax": {"dtype": "f32", "shape": ["seqs", 64]},
                    "pidx": {"dtype": "i32", "shape": ["seqs", 64]},
                },
                "launches": [
                    step("kern_argmax_rows_partial_bf16",
                         ["in buffer<bf16>", "i64", "out buffer<f32>",
                          "out buffer<i32>", "i32"],
                         [1024, 1, 1], [S, 64, 1],
                         [a(0), a(1), scr("pmax"), scr("pidx"), a(4)],
                         **hw("markov_rows")),
                    step("kern_argmax_rows_final_i64",
                         ["in buffer<f32>", "in buffer<i32>",
                          "out buffer<i64>", "i32", "i32"],
                         [64, 1, 1], [S, 1, 1],
                         [scr("pmax"), scr("pidx"), a(2), a(3), i32(64)],
                         **hw("markov_rows")),
                ],
            },
        }
        kernels["embedding_rows"] = single(
            "kern_embedding_rows_i64_bf16",
            ["in buffer<i64>", "i32", "in buffer<bf16>", "out buffer<bf16>",
             "i32"],
            [256, 1, 1], [S, 1, 1], **hw("markov_rows"))
        # c[m,n] += a[m,k] @ w[n,k]^T：β=1 累加版，喂 fc 分块和 markov 偏置
        kernels["gemm_acc"] = single(
            "extern:cublaslt_bf16_tn_acc",
            ["in buffer<bf16>", "in buffer<bf16>", "inout buffer<bf16>",
             "i32", "i32", "i32"],
            [1, 1, 1], [1, 1, 1])
        # 同上，第 7 参是 C 的行距（元素）：markov 链第 t 步只写每序列的第 t 行
        kernels["gemm_acc_ldc"] = single(
            "extern:cublaslt_bf16_tn_acc",
            ["in buffer<bf16>", "in buffer<bf16>", "inout buffer<bf16>",
             "i32", "i32", "i32", "i64"],
            [1, 1, 1], [1, 1, 1])

    def gemm(label, ab, w, c, m, n, k):
        return d(label, "gemm", [buf(ab), buf(w), buf(c), m, i32(n), i32(k)])

    def head_norm(label, kernel, out, off, w, heads):
        # 标量字面量与挖到的 launch 逐位一致。[3]=输入行距=QKV_DIM（q/k 视
        # 图都在融合 qkv 里，行距 6144——decode capture 里的 4096/1024 是
        # vLLM 自己的连续布局，tokens=1 时掩盖了差异，prefill capture 现形）；
        # [10]=tokens*heads（总 head 数上界），需要表达式标量
        return d(label, kernel,
                 [buf(out), buf("qkv", off), i64(HEAD_DIM), i64(QKV_DIM), i64(0),
                  i64(heads), i64(0), buf(w), i64(0), f32(eps),
                  expr(mul(T, heads)), i32(HEAD_DIM)])

    def fused(label, x, w):
        return d(label, "fused_add_rms_norm",
                 [buf(x), i64(HIDDEN), buf("residual"), buf(w), f32(eps), T,
                  i32(HIDDEN), i64(HIDDEN)])

    def forward(attn_kernel, tail, mp="model.", layers=LAYERS, kv_state="kv",
                block_stride=BLOCK_STRIDE, scales="kv_scales", taps=False,
                lm_head_w="lm_head.weight", final_norm_w="model.norm.weight",
                ids="token_ids"):
        """embed + 逐层的直线 dispatch 表，target/draft 共用（几何同构，
        激活 buffer 全部复用；差异全在权重名、层数、attention 实现、KV
        state 与收尾 tail）。tail: None=只落 KV（prefill）；"decode"=1 行
        logits+argmax；"verify"=8 行 logits+argmax；"draft"=7 行 logits +
        markov 链展开。taps=True 时在 layer 0/8/16/24/32 的 next_input_norm
        之后各放一个 fc 分块 GEMM（首块 β=0 初始化 fc_out，其余 β=1 累加）
        ——residual 此刻恰是 DSpark 要的 aux（hidden+residual）。"""
        # unified 的 num_seqs：3D 微程序（`attn`）的 reduce 是 num_seqs=1 特化
        # 实例，契约就是 bs=1，写死 1；2D 实例接 `seqs` var
        num_seqs = i32(1) if attn_kernel == "attn" else S
        ds = [
            d("embed", "embedding",
              [buf(ids), buf(mp + "embed_tokens.weight"),
               buf("residual"), T, i32(HIDDEN)]),
            d("l0.input_norm", "rms_norm",
              [buf("x"), buf("residual"), i64(HIDDEN), i64(0), i64(0), i64(0),
               i64(0), buf(mp + "layers.0.input_layernorm.weight"), i64(0),
               f32(eps), T, i32(HIDDEN)]),
        ]
        for i in range(layers):
            p = f"{mp}layers.{i}."
            l = f"l{i}."
            koff = i * LAYER_KV_BYTES
            ks, vs = buf(scales, i * 8), buf(scales, i * 8 + 4)
            last = i + 1 == layers
            ds += [
                gemm(l + "qkv_proj", "x", p + "self_attn.qkv_proj.weight",
                     "qkv", T, QKV_DIM, HIDDEN),
                head_norm(l + "q_norm", "rms_norm_qhead", "q_n", 0,
                          p + "self_attn.q_norm.weight", HEADS),
                head_norm(l + "k_norm", "rms_norm_khead", "k_n", Q_DIM * BF16,
                          p + "self_attn.k_norm.weight", KV_HEADS),
                d(l + "rope", "rotary_embedding",
                  [buf("positions"), buf("q_n"), buf("k_n"),
                   buf("rope.cos_sin_cache"),
                   i32(HEAD_DIM), i64(Q_DIM), i64(KV_DIM), i64(HEAD_DIM),
                   i32(HEADS), i32(KV_HEADS), i32(HEAD_DIM), i64(0), u8(0)]),
                d(l + "kv_write", "reshape_and_cache",
                  # [8]=value 行距=QKV_DIM（v 视图在融合 qkv 里；decode
                  # capture 的 1024 同样是 tokens=1 下的假常量）
                  [buf("k_n"), buf("qkv", (Q_DIM + KV_DIM) * BF16),
                   state(kv_state, koff), state(kv_state, koff + V_BYTE_OFF),
                   buf("slot_mapping"), ks, vs,
                   i64(KV_DIM), i64(QKV_DIM), i64(block_stride),
                   i64(2 * HEAD_DIM), i64(0), i64(0),
                   i64(KV_HEADS * 2 * HEAD_DIM), i64(0), i64(0)]),
                d(l + "attn", attn_kernel,
                  [buf("attn_out"), buf("q_n"),
                   state(kv_state, koff), state(kv_state, koff + V_BYTE_OFF),
                   buf("block_table"), buf("seq_lens"), f32(scale), ks, vs,
                   f32(1.0), f32(0.0),
                   i64(MAX_BLOCKS), i64(Q_DIM), i64(HEAD_DIM), i64(Q_DIM),
                   i64(HEAD_DIM), i64(0), buf("seq_lens"),
                   i64(block_stride), i64(KV_HEADS * 2 * HEAD_DIM),
                   i64(2 * HEAD_DIM),
                   i64(block_stride), i64(KV_HEADS * 2 * HEAD_DIM),
                   i64(2 * HEAD_DIM),
                   buf("cu_seqlens_q"), num_seqs,
                   i64(0), i64(0)]),
                gemm(l + "o_proj", "attn_out", p + "self_attn.o_proj.weight",
                     "y", T, HIDDEN, Q_DIM),
                fused(l + "post_attn_norm", "y",
                      p + "post_attention_layernorm.weight"),
                gemm(l + "gate_up", "y", p + "mlp.gate_up_proj.weight",
                     "gate_up", T, 2 * FFN, HIDDEN),
                d(l + "silu_mul", "silu_mul",
                  [buf("ffn_act"), buf("gate_up"), i32(FFN), i32(0), f32(1.0),
                   i32(0)]),
                gemm(l + "down_proj", "ffn_act", p + "mlp.down_proj.weight",
                     "x", T, HIDDEN, FFN),
            ]
            if last and tail is None:
                continue  # prefill：final_norm 喂 lm_head，没有 lm_head 就不需要
            ds.append(fused(l + ("final_norm" if last else "next_input_norm"),
                            "x",
                            f"{mp}layers.{i + 1}.input_layernorm.weight"
                            if not last else final_norm_w))
            if taps and i in TAPS:
                j = TAPS[i]
                ds.append(d(l + "fc_tap", "gemm" if j == 0 else "gemm_acc",
                            [buf("residual"), buf(f"draft.fc.{j}.weight"),
                             buf("fc_out"), T, i32(HIDDEN), i32(HIDDEN)]))
        if tail == "prefill":
            # the last row's token: a prefill that hands the first generated
            # token back (its taps are then in the draft KV before a round)
            ds.append(d("last_row", "last_row",
                        [buf("final_x"), buf("x"), i32(HIDDEN), i32(HIDDEN), T]))
            ds.append(gemm("lm_head", "final_x", lm_head_w, "logits",
                           i32(1), VOCAB, HIDDEN))
            ds.append(d("sample", "argmax_row",
                        [buf("logits"), buf("next_token"), i32(VOCAB)]))
        elif tail == "decode":
            # decode 的 tokens = seqs：x 的前 seqs 行各出一行 logits
            ds.append(gemm("lm_head", "x", lm_head_w, "logits",
                           S, VOCAB, HIDDEN))
            ds.append(d("sample", "argmax",
                        [buf("logits"), buf("next_token"), i32(VOCAB)]))
        elif tail == "verify":
            # 8·seqs 行 logits + 同样多行 argmax（argmax 行数 = env tokens）
            ds.append(gemm("lm_head", "x", lm_head_w, "logits_blk",
                           T, VOCAB, HIDDEN))
            ds.append(d("sample", "argmax",
                        [buf("logits_blk"), buf("verify_tokens"), i32(VOCAB)]))
        elif tail == "draft":
            # 7 行 base logits（m=T，env tokens=7），然后 markov 链展开：
            # embedding_row 取 markov_w1[prev] → gemm_acc 把 markov_w2 偏置
            # 累进该行 → argmax_row 出 draft token，喂下一步的 prev
            ds.append(gemm("lm_head", "x", lm_head_w, "logits_blk",
                           T, VOCAB, HIDDEN))
            # 每步作用在每序列的第 t 行（行集合 {s·7+t}，行距 7·V）；prev
            # 是各序列自己链上的上一个 token（anchor_token[s] 或
            # draft_tokens[s, t-1]，下标距 1 / 7）
            for t in range(BLOCK_TOKENS):
                prev, stride = (buf("anchor_token"), 1) if t == 0 \
                    else (buf("draft_tokens", (t - 1) * 8), BLOCK_TOKENS)
                ds += [
                    d(f"markov{t}.embed", "embedding_rows",
                      [prev, i32(stride), buf("draft.markov_w1"), buf("membed"),
                       i32(MARKOV_RANK)]),
                    d(f"markov{t}.bias", "gemm_acc_ldc",
                      [buf("membed"), buf("draft.markov_w2.weight"),
                       buf("logits_blk", t * VOCAB * BF16),
                       S, i32(VOCAB), i32(MARKOV_RANK),
                       i64(BLOCK_TOKENS * VOCAB)]),
                    d(f"markov{t}.sample", "argmax_rows",
                      [buf("logits_blk", t * VOCAB * BF16),
                       i64(BLOCK_TOKENS * VOCAB),
                       buf("draft_tokens", t * 8), i32(BLOCK_TOKENS),
                       i32(VOCAB)]),
                ]
        return ds

    def draft_precompute():
        """target 隐状态投影成 draft 的 context KV（DSpark 的关键结构）：
        hidden_norm(fc_out) → 融合 KV GEMM [n,10240] → 逐层：k_norm（打包
        写进 k_n）→ K-only rope（num_kv=0 跳过 key——vLLM 用 key=NULL，
        schema 无空指针，同一语义）→ reshape_and_cache 进 draft_kv。
        跑在 env tokens=n（fc_out 的有效行数）；positions/slot_mapping 沿用
        产生这批 fc_out 的那次 forward 的输入，caller 无需重写。"""
        ds = [
            d("hidden_norm", "rms_norm",
              [buf("x"), buf("fc_out"), i64(HIDDEN), i64(0), i64(0), i64(0),
               i64(0), buf("draft.hidden_norm.weight"), i64(0), f32(eps), T,
               i32(HIDDEN)]),
            gemm("fused_kv", "x", "draft.fused_kv.weight", "kv_flat",
                 T, DRAFT_KV_DIM, HIDDEN),
        ]
        for j in range(DRAFT_LAYERS):
            koff = j * LAYER_KV_BYTES
            ks = buf("draft.kv_scales", j * 8)
            vs = buf("draft.kv_scales", j * 8 + 4)
            ds += [
                # kv_flat 行内布局 [k0 v0 k1 v1 ...]（融合权重按层 cat [k;v]）：
                # 层 j 的 K 在列 j*2048、V 在列 j*2048+1024，行距 10240
                d(f"ctx{j}.k_norm", "rms_norm_khead",
                  [buf("k_n"), buf("kv_flat", j * 2 * KV_DIM * BF16),
                   i64(HEAD_DIM), i64(DRAFT_KV_DIM), i64(0), i64(KV_HEADS),
                   i64(0), buf(f"draft.layers.{j}.self_attn.k_norm.weight"),
                   i64(0), f32(eps), expr(mul(T, KV_HEADS)), i32(HEAD_DIM)]),
                d(f"ctx{j}.rope", "rotary_embedding",
                  [buf("positions"), buf("k_n"), buf("k_n"),
                   buf("rope.cos_sin_cache"), i32(HEAD_DIM), i64(KV_DIM),
                   i64(0), i64(HEAD_DIM), i32(KV_HEADS), i32(0),
                   i32(HEAD_DIM), i64(0), u8(0)]),
                d(f"ctx{j}.kv_write", "reshape_and_cache",
                  [buf("k_n"), buf("kv_flat", (j * 2 * KV_DIM + KV_DIM) * BF16),
                   state("draft_kv", koff), state("draft_kv", koff + V_BYTE_OFF),
                   buf("slot_mapping"), ks, vs,
                   i64(KV_DIM), i64(DRAFT_KV_DIM), i64(DRAFT_BLOCK_STRIDE),
                   i64(2 * HEAD_DIM), i64(0), i64(0),
                   i64(KV_HEADS * 2 * HEAD_DIM), i64(0), i64(0)]),
            ]
        return ds

    states = {"kv": {"bytes_per_token": KV_BYTES_PER_TOKEN}}
    programs = {
        # prefill: one sequence, as many rows as the chunk holds (tokens);
        # decode: the bs=1 contract, 3D split-KV kernels; decode_batch: up
        # to MAX_SEQS sequences of one row each, the 2D instance. Which bs
        # runs which kernel is the manifest's choice: the caller picks the
        # forward whose batch fits, the runtime does not know
        "prefill": program(forward("attn_prefill", None, taps=spec), groups=1, rows="tokens"),
        "decode": program(forward("attn", "decode"), groups=1, rows=1),
        "decode_batch": program(forward("attn_batch", "decode"), groups=MAX_SEQS, rows=1),
    }
    if spec:
        states["draft_kv"] = {"bytes_per_token": DRAFT_KV_BYTES_PER_TOKEN}

        def pre(prefix, calls):
            return [dict(c, label=f"{prefix}.{c['label']}") for c in calls]
        # prefill hands the first token back and projects every row's tap
        # into the draft KV (positions / slot_mapping are the chunk's), so a
        # round can start right after it
        programs["prefill"] = program(
            forward("attn_prefill", "prefill", taps=True) + pre("precompute", draft_precompute()),
            groups=1, rows="tokens")
        # round = one speculative round per sequence as one program: draft's
        # rows spliced from the anchor the caller staged, the non-causal
        # draft pass and its markov chain (7 drafts), verify's ids spliced
        # from the first 6, the causal target pass over [anchor, d0..d5] with
        # its taps, every row's tap into the draft KV (rejected rows land
        # past the sequence's position and the next round overwrites them),
        # and the count of rows taken. The caller reads verify_tokens and
        # nacc.
        programs["round"] = program(
            [d("splice_draft", "splice_draft",
               [buf("anchor_token"), buf("draft_ids"), i32(BLOCK_TOKENS), i64(MASK_TOKEN)])]
            + pre("draft", forward(
                "attn_draft", "draft", mp="draft.", layers=DRAFT_LAYERS,
                kv_state="draft_kv", block_stride=DRAFT_BLOCK_STRIDE,
                scales="draft.kv_scales", lm_head_w="draft.lm_head.weight",
                final_norm_w="draft.norm.weight", ids="draft_ids"))
            + [d("splice_verify", "splice_verify",
                 [buf("anchor_token"), buf("draft_tokens"), buf("verify_ids"), i32(VERIFY_TOKENS),
                  i32(BLOCK_TOKENS)])]
            + pre("verify", forward("attn_verify", "verify", taps=True, ids="verify_ids"))
            + pre("precompute", draft_precompute())
            + [d("count", "spec_count",
                 [buf("draft_tokens"), buf("verify_tokens"), buf("nacc"), i32(VERIFY_TOKENS),
                  i32(BLOCK_TOKENS)])],
            groups=MAX_SEQS, rows=BLOCK_TOKENS)
    for name, dom in DOMAINS.items():
        if name in buffers:
            buffers[name]["domain"] = dom
    # normalize: hoist inline cubin/sha256 into `modules`, fold the ABI
    # constants every call repeats into the impls, default identity wiring
    return normalize({
        "schema_version": SCHEMA_VERSION,
        "model": "qwen3-4b-dspark" if spec else "qwen3-4b",
        "vars": {"tokens": {"max": CHUNK_MAX}, "seqs": {"max": MAX_SEQS}},
        "states": states,
        "buffers": buffers,
        "ops": kernels,
        "programs": programs,
    })


MINED_MODULES = {
    "rms": "vllm_layernorm", "rms_head": "vllm_layernorm", "fused": "vllm_layernorm",
    "rope": "vllm_pos_encoding", "silu": "vllm_activation", "cache": "reshape_and_cache",
    "unified": "unified_decode", "reduce": "reduce_segments",
}


def main():
    repo = pathlib.Path(__file__).resolve().parent.parent
    jsonl = sys.argv[1] if len(sys.argv) > 1 else str(
        repo / "dumped-kernels" / "pid3977275" / "launches.jsonl")
    spec_dir = sys.argv[2] if len(sys.argv) > 2 else str(
        repo / "dumped-kernels" / "pid2633632")
    fwd, prefills = pick_forwards(jsonl)
    by = extract(fwd)
    eps, scale = check_topology(by)
    pf = check_prefill(prefills, by)

    # unified 实例消歧 + cubin 钉定：spec dump 证伪 draft/verify 假设，
    # cuobjdump 按 num_regs 定位两个同 ABI 实例各自的 module 文件
    causal_regs, draft_regs = check_spec(spec_dir, jsonl, pf)
    cfile, csha, _ = find_unified_module(pathlib.Path(jsonl).parent, causal_regs)
    dfile, dsha, _ = find_unified_module(spec_dir, draft_regs)
    # `cubin` is a label; the runtime resolves the sha256. extract_kernels.sh
    # finds the non-causal instance in the spec dump by hash.
    pins = {"causal": ("unified_causal.cubin", csha), "draft": ("unified_noncausal.cubin", dsha)}
    # Every mined launch pins its dump module too: the manifest's `modules`
    # table is the complete dependency list, the runtime loads nothing else.
    # Labels say what the module is (vLLM's layernorm_kernels.cu, …); the
    # sha256 is the identity.
    idx = DumpIndex(pathlib.Path(jsonl).parent)
    for tag, label in MINED_MODULES.items():
        r = by[tag][0]
        pins[tag] = (f"{label}.cubin", idx.pin(r["symbol"], r["attributes"]["num_regs"],
                                               [p["size"] for p in r["params"]]))
    for tag, (label, sha) in sorted(pins.items()):
        print(f"  pin {tag:<9} -> {label:<26} {sha[:12]}")

    for spec, silu, name in [(False, "hub", "qwen3-4b.json"),
                             (False, "mined", "qwen3-4b-silu-mined.json"),
                             (True, "mined", "qwen3-4b-dspark.json")]:
        manifest = build(by, eps, scale, pf, pins, spec=spec, silu=silu)
        out = repo / "examples" / name
        out.write_text(json.dumps(manifest, indent=1) + "\n")
        counts = {p: len(v["calls"]) for p, v in manifest["programs"].items()}
        print(f"wrote {out} ({out.stat().st_size // 1024} KiB, "
              f"{len(manifest['buffers'])} buffers, calls {counts})")
    print(f"topology checks passed (eps={eps!r}, attn scale={scale!r}, "
          f"prefill fwds={[t for t, _ in prefills]}; unified pins: "
          f"causal={cfile} {csha[:12]} regs={causal_regs}, "
          f"draft={dfile} {dsha[:12]} regs={draft_regs})")


if __name__ == "__main__":
    main()
