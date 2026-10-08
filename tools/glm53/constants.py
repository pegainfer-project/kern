"""GLM-5.3-Flash model geometry (checkpoint config.json, verified) + serving constants."""

HIDDEN = 4096
LAYERS = 45
MTP_LAYER = 45                    # exists in the checkpoint; manifest v1 does not serve it
VOCAB = 154880
RMS_EPS = 1e-5
MAX_POS = 1048576
EOS = (154820, 154827, 154829)
PAD = 154820

# attention pattern
DSA_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]
KDA_LAYERS = [i for i in range(LAYERS) if i not in DSA_LAYERS]
assert len(DSA_LAYERS) == 11 and len(KDA_LAYERS) == 34

# mHC (multi-stream hyper-connections), all 45 layers; MTP has none
HC_MULT = 4
HC_STATE = HIDDEN * HC_MULT       # 16384
HC_ROWS = 24                      # hc_attn_fn / hc_ffn_fn rows (mixing coefficients)
HC_SINKHORN = 20
HC_EPS = 1e-6

# dense MLP (layers 0..2), FP8 block-128x128
DENSE_LAYERS = [0, 1, 2]
FFN = 12288

# MoE (layers 3..44), FP8 block-128x128
MOE_LAYERS = [i for i in range(LAYERS) if i not in DENSE_LAYERS]
N_EXPERTS = 288
TOPK = 8
N_SHARED = 1
MOE_INTER = 2048
N_GROUP = 1
TOPK_GROUP = 1
ROUTED_SCALING = 2.5
NORM_TOPK_PROB = True
SWIGLU_LIMIT = 10.0

# DSA (NoPE MLA + indexer), per layer
Q_LORA = 1536
KV_LORA = 512
DSA_HEADS = 64
QK_HEAD = 256
V_HEAD = 256
Q_B_DIM = DSA_HEADS * QK_HEAD     # 16384 (q_b_proj output)
KV_B_DIM = DSA_HEADS * (QK_HEAD + V_HEAD)  # 32768
# indexer
IDX_HEADS = 32
IDX_DIM = 128
IDX_TOPK = 2048
IDX_KPOOL = 4                     # pool slot covers 4 tokens
IDX_QB_DIM = IDX_HEADS * IDX_DIM  # 4096 (wq_b output)

# KDA, per layer
KDA_HEADS = 64
KDA_DIM = 128
KDA_QKV = KDA_HEADS * KDA_DIM     # 8192
KDA_CONV_K = 4
KDA_GATE_LOWER = -5.0

# FP8 block quant
FP8_BLOCK = 128

# --- serving layout choices (ours, not sglang)
EP = 8
EP_EXPERTS = N_EXPERTS // EP      # 36 per rank

# KDA per-sequence state PER RANK (TP8: 8 local heads of 64, q/k/v shard 1024)
KDA_HEADS_PER_RANK = KDA_HEADS // EP                    # 8
KDA_QKV_PER_RANK = KDA_QKV // EP                        # 1024
KDA_CONV_BYTES = (KDA_CONV_K - 1) * (3 * KDA_QKV_PER_RANK) * 2   # 18432 bf16 [3, 3072]
KDA_SSM_BYTES = KDA_HEADS_PER_RANK * KDA_DIM * KDA_DIM * 4          # 524288 f32 [8,128,128]
KDA_LINE_BYTES = KDA_CONV_BYTES + KDA_SSM_BYTES                     # 542720
KDA_SEQ_BYTES = len(KDA_LAYERS) * KDA_LINE_BYTES                    # 18452480

# fused KDA projection output width per rank: q|k|v (1024 each) | b (8) | f_a | g_a (128 each)
KDA_FUSED_QKVBFG_A = 3 * KDA_QKV_PER_RANK + KDA_HEADS_PER_RANK + 2 * 128  # 3336

# DSA KV: latent 512 bf16 per token per layer -> bytes_per_token over 11 layers
KV_BYTES_PER_TOKEN = len(DSA_LAYERS) * KV_LORA * 2    # 11264
KV_PAGE_TOKENS = 64                                    # our page unit (sglang uses 64 too)
KV_PAGE_BYTES = KV_PAGE_TOKENS * KV_BYTES_PER_TOKEN
