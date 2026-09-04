# K3 decode 核 ABI（kern 自写核集，取代 pegainfer 的 TileLang 桶）

状态：2026-09-02 任务书 / 合同。每个核一份 `.cu`、一个 `extern "C"` 入口、一份 cubin；
manifest 只认 **入口名 + 参数表 + grid/block/smem 公式**，语言无关。本轮全部用 CUDA C++
（nvcc 13.1，`-arch=sm_103a`），验收看 harness、SASS 和 ncu，不要求与 TileLang 逐位一致。

目标：(1) 每 rank B>1 —— B 是运行时参数，一 block 一行，没有桶、没有 line shim；
(2) 把只做搬运/逐元素的 launch 融进邻居 —— 93 层现在 3792 个 launch，目标 ≤ 1500；
(3) MLA decode attention 在长上下文下 KV 只读一次，不是每头读一次。

## 0. 通用约定

- 入口 `extern "C" __global__ void kern_k3_<name>(...)`，文件 `tools/kernels-src/k3_<name>.cu`，
  单文件，只 include CUDA 自带头（`cuda_bf16.h`、`cuda_fp16.h`、`mma.h`），
  `tools/build_kernels.sh` 会编成 `target/cubins/k3_<name>.cubin`。
- 类型：`bf16 = __nv_bfloat16`，`f32 = float`，`i32 = int`，`i64 = long long`。
  标量按值传；指针一律 `__restrict__`。**所有 buffer 行主序，行距 = 宽度**，除非参数表里给了 stride。
- **B 是运行时参数** `int B`（decode 里 tokens == seqs）。约定 `grid.x = B`（一行一个 block.x），
  `grid.y` 按核自己的切分；block 和动态 smem 在文档里给公式，manifest 会照抄。
  runtime 传 grid 时 `tokens` 是变量，所以 grid.x 必须恰好是 B，不能是 ceil(B/k)。
- 权重（gamma、bias、卷积核、`w_f_b`、`w_kv_b`…）所有行共享，没有 batch 维。
- 输出 buffer 必须整块写满；不允许"没变就不写"。原地（同一指针既读又写）只在明确写了"原地安全"的核允许。
- **f32 partial**：cuBLAS extern `cublas_bf16_tn_f32` 的输出，`f32 [B, ldc]`，有效列从 `off` 起 `n` 列。
  所有从 GEMM 接数的核直接读 f32 partial，自己 landing 成 bf16（"land"），不再有单独的 land 核。
- 舍入链：累加 f32，**landing 点**（f32→bf16）按每个核列出的位置放——这是 pegainfer 的链，
  照着做误差最小，但**不要求逐位**；harness 的容差是验收标准。
- 常量（全局，编进核里也行，但入口签名要保持文档形状）：

| 名 | 值 | 名 | 值 |
|---|---|---|---|
| H | 7168 | HEADS / HEAD_DIM | 96 / 128 |
| INNER = HEADS·HEAD_DIM | 12288 | Q_LORA / KV_LORA / ROPE | 1536 / 512 / 64 |
| KV_A = KV_LORA + ROPE | 576 | Q_B = HEADS·192 | 18432 |
| MLA_FUSED = Q_LORA + KV_A + INNER | 14400 | KDA_FUSED = 4·INNER | 49152 |
| WSM | 256（b_proj 96 \| f_a 128 \| pad） | EXPERTS / TOPK | 224 / 16 |
| LATENT / INTER / SHARED | 3584 / 3072 / 6144 | DENSE_I | 33792 |
| V | 163840 | NB_MAX（attnres 块数上限） | 8 |
| EPS（所有 rms） | 1e-5 | LB（KDA gate lower bound） | -5.0 |
| PAGE | 64 token | LATENT_ROW | 576 |

### 状态布局

**KDA line**（每序列每 KDA 层一条，`bytes_per_seq = n_kda · LINE_BYTES`）：

```
offset 0                         : rec   f32 [96 head][128 dv][128 k]      6291456 B
REC_BYTES = 6291456              : win_q bf16 [3 tap][12288]                73728 B
REC_BYTES + 73728                : win_k bf16 [3][12288]
REC_BYTES + 2*73728              : win_v bf16 [3][12288]
LINE_BYTES = 6512640
```

tap 0 最旧，tap 2 最新。行 b 这一层的 line 地址 = `kda_base + (i64)line_index[b] * LINE_BYTES`；
`line_index` 是 `i32 [B]`（manifest 里是 `kda.line_index[n_kda, seqs]` 的一行，runtime 按层
偏移传进来）。核签名里统一写成 `(void* kda_base, const int* line_index, long long line_bytes)`。

**MLA latent slab**（`bytes_per_token = n_mla · 576 · 2`）：页 p 起点 `p * page_stride`（元素），
本层切片 `+ layer_off`，token t `+ t * 576`，行 = `latent 512 | rope 64` bf16。
`page_stride = n_mla · 64 · 576`，`layer_off = k · 64 · 576`（k = 本层在 MLA 层里的序号）。
`block_table i32 [B, max_pages]` 逻辑页→物理页；`seq_lens i32 [B]` = 含本步 token 的上下文长度
（kv_append 在 attention 之前跑）；`slot_mapping i64 [B]` = 本步 token 的 slot，页 = slot/64，行 = slot%64。

**attnres 快照** `blocks bf16 [B, NB_MAX, H]`，`scores` 不再需要（融进核里）。

### 数学原语（CPU 参考按这个写）

- `rms(x, gamma)`（round-before-scale）：`y = bf16(x · rsqrt(mean(x²) + EPS))`，再 `y · gamma`（bf16×bf16 → bf16）。
- `rms_nw(x)`：无权重版，`x · rsqrt(mean(x²) + EPS)`，留 f32。
- `sigmoid(x) = 1/(1+exp(-x))`；`situ(g,u) = 4·tanh(g/4)·σ(g)·25·tanh(u/25)`。
- attnres（NB 个候选 + prefix）：`score_c = Σ_i rms_nw(cand_c)[i] · sw[i]`，c = 0..NB-1 取 `blocks[b,c]`，
  c = NB 取 prefix；`p = softmax(score)`；`mixed = bf16(Σ_c p_c · cand_c)`（f32 累加）。NB = 0 时 mixed = prefix。

## 1. 核清单

每层的 launch 序列（KDA 层 / MoE 层为例）改成：

```
attnres_rms(nb_in, snapshot)                → normed                    [K1]
gemm normed·wbig  → kda_partial f32 [B, 49152]  (q|k|v|gate 一个 GEMM)
gemm normed·wsm   → wsm_partial  f32 [B, 256]
conv_silu(kda_partial, line)                → conv_q/k/v                [K2]
kda_core(conv_qkv, wsm_partial, kda_partial band 3, line) → gated       [K3]
gemm gated·w_o    → hidden_partial f32
land_add_attnres_rms(hidden_partial, hidden, snapshot, nb_mlp) → prefix2, normed   [K1]
gemm normed·w_router → router_partial;  router_topk → idx, wts          [K6]
gemm normed·w_lat_down → latent_partial; land → latent                  [K7]
MegaMoE ×3 → routed_latent;  rms(gamma_lat) → routed_latent_norm       [K7]
gemm → routed_partial;  gemm normed·wsh → shared_partial
land_situ(shared_partial) → shared_act                                  [K7]
gemm shared_act·sh_down → shared_partial2
land_add2(routed_partial, shared_partial2, prefix2) → hidden            [K1]
```

MLA 层把 conv/kda 换成：

```
gemm normed·wfu → mla_fused_partial f32 [B, 14400]
mla_prep(mla_fused_partial, slot_mapping, slab) → q_norm, mla_gate, slab 追加   [K4]
gemm q_norm·w_q_b → q_partial f32 [B, 18432]
mla_absorb(q_partial, w_kv_b) → q_abs bf16 [B, 96, 576]                            [K5a]
mla_attn(q_abs, slab, block_table, seq_lens, mla_bsk) → o_lat bf16 [B, 96, 512]     [K5, DSL 主核 + 归约]
mla_vup_gate(o_lat, w_kv_b, mla_gate) → gated                                        [K5c]
```

每步一次（embed 之后）：`mla_split_plan(seq_lens) → mla_bsk` [K5b]。尾部：`attnres_rms(8)` → `gemm w_lm` → `argmax_f32` [K6]。

### K1 残差流：`k3_attnres_rms` / `k3_land_add_attnres_rms` / `k3_land_add2`（一个 agent）

```c
// [K1a] mixed = attnres(blocks, prefix, nb); if (snapshot) blocks[b, nb] = prefix; normed = rms(mixed, gamma)
extern "C" __global__ void kern_k3_attnres_rms(
    const bf16* prefix,      // [B, H]  残差流（hidden）
    bf16*       blocks,      // [B, NB_MAX, H]  快照；snapshot != 0 时把 prefix 写进 blocks[b, nb]
    const f32*  sw,          // [H]  scoring 向量；nb == 0 时可为任意指针（不读）
    const bf16* gamma,       // [H]
    bf16*       normed,      // [B, H]
    int nb, int snapshot, int B);
// grid (B, 1, 1)，block 1024，smem 由你定（H=7168：每线程 7 个元素，NB+1 ≤ 9 个 score）

// [K1b] p = bf16(partial[b, :H]);  prefix2 = snapshot ? p : bf16(prefix + p);
//       mixed = attnres(blocks, prefix2, nb);  normed = rms(mixed, gamma)      （不写快照）
extern "C" __global__ void kern_k3_land_add_attnres_rms(
    const f32*  partial,     // [B, H]  o_proj 的 f32 partial
    const bf16* prefix,      // [B, H]
    const bf16* blocks,      // [B, NB_MAX, H]
    const f32*  sw, const bf16* gamma,
    bf16*       prefix2,     // [B, H]  必须写（层尾要用）
    bf16*       normed,      // [B, H]
    int nb, int snapshot, int B);

// [K1c] hidden = bf16( prefix2 + bf16(p1[b,:H]) + (two ? bf16(p2[b,:H]) : 0) )   （two == 0：dense 层，p2 不读，但传的是合法指针）
extern "C" __global__ void kern_k3_land_add2(
    const f32* p1, const f32* p2, const bf16* prefix2, bf16* hidden, int two, int B);
// grid (B, 1, 1)，block 1024
```

landing 点：attnres 的 mixed 落 bf16 一次；rms 内部 `bf16(x·rsqrt)` 再乘 gamma；K1b 的 p 先落 bf16
再与 prefix 相加落 bf16（两次舍入，与 pegainfer 同；愿意的话可以 f32 加完落一次，容差内都行）。
K1a 的 prefix 与 normed 可以是不同 buffer；`hidden` 在 K1c 之前不会被覆盖，所以生成器不再拷 prefix。

### K2 `k3_conv_silu`：三条流一次 launch，窗口在 KDA line 里

```c
// 对 s = 0,1,2（q,k,v）：x = bf16(partial[b, s*INNER + c]);
//   y = Σ_{t<3} f32(win_s[t][c])·cw[s][t][c] + f32(x)·cw[s][3][c];  sb = bf16(y);  out_s[b,c] = bf16(sb·σ(sb));
//   win_s[0..1] = win_s[1..2];  win_s[2] = x       （窗口原地更新）
extern "C" __global__ void kern_k3_conv_silu(
    const f32*  partial,     // [B, KDA_FUSED]  列 s*INNER.. 是流 s 的 f32 partial
    const f32*  cw,          // [3][4][INNER]   cw_q | cw_k | cw_v（生成器把三份权重拼成一块）
    void*       kda_base, const int* line_index, long long line_bytes,   // 窗口：line + REC_BYTES + s*73728
    bf16*       conv_q, bf16* conv_k, bf16* conv_v,   // [B, INNER] 各一
    int B, const int* span_at, int span);   // 行号在 [*span_at, +span) 的 block 直接返回（span 行由 K9 做）
// 交付：grid (B, 3, 24)（y = 流，z = 512 列一段），block 128，每线程 4 列，smem 0
```

### K3 `k3_kda_core`：delta rule，f_b 投影与 gate 的 landing 都在核内

```c
// 行 b、头 h（block = (b, h)，128 线程，线程 = dv）：
//   q,k,v = conv_q/k/v[b, h*128 .. +128]
//   qtot = Σ bf16(q·q) (f32)，kr 链全 bf16：qr = bf16(rsqrt(f32(bf16(qtot)) + 1e-6))
//   qs[d] = f32(bf16(q[d]·qr)) · 128^-0.5 ;  kn[d] = f32(bf16(k[d]·kr))
//   beta   = σ(f32(bf16(wsm_partial[b, h])))                          // 列 0..95
//   flow   = bf16(wsm_partial[b, 96 .. 224])                          // 列 96..223，128 个
//   ga[d]  = Σ_j f32(flow[j]) · f32(w_f_b[h*128+d, j])                // f_b GEMM 就地算，f32
//   raw[d] = f32(bf16(ga[d])) + dt_bias[h*128+d]
//   dec[d] = exp(LB · σ(exp(a_log[h]) · raw[d]))
//   m[dv]  = Σ_k S[h,dv,k]·dec[k]·kn[k];  dlt[dv] = (f32(v[dv]) - m[dv])·beta
//   S'[h,dv,k] = S[h,dv,k]·dec[k] + dlt[dv]·kn[k]      （原地写回 rec）
//   attn[dv] = bf16(Σ_k S'[h,dv,k]·qs[k])
//   o[d] = bf16( f32(attn[d]) · rsqrt(mean(attn²) + EPS) · gamma_o[d] ) · bf16(σ(f32(bf16(gate_partial[b, 3*INNER + h*128 + d]))))
extern "C" __global__ void kern_k3_kda_core(
    const bf16* conv_q, const bf16* conv_k, const bf16* conv_v,   // [B, INNER]
    const f32*  wsm_partial,   // [B, WSM]
    const f32*  gate_partial,  // [B, KDA_FUSED]  只读 band 3（列 3*INNER..）
    const bf16* w_f_b,         // [INNER, 128]
    const f32*  dt_bias,       // [INNER]
    const f32*  a_log,         // [HEADS]
    const f32*  gamma_o,       // [128]
    void*       kda_base, const int* line_index, long long line_bytes,   // rec 在 line 偏移 0
    bf16*       out,           // [B, INNER]
    int B, const int* span_at, int span);   // 行号在 [*span_at, +span) 的 block 直接返回（span 行由 K8 + K11 做）
// grid (B, HEADS, 1)，block 128
```

性能点：rec 每行每层读 + 写 6.3 MB，B=64 时一层 800 MB，这是 KDA 的带宽主项——访存模式（线程=dv、串行 k 是按行走，
warp 内 32 行同时读）请用 ncu 看 L1/L2 命中和 dram 吞吐，必要时换 k 分片 + shuffle 归约。
`w_f_b` 每 block 读 128×128 bf16 = 32 KB，B·96 个 block 共享，走 L2。

### K4 `k3_mla_prep`：MLA 融合投影的落地 + 两个 norm + 追加 latent

```c
// 从 P = mla_fused_partial[b, :]（14400 列）：
//   q_norm[b]   = rms(bf16(P[0 .. 1536]), gamma_q_a)                 // land 后 round-before-scale
//   kv_norm     = rms(bf16(P[1536 .. 2048]), gamma_kv_a)             // 512
//   rope        = bf16(P[2048 .. 2112])                              // 64
//   slab[slot]  = kv_norm | rope                                     // kv_append：行 = (slot/64)·page_stride + layer_off + (slot%64)·576
//   mla_gate[b] = bf16(P[2112 .. 14400])                             // 12288
extern "C" __global__ void kern_k3_mla_prep(
    const f32*  partial,       // [B, MLA_FUSED]
    const bf16* gamma_q_a,     // [Q_LORA]
    const bf16* gamma_kv_a,    // [KV_LORA]
    const i64*  slot_mapping,  // [B]
    bf16*       slab,          // state 基址
    long long layer_off, long long page_stride,   // 元素
    bf16*       q_norm,        // [B, Q_LORA]
    bf16*       mla_gate,      // [B, INNER]
    int B);
// 交付：grid (B, 4, 1)（y=0 做两个 norm + 追加，y=1..3 各落 1/3 的 gate），block 512，smem 12800（静态）
```

### K5 MLA decode attention：FlashInfer 的 CuTe-DSL Blackwell 核 + 三个配套小核

自写核 `k3_mla_paged_attn`（2026-09-02 交付，下文"旧核"）在 12.9k 上下文下每层每行 ~42 µs，
读 latent 只有 ~360 GB/s（<5% HBM）；agent 负载的 ctx p50 是 219k，这一项会是每步 ~17 ms/行，
比 E5 剩下的所有东西都大。2026-09-03 换成 NVIDIA 用 CuTe DSL 写、随 FlashInfer 发行的
Blackwell MLA decode 核（`flashinfer/cute_dsl/attention/monolithic/mla_decode_fp16.py`，BSD-3，
tcgen05 2-CTA MMA + TMA 分页加载 + split-KV 归约），**预编译成 cubin 收进仓库**
（`tools/kernels-bin/mla_decode_h96_p64.cubin`，构建配方 `tools/build_mla_dsl.py`，README 在同目录），
runtime 不加任何模型代码：它的 struct 参数 ABI 由 manifest 的 `bytes<n>` + `pack` 铺平，
五个 TMA 描述符是 `bytes<128>` 里的 `tensormap` 字段（`docs/manifest.md`）。

数学链和旧核一致处：q_abs 在 bf16 落地（f32 累加）、attention 输出 lat 在 bf16 落地、
o = bf16(W_UV·lat) 后乘 bf16(σ(gate))；不同处：分数/softmax 在核内 f32（scale 取
bf16(192^-0.5)·log2e 折进 exp2），P 以 bf16 进 MMA。

```c
// 一步一次：每行的 KV split 数（docs 里 K5b）
extern "C" __global__ void kern_k3_mla_split_plan(const int* seq_lens, int* block_split_kvs, int split_max, int B);
// grid (1,1,1) block 1024。行 b 按 128-token tile 计数，全 batch 的 tile 按 %nsmid/2 - B 个 cluster 摊
// （一波跑完好过第二波），split_b = clamp(ceil(tiles_b / per), 1, split_max)。

// 每个 MLA 层（K5a）：absorb
extern "C" __global__ void kern_k3_mla_absorb(const f32* q_partial, const bf16* w_kv_b, bf16* q_abs, int B);
// grid (ceil(B/32), 96, 8) block 128。q_abs[b, h, 0..512] = bf16(Σ_d bf16(q)[d]·W_UK_h[d, j])，[512..576] = rope
// 原样。带宽问题（W_UK 12 MB 全 batch 共用）：block = 头 × 64 列，线程握 8 行 d × 8 列的 16 B load
// 全部在飞，W 切片留在寄存器里连续吃最多 4 组 8 行；16 个 d 切片在 smem 归约。

// 每个 MLA 层：DSL 主核 + 归约（一个 op `mla_attn`，两个 launch）
//   主核  grid (2, B, split_max) block 384 cluster (2,1,1) smem 232448；行 b 只有 block_split_kvs[b] 个
//         split 有活，其余 CTA 读到 k_tile_count = 0 立刻退出（实测 split_max 8 → 32 时间不变）
//   归约  grid (96, 1, B) block 128 smem 1024；按 block_split_kvs[b] 合并 (acc_o, acc_lse) → o_lat[b, h, 512]
//   工作区 acc_o [tokens, split_max·128·512] f32 + acc_lse [tokens, split_max·128] f32，
//         生成器 --mla-split-max（默认 32）定 split_max，也定这块工作区（每行 split_max × 256 KiB）

// 每个 MLA 层（K5c）：v_up + gate
extern "C" __global__ void kern_k3_mla_vup_gate(const bf16* o_lat, const bf16* w_kv_b, const bf16* mla_gate, bf16* gated, int B);
// grid (ceil(B/32), 96, 4) block 256。gated[b, h*128+dv] = bf16(Σ_j W_UV_h[dv,j]·lat[j]) · bf16(σ(gate))。
// 同样的切法：block = 头 × 32 个 dv，线程握一个 dv 的 1/8 行（8 个 16 B load），8 个切片在 smem 归约。
```

DSL 主核的 28 个参数（生成器 `mla_attn_op` 写；PTX 里核真正读的只有描述符、页表、acc_o/acc_lse、
split_kv、cache_seqs、block_split_kvs、两个 scale 和 S=1 的 FastDivmod，其余按 DSL 的布局填 0
或常量，靠 `cuFuncGetParamInfo` 对齐字节数）：

| # | 类型 | 内容 |
|---|------|------|
| 0,1 | bytes<64> | TiledMMA 描述（空） |
| 2,4,6,8,10 | bytes<128>（tensormap 字段） | q latent / q rope / c latent / c rope / c latent 转置：bf16、swizzle 128B、L2 promotion 128B，box [64,64,1]（转置 [64,32,1]）；q 的 dims [512\|64, 96, tokens_max]，KV 的 dims [512\|64, 64, **0 = 铺满 state**]，页步长 = 全部 MLA 层的一页字节 |
| 3,5,7,9,11 | bytes<8/12> | TMA 坐标张量的动态 shape（核不读） |
| 12 | bytes<24> | 页表 {ptr, max_pages, B, i64 max_pages} |
| 13,14 | bytes<48/24> | o / lse 张量（split 路径不写） |
| 15 | bytes<48> | acc_o {ptr, 128, split_max, 512, 1, B; strides split_max·512, 512, split_max·128·512, 128·split_max·512} |
| 16 | bytes<40> | acc_lse {ptr, 128, split_max, 1, B; strides split_max, 128·split_max, 128·split_max} |
| 17 | i32 | split_max |
| 18,19 | bytes<16> | cache_seqs / block_split_kvs {ptr, i64 B} |
| 20,21 | f32 | softmax scale·log2e，输出 scale 1.0 |
| 22–24 | i32 | B, S=1, split_max（scheduler 参数） |
| 25–27 | bytes<12> | FastDivmod(B)（核不读，0）、(1)、(split_max)：{d, ceil(2^(32+l)/d)−2^32, 1, l−1} |

归约核 7 个参数：o {ptr, 96, 512, 1, B; i64 strides 512, 96·512, 96·512}、lse {ptr, 96, 1, B; i64 96, 96}、
acc_o、acc_lse、split_max、cache_seqs、block_split_kvs。

**ABI 是怎么拿到的**：DSL 的 JIT host 代码经 `libcute_dsl_runtime.so` 自己导出的
`_cudaLaunchKernelEx` / `_cuTensorMapEncodeTiled` 包装启动（内部静态链接 cudart，驱动符号走
export table），所以 LD_PRELOAD 钩 `cuLaunchKernelEx` 什么都看不到；钩那两个包装符号才拿到每个
参数的字节。描述符的 dims/strides/box 能从字节里反解，但 DSL 自己 encode 的描述符比驱动
`cuTensorMapEncodeTiled` 多两个 bit、box 字段错一字节——不追这个：用驱动 encode 的描述符
独立起 cubin，输出和 DSL 逐位一致（`target/mla-bench/abi/launch.c`，四种形状 0 mismatch）。

**实测（tray07 GB300，单卡，µs / 层，含归约，无 graph）**：

| 形状 | 旧核 | DSL | 备注 |
|------|------|-----|------|
| B=1 13k | ~42 | 16.5 | split 26 |
| B=1 65k | — | 31.9 / 26.5 | split 32 / 64 |
| B=1 200k | — | 78 / 58 | split 32 / 72（72 ≈ 3.9 TB/s） |
| B=16 13k 混合 | — | 55 / 67 / 87 | split 4 / 6 / 32：多一波 cluster 就慢，归约按 (行×split) 读 256 KB |

split 数的教训：cluster 数刚好一波（≤ nsm/2）最好，切得更细只多付归约；B=1 长上下文才需要
>32 的 split，`--mla-split-max` 默认 32 是工作区和长上下文之间的折中（64 时 B=1 200k 快 25%，
工作区翻倍）。

配套小核（tray03 GB300 单卡，µs，`target/mla-bench/abi/side.c`）：absorb B=1 6.5 / B=16 12 / B=64 38，
vup_gate 11 / 18 / 60，split_plan 2.5（每步一次）。第一版按"一个 block 一个头、16 行共用"写，
absorb B=1 17.7、vup B=16 58：每 block 在飞的 load 太少，纯延迟；改成上面的切法后 B=1 的下限是
12 MB 权重的一次读（~2 µs）+ 波次延迟，再压每步只剩 <1%，没做。旧核把这两段算在 attention
核里，短 ctx 下 B=1 全套 24.6 µs/层，新链 ≈ 6.5 + 12 + 3 + 11，持平；长 ctx 见上表。

### K6 `k3_router_topk` / `k3_argmax_f32`（一个 agent，顺带 embedding 与 rms）

```c
// sig = σ(S[b,e])；biased = sig + bias[e]；顺序扫描取 16 次 max（tie 取小 e）；
// wts[t] = sig[idx[t]] / (Σ_t sig[idx[t]] + 1e-20) · f32(rs[0])
extern "C" __global__ void kern_k3_router_topk(
    const f32* S,            // [B, EXPERTS]  f32 partial
    const f32* bias,         // [EXPERTS]
    const bf16* rs,          // [1]
    int* idx,                // [B, TOPK]
    f32* wts,                // [B, TOPK]
    int B);
// grid (B,1,1)，block 256

// argmax over f32 logits（不再 land 成 bf16）：两段
extern "C" __global__ void kern_k3_argmax_f32_partial(const f32* logits, f32* pmax, int* pidx, int n);  // grid (B, 64), block 1024
extern "C" __global__ void kern_k3_argmax_f32_final(const f32* pmax, const int* pidx, i64* out, int parts); // grid (B,1,1), block 64
// tie：取最小下标。

// 通用 rms（MoE 的 gamma_lat 用，h = 3584；也给 K1 之外任何地方）
extern "C" __global__ void kern_k3_rms(const bf16* x, const bf16* gamma, bf16* o, int h, int B);   // grid (B,1,1), block 1024
```

### K7 `k3_land` / `k3_land_situ`（归 K6 的 agent）

```c
// o[b, i] = bf16(p[b*ldc + off + i])，i < n
extern "C" __global__ void kern_k3_land(const f32* p, bf16* o, int n, int off, int ldc, int B);   // grid (B, ceil(n/1024)), block 1024
// act[b, i] = bf16( situ( f32(bf16(p[b*2n + i])), f32(bf16(p[b*2n + n + i])) ) )，i < n   （gate 在前 n 列，up 在后 n 列）
extern "C" __global__ void kern_k3_land_situ(const f32* p, bf16* act, int n, int B);               // grid (B, ceil(n/1024)), block 1024
```

### K8 `flash_kda_d128`：span 的 KDA 时间轴（vendored FlashKDA，两个核）

来源 `tools/flash-kda/`（MoonshotAI FlashKDA `7afb9f4`，MIT，`PROVENANCE.md`），cubin
`tools/kernels-bin/flash_kda_d128.cubin`。ABI 不是读模板推的，是 `tools/kernel-capture` 从 vendored 源码编的
probe（`tools/flash-kda/probe.cu`）跑一次直接提出来的（`lift.py`，见 `tools/kernel-capture/README.md`）；与
pegainfer shim 的捕获逐字段一致。数学（q/k L2 norm、β sigmoid、gate = `gate_scale·σ(g + dt_bias)`、
`a_log`）都在核内，输入是 conv+SiLU 之后的裸 q/k/v 和 `w_f_b` 投影后的 g，输出是 o_norm 之前的 attn；
状态 f32 `[H][128 v][128 k]`，与 K3 的 rec 同布局，可直接指到 KDA line 的 rec 区。

两个 entry（mangled 名见 cubin 的 `cuobjdump -symbols`，都是 `_Z22_flash_kda_fwd_prepareI…` /
`_Z25_flash_kda_fwd_recurrenceI…` 一个实例）：

| 核 | grid | block | dyn smem | 作用 |
|---|---|---|---|---|
| `_flash_kda_fwd_prepare` | `(tiles, H)`，tiles = ceil(T/16) | 256 | 21248 | 每 16 行 tile：kd/qd/kr、gt、inv、mqk 写 workspace |
| `_flash_kda_fwd_recurrence` | `(1, H)` | 192 | 98432 | 每 head 串行走 tile：state in → out，写 attn |

参数全是按值的 cute `TiledCopy`（`bytes<256>`：`CUtensorMap` 在 0，动态 stride 的 `int` 在 128，其余是
host 栈垃圾，pack 置 0）加尾部标量。描述符 dtype/dims/strides/box 如下（dims 内层在前，元素；strides 字节；
swizzle 0、L2 promotion 128 除注明外；oob none）：

```
prepare  0 q      bf16 [128, T, H] strides [H·256, 256] box [128,16,1]   +128: i32 1
         1 k      同 q                                                    +128: i32 1
         2 beta   bf16 [H·T]        box [32]                              （[H][T]，无动态 int）
         3 g      同 q                                                    +128: i32 1
         4 dt_bias f32 [128, H]     strides [512] box [128,1]
         5 ws_kd  bf16 [128, 16, tiles·H] strides [256, 4096] box [8,16,1]   ws + 0
         6 ws_qd  同 ws_kd                                                   ws + n·4096
         7 ws_kr  同 ws_kd                                                   ws + n·8192
         8 ws_gt  f32 [128, tiles·H] strides [512] box [128,1]               ws + n·12288
         9 ws_inv bf16 [16, 16, tiles·H] strides [32, 512] box [8,16,1]      ws + n·12800
        10 ws_mqk 同 ws_inv                                                  ws + n·13312
        11 f32 scale      12 i32 T      13 i32 H      14 i32 N=1      15 i64 cu_seqlens=0
        16 i32 tiles      17 ptr a_log  18 f32 gate_scale             19 ptr tile_prefix（varlen=false 不读，0）
recurrence 0 v    bf16 [128, T, H] strides [H·256, 256] box [8,16,1]     +128: i32 1
         1 beta   同上
         2..7     ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk 同 prepare 5..10（box 同）
         8 state_in  f32 [128, 128, H] strides [512, 65536] box [8,128,1] swizzle 32
         9 state_out 同 state_in
        10 out    同 v（TMA store 描述符，box [8,16,1]）                     +128: i32 1
        11 ptr out（同一块，bf16 [T, H, 128]）  12 i32 T  13 i32 H  14 i32 N=1  15 i64 cu_seqlens=0  16 i32 tiles
（q 只进 prepare——它落在 ws_qd 里；recurrence 里 out 既是 TMA store 的描述符又是裸指针。这两项最初写反了
（q/v、v/out），lift 出来的 `@A+0x0` 只是地址字母，是 probe 打印各 buffer 指针后对回去才定的；C2 门禁
`program_io` 逐位对拍就是抓这个的。）
```

n = tiles·H；workspace 每 (tile, head) 13824 B，六个数组分开连续（上游 `WS::kPerTile`，
`fwd_launch.cu`），末尾 128 B 的 tile_prefix 区 varlen=false 用不到。`scale` 乘在 q 上（bf16 相乘，
`r_qd = q·exp_cumsum·scale`），`gate_scale = lower_bound·log2 e`（K3 lower_bound −5 → −7.2135）。

放进 manifest 时 T 与 tiles 都取 span 上界（描述符装载时定死；核用标量 `T`/`tiles` 决定实际走多少行，
尾 tile 里 ≥ seq_len 的行核内自己清零——`fwd_kernel1.cuh` `actual_len`，`fwd_kernel2.cuh` 同——不靠 TMA
OOB 填零），beta 用 `[H][T]` 且行距是**实际** T（核内 `beta_linear = h·T + t`），所以写 beta 的 gather 核要按
当次 span 的 T 排。描述符维数、offset 全是 span.max 与 H 的算术，生成器写死。

### K9 / K10 / K11：span 的配套小核（`k3_span_gather` / `k3_span_state` / `k3_kda_out_gate`）

span 是 batch 里连续的一段行 `[*span_at, *span_at + span)`（同一序列的连续 token；`span_at` 是 `[1]` 的 i32
input，因为 tray 批"自己的行在前"，同一个 span 在 owner 上在行 0、在 peer 上在它的 block d），K2/K3 对这些行
直接返回，由下面三个核加 K8 接手。K8 的描述符基址装载时定死，所以 span 的 q/k/v/out 各有自己的 `[span, INNER]`
buffer（行 0..span），K9 把 span 行搬进去、K11 把结果搬回 batch 行。

```c
// K9：K2 逐行作用于 batch 行 at..at+span（tap 先取 line 窗口再取 span 自己的前几行，landing 与 K2 完全一样），
//     结果落在 span 自己的行 0..span；窗口最后留 span 的末三个输入；顺带写 K8 要的转置 beta 与 f_a 的 flow
extern "C" __global__ void kern_k3_span_gather(
    const f32* partial, const f32* cw,                                   // 同 K2，读行 at..at+span
    void* kda_base, const int* line_index, long long line_bytes,         // 只用 line_index[at] 的 line
    const f32* wsm_partial,                                              // [B, WSM]，读行 at..at+span
    bf16* span_q, bf16* span_k, bf16* span_v,                            // [span, INNER]
    bf16* span_beta,   // [HEADS * span]，h*span + i = bf16(wsm[at + i, h])
    bf16* span_flow,   // [span, 128]，bf16(wsm[at + i, 96 + j])
    const int* span_at, int span);
// grid (INNER/512, 4, ceil(span/8))，block 128；y<3 是流（每线程 4 列 × 8 行），y==3 写 beta/flow。
// 只有 z==0 的 block 读旧窗口（行 0..2 要），也只有它在读完之后写新窗口，block 之间不碰同一字节。

// K10：行 at 的 rec（line 偏移 0，f32 [HEADS][128][128]）与固定基址的 buf 互拷；K8 的 state 描述符指 buf
extern "C" __global__ void kern_k3_span_state(
    void* kda_base, const int* line_index, long long line_bytes, const int* span_at,
    f32* buf, int to_line);  // 0: buf=rec；1: rec=buf
// grid (HEADS, 32, 1)，block 128，每线程一个 float4

// K11：K3 的尾巴单独拿出来：K8 把裸 attn（o_norm 之前）写在 attn 的行 0..span，这里做完写进 gated 的行 b = at + i
//   o = bf16(f32(attn)·rsqrt(mean(attn²)+1e-5)·gamma_o[d])，gated = bf16(f32(o)·f32(bf16(σ(f32(bf16(gate_partial[b, 3·INNER + h·128 + d]))))))
extern "C" __global__ void kern_k3_kda_out_gate(const bf16* attn, const f32* gate_partial, const f32* gamma_o,
                                                bf16* gated, const int* span_at, int span);
// grid (span, HEADS, 1)，block 128（线程 = dv）
```

g（K8 的 gate 输入）不需要核：`span_flow · w_f_bᵀ` 用 cuBLAS 的 bf16 GEMM 出 bf16 `[span, INNER]`，
与 K3 核内 f_b 投影的 landing 相同（f32 累加落 bf16 一次）。harness：`tools/k3-harness` 的
`span_gather`（B = span，对 K2 参考逐行调用）、`span_state`（逐字节）、`kda_out_gate`，三个都把 span 放在
batch 行 3 起（`SPAN_AT`），前面的行必须原样。

## 2. 验收（每个核）

1. **harness 通过**：`tools/k3-harness/`（见其 README）——对每个核、每个规定形状，随机输入 + CPU 参考，
   B ∈ {1, 2, 8, 64}。容差：逐元素 |err| ≤ 3 bf16 ULP(|ref|) + 1e-3，且相对 RMS 误差 ≤ 2e-3。
   不许改 harness 与参考实现；觉得参考错了，写进 notes 找我。
2. **SASS**：`nvcc -Xptxas -v` 0 字节 spill、0 字节 local；`cuobjdump -sass` 里没有 `.MULTICAST`；
   寄存器数与 `__launch_bounds__` 一致；把 ptxas -v 输出贴进 notes。
3. **ncu**：`ncu --set full -k regex:kern_k3_<name> -c 1` 在 B=64（K5 在 ctx=32768，B=1）上，
   记录 dram 吞吐（GB/s 与占峰值 8 TB/s 的比例）、achieved occupancy、L2 命中率、总时长；报告存
   `tools/k3-harness/reports/k3_<name>.ncu.txt`。访存主导的核目标 ≥ 60% 峰值带宽；离目标远要在 notes 里说为什么。
4. **交付物**：`tools/kernels-src/k3_<name>.cu`（头注释 = 本文档的签名 + grid/block/smem 公式 + landing 点）、
   ncu 报告、`tools/k3-harness/notes/k3_<name>.md`（设计、访存模式、测得的数字、没做到的事）。
   不要动别的文件，不要 commit。
5. 机器：tray 由任务书指定，`CUDA_VISIBLE_DEVICES=<n>` 各用一张卡；`nvidia-smi` 先看一眼有没有人在跑。
   nvcc/ncu 在 `/usr/local/cuda-13.1/bin`。

## 3. 生成器侧配套（不归 agent）

- q|k|v|gate 一个 GEMM（wbig 全 49152 行）→ `kda_partial`；`cw_q/k/v` 拼成 `[3][4][INNER]` 一块权重。
- `prefix` / `mixed` / `mixed2` / `scores` / `conv_x` / `attn_out` / `mlp_out` / `logits` 等 workspace 删除。
- `seqs` 变成变量（max 64），`kda.line_index` 按层取行，`blocks` 变 `[seqs, 8, H]`。
- 每层 launch 数：KDA/MoE 层 3 + 2 + 1 + 1 + 1 + 3(MegaMoE) + 1 + 1 + 1 ≈ 14 + 8 GEMM；MLA 层少 1。

## 4. 交付状态与遗留（2026-09-02）

七族全部交付并入 master（`tools/kernels-src/k3_*.cu`，`tools/build_kernels.sh` 编成 `target/cubins/`）；
每个核在 harness 上 B ∈ {1, 2, 8, 64} 全过，0 spill，无 `.MULTICAST`；notes/ncu 报告在 `tools/k3-harness/`。
生成器 `tools/gen_k3_decode.py` 已切到这套核（manifest `examples/k3-*.json`，93 层 1855 launch，其中 742 GEMM）；
pegainfer 的 TileLang 桶核、line shim 和它们的 manifest 已从树里删除（git 历史里有）。
门禁数字见 roadmap E2 行。

遗留（都是契约层面，核本身不用改）：

- K6 `k3_router_topk` 把 `EXPERTS=224` 烤进了核；满血 K3（896 expert）要另编一版或改成参数。
- K5 自写核 `k3_mla_paged_attn` 2026-09-03 被 DSL 核替掉（上文 K5、roadmap M1）；源码和 harness 条目
  留着（`tools/k3-harness/run_all.sh` 还跑它），生成器不再引用。
- K1a `nb == 8` 与 snapshot 不能同时用（snapshot 写 blocks 的第 8 槽）；生成器只在 `nb < 8` 时带 snapshot。
- "snapshot" 在 K1a（把当前 hidden 存进 blocks[8]）和 K1b（把 landing 后的 hidden 存进 blocks[8]）里含义不同，
  签名同名不同义，改名的事等消融。
- manifest 传不了空指针：K1c 用 `int two` 标志代替 `p2 == NULL`。
