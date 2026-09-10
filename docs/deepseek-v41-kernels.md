# DeepSeek-V4.1-Flash kernel 选型与接入规划

日期：2026-09-10。状态：源码调研，尚未做 GPU benchmark 或数值验证。

目标是在 4×GB300 上统一采用 DP4 attention + EP4 MoE，先选定文本 backbone 的
prefill 和 batch decode kernel，再生成 kern manifest。模型语义放在 `tools/`
生成器和模型 artifact，runtime 保持模型无关。本文持续记录候选、取舍和实测结果。

## 1. 依据与待核对口径

- 模型自带 `config.json`、`inference/model.py`、`kernel.py`、`engram.py`、
  `convert.py`、`generate.py` 和技术报告。本文配置和 reference 行为来自这些文件；
  本地 checkpoint 文件仍在更新，正式验证前需记录完整性及文件 hash。
- [DeepGEMM PR #432](https://github.com/deepseek-ai/DeepGEMM/pull/432)，
  本次读到的 head 为 `ab69f76be5bb9ea3499bc755002b1a876cb0b3d9`。
  新增 Sparse Indexer、Mega Gate、Mega mHC，并优化 V4.1 MegaMoE。
  PR 描述中的性能数字是上游结果，不是本项目实测。
- [该版本 README](https://github.com/deepseek-ai/DeepGEMM/blob/ab69f76be5bb9ea3499bc755002b1a876cb0b3d9/README.md)
  描述了 FP8/FP4 GEMM、MegaMoE、量化布局和 DeepJIT 接口。
- 本仓库已有 [MoE 通信选型](moe-comm-survey.md)、
  [K3 kernel ABI](k3-kernel-abi.md) 和 `tools/k3-mega/` 的接入基础。

技术报告（`DeepSeek_V41_Tech_Report.pdf` §3.2，第 18–19 页）的原文：
"the vast majority of Transformer layers—those whose CSA2 operates in Reuse
Mode—execute with only 15 kernels during prefill and 11 during decode"，融合核
点名 FlashMLA 的 fused-RoPE-attention-RoPE-cast、DeepGEMM 的 Mega-Gate /
Mega-mHC / Mega-MoE、TileKernels、DeepSelect 的 TopK。口径是**每层的 launch
数，只算 CSA2 Reuse Mode 的层**（复用上游层 KV 与 Top-K 索引，自身不跑
indexer、不写 KV），不含 source 层、Engram、DSpark 和通信。2026-09-10 核对
（registry manifest，kern f38924e）：我们的 reuse 层 prefill 与 decode 都是
27 个 call、30 次 launch（mHC 与 Mega-Gate 各带一次 `zero_u64`），source 层
31 次，另有每步 250 余次不归属于层的 launch（indexer、compressor、Engram、
cache 写入、head）。与 11 的差距全在未融合的胶水：4 次独立的 MXFP8 动态
量化（wq_a / wq_b / wkv / wo_b）、2 次独立 RMSNorm、3 次独立 RoPE（q /
kv / o，FlashMLA 的融合版本把它们收进 attention）、O-A 按 8 组各发一次
cuBLAS（分组 GEMM 可收成一次）、MoE 入口一次独立量化。主算子数量与报告
同量级，融合是性能阶段的事，不为凑数划分算子。

## 2. DP4 / EP4 的已知形状

| 项目 | 全局配置 | DP4 / EP4 下 |
|---|---|---|
| Backbone | 40 层，hidden 5120 | 相同 |
| Attention | 64 heads，head dim 512，RoPE dim 64 | 64 heads/rank，按请求分工 |
| Q low-rank | 1280 | Q-A / Q-B 完整复制，每 rank 处理本地 token |
| O low-rank | 8 groups，每组 rank 1024 | 8 groups/rank |
| Routed MoE | 384 experts，top-6，intermediate 2304 | 96 experts/rank |
| Shared expert | 1，intermediate 2304 | 每 rank 完整复制，处理本地 token |
| Indexer | 32 heads，head dim 128，top-512 | 32 heads/rank，score/top-k 本地计算 |
| Window | 128 tokens | 每层独立 window state |
| mHC | mult 4，Sinkhorn 20 iterations | residual 每 token 为 4×5120 |
| Engram | 2 张 FP8 表，分别约 384M 行，行宽 256 | 暂按行四分预算；跨 owner lookup 通信待选 |

Routed experts 每卡 96 个，其余计算按本地请求执行。DP4 下 dense、attention、
indexer 和 shared expert 权重完整复制，需要重新核算显存；**还没有完整加载的显存结论**。
预算需包含 Engram、expert scales/repack、复制权重、KV、MegaMoE symmetric
buffer 和 workspace，并分别记录稳态及加载峰值。

以下层号均为 **0-based**：

- 0、1 层为 window-only；2–19 压缩比为 2，20–39 为 1。
- KV source：2、8、14、20。
- Index source：2、8、14、20、24、28、32、36。
- Candidate source：20，candidate top-2048 blocks，block size 8。
- Engram：1、14。

需要分别列出普通复用层、KV/index source 层、candidate source 层和 Engram
附加路径的 launch 表；同一模型不能简单用一个“每层 kernel 数”描述。

## 3. Kernel 候选表

以下是逻辑阶段，不是 15/11 kernel 清单。首选表示准备验证的候选，不代表可直接接入。

| 阶段 | Prefill 候选 | Batch decode 候选 | 需要确认的接口/语义 |
|---|---|---|---|
| mHC + RMSNorm | PR #432 Mega mHC | 同一 kernel family 的小 M 配置 | 延迟使用的 pre mix、post/residual、20 次 Sinkhorn、两种 epsilon，量化输出能否融合 |
| Q-A / Q-B / window KV projection | DeepGEMM FP8 GEMM | DeepGEMM 与 cuBLASLt 按小 M 比较 | checkpoint 32×32 block scale、activation scale、可合并 projection 的布局 |
| Q-A latent norm / KV prep | latent RMSNorm + KV norm/RoPE/quant/cache 写入融合候选 | 每序列独立 position 的 paged/ring prep | Q 的 RoPE 交给 FlashMLA fused 核；Q-A latent norm 仍保留；window FP8 与 compressed FP4 格式不同 |
| KV compressor | 先按 reference 建 oracle，再选融合边界 | 增量 compressor state kernel | ratio 2 的未满组、chunk 边界、跨层 cache 复用 |
| Index Q/K projection 与量化 | DeepGEMM GEMM + fused prep | 小 M GEMM + fused prep | index FP4 block 32 / UE8M0，与 compressed KV block 16 / E4M3 scale 区分 |
| Index scoring / candidate / top-k | PR #432 Sparse Indexer，核对完整链路 | 核对 paged 和变长 batch 路径 | PR 的 scoring 不默认包含 top-k；candidate 强制保留最新部分块、causal mask；DP4 下 score 本地计算 |
| Sparse attention + Q/O RoPE + output cast | FlashMLA #221 fused prefill，普通 sparse prefill 作对照 | FlashMLA #221 fused decode，普通 sparse decode 作对照 | 每 rank 64 heads；FP8 window + FP4 extra cache；sink、top-k sentinel、量化误差与输出布局 |
| O-A / O-B | DeepGEMM FP8 einsum + GEMM | 小 M einsum/GEMM | 每 rank 8 组，每组 K=4096、N=1024；接 FlashMLA FP8 输出需 permute Q-B/O-A 权重与 scales，reference BF16 O-A 作数值对照 |
| Router | PR #432 Mega Gate | 同一 kernel family 的小 M 配置 | sqrtsoftplus、top-6、bias 只影响选 expert、归一化后乘 1.5；不能复用 K3 router 语义 |
| Routed experts | PR #432 FP8×FP4 MegaMoE | 同一 MegaMoE 的低延迟配置 | hidden 5120/inter 2304/EP4、clamp 10、route weight 应用位置、权重 packing 和 symmetric buffer ABI |
| Shared expert | FP8 GEMM + SwiGLU + GEMM | 小 M GEMM；测试与 routed 路径重叠 | 完整复制、处理本地 token；不能假设 MegaMoE 已包含 shared expert |
| Engram | hash + lookup/dequant + projection + gate | 同一语义的增量 hash、批量 lookup | token 历史跨 chunk/请求隔离，lookup 通信、signed-sqrt gate |
| 通信与数据布局 | EP dispatch/combine + Engram lookup | 不均匀本地 batch 下的同步与通信 | 所有 layout/copy/barrier 单独计时，不隐藏在 GEMM 数字里 |

输入 embedding、末尾 mHC/head/sampling 另列全模型固定开销。Vision 与 DSpark
在文本 plain prefill/decode 基线之后接入；此阶段不把它们算作已支持。

### 3.1 完整执行清单与手写边界（plain text DP4/EP4）

下面按依赖顺序列逻辑算子，**编号不是 launch 数**：一个上游 API 可能启动多个
kernel，也可能一个融合核覆盖数项。量化、通信、gather 和 metadata 均算入路径。
来源固定为前述 DeepGEMM、FlashMLA 和 vLLM SHA；未编译/跑数值的项仍是候选。

记号：**复用** = 已有计算实现；**适配** = 改 ABI、布局或 state 接线；
**小核候选** = 可借上游实现，但为 kern 的布局/融合可能编写小核。

| 顺序 / 条件 | 逻辑算子与形状 | 来源与处理 |
|---|---|---|
| 请求/step 输入 | token/position/slot、每请求长度、页表、有效行 mask | 复用 kern caller；DP 每卡不同 batch 的同步接线要适配 |
| 模型入口 | token embedding → `[T,5120]`，初始 mHC residual `[T,4,5120]` 和 identity pre-mix | embedding 复用；broadcast/初始 mix 可适配 copy/fill 或写小核 |
| 入口，每步一次 | 计算两个 Engram 层的 n-gram hash，读取至多 3 个历史 token | vLLM `NgramHashState` / `_hash_ids_kernel`；适配 kern 请求状态，不能跨请求读取历史 |
| 入口，提前准备 | 两张 host 表查行 + per-32 scale 反量化 → GPU `[T,24,256]` | vLLM `_engram_lookup_kernel`；共享 host backing 是 host/runtime 绑定工作，不是新计算核 |
| 每层开始 | 上一子层 post/residual 混合；本层 mHC mixes/Sinkhorn、delayed pre-collapse、RMSNorm/FP8 quant | DeepGEMM `mega_mhc` shifted 模式；支持同时输出 BF16/FP8 及不同 scale 布局。首层和 Engram 边界见下 |
| 仅层 1/14，mHC pre 之前 | lookup flatten `[T,6144]` → WKV `[T,25600]` → normalized dot / signed sqrt / sigmoid / residual 注入 | GEMM 复用；vLLM `_fused_engram_post_wkv_kernel` 复用/适配。必须先完成上层 post 再注入，随后才能算本层 mixes |
| 每层 Attention | 合并 Q-A / window WKV projection：5120 → 1280+512 | vLLM 已合并权重；DeepGEMM GEMM，离线拼接权重/scales |
| 每层 Attention | Q latent RMSNorm+FP8 quant；window KV RMSNorm | vLLM `fused_q_kv_rmsnorm_quant`；复用/适配输出 strides/scales |
| 每层 Attention | Q-B：1280 → 64×512 | DeepGEMM GEMM；FlashMLA fused 路径要求离线 permute 权重/scales |
| 每层 Window | KV RoPE + FP8 quant + paged/ring cache insert | vLLM CUDA fused prep / cache insert；改为匹配 FlashMLA page packing。Q RoPE 若融合进 attention，此处不重复做 |
| 仅 KV source 2/8/14/20 | compressor projection：5120 → 1024（ratio2 的 KV+gate）或 512（ratio1） | BF16 GEMM，权重按 reference/vLLM 精度；不默认把所有 projection 都转 FP8 |
| 同上 | ratio2 保存未满组、softmax 加权 pooling + RMSNorm；ratio1 仅 norm | vLLM `fused_save_compress_norm`，ratio 1/2 已实现；适配 state slots 和 chunk 边界 |
| 同上，分支 A | latent RoPE + FP4 block16/E4M3 scale + compressed cache insert | vLLM `rope_quant_insert`；复用/适配 |
| 同上，分支 B | latent → index K projection 512→128、norm、RoPE、FP4 block32/UE8M0、index cache insert | GEMM + vLLM `indexer_k_norm_rope_store`；K 必须来自 RoPE 前 latent |
| 仅 8 个 index source | index Q projection 1280→4096；weights projection 5120→32；Q RoPE/FP4 quant | GEMM + vLLM index Q prep；各 dtype 和 scale 乘入位置逐项对照 |
| 同上 | index scoring：32 heads ×128，ReLU 后加权求和 | DeepGEMM MQA / #432 Sparse Indexer；DP4 本地完成，不做 TP score 归约 |
| 仅 layer20 | score 每 8 positions 求 max，最新部分块强制可见，取 top-2048 blocks | vLLM `candidate_blocks.py` 有 reduce/store 核，但选择仍调用 `scores.topk`；完整 device top-k 接入是明确待解决项 |
| 后续 index source | 仅对候选块评分/过滤，再取 top-512 positions | 优先 #432 sparse scoring；vLLM full-score + candidate mask 作对照。复用 top-k selector，核对输出位置排序、causal/invalid 语义 |
| 短 context | 可见 compressed positions 不超过 top-k 时，直接生成全部可见 indices | vLLM `_fill_short_context_topk_indices`；可复用，保留 candidate 语义检查 |
| 每层 Attention，cache/index 复用层跳过上面的生产步骤 | 构造 window indices、logical top-k→physical page offset、有效长度 | vLLM `compute_global_topk_indices_and_lens` / `combine_topk_swa_indices`；kern 页表适配，小核候选 |
| Prefill 专有 | 从量化 window/compressed cache gather/dequant 到 BF16 workspace，并 remap indices | vLLM `dequantize_and_gather_k_cache`；已有 Triton 实现。控制 chunk/workspace，不能漏算此成本 |
| 每层 Attention | Q RoPE → window+compressed sparse attention/sink → inverse O RoPE → FP8 cast | FlashMLA #221 fused prefill/decode 首选；普通 attention + vLLM inverse-RoPE/quant 作对照。Decode 直接读 FP8/FP4 两 cache |
| 每层 O projection | O-A 8 groups：4096→1024；O-B 8192→5120 | DeepGEMM FP8 einsum/GEMM；中间 BF16→FP8 quant 需要单列，能融合才消掉 |
| Attention→FFN | post/residual + delayed mHC pre/RMSNorm/quant | DeepGEMM `mega_mhc`；输出 BF16 给 gate、FP8/scales 给 MoE/shared，尽量直接落目标 buffer |
| 每层 FFN | router GEMM 5120→384 + sqrtsoftplus + selection bias + top6 + unbiased normalized weights×1.5 | DeepGEMM `mega_gate` 已支持 sqrtsoftplus，测试有 hidden5120 / 384 experts；复用，无需新写 router |
| 每层 Routed | EP dispatch → FC1(5120→4608) → clamp/SwiGLU → FC2(2304→5120) → combine 回 token owner | DeepGEMM FP8×FP4 MegaMoE；适配 EP4 symmetric memory、weight layout、TMA/launch ABI。96 experts/rank |
| 每层 Shared | FC1 5120→4608 → clamp/SwiGLU/quant → FC2 2304→5120 | GEMM 复用，确认可用的上游 clamped activation；现有 `silu_mul.cu` 无 clamp 且舍入顺序不同，不能原样使用；独立小核候选 |
| FFN 结束 | routed+shared 相加，post/residual 与下一子层衔接 | 可先用 add/post 核或下游融合；不假设 MegaMoE 自动包含独立 shared 输出 |
| 模型尾部 | 最后 post、按最终 pre-mix collapse、final norm → LM head 5120→129280 → greedy argmax | mHC collapse/norm/GEMM/argmax 复用；prefill 只为各请求末 token 算 head，last-row gather 适配 |

中间 scale repack、FP8 activation quant、EP 输入 staging、barrier/metadata 准备
均需要从 API 展开成实际 launch 表；优先离线 weight repack 和让前驱直接写目标布局。
有源代码不等于可以直接把 Python 调用放进 kern：需要导出 cubin，并把 host wrapper
做的 workspace、TMA descriptor 和调度参数计算移到 artifact builder/caller。

### 3.2 目前真正需要我们写什么

**尚未发现必须从零实现的 GEMM、attention、MoE、router、mHC、compressor 或
Engram 数学核心。** 明确新增的是模型生成器、加载期权重排列、ABI 包装、共享 host
表绑定和 DP/EP 调度；这些多数是 host/工具代码。

Device 侧先锁定下面几个小范围，选型后再决定移植还是手写：

1. **索引与 state glue**：每请求 window/physical indices、有效行 mask、compressor
   state slots、Engram lookback 接线。vLLM 都有参考实现，改成 kern 布局。
2. **Candidate block top-k**：vLLM 此处还依赖 PyTorch `topk`；为无 PyTorch
   runtime 接入现有 CUDA/CuTe selector，或写专用 selector。先核对 #432 是否
   已覆盖该选择环节。不能拿单个 argmax 或未经验证的现有 topk 核替代。
3. **Shared expert clamped SwiGLU + quant**：优先提取上游匹配语义的核；若输出
   scale layout 不匹配，写融合小核。gate 仅上界 clamp，up 双边 clamp，限值 10。
4. **mHC 特殊边界和输出合并**：普通层用 Mega mHC；首层 identity/broadcast、
   Engram 层 post→inject→pre、最终 collapse 先复用 vLLM 分开的核，按 profile
   决定是否写专用融合。routed/shared 相加也放在这里核对，避免重复或遗漏。

先完成以上路径的数值对照与 launch 展开，再决定优化性手写；不能提前承诺只需
固定数量的新 kernel。DSpark 额外 state 提交/回滚见后文，不计入 plain launch 表。

## 4. 统一 DP4 → EP4 → DP4 数据流

Prefill 和 batch decode 使用相同并行布局。每条请求归属一个 DP rank，
attention/indexer/mHC/shared expert、KV 和 compressor state 都由该 rank 管理。
每 rank 保留完整 64 Q heads、32 index heads 和 8 个 O groups。

```text
本地请求 → mHC / attention / indexer → mHC / router
                                      ├─ 本地 shared expert ─┐
                                      └─ EP4 MegaMoE ────────┤
                                           combine 回 owner ┘
                                      → 本地 residual / 下一层
```

EP4 每 rank 持有 96 个专家，接收所有 rank 路由过来的 token；结果返回原 token
owner。attention 前后不需要 TP all-reduce、reduce-scatter 或恢复复制 hidden。
单条长 prompt 的 attention 由一个 rank 完成；四卡通过多个请求并行提升吞吐。

小 batch、不能被 4 整除的 batch、某 rank 零 token、不同 rank 的 prefill chunk
长度都必须显式处理。即使某 rank 没有本地请求，仍须参与其他 rank 的 EP 计算与
协议同步。调度器需要统一 MoE collective 的层次/调用顺序，不能把四个 DP worker
当作完全独立的循环。先验证统一阶段与 bucket 的执行，再评估更灵活调度。

batch 大小统一记为 **全局有效 token/序列数**，同时记录每 rank 行数和 padding。
Prefill chunk 大小按每 rank 本地 token 数报告；同时记录单请求 TTFT 与四卡吞吐。

Engram 大表独立选型，尚未决定最终放置方案。vLLM #56201 的源码调研补充见下：
其默认 CPU offload，不是跨 DP 的 GPU 分表。初始按行四分仅是显存预算假设；
接下来比较共享 host 表的 UVA lookup 与 GPU 分表 lookup，计入实现、NUMA 和通信成本。

自带 reference 的 `MP=4` 会同时启用 TP，并不等于本方案。其转换产物也不能
直接用于 DP4/EP4；各 rank 的 manifest 应直接 bind 原 checkpoint 的完整
dense/attention/index/shared 张量及本 rank 的 experts，单独处理 Engram。
Reference 继续用于算子语义对照。

### FlashMLA #221 接口核对

[FlashMLA #221](https://github.com/deepseek-ai/FlashMLA/pull/221) 已合并。
本次检查固定 head `4f38f29ef6793c228363e4af5be66d44e81167ba`，来源为该版本
[README](https://github.com/deepseek-ai/FlashMLA/blob/4f38f29ef6793c228363e4af5be66d44e81167ba/README.md)
及 `csrc/api/{sparse_prefill,sparse_decode,fused_norm_rope_attn_rope_cast_fwd}.cpp`。

- 普通/融合 sparse prefill 与 decode 入口接受 64/128 Q heads；DP4 的本地
  64 heads 符合这一约束。源码包含 sm_103a 构建目标，实际 GB300 验证待做。
- 融合路径包含 Q-RoPE、attention、逆 O-RoPE、FP8 output cast。V4.1 设
  `enable_q_norm=False`，这里关闭的是 head Q-norm，不是 Q-A 后的 latent RMSNorm。
- Decode 主 cache 为 V4.1 FP8：每 token 512B data + 16B UE8M0 scales；
  compressed `extra_k_cache` 为 FP4：256B data + 32B E4M3 scales。
  每页先排 data rows，再排 scale rows，不能按普通逐 token interleaved packing 写入。
- Prefill 接口使用 BF16 KV，需核对如何从量化 cache 构造与 oracle 相同数值的
  输入以及相关成本；不能假设 decode 的双量化 cache 接口同样适用于 prefill。
- Decode indices 编码物理 page/offset，支持独立 extra indices 和长度。
  Prefill 批量请求按 token 维展平并调整 indices，必须保证请求间不可见。
- 融合路径要求提前 permute Q-B / O-A（上游称 Wv）及 scales，并让 DeepGEMM
  FP8 einsum 接收输出。README 示例的 q_lora_rank 等不等于 Flash 配置；使用
  本模型的 1280 / 1024，逐项验证量化粒度和排列约束。

### vLLM #56201：Engram 的实际实现

检查版本为 `61140208c5940195b5970fc44de77a9695d0baf7`；PR 仍在更新，
以下描述仅针对这个 SHA。
[配置](https://github.com/vllm-project/vllm/blob/61140208c5940195b5970fc44de77a9695d0baf7/vllm/config/engram.py)、
[Engram 实现](https://github.com/vllm-project/vllm/blob/61140208c5940195b5970fc44de77a9695d0baf7/vllm/models/deepseek_v4_1/common/engram.py)、
[模型入口](https://github.com/vllm-project/vllm/blob/61140208c5940195b5970fc44de77a9695d0baf7/vllm/models/deepseek_v4_1/nvidia/model.py)。

- 默认 `cpu_offload=True`。权重和 scales 分别分配在 pinned CPU memory；GPU
  通过 UVA view，在 Triton lookup kernel 中直接读取需要的行并反量化到 GPU BF16
  staging buffer。不是每步拷贝整张表，也没有在此路径按 EP/DP owner 发送请求。
- 分片维度是 **TP 内的完整 hash-head buckets**。每层 24 个 hash columns
  （3 种 n-gram 长度 × 8 heads）；TP4 时每 rank 6 个完整 buckets。每个 bucket
  行数略有不同，不是简单将总行数平均四分。
- TP>1 时各 rank 查询相同 token 的不同 hash heads，再 all-gather 拼回 head
  维；SP 路径 gather 后取本 rank 的 token 行。TP1 时直接返回本地 lookup 结果。
- **DP4/TP1/EP4 原样使用会产生四份完整 host 表**：源码每个实例独立
  `torch.empty(..., device="cpu", pin_memory=True)`，没有跨 DP 的表存储共享。
  按本模型两张表及 per-32 UE8M0 scales 计算，一份约 188.83 GiB，四份约
  755.33 GiB，仅为表的常驻数据，不含其他 host 内存和加载峰值。
- 模型 forward 开始先算整批 hash，再在进入 decoder layers 前为所有本地 Engram
  层调用 `prepare_embeddings`。当前这条路径在 main stream 预取；虽然 lookup
  提供 `background` 参数，不能据此称默认路径已经异步隐藏全部 host 访存。
- Lookup 将 FP8 + scale → BF16 融在一个 Triton kernel；采用受 SM 数限制的
  persistent grid。之后 replicated WKV projection，再用 fused kernel 做归一化
  dot、signed sqrt、sigmoid gate 和 residual 注入。
- n-gram 历史采用请求隔离的 lookback token IDs / slot cache，覆盖 chunk 边界；
  V2 runner 直接提供设备端历史，V1 用 slot cache 补足 async decode 的历史。

对本项目的启示：把 **单份只读 host 表共享映射 + 各 GPU UVA lookup** 列为
优先验证候选，可以省 HBM 并避免四份 host 表；这不是该 PR 已实现的能力。
kern 当前是一 tray 一进程、每卡一个 Runtime（见 [多卡设计](multi-gpu.md)），
因此同 tray 可由 caller 持有一份 host allocation，四个 Runtime 引用各自 GPU
可访问的映射；无需跨进程共享 backing。此 host 表绑定能力尚待实现。
需要验证 pin/register 生命周期、各 GPU 可见性、NUMA 放置、
随机 lookup 吞吐及 CUDA Graph 支持，再与 GPU 分表路径比较。

另一个修正：vLLM 的
[FlashMLA wrapper](https://github.com/vllm-project/vllm/blob/61140208c5940195b5970fc44de77a9695d0baf7/vllm/models/deepseek_v4_1/nvidia/flashmla.py)
会把少于 64 的本地 Q heads pad 到 64，外层 attention 在输出后切回有效 heads。
因此 TP4 可以通过 padding 使用上游核，并非必须新写 16-head kernel；计算效率
需要测量。我们按用户决定继续统一 DP4/EP4，此处仅修正接口可用性的判断。

## 5. 实施顺序与验收

### A. 源码与 launch 清单

- 提取技术报告的 15/11 原文，补上页码、层型和统计口径。
- 固定 PR SHA，逐项读 Mega Gate / Mega mHC / Sparse Indexer / MegaMoE 的
  API、实现和 tests；同时核对 FlashMLA #221 普通/融合 prefill 与 decode，
  标出不满足 V4.1-Flash 形状或语义的约束。
- 建立每种层型的 prefill/decode 数据流，记录 dtype、shape、scale layout、
  state owner、通信边界和实际 launch 数。区分 Python API 调用数与 GPU launch 数。

### B. 单模块选型

- 先测 MegaMoE、Mega Gate、Mega mHC，确认已有 K3 接入中可复用的 ABI 部分。
  K3 fork 带模型特定激活，不能直接作为 V4.1 算法实现。
- 随后测 Sparse Indexer + top-k + attention 完整链；同时补 compressor/prep。
- 再测 dense/shared/grouped O GEMM 与 Engram，按测到的占比确定融合优先级。
- 每个模块都做 oracle 对照和目标 shape 的 timing；数据格式转换成本计入路径。

初始测量矩阵（按内存预算裁剪）：

| 维度 | 初始取值 |
|---|---|
| Prefill chunk tokens | 128、512、2K、8K、16K |
| 已有 context | 0、4K、32K、128K |
| Decode 全局 batch | 1、4、16、32、64、128、256 |
| Decode context | 128、4K、32K、128K，另测混合长度 |
| Routing 分布 | 真实 reference 路由、均匀分布、热点/偏斜 |

先用代表点淘汰不合适候选，再扩展矩阵。记录 GPU 时间、全路径时间、
workspace、通信、padding、launch 数和数值误差；decode 测 CUDA Graph replay。
重复文本不能作为完整生成正确性样例。

### C. 单层 → 全模型 artifact

- 对 window-only、KV source、index-only source、复用层、candidate source、
  Engram 层分别验证，重点覆盖 ratio 2 奇偶位置、window wrap 和 chunk 边界。
- Oracle 除张量误差外，检查 routed expert IDs/weights、candidate 集合、top-k
  可见位置、KV 状态及最终 logits；近 tie 单独分析，不能只看输出“像中文”。
- Reference 当前主要区分 `start_pos == 0` 的 prefill 与单步 decode；任意 chunk
  prefill、变长 batch 需要扩展 oracle/test harness，不能假设原脚本已覆盖。
- 导出 pinned cubin/ABI 和直接绑定原权重的模型生成器，运行 manifest verify、单层对照、
  多请求状态隔离、完整 logits/生成对照，最后测 TTFT、TPOT 和吞吐。

实验统一落在 `~/bench_results/YYYY-MM-DD-deepseek-v41-kernels/`，包含
`README.md`、`scripts/`、`results/`；本文只记录结论和相对 artifact 名称。
使用 GPU 前逐卡确认空闲，不占用已有作业。当前尚未启动 GPU 实验。

## 6. DSpark：复用范围与后续接入

V4.1 checkpoint 自带 `mtp.0/1/2` 三层 DSpark，block size 5、Markov rank 256，
读取 target 的 37/38/39 层特征；draft routed experts 为 128、top-3（EP4 每卡
32 experts）。embedding / LM head 与 target 共享。阶段顺序仍为先完成 plain
prefill / batch decode，再接 DSpark；现在设计 state 和 artifact 时保留正确边界。

[vLLM DSpark 实现](https://github.com/vllm-project/vllm/blob/61140208c5940195b5970fc44de77a9695d0baf7/vllm/models/deepseek_v4_1/nvidia/dspark.py)
直接复用 Qwen3 DSpark 的 Markov / confidence head 类及 V4.1 decoder layer。
其 draft 用 sparse indices 表达块内非因果可见性；每层 context KV 均从同一份
投影后的 target 特征经该层自己的 WKV/norm/RoPE/quant 生成。

| 部分 | kern 可复用内容 | 需要适配/验证 |
|---|---|---|
| Round 输入拼接 | `spec_round.cu` 的 splice draft/verify | block、mask token、draft stride、anchor 位置 |
| Greedy 接受计数 | `kern_spec_count`、count 输出协议 | 仅限 greedy 前缀匹配；随机采样不能直接用 token equality 代替 rejection sampling |
| Markov 链 | `markov_rows.cu`、embedding/GEMM 累加/argmax，现有 Qwen3 DSpark 生成器模式 | vocab 129280、rank 256、5 行 block；重新生成形状与权重绑定 |
| Draft 主干 | V4.1 的 GEMM、mHC、router、MegaMoE kernel families | 128 experts / top-3 的实例，3 层参数；不是直接复用 target cubin 配置 |
| Draft attention | FlashMLA sparse attention / KV prep 候选 | window context + block 内非因果 indices；无 compressed indexer/Engram；验证融合输出布局 |
| Context precompute | GEMM、norm、RoPE、量化/cache insert | concat target taps → 15360→5120 projection + norm；每 draft 层独立 WKV，不能照搬 Qwen KV |
| State 提交 | 现有按接受数推进与 line 选择的组合方式 | V4.1 window、compressor、跨层 KV/index 和 Engram 历史的回滚/覆写必须独立验证 |
| Confidence head | 小 GEMM + sigmoid 候选 | 是否用于动态截断、权重缺省行为与策略需另定 |

`dspark_block_size=5` 不自动等于我们的 round rows=5：reference 输出五个 draft
预测，若全部验证还涉及 anchor 行。需明确 draft/verify 行数及丢弃最后一个预测的
取舍，再绑定现有通用 splice/count 核。验收先做 greedy 与 plain 的 logits/输出
对照、拒绝后 state 正确性，再测接受率及每秒有效 token；不能只报 draft 速度。

## 7. 决策记录

| 日期 | 决定 | 依据 / 后续门禁 |
|---|---|---|
| 2026-09-10 | 用户确定 prefill / batch decode 统一 DP4/EP4 | 每 rank 完整 64 heads，匹配 FlashMLA；重新核算复制权重与 Engram 通信 |
| 2026-09-10 | Attention 首选评估 FlashMLA #221 融合路径 | 普通路径作对照；KV packing、权重 permute、量化边界待验证 |
| 2026-09-10 | 优先评估 DeepGEMM #432 的融合核 | 上游已提供 V4.1 相关路径；本地形状、数值、性能待测 |
| 2026-09-10 | 自带 Python 作为语义 oracle | 包含完整新架构，但不是生产 batch decode 性能基线 |
| 2026-09-10 | 暂不认定 15/11 为实际 launch 目标 | 需先核对官方口径和各层型差异 |

## 8. 落地记录

2026-09-10：新增 `tools/dsv41_weights.py`，无需 Torch 或 GPU，读取全部 safetensors
header，校验索引映射、连续 payload 范围及文件长度，输出每 tensor 的 dtype/shape/
offset、header hash 和 DP4/EP4 放置计划。48 个 shard、96,085 个 tensor 检查通过。
这不等于 payload 校验或模型数值验证。

实测 checkpoint 原始字节：每 GPU 文本+DSpark 权重 **78.752 GiB**，共享 host
Engram **188.833 GiB**。此值包含原始 scales，不含额外 repack、副本、KV 和
workspace；dense 中保留了原 checkpoint 的辅助参数。Vision/aligner 单独分类。
完整 tensor inventory 和结果保存在本次实验目录 `results/weight-plan.json`。

构建环境检查：现有容器 Torch 2.11.0+cu130、TileLang 0.1.12；reference 要求
TileLang 0.1.8，后续固定隔离依赖后再做数值基线。尚未构建 V4.1 kernel artifact，
尚未跑 GPU/HTTP，也尚未验证 DSpark 接受长度。

### 加载与验证流程修正

已检查 `tools/qwen_weights.py`、`tools/kernels-src/weight_prep.cu`、
`crates/kern-runtime/src/weights.rs` / `load.rs` 和 `crates/kern-run/src/run.rs`。
当前正式路径是原 HF safetensors → `bind` 拼接/矩形切片 → 加载后执行 manifest
`once` program，派生结果放 `carry`。不需要另造 checkpoint，也不为 DP/EP 分片
导出权重。K3 的旧 exporter 不能作为新模型默认流程。

- 合并 Q-A/WKV、专家分配：生成 `bind` 即可。
- 非平凡 permutation/FP4 repack：复用上游重排核，放 `once`；现有
  `weight_prep.cu` 只提供 cast/fill/表生成等基础核，并未已有 V4.1 特定重排。
- 确认的加载缺口：`weights::dtype_of` 尚不接受 checkpoint 的 `I8` 和
  `F8_E8M0`。需要按类型契约补充支持，不能未经声明把它们当普通 U8。
- `once` 可以做转换，但不能据此假设原始与转换后权重的双份存储会自动释放；
  需核查并预算原始 weight 与 carry 生命周期。共享 host 表仍需专门的通用绑定能力。
- `tools/dsv41_weights.py` 仅保留清点/放置计划用途，不是新权重格式或离线转换前置步骤。

验证来源分三层：上游 pinned kernel tests 证明其算子契约；独立运行的 pinned
vLLM + 原 checkpoint 提供完整 prefill/变长 decode/DSpark 参考；自带 inference
用于解释模型语义和不一致处（其简化 batch/chunk 支持不能代替 serving 基线）。
这些参考当前尚未跑通，不宣称已有可信 V4.1 oracle。

现有 `kern test` 比较两份 manifest，默认 A 已可信，不会自动调用 vLLM 或官方
Python，也不能用自己生成的两个相同实现证明新模型正确。先建立上游执行的
输出/必要边界快照，验证 kern 的整体接线；只有差异、重排或改动融合边界处才做
局部数值对照。单层验证是定位工具，不是为全部上游 kernel 重写 reference 的
固定前置工程。可信 manifest 建立后，再用现有 cut/tap/noise-floor 机制验后续替换。

### 并行落地与 program 组合

用户授权三个 agent 并行处理 attention/indexer、MoE/mHC、Engram/auxiliary，
统一对照 checkpoint 自带 inference。GPU 测试使用 Slurm 已分配的四卡节点，
单核测试分卡运行，四卡 EP 测试需先协调独占窗口。

新增 `tools/dsv41/programs.py`，接收已降低为 call 列表的各阶段，组装
`load`（once）、`prefill`（动态 rows）、`decode_batch`（rows1）、`round`
（rows6）。Stages 仅为生成器内组合，没有 program 调 program。内部 draft
计算五行，verify 计算 anchor+全部五个预测；count 最大6，保留全部预测。
plain 和 prefill 同时刷新 draft context，避免切换投机模式时上下文陈旧。
目前阶段 call 尚待各模块和 buffer/state 绑定，不能作为完整 manifest 运行。

复用了现有 `spec_round.cu`，新增 GPU 验证脚本 `tools/dsv41/test_round.py`。
实际 sm_103a cubin 对 batch6/17/128 的输入拼接及接受计数1..6全部通过。
这是拼接/greedy count 的功能验证，不是 DSpark 的模型接受率证据。

新确认的接入事项：当前 Segment 是固定 tensor 名和固定矩形，rank 只可用于
call/launch 标量，不会自动插值 bind；同一 SPMD manifest 直接加载原 experts
需要补通用 rank-aware binding 或等价的加载期解析，不能隐含依赖已导出的
rank-local 文件。dtype、共享 host 只读映射也尚未接入。程序组装继续使用
现有协议，新增通用加载能力会单独验证。

### DeepSelect v1.0.0 接入调整（2026-09-10）

上游当天发布 [DeepSelect](https://github.com/deepseek-ai/DeepSelect)，固定到
[8e70df71d2a4b0c969ef96dc3b8998efa09a3315](https://github.com/deepseek-ai/DeepSelect/commit/8e70df71d2a4b0c969ef96dc3b8998efa09a3315)。
初始发布已包含 BF16 indexer 和 FP32 sampling TopK；最新提交删除
`do_check_nan` 参数及结构字段，提取 ABI 时必须使用相同 revision。

- Attention agent 优先从此上游提取 token TopK512 和 candidate TopK2048 的
  cubin/modules/ops，替代待实现的自写 TopK。
- 512 有按 batch waves 选择的配置；2048 使用 MAX_TOPK4096 档，
  上游明确称其为 correctness-only coverage，尚不能据此宣称 candidate
  路径性能达标。小 batch、超长行另有 cluster 分支，按实际形状选取。
- 请求 int32 indices、无需 values；排序取决于下游的实际要求。
  `end` 支持逐行有效长度；短行填充值必须显式设为 -1，
  默认 INT_MAX 不符合当前 attention indices 契约。
- `begin` / `hint` 尚不支持；遵守输入及输出的 stride 对齐。
  BF16 不支持按 value 排序。对照原模型时检查同分项选择及 mask，
  不能把无序 indices 的逐字节不同直接判为错误。
- DeepSelect 负责分数选择，不替代 DeepGEMM 的 indexer 打分或 FlashMLA。
  FP32 sampling 分支也不是完整 DSpark Markov head。
- 当前状态：已审阅上游接口和 dispatch，已委派编译及实际 kern 回放；
  尚未取得本项目接入精度或性能结果。README 的 2–20 倍相对
  torch.topk 加速是上游报告，不能作为本项目实测。

### 主存绑定与 FP32 head 进展

- Schema5 增加 `placement: "host"`，只允许非 export 的不可变 weight，
  不允许 rank-selected tensor。普通 EP expert 仍在设备上按 rank 选源。
- Runtime 在编译 program 之前分配稳定的 mapped host 地址；显式
  `HostWeights` scope 代表一个 checkpoint snapshot。serving 各 rank
  共用该 scope，CPU 按原 bind copy plan 初始化一次，后续 rank 复用。
  初次绑定完成之前执行 program 会报错；同一 runtime 不允许重载主存权重。
- Manifest 测试、runtime clippy、独立 serving workspace 的 cargo check
  通过。四 GPU 映射及 Runtime 层加载测试由 Auxiliary agent 继续验证；
  全尺寸 Engram 表加载和请求验证仍未完成。
- `tools/dsv41/head.py` 展开五步串行 Markov 链，使用现有 BF16-input /
  FP32-output GEMM，FP32 bias-add 与 greedy argmax 合并。避免把 head logits
  降成 BF16 后引入同分项。GPU driver 调用对照模型原始 `sample(..., 0)`：
  batch1/17/128、五个跨行 stride、FP32近邻和精确同分项均通过。
  此测试尚不覆盖真实权重 Markov 投影或完整投机接受率。

### 实际加载与整段 head 回放

- `loading.py` 从原始矩阵形状生成 226 个 dense scale packing once 调用，
  为不同 TMA 几何保留独立 op 名，模块同名不同 hash 会立即拒绝。
  `test_loading.py` 已用原始 checkpoint 的 226 个真实 scale 绑定运行
  kern once；全部 packed bytes 与独立展开/转置参考一致。包括 O-A 八组布局。
  新增 packed scale 合计 183,738,368 字节/rank。
- `test_head_program.py` 经实际 kern 执行完整五步 Markov 链和 CUDA Graph，
  batch1/17 与模型未修改的 `DSparkBlock.forward_head` 一致（替换的是
  synthetic linear weights 和上游归一化输入，temperature0）。不只是孤立 argmax。
- mapped host Runtime 集成测试通过：二维 strided bind、GPU GEMM 读取主存、
  加载前执行拒绝、重复加载拒绝、两 runtime 共享分配、释放首 runtime 与 scope
  后第二 runtime 的 graph 重放。四 context 底层映射读取测试也通过。
- 全尺寸 Engram 表、完整 target/draft 层流水线及 HTTP 请求仍待验证。

### Program 串接约束补充

- `forward.py` 生成 BF16 输入→动态 MXFP8 quant→dense/O-A 的实际调用，
  以及 Q-LoRA、Q norm、Q-B、KV projection、KV norm 的调用链。
  动态128行容量的 Q/KV manifest 已通过 JSON schema；真实权重 GPU
  projection 回放尚待四卡 MoE 测试窗口结束。
- TMA 的 dims/strides 是加载期静态值，必须用 buffer capacity；
  当前行数仅传 scalar/grid。Activation SF 的列 stride 固定为
  align(capacity,4)，不能按当前 batch 改写。
- `blocks.py` 维护待合并的 residual/update/post/comb/pre，通过下一次
  Mega mHC 合并上一个 hc_post；仅 Engram、target tap 和 final head
  边界 materialize。两个 buffer bank 避免输入输出别名，flush 后保留
  shifted pre，只重置 previous post/comb。
- 新增通用 `fill: "valid"`：real row=1，bucket padding=0，覆盖所有
  speculative rows。Token0、position0 和合法 pad lease 都不能用于判断
  padding。DP空rank与六行round布局测试通过；manifest tests和serving
  all-target clippy通过。Auxiliary metadata provider将用valid屏蔽状态写入。

### MoE 全层调用与权重准备

- `loading.expert_weights` 为40个target层和3个draft层生成86组
  routed/shared weight布局、11937个once调用，仅6个专用op定义。
  gate/up交错与UTCCP scale packing新增48.2853 GiB/rank。
  W2原始i8/FP8 buffer直接传入，避免为了指针类型转换再复制权重。
- `moe_forward.py` 将真实Gate直接写入exported byte slab，然后quant，
  最后调用EP4 MegaMoE。每个expert-count共用slab/peers/stats，层间重用；
  routed W2接口为i8，shared W2为FP8；Gate raw output接口与slab一致。
- layer0的load+完整MoE调用manifest通过Rust语义校验，作为probe没有
  serving fill，因此未声称它符合HTTP serving protocol。
- target384和draft128均完成四线程实际Runtime eager/capture/replay，
  counts1/1/1/1和1/5/0/3全部逐字节一致。该阶段fixtures预置gate/quant
  结果；原checkpoint + once + Gate + quant + MegaMoE整链验证另行进行。

### 首个完整层编排

- `attention_forward.py` 串接Q/KV投影、RoPE、paged window写入、FlashMLA、
  inverse RoPE、O-A/O-B；现用普通head布局，fused版本须配套加载期权重置换。
- `Blocks.layer` 内联attention与MoE provider，两个Mega mHC与pending
  residual贯穿同一program。`gen_layer.py`生成layer0完整30call诊断程序，
  Rust语义校验通过；输出HC residual/pre而非token，尚非完整serving模型。
- 真实checkpoint MoE整链EP4验证通过：原始bind→291次once→Gate直接slab→
  quant→MegaMoE→Graph。四rank相对平方误差0.000122506/0.000140288/
  0.000290526/0.000123945，最大相对L2约1.70%，参考原model.Gate+Expert。
- Q-A与KV真实权重的quant+dense链在rows1/5/65全部通过，最大相对平方
  误差8.84e-10。
- 完整layer0已完成真实checkpoint四rank的T1/T5验证，全部30calls及Graph
  通过。对原Block.forward，materialized相对平方误差最大1.36954e-5，
  即相对L2约0.37%；next-pre相对平方误差最大9.28769e-8。
  window-only的extra长度指针已修复为NULL，避免extra_topk=0时误访问extra TMA。
- 长prefill的空rank只重复首pad slot；不再索引超过pad lease页容量。

### 压缩注意力整链接线（待真实整层验证）

- `indexer.py`新增Q-LoRA投影、RoPE、MXFP4量化、weights投影及1/64缩放，
  接paged dense打分与top512；layer20发布top2048候选块，后续index源使用
  paged sparse打分、候选坐标还原及物理页索引。
- `compressed_attention.py`跟踪KV源2/8/14/20、index源与共享候选，收集
  compressor commit供verify接受数确定后执行。attention provider提供
  Q-LoRA之后、FlashMLA之前的准备接口。
- 上述新lowering仅完成Python语法检查，ABI审阅与真实权重调用链验证进行中。
  尚未完成40层、HTTP serving及DSpark接受长度/性能验证。

### 完整 target manifest 生成

- `target.py`串接40层与Engram、共享压缩KV/index结果、三个DSpark taps，
  最后shifted hc_pre和norm。`gen.py`生成load、prefill、decode_batch以及
  target后的draft context发布，均为具体op调用，无占位module。
- capacity128/max_seqs16/context32768实例：load12165calls、prefill1073、
  decode_batch1072；已通过Rust schema v5语义及serving protocol校验。
  这只证明静态编排有效，尚不证明40层GPU执行和HTTP输出正确。
- 无bias的FP32 target argmax已在batch1/17/128通过GPU原sample对照，
  原五步biased Markov argmax回归也通过；kern-serve开发构建成功。
- layer2 Indexer真实调用链scores量级异常来自测试遗漏once scale转换；
  补齐后12calls+Graph通过，score相对RMS0.2872%，top512重合99.8047%–100%。
  最多一个cutoff位置不同，不能声称topk逐位一致。
- `draft.py`与`gen.py`已接三层draft、Markov head、verify6、accept count、
  accepted context发布及compressor commit；round1167calls通过serving protocol。
- DSpark context已完成真实checkpoint prefill5/decode1/verify18验证，
  draft位置修正为anchor起的五行，窗口与官方环形索引转换结果一致。
- HTTP启动已开始排障：容器缺IMEX channel，改用既有EP4验证所用的宿主
  执行环境；随后发现逻辑context容量不能直接作为物理pool的TMA容量，
  已改为runtime已有的state span描述并重新通过serving protocol校验。
  最新服务进程已进入真实权重加载，尚无成功HTTP生成证据。

### 首次完整 HTTP 验证

- 原40层权重、DP4/EP4、prefill+batch decode已通过HTTP：单请求输出64token，
  八路普通chat全部成功（中英解释、算术、代码、短文），内容连贯；算术请求
  正确回答12+7=19并自然EOS，其他请求按64token截断。
- c8本轮响应耗时约0.83–1.15秒；这是debug构建的短请求smoke，包含不同请求
  长度，不能当作最终吞吐性能结论。decode graph覆盖每rank1/2行。
- 首次整链暴露rank串行issue与cuBLAS lazy setup的阻塞：rank0在layer2
  compressor GEMM等待前面的EP MoE，其他rank尚未提交，导致barrier timeout。
  Staged::run改为scoped线程并行issue后上述HTTP验证通过。
- 新增generic `kern server <target>`桥接。chat renderer 由 checkpoint 的
  `model_type` 自动选定（upstream vLLM 6ff479e1 起带 `deepseek_v41`，pegainfer
  a30543d 跟进），本仓库不再 vendor frontend 或 renderer。验证没有改动原始checkpoint。
- DSpark round已静态通过，正在进行原attention对照后切换HTTP六行round；
  接受长度及性能仍未证实。

### DSpark HTTP 初测

- 三层draft attention真实权重回放通过，位置和非因果窗口精确匹配参考；
  attention输出相对RMS4.49%–4.72%，包括额外O-A激活量化误差。
- 六行round的c8 HTTP全部成功；8个请求的输出文本与对应plain decode
  全部相同。mean accepted=2.96 token/round、draft accept39%；响应耗时
  0.62–0.92秒。短样本不能替代正式负载的接受率及性能对照。
- 首次graph capture覆盖每rank1/2sequence（6/12verify rows）；稳态仅在
  全部rank已有graph且未开启eager override时直接顺序cuGraphLaunch，
  其余情况并行issue，避免lazy库初始化等待尚未提交的peer。
- 下一步：跨chunk长prefill、更大batch、官方可比接受长度基线及V4.1
  多轮/系统/思考编码精确适配。

### 扩展 HTTP 验证与当前精度问题

- 上述后续项已有新证据：617 token 输入经五个128-token chunk完成prefill，
  DSpark正确提取指定事实并自然EOS；该单请求平均接受3.75 token/round。
- 独立 `deepseek_v41` renderer 已通过28个官方编码golden；HTTP系统指令、
  多轮记忆与thinking预算请求均通过。工具历史输入编码有参考测试，输出
  `tool_calls` parser尚未适配，不宣称完整工具调用支持。
- c32 plain与DSpark均32/32请求成功，但只有20/32输出文本相同。
  DSpark平均接受2.98 token/round。此前c8一致不能推广为大batch精度通过。
- 同历史teacher-forced检查比较四rank各八条sequence：192行verify logits中，
  七条sequence的六个位置全部逐值相同；仅prefix52的sequence六行不同，
  四rank一致复现。position54的top1由396变965，plain margin为0.16088，
  最大logit差2.95277。接受三个token后下一步仍保留该sequence差异。
  这一结果尚不能归因于常规浮点误差，正在逐层定位。
- 官方DSpark接受长度对照使用固定AR历史与tap，并补采kern在同历史上的
  proposal；官方推理脚本没有可直接比较的接受率或吞吐基线。
- 2026-09-10 tray05 A/B（`~/bench_results/2026-09-10-dsv41-fused-ab`）：A =
  paged attention + BF16 O-A（8 次 cuBLAS），B = FlashMLA fused
  RoPE-attention-RoPE-cast + FP8 O-A 一次 launch，reuse 层每步 30 → 21 次
  launch。16 条 prompt 贪心 256 token conc1：A、B 各自 rows=1 与 rows=6
  16/16 逐字同；A 对 B 0/16 同（3 条首字分叉），全部连贯、措辞级差异；
  rows=6 接受 2.71 → 2.76 token/步（34% → 35%）；TPOT rows=1 12.0 → 11.1
  ms，rows=6 5.38 → 4.93 ms。官方 `inference/model.py` 的 `wo_a` 是 BF16
  einsum，所以 A 对齐参考实现、B 对齐报告的生产核；分叉点的 top-1/top-2
  margin 没有量（kern-serve 无 logprobs，`kern test` 只单卡），决定切到 B，
  margin 留给多 rank `kern test`。
- 切到 B 之后的两步胶水收缩（同一 tray05 harness，rows=1 16/16 与 B 逐字
  同）：wq_a 与 wkv 共用一次 MXFP8 量化（4f607fc，21 → 20）；mHC 与 Mega-Gate
  的 barrier 从每次调用清零的 op 私有 scratch 改为 `load` 清零一次的 carry
  buffer，作为 op 的最后一个 `inout` 参数传入（20 → 17）。上游就是每个
  stream 分配一次、kernel 自行复位，这里只是把同一约定写进 manifest。
- 原checkpoint直接绑定与生成器`--bundle`产物已验证manifest一致；
  不需要离线导出权重。连续host权重段改为一次拷贝后，实际加载耗时
  从约154秒降至52–64秒。以上HTTP耗时仍为debug smoke，release性能验收待完成。
- 首个同历史DSpark对照已完成：autumn prompt生成72个AR token，插入draft
  采样前后canonical token IDs完全相同。67个可评估位置上，官方与kern
  均走23轮，平均接受2.913043 token；完整五proposal有45/67位置一致。
  两者接受长度直方图不同，均值一致不代表逐位置精度相同。
  此对照使用kern target taps，不能替代官方40层target精度验证；后者正在补齐。
- c32差异已完成因果定位：分叉前55份cache/state一致，layer0–19全部边界
  一致。layer20 compressor的BF16 wkv投影在M8/M48间仅元素296差1 ULP
  （0.046142578125与0.0458984375）；归一化后当前token的一个FP4码由5变4。
  索引分数、选中token集合与量化index key均一致。仅反事实替换该BF16元素，
  四rank全部224行verify及部分commit后logits恢复逐值一致。
  固定该投影M128后，全部224行也逐值一致。生产lowering已将ratio1 wkv的M
  固定为manifest row capacity，其他算子仍只处理live rows；生成物恰好只改
  prefill/decode/verify三处M参数，通过schema及serving protocol校验。
  这统一了数值路径，也可能改变旧plain的末位舍入结果；HTTP更大batch与
  release性能回归仍需验证，不能用224行诊断替代完整验收。

### 10k 输入的 release HTTP 性能

- 4×GB300，attention DP4 / MoE EP4，单请求，BF16 O-A，prefill chunk128。
  三个不同前缀请求均精确10,000输入token、256输出token，prefix hit均0；
  先独立短请求预热，再取三个测量的中位数。
- Plain：10k/HTTP TTFT吞吐5309.9 token/s，TTFT1.883秒，
  后续平均TPOT11.687 ms/token，即85.57 token/s。
- DSpark：吞吐5306.5 token/s，TTFT1.884秒，平均TPOT5.071 ms/token，
  即197.20 token/s；decode速度为plain的2.30倍。
- TPOT口径为首个到最后一个非空文本事件的时间除以输出token数减一，
  属于HTTP平均交付速度；投机模式可能一次交付多个token。
  三组输出有一组文本完全相同，另外两组不完全相同；此性能结果不等于
  声称bitwise/逐token一致，也不替代参考可靠性及质量验收。
