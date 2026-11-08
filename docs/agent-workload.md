# Agent workload：AgentX（Claude Code）trace 分析与对 NVL72 部署的含义

状态：2026-09-01；§2 成本口径 2026-09-03 按时间重算（见表下说明）。数据 = InferenceX（SemiAnalysis）AgentX 回放集，真实 Claude Code
会话的匿名 trace（prompt/代码/工具载荷剥离，保留每请求 token 数、64-token 块链哈希、
子 agent 分支、时间戳）。本地副本与分析脚本：
本地存档 `bench-results/2026-09-01-agentx-traces/`（不在仓库）（`analyze.py`、`analysis-full.txt`、
`analysis-256k.txt`）。来源：
[inferencex.semianalysis.com/datasets](https://inferencex.semianalysis.com/datasets)、
HF `semianalysisai/cc-traces-weka-062126`（full，≤1M ctx）/ `-256k`。
独立对照：UW TraceLab（arXiv 2606.30560，4.3k 会话）结论一致。

## 1. 数字（full 集：393 会话、98,827 请求、21.6G 输入 / 106.5M 输出 token）

| 量 | p50 | p90 | p99 |
|---|---:|---:|---:|
| ISL（每请求输入） | 142k | 550k | 863k |
| OSL | 444 | 2.7k | 9.3k |
| **extend = 输入 − 理想前缀命中** | **1.6k** | **5.9k** | 41k |
| 冷启动 ISL（会话首请求） | 448 | 60k | 103k |
| **decode 发生时的 ctx（按输出 token 加权）** | **219k** | **630k** | 889k |
| agent 流内空闲间隔（等工具/人） | 2.9 s | 49 s | 1533 s |
| 会话：请求数 / 墙钟 | 86 / 2.8 h | 618 / 47 h | |

- 理想前缀命中 **98.3%**（会话内连续块链；跨 agent 任意块 98.3%，同 agent 97.9%）。
- extend 分桶（请求占比 / extend token 占比）：≤256 10.8%/0.4%；≤1k 26.3%/4.3%；
  ≤4k 48.1%/28.9%；≤16k 11.9%/23.7%；≤64k 2.4%/19.5%；**>64k 0.5%/23.3%**；
  >256k 0.07%（72 个，最大 949k）。
- **>64k 的 whale**（524 个）：ISL p50 130k，**前缀命中 p50 = 0**，只有 26 个是会话
  首请求、29 个像 compaction（ISL 缩小）——主体是**整条链断掉的全量重 prefill**
  （提示词前部动态内容变化之类）。
- extend token : 输出 token = **3.4 : 1**。
- 输出 token 按 ctx 分：≤64k 14%、≤256k 43%、≤512k 27%、**>512k 16.5%**——
  **43% 的输出在 >256k 下生成**；256k 变体隐藏了一半负载，必须按 1M 设计。
- 子 agent = 43% 的请求，与会话共享 82% 的块；峰值并发 p90 4、max 10。
- 会话忙碌占比（Σapi_time / 墙钟）中位 **13.5%**。
- 参考（Anthropic 生产）：TTFT p50 2.6 s，api_time p50 6.7 s → TPOT ≈ 15 ms，SLO 松。

## 2. 成本模型（K3：24 层 MLA、96 头、latent 576 B/token/层；69 层 KDA）

每 (query, ctx-token) 对：prefill（非吸收，q/k 192、v 128）2×(192+128)×96 ≈ 61 kFLOP/层
→ **1.47 MFLOP**；decode（吸收，576/512）2×(576+512)×96 ≈ 209 kFLOP/层 → **5.0 MFLOP**。
KV 读 13.8 KB/token。dense+MoE 按 ~64 GFLOP/token（K2 量级，K3 待定）。

全集算账：

| | 量 |
|---|---:|
| prefill attention（extend×prefix + 自身三角） | 93 EFLOP |
| decode attention（每输出 token × ctx） | 152 EFLOP |
| dense + MoE（两侧） | 30 EFLOP |
| KV 读：decode 侧 / prefill 侧 | 420 PB / 0.3 PB（1400 : 1） |
| 合计 275 EFLOP ÷ (72 GPU × 1.5 PF) | 2548 s → **≤ 42k 输出 tok/s 整机上限**，同时对应 **~143k extend（prefill）tok/s**（3.4:1）；按峰值算，实际打 6 折 |

**这张表是 FLOP 口径，decode 侧两个分母都已被实测推翻（2026-09-03 重算）**：decode
attention 是带宽绑定（[`dcp-bench.md`](https://github.com/pegainfer-project/kern-k3-pp-prefill/blob/main/docs/dcp-bench.md) B1：0.27 µs/ktok/层，219k ctx 每 session 每步
1.4 ms），dense + MoE 在 decode 时是**权重带宽**绑定、与 token 数无关——EP32 下每 rank
每步读 dense 105 GiB + 28 个 expert 42 GiB = 147 GiB ≈ 24 ms，再加 EP dispatch/combine
5–7 ms，是每步 ~30 ms 的固定地板（E2b 实测 EP4 pruned B=1 步时 29.3 ms：dense 105 +
expert 84 GiB，同一个数）。按 FLOP 算它几乎为零，按时间算它是一半。按时间重算，
EP32、219k ctx、fp8 latent，每 rank 每步：

| B（每 rank 行数） | attention 1.4 ms × B | 权重读 + EP 同步 | attention 占比 | 步时 |
|---:|---:|---:|---:|---:|
| 4 | 6 ms | 30 ms | 17% | 36 ms |
| 16 | 22 ms | 30 ms | 43% | 52 ms |
| 64 | 90 ms | 30 ms | 75% | 120 ms |

全集折算（B=16）：decode attention ≈ 55 GPU·h、权重读 ≈ 55 GPU·h、extend attention
≈ 10（93 EFLOP 按 3 PF fp8）、KDA ≈ 4——attention 约一半，不是九成；要到 89% 得
B ≈ 100，那时步时 170 ms，TPOT 早破了。（dense 105 GiB 是导出文件的实际大小，
`kern-k3/full/dense/`；expert 每层每 rank 0.46 GiB@EP32 同样量自文件。）

口径核对：vLLM TP8×EP8 冷 128k 4.48 s = 29k prefill tok/s / 8 卡，×9 = 260k tok/s 整机。
本模型算冷 128k 为 20 PFLOP / 8 卡 = 1.7 s（vLLM 效率 ~40%），不矛盾；差别在
**每个 extend token 对 220k 前缀的 attention（0.32 TFLOP）是冷 128k 平均每 token
（0.094 + 0.064 MoE）的 2 倍**，且每个输出 token 背着 3.4 个 extend token 和
1.1 TFLOP 的 decode attention——所以 143k extend tok/s ≈ 冷 128k 口径的 ~300k tok/s。
| 一次冷 500k prefill | 184 PFLOP ≈ **122 GPU·s**（+MoE 21 GPU·s） |

三个推论：

1. ~~attention 是全部成本的 89%~~ **attention 的份额是 B 的函数，TPOT 允许的 B 下约一半**
   （上表）。另一半是每步固定的权重读，这一半是系统能动的：tray 内把 dense（含 KDA）
   按 TP4 分片，每 rank 每步从 147 GiB 降到 26 + 42 = 68 GiB，地板 30 → ~18 ms，B=16
   时步从 52 到 40 ms，或者固定 52 ms 的步下 B 从 16 涨到 24（每 rank 并发 +50%）。
   attention 仍留 DP：TP2 切头后一对卡各读整条 latent，attention 项 22 → 44 ms，换回
   dense 省的 8.5 ms，净亏（吸收式 MLA 每个头组都要读全部 latent）。模型侧的稀疏 /
   压缩 attention 仍然值钱，但只砍一半，不是九成。
2. ~~MLA 吸收式 decode 在 GB300 上是算力绑定~~ **已证伪**（[`dcp-bench.md`](https://github.com/pegainfer-project/kern-k3-pp-prefill/blob/main/docs/dcp-bench.md) B1）：
   b=1 时 786 TF/s、4 TB/s，是带宽/占用绑定，每层 0.27 µs/ktok。纸面 362 FLOP/B
   没兑现，因为一条 session 只有 96 行 query，GEMM 打不满。所以 decode attention
   按字节算：219k ctx 每 token 1.4 ms；共享 KV 的行批起来便宜 4×。"decode 吃带宽、
   prefill 吃算力"在这里成立，但两侧都要对 220k 前缀做 attention，互补换不来分离
   （见推论 3）。
3. **mix 的真正理由是前缀局部性**：98% 命中、extend 中位 1.6k，每个 extend 都要对
   220k 前缀做 attention，前缀住在 decode 所在的卡上；分离 P 意味着每请求搬 3 GB
   或跨 cell ship q。在前缀所在处算 extend 是唯一合理的选择。

（同日早些时候口头估的"2k chunk 对 220k 前缀 ≈ 17 ms"漏了 96 个头，真值 ≈ 440 ms
单卡；以下全部按修正后的数。）

## 3. 对部署的含义

- **不分 P/D。** 全部 GPU 是 mix cell；extend 在前缀所在 rank（或其条带组）作为
  decode superstep 的 filler 计算；whale 是"多走几步"，不是"走大步"。
- **chunk 是派生量，不是常量。** 每步 filler 预算 `b`（如 5 ms）、条带宽 `w`、前缀
  `P`：`chunk ≤ b × w × 1.5 PF / (P × 1.47 MFLOP)`。P=220k、w=4：~90 token/步；
  P=500k、w=36：~370 token/步。KDA 段效率下界（≥2k）与之冲突时以量子为准——
  线性层只占 ~4%，效率掉一半无所谓。
- **条带宽度按 ctx 定**（4 / 8 / 16 / 全 cell），prefix attention 是 partial+merge 同一个
  op，decode 与 extend 共用。43% 的输出在 >256k、单 session latent 12 GB@900k，条带化是
  延迟与 chunk 可行性的前提，不是可选项。
- **whale 的 TTFT 与背压**：一次 130k 重 prefill 对 220k 前缀 ≈ 40 PFLOP；冷 500k ≈
  184 PFLOP。在 36 卡 cell 里按每步预算分摊：TTFT ≈ FLOPs / (36 × b × 1.5 PF) × 步长。
  b=5 ms（TPOT 10→15 ms）：130k whale ≈ 2.2 s，500k ≈ 10 s；b 就是背压旋钮，多头 whale
  共享预算、排队、aging。这与专用 CP16 gang 的 TTFT（pegainfer 实测 256k 8.85 s）同量级，
  但不需要任何 rank 退出 decode，也不需要 gang/rendezvous/arming 那一层。
- **HBM 驻留按 13.5% 忙碌规划**：session 睡到本 tray DRAM（醒 30–60 ms ≪ 间隔 p50
  2.9 s），>60 s 的 9% 可再下一层；DRAM 挂 HBM 的 ~7 倍 session。
- **分支 CoW 是一等公民**：43% 请求来自子 agent，共享 82% 的块；从父 checkpoint 分叉 =
  latent 页引用计数 + KDA 状态快照（0.2 ms）。lease 池要加引用计数。
- **whale 的第二个来源可以在应用层消灭**：命中 0 的 130k 重 prefill 多半是提示词前部
  动态内容打断了块链，属于 agent 框架的缓存卫生问题，比任何 CP 都便宜。

## 4. 待验证

- K3 的 MLA 层是否有稀疏 attention（决定 attention 那一半能不能砍）。
- ~~decode attention 的实测 FLOP/B~~ 已测（[`dcp-bench.md`](https://github.com/pegainfer-project/kern-k3-pp-prefill/blob/main/docs/dcp-bench.md) B1，带宽绑定）；待测的是
  dense TP4 的实际地板（allreduce 93 层 × 2 次、KDA 分片后的 state 布局）。
- 真实 arrival：trace 是每会话相对时间，整机并发要靠回放器给到达率；AgentX 的回放
  脚本（aiperf `--public-dataset semianalysis_cc_traces_weka_with_subagents`）可直接用。
