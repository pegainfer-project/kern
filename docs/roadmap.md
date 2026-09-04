# Roadmap（2026-09-02 定稿；背景与实测见 multi-gpu.md / agent-workload.md / dcp-bench.md / moe-comm-survey.md）

**第一个多卡目标：K3 pruned（224 expert）@ EP4 单 tray 的 decode superstep**，与满血
K3 @ EP16 同构（56 expert/rank）。MoE 通信用 DeepGEMM MegaMoE（一层一个 launch）；
attention（MLA / KDA）、dense、norm 等 kernel 从 vLLM 的 K3 运行里挖（CUPTI capture →
manifest，与 qwen3 同一条流水）。两条线并行，每级带门禁。

## EP 线

| 级 | 内容 | 门禁 |
|---|---|---|
| E0 ✅ | runtime 原语：`export`（VMM + fabric handle）、`peer`（u64 数组）、`topology`、`{"rank"}`；`export_handles` / `import_peers` API；verifier 三条规则（peer 必须 of export、`.MULTICAST` SASS 扫描、extern 不接 peer） | 一进程 4 个 runtime，跨卡 barrier 作为 manifest op 跑通，≈3.8 µs —— **2026-09-02 tray03 实测 3.75 µs**（`ep0-k0-export-state`） |
| E1 ✅ | K3 pruned 的一个 MoE 层作为 program：quant → MegaMoE cubin → 输出，EP4 单 tray；`SymBuffer` 的偏移表由 kernel 在设备上从 peer 数组读；manifest 新增 `tensormap` launch 参与 `cluster` | EP4 每个 rank 与 EP1 逐位一致 —— **2026-09-02 tray04 实测通过**：EP4 227 µs/层（64 token/rank），EP1 733 µs/层（256 token），EP1 对 host 参考相对 RMS 1.7e-3（`k3_moe_ep`） |
| E2 ✅ | K3 pruned 完整 decode superstep @EP4：MLA + KDA + MoE 全 93 层一条 program（3792 launch），图捕获，每 rank 一条序列；核取自 pegainfer 认证 K3 核集（TileLang AOT + MegaMoE + 手写 MLA），稠密投影走新 extern `cublas_bf16_tn_f32` + 认证 `k3_land` | 与 pegainfer golden fixture 逐 token 一致（4 层 EP1/EP4 均 39/40 exact + 1 noise-floor，与 pegainfer 自身相同 —— **2026-09-02 tray04 通过**）。93 层 EP4 teacher-forced 三方对照（`tools/k3_oracle_dump.py`）：对 pegainfer EP4 prose 89/96、random 38/40 exact，对 vLLM prose 87/96 + 3 excused；所有不一致步都落在 vLLM top-1/top-2 margin ≤ 0.81 nats 的近平局上（kern 的答案均在 vLLM top-3，步 34/50 是 kern 与 vLLM 一致而 pegainfer 不一致），而 pegainfer 与 vLLM 自身也差 7/96 —— 三家核数值噪声，判为一致（用户 2026-09-02 定案）。步时 **37.2 ms/step**（4 rank × 1 seq，graph replay）vs pegainfer EP4 38.6 ms、vLLM DP4+EP4+MegaMoE 27.1 ms（vLLM TP4 12.5 ms 是另一种切法，不可比）；稠密权重每 rank 全量读的地板约 17 ms，步时对齐推后（用户：先不管性能）。lease 时 runtime 已把该序列的 KDA 槽清零。每 rank B=1 的限制由 E2b 解除 |
| E3 ✅ | 跨 tray EP8：一 tray 一进程，fabric handle 经 TCP rendezvous 交换（`k3_golden --world/--rank-base/--rendezvous`），MegaMoE 核加 R=8/16 世界；spin 超时 / 故障模型与 W=36 实测未做（用户：E3 只做简单验证） | **2026-09-02 tray07+tray08 通过**：4 层 EP8 跨 tray 8 个 rank 全部 39/40 exact + 1 excused，与单 tray EP4 同；93 层 EP8 跨 tray 对 pegainfer oracle 38/40，错的两步与单 tray EP4 完全相同（逐 token 一致）；步时 **36.8 ms/step** vs 单 tray EP4 37.2 ms —— 跨 tray dispatch+combine 税在 bs=1 下测不出（每 rank 少读一半 expert 抵掉了）|
| E2b ✅ | 每 rank B>1：K3 decode 核集用 CUDA C++ 重写（7 族，`tools/kernels-src/k3_*.cu`，契约 k3-kernel-abi.md，harness `tools/k3-harness/`；由 6 个并行 agent 各写一族，SASS 0 spill + ncu 报告），add / mul_sigmoid / norm / cuBLAS partial landing 全部融进邻居核，MLA decode 改 cluster-8 split-KV（32k ctx 258 µs vs 旧核 10 ms）；manifest `examples/k3-*.json`（TileLang 核集与其 manifest 已删），93 层 1855 launch（原 3792） | **2026-09-02 tray07/08 通过**：4 层 EP1 B=1 37/40 + 3 excused；4 层 EP1/EP4 B=8 mixed distinct 38/40 + 1 excused + 1 近平局（1.0 ULP，参考 top-5 外），无行间串扰；93 层 EP4 对 pegainfer oracle B=1/8/32/64 全部 38/40（错的两步与 TileLang 核集相同）；跨 tray EP8 B=8 同 38/40。步时 **B=1 29.3 ms**（TileLang 37.2，vLLM DP4+EP 27.1）、B=8 35.3、B=32 48.3、B=64 61.4 ms（EP8 跨 tray B=8 34.3）。长 prompt 门禁（forge bring-up log 前 46 KB = 12 897 token 逐 token 喂，pegainfer EP4 ctx 16k 做 teacher-forced 参考，`k3_oracle_dump.py --check-last 64` 只比最后 64 个 prompt 位置 + 64 步续写）：B=1 111/128、B=8 mixed 110/128，17 个不一致逐个 detokenize 全是同义替换 / 近平局（`;`↔`,`、`perf`↔`speed`、`kernels`↔`overhead`、数字），无乱码；`k3_golden --free 96` 让 kern 在 12 961 token 后自回归续写 96 token，文本连贯地接着日志写（"over the day (mbarrier arrivals, TMEM warpgroup, SMEM overlap); 1 self-merged flagged…"）。长 ctx 步时 B=1 31.5 ms、B=8 42.3 ms |
| E4 | GPU 自转：图尾 tail-launch、图头等 step flag、advance 小核；follower 无 host | 成员不变的步 host 零参与 |
| E5 | tray 内 TP4：tray 一个 batch，dense / shared expert / dense FFN 权重按列行切，KDA 投影与 state 按头切，MLA attention 按行归各卡（q/kv/o 投影第一版复制），行的 all-gather / reduce-scatter 是读 peer 数组的 kernel（`multi-gpu.md`"最终形态"）；导出按 tp 组切权重，manifest 两个组 `tp: 4, ep: N`；checkpoint / park / wake / fork 升为 tray 级 | 93 层 EP4×TP4 对 pegainfer oracle 逐 token 一致（E2b 同口径）；每 rank 每步权重读 147 → 68 GiB，B=1 步时 29.3 → ≤ 20 ms，B=16 52 → ≤ 42 ms（或同步时 B 16 → 24）；单 tray 与跨 tray EP8×TP4 同一致。**进度 2026-09-03**：第一块过——collective 核 + `--tp` 的 tray 批数据流（只加 gather、不切权重）4 层 EP4×TP4 与 EP4 逐 token 同（37/40）、混合行独立、四卡复制批一致、fork 正确；collective 实测见 multi-gpu.md"tray 内 collective"。第二块过——KDA 按头切（导出脚本 `tools/shard_k3_tp.py`、`HEADS=24` 核变体、每卡持 tray 批全部行的 state 分片）同一门禁全过，4 层 B=1 1.723 → 1.420 ms、B=8 2.176 → 1.633 ms。第三块过——shared expert / dense FFN 切列 + 每层一次 allreduce（lat_up 复制），4 层 B=1 1.188 ms；**93 层 EP4×TP4 对 12.9k oracle：B=1 108/128（EP4 111/128）、B=8 mixed 113/128（EP4 110/128）、B=16 107/128**，不一致全是同义替换；短 ctx 步时 **B=1 20.8 ms（EP4 29.3）、B=8 25.5（35.3）、B=16 31.9（~40）**，12.9k ctx 23.1 / 35.6 / 47.9（EP4 31.5 / 42.3 / —）。B=16 ≤ 42 过，B=1 差 0.8 ms（MLA 投影仍复制 ≈ 1.35 ms）。allreduce 换核（同日）——TensorRT-LLM `allreduce_fusion` 的协议（`tools/kernels-src/peer_allreduce.cu`：poison 当 flag 的 Lamport one-shot ≤192 行、两 barrier 的 two-shot 以上，按 own-rows-first 与 kern peer ABI 改造，`tp_init` 预填 -0.0）：单测 tray 64 行 24.9 → 15.0 µs、128 行 46 → 28；93 层像对像 A/B 步时 B=8 24.07 → 23.30、B=16 29.21 → **27.65 ms**，4 层三项门禁与换核前逐 token 相同；见 multi-gpu.md"allreduce 换成 TensorRT-LLM 的协议"。第四块写完、t=1 smoke 过、多卡门禁待跑（2026-09-03）——kern-serve 加 `tray.rs`：一个线程驱 n 个 `Runtime` 锁步，`Row` / `Snapshot` / `Sleeping` / `Rising` 是 tp 组每个成员一份的元组（owner 持页 + slot，peer 只持 slot：runtime 新增 `lease_slot` 与 slot-only 的 checkpoint / fork / restore / wake），lease / fork / checkpoint / park / wake 都是"全成或全不成"——靠类型构造（`Row` 只在每个 rank 都答应之后才存在，提前返回 drop 已拿到的）而不是回滚；park 用 runtime 新的两步 `room` + `park` 先在每张卡上找齐地方；`Prefix<R, P>` 泛型化按 tray 键；`Staged` 借住 tray 直到输出读完，stage / run / read 不可分；每步读 `tp_err`；owner 取页最少的 rank、终身不变；K3 没有 prefill program，prompt 逐 token 走 decode 步（正确、慢）。t=1 qwen3-4b 的 conc1 / 并发 / resident 与 host wake 命中 warm == cold 过，多卡门禁未跑：见 serve.md"tray 级"一节的待测清单。MLA decode 核在多行长 ctx 下每行每步 ~1 ms/12.9k 见下一行 M1 |

## KV 线

| 级 | 内容 | 门禁 |
|---|---|---|
| K0 ✅ | state 一律走 VMM（可导出）；per-seq 定长 state（KDA 状态）已有 `bytes_per_seq` | 现有 qwen3 / dspark 门禁不变（2026-09-02 过）；K3 的 KDA state 能装载 |
| K1 ✅ | 前缀缓存 = checkpoint 表：`Checkpoint` = (精确长度, 页链上的一个节点, 可选 state slot)，页链按引用计数还页（`Arc<Node>`，一页一节点，一条序列的所有 checkpoint 共一条链）；`Prefix` 表按页哈希链索引，序列自带滚动 `Chain`。纯 KV 模型每整页留一个、零拷贝；带循环状态的模型只在请求结束 `retire`（slot 易主，不拷），中途分叉不命中 | AgentX trace 回放（`crates/kern-run/examples/agentx_replay.rs`，393 session / 98 827 请求，2026-09-02）：纯 KV 页 64 命中 98.6%（缓存 7M token 时 98.1%），extend p50 640 / p90 5056 / p99 40704；带状态页 64 + 1024 slot 96.6%，页 784 + 130 slot（qwen3.8-27b 实机形状）91.0%。GPU 门禁在 `serve.md`：qwen3-4b / qwen3.8-27b 命中后 warm 与 cold 的 64 个 greedy token 一致 |
| K1b ✅ | 物理内存一个池：分页 state 与 per-seq state 各占一段虚拟地址，物理块（`cuMemCreate`，2 MiB 粒度的整数倍、≤ 最小对象的一半、封顶 64 MiB）谁用谁 map，块留在上次用它的地方，一类用光才从另一类的空闲对象上拆（`Remap`，后台线程等 stream 事件后 unmap → map → access，主线程收下时清零），`seqs.max` 只管每步 batch 行数、slot 从 `seqs.max + 2` 起按需长；拒绝分 `Remapping` / `Busy` / `ExceedsPool`。不做 vLLM 式"每个块边界都存一份 state"（它靠 TP 把 state 切小才付得起，且块边界与公域前缀的边界对不上；K3 EP-only 一份 605 MB）。实测（tray03 GB300，2026-09-02，`tools/vmm_bench.py`）：`cuMemMap` 1.3 µs/块、`cuMemSetAccess` ~30 µs/块、`cuMemUnmap` ~20 µs/块，都按块数计不按字节，所以块要尽量大：2 MiB 块时拼一个 qwen3.8 slot 2.3 ms / 一份 K3 state 8.9 ms，32 MiB 块时 0.2 / 0.7 ms；首次访问多 ~20 µs；连续 map/unmap 时同卡的带宽循环慢 3% | 回放（页 784 + 147 MiB slot，qwen3.8-27b 实机形状，2026-09-02）：同样的活跃负载（并发 32、页不设限 → 1024 GiB）命中率 91.0% → 96.9%，slot 峰值 2641、73 578 次 remap；单卡真实预算 250 GiB 并发 32 为 93.1%（页成了瓶颈，17 个请求等页），并发 16 为 95.8%；纯 KV 的 qwen3-4b 形状（147 KB/token、页 16）在 250 GiB = 1.9M token 下并发 32 只有 84.2%（2180 个请求等页）——单卡装不下 32 个 20 万 token 的会话，这是容量不是缓存的事。GPU：qwen3-4b `kern test` 位一致；qwen3-4b / qwen3.8-27b 的 serve 门禁输出与 K1 时逐字相同；qwen3.8 上 200 个不同请求结束后 slot 从 130 长出去、之后的输出与 cold 一致（`serve.md`） |
| K1c | 显式断点：请求可标断点（system prompt 末尾），prefill 把 chunk 末尾裁到断点后 `checkpoint`；带状态模型 `Busy` 只淘汰带 state 的 entry；"每 N 页自动存一份"留作旋钮，等有需要它的流量再加 | 共享 system prompt 的两个不同 session，第二个命中到断点 |
| K2 ✅ | fork = 引用计数 +1 + 状态快照拷贝：`Runtime::fork(&mut parent, len, tokens)` 从活着的序列分叉，整页共享、半页拷给孩子、per-seq state 拷进新 slot（带状态模型只能在父的当前位置分叉）；与 `lease_from` 同一套 `Pool` 决策，父不用先 checkpoint；`lease_from` 也接受纯 KV checkpoint 的更浅整页 | **2026-09-03 tray07 通过**（`k3_golden --fork 12 --seqs 4 --mixed --distinct`，93 层 EP4，4 卡）：第 12 步从行 0 分出两个孩子——喂同样 token 的"孪生"28 步逐 token 与行 0 一致（4 个 rank 都是），喂随机 token 的"走失"28 步与它从零跑的参照一致，行 0 对 oracle 仍 38/40 且错的两步（15、16）与 E2b/E3 相同，即孩子写自己的页、父不受影响；4 层 EP1 同样通过。参照 batch 与被测 batch 在同一步同样分叉，行与行的对照始终在同一 bucket（bucket 一变，随机行的近平局就翻，最初的"分叉后各行分歧"全是这个） |
| K3 ✅ | session 睡/醒到本 tray DRAM：`reserve_host` 一次 pin 一块（绑到卡所在 NUMA 节点），`park(checkpoint)` 把页和 slot 拷到 host 链上（同 session 再睡只拷新增的页），`wake` → `Waking` → `awake` 落地了才交出 `Lease`；拷贝走 transfer stream，compute stream 从不等它；`Prefix` 分 resident / parked 两层，纯 KV 链条目按页增长、部分命中只醒需要的页；kern-serve `--host-gib` | **2026-09-03 tray08 通过**：qwen3.8-27b 形状 98k token = 6.12 GiB，park 34 ms / wake 31 ms（180 / 197 GiB/s，逐字节验证；pinned 块落在另一颗 Grace 上时 53 / 52 ms，所以 runtime 自己绑 NUMA）；qwen3-4b 2500 页 5.49 GiB 31 / 28 ms。kern-serve：8 路 decode 下每秒 ~10 个 3.5 GiB 的 wake，token 间隔 p50 3.03 → 3.08 ms（+1.7%），p90/p99 不变；命中后 warm / cold 64 token 逐字一致。AgentX 回放（qwen3.8 形状，单卡 250 GiB + host 512 GiB，并发 32）命中 93.1% → 95.6%，p99 extend 430k → 208k。命中请求本身的 p90 85 ms 是 prefill 尾块在长上下文上的 40 ms（K5 的事） |
| M1 ✅ | MLA decode attention 换成 FlashInfer 的 CuTe-DSL Blackwell 核（NVIDIA 写，BSD-3），预编译 cubin 入库 `tools/kernels-bin/`；runtime 只加两样通用能力——`bytes<n>` + `pack`（struct 参数从接口铺平）和 tensormap 最外层维 0（铺满 state）——核的 28 参数 ABI 从 DSL 的 launch 里抓出来反解，写在 manifest 里（`k3-kernel-abi.md` K5）；配套自写 absorb / split 规划 / v_up+gate 三个小核，split 数每步在 GPU 上按行定 | 单核：B=1 13k 42 → 16.5 µs/层，200k 58 µs（split 72）、B=16 13k 55 µs；独立 launcher 与 DSL 输出逐位一致。**2026-09-03 tray07**：4 层 EP4×TP4 B=1 35/40 + 4 excused + 1 近平局（3 ulp），mixed4 37/40 + 2 + 1 近平局；**93 层 EP4×TP4 对 12.9k oracle：B=1 114/128（旧核 108）、B=8 mixed 111/128（113）、B=16 112/128（107）**，不一致仍是措辞级近平局；12.9k ctx 步时 **21.1 / 25.7 / 30.2 ms（旧核 23.1 / 35.6 / 47.9）**，短 ctx B=1 20.9、B=16 29.2（旧核 20.8 / 31.9；手头的 93 层 40 步 fixture 与 tray07 权重不是一个 checkpoint，新旧核、EP4/TP4 四种跑法逐 token 相同，短 ctx 只记步时）。每行每步 ~1 ms/12.9k 的斜率没了：B=16 长短 ctx 差 1 ms。配套核 absorb / vup_gate 带宽版 6.5 / 11 µs（B=1），4 层步时 1.195 ms（旧核 1.188） |
| K4 | DCP：ship q / 回 (O, LSE) 的 partial+merge op，w 按 span 定，flag-in-payload | 真 FlashMLA 复现 B2/B3：decode 税 ≤ 8%，extend W=4 ≥ 2× |
| K5 | **2026-09-03 定稿**（设计与 DAG 见下节"K5 规划"）：不做单独的 prefill program，decode 步的行从"一行一 token"变成"一行一 span"，extend 是 n>1 的 span 行与 decode 行同一步、同一张图；每步 span 长 c 由 caller 按稠密预算与 attention 预算派生。KDA 用 MoonshotAI FlashKDA（pegainfer / vLLM 同源，state 布局与 kern 的 rec 相同），MLA v1 用现有 DSL 核按行展开（c 行共用 block_table），物化路径按实验决定 | 4 层：fixture prompt 作一个 span 与逐 token 同口径（37/40）；93 层：12.9k oracle 按 c≈200 分 65 个 span，128 token 与逐 token 同；kern-serve：conc1 与 `kern run` 同，conc8 加冷 12.9k prompt ITL ≤ +25%，TTFT 记进 serve.md |

顺序：E0 + K0 一起先做（同一个分配器改动，K3 的 KDA state 也在关键路径上）；然后
E1–E2 与 K1–K2 并行；E3、K3；然后 **E5**（每步省 12 ms，先于 E4）与 K5 并行；
K4 改成按新 token 分段的 extend（待定稿）；E4 最后，其门禁按它实际买到的东西重写
（agent 负载下成员不变的步很少）。E4 依赖下面的 step 边界 GPU 化。

## 协议线

| 级 | 内容 | 门禁 |
|---|---|---|
| V4 ✅ | serving 协议进 manifest（schema v4，设计 v4-design.md）：buffer 的 `fill`、program 的 `batch` / `once`，`spec` 块删除；`Verified` newtype + `Protocol::check` 投影，kern-run / kern-serve 不读 JSON、不认名字（CI grep）；投机轮统一成 `round` program（dspark 也补了：splice / 计数 / prefill 收编 head 与 precompute 全在设备上），`--spec` → `--rows`；`kern verify` 打印协议事实 | **2026-09-03 tray03 通过**：`kern test` 位一致；qwen3-4b / qwen3.8-27b / dspark / dflash2 四份 conc1 与 `kern run` 逐字同；dflash2 与 v3 `--spec` 逐字节同；dspark 7 行 round 在一个 0.125 的 bf16 平局上与 v3 的 8 行 verify 分叉（核噪声）；conc32 接受率 34% / 24% 不塌，5930 / 1709 tok/s。记录 v4-design.md §9、serve.md |

## K5 规划（2026-09-03）

**依据**（`~/bench_results/2026-09-03-gb300-power-compute-vs-bw`，含补充 2 的 TP4 形状实测）：
稠密 GEMM 的 ridge ≈ 峰值算力 / HBM 带宽 ≈ 256 行 bf16，与 TP 切法无关；64→256 行的边际
0.046 µs/行/GEMM（TP4 切片），256 之上 0.082；2k 行并进 decode 步只省 10% 能量、步时 3 倍。
TP4 下 tray 是一个 batch，稠密相位只占步时 ~1/4（26 GiB ≈ 158 个 wbig 切片 ≈ 4.8 ms），
约束从 ridge 变成时间预算：整 tray 行数 64→256 多 1.4 ms、→512 多 5.1 ms、→1024 多 11 ms。
attention 按 owner rank 算：absorbed 每 (query, ctx) 对 209 kFLOP，P=220k 时 5 ms 预算下 c 只有
15–50；物化（kv_b 展开 + MHA）61 kFLOP 加每 ctx token 25 MFLOP 展开，交叉点 c≈170。

**判定**：
1. span 行放 batch 最前，行 0..c 连续，decode 行在后；FlashKDA 的 q/k/v/g 直接指向 conv 输出的行 0。
2. 每 rank 预留一个固定地址的 span slot（tensormap 基址装载时定死）；extend 开始 `line_copy` 拷入、结束拷回。
3. 两个 program：`decode` 与 `decode_span`（后者多 conv_span、f_b GEMM、FlashKDA 两核、kda_out_gate）；span 为 0 时不存在 grid 0。
4. decode 核收 `span_rows` 标量，行号小于它的 block 直接返回；padding 行的 g、beta 写负大数，对 state 是恒等更新。
5. MLA v1 用现有 DSL 核：span 行各带 `seq_lens = 前缀 + i + 1` 和复制的 block_table，零核工作先闭环正确性。

**调研事实**：FlashKDA（MIT，CUTLASS/CuTe，mma.sync，pegainfer 已在 GB300 编译）state `[H][128 v][128 k]` f32
与 kern rec 相同；kernel1 grid `(tiles16, H)`，kernel2 grid `(N, H)` 每 (seq, head) 串行走 tile；输入 q/k/v/g
`[T, H, 128]` bf16、beta `[H, T]`；q/k L2 norm、beta sigmoid、dt_bias / a_log gate 核内做，输出 o_norm 之前的 attn；
22 个 TMA 描述符作参数（M1 的 tensormap + pack 路）；shim 样板 `pegainfer-kernels/csrc/k3/k3_flash_kda.cu`。
第三方核进 manifest 的通用路（2026-09-03 沉淀）：vendor + PROVENANCE → 参考 launcher → `tools/kernel-capture`
捕获（launch 参数字节 + TMA encode 参数）→ `lift.py` 出 manifest 骨架 → 生成器 → 与 launcher dump 逐位对拍
（`tools/kernel-capture/README.md`）。
DSL MLA 核 `seq_len_q>1` 是 grid `(cluster, B×S_q, split)`，每个 q token 一个 CTA 各自流 KV，96 头 fold=1。
FlashMLA SM100 只有 sparse decode 与 dense prefill（MHA）；物化路径 shim `k3_flash_mla_prefill.cu`。
MegaMoE 协议上限 16896 token/rank。

**DAG**：

| 节点 | 内容 | 依赖 | 门禁 |
|---|---|---|---|
| A1 | DSL 核 c 行共用 block_table 的扩展曲线，c ∈ {1, 8, 32, 128, 256}，P ∈ {13k, 50k, 220k} | 无 | 每层时间表，决定 C5 |
| A2 | FlashKDA T 扫描，H=24 / 96，T=64..2048（pegainfer `k3_flash_kda_bench.rs`） | 无 | 每层时间表，进 D2 公式 |
| A3 | MegaMoE 256 / 512 token/rank（`k3_moe_ep`） | 无 | 每层时间 |
| B1 ✅ | manifest 侧不加新类型：`span` 是普通 var，`decode_span` 是普通 program，`span_at` 是 `[1]` 的 i32 input；runtime 新契约——program 的 env 只需带它的 launch 读到的 var（`CompiledProgram.vars`，其余归 MIN），所以 `decode` 不必给 `span` | 无 | 2026-09-03 过：4 层 EP4 / EP4×TP4 / EP1 与 93 层 span manifest verify 通过，runtime 53 单测 |
| B2 | runtime：span slot 预留成固定地址，bucket 键含 span | B1 | 单测 |
| B3 ✅ | `gen_k3_decode.py --span-max N`：每个 KDA 层 qkvg/wsm GEMM → conv_silu → span_gather → span_state_load → gemm（span_g）→ flash_kda → span_state_store → kda_core → kda_out_gate；span 的 q/k/v/out 是独立 buffer（TMA 描述符 load 时定死，不能指进批的行），`span_at` 定 span 在批里的位置（tray 批 own rows first，peer 上 span 在第 d 块） | B1 | 2026-09-03 过：`examples/k3-*.json` 全部带 `decode_span`（4 层 span 8、93 层 64） |
| C1 ✅ | `k3_span_gather`（并行 conv：前三行取 line 窗口、末三行写回，land g 为 bf16、写转置 beta、写 span_flow）+ `k3_span_state`（KDA slot ↔ f32 [h,128,128] 拷入拷出） | B1 | 2026-09-03 过：harness 对 CPU 参考，B ∈ {1,3,8,64}，span 在批中任意位置（`SPAN_AT=3`） |
| C2 ✅ | FlashKDA vendored 到 `tools/flash-kda/`（`7afb9f4`，MIT，PROVENANCE；只留 `<128,true,true,true,false>` 一个实例），cubin `flash_kda_d128.cubin`（sm_103a，host nvcc 13.1）；ABI 不反解模板，用 `tools/kernel-capture`（新增 `cuTensorMapEncodeTiled` 拦截 + `lift.py`）从 vendored probe 直接提出并与 pegainfer shim 捕获比对一致，记在 k3-kernel-abi.md K8。manifest 侧已消融：`tensormap` 参数类型与 `{"tensormap"}` 实参删除，改为 pack 字段 `{"at": 0, "tensormap": {...}}`（宽 128、偏移 64 对齐；裸描述符 = `bytes<128>` 一个字段，cute `TiledCopy` = `bytes<256>` 描述符 + 动态 stride int），生成器与 9 个 k3 example 同步改写，E1 MoE 门禁 EP4 bit-identical / EP1 rel RMS 1.66e-3 复跑通过（2026-09-03 tray03）。生成器 op 在 `tools/flash_kda_abi.py`（op 即数据：TiledCopy pack、workspace buffer、prepare / recurrence 两个 launch），`program_io` 例子跑通用 manifest 的任一 program 做门禁 | A2, B1 | **2026-09-03 过**：probe（`tools/flash-kda/probe.cu`）的 dump 与 kern op 的 out / state_out 逐位一致（第一次全零：capture 的分配表证明 recurrence p0 是 v、p10 是 out 的 TMA store 描述符，不是 q/v——见 lessons）；FlashKDA 数学对 K3 逐 token 参考（numpy f64）state 布局 = K3 rec，out relRMS 6.5e-3（核内 state 存 bf16） |
| C3 ✅ | `k3_kda_out_gate`：rms · gamma_o · σ(gate)，行并行，写回 `gated_kda` 的 span 行 | B1 | 2026-09-03 过：harness |
| C4 ✅ | conv_silu / kda_core 尾参 `const int* span_at, int span`，`(unsigned)(b - span_at[0]) < span` 的行跳过 | B1 | 2026-09-03 过：现有 harness 全过 |
| C5 | 条件项：kv_b 展开 GEMM + FlashMLA sm100 FMHA cubin | A1 结果差 | 与 DSL 路径逐 token 同 |
| D1 ✅ | kern-serve `tray.rs`：`Shape.span`（manifest 有 `decode_span` + `span` var + `span_at [1]` 输入）；`Layout` 每 cell 有行数，span cell 排在 owner 块最前，**组里没有这个 cell 的 rank 在块前垫 c 行 pad（`Layout.lead`）**——var 全 tray 一份、批每 rank 一份；行 (cell, j) 的位置 / slot / seq_len 按 `pos + j`；`span_at` = owner 在本 rank 块序里的下标 × b（没有即 0，指向 pad），每 rank 各写；`Capacity` 由 kern-serve 报 `(max_seqs + 1) × t`；tray 先把每个 rank 都 enqueue 再逐个 sync（MoE dispatch 在核里等 peer） | B2 | 2026-09-03：Layout / Shape 单测（含 lead 垫行）；E3 见下 |
| D2 ◐ | 第一版 policy：c = min(`--chunk`, `span.max`, 待喂 token 数, `seqs.max − (n − 1)`)，每步最老的还在喂 prompt 的序列拿 span，其余各一行；bucket 沿用现有表（span 行计入行数）。按稠密 / attention 预算派生 c 与 A1–A3 的曲线未做 | A1–A3 | 2026-09-03 E3 量到：256 行 span 步 ≈ 90 ms（decode B=8 38 ms，+52），64 行 +23 ms；冷 12.9k 并发者 mean ITL chunk 256 +37%、chunk 64 +60%——门（≤ +25%）要 D2 的预算版才可能过 |
| D3 ✅ | `Staged::read_i64` 一个 cell 多行时取末行；scheduler 把 span 的前 c−1 个 token 记为 prefill_tokens | D1 | 2026-09-03 单测（Layout 行序） |
| E1 ✅ | `k3_golden --span c`：行 0 把 fixture 的 token 按 c 个一步喂 `decode_span`（多 span），其余行逐 token；表每步按 cell 重铺 | B3, C1–C4 | **2026-09-03 tray07 过**（4 层 EP4）：逐 token 37/40 + 3 excused；`--span 16`（16/16/8）35/40 + 4 excused + 1 近平局（步 13，3.0 ulp，我们的 token 是参考的 #2）；`--span 40` 35/40 + 5 excused + 0 错；`--span 8 --seqs 4` 混合：行 0 37/40 + 2 + 1 近平局（5.5 ulp），行 1..3 只在步 13 的近平局上离开 fixture——无行间串扰。span 步时 8 行 2.08、16 行 1.95、40 行 2.40 ms（decode B=1 1.67 / B=4 1.80） |
| E2 ✅ | 93 层 EP4 12.9k oracle 按 c=200 分 65 个 span + 64 步续写 | E1 | **2026-09-03 tray07 过**：**115/128**（逐 token E2b 111/128），13 个不一致全在 scripted prompt 尾段（步 12834–12944），逐个 detokenize 是 `,`↔`;`、` speed`↔` perf`、`uted`↔`used` 一类的措辞近平局；**64 步自回归续写 64/64 与 oracle 逐 token同**。span 步时 **200 行 76.2 ms**（graph 捕获；逐 token 12 897 步 × 31.5 ms ≈ 406 s → 65 × 76 ms ≈ 5 s）。`--seqs 8 --mixed --distinct` 行 0 108/128（近平局带内），但 mixed 的复制批参照与含 span 的批形状不同、随机行第 2 步就翻，有 span 时这个检查不是门（参照批同形状未做）；行不串扰的证据是 k3_golden dump（span 旁的逐 token 行与单跑只差批形状噪声）与 harness `--span` |
| E3 ✅ | kern-serve conc1 与 `kern run` 同；conc8 加冷 12.9k | D1–D3, E2 | **2026-09-03 tray07**（93 层 EP4，表在 serve.md）：conc1 短 prompt 64 token 逐字同、ITL 32 ms 同；2k TTFT 0.83 s（逐 token 63.9 s），冷 12.9k **4.6 s**（416.8 s）；conc8 × 1k TTFT 0.56–3.2 s（38 s），稳态 ITL p50 38 ms 同；conc7 + 冷 12.9k：并发者 p50 38 / mean 52 / p90 91 ms（**mean +37%，门的 +25% 未达**，见 D2）；多轮命中第二轮 TTFT 112 ms。8 条相同短 prompt 并发的分歧是 0.2 logit 的近平局（k3_golden 倒 logits 证明），不是串扰 |

**风险**：长前缀 extend 的 attention 没有便宜的核，absorbed 每 extend token 对 220k 前缀 1.1 TFLOP，
任何核都逃不掉，A1 回答 DSL 核能否在 c ≤ 50 下把 KV 只流一遍；FlashKDA kernel2 每 rank 只 24 个 block，
T=512 时 32 个 tile 串行，A2 可能逼着 D2 把 KDA 项写进预算；FlashKDA 的 q/k norm 走 f32、kern decode 核走
bf16 链，E1 按近平局口径判而非逐位。明确不做：K4 DCP、空间 PCP gang、单独 prefill program、fp8 权重。

## 单卡遗留（未做）

- capture 补 launch→module id 映射（unified 双实例现靠 num_regs+cuobjdump
  间接定位，capture 直接记 module id 更干净）；生成器给自写核（argmax/
  embedding）也填 `sha256`（unified 双实例已钉哈希）。
- workspace 静态规划（liveness + 贪心 offset 复用；现在逐 buffer 独立分配）。
- step 边界 GPU 化（vs vLLM 差的 ~0.25ms/step）：token 反馈闭环进 graph——
  embedding 的 token_ids 直接由 next_token 喂，positions/slot_mapping/seq_lens
  可预知提前写，host 滞后一步异步取结果，步间不再 sync。E4 直接依赖它。
- attest 后续：kernel-as-package 目录里带上 attestation 当证据；bs>1 的
  workload（现在 bs=1 下 elementwise 核全是 launch 主导，roofline 列
  0.1%）；GEMM extern 的 FLOPs roofline（现在只算字节）；结构输入的
  domain 校验扩到 debug 模式下的设备侧 buffer（现在只查 host 写入）。
  多卡时：A、B 共用一个 runtime 装载；设备侧 compare op；rank-local 比较。
