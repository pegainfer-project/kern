# `kern-serve`：continuous batching + OpenAI 兼容 endpoint

```bash
# 独立 workspace（serving 栈不进 runtime 的依赖图和 CI）；binary 在 crates/kern-serve/target/
cd crates/kern-serve && cargo build --release
target/release/kern-serve --model-path /mnt/shared/weights/Qwen3-4B --gpus 3 --port 8000   # --gpus 0,1,2,3 drives a tray
# state 池默认按显存自动定：权重/激活/scratch 分完后，剩余显存减 1 GiB 全给 state，
# KV 页与 state slot 共用这份预算、按需互换（runtime.md）。`--capacity <tokens>` 或
# kern.toml 的 `capacity` 显式给则照旧。
# /v1/completions、/v1/chat/completions（流式 + chat template）、/v1/models、/metrics
```

manifest / kernels / weights 来自 kern.toml 的 target；`--model-path` 是给
**前端**的 HF 目录（tokenizer、chat template、`generation_config.json` 的
eos）。前端整个来自 pegainfer（`pegainfer-frontend`，底下是 vLLM 官方的
Rust server crates，git dep 钉 pegainfer main 的一个 rev），kern 只贡献引擎：`crates/kern-serve`。

## 分工

- **`kern-serve::scheduler::KernScheduler`** 实现 pegainfer 的 `Scheduler`
  契约（`submit` / `step` / `metrics`），跑在 pegainfer 的 `drive` 轮询线程
  上，独占一个 `Runtime`。策略刻意简单：
  - prefill 优先、不混批：每步先把 waiting 里的请求逐个（bs=1、chunk 级）
    prefill 到预算（`--prefill-budget`，默认 2048 token），再对全部 running
    序列做一步 decode；最后一个 prompt token 作为首个 decode 步的输入。
  - 准入即预留：请求在准入时向 runtime 租下最坏情况 `prompt + max_tokens`
    的全部 KV 页（`Runtime::lease` → `Lease`，序列结束即 drop 归还），
    decode 永远不缺页、不抢占。超过单序列上限（最窄页表行长 × 页）→
    `ContextLength`，超过整池 → `KvBudget`。`slot_mapping` / `block_table`
    的值只能从 `Lease` 算出来，scheduler 不碰裸页号。
  - decode 按 bucket（1,2,4,8,16,24,32,48,64,96,128,192,256）pad，每个
    bucket 首次使用时 capture 一张图；pad 行写进 scheduler 自己租的一页。
  - greedy：采样就是 manifest 里的 `argmax`。非 greedy 参数 warn 一次后按
    greedy 服务。EOS 本身不发出（pegainfer 约定），仍计入 `max_tokens`。
- **manifest**（`tools/gen_qwen3_decode.py`）多了一个 var `seqs`（≤256）和
  一个 program：
  - `decode`：原样，bs=1 契约（3D split-KV unified + reduce_segments——挖到
    的 reduce 是 Triton 在 `num_seqs=1` 下特化出的实例，ABI 里没有
    num_seqs，只能 bs=1）；
  - `decode_batch`：`seqs` 个序列各一行（tokens = seqs），attention 用
    prefill 的那份 2D causal 实例（vLLM 自己在 num_seqs 超过 3D 阈值时 decode
    就走它），grid.x = `ceil(5·tokens/4)`——盖住 vLLM 的 q-block 索引空间
    `tokens//4 + num_seqs`（seqs ≤ tokens 下恒成立），多出的 block 核内提前
    返回；表达式集合因此不用加"两个 var 相加"。
  - 元数据按序列：`block_table [seqs, 256]`、`seq_lens [seqs]`、
    `cu_seqlens_q [257]`（shape 不能是表达式，按上界声明）、`logits [seqs,
    V]`、`next_token [seqs]`；lm_head 的 m = `seqs`。
  - 哪个 bs 走哪个核是 manifest 的选择（两个 program），caller 按 bucket
    选，runtime 不知情。以后补 bs 2–16 的 split-KV = 再捕一次 bs=2 的
    decode 拿未特化的 reduce，多加一个 program。
- **runtime** 只改一处：CUDA graph 按 `(program, var 值)` 键控（原来一个
  program 一张图），加 `is_captured`。

## 实测（GB300 单卡，Qwen3-4B，2026-09-01）

- bs=1 `kern run` 不变：2.6 ms/step，输出逐字一致。
- `vllm bench serve --backend openai --dataset-name random --random-input-len
  1024 --random-output-len 128 --num-prompts 256 --max-concurrency 64
  --ignore-eos`：256/256 成功，**5138 tok/s 输出吞吐**（总 46k tok/s），TPOT
  中位 11.7 ms / P99 11.9 ms，TTFT 中位 62 ms / P99 1.0 s，E2E 中位 1.55 s。
  引擎侧：decode 5.5 ms/step @ ~60 seq，prefill 79.5k tok/s（chunk 512 走图）。
- 一致性：8 个不同 prompt 并发两轮逐字相同（确定性）；同 batch 内 16 份
  相同 prompt 里同 cohort 的 14 份逐字相同（行没有错位）。并发 vs 串行有
  2/8 在 ~25 token 处的近平局分叉，两边都连贯——batch 大小改变 GEMM 选核
  和 attention 归约顺序，和 vLLM 一样不 batch-invariant。

## 投机解码（`--rows`，2026-09-03）

```bash
target/release/kern-serve qwen3-4b-dspark --model-path /mnt/shared/weights/Qwen3-4B            # 缺省取最宽：7 行的 round
target/release/kern-serve qwen3.8-27b-dflash2 --model-path <Qwen3.8-27B 的 HF 目录> --rows 8   # 显式给；--rows 1 是 plain
```

投机不是模式开关，是 manifest 声明的一个形状：`round` program 的
`batch: {groups, rows}`（dspark 7 行、dflash2 8 行）。scheduler 装权重前
先算 `Plan`：`--rows`（缺省 manifest 声明的最宽）选出 `step` forward，
`rows − 1 ≤ MAX_SPEC_TOKENS`（前端计数器）、rows ≤ 页（pad 页要装下一组）、
rows > 1 时必须有 chunk program（prompt 逐 token 走 step 的模型只能 1 行），
`max_seqs` 按 `tokens.max / rows` 与 `groups` 上界收紧。之后 plain decode
与投机轮是同一段 `step()`：

- 每序列 stage `rows` 行（行轴的 token fill 里 `[next; rows]`，组轴的
  token fill 装 anchor），跑 `forward(b, rows)` 选中的 program，读
  `tokens [seqs, rows]` 与 `count [seqs]`（没 count 视为恒 1），每序列取
  `count` 个、`pos += count`。租约 = `prompt + max_tokens + rows − 1`：被
  拒的行落在新 pos 之后，下一轮覆写，target KV 与 draft KV 一样免费回滚。
- round 内的活全在设备上：`splice_draft`（anchor + mask 拼 draft 的
  ids）→ draft → `splice_verify`（anchor + draft 拼 verify 的 ids）→
  verify → precompute（verify 全部行的 tap 进 draft KV）→ `spec_count`
  （前缀匹配写 `count` 输出）→ GDN 模型再加 `spec_lines` + advance。一张
  图一次 sync；host 不再读 draft 再拼 verify，`nacc` 直接信。
- admission：chunk program 自己出第一个 token（`Forward.emits`：dspark
  的 prefill 尾部带 head + precompute，dflash2 本来如此）。
- greedy only；每轮 verify 是 `rows·b` 行的 target 前向，b 大到算力瓶颈
  后一轮比一步贵，`--rows 1` 关掉。分段的 draft / verify program 不再存
  在，逐轮 dump 用 `kern run --probe-dir`（按 label 后缀取 call 的输出）。

**验证**（tray03 GB300，2026-09-03，docs 段落做 prompt、512 token、
`ignore_eos`；stats 是 5 s 窗口的均值，取 running 最多的一条）：

| | conc1 == `kern run` | conc32 吞吐 | conc32 接受率 | 2026-09-01 基线 |
|---|---|---|---|---|
| qwen3-4b-dspark（rows 7，`--capacity 65536`） | 逐字同（683 tok/s，2.29 tok/step） | 5930 tok/s（32×512，含 prefill） | 3.02 tok/step、34%（running=22 的窗口） | conc32 5850 tok/s、19.2%（128 token） |
| qwen3.8-27b-dflash2（rows 8） | 逐字同（114 tok/s，1.75 tok/step） | 1709 tok/s | 2.65 tok/step、24%（running=32） | conc32 1236 tok/s |

512 token 的中文散文有几条掉进重复循环（一条 dspark conc1 接受 71%），
接受率偏高是它们抬的；不塌就是门禁。conc1 与 `kern run` 的对照 prompt
是英文 96 token（"In the summer of 1848…"），tok/s 是 `kern run` 的。
v3 的分相路径（host 轮流调 draft / verify / precompute，dspark 8 行
verify）：conc1 20.8% / conc32 19.2%，conc 1 / 8 / 32 ≈ 600 / 2560 / 5850
tok/s，普通模式 353 / 2048 / 6800——交叉点在 bs 16–32，v4 没有重测。

## 前缀缓存（K1，2026-09-02）

调度器持有一张 `Prefix` 表（kern-runtime，纯 host）：结束的序列留成 checkpoint，
新 prompt 从覆盖其真前缀的最长 checkpoint 起步（`Runtime::lease_from`），prefill
只补剩下的。两种模型一套机制，差别只在"何时留"：

- 纯 KV（qwen3-4b，页 16 token）：序列每填满一页就 `Runtime::checkpoint` 一次——页进
  共享链，不拷字节，所以任何早先的 prompt 或输出都按页粒度可复用；
- 带循环状态（qwen3.8-27b，页 784 token，GDN state 154 MB/序列）：只在请求结束时
  `Runtime::retire`——结束序列的 state slot 原样成为 checkpoint 的，不拷；因此只有
  "续着上一轮整段上下文"的 prompt 命中，同一 prompt 重发不命中（checkpoint 比它长）。
  slot 从 manifest 的 `seqs.max + 2` 个起，与 KV 页共用一份显存预算按需互换（K1b）：
  睡着的 session 的 checkpoint 拿着 slot，活跃请求再要 slot 就从空闲页拆，页不够
  再从空闲 slot 拆回来；租约 `Busy` 时按最久未命中淘汰，`Remapping` 时等它落地
  （stats 行的 `slots_used`/`slots`/`remaps`）。

门禁（本机 GB300，warm 服务与 cold 服务各一，greedy，`max_tokens 64`，prompt 为 46 KB
工程日志）：

| 模型 | prompt | cold prefill | 命中 | 命中后 prefill | 多轮 R3 warm vs cold |
|---|---:|---:|---:|---:|---|
| qwen3-4b | 13 983 tok | 888 ms（请求 1.23 s） | 13 968 tok | 27 ms（请求 0.36 s） | 64 token 逐字一致 |
| qwen3.8-27b | 14 256 tok | 请求 1.90 s | 14 319 tok（上一轮全文） | 请求 0.85 s | 64 token 逐字一致 |

qwen3.8 重发同一 prompt 不命中（唯一的 checkpoint 比它长），输出与首发一致。qwen3-4b
重发同一 prompt 命中 13 968，输出在第 10 个 token 后与首发分叉；两次命中之间完全一致。
分叉不是缓存的：同一 prompt 在 cold 服务上用 `--chunk 144`（末块同样是 15 个 token）
全量 prefill，得到第三种续写——三种切块三种 attention 归约顺序，这个位置是 bf16
近平局，与 `decode` / `decode_batch` 之间的分叉同类。
host 侧回放见 roadmap K1 / K1b 行（`crates/kern-run/examples/agentx_replay.rs`）。

K1b（页与 slot 共用预算）后的门禁（2026-09-02）：上表两行的输出逐字不变；qwen3.8-27b 单卡
预算 223.9 GiB = 9551 块 × 24 MiB，起步 4287 页 + 130 slot；200 个不同的短请求依次结束后
slot 长到 191（每个 checkpoint 拿一个，活跃请求再要就从空闲页拆，61 次 remap），此后重发
早先的 prompt 与首次输出一致、46 KB prompt 的 64 个 greedy token 与 cold 服务一致。
这条门禁第一次跑就抓到一个 manifest 的 bug：`gen_qwen35.py` 把 `(seqs.max + 2) × 48 = 6240`
当 vLLM conv kernel 的 `num_cache_lines` 写成字面量，kernel 用它做越界掩码，slot ≥ 130 的
conv state 静默丢掉（短 prompt 单块 prefill 从零态起步看不出来，14k 的 prompt 第二块起就错）。
按 slot 编号二分定位：页编号高没事、`kern run` 单序列没事，只有 kern-serve 里 slot ≥ 130
的多块 prefill 出错。manifest 里不许再出现 slot 数：conv kernel 现在拿 i32 最大值，掩码永不生效。

带状态模型的部分命中在算力上没有意义：state 快照之后的 token 必须整段重跑 forward
才能把状态推过去，attention 层的投影和注意力都省不掉，所以有效命中 = 最深的带
state 的 checkpoint，KV 页只是顺带共享（vLLM v1 也是这样：hit 取各层组最小值，
attention 的命中被截到 mamba 块边界）。快照放哪由调用方定——请求结束（agent 多轮）
和显式断点（roadmap K1b），不在每个块边界都存。

## 睡到 DRAM 的 session（K3，2026-09-03）

`--host-gib N` 在 tray DRAM 里 pin 一块 N GiB（`Runtime::reserve_host`，落在这张卡的
NUMA 节点上）。租约 `Busy` 时最久未命中的 resident checkpoint 不再直接丢，而是 park
到这块 host 内存（`Runtime::park`，页和 slot 都拷；host 放不下就先丢最冷的 parked），
runtime 攥着它直到拷贝落地才还页；命中 parked checkpoint 的 prompt 走 `Runtime::wake`：
租新页、把前缀拷回来，请求在 `waking` 队列里等 `Runtime::awake` 交出 `Lease`，其间
decode 步照常走——compute stream 从不等 transfer stream。stats 行多了
`parked / host_gib / parks / host_evictions / wakes / wake_tokens`。

门禁（tray08 GB300 单卡，qwen3-4b，`--capacity 32768 --host-gib 32`，greedy 64 token，
prompt 8189 token）：第一次请求 cold prefill；14 个不同的 12–13k 长 filler 把它挤到 host
（16 个 checkpoint 中 14 个 parked，24 GiB）；重发同一 prompt 命中 8176 token 的 parked
条目，wake 后 prefill 19 ms、请求 0.31 s（首发 0.67 s），64 个 token 与首发、与 cold 服务
逐字一致。

decode 抖动（同卡，`--capacity 65536 --host-gib 48`，8 路 stream 各 256 token 的 decode
负载，另一路 churn 轮流重发 3 个 25k-token 的 prompt、`max_tokens 1`、按 token id 发，
每个请求 wake 3.5 GiB；40 s 一组，`target/gate-logs/jitter.py`）：

| 负载 | token 间隔 mean / p50 / p90 / p99 (ms) | churn 请求 |
|---|---|---|
| 只有 decode（无 host 层） | 2.92 / 2.92 / 3.10 / 3.22 | — |
| decode + churn，命中 resident（无 host 层，池够大） | 31.3 / 3.03 / 85.4 / 85.6 | 89 ms |
| decode + churn，命中 parked（每个请求 wake 3.5 GiB） | 26.1 / 3.08 / 83.6 / 85.5 | 104 ms |
| 只有 decode（host 层开着） | 2.95 / 2.94 / 3.11 / 3.25 | — |

拷贝本身给 decode 添的抖动在 p50 上是 3.03 → 3.08 ms（+1.7%），p90/p99 不变，wake 的
成本（+15 ms）全落在被唤醒的那个请求自己身上。p90 的 85 ms 与 host 层无关：命中之后
剩下的几个 token 走 `prefill`，这一块在 25k 上下文上要 40 ms（挖来的 prefill attention
核只按 head 并行，长上下文下带宽吃不满），prefill-first 的每步又把它排在 decode 前面——
resident 命中同样如此。这是 K5（prefill 作为 decode 步的 filler）要解的，不是 K3 的。

同一 session 醒来后再睡，host 上按 device 页节点去重的链认不出它（醒来的是新页），会
再拷一份；旧的那份最冷，缺地方时先走。

## tray 级（E5 第四块，2026-09-03；t=4 门禁 2026-09-04 过）

`kern-serve --gpus 0,1,2,3` 一个进程驱一个 tray：`tray.rs` 持 n 个 `Runtime`，单线程
按序驱动（graph launch 1–3 µs，一线程驱四卡与四线程等价，实测见 multi-gpu.md），每步
所有 rank 跑同一个 program、同一组 var 值、同样的行数；manifest 的 `topology` 里 `tp`
组是"一个 batch 的组"（连续的 rank），其余组（`ep`）跨全部 rank；跨 tray 的 world
不在这里（rendezvous 是 harness 的事）。scheduler 只看得到 `Row` / `Snapshot` /
`Sleeping` / `Rising` 和 `Cell`，看不到 `Runtime`、`Lease` 或某个 rank 的输入。

- **先全部 launch 再逐个 sync**：一步里 n 个 rank 的图先全部 launch（`Runtime::enqueue` /
  `enqueue_captured`），再逐个 `synchronize`——EP 的 dispatch 与 tray collective 都在核里等
  peer，先 sync rank 0 就等到核的超时（`CUDA_ERROR_LAUNCH_FAILED`，2026-09-03 E3 第一次
  跑 K3 EP4 就撞上；qwen 单卡 smoke 看不出来）。
- **一行 = tp 组每个成员一份**：owner 卡持 MLA 页 + KDA slot（`Runtime::lease`），
  peer 卡只持 slot（`Runtime::lease_slot`，runtime 新增 slot-only 租约，见 runtime.md）。
  lease / lease_from / fork / checkpoint / retire / park / wake / awake 各在组里每个成员
  做同样的事；**全成或全不成靠构造**：`Row` 只在每个 rank 都给了 `Lease` 之后才存在，
  哪个 rank 说 `Denied`，`?` 提前返回把前面拿到的 drop 掉即回滚（各 rank 的 pool
  独立记账、slot 编号不跨卡比对，所以没有一致性要重建）。park 是唯一不能半途撤销
  的动作（拷贝已入队），runtime 拆成 `room`（找地方，`Room` drop 即退）+ `park`（拷），
  tray 先在每个成员上找齐再拷。`Waking` 提前 drop 会等拷贝落地，所以 wake 的回滚就是 drop。
- **owner**：新行落到"还开着（行数 < `--max-seqs`，per rank）且行数最少、再比页占用最少"的 rank
  （2026-09-04 之前只比页：页数含留着的快照，持有 12k 快照的卡在 conc8 里一行都分不到——`[3, 2, 0, 3]`），
  终身不变，后代（checkpoint、parked、wake 回来的）都跟着它——页在那张卡上，pinned
  块绑在那张卡的 NUMA 节点上。
- **`Prefix` 按 tray 键**：`Prefix<Snapshot, Sleeping>`，键还是 token 哈希链、与卡无关。
- **staging 按 fill 的轴**：fill 或 line 表跨过的第三个 var 就是 tray 轴（k3 的
  `rows`）；tray 轴上的 token fill / line 表 / tokens 输出跨组（本卡的行先、再按组序轮到
  其它成员的块，collective 假定的布局），行轴 / 组轴上的 slot / seq_len / 页表是本卡自己
  的行；qwen 的契约（没有 tray 轴）是 t=1 的特例，同一条代码。**块不等长**（2026-09-04）：
  每个 rank 的块是它自己的行数（至少 1 行 pad），组内块长之和垫到阶梯就是 tray 的行数
  （`rows` var；多组时各组垫到同一个数，垫在最小的块上；t=1 退化成每卡等块），块的前缀和写进
  `blocks` fill，collective 核按它换算行号（peer_collective.cu）；own 轴的 buffer 仍按最大块的
  bucket b 铺、pad 页每卡一页。一条序列 256 行的 run 于是让 tray 是 256 + 3 行而不是 4 × 256——
  之前 768 行 pad 走完整套核和每层 allreduce，是 TP4 span 步 142 对 t=1 87 ms 的全部差额。
  256 之上的阶梯加了 264 / 272 / 288 / 304 / 320，一条 run 加几张卡的一两行只垫几行（曾试过把 run 缩到
  阶梯值上一行不垫：6 个 token 的 prompt 被切成两步、1k 的 4 步变 5 步，省几行 pad 换整整一步，撤了）。`Staged` 借住 `&mut Tray` 直到输出读完，中间不能 lease / fork /
  再 stage。manifest 有 `error` fill 的输出时每步读一次，非零即该步失败。
- **K3 没有 prefill program，prompt 走 span**：`prefill` 可选；没有时 prompt 在 prefix
  命中之外的部分走 decode 步，该行的输出在最后一个 prompt token 进去之前丢掉。manifest 有
  带 `batch.span` 的 program（K5：`"batch": {"groups": …, "rows": 1, "span": "span"}` +
  `span_at` fill 的 `[1]` i32 输入，`Protocol::spanned(b)` 选它）时一步里**每个 tp 组一个** cell 可以是
  一段 run——同一序列的 c 个连续 token 各占一行（`Layout` 里每 cell 有行数，run 排在 owner 块
  的最前面，`span_at` = 本组 run 的 owner 块的偏移，每个 rank 各自算），位置 / slot / seq_len
  逐行递增，输出取末行；**组里没有 run 的 rank 在自己块最前面垫 c 行 pad**（`Layout.lead`）
  ——var 是全 tray 一份，每个 rank 都跑 span program、都跳过 `[span_at, span_at + c)`，
  EP4 下 run 落在 rank 1 时 rank 0 若不垫，它自己的第 0 行就被当 span 跳过、又被 span
  核当 span 算进那条序列的 state（2026-09-03 E3 第一版：8 条相同 prompt 出 8 种答案）。scheduler
  每步在每个 tp 组里挑最老的还在喂 prompt 的序列各喂一段 run（`runs`，纯函数；2026-09-04 之前整个 tray
  每步只放一条，t=1 的另外三张卡在 span 步里跑的是 pad），所有 run 一样长：
  c = min(`--chunk`, manifest 的 `span.max`, 各 run 待喂 token 数的最小值, 每个 rank 的余量)——
  余量按 rank：放 run 的 rank 是 `seqs.max + 1 − 本卡行数`，垫 lead 的 rank 是 `seqs.max − 本卡行数`；
  一条短 prompt 会把同步别人的 run 拉短一步，换它自己一步内出首 token，不排在别人后面。
  其余序列各一行。每 rank 的行数：k ≤ `--max-seqs` 时 bucket 再钉到 max_seqs（原规则），run 超过它时
  走阶梯（`rows_per_rank`）——同一个 run 不论同步还有什么都落同一个 bucket；cuBLAS 按 m 选核，2k prompt
  的尾块 223 行不垫到 256、12.9k 的尾块 160 不垫到 192，输出就在近平局处与 K5 线分道（2026-09-04 移植第
  二轮踩过，t=1 conc1 的 sha 全部对不上，runtime 用 93 层 12.9k oracle 证明是同的）。没有 span program 时
  逐 token（12.9k 的 prompt 要 12.9k 步）。b>1 走形状包含
  `(b, 1)` 的 program 里 `groups` 上界最紧的那个；没有 chunk program 时 `--rows` 只能是 1。
- 权重按 rank：kern.toml 的 `weights` 里 `{ep}` / `{tp}` 换成该 rank 在组里的下标，文件
  名里的 `*` 按名字序展开（`dense-tp4/r{tp}/l*.safetensors`），mmap 不读入。
  `--capacity` / `--host-gib` / `--max-seqs` 都是 per rank。

**t=1 smoke（2026-09-03，tray03 GPU 1，qwen3-4b，`--capacity 65536`）**：conc1 输出与 `kern run`
逐字同；4 路并发两次跑结果一致、与单跑一致（prompt 0 的分歧是 `decode` / `decode_batch`
两个核的近似平局，见 lessons）；prefix 命中三条路径——resident（8972 token 的 prompt 命中
4576）、host 全量 wake（命中 8960 后 12 token prefill）、host 半量 wake（4589 的 prompt 命中
4576）——warm 与 cold 24 token 逐字同；wake 回来的 937 页与 park 时逐页 digest 相等（运行时
`park_wake` 例子加了 `--wake` / `--every-page`，串接每页 checkpoint 的链与部分 wake 都逐字节回）。
`--host-gib 8` 下 c4 的填充就把 host 层打穿（一条 15k 的快照 2.06 GiB），测 wake 用 24。

**K3 93 层 EP4 span（E3，2026-09-03，tray07 4×GB300，`--max-seqs 16`，`--chunk 256` 对逐 token 的
`--chunk 1`；prompt 是 docs/*.md 语料按 K3 tokenizer 切的 12.9k / 2k / 8 × 1k token，脚本与
逐请求数据在 `~/bench_results/2026-09-03-k5-span-kernels/`）**：

| 场景 | span（chunk 256） | 逐 token（chunk 1） |
|---|---|---|
| conc1 短 prompt 64 token | 逐字同，ITL 32 ms | ITL 32 ms |
| conc1 2k prompt TTFT | **0.83 s** | 63.9 s |
| conc1 冷 12.9k prompt TTFT | **4.6 s**（51 个 span 步 ≈ 90 ms/步） | 416.8 s |
| conc8 × 1k prompt TTFT | 0.56–3.2 s（span 一步一条，排队） | 38 s（8 条同时逐 token） |
| conc8 稳态 decode ITL p50 | 38 ms（B=8） | 38 ms |
| conc7 decode + 3 s 后冷 12.9k 到达 | 12.9k 的 TTFT 4.8 s；其余 7 条 256 步的 ITL p50 38 / mean 52 / p90 91 ms | — |
| 同上，`--chunk 64` | 12.9k 的 TTFT 16.4 s；其余 7 条 ITL 61 ms 整段 | — |
| 多轮 prefix 命中（1k + 答案 + 新一轮） | 第二轮 TTFT 112 ms（第一轮 463 ms） | — |

- **span 步的税**：其余序列在有 256 行 span 的那一步 ITL ≈ 90 ms（+52 ms），64 行 ≈ 61 ms（+23 ms）。
  12.9k 按 256 切是 51 步、按 64 切是 202 步，所以 chunk 256 两头都好（TTFT 4.8 对 16.4 s，
  并发者 mean 52 对 61 ms）。K5 的门"ITL ≤ +25%"按 mean 算是 **+37%**（52 对 38，p50 不变）——
  未达；再往下要 D2 的预算 policy（span 长按稠密 / attention 预算定，或 span 步只带一部分
  decode 行），不是 span 实现的事。
- **MegaMoE BLOCK_M 96（2026-09-04）**：`k3_moe_bench` 在真实 router 分布下扫 ladder，BLOCK_M 96 在 8–512
  token/rank 全段最快（真实路由 −12～15%，输出逐位同），`pinned_config` 改钉它后复测：decode 步 ITL p50
  38.0 → 36.5 ms，12.9k 末尾 200 行 span 步 76.2 → 73.2 ms，冷 12.9k TTFT 4.62 → 4.48 s，conc1 输出 sha 同；
  门仍未过。数据在 bench_results 2026-09-04-k5-span-profile。
- **输出**：conc1 短 prompt 64 token 与逐 token 逐字同；2k / 12.9k 的输出与逐 token 在第 8 / 第 1
  个近平局后分道（两条数值路径，cuBLAS 按 m 选核，同 t=1 smoke 的注）；同一路径自己是确定的
  ——冷 12.9k 的 256 token 在 5 次不同并发环境下 sha 全同（ad2eea8cd2f0），2k conc1 重启服务后同。
- **相同 prompt 并发不是逐字相等的门**：8 条"The capital of France is"锁步，3 条答 " Paris."、5 条答
  " the capital of France is…"——`k3_golden` 把两种 batch 形状的 logits 倒出来比，两种形状的
  hidden / KDA state / KV 都只差 bf16 噪声，" the" 对 " Paris" 的 top-2 差 **0.2 logit**（13.53 对
  13.32；另一形状 13.63 对 13.90），是近平局，不是串扰。串扰（第一版没垫 lead pad）的样子是
  8 条 8 种、互不成句的答案。见 lessons。

**K3 EP4×TP4（t=4，2026-09-04，tray07 4×GB300，二进制与 t=1 同一份；等块版的脚本与逐请求数据在
`~/bench_results/2026-09-04-k5-v4-port-tp4/`，块不等长版在 `~/bench_results/2026-09-04-tp4-blocks/`）**：

- **4 层逐 token 对 `k3_golden`**（`k3-4l-ep4-tp4.json`，`--max-seqs 6 --chunk 8`，前 i 个 token 作 prompt、取 1 个，
  按 k3_golden 的近平局规则判）：EP4×TP4 35/40 + 4 excused + 1 近平局（step 13，3.0 ulp，与 `k3_golden` TP4
  同一个 token）；EP4 t=1 37/40 + 3，与 `k3_golden` 同。块不等长之后两者不变（tray 4 rank 等块时 `blocks` 表
  退化成 q × b，走的是新核）。
- **93 层 E3 场景在 EP4×TP4 上**（`k3-93l-tp4-span.json`，`--max-seqs 16 --chunk 256`；中列是块不等长之前同日的
  等块版，右列是同日同二进制的 EP4 t=1）：

| 场景 | EP4×TP4（块不等长） | EP4×TP4（等块，之前） | EP4 t=1（同日） |
|---|---|---|---|
| conc1 短 prompt 64 token | 逐字同（sha 与 t=1、K5 线同），ITL **22.7 ms** | 同 | 31.2 ms |
| conc1 2k prompt TTFT | 0.86 s | 1.26 s | 0.80 s |
| conc1 冷 12.9k prompt TTFT | **4.59 s**（51 个 span 步 ≈ 90 ms/步） | 7.39 s（≈ 145 ms/步） | 4.47 s（≈ 88 ms/步） |
| conc8 × 1k prompt TTFT | 0.64–3.32 s | 0.89–5.18 s | 0.67–3.29 s |
| conc8 稳态 decode ITL p50 | **28.5 ms** | 29.3 ms | 36.3 ms |
| conc7 decode + 3 s 后冷 12.9k 到达 | 12.9k 的 TTFT 4.87 s；其余 7 条 ITL p50 28.8 / mean 42–48 / p90 92–93 ms | 9.31 s；28.7 / 52–64 / 146 ms | 4.64 s；36.4 / 47–52 / 88–89 ms |
| 多轮 prefix 命中第二轮 TTFT（接在 E 之后，命中的是 D 里 conc8 步算的快照） | 487 ms | 722 ms | 462 ms |

  - decode 步 TP4 比 t=1 快 8–9 ms（E5 的账）。**span 步曾反过来慢 55 ms**（142 对 87，nsys 拆解在
    `~/bench_results/2026-09-04-tp4-prefill/`）：全部是 pad 行的账——tray batch 每 rank 行数相同，一条序列 256 行的
    run 让 tray 变成 1024 行、768 行是 pad，它们被 `kda_core` / `conv_silu` 当 decode 行算（+25 ms，t=1 的 span 行
    是跳过的）、进每层的 allreduce（+19）、land / rms（+8）、allgather（+4）；真实计算 TP4 66 ms 对 t=1 81（GEMM 切列
    快 7）。**块不等长之后（上表左列）span 步 ≈ 90 ms，与 t=1 持平**：conc7 里并发者的 ITL p90 就是 span 步
    （93 对 t=1 的 88），12.9k 冷 TTFT 4.59 对 4.47 s。持平而不是更快，因为 MoE 扫描、MLA、KDA 递推都不按 rank
    分摊；要更快得把 run 的 token 切到各 rank（K4/K5 重定义）。conc8 里 TP4 的 ITL p90 有四条是 84–88 ms：
    它们先出首 token，之后跟着别人的 span 步走（一步一个 span，四卡一个 tray），等块版 TTFT 慢到大家几乎一起
    出首 token，反而没这一段；总时长 8.9 → 6.6 s。`k3_golden` 的 tray batch 不接 span，TP4 的 span 路径没有
    oracle 门禁。
  - 输出：t=1 的 conc1 sha（短 / 2k / 12k 冷 / 12k 迟到）与 K5 线 2026-09-03 与 BLOCK_M 96 复测全同
    （ec2071a1bd49 / eed14fa316ea / 607802289f15 / ad2eea8cd2f0），块不等长之后 t=1 全部不变；TP4 短 prompt 同，
    2k / 12k 在第一个近平局后分道（TP 归约是另一条数值路径；块不等长之后 TP4 的 12k 冷 sha 反而与 t=1 同，
    2k 仍不同——近平局两边落哪边）。并发的 1k 条目每次跑的 sha 都不同（到达顺序决定 bucket 组合），K5 线自己
    两次跑也如此，不作门。无 panic、无 `tp_err`。

**每个 tp 组每步各一条 run（2026-09-04，tray07，同一天的 E3 场景；脚本与逐请求数据在
`~/bench_results/2026-09-04-t1-runs-per-rank/`）**：之前整个 tray 每步只放一条 run，t=1 的另外三张卡在 span 步里
跑的是 c 行 pad——两种配置的 decode 都是四卡并行，prefill 都只用一张卡的算力。改成每个 tp 组各挑最老的还在喂
prompt 的序列各喂一段 run（`scheduler::runs`），所有 run 一样长；同时 owner 从"页最少"改成"行最少、再页最少"
（页数含留着的快照，之前持有 12k 快照的卡在 conc8 里一行都分不到，`[3, 2, 0, 3]`，改后 `[2, 2, 2, 2]`）。
对照左列是块不等长那天的同二进制 t=1：

| 场景（EP4 t=1） | 之前 | 现在 |
|---|---|---|
| A 短 prompt TTFT / ITL p50 | 221 ms / 31.0 | 233 / 31.0（sha 同） |
| B 2k TTFT | 801 ms | 816（sha 同） |
| C 冷 12.9k TTFT | 4466 ms | 4476（sha 同） |
| D conc8×1k TTFT 均值 [min–max] / ITL p50 / p90 | 1962 [669–3294] / 36.4 / 65（四条 83） | **917 [612–1135]** / 34.5 / **34.5** |
| E 迟到 12.9k TTFT；其余 7 条 TTFT 均值 / p90 | 4644 ms；2075 / 88 | 4543；**1199** / 87 |
| F turn2 TTFT | 462 ms | 474（sha 同） |

  - conc8：8 条 1k 各 4 个 chunk，两条一卡，第二条等第一条的 4 步——最慢一条 1.13 s 就是 8 个 span 步；之前
    32 个 span 步串在一张卡上，最慢 3.3 s。ITL p90 从 65 回到 34：没有人再跟着别人的 span 步走。
  - 只对 span 步分摊，decode 步没变（ITL 34–36 同以前的抖动范围）。同 prompt 的 conc1 sha 四个场景全同；
    并发条目的 sha 变了，一是同以前"不作门"，二是 room 之前按整个 tray 的行数算（`seqs.max + 1 − n`），conc8
    下把 run 切成 249、247、…（旧日志里 span 249/247/246/244 的 capture 就是它），现在按 rank 算、整 256。
    TP4（一个组，每步仍一条 run）conc1 sha 同；conc8 按 TTFT 排序后相邻两条的间隔（一条 1k prompt 的 4 个
    span 步）三轮都是 ≈ 400 ms，首条的 TTFT 638 / 688 / 809 ms 在轮与轮之间漂，不是 span 步变了。
  - 一条短 prompt 会把同步别人的 run 拉短到它的长度（c 取最小）：换它一步出首 token。conc8 里没触发
    （8 条一样长）；混合长短的公平性是 K5 D2 预算策略的事。

**没测**（按 CLAUDE.md 的门禁排队）：
1. t=1 qwen3.8-27b（有 slot 的路径）conc1 与 `kern run` 同，K1/K3 门禁数字不变；
2. owner-only 页在 t>1 下：mixed 行、prefix 命中（retire → lease_from）、park / wake 之后 warm == cold；
3. 全成或全不成：某一卡 `--host-gib` 故意给小，park 整体退回、四卡 host 占用回到原值；
4. 93 层短 prompt 的 conc1 / conc8 步时对 k3_golden 的 20.8 / 25.5 ms（E3 量到 ITL 32 / 38 ms，含 tray 的 staging 与 HTTP，没拆）。

## 没做（按需要加）

span 长的预算 policy（K5 D2：按稠密 / attention 预算定 c，现在是 `--chunk` 上限）、抢占 / 动态页分配、
真采样（temperature/top-p 作为 manifest 内的 `sample` op；投机下是 rejection sampling）、logprobs / echo、
bs 2–16 的 split-KV decode、步间 host 空转（token 反馈进图）。
