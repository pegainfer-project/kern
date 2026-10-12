# `kern-serve`：continuous batching + OpenAI 兼容 endpoint

```bash
# workspace 成员但不在 default-members 里（serving 栈不进裸 cargo build / CI 的编译）；需要 protoc、libssl-dev
cargo build --release -p kern-serve
target/release/kern-serve --manifest examples/qwen3-4b.json --kernels kernels --weights weights/Qwen3-4B --gpu 3 --port 8000
# a manifest with a topology runs one process per GPU: rank 0 adds --listen host:port, rank q runs
# --gpu g --head host:port --rank q ("每卡一个进程")
# state 池默认按显存自动定：权重/激活/scratch 分完后，剩余显存减 1 GiB 全给 state，
# KV 页与 state slot 共用这份预算、按需互换（runtime.md）。`--capacity <tokens>` 显式给则照旧。
# /v1/completions、/v1/chat/completions（流式 + chat template）、/v1/models、/metrics
```

manifest / kernels / weights 走 `--manifest` `--kernels` `--weights`（与 `kern run`
同一套 flag）。kern-serve **不读 kern.toml**：那是改 kernel 的开发循环的文件
（reference、dumps、test seed），服务进程的输入全在命令行上，和 vLLM 一样。
API 里的模型名缺省是 manifest 的 `model`，`--served-model-name` 覆盖。
**前端**要的 HF 目录（config.json、tokenizer、chat template、`generation_config.json`
的 eos）就是 `--weights` 第一项所在的目录：目录本身，或去掉 `.safetensors` 文件名。前端整个来自 pegainfer（`pegainfer-frontend`，底下是 vLLM 官方的
Rust server crates，git dep 钉 pegainfer main 的一个 rev），kern 只贡献引擎：`crates/kern-serve`。

当前钉在 pegainfer `18d0c29d` / vLLM `f4dde313`（2026-10-08）。`18d0c29d` 是 pegainfer #1140：chat 层量出
`Request::history_tokens`（见下文"带循环状态"一条）。之前的 `f53b282f` bump 带来的：
停止条件改为请求自带的 `StopPolicy`（前端按 vLLM 的规则把 EOS、`ignore_eos`、显式 stop id
降成一份策略，scheduler 只 `classify`，结束时带 `stop_cause`），kern-serve 不再自己读
`generation_config.json` 的 eos，`--stop-tokens` 删除；DSV4.1 renderer 的 reasoning effort
映射跟着上游 #58316 的 Flash 版参考改了（缺省 75、low 50、high 75，`minimal` / `medium` 拒绝），
fixture 的期望随之更新。更早的 #56260 仍在：DSV4/V4.1 历史 tool call 的非法 JSON 或非对象参数
按原文包在 `arguments` 中，不再拒绝请求；双重编码的对象也保留原文，renderer 回归测试只对这一处
已知差异调整期望，仍比较完整对话文本。


## 分工

- **`kern-serve::scheduler::KernScheduler`** 实现 pegainfer 的 `Scheduler`
  契约（`submit` / `step` / `metrics`），跑在 pegainfer 的 `drive` 轮询线程
  上，独占一个 `Runtime`。策略刻意简单：
  - prefill 优先、不混批：每步先把 waiting 里能坐下的请求按序逐个（bs=1、
    按 `--chunk` 切块、逐 launch 不走图）整条 prefill 完，再对全部 running
    序列做一步 decode；最后一个 prompt token 作为首个 decode 步的输入。
    `--chunk` 缺省是 manifest 的 `tokens` 上界，只能改小。没有预算、没有分时：
    一条长 prompt 进来，正在 decode 的序列等它整条喂完（roadmap D2）。
    **2026-09-09 tray07 GB300 × 1，Qwen3.8-27B（registry cubins）**：`kern run` 35.9k token
    prompt，prefill 2048 块逐 launch 22.3k tok/s（走图 22.3k，512 块 18.2k），48 步 decode
    图 11.14 ms/step、`--eager` 11.13 ms/step，三种设置文本逐字相同；`--chunk 4096` 报
    "the manifest's `tokens` bound is 2048"（当时的 manifest）。kern-serve 同一 manifest：conc1
    completion 与之前逐字同，29.9k token 的 prompt 走 15 块 prefill 21.6k tok/s，日志里只有
    `decode` 捕图一次。同日 tray08，上界提到 8192 的 manifest（bdd2805）：5 块 23.5k tok/s
    对 18 块 22.4k，文本逐字节同；kern-serve chunk=8192，29.9k token 24.0k tok/s。
  - 准入即预留：请求在准入时向 runtime 租下最坏情况 `prompt + max_tokens`
    的全部 KV 页（`Runtime::lease` → `Lease`，序列结束即 drop 归还），
    decode 永远不缺页、不抢占。超过单序列上限（最窄页表行长 × 页）→
    `ContextLength`，超过整池 → `KvBudget`。`slot_mapping` / `block_table`
    的值只能从 `Lease` 算出来，scheduler 不碰裸页号。
  - decode 按 bucket（1,2,4,8,16,24,32,48,64,96,128,192,256）pad，声明了
    `graph` 的 program 每个 bucket 首次使用时 capture 一张图；pad 行写进
    scheduler 自己租的一页。
  - greedy：采样就是 manifest 里的 `argmax`。非 greedy 参数 warn 一次后按
    greedy 服务。停止 token 带着 `stop_cause` 交给 ledger（pegainfer 约定：前端像 vLLM
    一样把它计入 `completion_tokens`、从文本里剥掉），计入 `max_tokens`。
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
target/release/kern-serve --manifest examples/qwen3-4b-dspark.json --kernels kernels \
    --weights weights/Qwen3-4B --weights weights/dspark-qwen3-4b-b7                  # 缺省取最宽：7 行的 round
target/release/kern-serve --manifest examples/qwen3.8-27b-dflash2.json --kernels kernels-qwen38-dflash2 \
    --weights weights/Qwen3.8-27B --weights weights/Qwen3.8-27B-DFlash2 --rows 8     # 显式给；--rows 1 是 plain
```

投机不是模式开关，是 manifest 声明的一个形状：`round` program 的
`batch: {groups, rows}`（dspark 7 行、dflash2 8 行）。scheduler 装权重前
先算 `Plan`：`--rows`（缺省 manifest 声明的最宽）选出 `step` forward，
`rows − 1 ≤ MAX_SPEC_TOKENS`（前端计数器）、rows ≤ 页（pad 页要装下一组）、
必须有 chunk program（prompt 只走它），
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
- 带循环状态（qwen3.8-27b，页 784 token，GDN state 154 MB/序列）：state 回不去，只在
  两处留。一处在 prefill 途中、下一轮 prompt 与这一轮分开的地方 `Runtime::checkpoint` 一次
  （那半页和 state slot 拷给快照）：chat template 的 generation prompt（qwen3.8 是
  `<|im_start|>assistant\n<think>\n`）在下一轮的历史里渲染成别的样子（`<think>\n\n</think>`），
  所以两轮 prompt 共享的前缀到最后一条消息结束为止；这个位置由前端量，不由 scheduler 猜——
  pegainfer 的 chat 层在 `/v1/chat/completions` 前把请求的最后一条消息单独渲染两遍（带 / 不带
  generation prompt），两份 token 不共享的尾巴长度写进 `vllm_xargs.generation_prompt_tokens`，
  bridge 算成 `Request::history_tokens = prompt_len - 尾巴`（跨边界的 token 算进尾巴，快照只会
  提前不会越界；`add_generation_prompt=false` / `continue_final_message` 时尾巴为 0）。这是这一轮
  唯一的快照：下一轮不管渲染我们的回复还是录下来的回复（AgentX、改过历史的对话）都从这里续，
  只付回复和其后的部分；同一 prompt 重发也命中它，只付尾巴。量不出 history 末尾的请求（completion、
  前端读不懂的 chat body）才在结束时 `Runtime::retire`——结束序列的 state slot 原样成为
  checkpoint 的，不拷。会话走过去的旧轮留到要腾地方时先走（pool.md「带 state 的条目分档」）。history
  末尾的快照拿不到 slot 时按租约的规则腾地方，remap 在途就等它落地（`settle`，唯一一处
  阻塞等 remap：这个 state 过了这一刻就留不下了，而 prefill 刚读回输出、stream 已空，等的只是
  remap 本身的几毫秒），腾不出就不留，请求照跑（stats 行的 `prompt_checkpoints` / `prompt_skips`）。
  slot 从 manifest 的 `seqs.max + 2` 个起，与 KV 页共用一份显存预算按需互换（K1b）：
  睡着的 session 的 checkpoint 拿着 slot，活跃请求再要 slot 就从空闲页拆，页不够
  再从空闲 slot 拆回来；租约 `Busy` 时按最久未命中淘汰，`Remapping` 时等它落地
  （stats 行的 `slots_used`/`slots`/`remaps`）。

命中数逐请求报给调用方：`usage.prompt_tokens_details.cached_tokens` 就是这条请求
没有付钱的 prompt token 数（`Row::prefix()`，与 stats 行的 `prefix_hit_tokens`
同源）。DSv4.1 一次两轮（tray05，prompt 6159 token、生成 96）：第二轮 prompt
6263 token，报 6254 = 6159 + 96 − 1，快照不复用最后一个 token。未命中时上游的
序列化器不发 `prompt_tokens_details`，字段是 `null`。

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
按请求的最坏长度一次取够页、把前缀拷进去，请求在 `waking` 队列里等 `Runtime::awake` 交出
这份 `Lease`，落地即开跑（2026-09-15 起，`pool.md`；之前先醒成 `Checkpoint` 插回索引、再
`lease_from`，两次分配在 DSv4.1 EP4 的第二轮上 wake ↔ park 活锁，见 lessons.md），其间
decode 步照常走——compute stream 从不等 transfer stream。stats 行多了
`parked / host_pct / parks / host_evictions / wakes / wake_tokens`，以及按 tier 分的命中
`resident_hits / resident_hit_tokens / host_hits / host_hit_tokens`（host 层有没有被打到
一眼可见）。

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

同一 session 醒来后再睡，醒进 lease 的整页节点带着 host twin，只拷新增的页（`pool.md`）。

## DSv4.1 Flash 的 host 层命中（2026-09-11，tray05）

四卡 EP4、`--capacity 8192 --host-gib 4`：2013 token 的开场白 + 312 token 的回答（EOS
结束）留下 checkpoint；第二轮立刻问命中 resident（`prefix_hit=2325`，prefill 14 ms）；
32 条 2.5–3.3k 的 filler 把每张卡的池子打满（`parks=8`，日志 `parked tokens=2325`）后再问
同一轮，`woken=true`、新租 19 页、prefill 12 ms，**64 token 与 resident 命中逐字相同**。
新起的 server 上冷 prefill 同一 prompt 在第 35 个 token 处同义替换后重新汇合（回答的 312
token 一边走 decode 步一边走 prefill chunk，核和累加顺序不同，近平局翻转）。

有 per-seq state 的模型只在快照的整长上可用，所以命中还要求 chat template 重新渲染的
历史 re-tokenize 出模型当初生成的那串 id：被 max_tokens 截断在词中间的回答（64、400 都
试过）再问一轮全是 `prefix_hit=0`，resident 也不命中。记录与脚本在
`bench_results/2026-09-11-dsv41-host-tier/`。

## 每步的 host 空档（2026-09-11，tray05）

1M context / 每卡 256 seq 的 manifest 上，`stats` 报 `step_ms=7.73`，但 5 秒只跑完
534 步（每步 9.36 ms）——每步丢了 1.6 ms，32k 的 manifest 上没有。pegainfer 的
driver 每步调一次 `Scheduler::metrics()`，`KernScheduler::metrics` 读
`pages_used()` / `pages_total()`，而这两个数当时是把整条页状态数组扫一遍数出来的：
扫描量正比于池子，池子正比于 context（`--capacity 32768` 1132 页/卡空档 0.06 ms，
host 表 146k 页 0.13 ms，HBM 表 1.62M 页 0.91 ms，1M/256 seq 3.37M 页 1.64 ms）。

`Pool` 改成写状态时维护计数（`Inner::set_page` / `set_slot` 是唯一的写入口，
`Tally { live, held }` 随之移动），`total` / `used` / `slots` / `slots_used` /
`anything_held` 全部只读计数，不再扫描。同一 manifest 的门禁：

| | 步/5s | `step_ms` | 每步实际 | 空档 | tok/s |
|---|---|---|---|---|---|
| rows 1，扫描 | 534 | 7.73 | 9.36 | 1.63 | 107 |
| rows 1，计数 | 669 | 7.41 | 7.47 | 0.07 | 134 |
| rows 6，扫描 | 439 | 9.68 | 11.39 | 1.71 | 227 |
| rows 6，计数 | 518 | 9.42 | 9.65 | 0.23 | 274 |

16 条 greedy conc1 的输出两种 rows 都与改前逐字节相同。记录在
`bench_results/2026-09-11-kern-step-host-gap/`。

## 每步的拆分（2026-09-14，tray18）

打点版 kern-serve（每 rank 三个 CUDA event：写输入前、graph 前、graph 后，加各 host 段的
`Instant`），1M/256-seq manifest，16 条 greedy conc1，每步均值（µs）：

| | rows 1 | rows 6 | 32k manifest rows 1 |
|---|---:|---:|---:|
| 整步 | 7360 | 9340 | 7220 |
| GPU graph | 7070 | 9033 | 7063 |
| graph 前的 H2D | 245 | 242 | 88 |
| host 超出 GPU 的部分（stage 31、launch、wake、读回 18） | ~100 | ~110 | ~80 |
| 两步之间（driver、metrics、ledger） | 5 | 4 | 8 |

graph 占 96%，步间已无空档。H2D 里 155 µs 是 `write_input_at` 把整块 staging 拷上去：
page_table 是 256 × 8192 i32 = 8 MiB，每步每卡整块 DMA。改成只拷写入的字节后 H2D
245 → 109 µs，步时 7.36 → 7.22 / 9.34 → 9.21 ms，输出 16/16 逐字节相同；剩下的 ~100 µs
是 12 次 DMA 的串行发射延迟。记录在 `bench_results/2026-09-14-dsv41-step-host-split/`。

## 多卡锁步（E5 第四块，2026-09-03；t=4 门禁 2026-09-04 过）

> 2026-10-09 起一卡一进程（下一节）：`tray.rs` 删了，下面的组语义（行、owner、staging、run）原样搬进
> `ranks.rs` 的 `Ranks`，进程模型与 launch 顺序以下一节为准。

当时 `kern-serve --gpus 0,1,2,3` 一个进程驱四卡：`tray.rs` 持 n 个 `Runtime`，单线程
按序驱动（graph launch 1–3 µs，一线程驱四卡与四线程等价，实测见 multi-gpu.md），每步
所有 rank 跑同一个 program、同一组 var 值、同样的行数；manifest 的 `topology` 里 `tp`
组是"一个 batch 的组"（连续的 rank），其余组（`ep`）跨全部 rank；跨机的 world
不在这里（rendezvous 是 harness 的事）。scheduler 只看得到 `Row` / `Snapshot` /
`Sleeping` / `Rising` 和 `Cell`，看不到 `Runtime`、`Lease` 或某个 rank 的输入。

- **先全部 launch 再逐个 sync**：一步里 n 个 rank 的图先全部 launch（`Runtime::enqueue` /
  `enqueue_captured`），再逐个 `synchronize`——EP 的 dispatch 与组 collective 都在核里等
  peer，先 sync rank 0 就等到核的超时（`CUDA_ERROR_LAUNCH_FAILED`，2026-09-03 E3 第一次
  跑 K3 EP4 就撞上；qwen 单卡 smoke 看不出来）。
- **一行 = tp 组每个成员一份**：owner 卡持 MLA 页 + KDA slot（`Runtime::lease`），
  peer 卡只持 slot（`Runtime::lease_slot`，runtime 新增 slot-only 租约，见 runtime.md）。
  lease / lease_from / fork / checkpoint / retire / park / wake / awake 各在组里每个成员
  做同样的事；**全成或全不成靠构造**：`Row` 只在每个 rank 都给了 `Lease` 之后才存在，
  哪个 rank 说 `Denied`，`?` 提前返回把前面拿到的 drop 掉即回滚（各 rank 的 pool
  独立记账、slot 编号不跨卡比对，所以没有一致性要重建）。park 是唯一不能半途撤销
  的动作（拷贝已入队），runtime 拆成 `room`（找地方，`Room` drop 即退）+ `park`（拷），
  先在每个成员上找齐再拷。`Waking` 提前 drop 会等拷贝落地，所以 wake 的回滚就是 drop。
- **owner**：新行落到"还开着（行数 < `--max-seqs`，per rank）且行数最少、再比页占用最少"的 rank
  （2026-09-04 之前只比页：页数含留着的快照，持有 12k 快照的卡在 conc8 里一行都分不到——`[3, 2, 0, 3]`），
  终身不变，后代（checkpoint、parked、wake 回来的）都跟着它——页在那张卡上，pinned
  块绑在那张卡的 NUMA 节点上。
- **`Prefix` 按组键**：`Prefix<Snapshot, Sleeping>`，键还是 token 哈希链、与卡无关。
- **staging 按 fill 的轴**：fill 或 line 表跨过的第三个 var 就是 ranks 轴（k3 的
  `rows`）；ranks 轴上的 token fill / line 表 / tokens 输出跨组（本卡的行先、再按组序轮到
  其它成员的块，collective 假定的布局），行轴 / 组轴上的 slot / seq_len / 页表是本卡自己
  的行；qwen 的契约（没有 ranks 轴）是 t=1 的特例，同一条代码。**块不等长**（2026-09-04）：
  每个 rank 的块是它自己的行数（至少 1 行 pad），组内块长之和垫到阶梯就是组 batch 的行数
  （`rows` var；多组时各组垫到同一个数，垫在最小的块上；t=1 退化成每卡等块），块的前缀和写进
  `blocks` fill，collective 核按它换算行号（peer_collective.cu）；own 轴的 buffer 仍按最大块的
  bucket b 铺、pad 页每卡一页。`Staged` 是纯数据（每 rank 的 var 与 fill），`items` 把它变成每 rank 一个
  `Item`，`emitted` 从 `Answer` 里取各行的输出。manifest 有 `error` fill 的输出时每步读一次，非零即该步失败。
- **prompt 只走 chunk program**：manifest 必须有一个接受"1 组、行数随喂"的 program，没有就拒绝
  启动（2026-10-12 之前没有它的 decode-only manifest 逐 token 走 step，或走 span，K3 迁出后删了）。
  b>1 走形状包含 `(b, rows)` 的 program 里 `groups` 上界最紧的那个。
- 权重不分 rank 文件：每个 rank 打开同一份 checkpoint（目录 = 全部分片），manifest 的
  bind 按 rank 选行 / 列 / 张量（manifest.md"权重"）。
  `--capacity` / `--host-gib` / `--max-seqs` 都是 per rank。

**t=1 smoke（2026-09-03，tray03 GPU 1，qwen3-4b，`--capacity 65536`）**：conc1 输出与 `kern run`
逐字同；4 路并发两次跑结果一致、与单跑一致（prompt 0 的分歧是 `decode` / `decode_batch`
两个核的近似平局，见 lessons）；prefix 命中三条路径——resident（8972 token 的 prompt 命中
4576）、host 全量 wake（命中 8960 后 12 token prefill）、host 半量 wake（4589 的 prompt 命中
4576）——warm 与 cold 24 token 逐字同；wake 回来的 937 页与 park 时逐页 digest 相等（运行时
`park_wake` 例子加了 `--wake` / `--every-page`，串接每页 checkpoint 的链与部分 wake 都逐字节回）。
`--host-gib 8` 下 c4 的填充就把 host 层打穿（一条 15k 的快照 2.06 GiB），测 wake 用 24。

K3 在 tray 上的 span 实测（93 层 EP4 / EP4×TP4 的 E3 场景、每个 tp 组各一条 run，2026-09-03 – 09-04）
量的是 decode 步里夹 prompt 行的 run；run 与 K3 的生成器、manifest 在 2026-10-12 一起迁出（K3 现在在
[kern-k3-pp-prefill](https://github.com/pegainfer-project/kern-k3-pp-prefill)，走 chunk program），表格见 git 历史。

**没测**（按 CLAUDE.md 的门禁排队）：
1. ~~t=1 qwen3.8-27b（有 slot 的路径）conc1 与 `kern run` 同，K1/K3 门禁数字不变~~（e2e 门禁一节，2026-09-14）；
2. owner-only 页在 t>1 下：mixed 行、prefix 命中（retire → lease_from）、park / wake 之后 warm == cold；
3. 全成或全不成：某一卡 `--host-gib` 故意给小，park 整体退回、四卡 host 占用回到原值。

## 每卡一个进程（2026-10-09，tray03 4×GB300）

多卡也是一卡一进程，外壳与流水线同一种：rank 0 是 head（`--gpu g --listen host:port`：HTTP、scheduler、
**每个 rank 一个 `kern_pool::Pool`**），其余 rank 是 `--gpu g --head host:port --rank q`。`--gpus` 与
`tray.rs` 删了；rank 数是 manifest topology 里非 tp 组的大小（没有就是 tp 的大小，都没有就是 1）。

- **head 记全部的账**。rank 连上 head 发 `Hello { rank, manifest, chunk, chunks, first_slots, tokens, handles }`：
  manifest JSON 的摘要（对不上就拒）、这张卡 arena 的形状（`Runtime::budget`）、导出 buffer 的 fabric handle。
  `Pool::new` 是纯函数，head 拿同一个 `Budget` 建出同样的池，所以 head 的决定就是 rank 的决定，rank 自己
  不记账。head 回 `Table { peers, nccl, host }`：别的 rank 的 handle、每个 NCCL 组一个 id（每个 tp 组一个，
  其余组一个，按组名排序 join）、host 层每卡的单位数（host 页数、host slot 数，head 用 `--host-gib` 在 rank 0 的
  runtime 上按 `host_units` 切一次，head 的 `Host::new` 和每个 rank 的 `reserve_host` 用同一对数）。之后 rank 只收 `Order`、回 `Answer`。
- **决定变成 `Act`**。`Ranks`（`ranks.rs`，纯 host 代码，测试用假 rank）在组里每个成员的池上做
  lease / lease_from / checkpoint / retire / park / wake，每个池给出的 plan 按决定的顺序排进那个 rank 的队列：
  `Remap`、`Zero`（新 slot 清零）、`Copy`（半页 + slot 快照）、`Park`（页与 slot 拷到 host）、`Wake`（拷回）。
  每步给每个 rank 一个 `Order { acts, item, wait }`，rank 在 compute stream 上按序跑 acts 再跑 item，回
  `Answer { tokens, counts, remaps, landed }`：这一步的输出、上次以来落地的 remap、累计落地的 host 拷贝数。
  `Ranks::heard` 用后两个数结清 remap、放出 host 拷贝已落地的 parked checkpoint（页拷完之前不还池）。
- **锁步**。远端的 Order 先发，head 再跑自己那份，最后收全部 Answer——EP dispatch 与 tp collective 在核里等
  peer，规则与原来"先全部 launch 再逐个 sync"相同，只是换成了进程。每个 rank 的 item 是同一个 program、
  同一组 var，fill 是它自己的行（`Staged::items`）。线上是 TCP 帧（u32 长度 + JSON），与流水线的 item 同。
- **语义没动**：行、owner、staging、run、全成或全不成（`Group<Lease>` 等）从 tray.rs 原样搬进 `Ranks`；park 在
  任一成员 host 不够就整体退回 `Snapshot`。动了一处：lease 碰上 `Busy` 而有 park 还在路上时，现在先 `settle`
  等它落地再租，不再为腾地方多 park 一条。流水线的 flag 一并改成 `--listen` / `--head`。
- **代价**：(1) host 权重每进程一份（DSv4.1 的 host 表 × rank 数的 DRAM）；(2) 没有 topology 的 manifest
  不再在多卡上数据并行，要多卡就起多个 kern-serve；(3) 页表与 line 表每步以 JSON 过一次 TCP；(4) `zero_slot`
  的界放宽到 arena 的 `seq_slots`（哪个 slot 由 head 的池定）；(5) `Rising` 在 `awake` 之前 drop 不回滚（拷贝
  已在 rank 的队列上），scheduler 不这么做。

门禁（同一份二进制 `de4d20d`；脚本与日志 `~/bench_results/2026-10-09-ranks-executors/`）：

| 门 | 结果 |
|---|---|
| `cargo test -p kern-serve` | 17 + 4（`tests/ranks.rs`：一行全成或全不成且等在途的 slot；park 在拷贝落地前拿着页；任一成员 host 不够退回快照；每 rank 同 program 同 var、各自的行） |
| qwen3-4b e2e（GPU 3，oracle 同卡） | 10 门全过：conc1 12/12 = `kern run`，命中 16,16,16,0 / 80,80,80,64，park_wake parks 17、wakes 4、host_hits 4，wake_room 1/1——与 09-14 表同 |
| qwen3-4b-dspark e2e | 11 门全过：接受率 burst 24%（2.44 tok/轮）/ 串行 25.1%，rows1 4/4 |
| K3 4 层 EP4×TP4 逐 token（逐步 golden 脚本，随 K3 迁出，四进程） | **35/40 exact、4 excused、1 wrong**（step 14，3.0 ULP 近平局）——与 09-04 单进程同数；四 rank 15 s 起来；40 步每步命中前 i−1 个 token，9 次从 host 层醒 |
| K3 4 层 e2e（`k3-4l-ep4-tp4` / `k3-4l-ep4`，`--max-seqs 6 --chunk 8`，16 token） | 各 6 门全过：命中 0×4 / 38,35,37,31，12 并发全完成，park_wake parks 18、wakes 4、host_hits 4、12/12，wake_room 1/1；warm 对 cold 0/4 照旧只报（见 e2e 一节） |
| K3 4 层 EP4×TP4 ITL p50（96 token，`--max-seqs 8 --chunk 8`，conc 1 / 4 / 8） | 旧单进程（`29050aa`）1.39 / 1.46 / 1.55 ms → 四进程 1.40 / 1.44 / 1.53 ms；每步的页表 / line 表 JSON 过 TCP 不在关键路径上 |

**跨两台 tray（2026-10-09，tray08 + tray09，K3 pruned 93 层 `k3-ep8.json`，`--max-seqs 8 --chunk 64`）**：
rank 只连 head（星形），rank 之间走 fabric handle，没有地址要发现，代码不用改；head 在 tray08 `--listen 0.0.0.0:p`，
rank 1–7 `--head <tray08>:p`，8 进程 60 s 起来。每个 item 每个 rank 一行 `rank item`（issue / done 的墙钟 µs），head 一行
`ranks step`（自己做完后等远端回答的时间），`kern_serve=debug` 打开。脚本与日志 `~/bench_results/2026-10-09-pd-agentx/k4k6/`。

| 门 | 结果 |
|---|---|
| 同一份 manifest 两种 world 逐字节 | `k3_golden` EP4 单 tray 与 EP8 跨 tray 喂同一段 21 token：逐 token 与 span 18 两条路径，每步 logits 与全部 state 的 dump 逐字节相同（60 / 6 个文件） |
| conc1 文本（8 条 prose，64 token） | EP8 跨 tray 三次重启逐 token 相同；对 EP4 单 tray（kern-serve）7/8 相同，第 5 条第 3 个 token 翻：参考三甲 16.09 / 15.98 / 15.94（< 1.5 bf16 ULP），两边各自跨重启、换 rank、换 slot 都稳定。[推断] 空闲 rank 的 pad 行也进 MoE，pad 数随 rank 数变，专家的行数随之变 |
| decode ITL p50 conc 1 / 4 / 8 | EP8 跨 tray 30.1 / 31.2 / 31.2 ms，EP4 单 tray 30.8 / 31.4 / 33.5 ms |
| 每步拆解（conc1 decode，789 步中位数） | 每 rank device 29.8–30.1 ms，head 等远端 0.10 ms（span 步 0.15）；同 tray 四个 rank 的 done 相差 5 µs；两台 tray 的时钟差 5.1 ms（只是钟差，不是线上时间） |

## 复制组：paged state 按位置分给 tp 组（2026-10-09；K3 上卡见"全量 K3 P/D"）

manifest 的 paged state 写了 `shard: {group: "tp", by: "position"}`（`Protocol.shards.positions`），`Ranks` 就把
tp 组当成**一个复制的 batch** 来排（`ranks.rs` 模块文档"Replicated groups"）：

- **Row**：每个成员一个带页的 lease，本地长度 `⌈tokens/w⌉`；owner 是组的首成员，只用来读 Answer。
  `Row::leases()` 按成员序给出各成员的 lease（push 的目标页从这里取）。
- **Layout**：按"组 = 一个单独的 rank"排，所有成员同样的行、同样的顺序；不旋转，没有 `blocks`，没有 ranks 轴。
- **每成员的 fill**：位置 `p` 只有成员 `p % w` 写，slot 是它 lease 的本地位置 `p / w`，其余成员写 pad slot；
  `seq_len` 是本成员持有的位置数 `il.len(pos + per, m)`，可以是 0；页表、line 表都取本成员的 lease。
  token、position、var 全组相同。
- `max_seq_tokens = w × 成员本地上限的最小值`；`ExceedsRow` 的 limit 也乘 w。
- **什么都不留**：`checkpoint` 返回 `Ok(None)`，`retire` 返回 `None`（row 直接还池），前缀树一直是空的，
  所以 `lease_from` / `park` / `wake` 拿不到快照，走不到。
- `Ranks::adopt` 同 `lease`，但不排 `Act::Zero`：slot 由 push 整片写入，Zero 晚于 push 落地会把它清掉。
  两种模式都能用。

门禁：`cargo test -p kern-serve --test ranks` 6/6，其中性质测试在 (w ∈ {1,2,4,8}, 组数 1–2, 行数, 长度) 上随机跑
400 例：每个位置恰好一个成员写进它自己 lease 的 slot、其余成员只写进 pad 页、各成员本地长度之和 = L、
各成员 token 行序相同、各成员页表 / line 表都取自己的 lease、Answer 从组首成员读。

## 全量 K3 P/D：PP8 → TP8/EP8/DCP8（2026-10-09，P tray08+09，D tray17+18，16×GB300）

P = 262144 的 PP8 切分（`--chunk 8192 --max-seqs 128`，带 `--decode`），D = 93 层复制组 manifest
（`gen_stage.py --tp 8 --ranks 8 --dcp --experts 896`），8 个进程跨两台 tray，`--decode-listen`、
`--capacity 2000000`（本地位置，组内 16M token）、`--max-seqs 64`，每卡 263 GB。两边都是宿主机裸跑，
`NCCL_SOCKET_IFNAME=bond0.225`。D 加载 2–4 min（page cache 冷热），D 的第一步约 19 s（NCCL 懒连接）。

- **正确性**（P-oracle：P 对 prompt + D 输出前 i 个 token 的首 token，逐位对 D 的第 i 个）：
  D 单独（prompt 由 decode 步喂入）32 条 128–777 token × 64 步 1979/2048；P/D 同一组 1996/2048；
  P/D 2k / 32k / 128k / 200k 各 4 条 × 64 步 1012/1024。不一致处 D 自己（`k3_step` 同 manifest 教师强制）
  的 top-1/top-2 差：54 处中位 0.10 logit，47 处 ≤ 0.5，3 处 1.4–1.65（与 D2 的 MoE 路由平票同一来源）。
  图模式与 eager 输出逐 token 相同（32 短 + 16 长）。8 个成员的 token 每步相同（现在每步都查）。
- **交接**：TTFT 含推送，与 P 单独冷跑相同（200k 8.7 s，128k 5.3 s，32k 1.8 s，2k 0.46 s）；
  第二个 token 在首 token 之后 11–25 ms，就是 D 的一步。
- **D 每步**（图，短上下文；eager 只慢 0–1 ms）：1 行 22.3 ms、8 行 26.5、16 行 29.7、32 行 35.6–37.3、
  64 行 45 ms；200k 单行 22.6 ms。按 33 ms 的 SLO，短上下文约 24 行。
- **并发**：chat（K3 模板，散文 1–16k token）c8 / c16 / c32 0 错误，ITL p50 26.5 / 29.9 / 36.4 ms；
  stop 由 P 判定后 Abort、客户端中途挂断 8 条之后座位全部释放；多轮 chat 第二、三轮 P 命中
  14781 / 15444 token 后照常交接。
- **P 头转发曾是成批的**：P 头同时是 stage 0，冷长 prompt 在途时它每发一个 item 都要等 stage 1 取走上一个，
  D 的 token 每 0.4–0.8 s 才转给客户端一次（8 条 2k–200k 并发，`pd_client.py run --conc 8`）。
  原因：head 就是 pegainfer 的 `Scheduler`，driver 只在 `step` 返回后提交 ledger，而 `step` 卡在
  `Flow::issued` 的等待里，D 的 `Tokens` 在 head 的事件队列里排到下一步。现在（R1）head 跑在自己的线程上，
  driver 线程上是 `front::Front`：head 把对请求的写入按序作为 note 发给它，D 连接的读线程把 `Tokens`
  直接发进同一个通道，front 自己按 stop 策略转发（`pd::Relays`），head 在发 `Start` 之前把 relay 交过去，
  所以同一通道里 token 总在交接之后。客户端挂断由 front 看 ledger：解码中的它自己 retire 并 `Abort`，
  其余告诉 head 一次。宿主机门禁 `tests/front.rs`（假 head 每 500 ms 才听一次，假 D 每 25 ms 一步，
  5 条 × 40 token）：token 到客户端最晚比 D 发出晚 0.3 ms，间隔 p50 25.0 / max 25.2 ms；把 token 改走
  head 的同一测试是 p50 0 / max 500 ms。第二个 token 要等 head 收到首 token 和各段 `Pushed` 才发
  `Start`（原来 TTST 0.38–0.70 s 即此）：head 等 stage 1 取走上一个 item 时照常听各段和 D（`Flow::ready`，
  取走的消息顺带唤醒 head），只把 front 的提交与挂断留到下一轮。GPU 上未复测。

## e2e 门禁（`tools/e2e`，2026-09-14，tray06 / tray07 / tray09，各 4×GB300）

`python3 tools/e2e/e2e.py` 把一份 kern.toml 的每个 target 过同一组场景（表在 `tools/e2e/README.md`）：
`kern test`、conc1 对 `kern run`、重复命中、turn2（prompt + 答案 + 追问，发 id）warm 对 `kern run` 与对冷
server、12 条并发、流式中途挂断、小池子 + `--host-gib 2` 的 park / wake、刚好装一条 turn2 的池子上的 wake
（`wake_room`）、`--max-seqs 2` 起的 slot 增长、
投机 manifest 的 `--rows 1`。字节一致是门；单 rank target 的分歧再拿 `kern run --prompt-ids <分歧前的
上下文> --rows 1 --probe-dir` 倒那一步的 logits，两个 token 都在 top-1 的 4 个 bf16 ULP 内才放过。
每次 64 个 greedy token（K3 4 层 16 个）。结果与日志：`~/bench_results/2026-09-14-e2e-gate/`。

| target（tray） | rank / rows / checkpoint | conc1 = `kern run` | 命中：重复 / turn2 | warm = `kern run` / = 冷 | 12 并发 · 接受率 | park / wake（小池子） | slot 增长（`--max-seqs 2`） | `--rows 1` = `kern run` |
|---|---|---|---|---|---|---|---|---|
| qwen3-4b（tray06） | 1 / 1 / 每页 | 12/12（`kern test` 位一致） | 16,16,16,0 / 80,80,80,64 | 2/4 + 2 近平局 / 同 | 12/12 · — | parks 16、wakes 5、host_hits 5，答案 11/12 + 1 近平局 | — | — |
| qwen3-4b-dspark（tray06） | 1 / 7 / 每页 | 12/12 | 同上 | 1/4 + 3 近平局 / 同 | 12/12 · 24%（2.47 tok/步） | parks 14、wakes 5 | — | 4/4 |
| qwen3.8-27b（tray06） | 1 / 1 / 请求结束 | 12/12 | 0×4 / 87,83,87,79 | 4/4 / 4/4 | 12/12 · — | parks 19、wakes 4、host_hits 4，12/12 与 4/4 | remaps 15，slot 5 → 20，12/12 与 turn2 4/4 | — |
| qwen3.8-27b-dflash2（tray06） | 1 / 8 / 请求结束 | 12/12 | 0×4 / 87,84,88,0（6 个 `not kept`） | 4/4 / 4/4 | 12/12 · 20%（2.37） | parks 15、wakes 4、host_hits 3 | remaps 9，5 → 14 | 4/4 |
| dsv41-h152（tray09，EP4） | 4 / 6 / 请求结束 | 无 oracle（冷 12 条作基准） | 0×4 / 86,84,0,0（7 个 `not kept`） | — / 4/4（只报） | 12/12（12 条与 conc1 同）· 27%（2.33） | 26 条填满：parks 11、wakes 2、host_hits 2，12/12 与 4/4 | remaps 24，20 → 44（4 卡合计） | — |
| k3-4l-ep4（tray07，EP4，16 token） | 4 / 1 / 请求结束 | 无 oracle | 0×4 / 38,35,37,31 | — / 0/4（只报，见下） | 12/12（2 条与 conc1 同）· — | 17 条填满：parks 13、wakes 4、host_hits 4，12/12 | remaps 36，20 → 56 | — |

2026-09-15 合入前 DSv4.1 EP4 在 tray06 重跑（`--chunk 128 --max-seqs 16`）：第一遍 host 会话在 park_wake 的
turn2 上挂死——wake ↔ park 活锁（`server-host.log` 236 万行 `parked tokens=86`，见上面 toy 那节和 lessons.md）；
wake 改成一次分配、行上限减掉 pad 之后 8 门 3 报全过，新加的 `wake_room` parks=1、命中 86、醒进一条 2 页的租约
（`~/bench_results/2026-09-14-e2e-gate/results/tray06-r7-dsv41-final` → `tray06-r9-dsv41-wake`）。

近平局都有 logits 证据（`results/<tray>/<target>/probe-*/`），例如 qwen3-4b turn2 第 56 个 token：top-4
29.25 / 29.125 / 29.125 / 28.75，server 的在第 2、差 1 ULP。加载：qwen3-4b 10 s、qwen3.8 10 s、DSv4.1 37 s、
K3 4 层 4 s（`--capacity 262144`）。

过程中改了三处，都是 e2e 写不顺才发现的：

- **kern run 只吃文本**：turn2 按文本发，DSv4.1 上答案文本切回去的 id 与生成的 id 不同（4 条里 3 条命中
  0）——不是 cache 的错，是 client 造的 prompt 变了。turn2 改发 id，`kern run --prompt-ids` 让 oracle 跑
  server 跑过的那串。
- **投机轮多算的 checkpoint**：一轮接受的 token 可以越过 `max_tokens` 或 stop（`emit` 截断，state 里已经
  有了），请求结束的快照就以这串多出来的 token 为键——下一轮永远发不出这个前缀，快照白占一个 slot 到被
  淘汰（DSv4.1 4 条 turn2 命中 2 条、dflash2 3 条）。scheduler 现在不留这种快照（`not kept` debug 行），
  只多一个 stop token 的留着（chat template 下一轮就带着它）。e2e 的 turn2 门：命中要么 ≥ 第一轮 prompt
  要么 0，0 的条数 ≤ `not kept` 的条数。
- **scheduler 线程 panic 后端口还开着**：请求全挂到 client 超时（driver 干等了 15 分钟）；现在进程跟着退 101。

近平局的判法也改了一版：投机 manifest 的第 k 步是另一串（`--rows 1` 的 run 在更早的平局上已经分岔），
拿它的 logits 判分歧是错的；要用"prompt + 分歧前双方一致的 token"作 prompt 单独跑一步。

没有 oracle 的 tray target（EP4 没有 `kern run`），换了数值路径的一致性只报不门：K3 4 层 pruned
checkpoint 近乎平局遍地（12 条并发只有 1 条与 conc1 同、warm 对 cold 0/4 在第 0–1 个 token 就分），
同一条路的（重复、abort 后）4/4、1/1 一致照门。K3 4 层用默认预算：256 GiB 切成 131 079 个 2 MiB 块
（页 64 token × 4 层太小，块取最小对象的一半），每 rank map 40–60 s，一次在 rank 3 的 `cuMemSetAccess`
OOM——块大小该随预算长，先记着，e2e 里 K3 显式 `--capacity 262144`。

"没测"清单的第 1 条（t=1 qwen3.8-27b conc1 对 `kern run`、K1/K3 门禁）由这一节覆盖；2、3（t>1 的
owner-only 页、park 的全成或全不成）仍没测。

2026-10-12（tray03，4×GB300）去掉 span、prompt 只走 chunk program、K3 迁出之后重跑 qwen3-4b /
qwen3-4b-dspark / qwen3.8-27b：10 + 11 + 9 门全过（`kern test` PASS，conc1 12/12 与 `kern run` 同，dspark 接受率
并发 24% 对串行 24.9%，`--rows 1` 4/4，分歧都是有 logits 证据的近平局）；结果在
`~/bench_results/2026-10-12-drop-span-e2e/`。

## toy 门禁（`tools/toy`，2026-09-15，tray03 / tray06 GB300）

真模型的 e2e 一轮 20 分钟、要三台 tray，还有近平局要争。`tools/toy` 是一组不是模型的
manifest：整数 kernel，每个 token 槽开头存 64 个位置相关的 mark、其余到槽尾每个字都是
头部的函数，下一个 token 由序列所有位置的头字按位置旋转后的和决定（有 line 的再加上
line 的折叠，line 的尾部同样是折叠值的函数）；尾部哪个字对不上就把和毒掉，页序换了和
也变。Python 参考（`tools/toy/model.py`）逐 token 精确，只算头字。kern-serve、kern-pool、
runtime 走的是同一条路——它们本来就不认识模型。

```
python3 tools/e2e/e2e.py --config target/toy/kern.toml --reference tools/toy/model.py --gpus 0,1
```

把同一组场景过一遍，每个门都精确，包括真模型上只能"报不门"的：并发 12 条对 conc1、warm 对
cold、醒来的对 cold；第二张卡上 `kern run` 自己也被 reference 门住（`run_equals_reference`），
五种形状的每个 kernel 都过 `kern test` 的 A/B（256 对 128 线程块的两个 cubin）。

| target | 形状 | 命中：重复 / turn2 | park / wake / host_hit | slot 增长 | 投机 |
|---|---|---|---|---|---|
| toy-paged | 4 KiB/token、页 16 | 96,80,96,64 / 144,160,112,176 | 13 / 4 / 4 | — | — |
| toy-stateful | + 1 MiB/seq line | 0×4 / 117,219,160,107 | 14 / 4 / 4 | 5→20（15 remap） | — |
| toy-spec | 4 行 round | 同 paged / 144,160,112,160 | 13 / 4 / 4 | — | 50% 串行、50% 并发（2.49/轮）、rows1 4/4 |
| toy-stateful-spec | line + round | 0×4 / 117,220,0,0（req-2/3/14/15 `not kept`） | 15 / 2 / 2 | 5→15（10 remap） | 50% / 50%（2.5/轮） |
| toy-big | 64 KiB/token、页 64、281 GiB state | 64×4 / 128,128,64,128 | 12 / 4 / 4 | — | — |

2026-09-15 tray06，每条 256 token、12 条 prompt，五个 target 全部通过，整轮 3 分 09 秒，两张卡。
结果与日志：`~/bench_results/2026-09-15-toy-e2e/`（`results/tray06-r8`；qwen3-4b 同日同机 9 门全过，`tray06-r9-qwen3-4b`）。
同日 tray03 加上 `wake_room` 再跑一轮（`results/tray03-r14-wake-green`）：五个 target 全过，带 state 的两个
parks=1、命中 117、醒进租约，纯 KV 的三个 parks=0 原地续上；qwen3-4b / -dspark 10 门与 11 门全过（`tray03-r14-qwen`）。

第一版（tray03，`results/tray03-r4`）的表里五个 target 也"全过"，但并发 12 条那时只报不门，
toy-stateful 的 12 条并发答案其实**全错**（0/12 与 conc1 同）：`gen.py` 给 prefill 和 decode
的 `fold` 用的是同一个调用，rows 绑在 `tokens` 变量上，decode 的 tokens 是整批的行数，每个 group
把别人的行折进了自己的 line。设计评审（Opus）读出来的，把并发在精确 oracle 下改成门就复现
（`results/tray06-r5-red`，`FAIL: concurrent`），decode 改绑字面量 1 就过。同一轮 TDD 补的门：
近平局只放过一个 token、之后从 server 选的 token 续算继续比（qwen3-4b 上 repeat 第 46 个 token
两个 logit 相等 21.5，放过后其余 18 个逐字同）；`turn2_hit` 的 0 命中要配自己第一轮请求的
`not kept`，不再数条数；投机接受率对串行自己（同样 12 条 prompt，按 steps 加权）的 0.9 倍而不是固定 20%，toy 还要落在 40–60；
单 rank target 没 oracle 是 FAIL 不是只报；一个 target 都没跑退出码非零。这些裁决是纯函数，
`tools/e2e/test_e2e.py` 在 CI 里跑。driver 自己的两处竞态也是这轮撞出来的：小池子的填充停在
"日志里有 4 条 `parked`"，读日志的时机决定 turn2 要的 checkpoint park 没 park（一轮 wakes=0），
改成填到 turn2 命中的那几个长度都 `parked` 为止；`serving` 在 pegainfer 绑端口之前打出，看到就连
偶发 connection refused，ready 改为端口应答。

toy 抓到的几条：

- master 把 state 块的上限提到 64 MiB（bb50189，DSv4.1 的 836k 页从 13 s 降到 0.5 s）之后，
  `--capacity 2832`（177 页 × 64 KiB）变成了整整一个块的 1023 页——`--capacity <tokens>` 的契约破了，
  小池子什么都不 park。真模型的页都比 64 MiB 大得多，e2e 看不见；toy 的 4 KiB/token 一跑就翻。
  `Pool::new` 现在拿到要的 token 数，页数以它为上限，块的尾巴空着。
- `kern run` 与 kern-serve 对"权重是文件时 tokenizer / eos 在哪"的规则不一致，master 同日已统一
  （`kern_run::checkpoint_dir`）。
- toy-big 加上 `-ref` 之后 `kern test` 炸在 `position past the lease`：perf 的 prefill 扫描点取到
  manifest 的 `tokens.max`（8192），租约只有 `--capacity` 的 4096 个位置；真模型的 `tokens.max`
  从没超过 4096。扫描点现在以租约为界。
- wake ↔ park 活锁（DSv4.1 EP4 的 host 会话先撞到，toy-stateful 上 `wake_room` 场景 30 秒复现，
  608 万行 `parked`）：命中 parked 条目的请求先醒成快照再租，两次分配；现在 `Host::restore`
  一次醒进请求自己的租约（`pool.md`、lessons.md）。同一场景顺手抓到第二条：`max_request_tokens`
  没减 pad 那一页，恰好占满池子的请求既不被拒也永远坐不下；`Ranks::max_seq_tokens` 现在以 pad
  之外的页为界，超过的在租之前就按 ContextLength 拒。
- kern-pool 两处（评审读出、集成测试复现）：`Prefix::evict` park 成功那条路把"为腾地方丢掉的
  parked 条目数"扔了，scheduler 的 `host_evictions` 少计；`lookup` 给一条一页都没共享到的旁路
  候选盖时间戳，一次什么都没命中的查询把最冷的条目变热、淘汰错人。

toy 不测 kernel 数值和性能。多 rank 的 toy 只有 DCP 一种：

`toy-dcp`（2026-10-09，tray07 GPU 2,3）：同一个整数函数，页按位置发给 `tp` 组、16 头的 line 按头切，
各成员的部分和经 `nccl_allgather_i32` 收齐后按 u64 相加，走 kern-serve 的复制组（每卡一个进程，
decode 图里带 NCCL）。e2e 的 oracle 是单卡 twin（`toy-dcp-twin`，同一个函数的 w = 1 版本）的
`kern run`；twin 的 91 条答案先和 `model.py` 对过（91/91，309 s，每条约 3.4 s 一个进程），它自己
那套 toy 场景 12 门全过。w = 2 的 8 门全部逐 token 相同：conc1 12/12、short（L = 1..3，
max_tokens 1/2/64）9/9、pages（组页边界前后、十二页、比 chunk 长）6/6、concurrent 12/12、
many_rows 64/64、kept_nothing（cached 全 0）4/4、abort。结果：`~/bench_results/2026-10-09-pd-agentx/results/k7/w2-r3`。
同一轮抓到两条：默认池把设备只留 1 GiB，NCCL communicator 装不下（join 时 `ncclUnhandledCudaError`），
dealt 会话因此显式 `--capacity 262144`（默认池该给 NCCL 留地方，还没改）；`kern run` 首 token 就是 eos 时
不打 `generated ids`，e2e 当成失败（已修）。

`toy-dcp8`（同日，tray03 GPU 0–3 + tray10 GPU 0–3，rank 0 在 tray03，tray10 的四个 rank 经 ssh 起）：
NCCL 一个 8 rank 的 MNNVL communicator（nNodes 1、NVLS 可用），8 门全部逐 token 相同：conc1 12/12、
short（L = 1..9）27/27、pages 6/6、concurrent 12/12、many_rows 64/64、kept_nothing、abort；twin 109/109
对 `model.py`（352 s）。第一次跑全部 rank 都连上 head 之后卡在 NCCL join（10 分钟无输出）：节点上还有
docker0 / usb0，bootstrap 的网卡要钉 `NCCL_SOCKET_IFNAME=bond0.225`（e2e 把 `NCCL_*` 带给远端 rank）。
结果：`~/bench_results/2026-10-09-pd-agentx/results/k7/w8-r1`。

## 流水线（prefill-only PP v1，2026-10-07，tray07 GB300）

一段一个进程：`kern cut` 切出的 stage 0 是 head（`--listen host:port`：HTTP、scheduler、唯一的
`Pool`），其余段 `--head host:port`（2026-10-09 之前叫 `--pp-listen` / `--pp-head`）。握手：每段连上 head 发 `Hello { stage, cut, pages, port, arrivals }`
（`arrivals` = 本段 host 不填的 input / inout，即上一段要拷进来的，export handle 的 hex；`port` 是上一段
连本段的端口），head 收齐 S−1 个、按最少的页数装自己的池，回 `Table { pages, next }`（下一段的地址
与 arrivals）；之后同一条 TCP（u32 长度 + JSON）上 head 按发射顺序给每段发 `Item { program, vars, rows }`
（fill 与页表的值，页号全流水共用），末段对吐 token 的 item 回 token。
激活走 NVLink，信号走 host：每段把下一段的 arrivals 映射进来（`Runtime::map`），issue 完 item k 后等
下一段报告 k−1 已取走，再在 compute stream 上 `Runtime::push` 同名 buffer、打 `Runtime::mark`；mark 落地
时 watch 线程告诉下一段"k 到了"、上一段"k 取走了"。下一段的 host 不等这句话：它先在 compute stream 上
放一个"到达数 ≥ k+1"的门（`Runtime::hold`：一个在 pinned 计数上自旋的 kernel，runtime 自带的几行
PTX），接着就把 k 的 launch 全发出去，收到"到了"时写计数（`Gate::open`）放行——等数据的是设备，host 的
launch 不在关键路径上（qwen3 的 prefill 不走图，218 个 launch 要 0.6 ms，等到了再发会让 conc4 慢
3–30%；cuBLAS 的 eager launch 排在没开的门后面时会在调用里自旋到门开，所以 host 最多领先一个
item）。mark 也不走 driver：设备在 mark 处用一个 kernel 把序号写进 pinned 内存，watch 线程只读内存；
末段的 token 同样 issue 完就 `Runtime::fetch`（stream 上拷进 pinned），watch 线程在 mark 落地后读出来
答给 head；fault 由发射线程每个 item `Runtime::fault`（`cuStreamQuery`，不阻塞）问一次。为什么等待
设备的线程一个 driver 调用都不能做，见 lessons.md 2026-10-08。这换掉了设备上的邮箱核与 `pp.*` op：manifest 里只剩 stage 的签名。没有序号、stop、fault
消息：任何错误进程退出，连接断了对端退出；下一段 60 s 不报"取走"视为死，上一段死了表现为 head 退出。
只做 prefill：prompt 除末 token 按 chunk 走 chunk program，末 token 走一行的 step，首 token 即结束
（`max_tokens` 视为 1）。item 的编号是它在 TCP 上的次序，各段数得一样。每段 issue 完一个 item 就打一个
`Runtime::mark`（event），旁边一个线程等它，落地时记一行 `pp item`：stage、item、rows、`issued_us`、
`done_us`（墙钟）。host 上谁也不为它停：段的周期 = 相邻 done 之差；按 item 号把各段的行拼起来，段 s 开始
item k = max(本段 issue k、本段 done k−1、上游 done k)，于是每段每 item 的空等与占用都有了
（`tools/pp_timeline.py`，带 `--gantt`；跨 tray 的段用 `--skew` 扣两台机器的时钟差）。占用里含等下游取走的
时间，要拆出纯计算就拿该段单独跑这个形状的时间（`kern bench` 段 manifest）去比。
设备上的 fault 在该 item 的 mark 上冒出来，进程退出。下表的 busy / wait 列是当时
设备上 globaltimer 打点量的（`clock` fill，已删：它让输出不再是 forward 的纯函数）。

门禁（qwen3-4b，切在 `l18.qkv_proj` 前，2 段）：
- 只切不分进程：`pp_chain` 各段依次跑 vs 整份，7 条 prompt（1–2049 token），chunk 512 × 8 步、
  chunk 64 × 16 步（2049 token 32 个 chunk），每步 logits 与 token **逐位同**。
- 两进程 serve：7 条 gate prompt 首 token 与单卡 kern-serve 全同；杀末段 → head 立即退出；冻住末段
  （SIGSTOP）→ head 60 s 后退出（等"取走"超时），恢复的末段随后退出。

扫描（chunk 512，`--max-seqs 16`，每格 conc1 4 条 / conc4 16 条不同 prompt；TTFT = max_tokens=1 的
请求延迟；busy/wait 是每 item 均值）：

| conc | prompt | chunks | TTFT unsplit ms | TTFT pp ms | speedup | stage | chunk busy ms | step busy ms | recv_wait ms/item | send_wait ms/item |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| c1 | 513 | 1 | 8.1 | 8.5 | 0.95x | 0 | 2.99 | 1.32 | - | 0.00 |
| | | | | |  | 1 | 2.90 | 1.46 | 1.48 | - |
| c1 | 2049 | 4 | 38.2 | 25.3 | 1.51x | 0 | 4.27 | 1.46 | - | 0.00 |
| | | | | |  | 1 | 4.21 | 1.60 | 1.16 | - |
| c1 | 8193 | 16 | 313.4 | 173.8 | 1.80x | 0 | 9.57 | 1.99 | - | 0.00 |
| | | | | |  | 1 | 9.52 | 2.13 | 1.01 | - |
| c4 | 513 | 1 | 29.7 | 17.6 | 1.68x | 0 | 2.96 | 1.32 | - | 0.00 |
| | | | | |  | 1 | 2.91 | 1.46 | 0.09 | - |
| c4 | 2049 | 4 | 139.6 | 78.5 | 1.78x | 0 | 4.26 | 1.46 | - | 0.19 |
| | | | | |  | 1 | 4.20 | 1.60 | 0.30 | - |
| c4 | 8193 | 16 | 1194.0 | 664.5 | 1.80x | 0 | 9.57 | 1.99 | - | 0.61 |
| | | | | |  | 1 | 9.52 | 2.13 | 0.73 | - |

单请求内 chunk 之间就流水起来（stage 0 算第 k+1 块时 stage 1 算第 k 块），所以长 prompt conc1 也
接近 2×；1 块的 prompt 没有可重叠的东西，略慢（邮箱 + 两段 launch）。KV 每层一个 state 之后（同日，
`examples/qwen3-4b.json` 重生成，`kern test qwen3-4b` 位一致、`kern run` 文本同），每段只带自己的
18 层：不给 `--capacity` 时每段 122,993 → 245,760 页（2.0×）；切点、gate、扫描数字与上表一致
（`~/bench_results/2026-10-07-kern-pp-v1/perlayer/`）。v1 的限制：只认纯 paged 的 manifest；
chunk program 自己吐 token 的模型拒切；上游那份 box 是不用的占位。

传输搬到外壳之后（2026-10-08，`~/bench_results/2026-10-08-pp-inout/`）：门禁同上（tray08：8 条 gate
prompt 首 token 与单卡全同、498 item 两侧一致、杀 / 冻末段同前，head 退出后末段也退出）；K3 PP8
容量（tray12+17，c16、rate inf、200 条 2k–32k）44.3k tok/s、2.47 req/s（邮箱 44.9–46.8k），每段
period 175–179 ms 由算力定，host 不在关键路径上。qwen3 扫描中位 TTFT ms（邮箱 → 现在）：conc1
8.3 / 25.2 / 172.6 → 8.7 / 25.3 / 172.4，conc4 17.6 / 79.6 / 664.7 → 22.7 / 90.1 / 677.4——conc4 还差
5–13 ms：host 等下一段"取走"再 push 是一次 TCP 往返，而 cuBLAS 的 launch 排在没开的门后会阻塞，门一开
host 还得把剩下的 launch 发完，设备追上 host 就空转；要关上这段得弄清 cuBLAS 在 launch 里等什么。

落地缓冲（2026-10-12，GB300 单机 4 卡，`~/bench_results/2026-10-12-pp-buffers/`）：上一段原来直接 push
进下一段计算用的那份 buffer（`x` / `residual` 在中间段是 inout，下一段 push k−1 时还在读它），所以要等下一段
**做完并推走** k−1 才能推 k；下游一个大 item 会把上游每一段都按住。K3 c96 的 `pp item` 回放（1P2D 第一轮）
算出这一项占 1.15×。现在每个到达 buffer 有 `LANDINGS` 份导出的副本（`Runtime::landing`）：item k 推进副本
k % LANDINGS，下一段在 hold 之后先 `Runtime::land` 拷进计算用的 buffer、打 mark，mark 落地即报"取走"；
上一段只在下一段落后 `LANDINGS` 个 item 时才等。多一次片内拷贝（K3 8192 行的 hidden 约 117 MB，几十 µs，对
~227 ms 的 item 可忽略，估算）。复现用 Qwen3-4B 切 4 段（l9 / l18 / l27 前），chunk 2048（manifest 上限），
600 条 125–8000 token 的散文 prompt 闭环压满（K3 的长度分布缩 4 倍，item 大小混合比例不变），c32、150 s：

| 落地副本 | prompt tok/s | TTFT p50 / p90 s | 末段计算占比 |
|---|---:|---:|---:|
| 无（旧） | 90,480 | 0.97 / 1.17 | 0.49 |
| 1 | 106,395 | 0.83 / 1.01 | 0.58 |
| 2（取用） | 108,042 | 0.81 / 0.98 | 0.59 |
| 4 | 108,128 | 0.81 / 0.98 | 0.59 |

旧版本的回放模型（`scripts/pipe_sim.py`，K3 那份改成 4 段）预测 1.77–1.81×、实测 1.82–1.83× 于"不限缓冲"，
所以这一项在 Qwen 上比 K3 重：2048 行的 chunk 和 1 行的 step 差得更悬殊。两份副本之后四段都只用到 ~59%，剩下
的空闲在 head：两个 item 之间 host 要 ~4.5 ms 备下一个（stage 0 的间隔 p90 7.4 ms），K3 的 item 大两个数量级，
这一项在那里小。门禁：首 token 与旧版本逐位比，64 条串行 64/64 同，c32 并发 600 条 600/600 同。

head 的前缀缓存（2026-10-09，tray03 GPU 0+1，`~/bench_results/2026-10-09-pp-head-prefix/`）：head 的池是
整条流水唯一的账，索引也只在 head（`pp/cache.rs`，与单进程 scheduler 同一个 `Prefix`）。prompt 先查索引，
命中就 `lease_from`（整页共享、半页与 state slot 拷给新 lease），从命中处开始喂；最后一批行发出去时
lease 直接 `retire` 进索引（零拷贝，不等末段回 token：之后读这些页的 item 都排在写它们的后面）；带
recurrent state 的 manifest 改在 `history_tokens` 处停下做一次 checkpoint、结尾不留。另在它（无 `history_tokens` 时是结尾）前 4096 处停一下做 snapshot，与它一样腾地方（它随即被这一轮 supersede，腾地方时排在前面）：重试或改写的请求在上一个请求末尾前不远处分叉（AgentX 256k trace 同一条流内：p50 448、p75 4.7k token），没有它就只能从更早的轮次续，warmup 只回放过上一轮的流则什么都命中不了（2026-10-09 kern P/D c32 计时窗口里 9 条首请求 0 命中，占输入 1.6%）。实测（2026-10-09，GB300，P = PP8 两节点、host 层每段 128 GiB，D = TP8/EP8/DCP8 两节点，AgentX 256k，c24 后接 c32 各 900 s）：命中率 c24 91.8% → 94.1%、c32 91.8% → 94.1%（同数据集 SGLang 93.9%），c32 TTFT p90 3.91 → 3.06 s、p99 48.4 → 21.8 s；只用空闲 slot 的版本 c32 只到 92.5%（slot 常满，snapshot 245 次跳过、18 次拿到）。池满（`Busy`）按
`Prefix` 的分档丢一个条目再试（有 host 层时 park，见下）。拷贝由 head 的池决定、在 head 上当场执行，同时排进下一个
`Item` 的 `acts`（2026-10-09 之前是 `copy_pages` / `copy_slots` / `zero_slots`），各段在写 fill 与 launch 之前按序执行
（`Runtime::copy`），于是每段 stream 上的次序与 head 相同。段不记账，跟不了 remap：head 只在有空 slot 时取 slot（否则先丢条目），
池真 remap 了 head 直接退出——所以带 state 的流水能缓存的条目只有初始的 `--max-seqs` + 2 个 slot 减去在跑的。
stats 行报 `resident_hits` / `resident_hit_tokens` / `checkpoints` / `evictions`，命中也作为请求的 cached
tokens 回给客户端。门禁（qwen3-4b 切在 218，chunk 512，与单卡 kern-serve 冷跑比首 token）：`--capacity
65536` 15/15 同——7 条 gate prompt 冷跑；1500 / 2049 两条再发命中 1488 / 2048；四个第二轮（1500+200、
300+213、12+40、60+40）命中 1500 / 300 / 12 / 60，对照单卡冷跑。`--capacity 4096`（256 页，一条 2049
占 129 页）9/9 同，7 次淘汰、一次 2048 命中。对照组：把段上的拷贝去掉，12+40 那条首 token 变了（14/15），
门禁确实看得见半页拷贝。

head 的发射与次序（2026-10-09，tray06+07 full K3 PP8，262144 切分，`--max-seqs 128`，
`~/bench_results/2026-10-09-pd-agentx/p/`）：head 只在两次发射之间收答复、admit，每次 send 又等自己的设备，
所以领先多少就是决定得多晚。满 item 至多在途"段数"个（stage 0 前排一个），不满的只在少于段数时发
（`pack::goes_out`）——之前满 item 不设限，8 条 100–200k 冷 prompt 在第 39 s 一起返回，其间不 admit。
次序是剩余行最少者先（`pack::turn`）：命中长前缀的 prompt 只剩几千行，冷的剩整条，oldest-first 下排在
冷 prompt 后的命中要等完整个 backlog（200k 冷约 8.7 s 一条）。不饿死：被 `patience`（= 段数）个 item
跳过的序列排到所有被跳过更少的前面，被跳过最多的那个总放得下（自己的 context 在界内），所以有行的
序列每 `patience + n` 个 item 至少进一个（`tests/pack.rs` 对全部小到达序列逐条验证；去掉 patience 时
它报出饿死）。context 放不下的序列跳过而不是停下。混合负载（每会话 20 轮、每轮 +1.6k、思考 1 s，
预热后第 4 s 起每 3 s 来一条 200k 冷 prompt，共 4 条；TTFT ms）：

| 负载 | head | 命中 p50 / p90（全部） | 冷 prompt 在途时的命中 p50 / p90 | 冷 TTFT s |
|---|---|---:|---:|---|
| 16 会话 60–200k | FIFO（只有发射上限） | 1,020 / 7,932 | 22,620 / 23,428 | 8.9 / 15.3 / 19.8 / 24.2 |
| | 剩余最少先，chunk 8192 | 3,166 / 4,740 | 3,683 / 4,834 | 32.5 / 46.3 / 56.1 / 59.0 |
| | 剩余最少先，`--chunk 4096` | 1,948 / 2,397 | 1,996 / 2,404 | 38.1 / 55.0 / 55.4 / 55.9 |
| 8 会话 100–200k | FIFO | 947 / 8,275 | 21,871 / 22,493 | 8.7 / 14.9 / 19.4 / 23.5 |
| | 剩余最少先，chunk 8192 | 2,356 / 3,361 | 2,947 / 3,523 | 14.5 / 21.5 / 28.8 / 34.8 |
| | 剩余最少先，`--chunk 4096` | 1,754 / 2,114 | 1,784 / 2,138 | 15.8 / 26.0 / 33.8 / 40.5 |

FIFO 下冷 prompt 一来，会话整个卡在它后面 20 多秒；剩余最少先之后命中不再排队，剩下的是流水线本身的延迟：
命中的 item 走 8 段，每段的周期由同时在流水里的冷 item 定（8k@100–200k 每段 260–363 ms），
所以冷 prompt 在途时命中约 8 × 300 ms 加上等发射。`--chunk 4096` 把冷 item 减半，命中回到 2 s 左右，
代价是冷容量（2k–32k、c16、rate inf）44.3k → 38.1k tok/s（−14%）。chunk 8192 时冷容量 44.3k tok/s、
2.51 req/s（只有发射上限时 44.9k），TTFT p50/p90/p99 6.30/9.28/11.58 s（7.30/8.38 的 p90/p99 变长：长的
冷 prompt 让给短的）；frozen 192 首 token 与 pack7 184/192、与 run7 187/192（同一份 manifest 两次 run 之间
本来就是 180–187），45.3k tok/s。
`--max-seqs 128`：131 个 slot、216,632 页（13.86M token），中间段 272.8 GB。卡住的是 slot 不是页：
每轮命中留下一个带 state 的条目，上一轮的条目还在同一条路径上占着 slot；16 会话跑完 kv 10%、slot 130/131，
`--chunk 4096` 那次有 1 轮整条 miss（225.6k 冷重算 27.8 s）——淘汰按"链的叶子先走"，对带 state 的链正好反了。

带 state 的条目改成分档淘汰（2026-10-09，`~/bench_results/2026-10-09-pd-agentx/p2/`，pool.md「带 state 的
条目分档」）：每轮只在 history 末尾留一份，会话走过去的旧轮先走，再轮到闲得最久的会话。离线
`agentx_replay --pp-head`（Claude Code 轨迹 393 会话 98,827 请求，64 token 页、13.86M token、131 slot 不 carve，
请求在 prompt 末尾留快照）命中率：

| 并发 | LRU（之前） | 分档 | slot 无限（5000） |
|---|---:|---:|---:|
| 32 | 92.7% | **94.8%** | 97.0% |
| 64 | 90.6% | **94.3%** | 97.0% |
| 128 | 78.9% | **85.0%** | 96.8% |

c32 extend p99 429k → 257k token。先试过"新条目直接取代被续上的条目"：c32 92.4%，比 LRU 还低——
被取代的旧轮正是后面改写 / 分叉的请求要的，所以旧轮留到要腾地方时才走。full K3 PP8（tray06+07，
`--max-seqs 128`）32 会话 × 10 轮 chat（system 20–80k，每轮 +1.6k user、回放 300 token 的回复）：
后 9 轮 288 次全部命中、0 次整条 miss，命中率 96.8%；slot 用到 130 是设计如此（旧轮占着空位），
其中最新一轮的条目 ≈ 会话数（stats 行 `checkpoints − superseded` 37 = 32 会话 + 5 条 sanity）。
同一负载旧 head（K + R 两份、LRU）也是 0 miss、96.8%：它的 R 正好是下一轮的前缀，命中长 10 token
（K3 的 generation prompt 在下一轮的历史里原样出现）；这个负载每轮都很快，LRU 链的最新一条还没变冷
就被续上了，分档的差别要在 slot 紧、会话慢的时候才看得见（上表）。

head 的 host 层（2026-10-09，tray06+07，`~/bench_results/2026-10-09-pd-agentx/p3/`）：`--host-gib` 在流水线下
**按段算**（每个段进程一份 pinned 块，NUMA 本地），只在 head 上给。head 收齐各段 `Hello.host`（每段一个 host 页、
一个 host slot 的字节数），按最大的那段把 `--host-gib` 换成 host 单位数（`host_split`），放进 `Table.host` 发给
各段；每段按自己的单位字节数 reserve，没有 state 的段 slot 那一份是 0 字节。账只在 head 的 `Host` 上记一次。
`Busy` 时 `Prefix` 按 P2 的分档挑条目，有 host 层就整条 park（`Runtime::park_now`：先在 head 上把页和 slot 拷出，
再把同一份 `Copies` 作为 `Act::Park` 排进下一个 `Item`），host 满了才丢 host 上最冷的；命中 Parked 条目时
`wake_now` 只恢复要用的那些页和 slot 进新 lease（`Act::Wake`）。所有段的拷贝都排在**计算 stream**上、在本 item
的 fill 与 launch 之前，于是 park 一定读到写完的页、wake 一定先于读它的 launch，正确性不依赖任何计时；P→D
push 读的是 lease 的页，同一 stream 上排在 wake 之后。各段每 GiB 的 DRAM：K3 PP8 一段 64 GiB = 248,679 host 页
+ 234 host slot（reserve 3.1 s）；一台 4 段 × 128 GiB = 512 GiB，留一半给 OS / page cache，建议 64–128。

门禁：toy（toy-dcp-twin 2 段，`p3/scripts/gate.py`）`toy-host` 12/12 + 并发 6/6 exact（16 次 park、12 次
host 命中），`toy-tiny`（host 小到要丢）12/12 + 6/6（12 次 host 丢弃），`toy-pd`（P 的 host 层 → D toy-dcp w=2，
64 token）12/12。qwen3-4b 2 段 `--capacity 4096`，对照两台单卡冷跑：host 2 GiB 22/22 同（8 次 2048 token 唤醒、
21 次 park），host 0.4 GiB 13/13（1 次唤醒、7 次 host 丢弃）；对照组让段跳过 Park / Wake：17/22、12/13，
错的全是被唤醒的 prompt。full K3 PP8 `--max-seqs 8 --capacity 450000 --host-gib 64`：A/B/C 三条 200k
依次冷跑把彼此挤到 host，A+E+E2 / B+E / C+E 三次都从 host 醒（`host_hits=3`，601,600 token），首 token
与 `--capacity 1000000` 冷跑 4/4 同；TTFT 冷 8.8 s → 唤醒 0.91–0.92 s（resident 命中 0.96 s）。一条 200k
会话每段醒 750 MB（3125 页 + 58.6 MB slot），stage 0 同为 1600 行的 item 97.5 → 116 ms，即每段 ~18 ms、
~42 GB/s——拷贝是每个池化数组一条按页步长的 `copy_2d`，离 pinned 带宽差 4 倍。

离线 `agentx_replay --pp-head`（同上表的轨迹与 K3 段形状：3456 B/token、slot 58.6 MB、13.86M token、131 slot），
host 按段：

| host / 段 | c32 | c64 | c128 | c128 extend p99 | 拷出（整条轨迹，每段） |
|---|---:|---:|---:|---:|---:|
| 0 | 94.8% | 94.3% | 85.0% | 504k | — |
| 64 GiB | 96.7% | 96.1% | 88.6% | 417k | 13.8 TiB |
| 128 GiB | **97.0%** | **96.5%** | **90.7%** | 341k | 9.4 TiB |
| 256 GiB | 97.3% | 96.9% | 92.3% | 279k | 6.4 TiB |

c32/c64 的 host 层把命中率推到 slot 无限的水平（97.0%）；c128 还差 6 点。上表是五分之一给 slot 的旧切法，
K3 一条会话的 slot 58.6 MB 抵 265 页，slot 先用光，128 GiB 下 c32 峰值只用到 52 GiB。现在按 `--host-turn`
（默认 8192 token 的页配一个 slot，K3 上 slot 占 67%，`pool.md`）切，128 GiB c24/c32/c64 到 97.6 / 97.4 / 97.2%，
c128 89.9%。另试过"被续上的旧轮直接丢、只 park 最新一轮"：c128 好（92.0%）、c32/c64 差
（95.9 / 95.6%），park 一律走 P2 的次序。

## 没做（按需要加）

span 长的预算 policy（K5 D2：按稠密 / attention 预算定 c，现在是 `--chunk` 上限）、抢占 / 动态页分配、
真采样（temperature/top-p 作为 manifest 内的 `sample` op；投机下是 rejection sampling）、logprobs / echo、
bs 2–16 的 split-KV decode、步间 host 空转（token 反馈进图）。
