# 进程之间的协议

kern-serve 一个 GPU 一个进程。进程之间有三条边界：锁步组、流水线、P/D。这份文档把
每条边界当成设计对象来写，而不只是一段传输：上面跑哪些消息，每份状态归谁，顺序靠什么
保证，哪些不变量必须成立，以及用什么手段检查它们。

PR #50 的 review 找出的问题几乎都出在这里。原因是同一个：进程内的函数调用被序列化成
消息，调用方和被调方之间原本由类型和调用顺序保证的东西，过了边界就都没了。具体表现有
四种：一个请求的 payload 顺带捎上了连接级的状态；两个线程写同一条连接；对端发来的计划
不做校验就执行；等待没有期限。

## 三条边界

| 边界 | 两端 | 消息 | 状态的 owner | 顺序 |
|---|---|---|---|---|
| 锁步组 | ranks head（rank 0，`Ranks`）↔ 每个 rank 的 `Executor` | `Hello` / `Table`，然后 `Order { acts, item, wait }` ↔ `Answer { tokens, counts, remaps, landed }` | 每个 rank 的 pool 账（页、slot、remap、host）全在 head；rank 的 runtime 只执行 | 每个 rank 一条 TCP，一问一答；组内 kernel 靠 collective 互等 |
| 流水线 | pp head（stage 0）↔ stage 1..n | `Job::Item { acts, rows, vars }` / `Job::Push` ↔ `Report::Tokens` / `Report::Pushed` | 整条流水线只有一个 pool，在 head；stage 执行 acts 和 push | 每个 stage 一条 TCP；stream 顺序 |
| P/D | P head、P front ↔ D head；P 的每个 stage → D 的显存 | `Ask { Reserve, Start, Abort }` ↔ `Tell { Pools, Placed, Grant, Busy, Room, Refuse, Tokens }`；`Push` 经镜像写进 D 的 pool | D 的 seat 和 placement 在 D head；交接阶段在 P head（`Tiers`）；解码中的 relay 在 P front（`Clients`） | 一条 TCP，P 侧只有 head 一个写者，front 的 abort 经 head 转发；placement 按 epoch 在连接上先于用到它的 `Grant` 告知，镜像只接受下一个 epoch，seat 的 epoch 超过镜像的 push 被拒 |

## 不变量

下面每一条都写成检查器可以逐步判定的形式。

**安全**

- **S1 写目标。** 一次 push 写入 D 上的对象 o（页或 slot）时，o 被该请求持有；写入方的
  镜像在 o 覆盖的每个 chunk 位置上，与 D 的 pool 此刻的 placement 相同。
  kern-pool 保证被持有的对象不会被 remap 挪走（`chunks.rs` 模块文档），所以这一条
  只需要在写的那一刻检查。
- **S2 不死于正常事件。** 对端发来的消息序列只要合法，任何进程都不能因此退出。合法的
  情况包括客户端离开、P 拒掉 prompt、D 回 `Busy`，以及这些事件之间的任意交错。fail-fast
  针对的是真正的故障（对端死亡、计划违反自身事实），不是协议里的正常分支。
- **S3 seat。** push 在途时，它写的 seat 既不被释放，也不被 `Abort`。
- **S4 终结。** 每个请求恰好结束一次：finished、rejected 或 retired。finished 的 token
  就是 greedy 链，到 stop 或 `max_tokens` 为止，中间不重不漏。
- **S5 执行者校验。** 执行者拿自己掌握的事实校验对端发来的计划：位置到 chunk 是单射；
  host 页号小于预留数；页号和 slot 号在 arena 之内。矛盾就返回 `Api` 错误，而且在任何
  issue 之前返回。

**活性**

- **L1 死亡传播。** 组里任一成员死亡，其余成员都在期限内退出，客户端的流也在期限内结束。
- **L2 静止无残留。** 没有请求时，所有 seat、lease、relay、在途 push 都归零。

## 检查手段

| 手段 | 在哪 | 查什么 |
|---|---|---|
| 确定性仿真 | `crates/kern-serve/tests/sim.rs`（CPU） | S1–S4、L2。真实的 `Tiers`、`Clients`、`Decoder` 和 pool，每个线程是一个 actor，每条连接是一个 FIFO，调度由种子决定；反例会被收缩成最短调度，并逐步讲清每一步做了什么 |
| 执行者的 GPU 测试 | `crates/kern-runtime/tests/driven.rs`（`--ignored`，要 GPU） | S5 |
| 故障注入 | 多 rank 的 kern-serve 上，空闲或解码中 `SIGKILL` 一个 rank，计时 | L1 |

head 和 front 的决策是纯的状态机：`pd::Tiers` 返回 `(tier, Out)`，`front::Clients`
返回 `Write`，`PpHead` 和 `Front` 只负责执行，不做判断。仿真跑的就是这两份真代码，
手写的只剩 head admission 外围 P 自己的 pool 和 runtime（prompt 超过上限就拒，否则
lease）。head 只在听到事件之后才轮到一次 turn，和它的主循环一致，所以 admission 卡住
时报出来的是残留的 seat 或没结束的客户端，而不是 livelock。

检查器本身也要验：把真代码改坏，仿真必须抓到。去掉 front 对解码中离开客户端的
`Retire`，报 `request 1: None (gone true)`；让 `Tiers::release` 什么都不释放，报
`held at the end: [3]`。

穷举 BFS 还做不了：它要求状态是 `Clone + Eq + Hash`，`Decoder` 和 pool 还不满足，
所以目前只能做"随机调度 + 收缩"。

## 在现有代码上的复现（2026-10-12）

在动生产代码之前，检查手段必须先在现有代码上把 review 找出的四个问题都抓出来。

| # | 问题 | 结果 | 现在 |
|---|---|---|---|
| 1 | placement 增量挂在单个请求的 `Grant` 上（`decoder.rs` `told[q]` 在发送时就推进），请求被 release 或 abort 时它跟着丢失 | 仿真，种子 1，94 步：Grant 3 带着新 carve 出来的 slot 3 和两个 rank 的 placement；P 拒掉 3（prompt 太长）→ `release` → `Abort`；Grant 6 拿到同一个 slot 3，不带 placement；stage 0 透过旧镜像写 slot 3。违反 S1 | 已修：placement 属于连接，按 epoch 先于 `Grant` 告知。leave 0 和 3 各 1000 个种子全部干净；200 个种子里 196 / 145 个有 push 在 epoch 1 写入 |
| 2 | head 先发 `Note::Decode`，再锁 wire 发 `Start`（`pp/head.rs` `apply`）；front 在两者之间看到客户端离开，`Abort` 先上 wire | 仿真，种子 50，43 步：第 27 步 head 发 Decode → 第 28 步 front 处理 → 第 29 步 front sweep 发 `Abort 1` → 第 30 步 head 发 `Start 1` → D：`request 1 starts with no seat`，tier 退出。违反 S2 | 已修：head 是唯一写者，front 的 abort 经 head 转发。3000 个种子全部干净，其中 103 个有 front abort 解码中的请求；让 front 直接写 D 的变异体在种子 16 被抓到 |
| 3 | `Runtime::remap` 原样执行线上来的 `Remap`；`host_out` / `host_in` 跳过 `check_host` | GPU：一个把位置 0 的 chunk 也映射到位置 1 的 remap 返回 `Ok`，写位置 0、读位置 1 得到同样的字节；一个写进 host 第 1 页（只预留了 1 页）的 park 返回 `Ok`，覆盖的是 slot 区。违反 S5 | 已修：被驱动的 runtime 记着之前的 remap 留下的 placement，新 remap 先经 `Placement::apply`（kern-pool，纯函数）校验；`host_out` / `host_in` 走 `check_host`。tray07 上 `driven.rs` 3/3 通过：两份坏计划都返回 `Api`，twin pool 规划的 3 次 remap 照常执行。kern-pool 的属性测试在 20000 份随机 placement 和计划上与顺序 mapper 模型一致 |
| 4 | 一个 rank 在 issue 前就死了，同组成员在 collective 里永远等下去 | toy-dcp，2 rank：rank 1 空闲时 `SIGKILL`，再发一个请求；head 60 s 内没有退出，没有一行错误日志，客户端一个字节都没收到。违反 L1。对照：解码中 kill 时，rank 1 那一步的 kernel 已经发出，head 1.0 s 后读到 EOF，退出码 1 | 已修：tables 发出之后，每个成员用一个线程盯着它和对端的连接（head 盯每个 rank，rank 盯 head），对端一关就直接退出进程，不 unwind，因为 drop runtime 也要等设备。tray07 toy-dcp：rank 1 空闲或解码中被 kill，head 都在 0.8 s 内退出；head 空闲或解码中被 kill，rank 1 分别在 0.8 s 和 1.6 s 内退出。没有覆盖的情况：对端机器掉电、没有 FIN；流水线的 head↔stage 还没复现 |

对照组：不 remap、没有客户端离开、没有 prompt 被拒，300 个种子全部干净结束。所以上面
抓到的问题是对应配置自己的。

原有的 `tests/pd.rs` 随机调度测试为什么没抓到 1 和 2：

- 它把 head 和 front 合成了一个 actor，`apply` 的一串 Out 是原子执行的，两个写者之间的
  交错根本不存在。
- 它的 D 从来没有真正挪过 chunk。fixture 的 slot 只有 24 B，一个 256 B 的 chunk 能装
  十个，"carve 一个 slot"只是一次 access 授权，placement 一个位置都不变。所以
  "grant 带着两个 rank 的新 placement"那条断言，带的其实是没变的 placement。
- 它没有写目标的检查：push 只核对了长度和 seat 数。

顺带发现：kern-pool 的 `Arena::users` 是 `u16`。一个 chunk 里的对象超过 65535 个（比如
2 MiB 的 chunk 装 16 B 的页）时，debug 构建会 overflow panic，release 构建会回绕。

## 目标设计

- 每份状态只有一个决策者，写成纯的 `step(state, event) -> (state, effects)`。决策者之间
  只传事件，决策者交给执行者的只有计划。执行者先拿自己的事实校验整份计划，再 issue。
- 每个方向一条有序通道，只有一个写者。消息带序号；帧有长度上限；`Hello` 里带版本和
  manifest 的 sha256。
- 跨进程寻址用物理的 `(chunk, offset)`，持有期间 chunk 被钉住，不再靠镜像 placement。
- 异步完成是 ticket，所有 ack 到齐才提交。
- 每个等待都有期限；成员死亡，整组在期限内退出（fail-fast）。
- 每条边界的具体做法：
  - 锁步组：类型化的 `GroupOrder`；常驻的发送线程；`Answer` 带 placement 摘要。
  - 流水线：head 用 `Pool` 加上和 stage 相同的 `apply(plan)`。
  - P/D：每个 tier 一个 session，由一个单写者的 handoff actor 负责，front 的 abort 也经过它。

这些改动的门禁就是上面四行复现：每一行都要从"抓到"变成"抓不到"，同时对照组保持干净。
