# tools/e2e：kern 的 GPU 门禁，按消费方的用法跑

`e2e.py` 把一份 kern.toml 里的每个 target 过一遍同一组场景。它只用用户有的东西：
`kern`（`run` / `test` / `server`）、`kern-serve`、OpenAI 端点、server 的日志行。
场景像客户端一样写；一个场景写不出来，说明接口缺东西，先补接口。

```
python3 tools/e2e/e2e.py --gpus 0,1,2,3 --out results/ [--config kern.toml] [--targets a b]
    [--server-flags "--chunk 128 --max-seqs 16"] [--max-tokens 64] [--kern ..] [--kern-serve ..]
```

在有 GPU 的机器上跑。target 的 artifact 不在这台机器上就 skip（不算 fail）；
需要的 rank 数超过 `--gpus` 也 skip；一个 target 都没跑退出码非零。单 rank 的
target 用 `--gpus` 里的第二张卡跑 `kern run` 做 oracle（server 占满第一张卡），
没给第二张卡就是 `conc1_equals_run` FAIL，不会悄悄降成只报；tray target（EP4）没有
`kern run`，它冷启动的 conc1 答案就是 warm 答案的 oracle。`--reference <module.py>`
换成一个 Python 模块的 `generate(ids, max_tokens, manifest)` 做 oracle：模型自己的
算术，精确，任何 rank 数都有，分歧一律不放过，这时第二张卡上的 `kern run` 自己也
被 reference 门住（`run_equals_reference`）——`tools/toy` 的 manifest 就这么门
（README 在那里）：

```
python3 tools/e2e/e2e.py --config target/toy/kern.toml --reference tools/toy/model.py --gpus 0,1 --out results/
```

| 场景 | 门 | 看什么 |
|---|---|---|
| `kern_test` | 有 `reference` 的 target `kern test` 退出码 0 | logit-ulp 规则（docs/test.md） |
| `conc1_equals_run` | 12 条 prompt 逐条 serve == oracle（`kern run --prompt-ids` 或 `--reference`），token id 逐字同 | `return_token_ids` |
| `repeat_hit` | 同一条 prompt 再来一次：命中长度 = 分页规则允许的（每页 checkpoint：整页、去掉最后一个 token；request end：0），答案同冷 | `usage.prompt_tokens_details.cached_tokens` |
| `turn2_hit` / `turn2_warm_equals_run` | prompt 的 id + 答案的 id + 追问的 id，紧接着问（发 id 不发文本：答案文本不一定切回原来的 id）：命中 ≥ 第一轮 prompt，答案 == `kern run --prompt-ids` | 同上 |
| `concurrent` | 12 条同时发：都结束；精确 oracle 下答案 == conc1（`kern run` 做 oracle 时相同条数只报：batch 组成会换归约序） | |
| `spec_acceptance` | spec target 并发那个 5 s 窗口的 `accept_pct` ≥ 串行阶段（同样 12 条 prompt，各窗口按 steps 加权）的 0.9 倍——batch 只换归约序，不换一条序列接受什么；精确 oracle 给了区间（toy：40–60）串行的还要落在区间里 | stats 行 |
| `abort` | 流式请求读 3 个 chunk 就挂断，之后的请求答案同冷 | |
| `turn2_warm_equals_cold` | 新起一个 server 冷答 turn2 == 上面的 warm 答案 | |
| `park_wake` | 每 rank `--capacity` 只够它那份（4 条最长请求）、`--host-gib 2`、`--max-seqs 4`：12 条 prompt（答案同冷）加编号变体填到 server 记下 4 次 `parked`（上限 48 × ranks 条，有 state 的 manifest 首批 slot 的块也能当页用，光靠页数算不准）→ parks ≥ 1；再问 turn2 → wakes ≥ 1、host_hits ≥ 1，答案同冷 | stats 行 `parks/wakes/host_hits`，`admitted ... woken=true`，`parked` 行 |
| `slot_growth` | 带循环状态的 manifest 用 `--max-seqs 2` 起（首批 slot 只有几个）、池子默认：2 × 首批 slot 数 + 4 条结束后 `remaps` ≥ 1、`slots` 比 `scheduler ready` 时多，答案同冷，turn2 的命中与答案同第一个 session | stats 行 |
| `rows1_equals_run` | 投机 manifest `--rows 1` == `kern run --rows 1`；与整块 rows 的答案相同条数只报 | |

字节一致是所有门的标准。不一致时报第一个分歧 token；单 rank target 会再拿 `kern run
--prompt-ids <prompt + 分歧前的 token> --rows 1 --probe-dir` 把分歧那一步的 logits 倒出来
（这一步本身的上下文，不是另一条 run 的第 k 步——投机的第 k 步已经是另一串）：两个 token
都离 top-1 ≤ 4 个 bf16 ULP（按 top logit 自己的 ulp，`math.frexp`）才算近平局
（docs/test.md 的 logit 规则），否则就是 bug。近平局只放过那一个 token：之后拿
`kern run` 从 server 选的那个 token 续算，剩下的继续逐字比，一条答案最多放过 3 个。
tray target 没有这个出口：同一条路的一致性（重复、abort 后）照门，换了数值路径的
（warm 对 cold、醒来的对 cold）只报不门——K3 4 层的 checkpoint 近乎平局遍地（并发
12 条只有 1 条与 conc1 同）。有 state 的 target，投机轮多算到 max_tokens 之外的请求
server 不留 checkpoint（`not kept` 行带 `request=req-N`，N 是到达序），turn2 的命中要么
≥ 第一轮 prompt，要么 0 且它第一轮的请求（conc1 和 repeat 的）都 `not kept`；不能全是 0。
`test_e2e.py` 把这些裁决当纯函数测（CI 跑，不要 GPU）。

`--out` 下每个 target 一个目录：`report.json`（facts、每项结论、stats 计数）、
`server-*.log`（每个 session 一份，`RUST_LOG=kern_serve=debug`）、`oracle.json`
（`kern run` 的答案缓存）、`kern-test.log`、`probe-*/`（近平局证据）。`summary.md`
是一张表。CI 没有 GPU：结果连日期和机器写进 docs/serve.md。

仓库外的 target（DSv4.1 EP4、Kimi-K3 4 层 EP4）在 bench 目录自己的 kern.toml 里，
路径不进仓库。
