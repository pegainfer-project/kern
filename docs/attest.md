# kern test：一次 kernel 替换的证据

`kern test` 拿两份 manifest——A（参考，**默认正确**）和 B（候选）——
产出一份 attestation：换掉的东西在哪、每个 cut 数值上等不等价、随机输入
下是否一致、快了多少。老 program 就是 oracle；manifest 里**没有阈值**。

```bash
./target/release/kern test qwen3-4b --out attestation.json   # A/B/kernels/weights 来自 kern.toml 的 target
./target/release/kern test \                                 # 或全用 flag（没有 kern.toml 时）
  --reference examples/qwen3-4b.json --manifest examples/qwen3-4b-silu-mined.json \
  --kernels kernels --weights weights/qwen3-4b-decode.safetensors --out attestation.json
                                  # --diff-only 只看静态 diff；--no-perf 跳过计时
                                  # --no-graph-step / --no-sweep 关掉 TPOT graph 计时 / prefill 扫描
```

`kern.toml`（`crates/kern-run/src/config.rs` 有完整说明）里一个 target 就是
`manifest`（B）+ `reference`（A，用户自己拷一份信得过的）+ `kernels` +
`weights`。`kern test` 一次只测一个 target：只有一个就不用点名，有多个必须
点名；target 没有 `reference` 直接报错——没有 A 就没有测试。要跑全部，
shell 里 for 一下。target 的名字用户随便起，kern 不解释。

报告走 stdout，一行一个事实：行首是段名（`diff` / `tap` / `local` /
`logits` / `noise` / `fuzz` / `perf` / `sweep` / `roofline`），事实之间用
` · `，两边用 `a → b`，段耗时在行尾；**最后一行以 verdict 开头**（`PASS` /
`FAIL` / `INCONCLUSIVE`，同退出码 0 / 1 / 2），`tail -1 | cut -d' ' -f1`
就是结论。相同的只给计数（`900/900 bit-identical`），不同的才点名，最坏的
在前、最多 8 条，其余进 `--out`。`--json` 改成在末尾打一个 JSON 对象（同样
的段、同样的字段，`jq .verdict.pass`、`jq .perf.steps[0].graph_ms` 直接
取）。`--out` 写存档：`summary` 就是那个对象，`detail` 是每一条有差异的
比较（cut、buffer、差异元素数、ulp）——相同的不存，PASS 的存档只有几 KB。
stderr 默认安静（`RUST_LOG=debug` 看 runtime 装载）。

`examples/qwen3-4b-silu-mined.json` 是自带的 A/B fixture：和
`qwen3-4b.json` 唯一的区别是 `silu_mul` 的 impl 从 HF hub 的
`kernels-community/activation` 包换回挖矿得到的 vLLM cubin，接口与全部
call 一字不动——纯 impl 替换。两份 manifest 共用一个 `--kernels`
目录：module 按 sha256 解析，目录里放着两边各自钉的版本即可（`tools/
extract_kernels.sh` 对 A、B 各跑一次，只增不减）。`--capacity` 会向下
对齐到 manifest 的页单位，fuzz 的 `slot_mapping` 不会落进半页。

## 设计：tap 一次，之后全是 cut 级

整条流水线只有 **tap** 这一步跑完整 program；之后每一段都只重放 cut：
把快照里的 frontier 输入写回去、`run_range` 跑那几个 call、读写出的
buffer。成本随 cut 大小走，不随模型走——TP8 的大 MoE 换一个 kernel，
harness 付的是"模型装载一次 + 一次 prefill/decode + N × 几个 call"。

**端到端只看 logits，不生成。** 每个 cut bit 相同 ⇒ 整体必相同，不需要
再证；cut 有差异时，oracle 是 B 自由跑同一 workload 后每一步的 `logits`
（manifest 里的 buffer，读出来就行）：Δ 以 logits 自身尺度的 ulp 计，
argmax 的翻转按 A 自己的 margin 分成 near-tie 与真翻转。裁不了的（没有
logits 的 program、A 自己不确定）报 INCONCLUSIVE（退出码 2）。

## 五段

1. **DIFF（静态）**：逐 kernel 比接口（`params`）和实现（`impl`），分
   interface / impl / added / removed；逐 program 用 LCS 对齐两边的
   call 列表（变了的 op 两边永不对齐），切成 Same / Changed 段，
   Changed 段就是 **cut**。每个 cut 由数据流图给出 frontier：读了哪些
   外部 buffer、写了哪些——`a+b → c` 的中间 buffer 不在 frontier 上，自动
   略过。两边 frontier 不等会标 ⚠（表示不是 cut 内替换）。
2. **TAP（seeded workload）**：workload 完全由 `(seed, manifest, 选项)`
   决定，谁跑都一样、也不依赖 A 的数值：token 随机（vocab 从 `token_ids`
   的 `index_into` 域来），prefill 长度一半概率在 `[1, capacity − steps]`
   均匀抽、一半概率抽**结构边界**（一页 ±1、一个 chunk ±1、`tokens` max、
   上限——kernel 出错的地方），chunk 从 {`tokens` max, 512, 一页, 随机}
   抽，decode 步数在 `[steps/2, steps]` 抽（默认 32），decode 喂的 token
   也随机（不是 A 的 argmax，否则换个参考序列就变）。`--prompt` 可以换成
   真文本 prefill，`--prefill/--chunk/--decode-steps` 可以钉死。默认 seed
   固定是为了两次运行可比；覆盖靠换 seed（`--seed`），抓到问题的 seed
   写进报告钉成回归。
   A、B lockstep：Same 段两边各自跑；**每个 program run 之前 B 的 state 整
   体拷成 A 的**，每个 Changed 段跑之前从 A 读 frontier 输入（按当前
   var 值取活跃前缀）**并写进 B**，跑完读 A 的输出做参考、比 B 写出的
   buffer。所以每一行 cut 结果都是 **cut-local**：B 拿 A 的输入、A 的
   state 跑这一刀，差多少就是这一刀自己的事，不混前面层漂移下来的误差
   （早先不注入时，c7 那种 state 差会让下游每个 buffer 都显示 47/48 cuts
   differ、几十万 ulp，看不出哪一刀是根）。
   lockstep 结束后 **B 从零 state 自由跑一遍同样的 workload**，什么都不
   注入——表末的 `end-to-end` 行（output 类 buffer + 每个 state 全量字节
   差）就是调用方真正会拿到的东西。**state 按 cut 的 write-set 比**：A 跑前后各读一次 state，差异的字节区间就是这个 cut 的
   write-set（pre-image）；先把 A 的 pre-image 写进 B 再跑 B，然后只在
   write-set 上比 A、B 的 post-image，另报 B 在 write-set 之外写了多少
   字节。整个 state 不能拿来比——其余字节是别的层的历史，B 的历史又是
   B 自己的。第一个 prefill chunk 和 decode step 的每个 cut 都存成快照
   （输入 + 参考输出 + 参考 state + pre-image），后面的段全靠它。最后比
   output 类 buffer。
2b. **LOGITS（端到端判据）**：lockstep 里每个 run 之后读 A 的 `logits*`
   buffer（manifest 里本来就有，`logits`、spec 的 `logits_blk`），自由跑
   里读 B 的，逐 run（`logits_blk` 逐行）比：argmax 是否一致、A 的
   top-1 − top-2 margin、max |Δ|、KL(A‖B)。**尺度是 A 的 top logit 的
   ulp**（Δ / ulp(top1)）——逐元素 ulp 在这里没意义，近零的 logit 在
   hidden 差 1 ulp 时能动几千 bf16 ulp，却决定不了任何 token；c9 的求和序
   改写就是"31185 ulp"实为 0.156 = top logit 的 1.25 ulp。token 翻了但
   A 自己的 margin ≤ Δ 的叫 **near-tie**——A 自己就站在平局上，不算 B
   的错；margin 明显大于 Δ 的翻转才是 **FAIL**（第一个真正的 FAIL 档）。
3. **NOISE FLOOR**：每个快照写回 A，重跑 A 自己的 cut，和参考输出比。
   带 inout state 的 cut 不幂等（重放一次 conv 窗口再移一位、SSM 再递推
   一步），所以**每次重放先把 pre-image 写回，跑完把 A 的 post-image 写
   回**——A、B 皆然。只有 write-set 还不够：cut **读**的 slice 里有它没改
   的字节（step 0 恰好没变的 SSM 项），后面的 step 会改它们；所以每个被
   快照的 run 留一份 run 之前的全量 state 镜像（prefill chunk 0 = 全零，
   decode step 0 = A prefill 之后的 state，各一份），重放序列切换 run 时
   两边整体写回——各层 write-set 不相交，这份镜像就是该 run 每一刀的
   pre-state。这样参考自己
   逐位可复现，band 为零；仍不 clean 才是 A 真的不确定（atomics 之类），
   此时 B 按这条带子判（`--no-noise` 跳过）。早先没有复位时 A 自比差
   上百 MB，band 无限宽，B 在 state 上的真错误全躲在带内。
4. **FUZZ（围绕 tap 扰动）**：对每个快照，浮点 frontier 输入在 tap 到的
   值上扰动，轮流用 jitter（×(1+N(0,1)/64)，动低位尾数）/ noise（加 10%
   自身 rms 的高斯噪声）/ scale（整体 ×¼…×4）/ shuffle（按行打乱位置）/
   resample（自举，保边缘分布毁结构）/ outliers（1% 元素 ×16）；**整数
   输入一律保留 tap 值**（序列边界、索引、页表是结构不是值——随机的
   `cu_seqlens_q` 是没有调用方会产生的 workload，序列外的行 manifest 没
   定义，A 不碰 B 全写，早先 GDN 核在这上面"6143/6144 differ · 33k ulp"
   全是这么来的）。不再从 N(0,1) 合成：核只在它被造出来的分布里测。两边
   重放同一个 cut，
   比写出的 buffer；写出的 buffer 若声明了 domain，则检查每个元素落在域内
   （后置条件；A 违反说明参考本身有问题）。写出的 state 也比（A、B 全
   量，两边此时起点相同）。B 崩溃（IMA）直接 FAIL。
5. **PERF**：每个变了的 program 整步 eager 跑 N 次，逐 call event 计时
   取最小——同一份数据既给**整步**（Σ 全部）又给 **Σ cuts**（换掉的那
   块）。表里 `B measured` 旁边就是 **`B derived`** = `A − Σcut_A +
   Σcut_B`（只看 cut 就能推出的整步预估），实测与推导的差就是换 kernel
   带来的 launch 间隙 / L2 交互效应。decode 再两边各捕获 graph 跑 100 次
   取中位 = **TPOT**，同样给实测和推导两列（`--no-graph-step` 关）。prefill 扫 `tokens ∈ {1, 16, 128,
   512, 2048, 4096, max} ∩ [1, max]`（`--no-sweep` 只跑 tap 的 chunk 长），
   token id 按 `token_ids` 的 domain 随机、结构输入由 driver 填。对变了
   的 kernel 用 manifest 声明的读写 buffer 字节数算 roofline 下界——不需
   要任何模型知识（state 不透明，标为 "+ opaque state"）。
   整步计时不是 cut 级的，但成本只是"program 跑 N 次"，和 tap 同量级。

## driver

manifest 不规定 program 叫什么；能把真实 workload 喂进去的是 **driver**
（`crates/kern-run/src/lib.rs` 的 `Caller`：知道 `token_ids` /
`positions` / `slot_mapping` 怎么填、prefill 按 chunk 推位置、`tokens`
是 prefill 的尺寸符号）。它是模型家族契约，`kern run` 与 `kern test`
共用，目前只有 qwen3 一份。attest 遍历 manifest 的 programs；变了但
driver 不会 stage 的 program 在 TAP 里标红、判 INCONCLUSIVE。

## 判定

按顺序取第一条命中的：

- `FAIL`（退出码 1）：fuzz 下 B 崩溃 / 产出越出声明域。
- `FAIL`：端到端某一步 B 换了 argmax，而 A 的 top-1 − top-2 margin 明显
  大于 logits 的 Δ（不是 near-tie）；A 自己不确定（noise floor 脏）时不
  下这个结论。
- `INCONCLUSIVE`（退出码 2）：变了的 program driver 喂不了（覆盖缺口，
  再好的 logits 也只说明被喂到的那些）。
- `PASS: bit-identical`：每个 cut 在真实和扰动输入下逐 bit 相同。
- `PASS: value-identical`：只差 ±0 符号位（silu 类 kernel 常见）。
- `PASS: logits bit-identical`：cut 有差，端到端每一步 logits 逐 bit 相同。
- `PASS: logit evidence`：端到端 logits 的 max |Δ| ≤ `--logit-ulp`（默认
  4）个"logits 自身尺度的 ulp"（Δ / ulp(max |logit|)），argmax 一致或
  只有 near-tie 翻转（报告逐条列出）。合法的舍入序改写（c9 的 SSM 求和
  序：state 差 280 KB、logits 差 1 ulp）进这一档。
- `PASS: within noise floor`：cut 差异不超过 A 自己重跑的差异（且 fuzz
  值相同）。
- `INCONCLUSIVE`：其余——logits 动得超过阈值但没翻 argmax、或没有 logits
  可比。报告里有 Δ、near-tie 数、KL，交给上层判断。

`--out` 写完整 JSON（每个 cut 每个 buffer 的 n_diff / max ulp / max |Δ| /
nan / signed-zero、端到端每步 logits 的 Δ / argmax / margin / KL、noise
带、每轮 fuzz、逐 kernel roofline、sweep 曲线）。

## fixture 实测（GB300，2026-08-31）

HF hub `activation` 包 → 挖矿 vLLM cubin（packed 变体）：tap 72 个 cut
（prefill 36 + decode 36）全部 bit 相同、next_token 相同，快照 35.7 MB；
noise floor clean；fuzz 六种分布 bit 相同、`edge` 下只差 signed zero；
perf：decode eager 整步 4.231 → 4.167 ms，Σ36 cuts 306 → 232 µs
（−24%），推导 4.157 vs 实测 4.167（差 10 µs），graph TPOT 2.606 →
2.549 ms（384 → 392 tok/s）；prefill eager 整步从 tokens=1 的 −1.9% 到
2048 的 −3.4%（27.4 → 26.5 ms），Σ cuts −26% … −44%。roofline 列直观
展示 bs=1 下 58 KB 的 silu 只到峰值带宽的 0.1%——launch 主导，这正是
bs>1 才轮到 kernel 本身说话的原因。全程 7 s，其中 2.4 s 是装两个
runtime，PERF 1.8 s。

## 位置

- 静态 diff、frontier、快照、fuzz、比较全在
  `crates/kern-run/src/attest.rs`（caller 契约在
  `crates/kern-run/src/lib.rs`，和 `kern run` 共用）。
- runtime 只加了不在服务路径上的原语：`run_range`（按 call 区间
  eager 执行）、`read_buffer_prefix` / `write_buffer` / `read_state`（任意
  kind）、`time_range`（区间内逐 call event 计时）、`time_captured`
  （graph 中位数）、`check_domain`。元素编解码在
  `kern_runtime::values`（bf16/f16/f32/fp8e4m3/整数 ↔ f64，ulp 距离）。
