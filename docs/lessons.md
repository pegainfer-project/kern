# 教训（按时间倒序，每条：现象 → 原因 → 以后怎么做）

设计写在 runtime.md / serve.md / multi-gpu.md 里；这里只记那些"不写下来下次还会
再踩一遍"的事，以及它们落到了哪条规则上。

## 2026-09-03，v4 投机轮

**"旧路径与 plain 逐字同、新路径不同"不等于新路径有 bug。** dspark 的 verify 从 8 行改
7 行后，96 token 的散文在第 19 个 token 处分叉；先用 `kern run --probe-dir` dump plain 在
该位的 logits：top-1/top-2 差 0.125（bf16 一个 ulp）。cuBLAS 按 m 选核，m=8 与 m=7 各翻
一边，旧路径与 plain 的"逐字同"本来就是运气。规则：**分叉先读 margin 再定性**
（CLAUDE.md 门禁那句），margin ≤ 几个 ulp 就记进 docs 当核噪声，不追。

**压测 prompt 的散文要够长、够多样，报接受率要看窗口。** 512 token 的中文 docs 段落有
几条掉进重复循环（"用 `--show-raw` 显示原始数据。"×N），那条的接受率 71%；wall-clock 的
tok/s 按 `usage.completion_tokens` 算，不按 `max_tokens` 算。规则：
**报数字注明 prompt 集与窗口（running 多少）**，接受率异常高先看输出是不是循环。

## 2026-09-03，M1 MLA decode 核换成 CuTe DSL 预编译核

**LD_PRELOAD 钩 `cuLaunchKernelEx` / `cuTensorMapEncodeTiled` 抓不到 CuTe DSL 的启动。**
DSL 的 JIT host 代码调 `libcute_dsl_runtime.so` 自己导出的 `_cudaLaunchKernelEx` /
`_cuTensorMapEncodeTiled`，里面静态链接 cudart，驱动函数走 export table，没有一次按名字的
符号解析。规则：**抓一个闭源/JIT 栈的 ABI，先 `nm -D` 看它自己导出了哪层包装，钩最靠近
调用方的那层**；参数字节用 `cuFuncGetParamInfo` 按核的真实布局切，不猜。

**描述符字节对不上不等于参数错。** DSL 自己 encode 的 CUtensorMap 比驱动 encode 多两个 bit、
box 字段错一字节，穷举 dtype/swizzle/L2/OOB 也复现不出。不追字节：用驱动 encode 的描述符
独立起同一 cubin，输出逐位一致就是证明。规则：**反解 ABI 的验收是"独立 launcher 复现输出"，
不是字节相等。**

**split-KV 切多了反而慢。** B=16 × 13k：每行 4 个 split 55 µs，6 个 67 µs，32 个 87 µs——
cluster 数一超过一波（nsm/2）就排第二波，归约还按 (行 × split) 读 256 KB。规则：**split 规划
按"一波 cluster"摊 tile，B=1 长上下文才要 >32 的 split**；工作区大小跟 split 上限走，别默认开大。

**生成器的常量折叠只认顶层 `{"param": i}`。** `pack` 字段和 `tensormap` 里的 `param`
不改写，折掉一个标量参之后引用全错位，verifier 报"param #12 out of range"才发现。规则：
**manifest 里凡是能引用接口参的位置，normalize 的每个 pass 都得覆盖**（`fold_constants` 现在
遍历 pack 字段与 tensormap）。

## 2026-09-03，E5 tray 级 checkpoint 的 smoke

**随机 token 的 prompt 不能做 warm == cold 的相等测试。** 15k 个均匀随机 token id 的 prompt，
prefix 命中后 8 token 的短 prefill 与 30 块整 prefill 给出不同的首 token（' 1000…' 对
'Okay, I need…'），resident 命中和 host wake 一样，HEAD 一样，逐页 digest 证明 wake 回来的字节
与 park 时相等——花了两小时在 room / park / wake 里找一个不存在的 bug。随机上下文下每个
logit 都是近似平局，两条数值路径（chunk 形状不同）必翻。规则：**相等测试用自然语言 prompt
（docs/*.md 拼起来就够），随机 id 只用来占页**；见到 warm ≠ cold 先换 prompt 再读代码。

**字节问题用字节证明。** 这次真正切开问题的是给每页算 digest 对比 park 与 wake，五分钟；
读代码找了两小时。规则：**数据搬运路径上的疑点，先在两端 hash，再谈逻辑。**

## 2026-09-03，E5 tray 内 TP

**`asm volatile` 上的 `"memory"` clobber 把整条链路串行化。** LL collective 的 16 B
槽用内联 PTX 收发，加了 memory clobber 之后每个 pack 都是 load → 3 store → load，
B=16 的 allreduce 只有链路的三分之一。去掉 clobber（flag 随数据走，顺序由 volatile
访问自身保证）、一次轮询所有源，25 µs。规则：**collective 先测只发不收**，发的时间
就是链路上限，其余都是回读协议的代价，别在错的一半上优化。

**弱 load 看不到 peer 的写。** `ld.global.cg` 轮询 peer 写的槽永远"没到"，超时；
`ld.volatile`（sys scope）才是内存模型承诺的路径。gpu-scope 在 GB300 上也能看到且
更快，但没有承诺，不用。

**tp 组的每个 rank 必须跑一模一样的 launch 序列。** k3_golden 的参考跑（每个 distinct
feed 的复制批、stray 的 from-scratch 批）和 fork 都要在组里每个 rank 上做同样的次数，
不然 collective 在某个 rank 上等一个不会来的 epoch，超时报 err。规则：harness 里
"这个 rank 要不要跑"永远由组共有的量决定（feeds 数、fork 步），不看 rank 号。

**每卡持有 tray 批每一行的 state，slot 数就得跟行界走。** KDA 按头切后一张卡的 state
是 4B 行 × 24 头，pool 按 `seqs` 界开 slot，四卡各租 4B 行时第一步就 `lease denied:
remapping`。`Manifest::seq_slots` 改为跟 `rows` 界（没有 `rows` 时才是 `seqs`）。
规则：slot 数的来源是 line table 的宽度界，不是 batch 的序列数。

## 2026-09-03，K2 fork + K3 host 层

**拷贝流不能让计算流等。** 第一版 wake 让 compute stream 的下一次 launch 等 transfer
stream 上的拷贝，结果每次 6 GB 唤醒卡 decode 30 ms，8 路 decode 的 token 间隔 p50 从
3 ms 变成 20 ms。原因是"等事件"把两条流又串成了一条。改成 runtime 攥着 park 中的
checkpoint 直到落地才还页、wake 只给 `Waking`、`awake` 落地了才交 `Lease`：没有
`Lease` 就没有程序能读到路上的页，不用任何 stream 等待。规则：**异步资源的"可用"
用类型表达，不用事件等待表达**——一个句柄能被拿到，就意味着它背后的检查已经做完
（CLAUDE.md 里"a type exists because something was checked"的又一个实例）。

**pinned 内存要落在卡所在的 NUMA 节点。** 同一个 6 GB 唤醒在 GPU 0 上 31 ms、在
GPU 2 上 52 ms，一度以为是卡的差异；用 `numactl --membind` 一试，GPU 2 也是 31 ms。
一个 GB300 tray 两颗 Grace 各挂两张卡，`cuMemHostAlloc` 跟着调用线程的内存策略走，
跨到另一颗 CPU 的 C2C 只有 115 GiB/s，本地 180 / 197。规则：**runtime 自己绑**
（sysfs `numa_node` + `set_mempolicy` 包住分配），不指望调用方记得 numactl；任何
host 侧带宽数字都要写明块在哪个节点上。

**参照 batch 要和被测 batch 走同样的 bucket。** fork 门禁第一轮"分叉后各行都分歧"，
查了半天 fork 的页拷贝和 slot 拷贝，其实是 batch 从 4 行长到 6 行、bucket 从 4 变 8，
随机行的近平局翻了；同样 5 行不 fork 的对照只错 1 步（3 ULP）。改成参照 batch 在同一
步同样分叉，全部一致。规则：**行与行的对照只在同一 bucket 内有意义**，任何会改变
batch 大小的动作（fork、准入、结束）都要在参照里原样重演；比对失败先问"两边跑的是
不是同一组 kernel"，再问数值。

**基线要把变量隔开，先量再猜。** decode 抖动门禁一开始 p90 100 ms，第一反应是拷贝
在抢带宽；给 admitted 日志加了 lookup / lease / prefill 三段计时才看到：lookup 0.2 ms、
lease 0.3 ms、prefill 40 ms——命中之后剩下几个 token 的 prefill 尾块在 25k 上下文上
就要 40 ms（挖来的 prefill attention 只按 head 并行），而且 resident 命中一样慢，跟
host 层无关；wake 只多 15 ms，还全落在被唤醒的请求自己身上。规则：**门禁的对照组
必须是"同样的负载、只差被测的那一件事"**（这里是 resident 命中 vs parked 命中），
churn 请求用 token id 发，把前端 tokenizer 摘出去；先加计时再改设计。

**结论从数据里读，别从叙事里读。** 第一版 jitter 报告把 tokenization、prefill 尾块、
拷贝三件事混在一个 p90 里，看起来 K3 门禁差得远；分开之后 K3 本身只贡献 +1.7%，
另外两件是既有问题（K5 的事）。写 serve.md 时把三件事分行写清，谁的账归谁。

**`std::mem::zeroed()` 不能用在含 Rust enum 字段的 FFI 结构上。** cudarc 的
`CUDA_MEMCPY2D_st` 的 `srcMemoryType` 是 `CUmemorytype` 枚举，0 不是合法值，
release 构建直接 panic（"attempted to zero-initialize type … which is invalid"）。
规则：FFI 结构逐字段写全，宁可啰嗦。

**kern-serve 是独立 workspace，runtime 改了要单独重建。** 修了 device.rs 之后只重建
了 kern-runtime 和 examples，kern-serve 还是旧的 runtime，门禁跑到一半 panic。规则：
**运行门禁前先看 binary 的时间戳**，或者把两个 workspace 的构建写进同一条命令。

**四层 K3 的 fixture 不在这个仓库。** `k3_golden` 默认的 `tests/fixtures/k3_4l_greedy.json`
在 pegainfer-k3 里；tray08 的 K3 权重只有 EP8 shard，EP4 在 tray07。规则：门禁命令
连同它依赖的文件位置一起写进 roadmap 的门禁列，别只写结果。

**醒来的 session 再睡会再拷一份。** host 上的页链按 device 页节点去重，wake 出来的是
新页，所以同一 session 第二次 park 认不出上一次的副本，旧的那份只是变成最冷的先走。
没改，记在 serve.md 里；等真有 session 反复睡醒的负载数据再决定要不要让 wake 带着
host 节点的身份。

**没能复现的事也要记。** kern-serve 一致性门禁的某一次运行里一个 12k 的 filler 请求
花了 15 s（stats 里 prefill_tok_s 只有 1651），之后 4 次重跑都是 0.7 s。当时的日志是
INFO 级，看不到分段计时；以后门禁默认开 `kern_serve=debug`。

## 2026-09-02 之前

散在各处：qwen38-bringup.md（复现 ATen 数值时编译器的 FMA 缩并也是"顺序"的一部分）、
serve.md（manifest 里不许出现 slot 数：conv kernel 的 `num_cache_lines` 字面量让
slot ≥ 130 静默丢 state）、runtime.md（`cuMemAddressReserve` 的对齐必须是 2 的幂；
whole-state 读写只在第一次 remap 之前有效）、multi-gpu.md（peer / fabric 映射的显存
禁止 multicast TMA，会把发起的卡卡死到只能重启）。
