# tools/：从 vLLM 到可执行 manifest 的流水线

按执行顺序：

| # | 工具 | 输入 → 输出 |
|---|------|-------------|
| 1 | `capture_qwen3.sh` | vLLM 0.28（TRITON_ATTN，enforce_eager）跑 4 条递增 prompt → `dumped-kernels/pid<N>/`：全部 module cubin + `launches.jsonl`（每次 launch 的符号/grid/block/shmem/逐参数值 + `t_ns`） |
| 2 | `mine_capture.py` | `launches.jsonl` → 分析报告：按时间隙切 pass / 按核爆发切 forward、(range,offset) 指针稳定性分类、grid 表达式拟合（const/var/mul/ceil_div）。纯分析，无模型知识 |
| 2b | `capture_qwen3_spec.sh` | 同上但开 DSpark 投机（draft `weights/dspark_qwen3_4b_block7`，7 draft token，固定 k 验证）→ 第二个 dump：draft 的 non-causal unified 实例、context-KV precompute、verify pass |
| 3 | `gen_qwen3_decode.py` | 两个 `launches.jsonl` → `examples/qwen3-4b.json`（silu 用 HF hub 包）+ `qwen3-4b-silu-mined.json`（kern-test 的 A/B fixture，silu 用挖矿实例）+ `qwen3-4b-dspark.json`：真实 ABI + 手写连线，发射前用挖矿地址逐项断言证伪（q/k/v 视图偏移、KV 池布局、权重指针互异、precompute 的 K-only rope 与 grouped k_norm…）；顺带按 num_regs + cuobjdump 消歧两个同 ABI 的 unified 实例并钉哈希 |
| 3b | `kern_manifest.py` | 生成器共用件。`DumpIndex(dump).pin(symbol, regs, param_sizes)`：按内容索引 dump 的 module，给每个挖矿 launch 钉唯一的 module（寄存器数分不开的 Triton constexpr 实例再按 `.nv.info` 参数布局分）。`normalize()` 后处理：把 launch 内联的 cubin/sha256 提升到 `modules` 表、把每次 call 都相同的接口标量折进 impl 的 launch 字面量、抹掉恒等连线与重复 `params`、规范键序。生成器写长，wire form 写短 |
| 4 | `build_kernels.sh` + `handwritten.py` | `kernels-src/*.cu` → `target/cubins/`（nvcc，sm_103a）；生成器通过 `**hw("name")` 把当前 build 的 sha256 钉进 launch 的 module——换 nvcc / 换 flag / 改源码就是另一个核 |
| 5 | `extract_kernels.sh` | manifest + dump 目录 → `kernels/`：`modules` 表里每个 module 按 sha256 在 dump（递归）/ `target/cubins` 里找到文件，落地为 `<module>-<sha12>.cubin`；只增不减，同一目录可放每个版本，A/B 两份 manifest 共用（runtime 只装载各自点名的） |
| 6 | `export_weights.py` | HF checkpoint（+ draft checkpoint）→ `weights/`：qkv/gate_up 合并、rope cos_sin_cache 预计算、kv_scales 全 1、tied lm_head clone + tokenizer 文件；draft 侧另做 fc 按列切 5 块、融合 KV 权重 cat、markov 头原样 → `qwen3-4b-dspark.safetensors` |

## K3（多卡线，E1/E2）

kern 的 Kimi-K3 pruned decode 不是从 vLLM 挖的：vLLM 的 K3 路径是 fused
KDA / trtllm-gen MLA / DeepGEMM MegaMoE 的 struct ABI 与 torch.compile 混合，
不可 rebind。核来自 pegainfer 的认证 K3 核集，program 逐行照 pegainfer 的
`k3_step` 发射，oracle 是 pegainfer 的 golden fixture（`crates/kern-run/
examples/k3_golden.rs`）。

| 工具 | 输入 → 输出 |
|------|-------------|
| `kernels-src/k3_*.cu` + `build_kernels.sh` | K3 decode 核集（2026-09-02 CUDA C++ 重写，契约 ../docs/k3-kernel-abi.md；E2 时期 pegainfer 的 TileLang 桶核 + line shim 已删，见 git 历史）：`k3_residual`（attnres_rms / land_add_attnres_rms / land_add2）、`k3_conv_silu`、`k3_kda_core`、`k3_mla_prep`、`k3_mla_paged_attn`（cluster-8 split-KV，32k ctx 258 µs）、`k3_router_argmax`（router_topk / argmax / rms）、`k3_land`；elementwise（add、mul_sigmoid、norm、cuBLAS f32 partial 的 landing）都融进邻居核，每 rank B 是变量（≤ 64）。`k3_kv_append.cu`、`k3_mega_stage.cu`：latent 追加、MegaMoE 输入 staging |
| `k3-harness/` | 核的验收 harness：driver-API 加载 cubin，随机输入 + CPU 参考（`ref.h`），B ∈ {1,2,8,64}，容差 3 bf16 ULP + 1e-3 / 相对 RMS ≤ 2e-3；`run_all.sh` 跑全套；`notes/`、`reports/`（ncu）每核一份；`baseline/` 存 pegainfer 原版 MLA 核作对照 |
| `k3-mega/` + `build_k3_mega.sh` | DeepGEMM MegaMoE fork（SymBuffer 在设备上读 peer 表，R ∈ {1,4,8,16}）→ `k3_mega_moe.cubin` + `k3_mega_layout_dump`（E1） |
| `export_k3.py` | HF checkpoint → `dense/bookends + dense/l<i>`（所有 rank 共用）+ `experts/ep<R>-r<r>-l<i>`（按 rank 分片，MegaMoE 布局，复用 `export_k3_moe.py` 的变换）；slot 布局照 pegainfer `model/plan.rs`。权重放数据盘（tray04 `/data/<user>/kern-k3/`），跑在 vllm 镜像的 CPU 容器里 |
| `gen_k3_decode.py` | `--layers N --ranks R --max-ctx C --seqs S` → `examples/k3-<N>l-ep<R>.json` / `k3-ep<R>.json`：整条 decode program（93 层 1855 launch，742 个 cuBLAS GEMM），`tokens`/`seqs` 是变量，几何在 `GEOM` 表里；MoE 三步来自 `gen_k3_moe.mega_pieces`。核改了要先 `KERN_REBUILD=1 tools/build_kernels.sh` 再生成（manifest 钉 cubin 哈希） |
| `k3_oracle_dump.py` | 任一 OpenAI 兼容服务 → fixture（teacher-forced greedy）。vLLM 带 top-5 logprob，给 `k3_golden --margin-abs` 做 noise-floor 判定；pegainfer 的 K3 不出 logprob，用 `--no-logprobs`（`return_token_ids`，只记 argmax，逐步必须精确一致）。长 prompt 用 `--check-last N`：只对最后 N 个 prompt 位置和续写步请求参考，前面的位置记 `argmax=-1`（runner 只喂不比） |
| `flash_kda_abi.py` | FlashKDA（MoonshotAI，`tools/flash-kda/`）作为 kern op 的数据：prepare / recurrence 两个 launch 的 `bytes<256>` TiledCopy pack、workspace buffer、参数表（从 kernel-capture 提出，见 k3-kernel-abi.md K8）；`gen_k3_decode.py --span-max` 用它 |
| `gen_flash_kda_probe.py` | 只含 `flash_kda` 一个 op 的 manifest，`program_io` 例子喂 `tools/flash-kda/probe.cu` 的 dump 做逐位门禁（C2） |
| `flash_kda_ref.py` | numpy f64 的 K3 KDA 逐 token 参考，对 probe dump 报 out / state 的 relRMS（两种 state 朝向都报） |
| `k3_tokenizer_json.py` | Kimi-K3 的 tiktoken checkpoint → HF `tokenizer.json`（kern-serve 的前端只认这个），special token 与样例文本对 checkpoint 自己的 tokenizer round-trip 后才写出 |
| `vmm_bench.py` | CUDA VMM 块池的代价（roadmap K1b）：`cuMemMap` / `cuMemSetAccess` / `cuMemUnmap` 每块延迟，拼一页 KV / 一个 state slot / 一份 K3 state 的总耗时与首次访问，以及 map/unmap 进行时同卡带宽循环的抖动；只用 driver API（ctypes），不用编译，`python3 tools/vmm_bench.py <gpu>` |
| `crates/kern-run/examples/agentx_replay.rs` | K1 的 host 门禁：把 AgentX（Claude Code）trace（HF `semianalysisai/cc-traces-weka-062126`）按时间回放进 `Pool` + `Prefix`（只有页号，不碰 GPU），报命中率 / extend 分位 / 淘汰数 / remap 数与 slot 峰值；`--unit` 页大小、`--kv-bytes` 每 token 字节、`--state <每 slot 字节>` 带循环状态（只在请求结束留 checkpoint）、`--budget-gib` 页与 slot 共用的预算（或 `--capacity` token 数 + `--slots` 起始 slot 数）、`--concurrency` |
| `crates/kern-run/examples/k3_golden.rs` | 门禁 runner：`--manifest --weights --fixture --gpus --graph`；`--seqs B --mixed [--distinct]` 让每 rank 跑 B 行（行 0 是 fixture，其余随机 prompt），每行与"同 B 的自身拷贝批"逐 token 比（cuBLAS 按 m 选核，B=1 与 B=8 在近平局处可以不同）；跨 tray `--world 8 --rank-base 0|4 --rendezvous host:port` |


支撑件：

- `kernel-capture/`：CUPTI 注入库（vendored from pegainfer PR #982 + `t_ns` patch），`CUDA_INJECTION64_PATH` 挂进目标进程。
- `kernels-src/`：手写核（embedding、argmax、gemma norm、copy_rows、sigmoid_mul、DFlash2 的 conv/select/topk…），其余全部来自 vLLM dump。
- `capture_abi_probe.sh`：诊断用。快速抓某个 attention backend 的 ABI（`ABI_PROBE_BACKEND=FLASH_ATTN` 等），当初用它实锤 FA4/trtllm-gen 不可 rebind。
- `capture_sglang.sh` + `capture_sglang.py`：跨框架演示——同一注入库 dump SGLang（docker 镜像里跑）。实测 GB300 上 SGLang 几乎全员 struct ABI（trtllm-gen attention + nvjet GEMM + 单 struct 参数的自家 JIT 核），可挖性远差于 vLLM，见 ../docs/kernel-mining.md。
