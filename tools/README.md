# tools/：从 vLLM 到可执行 manifest 的流水线

单卡算子压测见 [`kern bench`](../docs/bench.md)，交互展示另见
[Performance Atlas](../docs/performance-atlas.md)：`kern bench --isolate` →
`profile_trace.py` → `profile_export.py` → `/perf/`。
`profile_standalone.mjs` 可导出不依赖网络的完整 HTML。

按执行顺序：

| # | 工具 | 输入 → 输出 |
|---|------|-------------|
| 1 | `capture_qwen3.sh` | vLLM 0.28（TRITON_ATTN，enforce_eager）跑 4 条递增 prompt → `dumped-kernels/pid<N>/`：全部 module cubin + `launches.jsonl`（每次 launch 的符号/grid/block/shmem/逐参数值 + `t_ns`） |
| 2 | `mine_capture.py` | `launches.jsonl` → 分析报告：按时间隙切 pass / 按核爆发切 forward、(range,offset) 指针稳定性分类、grid 表达式拟合（const/var/mul/ceil_div）。纯分析，无模型知识 |
| 2b | `capture_qwen3_spec.sh` | 同上但开 DSpark 投机（draft `weights/dspark_qwen3_4b_block7`，7 draft token，固定 k 验证）→ 第二个 dump：draft 的 non-causal unified 实例、context-KV precompute、verify pass |
| 3 | `gen_qwen3_decode.py` | 两个 `launches.jsonl` → `examples/qwen3-4b.json`（silu 用 HF hub 包）+ `qwen3-4b-silu-mined.json`（kern-test 的 A/B fixture，silu 用挖矿实例）+ `qwen3-4b-dspark.json`：真实 ABI + 手写连线，发射前用挖矿地址逐项断言证伪（q/k/v 视图偏移、KV 池布局、权重指针互异、precompute 的 K-only rope 与 grouped k_norm…）；顺带按 num_regs + cuobjdump 消歧两个同 ABI 的 unified 实例并钉哈希 |
| 3b | `kern_manifest.py` | 生成器共用件。`DumpIndex(dump).pin(symbol, regs, param_sizes)`：按内容索引 dump 的 module，给每个挖矿 launch 钉唯一的 module（寄存器数分不开的 Triton constexpr 实例再按 `.nv.info` 参数布局分）。`normalize()` 后处理：把 launch 内联的 cubin/sha256 提升到 `modules` 表、把每次 call 都相同的接口标量折进 impl 的 launch 字面量、抹掉恒等连线与重复 `params`、规范键序。生成器写长，wire form 写短 |
| 4 | `kernels/`（`index.py`、`import_handwritten.py`、`import_cubin.py`、`check.py`、`../kernel_index/<family>.toml`） | kernel registry（../docs/registry.md）：`import_handwritten.py` 在 kernel-lab 里 nvcc `kernels-src/*.cu`，字节进 `~/.cache/kern/blobs/<sha>`（`--upload` 再进 HF `Pegainfer/kern-kernels`），sha 进 `../kernel_index/<name>.toml`；生成器通过 `index.variant("name", K=V).module` 钉 sha——换 nvcc / 换 flag / 改源码就是另一个核，`import_handwritten.py` 会打印 MOVED 的 sha |
| 5 | `extract_kernels.sh` | manifest + dump 目录 → `kernels/`：`modules` 表里每个 module 按 sha256 在 dump（递归）/ `target/cubins` 里找到文件，落地为 `<module>-<sha12>.cubin`；只增不减，同一目录可放每个版本，A/B 两份 manifest 共用（runtime 只装载各自点名的） |
| 6 | `qwen_weights.py` | 生成器的最后一道（`gen_qwen35.py` / `gen_qwen3_decode.py` 的 build 都以它收尾，也可单跑改写一份 manifest）：给每个 weight buffer 写 `bind`——qkv / gate_up 拼接、tied lm_head 指回 embed、draft 的 fc 按列切 5 块、fused KV 按层 cat——并生成 `load` once program 算 checkpoint 里没有的表：Gemma norm 的 +1、A_log 的 f32、rope cos / sin、kv_scales 全 1、chunk 索引（核在 `kernels-src/weight_prep.cu`）。没有导出这一步：`--weights` 直接给 HF snapshot 目录，runtime 只读 safetensors header |

## 其他

Kimi-K3 的生成器、核源码、harness 与 manifest 在
[pegainfer-project/kern-k3-pp-prefill](https://github.com/pegainfer-project/kern-k3-pp-prefill)
（PP prefill，钉一个 kern commit）；本仓库删掉它们之前的最后一版见 git 历史。
`crates/kern-manifest/tests/fixtures/k3/` 是从那里 `kern cut` 出来的几层，CI 用它
守住 kern 对这条线的契约。

| 工具 | 输入 → 输出 |
|------|-------------|
| `once.py` + `kernels-src/weight_prep.cu` | 生成器共用的 `load` program 构造：checkpoint 没有的表（Qwen 的 norm+1 / rope 表）是 `carry`，`load` 在装载后算一次；权重本身全部 `bind` HF checkpoint（TP 分片 = 按 rank 选行 / 列，`interleave` 段），不再有导出中间态 |
| `trtllm_fmha_abi.py` | TensorRT-LLM gen 的 ragged context FMHA（FlashInfer 的 cubin，索引族 `trtllm_fmha_ctx_h192_v128`）作为 kern op 的数据：`bytes<1344>` 的 `KernelParams` pack（四个 tensormap + 指针 + 标量，从 kernel-capture 提出，见 k3-kernel-abi.md K13） |
| `gen_trtllm_fmha_probe.py` | 只含 `fmha` 一个 op 的 manifest，`program_io` 喂 `tools/trtllm-fmha/probe.py` 的 dump 做逐位门禁（K13） |
| `kernels/pick.py` | 把 `kern bench` 选出的变体写进索引的 `[[pick]]`，report 进 blob store |
| `vmm_bench.py` | CUDA VMM 块池的代价（roadmap K1b）：`cuMemMap` / `cuMemSetAccess` / `cuMemUnmap` 每块延迟，拼一页 KV / 一个 state slot / 一份 K3 state 的总耗时与首次访问，以及 map/unmap 进行时同卡带宽循环的抖动；只用 driver API（ctypes），不用编译，`python3 tools/vmm_bench.py <gpu>` |
| `crates/kern-run/examples/agentx_replay.rs` | K1 的 host 门禁：把 AgentX（Claude Code）trace（HF `semianalysisai/cc-traces-weka-062126`）按时间回放进 `Pool` + `Prefix`（只有页号，不碰 GPU），报命中率 / extend 分位 / 淘汰数 / remap 数与 slot 峰值；`--unit` 页大小、`--kv-bytes` 每 token 字节、`--state <每 slot 字节>` 带循环状态（只在请求结束留 checkpoint）、`--budget-gib` 页与 slot 共用的预算（或 `--capacity` token 数 + `--slots` 起始 slot 数）、`--concurrency` |

支撑件：

- `kernel-capture/`：CUPTI 注入库（vendored from pegainfer PR #982 + `t_ns` patch），`CUDA_INJECTION64_PATH` 挂进目标进程。
- `kernels-src/`：手写核（embedding、argmax、gemma norm、copy_rows、sigmoid_mul、DFlash2 的 conv/select/topk…），其余全部来自 vLLM dump。
- `capture_abi_probe.sh`：诊断用。快速抓某个 attention backend 的 ABI（`ABI_PROBE_BACKEND=FLASH_ATTN` 等），当初用它实锤 FA4/trtllm-gen 不可 rebind。
- `capture_sglang.sh` + `capture_sglang.py`：跨框架演示——同一注入库 dump SGLang（docker 镜像里跑）。实测 GB300 上 SGLang 几乎全员 struct ABI（trtllm-gen attention + nvjet GEMM + 单 struct 参数的自家 JIT 核），可挖性远差于 vLLM，见 ../docs/kernel-mining.md。
