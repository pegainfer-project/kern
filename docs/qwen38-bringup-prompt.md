# 任务：把 Qwen3.8-27B（+ DFlash2 投机）带上 kern，并把"支持一个新模型要多久"记录成证据

你在 `~/kern`（GB300 tray，aarch64）。先读 `README.md`、`docs/`（design / manifest /
kernel-mining / runtime / spec-decode / test / roadmap）和 `tools/README.md`，
再读 `~/.claude/CLAUDE.md`（集群规矩：GPU 共享、用前 `nvidia-smi`；host 无 cargo，
Rust 在 `kernel-lab` 容器里构建，home 挂在 `/work`；binary 宿主机裸跑）。

## 这件事为什么重要

kern 卖的是 model-agnostic runtime，但到今天只跑过 Qwen3-4B。这次的目标不只是
"跑通"，而是拿到一份可公开的证据：**一个 agent 从零把一个结构完全不同的新模型
带上 kern，花了多久、改了什么、没改什么。** 整个 session trace 会开源。所以：

- 全程在 `docs/qwen38-bringup.md` 记时间线：每个阶段的 wall-clock 起止、卡在哪、
  怎么解的、人工介入了几次（理想是 0）。用绝对时间戳。
- 最后统计：**kern-runtime / kern-manifest 改了多少行（期望接近 0 或只是 schema
  扩展），生成器（tools/）新写了多少行**。"新模型的成本落在生成器而不是 runtime"
  就是要证明的命题。运行时能不改就不改；必须改时先在时间线里写清楚为什么。

## 已经替你验证过的事实（2026-08-31，不用重复）

- 权重都在本地（HF cache，离线可用）：
  - target：`$HF_HUB/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`（18 shard，52 GB bf16）
  - draft：`$HF_HUB/models--incoai--Qwen3.8-27B-DFlash2/snapshots/dedf8df68adfb1afeaf7b7480c0a0243108177b4`（单文件 3.6 GB；HF 上 z-lab/ 与 incoai/ 同一份）
- vLLM 0.28.0 在 `~/kern/.venv`（host 裸跑，现有 `tools/capture_qwen3*.sh` 就是用它），
  registry 有 `DFlash2DraftModel → qwen3_dflash2.DFlash2Qwen3ForCausalLM`；
  `speculative_config={"method":"dflash","model":<draft>,"num_speculative_tokens":7}`。
- 单张 GB300 装得下（权重 54 GiB），TP1，`enforce_eager`，`max_model_len=4096`，
  `limit_mm_per_prompt={"image":0,"video":0}`（它是 VL 模型，我们只做文本）。四种组合
  都跑通：裸跑、裸跑+TRITON_ATTN、spec（默认 backend）、spec+TRITON_ATTN+triton GDN。
  参考脚本：`docs/qwen38-smoke.py`（本 prompt 同目录）。
- **挖矿必须的配置**（默认 backend 全是 struct ABI 挖不动）：
  `attention_config=AttentionConfig(backend="TRITON_ATTN")` +
  `additional_config={"gdn_prefill_backend":"triton"}`（默认 FlashInfer GDN prefill
  是 CuTe DSL；trtllm-gen / FA4 同理）。GDN decode 默认走 `torch.ops._C.fused_gdn_decode_post_conv_mtp`
  （vLLM 自家 CUDA op，ABI 是否 flat 自己抓了看；不行就找 triton
  `fused_recurrent_gated_delta_rule_packed_decode_kernel` 路径）。
- 注意：spec 配置下 draft 模型的 attention 被选成了 FLASH_ATTN（target 是 TRITON_ATTN；
  `triton_attn.supports_non_causal()` 明明是 True，看 `v1/spec_decode/llm_base_proposer.py`
  的 `_create_draft_vllm_config` 为什么没继承）。抓 draft 之前先把它钉到 TRITON_ATTN。
- 贪心输出：裸跑三种 backend 组合彼此一致（sky 那条 prompt），但 vLLM 自己的 spec
  输出和裸跑**不逐字节相等**——所以 kern 侧的 oracle 是 kern 自己的 decode program，
  不是 vLLM 的 spec 输出（和 Qwen3-4B DSpark 时一样）。
- 性能参考（eager，bs=2，128 token）：裸跑 15–20 tok/s，spec 45–55 tok/s。正式对比时
  用 vLLM bs=1、graph 默认开。

## 这个模型对 kern 来说结构上新的东西（按难度排）

target = Qwen3.5 混合架构（`text_config`：64 层，`full_attention_interval=4` → 16 层
full attention + 48 层 GatedDeltaNet；hidden 5120，MLP 17408，vocab 248320，lm_head 不 tied）：

1. **per-seq 定长 state**：GDN 层有 conv state（kernel 4）+ 递归 state（16 k-heads×128，
   48 v-heads×128）。kern 的 state 目前只有 per-token `bytes_per_token`（KV）；
   roadmap 里写明 Mamba 类 per-seq 定长 state 是已知的 schema 扩展点。这是本次唯一
   预期要动 schema 的地方——设计要保持"state 对 runtime 不透明"。
2. **投机下的递归 state**：verify 8 个 token 后被拒的 token 已经推进了 GDN state，
   需要回滚/重放。看 vLLM 怎么做（`mamba` cache mode `align`、`fused_gdn_decode_post_conv_mtp`
   的 mtp 语义），kern 里用 program 组合解决（快照 carry / 从 checkpoint 只重放接受的
   token），不要发明新 kernel。
3. full attention 层：24 q heads × head_dim 256，4 kv heads，`partial_rotary_factor=0.25`，
   mrope（文本 only 时三路 position 相同，验证它退化成普通 rope），`attn_output_gate`
   （gated attention，q 投影带 gate）。
4. draft = DFlash2（`config.json` 的 `dflash_config`）：5 层 qwen3 sliding-window(2048)
   非因果 block-diffusion drafter，block 8（7 draft token），`mask_token_id=248070`，
   context KV 来自 target 隐状态 tap `target_layer_ids=[5,19,33,47,61]`（对照 DSpark 的
   5 tap 方案，`docs/spec-decode.md`）；新增两样：backbone 里的 two-tap grouped conv
   （`conv_group_size=16, kernel 2`）和 candidate selector（rank 256, top_k 16，
   predecessor/successor codebook + einsum 打分）。实现见
   `.venv/.../vllm/model_executor/models/qwen3_dflash2.py`（继承 `qwen3_dflash.py`）、
   proposer `v1/spec_decode/dflash.py`。先搞清 selector 输出的是单条路径还是多候选，
   再决定 verify program 的形状。

## 分阶段（每阶段自成交付，先后顺序不要变）

**Stage 1 — target 本尊**：prefill（chunked，tokens∈[1,2048]）+ decode 两个 program，
`examples/qwen3.8-27b.json`。走现有流水线：capture → `mine_capture.py` → 新生成器
（`tools/gen_qwen35.py`，可以抄 `gen_qwen3_decode.py` 的骨架但别把它改成一锅粥）→
`extract_kernels.sh` → `export_weights.py`。验收：`kern-run` 贪心输出与 vLLM 0.28
裸跑（TRITON_ATTN + triton GDN）逐字节一致，≥300 token，多条散文 prompt；chunk=1 /
chunk=512 / eager 三路一致；跑 `kern test --a --b`（自己对自己）确认 harness 能
遍历这份 manifest 的 program。Stage 1 完成本身就证明了 model-agnostic（新的 state
类别 + 混合层）。

**Stage 2 — DFlash2**：`examples/qwen3.8-27b-dflash2.json`，六个 program 那套（参考
dspark 的 prefill/decode/decode_spec/verify/draft/draft_precompute），零新增手写核为
目标（能复用就复用；实在要写，放 `tools/kernels-src/`，并说明为什么没有现成的）。
验收：贪心输出与 Stage 1 decode 逐字节一致（greedy 投机无损=天然 oracle），报接受
长度和 tok/s。

**Stage 3 — 证据**：`docs/qwen38-bringup.md` 收尾（时间线、LOC 统计、runtime 改动
清单、性能表：kern decode vs vLLM bs=1 graph、spec vs 裸跑）；README 的 Proof 段和
website 的 06/MEASURED 加一行（网站构建用 `~/tools/node-v22.14.0-linux-arm64/bin`，
分寸线：不点名踩 vLLM、不用绝对句）。

## 工程规矩

- 权重产物写到 `weights/qwen3.8-27b/`（`weights/` 在 .gitignore，
  54 GB 别放 repo 里），dump 放 `dumped-kernels/`（也 ignore）。
- 用哪张 GPU 先 `nvidia-smi` 看，`CUDA_VISIBLE_DEVICES` 钉死一张；vLLM 挖矿用
  `gpu_memory_utilization` 留余量给 kern-run 同卡对比。
- bench prompt 必须多样化散文（重复句会首 token 出 EOS 假阳性）。
- `bash script | tail` 会吞退出码；`pkill -f` 小心杀到自己。
- 提交粒度：每个 Stage 一个 commit，message 讲"为什么"。不要等全做完再提。
- 卡住超过 30 分钟的问题：写进时间线（这是数据），换路径，别原地磨。
