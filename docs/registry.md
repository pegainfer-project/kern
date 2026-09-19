# Kernel registry：blob store 与 index（2026-09-19 定稿）

kern 从别人的工程里挖 kernel（trtllm-gen 的 cubin、CuTe-DSL 的 JIT 产物、
vendored 的 CUTLASS 核、自己的手写核），每挖一个都要花一次 ABI 提取、一次
数值门禁、一次 bench。挖出来的东西今天散在三处：`tools/kernels-bin/`（三个
预编译 cubin + README 里的配方）、每个核一份专属 python（`trtllm_fmha_abi.py`、
`flash_kda_abi.py`、`gen_k3.py` 里的 mla decode）、HF 上按模型建的仓库
（`Pegainfer/kern-qwen38-sm103`，cubin 与 manifest 混放）。下一次要用，得
重新翻 README、重新对 sha、重新跑 bench。本页定两样东西，让"挖到的"变成
"可取的"：**blob store**（字节，内容寻址）与 **index**（事实，进 git）。

前提是 `manifest.md` 已经定下的两条：module 的身份是 sha256，`source` 只是
取字节的提示；runtime 只认 sha，不认名字。所以本页不改 manifest，不改 runtime
的加载语义，只定字节放哪、事实记哪、生成器怎么取。

## 1. 决定

| 事项 | 决定 |
|---|---|
| 字节放哪 | HF private 仓库 `Pegainfer/kern-kernels`，`blobs/<sha256>` 平铺、无扩展名，与本地缓存 `~/.cache/kern/blobs/` 同构 |
| 事实记哪 | 独立 private Git 仓库的 `index/<family>.toml`（`KERN_INDEX_DIR` 指向该目录），一族一文件，由导入工具生成、CI 检查、人 review |
| ABI 在哪 | `tools/kernels/abi/<family>.py`，一族一个 Python 模块，输入 index 里的 variant 与形状、输出 manifest op 的 `impl`（今天的 `trtllm_fmha_abi.py` 就是这个形状） |
| runtime 改什么 | `source` 除 `hf:` 外接受裸 `https://` URL（`hf:` 退化为 URL 模板的语法糖）；modules 全是 registry ref 时不再要求 `--kernels` 目录 |
| 换主机怎么办 | 改 `source` 标签、重生成 manifest；sha 不变，index 不变。on-prem 交付 = 拷 `blobs/` 目录，不需要任何 registry |
| 不做什么 | 不做查询服务；TOML 里不做模板语言（ABI 留在 Python）；不按模型建索引；runtime 不知道 index 的存在 |

## 2. Blob store

**一个仓库，一个目录，只增不删。** `blobs/<sha256>` 是文件的全部：没有扩展名，
没有人读名，没有 revision。一个 blob 是 cubin、host 共享库、capture 的
`launches.jsonl`、`kern bench --out` 的 JSON，或一份 LICENSE 文本——凡是 index
要引用、又不适合进 git 的字节。源码不进 blob store（vendored 源码在
`tools/<name>/`，随 PROVENANCE 一起 review）。

manifest 里一个 module 写成：

```json
"trtllm_fmha_ctx_h192_v128": {
  "source": "hf:Pegainfer/kern-kernels/blobs/240811d976c01e4a9bee00041d98343f6f9254fefae598df46a97ff07c7df508",
  "sha256": "240811d976c01e4a9bee00041d98343f6f9254fefae598df46a97ff07c7df508"
}
```

sha 出现两次是故意的：键是人读名，`source` 是取字节的地址，`sha256` 是身份；
runtime 校验的是第三个，前两个换了都不影响运行。同一个 blob 镜像到别处，
`source` 换成 `https://<host>/blobs/<sha256>` 即可。

**为什么不沿用 `cubins/<name>-<sha12>.cubin`。** 那是给人在 HF 网页上翻目录用的；
有了 index 之后翻目录的场景消失，而平铺 sha 换来三件事：镜像是 `rsync`、
本地缓存目录就是完整的离线分发单元、永远没有重命名与冲突。

**为什么 HF、为什么 private。** runtime 已经会读 `HF_TOKEN` 拉 `hf:`，零改动；
第三方 cubin（Apache-2.0）与我们自己的 ABI、报告混放在一个仓库里最省事，
所以整仓 private。网络不是考虑项（海外机器）。等有第一个外部客户，按客户的
交付形态把 `blobs/` 镜像到自己域名下的对象存储或客户的 OCI 仓库，只改
`source` 标签。

**许可证。** 再分发 Apache-2.0 的 cubin 要带 NOTICE：每个 family 的 index 记
`license` 与 `license_blob`（LICENSE 文本的 sha），仓库 README 列全部来源。
`Pegainfer/kern-qwen38-sm103` 里的 `licenses/LICENSE.trtllm-gen.txt` 就是第一份。

**上传是工具的事，不是人的事。** `tools/kernels/import_*.py` 先把 blob 传上去、
确认可取，再写 index 条目；index 里出现的 sha 一定在 blob store 里，这是 index
的第一条不变量。

## 3. Index

索引单独 clone 到 kern checkout 外。将 `KERN_INDEX_DIR` 设为私有仓库内的
`index/` 目录；未设置时默认 `~/.local/share/kern/index`。例如：

```sh
export KERN_INDEX_DIR=/path/to/private-checkout/index
python3 tools/kernels/check.py
```

导入、选型和生成器共用这个目录。更新 TOML 后在私有仓库提交；HF
`Pegainfer/kern-kernels` 继续存放 `blobs/<sha256>`。运行已有 manifest
不需要索引仓库。不要把私有 checkout 放回公开仓库的 `tools/kernels/index/`。

`$KERN_INDEX_DIR/<family>.toml`，一族一文件。**族**是一批共享同一个参数
ABI 的核：trtllm-gen 的一个 dtype 组合是一族（`Bmm_MxE4m3_MxE2m1MxE4m3` 的
1460 个 tile 变体共一个 `KernelParams`），FlashKDA 的一个模板实例是一族，
我们的手写核每个 `.cu` 是一族。族名也是 ABI 模块名。

```toml
[family]
name = "trtllm_bmm_mxe4m3_mxe2m1_mxe4m3"
kind = "cubin"                         # cubin | so
sm = "sm100f"
abi = "trtllm_bmm"                     # tools/kernels/abi/trtllm_bmm.py
upstream = "flashinfer-cubin 0.6.18, bundle 8ec29a98612c3670f9f28825d1ed19f09496073b, batched_gemm-fa419f4-31ee4e5"
license = "Apache-2.0"
license_blob = "<sha256>"
abi_source = "bundle include/trtllmGen_bmm_export/KernelParams.h + flashinferMetaInfo.h"
abi_capture = "<sha256>"               # 探针 launch 的 launches.jsonl，证明 abi 模块与参考 launcher 逐字段一致
imported = "2026-09-19"

[[variant]]
name = "Bmm_MxE4m3_MxE2m1MxE4m3_Fp32_Ab32_Bb32_Cb32_t128x128x128_s7_et128x32_m256x128x32_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_tma_ldgstsSf_rgTma_clmp_siTuGlu_lbW8_lsfbW4_dynB_sm100f"
sha256 = "<sha256>"
entry = "<cuobjdump -symbols 里的那一个>"
launch = { block = 384, shared_mem = 227328, cluster = [2, 1, 1] }
tags = { a = "MxE4m3", b = "MxE2m1", c = "MxE4m3", act = "siTuGlu", tile = [128, 128, 128], stages = 7, epilogue_tile = [128, 32], mma = [256, 128, 32], dyn_batch = true, bias = "Fp32M", sched = "Pd2x1x2x3" }

[[pick]]
op = "moe_fc1"                         # 生成器问的名字
shape = "k3-prefill-16k-ep4"           # kern bench 的 workload 名
variant = "Bmm_MxE4m3_MxE2m1MxE4m3_…_siTuGlu_…_sm100f"
report = "<sha256>"                    # kern bench --out 的 JSON
measured = "2026-09-xx tray07"
```

三种表，三种事实：

- `[family]`：**来源与许可**。从哪个工程、哪个版本、哪个 bundle 拿的，ABI
  的源头是什么（上游头文件、DSL 的 launch、或只有 capture），哪次 capture 证明了
  ABI 模块是对的。
- `[[variant]]`：**字节与几何**。sha、entry、launch 几何（block / 动态 smem /
  cluster：这些不在 cubin 里，在上游的 launcher 或 metadata 里，导入工具从那里读），
  `tags` 是从上游文件名或构建参数机械解析出来的键值，供选择用。trtllm-gen 一族
  整批导入，不挑；手写核一族一个 variant 加 `defines`。
- `[[pick]]`：**测量过的选择**。生成器不在 730 个 variant 里自己挑：它按
  `(op, shape)` 取一个 pick，pick 一定带 `report`。没有测量就没有选择，这是
  index 的第二条不变量，也是 roadmap 里"kernel-as-package 目录带 test report"
  的落地形式。生成时不需要 GPU，也不需要网络。

**index 是生成产物，不手写。** `tools/kernels/import_flashinfer.py <bundle> <family>`
从 wheel 的 cubin 目录导入一族：解析文件名成 tags、算 sha、从 metadata 取
launch 几何、上传 blob、写 TOML（variant 按 name 排序）。`import_handwritten.py`
在 kernel-lab 容器里 build `tools/kernels-src/<name>.cu`、上传、写条目——从此
生成器不再在本机 nvcc（`handwritten.hw()` 今天每次生成都重编，且 sha 只有
kernel-lab 的 nvcc 13.0 能复现；改成"CI 编一次、index 钉 sha"之后这个坑消失）。
`[[pick]]` 由 `kern bench` 的结果写入，是唯一由人触发的条目，也要带 report。

**索引检查（无 GPU）：** 私有仓库 CI checkout 指定版本的 kern，设置 `KERN_INDEX_DIR` 后运行下述检查；公开仓库 CI 用合成数据测试索引工具，无需私有仓库权限。 `tools/kernels/check.py`——TOML 可解析；sha 是
64 位 hex 且全局唯一；每个 pick 引用存在的 variant 与 64 位 hex 的 report；
被 pick 的 variant 有 launch 几何；variant 按 name 排序（重新生成零 diff）。
`check.py --online` 对每个 sha HEAD 一次 blob store，上传前与 release 前手动跑。

**生成器怎么取：**

```python
from kernels import index, abi
v = index.pick("trtllm_bmm_mxe4m3_mxe2m1_mxe4m3", op="moe_fc1", shape="k3-prefill-16k-ep4")
launch = abi.trtllm_bmm.op(v, n=6144, k=3584, ...)   # v.module 给 {"cubin": "hf:…/blobs/<sha>", "sha256": …}
```

`index.pick` 返回的 variant 带 module 字典、entry、launch 几何与 tags；ABI 模块
负责把它变成 `impl`（pack 布局、tensormap、grid 表达式）。`handwritten.hw()` 与
`prebuilt()` 收编成 `index.variant("<family>")`，`tools/kernels-bin/` 与
`build_kernels.sh` 的 nvcc 路径随之删除（不留兼容层）。

## 4. 与现有约定的关系

- `manifest.md` 不变：`modules` 的语义、`hf:` 的语法、缓存目录、零信任传输都照旧。
  新增的只是 `source` 也可以是 `https://` URL。
- `kernel-mining.md` / `tools/kernel-capture/README.md` 的流程不变：vendor →
  探针 → capture → lift → op 数据。变的是最后一步的去处——lift 出的 ABI 进
  `tools/kernels/abi/`，capture 的 `launches.jsonl` 作为 blob 被 family 引用，
  cubin 进 blob store 而不是 `tools/kernels-bin/`。
- `k3-kernel-abi.md` 仍是 K3 那些核的契约文档；index 记的是"哪个字节、哪个
  几何、谁测过"，不重复 ABI 的散文。
- 模型仓库（`Pegainfer/kern-qwen38-sm103` 一类）以后只放 manifest 与模型自己的
  东西，cubin 一律指向 `kern-kernels`；现有仓库不动，新 manifest 按新写法生成。

## 5. 商业上意味着什么

index 是资产，blob 不是：第三方 cubin 谁都能从 wheel 里拿，我们卖的是"这个
形状用哪个变体、ABI 怎么摆、测过多少"。index 进 git 意味着它有 review 和历史；
TOML 已移到独立 private Git 仓库，格式不变；公开仓库保留索引工具与 ABI 代码。blob store
换主机是标签改写，离线交付是一个目录。继续遵守：不把事实写进 blob 的文件名，不让 runtime 依赖 HF 的任何特性。

## 6. 门禁

见 `roadmap.md` "Kernel registry 线"：R1 runtime 的 `https://` 与免 `--kernels`；
R2 index 格式、导入工具、三个预编译核迁入并删除 `tools/kernels-bin/`，重生成的
K3 manifest 除 module `source` 外逐字节相同、4 层 TP4 prefill 门禁不变；R3 第一
个新族 `trtllm_bmm_mxe4m3_mxe2m1_mxe4m3`（K3 prefill MoE 的 FC1 / FC2）。
