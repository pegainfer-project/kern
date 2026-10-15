# Release：怎么发一个 kern binary，什么时候算发完

kern 的 release 是两样东西，Linux x86_64 与 aarch64 各一份，都不含 CUDA 库（runtime
在首次使用时 dlopen driver 与 cuBLAS），不含模型：

- 两个 binary：`kern` 与 `kern-serve`，不含 Python；目标机器是 Ubuntu 24.04 或更新。
- PyPI 包 `kern-llm`：一个 abi3 wheel（Python 3.10+，manylinux 2.28），装的是
  `import kern` 与 vLLM、SGLang 两个插件（`kern_vllm`、`kern_sglang`，entry point 注册，
  不设 `KERN_MANIFEST` 时不做事）。torch 由引擎带，wheel 不依赖它。

装完能跑什么由 manifest 决定，不在本页范围。

流水线在 `.github/workflows/release.yml`，binary 的打包与自检在
`scripts/package_release.sh`，wheel 的自检在 `scripts/check_wheel.sh`，安装脚本在
`scripts/install.sh`。

## 用户看到的

```bash
curl -fsSL https://kern-baa.pages.dev/install.sh | sh
kern --version          # kern 0.3.0 (<commit>, cuda 13.0)
pip install kern-llm    # import kern; vllm / sglang serve 时插件自动加载
```

- 网站的 `/install.sh` 是一条 302，指向最新 release 里的 `install.sh`；
  脚本随每个 release 一起发，网站不另存一份。
- 脚本下载 `kern-<target>.tar.gz` 与 `SHA256SUMS`，校验后装两个文件到
  `~/.local/bin/`（`kern server` 在自己旁边找 `kern-serve`）。`KERN_VERSION=v0.1.0` 钉版本，`KERN_INSTALL_DIR` 换目录，
  `KERN_BASE_URL` 换下载源（镜像、离线包）。不 sudo，不改 shell rc，
  不在 PATH 上时只打印一行 export。
- 装完只 warn 不 fail 地说三件 `kern run` 会需要而它看不到的事：
  driver 是否支持 CUDA 13（r580+）、cuBLAS 13 是否在 loader path 上、
  找到了但不在 path 上时 `LD_LIBRARY_PATH` 该指哪。
- 升级是重跑同一行，卸载是删一个文件。

| 要求 | 值 |
| --- | --- |
| OS / arch | Linux x86_64、aarch64 |
| glibc | runner 的（Ubuntu 24.04，2.39）及更新 |
| NVIDIA driver | 支持 CUDA 13（r580+） |
| 运行时库 | `libcublas.so.13` / `libcublasLt.so.13` 在 loader path 上（toolkit，或 pip 的 `nvidia-cublas-cu13` 加 `LD_LIBRARY_PATH`）；kern-serve 还要系统的 `libssl3` |

## 流水线做什么

1. **触发**：push 一个 `v*` tag。改到 workflow、`scripts/` 或 `crates/kern-py` 的 PR
   也跑 build、打包与 wheel（不 publish），所以流水线本身的改动在合并前就被 CI 验过。
2. **构建**：两个 target 各自在 Ubuntu 24.04 runner 上原生编（aarch64 用
   GitHub 的 arm runner，不交叉），apt 装 protoc 与 libssl-dev，`--locked` 一次编出
   kern-run 与 kern-serve。cudarc 绑定的 CUDA 版本由 Cargo.toml 的
   `cuda-13000` feature 钉死，构建机不装 toolkit；`KERN_COMMIT` 传 tag 的 commit。
   tag 名必须等于 workspace version（`v` + `cargo pkgid`），不等就停。
3. **自检**（`package_release.sh`，在 runner 上跑）：两个 binary `--version` 报的版本 =
   tag；`NEEDED` 里没有任何 CUDA 库；`kern verify examples/qwen3-4b.json` 与
   `kern-serve --help` 跑通（无 GPU，证明 loader 满足）。
   任一条不过，release 不出。
4. **wheel**：两个 arch 各在原生 runner 上用 maturin-action 的 manylinux 2.28 容器编
   `crates/kern-py`；`check_wheel.sh` 装进干净 venv，查 `kern.__version__` = tag、
   两个插件包在、两个 entry point 在、扩展模块 `NEEDED` 里没有 CUDA 库。
5. **发布**：合并两份 `.sha256` 与 wheel 的 sha256 成一份 `SHA256SUMS`，附上
   `install.sh`，对 tar.gz 与 wheel 做 GitHub build provenance attestation，
   `gh release create --verify-tag`，notes 是固定的一段要求说明 + 自动生成的 commit 列表。
   同名 release 已存在则拒绝，不覆盖资产。
6. **PyPI**：GitHub release 出了之后，同一批 wheel 用 trusted publishing 上传
   `kern-llm`：没有 token，PyPI 认的是 `pegainfer-project/kern` 的 `release.yml`
   在 `pypi` environment 里跑出的 OIDC 身份（pypi.org 项目页 Publishing 里登记）。
   单独一个 job：这步失败只红它自己，修好后 `gh run rerun <run-id> --failed`，
   release 不用重发。PyPI 不允许覆盖同版本。

## 怎么发

```bash
# 1. bump：workspace version 在根 Cargo.toml 的 [workspace.package]
git commit -am "release: v0.3.0"
git push origin master
# 2. tag
git tag -a v0.3.0 -m "kern v0.3.0" && git push origin v0.3.0
# 3. 看 Actions 的 Release 跑绿
gh run watch
```

## 发完之后的 gate（手动，有 GPU 才能做）

GitHub runner 没有 GPU，流水线只能证明 binary 起得来。release 发出后在一台
空闲的 tray 上（先 `nvidia-smi`）：

```bash
curl -fsSL https://kern-baa.pages.dev/install.sh | sh
kern --version                  # commit 与 tag 一致，cuda 13.0
cd <kern checkout> && kern test qwen3-4b   # 最后一行 PASS
pip install kern-llm==0.3.0 && python -c 'import kern; print(kern.__version__)'   # 引擎的 venv 里
```

再在引擎里过一遍 conc1：[vllm.md](vllm.md) 与 [sglang.md](sglang.md) 开头那条命令起
Qwen3.8-27B，greedy 输出与 `kern run` 逐 token 相同。都过，release 才算发完。

不过时：PyPI 上的版本号用过就不能再传，所以不是删了重发，而是 yank 掉 PyPI 上那一版、
删掉 GitHub release 与 tag，修好后 bump 一个 patch 版本再发：

```bash
gh release delete v0.3.0 --yes --cleanup-tag   # PyPI 那版在 pypi.org 的项目页 yank
```

## 不做的

macOS、Windows、musl（静态 musl 的 dlopen 不能用）、老 glibc、Homebrew、sdist、
docker 镜像。
