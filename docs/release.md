# Release：怎么发一个 kern binary，什么时候算发完

kern 的 release 是一个 binary：`kern`，Linux x86_64 与 aarch64 各一份，
不含 CUDA 库（runtime 在首次使用时 dlopen driver 与 cuBLAS），不含 Python，
不含模型。装完能跑什么由 manifest 决定，不在本页范围。

流水线在 `.github/workflows/release.yml`，打包与自检在
`scripts/package_release.sh`，安装脚本在 `scripts/install.sh`。

## 用户看到的

```bash
curl -fsSL https://kern-baa.pages.dev/install.sh | sh
kern --version          # kern 0.1.0 (<commit>, cuda 13.0)
```

- 网站的 `/install.sh` 是一条 302，指向最新 release 里的 `install.sh`；
  脚本随每个 release 一起发，网站不另存一份。
- 脚本下载 `kern-<target>.tar.gz` 与 `SHA256SUMS`，校验后装一个文件到
  `~/.local/bin/kern`。`KERN_VERSION=v0.1.0` 钉版本，`KERN_INSTALL_DIR` 换目录，
  `KERN_BASE_URL` 换下载源（镜像、离线包）。不 sudo，不改 shell rc，
  不在 PATH 上时只打印一行 export。
- 装完只 warn 不 fail 地说三件 `kern run` 会需要而它看不到的事：
  driver 是否支持 CUDA 13（r580+）、cuBLAS 13 是否在 loader path 上、
  找到了但不在 path 上时 `LD_LIBRARY_PATH` 该指哪。
- 升级是重跑同一行，卸载是删一个文件。

| 要求 | 值 |
| --- | --- |
| OS / arch | Linux x86_64、aarch64 |
| glibc | ≥ 2.28（RHEL 8、Ubuntu 20.04 及更新） |
| NVIDIA driver | 支持 CUDA 13（r580+） |
| 运行时库 | `libcublas.so.13` / `libcublasLt.so.13` 在 loader path 上（toolkit，或 pip 的 `nvidia-cublas-cu13` 加 `LD_LIBRARY_PATH`） |

## 流水线做什么

1. **触发**：push 一个 `v*` tag。改到 workflow 或 `scripts/` 的 PR 也跑 build 与
   打包（不 publish），所以流水线本身的改动在合并前就被 CI 验过。
2. **构建**：两个 target 各自在 `manylinux_2_28` 容器里原生编（aarch64 用
   GitHub 的 arm runner，不交叉）。cudarc 绑定的 CUDA 版本由 Cargo.toml 的
   `cuda-13000` feature 钉死，构建机不装 toolkit；`KERN_COMMIT` 传 tag 的 commit。
   tag 名必须等于 workspace version（`v` + `cargo pkgid`），不等就停。
3. **自检**（`package_release.sh`，在构建容器里跑）：`kern --version` 报的版本 =
   tag；`NEEDED` 里没有任何 CUDA 库；要的 glibc 符号版本 ≤ 2.28；
   `kern verify examples/qwen3-4b.json` 跑通（无 GPU，证明 loader 在 2.28 上满足）。
   任一条不过，release 不出。
4. **发布**：合并两份 `.sha256` 成一份 `SHA256SUMS`，附上 `install.sh`，
   对两个 tar.gz 做 GitHub build provenance attestation，`gh release create
   --verify-tag`，notes 是固定的一段要求说明 + 自动生成的 commit 列表。
   同名 release 已存在则拒绝，不覆盖资产。

## 怎么发

```bash
# 1. bump：workspace version 在根 Cargo.toml 的 [workspace.package]
git commit -am "release: v0.1.0"
git push origin master
# 2. tag
git tag -a v0.1.0 -m "kern v0.1.0" && git push origin v0.1.0
# 3. 看 Actions 的 Release 跑绿
gh run watch
```

## 发完之后的 gate（手动，有 GPU 才能做）

GitHub runner 没有 GPU，流水线只能证明 binary 起得来。release 发出后在一台
空闲的 tray 上（先 `nvidia-smi`）：

```bash
curl -fsSL https://kern-baa.pages.dev/install.sh | sh
kern --version                  # commit 与 tag 一致，cuda 13.0
cd ~/kern-1 && kern test qwen3-4b   # 最后一行 PASS
```

三行都过，release 才算发完。不过就删掉 release 与 tag 重发，0.x 不做 hotfix 版本：

```bash
gh release delete v0.1.0 --yes --cleanup-tag
```

## 不做的

macOS、Windows、musl（静态 musl 的 dlopen 不能用）、Homebrew、PyPI、docker 镜像、
`kern-serve`（它拉整个 pegainfer 栈，只能在 kernel-lab 容器里编，另发）。
