# Install

`kern` is one static-looking binary: it links no CUDA library and loads the
NVIDIA driver and cuBLAS at first use. Install a release, or build it from
source; either way there is no toolkit and no Python to set up.

## Requirements

| | |
| --- | --- |
| OS / arch | Linux x86_64 or aarch64, glibc 2.28 or newer |
| Driver | supports CUDA 13 (r580 or newer) |
| Runtime libraries | `libcublas.so.13` and `libcublasLt.so.13` on the loader path: a CUDA 13 toolkit, an `nvidia/cuda:13.*-runtime` image, or `pip install nvidia-cublas-cu13` with `LD_LIBRARY_PATH` pointing into it |
| GPU | whatever the kernels you load were built for (the published Qwen3.8 registry is `sm_103a`) |

## From a release

```sh
curl -fsSL https://kern-baa.pages.dev/install.sh | sh
kern --version
```

The script downloads `kern-<arch>-unknown-linux-gnu.tar.gz` and
`SHA256SUMS` from the latest GitHub release, verifies the checksum and
installs one file to `~/.local/bin/kern`. It does not use sudo or edit your
shell profile; if the directory is not on `PATH` it prints the `export` to
add. Afterwards it looks at the machine and warns, without failing, about a
missing driver or cuBLAS.

- `KERN_VERSION=v0.1.0` pins a release
- `KERN_INSTALL_DIR=/opt/bin` picks the directory
- `KERN_BASE_URL=…` points at a mirror or an offline copy of the assets

Upgrading is the same line again; uninstalling is deleting the file.

## From source

```sh
git clone https://github.com/pegainfer-project/kern.git
cd kern
cargo build --release
./target/release/kern --version
```

Needs Rust and a C++ compiler. The CUDA API the runtime binds is fixed by
the pinned `cudarc` feature, so no CUDA toolkit is involved in the build.
The repository's `kern.toml` names fixture targets; with their kernels and
checkpoints in place, `./target/release/kern run qwen3-4b` works as-is.

## kern-serve

The HTTP server is a separate workspace (it carries the OpenAI front end and
its dependencies) and is not in the release archive yet. It builds from the
same checkout and needs `pkg-config`, `libssl-dev` and `protobuf-compiler`
on Debian/Ubuntu:

```sh
cd crates/kern-serve && cargo build --release
target/release/kern-serve --help
```

## Verify a manifest without a GPU

The manifest verifier is a pure Rust crate. It is what CI and editors can
run on a machine with no CUDA at all:

```sh
cargo run -p kern-manifest --example verify -- examples/qwen3-4b.json
# examples/qwen3-4b.json: ok, 3 forwards, 6 fills

./target/release/kern verify path/to/manifest.json   # the CLI's version: also prints the serving protocol
```

The published [JSON Schema](https://kern-baa.pages.dev/schema/manifest-v4.schema.json)
gives editor completion and early validation; `kern verify` remains
authoritative, since it runs the verifier the runtime itself uses.

## Next

- [Quick start](/getting-started/): run Qwen3.8-27B end to end
- [The model artifact](/concepts/artifact): what a manifest declares
