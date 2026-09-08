# Build and verify

This path proves that the manifest crate builds and accepts a known artifact.
It intentionally stops before compiling or loading CUDA support.

## Requirements

- Rust and Cargo
- A checkout of the repository

## Verify without CUDA

```sh
git clone https://github.com/pegainfer-project/kern.git
cd kern
cargo run -p kern-manifest --example verify -- examples/qwen3-4b.json
```

Expected output:

```text
examples/qwen3-4b.json: ok, 3 forwards, 6 fills
```

This compiles only `kern-manifest`, the pure Rust parser and verifier used by
the runtime.

## Install the CLI

Releases ship one binary for Linux x86_64 and aarch64 (glibc 2.28 or newer).
It links no CUDA library and needs no toolkit or Python to install; at first
use it loads the NVIDIA driver (CUDA 13, r580 or newer) and cuBLAS 13 from
the loader path.

```sh
curl -fsSL https://kern-baa.pages.dev/install.sh | sh
kern --version
```

`KERN_VERSION=v0.1.0` pins a release and `KERN_INSTALL_DIR` chooses the
directory (default `~/.local/bin`). The script verifies the archive's
checksum, installs one file, and warns about a missing driver or cuBLAS
without failing.

## Build the complete CLI

The CUDA API the runtime binds is fixed by the pinned `cudarc` feature, so
building needs Rust and a C++ compiler but no CUDA toolkit.

```sh
cargo build --release
./target/release/kern verify examples/qwen3-4b.json
```

Successful output identifies the model and schema version, then lists the
declared row and group axes, input fills, state tables, and forward programs.
The command exits with status 1 when either structural verification or serving
protocol validation fails.

## Verify your own manifest

```sh
./target/release/kern verify path/to/manifest.json
```

Use the published [JSON Schema](https://kern-baa.pages.dev/schema/manifest-v4.schema.json)
for editor completion and early validation. `kern verify` remains authoritative:
it runs the verifier implemented by the runtime rather than relying only on a
generic JSON Schema validator.

## Next

- [Run a complete artifact](/guides/run-a-model)
- [Read the manifest reference](/reference/manifest)
