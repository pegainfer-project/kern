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

## Build the complete CLI

The complete workspace also requires an NVIDIA CUDA development toolkit whose
version is supported by the pinned `cudarc` dependency.

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
