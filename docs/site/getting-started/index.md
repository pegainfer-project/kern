# Getting started

Kern runs an inference workload from an artifact rather than from model code
compiled into the engine. Start by building the manifest crate and verifying a
checked-in artifact; this first path does not need CUDA, weights, or kernel
binaries.

## What you will use

| Component | Job |
| --- | --- |
| `kern-manifest` | Parse and verify the typed wire format |
| `kern-runtime` | Allocate buffers, load modules, and execute programs |
| `kern` | Verify, run, benchmark, and A/B test artifacts |

## The first useful command

```sh
cargo run -p kern-manifest --example verify -- examples/qwen3-4b.json
```

The standalone verifier checks the manifest and its serving protocol. It is
suitable for development and CI that has no CUDA installation.

Continue with [Build and verify](/getting-started/build-and-verify), then read
[The model artifact](/concepts/artifact) before assembling a runnable model.

## Two ways to provide inputs

Commands accept explicit flags, or they can resolve a named target from the
nearest `kern.toml` at or above the current directory:

```sh
kern run demo
kern test demo
```

The config file contains host-specific locations. The manifest remains the
portable model contract. See the [`kern.toml` reference](/reference/config).
