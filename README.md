# kern

**Why does an inference engine need to understand every model it runs?**

Today's engines are million-line programs that carry, in one codebase:

```
models × precisions × GPUs × parallelism × decoding tricks × …
```

Every new factor multiplies the matrix. The engine absorbs all of it.
To run *one* model, you first teach an engine about *all* of them —
and one kernel change re-certifies the entire product.

It was never supposed to work this way.

---

## A model is not code to merge. It is a program to verify.

A model ships as three files:

```
manifest.json     one typed declaration — buffers, ops, programs
kernels/          compiled device code, from anywhere
weights           the model's own checkpoint; the manifest says which tensor lands where
```

**A manifest is one point in that exponential space — declared, verified,
shipped alone.** The combination lives in the artifact, not the engine.

The runtime reads the manifest the way a compiler reads source:
verify everything, refuse anything inconsistent, then execute blindly.
It contains no model. It never will.

**One manifest. Any kernel. Zero trust.**

## Proof

One line of the manifest points at a kernel package on the Hugging Face
hub — the stock torch extension the PyTorch ecosystem uses:

```diff
  "silu_mul": { "params": ["out buffer<bf16>", "in buffer<bf16>"],
    "impl": { "launches": [{
-     "entry": "_ZN4vllm18act_and_mul_kernel…packed_silu_kernel…",
+     "module": "activation",
+     "entry": "_ZN4vllm18act_and_mul_kernel…",
  "modules": {
+   "activation": { "source": "hf:kernels-community/activation/…/_activation_320b408.abi3.so",
+                   "sha256": "73748b54…b1fe49aa" }
```

A runtime with no torch and no Python fetched it, verified it, ran it.
Output: byte-identical. Calls touched: zero.

And:

- The entire runtime is **under 3,000 lines of Rust**.
- Speculative decoding took **six programs and zero new kernels** — composed, not implemented.
- **92%** of vLLM's decode throughput, **37×** faster prefill than the naive path. *(Qwen3-4B · GB300 · bs=1)*
- A second model family — **Qwen3.8-27B** (hybrid linear attention, 64 layers) plus its DFlash2 speculative draft — cost the runtime and schema **49 lines**. Everything model-specific landed in a 1.4k-line generator and six kernels under 150 lines; decode 81 vs 95 tok/s, speculative 178 vs 176. *(timeline: [docs/qwen38-bringup.md](docs/qwen38-bringup.md))*

## The loop

Machines write kernels now. Shipping one still takes a human review cycle.

Here, a kernel change is not an engine change:

```
swap the impl → verify (ms) → byte-diff (s) → shipped
```

No PR. No review queue. No CI across every model.
The loop runs unattended. The engine goes back to being an engine.

---

## Try it

```bash
curl -fsSL https://kern-baa.pages.dev/install.sh | sh   # Linux x86_64 / aarch64, one binary
kern --version                                          # kern 0.2.0 (<commit>, cuda 13.0)
```

Then the [quick start](https://kern-baa.pages.dev/docs/getting-started/):
Qwen3.8-27B from the published kernels and the checkpoint you already have,
plain and with its DFlash2 draft, in four commands.

The binary links no CUDA library; it dlopens the driver and cuBLAS at
first use, so it needs an NVIDIA driver for CUDA 13 (r580+) and cuBLAS 13
on the loader path, nothing else. `KERN_VERSION=v0.2.0` pins a release,
`KERN_INSTALL_DIR` picks the directory (default `~/.local/bin`). How a
release is cut and gated: [docs/release.md](docs/release.md).

Or from source:

```bash
cargo build --release

# kern.toml at the repo root names the fixture target (manifest, reference,
# kernels dir, weights); every flag can still override it.
./target/release/kern run --steps 320
./target/release/kern run qwen3-4b-dspark --steps 320   # speculative decoding: the manifest's 7-row round, same runtime

# the loop: evidence for a kernel swap — diff, tap a seeded workload once
# (random tokens, multi-chunk prefill, N decode steps), then per cut: noise
# floor, bit-diff, fuzz around the tap; end-to-end logits are the verdict;
# eager/TPOT/sweep timing. ~10 s, one line per fact, the last line is the
# verdict (exit 0 PASS / 1 FAIL / 2 INCONCLUSIVE); --json for one object
./target/release/kern test qwen3-4b
```

Serve a configured target with the independent HTTP server:

```bash
cargo build --release --manifest-path crates/kern-serve/Cargo.toml   # or use the kern-serve the release installs
KERN_SERVE_BIN="$PWD/crates/kern-serve/target/release/kern-serve" \
  ./target/release/kern server qwen3-4b --port 8000
```

`kern server` resolves the target's manifest, kernels and weights from
`kern.toml`, then forwards server arguments unchanged. Explicit `--manifest`,
`--kernels` or `--weights` replace the corresponding target defaults.
`KERN_SERVE_BIN` selects the executable; otherwise kern looks beside itself,
then on `PATH`. On Unix the server replaces the CLI process, preserving signals
and exit status. Use `kern server <target> -- --help` for the server's flags.
`--renderer` explicitly selects a frontend renderer, such as `hf` or
`deepseek_v41`, when automatic detection does not recognize a checkpoint; its
chat format must match that checkpoint. The default is `auto`.
The V4.1 renderer matches the released text encoding for system messages,
multi-turn history and thinking. It accepts `low`, `high`, `xhigh` and `max`
reasoning effort; integer budgets from 1 to 100 can be supplied as
`chat_template_kwargs.reasoning_effort`. V4.1 tool schemas and historical tool
calls are encoded and tested, but its output `tool_calls` parser is not yet
adapted; this is not complete tool-calling support.

`kern <cmd> --help` lists the flags; `crates/kern-run/src/config.rs`
documents `kern.toml` (targets are names you pick — kern reads no meaning
into them; anything the manifest already knows stays out of it). Logs go
to stderr (`RUST_LOG`); stdout carries the generated text or the report.
The pipeline that produces `kernels/` and `weights/` from a live vLLM
process is in [docs/runtime.md](docs/runtime.md) (`kern kernels` drives
it from `kern.toml`); what `kern test` measures and how it decides is in
[docs/test.md](docs/test.md).

## The contract

The wire format is one JSON Schema, generated from the code and
golden-checked in CI:
[`schema/manifest-v5.schema.json`](schema/manifest-v5.schema.json)
· [rendered](https://kern-baa.pages.dev/schema/).

| Path | What it is |
| --- | --- |
| `crates/kern-manifest` | Schema + verifier (pure, no CUDA) |
| `crates/kern-runtime` | The executor: fetch, verify, replay, CUDA graphs |
| `crates/kern-run` | `kern run` (generation) and `kern test` (A/B evidence) over the example manifests |
| `examples/` | Generated manifests — the artifact a provider ships (`*-silu-mined.json` is the `kern test` fixture) |
| `docs/` | [design](docs/design.md) · [manifest](docs/manifest.md) · [kernel mining](docs/kernel-mining.md) · [runtime](docs/runtime.md) · [test](docs/test.md) · [spec decode](docs/spec-decode.md) · [roadmap](docs/roadmap.md) · [release](docs/release.md) |

**Website:** [kern-baa.pages.dev](https://kern-baa.pages.dev/)

### Named constants (manifest v4)

The optional `constants` table is recommended for readable model dimensions.
Use C-style `UPPER_SNAKE_CASE` names:

```json
"constants": {"VOCAB_SIZE": 151936, "HIDDEN_SIZE": 2560}
```

Use `"shape": ["seqs", "VOCAB_SIZE"]` or `{"i32": "HIDDEN_SIZE"}`.
Names work in numeric fields (including `i64`, `f32`, expressions, capacities
and byte offsets), except `schema_version`, which remains `5`. Values must be
numeric literals; aliases and constant expressions are not supported. Names
must not overlap `vars`. The loader expands references before typed validation,
so destination types and ranges still apply. String-only fields such as `buf`
are unaffected. Serialization emits resolved literals. The public wire schema
is generated by `kern_manifest::json_schema()` from the literal Rust types plus
these reference alternatives.
