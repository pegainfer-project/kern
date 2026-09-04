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
weights
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
cargo build --release

# kern.toml at the repo root names the fixture target (manifest, reference,
# kernels dir, weights); every flag can still override it.
./target/release/kern run --steps 320
./target/release/kern run --manifest examples/qwen3-4b-dspark.json \
  --weights weights/qwen3-4b-dspark.safetensors --steps 320   # speculative decoding: the manifest's 7-row round, same runtime

# the loop: evidence for a kernel swap — diff, tap a seeded workload once
# (random tokens, multi-chunk prefill, N decode steps), then per cut: noise
# floor, bit-diff, fuzz around the tap; end-to-end logits are the verdict;
# eager/TPOT/sweep timing. ~10 s, one line per fact, the last line is the
# verdict (exit 0 PASS / 1 FAIL / 2 INCONCLUSIVE); --json for one object
./target/release/kern test qwen3-4b
```

`kern <cmd> --help` lists the flags; `crates/kern-run/src/config.rs`
documents `kern.toml` (targets are names you pick — kern reads no meaning
into them; anything the manifest already knows stays out of it). Logs go
to stderr (`RUST_LOG`); stdout carries the generated text or the report.
The pipeline that produces `kernels/` and `weights/` from a live vLLM
process is in [docs/runtime.md](docs/runtime.md) (`kern kernels` drives
it from `kern.toml`); what `kern test` measures and how it decides is in
[docs/attest.md](docs/attest.md).

## The contract

The wire format is one JSON Schema, generated from the code and
golden-checked in CI:
[`schema/manifest-v4.schema.json`](schema/manifest-v4.schema.json)
· [rendered](https://kern-baa.pages.dev/schema/).

| Path | What it is |
| --- | --- |
| `crates/kern-manifest` | Schema + verifier (pure, no CUDA) |
| `crates/kern-runtime` | The executor: fetch, verify, replay, CUDA graphs |
| `crates/kern-run` | `kern run` (generation) and `kern test` (A/B evidence) over the example manifests |
| `examples/` | Generated manifests — the artifact a provider ships (`*-silu-mined.json` is the attest fixture) |
| `docs/` | [design](docs/design.md) · [manifest](docs/manifest.md) · [kernel mining](docs/kernel-mining.md) · [runtime](docs/runtime.md) · [attest](docs/attest.md) · [spec decode](docs/spec-decode.md) · [roadmap](docs/roadmap.md) |

**Website:** [kern-baa.pages.dev](https://kern-baa.pages.dev/)
