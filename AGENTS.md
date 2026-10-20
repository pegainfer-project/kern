# kern

A model-agnostic GPU runtime. A model ships as a manifest (a typed declaration
of buffers, states, ops and programs), compiled kernels, and weights. The
runtime verifies the manifest, then executes it blindly. **It contains no model
and never will.** Anything that knows a model's name, layer count, head size or
decoding trick belongs in a generator under `tools/` or in the manifest itself,
never in `crates/`. When a change to `crates/` wants a model-specific branch,
the design is wrong, not the model.

Everything is written in English: code, comments, commit messages and docs.
`docs/site/` is the public documentation (VitePress, built into the website);
the files at the root of `docs/` are the design record, indexed by purpose in
`docs/README.md`. Its contracts (`manifest.md`, `runtime.md`, `pool.md`,
`protocol.md`, `serve.md`, `test.md`, `bench.md`, `spec-decode.md`,
`registry.md`, `vllm.md`, `sglang.md`) say what the code promises;
`docs/roadmap.md` is what is being built and the gate that closes each item;
`docs/lessons.md` is what went wrong before and the rule each incident left
behind — read it before a gate.

Only judgment lives in this file. Anything a machine can check (formatting,
the schema golden, lints) belongs in CI, not here.

## Orientation

- `crates/kern-manifest` — the manifest as types and schema (`types`), the
  load-time `verify`, which collects every diagnostic and never stops at the
  first, the serving `protocol` a driver reads off a manifest, and `cut`,
  which splits a manifest into pipeline stages. Pure host code.
- `crates/kern-pool` — the states' accounting: physical chunks (`chunks`),
  token and sequence slots handed out as leases (`pages`), frozen page chains
  (`store`), the prefix radix index (`prefix`), the pinned host tier
  (`host`), positions dealt over a group (`deal`). Pure host code with no
  CUDA; every decision comes back as a plan the runtime executes.
- `crates/kern-runtime` — the only crate that touches CUDA. Loads a verified
  manifest (`load`, `weights`, `derive`, `cubin`), lowers programs to flat
  launch lists (`compile`) and runs them (`exec`); runs kern-pool's plans
  (`lease`, `park`, `push`); joins ranks over exported memory and NCCL
  (`peers`, `nccl`); provides the cuBLAS externs (`cublas`); runs inside
  another engine's process on its memory and stream (`host`). `harness`,
  `compare` and `profile` serve the test and bench tools, never the serving
  path.
- `crates/kern-test` — the `kern test` harness as a library: static diff of
  two manifests (`diff`), A recorded span by span with its noise floor
  (`harness`), B replayed and fuzzed against it (`replay`), perf and the
  verdict in one report (`report`), all spoken to a `Side`; and the
  e2e-logits scoring (`e2e`). Pure host code with no CUDA; its tests drive a
  fake side.
- `crates/kern-run` — the `kern` binary: `run`, `test` (and `test
  e2e-logits`), `bench`, `kernels`, `verify`, `cut`, and `server`, which
  launches kern-serve. Around it: `kern.toml` (`config`), where inputs and
  weights come from (`inputs`, `weights`), chats rendered through vLLM's
  renderers (`chat`), and the real `Side` over `Runtime`.
- `crates/kern-serve` — the OpenAI-compatible server: pegainfer's frontend
  over `KernScheduler` (`scheduler`), which decides every rank's pool in one
  place (`ranks`) while each GPU is a process of its own (`executor`, and
  `watch` to fail the group fast); pipeline-stage prefill over a cut
  manifest (`pp/`); prefill/decode disaggregation, a pipeline head handing
  prompts to decode tiers (`pd/`). Outside `default-members`: it needs
  protoc and the pegainfer/vLLM serving stack, so it builds by name where
  those are (the kernel-lab container, the release runner).
- `crates/kern-py` — `import kern`, the runtime for a Python host (a
  cdylib built by maturin), and the `kern_vllm` / `kern_sglang` plugins that
  run a manifest as vLLM's or SGLang's model; shipped as the `kern-llm`
  wheel. Outside `default-members`.
- `tools/` — where model knowledge lives: kernel capture and lifting
  (`kernel-capture/`, `capture_*`), the registry's tooling (`kernels/`),
  hand-written kernel sources (`kernels-src/`), manifest generators
  (`gen_*.py`, `dsv41/`), and gates that are not models (`toy/`, manifests
  with an exact oracle; `e2e/`, the whole serving path per tray). A new
  model's manifest work lives in a repo of its own, as K3 does in
  `kern-k3-pp-prefill`.
- `tools/dsv41/**/Cargo.toml` — GPU harnesses over the runtime's public API;
  workspace members, built by name.
- `kernel_index/` — the kernel registry's facts, one TOML per family
  (`docs/registry.md`); the bytes live in the HF blob store.
- `examples/` and `kern.toml` — the fixture manifests and the targets that
  name them with their kernels and weights; `kern test qwen3-4b` runs from
  here. `schema/` holds the manifest schema golden.
- GPU tests need a free GPU on a shared tray: `nvidia-smi` before `kern test`.

## What we prefer

The runtime is small and that is a feature. The project's thesis (forward is
a typed pure function; state lives only at the boundary) applies to the code
that implements it: `kern-pool` decides, `kern-runtime` runs what it decided.

**Functional core, imperative shell.** Logic is `fn(data) -> data`. Effects
(CUDA, the clock, the ledger, files, logging) sit in a thin layer that does no
branching. If a function needs a GPU to be tested, the decision it makes
should move into a pure function that returns a plan; the shell only stages,
runs and reads back. `Pool` / `Lease` in `kern-pool` are the model: pure host
code that `Runtime` wraps.

**A type exists because something was checked when it was built.**
`Verified`, `Protocol`, `Lease`: only `verify`, `Protocol::check` and the
`Pool` make one, each returning `Result`; after that the invariant travels
with the type and nothing downstream re-checks it. Parse, don't validate. A
plain `Vec<i64>` of slots is a smell; a `Lease` that can only hand out its own
slots is the fix.

**Errors say who acts.** `kern_runtime::Error` is grouped by who has to fix it
(provider, artifact, caller). A message names the manifest object in backticks
and states expected versus got. `unwrap` / `expect` only for states the code
has already ruled out; anything reachable from input is an `Err`.

**Deterministic by construction.** No clock reads, no randomness, no
iteration-order dependence inside logic. Same input, same bytes out. Running
twice and diffing is a legitimate test, and a difference is a bug even when
both outputs look right.

**Comments explain the contract and the why, never the what.** The module doc
is the module's design doc, in prose (scheduler.rs and kern-runtime/lib.rs set
the standard). Function docs: one to three sentences, sentence case, no "This
function…". An inline comment justifies a non-obvious choice ("padding rows
write into a page nobody reads"). Work that is not done goes in
`docs/roadmap.md`, not in a source comment.

**Names.** Short locals in short scopes (`m`, `rt`, `s`, `q`); full words for
fields and public API. A type is the noun for the thing (`Lease`, not
`LeaseManager`). No `Helper`, `Util`, `Manager`, `_v2`, `new_`, `_impl`.

**Small.** No abstraction for one caller. No trait until there are two
implementations (`Scheduler` is pegainfer's, not ours). Prefer deleting to
adding. When a file outgrows what its module doc describes, split along a
contract, not along a line count.

**No compatibility layers.** Break the schema, bump the version, regenerate the
golden. No shims, no deprecated paths, no flags that keep old behavior alive.

**Dependencies are decisions.** A new crate is named in the commit message
with the reason it is worth its weight. CUDA-adjacent crates stay pinned
exact; pegainfer stays pinned by rev and is bumped deliberately.

## Gates

Nothing is done until its gate closes; a PR is not done until CI is green.

- Tests are integration tests: `tests/` in each crate, driving the crate
  through its public API only, so a test reads like a caller and survives
  any refactor that keeps the contract. Fewer, broader integration tests
  over many unit tests; a `mod tests` beside the code is the exception,
  for a pure function the public surface cannot reach in one step. When a
  test wants a private item, either the item is part of the contract and
  goes public, or the test belongs one level up. Fixtures are tiny and
  shared, never copied. Assert tuples so one line states one fact.
- Anything with enumerable inputs gets a property test rather than another
  example: layout arithmetic, accounting, a state machine against a reference
  model. The fake behind it is a model simpler than the code, never a mirror
  of it.
- The GPU oracle is `kern test`: logits against the reference manifest.
  Acceptance is reasonable agreement, not bit-exactness. At a divergence, read
  the top-1 / top-2 margin before calling it a bug; a near-tie that flips is
  kernel noise, a confident token that flips is a bug.
- `kern test e2e-logits` is the end-to-end numerics gate: `record` samples a
  baseline from the production manifest over a `Pegainfer/kern-prompts`
  split, the candidate is teacher-forced along those tokens in batches
  through its own decode path, and it passes when mean KL(baseline ‖
  candidate) ≤ `--kl` (design, commands and calibration in `docs/test.md`).
- `kern test <target>` is one command and, on the qwen3-4b fixture, a 7 s
  PASS: record A, replay B span by span, noise floor, perf, verdict in one
  report, so a kernel swap is never discussed without its report. `kern bench
  <target> --workload <sweep.toml> --out <json>` says where a program's time
  goes per shape, and the mix moves with the shape (Qwen3-4B: attention 47%
  of a 2k chunk over an empty cache, 95% of 512 rows over 32k of hit), so a
  workload names the shapes the service actually sees, never one
  representative one.
- kern-serve: conc1 output is identical to `kern run`; on a manifest whose
  step is a speculative round, acceptance rate does not collapse under load.
  Measured numbers go into `docs/serve.md` with date and machine, or they did
  not happen.
- Visibility is the tightest that compiles. The CI `hawk` job
  ([astral-sh/hawk](https://github.com/astral-sh/hawk)) denies
  `hawk::unnecessary_restricted_visibility`: a `pub(crate)` or `pub(super)`
  that could be private. It is exact, since restricted visibility cannot cross
  a crate. `hawk::dead_public` and `hawk::unnecessary_public` are an audit
  list, not a gate: hawk cannot see callers outside Rust (kern-py is a cdylib
  driven from Python), so grep an item's users before narrowing it. Fix by
  hand, never `--fix`. Hawk is built against one exact rustc, pinned with it
  in `.github/workflows/ci.yml` and bumped together; locally, install that
  release and run
  `cargo +<rustc> hawk check -A hawk::dead_public -A hawk::unnecessary_public -D hawk::unnecessary_restricted_visibility`.

## Commits

`<area>: <what is true after the commit, lowercase, no period>`. Area is the
crate or the feature: `runtime:`, `kern-serve:`, `manifest:`, `pool:`,
`test:`, `kern-py:`, `release:`, `docs:`. The subject states the new fact,
not the activity (`runtime: own the token slots, hand them out as leases`).
The body carries the why and the gate result if one was measured. One idea
per commit. A PR is squash-merged, so its title is the commit subject and
follows the same form.
