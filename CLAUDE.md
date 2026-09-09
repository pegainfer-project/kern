# kern

A model-agnostic GPU runtime. A model ships as a manifest (a typed declaration
of buffers, states, ops and programs), compiled kernels, and weights. The
runtime verifies the manifest, then executes it blindly. **It contains no model
and never will.** Anything that knows a model's name, layer count, head size or
decoding trick belongs in a generator under `tools/` or in the manifest itself,
never in `crates/`. When a change to `crates/` wants a model-specific branch,
the design is wrong, not the model.

`docs/` is the design record (mostly Chinese); code, comments and commit
messages are English. `docs/manifest.md`, `runtime.md`, `serve.md`,
`spec-decode.md`, `attest.md` are the contracts; `docs/roadmap.md` is what is
being built and the gate that closes each item; `docs/lessons.md` is what went
wrong before and the rule each incident left behind — read it before a gate.

Only judgment lives in this file. Anything a machine can check (formatting,
the schema golden, lints) belongs in CI, not here.

## Orientation

- `crates/kern-manifest` — types, schema, `verify`. Verification collects every
  diagnostic; it never stops at the first.
- `crates/kern-runtime` — loads a verified manifest, allocates, lowers programs
  to flat launch lists, runs them. The only crate that touches CUDA.
- `crates/kern-run` — the `kern` binary (`run` / `test` / `kernels`),
  `kern.toml`, attestation.
- `crates/kern-serve` — its own workspace: the pegainfer/vLLM front end plus
  `KernScheduler`. Builds only inside the kernel-lab container.
- `tools/` — capture, extract, export, manifest generators. Model knowledge
  lives here.
- GPU tests need a free GPU on a shared tray: `nvidia-smi` before `kern test`.

## What we prefer

The runtime is under 3,000 lines and that is a feature. The project's thesis
(forward is a typed pure function; state lives only at the boundary) applies
to the code that implements it.

**Functional core, imperative shell.** Logic is `fn(data) -> data`. Effects
(CUDA, the clock, the ledger, files, logging) sit in a thin layer that does no
branching. If a function needs a GPU to be tested, the decision it makes
should move into a pure function that returns a plan; the shell only stages,
runs and reads back. `Pool` / `Lease` in pages.rs are the model: pure host
code that `Runtime` wraps.

**A type exists because something was checked when it was built.** `Lease`,
`Contract`, `Policy`, `Topology`: constructed by a `check` / `new` that returns
`Result`; after that the invariant travels with the type and nothing
downstream re-checks it. Parse, don't validate. A plain `Vec<i64>` of slots is
a smell; a `Lease` that can only hand out its own slots is the fix.

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
- kern-serve: conc1 output is identical to `kern run`; `--spec` acceptance
  rate does not collapse under load. Measured numbers go into `docs/serve.md`
  with date and machine, or they did not happen.

## Commits

`<area>: <what is true after the commit, lowercase, no period>`. Area is the
crate or the feature: `runtime:`, `kern-serve:`, `manifest:`, `attest:`,
`k3:`, `docs:`. The subject states the new fact, not the activity
(`runtime: own the token slots, hand them out as leases`). The body carries
the why and the gate result if one was measured. One idea per commit.
