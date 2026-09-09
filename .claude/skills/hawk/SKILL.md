---
name: hawk
description: Audit or tighten Rust visibility with cargo-hawk (astral-sh/hawk); use when asked to run hawk, find dead or over-public items, or when the CI hawk job fails.
---

# hawk

Workspace-wide `pub` lint. Closed-world: it starts from the `kern` binary and
reports what nobody reaches. Three lints: `hawk::dead_public` (unused `pub`),
`hawk::unnecessary_public` (only used inside its crate, could be `pub(crate)`),
`hawk::unnecessary_restricted_visibility` (`pub(crate)`/`pub(super)` that could
be private).

## Run

Needs the exact rustc it was built against (0.1.14 ↔ 1.98.1; bump both together,
in `.github/workflows/ci.yml` too).

```sh
rustup toolchain install 1.98.1 --profile minimal
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/astral-sh/hawk/releases/download/0.1.14/cargo-hawk-installer.sh | sh
cargo +1.98.1 hawk check                      # everything, ~30 s
cargo +1.98.1 hawk check --output-format json # for scripting
```

CI enforces only the restricted-visibility class:

```sh
cargo +1.98.1 hawk check -A hawk::dead_public -A hawk::unnecessary_public -D hawk::unnecessary_restricted_visibility
```

## The one thing to know

`crates/kern-serve` is outside the workspace and drives the runtime through
its public API (`Runtime::lease_slot`, `retire`, `landed`, `pages_used`, …).
Hawk cannot see it, so most `dead_public` / `unnecessary_public` findings in
kern-runtime, kern-run and kern-manifest are that API. Before trusting a
finding, grep the item in `crates/kern-serve/src`. Anything kern-serve uses
stays `pub`.

Never run `--fix` here: it would privatise kern-serve's API.

Fix by hand, then confirm kern-serve still builds (kernel-lab container) and
the CI command above is clean.
