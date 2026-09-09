# Command-line reference

```text
kern [--config <PATH>] <COMMAND>
```

`--config` selects a `kern.toml`. Without it, Kern walks upward from the current
directory and uses the nearest one. `kern verify` ignores this option and never
reads `kern.toml`; it requires an explicit manifest path.

## Commands

| Command | Purpose | GPU required |
| --- | --- | --- |
| `kern verify <MANIFEST>` | Verify a manifest and print its serving protocol | No |
| `kern run [TARGET]` | Greedy single-sequence generation | Yes |
| `kern test [TARGET]` | A/B a candidate manifest against a reference | Yes |
| `kern bench [TARGET]` | Export raw program and call measurements | Yes |
| `kern kernels [TARGET...]` | Build and collect modules pinned by targets | Depends on inputs |

Use `kern <command> --help` for all command-specific flags. That output is
generated from the same option types the command parses.

## Input precedence

For settings supported by both forms, an explicit command-line flag takes
precedence over the selected target or section in `kern.toml`; configuration
takes precedence over the built-in default.

## Output and exit status

- Diagnostics and logs are written to stderr.
- `kern run` writes generated text to stdout.
- `kern test --json` writes one JSON object to stdout.
- `kern verify` exits with `1` when verification fails.
- `kern verify` logs `verified` at INFO level only after both manifest and serving
  protocol checks pass. Failures are logged at ERROR level; both go to stderr.
  The serving protocol summary goes to stdout on success.
- `kern test` exits with `0` for PASS, `1` for FAIL, and `2` for INCONCLUSIVE.

## Environment

`RUST_LOG` controls log filtering. Device selection is explicit through `--gpu`
or the `gpu` field in `kern.toml`.
