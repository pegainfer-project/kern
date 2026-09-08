# `kern.toml`

`kern.toml` describes host-local paths and command defaults. Kern finds the
nearest file at or above the current directory unless `--config` names one.
Relative paths are resolved from the config file's directory.

Unknown fields are rejected.

## Complete shape

```toml
gpu = 0
capacity = 4096

[targets.demo]
manifest = "artifacts/candidate/manifest.json"
reference = "artifacts/reference/manifest.json"
kernels = "artifacts/kernels"
weights = ["artifacts/checkpoint"]   # HF snapshot dir(s) or .safetensors files
tokenizer = "artifacts/tokenizer.json"

[kernels]
dumps = ["captures/framework-run"]
sources = "tools/kernels-src"

[test]
seed = 0x5eed
decode_steps = 32
logit_ulp = 4
fuzz = 6
prompt = "Optional real-text prompt"

[run]
prompt = "The capital of France is"
steps = 32
chunk = 512
```

## Top-level fields

| Field | Meaning |
| --- | --- |
| `gpu` | CUDA device ordinal used when the command does not override it |
| `capacity` | State capacity in tokens |
| `targets` | Named artifact locations |
| `kernels` | Inputs used by `kern kernels` |
| `test` | Defaults for `kern test` |
| `run` | Defaults for `kern run` |

## Target fields

| Field | Required for | Meaning |
| --- | --- | --- |
| `manifest` | All target-based commands | Candidate or runnable manifest |
| `reference` | `kern test` | Trusted A-side manifest |
| `kernels` | Run, test, bench, kernels | Directory containing device modules |
| `weights` | Run, test, bench | One or more Safetensors files |
| `tokenizer` | Run; test with a prompt | Hugging Face tokenizer JSON |

Target names have no built-in meaning. If the config contains exactly one
target, commands may omit its name; otherwise a target must be selected.
