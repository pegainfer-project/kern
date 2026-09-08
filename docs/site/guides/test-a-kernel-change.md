# Test a kernel change

`kern test` compares a candidate artifact with a trusted reference. Use it
after changing a kernel implementation or its manifest wiring.

## Declare A and B

```toml
[targets.demo]
reference = "artifacts/reference/manifest.json"
manifest = "artifacts/candidate/manifest.json"
kernels = "artifacts/kernels"
weights = ["artifacts/checkpoint"]   # HF snapshot dir(s) or .safetensors files

[test]
seed = 0x5eed
decode_steps = 32
logit_ulp = 4
fuzz = 6
```

The kernel directory may hold modules for both versions. Each manifest resolves
the bytes it names by SHA-256.

## Run the comparison

```sh
./target/release/kern test demo
```

The command reports structural differences, numerical comparison, input
perturbation results, and timing evidence. Its exit status is part of the API:

| Status | Meaning |
| --- | --- |
| `0` | PASS |
| `1` | FAIL |
| `2` | INCONCLUSIVE |

For automation, write a portable report or emit one JSON object:

```sh
./target/release/kern test demo --out attestation.json
./target/release/kern test demo --json
```

## Reproduce the workload

The seed, manifest, and options determine the sampled workload. Keep them with
the result. A real prompt is optional; without one, prefill and decode tokens
come from the seeded generator.

Timing is evidence from the machine that ran the command. Re-measure a kernel
change on the intended hardware and workload before treating it as an
optimization.
