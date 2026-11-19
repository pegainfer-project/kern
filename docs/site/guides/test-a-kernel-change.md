# Test a kernel change

`kern test` compares a candidate artifact with a trusted reference. Use it
after changing a kernel implementation or its manifest wiring: span-level A/B
(`kern test <target>`) localizes numerical and timing differences on a seeded
workload, while `kern test e2e-logits` is the end-to-end numerics gate over
long real conversations.

![Verification gates](/figures/test-gates.svg)

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
logit_kl = 0.01
```

The kernel directory may hold modules for both versions. Each manifest resolves
the bytes it names by SHA-256.

## Run the comparison

```sh
./target/release/kern test demo
```

The command reports structural differences, numerical comparison and
timing evidence. Its exit status is part of the API:

| Status | Meaning |
| --- | --- |
| `0` | PASS |
| `1` | FAIL |
| `2` | INCONCLUSIVE |

For automation, write a portable report or emit one JSON object:

```sh
./target/release/kern test demo --out test-report.json
./target/release/kern test demo --json
```

## Reproduce the workload

The seed, manifest, and options determine the sampled workload. Keep them with
the result. A real prompt is optional; without one, prefill and decode tokens
come from the seeded generator.

Timing is evidence from the machine that ran the command. Re-measure a kernel
change on the intended hardware and workload before treating it as an
optimization.

## End-to-end logits (`e2e-logits`)

Span-level diffs show *where* an artifact changed, not whether the change holds
up over long contexts and multi-step decode. `kern test e2e-logits` gates a
candidate end to end by teacher-forcing it along a baseline's sampled tokens
through its own prefill and decode path, so recurrent-state and deep-context
errors accumulate naturally.

`kern test e2e-logits` takes explicit CLI flags only; it does not read
`kern.toml`.

### Record a baseline and its self-noise

Sample a baseline once over a question set (a Parquet file with `source`,
`messages`, and `tools`, such as a split from `Pegainfer/kern-prompts`):

```sh
kern test e2e-logits record \
  --manifest base.json \
  --kernels kernels/ \
  --weights <checkpoint-dir> \
  --questions standard.parquet \
  --seed 0 \
  --max-new 512 \
  --out base-standard.parquet
```

Optional sampling flags (`--temperature`, `--top-k`, `--top-p`) default to the
checkpoint's `generation_config.json`; `--max-new` defaults to `16384`. For a
multi-rank manifest, pass `--gpu 0,1,2,3` (or a starting ordinal such as
`--gpu 0`).

After sampling and saving the top-20 logprobs per output position, `record`
replays the baseline along its own tokens in half-sized decode batches so every
position lands in a different batch. The resulting per-position KL is stored in
the Parquet file as `noise`: the rounding bar any equivalent candidate lands on.

### Score a candidate

Replay the candidate manifest against the recorded baseline:

```sh
kern test e2e-logits \
  --manifest cand.json \
  --kernels kernels/ \
  --weights <checkpoint-dir> \
  --reference base-standard.parquet
```

Add `--out cand-standard.parquet` to keep the candidate's recording in the same
schema, or `--kl <nats>` to judge against an explicit mean KL ceiling instead of
the recording's declared noise (useful when trading precision for speed, or when
scoring against a recording with no noise).

### Interpret the verdict

At each output position, `e2e-logits` computes `KL(baseline ‖ candidate)` over
the baseline's top-20 tokens plus the tail bucket and compares the mean against
the baseline's self-noise:

```text
mean KL <= noise + 3 standard errors   ->  PASS (exit 0)
mean KL >  noise + 3 standard errors   ->  FAIL (exit 1)
```

Errors (usage, inputs, device) exit `2`. The report also prints mean KL by
context depth and by decode step, median and p99 KL, and argmax flips ranked by
regret in nats (the baseline's log-probability gap between its top-1 token and
the candidate's choice), so a near-tie flip is easy to distinguish from a
confident error.
