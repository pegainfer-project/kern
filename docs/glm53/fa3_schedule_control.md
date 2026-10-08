# FA3 preparation: isolated correctness checkpoint

**Status: experimental, not a default change or an online speedup.**
This checkpoint concerns the exact `GLM-5.3-Flash` FA3 cubins used in the
H100 control. Native full-model Responses correctness and competitive
performance gates remain open.

Here, **M is the number of packed attention queries**, including speculative
rows. It is not the number of HTTP requests. A small request batch can
therefore reach M=32.

## Two separate faults

### The one-warp preparation entry has capacity 31

The captured `prepare_varlen_num_blocks_kernel<1, true>` entry uses a
boundary lane for adjacent `cu_seqlens` loads. Lane 31 does not calculate
a work block. With M=32, the metadata contains 31 rows with one work block
and one row with zero work blocks.

The batch-index permutation still passes. A permutation check alone is
not a work-coverage check.

The **same cubin** exports `<2, true>`. Selecting that entry with block
`[64, 1, 1]` covers all 32 rows in the guarded control, without enlarging
the existing 32-entry metadata scratch buffers. Increasing the block size
while retaining the one-warp entry is not the tested intervention.

### Dynamic split selection changes the arithmetic partition

The exact forward entry computes:

```text
key_blocks       = ceil(effective_key_length / 64)
blocks_per_split = ceil(key_blocks / num_splits)
first_key_block  = split_index * blocks_per_split
```

With the original preparation SM hint of 132, the split count depends on
M. For example, 32 key blocks are partitioned into blocks of size 2 with
29 or 16 splits, but into blocks of size 3 with 11 splits. This changes
the floating-point reduction, even when every attention input is equal.

In the bounded control, a preparation hint of **4224 = 132 × 32** preserves
the unchanged M1 per-row rule over M=1..32:

```text
num_splits = min(ceil(effective_key_length / 64), 29)
```

This is a scheduler argument, **not a hardware SM count**, CPU-affinity
setting, or direct overwrite of the split-count buffer. It is not a
general scheduling recommendation for other cubins, shapes, or devices.

## Completed controls

1. **Preparation census:** 128 cases, M=1..32, four length profiles and
   eight replays. This exposed the missing-work row despite passing
   permutation checks.
2. **Preparation factorial:** 108 cases, four arms, three profiles and
   nine boundary-relevant M values, with four replays. Input and
   32-entry scratch guards passed.
3. **Actual FA3 factorial:** 576 cases, four arms and two replays per arm.
   This used the frozen `kern_runtime::Runtime` and the exact captured
   preparation, forward, and combine cubins, not a substitute attention
   implementation.

The attention cases cover M=1..32, lengths 2043, 2051 and mixed
`[3, 65, 512, 2051]`, one constant-latent analytic fixture, two random
fixtures, and eager/CUDA Graph execution. Each row's reference is the
unchanged decode op at M1, evaluated both eager and under CUDA Graph.

| Preparation arm | Comparison with the unchanged M1 decode reference |
| --- | --- |
| Original | Some cases differ at every M=8..32; last row unwritten at M=32 |
| Hint only | All M=1..31 exact; last row still unwritten at M=32 |
| Two-warp entry only | No unwritten row; some cases still differ at M=8..32 |
| **Both** | **All 576 cases finite, byte-exact, replay-exact and graph/eager-exact** |

All inputs, the complete KV allocation, and output guards were checked.
The independent audit covers all 2,304 arm comparisons and the unchanged
M1 bridge. Alternating NaN sentinels expose unwritten output: the original
and hint-only M32 replay failures are **not evidence of random arithmetic
nondeterminism**.

## Explicit manifest experiment

`tools/glm53/fa3_schedule.py` applies **only the paired intervention** to
the preparation launch in `dsa_attn`, `mtp16_dsa_attn`, and
`mtp32_dsa_attn`. Forward/combine launches, modules, scratch allocations,
program calls, weights, buffers, and state definitions stay unchanged.
No CUDA code is compiled and no launch is added.

The input must be an existing normalized MTP manifest with `tokens.max=32`
and 32-entry `i32` metadata scratch in all three attention ops. The tool
checks the full original preparation ABI and all three module hashes.
The CLI also hashes the actual cubin files. Unknown, partially modified,
or already transformed preparation policies are rejected.

```sh
python3 tools/glm53/fa3_schedule.py /path/to/audited-mtp.json \
  --out /path/to/audited-mtp-fa3-experiment.json
kern verify /path/to/audited-mtp-fa3-experiment.json
```

The output must be a **new file in the same directory** so that relative
module paths remain valid. The tool does not resize a narrower manifest:
the public four-layer fixture and default generator output are not
automatically eligible. Do not bypass its checks to force a different
configuration into the tested envelope.

The three prepared full-model variants (k2 fusion-off, k7 fusion-off and
k7 core fusion) passed serving-manifest verification. Their native
Responses runs have **not** completed. Verification is a structural
check, not an inference-correctness result.

CPU-only regression tests:

```sh
python3 -m unittest tools.glm53.test_fa3_schedule
python3 -O -m unittest tools.glm53.test_fa3_schedule
```

These tests reuse the public captured preparation ABI and synthetic op
clones. They test transformation scope, immutability, refusal paths and
file handling; they do **not** rerun the GPU factorial. The frozen GPU
harness and private experiment bundle are not part of this checkpoint.

## Evidence identities

SHA256 fingerprints of the retained experiment evidence:

| Artifact | SHA256 |
| --- | --- |
| Preparation cubin | `f691748aa14be4a6b1572b276b9d81a49729f34f6b7d87210c4b9f0ae7ef28d9` |
| Forward cubin | `6e1d657d3fa2bf11f71777e737e8cc00edcdc4e50f0a7ddd4c26582201c585bd` |
| Combine cubin | `0e1e3dbcaf559715e150d3c386f1ad8255bd091915cd87936d7e4dc7479575b6` |
| Preparation factorial report | `b70de92231f10822e0e9e3f7aa5a6ce214d3dd513a74518d2d687bf80f4b7a6e` |
| Attention summary | `dc83c2c276953ec45cf7c258495c037053cf220cb2fb1b19ee675d429a6f7fdd` |
| Attention measurements | `d47e17086e30197c09a6630e3aada1af0f08cc2c46c2ce5df38e2e70ca99c3ae` |
| Frozen harness source | `cf1ba1b53693fe0e9bbd5dcab4613f30b0879ed645fa3cb9657a558bfdb9aba0` |
| Frozen harness binary | `90950c241db8030f97ccefdf7a58dfe3d6d091060f4051e8b0a5555bd9057d12` |

No private paths, credentials, model-generated text, token arrays, or
tensor dumps are published here.

## Remaining acceptance gates

- Preserve the ordinary-decode full-token and native Responses controls.
- Pass k7 fusion-off/core and k2 fusion-off full-model gates against those
  unchanged references; isolated FA3 equality is not enough.
- Check the relevant batch sizes, cache/state behavior and repeatability.
- Measure non-instrumented online performance against both vLLM and
  SGLang with the same model and workload. No accepted online gain is
  claimed at this checkpoint.
