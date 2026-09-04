# `tools/k3-harness` — the K3 decode kernel acceptance harness

The contract is [`docs/k3-kernel-abi.md`](../../docs/k3-kernel-abi.md). This
directory decides whether a delivered kernel passes it.

```
ref.h        CPU references for every kernel, in the kernel's own buffer layout
harness.cu   the driver-API host program: generate -> reference -> launch -> compare -> time
naive/       a straightforward CUDA transcription of each reference + build.sh
baseline/    the retired pegainfer K5 (`git show HEAD:tools/kernels-src/k3_mla_paged_attn.cu`)
run_all.sh   every naive kernel at every required shape, as a table
```

Nothing here writes outside `tools/k3-harness/`, and the harness never links
`cudart` — it loads your cubin with `cuModuleLoad`, so it does not care how you
built it.

## Build

```bash
cd tools/k3-harness
/usr/local/cuda-13.1/bin/nvcc -O2 -std=c++17 -arch=sm_103a harness.cu -o harness -lcuda
./naive/build.sh                 # naive/<name>.cubin for every kernel
./baseline/build.sh              # baseline/mla_paged_attn_old.cubin
```

`ref.h` is plain C++17 (no CUDA headers), so it also compiles under a host
compiler if you want to poke at it.

## Running

```bash
export CUDA_VISIBLE_DEVICES=3          # tray03: one card each, check nvidia-smi first
./harness --kernel <name> --cubin <path> [options]
```

| option | default | meaning |
|---|---|---|
| `--kernel <name>` | — | the document's kernel name (see the table below) |
| `--cubin <path>` | — | any cubin exporting the documented entry |
| `--B N` | 1 | batch rows; the acceptance set is 1, 2, 8, 64 |
| `--span N` | 0 | K2/K3: rows [1, 1+N) are a span the kernel must skip (`span_at` = 1); their outputs and line bytes are not compared, every other row is |
| `--ctx N` | 2048 | K5 context length (max seq_lens); up to 32768 |
| `--nb N` | 4 | attnres candidate blocks, 0..8 |
| `--snapshot 0\|1` | 1 | K1a/K1b snapshot flag |
| `--two 0\|1` | 1 | K1c: add `p2` as well (`two == 0` is the dense layer) |
| `--nmla N` / `--layer K` | 1 / 0 | MLA layers in the slab; `page_stride = nmla*64*576`, `layer_off = K*64*576` |
| `--reps N` | 50 | timed launches |
| `--seed N` | 1234 | RNG seed; the whole input set is a pure function of it |
| `--grid gx,gy,gz` | per kernel | launch grid override (`gx` must be `B`) |
| `--block bx,by,bz` | per kernel | block override |
| `--smem bytes` | 0 | dynamic shared memory override |

Exit code is `0` if every output passed, `1` if any failed, `2` on a harness or
driver error (bad cubin, missing entry, CUDA failure).

## Kernels, entries and default geometry

`--grid/--block/--smem` override the geometry for the kernels the document lets
you tile yourself; write your geometry in your `.cu` header comment and pass it
here. All the naive kernels use static shared memory, so the default dynamic
smem is 0 everywhere.

| `--kernel` | entry | default grid | default block | fixed by the doc? |
|---|---|---|---|---|
| `attnres_rms` | `kern_k3_attnres_rms` | (B,1,1) | 1024 | grid+block yes, smem yours |
| `land_add_attnres_rms` | `kern_k3_land_add_attnres_rms` | (B,1,1) | 1024 | grid+block yes |
| `land_add2` | `kern_k3_land_add2` | (B,1,1) | 1024 | yes |
| `conv_silu` | `kern_k3_conv_silu` | (B,3,24) | 128 | yours (4 columns/thread) |
| `kda_core` | `kern_k3_kda_core` | (B,96,1) | 128 | yes |
| `mla_prep` | `kern_k3_mla_prep` | (B,1,1) | 1024 | yours |
| `mla_paged_attn` | `kern_k3_mla_paged_attn` | (B,96,1) | 128 | head grouping is yours |
| `mla_paged_attn_old` | `kern_k3_mla_paged_attn` (old ABI) | (B,96,1) | 128 | the retired kernel |
| `router_topk` | `kern_k3_router_topk` | (B,1,1) | 256 | yes |
| `argmax_f32` | `kern_k3_argmax_f32_partial` + `_final` | (B,64,1) / (B,1,1) | 1024 / 64 | yes |
| `rms` | `kern_k3_rms` | (B,1,1) | 1024 | yes |
| `land` | `kern_k3_land` | (B,ceil(n/1024),1) | 1024 | yes |
| `land_situ` | `kern_k3_land_situ` | (B,ceil(n/1024),1) | 1024 | yes |
| `span_gather` | `kern_k3_span_gather` (B = span) | (24,4,ceil(B/8)) | 128 | rows per block yours |
| `span_state` | `kern_k3_span_state` | (96,32,1) | 128 | yes |
| `kda_out_gate` | `kern_k3_kda_out_gate` (B = span) | (B,96,1) | 128 | yes |

Non-`B` shapes the harness pins (they are not CLI options because the model
fixes them): `rms` h = LATENT = 3584; `land` n = 3584, off = 128, ldc = 4096
(a deliberately non-trivial `off`/`ldc` so the addressing is exercised);
`land_situ` n = INTER = 3072; `argmax_f32` n = V = 163840.

`conv_silu`'s geometry: the ABI document's prose still says `(B, 3, INNER/256)`
with block 256; the agreed delivery geometry is `(B, 3, 24)` with block 128 and
that is the harness default. The naive kernel strides its column loop, so both
launch shapes work — if yours differs, pass `--grid/--block`.

## Inputs

Deterministic from `--seed` (splitmix64 + Box–Muller), with realistic
magnitudes:

* activations (`prefix`, `blocks`, `conv_q/k/v`, `x`, latent rows) — bf16 N(0,1)
* f32 GEMM partials — N(0, 2²)
* gammas — bf16 1 + N(0, 0.1²); `gamma_o` f32 1 + N(0, 0.1²)
* attnres `sw` — f32 N(0, 0.012²) (so scores over H = 7168 land at O(1))
* KDA recurrent state `rec` — f32 N(0, 0.1²); windows bf16 N(0,1)
* conv weights `cw` — f32 N(0, 0.3²)
* `w_f_b` — bf16 N(0, 0.044²); `dt_bias` N(0,1); `a_log` N(0, 0.5²)
* `w_kv_b` — bf16 N(0, 0.025²) (keeps the MLA logits at O(1), i.e. a softmax
  with a real tail rather than a one-hot)
* router logits — f32 N(0, 2²), `bias` N(0, 0.1²), `rs` = 2.5
* KDA `line_index` is a shuffled permutation of the `B` lines, so a kernel that
  assumes `line_index[b] == b` fails
* the MLA slab is filled with random latent rows, and every `block_table` entry
  is a distinct physical page in shuffled order, so a bug in the page walk shows
* `seq_lens` varies per row when `B > 1` (row 0 is always the full `--ctx`)
* every output buffer is memset to `0x5A` before the launch, so "didn't write
  the whole buffer" is a failure, per §0

Two tie rules are exercised deliberately: `router_topk` gets an exact tie on
experts 40/41 that is guaranteed into the top-16 (the smaller `e` must win), and
`argmax_f32` gets the row maximum duplicated at two indices (the smaller index
must win).

## What PASS means

Per output buffer, against the CPU reference, using §2's tolerance:

* elementwise `|err| <= 3 * bf16ULP(|ref|) + 1e-3`, **and**
* relative RMS error `sqrt(sum err^2 / sum ref^2) <= 2e-3`

Integer outputs (`router_topk.idx`, `argmax_f32.out`) must match exactly.
In-place state is compared after the launch too: the KDA `rec` block and the
conv windows inside the line, and the slab rows `mla_prep` appends.

`argmax_f32`'s intermediate `pmax`/`pidx` are checked by invariant rather than
against a fixed split (the document fixes the grid but not which logits a part
owns): every `(pmax, pidx)` must be a real value/index pair of its row, and the
fold over the parts must reproduce the row argmax under the smallest-index tie
rule.

The printout per output is `max|err|`, `maxULP` (the elementwise error in bf16
ULPs of the reference), `relRMS` and PASS/FAIL. A large `maxULP` next to a small
`max|err|` is normal and passing — it just means the reference value there is
near zero, where the `+1e-3` term of the tolerance is doing the work.

## Timing

After the correctness check the harness warms up 5 launches and then times
`--reps` launches individually between CUDA events, printing the **median µs**,
the kernel's **roofline bytes** (the minimum traffic implied by the shapes:
inputs + outputs + state touched, counting the MLA KV rows once) and the
implied GB/s. The GB300's DRAM peak is ~8 TB/s; §2 asks memory-bound kernels for
≥ 60% of it.

Stateful kernels (`conv_silu`, `kda_core`, `mla_prep`) mutate their state on
every repetition — correctness is checked on the first launch from a clean
state, and the timing repetitions run on the drifted state. That does not change
the traffic or the work.

`nvidia-smi` before you trust a number: this tray is shared, and a co-tenant at
100% GPU utilisation roughly doubled the K5 baseline in one of our measurement
passes.

## The K5 baseline to beat

`--kernel mla_paged_attn_old` runs the retired pegainfer kernel
(`baseline/k3_mla_paged_attn_old.cu`, a verbatim copy of
`git show HEAD:tools/kernels-src/k3_mla_paged_attn.cu`) on exactly the same
inputs. Its ABI differs — bf16 `q`, no gate, no `B` — so the harness lands
`q_partial` to bf16 before the launch and applies `mul_sigmoid(mla_gate)` to its
output on the CPU afterwards, then compares against the same K5 reference.

Measured on tray03 GPU 3, idle card, `--reps 50`:

| shape | old kernel | 3× target for the new K5 |
|---|---|---|
| ctx = 32768, B = 1 | **10101 µs** | **≤ 3367 µs** |
| ctx = 32768, B = 8 | 15861 µs | — |
| ctx = 2048, B = 1 | 685 µs | must not be slower |
| ctx = 65, B = 1 | 82 µs | must not be slower |
| ctx = 1, B = 1 | 61 µs | must not be slower |

## Running your own kernel

```bash
/usr/local/cuda-13.1/bin/nvcc -cubin -arch=sm_103a -O3 \
    -o /tmp/k3_kda_core.cubin ../kernels-src/k3_kda_core.cu
CUDA_VISIBLE_DEVICES=3 ./harness --kernel kda_core --cubin /tmp/k3_kda_core.cubin --B 64
```

Then sweep the acceptance set:

```bash
for B in 1 2 8 64; do ./harness --kernel kda_core --cubin /tmp/k3_kda_core.cubin --B $B || break; done
```

and for K5:

```bash
for ctx in 1 64 65 2048 32768; do for B in 1 8; do
  ./harness --kernel mla_paged_attn --cubin /tmp/k5.cubin --B $B --ctx $ctx || exit 1
done; done
```

If your geometry is not the default, add `--grid B,gy,gz --block bx,by,bz
--smem bytes` to match the header comment in your `.cu`.

Do not edit `ref.h` or `harness.cu`. If you think a reference is wrong, write it
up in `notes/k3_<name>.md` — see "Open questions" below for the ones already
raised.

## `run_all.sh`

```bash
./run_all.sh                 # everything; table on stdout, full log in run_all.log
./run_all.sh kda_core rms    # only these kernels
BIG=1 ./run_all.sh           # also K5 at ctx=32768, B=64 (~30 s, 2.4 GB slab)
REPS=200 ./run_all.sh        # more timing repetitions
```

It sweeps B ∈ {1,2,8,64}; `nb`/`snapshot` ∈ {(0,0),(1,1),(4,1),(8,0)} for the
attnres kernels; `two` ∈ {0,1} for `land_add2`; `nmla`/`layer` ∈ {(1,0),(2,1)}
for `mla_prep`; ctx ∈ {1,64,65,2048,32768} × B ∈ {1,8} for K5, plus B ∈ {2,64}
at ctx = 2048 and a `nmla=2, layer=1` case; and the same K5 shapes for the old
baseline. Exit status is non-zero if anything fails.

## Open questions on the ABI document

These are things the harness had to decide; the references follow the document
as literally as possible. They are **not** silent fixes — raise them with the
document's owner if any reading is wrong.

1. **K3 `kr` chain is never written out.** The document gives
   `qtot = Σ bf16(q·q)` and `qr = bf16(rsqrt(f32(bf16(qtot)) + 1e-6))`, then only
   says "kr 链全 bf16". The reference assumes `k` is exactly symmetric with `q`
   (`ktot = Σ bf16(k·k)`, `kr = bf16(rsqrt(f32(bf16(ktot)) + 1e-6))`). Note the
   epsilon there is `1e-6`, not `EPS = 1e-5`.
2. **K3's output norm is not the §0 `rms()` primitive.**
   `o[d] = bf16(f32(attn[d])·rsqrt(mean(attn²)+EPS)·gamma_o[d])` has one landing
   after the `gamma_o` multiply, whereas §0's `rms` rounds before the scale and
   multiplies two bf16s; `gamma_o` is also f32 here, not bf16. Taken literally.
3. **K1a: `nb == NB_MAX` and `snapshot != 0` cannot both hold.** `blocks` is
   `[B, 8, H]`, so writing `blocks[b, nb]` with `nb == 8` is out of bounds — and
   the tail of §1 does call `attnres_rms(8)`. The harness forces `--snapshot 0`
   when `nb == 8`. Either the tail call is snapshot-free (that is the assumption)
   or `blocks` needs 9 slots.
4. **`snapshot` means two different things.** In K1a it writes the prefix into
   `blocks`; in K1b it selects `prefix2 = p` instead of `bf16(prefix + p)` and
   writes nothing (`blocks` is `const` there). Implemented as written, but the
   shared name is a trap.
5. **`argmax_f32_partial`'s split is unspecified.** The document fixes
   grid (B,64) and block 1024 but not which logits a part owns, so `pmax`/`pidx`
   are not a well-defined function of the inputs. The harness therefore checks
   the final `out` exactly and the partials by invariant. If a fixed split is
   intended, the document should name it.
6. **§0 says partials are `f32 [B, ldc]` with `n` valid columns from `off`,**
   but only `kern_k3_land` takes `off`/`ldc`; every other partial-consuming
   signature has neither. The harness passes those partials tightly packed
   (`ldc == width`, `off == 0`). If the GEMM will ever hand them a padded `ldc`,
   the signatures need it.
7. **K5 `block_table` entries past the context.** The retired kernel tolerates a
   negative page id; the new ABI does not say whether `-1` can appear. The
   harness always fills every entry with a distinct valid physical page, so a
   kernel that needs `-1` handling is not tested for it.
8. **KDA `dec` index space.** `dec[d]` is defined over the value dim `d` and then
   used as `dec[k]` in `m[dv] = Σ_k S[h,dv,k]·dec[k]·kn[k]`. The reference treats
   `d` and `k` as the same 128-wide key index, which is the only consistent
   reading, but the document switches letters mid-block.
9. **The §2 tolerance applied to the f32 `rec` state is very loose.**
   `3·bf16ULP + 1e-3` on values of magnitude ~0.1 is ~2.5% — the `relRMS` column
   is the check that actually bites there (~1e-6 for the naive kernel). Flagging
   in case a tighter rec tolerance was meant.
10. **`conv_silu`'s geometry line is stale** — the prose says `(B, 3, INNER/256)`
    block 256, the delivery is `(B, 3, 24)` block 128.
11. **K1c changed under us** (already applied): `p2` is always a valid pointer
    and a new `int two` before `B` says whether it is added. `docs/` is updated;
    `ref.h`, `naive/k3_land_add2.cu` and the harness follow it, and `run_all.sh`
    covers both `two = 0` and `two = 1`.
