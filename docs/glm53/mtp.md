# GLM-5.3-Flash MTP, TP8 H100 sm_90a

Date: 2026-09-27. Scope: single-node TP8.

## Status and release gate

This is an **experimental correctness-first implementation**, not a measured
replacement for the decode champion. `gen.py --mtp` emits a full model manifest,
not a mock accept loop. It passes `kern verify`. The six derived layer-45 DSA
weights and all rank-local weight ranges have been checked.

GPU unit tests passed at S=1/2/4/8 for splice, prefix acceptance, compact packing,
conv verify/advance, KDA no-store/store, and hidden carry. The KDA comparison uses
the pinned original cubins, not just two new implementations. DSA prep and ordered
pool update passed boundary tests. The latest run also passed head/tie tests,
EH norm, shifted draft metadata, and a captured accept/pack graph with changing
and mixed counts. See `ops_spec.py test` and `/tmp/mtp-test-final.log`.
Head, EH and residual stores matched bitwise at M=1/2/3/4/8/12/16/24/32.
Residual-add RMSNorm had three one-BF16-ULP differences in 417,792 outputs
(2 at M12, 1 at M32); the test records this, rather than calling it bitwise.
The test limit is one ULP and <=0.1% mismatches. Full draft logits remain a gate.

**Remaining release gates:** draft stem/head numerical A/B against sglang,
M32 FA3/mHC/MoE per-op A/B, 32-row Lamport/NCCL transport A/B, greedy target
agreement against the champion manifest, prefix restore, and acceptance/latency
on the real agentic trace. A controlled TP8 smoke passed on GPUs 0-7: all 39.7
GiB/rank weights bound, rows=1 eager returned `Paris. In French`, rows=3 eager
returned `Paris. In French, Paris is spelled`, and rows=3 graph mode returned the
same text. The graph log captured `decode` with 1,200 calls and `round_k2` with
1,380 calls for all eight ranks. This is a functional smoke, not a numerical A/B
or performance result. The process was stopped after the smoke; no other process
was stopped.

The verify path deliberately uses isolated, unfused M32 ops. `--moe v2` and
`--mhc boundary` and `--kda fused` can still optimize the one-row bootstrap, but do **not** silently
port their 16-row kernels into verification. `--dsa v2a` is rejected with `--mtp`
until its state-stride and row-capacity audit is complete.

## 1. Source findings that amend arch_decode.md 4.3

Sglang root below is
`<sglang checkout>/python/sglang`.

1. **Target hidden is post-final-norm.** `srt/models/glm5_next.py:1193-1219`
   finishes the layer stack and applies `self.norm`; `:1494` sends that result to
   the logits processor. `srt/layers/logits_processor.py:802-838` captures this
   hidden when no pre-norm tensor or auxiliary hidden override is supplied.
   `eagle_worker_v2.py:1056,1169` explicitly requests
   `return_hidden_states_before_norm=False`. Kern carries `h_norm`, not `h` or a
   4-stream mHC residual. This design is EAGLE/MTP, not EAGLE3 auxiliary capture.
2. **MTP positions are shifted relative to its input token.** For target input
   `x[p]`, target hidden is `H[p]`. Draft-extend takes `embed(x[p+1]), H[p]` and
   writes at absolute position `p`. See `_draft_extend_for_prefill`, lines
   1018-1042: token IDs shift left; positions/cache locations retain target-row
   alignment. Decode preparation sets positions to `batch.seq_lens`
   (`eagle_worker_common.py:304`), and the draft loop assigns per-step
   `out_cache_loc[i]` then increments positions (`eagle_worker_v2.py:864-918`).
3. **Draft cache repair is mandatory.** `_draft_extend_for_decode` at
   `eagle_worker_v2.py:1130-1288` replaces autoregressive draft conditioning with
   verified target hidden and selects the last accepted row. Merely carrying the
   last target hidden is not equivalent. The implementation below uses a delayed
   first draft and repairs intermediate accepted cache rows.
4. **The fused KDA no-store flag is `DISABLE_STATE_UPDATE=True`.**
   `kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py:38-373` has no
   `STORE_FINAL_STATE` argument. That name belongs to a different recurrent
   kernel. Offline mining keeps the original alignment attributes, PDL code,
   warp count and compiler options. Removing stride divisibility attributes
   initially changed FP32 state bits although BF16 outputs matched. Restoring
   those attributes made both state and outputs bit-identical in the unit test.
5. **Layer 45 is mixed precision, not entirely BF16.** Inventory: 1,760 tensors:
   17 BF16, 871 F8_E4M3 and 872 F32. EH, norms, router and indexer are BF16;
   DSA q/kv/o projections and MoE matrices have FP8 weights and scales. The
   `glm5_next_nextn.py` BF16 override applies only if the config ignores
   `model.layers.45.*`; this checkpoint has individual ignore entries, not that
   wildcard. Do not inherit a blanket BF16 draft policy from the comment.
6. **`gdn_advance.cu` is not directly reusable.** Its history is already after
   the anchor, so it shifts by `nacc-1`. Here verify stores neither conv nor SSM
   state. Commit must consume **nacc** rows from the pre-anchor state.
7. **Prefix sums alone do not compact inputs.** If acceptance is `[1,3]`, new
   `cu=[0,1,4]` over unchanged sequence-major `[3S,...]` data would read rejected
   rows from sequence 0 for sequence 1. `spec_pack_kda` packs all accepted
   prefixes before the store-enabled delta pass.
8. **MTP index sharing is enabled in this checkpoint.**
   `config.json:text_config.index_share_for_mtp_iteration=true`;
   `deepseek_v2.py:2055-2057` sets NextN `skip_topk/next_skip_topk=true`.
   `forward_mla.py:155` runs the indexer only when no seed exists.
   `eagle_worker_v2.py:374-389,837-841` retains the draft-extend seed.
   `index_topk_share.py:24-30,79-97` carries it. In this delayed design,
   draft0 computes that seed. Draft1 still rebuilds row/page/length metadata for its new position, but skips
   the complete indexer and pool update. It reuses the already expanded physical
   token indices **including the seed tail**.
   This is not a fresh top-k or an append of the new token.
   `dsa_backend.py:853,3663+` and
   `dsa/dsa_backend_mtp_precompute.py:_precompute_decode_mode` build draft-decode
   attention lengths from the committed `seq_lens`, not from incremented draft
   positions. Kern rebuilds row metadata for the new position while retaining
   the shared physical index seed.
   Its latent KV still writes at position p. The generator checks the flag.
9. **Residual-add RMSNorm cannot be split at a BF16 sum.**
   CUDA `layernorm.py:583-610` calls `sgl_kernel.fused_add_rmsnorm`;
   the installed `sgl_kernel/elementwise.py:152-162` delegates to FlashInfer.
   `flashinfer/norm.cuh:FusedAddRMSNormKernel` uses the FP32 sum for both variance
   and normalized output, while storing a BF16 residual. `spec_add_norm`
   follows that arithmetic. Its small measured rounding difference is stated
   above. `spec_eh_norm` follows the pinned fused-EH vector/reduction layout.

Existing `tools/kernels-src/spec_round.cu` provides correct splice/prefix-match
semantics, but has no validity/compaction/pool-keyed carry. New kernels include
those interfaces explicitly. Neither that file nor `gdn_advance.cu` was edited.

## 2. State and row layout

| Object | Layout / size | Change |
|---|---|---|
| kv | 12,288 B/token, 12 x 512 BF16 | layer offset `i*1024`, slot stride 6144 BF16 |
| idx | 396 B/token; page 101,376 B | 12 x 8,448 B; scales at layer +8,192 |
| idx_tail | 12 x 4,096 B/sequence | eight entries per layer, not three |
| kda_conv | 34 x 18,432 B/sequence | unchanged committed history |
| kda_ssm | 34 x 524,288 B/sequence | unchanged committed state |
| mtp_hidden | 8,192 B/sequence | post-norm target hidden; pooled, copied on restore |
| spec_F/Q/A | [34,32,3336/3072/1024] BF16 | all-layer verify inputs, local to a round |
| pack_F/Q/A | [32,3336/3072/1024] BF16 | reused for one advance layer at a time |

**Do not put hidden carry in a batch-indexed `kind:carry` buffer.** Batch seats
move after completion/admission. The extra per-sequence state and line table
follow each lease. Round-local input carries may be workspace: they do not
survive a host step.

Keep `seqs.max=16`, hence the original 64-byte line-table row pitch. Set
`tokens.max=32`. Decode accepts 16 sequences x1; round accepts at most 8 x3=24
rows. All activation capacities, FP8 A/D/SFA descriptors and mHC partials needed
by verify have 32-row variants. MoE has 288 pair rows and 18,432 sorted slots.

**Router decision:** two existing M=16 tiny-GEMM launches, at input and output
row offsets 0 and 16. This preserves the tested per-row reduction
and covers all 32 allocated rows, including TMA padding. A new M=32 entry could
save one launch per MoE layer, but it has not been measured or validated here.
This is a correctness/risk decision, not a claim that two launches are faster.
Draft passes keep separate M16 ops. No read-only op module's globals or files
are mutated.

FA3 scheduler scratch arrays `nsd/nmb/vbi/sem` grow from 8 to 32. Partial-output
and LSE strides use the variant's static row pitch. Runtime FA3 batch count is
24 or less, below its 31-batch limit. DeepGEMM metadata receives 24 or less,
below its 32-row limit. A separate 32-row Lamport allocation is necessary:
merely widening the activation buffer would overflow the original 16-row peer
payload. The two collectives share one sticky `Fill::Error` word.

## 3. Round DAG and shifted draft cache

At entry: `p` target tokens are committed, `anchor=x[p]`, and the sequence's
hidden state is `H[p-1]`. Every real sequence must finish its prompt first.

```
load H[p-1] from per-sequence state
  -> draft0: EH(anchor,H[p-1]) at draft position p-1 -> d1, draft hidden
  -> draft1: EH(d1,draft hidden) at position p, reuse draft0 top-k/lens -> d2
  -> splice [anchor,d1,d2], target positions [p,p+1,p+2]
  -> target verify: all 45 layers, 3S rows
       KDA conv + delta: no committed-state stores
       DSA KV: 3 rows; ordered kpool per sequence; per-row queries
  -> greedy prefix count; tokens [S,3], nacc [S]; compact cu=sum(nacc)
  -> for each KDA layer: pack -> delta with store -> conv advance by nacc
  -> save target H[nacc-1] in the sequence's hidden state
  -> repair draft cache rows j < nacc-1:
       EH(verify_token[j], target_H[j]) at position p+j
       -> input norm -> latent KV and index key/gate -> cache stores
```

The final accepted draft-cache position is computed at the next round's
`draft0`. This is the same shifted conditioning as sglang, but moves its final
full draft-extend layer into the next round. Intermediate repair needs only the
**cache-producing stem**. In this one-layer decoder, KV/index keys are functions
of EH plus input norm, before attention or MoE. Repair therefore omits queries,
attention output, MoE and the head. It is not an approximation of those caches.

During one-row bootstrap, repair position `p-1` with the actual current prompt
token and the stored previous target hidden, then run target decode and save its
hidden. At `p=0` draft stores are invalid/masked. This avoids filling the draft
prompt cache with target predictions in place of actual prompt tokens.

### Rollback and prefix safety

- Target and draft rejected paged entries are overwritten on the next round.
- Each kpool CTA processes the sequence's rows in order and synchronizes before
  the next row. A close at row j sees all earlier ring writes.
- Pooled entries from rejected rows cannot be read before their pool is valid:
  query pool count is `floor((position+1)/4)`.
- Eight tail entries retain the prior three committed taps plus speculative
  writes. Repaired rows overwrite the same ring locations.
- Conv and SSM commit only accepted prefixes. No per-token state snapshots.
- **Shifted-cache exception to arch 4.3:** a restored request may write `p-1`.
  At a non-aligned prefix this is in the copied partial page. At `p%256==0`
  it is in an immutable shared page. Scheduler `finish` now skips such snapshots
  for replicated-round manifests, even under `--rows 1`. This loses that prefix
  hit, but avoids cache corruption. Removing the guard requires a pool/runtime
  writable-lookbehind API or a different cache layout; simply changing `pos`
  or restoring recurrent state at `p-1` is incorrect.

## 4. Kernel / op table

| New entry | Launch | Contract |
|---|---|---|
| spec_draft_meta | S CTAs | positions p-1/p, local physical slots, pad validity |
| spec_splice | S CTAs | group anchors plus two draft IDs -> sequence-major rows |
| spec_accept | one CTA | prefix match, bonus, device nacc and compact cu |
| spec_pack_kda | S x3 CTAs | pack F, conv QKV and forget for accepted prefixes |
| spec_conv_verify | S x12 CTAs | three causal outputs; reads committed history only |
| spec_delta_verify | 4 xS x8 CTAs | mined varlen KDA, state update disabled |
| spec_delta_advance | 4 xS x8 CTAs | same recurrence over packed accepted rows; stores |
| spec_conv_advance | S x12 CTAs | shift committed history by nacc, overlap-safe |
| spec_dsa_prep | 3S CTAs | row length, replicated BT, pool lengths, FA3 cu |
| spec_kpool_update | S CTAs, 128 threads | ordered rows=1/3, explicit page pitch |
| spec_kv_store | query-row CTAs | validity guard, explicit slot pitch/layer offset |
| spec_carry_load/store | S CTAs | pool-line keyed target hidden |
| spec_repair_meta | S CTAs | successor IDs, mask `j<nacc-1` |
| spec_eh_norm | row CTAs, 512 threads | two RMS reductions and EH concatenate |
| spec_add_norm | row CTAs, 512 threads | FP32 residual sum -> RMS; BF16 residual store |
| spec_head_local/global | row CTAs | rank-local argmax, then 8-way global argmax |
| spec_ar_lamport | 16 CTAs | 32-row TP8 peer payload, separate phase storage |

The head gathers an eight-byte `(f32 max,i32 global id)` per sequence/rank using
four BF16 transport lanes. All-gather copies the bits; it does not reduce them.
Tie-breaking selects the lowest global token ID. EH projection and other BF16
GEMMs retain the existing cuBLASLt extern. EH and residual-add norms have private
kernels; input/latent norms retain the existing norm kernel. Full draft numerical
A/B remains a gate even though the norm unit tests pass their stated limits.

Three different length arrays must stay distinct:

- KDA verify: host `[0,3,6,...,3S]`.
- FA3: device `[0,1,2,...,3S]`, because each row is a separate sparse query.
- KDA advance: device exclusive prefix sum of `nacc`, with packed inputs.

## 5. Protocol, scheduler and graphs

`Protocol::check` retains replicated mode's ban on tray axes, `blocks` and
`batch.span`. It now allows fixed-width rounds if they emit `[groups,rows]`
tokens plus `[groups]` count and have a one-row token-emitting bootstrap forward.
Variable-length replicated chunks remain unsupported. This supersedes the
one-row-only sentence in `tp_replicated.md`; no schema/type change is required.

`Plan::check` chooses the widest shape by default (`--rows 3`). Without a chunk
program it uses the one-row forward for prompt bootstrap, with chunk size one.
It completes the prompt before scheduling any round. Admission reserves two
extra token slots, as the existing headroom policy requires. The tray keeps S
logical cells on each TP member and stages 3S positions/slots/valid values,
`seq_lens=p+3`, and per-rank page/line mappings. Only the leader reads outputs.
It rejects a staged shape that exceeds the token-row bound. Padding slots are
now consecutive within the pad page, not the same slot repeated three times.

Graphs are per program and bucket: `(S, rows=1)` and `(S, rows=3)`. S is padded to
1/2/4/8 for rounds. All launch counts, grids and collectives are fixed by that
shape. Device nacc changes masks, compact addresses and the recurrent loop
length, not the graph DAG. No host reads nacc until the whole round completes;
there are no conditional graph nodes or host-selected advance paths.

## 6. Acceptance and timing model

Greedy chain verification gives
`A = E[nacc] = 1 + alpha1 + alpha1*alpha2_given_1`.
The older example .8/.65 yields **2.32**, not 2.4. The supplied agentic sglang
measurement A=3.00 is the right target, but it is not a kern measurement and is
not proof of acceptance on other prompts or numerics.

User-supplied measured baseline: sglang-EAGLE latency per emitted token is
2.78/3.10/3.44/6.46 ms at S=1/2/4/8. The following are **conditional budgets**,
not measured results or a forecast for the initial unfused verify implementation:

| S | Assumed complete round budget (ms) | R/3 (ms/token) | sglang EAGLE | Largest R that beats it at A=3 |
|---|---:|---:|---:|---:|
| 1 | 6.4 | 2.13 | 2.78 | 8.34 |
| 2 | 7.2 | 2.40 | 3.10 | 9.30 |
| 4 | 8.7 | 2.90 | 3.44 | 10.32 |
| 8 | 15.0 | 5.00 | 6.46 | 19.38 |

Only the 6.4 ms round hypothesis is supplied in the task. The other budgets
are explicit planning targets, not extrapolated GPU measurements. Every R must
include two draft layers/heads, 3-row target verify, acceptance, KDA packing and
advance, hidden carry, and cache repair. At A=2.32, 6.4/A=2.76 ms/token: most of
the bs1 margin vanishes. Report R and A separately. R/A is latency per sequence;
aggregate throughput at batch S is `1000*S*A/R`, not `1000*A/R`.

**Measured KDA advance is not ~0.1 ms in this implementation.** A short GPU0
microbenchmark captured all 34 layers (102 launches: pack, delta, conv), with
distinct per-layer states and 20 warm replays. It excludes draft, target verify,
TP collectives, carry and repair. Results in `advance-bench.json`:

| S | nacc=1, ms | nacc=3, ms |
|---|---:|---:|
| 1 | 0.458 | 0.693 |
| 2 | 0.658 | 0.927 |
| 4 | 0.781 | 1.007 |
| 8 | 0.839 | 1.057 |

These are single-rank component measurements, not end-to-end predictions. Warm
cache and GPU clocks affect them. In particular, the 6.4 ms total bs1 budget
leaves only 5.71 ms for everything else; it is aggressive. At bs8 this measured
commit alone consumes 0.35 ms/emitted token at A=3. The initial unfused M32 target
is **not** a demonstrated performance win. Porting tested v2 blocks and reducing
commit cost are separate gates. If recompute cannot meet the budget, evaluate
capturing intermediate SSM states during verify and selecting the accepted one;
this changes the design/memory tradeoff and is not implemented here.

## 7. Build and validation

All commands run from the kern repo root:

```sh
export PYTHONPATH=tools
PY=<sglang-python>  # the sglang venv's python
$PY -m glm53.ops_spec build       # nvcc + offline Triton AST; no GPU context
$PY -m glm53.ops_spec prepare     # new mtp-derived.safetensors; no original edits
$PY -m glm53.ops_spec cpu         # default unchanged + MTP wiring + all weight slices
python3 tools/glm53/gen.py --mtp --bundle $GLM53_SPEC_ARTIFACTS/bundle
cargo build --release -p kern-serve
target/release/kern verify examples/glm53-flash-mtp.json
```

Compiler: `/usr/local/cuda-13.0/bin/nvcc -cubin -arch=sm_90a -std=c++17 -O3
-ccbin /usr/bin/g++-14`. `build.json` pins sources, cubins and compile commands.
Triton mining extracts only the JIT function AST from the oracle source; it does
not import sglang's module-level device probes. All runtime scalar alignment
hints match the captured kernel; T remains non-specialized.

Use the production launcher's `LD_LIBRARY_PATH`, but do not execute
that launcher just to obtain its environment. GPU unit command:

```sh
export LD_LIBRARY_PATH=<nccl + cuda-compat libs for your node>
# This environment has flashinfer 0.6.18 with flashinfer-cubin 0.6.3.
# Test-only override: norms use JIT source, not the mismatched attention cubins.
FLASHINFER_DISABLE_VERSION_CHECK=1 CUDA_VISIBLE_DEVICES=0 $PY -m glm53.ops_spec test
CUDA_VISIBLE_DEVICES=0 $PY -m glm53.ops_spec bench-advance
```

The test checks tmux, processes and nvidia-smi before CUDA initialization. It
refuses active bench/serve processes or a busy selected GPU. Never stop another
agent's process to obtain a test slot.

### CPU/build results

- `kern verify`: PASS; protocol example: PASS (2 forwards, 11 fills).
- TP8 smoke: PASS for eager rows=1 and eager/graph rows=3. Graph capture used
  1,200 decode calls and 1,380 round calls per rank. Smoke counters for the
  short deterministic prompt were A=2.0 (eager) and A=2.0 (graph), with
  `accept_pct=50%`; scheduler step averages were 14.27/12.59 ms in eager and
  34.79 ms in graph mode. These include prompt/capture effects and are not a
  batch or agentic benchmark.
- `ops_spec cpu`: checks default generation before/after MTP in one process,
  private M32 wiring, top-k reuse and every weight dtype/shape/range at all ranks.
- `cargo test -p kern-manifest`: 106 tests passed, including two round tests.
- `cargo test -p kern-serve replicated_`: six tests passed. New staging test
  covers 3-row positions/slots/cu, group anchors and padding on all ranks with
  unequal allocator histories. No GPU is involved in these Rust tests.
- `cargo check -p kern-serve --tests` and release build: passed; the live server
  was not restarted. Binary: `target/release/kern-serve`.
- Existing `tools/glm53/test_cpu.py`: 4/5 pass. Its assertion that FA3 PDL is off
  fails against the owner's current PDL-on examples. No champion manifest or
  read-only test was edited to hide this mismatch. Default before/after MTP
  equality passes independently. Private FA3 variants respect `GLM53_FA3_PDL=0`.

### Full model and agentic acceptance plan (not executed)

1. Check `tmux ls`, pane commands and `nvidia-smi`. Wait for kbench/kserve and all
   affected GPUs. Run a short `--eager`, `--max-seqs 1`, `--rows 1` smoke first.
2. Start the experimental server on a separate unused port, using the MTP
   manifest and the persistent bundle. The load and one request already pass;
   repeat after any kernel or scheduler change:
   `kern-serve --manifest examples/glm53-flash-mtp.json --kernels
   $GLM53_SPEC_ARTIFACTS/bundle --weights weights/GLM-5.3-Flash
   --gpus 0,1,2,3,4,5,6,7 --capacity 8192 --max-seqs 1 --rows 3 --port 8008`.
3. Compare exact generated token IDs against target-only kern with greedy
   settings. Use prompts ending near 3/4, 7/8, 255/256, 2047/2048/2051, with
   rejection at draft 1 and draft 2. Include EOS and max_tokens inside a round.
4. Test batches 1/2/4/8 and a five-request batch padded to eight. Change batch
   membership between rounds. Confirm every rank agrees on tokens/counts.
5. Compare eager and CUDA graph replay. Change nacc in the same captured graph
   (all 1, all 2, all 3, mixed) without recapture. Verify KDA committed state
   against sequential target decode after each round.
6. Test retire/restore and host park/wake. Non-aligned prefixes must hit and
   match fresh execution. Aligned prefixes must miss safely under the guard.
7. Run the real text-only agentic prompt trace, same tokenizer/template and
   deterministic request order as the supplied sglang measurement. Do not use
   synthetic repeated tokens to report acceptance. Warm all graph buckets;
   exclude prefill, load and capture from decode timings.
8. Record rounds, count histogram (1/2/3), draft-1 matches, conditional draft-2
   matches, emitted tokens, GPU round ms and wall-clock decode ms. Report
   `A=sum(nacc)/rounds`, `alpha1=P(nacc>=2)`,
   `alpha2_given_1=P(nacc=3)/P(nacc>=2)`, R/A and p50/p95. The existing per-position speculative counters permit the histogram:
   `h1=num_drafts-accepted_pos0`, `h2=accepted_pos0-accepted_pos1`,
   `h3=accepted_pos1`. Per-window `accepted` logs already report A. Use device
   events/profiling for GPU time; scheduler `step_ms` includes host work. Increase `--capacity` only for the real long-context run.

## 8. Integration risks and next cuts

- Hypatia's current `ops_kda_v2.py::fuse_manifest` explicitly requires rows=1.
  Its fused core stores conv/SSM. Do not reuse it for verify without a no-store
  mode and persistent raw inputs. This implementation uses separately mined
  original delta arithmetic; projections and output norm remain unchanged.
- DSA-v2 needs per-query metadata, ordered per-sequence pools, twelve-layer
  pitches and <=24-query FA3 scratch. A decode-only fused kv_store cannot be
  reused with its baked 11-layer stride or without validity handling.
- mHC/MoE v2 row limits, scratch and peer payloads must be audited before
  widening. In particular, MoE num_valid is 9*T for target verify, not 9*S.
- EH unit parity is bitwise; residual-add norm has the small measured differences
  above. Full draft EH projection, attention, MoE and head logits remain untested.
  Index sharing now matches the checkpoint's seed mode; do not compare against
  a sglang run that disables it without a new acceptance baseline.
- Long-context sglang top-k can be nondeterministic past 2051. Separate kernel
  numerical tolerances from exact target-token agreement and acceptance.
- New state layout invalidates old checkpoints. Do not hot-swap this manifest
  into an existing tray. Rebuild does not restart the live server.
- Artifacts and the derived-weight link now use the persistent directory
  `$GLM53_SPEC_ARTIFACTS (default `glm53-artifacts/mtp`)`. `GLM53_SPEC_ARTIFACTS` can override it.
  The manifest references its content-addressed `bundle/`; the prior `/tmp`
  unit logs are evidence only, not runtime dependencies. Changing the artifact
  directory requires an explicit move of the owned derived-weight link; prepare
  refuses to replace an unrelated link/file. No original weights/index changed.

## 9. Post-review correctness and fusion gates

The MTP correctness comparison must use the same target manifest. The current
MTP manifest is intentionally `--dsa v1 --moe v1 --kda off`; the champion is
`--dsa v2a --moe v2 --kda fused`. Therefore, MTP rows=1 versus the champion is
a v1/v2a and fusion baseline, not an MTP round-path gate.

The decisive same-base gate is rows=3 versus rows=1 with the MTP manifest. On
8 greedy prompts at 128 generated tokens, this matched exactly: 8/8 prompts
and 100% token agreement. A max-seqs=1 wave run also matched rows=1 8/8.

For every MTP change, record the carry/index sentinel:

* Group the first-draft acceptance `p1` of round `t` by total `nacc` of round
  `t-1` (1, 2, and 3).
* The three rates must be flat within 2%.
* Track `p1` and `p2|1` per fusion commit. A change in either rate is an
  early numerical or state-alignment warning, before text mismatches appear.

The 2.54--2.59 mean acceptance measured on the parity prompts is not a gap
against sglang's 3.00 result. The sglang result uses a repetitive 141k-token
agentic trace. Acceptance must be measured on that same shared-prefix
workload before comparing implementations.

Review planning estimates place the single-sequence round floor near 7.4 ms
(about 6.7 ms fused three-row verify, 0.5 ms drafts, and 0.15 ms accept/carry/
repair). The current 12.8 ms step leaves about 5.4 ms of launch overhead. The
listed pair fusions may reduce this to roughly 9.7--10.7 ms, or 3.8--4.2
ms/token at A=2.55, still above the warm sglang 3.1 ms/token target. The
winning target is a roughly 60-call round graph, which requires per-layer
megakernel work, not only op-pair fusions. At bs8, a 9.4 ms floor needs A at
least 3.0 and a 24-row fused-champion MoE cost no more than 1.1 ms over the
8-row path. These are planning estimates, not measurements.


## 10. Same-base gate and absorb-only v2a audit

The saved same-manifest comparison is now decisive. `rows=3` versus `rows=1`
used the MTP manifest on the same eight greedy prompts and 128-token limit.
The captured outputs are in `glm53-mtp-artifacts/gate-b-captured.json`. The
comparison is 8/8 exact and 100% re-tokenized token agreement for every prompt.
The max-seqs=1 wave capture is also 8/8 exact. This clears the round-path
correctness gate for the captured workload. The MTP-versus-champion result is
not this gate: the champion uses DSA-v2a absorb, MoE-v2, and fused KDA, while
the tested MTP base uses DSA-v1, MoE-v1, and separate KDA.

The absorb-only v2a audit is a private candidate, not the production MTP
manifest. `mtp-v2a.json` passes `kern verify`. It changes only the K/V absorb
GEMM definitions for the 16-row bootstrap and the 32-row draft/verify paths:
`dsa_w_kc`, `dsa_w_vc`, and their `mtp16_`/`mtp32_` copies. The v2a kernels do
not access KV, index, tail, or KDA state. Candidate state layout stays:

* KV: 12 x 1024 = 12288 bytes/token;
* index: 12 x 33 = 396 bytes/token;
* index tail: 12 x 4096 = 49152 bytes/sequence.

The fixed act-quant extent remains `M=32*rows`; in verify it is `32*3*seqs`.
The candidate still uses the MTP per-row DSA cache operations and the MTP
state pitches. It does not enable the separate full DSA-v2 cache fusion. GPU
parity and acceptance are still required before promotion.

The 141k measurement remains pending the owner GPU lock. The measured kern
prefill rate is about 143 tokens/s, or about 16 minutes for a 141k-token warm
request. This is a major TTFT gap versus a warm sglang radix hit (about 0.5 s).
The champion `--host-gib 48` snapshot-restore result is also pending; use
`prefix_hit_tokens` to validate it.
