# DSpark acceptance: comparable evidence and conditional oracle

**Reference reproducibility is unresolved.** A later run produced different
original DSpark proposals at anchor 16 even though its target taps through
position 51 and token prefix are byte-identical to the first run. Neither
reference acceptance mean below currently constitutes a precision pass. Keep
both artifacts; investigate loading, initialized state and original kernel
execution before treating the comparison as stable.

The same old anchor 16 was subsequently tested in four independent processes:
plain execution 1, diagnostic hooks, explicit cache zeroing, and plain
execution 2. Plain 1 proposed `[270,36406,14,270,1840]`; the other three
proposed `[36406,14,9120,270,2900]` (the unchanged anchor is omitted).
Repeating the anchor within each process gave byte-identical logits. All
205 common loaded parameter hashes, including selected lazy experts, matched
between the differing executions; source hashes also matched. Plain 2 recovered
without hooks or extra zeroing, so this does not establish cache zeroing as a
fix. `oracle_stability.py --variant plain|hooks|zero-cache` preserves these
diagnostic alternatives without modifying the model or deleting its cache.

There is no established numerical acceptance threshold for this checkpoint's
five-proposal, greedy, non-thinking configuration in the supplied material.

The supplied **DeepSeek V4.1 Technical Report**, §2.4.3 (printed pages 13–14),
describes three DSpark blocks, a 128-token window, five predictions, sequential
Markov correction and confidence-guided adaptive verification. It does not give
a matching greedy acceptance benchmark with a specified workload and hardware.
Its §3.2 describes 15 prefill and 11 decode operators, not an acceptance target.

The authors' [DSpark paper](https://arxiv.org/html/2607.05147v1), §4.1, evaluates
other models with seven proposals, temperature 1 and non-thinking prompts.
Section 5 discusses Flash/Pro **preview** production deployments with adaptive
verification. Its verification length is a scheduling budget, not accepted
length. These experiments cannot define a pass threshold for V4.1 Flash with
five fixed greedy proposals. The paper is useful for the mechanism and the
definition that accepted length includes the target/bonus token.

The supplied `inference/generate.py` performs autoregressive generation.
`Transformer.forward_spec` implements the original draft, but there is no
supplied acceptance/rollback serving loop. Calling the original target on six
verification rows at a nonzero position is also unsuitable: its ratio-two
compressor assumes a single decode row. We therefore use canonical AR target
tokens for the first conditional acceptance comparison.

## Reproducible comparison

1. `ar_trace` loads the same unmodified serving manifest and checkpoint as the
   server, with one shared `HostWeights` scope and four Runtime instances. All
   four DP ranks receive the same prompt, run the declared once programs,
   prefill and plain decode. Page and compressor line IDs come from actual
   Runtime leases. Each step checks cross-rank token equality.
2. Rank zero writes the canonical token IDs and BF16 target taps concatenated
   from layers 37, 38 and 39. A tap row is associated with the **processed input
   position**, not the newly predicted output position.
3. `conditional_acceptance.py` calls `moe/dspark_oracle.py`, which loads the
   original three DSpark blocks and quantized checkpoint experts on demand.
   It seeds the context, then calls the original `Transformer.forward_spec`
   at every AR position to keep its original caches current.
4. For an anchor at position `a`, compare its five proposals against canonical
   IDs `a+1 .. a+5`. If the matching prefix has length `k`, the accepted length
   is `1+k`, in `[1,6]`. Advance the next evaluated speculative round by that
   accepted length. Report the prefix and total separately, plus histogram
   and prefix survival. Exclude incomplete tails and EOS crossings.

This measures **original DSpark conditioned on kern target taps and history**.
It can isolate draft implementation or scheduling discrepancies. It does not
validate the target model independently, and it is not a throughput benchmark:
the reference advances every AR position, including positions a speculative
scheduler would skip, and includes Python overhead and first-run compilation.
Small greedy logit differences can change histories; comparisons must use the
same canonical history before interpreting aggregate acceptance differences.

## First measured comparison

The English autumn-leaves prompt used the official chat encoder, 16 input
tokens, temperature zero and no thinking. A 72-token canonical trace yielded
67 eligible anchors. Running the draft prefix did not change any of its 88
prompt-plus-generation token IDs. Original DSpark and kern each took 23 rounds
with mean accepted length **2.913043** on that canonical history.

This agreement is not tokenwise equality. All five proposals matched at 45/67
anchors. At draft depths 1 through 5, disagreement counts were respectively
**1, 3, 11, 12, 19**. Among the 22 differing anchors, the first differing depth
counts were **1, 2, 8, 3, 8**. The matching prefix against the target was equal
at **65/67** anchors: at anchor 26 kern matched three proposals versus the
reference's two; at anchor 40 kern matched two versus the reference's three.
The other proposal differences occur after both drafts have already disagreed
with the target and therefore do not change that anchor's accepted prefix.

Accepted-length histograms for lengths 1–6 were `[3,4,11,3,1,1]` for the
reference and `[3,4,12,1,2,1]` for kern. Their equal mean on this small example
is an observation from that execution, not a reproducibly established result,
a general model threshold or proof of full-model numerical accuracy. The reference run including loading
took 6.35 seconds with the existing compiled-kernel cache.

`prepare_acceptance_suite.py` prepares independent English explanation,
Chinese explanation and Python-code prompts using the official encoder. This
extends the sample without mixing histories; each case gets its own canonical
target trace and both draft measurements.

## Tools

Build `ar_trace/Cargo.toml` with Cargo. Its JSON configuration contains
`manifest`, `kernels` (the staged serving cubin directory), `weights` (four
lists of safetensor files or directories), `prompt_ids`, `generation_tokens`
(at least six), `capacity_tokens`, and `output`. The output directory contains
`trace.json`, `tokens.json`, and little-endian `taps.bf16` of shape
`[tap_rows,15360]`. The diagnostic trace has fixed length; it records this
explicitly and the evaluator excludes EOS boundaries.
With `target_logits: true`, `target_logits.f32` stores FP32 logits of shape
`[generation_tokens,129280]`. Row `j` belongs to processed input position
`prompt_length-1+j` and predicts token `prompt_length+j`. Intermediate prefill
chunks do not create logits rows. This permits an independent original-target
teacher-forced comparison when available.
Optional `teacher_tokens` supplies exactly `generation_tokens` continuation
IDs. The harness advances that canonical history while recording the actual
target predictions separately in `actual_argmax.json`; `trace.json` marks
`teacher_forced: true`. Such a trace is for target numerical comparisons.
It cannot establish greedy draft acceptance when its teacher tokens differ
from the new target's argmax; `conditional_acceptance.py` rejects it.

Set `draft_proposals: true` to run the prefix of `round` before its labelled
`splice_verify` boundary after each target step. It stages the serving six-row
metadata, records `kern_proposals.json`, and then advances ordinary AR. Future
draft KV writes are replaced by the following target context publications.
Compare the resulting canonical IDs against the trace without drafts to check
that this instrumentation did not change the target history.

Run `conditional_acceptance.py --checkpoint CHECKPOINT --trace TRACE_DIR
--output RESULT_DIR`. It produces per-position JSONL and `acceptance.json`.
`--eos-id` can be supplied repeatedly; otherwise it reads the checkpoint
tokenizer. `--max-positions` supports a short initial check. Run the reference
after releasing the full target runtime to avoid overlapping allocations.
Pass `--kern-proposals TRACE_DIR/kern_proposals.json` to compare both drafts
against the same target history, reporting separate round paths and exact
five-proposal agreement. Do not compare a single-prompt reference mean against
an aggregate HTTP acceptance number from a different prompt mix.

Expected execution time is not a measured claim: target loading should be
similar to the existing serving load; a 64-token AR trace adds 63 decode calls.
The first reference execution additionally compiles original TileLang kernels
and lazily loads routed experts. Record actual load/JIT and execution times
before using this tool to plan a larger prompt suite.
