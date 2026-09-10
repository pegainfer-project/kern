# Full target accuracy reproduction

`target_oracle.py` runs the checkpoint's original forty-layer `Transformer`.
It loads original safetensors directly, applying the supplied `convert.py`
partition rules in memory. No exported checkpoint or kern hidden-state tensors
are used as the reference. Unused vision and draft modules are not constructed;
target forward computation remains the supplied implementation.

The original reference uses **TP4 + EP4**. The kern serving program uses
**DP4 + EP4**, so reduction order and GEMM shapes differ. Compare numerical
errors and top-token margins rather than requiring bitwise equality.

Use a checkpoint containing its `inference/` sources and a CUDA environment
with the dependencies required by those sources. Run on four available GPUs:

```sh
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  tools/dsv41/target_oracle.py \
  --checkpoint "$CHECKPOINT" --out "$REFERENCE_OUT" \
  --tokens prompt.json --teacher-tokens teacher.json \
  --steps 96 --ar-steps 12 --context 4096
```

`prompt.json` is an integer token array, or an array of equal-length token arrays
for a batch. `teacher.json` contains only continuation tokens, in the same
format and batch order. For 96 logit rows, at least 95 continuation tokens are
required as subsequent inputs; supplying all 96 also records the last chosen
token consistently with the kern trace. The full prompt is processed once,
followed by individual decode tokens. `--ar-steps` starts a separate greedy
trajectory from the original prompt, without teacher forcing.

The default keeps weights on GPU. `--host-engram` stores each original rank's
Engram shard on CPU and transfers the results of the original embedding lookup
to GPU; original dequantization and distributed reduction remain unchanged.
This reference option is distinct from kern's single shared host table.
`--inspect-world 4 --inspect-rank 3` checks construction and checkpoint shard
shapes on the meta device without executing GPU math. It is not an accuracy test.

Each trajectory writes `trace.json`, `tokens.json`, and `target_logits.f32`.
FP32 logits have shape `[steps, batch, vocab]`. Row `j` is generated at input
position `prompt_length - 1 + j`, predicting token `prompt_length + j`.
The trace records original source hashes, runtime config, parameter memory,
input positions, predictions, and per-step elapsed time. Teacher output is in
`$REFERENCE_OUT/teacher`; independent AR output is in `$REFERENCE_OUT/ar`.

For a kern trace generated with `target_logits: true`, compare identical causal
histories and every vocabulary entry with:

```sh
python tools/dsv41/compare_target.py \
  --kern "$KERN_TRACE" --reference "$REFERENCE_OUT/teacher" \
  --out comparison.json
```

The comparison currently supports batch one. It rejects mismatched token
histories, malformed output sizes, and nonfinite logits. The report includes
top-1 agreement, reference top-two margin, maximum absolute error, RMSE,
relative squared error, and reference-to-kern KL divergence for every row.
A successful harness exit alone does not establish numerical agreement.

For layer localization, add `--prefill-hooks --steps 1`. This captures only the
first prefill through ordinary PyTorch forward hooks on the original modules.
`prefill-hooks/tensors.json` describes native shapes and dtypes. Each layer has
`attn_norm`, `attn`, `ffn_norm`, `ffn`, `h`, and `pre_mix` files, plus the initial
embedding. Captures retain original BF16 or FP32 bytes, and hooks do not replace
any inputs or outputs. The four sublayer boundaries are full-width tensors after
the original TP collectives; rank zero writes the files. The extra leading
batch dimension can be flattened when comparing batch-one kern captures.

Reference repeatability can be measured with `--diagnose-prefill`. In one
original-model construction this runs two consecutive pure prefills, another
prefill followed by three decode steps, a prefill after that decode history,
and a final prefill after explicitly resetting mutable caches. Each phase
retains its logits and hooks; `reset.json` identifies the buffers reset only
for the final diagnostic. This option does not change original forward math.
Use `compare_reference_stability.py OUTPUT_DIRECTORY` to report the reference's
own numerical variation, before attributing differences to another runtime.
