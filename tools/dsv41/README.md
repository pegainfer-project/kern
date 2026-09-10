# DeepSeek V4.1 Flash

The generator emits one-process DP4/EP4 serving programs: `load`, `prefill`,
`decode_batch`, and a five-draft/six-verify `round`. It binds the original
safetensors checkpoint directly, including shared mapped-host Engram tables.
Required weight permutations and scale packing run once during loading.
There is no offline weight export.
Immutable host weights use aligned 512 MiB transparent huge pages, followed by
portable CUDA registration. Allocation fails if `AnonHugePages` does not cover
the new mapping; small-page fallback is a large random-lookup regression.
Physical mappings include huge-page padding, while tensor lengths stay unchanged.
O-A weights are dequantized to BF16 once, matching the supplied inference;
its grouped projection consumes BF16 attention output without adding FP8
activation quantization. Build `attention/build_woa.sh` with the other kernels.

Kernel builds and upstream pins are documented in [attention](attention/README.md),
[MoE](moe/README.md), and [the integration log](../../docs/deepseek-v41-kernels.md).
Build those artifacts before generating a manifest. Generate tokenizer/hash and
RoPE constants with `auxiliary/engram_constants.py`, using its GPU reference mode.

Given those built artifacts:

```bash
python3 tools/dsv41/gen.py \
  --checkpoint "$CHECKPOINT" \
  --constants "$CONSTANTS/engram_constants.json" \
  --cubin-dir "$MOE_CUBINS" \
  --auxiliary-cubin "$AUX_CUBIN" \
  --attention-dir "$ATTENTION_CUBINS" \
  --head-cubin "$HEAD_CUBIN" \
  --copy-cubin "$COPY_CUBIN" \
  --spec-cubin "$SPEC_CUBIN" \
  --capacity 128 --max-seqs 16 --context 32768 \
  --out model.json --bundle model-cubins
kern verify model.json
```

`--capacity` bounds live rows, including six verification rows per sequence;
`--context` bounds each sequence's logical page table. Physical state TMA spans
are resolved from the runtime allocation. `--bundle` copies cubins by their pinned
hash, so subsequent kernel builds cannot change an existing serving bundle.

Configure a local target in `kern.toml`:

```toml
[targets.dsv41]
manifest = "model.json"
kernels = "model-cubins"
weights = ["/path/to/original/checkpoint"]
```

Build the independent server with
`cargo build --release --manifest-path crates/kern-serve/Cargo.toml`.
Install `kern-serve` beside `kern` or on PATH, or select it through `KERN_SERVE_BIN`.
The execution environment must expose its GPUs and IMEX channel for peer memory.

```bash
kern server dsv41 --gpus 0,1,2,3 --capacity 32768 --chunk 128 \
  --max-seqs 16 --rows 6 --renderer deepseek_v41 --port 8000
```

Use `--rows 1` for plain decode. Both modes update DSpark context. The native
V4.1 renderer supports system messages, multi-turn conversations and thinking;
`chat_template_kwargs: {"thinking": false}` selects ordinary chat.
Numeric reasoning effort can be passed through `chat_template_kwargs`.
V4.1 tool history rendering has reference tests, but its output `tool_calls`
parser has not been adapted; full tool calling is not claimed.

Full-model HTTP smoke checks have exercised both modes and chunked prefill.
Performance and acceptance baselines, and a batch-dependent plain/verify output
difference, remain under investigation; see the integration log for evidence and
current limitations.
