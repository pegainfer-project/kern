# Qwen3.8-27B hosted manifest: what the transform changes

Details behind the transform summarized in [vllm.md](vllm.md).
`tools/qwen38_vllm.py` maps `examples/qwen3.8-27b.json` to
`manifests/qwen3.8-27b-vllm.json` in the HF repo `Pegainfer/kern-qwen38-sm103`.
Regenerating it from the base manifest reproduces the published file byte for
byte.

## Page geometry

vLLM gives every layer of a hybrid model one page per block, the same size for
the attention and GDN groups. A GDN line is conv `[3][10240]` bf16 plus ssm
`[48][128][128]` f32, 3,207,168 bytes. The attention kernel (TRTLLM-GEN, P64)
takes 64-token pages of 4 KV heads × 256 × k|v bf16, 262,144 bytes per layer.
vLLM rounds the GDN line up to whole attention pages: 13 kernel blocks,
832 tokens, 3,407,872 bytes per page.

## States

The pooled `kv` and `gdn` states become 64 per-layer host states:

| state | dtype | shape | strides (elements) |
|---|---|---|---|
| `kv.l<L>` (L % 4 == 3) | bf16 | `[0, 4, 64, 512]` | `[131072, 512, 2048, 1]` |
| `gdn.l<L>` (other layers) | u8 | `[0, 3207168]` | `[3407872, 1]` |

`kv.l<L>` is vLLM's LBNHC view of one layer's kernel block (`[token][head][k|v]`
inside the block). A call argument `{"state": "kv", "offset": l·262144 + o}`
becomes `{"state": "kv.l<L>", "offset": o}`; GDN arguments are rebound the same
way, with their line offset checked.

## Strides

Page strides are literals in the base manifest, so they are rewritten:

| what | base | hosted | why |
|---|---|---|---|
| attention block (tensormaps, `attn_prep`) | 4,194,304 | 262,144 | kern's page held 16 attention layers × 64 tokens; vLLM's kernel block is one layer × 64 tokens |
| GDN line | 3,211,264 | 3,407,872 | vLLM's page is padded to 13 attention blocks |

## Conv kernel

Every kernel takes these strides as arguments or tensormap fields except one:
the Triton `_causal_conv1d_fwd_kernel` captured from vLLM, whose line stride was
a constexpr of the captured instance. It is replaced by `gdn_conv_fwd`
(`sources/gdn_conv_fwd.cu` in the HF repo, pinned by sha in the generator).
The replacement uses Triton's bf16 product rounding and its lowering of exp
and division, and takes the stride and the sequence count as parameters. It
runs as two launches: the conv over the chunk, then the conv state update.

## Programs

- `prefill` and `decode_batch` end at the final norm, which writes
  `hidden` instead of feeding the lm_head and argmax. `next_token` is dropped.
- `head` is one gemm: `head_in` `[seqs, 5120]` × `lm_head.weight` →
  `logits` `[seqs, 248320]`.
- `slot_mapping` and `block_table` index `kv.l3` (1 and 64 tokens per id),
  `gdn.line_index` indexes `gdn.l0` (one page per id). Every host state of
  the same layout shares these ids, as vLLM's layers share its block ids, so
  a tool that is its own host (`kern bench`) can provision the states.

## Ragged prefill

The base `prefill` serves one sequence: `fla.chunk_indices` and
`fla.chunk_offsets` are written once by a load call, the chunk kernels' grids
are `ceil(tokens / 64)`, and `h` / `has_initial` hold one sequence. The
hosted `prefill` takes `seqs` sequences split by `cu_seqlens_q`:

- `fla_chunk_index` (`sources/fla_chunk_index.cu`) runs first in every call
  and builds both tables from `cu_seqlens_q`.
- Grids that walk `fla.chunk_indices` cover `ceil(tokens / 64) + seqs`, the
  most chunks `tokens` rows split into `seqs` sequences can make. Entries
  past the call's chunks name a chunk past the last sequence, whose rows
  are out of range, so they write nothing. `chunk_h` walks `seqs × 48`.
  This needs `add` of two expressions in the manifest grammar.
- `h` grows by one chunk state per sequence, `has_initial` to `seqs`.
- Where the chunk kernel takes the initial state by pointer (vLLM's Triton
  `chunk_h` in the base manifest), `line_gather` / `line_scatter` copy one
  line per sequence, one grid row each, into an `h0` of `seqs` states.
  Its `solve_tril` leaves the blocks above the diagonal unwritten too, and
  is not ours to rebuild, so `prefill` zeroes `Ai` once per call: every
  layer of a call writes the same places.

Two kernels were only correct for one sequence, and are rebuilt (pinned by
sha in the generator, sources in the HF repo):

- `chunk_h` built with `-DLINE` located a sequence's state line at
  `seq * 48 + head`; the line is already the sequence's, so the offset is
  `head`. Sequences after the first read and wrote past their line.
- `solve_tril` stored only the diagonal blocks of its inverse and left the
  zeros above them to a workspace nobody else writes. A ragged call moves a
  row's position within its 64-row chunk, so stale values from an earlier
  call reached `recompute`. It now stores whole rows.

One sequence per call stays bit-identical to the previous manifest.

## gemm16 in `decode_batch`

The base `decode_batch` declares 128 groups (sequences), but its `gemm16_*`
kernels compute at most 16 rows. Under vLLM a 32-request decode batch produced
garbage from row 17 on. The hosted manifest replaces them with cuBLASLt `gemm`:

| base | hosted |
|---|---|
| `gemm16_in_proj` | two `gemm` |
| `gemm16_gate_up_silu` | `gemm` + `silu_mul` |
| `gemm16_qkv`, `gemm16_o`, `gemm16_out` | `gemm` |

gemm16 is bit-identical to cuBLASLt at M ≤ 16, so numerics do not change. The
base manifest still has the mismatch; whether kern-serve hits it with more
than 16 concurrent sequences has not been checked.
