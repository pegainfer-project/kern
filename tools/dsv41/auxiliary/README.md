# Auxiliary kernel contract

`build.py OUTPUT` builds `dsv41_auxiliary.cubin` with CUDA 13 for SM103a.
`ops.definitions(cubin, rows=..., groups=..., heads=...)` returns schema modules
and typed ops. Expression-valued rows work for dynamic prefill and speculative
rounds. No complete checkpoint export is used.

These are initial standalone implementations, not yet a fused performance result.
`test_reference.py INFERENCE_DIRECTORY CUBIN` executes supplied model methods
verbatim (AST extracts classes to avoid constructing the entire model), and the
supplied TileLang quantization kernel for cache validation. Run on an allocated
GPU only. The reference source and checkpoint must refer to the same revision.

`test_fused.py AUXILIARY_CUBIN STAGE_CUBIN` asserts that `norm_quant` and
`norm_rope` are byte-identical to the launch pairs they replace, over row
counts that are not a multiple of four and over the full row capacity.

Arrays are contiguous except where an op takes an input row stride: those read
a column slice of a wider projection output in place, so the merged Q-LoRA/KV
GEMM needs no split. `rows` is the flattened local request-token dimension.

| Op | Arguments in order, after output(s) | Layout |
|---|---|---|
| engram_history | token IDs, physical slots, token map, mask, rows | inout history state is int64 per physical token; mask false writes DEAD=-1 |
| engram_hash | history, request IDs, positions, page table, multipliers, primes, offsets, rows, page size, page-table stride, layer count, head count, max ngram, pad ID | output int64 `[rows,layers,(ngram-1)*heads]`; multiplier `[layers,ngram]`, primes/offsets `[layers,hash_cols]` |
| engram_lookup | hashes, FP8 table, E8M0 scales, rows, columns, dim, hash-row stride, layer offset | output BF16 `[rows,columns,dim]`; scale groups32; table may be mapped pinned host memory |
| engram_inject | residual, projected KV, Q weight, K weight, mask, rows, copies, dim, epsilon | residual `[rows,copies,dim]`, projection `[rows,copies+1,dim]`, weights `[copies,dim]` |
| compressor_stage | two inout history states, projected KV, projected score, physical slots, rows, dim | float32 per-token projections; separate launch completes before gather |
| compressor_gather | output KV, score, validity; history states, request IDs, positions, pages, rows, dim, page size, stride | gathered `[rows,2,dim]`; valid iff position is odd; incomplete outputs zero |
| compress2 | gathered KV, score, norm weight, groups, dim, epsilon | output `[groups,dim]`; dim <=1024; pooling is rounded to BF16 before norm |
| compress1 | input, norm weight, rows, dim, epsilon | ordinary BF16 RMSNorm |
| norm_quant | input, norm weight, rows, dim, input row stride, scale row stride, epsilon | compress1 then the MXFP8 activation quantization of its result; outputs normalized BF16 `[rows,dim]`, E4M3 `[rows,dim]` and packed I32 scales `[dim/128,scale row stride]`; grid is align4(rows) so the four-row padding clears its scale words, as moe/stage.cu does |
| norm_rope | input, norm weight, cos/sin, positions, rows, dim, rope dim, input row stride, inverse, epsilon | compress1 then rope over one head; only the rotated rows reach a buffer |
| window_indices | request IDs, positions, block-end positions, window lines, rows, window size, output width, ring, noncausal, draft rows | output int32 ring token indices `[rows,width]` (`line*ring + position%ring`), invalid=-1 |
| map_indices | logical selected positions, request IDs, positions, compressed pages, rows, topk, ratio, page size, stride | checks selected positions against `(position+1)//ratio` |
| cache_fp8/cache_fp4 | input BF16, physical slots, rows, page size | state is page-major FlashMLA storage; negative slots skip writes |
| cache_gather | cache state, physical slots, rows, page size, fp4 flag | output BF16 `[rows,512]`; negative slots become zero |
| index_quant | packed bytes, E8M0 scales, BF16 dequant outputs; input, rows, dim | groups32; output packed `[rows,dim/2]`, scales `[rows,dim/32]` |
| rope | input, cos/sin, positions, rows, heads, dim, rope dim, inverse | cos/sin float32 `[max_position,rope_dim/2,2]`; output same shape as input; input/output may alias; configure grid heads |

Window FP8 pages store `[page_size,512]` bytes then `[page_size,16]` E8M0
scale bytes. Compressed FP4 pages store `[page_size,256]` packed bytes then
`[page_size,32]` E4M3 scale bytes. Thus logical page sizes are 528 and 288
bytes/token, respectively. Per-token records are **not** interleaved.

Window states are rings, not paged: each `{target,draft}.window.N` is a
`bytes_per_seq` state of `Layout.ring` tokens (the 128-token window plus
every row one step can write, rounded to whole pages: 256 at `max_tokens`
128), laid out as `ring/page_size` consecutive FlashMLA pages. A row's
window token index is `slot*ring + position%ring`, where `slot` is the
sequence's entry in the `window.lines` line table (`[1, seqs]`, stride the
ring bytes, so the runtime fills it with the slot itself; every window state
of a lease shares it). `metadata` emits it as `window_slot` (-1 for padding)
for `cache_fp8` and the draft context publication; `window_indices` derives
the visible window from it. A write at position p lands on p-ring, which is
older than any window a row of the same step reads, so no step can clobber
what it still needs. Nothing about a sequence's earlier positions survives
beyond the ring; only the last `ring` tokens are ever read.

Paged histories deliberately retain speculative projected positions. Only the
accepted sequence extent grants visibility; a subsequent round must overwrite
all its new token positions before hashing/pooling. Rejected rows cannot mutate
older history. Compressor output validity must mask compressed-cache writes and
index-key production. Its slot is `position//2` in the compressed page domain,
not the input token's page domain. The two projected histories have identical
logical page IDs and independent storage. This initial projected history is
large (4096 bytes/token/source); narrowing retained pages is a scheduler task.

Draft noncausal windows use width rounded up to a multiple of64 (192 for128+5),
with remaining entries -1. Pass draft_rows=5 and block-end equal to the
exclusive end of the draft block; each row then sees the same accepted context
and complete tentative draft block. Target verification remains causal. Draft
and target page domains must remain separate.

Limits: shared pinned-host mapping/lifetime is supplied by the runtime. GPU and pinned-host UVA
lookup correctness are tested; actual host-UVA bandwidth is not yet tested.
Cache inserts accept already-normalized/rotated BF16 inputs. GEMMs, selector,
final program integration are outside these
ops. State-domain/index compatibility remains the caller's responsibility.

## Bounded compressor state (preferred)

`compressor_short_gather` supersedes projected paged history for ratio2.
Persist one `[2,512]` FP32 last-token KV/score pair per source per sequence:
4096 bytes, independent of context length. Current invocation projections stay
in workspace until acceptance is known. All rows are contiguous per sequence,
with `row_starts[seqs+1]` and `request_ids[rows]` identifying each interval.

Arguments: output gathered KV `[rows,2,dim]`, output gathered score, output
validity `[rows]`, input state, current projected KV `[rows,dim]`, projected
score, line table `[seqs,line_width]`, row starts, request IDs, absolute positions,
rows, dim, line width. The first row reads committed state only if its absolute
position is odd; remaining complete pairs come from current workspace. No state
is mutated during gathering or pooling.

`compressor_commit(state, kv, score, line_table, row_starts, accepted, seqs,
dim, line_width)` runs only after acceptance. It saves the last accepted row's
KV/score in the selected line. With `line_width=1` it supports arbitrarily long
prefill; with `line_width=6`, accepted count1..6 selects entry `accepted-1`, using
the runtime's existing speculative line contract. Zero accepted is a no-op.
The next call reads the committed line at entry0. Aliased lines are safe because
commit runs after every compute stage that consumes the previous state.

Use `bytes_per_seq=4096` per source and a line table domain indexing that state
with stride4096. Combining sources into one state requires source-interleaved
line IDs and `bytes_per_seq=source_count*4096`. The two paged history operators
remain available for debugging, but are unnecessary for the bounded path.

Validation includes six rejected-prefix trajectories with accepted counts
1,2,5,6,3,4, both even/odd boundaries, unchanged state before commit, and actual
supplied `Compressor.forward` snapshots restored at each accepted prefix.
`test_reference.py --dump DIR` additionally emits representative manifest/input
cases; `replay.py PROGRAM_IO CUBINS DIR` executes them through kern's runtime and
requires byte-exact agreement with the CUDA-driver launches.

## Serving lowering and load-time constants

`serving.build(cubin, Layout(...))` exposes buffers/states/typed ops and concrete
call builders: `prepare`, `engram_hash`, `engram_lookup`, `window_write`,
`compressor`, `compressed_write`, `compressed_indices`, `index_write`, `commit`.
Call `pieces(final_programs)` to retain only definitions referenced by the final
programs; merge the result with other providers' definitions. The schema rejects
unused definitions, so do not merge the complete helper catalog verbatim.

`prepare(mode, ids_buffer)` follows the serving fills, including `valid` from the
host. It supports prefill chunks, decode1, verify6 and interior draft5. Draft
selects staged positions/slots0..4 starting at the anchor; the previous
accepted target row already supplies context through anchor_position-1. `window_slot` is the row's
ring token index (see the window ring above). `window_length` is128 or
less for target,133 or less for draft (indices padded to192); both window and
compressed lengths are zero for padding. Compressed length512 relies on each
invalid index remaining-1. Masked physical slots are-1, so cache/history writes
skip them. Token0 is valid vocabulary and is never a padding heuristic.

Use width1 compressor line tables for every mode. The serving loop initializes
only column0 of a wider table; later columns are null until a kernel populates
them. Explicit accepted-prefix commit needs no such extra versions: it selects
a saved projection row and writes the existing committed line. The kernel also
refuses line0, even if mistakenly passed a positive count.

Default original-token pages have128 slots. Window cache pages therefore hold128
FP8 tokens; ratio2 compressed pages hold64 tokens with144 bytes allocated per
original token. Indexer pages hold packed FP4 `[page,64]` followed by E8M0 scales
`[page,4]`, followed by padding to a512-byte page stride. Ratio2 page64
uses4608 bytes (36 per original token); ratio1 page128 uses8704 bytes
(68 per original token).
`index_write` consumes packed keys plus scales, not the BF16 debug cache layout.

`engram_constants.py MODEL INFERENCE OUTPUT --max-positions N` checks the real
compressed tokenizer vocabulary (99092), applies the supplied reference hash
layout/RNG, and compiles two initialization entries. Its JSON returns
`buffers/modules/ops/calls`; merge these and append calls to the load-once stage.
No original weight is exported. It embeds the small token map/hash metadata and
only32 inverse frequencies per RoPE family, then computes the full tables on GPU
once. Carry names are `rope.{window,compressed}.{interleaved,split}`. The default
frequency derivation uses the allocated CUDA device, because CPU Torch pow
rounding differs and produces visible phase errors at long positions.

`rope_name(layer_compress_ratio, split=False)` selects the family for the entire
layer. Ratio0 layers/draft use theta10000 without YaRN. Ratio>0 layers use
config theta160000 with YaRN for **all** Q, window KV, compressed KV, indexer and
inverse-O rotations. GPU validation over1,048,576 positions is bit-exact against
the supplied original GPU `precompute_freqs_cis`, for both families and layouts.

`index_metadata(mode, source)` expands the staged request page table into
`{mode}.c{ratio}_page_table[rows,pages]` and emits
`{mode}.c{ratio}_end[rows] = floor((position+1)/ratio)` for active rows.
This matches compressed-block visibility in the reference Indexer, and applies
it independently to every verification query. Padded queries have zero length;
pages beyond each query's visible extent are zeroed. Physical page IDs retain
their original-token allocation numbering.

The same call writes `{mode}.index_request_ids=row` so paged sparse scoring
cannot pair queries across independent virtual requests. It also writes
`{mode}.sparse_ends=16384` for active rows (zero for padding), which bounds
DeepSelect candidate coordinates. The scorer itself must use `c{ratio}_end`,
not `sparse_ends`, for its actual compressed context length. Sources sharing a
ratio can reuse query metadata. Draft layers have no compressed indexer.

`draft_context.capture` writes the BF16 mean of four materialized HC copies into
its target-layer slice of a15360-wide tap buffer. Capture layers37/38/39 after
Engram and before attention, using `Blocks.materialize` first. The first capture
initializes the buffer's write lifetime; all three slices must be captured
before calling `draft_context.publish`.

Context publication uses the original FP8 checkpoint projections and RMSNorm:
`mtp.0.main_proj/main_norm`, then each draft layer's `attn.wkv/kv_norm`, window
RoPE, and FP8 cache insertion. It runs after target forward in plain modes and
after `spec_count` in verification. Verification masks physical write slots by
each request's accepted prefix, so zero acceptance preserves the context cache.
`test_draft_context.py` executes the complete lowering through program_io and
CUDA graphs with real checkpoint bindings, comparing against the supplied
`DSparkBlock.forward_embed` and `DSparkAttention.forward` methods.
