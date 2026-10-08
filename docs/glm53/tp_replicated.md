# GLM-5.3-Flash replicated TP serving

## Contract

The decode manifest declares:

```json
{"topology":{"groups":{"ep":8,"tp":8},"replicated_rows":true}}
```

`tp` selects one eight-rank serving group. Existing `ep` weight bindings and
NCCL rank arguments remain unchanged: both groups name the same eight ranks.
This requires the updated Rust serving protocol; an older kern-serve binary
rejects the new field. Generation does not rebuild or restart the server.

Replicated rows are an explicit, opt-in alternative to the existing rotated
tray. The protocol accepts only rows=1 programs, no span, and no tray axis or
`blocks` fill. Prefill proceeds through one-token steps. All ranks receive the
same logical cell order, token IDs, positions, sequence lengths, validity and
cu_seqlens. One logical batch of five requests becomes eight rows per rank,
not eight separately padded rank blocks. The maximum batch is 16, not 128.

Each request has a full paged lease on every member, plus each member's
recurrent state. Slot IDs, block tables and line tables use that member's
lease and may differ numerically across ranks. Every rank writes its own DSA
latent/index cache and its own KDA head shard. Admission counts the request
against every member. The group leader is only the scheduling/readback owner.
Tokens are read once from that leader's output in the common row order.

Checkpoint, retire and park already visit every member. Restore and wake use
the requested prefix length for every full paged replica. This keeps local
page mappings local and does not require equal physical page IDs. GPU copy
execution and value parity still need a later device test. Aggregate page
metrics in the tray continue to count physical pages across GPUs.

## Why `blocks` is not the fix

In the original tray mode, `blocks` is a separate i32 input of length R+1.
The host stages exclusive prefix sums of member block sizes, including
padding, in member order (in token rows, multiplied by the program's rows).
It is not a token or validity buffer. `tray_rows` rotates the batch so each
rank's own block comes first.

`tools/kernels-src/peer_collective.cu` is an explicitly launched CUDA module,
not an automatically inserted runtime extern. Its all-gather implements that
rotation; `peer_allreduce.cu` remaps peer row coordinates before summing.
Plain NCCL does neither. Merely adding the fill would leave incompatible row
orders and owner-only KV storage. Replicated mode instead stages the canonical
row order on the CPU, so existing element-wise NCCL operations are correct
without a gather before embedding or a redistribution after argmax.

## Generator and kernel corrections

- The generator overrides runtime `num_valid_tokens` in both MoE GEMMs with
  `9 * seqs`; `lower_wire` lifts it onto every call. It is not the padded EM.
- The generator forces FA3 launch PDL off. Restore performance mode here when
  debugging ends; changing only the source default in ops_dsa.py is insufficient.
- These overrides intentionally avoid edits to ops_moe.py and ops_dsa.py while
  their FP8 scale bindings are maintained in parallel.
- NCCL logits remain BF16 [rank,batch,19360]. The existing cast launch also
  permutes them to FP32 [batch,154880]; no extra buffer or launch is required.
- kpool closes at positions 3,7,11,..., with an independent eight-slot tail ring.
  Physical index-page pitch is 11*8448 = 92928 bytes, not one layer's 8448 bytes.

## CPU validation

- `cargo check -p kern-serve --tests` checks the server and its tests without
  producing a new server executable.
- `cargo test -p kern-manifest` checks both protocol modes and invalid mixtures.
- `cargo test -p kern-pool --test pool` checks host-only lease/checkpoint behavior.
- `crates/kern-serve/src/tray.rs` has host-only tests for all batch buckets,
  legacy layouts, replicated staging with unequal allocator histories, and
  per-rank page/line/slot selection and checkpoint metadata. These can also be
  executed in an isolated CPU harness without linking kern-serve/CUDA.
- `python3 tools/glm53/test_cpu.py` checks generated manifests/module hashes,
  dynamic MoE counts, PDL, kpool source invariants, and the actual head-indexing
  function on the CPU at B=1,2,4,5,8,16.
- Regenerate both JSON files after compiling changed CUDA sources with nvcc
  (compilation only); pass `--bundle kernels-glm53` to preserve content-addressed
  artifacts. Run the kern-manifest `verify` example on both files and
  our bind-check script on the serving deployment.

## Remaining device checks

No CUDA graph, NCCL, attention or recurrent-state execution is validated by
these CPU tests. Test one row, a five-row batch padded to eight, full B=16,
then prefix restore and graph/eager agreement. The pre-existing FA3 scheduler
scratch arrays declared with eight elements also need an ABI capacity audit
before claiming B=16 kernel safety. This patch does not change those arrays.
