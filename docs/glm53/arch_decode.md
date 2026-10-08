# GLM-5.3-Flash decode serving architecture on kern (TP8, H100) — arch_decode.md

Owner: architecture. Status: design of record for the decode path, prefill organization, and the
performance roadmap. Supersedes the "option A" state layout adopted from the fa3 ABI review (§1.5
explains why), and flags ten defects in the generator code written so far (§0.2).

Conventions:
- **kern citations** are repo-relative (`crates/...`, `tools/...`, `docs/...`) at
  `the kern repo`, working tree as of 2026-09-26.
- **sglang citations**: `SP/` = `<sglang site-packages>/`.
  `TC/` = the Triton cache sglang compiled into, `$HOME/.cache/sglang/triton/`.
  `TL/` = the TileLang kernel cache, `$HOME/.cache/sglang/tilelang/<ver>/linux-x86_64/kernels/`.
  `DUMP/` = `dumped-kernels-glm53-sglang/` (one capture per directory).
- **UNVERIFIED** marks a claim I could not check from this box without GPU memory. Each one comes
  with the command that would check it.
- Numbers are per TP rank unless stated otherwise. `S` is sequences per call (the bucket), `T` is
  rows per call (`T = S` for decode, `T = S·(k+1)` for a speculative round). `S_MAX = 16` is the
  static row pitch.

---

## 0. Executive summary

### 0.1 The architecture in eight lines

1. **States: 2 paged + 3 per-sequence.**
   - `kv`: 11,264 B/token, token-interleaved `[token][layer][512 bf16]`.
   - `idx`: 363 B/token, 256-token pages, each page layered `[layer][64 pools × 132 B]`.
   - `kda_conv`, `kda_ssm`, `idx_tail` are per-sequence.
   - Page unit **256** (not 64): the kpool compression then writes only into the page holding its
     tokens. That makes kern-serve's automatic retire/restore (multi-turn prefix reuse) safe, and
     it removes the 3/4 kpool waste. fa3 slot-linearity is kept with a single `kv` arena (§1).
2. **One `decode` program** with a variable `seqs` (kern-serve pads to buckets 1/2/4/8 and
   captures one CUDA graph per bucket), plus a `prefill` chunk program, plus later a `round`
   program for MTP. Every mined kernel must therefore be runtime-generic in S. §2 shows that 7 of
   the 31 mined kernels are not, as pinned today.
3. **The allreduce is sglang's algorithm, not NCCL.** It is a one-shot Lamport push on kern peer
   buffers: bf16 in, fp32 sum in rank order, one rounding. That is bit-compatible with sglang,
   and it is about 2× faster than NCCL LL128 at 8 KB. P2 fuses it into the mHC boundary kernel.
4. **The step is latency-bound, not bandwidth-bound.**
   - At bs1 on 141k context, sglang spends ≈5.5–5.7 ms on 1,351 kernels.
   - The weight-streaming floor is ≈1.2 ms and the KV/indexer traffic is < 0.1 ms.
   - The lever is launch count plus PDL: 1,351 → ~590 launches in P2 (§3.2, §4.4).
5. **P2 target (≥30% better than sglang at bs1): ≈3.2–3.5 ms/step.** It comes from fusion of the MoE
   block (10 → 3 launches), the mHC boundary plus allreduce (5+2 → 4 per layer), the DSA glue
   (33 → ~16) and KDA (7 → 3), with PDL everywhere a kernel supports it.
6. **P3 target (≥2× better at bs < 8): MTP speculative decoding with k = 2**, on top of P2.
   - The round program uses per-row DSA queries, a KDA "advance" pass (the kern dflash2
     pattern) and an 8-entry kpool tail ring, which makes rollback free.
   - Expected ≈1.6 ms/token at bs1 (3.6×) and ≈0.7 ms/token at bs4 (2.5×). Both assume an MTP
     acceptance rate that is **UNVERIFIED**.
7. **Prefill is a separate chunk program** (T ≤ 8192, eager). It uses the unfused mHC chain
   (sglang disables the fused boundary above T = 16), chunked KDA, and the sparse prefill
   kernels. The capture has **no** long-context sparse-prefill kernels. That capture is the top
   evidence gap.
8. **Baseline reality check.** The server log of this box shows **bs1 = 176 tok/s = 5.7 ms/step
   and bs4 = 579 tok/s = 6.9 ms/step at 141k** (the sglang serve log,
   `:3320`). The 12 ms and 25 ms figures in the brief are about 2× slower and probably include
   client or streaming overhead. All targets below are stated against the stricter log numbers.
   The baseline must be re-measured cleanly (§4.6).

### 0.2 Defects found in the current generator (fix before P0 bring-up)

Each is silent unless noted.

| # | Where | Defect | Evidence | Fix |
|---|---|---|---|---|
| D1 | `ops_moe.py:220-232` (router) | The pinned `_router_triton_kernel` is **module_329**, which has **M = 1 baked in**: the TTIR has no `M` arg, and `mask_m = pid < 1` is a constant. Rows ≥ 1 are never routed at S > 1. | `TC/JINWNBXI…/_router_triton_kernel.ttir`; `ttir_diff` 329 vs 411 (§2.3). | Use **module_411** (M is a runtime arg; both cache variants compile to the same cubin). Add `M = var tokens` as arg 5. |
| D2 | `ops_moe.py:236-247` (align) | The pinned `_moe_align_small_numel` is module_485, whose padded-numel constexpr is **NP = 32** (module_331 has NP = 16). `numel = 9S`, so it breaks for S ≥ 4. | TTIR: `n_fill = 31`, `tensor<32xi32>` (485) and `15` / `<16>` (331). | Handwrite it: one CTA, ≤ 144 entries. It can be fused with the router topk (§4.2c). |
| D3 | `ops_moe.py:205-214` (router GEMM) | `tiny_n_gemm_kernel<GEMMTraitN<288,4096,3,32>, **M = 1**, f32, PDL>`. The template M is the row count (`SP/sglang/kernels/jit/csrc/gemm/tiny_gemm.cuh:71-72`, `:98-104`). | The skeleton pin is the `…ELj1EfLb1EEE` entry. | module_327 also contains the **M = 1..16** entries (SASS scan, §2.4). Use the M = 16 entry for every bucket: per-row arithmetic is identical (`tiny_gemm.cuh:103-140`). |
| D4 | `ops_kda.py:25` (conv) | module_473 bakes **`num_cache_lines = 581`** (`SP/sglang/kernels/ops/mamba/causal_conv1d_triton.py:606`, mask at `:732`; TTIR `cmpi slt idx, 581`). kern's conv line is `slot·34 + layer`, so any slot ≥ 17 loses its conv state. seq_slots = seqs_max + 2 = 18 already makes slot 17 exist (`types.rs:105-124`), and checkpoints add more. | `TC/LAWVJ2KQ…ttir` line 71 (`%mask = … %c581_i64`). | Recompile the same Triton source with `num_cache_lines = 2^31-1` (§2.5), or handwrite it (P2 fuses it anyway). |
| D5 | `ops_kda.py:26` (norm_gated) | The team chose **module_399**, whose `T` carries `tt.divisibility = 16`. module_227 has plain `T`, and the TTIR bodies are otherwise identical. At decode T = 8·S, so S odd violates the hint. The bs1 capture itself ran module_227. | `TC/NGMCRZTJ…` vs `TC/HQZKUMJZ…` | Pin **module_227**. |
| D6 | `tools/glm53/kernels/glm53_misc.cu:46-60` (hc_repack) + `ops_mhc.py:12-16` | The DeepGEMM prenorm writes `sqr_sum` **compact** at pitch `shape_m` (`SP/deep_gemm/include/deep_gemm/impls/sm90_tf32_hc_prenorm_gemm.cuh:114`, `:236-241`). Only `mul` (a TMA store through a static tensormap) has pitch S_MAX. The repack reads sqr at pitch 16, so it is **wrong for every split ≥ 1, even at S = 1**. | The probe log shows `sqrsum [64,97] stride [97,1]` (`the probe workspace/probe-103868.jsonl`, record 0). | Delete hc_repack. Run prenorm with `shape_m = S_MAX` and big_fuse64 with `T = S_MAX`, so both partials have pitch 16. The cost is zero: one 64-row M block either way (§3.3). |
| D7 | `ops_mhc.py:74-99` (big_fuse, fma) | The TileLang kernels take their buffers in **alphabetical** order, not Python order. | big_fuse: `(comb_mix, gemm_out_mul, gemm_out_sqrsum, hc_base, hc_scale, layer_input, norm_weight, post_mix, residual, num_tokens)`. fma: `(cur_residual_out, hidden_in, mixes_partial_out, pre_fn, prev_comb_mix, prev_post_mix, prev_residual, sqrsum_partial_out, num_tokens, split_k)` (`TL/*/device_kernel.cu` signatures; recipe sites 5 and 14 confirm the pointer order). The op param lists follow the Python wrapper and pass `a(i)` positionally. | Permute the launch args. `mhc_post` `(a,b,c,d,x,n)` is already correct. |
| D8 | `ops_dsa.py:278-305` (scratch + prep), `:392`, `:431` | `dsa_fa3_prep`, `dsa_fa3` and `dsa_fa3_combine` are **three ops each declaring `fa3_scratch`**. Scratch is **op-private** (`crates/kern-runtime/src/compile.rs:393-397`, resolved from `rop.scratch` at `:521-526`, `:648-651`). fa3 therefore reads uninitialized scheduler metadata and semaphore, and combine reads partials fa3 never wrote. | kern semantics. | Make **one op `dsa_attn` with 3 launches** (prepare → fa3 → combine) sharing the scratch, or promote the scratch to workspace buffers. |
| D9 | `ops_head.py:78-87` via `ops_dsa.py:147` | `glm53_rms_norm` assumes row pitch = d (`tools/glm53/kernels/glm53_misc.cu:63-66`). q_a and kv_a are column slices of the fused `[S,2048]` GEMM output (pitch 2048), so rows ≥ 1 are normalized from the wrong bytes at S > 1. | Code. | Add `in_pitch` / `out_pitch` args (and fuse the kv_store, §4.2c). |
| D10 | `gen.py:34-58` | Four problems: (a) every fill is over `seqs`, but the protocol requires `slot` over a rows var and `seq_len` over a **different** groups var (`crates/kern-manifest/src/protocol.rs:264-273`); (b) `block_table` is `[W, seqs]`, but a page table must be `[groups, W]` (`protocol.rs:400-405`); (c) one combined `kda` state with line = conv + ssm is incompatible with the conv kernel's baked 18,432-B line stride; (d) `idx` bytes_per_token = 11·512 is stale. | Code. | Use the state and buffer tables of §1 and §3. |

Also check the handwritten kpool_update (`tools/glm53/kernels/glm53_dsa.cu:136-238`):

- **Softmax lowering.** It uses `expf` and IEEE `/`. The mined Triton kernel lowers `tl.exp` to
  **`ex2.approx.f32(x·log2e)`** and `/` to **`div.full.f32`**
  (`TC/W2FILQN3…/_kpool_decode_update_and_maybe_write_cache_kernel.ptx` lines 260-336). The
  comment "tl.exp == libdevice expf" at `glm53_dsa.cu:173` is wrong. This is the same lesson as `NOTES.md` ("Matching
  Triton's lowering mattered"). The ue8m0 path is fine: libdevice `log2f`, `cvt.rpi`,
  `ex2.approx` of an integer, and a power-of-2 divide, all exact.
- **Page mapping.** It still uses the unit-64 page row `(pool>>6)<<2` (`glm53_dsa.cu:212`). That must become
  `bt[pool>>6]` with the unit-256 layout (§1.5).
- **Pad guard.** It keys its pad test on `slots[row] != 0` (`glm53_dsa.cu:146`). That only works because kern-serve
  leases the pad first (`crates/kern-serve/src/tray.rs:551-554`, `:815`). kern has no reserved
  token slot 0: `pages.rs:607` hands out page 0 first. Use `Fill::Valid` instead.

The `ops_dsa.py` docstring claims "fp8 GEMM kernels never write rows ≥ M" (`ops_dsa.py:24-26`).
**This is false.** The 1d2d epilogue TMA-stores whole BLOCK_M = 16 row tiles, clipped only at the
tensormap's M dim (`sm90_fp8_gemm_1d2d.cuh:430-441`: no shape_m guard). Rows [M, 16) are written
with junk. The consequences are in §2.3 (T7).

---

## 1. State and lease architecture

### 1.1 What kern's pool guarantees (the semantics we design against)

| # | Semantics | Citation |
|---|---|---|
| K1 | The page unit is the lcm of the `stride`s of every input whose domain `index_into`s an owned paged state. Every paged state is an arena of pages of `unit × bytes_per_token` bytes. | `crates/kern-pool/src/pages.rs:268-280`, `:302-307` |
| K2 | **One lease is the same page ids in every paged state.** All paged arenas hold the same number of pages, `budget / Σ page bytes`. | `pages.rs:15-19`, `:316-335` |
| K3 | A token slot id is `page × unit + pos % unit`, identical in every paged state. | `pages.rs:881-887` |
| K4 | A page-table row has one entry per `stride` tokens of each held page, then the first entry repeated to the row width. Its shape must be `i32 [groups, W]`. | `pages.rs:898-910`; `protocol.rs:400-405` |
| K5 | A per-sequence state is `bytes_per_seq` per slot. Slot 0 is never leased (the null line). A line table is `i32 [lines, groups(, w)]` with entry `slot × (bytes_per_seq/stride) + row`. | `pages.rs:21-31`, `:228-254`, `:927-938` |
| K6 | A line table's stride must divide `bytes_per_seq`. | `crates/kern-manifest/src/verify.rs:1032-1036` |
| K7 | Each pooled state rounds up to whole chunks on its own. The chunk is **64 MiB** whenever the smallest page or slot object is < 2 granules (4 MiB), which is always true here. | `pages.rs:282-291`; `crates/kern-runtime/src/load.rs:623-652` |
| K8 | A fresh lease **zeroes** its per-sequence slots. restore and fork copy the slot plus the partial page. retire moves the slot and pages with no copy. | `crates/kern-runtime/src/lease.rs:95-109`; `pages.rs:655-777` |
| K9 | Pages in a checkpoint chain are never written again. This is enforced **only** for host-computed slots (`Lease::slot` asserts), never for addresses a kernel computes on the device. | `pages.rs:33-55`, `:877-887` |
| K10 | There is **no truncation API**. A paged state rolls back for free: the next call overwrites positions past `count`. A per-sequence state has no rollback. | `pages.rs:809-956` (Lease API); `crates/kern-serve/src/scheduler.rs:49-53` |
| K11 | kern-serve **retires every finished request as a snapshot at `s.pos`** (not page-aligned) when the manifest has a recurrent state, and restores it for a prompt that continues it. That is exactly the agentic multi-turn pattern. | `scheduler.rs:55-80`, `:722-742`; `pages.rs:687-740` |
| K12 | A tensormap over a state is encoded once at load (state base + static call offset). An outer dim of 0 spans the reservation: `dims[-1] = (bytes − inner)/stride + 1`. | `crates/kern-manifest/src/types.rs:1131-1150`; `compile.rs:713-721`, `:814-821` |
| K13 | Line tables are staged whole at pitch `groups.max`. Page tables are staged `[b, W]` for the bucket b. The H2D copies run on the compute stream before the graph launch. | `tray.rs:829-866`; `crates/kern-runtime/src/lib.rs:258-279`; `crates/kern-runtime/src/exec.rs:12-17` |
| K14 | Op scratch is private to the op, allocated once at var max, and shared by every call of that op. | `types.rs:875-893`; `compile.rs:239-245`, `:393-397` |

### 1.2 The recommended state table

Each state is one arena. The last column gives the addressing formula every kernel uses.

| state | kind | bytes | object (unit 256) | layout inside the object | per-layer addressing (i = DSA index 0..10, l = KDA index 0..33) |
|---|---|---|---|---|---|
| `kv` | paged | `bytes_per_token = 11 × 1024 = 11,264` (+1,024 per MTP layer) | page = 2,883,584 B | `[256 tok][11 layers][512 bf16]` (token-interleaved) | layer i, slot s: `kv + i·1024 + s·11264`. fa3: `ptr_V = ptr_K = kv + i·1024` (call offset), `stride_K = stride_V = 5632` elements. kv_store: `slot_pitch = 11264`, `layer_off = i·1024` (the handwritten kernel already takes both, `tools/glm53/kernels/glm53_dsa.cu:76-84`). |
| `idx` | paged | `bytes_per_token = 11 × 33 = 363` | page = 92,928 B = 11 × 8,448 | `[11 layers][64 pools × 128 B e4m3 ‖ 64 × f32 scale]`, i.e. sglang's 8,448-B kpool page per layer | layer i, page p: `idx + p·92928 + i·8448`. DeepGEMM kv TMA: base `idx + i·8448`, dims `(128, 64, 0)`, strides `(128, 92928)`. Scales: base `+8192`, dims `(64, 0)`, stride `(92928)`. Pool g sits at page `bt[g>>6]`, slot `g & 63`. |
| `kda_conv` | per-seq | `bytes_per_seq = 34 × 18,432 = 626,688` | slot | `[34][3 tok][3072 bf16]` | line `= slot·34 + l`. Table `kda_conv_lines [34, seqs]`, stride 18,432 (the conv kernel bakes 9,216 elements per line). |
| `kda_ssm` | per-seq | `34 × 524,288 = 17,825,792` | slot | `[34][8 H][128 V][128 K] f32` | line `= slot·34 + l`. Table `kda_ssm_lines [34, seqs]`, stride 524,288. The delta kernel passes `stride_h0_source = 131072` elements at runtime. |
| `idx_tail` | per-seq | `11 × 4,096 = 45,056` (12 lines with MTP) | slot | `[11][8-entry ring][128 bf16 k ‖ 128 bf16 score]` | line `= slot·11 + i`. Table `idx_tail_lines [11, seqs]`, stride 4,096. Ring index is `pos & 7`. |

Other options considered:

- **Option A** (22 states, unit 64, idx 132 B/token) is what the team adopted. §1.5 explains why it
  is replaced.
- A single `kda` state holding conv+ssm per line (`gen.py:46`, `:56`) is **incompatible** with the
  mined conv kernel, which bakes a line stride of 9,216 elements.

### 1.3 Input tables: who fills what

| buffer | shape / dtype | domain | filled by | read by |
|---|---|---|---|---|
| `token_ids` | i64 `[tokens]` | `index_into` embed | driver, `Fill::Token` (`tray.rs:804`) | embed |
| `positions` | i32 `[tokens]` | `min 0` | driver, `Fill::Position` (`tray.rs:810`) | kpool update (`pos&3`, `pos&7`), per-row DSA lengths in spec rounds |
| `slot` | i32 `[tokens]` | `index_into kv` (stride 1) | driver, `Fill::Slot` = `Lease::slot` (`tray.rs:811-819`) | kv_store |
| `seq_lens` | i32 `[seqs]` | `min 1` | driver, `Fill::SeqLen` = pos + rows (`tray.rs:820`) | dsa_prep, kpool update |
| `cu_seqlens` | i32 `[seqs_max+1]` | monotone | driver, `Fill::CuSeqlens` (`tray.rs:821`) | delta rule (varlen), fa3/combine (`cu_seqlens_q`) |
| `valid` | i32 `[tokens]` | 0..1 | driver, `Fill::Valid` (`tray.rs:809`) | router (P2: drop padding rows' experts); kpool update |
| `block_table` | i32 `[seqs, W]`, W = 1024 (262,144 tokens) | `index_into kv, stride 256` | driver, `extend_row` (`tray.rs:829-836`) | kpool update, `mqa_logits` (block table **is** the pooled table), topk transform |
| `kda_conv_lines` | i32 `[34, seqs]` | `index_into kda_conv, stride 18432` | driver, `seq_line` (`tray.rs:841-858`) | conv |
| `kda_ssm_lines` | i32 `[34, seqs]` | `index_into kda_ssm, stride 524288` | driver | delta rule |
| `idx_tail_lines` | i32 `[11, seqs]` | `index_into idx_tail, stride 4096` | driver | kpool update |
| `next_token` | i64 `[seqs]` | `Fill::Tokens` | argmax kernel | driver |

Derived on the device once per step, by the `dsa_prep` op, as workspace:

- `pool_lens[s] = seq/4`
- `pool_ctx[s] = max(seq/4, 1)` (DeepGEMM requires ≥ 1)
- `dsa_lens[s] = min(4⌊seq/4⌋, 2048) + seq%4` (fa3 `seqused_k`)
- the DeepGEMM schedule `[133,2]`, and (P1, see §3.2) the fa3 scheduler arrays

With unit 256 there is **no pooled table and no `::4` subsample**. The kern block table has one
entry per 256 tokens, which is exactly one kpool page of 64 pools, the page DeepGEMM's
`BLOCK_KV = 64` expects.

H2D per step at bs8:

- block table: 8 × 1024 × 4 = 32 KB
- line tables: (34 + 34 + 11) × 16 × 4 = 5 KB
- fills: < 1 KB

That is ≈3 µs of PCIe time, serialized before `cuGraphLaunch` (K13). Size W to the served maximum,
not to `MAX_POS = 1M`: W = 4096 would be 128 KB/step, ≈10 µs.

### 1.4 Capacity math: 200k tokens × bs8, per rank

| item | formula | bytes |
|---|---|---|
| pages | ⌈200,000/256⌉ = 782 pages/seq × 8 seqs, + 1 pad page | 6,257 pages |
| `kv` | 1,601,536 tokens × 11,264 | **18.04 GB** |
| `idx` | 1,601,536 × 363 | 0.58 GB |
| per-seq | 10 slots (seqs 8 + pad + null) × (626,688 + 17,825,792 + 45,056) | 0.185 GB |
| chunk rounding | ≤ 1 chunk per arena × 5 arenas × 64 MiB | ≤ 0.34 GB |
| **total states** | | **≈19.1 GB** |

| same workload, option A | kv 18.04 + idx 1,601,536 × 1,452 = 2.33 GB + rounding ≤ 25 × 64 MiB = 1.68 GB | ≈22.1 GB |

The rest of the memory:

- Resident weights per rank ≈ **41.6 GB**:
  - MoE 42 layers × 289 experts × 3 MiB = 38.2 GB
  - KDA 34 × 36.3 MB = 1.23 GB
  - DSA 11 × 39.3 MB = 0.43 GB
  - embed (replicated) 1.27 GB
  - lm_head shard 0.16 GB
  - mHC 0.14 GB
- MTP adds ≈1.0 GB.
- Workspace, graphs, cuBLAS and comm buffers take ≈2–3 GB (prefill buffers at `tokens = 8192`
  dominate).
- The state budget is therefore ≈33–35 GB, about **2.8–3.0 M tokens** of 256-page capacity.

That is 14× the bs8 × 200k need. sglang on the same box reserves 787,328 tokens
(`serve_sglang.log`: `max_total_num_tokens=787328`).

### 1.5 Why unit 256 and 2 paged arenas, and not option A (22 arenas, unit 64)

**(a) The kpool stride-4 write pattern breaks page sharing (K9, K11). This is not only a
"checkpoint later" problem.**

- sglang writes pool g into the idx page of **token-page row `(g//64)·4`**
  (`SP/sglang/srt/layers/attention/dsa/kpool_fp8_index.py:1155-1168`). With unit 64 that row is up
  to 3 pages *before* the page that holds the current position.
- kern-serve retires every finished request and restores it for the next turn (K11). The restored
  lease shares all whole pages before `len`.
- Concrete case: `len = 141,120`.
  - Shared pages are 0..2204.
  - The next pool is g = 35,280, whose row is 551·4 = **2204, a shared page**.
  - The restored lease's kpool update writes into the snapshot and into any other branch of it.
  - Nothing catches this, because the address is computed on the device (K9).
- With unit 256, pool g lands in page `⌊4g/256⌋ = ⌊g/64⌋`. That is the page containing its own 4
  tokens, which is always the lease's own page or the freshly copied partial page. In the same
  case, pages 0..550 are shared, page 551 is copied, and pool 35,280 goes to page 551. **Safe.**
- Prefix reuse at 141k is not optional for agentic serving. The log shows sglang serving each new
  turn with `#cached-token: 141120`. A 141k re-prefill costs ≈4.5 s at 31k tok/s.

**(b) Memory.** idx drops from 1,452 to 363 B/token (−1.74 GB at bs8 × 200k).

**(c) Arena rounding.** Each arena wastes up to one 64 MiB chunk (K7): 25 arenas can waste up to
1.6 GiB, versus 5 arenas and 0.3 GiB.

**(d) The only reason for per-layer `kv.l{i}` states is gone.**

- Option B (token-interleaved) was rejected in the fa3 review solely because the Triton writer at
  site 87 bakes `buffer_stride = 512`.
- kv_store is now handwritten with `slot_pitch` and `layer_off` (`tools/glm53/kernels/glm53_dsa.cu:76-84`).
- fa3's `stride_V` is a runtime i64 (pack bytes 168-191).
- The fa3 gather reads 1 KB contiguous per selected slot in either layout, so DRAM efficiency is
  equal.

**(e) The DeepGEMM idx descriptors work with the layered page.**

- Strides of 92,928 and 8,448 are multiples of 16. Bases at `i·8448` and `i·8448 + 8192` are
  16-B aligned.
- The span formula (K12) gives exactly P pages for every layer offset. The worst case is
  `10·8448 + 8448 = 92,928 ≤ stride`, which I checked by hand against `compile.rs:816-821`.

**(f) What unit 256 costs.**

- Lease granularity is 256 tokens (≤ 255 × 11.6 KB ≈ 3 MB of slack per sequence).
- The partial-page copy on restore is 256 × 11,627 B = 2.98 MB.
- Both are negligible.

**The one remaining constraint for later:**

- kern-serve takes a recurrent-state checkpoint only at request end, where `retire` does no copy
  (`scheduler.rs:709-742`).
- If we ever checkpoint *mid-sequence* (explicit breakpoints), the kpool is safe with unit 256.
- The **idx_tail ring and the KDA slot** are copied at `checkpoint`/`fork` time (`pages.rs:680`,
  `:772-775`). That is consistent with the pages as of `len`.
- **Do not** checkpoint inside a speculative round (K10): the per-sequence state is only
  consistent after the advance pass.

### 1.6 The per-sequence states: keep 3, do not fold idx_tail into KDA

- **The conv line stride is fixed.** The conv kernel bakes a 9,216-element line stride and indexes
  `line = table[l][col]` (`TC/LAWVJ2KQ…ttir:52`: `conv_states_base = idx·9216`). A combined state
  would need `bytes_per_seq` to be a multiple of lcm(18,432, 4,096, 524,288), with padding and odd
  line tables. That is not worth ≤ 64 MiB of rounding.
- **Per-layer per-seq states are the wrong alternative.** 34 × 2 + 11 arenas would round to up to
  79 × 64 MiB ≈ 5 GB.
- **Ring 8, not sglang's 4.** A verify round of k ≤ 4 drafts writes ring entries
  `(p+j)&7`, j ≤ 4. After a rollback to accepted length p+a+1, the entries still needed are
  positions in `[p+a−3, p+a]`. They differ from every rejected position by 1..7 < 8, so nothing
  needed is overwritten and **rollback is free** (§4.3). This costs +2 KB per layer per sequence.
  The ring is handwritten-kernel-private, so sglang parity is unaffected.
- **The per-sequence states are zeroed at lease time** (K8). A fresh sequence therefore starts
  from zero KDA state and an empty tail, which matches sglang's `_fused_slot_clear_kernel`
  (`DUMP/launches.jsonl` launch 707).

### 1.7 Invariants gen.py must assert

1. Every paged table has `stride = 256`. No input may `index_into` a paged state with any stride
   other than 1 or 256, because K1 silently changes the unit.
2. `W · 256 ≥ max served context`. `Denied::ExceedsRow` otherwise (`pages.rs:588-594`).
3. Every call-site pointer offset into a Triton kernel is a multiple of 16 B: every Triton pointer
   arg carries `tt.divisibility = 16` (§2.2).
4. Every DeepGEMM D, A and SFA buffer is `≥ S_MAX` rows, and D rows [M, S_MAX) alias nothing live
   (T7).
5. idx page math: `8448·i + 8448 ≤ 92928` for i ≤ 10. For 12 layers with MTP the page is
   `12·8448 = 101,376`, and the same holds.

---

## 2. Specialization-trap audit (every mined kernel we plan to use)

### 2.1 Method

1. **Triton kernels.** The capture compiled into `TC/`. For each cache entry I:
   - matched the cubin sha256 against `DUMP/module_*.cubin`;
   - read the `tt.func` signature from the `.ttir`. The arg names are in it; `{tt.divisibility =
     16}` marks divisibility-16 specialization, and a dropped int arg means it was specialized
     equal to 1;
   - diffed the TTIR bodies of sibling variants to find baked constexprs, e.g.
     `arith.constant 581`.
   - Rule: variants that differ only in a `d16` attribute **but compile to byte-identical cubins**
     are proven generic in that argument.
2. **sglang JIT (C++), DeepGEMM, fa3.** I read the template arguments from the mangled symbol and
   checked the header source for baked shapes (`SP/sglang/kernels/jit/csrc/**`,
   `SP/deep_gemm/include/deep_gemm/**`).
3. **TileLang kernels.** I checked which args are `T.dynamic` in `SP/sglang/kernels/ops/layernorm/mhc.py`,
   and the real ABI order in `TL/*/device_kernel.cu`.
4. **Grid scaling.** I compared recipe `grid` against `grid_b2`
   (`$GLM53_ARTIFACTS/recipe.json`), and diffed per-step params
   (`$GLM53_ARTIFACTS/tracks.json`). Across 6 steps only pointer offsets and
   `kpool_topk_transform`'s `page_table_stride` (p7 = 2..7, the growing seq length) change.
5. **PDL safety.** I scanned the SASS of each pinned entry for `ACQBULK` (griddepcontrol.wait) and
   `PREEXIT` (launch_dependents) (§2.4).

Verdict classes:

| Class | Meaning |
|---|---|
| SAFE | Shape-generic by construction. |
| RISK-CONSTEXPR | A shape is baked in. |
| RISK-DIV | A divisibility-16 assumption on a batch-dependent int. |
| RISK-LAYOUT | Output or pitch semantics the manifest mis-assumed. |
| RISK-GRID | A grid or SM count is baked in. |
| RISK-ABI | Wiring or argument order. |
| RISK-SEM | Semantics that differ from what the call site assumes. |

### 2.2 Audit table

**mHC**

| # | Kernel (module; skeleton pin) | Runtime shape args | Baked (relevant) | Verdict | Action / check |
|---|---|---|---|---|---|
| 1 | `sm90_tf32_hc_prenorm_gemm<24,16384,64,32,64,64(splits),…>` (module_125) | `shape_m`; A/B/D tensormaps | N = 24, K = 16384, 64 splits. D is a TMA 3-D store clipped at the tensormap (S_MAX pitch). `sqr_sum` is written at **pitch shape_m**, guarded `m < shape_m` (`sm90_tf32_hc_prenorm_gemm.cuh:114`, `:236-241`). | SAFE / RISK-LAYOUT (D6) | Run with `shape_m = S_MAX`. Check: probe record 0 (`sqrsum [64,97] stride [97,1]`). |
| 2 | `mhc_pre_big_fuse_with_norm` (module_127: n_splits = 64; module_231: n_splits = 8; per the team, not re-derived) | `num_tokens` (`mhc.py:958`) | hidden, hc, n_splits, gemm_last_dim = 24, eps, iters (`mhc.py:930-951`) | SAFE / RISK-ABI (D7) | Fix the arg order. Check: `grep -m1 __global__ TL/*/device_kernel.cu`. |
| 3 | `mhc_fused_post_pre_fma` (module_229, tile_mix_outputs = 2) | `num_tokens`, `split_k` (`mhc.py:1401-1402`) | hc, hidden, tile_n = 2 | SAFE / RISK-ABI (D7) | Keep `split_k = 8` for all S. sglang itself switches to tile 3 / split 4 at T ≥ 8 and to post + prenorm at T > 32 (`mhc.py:1710-1715`). kern is batch-invariant; bs ≥ 8 differs from sglang in the last bits. |
| 4 | `mhc_post_tilelang` (module_291) | `num_tokens` | hc, hidden, h_blk | SAFE | none |

**Quantization and fp8 GEMMs**

| # | Kernel (module; skeleton pin) | Runtime shape args | Baked (relevant) | Verdict | Action / check |
|---|---|---|---|---|---|
| 5 | `per_token_group_quant_flat<…rowmajor=F…>` (module_233); `<…rowmajor=T…>` (module_333) | 80-B `QuantKernelParams` (num_tokens, hidden, strides) (`quant.cuh:195-203`) | group 128, e4m3, ue8m0 = F | SAFE | grid = `cdiv(T·groups·8, 256)`. The team's formulas are right. |
| 6 | `sm90_fp8_gemm_1d2d<K-major,0,0,0,1,16,16|32,128,…,132,Normal,bf16>` (module_293 BN = 16; module_237 BN = 32) | M, N, K runtime (`SHAPE_* = 0`) | kNumSMs = 132; BLOCK_M = 16. The epilogue TMA-stores **all 16 rows** of the tile (`sm90_fp8_gemm_1d2d.cuh:430-441`). | SAFE with invariants (T7) | Grid is exactly 132. D rows ≥ M are junk. The brief's "never writes rows ≥ M" is **refuted**. |

**MoE and dense MLP**

| # | Kernel (module; skeleton pin) | Runtime shape args | Baked (relevant) | Verdict | Action / check |
|---|---|---|---|---|---|
| 7 | `silu_mul_clamp<bf16,PDL>` (module_279) | 32-B params (out_vecs, blocks_per_row) | limit is a runtime field | SAFE | none |
| 8 | `tiny_n_gemm<GEMMTraitN<288,4096,3,32>, M, f32, PDL>` (module_327; pin = **M = 1**) | none | **M is a template** (`tiny_gemm.cuh:71-72`) | RISK-CONSTEXPR (D3) | Use the `…ELj16EfLb1EEE` entry in the same module. Check bit-exactness: standalone harness, row 0 of M = 16 vs M = 1. |
| 9 | `_router_triton_kernel` (pin **module_329**) | 329: none. 411: `M`. | 329: `mask_m = pid < 1` constant | RISK-CONSTEXPR (D1) | Use **module_411**: +1 i32 at arg 5. Check: `grep -m1 "tt.func" TC/54LSD4DA*/_router_triton_kernel.ttir` shows `%M`. |
| 10 | `_moe_align_small_numel` (pin module_485, NP = 32; module_331, NP = 16) | `numel` | the padded tensor size `NP` | RISK-CONSTEXPR (D2) | Handwrite. Only the per-expert order changes; numerics do not, since every (token, k) row is independent and sum_reduce indexes `token·9 + k`. |
| 11 | `fused_moe_kernel` w13 (module_335) | N, K, EM, num_valid; strides | top_k = 9; BLOCK 64/128/128; 3 stages; MUL_ROUTED_WEIGHT = F | SAFE | 4 cache variants differing in EM / num_valid d16 are one identical cubin. The docstring labels args 7 and 13 as `bk`/`bsk`; they are `stride_bn`/`stride_bsn` (TTIR). The values 4096 and 32 are right. |
| 12 | `fused_moe_kernel` w2 (module_337 ≡ module_465, same sha 2a9ebb3803f4) | as above | top_k = 1, MUL_ROUTED_WEIGHT = T | SAFE | none |
| 13 | `_moe_sum_reduce_kernel` (module_487) | token_num, strides | `routed_scaling_factor = 2.5` | SAFE | 3 variants are one cubin. |

**KDA**

| # | Kernel (module; skeleton pin) | Runtime shape args | Baked (relevant) | Verdict | Action / check |
|---|---|---|---|---|---|
| 14 | `_causal_conv1d_update_kernel` (module_473) | `batch` only | **`num_cache_lines = 581`**, `seqlen = 1`, `pad_slot_id = −1`, all strides (x 3336, o 3072, state 9216) | RISK-CONSTEXPR (D4) | Recompile with num_cache_lines = 2^31−1 (§2.5). module_361 (output stride = x stride) does not fit our layout. |
| 15 | `fused_sigmoid_gating_delta_rule_update` (module_363; 8 cache keys, one cubin) | T (do_not_specialize), all strides, `stride_h0_source`, `cache_steps` | K = V = 128, 8 heads, lower_bound path | SAFE | A spec verify needs a no-store variant (§4.3). |
| 16 | `layer_norm_gated_fwd` (module_399 vs module_227) | eps, T | D = 128, BT = 32 | RISK-DIV (D5) | Pin **module_227**. |

**DSA indexer**

| # | Kernel (module; skeleton pin) | Runtime shape args | Baked (relevant) | Verdict | Action / check |
|---|---|---|---|---|---|
| 17 | `fast_hadamard_transform<16,7,bf16>` (module_365) | 56-B params | dim 128 | SAFE (no PDL) | grid = 32·S |
| 18 | `_act_quant_kernel` (module_367) | M/16, N/16 | block 128, round_scale (ue8m0) | SAFE | M = 32·S is always ≡ 0 mod 16 (no PDL). |
| 19 | `_kpool_decode_update…` (module_369 pin; module_441) | 369: no `block_tables_stride_0` (== 1). 441: runtime stride. | 369: `min(row, 0)`. 441: `min(row, 1)`. That is BLOCK_TABLE_COLS − 1 (TTIR diff). | **RISK-CONSTEXPR, both** | Handwrite (already started). Beyond 64/128 tokens every pool is written into page row 0 or 1. |
| 20 | `sm90_paged_mqa_logits_metadata<32,256,132,false>` (module_357) | batch_size, next_n, ctx ptr | kAlignedBatchSize = 32; kNumSMs = 132 | SAFE for batch ≤ 32 | Spec rows S·(k+1) ≤ 32 (bs8 × 4 = 32 OK). |
| 21 | `sm90_fp8_paged_mqa_logits<1,32,128,64,T,F,3,3,256,128,512>` (module_377) | batch_size, logits_stride, block_table_stride, ctx ptr | next_n = 1, 32 heads, dim 128, BLOCK_KV = 64, kNumSMs = 132 | SAFE | `logits_stride = W·64` (a multiple of 256), `block_table_stride = W`. Static per manifest, no per-step change. |
| 22 | `kpool_topk_transform_kernel<512>` (module_379) | dst_stride, out_cols, page_table_stride (i64), lengths / seq_lens ptrs (`kpool_topk_transform.cuh:220-234`) | K = 512 | SAFE / RISK-SEM | (a) The page table is token-granular: `page_table_entry[raw_token]` (`:206-216`). (b) Only ≤ 4,096 threshold-bin candidates are kept (`:63`, `:150-152`); the rest are dropped. (c) The output order comes from `atomicAdd`, so it is not deterministic (`:118`, `:131`). (d) Pads are −1. No PDL. See T9. |

**fa3**

| # | Kernel (module; skeleton pin) | Runtime shape args | Baked (relevant) | Verdict | Action / check |
|---|---|---|---|---|---|
| 23 | `prepare_varlen_num_blocks<1,true>` (module_319) | num_batch, seqlens | NumWarps = 1, so ≤ 31 batches (upstream `flash_prepare_scheduler.cu`: kNumBatchPerWarp = 31; UNVERIFIED for the fork) | SAFE (decode) | No `ACQBULK`: launch without PDL. A prefill with more than 31 rows or batches needs the `<32,false>`-class entry. |
| 24 | `FlashAttnFwdSm90<…(64,64,64),512,…OnlyQv,PackGQA,Split…>` (module_383) | num_batch, seqused_k, cu_seqlens | page-table width 2051 (fine), split cap 29 | SAFE | ABI as in the fa3 review. For §1.2: `stride_V = stride_K = 5632`, `ptr_V = kv + i·1024`. |
| 25 | `FlashAttnFwdCombine<(8,128),5,256,1,F,T,bf16,f32,Sm90>` (module_385) | params (seqlen divmod, heads) | kLogMaxSplits = 5 (≤ 32 splits) | SAFE | grid `[1,4,S]` |

**Replaced (not minable or not portable)**

| # | Kernel (module; skeleton pin) | Runtime shape args | Baked (relevant) | Verdict | Action / check |
|---|---|---|---|---|---|
| 26 | `all_reduce_1shot_push` (module_123) | none | 8 IPC workspace ptrs, counter, rank inside the 112-B params | not portable | Replace (§4.2a). |
| 27 | `_vocab_parallel_embedding` (module_121) | none | **the rank's vocab window** (constexpr, `moe_head.md` §2), so rank-specific | not portable | Replaced by `glm53_embedding` over a replicated table. |
| 28 | `set_mla_kv_buffer_norope` (module_381 / 317) | loc | buffer_stride = 512 (381) or 2048 (317) | RISK-CONSTEXPR | Replaced by `glm53_kv_store`. |
| 29 | flashinfer CuTe RMSNorm / LayerNorm (307, 315, 311, 343) | none | shapes are encoded in the symbol (`…153615361…`) | RISK-CONSTEXPR | Replaced (mind D9). |
| 30 | nvjet / cublasLt (qkvbfg, fg_b, o_proj KDA, wq_b, wk, gate, bmm, lm_head) | not minable (np = 0) | n/a | n/a | `extern:cublaslt_bf16_tn`. The algorithm may differ from sglang's nvjet or splitK choice (T12). |
| 31 | `track_mamba_states_all_layers` (387/489) | none | radix bookkeeping | n/a | Not needed. kern checkpoints via retire. |

### 2.3 The confirmed traps, in detail

- **T1 (D1) router, module_329.** The TTIR `329 vs 411` diff:
  `%mask_m = arith.constant 1 : i32; %mask_m_12 = arith.cmpi slt, %pid, %mask_m` versus
  `%mask_m = arith.cmpi slt, %pid, %M`. At S > 1 rows 1..S−1 keep stale expert ids and weights,
  so the MoE reads garbage routing. This is silent.
- **T2 (D2) align NP.** `tt.make_range {end = 16}` / `{end = 32}`, and `%mask_p = offs_p < numel`.
  Entries beyond NP are simply not sorted, so their tokens are never computed.
- **T3 (D3) tiny_n_gemm M.** Each block loads `xv[M][kUnroll]` and writes `out[m·N + …]` for
  m < M only (`tiny_gemm.cuh:103-140`).
  - The M = 16 entry computes 16 rows. Rows ≥ S read junk x rows that are in bounds (S_MAX = 16).
  - The weights (2.36 MB) dominate either way, so there is no cost difference.
  - The per-row order (per-thread fma over kUnroll, warp reduce, then a sum over warps in order)
    is identical. Expected bit-exact: **UNVERIFIED**. Check with a standalone
    `cuModuleGetFunction` harness comparing both entries on random x.
- **T4 (D4) conv num_cache_lines.** For conv lines idx ≥ 581:
  - The output taps `col0..col2` are loaded under `mask_w` only, so this step's output is still
    right (`causal_conv1d_triton.py:700-714`).
  - The shift-load of the window is masked with `idx < num_cache_lines` and returns 0 (`:727-732`).
  - The store that follows is **not** masked by the bound (`:760-764`).
  - So every step writes `[0, 0, x_t]` and wipes the 3-token history. From the next step on the
    conv silently sees zero history. That is a quality bug, not a crash.
- **T5 (D5) norm_gated d16.** The TTIR is identical apart from `%T {tt.divisibility = 16}`, yet
  the cubins differ, so the backend used it. Do not rely on undefined behaviour.
- **T6 (D6) prenorm sqr_sum pitch.** See §0.2. Also note that `glm53_hc_repack` is launched with
  grid [64, 2] and 384 threads, so it covers only `t < S·24 ≤ 384` (S ≤ 16). That is fine, but it
  becomes moot once the op is deleted.
- **T7 (1d2d and prenorm junk rows).**
  - The TMA store box is BLOCK_M × TMA_D_BLOCK_N and is clipped by the **tensormap's** M dim.
  - Hence rows [M, S_MAX) of every D buffer get junk (possibly NaN, since the A and SFA rows there
    are stale).
  - This is harmless because GEMM rows are independent, **provided** that:
    - (i) every D buffer has ≥ S_MAX rows. For prefill ops the tensormap M dim must equal the
      prefill buffer's rows, not 16;
    - (ii) no live data aliases those rows;
    - (iii) nothing reduces over the row axis (the allreduce and argmax take `S·4096` and `[S]`
      only).
  - At M ≤ 16 the scheduler still sees one M-block, and `is_computation_valid` gates only whole
    WGMMA waves (`sm90_fp8_gemm_1d2d.cuh:260-275`).
- **T8 (D7) TileLang ABI.** TileLang orders buffer params by name. The recipe confirms it:
  - site 5: `p0 = +90112` is `comb_mix`, the output later read by site 14's `prev_comb_mix`, and
    `p2 = +90624` is the prenorm `sqr_sum`, which is site 4's p4.
  - site 14: `p3` is the `hc_ffn_fn` weight and `p6 = +30208` is `prev_residual`.
- **T9 (topk semantics).** Three consequences for kern:
  - (i) The token → slot table must exist. `glm53_dsa_prep` now builds `[S, 64·bt_cols]` per step,
    bounded by seq (`tools/glm53/kernels/glm53_dsa.cu:93-116`). That is ≈10 µs at 141k with one 128-thread CTA per
    sequence.
    - **P1:** recompile `kpool_topk_transform.cuh` with `transform_kpool_token = bt[t>>8]·256 +
      (t&255)`.
    - **P1:** pad with 0, which deletes the `clamp0` launch (site 88).
  - (ii) The 4,096-candidate cap means sglang's top-k is approximate when more than 4,096 pools
    share the threshold bin (fp16 top byte). kern must keep the same kernel for parity.
  - (iii) The selected slot order is non-deterministic, so fa3's online-softmax order varies. Even
    sglang is not run-to-run bitwise reproducible at context > 2051. The parity oracle must use
    tolerances beyond 2051 tokens (§5 R3).
- **T10 (kpool_update, both variants).**
  - 369: `packed_page = bt_ptr + row` (stride 1 baked) and `min(page_row, 0)`.
  - 441: `row·stride` but `min(page_row, 1)`.
  - Every pool beyond the first 64 or 128 tokens lands in the wrong page.
  - Handwritten replacement: see §0.2 for its three issues (lowering, unit 256, pad guard).

### 2.4 PDL safety (launch with `"pdl": true` only if the entry waits)

kern turns `pdl: true` into a programmatic edge (`exec.rs:256-264`). A kernel that never executes
`griddepcontrol.wait` would then read its inputs before the producer finishes.

| Entry | ACQBULK / PREEXIT | pdl? |
|---|---|---|
| prenorm 125, big_fuse 127/231, fma 229, post 291, quant 233/333, 1d2d 237/293, silu 279, tiny_gemm 327 (all 16 M entries), router 329/411, align 331/485, fused_moe 335/337/437/465, sum 487, delta 363, conv 361/473, norm_gated 227/399, mqa metadata 357, mqa logits 377, fa3 383 (site-90 entry), combine 385, AR 123 | 1 / 0-or-1 | **yes** |
| embedding 121, hadamard 365, act_quant 367, kpool_update 369/441, topk 379 | 0 / 0 | **no** |
| prepare_varlen 319 `<1,true>` | 0 / 1 (triggers early, never waits) | **no** (its successor, fa3, waits) |
| handwritten `glm53_*` | 0 today | Add `griddepcontrol.wait` at entry and `launch_dependents` after the last global read. Then set pdl = true. |

Check (per entry):

```bash
cuobjdump -sass DUMP/module_N.cubin | awk '/Function :/{f=$0} /ACQBULK|PREEXIT/{print f": "$0}'
```

gen.py should run this scan and refuse `pdl: true` without `ACQBULK`.

### 2.5 How to "re-mine" a generic variant without a GPU

The capture only ever JIT-compiled what bs1 and bs2 needed. Every kernel in §2.2 is open source in
the venv and can be recompiled offline for sm_90a, with no GPU memory. This is the preferred
remedy over handwriting whenever the fix is a constexpr:

- **Triton (conv, router if needed, norm_gated, fused_moe decode config).** Use
  `triton.compile(ASTSource(fn, signature, constexprs, attrs), target=GPUTarget("cuda", 90, 32))`
  with Triton 3.7.1 from the venv. Pass the *captured* constexprs, except the one being
  generalized (`num_cache_lines`, `BLOCK_SIZE_N`, …). Mark batch-dependent ints non-d16 and not
  equal to 1.
  - Then diff the TTIR against the captured variant: only the intended constant may change.
- **sglang JIT C++ (tiny_gemm, topk transform, quant, silu, hadamard).** Build the `.cuh` with the
  same nvcc flags sglang's JIT uses; the cache holds the command lines under `$HOME/.cache/sglang/jit/`.
- **DeepGEMM.** Build the template (e.g. prenorm with other kNumSplits, 1d2d other BLOCK_N)
  through `deep_gemm`'s JIT build path offline.
- **TileLang.** Use `tilelang.compile` on the `mhc.py` function with the same `pass_configs`.
- **Pinning.** Every re-mined cubin is pinned by sha256 exactly like a mined one
  (`crates/kern-runtime/src/compile.rs:326-374`: identity is the hash; param layout is checked).

---

## 3. Decode program structure

### 3.1 Vars and programs (protocol-legal)

- **vars**
  - `tokens`: rows. Max 16 for decode; `8192` if the prefill chunk program shares the var.
  - `seqs`: groups, max 16.
  - `span`: P2, optional.
- The protocol demands that `slot` be over the rows var and `seq_len` over the groups var, and
  that these differ (`protocol.rs:264-273`).
- **programs**

| program | batch | graph | notes |
|---|---|---|---|
| `load` | none | no (`once: true`) | Weight derivations (f32 conv, fn tables). The Lamport buffer poison init also runs here. |
| `decode` | `{groups: 16, rows: 1}` | **yes** | kern-serve pads to buckets {1, 2, 4, 8} (`scheduler.rs:99-109`) and captures one graph per bucket value of `seqs` (`exec.rs:62-83`). Every kernel must be S-generic (§2). |
| `prefill` | `{groups: 1, rows: "tokens"}` | no | Chunk program run launch by launch (`scheduler.rs:33-35`). §3.6. |
| `round_k2` | `{groups: 8, rows: 3}` | yes | P3 MTP (§4.3). Writes `tokens [seqs,3]` and `count [seqs]` (`protocol.rs:506-549`). |
| `decode_span` | `{groups: 16, rows: 1, span: "span"}` | yes | P2, optional. Carries a short prompt chunk (a tool output) inside a decode step (`protocol.rs:37-44`). The recurrent kernels see the run as one sequence. |

Why one var-seqs `decode` and not four bucketed programs:

- After the §2 fixes every kernel is S-generic.
- Graphs are already per bucket (`Dense` keys the graph by var values, `compile.rs:36-65`).
- Bucketed programs would only be needed for M-templated kernels. The one we have (tiny_n_gemm)
  is handled by the M = 16 entry.

### 3.2 The per-step DAG

The DAG is shown here at P0/P1 fidelity, with launch counts. The only run-time conditions are the
`kda`/`dsa` layer type and the `dense`/`moe` FFN type, and gen.py resolves both statically.

```
ONCE PER STEP (prologue)
  embed            glm53_embedding(token_ids -> x)                               1
  hc_expand        x -> R_a[S,4,4096]                                            1
  dsa_prep         seq_lens -> pool_lens, pool_ctx, dsa_lens (+ slot table, P0)  1
  mqa_sched   (P1) sm90_paged_mqa_logits_metadata(pool_ctx) -> sched[133,2]      1   (P0: per DSA layer)
  fa3_sched   (P1) prepare_varlen(dsa_lens, cu_seqlens) -> {nsd,nmb,vbi,sem}     1   (P0: per DSA layer; UNVERIFIED hoist, see note)

PER LAYER L (45x); R_a/R_b ping-pong, [S_MAX,4,4096] bf16 each
  attn-pre   L==0: hc_prenorm(R_a, hc_attn_fn, M=S_MAX) -> mul64,sqr64          1
                   hc_big_fuse64(T=S_MAX)  -> post,comb,x_norm                  1
             L>0 : [already produced by the previous layer's ffn->attn boundary]
  ATTN (KDA x34 | DSA x11)                                     7 | 40 (P0) / 26 (P1)
  AR_attn    Lamport one-shot (bf16 in, f32 rank-order sum)                     1
  attn->ffn  hc_fma(comb,R_a,post,attn_out, hc_ffn_fn; split 8) -> R_b, mul8,sqr8   1
             hc_big_fuse8(T=S) -> post,comb,x_norm (post_attention_layernorm)   1
  FFN (dense L<3 | MoE L>=3)                                                5 | 9
  AR_ffn                                                                        1
  ffn->attn  L<44 (P0): hc_post(R_b)->R_a; hc_prenorm(L+1); hc_big_fuse64       3
             L<44 (P1): hc_fma(comb,R_b,post,ffn_out, hc_attn_fn[L+1]) -> R_a, mul8,sqr8   1
                        hc_big_fuse8 (input_layernorm[L+1])                      1
             L==44    : hc_post(R_b) -> R_a                                     1

ONCE PER STEP (epilogue)
  hc_contract  mean over 4 streams -> h                                          1
  final_norm   glm53_rms_norm(h, model.norm)                                     1
  lm_head      cublasLt [S,4096]x[19360,4096]^T -> logits_shard                  1
  argmax  P0:  nccl_allgather(310 KB/seq) + cast + argmax                        3
          P1:  local (max,idx) per shard -> 8-B/seq allgather -> pick           2
```

Notes on the DAG:

- **fma at the ffn→attn boundary (P1).** This deviates numerically from sglang. sglang runs
  `hc_post` → DeepGEMM TF32 prenorm (64 splits) → `big_fuse64` there (`kda_mhc.md` §6.5). fma is
  the same math with 8 fp32 splits: the products are exact because the hc_fn table is
  bf16-representable, so only the summation order changes. Keep the sglang order in P0 for
  bitwise cuts. Switch in P1 once greedy agreement is proven.
- **Hoisting fa3_sched.** Hoisting prepare_varlen to once per step is valid only if combine resets
  `tile_count_semaphore` after every fa3 launch. Combine's params carry `semaphore_to_reset`
  (`ops_dsa.py:429`). **UNVERIFIED** for the fork. Check with a standalone harness: prepare once,
  then 2× (fa3 + combine), and compare against per-layer prepare. Otherwise keep prepare per
  layer; it is 1 warp, ≈2 µs.

**KDA attention (7 launches, `kda_mhc.md` §3; wiring `ops_kda.py`)**

| # | op | kernel | notes |
|---|---|---|---|
| 1 | qkvbfg | cublasLt `[S,4096]×[3336,4096]ᵀ` → `F[S_MAX,3336]` | |
| 2, 3 | fg_b | 2 × cublasLt (f_a → forget `[S,1024]`, g_a → gproj `[S,1024]`), a_stride 3336 | |
| 4 | conv | re-mined conv (D4): x = F[:, :3072] (pitch 3336), state = kda_conv line `table[l]` → `q\|k\|v [S,3072]` | |
| 5 | delta | module_363: q/k/v views (+0 / +2048 / +4096 B), a = forget, b = F + 6144 B, h0 = kda_ssm line `table[l]`, cu_seqlens → o `[S,8,128]` | |
| 6 | norm_gated | module_227: o × σ(gproj), T = 8S | |
| 7 | o_proj | cublasLt `[S,1024]×[4096,1024]ᵀ` → attn_out `[S,4096]` | partial, to AR |

**DSA attention (40 launches in P0 as wired today, 26 with the P1 grouped GEMV)**

| # | op | kernel | notes |
|---|---|---|---|
| 1 | quant_x | module_233 → `a8 [S_MAX,4096]`, `sfa [32,S_MAX]` | |
| 2 | qkv_a | module_293 N = 2048 K = 4096 → `QKV [S_MAX,2048]` | |
| 3 | q_a_norm | glm53_rms_norm d = 1536, **pitch 2048** (D9) → `qa [S,1536]` | |
| 4 | kv_a_norm | d = 512, pitch 2048 → `knope [S,512]` | fused kv_store in P1 |
| 5 | quant_q | module_233 → `qa8`, `sfa12` | |
| 6 | q_b | module_293 N = 2048 K = 1536 → `q [S_MAX, 8×256]` | |
| 7 | wq_b | cublasLt → `iq [S,4096]` | |
| 8 | wk | cublasLt → `ik [S,128]` | |
| 9 | k_norm | glm53_layer_norm_128 | |
| 10 | hadamard | module_365 on iq | |
| 11 | act_quant | module_367 → `iq8 [S,32,128]`, `qs [S,32]` | |
| 12 | kpool_gate | cublasLt → `gs [S,128]` | |
| 13 | kpool_update | handwritten (unit 256, ring 8) | writes idx page, idx_tail line |
| 14 | weights | glm53_weights_proj → `w [S,32]` f32 | |
| 15 | (mqa_sched) | per-step op | |
| 16 | logits | module_377 → `lg [S_MAX, W·64]` | |
| 17 | topk | module_379 → `tk [S_MAX, 2051]` | |
| 18 | clamp0 | glm53_clamp0 | P1: folded into recompiled topk |
| 19 | w_kc | 8 × cublasLt per head (`ops_dsa.py:442-448`) → `qv [S,8,512]` token-major | P1: one grouped GEMV (−7) |
| 20 | kv_store | glm53_kv_store(slot_pitch 11264, layer_off i·1024) | |
| 21 | dsa_attn | **one op**: prepare_varlen → fa3 (2944-B pack) → combine | D8 |
| 22 | w_vc | 8 × cublasLt → `av [S,2048]` | P1: grouped GEMV |
| 23 | quant_o | module_233 | |
| 24 | o_proj | module_237 N = 4096 K = 2048 → attn_out | |

In P0, rows 19 and 22 add +14 launches per layer over sglang's single bmm. That is why the grouped
GEMV is a P1 item.

**MoE (9 launches, `moe_head.md` §4, with the D1/D2/D3 fixes)**

| # | op | kernel |
|---|---|---|
| 1 | router | tiny_n_gemm M = 16 entry → `scores [S_MAX,288]` f32 |
| 2 | topk | module_411 (M = S) → `ids [S,9]`, `wts [S,9]` |
| 3 | align | handwritten → sorted `[9S·64]`, expert_ids, n_post |
| 4 | quant_a | module_333 → `a8`, `as [S,32]` |
| 5 | w13 | module_335 → `c1 [9S,512]` |
| 6 | silu | module_279 → `[9S,256]` |
| 7 | quant_b | module_333 → `[9S,256]` fp8, `[9S,2]` |
| 8 | w2 | module_337 → `c2 [9S,4096]` |
| 9 | sum_reduce | module_487 → ffn_out `[S,4096]` |

**Dense MLP (layers 0–2):** quant → 1d2d gate_up (module_237) → silu → quant → 1d2d down → 5
launches.

**Launch counts per step:**

| phase | computation | launches |
|---|---|---|
| P0 | 3 (prologue) + 90 (AR) + 238 (KDA 34×7) + 440 (DSA 11×40) + 15 (dense 3×5) + 378 (MoE 42×9) + 225 (mHC 2 + 45×2 + 44×3 + 1) + 6 (epilogue) | ≈1,395 (more than sglang's 1,351 because of the 16 per-head bmm GEMMs) |
| P1 | P0 − 10 (hoisted mqa_sched) − 44 (fma boundary) − 154 (grouped GEMV) − 11 (clamp folded into topk) − 11 (kv_store fused into the kv_a norm) − 1 (argmax-first) | ≈1,165 |
| P2 | mHC 4/layer (AR fused) 180 + KDA 3×34 = 102 + DSA ≈15×11 = 165 + MoE 3×42 = 126 + dense 9 + pro/epilogue 7 | ≈590 (§4.4) |

### 3.3 Buffer plan

Workspace sizing:

- Workspace buffers are sized at var max (`load.rs:156-168`).
- Activations use the S_MAX = 16 static pitch.
- Nothing per-layer: every layer reuses the same activation buffers (sequential stream order).

Scratch and carries:

- Op-private scratch (K14) is used only for intra-op temporaries. Examples: the fa3
  `sched`/`oacc`/`lseacc`/`lse` inside the single `dsa_attn` op, and the DeepGEMM schedule inside
  `mqa`.
- Nothing crosses steps except the states.
- `streams` does **not** need to be a `carry` (`gen.py:51`). The residual is rebuilt from the
  embedding every step, so it can be workspace.

| buffer | shape (dtype) | producer → consumers | shared between attn-pre and ffn-pre? |
|---|---|---|---|
| `R_a`, `R_b` | [16,4,4096] bf16, 512 KB each | ping-pong: expand/fma/post write one, and big_fuse/fma/post read the other (fma and post must not run in place: 12 × 8 CTAs of fma read the same residual slice that others overwrite) | n/a |
| `mix_post [16,4]`, `mix_comb [16,16]` | f32 | big_fuse → next fma/post | **yes**, one pair. The attn-pre set is dead once fma(attn→ffn) has read it, and big_fuse(ffn-pre) writes after that fma completes: big_fuse waits on ACQBULK. |
| `part8_mul [8,16,24]`, `part8_sqr [8,16]` | f32 | fma → big_fuse8 | **yes** (both boundaries) |
| `part64_mul [64,16,24]`, `part64_sqr [64,16]` | f32 | prenorm → big_fuse64 (layer 0; P0 also every ffn→attn) | yes |
| `x_norm` | [16,4096] bf16 | big_fuse → the sublayer's first GEMMs | **yes** (attn input and ffn input have disjoint lifetimes) |
| `a8 [16,4096]` u8 + `sfa [32,16]` f32 | | quant → 1d2d / fused_moe | yes (DSA qkv_a, dense, MoE a1) |
| `sub_out` | [16,4096] bf16 | o_proj / MoE sum → AR (in place) → fma | yes (attn and ffn) |
| KDA: `F [16,3336]`, `fg [2][16,1024]`, `qkvc [16,3072]`, `o [16,1024]` | bf16 | within the KDA block | no |
| DSA: `QKV [16,2048]`, `qa [16,1536]`, `knope [16,512]`, `qa8`, `q [16,2048]`, `iq [16,4096]`, `iq8`, `qs [16,32]`, `ik`, `gs [16,128]`, `w [16,32]`, `lg [16, W·64]` f32 (4 MB at W = 1024), `tk [16,2051]` i32, `qv [16,4096]`, `ao [16,4096]`, `av [16,2048]` | | within the DSA block | no |
| MoE: `scores [16,288]` f32, `ids/wts [16,9]`, `sorted [144·64]` i32, `c1 [144,512]`, `h1 [144,256]`, `h1_8`, `c2 [144,4096]` bf16 | | within the MoE block | no |
| per-step: `pool_lens/pool_ctx/dsa_lens [16]` i32, `mqa_sched [133,2]`, `fa3 sched` (in the `dsa_attn` op) | | dsa_prep → all 11 DSA layers | n/a |

Invariant T7: every DeepGEMM D and A tensormap uses M dim = 16 in decode ops. `S_MAX` must equal
the max of the `seqs` var used by decode (16).

### 3.4 Op granularity and how gen.py composes layers

**P0: one op per kernel.** Use one op per kernel (one call per launch), except the fa3 triple
(D8). The reason is observability. `kern test` compares every buffer at every call boundary
("cut") against the oracle (`NOTES.md`, "bit-identical at every cut"). Fine cuts localize a
mismatch to one kernel.

**P2: fused impls behind the same stage interface.** A fused impl replaces a *sequence* of calls
while keeping the stage's input and output buffers (e.g. the MoE stage `x_norm → sub_out`). The
oracle cuts at stage boundaries therefore stay comparable. Intra-stage temporaries move into op
scratch.

**gen.py structure.**

- A `layer(L)` function emits `pre(L)`, `attn(L)` (`kda(l)` or `dsa(i)`), `ar`, `boundary(L)`,
  `ffn(L)`, `ar`, and `exit(L)`.
- Per-layer differences are only:
  - weight buffer names (`bind`);
  - state call offsets: `kv` offset `i·1024`; `idx` offset `i·8448` for the DeepGEMM
    descriptors and the kpool kernel;
  - line-table row offsets: `row × 16 × 4 B` into `kda_*_lines` / `idx_tail_lines`, because the
    driver stages them at pitch `groups.max`, K13.
- One op declaration per kernel variant serves all its layers. The tensormaps are encoded per call
  (`compile.rs:616-698`).
- gen.py must also assert the invariants of §1.7 and the PDL rule of §2.4.

### 3.5 CUDA-graph boundaries: what is graph-safe

- **The whole `decode` step is one graph** (`"graph": true`, `types.rs:151-153`). This is already
  what sglang does ("cuda graph: True" in its log), so capture alone is **not** a kern advantage.
  The advantages are the PDL edges (`exec.rs:256-264`) and fewer nodes.
- **Fills, page tables and line tables** are H2D writes on the compute stream before
  `cuGraphLaunch`. The graph reads buffer contents at replay (`exec.rs:12-17`, `:101-105`;
  `lib.rs:258-279`). So every per-step value (positions, lengths, slots, tables) **must reach
  kernels through device buffers**.
  - Scalars and vars are baked at capture.
  - Therefore no `context` var in decode (`types.rs:169-171`).
  - `tokens`/`seqs` are fine: one graph per bucket.
- **NCCL externs** are capturable: kern pins `NCCL_RUNTIME_CONNECT=0` so connects never happen in a
  capture, and `NCCL_PROTO=LL128` (`docs/multi-gpu.md:279-285`). But decode should use the Lamport
  kernel anyway (§4.2a).
- **Lamport / peer kernels** are capturable. The epoch lives in a carry the kernel advances itself
  (`tools/kernels-src/peer_barrier.cu:1-14`; `tools/kernels-src/peer_allreduce.cu` stage rotation).
  The poison init runs once in `load`.
- **cuBLASLt externs** are capturable. The heuristic is chosen at capture, per bucket.
- **Pool remaps** run on their own thread and stream. State pointers span the reservation, so
  graphs survive a remap (`compile.rs:713-721`). Fresh chunks are zeroed on the stream before use
  (`lease.rs:212-254`).
- **What would break capture or correctness:**
  - any host readback mid-step (none in the design);
  - a context-length scalar;
  - allocating scratch per call (kern never does);
  - a kernel with `pdl: true` that lacks `ACQBULK` (§2.4).

### 3.6 Prefill organization

**sglang's mHC switch.** sglang changes the mHC decomposition with T:

- the fused attn→ffn boundary is used only when `T ≤ _MHC_FUSED_BOUNDARY_MAX_TOKENS = 16`
  (`SP/sglang/srt/models/glm5_next.py:140`, `:856-863`);
- inside `mhc_fused_post_pre`, T > 32 switches the FMA kernel to `mhc_post` plus the DeepGEMM
  prenorm with `n_splits = max(1, min(132 // ⌈T/64⌉, 64))` (`SP/sglang/kernels/ops/layernorm/mhc.py:1708-1715`,
  `:853-860`);
- the `big_fuse` n_splits constexpr follows it: 64 for T ≤ 128, 33 at 256, … 1 at 8192.

**Recommendation: a separate `prefill` chunk program** (T ≤ 8192, eager). It must not reuse the
decode ops, for three reasons:

1. **mHC:** use the unfused chain at every boundary: `mhc_post` (T dynamic) → prenorm → big_fuse.
   - Use the **mined kNumSplits = 64 prenorm and big_fuse64 for all T**. The grid is
     `⌈T/64⌉ × 64` CTAs. The extra partial traffic at T = 8192 is 2 × 50 MB per call, ≈1% of a
     chunk. This avoids JIT-compiling the n_splits = 1 variants.
   - The D tensormap M dim is the prefill buffer's rows (8192). big_fuse reads `[64, T, 24]` at
     pitch T. So either (a) recompile big_fuse with a static partial pitch (TileLang source,
     `mhc.py:930`), or (b) run a repack of `mul` only (6 MB at T = 1024).
   - sqr_sum is compact at pitch T already (D6).
2. **KDA:** the decode delta kernel is recurrent per token. At T = 8192 it would run 8192
   sequential steps per head, ≈200 ms/chunk for 34 layers, while sglang's whole chunk costs
   ≈250 ms. Use sglang's chunked kernels, all present in `TC/` and `DUMP/`:
   `kda_gate_chunk_cumsum_vector`, `chunk_kda_fwd_kernel_intra_sub_chunk`,
   `chunk_kda_fwd_kernel_inter_solve_fused`, `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`,
   `chunk_gla_fwd_kernel_o`, `l2norm_fwd_kernel`, `_causal_conv1d_fwd_kernel`.
   - Audit them exactly as in §2. Their T-dependent specializations are unaudited.
   - The first chunk must use `has_initial_state = False`. The slot is zeroed by the lease (K8),
     so True-with-zeros is also correct.
3. **DSA:** decode kernels cap the batch. mqa metadata allows ≤ 32 rows and prepare `<1,true>`
   allows ≤ 31. Long prompts need the prefill path that sglang used (`prefill=flashmla_sparse`,
   `serve_sglang.log:1`): DeepGEMM `fp8_mqa_logits` (ragged), top-k over T query rows,
   `flash_mla_sparse_fwd`, and `_kpool_assemble_softmax_rotate_write_cache` plus
   `_scatter_kpool_tail_updates` to build pooled k.
   - **The capture has none of the long-context sparse kernels.** Its prompts were short, and it
     used dense FA3 `<128,80,256>` over kv_b-expanded K/V (`DUMP/launches.jsonl` 219347+, 264866+).
   - **Action:** capture a ≥ 16k-token prompt with `tools/capture_sglang_glm53.sh` before writing
     the prefill manifest.
4. **MoE at large M:** sglang's large-M path is `moe_align_block_size` + `count_and_sort` +
   `fused_moe` (large-M configs) + `moe_sum_reduce<bf16,9>`, all in the dump (264879-264886).
   The DeepGEMM 1d2d is shape-generic (SHAPE_M = 0). Pick one BLOCK config for all T
   (e.g. `<…,128,128,…>`, present in the dump) and set the tensormap M = `tokens` max.

**Steady-state agentic turns.**

- Each turn adds a short suffix (tool output, 100–3k tokens) to a restored 141k context.
- These should ride decode steps as a **span** (`decode_span`, `protocol.rs:37-44`) rather than
  interrupt decoding with a prefill call.
- The KDA kernels see the span as one sequence (the delta rule is varlen; conv needs a
  `seqlen = L` variant). The DSA kernels treat each span row as its own query with its own length.
  That needs the per-row `dsa_prep` and batch-generic metadata `<64,…>`/prepare variants.
- This is P2. P0 and P1 can use the prefill program for suffixes.

**MTP prefill (P3).** The `prefill` program must also run layer 45 over the chunk, fed with
`emb(t+1)` and `h(t)`, to fill its kv/idx/tail. That adds about 2.5% of prefill cost.

---

## 4. Performance roadmap

### 4.1 Where sglang's 5.7 ms goes (bs1, 141k context), and the floor

**Bytes streamed per step, per rank:**

| item | bs1 | bs4 | bs8 | basis |
|---|---|---|---|---|
| MoE experts (distinct × 3 MiB × 42 layers) | 9 → 1.19 GB | ≈31.3 → 4.14 GB | ≈58.5 → 7.73 GB | distinct = 288·(1−(287/288)^(8S)) + 1, uniform-routing assumption (real routing is skewed, so this is an upper bound) |
| non-MoE weights | 2.12 GB | 2.12 | 2.12 | KDA 1.23 (bf16), DSA 0.43, mHC fn 0.14 (f32), router 0.10, dense 0.06, lm_head 0.16 |
| fa3 KV (2051 × 1 KB × 11 layers) | 0.02 GB | 0.09 | 0.19 | |
| kpool (35,280 × 132 B × 11) | 0.05 GB | 0.21 | 0.41 | |
| KDA state (read + write 1.05 MB × 34) | 0.04 GB | 0.14 | 0.29 | |
| **total → floor at 2.8 TB/s** | **3.42 GB → 1.22 ms** | 6.71 GB → 2.40 ms | 10.73 GB → 3.83 ms | |

**sglang latency model.** This is my model, not a measurement. Each launch is priced at
bandwidth time or at the small-kernel latency floor of ≈2–3 µs in a graph, whichever is larger.
The per-launch prices come from the recipe grids and the sizes above. It reproduces the logged
bs1 and bs4 numbers.

| component | launches/step | per-call (bs1) | bs1 ms | bs4 ms | bs8 ms |
|---|---|---|---|---|---|
| mHC chain (prenorm 4, big_fuse 6, fma 4, big_fuse 6, post 3 µs) ×45 | 225 | 23 µs/layer; the 20-iteration Sinkhorn on 1 warp is ~2–3 µs of each big_fuse | 1.04 | 1.06 | 1.08 |
| TP allreduce (sglang 1-shot push) ×91 | 91 | 7 µs (8 KB) → 9 µs (64 KB) | 0.64 | 0.73 | 0.82 |
| KDA attention ×34 (qkvbfg 11 + reduce 2, fg_b 3, conv 2.5, delta 4, norm 2, o_proj 4) | 238 | 28.5 µs | 0.97 | 1.00 | 1.04 |
| DSA projections and glue ×11 | 330 | 71 µs | 0.78 | 0.80 | 0.82 |
| DSA indexer ×11: mqa_logits (35k pools: 138 × 256-slot segments/row) + kpool_topk radix | 22 | logits 8 → 18 µs; **topk ≈20 µs (UNVERIFIED)**: one 1024-thread CTA/row, ≥ 3 passes over 141 KB | 0.31 | 0.35 | 0.42 |
| fa3 ×11 (prepare 3 + fwd over 2051 slots 8 → 12 + combine 3) | 33 | 14 µs | 0.15 | 0.18 | 0.20 |
| MoE ×42 (router 3, topk 3, cat 2, align 3, quant 2, w13 12 → 42, silu 2, quant 2, w2 5 → 22, sum 2) | 462 | 36 → 83 µs. w13 at bs1 has only 36 CTAs (9 × 4 N-tiles), so it is occupancy-bound. | 1.51 | 2.31 | 3.49 |
| dense MLP ×3 | 15 | 14.5 µs | 0.04 | 0.05 | 0.06 |
| head (embed, expand, contract, norm, lm_head 159 MB, allgather, cast, argmax) | 9 | | 0.08 | 0.09 | 0.10 |
| **total (model)** | **1,351** | | **5.52** | **6.57** | **8.03** |
| **measured** (`serve_sglang.log:3299-3300`, `:3320`) | | | **5.68** | **6.9** | n/a (bs6 mixed ≈ 9.5) |
| the brief's figures | | | 12 | | 25 |

**Conclusion.** At bs ≤ 8 sglang runs at 4.5× (bs1) to 2.1× (bs8) above the bandwidth floor. The
excess is launch latency and serialization across 1,351 dependent kernels. Attention and the
indexer are only 0.46 ms at bs1: long context is **not** the problem at 141k with DSA. The MoE's
bytes become the problem only at bs ≥ 4.

### 4.2 Where kern can beat sglang

**(a) Allreduce: 2 per layer, 91 per step.**

- **Today's generator uses `nccl_allreduce_bf16`** (`ops_kda.py:113`, `ops_moe.py:196`,
  `ops_dsa.py:468`).
  - kern pins LL128 (`docs/multi-gpu.md:279-285`). At 8–64 KB on TP8 that is ≈12–20 µs per call,
    ≈1.4 ms/step: **+0.7 ms against sglang before anything else** (UNVERIFIED; check with
    `all_reduce_perf -b 8K -e 64K -f 2 -g 8` from nccl-tests, or a 91-AR kern program).
  - Its ring does 7 bf16 roundings, while sglang does one fp32 sum. Parity drifts (§5 R4).
- **P1: port sglang's one-shot push onto kern peer buffers** (`export` + `peer` arrays,
  `docs/multi-gpu.md:316-353`).
  - Lamport +0 poison with the +0 → −0 rewrite (`SP/sglang/kernels/jit/include/sgl_kernel/distributed/communicator.cuh:124-146`).
  - Push to 8 slots, poll, then `reduce_vec` in **fp32, rank order 0..7, one bf16 rounding**
    (`SP/sglang/kernels/jit/csrc/distributed/custom_all_reduce.cuh:136-209`;
    `SP/sglang/kernels/jit/include/sgl_kernel/vec.cuh:141-161`).
  - This is bit-identical to sglang by construction, and every rank gets identical bytes.
  - Start from kern's own TRT-LLM Lamport kernel (`tools/kernels-src/peer_allreduce.cu`: f32,
    NRANKS = 4, measured 5.1 µs at 4 rows on GB300 × 4) and make a bf16 / NRANKS = 8 / sm_90a
    variant with a timeout-to-`err` path.
  - Target 5–6 µs with PDL, **−0.14 ms against sglang and −0.8 ms against NCCL**.
- **P2: fuse into the consumer, not the producer.**
  - The producers are cuBLAS, DeepGEMM and Triton binaries, and a peer-store epilogue with the
    Lamport rewrite would mean recompiling each one.
  - The consumer is always the mHC boundary (fma, or post at L = 44). One kernel does:
    push own partial → poll 8 slots → rank-order sum → post + pre-GEMM partials.
  - This removes 91 launch boundaries: **−0.25 to −0.35 ms**.
  - The hard rule on peer memory stands: no multicast TMA on peer-mapped memory
    (`docs/multi-gpu.md:496-503`). Use plain `st.relaxed.sys` / `ld`.

**(b) MTP speculative decoding.** See §4.3. This is the only lever that breaks the per-step
latency floor at bs ≤ 4.

**(c) Tiny-op fusion.** Concrete list and savings at bs1, launches removed per step:

| fusion | removes | est. saving |
|---|---|---|
| quant(x) into the big_fuse epilogue: emit bf16 `x_norm` **and** e4m3 + SFA (col-major for DSA and dense, row-major for MoE) | 45 + 42 quant launches | 0.20 ms |
| q_a/kv_a norms + quant_q + kv_store → one kernel (norm reads QKV at pitch 2048, writes qa, qa8/sfa, and the kv slot) | 3 × 11 | 0.08 |
| wk ‖ gate ‖ weights_proj → one bf16 GEMM N = 288 over the same x, plus a k_norm / kpool-update / head-weight epilogue kernel | 4 × 11 | 0.10 |
| hadamard + act_quant → one kernel (per 128-row: 7 butterflies + ue8m0 quant) | 11 | 0.03 |
| topk: pad with 0 and map slots from the block table (recompiled), so the clamp0 and the slot table disappear | 11 + 1 | 0.04 |
| mqa metadata and dsa_prep once per step (fa3 prepare once per step if combine resets its semaphore, UNVERIFIED) | 10 (+10) | 0.03 (+0.03) |
| W_kc / W_vc: one grouped GEMV kernel per bmm instead of 8 extern GEMMs (`ops_dsa.py:117-123`) | 14 × 11 | 0.35 against P0 |
| MoE: router GEMM + topk + align → 1 (last-CTA-finishes pattern); w13 + silu-clamp + quant → 1 (interleave gate/up rows in blocks of 64 at load with `bind.interleave`, `types.rs:388-401`, so one N-tile holds both halves); w2 + sum-reduce → 1 (the CTA loops over a token's 9 experts, so the summation stays in topk order) | 7 × 42 | 0.70 |
| MoE: drop padding rows' experts (`Fill::Valid` = 0 → expert id −1 in align) | bucket-pad waste | up to 0.4 at bs 3/5/6/7 |
| KDA: fg_b + conv + delta + norm_gated → 1 kernel per layer (head-local: 8 heads × 4 V-tiles). qkvbfg with no splitK reduce. | 4 × 34 | 0.35 |
| lm_head argmax-first: local (max, idx) per vocab shard → 8-B allgather → pick lowest rank on ties (the same as torch's first-max) | 2 | 0.03 |

**Numerics.** Every fusion must keep the *order* of floating-point reductions of the kernels it
replaces, or accept a tolerance-based oracle for that stage. `NOTES.md` shows how the qwen38
bring-up mirrored Triton's PTX to stay bit-exact.

**(d) Persistent megakernels.**

- kern runs any cubin as one launch, but `LaunchKind::Cubin` only knows `cluster` and `pdl`
  (`exec.rs:244-264`). There is no cooperative-launch attribute.
- So a grid-wide barrier must rely on co-residency (grid ≤ 132 × occupancy) and must **not** be
  PDL-overlapped with a neighbour that holds SMs. That combination can deadlock.
- Recommendation: *block* megakernels only.
  - The mHC boundary + AR (P2/P3).
  - The MoE block. DeepGEMM ships an SM90 FP8 MegaMoE with a swiglu clamp
    (`SP/deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:30-60`), but it is built for EP
    symmetric buffers; the TP-mode adaptation is UNVERIFIED.
- No whole-model megakernel before P3 exit.

**(e) Whole-step CUDA graph.**

- Already true for both engines. Nothing in §3's design breaks capture (§3.5).
- The real win is **PDL on every edge** (`exec.rs:256-264`). Kernels that prefetch weights before
  `griddepcontrol.wait` (e.g. `tiny_gemm.cuh:85-94`; DeepGEMM TMA producers) overlap their weight
  streaming with the predecessor.
- At ~1,100 edges × ~0.7–1.5 µs that is **−0.5 to −0.8 ms**, subject to the §2.4 safety rule.

**(f) Indexer skip.**

- It is exact only when every row has ≤ 512 pools (seq ≤ 2051). The transform already takes the
  identity path then (`kpool_topk_transform.cuh:243-261`).
- Above that, full attention is a *different model*, so it is not allowed.
- Skipping the logits launch would need a per-step host decision (a graph per context class),
  and saves ~3 µs × 11 at short context. **Not worth it.** At 141k the topk radix (≈20 µs × 11)
  is the target instead. The P3 fix is a multi-CTA radix (per-CTA histograms over row slices plus
  one merge) or a streaming top-512 inside the logits kernel's epilogue.

### 4.3 MTP speculative-decoding design (P3)

**Model facts.**

- Layer 45 is a DSA + MoE decoder with `enorm`, `hnorm`, `eh_proj [4096, 8192]` bf16 (replicated,
  67 MB per rank) and `shared_head.norm`. It has **no mHC**: 1,760 tensors,
  `$GLM53_ARTIFACTS/inventory.json`.
- sglang wiring: `SP/sglang/srt/models/glm5_next_nextn.py:24-88`, and
  `SP/sglang/srt/models/deepseek_nextn.py:87-238`, where `eh_proj(cat(enorm(emb), hnorm(prev_hidden)))`
  is a ReplicatedLinear.
- Which target hidden feeds `hnorm` (post-`hc_contract`, pre- or post-final-norm; `glm5_next.py:1205-1212`
  returns the post-norm hidden) is **UNVERIFIED**. Check:
  `grep -n "hidden_states" SP/sglang/srt/speculative/eagle_worker*.py SP/sglang/srt/models/glm5_next.py`.

**State additions.**

- `kv` grows to 12 layers (12,288 B/token), `idx` to 12 × 33, and `idx_tail` to 12 lines.
- The draft layer has no recurrent state.

**Row count.** A round has T = 3S rows, up to 24 at bs8, which exceeds S_MAX = 16. It needs:

- its own op variants with M-dim 32 tensormaps (the cubins are the same);
- a tiny_gemm M = 32 entry (recompile, §2.5) or two M = 16 launches;
- activation buffers sized 32 rows.

**The `round_k2` program** (`{groups: 8, rows: 3}`, one graph per bucket, one host sync per
round):

1. **splice.** The anchor (`Fill::Token` over groups, `protocol.rs:318`, `:344-355`) becomes
   row 0. This is `kern_splice_draft`/`splice_verify` (`tools/kernels-src/spec_round.cu:1-60`).
2. **draft ×2.**
   - Run the MTP layer on 1 row per sequence, then norm → lm_head (shared) → local argmax → d1,
     then again for d2.
   - Each writes the MTP layer's kv, idx and tail at its position. The exact position offset
     convention is **UNVERIFIED**; check sglang `eagle_worker` `positions`/`out_cache_loc`.
3. **verify.** The main model runs on `[anchor, d1, d2]`, i.e. 3 rows per sequence.
   - **KDA:** the delta kernel's **no-store** variant (a re-mine with `STORE_FINAL_STATE = False`)
     runs over 3 tokens per sequence (varlen `cu_seqlens`). It computes outputs from the committed
     state. q, k, v, a, b and the conv inputs go into carries (the dflash2 `k_save`/`v_save`/
     `a_save`/`b_save` pattern). conv uses the `seqlen = 3` variant.
   - **DSA:** each row is an independent query.
     - Row length = position + 1. `dsa_prep` becomes per row, reading `positions`.
     - The DeepGEMM metadata allows ≤ 32 rows (bs8 × 3 = 24 is fine).
     - fa3 sees 3S "batches" (prepare `<1,true>` allows ≤ 31; bs8 × 3 = 24 is fine).
     - kv_store writes all 3 rows.
     - kpool_update runs a sequence's rows **in order in one CTA**. A pool closed at row j must
       see the ring entries written by rows < j.
   - **MoE:** 3S rows.
4. **accept.** Argmax per row, then `spec_count` (prefix match; `spec_round.cu`) gives
   `nacc ∈ [1, 3]`. `tokens [seqs,3]` is the `Fill::Tokens` output and `nacc [seqs]` is the
   `Fill::Count` output (`protocol.rs:536-549`).
5. **advance.**
   - The KDA delta kernel **with store** runs over the first `nacc` rows per sequence.
     `cu_seqlens` is rebuilt on the device from `nacc`; the kernel reads it from a buffer.
   - conv shifts by `nacc` (the `kern_conv_shift` pattern, `tools/kernels-src/gdn_advance.cu:1-60`).
6. **carry.** The main-model hidden of row `nacc−1` is carried as the next round's draft input.

**Rollback.**

- Paged `kv`/`idx`: free. The next round overwrites every position past `count` (`scheduler.rs:49-53`;
  the lease already holds `rows − 1` extra tokens, `scheduler.rs:36-38`).
- `idx_tail`: free with the 8-entry ring (§1.6).
- A pool closed by a rejected row leaves a garbage entry that nobody reads until the pool closes
  again for real, because the indexer reads only pools < ⌊len/4⌋.
- KDA: recompute-advance. Per-token state snapshots would write (k+1) × 17.8 MB per sequence per
  round (0.43 GB at bs8, k = 2). The advance re-reads 3 rows of inputs and 1 MB of state per
  layer and sequence: ≈0.1 ms/round.
- kern has no truncation API (K10), and this design does not need one.

**Draft length.** k = 2 at bs 1–2, and k = 1 or 2 at bs 4–8. Each extra verify row adds distinct
experts, i.e. MoE bytes; §4.4 has the numbers.

**Acceptance is UNVERIFIED.** Measure on an agentic trace with:

```bash
sglang.launch_server … --speculative-algorithm EAGLE --speculative-num-steps 2 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 3
```

and read `accept len` in the decode log. The planning figures assume α1 = 0.8 and α2|1 = 0.65,
i.e. 2.4 tokens per round at k = 2 and 1.8 at k = 1.

### 4.4 Phased plan with per-component budgets

All budgets are ms/step at 141k context. P3+MTP rows are ms per generated token.

| component | sglang model | P0 parity | P1 match | P2 (≥30% at bs1) | P3 plain |
|---|---|---|---|---|---|
| mHC + AR | 1.68 / 1.79 / 1.90 | 1.72 / 1.83 / 1.94 (Lamport AR; **+0.7 if NCCL**) | 1.40 / 1.50 / 1.60 | 0.95 / 1.02 / 1.10 | 0.70 / 0.76 / 0.83 |
| KDA attention | 0.97 / 1.00 / 1.04 | 0.97 / 1.00 / 1.04 | 0.80 / 0.83 / 0.87 | 0.62 / 0.66 / 0.72 | 0.58 / 0.62 / 0.68 |
| DSA projections and glue | 0.78 / 0.80 / 0.82 | 0.98 / 1.00 / 1.02 | 0.62 / 0.64 / 0.66 | 0.40 / 0.42 / 0.44 | 0.36 / 0.38 / 0.40 |
| DSA indexer (logits + topk) | 0.31 / 0.35 / 0.42 | 0.31 / 0.35 / 0.42 | 0.30 / 0.34 / 0.41 | 0.30 / 0.34 / 0.41 | 0.15 / 0.20 / 0.28 |
| fa3 + prepare + combine | 0.15 / 0.18 / 0.20 | 0.15 / 0.18 / 0.20 | 0.12 / 0.15 / 0.17 | 0.12 / 0.15 / 0.17 | 0.10 / 0.13 / 0.15 |
| MoE block (floor 0.46 / 1.51 / 2.80) | 1.51 / 2.31 / 3.49 | 1.51 / 2.31 / 3.49 | 1.10 / 1.85 / 3.00 | 0.76 / 1.60 / 3.00 | 0.60 / 1.50 / 2.90 |
| dense + head | 0.12 / 0.14 / 0.16 | 0.12 / 0.14 / 0.16 | 0.10 / 0.12 / 0.12 | 0.08 / 0.08 / 0.08 | 0.08 / 0.08 / 0.08 |
| **ms/step (bs1 / bs4 / bs8)** | **5.52 / 6.57 / 8.03** | **5.76 / 6.81 / 8.27** | **4.44 / 5.43 / 6.83** | **3.23 / 4.27 / 5.92** | **2.57 / 3.67 / 5.32** |
| against sglang as logged (5.68 / 6.9 / ≈8) | | +1% | −22% / −21% / −15% | **−43% / −38% / −26%** | −55% / −47% / −34% |
| **P3 + MTP, ms/token** (sglang column = step time / bs) | 5.68 / 1.73 / ≈1.0 | | | | **1.59 (k=2) / 0.70 (k=2) / 0.52 (k=2)**, i.e. 3.6× / 2.5× / 1.9× |

MTP round costs behind the last row:

| bs, k | verify cost | round time | tokens/round | ms/token |
|---|---|---|---|---|
| bs1, k = 2 | P3 plain + 0.75 ms MoE (≈25 vs 9 experts) + 0.15 advance/DSA + 2 × 0.17 draft | 3.81 ms | 2.4 | 1.59 |
| bs4, k = 2 | 12 rows ≈ 83 experts: +2.43 ms | 6.75 ms | 9.6 | 0.70 |
| bs8, k = 2 | 24 rows ≈ 141 experts: +3.9 ms | 9.9 ms | 19.2 | 0.52 |

The bs8 MoE floor is 2.8 ms of pure expert bytes. That is why bs8 lands at ≈1.9× and not 2×:
reaching 2× at bs8 needs either real routing skew (likely) or fewer bytes per expert (P3+, e.g.
the FP8 KDA projections of R10). Against the brief's 12 / 25 ms the ratios are 7.5× and 6×.

**Phase exit criteria.**

- **P0: correctness parity.**
  - Every §0.2 defect fixed.
  - Buffer-exact at every call cut against the sglang probe for bs1 and context ≤ 2051. topk is
    the identity there and deterministic.
  - Tolerance-based beyond that (§5 R3).
  - Greedy agreement ≥ 99.5% over 1k tokens at 141k context and bs 1/4/8.
  - Lamport AR in place, or NCCL with a documented drift.
- **P1: ≤ sglang at every bs.** PDL where safe, per-step hoists, the fma ffn→attn boundary,
  grouped GEMV, argmax-first, and the recompiled topk.
- **P2: ≤ 0.7 × sglang at bs1 (≤ 4.0 ms).** The MoE block (3 launches), the mHC boundary + AR
  fusion, DSA glue fusion and KDA fusion. Measured, not modeled.
- **P3: ≥ 2× tokens/s at bs 1..4.** The MTP `round` program, the multi-CTA topk and the
  single-kernel mHC boundary. It depends on acceptance.

### 4.5 Top-5 optimizations (ranked by expected gain / risk)

| rank | optimization | gain (bs1, beyond P0) | risk | why this rank |
|---|---|---|---|---|
| 1 | **PDL on every capable edge**, plus `griddepcontrol` in every handwritten kernel, gated by a SASS scan in gen.py | −0.5..−0.8 ms (all bs) | low | Pure manifest flag plus a small kernel edit. The mined kernels already contain the code (§2.4). |
| 2 | **MoE block → 3 launches**: the router/topk/align megakernel; w13 + silu + quant with interleaved gate/up and a decode-shaped N split (BLOCK_N = 32 → 144 CTAs, instead of 36 at bs1); w2 + sum in topk order; dropping padding rows' experts | −0.75 ms (bs1), −0.5 ms (bs8) | low-med | Triton sources exist (re-mine, §2.5). Numerics keep per-(token, expert) fp32 accumulation. |
| 3 | **Lamport one-shot AR (sglang numerics) → fused into the mHC boundary kernel** | −0.7 ms (all bs); −1.4 ms against an NCCL P0 | medium | Needs peer buffers, a timeout/err path and a handwritten fused kernel. It also fixes parity drift. |
| 4 | **DSA glue**: grouped absorb GEMV, fused norms/quant/kv_store, a merged x-side GEMM, per-step hoists, the recompiled topk | −0.6 ms against P0 (−0.4 against sglang) | low | Mostly handwritten glue with simple math. |
| 5 | **MTP k = 2 speculative rounds** | ×2.4 tokens per round at 1.2–1.5× round cost: **−1.6 to −2.5 ms/token equivalent at bs1** | high | Largest absolute gain. It depends on unmeasured acceptance and on three new kernel variants (no-store delta, seqlen-3 conv, per-row DSA prep). Do it last, on top of P2. |

Runners-up:

- mHC boundary as a single kernel (last-CTA Sinkhorn): −0.3 ms.
- Multi-CTA topk: −0.15 ms.
- hc_fn stored as bf16: the checkpoint is bf16, and bf16×bf16 products are exact in fp32, so it
  halves 141 MB/step at no numeric cost *if* the kernel accumulates in fp32. Worth −0.05 ms.

### 4.6 Measurement plan (profile first, then build)

1. **Clean sglang baseline** (after the probe server is stopped; nothing here allocates GPU memory
   from this session): steady-state per-bucket step time at 141k.

   ```bash
   python -m sglang.bench_one_batch_server --base-url http://127.0.0.1:30000 \
     --batch-size 1 2 4 8 --input-len 141000 --output-len 256
   ```

   Alternatively, read only `Decode batch` lines whose previous line has the same `#running-req`.
   Flag spelling is UNVERIFIED for this fork: check `--help`.
2. **Per-kernel sglang trace:**

   ```bash
   curl -X POST http://127.0.0.1:30000/start_profile -H 'Content-Type: application/json' \
     -d '{"output_dir":"/tmp/sgl_prof","num_steps":10,"activities":["GPU"]}'
   ```

   Or `nsys profile --cuda-graph-trace=node -t cuda` around the server. This validates every row
   of §4.1.
3. **Microbenchmarks of the unknowns** (standalone harnesses in the `NOTES.md` style):
   - topk radix at 35k/50k pools;
   - Lamport AR bf16 on 8 × H100 at 8/32/64 KB, against NCCL LL128;
   - tiny_gemm M = 16 against M = 1 (bit-exactness);
   - fused_moe w13 at BLOCK_N 32 against 128 at bs1.
4. **kern:** `kern bench glm53 --manifest <m> --program-only --workload <141k>` with the ablation
   method of `NOTES.md` (replace one op with a 1-block no-op to price it in-graph).

---

## 5. Risks not yet named

**Baseline and parity**

| # | Risk | Impact | Mitigation / check |
|---|---|---|---|
| R1 | **Baseline mismatch.** The log gives 5.7 ms (bs1) and 6.9 ms (bs4); the brief gives 12 ms and 25 ms. | Wrong targets. | Re-measure (§4.6.1). All phase targets here use the log. |
| R2 | **Protocol rejects the manifest on day one** (D10: rows and groups on one var; page table transposed). | P0 blocked. | `kern verify <manifest>` prints the protocol (`docs/spec-decode.md`, last block). Fix per §3.1. |
| R3 | **Bitwise parity is impossible beyond 2051 tokens, even sglang against itself.** The topk slot order comes from `atomicAdd` (`kpool_topk_transform.cuh:118`, `:131`), so fa3's online-softmax order varies. sglang also changes mHC split counts by T (`mhc.py:1710-1715`). | A flaky oracle. | Bitwise cuts only at bs1 and ctx ≤ 2051. Beyond: per-stage relative-error bounds plus greedy agreement. Calibrate the tolerance by running sglang twice on the same 141k prompt and comparing. Optional kern determinism: sort the 512 selected pools inside the recompiled topk. |
| R4 | **NCCL ring reductions differ from sglang's one-shot.** NCCL does 7 bf16 roundings in ring order; sglang does one fp32 sum in rank order, identical on every rank. | ≈1-ulp drift per AR × 91 per step. Greedy divergence over long generations. | Lamport AR with sglang numerics (§4.2a). |
| R5 | **−0.0 from the Lamport rewrite** (+0 is the poison; `communicator.cuh:131-137`). sglang's allreduce can emit −0 where a replicated embedding or NCCL gives +0. | Bitwise compares flag it; the math is unaffected. | Compare with `-0 == +0` semantics, or reproduce the rewrite. |
| R6 | **argmax semantics.** torch returns the first max and treats NaN as max. `glm53_argmax_f32` ignores NaN (`v > best` is false) and starts from −FLT_MAX. The split argmax must break ties by lowest rank first. | A mismatch only on NaN or exact ties. | Document it. For ties, local first-index then lowest rank equals torch. Assert no NaN in debug builds. |
| R7 | **Handwritten ports of Triton math drift by 1 ulp.** kpool update (§0.2): the softmax uses `ex2.approx` and `div.full` in Triton. The ue8m0 path is exact (libdevice `log2f`, `cvt.rpi`, `ex2.approx` of an integer, a power-of-2 divide; the `max(absmax, 1e-4)` constant `0f38D1B717` matches). | A different fp8 code, so a different pooled key, so a topk flip on near-ties. | Mirror the PTX (`NOTES.md`). Check: `grep -nE "ex2|div\.|lg2" TC/W2FILQN3*/…ptx`. |
| R12 | **cuBLASLt algorithm choice** for the KDA and indexer GEMMs and lm_head. kern links CUDA-13 cuBLASLt with its own workspace; sglang uses torch's bundled cuBLASLt with torch's workspace. The heuristic may pick another split-K than `nvjet_…splitK` (recipe sites 6-7). | Numerics and speed differ. | Log `cublasLtMatmulAlgoGetHeuristic` names per shape and compare with recipe `nvjet_*` names. If needed, pin the algo via a manifest extern variant (runtime change). |
| R18 | **The fixed fma split 8** (kern) against sglang's split 4 at T ≥ 8 changes the Sinkhorn inputs in the last bits. | bs ≥ 8 is not bit-comparable with sglang, even at ctx ≤ 2051. | This is the accepted batch-invariance tradeoff. The oracle at bs8 uses tolerance. |

**Arithmetic and sizing**

| # | Risk | Impact | Mitigation / check |
|---|---|---|---|
| R8 | **fp8 GEMM with K = 1536** (q_b, dense down). It is safe: K % 128 = 0, SFA `[16,12]` f32 with a 64-B stride, and a TMA A stride of 1536 B (a multiple of 16). `DG_TRAP_ONLY_DEVICE_ASSERT(shape_n % 8 == 0)` holds for every N (`sm90_fp8_gemm_1d2d.cuh:231`). | none | none |
| R9 | **Integer widths at long context.** Byte offsets `slot × 11264` overflow i32 beyond 190k slots. `glm53_kv_store` uses `long long` (`tools/glm53/kernels/glm53_dsa.cu:82`), and fa3 strides are i64. The kpool `page × 92928` fits i32 up to 23k pages, but the kpool kernel already uses `long long page`. Keep every new kernel's address math in i64. | A silent wraparound at 3M-token pools. | Review each handwritten kernel. |
| R16 | **Page-table width.** W sized for `MAX_POS = 1M` means 4096 entries: 128 KB of H2D per step at bs8 on the compute stream (K13), and a host domain check over 32k entries. | +10–30 µs/step. | W = 1024 (262k tokens). Raise it only when needed. |
| R21 | **Buffer sizing at var max** (`load.rs:156-168`). If decode activations are declared over `tokens` and prefill raises `tokens.max` to 8192, every decode buffer becomes 8192 rows (e.g. `c2 [9·tokens,4096]` = 604 MB). | Wasted GBs. | Declare decode activations with **constant** S_MAX rows. Only fills and tables go over vars. Prefill gets its own buffer set. |
| R10 | **The KDA projections stay bf16**: 1.23 GB/step, 36% of the bs1 byte floor. FP8 block-128 would halve them (≈−0.2 ms at bs1) but changes numerics. | A missed gain only. | P3+ option behind an accuracy gate. Not in the parity phases. |
| R11 | **The embedding is replicated** (1.27 GB/rank) instead of sglang's vocab shard + allreduce. | Memory only. Numerically identical (x + 0 = x) except −0 (R5). | Keep it. Revisit only if the state budget binds. |
| R23 | **kNumSMs = 132 is baked** into DeepGEMM 1d2d, mqa metadata and logits, and the fa3 `hw_info`. | Breaks under MPS or green contexts, or on H100 PCIe (114 SMs). | Pin the deployment to H100 SXM. Assert the SM count at load. |

**Operations and new features**

| # | Risk | Impact | Mitigation / check |
|---|---|---|---|
| R13 | **Graph capture cost.** ~1,000–1,400 nodes × 4 buckets (+ rounds), and capture is serialized across ranks (`exec.rs:71-76`). | The first request of each bucket stalls. | Warm-capture all buckets in the load path, driven by the scheduler before serving. |
| R14 | **A dead rank under the Lamport AR** spins forever unless the kernel times out into an `err` buffer. There is no NCCL-style abort. | A hung tray. | Timeout plus `Fill::Error` output (`types.rs:198-199`), as in `peer_allreduce.cu`'s sticky `err`. |
| R15 | **Remaps at admission** (`Denied::Remapping` retry). Fresh chunks are zeroed on the compute stream (`lease.rs:212-254`). | Admission-time latency spikes, not per step. | Size the budget with `capacity.tokens` so that steady state never remaps. |
| R19 | **Evidence gap for long prefill.** No flashmla_sparse, no DeepGEMM `fp8_mqa_logits`, no prefill topk in the capture (§3.6). | The prefill schedule is unknown. | Capture a ≥ 16k prompt. |
| R20 | **MTP details UNVERIFIED**: which hidden feeds `hnorm`, the position offsets of the draft KV, the acceptance rate. | The P3 gain is uncertain. | §4.3 check commands. |
| R17 | **A prefix-hit restore copies** the KDA slot (18.5 MB) and the partial page (3 MB) on the compute stream (`pages.rs:706-740`; K8). | ≈8 µs per admission. | None needed. |
| R22 | **topk pads map to slot 0**, which in kern-serve is the pad lease's first slot (finite junk KV). This is only correct while fa3 masks by `seqused_k`. | none | Assert the pad lease is taken first (it is: `tray.rs:551-554`). |
| R24 | **Enabling MTP changes the state layout** (kv 11 → 12 layers, idx 363 → 396 B). Checkpoints and snapshots do not survive a manifest change. | A migration later. | Decide before P1. Recommendation: reserve the 12th layer from day one (+9% KV bytes, 1.7 GB at bs8 × 200k). |
| R25 | **Padding rows run real kernels** (pad lease page/slot, `tray.rs:421-422`, `:551-554`). They pull up to 9 extra experts each (MoE bytes), and their kpool/tail writes go to the pad lease. | Bandwidth waste at non-power-of-2 batches. | `Fill::Valid` → drop their experts in align (§4.2c). The pad's own writes are harmless. |

---

## 6. Decision summary (one page)

Each line reads DECISION, then BECAUSE (the `→`).

**States and tables**

1. **Two paged states.** `kv` (11,264 B/token, token-interleaved; layer i at call offset i·1024)
   and `idx` (363 B/token; the 256-token page holds 11 × 8,448 B layer blocks)
   → one lease covers both (K2), rounding waste is ≤ 2 chunks instead of ≤ 22, fa3 stays
   slot-linear with `stride_V = 5632` elements, and kv_store is already handwritten with
   `slot_pitch` and `layer_off`.
2. **Page unit = 256 tokens; `block_table` is `[seqs, 1024]` with stride 256; nothing else
   indexes a paged state with a stride other than 1 or 256**
   → each pool lands in the page holding its own tokens. That makes kern-serve's automatic
   retire/restore of 141k agentic prefixes safe (K9, K11). It also removes the 3/4 kpool waste
   and the `::4` pooled table.
3. **Three per-seq states: `kda_conv` (34 lines × 18,432 B), `kda_ssm` (34 × 524,288 B),
   `idx_tail` (11 × 4,096 B, ring of 8). No folding.**
   → The mined conv kernel bakes a line stride of 9,216 elements. A mixed-stride fold needs lcm
   padding. Ring 8 makes speculative rollback free.
4. **The driver fills tokens, positions, slots, seq_lens, cu_seqlens, valid, the block table and
   the line tables. The device (`dsa_prep`, once per step) derives pool lengths, clamped context,
   `dsa_lens` and the mqa schedule.**
   → The protocol fills only these roles (`tray.rs:801-866`). Everything else must be computed
   on the device to stay graph-safe.
5. **Reserve the MTP layer's slot in kv, idx and idx_tail from P1 on**
   → a later layout change invalidates snapshots and checkpoints (R24).

**Kernels**

6. **Every mined kernel is runtime-generic in S, or it is re-mined or replaced:**
   - router → module_411;
   - tiny_gemm → the M = 16 entry;
   - align → handwritten;
   - conv → re-mined with `num_cache_lines = 2^31−1`;
   - norm_gated → module_227;
   - kpool_update → handwritten;
   - hc_repack → deleted (prenorm and big_fuse64 run at S_MAX)

   → the §2 audit found these seven silent S- or length-specializations.
7. **Re-mine by recompiling the venv sources offline (Triton, sglang JIT, DeepGEMM, TileLang) with
   the captured constexprs except the one being generalized. Pin by sha256.**
   → exact numerics, no GPU needed, and it is the same identity model as mining.
8. **Keep DeepGEMM D and A tensormaps at M = S_MAX = 16. Treat rows [M, 16) as junk. Never alias
   them.**
   → the 1d2d and prenorm epilogues TMA-store whole tiles clipped only by the tensormap
   (`sm90_fp8_gemm_1d2d.cuh:430-441`).
9. **Fix the TileLang arg order** (alphabetical; D7). **Make prepare, fa3 and combine one op**
   (D8). **Give the q_a/kv_a norm a pitch** (D9).
   → verified ABI (recipe sites 5 and 14); scratch is op-private (K14).
10. **Set `pdl: true` only on entries whose SASS has `ACQBULK`. Add griddepcontrol to every
    handwritten kernel.**
    → otherwise a kernel reads its producer's output early (§2.4).
11. **Recompile `kpool_topk_transform` to map slots from the block table and pad with 0.**
    → this removes the token → slot table build (≈10 µs/step at 141k) and the clamp0 launch.

**Programs**

12. **One `decode` program over vars `tokens` (rows) and `seqs` (groups), graph = true, one graph
    per kern-serve bucket {1, 2, 4, 8}.**
    → the protocol needs distinct rows and groups vars, graphs are keyed by var values, and
    every kernel is S-generic after decision 6.
13. **Declare decode activation buffers with constant S_MAX rows. Prefill gets its own buffer
    set.**
    → buffers are sized at var max (R21).
14. **P0 follows sglang's mHC decomposition exactly** (fma only at attn→ffn; post + prenorm64 +
    big_fuse64 at ffn→attn). **P1 switches ffn→attn to fma + big_fuse8.**
    → bitwise cuts first. The switch saves 44 launches but changes summation order only.
15. **Prefill is a separate eager chunk program (T ≤ 8192):**
    - unfused mHC at every boundary, with 64-split prenorm and big_fuse for all T;
    - chunked KDA kernels;
    - sparse-prefill DSA kernels, to be captured from a ≥ 16k-token prompt first.

    → sglang switches decomposition above T = 16/32, decode kernels cap the batch at 31/32 rows,
    and the capture has no long sparse prefill.
16. **Steady-state agentic suffixes ride decode steps as spans (P2).**
    → avoids stalling the batch for 100–3k-token tool outputs on a restored 141k prefix.

**Communication**

17. **The allreduce is a Lamport one-shot on kern peer buffers: bf16 in, fp32 rank-order sum, one
    rounding, timeout → `err`. No NCCL in decode.**
    → bit-compatible with sglang and ≈2× faster than NCCL LL128 at 8–64 KB. The generator's NCCL
    choice costs ≈+0.7 ms/step and adds parity drift.
18. **P2 fuses the AR into the consumer (the mHC boundary kernel), not the GEMM producer.**
    → producers are binary cuBLAS, DeepGEMM and Triton kernels. Consumer fusion removes 91
    boundaries.

**Performance**

19. **Optimize launches before bytes.** Order: PDL → MoE block (3 launches, BLOCK_N = 32 at decode,
    padding-row experts dropped) → mHC + AR fusion → DSA glue (grouped absorb GEMV, fused
    norms/quant/kv_store, merged x-side GEMM, per-step hoists) → KDA fusion.
    → the bs1 step is 4.5× above the bandwidth floor. Launch latency is the dominant cost (§4.1).
20. **MTP k = 2 speculative `round` program in P3:**
    - per-row DSA queries;
    - KDA no-store verify + advance pass (dflash2 pattern);
    - kpool tail ring 8;
    - count/tokens outputs.

    → the only lever past the latency floor at bs ≤ 4 (≈3.6× at bs1, 2.5× at bs4, assuming an
    acceptance to be measured first).
21. **No indexer skip and no whole-model megakernel. Build block megakernels (MoE, mHC + AR) and a
    multi-CTA topk.**
    → the skip is exact only at ≤ 2051 tokens. kern has no cooperative launch, so grid barriers
    under PDL can deadlock.
22. **Keep the bf16 KDA projections and f32 hc_fn for parity. FP8 KDA and bf16 hc_fn are P3+
    options behind an accuracy gate.**
    → these are numerics changes, not transcription.

**Oracle and targets**

23. **Oracle:** bitwise at bs1 and ctx ≤ 2051. Tolerance plus greedy agreement beyond that, and at
    bs ≥ 8.
    → sglang itself is non-deterministic past 2051 tokens (topk atomics) and T-dependent in mHC.
24. **Targets** are stated against the logged sglang step (5.7 / 6.9 / ≈8 ms at bs 1 / 4 / 8):
    P1 ≤ 1.0×, P2 ≤ 0.7× at bs1 (≈3.2–4.0 ms), P3 ≥ 2× tokens/s at bs 1–4.
    → the 12 / 25 ms in the brief are not reproduced by the server log.

---

## Appendix A — reproduce the audit (read-only, no GPU)

```bash
# Triton: cache-entry signature (arg names, d16 flags) mapped to dump modules
python3 - <<'EOF'
import os,hashlib,re,glob
TC="$HOME/.cache/sglang/triton"; D="dumped-kernels-glm53-sglang"
dump={hashlib.sha256(open(f,'rb').read()).hexdigest():os.path.basename(f) for f in glob.glob(D+"/module_*.cubin")}
for d in sorted(os.listdir(TC)):
    c=glob.glob(f"{TC}/{d}/*.cubin")
    if not c: continue
    mod=dump.get(hashlib.sha256(open(c[0],'rb').read()).hexdigest())
    if not mod: continue
    t=open(glob.glob(f"{TC}/{d}/*.ttir")[0]).read()
    sig=re.search(r"tt\.func public @(\w+)\((.*?)\)\s*attributes",t,re.S)
    args=[n+("/16" if "divisibility = 16" in (a or "") else "") for n,_,a in re.findall(r"%(\w+):\s*([^\s{]+)\s*(\{[^}]*\})?",sig.group(2))]
    print(mod, sig.group(1), d[:8], ", ".join(args))
EOF
# baked constants of one variant (e.g. conv num_cache_lines)
grep -o "arith.constant [0-9-]* : i[0-9]*" $HOME/.cache/sglang/triton/LAWVJ2KQ*/_causal_conv1d_update_kernel.ttir | sort -u
# TileLang real ABI order
grep -h -m1 "__global__" $HOME/.cache/sglang/tilelang/*/linux-x86_64/kernels/*/device_kernel.cu
# PDL safety per entry
cuobjdump -sass DUMP/module_N.cubin | awk '/Function :/{f=$NF} /ACQBULK/{a[f]++} /PREEXIT/{p[f]++} END{for(k in a)print k, "ACQBULK="a[k], "PREEXIT="p[k]}'   # entries not listed never wait: pdl must be false
```
