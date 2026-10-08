"""TP decode/verify. Non-interleaved weights; 128-wide K groups as module_335/337.
Retain bf16 cuts after W13, activation and each weighted W2 expert.

v3 (fused4): universal 32-pair align tiles (cap=32). At rows<=16 counts<=16,
so tile counts are identical to the old 16-pair scheme; at rows 17..32 each
expert still gets exactly one tile, so the W13/W2 shards are read once per
touched expert (no 2x re-read). as/hs are stored transposed (kb-major) so the
per-kb scale loads in the main loops coalesce instead of gathering.
v4: W13 maps 8 CTAs per tile (64 output cols each, wgmma n64) instead of 16;
grid is tiles*8. Per-element math is unchanged (same kb-ascending dot order,
same per-128-group scales), so c1/h8/hs stay bitwise identical; A-tile L2
traffic per expert halves. Launch config otherwise unchanged (4 warps,
3 stages). Epilogue ticket threshold is 8.
v5: fine-grained W13->W2 overlap. W13 calls gdc_launch_dependents at entry so
the pdl-marked W2 grid launches while W13 runs. The W13 epilogue CTA bumps
Counts[tile] a second time (8 tickets + 1 = 9) after fencing its H8/HS
stores; each active W2 CTA spins on Counts[tile] >= 9 (volatile read +
membar), then reads H8/HS with .cg (no kernel boundary invalidates L1 during
overlap). Deadlock-free: the W13 grid is queued before W2 and the W2 grid
launches only after every W13 CTA has started, so producers always hold or
get SMs first. Without a pdl launch attribute the spin is a no-op (counts
already 9 at W2 start), so behavior falls back to the serial version.
"""
import triton
import triton.language as tl

@triton.jit
def fast_div(a, b):
    return tl.inline_asm_elementwise("div.approx.ftz.f32 $0, $1, $2;", "=f,f,f",
                                   [a, b], dtype=tl.float32, is_pure=True, pack=1)

@triton.jit
def glm53_moe_v2_w13(A, W, AS, WS, Sorted, Experts, NPost, Counts, C1, H8, HS, Rows):
    tl.extra.cuda.gdc_wait()
    # Release the pdl W2 grid as soon as every W13 CTA has started; W2 CTAs
    # spin on per-tile ready flags instead of a whole-grid wait.
    tl.extra.cuda.gdc_launch_dependents()
    tile = tl.program_id(0) // 8
    nt = tl.program_id(0) % 8
    if tile * 32 < tl.load(NPost):
        m = tl.arange(0, 64)
        pairs = tl.load(Sorted + tile * 32 + m, mask=m < 32, other=Rows * 9)
        mask = pairs < Rows * 9
        expert = tl.load(Experts + tile).to(tl.int64)
        n = nt * 64 + tl.arange(0, 64)
        k = tl.arange(0, 128)
        acc = tl.full((64, 64), 0, tl.float32)
        for kb in range(32):
            aa = tl.load(A + (pairs[:, None] // 9) * 4096 + kb * 128 + k[None, :],
                         mask=mask[:, None], other=0.0)
            ww = tl.load(W + expert * 512 * 4096 + n[None, :] * 4096 + kb * 128 + k[:, None])
            # as is [32 kb][32 rows] transposed: one 128B segment per kb step.
            sa = tl.load(AS + kb * 32 + pairs // 9, mask=mask, other=0.0)
            sw = tl.load(WS + expert * 128 + (nt // 2) * 32 + kb)
            acc += tl.dot(aa, ww) * (sa[:, None] * sw)
        tl.store(C1 + pairs[:, None] * 512 + n[None, :], acc.to(tl.bfloat16), mask=mask[:, None])
        # Every lane fences its writes before the scalar acq_rel ticket.
        tl.inline_asm_elementwise("membar.gl; mov.u32 $0, $1;", "=r,r",
                                 [tl.arange(0, 128)], dtype=tl.int32, is_pure=False, pack=1)
        tl.debug_barrier()
        old = tl.atomic_add(Counts + tile, 1, sem="acq_rel", scope="gpu")
        if old == 7:
            em = tl.arange(0, 32)
            epairs = tl.load(Sorted + tile * 32 + em)
            emask = epairs < Rows * 9
            # .cg avoids a stale L1 line from a previous decode/layer.
            for group in range(2):
                h = group * 128 + tl.arange(0, 128)
                gate = tl.load(C1 + epairs[:, None] * 512 + h[None, :], mask=emask[:, None], other=0, cache_modifier=".cg").to(tl.float32)
                up = tl.load(C1 + epairs[:, None] * 512 + 256 + h[None, :], mask=emask[:, None], other=0, cache_modifier=".cg").to(tl.float32)
                gate = tl.minimum(gate, 10.)
                up = tl.minimum(tl.maximum(up, -10.), 10.)
                exp = tl.inline_asm_elementwise(
                    "{ mul.ftz.f32 $0, $1, 0fBFB8AA3B; ex2.approx.ftz.f32 $0, $0; }",
                    "=f,f", [gate], dtype=tl.float32, is_pure=True, pack=1)
                hidden = (fast_div(gate, 1. + exp) * up).to(tl.bfloat16).to(tl.float32)
                amax = tl.maximum(tl.max(tl.abs(hidden), 1), 1.e-10)
                sf = amax * (1. / 448.)
                q = fast_div(448., amax)
                fp8 = (hidden * q[:, None]).to(tl.float8e4nv)
                tl.store(H8 + epairs[:, None] * 256 + h[None, :], fp8, mask=emask[:, None])
                # hs is [2 groups][288 pairs] transposed: coalesced in w2.
                tl.store(HS + group * 288 + epairs, sf, mask=emask)
            # Publish this tile: fence all lanes' H8/HS stores, sync the CTA,
            # then one release bump (8 tickets + 1 = 9) for the W2 spin.
            tl.inline_asm_elementwise("membar.gl; mov.u32 $0, $1;", "=r,r",
                                     [tl.arange(0, 128)], dtype=tl.int32, is_pure=False, pack=1)
            tl.debug_barrier()
            tl.atomic_add(Counts + tile, 1, sem="release", scope="gpu")

@triton.jit
def glm53_moe_v2_w2(H8, W, HS, WS, Weights, Sorted, Experts, NPost, Counts, C2, Valid, Out, Ready, Rows):
    # No gdc_wait: under a pdl launch this grid starts while W13 runs. Each
    # active CTA waits only for its own tile via the ready count (8 W13
    # tickets + 1 epilogue release = 9). Serial fallback: W13 has completed,
    # the count is already 9, and the spin exits immediately.
    pid = tl.program_id(0)
    # Each invalid output row has exactly one zeroing CTA. No expert load.
    if pid < Rows:
        if tl.load(Valid + pid) == 0:
            z = tl.arange(0, 4096)
            tl.store(Out + pid * 4096 + z, tl.full((4096,), 0, tl.bfloat16))
    tile = pid // 32
    nt = pid % 32
    if tile * 32 < tl.load(NPost):
        m = tl.arange(0, 64)
        pairs = tl.load(Sorted + tile * 32 + m, mask=m < 32, other=Rows * 9)
        mask = pairs < Rows * 9
        expert = tl.load(Experts + tile).to(tl.int64)
        n = nt * 128 + tl.arange(0, 128)
        k = tl.arange(0, 128)
        # Weights may be read before the wait (pdl contract): stream this
        # CTA's W shard and scales while W13 finishes the tile.
        ww_a = tl.load(W + expert * 4096 * 256 + n[None, :] * 256 + 0 * 128 + k[:, None])
        ww_b = tl.load(W + expert * 4096 * 256 + n[None, :] * 256 + 1 * 128 + k[:, None])
        sw_a = tl.load(WS + expert * 64 + nt * 2 + 0)
        sw_b = tl.load(WS + expert * 64 + nt * 2 + 1)
        ready = tl.load(Ready + tile, volatile=True)
        while ready < 9:
            ready = tl.load(Ready + tile, volatile=True)
        # Acquire fence before the overlapped H8/HS reads; .cg on the loads
        # skips any stale L1 line from a previous decode/layer.
        tl.inline_asm_elementwise("membar.gl; mov.u32 $0, $1;", "=r,r",
                                 [tl.arange(0, 128)], dtype=tl.int32, is_pure=False, pack=1)
        acc = tl.full((64, 128), 0, tl.float32)
        aa = tl.load(H8 + pairs[:, None] * 256 + 0 * 128 + k[None, :],
                     mask=mask[:, None], other=0.0, cache_modifier=".cg")
        sa = tl.load(HS + 0 * 288 + pairs, mask=mask, other=0.0, cache_modifier=".cg")
        acc += tl.dot(aa, ww_a) * (sa[:, None] * sw_a)
        aa = tl.load(H8 + pairs[:, None] * 256 + 1 * 128 + k[None, :],
                     mask=mask[:, None], other=0.0, cache_modifier=".cg")
        sa = tl.load(HS + 1 * 288 + pairs, mask=mask, other=0.0, cache_modifier=".cg")
        acc += tl.dot(aa, ww_b) * (sa[:, None] * sw_b)
        weight = tl.load(Weights + pairs, mask=mask, other=0.0)
        partial = (acc * weight[:, None]).to(tl.bfloat16)
        tl.store(C2 + pairs[:, None] * 4096 + n[None, :], partial, mask=mask[:, None])
        tl.inline_asm_elementwise("membar.gl; mov.u32 $0, $1;", "=r,r",
                                 [tl.arange(0, 128)], dtype=tl.int32, is_pure=False, pack=1)
        tl.debug_barrier()
        tickets = tl.atomic_add(Counts + (pairs // 9) * 32 + nt, 1,
                                mask=mask, sem="acq_rel", scope="gpu")
        last = mask & (tickets == 8)
        if tl.sum(last.to(tl.int32), 0) > 0:
            total = tl.full((64, 128), 0, tl.float32)
            for slot in range(9):
                part = tl.load(C2 + ((pairs // 9)[:, None] * 9 + slot) * 4096 + n[None, :],
                               mask=last[:, None], other=0.0, cache_modifier=".cg").to(tl.float32)
                total += part
            tl.store(Out + (pairs // 9)[:, None] * 4096 + n[None, :],
                     (total * 2.5).to(tl.bfloat16), mask=last[:, None])
