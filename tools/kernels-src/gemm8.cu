// Streaming bf16 GEMM for M <= 8 token rows (bs=1 decode and the 8-row
// draft / verify blocks of DFlash2), Blackwell sm_103a.  At these shapes the
// weight matrix is the only real traffic: one persistent CTA per SM pulls its
// weight rows through a smem ring with cp.async.bulk (TMA 1D, plain pointers,
// 40 KB stages of 8 rows) and 8 compute warps multiply each stage with
// mma.m16n8k16 via ldmatrix.x4 (bf16 in, f32 accumulate; the 8 weight rows
// are the n8 side, the <= 8 tokens fill half of m16).  x is staged once into
// a dedicated smem area (<= 40960 elements); when M*K exceeds it -- or N is
// too small to balance the grid -- K is cut into ranges, tasks are ordered
// range-major so a CTA reloads x only on a range switch, and the f32 range
// partials go to impl scratch with no atomics: the tail that already reads
// every output element sums them there.  Smem rows are padded by 16 B so the
// ldmatrix reads are bank-conflict-free (measured 3.5x slower without).
//
// Tasks: whole 8-row blocks, a multiple of the grid so every CTA streams the
// same number, plus the leftover rows spread over the grid in tasks one
// granule apart -- scheduled first, where their latency-bound stages hide
// under the blocks behind them.  Every task walks the same K stages, so the
// per-row accumulation order does not depend on the geometry.  The ring/x
// split of the shared area is per launch, so bs=1 gets 5-6 stages in flight.
//
// The tail runs after a single end-of-CTA fence + task-count atomic.  With
// grid <= %nsmid (one ~200 KB-smem CTA per SM, the launch geometry of every
// call site) all CTAs are co-resident, so CTAs 0..M-1 spin on the counter and
// each finishes one token row in parallel; on any other geometry the CTA that
// finished last does all rows.  Counters are back to zero at kernel end.
// The tail is pure latency: the norm weight is fetched under the spin and
// each global round trip of a row is one batch of loads.
//
// Programmatic dependent launch (manifest step `pdl`): the ring is filled
// with the first task's weights before griddepcontrol.wait, everything that
// reads an upstream product or writes comes after it, and the CTA signals
// launch_dependents once its stream is issued, so the next launch's ramp
// overlaps this launch's drain and tail.
//
// Entry points differ in what is fused around the GEMM (rounding chains
// reproduce the vLLM / ATen kernels they replace):
//
//   kern_gemm8_bf16               y[M,N] = x W^T
//   kern_gemm8_gateup_silu_bf16   act = silu(gate) * up over the fused [2F,K]
//                                 gate_up weight (vLLM act_and_mul: bf16(gate),
//                                 bf16(up), bf16(silu), bf16(product))
//   kern_gemm8_dual_bf16          y1 = x W1^T, y2 = x W2^T (one weight stream)
//   kern_gemm8_add_norm_bf16      y = x W^T, then per token row the Gemma
//                                 fused_add_rms_norm (kern_gemma_rms_norm.cu
//                                 arithmetic: z = f32(bf16(y)) + f32(res),
//                                 res = bf16(z), ATen-order mean of z^2,
//                                 out = bf16((z * rsqrt(var + eps)) * w1))
//   kern_gemm8_sgate_add_norm_bf16  the add_norm chain with x computed on
//                                 the fly as attn * sigmoid(gate) (the
//                                 sigmoid_mul.cu chain: g = bf16(sigmoid
//                                 f32), x = bf16(f32(attn) * g)) -- o_proj
//                                 of the gated-attention layers
//
//   nvcc -cubin -arch=sm_103a -o kernels-qwen38/gemm8.cubin tools/kernels-src/gemm8.cu
#include <cuda_bf16.h>
#include <cstdint>

namespace {
constexpr int MMAX = 8;                  // token rows per launch
constexpr int R = 8;                     // weight rows per task (one n8 tile)
constexpr int SEG_MAX = 2560;            // K elements per weight row per stage
constexpr int PAD = 8;                   // 16 B row padding: bank shift 4
constexpr int X_ELEMS = 40960;           // x area capacity (M * kseg elements)
constexpr int X_SLOT = X_ELEMS + MMAX * PAD;
constexpr int RING_ELEMS = 69888;        // ring minimum: 3 stages of 8 rows at SEG_MAX (x at capacity)
constexpr int BUF_ELEMS = RING_ELEMS + X_SLOT;  // ring + x share one area, split per launch by M * kseg
constexpr int STAGES_MAX = 6;
constexpr int NCW = 8, CTHREADS = NCW * 32, NTHREADS = CTHREADS + 32;
constexpr int BAR_C = 1;                 // named barrier of the compute warps
constexpr int NT_ATEN = 512;             // ATen reduce block (emulated)

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n.reg .pred p;\nWAIT_%=:\nmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n@!p bra WAIT_%=;\n}" ::"r"(bar),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void fence_barrier_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
__device__ __forceinline__ void bulk_copy_1d(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst), "l"(src),
               "r"(bytes), "r"(bar)
               : "memory");
}
__device__ __forceinline__ void named_bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void mma_bf16_16816(float& c0, float& c1, float& c2, float& c3, uint32_t a0, uint32_t a1, uint32_t a2,
                                               uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint32_t nsmid() {
  uint32_t n;
  asm("mov.u32 %0, %%nsmid;" : "=r"(n));
  return n;
}
__device__ __forceinline__ float bf16r(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
// Programmatic dependent launch: the caller may launch this kernel with
// CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION.  Every global read of
// an upstream product and every global write sit behind pdl_wait(); only the
// weight stream (this launch's own inputs) runs ahead of it.  Without the
// attribute both instructions are no-ops.
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

struct Smem {
  alignas(128) __nv_bfloat16 buf[BUF_ELEMS];  // [ring: stage s at s*8*(seg+PAD)] ... [x: M rows, stride kseg+PAD]
  alignas(8) uint64_t full[STAGES_MAX], empty[STAGES_MAX], xfull, xempty;
  float red[NCW][MMAX][R];
  int flag;
};
static_assert(sizeof(Smem) + 128 <= 224128, "manifest shared_mem (GEMM8_SMEM) covers Smem");

enum Mode { PLAIN = 0, GATEUP = 1, DUAL = 2, NORM = 3, SGATE = 4 };

struct Args {
  const __nv_bfloat16* x;     // [M, ldx]
  const __nv_bfloat16* W;     // [N, K] (GATEUP: [2F, K])
  const __nv_bfloat16* W2;    // DUAL: rows >= N1
  __nv_bfloat16* y;           // PLAIN out [M, N]; DUAL y1; GATEUP act [M, F]; NORM: y scratch [M, N]
  __nv_bfloat16* y2;          // DUAL y2 [M, N2]
  float* partial;             // [nranges, MMAX, N] f32 scratch, or nullptr (GATEUP / DUAL: never split)
  int* gcnt;                  // [2] i32 scratch: finished tasks / finished rows, zero at rest
  int M, N, N1, K, ldx;
  const __nv_bfloat16* gate;  // SGATE: strided gate view; x is then attn [M, ldx]
  int head_dim, gate_ts, gate_hs;
};

// Tasks of one K range: nfull whole 8-row blocks, a multiple of the grid so
// every CTA streams the same number, plus the leftover rows spread over the
// grid in tasks of `rem_g` rows (GATEUP: whole gate/up pairs).  The small
// tasks come first: their stages are latency-bound, and at the head of a
// CTA's queue that latency hides under the full blocks streaming behind
// them instead of being the launch's last few microseconds.  The per-row
// accumulation order is the same either way.
struct Geom {
  int nrows, nfull, nsmall, na, hi, lo, nps;  // rows; full blocks; small tasks: na of hi rows then hi-gran rows; tasks per range
};
template <int MODE>
__device__ __forceinline__ Geom task_geom(const Args& a) {
  Geom g;
  g.nrows = MODE == GATEUP ? 2 * a.N : a.N;
  const int nblk = (g.nrows + R - 1) / R, grid = gridDim.x;
  g.nfull = nblk / grid * grid;
  // leftover rows over the whole grid in two task widths differing by one
  // granule, so no CTA streams more than a granule beyond another
  const int rem = g.nrows - g.nfull * R, gran = MODE == GATEUP ? 2 : 1;
  g.hi = min(R, ((rem + grid - 1) / grid + gran - 1) / gran * gran);
  g.lo = g.hi - gran;
  if (rem == 0) { g.na = g.nsmall = 0; }
  else if (a.M > 1) { g.hi = R; g.lo = 0; g.na = g.nsmall = (rem + R - 1) / R; }  // whole blocks: a narrow task costs full MMA rounds
  else if (g.lo == 0) { g.na = g.nsmall = (rem + g.hi - 1) / g.hi; }
  else { g.na = (rem - g.lo * grid + gran - 1) / gran; g.nsmall = grid; }
  g.nps = g.nfull + g.nsmall;
  return g;
}
__device__ __forceinline__ void task_rows(const Geom& g, int u, int& r0, int& rows) {
  if (u < g.nsmall) {
    r0 = g.nfull * R + (u < g.na ? u * g.hi : g.na * g.hi + (u - g.na) * g.lo);
    rows = min(u < g.na ? g.hi : g.lo, g.nrows - r0);
  } else { r0 = (u - g.nsmall) * R; rows = R; }
}


// K ranges: as many as x capacity demands, and at least enough tasks to
// balance the grid when the mode can split (partial != nullptr).
__device__ __forceinline__ int pick_nranges(const Args& a, int nblk) {
  int ks = 1;
  while ((long long)a.M * ((a.K + ks - 1) / ks + 31 & ~31) > X_ELEMS) ks++;
  const int kseg = (a.K + ks - 1) / ks + 31 & ~31;   // K % 32 == 0 keeps every stage in whole ldmatrix pairs
  return (a.K + kseg - 1) / kseg;
}

template <int MODE>
__device__ __forceinline__ const __nv_bfloat16* row_ptr(const Args& a, int row) {
  if (MODE == GATEUP) {  // logical rows are (gate j, up j) pairs
    const int j = row >> 1;
    return a.W + (size_t)((row & 1) ? a.N + j : j) * a.K;
  }
  if (MODE == DUAL) return row < a.N1 ? a.W + (size_t)row * a.K : a.W2 + (size_t)(row - a.N1) * a.K;
  return a.W + (size_t)row * a.K;
}

// Output (m, row) of one range.  Called by 64 threads (m = ct/8, r = ct%8) of
// two full warps, so the pair shuffle of GATEUP is convergent.
template <int MODE>
__device__ __forceinline__ void emit(const Args& a, int split, int nranges, int m, int row, bool valid, float v) {
  if (nranges > 1) {  // f32 range partial; the tail folds them
    if (valid) a.partial[((size_t)split * MMAX + m) * a.N + row] = v;
    return;
  }
  if (MODE == GATEUP) {
    const float other = __shfl_down_sync(0xffffffffu, v, 1);
    if (valid && (row & 1) == 0) {
      const float g = bf16r(v), u = bf16r(other);
      const float sg = bf16r(g / (1.0f + expf(-g)));
      a.y[(size_t)m * a.N + (row >> 1)] = __float2bfloat16_rn(sg * u);
    }
    return;
  }
  if (!valid) return;
  if (MODE == DUAL) {
    if (row < a.N1) a.y[(size_t)m * a.N1 + row] = __float2bfloat16_rn(v);
    else a.y2[(size_t)m * (a.N - a.N1) + row - a.N1] = __float2bfloat16_rn(v);
  } else {
    a.y[(size_t)m * a.N + row] = __float2bfloat16_rn(v);
  }
}

// The streaming GEMM proper.  Returns (in every thread) whether this CTA
// finished the grid's last task (tail duty when the grid is oversubscribed).
template <int MODE>
__device__ __forceinline__ bool gemm8_core(const Args& a, Smem& s) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const Geom g = task_geom<MODE>(a);
  const int nrows = g.nrows, nps = g.nps;
  const int nblk = (nrows + R - 1) / R;
  const int nranges = pick_nranges(a, nblk);
  const int kseg = (a.K + nranges - 1) / nranges + 31 & ~31;
  const int nst0 = (kseg + SEG_MAX - 1) / SEG_MAX;
  const int seg = (kseg + nst0 - 1) / nst0 + 31 & ~31;          // stage slot width, whole ldmatrix pairs
  const int srow = seg + PAD, xrow = kseg + PAD;                // smem row strides
  // x sits at the top of the shared area; whatever it leaves (never less
  // than RING_ELEMS) is ring depth -- at M = 1 that is 5-6 stages
  const int xbeg = (BUF_ELEMS - a.M * xrow) & ~63;
  __nv_bfloat16* const ring = s.buf;
  __nv_bfloat16* const xs = s.buf + xbeg;
  const int nstages = min(STAGES_MAX, xbeg / (R * srow));
  const int ntasks = nps * nranges;
  if (threadIdx.x == 0) {
    for (int i = 0; i < nstages; i++) {
      mbar_init(smem_u32(&s.full[i]), 1);
      mbar_init(smem_u32(&s.empty[i]), NCW);
    }
    mbar_init(smem_u32(&s.xfull), 1);
    mbar_init(smem_u32(&s.xempty), NCW);
    fence_barrier_init();
    s.flag = 0;
  }
  __syncthreads();
  if (warp == 0) {
    if (lane == 0) {
      int stage = 0;
      uint32_t phase = 0, xphase = 0;
      int cur_split = -1;
      // Weight stages of this CTA's first task fill the ring before the
      // upstream dependency resolves; the main loop skips those `pre` stages.
      int pre = 0;
      if ((int)blockIdx.x < ntasks) {
        const int t = blockIdx.x;
        const int split = t / nps;
        int r0, rows;
        task_rows(g, t - split * nps, r0, rows);
        const int k0 = split * kseg, klen = min(kseg, a.K - k0);
        for (int off = 0; off < klen && pre < nstages; off += seg, pre++) {
          const int len = min(seg, klen - off);
          mbar_wait(smem_u32(&s.empty[stage]), phase ^ 1);
          mbar_expect_tx(smem_u32(&s.full[stage]), rows * len * 2);
          __nv_bfloat16* dst = &ring[stage * R * srow];
          for (int r = 0; r < rows; r++)
            bulk_copy_1d(smem_u32(dst + r * srow), row_ptr<MODE>(a, r0 + r) + k0 + off, len * 2, smem_u32(&s.full[stage]));
          if (++stage == nstages) { stage = 0; phase ^= 1; }
        }
      }
      pdl_wait();
      for (int t = blockIdx.x; t < ntasks; t += gridDim.x) {
        const int split = t / nps;                                // range-major
        int r0, rows;
        task_rows(g, t - split * nps, r0, rows);
        const int k0 = split * kseg, klen = min(kseg, a.K - k0);
        if (MODE != SGATE && split != cur_split) {  // stage this range of x (consumers have released the old one)
          mbar_wait(smem_u32(&s.xempty), xphase);
          mbar_expect_tx(smem_u32(&s.xfull), a.M * klen * 2);
          for (int m = 0; m < a.M; m++)
            bulk_copy_1d(smem_u32(&xs[m * xrow]), a.x + (size_t)m * a.ldx + k0, klen * 2, smem_u32(&s.xfull));
          xphase ^= 1;
          cur_split = split;
        }
        for (int off = 0; off < klen; off += seg) {
          if (pre > 0) { pre--; continue; }  // already in flight (first task only)
          const int len = min(seg, klen - off);
          mbar_wait(smem_u32(&s.empty[stage]), phase ^ 1);
          mbar_expect_tx(smem_u32(&s.full[stage]), rows * len * 2);
          __nv_bfloat16* dst = &ring[stage * R * srow];
          for (int r = 0; r < rows; r++)
            bulk_copy_1d(smem_u32(dst + r * srow), row_ptr<MODE>(a, r0 + r) + k0 + off, len * 2, smem_u32(&s.full[stage]));
          if (++stage == nstages) { stage = 0; phase ^= 1; }
        }
      }
      pdl_trigger();  // our weight stream is issued: the dependent launch may take the SMs we vacate
    }
    __syncthreads();  // matches the compute warps' final barrier
    return s.flag != 0;
  }
  const int ct = threadIdx.x - 32, cw = warp - 1;
  pdl_wait();  // everything from here on touches upstream products or writes
  int stage = 0;
  uint32_t phase = 0, xphase = 0;
  int cur_split = -1;
  for (int t = blockIdx.x; t < ntasks; t += gridDim.x) {
    const int split = t / nps;
    int r0, rows;
    task_rows(g, t - split * nps, r0, rows);
    const int klen = min(kseg, a.K - split * kseg);
    if (split != cur_split) {  // release the old x range, wait for the new one
      if (MODE == SGATE) {  // the compute warps make this range of x themselves
        const int k0 = split * kseg, total = a.M * klen;
        named_bar_sync(BAR_C, CTHREADS);  // everyone is done reading the old range
        // 8 consecutive elements per thread: klen, head_dim and the batch
        // stride are all multiples of 8, so a batch never crosses a token row
        // or a head boundary and both loads and the smem store are one 16 B
        // vector each (elementwise scalar code here is latency-bound)
        for (int i0 = ct * 8; i0 < total; i0 += CTHREADS * 8) {
          const int m = i0 / klen, kk = i0 - m * klen, k = k0 + kk;
          const int h = k / a.head_dim, hd = k - h * a.head_dim;
          const uint4 gq = *reinterpret_cast<const uint4*>(a.gate + (size_t)m * a.gate_ts + (size_t)h * a.gate_hs + hd);
          const uint4 aq = *reinterpret_cast<const uint4*>(a.x + (size_t)m * a.ldx + k);
          const __nv_bfloat16* gv = reinterpret_cast<const __nv_bfloat16*>(&gq);
          const __nv_bfloat16* av = reinterpret_cast<const __nv_bfloat16*>(&aq);
          uint4 oq;
          __nv_bfloat16* ov = reinterpret_cast<__nv_bfloat16*>(&oq);
#pragma unroll
          for (int j = 0; j < 8; j++) {
            const float g = bf16r(1.0f / (1.0f + expf(-__bfloat162float(gv[j]))));
            ov[j] = __float2bfloat16_rn(__bfloat162float(av[j]) * g);
          }
          *reinterpret_cast<uint4*>(&xs[m * xrow + kk]) = oq;
        }
        named_bar_sync(BAR_C, CTHREADS);
      } else {
        if (lane == 0) mbar_arrive(smem_u32(&s.xempty));
        mbar_wait(smem_u32(&s.xfull), xphase);
        xphase ^= 1;
      }
      cur_split = split;
    }
    // ldmatrix.x4 row addresses: lane l supplies row l%8 of the 8x8 matrix
    // (l/8) = k tile-half; matrices (k 0-7, 8-15, 16-23, 24-31) of a 32-k
    // pair of tiles come back as (b0, b1, b0', b1') / (a0, a2, a0', a2').
    // x rows beyond M are never emitted: clamp them onto row M-1 (broadcast).
    const uint32_t xbase = smem_u32(&xs[min(lane & 7, a.M - 1) * xrow + (lane >> 3) * 8]);
    float c[4][4];
#pragma unroll
    for (int j = 0; j < 4; j++) c[j][0] = c[j][1] = c[j][2] = c[j][3] = 0.f;
    const int wrow = min(lane & 7, rows - 1);  // rows beyond the task: clamp (never emitted)
    for (int off = 0; off < klen; off += seg) {
      const int npairs = min(seg, klen - off) / 32;              // len % 32 == 0 (every K is)
      mbar_wait(smem_u32(&s.full[stage]), phase);
      const uint32_t wbase = smem_u32(&ring[stage * R * srow + wrow * srow + (lane >> 3) * 8]);
      const uint32_t xoff = xbase + off * 2;
      int pr = cw;
      for (; pr + NCW < npairs; pr += 2 * NCW) {
        uint32_t a0[4], a2[4], b0[4], b1[4];
#pragma unroll
        for (int j = 0; j < 2; j++) {
          const uint32_t o = (pr + j * NCW) * 64;                // 32 elements * 2 B
          ldsm_x4(a0[2 * j], a2[2 * j], a0[2 * j + 1], a2[2 * j + 1], xoff + o);
          ldsm_x4(b0[2 * j], b1[2 * j], b0[2 * j + 1], b1[2 * j + 1], wbase + o);
        }
#pragma unroll
        for (int j = 0; j < 4; j++) mma_bf16_16816(c[j][0], c[j][1], c[j][2], c[j][3], a0[j], 0u, a2[j], 0u, b0[j], b1[j]);
      }
      if (pr < npairs) {
        uint32_t a0[2], a2[2], b0[2], b1[2];
        ldsm_x4(a0[0], a2[0], a0[1], a2[1], xoff + pr * 64);
        ldsm_x4(b0[0], b1[0], b0[1], b1[1], wbase + pr * 64);
        mma_bf16_16816(c[0][0], c[0][1], c[0][2], c[0][3], a0[0], 0u, a2[0], 0u, b0[0], b1[0]);
        mma_bf16_16816(c[1][0], c[1][1], c[1][2], c[1][3], a0[1], 0u, a2[1], 0u, b0[1], b1[1]);
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(smem_u32(&s.empty[stage]));
      if (++stage == nstages) { stage = 0; phase ^= 1; }
    }
    // (c0, c1) = out[token q = lane/4][rows 2p, 2p+1], p = lane%4
    s.red[cw][lane >> 2][2 * (lane & 3)] = (c[0][0] + c[1][0]) + (c[2][0] + c[3][0]);
    s.red[cw][lane >> 2][2 * (lane & 3) + 1] = (c[0][1] + c[1][1]) + (c[2][1] + c[3][1]);
    named_bar_sync(BAR_C, CTHREADS);
    if (ct < MMAX * R) {
      const int m = ct >> 3, r = ct & 7;
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < NCW; w++) v += s.red[w][m][r];
      emit<MODE>(a, split, nranges, m, r0 + r, m < a.M && r < rows, v);
    }
    named_bar_sync(BAR_C, CTHREADS);
  }
  pdl_trigger();
  if (ct == 0) {  // one thread of the compute warps
    const int my = (ntasks - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x;
    if (my > 0 && a.gcnt) {
      __threadfence();
      const int old = atomicAdd(a.gcnt, my);
      if (old + my == ntasks) { __threadfence(); s.flag = 1; }
    }
  }
  __syncthreads();
  return s.flag != 0;
}

// ---- tail: fold range partials, optionally Gemma fused_add_rms_norm ----
__device__ __forceinline__ int last_pow2(int n) {
  n |= (n >> 1); n |= (n >> 2); n |= (n >> 4); n |= (n >> 8); n |= (n >> 16);
  return (n - (n >> 1)) > 0 ? (n - (n >> 1)) : 1;
}
// ATen ReduceConfig::set_block_dimension(dim0 = N/4, dim1 = rows), mnt=512.
__device__ __forceinline__ int aten_block_width(int dim0, int dim1) {
  const int max_threads = NT_ATEN;
  int dim0_pow2 = dim0 < max_threads ? last_pow2(dim0) : max_threads;
  int dim1_pow2 = dim1 < max_threads ? last_pow2(dim1) : max_threads;
  int block_width = min(dim0_pow2, 32);
  int block_height = min(dim1_pow2, max_threads / block_width);
  block_width = min(dim0_pow2, max_threads / block_height);
  return block_width;
}
// Sum of z[j]^2 over j in [0, N) in ATen's order (sq[j] = fmul_rn(z, z), then
// the vectorized-by-4 reduction of kern_gemma_rms_norm.cu) for W virtual
// lanes; the result is returned in every thread.  (Folding the W lanes in
// the registers of one warp instead costs 40 registers and a local-memory
// tree in every add_norm variant, and the M = 8 verify pass pays 3 us per
// launch for it; the smem tree is the same additions in the same order.)
__device__ float aten_row_sumsq(const float* __restrict__ z, int N, int W, float* __restrict__ s) {
  const int t = threadIdx.x;
  constexpr int VL = (NT_ATEN + NTHREADS - 1) / NTHREADS;
  if (t < 32) s[t] = 0.f;
  __syncthreads();
#pragma unroll
  for (int v = 0; v < VL; v++) {
    const int l = t + v * NTHREADS;
    if (l < W) {
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
      for (int idx = l; idx * 4 + 3 < N; idx += W) {
        const float4 x4 = *reinterpret_cast<const float4*>(z + idx * 4);
        acc[0] += __fmul_rn(x4.x, x4.x); acc[1] += __fmul_rn(x4.y, x4.y);
        acc[2] += __fmul_rn(x4.z, x4.z); acc[3] += __fmul_rn(x4.w, x4.w);
      }
      s[l] = ((acc[0] + acc[1]) + acc[2]) + acc[3];
    }
  }
  __syncthreads();
  for (int off = W / 2; off >= 32; off >>= 1) {
#pragma unroll
    for (int v = 0; v < VL; v++) {
      const int l = t + v * NTHREADS;
      if (l < off && l + off < W) s[l] += s[l + off];
    }
    __syncthreads();
  }
  float value = t < 32 ? s[t] : 0.f;
  if (t < 32) {
    for (int off = 16; off > 0; off >>= 1) value += __shfl_down_sync(0xffffffffu, value, off);
    if (t == 0) s[0] = value;
  }
  __syncthreads();
  return s[0];
}

// The f32 GEMM value of elements [j8, j8+4) of row m: the folded range
// partials, or the y scratch when the GEMM ran unsplit.
__device__ __forceinline__ float4 ysum4(const Args& a, int nranges, int m, int j) {
  if (nranges == 1) {
    const __nv_bfloat162 lo = __ldcg(reinterpret_cast<const __nv_bfloat162*>(a.y + (size_t)m * a.N + j));
    const __nv_bfloat162 hi = __ldcg(reinterpret_cast<const __nv_bfloat162*>(a.y + (size_t)m * a.N + j + 2));
    const float2 l = __bfloat1622float2(lo), h = __bfloat1622float2(hi);
    return make_float4(l.x, l.y, h.x, h.y);
  }
  // guarded full unroll: all range loads issue before the adds need them
  float4 p[8];
#pragma unroll
  for (int sp = 0; sp < 8; sp++)
    if (sp < nranges) p[sp] = __ldcg(reinterpret_cast<const float4*>(a.partial + ((size_t)sp * MMAX + m) * a.N + j));
  float4 v = p[0];
#pragma unroll
  for (int sp = 1; sp < 8; sp++)
    if (sp < nranges) { v.x += p[sp].x; v.y += p[sp].y; v.z += p[sp].z; v.w += p[sp].w; }
  return v;
}

// Rows [m0, m1) of the tail.  NORM: vLLM's Gemma fused_add_rms_norm with
// ATen's reduction width for the launch's M rows; z lives in smem (ring).
// PLAIN: just fold the partials to bf16 y.
// The tail is latency, not bandwidth: every global round trip of a row is
// issued for all of the thread's groups at once (TG = 6 float4 per thread
// covers N <= 6144), and w1 is already in smem (loaded during the wait).
constexpr int TG = (6144 / 4 + NTHREADS - 1) / NTHREADS;
template <int MODE>
__device__ void tail_rows(const Args& a, int nranges, int m0, int m1, const float* __restrict__ w1s, __nv_bfloat16* res,
                          __nv_bfloat16* out, float eps, float* __restrict__ z, float* __restrict__ s) {
  const int N = a.N;
  for (int m = m0; m < m1; m++) {
    float4 v[TG];
    float2 r0[TG], r1[TG];
    if (MODE >= NORM) {  // the residual pair of every group, issued as one batch
      uint32_t rw[TG][2];
#pragma unroll
      for (int gi = 0; gi < TG; gi++) {
        const int j = (threadIdx.x + gi * NTHREADS) * 4;
        const uint32_t* rr = reinterpret_cast<const uint32_t*>(res + (size_t)m * N + j);
        rw[gi][0] = j < N ? rr[0] : 0u;
        rw[gi][1] = j < N ? rr[1] : 0u;
      }
#pragma unroll
      for (int gi = 0; gi < TG; gi++) {
        r0[gi] = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&rw[gi][0]));
        r1[gi] = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&rw[gi][1]));
      }
    }
    if (nranges == 1) {  // y scratch, bf16: one batch of plain loads
      uint32_t yw[TG][2];
#pragma unroll
      for (int gi = 0; gi < TG; gi++) {
        const int j = (threadIdx.x + gi * NTHREADS) * 4;
        const uint32_t* yy = reinterpret_cast<const uint32_t*>(a.y + (size_t)m * N + j);
        yw[gi][0] = j < N ? __ldcg(yy) : 0u;
        yw[gi][1] = j < N ? __ldcg(yy + 1) : 0u;
      }
#pragma unroll
      for (int gi = 0; gi < TG; gi++) {
        const float2 l = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yw[gi][0]));
        const float2 h = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&yw[gi][1]));
        v[gi] = make_float4(l.x, l.y, h.x, h.y);
      }
    } else {
#pragma unroll
      for (int gi = 0; gi < TG; gi++) {
        const int j = (threadIdx.x + gi * NTHREADS) * 4;
        if (j < N) v[gi] = ysum4(a, nranges, m, j);
      }
    }
#pragma unroll
    for (int gi = 0; gi < TG; gi++) {
      const int j = (threadIdx.x + gi * NTHREADS) * 4;
      if (j >= N) continue;
      float4 y = v[gi];
      if (MODE >= NORM) {
        __nv_bfloat16* rr = res + (size_t)m * N + j;
        y.x = bf16r(y.x); y.y = bf16r(y.y); y.z = bf16r(y.z); y.w = bf16r(y.w);   // the bf16 y rounding
        y.x = __fadd_rn(y.x, r0[gi].x); y.y = __fadd_rn(y.y, r0[gi].y);
        y.z = __fadd_rn(y.z, r1[gi].x); y.w = __fadd_rn(y.w, r1[gi].y);
        *reinterpret_cast<__nv_bfloat162*>(rr) = __floats2bfloat162_rn(y.x, y.y);
        *reinterpret_cast<__nv_bfloat162*>(rr + 2) = __floats2bfloat162_rn(y.z, y.w);
        *reinterpret_cast<float4*>(z + j) = y;
      } else {
        __nv_bfloat16* yy = a.y + (size_t)m * N + j;
        *reinterpret_cast<__nv_bfloat162*>(yy) = __floats2bfloat162_rn(y.x, y.y);
        *reinterpret_cast<__nv_bfloat162*>(yy + 2) = __floats2bfloat162_rn(y.z, y.w);
      }
    }
    if (MODE < NORM) continue;
    __syncthreads();
    const int W = aten_block_width(N / 4, a.M);
    const float sum = aten_row_sumsq(z, N, W, s);
    const float factor = (float)a.M / (float)((long long)a.M * (long long)N);
    const float var = __fmul_rn(sum, factor);
    const float r = rsqrtf(__fadd_rn(var, eps));
#pragma unroll
    for (int gi = 0; gi < TG; gi++) {
      const int j = (threadIdx.x + gi * NTHREADS) * 4;
      if (j >= N) continue;
      const float4 zv = *reinterpret_cast<const float4*>(z + j);
      const float4 w0 = *reinterpret_cast<const float4*>(w1s + j);
      __nv_bfloat162 o0 = __floats2bfloat162_rn(__fmul_rn(__fmul_rn(zv.x, r), w0.x), __fmul_rn(__fmul_rn(zv.y, r), w0.y));
      __nv_bfloat162 o1 = __floats2bfloat162_rn(__fmul_rn(__fmul_rn(zv.z, r), w0.z), __fmul_rn(__fmul_rn(zv.w, r), w0.w));
      *reinterpret_cast<__nv_bfloat162*>(out + (size_t)m * N + j) = o0;
      *reinterpret_cast<__nv_bfloat162*>(out + (size_t)m * N + j + 2) = o1;
    }
    __syncthreads();
  }
}

// Tail driver: cooperative one-row-per-CTA when the whole grid is resident,
// else all rows in the CTA that finished last.
template <int MODE>
__device__ __forceinline__ void run_tail(const Args& a, Smem& s, bool last, __nv_bfloat16* res, const float* w1,
                                         __nv_bfloat16* out, float eps) {
  const Geom g = task_geom<MODE>(a);
  const int nblk = (g.nrows + R - 1) / R;
  const int nranges = pick_nranges(a, nblk);
  const int ntasks = g.nps * nranges;
  if (MODE == PLAIN && nranges == 1) return;  // nothing left to do
  static_assert(RING_ELEMS * 2 >= 2 * 6144 * 4, "z row + w1 copy: N <= 6144");
  float* z = reinterpret_cast<float*>(&s.buf[0]);
  float* w1s = z + 6144;
  if (gridDim.x <= nsmid()) {
    if ((int)blockIdx.x >= a.M) return;
    if (MODE >= NORM)  // the norm weight is not waiting on anyone: fetch it under the spin
      for (int j = threadIdx.x * 4; j < a.N; j += NTHREADS * 4) *reinterpret_cast<float4*>(w1s + j) = *reinterpret_cast<const float4*>(w1 + j);
    if (threadIdx.x == 0)
      while (ld_acquire(a.gcnt) < ntasks) __nanosleep(64);
    __syncthreads();
    tail_rows<MODE>(a, nranges, blockIdx.x, blockIdx.x + 1, w1s, res, out, eps, z, &s.red[0][0][0]);
    __syncthreads();
    if (threadIdx.x == 0 && atomicAdd(a.gcnt + 1, 1) == a.M - 1) {
      a.gcnt[1] = 0;
      a.gcnt[0] = 0;
      __threadfence();
    }
  } else if (last) {
    __threadfence();
    if (MODE >= NORM)
      for (int j = threadIdx.x * 4; j < a.N; j += NTHREADS * 4) *reinterpret_cast<float4*>(w1s + j) = *reinterpret_cast<const float4*>(w1 + j);
    __syncthreads();
    tail_rows<MODE>(a, nranges, 0, a.M, w1s, res, out, eps, z, &s.red[0][0][0]);
    __syncthreads();
    if (threadIdx.x == 0) { *a.gcnt = 0; __threadfence(); }
  }
}
}  // namespace

#define GEMM8_SMEM                                                                                     \
  extern __shared__ char smem_raw[];                                                                  \
  Smem& s = *reinterpret_cast<Smem*>((reinterpret_cast<uintptr_t>(smem_raw) + 127) & ~uintptr_t(127))

extern "C" __global__ void __launch_bounds__(NTHREADS, 1)
kern_gemm8_bf16(__nv_bfloat16* __restrict__ y, const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ W,
                float* __restrict__ partial, int* __restrict__ gcnt, int M, int N, int K) {
  GEMM8_SMEM;
  Args a{x, W, nullptr, y, nullptr, partial, gcnt, M, N, 0, K, K};
  const bool last = gemm8_core<PLAIN>(a, s);
  run_tail<PLAIN>(a, s, last, nullptr, nullptr, nullptr, 0.f);
}

extern "C" __global__ void __launch_bounds__(NTHREADS, 1)
kern_gemm8_gateup_silu_bf16(__nv_bfloat16* __restrict__ act, const __nv_bfloat16* __restrict__ x,
                            const __nv_bfloat16* __restrict__ W, int M, int F, int K) {
  GEMM8_SMEM;
  Args a{x, W, nullptr, act, nullptr, nullptr, nullptr, M, F, 0, K, K};   // M*K <= X_ELEMS: never splits
  gemm8_core<GATEUP>(a, s);
}

extern "C" __global__ void __launch_bounds__(NTHREADS, 1)
kern_gemm8_dual_bf16(__nv_bfloat16* __restrict__ y1, __nv_bfloat16* __restrict__ y2, const __nv_bfloat16* __restrict__ x,
                     const __nv_bfloat16* __restrict__ W1, const __nv_bfloat16* __restrict__ W2, int M, int N1, int N2, int K) {
  GEMM8_SMEM;
  Args a{x, W1, W2, y1, y2, nullptr, nullptr, M, N1 + N2, N1, K, K};      // M*K <= X_ELEMS: never splits
  gemm8_core<DUAL>(a, s);
}

extern "C" __global__ void __launch_bounds__(NTHREADS, 1)
kern_gemm8_sgate_add_norm_bf16(__nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ attn,
                               const __nv_bfloat16* __restrict__ gate, const __nv_bfloat16* __restrict__ W,
                               __nv_bfloat16* __restrict__ res, const float* __restrict__ w1, __nv_bfloat16* __restrict__ y,
                               float* __restrict__ partial, int* __restrict__ gcnt, int M, int N, int K, int head_dim,
                               int gate_tstride, int gate_hstride, float eps) {
  GEMM8_SMEM;
  Args a{attn, W, nullptr, y, nullptr, partial, gcnt, M, N, 0, K, K, gate, head_dim, gate_tstride, gate_hstride};
  const bool last = gemm8_core<SGATE>(a, s);
  run_tail<SGATE>(a, s, last, res, w1, out, eps);
}

extern "C" __global__ void __launch_bounds__(NTHREADS, 1)
kern_gemm8_add_norm_bf16(__nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ W,
                         __nv_bfloat16* __restrict__ res, const float* __restrict__ w1, __nv_bfloat16* __restrict__ y,
                         float* __restrict__ partial, int* __restrict__ gcnt, int M, int N, int K, float eps) {
  GEMM8_SMEM;
  Args a{x, W, nullptr, y, nullptr, partial, gcnt, M, N, 0, K, K};
  const bool last = gemm8_core<NORM>(a, s);
  run_tail<NORM>(a, s, last, res, w1, out, eps);
}
