// ref.h — CPU references for the K3 decode kernel set (docs/k3-kernel-abi.md).
//
// Every function here takes host arrays in *exactly* the layout the kernel
// sees: bf16 is a raw uint16_t, the KDA line is a byte blob indexed through
// line_index/line_bytes, the MLA slab is walked through a block table.  The
// harness compares layout-for-layout, so these are the contract.
//
// Accumulation is double unless the document explicitly places a bf16 landing
// point (f32->bf16 round-to-nearest-even, `f2b` below); the landing points are
// where §0 "数学原语" and each kernel's comment block put them.  The document
// says the rounding chain is not required to be bit-exact — the harness
// tolerance is the acceptance criterion — so the reference takes the most
// accurate accumulation and lands where the document lands.
//
// Nothing here includes CUDA headers: it is plain C++17 and can be compiled by
// a host compiler on its own.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

namespace k3ref {

// ---------------------------------------------------------------- constants
// docs/k3-kernel-abi.md §0 表
enum : int {
  H = 7168,
  HEADS = 96,
  HEAD_DIM = 128,
  INNER = HEADS * HEAD_DIM,   // 12288
  Q_LORA = 1536,
  KV_LORA = 512,
  ROPE = 64,
  KV_A = KV_LORA + ROPE,      // 576
  Q_B = HEADS * 192,          // 18432
  MLA_FUSED = Q_LORA + KV_A + INNER,  // 14400
  KDA_FUSED = 4 * INNER,      // 49152
  WSM = 256,
  EXPERTS = 224,
  TOPK = 16,
  LATENT = 3584,
  INTER = 3072,
  SHARED = 6144,
  DENSE_I = 33792,
  NB_MAX = 8,
  PAGE = 64,
  LATENT_ROW = 576,
};
static const int V_VOCAB = 163840;
static const float EPS = 1e-5f;
static const float LB = -5.0f;

// KDA line (§0 状态布局)
static const long long REC_BYTES = 6291456LL;              // f32[96][128][128]
static const long long WIN_BYTES = 3LL * INNER * 2;        // 73728 = bf16[3][12288]
static const long long LINE_BYTES = REC_BYTES + 3 * WIN_BYTES;  // 6512640

// ------------------------------------------------------------------- bf16
typedef uint16_t bf16;

static inline float b2f(bf16 h) {
  uint32_t u = (uint32_t)h << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// round-to-nearest-even f32 -> bf16, the same rule as __float2bfloat16_rn.
static inline bf16 f2b(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;  // NaN -> canonical qNaN
  uint32_t lsb = (u >> 16) & 1u;
  uint32_t rounded = u + 0x7fffu + lsb;
  return (bf16)(rounded >> 16);
}

// bf16 * bf16 -> bf16 (one rounding), i.e. __hmul.
static inline bf16 bmul(bf16 a, bf16 b) { return f2b(b2f(a) * b2f(b)); }

// One bf16 ULP at |x|; used by the harness tolerance.
static inline double bf16_ulp(double x) {
  double a = std::fabs(x);
  if (!(a > 1.17549435e-38)) return 9.183549616e-41;  // subnormal floor
  int e;
  std::frexp(a, &e);          // a in [0.5,1) * 2^e  =>  exponent field e-1
  return std::ldexp(1.0, (e - 1) - 7);
}

// --------------------------------------------------------------- primitives
static inline double sigmoidd(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// situ(g,u) = 4*tanh(g/4)*sigma(g) * 25*tanh(u/25)
static inline double situ(double g, double u) {
  return 4.0 * std::tanh(g / 4.0) * sigmoidd(g) * 25.0 * std::tanh(u / 25.0);
}

// rms(x, gamma) — round-before-scale: y = bf16(x * rsqrt(mean(x^2)+EPS));
// out = bf16(y * gamma)  (bf16 x bf16 -> bf16).
static inline void rms_g(const bf16* x, const bf16* gamma, bf16* out, int n) {
  double ss = 0.0;
  for (int i = 0; i < n; ++i) {
    double v = b2f(x[i]);
    ss += v * v;
  }
  double r = 1.0 / std::sqrt(ss / n + (double)EPS);
  for (int i = 0; i < n; ++i) {
    bf16 y = f2b((float)(b2f(x[i]) * r));
    out[i] = bmul(y, gamma[i]);
  }
}

// rms_nw(x): no weight, stays f32 (returned as double here).
static inline void rms_nw(const bf16* x, double* out, int n) {
  double ss = 0.0;
  for (int i = 0; i < n; ++i) {
    double v = b2f(x[i]);
    ss += v * v;
  }
  double r = 1.0 / std::sqrt(ss / n + (double)EPS);
  for (int i = 0; i < n; ++i) out[i] = b2f(x[i]) * r;
}

// ---------------------------------------------------------------- threading
template <class F>
static void parallel_for(int n, F f) {
  int nt = (int)std::thread::hardware_concurrency();
  if (nt < 1) nt = 1;
  if (nt > n) nt = n;
  if (n <= 0) return;
  if (nt <= 1) {
    for (int i = 0; i < n; ++i) f(i);
    return;
  }
  std::vector<std::thread> th;
  th.reserve(nt);
  for (int t = 0; t < nt; ++t) {
    th.emplace_back([=] {
      for (int i = t; i < n; i += nt) f(i);
    });
  }
  for (auto& x : th) x.join();
}

// =========================================================================
// K1  residual stream
// =========================================================================
//
// attnres (§0 数学原语): candidates are blocks[b,0..nb-1] then the prefix as
// candidate nb.  score_c = sum_i rms_nw(cand_c)[i]*sw[i]; p = softmax(score);
// mixed = bf16(sum_c p_c * cand_c) with f32 accumulation over the *raw*
// candidates (not the normalised ones).  nb == 0 => mixed = prefix.
static inline void attnres_row(const bf16* blocks_b,  // [NB_MAX, H] or null
                               const bf16* prefix_b,  // [H]
                               const float* sw, int nb, bf16* mixed) {
  if (nb <= 0) {
    for (int i = 0; i < H; ++i) mixed[i] = prefix_b[i];
    return;
  }
  int nc = nb + 1;
  std::vector<double> score(nc);
  std::vector<double> tmp(H);
  for (int c = 0; c < nc; ++c) {
    const bf16* cand = (c < nb) ? (blocks_b + (size_t)c * H) : prefix_b;
    rms_nw(cand, tmp.data(), H);
    double s = 0.0;
    for (int i = 0; i < H; ++i) s += tmp[i] * (double)sw[i];
    score[c] = s;
  }
  double m = score[0];
  for (int c = 1; c < nc; ++c) m = std::max(m, score[c]);
  double l = 0.0;
  for (int c = 0; c < nc; ++c) { score[c] = std::exp(score[c] - m); l += score[c]; }
  for (int c = 0; c < nc; ++c) score[c] /= l;
  for (int i = 0; i < H; ++i) {
    double acc = 0.0;
    for (int c = 0; c < nc; ++c) {
      const bf16* cand = (c < nb) ? (blocks_b + (size_t)c * H) : prefix_b;
      acc += score[c] * (double)b2f(cand[i]);
    }
    mixed[i] = f2b((float)acc);
  }
}

// [K1a] kern_k3_attnres_rms
//   mixed = attnres(blocks, prefix, nb);
//   if (snapshot) blocks[b, nb] = prefix;      (state, checked after)
//   normed = rms(mixed, gamma)
static inline void ref_attnres_rms(const bf16* prefix, bf16* blocks,
                                   const float* sw, const bf16* gamma,
                                   bf16* normed, int nb, int snapshot, int B) {
  std::vector<bf16> save(blocks ? (size_t)B * NB_MAX * H : 0);
  if (blocks) std::memcpy(save.data(), blocks, save.size() * 2);
  parallel_for(B, [&](int b) {
    std::vector<bf16> mixed(H);
    const bf16* blk = blocks ? (save.data() + (size_t)b * NB_MAX * H) : nullptr;
    attnres_row(blk, prefix + (size_t)b * H, sw, nb, mixed.data());
    rms_g(mixed.data(), gamma, normed + (size_t)b * H, H);
  });
  if (snapshot && blocks) {
    for (int b = 0; b < B; ++b)
      std::memcpy(blocks + ((size_t)b * NB_MAX + nb) * H, prefix + (size_t)b * H,
                  (size_t)H * 2);
  }
}

// [K1b] kern_k3_land_add_attnres_rms
//   p = bf16(partial[b,:H]); prefix2 = snapshot ? p : bf16(prefix + p);
//   mixed = attnres(blocks, prefix2, nb); normed = rms(mixed, gamma)
// The p landing then the add landing are two roundings (pegainfer's chain).
static inline void ref_land_add_attnres_rms(const float* partial,
                                            const bf16* prefix,
                                            const bf16* blocks, const float* sw,
                                            const bf16* gamma, bf16* prefix2,
                                            bf16* normed, int nb, int snapshot,
                                            int B) {
  parallel_for(B, [&](int b) {
    bf16* p2 = prefix2 + (size_t)b * H;
    for (int i = 0; i < H; ++i) {
      bf16 p = f2b(partial[(size_t)b * H + i]);
      p2[i] = snapshot ? p : f2b(b2f(prefix[(size_t)b * H + i]) + b2f(p));
    }
    std::vector<bf16> mixed(H);
    const bf16* blk = blocks ? (blocks + (size_t)b * NB_MAX * H) : nullptr;
    attnres_row(blk, p2, sw, nb, mixed.data());
    rms_g(mixed.data(), gamma, normed + (size_t)b * H, H);
  });
}

// [K1c] kern_k3_land_add2
//   hidden = bf16( prefix2 + bf16(p1) + (two ? bf16(p2) : 0) )
// `two == 0` is the dense layer: p2 is a valid pointer but must not be read.
static inline void ref_land_add2(const float* p1, const float* p2,
                                 const bf16* prefix2, bf16* hidden, int two,
                                 int B) {
  parallel_for(B, [&](int b) {
    for (int i = 0; i < H; ++i) {
      size_t k = (size_t)b * H + i;
      double a = (double)b2f(prefix2[k]) + (double)b2f(f2b(p1[k]));
      if (two) a += (double)b2f(f2b(p2[k]));
      hidden[k] = f2b((float)a);
    }
  });
}

// =========================================================================
// K2  kern_k3_conv_silu
// =========================================================================
// Line helpers: window s of row b lives at line + REC_BYTES + s*73728, and is
// bf16[3 tap][INNER] with tap 0 the oldest.
static inline bf16* win_ptr(void* kda_base, const int* line_index,
                            long long line_bytes, int b, int s) {
  uint8_t* p = (uint8_t*)kda_base + (long long)line_index[b] * line_bytes +
               REC_BYTES + (long long)s * WIN_BYTES;
  return (bf16*)p;
}
static inline float* rec_ptr(void* kda_base, const int* line_index,
                             long long line_bytes, int b) {
  return (float*)((uint8_t*)kda_base + (long long)line_index[b] * line_bytes);
}

// for s in {0,1,2} (q,k,v), column c:
//   x  = bf16(partial[b, s*INNER + c])
//   y  = sum_{t<3} f32(win_s[t][c])*cw[s][t][c] + f32(x)*cw[s][3][c]
//   sb = bf16(y);  out_s[b,c] = bf16(sb * sigma(sb))
//   window shifts: win[0]=win[1]; win[1]=win[2]; win[2]=x
static inline void ref_conv_silu(const float* partial, const float* cw,
                                 void* kda_base, const int* line_index,
                                 long long line_bytes, bf16* conv_q,
                                 bf16* conv_k, bf16* conv_v, int B) {
  bf16* outs[3] = {conv_q, conv_k, conv_v};
  parallel_for(B, [&](int b) {
    for (int s = 0; s < 3; ++s) {
      bf16* w = win_ptr(kda_base, line_index, line_bytes, b, s);
      for (int c = 0; c < INNER; ++c) {
        bf16 x = f2b(partial[(size_t)b * KDA_FUSED + (size_t)s * INNER + c]);
        double y = 0.0;
        for (int t = 0; t < 3; ++t)
          y += (double)b2f(w[(size_t)t * INNER + c]) *
               (double)cw[((size_t)s * 4 + t) * INNER + c];
        y += (double)b2f(x) * (double)cw[((size_t)s * 4 + 3) * INNER + c];
        bf16 sb = f2b((float)y);
        double sv = b2f(sb);
        outs[s][(size_t)b * INNER + c] = f2b((float)(sv * sigmoidd(sv)));
        w[(size_t)0 * INNER + c] = w[(size_t)1 * INNER + c];
        w[(size_t)1 * INNER + c] = w[(size_t)2 * INNER + c];
        w[(size_t)2 * INNER + c] = x;
      }
    }
  });
}

// =========================================================================
// K3  kern_k3_kda_core
// =========================================================================
// Per row b, head h (see the kernel comment block in the ABI doc).  The q and
// k normalisation chains are bf16 throughout; `rec` is updated in place.
static inline void ref_kda_core(const bf16* conv_q, const bf16* conv_k,
                                const bf16* conv_v, const float* wsm_partial,
                                const float* gate_partial, const bf16* w_f_b,
                                const float* dt_bias, const float* a_log,
                                const float* gamma_o, void* kda_base,
                                const int* line_index, long long line_bytes,
                                bf16* out, int B) {
  parallel_for(B * HEADS, [&](int bh) {
    int b = bh / HEADS, h = bh % HEADS;
    const bf16* q = conv_q + (size_t)b * INNER + (size_t)h * 128;
    const bf16* k = conv_k + (size_t)b * INNER + (size_t)h * 128;
    const bf16* v = conv_v + (size_t)b * INNER + (size_t)h * 128;

    // qtot = sum_d bf16(q[d]*q[d]) accumulated in f32; qr = bf16(rsqrt(f32(bf16(qtot))+1e-6))
    double qtot = 0.0, ktot = 0.0;
    for (int d = 0; d < 128; ++d) {
      qtot += (double)b2f(bmul(q[d], q[d]));
      ktot += (double)b2f(bmul(k[d], k[d]));
    }
    bf16 qr = f2b((float)(1.0 / std::sqrt((double)b2f(f2b((float)qtot)) + 1e-6)));
    bf16 kr = f2b((float)(1.0 / std::sqrt((double)b2f(f2b((float)ktot)) + 1e-6)));

    double qs[128], kn[128];
    const double inv_sqrt_128 = 1.0 / std::sqrt(128.0);
    for (int d = 0; d < 128; ++d) {
      qs[d] = (double)b2f(bmul(q[d], qr)) * inv_sqrt_128;
      kn[d] = (double)b2f(bmul(k[d], kr));
    }

    // beta from wsm column h; flow from wsm columns 96..224
    double beta = sigmoidd((double)b2f(f2b(wsm_partial[(size_t)b * WSM + h])));
    bf16 flow[128];
    for (int j = 0; j < 128; ++j)
      flow[j] = f2b(wsm_partial[(size_t)b * WSM + 96 + j]);

    double dec[128];
    double ea = std::exp((double)a_log[h]);
    for (int d = 0; d < 128; ++d) {
      double ga = 0.0;
      for (int j = 0; j < 128; ++j)
        ga += (double)b2f(flow[j]) *
              (double)b2f(w_f_b[((size_t)h * 128 + d) * 128 + j]);
      double raw = (double)b2f(f2b((float)ga)) + (double)dt_bias[(size_t)h * 128 + d];
      dec[d] = std::exp((double)LB * sigmoidd(ea * raw));
    }

    float* S = rec_ptr(kda_base, line_index, line_bytes, b) +
               (size_t)h * 128 * 128;
    bf16 attn[128];
    double attn_f[128];
    for (int dv = 0; dv < 128; ++dv) {
      float* Srow = S + (size_t)dv * 128;
      double m = 0.0;
      for (int kk = 0; kk < 128; ++kk) m += (double)Srow[kk] * dec[kk] * kn[kk];
      double dlt = ((double)b2f(v[dv]) - m) * beta;
      double a = 0.0;
      for (int kk = 0; kk < 128; ++kk) {
        float sp = (float)((double)Srow[kk] * dec[kk] + dlt * kn[kk]);
        Srow[kk] = sp;            // rec is f32 state: attn sees the stored value
        a += (double)sp * qs[kk];
      }
      attn[dv] = f2b((float)a);
      attn_f[dv] = b2f(attn[dv]);
    }

    double ss = 0.0;
    for (int dv = 0; dv < 128; ++dv) ss += attn_f[dv] * attn_f[dv];
    double r = 1.0 / std::sqrt(ss / 128.0 + (double)EPS);
    for (int d = 0; d < 128; ++d) {
      bf16 o = f2b((float)(attn_f[d] * r * (double)gamma_o[d]));
      double g = (double)b2f(f2b(gate_partial[(size_t)b * KDA_FUSED +
                                              3 * INNER + (size_t)h * 128 + d]));
      bf16 gt = f2b((float)sigmoidd(g));
      out[(size_t)b * INNER + (size_t)h * 128 + d] = bmul(o, gt);
    }
  });
}

// =========================================================================
// K9  kern_k3_span_gather:  K2 applied to batch rows at..at+span one after
//     another, all on row at's line, into the span's own rows 0..span;
//     beta transposed [HEADS][span], flow bf16 [span][128]
// =========================================================================
static inline void ref_span_gather(const float* partial, const float* cw,
                                   void* kda_base, const int* line_index,
                                   long long line_bytes, const float* wsm,
                                   bf16* span_q, bf16* span_k, bf16* span_v,
                                   bf16* span_beta, bf16* span_flow,
                                   const int* span_at, int span) {
  const int at = *span_at;
  partial += (size_t)at * KDA_FUSED;
  wsm += (size_t)at * WSM;
  line_index += at;
  for (int i = 0; i < span; ++i)
    ref_conv_silu(partial + (size_t)i * KDA_FUSED, cw, kda_base, line_index,
                  line_bytes, span_q + (size_t)i * INNER,
                  span_k + (size_t)i * INNER, span_v + (size_t)i * INNER, 1);
  for (int i = 0; i < span; ++i) {
    for (int h = 0; h < HEADS; ++h)
      span_beta[(size_t)h * span + i] = f2b(wsm[(size_t)i * WSM + h]);
    for (int j = 0; j < 128; ++j)
      span_flow[(size_t)i * 128 + j] = f2b(wsm[(size_t)i * WSM + 96 + j]);
  }
}

// =========================================================================
// K10 kern_k3_span_state:  rec of row at's line <-> buf [HEADS][128][128] f32
// =========================================================================
static inline void ref_span_state(void* kda_base, const int* line_index,
                                  long long line_bytes, const int* span_at,
                                  float* buf, int to_line) {
  float* rec = rec_ptr(kda_base, line_index, line_bytes, *span_at);
  if (to_line) std::memcpy(rec, buf, (size_t)REC_BYTES);
  else std::memcpy(buf, rec, (size_t)REC_BYTES);
}

// =========================================================================
// K11 kern_k3_kda_out_gate:  the K3 epilogue on the span's raw bf16 attention
//     rows 0..span, into batch rows at..at+span of `gated`
// =========================================================================
static inline void ref_kda_out_gate(const bf16* attn, const float* gate_partial,
                                    const float* gamma_o, bf16* gated,
                                    const int* span_at, int span) {
  parallel_for(span * HEADS, [&](int ih) {
    int i = ih / HEADS, h = ih % HEADS, b = *span_at + i;
    const bf16* row = attn + (size_t)i * INNER + (size_t)h * 128;
    bf16* dst = gated + (size_t)b * INNER + (size_t)h * 128;
    double ss = 0.0;
    for (int dv = 0; dv < 128; ++dv) ss += (double)b2f(row[dv]) * (double)b2f(row[dv]);
    double r = 1.0 / std::sqrt(ss / 128.0 + (double)EPS);
    for (int d = 0; d < 128; ++d) {
      bf16 o = f2b((float)((double)b2f(row[d]) * r * (double)gamma_o[d]));
      double g = (double)b2f(f2b(gate_partial[(size_t)b * KDA_FUSED +
                                              3 * INNER + (size_t)h * 128 + d]));
      dst[d] = bmul(o, f2b((float)sigmoidd(g)));
    }
  });
}

// =========================================================================
// K4  kern_k3_mla_prep
// =========================================================================
//   q_norm  = rms(bf16(P[0..1536]),    gamma_q_a)
//   kv_norm = rms(bf16(P[1536..2048]), gamma_kv_a)
//   rope    = bf16(P[2048..2112])
//   slab row (slot/64)*page_stride + layer_off + (slot%64)*576 = kv_norm|rope
//   mla_gate = bf16(P[2112..14400])
static inline void ref_mla_prep(const float* partial, const bf16* gamma_q_a,
                                const bf16* gamma_kv_a, const int64_t* slot_mapping,
                                bf16* slab, long long layer_off,
                                long long page_stride, bf16* q_norm,
                                bf16* mla_gate, int B) {
  parallel_for(B, [&](int b) {
    const float* P = partial + (size_t)b * MLA_FUSED;
    std::vector<bf16> qa(Q_LORA), kva(KV_LORA), kvn(KV_LORA);
    for (int i = 0; i < Q_LORA; ++i) qa[i] = f2b(P[i]);
    rms_g(qa.data(), gamma_q_a, q_norm + (size_t)b * Q_LORA, Q_LORA);
    for (int i = 0; i < KV_LORA; ++i) kva[i] = f2b(P[Q_LORA + i]);
    rms_g(kva.data(), gamma_kv_a, kvn.data(), KV_LORA);
    long long slot = slot_mapping[b];
    bf16* row = slab + (slot / PAGE) * page_stride + layer_off +
                (slot % PAGE) * (long long)LATENT_ROW;
    for (int i = 0; i < KV_LORA; ++i) row[i] = kvn[i];
    for (int i = 0; i < ROPE; ++i) row[KV_LORA + i] = f2b(P[Q_LORA + KV_LORA + i]);
    for (int i = 0; i < INNER; ++i)
      mla_gate[(size_t)b * INNER + i] = f2b(P[Q_LORA + KV_A + i]);
  });
}

// =========================================================================
// K5  kern_k3_mla_paged_attn
// =========================================================================
// `cache` is the slab already shifted to this layer (base + layer_off).
// If `gated` is null the ungated o (bf16[B,INNER]) is written to `ungated`
// instead — that is the old pegainfer ABI, used for the baseline compare.
static inline void ref_mla_paged_attn(const float* q_partial, const bf16* w_kv_b,
                                      const bf16* cache, const int* block_table,
                                      int max_pages, long long page_stride,
                                      const int* seq_lens, const bf16* scale,
                                      const bf16* mla_gate, bf16* gated,
                                      bf16* ungated, int B) {
  parallel_for(B * HEADS, [&](int bh) {
    int b = bh / HEADS, h = bh % HEADS;
    const float* qp = q_partial + (size_t)b * Q_B + (size_t)h * 192;
    bf16 qh[192];
    for (int d = 0; d < 192; ++d) qh[d] = f2b(qp[d]);

    // q_abs = [ bf16(sum_d q_h[d]*W_UK_h[d,j]) | q_h[128..192] ]
    // W_UK_h = w_kv_b[h*256 + 0..128, :]
    std::vector<bf16> qabs(LATENT_ROW);
    for (int j = 0; j < KV_LORA; ++j) {
      double acc = 0.0;
      for (int d = 0; d < 128; ++d)
        acc += (double)b2f(qh[d]) *
               (double)b2f(w_kv_b[((size_t)h * 256 + d) * KV_LORA + j]);
      qabs[j] = f2b((float)acc);
    }
    for (int j = 0; j < ROPE; ++j) qabs[KV_LORA + j] = qh[128 + j];

    int n = seq_lens[b];
    std::vector<double> s(n);
    for (int t = 0; t < n; ++t) {
      int page = block_table[(size_t)b * max_pages + t / PAGE];
      const bf16* row = cache + (long long)page * page_stride +
                        (long long)(t % PAGE) * LATENT_ROW;
      double acc = 0.0;
      for (int d = 0; d < LATENT_ROW; ++d)
        acc += (double)b2f(qabs[d]) * (double)b2f(row[d]);
      // bf16 landing of the dot, then a bf16 multiply by the bf16 scale.
      s[t] = (double)b2f(bmul(f2b((float)acc), scale[0]));
    }
    double m = s.empty() ? 0.0 : s[0];
    for (int t = 1; t < n; ++t) m = std::max(m, s[t]);
    double l = 0.0;
    for (int t = 0; t < n; ++t) l += std::exp(s[t] - m);
    std::vector<double> p(n);
    for (int t = 0; t < n; ++t) p[t] = b2f(f2b((float)(std::exp(s[t] - m) / l)));

    std::vector<bf16> lat(KV_LORA);
    std::vector<double> acc(KV_LORA, 0.0);
    for (int t = 0; t < n; ++t) {
      int page = block_table[(size_t)b * max_pages + t / PAGE];
      const bf16* row = cache + (long long)page * page_stride +
                        (long long)(t % PAGE) * LATENT_ROW;
      double pt = p[t];
      for (int j = 0; j < KV_LORA; ++j) acc[j] += pt * (double)b2f(row[j]);
    }
    for (int j = 0; j < KV_LORA; ++j) lat[j] = f2b((float)acc[j]);

    for (int dv = 0; dv < 128; ++dv) {
      double o = 0.0;
      for (int j = 0; j < KV_LORA; ++j)
        o += (double)b2f(w_kv_b[((size_t)h * 256 + 128 + dv) * KV_LORA + j]) *
             (double)b2f(lat[j]);
      bf16 ob = f2b((float)o);
      size_t k = (size_t)b * INNER + (size_t)h * 128 + dv;
      if (ungated) ungated[k] = ob;
      if (gated) {
        bf16 g = f2b((float)sigmoidd((double)b2f(mla_gate[k])));
        gated[k] = bmul(ob, g);
      }
    }
  });
}

// =========================================================================
// K6  router / argmax / rms
// =========================================================================
// sig = sigma(S[b,e]); biased = sig + bias[e]; 16 sequential max passes,
// ties take the smaller e; wts[t] = sig[idx[t]] / (sum + 1e-20) * f32(rs[0]).
static inline void ref_router_topk(const float* S, const float* bias,
                                   const bf16* rs, int* idx, float* wts, int B) {
  parallel_for(B, [&](int b) {
    std::vector<double> sig(EXPERTS), biased(EXPERTS);
    for (int e = 0; e < EXPERTS; ++e) {
      sig[e] = sigmoidd((double)S[(size_t)b * EXPERTS + e]);
      biased[e] = sig[e] + (double)bias[e];
    }
    std::vector<char> taken(EXPERTS, 0);
    double sum = 0.0;
    for (int t = 0; t < TOPK; ++t) {
      int best = -1;
      double bv = 0.0;
      for (int e = 0; e < EXPERTS; ++e) {
        if (taken[e]) continue;
        if (best < 0 || biased[e] > bv) { best = e; bv = biased[e]; }
      }
      taken[best] = 1;
      idx[(size_t)b * TOPK + t] = best;
      sum += sig[best];
    }
    double sc = (double)b2f(rs[0]);
    for (int t = 0; t < TOPK; ++t)
      wts[(size_t)b * TOPK + t] =
          (float)(sig[idx[(size_t)b * TOPK + t]] / (sum + 1e-20) * sc);
  });
}

// argmax over f32 logits, tie -> smallest index.  The two-stage kernel's
// partial split is the author's choice (the doc fixes only grid (B,64)); the
// reference produces the final answer plus, for the default grid-stride split,
// the partial pair — the harness validates partials by invariant.
static inline void ref_argmax_f32(const float* logits, int64_t* out, int n, int B) {
  parallel_for(B, [&](int b) {
    const float* row = logits + (size_t)b * n;
    int bi = 0;
    float bv = row[0];
    for (int i = 1; i < n; ++i)
      if (row[i] > bv) { bv = row[i]; bi = i; }
    out[b] = (int64_t)bi;
  });
}

static inline void ref_rms(const bf16* x, const bf16* gamma, bf16* o, int h,
                           int B) {
  parallel_for(B, [&](int b) {
    rms_g(x + (size_t)b * h, gamma, o + (size_t)b * h, h);
  });
}

// =========================================================================
// K7  land / land_situ
// =========================================================================
// o[b,i] = bf16(p[b*ldc + off + i])
static inline void ref_land(const float* p, bf16* o, int n, int off, int ldc,
                            int B) {
  parallel_for(B, [&](int b) {
    for (int i = 0; i < n; ++i)
      o[(size_t)b * n + i] = f2b(p[(size_t)b * ldc + off + i]);
  });
}

// act[b,i] = bf16( situ( f32(bf16(p[b*2n+i])), f32(bf16(p[b*2n+n+i])) ) )
static inline void ref_land_situ(const float* p, bf16* act, int n, int B) {
  parallel_for(B, [&](int b) {
    for (int i = 0; i < n; ++i) {
      double g = b2f(f2b(p[(size_t)b * 2 * n + i]));
      double u = b2f(f2b(p[(size_t)b * 2 * n + n + i]));
      act[(size_t)b * n + i] = f2b((float)situ(g, u));
    }
  });
}

}  // namespace k3ref
