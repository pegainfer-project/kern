// harness.cu — the acceptance harness for the K3 decode kernel set.
//
//   /usr/local/cuda-13.1/bin/nvcc -O2 -std=c++17 -arch=sm_103a harness.cu -o harness -lcuda
//
// It knows nothing about how a kernel was built: it loads a cubin through the
// CUDA driver API (cuModuleLoad / cuModuleGetFunction / cuLaunchKernel) and
// launches the documented entry with the documented grid/block/smem.
//
//   ./harness --kernel <name> --cubin <path> [--B 1|2|8|64] [--ctx N] [--nb N]
//             [--snapshot 0|1] [--two 0|1] [--reps N] [--seed N]
//             [--nmla N] [--layer K]
//             [--grid gx,gy,gz] [--block bx,by,bz] [--smem bytes]
//
// See README.md.  ref.h holds the CPU references; this file only generates
// inputs, launches, compares and times.
#include <cuda.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "ref.h"

using namespace k3ref;

// ------------------------------------------------------------------ driver
static void cu_check(CUresult r, const char* what, int line) {
  if (r == CUDA_SUCCESS) return;
  const char *n = nullptr, *s = nullptr;
  cuGetErrorName(r, &n);
  cuGetErrorString(r, &s);
  std::fprintf(stderr, "CUDA error at line %d: %s -> %s (%s)\n", line, what,
               n ? n : "?", s ? s : "?");
  std::exit(2);
}
#define CU(x) cu_check((x), #x, __LINE__)

static std::vector<CUdeviceptr> g_allocs;
static CUdeviceptr dmalloc(size_t bytes) {
  CUdeviceptr p = 0;
  CU(cuMemAlloc(&p, bytes ? bytes : 1));
  g_allocs.push_back(p);
  return p;
}
static CUdeviceptr dput(const void* h, size_t bytes) {
  CUdeviceptr p = dmalloc(bytes);
  if (bytes) CU(cuMemcpyHtoD(p, h, bytes));
  return p;
}
// output buffer: allocated and poisoned so a kernel that fails to write the
// whole thing is caught ("输出 buffer 必须整块写满").
static CUdeviceptr dpoison(size_t bytes) {
  CUdeviceptr p = dmalloc(bytes);
  CU(cuMemsetD8(p, 0x5A, bytes));
  return p;
}
static void dget(void* h, CUdeviceptr p, size_t bytes) {
  if (bytes) CU(cuMemcpyDtoH(h, p, bytes));
}

// --------------------------------------------------------------------- rng
struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 0x1234567ull) {}
  uint64_t next() {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  double u01() { return (double)(next() >> 11) * (1.0 / 9007199254740992.0); }
  double normal() {
    double u1 = u01();
    if (u1 < 1e-12) u1 = 1e-12;
    double u2 = u01();
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
  }
  uint32_t below(uint32_t n) { return (uint32_t)(next() % n); }
};

static void fill_f32(std::vector<float>& v, Rng& r, double mu, double sd) {
  for (auto& x : v) x = (float)(mu + sd * r.normal());
}
static void fill_bf16(std::vector<bf16>& v, Rng& r, double mu, double sd) {
  for (auto& x : v) x = f2b((float)(mu + sd * r.normal()));
}
static std::vector<int> shuffled_iota(int n, Rng& r) {
  std::vector<int> v(n);
  for (int i = 0; i < n; ++i) v[i] = i;
  for (int i = n - 1; i > 0; --i) std::swap(v[i], v[r.below((uint32_t)i + 1)]);
  return v;
}

// ---------------------------------------------------------------- compare
// docs §2: |err| <= 3 * bf16ULP(|ref|) + 1e-3  AND  relative RMS <= 2e-3.
static const double kUlpK = 3.0;
static const double kAbsAdd = 1e-3;
static const double kRelRms = 2e-3;

static bool g_fail = false;
static double g_median_us = 0.0, g_roof_bytes = 0.0;

template <class G, class R>
static void cmp_gen(const char* name, size_t n, G got, R ref) {
  double max_abs = 0.0, max_ulp = 0.0, se = 0.0, sr = 0.0;
  size_t bad = 0, first_bad = 0;
  double bad_g = 0, bad_r = 0;
  for (size_t i = 0; i < n; ++i) {
    double g = got(i), rv = ref(i);
    double e = std::fabs(g - rv);
    if (!(g == g) || !(rv == rv)) {  // NaN
      if (!bad++) { first_bad = i; bad_g = g; bad_r = rv; }
      continue;
    }
    double u = e / bf16_ulp(rv);
    if (e > max_abs) max_abs = e;
    if (u > max_ulp) max_ulp = u;
    se += e * e;
    sr += rv * rv;
    if (e > kUlpK * bf16_ulp(rv) + kAbsAdd) {
      if (!bad++) { first_bad = i; bad_g = g; bad_r = rv; }
    }
  }
  double rms = (sr > 0.0) ? std::sqrt(se / sr) : std::sqrt(se / (double)(n ? n : 1));
  bool ok = (bad == 0) && (rms <= kRelRms);
  if (!ok) g_fail = true;
  std::printf("  %-12s n=%-11zu max|err|=%.3e  maxULP=%6.2f  relRMS=%.3e  %s\n",
              name, n, max_abs, max_ulp, rms, ok ? "PASS" : "FAIL");
  if (bad)
    std::printf("      %zu elements out of tolerance; first at %zu: got %.9g ref %.9g\n",
                bad, first_bad, bad_g, bad_r);
}

static void cmp_bf16(const char* name, const std::vector<bf16>& got,
                     const std::vector<bf16>& ref) {
  cmp_gen(name, ref.size(), [&](size_t i) { return (double)b2f(got[i]); },
          [&](size_t i) { return (double)b2f(ref[i]); });
}
static void cmp_f32(const char* name, const std::vector<float>& got,
                    const std::vector<float>& ref) {
  cmp_gen(name, ref.size(), [&](size_t i) { return (double)got[i]; },
          [&](size_t i) { return (double)ref[i]; });
}
template <class T>
static void cmp_exact(const char* name, const std::vector<T>& got,
                      const std::vector<T>& ref) {
  size_t bad = 0, first = 0;
  for (size_t i = 0; i < ref.size(); ++i)
    if (got[i] != ref[i]) { if (!bad++) first = i; }
  bool ok = bad == 0;
  if (!ok) g_fail = true;
  std::printf("  %-12s n=%-11zu exact-match                                 %s\n",
              name, ref.size(), ok ? "PASS" : "FAIL");
  if (bad)
    std::printf("      %zu mismatches; first at %zu: got %lld ref %lld\n", bad,
                first, (long long)got[first], (long long)ref[first]);
}

// ------------------------------------------------------------------ options
struct Opt {
  std::string kernel, cubin;
  int B = 1, ctx = 2048, nb = 4, snapshot = 1, two = 1, reps = 50;
  int nmla = 1, layer = 0;
  int span = 0;  // K2 / K3: rows [1, 1 + span) are a span the kernel must skip
  uint64_t seed = 1234;
  int gx = -1, gy = -1, gz = -1, bx = -1, by = -1, bz = -1;
  long long smem = -1;
};

struct Geo {
  unsigned gx, gy, gz, bx, by, bz, smem;
};
// Launch geometry with the CLI overrides applied; `gx` is the document's
// grid.x (the batch for most kernels).
static Geo geo_at(const Opt& o, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by,
                  unsigned bz, unsigned smem) {
  Geo g;
  g.gx = o.gx > 0 ? (unsigned)o.gx : gx;
  g.gy = o.gy > 0 ? (unsigned)o.gy : gy;
  g.gz = o.gz > 0 ? (unsigned)o.gz : gz;
  g.bx = o.bx > 0 ? (unsigned)o.bx : bx;
  g.by = o.by > 0 ? (unsigned)o.by : by;
  g.bz = o.bz > 0 ? (unsigned)o.bz : bz;
  g.smem = o.smem >= 0 ? (unsigned)o.smem : smem;
  std::printf("  launch       grid(%u,%u,%u) block(%u,%u,%u) smem=%u\n", g.gx,
              g.gy, g.gz, g.bx, g.by, g.bz, g.smem);
  return g;
}
static Geo geo(const Opt& o, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz, unsigned smem) {
  return geo_at(o, (unsigned)o.B, gy, gz, bx, by, bz, smem);
}

static CUfunction getfn(CUmodule m, const char* name) {
  CUfunction f = nullptr;
  CUresult r = cuModuleGetFunction(&f, m, name);
  if (r != CUDA_SUCCESS) {
    std::fprintf(stderr, "cubin has no entry `%s`\n", name);
    std::exit(2);
  }
  return f;
}
static void launch(CUfunction f, const Geo& g, void** args) {
  CU(cuLaunchKernel(f, g.gx, g.gy, g.gz, g.bx, g.by, g.bz, g.smem, 0, args,
                    nullptr));
  CU(cuCtxSynchronize());
}

// ------------------------------------------------------------------- timing
static void time_and_report(const Opt& o, const std::function<void()>& body,
                            double roof_bytes) {
  for (int i = 0; i < 5; ++i) body();
  CU(cuCtxSynchronize());
  CUevent a, b;
  CU(cuEventCreate(&a, CU_EVENT_DEFAULT));
  CU(cuEventCreate(&b, CU_EVENT_DEFAULT));
  std::vector<double> us;
  us.reserve(o.reps);
  for (int i = 0; i < o.reps; ++i) {
    CU(cuEventRecord(a, 0));
    body();
    CU(cuEventRecord(b, 0));
    CU(cuEventSynchronize(b));
    float ms = 0.f;
    CU(cuEventElapsedTime(&ms, a, b));
    us.push_back((double)ms * 1000.0);
  }
  std::sort(us.begin(), us.end());
  double med = us[us.size() / 2];
  g_median_us = med;
  g_roof_bytes = roof_bytes;
  std::printf("  TIME         median %10.2f us   (min %.2f)  roofline %.2f MB"
              "   implied %.1f GB/s\n",
              med, us.front(), roof_bytes / 1e6, roof_bytes / (med * 1e-6) / 1e9);
  CU(cuEventDestroy(a));
  CU(cuEventDestroy(b));
}

// ======================================================================
// per-kernel drivers
// ======================================================================

// ---- shared input builders -------------------------------------------
struct Line {                 // the KDA line blob: B lines, shuffled indices
  std::vector<uint8_t> h;     // B * LINE_BYTES
  std::vector<int> index;     // [B]
  CUdeviceptr d = 0, dindex = 0;
};
static Line make_lines(int B, Rng& r) {
  Line L;
  L.h.assign((size_t)B * LINE_BYTES, 0);
  for (int i = 0; i < B; ++i) {
    uint8_t* p = L.h.data() + (size_t)i * LINE_BYTES;
    float* rec = (float*)p;
    for (long long j = 0; j < REC_BYTES / 4; ++j) rec[j] = (float)(0.1 * r.normal());
    bf16* win = (bf16*)(p + REC_BYTES);
    for (long long j = 0; j < 3 * WIN_BYTES / 2; ++j) win[j] = f2b((float)r.normal());
  }
  L.index = shuffled_iota(B, r);
  L.d = dput(L.h.data(), L.h.size());
  L.dindex = dput(L.index.data(), sizeof(int) * B);
  return L;
}

// ---- K1a attnres_rms --------------------------------------------------
static void run_attnres_rms(const Opt& o, CUmodule mod) {
  const int B = o.B;
  Rng r(o.seed);
  std::vector<bf16> prefix((size_t)B * H), blocks((size_t)B * NB_MAX * H),
      gamma(H);
  std::vector<float> sw(H);
  fill_bf16(prefix, r, 0, 1.0);
  fill_bf16(blocks, r, 0, 1.0);
  fill_f32(sw, r, 0, 0.012);
  fill_bf16(gamma, r, 1.0, 0.1);

  std::vector<bf16> normed_ref((size_t)B * H), blocks_ref = blocks;
  ref_attnres_rms(prefix.data(), blocks_ref.data(), sw.data(), gamma.data(),
                  normed_ref.data(), o.nb, o.snapshot, B);

  CUdeviceptr dprefix = dput(prefix.data(), prefix.size() * 2);
  CUdeviceptr dblocks = dput(blocks.data(), blocks.size() * 2);
  CUdeviceptr dsw = dput(sw.data(), sw.size() * 4);
  CUdeviceptr dgamma = dput(gamma.data(), gamma.size() * 2);
  CUdeviceptr dnormed = dpoison(normed_ref.size() * 2);

  int nb = o.nb, snap = o.snapshot, b_ = B;
  void* args[] = {&dprefix, &dblocks, &dsw, &dgamma, &dnormed, &nb, &snap, &b_};
  Geo g = geo(o, 1, 1, 1024, 1, 1, 0);
  CUfunction f = getfn(mod, "kern_k3_attnres_rms");
  launch(f, g, args);

  std::vector<bf16> normed(normed_ref.size()), blocks_got(blocks.size());
  dget(normed.data(), dnormed, normed.size() * 2);
  dget(blocks_got.data(), dblocks, blocks_got.size() * 2);
  cmp_bf16("normed", normed, normed_ref);
  cmp_bf16("blocks(state)", blocks_got, blocks_ref);

  double roof = (double)B * H * 2                    // prefix
              + (double)B * o.nb * H * 2             // candidate blocks
              + (double)H * 4 + (double)H * 2        // sw, gamma
              + (double)B * H * 2                    // normed
              + (o.snapshot ? (double)B * H * 2 : 0);
  time_and_report(o, [&] { CU(cuLaunchKernel(f, g.gx, g.gy, g.gz, g.bx, g.by, g.bz,
                                             g.smem, 0, args, nullptr)); }, roof);
}

// ---- K1b land_add_attnres_rms ----------------------------------------
static void run_land_add_attnres_rms(const Opt& o, CUmodule mod) {
  const int B = o.B;
  Rng r(o.seed);
  std::vector<float> partial((size_t)B * H), sw(H);
  std::vector<bf16> prefix((size_t)B * H), blocks((size_t)B * NB_MAX * H), gamma(H);
  fill_f32(partial, r, 0, 2.0);
  fill_bf16(prefix, r, 0, 1.0);
  fill_bf16(blocks, r, 0, 1.0);
  fill_f32(sw, r, 0, 0.012);
  fill_bf16(gamma, r, 1.0, 0.1);

  std::vector<bf16> p2_ref((size_t)B * H), normed_ref((size_t)B * H);
  ref_land_add_attnres_rms(partial.data(), prefix.data(), blocks.data(), sw.data(),
                           gamma.data(), p2_ref.data(), normed_ref.data(), o.nb,
                           o.snapshot, B);

  CUdeviceptr dpart = dput(partial.data(), partial.size() * 4);
  CUdeviceptr dprefix = dput(prefix.data(), prefix.size() * 2);
  CUdeviceptr dblocks = dput(blocks.data(), blocks.size() * 2);
  CUdeviceptr dsw = dput(sw.data(), sw.size() * 4);
  CUdeviceptr dgamma = dput(gamma.data(), gamma.size() * 2);
  CUdeviceptr dp2 = dpoison(p2_ref.size() * 2);
  CUdeviceptr dnormed = dpoison(normed_ref.size() * 2);

  int nb = o.nb, snap = o.snapshot, b_ = B;
  void* args[] = {&dpart, &dprefix, &dblocks, &dsw,  &dgamma,
                  &dp2,   &dnormed, &nb,      &snap, &b_};
  Geo g = geo(o, 1, 1, 1024, 1, 1, 0);
  CUfunction f = getfn(mod, "kern_k3_land_add_attnres_rms");
  launch(f, g, args);

  std::vector<bf16> p2(p2_ref.size()), normed(normed_ref.size());
  dget(p2.data(), dp2, p2.size() * 2);
  dget(normed.data(), dnormed, normed.size() * 2);
  cmp_bf16("prefix2", p2, p2_ref);
  cmp_bf16("normed", normed, normed_ref);

  double roof = (double)B * H * 4 + (double)B * H * 2 +
                (double)B * o.nb * H * 2 + (double)H * 4 + (double)H * 2 +
                (double)B * H * 2 * 2;
  time_and_report(o, [&] { CU(cuLaunchKernel(f, g.gx, g.gy, g.gz, g.bx, g.by, g.bz,
                                             g.smem, 0, args, nullptr)); }, roof);
}

// ---- K1c land_add2 ----------------------------------------------------
static void run_land_add2(const Opt& o, CUmodule mod) {
  const int B = o.B;
  Rng r(o.seed);
  std::vector<float> p1((size_t)B * H), p2((size_t)B * H);
  std::vector<bf16> prefix2((size_t)B * H);
  fill_f32(p1, r, 0, 2.0);
  fill_f32(p2, r, 0, 2.0);
  fill_bf16(prefix2, r, 0, 1.0);

  std::vector<bf16> hid_ref((size_t)B * H);
  ref_land_add2(p1.data(), p2.data(), prefix2.data(), hid_ref.data(), o.two, B);

  CUdeviceptr dp1 = dput(p1.data(), p1.size() * 4);
  CUdeviceptr dp2 = dput(p2.data(), p2.size() * 4);
  CUdeviceptr dpre = dput(prefix2.data(), prefix2.size() * 2);
  CUdeviceptr dhid = dpoison(hid_ref.size() * 2);
  int two = o.two, b_ = B;
  void* args[] = {&dp1, &dp2, &dpre, &dhid, &two, &b_};
  Geo g = geo(o, 1, 1, 1024, 1, 1, 0);
  CUfunction f = getfn(mod, "kern_k3_land_add2");
  launch(f, g, args);

  std::vector<bf16> hid(hid_ref.size());
  dget(hid.data(), dhid, hid.size() * 2);
  cmp_bf16("hidden", hid, hid_ref);

  double roof = (double)B * H * 4 * (o.two ? 2 : 1) + (double)B * H * 2 * 2;
  time_and_report(o, [&] { CU(cuLaunchKernel(f, g.gx, g.gy, g.gz, g.bx, g.by, g.bz,
                                             g.smem, 0, args, nullptr)); }, roof);
}

// ---- K2 conv_silu -----------------------------------------------------
static void run_conv_silu(const Opt& o, CUmodule mod) {
  const int B = o.B;
  Rng r(o.seed);
  std::vector<float> partial((size_t)B * KDA_FUSED), cw((size_t)3 * 4 * INNER);
  fill_f32(partial, r, 0, 2.0);
  fill_f32(cw, r, 0, 0.3);
  Line L = make_lines(B, r);

  std::vector<uint8_t> line_ref = L.h;
  std::vector<bf16> cq_ref((size_t)B * INNER), ck_ref((size_t)B * INNER),
      cv_ref((size_t)B * INNER);
  ref_conv_silu(partial.data(), cw.data(), line_ref.data(), L.index.data(),
                LINE_BYTES, cq_ref.data(), ck_ref.data(), cv_ref.data(), B);

  CUdeviceptr dpart = dput(partial.data(), partial.size() * 4);
  CUdeviceptr dcw = dput(cw.data(), cw.size() * 4);
  CUdeviceptr dq = dpoison(cq_ref.size() * 2);
  CUdeviceptr dk = dpoison(ck_ref.size() * 2);
  CUdeviceptr dv = dpoison(cv_ref.size() * 2);
  long long lb = LINE_BYTES;
  int b_ = B, span = o.span, at0 = span ? 1 : 0;
  if (at0 + span > B) { fprintf(stderr, "--span %d needs B >= %d\n", span, at0 + span); exit(1); }
  CUdeviceptr dat = dput(&at0, 4);
  void* args[] = {&dpart, &dcw, &L.d, &L.dindex, &lb, &dq, &dk, &dv, &b_, &dat, &span};
  // document default: grid (B, 3, 24), block 128 -> 4 columns per thread.
  Geo g = geo(o, 3, 24, 128, 1, 1, 0);
  CUfunction f = getfn(mod, "kern_k3_conv_silu");
  launch(f, g, args);

  std::vector<bf16> cq(cq_ref.size()), ck(ck_ref.size()), cv(cv_ref.size());
  dget(cq.data(), dq, cq.size() * 2);
  dget(ck.data(), dk, ck.size() * 2);
  dget(cv.data(), dv, cv.size() * 2);
  // The span's rows are K9's: whatever the kernel left there is right.
  for (int b = at0; b < at0 + span; ++b) {
    for (size_t i = (size_t)b * INNER; i < (size_t)(b + 1) * INNER; ++i) {
      cq_ref[i] = cq[i];
      ck_ref[i] = ck[i];
      cv_ref[i] = cv[i];
    }
  }
  cmp_bf16("conv_q", cq, cq_ref);
  cmp_bf16("conv_k", ck, ck_ref);
  cmp_bf16("conv_v", cv, cv_ref);

  // windows only: rec is untouched by K2, so compare the window bytes.
  std::vector<uint8_t> line_got(L.h.size());
  dget(line_got.data(), L.d, line_got.size());
  for (int b = at0; b < at0 + span; ++b) {
    size_t off = (size_t)L.index[b] * LINE_BYTES;
    std::copy(line_got.begin() + off, line_got.begin() + off + LINE_BYTES, line_ref.begin() + off);
  }
  {
    size_t nwin = (size_t)B * 3 * 3 * INNER;
    auto at = [&](std::vector<uint8_t>& v, size_t i) -> double {
      size_t per = (size_t)3 * 3 * INNER;
      int b = (int)(i / per);
      size_t k = i % per;
      int s = (int)(k / (3 * INNER));
      size_t off = k % (3 * INNER);
      bf16* w = (bf16*)(v.data() + (size_t)b * LINE_BYTES + REC_BYTES +
                        (size_t)s * WIN_BYTES);
      return (double)b2f(w[off]);
    };
    // NOTE: line index b here is the *physical* line, addressed directly, so
    // both sides walk the same bytes.
    cmp_gen("win(state)", nwin, [&](size_t i) { return at(line_got, i); },
            [&](size_t i) { return at(line_ref, i); });
  }

  double roof = (double)B * 3 * INNER * 4          // partial bands 0..2
              + (double)3 * 4 * INNER * 4          // cw
              + (double)B * 3 * 3 * INNER * 2 * 2  // windows read + write
              + (double)B * 3 * INNER * 2;         // conv_q/k/v
  time_and_report(o, [&] { CU(cuLaunchKernel(f, g.gx, g.gy, g.gz, g.bx, g.by, g.bz,
                                             g.smem, 0, args, nullptr)); }, roof);
}

// ---- K3 kda_core ------------------------------------------------------
static void run_kda_core(const Opt& o, CUmodule mod) {
  const int B = o.B;
  Rng r(o.seed);
  std::vector<bf16> cq((size_t)B * INNER), ck((size_t)B * INNER),
      cv((size_t)B * INNER), w_f_b((size_t)INNER * 128);
  std::vector<float> wsm((size_t)B * WSM), gate((size_t)B * KDA_FUSED),
      dt_bias(INNER), a_log(HEADS), gamma_o(128);
  fill_bf16(cq, r, 0, 1.0);
  fill_bf16(ck, r, 0, 1.0);
  fill_bf16(cv, r, 0, 1.0);
  fill_f32(wsm, r, 0, 2.0);
  fill_f32(gate, r, 0, 2.0);
  fill_bf16(w_f_b, r, 0, 0.044);
  fill_f32(dt_bias, r, 0, 1.0);
  fill_f32(a_log, r, 0, 0.5);
  fill_f32(gamma_o, r, 1.0, 0.1);
  Line L = make_lines(B, r);

  std::vector<uint8_t> line_ref = L.h;
  std::vector<bf16> out_ref((size_t)B * INNER);
  ref_kda_core(cq.data(), ck.data(), cv.data(), wsm.data(), gate.data(),
               w_f_b.data(), dt_bias.data(), a_log.data(), gamma_o.data(),
               line_ref.data(), L.index.data(), LINE_BYTES, out_ref.data(), B);

  CUdeviceptr dcq = dput(cq.data(), cq.size() * 2);
  CUdeviceptr dck = dput(ck.data(), ck.size() * 2);
  CUdeviceptr dcv = dput(cv.data(), cv.size() * 2);
  CUdeviceptr dwsm = dput(wsm.data(), wsm.size() * 4);
  CUdeviceptr dgate = dput(gate.data(), gate.size() * 4);
  CUdeviceptr dwfb = dput(w_f_b.data(), w_f_b.size() * 2);
  CUdeviceptr ddt = dput(dt_bias.data(), dt_bias.size() * 4);
  CUdeviceptr dal = dput(a_log.data(), a_log.size() * 4);
  CUdeviceptr dgo = dput(gamma_o.data(), gamma_o.size() * 4);
  CUdeviceptr dout = dpoison(out_ref.size() * 2);
  long long lb = LINE_BYTES;
  int b_ = B, span = o.span, at0 = span ? 1 : 0;
  if (at0 + span > B) { fprintf(stderr, "--span %d needs B >= %d\n", span, at0 + span); exit(1); }
  CUdeviceptr dat = dput(&at0, 4);
  void* args[] = {&dcq, &dck, &dcv, &dwsm,     &dgate, &dwfb, &ddt,
                  &dal, &dgo, &L.d, &L.dindex, &lb,    &dout, &b_, &dat, &span};
  Geo g = geo(o, HEADS, 1, 128, 1, 1, 0);
  CUfunction f = getfn(mod, "kern_k3_kda_core");
  launch(f, g, args);

  std::vector<bf16> out(out_ref.size());
  dget(out.data(), dout, out.size() * 2);
  std::vector<uint8_t> line_got(L.h.size());
  dget(line_got.data(), L.d, line_got.size());
  // The span's rows are K8 / K11's: whatever the kernel left there is right.
  for (int b = at0; b < at0 + span; ++b) {
    for (size_t i = (size_t)b * INNER; i < (size_t)(b + 1) * INNER; ++i) out_ref[i] = out[i];
    size_t off = (size_t)L.index[b] * LINE_BYTES;
    std::copy(line_got.begin() + off, line_got.begin() + off + LINE_BYTES, line_ref.begin() + off);
  }
  cmp_bf16("out", out, out_ref);
  {
    size_t nrec = (size_t)B * (REC_BYTES / 4);
    auto at = [&](std::vector<uint8_t>& v, size_t i) -> double {
      size_t per = (size_t)REC_BYTES / 4;
      int b = (int)(i / per);
      float* rec = (float*)(v.data() + (size_t)b * LINE_BYTES);
      return (double)rec[i % per];
    };
    cmp_gen("rec(state)", nrec, [&](size_t i) { return at(line_got, i); },
            [&](size_t i) { return at(line_ref, i); });
  }

  double roof = (double)B * 3 * INNER * 2        // conv_q/k/v
              + (double)B * WSM * 4              // wsm partial
              + (double)B * INNER * 4            // gate band 3
              + (double)INNER * 128 * 2          // w_f_b
              + (double)INNER * 4 + HEADS * 4 + 128 * 4
              + (double)B * REC_BYTES * 2        // rec read + write
              + (double)B * INNER * 2;           // out
  time_and_report(o, [&] { CU(cuLaunchKernel(f, g.gx, g.gy, g.gz, g.bx, g.by, g.bz,
                                             g.smem, 0, args, nullptr)); }, roof);
}

// ---- K4 mla_prep ------------------------------------------------------
static void run_mla_prep(const Opt& o, CUmodule mod) {
  const int B = o.B;
  Rng r(o.seed);
  const long long page_stride = (long long)o.nmla * PAGE * LATENT_ROW;
  const long long layer_off = (long long)o.layer * PAGE * LATENT_ROW;
  const int npages = std::max(B, 8);

  std::vector<float> partial((size_t)B * MLA_FUSED);
  std::vector<bf16> gq(Q_LORA), gkv(KV_LORA);
  fill_f32(partial, r, 0, 2.0);
  fill_bf16(gq, r, 1.0, 0.1);
  fill_bf16(gkv, r, 1.0, 0.1);

  std::vector<int> perm = shuffled_iota(npages, r);
  std::vector<int64_t> slot(B);
  for (int b = 0; b < B; ++b)
    slot[b] = (int64_t)perm[b] * PAGE + (int64_t)((b * 13 + 5) % PAGE);

  std::vector<bf16> slab((size_t)npages * page_stride);
  fill_bf16(slab, r, 0, 1.0);
  std::vector<bf16> slab_ref = slab;
  std::vector<bf16> qn_ref((size_t)B * Q_LORA), gate_ref((size_t)B * INNER);
  ref_mla_prep(partial.data(), gq.data(), gkv.data(), slot.data(),
               slab_ref.data(), layer_off, page_stride, qn_ref.data(),
               gate_ref.data(), B);

  CUdeviceptr dpart = dput(partial.data(), partial.size() * 4);
  CUdeviceptr dgq = dput(gq.data(), gq.size() * 2);
  CUdeviceptr dgkv = dput(gkv.data(), gkv.size() * 2);
  CUdeviceptr dslot = dput(slot.data(), slot.size() * 8);
  CUdeviceptr dslab = dput(slab.data(), slab.size() * 2);
  CUdeviceptr dqn = dpoison(qn_ref.size() * 2);
  CUdeviceptr dgate = dpoison(gate_ref.size() * 2);
  long long lo = layer_off, ps = page_stride;
  int b_ = B;
  void* args[] = {&dpart, &dgq, &dgkv, &dslot, &dslab,
                  &lo,    &ps,  &dqn,  &dgate, &b_};
  Geo g = geo(o, 1, 1, 1024, 1, 1, 0);
  CUfunction f = getfn(mod, "kern_k3_mla_prep");
  launch(f, g, args);

  std::vector<bf16> qn(qn_ref.size()), gate(gate_ref.size()), slab_got(slab.size());
  dget(qn.data(), dqn, qn.size() * 2);
  dget(gate.data(), dgate, gate.size() * 2);
  dget(slab_got.data(), dslab, slab_got.size() * 2);
  cmp_bf16("q_norm", qn, qn_ref);
  cmp_bf16("mla_gate", gate, gate_ref);
  cmp_bf16("slab(state)", slab_got, slab_ref);

  double roof = (double)B * MLA_FUSED * 4 + (double)(Q_LORA + KV_LORA) * 2 +
                (double)B * 8 + (double)B * Q_LORA * 2 + (double)B * INNER * 2 +
                (double)B * LATENT_ROW * 2;
  time_and_report(o, [&] { CU(cuLaunchKernel(f, g.gx, g.gy, g.gz, g.bx, g.by, g.bz,
                                             g.smem, 0, args, nullptr)); }, roof);
}

// ---- K5 mla_paged_attn (and the old pegainfer kernel) -----------------
static void run_mla_paged_attn(const Opt& o, CUmodule mod, bool old_abi) {
  const int B = o.B;
  const int ctx = o.ctx;
  Rng r(o.seed);
  const int max_pages = (ctx + PAGE - 1) / PAGE;
  const long long page_stride = (long long)o.nmla * PAGE * LATENT_ROW;
  const long long layer_off = (long long)o.layer * PAGE * LATENT_ROW;
  const int npages = B * max_pages;

  std::vector<float> qp((size_t)B * Q_B);
  std::vector<bf16> wkv((size_t)HEADS * 256 * KV_LORA), gate((size_t)B * INNER);
  fill_f32(qp, r, 0, 2.0);
  fill_bf16(wkv, r, 0, 0.025);
  fill_bf16(gate, r, 0, 1.0);

  std::vector<bf16> slab((size_t)npages * page_stride);
  fill_bf16(slab, r, 0, 1.0);

  // block table: every logical page maps to a distinct physical page, in a
  // shuffled order, so a bug in the page walk shows up.
  std::vector<int> pool = shuffled_iota(npages, r);
  std::vector<int> table((size_t)B * max_pages);
  for (int b = 0; b < B; ++b)
    for (int p = 0; p < max_pages; ++p)
      table[(size_t)b * max_pages + p] = pool[(size_t)b * max_pages + p];

  std::vector<int> seq(B);
  seq[0] = ctx;
  for (int b = 1; b < B; ++b) seq[b] = 1 + (int)r.below((uint32_t)ctx);

  std::vector<bf16> scale(1, f2b(1.0f / std::sqrt(192.0f)));

  std::vector<bf16> gated_ref((size_t)B * INNER);
  ref_mla_paged_attn(qp.data(), wkv.data(), slab.data() + layer_off, table.data(),
                     max_pages, page_stride, seq.data(), scale.data(),
                     gate.data(), gated_ref.data(), nullptr, B);

  CUdeviceptr dwkv = dput(wkv.data(), wkv.size() * 2);
  CUdeviceptr dslab = dput(slab.data(), slab.size() * 2);
  CUdeviceptr dcache = dslab + (CUdeviceptr)(layer_off * 2);
  CUdeviceptr dtab = dput(table.data(), table.size() * 4);
  CUdeviceptr dseq = dput(seq.data(), seq.size() * 4);
  CUdeviceptr dsc = dput(scale.data(), 2);
  CUdeviceptr dgate = dput(gate.data(), gate.size() * 2);
  CUdeviceptr dout = dpoison((size_t)B * INNER * 2);
  int mp = max_pages, b_ = B;
  long long ps = page_stride;

  std::vector<bf16> got((size_t)B * INNER);
  std::vector<bf16> qb;
  CUdeviceptr dq = 0;
  std::vector<void*> args;
  if (!old_abi) {
    dq = dput(qp.data(), qp.size() * 4);
    args = {&dq,  &dwkv, &dcache, &dtab, &mp, &ps,
            &dseq, &dsc, &dgate,  &dout, &b_};
  } else {
    // old ABI: bf16 q, no gate, no B (heads come from gridDim.y).
    qb.resize(qp.size());
    for (size_t i = 0; i < qp.size(); ++i) qb[i] = f2b(qp[i]);
    dq = dput(qb.data(), qb.size() * 2);
    args = {&dq, &dwkv, &dcache, &dtab, &mp, &ps, &dseq, &dsc, &dout};
  }
  Geo g = geo(o, HEADS, 1, 128, 1, 1, 0);
  CUfunction f = getfn(mod, "kern_k3_mla_paged_attn");
  launch(f, g, args.data());
  dget(got.data(), dout, got.size() * 2);
  if (old_abi) {
    // apply the gate on the CPU so the comparison is against the same K5 ref.
    for (size_t i = 0; i < got.size(); ++i)
      got[i] = bmul(got[i],
                    f2b((float)(1.0 / (1.0 + std::exp(-(double)b2f(gate[i]))))));
  }
  std::function<void()> body = [&] {
    CU(cuLaunchKernel(f, g.gx, g.gy, g.gz, g.bx, g.by, g.bz, g.smem, 0,
                      args.data(), nullptr));
  };
  cmp_bf16("gated", got, gated_ref);

  double kvtok = 0;
  for (int b = 0; b < B; ++b) kvtok += seq[b];
  double roof = (double)B * Q_B * (old_abi ? 2.0 : 4.0)   // q
              + (double)HEADS * 256 * KV_LORA * 2          // w_kv_b
              + kvtok * LATENT_ROW * 2                     // KV rows, once
              + (double)B * max_pages * 4 + (double)B * 4 + 2
              + (old_abi ? 0.0 : (double)B * INNER * 2)    // mla_gate
              + (double)B * INNER * 2;                     // out
  std::printf("  shape        ctx=%d max_pages=%d npages=%d seq_lens[0]=%d kv_tokens=%.0f\n",
              ctx, max_pages, npages, seq[0], kvtok);
  time_and_report(o, body, roof);
}

// ---- K6 router_topk ---------------------------------------------------
static void run_router_topk(const Opt& o, CUmodule mod) {
  const int B = o.B;
  Rng r(o.seed);
  std::vector<float> S((size_t)B * EXPERTS), bias(EXPERTS);
  fill_f32(S, r, 0, 2.0);
  fill_f32(bias, r, 0, 0.1);
  // Force an exact tie on experts 40/41 that is guaranteed to land in the
  // top-16, so the documented "tie takes the smaller e" rule is exercised.
  bias[40] = bias[41] = 0.5f;
  for (int b = 0; b < B; ++b) {
    S[(size_t)b * EXPERTS + 40] = 6.0f;
    S[(size_t)b * EXPERTS + 41] = 6.0f;
  }
  std::vector<bf16> rs(1, f2b(2.5f));

  std::vector<int> idx_ref((size_t)B * TOPK);
  std::vector<float> wts_ref((size_t)B * TOPK);
  ref_router_topk(S.data(), bias.data(), rs.data(), idx_ref.data(),
                  wts_ref.data(), B);

  CUdeviceptr dS = dput(S.data(), S.size() * 4);
  CUdeviceptr dbias = dput(bias.data(), bias.size() * 4);
  CUdeviceptr drs = dput(rs.data(), 2);
  CUdeviceptr didx = dpoison(idx_ref.size() * 4);
  CUdeviceptr dwts = dpoison(wts_ref.size() * 4);
  int b_ = B;
  void* args[] = {&dS, &dbias, &drs, &didx, &dwts, &b_};
  Geo g = geo(o, 1, 1, 256, 1, 1, 0);
  CUfunction f = getfn(mod, "kern_k3_router_topk");
  launch(f, g, args);

  std::vector<int> idx(idx_ref.size());
  std::vector<float> wts(wts_ref.size());
  dget(idx.data(), didx, idx.size() * 4);
  dget(wts.data(), dwts, wts.size() * 4);
  cmp_exact("idx", idx, idx_ref);
  cmp_f32("wts", wts, wts_ref);

  double roof = (double)B * EXPERTS * 4 + EXPERTS * 4 + 2 +
                (double)B * TOPK * 8;
  time_and_report(o, [&] { CU(cuLaunchKernel(f, g.gx, g.gy, g.gz, g.bx, g.by, g.bz,
                                             g.smem, 0, args, nullptr)); }, roof);
}

// ---- K6 argmax_f32 ----------------------------------------------------
static void run_argmax(const Opt& o, CUmodule mod) {
  const int B = o.B;
  const int n = V_VOCAB;
  const int parts = (o.gy > 0) ? o.gy : 64;
  Rng r(o.seed);
  std::vector<float> logits((size_t)B * n);
  fill_f32(logits, r, 0, 2.0);
  // Force a tie for the maximum at two indices per row: the smaller index wins.
  std::vector<int> tie_lo(B);
  for (int b = 0; b < B; ++b) {
    int i1 = (int)r.below((uint32_t)n / 2);
    int i2 = i1 + 1 + (int)r.below((uint32_t)n / 2 - 1);
    logits[(size_t)b * n + i1] = 40.0f;
    logits[(size_t)b * n + i2] = 40.0f;
    tie_lo[b] = i1;
  }

  std::vector<int64_t> out_ref(B);
  ref_argmax_f32(logits.data(), out_ref.data(), n, B);
  for (int b = 0; b < B; ++b) {
    if (out_ref[b] != tie_lo[b]) {
      std::printf("  internal: tie injection failed on row %d\n", b);
      g_fail = true;
    }
  }

  CUdeviceptr dlog = dput(logits.data(), logits.size() * 4);
  CUdeviceptr dpmax = dpoison((size_t)B * parts * 4);
  CUdeviceptr dpidx = dpoison((size_t)B * parts * 4);
  CUdeviceptr dout = dpoison((size_t)B * 8);
  int n_ = n, parts_ = parts;
  void* a1[] = {&dlog, &dpmax, &dpidx, &n_};
  void* a2[] = {&dpmax, &dpidx, &dout, &parts_};
  Geo g1 = geo(o, (unsigned)parts, 1, 1024, 1, 1, 0);
  Opt o2 = o;
  o2.gy = 1; o2.gz = 1; o2.bx = 64; o2.by = 1; o2.bz = 1; o2.smem = 0;
  Geo g2 = geo(o2, 1, 1, 64, 1, 1, 0);
  CUfunction f1 = getfn(mod, "kern_k3_argmax_f32_partial");
  CUfunction f2 = getfn(mod, "kern_k3_argmax_f32_final");
  launch(f1, g1, a1);
  launch(f2, g2, a2);

  std::vector<float> pmax((size_t)B * parts);
  std::vector<int> pidx((size_t)B * parts);
  std::vector<int64_t> out(B);
  dget(pmax.data(), dpmax, pmax.size() * 4);
  dget(pidx.data(), dpidx, pidx.size() * 4);
  dget(out.data(), dout, out.size() * 8);
  cmp_exact("out", out, out_ref);

  // The doc fixes grid (B,64) but not which elements a part owns, so the
  // partials are validated by invariant rather than against a fixed split:
  // every (pmax,pidx) must be a real (value,index) pair of the row, and the
  // fold over parts must reproduce the row argmax with the smallest index.
  size_t bad = 0;
  for (int b = 0; b < B; ++b) {
    int best = -1;
    for (int p = 0; p < parts; ++p) {
      int i = pidx[(size_t)b * parts + p];
      float v = pmax[(size_t)b * parts + p];
      if (i < 0 || i >= n) { ++bad; continue; }
      if (logits[(size_t)b * n + i] != v) { ++bad; continue; }
      if (best < 0 || v > logits[(size_t)b * n + best] ||
          (v == logits[(size_t)b * n + best] && i < best))
        best = i;
    }
    if (best != (int)out_ref[b]) ++bad;
  }
  if (bad) g_fail = true;
  std::printf("  %-12s n=%-11d invariant (value/index pair, fold == argmax)  %s\n",
              "pmax/pidx", B * parts, bad ? "FAIL" : "PASS");

  double roof = (double)B * n * 4 + (double)B * parts * 8 * 2 + (double)B * 8;
  time_and_report(o,
                  [&] {
                    CU(cuLaunchKernel(f1, g1.gx, g1.gy, g1.gz, g1.bx, g1.by,
                                      g1.bz, g1.smem, 0, a1, nullptr));
                    CU(cuLaunchKernel(f2, g2.gx, g2.gy, g2.gz, g2.bx, g2.by,
                                      g2.bz, g2.smem, 0, a2, nullptr));
                  },
                  roof);
}

// ---- K6 rms -----------------------------------------------------------
static void run_rms(const Opt& o, CUmodule mod) {
  const int B = o.B, h = LATENT;
  Rng r(o.seed);
  std::vector<bf16> x((size_t)B * h), gamma(h), o_ref((size_t)B * h);
  fill_bf16(x, r, 0, 1.0);
  fill_bf16(gamma, r, 1.0, 0.1);
  ref_rms(x.data(), gamma.data(), o_ref.data(), h, B);

  CUdeviceptr dx = dput(x.data(), x.size() * 2);
  CUdeviceptr dg = dput(gamma.data(), gamma.size() * 2);
  CUdeviceptr dd = dpoison(o_ref.size() * 2);
  int h_ = h, b_ = B;
  void* args[] = {&dx, &dg, &dd, &h_, &b_};
  Geo g = geo(o, 1, 1, 1024, 1, 1, 0);
  CUfunction f = getfn(mod, "kern_k3_rms");
  launch(f, g, args);
  std::vector<bf16> got(o_ref.size());
  dget(got.data(), dd, got.size() * 2);
  cmp_bf16("o", got, o_ref);
  double roof = (double)B * h * 2 * 2 + (double)h * 2;
  time_and_report(o, [&] { CU(cuLaunchKernel(f, g.gx, g.gy, g.gz, g.bx, g.by, g.bz,
                                             g.smem, 0, args, nullptr)); }, roof);
}

// ---- K7 land ----------------------------------------------------------
static void run_land(const Opt& o, CUmodule mod) {
  const int B = o.B, n = LATENT, off = 128, ldc = 4096;
  Rng r(o.seed);
  std::vector<float> p((size_t)B * ldc);
  fill_f32(p, r, 0, 2.0);
  std::vector<bf16> o_ref((size_t)B * n);
  ref_land(p.data(), o_ref.data(), n, off, ldc, B);

  CUdeviceptr dp = dput(p.data(), p.size() * 4);
  CUdeviceptr dd = dpoison(o_ref.size() * 2);
  int n_ = n, off_ = off, ldc_ = ldc, b_ = B;
  void* args[] = {&dp, &dd, &n_, &off_, &ldc_, &b_};
  Geo g = geo(o, (unsigned)((n + 1023) / 1024), 1, 1024, 1, 1, 0);
  CUfunction f = getfn(mod, "kern_k3_land");
  launch(f, g, args);
  std::vector<bf16> got(o_ref.size());
  dget(got.data(), dd, got.size() * 2);
  cmp_bf16("o", got, o_ref);
  double roof = (double)B * n * 4 + (double)B * n * 2;
  std::printf("  shape        n=%d off=%d ldc=%d\n", n, off, ldc);
  time_and_report(o, [&] { CU(cuLaunchKernel(f, g.gx, g.gy, g.gz, g.bx, g.by, g.bz,
                                             g.smem, 0, args, nullptr)); }, roof);
}

// ---- K7 land_situ -----------------------------------------------------
static void run_land_situ(const Opt& o, CUmodule mod) {
  const int B = o.B, n = INTER;
  Rng r(o.seed);
  std::vector<float> p((size_t)B * 2 * n);
  fill_f32(p, r, 0, 2.0);
  std::vector<bf16> a_ref((size_t)B * n);
  ref_land_situ(p.data(), a_ref.data(), n, B);

  CUdeviceptr dp = dput(p.data(), p.size() * 4);
  CUdeviceptr dd = dpoison(a_ref.size() * 2);
  int n_ = n, b_ = B;
  void* args[] = {&dp, &dd, &n_, &b_};
  Geo g = geo(o, (unsigned)((n + 1023) / 1024), 1, 1024, 1, 1, 0);
  CUfunction f = getfn(mod, "kern_k3_land_situ");
  launch(f, g, args);
  std::vector<bf16> got(a_ref.size());
  dget(got.data(), dd, got.size() * 2);
  cmp_bf16("act", got, a_ref);
  double roof = (double)B * 2 * n * 4 + (double)B * n * 2;
  std::printf("  shape        n=%d\n", n);
  time_and_report(o, [&] { CU(cuLaunchKernel(f, g.gx, g.gy, g.gz, g.bx, g.by, g.bz,
                                             g.smem, 0, args, nullptr)); }, roof);
}

// The span's first batch row in the K9 / K10 / K11 runs: the rows before it
// are decode rows the kernels must leave alone.
static const int SPAN_AT = 3;

// ---- K9 span_gather (B = span) ------------------------------------------
static void run_span_gather(const Opt& o, CUmodule mod) {
  const int S = o.B, N = SPAN_AT + S;
  Rng r(o.seed);
  std::vector<float> partial((size_t)N * KDA_FUSED), cw((size_t)3 * 4 * INNER), wsm((size_t)N * WSM);
  fill_f32(partial, r, 0, 2.0);
  fill_f32(cw, r, 0, 0.3);
  fill_f32(wsm, r, 0, 2.0);
  Line L = make_lines(N, r);
  int at = SPAN_AT;
  CUdeviceptr dat = dput(&at, 4);

  std::vector<uint8_t> line_ref = L.h;
  std::vector<bf16> cq_ref((size_t)S * INNER), ck_ref((size_t)S * INNER), cv_ref((size_t)S * INNER),
      beta_ref((size_t)HEADS * S), flow_ref((size_t)S * 128);
  ref_span_gather(partial.data(), cw.data(), line_ref.data(), L.index.data(), LINE_BYTES, wsm.data(),
                  cq_ref.data(), ck_ref.data(), cv_ref.data(), beta_ref.data(), flow_ref.data(), &at, S);

  CUdeviceptr dpart = dput(partial.data(), partial.size() * 4);
  CUdeviceptr dcw = dput(cw.data(), cw.size() * 4);
  CUdeviceptr dwsm = dput(wsm.data(), wsm.size() * 4);
  CUdeviceptr dq = dpoison(cq_ref.size() * 2), dk = dpoison(ck_ref.size() * 2), dv = dpoison(cv_ref.size() * 2);
  CUdeviceptr dbeta = dpoison(beta_ref.size() * 2), dflow = dpoison(flow_ref.size() * 2);
  long long lb = LINE_BYTES;
  int span = S;
  void* args[] = {&dpart, &dcw, &L.d, &L.dindex, &lb, &dwsm, &dq, &dk, &dv, &dbeta, &dflow, &dat, &span};
  // document default: grid (INNER/512, 4, ceil(span/8)), block 128; grid.x is not B here.
  Geo g = geo_at(o, (unsigned)(INNER / 512), 4, (unsigned)((S + 7) / 8), 128, 1, 1, 0);
  CUfunction f = getfn(mod, "kern_k3_span_gather");
  launch(f, g, args);

  std::vector<bf16> cq(cq_ref.size()), ck(ck_ref.size()), cv(cv_ref.size()), beta(beta_ref.size()),
      flow(flow_ref.size());
  dget(cq.data(), dq, cq.size() * 2);
  dget(ck.data(), dk, ck.size() * 2);
  dget(cv.data(), dv, cv.size() * 2);
  dget(beta.data(), dbeta, beta.size() * 2);
  dget(flow.data(), dflow, flow.size() * 2);
  cmp_bf16("conv_q", cq, cq_ref);
  cmp_bf16("conv_k", ck, ck_ref);
  cmp_bf16("conv_v", cv, cv_ref);
  cmp_bf16("span_beta", beta, beta_ref);
  cmp_bf16("span_flow", flow, flow_ref);
  std::vector<uint8_t> line_got(L.h.size());
  dget(line_got.data(), L.d, line_got.size());
  {
    // the windows of row at's line (the only line the kernel touches)
    size_t nwin = (size_t)3 * 3 * INNER;
    size_t base = (size_t)L.index[at] * LINE_BYTES + REC_BYTES;
    auto win_at = [&](std::vector<uint8_t>& v, size_t i) -> double {
      int s = (int)(i / (3 * INNER));
      size_t off = i % (3 * INNER);
      return (double)b2f(((bf16*)(v.data() + base + (size_t)s * WIN_BYTES))[off]);
    };
    cmp_gen("win(state)", nwin, [&](size_t i) { return win_at(line_got, i); },
            [&](size_t i) { return win_at(line_ref, i); });
    cmp_gen("other lines untouched", L.h.size() - LINE_BYTES,
            [&](size_t i) { size_t j = i < (size_t)L.index[at] * LINE_BYTES ? i : i + LINE_BYTES; return (double)line_got[j]; },
            [&](size_t i) { size_t j = i < (size_t)L.index[at] * LINE_BYTES ? i : i + LINE_BYTES; return (double)line_ref[j]; });
  }
  double roof = (double)S * 3 * INNER * 4 + (double)3 * 4 * INNER * 4 + (double)3 * 3 * INNER * 2 * 2
              + (double)S * 3 * INNER * 2 + (double)S * WSM * 4 + (double)S * (HEADS + 128) * 2;
  time_and_report(o, [&] { CU(cuLaunchKernel(f, g.gx, g.gy, g.gz, g.bx, g.by, g.bz, g.smem, 0, args, nullptr)); }, roof);
}

// ---- K10 span_state ------------------------------------------------------
static void run_span_state(const Opt& o, CUmodule mod) {
  Rng r(o.seed);
  Line L = make_lines(SPAN_AT + o.B, r);
  int at = SPAN_AT;
  CUdeviceptr dat = dput(&at, 4);
  std::vector<float> buf_ref((size_t)REC_BYTES / 4), buf_in((size_t)REC_BYTES / 4);
  fill_f32(buf_in, r, 0, 0.1);
  std::vector<uint8_t> line_ref = L.h;
  long long lb = LINE_BYTES;
  CUdeviceptr dbuf = dpoison(REC_BYTES);
  CUfunction f = getfn(mod, "kern_k3_span_state");
  Geo g = geo_at(o, (unsigned)HEADS, 32, 1, 128, 1, 1, 0);

  int to_line = 0;
  void* args[] = {&L.d, &L.dindex, &lb, &dat, &dbuf, &to_line};
  ref_span_state(line_ref.data(), L.index.data(), LINE_BYTES, &at, buf_ref.data(), 0);
  launch(f, g, args);
  std::vector<float> buf(buf_ref.size());
  dget(buf.data(), dbuf, buf.size() * 4);
  cmp_exact("buf = rec", buf, buf_ref);

  to_line = 1;
  CU(cuMemcpyHtoD(dbuf, buf_in.data(), REC_BYTES));
  ref_span_state(line_ref.data(), L.index.data(), LINE_BYTES, &at, buf_in.data(), 1);
  launch(f, g, args);
  std::vector<uint8_t> line_got(L.h.size());
  dget(line_got.data(), L.d, line_got.size());
  cmp_exact("rec = buf (whole line blob)", line_got, line_ref);
  to_line = 0;
  time_and_report(o, [&] { CU(cuLaunchKernel(f, g.gx, g.gy, g.gz, g.bx, g.by, g.bz, g.smem, 0, args, nullptr)); },
                  2.0 * REC_BYTES);
}

// ---- K11 kda_out_gate (B = span) ----------------------------------------
static void run_kda_out_gate(const Opt& o, CUmodule mod) {
  const int S = o.B, N = SPAN_AT + S;
  Rng r(o.seed);
  std::vector<bf16> attn((size_t)S * INNER), gated((size_t)N * INNER);
  std::vector<float> gate((size_t)N * KDA_FUSED), gamma_o(128);
  fill_bf16(attn, r, 0, 1.0);
  fill_bf16(gated, r, 0, 1.0);
  fill_f32(gate, r, 0, 2.0);
  fill_f32(gamma_o, r, 1.0, 0.1);
  int at = SPAN_AT;
  std::vector<bf16> out_ref = gated;  // rows before the span stay as they were
  ref_kda_out_gate(attn.data(), gate.data(), gamma_o.data(), out_ref.data(), &at, S);

  CUdeviceptr da = dput(attn.data(), attn.size() * 2);
  CUdeviceptr dg = dput(gated.data(), gated.size() * 2);
  CUdeviceptr dgate = dput(gate.data(), gate.size() * 4);
  CUdeviceptr dgo = dput(gamma_o.data(), gamma_o.size() * 4);
  CUdeviceptr dat = dput(&at, 4);
  int span = S;
  void* args[] = {&da, &dgate, &dgo, &dg, &dat, &span};
  Geo g = geo(o, HEADS, 1, 128, 1, 1, 0);
  CUfunction f = getfn(mod, "kern_k3_kda_out_gate");
  launch(f, g, args);
  std::vector<bf16> out(out_ref.size());
  dget(out.data(), dg, out.size() * 2);
  cmp_bf16("gated", out, out_ref);
  double roof = (double)S * INNER * 2 * 2 + (double)S * INNER * 4 + 128 * 4;
  time_and_report(o, [&] { CU(cuLaunchKernel(f, g.gx, g.gy, g.gz, g.bx, g.by, g.bz, g.smem, 0, args, nullptr)); }, roof);
}

// ======================================================================
static const char* kKernels[] = {
    "attnres_rms", "land_add_attnres_rms", "land_add2", "conv_silu",
    "kda_core",    "mla_prep",             "mla_paged_attn",
    "mla_paged_attn_old", "router_topk", "argmax_f32", "rms", "land",
    "land_situ",   "span_gather",          "span_state",    "kda_out_gate"};

static void usage() {
  std::printf(
      "usage: harness --kernel <name> --cubin <path> [options]\n"
      "  --B N            batch rows (1|2|8|64), default 1\n"
      "  --ctx N          K5 context length, default 2048\n"
      "  --nb N           attnres candidate blocks 0..8, default 4\n"
      "  --snapshot 0|1   K1a/K1b snapshot flag, default 1\n"
      "  --two 0|1        K1c: add p2 as well, default 1\n"
      "  --nmla N         MLA layers in the slab, default 1\n"
      "  --layer K        which MLA layer (layer_off = K*64*576), default 0\n"
      "  --reps N         timed launches, default 50\n"
      "  --seed N         rng seed, default 1234\n"
      "  --grid gx,gy,gz  override the launch grid (gx must be B)\n"
      "  --block bx,by,bz override the block\n"
      "  --smem bytes     override the dynamic shared memory\n"
      "kernels:");
  for (const char* k : kKernels) std::printf(" %s", k);
  std::printf("\n");
}

static void parse3(const char* s, int* a, int* b, int* c) {
  if (std::sscanf(s, "%d,%d,%d", a, b, c) != 3) {
    std::fprintf(stderr, "bad triple `%s`\n", s);
    std::exit(2);
  }
}

int main(int argc, char** argv) {
  Opt o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&]() -> const char* {
      if (i + 1 >= argc) { usage(); std::exit(2); }
      return argv[++i];
    };
    if (a == "--kernel") o.kernel = need();
    else if (a == "--cubin") o.cubin = need();
    else if (a == "--B") o.B = std::atoi(need());
    else if (a == "--span") o.span = std::atoi(need());
    else if (a == "--ctx") o.ctx = std::atoi(need());
    else if (a == "--nb") o.nb = std::atoi(need());
    else if (a == "--snapshot") o.snapshot = std::atoi(need());
    else if (a == "--two") o.two = std::atoi(need());
    else if (a == "--nmla") o.nmla = std::atoi(need());
    else if (a == "--layer") o.layer = std::atoi(need());
    else if (a == "--reps") o.reps = std::atoi(need());
    else if (a == "--seed") o.seed = (uint64_t)std::strtoull(need(), nullptr, 10);
    else if (a == "--grid") parse3(need(), &o.gx, &o.gy, &o.gz);
    else if (a == "--block") parse3(need(), &o.bx, &o.by, &o.bz);
    else if (a == "--smem") o.smem = std::atoll(need());
    else if (a == "-h" || a == "--help") { usage(); return 0; }
    else { std::fprintf(stderr, "unknown option `%s`\n", a.c_str()); usage(); return 2; }
  }
  if (o.kernel.empty() || o.cubin.empty()) { usage(); return 2; }
  if (o.reps < 1) o.reps = 1;
  if (o.nb < 0) o.nb = 0;
  if (o.nb > NB_MAX) o.nb = NB_MAX;
  if (o.nb == NB_MAX && o.snapshot) {
    std::printf("  note: nb == NB_MAX(8) leaves no snapshot slot; forcing --snapshot 0\n");
    o.snapshot = 0;
  }
  if (o.layer >= o.nmla) { std::fprintf(stderr, "--layer must be < --nmla\n"); return 2; }

  CU(cuInit(0));
  CUdevice dev;
  CU(cuDeviceGet(&dev, 0));
  char devname[128];
  CU(cuDeviceGetName(devname, sizeof(devname), dev));
  CUcontext ctx;
  CU(cuDevicePrimaryCtxRetain(&ctx, dev));
  CU(cuCtxSetCurrent(ctx));
  CUmodule mod;
  CUresult mr = cuModuleLoad(&mod, o.cubin.c_str());
  if (mr != CUDA_SUCCESS) {
    const char* s = nullptr;
    cuGetErrorString(mr, &s);
    std::fprintf(stderr, "cuModuleLoad(%s): %s\n", o.cubin.c_str(), s ? s : "?");
    return 2;
  }

  std::printf("== %s  B=%d  cubin=%s  device=%s  seed=%llu\n", o.kernel.c_str(),
              o.B, o.cubin.c_str(), devname, (unsigned long long)o.seed);

  const std::string& k = o.kernel;
  if (k == "attnres_rms") run_attnres_rms(o, mod);
  else if (k == "land_add_attnres_rms") run_land_add_attnres_rms(o, mod);
  else if (k == "land_add2") run_land_add2(o, mod);
  else if (k == "conv_silu") run_conv_silu(o, mod);
  else if (k == "kda_core") run_kda_core(o, mod);
  else if (k == "mla_prep") run_mla_prep(o, mod);
  else if (k == "mla_paged_attn") run_mla_paged_attn(o, mod, false);
  else if (k == "mla_paged_attn_old") run_mla_paged_attn(o, mod, true);
  else if (k == "router_topk") run_router_topk(o, mod);
  else if (k == "argmax_f32") run_argmax(o, mod);
  else if (k == "rms") run_rms(o, mod);
  else if (k == "land") run_land(o, mod);
  else if (k == "land_situ") run_land_situ(o, mod);
  else if (k == "span_gather") run_span_gather(o, mod);
  else if (k == "span_state") run_span_state(o, mod);
  else if (k == "kda_out_gate") run_kda_out_gate(o, mod);
  else { std::fprintf(stderr, "unknown kernel `%s`\n", k.c_str()); usage(); return 2; }

  // one machine-readable line per run, for run_all.sh
  std::printf("RESULT\t%s\tB=%d\tctx=%d\tnb=%d\tsnap=%d\ttwo=%d\t%s\t%.2f us\t%.1f GB/s\n",
              o.kernel.c_str(), o.B, o.ctx, o.nb, o.snapshot, o.two,
              g_fail ? "FAIL" : "PASS", g_median_us,
              g_median_us > 0 ? g_roof_bytes / (g_median_us * 1e-6) / 1e9 : 0.0);
  std::printf("== %s  %s\n\n", o.kernel.c_str(), g_fail ? "FAIL" : "PASS");
  for (CUdeviceptr p : g_allocs) cuMemFree(p);
  cuDevicePrimaryCtxRelease(dev);
  return g_fail ? 1 : 0;
}
