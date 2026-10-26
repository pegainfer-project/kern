// Drives the flashinfer_sampling cubin the way the runtime will: the driver
// API, the entries by name, the launch geometry the manifest will carry.
// Checks, over random logits with per-row temperature / top-k / top-p /
// min-p: the sampled token of every row lies in the set those filters keep
// (a CPU softmax decides the set); a second run over the same inputs gives
// the same tokens; one row's samples over many offsets follow its softmax
// (chi-square over the top tokens); and how long the chain takes per step at
// a few batch sizes. Prints the radix top-k geometry a generator has to
// write for this vocab.
//
//   nvcc -std=c++17 -O2 -I$FLASHINFER_INCLUDE -o probe tools/flashinfer-sampling/probe.cu -lcuda
//   ./probe <cubin> <entries.txt> [vocab=151936]
#include <cuda.h>
#include <cuda_runtime.h>
#include <flashinfer/sampling.cuh>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace fs = flashinfer::sampling;
constexpr uint32_t BLOCK = 1024, VEC = 4;

#define CU(x)                                                                        \
  do {                                                                               \
    CUresult r = (x);                                                                \
    if (r != CUDA_SUCCESS) {                                                         \
      const char* s;                                                                 \
      cuGetErrorString(r, &s);                                                       \
      fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, s);                \
      exit(1);                                                                       \
    }                                                                                \
  } while (0)
#define RT(x)                                                                        \
  do {                                                                               \
    cudaError_t r = (x);                                                             \
    if (r != cudaSuccess) {                                                          \
      fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(r)); \
      exit(1);                                                                       \
    }                                                                                \
  } while (0)

struct Kernel {
  CUfunction f;
  unsigned smem;
  void launch(unsigned grid, void** args, cudaStream_t s = 0) {
    CU(cuLaunchKernel(f, grid, 1, 1, BLOCK, 1, 1, smem, s, args, nullptr));
  }
};

static Kernel kernel(CUmodule m, const std::vector<std::string>& entries, const char* tag, unsigned smem) {
  for (auto& e : entries) {
    if (e.find(tag) != std::string::npos) {
      Kernel k;
      CU(cuModuleGetFunction(&k.f, m, e.c_str()));
      CU(cuFuncSetAttribute(k.f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem));
      k.smem = smem;
      return k;
    }
  }
  fprintf(stderr, "no entry matches %s\n", tag);
  exit(1);
}

/// What RadixTopKRenormProbMultiCTA derives for this vocab and GPU: the
/// constants a manifest launch carries.
struct Radix {
  uint32_t chunk, ctas, groups_max;
  unsigned smem;
};

static Radix radix_geometry(uint32_t vocab) {
  int dev, sms, optin;
  RT(cudaGetDevice(&dev));
  RT(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  RT(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  constexpr size_t fixed = sizeof(uint32_t) * (256 + 256 + 4) + sizeof(float);
  constexpr size_t fixed_aligned = (fixed + 15) / 16 * 16;
  size_t avail = fs::GetRadixTopKAvailableOrderedSmemBytes<BLOCK>(optin, fixed_aligned, false);
  uint32_t max_chunk = std::max<uint32_t>(avail / sizeof(uint32_t) / VEC * VEC, VEC * BLOCK);
  uint32_t ctas = (vocab + max_chunk - 1) / max_chunk;
  uint32_t chunk = std::min<uint32_t>(((vocab + ctas - 1) / ctas + VEC - 1) / VEC * VEC, max_chunk);
  return {chunk, ctas, std::max<uint32_t>(sms / ctas, 1), unsigned(fixed_aligned + chunk * sizeof(uint32_t))};
}

struct Chain {
  Kernel softmax, topk, topp, minp;
  Radix rx;
  uint32_t vocab;
  fs::RadixRowState* states;
  void run(float* probs, float* temp, int* top_k, float* top_p, float* min_p, uint64_t* seed, uint64_t* offset,
           int* out, bool* valid, uint32_t b) {
    float one = 1.0f, zero = 0.0f;
    uint32_t zero_u = 0, one_u = 1;
    uint64_t zero64 = 0;
    void* nul = nullptr;
    void* a0[] = {&probs, &probs, &temp, &one, &vocab};
    softmax.launch(b, a0);
    uint32_t groups = std::min(rx.groups_max, b);
    void* a1[] = {&probs, &probs, &top_k, &zero_u, &vocab, &b, &states, &rx.chunk, &rx.ctas};
    topk.launch(groups * rx.ctas, a1);
    void* a2[] = {&probs, &probs, &top_p, &zero, &vocab};
    topp.launch(b, a2);
    void* a3[] = {&probs, &min_p, &out, &valid, &nul, &zero, &vocab, &seed, &zero64, &offset, &zero64, &one_u, &one_u};
    minp.launch(b, a3);
  }
};

/// The tokens a row's filters keep, as kern's contract reads them: top-k
/// largest, then the top-p mass (the crossing token included), then
/// p >= min_p * max over what is left, renormalised at each stage.
static std::vector<char> kept(const std::vector<float>& logits, float temp, int top_k, float top_p, float min_p,
                              std::vector<double>* probs_out) {
  size_t v = logits.size();
  std::vector<double> p(v);
  double mx = -1e30;
  for (size_t i = 0; i < v; i++) mx = std::max(mx, double(logits[i]) / temp);
  double z = 0;
  for (size_t i = 0; i < v; i++) z += (p[i] = std::exp(double(logits[i]) / temp - mx));
  for (auto& x : p) x /= z;
  std::vector<int> order(v);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) { return p[a] > p[b]; });
  std::vector<char> keep(v, 1);
  for (size_t r = top_k; r < v; r++) keep[order[r]] = 0;
  double mass = 0;
  for (size_t r = 0; r < v; r++) mass += keep[order[r]] ? p[order[r]] : 0;
  double cum = 0;
  for (size_t r = 0; r < v; r++) {
    int i = order[r];
    if (!keep[i]) continue;
    if (cum >= top_p * mass - 1e-9) keep[i] = 0;
    cum += p[i];
  }
  double pmax = 0, total = 0;
  for (size_t i = 0; i < v; i++)
    if (keep[i]) pmax = std::max(pmax, p[i]), total += p[i];
  for (size_t i = 0; i < v; i++)
    if (keep[i] && p[i] < min_p * pmax * (1 - 1e-4)) keep[i] = 0;
  if (probs_out) *probs_out = p;
  return keep;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "probe <cubin> <entries.txt> [vocab]\n");
    return 2;
  }
  uint32_t vocab = argc > 3 ? atoi(argv[3]) : 151936;
  std::vector<std::string> entries;
  std::ifstream in(argv[2]);
  for (std::string line; std::getline(in, line);)
    if (!line.empty()) entries.push_back(line);
  RT(cudaFree(0));
  CU(cuInit(0));
  CUmodule mod;
  CU(cuModuleLoad(&mod, argv[1]));
  Radix rx = radix_geometry(vocab);
  printf("radix top-k geometry: vocab %u chunk %u ctas_per_row %u groups_max %u smem %u\n", vocab, rx.chunk, rx.ctas,
         rx.groups_max, rx.smem);
  Chain chain{
      kernel(mod, entries, "OnlineSoftmaxFusedKernel", sizeof(fs::OnlineSoftmaxTempStorage<BLOCK>)),
      kernel(mod, entries, "RadixTopKRenormProbKernel_MultiCTA", rx.smem),
      kernel(mod, entries, "TopPRenormProbKernel", sizeof(fs::RenormTempStorage<BLOCK, fs::REDUCE_ALGO>)),
      kernel(mod, entries, "MinPSamplingFromProbKernel",
             sizeof(fs::SamplingTempStorage<BLOCK, fs::SCAN_ALGO, fs::REDUCE_ALGO>)),
      rx, vocab, nullptr};
  RT(cudaMalloc(&chain.states, rx.groups_max * sizeof(fs::RadixRowState)));
  RT(cudaMemset(chain.states, 0, rx.groups_max * sizeof(fs::RadixRowState)));
  printf("smem: softmax %u topk %u topp %u minp %u\n", chain.softmax.smem, chain.topk.smem, chain.topp.smem,
         chain.minp.smem);

  const uint32_t B = 256;
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.f, 3.f);
  std::vector<float> logits(size_t(B) * vocab);
  for (auto& x : logits) x = nd(rng);
  const float temps[] = {1.0f, 0.7f, 1.5f, 1.0f};
  const int ks[] = {int(vocab), 50, int(vocab), 10};
  const float ps[] = {1.0f, 0.9f, 0.5f, 1.0f};
  const float mps[] = {0.0f, 0.05f, 0.0f, 0.1f};
  std::vector<float> temp(B), top_p(B), min_p(B);
  std::vector<int> top_k(B);
  std::vector<uint64_t> seed(B), offset(B);
  for (uint32_t i = 0; i < B; i++) {
    temp[i] = temps[i % 4], top_k[i] = ks[i % 4], top_p[i] = ps[i % 4], min_p[i] = mps[i % 4];
    seed[i] = 1000 + i, offset[i] = 7 * i;
  }
  float *d_logits, *d_probs, *d_temp, *d_top_p, *d_min_p;
  int *d_top_k, *d_out;
  bool* d_valid;
  uint64_t *d_seed, *d_offset;
  size_t bytes = logits.size() * sizeof(float);
  RT(cudaMalloc(&d_logits, bytes));
  RT(cudaMalloc(&d_probs, bytes));
  RT(cudaMalloc(&d_temp, B * 4));
  RT(cudaMalloc(&d_top_p, B * 4));
  RT(cudaMalloc(&d_min_p, B * 4));
  RT(cudaMalloc(&d_top_k, B * 4));
  RT(cudaMalloc(&d_out, B * 4));
  RT(cudaMalloc(&d_valid, B));
  RT(cudaMalloc(&d_seed, B * 8));
  RT(cudaMalloc(&d_offset, B * 8));
  RT(cudaMemcpy(d_logits, logits.data(), bytes, cudaMemcpyHostToDevice));
  RT(cudaMemcpy(d_temp, temp.data(), B * 4, cudaMemcpyHostToDevice));
  RT(cudaMemcpy(d_top_p, top_p.data(), B * 4, cudaMemcpyHostToDevice));
  RT(cudaMemcpy(d_min_p, min_p.data(), B * 4, cudaMemcpyHostToDevice));
  RT(cudaMemcpy(d_top_k, top_k.data(), B * 4, cudaMemcpyHostToDevice));
  RT(cudaMemcpy(d_seed, seed.data(), B * 8, cudaMemcpyHostToDevice));
  RT(cudaMemcpy(d_offset, offset.data(), B * 8, cudaMemcpyHostToDevice));

  auto run = [&](uint32_t b, std::vector<int>& out, std::vector<char>& valid) {
    RT(cudaMemcpy(d_probs, d_logits, size_t(b) * vocab * 4, cudaMemcpyDeviceToDevice));
    chain.run(d_probs, d_temp, d_top_k, d_top_p, d_min_p, d_seed, d_offset, d_out, d_valid, b);
    RT(cudaDeviceSynchronize());
    out.resize(b), valid.resize(b);
    RT(cudaMemcpy(out.data(), d_out, b * 4, cudaMemcpyDeviceToHost));
    RT(cudaMemcpy(valid.data(), d_valid, b, cudaMemcpyDeviceToHost));
  };

  // 1. Every row's token is one its filters keep; the run replays.
  std::vector<int> out1, out2;
  std::vector<char> valid1, valid2;
  run(B, out1, valid1);
  run(B, out2, valid2);
  int bad = 0, invalid = 0, replay = 0;
  for (uint32_t i = 0; i < B; i++) {
    std::vector<float> row(logits.begin() + size_t(i) * vocab, logits.begin() + size_t(i + 1) * vocab);
    auto keep = kept(row, temp[i], top_k[i], top_p[i], min_p[i], nullptr);
    if (!valid1[i]) invalid++;
    if (out1[i] < 0 || out1[i] >= int(vocab) || !keep[out1[i]]) {
      bad++;
      if (bad <= 5) printf("row %u: token %d outside the kept set (T %.2f k %d p %.2f min_p %.2f)\n", i, out1[i], temp[i], top_k[i], top_p[i], min_p[i]);
    }
    if (out1[i] != out2[i]) replay++;
  }
  printf("kept-set check: %d/%u rows outside, %d invalid, %d differ on replay\n", bad, B, invalid, replay);

  // 2. One unfiltered row over many offsets follows its softmax.
  {
    std::vector<double> p;
    std::vector<float> row(logits.begin(), logits.begin() + vocab);
    kept(row, 1.0f, vocab, 1.0f, 0.0f, &p);
    std::vector<int> order(vocab);
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + 30, order.end(), [&](int a, int b) { return p[a] > p[b]; });
    std::vector<float> t1(B, 1.0f), p1(B, 1.0f), m0(B, 0.0f);
    std::vector<int> kv(B, int(vocab));
    RT(cudaMemcpy(d_temp, t1.data(), B * 4, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_top_p, p1.data(), B * 4, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_min_p, m0.data(), B * 4, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_top_k, kv.data(), B * 4, cudaMemcpyHostToDevice));
    for (uint32_t i = 1; i < B; i++) RT(cudaMemcpy(d_logits + size_t(i) * vocab, d_logits, vocab * 4, cudaMemcpyDeviceToDevice));
    const int rounds = 100;
    std::vector<uint64_t> counts(vocab, 0);
    std::vector<int> out;
    std::vector<char> valid;
    for (int r = 0; r < rounds; r++) {
      for (uint32_t i = 0; i < B; i++) offset[i] = uint64_t(r) * B + i;
      RT(cudaMemcpy(d_offset, offset.data(), B * 8, cudaMemcpyHostToDevice));
      run(B, out, valid);
      for (int t : out) counts[t]++;
    }
    double n = double(rounds) * B, chi = 0, rest_p = 1, rest_n = n, tv = 0;
    for (int r = 0; r < 30; r++) {
      int t = order[r];
      double e = p[t] * n;
      chi += (counts[t] - e) * (counts[t] - e) / e;
      rest_p -= p[t], rest_n -= counts[t];
      tv += std::fabs(counts[t] / n - p[t]);
    }
    chi += (rest_n - rest_p * n) * (rest_n - rest_p * n) / (rest_p * n);
    printf("distribution: %d samples, top-30 chi-square %.1f (df 30, 99%% at 50.9), top-30 TV %.4f, top-1 p %.4f got %.4f\n",
           int(n), chi, tv, p[order[0]], counts[order[0]] / n);
  }

  // 3. Time per step at a few batch sizes, every filter on.
  RT(cudaMemcpy(d_temp, temp.data(), B * 4, cudaMemcpyHostToDevice));
  RT(cudaMemcpy(d_top_p, top_p.data(), B * 4, cudaMemcpyHostToDevice));
  RT(cudaMemcpy(d_min_p, min_p.data(), B * 4, cudaMemcpyHostToDevice));
  RT(cudaMemcpy(d_top_k, top_k.data(), B * 4, cudaMemcpyHostToDevice));
  cudaEvent_t e0, e1;
  RT(cudaEventCreate(&e0));
  RT(cudaEventCreate(&e1));
  for (uint32_t b : {1u, 32u, 256u}) {
    const int iters = 50;
    std::vector<int> out;
    std::vector<char> valid;
    run(b, out, valid);
    RT(cudaEventRecord(e0));
    for (int i = 0; i < iters; i++)
      chain.run(d_probs, d_temp, d_top_k, d_top_p, d_min_p, d_seed, d_offset, d_out, d_valid, b);
    RT(cudaEventRecord(e1));
    RT(cudaEventSynchronize(e1));
    float ms;
    RT(cudaEventElapsedTime(&ms, e0, e1));
    printf("batch %3u: %.1f us per step (4 launches)\n", b, ms * 1000 / iters);
  }
  return bad || replay || invalid ? 1 : 0;
}
