// Reference launcher for the vendored FlashKDA kernels: one forward on
// deterministic inputs through upstream's own `launch_fwd`, so the launch can
// be captured (tools/kernel-capture) and the outputs kept as the bit-exact
// oracle for kern's manifest op. Not part of any build; see ../kernel-capture/README.md.
//
//   probe <T> <H> [outdir]      writes q,k,v,g,beta_ht,a_log,dt_bias,state_in,
//                               state_out,out as raw little-endian .bin files
#include "csrc/fwd.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)

// Deterministic normal-ish samples (sum of uniforms), independent of libc.
struct Rng {
    uint64_t s;
    float uniform() { s = s * 6364136223846793005ull + 1442695040888963407ull; return float(s >> 40) / float(1u << 24); }
    float normal() { float a = 0; for (int i = 0; i < 12; i++) a += uniform(); return a - 6.f; }
};
static uint16_t bf16(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return uint16_t(u >> 16); }

template <class T> static T* upload(const std::vector<T>& h) {
    T* d; CK(cudaMalloc(&d, h.size() * sizeof(T))); CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); return d;
}
template <class T> static void dump(const std::string& dir, const char* name, const void* d, size_t n) {
    if (dir.empty()) return;
    std::vector<T> h(n); CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    FILE* f = fopen((dir + "/" + name + ".bin").c_str(), "wb"); if (!f) { perror(name); exit(1); }
    fwrite(h.data(), sizeof(T), n, f); fclose(f);
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: probe <T> <H> [outdir]\n"); return 2; }
    const int T = atoi(argv[1]), H = atoi(argv[2]), D = 128, CHUNK = 16;
    const std::string dir = argc > 3 ? argv[3] : "";
    const float scale = 1.f / 128.f, lower_bound = -5.f, gate_scale = lower_bound * 1.4426950408889634f;
    Rng rng{1};
    const size_t n = size_t(T) * H * D;
    std::vector<uint16_t> q(n), k(n), v(n), g(n), beta(size_t(T) * H);
    for (auto& x : q) x = bf16(rng.normal());
    for (auto& x : k) x = bf16(rng.normal());
    for (auto& x : v) x = bf16(rng.normal());
    for (auto& x : g) x = bf16(rng.normal());
    for (auto& x : beta) x = bf16(rng.normal());
    std::vector<float> state(size_t(H) * D * D), a_log(H), dt_bias(size_t(H) * D);
    for (auto& x : state) x = 0.1f * rng.normal();
    for (auto& x : a_log) x = 0.5f * rng.normal();
    for (auto& x : dt_bias) x = 0.5f * rng.normal();

    // Same workspace arithmetic as upstream's binding (varlen slack kept: +N tiles).
    const int N = 1, total_tiles = (T + CHUNK - 1) / CHUNK;
    const long long per_tile = 3LL * CHUNK * D * 2 + D * 4 + 2LL * CHUNK * CHUNK * 2;
    const long long ws_bytes = (long long)H * (total_tiles + N) * per_tile + 128;

    auto dq = upload(q), dk = upload(k), dv = upload(v), dg = upload(g), dbeta = upload(beta);
    auto dstate_in = upload(state), da_log = upload(a_log), ddt_bias = upload(dt_bias);
    float* dstate_out; uint16_t* dout; void* dws;
    CK(cudaMalloc(&dstate_out, state.size() * 4)); CK(cudaMalloc(&dout, n * 2)); CK(cudaMalloc(&dws, ws_bytes));
    CK(cudaMemset(dstate_out, 0, state.size() * 4)); CK(cudaMemset(dout, 0, n * 2)); CK(cudaMemset(dws, 0, ws_bytes));

    launch_fwd<128, true, true, true, false>(
        (cutlass::bfloat16_t const*)dq, (cutlass::bfloat16_t const*)dk, (cutlass::bfloat16_t const*)dv,
        (cutlass::bfloat16_t const*)dg, (cutlass::bfloat16_t const*)dbeta, dstate_in, scale, dstate_out,
        (cutlass::bfloat16_t*)dout, dws, total_tiles, T, H, N, nullptr, da_log, ddt_bias, gate_scale, 0);
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());

    dump<uint16_t>(dir, "q", dq, n); dump<uint16_t>(dir, "k", dk, n); dump<uint16_t>(dir, "v", dv, n);
    dump<uint16_t>(dir, "g", dg, n); dump<uint16_t>(dir, "beta_ht", dbeta, beta.size());
    dump<float>(dir, "a_log", da_log, H); dump<float>(dir, "dt_bias", ddt_bias, dt_bias.size());
    dump<float>(dir, "state_in", dstate_in, state.size()); dump<float>(dir, "state_out", dstate_out, state.size());
    dump<uint16_t>(dir, "out", dout, n);
    printf("T=%d H=%d tiles=%d workspace=%lld B%s\n", T, H, total_tiles, ws_bytes, dir.empty() ? "" : " (dumped)");
    // Each allocation by name, for lift.py --names (a captured pointer is
    // otherwise only a letter, and letters get mapped to the wrong buffer).
    printf("q %p\nk %p\nv %p\ng %p\nbeta %p\nstate_in %p\na_log %p\ndt_bias %p\nstate_out %p\nout %p\nws %p\n",
           (void*)dq, (void*)dk, (void*)dv, (void*)dg, (void*)dbeta, (void*)dstate_in, (void*)da_log, (void*)ddt_bias,
           (void*)dstate_out, (void*)dout, dws);
    return 0;
}
