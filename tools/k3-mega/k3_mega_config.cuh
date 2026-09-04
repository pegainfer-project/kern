// K3 MegaMoE build constants: the constexpr replicas of DeepGEMM's host
// heuristics (`get_block_config_for_mega_moe`, `get_pipeline_config_for_mega_moe`,
// `get_symm_buffer_size_for_mega_moe`), copied from pegainfer's
// csrc/k3/k3_mega_moe_sm100_common.cuh so the AOT instantiation, the slab
// layout the generator emits, and the tensor maps the runtime encodes all
// derive from one place. Host and device compilable.
#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>

#if defined(__CUDA_ARCH__)
#define K3_MEGA_TRAP() __trap()
#else
#define K3_MEGA_TRAP() __builtin_trap()
#endif
#ifndef DG_UNIFIED_ASSERT
#define DG_UNIFIED_ASSERT(cond)    \
  do {                             \
    if (not(cond)) K3_MEGA_TRAP(); \
  } while (0)
#endif

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>

namespace k3_mega {

using namespace deep_gemm;

// Per-expert K3 latent-MoE shapes. L1 is the fused gate|up projection.
constexpr int kHidden = 3584;
constexpr int kIntermediate = 3072;
constexpr int kNumTopk = 16;
// Protocol maximum tokens per rank per launch; the ring capacities derived
// from it are template parameters, so every rank of a world must agree.
constexpr int kMaxTokensPerRank = 16896;
constexpr int kSfGroupK = 32;
constexpr int kSfPerWord = 4;
constexpr int kSfWordK = kSfGroupK * kSfPerWord;  // 128
constexpr int kSmemCapacity = 232448;  // SM100ArchSpec::smem_capacity
constexpr int kMegaBlockN = 128;
constexpr int kNumDispatchThreads = 128;
constexpr int kNumNonEpilogueThreads = 128;
constexpr int kGb300Sms = 152;

constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }
constexpr int alignup(int a, int b) { return cdiv(a, b) * b; }

constexpr int mega_ring_tokens(int num_ranks, int num_experts, int num_max_tokens_per_rank,
                               int num_topk, int hidden, int intermediate, int num_sms) {
  const int per_rank = num_experts / num_ranks;
  const int active_topk = num_topk < per_rank ? num_topk : per_rank;
  const int routed = num_max_tokens_per_rank * num_ranks * active_topk;
  int best = 0;
  for (int i = 0; i < layout::kNumCandidateBlockMs; ++i) {
    const int bm = layout::kCandidateBlockM[i];
    const int pool = cdiv(routed, bm) + per_rank;
    const int live = sched::get_num_max_live_pool_blocks(pool, num_sms, hidden, intermediate);
    const int tokens = live * bm;
    if (tokens > best) best = tokens;
  }
  return alignup(best, layout::kLCMCandidateBlockM);
}

constexpr int mega_sf_ring_tokens(int num_ring_tokens) {
  int best = 0;
  for (int i = 0; i < layout::kNumCandidateBlockMs; ++i) {
    const int t = layout::get_num_sf_ring_tokens(num_ring_tokens, layout::kCandidateBlockM[i]);
    if (t > best) best = t;
  }
  return best;
}

constexpr int mega_bytes_per_pull(int hidden) {
  int b = hidden;
  while (b > 4096) b /= 2;
  return b;
}

struct BlockConfig {
  int bucket_x2;
  int block_m, store_block_m, block_k, num_epilogue_warpgroups;
};

constexpr BlockConfig kBlockLadder[] = {
    {17, 16, 8, 256, 2},    // <= 8.5 expected tokens/expert
    {33, 32, 16, 128, 2},   // <= 16.5
    {65, 64, 32, 128, 1},   // <= 32.5
    {129, 96, 16, 128, 2},  // <= 64.5
    {193, 128, 32, 128, 2}, // <= 96.5
    {0, 192, 32, 128, 2},   // otherwise
};
constexpr int kNumBlockConfigs = (int)(sizeof(kBlockLadder) / sizeof(kBlockLadder[0]));

constexpr int mega_block_config_index(int num_tokens, int num_ranks, int num_experts,
                                      int num_topk) {
  const long long lhs = 2LL * num_tokens * num_ranks * num_topk;
  for (int i = 0; i < kNumBlockConfigs - 1; ++i) {
    if (lhs <= (long long)kBlockLadder[i].bucket_x2 * num_experts) return i;
  }
  return kNumBlockConfigs - 1;
}

struct PipelineConfig {
  int num_stages;
  int smem_size;
};

constexpr PipelineConfig mega_pipeline(int num_experts, int block_m, int block_n, int block_k,
                                       int store_block_m, int sf_block_m, int sf_block_n,
                                       int num_dispatch_warps, int num_epilogue_warps,
                                       int num_bytes_per_pull) {
  constexpr int kSmemAlignment = 1024;
  const int smem_dispatch = alignup(num_experts * (int)sizeof(uint32_t), kSmemAlignment) +
                            alignup(num_bytes_per_pull * num_dispatch_warps, kSmemAlignment);
  const int wg = num_epilogue_warps / 4;
  const int smem_cd_l1 = wg * store_block_m * (block_n / 2) * 2;
  const int smem_cd_l2 = wg * store_block_m * block_n * (int)sizeof(nv_bfloat16);
  const int smem_cd = alignup(smem_cd_l1 > smem_cd_l2 ? smem_cd_l1 : smem_cd_l2, kSmemAlignment);
  const int smem_task_info = 2 * (int)sizeof(sched::TaskInfo<true>);
  const int smem_barriers = (num_dispatch_warps + 2 * 2 + num_epilogue_warps * 2 + 2 * 2) * 8;
  const int smem_amax = store_block_m * num_epilogue_warps * (int)sizeof(float);
  const int smem_fixed =
      smem_dispatch + smem_cd + smem_amax + smem_barriers + smem_task_info + 4;
  const int per_stage = (block_m / 2) * block_k + block_n * block_k +
                        sf_block_m * (block_k / kSfGroupK) + sf_block_n * (block_k / kSfGroupK) +
                        2 * 8;
  const int stages = (kSmemCapacity - smem_fixed) / per_stage;
  return {stages, smem_fixed + stages * per_stage};
}

static_assert(kMaxTokensPerRank % layout::kLCMCandidateBlockM == 0,
              "the protocol maximum must satisfy the upstream token alignment");
constexpr int kBytesPerPull = mega_bytes_per_pull(kHidden);

template <int kExperts, int kRanks>
struct MegaRing {
  static constexpr int kTokens = mega_ring_tokens(kRanks, kExperts, kMaxTokensPerRank, kNumTopk,
                                                  kHidden, kIntermediate, kGb300Sms);
  static constexpr int kSfTokens = mega_sf_ring_tokens(kTokens);
};

// One AOT kernel's geometry: a world (GLOBAL expert count x rank count)
// crossed with a ladder entry.
template <int kExperts, int kRanks, int kCfgIdx>
struct MegaGeom {
  static constexpr BlockConfig kCfg = kBlockLadder[kCfgIdx];
  static constexpr int kBlockM = kCfg.block_m;
  static constexpr int kBlockK = kCfg.block_k;
  static constexpr int kStoreBlockM = kCfg.store_block_m;
  static constexpr int kEpilogueThreads = kCfg.num_epilogue_warpgroups * 128;
  static constexpr int kSfBlockM = alignup(kBlockM, 128);
  static constexpr int kSfBlockN = alignup(kMegaBlockN, 128);
  static constexpr PipelineConfig kPipe =
      mega_pipeline(kExperts, kBlockM, kMegaBlockN, kBlockK, kStoreBlockM, kSfBlockM,
                    kSfBlockN, kNumDispatchThreads / 32, kEpilogueThreads / 32, kBytesPerPull);
  static constexpr int kSmemSize = kPipe.smem_size;
  static constexpr int kNumThreads =
      kNumDispatchThreads + kNumNonEpilogueThreads + kEpilogueThreads;
  static_assert(kPipe.num_stages >= 2, "MegaMoE pipeline needs at least 2 stages");
  static_assert(kSmemSize <= kSmemCapacity, "MegaMoE smem budget overflow");
};

// Every world pins ONE config, so peers of a collective launch agree on
// BLOCK_M and a row's MMA K-accumulation order does not depend on the world
// size (the EP-vs-EP1 oracle leans on this). The entry is BLOCK_M 96: on the
// K3 layer shape it is fastest or tied from 8 to 512 tokens per rank under
// real routing (-12..15% against the protocol-max entry, 192; -34% at 256
// uniform), with bit-identical output — see the k3_moe_bench sweep,
// bench_results 2026-09-04-k5-span-profile. `-DK3_MEGA_CFG=<i>` builds
// another ladder entry for such sweeps.
constexpr int index_of_block_m(int block_m) {
  for (int i = 0; i < kNumBlockConfigs; ++i)
    if (kBlockLadder[i].block_m == block_m) return i;
  return -1;
}

template <int kExperts, int kRanks>
constexpr int pinned_config() {
#ifdef K3_MEGA_CFG
  return K3_MEGA_CFG;
#else
  constexpr int cfg = index_of_block_m(96);
  static_assert(cfg >= 0, "the ladder lost its BLOCK_M 96 entry");
  return cfg;
#endif
}

}  // namespace k3_mega
