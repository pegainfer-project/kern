"""Adapt only MegaMoE ABI to kern device peer table; retain upstream body."""
import argparse,pathlib,hashlib
p=argparse.ArgumentParser();p.add_argument('deepgemm',type=pathlib.Path);p.add_argument('out',type=pathlib.Path);a=p.parse_args()
s=(a.deepgemm/'deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh').read_text()
assert hashlib.sha256(s.encode()).hexdigest() == '7745284efdced130f4c6e40acd47b34f846ccdb8ecd1959446fd53403894dd15', 'unreviewed DeepGEMM source'
s=s.replace('CUTLASS_GLOBAL __launch_bounds__(kNumThreads, 1) void\nsm100_fp8_fp4_mega_moe_impl(', 'CUTLASS_DEVICE void\nsm100_fp8_fp4_mega_moe_body(')
old='const __grid_constant__ layout::SymBuffer<kNumRanks> sym_buffer,'
assert s.count(old)==1;s=s.replace(old,'const int64_t* kern_peer_bases, uint32_t kern_rank,')
old='const __grid_constant__ cute::TmaDescriptor tensor_map_';assert s.count(old)==18;s=s.replace(old,'const cute::TmaDescriptor& tensor_map_')
anchor='    using Barrier = cutlass::arch::ClusterTransactionBarrier;';assert s.count(anchor)==1;s=s.replace(anchor,'    const layout::SymBuffer<kNumRanks> sym_buffer(kern_peer_bases, kern_rank);\n'+anchor)
a.out.write_text(s)
