# Patches applied to upstream kernel sources before instantiation

The tuned DSv4.1 modules are instantiated from upstream sources carrying these
patches (`diff -ru`, apply with `patch -p1` inside the upstream checkout). A
build from the unpatched source produces a different cubin and therefore a
different sha; the manifest pins what was built, and whether the two agree is
`kern test`'s verdict. The same files are published beside the cubins in the
registry (`patches/` of `Pegainfer/kern-deepseek-v41-flash-sm103`).

| patch | upstream | what it changes | consumer |
|---|---|---|---|
| `deepgemm-pdl-window.patch` | [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) `ab69f76be5bb9ea3499bc755002b1a876cb0b3d9` | `sm100_fp8_fp4_gemm_1d1d`: programmatic dependent launch, the dependent grid is released as the producer drains (its own `griddepcontrol.wait` still orders the data) | `moe/dense.cu` (`dsv41_dense`), built by `moe/build.sh` with `DEEPGEMM_ROOT` at the patched copy |
| `deepgemm-mhc-gate-early-trigger.patch` | DeepGEMM, same commit | Mega mHC and Mega Gate issue `griddepcontrol.launch_dependents` right after their grid-dependency wait, so a PDL successor's prologue overlaps the whole kernel | `moe/mega_mhc.cu`, `moe/mega_gate.cu` (`dsv41_mhc`, `dsv41_gate`) |
| `flashmla-split-kv.patch` | [FlashMLA](https://github.com/deepseek-ai/FlashMLA) `4f38f29ef6793c228363e4af5be66d44e81167ba` | a `DecodeWithSplitKV` mode of the fused V4.1 decode kernel: `parts` CTAs per row over disjoint KV block ranges into fp32 partials, plus `fused_split_combine_kernel` (lse merge, sink, inverse O RoPE, MXFP8 cast) producing the one-CTA path's output layout | `attention/fused_split_probe.cu` via `attention/build_fused.sh` (`dsv41_fused_split`); the one-CTA `dsv41_fused_decode` is built from the unpatched source |

Records of the builds and their A/B measurements (not in the repository):
`bench_results/2026-09-10-dsv41-gemm-tiles` (tile table, PDL window),
`2026-09-10-dsv41-mhc-gate` (early trigger), `2026-09-10-dsv41-attn-split`
(split-KV, bit-identical output at 2/5/10 parts), `2026-09-10-dsv41-pdl`
(PDL flags), `2026-09-10-dsv41-batch` (the 1M / 256-sequence manifest).
