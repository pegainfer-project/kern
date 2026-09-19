#!/usr/bin/env python3
"""Reference launcher for TRT-LLM gen's batched GEMMs the way FlashInfer's
fused MoE runs them (kernel index families `trtllm_bmm_mxe4m3_mxe2m1_mxe4m3`
for FC1 with the fused SiTU gate and `trtllm_bmm_bf16_mxe2m1_mxe4m3` for
FC2), on K3's latent MoE shape: mxfp8 tokens x mxfp4 experts, 224 experts,
top-16, one rank's 56 local experts, precomputed routing. Checks the kernels
against a torch dequantised reference, dumps inputs and outputs for the
manifest ops' gate, and prints the device pointers the capture's launches
name. Run under tools/kernel-capture to lift the launch ABI.

    python3 probe.py <dump dir> [--tokens 300] [--rank 1] [--local 56] [--autotune] [--no-dump]
"""
import argparse
import pathlib

import torch
import flashinfer
from flashinfer import reorder_rows_for_gated_act_gemm, shuffle_matrix_a, shuffle_matrix_sf_a
from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
from flashinfer.tllm_enums import ActivationType

H, I, E, K = 3584, 3072, 224, 16
ALPHA, BETA = 4.0, 25.0  # situ(g, u) = ALPHA tanh(g/ALPHA) sigmoid(g) * BETA tanh(u/BETA)
E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])


def deq_fp4(packed, sf):
    """[n, k/2] u8 nibbles (element 2j in the low nibble) with [n, k/32] UE8M0 scales -> [n, k] f32."""
    nib = torch.stack([packed & 0xF, packed >> 4], -1).reshape(packed.shape[0], -1).long()
    val = E2M1.to(packed.device)[nib & 7] * torch.where(nib & 8 != 0, -1.0, 1.0)
    return val * torch.exp2(sf.float() - 127).repeat_interleave(32, dim=1)


def deq_fp8(x, sf):
    return x.float() * torch.exp2(sf.float() - 127).repeat_interleave(32, dim=1)


def mxfp8(x):
    """Quantise [n, k] f32 to mxfp8 and back: one UE8M0 scale per 32 elements, 2^(floor(log2 amax) - 8)."""
    blocks = x.reshape(x.shape[0], -1, 32)
    amax = blocks.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    scale = torch.exp2(torch.floor(torch.log2(amax)) - 8)
    return ((blocks / scale).to(torch.float8_e4m3fn).float() * scale).reshape(x.shape)


def situ(g, u):
    return ALPHA * torch.tanh(g / ALPHA) * torch.sigmoid(g) * BETA * torch.tanh(u / BETA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("--tokens", type=int, default=300)
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--local", type=int, default=56)
    ap.add_argument("--autotune", action="store_true")
    ap.add_argument("--no-dump", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    T, EL, off = a.tokens, a.local, a.rank * a.local
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(7)
    rnd = lambda *s: torch.rand(*s, device=dev, generator=g)
    x = (torch.randn(T, H, device=dev, generator=g) * 0.5).to(torch.bfloat16)
    x_fp8, x_sf = flashinfer.mxfp8_quantize(x, is_sf_swizzled_layout=False)
    x_sf = x_sf.reshape(T, H // 32)
    w13 = (rnd(EL, 2 * I, H // 2) * 256).to(torch.uint8)
    w13_sf = (rnd(EL, 2 * I, H // 32) * 8 + 121).to(torch.uint8)
    w2 = (rnd(EL, H, I // 2) * 256).to(torch.uint8)
    w2_sf = (rnd(EL, H, I // 32) * 8 + 121).to(torch.uint8)
    ids = torch.stack([torch.randperm(E, device=dev, generator=g)[:K] for _ in range(T)]).int()
    wts = rnd(T, K) + 0.1
    wts = wts / wts.sum(-1, keepdim=True)
    # FlashInfer's static weight prep (vLLM flashinfer_fp4_moe.prepare_static_weights_for_trtllm_fp4_moe):
    # FC1 rows [gate; up] interleaved pairwise, then every matrix row-shuffled for the epilogue tile,
    # the scales likewise and re-laid in the 128x4 block layout.
    w13s = torch.stack([shuffle_matrix_a(reorder_rows_for_gated_act_gemm(w13[e]), 128) for e in range(EL)])
    w13_sfs = torch.stack([shuffle_matrix_sf_a(reorder_rows_for_gated_act_gemm(w13_sf[e]), 128, 32).reshape(2 * I, H // 32)
                           for e in range(EL)])
    w2s = torch.stack([shuffle_matrix_a(w2[e], 128) for e in range(EL)])
    w2_sfs = torch.stack([shuffle_matrix_sf_a(w2_sf[e], 128, 32).reshape(H, I // 32) for e in range(EL)])
    alpha = torch.full((EL,), ALPHA, device=dev)
    beta = torch.full((EL,), BETA, device=dev)

    def run():
        return trtllm_fp4_block_scale_routed_moe(
            (ids, wts), None, x_fp8, x_sf.view(torch.float8_e4m3fn), w13s, w13_sfs.view(torch.float8_e4m3fn), None,
            alpha, beta, None, w2s, w2_sfs.view(torch.float8_e4m3fn), None, None, None, None, E, K, None, None, I,
            off, EL, None, routing_method_type=1, do_finalize=True, activation_type=ActivationType.Situ.value)[0]

    if a.autotune:
        from flashinfer.autotuner import autotune
        with autotune(True):
            out = run()
    out = run()
    torch.cuda.synchronize()
    print("out", tuple(out.shape), out.dtype)

    # reference, both gate/up pairings
    x_deq = deq_fp8(x_fp8, x_sf)
    ref = [torch.zeros(T, H, device=dev) for _ in range(2)]
    for le in range(EL):
        t_idx, k_idx = (ids == off + le).nonzero(as_tuple=True)
        if t_idx.numel() == 0:
            continue
        h = x_deq[t_idx] @ deq_fp4(w13[le], w13_sf[le]).T
        w2d = deq_fp4(w2[le], w2_sf[le])
        for r, (gg, uu) in zip(ref, [(h[:, :I], h[:, I:]), (h[:, I:], h[:, :I])]):
            r.index_add_(0, t_idx, (mxfp8(situ(gg, uu)) @ w2d.T) * wts[t_idx, k_idx][:, None])
    for name, r in zip(["gate=first", "gate=second"], ref):
        err = (out.float() - r).abs()
        print(f"{name}: max|err| {err.max().item():.4f}  rel rms {(err.pow(2).mean().sqrt() / r.pow(2).mean().sqrt()).item():.2e}")
    if not a.no_dump:
        for name, t in [("x", x), ("x_fp8", x_fp8), ("x_sf", x_sf), ("ids", ids), ("wts", wts), ("w13", w13), ("w13_sf", w13_sf),
                        ("w2", w2), ("w2_sf", w2_sf), ("w13s", w13s), ("w13_sfs", w13_sfs), ("w2s", w2s),
                        ("w2_sfs", w2_sfs), ("out", out)]:
            (a.out / f"{name}.bin").write_bytes(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    print(" ".join(f"{n}={t.data_ptr():#x}" for n, t in [("x_fp8", x_fp8), ("x_sf", x_sf), ("ids", ids), ("wts", wts),
                                                          ("w13s", w13s), ("w13_sfs", w13_sfs), ("w2s", w2s),
                                                          ("w2_sfs", w2_sfs), ("alpha", alpha), ("beta", beta),
                                                          ("out", out)]))


if __name__ == "__main__":
    main()
