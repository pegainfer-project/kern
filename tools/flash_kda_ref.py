"""K3's KDA time axis in float64, over a span, from tools/flash-kda/probe.cu's dumps.

Checks the vendored FlashKDA against the per-token math kern's K3 kernel implements
(tools/k3-harness/ref.h `ref_kda_core`, docs/k3-kernel-abi.md K3): q/k L2-normalised,
q scaled, beta = sigmoid(raw), decay = exp(LB * sigmoid(exp(a_log) * (g + dt_bias))),
S = S * decay + (v - S (decay * k)) beta k^T, attn = S q. Reports the relative RMS of
FlashKDA's `out` against this and which orientation of `state_out` ([h][dv][dk], K3's
rec, or its transpose) FlashKDA writes.

    python3 tools/flash_kda_ref.py <dump dir> <T> <H> [--scale 0.0078125] [--lb -5]
"""
import argparse
import numpy as np


def bf16(path, shape):
    u = np.fromfile(path, dtype=np.uint16).astype(np.uint32) << 16
    return u.view(np.float32).reshape(shape).astype(np.float64)


def to_bf16(x):
    """Round to bf16 (round-to-nearest-even), as float64."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16
    return r.astype(np.uint32).view(np.float32).astype(np.float64)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def rel_rms(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)) / (np.sqrt(np.mean(b ** 2)) + 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("T", type=int)
    ap.add_argument("H", type=int)
    ap.add_argument("--scale", type=float, default=1.0 / 128)
    ap.add_argument("--lb", type=float, default=-5.0)
    a = ap.parse_args()
    T, H, D = a.T, a.H, 128
    d = a.dir
    q, k, v, g = (bf16(f"{d}/{n}.bin", (T, H, D)) for n in ["q", "k", "v", "g"])
    beta = sigmoid(bf16(f"{d}/beta_ht.bin", (H, T)))
    a_log = np.fromfile(f"{d}/a_log.bin", dtype=np.float32).astype(np.float64)
    dt_bias = np.fromfile(f"{d}/dt_bias.bin", dtype=np.float32).reshape(H, D).astype(np.float64)
    s_in = np.fromfile(f"{d}/state_in.bin", dtype=np.float32).reshape(H, D, D).astype(np.float64)
    out_k = bf16(f"{d}/out.bin", (T, H, D))
    s_out_k = np.fromfile(f"{d}/state_out.bin", dtype=np.float32).reshape(H, D, D).astype(np.float64)

    qn = to_bf16(q * to_bf16(1.0 / np.sqrt(to_bf16(np.sum(to_bf16(q * q), -1, keepdims=True)) + 1e-6))) * a.scale
    kn = to_bf16(k * to_bf16(1.0 / np.sqrt(to_bf16(np.sum(to_bf16(k * k), -1, keepdims=True)) + 1e-6)))
    dec = np.exp(a.lb * sigmoid(np.exp(a_log)[None, :, None] * (g + dt_bias[None])))  # [T, H, D]
    out = np.zeros((T, H, D))
    for orient, name in [(False, "S[h][dv][dk] (K3 rec)"), (True, "S[h][dk][dv] (transposed)")]:
        S = s_in.transpose(0, 2, 1).copy() if orient else s_in.copy()  # work in [h][dv][dk]
        for t in range(T):
            for h in range(H):
                Sd = S[h] * dec[t, h][None, :]  # decay along dk
                m = Sd @ kn[t, h]  # [dv]
                dlt = (v[t, h] - m) * beta[h, t]
                S[h] = Sd + np.outer(dlt, kn[t, h])
                out[t, h] = S[h] @ qn[t, h]
        S_cmp = S.transpose(0, 2, 1) if orient else S
        print(f"{name}: out relRMS {rel_rms(out_k, out):.3e}   state_out relRMS {rel_rms(s_out_k, S_cmp):.3e}")
    if T == 1:
        print("first-token check (state-orientation-free):", rel_rms(out_k, out))


if __name__ == "__main__":
    main()
