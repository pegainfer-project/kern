#!/usr/bin/env python3
"""Reference launcher for TRT-LLM gen's ragged MLA context FMHA (FlashInfer
`trtllm_ragged_attention_deepseek`): one sequence, Q [T, 96, 192] against K/V
that are two views of one expanded [n, 96, 192 | 128] buffer, causal aligned
to the end of the sequence (the chunk's rows are the last T of n), no rope.
Dumps inputs and outputs for the manifest op's bit-exact gate and checks the
kernel against a torch reference. Run under tools/kernel-capture to lift the
launch ABI.

    python3 probe.py <dump dir> [T=512] [n=1536] [max_kv=n]
"""
import pathlib
import sys

import torch
import flashinfer

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
T = int(sys.argv[2]) if len(sys.argv) > 2 else 512
n = int(sys.argv[3]) if len(sys.argv) > 3 else 1536
max_kv = int(sys.argv[4]) if len(sys.argv) > 4 else n
H, DQK, DV = 96, 192, 128
dev = torch.device("cuda")
g = torch.Generator(device=dev).manual_seed(7)
q = (torch.randn(T, H, DQK, device=dev, generator=g) * 0.5).to(torch.bfloat16)
kv = (torch.randn(n, H, DQK + DV, device=dev, generator=g) * 0.5).to(torch.bfloat16)
k, v = kv[:, :, :DQK], kv[:, :, DQK:]
seq_lens = torch.tensor([n], dtype=torch.int32, device=dev)
cum_q = torch.tensor([0, T], dtype=torch.int32, device=dev)
cum_kv = torch.tensor([0, n], dtype=torch.int32, device=dev)
workspace = torch.zeros(128 << 20, dtype=torch.uint8, device=dev)
scale = DQK ** -0.5
o = flashinfer.prefill.trtllm_ragged_attention_deepseek(
    query=q, key=k, value=v, workspace_buffer=workspace, seq_lens=seq_lens,
    max_q_len=T, max_kv_len=max_kv, bmm1_scale=scale, bmm2_scale=1.0, o_sf_scale=1.0,
    batch_size=1, window_left=-1, cum_seq_lens_q=cum_q, cum_seq_lens_kv=cum_kv,
    enable_pdl=False, is_causal=True, return_lse=False)
torch.cuda.synchronize()
print("out", tuple(o.shape), o.dtype, "strides k", k.stride(), "v", v.stride())
# reference: row i attends kv j <= n - T + i
s = torch.einsum("thd,nhd->htn", q.float(), k.float()) * scale
i = torch.arange(T, device=dev)[:, None]
j = torch.arange(n, device=dev)[None, :]
s = s.masked_fill((j > n - T + i)[None], float("-inf"))
ref = torch.einsum("htn,nhd->thd", torch.softmax(s, dim=-1), v.float())
err = (o.float() - ref).abs()
print(f"max|err| {err.max().item():.4f}  rel rms {(err.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item():.2e}")
# top-left alignment would be a different answer: report it too
s2 = torch.einsum("thd,nhd->htn", q.float(), k.float()) * scale
s2 = s2.masked_fill((j > i)[None], float("-inf"))
ref2 = torch.einsum("htn,nhd->thd", torch.softmax(s2, dim=-1), v.float())
print(f"vs top-left causal: max|err| {(o.float() - ref2).abs().max().item():.4f}")
for name, t in [("q", q), ("kv", kv), ("out", o), ("seq_lens", seq_lens), ("cum_q", cum_q), ("cum_kv", cum_kv)]:
    (out / f"{name}.bin").write_bytes(t.contiguous().view(torch.uint8).cpu().numpy().tobytes() if t.dtype != torch.int32 else t.cpu().numpy().tobytes())
print("q", q.data_ptr(), "kv", kv.data_ptr(), "out", o.data_ptr(), "seq_lens", seq_lens.data_ptr(), "cum_q", cum_q.data_ptr(), "cum_kv", cum_kv.data_ptr(), "workspace", workspace.data_ptr())
