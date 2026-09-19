#!/usr/bin/env python3
"""Compile FlashInfer's CuTe-DSL MLA decode kernel for K3 and land the cubin
for the kernel index (then tools/kernels/import_cubin.py --family mla_decode_h96_p64).

    python3 tools/build_mla_dsl.py <out_dir>

Needs FlashInfer with its cute_dsl attention package, the CuTe DSL
(nvidia-cutlass-dsl) and a GPU: the DSL JIT-compiles on first call. The
kernel is exercised once on a tiny problem so the compile happens, then the
cubin the DSL kept (CUTE_DSL_KEEP_CUBIN) is copied out under its stable name.
"""
import math
import os
import pathlib
import shutil
import sys
import tempfile

out = pathlib.Path(sys.argv[1])
dump = tempfile.mkdtemp(prefix="cute-dsl-")
os.environ["CUTE_DSL_KEEP_CUBIN"] = "1"
os.environ["CUTE_DSL_DUMP_DIR"] = dump

import torch  # noqa: E402
from cutlass import Float32, Int32  # noqa: E402
from flashinfer.cute_dsl.attention.monolithic.mla_decode import _get_compiled_mla_kernel  # noqa: E402

PAGE, L, R, H = 64, 512, 64, 96
dev = "cuda"
B, maxp, npages, split = 2, 2, 4, 2
pt = torch.arange(B * maxp, device=dev, dtype=torch.int32).reshape(B, maxp)
lens = torch.tensor([70, 100], device=dev, dtype=torch.int32)
bsk = torch.tensor([1, 2], device=dev, dtype=torch.int32)
q = torch.randn(B, 1, H, L + R, device=dev).to(torch.bfloat16)
kv = torch.randn(npages, PAGE, L + R, device=dev).to(torch.bfloat16)
ws = torch.zeros(128 * split * L * B * 4 + 128 * split * B * 4, dtype=torch.int8, device=dev)
o = torch.empty(B, 1, H, L, dtype=torch.bfloat16, device=dev)
lse = torch.empty(B, 1, H, dtype=torch.float32, device=dev)
k = _get_compiled_mla_kernel(torch_dtype=torch.bfloat16, torch_out_dtype=torch.bfloat16, page_size=PAGE, kv_lora_rank=L,
                             qk_rope_head_dim=R, num_heads=H, seq_len_q=1, is_persistent=False, is_var_seq=True,
                             is_var_split_kv=True, reducer_d_tiles=1, reducer_max_splits=256, skip_correction_threshold=0.0,
                             is_workspace_size_zero=False, enable_pdl=False)
k(q[..., :L], q[..., L:], kv[:, :, :L], kv[:, :, L:], pt, o, lse, ws, Int32(split), lens, bsk, Float32(1 / math.sqrt(192)),
  Float32(1.0))
torch.cuda.synchronize()
cubins = sorted(pathlib.Path(dump).glob("*.sm_*.cubin"))
assert len(cubins) == 1, cubins
dst = out / f"mla_decode_h{H}_p{PAGE}.cubin"
shutil.copyfile(cubins[0], dst)
print(f"{dst}: {dst.stat().st_size} bytes from {cubins[0].name}")
