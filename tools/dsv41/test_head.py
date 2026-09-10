"""Check FP32 head selection against the supplied inference.sample."""
import argparse
import ast
import ctypes
from pathlib import Path
import torch
from cuda.bindings import driver as cu


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inference", type=Path, required=True)
    p.add_argument("--cubin", type=Path, required=True)
    args = p.parse_args()
    tree = ast.parse((args.inference / "model.py").read_text())
    scope = {"torch": torch}
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "sample")
    exec(compile(ast.Module(body=[node], type_ignores=[]), "reference_sample", "exec"), scope)
    torch.manual_seed(31)
    torch.empty(1, device="cuda")
    def check(r):
        assert r[0] == cu.CUresult.CUDA_SUCCESS, r
        return r[1] if len(r) == 2 else r[1:]
    mod = check(cu.cuModuleLoad(str(args.cubin).encode()))
    def launch(name, grid, block, vals, types):
        fn = check(cu.cuModuleGetFunction(mod, name.encode()))
        vals = tuple(v.data_ptr() if isinstance(v, torch.Tensor) else v for v in vals)
        check(cu.cuLaunchKernel(fn, *grid, block, 1, 1, 0,
                               cu.CUstream(torch.cuda.current_stream().cuda_stream),
                               (vals, tuple(types)), 0))
    ptr, i64, i32 = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int
    for batch in (1, 17, 128):
        vocab = 129280
        logits = torch.randn(batch, 5, vocab, device="cuda")
        bias = torch.randn(batch, vocab, device="cuda")
        out = torch.full((batch, 5), -99, device="cuda", dtype=torch.int64)
        maxima = torch.empty(batch, 64, device="cuda")
        indices = torch.empty(batch, 64, device="cuda", dtype=torch.int32)
        # Same BF16 bin but different FP32 values: selection must retain precision.
        logits[:, :, 17:19] = torch.tensor([32.001, 32.002], device="cuda")
        bias[:, 17:19] = 0
        # An exact tie separately verifies the smallest-index convention.
        logits[0, 0, 17:19] = 32
        for step in range(5):
            launch("kern_argmax_rows_partial_f32_bias", (batch, 64, 1), 1024,
                   [logits[:, step], 5*vocab, bias, vocab, maxima, indices, vocab],
                   [ptr,i64,ptr,i64,ptr,ptr,i32])
            launch("kern_argmax_rows_final_i64", (batch, 1, 1), 64,
                   [maxima, indices, out[:, step], 5, 64], [ptr,ptr,ptr,i32,i32])
            expected = scope["sample"](logits[:, step] + bias, 0)
            torch.testing.assert_close(out[:, step], expected, atol=0, rtol=0)
        print(f"batch={batch}: five FP32 biased rows, close values and ties PASS", flush=True)
        launch("kern_argmax_rows_partial_f32_bias", (batch, 64, 1), 1024,
               [logits[:, 0], 5*vocab, 0, 0, maxima, indices, vocab],
               [ptr,i64,ptr,i64,ptr,ptr,i32])
        launch("kern_argmax_rows_final_i64", (batch, 1, 1), 64,
               [maxima, indices, out[:, 0], 5, 64], [ptr,ptr,ptr,i32,i32])
        torch.testing.assert_close(out[:, 0], scope["sample"](logits[:, 0], 0), atol=0, rtol=0)
        print(f"batch={batch}: target head without bias PASS", flush=True)
    check(cu.cuModuleUnload(mod))


if __name__ == "__main__":
    main()
