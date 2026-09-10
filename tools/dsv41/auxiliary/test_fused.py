"""Byte equality of the fused attention-input kernels against the pairs they replace.

`dsv41_norm_quant` must reproduce `dsv41_norm` followed by `dsv41_dense_quant_x`
(moe/stage.cu) and `dsv41_norm_rope` must reproduce `dsv41_norm` followed by
`dsv41_rope`, byte for byte, including the rows the four-row scale padding
leaves untouched. Every output buffer starts from the same poison so a write
outside the region either kernel owns shows up as a difference.

    test_fused.py AUXILIARY_CUBIN STAGE_CUBIN

Run on an allocated GPU. Row counts cover one row, counts that are not a
multiple of four, and the full capacity; strides cover both a tight row and a
slice of a wider projection output.
"""
import argparse
import ctypes
from pathlib import Path
import torch
from cuda.bindings import driver as cu

CAPACITY, POISON = 128, 0xCD


def check(result):
    assert result[0] == cu.CUresult.CUDA_SUCCESS, result
    return result[1] if len(result) == 2 else result[1:]


def launcher(cubin):
    module = check(cu.cuModuleLoad(str(cubin).encode()))
    def launch(entry, grid, *values):
        vals = [v.data_ptr() if isinstance(v, torch.Tensor) else v for v in values]
        types = [ctypes.c_void_p if isinstance(v, torch.Tensor) else
                 ctypes.c_float if isinstance(v, float) else ctypes.c_int for v in values]
        fn = check(cu.cuModuleGetFunction(module, entry.encode()))
        check(cu.cuLaunchKernel(fn, grid, 1, 1, 256, 1, 1, 0,
                                cu.CUstream(torch.cuda.current_stream().cuda_stream),
                                (tuple(vals), tuple(types)), 0))
        torch.cuda.synchronize()
    return launch


def align4(rows):
    return (rows + 3) // 4 * 4


def poison(shape, dtype):
    width = torch.empty(0, dtype=dtype).element_size()
    bytes_ = (*shape[:-1], shape[-1] * width)
    return torch.full(bytes_, POISON, device="cuda", dtype=torch.uint8).view(dtype)


def rows_and_strides(dim):
    return [(rows, stride) for rows in (1, 5, 17, 65, CAPACITY) for stride in (dim, 1792)]


def norm_quant(aux, stage, dim=1280, eps=1e-20):
    """q_norm + wq_b activation quantization."""
    sf_stride = align4(CAPACITY)
    for rows, stride in rows_and_strides(dim):
        x = torch.randn(CAPACITY, stride, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(dim, device="cuda", dtype=torch.bfloat16)
        want = [poison((CAPACITY, dim), torch.bfloat16), poison((CAPACITY, dim), torch.uint8),
                poison((dim // 128, sf_stride), torch.int32)]
        got = [t.clone() for t in want]
        stage_x = x[:, :dim].contiguous() if stride != dim else x
        aux("dsv41_norm", rows, want[0], stage_x, weight, rows, dim, eps)
        stage("dsv41_dense_quant_x", -(-align4(rows) * (dim // 128) // 8),
              want[0], want[1], want[2], rows, dim, dim, sf_stride)
        aux("dsv41_norm_quant", align4(rows), got[0], got[1], got[2], x, weight,
            rows, dim, stride, sf_stride, eps)
        equal = [torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(want, got)]
        assert all(equal), (rows, stride, equal)
        print("norm_quant", rows, stride, "byte-equal", flush=True)


def norm_rope(aux, dim=512, rope_dim=64, eps=1e-20):
    """kv_norm + kv_rope."""
    for rows, stride in rows_and_strides(dim):
        x = torch.randn(CAPACITY, stride, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(dim, device="cuda", dtype=torch.bfloat16)
        cos_sin = torch.randn(4096, rope_dim, device="cuda", dtype=torch.float32)
        positions = torch.randint(0, 4096, (CAPACITY,), device="cuda", dtype=torch.int32)
        want, got = poison((CAPACITY, dim), torch.bfloat16), poison((CAPACITY, dim), torch.bfloat16)
        normed = poison((CAPACITY, dim), torch.bfloat16)
        aux("dsv41_norm", rows, normed, x[:, :dim].contiguous() if stride != dim else x,
            weight, rows, dim, eps)
        aux("dsv41_rope", rows, want, normed, cos_sin, positions, rows, 1, dim, rope_dim, 0)
        aux("dsv41_norm_rope", rows, got, x, weight, cos_sin, positions,
            rows, dim, rope_dim, stride, 0, eps)
        assert torch.equal(want.view(torch.uint8), got.view(torch.uint8)), (rows, stride)
        print("norm_rope", rows, stride, "byte-equal", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("auxiliary", type=Path)
    p.add_argument("stage", type=Path)
    args = p.parse_args()
    torch.manual_seed(17)
    torch.zeros(1, device="cuda")
    aux, stage = launcher(args.auxiliary), launcher(args.stage)
    norm_quant(aux, stage)
    norm_rope(aux)


if __name__ == "__main__":
    main()
