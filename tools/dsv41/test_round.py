"""Exercise the existing device glue with five draft / six verify rows."""

import argparse
import ctypes
from pathlib import Path

import torch
from cuda.bindings import driver as cu


def checked(result):
    if result[0] != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(str(result[0]))
    return result[1] if len(result) == 2 else result[1:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cubin", type=Path, required=True)
    args = parser.parse_args()
    torch.empty(1, device="cuda")
    module = checked(cu.cuModuleLoad(str(args.cubin).encode()))

    def launch(name, batch, *values):
        function = checked(cu.cuModuleGetFunction(module, name.encode()))
        params = tuple(v.data_ptr() if isinstance(v, torch.Tensor) else v for v in values)
        types = tuple(ctypes.c_void_p if isinstance(v, torch.Tensor) else
                      ctypes.c_longlong if name == "kern_splice_draft" and i == 3 else
                      ctypes.c_int for i, v in enumerate(values))
        checked(cu.cuLaunchKernel(function, batch, 1, 1, 32, 1, 1, 0,
                                 cu.CUstream(torch.cuda.current_stream().cuda_stream),
                                 (params, types), 0))

    # One sequence for each possible accepted length, plus a padded-sized batch.
    for batch in (6, 17, 128):
        anchor = torch.arange(batch, device="cuda", dtype=torch.int64) + 100
        draft_ids = torch.empty((batch, 5), device="cuda", dtype=torch.int64)
        launch("kern_splice_draft", batch, anchor, draft_ids, 5, 128799)
        torch.testing.assert_close(draft_ids[:, 0], anchor, rtol=0, atol=0)
        assert (draft_ids[:, 1:] == 128799).all().item()
        predictions = torch.arange(batch * 5, device="cuda", dtype=torch.int64).view(batch, 5)
        verify_ids = torch.empty((batch, 6), device="cuda", dtype=torch.int64)
        launch("kern_splice_verify", batch, anchor, predictions, verify_ids, 6, 5)
        torch.testing.assert_close(verify_ids[:, 0], anchor, rtol=0, atol=0)
        torch.testing.assert_close(verify_ids[:, 1:], predictions, rtol=0, atol=0)
        logits_argmax = torch.cat((predictions.clone(), anchor[:, None]), dim=1)
        matched = torch.arange(batch, device="cuda") % 6
        for row in range(batch):
            length = row % 6
            if length < 5:
                logits_argmax[row, length] = -1
        count = torch.empty(batch, device="cuda", dtype=torch.int32)
        launch("kern_spec_count", batch, predictions, logits_argmax, count, 6, 5)
        torch.testing.assert_close(count, (matched + 1).int(), rtol=0, atol=0)
        print(f"batch={batch}: draft5/verify6 splice and all accepted lengths 1..6 PASS")
    checked(cu.cuModuleUnload(module))


if __name__ == "__main__":
    main()
