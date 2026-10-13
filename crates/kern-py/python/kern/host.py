"""A manifest's contract as torch sees it.

The manifest JSON names every buffer's dtype and shape (dims may be vars,
read at their max, or constants); `region` turns one into a view over the
runtime's memory, which an engine writes inputs into and reads outputs
out of in its own stream order. `host_region` is the other direction: an
engine tensor described the way `Runtime.bind_host` takes it.
"""
import torch

DTYPES = {"bf16": torch.bfloat16, "u8": torch.uint8, "f32": torch.float32, "i32": torch.int32, "i64": torch.int64}


def resolve(m: dict, dim):
    return m.get("constants", {}).get(dim, dim) if isinstance(dim, str) else dim


class Bytes:
    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (ptr, False), "version": 3}


def region(rt, m: dict, name: str) -> torch.Tensor:
    """Buffer `name` as a tensor of its declared dtype, at its largest shape."""
    ptr, nbytes = rt.region(name)
    b = m["buffers"][name]
    shape = [resolve(m, m["vars"][d]["max"] if d in m.get("vars", {}) else d) for d in b["shape"]]
    return torch.as_tensor(Bytes(ptr, nbytes), device="cuda").view(DTYPES[b["dtype"]]).view(*shape)


def host_region(t: torch.Tensor) -> tuple:
    """`(address, dtype, shape, element strides)` of `t`, as `bind_host` takes a host state."""
    name = {v: k for k, v in DTYPES.items()}[t.dtype]
    return t.data_ptr(), name, list(t.shape), list(t.stride())


def once(m: dict) -> list[str]:
    """The programs run once after load, in name order."""
    return sorted(n for n, p in m["programs"].items() if p.get("once"))
