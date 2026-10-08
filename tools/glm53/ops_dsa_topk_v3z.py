"""DSA kpool topk v3z (cluster z-split) — build, byte-exact gates vs pinned
glm53_dsa_topk_v2, and isolated bench. Design: memory
glm53-zsplit-dsmem-design-20260928.md (Tier 1). New file; no shared files touched.

  build   nvcc -> glm53_dsa_topk_v3z.cubin
  test    adversarial matrix, byte-exact vs pinned v2 + determinism
  bench   isolated timing v2 vs v3z
"""
import os
import argparse, ctypes as C, hashlib, json, os, subprocess, sys
from pathlib import Path

NVCC = os.environ.get("NVCC", "/usr/local/cuda-13.0/bin/nvcc")
HERE = Path("/tmp/glm53-topkv3z")
SRC_REL = "tools/glm53/kernels/glm53_dsa_topk_v3z.cu"
CUBIN = HERE / "glm53_dsa_topk_v3z.cubin"
STAMP = HERE / "build_v3z.json"
V2CUBIN = HERE / "glm53_dsa_topk_v2.cubin"          # pinned reference (sha below)
V2_SHA = "3bc73eed3c8e4246828b2641d277fc3059e331bff1a904a45ecf705960c4a5a1"
REPO = Path(os.environ.get("KERN_REPO", Path(__file__).resolve().parents[2]))


def _sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def build():
    HERE.mkdir(parents=True, exist_ok=True)
    src = REPO / SRC_REL
    cmd = [NVCC, "-cubin", "-arch=sm_90a", "-std=c++17", "-O3",
           "-ccbin", "/usr/bin/g++-14", str(src), "-o", str(CUBIN)]
    subprocess.run(cmd, check=True)
    STAMP.write_text(json.dumps({"source": str(src), "source_sha256": _sha(src),
                                 "cubin_sha256": _sha(CUBIN), "cmd": cmd}, indent=1))
    print(f"built {CUBIN} sha {_sha(CUBIN)[:16]}")

def check_build():
    if not CUBIN.exists() or _sha(CUBIN) != json.loads(STAMP.read_text())["cubin_sha256"]:
        build()
    return json.loads(STAMP.read_text())


# ---------------- GPU plumbing (self-contained Driver, slab-harness pattern) ----
class Driver:
    def __init__(self, torch):
        self.torch = torch
        self.lib = C.CDLL("libcuda.so.1")
        self.lib.cuModuleLoad.argtypes = [C.POINTER(C.c_void_p), C.c_char_p]
        self.lib.cuModuleGetFunction.argtypes = [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p]
        self.lib.cuLaunchKernel.argtypes = [C.c_void_p] + [C.c_uint]*7 + [C.c_void_p]*3
        self.lib.cuLaunchKernelEx.argtypes = [C.POINTER(LaunchConfig), C.c_void_p,
                                              C.POINTER(C.c_void_p), C.c_void_p]
    @staticmethod
    def check(rc, what=""):
        if rc: raise RuntimeError(f"CUDA error {rc} {what}")
    def module(self, path):
        m = C.c_void_p(); self.check(self.lib.cuModuleLoad(C.byref(m), str(path).encode()), "load")
        return m
    def func(self, mod, entry):
        fn = C.c_void_p(); self.check(self.lib.cuModuleGetFunction(C.byref(fn), mod, entry.encode()), entry)
        return fn

class LaunchAttr(C.Structure):
    _fields_ = [("id", C.c_uint), ("_pad", C.c_uint), ("value", C.c_uint64 * 8)]
class LaunchConfig(C.Structure):
    _fields_ = [("gridX", C.c_uint), ("gridY", C.c_uint), ("gridZ", C.c_uint),
                ("blockX", C.c_uint), ("blockY", C.c_uint), ("blockZ", C.c_uint),
                ("smem", C.c_uint), ("stream", C.c_void_p),
                ("attrs", C.POINTER(LaunchAttr)), ("nattrs", C.c_uint)]

def launch(D, fn, grid, block, args, cluster=None):
    vals = [C.c_uint64(x.data_ptr()) if hasattr(x, "data_ptr") else C.c_uint64(x)
            for x in args]
    av = (C.c_void_p * len(vals))(*[C.addressof(x) for x in vals])
    stream = C.c_void_p(D.torch.cuda.current_stream().cuda_stream)
    if cluster is None:
        D.check(D.lib.cuLaunchKernel(fn, grid[0], grid[1], grid[2], block, 1, 1, 0,
                                     stream, av, None), "launch")
    else:
        attr = LaunchAttr(); attr.id = 2  # CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        attr.value[0] = cluster[0]; attr.value[1] = cluster[1]; attr.value[2] = cluster[2]
        cfg = LaunchConfig(grid[0], grid[1], grid[2], block, 1, 1, 0, stream,
                           C.pointer(attr), 1)
        D.check(D.lib.cuLaunchKernelEx(C.byref(cfg), fn, av, None), "launchEx")
    launch._keep = (vals, av)


# ---------------- test data ----------------
def align256(n): return (n + 255) // 256 * 256

def gen_logits(torch, lens, dist, gen):
    """[rows, stride] fp32; stride covers max len; +inf sentinel beyond each len."""
    rows = len(lens); stride = align256(max(max(lens), 1) + 1)
    x = torch.full((rows, stride), float("inf"))
    for r, ln in enumerate(lens):
        if ln == 0: continue
        g = torch.Generator(device="cpu"); g.manual_seed(gen + 1000 * r)
        if dist == "randn":
            v = torch.randn(ln, generator=g)
        elif dist == "equal":
            v = torch.ones(ln)
        elif dist == "cutdup":  # 60% identical value straddling rank 512
            n_same = (ln * 3) // 5
            v = torch.cat([torch.full((n_same,), 2.0),
                           torch.rand(ln - n_same, generator=g) * 2 - 1])
            v = v[torch.randperm(ln, generator=g)]
        elif dist == "pmzero":
            v = torch.randn(ln, generator=g) * 1e-8
            v[::7] = 0.0; v[::11] = -0.0
        elif dist == "asc":
            v = torch.linspace(-10, 10, ln)
        elif dist == "desc":
            v = torch.linspace(10, -10, ln)
        elif dist == "nan":
            v = torch.randn(ln, generator=g); v[::17] = float("nan")
        elif dist == "denorm":
            v = torch.rand(ln, generator=g) * 1e-38
        else:
            raise ValueError(dist)
        x[r, :ln] = v
    return x.cuda()

def gen_case(torch, rows, lens, dist, seed, tail_mode="rand"):
    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    tails = [int(torch.randint(0, 4, (1,), generator=g)) if tail_mode == "rand"
             else (seed + r) % 4 for r in range(rows)]
    seq = torch.tensor([ln * 4 + tails[r] for r, ln in enumerate(lens)], dtype=torch.int32)
    bt_cols = max((max(lens) * 4 + 3 + 255) // 256, 1)
    bt = torch.randint(0, 1 << 20, (rows, bt_cols), dtype=torch.int64,
                       generator=g).to(torch.int32)
    return (gen_logits(torch, lens, dist, seed),
            torch.tensor(lens, dtype=torch.int32).cuda(), bt.cuda(), seq.cuda())

def run_one(D, torch, fn, cluster, case):
    logits, pools, bt, seq = case
    rows = pools.numel()
    dst = torch.full((rows, 2051), 0x7f7f7f7f, dtype=torch.int32, device="cuda")
    launch(D, fn, (rows, cluster, 1), 1024 if cluster == 1 else 256,
           [logits, pools, bt, seq, dst, logits.shape[1], bt.shape[1]],
           cluster=None if cluster == 1 else (1, 8, 1))
    return dst


def test(args):
    import torch
    torch.cuda.init(); torch.zeros(1, device="cuda")
    D = Driver(torch)
    v3z_stamp = check_build()
    if _sha(V2CUBIN) != V2_SHA:
        raise RuntimeError(f"pinned v2 cubin sha mismatch: {_sha(V2CUBIN)[:16]} != {V2_SHA[:16]}")
    m2, m3 = D.module(V2CUBIN), D.module(CUBIN)
    f2 = D.func(m2, "glm53_dsa_topk_v2")
    f3 = D.func(m3, "glm53_dsa_topk_v3z")
    print(f"v3z cubin sha {v3z_stamp['cubin_sha256'][:16]}; v2 pinned {V2_SHA[:16]}")

    fails, total = [], 0
    def gate(name, case):
        nonlocal total; total += 1
        d2 = run_one(D, torch, f2, 1, case)
        d3 = run_one(D, torch, f3, 8, case)
        d3b = run_one(D, torch, f3, 8, case)   # determinism
        torch.cuda.synchronize()
        ok_ref = torch.equal(d2, d3)
        ok_det = torch.equal(d3, d3b)
        if not (ok_ref and ok_det):
            bad = (d2 != d3).nonzero()
            fails.append((name, ok_ref, ok_det,
                          bad.shape[0], bad[:5].tolist(),
                          d2[bad[0, 0], bad[0, 1]].item() if bad.shape[0] else None,
                          d3[bad[0, 0], bad[0, 1]].item() if bad.shape[0] else None))
        print(f"  {'PASS' if ok_ref and ok_det else 'FAIL'} {name}")

    COUNTS = [0, 1, 511, 512, 513, 1000, 4095, 4096, 4097, 35280, 65535]
    print("== A: single-row pool-count matrix ==")
    for ln in COUNTS:
        for dist in (["randn", "equal", "cutdup", "asc", "pmzero"] +
                     (["nan"] if ln in (513, 35280) else []) +
                     (["denorm", "desc"] if ln in (1000, 35280) else [])):
            gate(f"A len={ln} {dist}", gen_case(torch, 1, [ln], dist, args.seed + ln))

    print("== B: mixed per-row lens in one launch ==")
    mixed = [0, 1, 2, 511, 512, 513, 1000, 2048, 4095, 4096, 4097, 8192, 20000, 35280, 65535, 3]
    for dist in ("randn", "cutdup"):
        gate(f"B mixed {dist}", gen_case(torch, len(mixed), mixed, dist, args.seed + 7))

    print("== C: T=3S verify shapes ==")
    for S in (1, 2, 3, 4, 5, 8, 15, 16):
        rows = 3 * S
        g = torch.Generator(device="cpu"); g.manual_seed(args.seed + 100 * S)
        lens = [int(torch.randint(513, 35281, (1,), generator=g)) for _ in range(rows)]
        lens[0] = 300  # one identity row mixed in
        if rows > 2: lens[2] = 512
        for dist in ("randn", "cutdup"):
            gate(f"C S={S} T={rows} {dist}", gen_case(torch, rows, lens, dist, args.seed + S))

    print("== D: repeated-launch determinism x20 (largest case) ==")
    case = gen_case(torch, 48, [35280] * 48, "randn", args.seed + 999)
    base = run_one(D, torch, f3, 8, case); torch.cuda.synchronize()
    for i in range(19):
        d = run_one(D, torch, f3, 8, case); torch.cuda.synchronize()
        total += 1
        if not torch.equal(base, d): fails.append((f"D iter{i}", False, False, -1, [], None, None))
    print("  PASS D" if not any(f[0].startswith("D ") for f in fails) else "  FAIL D")

    print(f"\n{'ALL PASS' if not fails else 'FAILURES'}: {total - len(fails)}/{total}")
    for f in fails[:12]: print("  FAIL", f)
    sys.exit(1 if fails else 0)


def bench(args):
    import torch
    torch.cuda.init(); torch.zeros(1, device="cuda")
    D = Driver(torch)
    check_build()
    m2, m3 = D.module(V2CUBIN), D.module(CUBIN)
    f2 = D.func(m2, "glm53_dsa_topk_v2")
    f3 = D.func(m3, "glm53_dsa_topk_v3z")

    def time_it(fn, cluster, case, iters=100):
        for _ in range(10): run_one(D, torch, fn, cluster, case)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters): run_one(D, torch, fn, cluster, case)
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e) * 1000 / iters  # us

    print(f"{'rows':>5} {'len':>7} {'v2 us':>9} {'v3z us':>9} {'speedup':>8}")
    for rows, ln in [(1, 35280), (3, 35280), (12, 35280), (24, 35280), (48, 35280),
                     (12, 513), (12, 4096), (48, 65535), (12, 20000)]:
        case = gen_case(torch, rows, [ln] * rows, "randn", args.seed + rows)
        t2 = time_it(f2, 1, case)
        t3 = time_it(f3, 8, case)
        print(f"{rows:>5} {ln:>7} {t2:>9.2f} {t3:>9.2f} {t2/t3:>7.2f}x")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "test", "bench"])
    ap.add_argument("--seed", type=int, default=555555)
    args = ap.parse_args()
    {"build": build, "test": test, "bench": bench}[args.cmd](args) if args.cmd != "build" else build()
