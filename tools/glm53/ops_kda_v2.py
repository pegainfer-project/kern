"""Opt-in GLM-5.3 KDA fusion. No default generator or bundle is modified.

CPU build: PYTHONPATH=tools python -m glm53.ops_kda_v2 build
GPU A/B:   PYTHONPATH=tools python -m glm53.ops_kda_v2 test --batches 1,4,8,16
See docs/glm53/kda_fusion.md for the integration hook and acceptance gates.

MTP verify (rows=3) fusion + advance 3->1 select (v3):
  CPU build:  PYTHONPATH=tools python -m glm53.ops_kda_v2 build-v3
  GPU gates:  PYTHONPATH=tools python -m glm53.ops_kda_v2 test-v3
  Manifest:   PYTHONPATH=tools python -m glm53.ops_kda_v2 fuse-round in.json out.json
Site-124 note: verify reads committed conv taps without stores and the advance
is a pure copy, so per-row stash + select is arithmetic-free (bitwise gate 1).
"""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
NVCC = os.environ.get("NVCC", "/usr/local/cuda-13.0/bin/nvcc")
CCBIN = os.environ.get("NVCC_CCBIN", "/usr/bin/g++-14")
ART = Path(os.environ.get("GLM53_KDA_ARTIFACTS", "/tmp/glm53-kda-v2")).expanduser().resolve()
CUBIN = ART / "glm53_kda_v2.cubin"
SOURCE = Path(__file__).with_name("kernels") / "glm53_kda_v2.cu"
V3ART = Path(os.environ.get("GLM53_KDA_V3_ARTIFACTS", "/tmp/glm53-kda-v3")).expanduser().resolve()
V3CUBIN = V3ART / "glm53_kda_v3.cubin"
V3SOURCE = Path(__file__).with_name("kernels") / "glm53_kda_v3.cu"
SPECART = Path(os.environ.get("GLM53_SPEC_ARTIFACTS", "glm53-artifacts/mtp")).resolve()


def build():
    ART.mkdir(parents=True, exist_ok=True)
    cmd = [NVCC, "-cubin", "-arch=sm_90a",
           "-std=c++17", "-O3", "-ccbin", CCBIN, "-Xptxas=-v",
           str(SOURCE), "-o", str(CUBIN)]
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (ART / "build.log").write_text(" ".join(cmd) + "\n" + p.stdout)
    print(p.stdout, end="")
    p.check_returncode()
    stamp={"source_sha256":hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
           "cubin_sha256":hashlib.sha256(CUBIN.read_bytes()).hexdigest(),"command":cmd}
    (ART/"build.json").write_text(json.dumps(stamp,indent=2))
    print("sha256",stamp["cubin_sha256"])


def check_build():
    stamp=json.loads((ART/"build.json").read_text())
    if stamp["source_sha256"]!=hashlib.sha256(SOURCE.read_bytes()).hexdigest() or stamp["cubin_sha256"]!=hashlib.sha256(CUBIN.read_bytes()).hexdigest():
        raise RuntimeError("stale build: run the CPU-only build command first")
    return stamp


def build_v3():
    V3ART.mkdir(parents=True, exist_ok=True)
    cmd = [NVCC, "-cubin", "-arch=sm_90a",
           "-std=c++17", "-O3", "-ccbin", CCBIN, "-Xptxas=-v",
           str(V3SOURCE), "-o", str(V3CUBIN)]
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (V3ART / "build.log").write_text(" ".join(cmd) + "\n" + p.stdout)
    print(p.stdout, end="")
    p.check_returncode()
    stamp = {"source_sha256": hashlib.sha256(V3SOURCE.read_bytes()).hexdigest(),
             "cubin_sha256": hashlib.sha256(V3CUBIN.read_bytes()).hexdigest(), "command": cmd}
    (V3ART / "build.json").write_text(json.dumps(stamp, indent=2))
    print("v3 sha256", stamp["cubin_sha256"])


def check_build_v3():
    stamp = json.loads((V3ART/"build.json").read_text())
    if stamp["source_sha256"] != hashlib.sha256(V3SOURCE.read_bytes()).hexdigest() or \
       stamp["cubin_sha256"] != hashlib.sha256(V3CUBIN.read_bytes()).hexdigest():
        raise RuntimeError("stale v3 build: run `ops_kda_v2 build-v3` first")
    return stamp


# Stash geometry: one [3,8,128,128] f32 SSM row-state plus one [3,3,3072] bf16
# conv row-state per (layer, sequence). Sized for the round's groups<=8.
STASH_SEQS = 8
SSM_LINE_F32 = 131072           # [8,128,128] f32
CONV_LINE_BF16 = 9216           # [3,3072] bf16


def ops_v3(mode="core", threads=512, rows=3):
    """Round (rows=N) verify+stash and select ops. Integrate like ops():
    after resolve_modules, before lower_wire. Uses the v3 cubin."""
    if mode not in ("core", "fused") or threads not in (256, 512, 1024):
        raise ValueError("mode core/fused, threads 256/512/1024 required")
    if not 1 <= rows <= 8:
        raise ValueError("v3 verify rows must be 1..8 (one fg mma tile)")
    stamp = check_build_v3()
    bf, fp, ix = "in buffer<bf16>", "in buffer<f32>", "in buffer<i32>"
    sha = stamp["cubin_sha256"]
    def launch(entry, p, grid, block):
        return {"cubin": str(V3CUBIN), "sha256": sha, "label": entry, "entry": entry,
                "params": p.copy(), "block": [block, 1, 1], "grid": grid,
                "args": [{"param": i} for i in range(len(p))], "pdl": True}
    params = [bf, bf, bf, fp, fp, fp, bf, "in state", "in state", ix, ix, ix,
              "out buffer<bf16>", "out buffer<f32>", "out buffer<bf16>", "i32", "i32"]
    out = {"spec_kda_fused_v3": {"params": params, "impl": {"launches": [
        launch(f"glm53_kda_verify_{mode}{threads}", params, ["seqs", 8, 1], threads)]}}}
    sp = [fp, bf, "inout state", "inout state", ix, ix, ix, ix, "i32"]
    out["spec_kda_select"] = {"params": sp, "impl": {"launches": [
        launch("glm53_kda_select", sp, ["seqs", 8, 1], 256)]}}
    return out


def fuse_round_manifest(m, mode="core", threads=512, stash_seqs=STASH_SEQS, program="round_k2"):
    """Rewrite the round program's KDA groups to the fused verify+stash + select form.

    Per KDA layer: [fg_b, conv_verify, delta_verify, norm_gated] becomes one
    spec_kda_fused_v3 call (core mode retains the fg_b call), and
    [pack_kda, delta_advance, conv_advance] becomes one spec_kda_select call.
    Call-count delta: -5/layer fused, -4/layer core (x34 = -170/-136).
    """
    if mode not in ("core", "fused") or threads not in (256, 512, 1024):
        raise ValueError("mode core/fused, threads 256/512/1024 required")
    if program not in m["programs"]:
        raise ValueError(f"no {program} program: not an --mtp manifest")
    rows = m["programs"][program].get("batch", {}).get("rows")
    if not isinstance(rows, int) or not 1 <= rows <= 8:
        raise ValueError(
            f"spec_kda_fused_v3/spec_kda_select are rows=N ABI (1..8, glm53_kda_v3.cu: "
            f"fgN mma B-columns, gates[8]/qkv[8]/raw[8] shared arrays, rows*s row "
            f"pitches, stash dims [34,S,rows,...]); {program} has rows={rows}.")
    ssm_off = stash_seqs * rows * SSM_LINE_F32 * 4
    conv_off = stash_seqs * rows * CONV_LINE_BF16 * 2
    m["buffers"]["spec_ssm_stash"] = {"dtype": "f32", "kind": "workspace",
        "shape": [34, stash_seqs, rows, 8, 128, 128]}
    m["buffers"]["spec_conv_stash"] = {"dtype": "bf16", "kind": "workspace",
        "shape": [34, stash_seqs, rows, 3, 3072]}
    calls = m["programs"][program]["calls"]
    rewritten, dead = [], set()
    i = nv = na = 0
    while i < len(calls):
        if calls[i]["op"] == "mtp32_kda_fg_b":
            group = calls[i:i+4]
            if [c["op"] for c in group] != ["mtp32_kda_fg_b", "spec_conv_verify",
                                            "spec_delta_verify", "mtp32_kda_norm_gated"]:
                raise ValueError(f"KDA verify cut changed at {calls[i]['label']}; refusing implicit fuse")
            fg, cv, de, no = [c["args"] for c in group]
            assert fg[0] == cv[0] == de[4], "spec_F identity mismatch"
            assert fg[3] == de[1] and cv[4] == de[3] and de[5] == no[0] and fg[4] == no[1]
            koff = cv[3].get("offset", 0)
            assert de[7].get("offset", 0) == koff and koff % 64 == 0
            k = koff // 64
            if mode == "core":
                rewritten.append(group[0]); f_in, g_in = fg[3], fg[4]
            else:
                f_in, g_in = fg[1], fg[2]
                dead.update(x["buf"] for x in (fg[3], fg[4]))
            dead.update(x["buf"] for x in (cv[4], no[3]))
            args = [fg[0], f_in, g_in, cv[1], de[0], de[2], no[2], cv[2], de[6],
                    cv[3], de[7], de[8], de[5],
                    {"buf": "spec_ssm_stash", "offset": k * ssm_off},
                    {"buf": "spec_conv_stash", "offset": k * conv_off},
                    {"var": "seqs"}, {"i32": rows}]
            rewritten.append({"label": group[0]["label"].replace("fg_b", "fused_v3"),
                              "op": "spec_kda_fused_v3", "args": args})
            nv += 1; i += 4
        elif calls[i]["op"] == "spec_pack_kda":
            group = calls[i:i+3]
            if [c["op"] for c in group] != ["spec_pack_kda", "spec_delta_advance", "spec_conv_advance"]:
                raise ValueError(f"KDA advance cut changed at {calls[i]['label']}; refusing implicit fuse")
            pk, da, ca = [c["args"] for c in group]
            koff = ca[2].get("offset", 0)
            assert da[7].get("offset", 0) == koff and koff % 64 == 0
            k = koff // 64
            assert pk[0] == ca[0], "advance F identity mismatch"
            dead.update(x["buf"] for x in (pk[3], pk[4], pk[5], da[5]))
            args = [{"buf": "spec_ssm_stash", "offset": k * ssm_off},
                    {"buf": "spec_conv_stash", "offset": k * conv_off},
                    da[6], ca[1], da[7], ca[2], ca[3], ca[4], {"i32": rows}]
            rewritten.append({"label": group[0]["label"].replace("pack", "select"),
                              "op": "spec_kda_select", "args": args})
            na += 1; i += 3
        else:
            rewritten.append(calls[i]); i += 1
    if nv != 34 or na != 34:
        raise ValueError(f"expected 34 verify + 34 advance groups, got {nv}/{na}")
    for call in rewritten:
        if any(arg.get("buf") in dead for arg in call["args"]):
            raise ValueError(f"live consumer of removed KDA cut: {call['label']}")
    m["ops"].update(ops_v3(mode, threads, rows))
    m["programs"][program]["calls"] = rewritten
    # Mirror ops_spec's pruning: drop ops and buffers the rewrite made unused.
    used = {c["op"] for p in m["programs"].values() for c in p["calls"]}
    m["ops"] = {n: o for n, o in m["ops"].items() if n in used}
    used_buf = {a["buf"] for p in m["programs"].values() for c in p["calls"]
                for a in c["args"] if "buf" in a}
    peer_of = {m["buffers"][n]["of"] for n in used_buf if m["buffers"][n]["kind"] == "peer"}
    used_buf |= peer_of
    m["buffers"] = {n: v for n, v in m["buffers"].items()
                    if n in used_buf or v["kind"] == "input"}
    used_mod = {l["module"] for o in m["ops"].values() for l in o["impl"]["launches"]
                if "module" in l}
    m["modules"] = {n: v for n, v in m["modules"].items() if n in used_mod}
    return m


def ops(mode="core", threads=512, direct=False):
    """Return only new ops; integrate AFTER gen.resolve_modules (absolute path)."""
    if mode not in ("core", "fused") or threads not in (128, 256, 512, 1024):
        raise ValueError("mode core/fused, threads 128/256/512/1024 required")
    check_build()
    bf, fp, ix = "in buffer<bf16>", "in buffer<f32>", "in buffer<i32>"
    params = [bf, bf, bf, fp, fp, fp, bf, "inout state", "inout state",
              ix, ix, ix, "out buffer<bf16>", "i32"]
    def launch(entry, p, grid, block):
        return {"cubin": str(CUBIN), "sha256": hashlib.sha256(CUBIN.read_bytes()).hexdigest(),
                "label": entry, "entry": entry, "params": p.copy(), "block": [block, 1, 1],
                "grid": grid, "args": [{"param": i} for i in range(len(p))], "pdl": True}
    out = {"kda_fused_v2": {"params": params, "impl": {"launches": [
        launch(f"glm53_kda_{mode}{threads}", params, ["seqs", 8, 1], threads)]}}}
    if direct:
        p = [bf, bf, "out buffer<bf16>", "i32", "i32", "i32"]
        out["kda_qkvbfg"] = {"params": p, "impl": {"launches": [
            launch("glm53_kda_qkv_direct", p, [834, 1, 1], 128)]}}
    return out


def fuse_manifest(m, mode="core", threads=512, direct=False):
    """Call after resolve_modules and before lower_wire. Keep o_proj/AR intact.

    core: retain two original fg_b GEMMs; replace conv/delta/norm with one CTA.
    fused: also absorb fg_b, subject to the independent projection parity gate.
    """
    if m["programs"]["decode"]["batch"] != {"groups": 16, "rows": 1}:
        raise ValueError("KDA v2 supports decode groups<=16, rows=1 only")
    calls = m["programs"]["decode"]["calls"]
    rewritten = []
    i = 0
    count = 0
    dead = set()
    while i < len(calls):
        if calls[i]["op"] != "kda_fg_b":
            rewritten.append(calls[i]); i += 1; continue
        group = calls[i:i+4]
        if [c["op"] for c in group] != ["kda_fg_b", "kda_conv", "kda_delta", "kda_norm_gated"]:
            raise ValueError("KDA cut changed or probes interposed; do not fuse implicitly")
        fg, cv, de, no = [c["args"] for c in group]
        # Validate identities; this rejects a silent layout/view mismatch.
        assert fg[0] == cv[0] == de[4] and fg[3] == de[1]
        assert cv[4] == de[3] and de[5] == no[0] and fg[4] == no[1]
        if mode == "core":
            rewritten.append(group[0]); f_in, g_in = fg[3], fg[4]
        else:
            f_in, g_in = fg[1], fg[2]
        dead.update(x["buf"] for x in (cv[4], no[3]))
        if mode == "fused": dead.update(x["buf"] for x in (fg[3],fg[4]))
        args = [fg[0], f_in, g_in, cv[1], de[0], de[2], no[2], cv[2], de[6],
                cv[3], de[7], de[8], de[5], fg[5]]
        rewritten.append({"label": group[0]["label"].replace("fg_b", "fused_v2"),
                          "op": "kda_fused_v2", "args": args})
        count += 1; i += 4
    if not count:
        raise ValueError("no unfused KDA calls found")
    for call in rewritten:
        if any(arg.get("buf") in dead for arg in call["args"]):
            raise ValueError(f"live consumer of removed KDA cut: {call['label']}")
    m["ops"].update(ops(mode, threads, direct))
    m["programs"]["decode"]["calls"] = rewritten
    return m


def gpu_check():
    """No GPU initialization before checking shared-box use. Never kill others."""
    for cmd in (["tmux", "ls"], ["tmux", "list-panes", "-a", "-F", "#S #{pane_current_command}"],
                ["nvidia-smi"]):
        try: subprocess.run(cmd, check=False)
        except FileNotFoundError: pass
    details = subprocess.check_output(["ps", "-eo", "comm=,args="], text=True)
    if any(line.split()[0].startswith("python") and ("sglang.launch_server" in line or "integration" in line) for line in details.splitlines() if line.split()):
        raise RuntimeError("GPU run deferred: serving/integration Python process is active")
    ps = subprocess.check_output(["ps", "-eo", "comm="], text=True)
    if any(line.strip() in ("kern", "kern-run", "kern-serve", "kbench", "kserve") for line in ps.splitlines()):
        raise RuntimeError("GPU run deferred: kern bench/serve process is active")
    gpu = os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    if not gpu.isdigit():
        raise RuntimeError("use a single numeric CUDA_VISIBLE_DEVICES GPU for this test")
    stats = subprocess.check_output(["nvidia-smi", "-i", gpu,
        "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    mem, util = map(int, stats.strip().split(","))
    if mem > 100 or util > 0:
        raise RuntimeError(f"GPU {gpu} busy ({stats.strip()}); wait and retry")


class Driver:
    def __init__(self, torch):
        self.torch = torch
        self.lib = C.CDLL("libcuda.so.1")
        self.modules = []
        self.lib.cuModuleLoad.argtypes = [C.POINTER(C.c_void_p), C.c_char_p]
        self.lib.cuModuleGetFunction.argtypes = [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p]
        self.lib.cuLaunchKernel.argtypes = [C.c_void_p] + [C.c_uint]*7 + [C.c_void_p, C.c_void_p, C.c_void_p]

    @staticmethod
    def check(rc):
        if rc: raise RuntimeError(f"CUDA/library error {rc}")

    def module(self, path):
        m = C.c_void_p(); self.check(self.lib.cuModuleLoad(C.byref(m), str(path).encode()))
        self.modules.append(m); return m

    def kernel(self, module, name, grid, block, args, smem=0):
        fn = C.c_void_p(); self.check(self.lib.cuModuleGetFunction(C.byref(fn), module, name.encode()))
        vals = [C.c_uint64(x.data_ptr()) if hasattr(x, "data_ptr") else x for x in args]
        av = (C.c_void_p*len(vals))(*[C.addressof(x) for x in vals])
        def launch():
            self.check(self.lib.cuLaunchKernel(fn, *grid, block, 1, 1, smem,
                C.c_void_p(self.torch.cuda.current_stream().cuda_stream), av, None))
        launch._keep = (vals, av, args)
        return launch


class Lt:
    """Same CUBLAS_COMPUTE_32F and 32 MiB workspace as cudarc on H100."""
    class Result(C.Structure):
        _fields_ = [("algo", C.c_uint64*8), ("workspace", C.c_size_t), ("state", C.c_int),
                    ("waves", C.c_float), ("reserved", C.c_int*4)]

    def __init__(self, torch):
        self.t = torch
        try: self.lib = C.CDLL("libcublasLt.so.13")
        except OSError:
            import glob
            base = Path(torch.__file__).parent.parent
            cands = (glob.glob(str(base/"nvidia"/"cublas"/"lib"/"libcublasLt.so*"))
                     + glob.glob(str(base/"torch"/"lib"/"libcublasLt.so*")))
            self.lib = C.CDLL(cands[0])
        self.handle = C.c_void_p()
        Driver.check(self.lib.cublasLtCreate(C.byref(self.handle)))
        self.ws = torch.empty(32 << 20, device="cuda", dtype=torch.uint8)
        self.objects = []

    def gemm(self, x, w, y, M, N, K, stride=None):
        L = self.lib; ck = Driver.check
        d, a, b, c, p = [C.c_void_p() for _ in range(5)]
        ck(L.cublasLtMatmulDescCreate(C.byref(d), C.c_int(68), C.c_int(0)))
        trans, nt = C.c_int(1), C.c_int(0)
        ck(L.cublasLtMatmulDescSetAttribute(d, C.c_int(3), C.byref(trans), C.c_size_t(4)))
        ck(L.cublasLtMatmulDescSetAttribute(d, C.c_int(4), C.byref(nt), C.c_size_t(4)))
        for dst, rows, cols, ld in ((a,K,N,K),(b,K,M,stride or K),(c,N,M,N)):
            ck(L.cublasLtMatrixLayoutCreate(C.byref(dst), C.c_int(14), C.c_uint64(rows),
                                            C.c_uint64(cols), C.c_int64(ld)))
        ck(L.cublasLtMatmulPreferenceCreate(C.byref(p)))
        ws = C.c_size_t(self.ws.numel())
        ck(L.cublasLtMatmulPreferenceSetAttribute(p, C.c_int(1), C.byref(ws), C.c_size_t(8)))
        r, n = self.Result(), C.c_int()
        ck(L.cublasLtMatmulAlgoGetHeuristic(self.handle,d,a,b,c,c,p,C.c_int(1),C.byref(r),C.byref(n)))
        if n.value != 1 or r.state: raise RuntimeError("no valid cuBLASLt heuristic")
        split, written = C.c_int(), C.c_size_t()
        ck(L.cublasLtMatmulAlgoConfigGetAttribute(C.byref(r.algo), C.c_int(2),
                                                C.byref(split), C.c_size_t(4), C.byref(written)))
        alpha, beta = C.c_float(1), C.c_float(0)
        def run():
            ck(L.cublasLtMatmul(self.handle,d,C.byref(alpha),C.c_void_p(w.data_ptr()),a,
                C.c_void_p(x.data_ptr()),b,C.byref(beta),C.c_void_p(y.data_ptr()),c,
                C.c_void_p(y.data_ptr()),c,C.byref(r.algo),C.c_void_p(self.ws.data_ptr()),ws,
                C.c_void_p(self.t.cuda.current_stream().cuda_stream)))
        run.info = {"split_k": split.value, "workspace": r.workspace, "waves": r.waves}
        self.objects.append((d,a,b,c,p,r,run))
        return run


def test(args):
    check_build()
    gpu_check()
    import torch
    torch.cuda.init(); torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    D, lt = Driver(torch), Lt(torch)
    mod = D.module(CUBIN)
    dump = ROOT / "dumped-kernels-glm53-sglang"
    conv = D.module(ROOT / "kernels-glm53-handwritten/module_conv_generic.cubin")
    delta = D.module(dump / "module_363.cubin")
    norm = D.module(dump / "module_227.cubin")
    I, J, F = C.c_int, C.c_int64, C.c_float
    bf, fp = torch.bfloat16, torch.float32
    def rnd(shape, scale=1, dtype=bf):
        return (torch.randn(shape, device="cuda", dtype=fp)*scale).to(dtype)
    def bits_equal(a,b):
        dtype = torch.int16 if a.element_size() == 2 else torch.int32
        return torch.equal(a.view(dtype),b.view(dtype))
    def stats(a,b):
        aa,bb=a.float(),b.float(); diff=(aa-bb).abs()
        return {"unequal": int((a!=b).sum()), "bit_equal": bits_equal(a,b), "max_abs": diff.max().item(),
                "rms_rel": (diff.square().mean().sqrt()/(aa.square().mean().sqrt()+1e-30)).item()}
    def timing(fn):
        for _ in range(3): fn()
        torch.cuda.synchronize()
        stream=torch.cuda.Stream()
        with torch.cuda.stream(stream):
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(args.repeats): fn()
        stream.synchronize()
        vals=[]
        for _ in range(5):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record(); graph.replay(); end.record(); end.synchronize()
            vals.append(start.elapsed_time(end)*1000/args.repeats)
        return sorted(vals)[2]
    results=[]
    for S in map(int,args.batches.split(",")):
        if not 1 <= S <= 16: raise ValueError("batch must be 1..16")
        # Different, non-contiguous conv/SSM line ids. Include >581 (old conv bug).
        cl=torch.tensor([600]+list(range(2,2*S,2)),dtype=torch.int32,device="cuda")
        sl=torch.tensor([601]+list(range(3,2*S+1,2)),dtype=torch.int32,device="cuda")
        cu=torch.arange(S+1,dtype=torch.int32,device="cuda")
        X=rnd((S,3336),.4); wf=rnd((1024,128),.08); wg=rnd((1024,128),.08)
        cw=rnd((3072,4),.4,fp); al=rnd((8,),.5,fp); dt=rnd((1024,),1,fp); nw=rnd((128,),.5)
        if args.layer is not None:
            from safetensors import safe_open
            cp=Path(args.checkpoint)
            index=json.loads((cp/"model.safetensors.index.json").read_text())["weight_map"]
            prefix=f"model.language_model.layers.{args.layer}.self_attn."
            def weight(name, start=None, count=None):
                key=prefix+name
                with safe_open(cp/index[key],framework="pt",device="cpu") as sf:
                    x=sf.get_tensor(key) if start is None else sf.get_slice(key)[start:start+count]
                return x.contiguous().cuda()
            off=args.rank*1024
            wf=weight("f_b_proj.weight",off,1024);wg=weight("g_b_proj.weight",off,1024)
            cw=torch.cat([weight(f"{q}_conv1d.weight",off,1024) for q in ("q","k","v")]).reshape(3072,4).float()
            al=weight("A_log",args.rank*8,8).flatten().float()
            dt=weight("dt_bias",off,1024).flatten().float();nw=weight("o_norm.weight")
        if args.stress:
            al.copy_(torch.tensor([-8,-2,-.5,0,.5,2,4,8],device="cuda"))
            dt.copy_(torch.linspace(-40,40,1024,device="cuda"))
        def refresh():
            X.copy_(rnd((S,3336),.4))
            if args.stress:
                X[0,:3072]=0
                X[:,3072:3080]=torch.linspace(-80,80,8,device="cuda").to(bf)
        # Non-zero recurrent states catch decay/update order errors on the first call.
        cs0=torch.zeros((602,3,3072),device="cuda",dtype=bf)
        ss0=torch.zeros((602,8,128,128),device="cuda",dtype=fp)
        cs0[cl.long()]=rnd((S,3,3072),.3)
        ss0[sl.long()]=rnd((S,8,128,128),.1,fp)
        cs1,ss1=cs0.clone(),ss0.clone()
        cv=torch.empty((S,3072),device="cuda",dtype=bf)
        fo,go,ft,gt=[torch.empty((S,1024),device="cuda",dtype=bf) for _ in range(4)]
        out,other=[torch.empty((S,1024),device="cuda",dtype=bf) for _ in range(2)]
        rawref,rawnew=[torch.empty_like(out) for _ in range(2)]
        cvnew=torch.empty_like(cv)
        rsnew=torch.empty((S,8),device="cuda",dtype=fp)
        rs=torch.empty((S,8),device="cuda",dtype=fp)
        fref=lt.gemm(X[:,3080:],wf,fo,S,1024,128,3336)
        gref=lt.gemm(X[:,3208:],wg,go,S,1024,128,3336)
        fnew=D.kernel(mod,"glm53_kda_fg",(S,8,1),128,[X,wf,wg,ft,gt,I(S)])
        def refs(fgate,ggate):
            c=D.kernel(conv,"_causal_conv1d_update_kernel",(S,12,1),128,
                       [X,cw,cs0,cl,X,cv,I(S),J(0),J(0)])
            d=D.kernel(delta,"fused_sigmoid_gating_delta_rule_update_kernel",(4,S,8),32,
                [al,fgate,dt,F(1),F(20),F(-5),cv,cv[:,1024:],cv[:,2048:],X[:,3072:],rawref,
                 ss0,sl,I(131072),cu,I(0),F(128**-.5),I(S),I(1024),I(3072),I(3072),I(3072),
                 I(3336),J(0),J(0)],64)
            n=D.kernel(norm,"layer_norm_gated_fwd_kernel",((S*8+31)//32,1,1),128,
                       [rawref,ggate,out,nw,rs,F(1e-5),I(S*8),J(0),J(0)],256)
            def run(): c();d();n()
            return run
        reference=refs(fo,go); matching=refs(ft,gt)
        def new(mode,nt,debug=False):
            suffix="_debug" if debug else ""
            return D.kernel(mod,f"glm53_kda_{mode}{nt}{suffix}",(S,8,1),nt,
                [X,wf if mode=="fused" else fo,wg if mode=="fused" else go,cw,al,dt,nw,
                 cs1,ss1,cl,sl,cu,other,I(S)]+([rawnew,cvnew,rsnew] if debug else []))
        fref();gref();fnew();torch.cuda.synchronize()
        result={"batch":S,"fg_f":stats(fo,ft),"fg_g":stats(go,gt),
                "fg_cublas":fref.info,"core":{},"fused_same_fg":{}}
        # Start each variant from IDENTICAL current oracle state, not from stale allocation.
        for mode in ("core","fused"):
            for nt in map(int,args.threads.split(",")):
                cs1.copy_(cs0);ss1.copy_(ss0)
                run=new(mode,nt,True)
                oracle=reference if mode=="core" else matching
                for step in range(args.steps):
                    refresh()
                    if mode=="core": fref();gref()
                    else: fnew()
                    oracle();run()
                    if not bits_equal(out,other) and not args.keep_going:
                        raise AssertionError(f"output bits differ at batch={S} mode={mode} threads={nt} step={step}")
                torch.cuda.synchronize()
                cut={"output":stats(out,other), "state":stats(ss0[sl.long()],ss1[sl.long()]),
                     "conv_equal":bits_equal(cs0,cs1),
                     "raw":stats(rawref,rawnew),"conv_output":stats(cv,cvnew),"rstd":stats(rs,rsnew)}
                if not all((bits_equal(ss0,ss1),cut["conv_equal"],bits_equal(out,other),bits_equal(rawref,rawnew),bits_equal(cv,cvnew),bits_equal(rs,rsnew))):
                    result["FAILED"]={"mode":mode,"threads":nt,"cut":cut}
                    print(json.dumps(result),flush=True)
                    if not args.keep_going: raise AssertionError("same-input mined parity failed")
                result["core" if mode=="core" else "fused_same_fg"][nt]=cut
        # cuBLAS fg versus fused fg: report separately; not a bitwise claim.
        cs1.copy_(cs0);ss1.copy_(ss0);full=new("fused",int(args.threads.split(",")[0]))
        for step in range(args.steps):
            refresh();fref();gref();reference();full()
        torch.cuda.synchronize()
        result["fused_vs_cublas"]={"output":stats(out,other),"state":stats(ss0[sl.long()],ss1[sl.long()])}
        def base(): fref();gref();reference()
        result["us"]={"baseline_chain":timing(base),"mined_core":timing(reference)}
        for mode in ("core","fused"):
            for nt in map(int,args.threads.split(",")): result["us"][f"{mode}{nt}"]=timing(new(mode,nt))
        xx=rnd((S,4096),.1); ww=rnd((3336,4096),.05)
        if args.layer is not None:
            ww=torch.cat([weight(f"{q}_proj.weight",args.rank*1024,1024) for q in ("q","k","v")]
                         +[weight("b_proj.weight",args.rank*8,8),weight("f_a_proj.weight"),weight("g_a_proj.weight")])
        yy,zz=[torch.empty((S,3336),device="cuda",dtype=bf) for _ in range(2)]
        qr=lt.gemm(xx,ww,yy,S,3336,4096)
        qn=D.kernel(mod,"glm53_kda_qkv_direct",(834,1,1),128,[xx,ww,zz,I(S),I(3336),I(4096)])
        qr();qn();torch.cuda.synchronize()
        result["qkv"]={"cublas":qr.info,"diff":stats(yy,zz),"baseline_us":timing(qr),"direct_us":timing(qn)}
        print(json.dumps(result),flush=True);results.append(result)
        # Lt closures retain their operands. Release them before the next bucket.
        lt.objects.clear()
        del ss0,ss1,cs0,cs1,ww,xx
    ART.mkdir(parents=True,exist_ok=True)
    (ART/args.report).write_text(json.dumps({"args":vars(args),
        "cubin_sha256":hashlib.sha256(CUBIN.read_bytes()).hexdigest(),
        "source_sha256":hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "gpu":torch.cuda.get_device_name(0),"torch_cuda":torch.version.cuda,
        "timing":"median of 5 ordinary CUDA graph replays; warm weights; us/call",
        "results":results},indent=2))


DELTA_ENTRY = "fused_sigmoid_gating_delta_rule_update_kernel"


def test_v3(args):
    """MTP verify-fusion gates (rows=N; --rows 7 is the k6 fused-path gate,
    --rows 3 the champion regression). References, all from production kernels:
    A) the unfused round chain (Lt fg_b -> spec_conv_verify -> spec_delta_verify
       -> module_227 norm -> spec_pack_kda -> spec_delta_advance -> spec_conv_advance)
    B) spec_delta_advance/spec_conv_advance snapshots with forced nacc=j+1
    C) the sequential 1-row reference of gate 1: decode module_363 + mined conv
       run row by row, state snapshotted after each row.
    The v3 verify+stash must be BITWISE equal to A/B/C in core mode; the select
    commit is a pure copy, so final states must be bitwise equal for every
    forced nacc pattern, including invalid sequences (state untouched).
    """
    check_build_v3()
    gpu_check()
    import torch
    torch.cuda.init(); torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    D, lt = Driver(torch), Lt(torch)
    v3 = D.module(V3CUBIN)
    dump = ROOT / "dumped-kernels-glm53-sglang"
    spec_conv = D.module(SPECART / "spec_kda.cubin")
    spec_dv = D.module(SPECART / "spec_delta_verify.cubin")
    spec_da = D.module(SPECART / "spec_delta_advance.cubin")
    spec_glue = D.module(SPECART / "spec_round.cubin")
    norm_m = D.module(dump / "module_227.cubin")
    conv_dec = D.module(ROOT / "kernels-glm53-handwritten/module_conv_generic.cubin")
    delta_dec = D.module(dump / "module_363.cubin")
    I, J, F32 = C.c_int, C.c_int64, C.c_float
    bf, fp = torch.bfloat16, torch.float32
    SCALE = 128**-.5
    NL = 602  # synthetic state-pool lines, mirrors test()
    def rnd(shape, scale=1, dtype=bf):
        return (torch.randn(shape, device="cuda", dtype=fp)*scale).to(dtype)
    def bits_equal(a, b):
        dtype = torch.int16 if a.element_size() == 2 else torch.int32
        return torch.equal(a.view(dtype), b.view(dtype))
    def rel_err(a, b):
        aa, bb = a.float(), b.float()
        diff = (aa-bb).abs()
        l2 = (diff.square().sum().sqrt()/(bb.square().sum().sqrt()+1e-30)).item()
        el = (diff/(bb.abs()+1e-6)).max().item()
        return {"max_abs": diff.max().item(), "l2_rel": l2, "max_elem_rel": el}
    def timing(fn, repeats):
        for _ in range(3): fn()
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(repeats): fn()
        stream.synchronize()
        vals = []
        for _ in range(5):
            s0, e0 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s0.record(); graph.replay(); e0.record(); e0.synchronize()
            vals.append(s0.elapsed_time(e0)*1000/repeats)
        return sorted(vals)[2]
    results = []
    R = args.rows
    threads = [int(x) for x in args.threads.split(",")]
    for S in map(int, args.batches.split(",")):
        if not 1 <= S <= 8: raise ValueError("round batch must be 1..8")
        T = R*S
        cl = torch.tensor([600]+list(range(2, 2*S, 2)), dtype=torch.int32, device="cuda")
        sl = torch.tensor([601]+list(range(3, 2*S+1, 2)), dtype=torch.int32, device="cuda")
        clv = [int(x) for x in cl.cpu()]; slv = [int(x) for x in sl.cpu()]
        cu_v = torch.arange(S+1, dtype=torch.int32, device="cuda")*R
        cu1 = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
        Fin = rnd((T, 3336), .4)
        wf, wg = rnd((1024, 128), .08), rnd((1024, 128), .08)
        cw = rnd((3072, 4), .4, fp); al = rnd((8,), .5, fp)
        dt = rnd((1024,), 1, fp); nw = rnd((128,), .5)
        if args.stress:
            al.copy_(torch.tensor([-8, -2, -.5, 0, .5, 2, 4, 8], device="cuda"))
            dt.copy_(torch.linspace(-40, 40, 1024, device="cuda"))
        if args.layer is not None:
            from safetensors import safe_open
            cp = Path(args.checkpoint)
            index = json.loads((cp/"model.safetensors.index.json").read_text())["weight_map"]
            prefix = f"model.language_model.layers.{args.layer}.self_attn."
            def weight(name, start=None, count=None):
                key = prefix+name
                with safe_open(cp/index[key], framework="pt", device="cpu") as sf:
                    x = sf.get_tensor(key) if start is None else sf.get_slice(key)[start:start+count]
                return x.contiguous().cuda()
            off = args.rank*1024
            wf = weight("f_b_proj.weight", off, 1024); wg = weight("g_b_proj.weight", off, 1024)
            cw = torch.cat([weight(f"{q}_conv1d.weight", off, 1024)
                            for q in ("q", "k", "v")]).reshape(3072, 4).float()
            al = weight("A_log", args.rank*8, 8).flatten().float()
            dt = weight("dt_bias", off, 1024).flatten().float(); nw = weight("o_norm.weight")
        cs_init = torch.zeros((NL, 3, 3072), device="cuda", dtype=bf)
        ss_init = torch.zeros((NL, 8, 128, 128), device="cuda", dtype=fp)
        cs_init[cl.long()] = rnd((S, 3, 3072), .3)
        ss_init[sl.long()] = rnd((S, 8, 128, 128), .1, fp)
        forget, gproj = torch.empty((T, 1024), device="cuda", dtype=bf), torch.empty((T, 1024), device="cuda", dtype=bf)
        fgem = lt.gemm(Fin[:, 3080:], wf, forget, T, 1024, 128, 3336)
        ggem = lt.gemm(Fin[:, 3208:], wg, gproj, T, 1024, 128, 3336)
        fgem(); ggem()
        valid_all = torch.ones(R*S, dtype=torch.int32, device="cuda")
        # ---- Reference A: unfused round chain (verify section is nacc-free).
        cs0, ss0 = cs_init.clone(), ss_init.clone()
        Q = torch.empty((T, 3072), device="cuda", dtype=bf)
        o_ref = torch.empty((T, 1024), device="cuda", dtype=bf)
        rstd = torch.empty((T*8,), device="cuda", dtype=fp)
        cvk = D.kernel(spec_conv, "spec_conv_verify", (S, 12, 1), 256,
                       [Fin, cw, cs0, cl, Q, valid_all, I(R)])
        dvk = D.kernel(spec_dv, DELTA_ENTRY, (4, S, 8), 32,
            [al, forget, dt, F32(1), F32(20), F32(-5), Q, Q[:, 1024:], Q[:, 2048:],
             Fin[:, 3072:], o_ref, ss0, sl, I(131072), cu_v, I(0), F32(SCALE), I(T),
             I(1024), I(3072), I(3072), I(3072), I(3336), J(0), J(0)], 64)
        nmk = D.kernel(norm_m, "layer_norm_gated_fwd_kernel", ((T*8+31)//32, 1, 1), 128,
            [o_ref, gproj, o_ref, nw, rstd, F32(1e-5), I(T*8), J(0), J(0)], 256)
        cvk(); dvk()
        raw_ref = o_ref.clone()
        nmk(); torch.cuda.synchronize()
        # ---- Reference C: sequential 1-row snapshots (gate-1 reference).
        ss_seq, cs_seq = ss_init.clone(), cs_init.clone()
        snap_ssm = torch.empty((S, R, 8, 128, 128), device="cuda", dtype=fp)
        snap_conv = torch.empty((S, R, 3, 3072), device="cuda", dtype=bf)
        o_d = torch.empty((1, 1024), device="cuda", dtype=bf)
        cv_d = torch.empty((1, 3072), device="cuda", dtype=bf)
        for s in range(S):
            l_ss = sl[s:s+1].contiguous(); l_cs = cl[s:s+1].contiguous()
            for j in range(R):
                r = R*s+j
                drow = D.kernel(delta_dec, DELTA_ENTRY, (4, 1, 8), 32,
                    [al, forget[r:r+1], dt, F32(1), F32(20), F32(-5), Q[r:r+1],
                     Q[r:r+1, 1024:], Q[r:r+1, 2048:], Fin[r:r+1, 3072:], o_d, ss_seq,
                     l_ss, I(131072), cu1, I(0), F32(SCALE), I(1), I(1024), I(3072),
                     I(3072), I(3072), I(3336), J(0), J(0)], 64)
                crow = D.kernel(conv_dec, "_causal_conv1d_update_kernel", (1, 12, 1), 128,
                    [Fin[r:r+1], cw, cs_seq, l_cs, Fin[r:r+1], cv_d, I(1), J(0), J(0)])
                drow(); crow()
                snap_ssm[s, j].copy_(ss_seq[slv[s]]); snap_conv[s, j].copy_(cs_seq[clv[s]])
        torch.cuda.synchronize()
        # ---- Reference B: forced-nacc advance snapshots (j+1 rows packed).
        packF = torch.empty((T, 3336), device="cuda", dtype=bf)
        packQ = torch.empty((T, 3072), device="cuda", dtype=bf)
        packA = torch.empty((T, 1024), device="cuda", dtype=bf)
        adv_o = torch.empty((T, 1024), device="cuda", dtype=bf)
        b_ssm = torch.empty_like(snap_ssm); b_conv = torch.empty_like(snap_conv)
        def ref_advance(nacc_t, acu_t, cs, ss):
            pk = D.kernel(spec_glue, "spec_pack_kda", (S, R, 1), 256,
                          [Fin, Q, forget, packF, packQ, packA, nacc_t, acu_t, I(R)])
            da = D.kernel(spec_da, DELTA_ENTRY, (4, S, 8), 32,
                [al, packA, dt, F32(1), F32(20), F32(-5), packQ, packQ[:, 1024:],
                 packQ[:, 2048:], packF[:, 3072:], adv_o, ss, sl, I(131072), acu_t,
                 I(0), F32(SCALE), I(T), I(1024), I(3072), I(3072), I(3072), I(3336),
                 J(0), J(0)], 64)
            ca = D.kernel(spec_conv, "spec_conv_advance", (S, 12, 1), 256,
                          [Fin, cs, cl, nacc_t, valid_all, I(R)])
            def run(): pk(); da(); ca()
            return run
        for j in range(R):
            cs_b, ss_b = cs_init.clone(), ss_init.clone()
            nj = torch.full((S,), j+1, dtype=torch.int32, device="cuda")
            acu = torch.zeros(S+1, dtype=torch.int32, device="cuda")
            acu[1:] = torch.cumsum(nj, 0)
            ref_advance(nj, acu, cs_b, ss_b)()
            for s in range(S):
                b_ssm[s, j].copy_(ss_b[slv[s]]); b_conv[s, j].copy_(cs_b[clv[s]])
        torch.cuda.synchronize()
        res = {"batch": S, "refs": {
            "advance_vs_seq_ssm_bits": bits_equal(b_ssm, snap_ssm),
            "advance_vs_seq_conv_bits": bits_equal(b_conv, snap_conv),
            "advance_vs_seq_ssm_rel": rel_err(b_ssm, snap_ssm)}}
        # ---- v3 verify + select.
        stash_ssm = torch.empty((S, R, 8, 128, 128), device="cuda", dtype=fp)
        stash_conv = torch.empty((S, R, 3, 3072), device="cuda", dtype=bf)
        o_new = torch.empty((T, 1024), device="cuda", dtype=bf)
        dbg_raw = torch.empty((T, 1024), device="cuda", dtype=bf)
        dbg_cv = torch.empty((T, 3072), device="cuda", dtype=bf)
        dbg_rs = torch.empty((S, 8*R), device="cuda", dtype=fp)
        cs1, ss1 = cs_init.clone(), ss_init.clone()
        def verify(mode, nt, debug=True):
            f_in, g_in = (forget, gproj) if mode == "core" else (wf, wg)
            suffix = "_debug" if debug else ""
            args_l = [Fin, f_in, g_in, cw, al, dt, nw, cs1, ss1, cl, sl, cu_v,
                      o_new, stash_ssm, stash_conv, I(S), I(R)]
            if debug: args_l += [dbg_raw, dbg_cv, dbg_rs]
            return D.kernel(v3, f"glm53_kda_verify_{mode}{nt}{suffix}", (S, 8, 1), nt, args_l)
        def select(nacc_t):
            return D.kernel(v3, "glm53_kda_select", (S, 8, 1), 256,
                            [stash_ssm, stash_conv, ss1, cs1, sl, cl, nacc_t, valid_all, I(R)])
        if R == 3:
            pats = {"all1": [1]*S, "all2": [2]*S, "all3": [3]*S,
                    "mixed": ([1, 2, 3, 1, 3, 2, 1, 2]*2)[:S]}
        else:
            # k6 coverage: extremes + the {1,7,3,5} forced-nacc set first.
            pats = {"all1": [1]*S, f"all{R}": [R]*S,
                    "mixed": ([1, R, 3, 5, 2, 6, 4, 7]*2)[:S]}
        if S > 1:
            inv = list(pats["mixed"])
            for s in ([1, 3] if S > 3 else [1]): inv[s] = 0
            pats["invalid"] = inv
        for mode in ("core", "fused"):
            for nt in (threads if mode == "core" else threads[:1]):
                bitwise = mode == "core"
                cs1.copy_(cs_init); ss1.copy_(ss_init)
                verify(mode, nt)()
                torch.cuda.synchronize()
                cut = {"o": rel_err(o_new, o_ref), "o_bits": bits_equal(o_new, o_ref),
                       "raw_bits": bits_equal(dbg_raw, raw_ref),
                       "cv_bits": bits_equal(dbg_cv, Q),
                       "stash_ssm_bits": bits_equal(stash_ssm, snap_ssm),
                       "stash_conv_bits": bits_equal(stash_conv, snap_conv),
                       "stash_ssm_rel": rel_err(stash_ssm, snap_ssm)}
                if bitwise and not all([cut["o_bits"], cut["raw_bits"], cut["cv_bits"],
                                        cut["stash_ssm_bits"], cut["stash_conv_bits"]]):
                    res.setdefault("FAILED", {})[f"verify_{mode}{nt}"] = cut
                    print(json.dumps(res), flush=True)
                    if not args.keep_going: raise AssertionError("v3 verify parity failed")
                # Gate 1: per-row stash vs sequential 1-row reference.
                gate1 = rel_err(stash_ssm, snap_ssm)
                cut["gate1_rel_le_1e-3"] = gate1["max_elem_rel"] <= 1e-3 or cut["stash_ssm_bits"]
                # Forced-nacc commit checks on the SAME stash (select is copy-only).
                for name, pl in pats.items():
                    nacc_t = torch.tensor(pl, dtype=torch.int32, device="cuda")
                    if name == "invalid":
                        valid_t = valid_all.clone()
                        for s in range(S):
                            if pl[s] == 0: valid_t[R*s:R*s+R] = 0
                    else:
                        valid_t = valid_all
                    acu = torch.zeros(S+1, dtype=torch.int32, device="cuda")
                    acu[1:] = torch.cumsum(nacc_t, 0)
                    cs0.copy_(cs_init); ss0.copy_(ss_init)
                    pk = D.kernel(spec_glue, "spec_pack_kda", (S, R, 1), 256,
                                  [Fin, Q, forget, packF, packQ, packA, nacc_t, acu, I(R)])
                    da = D.kernel(spec_da, DELTA_ENTRY, (4, S, 8), 32,
                        [al, packA, dt, F32(1), F32(20), F32(-5), packQ, packQ[:, 1024:],
                         packQ[:, 2048:], packF[:, 3072:], adv_o, ss0, sl, I(131072),
                         acu, I(0), F32(SCALE), I(T), I(1024), I(3072), I(3072),
                         I(3072), I(3336), J(0), J(0)], 64)
                    ca = D.kernel(spec_conv, "spec_conv_advance", (S, 12, 1), 256,
                                  [Fin, cs0, cl, nacc_t, valid_t, I(R)])
                    cs1.copy_(cs_init); ss1.copy_(ss_init)
                    sel = D.kernel(v3, "glm53_kda_select", (S, 8, 1), 256,
                                   [stash_ssm, stash_conv, ss1, cs1, sl, cl, nacc_t, valid_t, I(R)])
                    pk(); da(); ca(); sel(); torch.cuda.synchronize()
                    cut[f"commit_{name}_ssm_bits"] = bits_equal(ss0, ss1)
                    cut[f"commit_{name}_conv_bits"] = bits_equal(cs0, cs1)
                    cut[f"commit_{name}_ssm_rel"] = rel_err(ss1[sl.long()], ss0[sl.long()])
                    if not cut[f"commit_{name}_ssm_bits"] or not cut[f"commit_{name}_conv_bits"]:
                        res.setdefault("FAILED", {})[f"commit_{mode}{nt}_{name}"] = cut
                        print(json.dumps(res), flush=True)
                        if not args.keep_going: raise AssertionError(f"select commit failed: {name}")
                res[f"{mode}{nt}"] = cut
        # ---- Drift loop: repeated fused rounds vs unfused chain (gate-4 proxy).
        if args.drift_rounds:
            nt = threads[0]
            cs0.copy_(cs_init); ss0.copy_(ss_init)
            cs1.copy_(cs_init); ss1.copy_(ss_init)
            drift = {"rounds": args.drift_rounds, "o_bits_fail": 0, "checks": []}
            g = torch.Generator(device="cuda").manual_seed(args.seed+1)
            for rnd_i in range(args.drift_rounds):
                Fin.copy_(rnd((T, 3336), .4))
                fgem(); ggem(); cvk(); dvk(); nmk()
                nl = torch.randint(1, R+1, (S,), generator=g, device="cuda", dtype=torch.int32)
                if S > 1 and rnd_i % 17 == 3: nl[1] = 0
                valid_t = valid_all.clone()
                for s in range(S):
                    if int(nl[s]) == 0: valid_t[R*s:R*s+R] = 0
                acu = torch.zeros(S+1, dtype=torch.int32, device="cuda")
                acu[1:] = torch.cumsum(nl, 0)
                pk = D.kernel(spec_glue, "spec_pack_kda", (S, R, 1), 256,
                              [Fin, Q, forget, packF, packQ, packA, nl, acu, I(R)])
                da = D.kernel(spec_da, DELTA_ENTRY, (4, S, 8), 32,
                    [al, packA, dt, F32(1), F32(20), F32(-5), packQ, packQ[:, 1024:],
                     packQ[:, 2048:], packF[:, 3072:], adv_o, ss0, sl, I(131072), acu,
                     I(0), F32(SCALE), I(T), I(1024), I(3072), I(3072), I(3072),
                     I(3336), J(0), J(0)], 64)
                ca = D.kernel(spec_conv, "spec_conv_advance", (S, 12, 1), 256,
                              [Fin, cs0, cl, nl, valid_t, I(R)])
                verify("core", nt, False)()
                sel = D.kernel(v3, "glm53_kda_select", (S, 8, 1), 256,
                               [stash_ssm, stash_conv, ss1, cs1, sl, cl, nl, valid_t, I(R)])
                pk(); da(); ca(); sel(); torch.cuda.synchronize()
                if not bits_equal(o_new, o_ref): drift["o_bits_fail"] += 1
                if (rnd_i+1) % 25 == 0 or rnd_i == args.drift_rounds-1:
                    ok = bits_equal(ss0, ss1) and bits_equal(cs0, cs1)
                    drift["checks"].append({"round": rnd_i+1, "state_bits": ok,
                        "ssm_rel": rel_err(ss1[sl.long()], ss0[sl.long()])})
                    if not ok and not args.keep_going:
                        res["FAILED"] = {"drift": drift}
                        print(json.dumps(res), flush=True)
                        raise AssertionError(f"drift divergence at round {rnd_i+1}")
            drift["final_state_bits"] = bits_equal(ss0, ss1) and bits_equal(cs0, cs1)
            res["drift"] = drift
        # ---- Launch-bound timing: one KDA layer chain, us; x34 for the round.
        nacc3 = torch.full((S,), R, dtype=torch.int32, device="cuda")
        acu3 = torch.zeros(S+1, dtype=torch.int32, device="cuda")
        acu3[1:] = torch.cumsum(nacc3, 0)
        pkk = D.kernel(spec_glue, "spec_pack_kda", (S, R, 1), 256,
                       [Fin, Q, forget, packF, packQ, packA, nacc3, acu3, I(R)])
        dak = D.kernel(spec_da, DELTA_ENTRY, (4, S, 8), 32,
            [al, packA, dt, F32(1), F32(20), F32(-5), packQ, packQ[:, 1024:],
             packQ[:, 2048:], packF[:, 3072:], adv_o, ss0, sl, I(131072), acu3,
             I(0), F32(SCALE), I(T), I(1024), I(3072), I(3072), I(3072), I(3336),
             J(0), J(0)], 64)
        cak = D.kernel(spec_conv, "spec_conv_advance", (S, 12, 1), 256,
                       [Fin, cs0, cl, nacc3, valid_all, I(R)])
        vf_c = verify("core", threads[0], False)
        vf_f = verify("fused", threads[0], False)
        selk = select(nacc3)
        def unfused(): fgem(); ggem(); cvk(); dvk(); nmk(); pkk(); dak(); cak()
        def core_chain(): fgem(); ggem(); vf_c(); selk()
        def fused_chain(): vf_f(); selk()
        res["us_layer"] = {"unfused": timing(unfused, args.repeats),
                           f"core{threads[0]}": timing(core_chain, args.repeats),
                           f"fused{threads[0]}": timing(fused_chain, args.repeats)}
        res["us_round34"] = {k: v*34 for k, v in res["us_layer"].items()}
        res["saved_us_round34"] = {k: res["us_round34"]["unfused"]-v
                                   for k, v in res["us_round34"].items() if k != "unfused"}
        print(json.dumps(res), flush=True)
        results.append(res)
        lt.objects.clear()
        del ss0, ss1, cs0, cs1, ss_seq, cs_seq, ss_init, cs_init
    V3ART.mkdir(parents=True, exist_ok=True)
    (V3ART/args.report).write_text(json.dumps({"args": vars(args),
        "cubin_sha256": hashlib.sha256(V3CUBIN.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(V3SOURCE.read_bytes()).hexdigest(),
        "gpu": torch.cuda.get_device_name(0), "torch_cuda": torch.version.cuda,
        "gate1": "stash_ssm vs sequential 1-row module_363 snapshots; bitwise expected, rel<=1e-3 required",
        "timing": "median of 5 CUDA graph replays, warm; us per KDA layer; us_round34 = x34",
        "results": results}, indent=2))


def fuse_round_cmd(args):
    m = json.loads(Path(args.manifest).read_text())
    m = fuse_round_manifest(m, args.mode, args.threads, program=args.program)
    lower_v3(m)
    Path(args.out).write_text(json.dumps(m))
    n = len(m["programs"][args.program]["calls"])
    print(f"{args.program} calls: {n} (mode={args.mode} threads={args.threads})")


def lower_v3(m):
    """Standalone path only: gen.py's lower_wire does this for the integrated
    hook. Register the v3 cubin as a module and rewrite the two new ops'
    launches to the lowered {module, entry, params, args, block, grid, pdl}
    shape the runtime's Launch enum accepts."""
    stamp = check_build_v3()
    m.setdefault("modules", {})["glm53_kda_v3"] = {
        "source": str(V3CUBIN), "sha256": stamp["cubin_sha256"]}
    for name in ("spec_kda_fused_v3", "spec_kda_select"):
        for l in m["ops"][name]["impl"]["launches"]:
            for k in ("cubin", "sha256", "label"):
                l.pop(k, None)
            l["module"] = "glm53_kda_v3"


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="cmd",required=True)
    sub.add_parser("build")
    sub.add_parser("build-v3")
    t3=sub.add_parser("test-v3")
    t3.add_argument("--rows",type=int,default=3)
    t3.add_argument("--batches",default="1,2,4,8")
    t3.add_argument("--threads",default="512")
    t3.add_argument("--drift-rounds",type=int,default=200)
    t3.add_argument("--repeats",type=int,default=34)
    t3.add_argument("--layer",type=int)
    t3.add_argument("--rank",type=int,default=0)
    t3.add_argument("--checkpoint",
        default=os.environ.get("GLM53_CHECKPOINT","weights/GLM-5.3-Flash"))
    t3.add_argument("--stress",action="store_true")
    t3.add_argument("--seed",type=int,default=5300)
    t3.add_argument("--report",default="ab_v3.json")
    t3.add_argument("--keep-going",action="store_true")
    fr=sub.add_parser("fuse-round")
    fr.add_argument("manifest")
    fr.add_argument("out")
    fr.add_argument("--mode",default="core",choices=["core","fused"])
    fr.add_argument("--threads",type=int,default=512,choices=[256,512,1024])
    fr.add_argument("--program",default="round_k2")
    t=sub.add_parser("test")
    t.add_argument("--batches",default="1,2,4,8,16")
    t.add_argument("--threads",default="256,512,1024")
    t.add_argument("--steps",type=int,default=8)
    t.add_argument("--repeats",type=int,default=40)
    t.add_argument("--layer",type=int)
    t.add_argument("--rank",type=int,default=0)
    t.add_argument("--checkpoint",default=os.environ.get("GLM53_CHECKPOINT","weights/GLM-5.3-Flash"))
    t.add_argument("--stress",action="store_true")
    t.add_argument("--seed",type=int,default=5300)
    t.add_argument("--report",default="ab.json")
    t.add_argument("--keep-going",action="store_true")
    a=p.parse_args()
    if a.cmd=="build":build()
    elif a.cmd=="build-v3":build_v3()
    elif a.cmd=="fuse-round":fuse_round_cmd(a)
    elif a.cmd=="test-v3":
        if not 1 <= a.rows <= 8: p.error("rows must be 1..8")
        if not 0 <= a.rank < 8: p.error("rank must be 0..7")
        if a.layer is not None and (not 0 <= a.layer < 45 or a.layer % 4 == 3):
            p.error("layer must be a KDA layer in 0..44")
        if any(int(n) not in (256,512,1024) for n in a.threads.split(",")):
            p.error("v3 threads must be selected from 256,512,1024")
        if Path(a.report).name != a.report or not a.report.endswith(".json") or a.report == "build.json":
            p.error("report must be a JSON filename other than build.json")
        if a.drift_rounds < 0 or a.repeats < 1: p.error("drift-rounds>=0, repeats>=1")
        test_v3(a)
    else:
        if not 0 <= a.rank < 8: p.error("rank must be 0..7")
        if a.layer is not None and (not 0 <= a.layer < 45 or a.layer % 4 == 3):
            p.error("layer must be a KDA layer in 0..44")
        if a.steps < 1 or a.repeats < 1: p.error("steps/repeats must be positive")
        if any(int(n) not in (128,256,512,1024) for n in a.threads.split(",")):
            p.error("threads must be selected from 128,256,512,1024")
        if Path(a.report).name != a.report or not a.report.endswith(".json") or a.report == "build.json":
            p.error("report must be a JSON filename other than build.json")
        test(a)

if __name__=="__main__":main()
