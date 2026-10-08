#!/usr/bin/env python3
"""Single-GPU cubin A/B for the MTP-verify MoE (M<=32, rows=tokens).

Oracle = the unfused mtp32 MoE: ops_moe.ops() plus the ops_spec._base_ops(32)
transformation (second M16 router launch on rows 16..31). Fused = the same
three v2 cubins as decode, rows<=32 align tiles. No serving; no checkpoint
or shared-file writes. Run with any CUDA torch; no triton needed.
"""
import argparse
import ctypes as C
import json
import os
import pathlib
import re
import struct
import subprocess
import sys
from copy import deepcopy

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "tools"))

def guard():
    if not os.environ.get("MOE_VERIFY_SKIP_GUARD"):
        ps = subprocess.check_output(["ps", "-eo", "comm,args"], text=True)
        for line in ps.splitlines():
            name = line.split()[0] if line.split() else ""
            if name in ("kern", "kern-serve", "kserve", "kbench") or (name.startswith(("python", "sglang", "EngineCore")) and re.search(r"(integration|bench_decode|serve_sglang|sglang)", line)):
                raise SystemExit("GPU test deferred: active integration/serving/bench: " + line)
    used = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
    gpu = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    if int(used.splitlines()[gpu]) > 1024:
        raise SystemExit("GPU test deferred: selected GPU is occupied")
    util = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    if any(int(x) > 3 for x in util.split()):
        raise SystemExit("GPU test deferred: active GPU work")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="1,2,3,6,7,12,16,24,32")
    ap.add_argument("--seed", type=int, default=124)
    ap.add_argument("--report", type=pathlib.Path, default=HERE / "moe_verify_v2_test.json")
    ap.add_argument("--replays", type=int, default=20)
    ap.add_argument("--timing-repeats", type=int, default=30)
    ap.add_argument("--zero-input", action="store_true", help="Tie/clamp/zero-amax test")
    args = ap.parse_args()
    guard()
    import torch
    from glm53 import ops_moe, ops_moe_v2, ops_common
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    cuda = C.CDLL("libcuda.so.1")
    def check(err):
        if err:
            msg = C.c_char_p(); cuda.cuGetErrorString(err, C.byref(msg))
            raise RuntimeError((err, msg.value))
    cache = {}
    def function(launch):
        path = pathlib.Path(launch["cubin"])
        if not path.exists(): path = ops_common.HAND_DIR / path.name
        if not path.exists(): path = ops_common.DUMP_DIR / path.name
        key = (str(path), launch["entry"])
        if key not in cache:
            mod = C.c_void_p(); check(cuda.cuModuleLoad(C.byref(mod), str(path).encode()))
            fn = C.c_void_p(); check(cuda.cuModuleGetFunction(C.byref(fn), mod, launch["entry"].encode()))
            smem = launch.get("shared_mem", 0)
            if smem > 48000: check(cuda.cuFuncSetAttribute(fn, 8, smem))
            cache[key] = (mod, fn)
        return cache[key][1]
    def ev(e, b):
        if isinstance(e, int): return e
        if isinstance(e, str): return b
        if "mul" in e: return ev(e["mul"][0], b) * ev(e["mul"][1], b)
        if "ceil_div" in e: return (ev(e["ceil_div"][0], b) + e["ceil_div"][1] - 1) // e["ceil_div"][1]
        raise ValueError(e)
    def run(launch, params, b, scratch=None):
        def argval(a):
            if "param" in a: return params[a["param"]].data_ptr() + a.get("offset", 0)
            if "scratch" in a: return scratch[a["scratch"]].data_ptr()
            if "i32" in a: return a["i32"]
            if "i64" in a: return a["i64"]
            if "f32" in a: return a["f32"]
            if "var" in a: return b
            if "expr" in a: return ev(a["expr"], b)
            raise ValueError(a)
        vals = []
        for ty, a in zip(launch["params"], launch["args"]):
            if "pack" in a:
                data = bytearray(a["pack"]["size"])
                for f in a["pack"]["fields"]:
                    kind = "f" if "f32" in f else ("i" if "i32" in f or f.get("width") == 4 else "q")
                    struct.pack_into("<" + kind, data, f["at"], argval(f))
                vals.append(C.create_string_buffer(bytes(data)))
            elif ty == "i32": vals.append(C.c_int32(argval(a)))
            elif ty == "f32": vals.append(C.c_float(argval(a)))
            else: vals.append(C.c_uint64(argval(a)))
        argv = (C.c_void_p * len(vals))(*[C.addressof(v) for v in vals])
        grid = [ev(v, b) for v in launch["grid"]]
        check(cuda.cuLaunchKernel(function(launch), *grid, *launch["block"], launch.get("shared_mem", 0),
                                  C.c_void_p(torch.cuda.current_stream().cuda_stream), argv, None))
    dt = {"f32": torch.float32, "i32": torch.int32, "bf16": torch.bfloat16, "fp8e4m3": torch.float8_e4m3fn}
    # Shorthand cubin names; the harness resolves HAND_DIR/DUMP_DIR by basename.
    newop = ops_moe_v2._op(32, "tokens", resolve=False)
    sc = {n: torch.zeros(d["shape"], dtype=dt[d["dtype"]], device="cuda") for n, d in newop["impl"]["scratch"].items()}
    # Oracle: ops_spec._base_ops(32) MoE ops, without importing gen.
    oldops = {n: o for n, o in ops_moe.ops().items() if n.startswith("moe_")}
    for name in ("moe_w13", "moe_w2"):
        oldops[name]["impl"]["launches"][0]["args"][12] = {"expr": {"mul": ["tokens", 9]}}
    router = oldops["moe_router"]
    second = deepcopy(router["impl"]["launches"][0])
    router["params"] += ["in buffer<bf16>", "out buffer<f32>"]
    for field in second["args"][0]["pack"]["fields"]:
        if field.get("param") == 0: field["param"] = 3
        elif field.get("param") == 2: field["param"] = 4
    router["impl"]["launches"].append(second)
    def zeros(shape, dtype): return torch.zeros(shape, dtype=dtype, device="cuda")
    x = torch.randn((32, 4096), device="cuda", dtype=torch.bfloat16)
    rw = torch.randn((288, 4096), device="cuda", dtype=torch.bfloat16) * .03
    bias = torch.randn((288,), device="cuda") * .1
    w13 = torch.randn((289, 512, 4096), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    s13 = torch.rand((289, 4, 32), device="cuda") * .03 + .01
    w2 = torch.randn((289, 4096, 256), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    s2 = torch.rand((289, 32, 2), device="cuda") * .06 + .01
    valid = torch.ones((32,), device="cuda", dtype=torch.int32)
    out = zeros((32, 4096), torch.bfloat16)
    if args.zero_input:
        x.zero_(); bias.zero_()
    ref = {n: zeros(shape, dtype) for n, shape, dtype in [
        ("scores", (32, 288), torch.float32), ("weights", (32, 9), torch.float32), ("ids", (32, 9), torch.int32),
        ("packed", (32, 9), torch.int32), ("sorted", (288 * 64,), torch.int32), ("experts", (288,), torch.int32), ("npost", (1,), torch.int32),
        ("a8", (32, 4096), torch.float8_e4m3fn), ("as", (32, 32), torch.float32),
        ("c1", (288, 512), torch.bfloat16), ("h", (288, 256), torch.bfloat16), ("h8", (288, 256), torch.float8_e4m3fn),
        ("hs", (288, 2), torch.float32), ("c2", (288, 4096), torch.bfloat16), ("out", (32, 4096), torch.bfloat16)]}
    stages = [("moe_router", [x, rw, ref["scores"], x[16:], ref["scores"][16:]]),
              ("moe_topk", [ref["scores"], bias, ref["weights"], ref["ids"], ref["packed"]]),
              ("moe_align", [ref["ids"], ref["sorted"], ref["experts"], ref["npost"]]),
              ("moe_quant_a", [x, ref["a8"], ref["as"]]),
              ("moe_w13", [ref["a8"], w13, ref["c1"], ref["as"], s13, ref["weights"], ref["sorted"], ref["experts"], ref["npost"]]),
              ("moe_silu", [ref["c1"], ref["h"]]),
              ("moe_quant_b", [ref["h"], ref["h8"], ref["hs"]]),
              ("moe_w2", [ref["h8"], w2, ref["c2"], ref["hs"], s2, ref["weights"], ref["sorted"], ref["experts"], ref["npost"]]),
              ("moe_sum_reduce", [ref["c2"], ref["out"]])]
    params = [x, rw, bias, w13, s13, w2, s2, valid, out]
    def baseline(b):
        for name, ps in stages:
            for launch in oldops[name]["impl"]["launches"]:
                run(launch, ps, b)
    def fused(b):
        for launch in newop["impl"]["launches"]:
            run(launch, params, b, sc)
    for name, _ in stages:
        for launch in oldops[name]["impl"]["launches"]: function(launch)
    for launch in newop["impl"]["launches"]: function(launch)
    def metric(a, b):
        af = a.float(); bf = b.float(); dif = (af - bf).abs()
        return {"neq": int((af != bf).sum()), "numel": a.numel(), "max_abs": float(dif.max()),
                "rms_rel": float(dif.square().mean().sqrt() / (bf.square().mean().sqrt() + 1e-20)),
                "finite": bool(torch.isfinite(af).all())}
    result = {"seed": args.seed, "build": json.loads((HERE / "moe_v2_build.json").read_text()), "cases": []}
    for b in map(int, args.batches.split(",")):
        for live in sorted(set([b, max(1, b - 1), 0] + ([17, 23] if b == 32 else []) + ([5] if b == 7 else []))):
            valid.zero_(); valid[:live] = 1
            baseline(b); fused(b); torch.cuda.synchronize()
            ids = sc["ids"][:live]; rid = ref["ids"][:live]
            case = {"bucket": b, "live": live, "ids_equal": bool(torch.equal(ids, rid)),
                    "scores_equal": bool(torch.equal(sc["scores"][:b], ref["scores"][:b])),
                    "counter_zero": int(sc["route_counter"][0]) == 0,
                    "n_tiles": int(sc["npost"][0]) // 32,
                    "pad_zero": bool((out[live:b] == 0).all()),
                    "pad_ids_minus_one": bool((sc["ids"][live:b] == -1).all())}
            if live:
                for name in ["weights", "a8", "as", "c1", "h8", "hs"]:
                    nr = live * 9 if name in ("c1", "h8", "hs") else live
                    # v3 keeps as ([32 kb][32 rows]) and hs ([2][288]) transposed.
                    lhs = sc[name].T[:nr] if name in ("as", "hs") else sc[name][:nr]
                    case[name] = metric(lhs, ref[name][:nr])
                case["out"] = metric(out[:live], ref["out"][:live])
                isolated = dict(sc)
                for name in ("h8", "weights"): isolated[name] = ref[name]
                isolated["hs"] = ref["hs"].T.contiguous()
                sc["w2_counts"].zero_()
                run(newop["impl"]["launches"][2], params, b, isolated)
                torch.cuda.synchronize()
                case["w2_isolated"] = metric(out[:live], ref["out"][:live])
                fused(b)
            nn = int(sc["npost"][0]) // 32
            aligned = sc["sorted"].reshape(-1, 32)[:nn].cpu()
            experts = sc["experts"][:nn].cpu()
            pairs = aligned[aligned < b * 9]
            case["align_bijection"] = sorted(pairs.tolist()) == list(range(live * 9))
            flatids = sc["ids"].flatten().cpu()
            case["align_experts"] = all(int(flatids[p]) == int(experts[e]) for e in range(nn) for p in aligned[e] if p < b * 9)
            expected = out.clone()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g): fused(b)
            for _ in range(args.replays): g.replay()
            torch.cuda.synchronize()
            case["replay_exact"] = bool(torch.equal(out[:b], expected[:b]))
            gb = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gb): baseline(b)
            def timing(graph):
                start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.timing_repeats): graph.replay()
                end.record(); end.synchronize()
                return start.elapsed_time(end) * 1000 / args.timing_repeats
            case["baseline_warm_us"] = timing(gb)
            case["v2_warm_us"] = timing(g)
            print(json.dumps(case), flush=True)
            result["cases"].append(case)
    # Adversarial routing skew: every live row picks expert 7 -> two 16-pair tiles.
    b = 32
    bias[7] += 100.
    valid.fill_(1)
    baseline(b); fused(b); torch.cuda.synchronize()
    nn = int(sc["npost"][0]) // 32
    ex = sc["experts"][:nn].cpu().tolist()
    case = {"skew_bias_expert": 7, "live": b, "ids_equal": bool(torch.equal(sc["ids"], ref["ids"])),
            "expert7_tiles": ex.count(7), "shared_tiles": ex.count(288),
            "out": metric(out, ref["out"]),
            "align_bijection": sorted(sc["sorted"].reshape(-1, 32)[:nn].flatten()[sc["sorted"].reshape(-1, 32)[:nn].flatten() < b * 9].cpu().tolist()) == list(range(b * 9))}
    print(json.dumps(case), flush=True)
    result["skew"] = case
    bias[7] -= 100.
    # Row invariance: a row inside b=24 must match its solo b=1 run bit-exactly.
    x24 = torch.randn((24, 4096), device="cuda", dtype=torch.bfloat16)
    saved = x[:24].clone(); x[:24] = x24
    valid.fill_(1)
    fused(24); torch.cuda.synchronize()
    out24 = out[:24].clone(); ids24 = sc["ids"][:24].clone()
    rows = {}
    for r in (0, 7, 23):
        x.zero_(); x[0] = x24[r]; valid.zero_(); valid[0] = 1
        fused(1); torch.cuda.synchronize()
        rows[r] = {"out_equal": bool(torch.equal(out[0], out24[r])),
                   "ids_equal": bool(torch.equal(sc["ids"][0], ids24[r]))}
    print(json.dumps({"row_invariance": rows}), flush=True)
    result["row_invariance"] = rows
    x[:24] = saved; valid.fill_(1)
    # Dynamic valid in the same capture at b=32.
    b = 32
    fused(b); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fused(b)
    valid[:b] = torch.arange(b, device="cuda") % 2
    g.replay(); torch.cuda.synchronize()
    selected = valid[:b].bool()
    baseline(b); torch.cuda.synchronize()
    result["dynamic_valid"] = {"hole_pad_zero": bool((out[:b][~selected] == 0).all()),
                               "hole_ids_minus_one": bool((sc["ids"][:b][~selected] == -1).all())}
    if selected.any(): result["dynamic_valid"]["out"] = metric(out[:b][selected], ref["out"][:b][selected])
    valid.fill_(1); g.replay(); torch.cuda.synchronize()
    result["dynamic_valid"]["restore_out"] = metric(out[:b], ref["out"][:b])
    # Kernel-launch counts per block, from a CUDA-profiled graph replay.
    def launch_count(fn, b):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): fn(b)
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            graph.replay(); torch.cuda.synchronize()
        return sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA and "memset" not in e.name.lower() and "memcpy" not in e.name.lower())
    result["launches_per_block"] = {"baseline": launch_count(baseline, 24), "fused": launch_count(fused, 24)}
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    keys = ("weights", "a8", "as", "c1", "h8", "hs", "out", "w2_isolated")
    ok = all(c["ids_equal"] and c["scores_equal"] and c["counter_zero"] and c["pad_zero"] and c["pad_ids_minus_one"]
             and c["align_bijection"] and c["align_experts"] and c["replay_exact"]
             and (not c["live"] or (c["out"]["finite"] and all(c[k]["neq"] == 0 for k in keys))) for c in result["cases"])
    ok = ok and result["skew"]["ids_equal"] and result["skew"]["expert7_tiles"] == 1 and result["skew"]["shared_tiles"] == 1 and result["skew"]["out"]["neq"] == 0 and result["skew"]["align_bijection"]
    ok = ok and all(r["out_equal"] and r["ids_equal"] for r in result["row_invariance"].values())
    ok = ok and result["dynamic_valid"]["hole_pad_zero"] and result["dynamic_valid"]["hole_ids_minus_one"] and result["dynamic_valid"]["restore_out"]["neq"] == 0
    if "out" in result["dynamic_valid"]: ok = ok and result["dynamic_valid"]["out"]["neq"] == 0
    if not ok: raise SystemExit("A/B gate failed; see report")
    print("ALL GATES PASSED", json.dumps(result["launches_per_block"]))
if __name__ == "__main__":
    main()
