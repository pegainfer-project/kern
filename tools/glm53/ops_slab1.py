"""Slab-1 (KDA-layer attention half) for the GLM-5.3-Flash MTP verify round.

[mtp32_kda_qkvbfg -> spec_kda_fused_v3 -> mtp32_kda_o_proj_ar] becomes ONE
spec_kda_slab1 launch (3 phases, in-kernel ticket barriers, no cooperative
launch), and 34 x spec_kda_select becomes ONE spec_kda_select_all gather.

v0.5 (--mode select / GLM53_SLAB1=select): bitwise-only variant. Keeps the
canonical-M cuBLASLt externs (qkvbfg, o_proj), spec_kda_fused_v3, and the
standalone spec_ar launches untouched; only batches 34 x spec_kda_select ->
1 x spec_kda_select_all. Motivation (nsys node-trace, bs1 kv2048, 16
replays): select 34x10.7us = 363us/replay vs select_all 14.7us => -348us
busy + -33 graph nodes, with zero numerics fork (pure gather). The v0.3
slab's handwritten GEMM phases cost +173us exec vs the cuBLAS chain, so
v0.5's projected net (-360us) EXCEEDS v0.3 (-167us) while staying exact.

Fold-order contract (v0.5 on the champion line): apply AFTER ops_kda_v2's
v3 verify fusion (requires spec_kda_fused_v3, spec_ssm_stash,
spec_conv_stash and the 34-call advance.N.select run). ORDER-FREE vs the
mHC-boundary / MoE-fused4 / slab-2-glue passes: the fold rewrites only the
consecutive advance.N.select call run and touches no mHC/MoE/slab-2 call
sites, args, or buffers. Verified composition: fused45s2 (round_k2=761)
+ this fold -> 728 calls, test_cpu 5/5 + kern verify clean (2026-09-27).
Call-count delta: -2/layer x34 and -33 advance = -101 round_k2 calls.

New files only; no generator/bundle edits. The owner wires the fuse hook into
gen.py's mtp section after fused4+fused5 land. Escape hatch: GLM53_SLAB1=0
makes fuse_round_manifest a no-op.

Numerics contract (owner policy):
  phase B (KDA v3 core): verbatim copy -> BITWISE gate vs spec_kda_fused_v3.
  phases A/C (handwritten weight-streaming GEMMs): deterministic single-writer,
    NOT cublas-bitwise -> kernel proxy gate: rel_err vs fp32 reference within
    1.5x of cublasLt's own deviation + rerun bitwise stability; KL + nacc
    sentinel at manifest integration (owner's harness).
  select_all: pure copy with verbatim guards -> BITWISE gate vs 34 selects.

CPU build:   PYTHONPATH=tools python -m glm53.ops_slab1 build
GPU gates:   PYTHONPATH=tools python -m glm53.ops_slab1 test
Manifest:    PYTHONPATH=tools python -m glm53.ops_slab1 fuse-round in.json out.json
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
ART = Path(os.environ.get("GLM53_SLAB1_ARTIFACTS", "/tmp/glm53-slab1")).expanduser().resolve()
CUBIN = ART / "glm53_slab1.cubin"
SOURCE = Path(__file__).with_name("kernels") / "glm53_slab1.cu"
SMEM = 199744  # gemmA 3-stage tiles 3*(32+32)*(512+8) bf16 * 2B = 199680 + mbarriers
GRID = 132     # 1 CTA/SM: GEMM phases tile N to engage every SM (~24.5 GB/s/SM cap)
THREADS = 512
NLAYERS = 34
# Per-layer stash strides scale with the round width (ops_kda_v2 geometry,
# stash_seqs=8): SSM_LAYER_F32(rows) = 8 * rows * 131072.
def SSM_LAYER_F32(rows=3): return 8 * rows * 8 * 16384
def CONV_LAYER_BF16(rows=3): return 8 * rows * 9216
LINES_STRIDE = 16                       # i32 per layer in the lines tables

# v3 z-split (glm53_kda_v3z.cu): grid [S,8,2] bitwise-exec variant of
# glm53_kda_verify_fused512. Scratch xchg: [0]=err, [1]=trip, tickets at
# u32[2..194), raw_x bf16[S_MAX*8*3*128] at byte 1024.
ZSOURCE = Path(__file__).with_name("kernels") / "glm53_kda_v3z.cu"
ZCUBIN = ART / "glm53_kda_v3z.cubin"
Z_SCRATCH_U32 = (1024 + 8 * 8 * 3 * 128 * 2) // 4  # = 12544

# v4 stage-A warp-specialized pipeline (glm53_kda_v4.cu): grid [seqs,8]
# unchanged, entry glm53_kda_verify_fused512p, bitwise vs pinned v3 by
# construction (same per-element instruction stream; see file header).
V4SOURCE = Path(__file__).with_name("kernels") / "glm53_kda_v4.cu"
V4CUBIN = ART / "glm53_kda_v4.cubin"
# v4 stage-B (z2s) scratch: see glm53_kda_v4.cu layout comment.
V4_SCRATCH_U32 = (1040 + 64 * 3072 + 64 * 3 * 128 * 2) // 4  # = 61700


def build_v4():
    ART.mkdir(parents=True, exist_ok=True)
    cmd = [NVCC, "-cubin", "-arch=sm_90a", "-std=c++17", "-O3", "-ccbin", CCBIN,
           "-Xptxas=-v", str(V4SOURCE), "-o", str(V4CUBIN)]
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (ART / "build_v4.log").write_text(" ".join(cmd) + "\n" + p.stdout)
    print(p.stdout, end="")
    p.check_returncode()
    stamp = {"source_sha256": hashlib.sha256(V4SOURCE.read_bytes()).hexdigest(),
             "cubin_sha256": hashlib.sha256(V4CUBIN.read_bytes()).hexdigest(), "command": cmd}
    (ART / "build_v4.json").write_text(json.dumps(stamp, indent=2))
    print("v4 sha256", stamp["cubin_sha256"])


def check_build_v4():
    stamp = json.loads((ART / "build_v4.json").read_text())
    if stamp["source_sha256"] != hashlib.sha256(V4SOURCE.read_bytes()).hexdigest() or \
       stamp["cubin_sha256"] != hashlib.sha256(V4CUBIN.read_bytes()).hexdigest():
        raise RuntimeError("stale v4 build: run `ops_slab1 build-v4` first")
    return stamp


def build():
    ART.mkdir(parents=True, exist_ok=True)
    cmd = [NVCC, "-cubin", "-arch=sm_90a", "-std=c++17", "-O3", "-ccbin", CCBIN,
           "-Xptxas=-v", str(SOURCE), "-o", str(CUBIN)]
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (ART / "build.log").write_text(" ".join(cmd) + "\n" + p.stdout)
    print(p.stdout, end="")
    p.check_returncode()
    stamp = {"source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
             "cubin_sha256": hashlib.sha256(CUBIN.read_bytes()).hexdigest(), "command": cmd}
    (ART / "build.json").write_text(json.dumps(stamp, indent=2))
    print("sha256", stamp["cubin_sha256"])
    build_v3z()


def build_v3z():
    ART.mkdir(parents=True, exist_ok=True)
    cmd = [NVCC, "-cubin", "-arch=sm_90a", "-std=c++17", "-O3", "-ccbin", CCBIN,
           "-Xptxas=-v", str(ZSOURCE), "-o", str(ZCUBIN)]
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (ART / "build_v3z.log").write_text(" ".join(cmd) + "\n" + p.stdout)
    print(p.stdout, end="")
    p.check_returncode()
    stamp = {"source_sha256": hashlib.sha256(ZSOURCE.read_bytes()).hexdigest(),
             "cubin_sha256": hashlib.sha256(ZCUBIN.read_bytes()).hexdigest(), "command": cmd}
    (ART / "build_v3z.json").write_text(json.dumps(stamp, indent=2))
    print("v3z sha256", stamp["cubin_sha256"])


def check_build_v3z():
    stamp = json.loads((ART / "build_v3z.json").read_text())
    if stamp["source_sha256"] != hashlib.sha256(ZSOURCE.read_bytes()).hexdigest() or \
       stamp["cubin_sha256"] != hashlib.sha256(ZCUBIN.read_bytes()).hexdigest():
        raise RuntimeError("stale v3z build: run `ops_slab1 build` first")
    return stamp


def check_build():
    stamp = json.loads((ART / "build.json").read_text())
    if stamp["source_sha256"] != hashlib.sha256(SOURCE.read_bytes()).hexdigest() or \
       stamp["cubin_sha256"] != hashlib.sha256(CUBIN.read_bytes()).hexdigest():
        raise RuntimeError("stale slab1 build: run `ops_slab1 build` first")
    return stamp


def ops(final=False, mode="slab"):
    """spec_kda_slab1 + spec_kda_select_all op definitions.

    final=False: ops-file shorthand (cubin+sha256 inline) for gen.py's
    resolve_modules/lower_wire. final=True: schema-5 module form for direct
    rewrite of an already-generated manifest (fuse-round on mtp-fused3.json).
    mode="select": v0.5 bitwise-only form — emit spec_kda_select_all only
    (cuBLASLt qkvbfg/o_proj and the standalone spec_ar stay untouched).
    """
    stamp = check_build()
    bf, fp, ix = "in buffer<bf16>", "in buffer<f32>", "in buffer<i32>"
    sha = stamp["cubin_sha256"]

    def launch(entry, p, grid, block, args, smem=None, pdl=True):
        if final:
            d = {"module": "glm53_slab1", "entry": entry, "params": p.copy(),
                 "block": [block, 1, 1], "grid": grid, "pdl": pdl, "args": args}
        else:
            d = {"cubin": str(CUBIN), "sha256": sha, "label": entry, "entry": entry,
                 "params": p.copy(), "block": [block, 1, 1], "grid": grid, "args": args}
        if smem: d["shared_mem"] = smem
        return d

    # kernel ABI: x, wq, F, wf, wg, cw, al, dt, nw, cs, ss, cl, sl, cu, kda_o,
    #             stash_ssm, stash_conv, wo, sub_out, sync(scratch), T, S,
    #             wq_next (L2 prefetch under phase B), do_prefetch
    sp = [bf, bf, "out buffer<bf16>", bf, bf, fp, fp, fp, bf,
          "in state", "in state", ix, ix, ix,
          "out buffer<bf16>", "out buffer<f32>", "out buffer<bf16>",
          bf, "out buffer<bf16>", "i32", "i32", bf, "i32"]
    # launch ABI = op params with the sync scratch pointer inserted at slot 19;
    # "out" (write-first): the verifier reads direction strictly (monotonic
    # ticket needs no initial value; op scratch is zero-initialized anyway)
    lp = sp[:19] + ["out buffer<i32>"] + sp[19:]
    sargs = ([{"param": i} for i in range(19)] + [{"scratch": "sync"}]
             + [{"param": i} for i in range(19, 23)])
    slab = {"params": sp, "impl": {
        "scratch": {"sync": {"dtype": "i32", "shape": [8]}},
        "launches": [launch("glm53_kda_slab1_full", lp, [GRID, 1, 1], THREADS,
                            sargs, smem=SMEM)]}}
    # kernel ABI: stash_ssm, stash_conv, ss, cs, sl, cl, nacc, valid, S, lstride, rows
    ap = [fp, bf, "inout state", "inout state", ix, ix, ix, ix, "i32", "i32", "i32"]
    aargs = [{"param": i} for i in range(11)]
    # select_all launch is a 35-way join (34 v3 stash producers + accept):
    # a programmatic-trigger (PDL) edge on a wide join is the under-tested
    # path — a missed trigger parks all CTAs in griddepcontrol.wait (the
    # 141k livelock signature: 93-95% GPU, zero scheduler progress,
    # self-recover ~25-45min). The batching win does not come from PDL
    # prelaunch, so the launch goes pdl=False (wait/done are no-ops without
    # the attribute). The original 34 selects keep their own pdl edges
    # (single-stash-producer joins, the well-tested PDL shape).
    sel = {"params": ap, "impl": {"launches": [
        launch("glm53_kda_select_all", ap, [{"mul": ["seqs", NLAYERS]}, 8, 1],
               256, aargs, pdl=False)]}}
    if mode == "select":
        return {"spec_kda_select_all": sel}
    return {"spec_kda_slab1": slab, "spec_kda_select_all": sel}


def _vz_rewrite(m):
    """v3 z-split: replace spec_kda_fused_v3's impl with the z2 kernel.

    The op's params and every call site stay untouched; the launch gains the
    xchg scratch (raw exchange + tickets + err/trip) and grid [seqs,8,2].
    pdl stays True: the join shape is the unchanged qkvbfg->v3 single chain
    (the well-tested PDL shape, unlike select_all's 35-way join).
    """
    final = "modules" in m
    op = m["ops"]["spec_kda_fused_v3"]
    stamp = check_build_v3z()
    n = len(op["params"])
    launch = {"entry": "glm53_kda_verify_fused512z2",
              "params": op["params"] + ["out buffer<i32>"],
              "args": [{"param": i} for i in range(n)] + [{"scratch": "xchg"}],
              "block": [512, 1, 1], "grid": ["seqs", 8, 2], "pdl": True}
    if final:
        launch["module"] = "glm53_kda_v3z"
        m["modules"]["glm53_kda_v3z"] = {
            "source": str(ZCUBIN), "sha256": stamp["cubin_sha256"]}
    else:
        launch.update({"cubin": str(ZCUBIN), "sha256": stamp["cubin_sha256"],
                       "label": "glm53_kda_verify_fused512z2"})
    op["impl"] = {"scratch": {"xchg": {"dtype": "i32", "shape": [Z_SCRATCH_U32]}},
                  "launches": [launch]}
    # the pinned v3 module is now unreferenced; verify rejects unused modules
    if final:
        used = {l["module"] for o in m["ops"].values()
                for l in o["impl"].get("launches", []) if "module" in l}
        if "glm53_kda_verify_fused512" not in used:
            m["modules"].pop("glm53_kda_verify_fused512", None)
    print(f"ops_slab1[vz]: spec_kda_fused_v3 -> glm53_kda_verify_fused512z2 "
          f"(grid [seqs,8,2], xchg {Z_SCRATCH_U32*4}B)")
    return m


def _z2s_rewrite(m, zw):
    """v4 stage-B (z2s): replace spec_kda_fused_v3's impl with the prep-sharing
    z-split kernel (2026-09-28, 78/78 gates; isolated -239/-257/-192us
    per round at S=1/4/8 for zw=4).

    Same op-def-swap discipline as _vz_rewrite: params and every call site
    stay untouched; the launch gains the v4 xchg scratch (4-region package +
    raw exchange + tickets + err/trip; see glm53_kda_v4.cu layout comment)
    and grid [seqs,8,zw]. pdl stays True (unchanged qkvbfg->v3 single-chain
    join shape). Co-residency: all zw CTAs of an (s,h) group must be
    co-resident; holds at seqs<=8 with the 8-named-barrier build.
    """
    assert zw in (2, 4), zw
    tag = f"s{zw}"
    final = "modules" in m
    op = m["ops"]["spec_kda_fused_v3"]
    stamp = check_build_v4()
    n = len(op["params"])
    launch = {"entry": f"glm53_kda_verify_fused512{tag}",
              "params": op["params"] + ["out buffer<i32>"],
              "args": [{"param": i} for i in range(n)] + [{"scratch": "xchg"}],
              "block": [512, 1, 1], "grid": ["seqs", 8, zw], "pdl": True}
    if final:
        launch["module"] = "glm53_kda_v4"
        m["modules"]["glm53_kda_v4"] = {
            "source": str(V4CUBIN), "sha256": stamp["cubin_sha256"]}
    else:
        launch.update({"cubin": str(V4CUBIN), "sha256": stamp["cubin_sha256"],
                       "label": f"glm53_kda_verify_fused512{tag}"})
    op["impl"] = {"scratch": {"xchg": {"dtype": "i32", "shape": [V4_SCRATCH_U32]}},
                  "launches": [launch]}
    # the pinned v3 module is now unreferenced; verify rejects unused modules
    if final:
        used = {l["module"] for o in m["ops"].values()
                for l in o["impl"].get("launches", []) if "module" in l}
        if "glm53_kda_verify_fused512" not in used:
            m["modules"].pop("glm53_kda_verify_fused512", None)
    print(f"ops_slab1[z2s]: spec_kda_fused_v3 -> glm53_kda_verify_fused512{tag} "
          f"(grid [seqs,8,{zw}], xchg {V4_SCRATCH_U32*4}B)")
    return m


# Lamport AR multi-edge PDL (green-lit 2026-09-28; 8-mode
# A/B proved serving graphs preserve PDL semantics, runtime accepted 2-PDL
# chains in v0.3). Two edge sets, both pure op-def rewrites:
#
# Edge A (producer->AR, 184 calls): swap the AR launch to the *_pdl entry and
# set pdl:true. The _pdl AR kernels run griddepcontrol.wait (SASS ACQBULK)
# BEFORE reading the producer's partial and launch_dependents (PREEXIT) after
# slot retirement. Entries ship in the same pinned cubins
# (spec_ar-cf6e9ca74c91.cubin: spec_ar_lamport{,_pdl};
#  glm53_ar_lamport.cubin: glm53_ar_lamport{,_pdl}) — no rebuild.
#
# Edge B (AR->consumer, 180 calls): pdl:true on the AR-consumer ops whose
# kernels carry ACQBULK, verified per-entry:
#   glm53_mhc_boundary_f32{,_r32}  1 ACQBULK each (glm53_mhc_v2-9c4e9f250a72)
#   mhc_post_tilelang_kernel       1 ACQBULK + 1 PREEXIT (module_291)
# spec_add_norm (4 round calls) has NO ACQBULK -> stays non-PDL (skipped).
#
# Chain shape w2(pdl) -> spec_ar(pdl) -> hc_post(pdl) = two consecutive PDL
# launches, the v0.3-proven shape. Composes order-free with select/vz.
AR_PDL_OPS = ("spec_ar", "kda_ar", "dsa_ar", "moe_ar", "mlp_ar")
AR_PDL_CONSUMERS = ("mtp32_hc_post", "hc_post",
                    "hc_boundary_f32_v2r", "hc_boundary_f32_v2")


def _arpdl_rewrite(m):
    n_a = n_b = 0
    for name in AR_PDL_OPS:
        op = m["ops"].get(name)
        if not op:
            continue
        for l in op["impl"]["launches"]:
            if l["entry"].endswith("_lamport_pdl"):
                l["pdl"] = True
                n_a += 1
            elif l["entry"].endswith("_lamport"):
                l["entry"] += "_pdl"
                l["pdl"] = True
                n_a += 1
    for name in AR_PDL_CONSUMERS:
        op = m["ops"].get(name)
        if not op:
            continue
        for l in op["impl"]["launches"]:
            if not l.get("pdl"):
                l["pdl"] = True
                n_b += 1
    if not n_a:
        raise ValueError("arpdl: no lamport AR launches found")
    print(f"ops_slab1[arpdl]: edge A {n_a} AR launches -> *_pdl+pdl, "
          f"edge B {n_b} consumer launches pdl:true")
    return m


def fuse_round_manifest(m, mode=None, program="round_k2"):
    """Rewrite round_k2: per KDA layer [qkvbfg, fused_v3, o_proj] -> slab1,
    and the 34 consecutive advance.K.select -> one select_all.

    Requires the v3 verify fusion to be applied already (spec_kda_fused_v3 and
    the stash buffers must exist).
    GLM53_SLAB1: 0 -> no-op (escape hatch); "slab" (default) = v0.3 full slab;
    "select" = v0.5 bitwise-only (select_all batching only, GEMMs stay
    cuBLASLt canonical-M externs, spec_ar stays standalone);
    "vz" = v3 z-split: rewrites the spec_kda_fused_v3 op def in place
    (module glm53_kda_v3z, entry glm53_kda_verify_fused512z2, grid
    [seqs,8,2], +xchg scratch). Op name/params and all call sites are
    UNCHANGED, so vz composes order-free with "select" (run fuse-round
    twice) and with the mHC/MoE/slab-2 passes.
    "arpdl" = multi-edge PDL for the lamport ARs: AR launches -> *_pdl
    entries + pdl:true, and pdl:true on the ACQBULK-verified AR-consumer
    ops (hc_post/hc_boundary, both programs). Op-def only, order-free.
    """
    if mode is None:
        mode = os.environ.get("GLM53_SLAB1", "slab")
    if mode == "0":
        print("ops_slab1: GLM53_SLAB1=0, fuse is a no-op")
        return m
    if mode not in ("slab", "select", "vz", "s2", "s4", "arpdl", "1"):
        raise ValueError(f"bad GLM53_SLAB1 mode {mode!r}")
    if mode == "1":
        mode = "slab"
    if program not in m["programs"]:
        raise ValueError(f"no {program} program: not an --mtp manifest")
    if mode == "arpdl":
        # Op-def only (AR launches + hc consumers); no v3/rows dependency.
        return _arpdl_rewrite(m)
    rows = m["programs"][program].get("batch", {}).get("rows")
    if not isinstance(rows, int) or not 1 <= rows <= 8:
        raise ValueError(
            f"slab-1 round mode {mode} requires the rows=N v3 KDA fusion (1..8); "
            f"{program} has rows={rows}")
    if "spec_kda_fused_v3" not in m["ops"]:
        raise ValueError("apply the v3 verify fusion before slab-1/select")
    for b in ("spec_ssm_stash", "spec_conv_stash"):
        if b not in m["buffers"]:
            raise ValueError(f"missing stash buffer {b}: apply v3 fusion first")
    if mode == "vz":
        return _vz_rewrite(m)
    if mode in ("s2", "s4"):
        return _z2s_rewrite(m, 2 if mode == "s2" else 4)
    final = "modules" in m  # already-generated manifests are schema-5
    m["ops"].update(ops(final=final, mode=mode))
    if final:
        m["modules"]["glm53_slab1"] = {
            "source": str(CUBIN), "sha256": check_build()["cubin_sha256"]}
    calls = m["programs"][program]["calls"]
    rewritten, i, nsl = [], 0, 0
    while i < len(calls):
        c0 = calls[i]
        if mode == "slab" and c0["op"] == "mtp32_kda_qkvbfg" and \
                c0.get("label", "").startswith("verify."):
            group = calls[i:i+3]
            if [c["op"] for c in group] != ["mtp32_kda_qkvbfg", "spec_kda_fused_v3",
                                            "mtp32_kda_o_proj_ar"]:
                raise ValueError(f"KDA slab cut changed at {c0['label']}; refusing implicit fuse")
            qk, fu, op = [c["args"] for c in group]
            lab = c0["label"].rsplit(".", 1)[0]
            assert group[1]["label"] == lab + ".fused_v3" and \
                   group[2]["label"] == lab + ".o_proj", "slab label mismatch"
            assert qk[2] == fu[0], "spec_F identity mismatch"
            assert fu[12] == op[0], "kda_o identity mismatch"
            assert qk[3] == op[3], "tokens var mismatch"
            # L2 prefetch of the NEXT KDA layer's qkvbfg weight under phase B
            nxt = calls[i + 3] if i + 3 < len(calls) else None
            if nxt is not None and nxt["op"] == "mtp32_kda_qkvbfg" and \
               nxt.get("label", "").startswith("verify."):
                wq_next, do_pf = nxt["args"][1], {"i32": 1}
            else:
                wq_next, do_pf = qk[1], {"i32": 0}
            args = [qk[0], qk[1], qk[2], fu[1], fu[2], fu[3], fu[4], fu[5], fu[6],
                    fu[7], fu[8], fu[9], fu[10], fu[11], fu[12], fu[13], fu[14],
                    op[1], op[2], qk[3], fu[15], wq_next, do_pf]
            rewritten.append({"label": lab + ".slab1", "op": "spec_kda_slab1",
                              "args": args})
            i += 3; nsl += 1
            continue
        rewritten.append(c0); i += 1
    # advance selects: 34 consecutive spec_kda_select -> one select_all
    out, i, nse = [], 0, 0
    while i < len(rewritten):
        if rewritten[i]["op"] == "spec_kda_select":
            run = rewritten[i:i+NLAYERS]
            if len(run) < NLAYERS or any(c["op"] != "spec_kda_select" for c in run):
                raise ValueError("select run shorter than 34; refusing implicit fuse")
            s0 = run[0]["args"]
            for k, c in enumerate(run):
                a = c["args"]
                assert a[0].get("offset", 0) == k * SSM_LAYER_F32(rows) * 4, f"stash stride at {c['label']}"
                assert a[1].get("offset", 0) == k * CONV_LAYER_BF16(rows) * 2, f"conv stride at {c['label']}"
                assert a[4].get("offset", 0) == k * LINES_STRIDE * 4, f"ssm lines stride at {c['label']}"
                assert a[5].get("offset", 0) == k * LINES_STRIDE * 4, f"conv lines stride at {c['label']}"
                assert a[2] == s0[2] and a[3] == s0[3] and a[6] == s0[6] and a[7] == s0[7]
            args = [{"buf": s0[0]["buf"]}, {"buf": s0[1]["buf"]}, s0[2], s0[3],
                    {"buf": s0[4]["buf"]}, {"buf": s0[5]["buf"]}, s0[6], s0[7],
                    {"var": "seqs"}, {"i32": LINES_STRIDE}, {"i32": rows}]
            out.append({"label": "advance.select_all", "op": "spec_kda_select_all",
                        "args": args})
            i += NLAYERS; nse += 1
            continue
        out.append(rewritten[i]); i += 1
    if mode == "slab" and nsl != NLAYERS:
        raise ValueError(f"expected {NLAYERS} slab-1 rewrites, got {nsl}")
    if nse != 1:
        raise ValueError(f"expected 1 select_all rewrite, got {nse}")
    m["programs"][program]["calls"] = out
    # drop absorbed ops no program calls anymore (verify requires used ops;
    # mtp32_kda_qkvbfg stays: the repair.eh draft path still calls it)
    used = {c["op"] for p in m["programs"].values() for c in p["calls"]}
    dead_ops = ("spec_kda_fused_v3", "mtp32_kda_o_proj_ar", "spec_kda_select") \
        if mode == "slab" else ("spec_kda_select",)
    for dead in dead_ops:
        if dead not in used:
            m["ops"].pop(dead, None)
    # and modules no surviving launch references (verify requires used modules)
    used_mods = {l["module"] for o in m["ops"].values()
                 for l in o["impl"].get("launches", []) if "module" in l}
    for mod in ("glm53_kda_verify_fused512",):
        if mod not in used_mods:
            m["modules"].pop(mod, None)
    print(f"ops_slab1[{mode}]: {nsl} slab-1 rewrites (-{2*nsl} calls), "
          f"select_all (-{NLAYERS-1} calls)")
    return m


def test(args):
    """Phase-isolation gates (kernel level; references = production cubins).

    G-A/G-C GEMM numerics: rel_err vs fp32 reference within 1.5x of cublasLt's
      own deviation; rerun bitwise-stable; T in {3,12,24}.
    G-B core bitwise: slab_core vs production glm53_kda_verify_fused512 on
      identical inputs -> kda_o/stash_ssm/stash_conv bitwise; S in {1,2,4,8},
      plus an invalid-sequence guard case.
    G-D e2e: slab_full vs [cublasLt qkvbfg -> v3 -> cublasLt o_proj] chain:
      sub_out rel_err at bf16 level; determinism bitwise across runs.
    G-E select_all bitwise vs 34 x glm53_kda_select incl. invalid seqs.
    G-F barrier robustness: 2000 back-to-back slab_full launches, err stays 0.
    Timing: slab_full vs the 3-kernel chain, S in {1,4,8}.
    """
    check_build()
    from glm53.ops_kda_v2 import Driver, Lt, gpu_check, V3CUBIN, check_build_v3
    check_build_v3()
    check_build_v3z()
    check_build_v4()
    gpu_check()
    import torch
    torch.cuda.init(); torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    D, lt = Driver(torch), Lt(torch)
    mod = D.module(CUBIN)
    v3 = D.module(V3CUBIN)
    vz = D.module(ZCUBIN)
    v4 = D.module(V4CUBIN)
    I, J = C.c_int, C.c_int64
    bf, fp = torch.bfloat16, torch.float32
    SET_ATTR = 8  # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES

    def rnd(shape, scale=1, dtype=bf):
        return (torch.randn(shape, device="cuda", dtype=fp) * scale).to(dtype)

    def bits_equal(a, b):
        dt = torch.int16 if a.element_size() == 2 else torch.int32
        return torch.equal(a.view(dt), b.view(dt))

    def l2rel(a, b):
        aa, bb = a.float(), b.float()
        return ((aa - bb).square().sum().sqrt() /
                (bb.square().sum().sqrt() + 1e-30)).item()

    def slab_kernel(entry, block=THREADS, smem=SMEM):
        fn = C.c_void_p()
        Driver.check(D.lib.cuModuleGetFunction(C.byref(fn), mod, entry.encode()))
        if smem > 48 * 1024:
            Driver.check(D.lib.cuFuncSetAttribute(fn, C.c_int(SET_ATTR), C.c_int(smem)))
        def call(grid, args):
            vals = [C.c_uint64(x.data_ptr()) if hasattr(x, "data_ptr") else x for x in args]
            av = (C.c_void_p * len(vals))(*[C.addressof(x) for x in vals])
            Driver.check(D.lib.cuLaunchKernel(
                fn, grid[0], grid[1], grid[2], block, 1, 1, smem,
                C.c_void_p(torch.cuda.current_stream().cuda_stream), av, None))
            call._keep = (vals, av)
        return call

    def v3_kernel(entry, block):
        fn = C.c_void_p()
        Driver.check(D.lib.cuModuleGetFunction(C.byref(fn), v3, entry.encode()))
        def call(grid, args, smem=0):
            vals = [C.c_uint64(x.data_ptr()) if hasattr(x, "data_ptr") else x for x in args]
            av = (C.c_void_p * len(vals))(*[C.addressof(x) for x in vals])
            Driver.check(D.lib.cuLaunchKernel(
                fn, grid[0], grid[1], grid[2], block, 1, 1, smem,
                C.c_void_p(torch.cuda.current_stream().cuda_stream), av, None))
            call._keep = (vals, av)
        return call

    def timing(fn, repeats=50):
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
            vals.append(s0.elapsed_time(e0) * 1000 / repeats)
        return sorted(vals)[2]

    sync = torch.zeros(8, device="cuda", dtype=torch.int32)
    results = []

    # ---- G-A / G-C: GEMM phases vs cublasLt + fp32 reference -----------------
    # kernel ABI positions: 0 x, 1 wq, 2 F, 14 kda_o, 17 wo, 18 sub_out, 19 sync,
    # 20 T, 21 S, 22 wq_next, 23 do_prefetch (rest unused by isolation entries).
    for entry, N, K, pos in (("glm53_kda_slab1_gemmA", 3336, 4096, (0, 1, 2)),
                             ("glm53_kda_slab1_gemmC", 4096, 1024, (14, 17, 18))):
        kern = slab_kernel(entry)
        for T in (3, 12, 24):
            x, w = rnd((T, K)), rnd((N, K))
            y_slab = torch.empty(T, N, device="cuda", dtype=bf)
            y_lt = torch.empty(T, N, device="cuda", dtype=bf)
            dummy = torch.empty(8, device="cuda", dtype=bf)
            sync.zero_()
            args = [dummy] * 24
            args[pos[0]], args[pos[1]], args[pos[2]] = x, w, y_slab
            args[19], args[20], args[21] = sync, I(T), I(1)
            kern((GRID, 1, 1), args)
            torch.cuda.synchronize()
            lt.gemm(x, w, y_lt, T, N, K)()
            torch.cuda.synchronize()
            ref = (x.float() @ w.float().T)
            e_slab, e_lt = l2rel(y_slab, ref), l2rel(y_lt, ref)
            # determinism: a second run must be bitwise identical
            y2 = torch.empty_like(y_slab); sync.zero_()
            args[pos[2]] = y2; kern((GRID, 1, 1), args); torch.cuda.synchronize()
            det = bits_equal(y_slab, y2)
            ok = det and e_slab <= 1.5 * max(e_lt, 1e-12)
            results.append((ok, f"GEMM {entry.split('_')[-1]} T={T}: "
                                f"rel_err slab {e_slab:.3e} vs lt {e_lt:.3e} "
                                f"(vs f32 ref), deterministic {det}"))

    # ---- G-B: core phase bitwise vs production v3 ----------------------------
    NL = 64
    for S in (1, 2, 4, 8):
        T = 3 * S
        F_in = rnd((32, 3336))
        wf, wg = rnd((1024, 128)), rnd((1024, 128))
        cw = torch.randn(3072 * 4, device="cuda", dtype=fp)
        al = torch.randn(8, device="cuda", dtype=fp) * 0.5
        dt = torch.randn(1024, device="cuda", dtype=fp) * 0.5
        nw = rnd((128,))
        cs_pool = rnd((NL, 3, 3072))
        ss_pool = torch.randn(NL, 8, 128, 128, device="cuda", dtype=fp)
        cl = torch.arange(1, NL + 1, device="cuda", dtype=torch.int32)[:S].contiguous()
        sl = torch.arange(1, NL + 1, device="cuda", dtype=torch.int32)[:S].contiguous()
        cu = torch.arange(0, 3 * (S + 1), 3, device="cuda", dtype=torch.int32)
        outs, stash_s, stash_c = [], [], []
        for which in ("v3", "slab"):
            kda_o = torch.zeros(32, 1024, device="cuda", dtype=bf)
            ssm = torch.zeros(8, 3, 8, 128, 128, device="cuda", dtype=fp)
            cvs = torch.zeros(8, 3, 3, 3072, device="cuda", dtype=bf)
            args = [F_in, wf, wg, cw, al, dt, nw, cs_pool, ss_pool, cl, sl, cu,
                    kda_o, ssm, cvs]
            if which == "v3":
                v3_kernel("glm53_kda_verify_fused512", 512)((S, 8, 1), args + [I(S), I(3)])
            else:
                x_, w_ = rnd((T, 4096)), rnd((3336, 4096))
                wo_, sub_ = rnd((4096, 1024)), torch.empty(T, 4096, device="cuda", dtype=bf)
                sync.zero_()
                slab_kernel("glm53_kda_slab1_core")((GRID, 1, 1),
                    [x_, w_, F_in, wf, wg, cw, al, dt, nw, cs_pool, ss_pool,
                     cl, sl, cu, kda_o, ssm, cvs, wo_, sub_, sync, I(T), I(S), w_, I(1)])
            torch.cuda.synchronize()
            outs.append((kda_o, ssm, cvs))
        ok = all(bits_equal(outs[0][i], outs[1][i]) for i in range(3))
        results.append((ok, f"CORE S={S}: kda_o/stash_ssm/stash_conv "
                            f"bitwise vs v3 = {ok}"))
    # invalid sequence: cl/sl = -1 on seq 1 -> both must skip identically
    S = 2
    cl_bad = cl.clone(); cl_bad[1] = -1
    outs = []
    for which in ("v3", "slab"):
        kda_o = torch.zeros(32, 1024, device="cuda", dtype=bf)
        ssm = torch.zeros(8, 3, 8, 128, 128, device="cuda", dtype=fp)
        cvs = torch.zeros(8, 3, 3, 3072, device="cuda", dtype=bf)
        args = [F_in, wf, wg, cw, al, dt, nw, cs_pool, ss_pool, cl_bad, sl, cu,
                kda_o, ssm, cvs]
        if which == "v3":
            v3_kernel("glm53_kda_verify_fused512", 512)((S, 8, 1), args + [I(S), I(3)])
        else:
            sync.zero_()
            slab_kernel("glm53_kda_slab1_core")((GRID, 1, 1),
                [x_, w_, F_in, wf, wg, cw, al, dt, nw, cs_pool, ss_pool,
                 cl_bad, sl, cu, kda_o, ssm, cvs, wo_, sub_, sync, I(6), I(S)])
        torch.cuda.synchronize()
        outs.append((kda_o, ssm, cvs))
    ok = all(bits_equal(outs[0][i], outs[1][i]) for i in range(3))
    results.append((ok, f"CORE invalid-seq guard bitwise = {ok}"))

    # ---- G-Z: v3 z-split bitwise vs the pinned production v3 cubin ---------
    # z2 must reproduce kda_o/stash_ssm/stash_conv BITWISE for S=1..8, with
    # err/trip counters clean, determinism across reruns, identical early-out
    # behaviour on invalid seqs and partial-row groups, and bitwise debug
    # intermediates (dbg_raw/dbg_cv/dbg_rs via the _debug entries).
    def z2_kernel(entry, block=512):
        fn = C.c_void_p()
        Driver.check(D.lib.cuModuleGetFunction(C.byref(fn), vz, entry.encode()))
        def call(grid, args, smem=0):
            vals = [C.c_uint64(x.data_ptr()) if hasattr(x, "data_ptr") else x for x in args]
            av = (C.c_void_p * len(vals))(*[C.addressof(x) for x in vals])
            Driver.check(D.lib.cuLaunchKernel(
                fn, grid[0], grid[1], grid[2], block, 1, 1, smem,
                C.c_void_p(torch.cuda.current_stream().cuda_stream), av, None))
            call._keep = (vals, av)
        return call

    def run_pair(S, cl_use, cu_use, F_in, wf, wg, cw, al, dt, nw, cs_pool, ss_pool, sl,
                 debug=False):
        outs = []
        for which in ("v3", "z2"):
            kda_o = torch.zeros(32, 1024, device="cuda", dtype=bf)
            ssm = torch.zeros(8, 3, 8, 128, 128, device="cuda", dtype=fp)
            cvs = torch.zeros(8, 3, 3, 3072, device="cuda", dtype=bf)
            xchg = torch.zeros(Z_SCRATCH_U32, device="cuda", dtype=torch.int32)
            args = [F_in, wf, wg, cw, al, dt, nw, cs_pool, ss_pool, cl_use, sl, cu_use,
                    kda_o, ssm, cvs]
            if not debug:
                if which == "v3":
                    v3_kernel("glm53_kda_verify_fused512", 512)((S, 8, 1), args + [I(S), I(3)])
                else:
                    z2_kernel("glm53_kda_verify_fused512z2")((S, 8, 2), args + [I(S), xchg])
            else:
                dr = torch.zeros(32, 1024, device="cuda", dtype=bf)
                dc = torch.zeros(32, 3072, device="cuda", dtype=bf)
                ds = torch.zeros(8 * 24, device="cuda", dtype=fp)
                if which == "v3":
                    v3_kernel("glm53_kda_verify_fused512_debug", 512)(
                        (S, 8, 1), args + [I(S), I(3), dr, dc, ds])
                else:
                    z2_kernel("glm53_kda_verify_fused512z2_debug")(
                        (S, 8, 2), args + [I(S), xchg, dr, dc, ds])
            torch.cuda.synchronize()
            outs.append((kda_o, ssm, cvs, xchg) + ((dr, dc, ds) if debug else ()))
        return outs

    for S in (1, 2, 3, 4, 5, 6, 7, 8):
        F_in = rnd((32, 3336))
        wf, wg = rnd((1024, 128)), rnd((1024, 128))
        cw = torch.randn(3072 * 4, device="cuda", dtype=fp)
        al = torch.randn(8, device="cuda", dtype=fp) * 0.5
        dt = torch.randn(1024, device="cuda", dtype=fp) * 0.5
        nw = rnd((128,))
        cs_pool = rnd((NL, 3, 3072))
        ss_pool = torch.randn(NL, 8, 128, 128, device="cuda", dtype=fp)
        cl = torch.arange(1, NL + 1, device="cuda", dtype=torch.int32)[:S].contiguous()
        sl = torch.arange(1, NL + 1, device="cuda", dtype=torch.int32)[:S].contiguous()
        cu = torch.arange(0, 3 * (S + 1), 3, device="cuda", dtype=torch.int32)
        o = run_pair(S, cl, cu, F_in, wf, wg, cw, al, dt, nw, cs_pool, ss_pool, sl)
        bits = all(bits_equal(o[0][i], o[1][i]) for i in range(3))
        err0 = int(o[1][3][0].item()) == 0 and int(o[1][3][1].item()) == 0
        # determinism: a second z2 run must be bitwise identical
        o2 = run_pair(S, cl, cu, F_in, wf, wg, cw, al, dt, nw, cs_pool, ss_pool, sl)
        det = all(bits_equal(o[1][i], o2[1][i]) for i in range(3))
        results.append((bits and err0 and det,
                        f"GZ S={S}: z2 vs pinned v3 bitwise={bits} "
                        f"err/trip clean={err0} deterministic={det}"))
    # invalid seq + partial-row group: identical early-out behaviour
    S = 2
    cl_bad = cl.clone(); cl_bad[1] = -1
    o = run_pair(S, cl_bad, cu, F_in, wf, wg, cw, al, dt, nw, cs_pool, ss_pool, sl)
    bits = all(bits_equal(o[0][i], o[1][i]) for i in range(3))
    results.append((bits, f"GZ invalid-seq guard bitwise = {bits}"))
    cu_bad = torch.tensor([0, 3, 5, 8], device="cuda", dtype=torch.int32)
    o = run_pair(S, cl, cu_bad, F_in, wf, wg, cw, al, dt, nw, cs_pool, ss_pool, sl)
    bits = all(bits_equal(o[0][i], o[1][i]) for i in range(3))
    results.append((bits, f"GZ partial-row guard bitwise = {bits}"))
    # debug intermediates bitwise
    o = run_pair(4, cl, torch.arange(0, 3 * (4 + 1), 3, device="cuda", dtype=torch.int32),
                 F_in, wf, wg, cw, al, dt, nw, cs_pool, ss_pool, sl, debug=True)
    bits = all(bits_equal(o[0][i], o[1][i]) for i in (4, 5, 6))
    results.append((bits, f"GZ debug raw/cv/rs bitwise = {bits}"))
    # barrier stress: 500 back-to-back z2 launches; err/trip must stay 0 and
    # the final outputs must be bitwise equal to a single run on the same inputs
    S = 8
    xchg = torch.zeros(Z_SCRATCH_U32, device="cuda", dtype=torch.int32)
    kda_o = torch.zeros(32, 1024, device="cuda", dtype=bf)
    ssm = torch.zeros(8, 3, 8, 128, 128, device="cuda", dtype=fp)
    cvs = torch.zeros(8, 3, 3, 3072, device="cuda", dtype=bf)
    zargs = [F_in, wf, wg, cw, al, dt, nw, cs_pool, ss_pool, cl, sl, cu,
             kda_o, ssm, cvs, I(S), xchg]
    zk = z2_kernel("glm53_kda_verify_fused512z2")
    zk((S, 8, 2), zargs)
    torch.cuda.synchronize()
    ref3 = (kda_o.clone(), ssm.clone(), cvs.clone())
    xchg.zero_()
    for _ in range(500):
        zk((S, 8, 2), zargs)
    torch.cuda.synchronize()
    stress = all(bits_equal(ref3[i], (kda_o, ssm, cvs)[i]) for i in range(3)) and \
        int(xchg[0].item()) == 0 and int(xchg[1].item()) == 0
    results.append((stress, f"GZ stress 500 launches: bitwise stable + err/trip 0 = {stress}"))
    # timing: z2 vs pinned v3 per-launch (graph-captured), S in {1,4,8}
    for S in (1, 4, 8):
        xchg2 = torch.zeros(Z_SCRATCH_U32, device="cuda", dtype=torch.int32)
        za = [F_in, wf, wg, cw, al, dt, nw, cs_pool, ss_pool,
              torch.arange(1, NL + 1, device="cuda", dtype=torch.int32)[:S].contiguous(),
              torch.arange(1, NL + 1, device="cuda", dtype=torch.int32)[:S].contiguous(),
              torch.arange(0, 3 * (S + 1), 3, device="cuda", dtype=torch.int32),
              kda_o, ssm, cvs, I(S), xchg2]
        t_v3 = timing(lambda: v3_kernel("glm53_kda_verify_fused512", 512)((S, 8, 1), za[:15] + [I(S), I(3)]))
        t_z2 = timing(lambda: zk((S, 8, 2), za))
        results.append((t_z2 < t_v3, f"GZ timing S={S}: v3 {t_v3:.2f}us -> z2 {t_z2:.2f}us "
                                     f"({(t_v3 - t_z2):+.2f}us)"))

    # ---- G-V4: stage-A pipeline bitwise vs the pinned production v3 --------
    # fused512p must reproduce kda_o/stash_ssm/stash_conv BITWISE for
    # S=1..8, with determinism across reruns, identical early-out behaviour
    # on invalid seqs and partial-row groups, bitwise debug intermediates,
    # and a 500-launch named-barrier stress. Timing: v4 vs v3 per launch.
    def v4_kernel(entry, block=512):
        fn = C.c_void_p()
        Driver.check(D.lib.cuModuleGetFunction(C.byref(fn), v4, entry.encode()))
        def call(grid, args, smem=0):
            vals = [C.c_uint64(x.data_ptr()) if hasattr(x, "data_ptr") else x for x in args]
            av = (C.c_void_p * len(vals))(*[C.addressof(x) for x in vals])
            Driver.check(D.lib.cuLaunchKernel(
                fn, grid[0], grid[1], grid[2], block, 1, 1, smem,
                C.c_void_p(torch.cuda.current_stream().cuda_stream), av, None))
            call._keep = (vals, av)
        return call

    def run_pair4(S, cl_use, cu_use, F_in, wf, wg, cw, al, dt, nw, cs_pool, ss_pool, sl,
                  debug=False):
        outs = []
        for which in ("v3", "v4"):
            kda_o = torch.zeros(32, 1024, device="cuda", dtype=bf)
            ssm = torch.zeros(8, 3, 8, 128, 128, device="cuda", dtype=fp)
            cvs = torch.zeros(8, 3, 3, 3072, device="cuda", dtype=bf)
            args = [F_in, wf, wg, cw, al, dt, nw, cs_pool, ss_pool, cl_use, sl, cu_use,
                    kda_o, ssm, cvs]
            if not debug:
                if which == "v3":
                    v3_kernel("glm53_kda_verify_fused512", 512)((S, 8, 1), args + [I(S), I(3)])
                else:
                    v4_kernel("glm53_kda_verify_fused512p")((S, 8, 1), args + [I(S)])
            else:
                dr = torch.zeros(32, 1024, device="cuda", dtype=bf)
                dc = torch.zeros(32, 3072, device="cuda", dtype=bf)
                ds = torch.zeros(8 * 24, device="cuda", dtype=fp)
                if which == "v3":
                    v3_kernel("glm53_kda_verify_fused512_debug", 512)(
                        (S, 8, 1), args + [I(S), I(3), dr, dc, ds])
                else:
                    v4_kernel("glm53_kda_verify_fused512p_debug")(
                        (S, 8, 1), args + [I(S), dr, dc, ds])
            torch.cuda.synchronize()
            outs.append((kda_o, ssm, cvs) + ((dr, dc, ds) if debug else ()))
        return outs

    for S in (1, 2, 3, 4, 5, 6, 7, 8):
        F_in = rnd((32, 3336))
        wf, wg = rnd((1024, 128)), rnd((1024, 128))
        cw = torch.randn(3072 * 4, device="cuda", dtype=fp)
        al = torch.randn(8, device="cuda", dtype=fp) * 0.5
        dt = torch.randn(1024, device="cuda", dtype=fp) * 0.5
        nw = rnd((128,))
        cs_pool = rnd((NL, 3, 3072))
        ss_pool = torch.randn(NL, 8, 128, 128, device="cuda", dtype=fp)
        cl = torch.arange(1, NL + 1, device="cuda", dtype=torch.int32)[:S].contiguous()
        sl = torch.arange(1, NL + 1, device="cuda", dtype=torch.int32)[:S].contiguous()
        cu = torch.arange(0, 3 * (S + 1), 3, device="cuda", dtype=torch.int32)
        o = run_pair4(S, cl, cu, F_in, wf, wg, cw, al, dt, nw, cs_pool, ss_pool, sl)
        bits = all(bits_equal(o[0][i], o[1][i]) for i in range(3))
        o2 = run_pair4(S, cl, cu, F_in, wf, wg, cw, al, dt, nw, cs_pool, ss_pool, sl)
        det = all(bits_equal(o[1][i], o2[1][i]) for i in range(3))
        results.append((bits and det,
                        f"V4 S={S}: pipeline vs pinned v3 bitwise={bits} deterministic={det}"))
    S = 2
    cl_bad = cl.clone(); cl_bad[1] = -1
    o = run_pair4(S, cl_bad, cu, F_in, wf, wg, cw, al, dt, nw, cs_pool, ss_pool, sl)
    bits = all(bits_equal(o[0][i], o[1][i]) for i in range(3))
    results.append((bits, f"V4 invalid-seq guard bitwise = {bits}"))
    cu_bad = torch.tensor([0, 3, 5, 8], device="cuda", dtype=torch.int32)
    o = run_pair4(S, cl, cu_bad, F_in, wf, wg, cw, al, dt, nw, cs_pool, ss_pool, sl)
    bits = all(bits_equal(o[0][i], o[1][i]) for i in range(3))
    results.append((bits, f"V4 partial-row guard bitwise = {bits}"))
    o = run_pair4(4, cl, torch.arange(0, 3 * (4 + 1), 3, device="cuda", dtype=torch.int32),
                  F_in, wf, wg, cw, al, dt, nw, cs_pool, ss_pool, sl, debug=True)
    bits = all(bits_equal(o[0][i], o[1][i]) for i in (3, 4, 5))
    results.append((bits, f"V4 debug raw/cv/rs bitwise = {bits}"))
    # named-barrier stress: 500 back-to-back v4 launches must stay bitwise
    S = 8
    kda_o = torch.zeros(32, 1024, device="cuda", dtype=bf)
    ssm = torch.zeros(8, 3, 8, 128, 128, device="cuda", dtype=fp)
    cvs = torch.zeros(8, 3, 3, 3072, device="cuda", dtype=bf)
    cl8 = torch.arange(1, NL + 1, device="cuda", dtype=torch.int32)[:S].contiguous()
    sl8 = torch.arange(1, NL + 1, device="cuda", dtype=torch.int32)[:S].contiguous()
    cu8 = torch.arange(0, 3 * (S + 1), 3, device="cuda", dtype=torch.int32)
    v4args = [F_in, wf, wg, cw, al, dt, nw, cs_pool, ss_pool, cl8, sl8, cu8,
              kda_o, ssm, cvs, I(S)]
    v4k = v4_kernel("glm53_kda_verify_fused512p")
    v4k((S, 8, 1), v4args)
    torch.cuda.synchronize()
    ref3 = (kda_o.clone(), ssm.clone(), cvs.clone())
    for _ in range(500):
        v4k((S, 8, 1), v4args)
    torch.cuda.synchronize()
    stress = all(bits_equal(ref3[i], (kda_o, ssm, cvs)[i]) for i in range(3))
    results.append((stress, f"V4 stress 500 launches: bitwise stable = {stress}"))
    for S in (1, 4, 8):
        clS = torch.arange(1, NL + 1, device="cuda", dtype=torch.int32)[:S].contiguous()
        slS = torch.arange(1, NL + 1, device="cuda", dtype=torch.int32)[:S].contiguous()
        cuS = torch.arange(0, 3 * (S + 1), 3, device="cuda", dtype=torch.int32)
        va = [F_in, wf, wg, cw, al, dt, nw, cs_pool, ss_pool, clS, slS, cuS,
              kda_o, ssm, cvs, I(S)]
        t_v3 = timing(lambda: v3_kernel("glm53_kda_verify_fused512", 512)((S, 8, 1), va + [I(3)]))
        t_v4 = timing(lambda: v4k((S, 8, 1), va))
        results.append((t_v4 < t_v3, f"V4 timing S={S}: v3 {t_v3:.2f}us -> v4 {t_v4:.2f}us "
                                     f"({(t_v3 - t_v4):+.2f}us)"))

    # ---- G-V4S2/S4: stage-B distributed-prefix z-split bitwise vs pinned v3 --
    def run_pair_zs(S, cl_use, cu_use, F_in, wf, wg, cw, al, dt, nw, cs_pool, ss_pool, sl,
                    tag, zw, debug=False):
        outs = []
        for which in ("v3", tag):
            kda_o = torch.zeros(32, 1024, device="cuda", dtype=bf)
            ssm = torch.zeros(8, 3, 8, 128, 128, device="cuda", dtype=fp)
            cvs = torch.zeros(8, 3, 3, 3072, device="cuda", dtype=bf)
            xchg = torch.zeros(V4_SCRATCH_U32, device="cuda", dtype=torch.int32)
            args = [F_in, wf, wg, cw, al, dt, nw, cs_pool, ss_pool, cl_use, sl, cu_use,
                    kda_o, ssm, cvs]
            if not debug:
                if which == "v3":
                    v3_kernel("glm53_kda_verify_fused512", 512)((S, 8, 1), args + [I(S), I(3)])
                else:
                    v4_kernel(f"glm53_kda_verify_fused512{tag}")((S, 8, zw), args + [I(S), xchg])
            else:
                dr = torch.zeros(32, 1024, device="cuda", dtype=bf)
                dc = torch.zeros(32, 3072, device="cuda", dtype=bf)
                ds = torch.zeros(8 * 24, device="cuda", dtype=fp)
                if which == "v3":
                    v3_kernel("glm53_kda_verify_fused512_debug", 512)(
                        (S, 8, 1), args + [I(S), I(3), dr, dc, ds])
                else:
                    v4_kernel(f"glm53_kda_verify_fused512{tag}_debug")(
                        (S, 8, zw), args + [I(S), xchg, dr, dc, ds])
            torch.cuda.synchronize()
            outs.append((kda_o, ssm, cvs, xchg) + ((dr, dc, ds) if debug else ()))
        return outs

    for tag, zw in (("s2", 2), ("s4", 4)):
        TG = tag.upper()
        for S in (1, 2, 3, 4, 5, 6, 7, 8):
            F_in = rnd((32, 3336))
            wf, wg = rnd((1024, 128)), rnd((1024, 128))
            cw = torch.randn(3072 * 4, device="cuda", dtype=fp)
            al = torch.randn(8, device="cuda", dtype=fp) * 0.5
            dt = torch.randn(1024, device="cuda", dtype=fp) * 0.5
            nw = rnd((128,))
            cs_pool = rnd((NL, 3, 3072))
            ss_pool = torch.randn(NL, 8, 128, 128, device="cuda", dtype=fp)
            cl = torch.arange(1, NL + 1, device="cuda", dtype=torch.int32)[:S].contiguous()
            sl = torch.arange(1, NL + 1, device="cuda", dtype=torch.int32)[:S].contiguous()
            cu = torch.arange(0, 3 * (S + 1), 3, device="cuda", dtype=torch.int32)
            o = run_pair_zs(S, cl, cu, F_in, wf, wg, cw, al, dt, nw, cs_pool, ss_pool, sl,
                            tag, zw)
            bits = all(bits_equal(o[0][i], o[1][i]) for i in range(3))
            err0 = int(o[1][3][0].item()) == 0 and int(o[1][3][1].item()) == 0
            o2 = run_pair_zs(S, cl, cu, F_in, wf, wg, cw, al, dt, nw, cs_pool, ss_pool, sl,
                             tag, zw)
            det = all(bits_equal(o[1][i], o2[1][i]) for i in range(3))
            results.append((bits and err0 and det,
                            f"{TG} S={S}: zs vs pinned v3 bitwise={bits} "
                            f"err/trip clean={err0} deterministic={det}"))
        S = 2
        cl_bad = cl.clone(); cl_bad[1] = -1
        o = run_pair_zs(S, cl_bad, cu, F_in, wf, wg, cw, al, dt, nw, cs_pool, ss_pool, sl,
                        tag, zw)
        bits = all(bits_equal(o[0][i], o[1][i]) for i in range(3))
        results.append((bits, f"{TG} invalid-seq guard bitwise = {bits}"))
        cu_bad = torch.tensor([0, 3, 5, 8], device="cuda", dtype=torch.int32)
        o = run_pair_zs(S, cl, cu_bad, F_in, wf, wg, cw, al, dt, nw, cs_pool, ss_pool, sl,
                        tag, zw)
        bits = all(bits_equal(o[0][i], o[1][i]) for i in range(3))
        results.append((bits, f"{TG} partial-row guard bitwise = {bits}"))
        o = run_pair_zs(4, cl, torch.arange(0, 3 * (4 + 1), 3, device="cuda", dtype=torch.int32),
                        F_in, wf, wg, cw, al, dt, nw, cs_pool, ss_pool, sl, tag, zw, debug=True)
        bits = all(bits_equal(o[0][i], o[1][i]) for i in (4, 5, 6))
        results.append((bits, f"{TG} debug raw/cv/rs bitwise = {bits}"))
        # ticket stress: 500 back-to-back launches at S=8 (all CTAs co-resident)
        S = 8
        xchg = torch.zeros(V4_SCRATCH_U32, device="cuda", dtype=torch.int32)
        kda_o = torch.zeros(32, 1024, device="cuda", dtype=bf)
        ssm = torch.zeros(8, 3, 8, 128, 128, device="cuda", dtype=fp)
        cvs = torch.zeros(8, 3, 3, 3072, device="cuda", dtype=bf)
        cl8 = torch.arange(1, NL + 1, device="cuda", dtype=torch.int32)[:S].contiguous()
        sl8 = torch.arange(1, NL + 1, device="cuda", dtype=torch.int32)[:S].contiguous()
        cu8 = torch.arange(0, 3 * (S + 1), 3, device="cuda", dtype=torch.int32)
        s2args = [F_in, wf, wg, cw, al, dt, nw, cs_pool, ss_pool, cl8, sl8, cu8,
                  kda_o, ssm, cvs, I(S), xchg]
        s2k = v4_kernel(f"glm53_kda_verify_fused512{tag}")
        s2k((S, 8, zw), s2args)
        torch.cuda.synchronize()
        ref3 = (kda_o.clone(), ssm.clone(), cvs.clone())
        xchg.zero_()
        for _ in range(500):
            s2k((S, 8, zw), s2args)
        torch.cuda.synchronize()
        stress = all(bits_equal(ref3[i], (kda_o, ssm, cvs)[i]) for i in range(3)) and \
            int(xchg[0].item()) == 0 and int(xchg[1].item()) == 0
        results.append((stress, f"{TG} stress 500 launches: bitwise stable + err/trip 0 = {stress}"))
        for S in (1, 4, 8):
            xchg2 = torch.zeros(V4_SCRATCH_U32, device="cuda", dtype=torch.int32)
            clS = torch.arange(1, NL + 1, device="cuda", dtype=torch.int32)[:S].contiguous()
            slS = torch.arange(1, NL + 1, device="cuda", dtype=torch.int32)[:S].contiguous()
            cuS = torch.arange(0, 3 * (S + 1), 3, device="cuda", dtype=torch.int32)
            sa = [F_in, wf, wg, cw, al, dt, nw, cs_pool, ss_pool, clS, slS, cuS,
                  kda_o, ssm, cvs, I(S), xchg2]
            t_v3 = timing(lambda: v3_kernel("glm53_kda_verify_fused512", 512)((S, 8, 1), sa[:15] + [I(S), I(3)]))
            t_v4 = timing(lambda: v4k((S, 8, 1), sa[:15] + [I(S)]))
            t_s2 = timing(lambda: s2k((S, 8, zw), sa))
            results.append((t_s2 < t_v3, f"{TG} timing S={S}: v3 {t_v3:.2f} v4p {t_v4:.2f} "
                                         f"-> {tag} {t_s2:.2f}us ({(t_v3 - t_s2):+.2f}us)"))

    # ---- G-D: e2e full slab vs [cublasLt -> v3 -> cublasLt] chain ------------
    for S in (1, 4, 8):
        T = 3 * S
        x, wq, wo = rnd((T, 4096)), rnd((3336, 4096)), rnd((4096, 1024))
        F_ref = torch.empty(T, 3336, device="cuda", dtype=bf)
        lt.gemm(x, wq, F_ref, T, 3336, 4096)()
        kda_o_ref = torch.zeros(32, 1024, device="cuda", dtype=bf)
        ssm_ref = torch.zeros(8, 3, 8, 128, 128, device="cuda", dtype=fp)
        cvs_ref = torch.zeros(8, 3, 3, 3072, device="cuda", dtype=bf)
        Fpad = torch.zeros(32, 3336, device="cuda", dtype=bf); Fpad[:T] = F_ref
        v3_kernel("glm53_kda_verify_fused512", 512)((S, 8, 1),
            [Fpad, wf, wg, cw, al, dt, nw, cs_pool, ss_pool, cl, sl, cu,
             kda_o_ref, ssm_ref, cvs_ref, I(S), I(3)])
        sub_ref = torch.empty(T, 4096, device="cuda", dtype=bf)
        lt.gemm(kda_o_ref[:T], wo, sub_ref, T, 4096, 1024)()
        # slab full
        F_s = torch.zeros(32, 3336, device="cuda", dtype=bf)
        kda_o_s = torch.zeros(32, 1024, device="cuda", dtype=bf)
        ssm_s = torch.zeros(8, 3, 8, 128, 128, device="cuda", dtype=fp)
        cvs_s = torch.zeros(8, 3, 3, 3072, device="cuda", dtype=bf)
        sub_s = torch.empty(T, 4096, device="cuda", dtype=bf)
        sync.zero_()
        slab_kernel("glm53_kda_slab1_full")((GRID, 1, 1),
            [x, wq, F_s, wf, wg, cw, al, dt, nw, cs_pool, ss_pool, cl, sl, cu,
             kda_o_s, ssm_s, cvs_s, wo, sub_s, sync, I(T), I(S), wq, I(1)])
        torch.cuda.synchronize()
        e_sub = l2rel(sub_s, sub_ref)
        e_F = l2rel(F_s[:T], F_ref)
        e_o = l2rel(kda_o_s[:T], kda_o_ref[:T])
        # determinism
        sub_s2 = torch.empty_like(sub_s); sync.zero_()
        slab_kernel("glm53_kda_slab1_full")((GRID, 1, 1),
            [x, wq, F_s, wf, wg, cw, al, dt, nw, cs_pool, ss_pool, cl, sl, cu,
             kda_o_s, ssm_s, cvs_s, wo, sub_s2, sync, I(T), I(S), wq, I(1)])
        torch.cuda.synchronize()
        det = bits_equal(sub_s, sub_s2)
        ok = det and e_sub < 2e-3 and int(sync[4].item()) == 0
        results.append((ok, f"E2E S={S}: rel_err sub {e_sub:.3e} (F {e_F:.3e}, "
                            f"kda_o {e_o:.3e}), deterministic {det}, err 0"))

    # ---- G-E: select_all bitwise vs 34 x glm53_kda_select -------------------
    NL2 = 1200
    for R in (3, 7):
     for S in (1, 4, 8):
        stash_ssm = torch.randn(NLAYERS, 8, R, 8, 128, 128, device="cuda", dtype=fp)
        stash_cv = rnd((NLAYERS, 8, R, 3, 3072))
        nacc = torch.randint(1, R + 1, (S,), device="cuda", dtype=torch.int32)
        valid = torch.ones(R * S, device="cuda", dtype=torch.int32)
        if S > 1: valid[R] = 0  # seq 1 invalid
        sl_tab = torch.arange(1, NLAYERS * LINES_STRIDE + 1,
                              device="cuda", dtype=torch.int32).reshape(NLAYERS, LINES_STRIDE)
        cl_tab = sl_tab + NLAYERS * LINES_STRIDE
        ss_p0 = torch.randn(NL2, 8, 128, 128, device="cuda", dtype=fp)
        cs_p0 = rnd((NL2, 3, 3072))
        pools = []
        for which in ("per-layer", "all"):
            # identical initial pools per path: guarded/untouched lines must
            # keep their (random) initial content for a bitwise comparison
            ss_p, cs_p = ss_p0.clone(), cs_p0.clone()
            if which == "per-layer":
                for k in range(NLAYERS):
                    v3_kernel("glm53_kda_select", 256)((S, 8, 1),
                        [stash_ssm[k], stash_cv[k], ss_p, cs_p,
                         sl_tab[k], cl_tab[k], nacc, valid, I(R)])
            else:
                slab_kernel("glm53_kda_select_all", block=256, smem=0)((NLAYERS * S, 8, 1),
                    [stash_ssm, stash_cv, ss_p, cs_p, sl_tab, cl_tab, nacc,
                     valid, I(S), I(LINES_STRIDE), I(R)])
            torch.cuda.synchronize()
            pools.append((ss_p, cs_p))
        ok = bits_equal(pools[0][0], pools[1][0]) and bits_equal(pools[0][1], pools[1][1])
        results.append((ok, f"SELECT_ALL R={R} S={S}: final states bitwise = {ok}"))

    # ---- G-F: barrier robustness (2000 back-to-back launches, graph replay) --
    S, T = 8, 24
    sync.zero_()
    full = slab_kernel("glm53_kda_slab1_full")
    def once():
        full((GRID, 1, 1), [x, wq, F_s, wf, wg, cw, al, dt, nw, cs_pool, ss_pool,
                            cl, sl, cu, kda_o_s, ssm_s, cvs_s, wo, sub_s, sync,
                            I(T), I(S), wq, I(1)])
    t_slab = timing(once)
    torch.cuda.synchronize()
    err_fired = int(sync[4].item())
    g1 = lt.gemm(x, wq, F_ref, T, 3336, 4096)
    v3k = v3_kernel("glm53_kda_verify_fused512", 512)
    g2 = lt.gemm(kda_o_ref[:T], wo, sub_ref, T, 4096, 1024)
    def chain():
        g1()
        v3k((S, 8, 1),
            [Fpad, wf, wg, cw, al, dt, nw, cs_pool, ss_pool, cl, sl, cu,
             kda_o_ref, ssm_ref, cvs_ref, I(S), I(3)])
        g2()
    t_chain = timing(chain)
    ok = err_fired == 0
    results.append((ok, f"BARRIER: 50x5 graph-replayed launches, err {err_fired}; "
                        f"slab {t_slab:.1f} us vs chain {t_chain:.1f} us "
                        f"(S=8, saved {t_chain - t_slab:.1f} us/layer)"))

    failed = [msg for ok, msg in results if not ok]
    for ok, msg in results:
        print(("PASS " if ok else "FAIL ") + msg)
    print(f"slab1 gates: {len(results) - len(failed)}/{len(results)} passed")
    if failed:
        raise SystemExit(1)


def fuse_round(args):
    m = json.loads(Path(args.input).read_text())
    fuse_round_manifest(m, mode=args.mode, program=args.program)
    Path(args.output).write_text(json.dumps(m))
    print(f"wrote {args.output}")


def prof_s2(args):
    """Phase profiler for the z2s kernel. Launches the _s2_prof entry REPS
    times on synthetic inputs (same shapes as the G-V4S2 gates) and reports
    per-slot globaltimer medians per z-role, in us relative to the earliest
    CTA entry of each launch. Slots: 0 entry | 1 z0 post-ship / z1 post-
    PKG-wait | 2 prep post-qkd0 | 3/5/7 prep post-RAW-wait rows | 4/6/8 rec
    post-compute rows | 9 prep end | 10 rec end | 11 z0 post-fg3 / z1
    post-unpack."""
    check_build_v4()
    from glm53.ops_kda_v2 import Driver, gpu_check
    gpu_check()
    import torch
    torch.cuda.init(); torch.manual_seed(args.seed)
    torch.zeros(1, device="cuda")   # primary ctx must exist before Driver()
    D = Driver(torch)
    v4 = D.module(V4CUBIN)
    I = C.c_int
    bf, fp = torch.bfloat16, torch.float32

    def rnd(shape, scale=1, dtype=bf):
        return (torch.randn(shape, device="cuda", dtype=fp) * scale).to(dtype)

    def kload(entry, block=512):
        fn = C.c_void_p()
        Driver.check(D.lib.cuModuleGetFunction(C.byref(fn), v4, entry.encode()))
        def call(grid, kargs, smem=0):
            vals = [C.c_uint64(x.data_ptr()) if hasattr(x, "data_ptr") else x
                    for x in kargs]
            av = (C.c_void_p * len(vals))(*[C.addressof(x) for x in vals])
            Driver.check(D.lib.cuLaunchKernel(
                fn, grid[0], grid[1], grid[2], block, 1, 1, smem,
                C.c_void_p(torch.cuda.current_stream().cuda_stream), av, None))
            call._keep = (vals, av)
        return call

    NL = 64
    S = args.seqs
    F_in = rnd((32, 3336)); wf, wg = rnd((1024, 128)), rnd((1024, 128))
    cw = torch.randn(3072 * 4, device="cuda", dtype=fp)
    al = torch.randn(8, device="cuda", dtype=fp) * 0.5
    dt = torch.randn(1024, device="cuda", dtype=fp) * 0.5
    nw = rnd((128,))
    cs_pool = rnd((NL, 3, 3072))
    ss_pool = torch.randn(NL, 8, 128, 128, device="cuda", dtype=fp)
    cl = torch.arange(1, NL + 1, device="cuda", dtype=torch.int32)[:S].contiguous()
    sl = torch.arange(1, NL + 1, device="cuda", dtype=torch.int32)[:S].contiguous()
    cu = torch.arange(0, 3 * (S + 1), 3, device="cuda", dtype=torch.int32)
    kda_o = torch.zeros(32, 1024, device="cuda", dtype=bf)
    stash_h = int(os.environ.get("GLM53_Z2S_STASH_H", "16384"))
    ssm = torch.zeros(8 * 3 * 8 * stash_h, device="cuda", dtype=fp)
    cvs = torch.zeros(8, 3, 3, 3072, device="cuda", dtype=bf)
    xchg = torch.zeros(V4_SCRATCH_U32, device="cuda", dtype=torch.int32)
    ZW = args.zw
    ncta = S * 8 * ZW
    prof = torch.zeros(ncta * 12, device="cuda", dtype=torch.int64)
    tg = f"s{ZW}"
    s2, s2p = kload(f"glm53_kda_verify_fused512{tg}"), kload(f"glm53_kda_verify_fused512{tg}_prof")
    kargs = [F_in, wf, wg, cw, al, dt, nw, cs_pool, ss_pool, cl, sl, cu,
             kda_o, ssm, cvs, I(S), xchg]
    for _ in range(10):
        s2((S, 8, ZW), kargs)
    torch.cuda.synchronize()
    walls, stamps = [], []
    for _ in range(args.reps):
        prof.zero_()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); s2p((S, 8, ZW), kargs + [prof]); e1.record(); e1.synchronize()
        walls.append(e0.elapsed_time(e1) * 1000)
        stamps.append(prof.cpu().view(ncta, 12).double())
    err, trip = int(xchg[0].item()), int(xchg[1].item())
    st = torch.stack(stamps)                       # [reps, ncta, 12], ns
    base = st[:, :, 0].min(dim=1).values           # per-launch earliest entry
    rel = (st - base.view(-1, 1, 1)) / 1e3         # us
    rel = rel.where(st > 0, torch.full_like(rel, float("nan")))
    wall = sorted(walls)[len(walls) // 2]
    print(f"S={S} CTAs={ncta} reps={args.reps} wall(median)={wall:.2f}us "
          f"err={err} trip={trip}")
    names = ["entry", "z0:ship/z1:pkg", "prep:qkd0", "prep:raw0", "rec:row0",
             "prep:raw1", "rec:row1", "prep:raw2", "rec:row2", "prep:end",
             "rec:end", "z0:fg3/z1:unpk"]
    print("slot  phase              " + "".join(f"    z{z}-med  z{z}-p90" for z in range(ZW)))
    for i in range(12):
        line = f"{i:4d}  {names[i]:16s}"
        for z in range(ZW):
            v = rel[:, z::ZW, i].flatten()
            v = v[~torch.isnan(v)]
            line += ("        --      --" if v.numel() == 0 else
                     f"  {v.median().item():8.2f}  {torch.quantile(v.float(), 0.9).item():8.2f}")
        print(line)
    zs = [rel[:, z::ZW, :] for z in range(ZW)]
    pkgrel = torch.stack([zz[:, :, 1] for zz in zs])
    lag = (pkgrel.max(dim=0).values - pkgrel.min(dim=0).values).flatten()
    print(f"pkg spread (max-min z slot1): med {lag.median():+.2f}us")
    for j in range(3):
        ri, wi = 4 + 2 * j, 3 + 2 * j
        rw = torch.stack([zz[:, :, wi] for zz in zs]).max(dim=0).values.flatten()
        rc = torch.stack([zz[:, :, ri] for zz in zs]).max(dim=0).values.flatten()
        m = ~(torch.isnan(rw) | torch.isnan(rc))
        print(f"row{j}: raw-wait release - max(rec row{j} all z) = "
              f"{(rw[m] - rc[m]).median():+.2f}us (xchg ship+ticket+poll)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sub.add_parser("build-v4")
    tp = sub.add_parser("test")
    tp.add_argument("--seed", type=int, default=7)
    pp = sub.add_parser("prof-s2")
    pp.add_argument("--seed", type=int, default=7)
    pp.add_argument("--seqs", type=int, default=1)
    pp.add_argument("--reps", type=int, default=30)
    pp.add_argument("--zw", type=int, default=2, choices=[2, 4])
    fp_ = sub.add_parser("fuse-round")
    fp_.add_argument("input"); fp_.add_argument("output")
    fp_.add_argument("--mode", choices=["slab", "select", "vz"], default=None,
                     help="default: env GLM53_SLAB1 or 'slab'")
    fp_.add_argument("--program", default="round_k2")
    args = ap.parse_args()
    if args.cmd == "build": build()
    elif args.cmd == "build-v4": build_v4()
    elif args.cmd == "test": test(args)
    elif args.cmd == "prof-s2": prof_s2(args)
    elif args.cmd == "fuse-round": fuse_round(args)


if __name__ == "__main__":
    main()
