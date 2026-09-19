#!/usr/bin/env python3
"""Put a family of TRT-LLM gen batched GEMM cubins from a flashinfer-cubin
bundle into the registry: the cubins into the cache (and, with --upload,
the blob store), one variant per kernel with its launch geometry and the
options its `flashinferMetaInfo.h` entry records, the bundle's ABI headers
as one blob the family's `abi_source` names.

    tools/kernels/import_flashinfer.py --bundle <dir> --family trtllm_bmm_mxe4m3_mxe2m1_mxe4m3 \\
        --match 'bmm_MxE4m3_MxE2m1MxE4m3_.*_siTuGlu_.*_sm100f$' --license-file <apache-2.0.txt> [--upload]

`<dir>` is the bundle (`flashinfer_cubin/cubins/<hash>/batched_gemm-*/`):
`include/flashinferMetaInfo.h`, `include/trtllmGen_bmm_export/`,
`checksums.txt` and the cubins. A variant is named by its kernel's entry;
its tags are the launcher-relevant options (tools/kernels/abi/trtllm_bmm.py
reads them), enums by name. The MetaInfo hash is checked against the bytes.
"""
import argparse
import datetime
import pathlib
import re
import subprocess
import sys
import tarfile
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import import_cubin  # noqa: E402
import index  # noqa: E402
import store  # noqa: E402

ENTRY = re.compile(r'\{nullptr, 0, (\d+), "([^"]+)", (\d+), "([0-9a-f]{64})",.*?\{(.*?)\}, gemm::SmVersion::(\w+)\}',
                   re.S)
OPTION = re.compile(r"/\* (m\w+) \*/ ([^,\n]+)")
DTYPE = ["bf16", "bool", "e2m1", "e2m3", "e3m2", "e4m3", "e5m2", "f16", "f32", "i8", "i32", "i64", "mxe2m1", "mxe4m3",
         "mxint4", "ue8m0", "u8", "u16", "u32", "u64", "u128", "void"]  # trtllm::gen::Dtype by uid (DtypeDecl.h)
ENUMS = {
    "SfLayout": ["linear", "r8c4", "r8c16", "r128c4"],
    "RouteImpl": ["none", "ldgsts", "tma", "ldg_sts"],
    "TileScheduler": ["static", "persistent", "static_persistent", "persistent_sm90"],
    "MatrixLayout": ["major_k", "major_mn", "block_major_k"],
    "BatchMode": ["batch_m", "batch_n"],
    "ActType": ["swiGlu", "geGlu", "siTuGlu", "none"],
    "BiasType": ["none", "m", "n", "mn"],
    "MmaKind": ["auto", "fp16", "fp8fp6fp4", "int8", "tf32", "mxfp8fp6fp4", "mxfp4nvfp4"],
}
# option -> tag; a tuple of options folds into one list tag
TAGS = {
    "tile": ("mTileM", "mTileN", "mTileK"), "epilogue_tile": ("mEpilogueTileM", "mEpilogueTileN"),
    "mma": ("mMmaM", "mMmaN", "mMmaK"), "mma_tile_k": "mMmaTileK", "cluster": ("mClusterDimX", "mClusterDimY", "mClusterDimZ"),
    "stages": ("mNumStagesA", "mNumStagesB"), "mma_kind": "mMmaKind", "dtype_a": "mDtypeA", "dtype_b": "mDtypeB",
    "dtype_c": "mDtypeC", "dtype_sf_c": "mDtypeSfC", "sf_block": ("mSfBlockSizeA", "mSfBlockSizeB", "mSfBlockSizeC"),
    "sf_layout": ("mSfLayoutA", "mSfLayoutB", "mSfLayoutC"), "sf_reshape": "mSfReshapeFactor", "layout": ("mLayoutA", "mLayoutB"),
    "block_k": "mBlockK", "batch_mode": "mBatchMode", "static_batch": "mIsStaticBatch", "route": "mRouteImpl",
    "route_sf": "mRouteSfsImpl", "sched": "mTileScheduler", "split_k": "mNumSlicesForSplitK", "transpose_out": "mTransposeMmaOutput",
    "fused_act": "mFusedAct", "act": "mActType", "clamp_before_act": "mClampBeforeAct", "bias": "mBiasType",
    "tma_store": "mUseTmaStore", "tma_oob": "mUseTmaOobOpt", "c_multicast": "mUseCMultiCast", "early_exit": "mEnablesEarlyExit",
    "delayed_early_exit": "mEnablesDelayedEarlyExit", "shuffled_matrix": "mUseShuffledMatrix", "deepseek_fp8": "mUseDeepSeekFp8",
    "per_token_sf": ("mUsePerTokenSfA", "mUsePerTokenSfB"),
}


def value(v):
    """One MetaInfo option value: an int, a bool, an enum's name, or a dtype's."""
    v = v.strip().strip("{}").strip()
    if not v:
        return None
    if m := re.fullmatch(r"trtllm::gen::Dtype\((\d+)\)", v):
        return DTYPE[int(m.group(1)) & 0xFF]
    if m := re.fullmatch(r"[\w:]*::(\w+)\((\d+)\)", v):
        names = ENUMS.get(m.group(1))
        return names[int(m.group(2))] if names else int(m.group(2))
    if v in ("true", "false"):
        return v == "true"
    return int(v)


def meta_info(bundle):
    """kernel entry -> (shared_mem, threads, sha256, {option: value}, sm) for every MetaInfo entry."""
    text = (bundle / "include" / "flashinferMetaInfo.h").read_text()
    out = {}
    for smem, name, threads, sha, opts, sm in ENTRY.findall(text):
        out[name] = (int(smem), int(threads), sha, {k: x for k, v in OPTION.findall(opts) if (x := value(v)) is not None}, sm)
    return out


def tags(opts):
    t = {}
    for tag, keys in TAGS.items():
        if isinstance(keys, tuple):
            t[tag] = [opts[k] for k in keys]
        else:
            t[tag] = opts[keys]
    return t


def headers_blob(bundle):
    """The bundle's ABI headers as one tar in the store (uncompressed, zero mtimes: the same bytes every import)."""
    with tempfile.TemporaryDirectory() as d:
        tgz = pathlib.Path(d) / "trtllmGen_bmm_export.tar"
        with tarfile.open(tgz, "w", format=tarfile.PAX_FORMAT) as tf:
            for p in sorted((bundle / "include" / "trtllmGen_bmm_export").rglob("*")):
                if p.is_file():
                    info = tf.gettarinfo(p, arcname=str(p.relative_to(bundle / "include")))
                    info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
                    with open(p, "rb") as f:
                        tf.addfile(info, f)
        return store.put(tgz)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True, type=pathlib.Path)
    ap.add_argument("--family", required=True)
    ap.add_argument("--match", required=True, help="regex over the kernel entry name")
    ap.add_argument("--upstream", required=True, help="e.g. 'flashinfer-cubin 0.6.18, cubins/<hash>/batched_gemm-<v>'")
    ap.add_argument("--license", default="Apache-2.0")
    ap.add_argument("--license-file", type=pathlib.Path)
    ap.add_argument("--abi-capture")
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args()
    meta = meta_info(a.bundle)
    picked = sorted(n for n in meta if re.search(a.match, n))
    if not picked:
        sys.exit(f"no MetaInfo entry matches {a.match!r}")
    version = re.search(r'TLLM_GEN_EXPORT_VERSION "([^"]+)"', (a.bundle / "include" / "flashinferMetaInfo.h").read_text())
    hdr = headers_blob(a.bundle)
    args = argparse.Namespace(
        kind="cubin", sm=None, abi="trtllm_bmm", license=a.license, license_file=a.license_file, toolchain=None,
        rebuild=None, abi_capture=a.abi_capture,
        upstream=f"{a.upstream}, TLLM_GEN_EXPORT_VERSION {version.group(1)}",
        abi_source=f"blob {hdr} (trtllmGen_bmm_export/): KernelParamsDecl.h is the pack, BatchedGemmInterface.h "
                   "setKernelParams fills it, TmaDescriptor.h encodes the maps; tools/kernels/abi/trtllm_bmm.py is the port")
    shas, sms = [hdr], set()
    for name in picked:
        smem, threads, sha, opts, sm = meta[name]
        cubin = a.bundle / f"B{name[1:]}.cubin"
        if not cubin.exists():
            sys.exit(f"{name}: no {cubin.name} in the bundle")
        if store.sha256_of(cubin) != sha:
            sys.exit(f"{name}: MetaInfo hash {sha[:12]} is not the cubin's")
        sms.add(sm.lower())
        args.sm = sm.lower()
        launch = {"block": threads, "shared_mem": smem, "cluster": [opts["mClusterDimX"], opts["mClusterDimY"], opts["mClusterDimZ"]]}
        got, _ = import_cubin.import_cubin(a.family, cubin, name=name, launch=launch, tags=tags(opts), args=args)
        shas.append(got)
    doc = index.load(a.family)
    doc["family"]["sm"] = "+".join(sorted(sms))
    doc["family"]["imported"] = datetime.date.today().isoformat()
    index.save(doc)
    print(f"{a.family}: {len(picked)} variants, headers @{hdr[:12]}", file=sys.stderr)
    if a.upload:
        store.upload(shas, f"{a.family}: {len(picked)} trtllm-gen variants, {args.upstream}")


if __name__ == "__main__":
    main()
