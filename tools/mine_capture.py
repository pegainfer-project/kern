#!/usr/bin/env python3
"""launches.jsonl -> manifest 生成器的分析前端（三步）。

1. 切 pass：>gap-ms 时间空隙切出请求/独立 dummy 窗口；窗口内按 launch
   符号序列的周期锚点切 forward pass（时间只够切请求级，见 README）。
2. 跨 pass 指针稳定性分类：同一 call site（symbol + 窗口内出现序号，
   enforce_eager 下 launch 序列确定，出现序号即 call site）的指针身份
   (range_start, offset) 跨 forward 对比 —— 恒定且逐层不同 = weight 候选，
   恒定且全层相同 = 持久 state/input 候选，随 pass 变 = workspace。
3. 表达式拟合：grid 各轴与标量参数在多个 token 数的 forward 上采样，
   对封闭表达式集合（const / sym / mul / ceil_div / ceil_div∘mul /
   mul∘ceil_div）做一致性筛选。

输出人读报告到 stdout；--json 落全量结果。不产出 manifest —— 产出的是
写 manifest 生成器所需的全部事实。
"""

import argparse
import collections
import json
import math
import struct
import sys

GAP_MS_DEFAULT = 5.0
MIN_FORWARD_WINDOW = 50  # 小于这个 launch 数的窗口是 init/JIT 杂音，不算 forward
CEIL_DIV_CANDS = [2 ** i for i in range(1, 17)]


def load(path):
    recs = []
    with open(path) as f:
        for line in f:
            recs.append(json.loads(line))
    if not recs or "t_ns" not in recs[0]:
        sys.exit("input has no t_ns timestamps; re-capture with the current capture.c")
    return recs


# ---------------------------------------------------------------- step 1: pass 切分

def slice_windows(recs, gap_ms):
    windows, cur = [], [recs[0]]
    for prev, r in zip(recs, recs[1:]):
        if r["t_ns"] - prev["t_ns"] > gap_ms * 1e6:
            windows.append(cur)
            cur = []
        cur.append(r)
    windows.append(cur)
    return windows


BURST_GAP = 25  # core 核相邻出现的 launch 序号间隔超过它即视为跨 forward


def slice_forwards(windows):
    """窗口内按 core 核的 burst 切 forward。

    core = 全局最高频核（必然逐层跑，每 forward 以一个密集 burst 出现，
    burst 内相邻出现相隔 ~10 个 launch）。burst 之间是 forward 尾部
    （final norm / logits / 采样）+ 下一个 forward 头部（embedding 等），
    从空档中点切开。边界带里的采样/簿记核可能被划错半格 —— 它们不是
    要收编的核，对齐失败会单独报告，不污染层内核的分析。
    """
    total = collections.Counter(r["symbol"] for w in windows for r in w)
    core = total.most_common(1)[0][0]
    big = [w for w in windows
           if len(w) >= MIN_FORWARD_WINDOW and any(r["symbol"] == core for r in w)]
    if not big:
        sys.exit("no window has >= %d launches" % MIN_FORWARD_WINDOW)

    forwards = []  # list of (window_idx, recs)
    n_per_window = []
    for wi, w in enumerate(big):
        pos = [i for i, r in enumerate(w) if r["symbol"] == core]
        bursts = [[pos[0]]]
        for a, b in zip(pos, pos[1:]):
            if b - a > BURST_GAP:
                bursts.append([])
            bursts[-1].append(b)
        cuts = [(bursts[k][-1] + bursts[k + 1][0]) // 2 for k in range(len(bursts) - 1)]
        starts = [0] + cuts
        ends = cuts + [len(w)]
        for s, e in zip(starts, ends):
            forwards.append((wi, w[s:e]))
        n_per_window.append(len(bursts))
    return core, n_per_window, forwards


def pick_tokens_reference(forwards):
    """tokens 参照 = 某个每 forward 恰出现一次、grid.x 跨 forward 变化的
    核的 grid.x。选哪个不能靠先验（ceil_div 形的核尺度更小但不可逆，会
    让其他核大面积拟合不出）—— 所以自验证：对每个候选试拟合全部 call
    site 的 grid.x，取失败数最少者；平手取覆盖 forward 多、尺度小的。"""
    per_sym = collections.defaultdict(list)
    for fi, (_, f) in enumerate(forwards):
        seen = collections.Counter()
        gx = {}
        for r in f:
            seen[r["symbol"]] += 1
            gx[r["symbol"]] = r["grid"][0]
        for s in seen:
            if seen[s] == 1:
                per_sym[s].append((fi, gx[s]))
    cands = [(s, dict(vals)) for s, vals in per_sym.items()
             if len({g for _, g in vals}) > 1]
    if not cands:
        sys.exit("no per-forward-once symbol with varying grid.x; cannot infer tokens")
    cands.sort(key=lambda sv: -len(sv[1]))

    best = None
    for s, tok_of in cands[:8]:
        kept = [f for fi, f in enumerate(forwards) if fi in tok_of]
        tokens = [g for _, g in sorted(tok_of.items())]
        analysis, _ = analyze_sites(kept, tokens)
        # 拟合不出和欠定都算失败：不可逆（ceil_div 形）的参照造成前者，
        # 分辨率差的参照（采样点坍缩）造成后者
        failures = sum(
            1
            for a in analysis.values()
            for site in a["sites"]
            if site["grid"][0] is None
            or site["grid"][0].get("form") == "underdetermined"
        )
        key = (failures, -len(set(tokens)), len(forwards) - len(kept), max(tokens))
        if best is None or key < best[0]:
            best = (key, s, tok_of, kept, tokens)
    _, ref, tok_of, kept, tokens = best
    return ref, tokens, kept, len(forwards) - len(kept)


# ---------------------------------------------------------------- step 3: 表达式拟合
# （step 2 的指针分类要用同样的 call-site 索引，实现顺序上放在一起）

def _interval_ceil_div(t, g):
    """{c : ceil(t/c) == g} 的整数区间，None 表示空。"""
    if g == 1:
        return (t, None)  # c >= t，无上界
    lo = math.ceil(t / g)
    hi = (t - 1) // (g - 1)
    return (lo, hi) if lo <= hi else None


def _intersect(ivs):
    lo, hi = 1, None
    for iv in ivs:
        if iv is None:
            return None
        a, b = iv
        lo = max(lo, a)
        if b is not None:
            hi = b if hi is None else min(hi, b)
    if hi is not None and lo > hi:
        return None
    return (lo, hi)


def _canon(lo, hi):
    """区间里挑个规范常数：优先 2 的幂。"""
    p = 1
    while p < lo:
        p *= 2
    if hi is None or p <= hi:
        return p
    return lo


def fit(samples):
    """samples: [(tokens, value)] -> 表达式描述 dict，或 None（拟合不出）。"""
    ts = [t for t, _ in samples]
    vs = [v for _, v in samples]
    if len(set(vs)) == 1:
        return {"form": "const", "c": vs[0]}
    if len(set(ts)) < 3:
        return {"form": "underdetermined", "distinct_tokens": len(set(ts))}
    if all(v == t for t, v in samples):
        return {"form": "sym"}
    if any(t < 1 for t in ts) or any(v is None or v < 1 for v in vs):
        return None  # 0/负值/缺失：不属于任何几何表达式形态
    muls = {v // t for t, v in samples if t and v % t == 0}
    if len(muls) == 1 and all(t and v % t == 0 for t, v in samples):
        return {"form": "mul", "c": muls.pop()}
    iv = _intersect([_interval_ceil_div(t, v) for t, v in samples])
    if iv:
        return {"form": "ceil_div", "c": _canon(*iv), "c_range": iv}
    # ceil_div(mul(sym, m), c)
    for c in CEIL_DIV_CANDS:
        mivs = []
        for t, v in samples:
            lo = (v - 1) * c // t + 1
            hi = v * c // t
            mivs.append((lo, hi) if lo <= hi else None)
        iv = _intersect(mivs)
        if iv:
            return {"form": "ceil_div_mul", "m": _canon(*iv), "m_range": iv, "c": c}
    # mul(ceil_div(sym, c), m)
    for c in CEIL_DIV_CANDS:
        ms = set()
        ok = True
        for t, v in samples:
            q = -(-t // c)
            if v % q:
                ok = False
                break
            ms.add(v // q)
        if ok and len(ms) == 1:
            return {"form": "mul_ceil_div", "c": c, "m": ms.pop()}
    return None


def le_int(hexstr):
    return int.from_bytes(bytes.fromhex(hexstr), "little") if hexstr else None


def analyze_sites(forwards, tokens_per_forward):
    """按 (symbol, 窗口内出现序号) 对齐 call site，跨 forward 收集
    grid / 标量 / 指针样本。出现次数跨 forward 不一致的符号对齐不了，
    单独报告。"""
    # sym -> forward idx -> [recs in order]
    occ = collections.defaultdict(lambda: collections.defaultdict(list))
    for fi, (_, f) in enumerate(forwards):
        for r in f:
            occ[r["symbol"]][fi].append(r)

    # 按众数出现次数对齐：切分边界有 ±1 launch 抖动、prefill/decode 序列
    # 也不同，某符号次数非众数的 forward 只对该符号剔除，不整体作废。
    aligned, unaligned = {}, {}
    for sym, per_fw in occ.items():
        modal, _ = collections.Counter(len(v) for v in per_fw.values()).most_common(1)[0]
        kept = {fi: v for fi, v in per_fw.items() if len(v) == modal}
        if len(kept) < 2:
            unaligned[sym] = {
                "forwards_present": len(per_fw),
                "occurrence_counts": sorted({len(v) for v in per_fw.values()}),
            }
            continue
        aligned[sym] = kept

    out = {}
    for sym, per_fw in aligned.items():
        fis = sorted(per_fw)
        n_occ = len(per_fw[fis[0]])
        sites = []
        for oi in range(n_occ):
            recs = [(tokens_per_forward[fi], per_fw[fi][oi]) for fi in fis]
            site = {"grid": [fit([(t, r["grid"][ax]) for t, r in recs]) for ax in range(3)]}
            p0 = recs[0][1].get("params")
            if isinstance(p0, list):
                # Param staging can differ across forwards at one call site
                # (e.g. a launch that staged no bytes once); classify only
                # the prefix every sample has.
                n_params = min(len(r.get("params") or []) for _, r in recs)
                n_params = min(n_params, len(p0))
                site["params"] = [
                    classify_param(pi, [(t, r["params"][pi]) for t, r in recs])
                    for pi in range(n_params)
                ]
            else:
                site["params"] = p0  # null 或 "unknown-layout"，原样透传
            sites.append(site)
        # 相邻 site 往往逐层重复；相同的压成一组报告
        out[sym] = {"n_sites": n_occ, "coverage": len(fis),
                    "tokens_sampled": sorted({tokens_per_forward[fi] for fi in fis}),
                    "sites": sites}
    return out, unaligned


def classify_param(pi, samples):
    """samples: [(tokens, param-json)]。指针走稳定性分类，标量走拟合。"""
    ptr = [(t, p) for t, p in samples if "pointer" in p]
    if len(ptr) == len(samples):
        idents = set()
        for _, p in samples:
            rs = int(p["pointer"]["range_start"], 16)
            idents.add((rs, le_int(p["data"]) - rs, p["pointer"]["range_size"]))
        if len(idents) == 1:
            rs, off, size = idents.pop()
            return {
                "kind": "ptr_stable",
                "range_start": hex(rs),
                "offset": off,
                "range_size": size,
                "memory_type": samples[0][1]["pointer"]["memory_type"],
            }
        return {"kind": "ptr_varying", "n_identities": len(idents),
                "ranges": sorted({hex(i[0]) for i in idents})}
    if ptr:
        return {"kind": "ptr_sometimes", "with_pointer": len(ptr), "of": len(samples)}
    size = samples[0][1]["size"]
    vals = [(t, le_int(p["data"])) for t, p in samples]
    host_like = any(v is not None and (v >> 40) == 0xFFFF for _, v in vals)
    r = {"kind": "scalar", "size": size, "fit": fit(vals)}
    if host_like:
        r["kind"] = "host_ptr_suspect"  # 0xffff... 开头：未归类 host 指针盲点
    if size == 4 and r.get("fit", {}) and r["fit"].get("form") == "const":
        r["as_f32"] = struct.unpack("<f", struct.pack("<I", r["fit"]["c"]))[0]
    return r


# ---------------------------------------------------------------- 报告

def summarize_pointer(site_params):
    """一个 symbol 的各 site 同一参数位的指针分类汇总。"""
    kinds = collections.Counter()
    idents = set()
    for p in site_params:
        kinds[p["kind"]] += 1
        if p["kind"] == "ptr_stable":
            idents.add((p["range_start"], p["offset"]))
    if set(kinds) == {"ptr_stable"}:
        if len(idents) == len(site_params) > 1:
            return "weight-like（逐 site 恒定且互不相同）"
        if len(idents) == 1:
            return "global persistent（全 site 同一指针，state/input 候选）"
        return "persistent（%d 个身份 / %d site）" % (len(idents), len(site_params))
    if set(kinds) == {"ptr_varying"}:
        return "workspace（跨 forward 变化）"
    return "混合: " + ", ".join("%s×%d" % kv for kv in kinds.items())


def expr_str(e):
    if e is None:
        return "!! 拟合不出（超出封闭表达式集合）"
    f = e["form"]
    if f == "const":
        return str(e["c"])
    if f == "sym":
        return "tokens"
    if f == "mul":
        return "tokens*%d" % e["c"]
    if f == "ceil_div":
        r = e["c_range"]
        return "ceil_div(tokens,%d)[c∈%s..%s]" % (e["c"], r[0], r[1] if r[1] else "∞")
    if f == "ceil_div_mul":
        return "ceil_div(tokens*%d,%d)" % (e["m"], e["c"])
    if f == "mul_ceil_div":
        return "ceil_div(tokens,%d)*%d" % (e["c"], e["m"])
    if f == "underdetermined":
        return "欠定（仅 %d 个 token 采样点）" % e["distinct_tokens"]
    return json.dumps(e)


def report(core, n_per_window, forwards, tokens, ref, dropped, analysis,
           unaligned, only_vllm):
    print("== pass 切分 ==")
    print("core 核: %s" % core[:90])
    print("各窗口 forward 数: %s -> 共 %d 个 forward（另剔除 %d 个无 tokens 参照的）"
          % (n_per_window, len(forwards), dropped))
    print("tokens 参照核: %s" % ref[:90])
    print("各 forward tokens: %s" % tokens)
    print()

    syms = sorted(analysis)
    if only_vllm:
        syms = [s for s in syms if "vllm" in s]
    for sym in syms:
        a = analysis[sym]
        print("== %s ==" % sym[:90])
        print("  %d call sites, 覆盖 %d/%d forward, tokens 采样点 %s" % (
            a["n_sites"], a["coverage"], len(forwards), a["tokens_sampled"]))
        grids = collections.Counter(
            tuple(expr_str(g) for g in s["grid"]) for s in a["sites"]
        )
        for g, n in grids.most_common():
            print("  grid [%s] × %d site" % (", ".join(g), n))
        params0 = a["sites"][0]["params"]
        if not isinstance(params0, list):
            print("  params: %s" % params0)
            print()
            continue
        for pi in range(len(params0)):
            col = [s["params"][pi] for s in a["sites"]]
            kinds = {p["kind"] for p in col}
            if kinds <= {"ptr_stable", "ptr_varying", "ptr_sometimes"}:
                print("  param[%d] ptr: %s" % (pi, summarize_pointer(col)))
            elif kinds & {"ptr_stable", "ptr_varying", "ptr_sometimes"}:
                # 同一参数位有的 site 是指针有的是标量（如可选指针传 NULL）
                print("  param[%d] 混合: %s" % (
                    pi, ", ".join("%s×%d" % kv for kv in
                                  collections.Counter(p["kind"] for p in col).items())))
            else:
                fits = collections.Counter(expr_str(p.get("fit")) for p in col)
                extra = " !! host 指针嫌疑" if "host_ptr_suspect" in kinds else ""
                print("  param[%d] scalar(%dB): %s%s" % (
                    pi, col[0].get("size", 0),
                    "; ".join("%s ×%d" % kv for kv in fits.most_common(3)), extra))
        print()

    if unaligned:
        print("== 无法按出现序号对齐的符号（次数跨 forward 不一致，已跳过）==")
        for s, info in sorted(unaligned.items()):
            print("  %s  出现于 %d/%d forward, 次数 %s" % (
                s[:80], info["forwards_present"], len(forwards),
                info["occurrence_counts"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--gap-ms", type=float, default=GAP_MS_DEFAULT)
    ap.add_argument("--json", help="全量分析结果落到这个文件")
    ap.add_argument("--all-symbols", action="store_true",
                    help="报告所有符号（默认只报 vllm flat 核）")
    args = ap.parse_args()

    recs = load(args.jsonl)
    windows = slice_windows(recs, args.gap_ms)
    core, n_per_window, forwards = slice_forwards(windows)
    ref, tokens, forwards, dropped = pick_tokens_reference(forwards)
    analysis, unaligned = analyze_sites(forwards, tokens)
    report(core, n_per_window, forwards, tokens, ref, dropped, analysis,
           unaligned, only_vllm=not args.all_symbols)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"core": core, "tokens": tokens, "ref": ref,
                       "symbols": analysis, "unaligned": unaligned}, f, indent=1)
        print("\nfull analysis -> %s" % args.json)


if __name__ == "__main__":
    main()
