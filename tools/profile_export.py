#!/usr/bin/env python3
"""Turn kern bench evidence into a portable explorer and AI quick view.

Composition calibration withholds whole-program times for holdout scenarios.
Their op microbenchmarks ARE used: this tests composition, not unseen-shape
interpolation. Report that distinction along with every error and raw tail.
"""
import argparse
import gzip
import hashlib
import json
import math
import pathlib
import statistics


def median(xs):
    return statistics.median(xs)


def attach_control(raw, control):
    assert control.get("program_only") and control.get("calibration_after"), "incomplete program-only control"
    for key in ("model", "manifest_sha256", "workload", "hardware"):
        assert raw[key] == control[key], f"control differs in {key}"
    others = {r["scenario"]["id"]: r for r in control["scenarios"]}
    assert len(others) == len(raw["scenarios"]), "control scenario count differs"
    deltas = []
    for r in raw["scenarios"]:
        other = others[r["scenario"]["id"]]
        assert r["outputs"] == other["outputs"], "untraced repeat token outputs differ"
        r["untraced_graph"] = other["graph"]
        r["trace_vs_untraced_pct"] = (r["graph"]["stats"]["p50"] / other["graph"]["stats"]["p50"] - 1) * 100
        deltas.append(abs(r["trace_vs_untraced_pct"]))
    raw["repeat_check"] = dict(scenarios=len(deltas), median_abs_delta_pct=median(deltas),
        max_abs_delta_pct=max(deltas), token_outputs_match=True, runner_sha256=control["runner_sha256"],
        interpretation="Separate untraced program-only run with the same protocol; difference is not pure tracer overhead")


def aggregate(record):
    ops = {}
    for call in record["calls"]:
        case = record["cases"][call["case"]]
        op = ops.setdefault(call["op"], dict(name=call["op"], count=0, launches=0,
            cold_us=0., warm_us=0., attributed_us=0., case_indices=set(),
            max_tail_ratio=1., max_cv=0., calls=[]))
        op["count"] += 1
        op["launches"] += len(call["launches"])
        op["case_indices"].add(call["case"])
        op["calls"].append(call["index"])
        op["cold_us"] += case["cold"]["stats"]["p50"]
        op["warm_us"] += case["warm"]["stats"]["p50"]
        op["attributed_us"] += call["in_program"]["stats"]["p50"]
        op["max_tail_ratio"] = max(op["max_tail_ratio"], *(case[x]["stats"]["tail_ratio"] for x in ("cold", "warm")))
        op["max_cv"] = max(op["max_cv"], *(case[x]["stats"]["cv"] for x in ("cold", "warm")))
        op["max_cv"] = max(op["max_cv"],call["in_program"]["stats"]["cv"])
        op["max_tail_ratio"] = max(op["max_tail_ratio"],call["in_program"]["stats"]["tail_ratio"])
    total = sum(o["attributed_us"] for o in ops.values())
    for op in ops.values():
        op["share"] = op["attributed_us"] / total if total else 0.
        op["estimated_program_us"] = op["share"] * record["graph"]["stats"]["p50"]
        op["case_indices"] = sorted(op["case_indices"])
        op["variable"] = op["max_tail_ratio"] > 1.15 or op["max_cv"] > .10
        op["saving_at_2x_us"] = op["estimated_program_us"] / 2
    return sorted(ops.values(), key=lambda x: -x["attributed_us"])


def export(raw):
    assert not raw.get("program_only"), "whole-program-only reports have no operator evidence"
    assert raw.get("calibration_after"), "incomplete sweep: final hardware calibration missing"
    assert raw.get("trace_validation",{}).get("all_program_activity_sequences_matched"), "attach and validate GPU activity timing first"
    records = raw["scenarios"]
    assert len(records) == len(raw["workload"]["scenarios"]), "missing scenarios"
    for r in records:
        assert r.get("output_check"), "trajectory output validation missing"
        assert all(0 <= c["case"] < len(r["cases"]) for c in r["calls"])
        for c in r["cases"]:
            for mode in ("cold", "warm"):
                assert len(c[mode]["samples_us"]) == raw["workload"]["samples"]
                assert all(math.isfinite(x) and x > 0 for x in c[mode]["samples_us"])
        r["ops"] = aggregate(r)
        r["cold_sum_us"] = sum(o["cold_us"] for o in r["ops"])
        r["warm_sum_us"] = sum(o["warm_us"] for o in r["ops"])
        r["instrumentation_ratio"] = r["instrumented"]["stats"]["p50"] / r["graph"]["stats"]["p50"]
    # Fit a mixture of measured cold and warm op costs with one scale per
    # actual program. No target holdout graph value is read in this fit.
    fits = {}
    for program in sorted({r["program"] for r in records}):
        training = [r for r in records if r["program"] == program and not r["scenario"]["holdout"]]
        if not training:
            continue
        best = None
        for i in range(101):
            alpha = i / 100
            costs = [alpha*r["cold_sum_us"] + (1-alpha)*r["warm_sum_us"] for r in training]
            scale = median([r["graph"]["stats"]["p50"]/x for r, x in zip(training, costs)])
            errors = [abs(math.log(scale*x/r["graph"]["stats"]["p50"])) for r, x in zip(training,costs)]
            loss = sum(errors)/len(errors)
            if best is None or loss < best[0]:
                best = loss, alpha, scale
        _, alpha, scale = best
        residuals = [r["graph"]["stats"]["p50"] / (scale*(alpha*r["cold_sum_us"]+(1-alpha)*r["warm_sum_us"])) for r in training]
        fits[program] = dict(cold_weight=alpha, scale=scale, training_ids=[r["scenario"]["id"] for r in training],
            residual_min=min(residuals), residual_max=max(residuals))
    for r in records:
        fit = fits.get(r["program"])
        if fit:
            predicted = fit["scale"]*(fit["cold_weight"]*r["cold_sum_us"]+(1-fit["cold_weight"])*r["warm_sum_us"])
            r["prediction"] = dict(us=predicted, error_pct=(predicted/r["graph"]["stats"]["p50"]-1)*100,
                range_us=[predicted*fit["residual_min"],predicted*fit["residual_max"]])
    before = {a["name"]:a for a in raw["calibration_before"]}
    after = {a["name"]:a for a in raw["calibration_after"]}
    raw["calibration_drift_pct"] = {name:(after[name]["timing"]["stats"]["p50"]/a["timing"]["stats"]["p50"]-1)*100 for name,a in before.items()}
    raw["eviction_ratio"] = before["evicted_read"]["timing"]["stats"]["p50"]/before["l2_read"]["timing"]["stats"]["p50"]
    raw["composition"] = dict(method="nonnegative cold/warm mixture, median scale per program",
        validation="held-out whole-program times; local op times remain measured",
        range="training residual envelope; not a request-latency percentile or confidence interval", fits=fits)
    raw["coverage"] = dict(programs=len({r["program"] for r in records}),scenarios=len(records),
        call_observations=sum(len(r["calls"]) for r in records),cases=sum(len(r["cases"]) for r in records),
        distinct_ops=len({c["op"] for r in records for c in r["calls"]}),
        holdouts=sum(r["scenario"]["holdout"] for r in records))
    return raw


def quick_view(raw):
    return dict(model=raw["model"],manifest_sha256=raw["manifest_sha256"], hardware=raw["hardware"],coverage=raw["coverage"],
        calibration_drift_pct=raw["calibration_drift_pct"],
        repeat_check=raw.get("repeat_check"),
        caveats=["Shares come from uninstrumented program GPU activity durations and are scaled to measured graph time.",
            "A 2x saving is a what-if, not evidence that an implementation can attain it.",
            "Variability flags retain tails; they do not establish whether the cause is intrinsic or environmental.",
            "Token output comparison checks profiling restoration, not independent model accuracy.",
            raw["composition"]["validation"]],
        scenarios=[dict(id=r["scenario"]["id"], workload=r["scenario"],program=r["program"],
            graph_us=r["graph"]["stats"],prediction=r.get("prediction"),top_ops=r["ops"][:8],
            variation_watchlist=[dict(name=o["name"],share=o["share"],max_cv=o["max_cv"],
                max_tail_ratio=o["max_tail_ratio"],case_indices=o["case_indices"])
                for o in sorted(r["ops"],key=lambda x:-x["max_cv"]) if o["variable"]]) for r in raw["scenarios"]])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs",nargs="+",type=pathlib.Path)
    p.add_argument("--controls",nargs="*",default=[],type=pathlib.Path,help="optional untraced program-only repeats, matched by model")
    p.add_argument("--out",required=True,type=pathlib.Path)
    args=p.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    index=[];controls={}
    for path in args.controls:
        source=path.read_bytes();control=json.loads(source)
        assert control["model"] not in controls,"duplicate control model"
        controls[control["model"]]=(control,hashlib.sha256(source).hexdigest())
    for path in args.inputs:
        source=path.read_bytes(); raw=export(json.loads(source))
        if raw["model"] in controls:
            control,control_hash=controls.pop(raw["model"])
            attach_control(raw,control)
            raw["repeat_check"]["source_sha256"]=control_hash
        # Model labels are provider data: never turn them directly into paths.
        slug="".join(c if c.isalnum() or c in ".-_" else "-" for c in raw["model"])
        raw["raw_sha256"]=hashlib.sha256(source).hexdigest()
        # Raw samples compress well; small static assets also fit static-host
        # per-file limits. No samples are dropped or rounded for publication.
        payload=json.dumps(raw,separators=(",",":"),allow_nan=False).encode()
        (args.out/f"{slug}.json.gz").write_bytes(gzip.compress(payload,mtime=0))
        (args.out/f"{slug}-quick.json").write_text(json.dumps(quick_view(raw),indent=2,allow_nan=False))
        index.append(dict(model=raw["model"],file=f"{slug}.json.gz",quick=f"{slug}-quick.json",coverage=raw["coverage"]))
        errors=[abs(r["prediction"]["error_pct"]) for r in raw["scenarios"] if r["scenario"]["holdout"] and r.get("prediction")]
        print(json.dumps(dict(model=raw["model"],coverage=raw["coverage"],eviction_ratio=raw["eviction_ratio"],
            holdout_median_error_pct=median(errors) if errors else None,holdout_max_error_pct=max(errors,default=None))))
    assert not controls,"control supplied without a matching main report"
    (args.out/"index.json").write_text(json.dumps(index,indent=2))


if __name__=="__main__":main()
