#!/usr/bin/env python3
"""Attach GPU activity durations to a kern bench report from an Nsight SQLite export.

Empty named GPU kernels delimit samples. Their durations, event nodes, cache
eviction and restoration are excluded. Model op spans include the interval
from the first enclosed device activity to the last, including internal gaps.
Whole-program call attribution comes from the uninstrumented graph; kernel
sequences are checked against each isolated call's observed implementation.
"""
import argparse
import json
import pathlib
import sqlite3
import statistics


def series(values):
    v=sorted(values); assert v and v[0]>0
    def q(p):
        at=p*(len(v)-1); lo=int(at); hi=min(lo+1,len(v)-1)
        return v[lo]+(v[hi]-v[lo])*(at-lo)
    mean=statistics.mean(v); size=(len(v)+3)//4
    return dict(samples_us=values,stats=dict(n=len(v),min=v[0],p10=q(.1),p50=q(.5),p90=q(.9),max=v[-1],mean=mean,
        cv=statistics.pstdev(v)/mean,tail_ratio=q(.9)/q(.5),block_medians=[statistics.median(values[i:i+size]) for i in range(0,len(v),size)]))


def segments(database):
    db=sqlite3.connect(f"file:{database}?mode=ro",uri=True)
    query="""SELECT k.start,k.end,s.value,k.deviceId,k.contextId,k.streamId
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id
        UNION ALL SELECT start,end,'memcpy',deviceId,contextId,streamId FROM CUPTI_ACTIVITY_KIND_MEMCPY
        UNION ALL SELECT start,end,'memset',deviceId,contextId,streamId FROM CUPTI_ACTIVITY_KIND_MEMSET
        ORDER BY 1"""
    active=None; body=[]
    for start,end,name,*stream in db.execute(query):
        if name.startswith("profile_"):
            if name=="profile_end":
                assert active is not None and active[1]==stream,"unmatched trace marker"
                assert body,"no device activities in measured range"
                yield active[0],body
                active=None;body=[]
            else:
                assert active is None,"nested trace marker"
                active=(name,stream)
        elif active is not None and active[1]==stream:
            body.append((start,end,name))
    assert active is None,"truncated activity trace"
    db.close()


def duration(body):return (max(x[1] for x in body)-min(x[0] for x in body))/1000


def attach(raw,database):
    groups=iter(segments(database)); n=raw["workload"]["samples"]
    def take(kind,count):
        out=[]
        for _ in range(count):
            found,body=next(groups)
            assert found==kind,("unexpected segment",kind,found)
            out.append(body)
        return out
    def calibrate(anchors):
        for a in anchors:
            repetitions=1 if a["name"]=="evicted_read" else 4
            bodies=take("profile_anchor_start",n*repetitions)
            a["event_timing"]=a["timing"]
            a["timing"]=series([duration(b) for b in bodies[-n:]])
    calibrate(raw["calibration_before"])
    for r in raw["scenarios"]:
        program_bodies=take("profile_program_start",n+4)[4:]
        for case in r["cases"]:
            samples={"cold":[],"warm":[]}; sequence=None
            for _ in range(n*2):
                kind,body=next(groups)
                assert kind in ("profile_cold_start","profile_warm_start"),kind
                mode="cold" if kind=="profile_cold_start" else "warm"
                observed=[x[2] for x in body]
                assert sequence is None or sequence==observed,"implementation changed between samples"
                sequence=observed; samples[mode].append(duration(body))
            case["activity_sequence"]=sequence
            for mode in ("cold","warm"):
                assert len(samples[mode])==n
                case[f"event_{mode}"]=case[mode]
                case[mode]=series(samples[mode])
        call_samples=[[] for _ in r["calls"]]
        for body in program_bodies:
            cursor=0
            for call in r["calls"]:
                expected=r["cases"][call["case"]]["activity_sequence"]
                activities=body[cursor:cursor+len(expected)]
                assert [x[2] for x in activities]==expected,(r["scenario"]["id"],call["index"],"activity sequence mismatch")
                call_samples[call["index"]].append(duration(activities))
                cursor+=len(expected)
            assert cursor==len(body),"unattributed program activities"
        for call,values in zip(r["calls"],call_samples):
            call["instrumented_events"]=call["in_program"]
            call["in_program"]=series(values)
        r["activity_graph"]=series([duration(b) for b in program_bodies])
        # Keep the independent whole-graph CUDA event measurement as the
        # prediction target. Activity duration is a cross-check of that timer.
        r["activity_vs_event_pct"]=(r["activity_graph"]["stats"]["p50"]/r["graph"]["stats"]["p50"]-1)*100
        r["attribution_source"]="unmodified model graph GPU activities; isolated implementation sequences verified"
        r["call_gap_us"]=[duration(b)-sum(v[i] for v in call_samples) for i,b in enumerate(program_bodies)]
    calibrate(raw["calibration_after"])
    assert next(groups,None) is None,"unconsumed trace segments"
    raw["protocol"]["op_timer"]="GPU activity span, excluding markers and event records; event samples retained separately"
    raw["protocol"]["attribution"]="GPU activities from the same graph as whole-program samples; no events between calls"
    raw["trace_validation"]={"all_samples_mapped":True,"all_program_activity_sequences_matched":True,
        "marker_time_excluded":True,"source":"Nsight Systems CUDA graph node trace"}
    return raw


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("report",type=pathlib.Path);p.add_argument("sqlite",type=pathlib.Path);p.add_argument("--out",required=True,type=pathlib.Path)
    a=p.parse_args();raw=attach(json.loads(a.report.read_text()),a.sqlite)
    a.out.write_text(json.dumps(raw,separators=(",",":"),allow_nan=False))
    print(json.dumps({"model":raw["model"],"scenarios":len(raw["scenarios"]),"trace_validation":raw["trace_validation"],
        "max_graph_timer_delta_pct":max(abs(r["activity_vs_event_pct"]) for r in raw["scenarios"])}))


if __name__=="__main__":main()
