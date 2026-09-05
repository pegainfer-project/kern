"""CPU-only regression checks for trace mapping and statistical evidence."""
import copy
import pathlib
import sqlite3
import tempfile
import unittest

from profile_export import export, aggregate, attach_control
from profile_trace import series, segments, duration


def record(name, cold, warm, graph, holdout=False):
    return dict(scenario=dict(id=name, holdout=holdout), program="example",
        graph=series([graph]*12), instrumented=series([graph*2]*3), output_check="matched",
        calls=[dict(index=0, case=0, op="matmul", launches=[{}], in_program=series([graph]*12))],
        cases=[dict(cold=series([cold]*12), warm=series([warm]*12))])


def report():
    records=[record("train-a",10,8,9),record("train-b",20,12,16),record("held",30,16,23,True)]
    anchors=[dict(name="l2_read",timing=series([2]*12)),dict(name="evicted_read",timing=series([6]*12))]
    return dict(workload=dict(samples=12,scenarios=[r["scenario"] for r in records]), scenarios=records,
        calibration_before=anchors,calibration_after=copy.deepcopy(anchors),
        trace_validation=dict(all_program_activity_sequences_matched=True))


class StatisticsTests(unittest.TestCase):
    def test_tails_order_and_quantiles(self):
        values=[1.]*8+[10.,30.,50.,100.]
        measured=series(values)
        self.assertEqual(measured["samples_us"],values)
        self.assertEqual(measured["stats"]["p50"],1.)
        self.assertEqual(measured["stats"]["max"],100.)
        self.assertGreater(measured["stats"]["cv"],1.)
        self.assertGreater(measured["stats"]["block_medians"][-1],1.)

    def test_holdout_target_cannot_change_fit_or_prediction(self):
        a=report();b=copy.deepcopy(a)
        b["scenarios"][-1]["graph"]=series([23000.]*12)
        x,y=export(a),export(b)
        self.assertEqual(x["composition"]["fits"],y["composition"]["fits"])
        self.assertEqual(x["scenarios"][-1]["prediction"]["us"],y["scenarios"][-1]["prediction"]["us"])
        self.assertNotEqual(x["scenarios"][-1]["prediction"]["error_pct"],y["scenarios"][-1]["prediction"]["error_pct"])

    def test_program_variation_is_not_hidden_by_stable_microbench(self):
        r=record("variable",10,8,9)
        r["calls"][0]["in_program"]=series([9.]*10+[40.,90.])
        self.assertTrue(aggregate(r)[0]["variable"])

    def test_trace_attribution_cannot_change_cost_prediction(self):
        a=report();b=copy.deepcopy(a)
        for r in b["scenarios"]:
            r["calls"][0]["in_program"]=series([9000.]*12)
        x,y=export(a),export(b)
        self.assertEqual(x["composition"]["fits"],y["composition"]["fits"])
        self.assertEqual([r["prediction"] for r in x["scenarios"]],
                         [r["prediction"] for r in y["scenarios"]])
        self.assertNotEqual(x["scenarios"][0]["ops"][0]["attributed_us"],
                            y["scenarios"][0]["ops"][0]["attributed_us"])

    def test_holdout_prediction_still_uses_measured_microbench_costs(self):
        a=report();b=copy.deepcopy(a)
        for mode in ("cold","warm"):
            case=b["scenarios"][-1]["cases"][0]
            case[mode]=series([v*2 for v in case[mode]["samples_us"]])
        x,y=export(a),export(b)
        self.assertEqual(x["composition"]["fits"],y["composition"]["fits"])
        self.assertAlmostEqual(y["scenarios"][-1]["prediction"]["us"],
                               2*x["scenarios"][-1]["prediction"]["us"])

    def test_incomplete_and_untraced_reports_are_rejected(self):
        for field in ("calibration_after","trace_validation"):
            r=report();del r[field]
            with self.assertRaises(AssertionError):export(r)
        r=report();r["scenarios"].pop()
        with self.assertRaises(AssertionError):export(r)

    def test_missing_tail_sample_is_rejected(self):
        r=report();r["scenarios"][0]["cases"][0]["cold"]["samples_us"].pop()
        with self.assertRaises(AssertionError):export(r)

    def test_control_must_match_workload_and_outputs(self):
        raw=report()
        raw.update(model="example",manifest_sha256="abc",hardware={})
        for r in raw["scenarios"]:r["outputs"]=[1]
        control=copy.deepcopy(raw);control.update(program_only=True,runner_sha256="def")
        attach_control(raw,control)
        self.assertEqual(raw["repeat_check"]["max_abs_delta_pct"],0.)
        control["scenarios"][0]["outputs"]=[2]
        with self.assertRaises(AssertionError):attach_control(raw,control)


class TraceTests(unittest.TestCase):
    def test_markers_excluded_internal_gaps_preserved_other_stream_ignored(self):
        with tempfile.TemporaryDirectory() as folder:
            path=pathlib.Path(folder)/"trace.sqlite"
            db=sqlite3.connect(path)
            db.executescript("""
                CREATE TABLE StringIds(id INTEGER,value TEXT);
                CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,shortName INTEGER,deviceId INTEGER,contextId INTEGER,streamId INTEGER);
                CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER,end INTEGER,deviceId INTEGER,contextId INTEGER,streamId INTEGER);
                CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET(start INTEGER,end INTEGER,deviceId INTEGER,contextId INTEGER,streamId INTEGER);
                INSERT INTO StringIds VALUES (0,'profile_cold_start'),(1,'op_a'),(2,'op_b'),(3,'profile_end');
                INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
                    (0,1000,0,0,1,1),(2000,3000,1,0,1,1),(2500,999999,1,0,1,2),
                    (4000,6000,2,0,1,1),(7000,8000,3,0,1,1);
            """)
            db.commit();db.close()
            groups=list(segments(path))
            self.assertEqual(len(groups),1)
            kind,body=groups[0]
            self.assertEqual(kind,"profile_cold_start")
            self.assertEqual([x[2] for x in body],["op_a","op_b"])
            self.assertEqual(duration(body),4.)


if __name__=="__main__":unittest.main()
