//! Lanes, driven through `verify`: a program whose calls run side by side
//! is accepted exactly when every serial order of it is the list order.
//! The reference is a graph model: explicit edges, reachability by search,
//! every pair of calls compared.

use kern_manifest::{verify, Manifest, Verified, VerifyErrors};
use serde_json::{json, Value};

/// Two inout buffers and three ops: a reader, a writer, and a reader with
/// impl scratch. Program `all` uses every declaration on lane 0, so a
/// program under test only has to be about its lanes.
fn base(calls: Vec<Value>) -> Value {
    let launch = json!({ "module": "toy", "entry": "k", "block": [32, 1, 1], "grid": [1, 1, 1] });
    let scratch_launch = |dir: &str| {
        json!({ "module": "toy", "entry": "k", "block": [32, 1, 1], "grid": [1, 1, 1],
                "params": ["in buffer<bf16>", format!("{dir} buffer<f32>")],
                "args": [{ "param": 0 }, { "scratch": "part" }] })
    };
    json!({
        "schema_version": 7, "model": "lanes",
        "buffers": {
            "u": { "dtype": "bf16", "shape": [64], "kind": "inout" },
            "v": { "dtype": "bf16", "shape": [64], "kind": "inout" }
        },
        "modules": { "toy": { "source": "toy.cubin", "sha256": "ab".repeat(32) } },
        "ops": {
            "r": { "params": ["in buffer<bf16>"], "impl": { "launches": [launch] } },
            "w": { "params": ["out buffer<bf16>"], "impl": { "launches": [launch] } },
            "s": { "params": ["in buffer<bf16>"],
                   "impl": { "scratch": { "part": { "dtype": "f32", "shape": [8] } },
                             "launches": [scratch_launch("out"), scratch_launch("in")] } }
        },
        "programs": {
            "all": { "calls": [call("w", "u", 0, &[]), call("w", "v", 0, &[]), call("r", "u", 0, &[]), call("s", "v", 0, &[])] },
            "p": { "calls": calls }
        }
    })
}

fn call(op: &str, buf: &str, lane: u8, after: &[u8]) -> Value {
    let mut c = json!({ "op": op, "args": [{ "buf": buf }] });
    if lane != 0 {
        c["lane"] = json!(lane);
    }
    if !after.is_empty() {
        c["after"] = json!(after);
    }
    c
}

fn check(v: Value) -> Result<Verified, VerifyErrors> {
    let m: Manifest = serde_json::from_value(v).map_err(|e| VerifyErrors(vec![e.to_string()]))?;
    verify(m)
}

fn errors(v: Value) -> Vec<String> {
    check(v).map(|_| Vec::new()).unwrap_or_else(|e| e.0)
}

fn has(errs: &[String], needle: &str) -> bool {
    errs.iter().any(|e| e.contains(needle))
}

/// One call of the model: its lane, the lanes it runs after, the buffer
/// it touches, whether it writes it, and whether it holds the op scratch.
#[derive(Clone, Copy, Debug)]
struct C {
    lane: u8,
    after: u8,
    buf: u8,
    writes: bool,
    scratch: bool,
}

impl C {
    fn after(&self) -> Vec<u8> {
        (0..3).filter(|l| self.after & (1 << l) != 0).collect()
    }

    fn json(&self) -> Value {
        let (op, buf) = (
            if self.scratch {
                "s"
            } else if self.writes {
                "w"
            } else {
                "r"
            },
            ["u", "v"][self.buf as usize],
        );
        call(op, buf, self.lane, &self.after())
    }
}

/// The model's verdict: whether the program is well formed and race free.
fn model(p: &[C]) -> bool {
    let n = p.len();
    let mut edges = vec![Vec::new(); n];
    let mut ok = true;
    for (i, c) in p.iter().enumerate() {
        let latest = |l: u8| (0..i).rev().find(|&j| p[j].lane == l);
        if let Some(j) = latest(c.lane) {
            edges[i].push(j);
        } else if c.lane != 0 && c.after().is_empty() {
            ok = false;
        }
        for l in c.after() {
            match latest(l) {
                _ if l == c.lane => ok = false,
                Some(j) => edges[i].push(j),
                None if l != 0 => ok = false,
                None => {}
            }
        }
    }
    let reaches = |from: usize, to: usize| {
        let mut stack = vec![from];
        let mut seen = vec![false; n];
        while let Some(x) = stack.pop() {
            if x == to {
                return true;
            }
            if !std::mem::replace(&mut seen[x], true) {
                stack.extend(&edges[x]);
            }
        }
        false
    };
    let last = n - 1;
    ok &= p[last].lane == 0 && (0..n).all(|i| reaches(last, i));
    let lanes: Vec<u8> = p.iter().map(|c| c.lane).collect();
    let top = lanes.iter().copied().max().unwrap_or(0);
    ok &= (1..=top).all(|l| lanes.contains(&l));
    for b in 0..n {
        for a in 0..b {
            let concurrent = !reaches(b, a);
            let (x, y) = (&p[a], &p[b]);
            let shares = (x.buf == y.buf && (x.writes || y.writes)) || (x.scratch && y.scratch);
            ok &= !(concurrent && shares);
        }
    }
    ok
}

/// A fixed-seed generator: the same programs every run.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self, n: u64) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) % n
    }

    fn call(&mut self) -> C {
        let scratch = self.next(5) == 0;
        C {
            lane: [0, 0, 0, 1, 1, 2][self.next(6) as usize],
            after: [0, 0, 1, 2, 4, 3, 5, 6][self.next(8) as usize],
            buf: self.next(2) as u8,
            writes: !scratch && self.next(2) == 0,
            scratch,
        }
    }
}

#[test]
fn verify_accepts_exactly_the_programs_the_model_does() {
    let mut rng = Lcg(7);
    let mut verdicts = [0usize; 2];
    for _ in 0..20_000 {
        let n = 1 + rng.next(6) as usize;
        let mut p: Vec<C> = (0..n).map(|_| rng.call()).collect();
        if rng.next(2) == 0 {
            // Half the programs end joining every side lane they use.
            let used = p[..n - 1].iter().filter(|c| c.lane != 0).fold(0, |m, c| m | 1 << c.lane);
            p[n - 1].lane = 0;
            p[n - 1].after = used;
        }
        let errs = errors(base(p.iter().map(C::json).collect()));
        let expected = model(&p);
        assert_eq!(errs.is_empty(), expected, "{p:?}: {errs:#?}");
        verdicts[expected as usize] += 1;
    }
    assert!(verdicts.iter().all(|&v| v > 1000), "both verdicts are exercised: {verdicts:?}");
}

#[test]
fn a_fork_and_a_join_verify() {
    let p = vec![
        call("w", "u", 0, &[]),
        call("r", "u", 1, &[0]),
        call("w", "v", 1, &[]),
        call("r", "u", 0, &[]),
        call("r", "v", 0, &[1]),
    ];
    let m = check(base(p)).unwrap();
    assert_eq!((m.programs["p"].calls[1].lane, m.programs["p"].calls[4].after.as_slice()), (1, &[1u8][..]));
    let j = m.to_json();
    assert_eq!(
        (j.matches("\"lane\"").count(), j.matches("\"after\"").count()),
        (2, 2),
        "lane 0 and no `after` stay implicit"
    );
}

#[test]
fn a_race_names_both_calls_and_the_writer() {
    let p = vec![call("w", "u", 0, &[]), call("w", "v", 1, &[0]), call("r", "v", 0, &[]), call("r", "u", 0, &[1])];
    let errs = errors(base(p));
    assert!(
        has(&errs, "program `p` call #1 (lane 1) and program `p` call #2 (lane 0) both touch buffer `v` and neither runs after the other; program `p` call #1 (lane 1) writes it"),
        "{errs:#?}"
    );
}

#[test]
fn two_calls_of_an_op_with_scratch_never_overlap() {
    let p = vec![call("w", "u", 0, &[]), call("s", "u", 1, &[0]), call("s", "v", 0, &[]), call("r", "u", 0, &[1])];
    assert!(has(&errors(base(p)), "both touch the scratch of op `s`"));
}

#[test]
fn every_lane_joins_before_the_program_ends() {
    let p = vec![call("w", "u", 0, &[]), call("r", "u", 1, &[0]), call("r", "v", 0, &[])];
    assert!(has(&errors(base(p)), "lane 1 is never joined: 1 of its 1 calls"));
    let p = vec![call("w", "u", 0, &[]), call("r", "u", 1, &[0])];
    assert!(has(&errors(base(p)), "a program ends on lane 0, after every lane"));
}

#[test]
fn after_names_lanes_that_ran_before() {
    let p = vec![call("r", "u", 0, &[1]), call("r", "u", 1, &[0]), call("r", "v", 0, &[1])];
    assert!(has(&errors(base(p)), "`after` names lane 1, which runs no call before it"));
    let p = vec![call("r", "u", 1, &[0, 1]), call("r", "v", 0, &[1])];
    assert!(has(&errors(base(p)), "`after` names its own lane"));
    let p = vec![call("r", "u", 0, &[]), call("r", "u", 1, &[]), call("r", "v", 0, &[1])];
    assert!(has(&errors(base(p)), "a side lane's first call names the lanes it runs `after`"));
    let p = vec![call("r", "u", 0, &[]), call("r", "u", 1, &[0, 0]), call("r", "v", 0, &[1])];
    assert!(has(&errors(base(p)), "`after` names lane 0 twice"));
}

#[test]
fn lanes_are_numbered_densely() {
    let p = vec![call("r", "u", 0, &[]), call("r", "u", 2, &[0]), call("r", "v", 0, &[2])];
    assert!(has(&errors(base(p)), "lane 1 runs no call, but lane 2 does"));
}

#[test]
fn cross_rank_work_stays_on_lane_0() {
    let mut v = base(vec![call("r", "u", 0, &[]), call("x", "f", 1, &[0]), call("r", "v", 0, &[1])]);
    v["topology"] = json!({ "groups": { "tp": 2 } });
    v["buffers"]["f"] = json!({ "dtype": "bf16", "shape": [64], "kind": "inout", "export": true });
    v["buffers"]["fp"] = json!({ "dtype": "u64", "shape": [2], "kind": "peer", "of": "f", "group": "tp" });
    v["ops"]["x"] = json!({ "params": ["in buffer<bf16>"], "impl": { "launches": [{ "module": "toy", "entry": "k", "block": [32, 1, 1], "grid": [1, 1, 1] }] } });
    v["ops"]["y"] = json!({ "params": ["in buffer<u64>", "i32"], "impl": { "launches": [{ "module": "toy", "entry": "k", "block": [32, 1, 1], "grid": [1, 1, 1] }] } });
    v["ops"]["ar"] = json!({ "params": ["in buffer<bf16>", "out buffer<bf16>", "i32", "i32"],
                             "impl": { "launches": [{ "entry": "extern:nccl_allreduce_bf16" }] } });
    let mut lane0 = v.clone();
    lane0["programs"]["p"]["calls"][1] = call("x", "f", 0, &[]);
    lane0["programs"]["p"]["calls"][2] = call("r", "v", 0, &[]);
    lane0["programs"]["q"] = json!({ "calls": [
        { "op": "y", "args": [{ "buf": "fp" }, { "rank": "tp" }] },
        { "op": "ar", "args": [{ "buf": "u" }, { "buf": "v" }, { "i32": 64 }, { "rank": "tp" }] } ] });
    check(lane0.clone()).unwrap();
    assert!(has(&errors(v), "takes `f`, which a peer buffer maps; cross-rank work stays on lane 0"));
    let mut peer = lane0.clone();
    peer["programs"]["q"]["calls"][0]["lane"] = json!(1);
    peer["programs"]["q"]["calls"][0]["after"] = json!([0]);
    peer["programs"]["q"]["calls"][1]["after"] = json!([1]);
    assert!(has(&errors(peer), "takes peer buffer `fp`"));
    let mut coll = lane0;
    coll["programs"]["q"]["calls"][1]["lane"] = json!(1);
    coll["programs"]["q"]["calls"][1]["after"] = json!([0]);
    coll["programs"]["q"]["calls"].as_array_mut().unwrap().push(call("r", "u", 0, &[1]));
    assert!(has(&errors(coll), "op `ar` is a collective"));
}
