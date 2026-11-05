//! Lanes gate: one program written twice, once on lane 0 alone and once
//! as two lanes that fork and join, must give the same bytes and the laned
//! one must take the time of its longest lane, eagerly and as a graph.
//!
//!   nvcc -cubin -arch=sm_103a -O3 -o target/cubins/lanes.cubin tools/kernels-src/lanes.cu
//!   cargo run --release -p kern-runtime --example lanes -- --cubin target/cubins/lanes.cubin [--gpu 0]
//!
//! Lane 0 runs two steps of `--us` (a globaltimer spin), lane 1 two of
//! twice that, and each a cuBLAS GEMM, so the two GEMMs run side by side on
//! their lanes' own handles and workspaces. The join is a PDL launch whose
//! predecessors are both branches (under capture, programmatic edges from
//! each); every checked run takes a new input, so a join that read lane
//! 1's output before lane 1 wrote it would read the previous run's.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use kern_runtime::{Capacity, Runtime};
use serde_json::{json, Value};

const N: usize = 1024;
const M: usize = 64;
const K: usize = 4096;
const NN: usize = 4096;

fn main() {
    let mut cubin = PathBuf::from("target/cubins/lanes.cubin");
    let mut gpu = 0usize;
    let mut us = 500i64;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut v = || args.next().expect("value");
        match a.as_str() {
            "--cubin" => cubin = PathBuf::from(v()),
            "--gpu" => gpu = v().parse().unwrap(),
            "--us" => us = v().parse().unwrap(),
            _ => panic!("unknown arg {a}"),
        }
    }
    let bytes = std::fs::read(&cubin).expect("cubin");
    let sha = hex::encode(<sha2::Sha256 as sha2::Digest>::digest(&bytes));
    let kernels = std::env::temp_dir().join(format!("kern-lanes-{}", std::process::id()));
    std::fs::create_dir_all(&kernels).unwrap();
    std::fs::write(kernels.join(format!("lanes-{}.cubin", &sha[..12])), &bytes).unwrap();

    let ns = us * 1000;
    let step = |x: &str, y: &str, lane: u8, after: &[u8]| {
        let ns = ns * (1 + lane as i64);
        json!({"label": format!("{x}>{y}"), "op": "step", "args": [{"buf": x}, {"buf": y}, {"i32": N}, {"i64": ns}],
               "lane": lane, "after": after})
    };
    let gemm = |c: &str, lane: u8| {
        json!({"label": c, "op": "gemm", "args": [{"buf": "ga"}, {"buf": "gw"}, {"buf": c}, {"i32": M}, {"i32": NN}, {"i32": K}],
               "lane": lane})
    };
    let join = |after: &[u8]| json!({"label": "join", "op": "join", "args": [{"buf": "b"}, {"buf": "d"}, {"buf": "y"}, {"i32": N}], "after": after});
    // Lane 1 forks at the program's start: lane 0 has no call before it.
    let laned = vec![
        step("x", "c", 1, &[0]),
        step("x", "a", 0, &[]),
        gemm("g1", 1),
        step("c", "d", 1, &[]),
        gemm("g0", 0),
        step("a", "b", 0, &[]),
        join(&[1]),
    ];
    let serial: Vec<Value> = laned
        .iter()
        .map(|c| {
            let mut c = c.clone();
            c["lane"] = json!(0);
            c["after"] = json!([]);
            c
        })
        .collect();
    let pdl = |entry: &str, params: Value| {
        json!({"params": params, "impl": {"launches": [
            {"module": "lanes", "entry": entry, "block": [256, 1, 1], "grid": [1, 1, 1], "pdl": true}]}})
    };
    let f32n = |kind: &str| json!({"dtype": "f32", "shape": [N], "kind": kind});
    let program = |calls: Vec<Value>| json!({"batch": {"groups": 1, "rows": 1}, "graph": true, "calls": calls});
    let manifest = json!({
        "schema_version": 7, "model": "lanes",
        "buffers": {
            "x": f32n("input"), "a": f32n("workspace"), "b": f32n("workspace"), "c": f32n("workspace"),
            "d": f32n("workspace"), "y": f32n("output"),
            "ga": {"dtype": "bf16", "shape": [M, K], "kind": "input"},
            "gw": {"dtype": "bf16", "shape": [NN, K], "kind": "input"},
            "g0": {"dtype": "bf16", "shape": [M, NN], "kind": "output"},
            "g1": {"dtype": "bf16", "shape": [M, NN], "kind": "output"}
        },
        "modules": {"lanes": {"source": "lanes.cubin", "sha256": sha}},
        "ops": {
            "step": pdl("kern_lanes_step", json!(["in buffer<f32>", "out buffer<f32>", "i32", "i64"])),
            "join": pdl("kern_lanes_join", json!(["in buffer<f32>", "in buffer<f32>", "out buffer<f32>", "i32"])),
            "gemm": {"params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i32"],
                     "impl": {"launches": [{"entry": "extern:cublaslt_bf16_tn"}]}}
        },
        "programs": {"serial": program(serial), "laned": program(laned)}
    });
    let json = serde_json::to_string_pretty(&manifest).unwrap();
    let verified = kern_manifest::Verified::from_json(&json).unwrap_or_else(|e| panic!("{e}"));

    let capacity = Capacity { tokens: Some(1), slots: 1, landings: 0 };
    let mut rt = Runtime::load(&verified, Some(&kernels), gpu, Some(capacity), None).unwrap();
    let x =
        |run: usize| -> Vec<u8> { (0..N).flat_map(|i| (i as f32 * 0.25 - 7.0 + run as f32).to_le_bytes()).collect() };
    let bf16 = |n: usize, seed: u32| -> Vec<u8> {
        (0..n as u32)
            .flat_map(|i| {
                let h = (i ^ seed).wrapping_mul(2654435761) >> 20;
                half::bf16::from_f32(h as f32 / 4096.0 - 0.5).to_le_bytes()
            })
            .collect()
    };
    rt.write_input("ga", &bf16(M * K, 1)).unwrap();
    rt.write_input("gw", &bf16(NN * K, 2)).unwrap();
    let vars = BTreeMap::new();

    let mut ok = true;
    let mut outputs = Vec::new();
    for graph in [false, true] {
        rt.set_eager(!graph);
        for program in ["serial", "laned"] {
            for run in 1..=4 {
                rt.write_input("x", &x(run)).unwrap();
                rt.issue(program, &vars).unwrap();
                rt.synchronize().unwrap();
                // Each branch computes 4 x + 3, exact in f32; the join adds them.
                let want: Vec<u8> =
                    (0..N).flat_map(|i| (8.0 * (i as f32 * 0.25 - 7.0 + run as f32) + 6.0).to_le_bytes()).collect();
                if rt.read_output("y").unwrap() != want {
                    println!("{} {program} run {run}: the join read a branch before it finished", mode(graph));
                    ok = false;
                }
            }
            rt.write_input("x", &x(0)).unwrap();
            let runs = 20;
            let t = Instant::now();
            for _ in 0..runs {
                rt.issue(program, &vars).unwrap();
            }
            rt.synchronize().unwrap();
            let ms = t.elapsed().as_secs_f64() * 1e3 / runs as f64;
            let out: Vec<Vec<u8>> = ["y", "g0", "g1"].iter().map(|b| rt.read_output(b).unwrap()).collect();
            println!("{} {program}: {ms:.3} ms a run (steps of {us} and {} us)", mode(graph), 2 * us);
            outputs.push((graph, program, ms, out));
        }
    }
    let first = &outputs[0].3;
    for (graph, program, _, out) in &outputs {
        if out != first {
            println!("{program} (graph {graph}) differs from eager serial");
            ok = false;
        }
    }
    if first[1] != first[2] {
        println!("the two lanes' GEMMs differ");
        ok = false;
    }
    for pair in outputs.chunks(2) {
        let (serial, laned) = (pair[0].2, pair[1].2);
        if laned > 0.75 * serial {
            println!("{}: laned {laned:.3} ms is not below 0.75 x serial {serial:.3} ms", mode(pair[0].0));
            ok = false;
        }
    }
    println!("{}", if ok { "PASS" } else { "FAIL" });
    std::process::exit(if ok { 0 } else { 1 });
}

fn mode(graph: bool) -> &'static str {
    if graph {
        "graph"
    } else {
        "eager"
    }
}
