//! E5 gate, first stone: the tray-local collectives (tools/kernels-src/
//! peer_collective.cu) on N runtimes of one tray. Every rank fills its input
//! with a rank-and-row pattern, runs the one-shot allreduce and the
//! allgather once eagerly and checks every element against the host in the
//! "own rows first" layout, then times captured bursts of each at the
//! sizes a K3 decode step exchanges (f32 [R*B, 7168] partials, bf16
//! [B, 3584] and f32 [B, 7168] rows).
//!
//!   cargo run --release -p kern-runtime --example peer_collective -- \
//!       --cubin target/cubins/peer_collective.cubin [--gpus 0,1,2,3] [--iters 50] [--burst 32]
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Barrier, Mutex};

use kern_runtime::{PeerHandle, Runtime, Topology};

const H: usize = 7168;
const B_MAX: usize = 64;
const RB_MAX: usize = 4 * H;
const TIMEOUT_NS: i64 = 2_000_000_000;

/// Rank `r`'s allreduce input at tray row `t`, column `i`: small integers, so
/// the f32 sum over ranks is exact in any order.
fn ar_value(r: usize, t: usize, i: usize) -> f32 {
    (((r + 1) * (t * 7 + i)) % 13) as f32
}

fn ag_byte(r: usize, j: usize, k: usize) -> u8 {
    (r * 31 + j * 7 + k) as u8
}

fn main() {
    let mut cubin = PathBuf::from("target/cubins/peer_collective.cubin");
    let mut gpus: Vec<usize> = vec![0, 1, 2, 3];
    let mut iters = 50usize;
    let mut burst = 32usize;
    let mut verify = true;
    let mut grid = 128usize;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut v = || args.next().expect("value");
        match a.as_str() {
            "--cubin" => cubin = PathBuf::from(v()),
            "--gpus" => gpus = v().split(',').map(|s| s.parse().unwrap()).collect(),
            "--iters" => iters = v().parse().unwrap(),
            "--burst" => burst = v().parse().unwrap(),
            "--no-verify" => verify = false,
            "--grid" => grid = v().parse().unwrap(),
            _ => panic!("unknown arg {a}"),
        }
    }
    let n = gpus.len();
    let bytes = std::fs::read(&cubin).expect("cubin");
    let sha = format!("{:x}", <sha2::Sha256 as sha2::Digest>::digest(&bytes));
    let kernels = std::env::temp_dir().join(format!("kern-peer-collective-{}", std::process::id()));
    std::fs::create_dir_all(&kernels).unwrap();
    std::fs::write(kernels.join(format!("peer_collective-{}.cubin", &sha[..12])), &bytes).unwrap();

    let ar_region = n * B_MAX * H / 2;
    let ag_region = B_MAX * RB_MAX / 8;
    let ar = serde_json::json!({ "op": "allreduce", "args": [
        { "buf": "ar_x" }, { "buf": "ar_y" }, { "buf": "ar_sym" }, { "buf": "ar_peers" }, { "buf": "ar_epochs" },
        { "buf": "err" }, { "rank": "tp" }, { "i32": n }, { "var": "tokens" }, { "i32": H }, { "i32": ar_region },
        { "i64": TIMEOUT_NS }
    ]});
    let ag = serde_json::json!({ "op": "allgather", "args": [
        { "buf": "ag_x" }, { "buf": "ag_y" }, { "buf": "ag_sym" }, { "buf": "ag_peers" }, { "buf": "ag_epochs" },
        { "buf": "err" }, { "rank": "tp" }, { "i32": n }, { "var": "tokens" }, { "var": "rb" }, { "i32": ag_region },
        { "i64": TIMEOUT_NS }
    ]});
    let params = |io: &str| -> Vec<String> {
        [
            &format!("in buffer<{io}>"),
            &format!("out buffer<{io}>"),
            "inout buffer<u8>",
            "in buffer<u64>",
            "inout buffer<u32>",
            "out buffer<i32>",
            "i32",
            "i32",
            "i32",
            "i32",
            "i32",
            "i64",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    };
    let (ar_params, ag_params) = (params("f32"), params("u8"));
    let launch = |entry: &str| serde_json::json!([{ "module": "peer_collective", "entry": entry, "block": [256, 1, 1], "grid": [grid, 1, 1] }]);
    let manifest = serde_json::json!({
        "schema_version": 3,
        "model": "peer-collective",
        "vars": { "tokens": { "max": B_MAX }, "rows": { "max": n * B_MAX }, "rb": { "max": RB_MAX } },
        "topology": { "groups": { "tp": n } },
        "buffers": {
            "ar_x": { "dtype": "f32", "shape": ["rows", H], "kind": "input" },
            "ar_y": { "dtype": "f32", "shape": ["rows", H], "kind": "output" },
            "ar_sym": { "dtype": "u8", "shape": [2 * n * ar_region * 16], "kind": "carry", "export": true },
            "ar_peers": { "dtype": "u64", "shape": [n], "kind": "peer", "of": "ar_sym", "group": "tp" },
            "ar_epochs": { "dtype": "u32", "shape": [grid], "kind": "carry" },
            "ag_x": { "dtype": "u8", "shape": ["tokens", "rb"], "kind": "input" },
            "ag_y": { "dtype": "u8", "shape": ["rows", "rb"], "kind": "output" },
            "ag_sym": { "dtype": "u8", "shape": [2 * n * ag_region * 16], "kind": "carry", "export": true },
            "ag_peers": { "dtype": "u64", "shape": [n], "kind": "peer", "of": "ag_sym", "group": "tp" },
            "ag_epochs": { "dtype": "u32", "shape": [grid], "kind": "carry" },
            "err": { "dtype": "i32", "shape": [1], "kind": "output" }
        },
        "modules": { "peer_collective": { "source": "peer_collective.cubin", "sha256": sha } },
        "ops": {
            "allreduce": {
                "params": ar_params,
                "impl": { "launches": launch("kern_peer_allreduce_f32") }
            },
            "allgather": {
                "params": ag_params,
                "impl": { "launches": launch("kern_peer_allgather") }
            }
        },
        "programs": {
            "allreduce": [ar.clone()],
            "allgather": [ag.clone()],
            "ar_burst": std::iter::repeat_n(ar, burst).collect::<Vec<_>>(),
            "ag_burst": std::iter::repeat_n(ag, burst).collect::<Vec<_>>()
        }
    });
    let json = serde_json::to_string_pretty(&manifest).unwrap();
    std::fs::write(kernels.join("peer-collective.json"), &json).unwrap();
    eprintln!("manifest + cubin in {}", kernels.display());

    // (rows per rank, allgather row bytes): the shapes a K3 step exchanges.
    let configs: Vec<(usize, usize)> = vec![(1, 2 * 3584), (4, 2 * 3584), (16, 2 * 3584), (16, 4 * H), (64, 4 * H)];

    let posted: Arc<Mutex<Vec<Option<BTreeMap<String, PeerHandle>>>>> = Arc::new(Mutex::new(vec![None; n]));
    let gate = Arc::new(Barrier::new(n));
    let results: Arc<Mutex<Vec<Option<Result<Vec<String>, String>>>>> = Arc::new(Mutex::new(vec![None; n]));
    let mut threads = Vec::new();
    for (rank, &gpu) in gpus.iter().enumerate() {
        let (json, kernels, posted, gate, results, configs) =
            (json.clone(), kernels.clone(), posted.clone(), gate.clone(), results.clone(), configs.clone());
        threads.push(std::thread::spawn(move || {
            let run = || -> Result<Vec<String>, String> {
                let topo = Topology::one("tp", rank as u64, n as u64);
                let e = |e: kern_runtime::Error| e.to_string();
                let mut rt = Runtime::load(&json, &kernels, gpu, Some(kern_runtime::Capacity { tokens: Some(1), seqs: 1 }), Some(&topo)).map_err(e)?;
                let mine = rt.export_handles().map_err(e)?;
                posted.lock().unwrap()[rank] = Some(mine);
                gate.wait();
                let members: Vec<_> = posted.lock().unwrap().iter().map(|m| m.clone().unwrap()).collect();
                rt.import_peers("tp", &members).map_err(e)?;
                gate.wait();
                let mut lines = Vec::new();
                for &(b, rb) in &configs {
                    let rows = n * b;
                    let env = BTreeMap::from([
                        ("tokens".to_string(), b as u64),
                        ("rows".to_string(), rows as u64),
                        ("rb".to_string(), rb as u64),
                    ]);
                    // Local row j is tray row (rank*b + j) mod rows.
                    let tray = |j: usize| (rank * b + j) % rows;
                    let x: Vec<u8> =
                        (0..rows).flat_map(|j| (0..H).flat_map(move |i| ar_value(rank, tray(j), i).to_le_bytes())).collect();
                    rt.write_input_at("ar_x", &x, &env).map_err(e)?;
                    let gx: Vec<u8> = (0..b).flat_map(|j| (0..rb).map(move |k| ag_byte(rank, j, k))).collect();
                    rt.write_input_at("ag_x", &gx, &env).map_err(e)?;
                    gate.wait();
                    rt.run("allreduce", &env).map_err(e)?;
                    rt.run("allgather", &env).map_err(e)?;
                    let err = i32::from_le_bytes(rt.read_output("err").map_err(e)?[..4].try_into().unwrap());
                    if err != 0 { return Err(format!("B={b}: a slot from rank {} never arrived", err - 1)); }
                    let y = rt.read_output("ar_y").map_err(e)?;
                    let mut bad = 0usize;
                    for j in 0..if verify { rows } else { 0 } {
                        for i in 0..H {
                            let got = f32::from_le_bytes(y[(j * H + i) * 4..][..4].try_into().unwrap());
                            let want: f32 = (0..n).map(|r| ar_value(r, tray(j), i)).sum();
                            if got != want {
                                if bad == 0 {
                                    eprintln!("rank {rank} B={b}: allreduce row {j} col {i}: got {got}, want {want}");
                                }
                                bad += 1;
                            }
                        }
                    }
                    if bad != 0 { return Err(format!("B={b}: {bad} wrong allreduce elements")); }
                    let gy = rt.read_output("ag_y").map_err(e)?;
                    let mut bad = 0usize;
                    for d in 0..if verify { n } else { 0 } {
                        let src = (rank + d) % n;
                        for j in 0..b {
                            for k in 0..rb {
                                if gy[(d * b + j) * rb + k] != ag_byte(src, j, k) {
                                    if bad == 0 {
                                        eprintln!("rank {rank} B={b} rb={rb}: allgather block {d} row {j} byte {k} wrong");
                                    }
                                    bad += 1;
                                }
                            }
                        }
                    }
                    if bad != 0 { return Err(format!("B={b} rb={rb}: {bad} wrong allgather bytes")); }
                    rt.capture("ar_burst", &env).map_err(e)?;
                    rt.capture("ag_burst", &env).map_err(e)?;
                    gate.wait();
                    let ar_us = rt.time_captured("ar_burst", &env, iters).map_err(e)? as f64 * 1e3 / burst as f64;
                    gate.wait();
                    let ag_us = rt.time_captured("ag_burst", &env, iters).map_err(e)? as f64 * 1e3 / burst as f64;
                    let err = i32::from_le_bytes(rt.read_output("err").map_err(e)?[..4].try_into().unwrap());
                    if err != 0 { return Err(format!("B={b}: burst: a slot from rank {} never arrived", err - 1)); }
                    lines.push(format!(
                        "B={b:>2} rows={rows:>3}: allreduce f32 [{rows},{H}] ({:.2} MB) {ar_us:.2} us; allgather [{b},{rb}] -> [{rows},{rb}] ({:.2} MB) {ag_us:.2} us",
                        (rows * H * 4) as f64 / 1e6,
                        (rows * rb) as f64 / 1e6
                    ));
                }
                Ok(lines)
            };
            let r = run();
            results.lock().unwrap()[rank] = Some(r);
        }));
    }
    for t in threads {
        t.join().unwrap();
    }
    let mut ok = true;
    for (rank, r) in results.lock().unwrap().iter().enumerate() {
        match r {
            Some(Ok(lines)) => {
                println!("rank {rank} gpu {}: all elements verified", gpus[rank]);
                if rank == 0 {
                    for l in lines {
                        println!("  {l}");
                    }
                }
            }
            Some(Err(e)) => {
                ok = false;
                println!("rank {rank}: {e}");
            }
            None => {
                ok = false;
                println!("rank {rank}: no result");
            }
        }
    }
    println!("{}", if ok { "PASS" } else { "FAIL" });
    std::process::exit(if ok { 0 } else { 1 });
}
