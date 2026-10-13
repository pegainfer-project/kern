//! Collectives across processes and nodes, one process per GPU as kern-serve
//! runs them: the K4 gate for `extern:nccl_alltoall_{bf16,f32}` and the
//! latency table of the allreduce choices a decode step has.
//!
//! The all-to-all is exchanged at 0.3-8 MiB per rank, `--iters` times
//! eagerly and `--iters` times as a graph replay per size and type, each
//! time with a fresh pattern every received byte is checked against, then
//! timed as a captured burst. The allreduce table puts NCCL (bf16 and f32)
//! beside the Lamport one-shot and the two-shot of peer_allreduce.cu when
//! `--ar-cubin` (built with `-DNRANKS=<world>`) is given; every size is
//! checked once before it is timed. `--capture-first` captures the first
//! all-to-all before anything ran eagerly, the order a server that warms
//! up by capturing has.
//!
//! Rank 0 listens on the rendezvous port, collects every rank's offer
//! (its fabric handles, rank 0's with the communicator's id) and sends
//! the list back.
//!
//!   cross_collective --rank R --world W --gpu G --rendezvous host:port \
//!       [--ar-cubin peer_allreduce.cubin] [--iters 100] [--burst 16] [--capture-first]
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;

use kern_runtime::{Capacity, Offer, Runtime, Topology};

const H: usize = 7168;
const R_MAX: usize = 64;
const A2A_MAX: usize = 8 << 20;
const TIMEOUT_NS: i64 = 2_000_000_000;
const MIB: f64 = (1 << 20) as f64;

type Vars = BTreeMap<String, u64>;

fn vars(kv: &[(&str, usize)]) -> Vars {
    kv.iter().map(|(k, v)| (k.to_string(), *v as u64)).collect()
}

/// A word that names its sender, receiver, element and round, with the
/// exponent kept below all-ones so neither type ever holds a NaN.
fn word(from: usize, to: usize, i: usize, round: usize) -> u32 {
    let mut h = (from as u32).wrapping_mul(0x9E37_79B1)
        ^ (to as u32).wrapping_mul(0x85EB_CA77)
        ^ (i as u32).wrapping_mul(0xC2B2_AE3D)
        ^ (round as u32).wrapping_mul(0x27D4_EB2F);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 13;
    h & 0x7EFF_FFFF
}

fn a2a_bytes(esize: usize, from: usize, to: usize, n: usize, round: usize) -> Vec<u8> {
    let w = |i| word(from, to, i, round);
    match esize {
        2 => (0..n).flat_map(|i| ((w(i) >> 16) as u16).to_le_bytes()).collect(),
        _ => (0..n).flat_map(|i| w(i).to_le_bytes()).collect(),
    }
}

/// Rank `r`'s allreduce input: small integers, exact summed in any order
/// and in either type.
fn ar_value(r: usize, i: usize, round: usize) -> f32 {
    (((r + 1) * (i * 7 + round)) % 13) as f32
}

/// Rank `q`'s block of a peer allreduce's `rows`: an even deal, the last
/// entry `rows` itself.
fn blocks(world: usize, rows: usize) -> Vec<usize> {
    (0..=world).map(|q| q * rows / world).collect()
}

fn manifest(world: usize, peer: Option<&str>, burst: usize) -> serde_json::Value {
    let io = |dt: &str| vec![format!("in buffer<{dt}>"), format!("out buffer<{dt}>"), "i64".into()];
    let nccl = |coll: &str, dt: &str| {
        let mut params = io(dt);
        params.push("i32".into());
        serde_json::json!({ "params": io(dt), "impl": { "launches": [{
            "entry": format!("extern:nccl_{coll}_{dt}"), "params": params,
            "args": [{ "param": 0 }, { "param": 1 }, { "param": 2 }, { "rank": "w" }] }] } })
    };
    let call = |op: &str, x: &str, y: &str, count: serde_json::Value| serde_json::json!({ "op": op, "args": [{ "buf": x }, { "buf": y }, count] });
    let rows_h = serde_json::json!({ "expr": { "mul": ["rows", H] } });
    let mut programs = serde_json::Map::new();
    let mut add = |name: &str, c: serde_json::Value| {
        programs.insert(name.into(), serde_json::json!({ "calls": [c.clone()] }));
        programs.insert(format!("{name}_burst"), serde_json::json!({ "calls": vec![c; burst] }));
    };
    add("a2a_bf16", call("nccl_alltoall_bf16", "xb", "yb", serde_json::json!({ "var": "cb" })));
    add("a2a_f32", call("nccl_alltoall_f32", "xf", "yf", serde_json::json!({ "var": "cf" })));
    add("ar_nccl_f32", call("nccl_allreduce_f32", "arx", "ary", rows_h.clone()));
    add("ar_nccl_bf16", call("nccl_allreduce_bf16", "arbx", "arby", rows_h));
    let mut buffers = serde_json::json!({
        "xb": { "dtype": "bf16", "shape": [world, "cb"], "kind": "input" },
        "yb": { "dtype": "bf16", "shape": [world, "cb"], "kind": "output" },
        "xf": { "dtype": "f32", "shape": [world, "cf"], "kind": "input" },
        "yf": { "dtype": "f32", "shape": [world, "cf"], "kind": "output" },
        "arx": { "dtype": "f32", "shape": ["rows", H], "kind": "input" },
        "ary": { "dtype": "f32", "shape": ["rows", H], "kind": "output" },
        "arbx": { "dtype": "bf16", "shape": ["rows", H], "kind": "input" },
        "arby": { "dtype": "bf16", "shape": ["rows", H], "kind": "output" }
    });
    let mut ops = serde_json::json!({
        "nccl_alltoall_bf16": nccl("alltoall", "bf16"),
        "nccl_alltoall_f32": nccl("alltoall", "f32"),
        "nccl_allreduce_f32": nccl("allreduce", "f32"),
        "nccl_allreduce_bf16": nccl("allreduce", "bf16")
    });
    let mut modules = serde_json::json!({});
    if let Some(sha) = peer {
        let stage = world * R_MAX * H * 4;
        let b = buffers.as_object_mut().unwrap();
        for (k, v) in [
            (
                "tr_comm",
                serde_json::json!({ "dtype": "u8", "shape": [2 * world * R_MAX * H * 4], "kind": "carry", "export": true }),
            ),
            (
                "tr_comm_peers",
                serde_json::json!({ "dtype": "u64", "shape": [world], "kind": "peer", "of": "tr_comm", "group": "w" }),
            ),
            (
                "tr_flags",
                serde_json::json!({ "dtype": "i32", "shape": [world * 256], "kind": "carry", "export": true }),
            ),
            (
                "tr_flag_peers",
                serde_json::json!({ "dtype": "u64", "shape": [world], "kind": "peer", "of": "tr_flags", "group": "w" }),
            ),
            ("tr_lamport", serde_json::json!({ "dtype": "u8", "shape": [3 * stage], "kind": "carry", "export": true })),
            (
                "tr_lamport_peers",
                serde_json::json!({ "dtype": "u64", "shape": [world], "kind": "peer", "of": "tr_lamport", "group": "w" }),
            ),
            ("tr_state", serde_json::json!({ "dtype": "i32", "shape": [8], "kind": "carry" })),
            ("tr_err", serde_json::json!({ "dtype": "i32", "shape": [1], "kind": "output" })),
            ("tr_blocks", serde_json::json!({ "dtype": "i32", "shape": [world + 1], "kind": "input" })),
        ] {
            b.insert(k.into(), v);
        }
        modules = serde_json::json!({ "peer_allreduce": { "source": "peer_allreduce.cubin", "sha256": sha } });
        let o = ops.as_object_mut().unwrap();
        o.insert("allreduce".into(), serde_json::json!({
            "params": ["in buffer<f32>", "out buffer<f32>", "inout buffer<u8>", "in buffer<u64>", "inout buffer<i32>",
                       "in buffer<u64>", "inout buffer<u8>", "in buffer<u64>", "inout buffer<i32>", "out buffer<i32>",
                       "in buffer<i32>", "i32", "i32", "i32", "i64", "i32", "i64"],
            "impl": { "launches": [{ "module": "peer_allreduce", "entry": "kern_peer_allreduce_f32",
                                     "block": [H / 4 / 8, 1, 1], "grid": [152, 1, 1], "cluster": [8, 1, 1] }] }
        }));
        o.insert(
            "lamport_init".into(),
            serde_json::json!({
                "params": ["inout buffer<u8>", "i64"],
                "impl": { "launches": [{ "module": "peer_allreduce", "entry": "kern_peer_lamport_init",
                                         "block": [256, 1, 1], "grid": [256, 1, 1] }] }
            }),
        );
        let tr = |mode: i32| {
            serde_json::json!({ "op": "allreduce", "args": [
                { "buf": "arx" }, { "buf": "ary" }, { "buf": "tr_comm" }, { "buf": "tr_comm_peers" },
                { "buf": "tr_flags" }, { "buf": "tr_flag_peers" }, { "buf": "tr_lamport" }, { "buf": "tr_lamport_peers" },
                { "buf": "tr_state" }, { "buf": "tr_err" }, { "buf": "tr_blocks" }, { "rank": "w" }, { "var": "rows" },
                { "i32": H }, { "i64": stage }, { "i32": mode }, { "i64": TIMEOUT_NS }
            ]})
        };
        add("ar_oneshot", tr(1));
        add("ar_twoshot", tr(2));
        add(
            "tr_init",
            serde_json::json!({ "op": "lamport_init", "args": [{ "buf": "tr_lamport" }, { "i64": 3 * stage }] }),
        );
    }
    serde_json::json!({
        "schema_version": 7,
        "model": "cross-collective",
        "vars": {
            "cb": { "max": A2A_MAX / world / 2 },
            "cf": { "max": A2A_MAX / world / 4 },
            "rows": { "max": R_MAX }
        },
        "topology": { "groups": { "w": world } },
        "buffers": buffers,
        "modules": modules,
        "ops": ops,
        "programs": programs
    })
}

fn write_frame(w: &mut impl Write, b: &[u8]) -> std::io::Result<()> {
    w.write_all(&(b.len() as u32).to_le_bytes())?;
    w.write_all(b)
}

fn read_frame(r: &mut impl Read) -> std::io::Result<Vec<u8>> {
    let mut n = [0u8; 4];
    r.read_exact(&mut n)?;
    let mut b = vec![0u8; u32::from_le_bytes(n) as usize];
    r.read_exact(&mut b)?;
    Ok(b)
}

/// Every rank's offer in rank order: rank 0 collects them and sends the
/// whole list back.
fn rendezvous(addr: &str, rank: usize, world: usize, mine: Vec<u8>) -> std::io::Result<Vec<Vec<u8>>> {
    if rank == 0 {
        let port = addr.rsplit(':').next().unwrap();
        let listener = TcpListener::bind(format!("0.0.0.0:{port}"))?;
        let mut offers = vec![Vec::new(); world];
        offers[0] = mine;
        let mut streams = Vec::new();
        while streams.len() + 1 < world {
            let (mut s, _) = listener.accept()?;
            let r = u32::from_le_bytes(read_frame(&mut s)?.try_into().unwrap()) as usize;
            offers[r] = read_frame(&mut s)?;
            streams.push(s);
        }
        for s in &mut streams {
            offers.iter().try_for_each(|o| write_frame(s, o))?;
        }
        return Ok(offers);
    }
    let mut s = (0..600)
        .find_map(|_| {
            TcpStream::connect(addr).map_err(|_| std::thread::sleep(std::time::Duration::from_millis(500))).ok()
        })
        .ok_or_else(|| std::io::Error::other(format!("rendezvous {addr}: no listener after 5 minutes")))?;
    write_frame(&mut s, &(rank as u32).to_le_bytes())?;
    write_frame(&mut s, &mine)?;
    (0..world).map(|_| read_frame(&mut s)).collect()
}

struct A2a {
    name: &'static str,
    var: &'static str,
    x: &'static str,
    y: &'static str,
    esize: usize,
}

const A2AS: [A2a; 2] = [
    A2a { name: "a2a_bf16", var: "cb", x: "xb", y: "yb", esize: 2 },
    A2a { name: "a2a_f32", var: "cf", x: "xf", y: "yf", esize: 4 },
];

/// One all-to-all at `round`: a fresh pattern in, every received block
/// checked against the word its sender wrote for this rank.
fn a2a_round(
    rt: &mut Runtime,
    a: &A2a,
    rank: usize,
    world: usize,
    n: usize,
    round: usize,
    graph: bool,
) -> Result<(), String> {
    let e = |e: kern_runtime::Error| e.to_string();
    let v = vars(&[(a.var, n)]);
    let x: Vec<u8> = (0..world).flat_map(|p| a2a_bytes(a.esize, rank, p, n, round)).collect();
    rt.write_input_at(a.x, &x, &v).map_err(e)?;
    if graph { rt.run_captured(a.name, &v) } else { rt.run(a.name, &v) }.map_err(e)?;
    let y = rt.read_output(a.y).map_err(e)?;
    let block = n * a.esize;
    match (0..world).find(|&p| y[p * block..][..block] != a2a_bytes(a.esize, p, rank, n, round)[..]) {
        Some(p) => Err(format!(
            "{} {} B per peer, {} round {round}: block from rank {p} wrong",
            a.name,
            block,
            if graph { "graph" } else { "eager" }
        )),
        None => Ok(()),
    }
}

/// One allreduce of `prog` at `round`, every element checked. A peer
/// allreduce keeps its rows "own rows first": local row `j` of rank `r` is
/// group row `(blocks[r] + j) mod rows`, so input and output rotate by it.
fn ar_check(
    rt: &mut Runtime,
    prog: &str,
    peer: bool,
    rank: usize,
    world: usize,
    rows: usize,
    round: usize,
) -> Result<(), String> {
    let e = |e: kern_runtime::Error| e.to_string();
    let v = vars(&[("rows", rows)]);
    let bf16 = prog.ends_with("bf16");
    let bl = blocks(world, rows);
    let shift = if peer { bl[rank] } else { 0 };
    let elem = |i: usize| ((i / H + shift) % rows) * H + i % H;
    let (x, y) = if bf16 { ("arbx", "arby") } else { ("arx", "ary") };
    let input: Vec<u8> = match bf16 {
        true => (0..rows * H)
            .flat_map(|i| ((ar_value(rank, elem(i), round).to_bits() >> 16) as u16).to_le_bytes())
            .collect(),
        false => (0..rows * H).flat_map(|i| ar_value(rank, elem(i), round).to_le_bytes()).collect(),
    };
    if peer {
        let b: Vec<u8> = bl.iter().flat_map(|&b| (b as i32).to_le_bytes()).collect();
        rt.write_input_at("tr_blocks", &b, &v).map_err(e)?;
    }
    rt.write_input_at(x, &input, &v).map_err(e)?;
    rt.run(prog, &v).map_err(e)?;
    let out = rt.read_output(y).map_err(e)?;
    let got = |i: usize| match bf16 {
        true => f32::from_bits((u16::from_le_bytes([out[2 * i], out[2 * i + 1]]) as u32) << 16),
        false => f32::from_le_bytes(out[4 * i..][..4].try_into().unwrap()),
    };
    let bad = (0..rows * H).filter(|&i| got(i) != (0..world).map(|r| ar_value(r, elem(i), round)).sum::<f32>()).count();
    if bad != 0 {
        return Err(format!("{prog} rows={rows}: {bad} wrong elements"));
    }
    if peer {
        let err = i32::from_le_bytes(rt.read_output("tr_err").map_err(e)?[..4].try_into().unwrap());
        if err != 0 {
            return Err(format!("{prog} rows={rows}: rank {} never arrived", err - 1));
        }
    }
    Ok(())
}

/// Microseconds per call of `prog`, timed as `iters` replays of its
/// captured burst.
fn burst_us(rt: &mut Runtime, prog: &str, v: &Vars, iters: usize, burst: usize) -> Result<f64, String> {
    let e = |e: kern_runtime::Error| e.to_string();
    let b = format!("{prog}_burst");
    rt.capture(&b, v).map_err(e)?;
    rt.run_captured(&b, v).map_err(e)?;
    Ok(rt.time_captured(&b, v, iters).map_err(e)? as f64 * 1e3 / burst as f64)
}

fn main() {
    let (mut rank, mut world, mut gpu) = (0usize, 0usize, 0usize);
    let mut addr = String::new();
    let mut ar_cubin: Option<PathBuf> = None;
    let (mut iters, mut burst, mut capture_first) = (100usize, 16usize, false);
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut v = || args.next().expect("value");
        match a.as_str() {
            "--rank" => rank = v().parse().unwrap(),
            "--world" => world = v().parse().unwrap(),
            "--gpu" => gpu = v().parse().unwrap(),
            "--rendezvous" => addr = v(),
            "--ar-cubin" => ar_cubin = Some(PathBuf::from(v())),
            "--iters" => iters = v().parse().unwrap(),
            "--burst" => burst = v().parse().unwrap(),
            "--capture-first" => capture_first = true,
            _ => panic!("unknown arg {a}"),
        }
    }
    assert!(world > 1 && rank < world && !addr.is_empty(), "--rank, --world and --rendezvous are required");
    let kernels = std::env::temp_dir().join(format!("kern-cross-collective-{}", std::process::id()));
    std::fs::create_dir_all(&kernels).unwrap();
    let sha = ar_cubin.as_ref().map(|p| {
        let bytes = std::fs::read(p).expect("allreduce cubin");
        let sha = hex::encode(<sha2::Sha256 as sha2::Digest>::digest(&bytes));
        std::fs::write(kernels.join(format!("peer_allreduce-{}.cubin", &sha[..12])), &bytes).unwrap();
        sha
    });
    let json = serde_json::to_string(&manifest(world, sha.as_deref(), burst)).unwrap();
    let verified = kern_manifest::Verified::from_json(&json).unwrap_or_else(|e| panic!("{e}"));

    let run = || -> Result<Vec<String>, String> {
        let e = |e: kern_runtime::Error| e.to_string();
        let topo = Topology::one("w", rank as u64, world as u64);
        let mut rt = Runtime::load(
            &verified,
            Some(&kernels),
            gpu,
            Some(Capacity { tokens: Some(1), seqs: 1, landings: 0 }),
            Some(&topo),
        )
        .map_err(e)?;
        let offers = rendezvous(&addr, rank, world, rt.offer().and_then(|o| o.to_bytes()).map_err(e)?)
            .map_err(|e| e.to_string())?
            .iter()
            .map(|b| Offer::from_bytes(b))
            .collect::<kern_runtime::Result<Vec<_>>>()
            .map_err(e)?;
        rt.connect(&BTreeMap::from([("w".to_string(), offers)])).map_err(e)?;
        let mut lines = vec![format!(
            "world {world}, NCCL_PROTO={}, NCCL_P2P_LL_THRESHOLD={}",
            std::env::var("NCCL_PROTO").unwrap_or_default(),
            std::env::var("NCCL_P2P_LL_THRESHOLD").unwrap_or_else(|_| "default".into())
        )];

        let per_rank = [0.3, 0.5, 1.0, 2.0, 4.0, 8.0];
        let count = |a: &A2a, mib: f64| ((mib * MIB) as usize / world / a.esize) / 64 * 64;
        if capture_first {
            let a = &A2AS[0];
            let n = count(a, per_rank[0]);
            rt.capture(a.name, &vars(&[(a.var, n)])).map_err(e)?;
            a2a_round(&mut rt, a, rank, world, n, 0, true)?;
            lines.push("capture before any eager all-to-all: exact".into());
        }
        lines.push(format!("all-to-all: {iters} eager + {iters} graph rounds per size, every byte checked"));
        lines.push("  type  MiB/rank  B/peer      us    GB/s (sent off-rank)".into());
        for a in &A2AS {
            for &mib in &per_rank {
                let n = count(a, mib);
                (0..iters).try_for_each(|r| a2a_round(&mut rt, a, rank, world, n, r, false))?;
                rt.capture(a.name, &vars(&[(a.var, n)])).map_err(e)?;
                (0..iters).try_for_each(|r| a2a_round(&mut rt, a, rank, world, n, iters + r, true))?;
                let us = burst_us(&mut rt, a.name, &vars(&[(a.var, n)]), 50, burst)?;
                let off = (world - 1) * n * a.esize;
                lines.push(format!(
                    "  {:<5} {:>8.2} {:>7} {:>7.1} {:>7.1}",
                    &a.name[4..],
                    (world * n * a.esize) as f64 / MIB,
                    n * a.esize,
                    us,
                    off as f64 / us / 1e3
                ));
            }
        }

        if sha.is_some() {
            rt.run("tr_init", &vars(&[])).map_err(e)?;
        }
        lines.push("allreduce, every size checked once, us per call (captured burst):".into());
        lines.push("  rows   bytes f32 | nccl f32  nccl bf16  one-shot  two-shot".into());
        let peer: &[&str] = if sha.is_some() { &["ar_oneshot", "ar_twoshot"] } else { &[] };
        for (round, rows) in [1usize, 2, 4, 8, 16, 24, 32, 36].into_iter().enumerate() {
            let v = vars(&[("rows", rows)]);
            ar_check(&mut rt, "ar_nccl_f32", false, rank, world, rows, round)?;
            ar_check(&mut rt, "ar_nccl_bf16", false, rank, world, rows, round)?;
            let mut cells = vec![
                burst_us(&mut rt, "ar_nccl_f32", &v, 50, burst)?,
                burst_us(&mut rt, "ar_nccl_bf16", &v, 50, burst)?,
            ];
            for p in peer {
                ar_check(&mut rt, p, true, rank, world, rows, round)?;
                cells.push(burst_us(&mut rt, p, &v, 50, burst)?);
                ar_check(&mut rt, p, true, rank, world, rows, round + 100)?;
            }
            let cells: Vec<String> = cells.iter().map(|c| format!("{c:>9.1}")).collect();
            lines.push(format!("  {rows:>4} {:>11} | {}", rows * H * 4, cells.join(" ")));
        }
        Ok(lines)
    };
    match run() {
        Ok(lines) => {
            if rank == 0 {
                lines.iter().for_each(|l| println!("{l}"));
            }
            println!("rank {rank} gpu {gpu}: PASS");
        }
        Err(err) => {
            println!("rank {rank} gpu {gpu}: FAIL {err}");
            std::process::exit(1);
        }
    }
}
