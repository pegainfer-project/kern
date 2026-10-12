//! `--rows`: a sequence's rows dealt by position over a group of `w`
//! members, each member its own runtime and pool, pushed one member and
//! one state per [`Runtime::push_rows`]. The states are `--layers` paged
//! states of `--row` bytes per token over 64-row pages, a manifest made
//! here (nothing runs but the push). The source fills its pages with
//! bytes keyed by (state, position, word); every member reads its pages
//! back and finds position `q·w + m` at local position `q`, and the rows
//! of its last page past its share as they were before the push.
//!
//! Cases: w = 1, 3, 8 over scattered source pages (runs that never fold),
//! w = 8 over adjacent ones (one long run per member), lengths that leave
//! the last page partial on both sides. `--bench` instead pushes a
//! `--tokens` prompt (219000) over eight members, adjacent and scattered,
//! `--iters` times each, and reports host issue time and time to landed.
//!
//!   push_states --rows [--bench] [--gpu G] [--peer-gpu H]            one process
//!   push_states --rows [--bench] --cross [--gpu G] [--peer-gpu H]    the group in a child process
//!   push_states --rows [--bench] --destination DIR [--gpu H]         the group, another host
//!   push_states --rows [--bench] --source DIR [--gpu G]              the source, against it
//!
//! `DIR` is a directory both hosts see; both sides take the same case
//! flags.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use kern_manifest::Verified;
use kern_pool::{deal, Deal, Interleave, Lease, Pages, Placement};
use kern_runtime::{Capacity, MappedPool, PoolHandle, Rows, Runtime};

use crate::{appear, publish};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

const UNIT: u64 = 64;

/// What every case shares: the states' shape.
#[derive(Clone, Copy)]
struct Shape {
    layers: usize,
    row: u64,
}

/// One push: `len` positions over `w` members, the source's pages in
/// lease order or scattered.
#[derive(Debug, Clone, Copy)]
struct Case {
    w: u64,
    len: u64,
    scattered: bool,
}

fn cases(bench: bool, tokens: u64) -> Vec<Case> {
    let c = |w, len, scattered| Case { w, len, scattered };
    if bench {
        vec![c(8, tokens, false), c(8, tokens, true)]
    } else {
        vec![c(1, 300, true), c(3, 1001, true), c(8, 4999, true), c(8, 4999, false)]
    }
}

fn state(l: usize) -> String {
    format!("kv.{l}")
}

/// `layers` paged states of `row` bytes per token, one 64-row page table
/// wide enough for `tokens`, and a program that names them all.
fn manifest(s: Shape, tokens: u64) -> Res<Verified> {
    let names: Vec<String> = (0..s.layers).map(state).collect();
    let states: serde_json::Map<String, serde_json::Value> =
        names.iter().map(|n| (n.clone(), serde_json::json!({"bytes_per_token": s.row}))).collect();
    let square = |kind: &str| serde_json::json!({"kind": kind, "dtype": "bf16", "shape": [16, 16]});
    let params: Vec<&str> = ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "in buffer<i32>"]
        .into_iter()
        .chain(names.iter().map(|_| "inout state"))
        .collect();
    let args: Vec<serde_json::Value> = ["a", "b", "c", "table"]
        .iter()
        .map(|b| serde_json::json!({"buf": b}))
        .chain(names.iter().map(|n| serde_json::json!({"state": n})))
        .collect();
    let m = serde_json::json!({
        "schema_version": 7, "model": "rows", "vars": {}, "states": states,
        "buffers": {
            "a": square("input"), "b": square("input"), "c": square("output"),
            "table": {"kind": "input", "dtype": "i32", "shape": [1, tokens.div_ceil(UNIT)],
                      "domain": {"index_into": names[0], "stride": UNIT}}
        },
        "modules": {},
        "ops": {"touch": {"params": params, "impl": {"launches": [{
            "entry": "extern:cublaslt_bf16_tn",
            "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i32"],
            "args": [{"param": 0}, {"param": 1}, {"param": 2}, {"i32": 16}, {"i32": 16}, {"i32": 16}]
        }]}}},
        "programs": {"step": {"calls": [{"op": "touch", "args": args}]}}
    });
    Ok(Verified::from_json(&m.to_string()).map_err(|e| e.to_string())?)
}

fn mix(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// The bytes of position `p` of state `l`.
fn pattern(l: usize, p: u64, row: u64) -> Vec<u8> {
    (0..row / 8).flat_map(|i| mix(((l as u64) << 56) ^ (p << 20) ^ i).to_le_bytes()).collect()
}

/// A Fisher–Yates shuffle under a fixed xorshift: the same order every run.
fn shuffle(ids: &mut [i32]) {
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    for i in (1..ids.len()).rev() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        ids.swap(i, (x % (i as u64 + 1)) as usize);
    }
}

pub fn run(args: Vec<String>) -> Res<()> {
    let (mut gpu, mut peer_gpu) = (0usize, None);
    let (mut bench, mut cross) = (false, false);
    let (mut tokens, mut iters) = (219_000u64, 20usize);
    let mut shape = Shape { layers: 3, row: 1152 };
    let (mut source, mut destination) = (None::<PathBuf>, None::<PathBuf>);
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        let mut v = || it.next().ok_or(format!("{a} takes a value"));
        match a.as_str() {
            "--rows" => {}
            "--gpu" => gpu = v()?.parse()?,
            "--peer-gpu" => peer_gpu = Some(v()?.parse()?),
            "--bench" => bench = true,
            "--cross" => cross = true,
            "--tokens" => tokens = v()?.parse()?,
            "--iters" => iters = v()?.parse()?,
            "--layers" => shape.layers = v()?.parse()?,
            "--row" => shape.row = v()?.parse()?,
            "--source" => source = Some(PathBuf::from(v()?)),
            "--destination" => destination = Some(PathBuf::from(v()?)),
            _ => return Err(format!("unknown arg {a}").into()),
        }
    }
    let cases = cases(bench, tokens);
    let iters = if bench { iters } else { 1 };
    let peer_gpu = peer_gpu.unwrap_or(gpu);
    let flags: Vec<String> = [
        bench.then(|| vec!["--bench".to_string()]),
        Some(vec!["--tokens".into(), tokens.to_string(), "--layers".into(), shape.layers.to_string()]),
        Some(vec!["--row".into(), shape.row.to_string()]),
    ]
    .into_iter()
    .flatten()
    .flatten()
    .collect();
    println!("rows: {} states × {} B/row, {UNIT}-row pages; cases {:?}", shape.layers, shape.row, cases);
    if let Some(dir) = destination {
        return be_group(shape, &cases, gpu, &dir);
    }
    let src_tokens = cases.iter().map(|c| c.len).max().unwrap_or(0) + UNIT;
    let mut a = Runtime::load(&manifest(shape, src_tokens)?, None, gpu, Some(cap(src_tokens)), None)?;
    let mut group = if let Some(dir) = source {
        Group::There { dir, child: None }
    } else if cross {
        let dir = std::env::temp_dir().join(format!("kern-push-rows-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let child = std::process::Command::new(std::env::current_exe()?)
            .arg("--rows")
            .args(&flags)
            .args(["--gpu".to_string(), peer_gpu.to_string(), "--destination".into(), dir.display().to_string()])
            .spawn()?;
        Group::There { dir, child: Some(child) }
    } else {
        Group::Here { members: members(shape, &cases, peer_gpu)?, seats: Vec::new() }
    };
    let mut mirrors: Vec<MappedPool> = Vec::new();
    for (m, h) in group.handles(cases.iter().map(|c| c.w).max().unwrap_or(1))?.iter().enumerate() {
        mirrors.push(a.map_pool(h, &format!("member {m}"))?);
    }
    println!("source on GPU {gpu}: {} members mapped", mirrors.len());
    for (i, c) in cases.iter().enumerate() {
        push_case(&mut a, shape, i, *c, iters, &mut group, &mut mirrors)?;
    }
    group.finish()?;
    println!("PASS");
    Ok(())
}

fn cap(tokens: u64) -> Capacity {
    Capacity { tokens: Some(tokens), seqs: 1, landings: 0 }
}

/// Member runtimes for the largest group any case deals over, each with
/// room for the largest share.
fn members(s: Shape, cases: &[Case], gpu: usize) -> Res<Vec<Runtime>> {
    let w = cases.iter().map(|c| c.w).max().unwrap_or(1);
    let share = cases.iter().map(|c| c.len.div_ceil(c.w)).max().unwrap_or(0) + UNIT;
    let m = manifest(s, share)?;
    Ok((0..w).map(|_| Runtime::load(&m, None, gpu, Some(cap(share)), None)).collect::<Result<_, _>>()?)
}

/// A member's lease for a case, and the bytes past its share in its last
/// page per state, as they were before the push.
struct Seat {
    lease: Lease,
    tails: Vec<Vec<u8>>,
}

fn seat(rt: &mut Runtime, s: Shape, c: Case, m: u64) -> Res<Seat> {
    let n = Interleave::new(c.w)?.len(c.len, m);
    let lease = rt.lease(n as usize)?;
    rt.synchronize()?;
    let tails = (0..s.layers).map(|l| tail(rt, s, &lease, n, l)).collect::<Res<Vec<_>>>()?;
    Ok(Seat { lease, tails })
}

fn tail(rt: &Runtime, s: Shape, lease: &Lease, n: u64, l: usize) -> Res<Vec<u8>> {
    let Some(&last) = lease.page_ids().last() else { return Ok(Vec::new()) };
    let at = (last as u64 * UNIT + n % UNIT) * s.row;
    let rows = if n.is_multiple_of(UNIT) { 0 } else { UNIT - n % UNIT };
    Ok(rt.read_state_at(&state(l), at as usize, (rows * s.row) as usize)?)
}

/// Member `m`'s pages hold its positions of the case, and nothing past them moved.
fn check(rt: &Runtime, s: Shape, c: Case, m: u64, seat: &Seat) -> Res<u64> {
    let n = Interleave::new(c.w)?.len(c.len, m);
    let ids = seat.lease.page_ids();
    for l in 0..s.layers {
        for (k, &page) in ids.iter().enumerate() {
            let rows = (n - k as u64 * UNIT).min(UNIT);
            let got = rt.read_state_at(&state(l), (page as u64 * UNIT * s.row) as usize, (rows * s.row) as usize)?;
            for (i, r) in got.chunks_exact(s.row as usize).enumerate() {
                let q = k as u64 * UNIT + i as u64;
                if r != pattern(l, q * c.w + m, s.row) {
                    return Err(format!(
                        "{c:?} member {m} state `{}`: local row {q} is not position {}",
                        state(l),
                        q * c.w + m
                    )
                    .into());
                }
            }
        }
        if tail(rt, s, &seat.lease, n, l)? != seat.tails[l] {
            return Err(format!("{c:?} member {m} state `{}`: the rows past its share moved", state(l)).into());
        }
    }
    Ok(n * s.row * s.layers as u64)
}

/// The group as the source sees it: runtimes in this process, or a
/// process that answers through files in `dir`.
enum Group {
    Here { members: Vec<Runtime>, seats: Vec<Seat> },
    There { dir: PathBuf, child: Option<std::process::Child> },
}

impl Group {
    fn handles(&self, w: u64) -> Res<Vec<PoolHandle>> {
        match self {
            Group::Here { members, .. } => Ok(members.iter().map(Runtime::export_pool).collect::<Result<_, _>>()?),
            Group::There { dir, .. } => (0..w)
                .map(|m| {
                    PoolHandle::from_bytes(&appear(&dir.join(format!("pool.{m}")))?).ok_or("not a pool handle".into())
                })
                .collect(),
        }
    }

    /// Every member's pages and placement for case `i`, once it has leased them.
    fn seat(&mut self, s: Shape, i: usize, c: Case) -> Res<Vec<(Vec<i32>, Placement)>> {
        match self {
            Group::Here { members, seats } => {
                *seats = (0..c.w).map(|m| seat(&mut members[m as usize], s, c, m)).collect::<Res<_>>()?;
                Ok(seats
                    .iter()
                    .zip(members.iter())
                    .map(|(s, rt)| (s.lease.page_ids().to_vec(), rt.placement()))
                    .collect())
            }
            Group::There { dir, .. } => (0..c.w)
                .map(|m| {
                    let ids = appear(&dir.join(format!("c{i}.ids.{m}")))?;
                    let p = Placement::from_bytes(&appear(&dir.join(format!("c{i}.placement.{m}")))?)
                        .ok_or("not a placement")?;
                    Ok((ids.as_chunks::<4>().0.iter().map(|&b| i32::from_le_bytes(b)).collect(), p))
                })
                .collect(),
        }
    }

    /// The members' verdict on case `i` once the source's pushes have landed.
    fn check(&mut self, s: Shape, i: usize, c: Case) -> Res<String> {
        match self {
            Group::Here { members, seats } => {
                let bytes: u64 =
                    (0..c.w).map(|m| check(&members[m as usize], s, c, m, &seats[m as usize])).sum::<Res<u64>>()?;
                seats.clear();
                Ok(format!("{bytes} bytes identical"))
            }
            Group::There { dir, .. } => {
                publish(&dir.join(format!("c{i}.pushed")), b"")?;
                let t = Instant::now();
                loop {
                    if let Ok(b) = std::fs::read(dir.join(format!("c{i}.checked"))) {
                        return Ok(String::from_utf8(b)?);
                    }
                    if let Ok(b) = std::fs::read(dir.join(format!("c{i}.failed"))) {
                        return Err(String::from_utf8(b)?.into());
                    }
                    if t.elapsed() > Duration::from_secs(600) {
                        return Err(format!("case {i} never checked").into());
                    }
                    std::thread::sleep(Duration::from_millis(5));
                }
            }
        }
    }

    fn finish(self) -> Res<()> {
        if let Group::There { dir, child: Some(mut child) } = self {
            let status = child.wait()?;
            std::fs::remove_dir_all(&dir)?;
            if !status.success() {
                return Err(format!("group process: {status}").into());
            }
        }
        Ok(())
    }
}

/// The group process: load the members, publish their pools, and for
/// every case lease, publish, wait for the push, check.
fn be_group(s: Shape, cases: &[Case], gpu: usize, dir: &Path) -> Res<()> {
    std::fs::create_dir_all(dir)?;
    if std::fs::read_dir(dir)?.next().is_some() {
        return Err(format!("{} is not empty: the source would read another run's files", dir.display()).into());
    }
    let mut members = members(s, cases, gpu)?;
    for (m, rt) in members.iter().enumerate() {
        let h = rt.export_pool()?.to_bytes().ok_or("local handles do not cross a process")?;
        publish(&dir.join(format!("pool.{m}")), &h)?;
    }
    for (i, &c) in cases.iter().enumerate() {
        let seats = (0..c.w).map(|m| seat(&mut members[m as usize], s, c, m)).collect::<Res<Vec<_>>>()?;
        for (m, (st, rt)) in seats.iter().zip(&members).enumerate() {
            let ids: Vec<u8> = st.lease.page_ids().iter().flat_map(|i| i.to_le_bytes()).collect();
            publish(&dir.join(format!("c{i}.ids.{m}")), &ids)?;
            publish(&dir.join(format!("c{i}.placement.{m}")), &rt.placement().to_bytes())?;
        }
        appear(&dir.join(format!("c{i}.pushed")))?;
        let verdict = (0..c.w).map(|m| check(&members[m as usize], s, c, m, &seats[m as usize])).sum::<Res<u64>>();
        match verdict {
            Ok(bytes) => {
                let line = format!("{bytes} bytes identical on GPU {gpu} of the group process");
                println!("group: {c:?}: {line}");
                publish(&dir.join(format!("c{i}.checked")), line.as_bytes())?;
            }
            Err(e) => {
                publish(&dir.join(format!("c{i}.failed")), e.to_string().as_bytes())?;
                return Err(e);
            }
        }
    }
    Ok(())
}

/// Lease and fill the source sequence, deal it to every member, push
/// every state to every member `iters` times, and have the group check.
fn push_case(
    a: &mut Runtime,
    s: Shape,
    i: usize,
    c: Case,
    iters: usize,
    group: &mut Group,
    mirrors: &mut [MappedPool],
) -> Res<()> {
    let lease = a.lease(c.len as usize)?;
    let mut src = lease.page_ids().to_vec();
    if c.scattered {
        shuffle(&mut src);
    }
    for l in 0..s.layers {
        for (k, &page) in src.iter().enumerate() {
            let rows: Vec<u8> = (0..UNIT).flat_map(|r| pattern(l, k as u64 * UNIT + r, s.row)).collect();
            a.write_state_at(&state(l), (page as u64 * UNIT * s.row) as usize, &rows)?;
        }
    }
    let seats = group.seat(s, i, c)?;
    for ((_, p), mp) in seats.iter().zip(mirrors.iter_mut()) {
        a.place(mp, p)?;
    }
    let il = Interleave::new(c.w)?;
    let t = Instant::now();
    let deals: Vec<Deal> = seats
        .iter()
        .enumerate()
        .map(|(m, (ids, _))| deal(Pages { ids: &src, unit: UNIT }, Pages { ids, unit: UNIT }, il, m as u64, c.len))
        .collect::<Result<_, _>>()?;
    let planned = t.elapsed();
    let runs: usize = deals.iter().map(|d| d.runs().len()).sum();
    let pieces: usize = deals.iter().map(|d| Rows::dealt(d, s.row).len()).sum();
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        a.synchronize()?;
        let t = Instant::now();
        for l in 0..s.layers {
            for (d, mp) in deals.iter().zip(mirrors.iter()) {
                a.push_rows(&state(l), d, mp)?;
            }
        }
        let issued = t.elapsed();
        a.mark()?.wait()?;
        times.push((issued, t.elapsed()));
        a.fault()?;
    }
    let verdict = group.check(s, i, c)?;
    let bytes = c.len * s.row * s.layers as u64;
    times.sort_by_key(|&(_, landed)| landed);
    let (issued, landed) = times[times.len() / 2];
    let (_, best) = times[0];
    println!(
        "case {i} {c:?}: {runs} runs, {pieces} pieces per state, deals planned in {planned:?}; \
         {} pushes of {bytes} B: issued in {issued:?}, landed in {landed:?} (median of {iters}, best {best:?}), \
         {:.0} GB/s; {verdict}",
        s.layers * c.w as usize,
        bytes as f64 / landed.as_secs_f64() / 1e9
    );
    Ok(())
}
