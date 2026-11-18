//! How long a pipeline head takes to size one item, on one core: headway's
//! four rungs over queues from an idle head to a deep backlog.
//!
//! ```text
//! cargo bench -p kern-serve --bench pp_decide [-- cost=<costfit>] [context=<rows>] [waiting=<n>,...]
//! ```
//!
//! Each row is one queue: its waiting prompts, their rows to go, the cap
//! each rung chose and the median time of that decision. The queue is
//! drawn from a fixed seed (half the prompts cold, half behind a kept
//! prefix of up to half the context, passed over by up to twice the
//! stages) and the pipeline is idle. The cost model is DeepSeek V4.1
//! Flash PP4's and the context 2^20 rows unless named; `waiting=` keeps
//! only the queues of that many prompts. Without `--bench` (as
//! `cargo test --benches` runs it) it sizes the two smallest queues once.

use std::collections::VecDeque;
use std::hint::black_box;
use std::time::{Duration, Instant};

use kern_serve::cost::Cost;
use kern_serve::frontier::{Pipe, Seq};
use kern_serve::policy::{Decision, Policy};

const AHEAD: usize = 4;
const QUEUES: [(usize, i64); 7] = [
    (2, 8_000),
    (8, 100_000),
    (32, 1_000_000),
    (80, 6_000_000),
    (256, 6_000_000),
    (1024, 24_000_000),
    (4096, 24_000_000),
];
/// An offered load on each of headway's rungs.
const LOADS: [f64; 4] = [0.3, 0.5, 0.7, 0.9];
const DSV41: &str = include_str!("../tests/fixtures/pp/dsv41-pp4-cost.json");

/// Uniform values in [0, 1) from `seed`.
fn uniform(seed: u64) -> impl Iterator<Item = f64> {
    std::iter::successors(Some(seed), |x| Some(x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)))
        .skip(1)
        .map(|x| (x >> 11) as f64 / (1u64 << 53) as f64)
}

/// `waiting` prompts with about `togo` rows to go between them, none over a
/// quarter of the context on average.
fn queue(waiting: usize, togo: i64, stages: usize, context: i64) -> Vec<Seq> {
    let mut u = uniform(waiting as u64);
    let mut next = || u.next().expect("the stream does not end");
    let mean = (togo / waiting as i64).min(context / 4) as f64;
    (0..waiting)
        .map(|i| {
            let left = ((mean * (0.25 + 1.5 * next())) as i64).max(64);
            let hit = if next() < 0.5 { 0 } else { (next() * (context - left) as f64 / 2.0) as i64 };
            let passed = (next() * (2 * stages + 1) as f64) as u32;
            Seq::new(i, hit, hit + left, 0.0).at(hit, hit + left, passed)
        })
        .collect()
}

/// The cap `decide` returns and its median time over `budget`, µs.
fn time(decide: impl Fn() -> i64, budget: Duration) -> (i64, f64) {
    let (began, mut runs) = (Instant::now(), Vec::new());
    let cap = loop {
        let t = Instant::now();
        let cap = black_box(decide());
        runs.push(t.elapsed().as_secs_f64() * 1e6);
        if began.elapsed() >= budget || runs.len() == 2000 {
            break cap;
        }
    };
    runs.sort_by(f64::total_cmp);
    (cap, runs[runs.len() / 2])
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let named = |name: &str| args.iter().find_map(|a| a.strip_prefix(name)?.strip_prefix('='));
    let full = args.iter().any(|a| a == "--bench");
    let cost = match named("cost") {
        Some(path) => Cost::parse(&std::fs::read_to_string(path).expect("the costfit reads")),
        None => Cost::parse(DSV41),
    }
    .expect("the costfit parses");
    let context: i64 = named("context").map_or(1 << 20, |c| c.parse().expect("context is a row count"));
    let only: Vec<usize> = named("waiting")
        .map(|w| w.split(',').map(|n| n.parse().expect("waiting is prompt counts")).collect())
        .unwrap_or_default();
    let (stages, policy) = (cost.stages(), Policy::parse("headway10240/12288").expect("headway parses"));
    let (queues, budget) =
        if full { (&QUEUES[..], Duration::from_millis(400)) } else { (&QUEUES[..2], Duration::ZERO) };
    println!("{stages} stages, context {context}\n{:>7} {:>10}   cap and µs on each rung", "waiting", "rows");
    for &(waiting, togo) in queues.iter().filter(|q| only.is_empty() || only.contains(&q.0)) {
        let (seqs, recent) = (queue(waiting, togo, stages, context), VecDeque::new());
        let decision = |load| Decision {
            now: 0.0,
            active: &seqs,
            frontier: Pipe::idle(stages, 0.0),
            answers: Vec::new(),
            model: &cost,
            stages,
            ahead: AHEAD,
            context,
            recent: &recent,
            rate: 5e-6,
            load,
            util: 0.9,
        };
        let cells: Vec<String> = LOADS
            .iter()
            .map(|&load| {
                let d = decision(load);
                let (cap, us) = time(|| policy.cap(black_box(&d)), budget);
                format!("{cap:>6} {us:>10.1}")
            })
            .collect();
        let rows: i64 = seqs.iter().map(Seq::left).sum();
        println!("{waiting:>7} {rows:>10}   {}", cells.join("   "));
    }
}
