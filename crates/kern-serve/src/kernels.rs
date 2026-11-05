//! What each rank's device did, as four counters, and every kernel on disk.
//!
//! With `--trace` the process subscribes CUPTI's hardware event system
//! before any context exists ([`kern_runtime::trace`]), and a thread folds
//! what it collects into a [`Board`] every 200 ms. A step is named by its
//! program and shape: a graph program's var values (its graph key), an
//! eager one's rounded up to a power of two, so a prefill chunk's label
//! set stays small.
//!
//! `/kern/metrics` is four counters per rank, the health view, low enough
//! in cardinality to keep for months; no op, call or kernel name is a
//! label, those are what the files on disk are for.
//! `kern_device_seconds_total{class}` splits the device's wall clock into
//! compute, comm and idle: comm is a call that reads a peer buffer or runs
//! an NCCL collective, and includes waiting for the peers, so a slow rank
//! shows as compute on itself and comm on the rest; where both run, the
//! time is compute; idle is the clock no kernel covers, settled [`LAG`]
//! behind CUPTI's clock so the kernels that ran inside it have arrived.
//! `kern_step_compute_seconds_total` and `kern_steps_total` per (program,
//! shape, kv) give a step's compute cost for the same work, across ranks
//! and across days: `kv` is the step's `seq_len` sum rounded up to a power
//! of two, since a step that reads more context costs more without
//! anything being wrong. `kern_trace_dropped_records_total` is what the
//! counters miss: records CUPTI dropped, and kernels that arrived after
//! their time was settled.
//!
//! With `--trace-dir` every kernel is also kept on disk
//! ([`crate::recorder`]); `POST /kern/trace/rotate` closes the files being
//! written so the last minutes can be read without waiting for the
//! quarter hour. Both endpoints are served on `--trace-port`, on every
//! rank alike; without it a rank still traces and records.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use kern_manifest::types::{Arg, BufferKind, Call, Launch, Manifest};
use kern_runtime::trace::{self, Collected, Step};
use serde_json::json;
use tracing::{info, warn};

use crate::recorder::{Now, Recorder};

const EVERY: Duration = Duration::from_millis(200);
/// How far behind CUPTI's clock the device's time is settled.
const LAG: u64 = 2_000_000_000;

/// What the device ran, as intervals in nanoseconds: `true` for comm.
type Interval = (u64, u64, bool);

/// The device's wall clock, settled up to `from`.
#[derive(Debug, Default)]
struct Clock {
    from: Option<u64>,
    /// Intervals ending past `from`, clipped to start at it.
    open: Vec<Interval>,
    compute: f64,
    comm: f64,
    idle: f64,
    late: u64,
}

/// Nanoseconds the intervals cover together.
fn union(mut v: Vec<(u64, u64)>) -> u64 {
    v.sort_unstable();
    let (total, _) = v
        .into_iter()
        .fold((0u64, 0u64), |(total, reach), (a, b)| (total + b.saturating_sub(a.max(reach)), reach.max(b)));
    total
}

impl Clock {
    /// Takes the intervals that arrived and settles the clock up to
    /// `horizon`; an interval ending before the settled clock is late.
    fn advance(&mut self, arrived: Vec<Interval>, horizon: u64) {
        let from = *self.from.get_or_insert_with(|| arrived.iter().map(|i| i.0).min().unwrap_or(horizon).min(horizon));
        let (kept, late): (Vec<_>, Vec<_>) = arrived.into_iter().partition(|&(_, end, _)| end > from);
        self.late += late.len() as u64;
        self.open.extend(kept.into_iter().map(|(a, b, c)| (a.max(from), b, c)));
        if horizon <= from {
            return;
        }
        let inside = |comm: Option<bool>| -> u64 {
            let v = self.open.iter().filter(|i| comm.is_none_or(|c| i.2 == c)).map(|&(a, b, _)| (a, b.min(horizon)));
            union(v.filter(|(a, b)| a < b).collect())
        };
        let (all, compute) = (inside(None), inside(Some(false)));
        self.compute += secs(compute);
        self.comm += secs(all - compute);
        self.idle += secs((horizon - from).saturating_sub(all));
        self.open.retain(|i| i.1 > horizon);
        self.open.iter_mut().for_each(|i| i.0 = i.0.max(horizon));
        self.from = Some(horizon);
    }
}

/// A context length bucket: `<=` the next power of two, in k past 1024.
fn kv(rows: Option<u64>) -> String {
    match rows.map(u64::next_power_of_two) {
        None => String::new(),
        Some(n) if n >= 1 << 20 => format!("<={}M", n >> 20),
        Some(n) if n >= 1 << 10 => format!("<={}k", n >> 10),
        Some(n) => format!("<={n}"),
    }
}

/// The counters so far, and what the manifest says each call is.
pub struct Board {
    rank: String,
    /// Per program, per call: (label, op).
    sites: BTreeMap<String, Vec<(String, String)>>,
    /// Per program, per call: whether it communicates.
    comm: BTreeMap<String, Vec<bool>>,
    clock: Clock,
    /// Per (program, shape, kv): compute seconds and steps.
    work: BTreeMap<(String, String, String), (f64, u64)>,
}

/// An issue's shape label: its graph key, or its vars rounded up to a
/// power of two.
pub(crate) fn shape(s: &Step) -> String {
    let vars = s.issue.vars.iter().map(|(n, v)| match s.issue.graph {
        true => format!("{n}={v}"),
        false => format!("{n}<={}", v.next_power_of_two()),
    });
    vars.collect::<Vec<_>>().join(",")
}

fn secs(ns: u64) -> f64 {
    ns as f64 * 1e-9
}

/// Whether a call reads a peer buffer or runs an NCCL collective.
fn communicates(m: &Manifest, c: &Call) -> bool {
    let peer = c
        .args
        .iter()
        .any(|a| matches!(a, Arg::Buf { buf, .. } if m.buffers.get(buf).is_some_and(|b| b.kind == BufferKind::Peer)));
    let nccl = m.ops.get(&c.op).is_some_and(|op| {
        op.imp.launches.iter().any(|l| matches!(l, Launch::Extern(e) if e.entry.starts_with("extern:nccl_")))
    });
    peer || nccl
}

/// `f` of every call, by program and call index.
fn per_call<T>(m: &Manifest, f: impl Fn(usize, &Call) -> T) -> BTreeMap<String, Vec<T>> {
    m.programs
        .iter()
        .map(|(p, prog)| (p.clone(), prog.calls.iter().enumerate().map(|(i, c)| f(i, c)).collect()))
        .collect()
}

impl Board {
    pub fn new(m: &Manifest, rank: usize) -> Self {
        Self {
            rank: rank.to_string(),
            sites: per_call(m, |i, c| (c.label.clone().unwrap_or_else(|| format!("{}#{i}", c.op)), c.op.clone())),
            comm: per_call(m, |_, c| communicates(m, c)),
            clock: Clock::default(),
            work: BTreeMap::new(),
        }
    }

    fn site(&self, program: &str, call: Option<u32>) -> (String, String) {
        call.and_then(|c| self.sites.get(program)?.get(c as usize).cloned()).unwrap_or_else(|| ("?".into(), "?".into()))
    }

    fn is_comm(&self, program: &str, call: Option<u32>) -> bool {
        call.and_then(|c| self.comm.get(program)?.get(c as usize).copied()).unwrap_or(false)
    }

    /// Folds what was collected, then settles the device's clock to `now`
    /// (CUPTI's clock) less [`LAG`].
    pub fn fold(&mut self, c: Collected, now: u64) {
        let mut arrived: Vec<Interval> = c.unclaimed.iter().map(|k| (k.start, k.end, false)).collect();
        for step in c.steps.iter().filter(|s| !s.spans.is_empty()) {
            let p = &step.issue.program;
            let marked: Vec<Interval> = step.spans.iter().map(|s| (s.start, s.end, self.is_comm(p, s.call))).collect();
            let compute = union(marked.iter().filter(|i| !i.2).map(|&(a, b, _)| (a, b)).collect());
            let w = self.work.entry((p.clone(), shape(step), kv(step.issue.kv))).or_default();
            *w = (w.0 + secs(compute), w.1 + 1);
            arrived.extend(marked);
        }
        self.clock.advance(arrived, now.saturating_sub(LAG));
    }

    /// The board in Prometheus' text exposition format.
    pub fn render(&self, dropped: u64) -> String {
        let mut o = String::new();
        let r = &self.rank;
        let _ = writeln!(
            o,
            "# HELP kern_device_seconds_total The device's wall clock: compute, comm (waiting included) or idle."
        );
        let _ = writeln!(o, "# TYPE kern_device_seconds_total counter");
        for (class, v) in [("compute", self.clock.compute), ("comm", self.clock.comm), ("idle", self.clock.idle)] {
            let _ = writeln!(o, "kern_device_seconds_total{{{}}} {v}", labels(&[("rank", r), ("class", class)]));
        }
        let _ = writeln!(o, "# HELP kern_step_compute_seconds_total Time a step's compute kernels cover.");
        let _ = writeln!(o, "# TYPE kern_step_compute_seconds_total counter");
        for ((p, s, k), (v, _)) in &self.work {
            let _ = writeln!(
                o,
                "kern_step_compute_seconds_total{{{}}} {v}",
                labels(&[("rank", r), ("program", p), ("shape", s), ("kv", k)])
            );
        }
        let _ = writeln!(o, "# HELP kern_steps_total Steps run.\n# TYPE kern_steps_total counter");
        for ((p, s, k), (_, n)) in &self.work {
            let _ = writeln!(
                o,
                "kern_steps_total{{{}}} {n}",
                labels(&[("rank", r), ("program", p), ("shape", s), ("kv", k)])
            );
        }
        let _ = writeln!(o, "# HELP kern_trace_dropped_records_total Kernel records the counters miss.");
        let _ = writeln!(o, "# TYPE kern_trace_dropped_records_total counter");
        let _ =
            writeln!(o, "kern_trace_dropped_records_total{{{}}} {}", labels(&[("rank", r)]), dropped + self.clock.late);
        o
    }
}

fn escape(v: &str) -> String {
    v.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n")
}

fn labels(l: &[(&str, &str)]) -> String {
    l.iter().map(|(k, v)| format!("{k}=\"{}\"", escape(v))).collect::<Vec<_>>().join(",")
}

/// The board, and the files it is kept in with `--trace-dir`.
pub struct Tracing {
    board: Mutex<Board>,
    recorder: Option<Mutex<Recorder>>,
}

pub type Shared = Arc<Tracing>;

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Subscribes CUPTI (before any context) and starts the thread that folds
/// what it collects, and writes it under `dir` when given.
pub fn start(m: &Manifest, rank: usize, dir: Option<PathBuf>, meta: Vec<(String, String)>) -> anyhow::Result<Shared> {
    trace::start()?;
    let t = Arc::new(Tracing {
        board: Mutex::new(Board::new(m, rank)),
        recorder: dir.map(|d| Mutex::new(Recorder::new(d, rank, meta))),
    });
    let shared = t.clone();
    std::thread::Builder::new().name("kern-trace".into()).spawn(move || {
        let (mut told, mut failing) = (false, false);
        loop {
            std::thread::sleep(EVERY);
            let (c, cupti) = match trace::collect().and_then(|c| Ok((c, trace::now()?))) {
                Ok(v) => v,
                Err(e) => {
                    warn!(error = %e, "trace collect");
                    continue;
                }
            };
            let unix =
                std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos() as i64);
            if !told && !c.steps.is_empty() {
                let last = c.steps.iter().flat_map(|s| s.spans.iter().map(|k| k.end)).max().unwrap_or(0);
                let behind_ms = (cupti as i128 - last as i128) / 1_000_000;
                info!(hes = ?trace::hes(), behind_ms, "tracing kernels");
                told = true;
            }
            let mut board = lock(&t.board);
            if let Some(r) = &t.recorder {
                let site = |p: &str, call: Option<u32>| board.site(p, call);
                match lock(r).write(&c, &site, Now { unix, cupti }) {
                    Ok(()) => failing = false,
                    Err(e) if !failing => {
                        warn!(error = %format!("{e:#}"), "trace dir");
                        failing = true;
                    }
                    Err(_) => {}
                }
            }
            board.fold(c, cupti);
        }
    })?;
    Ok(shared)
}

/// `/kern/metrics` and `/kern/trace/rotate` on `port`, from a thread of
/// their own: the rank's own HTTP front end, if it has one, is infront's.
pub fn serve(board: Shared, port: u16) -> anyhow::Result<()> {
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
    let listener = rt.block_on(tokio::net::TcpListener::bind(("0.0.0.0", port)))?;
    let routes =
        Router::new().route("/kern/metrics", get(metrics)).route("/kern/trace/rotate", post(rotate)).with_state(board);
    std::thread::Builder::new().name("kern-trace-http".into()).spawn(move || {
        if let Err(e) = rt.block_on(async move { axum::serve(listener, routes).await }) {
            warn!(error = %e, "trace http");
        }
    })?;
    Ok(())
}

async fn metrics(State(t): State<Shared>) -> impl IntoResponse {
    let body = lock(&t.board).render(trace::dropped());
    ([(header::CONTENT_TYPE, "text/plain; version=0.0.4")], body)
}

/// Closes the files being written so they can be read now; answers the
/// paths closed.
async fn rotate(State(t): State<Shared>) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let r = t.recorder.as_ref().ok_or((StatusCode::NOT_FOUND, "this rank runs without --trace-dir".to_string()))?;
    let paths = lock(r).rotate().map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")))?;
    Ok(Json(json!({"closed": paths})))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Per nanosecond of `[from, to)`: compute where a compute interval
    /// covers it, else comm where a comm one does, else idle.
    fn reference(all: &[Interval], from: u64, to: u64) -> [u64; 3] {
        (from..to).fold([0; 3], |mut n, t| {
            let at = |c: bool| all.iter().any(|&(a, b, k)| k == c && a <= t && t < b);
            n[if at(false) {
                0
            } else if at(true) {
                1
            } else {
                2
            }] += 1;
            n
        })
    }

    #[test]
    fn the_clock_splits_every_settled_nanosecond_once() {
        // A small LCG: overlapping intervals, some straddling a horizon,
        // each arriving at the fold after the one its start falls in.
        let mut x = 7u64;
        let mut next = |m: u64| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (x >> 33) % m
        };
        let all: Vec<Interval> = (0..200)
            .map(|_| {
                let a = 100 + next(4000);
                (a, a + 1 + next(60), next(3) == 0)
            })
            .collect();
        let horizons: Vec<u64> = (0..=10).map(|i| 100 + i * 450).collect();
        let mut c = Clock::default();
        c.advance(Vec::new(), horizons[0]);
        // Fold `i` brings the intervals starting in `[h[i-2], h[i-1])`,
        // with the clock settled at `h[i-1]`, and settles it to `h[i]`;
        // an interval counts from the settled clock on, if it ends past it.
        let mut counted = Vec::new();
        let mut late = 0;
        for i in 1..horizons.len() {
            let due: Vec<Interval> = match i {
                1 => Vec::new(),
                _ => all.iter().copied().filter(|v| (horizons[i - 2]..horizons[i - 1]).contains(&v.0)).collect(),
            };
            for &(a, b, k) in &due {
                match b > horizons[i - 1] {
                    true => counted.push((a.max(horizons[i - 1]), b, k)),
                    false => late += 1,
                }
            }
            c.advance(due, horizons[i]);
        }
        let [compute, comm, idle] = reference(&counted, horizons[0], horizons[10]);
        let ns = |s: f64| (s * 1e9).round() as u64;
        assert_eq!((ns(c.compute), ns(c.comm), ns(c.idle), c.late), (compute, comm, idle, late));
    }

    #[test]
    fn kv_rounds_up_to_a_power_of_two() {
        let got: Vec<String> =
            [None, Some(1), Some(1000), Some(1025), Some(65536), Some(3 << 20)].into_iter().map(kv).collect();
        assert_eq!(got, ["", "<=1", "<=1k", "<=2k", "<=64k", "<=4M"]);
    }
}
