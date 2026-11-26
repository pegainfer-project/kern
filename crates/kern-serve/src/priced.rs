//! Serving without a device: every call takes what a bench report
//! measured for its shape.
//!
//! A server is what it decides (admission, batching, chunking, the prefix
//! index, eviction) and what its GPU takes to run each decision.
//! `KernScheduler` runs its ranks' orders through [`Run`], so the second
//! is replaced here without touching the first: [`Priced`] answers an
//! order once the time `kern bench` measured for its call's shape has
//! passed. Everything above it is the server itself (the frontend, the
//! scheduler, the rank's pool), so a question about a policy is asked of
//! the code that will run it, at the speed of the GPU the report was
//! measured on, on a machine that has none.
//!
//! # A price
//!
//! [`Prices`] reads the report of the workload `kern workload` writes
//! from a manifest's bounds (docs/bench.md, "A workload from the manifest
//! alone"): every call size measured over a shallow context and a deep
//! one. A context adds to a call linearly, so a size is two numbers, its
//! time over the shallow context and what a token of context adds.
//!
//! - A step is priced at its own size, sequences and rows, and at no
//!   other: the scheduler pads a batch to the sizes it knows, so a size
//!   the report lacks is a size nobody measured, and an error naming it.
//! - A chunk's time over the shallow context is read between the two
//!   measured chunks around its rows. What a token adds is the measured
//!   chunk's at or above it: that moves with the rows in steps, not
//!   smoothly, and the chunk above is past the step.
//!
//! # What is not priced
//!
//! Only programs are. The pool's acts (a slot zeroed, pages copied) take
//! no time here, and the host's own work between two orders takes what it
//! takes on this machine. A padded batch is priced as the report measured
//! its size, every row real. The pooled tier, a second rank, and a
//! speculative round (whose acceptance is the model's, not the report's)
//! are refused.
//!
//! # What is answered
//!
//! No model runs, so every row hands back one token, the same for all: a
//! sequence repeats it until the request's own limit ends it. The lengths
//! a client asks for are the workload; the text is not. The token is an
//! ordinary word's, given at [`Priced::new`]: the frontend holds back a
//! first token that is white space, and a first token that reaches the
//! client one step late is a time to first token one step, and every
//! prefill queued before it, too long.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use anyhow::{bail, ensure, Context, Result};
use kern_manifest::protocol::{Axis, Rows};
use kern_manifest::types::{Fill, Manifest};
use kern_manifest::Protocol;
use kern_pool::Pool;
use serde::Deserialize;
use tracing::debug;

use crate::logline;
use crate::pd::Run;
use crate::pp::Item;
use crate::ranks::{Act, Answer, Order};

/// How close to its deadline a wait stops sleeping and spins: a sleep
/// overshoots by tens of microseconds, a tenth of a percent of a step.
const SPIN: Duration = Duration::from_micros(200);

#[derive(Deserialize)]
struct Report {
    manifest_sha256: String,
    scenarios: Vec<Measured>,
}

#[derive(Deserialize)]
struct Measured {
    scenario: Shape,
    rank: usize,
    graph: Graph,
}

#[derive(Deserialize)]
struct Shape {
    program: String,
    groups: u64,
    rows: u64,
    chunk: bool,
    context: Vec<u64>,
}

#[derive(Deserialize)]
struct Graph {
    stats: Stats,
}

#[derive(Deserialize)]
struct Stats {
    p50: f64,
}

/// One call, as an order carries it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Call {
    pub program: String,
    pub groups: u64,
    /// Rows of every sequence.
    pub rows: u64,
    /// Whether the rows are a prompt's chunk rather than a step's.
    pub chunk: bool,
    /// Tokens every sequence holds before the call.
    pub context: Vec<u64>,
}

/// A call size's time: over `at` tokens of context in all, and what each
/// token past them adds.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Line {
    at: f64,
    us: f64,
    per_token: Option<f64>,
}

impl Line {
    /// The line through a size's measured points, `(tokens of context, µs)`:
    /// the shallowest, and the slope to the deepest when there is another.
    fn through(points: &[(f64, f64)]) -> Line {
        let lo = points.iter().copied().fold((f64::INFINITY, 0.0), |a, b| if b.0 < a.0 { b } else { a });
        let hi = points.iter().copied().fold((f64::NEG_INFINITY, 0.0), |a, b| if b.0 > a.0 { b } else { a });
        Line { at: lo.0, us: lo.1, per_token: (hi.0 > lo.0).then(|| (hi.1 - lo.1) / (hi.0 - lo.0)) }
    }
}

/// What every call of a report's manifest takes.
#[derive(Debug, Clone, PartialEq)]
pub struct Prices {
    /// The manifest the report was measured on (sha256, hex).
    pub manifest: String,
    /// By program, sequences and rows.
    steps: BTreeMap<(String, u64, u64), Line>,
    /// By program and sequences, in rows order.
    chunks: BTreeMap<(String, u64), Vec<(u64, Line)>>,
}

impl Prices {
    /// Read a `kern bench` report: rank 0's median of every scenario.
    pub fn read(json: &str) -> Result<Prices> {
        let r: Report = serde_json::from_str(json).context("not a `kern bench` report")?;
        let mut points: BTreeMap<(String, u64, u64, bool), Vec<(f64, f64)>> = BTreeMap::new();
        for m in r.scenarios.iter().filter(|m| m.rank == 0) {
            let s = &m.scenario;
            let held = s.context.iter().sum::<u64>() as f64;
            points.entry((s.program.clone(), s.groups, s.rows, s.chunk)).or_default().push((held, m.graph.stats.p50));
        }
        ensure!(!points.is_empty(), "the report measured no scenario");
        let mut prices = Prices { manifest: r.manifest_sha256, steps: BTreeMap::new(), chunks: BTreeMap::new() };
        for ((program, groups, rows, chunk), pts) in points {
            let line = Line::through(&pts);
            match chunk {
                true => prices.chunks.entry((program, groups)).or_default().push((rows, line)),
                false => drop(prices.steps.insert((program, groups, rows), line)),
            }
        }
        Ok(prices)
    }

    /// Microseconds `c` takes.
    pub fn price(&self, c: &Call) -> Result<f64> {
        let held = c.context.iter().sum::<u64>() as f64;
        if !c.chunk {
            let Some(l) = self.steps.get(&(c.program.clone(), c.groups, c.rows)) else {
                let have: Vec<u64> =
                    self.steps.keys().filter(|k| k.0 == c.program && k.2 == c.rows).map(|k| k.1).collect();
                bail!(
                    "the report has no `{}` step of {} sequences (it has {have:?}): measure it, `kern workload --groups` takes the sizes",
                    c.program,
                    c.groups
                )
            };
            return Ok(l.us + l.per_token.unwrap_or(0.0) * (held - l.at));
        }
        let table = self.chunks.get(&(c.program.clone(), c.groups)).map_or(&[][..], Vec::as_slice);
        let below = table.iter().rev().find(|(rows, _)| *rows <= c.rows);
        let above = table.iter().find(|(rows, _)| *rows >= c.rows);
        let (Some((r0, lo)), Some((r1, hi))) = (below, above) else {
            bail!(
                "the report prices `{}` chunks of {} rows, and this one has {}: serve with a `--chunk` it measured, or measure more with `kern workload --chunk`",
                c.program,
                match (table.first(), table.last()) {
                    (Some(a), Some(b)) => format!("{} to {}", a.0, b.0),
                    _ => "no number of".into(),
                },
                c.rows
            )
        };
        let t = if r1 == r0 { 0.0 } else { (c.rows - r0) as f64 / (r1 - r0) as f64 };
        let mix = |a: f64, b: f64| a + (b - a) * t;
        let above = table.iter().filter(|(rows, _)| *rows >= c.rows).find_map(|(_, l)| l.per_token);
        let per_token = above.or_else(|| table.iter().rev().find_map(|(_, l)| l.per_token)).unwrap_or(0.0);
        Ok(mix(lo.us, hi.us) + per_token * (held - mix(lo.at, hi.at)))
    }
}

/// The pool a rank's load would cut for `tokens` tokens of paged state and
/// `slots` sequence slots, as `kern_runtime`'s load cuts it: whole pages,
/// and the null line on top of the slots when a state is per sequence.
pub fn pool(m: &Manifest, tokens: u64, slots: u64) -> Result<Pool> {
    let unit = kern_pool::page_unit(m);
    let per_seq = m.states.values().any(|s| s.bytes_per_seq > 0);
    Ok(Pool::new(m, (tokens / unit) as usize, if per_seq { slots as usize + 1 } else { 0 })?)
}

/// A rank with no device: it answers an order once its call's price has
/// passed.
pub struct Priced {
    p: Protocol,
    prices: Prices,
    token: i64,
}

impl Priced {
    /// A rank that answers every row with `token`. Refuses a protocol whose
    /// answers a report cannot stand for.
    pub fn new(p: Protocol, prices: Prices, token: i64) -> Result<Priced> {
        ensure!(
            p.ranks.is_none(),
            "a manifest of several ranks is not priced yet: one rank's report says nothing of its peers"
        );
        if let Some(f) = p.forwards.iter().find(|f| f.count.is_some()) {
            bail!("`{}` hands back a count of accepted tokens, which is the model's to say and no report's", f.name);
        }
        ensure!(
            p.any(Fill::SeqLen).is_some(),
            "the manifest declares no `seq_len` fill, so an order does not say how deep its sequences are"
        );
        Ok(Priced { p, prices, token })
    }

    /// The call `item` is: its shape off the vars, its sequences' depths
    /// off their lengths after it.
    pub fn call(&self, item: &Item) -> Result<Call> {
        let var = |name: &str| item.vars.get(name).copied().with_context(|| format!("`{}`: no `{name}`", item.program));
        let f = self
            .p
            .forwards
            .iter()
            .find(|f| f.name == item.program)
            .with_context(|| format!("`{}` is no program a caller drives", item.program))?;
        let groups = var(&self.p.groups.var)?;
        let rows = var(&self.p.rows.var)? / groups.max(1);
        let lens = self.p.any(Fill::SeqLen).expect("checked at new");
        let after = item
            .rows
            .get(&lens.name)
            .and_then(|v| v.ints())
            .with_context(|| format!("`{}`: no `{}`", item.program, lens.name))?;
        Ok(Call {
            program: item.program.clone(),
            groups,
            rows,
            chunk: f.rows == Rows::Var,
            context: after.iter().map(|&n| (n as u64).saturating_sub(rows)).collect(),
        })
    }

    /// What `item`'s program hands back: the token, in every entry of its
    /// `tokens` output.
    fn answer(&self, item: &Item) -> Result<Answer> {
        let f = self.p.forwards.iter().find(|f| f.name == item.program);
        let Some(out) = f.and_then(|f| f.emits).map(|i| &self.p.fills[i]) else { return Ok(Answer::default()) };
        let var = |name: &str| item.vars.get(name).copied().unwrap_or(0);
        let n = out.width
            * match out.axis {
                Axis::Rows => var(&self.p.rows.var),
                Axis::Groups => var(&self.p.groups.var),
                Axis::Ranks | Axis::Fixed(_) => bail!("`{}` is not over this rank's rows or sequences", out.name),
            };
        Ok(Answer { tokens: vec![self.token; n as usize], ..Answer::default() })
    }
}

/// Returns at `deadline`, to the microsecond: asleep until just before
/// it, spinning from there.
fn wait(deadline: Instant) {
    if let Some(d) = deadline.checked_duration_since(Instant::now()).and_then(|d| d.checked_sub(SPIN)) {
        std::thread::sleep(d);
    }
    while Instant::now() < deadline {
        std::hint::spin_loop();
    }
}

impl Run for Priced {
    fn run(&mut self, orders: Vec<Order>) -> Result<Vec<Answer>> {
        let (issued, issued_us) = (Instant::now(), logline::now_us());
        ensure!(orders.len() == 1, "{} orders for one rank", orders.len());
        let o = &orders[0];
        ensure!(
            o.acts.iter().all(|a| matches!(a, Act::Zero(_) | Act::Copy(_))),
            "an act on the pooled tier, and a priced rank joins none"
        );
        let Some(item) = &o.item else { return Ok(vec![Answer::default()]) };
        let us = self.prices.price(&self.call(item)?)?;
        let a = self.answer(item)?;
        wait(issued + Duration::from_secs_f64(us / 1e6));
        // The line a rank with a device leaves, so the two read alike.
        let vars = item.vars.values().map(u64::to_string).collect::<Vec<_>>().join("x");
        debug!(rank = 0, program = %item.program, acts = o.acts.len(), %vars, issued_us, priced_us = us, done_us = logline::now_us(), "rank item");
        Ok(vec![a])
    }
}
