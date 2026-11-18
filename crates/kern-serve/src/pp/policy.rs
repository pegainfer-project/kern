//! Item-size policies: the most rows the next item may hold, from what the
//! head sees when it sizes it (`--pp-policy`, the simulator's syntax).
//!
//! ```text
//! fixed<N>               N
//! dck<init>/<sigma>      SGLang's dynamic chunking: the rows at which the first sequence in turn, at its
//!                        context, costs what init rows cost at context 0, moved from init toward them by
//!                        sigma, at least init / 4
//! hwang<slo>/<fresh>     Hwang & Ahn's predictive chunking (OSDI '26) on the prefill side: among 1024 to
//!                        16384 rows in steps of 1024, the smallest cap whose packed item's rows per
//!                        slowest-stage time meet the demand, within the caps whose item crosses an empty
//!                        pipeline (stages times its slowest stage) within slo ms; demand is the new rows
//!                        admitted per µs over the last minute plus the rows to go of the prompts waiting
//!                        more than fresh ms over slo less their mean wait; with no cap meeting it (or no
//!                        time left), the largest within slo, else 1024
//! depth<lo>/<hi>/<n>     the admitted prompts' rows to go spread over n items, within [lo, hi]
//! time<N>                the most rows whose item's slowest stage costs at most what N rows cost at
//!                        context 0, up to 16384
//! mpc                    for each cap in 2048 / 4096 / 8192 / 16384, roll every waiting prompt out to its
//!                        end from the frontier, take the cap finishing them soonest in sum
//! mpcb                   mpc whose candidates are those caps' time budgets
//! mpch<N>                mpc, cut to time<N>'s rows when those are fewer
//! mpca<N>/<K>[/<W>]      mpch<N> unless the waiting prompts but the longest have more than K rows to go:
//!                        then mpch<W>, or mpc without W
//! mpcl<x>                mpc whose rollout also charges x times the admission rate of the last minute for
//!                        each µs an arrival would ride behind the items it sends
//! mpclb<x>               mpcl<x> over mpcb's candidates
//! mpcp<k>[/<x>]          mpcb whose rollout also pays k u / (1 - u) per µs of slowest-stage time its items
//!                        commit, u the head's utilization (at most 0.95), plus mpclb's term at weight x
//! <policy>^<N>           the policy's cap cut to time<N>'s rows when those are fewer
//! lad<p1>@<l1>+...+<top> the first rung p_i whose bound l_i exceeds the measured offered load, else top
//! headway<N1>/<N2>       lad mpclb1@0.45 + mpch<N1>@0.55 + time<N2>@0.85 + mpcp1: the rollout free, under
//!                        a ceiling, replaced by equal-time items, and priced, as the load rises
//! ```
//!
//! The two prior-work rules keep no state across decisions: Hwang & Ahn's
//! online refit of the predictor is the head's own model here.
//!
//! The head measures two loads from its own records ([`Meter`]). Offered
//! load is the slowest-stage time, under its model, that the admissions
//! of the last minute need in 8192-row chunks, each prompt counted for at
//! most 32768 new rows, per second; the largest of that over 60 s, over
//! 15 s, and the waiting prompts' remaining work spread over
//! `--pp-backlog` seconds, so it rises fast and falls slowly. Utilization
//! is the slowest-stage time of the items it sent in the last minute, per
//! second. Neither needs a stage to report: the head knows what it
//! admitted and sent, and what its model says that costs.
//!
//! A rollout must read the state as it is when the item goes: it runs on
//! the head's thread, after admission and before the send. A decision is
//! a pure function of a [`Decision`], the same bits the simulator computes
//! from the same events.

use std::collections::VecDeque;

use anyhow::{bail, Context, Result};

use super::cost::{sum, Cost};
use super::frontier::{self, pack, parts_of, Pipe, Rule, Sent, Seq, Start, Terms, Turn, CAPS, PAGE};

/// How far back the head counts admissions and sends, µs.
const WINDOW: f64 = 60e6;
/// The short window of offered load, s.
const FAST: f64 = 15.0;
/// The chunk offered load prices a prompt's rows in, and the rows a
/// prompt counts for at most.
const LOAD_CHUNK: i64 = 8192;
const LOAD_CAP: i64 = 32768;
/// The offered loads at which headway's rungs end.
const RUNGS: [f64; 3] = [0.45, 0.55, 0.85];

#[derive(Clone, Debug, PartialEq)]
pub enum Policy {
    Fixed(i64),
    Dck { init: i64, sigma: f64 },
    Hwang { slo: f64, fresh: f64 },
    Depth { lo: i64, hi: i64, n: i64 },
    Time(i64),
    Mpc,
    Mpcb,
    Mpch(i64),
    Mpca { n: i64, backlog: i64, w: Option<i64> },
    Mpcl(f64),
    Mpclb(f64),
    Mpcp { k: f64, x: f64 },
    Cut(Box<Policy>, i64),
    Ladder { rungs: Vec<(f64, Policy)>, top: Box<Policy> },
}

/// What the head sees when it sizes the next item.
pub struct Decision<'a> {
    pub now: f64,
    pub active: &'a [Seq],
    pub frontier: Pipe,
    /// The answer times of the items in flight, as the head places them.
    pub answers: Vec<f64>,
    pub model: &'a Cost,
    pub stages: usize,
    pub ahead: usize,
    /// The most context an item may hold.
    pub context: i64,
    /// The admissions of the last minute: (time, hit, prompt).
    pub recent: &'a VecDeque<(f64, i64, i64)>,
    /// Admissions per µs over the last minute.
    pub rate: f64,
    /// Offered load: slowest-stage time per second the recent admissions need.
    pub load: f64,
    /// Utilization: slowest-stage time per second of the items sent recently.
    pub util: f64,
}

impl Policy {
    pub fn parse(p: &str) -> Result<Policy> {
        if let Some(spec) = p.strip_prefix("headway") {
            let bound = |v: &str| v.parse::<i64>().with_context(|| format!("policy `{p}`: `{v}` is not an integer"));
            let (n1, n2) = spec.split_once('/').with_context(|| format!("policy `{p}` is not headway<N1>/<N2>"))?;
            let rungs = vec![Policy::Mpclb(1.0), Policy::Mpch(bound(n1)?), Policy::Time(bound(n2)?)];
            let top = Box::new(Policy::Mpcp { k: 1.0, x: 0.0 });
            return Ok(Policy::Ladder { rungs: RUNGS.into_iter().zip(rungs).collect(), top });
        }
        if let Some(spec) = p.strip_prefix("lad") {
            let mut parts: Vec<&str> = spec.split('+').collect();
            let top = Box::new(Policy::parse(parts.pop().unwrap_or_default())?);
            let rungs = parts
                .iter()
                .map(|r| {
                    let (q, l) =
                        r.rsplit_once('@').with_context(|| format!("policy `{p}`: rung `{r}` has no `@<load>`"))?;
                    let l = l.parse().with_context(|| format!("policy `{p}`: rung `{r}`: `{l}` is not a number"))?;
                    Ok((l, Policy::parse(q)?))
                })
                .collect::<Result<_>>()?;
            return Ok(Policy::Ladder { rungs, top });
        }
        if let Some((q, n)) = p.rsplit_once('^') {
            let n = n.parse().with_context(|| format!("policy `{p}`: cut `{n}` is not an integer"))?;
            return Ok(Policy::Cut(Box::new(Policy::parse(q)?), n));
        }
        let num = |v: &str| v.parse::<f64>().with_context(|| format!("policy `{p}`: `{v}` is not a number"));
        if let Some(spec) = p.strip_prefix("hwang") {
            let (slo, fresh) =
                spec.split_once('/').with_context(|| format!("policy `{p}` is not hwang<slo>/<fresh>"))?;
            return Ok(Policy::Hwang { slo: num(slo)? * 1e3, fresh: num(fresh)? * 1e3 });
        }
        if let Some(spec) = p.strip_prefix("mpcp") {
            let (k, x) = spec.split_once('/').unwrap_or((spec, "0"));
            return Ok(Policy::Mpcp { k: num(k)?, x: num(x)? });
        }
        let names = ["fixed", "dck", "depth", "time", "mpch", "mpca", "mpclb", "mpcl"];
        let Some(name) = names.iter().find(|n| p.starts_with(*n)) else {
            return match p {
                "mpc" => Ok(Policy::Mpc),
                "mpcb" => Ok(Policy::Mpcb),
                _ => bail!("unknown policy `{p}`"),
            };
        };
        let args: Vec<&str> = p[name.len()..].split('/').collect();
        let int = |i: usize| -> Result<i64> {
            let x = args.get(i).with_context(|| format!("policy `{p}` needs {} argument(s)", i + 1))?;
            x.parse().with_context(|| format!("policy `{p}`: argument `{x}` is not an integer"))
        };
        let want = |n: &[usize]| match n.contains(&args.len()) {
            true => Ok(()),
            false => Err(anyhow::anyhow!("policy `{p}` takes {n:?} argument(s), got {}", args.len())),
        };
        Ok(match *name {
            "fixed" => want(&[1]).and(int(0).map(Policy::Fixed))?,
            "dck" => {
                want(&[2])?;
                Policy::Dck { init: int(0)?, sigma: num(args[1])? }
            }
            "time" => want(&[1]).and(int(0).map(Policy::Time))?,
            "mpch" => want(&[1]).and(int(0).map(Policy::Mpch))?,
            "mpcl" => want(&[1]).and(num(args[0]).map(Policy::Mpcl))?,
            "mpclb" => want(&[1]).and(num(args[0]).map(Policy::Mpclb))?,
            "mpca" => {
                want(&[2, 3])?;
                Policy::Mpca { n: int(0)?, backlog: int(1)?, w: if args.len() == 3 { Some(int(2)?) } else { None } }
            }
            _ => {
                want(&[3])?;
                Policy::Depth { lo: int(0)?, hi: int(1)?, n: int(2)? }
            }
        })
    }

    /// Whether the policy reads the cost model; one that does not runs
    /// with none.
    pub fn needs_cost(&self) -> bool {
        match self {
            Policy::Fixed(_) | Policy::Depth { .. } => false,
            Policy::Cut(..) => true,
            Policy::Ladder { rungs, top } => top.needs_cost() || rungs.iter().any(|r| r.1.needs_cost()),
            _ => true,
        }
    }

    /// The ladder rung a load picks (`rungs.len()` for the top), none for
    /// a policy that is no ladder.
    pub fn rung(&self, load: f64) -> Option<usize> {
        match self {
            Policy::Ladder { rungs, .. } => Some(rungs.iter().position(|r| load < r.0).unwrap_or(rungs.len())),
            _ => None,
        }
    }

    pub fn cap(&self, d: &Decision) -> i64 {
        let (m, s) = (d.model, d.stages);
        match self {
            Policy::Ladder { rungs, top } => {
                return rungs.iter().find(|r| d.load < r.0).map_or(&**top, |r| &r.1).cap(d);
            }
            Policy::Cut(q, n) => return q.cap(d).min(Policy::Time(*n).cap(d)),
            _ => {}
        }
        match *self {
            Policy::Fixed(n) => n,
            Policy::Dck { init, sigma } => {
                Turn::of(d.active, s).first().map_or(init, |k| dck(d.active[k].pos, init, sigma, m))
            }
            Policy::Hwang { slo, fresh } => hwang(d, slo, fresh),
            Policy::Depth { lo, hi, n } => {
                let togo: i64 = d.active.iter().map(|s| (s.end - s.pos).max(0)).sum();
                hi.min(lo.max(-(-togo).div_euclid(n * PAGE) * PAGE))
            }
            Policy::Time(n) => frontier::fill(
                d.active,
                &Turn::of(d.active, s),
                m.slowest(&[(0, n)]),
                m,
                CAPS[CAPS.len() - 1],
                d.context,
            ),
            Policy::Mpc | Policy::Mpcb | Policy::Mpcl(_) | Policy::Mpclb(_) | Policy::Mpcp { .. } => self.mpc(d),
            Policy::Mpch(n) => self.mpc(d).min(Policy::Time(n).cap(d)),
            Policy::Mpca { n, backlog, w } => {
                let cap = self.mpc(d);
                let lefts = d.active.iter().map(|x| (x.end - x.pos).max(0));
                let longest = lefts.clone().max().unwrap_or(0);
                let lim = if lefts.sum::<i64>() - longest <= backlog { n } else { w.unwrap_or(0) };
                // A zero limit means no cut.
                if lim != 0 {
                    cap.min(Policy::Time(lim).cap(d))
                } else {
                    cap
                }
            }
            Policy::Ladder { .. } | Policy::Cut(..) => unreachable!("dispatched above"),
        }
    }

    fn mpc(&self, d: &Decision) -> i64 {
        let m = d.model;
        let cands: Vec<Rule> = match self {
            Policy::Mpcb | Policy::Mpclb(_) | Policy::Mpcp { .. } => {
                CAPS.iter().map(|&c| Rule::Budget(m.slowest(&[(0, c)]))).collect()
            }
            _ => CAPS.iter().map(|&c| Rule::Rows(c)).collect(),
        };
        let lam = match *self {
            Policy::Mpcl(x) | Policy::Mpclb(x) | Policy::Mpcp { x, .. } => x * d.rate,
            _ => 0.0,
        };
        let price = match *self {
            Policy::Mpcp { k, .. } => {
                let u = d.util.min(0.95);
                k * u / (1.0 - u)
            }
            _ => 0.0,
        };
        let at = Start {
            seqs: d.active,
            pipe: &d.frontier,
            now: d.now,
            answers: &d.answers,
            ahead: d.ahead,
            context: d.context,
        };
        let (best, _) = cands
            .iter()
            .map(|&r| (r, frontier::plan_cost(at, r, m, Terms { lam, price })))
            .reduce(|b, x| if x.1 < b.1 { x } else { b })
            .expect("there is a candidate");
        best.cap(d.active, &Turn::of(d.active, d.stages), m, d.context)
    }
}

/// `Policy::Dck`'s cap for a first sequence over `l` rows of context.
fn dck(l: i64, init: i64, sigma: f64, m: &Cost) -> i64 {
    let rows = init as f64 - sigma * (init as f64 - m.equal(l, init));
    init.div_euclid(4).max((rows as i64).div_euclid(PAGE) * PAGE).max(PAGE)
}

/// `Policy::Hwang`'s cap.
fn hwang(d: &Decision, slo: f64, fresh: f64) -> i64 {
    let turn = Turn::of(d.active, d.stages);
    let cands: Vec<(i64, f64)> = (1..=16)
        .map(|k| {
            let (plan, rows) = pack(d.active, &turn, k * 1024, d.context);
            (k * 1024, rows, d.model.slowest(&parts_of(d.active, &plan)))
        })
        .filter(|&(_, _, t)| d.stages as f64 * t <= slo)
        .map(|(c, rows, t)| (c, if t > 0.0 { rows as f64 / t } else { 0.0 }))
        .collect();
    let arriving = sum(d.recent.iter().filter(|r| r.0 >= d.now - WINDOW).map(|r| (r.2 - r.1) as f64)) / WINDOW;
    let stale: Vec<(f64, f64)> = (d.active.iter())
        .map(|q| ((q.end - q.pos) as f64, d.now - q.since))
        .filter(|&(rows, wait)| rows > 0.0 && wait > fresh)
        .collect();
    let need = match stale.is_empty() {
        true => Some(arriving),
        false => {
            let left = slo - sum(stale.iter().map(|x| x.1)) / stale.len() as f64;
            (left > 0.0).then(|| arriving + sum(stale.iter().map(|x| x.0)) / left)
        }
    };
    need.and_then(|n| cands.iter().find(|x| x.1 >= n)).or(cands.last()).map_or(1024, |x| x.0)
}

/// The head's own records, all a policy needs of the past: the
/// admissions and the sends of the last minute, and the sends a frontier
/// is placed from, each with what the head heard of it since.
pub struct Meter {
    stages: usize,
    ahead: usize,
    /// Seconds the waiting prompts' remaining work is spread over for the
    /// backlog load reading (0: none).
    backlog: f64,
    /// Admissions: (time, hit, prompt), and each one's slowest-stage time
    /// as offered load counts it.
    recent: VecDeque<(f64, i64, i64)>,
    work: VecDeque<f64>,
    /// Sends: (time, slowest-stage time).
    busy: VecDeque<(f64, f64)>,
    sent: VecDeque<Sent>,
}

impl Meter {
    pub fn new(stages: usize, ahead: usize, backlog: f64) -> Meter {
        Meter {
            stages,
            ahead,
            backlog,
            recent: VecDeque::new(),
            work: VecDeque::new(),
            busy: VecDeque::new(),
            sent: VecDeque::new(),
        }
    }

    /// A prompt admitted at `now` with `hit` of its `prompt` rows kept.
    pub fn admit(&mut self, now: f64, hit: i64, prompt: i64, m: &Cost) {
        self.recent.push_back((now, hit, prompt));
        self.work.push_back(work(m, hit, prompt));
        self.forget(now);
    }

    /// Admissions older than a minute at `now` drop out of the rate.
    pub fn forget(&mut self, now: f64) {
        while self.recent.front().is_some_and(|s| s.0 < now - WINDOW) {
            self.recent.pop_front();
            self.work.pop_front();
        }
    }

    /// Item `item` of `parts` (context, rows) sent at `go`.
    pub fn sent(&mut self, item: u64, go: f64, parts: &[(i64, i64)], m: &Cost) {
        let model = m.times(parts);
        self.busy.push_back((go, model[..self.stages].iter().fold(0.0, |a: f64, &b| a.max(b))));
        while self.busy.front().is_some_and(|x| x.0 < go - WINDOW) {
            self.busy.pop_front();
        }
        self.sent.push_back(Sent { item, issued: go, model, start: None, done: None, exit: None });
        if self.sent.len() > self.stages + self.ahead + 3 + 64 {
            self.sent.pop_front();
        }
    }

    /// The head's own device started item `item` at `t`.
    pub fn began(&mut self, item: u64, t: f64) {
        self.heard(item, |s| s.start = Some(t));
    }

    /// The head's own device finished item `item` at `t`.
    pub fn ended(&mut self, item: u64, t: f64) {
        self.heard(item, |s| s.done = Some(t));
    }

    /// The last stage finished item `item` at `t`: its answer came back.
    pub fn answered(&mut self, item: u64, t: f64) {
        self.heard(item, |s| s.exit = Some(t));
    }

    /// An item the meter no longer holds, or never did (a step), is not
    /// placed and needs no time.
    fn heard(&mut self, item: u64, set: impl FnOnce(&mut Sent)) {
        if let Some(s) = self.sent.iter_mut().rev().find(|s| s.item == item) {
            set(s);
        }
    }

    /// Offered load at `now` with `active` waiting: the larger of the last
    /// minute's and the last 15 s's admissions, and, with a backlog
    /// horizon, the waiting prompts' remaining work spread over it, which
    /// sees a queue the admissions stopped showing.
    pub fn load(&self, now: f64, active: &[Seq], m: &Cost) -> f64 {
        let over = |secs: f64| {
            let since = now - secs * 1e6;
            let busy: f64 = self.recent.iter().zip(&self.work).filter(|(s, _)| s.0 >= since).map(|(_, x)| x).sum();
            busy / (secs * 1e6)
        };
        let l = over(WINDOW / 1e6).max(over(FAST));
        match self.backlog > 0.0 {
            true => l.max(active.iter().map(|q| work(m, q.pos, q.end)).sum::<f64>() / (self.backlog * 1e6)),
            false => l,
        }
    }

    /// Utilization at `now`.
    pub fn util(&self, now: f64) -> f64 {
        let since = now - WINDOW;
        self.busy.iter().filter(|x| x.0 >= since).map(|x| x.1).sum::<f64>() / WINDOW
    }

    /// What the head sees at `now` of `active`, `flying` items in flight.
    pub fn decision<'a>(
        &'a self,
        now: f64,
        active: &'a [Seq],
        flying: usize,
        m: &'a Cost,
        context: i64,
    ) -> Decision<'a> {
        let (frontier, answers) = match self.sent.is_empty() {
            true => (Pipe::idle(self.stages, 0.0), Vec::new()),
            false => {
                let (view, mut seen) = frontier::frontier(&self.sent, self.stages, now);
                seen.sort_by(|a, b| a.partial_cmp(b).expect("answer times are not NaN"));
                let n = flying.min(seen.len());
                (view, seen[seen.len() - n..].to_vec())
            }
        };
        Decision {
            now,
            active,
            frontier,
            answers,
            model: m,
            stages: self.stages,
            ahead: self.ahead,
            context,
            recent: &self.recent,
            rate: self.recent.len() as f64 / WINDOW,
            load: self.load(now, active, m),
            util: self.util(now),
        }
    }
}

/// Slowest-stage time of rows [`from`, `to`) in `LOAD_CHUNK`-row chunks, at
/// most `LOAD_CAP` rows: what offered load prices a prompt at.
fn work(m: &Cost, from: i64, to: i64) -> f64 {
    let end = to.min(from + LOAD_CAP);
    (from..end).step_by(LOAD_CHUNK as usize).map(|p| m.slowest(&[(p, LOAD_CHUNK.min(end - p))])).sum()
}
