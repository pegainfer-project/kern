//! The pipeline head's rules as pure functions: which sequences an item
//! packs, when it goes out, how the stage frontier moves when it enters,
//! and the rollout a policy prices a row cap with ([`plan_cost`]).
//!
//! An item reads a stage's weights once whatever its rows, so a pipeline
//! fed one prompt per item pays a whole weight pass for a prompt's last
//! few hundred rows. Packing fills each item instead: sequences in turn
//! each give their next rows, the last one a partial chunk if that is what
//! fits, until the item has its rows, [`SEQS`] sequences or its context
//! (the sum of its sequences' lengths after the call). A sequence the
//! context has no room for is passed over, not waited on.
//!
//! The turn is shortest remaining first. A prompt continuing from a long
//! kept prefix has a few thousand rows left where a cold one has its whole
//! length, and a cold 200k prompt holds a pipeline for seconds: oldest
//! first, every hit admitted behind cold prompts waits all of them out.
//! Nothing starves: a sequence passed over by `stages` items goes ahead of
//! every sequence passed over fewer times, most passed first, and the one
//! passed over most always fits (its own context is below the bound).
//!
//! A full item (its cap or all its sequences) goes out while fewer than
//! the stages and `ahead` more are in flight, a partial one only while
//! fewer than the stages ([`goes_out`]): packing waits for rows only when
//! the stages have work.
//!
//! The head hears no stage but its answers, so it places every item it
//! sent with its cost model from the moment it sent it ([`Sent`],
//! [`frontier`]): stage s starts item k once it finished k-1, stage s-1
//! finished k ([`HOP`] later) and stage s+1 started k-2, and the last
//! stage's finish plus [`REPLY`] is the answer. Which of those answers are
//! still out is counted by the replies, not modelled. A rollout runs every
//! waiting prompt to its end from that frontier under one row rule and
//! sums when each is answered ([`plan_cost`]); it is the simulator's
//! (`ppsim`, `head.rs`), expression for expression, so a decision replayed
//! from the same events is the same bits.

use std::collections::VecDeque;

use super::cost::{max, Cost, Item, Times, MAX_STAGES};

/// µs from one stage's finish to the next stage's start, as the rollout
/// believes.
pub const HOP: f64 = 300.0;
/// µs from the last stage's finish to its answer at the head.
pub const REPLY: f64 = 6000.0;
/// The granularity of every row cap.
pub const PAGE: i64 = 64;
/// Sequences an item packs at most.
pub const SEQS: usize = 16;
/// The row caps a rollout chooses among.
pub const CAPS: [i64; 4] = [2048, 4096, 8192, 16384];

/// A waiting prompt: rows `[start, end)` of request `req`, done up to
/// `pos`, its rows stopping at `stop` and then on a snapshot grid of
/// `grid` rows (0: only at its end), skipped by `passed` items.
#[derive(Clone, Copy, Debug)]
pub struct Seq {
    pub req: usize,
    pub start: i64,
    pub pos: i64,
    pub end: i64,
    pub passed: u32,
    stop: i64,
    grid: i64,
}

impl Seq {
    /// A prompt just admitted with `hit` rows kept.
    pub fn new(req: usize, hit: i64, prompt: i64, grid: i64) -> Seq {
        let mut s = Seq { req, start: hit, pos: hit, end: prompt, passed: 0, stop: 0, grid };
        s.stop = s.next_stop();
        s
    }

    /// A prompt as the head holds it: at `pos`, its rows stopping next at
    /// `stop` (a checkpoint, or its end).
    pub fn at(req: usize, start: i64, pos: i64, end: i64, stop: i64, passed: u32, grid: i64) -> Seq {
        Seq { req, start, pos, end, passed, stop, grid }
    }

    /// The next snapshot past `pos`: the grid past the start, then the last
    /// page boundary, then the end.
    fn next_stop(&self) -> i64 {
        if self.grid == 0 {
            return self.end;
        }
        let tail = self.end.div_euclid(PAGE) * PAGE;
        if self.pos >= tail {
            return self.end;
        }
        let g = (self.pos.div_euclid(self.grid) + 1) * self.grid;
        if g < tail {
            g
        } else {
            tail
        }
    }

    fn advance(&mut self, n: i64) {
        self.pos += n;
        self.stop = self.next_stop();
    }

    /// Rows to its next stop.
    pub fn left(&self) -> i64 {
        self.stop - self.pos
    }
}

/// At most [`SEQS`] values, inline.
#[derive(Clone, Copy, Debug)]
pub struct Few<T: Copy + Default> {
    items: [T; SEQS],
    len: usize,
}

impl<T: Copy + Default> Default for Few<T> {
    fn default() -> Self {
        Few { items: [T::default(); SEQS], len: 0 }
    }
}

impl<T: Copy + Default> Few<T> {
    fn push(&mut self, x: T) {
        self.items[self.len] = x;
        self.len += 1;
    }
}

impl<T: Copy + Default> std::ops::Deref for Few<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        &self.items[..self.len]
    }
}

/// The order in which sequences are served, and each one's rows to its
/// next stop. An item takes at most [`SEQS`] sequences, so only a short
/// prefix of the order is sorted up front; [`Turn::walk`] sorts the rest
/// only if a caller gets that far.
#[derive(Default, Clone)]
pub struct Turn {
    /// The order as sort keys, each sequence's index in its low bits.
    keys: Vec<u64>,
    pub left: Vec<i64>,
    sorted: usize,
}

const SORTED: usize = 2 * SEQS;
const INDEX_BITS: u32 = 22;

impl Turn {
    pub fn of(seqs: &[Seq], stages: usize) -> Turn {
        let mut t = Turn::default();
        t.fill(seqs, stages);
        t
    }

    fn fill(&mut self, seqs: &[Seq], stages: usize) {
        assert!(seqs.len() < 1 << INDEX_BITS, "{} waiting prompts, at most 2^{INDEX_BITS}", seqs.len());
        self.left.clear();
        self.left.extend(seqs.iter().map(Seq::left));
        self.keys.clear();
        let s = stages as u32;
        self.keys.extend(seqs.iter().zip(&self.left).enumerate().filter(|(_, (_, l))| **l > 0).map(|(k, (q, &l))| {
            // Past patience ranks below 2^32 by most skipped; the rest above, by rows left (< 2^41).
            let rank = if q.passed >= s { u64::from(u32::MAX - q.passed) } else { 1 << 41 | l as u64 };
            rank << INDEX_BITS | k as u64
        }));
        if self.keys.len() > SORTED {
            self.keys.select_nth_unstable(SORTED);
        }
        self.sorted = self.keys.len().min(SORTED);
        self.keys[..self.sorted].sort_unstable();
    }

    /// Every waiting sequence in turn.
    pub fn walk(&self) -> impl Iterator<Item = usize> + '_ {
        let rest = std::iter::once(()).flat_map(move |_| {
            let mut t = self.keys[self.sorted..].to_vec();
            t.sort_unstable();
            t.into_iter().map(index)
        });
        self.keys[..self.sorted].iter().map(|&k| index(k)).chain(rest)
    }
}

fn index(key: u64) -> usize {
    (key & ((1 << INDEX_BITS) - 1)) as usize
}

/// The next item under `cap` rows and `context`: (sequence index, rows) in
/// turn order, and its total rows.
pub fn pack(seqs: &[Seq], turn: &Turn, cap: i64, context: i64) -> (Few<(usize, i64)>, i64) {
    let mut plan = Few::default();
    let (mut rows, mut ctx) = (0i64, 0i64);
    for k in turn.walk() {
        if plan.len() == SEQS || rows == cap {
            break;
        }
        let n = turn.left[k].min(cap - rows).min((context - ctx - seqs[k].pos).max(0));
        if n > 0 {
            plan.push((k, n));
            rows += n;
            ctx += seqs[k].pos + n;
        }
    }
    (plan, rows)
}

/// Whether an item of `rows` rows and `seqs` sequences under `cap` goes
/// out with `flying` items in a pipeline of `stages`: a full one while
/// fewer than `stages + ahead` are in flight, a partial one while fewer
/// than `stages`. Item times vary with the context their rows attend to,
/// and the items queued ahead absorb that.
pub fn goes_out(rows: i64, seqs: usize, cap: i64, flying: usize, stages: usize, ahead: usize) -> bool {
    let full = rows == cap || seqs == SEQS;
    seqs > 0 && flying < stages + if full { ahead } else { 0 }
}

/// Each packed sequence's (context, rows).
pub fn parts_of(seqs: &[Seq], plan: &[(usize, i64)]) -> Few<(i64, i64)> {
    let mut parts = Few::default();
    plan.iter().for_each(|&(k, n)| parts.push((seqs[k].pos, n)));
    parts
}

/// Sends `plan`: every other sequence with rows left is passed once more,
/// the packed ones advance, and the finished ones leave. Returns the
/// item's parts and the finished requests, in plan order.
pub fn apply(seqs: &mut Vec<Seq>, plan: &[(usize, i64)], left: &[i64]) -> (Few<(i64, i64)>, Few<usize>) {
    for (s, &l) in seqs.iter_mut().zip(left) {
        s.passed += u32::from(l > 0);
    }
    for &(k, _) in plan {
        seqs[k].passed = 0;
    }
    let mut parts = Few::default();
    let mut fin = Few::default();
    let mut gone = Few::<usize>::default();
    for &(k, n) in plan {
        let s = &mut seqs[k];
        parts.push((s.pos, n));
        s.advance(n);
        if s.pos == s.end {
            fin.push(s.req);
            gone.push(k);
        }
    }
    if !gone.is_empty() {
        let mut k = 0;
        seqs.retain(|_| {
            k += 1;
            !gone.contains(&(k - 1))
        });
    }
    (parts, fin)
}

/// One stage: its last finish, and the starts of its last two items once
/// that many entered it.
#[derive(Clone, Copy, Debug, Default)]
struct Stage {
    v: [f64; 3],
    len: u8,
}

/// The stage frontier.
#[derive(Clone, Copy, Debug)]
pub struct Pipe {
    stage: [Stage; MAX_STAGES],
    len: usize,
}

impl Pipe {
    pub fn idle(stages: usize, t: f64) -> Pipe {
        assert!(stages <= MAX_STAGES, "{stages} stages, at most {MAX_STAGES}");
        Pipe { stage: [Stage { v: [t, 0.0, 0.0], len: 1 }; MAX_STAGES], len: stages }
    }

    /// Each stage's last finish.
    pub fn dones(&self) -> Vec<f64> {
        self.stage[..self.len].iter().map(|s| s.v[0]).collect()
    }
}

/// An item entering the frontier at `now` with stage times `ds`: the new
/// frontier, and when the last stage finishes it.
pub fn place(pipe: &Pipe, now: f64, ds: &Times) -> (Pipe, f64) {
    let n = pipe.len;
    let mut out = *pipe;
    let mut prev = 0.0;
    for s in 0..n {
        let st = &pipe.stage[s];
        let ready = if s == 0 { now } else { prev + HOP };
        let mut t = max(st.v[0], ready);
        if s + 1 < n && pipe.stage[s + 1].len == 3 {
            t = max(t, pipe.stage[s + 1].v[1]);
        }
        prev = t + ds[s];
        out.stage[s] = match st.len {
            3 => Stage { v: [prev, st.v[2], t], len: 3 },
            2 => Stage { v: [prev, st.v[1], t], len: 3 },
            _ => Stage { v: [prev, t, 0.0], len: 2 },
        };
    }
    (out, prev)
}

/// An item the head sent: when, and its model times.
#[derive(Clone, Copy, Debug)]
pub struct Sent {
    pub issued: f64,
    pub model: Times,
}

/// The frontier and the answer times of every item in `sent` as the head
/// places them: no stage heard, each item from its send with its model
/// times behind the items before it.
pub fn frontier(sent: &VecDeque<Sent>, stages: usize) -> (Pipe, Vec<f64>) {
    let m = sent.len();
    let mut es = vec![[0.0; MAX_STAGES]; m];
    let mut ed = vec![[0.0; MAX_STAGES]; m];
    for (k, h) in sent.iter().enumerate() {
        for s in 0..stages {
            let mut c = if s == 0 { max(f64::NEG_INFINITY, h.issued) } else { f64::NEG_INFINITY };
            if k > 0 {
                c = max(c, ed[k - 1][s]);
            }
            if s > 0 {
                c = max(c, ed[k][s - 1] + HOP);
            }
            if s + 1 < stages && k >= 2 {
                c = max(c, es[k - 2][s + 1]);
            }
            es[k][s] = c;
            ed[k][s] = max(c + h.model[s], f64::NEG_INFINITY);
        }
    }
    let mut view = Pipe::idle(stages, 0.0);
    for s in 0..stages {
        view.stage[s] = if m > 1 {
            Stage { v: [ed[m - 1][s], es[m - 2][s], es[m - 1][s]], len: 3 }
        } else {
            Stage { v: [ed[m - 1][s], es[m - 1][s], 0.0], len: 2 }
        };
    }
    (view, ed.iter().map(|d| d[stages - 1] + REPLY).collect())
}

/// The most rows (at least a page, up to `hi`) whose packed item's slowest
/// stage costs at most `budget`.
pub fn fill(seqs: &[Seq], turn: &Turn, budget: f64, m: &Cost, hi: i64, context: i64) -> i64 {
    let mut item = Item::default();
    let (mut count, mut rows, mut ctx) = (0, 0i64, 0i64);
    for k in turn.walk() {
        if count == SEQS || rows == hi {
            break;
        }
        let p = seqs[k].pos;
        let n = turn.left[k].min(hi - rows).min((context - ctx - p).max(0));
        if n <= 0 {
            continue;
        }
        let over = |it: &Item| m.slowest_of(it) > budget;
        let with = item.with(p, n);
        if over(&with) {
            let (mut lo, mut top) = (0, n.div_euclid(PAGE));
            while lo < top {
                let mid = (lo + top + 1).div_euclid(2);
                if over(&item.with(p, mid * PAGE)) {
                    top = mid - 1
                } else {
                    lo = mid
                }
            }
            rows += lo * PAGE;
            break;
        }
        item = with;
        count += 1;
        rows += n;
        ctx += p + n;
    }
    rows.max(PAGE)
}

/// How a rollout sizes each item: a row cap, or the most rows within a
/// slowest-stage time budget.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rule {
    Rows(i64),
    Budget(f64),
}

impl Rule {
    pub fn cap(&self, seqs: &[Seq], turn: &Turn, m: &Cost, context: i64) -> i64 {
        match *self {
            Rule::Rows(c) => c,
            Rule::Budget(b) => fill(seqs, turn, b, m, CAPS[CAPS.len() - 1], context),
        }
    }
}

/// What a rollout charges beyond the waiting prompts' answer times: `lam`
/// per µs an arrival rides behind the items it sends, `price` per µs of
/// slowest-stage time its items commit.
#[derive(Clone, Copy, Debug, Default)]
pub struct Terms {
    pub lam: f64,
    pub price: f64,
}

/// Where a rollout starts: the waiting prompts, the frontier and the
/// answers still out at `now`, the window and the item bounds.
#[derive(Clone, Copy)]
pub struct Start<'a> {
    pub seqs: &'a [Seq],
    pub pipe: &'a Pipe,
    pub now: f64,
    pub answers: &'a [f64],
    pub ahead: usize,
    pub context: i64,
}

/// A rollout of every waiting prompt under `rule` from `at`: the sum of
/// their answer times after `now`, plus what `terms` charges.
pub fn plan_cost(at: Start, rule: Rule, m: &Cost, terms: Terms) -> f64 {
    let Start { seqs, pipe, now, answers, ahead, context } = at;
    let stages = pipe.len;
    let mut pipe = *pipe;
    let mut answers = answers.to_vec();
    let mut seqs = seqs.to_vec();
    answers.sort_by(|a, b| a.partial_cmp(b).expect("answer times are not NaN"));
    let (mut total, window) = (0.0, stages + ahead);
    let mut turn = Turn::default();
    let (mut ride, mut last, mut busy) = (0.0, None::<(f64, f64)>, 0.0);
    while !seqs.is_empty() {
        turn.fill(&seqs, stages);
        let c = rule.cap(&seqs, &turn, m, context);
        let (plan, _) = pack(&seqs, &turn, c, context);
        if plan.is_empty() {
            break;
        }
        let go = if answers.len() < window { now } else { max(now, answers[answers.len() - window]) };
        let (parts, fin) = apply(&mut seqs, &plan, &turn.left);
        let ds = m.times(&parts);
        busy += ds[..stages].iter().fold(0.0, |a: f64, &b| a.max(b));
        let (next, end) = place(&pipe, go, &ds);
        pipe = next;
        if let Some((g, e)) = last {
            ride += (go - g) * (e - g) - (go - g) * (go - g) / 2.0;
        }
        last = Some((go, end));
        answers.push(end + REPLY);
        total += (end + REPLY - now) * fin.len() as f64;
    }
    if let Some((g, e)) = last {
        ride += (e - g) * (e - g) / 2.0;
    }
    let base = if terms.lam > 0.0 { total + terms.lam * ride } else { total };
    if terms.price > 0.0 {
        base + terms.price * busy
    } else {
        base
    }
}
