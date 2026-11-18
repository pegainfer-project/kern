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
//! The head hears two ends of the pipeline: when its own device starts
//! and finishes an item, and, by the item's answer, when the last stage
//! finished it. Everything between it places with its cost model
//! ([`Sent`], [`frontier`]): stage s starts item k once it finished k-1,
//! stage s-1 finished k ([`HOP`] later) and stage s+1 started k-2, and the
//! last stage's finish plus [`REPLY`] is the answer. A time it heard
//! replaces the model's, the items behind are placed again from it, and an
//! answered item's middle stages are pulled back to fit the finish it
//! heard, so the model's error on one item does not carry into the next.
//! Which answers are still out is counted by the replies, not modelled. A
//! rollout runs every waiting prompt to its end from that frontier under
//! one row rule and sums when each is answered ([`plan_cost`]); its
//! arithmetic is the simulator's (`ppsim`, `head.rs`, `view=ends`),
//! expression for expression, so a decision replayed from the same events
//! is the same bits.
//!
//! A rollout sends as many items as the waiting rows fill, and the head
//! runs several before each item it sends, so an item of one costs little:
//! the turn is sorted once and kept ([`Turn`]), and a budget's rows start
//! from a root of the cost model and are checked as a bisection would
//! check them ([`fill`]). `benches/pp_decide.rs` times a decision.

use std::collections::VecDeque;

use super::cost::{max, min, Cost, Item, Times, MAX_STAGES};

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

/// A waiting prompt: rows `[start, end)` of request `req`, admitted at
/// `since`, done up to `pos`, its rows stopping at `stop` and then at its
/// end, skipped by `passed` items.
#[derive(Clone, Copy, Debug)]
pub struct Seq {
    pub req: usize,
    pub start: i64,
    pub pos: i64,
    pub end: i64,
    pub passed: u32,
    pub since: f64,
    stop: i64,
}

impl Seq {
    /// A prompt admitted at `since` with `hit` rows kept.
    pub fn new(req: usize, hit: i64, prompt: i64, since: f64) -> Seq {
        Seq { req, start: hit, pos: hit, end: prompt, passed: 0, since, stop: prompt }
    }

    /// The prompt at `pos`, its rows stopping next at `stop` (a
    /// checkpoint, or its end), skipped by `passed` items.
    pub fn at(self, pos: i64, stop: i64, passed: u32) -> Seq {
        Seq { pos, stop, passed, ..self }
    }

    fn advance(&mut self, n: i64) {
        self.pos += n;
        self.stop = self.end;
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
/// next stop. [`Turn::send`] keeps both as items go, so a rollout sorts
/// its queue once: an item moves the few sequences it packs and the few
/// its passing puts out of patience, and no other.
#[derive(Clone)]
pub struct Turn {
    /// The sequences out of patience, most passed over first.
    starved: VecDeque<usize>,
    /// The patient ones behind them as sort keys, fewest rows left first,
    /// each one's index in the low bits.
    patient: Vec<u64>,
    pub left: Vec<i64>,
    /// The items sent when each sequence was last passed over by none.
    fresh: Vec<i64>,
    /// The patient sequences and the items sent when each stops being
    /// one, in that order.
    due: VecDeque<(i64, usize)>,
    sent: i64,
    patience: i64,
}

const INDEX_BITS: u32 = 22;

fn key(left: i64, k: usize) -> u64 {
    (left as u64) << INDEX_BITS | k as u64
}

impl Turn {
    pub fn of(seqs: &[Seq], stages: usize) -> Turn {
        assert!(seqs.len() < 1 << INDEX_BITS, "{} waiting prompts, at most 2^{INDEX_BITS}", seqs.len());
        let patience = stages as i64;
        let fresh = |q: &Seq| -i64::from(q.passed);
        let live =
            |out: bool| seqs.iter().enumerate().filter(move |(_, q)| q.left() > 0 && (fresh(q) + patience <= 0) == out);
        let mut starved: Vec<(i64, usize)> = live(true).map(|(k, q)| (fresh(q), k)).collect();
        let mut patient: Vec<u64> = live(false).map(|(k, q)| key(q.left(), k)).collect();
        let mut due: Vec<(i64, usize)> = live(false).map(|(k, q)| (fresh(q) + patience, k)).collect();
        starved.sort_unstable();
        patient.sort_unstable();
        due.sort_unstable();
        Turn {
            starved: starved.into_iter().map(|(_, k)| k).collect(),
            patient,
            left: seqs.iter().map(Seq::left).collect(),
            fresh: seqs.iter().map(fresh).collect(),
            due: due.into(),
            sent: 0,
            patience,
        }
    }

    /// The first sequence in turn.
    pub fn first(&self) -> Option<usize> {
        self.walk().next()
    }

    /// Every waiting sequence in turn.
    pub fn walk(&self) -> impl Iterator<Item = usize> + '_ {
        self.starved.iter().copied().chain(self.patient.iter().map(|&k| (k & ((1 << INDEX_BITS) - 1)) as usize))
    }

    /// Sends `plan` as [`apply`] does, the finished sequences staying in
    /// `seqs` out of turn: the packed ones advance and are patient again,
    /// and a sequence this item is the last to pass in patience goes
    /// behind those already out of it. Returns the item and how many
    /// requests it finishes.
    fn send(&mut self, seqs: &mut [Seq], plan: &[(usize, i64)]) -> (Item, usize) {
        let (mut item, mut finished) = (Item::default(), 0);
        for &(k, n) in plan {
            if self.sent - self.fresh[k] >= self.patience {
                let at = self.starved.iter().position(|&x| x == k).expect("a sequence out of patience is in turn");
                self.starved.remove(at);
            } else {
                self.wait(key(self.left[k], k), false);
            }
            item = item.with(seqs[k].pos, n);
            seqs[k].advance(n);
            (self.left[k], self.fresh[k]) = (seqs[k].left(), self.sent + 1);
            if self.left[k] > 0 {
                self.wait(key(self.left[k], k), true);
                self.due.push_back((self.sent + 1 + self.patience, k));
            } else {
                finished += 1;
            }
        }
        self.sent += 1;
        while let Some(&(at, k)) = self.due.front().filter(|d| d.0 <= self.sent) {
            self.due.pop_front();
            if self.left[k] > 0 && self.fresh[k] + self.patience == at {
                self.wait(key(self.left[k], k), false);
                let later = |&&x: &&usize| self.fresh[x] == self.fresh[k] && x > k;
                let behind = self.starved.iter().rev().take_while(later).count();
                self.starved.insert(self.starved.len() - behind, k);
            }
        }
        (item, finished)
    }

    /// Puts `key` among the patient, or takes it from them.
    fn wait(&mut self, key: u64, put: bool) {
        let at = self.patient.partition_point(|&x| x < key);
        if put {
            self.patient.insert(at, key);
        } else {
            assert_eq!(self.patient.remove(at), key, "a patient sequence has its key");
        }
    }
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

    /// Takes an item at `now` with stage times `ds`; when the last stage
    /// finishes it.
    fn enter(&mut self, now: f64, ds: &Times) -> f64 {
        let n = self.len;
        let mut prev = 0.0;
        for s in 0..n {
            let st = self.stage[s];
            let ready = if s == 0 { now } else { prev + HOP };
            let mut t = max(st.v[0], ready);
            if s + 1 < n && self.stage[s + 1].len == 3 {
                t = max(t, self.stage[s + 1].v[1]);
            }
            prev = t + ds[s];
            self.stage[s] = match st.len {
                3 => Stage { v: [prev, st.v[2], t], len: 3 },
                2 => Stage { v: [prev, st.v[1], t], len: 3 },
                _ => Stage { v: [prev, t, 0.0], len: 2 },
            };
        }
        prev
    }

    /// Each stage's last finish.
    pub fn dones(&self) -> Vec<f64> {
        self.stage[..self.len].iter().map(|s| s.v[0]).collect()
    }
}

/// An item entering the frontier at `now` with stage times `ds`: the new
/// frontier, and when the last stage finishes it.
pub fn place(pipe: &Pipe, now: f64, ds: &Times) -> (Pipe, f64) {
    let mut out = *pipe;
    let end = out.enter(now, ds);
    (out, end)
}

/// An item the head sent: when, its model times, and what the head has
/// heard of it since: its own device's start and finish, and the last
/// stage's finish.
#[derive(Clone, Copy, Debug)]
pub struct Sent {
    pub item: u64,
    pub issued: f64,
    pub model: Times,
    pub start: Option<f64>,
    pub done: Option<f64>,
    pub exit: Option<f64>,
}

/// The frontier and the answer times of every item in `sent` as the head
/// places them at `now`: each time it heard stands, the rest come from the
/// model behind the items before, a stage it hears of never earlier than
/// `now`, and an item whose exit it heard is pulled back through the
/// stages it did not hear to fit that exit.
pub fn frontier(sent: &VecDeque<Sent>, stages: usize, now: f64) -> (Pipe, Vec<f64>) {
    let (m, last) = (sent.len(), stages - 1);
    let mut es = vec![[0.0; MAX_STAGES]; m];
    let mut ed = vec![[0.0; MAX_STAGES]; m];
    for (k, h) in sent.iter().enumerate() {
        let heard = |s: usize| match s {
            _ if s == last => (if s == 0 { h.start } else { None }, h.exit, now - REPLY),
            0 => (h.start, h.done, now),
            _ => (None, None, f64::NEG_INFINITY),
        };
        for s in 0..stages {
            let (start, done, upto) = heard(s);
            (es[k][s], ed[k][s]) = match (start, done) {
                (start, Some(d)) => (start.unwrap_or(d - h.model[s]), d),
                (Some(t), None) => (t, max(upto, t + h.model[s])),
                (None, None) => {
                    let mut c = if s == 0 { max(now, h.issued) } else { f64::NEG_INFINITY };
                    if k > 0 {
                        c = max(c, ed[k - 1][s]);
                    }
                    if s > 0 {
                        c = max(c, ed[k][s - 1] + HOP);
                    }
                    if s + 1 < stages && k >= 2 {
                        c = max(c, es[k - 2][s + 1]);
                    }
                    (c, max(c + h.model[s], upto))
                }
            };
        }
        if h.exit.is_some() {
            for s in (0..last).rev().filter(|&s| heard(s).1.is_none()) {
                ed[k][s] = min(ed[k][s], es[k][s + 1] - HOP);
                es[k][s] = min(es[k][s], ed[k][s] - h.model[s]);
            }
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
            let top = n.div_euclid(PAGE);
            let fits = |pages: i64| pages == 0 || !over(&item.with(p, pages * PAGE));
            rows += PAGE
                * if m.rises(context) {
                    from(guess(m.room(&item, p, budget), top), top, fits)
                } else {
                    bisect(top, fits)
                };
            break;
        }
        item = with;
        count += 1;
        rows += n;
        ctx += p + n;
    }
    rows.max(PAGE)
}

/// The most pages of `0..=top` that fit, where no more fit past the first
/// that does not, found from `guess`.
fn from(guess: i64, top: i64, fits: impl Fn(i64) -> bool) -> i64 {
    let up = (guess..top).find(|&g| !fits(g + 1)).unwrap_or(top);
    (0..=up).rev().find(|&g| fits(g)).expect("no pages fit")
}

fn guess(rows: f64, top: i64) -> i64 {
    ((rows / PAGE as f64) as i64).clamp(0, top)
}

/// The most pages of `0..=top` that fit, as a bisection finds them.
fn bisect(top: i64, fits: impl Fn(i64) -> bool) -> i64 {
    let (mut lo, mut hi) = (0, top);
    while lo < hi {
        let mid = (lo + hi + 1).div_euclid(2);
        if fits(mid) {
            lo = mid
        } else {
            hi = mid - 1
        }
    }
    lo
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
    let mut turn = Turn::of(&seqs, stages);
    let (mut ride, mut last, mut busy) = (0.0, None::<(f64, f64)>, 0.0);
    loop {
        let c = rule.cap(&seqs, &turn, m, context);
        let (plan, _) = pack(&seqs, &turn, c, context);
        if plan.is_empty() {
            break;
        }
        let go = if answers.len() < window { now } else { max(now, answers[answers.len() - window]) };
        let (item, finished) = turn.send(&mut seqs, &plan);
        let ds = m.of(&item);
        busy += ds[..stages].iter().fold(0.0, |a: f64, &b| a.max(b));
        let end = pipe.enter(go, &ds);
        if let Some((g, e)) = last {
            ride += (go - g) * (e - g) - (go - g) * (go - g) / 2.0;
        }
        last = Some((go, end));
        answers.push(end + REPLY);
        total += (end + REPLY - now) * finished as f64;
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
