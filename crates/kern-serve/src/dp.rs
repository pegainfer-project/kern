//! The prefill tier's scheduler: every rank its own batch, one step for all.
//!
//! A prefill-only manifest over an expert-parallel group (`ep`, no `tp`)
//! is data parallel: the ranks share nothing but the MoE's collective,
//! which asks only that every rank launch the same program once a step,
//! each with however many rows it has. So this scheduler keeps one
//! [`Pool`] per rank and, per step, packs each rank's resident prompts
//! into that rank's own call ([`pack`]: shortest remaining first, bounded
//! rows and sequences, nothing starves), stages each rank's inputs from
//! its own leases alone ([`kern_run::fills`]: real varlen, a `cu_seqlens`
//! from the sequences' lengths) and hands every rank its order at once
//! ([`Executors::run`]). A rank with nothing to feed runs one padding row
//! so the collective sees it; its vars are its own, as every rank's are.
//!
//! The one decision that spans ranks is placement, made once at
//! admission: a prompt whose prefix some rank kept goes to that rank (the
//! pages are there), any other to the rank with the fewest prompt rows
//! still to prefill, then the fewest pages in use. Step time is the
//! slowest rank's, so keeping the ranks' backlogs level is the whole
//! policy. A prompt leases its own length at admission, nothing more: the
//! completion's pages are a decode tier's business. Its last rows out, the
//! call's token for it is the answer, the request finishes, and its lease
//! retires into the rank's [`Prefix`] keyed by the prompt, where a longer
//! prompt extending it continues from. Room for a lease is made by
//! dropping the rank's coldest entry.
//!
//! With a pooled tier (`--pool-master`) a leased prompt is probed before
//! it runs, on its rank: the marks its frontend named and every page past
//! the device hit, in one batch; the longest usable is fetched into the
//! lease and the prompt starts past it, a fetch that failed puts the
//! lease back as leased and the prompt starts from the device hit. What
//! goes in: with a state, a snapshot at the frontend's parting point
//! (`history_tokens`), taken as the rows reach it and put from the rank,
//! held only until the put lands (the device keeps the prompt's end, as
//! it did, and a slot is scarce); without one, every whole page at
//! retirement. The acts ride the rank's orders and its answers bring the
//! landings, so a rank with nothing to feed still gets an order while an
//! act of its is out.
//!
//! What this module is not: the group scheduler (`scheduler.rs`), which
//! runs one batch across a group in lockstep rows and steps a decode; or
//! the pipeline head (`pp/head.rs`), one rank packing for stages behind
//! it. Both stay for the shapes they serve.

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, ensure, Result};
use infront_engine::{
    FinishReason, QueuedRequest, RejectReason, RequestId, RequestLedger, Scheduler, SchedulerMetrics,
};
use kern_manifest::protocol::{Axis, Forward, Rows, Sampling, Values};
use kern_manifest::types::Fill;
use kern_manifest::Protocol;
use kern_pool::{landed, objects, Chain, Checkpoint, Denied, Keys, Lease, Pool, Prefix, Probe};
use tracing::{debug, info, warn};

use crate::executor::{Executors, Joined};
use crate::logline;
use crate::pack::{self, Bounds, Waiting};
use crate::pp::Item;
use crate::ranks::{Act, Answer, Landed, Order};
use crate::scheduler::{marks, Facts};

/// Items a sequence with rows left is passed over before it goes ahead
/// of every sequence passed over fewer times.
const PATIENCE: usize = 8;

pub struct Policy {
    /// Rows per rank per step: the manifest's `tokens` bound, or fewer.
    pub chunk: Option<usize>,
    /// Sequences per rank per step (≤ the manifest's `seqs` bound).
    pub max_seqs: usize,
}

/// A prompt resident on a rank: its lease, and how far it is fed.
struct Seq {
    id: RequestId,
    lease: Lease,
    ids: Vec<i64>,
    /// Rows in the state: the prefix it started from plus what was fed.
    pos: usize,
    /// Where its rows stop until the pooled tier has its snapshot there.
    mark: Option<usize>,
    passed: usize,
    stop: infront_engine::StopPolicy,
    sampling: Sampling,
    admitted: Instant,
}

impl Seq {
    fn left(&self) -> usize {
        self.mark.unwrap_or(self.ids.len()) - self.pos
    }
}

/// A leased prompt waiting on the pooled tier before it runs on its rank.
struct Pending {
    q: QueuedRequest,
    rank: usize,
    lease: Lease,
    /// The entry the lease continues from, the fallback's copies.
    hit: Option<Checkpoint>,
    ids: Vec<i64>,
    asked: Asked,
}

#[derive(Clone)]
enum Asked {
    Probing(Probe),
    /// Fetching `len` tokens, pages `first..` of them.
    Fetching {
        len: usize,
        first: usize,
    },
}

/// A snapshot being put, held so its pages and slot stay until it landed;
/// `mark` is written then, for one with a state.
struct Putting {
    cp: Checkpoint,
    len: usize,
    mark: Option<String>,
}

#[derive(Default)]
struct PoolCounts {
    probes: u64,
    hits: u64,
    fetched: u64,
    fetched_tokens: u64,
    fetch_fails: u64,
    puts: u64,
    put_fails: u64,
}

/// The pooled tier as the tier sees it: the namespace its keys are in
/// and every act awaiting its rank's answer.
struct Pooled {
    keys: Keys,
    next: u64,
    pending: BTreeMap<u64, Pending>,
    puts: BTreeMap<u64, Putting>,
    marks: BTreeMap<u64, Checkpoint>,
    counts: PoolCounts,
}

impl Pooled {
    fn id(&mut self) -> u64 {
        self.next += 1;
        self.next
    }

    fn busy(&self) -> bool {
        !self.pending.is_empty() || !self.puts.is_empty() || !self.marks.is_empty()
    }

    fn pending_on(&self, rank: usize) -> usize {
        self.pending.values().filter(|p| p.rank == rank).count()
    }
}

/// One rank: its pool, its kept prefixes, its resident prompts and what
/// its pool decided since the last order.
struct Rank {
    pool: Arc<Pool>,
    /// A page and a slot no prompt owns: the row an idle rank feeds.
    pad: Lease,
    prefix: Prefix,
    seqs: Vec<Seq>,
    acts: Vec<Act>,
    /// The line tables the rank's buffers hold; an item carries one only
    /// when its sequences change it.
    lines: BTreeMap<String, Vec<i64>>,
}

impl Rank {
    /// The rank's order: its queued acts and `item`.
    fn order(&mut self, item: Option<Item>) -> Order {
        Order { acts: std::mem::take(&mut self.acts), item }
    }

    fn backlog(&self) -> usize {
        self.seqs.iter().map(Seq::left).sum()
    }

    fn max_seq_tokens(&self) -> usize {
        self.pool.max_seq_tokens().min((self.pool.pages() - self.pad.pages()) * self.pool.unit() as usize)
    }

    /// Drop the coldest kept prefix; `false` when none is kept.
    fn make_room(&mut self) -> bool {
        match self.prefix.evict() {
            Some(key) => {
                debug!(tokens = key.len(), "dropped");
                true
            }
            None => false,
        }
    }
}

pub struct Dp {
    p: Protocol,
    exec: Executors,
    /// The program every rank runs each step: as many sequences as a
    /// rank may hold, of rows as fed.
    forward: Forward,
    bounds: Bounds,
    ranks: Vec<Rank>,
    waiting: VecDeque<QueuedRequest>,
    pool: Option<Pooled>,
    warned_sampling: bool,
    stats: Stats,
}

struct Stats {
    since: Instant,
    steps: u64,
    step_ns: u128,
    rows: u64,
    finished: u64,
    hits: u64,
    hit_tokens: u64,
    evictions: u64,
}

impl Stats {
    fn new() -> Stats {
        Stats {
            since: Instant::now(),
            steps: 0,
            step_ns: 0,
            rows: 0,
            finished: 0,
            hits: 0,
            hit_tokens: 0,
            evictions: 0,
        }
    }
}

/// Whether `p` is a prefill tier's protocol: a program taking sequences
/// of rows as fed that hands the first token back, and no program taking
/// a fixed number of rows to step with.
pub fn fits(p: &Protocol) -> bool {
    p.row_shapes().is_empty() && p.chunk().is_some_and(|f| f.emits.is_some())
}

impl Dp {
    pub fn new(j: Joined, policy: Policy) -> Result<Dp> {
        let Joined { protocol: p, groups, pools, exec, keys, .. } = j;
        ensure!(
            groups.size() == 1,
            "a prefill tier's ranks each run their own batch; this manifest's `tp` group has {} members",
            groups.size()
        );
        ensure!(
            p.ranks.is_none(),
            "a prefill tier's ranks each run their own batch; this manifest's batch spans a rank group"
        );
        ensure!(
            p.shards.positions.is_none(),
            "a prefill tier owns its rows; this manifest deals positions over a group"
        );
        let Some(chunk) = p.chunk().cloned() else {
            bail!("no program takes one sequence of as many rows as fed: nothing to feed a prompt through")
        };
        ensure!(
            chunk.emits.is_some(),
            "`{}` hands no token back; a prefill tier's call answers the first token",
            chunk.name
        );
        let max = p.rows.max as usize;
        let rows = match policy.chunk {
            None => max,
            Some(c) if (1..=max).contains(&c) => c,
            Some(c) => bail!("--chunk {c}: the manifest's `tokens` bound is {max}; a chunk can only be smaller"),
        };
        let seqs = policy.max_seqs.clamp(1, p.groups.max as usize);
        let Some(forward) = p.forward(seqs as u64, Rows::Var).cloned() else {
            bail!("no program takes {seqs} sequences of rows as fed; `{}` takes {}", chunk.name, chunk.groups)
        };
        let unit = pools.first().map_or(0, |pl| pl.unit() as usize);
        let ranks = pools
            .into_iter()
            .enumerate()
            .map(|(q, pool)| {
                let pool = Arc::new(pool);
                let pad = pool.lease(1).map_err(|d| anyhow::anyhow!("rank {q}: no page for the padding row: {d}"))?;
                ensure!(pool.used() < pool.pages(), "rank {q}: the pool holds one page; nothing left to serve from");
                let acts = pad.seq_slot().map(Act::Zero).into_iter().collect();
                Ok(Rank { pool, pad, prefix: Prefix::new(unit), seqs: Vec::new(), acts, lines: BTreeMap::new() })
            })
            .collect::<Result<Vec<_>>>()?;
        let bounds = Bounds { rows, seqs, context: usize::MAX };
        let d = Dp {
            p,
            exec,
            forward,
            bounds,
            ranks,
            waiting: VecDeque::new(),
            pool: keys.map(|keys| Pooled {
                keys,
                next: 0,
                pending: BTreeMap::new(),
                puts: BTreeMap::new(),
                marks: BTreeMap::new(),
                counts: PoolCounts::default(),
            }),
            warned_sampling: false,
            stats: Stats::new(),
        };
        info!(
            ranks = d.ranks.len(),
            pages = d.facts().total_blocks,
            page = unit,
            rows_per_rank = rows,
            seqs_per_rank = seqs,
            max_request_tokens = d.facts().max_request_tokens,
            program = %d.forward.name,
            pooled = d.pool.is_some(),
            "prefill tier ready"
        );
        Ok(d)
    }

    pub fn facts(&self) -> Facts {
        Facts {
            total_blocks: self.ranks.iter().map(|r| r.pool.pages() - r.pad.pages()).sum(),
            block_size: self.ranks[0].pool.unit() as usize,
            max_request_tokens: self.ranks.iter().map(Rank::max_seq_tokens).min().unwrap_or(0),
        }
    }

    /// Admit waiting requests in order, each onto its rank; one that
    /// cannot be seated stops the scan.
    fn admit(&mut self, ledger: &mut RequestLedger) -> Result<()> {
        while let Some(q) = self.waiting.front() {
            let id = q.id;
            if ledger.is_aborted(id) {
                ledger.retire(id);
                self.waiting.pop_front();
                continue;
            }
            let prompt = q.request.prompt_tokens.len();
            let limit = self.facts().max_request_tokens;
            if prompt == 0 || prompt > limit {
                ledger.reject(
                    id,
                    RejectReason::ContextLength { prompt_tokens: prompt, max_tokens: q.request.max_tokens, limit },
                );
                self.waiting.pop_front();
                continue;
            }
            let ids: Vec<i64> = q.request.prompt_tokens.iter().map(|&t| t as i64).collect();
            let seqs = self.bounds.seqs;
            // The longest kept prefix over every rank names the rank; a miss
            // goes to the rank with the least left to prefill.
            let hit =
                self.ranks.iter_mut().enumerate().filter_map(|(q, r)| r.prefix.lookup(&ids).map(|h| (h.len, q))).max();
            let seated = |q: usize| self.ranks[q].seqs.len() + self.pool.as_ref().map_or(0, |p| p.pending_on(q));
            let q_at = match hit {
                Some((_, q)) if seated(q) >= seqs => break,
                Some((_, q)) => q,
                None => match self
                    .ranks
                    .iter()
                    .enumerate()
                    .filter(|&(q, _)| seated(q) < seqs)
                    .map(|(q, r)| (r.backlog(), r.pool.used(), q))
                    .min()
                {
                    Some((.., q)) => q,
                    None => break,
                },
            };
            let r = &mut self.ranks[q_at];
            let got = loop {
                let attempt = match r.prefix.lookup(&ids) {
                    Some(h) => r.pool.restore(&h.found, h.len, prompt).map(|(l, c)| (l, Some((c, h.found)))),
                    None => r.pool.lease(prompt).map(|l| (l, None)),
                };
                match attempt {
                    Err(Denied::Busy) if r.make_room() => self.stats.evictions += 1,
                    a => break a,
                }
            };
            let (lease, hit) = match got {
                Ok(x) => x,
                Err(Denied::Busy) => break,
                Err(Denied::ExceedsRow { limit }) => {
                    ledger.reject(
                        id,
                        RejectReason::ContextLength { prompt_tokens: prompt, max_tokens: q.request.max_tokens, limit },
                    );
                    self.waiting.pop_front();
                    continue;
                }
                Err(Denied::ExceedsPool) => {
                    ledger.reject(id, RejectReason::KvBudget { prompt_tokens: prompt, worst_case_tokens: prompt });
                    self.waiting.pop_front();
                    continue;
                }
            };
            let hit = match hit {
                Some((c, found)) => {
                    r.acts.push(Act::Copy(c.into()));
                    Some(found)
                }
                None => {
                    r.acts.extend(lease.seq_slot().map(Act::Zero));
                    None
                }
            };
            let start = lease.prefix();
            if start > 0 {
                self.stats.hits += 1;
                self.stats.hit_tokens += start as u64;
            }
            let q = self.waiting.pop_front().expect("peeked");
            if !q.request.params.is_greedy() && !self.p.samples() && !self.warned_sampling {
                warn!(request = %id, "non-greedy sampling params, but this manifest takes no sampling fills: served greedily; further requests are not warned about");
                self.warned_sampling = true;
            }
            let probe = self.pool.as_ref().and_then(|_| {
                let unit = self.ranks[q_at].pool.unit() as usize;
                Probe::plan(lease.seq_slot().is_some(), unit, start, &marks(&q.request), prompt, None)
            });
            match probe {
                Some(probe) => self.probe(Pending { q, rank: q_at, lease, hit, ids, asked: Asked::Probing(probe) }),
                None => self.activate(q, q_at, lease, ids, ledger),
            }
        }
        Ok(())
    }

    /// A leased prompt starts on its rank past its lease's prefix, unless
    /// its client left meanwhile: with a state, its rows stop at the
    /// frontend's parting point past the prefix for the pooled tier's
    /// snapshot.
    fn activate(&mut self, q: QueuedRequest, rank: usize, lease: Lease, ids: Vec<i64>, ledger: &mut RequestLedger) {
        let (id, prompt, start) = (q.id, ids.len(), lease.prefix());
        if ledger.is_aborted(id) {
            ledger.retire(id);
            return;
        }
        ledger.admit(id);
        ledger.set_cached_tokens(id, start);
        debug!(request = %id, prompt, rank, prefix_hit = start, "admitted");
        let stateful = lease.seq_slot().is_some();
        let mark = self.pool.as_ref().and(q.request.history_tokens).filter(|&m| stateful && m > start && m < prompt);
        self.ranks[rank].seqs.push(Seq {
            id,
            lease,
            ids,
            pos: start,
            mark,
            passed: 0,
            stop: q.request.stop_policy.clone(),
            sampling: crate::ranks::sampling(&q.request.params),
            admitted: Instant::now(),
        });
    }

    /// Ask the pooled tier, through `p`'s rank, what it holds of the
    /// prompt past the device hit.
    fn probe(&mut self, p: Pending) {
        let pool = self.pool.as_mut().expect("probing with a pool");
        let Asked::Probing(probe) = &p.asked else { unreachable!("a pending prompt starts probing") };
        let unit = self.ranks[p.rank].pool.unit() as usize;
        let keys = probe.keys(&pool.keys, &Chain::of(unit, &p.ids), &p.ids);
        debug!(request = %p.q.id, rank = p.rank, device_hit = p.lease.prefix(), marks = ?probe.marks, pages = ?probe.pages, "probing");
        let id = pool.id();
        self.ranks[p.rank].acts.push(Act::Probe { id, keys });
        pool.pending.insert(id, p);
    }

    /// Fetch `len` tokens of `p`'s prompt into its lease, pages `first..`.
    fn fetch(&mut self, p: Pending, len: usize, first: usize) {
        let pool = self.pool.as_mut().expect("fetching with a pool");
        let unit = self.ranks[p.rank].pool.unit() as usize;
        let chain = Chain::of(unit, &p.ids[..len]);
        let objs =
            objects(&pool.keys, None, &chain, &p.ids[..len], first, 0, 0, p.lease.page_ids(), p.lease.seq_slot());
        debug!(request = %p.q.id, rank = p.rank, device_hit = p.lease.prefix(), len, "fetching");
        let id = pool.id();
        self.ranks[p.rank].acts.push(Act::Get { id, objects: objs });
        pool.pending.insert(id, Pending { asked: Asked::Fetching { len, first }, ..p });
    }

    /// Put `cp`, which holds `tokens`, from rank `q`: every whole page and,
    /// with a state, the state after them, marked once landed.
    fn put(&mut self, q: usize, cp: Checkpoint, tokens: &[i64]) {
        let Some(pool) = self.pool.as_mut() else { return };
        let unit = self.ranks[q].pool.unit() as usize;
        let chain = Chain::of(unit, tokens);
        let objs = objects(&pool.keys, None, &chain, tokens, 1, 0, 0, &cp.page_ids(), cp.seq_slot());
        let whole = tokens.len() / unit;
        let mark = cp.seq_slot().map(|_| pool.keys.mark(chain.at(whole), &tokens[whole * unit..]));
        let id = pool.id();
        self.ranks[q].acts.push(Act::Put { id, objects: objs });
        pool.puts.insert(id, Putting { cp, len: tokens.len(), mark });
    }

    /// What the ranks answered for the pooled tier: a probe's choice
    /// becomes a fetch, a fetch landed starts its prompt past what landed
    /// (or restores the lease and starts it from the device hit), a put
    /// landed is marked.
    fn heard(&mut self, ledger: &mut RequestLedger, answers: &[Answer]) -> Result<()> {
        for (q, a) in answers.iter().enumerate() {
            for l in &a.landed {
                self.landed(ledger, q, l)?;
            }
        }
        Ok(())
    }

    fn landed(&mut self, ledger: &mut RequestLedger, q: usize, l: &Landed) -> Result<()> {
        let Some(pool) = self.pool.as_mut() else {
            bail!("rank {q} answered pooled act {}, and no pooled tier was joined", l.id)
        };
        if let Some(mut p) = pool.pending.remove(&l.id) {
            ensure!(p.rank == q, "rank {q} answered act {} of rank {}", l.id, p.rank);
            match p.asked.clone() {
                Asked::Probing(probe) => {
                    pool.counts.probes += 1;
                    match probe.chosen(&l.ok) {
                        Some(len) => {
                            pool.counts.hits += 1;
                            self.fetch(p, len, probe.pages.start);
                        }
                        None => {
                            debug!(request = %p.q.id, "pool miss");
                            self.activate(p.q, p.rank, p.lease, p.ids, ledger);
                        }
                    }
                }
                Asked::Fetching { len, first } => {
                    let unit = self.ranks[q].pool.unit() as usize;
                    let got = landed(&l.ok, p.lease.seq_slot().is_some(), len, first, unit);
                    let device = p.lease.prefix();
                    if got > device {
                        pool.counts.fetched += 1;
                        pool.counts.fetched_tokens += (got - device) as u64;
                        debug!(request = %p.q.id, device_hit = device, fetched = got, "pool hit");
                        p.lease.filled(got);
                    } else {
                        pool.counts.fetch_fails += 1;
                        debug!(request = %p.q.id, device_hit = device, asked = len, "pool fetch did not land");
                        let r = &mut self.ranks[q];
                        match &p.hit {
                            Some(h) => r.acts.push(Act::Copy(p.lease.copies_from(h).into())),
                            None => r.acts.extend(p.lease.seq_slot().map(Act::Zero)),
                        }
                    }
                    self.activate(p.q, p.rank, p.lease, p.ids, ledger);
                }
            }
        } else if let Some(put) = pool.puts.remove(&l.id) {
            match l.ok.iter().all(|&b| b) {
                true => {
                    pool.counts.puts += 1;
                    if let Some(key) = put.mark {
                        let id = pool.id();
                        self.ranks[q].acts.push(Act::Mark { id, key });
                        pool.marks.insert(id, put.cp);
                    }
                }
                false => {
                    pool.counts.put_fails += 1;
                    debug!(tokens = put.len, "pool put did not land");
                }
            }
        } else if pool.marks.remove(&l.id).is_some() {
            pool.counts.put_fails += u64::from(!l.ok.iter().all(|&b| b));
        } else {
            bail!("rank {q}: an answer to pooled act {} nobody asked", l.id);
        }
        Ok(())
    }

    /// Drop aborted prompts before a step so they neither feed nor wait.
    fn drop_aborted(&mut self, ledger: &mut RequestLedger) {
        for r in &mut self.ranks {
            r.seqs.retain(|s| {
                let gone = ledger.is_aborted(s.id);
                if gone {
                    ledger.retire(s.id);
                }
                !gone
            });
        }
    }

    /// Rank `q`'s item for `plan` (sequence index, rows), or its padding
    /// row when the plan is empty.
    fn item(&mut self, q: usize, plan: &[(usize, usize)]) -> Result<Item> {
        let (p, r) = (&self.p, &mut self.ranks[q]);
        let parts: Vec<(&Lease, usize, &[i64], Sampling)> = match plan {
            [] => vec![(&r.pad, 0, &[0][..], Sampling::GREEDY)],
            plan => plan
                .iter()
                .map(|&(i, n)| {
                    let s = &r.seqs[i];
                    (&s.lease, s.pos, &s.ids[s.pos..s.pos + n], s.sampling)
                })
                .collect(),
        };
        let leases: Vec<&Lease> = parts.iter().map(|&(l, ..)| l).collect();
        let positions: Vec<usize> = parts.iter().map(|&(_, pos, ..)| pos).collect();
        let lens: Vec<usize> = parts.iter().map(|(_, _, ids, _)| ids.len()).collect();
        let ids: Vec<i64> = parts.iter().flat_map(|(_, _, ids, _)| ids.iter().copied()).collect();
        let samplings: Vec<Sampling> = parts.iter().map(|&(.., s)| s).collect();
        let (vars, values) = kern_run::fills(p, &leases, &positions, &lens, &ids, &samplings);
        let mut rows: BTreeMap<String, Values> = values
            .into_iter()
            .map(|(f, v)| match (plan.is_empty(), f.fill) {
                (true, Fill::Valid) => (f.name.clone(), Values::Ints(vec![0; v.len()])),
                _ => (f.name.clone(), v),
            })
            .collect();
        for t in &p.page_tables {
            let mut v = Vec::with_capacity(t.width * leases.len());
            for l in &leases {
                l.extend_row(&t.name, &mut v)?;
            }
            rows.insert(t.name.clone(), Values::Ints(v.into_iter().map(i64::from).collect()));
        }
        for t in &p.line_tables {
            let cols = match t.axis {
                Axis::Ranks => p.ranks.as_ref().map_or(1, |b| b.max),
                _ => p.groups.max,
            };
            let v: Vec<i64> = kern_run::line_rows(t, &leases, cols as usize)?.into_iter().map(i64::from).collect();
            if r.lines.get(&t.name) != Some(&v) {
                r.lines.insert(t.name.clone(), v.clone());
                rows.insert(t.name.clone(), Values::Ints(v));
            }
        }
        Ok(Item { program: self.forward.name.clone(), vars, rows, acts: Vec::new() })
    }

    /// One step: every rank packs its own call, all run together, and the
    /// prompts whose last rows went in get their token.
    fn step(&mut self, ledger: &mut RequestLedger) -> Result<()> {
        if self.ranks.iter().all(|r| r.seqs.is_empty()) {
            // Nothing to feed; the acts queued ride with the next item,
            // unless the pooled tier owes an answer: then they go now, and
            // its answers come back the same way.
            if self.pool.as_ref().is_some_and(Pooled::busy) {
                let orders = self.ranks.iter_mut().map(|r| r.order(None)).collect();
                let answers = self.exec.run(orders)?;
                self.heard(ledger, &answers)?;
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            return Ok(());
        }
        let t0 = Instant::now();
        let plans: Vec<Vec<(usize, usize)>> = self
            .ranks
            .iter_mut()
            .map(|r| {
                let w: Vec<Waiting> =
                    r.seqs.iter().map(|s| Waiting { pos: s.pos, left: s.left(), passed: s.passed }).collect();
                let plan = pack::pack(&w, self.bounds, PATIENCE);
                for (s, passed) in r.seqs.iter_mut().zip(pack::passed(&w, &plan)) {
                    s.passed = passed;
                }
                plan
            })
            .collect();
        let mut orders = Vec::with_capacity(self.ranks.len());
        for (q, plan) in plans.iter().enumerate() {
            let item = self.item(q, plan)?;
            orders.push(self.ranks[q].order(Some(item)));
        }
        let answers = self.exec.run(orders)?;
        ensure!(answers.len() == self.ranks.len(), "{} answers from {} ranks", answers.len(), self.ranks.len());
        self.stats.steps += 1;
        self.stats.step_ns += t0.elapsed().as_nanos();
        self.heard(ledger, &answers)?;
        let emits = &self.p.fills[self.forward.emits.expect("checked at construction")];
        let width = emits.width as usize;
        let mut puts = Vec::new();
        for (q, ((r, plan), a)) in self.ranks.iter_mut().zip(&plans).zip(&answers).enumerate() {
            let mut done = Vec::new();
            for (k, &(i, n)) in plan.iter().enumerate() {
                let s = &mut r.seqs[i];
                s.pos += n;
                self.stats.rows += n as u64;
                if s.mark == Some(s.pos) {
                    s.mark = None;
                    // The snapshot for the pool alone: a rank's slots are
                    // few, and the device keeps the prompt's end. Room is
                    // made for it as for a lease: a kept entry goes first.
                    let got = loop {
                        match r.pool.checkpoint(&mut s.lease, s.pos) {
                            Err(Denied::Busy) if r.prefix.evict().is_some() => self.stats.evictions += 1,
                            got => break got,
                        }
                    };
                    match got {
                        Ok((cp, copies)) => {
                            r.acts.push(Act::Copy(copies.into()));
                            puts.push((q, cp, s.ids[..s.pos].to_vec()));
                        }
                        Err(d) => debug!(request = %s.id, tokens = s.pos, "no snapshot for the pool: {d}"),
                    }
                }
                if s.pos == s.ids.len() {
                    let Some(&tok) = a.tokens.get(k * width) else {
                        bail!("rank: `{}` holds no token for sequence {k} of {}", emits.name, plan.len())
                    };
                    done.push((i, tok as u32));
                }
            }
            // Highest index first, so the removals do not move the rest.
            done.sort_unstable_by_key(|&(i, _)| std::cmp::Reverse(i));
            for (i, tok) in done {
                let s = r.seqs.swap_remove(i);
                let why = s.stop.classify(tok, |_| false).map_or(FinishReason::Length, FinishReason::Stop);
                ledger.push_tokens(s.id, &[tok], &[]);
                ledger.finish(s.id, why);
                self.stats.finished += 1;
                debug!(request = %s.id, prompt = s.ids.len(), elapsed_ms = logline::ms(s.admitted.elapsed()), "finished");
                let cp = r.pool.retire(s.lease, s.pos);
                r.prefix.insert(&s.ids, cp.clone());
                if cp.seq_slot().is_none() {
                    puts.push((q, cp, s.ids));
                }
            }
        }
        for (q, cp, tokens) in puts {
            self.put(q, cp, &tokens);
        }
        Ok(())
    }

    fn log_stats(&mut self) {
        let st = &self.stats;
        let dt = st.since.elapsed();
        if dt.as_secs() < 5 {
            return;
        }
        if st.steps > 0 {
            let round = |x: f64, d: f64| (x * d).round() / d;
            let (pages, slots) = self.ranks.iter().fold(((0, 0), (0, 0)), |((pu, pt), (su, st)), r| {
                (
                    (pu + r.pool.used() - r.pad.pages(), pt + r.pool.pages() - r.pad.pages()),
                    (su + r.pool.slots_used(), st + r.pool.slots()),
                )
            });
            info!(
                running = self.ranks.iter().map(|r| r.seqs.len()).sum::<usize>(),
                backlog_per_rank = ?self.ranks.iter().map(Rank::backlog).collect::<Vec<_>>(),
                waiting = self.waiting.len(),
                pages = format_args!("{}/{}", pages.0, pages.1),
                slots = format_args!("{}/{}", slots.0, slots.1),
                steps = st.steps,
                step_ms = round(st.step_ns as f64 / 1e6 / st.steps as f64, 100.0),
                rows = st.rows,
                rows_per_s = round(st.rows as f64 / (st.step_ns as f64 / 1e9).max(1e-9), 1.0),
                finished = st.finished,
                hits = st.hits,
                hit_tokens = st.hit_tokens,
                evictions = st.evictions,
                checkpoints = self.ranks.iter().map(|r| r.prefix.entries()).sum::<usize>(),
                pool_pending = self.pool.as_ref().map(|p| p.pending.len()),
                pool_probes = self.pool.as_ref().map(|p| p.counts.probes),
                pool_hits = self.pool.as_ref().map(|p| p.counts.hits),
                pool_fetched = self.pool.as_ref().map(|p| p.counts.fetched),
                pool_fetched_tokens = self.pool.as_ref().map(|p| p.counts.fetched_tokens),
                pool_fetch_fails = self.pool.as_ref().map(|p| p.counts.fetch_fails),
                pool_puts = self.pool.as_ref().map(|p| p.counts.puts),
                pool_put_fails = self.pool.as_ref().map(|p| p.counts.put_fails),
                "stats"
            );
        }
        self.stats = Stats::new();
        if let Some(p) = &mut self.pool {
            p.counts = PoolCounts::default();
        }
    }
}

impl Scheduler for Dp {
    fn submit(&mut self, request: QueuedRequest) {
        self.waiting.push_back(request);
    }

    fn step(&mut self, ledger: &mut RequestLedger) -> Result<()> {
        self.drop_aborted(ledger);
        self.admit(ledger)?;
        self.step(ledger)?;
        self.log_stats();
        Ok(())
    }

    fn metrics(&self) -> SchedulerMetrics {
        SchedulerMetrics {
            kv_used_blocks: self.ranks.iter().map(|r| (r.pool.used() - r.pad.pages()) as u64).sum(),
            kv_total_blocks: self.ranks.iter().map(|r| (r.pool.pages() - r.pad.pages()) as u64).sum(),
            num_running_reqs: self.ranks.iter().map(|r| r.seqs.len() as u64).sum(),
            num_waiting_reqs: self.waiting.len() as u64,
            spec_decode: None,
            prefix_cache: None,
        }
    }
}
