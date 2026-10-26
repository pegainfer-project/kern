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
//! dropping the rank's coldest entry; no host tier yet.
//!
//! What this module is not: the group scheduler (`scheduler.rs`), which
//! runs one batch across a group in lockstep rows and steps a decode; or
//! the pipeline head (`pp/head.rs`), one rank packing for stages behind
//! it. Both stay for the shapes they serve.

use std::collections::{BTreeMap, VecDeque};
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, ensure, Result};
use kern_manifest::protocol::{Axis, Forward, Rows, Sampling, Values};
use kern_manifest::types::Fill;
use kern_manifest::Protocol;
use kern_pool::{Checkpoint, Denied, Evicted, Found, Lease, Parked, Pool, Prefix, Remap};
use pegainfer_frontend::engine::{
    FinishReason, QueuedRequest, RejectReason, RequestId, RequestLedger, Scheduler, SchedulerMetrics,
};
use tracing::{debug, info, warn};

use crate::executor::{Executors, Joined};
use crate::logline;
use crate::pack::{self, Bounds, Waiting};
use crate::pp::Item;
use crate::ranks::{Act, Answer, Mapping, Order};
use crate::scheduler::Facts;

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
    passed: usize,
    stop: pegainfer_frontend::engine::StopPolicy,
    sampling: Sampling,
    admitted: Instant,
}

impl Seq {
    fn left(&self) -> usize {
        self.ids.len() - self.pos
    }
}

/// One rank: its pool, its kept prefixes, its resident prompts and what
/// its pool decided since the last order.
struct Rank {
    pool: Arc<Pool>,
    /// A page and a slot no prompt owns: the row an idle rank feeds.
    pad: Lease,
    prefix: Prefix<Checkpoint, Parked>,
    seqs: Vec<Seq>,
    acts: Vec<Act>,
    /// The remap sent and not yet landed.
    remap: Option<Remap>,
    /// The line tables the rank's buffers hold; an item carries one only
    /// when its sequences change it.
    lines: BTreeMap<String, Vec<i64>>,
}

impl Rank {
    /// The rank's order: its queued acts, a remap its pool has pending, `item`.
    fn order(&mut self, item: Option<Item>) -> Order {
        if let Some(plan) = self.pool.take_pending() {
            self.acts.push(Act::Remap(Mapping::of(&plan)));
            self.remap = Some(plan);
        }
        Order { acts: std::mem::take(&mut self.acts), item, wait: self.remap.is_some() }
    }

    /// What the rank answered: a remap it landed is complete.
    fn heard(&mut self, q: usize, a: &Answer) -> Result<()> {
        match (a.remaps, self.remap.take()) {
            (0, r) => self.remap = r,
            (1, Some(plan)) => self.pool.complete(plan),
            (k, r) => bail!("rank {q} landed {k} remaps, {} in flight", r.is_some() as u8),
        }
        Ok(())
    }

    fn backlog(&self) -> usize {
        self.seqs.iter().map(Seq::left).sum()
    }

    fn max_seq_tokens(&self) -> usize {
        self.pool.max_seq_tokens().min((self.pool.pages_max() - self.pad.pages()) * self.pool.unit() as usize)
    }

    /// Drop the coldest kept prefix; `false` when none is kept.
    fn make_room(&mut self) -> bool {
        let park = None::<fn(Checkpoint) -> std::result::Result<std::result::Result<Parked, Checkpoint>, Infallible>>;
        match self.prefix.evict(park).unwrap_or_else(|e| match e {}) {
            Some(Evicted::Dropped { key, .. }) => {
                debug!(tokens = key.len(), "dropped");
                true
            }
            Some(Evicted::Parked { .. }) => unreachable!("no host tier parks"),
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
        let Joined { protocol: p, groups, pools, host, exec } = j;
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
        ensure!(host == (0, 0), "the prefill tier has no host tier yet");
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
                ensure!(pool.used() < pool.total(), "rank {q}: the pool holds one page; nothing left to serve from");
                let acts = pad.seq_slot().map(Act::Zero).into_iter().collect();
                Ok(Rank {
                    pool,
                    pad,
                    prefix: Prefix::new(unit),
                    seqs: Vec::new(),
                    acts,
                    remap: None,
                    lines: BTreeMap::new(),
                })
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
            "prefill tier ready"
        );
        Ok(d)
    }

    pub fn facts(&self) -> Facts {
        Facts {
            total_blocks: self.ranks.iter().map(|r| r.pool.total() - r.pad.pages()).sum(),
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
            let q_at = match hit {
                Some((_, q)) if self.ranks[q].seqs.len() >= seqs => break,
                Some((_, q)) => q,
                None => match self
                    .ranks
                    .iter()
                    .enumerate()
                    .filter(|(_, r)| r.seqs.len() < seqs)
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
                    Some(h) => match h.found {
                        Found::Resident(cp) => r.pool.restore(&cp, h.len, prompt).map(|(l, c)| (l, h.len, Some(c))),
                        Found::Parked(_) => unreachable!("no host tier parks"),
                    },
                    None => r.pool.lease(prompt).map(|l| (l, 0, None)),
                };
                match attempt {
                    Err(Denied::Busy) if r.make_room() => self.stats.evictions += 1,
                    a => break a,
                }
            };
            let (lease, start, copies) = match got {
                Ok(x) => x,
                Err(Denied::Busy | Denied::Remapping) => break,
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
                Err(d @ Denied::HostFull) => bail!("a lease denied {d}"),
            };
            match copies {
                Some(c) => r.acts.push(Act::Copy(c.into())),
                None => r.acts.extend(lease.seq_slot().map(Act::Zero)),
            }
            let q = self.waiting.pop_front().expect("peeked");
            if !q.request.params.is_greedy() && !self.p.samples() && !self.warned_sampling {
                warn!(request = %id, "non-greedy sampling params, but this manifest takes no sampling fills: served greedily; further requests are not warned about");
                self.warned_sampling = true;
            }
            ledger.admit(id);
            ledger.set_cached_tokens(id, start);
            if start > 0 {
                self.stats.hits += 1;
                self.stats.hit_tokens += start as u64;
            }
            debug!(request = %id, prompt, rank = q_at, prefix_hit = start, "admitted");
            r.seqs.push(Seq {
                id,
                lease,
                ids,
                pos: start,
                passed: 0,
                stop: q.request.stop_policy.clone(),
                sampling: crate::ranks::sampling(&q.request.params),
                admitted: Instant::now(),
            });
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
            // Nothing to feed: only what the pools decided, and a remap
            // on its way waited for, so the lease it serves can be retried.
            if self.ranks.iter().any(|r| !r.acts.is_empty() || r.remap.is_some() || r.pool.remapping()) {
                let orders = self.ranks.iter_mut().map(|r| r.order(None)).collect();
                let answers = self.exec.run(orders)?;
                for (q, (r, a)) in self.ranks.iter_mut().zip(&answers).enumerate() {
                    r.heard(q, a)?;
                }
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
        for (q, (r, a)) in self.ranks.iter_mut().zip(&answers).enumerate() {
            r.heard(q, a)?;
        }
        let emits = &self.p.fills[self.forward.emits.expect("checked at construction")];
        let width = emits.width as usize;
        self.stats.steps += 1;
        self.stats.step_ns += t0.elapsed().as_nanos();
        for ((r, plan), a) in self.ranks.iter_mut().zip(&plans).zip(&answers) {
            let mut done = Vec::new();
            for (k, &(i, n)) in plan.iter().enumerate() {
                let s = &mut r.seqs[i];
                s.pos += n;
                self.stats.rows += n as u64;
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
                let cause = s.stop.classify(tok, |_| false);
                ledger.push_tokens(s.id, &[tok], &[]);
                let why = if cause.is_some() { FinishReason::Stop } else { FinishReason::Length };
                ledger.finish_with_cause(s.id, why, cause);
                self.stats.finished += 1;
                debug!(request = %s.id, prompt = s.ids.len(), elapsed_ms = logline::ms(s.admitted.elapsed()), "finished");
                r.prefix.insert(&s.ids, r.pool.retire(s.lease, s.pos));
            }
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
            // Pages and slots trade chunks: a kept prefix holds a slot, and a
            // slot made from free pages shrinks what the KV can hold.
            let (pages, slots) = self.ranks.iter().fold(((0, 0), (0, 0)), |((pu, pt), (su, st)), r| {
                (
                    (pu + r.pool.used() - r.pad.pages(), pt + r.pool.total() - r.pad.pages()),
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
                "stats"
            );
        }
        self.stats = Stats::new();
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
            kv_total_blocks: self.ranks.iter().map(|r| (r.pool.total() - r.pad.pages()) as u64).sum(),
            num_running_reqs: self.ranks.iter().map(|r| r.seqs.len() as u64).sum(),
            num_waiting_reqs: self.waiting.len() as u64,
            spec_decode: None,
            prefix_cache: None,
        }
    }
}
