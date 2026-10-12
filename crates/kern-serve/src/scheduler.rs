//! The kern scheduler: every rank, many sequences.
//!
//! Implements the pegainfer frontend's [`Scheduler`] contract — `submit`,
//! `step`, `metrics` — over a kern manifest whose every rank's accounting
//! is a [`Ranks`] (`ranks.rs` is the design: lockstep, owners, staging)
//! and whose ranks run what it decides through [`Executors`] (one process
//! per GPU, `executor.rs`). What the
//! manifest must speak is its [`Protocol`]: fills naming which buffer
//! carries the tokens, positions, slots and lengths of a call and which
//! output hands tokens back, and a `batch` on every program a step may
//! run — `groups` sequences of `rows` rows each. The scheduler picks a
//! forward by shape and never by name: a decode step and a speculative
//! round are the same motion — stage `rows` rows per sequence at its
//! position, run the forward that accepts `(b, rows)`, read the tokens it
//! hands each sequence, `count` of them when it says so — and differ only
//! in the rows the operator asked for (`--rows`, a shape the manifest
//! declares; the widest by default). A prompt goes through the forward
//! that takes one sequence of as many rows as fed, in chunks; when that
//! forward hands a token back (a hybrid GDN model, whose chunked kernels
//! must see every prompt token) every prompt token goes through it and
//! the first generated token comes from it, otherwise the last prompt
//! token is the first step's input.
//!
//! Policy, deliberately simple:
//! - prefill first: each step admits every waiting request it can seat,
//!   in order, and prefills each whole (one sequence, `chunk` tokens per
//!   call, launch by launch: no graph per chunk length), then runs one
//!   step over every running sequence;
//! - a request leases every KV page its worst case (`prompt + max_tokens`,
//!   plus `rows - 1` for the last step's rows past the end) needs at
//!   admission (`Ranks::lease`, on the rank with the fewest rows, then
//!   pages),
//!   so a step never runs out of pages and nothing is ever preempted; the
//!   row drops with the sequence;
//! - batches are padded up to a bucket size — the same bucket on every
//!   rank, from the most loaded one — and each bucket's forward is
//!   CUDA-graph-captured once; padding rows write into a page leased for
//!   them on each rank that nobody reads;
//! - greedy only: the manifest's `argmax` is the sampler. Non-greedy
//!   sampling params are logged once and served greedily;
//! - a speculative round is a forward of several rows per sequence whose
//!   `tokens` output is `[groups, rows]` and whose `count` output says how
//!   many of a sequence's the device accepted: the rows past the count
//!   land past the sequence's position and the next step overwrites them,
//!   the paged state's free rollback. Whether a round beats a plain step
//!   at a given batch size is the operator's call (`--rows`), not the
//!   scheduler's;
//! - prefix reuse: a finished sequence's KV lives on as snapshots
//!   (`Ranks::checkpoint` / `retire`, indexed by their tokens in a `Prefix`
//!   tree over every rank), and a new prompt starts from the
//!   longest snapshot holding a proper prefix of it (`Ranks::lease_from`;
//!   prefill covers the rest). A paged-only manifest checkpoints every
//!   whole page as a sequence fills it — free, a shared page — so any
//!   earlier prompt or output is reusable at page granularity. A recurrent
//!   state cannot roll back, so a manifest with one checkpoints where its
//!   state is worth continuing and nowhere else. Once where the next
//!   turn's prompt parts from this one, `Request::history_tokens`: the
//!   frontend's chat layer measured where the rendered conversation ends
//!   and the template's generation prompt begins, since the next turn
//!   renders the reply as history and the generation prompt anew; the
//!   prefill stops there for it, and the page it ends inside and the
//!   state slots are copied into the snapshot's own. That is the turn's
//!   one snapshot: the next turn renders a reply after it (ours or a
//!   recorded one, as an agent trace replays) and pays for the reply and
//!   what follows it. Only a request with no such point is kept where it
//!   ends (the finished sequence's state slots become the snapshot's,
//!   nothing is copied), which a next turn continuing the whole context
//!   hits. The turns a conversation moved past stay until room is needed
//!   and go first (`Prefix`). A speculative round that
//!   accepted past `max_tokens` or the stop leaves a state no next turn
//!   continues, which is not kept. The prompt's snapshot is an
//!   optimization, never a reason to fail its request: `Busy` makes room
//!   as a lease does, `Remapping` waits for the slot being carved (this
//!   is the one place that waits: the state cannot be snapshotted later,
//!   and the wait is the remap's own milliseconds), and when nothing is
//!   left to make room with it is skipped. A `Busy` lease makes room and
//!   retries until it fits or
//!   nothing is left: the least recently hit snapshot is parked into the
//!   host tier (`--host-gib`: pinned DRAM per rank, `Ranks::park`, all of
//!   its members' pieces or none; the coldest parked ones are dropped when
//!   it is full) or, without one, dropped; a lease still short while a
//!   park's pages are on their way back waits for them rather than parking
//!   more. A prompt hitting a parked
//!   snapshot wakes it (`Ranks::wake`) straight into its own row: one
//!   lease sized for the request, the prefix's pages copied in over the
//!   transfer streams, so room is decided once (waking the prefix and then
//!   leasing from it would ask twice, and the second ask could park what
//!   the first woke, without end). The request waits in `waking` until
//!   `Ranks::awake` hands the row back and starts from it; no step queues
//!   behind them. With nothing running, a step only polls the ranks for
//!   what landed, and only while something is in flight. The parked snapshot stays parked; the request's own
//!   finish indexes the longer context resident. A replicated group
//!   (positions dealt over its members) keeps nothing: its checkpoints
//!   and retires answer `None` and the prefix tree stays empty.

use std::collections::VecDeque;
use std::time::Instant;

use anyhow::{bail, Result};
use kern_manifest::protocol::{Forward, Rows};
use kern_manifest::Protocol;
use kern_pool::{Denied, Evicted, Found, Prefix, Tier};
use pegainfer_frontend::engine::{
    FinishReason, QueuedRequest, RejectReason, RequestId, RequestLedger, Scheduler, SchedulerMetrics,
    SpecDecodeCounters, StopCause, StopPolicy, MAX_SPEC_TOKENS,
};
use tracing::{debug, info, warn};

use crate::executor::Executors;
use crate::logline;
use crate::ranks::{Cell, Ranks, Rising, Row, Sleeping, Snapshot, Staged};

/// Decode batch buckets; a batch is padded up to the smallest one that
/// fits and each bucket owns one captured graph.
const BUCKETS: [usize; 18] = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 640, 768, 1024];

pub(crate) fn bucket(n: usize) -> usize {
    BUCKETS.iter().copied().find(|&b| b >= n).unwrap_or(n)
}

pub struct Policy {
    /// Prefill chunk (tokens per `prefill` call): the manifest's `tokens`
    /// bound, or a smaller one.
    pub chunk: Option<usize>,
    /// Cap on concurrently running sequences per rank (≤ the manifest's
    /// `seqs` bound).
    pub max_seqs: usize,
    /// Rows per sequence of a step, a shape the manifest declares; the
    /// widest by default.
    pub rows: Option<u64>,
    /// Pinned host memory per rank for parked snapshots (0: none).
    pub host_bytes: u64,
}

/// How this process drives the manifest: the forwards a step and a
/// prompt go through and the batch they allow. Pure over the protocol and
/// the ranks' size, so a synthetic manifest exercises every rejection.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Plan {
    /// Rows per sequence of a step.
    rows: u64,
    /// The forward that takes one sequence of `rows` rows; a batch of `b`
    /// runs the one the protocol picks for `(b, rows)`.
    step: Forward,
    /// The forward a prompt chunk goes through.
    chunk: Forward,
    /// Concurrent sequences per rank.
    max_seqs: usize,
}

impl Plan {
    /// `page` is the ranks' page in tokens (the pad page a step's rows
    /// must fit), `seqs_max` its sequences per rank, `want_rows` and
    /// `want_seqs` the operator's asks.
    fn check(p: &Protocol, page: usize, seqs_max: usize, want_rows: Option<u64>, want_seqs: usize) -> Result<Plan> {
        let shapes = p.row_shapes();
        let Some(rows) = want_rows.or(shapes.last().copied()) else {
            bail!("no program takes a fixed number of rows per sequence: nothing to run a step with")
        };
        let Some(step) = p.forward(1, Rows::Const(rows)).cloned() else {
            bail!("no program takes one sequence of {rows} rows; the manifest declares rows {shapes:?}")
        };
        if rows as usize > page {
            bail!("{rows} rows per sequence per step exceed the {page}-token pad page");
        }
        if rows as usize - 1 > MAX_SPEC_TOKENS {
            bail!("{} rows past the first per step, the frontend's metrics hold {MAX_SPEC_TOKENS}", rows - 1);
        }
        let Some(chunk) = p.chunk().cloned() else {
            bail!("no program takes one sequence of as many rows as fed: nothing to feed a prompt through")
        };
        let max_seqs = want_seqs
            .clamp(1, seqs_max)
            .min(p.rows.max as usize / rows as usize)
            .min(p.max_groups(Rows::Const(rows)) as usize)
            .max(1);
        Ok(Plan { rows, step, chunk, max_seqs })
    }

    /// Rows a step writes past the token it feeds: what a lease holds past
    /// `prompt + max_tokens` so the last step's rows have slots.
    fn headroom(&self) -> usize {
        self.rows as usize - 1
    }

    /// The acceptance counters, for a plan whose steps take several rows.
    fn counters(&self) -> Option<SpecDecodeCounters> {
        (self.rows > 1).then(|| SpecDecodeCounters { num_spec_tokens: self.rows - 1, ..Default::default() })
    }
}

struct Seq {
    id: RequestId,
    /// Tokens already in the KV state.
    pos: usize,
    /// The token the next decode step feeds at `pos`.
    next: u32,
    /// Tokens counted against `max_tokens`, the stop token included; what
    /// a speculative round accepted past it or `max_tokens` is not.
    generated: usize,
    /// Tokens the client got as text: `generated` less the stop token.
    /// The state can hold more (a round's accepted tokens past the end),
    /// which is what `finish` checks against.
    emitted: usize,
    max_tokens: usize,
    /// The request's EOS and explicit stop ids, as the frontend lowered
    /// them (`ignore_eos` arrives as a policy without an EOS).
    stop: StopPolicy,
    /// Its KV pages and state slots across its group; returned when the
    /// sequence drops.
    row: Row,
    prompt_len: usize,
    /// The tokens in the state, `pos` of them; the prefix tree keys this
    /// sequence's snapshots by them.
    history: Vec<i64>,
    /// Tokens already checkpointed, a whole number of pages.
    checkpointed: usize,
    /// Its history ends before its prompt does, so its end is not kept.
    parts: bool,
    admitted: Instant,
}

impl Seq {
    /// `fed` went through a program and is in the state now.
    fn advance(&mut self, fed: impl IntoIterator<Item = u32>) {
        for t in fed {
            self.pos += 1;
            self.history.push(t as i64);
        }
    }

    /// Account `toks` as generated, as far as [`cut`] takes them. The stop
    /// token goes to the ledger with its cause (pegainfer convention: the
    /// frontend counts it as vLLM does and strips it from the text) but not
    /// into `emitted`. Finishes the request in the ledger when it is done,
    /// otherwise the last token is the next step's input. Returns how many
    /// tokens were emitted and whether the sequence finished.
    fn emit(&mut self, toks: &[u32], ledger: &mut RequestLedger) -> (u64, bool) {
        let (taken, end) = cut(toks, self.generated, self.max_tokens, &self.stop);
        let emitted = taken - matches!(end, Some(End::Stop(_))) as usize;
        self.generated += taken;
        self.emitted += emitted;
        if taken > 0 {
            ledger.push_tokens(self.id, &toks[..taken], &[]);
        }
        let done = match end {
            Some(end) => {
                let (reason, cause) = match end {
                    End::Stop(c) => (FinishReason::Stop, Some(c)),
                    End::Length => (FinishReason::Length, None),
                };
                debug!(
                    request = %self.id,
                    reason = ?reason,
                    prompt = self.prompt_len,
                    generated = self.generated,
                    elapsed_s = logline::secs(self.admitted.elapsed()),
                    "finished"
                );
                ledger.finish_with_cause(self.id, reason, cause);
                true
            }
            None => {
                if let Some(&t) = toks.last() {
                    self.next = t;
                }
                false
            }
        };
        (emitted as u64, done)
    }
}

/// Why a sequence's tokens end.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum End {
    Stop(StopCause),
    Length,
}

/// How many of `toks` a sequence that has generated `generated` of
/// `max_tokens` takes: through the first stop token or the one reaching
/// `max_tokens`, and why it ends there; all of them and `None` otherwise.
/// A stop token wins over `max_tokens` when it is both.
fn cut(toks: &[u32], generated: usize, max_tokens: usize, stop: &StopPolicy) -> (usize, Option<End>) {
    toks.iter()
        .enumerate()
        .find_map(|(i, &t)| match stop.classify(t, |_| false) {
            Some(c) => Some((i + 1, End::Stop(c))),
            None => (generated + i + 1 >= max_tokens).then_some((i + 1, End::Length)),
        })
        .map_or((toks.len(), None), |(n, end)| (n, Some(end)))
}

pub struct KernScheduler {
    ranks: Ranks,
    exec: Executors,
    policy: Policy,
    /// Tokens per `prefill` call, within the manifest's bound.
    chunk: usize,
    plan: Plan,
    /// Draft acceptance, for a plan whose steps take several rows.
    counters: Option<SpecDecodeCounters>,
    waiting: VecDeque<QueuedRequest>,
    /// Requests whose rows are still on the way in from the host tier.
    waking: VecDeque<(QueuedRequest, Rising)>,
    running: Vec<Seq>,
    /// Snapshots of finished prefixes, for the next prompt that shares
    /// one (see the module doc).
    prefix: Prefix<Snapshot, Sleeping>,
    /// Checkpoint at every page (a paged-only manifest) rather than only
    /// where a prompt's history and a request end (one with a recurrent state).
    every_page: bool,
    warned_sampling: bool,
    stats: Stats,
}

/// Rolling counters for the periodic log line, reset when it prints.
struct Stats {
    since: Instant,
    /// Steps and their time.
    steps: u64,
    step_ns: u128,
    /// Tokens emitted to the ledger.
    tokens: u64,
    /// Prompt tokens fed through the chunk forward, and their time.
    prefill_tokens: u64,
    prefill_ns: u128,
    /// Prompt tokens found in a snapshot, and snapshots evicted for room.
    prefix_hit_tokens: u64,
    evictions: u64,
    /// Requests whose prompt was found in a resident snapshot, and in one
    /// woken from the host, with the tokens found.
    resident_hits: u64,
    resident_hit_tokens: u64,
    host_hits: u64,
    host_hit_tokens: u64,
    /// Prompts checkpointed where their history ends (a recurrent state),
    /// and those skipped for want of room.
    prompt_checkpoints: u64,
    prompt_skips: u64,
    /// Snapshots parked to the host, dropped from it for room, woken
    /// from it, and the tokens woken.
    parks: u64,
    host_evictions: u64,
    wakes: u64,
    wake_tokens: u64,
    /// The acceptance counters at the window's start, so the window's
    /// acceptance is reported rather than the process's.
    spec_at: (u64, u64, u64),
}

impl Stats {
    fn new(counters: &Option<SpecDecodeCounters>) -> Stats {
        let spec_at =
            counters.as_ref().map_or((0, 0, 0), |c| (c.num_drafts, c.num_draft_tokens, c.num_accepted_tokens));
        Stats {
            since: Instant::now(),
            steps: 0,
            step_ns: 0,
            tokens: 0,
            prefill_tokens: 0,
            prefill_ns: 0,
            prefix_hit_tokens: 0,
            evictions: 0,
            resident_hits: 0,
            resident_hit_tokens: 0,
            host_hits: 0,
            host_hit_tokens: 0,
            prompt_checkpoints: 0,
            prompt_skips: 0,
            parks: 0,
            host_evictions: 0,
            wakes: 0,
            wake_tokens: 0,
            spec_at,
        }
    }
}

/// Public facts the frontend wants at launch.
pub struct Facts {
    pub total_blocks: usize,
    pub block_size: usize,
    /// Longest request (prompt + completion) one sequence can hold.
    pub max_request_tokens: usize,
}

/// What a lease attempt handed out.
enum Got {
    Row(Row),
    /// A row in flight from the host tier.
    Rising(Rising),
}

impl KernScheduler {
    /// Wrap the loaded ranks (weights bound, peers connected, pads
    /// leased): settle how the head drives the manifest, within its bounds.
    pub fn new(ranks: Ranks, exec: Executors, policy: Policy) -> Result<KernScheduler> {
        let plan = Plan::check(ranks.protocol(), ranks.page(), ranks.seqs_max(), policy.rows, policy.max_seqs)?;
        let max = ranks.protocol().rows.max as usize;
        let chunk = match policy.chunk {
            None => max,
            Some(c) if (1..=max).contains(&c) => c,
            Some(c) => bail!("--chunk {c}: the manifest's `tokens` bound is {max}; a chunk can only be smaller"),
        };
        let policy = Policy { max_seqs: plan.max_seqs, ..policy };
        let counters = plan.counters();
        let stats = Stats::new(&counters);
        let every_page = !ranks.has_seq_state();
        let prefix = Prefix::new(ranks.page());
        let s = KernScheduler {
            ranks,
            exec,
            policy,
            chunk,
            plan,
            counters,
            waiting: VecDeque::new(),
            waking: VecDeque::new(),
            running: Vec::new(),
            prefix,
            every_page,
            warned_sampling: false,
            stats,
        };
        s.log_ready();
        Ok(s)
    }

    /// What the frontend advertises at launch.
    pub fn facts(&self) -> Facts {
        // What admit() leases is `prompt + max_tokens + headroom`;
        // advertise the request-shaped remainder so the frontend clamps
        // `max_tokens` to something admissible instead of the scheduler
        // bouncing it (the wire turns a scheduler reject into a 500).
        Facts {
            total_blocks: self.ranks.pages_total(),
            block_size: self.ranks.page(),
            max_request_tokens: self.ranks.max_seq_tokens() - self.headroom(),
        }
    }

    fn headroom(&self) -> usize {
        self.plan.headroom()
    }

    fn log_ready(&self) {
        let (ranks, policy, plan, facts) = (&self.ranks, &self.policy, &self.plan, self.facts());
        let (slots, _) = ranks.seq_slots();
        // Pages: what requests can lease (one more per rank holds the
        // padding rows). Graphs are captured per bucket on first use
        // unless `eager`.
        info!(
            ranks = ranks.groups().ranks(),
            group = ranks.groups().size(),
            pages = facts.total_blocks,
            page = facts.block_size,
            seq_slots = (slots > 0).then_some(slots),
            max_seqs_per_rank = policy.max_seqs,
            max_request_tokens = facts.max_request_tokens,
            chunk = self.chunk,
            buckets = ?BUCKETS.iter().filter(|&&b| b <= policy.max_seqs).collect::<Vec<_>>(),
            rows = plan.rows,
            step = %plan.step.name,
            steps = ?ranks.protocol().forwards.iter().filter(|f| f.rows == plan.step.rows).map(|f| (f.name.as_str(), f.groups)).collect::<Vec<_>>(),
            prefill = match plan.chunk.emits {
                Some(_) => format!("`{}`, emits the first token", plan.chunk.name),
                None => format!("`{}`, state only", plan.chunk.name),
            },
            checkpoints = if self.every_page { "every page" } else { "at history end and request end" },
            host_gib_per_rank = (policy.host_bytes > 0).then_some(policy.host_bytes >> 30),
            "scheduler ready"
        );
    }

    /// Rows each rank has (running, and woken ones on the way in).
    fn rows_per_rank(&self) -> Vec<usize> {
        let mut rows = vec![0usize; self.ranks.groups().ranks()];
        for s in &self.running {
            rows[s.row.owner().index()] += 1;
        }
        for (_, r) in &self.waking {
            rows[r.owner().index()] += 1;
        }
        rows
    }

    /// Admit waiting requests in order and prefill each (single sequence,
    /// chunked); one that cannot be seated stops the scan.
    fn admit(&mut self, ledger: &mut RequestLedger) -> Result<()> {
        // Wakes land in order: the first still in flight stops the scan;
        // a landed one starts from its row ahead of the line.
        while let Some((q, r)) = self.waking.pop_front() {
            match self.ranks.awake(r) {
                Ok(row) => {
                    self.stats.wakes += 1;
                    self.stats.wake_tokens += row.prefix() as u64;
                    if ledger.is_aborted(q.id) {
                        ledger.retire(q.id);
                        continue;
                    }
                    self.admit_one(q, row, true, ledger)?;
                }
                Err(r) => {
                    self.waking.push_front((q, r));
                    break;
                }
            }
        }
        while let Some(q) = self.waiting.front() {
            let id = q.id;
            if ledger.is_aborted(id) {
                ledger.retire(id);
                self.waiting.pop_front();
                continue;
            }
            let cap = self.policy.max_seqs;
            let rows = self.rows_per_rank();
            if rows.iter().all(|&r| r >= cap) {
                break;
            }
            let prompt = q.request.prompt_tokens.len();
            let max_tokens = q.request.max_tokens;
            let worst = prompt + max_tokens + self.headroom();
            let limit = self.ranks.max_seq_tokens();
            if prompt == 0 || worst > limit {
                ledger.reject(id, RejectReason::ContextLength { prompt_tokens: prompt, max_tokens, limit });
                self.waiting.pop_front();
                continue;
            }
            let ids: Vec<i64> = q.request.prompt_tokens.iter().map(|&t| t as i64).collect();
            // A hit continues on the snapshot's owner, which must have a row free.
            let owner = self.prefix.lookup(&ids).map(|h| match h.found {
                Found::Resident(snap) => snap.owner(),
                Found::Parked(p) => p.owner(),
            });
            if owner.is_some_and(|o| rows[o.index()] >= cap) {
                break;
            }
            // Room made for a retry may have parked or dropped the
            // snapshot hit above: look it up again where it is now.
            let got = loop {
                let ranks = &mut self.ranks;
                let attempt = match self.prefix.lookup(&ids) {
                    Some(h) => match h.found {
                        Found::Resident(snap) => ranks.lease_from(&snap, h.len, worst).map(Got::Row),
                        Found::Parked(p) => ranks.wake(&p, h.len, worst).map(Got::Rising),
                    },
                    None => ranks.lease(worst, |r| Some(rows[r.index()]).filter(|&n| n < cap)).map(Got::Row),
                };
                match attempt {
                    Err(Denied::Busy) if self.ranks.parking() => self.settle()?,
                    Err(Denied::Busy) if self.make_room()? => {}
                    r => break r,
                }
            };
            let row = match got {
                Ok(Got::Row(row)) => row,
                Ok(Got::Rising(r)) => {
                    // Its copies are in flight; it starts once they land.
                    let q = self.waiting.pop_front().unwrap();
                    self.waking.push_back((q, r));
                    continue;
                }
                Err(Denied::Busy) => break,      // wait for pages / a slot
                Err(Denied::Remapping) => break, // pages or a slot are on their way
                Err(Denied::ExceedsRow { limit }) => {
                    ledger.reject(id, RejectReason::ContextLength { prompt_tokens: prompt, max_tokens, limit });
                    self.waiting.pop_front();
                    continue;
                }
                Err(Denied::ExceedsPool) => {
                    ledger.reject(id, RejectReason::KvBudget { prompt_tokens: prompt, worst_case_tokens: worst });
                    self.waiting.pop_front();
                    continue;
                }
                Err(d @ Denied::HostFull) => bail!("a lease denied {d}"),
            };
            let q = self.waiting.pop_front().unwrap();
            self.admit_one(q, row, false, ledger)?;
        }
        Ok(())
    }

    /// Start `q` running in `row` (past the row's prefix, `woken` from
    /// the host tier or resident): through the chunk forward when the
    /// manifest has one, else with the prompt queued for the steps.
    fn admit_one(&mut self, q: QueuedRequest, mut row: Row, woken: bool, ledger: &mut RequestLedger) -> Result<()> {
        let id = q.id;
        let prompt = q.request.prompt_tokens.len();
        let max_tokens = q.request.max_tokens;
        let ids: Vec<i64> = q.request.prompt_tokens.iter().map(|&t| t as i64).collect();
        if !q.request.params.is_greedy() && !self.warned_sampling {
            warn!(
                request = %id,
                temperature = q.request.params.temperature,
                top_p = q.request.params.top_p,
                top_k = q.request.params.top_k,
                "non-greedy sampling params; this engine samples greedily (argmax in the manifest); further requests are not warned about"
            );
            self.warned_sampling = true;
        }
        ledger.admit(id);
        let t0 = Instant::now();
        let start = row.prefix();
        // The prompt tokens this request did not pay for: its `usage` says so
        // (`prompt_tokens_details.cached_tokens`), zero included, so a caller
        // can tell a miss from an engine that does not report hits.
        ledger.set_cached_tokens(id, start);
        // Every prompt token goes through the chunk forward when it hands
        // the first generated token back itself; otherwise everything but
        // the last, which is the first step's input. A snapshot hit skips
        // its tokens (never the last one). A recurrent state is kept where
        // the frontend says the next turn parts from this prompt, so the
        // prefill stops there once.
        let parts = !self.every_page && q.request.history_tokens.is_some_and(|m| m < prompt);
        let f = self.plan.chunk.clone();
        let n_pre = if f.emits.is_some() { prompt } else { prompt - 1 };
        let mark =
            (!self.every_page).then_some(q.request.history_tokens).flatten().filter(|&m| m > start && m <= n_pre);
        let first = match mark {
            Some(m) => {
                let first = self.prefill(&f, &row, &ids[..m], start)?;
                self.keep_prompt(&mut row, &ids[..m], id)?;
                match m < n_pre {
                    true => self.prefill(&f, &row, &ids[..n_pre], m)?,
                    false => first,
                }
            }
            None => self.prefill(&f, &row, &ids[..n_pre], start)?,
        };
        self.stats.prefill_ns += t0.elapsed().as_nanos();
        self.stats.prefill_tokens += (n_pre - start) as u64;
        self.stats.prefix_hit_tokens += start as u64;
        if woken {
            self.stats.host_hits += 1;
            self.stats.host_hit_tokens += start as u64;
        } else if start > 0 {
            self.stats.resident_hits += 1;
            self.stats.resident_hit_tokens += start as u64;
        }
        debug!(
            request = %id,
            prompt,
            max_tokens,
            row = ?row,
            prefix_hit = start,
            woken,
            prefill_ms = logline::ms(t0.elapsed()),
            "admitted"
        );
        let page = self.ranks.page();
        let mut seq = Seq {
            id,
            pos: n_pre,
            next: q.request.prompt_tokens[n_pre.min(prompt - 1)],
            generated: 0,
            emitted: 0,
            max_tokens,
            stop: q.request.stop_policy.clone(),
            row,
            prompt_len: prompt,
            history: ids[..n_pre].to_vec(),
            checkpointed: start / page * page,
            parts,
            admitted: t0,
        };
        self.checkpoint(&mut seq)?;
        if let Some(tok) = first {
            let (emitted, done) = seq.emit(&[tok], ledger);
            self.stats.tokens += emitted;
            if done {
                self.finish(seq);
                return Ok(());
            }
        }
        self.running.push(seq);
        Ok(())
    }

    /// The `key.len()` tokens `row` holds, which `key` names, as a snapshot
    /// of its recurrent state. `Busy` waits for a park's pages or makes
    /// room and retries, `Remapping` waits for the slot on its way (the
    /// prefill just read its output back, so the stream the remap waits on
    /// is drained; what is left is the remap's own milliseconds); with
    /// nothing left to make room with or any other denial, the request runs
    /// on without it.
    fn keep_prompt(&mut self, row: &mut Row, key: &[i64], id: RequestId) -> Result<()> {
        loop {
            match self.ranks.checkpoint(row, key.len()) {
                Ok(Some(snap)) => {
                    self.prefix.insert(key, snap);
                    self.stats.prompt_checkpoints += 1;
                    return Ok(());
                }
                Ok(None) => return Ok(()),
                Err(Denied::Busy) if self.ranks.parking() => self.settle()?,
                Err(Denied::Busy) if self.make_room()? => {}
                Err(Denied::Remapping) => self.settle()?,
                Err(d) => {
                    debug!(request = %id, tokens = key.len(), denied = %d, "history not kept");
                    self.stats.prompt_skips += 1;
                    return Ok(());
                }
            }
        }
    }

    /// Hand every rank what is queued for it, hear what landed; `wait`
    /// blocks each rank with something in flight until it has.
    fn poll(&mut self, wait: bool) -> Result<()> {
        let orders = self.ranks.orders(None, wait);
        let answers = self.exec.run(orders)?;
        self.ranks.heard(&answers)
    }

    /// Every remap and host transfer landed, on every rank.
    fn settle(&mut self) -> Result<()> {
        while self.ranks.busy() {
            self.poll(true)?;
        }
        Ok(())
    }

    /// Run `f` over a staged step on every rank: what it handed each cell.
    fn run(&mut self, st: &mut Staged, f: &Forward) -> Result<Vec<Vec<i64>>> {
        let orders = self.ranks.orders(Some(st.items(f)), false);
        let answers = self.exec.run(orders)?;
        self.ranks.heard(&answers)?;
        st.emitted(self.ranks.protocol(), f, &answers)
    }

    /// Room for a `Busy` lease: the coldest resident snapshot goes to
    /// the host tier when there is one (the coldest parked ones dropped
    /// until it fits), else is dropped. `false` when nothing is resident.
    fn make_room(&mut self) -> Result<bool> {
        let ranks = &mut self.ranks;
        let park = (self.policy.host_bytes > 0).then_some(|snap| Ok::<_, std::convert::Infallible>(ranks.park(snap)));
        let dropped = match self.prefix.evict(park).unwrap_or_else(|e| match e {}) {
            Some(Evicted::Parked { key, dropped }) => {
                debug!(tokens = key.len(), dropped, "parked");
                self.stats.parks += 1;
                dropped
            }
            Some(Evicted::Dropped { key, dropped }) => {
                debug!(tokens = key.len(), dropped, "dropped");
                self.stats.evictions += 1;
                dropped
            }
            None => return Ok(false),
        };
        self.stats.host_evictions += dropped as u64;
        Ok(true)
    }

    /// Checkpoint every whole page of `s` not yet checkpointed, when the
    /// manifest checkpoints every page.
    fn checkpoint(&mut self, s: &mut Seq) -> Result<()> {
        if !self.every_page {
            return Ok(());
        }
        let unit = self.ranks.page();
        while s.checkpointed + unit <= s.pos {
            let len = s.checkpointed + unit;
            let Some(snap) = self.ranks.checkpoint(&mut s.row, len)? else { return Ok(()) };
            self.prefix.insert(&s.history[..len], snap);
            s.checkpointed = len;
        }
        Ok(())
    }

    /// A sequence is done: with a recurrent state and no history point
    /// kept before its end, its whole context becomes a snapshot (the
    /// slots move, nothing is copied); without one, every whole page
    /// already is. A state holding tokens the client
    /// never got (a speculative round accepted past `max_tokens` or the
    /// stop) is a key no next turn can send: it returns to the pool.
    fn finish(&mut self, s: Seq) {
        if self.every_page || s.pos == 0 || s.parts {
            return;
        }
        // The last token is normally the next input, not in the state
        // yet; the stop token itself may be in it, and stays: a chat's
        // next turn ends the answer with it.
        let visible = s.prompt_len + s.emitted;
        let past = s.history.get(visible..).unwrap_or_default();
        if s.pos > visible + 1 || past.iter().any(|&t| s.stop.classify(t as u32, |_| false).is_none()) {
            debug!(request = %s.id, tokens = s.pos, visible, "not kept");
            return;
        }
        if let Some(snap) = self.ranks.retire(s.row, s.pos) {
            self.prefix.insert(&s.history, snap);
        }
    }

    /// Drop aborted sequences before a step so they neither pad nor compute.
    fn drop_aborted(&mut self, ledger: &mut RequestLedger) {
        let running = std::mem::take(&mut self.running);
        for s in running {
            if ledger.is_aborted(s.id) {
                ledger.retire(s.id);
                self.finish(s);
            } else {
                self.running.push(s);
            }
        }
    }

    /// Chunked single-sequence prefill of `ids[start..]` at positions
    /// `start..` of `row` (`start` is the row's prefix) through `f`; the
    /// first generated token when `f` hands it back.
    fn prefill(&mut self, f: &Forward, row: &Row, ids: &[i64], start: usize) -> Result<Option<u32>> {
        let chunk = self.chunk;
        let mut first = None;
        let mut pos = start;
        while pos < ids.len() {
            let c = (ids.len() - pos).min(chunk);
            let cells = [Cell { row, ids: ids[pos..pos + c].to_vec(), pos }];
            let mut st = self.ranks.stage(&cells, c, |_| 1)?;
            let out = self.run(&mut st, f)?;
            pos += c;
            if pos == ids.len() {
                first = out[0].first().map(|&t| t as u32);
            }
        }
        Ok(first)
    }

    /// One step over every running sequence: `rows` rows per sequence at
    /// its position, the forward the protocol picks for the batch, and
    /// what it hands each sequence back.
    fn step(&mut self, ledger: &mut RequestLedger) -> Result<()> {
        self.drop_aborted(ledger);
        if self.running.is_empty() {
            return Ok(());
        }
        let rows = self.plan.rows;
        let t0 = Instant::now();
        let max_seqs = self.policy.max_seqs;
        // Each sequence feeds its next token `rows` times; a speculative
        // round drafts the rows past the first on the device.
        let cells: Vec<Cell> = self
            .running
            .iter()
            .map(|s| Cell { row: &s.row, ids: vec![s.next as i64; rows as usize], pos: s.pos })
            .collect();
        let mut st = self.ranks.stage(&cells, rows as usize, |k| bucket(k).min(max_seqs))?;
        drop(cells);
        let f = st.forward(self.ranks.protocol(), rows).expect("planned: every bucket up to max_seqs has a forward");
        let out = self.run(&mut st, &f)?;
        self.stats.step_ns += t0.elapsed().as_nanos();
        self.stats.steps += 1;

        let running = std::mem::take(&mut self.running);
        for (i, mut s) in running.into_iter().enumerate() {
            let toks: Vec<u32> = out[i].iter().map(|&t| t as u32).collect();
            if let Some(c) = &mut self.counters {
                // The first token is the next input's answer; the rest are
                // accepted drafts.
                let accepted = toks.len().saturating_sub(1);
                for p in &mut c.num_accepted_tokens_per_pos[..accepted] {
                    *p += 1;
                }
                c.num_drafts += 1;
                c.num_draft_tokens += rows - 1;
                c.num_accepted_tokens += accepted as u64;
            }
            // The tokens taken are in the state now: what was fed and
            // every accepted one after it.
            let taken = toks.len().max(1);
            s.advance(std::iter::once(s.next).chain(toks[..taken - 1].iter().copied()));
            let (n, done) = s.emit(&toks, ledger);
            self.stats.tokens += n;
            if done {
                self.finish(s);
            } else {
                self.checkpoint(&mut s)?;
                self.running.push(s);
            }
        }
        Ok(())
    }

    /// One line per 5 s window in which anything happened; a window that
    /// only idled is dropped (and restarted, so the next line's rates are
    /// over its own window). `accepted` / `accept_pct` are the window's.
    fn log_stats(&mut self) {
        let st = &self.stats;
        let dt = st.since.elapsed();
        if dt.as_secs() < 5 {
            return;
        }
        if st.tokens > 0 || st.prefill_tokens > 0 {
            let round = |x: f64, d: f64| (x * d).round() / d;
            let host = self.ranks.host_tier();
            let (slots, slots_used) = self.ranks.seq_slots();
            let (drafts, draft_tokens, accepted) = self.counters.as_ref().map_or((0, 0, 0), |c| {
                (c.num_drafts - st.spec_at.0, c.num_draft_tokens - st.spec_at.1, c.num_accepted_tokens - st.spec_at.2)
            });
            info!(
                running = self.running.len(),
                rows_per_rank = ?self.rows_per_rank(),
                waiting = self.waiting.len() + self.waking.len(),
                kv_pct = round(self.ranks.pages_used() as f64 * 100.0 / self.ranks.pages_total().max(1) as f64, 10.0),
                steps = st.steps,
                step_ms = round(st.step_ns as f64 / 1e6 / st.steps.max(1) as f64, 100.0),
                tok_s = round(st.tokens as f64 / dt.as_secs_f64(), 1.0),
                prefill_tokens = st.prefill_tokens,
                prefill_tok_s =
                    (st.prefill_ns > 0).then(|| round(st.prefill_tokens as f64 / (st.prefill_ns as f64 / 1e9), 1.0)),
                prefix_hit_tokens = st.prefix_hit_tokens,
                resident_hits = st.resident_hits,
                resident_hit_tokens = st.resident_hit_tokens,
                host_hits = host.map(|_| st.host_hits),
                host_hit_tokens = host.map(|_| st.host_hit_tokens),
                checkpoints = self.prefix.entries(),
                prompt_checkpoints = (!self.every_page).then_some(st.prompt_checkpoints),
                prompt_skips = (!self.every_page).then_some(st.prompt_skips),
                evictions = st.evictions,
                parked = host.map(|_| self.prefix.count(Tier::Parked)),
                host_pct = host.map(|(u, n)| round(u as f64 * 100.0 / n.max(1) as f64, 10.0)),
                parks = host.map(|_| st.parks),
                host_evictions = host.map(|_| st.host_evictions),
                wakes = host.map(|_| st.wakes),
                wake_tokens = host.map(|_| st.wake_tokens),
                slots_used = self.ranks.has_seq_state().then_some(slots_used),
                slots = self.ranks.has_seq_state().then_some(slots),
                remaps = self.ranks.remaps(),
                accepted = self.counters.as_ref().map(|_| round((accepted + drafts) as f64 / drafts.max(1) as f64, 100.0)),
                accept_pct =
                    self.counters.as_ref().map(|_| round(accepted as f64 * 100.0 / draft_tokens.max(1) as f64, 1.0)),
                "stats"
            );
        }
        self.stats = Stats::new(&self.counters);
    }
}

impl Scheduler for KernScheduler {
    fn submit(&mut self, request: QueuedRequest) {
        self.waiting.push_back(request);
    }

    fn step(&mut self, ledger: &mut RequestLedger) -> Result<()> {
        self.admit(ledger)?;
        if self.running.is_empty() && self.ranks.outstanding() {
            self.poll(false)?;
        }
        self.step(ledger)?;
        self.log_stats();
        Ok(())
    }

    fn metrics(&self) -> SchedulerMetrics {
        SchedulerMetrics {
            kv_used_blocks: self.ranks.pages_used() as u64,
            kv_total_blocks: self.ranks.pages_total() as u64,
            num_running_reqs: self.running.len() as u64,
            num_waiting_reqs: self.waiting.len() as u64,
            spec_decode: self.counters,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kern_manifest::types::Manifest;
    use kern_manifest::verify;

    #[test]
    fn a_sequence_takes_tokens_through_its_stop_or_its_length() {
        use pegainfer_frontend::engine::EosPolicy;
        let stop = StopPolicy::new(EosPolicy::Token(9), vec![7]);
        assert_eq!(cut(&[1, 2, 3], 0, 10, &stop), (3, None));
        assert_eq!(cut(&[1, 9, 3], 0, 10, &stop), (2, Some(End::Stop(StopCause::Eos(9)))));
        assert_eq!(cut(&[1, 2, 3], 8, 10, &stop), (2, Some(End::Length)));
        assert_eq!(cut(&[1, 7], 8, 10, &stop), (2, Some(End::Stop(StopCause::Token(7)))));
        assert_eq!(cut(&[], 0, 10, &stop), (0, None));
        // Every token string over {plain, stop, eos} up to four long, at
        // every count already generated short of the limit.
        let alphabet = [1u32, 7, 9];
        let strings = (0..=4u32).flat_map(|n| {
            (0..3usize.pow(n)).map(move |k| (0..n).map(|i| alphabet[k / 3usize.pow(i) % 3]).collect::<Vec<_>>())
        });
        for toks in strings {
            for (generated, max) in (1..=5).flat_map(|m| (0..m).map(move |g| (g, m))) {
                let (taken, end) = cut(&toks, generated, max, &stop);
                let stops = |s: &[u32]| s.iter().filter(|&&t| t != 1).count();
                let fact = match end {
                    None => taken == toks.len() && stops(&toks) == 0 && generated + taken < max,
                    Some(End::Stop(_)) => {
                        stops(&toks[..taken]) == 1 && toks[taken - 1] != 1 && generated + taken <= max
                    }
                    Some(End::Length) => stops(&toks[..taken]) == 0 && generated + taken == max,
                };
                assert!(fact, "{toks:?} at {generated}/{max}: ({taken}, {end:?})");
            }
        }
    }

    /// The plain contract of kern-manifest's own tests: a prefill chunk
    /// that only writes state, a one-row step for one sequence and one
    /// for four.
    fn plain() -> Manifest {
        Manifest::from_json(include_str!("../../kern-manifest/tests/fixtures/plain.json")).unwrap()
    }

    /// The same plus a speculative round: 4 rows per sequence for up to 2
    /// sequences, handing back 4 tokens and a count.
    fn speculative() -> Manifest {
        let mut v: serde_json::Value = serde_json::from_str(&serde_json::to_string(&plain()).unwrap()).unwrap();
        v["buffers"]["verify_tokens"] =
            serde_json::json!({"kind": "output", "dtype": "i64", "shape": ["seqs", 4], "fill": "tokens"});
        v["buffers"]["nacc"] =
            serde_json::json!({"kind": "output", "dtype": "i32", "shape": ["seqs"], "fill": "count"});
        v["ops"]["accept"] = serde_json::json!({"params": ["out buffer<i64>", "out buffer<i32>"], "impl": {"launches": [{"entry": "extern:x"}]}});
        v["programs"]["round"] = serde_json::json!({"batch": {"groups": 2, "rows": 4}, "calls": [
            {"op": "accept", "args": [{"buf": "verify_tokens"}, {"buf": "nacc"}]}]});
        Manifest::from_json(&v.to_string()).unwrap()
    }

    fn check(m: &Manifest, page: usize, rows: Option<u64>, seqs: usize) -> Result<Plan> {
        Plan::check(&Protocol::check(&verify(m.clone())?)?, page, 4, rows, seqs)
    }

    fn rejects(m: &Manifest, page: usize, rows: Option<u64>, what: &str) {
        let Err(e) = check(m, page, rows, 4) else { panic!("accepted, expected `{what}`") };
        let e = format!("{e:#}");
        assert!(e.contains(what), "{e}");
    }

    #[test]
    fn plain_plan() {
        let p = check(&plain(), 16, None, 4).unwrap();
        assert_eq!((p.rows, p.step.name.as_str(), p.chunk.name.as_str()), (1, "decode", "prefill"));
        assert_eq!((p.headroom(), p.counters().is_none(), p.chunk.emits), (0, true, None));
        // The operator's ask, within the sequences bound.
        assert_eq!(
            (check(&plain(), 16, None, 0).unwrap().max_seqs, check(&plain(), 16, None, 100).unwrap().max_seqs),
            (1, 4)
        );
    }

    #[test]
    fn the_widest_rows_by_default() {
        let p = check(&speculative(), 16, None, 4).unwrap();
        assert_eq!((p.rows, p.step.name.as_str(), p.headroom()), (4, "round", 3));
        assert_eq!(p.counters().map(|c| c.num_spec_tokens), Some(3));
        // Four rows per sequence fit twice in 8, and the round takes two sequences.
        assert_eq!(p.max_seqs, 2);
        // The plain step on the same manifest, on request.
        let p = check(&speculative(), 16, Some(1), 4).unwrap();
        assert_eq!((p.rows, p.step.name.as_str(), p.max_seqs), (1, "decode", 4));
    }

    #[test]
    fn plan_rejections() {
        rejects(&plain(), 16, Some(4), "no program takes one sequence of 4 rows; the manifest declares rows [1]");
        rejects(&speculative(), 2, Some(4), "exceed the 2-token pad page");
        let mut m = plain();
        m.programs.remove("prefill");
        m.ops.remove("write");
        rejects(&m, 16, None, "nothing to feed a prompt through");
        // Only a chunk forward, one that hands a token back: nothing to step with.
        let mut m = plain();
        let head = m.programs["decode"].calls[0].clone();
        m.programs.get_mut("prefill").unwrap().calls.push(head);
        m.programs.remove("decode");
        m.programs.remove("decode_batch");
        rejects(&m, 16, None, "no program takes a fixed number of rows");
    }
}
