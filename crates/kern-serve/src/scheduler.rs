//! The kern scheduler: one tray, many sequences.
//!
//! Implements the pegainfer frontend's [`Scheduler`] contract — `submit`,
//! `step`, `metrics` — over a kern manifest driven by a [`Tray`] (one
//! runtime per GPU, in lockstep; `tray.rs` is the design). The caller
//! contract the manifest must speak: input buffers `token_ids` /
//! `slot_mapping` / `seq_lens` (`positions` and `cu_seqlens_q` when it has
//! them), a `block_table` page table, a `next_token` output, and programs
//! `decode` (one row per sequence; `decode_batch` for batches of more than
//! one when the manifest has it, the bs=1 microprogram otherwise) and,
//! optionally, `prefill` (single sequence, chunked). Two prefill contracts
//! when there is one: state only (the last prompt token is the first
//! decode step's input) or, when `prefill` writes `next_token` (hybrid GDN
//! models, whose chunked kernels must see every prompt token), every
//! prompt token through prefill and the first generated token from it.
//! Without a `prefill` program the prompt goes through decode steps one
//! token at a time — a row feeding its prompt is a row like any other,
//! whose outputs are dropped until the last prompt token is in — or, when
//! the manifest has `decode_span`, one sequence per step feeds a run of up
//! to `chunk` prompt tokens as consecutive rows (the span contract in
//! `tray.rs`), the first generated token being the last row's. That is
//! what a decode-only tray manifest (K3 today) gets: correct, and slow
//! without the span. A
//! manifest with per-sequence states (`bytes_per_seq`) also has line
//! tables indexing them; the tray stages them from the rows.
//!
//! Policy, deliberately simple:
//! - prefill first: each step admits waiting requests (up to a token
//!   budget when there is a `prefill` program) and prefills them one at a
//!   time, then runs one decode step over every running sequence;
//! - a request leases every KV page its worst case (`prompt + max_tokens`)
//!   needs at admission (`Tray::lease`, on the rank with the fewest pages
//!   in use), so decode never runs out of pages and nothing is ever
//!   preempted; the row drops with the sequence;
//! - decode batches are padded up to a bucket size — the same bucket on
//!   every rank of the tray, from the most loaded one — and each bucket's
//!   program is CUDA-graph-captured once; padding rows write into a page
//!   the tray leases for them and nobody reads;
//! - greedy only: the manifest's `argmax` is the sampler. Non-greedy
//!   sampling params are logged once and served greedily;
//! - `--spec` (a manifest with a `spec` block and `draft` / `verify` /
//!   `draft_precompute` / `decode_spec`; one rank only): every step is one
//!   speculative round over the batch — `draft` proposes `n` tokens per
//!   sequence, `verify` runs the target over `[anchor, drafts]` per
//!   sequence, `draft_precompute` projects the target's taps into the
//!   draft's context KV for every row (rejected rows land past the
//!   sequence's position and are overwritten next round, exactly like the
//!   target KV's free rollback), and the host accepts each sequence's
//!   longest matching prefix. The lease grows by `n` tokens so the last
//!   round's rejected rows have slots. A manifest with a `round` program
//!   runs the whole round as one graph: draft, verify's ids spliced on
//!   device, verify, precompute, accept on device, `advance` from the
//!   device's `num_accepted` — one launch and one sync per round instead
//!   of four; the host reads `draft_tokens` / `verify_tokens` and accepts
//!   the same prefix. Whether a round beats a plain step at a given batch
//!   size is the operator's call, not the scheduler's: the flag picks the
//!   mode for the process.
//! - prefix reuse: a finished sequence's KV lives on as snapshots
//!   (`Tray::checkpoint` / `retire`, indexed by token hash in a `Prefix`
//!   table keyed over the whole tray), and a new prompt starts from the
//!   longest snapshot holding a proper prefix of it (`Tray::lease_from`;
//!   prefill covers the rest). A paged-only manifest checkpoints every
//!   whole page as a sequence fills it — free, a shared page — so any
//!   earlier prompt or output is reusable at page granularity; a manifest
//!   with a recurrent state checkpoints only where a request ends (the
//!   finished sequence's state slots become the snapshot's, nothing is
//!   copied), so only a prompt that continues an earlier request's whole
//!   context hits. A `Busy` lease makes room and retries until it fits or
//!   nothing is left: the least recently hit snapshot is parked into the
//!   host tier (`--host-gib`: pinned DRAM per rank, `Tray::park`, all of
//!   its members' pieces or none; the coldest parked ones are dropped when
//!   it is full) or, without one, dropped. A prompt hitting a parked
//!   snapshot wakes it (`Tray::wake`): the copies ride the transfer
//!   streams and the request waits in `waking` until `Tray::awake` hands
//!   out the row, so no step queues behind them.

use std::collections::VecDeque;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use kern_manifest::types::{Arg, Dim, Manifest};
use kern_runtime::{Chain, Denied, Error, Prefix, Tier};
use pegainfer_frontend::engine::{
    FinishReason, QueuedRequest, RejectReason, RequestId, RequestLedger, Scheduler, SchedulerMetrics,
    SpecDecodeCounters, MAX_SPEC_TOKENS,
};
use tracing::{debug, info, warn};

use crate::logline;
use crate::tray::{Cell, Rising, Row, Shape, Sleeping, Snapshot, Tray};

/// Decode batch buckets; a batch is padded up to the smallest one that
/// fits and each bucket owns one captured graph.
const BUCKETS: [usize; 13] = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256];

fn bucket(n: usize) -> usize {
    BUCKETS.iter().copied().find(|&b| b >= n).unwrap_or(n)
}

pub struct Policy {
    /// Prefill chunk (tokens per `prefill` call), clamped to the manifest's
    /// `tokens` bound.
    pub chunk: usize,
    /// Prompt tokens one step may prefill before it runs decode (at least
    /// one request is always admitted when one fits).
    pub prefill_budget: usize,
    /// Launch every call eagerly instead of capturing graphs.
    pub eager: bool,
    /// Cap on concurrently running sequences per rank (≤ the manifest's
    /// `seqs` bound).
    pub max_seqs: usize,
    /// Token ids that end a request unless it asked `ignore_eos`.
    pub stop_tokens: Vec<u32>,
    /// Speculative rounds instead of decode steps (needs the spec programs).
    pub spec: bool,
    /// Pinned host memory per rank for parked snapshots (0: none).
    pub host_bytes: u64,
}

/// The manifest's speculative contract, one row-group per sequence.
struct SpecPlan {
    /// Tokens `draft` proposes per sequence (`draft_tokens` is `[seqs, n]`).
    n_drafts: usize,
    /// Rows per sequence in `draft`: `[anchor, mask, ...]`.
    draft_rows: usize,
    /// Rows per sequence in `verify`: `[anchor, drafts...]` = n + 1.
    verify_rows: usize,
    /// Fills the undrafted rows of `draft`.
    mask_token: i64,
    /// The target resumes a recurrent state from `num_accepted_tokens`
    /// (one per sequence) and commits the accepted rows with `advance`.
    advance: bool,
    /// The manifest has `round`: the whole round is one program (draft
    /// and verify rows per sequence coincide, so one staging serves both).
    fused: bool,
    counters: SpecDecodeCounters,
}

impl SpecPlan {
    /// The manifest's speculative contract; a round's rows per sequence
    /// must fit the `page`-token pad page.
    fn check(m: &Manifest, page: usize) -> Result<SpecPlan> {
        let fused = m.programs.contains_key("round");
        need_programs(
            m,
            if fused {
                &["decode_spec", "draft_precompute"]
            } else {
                &["decode_spec", "draft_precompute", "draft", "verify"]
            },
        )?;
        let n_drafts = seqs_rows(m, "draft_tokens")?;
        let verify_rows = seqs_rows(m, "verify_tokens")?;
        if verify_rows != n_drafts + 1 {
            bail!("verify_tokens has {verify_rows} rows per sequence, expected {} (anchor + drafts)", n_drafts + 1);
        }
        if n_drafts > MAX_SPEC_TOKENS {
            bail!("{n_drafts} drafts per round, the frontend's metrics hold {MAX_SPEC_TOKENS}");
        }
        per_seq(m, "anchor_token")?;
        // The target resumes a recurrent state from `num_accepted_tokens`
        // and commits the accepted rows with `advance`: both or neither.
        let advance = m.programs.contains_key("advance");
        match (m.buffers.contains_key("num_accepted_tokens"), advance) {
            (true, true) => per_seq(m, "num_accepted_tokens")?,
            (true, false) => bail!(
                "`num_accepted_tokens` resumes a recurrent state but no `advance` program commits the accepted rows"
            ),
            (false, true) => bail!("program `advance` without a `num_accepted_tokens` input"),
            (false, false) => {}
        }
        let Some(spec) = &m.spec else {
            bail!("the manifest has no `spec` block (draft rows per sequence and the mask token)");
        };
        let (draft_rows, mask_token) = (spec.block as usize, spec.mask_token);
        if fused && draft_rows != verify_rows {
            bail!("`round` needs draft and verify rows per sequence to coincide, got {draft_rows} and {verify_rows}");
        }
        let plan = SpecPlan {
            n_drafts,
            draft_rows,
            verify_rows,
            mask_token,
            advance,
            fused,
            counters: SpecDecodeCounters { num_spec_tokens: n_drafts as u64, ..Default::default() },
        };
        if plan.rows() > page {
            bail!("{} rows per sequence per round exceed the {page}-token pad page", plan.rows());
        }
        Ok(plan)
    }

    /// Rows per sequence a round stages: the wider of draft's and verify's.
    fn rows(&self) -> usize {
        self.draft_rows.max(self.verify_rows)
    }
}

struct Seq {
    id: RequestId,
    /// Tokens already in the KV state.
    pos: usize,
    /// The token the next decode step feeds at `pos`.
    next: u32,
    /// Prompt tokens still to feed after `next`, one per step, when the
    /// prompt goes through decode; the outputs of those steps are dropped.
    pending: VecDeque<u32>,
    generated: usize,
    max_tokens: usize,
    ignore_eos: bool,
    /// Its KV pages and state slots across the tray; returned when the
    /// sequence drops.
    row: Row,
    prompt_len: usize,
    /// The hash chain over the tokens in the state, `pos` of them; the
    /// prefix table keys this sequence's snapshots by it.
    chain: Chain,
    /// Tokens already checkpointed, a whole number of pages.
    checkpointed: usize,
    admitted: Instant,
}

impl Seq {
    /// `fed` went through a program and is in the state now.
    fn advance(&mut self, fed: impl IntoIterator<Item = u32>) {
        for t in fed {
            self.pos += 1;
            self.chain.push(t as i64);
        }
    }

    /// Account `toks` as generated, in order: each is emitted until a stop
    /// token (itself not emitted, pegainfer convention; it still counts
    /// against `max_tokens` like vLLM's) or `max_tokens`. Finishes the
    /// request in the ledger when it is done, otherwise the last token is
    /// the next step's input. Returns how many tokens were emitted and
    /// whether the sequence finished.
    fn emit(&mut self, toks: &[u32], stop: &[u32], ledger: &mut RequestLedger) -> (u64, bool) {
        let mut out = Vec::with_capacity(toks.len());
        let mut reason = None;
        for &tok in toks {
            self.generated += 1;
            if !self.ignore_eos && stop.contains(&tok) {
                reason = Some(FinishReason::Stop);
                break;
            }
            out.push(tok);
            if self.generated >= self.max_tokens {
                reason = Some(FinishReason::Length);
                break;
            }
        }
        if !out.is_empty() {
            ledger.push_tokens(self.id, &out, &[]);
        }
        let done = match reason {
            Some(r) => {
                debug!(
                    request = %self.id,
                    reason = ?r,
                    prompt = self.prompt_len,
                    generated = self.generated,
                    elapsed_s = logline::secs(self.admitted.elapsed()),
                    "finished"
                );
                ledger.finish(self.id, r);
                true
            }
            None => {
                if let Some(&t) = toks.last() {
                    self.next = t;
                }
                false
            }
        };
        (out.len() as u64, done)
    }
}

pub struct KernScheduler {
    tray: Tray,
    policy: Policy,
    spec: Option<SpecPlan>,
    /// `Some(emits)`: the manifest has `prefill`, and whether it writes
    /// `next_token` itself (see the module doc). `None`: prompts go
    /// through decode steps.
    prefill: Option<bool>,
    /// The manifest has `decode_batch` for batches of more than one row.
    decode_batch: bool,
    /// The manifest has `decode_span`: one sequence per step may feed a run
    /// of up to this many prompt tokens, a row each.
    span: Option<usize>,
    waiting: VecDeque<QueuedRequest>,
    /// Admitted requests whose woken rows are still on the way in.
    waking: VecDeque<(QueuedRequest, Rising)>,
    running: Vec<Seq>,
    /// Snapshots of finished prefixes, for the next prompt that shares
    /// one (see the module doc).
    prefix: Prefix<Snapshot, Sleeping>,
    /// Checkpoint at every page (a paged-only manifest) rather than only
    /// where a request ends (one with a recurrent state).
    every_page: bool,
    warned_sampling: bool,
    stats: Stats,
}

/// Rolling counters for the periodic log line, reset when it prints.
struct Stats {
    since: Instant,
    /// Decode steps (speculative rounds under `--spec`) and their time.
    steps: u64,
    step_ns: u128,
    /// Tokens emitted to the ledger.
    tokens: u64,
    /// Prompt tokens fed: through `prefill` (timed) or through decode steps.
    prefill_tokens: u64,
    prefill_ns: u128,
    /// Prompt tokens found in a snapshot, and snapshots evicted for room.
    prefix_hit_tokens: u64,
    evictions: u64,
    /// Snapshots parked to the host, dropped from it for room, woken
    /// from it, and the tokens woken.
    parks: u64,
    host_evictions: u64,
    wakes: u64,
    wake_tokens: u64,
    /// The speculative counters at the window's start, so the window's
    /// acceptance is reported rather than the process's.
    spec_at: (u64, u64, u64),
}

impl Stats {
    fn new(spec: &Option<SpecPlan>) -> Stats {
        let spec_at = spec.as_ref().map_or((0, 0, 0), |p| {
            let c = &p.counters;
            (c.num_drafts, c.num_draft_tokens, c.num_accepted_tokens)
        });
        Stats {
            since: Instant::now(),
            steps: 0,
            step_ns: 0,
            tokens: 0,
            prefill_tokens: 0,
            prefill_ns: 0,
            prefix_hit_tokens: 0,
            evictions: 0,
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

/// The manifest's fit to this caller contract, settled before the GPU is
/// touched: pure over the manifest, the tray's shape and size, so a
/// synthetic manifest exercises every rejection.
struct Contract {
    prefill: Option<bool>,
    decode_batch: bool,
    spec: Option<SpecPlan>,
}

impl Contract {
    /// `page` is the tray's page in tokens, `ranks` how many it drives;
    /// `spec` asks for the speculative contract too.
    fn check(m: &Manifest, page: usize, ranks: usize, spec: bool) -> Result<Contract> {
        need_programs(m, &["decode"])?;
        let prefill = m.programs.get("prefill").map(|calls| {
            calls.iter().flat_map(|c| &c.args).any(|a| matches!(a, Arg::Buf { buf, .. } if buf == "next_token"))
        });
        if spec && ranks > 1 {
            bail!("--spec drives one rank; this tray has {ranks}");
        }
        if spec && prefill.is_none() {
            bail!("--spec needs a `prefill` program");
        }
        let spec = spec
            .then(|| SpecPlan::check(m, page).context("--spec: the manifest's speculative contract"))
            .transpose()?;
        Ok(Contract { prefill, decode_batch: m.programs.contains_key("decode_batch"), spec })
    }

    /// Concurrent sequences per rank: `want` within the `seqs` bound and
    /// what one call's rows — one per sequence, a round's row-group under
    /// speculation — fit in `tokens`.
    fn max_seqs(&self, shape: &Shape, want: usize) -> usize {
        let rows = self.spec.as_ref().map_or(1, SpecPlan::rows);
        want.clamp(1, shape.seqs_max).min(shape.tokens_max / rows).max(1)
    }
}

/// What a lease attempt handed out.
enum Got {
    Row(Row),
    Rising(Rising),
}

impl KernScheduler {
    /// Wrap a loaded tray (weights bound, peers connected, pads leased):
    /// check the manifest against this caller contract and settle the
    /// policy within its bounds.
    pub fn new(tray: Tray, policy: Policy) -> Result<KernScheduler> {
        let c = Contract::check(tray.manifest(), tray.page(), tray.len(), policy.spec)?;
        let policy = Policy {
            max_seqs: c.max_seqs(tray.shape(), policy.max_seqs),
            chunk: policy.chunk.clamp(1, tray.shape().tokens_max),
            ..policy
        };
        let stats = Stats::new(&c.spec);
        let every_page = !tray.has_seq_state();
        let span = tray.shape().span;
        let prefix = Prefix::new(tray.page());
        let s = KernScheduler {
            tray,
            policy,
            spec: c.spec,
            prefill: c.prefill,
            decode_batch: c.decode_batch,
            span,
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
            total_blocks: self.tray.pages_total(),
            block_size: self.tray.page(),
            max_request_tokens: self.tray.max_seq_tokens() - self.headroom(),
        }
    }

    /// Slots a lease holds past `prompt + max_tokens`: a speculative
    /// round writes `n_drafts` rows past the token it may still have to
    /// emit, and the last round's rejects need slots.
    fn headroom(&self) -> usize {
        self.spec.as_ref().map_or(0, |s| s.n_drafts)
    }

    fn log_ready(&self) {
        let (tray, policy, spec, facts) = (&self.tray, &self.policy, self.spec.as_ref(), self.facts());
        let (slots, _) = tray.seq_slots();
        // Pages: what requests can lease (one more per rank holds the
        // padding rows). Graphs are captured per bucket on first use
        // unless `eager`.
        info!(
            ranks = tray.len(),
            tray = tray.group_size(),
            pages = facts.total_blocks,
            page = facts.block_size,
            seq_slots = (slots > 0).then_some(slots),
            max_seqs_per_rank = policy.max_seqs,
            max_request_tokens = facts.max_request_tokens,
            chunk = policy.chunk,
            buckets = ?BUCKETS.iter().filter(|&&b| b <= policy.max_seqs).collect::<Vec<_>>(),
            eager = policy.eager,
            prefill = match self.prefill {
                Some(true) => "emits next_token",
                Some(false) => "state only",
                None => "through decode steps",
            },
            decode_batch = self.decode_batch,
            checkpoints = if self.every_page { "every page" } else { "at request end" },
            host_gib_per_rank = (policy.host_bytes > 0).then_some(policy.host_bytes >> 30),
            drafts = spec.map(|s| s.n_drafts),
            draft_rows = spec.map(|s| s.draft_rows),
            verify_rows = spec.map(|s| s.verify_rows),
            fused_round = spec.map(|s| s.fused),
            "scheduler ready"
        );
    }

    /// Rows each rank has (running, and woken ones on the way in).
    fn rows_per_rank(&self) -> Vec<usize> {
        let mut rows = vec![0usize; self.tray.len()];
        for s in &self.running {
            rows[s.row.owner().index()] += 1;
        }
        for (_, r) in &self.waking {
            rows[r.owner().index()] += 1;
        }
        rows
    }

    /// Admit waiting requests in order and prefill each (single sequence,
    /// chunked) up to the step's token budget.
    fn admit(&mut self, ledger: &mut RequestLedger) -> Result<()> {
        let mut budget_used = 0usize;
        // Wakes land in order: the first still in flight stops the scan.
        while let Some((q, r)) = self.waking.pop_front() {
            match self.tray.awake(r)? {
                Ok(row) => {
                    if ledger.is_aborted(q.id) {
                        ledger.retire(q.id);
                        continue;
                    }
                    self.stats.wakes += 1;
                    self.stats.wake_tokens += row.prefix() as u64;
                    budget_used += self.admit_one(q, row, true, ledger)?;
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
            if prompt == 0 {
                let limit = self.tray.max_seq_tokens();
                ledger.reject(id, RejectReason::ContextLength { prompt_tokens: prompt, max_tokens, limit });
                self.waiting.pop_front();
                continue;
            }
            if self.prefill.is_some() && budget_used > 0 && budget_used + prompt - 1 > self.policy.prefill_budget {
                break; // enough prefill for this step; decode must run
            }
            let ids: Vec<i64> = q.request.prompt_tokens.iter().map(|&t| t as i64).collect();
            let hit = self.prefix.lookup(&ids);
            // A hit continues on the snapshot's owner, which must have a row free.
            let owner = hit.and_then(|h| match h.tier {
                Tier::Resident => self.prefix.resident(h.id).map(Snapshot::owner),
                Tier::Parked => self.prefix.parked(h.id).map(Sleeping::owner),
            });
            if owner.is_some_and(|o| rows[o.index()] >= cap) {
                break;
            }
            let got = loop {
                let tray = &mut self.tray;
                let attempt = match hit {
                    Some(h) => match h.tier {
                        Tier::Resident => {
                            let snap = self.prefix.resident(h.id).expect("hit");
                            tray.lease_from(snap, h.len, worst).map(Got::Row)
                        }
                        Tier::Parked => {
                            let p = self.prefix.parked(h.id).expect("hit");
                            tray.wake(p, h.len, worst).map(Got::Rising)
                        }
                    },
                    None => tray.lease(worst, |r| rows[r.index()] < cap).map(Got::Row),
                };
                match attempt {
                    Err(Error::Denied(Denied::Busy)) if self.make_room()? => {}
                    r => break r,
                }
            };
            let row = match got {
                Ok(Got::Row(row)) => row,
                Ok(Got::Rising(r)) => {
                    // Its copies are in flight; it is admitted once they land.
                    let q = self.waiting.pop_front().unwrap();
                    self.waking.push_back((q, r));
                    continue;
                }
                Err(Error::Denied(Denied::Busy)) => break, // wait for pages / a slot
                Err(Error::Denied(Denied::Remapping)) => break, // pages or a slot are on their way
                Err(Error::Denied(Denied::ExceedsRow { limit })) => {
                    ledger.reject(id, RejectReason::ContextLength { prompt_tokens: prompt, max_tokens, limit });
                    self.waiting.pop_front();
                    continue;
                }
                Err(Error::Denied(Denied::ExceedsPool)) => {
                    ledger.reject(id, RejectReason::KvBudget { prompt_tokens: prompt, worst_case_tokens: worst });
                    self.waiting.pop_front();
                    continue;
                }
                Err(e) => return Err(e.into()),
            };
            let q = self.waiting.pop_front().unwrap();
            budget_used += self.admit_one(q, row, false, ledger)?;
        }
        Ok(())
    }

    /// Start `q` running in `row` (past the row's prefix): through
    /// `prefill` when the manifest has one, else with the prompt queued
    /// for the decode steps. Returns the prompt tokens prefilled, for the
    /// step's budget.
    fn admit_one(&mut self, q: QueuedRequest, row: Row, woken: bool, ledger: &mut RequestLedger) -> Result<usize> {
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
        // With a prefill program every prompt token goes through it when
        // it emits the first generated token itself; otherwise everything
        // but the last, which is the first decode step's input. A snapshot
        // hit skips its tokens (never the last one). Without one the
        // prompt past the hit is fed a token per decode step.
        let (n_pre, first, pending) = match self.prefill {
            Some(emits) => {
                let n_pre = if emits { prompt } else { prompt - 1 };
                let first = self.prefill(&row, &ids[..n_pre], start)?;
                self.stats.prefill_ns += t0.elapsed().as_nanos();
                self.stats.prefill_tokens += (n_pre - start) as u64;
                (n_pre, first, VecDeque::new())
            }
            None => (start, None, q.request.prompt_tokens[start + 1..].iter().copied().collect()),
        };
        self.stats.prefix_hit_tokens += start as u64;
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
        let page = self.tray.page();
        let mut seq = Seq {
            id,
            pos: n_pre,
            next: q.request.prompt_tokens[n_pre.min(prompt - 1)],
            pending,
            generated: 0,
            max_tokens,
            ignore_eos: q.request.params.ignore_eos,
            row,
            prompt_len: prompt,
            chain: Chain::over(page, &ids[..n_pre]),
            checkpointed: start / page * page,
            admitted: t0,
        };
        self.checkpoint(&mut seq)?;
        let first = match first {
            Some(tok) => Some(tok),
            None if self.spec.is_some() => {
                // The last prompt token goes through `decode_spec` now
                // (a round needs an anchor and its tap in the draft
                // KV); its token is the first one emitted.
                let tok = self.first_token(&seq)?;
                seq.advance([seq.next]);
                Some(tok)
            }
            None => None,
        };
        if let Some(tok) = first {
            let (emitted, done) = seq.emit(&[tok], &self.policy.stop_tokens, ledger);
            self.stats.tokens += emitted;
            if done {
                self.finish(seq);
                return Ok(n_pre - start);
            }
        }
        self.running.push(seq);
        Ok(n_pre - start)
    }

    /// Room for a `Busy` lease: the coldest resident snapshot goes to
    /// the host tier when there is one (the coldest parked ones dropped
    /// until it fits), else is dropped. `false` when nothing is resident.
    fn make_room(&mut self) -> Result<bool> {
        let Some(id) = self.prefix.coldest(Tier::Resident) else { return Ok(false) };
        if self.policy.host_bytes > 0 {
            loop {
                let tray = &mut self.tray;
                if self.prefix.park(id, |snap| tray.park(snap))? {
                    debug!(tokens = self.prefix.parked(id).map_or(0, kern_runtime::Kept::tokens), "parked");
                    self.stats.parks += 1;
                    return Ok(true);
                }
                match self.prefix.coldest(Tier::Parked) {
                    Some(c) => {
                        self.prefix.remove(c);
                        self.stats.host_evictions += 1;
                    }
                    None => break,
                }
            }
        }
        self.prefix.remove(id);
        self.stats.evictions += 1;
        Ok(true)
    }

    /// Checkpoint every whole page of `s` not yet checkpointed, when the
    /// manifest checkpoints every page.
    fn checkpoint(&mut self, s: &mut Seq) -> Result<()> {
        if !self.every_page {
            return Ok(());
        }
        let unit = self.tray.page();
        while s.checkpointed + unit <= s.pos {
            let len = s.checkpointed + unit;
            let snap = self.tray.checkpoint(&mut s.row, len)?;
            self.prefix.insert(&s.chain, snap);
            s.checkpointed = len;
        }
        Ok(())
    }

    /// A sequence is done: with a recurrent state, its whole context
    /// becomes a snapshot (the slots move, nothing is copied); without
    /// one, every whole page already is.
    fn finish(&mut self, s: Seq) {
        if self.every_page || s.pos == 0 {
            return;
        }
        let snap = self.tray.retire(s.row, s.pos);
        self.prefix.insert(&s.chain, snap);
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
    /// `start..` of `row` (`start` is the row's prefix); the first
    /// generated token when the manifest's prefill emits it.
    fn prefill(&mut self, row: &Row, ids: &[i64], start: usize) -> Result<Option<u32>> {
        let chunk = self.policy.chunk;
        let emits = self.prefill == Some(true);
        let spec = self.spec.is_some();
        let mut first = None;
        let mut pos = start;
        while pos < ids.len() {
            let c = (ids.len() - pos).min(chunk);
            let cells = [Cell { row, ids: ids[pos..pos + c].to_vec(), pos, col: 0 }];
            let eager = self.policy.eager || c != chunk;
            let mut st = self.tray.stage(&cells, bucket)?;
            st.run("prefill", eager)?;
            if spec {
                // The chunk's taps (`fc_out`) into the draft's context KV;
                // positions/slot_mapping are still the chunk's.
                st.run("draft_precompute", eager)?;
            }
            pos += c;
            if emits && pos == ids.len() {
                first = Some(st.read_i64("next_token")?[0][0] as u32);
            }
        }
        Ok(first)
    }

    /// Speculative admission: the last prompt token through `decode_spec`
    /// (bs=1, taps) and its row into the draft KV; returns the first
    /// generated token.
    fn first_token(&mut self, s: &Seq) -> Result<u32> {
        let cells = [Cell { row: &s.row, ids: vec![s.next as i64], pos: s.pos, col: 0 }];
        let mut st = self.tray.stage(&cells, bucket)?;
        st.run("decode_spec", self.policy.eager)?;
        st.run("draft_precompute", self.policy.eager)?;
        Ok(st.read_i64("next_token")?[0][0] as u32)
    }

    /// One speculative round over every running sequence: draft, verify,
    /// precompute, accept — the `round` program, or the four phased ones.
    fn spec_round(&mut self, ledger: &mut RequestLedger) -> Result<()> {
        self.drop_aborted(ledger);
        let n = self.running.len();
        if n == 0 {
            return Ok(());
        }
        let plan = self.spec.as_ref().unwrap();
        let (nd, dr, mask, advance, fused) =
            (plan.n_drafts, plan.draft_rows, plan.mask_token, plan.advance, plan.fused);
        let eager = self.policy.eager;
        let t0 = Instant::now();
        let running = &self.running;
        let tray = &mut self.tray;

        // Draft: [anchor, mask × (dr-1)] per sequence at pos.., non-causal.
        let cells: Vec<Cell> = running
            .iter()
            .map(|s| {
                let mut ids = vec![mask; dr];
                ids[0] = s.next as i64;
                Cell { row: &s.row, ids, pos: s.pos, col: 0 }
            })
            .collect();
        let mut st = tray.stage(&cells, bucket)?;
        st.write_seqs("anchor_token", |i| running[i].next as i64, 0i64)?;
        let (drafts, vt) = if fused {
            // Verify resumes from the committed state; the round's accept
            // writes advance's own `num_accepted` and line table.
            if advance {
                st.write_seqs("num_accepted_tokens", |_| 1i32, 1)?;
            }
            st.run("round", eager)?;
            let out = (st.read_i64("draft_tokens")?, st.read_i64("verify_tokens")?);
            drop(st);
            out
        } else {
            st.run("draft", eager)?;
            let drafts = st.read_i64("draft_tokens")?;
            drop(st);

            // Verify: [anchor, d0..] per sequence at pos.., causal; row i of
            // a group answers "what follows position pos+i".
            let cells: Vec<Cell> = running
                .iter()
                .zip(&drafts)
                .map(|(s, d)| {
                    let mut ids = vec![s.next as i64];
                    ids.extend_from_slice(d);
                    Cell { row: &s.row, ids, pos: s.pos, col: 0 }
                })
                .collect();
            let mut st = tray.stage(&cells, bucket)?;
            if advance {
                st.write_seqs("num_accepted_tokens", |_| 1i32, 1)?;
            }
            st.run("verify", eager)?;
            let vt = st.read_i64("verify_tokens")?;
            // Every row's tap into the draft KV (positions/slot_mapping are
            // still verify's): rejected rows land past the sequence's new
            // position and the next round overwrites them.
            st.run("draft_precompute", eager)?;
            (drafts, vt)
        };

        // Accept the longest matching prefix; vt[a] is the correction (or
        // the bonus token when everything matched).
        let accepted: Vec<usize> =
            drafts.iter().zip(&vt).map(|(d, v)| d.iter().zip(v).take_while(|(x, y)| x == y).count()).collect();
        if advance && !fused {
            // Commit the accepted rows into the recurrent state: the
            // target re-runs verify's rows from the state after the anchor
            // and stores after the last accepted one — the line moves to
            // entry `a` of its cell, `num_accepted_tokens` = a + 1.
            let cells: Vec<Cell> = running
                .iter()
                .zip(&drafts)
                .zip(&accepted)
                .map(|((s, d), &a)| {
                    let mut ids = vec![s.next as i64];
                    ids.extend_from_slice(d);
                    Cell { row: &s.row, ids, pos: s.pos, col: a }
                })
                .collect();
            let mut st = tray.stage(&cells, bucket)?;
            st.write_seqs("num_accepted_tokens", |i| accepted[i] as i32 + 1, 1)?;
            st.run("advance", eager)?;
        }
        debug_assert_eq!(vt.len(), n);
        self.stats.step_ns += t0.elapsed().as_nanos();
        self.stats.steps += 1;

        let running = std::mem::take(&mut self.running);
        for (i, mut s) in running.into_iter().enumerate() {
            let v = &vt[i];
            let a = accepted[i];
            let plan = self.spec.as_mut().unwrap();
            for p in &mut plan.counters.num_accepted_tokens_per_pos[..a] {
                *p += 1;
            }
            plan.counters.num_drafts += 1;
            plan.counters.num_draft_tokens += nd as u64;
            plan.counters.num_accepted_tokens += a as u64;
            let toks: Vec<u32> = v[..=a].iter().map(|&t| t as u32).collect();
            s.advance(std::iter::once(s.next).chain(toks[..a].iter().copied()));
            let (n, done) = s.emit(&toks, &self.policy.stop_tokens, ledger);
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

    /// One decode step over every running sequence. Under the span
    /// contract the oldest sequence still feeding its prompt feeds a run of
    /// up to `chunk` tokens, a row each, within the step's row bound; only
    /// the run's last output matters, as with a prefill chunk.
    fn decode(&mut self, ledger: &mut RequestLedger) -> Result<()> {
        self.drop_aborted(ledger);
        let n = self.running.len();
        if n == 0 {
            return Ok(());
        }
        let t0 = Instant::now();
        let room = self.tray.shape().seqs_max + 1 - n;
        let cap = self.span.map_or(1, |mx| mx.min(self.policy.chunk).min(room));
        let runner = (cap > 1).then(|| self.running.iter().position(|s| !s.pending.is_empty())).flatten();
        let fed: Vec<Vec<u32>> = self
            .running
            .iter_mut()
            .enumerate()
            .map(|(i, s)| {
                let run = if Some(i) == runner { (cap - 1).min(s.pending.len()) } else { 0 };
                std::iter::once(s.next).chain(s.pending.drain(..run)).collect()
            })
            .collect();
        let cells: Vec<Cell> = self
            .running
            .iter()
            .zip(&fed)
            .map(|(s, ids)| Cell { row: &s.row, ids: ids.iter().map(|&t| t as i64).collect(), pos: s.pos, col: 0 })
            .collect();
        let mut st = self.tray.stage(&cells, bucket)?;
        let program = match (runner.is_some(), st.b() > 1 && self.decode_batch) {
            (true, _) => "decode_span",
            (false, true) => "decode_batch",
            (false, false) => "decode",
        };
        st.run(program, self.policy.eager)?;
        let out = st.read_i64("next_token")?;
        drop(st);
        self.stats.step_ns += t0.elapsed().as_nanos();
        self.stats.steps += 1;

        let running = std::mem::take(&mut self.running);
        for (i, mut s) in running.into_iter().enumerate() {
            s.advance(fed[i].iter().copied());
            self.stats.prefill_tokens += (fed[i].len() - 1) as u64;
            // A row still feeding its prompt: this step's output is
            // dropped, the next prompt token is the next input.
            let done = match s.pending.pop_front() {
                Some(t) => {
                    s.next = t;
                    self.stats.prefill_tokens += 1;
                    false
                }
                None => {
                    let (n, done) = s.emit(&[out[i][0] as u32], &self.policy.stop_tokens, ledger);
                    self.stats.tokens += n;
                    done
                }
            };
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
    /// over its own window). `steps` are speculative rounds under
    /// `--spec`, and `accepted` / `accept_pct` are the window's.
    fn log_stats(&mut self) {
        let st = &self.stats;
        let dt = st.since.elapsed();
        if dt.as_secs() < 5 {
            return;
        }
        if st.tokens > 0 || st.prefill_tokens > 0 {
            let round = |x: f64, d: f64| (x * d).round() / d;
            let host = self.tray.host_tier();
            let (slots, slots_used) = self.tray.seq_slots();
            let (drafts, draft_tokens, accepted) = self.spec.as_ref().map_or((0, 0, 0), |p| {
                let c = &p.counters;
                (c.num_drafts - st.spec_at.0, c.num_draft_tokens - st.spec_at.1, c.num_accepted_tokens - st.spec_at.2)
            });
            info!(
                running = self.running.len(),
                rows_per_rank = ?self.rows_per_rank(),
                waiting = self.waiting.len() + self.waking.len(),
                kv_pct = round(self.tray.pages_used() as f64 * 100.0 / self.tray.pages_total().max(1) as f64, 10.0),
                steps = st.steps,
                step_ms = round(st.step_ns as f64 / 1e6 / st.steps.max(1) as f64, 100.0),
                tok_s = round(st.tokens as f64 / dt.as_secs_f64(), 1.0),
                prefill_tokens = st.prefill_tokens,
                prefill_tok_s =
                    (st.prefill_ns > 0).then(|| round(st.prefill_tokens as f64 / (st.prefill_ns as f64 / 1e9), 1.0)),
                prefix_hit_tokens = st.prefix_hit_tokens,
                checkpoints = self.prefix.len(),
                evictions = st.evictions,
                parked = host.map(|_| self.prefix.count(Tier::Parked)),
                host_gib = host.map(|(u, _)| round(u as f64 / (1u64 << 30) as f64, 10.0)),
                parks = host.map(|_| st.parks),
                host_evictions = host.map(|_| st.host_evictions),
                wakes = host.map(|_| st.wakes),
                wake_tokens = host.map(|_| st.wake_tokens),
                slots_used = self.tray.has_seq_state().then_some(slots_used),
                slots = self.tray.has_seq_state().then_some(slots),
                remaps = self.tray.remaps(),
                accepted = self.spec.as_ref().map(|_| round((accepted + drafts) as f64 / drafts.max(1) as f64, 100.0)),
                accept_pct =
                    self.spec.as_ref().map(|_| round(accepted as f64 * 100.0 / draft_tokens.max(1) as f64, 1.0)),
                "stats"
            );
        }
        self.stats = Stats::new(&self.spec);
    }
}

impl Scheduler for KernScheduler {
    fn submit(&mut self, request: QueuedRequest) {
        self.waiting.push_back(request);
    }

    fn step(&mut self, ledger: &mut RequestLedger) -> Result<()> {
        self.admit(ledger)?;
        if self.spec.is_some() {
            self.spec_round(ledger)?;
        } else {
            self.decode(ledger)?;
        }
        self.log_stats();
        Ok(())
    }

    fn metrics(&self) -> SchedulerMetrics {
        SchedulerMetrics {
            kv_used_blocks: self.tray.pages_used() as u64,
            kv_total_blocks: self.tray.pages_total() as u64,
            num_running_reqs: self.running.len() as u64,
            num_waiting_reqs: self.waiting.len() as u64,
            spec_decode: self.spec.as_ref().map(|p| p.counters),
        }
    }
}

fn need_programs(m: &Manifest, names: &[&str]) -> Result<()> {
    match names.iter().find(|p| !m.programs.contains_key(**p)) {
        Some(p) => bail!("manifest has no program `{p}`"),
        None => Ok(()),
    }
}

fn shape<'m>(m: &'m Manifest, name: &str) -> Result<&'m [Dim]> {
    m.buffers.get(name).map(|b| b.shape.as_slice()).with_context(|| format!("manifest has no buffer `{name}`"))
}

/// A buffer shaped `[seqs, n]`, one row per sequence: `n`.
fn seqs_rows(m: &Manifest, name: &str) -> Result<usize> {
    match shape(m, name)? {
        [Dim::Var(v), Dim::Const(n)] if v == "seqs" => Ok(*n as usize),
        s => bail!("`{name}` shaped {s:?}, expected [seqs, n]"),
    }
}

/// A buffer shaped `[seqs]`, one entry per sequence.
fn per_seq(m: &Manifest, name: &str) -> Result<()> {
    match shape(m, name)? {
        [Dim::Var(v)] if v == "seqs" => Ok(()),
        s => bail!("`{name}` shaped {s:?}, expected [seqs]"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kern_manifest::types::{Buffer, Spec};

    /// The plain contract: 8 tokens, 4 sequences, a 3-page `block_table`
    /// and a 3-line table over a recurrent state.
    fn plain() -> Manifest {
        Manifest::from_json(
            r#"{
            "schema_version": 3, "model": "t", "vars": {"tokens": {"max": 8}, "seqs": {"max": 4}},
            "states": {"kv": {"bytes_per_token": 1}, "gdn": {"bytes_per_seq": 24}},
            "buffers": {
                "token_ids": {"kind": "input", "dtype": "i64", "shape": ["tokens"]},
                "positions": {"kind": "input", "dtype": "i64", "shape": ["tokens"]},
                "slot_mapping": {"kind": "input", "dtype": "i64", "shape": ["tokens"], "domain": {"index_into": "kv"}},
                "seq_lens": {"kind": "input", "dtype": "i32", "shape": ["seqs"]},
                "cu_seqlens_q": {"kind": "input", "dtype": "i32", "shape": ["seqs"]},
                "block_table": {"kind": "input", "dtype": "i32", "shape": ["seqs", 3], "domain": {"index_into": "kv", "stride": 16}},
                "line_index": {"kind": "input", "dtype": "i32", "shape": [3, "seqs"], "domain": {"index_into": "gdn", "stride": 8}},
                "next_token": {"kind": "output", "dtype": "i64", "shape": ["seqs"]}
            },
            "modules": {}, "ops": {}, "programs": {"prefill": [], "decode": [], "decode_batch": []}
        }"#,
        )
        .unwrap()
    }

    fn buffer(kind: &str, shape: &str) -> Buffer {
        serde_json::from_str(&format!(r#"{{"kind": "{kind}", "dtype": "i64", "shape": {shape}}}"#)).unwrap()
    }

    /// The same plus DSpark's contract: 3 drafts, `spec.block` 4.
    fn speculative() -> Manifest {
        let mut m = plain();
        m.spec = Some(Spec { block: 4, mask_token: 7 });
        m.buffers.insert("draft_tokens".into(), buffer("output", r#"["seqs", 3]"#));
        m.buffers.insert("verify_tokens".into(), buffer("output", r#"["seqs", 4]"#));
        m.buffers.insert("anchor_token".into(), buffer("input", r#"["seqs"]"#));
        for p in ["decode_spec", "draft_precompute", "draft", "verify"] {
            m.programs.insert(p.into(), vec![]);
        }
        m
    }

    fn check_on(m: &Manifest, ranks: usize, spec: bool) -> Result<(Shape, Contract)> {
        let shape = Shape::check(m, &["line_index"], &["block_table"], 1)?;
        let c = Contract::check(m, 16, ranks, spec)?;
        Ok((shape, c))
    }

    fn check(m: &Manifest, spec: bool) -> Result<Contract> {
        check_on(m, 1, spec).map(|(_, c)| c)
    }

    fn rejects(m: &Manifest, spec: bool, what: &str) {
        let Err(e) = check(m, spec) else { panic!("accepted, expected `{what}`") };
        let e = format!("{e:#}");
        assert!(e.contains(what), "{e}");
    }

    #[test]
    fn plain_contract() {
        let (shape, c) = check_on(&plain(), 1, false).unwrap();
        assert_eq!((c.prefill, c.decode_batch, c.spec.is_none()), (Some(false), true, true));
        assert_eq!((c.max_seqs(&shape, 0), c.max_seqs(&shape, 3), c.max_seqs(&shape, 100)), (1, 3, 4));
    }

    #[test]
    fn prefill_emits_when_it_writes_next_token() {
        let mut m = plain();
        let call = serde_json::from_str(r#"{"op": "head", "args": [{"buf": "next_token"}]}"#).unwrap();
        m.programs.insert("prefill".into(), vec![call]);
        assert_eq!(check(&m, false).unwrap().prefill, Some(true));
    }

    #[test]
    fn decode_only_manifests_feed_the_prompt_through_decode() {
        let mut m = plain();
        m.programs.remove("prefill");
        m.programs.remove("decode_batch");
        let c = check(&m, false).unwrap();
        assert_eq!((c.prefill, c.decode_batch), (None, false));
        rejects(&m, true, "--spec needs a `prefill` program");
    }

    #[test]
    fn plain_rejections() {
        let mut m = plain();
        m.programs.remove("decode");
        rejects(&m, false, "no program `decode`");
        let Err(e) = check_on(&speculative(), 4, true) else { panic!("--spec on 4 ranks accepted") };
        assert!(format!("{e:#}").contains("--spec drives one rank; this tray has 4"), "{e:#}");
    }

    #[test]
    fn speculative_contract() {
        let (shape, c) = check_on(&speculative(), 1, true).unwrap();
        let s = c.spec.as_ref().unwrap();
        assert_eq!(
            (s.n_drafts, s.draft_rows, s.verify_rows, s.mask_token, s.advance, s.fused),
            (3, 4, 4, 7, false, false)
        );
        assert_eq!(s.counters.num_spec_tokens, 3);
        // Four rows per sequence per round fit twice in 8 tokens.
        assert_eq!((c.max_seqs(&shape, 1), c.max_seqs(&shape, 4)), (1, 2));
        // A plain manifest has no `decode_batch` to miss under --spec.
        let mut m = speculative();
        m.programs.remove("decode_batch");
        assert!(check(&m, true).is_ok());
    }

    #[test]
    fn speculative_rejections() {
        // The draft rows and the mask token come from the manifest, nowhere else.
        let mut m = speculative();
        m.spec = None;
        rejects(&m, true, "no `spec` block");
        let mut m = speculative();
        m.programs.remove("verify");
        rejects(&m, true, "--spec: the manifest's speculative contract: manifest has no program `verify`");
        let mut m = speculative();
        m.buffers.get_mut("verify_tokens").unwrap().shape = vec![Dim::Var("seqs".into()), Dim::Const(5)];
        rejects(&m, true, "verify_tokens has 5 rows per sequence, expected 4");
        let mut m = speculative();
        m.buffers.get_mut("anchor_token").unwrap().shape = vec![Dim::Const(4)];
        rejects(&m, true, "`anchor_token` shaped [Const(4)], expected [seqs]");
        let mut m = speculative();
        m.programs.insert("advance".into(), vec![]);
        rejects(&m, true, "`advance` without a `num_accepted_tokens`");
        let mut m = speculative();
        m.buffers.insert("num_accepted_tokens".into(), buffer("input", r#"["seqs"]"#));
        rejects(&m, true, "no `advance` program");
        m.programs.insert("advance".into(), vec![]);
        assert!(check(&m, true).unwrap().spec.unwrap().advance);
        // `round` fuses draft and verify: their rows must coincide.
        let mut m = speculative();
        m.programs.insert("round".into(), vec![]);
        m.programs.remove("draft");
        assert!(check(&m, true).unwrap().spec.unwrap().fused);
        m.spec = Some(Spec { block: 3, mask_token: 7 });
        rejects(&m, true, "coincide, got 3 and 4");
        // A round's rows must fit the pad page.
        let Err(e) = Contract::check(&speculative(), 2, 1, true) else { panic!("4 rows in a 2-token page") };
        assert!(format!("{e:#}").contains("exceed the 2-token pad page"), "{e:#}");
    }
}
