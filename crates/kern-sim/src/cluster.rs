//! Pure-CPU discrete-event cluster simulator over [`kern_manifest::Protocol`]
//! and [`kern_pool::{Pool, Prefix}`].
//!
//! Each simulated endpoint runs the same scheduling state machine and memory
//! invariants as `kern-serve`:
//! - static KV page and sequence-slot capacity from [`MemoryBudget`], with one
//!   pad page leased up front;
//! - all-or-nothing worst-case page + slot reservation at admission
//!   (`prompt + output + rows - 1`), evicting cold [`Prefix`] snapshots on
//!   [`Denied::Busy`] and never preempting running sequences;
//! - chunk-by-chunk prefill (`Rows::Var`) interleaved with bucket-padded
//!   decode / speculative steps (`Rows::Const(rows)`), priced in nanoseconds
//!   per step via [`PreparedProgram`];
//! - asynchronous KV insert/evict notifications to [`Router`] after
//!   `kv_event_lag_ns`, plus closed-loop multi-turn session wakeup after
//!   `think_time_ns`.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap, VecDeque};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use kern_manifest::protocol::{Forward, Rows};
use kern_manifest::{Protocol, Verified};
use kern_pool::{Checkpoint, Denied, Lease, Pool, Prefix};
use serde::Serialize;

use crate::cost::{CostBook, MatchTier, PreparedProgram, PriceOptions};
use crate::memory::{DeviceMemorySpec, MemoryBudget};
use crate::router::{EndpointLoad, Router, RouterPolicy, ShadowTrie};
use crate::trace::{generated_token_id, ArrivalMode, RequestTrace, TraceRequest};

const BUCKETS: [usize; 18] = [
    1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 640, 768, 1024,
];

fn bucket_size(n: usize) -> usize {
    BUCKETS.iter().copied().find(|&b| b >= n).unwrap_or(n)
}

/// Configuration for one simulated serving endpoint.
#[derive(Debug, Clone)]
pub struct EndpointConfig {
    pub name: String,
    pub manifest: Verified,
    pub memory: DeviceMemorySpec,
    /// Optional prefill chunk cap (`None` uses the manifest's `rows.max`).
    pub chunk_size: Option<usize>,
    /// Optional concurrent sequence cap (`None` uses the manifest's `groups` bound).
    pub max_seqs: Option<usize>,
    /// Optional fixed-row step width (`None` uses the widest declared `Rows::Const`).
    pub step_rows: Option<u64>,
    /// Mean accepted output tokens per step (`1.0` for standard decode; `> 1.0`
    /// for speculative draft-verify rounds).
    pub mean_accepted_tokens: f64,
}

impl EndpointConfig {
    pub fn standard(name: impl Into<String>, manifest: Verified, memory: DeviceMemorySpec) -> Self {
        Self {
            name: name.into(),
            manifest,
            memory,
            chunk_size: None,
            max_seqs: None,
            step_rows: None,
            mean_accepted_tokens: 1.0,
        }
    }
}

/// Optional cluster-wide L2 KV pool (e.g. Mooncake Store / NVMe tier) shared across endpoints.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct SharedPoolConfig {
    /// Effective read bandwidth per endpoint when fetching cached pages/slots, in GiB/s.
    pub bandwidth_gib_s: f64,
    /// Fixed probe + RPC round-trip overhead per fetch, in microseconds.
    pub lookup_us: f64,
}

/// Full multi-endpoint simulation configuration.
#[derive(Debug, Clone)]
pub struct ClusterConfig {
    pub endpoints: Vec<EndpointConfig>,
    pub router: RouterPolicy,
    pub arrival_mode: ArrivalMode,
    pub price_options: PriceOptions,
    pub shared_pool: Option<SharedPoolConfig>,
}

/// Summary distribution metrics (in milliseconds).
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct LatencySummary {
    pub mean_ms: f64,
    pub p50_ms: f64,
    pub p90_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
}

/// Per-endpoint simulation telemetry.
#[derive(Debug, Clone, Serialize)]
pub struct EndpointReport {
    pub name: String,
    pub model: String,
    pub routed_requests: usize,
    pub completed_requests: usize,
    pub rejected_requests: usize,
    pub busy_time_s: f64,
    pub busy_fraction: f64,
    pub prefill_chunks: u64,
    pub decode_steps: u64,
    pub mean_decode_batch: f64,
    pub prompt_tokens: u64,
    pub local_prefix_hit_tokens: u64,
    pub shared_pool_hit_tokens: u64,
    pub prefix_hit_rate: f64,
    pub evictions: u64,
    pub peak_pages_used: usize,
    pub scheduler_pages: usize,
    pub peak_slots_used: usize,
    pub provisioned_slots: u64,
}

/// Cluster-wide simulation report.
#[derive(Debug, Clone, Serialize)]
pub struct ClusterReport {
    pub trace: String,
    pub arrival_mode: ArrivalMode,
    pub router_policy: RouterPolicy,
    pub wall_time_s: f64,
    pub total_requests: usize,
    pub completed_requests: usize,
    pub rejected_requests: usize,
    pub total_prompt_tokens: u64,
    pub total_output_tokens: u64,
    pub total_prefix_hit_tokens: u64,
    pub prefix_hit_rate: f64,
    pub total_evictions: u64,
    pub output_tok_per_s: f64,
    pub total_tok_per_s: f64,
    pub ttft: LatencySummary,
    pub tpot: LatencySummary,
    pub e2e: LatencySummary,
    pub exact_time_coverage: f64,
    pub calibrated_time_coverage: f64,
    pub endpoints: Vec<EndpointReport>,
}

/// Pure-CPU discrete-event cluster simulator.
pub struct SimCluster;

impl SimCluster {
    /// Replay `trace` across `cfg` priced by `book`.
    pub fn run(book: &CostBook, cfg: &ClusterConfig, trace: &RequestTrace) -> Result<ClusterReport> {
        if cfg.endpoints.is_empty() {
            bail!("cluster configuration requires at least one endpoint");
        }

        let mut endpoints = Vec::with_capacity(cfg.endpoints.len());
        for (idx, ep_cfg) in cfg.endpoints.iter().enumerate() {
            endpoints.push(
                SimEndpoint::new(idx, ep_cfg, book, cfg.price_options)
                    .with_context(|| format!("initializing endpoint `{}`", ep_cfg.name))?,
            );
        }

        let mut router = Router::new(cfg.router.clone(), endpoints.len());
        let lag_ns = cfg.router.kv_event_lag_ns();
        let mut shared_prefixes = SharedPrefixIndex::default();

        // Build closed-loop session chains (`session_id` -> ordered slice of request indices).
        let mut session_chains: BTreeMap<String, VecDeque<usize>> = BTreeMap::new();
        if cfg.arrival_mode == ArrivalMode::ClosedLoop {
            for (idx, req) in trace.requests.iter().enumerate() {
                if let Some(sid) = &req.session_id {
                    session_chains.entry(sid.clone()).or_default().push_back(idx);
                }
            }
        }

        let mut events: BinaryHeap<Reverse<SimEvent>> = BinaryHeap::new();
        let mut seq_counter = 0u64;

        match cfg.arrival_mode {
            ArrivalMode::OpenLoop => {
                for (idx, req) in trace.requests.iter().enumerate() {
                    push_event(
                        &mut events,
                        &mut seq_counter,
                        req.arrival_ns,
                        EventKind::RequestArrival { req_idx: idx },
                    );
                }
            }
            ArrivalMode::ClosedLoop => {
                for (idx, req) in trace.requests.iter().enumerate() {
                    if req.session_id.is_none() {
                        push_event(
                            &mut events,
                            &mut seq_counter,
                            req.arrival_ns,
                            EventKind::RequestArrival { req_idx: idx },
                        );
                    }
                }
                for q in session_chains.values_mut() {
                    if let Some(first_idx) = q.pop_front() {
                        push_event(
                            &mut events,
                            &mut seq_counter,
                            trace.requests[first_idx].arrival_ns,
                            EventKind::RequestArrival { req_idx: first_idx },
                        );
                    }
                }
            }
        }

        let mut actual_arrival_ns = vec![0u64; trace.requests.len()];
        let mut ttft_samples_ms = Vec::with_capacity(trace.requests.len());
        let mut tpot_samples_ms = Vec::with_capacity(trace.requests.len());
        let mut e2e_samples_ms = Vec::with_capacity(trace.requests.len());
        let mut total_output_tokens = 0u64;
        let mut us_by_tier: BTreeMap<MatchTier, f64> = BTreeMap::new();

        let mut first_arrival_ns: Option<u64> = None;
        let mut last_finish_ns = 0u64;

        while let Some(Reverse(ev)) = events.pop() {
            let now_ns = ev.time_ns;
            match ev.kind {
                EventKind::KvIndexUpdate { endpoint_id, update } => match update {
                    KvUpdate::Insert { key, has_slot } => {
                        router.on_insert(endpoint_id, &key, has_slot);
                    }
                    KvUpdate::Evict { key } => {
                        router.on_evict(endpoint_id, &key);
                    }
                },
                EventKind::RequestArrival { req_idx } => {
                    let req = &trace.requests[req_idx];
                    first_arrival_ns.get_or_insert(now_ns);
                    actual_arrival_ns[req_idx] = now_ns;

                    let loads: Vec<EndpointLoad> = endpoints.iter().map(SimEndpoint::load_snapshot).collect();
                    let ep_id = router.route(&req.prompt_tokens, req.output_tokens, &loads);
                    endpoints[ep_id].enqueue(req_idx, req);
                    if !endpoints[ep_id].busy {
                        drive_endpoint(
                            &mut endpoints[ep_id],
                            now_ns,
                            book,
                            trace,
                            cfg.shared_pool.as_ref(),
                            &mut shared_prefixes,
                            lag_ns,
                            &mut events,
                            &mut seq_counter,
                            &mut us_by_tier,
                        )?;
                    }
                }
                EventKind::EndpointStepDone { endpoint_id } => {
                    let completed = endpoints[endpoint_id].complete_inflight(
                        now_ns,
                        trace,
                        &mut shared_prefixes,
                        lag_ns,
                        &mut events,
                        &mut seq_counter,
                    );
                    for done in completed {
                        last_finish_ns = last_finish_ns.max(done.finish_ns);
                        let arr = actual_arrival_ns[done.req_idx];
                        let ttft_ns = done.first_token_ns.saturating_sub(arr);
                        let e2e_ns = done.finish_ns.saturating_sub(arr);
                        ttft_samples_ms.push(ttft_ns as f64 / 1e6);
                        e2e_samples_ms.push(e2e_ns as f64 / 1e6);
                        if done.output_tokens > 1 {
                            let decode_ns = done.finish_ns.saturating_sub(done.first_token_ns);
                            tpot_samples_ms
                                .push((decode_ns as f64 / 1e6) / (done.output_tokens - 1) as f64);
                        }
                        total_output_tokens += done.output_tokens as u64;

                        if cfg.arrival_mode == ArrivalMode::ClosedLoop {
                            if let Some(sid) = &trace.requests[done.req_idx].session_id {
                                if let Some(next_idx) = session_chains.get_mut(sid).and_then(VecDeque::pop_front) {
                                    let next_req = &trace.requests[next_idx];
                                    push_event(
                                        &mut events,
                                        &mut seq_counter,
                                        done.finish_ns.saturating_add(next_req.think_time_ns),
                                        EventKind::RequestArrival { req_idx: next_idx },
                                    );
                                }
                            }
                        }
                    }

                    drive_endpoint(
                        &mut endpoints[endpoint_id],
                        now_ns,
                        book,
                        trace,
                        cfg.shared_pool.as_ref(),
                        &mut shared_prefixes,
                        lag_ns,
                        &mut events,
                        &mut seq_counter,
                        &mut us_by_tier,
                    )?;
                }
            }
        }

        let start_ns = first_arrival_ns.unwrap_or(0);
        let wall_ns = last_finish_ns.saturating_sub(start_ns);
        let wall_time_s = (wall_ns as f64 / 1e9).max(1e-9);

        let mut ep_reports = Vec::with_capacity(endpoints.len());
        let mut completed_requests = 0usize;
        let mut rejected_requests = 0usize;
        let mut total_prompt_tokens = 0u64;
        let mut total_prefix_hit_tokens = 0u64;
        let mut total_evictions = 0u64;

        for ep in &endpoints {
            let busy_s = ep.busy_ns as f64 / 1e9;
            let hit_toks = ep.local_prefix_hit_tokens + ep.shared_pool_hit_tokens;
            let hit_rate = if ep.prompt_tokens > 0 {
                hit_toks as f64 / ep.prompt_tokens as f64
            } else {
                0.0
            };
            completed_requests += ep.completed_requests;
            rejected_requests += ep.rejected_requests;
            total_prompt_tokens += ep.prompt_tokens;
            total_prefix_hit_tokens += hit_toks;
            total_evictions += ep.evictions;

            ep_reports.push(EndpointReport {
                name: ep.name.clone(),
                model: ep.manifest.model.clone(),
                routed_requests: ep.routed_requests,
                completed_requests: ep.completed_requests,
                rejected_requests: ep.rejected_requests,
                busy_time_s: busy_s,
                busy_fraction: if wall_ns > 0 {
                    (ep.busy_ns as f64 / wall_ns as f64).min(1.0)
                } else {
                    0.0
                },
                prefill_chunks: ep.prefill_chunks,
                decode_steps: ep.decode_steps,
                mean_decode_batch: if ep.decode_steps > 0 {
                    ep.decode_batch_sum as f64 / ep.decode_steps as f64
                } else {
                    0.0
                },
                prompt_tokens: ep.prompt_tokens,
                local_prefix_hit_tokens: ep.local_prefix_hit_tokens,
                shared_pool_hit_tokens: ep.shared_pool_hit_tokens,
                prefix_hit_rate: hit_rate,
                evictions: ep.evictions,
                peak_pages_used: ep.peak_pages_used,
                scheduler_pages: ep.budget.scheduler_pages,
                peak_slots_used: ep.peak_slots_used,
                provisioned_slots: ep.budget.provisioned_slots,
            });
        }

        let tier_sum: f64 = us_by_tier.values().sum();
        let exact_us = us_by_tier.get(&MatchTier::ExactSha256).copied().unwrap_or(0.0);
        let abi_us = us_by_tier.get(&MatchTier::EntryAbi).copied().unwrap_or(0.0);

        Ok(ClusterReport {
            trace: trace.name.clone(),
            arrival_mode: cfg.arrival_mode,
            router_policy: cfg.router.clone(),
            wall_time_s: if wall_ns > 0 { wall_ns as f64 / 1e9 } else { 0.0 },
            total_requests: trace.requests.len(),
            completed_requests,
            rejected_requests,
            total_prompt_tokens,
            total_output_tokens,
            total_prefix_hit_tokens,
            prefix_hit_rate: if total_prompt_tokens > 0 {
                total_prefix_hit_tokens as f64 / total_prompt_tokens as f64
            } else {
                0.0
            },
            total_evictions,
            output_tok_per_s: if wall_ns > 0 {
                total_output_tokens as f64 / wall_time_s
            } else {
                0.0
            },
            total_tok_per_s: if wall_ns > 0 {
                (total_prompt_tokens + total_output_tokens) as f64 / wall_time_s
            } else {
                0.0
            },
            ttft: summarize_ms(&ttft_samples_ms),
            tpot: summarize_ms(&tpot_samples_ms),
            e2e: summarize_ms(&e2e_samples_ms),
            exact_time_coverage: if tier_sum > 0.0 { exact_us / tier_sum } else { 1.0 },
            calibrated_time_coverage: if tier_sum > 0.0 {
                (exact_us + abi_us) / tier_sum
            } else {
                1.0
            },
            endpoints: ep_reports,
        })
    }
}

// --- Discrete-event priority queue types ---

#[derive(Clone, Eq, PartialEq)]
struct SimEvent {
    time_ns: u64,
    priority: u8,
    seq: u64,
    kind: EventKind,
}

impl Ord for SimEvent {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.time_ns, self.priority, self.seq).cmp(&(other.time_ns, other.priority, other.seq))
    }
}

impl PartialOrd for SimEvent {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Eq, PartialEq)]
enum EventKind {
    KvIndexUpdate { endpoint_id: usize, update: KvUpdate },
    EndpointStepDone { endpoint_id: usize },
    RequestArrival { req_idx: usize },
}

#[derive(Clone, Eq, PartialEq)]
enum KvUpdate {
    Insert { key: Arc<[i64]>, has_slot: bool },
    Evict { key: Arc<[i64]> },
}

fn push_event(
    heap: &mut BinaryHeap<Reverse<SimEvent>>,
    seq_counter: &mut u64,
    time_ns: u64,
    kind: EventKind,
) {
    let priority = match &kind {
        EventKind::KvIndexUpdate { .. } => 0,
        EventKind::EndpointStepDone { .. } => 1,
        EventKind::RequestArrival { .. } => 2,
    };
    let seq = *seq_counter;
    *seq_counter += 1;
    heap.push(Reverse(SimEvent {
        time_ns,
        priority,
        seq,
        kind,
    }));
}

// --- Shared L2 pool index ---

#[derive(Default)]
struct SharedPrefixIndex {
    entries: Vec<(Arc<[i64]>, bool)>,
}

impl SharedPrefixIndex {
    fn put(&mut self, key: Arc<[i64]>, has_slot: bool) {
        if !self.entries.iter().any(|(k, _)| k.as_ref() == key.as_ref()) {
            self.entries.push((key, has_slot));
        }
    }

    fn probe(&self, prompt: &[i64], unit: usize) -> usize {
        let Some(usable) = prompt.len().checked_sub(1) else {
            return 0;
        };
        let q = &prompt[..usable];
        let mut best = 0usize;
        for (k, has_slot) in &self.entries {
            if *has_slot {
                if k.len() <= q.len() && q[..k.len()] == k[..] {
                    best = best.max(k.len());
                }
            } else {
                let shared = k.iter().zip(q).take_while(|(a, b)| a == b).count();
                let aligned = if shared == k.len() && k.len() <= q.len() {
                    k.len()
                } else {
                    (shared / unit) * unit
                };
                best = best.max(aligned);
            }
        }
        best
    }
}

// --- Endpoint state machine ---

struct QueuedItem {
    req_idx: usize,
    prompt_len: usize,
    est_uncached_len: usize,
    output_tokens: usize,
}

struct ActiveSeq {
    req_idx: usize,
    output_tokens: usize,
    /// Current token count materialized in the KV state.
    pos: usize,
    /// Prompt tokens that must go through the variable-row `chunk` program.
    prefill_target: usize,
    generated: usize,
    first_token_ns: Option<u64>,
    frac_carry: f64,
    lease: Lease,
}

enum InflightOp {
    PrefillChunk { seq_slot: usize, rows: usize },
    DecodeStep { rows_per_seq: usize },
}

struct CompletedReq {
    req_idx: usize,
    first_token_ns: u64,
    finish_ns: u64,
    output_tokens: usize,
}

struct SimEndpoint {
    id: usize,
    name: String,
    manifest: Verified,
    protocol: Protocol,
    budget: MemoryBudget,
    pool: Arc<Pool>,
    _pad_lease: Option<Lease>,
    prefix: Prefix<Checkpoint>,
    local_shadow: ShadowTrie,
    price_options: PriceOptions,
    prepared_cache: BTreeMap<(String, BTreeMap<String, u64>), PreparedProgram>,

    chunk_prog: Forward,
    chunk_cap: usize,
    step_rows: u64,
    max_seqs: usize,
    mean_accepted: f64,
    rank_count: u64,

    waiting: VecDeque<QueuedItem>,
    queued_prefill_tokens: usize,
    prefilling: VecDeque<ActiveSeq>,
    running: Vec<ActiveSeq>,
    busy: bool,
    inflight: Option<InflightOp>,

    // Telemetry
    routed_requests: usize,
    completed_requests: usize,
    rejected_requests: usize,
    busy_ns: u64,
    prefill_chunks: u64,
    decode_steps: u64,
    decode_batch_sum: u64,
    prompt_tokens: u64,
    local_prefix_hit_tokens: u64,
    shared_pool_hit_tokens: u64,
    evictions: u64,
    peak_pages_used: usize,
    peak_slots_used: usize,
}

impl SimEndpoint {
    fn new(
        id: usize,
        cfg: &EndpointConfig,
        _book: &CostBook,
        price_options: PriceOptions,
    ) -> Result<Self> {
        let protocol = Protocol::check_unsampled(&cfg.manifest)?;
        let budget = MemoryBudget::compute(&cfg.manifest, cfg.memory)?;
        let pool = Arc::new(Pool::new(
            &cfg.manifest,
            budget.raw_pages,
            budget.provisioned_slots as usize,
        )?);
        let pad_lease = if budget.page_bytes > 0 {
            Some(
                pool.lease(budget.page_unit as usize)
                    .map_err(|e| anyhow::anyhow!("reserving pad page: {e}"))?,
            )
        } else {
            None
        };

        let shapes = protocol.row_shapes();
        let Some(step_rows) = cfg.step_rows.or_else(|| shapes.last().copied()) else {
            bail!("manifest declares no fixed-row step program");
        };
        if protocol.forward(1, Rows::Const(step_rows)).is_none() {
            bail!("manifest has no program accepting 1 sequence of {step_rows} rows");
        }
        let Some(chunk_prog) = protocol.chunk().cloned() else {
            bail!("manifest has no variable-row chunk program for prefill");
        };

        let chunk_cap = cfg
            .chunk_size
            .unwrap_or(protocol.rows.max as usize)
            .clamp(1, protocol.rows.max as usize);
        let max_groups = protocol.max_groups(Rows::Const(step_rows)) as usize;
        let slot_cap = if pool.has_slots() {
            budget.provisioned_slots.saturating_sub(1) as usize
        } else {
            usize::MAX
        };
        let max_seqs = cfg
            .max_seqs
            .unwrap_or(max_groups)
            .min(max_groups)
            .min(protocol.rows.max as usize / step_rows as usize)
            .min(slot_cap)
            .max(1);

        let mean_accepted = if step_rows == 1 {
            1.0
        } else {
            cfg.mean_accepted_tokens.clamp(1.0, step_rows as f64)
        };
        let rank_count = cfg
            .manifest
            .topology
            .as_ref()
            .map_or(1u64, |t| t.groups.values().copied().product::<u64>().max(1));

        Ok(Self {
            id,
            name: cfg.name.clone(),
            manifest: cfg.manifest.clone(),
            protocol,
            prefix: Prefix::new(budget.page_unit.max(1) as usize),
            local_shadow: ShadowTrie::default(),
            budget,
            pool,
            _pad_lease: pad_lease,
            price_options,
            prepared_cache: BTreeMap::new(),
            chunk_prog,
            chunk_cap,
            step_rows,
            max_seqs,
            mean_accepted,
            rank_count,
            waiting: VecDeque::new(),
            queued_prefill_tokens: 0,
            prefilling: VecDeque::new(),
            running: Vec::new(),
            busy: false,
            inflight: None,
            routed_requests: 0,
            completed_requests: 0,
            rejected_requests: 0,
            busy_ns: 0,
            prefill_chunks: 0,
            decode_steps: 0,
            decode_batch_sum: 0,
            prompt_tokens: 0,
            local_prefix_hit_tokens: 0,
            shared_pool_hit_tokens: 0,
            evictions: 0,
            peak_pages_used: 0,
            peak_slots_used: 0,
        })
    }

    fn load_snapshot(&self) -> EndpointLoad {
        let active_prefill_rem: usize = self
            .prefilling
            .iter()
            .map(|s| s.prefill_target.saturating_sub(s.pos))
            .sum();
        EndpointLoad {
            endpoint_id: self.id,
            page_unit: self.budget.page_unit as usize,
            max_seq_tokens: self.budget.max_seq_tokens,
            queued_requests: self.waiting.len() + self.prefilling.len(),
            queued_prefill_tokens: self.queued_prefill_tokens + active_prefill_rem,
            running_sequences: self.running.len(),
            max_sequences: self.max_seqs,
        }
    }

    fn enqueue(&mut self, req_idx: usize, req: &TraceRequest) {
        self.routed_requests += 1;
        let hit_len = self
            .local_shadow
            .longest_hit(&req.prompt_tokens, self.budget.page_unit.max(1) as usize);
        let est_uncached_len = req.prompt_tokens.len().saturating_sub(hit_len);
        self.queued_prefill_tokens += est_uncached_len;
        self.waiting.push_back(QueuedItem {
            req_idx,
            prompt_len: req.prompt_tokens.len(),
            est_uncached_len,
            output_tokens: req.output_tokens,
        });
    }

    fn record_pool_peaks(&mut self) {
        // Subtract 1 pad page so `peak_pages_used` reflects request/checkpoint KV pages.
        let used_pages = self
            .pool
            .used()
            .saturating_sub(usize::from(self._pad_lease.is_some()));
        self.peak_pages_used = self.peak_pages_used.max(used_pages);
        self.peak_slots_used = self.peak_slots_used.max(self.pool.slots_used());
    }

    fn price_step(
        &mut self,
        book: &CostBook,
        program: &str,
        vars: BTreeMap<String, u64>,
        sum_kv: u64,
        us_by_tier: &mut BTreeMap<MatchTier, f64>,
    ) -> Result<u64> {
        let key = (program.to_string(), vars);
        if !self.prepared_cache.contains_key(&key) {
            let prep = book.prepare_program(
                &self.manifest,
                &self.protocol.rows.var,
                &key.0,
                &key.1,
                self.price_options,
            )?;
            self.prepared_cache.insert(key.clone(), prep);
        }
        let timing = self.prepared_cache[&key].eval(book, sum_kv);
        for (&t, &u) in &timing.us_by_tier {
            *us_by_tier.entry(t).or_default() += u;
        }
        Ok(((timing.total_us * 1_000.0).round() as u64).max(1))
    }

    fn complete_inflight(
        &mut self,
        now_ns: u64,
        trace: &RequestTrace,
        shared_prefixes: &mut SharedPrefixIndex,
        lag_ns: u64,
        events: &mut BinaryHeap<Reverse<SimEvent>>,
        seq_counter: &mut u64,
    ) -> Vec<CompletedReq> {
        self.busy = false;
        let mut finished = Vec::new();
        let Some(op) = self.inflight.take() else {
            return finished;
        };

        match op {
            InflightOp::PrefillChunk { seq_slot, rows } => {
                let seq = &mut self.prefilling[seq_slot];
                seq.pos += rows;
                if seq.pos >= seq.prefill_target {
                    let mut seq = self.prefilling.remove(seq_slot).expect("active prefill seq");
                    if self.chunk_prog.emits.is_some() {
                        seq.generated = 1;
                        seq.first_token_ns = Some(now_ns);
                    }
                    if seq.generated >= seq.output_tokens {
                        let done = self.retire_sequence(
                            seq,
                            now_ns,
                            trace,
                            shared_prefixes,
                            lag_ns,
                            events,
                            seq_counter,
                        );
                        finished.push(done);
                    } else {
                        self.running.push(seq);
                    }
                }
            }
            InflightOp::DecodeStep { rows_per_seq } => {
                let mut idx = 0;
                while idx < self.running.len() {
                    let seq = &mut self.running[idx];
                    let raw = self.mean_accepted + seq.frac_carry;
                    let accepted = (raw.floor() as usize).clamp(1, rows_per_seq);
                    seq.frac_carry = raw - accepted as f64;
                    if seq.first_token_ns.is_none() {
                        seq.first_token_ns = Some(now_ns);
                    }
                    let rem = seq.output_tokens.saturating_sub(seq.generated);
                    let take = accepted.min(rem);
                    seq.pos += take;
                    seq.generated += take;

                    if seq.generated >= seq.output_tokens {
                        let done_seq = self.running.remove(idx);
                        let done = self.retire_sequence(
                            done_seq,
                            now_ns,
                            trace,
                            shared_prefixes,
                            lag_ns,
                            events,
                            seq_counter,
                        );
                        finished.push(done);
                    } else {
                        idx += 1;
                    }
                }
            }
        }
        finished
    }

    #[allow(clippy::too_many_arguments)]
    fn retire_sequence(
        &mut self,
        seq: ActiveSeq,
        now_ns: u64,
        trace: &RequestTrace,
        shared_prefixes: &mut SharedPrefixIndex,
        lag_ns: u64,
        events: &mut BinaryHeap<Reverse<SimEvent>>,
        seq_counter: &mut u64,
    ) -> CompletedReq {
        self.completed_requests += 1;
        let req = &trace.requests[seq.req_idx];
        let first_token_ns = seq.first_token_ns.unwrap_or(now_ns);

        if self.protocol.shards.positions.is_none() {
            let final_len = req.prompt_tokens.len() + seq.output_tokens;
            let mut full_tokens = Vec::with_capacity(final_len);
            full_tokens.extend_from_slice(&req.prompt_tokens);
            let sess_key = req
                .session_id
                .as_deref()
                .unwrap_or("anon");
            for off in 0..seq.output_tokens {
                full_tokens.push(generated_token_id(sess_key, req.turn_index, off));
            }
            let cp = self.pool.retire(seq.lease, final_len);
            let has_slot = cp.has_slot();
            self.prefix.insert(&full_tokens, cp);
            self.local_shadow.insert(&full_tokens, has_slot);
            let arc_key: Arc<[i64]> = Arc::from(full_tokens);
            shared_prefixes.put(Arc::clone(&arc_key), has_slot);
            push_event(
                events,
                seq_counter,
                now_ns.saturating_add(lag_ns),
                EventKind::KvIndexUpdate {
                    endpoint_id: self.id,
                    update: KvUpdate::Insert {
                        key: arc_key,
                        has_slot,
                    },
                },
            );
            self.record_pool_peaks();
        }

        CompletedReq {
            req_idx: seq.req_idx,
            first_token_ns,
            finish_ns: now_ns,
            output_tokens: seq.output_tokens,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn drive_endpoint(
    ep: &mut SimEndpoint,
    now_ns: u64,
    book: &CostBook,
    trace: &RequestTrace,
    shared_pool_cfg: Option<&SharedPoolConfig>,
    shared_prefixes: &mut SharedPrefixIndex,
    lag_ns: u64,
    events: &mut BinaryHeap<Reverse<SimEvent>>,
    seq_counter: &mut u64,
    us_by_tier: &mut BTreeMap<MatchTier, f64>,
) -> Result<()> {
    if ep.busy {
        return Ok(());
    }

    // 1. Admit as many waiting requests as fit `max_seqs` and `Pool` capacity.
    while ep.running.len() + ep.prefilling.len() < ep.max_seqs {
        let Some(front) = ep.waiting.front() else {
            break;
        };
        let req = &trace.requests[front.req_idx];
        let headroom = (ep.step_rows as usize).saturating_sub(1);
        let need_tokens = front.prompt_len + front.output_tokens + headroom;

        let local_hit = ep.prefix.lookup(&req.prompt_tokens);
        let mut transfer_extra_ns = 0u64;

        let lease_attempt = loop {
            let res = if let Some(hit) = &local_hit {
                ep.pool
                    .restore(&hit.found, hit.len, need_tokens)
                    .map(|(l, _)| (l, hit.len, 0usize))
            } else {
                ep.pool.lease(need_tokens).map(|l| (l, 0usize, 0usize))
            };
            match res {
                Ok(ok) => break Ok(ok),
                Err(Denied::Busy) => {
                    if let Some(evicted_key) = ep.prefix.evict() {
                        ep.evictions += 1;
                        ep.local_shadow.remove(&evicted_key);
                        push_event(
                            events,
                            seq_counter,
                            now_ns.saturating_add(lag_ns),
                            EventKind::KvIndexUpdate {
                                endpoint_id: ep.id,
                                update: KvUpdate::Evict { key: evicted_key },
                            },
                        );
                        continue;
                    }
                    break Err(Denied::Busy);
                }
                Err(fatal) => break Err(fatal),
            }
        };

        match lease_attempt {
            Ok((lease, local_len, _)) => {
                let item = ep.waiting.pop_front().expect("front exists");
                ep.queued_prefill_tokens =
                    ep.queued_prefill_tokens.saturating_sub(item.est_uncached_len);
                ep.prompt_tokens += item.prompt_len as u64;

                let mut start_pos = local_len;
                ep.local_prefix_hit_tokens += local_len as u64;

                if local_len == 0 {
                    if let Some(sp_cfg) = shared_pool_cfg {
                        let remote_len =
                            shared_prefixes.probe(&req.prompt_tokens, ep.budget.page_unit.max(1) as usize);
                        if remote_len > 0 {
                            start_pos = remote_len;
                            ep.shared_pool_hit_tokens += remote_len as u64;
                            let kv_bytes = (remote_len as u64) * ep.budget.token_unit_bytes
                                + if ep.pool.has_slots() { ep.budget.slot_unit_bytes } else { 0 };
                            let bw_bytes_per_ns = sp_cfg.bandwidth_gib_s.max(0.1) * ((1u64 << 30) as f64 / 1e9);
                            transfer_extra_ns = (sp_cfg.lookup_us * 1_000.0
                                + kv_bytes as f64 / bw_bytes_per_ns)
                                .round() as u64;
                        }
                    }
                }

                let prefill_target = if ep.chunk_prog.emits.is_some() {
                    item.prompt_len
                } else {
                    item.prompt_len.saturating_sub(1)
                };

                let seq = ActiveSeq {
                    req_idx: item.req_idx,
                    output_tokens: item.output_tokens,
                    pos: start_pos.min(prefill_target),
                    prefill_target,
                    generated: 0,
                    first_token_ns: None,
                    frac_carry: 0.0,
                    lease,
                };
                ep.record_pool_peaks();

                if seq.pos < seq.prefill_target {
                    ep.prefilling.push_back(seq);
                } else {
                    ep.running.push(seq);
                }

                if transfer_extra_ns > 0 {
                    // Account for pooled KV fetch time before the first chunk runs.
                    ep.busy = true;
                    ep.busy_ns += transfer_extra_ns;
                    push_event(
                        events,
                        seq_counter,
                        now_ns.saturating_add(transfer_extra_ns),
                        EventKind::EndpointStepDone {
                            endpoint_id: ep.id,
                        },
                    );
                    return Ok(());
                }
            }
            Err(Denied::Busy) => break,
            Err(Denied::ExceedsRow { .. } | Denied::ExceedsPool) => {
                let item = ep.waiting.pop_front().expect("front exists");
                ep.queued_prefill_tokens =
                    ep.queued_prefill_tokens.saturating_sub(item.est_uncached_len);
                ep.rejected_requests += 1;
            }
        }
    }

    // 2. Prefill-first policy: run one chunk of the head prefilling sequence, if any.
    if let Some(seq) = ep.prefilling.front() {
        let rem = seq.prefill_target - seq.pos;
        let chunk_rows = rem.min(ep.chunk_cap);
        let sum_kv = (seq.pos + chunk_rows) as u64;
        let rows_u64 = chunk_rows as u64;
        let vars = ep
            .protocol
            .vars(1, rows_u64, rows_u64 * ep.rank_count, sum_kv);
        let prog_name = ep.chunk_prog.name.clone();
        let dur_ns = ep.price_step(book, &prog_name, vars, sum_kv, us_by_tier)?;

        ep.busy = true;
        ep.busy_ns += dur_ns;
        ep.prefill_chunks += 1;
        ep.inflight = Some(InflightOp::PrefillChunk {
            seq_slot: 0,
            rows: chunk_rows,
        });
        push_event(
            events,
            seq_counter,
            now_ns.saturating_add(dur_ns),
            EventKind::EndpointStepDone {
                endpoint_id: ep.id,
            },
        );
        return Ok(());
    }

    // 3. Otherwise run one bucketed decode / speculative step across `running`.
    if !ep.running.is_empty() {
        let active_n = ep.running.len();
        let b = bucket_size(active_n).min(ep.max_seqs) as u64;
        let step_rows = ep.step_rows;
        let fwd_name = ep
            .protocol
            .forward(b, Rows::Const(step_rows))
            .or_else(|| ep.protocol.forward(active_n as u64, Rows::Const(step_rows)))
            .map(|f| f.name.clone())
            .ok_or_else(|| anyhow::anyhow!("no forward program for batch {b} × {step_rows} rows"))?;

        let pad_n = (b as usize).saturating_sub(active_n);
        let sum_kv: u64 = ep
            .running
            .iter()
            .map(|s| s.pos as u64 + step_rows)
            .sum::<u64>()
            + (pad_n as u64) * step_rows;
        let total_rows = b * step_rows;
        let vars = ep
            .protocol
            .vars(b, total_rows, total_rows * ep.rank_count, sum_kv);
        let dur_ns = ep.price_step(book, &fwd_name, vars, sum_kv, us_by_tier)?;

        ep.busy = true;
        ep.busy_ns += dur_ns;
        ep.decode_steps += 1;
        ep.decode_batch_sum += active_n as u64;
        ep.inflight = Some(InflightOp::DecodeStep {
            rows_per_seq: step_rows as usize,
        });
        push_event(
            events,
            seq_counter,
            now_ns.saturating_add(dur_ns),
            EventKind::EndpointStepDone {
                endpoint_id: ep.id,
            },
        );
    }

    Ok(())
}

fn summarize_ms(samples: &[f64]) -> LatencySummary {
    if samples.is_empty() {
        return LatencySummary::default();
    }
    let mut v = samples.to_vec();
    v.sort_by(f64::total_cmp);
    let quantile = |p: f64| -> f64 {
        let at = p * (v.len() - 1) as f64;
        let lo = at.floor() as usize;
        let hi = at.ceil() as usize;
        v[lo] + (v[hi] - v[lo]) * (at - lo as f64)
    };
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    LatencySummary {
        mean_ms: mean,
        p50_ms: quantile(0.50),
        p90_ms: quantile(0.90),
        p99_ms: quantile(0.99),
        max_ms: *v.last().expect("non-empty"),
    }
}
