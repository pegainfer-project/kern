//! Replay the AgentX (Claude Code) traces through the checkpoint table:
//! page numbers only, no GPU, no model. The roadmap's K1 gate.
//!
//!     agentx_replay --traces <cc-traces-weka-062126>/traces.jsonl   # HF semianalysisai/cc-traces-weka-062126 \
//!         [--unit 64] [--kv-bytes 65536] [--state <bytes per slot>] [--budget-gib 250 | --capacity <tokens>]
//!         [--slots 130] [--concurrency 32] [--sessions N] [--pp-head]
//!
//! Every request of a session is a sequence: leased at its timestamp for
//! `in + out` tokens, prefilled from the longest checkpoint holding a
//! prefix of its input, dropped `api_time` later. A paged-only model
//! (default) leaves a checkpoint at every page boundary of the prompt as it
//! is admitted and of the output as it finishes; `--state` models a
//! recurrent state too — a checkpoint costs a sequence slot, so only the
//! finished request is retired into one. Tokens are synthesized from the
//! trace's 64-token block hashes (a request's output tokens are what the
//! next request of the session carries at those positions, when one
//! extends it); positions no block hash covers get tokens nothing else
//! shares. Sessions start `--concurrency` at a time, in file order, each
//! when the one that many before it ends. A `Busy` lease evicts
//! checkpoints until it fits, or waits for the next sequence to finish.
//! The pool is `--slots` slots and the pages `--budget-gib` has room for
//! beside them (or `--capacity` tokens of pages), fixed as the runtime's
//! are: a lease short of a slot or pages is `Busy`.
//!
//! `--pp-head` (with `--state`) replays as a pipeline head serving the
//! prefill of a P/D split does: a request leases its prompt only, is kept
//! where its prompt ends (the next request carries its output, which the
//! decode tier made).
//!
//! Reported: hit rate (prefix tokens found / input tokens), the hits and
//! their tokens, extend percentiles, checkpoints kept, evictions, requests
//! that had to wait; the most entries and slots held at once, the entries
//! a later turn of their session moved past at the end, and the requests
//! that found nothing although an earlier one of their session had
//! finished.

mod agentx;

use std::collections::{BTreeMap, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;

use agentx::{manifest, pct, Trace};
use kern_pool::{Denied, Lease, Pool, Prefix};

struct Live {
    lease: Lease,
    req: usize,
    hit: usize,
    /// The prompt and the output both: the output's tokens are known up
    /// front here.
    tokens: Vec<i64>,
}

#[derive(Default)]
struct Tally {
    input: u64,
    hit: u64,
    extend: Vec<usize>,
    /// Requests whose prefix was found, and the tokens found.
    hits: u64,
    hit_tokens: u64,
    checkpoints: u64,
    evictions: u64,
    waited: u64,
    rejected: u64,
    max_live: usize,
    peak_entries: usize,
    peak_slots_used: usize,
    /// Requests that found nothing after an earlier one of their session finished.
    lost: u64,
}

fn main() {
    let mut traces = PathBuf::new();
    let mut unit = 64usize;
    let mut kv_bytes = 65536u64;
    let mut state = 0u64;
    let mut capacity: Option<u64> = None;
    let mut budget_gib = 250u64;
    let mut slots = 130usize;
    let mut concurrency = 32usize;
    let mut sessions_cap = usize::MAX;
    let mut pp_head = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut v = || args.next().expect("value");
        match a.as_str() {
            "--traces" => traces = PathBuf::from(v()),
            "--unit" => unit = v().parse().unwrap(),
            "--kv-bytes" => kv_bytes = v().parse().unwrap(),
            "--state" => state = v().parse().unwrap(),
            "--capacity" => capacity = Some(v().parse().unwrap()),
            "--budget-gib" => budget_gib = v().parse().unwrap(),
            "--slots" => slots = v().parse().unwrap(),
            "--concurrency" => concurrency = v().parse().unwrap(),
            "--sessions" => sessions_cap = v().parse().unwrap(),
            "--pp-head" => pp_head = true,
            _ => panic!("unknown arg {a}"),
        }
    }
    let trace = Trace::load(&traces, sessions_cap, concurrency);
    let (reqs, sessions, order) = (trace.reqs.as_slice(), trace.sessions, &trace.order);
    let row = (1usize << 20) / unit + 64;
    let m = manifest(unit, kv_bytes, state, slots, row);
    let page_bytes = kv_bytes * unit as u64;
    let slots = if state > 0 { slots } else { 0 };
    let pages = match capacity {
        Some(tokens) => tokens / unit as u64,
        None => (budget_gib << 30).saturating_sub(slots as u64 * state) / page_bytes,
    } as usize;
    assert!(pages > 0, "no page left beside {slots} slots of {state} bytes: a bigger --budget-gib or fewer --slots");
    let pool = Arc::new(Pool::new(&m, pages, slots).expect("the budget lays out"));
    let mut prefix = Prefix::new(unit);
    let mut tally = Tally::default();
    // Finish events keyed by (time bits, live index) so the order is total.
    let mut finishing: BTreeMap<(u64, usize), Live> = BTreeMap::new();
    let mut waiting: VecDeque<usize> = VecDeque::new();
    let mut live_ids = 0usize;

    assert!(!pp_head || state > 0, "--pp-head models a recurrent state: pass --state");
    let mut finished = vec![false; sessions];
    let finish = |live: Live, prefix: &mut Prefix, tally: &mut Tally, finished: &mut [bool]| {
        let r = &reqs[live.req];
        finished[r.session] = true;
        let total = if pp_head { r.input } else { r.input + r.out };
        if state > 0 {
            if total >= 1 {
                let cp = pool.retire(live.lease, total);
                prefix.insert(&live.tokens[..total], cp);
                tally.checkpoints += 1;
            }
        } else {
            let mut lease = live.lease;
            let first = r.input.div_ceil(unit).max(live.hit / unit + 1);
            for k in first..=total / unit {
                if let Ok((cp, _)) = pool.checkpoint(&mut lease, k * unit) {
                    prefix.insert(&live.tokens[..k * unit], cp);
                    tally.checkpoints += 1;
                }
            }
        }
    };

    let mut i = 0usize;
    let mut now = 0.0f64;
    while i < order.len() || !finishing.is_empty() || !waiting.is_empty() {
        let next_admit = order.get(i).map(|&r| trace.at(r));
        let next_finish = finishing.keys().next().map(|&(bits, _)| f64::from_bits(bits));
        let admit_now = match (next_admit, next_finish) {
            (Some(a), Some(f)) => a <= f && waiting.is_empty(),
            (Some(_), None) => true,
            (None, _) => false,
        };
        if admit_now || (!waiting.is_empty() && next_finish.is_none()) {
            let ri = if let Some(w) = waiting.pop_front() {
                w
            } else {
                i += 1;
                order[i - 1]
            };
            let r = &reqs[ri];
            now = now.max(trace.at(ri));
            let toks = trace.tokens(ri);
            let worst = if pp_head { r.input } else { r.input + r.out };
            let lease = loop {
                // Room made for a retry may have dropped the checkpoint
                // hit: look it up again.
                let attempt = match prefix.lookup(&toks[..r.input]) {
                    Some(h) => pool.restore(&h.found, h.len, worst).map(|(l, _)| l),
                    None => pool.lease(worst),
                };
                match attempt {
                    Ok(l) => break Some(l),
                    Err(Denied::Busy) => match prefix.evict() {
                        Some(_) => tally.evictions += 1,
                        None => break None,
                    },
                    Err(_) => {
                        tally.rejected += 1;
                        break None;
                    }
                }
            };
            let Some(mut lease) = lease else {
                if finishing.is_empty() {
                    tally.rejected += 1; // nothing to wait for
                } else {
                    tally.waited += 1;
                    waiting.push_front(ri);
                    // Fall through to the next finish.
                    let ((bits, _), live) = finishing.pop_first().unwrap();
                    now = f64::from_bits(bits);
                    finish(live, &mut prefix, &mut tally, &mut finished);
                }
                continue;
            };
            let hit_len = lease.prefix();
            tally.lost += (hit_len == 0 && finished[r.session]) as u64;
            tally.input += r.input as u64;
            tally.hit += hit_len as u64;
            tally.extend.push(r.input - hit_len);
            if hit_len > 0 {
                tally.hits += 1;
                tally.hit_tokens += hit_len as u64;
            }
            if state == 0 {
                for k in (hit_len / unit + 1)..=r.input / unit {
                    if let Ok((cp, _)) = pool.checkpoint(&mut lease, k * unit) {
                        prefix.insert(&toks[..k * unit], cp);
                        tally.checkpoints += 1;
                    }
                }
            }
            let end = now + r.api;
            let id = live_ids;
            live_ids += 1;
            finishing.insert((end.to_bits(), id), Live { lease, req: ri, hit: hit_len, tokens: toks });
            tally.max_live = tally.max_live.max(finishing.len());
        } else {
            let ((bits, _), live) = finishing.pop_first().unwrap();
            now = f64::from_bits(bits);
            finish(live, &mut prefix, &mut tally, &mut finished);
        }
        tally.peak_entries = tally.peak_entries.max(prefix.entries());
        tally.peak_slots_used = tally.peak_slots_used.max(pool.slots_used());
    }
    let main_reqs = reqs.iter().filter(|r| !r.subagent).count();
    println!(
        "{} sessions, {} requests ({main_reqs} main), unit {unit} ({} MiB/page), {}, {:.1} GiB: {} pages ({} tokens) and {} slots, concurrency {concurrency}",
        sessions,
        reqs.len(),
        page_bytes >> 20,
        if state > 0 { format!("recurrent state {} MiB/slot (checkpoint at request end)", state >> 20) } else { "paged only (checkpoint every page)".into() },
        (pages as u64 * page_bytes + slots as u64 * state) as f64 / (1u64 << 30) as f64,
        pool.pages(),
        pool.pages() as u64 * unit as u64,
        pool.slots(),
    );
    println!(
        "hit {:.1}% of {} input tokens: {} requests hit ({} tokens); extend p50 {} p90 {} p99 {}; {} checkpoints made, {} kept, {} evicted; {} requests waited, {} rejected; max live {}",
        100.0 * tally.hit as f64 / tally.input.max(1) as f64,
        tally.input,
        tally.hits,
        tally.hit_tokens,
        pct(&mut tally.extend, 0.5).unwrap_or(0),
        pct(&mut tally.extend, 0.9).unwrap_or(0),
        pct(&mut tally.extend, 0.99).unwrap_or(0),
        tally.checkpoints,
        prefix.entries(),
        tally.evictions,
        tally.waited,
        tally.rejected,
        tally.max_live,
    );
    println!(
        "at most {} entries and {} slots held at once ({} and {} at the end, {} entries superseded); {} requests found nothing after their session had finished one",
        tally.peak_entries,
        tally.peak_slots_used,
        prefix.entries(),
        pool.slots_used(),
        prefix.superseded(),
        tally.lost,
    );
}
