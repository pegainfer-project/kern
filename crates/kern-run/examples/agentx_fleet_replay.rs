//! Replay the AgentX traces over a fleet of instances that share one pooled
//! KV store and route requests without looking at any cache: page numbers
//! only, no GPU, no model.
//!
//!     agentx_fleet_replay --traces <traces.jsonl> --instances N [--route random|session]
//!         [--pool-gib G] [--block <pages>] [--put-ms L] [--unit 64] [--kv-bytes 65536]
//!         [--state <bytes per slot>] [--budget-gib 250] [--slots 130] [--concurrency 32] [--sessions N]
//!
//! Every instance keeps its own device pool and prefix index, as
//! `agentx_replay` does for one, and nothing else: its DRAM is the pool's. The pool is content-addressed and write-through: a
//! block (`--block` pages) is keyed by the hash chain of its tokens and
//! put the moment a prefill seals it (at admission for the prompt, at the
//! end for the output), landing `--put-ms` later; a recurrent state is put
//! at the request's end as one object at its exact length (the slot and
//! the tail past the last whole block). A prompt with a state looks for
//! the session's marks (where its earlier turns ended) newest first,
//! skipping the ones it has diverged from; the device index is asked
//! first, the pool for a longer mark it has landed, fetched into a fresh
//! lease. A paged-only prompt takes the longer of its local hit and the
//! pool's landed prefix of whole blocks. The pool is one LRU of
//! `--pool-gib`.
//!
//! Reported: hit rate split into local and pool, and every input token
//! not hit by its cause (the session's first turn, its last turn still
//! running, the prompt diverged from every mark, the mark evicted from
//! both tiers, its put in flight, a block under it evicted, or the tail
//! past a hit), put and get bytes with the peak and p99 of the busy
//! seconds, and the per-instance device tallies.

mod agentx;

use std::collections::{BTreeMap, VecDeque};
use std::ops::RangeInclusive;
use std::path::PathBuf;
use std::sync::Arc;

use agentx::{manifest, pct, Trace};
use kern_pool::{Denied, Lease, Pool, Prefix};

const SEED: u64 = 0xcbf29ce484222325;

/// FNV-1a over `words` from `seed`, a word at a time.
fn fnv(seed: u64, words: impl IntoIterator<Item = u64>) -> u64 {
    words.into_iter().fold(seed, |h, w| (h ^ w).wrapping_mul(0x100000001b3))
}

/// A sequence's tokens and their hash chain by page: `chain[p]` covers
/// pages `0..=p`, so equal prefixes give equal values whatever session
/// wrote them.
struct Seq {
    toks: Vec<i64>,
    chain: Vec<u64>,
}

impl Seq {
    fn new(toks: Vec<i64>, unit: usize) -> Seq {
        let chain = toks
            .chunks_exact(unit)
            .scan(SEED, |h, page| {
                *h = fnv(*h, page.iter().map(|&t| t as u64));
                Some(*h)
            })
            .collect();
        Seq { toks, chain }
    }

    /// The key of whole block `j` of `pages` pages.
    fn block_key(&self, j: usize, pages: usize) -> u64 {
        self.chain[(j + 1) * pages - 1]
    }

    /// The key of a recurrent state after exactly `len` tokens: the chain
    /// through its whole pages, then its tail, marked apart from a block.
    fn state_key(&self, len: usize, unit: usize) -> u64 {
        let k = len / unit;
        let h = if k > 0 { self.chain[k - 1] } else { SEED };
        fnv(h ^ 0x5bd1e995, self.toks[k * unit..len].iter().map(|&t| t as u64).chain([u64::MAX]))
    }
}

struct Obj {
    bytes: u64,
    landed: f64,
    tick: u64,
}

/// The pooled store: one LRU of content-keyed objects, each landing a
/// put latency after it was put.
struct Store {
    cap: u64,
    used: u64,
    put_ms: f64,
    clock: u64,
    // Never iterated: the LRU below is the only order anything reads.
    #[allow(clippy::disallowed_types)]
    objs: std::collections::HashMap<u64, Obj>,
    lru: BTreeMap<(u64, u64), ()>,
    put_bytes: u64,
    get_bytes: u64,
    evicted_bytes: u64,
    /// Bytes put and got per second of replay time.
    traffic: BTreeMap<u64, (u64, u64)>,
}

impl Store {
    fn new(cap: u64, put_ms: f64) -> Store {
        Store {
            cap,
            used: 0,
            put_ms,
            clock: 0,
            objs: Default::default(),
            lru: BTreeMap::new(),
            put_bytes: 0,
            get_bytes: 0,
            evicted_bytes: 0,
            traffic: BTreeMap::new(),
        }
    }

    fn touch(&mut self, key: u64) {
        self.clock += 1;
        let o = self.objs.get_mut(&key).expect("a kept object");
        self.lru.remove(&(o.tick, key));
        o.tick = self.clock;
        self.lru.insert((o.tick, key), ());
    }

    /// Put `bytes` under `key` at `now` unless it is there; the coldest
    /// objects go to make room.
    fn put(&mut self, key: u64, bytes: u64, now: f64) {
        if self.cap == 0 {
            return;
        }
        if self.objs.contains_key(&key) {
            self.touch(key);
            return;
        }
        while self.used + bytes > self.cap {
            let Some((&(tick, k), _)) = self.lru.iter().next() else { break };
            self.lru.remove(&(tick, k));
            let o = self.objs.remove(&k).expect("the lru names kept objects");
            self.used -= o.bytes;
            self.evicted_bytes += o.bytes;
        }
        self.clock += 1;
        self.objs.insert(key, Obj { bytes, landed: now + self.put_ms / 1e3, tick: self.clock });
        self.lru.insert((self.clock, key), ());
        self.used += bytes;
        self.put_bytes += bytes;
        self.traffic.entry(now as u64).or_default().0 += bytes;
    }

    /// Whether `key` is there, and landed by `now`.
    fn probe(&self, key: u64, now: f64) -> Option<bool> {
        self.objs.get(&key).map(|o| o.landed <= now)
    }

    /// Fetch `key` at `now`, touching it.
    fn get(&mut self, key: u64, now: f64) {
        self.touch(key);
        let bytes = self.objs[&key].bytes;
        self.get_bytes += bytes;
        self.traffic.entry(now as u64).or_default().1 += bytes;
    }
}

/// What became of the tokens a prompt did not hit.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Cause {
    /// Hit on the device; the tokens past the hit are the prompt's own.
    Local,
    /// Fetched from the pool; likewise.
    Pool,
    /// The session had finished nothing a prompt could continue.
    First,
    /// The session's last turn was still running.
    Running,
    /// The prompt diverged from every mark the session left.
    Diverged,
    /// The mark was gone from the device and the pool both.
    Evicted,
    /// The mark's put had not landed.
    Inflight,
    /// The mark was in the pool but a block under it was not.
    Blocks,
}

const CAUSES: [Cause; 8] = [
    Cause::Local,
    Cause::Pool,
    Cause::First,
    Cause::Running,
    Cause::Diverged,
    Cause::Evicted,
    Cause::Inflight,
    Cause::Blocks,
];

impl Cause {
    fn name(self) -> &'static str {
        match self {
            Cause::Local => "local",
            Cause::Pool => "pool",
            Cause::First => "first",
            Cause::Running => "running",
            Cause::Diverged => "diverged",
            Cause::Evicted => "evicted",
            Cause::Inflight => "inflight",
            Cause::Blocks => "blocks",
        }
    }
}

struct Live {
    instance: usize,
    lease: Lease,
    req: usize,
    hit: usize,
    seq: Seq,
}

#[derive(Default)]
struct Tally {
    input: u64,
    local_hit_tokens: u64,
    pool_hit_tokens: u64,
    /// Requests and input tokens not hit, by cause.
    causes: [(u64, u64); 8],
    extend: Vec<usize>,
    checkpoints: u64,
    waited: u64,
    rejected: u64,
}

struct Instance {
    pool: Arc<Pool>,
    prefix: Prefix,
    evictions: u64,
    requests: u64,
}

impl Instance {
    fn make_room(&mut self) -> bool {
        let dropped = self.prefix.evict().is_some();
        self.evictions += dropped as u64;
        dropped
    }

    /// Leaves a checkpoint of `lease` at the end of each of `pages` of
    /// `unit` tokens, indexed under `toks`; how many it could.
    fn index(&mut self, lease: &mut Lease, toks: &[i64], unit: usize, pages: RangeInclusive<usize>) -> u64 {
        let mut made = 0;
        for k in pages {
            if let Ok((cp, _)) = self.pool.checkpoint(lease, k * unit) {
                self.prefix.insert(&toks[..k * unit], cp);
                made += 1;
            }
        }
        made
    }
}

/// The byte shape every object follows.
#[derive(Clone, Copy)]
struct Shape {
    unit: usize,
    kv_bytes: u64,
    state: u64,
    /// Pages per pooled block.
    pages: usize,
}

impl Shape {
    fn block(&self) -> usize {
        self.pages * self.unit
    }
}

enum Admit {
    Live(Live),
    /// Nothing to evict and nothing to wait for on the instance.
    Busy,
    Rejected,
}

/// What the pool has of a mark.
enum Fetch {
    Landed(Vec<u64>),
    Inflight,
    Blocks,
    Absent,
}

struct Sim<'a> {
    trace: &'a Trace,
    shape: Shape,
    fleet: Vec<Instance>,
    store: Store,
    /// Where each session's turns left a state: (length, key), newest last.
    marks: Vec<Vec<(usize, u64)>>,
    /// Requests of each session in flight.
    running: Vec<u32>,
    tally: Tally,
}

impl Sim<'_> {
    /// The blocks of `seq.toks[from..to]` sealed at `now` go to the pool.
    fn put_blocks(&mut self, seq: &Seq, from: usize, to: usize, now: f64) {
        let (block, pages) = (self.shape.block(), self.shape.pages);
        for j in from / block..to / block {
            self.store.put(seq.block_key(j, pages), block as u64 * self.shape.kv_bytes, now);
        }
    }

    /// The newest mark of `session` within `usable` that the prompt still
    /// follows, or why there is none.
    fn best_mark(&self, session: usize, seq: &Seq, usable: usize) -> Result<(usize, u64), Cause> {
        let unit = self.shape.unit;
        let within = self.marks[session].iter().rev().filter(|&&(len, _)| len <= usable);
        let mut any = false;
        for &(len, key) in within {
            any = true;
            if seq.state_key(len, unit) == key {
                return Ok((len, key));
            }
        }
        Err(match (any, self.running[session] > 0) {
            (true, _) => Cause::Diverged,
            (false, true) => Cause::Running,
            (false, false) => Cause::First,
        })
    }

    /// The pool's holding of the state at `len`, keyed `sk`, with the
    /// blocks under it.
    fn fetch_state(&self, seq: &Seq, len: usize, sk: u64, now: f64) -> Fetch {
        let (store, Shape { pages, .. }) = (&self.store, self.shape);
        match store.probe(sk, now) {
            None => Fetch::Absent,
            Some(false) => Fetch::Inflight,
            Some(true) => {
                let keys: Vec<u64> = (0..len / self.shape.block()).map(|j| seq.block_key(j, pages)).collect();
                match keys.iter().all(|&k| store.probe(k, now) == Some(true)) {
                    true => Fetch::Landed(keys.into_iter().chain([sk]).collect()),
                    false => Fetch::Blocks,
                }
            }
        }
    }

    /// The pool's landed prefix of whole blocks under `usable`: its
    /// length, the length its keys reach landed or not, and the objects.
    fn fetch_blocks(&self, seq: &Seq, usable: usize, now: f64) -> (usize, usize, Vec<u64>) {
        let (store, Shape { pages, .. }) = (&self.store, self.shape);
        let block = self.shape.block();
        let keys: Vec<u64> = (0..usable / block).map(|j| seq.block_key(j, pages)).collect();
        let run = |f: &dyn Fn(u64) -> bool| keys.iter().take_while(|&&k| f(k)).count();
        let n = run(&|k| store.probe(k, now) == Some(true));
        let any = run(&|k| store.probe(k, now).is_some());
        (n * block, any * block, keys[..n].to_vec())
    }

    /// Request `ri` arrives at instance `inst_id` at `now`.
    fn admit(&mut self, ri: usize, inst_id: usize, now: f64) -> Admit {
        let trace = self.trace;
        let r = &trace.reqs[ri];
        let Shape { unit, state, .. } = self.shape;
        let seq = Seq::new(trace.tokens(ri), unit);
        let worst = r.input + r.out;
        let usable = r.input.saturating_sub(1);
        let stateful = state > 0;
        // What the pool offers: a length, its objects, and what to call
        // the shortfall when the device has nothing longer.
        let best = if stateful { self.best_mark(r.session, &seq, usable) } else { Err(Cause::First) };
        let (pool_len, pool_any, pool_objs, cause) = match (self.store.cap > 0, stateful, best) {
            (false, _, Ok(_)) => (0, 0, Vec::new(), Cause::Evicted),
            (false, _, Err(c)) => (0, 0, Vec::new(), c),
            (true, true, Ok((len, sk))) => match self.fetch_state(&seq, len, sk, now) {
                Fetch::Landed(objs) => (len, len, objs, Cause::Pool),
                Fetch::Inflight => (0, len, Vec::new(), Cause::Inflight),
                Fetch::Blocks => (0, 0, Vec::new(), Cause::Blocks),
                Fetch::Absent => (0, 0, Vec::new(), Cause::Evicted),
            },
            (true, true, Err(c)) => (0, 0, Vec::new(), c),
            (true, false, _) => {
                let (n, any, objs) = self.fetch_blocks(&seq, usable, now);
                (n, any, objs, if n > 0 { Cause::Pool } else { Cause::First })
            }
        };

        let inst = &mut self.fleet[inst_id];
        let mut from_pool = false;
        let mut lease = loop {
            let attempt = match inst.prefix.lookup(&seq.toks[..r.input]) {
                Some(h) if h.len >= pool_len => inst.pool.restore(&h.found, h.len, worst).map(|(l, _)| l),
                _ => inst.pool.lease(worst).inspect(|_| from_pool = pool_len > 0),
            };
            match attempt {
                Ok(l) => break l,
                Err(Denied::Busy) => {
                    if !inst.make_room() {
                        return Admit::Busy;
                    }
                }
                Err(_) => return Admit::Rejected,
            }
        };
        inst.requests += 1;
        let hit_len = if from_pool {
            for &k in &pool_objs {
                self.store.get(k, now);
            }
            pool_len
        } else {
            lease.prefix()
        };
        let cause = match (from_pool, hit_len > 0, stateful, best) {
            (true, ..) => Cause::Pool,
            (false, true, true, Ok((len, _))) if hit_len == len => Cause::Local,
            (false, true, false, _) => Cause::Local,
            _ => cause,
        };
        let tally = &mut self.tally;
        tally.input += r.input as u64;
        tally.extend.push(r.input - hit_len);
        let slot = &mut tally.causes[cause as usize];
        slot.0 += 1;
        slot.1 += (r.input - hit_len) as u64;
        match cause {
            Cause::Pool => tally.pool_hit_tokens += hit_len as u64,
            _ => tally.local_hit_tokens += hit_len as u64,
        }
        if !stateful {
            tally.causes[Cause::Inflight as usize].1 += pool_any.saturating_sub(hit_len.max(pool_len)) as u64;
            // Pages fetched from the pool are whole pages of a fresh lease:
            // they are indexed here too, so the instance hits them itself
            // next time.
            let first = if from_pool { 1 } else { hit_len / unit + 1 };
            tally.checkpoints += inst.index(&mut lease, &seq.toks, unit, first..=r.input / unit);
            self.put_blocks(&seq, hit_len, r.input, now);
        }
        self.running[r.session] += 1;
        Admit::Live(Live { instance: inst_id, lease, req: ri, hit: hit_len, seq })
    }

    /// `live` ends at `now`: its output is indexed and pooled.
    fn finish(&mut self, live: Live, now: f64) {
        let r = &self.trace.reqs[live.req];
        let Shape { unit, kv_bytes, state, .. } = self.shape;
        let block = self.shape.block();
        self.running[r.session] -= 1;
        let total = r.input + r.out;
        let inst = &mut self.fleet[live.instance];
        if state > 0 {
            if total >= 1 {
                let cp = inst.pool.retire(live.lease, total);
                inst.prefix.insert(&live.seq.toks[..total], cp);
                self.tally.checkpoints += 1;
                self.put_blocks(&live.seq, live.hit, total, now);
                let sk = live.seq.state_key(total, unit);
                self.store.put(sk, state + (total % block) as u64 * kv_bytes, now);
                self.marks[r.session].push((total, sk));
            }
        } else {
            let mut lease = live.lease;
            let first = r.input.div_ceil(unit).max(live.hit / unit + 1);
            self.tally.checkpoints += inst.index(&mut lease, &live.seq.toks, unit, first..=total / unit);
            self.put_blocks(&live.seq, r.input.max(live.hit), total, now);
        }
    }
}

fn main() {
    let mut traces = PathBuf::new();
    let mut unit = 64usize;
    let mut kv_bytes = 65536u64;
    let mut state = 0u64;
    let mut budget_gib = 250u64;
    let mut slots = 130usize;
    let mut concurrency = 32usize;
    let mut sessions_cap = usize::MAX;
    let mut instances = 1usize;
    let mut route = "random".to_string();
    let mut pool_gib = 0u64;
    let mut block_pages = 1usize;
    let mut put_ms = 0.0f64;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut v = || args.next().expect("value");
        match a.as_str() {
            "--traces" => traces = PathBuf::from(v()),
            "--unit" => unit = v().parse().unwrap(),
            "--kv-bytes" => kv_bytes = v().parse().unwrap(),
            "--state" => state = v().parse().unwrap(),
            "--budget-gib" => budget_gib = v().parse().unwrap(),
            "--slots" => slots = v().parse().unwrap(),
            "--concurrency" => concurrency = v().parse().unwrap(),
            "--sessions" => sessions_cap = v().parse().unwrap(),
            "--instances" => instances = v().parse().unwrap(),
            "--route" => route = v(),
            "--pool-gib" => pool_gib = v().parse().unwrap(),
            "--block" => block_pages = v().parse().unwrap(),
            "--put-ms" => put_ms = v().parse().unwrap(),
            _ => panic!("unknown arg {a}"),
        }
    }
    assert!(instances >= 1 && block_pages >= 1);
    let shape = Shape { unit, kv_bytes, state, pages: block_pages };

    let trace = Trace::load(&traces, sessions_cap, concurrency);
    let (reqs, sessions, order) = (&trace.reqs, trace.sessions, &trace.order);
    let row = (1usize << 20) / unit + 64;
    let m = manifest(unit, kv_bytes, state, slots, row);
    let page_bytes = kv_bytes * unit as u64;
    let slots = if state > 0 { slots } else { 0 };
    let pages = ((budget_gib << 30).saturating_sub(slots as u64 * state) / page_bytes) as usize;
    assert!(pages > 0, "no page left beside {slots} slots of {state} bytes: a bigger --budget-gib or fewer --slots");
    let fleet = (0..instances)
        .map(|_| Instance {
            pool: Arc::new(Pool::new(&m, pages, slots).expect("the budget lays out")),
            prefix: Prefix::new(unit),
            evictions: 0,
            requests: 0,
        })
        .collect();
    let mut sim = Sim {
        trace: &trace,
        shape,
        fleet,
        store: Store::new(pool_gib << 30, put_ms),
        marks: vec![Vec::new(); sessions],
        running: vec![0; sessions],
        tally: Tally::default(),
    };
    let pick = |ri: usize| match route.as_str() {
        "session" => reqs[ri].session % instances,
        "random" => ((ri as u64).wrapping_mul(0x9E3779B97F4A7C15) >> 33) as usize % instances,
        r => panic!("unknown route {r}"),
    };
    // Finish events keyed by (time bits, live index) so the order is total.
    let mut finishing: BTreeMap<(u64, usize), Live> = BTreeMap::new();
    let mut waiting: VecDeque<usize> = VecDeque::new();
    let mut live_ids = 0usize;

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
        if !(admit_now || (!waiting.is_empty() && next_finish.is_none())) {
            let ((bits, _), live) = finishing.pop_first().unwrap();
            now = f64::from_bits(bits);
            sim.finish(live, now);
            continue;
        }
        let ri = waiting.pop_front().unwrap_or_else(|| {
            i += 1;
            order[i - 1]
        });
        now = now.max(trace.at(ri));
        match sim.admit(ri, pick(ri), now) {
            Admit::Live(live) => {
                let end = now + reqs[ri].api;
                finishing.insert((end.to_bits(), live_ids), live);
                live_ids += 1;
            }
            Admit::Busy if !finishing.is_empty() => {
                sim.tally.waited += 1;
                waiting.push_front(ri);
                let ((bits, _), live) = finishing.pop_first().unwrap();
                now = f64::from_bits(bits);
                sim.finish(live, now);
            }
            Admit::Busy | Admit::Rejected => sim.tally.rejected += 1,
        }
    }

    let Sim { fleet, store, mut tally, .. } = sim;
    let mut puts: Vec<u64> = store.traffic.values().map(|x| x.0).collect();
    let mut gets: Vec<u64> = store.traffic.values().map(|x| x.1).collect();
    let span = store.traffic.keys().last().map_or(0, |&s| s + 1);
    let gb = |b: u64| b as f64 / 1e9;
    let share = |n: u64| 100.0 * n as f64 / tally.input.max(1) as f64;
    println!(
        "{sessions} sessions, {} requests over {instances} instances ({route} routing), concurrency {concurrency}, unit {unit} ({} MiB/page), {}, {budget_gib} GiB device per instance; pool {pool_gib} GiB in blocks of {} tokens ({} MiB), put lands after {put_ms} ms",
        reqs.len(),
        page_bytes >> 20,
        if state > 0 { format!("recurrent state {} MiB/slot", state >> 20) } else { "paged only".into() },
        shape.block(),
        (shape.block() as u64 * kv_bytes) >> 20,
    );
    println!(
        "hit {:.1}% of {} input tokens: local {:.1}%, pool {:.1}%; extend p50 {} p90 {} p99 {}; {} checkpoints made; {} waited, {} rejected",
        share(tally.local_hit_tokens + tally.pool_hit_tokens),
        tally.input,
        share(tally.local_hit_tokens),
        share(tally.pool_hit_tokens),
        pct(&mut tally.extend, 0.5).unwrap_or(0),
        pct(&mut tally.extend, 0.9).unwrap_or(0),
        pct(&mut tally.extend, 0.99).unwrap_or(0),
        tally.checkpoints,
        tally.waited,
        tally.rejected,
    );
    let causes: Vec<String> = CAUSES
        .iter()
        .map(|&c| {
            let (n, t) = tally.causes[c as usize];
            format!("{} {:.2}% ({n} req)", c.name(), share(t))
        })
        .collect();
    println!("not hit, by cause (share of input tokens): {}", causes.join(", "));
    println!(
        "pool: {} objects ({:.1} GB) at the end, put {:.1} TB, got {:.1} TB, evicted {:.1} TB; over {span} s of replay: put mean {:.2} GB/s, busy-second p99 {:.1} max {:.1}; get mean {:.2} GB/s, busy-second p99 {:.1} max {:.1}",
        store.objs.len(),
        gb(store.used),
        gb(store.put_bytes) / 1e3,
        gb(store.get_bytes) / 1e3,
        gb(store.evicted_bytes) / 1e3,
        gb(store.put_bytes) / span.max(1) as f64,
        gb(pct(&mut puts, 0.99).unwrap_or(0)),
        gb(puts.last().copied().unwrap_or(0)),
        gb(store.get_bytes) / span.max(1) as f64,
        gb(pct(&mut gets, 0.99).unwrap_or(0)),
        gb(gets.last().copied().unwrap_or(0)),
    );
    for (k, inst) in fleet.iter().enumerate() {
        println!(
            "instance {k}: {} requests, {} entries kept, {} evicted, {} of {} pages used, {} slots",
            inst.requests,
            inst.prefix.entries(),
            inst.evictions,
            inst.pool.used(),
            inst.pool.pages(),
            inst.pool.slots_used(),
        );
    }
}
