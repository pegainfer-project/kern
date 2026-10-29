//! The head: stage 0 of a pipeline, the scheduler every stage follows.
//!
//! A request leases its prompt's pages (and slot) from the head's pool,
//! the one pool of the pipeline, and goes out as items: the prompt but its
//! last token in chunks through the chunk program, then the last token
//! through the one-row step, whose tokens the last stage hands back; or,
//! when the chunk program hands a token back itself (a model whose chunked
//! kernels must see every prompt token), the whole prompt in chunks, the
//! last chunk's token the answer. A chunk program over several sequences
//! takes the admitted prompts' next rows packed back to back, shortest
//! remaining first, and goes out only a bounded way ahead of the stages
//! ([`pack`]). The head sends one chunk a turn and admits between turns, up
//! to as many prompts as a chunk carries, so a prompt arriving while
//! another goes out shares its items.
//!
//! A prompt starts past the longest prefix of it the head has kept
//! ([`Cache`]): the copies that costs run on the head as they are decided
//! and ride the next item to every stage. Once its last rows are out it
//! leaves all of itself in the cache, except that a recurrent state is
//! kept only at the last page boundary before the frontend says the next
//! turn parts from the prompt: its rows stop there until the checkpoint is
//! taken, and the state at its end goes back to the pool: one checkpoint a
//! turn, where every next turn of the conversation passes whatever reply
//! it renders.
//!
//! Each item is sent to every stage before the head stages and issues it
//! itself, so a stage waiting for an item always has it on the way. The
//! last stage answers every item, so the head knows how many are in the
//! pipeline and when the next goes out ([`goes_out`]). The first token
//! finishes the request. The head's own device is waited on only
//! by its [`Flow`], to push each item to stage 1.
//!
//! That wait is as long as an item, so the head is not the scheduler: it
//! runs on a thread of its own ([`spawn`]), hears submissions and departed
//! clients among its events, and writes what it decides as notes to its
//! [`Front`], which the HTTP side's driver steps. While it waits for stage
//! 1 to take an item it goes on hearing the stages, the decode tier and
//! the front's aborts, so a first token or a landed push starts its
//! request on D at once; submissions and departures wait for the next
//! turn, as admission does. The head is the only writer to a decode tier:
//! the front's aborts come through it, so every ask reaches D in the order
//! the head decided it, an abort never ahead of its request's start.
//!
//! With a decode tier (`--decode`, `pd`) a request of more than one token
//! is handed off instead of finished: admission waits for its seats there
//! ([`Tiers`]: one reserve at a time, the waiting front's), the item
//! that writes its last rows is followed at once by a push to every stage
//! and to the head's own device, before its lease goes to the cache, and
//! its first token goes to the client as it comes; the decode tier's after
//! it are the front's to relay. `cached_tokens` and the prompt's usage stay
//! the head's. With several decode tiers (`--decode` repeated) each request
//! reserves on one of those not waiting on an answer or busy, so a tier
//! that says `Busy` passes the request to another: of two, the one holding
//! fewer seats as it last said ([`pick`]: every grant and every step tells
//! the tier's [`Load`]). Every stage mirrors every tier's pools, and a push
//! names its tier.
//!
//! With a pooled tier (`--pool-master`) a leased prompt is probed before
//! it runs: the head asks the pool, once, for every page past the device
//! hit on every stage and, with a state, the mark on each of their
//! boundaries, and fetches the longest it can into the lease on every
//! stage at once, each stage's own layers under its own keys
//! ([`kern_pool::pooled`]). What landed is kept as an entry of the head's
//! cache and the prompt starts past it; a fetch that failed anywhere puts
//! the lease back as the cache leased it and the prompt starts from the
//! device hit. Every entry the cache keeps is put from every stage (a
//! retired prompt's pages alone with a state, whose end is off the page
//! grid), and a state's put is marked once every stage landed it. A
//! recurrent prompt also stops on the pool's grid and at its last page
//! boundary ([`snapshots`]) for snapshots the pool alone keeps: each is
//! put and let go, so a prompt that parts from it anywhere resumes within
//! a grid's length of where it parts, and one that passes all of it within
//! a page of its end. The pooled acts ride to the stages as jobs of their
//! own and are answered by every stage; the head waits for no answer in a
//! turn, so a prompt waits on the pool only while others go out.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::{bail, ensure, Context, Result};
use infront_engine::{FinishReason, PrefixCacheCounters, QueuedRequest, RejectReason, RequestId, StopPolicy};
use kern_manifest::protocol::{Axis, Forward, Rows, Sampling, Values};
use kern_manifest::{Protocol, Verified};
use kern_pool::{landed, objects, snapshots, Chain, Checkpoint, Denied, Keys, Landing, Lease, Object, Probe};
use kern_run::Weights;
use kern_runtime::{Capacity, Error, Runtime};
use tracing::{debug, error, info};

use super::cache::Cache;
use super::pack::{goes_out, pack, passed, Bounds, Waiting};
use super::{connect, recv, send, Flow, Hello, Item, Job, Next, Report, Table};
use crate::front::{Call, Front, Note};
use crate::pd::mirror::Mirrors;
use crate::pd::{cover, shapes, Gate, Load, Out, Pools, Push, Refusal, Req, Tell, Tiers};
use crate::pool;
use crate::ranks::{Act, Landed};
use crate::scheduler::Facts;

/// How long the head keeps trying to reach a decode tier still loading.
const CONNECT: Duration = Duration::from_secs(600);

/// How long the head sleeps with nothing heard before it looks again.
const IDLE: Duration = Duration::from_millis(1);

/// How the head loads and what it may run at once.
pub struct Head {
    pub listen: String,
    pub gpu: usize,
    pub kernels: Option<std::path::PathBuf>,
    pub capacity: Option<u64>,
    /// State slots on every stage: the running prompts' and the kept prefixes'.
    pub slots: u64,
    pub chunk: Option<u64>,
    pub max_seqs: usize,
    pub eager: bool,
    /// The decode tiers' heads to hand prompts to, each request to one.
    pub decode: Vec<String>,
    /// The pooled tier every stage joins, when one does.
    pub pool: Option<pool::Config>,
}

/// An admitted prompt with rows still to send.
struct Active {
    id: RequestId,
    lease: Lease,
    ids: Vec<i64>,
    pos: usize,
    stop: StopPolicy,
    sampling: Sampling,
    /// Where its rows stop for a checkpoint, ascending, each gone once
    /// attempted.
    stops: VecDeque<usize>,
    /// The stop kept in the cache, the turn's own; the others are the
    /// pool's alone.
    keep: Option<usize>,
    /// The frontend says where the next turn parts from it, before its
    /// end: kept there, its end is not kept.
    parts: bool,
    /// Handed off to the decode tier once prefilled.
    handoff: bool,
    /// Items gone out without it since it last had rows in one.
    passed: usize,
}

/// A leased prompt waiting on the pooled tier before it runs; its lease
/// holds the pages the fetch lands in, on every stage.
struct Pending {
    q: QueuedRequest,
    lease: Lease,
    /// The entry the lease continues from, the fallback's copies.
    hit: Option<Checkpoint>,
    ids: Vec<i64>,
    tier: Option<usize>,
    asked: Asked,
}

#[derive(Clone)]
enum Asked {
    Probing(Probe),
    /// Fetching `len` tokens, pages `first..` of them.
    Fetching {
        len: usize,
        first: usize,
        landing: Landing,
    },
}

/// A checkpoint being written to the shared cache. Its pages and slot remain
/// held until every stage completes; then `mark` publishes the stateful entry.
struct Writeback {
    cp: Checkpoint,
    len: usize,
    landing: Landing,
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

/// The pooled tier as the head sees it: its own client (stage 0's), the
/// namespace its keys are in, and every act awaiting the stages' answers.
struct Pooled {
    client: pool::Client,
    keys: Keys,
    stages: u8,
    unit: usize,
    /// The spacing of a recurrent prompt's snapshots for the pool.
    grid: usize,
    next: u64,
    pending: BTreeMap<u64, Pending>,
    puts: BTreeMap<u64, Writeback>,
    /// Marks being written, each holding its entry until it is.
    marks: BTreeMap<u64, Checkpoint>,
    counts: PoolCounts,
}

impl Pooled {
    fn id(&mut self) -> u64 {
        self.next += 1;
        self.next
    }

    /// The objects of `tokens` on every stage, from page `first`, as
    /// `pages` and `slot` hold them (the same ids on every stage).
    fn objects(&self, tokens: &[i64], first: usize, pages: &[i32], slot: Option<i32>) -> Vec<Vec<Object>> {
        let chain = Chain::of(self.unit, tokens);
        (0..self.stages).map(|s| objects(&self.keys, Some(s), &chain, tokens, first, 0, 0, pages, slot)).collect()
    }
}

/// A prompt whose last rows are out, its lease left: its token is entry
/// `at` of the reply to the item that carried them.
struct Finish {
    at: usize,
    id: RequestId,
    stop: StopPolicy,
    handoff: bool,
}

impl Finish {
    /// Complete a request on P, preserving why its first token stopped it.
    fn notes(&self, tok: u32) -> [Note; 2] {
        let why = self.stop.classify(tok, |_| false).map_or(FinishReason::Length, FinishReason::Stop);
        [Note::Emit { id: self.id, tok }, Note::Finish { id: self.id, why }]
    }
}

/// What reaches the head from the stages, the decode tier and the front.
enum Event {
    /// The last stage's tokens for the oldest item in flight.
    Answer(Vec<i64>),
    Pushed {
        stage: usize,
        req: Req,
    },
    /// A decode tier's words but its tokens, which go to the front.
    Tell {
        tier: usize,
        t: Tell,
    },
    /// A decode tier's load after a step.
    Load {
        tier: usize,
        load: Load,
    },
    Call(Call),
    /// Stage 1 took an item: a wake for a head waiting on that.
    Taken,
    /// A stage's answer to a pooled act, the head's own as stage 0.
    Landed {
        stage: usize,
        l: Landed,
    },
}

/// A decode tier's connection and mirrors: the head alone writes to it,
/// so what it asks reaches D in the order the head decided it.
struct Pd {
    d: TcpStream,
    pools: Pools,
    mirrors: Mirrors,
}

/// Rows fed since the stats line was last logged.
struct Window {
    since: Instant,
    fed: u64,
}

struct PpHead {
    rt: Runtime,
    p: Protocol,
    prefill: Forward,
    /// The one-row step the last prompt token goes through; none when the
    /// chunk program hands the first token back itself.
    step: Option<Forward>,
    bounds: Bounds,
    /// Items in flight below which a partial item goes out, and items a
    /// sequence may be passed over before it goes first: the stages.
    depth: usize,
    max_seqs: usize,
    stages: Vec<TcpStream>,
    /// Events from the front, stages, decode tiers and local completions.
    event_rx: mpsc::Receiver<Event>,
    /// A sender for the same channel, handed to the front.
    event_tx: mpsc::Sender<Event>,
    notes: mpsc::Sender<Note>,
    /// Requests the head holds whose clients left, retired at the next touch.
    gone: BTreeSet<RequestId>,
    /// The pool the front was last told of: pages in use and the cache's counters.
    pool_told: (u64, PrefixCacheCounters),
    /// Received events awaiting handling, including those deferred during issue.
    pending_events: VecDeque<Event>,
    /// Where each handed-off request stands on its decode tier.
    tiers: Tiers,
    decode: Vec<Pd>,
    waiting: VecDeque<QueuedRequest>,
    active: VecDeque<Active>,
    /// Per item in flight, oldest first, the prompts its reply finishes.
    in_flight_items: VecDeque<Vec<Finish>>,
    /// The line tables as the stages last had them written.
    lines: BTreeMap<String, Vec<i64>>,
    cache: Cache,
    pool: Option<Pooled>,
    window: Window,
    flow: Flow,
}

/// Load the head on a thread of its own, which feeds the pipeline for the
/// process's life, and hand back its front once it is ready.
pub fn spawn(m: Verified, weights: Weights, h: Head) -> Result<(Front, Facts)> {
    let (notes, front_notes) = mpsc::channel();
    let (ready, loaded) = mpsc::channel();
    std::thread::Builder::new()
        .name("kern-pp-head".into())
        .spawn(move || match load(m, &weights, h, notes) {
            Ok(head) => {
                let _ = ready.send(Ok((head.facts(), head.event_tx.clone(), head.rt.pages_total() as u64)));
                head.run()
            }
            Err(e) => {
                let _ = ready.send(Err(e));
            }
        })
        .context("spawning the head's thread")?;
    let (facts, event_tx, pages) = loaded.recv().context("the head's thread died loading")??;
    let head = move |c: Call| event_tx.send(Event::Call(c)).map_err(|_| anyhow::anyhow!("the pipeline head is gone"));
    Ok((Front::new(front_notes, head, pages), facts))
}

/// Wait for every other stage, load stage 0 at the pages they all have,
/// answer each stage with its table and join stage 1.
fn load(m: Verified, weights: &Weights, h: Head, notes: mpsc::Sender<Note>) -> Result<PpHead> {
    let cut = m.cut.clone().context("the head serves a cut manifest")?;
    ensure!(cut.stage == 0, "`--listen` serves stage 0; this manifest is stage {} of {}", cut.stage, cut.stages);
    let listener = TcpListener::bind(&h.listen).with_context(|| format!("binding {}", h.listen))?;
    info!(listen = %h.listen, stages = cut.stages, cut = %cut.id, "waiting for the stages");
    let mut hellos: BTreeMap<u64, (Hello, TcpStream, String)> = BTreeMap::new();
    while (hellos.len() as u64) < cut.stages - 1 {
        let (mut s, addr) = listener.accept()?;
        let _ = s.set_nodelay(true);
        let hello: Hello = recv(&mut s)?.with_context(|| format!("{addr} closed before its hello"))?;
        ensure!(
            hello.cut == cut.id,
            "{addr}: stage {} is of cut `{}`, the head of `{}`",
            hello.stage,
            hello.cut,
            cut.id
        );
        ensure!(
            (1..cut.stages).contains(&hello.stage) && !hellos.contains_key(&hello.stage),
            "{addr}: stage {} is not a stage still missing of 1..{}",
            hello.stage,
            cut.stages
        );
        info!(stage = hello.stage, %addr, pages = hello.pages, "stage joined");
        let reach = format!("{}:{}", addr.ip(), hello.port);
        hellos.insert(hello.stage, (hello, s, reach));
    }
    let unit = kern_pool::page_unit(&m);
    let fewest = hellos.values().filter_map(|(x, ..)| x.pages).min();
    let tokens = match (h.capacity, fewest) {
        (c, Some(f)) => Some(c.map_or(f * unit, |c| c.min(f * unit))),
        (c, None) => c,
    };
    let slots = h.slots;
    let mut rt = Runtime::load(&m, h.kernels.as_deref(), h.gpu, Some(Capacity { tokens, slots, landings: 0 }), None)?;
    rt.set_eager(h.eager);
    weights.bind(&mut rt)?;
    let p = Protocol::check_unsampled(&rt.manifest)?;
    kern_run::run_once(&rt, &p)?;
    let pages = rt.pages_total() as u64;

    let prefill =
        p.chunk().cloned().context("a pipeline prefills through a program taking one sequence's rows as fed")?;
    // The head is stage 0: whether the last stage's chunk emits is not in its manifest, so
    // a pipeline with no one-row step feeds every prompt row as chunks and the last stage answers.
    let step = p.forward(1, Rows::Const(1)).cloned();
    let max = p.rows.max as usize;
    let chunk = match h.chunk {
        None => max,
        Some(c) if (1..=max as u64).contains(&c) => c as usize,
        Some(c) => bail!("--chunk {c}: the manifest's `tokens` bound is {max}; a chunk can only be smaller"),
    };
    let context = p.context.as_ref().map_or(usize::MAX, |c| c.max as usize);
    let bounds = Bounds { rows: chunk, seqs: prefill.groups as usize, context };
    let (tx, event_rx) = mpsc::channel();
    let mut states = vec![shapes(&rt.manifest)];
    states.extend(hellos.values().map(|(x, ..)| x.states.clone()));
    let decode: Vec<Pd> = (h.decode.iter().enumerate())
        .map(|(i, addr)| decode_tier(&rt, addr, i, &states, tx.clone(), notes.clone()))
        .collect::<Result<_>>()?;
    let next_of = |s: u64| hellos.get(&(s + 1)).map(|(x, _, reach)| (reach.clone(), x.arrivals.clone()));
    let pools: Vec<Pools> = decode.iter().map(|p| p.pools.clone()).collect();
    let tables: Vec<Table> =
        (1..cut.stages).map(|s| Table { pages, next: next_of(s), decode: pools.clone() }).collect();
    let first = next_of(0).expect("stage 1 said hello");
    let mut stages = Vec::new();
    for ((s, (_, mut w, _)), table) in hellos.into_iter().zip(tables) {
        send(&mut w, &table)?;
        std::thread::spawn({
            let (r, tx) = (w.try_clone()?, tx.clone());
            move || listen(s, r, tx, s == cut.stages - 1)
        });
        stages.push(w);
    }
    let next = Next::connect(&rt, &first.0, &first.1, {
        let tx = tx.clone();
        move || {
            let _ = tx.send(Event::Taken);
        }
    })?;
    let event_tx = tx.clone();
    let pool = match &h.pool {
        Some(cfg) => {
            let mut client = pool::Client::open(cfg, rt.pooled_spans())?;
            let tx = tx.clone();
            client.forward(move |l| tx.send(Event::Landed { stage: 0, l }).is_ok())?;
            Some(Pooled {
                client,
                keys: crate::executor::keys(&m, weights)?,
                stages: cut.stages as u8,
                unit: rt.page() as usize,
                grid: {
                    let bytes = |f: fn(&kern_manifest::types::State) -> u64| m.states.values().map(f).sum();
                    kern_pool::grid(rt.page() as usize, bytes(|s| s.bytes_per_seq), bytes(|s| s.bytes_per_token))
                },
                next: 0,
                pending: BTreeMap::new(),
                puts: BTreeMap::new(),
                marks: BTreeMap::new(),
                counts: PoolCounts::default(),
            })
        }
        None => None,
    };
    let report: super::Sink = Box::new(move |r| match r {
        Report::Pushed(req) => {
            tx.send(Event::Pushed { stage: 0, req }).map_err(|_| anyhow::anyhow!("the head is gone"))
        }
        Report::Tokens(_) => bail!("the head answers no items"),
        Report::Landed(_) => bail!("the head's own pool answers come by its client"),
    });

    let step_name = step.as_ref().map_or("-", |f| f.name.as_str()).to_string();
    info!(pages, page = rt.page(), chunk, seqs = bounds.seqs, prefill = %prefill.name, step = %step_name,
          seq_slots = rt.has_seq_state().then(|| rt.seq_slots()), pooled = pool.is_some(), "pipeline head ready");
    let cache = Cache::new(rt.page() as usize);
    Ok(PpHead {
        rt,
        p,
        prefill,
        step,
        bounds,
        depth: cut.stages as usize,
        max_seqs: h.max_seqs,
        stages,
        event_rx,
        event_tx,
        notes,
        gone: BTreeSet::new(),
        pool_told: (0, PrefixCacheCounters::default()),
        pending_events: VecDeque::new(),
        tiers: Tiers::new(decode.len(), cut.stages as usize),
        decode,
        waiting: VecDeque::new(),
        active: VecDeque::new(),
        in_flight_items: VecDeque::new(),
        lines: BTreeMap::new(),
        cache,
        pool,
        window: Window { since: Instant::now(), fed: 0 },
        flow: Flow::new(0, None, Some(next), report, false),
    })
}

/// Connect to decode tier `tier` at `addr`, check that `stages` (the
/// head's states first) cover its states, mirror its pools here, and hear
/// it on a thread of its own: its tokens go to the front, the rest to the
/// head.
fn decode_tier(
    rt: &Runtime,
    addr: &str,
    tier: usize,
    stages: &[BTreeMap<String, crate::pd::Shape>],
    tx: mpsc::Sender<Event>,
    notes: mpsc::Sender<Note>,
) -> Result<Pd> {
    let mut d = connect(addr, CONNECT)?;
    let pools = match recv::<Tell>(&mut d)?.context("the decode tier closed before its pools")? {
        Tell::Pools(p) => p,
        t => bail!("the decode tier opened with {t:?}, not its pools"),
    };
    cover(stages, &pools)?;
    let mirrors = Mirrors::map(rt, pools.clone(), &stages[0])?;
    info!(
        addr,
        tier,
        ranks = pools.handles.len(),
        group = pools.group,
        dealt = pools.dealt,
        max_seq_tokens = pools.max_seq_tokens,
        "decode tier mapped"
    );
    std::thread::spawn({
        let mut r = d.try_clone()?;
        move || {
            let why = loop {
                match recv::<Tell>(&mut r) {
                    Ok(Some(t)) => {
                        let sent = match t {
                            Tell::Tokens { step, load } => {
                                notes.send(Note::Tokens(step)).is_ok() && tx.send(Event::Load { tier, load }).is_ok()
                            }
                            t => tx.send(Event::Tell { tier, t }).is_ok(),
                        };
                        if !sent {
                            return;
                        }
                    }
                    Ok(None) => break format!("decode tier {tier} closed its connection"),
                    Err(e) => break format!("decode tier {tier} connection: {e:#}"),
                }
            };
            error!("{why}");
            std::process::exit(1);
        }
    });
    Ok(Pd { d, pools, mirrors })
}

/// Read one stage's connection to its end: the last stage's answers and
/// every stage's pushes go to the head; anything closing ends the process.
fn listen(stage: u64, mut r: TcpStream, tx: mpsc::Sender<Event>, last: bool) {
    let why = loop {
        let e = match recv::<Report>(&mut r) {
            Ok(Some(Report::Tokens(t))) if last => Event::Answer(t),
            Ok(Some(Report::Tokens(_))) => break "a stage other than the last answered an item".to_string(),
            Ok(Some(Report::Pushed(req))) => Event::Pushed { stage: stage as usize, req },
            Ok(Some(Report::Landed(l))) => Event::Landed { stage: stage as usize, l },
            Ok(None) => break "stage closed its connection; the pipeline is gone".to_string(),
            Err(e) => break format!("stage connection: {e:#}"),
        };
        if tx.send(e).is_err() {
            return;
        }
    };
    error!(stage, "{why}");
    std::process::exit(1);
}

impl PpHead {
    fn facts(&self) -> Facts {
        let d = self.decode.iter().map(|p| p.pools.max_seq_tokens).min().unwrap_or(usize::MAX);
        Facts {
            total_blocks: self.rt.pages_total(),
            block_size: self.rt.page() as usize,
            max_request_tokens: self.rt.max_seq_tokens().min(d),
        }
    }

    /// Prompts leased and not finished.
    fn live(&self) -> usize {
        self.active.len() + self.pending() + self.in_flight_items.iter().map(Vec::len).sum::<usize>()
    }

    /// Prompts leased and waiting on the pooled tier.
    fn pending(&self) -> usize {
        self.pool.as_ref().map_or(0, |p| p.pending.len())
    }

    fn note(&self, n: Note) -> Result<()> {
        self.notes.send(n).map_err(|_| anyhow::anyhow!("the front is gone"))
    }

    /// Whether `id` is still the head's to answer.
    fn holds(&self, id: RequestId) -> bool {
        self.waiting.iter().any(|q| q.id == id)
            || self.active.iter().any(|a| a.id == id)
            || self.pool.as_ref().is_some_and(|p| p.pending.values().any(|x| x.q.id == id))
            || self.in_flight_items.iter().flatten().any(|f| f.id == id)
            || self.tiers.open().contains(&id.raw())
    }

    /// Lease each waiting prompt in order, from what the cache kept of it;
    /// one that cannot be seated stops the scan.
    fn admit(&mut self) -> Result<()> {
        while let Some(q) = self.waiting.front() {
            let id = q.id;
            if self.gone.remove(&id) {
                self.note(Note::Retire(id))?;
                self.waiting.pop_front();
                let outs = self.tiers.abort(id.raw());
                self.apply(outs)?;
                continue;
            }
            // More prompts feeding than an item carries would only hold pages.
            if self.live() >= self.max_seqs || self.active.len() >= self.bounds.seqs {
                break;
            }
            let (prompt, max_tokens) = (q.request.prompt_tokens.len(), q.request.max_tokens);
            let (gate, outs) = self.tiers.gate(id.raw(), prompt + max_tokens, max_tokens, &q.request.stop_policy)?;
            self.apply(outs)?;
            let Gate::Lease(tier) = gate else { break };
            let Some(q) = self.waiting.front() else { unreachable!("the request gated is still at the front") };
            let limit = self.rt.max_seq_tokens();
            let ids: Vec<i64> = q.request.prompt_tokens.iter().map(|&t| t as i64).collect();
            let leased = match prompt {
                0 => Err(Error::Denied(Denied::ExceedsRow { limit })),
                _ => self.cache.lease(&mut self.rt, &ids),
            };
            let reject = match leased {
                Ok((lease, start, hit)) => {
                    let q = self.waiting.pop_front().unwrap();
                    let probe = self
                        .pool
                        .as_ref()
                        .and_then(|p| Probe::plan(self.rt.has_seq_state(), p.unit, start, prompt, Some(p.stages)));
                    match probe {
                        Some(probe) => self.probe(Pending { q, lease, hit, ids, tier, asked: Asked::Probing(probe) }),
                        None => self.activate(q, lease, ids, tier)?,
                    }
                    continue;
                }
                Err(Error::Denied(Denied::Busy)) => break,
                Err(Error::Denied(Denied::ExceedsRow { limit })) => {
                    Some(RejectReason::ContextLength { prompt_tokens: prompt, max_tokens, limit })
                }
                Err(Error::Denied(Denied::ExceedsPool)) => {
                    Some(RejectReason::KvBudget { prompt_tokens: prompt, worst_case_tokens: prompt })
                }
                Err(e) => return Err(e.into()),
            };
            if let Some(why) = reject {
                self.note(Note::Reject { id, why })?;
                self.waiting.pop_front();
                if let Some(t) = tier {
                    let outs = self.tiers.release(t, id.raw());
                    self.apply(outs)?;
                }
            }
        }
        Ok(())
    }

    /// A leased prompt starts past its lease's prefix: its rows stop at
    /// the page boundary before the frontend's parting point when it is
    /// past the prefix, and with a pooled tier at its snapshots too.
    fn activate(&mut self, q: QueuedRequest, lease: Lease, ids: Vec<i64>, tier: Option<usize>) -> Result<()> {
        let (id, prompt, start) = (q.id, ids.len(), lease.prefix());
        // Admitted on P now, not at its lease: a prompt the pooled tier is
        // still writing into is the pool's to retire, and its tier's once it
        // starts.
        if let Some(t) = tier {
            self.tiers.admitted(t, id.raw());
        }
        self.note(Note::Admit { id, cached: start })?;
        debug!(request = %id, prompt, prefix_hit = start, "admitted");
        // The prompt's whole self is kept at its end; a parting point is
        // worth a checkpoint of its own only to a recurrent state, which
        // cannot be continued from part of a longer one, and is then the
        // only one kept.
        let (stateful, unit) = (self.rt.has_seq_state(), self.rt.page() as usize);
        let parted = q.request.history_tokens.filter(|&m| stateful && m < prompt);
        let keep = parted.map(|m| m / unit * unit).filter(|&m| m > start);
        let stops = match self.pool.as_ref().filter(|_| stateful) {
            Some(p) => {
                snapshots(unit, p.grid, start, parted.unwrap_or(prompt - usize::from(self.step.is_some()))).into()
            }
            None => keep.into_iter().collect(),
        };
        let parts = parted.is_some();
        let stop = q.request.stop_policy.clone();
        let sampling = crate::ranks::sampling(&q.request.params);
        let handoff = tier.is_some();
        let a = Active { id, lease, ids, pos: start, stop, sampling, stops, keep, parts, handoff, passed: 0 };
        self.active.push_back(a);
        Ok(())
    }

    /// A prompt done waiting on the pooled tier starts, unless its client
    /// left meanwhile: then its lease goes back, now that no stage writes it.
    fn resume(&mut self, p: Pending) -> Result<()> {
        if self.gone.remove(&p.q.id) {
            self.note(Note::Retire(p.q.id))?;
            let outs = self.tiers.abort(p.q.id.raw());
            return self.apply(outs);
        }
        self.activate(p.q, p.lease, p.ids, p.tier)
    }

    /// Ask the pooled tier what it holds of `p`'s prompt past its lease's
    /// prefix, through the head's own client: the keys name every stage.
    fn probe(&mut self, p: Pending) {
        let pool = self.pool.as_mut().expect("probing with a pool");
        let Asked::Probing(probe) = &p.asked else { unreachable!("a pending prompt starts probing") };
        let keys = probe.keys(&pool.keys, &Chain::of(pool.unit, &p.ids));
        debug!(request = %p.q.id, device_hit = p.lease.prefix(), pages = ?probe.pages, "probing");
        let id = pool.id();
        pool.client.probe(id, keys);
        pool.pending.insert(id, p);
    }

    /// Fetch `len` tokens of `p`'s prompt into its lease on every stage,
    /// pages `first..` of them.
    fn fetch(&mut self, p: Pending, len: usize, first: usize) -> Result<()> {
        let pool = self.pool.as_mut().expect("fetching with a pool");
        let per = pool.objects(&p.ids[..len], first, p.lease.page_ids(), p.lease.seq_slot());
        let (id, landing) = (pool.id(), Landing::new(0..pool.stages as usize));
        debug!(request = %p.q.id, device_hit = p.lease.prefix(), len, "fetching");
        self.transfer(id, per, false)?;
        let pool = self.pool.as_mut().expect("fetching with a pool");
        pool.pending.insert(id, Pending { asked: Asked::Fetching { len, first, landing }, ..p });
        Ok(())
    }

    /// Put the entry `cp`, which holds `tokens`, from every stage: every
    /// whole page and, with a state, the state after them, marked once
    /// every stage landed it. Without a pool, nothing.
    fn put(&mut self, cp: Checkpoint, tokens: &[i64]) -> Result<()> {
        let Some(pool) = self.pool.as_mut() else { return Ok(()) };
        let per = pool.objects(tokens, 1, &cp.page_ids(), cp.seq_slot());
        let mark = cp.seq_slot().map(|_| pool.keys.mark(Chain::of(pool.unit, tokens).at(tokens.len() / pool.unit)));
        let (id, landing) = (pool.id(), Landing::new(0..pool.stages as usize));
        self.transfer(id, per, true)?;
        let pool = self.pool.as_mut().expect("putting with a pool");
        pool.puts.insert(id, Writeback { cp, len: tokens.len(), landing, mark });
        Ok(())
    }

    /// One transfer act, `id`, with each stage's objects: the stages get
    /// theirs behind the cache's acts since the last item, the head's
    /// go to its client once its own device is through those.
    fn transfer(&mut self, id: u64, mut per: Vec<Vec<Object>>, put: bool) -> Result<()> {
        let act = |objects| match put {
            true => Act::Put { id, objects },
            false => Act::Get { id, objects },
        };
        let mine = per.remove(0);
        let shared = self.cache.take_acts();
        for (w, objects) in self.stages.iter_mut().zip(per) {
            let acts = shared.iter().cloned().chain([act(objects)]).collect();
            send(w, &Job::Acts(acts))?;
        }
        let after = self.rt.mark()?;
        let client = &self.pool.as_ref().expect("a transfer needs a pool").client;
        match put {
            true => client.put(id, after, &mine),
            false => client.get(id, after, &mine),
        }
        Ok(())
    }

    /// Put `lease` back as the cache leased it, on the head and the
    /// stages: a fetch that failed left who knows what in its pages.
    fn restore(&mut self, lease: &Lease, hit: Option<&Checkpoint>) -> Result<()> {
        let act = match hit {
            Some(h) => Act::Copy(lease.copies_from(h).into()),
            None => match lease.seq_slot() {
                Some(s) => Act::Zero(s),
                None => return Ok(()),
            },
        };
        match &act {
            Act::Copy(m) => self.rt.copy(&m.pages, m.slot.as_slice())?,
            Act::Zero(s) => self.rt.zero_slot(*s)?,
            _ => unreachable!("built above"),
        }
        self.cache.act(act);
        Ok(())
    }

    /// A stage's answer to a pooled act: a probe's choice becomes a fetch,
    /// a fetch landed on every stage starts its prompt past what landed,
    /// kept as an entry (the pool holds it already, so it is not put
    /// again), or failed anywhere restores the lease and starts it from
    /// the device hit; a put landed on every stage is marked.
    fn landed(&mut self, stage: usize, l: Landed) -> Result<()> {
        let pool = self.pool.as_mut().context("a stage answered a pooled act, and the head joined no pooled tier")?;
        if let Some(mut p) = pool.pending.remove(&l.id) {
            match p.asked.clone() {
                Asked::Probing(probe) => {
                    pool.counts.probes += 1;
                    self.cache.pooled(p.ids.len() - p.lease.prefix(), 0);
                    match probe.chosen(&l.ok) {
                        Some(len) => {
                            pool.counts.hits += 1;
                            self.fetch(p, len, probe.pages.start)
                        }
                        None => {
                            debug!(request = %p.q.id, "pool miss");
                            self.resume(p)
                        }
                    }
                }
                Asked::Fetching { len, first, landing } => {
                    let got = landed(&l.ok, self.rt.has_seq_state(), len, first, pool.unit);
                    let landing = landing.heard(stage, got).context("a fetch's landing")?;
                    let device = p.lease.prefix();
                    match landing.done() {
                        None => {
                            p.asked = Asked::Fetching { len, first, landing };
                            pool.pending.insert(l.id, p);
                            Ok(())
                        }
                        Some(got) if got > device => {
                            pool.counts.fetched += 1;
                            pool.counts.fetched_tokens += (got - device) as u64;
                            self.cache.pooled(0, got - device);
                            debug!(request = %p.q.id, device_hit = device, fetched = got, "pool hit");
                            p.lease.filled(got);
                            self.cache.keep(&mut self.rt, &mut p.lease, &p.ids[..got])?;
                            self.resume(p)
                        }
                        Some(_) => {
                            pool.counts.fetch_fails += 1;
                            debug!(request = %p.q.id, device_hit = device, asked = len, "pool fetch did not land");
                            self.restore(&p.lease, p.hit.as_ref())?;
                            self.resume(p)
                        }
                    }
                }
            }
        } else if let Some(mut put) = pool.puts.remove(&l.id) {
            let whole = l.ok.iter().all(|&b| b);
            put.landing = put.landing.heard(stage, if whole { put.len } else { 0 }).context("a put's landing")?;
            match put.landing.done() {
                None => {
                    pool.puts.insert(l.id, put);
                }
                Some(got) if got == put.len => {
                    pool.counts.puts += 1;
                    if let Some(key) = put.mark {
                        let id = pool.id();
                        pool.client.mark(id, key);
                        pool.marks.insert(id, put.cp);
                    }
                }
                Some(_) => {
                    pool.counts.put_fails += 1;
                    debug!(tokens = put.len, "pool put did not land");
                }
            }
            Ok(())
        } else if pool.marks.remove(&l.id).is_some() {
            pool.counts.put_fails += u64::from(!l.ok.iter().all(|&b| b));
            Ok(())
        } else {
            bail!("stage {stage}: an answer to pooled act {} nobody asked", l.id)
        }
    }

    /// Carry out what the tiers decided, in order.
    fn apply(&mut self, outs: Vec<(usize, Out)>) -> Result<()> {
        for (tier, o) in outs {
            match o {
                Out::Ask(a) => send(&mut self.decode[tier].d, &a)?,
                Out::Push(p) => self.push(&p)?,
                Out::Emit { req, tok } => self.note(Note::Emit { id: RequestId::new(req), tok })?,
                Out::Finish { req, why } => {
                    let id = RequestId::new(req);
                    self.gone.remove(&id);
                    self.note(Note::Finish { id, why })?;
                }
                Out::Decode { req, relay } => {
                    let id = RequestId::new(req);
                    self.gone.remove(&id);
                    self.note(Note::Decode { id, relay, tier })?;
                }
                Out::Reject { req, why } => {
                    let i = self.waiting.iter().position(|q| q.id.raw() == req);
                    let q = self.waiting.remove(i.context("a refused request no longer waiting")?).expect("found");
                    let (prompt_tokens, max_tokens) = (q.request.prompt_tokens.len(), q.request.max_tokens);
                    let why = match why {
                        Refusal::ContextLength { limit } => {
                            RejectReason::ContextLength { prompt_tokens, max_tokens, limit }
                        }
                        Refusal::KvBudget => {
                            RejectReason::KvBudget { prompt_tokens, worst_case_tokens: prompt_tokens + max_tokens }
                        }
                    };
                    self.gone.remove(&q.id);
                    self.note(Note::Reject { id: q.id, why })?;
                }
            }
        }
        Ok(())
    }

    /// Every stage, the head among them, pushes its states for `p` into
    /// its decode tier, behind the item just issued.
    fn push(&mut self, p: &Push) -> Result<()> {
        let job = Job::Push(p.clone());
        for w in &mut self.stages {
            send(w, &job)?;
        }
        let pd = self.decode.get_mut(p.tier).with_context(|| format!("a push to decode tier {}, of none", p.tier))?;
        pd.mirrors.push(&self.rt, p)?;
        self.flow.pushed(&self.rt, p.req)
    }

    /// The rows of a prompt that go through the chunk program: all of them,
    /// or all but the last when the step takes that one.
    fn chunked(&self, a: &Active) -> usize {
        a.ids.len() - usize::from(self.step.is_some())
    }

    /// Where the rows stop for now: the next checkpoint, then the chunked rows.
    fn upto(&self, a: &Active) -> usize {
        a.stops.front().copied().unwrap_or(self.chunked(a))
    }

    /// Send what is ready: the next packed chunk while the pipeline takes
    /// it, then the step of each prompt whose chunks are out. One chunk a
    /// turn, so admission runs between every two items: a turn that drained
    /// the admitted prompts would leave each one's last rows (the few after
    /// its checkpoint) to go alone, with nothing admitted to pack them with.
    /// A prompt whose last rows are out goes into the cache at once:
    /// whatever reads its pages next is an item behind the ones writing them.
    fn issue(&mut self) -> Result<()> {
        let waiting: Vec<Waiting> =
            self.active.iter().map(|a| Waiting { pos: a.pos, left: self.upto(a) - a.pos, passed: a.passed }).collect();
        let plan = pack(&waiting, self.bounds, self.depth);
        if goes_out(&plan, self.bounds, self.in_flight_items.len(), self.depth) {
            for (a, n) in self.active.iter_mut().zip(passed(&waiting, &plan)) {
                a.passed = n;
            }
            self.send_chunk(&plan)?;
        }
        while let Some(i) = self.step.as_ref().and(self.active.iter().position(|a| a.pos == self.chunked(a))) {
            let a = self.active.remove(i).expect("found");
            let f = self.step.clone().expect("checked");
            self.send_item(&f, &[(&a.lease, a.pos, &a.ids[a.pos..], a.sampling)])?;
            self.prefilled(&a)?;
            let done = self.leave(a, 0)?;
            self.in_flight_items.push_back(vec![done]);
        }
        Ok(())
    }

    /// One packed chunk item: a prompt that reaches a stop is checkpointed
    /// there and the checkpoint put, the turn's own kept in the cache too.
    /// Prompts whose last rows are sent leave the active set to await the reply.
    fn send_chunk(&mut self, plan: &[(usize, usize)]) -> Result<()> {
        let f = self.prefill.clone();
        let active = std::mem::take(&mut self.active);
        let parts: Vec<(&Lease, usize, &[i64], Sampling)> = plan
            .iter()
            .map(|&(i, n)| {
                let a = &active[i];
                (&a.lease, a.pos, &a.ids[a.pos..a.pos + n], a.sampling)
            })
            .collect();
        let sent = self.send_item(&f, &parts);
        drop(parts);
        self.active = active;
        sent?;
        let emits = self.step.is_none();
        let mut finished = Vec::new();
        let mut kept = Vec::new();
        for (at, &(i, n)) in plan.iter().enumerate() {
            let a = &mut self.active[i];
            a.pos += n;
            if a.stops.front() == Some(&a.pos) {
                a.stops.pop_front();
                let cp = match a.keep == Some(a.pos) {
                    true => self.cache.keep(&mut self.rt, &mut a.lease, &a.ids[..a.pos])?,
                    false => self.cache.snapshot(&mut self.rt, &mut a.lease, a.pos)?,
                };
                kept.extend(cp.map(|cp| (cp, a.ids[..a.pos].to_vec())));
            }
            if emits && a.pos == a.ids.len() {
                finished.push((i, at));
            }
        }
        for (cp, key) in kept {
            self.put(cp, &key)?;
        }
        // Highest index first, so each removal leaves the next one's index in place.
        finished.sort_unstable_by(|x, y| y.cmp(x));
        let mut completions = Vec::with_capacity(finished.len());
        for (i, at) in finished {
            let a = self.active.remove(i).expect("planned");
            self.prefilled(&a)?;
            completions.push(self.leave(a, at)?);
        }
        completions.sort_unstable_by_key(|f| f.at);
        self.in_flight_items.push_back(completions);
        Ok(())
    }

    /// A prompt whose last rows are out, its token entry `at` of the item's
    /// reply: its lease becomes an entry, or goes back to the pool when the
    /// next turn parts from it before its end. A state's end is off the
    /// page grid, so a recurrent entry is the device's alone: the pooled
    /// tier took its last boundary as a snapshot.
    fn leave(&mut self, a: Active, at: usize) -> Result<Finish> {
        match a.parts {
            true => drop(a.lease),
            false => {
                let cp = self.cache.retire(&mut self.rt, a.lease, &a.ids);
                if cp.seq_slot().is_none() {
                    self.put(cp, &a.ids)?;
                }
            }
        }
        Ok(Finish { at, id: a.id, stop: a.stop, handoff: a.handoff })
    }

    /// A prompt whose last rows were just issued: handed off, it is pushed
    /// now, from its lease, before the lease goes to the cache.
    fn prefilled(&mut self, a: &Active) -> Result<()> {
        if !a.handoff {
            return Ok(());
        }
        let outs = self.tiers.prefilled(a.id.raw(), &a.lease, a.ids.len(), a.sampling);
        self.apply(outs)
    }

    /// Stage one item of `parts` (a lease, its position, its rows, how its
    /// token is drawn) on every stage and on the head. The acts the cache
    /// ran on the head since the item before go first (the stages run
    /// them, the head has); a line table goes out only when the item's
    /// sequences change it.
    fn send_item(&mut self, f: &Forward, parts: &[(&Lease, usize, &[i64], Sampling)]) -> Result<()> {
        let leases: Vec<&Lease> = parts.iter().map(|&(l, ..)| l).collect();
        let positions: Vec<usize> = parts.iter().map(|&(_, pos, ..)| pos).collect();
        let lens: Vec<usize> = parts.iter().map(|(_, _, ids, _)| ids.len()).collect();
        let ids: Vec<i64> = parts.iter().flat_map(|(_, _, ids, _)| ids.iter().copied()).collect();
        let samplings: Vec<Sampling> = parts.iter().map(|&(.., s)| s).collect();
        let (vars, values) = kern_run::fills(&self.p, &leases, &positions, &lens, &ids, &samplings);
        let mut rows: BTreeMap<String, Values> = values.into_iter().map(|(f, v)| (f.name.clone(), v)).collect();
        for t in &self.p.page_tables {
            let mut v = Vec::with_capacity(t.width * leases.len());
            for l in &leases {
                l.extend_row(&t.name, &mut v)?;
            }
            rows.insert(t.name.clone(), Values::Ints(v.into_iter().map(i64::from).collect()));
        }
        for t in &self.p.line_tables {
            let cols = match t.axis {
                Axis::Ranks => self.p.ranks.as_ref().map_or(1, |b| b.max),
                _ => self.p.groups.max,
            };
            let v: Vec<i64> = kern_run::line_rows(t, &leases, cols as usize)?.into_iter().map(i64::from).collect();
            if self.lines.get(&t.name) != Some(&v) {
                self.lines.insert(t.name.clone(), v.clone());
                rows.insert(t.name.clone(), Values::Ints(v));
            }
        }
        let job = Job::Item(Item { program: f.name.clone(), vars, rows, acts: self.cache.take_acts() });
        for w in &mut self.stages {
            send(w, &job)?;
        }
        let Job::Item(item) = job else { unreachable!("built above") };
        for (name, v) in &item.rows {
            kern_run::write_named(&mut self.rt, &self.p, name, v, &item.vars)?;
        }
        self.flow.begin(&self.rt, &item.vars)?;
        self.rt.issue(&item.program, &item.vars).with_context(|| format!("`{}`", item.program))?;
        self.taken()?;
        self.flow.issued(&self.rt, &item.vars, ids.len() as u64, None)?;
        self.window.fed += ids.len() as u64;
        Ok(())
    }

    /// The head's life: handle events, admit, issue, until the process ends.
    /// A failed tick ends the process: the stages see the connection close
    /// and follow, and no half-fed pipeline lingers behind an open port.
    fn run(mut self) {
        loop {
            if let Err(e) = self.tick() {
                error!("pipeline head: {e:#}");
                std::process::exit(1);
            }
        }
    }

    fn tick(&mut self) -> Result<()> {
        if self.pending_events.is_empty() {
            match self.event_rx.recv_timeout(IDLE) {
                Ok(e) => self.pending_events.push_back(e),
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => bail!("every connection to the head is gone"),
            }
        }
        self.pending_events.extend(self.event_rx.try_iter());
        while let Some(e) = self.pending_events.pop_front() {
            self.handle_event(e)?;
        }
        self.retire_gone()?;
        self.admit()?;
        self.issue()?;
        let pool = (self.rt.pages_used() as u64, self.cache.prefix_cache());
        if pool != self.pool_told {
            self.pool_told = pool;
            self.note(Note::Pool { pages: pool.0, prefix_cache: pool.1 })?;
        }
        self.log_stats();
        Ok(())
    }

    /// One line per 5 s window in which rows went out, as the single-process
    /// scheduler logs: what the cache served and kept.
    fn log_stats(&mut self) {
        let dt = self.window.since.elapsed();
        if dt.as_secs() < 5 {
            return;
        }
        let c = self.cache.take_counts();
        if self.window.fed > 0 {
            let round = |x: f64, d: f64| (x * d).round() / d;
            let state = self.rt.has_seq_state();
            info!(
                running = self.live(),
                waiting = self.waiting.len(),
                kv_pct = round(self.rt.pages_used() as f64 * 100.0 / self.rt.pages_total().max(1) as f64, 10.0),
                prefill_tokens = self.window.fed,
                prefill_tok_s = round(self.window.fed as f64 / dt.as_secs_f64(), 1.0),
                resident_hits = c.hits,
                resident_hit_tokens = c.hit_tokens,
                checkpoints = self.cache.entries(),
                prompt_checkpoints = state.then_some(c.kept),
                prompt_skips = state.then_some(c.skipped),
                pool_snapshots = self.pool.as_ref().filter(|_| state).map(|_| c.snapshots),
                pool_snapshot_skips = self.pool.as_ref().filter(|_| state).map(|_| c.snapshot_skips),
                evictions = c.evictions,
                superseded = state.then(|| self.cache.superseded()),
                slots_used = state.then(|| self.rt.seq_slots_used()),
                slots = state.then(|| self.rt.seq_slots()),
                decode_seats = ?self.tiers.loads().iter().map(|l| l.seats).collect::<Vec<_>>(),
                decode_kv_pct = ?self.tiers.loads().iter().map(|l| round(l.pages as f64 * 100.0 / l.pages_total.max(1) as f64, 10.0)).collect::<Vec<_>>(),
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
        if let Some(p) = &mut self.pool {
            p.counts = PoolCounts::default();
        }
        self.window = Window { since: Instant::now(), fed: 0 };
    }

    /// Wait until stage 1 has taken the item before the one just issued,
    /// hearing all but the front's submissions and departures meanwhile.
    fn taken(&mut self) -> Result<()> {
        let t0 = Instant::now();
        while !self.flow.ready()? {
            match self.event_rx.recv_timeout(super::TIMEOUT.saturating_sub(t0.elapsed())) {
                // A landing admits, and admission waits for the next tick.
                Ok(e @ (Event::Call(Call::Submit(_) | Call::Gone(_)) | Event::Landed { .. })) => {
                    self.pending_events.push_back(e)
                }
                Ok(e) => self.handle_event(e)?,
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    bail!("stage 1 took nothing in {:?}; the stage is gone", super::TIMEOUT)
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => bail!("every connection to the head is gone"),
            }
        }
        Ok(())
    }

    /// One thing heard: an answer retires its item and finishes the prompts
    /// it carried last (the last stage answers in issue order), or hands
    /// their first token on; a push landed; the decode tier's words; a
    /// submission; a client gone.
    fn handle_event(&mut self, e: Event) -> Result<()> {
        let t = match e {
            Event::Answer(t) => t,
            Event::Pushed { stage, req } => {
                let outs = self.tiers.pushed(req, stage)?;
                return self.apply(outs);
            }
            Event::Tell { tier, t } => {
                let outs = self.tiers.hear(tier, t)?;
                return self.apply(outs);
            }
            Event::Load { tier, load } => {
                self.tiers.load(tier, load);
                return Ok(());
            }
            Event::Call(Call::Submit(q)) => {
                self.waiting.push_back(*q);
                return Ok(());
            }
            Event::Call(Call::Gone(id)) => {
                if self.holds(id) {
                    self.gone.insert(id);
                }
                return Ok(());
            }
            Event::Call(Call::Abort { tier, req }) => {
                let out = self.tiers.stop(tier, req)?;
                return self.apply(vec![out]);
            }
            Event::Taken => return Ok(()),
            Event::Landed { stage, l } => return self.landed(stage, l),
        };
        let Some(completions) = self.in_flight_items.pop_front() else { bail!("an answer came back for no item") };
        for r in completions {
            let Some(&tok) = t.get(r.at) else {
                bail!("the last stage handed back {} tokens, no token {}", t.len(), r.at)
            };
            let tok = tok as u32;
            if r.handoff {
                let outs = self.tiers.first(r.id.raw(), tok)?;
                self.apply(outs)?;
                continue;
            }
            if self.gone.remove(&r.id) {
                self.note(Note::Retire(r.id))?;
                continue;
            }
            for note in r.notes(tok) {
                self.note(note)?;
            }
        }
        Ok(())
    }

    /// The clients gone from handed-off requests the head still holds.
    fn retire_gone(&mut self) -> Result<()> {
        let gone: Vec<Req> =
            self.tiers.open().into_iter().filter(|&r| self.gone.contains(&RequestId::new(r))).collect();
        for req in gone {
            let id = RequestId::new(req);
            self.gone.remove(&id);
            self.note(Note::Retire(id))?;
            let outs = self.tiers.abort(req);
            self.apply(outs)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use infront_engine::SamplingParams;
    use infront_engine::{spawn_scheduler, EosPolicy, Request, StopCause, Terminal};

    /// Exercise the actual head completion notes through Front and the ledger,
    /// without loading a model. The bridge appends a compatibility EOS for a
    /// Stop terminal without a typed cause; losing that cause is observable on
    /// this stream even though its token/count fields still say one token.
    fn first_token(stop: StopPolicy, tok: u32) -> (Vec<u32>, Terminal) {
        let (notes, incoming) = mpsc::channel();
        let head = move |call| {
            let Call::Submit(q) = call else { return Ok(()) };
            notes.send(Note::Admit { id: q.id, cached: 0 })?;
            let done = Finish { at: 0, id: q.id, stop: q.request.stop_policy, handoff: false };
            for note in done.notes(tok) {
                notes.send(note)?;
            }
            Ok(())
        };
        let mut live = spawn_scheduler("first-token-stop", Front::new(incoming, head, 1));
        let mut steps = live.handle.take_steps().unwrap();
        let control = live.handle.submit(Request {
            prompt_tokens: vec![3; 16],
            params: SamplingParams::default(),
            history_tokens: None,
            history_marks: Vec::new(),
            stop_policy: stop,
            max_tokens: 1,
            logprobs: None,
            prompt_logprobs: None,
        });
        let mut tokens = Vec::new();
        let mut terminal = None;
        let deadline = Instant::now() + Duration::from_secs(5);
        while terminal.is_none() && Instant::now() < deadline {
            if let Ok(step) = steps.try_recv() {
                for update in step.updates {
                    assert_eq!(update.id, control.id());
                    tokens.extend(update.tokens);
                    if let Some(end) = update.terminal {
                        assert!(terminal.replace(end).is_none(), "duplicate terminal");
                    }
                }
            } else {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        let terminal = terminal.expect("first-token terminal before deadline");
        drop(live.handle);
        live.join.join().unwrap();
        for step in std::iter::from_fn(|| steps.try_recv().ok()) {
            assert!(step.updates.is_empty(), "output after terminal");
        }
        (tokens, terminal)
    }

    #[test]
    fn first_token_stop_cause_survives_the_front_stream() {
        // EOS wins over an explicit stop for the same token; an explicit stop
        // remains active when EOS is ignored.
        let cases = [
            (EosPolicy::Token(2), vec![2, 43], 2, StopCause::Eos(2)),
            (EosPolicy::Token(2), vec![43], 43, StopCause::Token(43)),
            (EosPolicy::Ignore, vec![43], 43, StopCause::Token(43)),
        ];
        for (eos, stops, tok, cause) in cases {
            let (tokens, terminal) = first_token(StopPolicy::new(eos, stops), tok);
            assert_eq!(tokens, vec![tok]);
            assert!(
                matches!(terminal, Terminal::Finished {
                    reason, prompt_tokens: 16, completion_tokens: 1,
                } if reason == FinishReason::Stop(cause)),
                "first token {tok}: {terminal:?}, expected a stop on {cause:?}"
            );
        }
    }

    #[test]
    fn first_token_non_stop_still_finishes_by_length() {
        for (eos, tok) in [(EosPolicy::Token(2), 7), (EosPolicy::Ignore, 2)] {
            let (tokens, terminal) = first_token(StopPolicy::new(eos, vec![43]), tok);
            assert_eq!(tokens, vec![tok]);
            assert!(matches!(
                terminal,
                Terminal::Finished { reason: FinishReason::Length, prompt_tokens: 16, completion_tokens: 1 }
            ));
        }
    }
}
