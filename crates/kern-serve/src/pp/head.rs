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
//! ([`Cache`]), resident or woken from the host tier: the copies that
//! costs, and the parks that made room for it, run on the head as they are
//! decided and ride the next item to every stage. The head sizes the host
//! tier in units every stage holds: `--host-gib` per stage, cut into host
//! pages and host slots by the largest page and slot of any stage. Once its last rows are out it
//! leaves all of itself in the cache, except that a recurrent state is
//! kept only where the frontend says the next turn parts from the prompt:
//! its rows stop there until the checkpoint is taken, and the state at its
//! end goes back to the pool: one checkpoint a turn, where every next turn
//! of the conversation passes whatever reply it renders.
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

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::{bail, ensure, Context, Result};
use kern_manifest::protocol::{Axis, Forward, Rows};
use kern_manifest::{Protocol, Verified};
use kern_pool::{Denied, Lease};
use kern_run::Weights;
use kern_runtime::{Capacity, Error, Runtime};
use pegainfer_frontend::engine::{FinishReason, QueuedRequest, RejectReason, RequestId, StopPolicy};
use tracing::{debug, error, info};

use super::cache::Cache;
use super::pack::{goes_out, pack, passed, Bounds, Waiting};
use super::{connect, recv, send, Flow, Hello, Item, Job, Next, Report, Table};
use crate::front::{Call, Front, Note};
use crate::pd::mirror::Mirrors;
use crate::pd::{cover, shapes, Gate, Load, Out, Pools, Push, Refusal, Req, Tell, Tiers};
use crate::scheduler::Facts;

/// How long the head keeps trying to reach a decode tier still loading.
const CONNECT: Duration = Duration::from_secs(600);
/// How far short of where a prompt is kept its snapshot goes. A retried or
/// edited request parts from the request before it near that one's end:
/// in AgentX's traces half of them within 448 tokens, three in four within
/// 4.7k, and a snapshot 4096 short serves 3.6% of all input tokens of the
/// 4.1% they share.
const SNAPSHOT_SHORT: usize = 4096;

/// How long the head sleeps with nothing heard before it looks again.
const IDLE: Duration = Duration::from_millis(1);

/// How the head loads and what it may run at once.
pub struct Head {
    pub listen: String,
    pub gpu: usize,
    pub kernels: Option<std::path::PathBuf>,
    pub capacity: Option<u64>,
    pub chunk: Option<u64>,
    pub max_seqs: usize,
    pub eager: bool,
    /// The decode tiers' heads to hand prompts to, each request to one.
    pub decode: Vec<String>,
    /// Pinned host memory per stage for entries parked off the devices.
    pub host_bytes: u64,
    /// Tokens of pages the host tier holds per host slot.
    pub host_turn: u64,
}

/// An admitted prompt with rows still to send.
struct Active {
    id: RequestId,
    lease: Lease,
    ids: Vec<i64>,
    pos: usize,
    stop: StopPolicy,
    /// Where its rows stop until the prompt so far is kept, gone once it is.
    mark: Option<usize>,
    /// Where they stop before that for a snapshot, gone once it is taken.
    snap: Option<usize>,
    /// The frontend says where the next turn parts from it, before its
    /// end: kept there, its end is not kept.
    parts: bool,
    /// Handed off to the decode tier once prefilled.
    handoff: bool,
    /// Items gone out without it since it last had rows in one.
    passed: usize,
}

/// A prompt whose last rows are out, its lease left: its token is entry
/// `at` of the reply to the item that carried them.
struct Finish {
    at: usize,
    id: RequestId,
    stop: StopPolicy,
    handoff: bool,
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
    events: mpsc::Receiver<Event>,
    /// Into the head's own events, for the front's calls.
    calls: mpsc::Sender<Event>,
    notes: mpsc::Sender<Note>,
    /// Requests the head holds whose clients left, retired at the next touch.
    gone: BTreeSet<RequestId>,
    /// The pages in use the front was last told.
    pages_told: u64,
    /// Events heard while waiting on stage 1 that wait for the next turn.
    later: VecDeque<Event>,
    /// Where each handed-off request stands on its decode tier.
    tiers: Tiers,
    decode: Vec<Pd>,
    waiting: VecDeque<QueuedRequest>,
    active: VecDeque<Active>,
    /// Per item in flight, oldest first, the prompts its reply finishes.
    due: VecDeque<Vec<Finish>>,
    /// The line tables as the stages last had them written.
    lines: BTreeMap<String, Vec<i64>>,
    cache: Cache,
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
                let _ = ready.send(Ok((head.facts(), head.calls.clone(), head.rt.pages_total() as u64)));
                head.run()
            }
            Err(e) => {
                let _ = ready.send(Err(e));
            }
        })
        .context("spawning the head's thread")?;
    let (facts, calls, pages) = loaded.recv().context("the head's thread died loading")??;
    let head = move |c: Call| calls.send(Event::Call(c)).map_err(|_| anyhow::anyhow!("the pipeline head is gone"));
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
    let seqs = (h.max_seqs + 1) as u64;
    let mut rt = Runtime::load(&m, h.kernels.as_deref(), h.gpu, Some(Capacity { tokens, seqs, landings: 0 }), None)?;
    rt.set_eager(h.eager);
    weights.bind(&mut rt)?;
    let p = Protocol::check_unsampled(&rt.manifest)?;
    kern_run::run_once(&rt, &p)?;
    let pages = rt.pages_total() as u64;
    let turn = h.host_turn.div_ceil(unit);
    let host = host_units(h.host_bytes, turn, hellos.values().map(|(x, ..)| x.host).chain([rt.host_unit()]));

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
    let (tx, events) = mpsc::channel();
    let mut states = vec![shapes(&rt.manifest)];
    states.extend(hellos.values().map(|(x, ..)| x.states.clone()));
    let decode: Vec<Pd> = (h.decode.iter().enumerate())
        .map(|(i, addr)| decode_tier(&rt, addr, i, &states, tx.clone(), notes.clone()))
        .collect::<Result<_>>()?;
    let next_of = |s: u64| hellos.get(&(s + 1)).map(|(x, _, reach)| (reach.clone(), x.arrivals.clone()));
    let pools: Vec<Pools> = decode.iter().map(|p| p.pools.clone()).collect();
    let tables: Vec<Table> =
        (1..cut.stages).map(|s| Table { pages, next: next_of(s), decode: pools.clone(), host }).collect();
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
    reserve(&mut rt, host)?;
    let next = Next::connect(&rt, &first.0, &first.1, {
        let tx = tx.clone();
        move || {
            let _ = tx.send(Event::Taken);
        }
    })?;
    let tx_calls = tx.clone();
    let report: super::Sink = Box::new(move |r| match r {
        Report::Pushed(req) => {
            tx.send(Event::Pushed { stage: 0, req }).map_err(|_| anyhow::anyhow!("the head is gone"))
        }
        Report::Tokens(_) => bail!("the head answers no items"),
    });

    let step_name = step.as_ref().map_or("-", |f| f.name.as_str()).to_string();
    info!(pages, page = rt.page(), chunk, seqs = bounds.seqs, prefill = %prefill.name, step = %step_name,
          seq_slots = rt.has_seq_state().then(|| rt.seq_slots()), "pipeline head ready");
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
        events,
        calls: tx_calls,
        notes,
        gone: BTreeSet::new(),
        pages_told: 0,
        later: VecDeque::new(),
        tiers: Tiers::new(decode.len(), cut.stages as usize),
        decode,
        waiting: VecDeque::new(),
        active: VecDeque::new(),
        due: VecDeque::new(),
        lines: BTreeMap::new(),
        cache,
        window: Window { since: Instant::now(), fed: 0 },
        flow: Flow::new(0, None, Some(next), report, false),
    })
}

/// The host pages and host slots of `bytes` per stage, one slot per `turn`
/// pages, the same unit ids on every stage: the split the largest page and
/// the largest slot of any stage (each `(page bytes, slot bytes)`) make, so
/// each stage's block, those units times its own sizes, is at most `bytes`.
fn host_units(bytes: u64, turn: u64, stages: impl Iterator<Item = (u64, u64)>) -> (u64, u64) {
    let (page, slot) = stages.fold((0, 0), |(p, s), (q, t)| (p.max(q), s.max(t)));
    kern_runtime::host_split(bytes, page, slot, turn)
}

/// Reserve this stage's host block of `units` (host pages, host slots), none for (0, 0).
pub(super) fn reserve(rt: &mut Runtime, units: (u64, u64)) -> Result<()> {
    if units == (0, 0) {
        return Ok(());
    }
    let t0 = Instant::now();
    rt.reserve_host(units.0, units.1).context("reserving the host tier")?;
    let gib = rt.host_tier().map_or(0.0, |(_, b)| b as f64 / (1u64 << 30) as f64);
    info!(pages = units.0, slots = units.1, gib, reserve_s = t0.elapsed().as_secs_f64(), "host tier reserved");
    Ok(())
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
        self.active.len() + self.due.iter().map(Vec::len).sum::<usize>()
    }

    fn note(&self, n: Note) -> Result<()> {
        self.notes.send(n).map_err(|_| anyhow::anyhow!("the front is gone"))
    }

    /// Whether `id` is still the head's to answer.
    fn holds(&self, id: RequestId) -> bool {
        self.waiting.iter().any(|q| q.id == id)
            || self.active.iter().any(|a| a.id == id)
            || self.due.iter().flatten().any(|f| f.id == id)
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
            let handoff = tier.is_some();
            let limit = self.rt.max_seq_tokens();
            let ids: Vec<i64> = q.request.prompt_tokens.iter().map(|&t| t as i64).collect();
            let leased = match prompt {
                0 => Err(Error::Denied(Denied::ExceedsRow { limit })),
                _ => self.cache.lease(&mut self.rt, &ids),
            };
            let reject = match leased {
                Ok((lease, start)) => {
                    let q = self.waiting.pop_front().unwrap();
                    self.note(Note::Admit { id, cached: start })?;
                    debug!(request = %id, prompt, prefix_hit = start, "admitted");
                    // The prompt's whole self is kept at its end; a parting
                    // point is worth a checkpoint of its own only to a
                    // recurrent state, which cannot be continued from part
                    // of a longer one, and is then the only one kept.
                    let parted = q.request.history_tokens.filter(|&m| self.rt.has_seq_state() && m < prompt);
                    let mark = parted.filter(|&m| m > start);
                    let parts = parted.is_some();
                    let snap = match self.rt.has_seq_state() {
                        true => parted.unwrap_or(prompt).checked_sub(SNAPSHOT_SHORT).filter(|&s| s > start),
                        false => None,
                    };
                    let stop = q.request.stop_policy.clone();
                    let a = Active { id, lease, ids, pos: start, stop, mark, snap, parts, handoff, passed: 0 };
                    self.active.push_back(a);
                    if let Some(t) = tier {
                        self.tiers.admitted(t, id.raw());
                    }
                    continue;
                }
                Err(Error::Denied(Denied::Busy | Denied::Remapping)) => break,
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

    /// Carry out what the tiers decided, in order.
    fn apply(&mut self, outs: Vec<(usize, Out)>) -> Result<()> {
        for (tier, o) in outs {
            match o {
                Out::Ask(a) => send(&mut self.decode[tier].d, &a)?,
                Out::Push(p) => self.push(&p)?,
                Out::Place { tier, rank, epoch, placement } => {
                    let job = Job::Place { tier, rank, epoch, placement };
                    for w in &mut self.stages {
                        send(w, &job)?;
                    }
                    let Job::Place { placement, .. } = &job else { unreachable!("built above") };
                    self.decode[tier].mirrors.place(&self.rt, rank, epoch, placement)?;
                }
                Out::Emit { req, tok } => self.note(Note::Emit { id: RequestId::new(req), tok })?,
                Out::Finish { req, why, cause } => {
                    let id = RequestId::new(req);
                    self.gone.remove(&id);
                    self.note(Note::Finish { id, why, cause })?;
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

    /// Where the rows of a prompt stop for now: its snapshot, its mark, then
    /// its chunked rows.
    fn upto(&self, a: &Active) -> usize {
        a.snap.or(a.mark).unwrap_or(self.chunked(a))
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
        if goes_out(&plan, self.bounds, self.due.len(), self.depth) {
            for (a, n) in self.active.iter_mut().zip(passed(&waiting, &plan)) {
                a.passed = n;
            }
            self.send_chunk(&plan)?;
        }
        while let Some(i) = self.step.as_ref().and(self.active.iter().position(|a| a.pos == self.chunked(a))) {
            let a = self.active.remove(i).expect("found");
            let f = self.step.clone().expect("checked");
            self.send_item(&f, &[(&a.lease, a.pos, &a.ids[a.pos..])])?;
            self.prefilled(&a)?;
            let done = self.leave(a, 0);
            self.due.push_back(vec![done]);
        }
        Ok(())
    }

    /// One packed chunk item: each planned prompt's next rows; a prompt
    /// that reaches its mark is kept there, and the prompts the item
    /// finishes leave the active set for the item's reply.
    fn send_chunk(&mut self, plan: &[(usize, usize)]) -> Result<()> {
        let f = self.prefill.clone();
        let active = std::mem::take(&mut self.active);
        let parts: Vec<(&Lease, usize, &[i64])> = plan
            .iter()
            .map(|&(i, n)| {
                let a = &active[i];
                (&a.lease, a.pos, &a.ids[a.pos..a.pos + n])
            })
            .collect();
        let sent = self.send_item(&f, &parts);
        drop(parts);
        self.active = active;
        sent?;
        let emits = self.step.is_none();
        let mut finished = Vec::new();
        for (at, &(i, n)) in plan.iter().enumerate() {
            let a = &mut self.active[i];
            a.pos += n;
            if a.snap == Some(a.pos) {
                a.snap = None;
                self.cache.snapshot(&mut self.rt, &mut a.lease, &a.ids[..a.pos])?;
            }
            if a.mark == Some(a.pos) {
                a.mark = None;
                self.cache.keep(&mut self.rt, &mut a.lease, &a.ids[..a.pos])?;
            }
            if emits && a.pos == a.ids.len() {
                finished.push((i, at));
            }
        }
        // Highest index first, so each removal leaves the next one's index in place.
        finished.sort_unstable_by(|x, y| y.cmp(x));
        let mut due = Vec::with_capacity(finished.len());
        for (i, at) in finished {
            let a = self.active.remove(i).expect("planned");
            self.prefilled(&a)?;
            due.push(self.leave(a, at));
        }
        due.sort_unstable_by_key(|f| f.at);
        self.due.push_back(due);
        Ok(())
    }

    /// A prompt whose last rows are out, its token entry `at` of the item's
    /// reply: its lease becomes an entry, or goes back to the pool when the
    /// next turn parts from it before its end.
    fn leave(&mut self, a: Active, at: usize) -> Finish {
        match a.parts {
            true => drop(a.lease),
            false => self.cache.retire(&mut self.rt, a.lease, &a.ids),
        }
        Finish { at, id: a.id, stop: a.stop, handoff: a.handoff }
    }

    /// A prompt whose last rows were just issued: handed off, it is pushed
    /// now, from its lease, before the lease goes to the cache.
    fn prefilled(&mut self, a: &Active) -> Result<()> {
        if !a.handoff {
            return Ok(());
        }
        let outs = self.tiers.prefilled(a.id.raw(), &a.lease, a.ids.len());
        self.apply(outs)
    }

    /// Stage one item of `parts` (a lease, its position, its rows) on every
    /// stage and on the head. The acts the cache ran on the head since the
    /// item before go first (the stages run them, the head has); a line
    /// table goes out only when the item's sequences change it.
    fn send_item(&mut self, f: &Forward, parts: &[(&Lease, usize, &[i64])]) -> Result<()> {
        let leases: Vec<&Lease> = parts.iter().map(|&(l, ..)| l).collect();
        let positions: Vec<usize> = parts.iter().map(|&(_, pos, _)| pos).collect();
        let lens: Vec<usize> = parts.iter().map(|(.., ids)| ids.len()).collect();
        let ids: Vec<i64> = parts.iter().flat_map(|(.., ids)| ids.iter().copied()).collect();
        let (vars, values) = kern_run::fills(&self.p, &leases, &positions, &lens, &ids);
        let mut rows: BTreeMap<String, Vec<i64>> = values.into_iter().map(|(f, v)| (f.name.clone(), v)).collect();
        for t in &self.p.page_tables {
            let mut v = Vec::with_capacity(t.width * leases.len());
            for l in &leases {
                l.extend_row(&t.name, &mut v)?;
            }
            rows.insert(t.name.clone(), v.into_iter().map(i64::from).collect());
        }
        for t in &self.p.line_tables {
            let cols = match t.axis {
                Axis::Ranks => self.p.ranks.as_ref().map_or(1, |b| b.max),
                _ => self.p.groups.max,
            };
            let v: Vec<i64> = kern_run::line_rows(t, &leases, cols as usize)?.into_iter().map(i64::from).collect();
            if self.lines.get(&t.name) != Some(&v) {
                self.lines.insert(t.name.clone(), v.clone());
                rows.insert(t.name.clone(), v);
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

    /// The head's life: hear what came, admit, issue, until the process ends.
    /// A failed turn ends the process: the stages see the connection close
    /// and follow, and no half-fed pipeline lingers behind an open port.
    fn run(mut self) {
        loop {
            if let Err(e) = self.turn() {
                error!("pipeline head: {e:#}");
                std::process::exit(1);
            }
        }
    }

    fn turn(&mut self) -> Result<()> {
        if self.later.is_empty() {
            match self.events.recv_timeout(IDLE) {
                Ok(e) => self.later.push_back(e),
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => bail!("every connection to the head is gone"),
            }
        }
        self.later.extend(self.events.try_iter());
        while let Some(e) = self.later.pop_front() {
            self.hear(e)?;
        }
        self.retire_gone()?;
        self.admit()?;
        self.issue()?;
        ensure!(self.rt.remaps() == 0, "the head's pool remapped its chunks, which the stages cannot follow");
        let used = self.rt.pages_used() as u64;
        if used != self.pages_told {
            self.pages_told = used;
            self.note(Note::Pages(used))?;
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
            let host = self.rt.host_tier();
            info!(
                running = self.live(),
                waiting = self.waiting.len(),
                kv_pct = round(self.rt.pages_used() as f64 * 100.0 / self.rt.pages_total().max(1) as f64, 10.0),
                prefill_tokens = self.window.fed,
                prefill_tok_s = round(self.window.fed as f64 / dt.as_secs_f64(), 1.0),
                resident_hits = c.hits,
                resident_hit_tokens = c.hit_tokens,
                host_hits = host.and(Some(c.host_hits)),
                host_hit_tokens = host.and(Some(c.host_hit_tokens)),
                checkpoints = self.cache.entries(),
                parked = host.and(Some(self.cache.parked())),
                host_pct = host.map(|(u, r)| round(u as f64 * 100.0 / r.max(1) as f64, 10.0)),
                parks = host.and(Some(c.parks)),
                host_evictions = host.and(Some(c.host_evictions)),
                prompt_checkpoints = state.then_some(c.kept),
                prompt_skips = state.then_some(c.skipped),
                snapshots = state.then_some(c.snapshots),
                snapshot_skips = state.then_some(c.snapshot_skips),
                evictions = c.evictions,
                superseded = state.then(|| self.cache.superseded()),
                slots_used = state.then(|| self.rt.seq_slots_used()),
                slots = state.then(|| self.rt.seq_slots()),
                decode_seats = ?self.tiers.loads().iter().map(|l| l.seats).collect::<Vec<_>>(),
                decode_kv_pct = ?self.tiers.loads().iter().map(|l| round(l.pages as f64 * 100.0 / l.pages_total.max(1) as f64, 10.0)).collect::<Vec<_>>(),
                "stats"
            );
        }
        self.window = Window { since: Instant::now(), fed: 0 };
    }

    /// Wait until stage 1 has taken the item before the one just issued,
    /// hearing all but the front's submissions and departures meanwhile.
    fn taken(&mut self) -> Result<()> {
        let t0 = Instant::now();
        while !self.flow.ready()? {
            match self.events.recv_timeout(super::TIMEOUT.saturating_sub(t0.elapsed())) {
                Ok(e @ Event::Call(Call::Submit(_) | Call::Gone(_))) => self.later.push_back(e),
                Ok(e) => self.hear(e)?,
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
    fn hear(&mut self, e: Event) -> Result<()> {
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
        };
        let Some(due) = self.due.pop_front() else { bail!("an answer came back for no item") };
        for r in due {
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
            let why = if r.stop.classify(tok, |_| false).is_some() { FinishReason::Stop } else { FinishReason::Length };
            self.note(Note::Emit { id: r.id, tok })?;
            self.note(Note::Finish { id: r.id, why, cause: None })?;
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
