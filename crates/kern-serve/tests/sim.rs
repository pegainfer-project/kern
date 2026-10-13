//! A deterministic simulation of one prefill pipeline handing requests to
//! one decode tier, built to find what the P/D protocol lets go wrong
//! (docs/protocol.md).
//!
//! Every thread of the deployment is an actor and every connection a FIFO:
//! the P head (its [`Handoffs`], carrying out each [`Out`] as a step of its
//! own, as `apply` does), the front (its [`Relays`] and its sweep of
//! departed clients), the D tier's reader and scheduler (a real
//! [`Decoder`] over real pools), the reader of D's connection on P, the
//! stages pushing into their mirrors of D's pools, the last stage's first
//! tokens, and the clients, who may leave at any step. A seeded schedule
//! picks which enabled actor moves next. The head's and the front's glue
//! around their pure halves is modelled here after `pp/head.rs` and
//! `front.rs`: those halves are bound to a CUDA runtime and a pegainfer
//! ledger, so no test can run them.
//!
//! What it checks, at every step: no process dies on what a peer sent; a
//! stage writes into a D object only through a mirror that maps that
//! object's chunk positions where D's pool has them; no seat is freed and
//! no abort sent while a push into it is in flight. At quiescence: every
//! client ended exactly once, with the greedy chain when it finished, and
//! nothing is held anywhere. A violation is shrunk to the shortest
//! schedule that still shows it.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;

use kern_manifest::types::Manifest;
use kern_pool::{Kind, Placement, Pool, Pooled};
use kern_serve::pd::{Ask, Decoder, Handoffs, Out, Push, Relay, Relays, Req, Run, Seat, Standing, Tell};
use kern_serve::ranks::{Act, Answer, Order};
use pegainfer_frontend::engine::{EosPolicy, FinishReason, StopPolicy};

mod common;
use common::{dealt_with, replicated_of, Rand};

const STAGES: usize = 3;
/// D's group: two members, each holding every row's every other position.
const W: usize = 2;
const STOP: u32 = 7;
/// Prompts P seats; longer ones it rejects after their grant.
const P_LIMIT: usize = 64;

/// The fake model: a row fed `t` hands back `next(t)`.
fn next(t: i64) -> i64 {
    t * 7 % 97 + 1
}

fn first_token(req: Req) -> u32 {
    (req * 13 % 97 + 1) as u32
}

fn stop() -> StopPolicy {
    StopPolicy::new(EosPolicy::Ignore, vec![STOP])
}

/// Ranks that land every remap the moment they hear it and answer every
/// row with [`next`].
struct Fake;

impl Run for Fake {
    fn run(&mut self, orders: Vec<Order>) -> anyhow::Result<Vec<Answer>> {
        Ok(orders
            .iter()
            .map(|o| {
                let remaps = o.acts.iter().filter(|a| matches!(a, Act::Remap(_))).count() as u64;
                let tokens =
                    o.item.as_ref().map_or(Vec::new(), |i| i.rows["token_ids"].iter().map(|&t| next(t)).collect());
                Answer { tokens, remaps, ..Answer::default() }
            })
            .collect())
    }
}

/// How a deployment is set up.
#[derive(Debug, Clone, Copy)]
struct Config {
    requests: u64,
    /// Slots per D member at the start, the null one and the pad's among
    /// them: three leave one, so a second seat makes another slot (a remap).
    slots: usize,
    /// Bytes of a member's slot. A chunk is 256: a slot of 24 shares a
    /// chunk mapped already, one of 768 takes three from free pages, and
    /// the placement moves.
    slot: u64,
    seats: usize,
    /// How many clients may leave before their end.
    leave: usize,
    /// Whether some prompts are longer than P seats (rejected after their grant).
    long: bool,
}

/// What the protocol let go wrong.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Violation {
    /// The decode tier's process exits on what P sent it.
    DecodeDied(String),
    /// The P head's process exits on what D or a stage told it.
    PrefillDied(String),
    /// A stage writes request `req`'s state into D rank `rank` through a
    /// mirror that maps one of the seat's chunk positions elsewhere.
    StaleMirror {
        stage: usize,
        req: Req,
        rank: usize,
        arena: usize,
        position: usize,
    },
    FreedMidPush(Req),
    AbortMidPush(Req),
    /// At quiescence, a client's end is wrong or missing, or something is held.
    End(String),
}

impl Violation {
    fn kind(&self) -> &'static str {
        match self {
            Violation::DecodeDied(_) => "decode died",
            Violation::PrefillDied(_) => "prefill died",
            Violation::StaleMirror { .. } => "stale mirror",
            Violation::FreedMidPush(_) => "freed mid-push",
            Violation::AbortMidPush(_) => "abort mid-push",
            Violation::End(_) => "end",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum End {
    Finished(Vec<u32>, FinishReason),
    Rejected,
    Retired,
}

struct Client {
    prompt: usize,
    max_tokens: usize,
    tokens: Vec<u32>,
    end: Option<End>,
    /// The ledger's abort flag: the client left.
    gone: bool,
}

/// What reaches the P head's event channel.
#[derive(Debug)]
enum Event {
    Tell(Tell),
    Pushed { stage: usize, req: Req },
    First(Req),
    Gone(Req),
}

/// What reaches the front's note channel.
#[derive(Debug)]
enum Note {
    Admit,
    Reject(Req),
    Emit(Req, u32),
    Finish(Req, FinishReason),
    Retire(Req),
    Decode(Req, Relay),
    Tokens(Vec<(Req, u32, bool)>),
}

/// One thing that can happen next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Step {
    /// The head carries out the next [`Out`] it decided.
    HeadOut,
    HeadHear,
    /// The head's turn: retire the departed it holds, admit.
    HeadTurn,
    /// The head issues `req`'s last rows: its push, its first token on the way.
    HeadPrefill(Req),
    FrontNote,
    FrontSweep,
    Leave(Req),
    DHear,
    Decode,
    /// P's reader of D's connection: tokens to the front, the rest to the head.
    PRead,
    Stage(usize),
    /// The last stage answers the item that carried `req`'s last rows.
    Last(Req),
}

/// Where chunk positions are on each D rank, and which a stage writes.
struct Geometry {
    pooled: Vec<Pooled>,
    chunk: u64,
    /// Per stage, the arenas of the states it holds.
    arenas: Vec<Vec<usize>>,
}

impl Geometry {
    fn new(m: &Manifest, slots: usize) -> Geometry {
        let twin = Pool::new(m, 256, 128, slots, None).unwrap().0;
        let pooled = twin.pooled().to_vec();
        let mut names: Vec<&str> = pooled.iter().map(|p| p.state.as_str()).collect();
        names.sort_unstable();
        // One state a stage, in name order.
        let arenas = (0..STAGES)
            .map(|s| {
                pooled
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| names.get(s) == Some(&p.state.as_str()))
                    .map(|(a, _)| a)
                    .collect()
            })
            .collect();
        Geometry { pooled, chunk: twin.chunk(), arenas }
    }

    /// The chunk positions `seat`'s objects cover in arena `a`.
    fn positions(&self, a: usize, seat: &Seat) -> Vec<usize> {
        let p = &self.pooled[a];
        let ids: Vec<usize> = match p.kind {
            Kind::Page => seat.pages.iter().map(|&i| i as usize).collect(),
            Kind::Slot => seat.slot.iter().map(|&i| i as usize).collect(),
        };
        ids.iter()
            .flat_map(|&i| {
                let lo = p.object * i as u64;
                (lo / self.chunk) as usize..(lo + p.object).div_ceil(self.chunk) as usize
            })
            .collect()
    }
}

fn placement(hex: &str) -> Placement {
    Placement::from_bytes(&hex::decode(hex).unwrap()).unwrap()
}

struct Head {
    h: Handoffs,
    outbox: VecDeque<Out>,
    events: VecDeque<Event>,
    waiting: VecDeque<Req>,
    /// Admitted, last rows not yet issued: the lease and whether it hands off.
    admitted: BTreeMap<Req, (kern_pool::Lease, bool)>,
    gone: BTreeSet<Req>,
}

struct Front {
    r: Relays,
    notes: VecDeque<Note>,
    /// Submitted, not answered; whether the head was told the client left.
    open: BTreeMap<Req, bool>,
}

/// What the head sends a stage, in order.
#[derive(Debug)]
enum Job {
    Push(Push),
    Place { rank: usize, epoch: u64, placement: String },
}

struct Stage {
    inbox: VecDeque<Job>,
    /// Every D rank's pool as this stage last placed it, and the epoch.
    mirror: Vec<(Placement, u64)>,
}

struct Sim {
    geo: Geometry,
    clients: BTreeMap<Req, Client>,
    head: Head,
    front: Front,
    d: Decoder<Fake>,
    to_d: VecDeque<Ask>,
    to_p: VecDeque<Tell>,
    stages: Vec<Stage>,
    /// Requests whose last rows are issued and whose first token is not back.
    firsts: BTreeSet<Req>,
    p_pool: Arc<Pool>,
    /// Stages yet to push, by request.
    in_flight: BTreeMap<Req, usize>,
    left: usize,
    log: Vec<String>,
}

type Checked<T> = Result<T, Violation>;

impl Sim {
    fn new(cfg: Config, rand: &mut Rand) -> Sim {
        let m = dealt_with(W, cfg.slot);
        let plain =
            include_str!("../../kern-manifest/tests/fixtures/plain.json").replace("[\"seqs\", 3]", "[\"seqs\", 8]");
        let plain = Manifest::from_json(&plain).unwrap();
        let mut clients = BTreeMap::new();
        for req in 1..=cfg.requests {
            let prompt = match rand.next(10) {
                0 if cfg.long => 97,
                1 if cfg.long => P_LIMIT + 1,
                _ => 1 + rand.next(60),
            };
            let max_tokens = [1, 2, 3, 5, 9][rand.next(5)];
            clients.insert(req, Client { prompt, max_tokens, tokens: Vec::new(), end: None, gone: false });
        }
        let d = Decoder::new(replicated_of(m.clone(), W, 1, cfg.slots), Fake, cfg.seats).unwrap();
        let pools = d.pools(vec![String::new(); W], kern_serve::pd::shapes(&m));
        let mirror: Vec<(Placement, u64)> = pools.placements.iter().map(|h| (placement(h), 0)).collect();
        Sim {
            geo: Geometry::new(&m, cfg.slots),
            head: Head {
                h: Handoffs::new(STAGES, 0),
                outbox: VecDeque::new(),
                events: VecDeque::new(),
                waiting: clients.keys().copied().collect(),
                admitted: BTreeMap::new(),
                gone: BTreeSet::new(),
            },
            front: Front {
                r: Relays::new(),
                notes: VecDeque::new(),
                open: clients.keys().map(|&r| (r, false)).collect(),
            },
            clients,
            d,
            to_d: VecDeque::new(),
            to_p: VecDeque::new(),
            stages: (0..STAGES).map(|_| Stage { inbox: VecDeque::new(), mirror: mirror.clone() }).collect(),
            firsts: BTreeSet::new(),
            p_pool: Arc::new(Pool::new(&plain, 64, 1024, 16, None).unwrap().0),
            in_flight: BTreeMap::new(),
            left: cfg.leave,
            log: Vec::new(),
        }
    }

    /// Say what the current step did, on its line of the log.
    fn say(&mut self, what: String) {
        if let Some(l) = self.log.last_mut() {
            l.push_str(&format!(" | {what}"));
        }
    }

    /// What can happen now, in a fixed order.
    fn enabled(&self) -> Vec<Step> {
        let mut v = Vec::new();
        match self.head.outbox.is_empty() {
            // `apply` runs its outs back to back on the head's thread.
            false => v.push(Step::HeadOut),
            true => {
                if !self.head.events.is_empty() {
                    v.push(Step::HeadHear);
                }
                if !self.head.waiting.is_empty() || self.head.h.open().iter().any(|r| self.head.gone.contains(r)) {
                    v.push(Step::HeadTurn);
                }
                v.extend(self.head.admitted.keys().map(|&r| Step::HeadPrefill(r)));
            }
        }
        if !self.front.notes.is_empty() {
            v.push(Step::FrontNote);
        }
        if self.front.open.iter().any(|(r, &told)| self.clients[r].gone && (!told || self.front.r.holds(*r))) {
            v.push(Step::FrontSweep);
        }
        if self.left > 0 {
            v.extend(self.clients.iter().filter(|(_, c)| c.end.is_none() && !c.gone).map(|(&r, _)| Step::Leave(r)));
        }
        if !self.to_d.is_empty() {
            v.push(Step::DHear);
        }
        if self.d.busy() {
            v.push(Step::Decode);
        }
        if !self.to_p.is_empty() {
            v.push(Step::PRead);
        }
        v.extend((1..STAGES).filter(|&s| !self.stages[s].inbox.is_empty()).map(Step::Stage));
        v.extend(self.firsts.iter().map(|&r| Step::Last(r)));
        v
    }

    fn step(&mut self, s: Step) -> Checked<()> {
        self.log.push(format!("{s:?}"));
        match s {
            Step::HeadOut => {
                let o = self.head.outbox.pop_front().expect("enabled");
                if !matches!(o, Out::Ask(_) | Out::Push(_)) {
                    self.say(format!("{o:?}"));
                }
                self.carry(o)
            }
            Step::HeadHear => {
                let e = self.head.events.pop_front().expect("enabled");
                self.hear(e)
            }
            Step::HeadTurn => {
                self.retire_gone()?;
                self.admit()
            }
            Step::HeadPrefill(req) => {
                let (lease, handoff) = self.head.admitted.remove(&req).expect("enabled");
                self.say(format!("prompt {}", self.clients[&req].prompt));
                if handoff {
                    let outs = self.head.h.prefilled(req, &lease, self.clients[&req].prompt);
                    self.head.outbox.extend(outs);
                }
                // The head's lease goes to its cache: P's pages are not checked here.
                drop(lease);
                self.firsts.insert(req);
                Ok(())
            }
            Step::FrontNote => {
                let n = self.front.notes.pop_front().expect("enabled");
                self.say(format!("{n:?}"));
                self.note(n)
            }
            Step::FrontSweep => self.sweep(),
            Step::Leave(req) => {
                self.say(format!("{:?}", self.head.h.standing(req)));
                self.left -= 1;
                self.clients.get_mut(&req).unwrap().gone = true;
                Ok(())
            }
            Step::DHear => {
                let a = self.to_d.pop_front().expect("enabled");
                let before = self.d.seated();
                let tells = self.d.ask(a).map_err(|e| Violation::DecodeDied(format!("{e:#}")))?;
                self.told(tells, before)
            }
            Step::Decode => {
                let before = self.d.seated();
                let tells = self.d.step().map_err(|e| Violation::DecodeDied(format!("{e:#}")))?;
                self.told(tells, before)
            }
            Step::PRead => {
                match self.to_p.pop_front().expect("enabled") {
                    Tell::Tokens { step, .. } => self.front.notes.push_back(Note::Tokens(step)),
                    t => self.head.events.push_back(Event::Tell(t)),
                }
                Ok(())
            }
            Step::Stage(s) => match self.stages[s].inbox.pop_front().expect("enabled") {
                Job::Push(p) => self.push(s, &p),
                Job::Place { rank, epoch, placement } => self.place(s, rank, epoch, &placement),
            },
            Step::Last(req) => {
                self.firsts.remove(&req);
                self.head.events.push_back(Event::First(req));
                Ok(())
            }
        }
    }

    /// The head carries out one decision, as `PpHead::apply`.
    fn carry(&mut self, o: Out) -> Checked<()> {
        let n = match o {
            Out::Ask(a) => return self.ask(a),
            Out::Push(p) => {
                self.in_flight.insert(p.req, STAGES);
                for s in 1..STAGES {
                    self.stages[s].inbox.push_back(Job::Push(p.clone()));
                }
                // The head is stage 0: it pushes from its own mirrors at once.
                return self.push(0, &p);
            }
            Out::Place { rank, epoch, placement, .. } => {
                for s in 1..STAGES {
                    self.stages[s].inbox.push_back(Job::Place { rank, epoch, placement: placement.clone() });
                }
                return self.place(0, rank, epoch, &placement);
            }
            Out::Emit { req, tok } => Note::Emit(req, tok),
            Out::Finish { req, why, .. } => {
                self.head.gone.remove(&req);
                Note::Finish(req, why)
            }
            Out::Decode { req, relay } => {
                self.head.gone.remove(&req);
                Note::Decode(req, relay)
            }
            Out::Reject { req, .. } => {
                let i = self.head.waiting.iter().position(|&r| r == req);
                let i =
                    i.ok_or_else(|| Violation::PrefillDied(format!("a refused request {req} no longer waiting")))?;
                self.head.waiting.remove(i);
                self.head.gone.remove(&req);
                Note::Reject(req)
            }
        };
        self.front.notes.push_back(n);
        Ok(())
    }

    /// An ask onto the one connection to D, whichever thread writes it.
    fn ask(&mut self, a: Ask) -> Checked<()> {
        if let Ask::Abort { req } = a {
            if self.in_flight.contains_key(&req) {
                return Err(Violation::AbortMidPush(req));
            }
        }
        self.say(format!("→D {a:?}"));
        self.to_d.push_back(a);
        Ok(())
    }

    fn told(&mut self, tells: Vec<Tell>, before: Vec<Req>) -> Checked<()> {
        let after: BTreeSet<Req> = self.d.seated().into_iter().collect();
        if let Some(&r) = before.iter().find(|r| !after.contains(r) && self.in_flight.contains_key(r)) {
            return Err(Violation::FreedMidPush(r));
        }
        for t in &tells {
            let said = match t {
                Tell::Grant { req, seats, .. } => format!(
                    "D→ Grant {req} slots {:?} at epochs {:?}",
                    seats.iter().map(|s| s.slot).collect::<Vec<_>>(),
                    seats.iter().map(|s| s.epoch).collect::<Vec<_>>()
                ),
                Tell::Placed { rank, epoch, .. } => format!("D→ Placed rank {rank} at {epoch}"),
                Tell::Tokens { step, .. } => format!("D→ Tokens {step:?}"),
                t => format!("D→ {t:?}"),
            };
            self.say(said);
        }
        self.to_p.extend(tells);
        Ok(())
    }

    /// Stage `s` places D rank `rank`'s mirror at `epoch`, as `Mirrors::place`.
    fn place(&mut self, s: usize, rank: usize, epoch: u64, hex: &str) -> Checked<()> {
        let at = self.stages[s].mirror[rank].1;
        self.say(format!("stage {s} places rank {rank} at {epoch}"));
        if epoch != at + 1 {
            return Err(Violation::PrefillDied(format!("stage {s}: rank {rank}'s placement {epoch} after {at}")));
        }
        self.stages[s].mirror[rank] = (placement(hex), epoch);
        Ok(())
    }

    /// Stage `s` writes every seat through its mirrors once they reach the
    /// seats' epochs, as `Mirrors::push`: each position it writes must be
    /// where D has it.
    fn push(&mut self, s: usize, p: &Push) -> Checked<()> {
        self.say(format!(
            "stage {s} pushes {} into slots {:?} at epochs {:?}",
            p.req,
            p.seats.iter().map(|s| s.slot).collect::<Vec<_>>(),
            p.seats.iter().map(|s| s.epoch).collect::<Vec<_>>()
        ));
        if let Some(seat) = p.seats.iter().find(|seat| self.stages[s].mirror[seat.rank].1 < seat.epoch) {
            let at = self.stages[s].mirror[seat.rank].1;
            return Err(Violation::PrefillDied(format!(
                "stage {s}: request {} at placement {} of rank {}, the mirror at {at}",
                p.req, seat.epoch, seat.rank
            )));
        }
        let truth = self.d.pools(vec![String::new(); W], BTreeMap::new()).placements;
        for seat in &p.seats {
            let now = placement(&truth[seat.rank]);
            for &a in &self.geo.arenas[s] {
                let bad = self
                    .geo
                    .positions(a, seat)
                    .into_iter()
                    .find(|&x| self.stages[s].mirror[seat.rank].0.at(a, x) != now.at(a, x));
                if let Some(position) = bad {
                    return Err(Violation::StaleMirror { stage: s, req: p.req, rank: seat.rank, arena: a, position });
                }
            }
        }
        let left = self.in_flight.get_mut(&p.req).expect("a push in flight");
        *left -= 1;
        if *left == 0 {
            self.in_flight.remove(&p.req);
        }
        self.head.events.push_back(Event::Pushed { stage: s, req: p.req });
        Ok(())
    }

    fn died<T>(r: anyhow::Result<T>) -> Checked<T> {
        r.map_err(|e| Violation::PrefillDied(format!("{e:#}")))
    }

    /// One event heard, as `PpHead::hear`.
    fn hear(&mut self, e: Event) -> Checked<()> {
        let outs = match e {
            Event::Tell(t) => Self::died(self.head.h.hear(t))?,
            Event::Pushed { stage, req } => {
                if self.head.h.standing(req) == Standing::Unasked {
                    return Err(Violation::PrefillDied(format!("stage {stage} pushed request {req}, on no tier")));
                }
                Self::died(self.head.h.pushed(req, stage))?
            }
            Event::First(req) => {
                let tok = first_token(req);
                if Relay::hands_off(self.clients[&req].max_tokens) {
                    match self.head.h.standing(req) {
                        Standing::Unasked => Vec::new(),
                        _ => Self::died(self.head.h.first(req, tok))?,
                    }
                } else if self.head.gone.remove(&req) {
                    self.front.notes.push_back(Note::Retire(req));
                    Vec::new()
                } else {
                    let why = if tok == STOP { FinishReason::Stop } else { FinishReason::Length };
                    self.front.notes.push_back(Note::Emit(req, tok));
                    self.front.notes.push_back(Note::Finish(req, why));
                    Vec::new()
                }
            }
            Event::Gone(req) => {
                let holds = self.head.waiting.contains(&req)
                    || self.head.admitted.contains_key(&req)
                    || self.firsts.contains(&req)
                    || self.head.h.open().contains(&req);
                if holds {
                    self.head.gone.insert(req);
                }
                Vec::new()
            }
        };
        self.head.outbox.extend(outs);
        Ok(())
    }

    /// As `PpHead::retire_gone`.
    fn retire_gone(&mut self) -> Checked<()> {
        let gone: Vec<Req> = self.head.h.open().into_iter().filter(|r| self.head.gone.contains(r)).collect();
        for req in gone {
            self.head.gone.remove(&req);
            self.front.notes.push_back(Note::Retire(req));
            let outs = self.head.h.abort(req);
            self.head.outbox.extend(outs);
        }
        Ok(())
    }

    /// As `PpHead::admit`, over the one decode tier.
    fn admit(&mut self) -> Checked<()> {
        while let Some(&req) = self.head.waiting.front() {
            if self.head.gone.remove(&req) {
                self.front.notes.push_back(Note::Retire(req));
                self.head.waiting.pop_front();
                if self.head.h.standing(req) != Standing::Unasked {
                    let outs = self.head.h.abort(req);
                    self.head.outbox.extend(outs);
                }
                continue;
            }
            let c = &self.clients[&req];
            let handoff = Relay::hands_off(c.max_tokens);
            match (handoff, self.head.h.standing(req)) {
                (true, Standing::Unasked) => {
                    let relay = Relay::new(c.max_tokens, stop());
                    if let Some(a) = self.head.h.reserve(req, c.prompt + c.max_tokens, relay) {
                        self.head.outbox.push_back(Out::Ask(a));
                    }
                    return Ok(());
                }
                (true, Standing::Reserving) => return Ok(()),
                (true, Standing::Underway) => {
                    return Err(Violation::PrefillDied(format!("request {req} is underway while waiting")))
                }
                _ => {}
            }
            self.head.waiting.pop_front();
            if c.prompt > P_LIMIT {
                self.front.notes.push_back(Note::Reject(req));
                if handoff {
                    let outs = self.head.h.release(req);
                    self.head.outbox.extend(outs);
                }
                continue;
            }
            let lease = self.p_pool.lease(c.prompt).expect("P's pool holds every prompt");
            self.front.notes.push_back(Note::Admit);
            if handoff {
                self.head.h.admitted(req);
            }
            self.head.admitted.insert(req, (lease, handoff));
        }
        Ok(())
    }

    /// One note, as `Front::note` and `Front::relay`.
    fn note(&mut self, n: Note) -> Checked<()> {
        match n {
            Note::Admit => {}
            Note::Reject(req) => self.end(req, End::Rejected)?,
            Note::Emit(req, tok) => self.emit(req, tok)?,
            Note::Finish(req, why) => {
                let t = self.clients[&req].tokens.clone();
                self.end(req, End::Finished(t, why))?;
            }
            Note::Retire(req) => self.end(req, End::Retired)?,
            Note::Decode(req, relay) => self.front.r.start(req, relay),
            Note::Tokens(step) => {
                let outs = self.front.r.hear(step).map_err(|e| Violation::PrefillDied(format!("{e:#}")))?;
                self.relay(outs)?;
            }
        }
        Ok(())
    }

    fn relay(&mut self, outs: Vec<Out>) -> Checked<()> {
        for o in outs {
            match o {
                Out::Emit { req, tok } => self.emit(req, tok)?,
                Out::Finish { req, why, .. } => {
                    let t = self.clients[&req].tokens.clone();
                    self.end(req, End::Finished(t, why))?;
                }
                Out::Ask(a @ Ask::Abort { .. }) => self.ask(a)?,
                o => return Err(Violation::PrefillDied(format!("the relays asked for {o:?}"))),
            }
        }
        Ok(())
    }

    /// As `Front::sweep`.
    fn sweep(&mut self) -> Checked<()> {
        let gone: Vec<Req> = self
            .front
            .open
            .iter()
            .filter(|(r, &told)| self.clients[r].gone && (!told || self.front.r.holds(**r)))
            .map(|(&r, _)| r)
            .collect();
        for req in gone {
            if self.front.r.holds(req) {
                self.end(req, End::Retired)?;
                let outs = self.front.r.abort(req);
                self.relay(outs)?;
            } else {
                self.front.open.insert(req, true);
                self.head.events.push_back(Event::Gone(req));
            }
        }
        Ok(())
    }

    fn emit(&mut self, req: Req, tok: u32) -> Checked<()> {
        let c = self.clients.get_mut(&req).unwrap();
        if c.end.is_some() {
            return Err(Violation::End(format!("a token for {req} after its end")));
        }
        c.tokens.push(tok);
        Ok(())
    }

    fn end(&mut self, req: Req, e: End) -> Checked<()> {
        let c = self.clients.get_mut(&req).unwrap();
        if let Some(was) = &c.end {
            return Err(Violation::End(format!("{req} ended twice: {was:?}, then {e:?}")));
        }
        c.end = Some(e);
        self.front.open.remove(&req);
        Ok(())
    }

    /// At quiescence: every client's end is right and nothing is held.
    fn settled(&self) -> Checked<()> {
        for (req, c) in &self.clients {
            let mut chain = vec![first_token(*req)];
            while chain.len() < c.max_tokens && *chain.last().unwrap() != STOP {
                chain.push(next(*chain.last().unwrap() as i64) as u32);
            }
            let why = if *chain.last().unwrap() == STOP { FinishReason::Stop } else { FinishReason::Length };
            let ok = match &c.end {
                Some(End::Finished(t, r)) => (t, *r) == (&chain, why),
                Some(End::Rejected) => c.prompt > P_LIMIT,
                Some(End::Retired) => c.gone,
                None => false,
            };
            if !ok {
                return Err(Violation::End(format!(
                    "request {req}: {:?} (prompt {}, gone {})",
                    c.end, c.prompt, c.gone
                )));
            }
        }
        let held = (self.head.h.live(), self.front.r.live(), self.d.seated());
        match held {
            (0, 0, ref s) if s.is_empty() => Ok(()),
            _ => Err(Violation::End(format!("held at the end: {held:?}"))),
        }
    }
}

/// Run `cfg` under `choices`, each picking among the enabled steps, then
/// the first enabled step until nothing is: the steps taken, or the first
/// violation and the steps up to it.
fn run(cfg: Config, seed: u64, choices: &[usize]) -> (Vec<String>, Option<Violation>) {
    let mut sim = Sim::new(cfg, &mut Rand(seed));
    let mut it = choices.iter();
    for _ in 0..20_000 {
        let v = sim.enabled();
        if v.is_empty() {
            let end = sim.settled().err();
            return (sim.log, end);
        }
        let s = v[it.next().map_or(0, |&c| c % v.len())];
        if let Err(e) = sim.step(s) {
            return (sim.log, Some(e));
        }
    }
    (sim.log, Some(Violation::End("no quiescence in 20000 steps".into())))
}

/// The first of `seeds` schedules of `cfg` that violates something, shrunk:
/// (seed, choices, steps, violation).
fn explore(cfg: Config, seeds: u64) -> Option<(u64, Vec<usize>, Vec<String>, Violation)> {
    let mut rand = Rand(0x9E37_79B9_7F4A_7C15);
    for seed in 1..=seeds {
        let choices: Vec<usize> = (0..400).map(|_| rand.next(1 << 16)).collect();
        if let (_, Some(v)) = run(cfg, seed, &choices) {
            let choices = shrink(cfg, seed, choices, v.kind());
            let (log, v) = run(cfg, seed, &choices);
            return Some((seed, choices, log, v.expect("shrunk to a violation")));
        }
    }
    None
}

/// Drop runs of choices while the schedule still violates `kind`, halving
/// the run length down to one choice.
fn shrink(cfg: Config, seed: u64, mut c: Vec<usize>, kind: &str) -> Vec<usize> {
    let still = |c: &[usize]| run(cfg, seed, c).1.is_some_and(|v| v.kind() == kind);
    let mut n = c.len().div_ceil(2).max(1);
    loop {
        let mut i = 0;
        while i < c.len() {
            let t: Vec<usize> = c[..i].iter().chain(&c[(i + n).min(c.len())..]).copied().collect();
            if still(&t) {
                c = t;
            } else {
                i += n;
            }
        }
        if n == 1 {
            return c;
        }
        n = n.div_ceil(2);
    }
}

/// How many of `seeds` schedules of `cfg` have a step whose log line
/// holds `what`: that the schedules reach the case a test is about.
fn reach(cfg: Config, seeds: u64, what: &str) -> usize {
    let mut rand = Rand(0x9E37_79B9_7F4A_7C15);
    (1..=seeds)
        .filter(|&seed| {
            let choices: Vec<usize> = (0..400).map(|_| rand.next(1 << 16)).collect();
            run(cfg, seed, &choices).0.iter().any(|l| l.contains(what))
        })
        .count()
}

fn report(name: &str, found: &(u64, Vec<usize>, Vec<String>, Violation)) {
    let (seed, choices, log, v) = found;
    eprintln!("{name}: seed {seed}, {} choices, {} steps: {v:?}", choices.len(), log.len());
    for (i, s) in log.iter().enumerate() {
        eprintln!("  {i:>3} {s}");
    }
}

/// The control: no remap (slots to spare), no client leaving, no prompt P
/// rejects. Every schedule ends cleanly, so what the other configurations
/// find is theirs.
#[test]
fn with_nothing_leaving_and_nothing_remapping_every_schedule_ends_cleanly() {
    let cfg = Config { requests: 6, slots: 10, slot: 24, seats: 3, leave: 0, long: false };
    if let Some(f) = explore(cfg, 300) {
        report("control", &f);
        panic!("the control violates {:?}", f.3);
    }
}

/// Review finding 1, closed: a D rank's placement is the connection's,
/// told in order ahead of the grant that needs it, so a grant released
/// (P rejects its prompt) or abandoned (its client leaves) takes nothing
/// with it, and every push writes through a mirror at its seats' epochs.
#[test]
fn a_placement_moved_on_d_reaches_every_mirror_before_a_push_needs_it() {
    for leave in [0, 3] {
        let cfg = Config { requests: 6, slots: 3, slot: 768, seats: 3, leave, long: true };
        if let Some(f) = explore(cfg, 1000) {
            report("placement", &f);
            panic!("leave {leave}: {:?}", f.3);
        }
        let reached = (reach(cfg, 200, "D→ Placed"), reach(cfg, 200, "pushes"), reach(cfg, 200, "at epochs [1"));
        eprintln!(
            "leave {leave}: of 200 schedules, {} place, {} push, {} push at epoch 1",
            reached.0, reached.1, reached.2
        );
        assert!(
            reached.0 > 20 && reached.2 > 20,
            "leave {leave}: (placed, pushed, pushed at epoch 1) in {reached:?} of 200"
        );
    }
}

/// Review finding 2: the head hands the request to the front before it
/// writes `Start`; the front, seeing the client gone, writes `Abort` on the
/// same connection in between, and D, finding no seat to start, exits.
#[test]
fn an_abort_from_the_front_overtakes_the_heads_start() {
    let cfg = Config { requests: 4, slots: 10, slot: 24, seats: 3, leave: 4, long: false };
    let f = explore(cfg, 3000).expect("the overtaking abort is found");
    report("abort overtakes start", &f);
    assert!(matches!(&f.3, Violation::DecodeDied(m) if m.contains("starts with no seat")), "{:?}", f.3);
}
