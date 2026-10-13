//! A deterministic simulation of one prefill pipeline handing requests to
//! one decode tier, built to find what the P/D protocol lets go wrong
//! (docs/protocol.md).
//!
//! Every thread of the deployment is an actor and every connection a FIFO:
//! the P head (the real [`Tiers`], carrying out each [`Out`] as a step of
//! its own, as `apply` does), the front (the real [`Clients`], its writes
//! carried out against the clients and the head), the D tier's reader and
//! scheduler (a real [`Decoder`] over real pools), the reader of D's
//! connection on P, the stages pushing into their mirrors of D's pools,
//! the last stage's first tokens, and the clients, who may leave at any
//! step. A seeded schedule picks which enabled actor moves next. What is
//! modelled is only what the head's admission does around [`Tiers`] with
//! P's own pool and runtime: a prompt longer than [`P_LIMIT`] is rejected,
//! any other leased.
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
use kern_serve::front::{Clients, Note, Write};
use kern_serve::pd::{Ask, Decoder, Gate, Out, Push, Refusal, Req, Run, Seat, Tell, Tiers};
use kern_serve::ranks::{Act, Answer, Order};
use pegainfer_frontend::engine::{EosPolicy, FinishReason, RejectReason, RequestId, StopPolicy};

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
    Pushed {
        stage: usize,
        req: Req,
    },
    /// The last stage's token for `req`, issued handed off or not.
    First {
        req: Req,
        handoff: bool,
    },
    Gone(Req),
    /// The front ended a request decoding on `tier`: free its seats.
    Abort {
        tier: usize,
        req: Req,
    },
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
    /// A client leaves: of those still there, the one the step count picks,
    /// so departures spread over every phase rather than crowd the start.
    Leave,
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
    tiers: Tiers,
    outbox: VecDeque<(usize, Out)>,
    events: VecDeque<Event>,
    waiting: VecDeque<Req>,
    /// Admitted, last rows not yet issued: the lease and whether it hands off.
    admitted: BTreeMap<Req, (kern_pool::Lease, bool)>,
    gone: BTreeSet<Req>,
    /// Something was heard since the last turn: the head's loop admits
    /// after every event, and otherwise blocks on its channel.
    turn: bool,
}

struct Front {
    c: Clients,
    notes: VecDeque<Note>,
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
    /// Requests whose last rows are issued and whose first token is not
    /// back, and whether each is handed off.
    firsts: BTreeMap<Req, bool>,
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
                tiers: Tiers::new(1, STAGES),
                outbox: VecDeque::new(),
                events: VecDeque::new(),
                waiting: clients.keys().copied().collect(),
                admitted: BTreeMap::new(),
                gone: BTreeSet::new(),
                turn: true,
            },
            front: Front {
                c: clients.keys().fold(Clients::new(0), |mut c, &r| {
                    c.submit(RequestId::new(r));
                    c
                }),
                notes: VecDeque::new(),
            },
            clients,
            d,
            to_d: VecDeque::new(),
            to_p: VecDeque::new(),
            stages: (0..STAGES).map(|_| Stage { inbox: VecDeque::new(), mirror: mirror.clone() }).collect(),
            firsts: BTreeMap::new(),
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
                if self.head.turn {
                    v.push(Step::HeadTurn);
                }
                v.extend(self.head.admitted.keys().map(|&r| Step::HeadPrefill(r)));
            }
        }
        if !self.front.notes.is_empty() {
            v.push(Step::FrontNote);
        }
        if self.front.c.clone().sweep(|id| self.clients[&id.raw()].gone).is_ok_and(|w| !w.is_empty()) {
            v.push(Step::FrontSweep);
        }
        if self.left > 0 && self.clients.values().any(|c| c.end.is_none() && !c.gone) {
            v.push(Step::Leave);
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
        v.extend(self.firsts.keys().map(|&r| Step::Last(r)));
        v
    }

    fn step(&mut self, s: Step) -> Checked<()> {
        self.log.push(format!("{s:?}"));
        match s {
            Step::HeadOut => {
                let o = self.head.outbox.pop_front().expect("enabled");
                if !matches!(o.1, Out::Ask(_) | Out::Push(_)) {
                    self.say(format!("{o:?}"));
                }
                self.carry(o)
            }
            Step::HeadHear => {
                let e = self.head.events.pop_front().expect("enabled");
                self.head.turn = true;
                self.hear(e)
            }
            Step::HeadTurn => {
                self.head.turn = false;
                self.retire_gone()?;
                self.admit()
            }
            Step::HeadPrefill(req) => {
                let (lease, handoff) = self.head.admitted.remove(&req).expect("enabled");
                self.say(format!("prompt {}", self.clients[&req].prompt));
                if handoff {
                    let outs = self.head.tiers.prefilled(req, &lease, self.clients[&req].prompt);
                    self.head.outbox.extend(outs);
                }
                // The head's lease goes to its cache: P's pages are not checked here.
                drop(lease);
                self.firsts.insert(req, handoff);
                Ok(())
            }
            Step::FrontNote => {
                let n = self.front.notes.pop_front().expect("enabled");
                if !matches!(n, Note::Tokens(_)) {
                    self.say(format!("{n:?}"));
                }
                let w = self.front.c.note(n).map_err(|e| Violation::PrefillDied(format!("{e:#}")))?;
                self.write(w)
            }
            Step::FrontSweep => {
                let clients = &self.clients;
                let w = self.front.c.sweep(|id| clients[&id.raw()].gone);
                let w = w.map_err(|e| Violation::PrefillDied(format!("{e:#}")))?;
                self.write(w)
            }
            Step::Leave => {
                let live: Vec<Req> =
                    self.clients.iter().filter(|(_, c)| c.end.is_none() && !c.gone).map(|(&r, _)| r).collect();
                let req = live[self.log.len() % live.len()];
                self.say(format!("{req}"));
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
                let handoff = self.firsts.remove(&req).expect("enabled");
                self.head.events.push_back(Event::First { req, handoff });
                Ok(())
            }
        }
    }

    /// The head carries out one decision, as `PpHead::apply`.
    fn carry(&mut self, (tier, o): (usize, Out)) -> Checked<()> {
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
            Out::Emit { req, tok } => Note::Emit { id: RequestId::new(req), tok },
            Out::Finish { req, why, cause } => {
                self.head.gone.remove(&req);
                Note::Finish { id: RequestId::new(req), why, cause }
            }
            Out::Decode { req, relay } => {
                self.head.gone.remove(&req);
                Note::Decode { id: RequestId::new(req), relay, tier }
            }
            Out::Reject { req, why } => {
                let i = self.head.waiting.iter().position(|&r| r == req);
                let i =
                    i.ok_or_else(|| Violation::PrefillDied(format!("a refused request {req} no longer waiting")))?;
                self.head.waiting.remove(i);
                self.head.gone.remove(&req);
                let c = &self.clients[&req];
                let why = match why {
                    Refusal::ContextLength { limit } => {
                        RejectReason::ContextLength { prompt_tokens: c.prompt, max_tokens: c.max_tokens, limit }
                    }
                    Refusal::KvBudget => {
                        RejectReason::KvBudget { prompt_tokens: c.prompt, worst_case_tokens: c.prompt + c.max_tokens }
                    }
                };
                Note::Reject { id: RequestId::new(req), why }
            }
        };
        self.front.notes.push_back(n);
        Ok(())
    }

    /// An ask onto the one connection to D.
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
            Event::Tell(t) => Self::died(self.head.tiers.hear(0, t))?,
            Event::Pushed { stage, req } => Self::died(self.head.tiers.pushed(req, stage))?,
            Event::First { req, handoff: true } => Self::died(self.head.tiers.first(req, first_token(req)))?,
            Event::First { req, handoff: false } => {
                let id = RequestId::new(req);
                let tok = first_token(req);
                let why = if tok == STOP { FinishReason::Stop } else { FinishReason::Length };
                match self.head.gone.remove(&req) {
                    true => self.front.notes.push_back(Note::Retire(id)),
                    false => self.front.notes.extend([Note::Emit { id, tok }, Note::Finish { id, why, cause: None }]),
                }
                Vec::new()
            }
            Event::Abort { tier, req } => vec![Self::died(self.head.tiers.stop(tier, req))?],
            Event::Gone(req) => {
                let holds = self.head.waiting.contains(&req)
                    || self.head.admitted.contains_key(&req)
                    || self.firsts.contains_key(&req)
                    || self.head.tiers.open().contains(&req);
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
        let gone: Vec<Req> = self.head.tiers.open().into_iter().filter(|r| self.head.gone.contains(r)).collect();
        for req in gone {
            self.head.gone.remove(&req);
            self.front.notes.push_back(Note::Retire(RequestId::new(req)));
            let outs = self.head.tiers.abort(req);
            self.head.outbox.extend(outs);
        }
        Ok(())
    }

    /// As `PpHead::admit`: P rejects a prompt past [`P_LIMIT`] and leases any other.
    fn admit(&mut self) -> Checked<()> {
        while let Some(&req) = self.head.waiting.front() {
            let id = RequestId::new(req);
            if self.head.gone.remove(&req) {
                self.front.notes.push_back(Note::Retire(id));
                self.head.waiting.pop_front();
                let outs = self.head.tiers.abort(req);
                self.head.outbox.extend(outs);
                continue;
            }
            let (prompt, max_tokens) = (self.clients[&req].prompt, self.clients[&req].max_tokens);
            let (gate, outs) = Self::died(self.head.tiers.gate(req, prompt + max_tokens, max_tokens, &stop()))?;
            self.head.outbox.extend(outs);
            let Gate::Lease(tier) = gate else { return Ok(()) };
            self.head.waiting.pop_front();
            if prompt > P_LIMIT {
                let why = RejectReason::ContextLength { prompt_tokens: prompt, max_tokens, limit: P_LIMIT };
                self.front.notes.push_back(Note::Reject { id, why });
                if let Some(t) = tier {
                    let outs = self.head.tiers.release(t, req);
                    self.head.outbox.extend(outs);
                }
                continue;
            }
            let lease = self.p_pool.lease(prompt).expect("P's pool holds every prompt");
            self.front.notes.push_back(Note::Admit { id, cached: 0 });
            if let Some(t) = tier {
                self.head.tiers.admitted(t, req);
            }
            self.head.admitted.insert(req, (lease, tier.is_some()));
        }
        Ok(())
    }

    /// The front's writes carried out, as `Front::carry`: onto the clients'
    /// ledger, or to the head.
    fn write(&mut self, writes: Vec<Write>) -> Checked<()> {
        for w in writes {
            match w {
                Write::Admit { .. } => {}
                Write::Reject { id, .. } => self.end(id.raw(), End::Rejected)?,
                Write::Tokens { id, tok } => self.emit(id.raw(), tok)?,
                Write::Finish { id, why, .. } => {
                    let t = self.clients[&id.raw()].tokens.clone();
                    self.end(id.raw(), End::Finished(t, why))?;
                }
                Write::Retire(id) => self.end(id.raw(), End::Retired)?,
                Write::Gone(id) => self.head.events.push_back(Event::Gone(id.raw())),
                Write::Abort { tier, req } => {
                    self.say(format!("abort {req} through the head"));
                    self.head.events.push_back(Event::Abort { tier, req });
                }
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
        let held = (self.head.tiers.live(), self.front.c.open(), self.front.c.decoding(), self.d.seated());
        match held {
            (0, 0, 0, ref s) if s.is_empty() => Ok(()),
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
    for (i, s) in log.iter().enumerate().skip(log.len().saturating_sub(200)) {
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

/// Review finding 2, closed: the head is the one writer to D and the
/// front's aborts go through it, so the abort of a request the head handed
/// to the front lands behind the head's `Start`, never ahead of it.
#[test]
fn an_abort_from_the_front_comes_behind_the_heads_start() {
    let cfg = Config { requests: 4, slots: 10, slot: 24, seats: 3, leave: 4, long: false };
    if let Some(f) = explore(cfg, 3000) {
        report("abort and start", &f);
        panic!("{:?}", f.3);
    }
    let reached = reach(cfg, 3000, "FrontSweep | abort");
    eprintln!("of 3000 schedules, {reached} have the front abort a decoding request");
    assert!(reached > 20, "{reached} of 3000");
}
