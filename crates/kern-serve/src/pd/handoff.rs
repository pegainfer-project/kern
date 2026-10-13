//! The P head's half of the protocol: where each handed-off request
//! stands until it decodes, and what follows from each thing heard.
//!
//! A request moves through three phases here: `Reserving` (a `Reserve`
//! sent, no answer yet), `Granted` (seats held on D, the prompt waiting for
//! or in P's prefill) and `Pushing` (every stage told to push, counting
//! their `Pushed` and waiting for the first token). Once it starts on D it
//! leaves, its [`Relay`] handed to the front ([`Out::Decode`], before the
//! `Start`), which relays D's tokens without waiting on the head. Two rules
//! make it safe, and both are this type's, so no caller can break them:
//!
//! - an `Abort` goes out only while nothing pushes into the seats: in
//!   `Granted`, or in `Pushing` once every stage said `Pushed`. A client
//!   that goes away mid-push, or a first token that ends the request, waits
//!   for the last `Pushed`.
//! - a push is only made from a live lease ([`Handoffs::prefilled`] takes
//!   one), so the caller cannot have given its pages back before the push
//!   is on the stages' streams.
//!
//! One `Reserve` is in flight at a time, the waiting queue's front's, so
//! seats are granted in arrival order. `Busy` drops the request back to
//! not asked and stops all asking until D says `Room`.
//!
//! Who answers the client: the waiting queue until the request is
//! admitted on P, these handoffs after, until they finish it, the caller
//! reports the client gone ([`Handoffs::abort`]) or it starts decoding.
//! Everything it wants done comes back as [`Out`]s, in order.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{bail, Result};
use kern_pool::Lease;
use pegainfer_frontend::engine::{FinishReason, StopCause};

use super::{Ask, Push, Refusal, Relay, Req, Seat, Tell};

/// What the P head does next, in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Out {
    Ask(Ask),
    /// To every stage, the head among them.
    Push(Push),
    Emit {
        req: Req,
        tok: u32,
    },
    Finish {
        req: Req,
        why: FinishReason,
        cause: Option<StopCause>,
    },
    /// Decode tier `tier`'s rank `rank` is at placement `epoch` (hex): the
    /// head's mirror and every stage's, ahead of any push behind it.
    Place {
        tier: usize,
        rank: usize,
        epoch: u64,
        placement: String,
    },
    /// The waiting front's reserve was refused: reject it.
    Reject {
        req: Req,
        why: Refusal,
    },
    /// `req` starts on D: its tokens are the front's to relay from here on.
    Decode {
        req: Req,
        relay: Relay,
    },
}

#[derive(Debug)]
enum Phase {
    Reserving,
    Granted { seats: Vec<Seat> },
    Pushing { pushed: BTreeSet<usize>, first: Option<u32>, len: usize },
}

#[derive(Debug)]
struct Handoff {
    phase: Phase,
    relay: Relay,
    /// Admitted on P: these handoffs answer the client.
    admitted: bool,
    /// Finished or retired: nothing more goes to the client.
    closed: bool,
}

/// Where a request stands, as the P head's admission sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Standing {
    /// Never asked, or asked and told `Busy`.
    Unasked,
    Reserving,
    Granted,
    /// Past its grant: admitted or prefilled.
    Underway,
}

pub struct Handoffs {
    stages: usize,
    tier: usize,
    asking: Option<Req>,
    blocked: bool,
    live: BTreeMap<Req, Handoff>,
}

impl Handoffs {
    /// Handoffs to decode tier `tier`, pushed by `stages` stages.
    pub fn new(stages: usize, tier: usize) -> Handoffs {
        Handoffs { stages, tier, asking: None, blocked: false, live: BTreeMap::new() }
    }

    /// Requests D holds seats for or is asked about.
    pub fn live(&self) -> usize {
        self.live.len()
    }

    /// Requests whose client these handoffs answer: admitted, not yet
    /// finished or retired.
    pub fn open(&self) -> Vec<Req> {
        self.live.iter().filter(|(_, h)| h.admitted && !h.closed).map(|(&r, _)| r).collect()
    }

    pub fn standing(&self, req: Req) -> Standing {
        match self.live.get(&req).map(|h| &h.phase) {
            None => Standing::Unasked,
            Some(Phase::Reserving) => Standing::Reserving,
            Some(Phase::Granted { .. }) if !self.live[&req].admitted => Standing::Granted,
            Some(_) => Standing::Underway,
        }
    }

    /// Whether a reserve would go out now: none unanswered, D not busy.
    pub fn ready(&self) -> bool {
        self.asking.is_none() && !self.blocked
    }

    /// Ask seats for `req`, a sequence of up to `tokens`; `None` while
    /// another reserve is unanswered, D is busy, or `req` was asked.
    pub fn reserve(&mut self, req: Req, tokens: usize, relay: Relay) -> Option<Ask> {
        if !self.ready() || self.live.contains_key(&req) {
            return None;
        }
        self.asking = Some(req);
        self.live.insert(req, Handoff { phase: Phase::Reserving, relay, admitted: false, closed: false });
        Some(Ask::Reserve { req, tokens })
    }

    /// The granted `req` is admitted on P; its client is these handoffs' now.
    pub fn admitted(&mut self, req: Req) {
        if let Some(h) = self.live.get_mut(&req) {
            h.admitted = true;
        }
    }

    /// P will never seat the granted `req` (or its client left while it
    /// waited): its seats go back. Nothing is pushing into them.
    pub fn release(&mut self, req: Req) -> Vec<Out> {
        match self.live.get(&req).map(|h| &h.phase) {
            Some(Phase::Granted { .. }) => {
                self.live.remove(&req);
                vec![Out::Ask(Ask::Abort { req })]
            }
            _ => Vec::new(),
        }
    }

    /// The client of `req` went away (the caller retired it): nothing more
    /// goes to it, and its seats are freed as soon as nothing pushes into
    /// them.
    pub fn abort(&mut self, req: Req) -> Vec<Out> {
        let Some(h) = self.live.get_mut(&req) else { return Vec::new() };
        h.closed = true;
        match &h.phase {
            Phase::Reserving => Vec::new(),
            Phase::Granted { .. } => {
                self.live.remove(&req);
                vec![Out::Ask(Ask::Abort { req })]
            }
            Phase::Pushing { .. } => self.settle(req),
        }
    }

    /// The prompt's last rows of `req` are on their way through `lease`,
    /// `len` positions: the push to every stage, unless its client left.
    pub fn prefilled(&mut self, req: Req, lease: &Lease, len: usize) -> Vec<Out> {
        let Some(h) = self.live.get_mut(&req) else { return Vec::new() };
        let Phase::Granted { seats } =
            std::mem::replace(&mut h.phase, Phase::Pushing { pushed: BTreeSet::new(), first: None, len })
        else {
            unreachable!("a request is prefilled once, after its grant")
        };
        let (pages, slot, tier) = (lease.page_ids().to_vec(), lease.seq_slot(), self.tier);
        let push = Push { req, tier, pages, slot, len, seats };
        vec![Out::Push(push)]
    }

    /// Stage `stage`'s push for `req` landed.
    pub fn pushed(&mut self, req: Req, stage: usize) -> Result<Vec<Out>> {
        let Some(Handoff { phase: Phase::Pushing { pushed, .. }, .. }) = self.live.get_mut(&req) else {
            bail!("stage {stage} reports a push of request {req}, which is not pushing");
        };
        if stage >= self.stages || !pushed.insert(stage) {
            bail!("stage {stage} reports the push of request {req} twice, or is not one of {}", self.stages);
        }
        Ok(self.settle(req))
    }

    /// The first token of `req`, from the prefill: the client's first,
    /// unless it left.
    pub fn first(&mut self, req: Req, tok: u32) -> Result<Vec<Out>> {
        let Some(h) = self.live.get_mut(&req) else { return Ok(Vec::new()) };
        let Phase::Pushing { first, .. } = &mut h.phase else {
            bail!("the first token of request {req} came back before its push");
        };
        *first = Some(tok);
        let mut out = Vec::new();
        if !h.closed {
            out.push(Out::Emit { req, tok });
            if let Some((why, cause)) = h.relay.take(tok) {
                h.closed = true;
                out.push(Out::Finish { req, why, cause });
            }
        }
        out.extend(self.settle(req));
        Ok(out)
    }

    /// A pushing `req` whose stages all pushed and whose first token is in
    /// starts on D, its relay handed over first, or frees its seats when it
    /// is already over.
    fn settle(&mut self, req: Req) -> Vec<Out> {
        let h = &self.live[&req];
        let Phase::Pushing { pushed, first: Some(first), len } = &h.phase else { return Vec::new() };
        if pushed.len() < self.stages {
            return Vec::new();
        }
        if h.closed {
            self.live.remove(&req);
            return vec![Out::Ask(Ask::Abort { req })];
        }
        let start = Ask::Start { req, first: *first, len: *len, max_tokens: h.relay.max_tokens - h.relay.generated };
        let relay = self.live.remove(&req).expect("looked up").relay;
        vec![Out::Decode { req, relay }, Out::Ask(start)]
    }

    /// What D said.
    pub fn hear(&mut self, t: Tell) -> Result<Vec<Out>> {
        let answered = |req: Req, asking: &mut Option<Req>| -> Result<()> {
            match asking.take() {
                Some(r) if r == req => Ok(()),
                other => bail!("the decode tier answered a reserve of request {req}; the one asked is {other:?}"),
            }
        };
        match t {
            Tell::Pools(_) => bail!("the decode tier sent its pools twice"),
            Tell::Placed { rank, epoch, placement } => Ok(vec![Out::Place { tier: self.tier, rank, epoch, placement }]),
            Tell::Grant { req, seats, .. } => {
                answered(req, &mut self.asking)?;
                let h = self.live.get_mut(&req).expect("asking is live");
                h.phase = Phase::Granted { seats };
                Ok(match h.closed {
                    true => self.release(req),
                    false => Vec::new(),
                })
            }
            Tell::Busy { req } => {
                answered(req, &mut self.asking)?;
                self.live.remove(&req);
                self.blocked = true;
                Ok(Vec::new())
            }
            Tell::Room => {
                self.blocked = false;
                Ok(Vec::new())
            }
            Tell::Refuse { req, why } => {
                answered(req, &mut self.asking)?;
                let h = self.live.remove(&req).expect("asking is live");
                Ok(match h.closed {
                    true => Vec::new(),
                    false => vec![Out::Reject { req, why }],
                })
            }
            Tell::Tokens { .. } => bail!("the decode tier's tokens go to the front's relays, not the handoffs"),
        }
    }
}
