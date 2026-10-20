//! The pipeline head's side of SGLang's P/D: each request a bootstrap room
//! that every stage, as one of SGLang's prefill pp ranks, sends its own
//! layers into, and what follows from each thing the head hears.
//!
//! A request opens its room on every stage when it is admitted
//! ([`Out::Open`]), so each stage's sender meets the decode instance while
//! the prompt waits and runs. Once its last rows are issued the head hands
//! every stage the pages to send ([`Out::Transfer`]); each stage sends after
//! the item has landed on its device and says how the send ended. The last
//! stage also writes the first token into the room's metadata, from its
//! own answer to the item: no token travels from the head to a stage.
//!
//! A send is RDMA out of the stage's pages, off the compute stream, so
//! nothing on the device orders it before a later item that writes the
//! same pages. That is the difference from kern's own P/D (`pd`), where a
//! push is a copy on the stream and the head gives the lease to its cache
//! as soon as the push is issued. Here the lease is [`Rooms`]'s from the
//! moment its last rows are out until every stage has said how its send
//! ended, and only then goes back ([`Out::Release`]), with the reply to
//! the client ([`Out::Reply`]): SGLang answers a prefill request only once
//! its KV is on the decode side, and the decode instance starts only with
//! every pp rank's. Every room ends this way exactly once, sent or failed;
//! a client that leaves does not cut a send short.
//!
//! The held lease is `H`, opaque here: the head holds its lease and what
//! the cache needs to keep it, a test holds whatever it checks.

use std::collections::BTreeMap;

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

use crate::pd::Req;

/// What every stage sends for one request: the prompt's `len` positions in
/// `pages` and the state in sequence slot `slot` (none when the manifest
/// has no per-sequence state), into bootstrap room `room`, the request's
/// first token entry `at` of the last stage's answer to the item that
/// carried its last rows.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Transfer {
    pub req: Req,
    pub room: u64,
    pub pages: Vec<i32>,
    pub slot: Option<i32>,
    pub len: usize,
    pub at: usize,
}

/// What the head does next, in order.
#[derive(Debug, PartialEq, Eq)]
pub enum Out<H> {
    /// To every stage, the head among them: meet the decode side in `room`.
    Open { req: Req, room: u64 },
    /// To every stage, the head among them.
    Transfer(Transfer),
    /// To the client: its first token, and why the send failed if it did.
    Reply { req: Req, tok: u32, failure: Option<String> },
    /// The lease, no longer read by any send.
    Release { req: Req, held: H },
}

#[derive(Debug)]
struct Room<H> {
    room: u64,
    /// The lease from the transfer until the room ends.
    held: Option<H>,
    transferred: bool,
    first: Option<u32>,
    /// Every stage that said how its send ended: why it failed, if it did.
    sent: BTreeMap<usize, Option<String>>,
}

pub struct Rooms<H> {
    stages: usize,
    live: BTreeMap<Req, Room<H>>,
}

impl<H> Rooms<H> {
    /// Rooms sent into by `stages` stages, the head among them.
    pub fn new(stages: usize) -> Rooms<H> {
        Rooms { stages, live: BTreeMap::new() }
    }

    /// Requests whose room has not ended.
    pub fn live(&self) -> usize {
        self.live.len()
    }

    /// Leases held for sends in flight.
    pub fn held(&self) -> usize {
        self.live.values().filter(|r| r.held.is_some()).count()
    }

    /// `req` was admitted for `room`, which no live request has.
    pub fn open(&mut self, req: Req, room: u64) -> Result<Vec<Out<H>>> {
        if let Some((r, _)) = self.live.iter().find(|(_, x)| x.room == room) {
            bail!("request {req} opens room {room}, which request {r} still holds");
        }
        if self.live.contains_key(&req) {
            bail!("request {req} opens a second room");
        }
        let fresh = Room { room, held: None, transferred: false, first: None, sent: BTreeMap::new() };
        self.live.insert(req, fresh);
        Ok(vec![Out::Open { req, room }])
    }

    /// `req`'s last rows are issued from `held`, its `pages` holding `len`
    /// positions, its state in `slot`, and its token entry `at` of the
    /// item's answer: every stage sends them, and `held` stays here until
    /// every send ended.
    pub fn transfer(
        &mut self,
        req: Req,
        held: H,
        pages: Vec<i32>,
        slot: Option<i32>,
        len: usize,
        at: usize,
    ) -> Result<Vec<Out<H>>> {
        let Some(r) = self.live.get_mut(&req) else { bail!("request {req} is prefilled with no room open") };
        if r.transferred {
            bail!("request {req} is prefilled twice");
        }
        r.transferred = true;
        r.held = Some(held);
        Ok(vec![Out::Transfer(Transfer { req, room: r.room, pages, slot, len, at })])
    }

    /// The first token of `req`, from the last stage's answer.
    pub fn first(&mut self, req: Req, tok: u32) -> Result<Vec<Out<H>>> {
        let Some(r) = self.live.get_mut(&req).filter(|r| r.transferred) else {
            bail!("a first token for request {req}, which is not prefilled");
        };
        if r.first.replace(tok).is_some() {
            bail!("a second first token for request {req}");
        }
        Ok(self.settle(req))
    }

    /// Stage `stage`'s send of `req` ended, failed for `failure` if it did.
    pub fn sent(&mut self, req: Req, stage: usize, failure: Option<String>) -> Result<Vec<Out<H>>> {
        let Some(r) = self.live.get_mut(&req).filter(|r| r.transferred) else {
            bail!("stage {stage} ended a send of request {req}, which is not sending");
        };
        if stage >= self.stages || r.sent.contains_key(&stage) {
            bail!("stage {stage} ends the send of request {req} twice, or is not one of {}", self.stages);
        }
        r.sent.insert(stage, failure);
        Ok(self.settle(req))
    }

    /// A room every stage ended and whose first token is in: its reply,
    /// then its lease back.
    fn settle(&mut self, req: Req) -> Vec<Out<H>> {
        let r = &self.live[&req];
        let Some(tok) = r.first.filter(|_| r.sent.len() == self.stages) else { return Vec::new() };
        let mut r = self.live.remove(&req).expect("looked up");
        let failures: Vec<String> =
            r.sent.iter().filter_map(|(s, f)| f.as_ref().map(|f| format!("stage {s}: {f}"))).collect();
        let failure = (!failures.is_empty()).then(|| failures.join("; "));
        let held = r.held.take().expect("a prefilled room holds its lease");
        vec![Out::Reply { req, tok, failure }, Out::Release { req, held }]
    }
}
