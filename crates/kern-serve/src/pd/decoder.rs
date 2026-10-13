//! The D head's half of the protocol: seats, and the decode loop.
//!
//! A `Reserve` adopts a row ([`Ranks::adopt`]: a lease on every member,
//! nothing zeroed, since the push writes the slot whole and a zero landing
//! after it would clear it) and grants it, each seat naming its rank's
//! placement epoch; a seat's rank whose placement moved since P was last
//! told is told first (`Placed`). A reserve that finds no room is `Busy`
//! and owes a `Room` once a seat is freed or a remap lands.
//! A seat is freed only by `Abort` or by finishing, and P sends the first
//! only while nothing pushes into it (`handoff.rs`), so a seat is never
//! leased again under a push. A `Start` moves a seat into the running batch
//! at the next step, fed its first token at the prompt's length; every step
//! runs every running row one token (`Ranks::stage`, the forward the
//! protocol picks for the batch) and tells P each one's token. D keeps
//! nothing and judges no stop but `max_tokens`: the row is dropped when it
//! ends, and P aborts what its stop policy ends first.

use std::collections::BTreeMap;

use anyhow::{bail, Context, Result};
use kern_manifest::protocol::Rows;
use kern_pool::{Denied, Placement};

use super::{Ask, Load, Pools, Refusal, Req, Seat, Shape, Tell};
use crate::ranks::{Answer, Cell, Order, Ranks, Row};
use crate::scheduler::bucket;

/// What runs the ranks' orders: the ranks' processes, or a fake.
pub trait Run {
    /// Every rank's order in rank order, every rank's answer in rank order.
    fn run(&mut self, orders: Vec<Order>) -> Result<Vec<Answer>>;
}

struct Seq {
    req: Req,
    pos: usize,
    next: u32,
    left: usize,
    row: Row,
}

pub struct Decoder<E> {
    ranks: Ranks,
    exec: E,
    /// Seats a group holds at once: reserved and running.
    cap: usize,
    reserved: BTreeMap<Req, (Row, usize)>,
    running: Vec<Seq>,
    /// A `Busy` was told and no `Room` since.
    owed: bool,
    /// Each rank's placement as P was last told, and its epoch.
    told: Vec<(u64, Placement)>,
}

impl<E: Run> Decoder<E> {
    /// Decode over `ranks` through `exec`, at most `max_seqs` seats per
    /// group (within what the protocol's one-row step takes).
    pub fn new(ranks: Ranks, exec: E, max_seqs: usize) -> Result<Decoder<E>> {
        let p = ranks.protocol();
        if p.forward(1, Rows::Const(1)).is_none() {
            bail!("no program takes one sequence of one row: nothing to decode with");
        }
        let cap = max_seqs.clamp(1, ranks.seqs_max()).min(p.max_groups(Rows::Const(1)) as usize).max(1);
        let told = (0..ranks.groups().ranks()).map(|q| (0, ranks.placement(q))).collect();
        Ok(Decoder { ranks, exec, cap, reserved: BTreeMap::new(), running: Vec::new(), owed: false, told })
    }

    /// What runs the orders.
    pub fn exec(&self) -> &E {
        &self.exec
    }

    /// Rows in the running batch.
    pub fn running(&self) -> usize {
        self.running.len()
    }

    /// Seats a group holds at once.
    pub fn cap(&self) -> usize {
        self.cap
    }

    /// The requests holding seats, reserved or running, in request order.
    pub fn seated(&self) -> Vec<Req> {
        let mut v: Vec<Req> = self.reserved.keys().copied().chain(self.running.iter().map(|s| s.req)).collect();
        v.sort_unstable();
        v
    }

    /// Seats held and pages held, every rank together.
    pub fn load(&self) -> Load {
        let seats = self.reserved.len() + self.running.len();
        Load { seats, pages: self.ranks.pages_used(), pages_total: self.ranks.pages_total() }
    }

    /// Whether a step has work: rows to run or acts to land.
    pub fn busy(&self) -> bool {
        !self.running.is_empty() || self.ranks.outstanding()
    }

    /// What a prefill head is told on connecting: the pools behind
    /// `handles` (every rank's `PoolHandle`, hex) holding `states` (the
    /// manifest's [`shapes`](super::shapes)), their placements at epoch 0,
    /// and the states' shards.
    pub fn pools(&self, handles: Vec<String>, states: BTreeMap<String, Shape>) -> Pools {
        let (r, p) = (&self.ranks, self.ranks.protocol());
        Pools {
            handles,
            placements: self.told.iter().map(|(_, p)| hex::encode(p.to_bytes())).collect(),
            states,
            group: r.groups().size(),
            dealt: r.interleave().is_some(),
            heads: p.shards.heads.iter().map(|(n, h)| (n.clone(), h.segments.clone())).collect(),
            page: r.page() as u64,
            max_seq_tokens: r.max_seq_tokens(),
        }
    }

    /// Seats held per rank: a group's on its first member.
    fn held(&self) -> Vec<usize> {
        let mut n = vec![0; self.ranks.groups().ranks()];
        let rows = self.reserved.values().map(|(r, _)| r).chain(self.running.iter().map(|s| &s.row));
        for r in rows {
            n[r.owner().index()] += 1;
        }
        n
    }

    /// What P asked, answered.
    pub fn ask(&mut self, a: Ask) -> Result<Vec<Tell>> {
        match a {
            Ask::Reserve { req, tokens } => self.reserve(req, tokens),
            Ask::Start { req, first, len, max_tokens } => {
                let &(_, tokens) =
                    self.reserved.get(&req).with_context(|| format!("request {req} starts with no seat"))?;
                if max_tokens == 0 || len + max_tokens > tokens {
                    bail!("request {req} starts at {len} for {max_tokens} tokens in a seat of {tokens}");
                }
                let (row, _) = self.reserved.remove(&req).expect("looked up");
                self.running.push(Seq { req, pos: len, next: first, left: max_tokens, row });
                Ok(Vec::new())
            }
            Ask::Abort { req } => {
                // Finished here while P's abort was on its way: nothing is held.
                let before = self.reserved.len() + self.running.len();
                self.reserved.remove(&req);
                self.running.retain(|s| s.req != req);
                Ok(self.room(before != self.reserved.len() + self.running.len()))
            }
        }
    }

    fn reserve(&mut self, req: Req, tokens: usize) -> Result<Vec<Tell>> {
        if self.reserved.contains_key(&req) || self.running.iter().any(|s| s.req == req) {
            bail!("request {req} reserved twice");
        }
        let (held, cap) = (self.held(), self.cap);
        let got = self.ranks.adopt(tokens, |q| Some(held[q.index()]).filter(|&n| n < cap));
        let row = match got {
            Ok(row) => row,
            Err(Denied::Busy | Denied::Remapping) => {
                self.owed = true;
                return Ok(vec![Tell::Busy { req }]);
            }
            Err(Denied::ExceedsRow { limit }) => {
                return Ok(vec![Tell::Refuse { req, why: Refusal::ContextLength { limit } }]);
            }
            Err(Denied::ExceedsPool) => return Ok(vec![Tell::Refuse { req, why: Refusal::KvBudget }]),
            Err(d @ Denied::HostFull) => bail!("an adopt denied {d}"),
        };
        let first = self.ranks.groups().members(row.owner().index()).start;
        let mut tells = Vec::new();
        let mut seats = Vec::with_capacity(row.leases().len());
        for (m, l) in row.leases().iter().enumerate() {
            let q = first + m;
            let now = self.ranks.placement(q);
            if now != self.told[q].1 {
                let epoch = self.told[q].0 + 1;
                tells.push(Tell::Placed { rank: q, epoch, placement: hex::encode(now.to_bytes()) });
                self.told[q] = (epoch, now);
            }
            seats.push(Seat { rank: q, pages: l.page_ids().to_vec(), slot: l.seq_slot(), epoch: self.told[q].0 });
        }
        self.reserved.insert(req, (row, tokens));
        tells.push(Tell::Grant { req, seats, load: self.load() });
        Ok(tells)
    }

    /// A `Room` owed and room made.
    fn room(&mut self, made: bool) -> Vec<Tell> {
        match made && self.owed {
            true => {
                self.owed = false;
                vec![Tell::Room]
            }
            false => Vec::new(),
        }
    }

    /// One step: every running row one token, or with none running, the
    /// ranks polled for what is still landing.
    pub fn step(&mut self) -> Result<Vec<Tell>> {
        let was = self.ranks.busy();
        if self.running.is_empty() {
            if self.ranks.outstanding() {
                let orders = self.ranks.orders(None, false);
                let answers = self.exec.run(orders)?;
                self.ranks.heard(&answers)?;
            }
            let landed = was && !self.ranks.busy();
            return Ok(self.room(landed));
        }
        let t0 = std::time::Instant::now();
        let cells: Vec<Cell> =
            self.running.iter().map(|s| Cell { row: &s.row, ids: vec![s.next as i64], pos: s.pos }).collect();
        let cap = self.cap;
        let mut st = self.ranks.stage(&cells, 1, |k| bucket(k).min(cap))?;
        drop(cells);
        let p = self.ranks.protocol();
        let f = st.forward(p, 1).context("no forward for this batch of one-row sequences")?;
        let orders = self.ranks.orders(Some(st.items(&f)), false);
        let staged_us = t0.elapsed().as_micros() as u64;
        let answers = self.exec.run(orders)?;
        let ran_us = t0.elapsed().as_micros() as u64;
        self.ranks.heard(&answers)?;
        let out = st.emitted(self.ranks.protocol(), &f, &answers)?;
        tracing::debug!(
            rows = self.running.len(),
            staged_us,
            ran_us,
            heard_us = t0.elapsed().as_micros() as u64,
            "decode step"
        );
        let mut step = Vec::with_capacity(self.running.len());
        for (s, toks) in self.running.iter_mut().zip(&out) {
            let &[tok] = toks.as_slice() else {
                bail!("request {}: the step handed back {} tokens, expected one", s.req, toks.len());
            };
            s.pos += 1;
            s.next = tok as u32;
            s.left -= 1;
            step.push((s.req, tok as u32, s.left == 0));
        }
        let before = self.running.len();
        self.running.retain(|s| s.left > 0);
        let made = self.running.len() < before || (was && !self.ranks.busy());
        let mut tells = vec![Tell::Tokens { step, load: self.load() }];
        tells.extend(self.room(made));
        Ok(tells)
    }
}
