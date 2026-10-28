//! The head's accounting of every rank: one [`kern_pool::Pool`] per GPU,
//! decided here, run by the ranks' own processes.
//!
//! Every GPU is one process. The head holds what each rank's runtime would
//! otherwise decide for itself — its pages and slots — in a pool built
//! from the rank's own budget, so the pool here starts where the rank's
//! did and makes the decisions it would make. Nothing in this module
//! touches a device or a socket: a decision comes back as [`Act`]s queued
//! on the rank in the order they were made, and [`Ranks::orders`] hands
//! each rank its queue with the step's [`Item`]. The rank runs them in
//! that order on its compute stream ahead of the item, so a page freed
//! here and leased again is written there in the same order, and answers
//! with the item's outputs ([`Answer`]).
//!
//! # Lockstep
//!
//! A manifest with a `topology` is SPMD — every rank runs the same launch
//! list, and a collective inside it waits for every other rank — so the
//! ranks must be given the same program, the same var values and the same
//! number of rows at every step, or one of them spins on an epoch that
//! never comes. [`Ranks::stage`] is the only place a step's per-rank
//! inputs are made, every one of them from one [`Layout`] over the whole
//! batch, so the invariant is a property of the type layer rather than of
//! the scheduler's discipline.
//!
//! # Rows and their owners
//!
//! The `tp` group (`t` ranks; 1 without one) runs one batch: its rows are
//! spread over the group's ranks, every rank holding a slice of every
//! row's recurrent state (a head-sharded KDA state) and the pages of the
//! rows it owns (MLA attention is by row: the owner alone reads them). A
//! [`Row`] is therefore one paged [`Lease`] on its owner and a slot-only
//! lease on each of the owner's peers, and every act on it — lease,
//! checkpoint, retire — is the same act on every member of the group, all
//! or nothing: a `Row` is only ever constructed once every member's pool
//! has said yes, and an early return drops what the earlier members handed
//! out, before any act is queued. The pools decide independently (slot
//! numbers differ across ranks and are never compared). An owner is chosen
//! when a row is leased — the open rank with the fewest rows, then the
//! fewest pages in use (pages count the snapshots kept there, so a rank
//! holding a long prefix would otherwise sit idle while the others
//! prefill) — and is fixed for the row's life and everything that descends
//! from it: pages live on that GPU.
//!
//! # Staging
//!
//! A step is a list of [`Cell`]s, one per row: the row, the tokens it
//! feeds this step and its position. Rows per rank `b` is one bucket for
//! every rank (the most rows any rank has, rounded up), padded on every
//! rank with the rank's pad lease — a page and a slot no sequence owns,
//! whose junk nobody reads. Which buffer carries what, and which inputs
//! span the group and which are a rank's own, is the manifest's
//! [`Protocol`]: a fill over the ranks axis carries the group batch — this
//! rank's rows first, then the other members' in group order from it, the
//! layout the collectives assume — and one over the rows or sequences axis
//! carries this rank's rows alone. The tokens fed and the line tables of a
//! sharded state are of the first kind in a group manifest; the slots, the
//! sequence lengths and the page tables are always of the second. Outputs
//! likewise: a `tokens` fill over the ranks axis holds this rank's rows in
//! its first block, so every cell is read from its owner's answer.
//!
//! # Replicated groups
//!
//! A protocol whose paged states deal their positions over the `tp` group
//! (`shards.positions`, decode context parallelism) runs the group as one
//! replicated batch: every member takes the same rows in the same order,
//! with no own-rows-first rotation, no `blocks` and no ranks axis, so the
//! layout is the one a group of one would have and a collective finds the
//! same sequence at the same index on every member. A [`Row`] is then a
//! paged lease on every member, each of `⌈tokens / w⌉` local positions
//! (symmetric: at most one position wasted), and its owner is the group's
//! first member, whose answer is read. Only what addresses a member's own
//! positions differs per member: position `p` is written by member
//! `p % w` alone at its local `p / w`, every other member's slot for it
//! is its pad's; a member's sequence length is the count of positions it
//! holds, which may be 0; its page and line tables are its own lease's.
//! The vars stay one set for the group. Nothing is kept: such a group is a
//! decode tier whose next turn starts from a prefill elsewhere, so a
//! checkpoint or a retire answers that nothing was kept, and with no
//! snapshot there is nothing to restore.

use std::collections::hash_map::RandomState;
use std::collections::BTreeMap;
use std::hash::{BuildHasher, Hasher};
use std::ops::Range;
use std::sync::Arc;

use anyhow::{bail, ensure, Result};
use infront_engine::SamplingParams;
use kern_manifest::protocol::{Axis, Filled, Forward, Rows, Sampling, Values};
use kern_manifest::types::Fill;
use kern_manifest::Protocol;
use kern_pool::{Checkpoint, Copies, Denied, Interleave, Kept, Lease, Pool};
use serde::{Deserialize, Serialize};

use crate::pp::Item;

/// A rank: an index into the ranks. Only [`Ranks`] makes one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Rank(usize);

impl Rank {
    pub fn index(self) -> usize {
        self.0
    }
}

/// How `n` ranks split into group batches of `t`: rank `q` is member
/// `q % t` of group `q / t`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Groups {
    n: usize,
    t: usize,
}

impl Groups {
    pub fn new(n: usize, t: usize) -> Result<Groups> {
        ensure!(n >= 1 && t >= 1, "{n} ranks in groups of {t}");
        ensure!(n.is_multiple_of(t), "{n} ranks do not split into group batches of {t} (the manifest's `tp` group)");
        Ok(Groups { n, t })
    }

    /// Ranks in all.
    pub fn ranks(self) -> usize {
        self.n
    }

    /// Ranks per group batch.
    pub fn size(self) -> usize {
        self.t
    }

    pub fn member(self, q: usize) -> usize {
        q % self.t
    }

    /// The ranks of `q`'s group, in member order.
    pub fn members(self, q: usize) -> Range<usize> {
        let base = q / self.t * self.t;
        base..base + self.t
    }

    /// The ranks of `q`'s group in the order rank `q` lays their blocks
    /// out: itself first, then member `+1`, `+2`, ... around the group.
    fn blocks(self, q: usize) -> impl Iterator<Item = usize> {
        let (base, me, t) = (q / self.t * self.t, self.member(q), self.t);
        (0..t).map(move |d| base + (me + d) % t)
    }
}

/// [`Copies`] on the wire.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Moves {
    pub pages: Vec<(i32, i32)>,
    pub slot: Option<(i32, i32)>,
}

impl From<Copies> for Moves {
    fn from(c: Copies) -> Moves {
        Moves { pages: c.pages, slot: c.slot }
    }
}

impl From<Moves> for Copies {
    fn from(m: Moves) -> Copies {
        Copies { pages: m.pages, slot: m.slot }
    }
}

/// One decision of a rank's pool, as the rank runs it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Act {
    /// Zero a fresh lease's slot in every per-sequence state.
    Zero(i32),
    /// Pages and a slot within the device pool.
    Copy(Moves),
}

/// What a rank is sent: its acts in decision order, then the step's item
/// when there is one, without the tables the rank already holds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub acts: Vec<Act>,
    pub item: Option<Item>,
}

/// A rank's reply to an [`Order`], once its device is done with the item:
/// the item's `tokens` and `count` outputs (empty without).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Answer {
    pub tokens: Vec<i64>,
    pub counts: Vec<i64>,
}

/// One sequence's pieces across its owner's group, one per member in
/// member order: leases (a [`Row`]) or checkpoints (a [`Snapshot`]). The
/// owner's piece, at `me`, is the paged one; the peers' hold a slot
/// alone. Dropping it returns all of them.
pub struct Group<X> {
    owner: Rank,
    me: usize,
    parts: Vec<X>,
}

pub type Row = Group<Lease>;
pub type Snapshot = Group<Checkpoint>;

impl<X: Clone> Clone for Group<X> {
    fn clone(&self) -> Group<X> {
        Group { owner: self.owner, me: self.me, parts: self.parts.clone() }
    }
}

impl<X> Group<X> {
    pub fn owner(&self) -> Rank {
        self.owner
    }

    /// The owner's piece: the one that names positions.
    fn own(&self) -> &X {
        &self.parts[self.me]
    }

    fn by_ref(&self) -> Group<&X> {
        Group { owner: self.owner, me: self.me, parts: self.parts.iter().collect() }
    }

    fn by_mut(&mut self) -> Group<&mut X> {
        Group { owner: self.owner, me: self.me, parts: self.parts.iter_mut().collect() }
    }

    fn map<Y>(self, f: impl FnMut(X) -> Y) -> Group<Y> {
        Group { owner: self.owner, me: self.me, parts: self.parts.into_iter().map(f).collect() }
    }
}

impl Row {
    /// Positions already filled when the row was handed out.
    pub fn prefix(&self) -> usize {
        self.own().prefix()
    }

    /// Each member's lease, in member order: a replicated row's local
    /// positions on every member, else the owner's pages and its peers'
    /// slots.
    pub fn leases(&self) -> &[Lease] {
        &self.parts
    }
}

impl std::fmt::Debug for Row {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Row(rank {}, {:?}, {} members)", self.owner.0, self.own(), self.parts.len())
    }
}

impl<K: Kept> Kept for Group<K> {
    fn tokens(&self) -> usize {
        self.own().tokens()
    }

    fn has_slot(&self) -> bool {
        self.own().has_slot()
    }
}

/// One row's part of a step: the `per` tokens it feeds at `pos..`, and how
/// its token is drawn.
pub struct Cell<'a> {
    pub row: &'a Row,
    pub ids: Vec<i64>,
    pub pos: usize,
    pub sampling: Sampling,
}

/// How a request's tokens are drawn, as the manifest's fills take it: the
/// most likely token when the frontend calls the request greedy, else its
/// parameters with its seed, or one drawn here so two requests without one
/// draw apart. The draw is the shell's: nothing past admission is random.
pub fn sampling(p: &SamplingParams) -> Sampling {
    if p.is_greedy() {
        return Sampling::GREEDY;
    }
    let seed = p.seed.map_or_else(|| RandomState::new().build_hasher().finish() as i64, |s| s as i64);
    Sampling { temperature: p.temperature, top_k: p.top_k, top_p: p.top_p, min_p: p.min_p, seed }
}

/// One rank's pool and what is queued on it.
struct Account {
    pool: Arc<Pool>,
    /// A page and a slot no sequence owns: padding rows write here.
    pad: Lease,
    /// Decisions not yet sent, in the order they were made.
    acts: Vec<Act>,
    /// The tables the rank's buffers hold: the last item sent, its rows
    /// cut to the page and line tables.
    held: Option<Item>,
}

pub struct Ranks {
    protocol: Protocol,
    groups: Groups,
    /// How a replicated group deals positions; `None` when rows are owned.
    dealt: Option<Interleave>,
    accounts: Vec<Account>,
}

impl Ranks {
    /// The accounting of `pools`, one per rank in rank order (each built
    /// from that rank's budget); every rank's pad is leased and its slot
    /// zeroing queued.
    pub fn new(protocol: Protocol, groups: Groups, pools: Vec<Pool>) -> Result<Ranks> {
        ensure!(pools.len() == groups.n, "{} pools for {} ranks", pools.len(), groups.n);
        let dealt = match &protocol.shards.positions {
            Some(p) if p.group != "tp" || p.members != groups.t as u64 => bail!(
                "the paged states deal their positions over `{}` of {}; a replicated batch runs over the `tp` group of {}",
                p.group,
                p.members,
                groups.t
            ),
            Some(p) => Some(Interleave::new(p.members)?),
            None => None,
        };
        // A group of several members that owns its rows lays its blocks
        // out through the `blocks` fill (peer_collective.cu "own rows
        // first"); alone, or replicated, the block is the group batch.
        if dealt.is_none() && groups.t > 1 && !protocol.fills.iter().any(|f| f.fill == Fill::Blocks) {
            bail!(
                "the manifest's `tp` group has {} members but no input has fill `blocks` (the batch's rows per member)",
                groups.t
            );
        }
        let accounts = pools
            .into_iter()
            .enumerate()
            .map(|(q, pool)| {
                let pool = Arc::new(pool);
                let pad = pool.lease(1).map_err(|d| anyhow::anyhow!("rank {q}: no page for the padding rows: {d}"))?;
                if pool.used() == pool.pages() {
                    bail!("rank {q}: the pool holds one page; nothing left to serve from");
                }
                let acts = pad.seq_slot().map(Act::Zero).into_iter().collect();
                Ok(Account { pool, pad, acts, held: None })
            })
            .collect::<Result<_>>()?;
        Ok(Ranks { protocol, groups, dealt, accounts })
    }

    pub fn protocol(&self) -> &Protocol {
        &self.protocol
    }

    pub fn groups(&self) -> Groups {
        self.groups
    }

    /// How the group deals positions, when it runs a replicated batch.
    pub fn interleave(&self) -> Option<Interleave> {
        self.dealt
    }

    /// The groups a step lays out, and the ranks per layout rank: a
    /// replicated group lays out as one rank alone, every member a copy.
    fn laid(&self) -> (Groups, usize) {
        match self.dealt {
            Some(_) => (Groups { n: self.groups.n / self.groups.t, t: 1 }, self.groups.t),
            None => (self.groups, 1),
        }
    }

    /// Sequences one rank may run at once: the manifest's bound, and its
    /// share of the group batch when there is one.
    pub fn seqs_max(&self) -> usize {
        seqs_max(&self.protocol, self.groups.t)
    }

    // ---- accounting, summed over the ranks.

    /// Page unit in tokens.
    pub fn page(&self) -> usize {
        self.accounts[0].pool.unit() as usize
    }

    /// Pages held by rows, every rank together (the pads excluded).
    pub fn pages_used(&self) -> usize {
        self.accounts.iter().map(|a| a.pool.used() - a.pad.pages()).sum()
    }

    /// Pages rows can hold, every rank together (the pads excluded).
    pub fn pages_total(&self) -> usize {
        self.accounts.iter().map(|a| a.pool.pages() - a.pad.pages()).sum()
    }

    /// Longest sequence a row can hold on any rank: what its page table
    /// addresses, within the pages beside the pad; `w` times the shortest
    /// member's when a replicated group deals positions over `w`.
    pub fn max_seq_tokens(&self) -> usize {
        let page = self.page();
        let local = self
            .accounts
            .iter()
            .map(|a| a.pool.max_seq_tokens().min((a.pool.pages() - a.pad.pages()) * page))
            .min()
            .unwrap_or(0);
        local * self.dealt.map_or(1, |il| il.members() as usize)
    }

    pub fn has_seq_state(&self) -> bool {
        self.accounts[0].pool.has_slots()
    }

    /// (slots, slots held by rows) over every rank; (0, 0) without
    /// per-sequence states.
    pub fn seq_slots(&self) -> (usize, usize) {
        if !self.has_seq_state() {
            return (0, 0);
        }
        let all = self.accounts.iter().map(|a| a.pool.slots()).sum();
        (all, self.accounts.iter().map(|a| a.pool.slots_used()).sum::<usize>() - self.groups.n)
    }

    /// Whether any rank has acts not yet sent.
    pub fn outstanding(&self) -> bool {
        self.accounts.iter().any(|a| !a.acts.is_empty())
    }

    // ---- rows: every act is the same act on every member of a group.

    /// `f(account, member, piece)` on every member of `g`'s group in
    /// member order, as a group of what it returns; the first `Err` drops
    /// what the earlier members handed out.
    fn each<X, Y>(
        &self,
        g: Group<X>,
        mut f: impl FnMut(&Account, usize, X) -> Result<Y, Denied>,
    ) -> Result<Group<Y>, Denied> {
        let Group { owner, me, parts } = g;
        let base = self.groups.members(owner.0).start;
        let parts: Result<Vec<Y>, Denied> =
            parts.into_iter().enumerate().map(|(m, x)| f(&self.accounts[base + m], m, x)).collect();
        Ok(Group { owner, me, parts: parts? })
    }

    /// Queue `act(member, piece)` on each member of `g`'s group.
    fn queue<X>(&mut self, g: &Group<X>, mut act: impl FnMut(usize, &X) -> Option<Act>) {
        let base = self.groups.members(g.owner.0).start;
        for (m, x) in g.parts.iter().enumerate() {
            self.accounts[base + m].acts.extend(act(m, x));
        }
    }

    /// A fresh row of `tokens` on the open rank with the fewest rows, then
    /// the fewest pages in use (`rows` says how many a rank holds, `None`
    /// when it can take no more), a slot-only lease on each of its peers;
    /// in a replicated group, on the open group's first member with the
    /// fewest, a paged lease on every member. Every slot is zeroed.
    /// [`Denied::Busy`] when no rank is open.
    pub fn lease(&mut self, tokens: usize, rows: impl Fn(Rank) -> Option<usize>) -> Result<Row, Denied> {
        let row = self.seat(tokens, rows)?;
        self.queue(&row, |_, l| l.seq_slot().map(Act::Zero));
        Ok(row)
    }

    /// [`Ranks::lease`] for a row whose state is pushed in whole from
    /// elsewhere: no slot is zeroed, since a zero landing after the push
    /// would clear it.
    pub fn adopt(&mut self, tokens: usize, rows: impl Fn(Rank) -> Option<usize>) -> Result<Row, Denied> {
        self.seat(tokens, rows)
    }

    fn seat(&self, tokens: usize, rows: impl Fn(Rank) -> Option<usize>) -> Result<Row, Denied> {
        let owner = (0..self.groups.n)
            .filter(|&q| self.dealt.is_none() || self.groups.member(q) == 0)
            .filter_map(|q| rows(Rank(q)).map(|n| (n, self.accounts[q].pool.used(), q)))
            .min()
            .map(|(_, _, q)| q)
            .ok_or(Denied::Busy)?;
        let me = self.groups.member(owner);
        let slots = Group { owner: Rank(owner), me, parts: vec![(); self.groups.t] };
        match self.dealt {
            None => self.each(slots, |a, m, ()| if m == me { a.pool.lease(tokens) } else { a.pool.lease_slot() }),
            Some(il) => {
                let w = il.members() as usize;
                self.each(slots, |a, _, ()| {
                    a.pool.lease(tokens.div_ceil(w)).map_err(|d| match d {
                        Denied::ExceedsRow { limit } => Denied::ExceedsRow { limit: limit * w },
                        d => d,
                    })
                })
            }
        }
    }

    /// A row continuing from the first `len` tokens of `snap` with room
    /// for `tokens`, on the snapshot's owner; the peers continue their
    /// slots at the snapshot's length.
    pub fn lease_from(&mut self, snap: &Snapshot, len: usize, tokens: usize) -> Result<Row, Denied> {
        let me = snap.me;
        let got =
            self.each(snap.by_ref(), |a, m, cp| a.pool.restore(cp, if m == me { len } else { cp.tokens() }, tokens))?;
        Ok(self.copied(got))
    }

    /// The first `len` tokens of `row` as a snapshot it keeps running
    /// past; `None` in a replicated group, which keeps nothing.
    pub fn checkpoint(&mut self, row: &mut Row, len: usize) -> Result<Option<Snapshot>, Denied> {
        if self.dealt.is_some() {
            return Ok(None);
        }
        let got = self.each(row.by_mut(), |a, _, l| a.pool.checkpoint(l, len))?;
        Ok(Some(self.copied(got)))
    }

    /// Queue each member's copies, keep what they made.
    fn copied<X>(&mut self, got: Group<(X, Copies)>) -> Group<X> {
        let mut copies: Vec<Copies> = Vec::with_capacity(got.parts.len());
        let got = got.map(|(x, c)| {
            copies.push(c);
            x
        });
        let mut copies = copies.into_iter();
        self.queue(&got, |_, _| copies.next().map(|c| Act::Copy(c.into())));
        got
    }

    /// The first `len` tokens of a finished row as a snapshot; nothing is
    /// copied. `None` in a replicated group: the row is dropped.
    pub fn retire(&mut self, row: Row, len: usize) -> Option<Snapshot> {
        if self.dealt.is_some() {
            return None;
        }
        let base = self.groups.members(row.owner.0).start;
        let mut m = 0..;
        Some(row.map(|l| self.accounts[base + m.next().expect("unbounded")].pool.retire(l, len)))
    }

    // ---- orders.

    /// Every rank's order: its queued acts and its part of the step
    /// (`items[q]`, from [`Staged::items`]).
    pub fn orders(&mut self, items: Option<Vec<Item>>) -> Vec<Order> {
        let mut items = items.map(Vec::into_iter);
        let p = &self.protocol;
        self.accounts
            .iter_mut()
            .map(|a| {
                let item = items.as_mut().and_then(Iterator::next).map(|i| elide(p, &mut a.held, i));
                Order { acts: std::mem::take(&mut a.acts), item }
            })
            .collect()
    }

    /// Every rank answered, in rank order.
    pub fn heard(&self, answers: &[Answer]) -> Result<()> {
        ensure!(answers.len() == self.groups.n, "{} answers from {} ranks", answers.len(), self.groups.n);
        Ok(())
    }

    // ---- steps.

    /// Lay a step out on every rank and decide its inputs: `per` tokens
    /// per cell, rows per rank `bucket` of the most any rank holds.
    pub fn stage(&self, cells: &[Cell<'_>], per: usize, bucket: impl Fn(usize) -> usize) -> Result<Staged> {
        if per == 0 {
            bail!("a cell feeds at least one token");
        }
        if let Some(c) = cells.iter().find(|c| c.ids.len() != per) {
            bail!("every cell of a step feeds {per} tokens, one feeds {}", c.ids.len());
        }
        let (laid, stride) = self.laid();
        let owners: Vec<usize> = cells.iter().map(|c| c.row.owner.0 / stride).collect();
        let layout = Layout::new(&owners, per, &laid, bucket);
        let b = layout.b;
        if b > self.protocol.groups.max as usize {
            bail!("{b} rows per rank, the manifest allows {}", self.protocol.groups.max);
        }
        if let Some(t) = &self.protocol.ranks {
            if (layout.batch * per) as u64 > t.max {
                bail!("{} rows in the group batch, the manifest allows {}", layout.batch * per, t.max);
            }
        }
        let context: usize = cells.iter().map(|c| c.pos + c.ids.len()).sum();
        let vars = self.protocol.vars(b as u64, (b * per) as u64, (layout.batch * per) as u64, context as u64);
        let rows = (0..self.groups.n).map(|q| self.stage_rank(q, cells, &layout)).collect::<Result<_>>()?;
        Ok(Staged { layout, stride, vars, rows })
    }

    /// Rank `q`'s inputs for a step, by name: every fill, the page tables
    /// and the line tables (see the module doc for which span the group).
    fn stage_rank(&self, q: usize, cells: &[Cell<'_>], l: &Layout) -> Result<BTreeMap<String, Values>> {
        let (p, pad, me) = (&self.protocol, &self.accounts[q].pad, self.groups.member(q));
        let ((groups, stride), per, b) = (self.laid(), l.per, l.b);
        // This rank's place in the layout: a replicated member is its group's.
        let q = q / stride;
        // Every rank's pages and slot for a row are the lease at its own
        // member index: the owner's alone lay a row out unless replicated.
        let lease_on = |i: usize| -> &Lease { &cells[i].row.parts[me] };
        // Where this member writes position `p` (`None`: another member
        // holds it), and how many of a sequence's first `n` it holds.
        let local = |p: usize| match self.dealt {
            Some(il) => (il.owner(p as u64) == me as u64).then(|| il.local(p as u64) as usize),
            None => Some(p),
        };
        let held = |n: usize| self.dealt.map_or(n, |il| il.len(n as u64, me as u64) as usize);
        let ids_of = |c: Option<usize>| -> Vec<i64> { c.map_or(vec![0; per], |i| cells[i].ids.clone()) };
        let pos_of = |c: Option<usize>| -> usize { c.map_or(0, |i| cells[i].pos) };
        let samplings: Vec<Sampling> = l.rows(q).map(|c| c.map_or(Sampling::GREEDY, |i| cells[i].sampling)).collect();
        let mut out: BTreeMap<String, Values> = BTreeMap::new();
        for f in &p.fills {
            if let Some(v) = Sampling::column(f.fill, &samplings) {
                out.insert(f.name.clone(), v);
                continue;
            }
            let v: Vec<i64> = match (f.fill, f.axis) {
                (Fill::Token, Axis::Rows) => l.rows(q).flat_map(ids_of).collect(),
                (Fill::Token, Axis::Ranks) => l.batch_rows(&groups, q).flat_map(ids_of).collect(),
                // Each sequence's first token, the anchor a drafting program
                // splices its own rows from.
                (Fill::Token, Axis::Groups) => l.rows(q).map(|c| ids_of(c)[0]).collect(),
                (Fill::Valid, _) => l.valid(q),
                (Fill::Position, _) => l.rows(q).flat_map(|c| (0..per).map(move |j| (pos_of(c) + j) as i64)).collect(),
                (Fill::Slot, _) => l
                    .rows(q)
                    .flat_map(|c| {
                        let pos = pos_of(c);
                        (pos..pos + per).map(move |p| match (c, local(p)) {
                            (Some(i), Some(at)) => lease_on(i).slot(at),
                            _ => pad.slot(0),
                        })
                    })
                    .collect(),
                (Fill::SeqLen, _) => l.rows(q).map(|c| held(pos_of(c) + per) as i64).collect(),
                (Fill::CuSeqlens, _) => (0..=b as i64).map(|i| i * per as i64).collect(),
                (Fill::Blocks, _) => l.offsets(&groups, q).into_iter().map(|o| (o * per) as i64).collect(),
                (Fill::Tokens | Fill::Count | Fill::Error, _) => continue,
                (Fill::Token, Axis::Fixed(_)) => unreachable!("the protocol checks fill shapes"),
                (Fill::Temperature | Fill::TopK | Fill::TopP | Fill::MinP | Fill::Seed, _) => {
                    unreachable!("staged above")
                }
            };
            out.insert(f.name.clone(), Values::Ints(v));
        }
        for t in &p.page_tables {
            let mut table = Vec::new();
            for c in l.rows(q) {
                c.map_or(pad, |i| lease_on(i)).extend_row(&t.name, &mut table)?;
            }
            out.insert(t.name.clone(), Values::Ints(table.into_iter().map(i64::from).collect()));
        }
        // Line tables are written whole: cell `[r, c]` carries the line of
        // the row in column `c` in entry 0 (a program that moves along a
        // wide cell does so on the device), the null line 0 in the rest,
        // and the pad's past the batch.
        for t in &p.line_tables {
            let (cols_max, cols): (usize, Vec<Option<usize>>) = match t.axis {
                Axis::Ranks => (p.ranks.as_ref().map_or(0, |b| b.max) as usize, l.batch_rows(&groups, q).collect()),
                _ => (p.groups.max as usize, l.rows(q).collect()),
            };
            let (name, w) = (t.name.as_str(), t.width);
            let fill = pad.seq_line_row(name)?;
            let mut lines = Vec::with_capacity(cols.len());
            for c in &cols {
                lines.push(c.map(|i| lease_on(i).seq_line_row(name)).transpose()?);
            }
            let mut table = vec![0i64; t.lines * cols_max * w];
            for r in 0..t.lines {
                for c in 0..cols_max {
                    let col = lines.get(c).and_then(Option::as_ref).unwrap_or(&fill);
                    table[(r * cols_max + c) * w] = i64::from(col[r]);
                }
            }
            out.insert(name.to_string(), Values::Ints(table));
        }
        Ok(out)
    }
}

/// `item` as its rank is sent it: without the page and line tables its
/// buffers already hold (an input an item leaves out keeps its values),
/// which is most of a decode step's bytes, since a row's tables change
/// only when it crosses a page or the batch changes. Only an item of the
/// same program and vars as the last one leaves a table out, a table
/// being written at the item's shape. `held` becomes what the buffers
/// hold after it.
fn elide(p: &Protocol, held: &mut Option<Item>, mut item: Item) -> Item {
    let table = |n: &str| p.page_tables.iter().any(|t| t.name == n) || p.line_tables.iter().any(|t| t.name == n);
    let h = match held {
        Some(h) if h.program == item.program && h.vars == item.vars => h,
        _ => held.insert(Item { program: item.program.clone(), vars: item.vars.clone(), ..Item::default() }),
    };
    item.rows.retain(|n, v| match table(n) {
        false => true,
        true if h.rows.get(n) == Some(v) => false,
        true => {
            h.rows.insert(n.clone(), v.clone());
            true
        }
    });
    item
}

/// Sequences per rank: the sequences bound, and the group batch bound's
/// share when the batch spans `t` ranks.
fn seqs_max(p: &Protocol, t: usize) -> usize {
    let g = p.groups.max as usize;
    p.ranks.as_ref().map_or(g, |r| g.min(r.max as usize / t)).max(1)
}

/// A step laid out on every rank: each rank's inputs, and where each
/// cell's answer is read back from.
pub struct Staged {
    layout: Layout,
    /// Ranks per layout rank: the group's size when replicated, else 1.
    stride: usize,
    vars: BTreeMap<String, u64>,
    rows: Vec<BTreeMap<String, Values>>,
}

impl Staged {
    /// The forward the protocol picks for this step's shape: `b` sequences
    /// of `rows` rows on every rank.
    pub fn forward(&self, p: &Protocol, rows: u64) -> Option<Forward> {
        p.forward(self.layout.b as u64, Rows::Const(rows)).cloned()
    }

    /// Every rank's item running `f` over the staged inputs, in rank order.
    pub fn items(&mut self, f: &Forward) -> Vec<Item> {
        self.rows
            .iter_mut()
            .map(|rows| Item {
                program: f.name.clone(),
                vars: self.vars.clone(),
                rows: std::mem::take(rows),
                ..Item::default()
            })
            .collect()
    }

    /// What `f` handed each cell, in cell order, read from its owner's
    /// answer: its `tokens` output's cell, cut to its
    /// `count` (one without a count); empty for a forward that only
    /// advances state. A replicated group's members ran the same rows on
    /// the same reduced logits, so a member whose outputs differ from the
    /// first member's is an error, not a choice of whom to believe.
    pub fn emitted(&self, p: &Protocol, f: &Forward, answers: &[Answer]) -> Result<Vec<Vec<i64>>> {
        let mut out: Vec<Vec<i64>> = vec![Vec::new(); self.layout.own.iter().map(Vec::len).sum()];
        let Some(i) = f.emits else { return Ok(out) };
        let (t, c): (&Filled, Option<&Filled>) = (&p.fills[i], f.count.map(|c| &p.fills[c]));
        let k = t.width as usize;
        for (g, own) in self.layout.own.iter().enumerate() {
            let q = g * self.stride;
            if own.is_empty() {
                continue;
            }
            let a = answers.get(q).ok_or_else(|| anyhow::anyhow!("no answer from rank {q}"))?;
            for m in q + 1..q + self.stride {
                let b = answers.get(m).ok_or_else(|| anyhow::anyhow!("no answer from rank {m}"))?;
                if (&b.tokens, &b.counts) != (&a.tokens, &a.counts) {
                    bail!("rank {m}: `{}` differs from rank {q}'s in its replicated group", t.name);
                }
            }
            for (r, i) in self.layout.cell_rows(g) {
                let n = match c {
                    Some(c) => {
                        *a.counts.get(r).ok_or_else(|| anyhow::anyhow!("rank {q}: no `{}` for row {r}", c.name))?
                    }
                    None => 1,
                };
                if n < 1 || n > k as i64 {
                    bail!("rank {q}: `{}` says {n} of the {k} rows are taken", c.expect("counted").name);
                }
                let Some(v) = a.tokens.get(r * k..r * k + n as usize) else {
                    bail!("rank {q}: `{}` holds {} values, row {r} reads past them", t.name, a.tokens.len())
                };
                out[i] = v.to_vec();
            }
        }
        Ok(out)
    }
}

/// Which cells each rank owns and where their rows sit, and the bucket
/// they were padded to: the pure half of [`Ranks::stage`].
#[derive(Debug, Clone, PartialEq, Eq)]
struct Layout {
    /// `own[q]` = indices of the cells rank `q` owns, in block order.
    own: Vec<Vec<usize>>,
    /// Rows per rank for the buffers over this rank's own rows: the
    /// bucket of the largest block.
    b: usize,
    /// Tokens per row.
    per: usize,
    /// Rows in rank `q`'s block of the group batch: its cells, then
    /// padding up to the batch's rows.
    block: Vec<usize>,
    /// Rows of the group batch, the same in every group.
    batch: usize,
}

impl Layout {
    /// Row validity is independent of token id, position and pad lease address.
    fn valid(&self, rank: usize) -> Vec<i64> {
        self.rows(rank).flat_map(|cell| std::iter::repeat_n(i64::from(cell.is_some()), self.per)).collect()
    }

    /// `owners[i]` = the rank that owns cell `i`. `bucket` pads a row
    /// count up to its graph's.
    fn new(owners: &[usize], per: usize, groups: &Groups, bucket: impl Fn(usize) -> usize) -> Layout {
        let mut own = vec![Vec::new(); groups.n];
        for (i, &q) in owners.iter().enumerate() {
            own[q].push(i);
        }
        // Every rank holds at least its pad row; a group's blocks add up
        // to the batch's rows, the largest group's bucket, the padding on
        // the smallest blocks first. A bucket short of the rows would
        // drop rows on the floor: never.
        let mut block: Vec<usize> = own.iter().map(|o| o.len().max(1)).collect();
        let total = |block: &[usize], q: usize| groups.members(q).map(|r| block[r]).sum::<usize>();
        let most = (0..groups.n).step_by(groups.t).map(|g| total(&block, g)).max().unwrap_or(1);
        let batch = bucket(most).max(most);
        for g in (0..groups.n).step_by(groups.t) {
            for _ in total(&block, g)..batch {
                let q = groups.members(g).min_by_key(|&r| (block[r], r)).expect("a group has members");
                block[q] += 1;
            }
        }
        let widest = block.iter().copied().max().unwrap_or(1);
        Layout { own, b: bucket(widest).max(widest), per, block, batch }
    }

    /// Rank `q`'s block: its own cells, then `None` for each padding row.
    fn block_rows(&self, q: usize) -> impl Iterator<Item = Option<usize>> + '_ {
        self.own[q].iter().copied().map(Some).chain(std::iter::repeat(None)).take(self.block[q])
    }

    /// Rank `q`'s rows for the buffers over its own rows: its block, then
    /// padding up to `b`.
    fn rows(&self, q: usize) -> impl Iterator<Item = Option<usize>> + '_ {
        self.block_rows(q).chain(std::iter::repeat(None)).take(self.b)
    }

    /// The group batch as rank `q` lays it out: its own block first, then
    /// the other members' around the group.
    fn batch_rows(&self, groups: &Groups, q: usize) -> impl Iterator<Item = Option<usize>> + '_ {
        groups.blocks(q).flat_map(move |r| self.block_rows(r))
    }

    /// The batch's blocks in member order for rank `q`'s group, as
    /// exclusive prefix sums of rows: the `blocks` fill.
    fn offsets(&self, groups: &Groups, q: usize) -> Vec<usize> {
        groups
            .members(q)
            .scan(0, |acc, r| {
                let o = *acc;
                *acc += self.block[r];
                Some(o)
            })
            .chain(std::iter::once(self.batch))
            .collect()
    }

    /// `(row, cell)` for each cell rank `q` owns: where its token is read.
    fn cell_rows(&self, q: usize) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.rows(q).enumerate().filter_map(|(r, c)| c.map(|i| (r, i)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kern_manifest::types::Manifest;
    use kern_manifest::verify;

    /// A group manifest, kern-manifest's own: 4 ranks' rows in `rows`, the
    /// tokens and the line table over it, the page table and the slots a
    /// rank's own.
    fn grouped() -> Manifest {
        Manifest::from_json(include_str!("../../kern-manifest/tests/fixtures/ranks.json")).unwrap()
    }

    fn groups(n: usize, t: usize) -> Groups {
        Groups::new(n, t).unwrap()
    }

    #[test]
    fn a_group_batch_shares_the_rows_bound() {
        let p = Protocol::check(&verify(grouped()).unwrap()).unwrap();
        // 32 rows over a group of 4 is 8 per rank, the sequences bound; over 8 it is 4.
        assert_eq!((seqs_max(&p, 4), seqs_max(&p, 8), seqs_max(&p, 1)), (8, 4, 8));
    }

    #[test]
    fn groups_lay_blocks_out_own_first() {
        let g = groups(8, 4);
        assert_eq!((g.member(5), g.members(5)), (1, 4..8));
        assert_eq!(g.blocks(5).collect::<Vec<_>>(), [5, 6, 7, 4]);
        assert_eq!(g.blocks(0).collect::<Vec<_>>(), [0, 1, 2, 3]);
        let one = groups(4, 1);
        assert_eq!((one.members(2), one.blocks(2).collect::<Vec<_>>()), (2..3, vec![2]));
        assert!(Groups::new(6, 4).is_err());
    }

    #[test]
    fn layout_buckets_the_most_loaded_rank() {
        let bucket = |k: usize| [1usize, 2, 4, 8].into_iter().find(|&b| b >= k).unwrap_or(k);
        let l = Layout::new(&[0, 0, 1, 3, 0], 1, &groups(4, 1), bucket);
        assert_eq!((l.b, l.per, l.batch), (4, 1, 4));
        assert_eq!(l.own, vec![vec![0, 1, 4], vec![2], vec![], vec![3]]);
        // Alone, a rank's block is the batch: every rank pads to the same rows.
        assert_eq!(l.block, vec![4, 4, 4, 4]);
        assert_eq!(l.rows(0).collect::<Vec<_>>(), [Some(0), Some(1), Some(4), None]);
        assert_eq!(l.rows(2).collect::<Vec<_>>(), [None, None, None, None]);
        assert_eq!(l.cell_rows(0).collect::<Vec<_>>(), [(0, 0), (1, 1), (2, 4)]);
        assert_eq!(l.offsets(&groups(4, 1), 2), vec![0, 4]);
        // No cells still stage one padding row per rank.
        assert_eq!(Layout::new(&[], 1, &groups(2, 1), bucket).b, 1);
    }

    #[test]
    fn a_groups_blocks_add_up_to_its_rows() {
        let bucket = |k: usize| [1usize, 2, 4, 8, 16, 24, 32].into_iter().find(|&b| b >= k).unwrap_or(k);
        let g = groups(4, 4);
        // Rank 1 holds five cells, the others one each: 8 batch rows, no
        // padding, rank 1's block the widest.
        let l = Layout::new(&[1, 1, 1, 1, 1, 0, 2, 3], 1, &g, bucket);
        assert_eq!((l.block.clone(), l.batch, l.b), (vec![1, 5, 1, 1], 8, 8));
        assert_eq!(l.offsets(&g, 2), vec![0, 1, 6, 7, 8]);
        // Rank 3 lays the batch out as its own block, then 0, 1, 2.
        assert_eq!(l.batch_rows(&g, 3).count(), 8);
        assert_eq!(l.batch_rows(&g, 3).take(3).collect::<Vec<_>>(), [Some(7), Some(5), Some(0)]);
        // Padding to the bucket lands on the smallest blocks first, and a
        // rank without cells still holds a row.
        let l = Layout::new(&[0, 0, 0, 0, 0, 1, 1], 1, &g, bucket);
        assert_eq!((l.block.clone(), l.batch, l.b), (vec![5, 4, 4, 3], 16, 8));
        assert_eq!(l.rows(1).collect::<Vec<_>>(), [Some(5), Some(6), None, None, None, None, None, None]);
        // Two groups pad to the same batch rows.
        let g = groups(8, 4);
        let l = Layout::new(&[0, 0, 0, 5], 1, &g, bucket);
        assert_eq!((l.block.clone(), l.batch), (vec![3, 2, 2, 1, 2, 2, 2, 2], 8));
        assert_eq!(l.offsets(&g, 5), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn valid_rows_distinguish_bucket_padding_and_empty_ranks() {
        let layout = Layout::new(&[0, 2, 0], 6, &groups(4, 1), |n| n.next_power_of_two());
        let valid = layout.valid(0);
        assert_eq!(&valid[..12], &[1; 12]);
        assert!(valid[12..].iter().all(|&x| x == 0));
        assert!(layout.valid(1).iter().all(|&x| x == 0));
        let other = layout.valid(2);
        assert_eq!(&other[..6], &[1; 6]);
        assert!(other[6..].iter().all(|&x| x == 0));
    }
}
