//! Token-slot and sequence-slot ownership of the states.
//!
//! The runtime provisions every paged state as `capacity` tokens and every
//! per-sequence state as `seq_slots` slots; the kernels address them
//! through the manifest's tables — inputs whose domain `index_into`s a
//! state: page tables (`stride` tokens per entry) and slot lists (stride 1)
//! over a paged state, line tables (`stride` bytes per line) over a
//! per-sequence one. Which slot holds what is the caller's business, but
//! the only way to name a slot is a [`Lease`]: pages and a sequence slot
//! come out of the pool as a lease, slots, table rows and line indices are
//! computed from it, and everything goes back when it drops. Nothing can
//! free a page twice, free a page it never leased, or address a slot past
//! its lease.
//!
//! Pages are in the runtime's page unit — the lcm of every page table's
//! stride — so one lease serves every paged state at once (a 16-token
//! draft table sees 49 entries per 784-token page of the target's table).
//! A lease is all-or-nothing: a caller takes the pages its worst case
//! needs and holds them, so the pool never fragments.
//!
//! A per-sequence state is `seq_slots` slots of `bytes_per_seq`; slot 0 is
//! never leased (a kernel may read line index 0 as the null line), the
//! rest go one per lease. A line table is shaped `[lines, seqs]` (or
//! `[lines]`): row `r` names, for every sequence of the batch, line `r` of
//! its slot — `slot × lines_per_slot + r`. A wide table `[lines, seqs, w]`
//! has `w` entries per (line, sequence) cell for kernels that take a
//! per-sequence list of lines: the caller puts the line in one of them (the
//! contract of the program says which) and 0, the null line, in the rest.
//!
//! # Checkpoints
//!
//! A [`Checkpoint`] is the first `len` tokens of a sequence kept after the
//! sequence is gone, so a later sequence with the same prefix starts at
//! `len` instead of 0: the pages holding those tokens and, when the
//! manifest has per-sequence states, a slot holding the recurrent state as
//! it was after token `len - 1`. Shared pages live in chains — a page and
//! the chain before it, reference-counted — so a checkpoint is one node
//! however deep it sits, a sequence's checkpoints at every page share one
//! chain, and a page returns to the pool when the last lease or checkpoint
//! holding its node drops. A paged state alone makes a checkpoint free — a
//! node, no bytes move — so a caller leaves one at every page boundary; a
//! recurrent state makes it cost a slot, so a caller leaves one where a
//! request ends ([`Pool::retire`] hands the finished sequence's slot over
//! without a copy) and only there.
//!
//! Restoring ([`Pool::restore`]) shares the checkpoint's whole pages, copies
//! its last page when `len` ends inside one (the new sequence appends into
//! that page, the checkpoint keeps its own), copies the state slot, and
//! hands out a lease whose first `len` positions are read-only — the lease
//! refuses to name a slot inside its prefix. Positions past a checkpoint's
//! `len` in its last page belong to whoever writes them next; a checkpoint
//! claims positions, not the page's tail. The pool decides all of this on
//! the host and returns the byte moves as [`Copies`]; the runtime is the
//! shell that runs them on the stream.
//!
//! # Memory
//!
//! Pages and slots come out of one budget of physical chunks
//! ([`crate::chunks`]): every pooled state is an address range reserved
//! once, a page or a slot exists while its chunks are mapped there, and
//! chunks stay where they were last used. When a lease finds pages (or a
//! slot) short but free objects of the other kind are holding chunks, the
//! pool plans a [`Remap`] that unmakes those and makes what is short, says
//! [`Denied::Remapping`], and the caller asks again once the shell has run
//! the plan and reported it landed ([`Pool::take_pending`],
//! [`Pool::complete`]). One remap at a time. [`Denied::Busy`] is only for
//! when everything is held: something has to go first. So the manifest's
//! `seqs` bound only sizes a step's batch; how many sequences can sleep
//! as checkpoints is the budget's business.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::ops::Range;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use kern_manifest::types::{BufferKind, Dim, Manifest};

use crate::chunks::{Chunks, Kind, Remap};
use crate::error::{bail, Result};

/// A page table input: `stride` tokens per entry, `width` entries per row.
struct Table {
    stride: u64,
    width: usize,
}

/// A line table input over a per-sequence state: `rows` lines per
/// sequence it names, out of `per_slot` lines a slot holds, `width`
/// entries per (line, sequence) cell.
struct SeqTable {
    rows: usize,
    per_slot: i32,
    width: usize,
}

/// A pooled state: its arena in the chunk pool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pooled {
    pub state: String,
    pub kind: Kind,
    /// Bytes per page or per slot.
    pub object: u64,
    /// Pages or slots the arena is reserved for.
    pub objects: usize,
    /// Chunk positions reserved.
    pub positions: usize,
}

/// Where an object stands: not backed, free, handed out, or in a remap
/// that makes or unmakes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    Absent,
    Free,
    Held,
    Arriving,
    Leaving,
}

impl Status {
    /// Usable now: a caller can be handed it (free) or holds it.
    fn exists(self) -> bool {
        matches!(self, Status::Free | Status::Held)
    }
}

/// The accounting behind one mutex: chunks, every page's and slot's
/// status, and the remap not yet landed.
struct Inner {
    chunks: Chunks,
    pages: Vec<Status>,
    /// Slot 0 is `Held` for good: the null line.
    slots: Vec<Status>,
    free_pages: BTreeSet<i32>,
    free_slots: BTreeSet<i32>,
    /// Built, not yet taken by the shell.
    pending: Option<Remap>,
    /// Taken, not yet completed.
    in_flight: bool,
    remapped: bool,
    next_node: u64,
}

/// Shared between the runtime and every live lease and checkpoint so a
/// drop returns pages directly. One caller thread; the mutex is for
/// `Send`, never contended.
pub struct Pool {
    /// Tokens per page.
    unit: u64,
    /// Pages one sequence may hold: what the narrowest table row fits.
    max_pages: usize,
    tables: BTreeMap<String, Table>,
    seq_tables: BTreeMap<String, SeqTable>,
    pooled: Vec<Pooled>,
    inner: Mutex<Inner>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The page tables: every input whose domain `index_into`s a paged state
/// (an index a kernel writes — a carry — is the manifest's business, not
/// the host's).
fn tables(m: &Manifest) -> BTreeMap<String, Table> {
    m.buffers
        .iter()
        .filter_map(|(name, b)| {
            if b.kind != BufferKind::Input {
                return None;
            }
            let d = b.domain.as_ref()?;
            if !m.states.get(d.index_into.as_deref()?).is_some_and(|s| !s.is_per_seq()) {
                return None;
            }
            let Some(Dim::Const(width)) = b.shape.last() else { return None };
            Some((name.clone(), Table { stride: d.stride.max(1), width: *width as usize }))
        })
        .collect()
}

/// The line tables: every input whose domain `index_into`s a per-sequence
/// state, shaped `[lines]`, `[lines, seqs]` or `[lines, seqs, w]`.
fn seq_tables(m: &Manifest) -> Result<BTreeMap<String, SeqTable>> {
    let mut out = BTreeMap::new();
    for (name, b) in &m.buffers {
        if b.kind != BufferKind::Input {
            continue;
        }
        let Some(d) = b.domain.as_ref() else { continue };
        let Some(st) = d.index_into.as_deref().and_then(|s| m.states.get(s)) else { continue };
        if !st.is_per_seq() {
            continue;
        }
        let (rows, width) = match b.shape.as_slice() {
            [Dim::Const(rows)] | [Dim::Const(rows), Dim::Var(_)] => (*rows, 1),
            [Dim::Const(rows), Dim::Var(_), Dim::Const(w)] => (*rows, *w as usize),
            s => bail!(
                Manifest,
                "`{name}` indexes a per-sequence state: expected shape [lines], [lines, seqs] or [lines, seqs, w], got {s:?}"
            ),
        };
        let per_slot = st.bytes_per_seq / d.stride.max(1);
        if rows > per_slot {
            bail!(Manifest, "`{name}` names {rows} lines per sequence, the state holds {per_slot}");
        }
        out.insert(name.clone(), SeqTable { rows: rows as usize, per_slot: per_slot as i32, width });
    }
    Ok(out)
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

fn lcm(a: u64, b: u64) -> u64 {
    a / gcd(a, b) * b
}

/// The page unit: the lcm of every page table's stride (a per-sequence
/// state's stride is bytes per line, not tokens: not a page).
pub fn page_unit(m: &Manifest) -> u64 {
    m.buffers
        .values()
        .filter_map(|b| b.domain.as_ref())
        .filter(|d| d.index_into.as_deref().and_then(|s| m.states.get(s)).is_some_and(|s| !s.is_per_seq()))
        .map(|d| d.stride.max(1))
        .fold(1u64, lcm)
}

/// Tokens one sequence can hold, in whole pages of `unit`: what the
/// narrowest page-table row references. `None` when nothing is paged.
pub(crate) fn row_tokens(m: &Manifest, unit: u64) -> Option<u64> {
    tables(m).values().map(|t| t.width as u64 * t.stride / unit * unit).min()
}

/// The pooled states of `m` in manifest order, paged ones first, sized for
/// a budget of `chunks` chunks: a state is paged, per-sequence or fixed,
/// never a mix.
fn pooled(m: &Manifest, unit: u64, chunk: u64, chunks: u32) -> Result<Vec<Pooled>> {
    let mut paged = Vec::new();
    let mut per_seq = Vec::new();
    for (name, s) in &m.states {
        match (s.bytes_per_token > 0, s.bytes > 0, s.bytes_per_seq > 0) {
            (true, false, false) => paged.push((name.clone(), unit * s.bytes_per_token)),
            (false, false, true) => per_seq.push((name.clone(), s.bytes_per_seq)),
            (false, _, false) => {}
            _ => bail!(
                Manifest,
                "state `{name}` mixes per-token, per-sequence and fixed bytes; a pooled state has one layout"
            ),
        }
    }
    let budget = chunk * chunks as u64;
    let page_bytes: u64 = paged.iter().map(|(_, b)| b).sum();
    let slot_bytes: u64 = per_seq.iter().map(|(_, b)| b).sum();
    let pages = budget.checked_div(page_bytes).map_or(chunks as usize, |n| n as usize);
    let slots = budget.checked_div(slot_bytes).map_or(0, |n| n as usize);
    let entry = |(state, object): (String, u64), kind, objects: usize| Pooled {
        state,
        kind,
        object,
        objects,
        positions: (object * objects as u64).div_ceil(chunk) as usize,
    };
    Ok(paged
        .into_iter()
        .map(|p| entry(p, Kind::Page, pages))
        .chain(per_seq.into_iter().map(|p| entry(p, Kind::Slot, slots)))
        .collect())
}

/// Device copies that realize a pool decision, in page and slot numbers:
/// `pages` are (from, to) for every paged state, `slot` is (from, to) for
/// every per-sequence state. Empty when nothing moves.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Copies {
    pub pages: Vec<(i32, i32)>,
    pub slot: Option<(i32, i32)>,
}

/// A shared page and the chain before it. Holding a node holds every page
/// up to it; the page returns when its last holder lets go.
struct Node {
    /// Unique for the pool's life: the host tier keys its copies by it.
    id: u64,
    page: i32,
    parent: Option<Arc<Node>>,
    pool: Arc<Pool>,
}

impl Drop for Node {
    /// Unwind the chain in a loop: a recursive drop of a 65k-page chain
    /// would overflow the stack.
    fn drop(&mut self) {
        self.pool.release(&[self.page], None);
        let mut next = self.parent.take();
        while let Some(n) = next {
            match Arc::try_unwrap(n) {
                Ok(mut node) => next = node.parent.take(),
                Err(_) => break,
            }
        }
    }
}

/// The pages of a chain, root first.
fn chain_pages(chain: &Option<Arc<Node>>) -> Vec<i32> {
    chain_nodes(chain).into_iter().map(|(_, p)| p).collect()
}

/// The (node id, page) of a chain, root first.
fn chain_nodes(chain: &Option<Arc<Node>>) -> Vec<(u64, i32)> {
    let mut out = Vec::new();
    let mut cur = chain.as_ref();
    while let Some(n) = cur {
        out.push((n.id, n.page));
        cur = n.parent.as_ref();
    }
    out.reverse();
    out
}

/// The node `depth` pages up the chain from `chain` (0: `chain` itself).
fn ancestor(chain: &Option<Arc<Node>>, depth: usize) -> Option<Arc<Node>> {
    let mut cur = chain.clone();
    for _ in 0..depth {
        cur = cur.and_then(|n| n.parent.clone());
    }
    cur
}

impl Inner {
    fn count(v: &[Status], f: impl Fn(Status) -> bool) -> usize {
        v.iter().filter(|&&s| f(s)).count()
    }

    fn anything_held(&self) -> bool {
        self.pages.contains(&Status::Held) || self.slots.iter().skip(1).any(|&s| s == Status::Held)
    }

    /// The `n` lowest absent objects of `v` (skipping slot 0 through
    /// `from`).
    fn absent(v: &[Status], from: usize, n: usize) -> Vec<usize> {
        v.iter().enumerate().skip(from).filter(|(_, &s)| s == Status::Absent).map(|(i, _)| i).take(n).collect()
    }

    /// Plan what a request for `need_p` pages and `need_s` slots is
    /// short of, out of free chunks and, when only one kind is short,
    /// chunks taken from the free objects of the other the request does
    /// not need itself — the highest first, so what stays is packed low.
    /// The plan waits in `pending`. `Busy` when what is held would have
    /// to go first, `ExceedsPool` when even then nothing could give.
    fn rebalance(&mut self, need_p: usize, need_s: usize) -> std::result::Result<(), Denied> {
        let never = |me: &Inner| if me.anything_held() { Denied::Busy } else { Denied::ExceedsPool };
        let (free_p, free_s) = (self.free_pages.len(), self.free_slots.len());
        let (d_p, d_s) = (need_p.saturating_sub(free_p), need_s.saturating_sub(free_s));
        let pages = Inner::absent(&self.pages, 0, d_p);
        let slots = Inner::absent(&self.slots, 1, d_s);
        if pages.len() < d_p || slots.len() < d_s {
            return Err(never(self));
        }
        let mut sim = self.chunks.clone();
        let mut plan = Remap::default();
        // Costs counted per object over-count a chunk two targets share:
        // a chunk too many taken, never too few.
        let cost: usize = pages.iter().map(|&p| sim.cost(Kind::Page, p)).sum::<usize>()
            + slots.iter().map(|&s| sim.cost(Kind::Slot, s)).sum::<usize>();
        let mut sources: Vec<(Kind, i32)> = Vec::new();
        let candidates: Vec<(Kind, i32)> = if d_s == 0 {
            self.free_slots.iter().rev().take(free_s - need_s).map(|&s| (Kind::Slot, s)).collect()
        } else if d_p == 0 {
            self.free_pages.iter().rev().take(free_p - need_p).map(|&p| (Kind::Page, p)).collect()
        } else {
            Vec::new()
        };
        for (k, o) in candidates {
            if sim.free() >= cost {
                break;
            }
            sim.unmake(k, o as usize, &mut plan);
            sources.push((k, o));
        }
        if sim.free() < cost {
            return Err(never(self));
        }
        for &p in &pages {
            sim.make(Kind::Page, p, &mut plan);
        }
        for &s in &slots {
            sim.make(Kind::Slot, s, &mut plan);
        }
        self.chunks = sim;
        for (k, o) in sources {
            match k {
                Kind::Page => {
                    self.free_pages.remove(&o);
                    self.pages[o as usize] = Status::Leaving;
                }
                Kind::Slot => {
                    self.free_slots.remove(&o);
                    self.slots[o as usize] = Status::Leaving;
                }
            }
        }
        for p in pages {
            self.pages[p] = Status::Arriving;
        }
        for s in slots {
            self.slots[s] = Status::Arriving;
        }
        self.pending = Some(plan);
        self.remapped = true;
        Ok(())
    }
}

impl Pool {
    /// The pool of `m` over `chunks` chunks of `chunk` bytes, in whole
    /// pages of [`page_unit`]: the first `first_slots` sequence slots
    /// (slot 0 among them) exist from the start, then as many pages as the
    /// chunks left hold. The [`Remap`] returned makes that initial layout;
    /// the pool already counts it as landed.
    pub fn new(m: &Manifest, chunk: u64, chunks: u32, first_slots: usize) -> Result<(Pool, Remap)> {
        let unit = page_unit(m);
        let tables = tables(m);
        let pooled = pooled(m, unit, chunk, chunks)?;
        let arenas: Vec<(Kind, u64, usize)> = pooled.iter().map(|p| (p.kind, p.object, p.objects)).collect();
        let pages_max = pooled.iter().find(|p| p.kind == Kind::Page).map_or(chunks as usize, |p| p.objects);
        let slots_max = pooled.iter().find(|p| p.kind == Kind::Slot).map_or(0, |p| p.objects);
        let max_pages = row_tokens(m, unit).map_or(pages_max, |t| (t / unit) as usize).min(pages_max);
        let mut inner = Inner {
            chunks: Chunks::new(chunk, &arenas, chunks),
            pages: vec![Status::Absent; pages_max],
            slots: vec![Status::Absent; slots_max],
            free_pages: BTreeSet::new(),
            free_slots: BTreeSet::new(),
            pending: None,
            in_flight: false,
            remapped: false,
            next_node: 0,
        };
        let mut plan = Remap::default();
        for s in 0..first_slots {
            if s >= slots_max || inner.chunks.cost(Kind::Slot, s) > inner.chunks.free() {
                bail!(Api, "{chunks} chunks of {chunk} bytes hold {s} sequence slots, not the {first_slots} asked for");
            }
            inner.chunks.make(Kind::Slot, s, &mut plan);
            inner.slots[s] = if s == 0 { Status::Held } else { Status::Free };
            if s > 0 {
                inner.free_slots.insert(s as i32);
            }
        }
        for p in 0..pages_max {
            if inner.chunks.cost(Kind::Page, p) > inner.chunks.free() {
                break;
            }
            inner.chunks.make(Kind::Page, p, &mut plan);
            inner.pages[p] = Status::Free;
            inner.free_pages.insert(p as i32);
        }
        let pool = Pool { unit, max_pages, tables, seq_tables: seq_tables(m)?, pooled, inner: Mutex::new(inner) };
        Ok((pool, plan))
    }

    pub fn unit(&self) -> u64 {
        self.unit
    }

    /// The pooled states, in arena order.
    pub fn pooled(&self) -> &[Pooled] {
        &self.pooled
    }

    pub fn chunk(&self) -> u64 {
        lock(&self.inner).chunks.chunk()
    }

    /// Pages that exist now (free or held).
    pub fn total(&self) -> usize {
        Inner::count(&lock(&self.inner).pages, Status::exists)
    }

    /// Pages the arena could hold if every chunk were a page.
    pub fn pages_max(&self) -> usize {
        lock(&self.inner).pages.len()
    }

    /// Pages held by a lease or a checkpoint.
    pub fn used(&self) -> usize {
        Inner::count(&lock(&self.inner).pages, |s| s == Status::Held)
    }

    pub fn max_seq_tokens(&self) -> usize {
        self.max_pages * self.unit as usize
    }

    pub fn tables(&self) -> impl Iterator<Item = &str> {
        self.tables.keys().map(String::as_str)
    }

    /// Whether the manifest has per-sequence states (slots at all).
    pub fn has_slots(&self) -> bool {
        self.pooled.iter().any(|p| p.kind == Kind::Slot)
    }

    /// Sequence slots that exist now, slot 0 among them (0 without
    /// per-sequence states).
    pub fn slots(&self) -> usize {
        Inner::count(&lock(&self.inner).slots, Status::exists)
    }

    /// Slots the arena could hold if every chunk were a slot.
    pub fn slots_max(&self) -> usize {
        lock(&self.inner).slots.len()
    }

    /// Sequence slots held by a lease or a checkpoint.
    pub fn slots_used(&self) -> usize {
        Inner::count(lock(&self.inner).slots.get(1..).unwrap_or(&[]), |s| s == Status::Held)
    }

    pub fn seq_tables(&self) -> impl Iterator<Item = &str> {
        self.seq_tables.keys().map(String::as_str)
    }

    /// Whether any remap was ever planned: the initial layout is gone.
    pub fn remapped(&self) -> bool {
        lock(&self.inner).remapped
    }

    /// The remap waiting to be executed, if any; from here until
    /// [`Pool::complete`] one is in flight and no other is planned.
    pub fn take_pending(&self) -> Option<Remap> {
        let mut g = lock(&self.inner);
        let plan = g.pending.take()?;
        g.in_flight = true;
        Some(plan)
    }

    /// The remap landed: what it made is free, what it unmade is gone.
    pub fn complete(&self, plan: Remap) {
        let mut g = lock(&self.inner);
        for (k, o) in plan.made {
            match k {
                Kind::Page => {
                    g.pages[o as usize] = Status::Free;
                    g.free_pages.insert(o);
                }
                Kind::Slot => {
                    g.slots[o as usize] = Status::Free;
                    g.free_slots.insert(o);
                }
            }
        }
        for (k, o) in plan.unmade {
            match k {
                Kind::Page => g.pages[o as usize] = Status::Absent,
                Kind::Slot => g.slots[o as usize] = Status::Absent,
            }
        }
        g.in_flight = false;
    }

    fn pages_for(&self, tokens: usize) -> std::result::Result<usize, Denied> {
        let need = tokens.div_ceil(self.unit as usize);
        if need > self.max_pages {
            return Err(Denied::ExceedsRow { limit: self.max_seq_tokens() });
        }
        Ok(need)
    }

    /// `fresh` free pages, the lowest, and, when the manifest has
    /// per-sequence states, a slot; all or nothing. Short of either, a
    /// remap is planned when free objects of the other kind can give the
    /// chunks.
    fn take(&self, fresh: usize) -> std::result::Result<(Vec<i32>, Option<i32>), Denied> {
        let want_slot = self.has_slots();
        let mut g = lock(&self.inner);
        if fresh <= g.free_pages.len() && (!want_slot || !g.free_slots.is_empty()) {
            let taken: Vec<i32> = (0..fresh).map(|_| g.free_pages.pop_first().expect("counted")).collect();
            for &p in &taken {
                g.pages[p as usize] = Status::Held;
            }
            let slot = want_slot.then(|| g.free_slots.pop_first().expect("counted"));
            if let Some(s) = slot {
                g.slots[s as usize] = Status::Held;
            }
            return Ok((taken, slot));
        }
        if g.pending.is_some() || g.in_flight {
            return Err(Denied::Remapping);
        }
        g.rebalance(fresh, want_slot as usize)?;
        Err(Denied::Remapping)
    }

    fn node_id(&self) -> u64 {
        let mut g = lock(&self.inner);
        g.next_node += 1;
        g.next_node
    }

    fn release(&self, pages: &[i32], slot: Option<i32>) {
        let mut g = lock(&self.inner);
        for &p in pages {
            debug_assert_eq!(g.pages[p as usize], Status::Held);
            g.pages[p as usize] = Status::Free;
            g.free_pages.insert(p);
        }
        if let Some(s) = slot {
            debug_assert_eq!(g.slots[s as usize], Status::Held);
            g.slots[s as usize] = Status::Free;
            g.free_slots.insert(s);
        }
    }

    /// A fresh sequence: the pages `tokens` need and a sequence slot.
    pub fn lease(self: &Arc<Pool>, tokens: usize) -> std::result::Result<Lease, Denied> {
        let need = self.pages_for(tokens)?;
        let (pages, slot) = self.take(need)?;
        Ok(Lease { chain: None, shared: 0, pages, slot, prefix: 0, pool: Arc::clone(self) })
    }

    /// A sequence slot alone, no pages: this rank's share of a sequence
    /// whose positions live on another rank (a tensor-parallel peer holds
    /// a slice of every row's recurrent state and none of its KV). Such a
    /// lease names no position; its checkpoints, forks and wakes move the
    /// slot only, at whatever length the caller says.
    pub fn lease_slot(self: &Arc<Pool>) -> std::result::Result<Lease, Denied> {
        let (pages, slot) = self.take(0)?;
        Ok(Lease { chain: None, shared: 0, pages, slot, prefix: 0, pool: Arc::clone(self) })
    }

    /// The first `len` tokens of `lease` as a checkpoint the lease's
    /// sequence keeps running past: its pages up to there become shared,
    /// its state slot (when the manifest has one) is copied into a fresh
    /// slot — the [`Copies`] say which. `len` is 1 to the lease's tokens
    /// (anything from 1 for a slot-only lease).
    pub fn checkpoint(
        self: &Arc<Pool>,
        lease: &mut Lease,
        len: usize,
    ) -> std::result::Result<(Checkpoint, Copies), Denied> {
        assert!(
            len >= 1 && (!lease.paged() || len <= lease.tokens()),
            "checkpoint of {len} tokens out of a lease of {}",
            lease.tokens()
        );
        let (_, slot) = self.take(0)?;
        let chain = lease.paged().then(|| lease.share(len.div_ceil(self.unit as usize)));
        let copies = Copies { pages: Vec::new(), slot: lease.slot.zip(slot) };
        Ok((Checkpoint { len, chain, slot, pool: Arc::clone(self) }, copies))
    }

    /// The first `len` tokens of a finished sequence as a checkpoint:
    /// the lease's pages past `len` return, the rest and its state slot
    /// move over as they are. Nothing is copied.
    pub fn retire(self: &Arc<Pool>, mut lease: Lease, len: usize) -> Checkpoint {
        assert!(
            len >= 1 && (!lease.paged() || len <= lease.tokens()),
            "retiring {len} tokens out of a lease of {}",
            lease.tokens()
        );
        let chain = lease.paged().then(|| lease.share(len.div_ceil(self.unit as usize)));
        Checkpoint { len, chain, slot: lease.slot.take(), pool: Arc::clone(self) }
    }

    /// A sequence continuing from the first `len` tokens of `cp` with room
    /// for `tokens` (more than `len`): those whole pages shared, a copy of
    /// the page `len` ends inside when it does, fresh pages for the rest,
    /// a fresh slot with the checkpoint's state copied in. `len` is the
    /// checkpoint's own length or, for one without a state slot, any
    /// whole number of its pages. The lease names positions from `len` on.
    /// A slot-only checkpoint restores to a slot-only lease at its own
    /// length; `tokens` is not its business.
    pub fn restore(
        self: &Arc<Pool>,
        cp: &Checkpoint,
        len: usize,
        tokens: usize,
    ) -> std::result::Result<(Lease, Copies), Denied> {
        let unit = self.unit as usize;
        let Some(cp_chain) = &cp.chain else {
            assert!(len == cp.len, "restoring {len} tokens of a slot-only checkpoint of {}", cp.len);
            let (_, slot) = self.take(0)?;
            let lease = Lease { chain: None, shared: 0, pages: Vec::new(), slot, prefix: len, pool: Arc::clone(self) };
            return Ok((lease, Copies { pages: Vec::new(), slot: cp.slot.zip(slot) }));
        };
        assert!(
            len == cp.len || (cp.slot.is_none() && len >= 1 && len < cp.len && len.is_multiple_of(unit)),
            "restoring {len} tokens of a checkpoint of {} ({} slot)",
            cp.len,
            if cp.slot.is_some() { "with a" } else { "no" }
        );
        assert!(tokens > len, "restoring {len} tokens into room for {tokens}");
        let need = self.pages_for(tokens)?;
        let full = len / unit;
        let partial = if len.is_multiple_of(unit) { None } else { Some(cp_chain.page) };
        let (fresh, slot) = self.take(need - full)?;
        // The chain through the whole pages: the node `full` pages deep.
        let chain = ancestor(&cp.chain, cp.pages() - full);
        let mut pages = chain_pages(&chain);
        pages.extend_from_slice(&fresh);
        let copies = Copies { pages: partial.map(|p| (p, fresh[0])).into_iter().collect(), slot: cp.slot.zip(slot) };
        Ok((Lease { chain, shared: full, pages, slot, prefix: len, pool: Arc::clone(self) }, copies))
    }

    /// A sequence branched off the first `len` tokens of `parent`, which
    /// keeps running, with room for `tokens`: the whole pages shared, the
    /// page `len` ends inside copied, a fresh slot with the parent's state
    /// copied in. A recurrent state is the parent's as of now, so with one
    /// `len` must be the parent's position. The lease names positions from
    /// `len` on. A slot-only parent forks a slot-only child: the state
    /// copied, nothing else.
    pub fn fork(
        self: &Arc<Pool>,
        parent: &mut Lease,
        len: usize,
        tokens: usize,
    ) -> std::result::Result<(Lease, Copies), Denied> {
        if !parent.paged() {
            assert!(len >= 1, "forking a slot-only lease at 0 tokens");
            let (_, slot) = self.take(0)?;
            let lease = Lease { chain: None, shared: 0, pages: Vec::new(), slot, prefix: len, pool: Arc::clone(self) };
            return Ok((lease, Copies { pages: Vec::new(), slot: parent.slot.zip(slot) }));
        }
        assert!(len >= 1 && len <= parent.tokens(), "forking {len} tokens out of a lease of {}", parent.tokens());
        assert!(tokens > len, "forking {len} tokens into room for {tokens}");
        let unit = self.unit as usize;
        let need = self.pages_for(tokens)?;
        let full = len / unit;
        let partial = if len.is_multiple_of(unit) { None } else { Some(parent.pages[full]) };
        let (fresh, slot) = self.take(need - full)?;
        let chain = if full > 0 { Some(parent.share(full)) } else { None };
        let mut pages = chain_pages(&chain);
        pages.extend_from_slice(&fresh);
        let copies =
            Copies { pages: partial.map(|p| (p, fresh[0])).into_iter().collect(), slot: parent.slot.zip(slot) };
        Ok((Lease { chain, shared: full, pages, slot, prefix: len, pool: Arc::clone(self) }, copies))
    }

    /// A fresh lease whose first `len` positions the caller fills from
    /// elsewhere (a parked checkpoint's bytes): its own pages, a slot, and
    /// a prefix of `len`.
    pub fn wake(self: &Arc<Pool>, len: usize, tokens: usize) -> std::result::Result<Lease, Denied> {
        assert!(len >= 1 && tokens > len, "waking {len} tokens into room for {tokens}");
        let need = self.pages_for(tokens)?;
        let (pages, slot) = self.take(need)?;
        Ok(Lease { chain: None, shared: 0, pages, slot, prefix: len, pool: Arc::clone(self) })
    }

    /// The slot-only counterpart of [`Pool::wake`]: a slot the caller
    /// fills with a parked state after `len` tokens, no pages.
    pub fn wake_slot(self: &Arc<Pool>, len: usize) -> std::result::Result<Lease, Denied> {
        assert!(len >= 1, "waking a slot-only lease at 0 tokens");
        let (pages, slot) = self.take(0)?;
        Ok(Lease { chain: None, shared: 0, pages, slot, prefix: len, pool: Arc::clone(self) })
    }
}

/// Why [`Runtime::lease`](crate::Runtime::lease) said no.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Denied {
    /// More tokens than one page-table row can reference; never fits.
    ExceedsRow { limit: usize },
    /// More pages than the pool has, even empty; never fits.
    ExceedsPool,
    /// Fits, but not right now: pages or sequence slots all held.
    Busy,
    /// Fits once the remap in flight lands; ask again, evict nothing.
    Remapping,
    /// The host tier has no room for the checkpoint; drop a parked one.
    HostFull,
}

impl fmt::Display for Denied {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Denied::ExceedsRow { limit } => write!(f, "longer than a page-table row ({limit} tokens)"),
            Denied::ExceedsPool => write!(f, "no layout of the state budget holds it"),
            Denied::Busy => write!(f, "pages or sequence slots busy"),
            Denied::Remapping => write!(f, "pages or sequence slots being remapped"),
            Denied::HostFull => write!(f, "no room in the host tier"),
        }
    }
}

impl std::error::Error for Denied {}

/// Pages and a sequence slot leased to one sequence: the only handle to
/// its token slots and its per-sequence state. Dropping it returns them
/// to the runtime. The first `shared` pages are held through a chain
/// (checkpoints hold them too), the rest are the lease's own. A lease
/// restored from a checkpoint starts with `prefix` positions already
/// filled; it never names a slot inside them.
pub struct Lease {
    chain: Option<Arc<Node>>,
    shared: usize,
    /// Every page in order: the chain's, then the lease's own.
    pages: Vec<i32>,
    slot: Option<i32>,
    prefix: usize,
    pool: Arc<Pool>,
}

impl Lease {
    /// The chain through the first `keep` pages, moving own pages into
    /// nodes as needed.
    fn share(&mut self, keep: usize) -> Arc<Node> {
        while self.shared < keep {
            let page = self.pages[self.shared];
            let node = Node { id: self.pool.node_id(), page, parent: self.chain.take(), pool: Arc::clone(&self.pool) };
            self.chain = Some(Arc::new(node));
            self.shared += 1;
        }
        ancestor(&self.chain, self.shared - keep).expect("keep is at least 1")
    }

    /// Pages held.
    pub fn pages(&self) -> usize {
        self.pages.len()
    }

    /// Whether the lease holds pages at all (a slot-only lease names no
    /// position).
    pub fn paged(&self) -> bool {
        !self.pages.is_empty()
    }

    /// The page ids, in position order (a harness reading a restored
    /// prefix back; programs get them through `extend_row`).
    pub fn page_ids(&self) -> &[i32] {
        &self.pages
    }

    /// Token slots held (whole pages, so at least what was asked for).
    pub fn tokens(&self) -> usize {
        self.pages.len() * self.pool.unit as usize
    }

    /// Positions already filled when the lease was handed out: 0 for a
    /// fresh sequence, the checkpoint's length for a restored one.
    pub fn prefix(&self) -> usize {
        self.prefix
    }

    /// The token slot of position `pos` of the sequence, to write into;
    /// `pos` is past the shared prefix.
    pub fn slot(&self, pos: usize) -> i64 {
        assert!(pos >= self.prefix, "position {pos} is inside the shared prefix of {} tokens", self.prefix);
        let unit = self.pool.unit as usize;
        let page = *self.pages.get(pos / unit).expect("position past the lease") as i64;
        page * unit as i64 + (pos % unit) as i64
    }

    /// The token slots of consecutive positions (a `slot_mapping` list).
    pub fn slots(&self, positions: Range<usize>) -> Vec<i64> {
        positions.map(|p| self.slot(p)).collect()
    }

    /// Append the sequence's row of page table `table`: one entry per
    /// `stride` tokens of every page held, then the first entry repeated to
    /// the row's width (entries past the sequence length are never
    /// dereferenced, but the domain wants valid page ids in them).
    pub fn extend_row(&self, table: &str, out: &mut Vec<i32>) -> Result<()> {
        let Some(t) = self.pool.tables.get(table) else {
            bail!(Api, "`{table}` is not a page table of this manifest");
        };
        let per_page = (self.pool.unit / t.stride) as i32;
        let start = out.len();
        for &p in &self.pages {
            out.extend((0..per_page).map(|k| p * per_page + k));
        }
        let fill = out.get(start).copied().unwrap_or(0);
        out.resize(start + t.width, fill);
        Ok(())
    }

    /// The sequence slot held in every per-sequence state (`None` when
    /// the manifest has none). Slot 0 is never handed out.
    pub fn seq_slot(&self) -> Option<i32> {
        self.slot
    }

    /// Byte range of this sequence's slot in a per-sequence state of
    /// `bytes_per_seq`.
    pub(crate) fn seq_bytes(&self, bytes_per_seq: u64) -> Option<Range<usize>> {
        let s = self.slot? as u64;
        Some((s * bytes_per_seq) as usize..((s + 1) * bytes_per_seq) as usize)
    }

    /// Line `row` of this sequence in line table `table`: the index its
    /// entry `[row, i]` holds when the sequence is column `i` of a batch.
    pub fn seq_line(&self, table: &str, row: usize) -> Result<i32> {
        let Some(t) = self.pool.seq_tables.get(table) else {
            bail!(Api, "`{table}` is not a line table of this manifest");
        };
        if row >= t.rows {
            bail!(Api, "`{table}` has {} lines per sequence, asked for line {row}", t.rows);
        }
        let Some(slot) = self.slot else {
            bail!(Api, "lease holds no sequence slot");
        };
        Ok(slot * t.per_slot + row as i32)
    }

    /// Lines per sequence line table `table` names.
    pub fn seq_lines(&self, table: &str) -> Result<usize> {
        match self.pool.seq_tables.get(table) {
            Some(t) => Ok(t.rows),
            None => bail!(Api, "`{table}` is not a line table of this manifest"),
        }
    }

    /// Entries per (line, sequence) cell of line table `table`: 1, or the
    /// `w` of a wide `[lines, seqs, w]` table.
    pub fn seq_width(&self, table: &str) -> Result<usize> {
        match self.pool.seq_tables.get(table) {
            Some(t) => Ok(t.width),
            None => bail!(Api, "`{table}` is not a line table of this manifest"),
        }
    }
}

impl fmt::Debug for Lease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Lease({} pages, {} shared", self.pages.len(), self.shared)?;
        if let Some(s) = self.slot {
            write!(f, ", slot {s}")?;
        }
        write!(f, ", prefix {})", self.prefix)
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        self.pool.release(&self.pages[self.shared..], self.slot.take());
    }
}

/// The first `len` tokens of a sequence that is gone: the pages holding
/// them (shared with whoever else holds them) and, when the manifest has
/// per-sequence states, a slot with the state after those tokens. Made
/// by [`Pool::checkpoint`] / [`Pool::retire`], spent by [`Pool::restore`];
/// dropping it releases what it alone holds.
pub struct Checkpoint {
    len: usize,
    /// The chain through the pages; `None` for a slot-only checkpoint.
    chain: Option<Arc<Node>>,
    slot: Option<i32>,
    pool: Arc<Pool>,
}

impl Checkpoint {
    /// Tokens the checkpoint holds; never 0.
    pub fn tokens(&self) -> usize {
        self.len
    }

    /// Pages held (0 for a slot-only checkpoint).
    pub fn pages(&self) -> usize {
        match self.chain {
            Some(_) => self.len.div_ceil(self.pool.unit as usize),
            None => 0,
        }
    }

    /// Whether pages are held at all.
    pub fn paged(&self) -> bool {
        self.chain.is_some()
    }

    /// The sequence slot holding the state, when the manifest has one.
    pub fn seq_slot(&self) -> Option<i32> {
        self.slot
    }

    /// (node id, page) of every page held, root first: what the host tier
    /// keys its copies by.
    pub fn nodes(&self) -> Vec<(u64, i32)> {
        chain_nodes(&self.chain)
    }
}

impl fmt::Debug for Checkpoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.slot {
            Some(s) => write!(f, "Checkpoint({} tokens, {} pages, slot {s})", self.len, self.pages()),
            None => write!(f, "Checkpoint({} tokens, {} pages)", self.len, self.pages()),
        }
    }
}

impl Drop for Checkpoint {
    fn drop(&mut self) {
        self.pool.release(&[], self.slot.take());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// kv paged in 16-token entries (row of 3), a draft state paged in
    /// 4-token entries (row of 16): page unit 16, one byte per token per
    /// state, so a page is 16 bytes in each of two arenas.
    fn manifest() -> Manifest {
        Manifest::from_json(
            r#"{
            "schema_version": 4, "model": "t", "vars": {"tokens": {"max": 8}, "seqs": {"max": 2}},
            "states": {"kv": {"bytes_per_token": 1}, "draft_kv": {"bytes_per_token": 1}},
            "buffers": {
                "slot_mapping": {"kind": "input", "dtype": "i64", "shape": ["tokens"], "domain": {"index_into": "kv"}},
                "block_table": {"kind": "input", "dtype": "i32", "shape": ["seqs", 3], "domain": {"index_into": "kv", "stride": 16}},
                "draft_block_table": {"kind": "input", "dtype": "i32", "shape": [16], "domain": {"index_into": "draft_kv", "stride": 4}}
            },
            "modules": {}, "ops": {}, "programs": {}
        }"#,
        )
        .unwrap()
    }

    /// One paged state of 16 bytes a page, plus a recurrent state of 3
    /// lines of 8 bytes per sequence and its line table.
    fn hybrid() -> Manifest {
        Manifest::from_json(
            r#"{
            "schema_version": 4, "model": "t", "vars": {"tokens": {"max": 8}, "seqs": {"max": 2}},
            "states": {"kv": {"bytes_per_token": 1}, "gdn": {"bytes_per_seq": 24}},
            "buffers": {
                "block_table": {"kind": "input", "dtype": "i32", "shape": ["seqs", 3], "domain": {"index_into": "kv", "stride": 16}},
                "line_index": {"kind": "input", "dtype": "i32", "shape": [3, "seqs"], "domain": {"index_into": "gdn", "stride": 8}}
            },
            "modules": {}, "ops": {}, "programs": {}
        }"#,
        )
        .unwrap()
    }

    /// 8-byte chunks: a page is 2 chunks per arena (4 for the two-state
    /// manifest), a slot 3. 16 chunks: 4 pages.
    fn pool() -> Arc<Pool> {
        Arc::new(Pool::new(&manifest(), 8, 16, 0).unwrap().0)
    }

    /// 20 chunks: 4 slots (slot 0 among them) and 4 pages, no chunk spare.
    fn hybrid_pool() -> Arc<Pool> {
        Arc::new(Pool::new(&hybrid(), 8, 20, 4).unwrap().0)
    }

    /// Land the remap the pool has planned.
    fn land(p: &Pool) {
        p.complete(p.take_pending().expect("a remap planned"));
    }

    #[test]
    fn geometry() {
        let p = pool();
        assert_eq!((p.total(), p.unit(), p.max_seq_tokens(), p.chunk()), (4, 16, 48, 8));
        assert_eq!(p.tables().collect::<Vec<_>>(), ["block_table", "draft_block_table"]);
        // The draft table's 16 entries × 4 tokens hold 4 pages; kv's 3 × 16 hold 3.
        assert_eq!(p.max_pages, 3);
        assert_eq!((p.has_slots(), p.slots()), (false, 0));
        assert_eq!(p.lease(1).unwrap().seq_slot(), None);
        // Two chunks over what four pages take are spare, not a torn page.
        let (p2, plan) = Pool::new(&manifest(), 8, 18, 0).unwrap();
        assert_eq!((p2.total(), p2.pages_max(), plan.map.len(), plan.unmap.len()), (4, 4, 16, 0));
        assert_eq!(plan.made, [(Kind::Page, 0), (Kind::Page, 1), (Kind::Page, 2), (Kind::Page, 3)]);
        let names: Vec<(&str, Kind, u64, usize, usize)> =
            p2.pooled().iter().map(|a| (a.state.as_str(), a.kind, a.object, a.objects, a.positions)).collect();
        assert_eq!(names, [("draft_kv", Kind::Page, 16, 4, 8), ("kv", Kind::Page, 16, 4, 8)]);
    }

    #[test]
    fn a_state_has_one_layout() {
        let mut m = hybrid();
        m.states.get_mut("gdn").unwrap().bytes = 8;
        let Err(e) = Pool::new(&m, 8, 20, 4) else { panic!("mixed state accepted") };
        assert!(e.to_string().contains("one layout"), "{e}");
    }

    #[test]
    fn drop_returns_pages() {
        let p = pool();
        let a = p.lease(17).unwrap(); // 2 pages
        let b = p.lease(1).unwrap();
        assert_eq!((a.pages(), a.tokens(), b.pages(), p.used()), (2, 32, 1, 3));
        drop(a);
        assert_eq!(p.used(), 1);
        drop(b);
        assert_eq!(p.used(), 0);
        let all: Vec<Lease> = (0..4).map(|_| p.lease(1).unwrap()).collect();
        let mut ids: Vec<i32> = all.iter().flat_map(|l| l.pages.clone()).collect();
        ids.sort();
        assert_eq!((ids, p.used()), (vec![0, 1, 2, 3], 4));
        drop(all);
        assert_eq!(p.used(), 0);
    }

    #[test]
    fn denials() {
        let p = pool();
        assert_eq!(p.lease(49).unwrap_err(), Denied::ExceedsRow { limit: 48 });
        let a = p.lease(48).unwrap();
        // One page free, no other kind to take chunks from: busy.
        assert_eq!(p.lease(17).unwrap_err(), Denied::Busy);
        let _b = p.lease(16).unwrap();
        assert_eq!(p.lease(1).unwrap_err(), Denied::Busy);
        drop(a);
        assert!(p.lease(33).is_ok());
        // Two pages of budget cap the row at two pages.
        let small = Arc::new(Pool::new(&manifest(), 8, 8, 0).unwrap().0);
        assert_eq!((small.pages_max(), small.lease(48).unwrap_err()), (2, Denied::ExceedsRow { limit: 32 }));
    }

    #[test]
    fn slots_and_rows() {
        let p = pool();
        let l = p.lease(20).unwrap(); // 2 pages
        let [p0, p1] = l.pages[..] else { panic!() };
        assert_eq!(l.slot(0), p0 as i64 * 16);
        assert_eq!(l.slot(15), p0 as i64 * 16 + 15);
        assert_eq!(l.slot(16), p1 as i64 * 16);
        assert_eq!(l.slots(15..17), [p0 as i64 * 16 + 15, p1 as i64 * 16]);
        let mut t = Vec::new();
        l.extend_row("block_table", &mut t).unwrap();
        assert_eq!(t, [p0, p1, p0]);
        let mut d = Vec::new();
        l.extend_row("draft_block_table", &mut d).unwrap();
        let mut want: Vec<i32> = (0..4).map(|k| p0 * 4 + k).chain((0..4).map(|k| p1 * 4 + k)).collect();
        want.resize(16, p0 * 4);
        assert_eq!(d, want);
        // Position 17's draft entry (row index 17/4) names the draft page its slot falls in.
        assert_eq!(d[17 / 4], (l.slot(17) / 4) as i32);
        assert!(l.extend_row("slot_mapping", &mut t).is_err());
        assert!(l.seq_line("block_table", 0).is_err());
    }

    #[test]
    #[should_panic(expected = "past the lease")]
    fn slot_past_lease() {
        let p = pool();
        p.lease(16).unwrap().slot(16);
    }

    #[test]
    fn seq_slots_and_lines() {
        let m = hybrid();
        // seqs 2 + pad + null: the manifest's first slots are 0..4, three of them leasable.
        assert_eq!(m.seq_slots(), 4);
        let p = hybrid_pool();
        assert_eq!((p.has_slots(), p.slots(), p.seq_tables().collect::<Vec<_>>()), (true, 4, vec!["line_index"]));
        let a = p.lease(16).unwrap();
        let b = p.lease(16).unwrap();
        let c = p.lease(16).unwrap();
        let (sa, sb, sc) = (a.seq_slot().unwrap(), b.seq_slot().unwrap(), c.seq_slot().unwrap());
        let mut got = vec![sa, sb, sc];
        got.sort();
        assert_eq!((got, p.slots_used()), (vec![1, 2, 3], 3));
        // A fourth page is free but no slot is, and one free page's two
        // chunks are short of a slot's three: busy, not exhausted.
        assert_eq!(p.lease(16).unwrap_err(), Denied::Busy);
        // Line r of a slot: slot × 3 + r; the null line 0 belongs to no lease.
        assert_eq!(a.seq_line("line_index", 2).unwrap(), sa * 3 + 2);
        assert_eq!(a.seq_lines("line_index").unwrap(), 3);
        assert!(a.seq_line("line_index", 3).is_err());
        assert_ne!(a.seq_line("line_index", 0).unwrap(), 0);
        assert_eq!(a.seq_bytes(24), Some(sa as usize * 24..sa as usize * 24 + 24));
        drop(b);
        let d = p.lease(16).unwrap();
        assert_eq!(d.seq_slot().unwrap(), sb);
    }

    #[test]
    fn line_table_shape_is_checked() {
        let mut m = hybrid();
        let b = m.buffers.get_mut("line_index").unwrap();
        b.shape = vec![Dim::Const(4), Dim::Var("seqs".into())];
        let Err(e) = Pool::new(&m, 8, 20, 4) else { panic!("4 lines of 3 accepted") };
        assert!(e.to_string().contains("state holds 3"), "{e}");
        let b = m.buffers.get_mut("line_index").unwrap();
        b.shape = vec![Dim::Var("seqs".into())];
        let Err(e) = Pool::new(&m, 8, 20, 4) else { panic!("[seqs] accepted") };
        assert!(e.to_string().contains("[lines, seqs, w]"), "{e}");
        let b = m.buffers.get_mut("line_index").unwrap();
        b.shape = vec![Dim::Const(3), Dim::Var("seqs".into()), Dim::Const(8)];
        let p = Arc::new(Pool::new(&m, 8, 20, 4).unwrap().0);
        let a = p.lease(16).unwrap();
        assert_eq!(a.seq_width("line_index").unwrap(), 8);
        assert_eq!(a.seq_lines("line_index").unwrap(), 3);
    }

    // ---- memory: pages and slots out of one budget

    #[test]
    fn the_initial_layout_is_the_first_plan() {
        let (p, plan) = Pool::new(&hybrid(), 8, 20, 4).unwrap();
        // Every chunk mapped, one access grant per object, nothing unmapped.
        assert_eq!((plan.map.len(), plan.access.len(), plan.unmap.len(), plan.unmade.len()), (20, 8, 0, 0));
        assert_eq!(plan.made.len(), 8);
        assert_eq!((p.total(), p.slots(), p.pages_max(), p.slots_max()), (4, 4, 10, 6));
        let arenas: Vec<(&str, Kind, usize)> =
            p.pooled().iter().map(|a| (a.state.as_str(), a.kind, a.positions)).collect();
        assert_eq!(arenas, [("kv", Kind::Page, 20), ("gdn", Kind::Slot, 18)]);
        // The budget must hold the first slots.
        let Err(e) = Pool::new(&hybrid(), 8, 5, 4) else { panic!("4 slots out of 5 chunks") };
        assert!(e.to_string().contains("hold 1 sequence slots"), "{e}");
    }

    #[test]
    fn pages_come_from_free_slots() {
        let p = hybrid_pool();
        let a = p.lease(48).unwrap(); // pages 0..3, slot 1
        assert_eq!((p.used(), p.slots_used()), (3, 1));
        // Two pages wanted, one free: the highest free slot's three chunks make page 4.
        assert_eq!(p.lease(32).unwrap_err(), Denied::Remapping);
        let plan = p.take_pending().unwrap();
        assert_eq!(
            (plan.unmap.len(), plan.map.len(), &plan.made[..], &plan.unmade[..]),
            (3, 2, &[(Kind::Page, 4)][..], &[(Kind::Slot, 3)][..])
        );
        assert_eq!(plan.access, [(0, 8..10)]);
        // Until it lands, a caller sees neither the page arriving nor the slot leaving.
        assert_eq!(p.lease(32).unwrap_err(), Denied::Remapping);
        assert!(p.take_pending().is_none());
        assert_eq!((p.total(), p.slots()), (4, 3));
        p.complete(plan);
        let b = p.lease(32).unwrap();
        assert_eq!(
            (b.pages[..].to_vec(), b.seq_slot(), p.total(), p.slots(), p.slots_used()),
            (vec![3, 4], Some(2), 5, 3, 2)
        );
        drop((a, b));
        assert_eq!((p.used(), p.slots_used()), (0, 0));
    }

    #[test]
    fn a_slot_comes_from_free_pages() {
        // 24 chunks: 4 slots and 6 pages.
        let p = Arc::new(Pool::new(&hybrid(), 8, 24, 4).unwrap().0);
        let held: Vec<Lease> = (0..3).map(|_| p.lease(16).unwrap()).collect();
        assert_eq!((p.total(), p.used(), p.slots_used()), (6, 3, 3));
        // A page is free but every slot is held: two free pages give slot 4.
        assert_eq!(p.lease(16).unwrap_err(), Denied::Remapping);
        let plan = p.take_pending().unwrap();
        assert_eq!(
            (plan.unmade.clone(), plan.made.clone()),
            (vec![(Kind::Page, 5), (Kind::Page, 4)], vec![(Kind::Slot, 4)])
        );
        assert_eq!((plan.unmap.len(), plan.map.len()), (4, 3));
        p.complete(plan);
        let d = p.lease(16).unwrap();
        assert_eq!((d.seq_slot(), p.total(), p.slots(), p.slots_used()), (Some(4), 4, 5, 4));
        drop(held);
        // Chunks stay where they were: pages 4 and 5 are gone, slots 1..3 free.
        assert_eq!((p.total(), p.slots(), p.used()), (4, 5, 1));
    }

    #[test]
    fn nothing_free_to_give_is_busy_or_exceeds() {
        let p = hybrid_pool();
        let _a = p.lease(32).unwrap();
        let _b = p.lease(16).unwrap();
        let _c = p.lease(16).unwrap();
        // Every page and slot held: only an eviction helps.
        assert_eq!(p.lease(16).unwrap_err(), Denied::Busy);
        assert!(p.take_pending().is_none());
        // Five chunks: slot 0 and one page, room for no other slot ever.
        let small = Arc::new(Pool::new(&hybrid(), 8, 5, 1).unwrap().0);
        assert_eq!((small.slots_max(), small.total()), (1, 1));
        assert_eq!(small.lease(16).unwrap_err(), Denied::ExceedsPool);
    }

    // ---- checkpoints

    #[test]
    fn checkpoint_shares_pages_and_outlives_the_lease() {
        let p = pool();
        let mut a = p.lease(40).unwrap(); // 3 pages
        let (cp, copies) = p.checkpoint(&mut a, 32).unwrap(); // the first 2
        assert_eq!((cp.tokens(), cp.pages(), cp.seq_slot(), copies), (32, 2, None, Copies::default()));
        assert_eq!((a.shared, a.pages(), p.used()), (2, 3, 3));
        drop(a);
        // The checkpoint keeps its 2 pages; the lease's third came back.
        assert_eq!(p.used(), 2);
        assert_eq!(p.lease(32).unwrap().pages(), 2);
        drop(cp);
        assert_eq!(p.used(), 0);
    }

    #[test]
    fn checkpoints_along_one_lease_share_one_chain() {
        let p = pool();
        let mut a = p.lease(48).unwrap();
        let c1 = p.checkpoint(&mut a, 16).unwrap().0;
        let c2 = p.checkpoint(&mut a, 32).unwrap().0;
        let c3 = p.checkpoint(&mut a, 48).unwrap().0;
        // A shallower checkpoint taken after a deeper one finds its node up the chain.
        let c2b = p.checkpoint(&mut a, 20).unwrap().0;
        let (n1, n2, n2b) = (c1.chain.as_ref().unwrap(), c2.chain.as_ref().unwrap(), c2b.chain.as_ref().unwrap());
        assert!(Arc::ptr_eq(n2, n2b));
        assert!(Arc::ptr_eq(n1, n2.parent.as_ref().unwrap()));
        assert_eq!((a.shared, p.used()), (3, 3));
        drop(a);
        drop(c3);
        assert_eq!(p.used(), 2);
        drop(c2);
        assert_eq!(p.used(), 2); // c2b still holds page 2's node
        drop(c2b);
        assert_eq!(p.used(), 1);
        drop(c1);
        assert_eq!(p.used(), 0);
    }

    #[test]
    fn restore_shares_whole_pages_and_copies_a_partial_one() {
        let p = pool();
        let mut a = p.lease(40).unwrap();
        let [a0, a1, _] = a.pages[..] else { panic!() };
        // 20 tokens: page a0 whole, a1 holds positions 16..20.
        let (cp, _) = p.checkpoint(&mut a, 20).unwrap();
        drop(a);
        let (b, copies) = p.restore(&cp, cp.tokens(), 40).unwrap();
        let [b0, b1, b2] = b.pages[..] else { panic!() };
        // Shares a0, gets a fresh copy of a1 (a1 itself stays the checkpoint's), one more fresh page.
        assert_eq!((b0, b.shared, b.prefix(), b.tokens()), (a0, 1, 20, 48));
        assert_ne!(b1, a1);
        assert_eq!(copies, Copies { pages: vec![(a1, b1)], slot: None });
        assert_eq!(p.used(), 4);
        // Positions from 20 on are the lease's to write, into its own copy.
        assert_eq!(b.slot(20), b1 as i64 * 16 + 4);
        assert_eq!(b.slots(31..33), [b1 as i64 * 16 + 15, b2 as i64 * 16]);
        drop(cp);
        // a1 is only the checkpoint's: freed with it; a0 is still b's.
        assert_eq!(p.used(), 3);
        drop(b);
        assert_eq!(p.used(), 0);
    }

    #[test]
    fn restore_at_a_page_boundary_copies_nothing() {
        let p = pool();
        let mut a = p.lease(32).unwrap();
        let (cp, _) = p.checkpoint(&mut a, 32).unwrap();
        drop(a);
        let (b, copies) = p.restore(&cp, cp.tokens(), 33).unwrap();
        assert_eq!((b.prefix(), b.shared, b.pages(), copies), (32, 2, 3, Copies::default()));
        assert_eq!(&b.pages[..2], &chain_pages(&cp.chain)[..]);
        assert_eq!(p.used(), 3);
    }

    #[test]
    fn restore_at_a_shallower_page_of_a_pages_only_checkpoint() {
        let p = pool();
        let mut a = p.lease(40).unwrap();
        let [a0, a1, _] = a.pages[..] else { panic!() };
        let (cp, _) = p.checkpoint(&mut a, 36).unwrap(); // 3 pages, 4 tokens into the third
        drop(a);
        // 32 tokens: a0 and a1 shared, one fresh page.
        let (c, copies) = p.restore(&cp, 32, 33).unwrap();
        assert_eq!(
            (&c.pages[..2], c.shared, c.prefix(), c.pages(), copies),
            (&[a0, a1][..], 2, 32, 3, Copies::default())
        );
        drop(c);
        // 16 tokens: page a0 shared, nothing copied.
        let (b, copies) = p.restore(&cp, 16, 20).unwrap();
        assert_eq!((b.pages[0], b.shared, b.prefix(), b.pages(), copies), (a0, 1, 16, 2, Copies::default()));
        assert_ne!(b.pages[1], a1);
    }

    #[test]
    #[should_panic(expected = "restoring 4 tokens of a checkpoint of 10 (with a slot)")]
    fn a_stateful_checkpoint_restores_at_its_length_only() {
        let p = Arc::new(Pool::new(&hybrid(), 8, 26, 4).unwrap().0);
        let mut a = p.lease(16).unwrap();
        let (cp, _) = p.checkpoint(&mut a, 10).unwrap();
        let _ = p.restore(&cp, 4, 20);
    }

    #[test]
    fn fork_shares_whole_pages_and_copies_the_partial_one_and_the_slot() {
        let p = Arc::new(Pool::new(&hybrid(), 8, 26, 4).unwrap().0); // 4 slots, 7 pages
        let mut a = p.lease(40).unwrap(); // 3 pages
        let [a0, a1, _] = a.pages[..] else { panic!() };
        let sa = a.seq_slot().unwrap();
        // The parent is 20 tokens in: page a0 whole, a1 holds 16..20.
        let (b, copies) = p.fork(&mut a, 20, 40).unwrap();
        let [b0, b1, _] = b.pages[..] else { panic!() };
        assert_eq!((b0, b.shared, b.prefix(), a.shared), (a0, 1, 20, 1));
        assert_ne!(b1, a1);
        assert_eq!(copies, Copies { pages: vec![(a1, b1)], slot: Some((sa, b.seq_slot().unwrap())) });
        // Both keep writing their own copy of the page; a0 stays until both are gone.
        assert_eq!((a.slot(20), b.slot(20)), (a1 as i64 * 16 + 4, b1 as i64 * 16 + 4));
        assert_eq!((p.used(), p.slots_used()), (5, 2));
        drop(a);
        assert_eq!(p.used(), 3);
        drop(b);
        assert_eq!((p.used(), p.slots_used()), (0, 0));
        // At a page boundary nothing is copied; before the first boundary nothing is shared.
        let mut a = p.lease(32).unwrap();
        let (b, copies) = p.fork(&mut a, 16, 32).unwrap();
        assert_eq!((b.shared, copies.pages.len(), b.pages[0]), (1, 0, a.pages[0]));
        let (c, copies) = p.fork(&mut a, 3, 32).unwrap();
        assert_eq!((c.shared, c.prefix(), copies.pages), (0, 3, vec![(a.pages[0], c.pages[0])]));
        assert_eq!(p.fork(&mut a, 3, 32).unwrap_err(), Denied::Busy);
    }

    #[test]
    fn wake_is_a_fresh_lease_with_a_prefix() {
        let p = pool();
        let l = p.wake(20, 40).unwrap();
        assert_eq!((l.pages(), l.shared, l.prefix(), l.chain.is_none(), p.used()), (3, 0, 20, true, 3));
        assert_eq!(l.page_ids().len(), 3);
        assert_eq!(l.slot(20), l.pages[1] as i64 * 16 + 4);
    }

    #[test]
    #[should_panic(expected = "inside the shared prefix")]
    fn restored_lease_refuses_its_prefix() {
        let p = pool();
        let mut a = p.lease(32).unwrap();
        let (cp, _) = p.checkpoint(&mut a, 20).unwrap();
        let (b, _) = p.restore(&cp, cp.tokens(), 40).unwrap();
        b.slot(19);
    }

    #[test]
    fn restore_denials() {
        let p = pool();
        let mut a = p.lease(16).unwrap();
        let (cp, _) = p.checkpoint(&mut a, 16).unwrap();
        assert_eq!(p.restore(&cp, cp.tokens(), 49).unwrap_err(), Denied::ExceedsRow { limit: 48 });
        let _b = p.lease(48).unwrap();
        // 3 more pages held: the one fresh page a 17-token restore needs is gone.
        assert_eq!(p.restore(&cp, cp.tokens(), 17).unwrap_err(), Denied::Busy);
        drop(a);
        assert_eq!(p.restore(&cp, cp.tokens(), 17).unwrap_err(), Denied::Busy);
    }

    #[test]
    fn hybrid_checkpoint_copies_the_slot_and_retire_moves_it() {
        // 26 chunks: 4 slots and 7 pages.
        let p = Arc::new(Pool::new(&hybrid(), 8, 26, 4).unwrap().0);
        let mut a = p.lease(16).unwrap();
        let sa = a.seq_slot().unwrap();
        let (cp, copies) = p.checkpoint(&mut a, 10).unwrap();
        let sc = cp.seq_slot().unwrap();
        assert_ne!(sc, sa);
        assert_eq!((copies, p.slots_used()), (Copies { pages: vec![], slot: Some((sa, sc)) }, 2));
        // A slot each for a and cp: one left; a restore takes it and copies the state in.
        let (b, copies) = p.restore(&cp, cp.tokens(), 17).unwrap();
        let sb = b.seq_slot().unwrap();
        assert_eq!((copies.slot, b.prefix(), p.slots_used()), (Some((sc, sb)), 10, 3));
        // No slot left; of the four free pages the restore needs two, the
        // other two hold a slot's worth of chunks.
        assert_eq!(p.restore(&cp, cp.tokens(), 17).unwrap_err(), Denied::Remapping);
        land(&p);
        let (c, copies) = p.restore(&cp, cp.tokens(), 17).unwrap();
        assert_eq!((copies.slot, p.slots(), p.slots_used(), p.total()), (Some((sc, 4)), 5, 4, 5));
        drop((b, c));
        // Retiring a moves its slot to the checkpoint: no copy; its one page is the same one cp shares.
        let a2 = p.retire(a, 10);
        assert_eq!((a2.seq_slot(), a2.pages(), p.slots_used(), p.used()), (Some(sa), 1, 2, 1));
        drop(cp);
        drop(a2);
        assert_eq!((p.slots_used(), p.used()), (0, 0));
    }

    #[test]
    fn checkpoint_without_a_free_slot_is_busy() {
        let p = hybrid_pool();
        let mut a = p.lease(16).unwrap();
        let _b = p.lease(16).unwrap();
        let _c = p.lease(16).unwrap();
        assert_eq!(p.checkpoint(&mut a, 16).unwrap_err(), Denied::Busy);
        // Retiring never needs a slot.
        let cp = p.retire(a, 16);
        assert!(cp.seq_slot().is_some());
    }

    #[test]
    fn retire_returns_pages_past_len() {
        let p = pool();
        let a = p.lease(48).unwrap();
        let cp = p.retire(a, 17); // pages 0..2 kept
        assert_eq!((cp.tokens(), cp.pages(), p.used()), (17, 2, 2));
    }

    #[test]
    fn a_long_chain_drops_without_recursion() {
        let m = Manifest::from_json(
            r#"{
            "schema_version": 4, "model": "t", "vars": {"tokens": {"max": 8}, "seqs": {"max": 2}},
            "states": {"kv": {"bytes_per_token": 1}},
            "buffers": {
                "block_table": {"kind": "input", "dtype": "i32", "shape": ["seqs", 200000], "domain": {"index_into": "kv", "stride": 1}}
            },
            "modules": {}, "ops": {}, "programs": {}
        }"#,
        )
        .unwrap();
        let p = Arc::new(Pool::new(&m, 1, 200_000, 0).unwrap().0);
        let mut a = p.lease(200_000).unwrap();
        let cp = p.checkpoint(&mut a, 200_000).unwrap().0;
        drop(a);
        assert_eq!(p.used(), 200_000);
        drop(cp);
        assert_eq!(p.used(), 0);
    }

    /// Random leases, checkpoints, restores, retirements, drops and remap
    /// landings against a model that only asks which pages and slots are
    /// reachable from a live handle and which chunks are mapped where:
    /// held, free and absent partition the objects; a chunk is free or at
    /// one position; a position is mapped exactly when an object exists
    /// over it.
    #[test]
    fn ownership_partitions_the_pool() {
        let m = hybrid();
        // 26 chunks: slots 0..3 (9), 8 pages (16), one spare.
        let p = Arc::new(Pool::new(&m, 8, 26, 3).unwrap().0);
        let mut leases: Vec<Lease> = Vec::new();
        let mut cps: Vec<Checkpoint> = Vec::new();
        let mut remaps = 0;
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        let mut rand = move |n: usize| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 33) as usize % n
        };
        for _ in 0..6000 {
            match rand(8) {
                0 | 6 => {
                    if let Ok(l) = p.lease(1 + rand(48)) {
                        leases.push(l);
                    }
                }
                1 if !leases.is_empty() => {
                    let at = rand(leases.len());
                    let l = &mut leases[at];
                    let len = 1 + rand(l.tokens());
                    if let Ok((cp, c)) = p.checkpoint(l, len) {
                        assert_eq!(c.slot.map(|(a, _)| a), l.seq_slot());
                        cps.push(cp);
                    }
                }
                2 if !cps.is_empty() => {
                    let cp = &cps[rand(cps.len())];
                    if cp.tokens() < 48 {
                        if let Ok((l, c)) = p.restore(cp, cp.tokens(), cp.tokens() + 1 + rand(48 - cp.tokens())) {
                            assert_eq!(
                                (l.prefix(), c.pages.len()),
                                (cp.tokens(), (!cp.tokens().is_multiple_of(16)) as usize)
                            );
                            leases.push(l);
                        }
                    }
                }
                3 if !leases.is_empty() => {
                    let l = leases.swap_remove(rand(leases.len()));
                    let len = 1 + rand(l.tokens());
                    cps.push(p.retire(l, len));
                }
                4 if !leases.is_empty() => {
                    leases.swap_remove(rand(leases.len()));
                }
                5 if !cps.is_empty() => {
                    cps.swap_remove(rand(cps.len()));
                }
                7 => {
                    if let Some(plan) = p.take_pending() {
                        remaps += 1;
                        p.complete(plan);
                    }
                }
                _ => {}
            }
            let mut held: Vec<i32> = Vec::new();
            for l in &leases {
                held.extend(&l.pages[l.shared..]);
                held.extend(chain_pages(&l.chain));
            }
            for cp in &cps {
                held.extend(chain_pages(&cp.chain));
            }
            held.sort();
            held.dedup();
            let mut slots: Vec<i32> = leases.iter().map(|l| l.seq_slot().unwrap()).collect();
            slots.extend(cps.iter().map(|c| c.seq_slot().unwrap()));
            slots.sort();
            let mut uniq = slots.clone();
            uniq.dedup();
            assert_eq!(uniq, slots, "a slot has one holder");
            let g = lock(&p.inner);
            let by = |v: &[Status], s: Status| -> Vec<i32> {
                v.iter().enumerate().filter(|(_, &x)| x == s).map(|(i, _)| i as i32).collect()
            };
            assert_eq!(by(&g.pages, Status::Held), held);
            assert_eq!(by(&g.pages, Status::Free), g.free_pages.iter().copied().collect::<Vec<_>>());
            assert_eq!(by(&g.slots, Status::Held)[1..], slots[..]);
            assert_eq!(by(&g.slots, Status::Free), g.free_slots.iter().copied().collect::<Vec<_>>());
            // The pending plan's objects are on their way.
            if let Some(plan) = &g.pending {
                for (k, o) in &plan.made {
                    let v = if *k == Kind::Page { &g.pages } else { &g.slots };
                    assert_eq!(v[*o as usize], Status::Arriving);
                }
                for (k, o) in &plan.unmade {
                    let v = if *k == Kind::Page { &g.pages } else { &g.slots };
                    assert_eq!(v[*o as usize], Status::Leaving);
                }
            } else {
                assert!(!g.pages.contains(&Status::Arriving) || g.in_flight);
            }
            // Chunks: free or at one position, all accounted for.
            let mut ids: Vec<u32> = g.chunks.free_ids().to_vec();
            ids.extend(g.chunks.mapped().iter().map(|&(_, _, c)| c));
            ids.sort();
            assert_eq!(ids, (0..26).collect::<Vec<u32>>());
            // A position is mapped exactly when an object that exists or
            // is arriving covers it (a leaving one has let go already).
            for (a, ar) in p.pooled().iter().enumerate() {
                let objects = if ar.kind == Kind::Page { &g.pages } else { &g.slots };
                let mut cover = vec![0u16; ar.positions];
                for (o, s) in objects.iter().enumerate() {
                    if s.exists() || *s == Status::Arriving {
                        for q in g.chunks.interval(a, o) {
                            cover[q] += 1;
                        }
                    }
                }
                assert_eq!(g.chunks.users(a), &cover[..], "arena {a}");
                let mapped: Vec<usize> = g.chunks.mapped().iter().filter(|m| m.0 == a).map(|m| m.1).collect();
                let covered: Vec<usize> = cover.iter().enumerate().filter(|(_, &u)| u > 0).map(|(q, _)| q).collect();
                assert_eq!(mapped, covered, "arena {a}");
            }
        }
        assert!(remaps > 20, "{remaps} remaps landed");
    }

    #[test]
    fn slot_only_leases_move_the_slot_alone() {
        let p = hybrid_pool();
        let mut l = p.lease_slot().unwrap();
        assert_eq!((l.pages(), l.tokens(), l.paged(), l.seq_slot().is_some()), (0, 0, false, true));
        assert_eq!((p.used(), p.slots_used()), (0, 1));
        // A checkpoint copies the slot and holds no page, at any length.
        let (cp, c) = p.checkpoint(&mut l, 5).unwrap();
        assert_eq!((cp.tokens(), cp.pages(), cp.paged(), cp.nodes().len()), (5, 0, false, 0));
        assert_eq!((c.pages.len(), c.slot.map(|(a, _)| a)), (0, l.seq_slot()));
        // Restoring is at its own length and gives a slot-only lease with that prefix.
        let (r, c) = p.restore(&cp, 5, 100).unwrap();
        assert_eq!((r.pages(), r.prefix(), c.pages.len(), c.slot.map(|(a, _)| a)), (0, 5, 0, cp.seq_slot()));
        assert_eq!(p.slots_used(), 3);
        drop(r);
        drop(cp);
        // A fork the same way.
        let (f, c) = p.fork(&mut l, 9, 100).unwrap();
        assert_eq!((f.pages(), f.prefix(), c.slot.map(|(a, _)| a)), (0, 9, l.seq_slot()));
        drop(f);
        // Retiring hands the slot over as it is.
        let slot = l.seq_slot();
        let cp = p.retire(l, 7);
        assert_eq!((cp.tokens(), cp.pages(), cp.seq_slot()), (7, 0, slot));
        drop(cp);
        assert_eq!((p.used(), p.slots_used()), (0, 0));
        // A woken slot-only lease: a slot and the prefix, no page.
        let w = p.wake_slot(11).unwrap();
        assert_eq!((w.pages(), w.prefix(), w.seq_slot().is_some()), (0, 11, true));
    }
}
