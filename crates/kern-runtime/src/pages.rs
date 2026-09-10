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

/// Chunks that hold `tokens` of every paged state and `first_slots` of
/// every per-sequence state. Every pooled state is its own arena of whole
/// chunks, so each rounds up on its own: rounding the sum once leaves the
/// last page or slot of a state short of a chunk.
pub fn chunks_for(m: &Manifest, tokens: u64, first_slots: u64, chunk: u64) -> u64 {
    m.states
        .values()
        .map(|s| (tokens * s.bytes_per_token).div_ceil(chunk) + (first_slots * s.bytes_per_seq).div_ceil(chunk))
        .sum()
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
    pub(crate) fn remapped(&self) -> bool {
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
    //! The pool against a reference model of its chunks: which pages and
    //! slots a handle can reach, and which chunks are mapped where. The
    //! model reads the ledger the public API keeps private; everything the
    //! API shows is tested in `tests/pool.rs`.

    use super::*;

    /// One paged state of 16 bytes a page, plus a recurrent state of 3
    /// lines of 8 bytes per sequence and its line table.
    fn hybrid() -> Manifest {
        Manifest::from_json(
            r#"{
            "schema_version": 5, "model": "t", "vars": {"tokens": {"max": 8}, "seqs": {"max": 2}},
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
}
