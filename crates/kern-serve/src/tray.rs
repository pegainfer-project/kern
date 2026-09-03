//! The tray: `n` runtimes, one per GPU, driven from one thread as one
//! machine.
//!
//! A manifest with a `topology` is SPMD — every rank runs the same launch
//! list, and a collective inside it waits for every other rank — so the
//! ranks of a tray must be given the same program, the same var values
//! and the same number of rows at every step, or one of them spins on an
//! epoch that never comes. This module is the only place that holds a
//! [`Runtime`]: the scheduler above it never sees a rank, a lease or a
//! per-rank input. It sees rows, snapshots and steps, and every one of
//! those is defined over the whole tray, so the lockstep invariant is a
//! property of the type layer rather than of the scheduler's discipline.
//!
//! # Rows and their owners
//!
//! The `tp` group (`t` ranks; 1 without one) runs one batch: its rows are
//! spread over the group's ranks, every rank holding a slice of every
//! row's recurrent state (a head-sharded KDA state) and the pages of the
//! rows it owns (MLA attention is by row: the owner alone reads them). A
//! [`Row`] is therefore one paged [`Lease`] on its owner and a slot-only
//! lease on each of the owner's peers, and every act on it — lease, fork,
//! checkpoint, retire, park, wake — is the same act on every member of
//! the group, all or nothing: a `Row` is only ever constructed once every
//! rank has said yes, and an early return drops what the earlier ranks
//! handed out. The decisions the ranks make are independent (each pool is
//! its own accounting; slot numbers differ across ranks and are never
//! compared), so nothing is rolled back but by `Drop`. Parking is the one
//! two-step act, since a park in flight cannot be undone: room is found on
//! every rank first ([`Runtime::room`]), and bytes move only once all of
//! it is. An owner is chosen when a row is leased — the open rank with the
//! fewest pages in use — and is fixed for the row's life and everything
//! that descends from it: pages live on that GPU, a parked copy lives in
//! that GPU's pinned block.
//!
//! # Staging
//!
//! A step is a list of [`Cell`]s, one per row of the tray: the row, the
//! tokens it feeds this step and its position. [`Tray::stage`] lays them
//! out per rank. Rows per rank `b` is one bucket for the whole tray (the
//! most rows any rank has, rounded up), padded on every rank with the
//! rank's pad lease — a page and a slot no sequence owns, whose junk
//! nobody reads. Which buffer carries what, and which inputs span the
//! group and which are a rank's own, is the manifest's [`Protocol`]: a
//! fill over the tray axis carries the tray batch — this rank's rows
//! first, then the other members' in group order from it, the layout the
//! collectives assume — and one over the rows or sequences axis carries
//! this rank's rows alone. The tokens fed and the line tables of a
//! sharded state are of the first kind in a tray manifest; the slots, the
//! sequence lengths and the page tables are always of the second. Outputs
//! likewise: a `tokens` fill over the tray axis holds this rank's rows in
//! its first block, so every cell is read from its owner. A manifest with
//! an `error` fill declares the collectives' error word; it is read after
//! every step and a nonzero value is a failed step, never a silent one.
//!
//! [`Staged`] borrows the tray for as long as the step lives: nothing can
//! lease, fork or stage again until the outputs are read and it drops,
//! which is what keeps "stage, run, read" one indivisible motion.

use std::collections::BTreeMap;
use std::ops::Range;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use kern_manifest::protocol::{Axis, Filled, Forward};
use kern_manifest::types::{Fill, Manifest};
use kern_manifest::{Protocol, Verified};
use kern_run::le_bytes_i32;
use kern_runtime::{
    Checkpoint, Denied, Error, GroupRank, Kept, Lease, Parked, PeerHandle, Room, Runtime, Topology, Waking,
};
use tracing::info;

use crate::logline;

/// A rank of the tray: an index into its runtimes. Only the tray makes one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Rank(usize);

impl Rank {
    pub fn index(self) -> usize {
        self.0
    }
}

/// How `n` ranks split into tray batch groups of `t`: rank `q` is member
/// `q % t` of group `q / t`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Groups {
    n: usize,
    t: usize,
}

impl Groups {
    fn member(self, q: usize) -> usize {
        q % self.t
    }

    /// The ranks of `q`'s group, in member order.
    fn members(self, q: usize) -> Range<usize> {
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

/// One sequence's pieces across its owner's group, one per member in
/// member order: leases (a [`Row`]), checkpoints (a [`Snapshot`]),
/// parked copies (a [`Sleeping`]) or wakes in flight (a [`Rising`]). The
/// owner's piece, at `me`, is the paged one; the peers' hold a slot
/// alone. Dropping it returns all of them.
pub struct Group<X> {
    owner: Rank,
    me: usize,
    parts: Vec<X>,
}

pub type Row = Group<Lease>;
pub type Snapshot = Group<Checkpoint>;
pub type Sleeping = Group<Parked>;
pub type Rising = Group<Waking>;

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

/// One row's part of a step: the tokens it feeds at `pos..`.
pub struct Cell<'a> {
    pub row: &'a Row,
    pub ids: Vec<i64>,
    pub pos: usize,
}

/// Which cells each rank owns, in cell order, and the bucket they were
/// padded to: the pure half of [`Tray::stage`].
#[derive(Debug, Clone, PartialEq, Eq)]
struct Layout {
    /// `own[q]` = indices of the cells rank `q` owns.
    own: Vec<Vec<usize>>,
    /// Rows per rank after padding.
    b: usize,
    /// Tokens per cell.
    per: usize,
    cells: usize,
}

impl Layout {
    fn new(owners: &[usize], per: usize, n: usize, bucket: impl Fn(usize) -> usize) -> Layout {
        let mut own = vec![Vec::new(); n];
        for (i, &q) in owners.iter().enumerate() {
            own[q].push(i);
        }
        let most = own.iter().map(Vec::len).max().unwrap_or(0);
        Layout { own, b: bucket(most).max(1), per, cells: owners.len() }
    }

    /// Rank `q`'s rows: its own cells, then `None` for each padding row.
    fn rows(&self, q: usize) -> impl Iterator<Item = Option<usize>> + '_ {
        (0..self.b).map(move |j| self.own[q].get(j).copied())
    }
}

/// The `Runtime` holds raw CUDA handles and is used from the tray's thread
/// only; every entry point rebinds the context to the calling thread, so
/// loading it elsewhere and moving it there once is sound.
struct Sent(Runtime);
#[allow(unsafe_code)]
unsafe impl Send for Sent {}

pub struct Tray {
    ranks: Vec<Runtime>,
    groups: Groups,
    protocol: Protocol,
    /// One page and slot per rank no sequence owns: padding rows write here.
    pad: Vec<Lease>,
}

// See `Sent`: the tray lives on the scheduler thread.
#[allow(unsafe_code)]
unsafe impl Send for Tray {}

impl Tray {
    /// Load the manifest on every GPU of `gpus` with its place in the
    /// topology (`tp` splits the ranks into consecutive groups; every
    /// other group spans them all), bind each rank's weights
    /// (`weights_of` names them for a rank's topology), connect the peers,
    /// run what the manifest runs once, reserve `host_bytes` of pinned
    /// memory per rank and lease the pad.
    pub fn load(
        m: &Verified,
        kernels: &Path,
        gpus: &[usize],
        capacity: Option<u64>,
        weights_of: &(dyn Fn(&Topology) -> Result<Vec<PathBuf>> + Sync),
        host_bytes: u64,
    ) -> Result<Tray> {
        let protocol = Protocol::check(m)?;
        let n = gpus.len();
        anyhow::ensure!(n >= 1, "no GPUs");
        let t = m.group_size("tp").unwrap_or(1) as usize;
        if !n.is_multiple_of(t) {
            bail!("{n} GPUs do not split into tray batch groups of {t} (the manifest's `tp` group)");
        }
        if let Some(topo) = &m.topology {
            for (g, &size) in &topo.groups {
                if g != "tp" && size as usize != n {
                    bail!("topology group `{g}` has {size} members; this process drives {n} GPUs and spans it with all of them");
                }
            }
        }
        let groups = Groups { n, t };
        let topology = |q: usize| -> Topology {
            let mut topo = Topology::default();
            for (g, &size) in m.topology.iter().flat_map(|t| &t.groups) {
                let index = if g == "tp" { groups.member(q) } else { q } as u64;
                topo.groups.insert(g.clone(), GroupRank { index, size });
            }
            topo
        };
        let has_topology = m.topology.is_some();
        let t0 = std::time::Instant::now();
        let loaded: Vec<Result<Sent>> = std::thread::scope(|s| {
            let handles: Vec<_> = gpus
                .iter()
                .enumerate()
                .map(|(q, &gpu)| {
                    let topo = topology(q);
                    s.spawn(move || -> Result<Sent> {
                        let mut rt = Runtime::load(m, kernels, gpu, capacity, has_topology.then_some(&topo))
                            .with_context(|| format!("rank {q} on gpu {gpu}"))?;
                        let files = weights_of(&topo)?;
                        let maps = files
                            .iter()
                            .map(|f| {
                                let file =
                                    std::fs::File::open(f).with_context(|| format!("weights {}", f.display()))?;
                                // Mapped, not read: the ranks share the page cache and no rank
                                // holds a copy of its shard in DRAM.
                                #[allow(unsafe_code)]
                                let map = unsafe { memmap2::Mmap::map(&file) }
                                    .with_context(|| format!("mapping weights {}", f.display()))?;
                                Ok(map)
                            })
                            .collect::<Result<Vec<_>>>()?;
                        let blobs: Vec<&[u8]> = maps.iter().map(|m| &m[..]).collect();
                        rt.load_weights(&blobs).with_context(|| format!("rank {q}: binding weights"))?;
                        Ok(Sent(rt))
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().expect("a rank's load thread panicked")).collect()
        });
        let mut ranks: Vec<Runtime> = loaded.into_iter().map(|r| r.map(|s| s.0)).collect::<Result<_>>()?;
        info!(ranks = n, tray = t, load_s = logline::secs(t0.elapsed()), "ranks loaded, weights bound");
        if let Some(topo) = &m.topology {
            let handles: Vec<BTreeMap<String, PeerHandle>> =
                ranks.iter().map(Runtime::export_handles).collect::<Result<_, _>>()?;
            for g in topo.groups.keys() {
                for q in 0..n {
                    let members: Vec<BTreeMap<String, PeerHandle>> = if g == "tp" {
                        groups.members(q).map(|r| handles[r].clone()).collect()
                    } else {
                        handles.clone()
                    };
                    ranks[q].import_peers(g, &members).with_context(|| format!("rank {q}: peers of `{g}`"))?;
                }
            }
            for (q, rt) in ranks.iter().enumerate() {
                let pending = rt.pending_peers();
                if !pending.is_empty() {
                    bail!("rank {q}: peer buffers {pending:?} still unfilled after every group was imported");
                }
            }
        }
        // Once after load, the peers mapped: a tray manifest's setup (the
        // allreduce's Lamport stages are poisoned, not zeroed).
        let env = protocol.env(1, 1, t as u64);
        for p in &protocol.once {
            for (q, rt) in ranks.iter().enumerate() {
                rt.run(p, &env).with_context(|| format!("rank {q}: `{p}`"))?;
            }
        }
        if host_bytes > 0 {
            let t0 = std::time::Instant::now();
            for (q, rt) in ranks.iter_mut().enumerate() {
                rt.reserve_host(host_bytes).with_context(|| format!("rank {q}: reserving the host tier"))?;
            }
            info!(gib_per_rank = host_bytes >> 30, reserve_s = logline::secs(t0.elapsed()), "host tiers reserved");
        }
        let pad = ranks
            .iter_mut()
            .enumerate()
            .map(|(q, rt)| rt.lease(1).map_err(|e| anyhow::anyhow!("rank {q}: no page for the padding rows: {e}")))
            .collect::<Result<Vec<_>>>()?;
        for (q, rt) in ranks.iter().enumerate() {
            if rt.pages_used() == rt.pages_total() {
                bail!("rank {q}: capacity {} tokens holds one page; nothing left to serve from", rt.capacity());
            }
        }
        Ok(Tray { ranks, groups, protocol, pad })
    }

    pub fn manifest(&self) -> &Manifest {
        &self.ranks[0].manifest
    }

    pub fn protocol(&self) -> &Protocol {
        &self.protocol
    }

    /// Sequences one rank may run at once: the manifest's bound, and its
    /// share of the tray batch when there is one.
    pub fn seqs_max(&self) -> usize {
        seqs_max(&self.protocol, self.groups.t)
    }

    /// Ranks driven.
    pub fn len(&self) -> usize {
        self.groups.n
    }

    /// Ranks per tray batch group.
    pub fn group_size(&self) -> usize {
        self.groups.t
    }

    // ---- accounting, summed over the ranks.

    /// Page unit in tokens.
    pub fn page(&self) -> usize {
        self.ranks[0].page() as usize
    }

    /// Pages held by rows, every rank together (the pads excluded).
    pub fn pages_used(&self) -> usize {
        self.ranks.iter().map(Runtime::pages_used).sum::<usize>() - self.pad.iter().map(Lease::pages).sum::<usize>()
    }

    /// Pages rows can hold, every rank together (the pads excluded).
    pub fn pages_total(&self) -> usize {
        self.ranks.iter().map(Runtime::pages_total).sum::<usize>() - self.pad.iter().map(Lease::pages).sum::<usize>()
    }

    /// Longest sequence one page-table row can address, on any rank.
    pub fn max_seq_tokens(&self) -> usize {
        self.ranks.iter().map(Runtime::max_seq_tokens).min().unwrap_or(0)
    }

    pub fn has_seq_state(&self) -> bool {
        self.ranks[0].has_seq_state()
    }

    /// (slots that exist, slots held by rows) over every rank; (0, 0)
    /// without per-sequence states.
    pub fn seq_slots(&self) -> (usize, usize) {
        if !self.has_seq_state() {
            return (0, 0);
        }
        (
            self.ranks.iter().map(Runtime::seq_slots).sum(),
            self.ranks.iter().map(Runtime::seq_slots_used).sum::<usize>() - self.pad.len(),
        )
    }

    pub fn remaps(&self) -> u64 {
        self.ranks.iter().map(Runtime::remaps).sum()
    }

    /// (bytes used, bytes reserved) of the host tiers together, when there are any.
    pub fn host_tier(&self) -> Option<(u64, u64)> {
        self.ranks.iter().map(Runtime::host_tier).try_fold((0, 0), |(u, r), t| t.map(|(a, b)| (u + a, r + b)))
    }

    // ---- rows: every act is the same act on every member of a group.

    /// `f(rank, member, piece)` on every member of `g`'s group in member
    /// order, as a group of what it returns; the first `Err` drops what the
    /// earlier members handed out.
    fn each<X, Y>(
        &mut self,
        g: Group<X>,
        mut f: impl FnMut(&mut Runtime, usize, X) -> Result<Y, Error>,
    ) -> Result<Group<Y>, Error> {
        let Group { owner, me, parts } = g;
        let base = self.groups.members(owner.0).start;
        let parts: Result<Vec<Y>, Error> =
            parts.into_iter().enumerate().map(|(m, x)| f(&mut self.ranks[base + m], m, x)).collect();
        Ok(Group { owner, me, parts: parts? })
    }

    /// A fresh row of `tokens` on the open rank with the fewest pages in
    /// use (`open` says which ranks can take another row), a slot-only
    /// lease on each of its peers. [`Denied::Busy`] when no rank is open.
    pub fn lease(&mut self, tokens: usize, open: impl Fn(Rank) -> bool) -> Result<Row, Error> {
        let owner = (0..self.groups.n)
            .filter(|&q| open(Rank(q)))
            .min_by_key(|&q| (self.ranks[q].pages_used(), q))
            .ok_or(Error::Denied(Denied::Busy))?;
        let me = self.groups.member(owner);
        let slots = Group { owner: Rank(owner), me, parts: vec![(); self.groups.t] };
        self.each(slots, |rt, m, ()| if m == me { rt.lease(tokens) } else { rt.lease_slot() })
    }

    /// A row continuing from the first `len` tokens of `snap` with room
    /// for `tokens`, on the snapshot's owner; the peers continue their
    /// slots at the snapshot's length.
    pub fn lease_from(&mut self, snap: &Snapshot, len: usize, tokens: usize) -> Result<Row, Error> {
        self.each(snap.by_ref(), |rt, m, cp| rt.lease_from(cp, if m == snap.me { len } else { cp.tokens() }, tokens))
    }

    /// A child of `parent` at its first `len` tokens with room for
    /// `tokens`, on every member. Nothing in `serve` branches a live
    /// sequence yet; the harness does (`k3_golden --fork`), and a session
    /// fork request would land here.
    #[allow(dead_code)]
    pub fn fork(&mut self, parent: &mut Row, len: usize, tokens: usize) -> Result<Row, Error> {
        self.each(parent.by_mut(), |rt, _, l| rt.fork(l, len, tokens))
    }

    /// The first `len` tokens of `row` as a snapshot it keeps running past.
    pub fn checkpoint(&mut self, row: &mut Row, len: usize) -> Result<Snapshot, Error> {
        self.each(row.by_mut(), |rt, _, l| rt.checkpoint(l, len))
    }

    /// The first `len` tokens of a finished row as a snapshot; nothing is copied.
    pub fn retire(&mut self, row: Row, len: usize) -> Snapshot {
        let base = self.groups.members(row.owner.0).start;
        let mut m = 0..;
        row.map(|l| self.ranks[base + m.next().unwrap()].retire(l, len))
    }

    /// Park `snap` in its members' host tiers: room on every one first,
    /// then the copies; `Err(snap)` hands it back untouched when any
    /// member is short of room.
    pub fn park(&mut self, snap: Snapshot) -> Result<std::result::Result<Sleeping, Snapshot>, Error> {
        let rooms = self.each(snap, |rt, _, cp| rt.room(cp))?;
        if rooms.parts.iter().any(Result::is_err) {
            return Ok(Err(rooms.map(|r| r.map_or_else(|cp| cp, Room::into_checkpoint))));
        }
        self.each(rooms, |rt, _, r| rt.park(r.unwrap_or_else(|_| unreachable!("every member found room")))).map(Ok)
    }

    /// Wake the first `len` tokens of `sleeping` into a row with room for
    /// `tokens`, on every member; the copies are in flight until
    /// [`Tray::awake`] says otherwise.
    pub fn wake(&mut self, sleeping: &Sleeping, len: usize, tokens: usize) -> Result<Rising, Error> {
        self.each(sleeping.by_ref(), |rt, m, p| rt.wake(p, if m == sleeping.me { len } else { p.tokens() }, tokens))
    }

    /// The row of a wake whose copies have all landed; `Err(r)` while any
    /// is still in flight. Does not block.
    pub fn awake(&mut self, r: Rising) -> Result<std::result::Result<Row, Rising>, Error> {
        for (m, q) in self.groups.members(r.owner.0).enumerate() {
            if !self.ranks[q].landed(&r.parts[m])? {
                return Ok(Err(r));
            }
        }
        // A landed event stays landed.
        self.each(r, |rt, _, w| rt.awake(w).map(|l| l.unwrap_or_else(|_| unreachable!("landed")))).map(Ok)
    }

    // ---- steps.

    /// Lay a step out on every rank and write its inputs: rows per rank
    /// is `bucket` of the most any rank owns. Every cell feeds the same
    /// number of tokens.
    pub fn stage(&mut self, cells: &[Cell<'_>], bucket: impl Fn(usize) -> usize) -> Result<Staged<'_>> {
        let per = cells.first().map_or(1, |c| c.ids.len());
        if per == 0 || cells.iter().any(|c| c.ids.len() != per) {
            bail!("every cell of a step feeds the same number of tokens, at least one");
        }
        let owners: Vec<usize> = cells.iter().map(|c| c.row.owner.0).collect();
        let layout = Layout::new(&owners, per, self.groups.n, bucket);
        let (b, t) = (layout.b, self.groups.t);
        if b > self.seqs_max() {
            bail!("{b} rows per rank, the manifest allows {}", self.seqs_max());
        }
        let env = self.protocol.env(b as u64, per as u64, t as u64);
        for q in 0..self.groups.n {
            self.stage_rank(q, cells, &layout, &env)?;
        }
        Ok(Staged { tray: self, layout, env })
    }

    /// Rank `q`'s inputs for a step: every fill, the page tables and the
    /// line tables (see the module doc for which span the group).
    fn stage_rank(&mut self, q: usize, cells: &[Cell<'_>], l: &Layout, env: &BTreeMap<String, u64>) -> Result<()> {
        let (per, b, me) = (l.per, l.b, self.groups.member(q));
        let p = &self.protocol;
        let pad = &self.pad[q];
        // Every rank's slot for a row is the lease at its own member index.
        let lease_on = |i: usize| -> &Lease { &cells[i].row.parts[me] };
        // The ranks whose rows a buffer over `axis` holds, in block order.
        let blocks = |axis: Axis| -> Vec<usize> {
            if axis == Axis::Tray {
                self.groups.blocks(q).collect()
            } else {
                vec![q]
            }
        };
        let mut writes: Vec<(&Filled, Vec<i64>)> = Vec::new();
        for f in &p.fills {
            let v: Vec<i64> = match (f.fill, f.axis) {
                (Fill::Token, Axis::Rows | Axis::Tray) => blocks(f.axis)
                    .iter()
                    .flat_map(|&r| l.rows(r))
                    .flat_map(|i| i.map_or(vec![0; per], |i| cells[i].ids.clone()))
                    .collect(),
                // Each sequence's first token, the anchor a drafting program
                // splices its own rows from.
                (Fill::Token, Axis::Groups) => l.rows(q).map(|i| i.map_or(0, |i| cells[i].ids[0])).collect(),
                (Fill::Position, _) => {
                    l.rows(q).flat_map(|i| (0..per).map(move |j| (i.map_or(0, |i| cells[i].pos) + j) as i64)).collect()
                }
                (Fill::Slot, _) => l
                    .rows(q)
                    .flat_map(|i| {
                        let (pos, lease) = i.map_or((0, pad), |i| (cells[i].pos, cells[i].row.own()));
                        lease.slots(pos..pos + per)
                    })
                    .collect(),
                (Fill::SeqLen, _) => l.rows(q).map(|i| (i.map_or(0, |i| cells[i].pos) + per) as i64).collect(),
                (Fill::CuSeqlens, _) => (0..=b as i64).map(|i| i * per as i64).collect(),
                (Fill::Tokens | Fill::Count | Fill::Error, _) => continue,
                (Fill::Token, Axis::Fixed(_)) => unreachable!("the protocol checks fill shapes"),
            };
            writes.push((f, v));
        }
        let mut tables: Vec<(&str, Vec<i32>)> = Vec::with_capacity(p.page_tables.len());
        for t in &p.page_tables {
            let mut table = Vec::new();
            for i in l.rows(q) {
                i.map_or(pad, |i| cells[i].row.own()).extend_row(&t.name, &mut table)?;
            }
            tables.push((&t.name, table));
        }
        // Line tables are written whole: cell `[r, c]` carries the line of
        // the row in column `c` in entry 0 (a program that moves along a
        // wide cell does so on the device), the null line 0 in the rest,
        // and the pad's past the batch.
        let mut lines: Vec<(&str, Vec<i32>)> = Vec::with_capacity(p.line_tables.len());
        for t in &p.line_tables {
            let cols_max = match t.axis {
                Axis::Tray => p.tray.as_ref().map_or(0, |b| b.max) as usize,
                _ => p.groups.max as usize,
            };
            let blocks = blocks(t.axis);
            let (name, w) = (t.name.as_str(), t.width);
            let mut table = vec![0i32; t.lines * cols_max * w];
            for r in 0..t.lines {
                let fill = pad.seq_line(name, r)?;
                for c in 0..cols_max {
                    let (d, j) = (c / b, c % b);
                    let cell = blocks.get(d).and_then(|&rank| l.rows(rank).nth(j).flatten());
                    table[(r * cols_max + c) * w] = match cell {
                        Some(i) => lease_on(i).seq_line(name, r)?,
                        None => fill,
                    };
                }
            }
            lines.push((name, table));
        }
        let rt = &mut self.ranks[q];
        for (f, v) in &writes {
            rt.write_input_at(&f.name, &f.encode(v), env)?;
        }
        for (name, table) in &tables {
            rt.write_input_at(name, &le_bytes_i32(table), env)?;
        }
        for (name, table) in &lines {
            rt.write_input(name, &le_bytes_i32(table))?;
        }
        Ok(())
    }
}

/// Sequences per rank: the sequences bound, and the tray bound's share
/// when the batch spans `t` ranks.
fn seqs_max(p: &Protocol, t: usize) -> usize {
    let g = p.groups.max as usize;
    p.tray.as_ref().map_or(g, |r| g.min(r.max as usize / t)).max(1)
}

/// A step staged on every rank: run a forward, read what it handed back.
/// Holds the tray until it drops.
pub struct Staged<'t> {
    tray: &'t mut Tray,
    layout: Layout,
    env: BTreeMap<String, u64>,
}

impl Staged<'_> {
    /// The forward the protocol picks for this step's shape: `b` sequences
    /// of `rows` rows on every rank.
    pub fn forward(&self, rows: u64) -> Option<Forward> {
        self.tray.protocol.forward(self.layout.b as u64, kern_manifest::protocol::Rows::Const(rows)).cloned()
    }

    /// Run `f` on every rank, eagerly or through its graph (captured on
    /// first use), then read the error word when the manifest has one.
    pub fn run(&mut self, f: &Forward, eager: bool) -> Result<()> {
        for (q, rt) in self.tray.ranks.iter_mut().enumerate() {
            run_program(rt, &f.name, &self.env, eager).with_context(|| format!("rank {q}"))?;
        }
        if let Some(e) = self.tray.protocol.any(Fill::Error) {
            for (q, rt) in self.tray.ranks.iter().enumerate() {
                let err = e.decode(&rt.read_output(&e.name)?)[0];
                if err != 0 {
                    bail!("rank {q}: `{}` reports collective error {err}", f.name);
                }
            }
        }
        Ok(())
    }

    /// What `f` handed each cell, in cell order, read from its owner: its
    /// `tokens` output's cell, cut to its `count` (one without a count);
    /// empty for a forward that only advances state.
    pub fn emitted(&self, f: &Forward) -> Result<Vec<Vec<i64>>> {
        let mut out: Vec<Vec<i64>> = vec![Vec::new(); self.layout.cells];
        let Some(i) = f.emits else { return Ok(out) };
        let p = &self.tray.protocol;
        let (t, c) = (&p.fills[i], f.count.map(|c| &p.fills[c]));
        let k = t.width as usize;
        for (q, rt) in self.tray.ranks.iter().enumerate() {
            if self.layout.own[q].is_empty() {
                continue;
            }
            let all = t.decode(&rt.read_output(&t.name)?);
            let counts = match c {
                Some(c) => Some(c.decode(&rt.read_output(&c.name)?)),
                None => None,
            };
            for (j, &i) in self.layout.own[q].iter().enumerate() {
                let n = counts.as_ref().map_or(1, |v| v[j]);
                if n < 1 || n > k as i64 {
                    bail!("rank {q}: `{}` says {n} of the {k} rows are taken", c.expect("counted").name);
                }
                out[i] = all[j * k..j * k + n as usize].to_vec();
            }
        }
        Ok(out)
    }
}

/// Run `program` at `env`: eagerly, or through its CUDA graph, captured
/// on first use.
fn run_program(rt: &mut Runtime, program: &str, env: &BTreeMap<String, u64>, eager: bool) -> Result<()> {
    if eager {
        return Ok(rt.run(program, env)?);
    }
    if !rt.is_captured(program, env) {
        let t = std::time::Instant::now();
        rt.capture(program, env)?;
        info!(program, env = ?env, capture_ms = logline::ms(t.elapsed()), "captured");
    }
    Ok(rt.run_captured(program, env)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tray manifest: 4 ranks' rows in `rows`, the tokens and the line
    /// table over it, the page table and the slots a rank's own.
    fn tray() -> Manifest {
        Manifest::from_json(
            r#"{
            "schema_version": 4, "model": "t", "vars": {"tokens": {"max": 8}, "seqs": {"max": 8}, "rows": {"max": 32}},
            "topology": {"groups": {"ep": 4, "tp": 4}},
            "states": {"kv": {"bytes_per_token": 1}, "kda": {"bytes_per_seq": 24}},
            "buffers": {
                "token_ids": {"kind": "input", "dtype": "i64", "shape": ["rows"], "fill": "token"},
                "slot_mapping": {"kind": "input", "dtype": "i64", "shape": ["tokens"], "fill": "slot", "domain": {"index_into": "kv"}},
                "seq_lens": {"kind": "input", "dtype": "i32", "shape": ["seqs"], "fill": "seq_len"},
                "block_table": {"kind": "input", "dtype": "i32", "shape": ["seqs", 3], "domain": {"index_into": "kv", "stride": 16}},
                "kda.line_index": {"kind": "input", "dtype": "i32", "shape": [3, "rows"], "domain": {"index_into": "kda", "stride": 8}},
                "next_token": {"kind": "output", "dtype": "i64", "shape": ["rows"], "fill": "tokens"},
                "tp_err": {"kind": "output", "dtype": "i32", "shape": [1], "fill": "error"}
            },
            "modules": {}, "ops": {"step": {"params": ["in buffer<i64>", "out buffer<i64>"], "impl": {"launches": []}}},
            "programs": {"decode": {"batch": {"groups": 8, "rows": 1}, "calls": [{"op": "step", "args": [{"buf": "token_ids"}, {"buf": "next_token"}]}]}}
        }"#,
        )
        .unwrap()
    }

    #[test]
    fn a_tray_batch_shares_the_rows_bound() {
        let p = Protocol::check(&tray()).unwrap();
        // 32 rows over a group of 4 is 8 per rank, the sequences bound; over 8 it is 4.
        assert_eq!((seqs_max(&p, 4), seqs_max(&p, 8), seqs_max(&p, 1)), (8, 4, 8));
    }

    #[test]
    fn groups_lay_blocks_out_own_first() {
        let g = Groups { n: 8, t: 4 };
        assert_eq!((g.member(5), g.members(5)), (1, 4..8));
        assert_eq!(g.blocks(5).collect::<Vec<_>>(), [5, 6, 7, 4]);
        assert_eq!(g.blocks(0).collect::<Vec<_>>(), [0, 1, 2, 3]);
        let one = Groups { n: 4, t: 1 };
        assert_eq!((one.members(2), one.blocks(2).collect::<Vec<_>>()), (2..3, vec![2]));
    }

    #[test]
    fn layout_buckets_the_most_loaded_rank() {
        let bucket = |k: usize| [1usize, 2, 4, 8].into_iter().find(|&b| b >= k).unwrap_or(k);
        let l = Layout::new(&[0, 0, 1, 3, 0], 1, 4, bucket);
        assert_eq!((l.b, l.per, l.cells), (4, 1, 5));
        assert_eq!(l.own, vec![vec![0, 1, 4], vec![2], vec![], vec![3]]);
        assert_eq!(l.rows(0).collect::<Vec<_>>(), [Some(0), Some(1), Some(4), None]);
        assert_eq!(l.rows(2).collect::<Vec<_>>(), [None, None, None, None]);
        // No cells still stage one padding row per rank.
        assert_eq!(Layout::new(&[], 1, 2, bucket).b, 1);
    }
}
