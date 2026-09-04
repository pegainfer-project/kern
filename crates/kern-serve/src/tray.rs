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
//! nobody reads. Which inputs span the group and which are a rank's own
//! is read off the manifest: a buffer over the `rows` var carries the
//! tray batch — this rank's rows first, then the other members' in group
//! order from it, the layout the collectives assume — and one over
//! `tokens` or `seqs` carries this rank's rows alone. `token_ids` and the
//! line tables of a sharded state are of the first kind in a tray
//! manifest; `slot_mapping`, `seq_lens` and the page tables are always of
//! the second. Outputs likewise: a `next_token` over `rows` holds this
//! rank's rows in its first block, so every cell is read from its owner.
//! A manifest with a `tp_err` output declares the collectives' error
//! word; it is read after every step and a nonzero value is a failed
//! step, never a silent one.
//!
//! [`Staged`] borrows the tray for as long as the step lives: nothing can
//! lease, fork or stage again until the outputs are read and it drops,
//! which is what keeps "stage, run, read" one indivisible motion.

use std::collections::BTreeMap;
use std::ops::Range;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use kern_manifest::types::{BufferKind, Dim, Manifest};
use kern_run::{i64_from_le, le_bytes_i32, le_bytes_i64};
use kern_runtime::{
    Capacity, Checkpoint, Denied, Error, GroupRank, Kept, Lease, Parked, PeerHandle, Room, Runtime, Topology, Waking,
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

/// A line table over a per-sequence state, `[lines, cols]` or
/// `[lines, cols, w]`: `rows` lines per sequence, `width` entries per
/// (line, sequence) cell, its columns the tray batch's rows or this
/// rank's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineTable {
    pub name: String,
    pub rows: usize,
    pub width: usize,
    pub tray: bool,
}

/// How this manifest's caller contract lays a step out: which of the
/// standard inputs it has and which axis each spans. Pure over the
/// manifest and the runtime's table names, so a synthetic manifest
/// exercises every rejection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    /// The `seqs` bound, less what the `rows` bound allows per rank.
    pub seqs_max: usize,
    pub tokens_max: usize,
    /// The manifest has a `rows` var: the tray batch's row axis.
    pub rows: bool,
    /// `token_ids` spans the tray batch rather than this rank.
    pub ids_tray: bool,
    pub positions: bool,
    pub cu_seqlens: bool,
    /// A `tp_err` output: the collectives' error word.
    pub err_word: bool,
    pub line_tables: Vec<LineTable>,
    pub page_tables: Vec<String>,
    /// The manifest has `decode_span`: one cell of a step may be a run of
    /// up to this many consecutive tokens of its sequence, each its own
    /// row, at the front of its owner's rows; `span_at` tells the kernels
    /// where that run sits in each rank's row order.
    pub span: Option<usize>,
}

impl Shape {
    /// `seq_tables` / `page_tables` are the runtime's, `t` the tray batch
    /// group's size.
    pub fn check(m: &Manifest, seq_tables: &[&str], page_tables: &[&str], t: usize) -> Result<Shape> {
        for name in ["token_ids", "slot_mapping", "seq_lens"] {
            if !m.buffers.contains_key(name) {
                bail!("manifest has no input buffer `{name}`");
            }
        }
        if !page_tables.contains(&"block_table") {
            bail!("`block_table` is not a page table (an input indexing a paged state)");
        }
        let seqs = var_max(m, "seqs")?;
        let tokens_max = var_max(m, "tokens")?;
        let rows = m.vars.get("rows").map(|v| v.max as usize);
        if t > 1 && rows.is_none() {
            bail!("a `tp` group of {t} needs a `rows` var: the tray batch's row axis");
        }
        let seqs_max = rows.map_or(seqs, |r| seqs.min(r / t)).max(1);
        let axis = |name: &str, kind: BufferKind, own: &str| -> Result<bool> {
            let b = m.buffers.get(name).with_context(|| format!("manifest has no buffer `{name}`"))?;
            if b.kind != kind {
                bail!("`{name}` is {}, expected {kind}", b.kind);
            }
            match b.shape.first() {
                Some(Dim::Var(v)) if v == own => Ok(false),
                Some(Dim::Var(v)) if v == "rows" && rows.is_some() => Ok(true),
                _ => bail!("`{name}` shaped {:?}, expected [{own}] or [rows]", b.shape),
            }
        };
        let ids_tray = axis("token_ids", BufferKind::Input, "tokens")?;
        // `next_token` over `rows` holds this rank's rows first, so a cell is
        // read from its owner the same way either way.
        axis("next_token", BufferKind::Output, "seqs")?;
        let line_tables = seq_tables
            .iter()
            .map(|name| {
                let (rows_, width, v) = match shape(m, name)? {
                    [Dim::Const(r), Dim::Var(v)] => (*r, 1, v),
                    [Dim::Const(r), Dim::Var(v), Dim::Const(w)] => (*r, *w, v),
                    s => {
                        bail!("line table `{name}` shaped {s:?}, expected [lines, seqs|rows] or [lines, seqs|rows, w]")
                    }
                };
                let tray = match v.as_str() {
                    "seqs" => false,
                    "rows" if rows.is_some() => true,
                    _ => bail!("line table `{name}` is over `{v}`, expected `seqs` or `rows`"),
                };
                Ok(LineTable { name: name.to_string(), rows: rows_ as usize, width: width as usize, tray })
            })
            .collect::<Result<Vec<_>>>()?;
        let page_tables = page_tables
            .iter()
            .map(|name| match shape(m, name)? {
                [Dim::Var(v), Dim::Const(_)] if v == "seqs" => Ok(name.to_string()),
                s => bail!("`{name}` shaped {s:?}, expected [seqs, n]"),
            })
            .collect::<Result<Vec<_>>>()?;
        let is_output = |name: &str| m.buffers.get(name).is_some_and(|b| b.kind == BufferKind::Output);
        let span = match m.programs.contains_key("decode_span") {
            false => None,
            true => {
                let at = m.buffers.get("span_at").context("`decode_span` without a `span_at` input")?;
                if at.kind != BufferKind::Input || !matches!(at.shape.as_slice(), [Dim::Const(1)]) {
                    bail!("`span_at` is {} shaped {:?}, expected an input [1]", at.kind, at.shape);
                }
                Some(var_max(m, "span")?)
            }
        };
        Ok(Shape {
            seqs_max,
            tokens_max,
            rows: rows.is_some(),
            ids_tray,
            positions: m.buffers.contains_key("positions"),
            cu_seqlens: m.buffers.contains_key("cu_seqlens_q"),
            err_word: is_output("tp_err"),
            line_tables,
            page_tables,
            span,
        })
    }
}

fn var_max(m: &Manifest, name: &str) -> Result<usize> {
    m.vars.get(name).map(|v| v.max as usize).with_context(|| format!("manifest has no var `{name}`"))
}

fn shape<'m>(m: &'m Manifest, name: &str) -> Result<&'m [Dim]> {
    m.buffers.get(name).map(|b| b.shape.as_slice()).with_context(|| format!("manifest has no buffer `{name}`"))
}

/// One row's part of a step: the tokens it feeds at `pos..`, and the
/// entry of a wide line table's cell that carries its line.
pub struct Cell<'a> {
    pub row: &'a Row,
    pub ids: Vec<i64>,
    pub pos: usize,
    pub col: usize,
}

/// Which cells each rank owns, widest first then in cell order, and the
/// bucket their rows were padded to: the pure half of [`Tray::stage`].
///
/// Every rank runs the step's program with the step's var values, so a
/// run of `span` rows exists on every rank: the owner's cell rows in its
/// group, and on a group that has no cell of the run, `span` padding rows
/// leading the block, which the span kernels chew on harmlessly and the
/// row kernels skip.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Layout {
    /// `own[q]` = indices of the cells rank `q` owns.
    own: Vec<Vec<usize>>,
    /// Rows per rank after padding.
    b: usize,
    /// Tokens per row.
    per: usize,
    /// Rows each cell fills: its tokens under the span contract, else 1.
    len: Vec<usize>,
    /// Padding rows leading rank `q`'s block in place of the run.
    lead: Vec<usize>,
}

impl Layout {
    /// `cells[i]` = (owner, rows) of cell `i`; at most one cell has more
    /// than one row.
    fn new(cells: &[(usize, usize)], per: usize, groups: &Groups, bucket: impl Fn(usize) -> usize) -> Layout {
        let mut own = vec![Vec::new(); groups.n];
        for (i, &(q, _)) in cells.iter().enumerate() {
            own[q].push(i);
        }
        let len: Vec<usize> = cells.iter().map(|&(_, l)| l).collect();
        // A run of rows sits at the front of its owner's block, where
        // `span_at` is a whole number of blocks.
        for o in &mut own {
            o.sort_by_key(|&i| std::cmp::Reverse(len[i]));
        }
        let run = cells.iter().find(|&&(_, l)| l > 1);
        let lead: Vec<usize> = (0..groups.n)
            .map(|q| match run {
                Some(&(owner, l)) if !groups.blocks(q).any(|r| r == owner) => l,
                _ => 0,
            })
            .collect();
        let most = (0..groups.n).map(|q| lead[q] + own[q].iter().map(|&i| len[i]).sum::<usize>()).max().unwrap_or(0);
        Layout { own, b: bucket(most).max(1), per, len, lead }
    }

    /// Rank `q`'s rows: the run's stand-in padding when its group has no
    /// cell of the run, `(cell, row within the cell)` for its own cells'
    /// rows, then `None` for each padding row.
    fn rows(&self, q: usize) -> impl Iterator<Item = Option<(usize, usize)>> + '_ {
        let mine = self.own[q].iter().flat_map(|&i| (0..self.len[i]).map(move |j| (i, j)));
        std::iter::repeat_n(None, self.lead[q]).chain(mine.map(Some)).chain(std::iter::repeat(None)).take(self.b)
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
    shape: Shape,
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
    /// reserve `host_bytes` of pinned memory per rank and lease the pad.
    pub fn load(
        manifest_json: &str,
        kernels: &Path,
        gpus: &[usize],
        capacity: Option<Capacity>,
        weights_of: &(dyn Fn(&Topology) -> Result<Vec<PathBuf>> + Sync),
        host_bytes: u64,
    ) -> Result<Tray> {
        let m = Manifest::from_json(manifest_json)?;
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
                        let mut rt =
                            Runtime::load(manifest_json, kernels, gpu, capacity, has_topology.then_some(&topo))
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
        let seq_tables: Vec<&str> = ranks[0].seq_tables().collect();
        let page_tables: Vec<&str> = ranks[0].page_tables().collect();
        let shape = Shape::check(&m, &seq_tables, &page_tables, t)?;
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
        Ok(Tray { ranks, groups, shape, pad })
    }

    pub fn manifest(&self) -> &Manifest {
        &self.ranks[0].manifest
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
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
    /// number of tokens, except that under the span contract one cell may
    /// feed a run of them, a row each (the `span` var), the others one.
    pub fn stage(&mut self, cells: &[Cell<'_>], bucket: impl Fn(usize) -> usize) -> Result<Staged<'_>> {
        if cells.iter().any(|c| c.ids.is_empty()) {
            bail!("a cell feeds at least one token");
        }
        let per = match self.shape.span {
            Some(_) => 1,
            None => cells.first().map_or(1, |c| c.ids.len()),
        };
        let runs: Vec<usize> = cells.iter().map(|c| c.ids.len()).filter(|&l| l != per).collect();
        let span = match (self.shape.span, runs.as_slice()) {
            (_, []) => None,
            (Some(mx), &[c]) if c <= mx => Some(c),
            (Some(mx), &[c]) => bail!("a run of {c} rows, the manifest's `span` allows {mx}"),
            (Some(_), _) => bail!("one cell per step may feed a run of tokens, {} do", runs.len()),
            (None, _) => bail!("every cell of a step feeds the same number of tokens"),
        };
        let placed: Vec<(usize, usize)> =
            cells.iter().map(|c| (c.row.owner.0, if self.shape.span.is_some() { c.ids.len() } else { 1 })).collect();
        let layout = Layout::new(&placed, per, &self.groups, bucket);
        let (b, t) = (layout.b, self.groups.t);
        if b > self.shape.seqs_max {
            bail!("{b} rows per rank, the manifest allows {}", self.shape.seqs_max);
        }
        let mut env = BTreeMap::from([("tokens".to_string(), (per * b) as u64), ("seqs".to_string(), b as u64)]);
        if self.shape.rows {
            env.insert("rows".to_string(), (t * b) as u64);
        }
        if let Some(c) = span {
            env.insert("span".to_string(), c as u64);
        }
        for q in 0..self.groups.n {
            self.stage_rank(q, cells, &layout, &env)?;
        }
        Ok(Staged { tray: self, layout, env })
    }

    /// Rank `q`'s inputs for a step (see the module doc for which span
    /// the group).
    fn stage_rank(&mut self, q: usize, cells: &[Cell<'_>], l: &Layout, env: &BTreeMap<String, u64>) -> Result<()> {
        let (per, b, me) = (l.per, l.b, self.groups.member(q));
        let shape = &self.shape;
        let pad = &self.pad[q];
        // Every rank's slot for a row is the lease at its own member index.
        let lease_on = |i: usize| -> &Lease { &cells[i].row.parts[me] };
        let blocks: Vec<usize> = if shape.ids_tray { self.groups.blocks(q).collect() } else { vec![q] };
        // A row is a cell's tokens (one under the span contract, the
        // `j`-th of its run) at the cell's position plus `j`.
        let mut ids = Vec::with_capacity(per * b * blocks.len());
        for &r in &blocks {
            for i in l.rows(r) {
                match i {
                    Some((i, j)) => ids.extend_from_slice(&cells[i].ids[j * per..(j + 1) * per]),
                    None => ids.extend(std::iter::repeat_n(0, per)),
                }
            }
        }
        let mut positions = Vec::with_capacity(per * b);
        let mut slots = Vec::with_capacity(per * b);
        let mut seq_lens = Vec::with_capacity(b);
        for i in l.rows(q) {
            let (pos, lease) = match i {
                Some((i, j)) => (cells[i].pos + j, cells[i].row.own()),
                None => (0, pad),
            };
            positions.extend((pos..pos + per).map(|p| p as i64));
            slots.extend(lease.slots(pos..pos + per));
            seq_lens.push((pos + per) as i32);
        }
        let cu: Vec<i32> = (0..=b as i32).map(|i| i * per as i32).collect();
        // The run's first row on this rank: block d of its owner, whose
        // cells it heads; the leading padding of the own block when no
        // block here has it.
        let span_at: i32 = (0..cells.len())
            .find(|&i| l.len[i] > 1)
            .and_then(|i| blocks.iter().position(|&r| r == cells[i].row.owner.0))
            .map_or(0, |d| (d * b) as i32);
        let mut tables: Vec<(String, Vec<i32>)> = Vec::with_capacity(shape.page_tables.len());
        for name in &shape.page_tables {
            let mut table = Vec::new();
            for i in l.rows(q) {
                i.map_or(pad, |(i, _)| cells[i].row.own()).extend_row(name, &mut table)?;
            }
            tables.push((name.clone(), table));
        }
        // Line tables are written whole: cell `[r, c]` carries the line of
        // the row in column `c` — entry `col` of a wide table's cell, the
        // null line 0 in the rest — and the pad's past the batch.
        let mut lines: Vec<(String, Vec<i32>)> = Vec::with_capacity(shape.line_tables.len());
        for LineTable { name, rows, width: w, tray } in &shape.line_tables {
            let (cols_max, blocks): (usize, Vec<usize>) = if *tray {
                (var_max(&self.ranks[q].manifest, "rows")?, self.groups.blocks(q).collect())
            } else {
                (var_max(&self.ranks[q].manifest, "seqs")?, vec![q])
            };
            let mut table = vec![0i32; rows * cols_max * w];
            for r in 0..*rows {
                let fill = pad.seq_line(name, r)?;
                for c in 0..cols_max {
                    let (d, j) = (c / b, c % b);
                    let cell = blocks.get(d).and_then(|&rank| l.rows(rank).nth(j).flatten());
                    let (line, col) = match cell {
                        Some((i, _)) => (lease_on(i).seq_line(name, r)?, cells[i].col),
                        None => (fill, 0),
                    };
                    if col >= *w {
                        bail!("line table `{name}`: entry {col} of a {w}-wide cell");
                    }
                    table[(r * cols_max + c) * w + col] = line;
                }
            }
            lines.push((name.clone(), table));
        }
        let rt = &mut self.ranks[q];
        rt.write_input_at("token_ids", &le_bytes_i64(&ids), env)?;
        if shape.positions {
            rt.write_input_at("positions", &le_bytes_i64(&positions), env)?;
        }
        rt.write_input_at("slot_mapping", &le_bytes_i64(&slots), env)?;
        rt.write_input_at("seq_lens", &le_bytes_i32(&seq_lens), env)?;
        if shape.cu_seqlens {
            rt.write_input_at("cu_seqlens_q", &le_bytes_i32(&cu), env)?;
        }
        if shape.span.is_some() {
            rt.write_input("span_at", &le_bytes_i32(&[span_at]))?;
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

/// A step staged on every rank: write what else it needs, run it, read
/// its outputs. Holds the tray until it drops.
pub struct Staged<'t> {
    tray: &'t mut Tray,
    layout: Layout,
    env: BTreeMap<String, u64>,
}

impl Staged<'_> {
    /// Rows per rank the step was padded to.
    pub fn b(&self) -> usize {
        self.layout.b
    }

    /// A per-sequence input (shaped `[seqs]`, this rank's rows): `of(i)`
    /// for cell `i`, `fill` in the padding rows.
    pub fn write_seqs<T: Le>(&mut self, name: &str, of: impl Fn(usize) -> T, fill: T) -> Result<()> {
        for q in 0..self.tray.groups.n {
            let v: Vec<T> = self.layout.rows(q).map(|i| i.map_or(fill, |(i, _)| of(i))).collect();
            self.tray.ranks[q].write_input_at(name, &T::le_bytes(&v), &self.env)?;
        }
        Ok(())
    }

    /// Run `program` on every rank, eagerly or through its graph (captured
    /// on first use), then read the error word when the manifest has one.
    pub fn run(&mut self, program: &str, eager: bool) -> Result<()> {
        // Every rank is issued before any is waited for: a rank's kernels
        // wait on its peers' (EP dispatch, the tray collectives), so
        // waiting on rank 0 alone would spin until the kernel's timeout.
        for (q, rt) in self.tray.ranks.iter_mut().enumerate() {
            enqueue_program(rt, program, &self.env, eager).with_context(|| format!("rank {q}"))?;
        }
        for (q, rt) in self.tray.ranks.iter().enumerate() {
            rt.synchronize().with_context(|| format!("rank {q}"))?;
        }
        if self.tray.shape.err_word {
            for (q, rt) in self.tray.ranks.iter().enumerate() {
                let err = i32::from_le_bytes(rt.read_output("tp_err")?[..4].try_into().unwrap());
                if err != 0 {
                    bail!("rank {q}: `{program}` reports collective error {err}");
                }
            }
        }
        Ok(())
    }

    /// A per-sequence output (`[seqs]`, `[seqs, k]`, or over `rows` with
    /// this rank's rows first): each cell's `k` values, in cell order,
    /// read from its owner; a run of rows yields its last row's.
    pub fn read_i64(&self, name: &str) -> Result<Vec<Vec<i64>>> {
        let k = match shape(self.tray.manifest(), name)? {
            [Dim::Var(_)] => 1,
            [Dim::Var(_), Dim::Const(k)] => *k as usize,
            s => bail!("output `{name}` shaped {s:?}, expected [seqs], [seqs, k], [rows] or [rows, k]"),
        };
        let mut out: Vec<Vec<i64>> = vec![Vec::new(); self.layout.len.len()];
        for (q, rt) in self.tray.ranks.iter().enumerate() {
            if self.layout.own[q].is_empty() {
                continue;
            }
            let all = i64_from_le(&rt.read_output(name)?);
            for (r, cell) in self.layout.rows(q).enumerate() {
                if let Some((i, _)) = cell.filter(|&(i, j)| j + 1 == self.layout.len[i]) {
                    out[i] = all[r * k..(r + 1) * k].to_vec();
                }
            }
        }
        Ok(out)
    }
}

/// An input element type: how a list of it is written to a buffer.
pub trait Le: Copy {
    fn le_bytes(v: &[Self]) -> Vec<u8>;
}

impl Le for i64 {
    fn le_bytes(v: &[i64]) -> Vec<u8> {
        le_bytes_i64(v)
    }
}

impl Le for i32 {
    fn le_bytes(v: &[i32]) -> Vec<u8> {
        le_bytes_i32(v)
    }
}

/// Run `program` at `env`: eagerly, or through its CUDA graph, captured
/// on first use.
fn enqueue_program(rt: &mut Runtime, program: &str, env: &BTreeMap<String, u64>, eager: bool) -> Result<()> {
    if eager {
        return Ok(rt.enqueue(program, env)?);
    }
    if !rt.is_captured(program, env) {
        let t = std::time::Instant::now();
        rt.capture(program, env)?;
        info!(
            program,
            seqs = env.get("seqs"),
            tokens = env.get("tokens"),
            rows = env.get("rows"),
            capture_ms = logline::ms(t.elapsed()),
            "captured"
        );
    }
    Ok(rt.enqueue_captured(program, env)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tray manifest: 4 ranks' rows in `rows`, `token_ids` and the line
    /// table over it, the page table and `slot_mapping` a rank's own.
    const TRAY: &str = r#"{
            "schema_version": 3, "model": "t", "vars": {"tokens": {"max": 8}, "seqs": {"max": 8}, "rows": {"max": 32}},
            "topology": {"groups": {"ep": 4, "tp": 4}},
            "states": {"kv": {"bytes_per_token": 1}, "kda": {"bytes_per_seq": 24}},
            "buffers": {
                "token_ids": {"kind": "input", "dtype": "i64", "shape": ["rows"]},
                "slot_mapping": {"kind": "input", "dtype": "i64", "shape": ["tokens"], "domain": {"index_into": "kv"}},
                "seq_lens": {"kind": "input", "dtype": "i32", "shape": ["seqs"]},
                "block_table": {"kind": "input", "dtype": "i32", "shape": ["seqs", 3], "domain": {"index_into": "kv", "stride": 16}},
                "kda.line_index": {"kind": "input", "dtype": "i32", "shape": [3, "rows"], "domain": {"index_into": "kda", "stride": 8}},
                "next_token": {"kind": "output", "dtype": "i64", "shape": ["rows"]},
                "tp_err": {"kind": "output", "dtype": "i32", "shape": [1]}
            },
            "modules": {}, "ops": {}, "programs": {"decode": []}
        }"#;

    fn tray() -> Manifest {
        Manifest::from_json(TRAY).unwrap()
    }

    /// `decode_span` makes the span contract: its `span` var bound, and a
    /// `span_at` input to tell the kernels where the run sits.
    #[test]
    fn span_contract() {
        let span = |programs: &str, buffers: &str| {
            let json = TRAY
                .replace(r#""programs": {"decode": []}"#, programs)
                .replace(r#""tp_err": {"kind": "output", "dtype": "i32", "shape": [1]}"#, buffers)
                .replace(r#""rows": {"max": 32}"#, r#""rows": {"max": 32}, "span": {"max": 4}"#);
            Shape::check(&Manifest::from_json(&json).unwrap(), &["kda.line_index"], &["block_table"], 4).map(|s| s.span)
        };
        let err = r#""tp_err": {"kind": "output", "dtype": "i32", "shape": [1]}"#;
        let at = r#""span_at": {"kind": "input", "dtype": "i32", "shape": [1]}, "#;
        let both = r#""programs": {"decode": [], "decode_span": []}"#;
        assert_eq!(span(r#""programs": {"decode": []}"#, err).unwrap(), None);
        assert_eq!(span(both, &format!("{at}{err}")).unwrap(), Some(4));
        let e = span(both, err).unwrap_err().to_string();
        assert!(e.contains("`span_at`"), "{e}");
    }

    /// The single-rank contract: everything over `tokens` / `seqs`.
    fn plain() -> Manifest {
        Manifest::from_json(
            r#"{
            "schema_version": 3, "model": "t", "vars": {"tokens": {"max": 8}, "seqs": {"max": 4}},
            "states": {"kv": {"bytes_per_token": 1}, "gdn": {"bytes_per_seq": 24}},
            "buffers": {
                "token_ids": {"kind": "input", "dtype": "i64", "shape": ["tokens"]},
                "positions": {"kind": "input", "dtype": "i64", "shape": ["tokens"]},
                "slot_mapping": {"kind": "input", "dtype": "i64", "shape": ["tokens"], "domain": {"index_into": "kv"}},
                "seq_lens": {"kind": "input", "dtype": "i32", "shape": ["seqs"]},
                "cu_seqlens_q": {"kind": "input", "dtype": "i32", "shape": ["seqs"]},
                "block_table": {"kind": "input", "dtype": "i32", "shape": ["seqs", 3], "domain": {"index_into": "kv", "stride": 16}},
                "line_index": {"kind": "input", "dtype": "i32", "shape": [3, "seqs"], "domain": {"index_into": "gdn", "stride": 8}},
                "next_token": {"kind": "output", "dtype": "i64", "shape": ["seqs"]}
            },
            "modules": {}, "ops": {}, "programs": {"prefill": [], "decode": [], "decode_batch": []}
        }"#,
        )
        .unwrap()
    }

    fn rejects(m: &Manifest, seq: &[&str], t: usize, what: &str) {
        let Err(e) = Shape::check(m, seq, &["block_table"], t) else { panic!("accepted, expected `{what}`") };
        let e = format!("{e:#}");
        assert!(e.contains(what), "{e}");
    }

    #[test]
    fn tray_shape() {
        let s = Shape::check(&tray(), &["kda.line_index"], &["block_table"], 4).unwrap();
        assert_eq!((s.seqs_max, s.tokens_max, s.rows), (8, 8, true));
        assert_eq!((s.ids_tray, s.positions, s.cu_seqlens, s.err_word), (true, false, false, true));
        let lines: Vec<_> = s.line_tables.iter().map(|t| (t.name.as_str(), t.rows, t.width, t.tray)).collect();
        assert_eq!((lines, s.page_tables.clone()), (vec![("kda.line_index", 3, 1, true)], vec!["block_table".into()]));
        // The `rows` bound caps rows per rank: 32 rows over 8 ranks is 4.
        assert_eq!(Shape::check(&tray(), &["kda.line_index"], &["block_table"], 8).unwrap().seqs_max, 4);
    }

    #[test]
    fn plain_shape() {
        let s = Shape::check(&plain(), &["line_index"], &["block_table"], 1).unwrap();
        assert_eq!((s.seqs_max, s.rows, s.ids_tray), (4, false, false));
        assert_eq!((s.positions, s.cu_seqlens, s.err_word, s.line_tables[0].tray), (true, true, false, false));
    }

    #[test]
    fn shape_rejections() {
        rejects(&plain(), &["line_index"], 4, "needs a `rows` var");
        let mut m = plain();
        m.buffers.remove("seq_lens");
        rejects(&m, &["line_index"], 1, "no input buffer `seq_lens`");
        let mut m = plain();
        m.buffers.get_mut("token_ids").unwrap().shape = vec![Dim::Const(8)];
        rejects(&m, &["line_index"], 1, "expected [tokens] or [rows]");
        let mut m = plain();
        m.buffers.get_mut("line_index").unwrap().shape = vec![Dim::Const(3)];
        rejects(&m, &["line_index"], 1, "expected [lines, seqs|rows]");
        let mut m = plain();
        m.buffers.get_mut("block_table").unwrap().shape = vec![Dim::Var("seqs".into())];
        rejects(&m, &["line_index"], 1, "expected [seqs, n]");
        let Err(e) = Shape::check(&plain(), &[], &[], 1) else { panic!("no page table accepted") };
        assert!(e.to_string().contains("block_table"), "{e}");
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
        let one = |q: usize| (q, 1);
        let l = Layout::new(&[one(0), one(0), one(1), one(3), one(0)], 1, &Groups { n: 4, t: 1 }, bucket);
        assert_eq!((l.b, l.per, l.len.len()), (4, 1, 5));
        assert_eq!(l.own, vec![vec![0, 1, 4], vec![2], vec![], vec![3]]);
        assert_eq!(l.rows(0).collect::<Vec<_>>(), [Some((0, 0)), Some((1, 0)), Some((4, 0)), None]);
        assert_eq!(l.rows(2).collect::<Vec<_>>(), [None, None, None, None]);
        // A run of rows heads its owner's block and counts as that many
        // rows; a group without it leads with as many padding rows.
        let l = Layout::new(&[one(1), (1, 3), one(0)], 1, &Groups { n: 2, t: 1 }, bucket);
        assert_eq!((l.b, l.own.clone(), l.lead.clone()), (4, vec![vec![2], vec![1, 0]], vec![3, 0]));
        assert_eq!(l.rows(1).collect::<Vec<_>>(), [Some((1, 0)), Some((1, 1)), Some((1, 2)), Some((0, 0))]);
        assert_eq!(l.rows(0).collect::<Vec<_>>(), [None, None, None, Some((2, 0))]);
        // Peers of the owner's tray group carry the run in the owner's
        // block already: no lead there, but a foreign group leads.
        let l = Layout::new(&[(1, 3), one(2)], 1, &Groups { n: 4, t: 2 }, bucket);
        assert_eq!((l.b, l.lead.clone()), (4, vec![0, 0, 3, 3]));
        assert_eq!(l.rows(0).collect::<Vec<_>>(), [None, None, None, None]);
        assert_eq!(l.rows(2).collect::<Vec<_>>(), [None, None, None, Some((1, 0))]);
        // No cells still stage one padding row per rank.
        assert_eq!(Layout::new(&[], 1, &Groups { n: 2, t: 1 }, bucket).b, 1);
    }
}
