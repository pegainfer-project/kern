//! Positions dealt over a group, and what a push does with them.
//!
//! A group that deals its paged states by position (a manifest's
//! `by: position` shard) keeps a sequence's position `p` on member
//! `p mod w`, at that member's local position `p / w` ([`Interleave`]).
//! Each member has a pool of its own and accounts in local positions: its
//! lease holds `len(n, m)` of a sequence's `n` positions in local pages.
//!
//! Handing a prompt from a whole-sequence pool (a prefill stage) to such a
//! group moves every row of the source's pages to the member that owns
//! it. [`deal`] folds that move, for one member, into runs over token rows
//! (a row is one token slot of a paged state, `bytes_per_token` bytes of
//! every paged state alike): consecutive local rows from source rows `w`
//! apart. A run never crosses a page boundary on either side unless the
//! next page is physically adjacent, so a prefill of 64-row pages over
//! eight members moves eight rows per source page into one eighth of a
//! local page, and whole runs of adjacent pages fold into one. The plan is
//! the same for every paged state, which share page ids; the runtime
//! scales it by each state's row bytes.

use crate::error::{bail, Result};

/// Positions dealt round-robin over `w` members: position `p` on member
/// `p % w` at local position `p / w`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Interleave {
    w: u64,
}

impl Interleave {
    /// Dealt over `w` members, at least one.
    pub fn new(w: u64) -> Result<Interleave> {
        if w == 0 {
            bail!(Api, "positions dealt over 0 members; expected at least 1");
        }
        Ok(Interleave { w })
    }

    pub fn members(self) -> u64 {
        self.w
    }

    /// The member holding position `p`.
    pub fn owner(self, p: u64) -> u64 {
        p % self.w
    }

    /// Where position `p` sits on its owner.
    pub fn local(self, p: u64) -> u64 {
        p / self.w
    }

    /// How many of a sequence's first `n` positions member `m` holds.
    pub fn len(self, n: u64, m: u64) -> u64 {
        n / self.w + u64::from(n % self.w > m)
    }
}

/// One side's pages: page ids in sequence order, `unit` rows each.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pages<'a> {
    pub ids: &'a [i32],
    pub unit: u64,
}

impl Pages<'_> {
    /// The row position `p` of the sequence is in.
    fn row(&self, p: u64) -> u64 {
        self.ids[(p / self.unit) as usize] as u64 * self.unit + p % self.unit
    }
}

/// `rows` rows into consecutive destination rows from `dst_row`, from
/// source rows `src_stride` apart from `src_row`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowRun {
    pub src_row: u64,
    pub src_stride: u64,
    pub dst_row: u64,
    pub rows: u64,
}

/// One member's share of a sequence, as maximal runs in local position
/// order: from [`deal`] alone, so every row it names is within the pages
/// it was dealt from and to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Deal {
    runs: Vec<RowRun>,
}

impl Deal {
    pub fn runs(&self) -> &[RowRun] {
        &self.runs
    }

    /// Rows the deal moves, member `m`'s positions of the sequence.
    pub fn rows(&self) -> u64 {
        self.runs.iter().map(|r| r.rows).sum()
    }
}

/// Member `m`'s rows of the first `len` positions of a sequence held
/// whole in `src`, into its local positions in `dst`. Refused when `m` is
/// not a member, a unit is 0, either side's pages hold fewer positions
/// than the deal names, or a page id it reads is negative.
pub fn deal(src: Pages, dst: Pages, il: Interleave, m: u64, len: u64) -> Result<Deal> {
    let w = il.members();
    if m >= w {
        bail!(Api, "member {m} of a group of {w}");
    }
    if src.unit == 0 || dst.unit == 0 {
        bail!(Api, "pages of {} source and {} destination rows; expected neither 0", src.unit, dst.unit);
    }
    let covers = |p: Pages, rows: u64, side: &str| -> Result<()> {
        let pages = rows.div_ceil(p.unit) as usize;
        if p.ids.len() < pages {
            bail!(Api, "{side} holds {} pages of {}, expected {pages} for {rows} rows", p.ids.len(), p.unit);
        }
        if let Some(id) = p.ids[..pages].iter().find(|&&id| id < 0) {
            bail!(Api, "{side} page id {id} among the {pages} pages dealt");
        }
        Ok(())
    };
    let n = il.len(len, m);
    covers(src, len, "source")?;
    covers(dst, n, "destination")?;
    let mut runs: Vec<RowRun> = Vec::new();
    let mut q = 0;
    while q < n {
        let p = q * w + m;
        let src_end = ((p / src.unit + 1) * src.unit - m).div_ceil(w);
        let dst_end = (q / dst.unit + 1) * dst.unit;
        let rows = src_end.min(dst_end).min(n) - q;
        let (s, d) = (src.row(p), dst.row(q));
        match runs.last_mut() {
            Some(r) if r.src_row + r.rows * w == s && r.dst_row + r.rows == d => r.rows += rows,
            _ => runs.push(RowRun { src_row: s, src_stride: w, dst_row: d, rows }),
        }
        q += rows;
    }
    Ok(Deal { runs })
}
