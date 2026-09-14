//! The host tier: checkpoints parked in pinned DRAM.
//!
//! A [`Parked`] checkpoint is a [`Checkpoint`]'s bytes — its pages and,
//! with a recurrent state, its slot — copied into one pinned block the
//! runtime reserves once, so the device pages and the slot go back to the
//! pool while the prefix stays findable. Pages are a chain here too, one
//! host node per device node, and a device node remembers its host twin:
//! a checkpoint whose pages were parked once already (an earlier turn of
//! the same session, or a wake) copies only the pages past them, and a
//! host page returns when its last parked holder drops. Waking copies the
//! first `len` tokens' pages back into fresh device pages — a resident
//! [`Checkpoint`] again, its nodes twinned with the host ones, so the
//! prefix is shared by every sequence that continues from it and parks
//! for free next time.
//!
//! A page on the host is every paged state's page back to back, in arena
//! order; a slot every per-sequence state's slot likewise. [`Host`] is the
//! allocator, pure host code over byte offsets in `grain` units: pages
//! taken from the low end, slots from the high end, first fit, free runs
//! coalesced. The runtime owns the pinned block and runs the copies a plan
//! names ([`Host::park`] says what to copy out, [`Host::wake`] what to
//! copy back in); [`runs`] folds consecutive pages into one copy each.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use crate::pages::{Checkpoint, Denied, Pool};
use crate::store::{chain_nodes, chain_pages, lock, sealed, Copies, Node, SlotOwn, Storage, Store};

struct Inner {
    /// Free runs: start → length, in grain units, never adjacent.
    free: BTreeMap<u64, u64>,
    /// Host pages alive.
    pages: usize,
}

/// The pinned block's accounting: `bytes` in all, handed out in `grain`
/// units, a page `page_bytes` and a slot `slot_bytes` of them.
pub struct Host {
    grain: u64,
    units: u64,
    page_bytes: u64,
    slot_bytes: u64,
    inner: Mutex<Inner>,
}

/// A checkpoint on the host: the first `len` tokens of a sequence, as
/// host pages and, when the manifest has per-sequence states, a slot.
pub type Parked = Store<Host>;

impl sealed::Sealed for Host {}

impl Storage for Host {
    type Page = u64;
    type Slot = u64;
    type Twin = ();

    fn give_page(&self, page: u64) {
        let mut g = lock(&self.inner);
        g.pages -= 1;
        self.give(&mut g, page, self.page_bytes);
    }

    fn give_slot(&self, slot: u64) {
        let mut g = lock(&self.inner);
        self.give(&mut g, slot, self.slot_bytes);
    }
}

impl Host {
    /// A block of `bytes`, handed out in `grain` units, for pages of
    /// `page_bytes` and slots of `slot_bytes`.
    pub fn new(bytes: u64, grain: u64, page_bytes: u64, slot_bytes: u64) -> Host {
        assert!(grain >= 1);
        let units = bytes / grain;
        let free = if units > 0 { BTreeMap::from([(0, units)]) } else { BTreeMap::new() };
        Host { grain, units, page_bytes, slot_bytes, inner: Mutex::new(Inner { free, pages: 0 }) }
    }

    pub fn bytes(&self) -> u64 {
        self.units * self.grain
    }

    /// Bytes handed out.
    pub fn used(&self) -> u64 {
        let g = lock(&self.inner);
        (self.units - g.free.values().sum::<u64>()) * self.grain
    }

    /// Host pages alive.
    pub fn pages(&self) -> usize {
        lock(&self.inner).pages
    }

    pub fn page_bytes(&self) -> u64 {
        self.page_bytes
    }

    pub fn slot_bytes(&self) -> u64 {
        self.slot_bytes
    }

    fn grains(&self, bytes: u64) -> u64 {
        bytes.div_ceil(self.grain).max(1)
    }

    /// A run of `bytes`, first fit from the low end (`high` false) or the
    /// high end.
    fn take(&self, g: &mut Inner, bytes: u64, high: bool) -> Option<u64> {
        let n = self.grains(bytes);
        let (start, len) = if high {
            g.free.iter().rev().find(|(_, &l)| l >= n).map(|(&s, &l)| (s, l))?
        } else {
            g.free.iter().find(|(_, &l)| l >= n).map(|(&s, &l)| (s, l))?
        };
        g.free.remove(&start);
        let at = if high { start + len - n } else { start };
        if high && len > n {
            g.free.insert(start, len - n);
        } else if len > n {
            g.free.insert(start + n, len - n);
        }
        Some(at * self.grain)
    }

    /// Return the run of `bytes` at `offset`, merging with its neighbours.
    fn give(&self, g: &mut Inner, offset: u64, bytes: u64) {
        let (mut start, mut len) = (offset / self.grain, self.grains(bytes));
        if let Some((&s, &l)) = g.free.range(..start).next_back() {
            if s + l == start {
                g.free.remove(&s);
                start = s;
                len += l;
            }
        }
        if let Some(&l) = g.free.get(&(start + len)) {
            g.free.remove(&(start + len));
            len += l;
        }
        g.free.insert(start, len);
    }

    fn take_page(self: &Arc<Host>, g: &mut Inner) -> Option<u64> {
        let at = self.take(g, self.page_bytes, false)?;
        g.pages += 1;
        Some(at)
    }

    /// Park `cp`: a host node per device node not parked already, a run
    /// for the slot, and the copies (device page, host offset) that fill
    /// them, root first. `HostFull` when the block cannot hold it; nothing
    /// is kept then.
    pub fn park(self: &Arc<Host>, cp: &Checkpoint) -> std::result::Result<(Parked, Copies<i32, u64>), Denied> {
        let mut plan = Copies::default();
        let mut chain: Option<Arc<Node<Host>>> = None;
        for dev in chain_nodes(&cp.chain) {
            let twin = lock(&dev.twin).upgrade();
            let node = match twin {
                Some(n) => n,
                None => {
                    let mut g = lock(&self.inner);
                    let Some(at) = self.take_page(&mut g) else { return Err(Denied::HostFull) };
                    drop(g);
                    plan.pages.push((dev.page, at));
                    let n = Node::new(at, chain.take(), self, ());
                    *lock(&dev.twin) = Arc::downgrade(&n);
                    n
                }
            };
            chain = Some(node);
        }
        let slot = match cp.seq_slot() {
            Some(s) => {
                let mut g = lock(&self.inner);
                let Some(at) = self.take(&mut g, self.slot_bytes, true) else { return Err(Denied::HostFull) };
                plan.slot = Some((s, at));
                Some(SlotOwn::new(at, self))
            }
            None => None,
        };
        Ok((Store { len: cp.len, pages: cp.pages, chain, slot }, plan))
    }

    /// The first `len` tokens of `p` back on the device as a checkpoint:
    /// fresh pages with those tokens' pages copied in, each twinned with
    /// its host node, and a fresh slot with the state when `len` is the
    /// whole checkpoint (a parked state is usable at its length only;
    /// a shorter `len` is a whole number of pages of a stateless one).
    /// The copies are (host offset, device page). A slot-only checkpoint
    /// wakes to a slot-only one.
    pub fn wake(
        self: &Arc<Host>,
        p: &Parked,
        pool: &Arc<Pool>,
        len: usize,
    ) -> std::result::Result<(Checkpoint, Copies<u64, i32>), Denied> {
        let unit = pool.unit() as usize;
        assert!(
            len >= 1 && (len == p.len || (p.paged() && !p.has_slot() && len < p.len && len.is_multiple_of(unit))),
            "waking {len} tokens of a parked checkpoint of {} ({} slot)",
            p.len,
            if p.has_slot() { "with a" } else { "no" }
        );
        let n = if p.paged() { len.div_ceil(unit) } else { 0 };
        let (fresh, slot) = pool.take(n)?;
        let hosts = chain_nodes(&p.chain);
        let mut chain: Option<Arc<Node<Pool>>> = None;
        let mut plan = Copies::default();
        for (h, &page) in hosts.iter().zip(&fresh) {
            plan.pages.push((h.page, page));
            chain = Some(Node::new(page, chain.take(), pool, Mutex::new(Arc::downgrade(h))));
        }
        plan.slot = p.slot_id().zip(slot.as_ref().map(|s| s.id));
        Ok((Store { len, pages: n, chain, slot }, plan))
    }
}

impl Parked {
    /// The host offset of each page, root first.
    pub fn offsets(&self) -> Vec<u64> {
        chain_pages(&self.chain)
    }

    /// The host offset of the slot, when one is held.
    pub fn slot_offset(&self) -> Option<u64> {
        self.slot_id()
    }
}

/// Consecutive (device page, host offset) pairs — page `p + 1` at host
/// offset `o + page_bytes` — folded into (first page, host offset, count)
/// runs, in order.
pub fn runs(pairs: &[(i32, u64)], page_bytes: u64) -> Vec<(i32, u64, usize)> {
    let mut out: Vec<(i32, u64, usize)> = Vec::new();
    for &(p, o) in pairs {
        match out.last_mut() {
            Some((p0, o0, n)) if *p0 as i64 + *n as i64 == p as i64 && *o0 + *n as u64 * page_bytes == o => *n += 1,
            _ => out.push((p, o, 1)),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    //! `runs` is used by the transfer stream, not by callers: the copies
    //! it folds are observable only as bytes on the host.

    use super::*;

    #[test]
    fn consecutive_pages_fold_into_one_run() {
        assert_eq!(runs(&[(10, 0), (11, 8), (12, 16)], 8), [(10, 0, 3)]);
        assert_eq!(runs(&[(10, 0), (11, 8), (13, 16), (14, 32)], 8), [(10, 0, 2), (13, 16, 1), (14, 32, 1)]);
    }
}
