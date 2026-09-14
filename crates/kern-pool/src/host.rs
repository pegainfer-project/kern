//! The host tier: checkpoints parked in pinned DRAM.
//!
//! A [`Parked`] checkpoint is the bytes a [`crate::Checkpoint`] holds — its
//! pages and, with a recurrent state, its slot — copied into one pinned
//! block the runtime reserves once, so the device pages and the slot go
//! back to the pool while the prefix stays findable; waking copies the
//! first `len` tokens' pages back into a fresh lease. Pages are a chain
//! here too: one host node per page, keyed by the device node it was
//! copied from, so a checkpoint whose pages were parked once already (an
//! earlier turn of the same session) copies only the pages past them, and
//! a node returns its bytes when its last parked holder drops.
//!
//! A page on the host is every paged state's page back to back, in arena
//! order; a slot every per-sequence state's slot likewise. [`Host`] is the
//! allocator, pure host code over byte offsets in `grain` units: pages
//! taken from the low end, slots from the high end, first fit, free runs
//! coalesced. The runtime owns the pinned block and runs the copies a plan
//! names ([`Host::park`] says what to copy out, [`Parked::pages`] what to
//! copy back in); [`runs`] folds consecutive pages into one copy each.

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};

use crate::pages::Denied;

/// A parked page: `range` bytes of the block holding the page of device
/// node `id`, and the chain before it.
struct Node {
    id: u64,
    range: Range<u64>,
    parent: Option<Arc<Node>>,
    host: Arc<Host>,
}

impl Drop for Node {
    fn drop(&mut self) {
        self.host.give(self.id, self.range.clone());
        let mut next = self.parent.take();
        while let Some(n) = next {
            match Arc::try_unwrap(n) {
                Ok(mut node) => next = node.parent.take(),
                Err(_) => break,
            }
        }
    }
}

struct Inner {
    /// Free runs: start → length, in grain units, never adjacent.
    free: BTreeMap<u64, u64>,
    /// The parked page of every device node still parked somewhere.
    nodes: BTreeMap<u64, Weak<Node>>,
}

/// The pinned block's accounting: `bytes` in all, handed out in `grain`
/// units.
pub struct Host {
    grain: u64,
    units: u64,
    inner: Mutex<Inner>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The copies that park a checkpoint: (device page, host offset) for
/// every page not on the host already, root first, and the slot's.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Park {
    pub pages: Vec<(i32, u64)>,
    pub slot: Option<(i32, u64)>,
}

impl Host {
    /// A block of `bytes`, handed out in `grain` units.
    pub fn new(bytes: u64, grain: u64) -> Host {
        assert!(grain >= 1);
        let units = bytes / grain;
        let free = if units > 0 { BTreeMap::from([(0, units)]) } else { BTreeMap::new() };
        Host { grain, units, inner: Mutex::new(Inner { free, nodes: BTreeMap::new() }) }
    }

    pub fn bytes(&self) -> u64 {
        self.units * self.grain
    }

    /// Bytes handed out.
    pub fn used(&self) -> u64 {
        let g = lock(&self.inner);
        (self.units - g.free.values().sum::<u64>()) * self.grain
    }

    /// Nodes parked.
    pub fn pages(&self) -> usize {
        lock(&self.inner).nodes.len()
    }

    /// A run of `bytes`, first fit from the low end (`high` false) or the
    /// high end.
    fn take(&self, g: &mut Inner, bytes: u64, high: bool) -> Option<Range<u64>> {
        let n = bytes.div_ceil(self.grain).max(1);
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
        Some(at * self.grain..(at + n) * self.grain)
    }

    /// Return `range`, merging with its neighbours; forget node `id`.
    fn give(&self, id: u64, range: Range<u64>) {
        let mut g = lock(&self.inner);
        g.nodes.remove(&id);
        let (mut start, mut len) = (range.start / self.grain, (range.end - range.start) / self.grain);
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

    /// Park the checkpoint whose pages are `nodes` (device node id and
    /// page, root first) and whose slot, if any, is `slot`: a host node per
    /// page not parked already, a run for the slot, and the copies that
    /// fill them. `HostFull` when the block cannot hold it; nothing is
    /// kept then.
    pub fn park(
        self: &Arc<Host>,
        nodes: &[(u64, i32)],
        page_bytes: u64,
        slot: Option<(i32, u64)>,
        len: usize,
    ) -> std::result::Result<(Parked, Park), Denied> {
        let mut plan = Park::default();
        let mut chain: Option<Arc<Node>> = None;
        for &(id, page) in nodes {
            let existing = lock(&self.inner).nodes.get(&id).and_then(Weak::upgrade);
            let node = match existing {
                Some(n) => n,
                None => {
                    let mut g = lock(&self.inner);
                    let Some(range) = self.take(&mut g, page_bytes, false) else { return Err(Denied::HostFull) };
                    plan.pages.push((page, range.start));
                    let n = Arc::new(Node { id, range, parent: chain.take(), host: Arc::clone(self) });
                    g.nodes.insert(id, Arc::downgrade(&n));
                    n
                }
            };
            chain = Some(node);
        }
        let slot = match slot {
            Some((s, bytes)) => {
                let mut g = lock(&self.inner);
                let Some(range) = self.take(&mut g, bytes, true) else { return Err(Denied::HostFull) };
                plan.slot = Some((s, range.start));
                Some(range)
            }
            None => None,
        };
        Ok((Parked { len, chain, slot, host: Arc::clone(self) }, plan))
    }
}

/// A checkpoint on the host: the first `len` tokens of a sequence, as
/// host pages and, when the manifest has per-sequence states, a slot.
/// Dropping it releases what it alone holds.
pub struct Parked {
    len: usize,
    chain: Option<Arc<Node>>,
    slot: Option<Range<u64>>,
    host: Arc<Host>,
}

impl Parked {
    /// Tokens held; never 0.
    pub fn tokens(&self) -> usize {
        self.len
    }

    /// Whether a slot's bytes are held.
    pub fn has_slot(&self) -> bool {
        self.slot.is_some()
    }

    /// Whether any page is held (a slot-only checkpoint parks its slot alone).
    pub fn paged(&self) -> bool {
        self.chain.is_some()
    }

    /// Host offsets of the first `n` pages, root first.
    pub fn pages(&self, n: usize) -> Vec<u64> {
        let mut out = Vec::new();
        let mut cur = self.chain.as_ref();
        while let Some(node) = cur {
            out.push(node.range.start);
            cur = node.parent.as_ref();
        }
        out.reverse();
        assert!(n <= out.len(), "{n} pages of a parked checkpoint of {}", out.len());
        out.truncate(n);
        out
    }

    /// Host offset of the slot's bytes.
    pub fn slot(&self) -> Option<u64> {
        self.slot.as_ref().map(|r| r.start)
    }
}

impl Drop for Parked {
    fn drop(&mut self) {
        if let Some(r) = self.slot.take() {
            self.host.give(u64::MAX, r);
        }
    }
}

impl std::fmt::Debug for Parked {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Parked({} tokens{})", self.len, if self.slot.is_some() { ", slot" } else { "" })
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
