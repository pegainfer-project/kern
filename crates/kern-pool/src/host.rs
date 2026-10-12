//! The host tier: checkpoints parked in pinned DRAM.
//!
//! A [`Parked`] checkpoint is a [`Checkpoint`]'s bytes — its pages and,
//! with a recurrent state, its slot — copied into one pinned block the
//! runtime reserves once, so the device pages and the slot go back to the
//! pool while the prefix stays findable. Pages are a chain here too, one
//! host node per device node, and a device node remembers its host twin:
//! a checkpoint whose pages were parked once already (an earlier turn of
//! the same session, or a wake) copies only the pages past them, and a
//! host page returns when its last parked holder drops. Waking is
//! [`Host::restore`]: the first `len` tokens' pages copied straight into
//! a [`Lease`] sized for the sequence continuing from them — one
//! allocation, the room decision made once — its whole pages sealed and
//! twinned with the host ones, so the sequence's next checkpoint parks
//! them for free. The parked checkpoint stays where it is.
//!
//! [`Host`] counts host pages and host slots, two fixed budgets of
//! numbered units, and knows no bytes. What a unit holds is the shell's
//! business (every paged state's page back to back, every per-sequence
//! state's slot likewise), so one accounting can stand for several
//! processes that each hold different layers of the model and multiply
//! the same unit numbers by their own sizes. Each budget hands out its
//! lowest free unit. The runtime owns the pinned block and runs the
//! copies a plan names ([`Host::park`] says what to copy out,
//! [`Host::restore`] what to copy back in); [`runs`] folds consecutive
//! pages into one copy each.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use crate::pages::{Checkpoint, Denied, Lease, Pool};
use crate::store::{chain_nodes, chain_pages, lock, sealed, Copies, Node, SlotOwn, Storage, Store};

/// One budget of `n` numbered units. The ones never handed out are
/// `next..n` and the ones handed back are `free`, all below `next`, so
/// the lowest free unit is `free`'s first or else `next`.
struct Units {
    n: u64,
    next: u64,
    free: BTreeSet<u64>,
}

impl Units {
    fn new(n: u64) -> Units {
        Units { n, next: 0, free: BTreeSet::new() }
    }

    fn used(&self) -> u64 {
        self.next - self.free.len() as u64
    }

    fn take(&mut self) -> Option<u64> {
        self.free.pop_first().or_else(|| {
            let u = self.next;
            (u < self.n).then(|| {
                self.next += 1;
                u
            })
        })
    }

    fn give(&mut self, u: u64) {
        assert!(u < self.next && self.free.insert(u), "host unit {u} handed back but not out");
    }
}

/// The host block's accounting: `pages` host pages and `slots` host
/// slots, each numbered from 0.
pub struct Host {
    pages: Mutex<Units>,
    slots: Mutex<Units>,
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
        lock(&self.pages).give(page);
    }

    fn give_slot(&self, slot: u64) {
        lock(&self.slots).give(slot);
    }
}

impl Host {
    /// A block of `pages` host pages and `slots` host slots.
    pub fn new(pages: u64, slots: u64) -> Host {
        Host { pages: Mutex::new(Units::new(pages)), slots: Mutex::new(Units::new(slots)) }
    }

    /// Host pages held by a parked checkpoint.
    pub fn used(&self) -> u64 {
        lock(&self.pages).used()
    }

    /// Host slots held by a parked checkpoint.
    pub fn slots_used(&self) -> u64 {
        lock(&self.slots).used()
    }

    /// Park `cp`: a host node per device node not parked already, a host
    /// slot for its state, and the copies (device page, host page) and
    /// (device slot, host slot) that fill them, root first. `HostFull` when the block cannot hold it; nothing
    /// is kept then.
    pub fn park(self: &Arc<Host>, cp: &Checkpoint) -> std::result::Result<(Parked, Copies<i32, u64>), Denied> {
        let mut plan = Copies::default();
        let mut chain: Option<Arc<Node<Host>>> = None;
        for dev in chain_nodes(&cp.chain) {
            let twin = lock(&dev.twin).upgrade();
            let node = match twin {
                Some(n) => n,
                None => {
                    let Some(at) = lock(&self.pages).take() else { return Err(Denied::HostFull) };
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
                let Some(at) = lock(&self.slots).take() else { return Err(Denied::HostFull) };
                plan.slot = Some((s, at));
                Some(SlotOwn::new(at, self))
            }
            None => None,
        };
        Ok((Store { len: cp.len, pages: cp.pages, chain, slot }, plan))
    }

    /// A sequence continuing from the first `len` tokens of `p` with room
    /// for `tokens` (more than `len`), those tokens' bytes on the way in:
    /// fresh pages, the ones `len` fills copied from the host — the whole
    /// ones sealed and twinned with their host pages — and a fresh slot
    /// with the state when `len` is the whole checkpoint (a parked state
    /// is usable at its length only; a shorter `len` is a whole number of
    /// pages of a stateless one). The copies are (host page, device
    /// page) and (host slot, device slot). The lease names positions from `len` on. A slot-only
    /// checkpoint restores to a slot-only lease at its own length;
    /// `tokens` is not its business.
    pub fn restore(
        self: &Arc<Host>,
        p: &Parked,
        pool: &Arc<Pool>,
        len: usize,
        tokens: usize,
    ) -> std::result::Result<(Lease, Copies<u64, i32>), Denied> {
        let unit = pool.unit() as usize;
        assert!(
            len >= 1 && (len == p.len || (p.paged() && !p.has_slot() && len < p.len && len.is_multiple_of(unit))),
            "restoring {len} tokens of a parked checkpoint of {} ({} slot)",
            p.len,
            if p.has_slot() { "with a" } else { "no" }
        );
        let need = if p.paged() {
            assert!(tokens > len, "restoring {len} tokens into room for {tokens}");
            pool.pages_for(tokens)?
        } else {
            0
        };
        let (fresh, slot) = pool.take(need)?;
        let hosts = chain_nodes(&p.chain);
        let full = if p.paged() { len / unit } else { 0 };
        let chain = hosts[..full]
            .iter()
            .zip(&fresh)
            .fold(None, |chain, (h, &page)| Some(Node::new(page, chain, pool, Mutex::new(Arc::downgrade(h)))));
        let plan = Copies {
            pages: hosts.iter().take(len.div_ceil(unit)).zip(&fresh).map(|(h, &page)| (h.page, page)).collect(),
            slot: p.slot_id().zip(slot.as_ref().map(|s| s.id)),
        };
        Ok((Lease::new(chain, full, fresh, slot, len, pool), plan))
    }
}

impl Parked {
    /// The host page of each page, root first (a test reading the block
    /// back; the runtime moves bytes by the plans).
    pub fn page_ids(&self) -> Vec<u64> {
        chain_pages(&self.chain)
    }
}

/// Consecutive (device page, host page) pairs — page `p + 1` at host page
/// `o + 1` — folded into (first page, first host page, count) runs, in
/// order.
pub fn runs(pairs: &[(i32, u64)]) -> Vec<(i32, u64, usize)> {
    let mut out: Vec<(i32, u64, usize)> = Vec::new();
    for &(p, o) in pairs {
        match out.last_mut() {
            Some((p0, o0, n)) if *p0 as i64 + *n as i64 == p as i64 && *o0 + *n as u64 == o => *n += 1,
            _ => out.push((p, o, 1)),
        }
    }
    out
}
