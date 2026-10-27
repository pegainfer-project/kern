//! The lease shell. The pool decides on the host which pages and slots
//! a sequence gets and which bytes move; the runtime runs those moves on
//! the compute stream.
//!
//! Every lease operation begins with `poll`: a checkpoint whose park
//! copies have landed goes back to the pool. A leased slot is zeroed on
//! the stream (a fresh sequence's recurrent state); a checkpoint, a
//! restore and a fork move whole pages and whole slots, as the pool's
//! [`Copies`] say.
//!
//! The pool may also live in another process: a serving head holds one
//! for every rank, built from the rank's [`Runtime::budget`], and the
//! rank's runtime only runs what it decided — [`Runtime::copy`] for the
//! byte moves, [`Runtime::zero_slot`] for a fresh lease's slot. Such a
//! runtime leases nothing of its own.

use cudarc::driver::sys;

use crate::device::landed;
use crate::error::bail;
use crate::{Result, Runtime};
use kern_pool::{Checkpoint, Copies, Lease};

impl Runtime {
    /// Lease the pages `tokens` slots need and, when the manifest has
    /// per-sequence states, a slot in each of them, all or nothing
    /// ([`Error::Denied`] says why not). The lease is the only source of
    /// `slot_mapping` values, page-table rows and line indices for the
    /// sequence; the slot is zeroed on the stream (a fresh sequence's
    /// recurrent state), and everything returns when the lease drops.
    pub fn lease(&mut self, tokens: usize) -> Result<Lease> {
        self.ctx.bind_to_thread()?;
        self.poll()?;
        let lease = self.pool.lease(tokens)?;
        for (name, st) in &self.manifest.states {
            let Some(range) = lease.seq_bytes(st.bytes_per_seq).filter(|_| st.is_per_seq()) else { continue };
            let s = self.states.get_mut(name).unwrap();
            let mut view = s.view(range)?;
            self.stream.memset_zeros(&mut view)?;
        }
        Ok(lease)
    }

    /// A slot in each per-sequence state and no pages, zeroed on the
    /// stream: this rank's share of a sequence whose positions live on
    /// another rank (a tensor-parallel peer holds a slice of every row's
    /// recurrent state). Empty when the manifest has no per-sequence state.
    pub fn lease_slot(&mut self) -> Result<Lease> {
        self.ctx.bind_to_thread()?;
        self.poll()?;
        let lease = self.pool.lease_slot()?;
        for (name, st) in &self.manifest.states {
            let Some(range) = lease.seq_bytes(st.bytes_per_seq).filter(|_| st.is_per_seq()) else { continue };
            let s = self.states.get_mut(name).unwrap();
            let mut view = s.view(range)?;
            self.stream.memset_zeros(&mut view)?;
        }
        Ok(lease)
    }

    /// Zero sequence slot `slot` of every per-sequence state on the
    /// stream, the way a lease zeroes its own: for a runtime whose slots
    /// another pool hands out (a pipeline stage holds the head's slot ids
    /// for its own layers' recurrent state; a decode rank, its own slots
    /// as the head's pool for it decides them). That pool knows which slots
    /// are free; this one only refuses a slot past its own.
    pub fn zero_slot(&mut self, slot: i32) -> Result<()> {
        self.ctx.bind_to_thread()?;
        let max = self.provision.seq_slots as usize;
        if slot < 1 || slot as usize >= max {
            bail!(Api, "slot {slot}: this runtime holds slots 1..{max}");
        }
        for (name, st) in &self.manifest.states {
            if !st.is_per_seq() {
                continue;
            }
            let b = st.bytes_per_seq as usize;
            let s = slot as usize;
            let mut view = self.states.get_mut(name).unwrap().view(s * b..(s + 1) * b)?;
            self.stream.memset_zeros(&mut view)?;
        }
        Ok(())
    }

    /// The first `len` tokens of `lease` as a [`Checkpoint`] its
    /// sequence keeps running past: pages shared, the per-sequence state
    /// copied into a fresh slot ([`Error::Denied`] when none is free).
    /// The copies it ran come back too, for a runtime holding the same
    /// ids to run after it ([`Runtime::copy`]).
    pub fn checkpoint(&mut self, lease: &mut Lease, len: usize) -> Result<(Checkpoint, Copies)> {
        self.ctx.bind_to_thread()?;
        self.poll()?;
        let (cp, copies) = self.pool.checkpoint(lease, len)?;
        self.run_copies(&copies.pages, copies.slot.as_slice())?;
        Ok((cp, copies))
    }

    /// The first `len` tokens of a finished sequence as a [`Checkpoint`]:
    /// its pages past `len` return, its state slot moves over as it is.
    pub fn retire(&mut self, lease: Lease, len: usize) -> Checkpoint {
        self.pool.retire(lease, len)
    }

    /// A sequence continuing from the first `len` tokens of `cp` with room
    /// for `tokens`: shares those pages (copying the one `len` ends
    /// inside), copies the state into a fresh slot, and hands out a lease
    /// that names positions from `len` on. `len` is the checkpoint's own
    /// length, or any whole number of its pages when it holds no state.
    /// [`Error::Denied`] as for [`Runtime::lease`]. The copies it ran come
    /// back too, as from [`Runtime::checkpoint`].
    pub fn lease_from(&mut self, cp: &Checkpoint, len: usize, tokens: usize) -> Result<(Lease, Copies)> {
        self.ctx.bind_to_thread()?;
        self.poll()?;
        let (lease, copies) = self.pool.restore(cp, len, tokens)?;
        self.run_copies(&copies.pages, copies.slot.as_slice())?;
        Ok((lease, copies))
    }

    /// A sequence branched off the first `len` tokens of `parent`, which
    /// keeps running: the whole pages shared, the page `len` ends inside
    /// copied, the parent's state copied into a fresh slot. With a
    /// recurrent state `len` is the parent's position — the state is the
    /// parent's as of now. The lease names positions from `len` on.
    pub fn fork(&mut self, parent: &mut Lease, len: usize, tokens: usize) -> Result<Lease> {
        self.ctx.bind_to_thread()?;
        self.poll()?;
        let (lease, copies) = self.pool.fork(parent, len, tokens)?;
        self.run_copies(&copies.pages, copies.slot.as_slice())?;
        Ok(lease)
    }

    /// Fill the first `len` positions of `dst`, a fresh lease, with a copy
    /// of `src`'s: pages and state slot copied on the stream, nothing
    /// shared. A bench reaches a long context at copy speed this way, and
    /// every sequence of a batch still reads its own pages.
    pub fn replicate(&mut self, src: &Lease, dst: &Lease, len: usize) -> Result<()> {
        self.ctx.bind_to_thread()?;
        let c = dst.replica_of(src, len);
        self.run_copies(&c.pages, c.slot.as_slice())
    }

    /// Run another runtime's pool decisions on this one's states, `(from,
    /// to)` in order: whole pages of every paged state, then whole slots
    /// of every per-sequence state. For a runtime whose pages and slots
    /// another runtime's pool hands out (a pipeline stage holds the head's
    /// ids for its own layers; a decode rank, its own as the head's pool
    /// for it decides them). That pool knows which are free; this one only
    /// refuses an id past its own.
    pub fn copy(&mut self, pages: &[(i32, i32)], slots: &[(i32, i32)]) -> Result<()> {
        self.ctx.bind_to_thread()?;
        let (pages_max, slots_max) = (self.pool.pages() as i32, self.provision.seq_slots as i32);
        let outside = |ids: &[(i32, i32)], arena: std::ops::Range<i32>| {
            ids.iter().copied().find(|(a, b)| !arena.contains(a) || !arena.contains(b))
        };
        if let Some((a, b)) = outside(pages, 0..pages_max) {
            bail!(Api, "copying page {a} to {b}: this runtime holds pages 0..{pages_max}");
        }
        if let Some((a, b)) = outside(slots, 1..slots_max) {
            bail!(Api, "copying slot {a} to {b}: this runtime holds slots 1..{slots_max}");
        }
        self.run_copies(pages, slots)
    }

    /// Run byte moves on the stream: whole pages of every paged state,
    /// whole slots of every per-sequence state.
    fn run_copies(&mut self, pages: &[(i32, i32)], slots: &[(i32, i32)]) -> Result<()> {
        let unit = self.pool.unit();
        for (name, st) in &self.manifest.states {
            let s = &self.states[name];
            let moves: Vec<(u64, u64, u64)> = if st.is_per_seq() {
                slots.iter().map(|&(a, b)| (a as u64, b as u64, st.bytes_per_seq)).collect()
            } else if st.bytes_per_token > 0 {
                pages.iter().map(|&(a, b)| (a as u64, b as u64, unit * st.bytes_per_token)).collect()
            } else {
                Vec::new()
            };
            for (a, b, bytes) in moves {
                let src = s.view((a * bytes) as usize..((a + 1) * bytes) as usize)?;
                let mut dst = s.view((b * bytes) as usize..((b + 1) * bytes) as usize)?;
                self.stream.memcpy_dtod(&src, &mut dst)?;
            }
        }
        Ok(())
    }

    /// Return the checkpoints whose park copies have landed to the pool.
    pub(crate) fn poll(&mut self) -> Result<()> {
        let mut parking = std::mem::take(&mut self.parking);
        for (cp, ev) in parking.drain(..) {
            if landed(ev)? {
                unsafe { sys::cuEventDestroy_v2(ev) };
                drop(cp);
            } else {
                self.parking.push((cp, ev));
            }
        }
        Ok(())
    }
}
