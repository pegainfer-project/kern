//! The lease shell. The pool decides on the host which pages and slots
//! a sequence gets and which bytes move; the runtime runs those moves on
//! the compute stream and the pool's remaps on a thread of its own.
//!
//! Every lease operation begins with `poll`: land the remaps the thread
//! has finished, their fresh chunks zeroed on the stream so a page or a
//! slot comes out of the pool as it did at load, then hand the thread the
//! plan the pool has pending, to run once both streams have passed
//! everything enqueued so far. One remap is in flight at a time; a
//! [`Denied::Remapping`] tells the caller to ask again once it lands. A
//! leased slot is zeroed on the stream (a fresh sequence's recurrent
//! state); a checkpoint, a restore and a fork move whole pages and whole
//! slots, as the pool's [`Copies`] say.
//!
//! The pool may also live in another process: a serving head holds one
//! for every rank, built from the rank's [`Runtime::budget`], and the
//! rank's runtime only runs what it decided — [`Runtime::remap`] /
//! [`Runtime::remapped`] for the remaps, [`Runtime::copy`] for the byte
//! moves, [`Runtime::zero_slot`] for a fresh lease's slot. Such a runtime
//! leases nothing of its own; its own pool stays as load left it.

use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::Arc;
use std::thread::JoinHandle;

use cudarc::driver::{sys, CudaContext};

use crate::device::{landed, record, Mapper};
use crate::error::{bail, cuda_check};
use crate::{Error, Result, Runtime};
use kern_pool::{Checkpoint, Copies, Denied, Lease, Remap};

/// What the remap thread is told: a plan to run once both streams have
/// passed `after` (recorded events, destroyed by the thread), or to stop.
enum Job {
    Run { plan: Remap, after: [u64; 2] },
    Stop,
}

/// The remap thread and the channels to it.
pub(crate) struct Remaps {
    jobs: Sender<Job>,
    done: Receiver<Result<Remap>>,
    thread: Option<JoinHandle<()>>,
}

impl Remaps {
    /// Start the thread that owns `mapper` from here on.
    pub(crate) fn spawn(ctx: Arc<CudaContext>, mapper: Mapper) -> Result<Remaps> {
        let (jobs, job_rx) = mpsc::channel();
        let (done_tx, done) = mpsc::channel();
        let thread = std::thread::Builder::new()
            .name("kern-remap".into())
            .spawn(move || remap_thread(ctx, mapper, job_rx, done_tx))
            .map_err(|e| Error::Cuda(format!("spawning the remap thread: {e}")))?;
        Ok(Remaps { jobs, done, thread: Some(thread) })
    }

    /// Tell the thread to stop and wait for it; the arenas go with it.
    pub(crate) fn stop(&mut self) {
        let _ = self.jobs.send(Job::Stop);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// Runs plans until told to stop, then lets the arenas and chunks go.
fn remap_thread(ctx: Arc<CudaContext>, mut mapper: Mapper, jobs: Receiver<Job>, done: Sender<Result<Remap>>) {
    if let Err(e) = ctx.bind_to_thread() {
        let _ = done.send(Err(e.into()));
        return;
    }
    while let Ok(job) = jobs.recv() {
        match job {
            Job::Run { plan, after } => {
                let evs = after.map(|e| e as sys::CUevent);
                let r = evs
                    .iter()
                    .try_for_each(|&ev| cuda_check(unsafe { sys::cuEventSynchronize(ev) }, "cuEventSynchronize"))
                    .and_then(|()| mapper.run(&plan));
                for ev in evs {
                    unsafe { sys::cuEventDestroy_v2(ev) };
                }
                if done.send(r.map(|()| plan)).is_err() {
                    break;
                }
            }
            Job::Stop => break,
        }
    }
    drop(mapper);
}

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
        let lease = match self.pool.lease(tokens) {
            Ok(l) => l,
            Err(d) => return self.denied(d),
        };
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
        let lease = match self.pool.lease_slot() {
            Ok(l) => l,
            Err(d) => return self.denied(d),
        };
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
    /// exist; this one only refuses a slot past the arena.
    pub fn zero_slot(&mut self, slot: i32) -> Result<()> {
        self.ctx.bind_to_thread()?;
        let max = self.provision.seq_slots as usize;
        if slot < 1 || slot as usize >= max {
            bail!(Api, "slot {slot}: this runtime's arena holds slots 1..{max}");
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
        let (cp, copies) = match self.pool.checkpoint(lease, len) {
            Ok(x) => x,
            Err(d) => return self.denied(d),
        };
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
        let (lease, copies) = match self.pool.restore(cp, len, tokens) {
            Ok(x) => x,
            Err(d) => return self.denied(d),
        };
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
        let (lease, copies) = match self.pool.fork(parent, len, tokens) {
            Ok(x) => x,
            Err(d) => return self.denied(d),
        };
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
    /// for it decides them). That pool knows which exist; this one only
    /// refuses an id past the arena.
    pub fn copy(&mut self, pages: &[(i32, i32)], slots: &[(i32, i32)]) -> Result<()> {
        self.ctx.bind_to_thread()?;
        let (pages_max, slots_max) = (self.pool.pages_max() as i32, self.provision.seq_slots as i32);
        let outside = |ids: &[(i32, i32)], arena: std::ops::Range<i32>| {
            ids.iter().copied().find(|(a, b)| !arena.contains(a) || !arena.contains(b))
        };
        if let Some((a, b)) = outside(pages, 0..pages_max) {
            bail!(Api, "copying page {a} to {b}: this runtime's arena holds pages 0..{pages_max}");
        }
        if let Some((a, b)) = outside(slots, 1..slots_max) {
            bail!(Api, "copying slot {a} to {b}: this runtime's arena holds slots 1..{slots_max}");
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

    /// A denial, after handing the remap it may have planned to the thread.
    pub(crate) fn denied<T>(&mut self, d: Denied) -> Result<T> {
        self.poll()?;
        Err(d.into())
    }

    /// Land the remaps the thread has run — their fresh chunks zeroed on
    /// the stream, so a page or a slot comes out of the pool as it did at
    /// load — and hand the thread the plan the pool has pending: it runs
    /// once the stream has passed everything enqueued so far.
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
        loop {
            match self.remaps.done.try_recv() {
                Ok(Ok(plan)) => {
                    self.zero_fresh(&plan)?;
                    self.pool.complete(plan);
                }
                Ok(Err(e)) => return Err(e),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => bail!(Cuda, "the remap thread is gone"),
            }
        }
        if let Some(plan) = self.pool.take_pending() {
            let after = [record(&self.stream)? as u64, record(&self.xfer)? as u64];
            self.remap_count += 1;
            tracing::debug!(unmap = plan.unmap.len(), map = plan.map.len(), "remap");
            if self.remaps.jobs.send(Job::Run { plan, after }).is_err() {
                bail!(Cuda, "the remap thread is gone");
            }
        }
        Ok(())
    }

    /// Land every remap, blocking until the thread has run the one in
    /// flight and whatever the pool plans after it: the ask a
    /// [`Denied::Remapping`] refused is answered on the next call.
    pub fn settle(&mut self) -> Result<()> {
        self.ctx.bind_to_thread()?;
        self.poll()?;
        while self.pool.remapping() {
            match self.remaps.done.recv() {
                Ok(Ok(plan)) => {
                    self.zero_fresh(&plan)?;
                    self.pool.complete(plan);
                }
                Ok(Err(e)) => return Err(e),
                Err(_) => bail!(Cuda, "the remap thread is gone"),
            }
            self.poll()?;
        }
        Ok(())
    }

    /// Zero every chunk `plan` mapped.
    pub(crate) fn zero_fresh(&mut self, plan: &Remap) -> Result<()> {
        let chunk = self.pool.chunk() as usize;
        for &(a, p, _) in &plan.map {
            let s = &self.states[&self.pool.pooled()[a].state];
            let mut v = s.view(p * chunk..(p + 1) * chunk)?;
            self.stream.memset_zeros(&mut v)?;
        }
        Ok(())
    }

    /// Remaps planned so far.
    pub fn remaps(&self) -> u64 {
        self.remap_count
    }

    /// Run `plan`, a remap another process's pool for this runtime planned
    /// (one built from [`Runtime::budget`]), on the remap thread once both
    /// streams have passed everything enqueued so far. One at a time, as
    /// that pool plans them; [`Runtime::remapped`] says when it has landed.
    /// A runtime driven this way leases nothing itself. A plan that does
    /// not follow from the placement the earlier ones left (a chunk at two
    /// positions, an unmap of an empty one) is refused before it runs.
    pub fn remap(&mut self, plan: Remap) -> Result<()> {
        self.ctx.bind_to_thread()?;
        let next = self.driven.apply(&plan, self.budget.chunks)?;
        let after = [record(&self.stream)? as u64, record(&self.xfer)? as u64];
        self.remap_count += 1;
        if self.remaps.jobs.send(Job::Run { plan, after }).is_err() {
            bail!(Cuda, "the remap thread is gone");
        }
        self.driven = next;
        Ok(())
    }

    /// How many of the remaps [`Runtime::remap`] was handed have landed
    /// since the last call, their fresh chunks zeroed on the stream ahead of
    /// anything issued after; `wait` blocks until one lands (call it so
    /// only with one in flight).
    pub fn remapped(&mut self, wait: bool) -> Result<u64> {
        self.ctx.bind_to_thread()?;
        let mut n = 0;
        loop {
            let done = match wait && n == 0 {
                true => self.remaps.done.recv().map_err(|_| TryRecvError::Disconnected),
                false => self.remaps.done.try_recv(),
            };
            match done {
                Ok(plan) => {
                    self.zero_fresh(&plan?)?;
                    n += 1;
                }
                Err(TryRecvError::Empty) => return Ok(n),
                Err(TryRecvError::Disconnected) => bail!(Cuda, "the remap thread is gone"),
            }
        }
    }
}
