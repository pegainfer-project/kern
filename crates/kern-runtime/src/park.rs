//! The host tier shell: checkpoints parked in pinned DRAM and woken
//! straight into the lease of the sequence continuing from them.
//! [`kern_pool::Host`] decides where a checkpoint's bytes go; this is the
//! runtime running the copies.
//!
//! Two streams: every program and every pool copy runs on the compute
//! stream in order; the host tier's copies ([`Runtime::park`] out,
//! [`Runtime::wake`] in) run on a transfer stream, and the compute stream
//! never waits for them. The transfer stream starts each batch of copies
//! after everything the compute stream has enqueued; a parked checkpoint
//! stays held (its pages and slot out of the pool) until its copy has
//! landed, and a woken lease is a [`Waking`] until [`Runtime::awake`]
//! finds its copy landed, so no program can read pages still in flight.
//! Parking is two steps, [`Runtime::room`] then [`Runtime::park`], so a
//! caller parking several checkpoints as one unit can find room for all
//! of them before any byte moves.
//!
//! [`kern_pool::Host`] counts host pages and host slots; the bytes are
//! this module's. The pinned block is the host pages, each every paged
//! state's page back to back in arena order, then the host slots, each
//! every per-sequence state's slot likewise, so unit `u` of a region is
//! `u` times the unit's bytes past the region's start.
//!
//! A head that keeps this runtime's accounting in another process holds
//! the [`Host`] itself (`Host::new` over the units [`host_split`] cuts from
//! the block it told this runtime to reserve) and sends only its plans:
//! [`Runtime::host_out`] and [`Runtime::host_in`] run them on the same
//! transfer stream and hand back a [`Landing`] the head waits on before it
//! reuses the pages.
//!
//! A pipeline has no decode step to keep off the copies' path, so its
//! host moves are stream-ordered instead: [`Runtime::park_now`] and
//! [`Runtime::wake_now`] decide and run on the compute stream, behind
//! everything enqueued and ahead of everything after, and
//! [`Runtime::copy_out`] / [`Runtime::copy_in`] run the same plans for a
//! stage that holds the head's ids for its own layers. A parked
//! checkpoint's pages return at once and a woken lease is usable at once:
//! whatever writes or reads them is a later launch on the same stream.

use std::sync::Arc;

use cudarc::driver::{sys, CudaStream};

use crate::device::{copy_2d, landed, record, wait_then_destroy, Pinned, Space};
use crate::error::bail;
use crate::{Error, Result, Runtime};
use kern_pool::{runs, Checkpoint, Copies, Denied, Host, Kind, Lease, Parked};

/// The host tier once reserved: its accounting, its pinned block, and
/// the host pages ahead of the slots in it.
pub(crate) struct Block {
    host: Arc<Host>,
    pinned: Pinned,
    pages: u64,
    slots: u64,
}

/// The host pages and host slots `bytes` hold, pages of `page_bytes` and
/// slots of `slot_bytes`: one slot per `turn` pages, the mix parked
/// checkpoints take when each holds its own slot and the pages of the one
/// turn it adds to the prefix it shares with its session's other turns;
/// all of it to the one kind a manifest has when it has only one.
pub fn host_split(bytes: u64, page_bytes: u64, slot_bytes: u64, turn: u64) -> (u64, u64) {
    match (page_bytes, slot_bytes) {
        (0, 0) => (0, 0),
        (0, s) => (0, bytes / s),
        (p, 0) => (bytes / p, 0),
        (p, s) => {
            let slots = bytes / turn.saturating_mul(p).saturating_add(s);
            ((bytes - slots * s) / p, slots)
        }
    }
}

/// A checkpoint with host room found for it and nothing copied yet: what
/// [`Runtime::room`] hands out and [`Runtime::park`] spends. Dropping it
/// frees the room and drops the checkpoint.
pub struct Room {
    cp: Checkpoint,
    parked: Parked,
    plan: Copies<i32, u64>,
}

impl Room {
    /// Give the room back and keep the checkpoint.
    pub fn into_checkpoint(self) -> Checkpoint {
        self.cp
    }
}

/// A lease being woken: its pages are taken, the parked prefix's bytes
/// still on the way in. [`Runtime::awake`] hands the lease out once they
/// have landed; dropping it earlier waits for them first, so the pages
/// never return to the pool with a copy still writing them.
pub struct Waking {
    lease: Option<Lease>,
    event: sys::CUevent,
}

// The event is only ever queried through the runtime that recorded it,
// on whatever thread drives that runtime; the runtime itself moves
// between threads the same way.
unsafe impl Send for Waking {}

impl Drop for Waking {
    fn drop(&mut self) {
        if !self.event.is_null() {
            unsafe {
                sys::cuEventSynchronize(self.event);
                sys::cuEventDestroy_v2(self.event);
            }
        }
    }
}

/// A batch of host transfers in flight. [`Landing::landed`] says when
/// its bytes are where the plan put them; dropping it earlier waits.
pub struct Landing {
    event: sys::CUevent,
}

// As for [`Waking`]: queried only on the thread that drives its runtime.
unsafe impl Send for Landing {}

impl Landing {
    /// Whether the copies have landed. Does not block.
    pub fn landed(&self) -> Result<bool> {
        landed(self.event)
    }
}

impl Drop for Landing {
    fn drop(&mut self) {
        unsafe {
            sys::cuEventSynchronize(self.event);
            sys::cuEventDestroy_v2(self.event);
        }
    }
}

impl Runtime {
    /// The host pages and host slots `bytes` of pinned memory hold for
    /// this runtime's states, one slot per `turn` tokens of pages
    /// ([`host_split`]).
    pub fn host_units(&self, bytes: u64, turn: u64) -> (u64, u64) {
        let (page_bytes, slot_bytes, _) = self.host_layout();
        host_split(bytes, page_bytes, slot_bytes, turn.div_ceil(self.pool.unit()))
    }

    /// Reserve page-locked host memory for `pages` host pages and `slots`
    /// host slots of parked checkpoints (once; ~100 ms per GiB on GB300).
    pub fn reserve_host(&mut self, pages: u64, slots: u64) -> Result<()> {
        self.ctx.bind_to_thread()?;
        if self.host.is_some() {
            bail!(Api, "the host tier is reserved already");
        }
        let (page_bytes, slot_bytes, _) = self.host_layout();
        let pinned = Pinned::alloc(pages * page_bytes + slots * slot_bytes, self.gpu as i32)?;
        self.host = Some(Block { host: Arc::new(Host::new(pages, slots)), pinned, pages, slots });
        Ok(())
    }

    /// (bytes used, bytes reserved) of the host tier, when there is one.
    pub fn host_tier(&self) -> Option<(u64, u64)> {
        let (page_bytes, slot_bytes, _) = self.host_layout();
        self.host.as_ref().map(|b| (b.host.used() * page_bytes + b.host.slots_used() * slot_bytes, b.pinned.bytes()))
    }

    /// Bytes per host page (every paged state's page back to back) and per
    /// host slot: 0 for a kind this runtime holds none of.
    pub fn host_unit(&self) -> (u64, u64) {
        let (page, slot, _) = self.host_layout();
        (page, slot)
    }

    /// Bytes per host page (every paged state's page back to back) and per
    /// host slot, and each pooled state's offset inside its one.
    fn host_layout(&self) -> (u64, u64, Vec<u64>) {
        let (mut page, mut slot) = (0u64, 0u64);
        let offsets = self
            .pool
            .pooled()
            .iter()
            .map(|a| match a.kind {
                Kind::Page => {
                    page += a.object;
                    page - a.object
                }
                Kind::Slot => {
                    slot += a.object;
                    slot - a.object
                }
            })
            .collect();
        (page, slot, offsets)
    }

    /// Host room for `cp` — its pages not parked already (an earlier turn
    /// of the same session shares them there) and its slot — with nothing
    /// copied yet: the first half of a park, so a caller parking several
    /// checkpoints as one unit can find room for all of them before any
    /// byte moves. `Err(cp)` hands the checkpoint back when the block
    /// cannot hold it. Dropping a [`Room`] frees the room and the
    /// checkpoint with it; [`Room::into_checkpoint`] keeps the checkpoint.
    pub fn room(&mut self, cp: Checkpoint) -> Result<std::result::Result<Room, Checkpoint>> {
        self.ctx.bind_to_thread()?;
        self.poll()?;
        let Some(Block { host, .. }) = &self.host else {
            bail!(Api, "no host tier: reserve_host first");
        };
        match host.park(&cp) {
            Ok((parked, plan)) => Ok(Ok(Room { cp, parked, plan })),
            Err(Denied::HostFull) => Ok(Err(cp)),
            Err(d) => Err(Error::Denied(d)),
        }
    }

    /// Copy a checkpoint into the room found for it. The checkpoint is
    /// held until the copies, on the transfer stream, have landed; then
    /// its device pages and slot return.
    pub fn park(&mut self, room: Room) -> Result<Parked> {
        self.ctx.bind_to_thread()?;
        let Room { cp, parked, plan } = room;
        self.transfer(&self.xfer, &plan.pages, plan.slot, true)?;
        self.parking.push((cp, record(&self.xfer)?));
        Ok(parked)
    }

    /// A sequence continuing from the first `len` tokens of a parked
    /// checkpoint with room for `tokens`: one lease, sized for the
    /// sequence, with those tokens' pages copied back in from the host
    /// and, when `len` is the whole checkpoint, its state into a fresh
    /// slot (a parked state is usable at its length only; a shorter
    /// `len` is a whole number of pages of a stateless one). A slot-only
    /// checkpoint wakes to a slot-only lease. The copies run on the
    /// transfer stream; [`Runtime::awake`] hands out the lease once they
    /// have landed, its parked pages twinned with their host copies, so
    /// its next checkpoint parks them for free. [`Error::Denied`] as for
    /// [`Runtime::lease`].
    pub fn wake(&mut self, p: &Parked, len: usize, tokens: usize) -> Result<Waking> {
        self.ctx.bind_to_thread()?;
        self.poll()?;
        let Some(Block { host, .. }) = &self.host else {
            bail!(Api, "no host tier: reserve_host first");
        };
        let unit = self.pool.unit() as usize;
        if len == 0 || len > p.tokens() || (len != p.tokens() && (p.has_slot() || !len.is_multiple_of(unit))) {
            bail!(Api, "waking {len} tokens of a parked checkpoint of {} ({p:?})", p.tokens());
        }
        let (lease, plan) = match Arc::clone(host).restore(p, &self.pool, len, tokens) {
            Ok(x) => x,
            Err(d) => return self.denied(d),
        };
        self.transfer(&self.xfer, &inward(&plan.pages), plan.slot.map(|(o, s)| (s, o)), false)?;
        Ok(Waking { lease: Some(lease), event: record(&self.xfer)? })
    }

    /// Run a park another process planned: its pages and slot out to the
    /// host block, after everything the compute stream has enqueued.
    pub fn host_out(&self, plan: &Copies<i32, u64>) -> Result<Landing> {
        self.ctx.bind_to_thread()?;
        self.check_host(plan.pages.iter().copied(), plan.slot)?;
        self.transfer(&self.xfer, &plan.pages, plan.slot, true)?;
        Ok(Landing { event: record(&self.xfer)? })
    }

    /// Run a wake another process planned: host offsets back into its
    /// pages and slot.
    pub fn host_in(&self, plan: &Copies<u64, i32>) -> Result<Landing> {
        self.ctx.bind_to_thread()?;
        let (pages, slot) = (inward(&plan.pages), plan.slot.map(|(o, s)| (s, o)));
        self.check_host(pages.iter().copied(), slot)?;
        self.transfer(&self.xfer, &pages, slot, false)?;
        Ok(Landing { event: record(&self.xfer)? })
    }

    /// Park `cp` in stream order: host room for its pages not parked
    /// already and its slot, copied out on the compute stream behind
    /// everything enqueued, so whatever writes its pages or slot next is a
    /// later launch and the checkpoint lets go of them at once. The plan
    /// comes back for runtimes holding the same ids to run after this one
    /// ([`Runtime::copy_out`]); `Ok(Err(cp))` hands the checkpoint back
    /// when the block cannot hold it.
    #[allow(clippy::type_complexity)]
    pub fn park_now(&mut self, cp: Checkpoint) -> Result<std::result::Result<(Parked, Copies<i32, u64>), Checkpoint>> {
        self.ctx.bind_to_thread()?;
        self.poll()?;
        let Some(Block { host, .. }) = &self.host else {
            bail!(Api, "no host tier: reserve_host first");
        };
        match host.park(&cp) {
            Ok((parked, plan)) => {
                self.transfer(&self.stream, &plan.pages, plan.slot, true)?;
                Ok(Ok((parked, plan)))
            }
            Err(Denied::HostFull) => Ok(Err(cp)),
            Err(d) => Err(Error::Denied(d)),
        }
    }

    /// A sequence continuing from the first `len` tokens of a parked
    /// checkpoint with room for `tokens`, as [`Runtime::wake`] makes it,
    /// copied in on the compute stream: the lease is usable at once by
    /// anything issued after. The plan comes back for runtimes holding the
    /// same ids ([`Runtime::copy_in`]). [`Error::Denied`] as for
    /// [`Runtime::lease`].
    pub fn wake_now(&mut self, p: &Parked, len: usize, tokens: usize) -> Result<(Lease, Copies<u64, i32>)> {
        self.ctx.bind_to_thread()?;
        self.poll()?;
        let Some(Block { host, .. }) = &self.host else {
            bail!(Api, "no host tier: reserve_host first");
        };
        let (lease, plan) = match Arc::clone(host).restore(p, &self.pool, len, tokens) {
            Ok(x) => x,
            Err(d) => return self.denied(d),
        };
        self.transfer(&self.stream, &inward(&plan.pages), plan.slot.map(|(o, s)| (s, o)), false)?;
        Ok((lease, plan))
    }

    /// Run a park another runtime's accounting decided on this one's
    /// states, on the compute stream in order, as [`Runtime::copy`] runs
    /// its copies. That accounting knows which ids exist; this one only
    /// refuses one past its arena or its block.
    pub fn copy_out(&mut self, plan: &Copies<i32, u64>) -> Result<()> {
        self.ctx.bind_to_thread()?;
        self.check_host(plan.pages.iter().copied(), plan.slot)?;
        self.transfer(&self.stream, &plan.pages, plan.slot, true)
    }

    /// Run a wake another runtime's accounting decided, as
    /// [`Runtime::copy_out`] runs a park.
    pub fn copy_in(&mut self, plan: &Copies<u64, i32>) -> Result<()> {
        self.ctx.bind_to_thread()?;
        let (pages, slot) = (inward(&plan.pages), plan.slot.map(|(o, s)| (s, o)));
        self.check_host(pages.iter().copied(), slot)?;
        self.transfer(&self.stream, &pages, slot, false)
    }

    /// Refuse a (device, host) page or slot past the arena or the block.
    fn check_host(&self, mut pages: impl Iterator<Item = (i32, u64)>, slot: Option<(i32, u64)>) -> Result<()> {
        let Some(b) = &self.host else {
            bail!(Api, "no host tier: reserve_host first");
        };
        let (pages_max, slots_max) = (self.pool.pages_max() as i32, self.provision.seq_slots as i32);
        if let Some((p, o)) = pages.find(|&(p, o)| !(0..pages_max).contains(&p) || o >= b.pages) {
            bail!(
                Api,
                "page {p} and host page {o}: this runtime holds pages 0..{pages_max} and host pages 0..{}",
                b.pages
            );
        }
        if let Some((s, o)) = slot.filter(|&(s, o)| !(1..slots_max).contains(&s) || o >= b.slots) {
            bail!(
                Api,
                "slot {s} and host slot {o}: this runtime holds slots 1..{slots_max} and host slots 0..{}",
                b.slots
            );
        }
        Ok(())
    }

    /// Whether a wake's copies have landed. Does not block.
    pub fn landed(&self, w: &Waking) -> Result<bool> {
        self.ctx.bind_to_thread()?;
        landed(w.event)
    }

    /// The lease of a wake whose copies have landed; `Err(w)` while they
    /// are still in flight. Does not block.
    pub fn awake(&self, mut w: Waking) -> Result<std::result::Result<Lease, Waking>> {
        self.ctx.bind_to_thread()?;
        if !landed(w.event)? {
            return Ok(Err(w));
        }
        unsafe { sys::cuEventDestroy_v2(w.event) };
        w.event = std::ptr::null_mut();
        Ok(Ok(w.lease.take().expect("a waking lease")))
    }

    /// (device page, host page) pages and a (device slot, host slot) slot
    /// between the pooled states and the host block, on `on`: the transfer
    /// stream after everything the compute stream has enqueued, or the
    /// compute stream itself. One strided copy per run of consecutive pages
    /// per state.
    fn transfer(
        &self,
        on: &Arc<CudaStream>,
        pages: &[(i32, u64)],
        slot: Option<(i32, u64)>,
        to_host: bool,
    ) -> Result<()> {
        let Some(block) = &self.host else {
            bail!(Api, "no host tier: reserve_host first");
        };
        let (page_bytes, slot_bytes, offsets) = self.host_layout();
        let (base, slots) = (block.pinned.ptr(), block.pinned.ptr() + block.pages * page_bytes);
        if !Arc::ptr_eq(on, &self.stream) {
            wait_then_destroy(on, record(&self.stream)?)?;
        }
        let stream = on.cu_stream();
        for (a, ar) in self.pool.pooled().iter().enumerate() {
            let st = &self.states[&ar.state];
            match ar.kind {
                Kind::Page => {
                    for (p, o, n) in runs(pages) {
                        let dev = (st.ptr + p as u64 * ar.object, ar.object, Space::Device);
                        let hst = (base + o * page_bytes + offsets[a], page_bytes, Space::Host);
                        let (dst, src) = if to_host { (hst, dev) } else { (dev, hst) };
                        copy_2d(stream, dst, src, ar.object, n as u64)?;
                    }
                }
                Kind::Slot => {
                    if let Some((s, o)) = slot {
                        let dev = (st.ptr + s as u64 * ar.object, ar.object, Space::Device);
                        let hst = (slots + o * slot_bytes + offsets[a], slot_bytes, Space::Host);
                        let (dst, src) = if to_host { (hst, dev) } else { (dev, hst) };
                        copy_2d(stream, dst, src, ar.object, 1)?;
                    }
                }
            }
        }
        Ok(())
    }
}

/// A wake's (host page, device page) pairs as (device page, host page).
fn inward(pages: &[(u64, i32)]) -> Vec<(i32, u64)> {
    pages.iter().map(|&(o, page)| (page, o)).collect()
}
