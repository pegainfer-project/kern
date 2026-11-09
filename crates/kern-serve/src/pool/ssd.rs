//! The pooled tier on this rank's local disk: one file, written through on
//! every snapshot the scheduler puts, read back into a lease when a
//! prompt's probe finds its prefix there.
//!
//! What moves where is [`Disk`]'s to decide (kern-pool: the file as a
//! ring evicted oldest first, the pinned block as rooms, and the plan each
//! act gets); this is the shell that moves the bytes. Nothing outlives the
//! process: the file is made without a name (`O_TMPFILE`), so no other
//! process opens it and no byte of another run is ever read. The tier is a
//! cache, not storage.
//!
//! The disk reaches no device memory, so every byte passes through the
//! pinned block, copied on the compute stream. A put copies its objects'
//! pages and slot into rooms behind everything issued so far and marks the
//! stream; the write thread waits for the mark before the disk reads the
//! rooms. The put is answered as soon as the stream passed the mark, so
//! the scheduler lets go of its pages while the disk is still writing:
//! write-through never holds prefill to the disk's speed, and a backlog
//! the block cannot stage is refused instead. A get is read into rooms by the read thread, and the next
//! [`Client::issue`], on the thread that drives the device, copies them
//! into the lease's pages and slot. That copy is in stream order ahead of
//! everything the scheduler issues once it hears the get landed, so the
//! get is answered as soon as the copy is issued; its rooms go back once
//! the stream has passed it. Neither disk thread calls CUDA (lessons.md,
//! 2026-10-08). A put whose mark the stream never passed keeps its rooms
//! forever: its copy may still land in them. A mark the stream is slow to
//! pass is waited for as long as the client lives; a stream that stopped
//! fails the rank by its own path.
//!
//! The disk is driven through io_uring with `O_DIRECT`, one ring per
//! thread. Each thread takes every job queued as one batch (a paged-only
//! manifest puts each page as it is sealed, one small job at a time), cuts
//! its objects into chunks and keeps them all in flight up to the queue
//! depth, so every member of a RAID works. A thread that stops (its
//! io_uring failed with requests in flight) fails the rank at the next
//! order rather than leave a fetch unanswered.

#![allow(unsafe_code)]

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::io::AsRawFd;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use io_uring::{opcode, types, IoUring};
use kern_pool::{Disk, Move, Object, Plan, Room};
use kern_runtime::{Mark, Pinned, Runtime, Span};
use tracing::{debug, info};

use super::{buffers, Buffers};
use crate::ranks::Landed;

/// `O_DIRECT`'s unit on 4Kn and 512e disks alike; the block and the file
/// place everything on it.
const ALIGN: u64 = 4096;
/// The most bytes one read or write moves.
const CHUNK: u64 = 2 << 20;
/// Chunks in flight per thread.
const DEPTH: u32 = 64;

#[derive(Debug, Clone)]
pub struct Config {
    /// The directory the file is made in.
    pub dir: PathBuf,
    /// Bytes of the file, all reserved at open.
    pub capacity: u64,
    /// Bytes of the pinned block every transfer passes through.
    pub staging: u64,
}

/// What the disk threads and the issuing thread share.
struct Shared {
    file: File,
    disk: Mutex<Disk>,
    /// The pinned block's address; the block itself stays with the client.
    base: u64,
    /// The client is dropping: a mark not passed yet never will be.
    closing: AtomicBool,
}

/// A put on its way to the write thread: written once `after` passed.
struct Write {
    plan: Plan,
    after: Mark,
}

/// A get on its way through the read thread, and each move's bytes on the
/// device to copy it into.
struct Read {
    id: u64,
    plan: Plan,
    device: Vec<Buffers>,
}

/// This rank's disk tier: the file, the block, a write and a read thread.
pub struct Client {
    shared: Arc<Shared>,
    spans: BTreeMap<String, Span>,
    writes: Option<Sender<Write>>,
    reads: Option<Sender<Read>>,
    /// Gets read into the block, for [`Client::issue`] to copy in.
    read: Receiver<Read>,
    done_tx: Sender<Landed>,
    done: Receiver<Landed>,
    /// Rooms of gets copied in, until the stream passes the copy.
    copying: RefCell<Vec<(Mark, Vec<Room>)>>,
    /// Answers to puts, once the stream passes their copy out.
    staged: RefCell<Vec<(Mark, Landed)>>,
    threads: Vec<JoinHandle<()>>,
    /// Freed once both threads stopped and every copy either way ran;
    /// leaked if one never did.
    block: Option<Pinned>,
}

impl Client {
    /// Make the file, reserve all of it, pin the block, start the threads.
    pub fn open(cfg: &Config, rt: &Runtime, spans: BTreeMap<String, Span>) -> Result<Client> {
        std::fs::create_dir_all(&cfg.dir).with_context(|| format!("creating {}", cfg.dir.display()))?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_TMPFILE | libc::O_DIRECT)
            .open(&cfg.dir)
            .with_context(|| format!("making a file in {} with O_TMPFILE | O_DIRECT", cfg.dir.display()))?;
        let t0 = Instant::now();
        let block = rt.pinned(cfg.staging).context("pinning the disk tier's staging block")?;
        let disk = Disk::new(cfg.capacity, block.bytes(), ALIGN);
        // Reserved whole, so a disk short of room fails here rather than
        // on every write.
        if unsafe { libc::fallocate(file.as_raw_fd(), 0, 0, disk.capacity() as libc::off_t) } != 0 {
            bail!(
                "reserving {} GiB in {}: {}",
                disk.capacity() >> 30,
                cfg.dir.display(),
                std::io::Error::last_os_error()
            );
        }
        info!(
            dir = %cfg.dir.display(),
            capacity_gib = disk.capacity() >> 30,
            staging_gib = disk.staging() >> 30,
            open_s = t0.elapsed().as_secs_f64(),
            states = spans.len(),
            "pool: disk tier open"
        );
        let shared =
            Arc::new(Shared { file, disk: Mutex::new(disk), base: block.ptr(), closing: AtomicBool::new(false) });
        let (done_tx, done) = channel();
        let (read_tx, read) = channel();
        let (writes, write_rx) = channel();
        let (reads, read_rx) = channel();
        let spawn = |name: &str, f: Box<dyn FnOnce() + Send>| {
            std::thread::Builder::new().name(format!("kern-pool-{name}")).spawn(f).context("spawning a disk thread")
        };
        let threads = vec![
            spawn("write", {
                let (sh, mut uring) = (Arc::clone(&shared), uring()?);
                Box::new(move || write_loop(&sh, &mut uring, write_rx))
            })?,
            spawn("read", {
                let (sh, mut uring) = (Arc::clone(&shared), uring()?);
                Box::new(move || read_loop(&sh, &mut uring, read_rx, &read_tx))
            })?,
        ];
        Ok(Client {
            shared,
            spans,
            writes: Some(writes),
            reads: Some(reads),
            read,
            done_tx,
            done,
            copying: RefCell::new(Vec::new()),
            staged: RefCell::new(Vec::new()),
            threads,
            block: Some(block),
        })
    }

    fn disk(&self) -> std::sync::MutexGuard<'_, Disk> {
        self.shared.disk.lock().expect("not poisoned")
    }

    fn answer(&self, id: u64, ok: Vec<bool>) {
        let _ = self.done_tx.send(Landed { id, ok });
    }

    pub fn probe(&self, id: u64, keys: &[String]) {
        let ok = self.disk().probe(keys);
        self.answer(id, ok);
    }

    pub fn mark(&self, id: u64, key: &str) {
        let ok = self.disk().mark(key);
        self.answer(id, vec![ok]);
    }

    /// Each object's bytes on this rank, and its key and size as the plan
    /// takes it.
    fn sized<'o>(&self, objects: &'o [Object]) -> (Vec<Buffers>, Vec<(&'o str, u64)>) {
        let bufs: Vec<Buffers> = objects.iter().map(|o| buffers(&self.spans, o)).collect();
        let sized = objects.iter().zip(&bufs).map(|(o, b)| (o.key.as_str(), bytes(b))).collect();
        (bufs, sized)
    }

    /// Copy every object the plan moves into its room behind what `rt`
    /// has issued, and hand the writes to the write thread.
    pub fn put(&self, rt: &Runtime, id: u64, objects: &[Object]) -> Result<()> {
        let (bufs, sized) = self.sized(objects);
        let plan = self.disk().put(&sized);
        if plan.moves.is_empty() {
            self.answer(id, plan.ok);
            return Ok(());
        }
        let device: Vec<&Buffers> = plan.moves.iter().map(|m| &bufs[m.at]).collect();
        let runs = runs(self.shared.base, &plan.moves, &device, Way::Out);
        // SAFETY: every run is between a pooled state of `rt` and a room of
        // this client's block that the plan holds.
        let after = match unsafe { rt.copy_bytes(&runs) }.and_then(|()| rt.mark()) {
            Ok(m) => m,
            Err(e) => {
                self.disk().abandon(plan);
                return Err(e.into());
            }
        };
        self.staged.borrow_mut().push((after.clone(), Landed { id, ok: plan.ok.clone() }));
        send(&self.writes, Write { plan, after }, "write")
    }

    /// Hand the plan's reads to the read thread; [`Client::issue`] copies
    /// what landed in.
    pub fn get(&self, id: u64, objects: &[Object]) -> Result<()> {
        let (mut bufs, sized) = self.sized(objects);
        let plan = self.disk().get(&sized);
        if plan.moves.is_empty() {
            self.answer(id, plan.ok);
            return Ok(());
        }
        let device = plan.moves.iter().map(|m| std::mem::take(&mut bufs[m.at])).collect();
        send(&self.reads, Read { id, plan, device }, "read")
    }

    /// Copy every get the read thread has landed into its lease, answer
    /// it, and take back the rooms of copies the stream has passed.
    pub fn issue(&self, rt: &Runtime) -> Result<()> {
        if let Some(t) = self.threads.iter().find(|t| t.is_finished()) {
            bail!("the disk tier's {} thread has stopped", t.thread().name().unwrap_or("?"));
        }
        let mut copying = self.copying.borrow_mut();
        let (passed, running): (Vec<_>, Vec<_>) =
            std::mem::take(&mut *copying).into_iter().partition(|(m, _)| m.passed());
        *copying = running;
        let mut give = passed.into_iter().flat_map(|(_, rooms)| rooms).collect::<Vec<_>>();
        for Read { id, plan, device } in self.read.try_iter() {
            let (landed, failed): (Vec<_>, Vec<_>) =
                plan.moves.into_iter().zip(device).partition(|(m, _)| plan.ok[m.at]);
            give.extend(failed.into_iter().map(|(m, _)| m.room));
            if !landed.is_empty() {
                let (moves, device): (Vec<Move>, Vec<Buffers>) = landed.into_iter().unzip();
                let runs = runs(self.shared.base, &moves, &device.iter().collect::<Vec<_>>(), Way::In);
                // SAFETY: as in `put`, the other way.
                unsafe { rt.copy_bytes(&runs) }?;
                copying.push((rt.mark()?, moves.into_iter().map(|m| m.room).collect()));
            }
            self.answer(id, plan.ok);
        }
        let mut disk = self.disk();
        give.into_iter().for_each(|r| disk.give(r));
        Ok(())
    }

    pub fn landed(&self) -> Vec<Landed> {
        let mut staged = self.staged.borrow_mut();
        let (passed, running): (Vec<_>, Vec<_>) =
            std::mem::take(&mut *staged).into_iter().partition(|(m, _)| m.passed());
        *staged = running;
        passed.into_iter().map(|(_, l)| l).chain(self.done.try_iter()).collect()
    }
}

impl Drop for Client {
    /// The block outlives everything that moves through it: the threads
    /// finish their jobs, and every copy either way is waited for. A copy
    /// the stream never ran leaves the block leaked rather than freed under
    /// it.
    fn drop(&mut self) {
        self.shared.closing.store(true, Ordering::Release);
        self.writes.take();
        self.reads.take();
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
        let mut copies =
            self.copying.take().into_iter().map(|(m, _)| m).chain(self.staged.take().into_iter().map(|(m, _)| m));
        if !copies.all(|m| m.wait().is_ok()) {
            std::mem::forget(self.block.take());
        }
    }
}

fn send<T>(to: &Option<Sender<T>>, job: T, what: &str) -> Result<()> {
    let tx = to.as_ref().expect("open until dropped");
    tx.send(job).ok().with_context(|| format!("the disk tier's {what} thread has stopped"))
}

fn bytes(b: &Buffers) -> u64 {
    b.iter().map(|&(_, n)| n as u64).sum()
}

#[derive(Clone, Copy)]
enum Way {
    Out,
    In,
}

/// The copies between each move's device bytes and its room, state by
/// state back to back, as `(dst, src, bytes)`.
fn runs(base: u64, moves: &[Move], device: &[&Buffers], way: Way) -> Vec<(u64, u64, u64)> {
    moves
        .iter()
        .zip(device)
        .flat_map(|(m, dev)| {
            let room = base + m.room.offset();
            dev.iter().scan(0u64, move |at, &(d, n)| {
                let host = room + *at;
                *at += n as u64;
                Some(match way {
                    Way::Out => (host, d, n as u64),
                    Way::In => (d, host, n as u64),
                })
            })
        })
        .collect()
}

fn uring() -> Result<IoUring> {
    IoUring::new(DEPTH).context("creating an io_uring for the disk tier")
}

/// Every job queued behind `first`: the disk threads take what has piled
/// up as one batch, so a stream of one-page puts still keeps the queue deep.
fn batch<T>(first: T, jobs: &Receiver<T>) -> Vec<T> {
    std::iter::once(first).chain(jobs.try_iter()).collect()
}

/// Write every queued put's rooms once the stream passed its mark, then
/// settle each put in the order it was planned.
fn write_loop(sh: &Shared, uring: &mut IoUring, jobs: Receiver<Write>) {
    while let Ok(first) = jobs.recv() {
        let jobs = batch(first, &jobs);
        let t0 = Instant::now();
        // Marks are passed in the order they were taken: the last one
        // passed, every one has.
        let ready = passes(sh, &jobs.last().expect("a batch of one at least").after);
        let moves: Vec<&Move> = jobs.iter().flat_map(|j| &j.plan.moves).collect();
        let (n, mib) = (moves.len(), moves.iter().map(|m| m.extent.len()).sum::<u64>() >> 20);
        let wrote = match ready {
            true => transfer(uring, sh, &moves, Way::Out),
            false => Vec::new(),
        };
        let mut wrote = wrote.into_iter();
        let mut disk = sh.disk.lock().expect("not poisoned");
        let kept: usize = jobs
            .into_iter()
            .map(|j| match ready {
                true => {
                    let landed: Vec<bool> = wrote.by_ref().take(j.plan.moves.len()).collect();
                    disk.wrote(j.plan, &landed)
                }
                false => {
                    disk.abandon(j.plan);
                    0
                }
            })
            .sum();
        drop(disk);
        log("write", n, kept, mib, t0);
    }
}

/// Read every queued get's extents into its rooms and hand each back to be
/// copied in.
fn read_loop(sh: &Shared, uring: &mut IoUring, jobs: Receiver<Read>, read: &Sender<Read>) {
    while let Ok(first) = jobs.recv() {
        let mut jobs = batch(first, &jobs);
        let t0 = Instant::now();
        let moves: Vec<&Move> = jobs.iter().flat_map(|j| &j.plan.moves).collect();
        let (n, mib) = (moves.len(), moves.iter().map(|m| m.extent.len()).sum::<u64>() >> 20);
        let mut got = transfer(uring, sh, &moves, Way::In).into_iter();
        let disk = sh.disk.lock().expect("not poisoned");
        for j in &mut jobs {
            let g: Vec<bool> = got.by_ref().take(j.plan.moves.len()).collect();
            disk.read(&mut j.plan, &g);
        }
        drop(disk);
        let kept = jobs.iter().map(|j| j.plan.moves.iter().filter(|m| j.plan.ok[m.at]).count()).sum();
        log("read", n, kept, mib, t0);
        for j in jobs {
            if let Err(e) = read.send(j) {
                let mut disk = sh.disk.lock().expect("not poisoned");
                e.0.plan.moves.into_iter().for_each(|m| disk.give(m.room));
            }
        }
    }
}

/// Whether the stream passes `m` before the client is dropped.
fn passes(sh: &Shared, m: &Mark) -> bool {
    loop {
        if m.clone().wait().is_ok() {
            return true;
        }
        if sh.closing.load(Ordering::Acquire) {
            return false;
        }
    }
}

fn log(what: &str, objects: usize, landed: usize, mib: u64, t0: Instant) {
    let s = t0.elapsed().as_secs_f64();
    debug!(what, objects, landed, mib, ms = (s * 1e3) as u64, gib_s = mib as f64 / 1024.0 / s.max(1e-9), "pool: disk");
}

/// Move every move's extent between the file and its room, `CHUNK` bytes
/// per request and `DEPTH` requests in flight: whether each move landed
/// whole. Returns only once nothing is in flight, so a room is never
/// given back while the disk still moves it.
fn transfer(uring: &mut IoUring, sh: &Shared, moves: &[&Move], way: Way) -> Vec<bool> {
    let fd = types::Fd(sh.file.as_raw_fd());
    let chunks: Vec<(usize, u64, u64, u32)> = moves
        .iter()
        .enumerate()
        .flat_map(|(i, m)| {
            let (file, host, len) = (m.extent.offset(), sh.base + m.room.offset(), m.extent.len());
            (0..len).step_by(CHUNK as usize).map(move |o| (i, file + o, host + o, CHUNK.min(len - o) as u32))
        })
        .collect();
    let mut ok = vec![true; moves.len()];
    let (mut next, mut inflight) = (0usize, 0u32);
    while next < chunks.len() || inflight > 0 {
        while next < chunks.len() && inflight < DEPTH {
            let (_, file, host, len) = chunks[next];
            let sqe = match way {
                Way::Out => opcode::Write::new(fd, host as *const u8, len).offset(file).build(),
                Way::In => opcode::Read::new(fd, host as *mut u8, len).offset(file).build(),
            }
            .user_data(next as u64);
            // SAFETY: the room is this job's until it is settled, and the
            // block outlives both threads (`Client::drop` joins them).
            if unsafe { uring.submission().push(&sqe) }.is_err() {
                break;
            }
            next += 1;
            inflight += 1;
        }
        match uring.submit_and_wait(1) {
            Ok(_) => {}
            // Interrupted or out of room: reap what completed and go on.
            Err(e) if matches!(e.raw_os_error(), Some(libc::EINTR | libc::EAGAIN | libc::EBUSY)) => {}
            // Requests may still be in flight into the rooms: nothing can
            // be given back, so the thread stops and the rank fails at its
            // next order.
            Err(e) => panic!("the disk tier's io_uring failed: {e}"),
        }
        for cqe in uring.completion() {
            let (i, _, _, len) = chunks[cqe.user_data() as usize];
            ok[i] &= cqe.result() == len as i32;
            inflight -= 1;
        }
    }
    ok
}
