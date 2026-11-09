//! The pooled tier on this rank's local disk: one file, written through on
//! every snapshot the scheduler puts, read back into a lease when a
//! prompt's probe finds its prefix there.
//!
//! The file is a [`Ring`]: placed at the head, evicted oldest first, found
//! through an index in memory. Nothing outlives the process. The file is
//! truncated at open, so no byte of another run is ever read; the tier is
//! a cache, not storage.
//!
//! The disk reaches no device memory, so every byte passes through one
//! pinned block whose rooms an [`Arena`] hands out, copied on the compute
//! stream. A put copies its objects' pages and slot into rooms behind
//! everything issued so far and marks the stream; the write thread waits
//! for the mark before the disk reads the rooms. A get is read into rooms
//! by the read thread, and the next [`Client::issue`], on the thread that
//! drives the device, copies them into the lease's pages and slot. That
//! copy is in stream order ahead of everything the scheduler issues once
//! it hears the get landed, so the get is answered as soon as the copy is
//! issued; its rooms go back once the stream has passed it. Neither disk
//! thread calls CUDA (lessons.md, 2026-10-08).
//!
//! What is decided on the issuing thread, under the ring's lock, never
//! waits for the disk: a probe asks the index, a mark is an entry of no
//! bytes, a key written or being written is not copied again (every
//! snapshot puts its whole prefix), and a put or get that finds no room in
//! the ring or the block fails at once, which the scheduler answers by
//! computing that prefix. A read is believed only when the ring still
//! holds its extent after it landed.
//!
//! The disk is driven through io_uring with `O_DIRECT`, one ring per
//! thread: a job's objects are cut into chunks and all of them are in
//! flight at once up to the queue depth, so every member of a RAID works.

#![allow(unsafe_code)]

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::io::AsRawFd;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use io_uring::{opcode, types, IoUring};
use kern_pool::{Arena, Extent, Object, Placed, Ring, Room};
use kern_runtime::{Mark, Pinned, Runtime, Span};
use tracing::{debug, info};

use super::{buffers, Buffers};
use crate::ranks::Landed;

/// `O_DIRECT`'s unit on 4Kn and 512e disks alike; the block and the ring
/// place everything on it.
const ALIGN: u64 = 4096;
/// The most bytes one read or write moves.
const CHUNK: u64 = 2 << 20;
/// Chunks in flight per thread.
const DEPTH: u32 = 64;

#[derive(Debug, Clone)]
pub struct Config {
    /// The directory the file goes in, `kern-pool-gpu<gpu>.ring`.
    pub dir: PathBuf,
    pub gpu: usize,
    /// Bytes of the file, all reserved at open.
    pub capacity: u64,
    /// Bytes of the pinned block every transfer passes through.
    pub staging: u64,
}

/// What the disk threads and the issuing thread share.
struct Shared {
    file: File,
    ring: Mutex<Ring>,
    arena: Mutex<Arena>,
    /// The pinned block's address; the block itself stays with the client.
    base: u64,
}

impl Shared {
    fn give(&self, rooms: impl IntoIterator<Item = Room>) {
        let mut a = self.arena.lock().expect("not poisoned");
        rooms.into_iter().for_each(|r| a.give(r));
    }
}

/// One object a job moves: object `at` of its act, its extent in the
/// file, its room in the block, its bytes on the device.
struct Move {
    at: usize,
    key: String,
    extent: Extent,
    room: Room,
    device: Buffers,
}

/// A transfer between the file and the block: the act, its flags so far,
/// and the objects that still move. A put also waits for its mark.
struct Job {
    id: u64,
    ok: Vec<bool>,
    moves: Vec<Move>,
    after: Option<Mark>,
}

/// This rank's disk tier: the file, the block, a write and a read thread.
pub struct Client {
    shared: Arc<Shared>,
    spans: BTreeMap<String, Span>,
    writes: Option<Sender<Job>>,
    reads: Option<Sender<Job>>,
    /// Gets read into the block, for [`Client::issue`] to copy in.
    read: Receiver<Job>,
    done_tx: Sender<Landed>,
    done: Receiver<Landed>,
    /// Rooms of gets copied in, until the stream passes the copy.
    copying: Mutex<Vec<(Mark, Vec<Room>)>>,
    threads: Vec<JoinHandle<()>>,
    /// Dropped last: both threads have stopped and every copy has run.
    _block: Pinned,
}

impl Client {
    /// Create the file, reserve all of it, pin the block, start the threads.
    pub fn open(cfg: &Config, rt: &Runtime) -> Result<Client> {
        let spans = rt.pooled_spans();
        if spans.is_empty() {
            bail!("pool: the manifest has no pooled state to keep");
        }
        std::fs::create_dir_all(&cfg.dir).with_context(|| format!("creating {}", cfg.dir.display()))?;
        let path = cfg.dir.join(format!("kern-pool-gpu{}.ring", cfg.gpu));
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .custom_flags(libc::O_DIRECT)
            .open(&path)
            .with_context(|| format!("opening {} with O_DIRECT", path.display()))?;
        let ring = Ring::new(cfg.capacity, ALIGN);
        // Reserved whole, so a disk short of room fails here rather than
        // on every write.
        if unsafe { libc::fallocate(file.as_raw_fd(), 0, 0, ring.capacity() as libc::off_t) } != 0 {
            bail!("reserving {} GiB in {}: {}", ring.capacity() >> 30, path.display(), std::io::Error::last_os_error());
        }
        let t0 = Instant::now();
        let block = rt.pinned(cfg.staging).context("pinning the disk tier's staging block")?;
        let arena = Arena::new(block.bytes(), ALIGN);
        info!(
            path = %path.display(),
            capacity_gib = ring.capacity() >> 30,
            staging_gib = arena.size() >> 30,
            pin_s = t0.elapsed().as_secs_f64(),
            states = spans.len(),
            "pool: disk tier open"
        );
        let shared = Arc::new(Shared { file, ring: Mutex::new(ring), arena: Mutex::new(arena), base: block.ptr() });
        let (done_tx, done) = channel();
        let (read_tx, read) = channel();
        let (writes, write_rx) = channel::<Job>();
        let (reads, read_rx) = channel::<Job>();
        let spawn = |name: &str, f: Box<dyn FnOnce() + Send>| {
            std::thread::Builder::new().name(format!("kern-pool-{name}")).spawn(f).context("spawning a disk thread")
        };
        let threads = vec![
            spawn("write", {
                let (sh, done, mut uring) = (Arc::clone(&shared), done_tx.clone(), uring()?);
                Box::new(move || write_loop(&sh, &mut uring, write_rx, &done))
            })?,
            spawn("read", {
                let (sh, mut uring) = (Arc::clone(&shared), uring()?);
                Box::new(move || read_loop(&sh, &mut uring, read_rx, &read_tx))
            })?,
        ];
        let spans = spans.into_iter().map(|s| (s.state.clone(), s)).collect();
        Ok(Client {
            shared,
            spans,
            writes: Some(writes),
            reads: Some(reads),
            read,
            done_tx,
            done,
            copying: Mutex::new(Vec::new()),
            threads,
            _block: block,
        })
    }

    fn answer(&self, id: u64, ok: Vec<bool>) {
        let _ = self.done_tx.send(Landed { id, ok });
    }

    pub fn probe(&self, id: u64, keys: &[String]) {
        let ring = self.shared.ring.lock().expect("not poisoned");
        let ok = keys.iter().map(|k| ring.find(k).is_some()).collect();
        drop(ring);
        self.answer(id, ok);
    }

    pub fn mark(&self, id: u64, key: &str) {
        let mut ring = self.shared.ring.lock().expect("not poisoned");
        let ok = match ring.place(key, 0) {
            Placed::New(e) => ring.commit(key, e, true),
            Placed::Held => true,
            Placed::Busy | Placed::TooLarge => false,
        };
        drop(ring);
        self.answer(id, vec![ok]);
    }

    /// Place every object not on disk yet, copy it into a room behind what
    /// `rt` has issued, and hand the writes to the write thread.
    pub fn put(&self, rt: &Runtime, id: u64, objects: &[Object]) -> Result<()> {
        let mut ok = vec![true; objects.len()];
        let mut moves = Vec::new();
        {
            let mut ring = self.shared.ring.lock().expect("not poisoned");
            let mut arena = self.shared.arena.lock().expect("not poisoned");
            for (at, o) in objects.iter().enumerate() {
                let device = buffers(&self.spans, o);
                if device.is_empty() {
                    continue;
                }
                match ring.place(&o.key, bytes(&device)) {
                    Placed::Held => {}
                    Placed::New(extent) => match arena.take(extent.len()) {
                        Some(room) => moves.push(Move { at, key: o.key.clone(), extent, room, device }),
                        None => {
                            ring.commit(&o.key, extent, false);
                            ok[at] = false;
                        }
                    },
                    Placed::Busy | Placed::TooLarge => ok[at] = false,
                }
            }
        }
        if moves.is_empty() {
            self.answer(id, ok);
            return Ok(());
        }
        rt.copy_bytes(&runs(self.shared.base, &moves, Way::Out))?;
        let after = Some(rt.mark()?);
        let tx = self.writes.as_ref().expect("open until dropped");
        tx.send(Job { id, ok, moves, after }).ok().context("the disk tier's write thread has stopped")
    }

    /// Hand every object the ring holds at its size to the read thread,
    /// each with a room to land in; [`Client::issue`] copies them in.
    pub fn get(&self, id: u64, objects: &[Object]) -> Result<()> {
        let mut ok = vec![false; objects.len()];
        let mut moves = Vec::new();
        {
            let ring = self.shared.ring.lock().expect("not poisoned");
            let mut arena = self.shared.arena.lock().expect("not poisoned");
            for (at, o) in objects.iter().enumerate() {
                let device = buffers(&self.spans, o);
                let want = bytes(&device).div_ceil(ALIGN) * ALIGN;
                match ring.find(&o.key) {
                    _ if want == 0 => ok[at] = true,
                    Some(extent) if extent.len() == want => {
                        if let Some(room) = arena.take(want) {
                            moves.push(Move { at, key: o.key.clone(), extent, room, device });
                        }
                    }
                    _ => {}
                }
            }
        }
        if moves.is_empty() {
            self.answer(id, ok);
            return Ok(());
        }
        let tx = self.reads.as_ref().expect("open until dropped");
        tx.send(Job { id, ok, moves, after: None }).ok().context("the disk tier's read thread has stopped")
    }

    /// Copy every get the read thread has landed into its lease, answer
    /// it, and take back the rooms of copies the stream has passed.
    pub fn issue(&self, rt: &Runtime) -> Result<()> {
        let mut copying = self.copying.lock().expect("not poisoned");
        let (passed, running): (Vec<_>, Vec<_>) =
            std::mem::take(&mut *copying).into_iter().partition(|(m, _)| m.passed());
        *copying = running;
        self.shared.give(passed.into_iter().flat_map(|(_, rooms)| rooms));
        for job in self.read.try_iter() {
            let (landed, failed): (Vec<Move>, Vec<Move>) = job.moves.into_iter().partition(|m| job.ok[m.at]);
            self.shared.give(failed.into_iter().map(|m| m.room));
            if !landed.is_empty() {
                rt.copy_bytes(&runs(self.shared.base, &landed, Way::In))?;
                copying.push((rt.mark()?, landed.into_iter().map(|m| m.room).collect()));
            }
            self.answer(job.id, job.ok);
        }
        Ok(())
    }

    pub fn landed(&self) -> Vec<Landed> {
        self.done.try_iter().collect()
    }
}

impl Drop for Client {
    /// The block outlives everything that moves through it: the threads
    /// finish their jobs, and every copy in is waited for.
    fn drop(&mut self) {
        self.writes.take();
        self.reads.take();
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
        let copying = std::mem::take(self.copying.get_mut().unwrap_or_else(|e| e.into_inner()));
        for (m, _) in copying {
            let _ = m.wait();
        }
    }
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
fn runs(base: u64, moves: &[Move], way: Way) -> Vec<(u64, u64, u64)> {
    moves
        .iter()
        .flat_map(|m| {
            let room = base + m.room.offset();
            m.device.iter().scan(0u64, move |at, &(dev, n)| {
                let host = room + *at;
                *at += n as u64;
                Some(match way {
                    Way::Out => (host, dev, n as u64),
                    Way::In => (dev, host, n as u64),
                })
            })
        })
        .collect()
}

fn uring() -> Result<IoUring> {
    IoUring::new(DEPTH).context("creating an io_uring for the disk tier")
}

/// Wait for each put's mark, write its rooms, commit what landed.
fn write_loop(sh: &Shared, uring: &mut IoUring, jobs: Receiver<Job>, done: &Sender<Landed>) {
    for mut job in jobs {
        let t0 = Instant::now();
        let ready = job.after.take().is_none_or(|m| m.wait().is_ok());
        let wrote = match ready {
            true => transfer(uring, sh, &job.moves, Way::Out),
            false => vec![false; job.moves.len()],
        };
        let mut ring = sh.ring.lock().expect("not poisoned");
        for (m, &w) in job.moves.iter().zip(&wrote) {
            job.ok[m.at] &= ring.commit(&m.key, m.extent, w);
        }
        drop(ring);
        log("write", &job, t0);
        sh.give(job.moves.into_iter().map(|m| m.room));
        let _ = done.send(Landed { id: job.id, ok: job.ok });
    }
}

/// Read each get's extents into its rooms and hand it back to be copied
/// in, an extent believed only when the ring still holds it.
fn read_loop(sh: &Shared, uring: &mut IoUring, jobs: Receiver<Job>, read: &Sender<Job>) {
    for mut job in jobs {
        let t0 = Instant::now();
        let got = transfer(uring, sh, &job.moves, Way::In);
        let ring = sh.ring.lock().expect("not poisoned");
        for (m, &g) in job.moves.iter().zip(&got) {
            job.ok[m.at] = g && ring.holds(m.extent);
        }
        drop(ring);
        log("read", &job, t0);
        if let Err(e) = read.send(job) {
            sh.give(e.0.moves.into_iter().map(|m| m.room));
        }
    }
}

fn log(what: &str, job: &Job, t0: Instant) {
    let moved: u64 = job.moves.iter().filter(|m| job.ok[m.at]).map(|m| m.extent.len()).sum();
    let s = t0.elapsed().as_secs_f64();
    debug!(
        what,
        id = job.id,
        objects = job.moves.len(),
        landed = job.moves.iter().filter(|m| job.ok[m.at]).count(),
        mib = moved >> 20,
        ms = (s * 1e3) as u64,
        gib_s = moved as f64 / (1u64 << 30) as f64 / s.max(1e-9),
        "pool: disk"
    );
}

/// Move every move's extent between the file and its room, `CHUNK` bytes
/// per request and `DEPTH` requests in flight: whether each move landed
/// whole. Returns only once nothing is in flight, so a room is never
/// given back while the disk still moves it.
fn transfer(uring: &mut IoUring, sh: &Shared, moves: &[Move], way: Way) -> Vec<bool> {
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
            // SAFETY: the room is this job's until it answers, and the
            // block outlives both threads (`Client::drop` joins them).
            if unsafe { uring.submission().push(&sqe) }.is_err() {
                break;
            }
            next += 1;
            inflight += 1;
        }
        match uring.submit_and_wait(1) {
            Ok(_) => {}
            Err(e) if matches!(e.raw_os_error(), Some(libc::EINTR | libc::EAGAIN | libc::EBUSY)) => continue,
            // Requests may still be in flight into the rooms: nothing
            // can be given back, so the thread stops and the next act
            // on the tier fails the rank.
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
