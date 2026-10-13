//! Pipeline serving: one process per stage of a cut manifest (`kern cut`),
//! prefill only — a request gets its first token and finishes.
//!
//! Stage 0 is the head: the HTTP front end, the scheduler and the one
//! [`kern_pool::Pool`] that decides every lease. The other stages are
//! executors that hold no accounting of their own. Each stage's KV pages
//! hold that stage's layers under the head's page ids, which is why the
//! head's pool is sized to the smallest stage: the stages say how many
//! pages they have in their [`Hello`], the head loads with the minimum.
//!
//! Two kinds of connection, each a TCP stream of frames (a `u32` length,
//! then JSON). Every stage connects to the head and sends its [`Hello`]:
//! its place, the cut it belongs to, its pages, the port its predecessor
//! may reach it on and the handles of its arriving buffers. Once every
//! stage has said hello the head loads and answers each with a [`Table`]:
//! the pages, and where the next stage listens with what arrives there.
//! From then on the head sends every stage every [`Item`] in issue order
//! — the program, its vars, the values of each input it staged and what
//! its pool decided since the item before (copies, parks and wakes into
//! each stage's own host block, sized by the head in host pages and
//! host slots) — and
//! the last stage answers every item with its tokens (none for a program
//! that hands none back). TCP keeps the order, so an item's number is its
//! place in it and every stage counts the same items alike; there is no
//! sequence number, no stop message and no fault message. A stage that
//! fails exits; a closed connection ends the process on the other side.
//! With a decode tier (`pd`) the head's table also carries its pools, the
//! head forwards every placement a tier moves to, and it sends every stage
//! a [`Push`] after the item that writes a prompt's last rows; each stage
//! answers it `Pushed` once it has landed.
//!
//! The second kind joins each stage to the next, and carries item
//! numbers both ways. Every arriving buffer of a stage (`kern_run::arrivals`:
//! the inputs nobody on the host stages) has [`LANDINGS`] exported twins
//! ([`Runtime::landing`]), and the stage before maps them at start
//! ([`Next`]). Having issued item `k`, that stage waits until the next
//! one has taken the item last in landing `k % LANDINGS`, copies each
//! buffer into it on its compute stream ([`Runtime::push`]) and marks;
//! when the mark lands its [`Watch`] thread reports `k` arrived to the
//! next stage. The next stage's host does not wait for that: it issues
//! `k` behind a hold on its count of arrived items ([`Runtime::hold`]),
//! moves the landing into the buffer it computes in ([`Runtime::land`])
//! and marks, the launches are in before the data, and the report opens
//! the gate ([`Gate::open`]); once that mark lands the item is taken and
//! its landing free, which the next stage reports to the one before. A
//! stage is held back only when the next is `LANDINGS` items behind,
//! never because it still reads the buffer the item goes to. The host is
//! in the loop once per edge and item, off the device's path. The last
//! stage reads its tokens, and every stage logs when each item left it. A
//! next stage that takes nothing for [`TIMEOUT`] is taken for dead; a
//! dead stage before shows as the head going.

mod cache;
pub mod head;
pub mod pack;
pub mod stage;

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

use anyhow::{bail, ensure, Context, Result};
use kern_manifest::protocol::Filled;
use kern_manifest::Protocol;
use kern_runtime::{Fetch, Gate, Mapped, Mark, PeerHandle, Runtime};
use serde::{Deserialize, Serialize};

use crate::logline::now_us;
use crate::pd::{Pools, Push, Req, Shape};
use crate::ranks::Act;

/// How long a stage waits for its neighbour to deliver or take an item.
const TIMEOUT: Duration = Duration::from_secs(60);

/// Landings of every arriving buffer: item `k` lands in `k % LANDINGS`, so
/// a stage pushes on as long as the next one has started all but
/// `LANDINGS - 1` of the items before. Pushing straight into the buffer
/// the next stage computes in held every stage behind the slowest item
/// downstream; on a 4-stage Qwen3-4B (2026-10-12) one landing took c32
/// from 90.5k to 106.4k prompt tokens a second, two to 108.0k, four 108.1k.
const LANDINGS: u64 = 2;

/// A stage introducing itself to the head.
#[derive(Debug, Serialize, Deserialize)]
pub struct Hello {
    pub stage: u64,
    pub cut: String,
    /// Its pages, or none when it holds no paged state and bounds nothing.
    pub pages: Option<u64>,
    /// The port the stage before it connects to.
    pub port: u16,
    /// The landings of its arriving buffers, by name, as hex fabric handles.
    pub arrivals: Landings,
    /// Its pooled states, which a decode tier's must match.
    pub states: BTreeMap<String, Shape>,
    /// Bytes of its host page and its host slot: every paged state's page
    /// back to back, every per-sequence state's slot likewise.
    pub host: (u64, u64),
}

/// The head's answer: the pages every stage addresses, where the next
/// stage listens with what arrives there (none for the last stage), the
/// decode tiers to push into, in the head's `--decode` order, and the host
/// pages and host slots of the host tier every stage reserves, (0, 0) for
/// none.
#[derive(Debug, Serialize, Deserialize)]
pub struct Table {
    pub pages: u64,
    pub next: Option<(String, Landings)>,
    pub decode: Vec<Pools>,
    pub host: (u64, u64),
}

/// What the head sends a stage, in issue order.
#[derive(Debug, Serialize, Deserialize)]
pub enum Job {
    Item(Item),
    Push(Push),
    /// Decode tier `tier`'s rank `rank` moved to placement `epoch` (hex).
    Place {
        tier: usize,
        rank: usize,
        epoch: u64,
        placement: String,
    },
}

/// What a stage tells the head: the last stage every item's tokens, every
/// stage each push that landed.
#[derive(Debug, Serialize, Deserialize)]
pub enum Report {
    Tokens(Vec<i64>),
    Pushed(Req),
}

/// Where a stage's watch thread reports.
type Sink = Box<dyn FnMut(Report) -> Result<()> + Send>;

/// One call, as the head staged it.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Item {
    pub program: String,
    pub vars: BTreeMap<String, u64>,
    /// Each input's values by name: the fills, the page tables and the
    /// line tables when the item's sequences change them.
    pub rows: BTreeMap<String, Vec<i64>>,
    /// What the head's pool decided since the item before, in the order
    /// it ran on the head's stream, for a stage to run before anything
    /// else of the item: a fresh sequence's slot zeroed, the copies of a
    /// prompt continuing from a kept prefix or of a prefix kept, an entry
    /// parked to the host tier, a parked one woken into a lease. Never a
    /// remap: the head's pool makes none.
    pub acts: Vec<Act>,
}

pub type Handles = BTreeMap<String, String>;

/// Every arriving buffer's landings, in slot order.
pub type Landings = BTreeMap<String, Vec<String>>;

/// The landings of this runtime's arriving buffers as wire handles: the
/// stage before pushes into a landing, never into a buffer this stage may
/// still be reading. A stage maps its neighbour across processes, so a
/// local handle is an error.
fn arrivals(rt: &Runtime, p: &Protocol) -> Result<Landings> {
    kern_run::arrivals(&rt.manifest, p)
        .into_iter()
        .map(|name| {
            let slots = (0..LANDINGS as usize)
                .map(|i| {
                    let h = rt.landing(&name, i).with_context(|| format!("landing {i} of `{name}`"))?;
                    let b = h.to_bytes().with_context(|| format!("landing {i} of `{name}` has no fabric handle"))?;
                    Ok(hex::encode(b))
                })
                .collect::<Result<_>>()?;
            Ok((name, slots))
        })
        .collect()
}

pub(crate) fn send<T: Serialize>(w: &mut TcpStream, msg: &T) -> Result<()> {
    let body = serde_json::to_vec(msg)?;
    w.write_all(&(body.len() as u32).to_le_bytes()).and_then(|()| w.write_all(&body)).context("sending")
}

/// The next message; `None` when the other side closed.
pub(crate) fn recv<T: for<'de> Deserialize<'de>>(r: &mut TcpStream) -> Result<Option<T>> {
    let mut len = [0u8; 4];
    match r.read_exact(&mut len) {
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        done => done.context("receiving")?,
    }
    let mut body = vec![0u8; u32::from_le_bytes(len) as usize];
    r.read_exact(&mut body).context("receiving")?;
    Ok(Some(serde_json::from_slice(&body)?))
}

/// Reach `addr`, retrying while the other side is still coming up.
pub(crate) fn connect(addr: &str, patience: Duration) -> Result<TcpStream> {
    let t0 = Instant::now();
    loop {
        match TcpStream::connect(addr) {
            Ok(s) => {
                let _ = s.set_nodelay(true);
                return Ok(s);
            }
            Err(e) if t0.elapsed() < patience => {
                tracing::debug!("{addr}: {e}");
                std::thread::sleep(Duration::from_millis(500));
            }
            Err(e) => return Err(e).with_context(|| format!("connecting to {addr}")),
        }
    }
}

/// Item numbers between neighbours: one `u64` a frame.
fn tell(w: &mut TcpStream, item: u64) -> Result<()> {
    w.write_all(&item.to_le_bytes()).context("telling the neighbour")
}

/// Read item numbers off `r` until the neighbour goes away.
fn hear(mut r: TcpStream, mut heard: impl FnMut(u64) -> bool) {
    let mut item = [0u8; 8];
    while r.read_exact(&mut item).is_ok() && heard(u64::from_le_bytes(item)) {}
}

/// Wait for `rx` to say `k`, the next item in order.
fn wait(rx: &mpsc::Receiver<u64>, k: u64, what: &str) -> Result<()> {
    match rx.recv_timeout(TIMEOUT) {
        Ok(n) if n == k => Ok(()),
        Ok(n) => bail!("{what} item {n}, expected {k}"),
        Err(mpsc::RecvTimeoutError::Timeout) => bail!("{what} nothing in {TIMEOUT:?} on item {k}; the stage is gone"),
        Err(mpsc::RecvTimeoutError::Disconnected) => bail!("{what}: the connection is gone"),
    }
}

/// The edge out of a stage: the landings of the next stage's arriving
/// buffers mapped (each one this stage holds under the same name), and
/// the items it reports taken.
struct Next {
    landings: Vec<(String, Vec<Mapped>)>,
    taken: mpsc::Receiver<u64>,
}

impl Next {
    /// Map the next stage's arrivals and connect to it; `wake` hears of
    /// every item it reports taken. Returns the edge and the write half the
    /// watch reports arrivals on.
    fn connect(
        rt: &Runtime,
        addr: &str,
        arrivals: &Landings,
        wake: impl Fn() + Send + 'static,
    ) -> Result<(Next, TcpStream)> {
        let mut landings = Vec::new();
        for (name, x) in arrivals {
            if !rt.manifest.buffers.contains_key(name) {
                bail!("the next stage takes `{name}` from this one, which has no such buffer");
            }
            let slots = x
                .iter()
                .enumerate()
                .map(|(i, x)| {
                    let b = hex::decode(x).with_context(|| format!("handle of landing {i} of `{name}`"))?;
                    let h = PeerHandle::from_bytes(&b).with_context(|| format!("handle of landing {i} of `{name}`"))?;
                    Ok(rt.map(&h, &format!("the next stage's landing {i} of `{name}`"))?)
                })
                .collect::<Result<Vec<_>>>()?;
            ensure!(slots.len() as u64 == LANDINGS, "`{name}` has {} landings, not {LANDINGS}", slots.len());
            landings.push((name.clone(), slots));
        }
        let s = connect(addr, TIMEOUT)?;
        let (tx, taken) = mpsc::channel();
        std::thread::spawn({
            let r = s.try_clone()?;
            move || {
                hear(r, |k| {
                    let sent = tx.send(k).is_ok();
                    wake();
                    sent
                })
            }
        });
        Ok((Next { landings, taken }, s))
    }
}

/// The edge into a stage: how many items the stage before reports
/// arrived, as a gate the device waits on, and the buffers they land in.
struct Prev {
    arrived: Arc<Gate>,
    landings: Vec<String>,
}

impl Prev {
    /// Accept the stage before. Returns the edge and the write half the
    /// watch reports taken items on.
    fn accept(rt: &Runtime, listener: &TcpListener, landings: Vec<String>) -> Result<(Prev, TcpStream)> {
        let (s, _) = listener.accept().context("accepting the stage before")?;
        let _ = s.set_nodelay(true);
        let arrived = Arc::new(rt.gate()?);
        std::thread::spawn({
            let r = s.try_clone()?;
            let gate = arrived.clone();
            move || {
                hear(r, |k| {
                    gate.open(k + 1);
                    true
                })
            }
        });
        Ok((Prev { arrived, landings }, s))
    }
}

/// What the last stage answers an item with: its token output on the way
/// to the host, decoded and cut to the item's groups once the item is done.
pub(super) struct Answer {
    pub fetch: Fetch,
    pub fill: Filled,
    pub keep: usize,
}

/// A stage's place in the flow of items: what it waits for before
/// issuing one and what it does after.
pub(super) struct Flow {
    prev: Option<Prev>,
    next: Option<Next>,
    watch: Watch,
    item: u64,
    /// Items the next stage reported taken, in order.
    taken: u64,
}

impl Flow {
    /// `report` is where the watch tells the head what landed; `answers`
    /// that this is the last stage, which answers every item.
    fn new(
        stage: u64,
        prev: Option<(Prev, TcpStream)>,
        next: Option<(Next, TcpStream)>,
        report: Sink,
        answers: bool,
    ) -> Flow {
        let (prev, prev_w) = prev.map_or((None, None), |(p, w)| (Some(p), Some(w)));
        let (next, next_w) = next.map_or((None, None), |(n, w)| (Some(n), Some(w)));
        Flow { prev, next, watch: Watch::new(stage, prev_w, next_w, report, answers), item: 0, taken: 0 }
    }

    /// A push for `req` is issued: mark it, and report it once it lands.
    fn pushed(&self, rt: &Runtime, req: Req) -> Result<()> {
        rt.fault()?;
        self.watch.send(Landing::Pushed { req, mark: rt.mark()? })
    }

    /// Hold the compute stream until the next item has arrived from the
    /// stage before, move it out of the landings at `vars`, and mark where
    /// the item starts: from there the stage before may push the item
    /// after. The host goes on issuing.
    fn begin(&self, rt: &Runtime, vars: &BTreeMap<String, u64>) -> Result<()> {
        if let Some(p) = &self.prev {
            rt.hold(&p.arrived, self.item + 1)?;
            for name in &p.landings {
                rt.land(name, (self.item % LANDINGS) as usize, vars).with_context(|| format!("landing `{name}`"))?;
            }
        }
        self.watch.send(Landing::Start { item: self.item, mark: rt.mark()? })
    }

    /// Whether a landing is free on the next stage for the next item, so
    /// that [`Flow::issued`] pushes it on without waiting.
    fn ready(&mut self) -> Result<bool> {
        let Some(n) = &self.next else { return Ok(true) };
        loop {
            match n.taken.try_recv() {
                Ok(t) if t == self.taken => self.taken += 1,
                Ok(t) => bail!("the next stage reported item {t}, expected {}", self.taken),
                Err(mpsc::TryRecvError::Empty) => return Ok(self.taken + LANDINGS > self.item),
                Err(mpsc::TryRecvError::Disconnected) => bail!("the next stage's connection is gone"),
            }
        }
    }

    /// The next item is issued, of `rows` rows at `vars`: once the next
    /// stage has taken the item last in its landing, push it on, and mark
    /// it.
    fn issued(&mut self, rt: &Runtime, vars: &BTreeMap<String, u64>, rows: u64, answer: Option<Answer>) -> Result<()> {
        let k = self.item;
        if let Some(n) = &self.next {
            while self.taken + LANDINGS <= k {
                wait(&n.taken, self.taken, "the next stage took")?;
                self.taken += 1;
            }
            for (name, slots) in &n.landings {
                rt.push(name, vars, &slots[(k % LANDINGS) as usize])
                    .with_context(|| format!("pushing `{name}` to the next stage"))?;
            }
        }
        rt.fault()?;
        self.watch.send(Landing::Item(Issued { item: k, rows, issued_us: now_us(), mark: rt.mark()?, answer }))?;
        self.item += 1;
        Ok(())
    }
}

/// When each item a stage issued ran on its device: one `pp item` line per
/// item (its number, its rows, and of the wall clock `issued_us`, when the
/// issuing thread was through with it, `start_us`, when the device reached
/// it past the hold for its arrival, and `done_us`, when it left), logged
/// by a thread that waits on the item's [`Mark`]s so the issuing thread
/// never does, and tells the neighbours: the next stage that the item
/// arrived, the previous that it was taken. On the last stage it also
/// answers the head with the item's tokens; on every stage it reports each
/// push to a decode tier once its mark lands. A stage's period is done to
/// done; `done - start` is the stage's own time on the item, the rest of
/// the period its pipeline bubble (with the acts and row writes before the
/// hold). A fault on the device surfaces at its item's mark and ends the
/// process.
struct Watch {
    issued: Option<mpsc::Sender<Landing>>,
}

enum Landing {
    /// Where item `item` starts, its arrivals out of the landings.
    Start {
        item: u64,
        mark: Mark,
    },
    Item(Issued),
    Pushed {
        req: Req,
        mark: Mark,
    },
}

struct Issued {
    item: u64,
    rows: u64,
    issued_us: u64,
    mark: Mark,
    answer: Option<Answer>,
}

impl Watch {
    fn new(
        stage: u64,
        mut prev: Option<TcpStream>,
        mut next: Option<TcpStream>,
        mut report: Sink,
        answers: bool,
    ) -> Watch {
        let (issued, rx) = mpsc::channel::<Landing>();
        std::thread::spawn(move || {
            let mut start_us = 0;
            for l in rx {
                let told = match l {
                    Landing::Start { item, mark } => {
                        if let Err(e) = mark.wait() {
                            tracing::error!(stage, "device: {e:#}");
                            std::process::exit(1);
                        }
                        start_us = now_us();
                        prev.as_mut().map_or(Ok(()), |w| tell(w, item)).with_context(|| format!("item {item} taken"))
                    }
                    Landing::Item(Issued { item, rows, issued_us, mark, answer }) => {
                        if let Err(e) = mark.wait() {
                            tracing::error!(stage, item, "device: {e:#}");
                            std::process::exit(1);
                        }
                        tracing::info!(stage, item, rows, issued_us, start_us, done_us = now_us(), "pp item");
                        let tokens = answer.map_or(Vec::new(), |a| {
                            let mut v = a.fill.decode(a.fetch.bytes());
                            v.truncate(a.keep);
                            v
                        });
                        next.as_mut()
                            .map_or(Ok(()), |w| tell(w, item))
                            .and(if answers { report(Report::Tokens(tokens)) } else { Ok(()) })
                            .with_context(|| format!("item {item}"))
                    }
                    Landing::Pushed { req, mark } => {
                        if let Err(e) = mark.wait() {
                            tracing::error!(stage, req, "device: {e:#}");
                            std::process::exit(1);
                        }
                        tracing::debug!(stage, req, done_us = now_us(), "pd pushed");
                        report(Report::Pushed(req)).with_context(|| format!("push of request {req}"))
                    }
                };
                if let Err(e) = told {
                    tracing::error!(stage, "{e:#}");
                    std::process::exit(1);
                }
            }
        });
        Watch { issued: Some(issued) }
    }

    fn send(&self, l: Landing) -> Result<()> {
        let issued = self.issued.as_ref().expect("taken only on drop");
        if issued.send(l).is_err() {
            bail!("the watch thread is gone");
        }
        Ok(())
    }
}

/// The watch goes with the process: an item still in flight when the head
/// is gone may be holding for data that will never come.
impl Drop for Watch {
    fn drop(&mut self) {
        drop(self.issued.take());
    }
}
