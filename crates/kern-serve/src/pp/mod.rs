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
//! — the program, its vars and the values of each input it staged — and
//! the last stage answers every item with its tokens (none for a program
//! that hands none back). TCP keeps the order, so an item's number is its
//! place in it and every stage counts the same items alike; there is no
//! sequence number, no stop message and no fault message. A stage that
//! fails exits; a closed connection ends the process on the other side.
//!
//! The second kind joins each stage to the next, and carries item
//! numbers both ways. A stage's arriving buffers (`kern_run::arrivals`:
//! the inputs nobody on the host stages) are exported, and the stage
//! before maps them at start ([`Next`]). Having issued item `k`, that
//! stage waits until the next one reports item `k - 1` done, copies each
//! arriving buffer over on its compute stream ([`Runtime::push`]) and
//! marks; when the mark lands its [`Watch`] thread reports `k` arrived to
//! the next stage and `k` done to the previous one ([`Flow`]). The next
//! stage's host does not wait for that: it issues `k` behind a hold on
//! its count of arrived items ([`Runtime::hold`]), the launches are in
//! before the data, and the report opens the gate ([`Gate::open`]). So
//! each crossing buffer has one copy per stage and the host is in the
//! loop once per edge and item, off the device's path. The last stage
//! reads its tokens, and every stage logs when each item left it. A next
//! stage that reports nothing done for [`TIMEOUT`] is taken for dead; a
//! dead stage before shows as the head going.

pub mod head;
mod pack;
pub mod stage;

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use kern_manifest::protocol::Filled;
use kern_manifest::Protocol;
use kern_runtime::{Fetch, Gate, Mapped, Mark, PeerHandle, Runtime};
use serde::{Deserialize, Serialize};

/// How long a stage waits for its neighbour to deliver or take an item.
const TIMEOUT: Duration = Duration::from_secs(60);

/// A stage introducing itself to the head.
#[derive(Debug, Serialize, Deserialize)]
pub struct Hello {
    pub stage: u64,
    pub cut: String,
    pub pages: u64,
    /// The port the stage before it connects to.
    pub port: u16,
    /// Its arriving buffers, by name, as hex fabric handles.
    pub arrivals: Handles,
}

/// The head's answer: the pages every stage addresses, and where the next
/// stage listens with what arrives there (none for the last stage).
#[derive(Debug, Serialize, Deserialize)]
pub struct Table {
    pub pages: u64,
    pub next: Option<(String, Handles)>,
}

/// One call, as the head staged it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Item {
    pub program: String,
    pub vars: BTreeMap<String, u64>,
    /// Each input's values by name: the fills, the page tables and the
    /// line tables when the item's sequences change them.
    pub rows: BTreeMap<String, Vec<i64>>,
    /// The sequence slots to zero before the call, the ones of sequences
    /// on their first item: the recurrent state a fresh lease starts from.
    pub zero_slots: Vec<i32>,
}

pub type Handles = BTreeMap<String, String>;

/// This runtime's arriving buffers as wire handles; a stage maps its
/// neighbour across processes, so a local handle is an error.
fn arrivals(rt: &Runtime, p: &Protocol) -> Result<Handles> {
    let all = rt.export_handles()?;
    kern_run::arrivals(&rt.manifest, p)
        .into_iter()
        .map(|name| {
            let h = all
                .get(&name)
                .with_context(|| format!("`{name}` arrives from the stage before but is not exported"))?;
            let b = h.to_bytes().with_context(|| format!("`{name}` has no fabric handle"))?;
            Ok((name, hex::encode(b)))
        })
        .collect()
}

fn send<T: Serialize>(w: &mut TcpStream, msg: &T) -> Result<()> {
    let body = serde_json::to_vec(msg)?;
    w.write_all(&(body.len() as u32).to_le_bytes()).and_then(|()| w.write_all(&body)).context("sending")
}

/// The next message; `None` when the other side closed.
fn recv<T: for<'de> Deserialize<'de>>(r: &mut TcpStream) -> Result<Option<T>> {
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
fn connect(addr: &str, patience: Duration) -> Result<TcpStream> {
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

/// The edge out of a stage: the next stage's arriving buffers mapped
/// (each one this stage holds under the same name), and the items it
/// reports done.
struct Next {
    boxes: Vec<(String, Mapped)>,
    done: mpsc::Receiver<u64>,
}

impl Next {
    /// Map the next stage's arrivals and connect to it. Returns the edge
    /// and the write half the watch reports arrivals on.
    fn connect(rt: &Runtime, addr: &str, arrivals: &Handles) -> Result<(Next, TcpStream)> {
        let mut boxes = Vec::new();
        for (name, x) in arrivals {
            if !rt.manifest.buffers.contains_key(name) {
                bail!("the next stage takes `{name}` from this one, which has no such buffer");
            }
            let b = hex::decode(x).with_context(|| format!("handle of `{name}`"))?;
            let h = PeerHandle::from_bytes(&b).with_context(|| format!("handle of `{name}`"))?;
            boxes.push((name.clone(), rt.map(&h, &format!("the next stage's `{name}`"))?));
        }
        let s = connect(addr, TIMEOUT)?;
        let (tx, done) = mpsc::channel();
        std::thread::spawn({
            let r = s.try_clone()?;
            move || hear(r, |k| tx.send(k).is_ok())
        });
        Ok((Next { boxes, done }, s))
    }
}

/// The edge into a stage: how many items the stage before reports
/// arrived, as a gate the device waits on.
struct Prev {
    arrived: Arc<Gate>,
}

impl Prev {
    /// Accept the stage before. Returns the edge and the write half the
    /// watch reports taken items on.
    fn accept(rt: &Runtime, listener: &TcpListener) -> Result<(Prev, TcpStream)> {
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
        Ok((Prev { arrived }, s))
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
}

impl Flow {
    /// `head` is the connection the last stage answers items on.
    fn new(
        stage: u64,
        prev: Option<(Prev, TcpStream)>,
        next: Option<(Next, TcpStream)>,
        head: Option<TcpStream>,
    ) -> Flow {
        let (prev, prev_w) = prev.map_or((None, None), |(p, w)| (Some(p), Some(w)));
        let (next, next_w) = next.map_or((None, None), |(n, w)| (Some(n), Some(w)));
        Flow { prev, next, watch: Watch::new(stage, prev_w, next_w, head), item: 0 }
    }

    /// Hold the compute stream until the next item has arrived from the
    /// stage before; the host goes on issuing.
    fn arrived(&self, rt: &Runtime) -> Result<()> {
        match &self.prev {
            Some(p) => Ok(rt.hold(&p.arrived, self.item + 1)?),
            None => Ok(()),
        }
    }

    /// The next item is issued, of `rows` rows at `vars`: once the next
    /// stage has taken the one before it, push it on, and mark it.
    fn issued(&mut self, rt: &Runtime, vars: &BTreeMap<String, u64>, rows: u64, answer: Option<Answer>) -> Result<()> {
        let k = self.item;
        if let Some(n) = &self.next {
            if k > 0 {
                wait(&n.done, k - 1, "the next stage reported")?;
            }
            for (name, to) in &n.boxes {
                rt.push(name, vars, to).with_context(|| format!("pushing `{name}` to the next stage"))?;
            }
        }
        rt.fault()?;
        self.watch.issued(Issued { item: k, rows, issued_us: now_us(), mark: rt.mark()?, answer })?;
        self.item += 1;
        Ok(())
    }
}

/// When each item a stage issued left its device: one `pp item` line per
/// item (its number, its rows, `issued_us` and `done_us` of the wall
/// clock), logged by a thread that waits on the item's [`Mark`] so the
/// issuing thread never does, and tells the neighbours: the next stage
/// that the item arrived, the previous that it was taken. On the last
/// stage it also answers the head with the item's tokens. A stage's
/// period is done to done; against what the stage alone takes for the
/// item's shape, the rest is its pipeline bubble. A fault on the device
/// surfaces at its item's mark and ends the process.
struct Watch {
    issued: Option<mpsc::Sender<Issued>>,
}

struct Issued {
    item: u64,
    rows: u64,
    issued_us: u64,
    mark: Mark,
    answer: Option<Answer>,
}

impl Watch {
    fn new(stage: u64, mut prev: Option<TcpStream>, mut next: Option<TcpStream>, mut head: Option<TcpStream>) -> Watch {
        let (issued, rx) = mpsc::channel::<Issued>();
        std::thread::spawn(move || {
            for Issued { item, rows, issued_us, mark, answer } in rx {
                if let Err(e) = mark.wait() {
                    tracing::error!(stage, item, "device: {e:#}");
                    std::process::exit(1);
                }
                tracing::info!(stage, item, rows, issued_us, done_us = now_us(), "pp item");
                let tokens = answer.map_or(Vec::new(), |a| {
                    let mut v = a.fill.decode(a.fetch.bytes());
                    v.truncate(a.keep);
                    v
                });
                let told = next
                    .as_mut()
                    .map_or(Ok(()), |w| tell(w, item))
                    .and(prev.as_mut().map_or(Ok(()), |w| tell(w, item)))
                    .and(head.as_mut().map_or(Ok(()), |h| send(h, &tokens)));
                if let Err(e) = told {
                    tracing::error!(stage, item, "{e:#}");
                    std::process::exit(1);
                }
            }
        });
        Watch { issued: Some(issued) }
    }

    fn issued(&self, i: Issued) -> Result<()> {
        let issued = self.issued.as_ref().expect("taken only on drop");
        if issued.send(i).is_err() {
            bail!("the watch thread is gone");
        }
        Ok(())
    }
}

fn now_us() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_micros() as u64)
}

/// The watch goes with the process: an item still in flight when the head
/// is gone may be holding for data that will never come.
impl Drop for Watch {
    fn drop(&mut self) {
        drop(self.issued.take());
    }
}
