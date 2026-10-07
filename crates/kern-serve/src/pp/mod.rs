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
//! One TCP connection per downstream stage, framed by `tokio-util`'s
//! length-delimited codec, each frame one JSON message. A stage connects
//! and sends its [`Hello`]: its place, the cut it belongs to, its pages and
//! the export handles of its allocations (its mailbox among them). Once
//! every stage has said hello the head loads, maps the first stage's
//! mailbox and answers each stage with a [`Table`]: the next stage's
//! handles, to map its own outgoing peer from, and the pages. From then on
//! the connection carries the log: the head sends every stage every
//! [`Item`] in issue order — the program, its vars and the values of each
//! input it staged — and the last stage answers every item with its tokens
//! (none for a program that hands none back). TCP keeps the order; there is no
//! sequence number, no stop message and no fault message. A stage that
//! fails exits; a closed connection ends the process on the other side.
//!
//! The activations never touch the connection: they go GPU to GPU through
//! the mailbox the cut wired in, and each stage's device waits for its
//! predecessor on its own. A stage only enqueues; it synchronizes when it
//! has nothing to enqueue (to read its clock) or when it must read tokens.

pub mod head;
mod pack;
pub mod stage;

use std::collections::{BTreeMap, VecDeque};

use anyhow::{bail, Context, Result};
use bytes::Bytes;
use futures_util::{SinkExt, StreamExt};
use kern_manifest::protocol::Axis;
use kern_manifest::types::Fill;
use kern_manifest::Protocol;
use kern_runtime::{GroupRank, PeerHandle, Runtime, Topology};
use serde::{Deserialize, Serialize};
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::TcpStream;
use tokio_util::codec::{FramedRead, FramedWrite, LengthDelimitedCodec};

/// A stage introducing itself to the head.
#[derive(Debug, Serialize, Deserialize)]
pub struct Hello {
    pub stage: u64,
    pub cut: String,
    pub pages: u64,
    /// Every allocation the stage exports, by name, as hex fabric handles.
    pub mailbox: Handles,
}

/// The head's answer: the next stage's exports (none for the last stage)
/// and the pages every stage addresses.
#[derive(Debug, Serialize, Deserialize)]
pub struct Table {
    pub downstream_mailbox: Option<Handles>,
    pub pages: u64,
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

/// This runtime's exports as wire handles; a stage's box must be mappable
/// from another process, so a local handle is an error.
fn handles(rt: &Runtime) -> Result<Handles> {
    rt.export_handles()?
        .into_iter()
        .map(|(name, h)| {
            let b = h.to_bytes().with_context(|| {
                format!("`{name}` has no fabric handle; a pipeline stage maps its neighbour across processes")
            })?;
            Ok((name, hex::encode(b)))
        })
        .collect()
}

fn peer_handles(h: &Handles) -> Result<BTreeMap<String, PeerHandle>> {
    h.iter()
        .map(|(name, x)| {
            let b = hex::decode(x).with_context(|| format!("handle of `{name}`"))?;
            Ok((name.clone(), PeerHandle::from_bytes(&b).with_context(|| format!("handle of `{name}`"))?))
        })
        .collect()
}

/// A stage's place in the group of its outgoing edge: always the
/// upstream member, rank 0 of 2.
fn topology(rt_manifest: &kern_manifest::Manifest) -> Option<Topology> {
    rt_manifest.topology.as_ref().map(|t| Topology {
        groups: t.groups.iter().map(|(g, &size)| (g.clone(), GroupRank { index: 0, size })).collect(),
    })
}

/// Map the next stage's box into this stage's outgoing peer buffer.
fn connect_downstream(rt: &mut Runtime, downstream: &Handles) -> Result<()> {
    let groups: Vec<String> = rt.manifest.topology.iter().flat_map(|t| t.groups.keys().cloned()).collect();
    let own = rt.export_handles()?;
    let theirs = peer_handles(downstream)?;
    for g in groups {
        rt.import_peers(&g, &[own.clone(), theirs.clone()])
            .with_context(|| format!("mapping the next stage over `{g}`"))?;
    }
    let pending = rt.pending_peers();
    if !pending.is_empty() {
        bail!("peer buffers {pending:?} still unfilled after the next stage was mapped");
    }
    Ok(())
}

type Reader = FramedRead<OwnedReadHalf, LengthDelimitedCodec>;
type Writer = FramedWrite<OwnedWriteHalf, LengthDelimitedCodec>;

fn wire(s: TcpStream) -> (Reader, Writer) {
    let _ = s.set_nodelay(true);
    let (r, w) = s.into_split();
    (FramedRead::new(r, LengthDelimitedCodec::new()), FramedWrite::new(w, LengthDelimitedCodec::new()))
}

async fn send<T: Serialize>(w: &mut Writer, msg: &T) -> Result<()> {
    w.send(Bytes::from(serde_json::to_vec(msg)?)).await.context("sending")
}

/// The next message; `None` when the other side closed.
async fn recv<T: for<'de> Deserialize<'de>>(r: &mut Reader) -> Result<Option<T>> {
    match r.next().await {
        None => Ok(None),
        Some(frame) => Ok(Some(serde_json::from_slice(&frame.context("receiving")?)?)),
    }
}

/// The IO runtime a process's connections live on, one thread beside the
/// one driving the GPU.
fn io() -> Result<tokio::runtime::Runtime> {
    Ok(tokio::runtime::Builder::new_multi_thread().worker_threads(1).enable_all().build()?)
}

/// What a stage's device spent per item, read off its clock output: one
/// log line per item, busy and waiting on either side.
struct Clock {
    name: String,
    ring: u64,
    logged: u64,
    /// Rows of each item issued and not yet logged, oldest first.
    rows: VecDeque<u64>,
}

impl Clock {
    fn new(p: &Protocol) -> Result<Clock> {
        let f = p.any(Fill::Clock).context("a cut stage declares a `clock` fill")?;
        let Axis::Fixed(n) = f.axis else { bail!("`{}`: a clock is a fixed ring", f.name) };
        Ok(Clock { name: f.name.clone(), ring: n / 5, logged: 0, rows: VecDeque::new() })
    }

    fn issued(&mut self, rows: u64) {
        self.rows.push_back(rows);
    }

    /// Items issued and not yet logged.
    fn pending(&self) -> usize {
        self.rows.len()
    }

    /// Whether the oldest unread entry is half a ring from being overwritten.
    fn half_full(&self) -> bool {
        self.rows.len() as u64 >= self.ring / 2
    }

    /// Read the clock (waiting for the stage's stream) and log every item
    /// issued since the last read.
    fn flush(&mut self, rt: &Runtime, stage: u64) -> Result<()> {
        if self.rows.is_empty() {
            return Ok(());
        }
        let raw = rt.read_output(&self.name)?;
        let e: Vec<i64> = raw.as_chunks::<8>().0.iter().map(|c| i64::from_le_bytes(*c)).collect();
        let span = |a: i64, b: i64| (a != 0 && b != 0).then_some(b - a);
        for rows in self.rows.drain(..) {
            let k = self.logged;
            self.logged += 1;
            let x = &e[(k % self.ring) as usize * 5..][..5];
            if x[0] != k as i64 + 1 {
                tracing::warn!(stage, item = k, "pp clock entry overwritten before it was read");
                continue;
            }
            tracing::info!(
                stage,
                item = k,
                rows,
                busy_ns = span(x[2], x[3]),
                recv_wait_ns = span(x[1], x[2]),
                send_wait_ns = span(x[3], x[4]),
                "pp item"
            );
        }
        Ok(())
    }
}
