//! A stage's transport thread: the embedded Python `Stage` and what the
//! stage told it about each request, until its send ends.
//!
//! [`Ledger`] is what the thread knows, kept apart from Python: a request
//! is ready once its transfer is in and the item before it landed, and on
//! the last stage its first token is the entry the transfer names of the
//! answer to that item, which the watch reported just before the landing
//! (it reports in issue order, and a transfer follows the item that wrote
//! the request's last rows). Python is asked about ready requests only,
//! every [`POLL`] while any is in flight.

use std::collections::BTreeMap;
use std::sync::mpsc;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use kern_runtime::Runtime;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use super::{Bootstrap, Msg, Rank, Sent, Sglang};
use crate::pd::Req;
use crate::rooms::Transfer;

/// How often a send in flight is looked at.
const POLL: Duration = Duration::from_millis(1);

/// One request as the thread knows it.
struct Pending {
    room: u64,
    transfer: Option<Transfer>,
    /// The item before the transfer landed; on the last stage, with the
    /// first token.
    landed: Option<Option<i64>>,
}

/// What a stage's transport thread was told.
#[derive(Default)]
struct Ledger {
    last: bool,
    /// The last stage's latest answer.
    tokens: Vec<i64>,
    rooms: BTreeMap<Req, Pending>,
}

impl Ledger {
    /// One message; the room to open when it opens one.
    fn hear(&mut self, m: Msg) -> Result<Option<u64>> {
        match m {
            Msg::Open { req, room } => {
                if self.rooms.insert(req, Pending { room, transfer: None, landed: None }).is_some() {
                    bail!("request {req} opened twice");
                }
                return Ok(Some(room));
            }
            Msg::Transfer(t) => {
                let p = self
                    .rooms
                    .get_mut(&t.req)
                    .with_context(|| format!("a transfer of request {}, never opened", t.req))?;
                p.transfer = Some(t);
            }
            Msg::Tokens(t) => self.tokens = t,
            Msg::Landed(req) => {
                let p = self.rooms.get_mut(&req).with_context(|| format!("request {req} landed, never opened"))?;
                let t = p.transfer.as_ref().with_context(|| format!("request {req} landed with no transfer"))?;
                let tok = match self.last {
                    true => Some(*self.tokens.get(t.at).with_context(|| {
                        format!("request {req}: no token {} in an answer of {}", t.at, self.tokens.len())
                    })?),
                    false => None,
                };
                p.landed = Some(tok);
            }
        }
        Ok(None)
    }

    /// The requests whose send can go or is going: room, transfer, token.
    fn ready(&self) -> Vec<(u64, &Transfer, Option<i64>)> {
        self.rooms.values().filter_map(|p| Some((p.room, p.transfer.as_ref()?, p.landed?))).collect()
    }
}

/// Load the stage's transport: Python's `Stage` over `rt`'s pooled states,
/// on a thread of its own that reports every ended send to `sent`. Returns
/// what the thread hears and, on pp rank 0, what it broadcast to the other
/// ranks.
pub(crate) fn start(
    s: &Sglang,
    rt: &Runtime,
    rank: Rank,
    boot: &Bootstrap,
    mut sent: Sent,
) -> Result<(mpsc::Sender<Msg>, Vec<serde_json::Value>)> {
    let memory = rt.pooled_memory()?;
    let registered: usize = memory.iter().map(|m| m.chunks.len()).sum();
    let (page, last) = (rt.page(), rank.stage + 1 == rank.stages);
    let (s, boot) = (s.clone(), boot.clone());
    let (tx, rx) = mpsc::channel();
    let (ready, made) = mpsc::channel();
    Python::initialize();
    std::thread::Builder::new()
        .name("kern-sglang".into())
        .spawn(move || {
            let stage = Python::attach(|py| -> PyResult<(Py<PyAny>, String)> {
                let states = PyDict::new(py);
                for m in &memory {
                    states.set_item(&m.state, (m.ptr, m.object, m.reserved))?;
                }
                let chunks: Vec<(u64, u64)> = memory.iter().flat_map(|m| m.chunks.iter().copied()).collect();
                let kw = PyDict::new(py);
                kw.set_item("model_path", s.model_path.to_string_lossy())?;
                kw.set_item("ip", &s.ip)?;
                kw.set_item("bootstrap_port", boot.port)?;
                kw.set_item("ib_device", &s.ib_device)?;
                kw.set_item("rank", rank.stage)?;
                kw.set_item("size", rank.stages)?;
                kw.set_item("page", page)?;
                kw.set_item("states", states)?;
                kw.set_item("chunks", PyList::new(py, chunks)?)?;
                kw.set_item("slots", boot.slots)?;
                let json = py.import("json")?;
                let consensus = json.call_method1("loads", (serde_json::to_string(&boot.consensus).expect("json"),))?;
                kw.set_item("consensus", consensus)?;
                let stage = py.import("kern_sglang.prefill")?.getattr("Stage")?.call((), Some(&kw))?;
                let told = json.call_method1("dumps", (stage.getattr("consensus")?,))?.extract::<String>()?;
                Ok((stage.unbind(), told))
            });
            match stage {
                Ok((stage, told)) => {
                    let _ = ready.send(Ok(told));
                    let ledger = Ledger { last, ..Ledger::default() };
                    if let Err(e) = run(stage, ledger, rx, &mut sent) {
                        tracing::error!(stage = rank.stage, "sglang transport: {e:#}");
                        std::process::exit(1);
                    }
                }
                Err(e) => {
                    let _ = ready.send(Err(anyhow::anyhow!("{}", Python::attach(|py| describe(py, e)))));
                }
            }
        })
        .context("spawning the sglang transport thread")?;
    let told = made.recv().context("the sglang transport thread died loading")??;
    let consensus = serde_json::from_str(&told).context("pp rank 0's broadcast")?;
    tracing::info!(stage = rank.stage, chunks = registered, "sglang prefill rank ready");
    Ok((tx, consensus))
}

/// A Python error with its traceback.
fn describe(py: Python<'_>, e: PyErr) -> String {
    let tb = e.traceback(py).and_then(|t| t.format().ok()).unwrap_or_default();
    format!("{tb}{e}")
}

/// Hear the stage and move every ready send on until the process ends.
fn run(stage: Py<PyAny>, mut ledger: Ledger, rx: mpsc::Receiver<Msg>, sent: &mut Sent) -> Result<()> {
    loop {
        let first = match ledger.ready().is_empty() {
            true => Some(rx.recv().context("the stage is gone")?),
            false => match rx.recv_timeout(POLL) {
                Ok(m) => Some(m),
                Err(mpsc::RecvTimeoutError::Timeout) => None,
                Err(mpsc::RecvTimeoutError::Disconnected) => bail!("the stage is gone"),
            },
        };
        let mut opens = Vec::new();
        for m in first.into_iter().chain(rx.try_iter()) {
            opens.extend(ledger.hear(m)?);
        }
        let ready: Vec<(u64, Transfer, Option<i64>)> =
            ledger.ready().into_iter().map(|(room, t, tok)| (room, t.clone(), tok)).collect();
        if opens.is_empty() && ready.is_empty() {
            continue;
        }
        let ended = Python::attach(|py| -> PyResult<Vec<(Req, Option<String>)>> {
            let s = stage.bind(py);
            for room in opens {
                s.call_method1("open", (room,))?;
            }
            let mut ended = Vec::new();
            for (room, t, tok) in ready {
                let r: Option<String> = s.call_method1("advance", (room, t.pages, t.len, tok))?.extract()?;
                if let Some(why) = r {
                    ended.push((t.req, Some(why).filter(|w| !w.is_empty())));
                }
            }
            Ok(ended)
        })
        .map_err(|e| anyhow::anyhow!("{}", Python::attach(|py| describe(py, e))))?;
        for (req, failure) in ended {
            ledger.rooms.remove(&req);
            sent(req, failure)?;
        }
    }
}
