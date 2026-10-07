//! The head: stage 0 of a pipeline, the scheduler every stage follows.
//!
//! A request leases its prompt's pages (and slot) from the head's pool,
//! the one pool of the pipeline, and goes out as items: the prompt but its
//! last token in chunks through the chunk program, then the last token
//! through the one-row step, whose tokens the last stage hands back; or,
//! when the chunk program hands a token back itself (a model whose chunked
//! kernels must see every prompt token), the whole prompt in chunks, the
//! last chunk's token the answer. Each item is sent
//! to every stage before the head stages and issues it itself, so a stage
//! waiting on its device for an item always has it on the way. The first
//! token finishes the request. Nothing waits on the head's device but the
//! clock read, done when nothing is in flight or the clock ring is half
//! full.

use std::collections::{BTreeMap, VecDeque};
use std::sync::mpsc;

use anyhow::{bail, ensure, Context, Result};
use kern_manifest::protocol::{Axis, Forward, Rows};
use kern_manifest::{Protocol, Verified};
use kern_pool::{Denied, Lease};
use kern_run::Weights;
use kern_runtime::{Capacity, Error, Runtime};
use pegainfer_frontend::engine::{
    FinishReason, QueuedRequest, RejectReason, RequestId, RequestLedger, Scheduler, SchedulerMetrics,
};
use tokio::net::TcpListener;
use tracing::{error, info};

use super::{connect_downstream, recv, send, topology, wire, Clock, Handles, Hello, Item, Reader, Table, Writer};
use crate::scheduler::Facts;

/// How the head loads and what it may run at once.
pub struct Head {
    pub listen: String,
    pub gpu: usize,
    pub kernels: Option<std::path::PathBuf>,
    pub capacity: Option<u64>,
    pub chunk: Option<u64>,
    pub max_seqs: usize,
    pub eager: bool,
    pub stop_tokens: Vec<u32>,
}

/// A request whose items are out and whose token is not back yet.
struct Inflight {
    id: RequestId,
    _lease: Lease,
}

pub struct PpHead {
    rt: Runtime,
    p: Protocol,
    prefill: Forward,
    /// The one-row step the last prompt token goes through; none when the
    /// chunk program hands the first token back itself.
    step: Option<Forward>,
    chunk: usize,
    max_seqs: usize,
    stop_tokens: Vec<u32>,
    io: tokio::runtime::Runtime,
    stages: Vec<Writer>,
    replies: mpsc::Receiver<Vec<i64>>,
    waiting: VecDeque<QueuedRequest>,
    inflight: VecDeque<Inflight>,
    clock: Clock,
}

/// Wait for every other stage, load stage 0 at the pages they all have,
/// map stage 1's mailbox and answer each stage with its table.
pub fn load(m: Verified, weights: &Weights, h: Head) -> Result<PpHead> {
    let cut = m.cut.clone().context("the head serves a cut manifest")?;
    ensure!(cut.stage == 0, "`--pp-listen` serves stage 0; this manifest is stage {} of {}", cut.stage, cut.stages);
    let io = super::io()?;
    let listener = io.block_on(TcpListener::bind(&h.listen)).with_context(|| format!("binding {}", h.listen))?;
    info!(listen = %h.listen, stages = cut.stages, cut = %cut.id, "waiting for the stages");
    let mut hellos: BTreeMap<u64, (Hello, Reader, Writer)> = BTreeMap::new();
    while (hellos.len() as u64) < cut.stages - 1 {
        let (s, addr) = io.block_on(listener.accept())?;
        let (mut r, w) = wire(s);
        let hello: Hello = io.block_on(recv(&mut r))?.with_context(|| format!("{addr} closed before its hello"))?;
        ensure!(
            hello.cut == cut.id,
            "{addr}: stage {} is of cut `{}`, the head of `{}`",
            hello.stage,
            hello.cut,
            cut.id
        );
        ensure!(
            (1..cut.stages).contains(&hello.stage) && !hellos.contains_key(&hello.stage),
            "{addr}: stage {} is not a stage still missing of 1..{}",
            hello.stage,
            cut.stages
        );
        info!(stage = hello.stage, %addr, pages = hello.pages, "stage joined");
        hellos.insert(hello.stage, (hello, r, w));
    }
    let unit = kern_pool::page_unit(&m);
    let fewest = hellos.values().map(|(x, ..)| x.pages).min().expect("at least one other stage");
    let tokens = h.capacity.map_or(fewest * unit, |c| c.min(fewest * unit));
    let seqs = (h.max_seqs + 1) as u64;
    let mut rt = Runtime::load(
        &m,
        h.kernels.as_deref(),
        h.gpu,
        Some(Capacity { tokens: Some(tokens), seqs }),
        topology(&m).as_ref(),
    )?;
    rt.set_eager(h.eager);
    weights.bind(&mut rt)?;
    connect_downstream(&mut rt, &hellos[&1].0.mailbox)?;
    let p = Protocol::check_unsampled(&rt.manifest)?;
    kern_run::run_once(&rt, &p)?;
    let pages = rt.pages_total() as u64;

    let downstream: Vec<Option<Handles>> =
        (1..cut.stages).map(|s| hellos.get(&(s + 1)).map(|(x, ..)| x.mailbox.clone())).collect();
    let (tx, replies) = mpsc::channel();
    let mut stages = Vec::new();
    for ((s, (_, r, mut w)), down) in hellos.into_iter().zip(downstream) {
        io.block_on(send(&mut w, &Table { downstream_mailbox: down, pages }))?;
        let tx = (s == cut.stages - 1).then(|| tx.clone());
        io.spawn(listen(s, r, tx));
        stages.push(w);
    }

    let prefill =
        p.chunk().cloned().context("a pipeline prefills through a program taking one sequence's rows as fed")?;
    let step = match prefill.emits {
        Some(_) => None,
        None => Some(p.forward(1, Rows::Const(1)).cloned().context("no program takes one sequence of one row")?),
    };
    let max = p.rows.max as usize;
    let chunk = match h.chunk {
        None => max,
        Some(c) if (1..=max as u64).contains(&c) => c as usize,
        Some(c) => bail!("--chunk {c}: the manifest's `tokens` bound is {max}; a chunk can only be smaller"),
    };
    let clock = Clock::new(&p)?;
    let step_name = step.as_ref().map_or("-", |f| f.name.as_str()).to_string();
    info!(pages, page = rt.page(), chunk, prefill = %prefill.name, step = %step_name, "pipeline head ready");
    Ok(PpHead {
        rt,
        p,
        prefill,
        step,
        chunk,
        max_seqs: h.max_seqs,
        stop_tokens: h.stop_tokens,
        io,
        stages,
        replies,
        waiting: VecDeque::new(),
        inflight: VecDeque::new(),
        clock,
    })
}

/// Read one stage's connection to its end: the last stage's tokens go to
/// the scheduler; anything closing ends the head.
async fn listen(stage: u64, mut r: Reader, tokens: Option<mpsc::Sender<Vec<i64>>>) {
    let why = loop {
        match (recv::<Vec<i64>>(&mut r).await, &tokens) {
            (Ok(Some(t)), Some(tx)) => {
                if tx.send(t).is_err() {
                    return;
                }
            }
            (Ok(Some(_)), None) => break "a stage other than the last handed tokens back".to_string(),
            (Ok(None), _) => break "stage closed its connection; the pipeline is gone".to_string(),
            (Err(e), _) => break format!("stage connection: {e:#}"),
        }
    };
    error!(stage, "{why}");
    std::process::exit(1);
}

impl PpHead {
    pub fn facts(&self) -> Facts {
        Facts {
            total_blocks: self.rt.pages_total(),
            block_size: self.rt.page() as usize,
            max_request_tokens: self.rt.max_seq_tokens(),
        }
    }

    /// Lease each waiting prompt in order and send its items out; one that
    /// cannot be seated stops the scan.
    fn admit(&mut self, ledger: &mut RequestLedger) -> Result<()> {
        while let Some(q) = self.waiting.front() {
            let id = q.id;
            if ledger.is_aborted(id) {
                ledger.retire(id);
                self.waiting.pop_front();
                continue;
            }
            if self.inflight.len() >= self.max_seqs {
                break;
            }
            let (prompt, max_tokens) = (q.request.prompt_tokens.len(), q.request.max_tokens);
            let limit = self.rt.max_seq_tokens();
            let leased =
                if prompt == 0 { Err(Error::Denied(Denied::ExceedsRow { limit })) } else { self.rt.lease(prompt) };
            let reject = match leased {
                Ok(lease) => {
                    let q = self.waiting.pop_front().unwrap();
                    ledger.admit(id);
                    ledger.set_cached_tokens(id, 0);
                    self.send_prompt(&q, &lease)?;
                    self.inflight.push_back(Inflight { id, _lease: lease });
                    continue;
                }
                Err(Error::Denied(Denied::Busy | Denied::Remapping)) => break,
                Err(Error::Denied(Denied::ExceedsRow { limit })) => {
                    Some(RejectReason::ContextLength { prompt_tokens: prompt, max_tokens, limit })
                }
                Err(Error::Denied(Denied::ExceedsPool)) => {
                    Some(RejectReason::KvBudget { prompt_tokens: prompt, worst_case_tokens: prompt })
                }
                Err(e) => return Err(e.into()),
            };
            if let Some(r) = reject {
                ledger.reject(id, r);
                self.waiting.pop_front();
            }
        }
        Ok(())
    }

    /// The prompt as items: all but its last token in chunks, the last
    /// through the step, whose token comes back; or all of it in chunks
    /// when the chunk program hands the token back.
    fn send_prompt(&mut self, q: &QueuedRequest, lease: &Lease) -> Result<()> {
        let ids: Vec<i64> = q.request.prompt_tokens.iter().map(|&t| t as i64).collect();
        let n = ids.len();
        let (prefill, step) = (self.prefill.clone(), self.step.clone());
        let chunked = if step.is_some() { n - 1 } else { n };
        let mut pos = 0;
        while pos < chunked {
            let c = (chunked - pos).min(self.chunk);
            let reply = step.is_none() && pos + c == n;
            self.send_item(&prefill, lease, pos, &ids[pos..pos + c], reply)?;
            pos += c;
        }
        match &step {
            Some(f) => self.send_item(f, lease, n - 1, &ids[n - 1..], true),
            None => Ok(()),
        }
    }

    fn send_item(&mut self, f: &Forward, lease: &Lease, pos: usize, ids: &[i64], reply: bool) -> Result<()> {
        let (vars, values) = kern_run::fills(&self.p, std::slice::from_ref(lease), &[pos], ids.len(), ids);
        let mut rows: BTreeMap<String, Vec<i64>> = values.into_iter().map(|(f, v)| (f.name.clone(), v)).collect();
        for t in &self.p.page_tables {
            let mut v = Vec::with_capacity(t.width);
            lease.extend_row(&t.name, &mut v)?;
            rows.insert(t.name.clone(), v.into_iter().map(i64::from).collect());
        }
        // A sequence's line tables and slot are the same on every item: its
        // first item carries them and zeroes the slot on every stage (the
        // head's own lease zeroed its slot already).
        let first = pos == 0;
        for t in self.p.line_tables.iter().filter(|_| first) {
            let cols = match t.axis {
                Axis::Tray => self.p.tray.as_ref().map_or(1, |b| b.max),
                _ => self.p.groups.max,
            };
            let v = kern_run::line_rows(t, std::slice::from_ref(lease), cols as usize)?;
            rows.insert(t.name.clone(), v.into_iter().map(i64::from).collect());
        }
        let item = Item { program: f.name.clone(), vars, rows, zero_slot: None, reply };
        let out = Item { zero_slot: lease.seq_slot().filter(|_| first), ..item.clone() };
        for w in &mut self.stages {
            self.io.block_on(send(w, &out))?;
        }
        for (name, v) in &item.rows {
            kern_run::write_named(&mut self.rt, &self.p, name, v, &item.vars)?;
        }
        self.rt.issue(&item.program, &item.vars).with_context(|| format!("`{}`", item.program))?;
        self.clock.issued(ids.len() as u64);
        Ok(())
    }

    /// Admit, collect, and read the clock whenever this step issued
    /// nothing: the read waits for the head's device, which is also how a
    /// mailbox wait that gave up on a dead stage surfaces here.
    fn advance(&mut self, ledger: &mut RequestLedger) -> Result<()> {
        let issued = self.clock.pending();
        self.admit(ledger)?;
        self.collect(ledger)?;
        if self.clock.pending() == issued || self.clock.half_full() {
            self.clock.flush(&self.rt, 0)?;
        }
        Ok(())
    }

    /// Every token back since the last step finishes its request, oldest
    /// first: the last stage answers in issue order.
    fn collect(&mut self, ledger: &mut RequestLedger) -> Result<()> {
        loop {
            let t = match self.replies.try_recv() {
                Ok(t) => t,
                Err(mpsc::TryRecvError::Empty) => return Ok(()),
                Err(mpsc::TryRecvError::Disconnected) => bail!("the last stage's connection is gone"),
            };
            let Some(r) = self.inflight.pop_front() else { bail!("a token came back for no request") };
            let Some(&tok) = t.first() else { bail!("the last stage handed back no token") };
            if ledger.is_aborted(r.id) {
                ledger.retire(r.id);
                continue;
            }
            let tok = tok as u32;
            ledger.push_tokens(r.id, &[tok], &[]);
            let why = if self.stop_tokens.contains(&tok) { FinishReason::Stop } else { FinishReason::Length };
            ledger.finish(r.id, why);
        }
    }
}

// Built on the scheduler thread and used there only, like the `Tray`.
#[allow(unsafe_code)]
unsafe impl Send for PpHead {}

impl Scheduler for PpHead {
    fn submit(&mut self, request: QueuedRequest) {
        self.waiting.push_back(request);
    }

    /// A failed step ends the process: the stages see the connection close
    /// and follow, and no half-fed pipeline lingers behind an open port.
    fn step(&mut self, ledger: &mut RequestLedger) -> Result<()> {
        if let Err(e) = self.advance(ledger) {
            error!("pipeline head: {e:#}");
            std::process::exit(1);
        }
        Ok(())
    }

    fn metrics(&self) -> SchedulerMetrics {
        SchedulerMetrics {
            kv_used_blocks: self.rt.pages_used() as u64,
            kv_total_blocks: self.rt.pages_total() as u64,
            num_running_reqs: self.inflight.len() as u64,
            num_waiting_reqs: self.waiting.len() as u64,
            spec_decode: None,
        }
    }
}
