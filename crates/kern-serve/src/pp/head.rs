//! The head: stage 0 of a pipeline, the scheduler every stage follows.
//!
//! A request leases its prompt's pages (and slot) from the head's pool,
//! the one pool of the pipeline, and goes out as items: the prompt but its
//! last token in chunks through the chunk program, then the last token
//! through the one-row step, whose tokens the last stage hands back; or,
//! when the chunk program hands a token back itself (a model whose chunked
//! kernels must see every prompt token), the whole prompt in chunks, the
//! last chunk's token the answer. A chunk program over several sequences
//! takes the admitted prompts' next rows packed back to back ([`pack`]).
//!
//! Each item is sent to every stage before the head stages and issues it
//! itself, so a stage waiting on its device for an item always has it on
//! the way. The last stage answers every item, so the head knows how many
//! are in the pipeline: a full item (all its rows or all its sequences)
//! goes out at once, a partial one only
//! while fewer items than stages are in flight, so packing waits for rows
//! only when the stages have work. The first token finishes the request.
//! Nothing waits on the head's device but the clock read, done when
//! nothing is in flight or the clock ring is half full.

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

use super::pack::{pack, Bounds, Waiting};
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

/// An admitted prompt with rows still to send.
struct Active {
    id: RequestId,
    lease: Lease,
    ids: Vec<i64>,
    pos: usize,
}

/// A prompt whose last rows are out: its token is entry `at` of the reply
/// to the item that carried them, and its lease lives until then.
struct Finish {
    at: usize,
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
    bounds: Bounds,
    /// Items in flight below which a partial item goes out: the stages.
    depth: usize,
    max_seqs: usize,
    stop_tokens: Vec<u32>,
    io: tokio::runtime::Runtime,
    stages: Vec<Writer>,
    replies: mpsc::Receiver<Vec<i64>>,
    waiting: VecDeque<QueuedRequest>,
    active: VecDeque<Active>,
    /// Per item in flight, oldest first, the prompts its reply finishes.
    due: VecDeque<Vec<Finish>>,
    /// The line tables as the stages last had them written.
    lines: BTreeMap<String, Vec<i64>>,
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

    let prefill =
        p.chunk().cloned().context("a pipeline prefills through a program taking one sequence's rows as fed")?;
    // The head is stage 0: whether the last stage's chunk emits is not in its manifest, so
    // a pipeline with no one-row step feeds every prompt row as chunks and the last stage answers.
    let step = p.forward(1, Rows::Const(1)).cloned();
    let max = p.rows.max as usize;
    let chunk = match h.chunk {
        None => max,
        Some(c) if (1..=max as u64).contains(&c) => c as usize,
        Some(c) => bail!("--chunk {c}: the manifest's `tokens` bound is {max}; a chunk can only be smaller"),
    };
    let context = p.context.as_ref().map_or(usize::MAX, |c| c.max as usize);
    let bounds = Bounds { rows: chunk, seqs: prefill.groups as usize, context };
    let clock = Clock::new(&p)?;
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

    let step_name = step.as_ref().map_or("-", |f| f.name.as_str()).to_string();
    info!(pages, page = rt.page(), chunk, seqs = bounds.seqs, prefill = %prefill.name, step = %step_name,
          "pipeline head ready");
    Ok(PpHead {
        rt,
        p,
        prefill,
        step,
        bounds,
        depth: cut.stages as usize,
        max_seqs: h.max_seqs,
        stop_tokens: h.stop_tokens,
        io,
        stages,
        replies,
        waiting: VecDeque::new(),
        active: VecDeque::new(),
        due: VecDeque::new(),
        lines: BTreeMap::new(),
        clock,
    })
}

/// Read one stage's connection to its end: the last stage's answers go to
/// the scheduler; anything closing ends the head.
async fn listen(stage: u64, mut r: Reader, tokens: Option<mpsc::Sender<Vec<i64>>>) {
    let why = loop {
        match (recv::<Vec<i64>>(&mut r).await, &tokens) {
            (Ok(Some(t)), Some(tx)) => {
                if tx.send(t).is_err() {
                    return;
                }
            }
            (Ok(Some(_)), None) => break "a stage other than the last answered an item".to_string(),
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

    /// Prompts leased and not finished.
    fn live(&self) -> usize {
        self.active.len() + self.due.iter().map(Vec::len).sum::<usize>()
    }

    /// Lease each waiting prompt in order; one that cannot be seated stops
    /// the scan.
    fn admit(&mut self, ledger: &mut RequestLedger) -> Result<()> {
        while let Some(q) = self.waiting.front() {
            let id = q.id;
            if ledger.is_aborted(id) {
                ledger.retire(id);
                self.waiting.pop_front();
                continue;
            }
            if self.live() >= self.max_seqs {
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
                    let ids = q.request.prompt_tokens.iter().map(|&t| t as i64).collect();
                    self.active.push_back(Active { id, lease, ids, pos: 0 });
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

    /// The rows of a prompt that go through the chunk program: all of them,
    /// or all but the last when the step takes that one.
    fn chunked(&self, a: &Active) -> usize {
        a.ids.len() - usize::from(self.step.is_some())
    }

    /// Send every item that is ready: packed chunks while they are full or
    /// the pipeline has room, then the step of each prompt whose chunks are out.
    fn issue(&mut self) -> Result<()> {
        loop {
            let waiting: Vec<Waiting> =
                self.active.iter().map(|a| Waiting { pos: a.pos, left: self.chunked(a) - a.pos }).collect();
            let plan = pack(&waiting, self.bounds);
            let rows: usize = plan.iter().map(|&(_, n)| n).sum();
            let full = rows == self.bounds.rows || plan.len() == self.bounds.seqs;
            if plan.is_empty() || (!full && self.due.len() >= self.depth) {
                break;
            }
            self.send_chunk(&plan)?;
        }
        while let Some(i) = self.step.as_ref().and(self.active.iter().position(|a| a.pos == self.chunked(a))) {
            let a = self.active.remove(i).expect("found");
            let f = self.step.clone().expect("checked");
            let pos = a.pos;
            let finish = Finish { at: 0, id: a.id, _lease: a.lease };
            self.send_item(&f, &[(&finish._lease, pos, &a.ids[pos..])])?;
            self.due.push_back(vec![finish]);
        }
        Ok(())
    }

    /// One packed chunk item: each planned prompt's next rows; the prompts
    /// it finishes leave the active set for the item's reply.
    fn send_chunk(&mut self, plan: &[(usize, usize)]) -> Result<()> {
        let f = self.prefill.clone();
        let active = std::mem::take(&mut self.active);
        let parts: Vec<(&Lease, usize, &[i64])> = plan
            .iter()
            .map(|&(i, n)| {
                let a = &active[i];
                (&a.lease, a.pos, &a.ids[a.pos..a.pos + n])
            })
            .collect();
        let sent = self.send_item(&f, &parts);
        drop(parts);
        self.active = active;
        sent?;
        let emits = self.step.is_none();
        let mut finished = Vec::new();
        for (at, &(i, n)) in plan.iter().enumerate().rev() {
            self.active[i].pos += n;
            if emits && self.active[i].pos == self.active[i].ids.len() {
                finished.push((at, i));
            }
        }
        // Highest index first, so each removal leaves the next one's index in place.
        let mut due: Vec<Finish> = finished
            .into_iter()
            .map(|(at, i)| {
                let a = self.active.remove(i).expect("planned");
                Finish { at, id: a.id, _lease: a.lease }
            })
            .collect();
        due.reverse();
        self.due.push_back(due);
        Ok(())
    }

    /// Stage one item of `parts` (a lease, its position, its rows) on every
    /// stage and on the head. A sequence's first item zeroes its slot on the
    /// stages (the head's own lease zeroed it already); a line table goes
    /// out only when the item's sequences change it.
    fn send_item(&mut self, f: &Forward, parts: &[(&Lease, usize, &[i64])]) -> Result<()> {
        let leases: Vec<&Lease> = parts.iter().map(|&(l, ..)| l).collect();
        let positions: Vec<usize> = parts.iter().map(|&(_, pos, _)| pos).collect();
        let lens: Vec<usize> = parts.iter().map(|(.., ids)| ids.len()).collect();
        let ids: Vec<i64> = parts.iter().flat_map(|(.., ids)| ids.iter().copied()).collect();
        let (vars, values) = kern_run::fills(&self.p, &leases, &positions, &lens, &ids);
        let mut rows: BTreeMap<String, Vec<i64>> = values.into_iter().map(|(f, v)| (f.name.clone(), v)).collect();
        for t in &self.p.page_tables {
            let mut v = Vec::with_capacity(t.width * leases.len());
            for l in &leases {
                l.extend_row(&t.name, &mut v)?;
            }
            rows.insert(t.name.clone(), v.into_iter().map(i64::from).collect());
        }
        for t in &self.p.line_tables {
            let cols = match t.axis {
                Axis::Tray => self.p.tray.as_ref().map_or(1, |b| b.max),
                _ => self.p.groups.max,
            };
            let v: Vec<i64> = kern_run::line_rows(t, &leases, cols as usize)?.into_iter().map(i64::from).collect();
            if self.lines.get(&t.name) != Some(&v) {
                self.lines.insert(t.name.clone(), v.clone());
                rows.insert(t.name.clone(), v);
            }
        }
        let zero_slots = leases.iter().zip(&positions).filter(|(_, &pos)| pos == 0).filter_map(|(l, _)| l.seq_slot());
        let item = Item { program: f.name.clone(), vars, rows, zero_slots: Vec::new() };
        let out = Item { zero_slots: zero_slots.collect(), ..item.clone() };
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

    /// Admit, send, collect, and read the clock whenever this step issued
    /// nothing: the read waits for the head's device, which is also how a
    /// mailbox wait that gave up on a dead stage surfaces here.
    fn advance(&mut self, ledger: &mut RequestLedger) -> Result<()> {
        let issued = self.clock.pending();
        self.admit(ledger)?;
        self.issue()?;
        self.collect(ledger)?;
        if self.clock.pending() == issued || self.clock.half_full() {
            self.clock.flush(&self.rt, 0)?;
        }
        Ok(())
    }

    /// Every answer back since the last step retires its item and finishes
    /// the prompts it carried last: the last stage answers in issue order.
    fn collect(&mut self, ledger: &mut RequestLedger) -> Result<()> {
        loop {
            let t = match self.replies.try_recv() {
                Ok(t) => t,
                Err(mpsc::TryRecvError::Empty) => return Ok(()),
                Err(mpsc::TryRecvError::Disconnected) => bail!("the last stage's connection is gone"),
            };
            let Some(due) = self.due.pop_front() else { bail!("an answer came back for no item") };
            for r in due {
                let Some(&tok) = t.get(r.at) else {
                    bail!("the last stage handed back {} tokens, no token {}", t.len(), r.at)
                };
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
            num_running_reqs: (self.active.len() + self.due.iter().map(Vec::len).sum::<usize>()) as u64,
            num_waiting_reqs: self.waiting.len() as u64,
            spec_decode: None,
        }
    }
}
