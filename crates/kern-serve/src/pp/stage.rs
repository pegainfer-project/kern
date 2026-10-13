//! A downstream stage: loads its slice of the model, says hello to the
//! head, joins the stage after it and the one before from the table it
//! gets back, then replays the head's items in order as they arrive. It
//! keeps no accounting: the page ids, positions and slots of every item
//! are the head's, and so are the zeroing of a fresh sequence's slot, the
//! pages and slots a kept prefix copies and the host units an entry parks
//! in or wakes from, in the block the head's table sizes. The last stage answers every
//! item with its tokens. With a decode tier, the stage mirrors its pools
//! at start and runs every push the head sends behind the item before it.

use std::net::TcpListener;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::{bail, ensure, Context, Result};
use kern_manifest::{Protocol, Verified};
use kern_run::{Vars, Weights};
use kern_runtime::{Capacity, Runtime};
use tracing::{error, info};

use super::{arrivals, connect, recv, send, Answer, Flow, Hello, Job, Next, Prev, Report, Table, LANDINGS};
use crate::pd::{mirror::Mirrors, shapes};
use crate::ranks::{Act, Moves};
use kern_pool::Copies;

/// How a stage loads and where its head listens.
pub struct Stage {
    pub head: String,
    pub gpu: usize,
    pub kernels: Option<std::path::PathBuf>,
    pub capacity: Option<u64>,
    pub max_seqs: usize,
    pub eager: bool,
}

/// How long a stage keeps trying to reach a head that is not up yet.
const CONNECT: Duration = Duration::from_secs(600);

pub fn run(m: Verified, weights: &Weights, o: Stage) -> Result<()> {
    let cut = m.cut.clone().context("`--head` serves a stage of a cut manifest")?;
    ensure!(cut.stage > 0, "stage 0 is the head; it takes `--listen`");
    let last = cut.stage + 1 == cut.stages;
    let t0 = Instant::now();
    let capacity = Capacity { tokens: o.capacity, seqs: (o.max_seqs + 1) as u64, landings: LANDINGS as usize };
    let mut rt = Runtime::load(&m, o.kernels.as_deref(), o.gpu, Some(capacity), None)?;
    rt.set_eager(o.eager);
    weights.bind(&mut rt)?;
    let p = Protocol::check_unsampled(&rt.manifest)?;
    info!(
        stage = cut.stage,
        gpu = o.gpu,
        pages = rt.pages_total(),
        load_s = t0.elapsed().as_secs_f64(),
        "stage loaded"
    );

    let listener = TcpListener::bind("0.0.0.0:0")?;
    let mut head = connect(&o.head, CONNECT)?;
    let hello = Hello {
        stage: cut.stage,
        cut: cut.id.clone(),
        pages: Some(rt.pages_total() as u64).filter(|&n| n > 0),
        port: listener.local_addr()?.port(),
        arrivals: arrivals(&rt, &p)?,
        states: shapes(&rt.manifest),
        host: rt.host_unit(),
    };
    send(&mut head, &hello)?;
    let table: Table = recv(&mut head)?.context("the head closed before its table")?;
    ensure!(
        hello.pages.is_none_or(|n| table.pages <= n),
        "the head addresses {} pages, this stage has {}",
        table.pages,
        rt.pages_total()
    );
    let next = match (table.next, last) {
        (Some((addr, h)), false) => Some(Next::connect(&rt, &addr, &h, || {})?),
        (None, true) => None,
        (n, _) => bail!(
            "stage {} of {}: the head sent {} next stage",
            cut.stage,
            cut.stages,
            if n.is_some() { "a" } else { "no" }
        ),
    };
    super::head::reserve(&mut rt, table.host)?;
    let mine = shapes(&rt.manifest);
    let mut mirrors: Vec<Mirrors> =
        table.decode.into_iter().map(|d| Mirrors::map(&rt, d, &mine)).collect::<Result<_>>()?;
    let prev = Prev::accept(&rt, &listener, hello.arrivals.keys().cloned().collect())?;
    kern_run::run_once(&rt, &p)?;
    info!(stage = cut.stage, head = %o.head, last, "stage serving");

    let (tx, jobs) = mpsc::channel::<Job>();
    std::thread::spawn({
        let mut r = head.try_clone()?;
        move || loop {
            match recv::<Job>(&mut r) {
                Ok(Some(job)) => {
                    if tx.send(job).is_err() {
                        return;
                    }
                }
                Ok(None) => {
                    info!("the head closed its connection");
                    return;
                }
                Err(e) => {
                    error!("head connection: {e:#}");
                    return;
                }
            }
        }
    });

    let mut w = head.try_clone()?;
    let report = Box::new(move |r: Report| send(&mut w, &r));
    let mut flow = Flow::new(cut.stage, Some(prev), next, report, last);
    let holds = (rt.pages_total() > 0, rt.seq_slots() > 0);
    for job in jobs {
        let item = match job {
            super::Job::Item(item) => item,
            super::Job::Push(push) => {
                let m = mirrors
                    .get_mut(push.tier)
                    .with_context(|| format!("a push to decode tier {}, of none", push.tier))?;
                m.push(&rt, &push)?;
                flow.pushed(&rt, push.req)?;
                continue;
            }
            super::Job::Place { tier, rank, epoch, placement } => {
                let m = mirrors.get_mut(tier).with_context(|| format!("a placement of decode tier {tier}, of none"))?;
                m.place(&rt, rank, epoch, &placement)?;
                continue;
            }
        };
        let vars: Vars = item.vars.into_iter().filter(|(k, _)| rt.manifest.vars.contains_key(k)).collect();
        // The acts, fills and tables are stream-ordered behind the item
        // before, so they go over while the activations are still on their way.
        for act in item.acts {
            self::act(&mut rt, act, holds)?;
        }
        for (name, v) in &item.rows {
            kern_run::write_named(&mut rt, &p, name, v, &vars)?;
        }
        flow.begin(&rt, &vars)?;
        rt.issue(&item.program, &vars).with_context(|| format!("`{}`", item.program))?;
        let emits = p.forwards.iter().find(|f| f.name == item.program).and_then(|f| f.emits);
        let answer = emits
            .filter(|_| last)
            .map(|i| {
                let fill = p.fills[i].clone();
                let keep = vars.get(&p.groups.var).copied().unwrap_or(1) as usize * fill.width as usize;
                Ok::<_, anyhow::Error>(Answer { fetch: rt.fetch(&fill.name)?, fill, keep })
            })
            .transpose()?;
        flow.issued(&rt, &vars, vars.get(&p.rows.var).copied().unwrap_or(0), answer)?;
    }
    Ok(())
}

/// Run one of the head's acts on this stage's states, the kinds it
/// `holds` (pages, slots) only: a stage without paged or per-sequence
/// state has no arena for the head's ids of that kind.
fn act(rt: &mut Runtime, act: Act, holds: (bool, bool)) -> Result<()> {
    fn held<S, D>(m: Moves<S, D>, holds: (bool, bool)) -> Copies<S, D> {
        let pages = if holds.0 { m.pages } else { Vec::new() };
        Copies { pages, slot: m.slot.filter(|_| holds.1) }
    }
    match act {
        Act::Zero(slot) if holds.1 => rt.zero_slot(slot)?,
        Act::Zero(_) => {}
        Act::Copy(m) => {
            let c = held(m, holds);
            rt.copy(&c.pages, c.slot.as_slice())?
        }
        Act::Park(m) => rt.copy_out(&held(m, holds))?,
        Act::Wake(m) => rt.copy_in(&held(m, holds))?,
        Act::Remap(_) => bail!("the head sent a remap, which its pool never makes"),
    }
    Ok(())
}
