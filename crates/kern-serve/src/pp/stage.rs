//! A downstream stage: loads its slice of the model, says hello to the
//! head, joins the stage after it and the one before from the table it
//! gets back, then replays the head's items in order as they arrive. It
//! keeps no accounting: the page ids and positions of every item are the
//! head's. A stage serves paged state only; a per-sequence state (its
//! slot zeroed on a sequence's first item) waits for a model that needs
//! it. The last stage answers every item with its tokens.

use std::net::TcpListener;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::{bail, ensure, Context, Result};
use kern_manifest::{Protocol, Verified};
use kern_run::{Vars, Weights};
use kern_runtime::{Capacity, Runtime};
use tracing::{error, info};

use super::{arrivals, connect, recv, send, Answer, Flow, Hello, Item, Next, Prev, Table};

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
    let cut = m.cut.clone().context("`--pp-head` serves a stage of a cut manifest")?;
    ensure!(cut.stage > 0, "stage 0 is the head; it takes `--pp-listen`");
    let last = cut.stage + 1 == cut.stages;
    let t0 = Instant::now();
    let capacity = Capacity { tokens: o.capacity, seqs: (o.max_seqs + 1) as u64 };
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
        pages: rt.pages_total() as u64,
        port: listener.local_addr()?.port(),
        arrivals: arrivals(&rt, &p)?,
    };
    send(&mut head, &hello)?;
    let table: Table = recv(&mut head)?.context("the head closed before its table")?;
    ensure!(
        table.pages <= rt.pages_total() as u64,
        "the head addresses {} pages, this stage has {}",
        table.pages,
        rt.pages_total()
    );
    let next = match (table.next, last) {
        (Some((addr, h)), false) => Some(Next::connect(&rt, &addr, &h)?),
        (None, true) => None,
        (n, _) => bail!(
            "stage {} of {}: the head sent {} next stage",
            cut.stage,
            cut.stages,
            if n.is_some() { "a" } else { "no" }
        ),
    };
    let prev = Prev::accept(&rt, &listener)?;
    kern_run::run_once(&rt, &p)?;
    info!(stage = cut.stage, head = %o.head, last, "stage serving");

    let (tx, items) = mpsc::channel::<Item>();
    std::thread::spawn({
        let mut r = head.try_clone()?;
        move || loop {
            match recv::<Item>(&mut r) {
                Ok(Some(item)) => {
                    if tx.send(item).is_err() {
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

    let answers = last.then(|| head.try_clone()).transpose()?;
    let mut flow = Flow::new(cut.stage, Some(prev), next, answers);
    for item in items {
        let vars: Vars = item.vars.into_iter().filter(|(k, _)| rt.manifest.vars.contains_key(k)).collect();
        // The fills and tables are stream-ordered behind the item before,
        // so they go over while the activations are still on their way.
        for (name, v) in &item.rows {
            kern_run::write_named(&mut rt, &p, name, v, &vars)?;
        }
        flow.arrived(&rt)?;
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
