//! A downstream stage: loads its slice of the model, says hello to the
//! head, maps the next stage's mailbox from the table it gets back, then
//! replays the head's items in order. It keeps no accounting: the page
//! ids and positions of every item are the head's. A stage serves paged
//! state only; a per-sequence state (its slot zeroed on a sequence's
//! first item) waits for a model that needs it. The last stage answers
//! every item with its tokens.

use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::{bail, ensure, Context, Result};
use kern_manifest::{Protocol, Verified};
use kern_run::{Vars, Weights};
use kern_runtime::{Capacity, Runtime};
use tokio::net::TcpStream;
use tracing::{error, info};

use super::{connect_downstream, handles, recv, send, topology, wire, Hello, Item, Table, Watch};

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
    let mut rt = Runtime::load(&m, o.kernels.as_deref(), o.gpu, Some(capacity), topology(&m).as_ref())?;
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

    let io = super::io()?;
    let stream = io.block_on(connect(&o.head))?;
    let (mut r, mut w) = wire(stream);
    let hello = Hello { stage: cut.stage, cut: cut.id.clone(), pages: rt.pages_total() as u64, mailbox: handles(&rt)? };
    io.block_on(send(&mut w, &hello))?;
    let table: Table = io.block_on(recv(&mut r))?.context("the head closed before its table")?;
    ensure!(
        table.pages <= rt.pages_total() as u64,
        "the head addresses {} pages, this stage has {}",
        table.pages,
        rt.pages_total()
    );
    match (&table.downstream_mailbox, last) {
        (Some(d), false) => connect_downstream(&mut rt, d)?,
        (None, true) => {}
        (d, _) => bail!(
            "stage {} of {}: the head sent {} next stage's mailbox",
            cut.stage,
            cut.stages,
            if d.is_some() { "a" } else { "no" }
        ),
    }
    kern_run::run_once(&rt, &p)?;
    info!(stage = cut.stage, head = %o.head, last, "stage serving");

    let (tx, items) = mpsc::channel::<Item>();
    io.spawn(async move {
        loop {
            match recv::<Item>(&mut r).await {
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

    let mut watch = Watch::new(cut.stage);
    for item in items {
        let vars: Vars = item.vars.into_iter().filter(|(k, _)| rt.manifest.vars.contains_key(k)).collect();
        for (name, v) in &item.rows {
            kern_run::write_named(&mut rt, &p, name, v, &vars)?;
        }
        rt.issue(&item.program, &vars).with_context(|| format!("`{}`", item.program))?;
        watch.issued(&rt, vars.get(&p.rows.var).copied().unwrap_or(0))?;
        if last {
            let emits = p.forwards.iter().find(|f| f.name == item.program).and_then(|f| f.emits);
            let tokens = match emits.map(|i| &p.fills[i]) {
                Some(t) => {
                    let mut v = t.decode(&rt.read_output(&t.name)?);
                    v.truncate(vars.get(&p.groups.var).copied().unwrap_or(1) as usize * t.width as usize);
                    v
                }
                None => Vec::new(),
            };
            io.block_on(send(&mut w, &tokens))?;
        }
    }
    Ok(())
}

/// Reach the head, retrying while it is still loading.
async fn connect(head: &str) -> Result<TcpStream> {
    let t0 = Instant::now();
    loop {
        match TcpStream::connect(head).await {
            Ok(s) => return Ok(s),
            Err(e) if t0.elapsed() < CONNECT => {
                tracing::debug!("head {head}: {e}");
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
            Err(e) => return Err(e).with_context(|| format!("connecting to the head at {head}")),
        }
    }
}
