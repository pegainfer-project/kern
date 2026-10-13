//! `kern-serve`: an OpenAI-compatible HTTP endpoint over a kern manifest.
//!
//! The HTTP/protocol stack is pegainfer's frontend (vLLM's Rust server
//! crates underneath: completions, chat completions, streaming, chat
//! templates, stop strings). This crate contributes the engine behind it:
//! `scheduler::KernScheduler`, the pegainfer `Scheduler` contract over
//! [`ranks::Ranks`] — every rank's accounting, kept by the head — and
//! `executor::Executors`, one process per GPU running what it decided.
//! A cut manifest is served by `pp` instead, one process per stage, its
//! head behind a [`front::Front`], and may hand each prompt to a decode
//! tier ([`pd`]). The crate's public
//! surface is [`serve`], its option structs, and the accounting,
//! protocols and packing policy its tests drive.

#![deny(unsafe_code)]

mod executor;
pub mod front;
pub mod logline;
pub mod pd;
mod pp;
pub mod ranks;
mod scheduler;
mod watch;

pub use pp::pack;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Args;
use kern_manifest::{Protocol, Verified};
use kern_runtime::Capacity;
use pegainfer_frontend::engine::{
    drive, scheduler_pair, Engine, EngineInfo, KvCapacity, LaunchedEngine, LiveScheduler, Scheduler,
};
use pegainfer_frontend::vllm::{self, ParserSelection};
use tracing::info;

use scheduler::{KernScheduler, Policy};

/// The manifest and its artifacts, as named on the command line.
pub struct Artifacts {
    pub manifest: PathBuf,
    pub kernels: Option<PathBuf>,
    /// Weight entries, files or a weight cache; see [`kern_run::Weights`].
    pub weights: Vec<String>,
}

#[derive(Args, Debug, Clone)]
pub struct ServeOpts {
    /// Model id served by the API (default: the manifest's `model`)
    #[arg(long)]
    pub served_model_name: Option<String>,

    #[arg(long, default_value_t = 8000)]
    pub port: u16,

    /// CUDA device this process drives: one GPU per process
    #[arg(long, default_value_t = 0)]
    pub gpu: usize,

    /// KV pool in tokens per rank (rounded down to the page); every request
    /// reserves its worst case `prompt + max_tokens` at admission. Default:
    /// whatever device memory is left once weights, activations and scratch
    /// are allocated, less 1 GiB
    #[arg(long)]
    pub capacity: Option<u64>,

    /// Prefill chunk in tokens: the manifest's `tokens` bound unless a
    /// smaller one is asked for
    #[arg(long)]
    pub chunk: Option<u64>,

    /// Cap on concurrently running sequences per rank (≤ the manifest's
    /// `seqs` bound)
    #[arg(long, default_value_t = 256)]
    pub max_seqs: usize,

    /// Debug: launch every program eagerly, ignoring the manifest's `graph`
    #[arg(long)]
    pub eager: bool,

    /// Rows per sequence of a step: a shape some program of the manifest
    /// declares (1 for a plain step, its block for a speculative round).
    /// Default: the widest declared
    #[arg(long)]
    pub rows: Option<u64>,

    /// Pinned host memory (GiB) per rank, or per stage on a pipeline's head,
    /// for snapshots parked off the device: a lease short of pages or slots
    /// parks the coldest snapshot there instead of dropping it, and a
    /// prompt hitting one wakes it (0: off)
    #[arg(long, default_value_t = 0.0)]
    pub host_gib: f64,

    /// Tokens a parked snapshot adds to the prefix its session's other
    /// snapshots share: the host tier holds one slot per this many tokens
    /// of pages. The default fits the AgentX traces on qwen3.8 and full K3
    #[arg(long, default_value_t = 8192)]
    pub host_turn: u64,

    /// Serve as the head of several processes: HTTP, scheduler and every
    /// process's accounting, waiting at this host:port for the others to
    /// join — the later stages of a cut manifest (`kern cut`), or ranks
    /// 1.. of a manifest whose topology spans several GPUs
    #[arg(long, conflicts_with = "head")]
    pub listen: Option<String>,

    /// Join the head at host:port: as a later stage of a cut manifest, or
    /// as rank `--rank` of a manifest with a topology
    #[arg(long)]
    pub head: Option<String>,

    /// This process's rank with `--head`, 1.. of the manifest's ranks
    /// (rank 0 is the head; `tp` groups are consecutive ranks)
    #[arg(long, requires = "head")]
    pub rank: Option<usize>,

    /// On the head of a pipeline: hand every prompt of more than one token
    /// to a decode tier whose head listens at host:port (`--decode-listen`)
    /// once prefilled, and relay what it generates; repeated, each request
    /// goes to one of the tiers
    #[arg(long, requires = "listen")]
    pub decode: Vec<String>,

    /// Serve as a decode tier: no HTTP; wait at this host:port for the one
    /// pipeline head (`--decode`) whose prompts it decodes. Its other ranks,
    /// if any, join at `--listen` as usual. `--max-seqs` is the seats a
    /// group holds, reserved and running
    #[arg(long, conflicts_with = "head")]
    pub decode_listen: Option<String>,
}

pub fn serve(o: ServeOpts, art: Artifacts) -> Result<()> {
    let weights = kern_run::Weights::parse(&art.weights)?;
    info!(
        manifest = %art.manifest.display(),
        kernels = ?art.kernels,
        %weights,
        kern = %*kern_run::VERSION,
        "loading"
    );
    let manifest_json = std::fs::read_to_string(&art.manifest)
        .with_context(|| format!("reading manifest {}", art.manifest.display()))?;
    let manifest =
        Verified::from_json(&manifest_json).with_context(|| format!("manifest {}", art.manifest.display()))?;
    log_manifest(&manifest);
    let model_path = weights.dirs().remove(0);
    let served_name = o.served_model_name.clone().unwrap_or_else(|| manifest.model.clone());

    if manifest.cut.is_some() {
        anyhow::ensure!(o.rank.is_none(), "a pipeline stage's place is its cut manifest's; `--rank` is a topology's");
        anyhow::ensure!(
            o.decode_listen.is_none(),
            "a decode tier serves an uncut manifest; this one is a pipeline stage"
        );
        if let Some(head) = &o.head {
            let stage = pp::stage::Stage {
                head: head.clone(),
                gpu: o.gpu,
                kernels: art.kernels.clone(),
                capacity: o.capacity,
                max_seqs: o.max_seqs,
                eager: o.eager,
            };
            return pp::stage::run(manifest, &weights, stage);
        }
        let listen =
            o.listen.clone().context("a pipeline stage runs with `--listen` (stage 0) or `--head` (the others)")?;
        let head = pp::head::Head {
            listen,
            gpu: o.gpu,
            kernels: art.kernels.clone(),
            capacity: o.capacity,
            chunk: o.chunk,
            max_seqs: o.max_seqs,
            eager: o.eager,
            decode: o.decode.clone(),
            host_bytes: (o.host_gib * (1u64 << 30) as f64) as u64,
            host_turn: o.host_turn,
        };
        let load = move || pp::head::spawn(manifest, weights, head);
        return launch(load, served_name, model_path, o.port);
    }

    // Every sequence of a group batch holds a token slot on each of its
    // `t` ranks, and each rank its pad. A replicated group's batch is
    // `--max-seqs` rows on every member, not `t` members' worth.
    let replicated = Protocol::check(&manifest)?.shards.positions.is_some();
    let t = if replicated { 1 } else { manifest.group_size("tp").unwrap_or(1) as usize };
    let load = executor::Load {
        gpu: o.gpu,
        kernels: art.kernels.clone(),
        capacity: Capacity { tokens: o.capacity, seqs: ((o.max_seqs + 1) * t) as u64, landings: 0 },
        eager: o.eager,
    };
    if let Some(head) = &o.head {
        let q = o.rank.context("`--head` with a topology manifest joins as `--rank <q>`")?;
        return executor::rank(&manifest, &weights, &load, head, q);
    }
    anyhow::ensure!(o.decode.is_empty(), "`--decode` hands prompts off from a pipeline head; this manifest is not cut");
    if let Some(at) = &o.decode_listen {
        let (ranks, exec) = executor::head(&manifest, &weights, &load, o.listen.as_deref(), (0, 0))?;
        let pools = exec.pools()?;
        return pd::tier::serve(ranks, exec, pools, pd::shapes(&manifest), at, o.max_seqs);
    }
    let host = ((o.host_gib * (1u64 << 30) as f64) as u64, o.host_turn);
    let policy = Policy { chunk: o.chunk.map(|c| c as usize), max_seqs: o.max_seqs, rows: o.rows, host_bytes: host.0 };
    let listen = o.listen.clone();
    let load = move || -> Result<(KernScheduler, scheduler::Facts)> {
        let t0 = Instant::now();
        let (ranks, exec) = executor::head(&manifest, &weights, &load, listen.as_deref(), host)?;
        info!(model = %manifest.model, ranks = ranks.groups().ranks(), load_s = logline::secs(t0.elapsed()), "ranks loaded");
        let sched = KernScheduler::new(ranks, exec, policy)?;
        let facts = sched.facts();
        Ok((sched, facts))
    };
    launch(load, served_name, model_path, o.port)
}

/// Logs the verified manifest before any GPU work: buffer kinds, module count, and each program's calls.
fn log_manifest(m: &Verified) {
    let mut kinds: BTreeMap<String, usize> = BTreeMap::new();
    for b in m.buffers.values() {
        *kinds.entry(b.kind.to_string()).or_default() += 1;
    }
    let kinds = kinds.iter().map(|(k, n)| format!("{k}={n}")).collect::<Vec<_>>().join(" ");
    let registry =
        m.modules.values().filter(|md| md.source.starts_with("hf:") || md.source.starts_with("https:")).count();
    info!(
        buffers = m.buffers.len(),
        kinds = %kinds,
        modules = m.modules.len(),
        registry,
        states = m.states.len(),
        ops = m.ops.len(),
        "manifest"
    );
    for (name, p) in &m.programs {
        info!(program = %name, calls = p.calls.len(), shape = %program_shape(p), "program");
    }
}

fn program_shape(p: &kern_manifest::types::Program) -> String {
    let mut parts = Vec::new();
    if let Some(b) = &p.batch {
        let rows = match &b.rows {
            kern_manifest::types::Dim::Const(n) => n.to_string(),
            kern_manifest::types::Dim::Var(v) => v.clone(),
        };
        parts.push(format!("batch groups={} rows={rows}", b.groups));
    }
    if p.once {
        parts.push("once".into());
    }
    if p.derive {
        parts.push("derive".into());
    }
    if p.graph {
        parts.push("graph".into());
    }
    parts.join(" ")
}

/// Run the scheduler `load` builds on a thread of its own, which owns it
/// (and its GPUs) for its whole life, and serve HTTP once it is ready.
fn launch<S, L>(load: L, served_name: String, model_path: PathBuf, port: u16) -> Result<()>
where
    S: Scheduler + 'static,
    L: FnOnce() -> Result<(S, scheduler::Facts)> + Send + 'static,
{
    let (handle, backend) = scheduler_pair();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<scheduler::Facts>>();
    let join = std::thread::Builder::new()
        .name("kern-scheduler".into())
        .spawn(move || match load() {
            Ok((sched, facts)) => {
                let _ = ready_tx.send(Ok(facts));
                // A scheduler that panicked would leave the port open
                // and every request hanging: the process goes with it.
                if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drive(sched, backend))).is_err() {
                    std::process::exit(101);
                }
            }
            Err(e) => {
                let _ = ready_tx.send(Err(e));
            }
        })
        .context("spawning the scheduler thread")?;

    // The port opens once the engine is up: `serving` follows readiness.
    let (model, model_dir) = (served_name.clone(), model_path.clone());
    let engine = async move {
        let facts = tokio::task::spawn_blocking(move || ready_rx.recv())
            .await
            .context("scheduler thread died before reporting readiness")?
            .context("scheduler thread died before reporting readiness")??;
        info!(model = %model, port, model_dir = %model_dir.display(), "serving");
        Ok(LaunchedEngine::Stepped(Engine {
            schedulers: vec![LiveScheduler { handle, join }],
            info: EngineInfo {
                kv_capacity: Some(KvCapacity { total_blocks: facts.total_blocks, block_size: facts.block_size }),
                servable_len: Some(facts.max_request_tokens as u32),
            },
            lora: None,
        }))
    };

    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    rt.block_on(async move {
        // Needs the runtime: it spawns the signal listener.
        let shutdown = vllm::shutdown_token_from_ctrl_c();
        vllm::serve_with_engine_count(
            engine,
            &model_path,
            vec![served_name],
            ParserSelection::default(),
            port,
            None,
            1,
            shutdown,
        )
        .await
    })
}
