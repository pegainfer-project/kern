//! `kern-serve`: an OpenAI-compatible HTTP endpoint over a kern manifest.
//!
//! The HTTP/protocol stack is infront (vLLM's Rust server crates
//! underneath: completions, chat completions, streaming, chat templates,
//! stop strings). This crate contributes the engine behind it:
//! `scheduler::KernScheduler`, infront's `Scheduler` contract over
//! [`ranks::Ranks`] — every rank's accounting, kept by the head — and
//! `executor::Executors`, one process per GPU running what it decided.
//! A cut manifest is served by `pp` instead, one process per stage, its
//! head behind a [`front::Front`], and may hand each prompt to a decode
//! tier ([`pd`]). What the frontend would admit but no manifest can serve
//! (logprobs; a draw from a manifest without sampling fills) is declared to
//! infront, which refuses it at the HTTP layer. The crate's public surface
//! is [`serve`], its option structs, and the accounting, protocols and
//! packing policy its tests drive.

#![deny(unsafe_code)]

mod dp;
mod executor;
pub mod front;
pub mod logline;
pub mod pd;
mod pool;
mod pp;
pub mod ranks;
pub mod scheduler;
mod watch;

pub use pp::pack;

use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Args;
use infront_engine::{drive, scheduler_pair, Engine, EngineInfo, KvCapacity, LiveScheduler, Scheduler};
use infront_vllm::{Capabilities, ParserSelection, ServeOptions};
use kern_manifest::{Protocol, Verified};
use kern_runtime::Capacity;
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

    /// Tool-call output parser, by the frontend's registry name
    /// (`qwen3_coder`, `hermes`, …) or `none`; `auto` takes the manifest's
    /// `chat.tool_parser`, and without one matches the model directory's name
    #[arg(long, default_value_t = ParserSelection::Auto)]
    pub tool_call_parser: ParserSelection,

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

    /// Per-sequence state slots per rank: the running sequences' plus the
    /// kept prefixes' (default: `--max-seqs` plus one)
    #[arg(long)]
    pub seq_slots: Option<usize>,

    /// Debug: launch every program eagerly, ignoring the manifest's `graph`
    #[arg(long)]
    pub eager: bool,

    /// Rows per sequence of a step: a shape some program of the manifest
    /// declares (1 for a plain step, its block for a speculative round).
    /// Default: the widest declared
    #[arg(long)]
    pub rows: Option<u64>,

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

    #[command(flatten)]
    pub pool: PoolOpts,
}

/// The pooled KV tier (Mooncake Store): every rank joins it when
/// `--pool-master` is given, and every snapshot the prefix tree keeps is
/// shared with every other instance of the same manifest and weights.
#[derive(Args, Debug, Clone)]
pub struct PoolOpts {
    /// Join the pooled KV tier whose Mooncake master is at host:port
    #[arg(long)]
    pub pool_master: Option<String>,

    /// The pool's transfer-engine metadata server
    /// (`http://host:port/metadata`, or `P2PHANDSHAKE`)
    #[arg(long, default_value = "P2PHANDSHAKE", requires = "pool_master")]
    pub pool_metadata: String,

    /// The RDMA device this rank moves pool bytes over
    #[arg(long, requires = "pool_master")]
    pub pool_device: Option<String>,

    /// This process's name in the pool, host:port; default: the hostname
    /// and a port from the GPU index
    #[arg(long, requires = "pool_master")]
    pub pool_hostname: Option<String>,

    /// The segment a put prefers (this tray's holder), by its name
    #[arg(long, requires = "pool_master")]
    pub pool_segment: Option<String>,

    /// `libmooncake_store.so`, when not on the loader's path
    #[arg(long, requires = "pool_master")]
    pub pool_library: Option<PathBuf>,

    /// Seconds every pool act fails outright after a client error
    #[arg(long, default_value_t = 30, requires = "pool_master")]
    pub pool_cooldown: u64,
}

impl PoolOpts {
    fn config(&self, gpu: usize) -> Result<Option<pool::Config>> {
        let Some(master) = &self.pool_master else { return Ok(None) };
        let device = self.pool_device.clone().context("`--pool-master` takes `--pool-device <rdma device>`")?;
        let hostname = match &self.pool_hostname {
            Some(h) => h.clone(),
            None => {
                let host = std::fs::read_to_string("/proc/sys/kernel/hostname").context("reading the hostname")?;
                format!("{}:{}", host.trim(), 17100 + gpu)
            }
        };
        Ok(Some(pool::Config {
            master: master.clone(),
            metadata: self.pool_metadata.clone(),
            device,
            hostname,
            segment: self.pool_segment.clone(),
            library: self.pool_library.clone(),
            cooldown: std::time::Duration::from_secs(self.pool_cooldown),
        }))
    }
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
    let samples = Protocol::check_unsampled(&manifest)?.samples();
    let parser = tool_parser(o.tool_call_parser.clone(), manifest.chat.as_ref().and_then(|c| c.tool_parser.as_deref()));
    info!(
        tool_call_parser = %parser,
        resolved = parser.resolve_tool_name(&model_path.to_string_lossy()).unwrap_or("none"),
        "chat"
    );

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
                pool: o.pool.config(o.gpu)?,
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
            pool: o.pool.config(o.gpu)?,
        };
        let load = move || pp::head::spawn(manifest, weights, head);
        return launch(load, served_name, model_path, o.port, samples, parser);
    }

    // Every sequence of a group batch holds a state slot on each of its
    // `t` ranks, and each rank its pad. A replicated group's batch is
    // `--max-seqs` rows on every member, not `t` members' worth.
    let replicated = Protocol::check(&manifest)?.shards.positions.is_some();
    let t = if replicated { 1 } else { manifest.group_size("tp").unwrap_or(1) as u64 };
    let load = executor::Load {
        gpu: o.gpu,
        kernels: art.kernels.clone(),
        capacity: Capacity { tokens: o.capacity, slots: o.seq_slots.map_or(slots_for(o.max_seqs), |n| (n as u64).max(slots_for(o.max_seqs))) * t, landings: 0 },
        eager: o.eager,
        pool: o.pool.config(o.gpu)?,
    };
    if let Some(head) = &o.head {
        let q = o.rank.context("`--head` with a topology manifest joins as `--rank <q>`")?;
        return executor::rank(&manifest, &weights, &load, head, q);
    }
    anyhow::ensure!(o.decode.is_empty(), "`--decode` hands prompts off from a pipeline head; this manifest is not cut");
    if let Some(at) = &o.decode_listen {
        let (ranks, exec) = executor::head(&manifest, &weights, &load, o.listen.as_deref())?.ranks()?;
        let pools = exec.pools()?;
        return pd::tier::serve(ranks, exec, pools, pd::shapes(&manifest), at, o.max_seqs);
    }
    let listen = o.listen.clone();
    // A manifest with nothing to step with is a prefill tier: every rank
    // its own batch (`dp`); any other runs the group scheduler.
    if dp::fits(&Protocol::check(&manifest)?) {
        let policy = dp::Policy { chunk: o.chunk.map(|c| c as usize), max_seqs: o.max_seqs };
        let load = move || -> Result<(dp::Dp, scheduler::Facts)> {
            let t0 = Instant::now();
            let joined = executor::head(&manifest, &weights, &load, listen.as_deref())?;
            info!(model = %manifest.model, ranks = joined.groups.ranks(), load_s = logline::secs(t0.elapsed()), "ranks loaded");
            let sched = dp::Dp::new(joined, policy)?;
            let facts = sched.facts();
            Ok((sched, facts))
        };
        return launch(load, served_name, model_path, o.port, samples, parser);
    }
    let policy = Policy { chunk: o.chunk.map(|c| c as usize), max_seqs: o.max_seqs, rows: o.rows };
    let load = move || -> Result<(KernScheduler, scheduler::Facts)> {
        let t0 = Instant::now();
        let joined = executor::head(&manifest, &weights, &load, listen.as_deref())?;
        let keys = joined.keys.clone();
        let (ranks, exec) = joined.ranks()?;
        info!(model = %manifest.model, ranks = ranks.groups().ranks(), load_s = logline::secs(t0.elapsed()), "ranks loaded");
        let sched = KernScheduler::new(ranks, exec, policy, keys)?;
        let facts = sched.facts();
        Ok((sched, facts))
    };
    launch(load, served_name, model_path, o.port, samples, parser)
}

/// State slots a rank holds for `seats` running sequences: the pad's and
/// one per seat. A slot is a running request's resource; a checkpoint a
/// request's end leaves for its next turn keeps one only while no seat
/// wants it. The slots are fixed at load; past them a lease waits for
/// room (`Busy`), and the scheduler makes it by parking or dropping a
/// kept prefix.
fn slots_for(seats: usize) -> u64 {
    seats as u64 + 1
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

/// The tool-call parser the frontend runs: the operator's flag when it
/// says more than `auto`, else the manifest's `chat.tool_parser`, else
/// `auto`, which the frontend matches to the model directory's name.
fn tool_parser(flag: ParserSelection, declared: Option<&str>) -> ParserSelection {
    match (flag, declared) {
        (ParserSelection::Auto, Some(name)) => ParserSelection::Explicit(name.into()),
        (flag, _) => flag,
    }
}

/// Run the scheduler `load` builds on a thread of its own, which owns it
/// (and its GPUs) for its whole life, and serve HTTP once it is ready.
/// Serve `load`'s scheduler on `port`; `samples` is whether its manifest
/// draws, so the endpoint can refuse a request for a draw it cannot make;
/// `parser` is what the frontend reads tool calls with.
fn launch<S, L>(
    load: L,
    served_name: String,
    model_path: PathBuf,
    port: u16,
    samples: bool,
    parser: ParserSelection,
) -> Result<()>
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
        let block_size = NonZeroUsize::new(facts.block_size).context("the pool's block size is zero")?;
        Ok(Engine {
            schedulers: vec![LiveScheduler { handle, join }],
            info: EngineInfo {
                kv_capacity: Some(KvCapacity { total_blocks: facts.total_blocks, block_size }),
                servable_len: Some(facts.max_request_tokens as u32),
            },
        })
    };

    // A manifest hands back token ids, never logits; one without sampling
    // fills draws nothing. The state cannot roll back, so every chat
    // request carries where its history ends.
    let options = ServeOptions {
        served_model_names: vec![served_name],
        tool_call_parser: parser,
        port,
        capabilities: Capabilities { logprobs: false, sampling: samples },
        history_boundary: true,
        ..ServeOptions::new(model_path.to_string_lossy())
    };
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    // The ctrl-c token needs the runtime: it spawns the signal listener.
    rt.block_on(async move { infront_vllm::serve(engine, options, infront_vllm::shutdown_token_from_ctrl_c()).await })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_flag_wins_over_the_manifest_and_the_manifest_over_auto() {
        let explicit = |s: &str| ParserSelection::Explicit(s.into());
        assert_eq!(tool_parser(ParserSelection::Auto, Some("qwen3_coder")), explicit("qwen3_coder"));
        assert_eq!(tool_parser(explicit("hermes"), Some("qwen3_coder")), explicit("hermes"));
        assert_eq!(tool_parser(ParserSelection::None, Some("qwen3_coder")), ParserSelection::None);
        assert_eq!(tool_parser(ParserSelection::Auto, None), ParserSelection::Auto);
    }
}
