//! `kern-serve`: an OpenAI-compatible HTTP endpoint over a kern manifest.
//!
//! The HTTP/protocol stack is pegainfer's frontend (vLLM's Rust server
//! crates underneath: completions, chat completions, streaming, chat
//! templates, stop strings). This crate contributes the engine behind it:
//! `scheduler::KernScheduler`, the pegainfer `Scheduler` contract over a
//! `tray::Tray` — one `kern_runtime::Runtime` per GPU, driven in lockstep
//! (KV pages are the runtimes' leases). The crate's public surface is
//! [`serve`] and its option structs.

#![deny(unsafe_code)]

pub mod logline;
mod scheduler;
mod tray;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::Args;
use kern_manifest::Verified;
use kern_runtime::{Capacity, Topology};
use pegainfer_frontend::engine::{
    drive, scheduler_pair, Engine, EngineInfo, KvCapacity, LaunchedEngine, LiveScheduler,
};
use pegainfer_frontend::vllm;
use tracing::info;

use scheduler::{KernScheduler, Policy};
use tray::Tray;

/// The manifest and its artifacts (from kern.toml's target or flags).
pub struct Artifacts {
    pub manifest: PathBuf,
    pub kernels: PathBuf,
    /// Weight files; see [`rank_weights`] for `{group}` and `*`.
    pub weights: Vec<PathBuf>,
}

/// Defaults a kern.toml may supply.
#[derive(Default)]
pub struct Defaults {
    pub gpu: Option<usize>,
    pub capacity: Option<u64>,
    pub chunk: Option<u64>,
}

#[derive(Args, Debug, Clone)]
pub struct ServeOpts {
    /// HF-layout model directory for the frontend: config.json, tokenizer,
    /// chat template, generation_config (stop tokens)
    #[arg(long)]
    pub model_path: PathBuf,

    /// Model id served by the API (default: the manifest's `model`)
    #[arg(long)]
    pub served_model_name: Option<String>,

    #[arg(long, default_value_t = 8000)]
    pub port: u16,

    /// CUDA device ordinals, one per rank of the tray, in rank order (a
    /// manifest with a topology needs every group's members; `tp` groups
    /// are consecutive ranks). Default: kern.toml's `gpu`, else 0
    #[arg(long, value_delimiter = ',')]
    pub gpus: Vec<usize>,

    /// KV pool in tokens per rank (rounded down to the page); every request
    /// reserves its worst case `prompt + max_tokens` at admission. Default:
    /// whatever device memory is left once weights, activations and scratch
    /// are allocated, less 1 GiB
    #[arg(long)]
    pub capacity: Option<u64>,

    /// Prefill chunk in tokens
    #[arg(long)]
    pub chunk: Option<u64>,

    /// Prompt tokens one step may prefill before it decodes
    #[arg(long, default_value_t = 2048)]
    pub prefill_budget: usize,

    /// Cap on concurrently running sequences per rank (≤ the manifest's
    /// `seqs` bound)
    #[arg(long, default_value_t = 256)]
    pub max_seqs: usize,

    /// Skip CUDA graph capture, launch every call eagerly
    #[arg(long)]
    pub eager: bool,

    /// Rows per sequence of a step: a shape some program of the manifest
    /// declares (1 for a plain step, its block for a speculative round).
    /// Default: the widest declared
    #[arg(long)]
    pub rows: Option<u64>,

    /// Extra stop token ids (generation_config.json's eos ids always apply)
    #[arg(long, value_delimiter = ',')]
    pub stop_tokens: Vec<u32>,

    /// Pinned host memory (GiB) per rank for snapshots parked off the
    /// device: a lease short of pages or slots parks the coldest snapshot
    /// there instead of dropping it, and a prompt hitting one wakes it
    /// (0: off)
    #[arg(long, default_value_t = 0.0)]
    pub host_gib: f64,
}

/// Stop tokens from the HF directory: generation_config.json `eos_token_id`
/// (int or list), else config.json's.
fn hf_stop_tokens(model_path: &Path) -> Vec<u32> {
    let read = |f: &str| -> Option<serde_json::Value> {
        serde_json::from_str(&std::fs::read_to_string(model_path.join(f)).ok()?).ok()
    };
    let ids = |v: &serde_json::Value| -> Vec<u32> {
        match v.get("eos_token_id") {
            Some(serde_json::Value::Number(n)) => n.as_u64().into_iter().map(|x| x as u32).collect(),
            Some(serde_json::Value::Array(a)) => a.iter().filter_map(|x| x.as_u64()).map(|x| x as u32).collect(),
            _ => Vec::new(),
        }
    };
    let mut out = read("generation_config.json").map(|v| ids(&v)).unwrap_or_default();
    if out.is_empty() {
        out = read("config.json").map(|v| ids(&v)).unwrap_or_default();
    }
    out
}

/// One rank's weight files from the target's list: every `{group}` in a
/// path is the rank's index in that topology group (`{ep}`, `{tp}`), and
/// a `*` in a file name matches that directory's files around it, in name
/// order. A manifest sharded per rank names its shards this way once for
/// every rank.
fn rank_weights(paths: &[PathBuf], topo: &Topology) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for p in paths {
        let mut s = p.to_string_lossy().into_owned();
        for (g, r) in &topo.groups {
            s = s.replace(&format!("{{{g}}}"), &r.index.to_string());
        }
        if let Some(open) = s.find('{') {
            let close = s[open..].find('}').map_or(s.len(), |c| open + c + 1);
            bail!("weights path {s}: `{}` is not a group of the manifest's topology", &s[open..close]);
        }
        let p = PathBuf::from(&s);
        let Some((pre, post)) = p.file_name().and_then(|f| f.to_str()).and_then(|f| f.split_once('*')) else {
            out.push(p);
            continue;
        };
        let dir = p.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(Path::new("."));
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .with_context(|| format!("weights {s}: listing {}", dir.display()))?
            .filter_map(|e| e.ok()?.file_name().into_string().ok())
            .filter(|f| f.len() >= pre.len() + post.len() && f.starts_with(pre) && f.ends_with(post))
            .collect();
        if names.is_empty() {
            bail!("weights {s}: nothing matches");
        }
        names.sort_unstable();
        out.extend(names.into_iter().map(|f| dir.join(f)));
    }
    Ok(out)
}

pub fn serve(o: ServeOpts, art: Artifacts, d: Defaults) -> Result<()> {
    let gpus = if o.gpus.is_empty() { vec![d.gpu.unwrap_or(0)] } else { o.gpus.clone() };
    let chunk = o.chunk.or(d.chunk).unwrap_or(512) as usize;
    let mut stop_tokens = hf_stop_tokens(&o.model_path);
    stop_tokens.extend(&o.stop_tokens);
    stop_tokens.sort_unstable();
    stop_tokens.dedup();
    anyhow::ensure!(
        !stop_tokens.is_empty(),
        "no stop tokens: none in {}/generation_config.json or config.json and no --stop-tokens",
        o.model_path.display()
    );
    info!(ids = ?stop_tokens, "stop tokens");

    let manifest_json = std::fs::read_to_string(&art.manifest)
        .with_context(|| format!("reading manifest {}", art.manifest.display()))?;
    let manifest =
        Verified::from_json(&manifest_json).with_context(|| format!("manifest {}", art.manifest.display()))?;
    let served_name = o.served_model_name.clone().unwrap_or_else(|| manifest.model.clone());
    // Every sequence of a tray batch group holds a token slot on each of
    // its `t` ranks, and each rank its pad.
    let t = manifest.group_size("tp").unwrap_or(1) as usize;
    let capacity = Capacity { tokens: o.capacity.or(d.capacity), seqs: ((o.max_seqs + 1) * t) as u64 };

    // The scheduler thread owns the tray for its whole life: load there,
    // report readiness, then drive.
    let (handle, backend) = scheduler_pair();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<scheduler::Facts>>();
    let host_bytes = (o.host_gib * (1u64 << 30) as f64) as u64;
    let policy = Policy {
        chunk,
        prefill_budget: o.prefill_budget,
        eager: o.eager,
        max_seqs: o.max_seqs,
        stop_tokens,
        rows: o.rows,
        host_bytes,
    };
    let join = std::thread::Builder::new()
        .name("kern-scheduler".into())
        .spawn(move || {
            let load = || -> Result<KernScheduler> {
                let t0 = Instant::now();
                let weights_of = |topo: &Topology| rank_weights(&art.weights, topo);
                let tray = Tray::load(&manifest, &art.kernels, &gpus, capacity, &weights_of, host_bytes)?;
                info!(model = %tray.manifest().model, gpus = ?gpus, load_s = logline::secs(t0.elapsed()), "tray loaded");
                KernScheduler::new(tray, policy)
            };
            match load() {
                Ok(sched) => {
                    let _ = ready_tx.send(Ok(sched.facts()));
                    drive(sched, backend);
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                }
            }
        })
        .context("spawning the scheduler thread")?;

    let engine = async move {
        let facts = tokio::task::spawn_blocking(move || ready_rx.recv())
            .await
            .context("scheduler thread died before reporting readiness")?
            .context("scheduler thread died before reporting readiness")??;
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
    info!(model = %served_name, port = o.port, model_dir = %o.model_path.display(), "serving");
    rt.block_on(async move {
        // Needs the runtime: it spawns the signal listener.
        let shutdown = vllm::shutdown_token_from_ctrl_c();
        vllm::serve_with_engine_count(engine, &o.model_path, vec![served_name], o.port, None, 1, shutdown).await
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use kern_runtime::GroupRank;

    #[test]
    fn rank_weights_substitute_groups_and_expand_stars() {
        let dir = std::env::temp_dir().join(format!("kern-serve-weights-{}", std::process::id()));
        let shard = dir.join("dense-tp4").join("r2");
        std::fs::create_dir_all(&shard).unwrap();
        for f in ["l10.safetensors", "l1.safetensors", "l0.safetensors", "notes.txt"] {
            std::fs::write(shard.join(f), b"").unwrap();
        }
        let mut topo = Topology::one("ep", 3, 4);
        topo.groups.insert("tp".into(), GroupRank { index: 2, size: 4 });
        let paths = [dir.join("bookends.safetensors"), dir.join("dense-tp4/r{tp}/l*.safetensors")];
        let got = rank_weights(&paths, &topo).unwrap();
        // Name order, not layer order: the runtime binds by tensor name.
        let want = [
            dir.join("bookends.safetensors"),
            shard.join("l0.safetensors"),
            shard.join("l1.safetensors"),
            shard.join("l10.safetensors"),
        ];
        assert_eq!(got, want);
        let e = rank_weights(&[dir.join("experts/ep{world}-r{ep}.safetensors")], &topo).unwrap_err();
        assert!(e.to_string().contains("`{world}` is not a group"), "{e}");
        let e = rank_weights(&[shard.join("x*.safetensors")], &topo).unwrap_err();
        assert!(e.to_string().contains("nothing matches"), "{e}");
    }
}
