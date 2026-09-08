//! `kern-serve`: OpenAI-compatible HTTP over a manifest.
//!
//! Same lookup as `kern run`: `--manifest`, `--kernels` and `--weights`
//! name the artifact, or a target in the nearest `kern.toml` does;
//! `--model-path` is the HF directory the frontend reads (tokenizer, chat
//! template, stop tokens). The API serves the model under the target's
//! name when there is one.

#![deny(unsafe_code)]

use std::path::PathBuf;

use anyhow::{anyhow, ensure, Result};
use clap::Parser;
use kern_run::config::Config;

#[derive(Parser)]
#[command(name = "kern-serve", version, about = "serve a kern manifest over an OpenAI-compatible HTTP endpoint")]
struct Cli {
    /// kern.toml to use (default: the nearest one at or above the cwd)
    #[arg(long)]
    config: Option<PathBuf>,
    /// Target in kern.toml (needed when it declares several)
    target: Option<String>,
    /// Manifest JSON (default: the target's)
    #[arg(long)]
    manifest: Option<PathBuf>,
    /// Directory of cubins, resolved by their pinned sha256 (default: the target's)
    #[arg(long)]
    kernels: Option<PathBuf>,
    /// Checkpoint directories or .safetensors files (default: the target's)
    #[arg(long)]
    weights: Vec<PathBuf>,
    #[command(flatten)]
    opts: kern_serve::ServeOpts,
}

fn main() -> Result<()> {
    kern_serve::logline::init();
    let cli = Cli::parse();
    let cfg = Config::find(cli.config.as_deref())?.filter(|c| !c.targets.is_empty());
    // A target is consulted when named, or when the flags do not already
    // say which manifest to serve.
    let target = match &cfg {
        Some(c) if cli.target.is_some() || cli.manifest.is_none() => Some(c.one(cli.target.as_deref())?),
        _ => None,
    };
    let need = |what: &str| anyhow!("no --{what} and no kern.toml target to take it from");
    let manifest = cli.manifest.or_else(|| target.map(|(_, t)| t.manifest.clone())).ok_or_else(|| need("manifest"))?;
    let kernels = cli.kernels.or_else(|| target.map(|(_, t)| t.kernels.clone())).ok_or_else(|| need("kernels"))?;
    let weights =
        if cli.weights.is_empty() { target.map(|(_, t)| t.weights.clone()).unwrap_or_default() } else { cli.weights };
    ensure!(!weights.is_empty(), "{}", need("weights"));
    let mut opts = cli.opts;
    if opts.served_model_name.is_none() {
        opts.served_model_name = target.map(|(name, _)| name.clone());
    }
    let d = cfg
        .as_ref()
        .map(|c| kern_serve::Defaults { gpu: c.gpu, capacity: c.capacity, chunk: c.run.chunk })
        .unwrap_or_default();
    kern_serve::serve(opts, kern_serve::Artifacts { manifest, kernels, weights }, d)
}
