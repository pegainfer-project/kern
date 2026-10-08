//! `kern-serve`: OpenAI-compatible HTTP over a manifest.
//!
//! `--manifest`, `--kernels` and `--weights` name the artifact, the same
//! three flags `kern run` takes; the first weights entry's checkpoint
//! directory is also what the frontend reads (config, tokenizer, chat
//! template, stop tokens). A server names every input on its command line; `kern.toml`
//! is the kernel-development loop's file and this binary never reads it.

#![deny(unsafe_code)]

use std::path::PathBuf;
use std::sync::LazyLock;

use anyhow::Result;
use clap::Parser;

/// What `--version` prints: this server's crate version and the runtime it was
/// built with, the two facts a bug report from a server needs.
static VERSION: LazyLock<String> =
    LazyLock::new(|| format!("{} (runtime {})", env!("CARGO_PKG_VERSION"), kern_runtime::ID));

#[derive(Parser)]
#[command(name = "kern-serve", version = VERSION.as_str(), about = "serve a kern manifest over an OpenAI-compatible HTTP endpoint")]
struct Cli {
    /// Manifest JSON (must pass verification)
    #[arg(long)]
    manifest: PathBuf,
    /// Directory of cubins, resolved by their pinned sha256; a manifest of
    /// registry refs needs none
    #[arg(long)]
    kernels: Option<PathBuf>,
    /// Checkpoint directories or .safetensors files, one flag each (a
    /// manifest with a topology may write `{ep}` / `{tp}` and `*` for the
    /// rank's shard)
    #[arg(long, required = true)]
    weights: Vec<String>,
    #[command(flatten)]
    opts: kern_serve::ServeOpts,
}

fn main() -> Result<()> {
    check_runtime()?;
    kern_serve::logline::init();
    let cli = Cli::parse();
    let art = kern_serve::Artifacts { manifest: cli.manifest, kernels: cli.kernels, weights: cli.weights };
    kern_serve::serve(cli.opts, art)
}

/// `kern server` hands over the runtime it was built with. A server that was
/// not rebuilt after the runtime changed would serve the older one, which is
/// how a gate run came to panic half way through (docs/lessons.md); the
/// launcher cannot see this alone, so the check is here, before any work.
fn check_runtime() -> Result<()> {
    let Some(launcher) = std::env::var_os(kern_runtime::ID_VAR) else {
        return Ok(());
    };
    let launcher = launcher.to_string_lossy();
    anyhow::ensure!(
        launcher == kern_runtime::ID,
        "kern runs runtime {launcher} and this kern-serve was built with {}: rebuild it with \
         `cargo build --manifest-path crates/kern-serve/Cargo.toml`",
        kern_runtime::ID
    );
    Ok(())
}
