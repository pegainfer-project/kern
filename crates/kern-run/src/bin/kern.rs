//! `kern`: run a manifest, test a kernel swap, gather cubins, or say
//! what a manifest declares without a GPU.
//!
//! Inputs come from flags, else from the nearest `kern.toml` (see
//! `kern_run::config`). Targets are names the user picks there.

#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, ensure, Context, Result};
use clap::{Parser, Subcommand};
use kern_run::attest::TestOpts;
use kern_run::config::{Config, Target};
use kern_run::run::RunOpts;

#[derive(Parser)]
#[command(
    name = "kern",
    version,
    about = "model-agnostic GPU runtime: run a manifest, test a kernel swap, gather cubins, verify a manifest"
)]
struct Cli {
    /// kern.toml to use (default: the nearest one at or above the cwd)
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Greedy bs=1 generation over a target's manifest
    Run {
        /// Target in kern.toml (needed when it declares several)
        target: Option<String>,
        #[command(flatten)]
        opts: RunOpts,
    },
    /// A/B a kernel swap — each target's `reference` (A) against its
    /// `manifest` (B); all targets when none is named. Exit 0 PASS, 1 FAIL,
    /// 2 INCONCLUSIVE (the worst over the targets)
    Test {
        targets: Vec<String>,
        #[command(flatten)]
        opts: TestOpts,
    },
    /// Build the handwritten cubins (`[kernels].sources`) and land every
    /// cubin pinned by each target's manifest and reference into its
    /// kernels dir, from `[kernels].dumps` and the builds
    Kernels { targets: Vec<String> },
    /// Verify a manifest and print its serving protocol: the axes, every
    /// fill, the tables, and the call shape each program accepts. No GPU.
    /// Exit 1 when verification or the protocol fails
    Verify {
        /// Manifest JSON, or a target's when none is given
        manifest: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .with_target(false)
        .without_time()
        .init();
    let cli = Cli::parse();
    let cfg = Config::find(cli.config.as_deref())?;
    match cli.cmd {
        Cmd::Run { target, opts } => {
            let t = match &cfg {
                Some(c) if !c.targets.is_empty() => Some(c.one(target.as_deref())?.1),
                _ => {
                    ensure!(
                        target.is_none(),
                        "no kern.toml with targets found; `{}` cannot be looked up",
                        target.unwrap_or_default()
                    );
                    None
                }
            };
            kern_run::run::run(opts, cfg.as_ref(), t)
        }
        Cmd::Test { targets, opts } => {
            let sel: Vec<(Option<&String>, Option<&Target>)> = match &cfg {
                Some(c) if !c.targets.is_empty() => {
                    c.select(&targets)?.into_iter().map(|(n, t)| (Some(n), Some(t))).collect()
                }
                _ => {
                    ensure!(targets.is_empty(), "no kern.toml with targets found; {targets:?} cannot be looked up");
                    vec![(None, None)]
                }
            };
            let multi = sel.len() > 1;
            let mut worst = 0;
            for (name, t) in sel {
                let mut o = opts.clone();
                if multi {
                    let name = name.unwrap();
                    if let Some(dir) = &opts.out {
                        std::fs::create_dir_all(dir)?;
                        o.out = Some(dir.join(format!("{name}.json")));
                    }
                    println!("\n════ {name} ════\n");
                }
                worst = worst.max(kern_run::attest::run(o, cfg.as_ref(), t)?);
            }
            if worst != 0 {
                std::process::exit(worst);
            }
            Ok(())
        }
        Cmd::Kernels { targets } => kernels(cfg.as_ref(), &targets),
        Cmd::Verify { manifest } => {
            let path = match (manifest, &cfg) {
                (Some(p), _) => p,
                (None, Some(c)) if !c.targets.is_empty() => c.one(None)?.1.manifest.clone(),
                _ => bail!("kern verify needs a manifest path or a kern.toml target"),
            };
            if !verify(&path)? {
                std::process::exit(1);
            }
            Ok(())
        }
    }
}

/// `kern verify`: verification, then the protocol, both reported in full.
fn verify(path: &Path) -> Result<bool> {
    use kern_manifest::protocol::{Axis, Rows};
    let json = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let m = match kern_manifest::Verified::from_json(&json) {
        Ok(m) => m,
        Err(e) => {
            println!("{}: {e}", path.display());
            return Ok(false);
        }
    };
    println!("{}: `{}` schema v{}, verified", path.display(), m.model, m.schema_version);
    let p = match kern_manifest::Protocol::check(&m) {
        Ok(p) => p,
        Err(e) => {
            println!("{e}");
            return Ok(false);
        }
    };
    let axis = |a: Axis| match a {
        Axis::Rows => format!("[{}]", p.rows.var),
        Axis::Groups => format!("[{}]", p.groups.var),
        Axis::Tray => format!("[{}]", p.tray.as_ref().map_or("tray", |t| t.var.as_str())),
        Axis::Fixed(n) => format!("[{n}]"),
    };
    println!("  rows      `{}` <= {}", p.rows.var, p.rows.max);
    println!("  groups    `{}` <= {}", p.groups.var, p.groups.max);
    if let Some(t) = &p.tray {
        println!("  tray      `{}` <= {}", t.var, t.max);
    }
    for f in &p.fills {
        let w = if f.width > 1 { format!(" x {}", f.width) } else { String::new() };
        println!("  fill      {:<11} `{}` {} {}{w}", f.fill.to_string(), f.name, f.dtype, axis(f.axis));
    }
    for t in &p.page_tables {
        println!("  pages     `{}` [{}, {}]", t.name, p.groups.var, t.width);
    }
    for t in &p.line_tables {
        let w = if t.width > 1 { format!(", {}", t.width) } else { String::new() };
        println!("  lines     `{}` [{}, {}{w}]", t.name, t.lines, axis(t.axis).trim_matches(['[', ']']));
    }
    for f in &p.forwards {
        let rows = match f.rows {
            Rows::Const(r) => format!("{r} rows"),
            Rows::Var => "rows as fed".into(),
        };
        let emits = match f.emits {
            Some(i) => format!(", hands back `{}`", p.fills[i].name),
            None => ", state only".into(),
        };
        let count = match f.count {
            Some(i) => format!(" counted by `{}`", p.fills[i].name),
            None => String::new(),
        };
        println!("  forward   `{}`: <= {} sequences of {rows}{emits}{count}", f.name, f.groups);
    }
    for o in &p.once {
        println!("  once      `{o}`");
    }
    Ok(true)
}

/// `kern kernels`: the two tools scripts, driven from kern.toml.
fn kernels(cfg: Option<&Config>, targets: &[String]) -> Result<()> {
    let Some(cfg) = cfg else { bail!("kern kernels needs a kern.toml ([targets], [kernels])") };
    let tools = tools_dir(cfg)?;
    if let Some(src) = &cfg.kernels.sources {
        sh(Command::new(tools.join("build_kernels.sh")).env("KERN_SRC", src))?;
    }
    let dumps: Vec<String> = cfg.kernels.dumps.iter().map(|p| p.display().to_string()).collect();
    for (name, t) in cfg.select(targets)? {
        for m in std::iter::once(&t.manifest).chain(t.reference.iter()) {
            eprintln!("{name}: {} → {}", m.display(), t.kernels.display());
            // the script wants at least one search dir; the kernels dir itself is harmless
            let d = if dumps.is_empty() { t.kernels.display().to_string() } else { dumps.join(":") };
            sh(Command::new(tools.join("extract_kernels.sh")).arg(m).arg(&d).arg(&t.kernels))?;
        }
    }
    Ok(())
}

/// The repo's `tools/`: next to the kern.toml or any directory above it,
/// else next to this binary's `target/` (never the build-time path — the
/// binary is built in a container).
fn tools_dir(cfg: &Config) -> Result<PathBuf> {
    let mut cands: Vec<PathBuf> = cfg.dir().ancestors().map(|d| d.join("tools")).collect();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(root) = exe.parent().and_then(Path::parent).and_then(Path::parent) {
            cands.push(root.join("tools"));
        }
    }
    cands.into_iter().find(|d| d.join("extract_kernels.sh").is_file()).ok_or_else(|| {
        anyhow::anyhow!("no tools/extract_kernels.sh above {} or next to the binary", cfg.path.display())
    })
}

fn sh(cmd: &mut Command) -> Result<()> {
    let st = cmd.status().with_context(|| format!("running {cmd:?}"))?;
    ensure!(st.success(), "{cmd:?} exited with {st}");
    Ok(())
}
