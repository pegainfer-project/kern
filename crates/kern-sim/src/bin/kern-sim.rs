use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use kern_manifest::Verified;
use kern_sim::cluster::{ClusterConfig, EndpointConfig, SharedPoolConfig, SimCluster};
use kern_sim::cost::{CostBook, PriceOptions};
use kern_sim::memory::{DeviceMemorySpec, MemoryBudget, DEFAULT_DRIVER_OVERHEAD_BYTES};
use kern_sim::router::RouterPolicy;
use kern_sim::trace::{ArrivalMode, JsonlTraceOptions, RequestTrace, SyntheticSessionConfig};
use kern_sim::workload::{price_workload, Workload};
use serde_json::json;

#[derive(Parser)]
#[command(
    name = "kern-sim",
    version,
    about = "Pure-CPU cost repricing, VRAM budgeting, and cluster routing simulator for kern"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Reprice one or more candidate manifests across workloads using calibrated bench/trace data.
    Price(PriceCmd),
    /// Compute static VRAM breakdown and KV page / session-slot budgets without a GPU.
    Budget(BudgetCmd),
    /// Replay request/session traces across single- or multi-endpoint clusters and routers.
    Replay(ReplayCmd),
}

#[derive(clap::Args)]
struct PriceCmd {
    /// Calibration `<manifest.json>=<bench.json>` pair (repeatable).
    #[arg(long = "bench", value_name = "MANIFEST=BENCH")]
    benches: Vec<String>,

    /// Calibration `<manifest.json>=<steps.parquet>,<kernels.parquet>` triple (repeatable).
    #[arg(long = "parquet", value_name = "MANIFEST=STEPS,KERNELS")]
    parquets: Vec<String>,

    /// Candidate `[label=]manifest.json` to price (repeatable).
    #[arg(long = "manifest", required = true, value_name = "[LABEL=]PATH")]
    manifests: Vec<String>,

    /// Workload `.toml` file to price against (repeatable).
    #[arg(long = "workload", value_name = "TOML")]
    workloads: Vec<PathBuf>,

    /// Closed `steps.parquet` file or directory to convert into a workload (repeatable).
    #[arg(long = "steps-workload", value_name = "PATH")]
    steps_workloads: Vec<PathBuf>,

    /// Optional measured `label=bench.json` ground truth to display delta % against.
    #[arg(long = "actual", value_name = "LABEL=BENCH")]
    actuals: Vec<String>,

    /// Require strict cubin sha256 matches only (disable EntryAbi fallback).
    #[arg(long)]
    strict_sha: bool,

    /// Optional JSON output path.
    #[arg(long)]
    json: Option<PathBuf>,
}

#[derive(clap::Args)]
struct BudgetCmd {
    /// Candidate `[label=]manifest.json` to inspect (repeatable).
    #[arg(long = "manifest", required = true, value_name = "[LABEL=]PATH")]
    manifests: Vec<String>,

    /// Device total VRAM in GiB (e.g. 71.2, 80.0).
    #[arg(long, default_value_t = 80.0)]
    vram_gib: f64,

    /// Active sequence slots to evaluate (repeatable; defaults to manifest's `seq_slots - 1`).
    #[arg(long = "slots")]
    slots: Vec<u64>,

    /// Optional 1-point driver overhead calibration `<manifest.json>:<raw_pages>[:<active_slots>]`.
    #[arg(long)]
    calibrate: Option<String>,

    /// Optional JSON output path.
    #[arg(long)]
    json: Option<PathBuf>,
}

#[derive(clap::Args)]
struct ReplayCmd {
    /// Calibration `<manifest.json>=<bench.json>` pair (repeatable).
    #[arg(long = "bench", value_name = "MANIFEST=BENCH")]
    benches: Vec<String>,

    /// Calibration `<manifest.json>=<steps.parquet>,<kernels.parquet>` triple (repeatable).
    #[arg(long = "parquet", value_name = "MANIFEST=STEPS,KERNELS")]
    parquets: Vec<String>,

    /// Endpoint `[name=]manifest.json` (repeatable).
    #[arg(long = "endpoint", required = true, value_name = "[NAME=]MANIFEST")]
    endpoints: Vec<String>,

    /// Replicate the given endpoint(s) N times (e.g. `--replicas 4` creates 4 homogeneous replicas).
    #[arg(long, default_value_t = 1)]
    replicas: usize,

    /// Total VRAM per endpoint in GiB.
    #[arg(long, default_value_t = 71.2)]
    vram_gib: f64,

    /// Optional 1-point driver overhead calibration `<manifest.json>:<raw_pages>[:<active_slots>]`.
    #[arg(long)]
    calibrate: Option<String>,

    /// Active sequence slots per endpoint (`None` uses manifest default).
    #[arg(long)]
    slots: Option<u64>,

    /// Optional prefill chunk cap per call.
    #[arg(long)]
    chunk: Option<usize>,

    /// Optional max concurrent sequences per endpoint.
    #[arg(long)]
    max_seqs: Option<usize>,

    /// Optional step rows override.
    #[arg(long)]
    step_rows: Option<u64>,

    /// Mean accepted tokens per speculative round (when `step_rows > 1`).
    #[arg(long, default_value_t = 3.2)]
    mean_accepted: f64,

    /// Router policy to evaluate (repeatable: `round-robin`, `least-work`, `kv-aware`,
    /// `kv-aware:lag_ms=50`, `length-split:2048`).
    #[arg(long = "router", default_values_t = vec!["round-robin".to_string(), "kv-aware".to_string()])]
    routers: Vec<String>,

    /// Optional shared L2 KV pool bandwidth in GiB/s across endpoints.
    #[arg(long)]
    shared_pool_gbps: Option<f64>,

    /// Arrival mode: `open` or `closed`.
    #[arg(long, default_value = "closed")]
    mode: String,

    /// Optional Mooncake or native JSONL request trace file; if omitted, uses synthetic multi-turn sessions.
    #[arg(long)]
    trace: Option<PathBuf>,

    /// Synthetic trace: number of concurrent/staggered sessions.
    #[arg(long, default_value_t = 24)]
    sessions: usize,

    /// Synthetic trace: turns per session.
    #[arg(long, default_value_t = 4)]
    turns: usize,

    /// Synthetic trace: shared system prompt tokens.
    #[arg(long, default_value_t = 512)]
    sys_tokens: usize,

    /// Synthetic trace: new user/tool tokens appended per turn.
    #[arg(long, default_value_t = 256)]
    user_tokens: usize,

    /// Synthetic trace: generated assistant tokens per turn.
    #[arg(long, default_value_t = 128)]
    out_tokens: usize,

    /// Synthetic trace: mean gap between new session arrivals, in ms.
    #[arg(long, default_value_t = 35)]
    gap_ms: u64,

    /// Synthetic trace: client/tool think time between turns, in ms.
    #[arg(long, default_value_t = 120)]
    think_ms: u64,

    /// Require strict cubin sha256 matches only.
    #[arg(long)]
    strict_sha: bool,

    /// Fallback launch duration (µs) if running uncalibrated.
    #[arg(long)]
    fallback_launch_us: Option<f64>,

    /// Optional JSON output path.
    #[arg(long)]
    json: Option<PathBuf>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Command::Price(cmd) => run_price(cmd),
        Command::Budget(cmd) => run_budget(cmd),
        Command::Replay(cmd) => run_replay(cmd),
    }
}

fn load_verified(path: &Path, cache: &mut BTreeMap<PathBuf, Verified>) -> Result<Verified> {
    if let Some(m) = cache.get(path) {
        return Ok(m.clone());
    }
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading manifest {}", path.display()))?;
    let v = Verified::from_json(&text)
        .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    cache.insert(path.to_path_buf(), v.clone());
    Ok(v)
}

fn parse_labeled_path(spec: &str) -> (String, PathBuf) {
    if let Some((lbl, rest)) = spec.split_once('=') {
        if !lbl.is_empty() && !lbl.contains('/') {
            return (lbl.to_string(), PathBuf::from(rest));
        }
    }
    let p = PathBuf::from(spec);
    let label = p
        .parent()
        .and_then(|d| d.file_name())
        .or_else(|| p.file_stem())
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| spec.to_string());
    (label, p)
}

fn build_cost_book(
    benches: &[String],
    parquets: &[String],
    cache: &mut BTreeMap<PathBuf, Verified>,
) -> Result<CostBook> {
    let mut book = CostBook::new();
    for b in benches {
        let (m_str, bench_str) = b
            .split_once('=')
            .with_context(|| format!("expected MANIFEST=BENCH, got `{b}`"))?;
        let m = load_verified(Path::new(m_str), cache)?;
        book.ingest_bench_file(&m, Path::new(bench_str))?;
    }
    for pq in parquets {
        let (m_str, rest) = pq
            .split_once('=')
            .with_context(|| format!("expected MANIFEST=STEPS,KERNELS, got `{pq}`"))?;
        let (steps_s, kernels_s) = rest
            .split_once(',')
            .with_context(|| format!("expected STEPS,KERNELS after `=`, got `{rest}`"))?;
        let m = load_verified(Path::new(m_str), cache)?;
        book.ingest_parquet_pair(&m, Path::new(steps_s), Path::new(kernels_s))?;
    }
    Ok(book)
}

fn read_bench_total_s(path: &Path) -> Result<f64> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let val: serde_json::Value = serde_json::from_slice(&bytes)?;
    if let Some(us) = val.pointer("/cost/total_us").and_then(|v| v.as_f64()) {
        return Ok(us / 1e6);
    }
    let scenarios = val
        .get("scenarios")
        .and_then(|v| v.as_array())
        .context("missing scenarios in bench report")?;
    let mut total_us = 0.0;
    for s in scenarios {
        let w = s.pointer("/scenario/weight").and_then(|v| v.as_f64()).unwrap_or(1.0);
        let p50 = s.pointer("/graph/stats/p50").and_then(|v| v.as_f64()).unwrap_or(0.0);
        total_us += w * p50;
    }
    Ok(total_us / 1e6)
}

fn run_price(cmd: PriceCmd) -> Result<()> {
    let t0 = Instant::now();
    let mut cache = BTreeMap::new();
    let book = build_cost_book(&cmd.benches, &cmd.parquets, &mut cache)?;

    let mut candidates = Vec::new();
    for spec in &cmd.manifests {
        let (lbl, path) = parse_labeled_path(spec);
        let m = load_verified(&path, &mut cache)?;
        candidates.push((lbl, m));
    }

    let mut workloads = Vec::new();
    for wpath in &cmd.workloads {
        workloads.push(Workload::from_toml_file(wpath)?);
    }
    if !cmd.steps_workloads.is_empty() {
        let (_, ref_m) = &candidates[0];
        workloads.push(Workload::from_steps_parquet(
            "traced-steps",
            &cmd.steps_workloads,
            ref_m,
        )?);
    }
    if workloads.is_empty() {
        bail!("provide at least one `--workload <toml>` or `--steps-workload <parquet>`");
    }

    let mut actual_map = BTreeMap::new();
    for a in &cmd.actuals {
        let (lbl, p) = a
            .split_once('=')
            .with_context(|| format!("expected LABEL=BENCH, got `{a}`"))?;
        actual_map.insert(lbl.to_string(), read_bench_total_s(Path::new(p))?);
    }

    let opts = PriceOptions {
        allow_abi_fallback: !cmd.strict_sha,
        uncalibrated_launch_us: None,
    };

    let mut all_json = Vec::new();
    println!(
        "CostBook: {} calibration source(s), {} exact active-launch signatures (loaded in {:.2}s)",
        book.sources_ingested(),
        book.unique_exact_signatures(),
        t0.elapsed().as_secs_f64()
    );

    for w in &workloads {
        println!(
            "\nWorkload `{}` ({} scenarios, {} calls):",
            w.name,
            w.shapes.len(),
            w.total_calls()
        );
        let has_actual = !actual_map.is_empty();
        if has_actual {
            println!(
                "  {:<24} {:>11} {:>11} {:>9} {:>9} {:>10} {:>10}",
                "Candidate", "Pred (s)", "Actual (s)", "Err (%)", "vs #1", "ExactCov", "CalibCov"
            );
            println!("  {}", "-".repeat(90));
        } else {
            println!(
                "  {:<24} {:>11} {:>9} {:>10} {:>10}  Top Ops",
                "Candidate", "Pred (s)", "vs #1", "ExactCov", "CalibCov"
            );
            println!("  {}", "-".repeat(90));
        }

        let mut base_s: Option<f64> = None;
        for (lbl, m) in &candidates {
            let rep = price_workload(&book, m, w, opts)?;
            let first = *base_s.get_or_insert(rep.total_s);
            let rel_pct = if first > 0.0 {
                100.0 * (rep.total_s - first) / first
            } else {
                0.0
            };

            if has_actual {
                let (act_str, err_str) = if let Some(&act_s) = actual_map.get(lbl) {
                    let err = 100.0 * (rep.total_s - act_s) / act_s;
                    (format!("{act_s:11.3}"), format!("{err:+8.2}%"))
                } else {
                    ("          -".to_string(), "        -".to_string())
                };
                println!(
                    "  {:<24} {:11.3} {} {} {:+8.2}% {:9.2}% {:9.2}%",
                    lbl,
                    rep.total_s,
                    act_str,
                    err_str,
                    rel_pct,
                    rep.exact_time_coverage * 100.0,
                    rep.calibrated_time_coverage * 100.0,
                );
            } else {
                let top_ops: String = rep
                    .by_op
                    .iter()
                    .take(3)
                    .map(|(op, sh)| format!("{op} {:.1}%", sh * 100.0))
                    .collect::<Vec<_>>()
                    .join(", ");
                println!(
                    "  {:<24} {:11.3} {:+8.2}% {:9.2}% {:9.2}%  {}",
                    lbl,
                    rep.total_s,
                    rel_pct,
                    rep.exact_time_coverage * 100.0,
                    rep.calibrated_time_coverage * 100.0,
                    top_ops
                );
            }
            all_json.push(json!({ "candidate": lbl, "report": rep }));
        }
    }

    if let Some(out) = &cmd.json {
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(out, serde_json::to_vec_pretty(&all_json)?)?;
    }
    Ok(())
}

fn parse_calibration(
    spec: &str,
    total_vram_bytes: u64,
    cache: &mut BTreeMap<PathBuf, Verified>,
) -> Result<u64> {
    let parts: Vec<&str> = spec.split(':').collect();
    if !(2..=3).contains(&parts.len()) {
        bail!("expected `--calibrate <manifest.json>:<raw_pages>[:<active_slots>]`, got `{spec}`");
    }
    let m = load_verified(Path::new(parts[0]), cache)?;
    let pages: usize = parts[1].parse().context("parsing observed raw_pages")?;
    let active_slots = if parts.len() == 3 {
        Some(parts[2].parse::<u64>().context("parsing active_slots")?)
    } else {
        None
    };
    MemoryBudget::calibrate_driver_overhead(&m, total_vram_bytes, active_slots, pages)
}

fn run_budget(cmd: BudgetCmd) -> Result<()> {
    let mut cache = BTreeMap::new();
    let total_vram_bytes = (cmd.vram_gib * (1u64 << 30) as f64).round() as u64;
    let driver_overhead_bytes = match &cmd.calibrate {
        Some(cal) => parse_calibration(cal, total_vram_bytes, &mut cache)?,
        None => DEFAULT_DRIVER_OVERHEAD_BYTES,
    };

    let slot_sweep: Vec<Option<u64>> = if cmd.slots.is_empty() {
        vec![None]
    } else {
        cmd.slots.iter().copied().map(Some).collect()
    };

    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    println!(
        "VRAM Budget ({:.2} GiB total, {:.2} GiB driver/ctx overhead, 1.00 GiB headroom):",
        cmd.vram_gib,
        gib(driver_overhead_bytes)
    );
    println!(
        "  {:<22} {:>7} {:>9} {:>9} {:>9} {:>9} {:>9} {:>11}",
        "Manifest", "Slots", "Weights", "Act+Scr", "SeqState", "RawPages", "SchedPgs", "KV Tokens"
    );
    println!("  {}", "-".repeat(92));

    let mut rows_json = Vec::new();
    for spec in &cmd.manifests {
        let (lbl, path) = parse_labeled_path(spec);
        let m = load_verified(&path, &mut cache)?;
        for &active_slots in &slot_sweep {
            let mem_spec = DeviceMemorySpec {
                total_vram_bytes,
                driver_overhead_bytes,
                active_slots,
                explicit_tokens: None,
            };
            let b = MemoryBudget::compute(&m, mem_spec)?;
            let act_scr = b.activations_bytes + b.op_scratch_bytes + b.cublas_workspace_bytes;
            println!(
                "  {:<22} {:>7} {:>8.2}G {:>8.2}G {:>8.2}G {:>9} {:>9} {:>11}",
                lbl,
                b.provisioned_slots,
                gib(b.weights_bytes),
                gib(act_scr),
                gib(b.seq_states_bytes + b.fixed_states_bytes),
                b.raw_pages,
                b.scheduler_pages,
                b.scheduler_tokens(),
            );
            rows_json.push(json!({ "candidate": lbl, "budget": b }));
        }
    }

    if let Some(out) = &cmd.json {
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(out, serde_json::to_vec_pretty(&rows_json)?)?;
    }
    Ok(())
}

fn parse_router_spec(spec: &str) -> Result<(String, RouterPolicy)> {
    if spec == "round-robin" || spec == "rr" {
        return Ok(("round-robin".into(), RouterPolicy::RoundRobin));
    }
    if spec == "least-work" || spec == "lw" {
        return Ok((
            "least-work".into(),
            RouterPolicy::LeastWork {
                active_seq_weight: 256.0,
            },
        ));
    }
    if spec == "kv-aware" || spec == "kv" {
        return Ok(("kv-aware".into(), RouterPolicy::kv_aware_default()));
    }
    if let Some(rest) = spec.strip_prefix("kv-aware:") {
        let mut lag_ns = 0u64;
        for part in rest.split(',') {
            if let Some(ms_str) = part.strip_prefix("lag_ms=") {
                let ms: f64 = ms_str.parse().context("parsing lag_ms")?;
                lag_ns = (ms.max(0.0) * 1e6).round() as u64;
            } else {
                bail!("unknown kv-aware option `{part}` (expected `lag_ms=<num>`)");
            }
        }
        return Ok((
            spec.to_string(),
            RouterPolicy::KvAware {
                prefill_weight: 2.0,
                queue_weight: 0.5,
                active_seq_weight: 128.0,
                kv_event_lag_ns: lag_ns,
            },
        ));
    }
    if let Some(tok_s) = spec.strip_prefix("length-split:") {
        let short_max_tokens: usize = tok_s.parse().context("parsing length-split threshold")?;
        return Ok((
            spec.to_string(),
            RouterPolicy::LengthSplit { short_max_tokens },
        ));
    }
    bail!(
        "unknown router policy `{spec}` (expected `round-robin`, `least-work`, `kv-aware`, `kv-aware:lag_ms=<ms>`, or `length-split:<tokens>`)"
    )
}

fn run_replay(cmd: ReplayCmd) -> Result<()> {
    let mut cache = BTreeMap::new();
    let book = build_cost_book(&cmd.benches, &cmd.parquets, &mut cache)?;

    let total_vram_bytes = (cmd.vram_gib * (1u64 << 30) as f64).round() as u64;
    let driver_overhead_bytes = match &cmd.calibrate {
        Some(cal) => parse_calibration(cal, total_vram_bytes, &mut cache)?,
        None => DEFAULT_DRIVER_OVERHEAD_BYTES,
    };
    let mem_spec = DeviceMemorySpec {
        total_vram_bytes,
        driver_overhead_bytes,
        active_slots: cmd.slots,
        explicit_tokens: None,
    };

    let mut endpoints = Vec::new();
    let replicas = cmd.replicas.max(1);
    for ep_spec in &cmd.endpoints {
        let (base_name, path) = parse_labeled_path(ep_spec);
        let m = load_verified(&path, &mut cache)?;
        for r in 0..replicas {
            let name = if replicas == 1 && cmd.endpoints.len() == 1 {
                base_name.clone()
            } else if replicas > 1 {
                format!("{base_name}-{r}")
            } else {
                base_name.clone()
            };
            endpoints.push(EndpointConfig {
                name,
                manifest: m.clone(),
                memory: mem_spec,
                chunk_size: cmd.chunk,
                max_seqs: cmd.max_seqs,
                step_rows: cmd.step_rows,
                mean_accepted_tokens: cmd.mean_accepted,
            });
        }
    }

    let arrival_mode = match cmd.mode.as_str() {
        "open" | "open-loop" => ArrivalMode::OpenLoop,
        "closed" | "closed-loop" => ArrivalMode::ClosedLoop,
        other => bail!("unknown `--mode {other}` (expected `open` or `closed`)"),
    };

    let trace = match &cmd.trace {
        Some(p) => RequestTrace::from_jsonl_file(p, JsonlTraceOptions::default())?,
        None => RequestTrace::synthetic_sessions(SyntheticSessionConfig {
            sessions: cmd.sessions,
            turns_per_session: cmd.turns,
            system_prompt_tokens: cmd.sys_tokens,
            user_tokens_per_turn: cmd.user_tokens,
            output_tokens_per_turn: cmd.out_tokens,
            session_arrival_gap_ms: cmd.gap_ms,
            think_time_ms: cmd.think_ms,
            system_groups: 2,
            seed: 42,
        }),
    };

    let price_options = PriceOptions {
        allow_abi_fallback: !cmd.strict_sha,
        uncalibrated_launch_us: cmd.fallback_launch_us.or_else(|| {
            if book.sources_ingested() == 0 {
                Some(5.0)
            } else {
                None
            }
        }),
    };
    let shared_pool = cmd.shared_pool_gbps.map(|bw| SharedPoolConfig {
        bandwidth_gib_s: bw,
        lookup_us: 150.0,
    });

    println!(
        "Cluster Replay: {} endpoint(s), trace `{}` ({} requests, {:?})",
        endpoints.len(),
        trace.name,
        trace.requests.len(),
        arrival_mode
    );
    println!(
        "  {:<22} {:>8} {:>9} {:>9} {:>9} {:>9} {:>8} {:>8}",
        "Router", "Wall(s)", "OutTok/s", "TTFT p50", "TTFT p99", "TPOT p50", "HitRate", "Evicts"
    );
    println!("  {}", "-".repeat(90));

    let mut reports_json = Vec::new();
    for r_spec in &cmd.routers {
        let (r_label, policy) = parse_router_spec(r_spec)?;
        let cfg = ClusterConfig {
            endpoints: endpoints.clone(),
            router: policy,
            arrival_mode,
            price_options,
            shared_pool,
        };
        let rep = SimCluster::run(&book, &cfg, &trace)?;
        println!(
            "  {:<22} {:>8.2} {:>9.1} {:>8.1}m {:>8.1}m {:>8.2}m {:>7.1}% {:>8}",
            r_label,
            rep.wall_time_s,
            rep.output_tok_per_s,
            rep.ttft.p50_ms,
            rep.ttft.p99_ms,
            rep.tpot.p50_ms,
            rep.prefix_hit_rate * 100.0,
            rep.total_evictions,
        );
        reports_json.push(json!({ "router": r_label, "report": rep }));
    }

    if let Some(out) = &cmd.json {
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(out, serde_json::to_vec_pretty(&reports_json)?)?;
    }
    Ok(())
}
