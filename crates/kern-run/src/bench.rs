//! Single-device performance evidence, driven by the manifest's protocol.
//! Workload JSON describes sequences, not ABI names. GPU samples are kept
//! verbatim, including slow tails; no minimum-only or outlier filtering.
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{ensure, Context, Result};
use kern_manifest::protocol::{Axis, Rows};
use kern_manifest::types::{Arg, BufferKind, Dim, Fill, Manifest};
use kern_manifest::{Protocol, Verified};
use kern_runtime::profile::{Anchor, Probe};
use kern_runtime::{Capacity, Lease, Runtime};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::config::{Config, Target};
use crate::{le_bytes_i32, Env};

#[derive(clap::Args, Debug)]
pub struct BenchOpts {
    /// Only time whole programs, for low-cost checks without an activity tracer
    #[arg(long)]
    pub program_only: bool,
    /// Only run hardware probes (useful for independent profiler validation)
    #[arg(long)]
    pub calibrate_only: bool,
    #[arg(long)]
    pub manifest: Option<PathBuf>,
    #[arg(long)]
    pub kernels: Option<PathBuf>,
    #[arg(long)]
    pub weights: Vec<PathBuf>,
    #[arg(long)]
    pub tokenizer: Option<PathBuf>,
    #[arg(long)]
    pub gpu: Option<usize>,
    /// Scenario list, sample count and seed; independent of the model ABI
    #[arg(long)]
    pub workload: PathBuf,
    /// Portable JSON with raw samples and all call locations; no machine paths
    #[arg(long)]
    pub out: PathBuf,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct Scenario {
    pub id: String,
    pub kind: String,
    pub batch: usize,
    pub query: usize,
    /// One previous-context length per sequence, or one broadcast length
    pub context: Vec<usize>,
    #[serde(default)]
    pub holdout: bool,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Workload {
    pub samples: usize,
    pub seed: u64,
    pub scenarios: Vec<Scenario>,
}

impl Scenario {
    fn lengths(&self) -> Result<Vec<usize>> {
        ensure!(self.batch > 0 && self.query > 0, "{}: batch and query must be positive", self.id);
        ensure!(
            self.context.len() == 1 || self.context.len() == self.batch,
            "{}: one context or one per sequence",
            self.id
        );
        ensure!(self.kind == "step" || self.kind == "prompt", "{}: kind must be step or prompt", self.id);
        ensure!(self.kind != "step" || self.query == 1, "{}: ordinary step has one row per sequence", self.id);
        Ok(if self.context.len() == 1 { vec![self.context[0]; self.batch] } else { self.context.clone() })
    }
}

#[derive(Serialize, Debug)]
pub struct Stats {
    pub n: usize,
    pub min: f64,
    pub p10: f64,
    pub p50: f64,
    pub p90: f64,
    pub max: f64,
    pub mean: f64,
    pub cv: f64,
    pub tail_ratio: f64,
    pub block_medians: Vec<f64>,
}

pub fn stats(samples: &[f64]) -> Stats {
    assert!(!samples.is_empty() && samples.iter().all(|x| x.is_finite() && *x >= 0.));
    let mut v = samples.to_vec();
    v.sort_by(f64::total_cmp);
    let q = |p: f64| {
        let at = p * (v.len() - 1) as f64;
        let lo = at.floor() as usize;
        let hi = at.ceil() as usize;
        v[lo] + (v[hi] - v[lo]) * (at - lo as f64)
    };
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    let variance = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / v.len() as f64;
    let block_medians = samples
        .chunks(samples.len().div_ceil(4))
        .map(|b| {
            let mut b = b.to_vec();
            b.sort_by(f64::total_cmp);
            b[b.len() / 2]
        })
        .collect();
    Stats {
        n: v.len(),
        min: v[0],
        p10: q(0.1),
        p50: q(0.5),
        p90: q(0.9),
        max: *v.last().unwrap(),
        mean,
        cv: if mean > 0. { variance.sqrt() / mean } else { 0. },
        tail_ratio: if q(0.5) > 0. { q(0.9) / q(0.5) } else { 1. },
        block_medians,
    }
}

fn series(v: Vec<f64>) -> Value {
    json!({"stats":stats(&v),"samples_us":v})
}
fn digest(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn anchors(v: Vec<Anchor>) -> Value {
    Value::Array(
        v.into_iter()
            .map(|a| {
                json!({"name":a.name,"payload_bytes":a.bytes,
        "traffic_bytes":a.traffic_bytes,"flops":a.flops,"timing":series(a.samples_us)})
            })
            .collect(),
    )
}

fn telemetry(gpu: usize) -> Value {
    let fields = "clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used";
    let output = std::process::Command::new("nvidia-smi")
        .args(["-i", &gpu.to_string(), &format!("--query-gpu={fields}"), "--format=csv,noheader,nounits"])
        .output();
    match output {
        Ok(o) if o.status.success() => {
            let text = String::from_utf8_lossy(&o.stdout);
            let values = text.trim().split(',').map(|v| v.trim().parse::<f64>().ok()).collect::<Vec<_>>();
            json!({"sm_mhz":values.first(),"memory_mhz":values.get(1),"temperature_c":values.get(2),
                "power_w":values.get(3),"power_limit_w":values.get(4),"gpu_util_pct":values.get(5),"memory_mib":values.get(6)})
        }
        _ => Value::Null,
    }
}

/// Conservative grouping: retain state offsets, scalar args, ABI and all
/// buffer shapes/offsets and alias relationships. Weight names can vary.
fn signature(m: &Manifest, program: &str, index: usize, env: &Env) -> Value {
    let c = &m.programs[program].calls[index];
    let mut aliases: BTreeMap<&str, usize> = BTreeMap::new();
    let args: Vec<Value> = c
        .args
        .iter()
        .map(|a| match a {
            Arg::Buf { buf, offset } => {
                let next = aliases.len();
                let alias = *aliases.entry(buf).or_insert(next);
                let b = &m.buffers[buf];
                let shape: Vec<u64> = b
                    .shape
                    .iter()
                    .map(|d| match d {
                        Dim::Const(n) => *n,
                        Dim::Var(v) => env[v],
                    })
                    .collect();
                json!({"alias":alias,"dtype":b.dtype,"shape":shape,"kind":b.kind,"offset":offset})
            }
            Arg::State { state, offset } => json!({"state":state,"offset":offset}),
            Arg::Var { var } => json!({"scalar":env[var]}),
            Arg::Expr { expr } => json!({"scalar":expr.eval(env).expect("verified expression")}),
            _ => serde_json::to_value(a).unwrap(),
        })
        .collect();
    // Env remains in the key because impl-private geometry may read vars
    // not forwarded by the call's public interface.
    json!({"op":c.op,"args":args,"env":env})
}

fn stage(
    rt: &mut Runtime,
    p: &Protocol,
    leases: &[Lease],
    positions: &[usize],
    rows: usize,
    ids: &[i64],
) -> Result<Env> {
    let b = leases.len();
    let env = p.env(b as u64, rows as u64, (b * rows) as u64);
    for f in &p.fills {
        let vals: Vec<i64> = match f.fill {
            Fill::Token => match f.axis {
                Axis::Groups => ids.chunks(rows).map(|x| x[0]).collect(),
                _ => ids.to_vec(),
            },
            Fill::Position => positions.iter().flat_map(|pos| (*pos..*pos + rows).map(|v| v as i64)).collect(),
            Fill::Slot => leases.iter().zip(positions).flat_map(|(l, pos)| l.slots(*pos..*pos + rows)).collect(),
            Fill::SeqLen => positions.iter().map(|pos| (*pos + rows) as i64).collect(),
            Fill::CuSeqlens => (0..=b).map(|i| (i * rows) as i64).collect(),
            Fill::SpanAt => vec![0],
            Fill::Blocks => anyhow::bail!("multi-device workload is outside this profiler"),
            _ => continue,
        };
        rt.write_input_at(&f.name, &f.encode(&vals), &env)?;
    }
    for t in &p.page_tables {
        let mut vals = Vec::new();
        for lease in leases {
            lease.extend_row(&t.name, &mut vals)?;
        }
        rt.write_input_at(&t.name, &le_bytes_i32(&vals), &env)?;
    }
    for t in &p.line_tables {
        let mut vals = Vec::new();
        // Line tables use their declared maximum column stride, even when
        // only a prefix of columns is active.
        for line in 0..t.lines {
            for col in 0..p.groups.max as usize {
                vals.push(leases[col.min(b - 1)].seq_line(&t.name, line)?);
                vals.extend(std::iter::repeat_n(0, t.width - 1));
            }
        }
        rt.write_input_at(&t.name, &le_bytes_i32(&vals), &env)?;
    }
    Ok(env)
}

fn corpus(tokenizer: &PathBuf, seed: u64) -> Result<Vec<i64>> {
    let tok = tokenizers::Tokenizer::from_file(tokenizer).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let topics = [
        "A botanist mapped the small plants growing between the stones. Some leaves stored water, while others curled away from the afternoon sunlight.",
        "The engineer compared three implementations using measured latency and memory traffic. A smaller matrix did not always finish sooner, because the scheduling algorithm changed.",
        "At the harbor, a wooden boat carried baskets of oranges and a box of weather instruments. The crew checked the wind before crossing the bay.",
        "The library catalog described an expedition through the mountains. Its notebooks contained sketches, temperature readings, and careful accounts of conversations in each village.",
        "A musician rehearsed the passage slowly, listening for uneven intervals. Later the ensemble adjusted its timing until the melody became clear across the room.",
        "An astronomer explained why a distant planet was hard to observe. Its atmosphere, orbital period, and reflected light each offered a different piece of evidence.",
    ];
    let text = (0..320)
        .map(|i| format!("Observation {}: {}\n", i as u64 + seed, topics[(i * 5 + seed as usize) % topics.len()]))
        .collect::<String>();
    Ok(tok
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?
        .get_ids()
        .iter()
        .map(|x| *x as i64)
        .collect())
}

fn ids(corpus: &[i64], start: usize, rows: usize) -> Vec<i64> {
    (0..rows).map(|i| corpus[(start + i) % corpus.len()]).collect()
}

fn output_fingerprints(rt: &Runtime, p: &Protocol, batch: usize) -> Result<Vec<Value>> {
    p.fills
        .iter()
        .filter(|f| rt.manifest.buffers[&f.name].kind == BufferKind::Output)
        .map(|f| {
            let bytes = rt.read_output(&f.name)?;
            let values: Vec<i64> = f.decode(&bytes).into_iter().take(batch * f.width as usize).collect();
            Ok(json!({"name":f.name,"values":values}))
        })
        .collect()
}

pub fn run(o: BenchOpts, cfg: Option<&Config>, target: Option<&Target>) -> Result<()> {
    let manifest = o.manifest.as_ref().or(target.map(|t| &t.manifest)).context("--manifest or target required")?;
    let kernels = o.kernels.as_ref().or(target.map(|t| &t.kernels)).context("--kernels or target required")?;
    let tokenizer = o
        .tokenizer
        .as_ref()
        .or(target.and_then(|t| t.tokenizer.as_ref()))
        .context("tokenizer required for prose workloads")?;
    let weights = if o.weights.is_empty() { target.map(|t| t.weights.as_slice()).unwrap_or(&[]) } else { &o.weights };
    ensure!(!weights.is_empty(), "weights required");
    let workload: Workload = serde_json::from_slice(&std::fs::read(&o.workload)?)?;
    ensure!((12..=256).contains(&workload.samples), "samples must be in 12..=256");
    ensure!(!workload.scenarios.is_empty(), "no scenarios");
    let mut names = BTreeSet::new();
    let mut max_batch = 0;
    for s in &workload.scenarios {
        ensure!(names.insert(&s.id), "duplicate scenario id {}", s.id);
        s.lengths()?;
        max_batch = max_batch.max(s.batch);
    }
    let json_bytes = std::fs::read(manifest)?;
    let m = Verified::from_json(std::str::from_utf8(&json_bytes)?)?;
    let unit = kern_runtime::page_unit(&m) as usize;
    let capacity = workload
        .scenarios
        .iter()
        .map(|s| s.lengths().unwrap().iter().map(|n| (n + s.query).div_ceil(unit) * unit).sum::<usize>() + unit)
        .max()
        .unwrap();
    ensure!(m.topology.is_none(), "this demo profiles single-device manifests");
    let p = Protocol::check(&m)?;
    for s in &workload.scenarios {
        let rows = if s.kind == "prompt" { Rows::Var } else { Rows::Const(1) };
        ensure!(p.forward(s.batch as u64, rows).is_some(), "{}: unsupported batch/program shape", s.id);
        ensure!(s.batch * s.query <= p.rows.max as usize, "{}: row capacity exceeded", s.id);
    }
    let gpu = o.gpu.or(cfg.and_then(|c| c.gpu)).unwrap_or(0);
    let mut rt = Runtime::load(
        &m,
        kernels,
        gpu,
        Some(Capacity { tokens: Some(capacity as u64), seqs: max_batch as u64 }),
        None,
    )?;
    let blobs = weights.iter().map(std::fs::read).collect::<std::io::Result<Vec<_>>>()?;
    rt.load_weights(&blobs.iter().map(Vec::as_slice).collect::<Vec<_>>())?;
    drop(blobs);
    let corpus = corpus(tokenizer, workload.seed)?;
    let probe = Probe::new(&rt)?;
    eprintln!("calibrating {} · L2 {} MiB", probe.device, probe.l2_bytes >> 20);
    let before = anchors(probe.calibrate(&rt, workload.samples)?);
    let mut report = json!({"version":1,"model":m.model,"manifest_sha256":digest(&json_bytes),
        "runner_sha256":digest(&std::fs::read(std::env::current_exe()?)?),
        "created_unix":SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        "hardware":{"device":probe.device,"sm_count":probe.sm_count,"l2_bytes":probe.l2_bytes,
            "eviction_bytes":probe.eviction_bytes,"driver_version":probe.driver_version},
        "protocol":{"timer":"CUDA graph events","op_cache_modes":["cold_l2","warm_replay"],
            "program_entry_cache":"cold_l2; natural reuse between calls","restore":"declared writes before each sample",
            "samples_per_mode":workload.samples,"tail_policy":"all measured samples retained; no outlier filtering",
            "warmup": "3 call executions; 4 whole-program executions",
            "grouping":"same op, resolved arguments, buffer shape, aliasing, offsets and env; weight names omitted",
            "context_data":"deterministic diverse prose, actual model prefix execution",
            "units":"microseconds; bandwidth GB/s uses read+write traffic"},
        "modules":m.modules.iter().map(|(n,x)|json!({"name":n,"sha256":x.sha256})).collect::<Vec<_>>(),
        "calibration_before":before,"workload":workload,"scenarios":[],"program_only":o.program_only});
    if o.calibrate_only {
        report["calibration_only"] = json!(true);
        if let Some(parent) = o.out.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&o.out, serde_json::to_vec(&report)?)?;
        return Ok(());
    }
    for s in &workload.scenarios {
        let started = Instant::now();
        let lengths = s.lengths()?;
        let rows = if s.kind == "prompt" { Rows::Var } else { Rows::Const(1) };
        let f = p.forward(s.batch as u64, rows).unwrap();
        eprintln!("{} · {} · batch {} · query {} · contexts {:?}", s.id, f.name, s.batch, s.query, lengths);
        let telemetry_before = telemetry(gpu);
        let leases = lengths.iter().map(|n| rt.lease(n + s.query)).collect::<kern_runtime::Result<Vec<_>>>()?;
        // Each sequence gets its own real prefix and private state. Never
        // duplicate one lease across batch rows to save preparation time.
        for (i, &length) in lengths.iter().enumerate() {
            let mut pos = 0;
            while pos < length {
                let q = (length - pos).min(p.rows.max as usize);
                let prefix = p.chunk().context("prefix workload needs a variable-row program")?;
                let env = stage(&mut rt, &p, &leases[i..i + 1], &[pos], q, &ids(&corpus, i * 173 + pos, q))?;
                if !rt.is_captured(&prefix.name, &env) {
                    rt.capture(&prefix.name, &env)?;
                }
                rt.run_captured(&prefix.name, &env)?;
                pos += q;
            }
        }
        let input: Vec<i64> =
            lengths.iter().enumerate().flat_map(|(i, pos)| ids(&corpus, i * 173 + pos, s.query)).collect();
        let env = stage(&mut rt, &p, &leases, &lengths, s.query, &input)?;
        let program_samples = probe.program(&rt, &f.name, &env, workload.samples)?;
        let reference_outputs = output_fingerprints(&rt, &p, s.batch)?;
        if o.program_only {
            report["scenarios"].as_array_mut().unwrap().push(json!({
                "scenario":s,"program":f.name,"env":env,"graph":series(program_samples.graph_us),
                "instrumented":series(program_samples.instrumented_us),"outputs":reference_outputs,
                "telemetry_before":telemetry_before,"telemetry_after":telemetry(gpu),
                "elapsed_s":started.elapsed().as_secs_f64()}));
            if let Some(parent) = o.out.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&o.out, serde_json::to_vec(&report)?)?;
            continue;
        }
        // program() leaves persistent state at its pre-image; stage restores
        // all caller inputs before walking the calls to obtain real inputs.
        stage(&mut rt, &p, &leases, &lengths, s.query, &input)?;
        let mut groups: BTreeMap<String, usize> = BTreeMap::new();
        let mut cases = Vec::new();
        let mut calls = Vec::new();
        for (i, c) in m.programs[&f.name].calls.iter().enumerate() {
            let sig = signature(&m, &f.name, i, &env);
            let key = digest(&serde_json::to_vec(&sig)?);
            let case = if let Some(&case) = groups.get(&key) {
                case
            } else {
                let sample = probe
                    .call(&rt, &f.name, &env, i, workload.samples)
                    .with_context(|| format!("{} call {i} ({})", s.id, c.op))?;
                let case = cases.len();
                groups.insert(key.clone(), case);
                cases.push(json!({"id":key,"signature":sig,"representative_call":i,
                    "cold":series(sample.cold_us),"warm":series(sample.warm_us)}));
                case
            };
            calls.push(json!({"index":i,"label":c.label,"op":c.op,"case":case,"args":c.args,
                "launches":m.ops[&c.op].imp.launches.iter().map(|l|json!({"entry":l.entry(),"module":l.module()})).collect::<Vec<_>>(),
                "in_program":series(program_samples.attributed_us[i].clone())}));
            rt.run_range(&f.name, &env, i, i + 1)?;
        }
        let outputs = output_fingerprints(&rt, &p, s.batch)?;
        ensure!(outputs == reference_outputs, "{}: profiled trajectory differs from whole program output", s.id);
        let record = json!({"scenario":s,"program":f.name,"env":env,"graph":series(program_samples.graph_us),
            "instrumented":series(program_samples.instrumented_us),"cases":cases,"calls":calls,"outputs":outputs,
            "output_check":"matches whole-program token outputs",
            "telemetry_before":telemetry_before,"telemetry_after":telemetry(gpu),"elapsed_s":started.elapsed().as_secs_f64()});
        eprintln!(
            "{} · {:.3} ms · {} calls / {} cases · {:.1}s",
            s.id,
            record["graph"]["stats"]["p50"].as_f64().unwrap() / 1000.,
            calls.len(),
            cases.len(),
            started.elapsed().as_secs_f64()
        );
        report["scenarios"].as_array_mut().unwrap().push(record);
        if let Some(parent) = o.out.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&o.out, serde_json::to_vec(&report)?)?;
        drop(leases);
    }
    report["calibration_after"] = anchors(probe.calibrate(&rt, workload.samples)?);
    std::fs::write(&o.out, serde_json::to_vec(&report)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tails_and_order_are_preserved() {
        let s = stats(&[1., 1., 1., 1., 1., 1., 1., 1., 10., 30., 50., 100.]);
        assert_eq!(s.n, 12);
        assert_eq!(s.max, 100.);
        assert_eq!(s.p50, 1.);
        assert!(s.tail_ratio > 40. && s.cv > 1.);
        assert!(s.block_medians[3] > s.block_medians[0]);
    }
    #[test]
    fn workload_rejects_impossible_context_cardinality() {
        let s =
            Scenario { id: "bad".into(), kind: "step".into(), batch: 4, query: 1, context: vec![1, 2], holdout: false };
        assert!(s.lengths().is_err());
    }
}
