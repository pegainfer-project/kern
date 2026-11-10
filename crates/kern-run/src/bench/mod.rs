//! Single-device performance evidence, driven by the manifest's protocol.
//!
//! Where does a program's time go: every sample is one traced replay of
//! the real graph, from the program's pre-image behind an L2 flush, and
//! the tracer names each kernel by the call that launched it, so the
//! graph's time and every call's come from the same replay with nothing
//! inserted between kernels.
//!
//! Nothing here names a program, a buffer or a model. A workload gives
//! shapes ([`workload`]), written by hand or read off a server's trace
//! ([`served`]), `Protocol` turns a shape into a program, and the fill
//! table says what to stage. Samples are kept verbatim, slow tails
//! included; no minimum-only or outlier filtering.
//!
//! A manifest with a topology runs as its ranks, every rank fed the same
//! shape with its own sequences, in lockstep: a call that reads its peers
//! finds them issuing. Attribution is per rank.

pub mod served;
pub mod workload;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{ensure, Context, Result};
use kern_manifest::types::{BufferKind, Manifest};
use kern_manifest::{Protocol, Verified};
use kern_pool::Lease;
use kern_runtime::profile::{Graph, PreImage, Probe};
use kern_runtime::trace::{self, Step};
use kern_runtime::{Capacity, HostWeights, Runtime};
use kern_test::report::row;
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::config::{Config, Target};
use crate::Vars;
use workload::{Plan, Scenario, Shape, Workload};

/// The report schema. Readers check it before they read anything else.
const VERSION: u32 = 5;

/// Untimed replays before a graph's samples.
const WARMUP: usize = 4;

/// Ops named on the mix line before it says how many are left.
const SHOW: usize = 6;

#[derive(clap::Args, Debug)]
pub struct BenchOpts {
    /// Sweep of call shapes, sample count and seed (TOML); no ABI names
    #[arg(long)]
    workload: PathBuf,
    /// Portable JSON with raw samples and all call locations; no machine paths
    #[arg(long)]
    out: PathBuf,
    /// Manifest JSON (must pass verification)
    #[arg(long)]
    manifest: Option<PathBuf>,
    /// Directory of cubins; steps resolve by their pinned sha256, so one dir
    /// holds every version (file names are labels)
    #[arg(long)]
    kernels: Option<PathBuf>,
    /// Checkpoint directories or .safetensors files, tensors bound by name
    /// across all of them
    #[arg(long)]
    weights: Vec<String>,
    /// HF tokenizer.json; the prose every context is built from goes through it
    #[arg(long)]
    tokenizer: Option<PathBuf>,
    /// CUDA device ordinal; a manifest with a topology takes this one and
    /// the next ranks-1 after it
    #[arg(long)]
    gpu: Option<usize>,
}

#[derive(Serialize, Debug)]
struct Stats {
    n: usize,
    min: f64,
    p10: f64,
    p50: f64,
    p90: f64,
    max: f64,
    mean: f64,
    cv: f64,
    tail_ratio: f64,
    block_medians: Vec<f64>,
}

fn stats(samples: &[f64]) -> Stats {
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

/// Share of a program's attributed time by op, largest first. The shares
/// are of the kernels' total, not of the graph: time between kernels
/// belongs to no call.
fn mix(calls: &[(String, f64)]) -> Vec<(String, f64)> {
    let mut by: BTreeMap<&str, f64> = BTreeMap::new();
    for (op, us) in calls {
        *by.entry(op).or_default() += us;
    }
    let total: f64 = by.values().sum();
    let mut v: Vec<(String, f64)> =
        by.into_iter().map(|(op, us)| (op.to_string(), if total > 0. { us / total } else { 0. })).collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    v
}

/// One scenario's result on its slowest rank: program p50 µs and the share
/// of it each op was attributed.
struct Priced {
    shape: Shape,
    us: f64,
    shares: Vec<(String, f64)>,
}

/// What the workload's traffic costs on this manifest: every shape's
/// program time times its weight, in µs; and that time's share by op.
fn cost(runs: &[Priced]) -> (f64, Vec<(String, f64)>) {
    let total = runs.iter().map(|r| r.shape.weight as f64 * r.us).sum();
    let by_op: Vec<(String, f64)> = runs
        .iter()
        .flat_map(|r| r.shares.iter().map(move |(op, f)| (op.clone(), r.shape.weight as f64 * r.us * f)))
        .collect();
    (total, mix(&by_op))
}

fn series(v: Vec<f64>) -> Value {
    json!({"stats":stats(&v),"samples_us":v})
}

fn digest(b: &[u8]) -> String {
    hex::encode(Sha256::digest(b))
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

fn corpus(tokenizer: &std::path::Path, seed: u64) -> Result<Vec<i64>> {
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

/// `rows` tokens of prose for sequence `seq` from position `pos`. Every
/// sequence reads the corpus from its own offset, so the rows of a batch
/// are different text and not one prompt copied across the batch.
fn tokens(corpus: &[i64], seq: usize, pos: usize, rows: usize) -> Vec<i64> {
    (0..rows).map(|i| corpus[(seq * 173 + pos + i) % corpus.len()]).collect()
}

fn output_fingerprints(rt: &Runtime, p: &Protocol, groups: usize) -> Result<Vec<Value>> {
    p.fills
        .iter()
        .filter(|f| rt.manifest.buffers[&f.name].kind == BufferKind::Output)
        .map(|f| {
            let bytes = rt.read_output(&f.name)?;
            let values: Vec<i64> = f.decode(&bytes).into_iter().take(groups * f.width as usize).collect();
            Ok(json!({"name":f.name,"values":values}))
        })
        .collect()
}

/// One real prefix of `length` tokens of prose, built by running the chunk
/// program over it: the source every scenario's sequences copy their
/// context from. Copying keeps each sequence on its own pages, so a batch
/// reads as much memory as distinct sequences would, and a long context
/// costs a copy instead of its prefill.
fn source(rt: &mut Runtime, p: &Protocol, length: usize, corpus: &[i64], seq: usize) -> Result<Option<Lease>> {
    if length == 0 {
        return Ok(None);
    }
    let chunk = p.chunk().context("a prefix needs a variable-row program")?.name.clone();
    let lease = rt.lease(length)?;
    let mut pos = 0;
    while pos < length {
        let q = (length - pos).min(p.rows.max as usize);
        let vars = crate::stage_batch(rt, p, std::slice::from_ref(&lease), &[pos], q, &tokens(corpus, seq, pos, q))?;
        if !rt.is_captured(&chunk, &vars) {
            rt.capture(&chunk, &vars)?;
        }
        rt.run_captured(&chunk, &vars)?;
        pos += q;
    }
    Ok(Some(lease))
}

/// One sample's whole-graph time, first kernel start to last kernel end,
/// and each of the program's `calls` calls' kernel time, in µs.
fn timing(step: &Step, calls: usize) -> Result<(f64, Vec<f64>)> {
    let (got, want) = (step.spans.len(), step.issue.kernels.len());
    ensure!(
        got == want && got > 0,
        "`{}`: the tracer returned {got} of the graph's {want} kernels ({} records dropped so far)",
        step.issue.program,
        trace::dropped()
    );
    let mut per = vec![0.; calls];
    for s in &step.spans {
        let c = s.call.filter(|&c| (c as usize) < calls).with_context(|| {
            format!("`{}`: kernel `{}` is not one of the program's calls", step.issue.program, s.name)
        })?;
        per[c as usize] += (s.end - s.start) as f64 / 1e3;
    }
    let start = step.spans.iter().map(|s| s.start).min().unwrap();
    let end = step.spans.iter().map(|s| s.end).max().unwrap();
    Ok(((end - start) as f64 / 1e3, per))
}

/// The tracer's steps for `seqs`, taken out of `steps`.
fn take(steps: &mut BTreeMap<u64, Step>, seqs: &[u64]) -> Result<Vec<Step>> {
    seqs.iter()
        .map(|q| steps.remove(q).with_context(|| format!("the tracer returned no kernels for issue {q}")))
        .collect()
}

/// `samples` traced replays of `g` after the warmup; their sequence numbers.
fn replays(r: &Rank, pre: &PreImage, g: &Graph, samples: usize) -> Result<Vec<u64>> {
    let all = (0..WARMUP + samples).map(|_| r.probe.sample(&r.rt, pre, g)).collect::<kern_runtime::Result<Vec<_>>>()?;
    Ok(all[WARMUP..].to_vec())
}

/// One line, one fact, in the shape `kern test` reports in.
fn say(key: &str, body: String) {
    println!("{}", row(key, body, None));
}

fn say_timed(key: &str, body: String, since: Instant) {
    println!("{}", row(key, body, Some(since.elapsed().as_secs_f32())));
}

fn program_line(g: &Stats, calls: usize) -> String {
    format!(
        "{:.3} ms · p10–p90 {:.3}–{:.3} · cv {:.1}% · {calls} calls",
        g.p50 / 1e3,
        g.p10 / 1e3,
        g.p90 / 1e3,
        g.cv * 100.
    )
}

/// The largest shares by name, then how many ops are left unnamed.
fn mix_line(shares: &[(String, f64)]) -> String {
    let named = shares.iter().take(SHOW).map(|(op, f)| format!("{op} {:.1}%", f * 100.)).collect::<Vec<_>>();
    match shares.len().saturating_sub(SHOW) {
        0 => named.join(" · "),
        rest => format!("{} · +{rest} more", named.join(" · ")),
    }
}

/// Everything about the report that is settled before the first sample:
/// what was measured, on what, and under which protocol.
fn preamble(m: &Verified, json_bytes: &[u8], w: &Workload, plan: &Plan, probe: &Probe, ranks: usize) -> Result<Value> {
    let timer = match probe.hes {
        true => "CUPTI kernel records, GPU timestamps (HES)",
        false => "CUPTI kernel records, software timestamps",
    };
    Ok(json!({"version":VERSION,"model":m.model,"manifest_sha256":digest(json_bytes),
        "runner_sha256":digest(&std::fs::read(std::env::current_exe()?)?),
        "created_unix":SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        "ranks":ranks,
        "hardware":{"device":probe.device,"sm_count":probe.sm_count,"l2_bytes":probe.l2_bytes,
            "driver_version":probe.driver_version},
        "protocol":{"timer":timer,"graph":"first kernel start to last kernel end",
            "call":"sum of the call's kernel durations in the same replay",
            "entry":"L2 flushed; declared carry and state writes restored","samples":w.samples,"warmup":WARMUP,
            "context_data":"deterministic diverse prose; one real prefix per rank, copied onto each sequence's own pages",
            "units":"microseconds"},
        "modules":m.modules.iter().map(|(n,x)|json!({"name":n,"sha256":x.sha256})).collect::<Vec<_>>(),
        "workload":{"samples":w.samples,"seed":w.seed,"scenarios":&plan.scenarios,"dropped":&plan.dropped},
        "scenarios":[]}))
}

/// One rank's issues of one scenario, before the tracer has their kernels.
struct Measured {
    vars: Vars,
    program: Vec<u64>,
    telemetry: (Value, Value),
}

/// One rank's evidence of one scenario: per sample the graph's time and
/// every call's.
struct Timed {
    vars: Vars,
    graph_us: Vec<f64>,
    calls_us: Vec<Vec<f64>>,
    telemetry: (Value, Value),
}

impl Measured {
    fn time(self, steps: &mut BTreeMap<u64, Step>, calls: usize) -> Result<Timed> {
        let timed = take(steps, &self.program)?.iter().map(|s| timing(s, calls)).collect::<Result<Vec<_>>>()?;
        let mut calls_us = vec![Vec::with_capacity(timed.len()); calls];
        for (_, per) in &timed {
            calls_us.iter_mut().zip(per).for_each(|(v, us)| v.push(*us));
        }
        let graph_us = timed.into_iter().map(|t| t.0).collect();
        Ok(Timed { vars: self.vars, graph_us, calls_us, telemetry: self.telemetry })
    }
}

/// The report's record of one scenario on one rank: the manifest's calls
/// annotated with what was measured of each.
fn record(m: &Manifest, s: &Scenario, rank: usize, x: Timed, elapsed_s: f64) -> Value {
    let calls: Vec<Value> = m.programs[&s.program]
        .calls
        .iter()
        .zip(x.calls_us)
        .enumerate()
        .map(|(i, (c, us))| json!({"index":i,"label":c.label,"op":c.op,"in_program":series(us)}))
        .collect();
    json!({"scenario":s,"program":s.program,"rank":rank,"vars":x.vars,"graph":series(x.graph_us),
        "calls":calls,
        "telemetry_before":x.telemetry.0,"telemetry_after":x.telemetry.1,"elapsed_s":elapsed_s})
}

/// One rank of the manifest: its runtime, the probe over it, the GPU it
/// is on and its index, which picks the prose its sequences read.
struct Rank {
    rt: Runtime,
    probe: Probe,
    gpu: usize,
    rank: usize,
    source: Option<Lease>,
}

/// One rank's part of a scenario: lease, prefix, stage, sample. Every
/// rank runs this at once for the same shape, so the graphs replay in
/// lockstep and a peer read finds its peer issuing. One more replay must
/// emit the tokens the last sample did, or restoring the pre-image does
/// not put the state back.
fn measure(r: &mut Rank, p: &Protocol, s: &Scenario, corpus: &[i64], samples: usize) -> Result<Measured> {
    let shape = &s.shape;
    let before = telemetry(r.gpu);
    let leases = shape.context.iter().map(|n| r.rt.lease(n + shape.rows)).collect::<kern_runtime::Result<Vec<_>>>()?;
    let seq = |i: usize| r.rank * shape.groups + i;
    if let Some(src) = &r.source {
        for (lease, &n) in leases.iter().zip(&shape.context) {
            r.rt.replicate(src, lease, n)?;
        }
    }
    let input: Vec<i64> =
        shape.context.iter().enumerate().flat_map(|(i, &pos)| tokens(corpus, seq(i), pos, shape.rows)).collect();
    let vars = crate::stage_batch(&mut r.rt, p, &leases, &shape.context, shape.rows, &input)?;
    let pre = PreImage::new(&r.rt, &s.program)?;
    let graph = Graph::new(&r.rt, &s.program, &vars)?;
    let program = replays(r, &pre, &graph, samples)?;
    let last = output_fingerprints(&r.rt, p, shape.groups)?;
    r.probe.sample(&r.rt, &pre, &graph)?;
    ensure!(
        output_fingerprints(&r.rt, p, shape.groups)? == last,
        "{}: a replay from the restored pre-image emitted different tokens",
        s.id
    );
    Ok(Measured { vars, program, telemetry: (before, telemetry(r.gpu)) })
}

/// What every scenario is measured against: loaded once, and unchanged by
/// any of them. The manifest is the runtimes' own, so a scenario cannot be
/// measured against a different one than it runs on.
struct Bench {
    ranks: Vec<Rank>,
    protocol: Protocol,
    corpus: Vec<i64>,
    samples: usize,
}

impl Bench {
    /// Every rank on its GPU, peers connected, what the manifest runs once
    /// run, a probe over each.
    fn load(m: &Verified, inputs: &crate::Inputs, protocol: Protocol, plan: &Plan) -> Result<Bench> {
        let n = crate::ranks_of(m)?;
        let gpus: Vec<usize> = (inputs.gpu..inputs.gpu + n).collect();
        let host_weights = HostWeights::new();
        let capacity = Capacity { tokens: Some(plan.tokens), slots: plan.seqs, landings: 0 };
        let loaded: Vec<Result<crate::Sent<Runtime>>> = std::thread::scope(|s| {
            let handles: Vec<_> = gpus
                .iter()
                .enumerate()
                .map(|(q, &gpu)| {
                    let (topo, host_weights) = (crate::topology_of(m, q), &host_weights);
                    s.spawn(move || -> Result<crate::Sent<Runtime>> {
                        let topology = m.topology.is_some().then_some(&topo);
                        let mut rt = Runtime::load_with_host_weights(
                            m,
                            inputs.kernels.as_deref(),
                            gpu,
                            Some(capacity),
                            topology,
                            host_weights,
                        )
                        .with_context(|| format!("rank {q} on gpu {gpu}"))?;
                        inputs.weights.bind(&mut rt).with_context(|| format!("rank {q}: binding weights"))?;
                        Ok(crate::Sent(rt))
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().expect("a rank's load thread panicked")).collect()
        });
        let mut rts: Vec<Runtime> = loaded.into_iter().map(|r| r.map(|s| s.0)).collect::<Result<_>>()?;
        crate::connect_peers(m, &mut rts)?;
        let probes = crate::each(&mut rts, "running what the manifest runs once", |rt| {
            crate::run_once(rt, &protocol)?;
            Ok(Probe::new(rt)?)
        })?;
        let ranks = rts
            .into_iter()
            .zip(probes)
            .zip(gpus)
            .enumerate()
            .map(|(rank, ((rt, probe), gpu))| Rank { rt, probe, gpu, rank, source: None })
            .collect();
        Ok(Bench { ranks, protocol, corpus: Vec::new(), samples: 0 })
    }

    /// Build every rank's source prefix, the longest context any scenario
    /// reaches.
    fn prime(&mut self, length: usize) -> Result<()> {
        let (p, corpus) = (&self.protocol, &self.corpus);
        crate::each(&mut self.ranks, "building the source prefix", |r| {
            r.source = source(&mut r.rt, p, length, corpus, r.rank)?;
            Ok(())
        })?;
        Ok(())
    }

    /// One scenario end to end on every rank: measure, say, record. The
    /// terminal shows the slowest rank, which is what a step waits for;
    /// the report keeps every rank.
    fn scenario(&mut self, s: &Scenario) -> Result<(Vec<Value>, Priced)> {
        let started = Instant::now();
        say("scenario", s.id.clone());
        let (p, corpus, samples) = (&self.protocol, &self.corpus, self.samples);
        let measured = crate::each(&mut self.ranks, &s.id, |r| measure(r, p, s, corpus, samples))?;
        let m = &self.ranks[0].rt.manifest;
        let calls = &m.programs[&s.program].calls;
        let mut steps: BTreeMap<u64, Step> = trace::finish()?.steps.into_iter().map(|t| (t.seq, t)).collect();
        let timed = measured.into_iter().map(|x| x.time(&mut steps, calls.len())).collect::<Result<Vec<_>>>()?;
        let p50 = |x: &Timed| stats(&x.graph_us).p50;
        let (q, slow) = timed.iter().enumerate().max_by(|a, b| p50(a.1).total_cmp(&p50(b.1))).unwrap();
        let ranks = match timed.len() {
            1 => String::new(),
            n => format!(" · rank {q} slowest of {n}"),
        };
        say_timed("program", program_line(&stats(&slow.graph_us), calls.len()) + ranks.as_str(), started);
        let attributed: Vec<(String, f64)> =
            calls.iter().zip(&slow.calls_us).map(|(c, us)| (c.op.clone(), stats(us).p50)).collect();
        let shares = mix(&attributed);
        say("mix", mix_line(&shares));
        let elapsed_s = started.elapsed().as_secs_f64();
        let priced = Priced { shape: s.shape.clone(), us: p50(slow), shares };
        Ok((timed.into_iter().enumerate().map(|(q, x)| record(m, s, q, x, elapsed_s)).collect(), priced))
    }
}

pub fn run(o: BenchOpts, cfg: Option<&Config>, target: Option<&Target>) -> Result<()> {
    let given = crate::Given {
        manifest: o.manifest.clone(),
        kernels: o.kernels.clone(),
        weights: o.weights.clone(),
        tokenizer: o.tokenizer.clone(),
        gpu: o.gpu,
        ..Default::default()
    };
    let inputs = crate::Inputs::resolve(given, cfg, target)?;
    let w = Workload::read(&o.workload)?;
    // Before any runtime loads: CUPTI picks its timestamps once per process.
    trace::start()?;
    let json_bytes = std::fs::read(&inputs.manifest)?;
    // A vLLM manifest runs as vLLM runs it: bench is its host, and hands
    // the host states to the runtime in their declared layouts.
    let m = Verified::from_json(std::str::from_utf8(&json_bytes)?)?.self_hosted()?;
    let protocol = Protocol::check_unsampled(&m)?;
    let unit = kern_pool::page_unit(&m);
    let plan = Plan::check(&w, &protocol, unit as usize, kern_pool::row_tokens(&m, unit))?;
    let mut bench = Bench::load(&m, &inputs, protocol, &plan)?;
    bench.corpus = corpus(inputs.tokenizer()?, w.seed)?;
    bench.samples = w.samples;
    let at = Instant::now();
    bench.prime(plan.source)?;
    say_timed("source", format!("{} tokens of real prefix per rank", plan.source), at);

    let started = Instant::now();
    let probe = &bench.ranks[0].probe;
    let ranks = match bench.ranks.len() {
        1 => String::new(),
        n => format!(" · rank 0 of {n}"),
    };
    let hes = if probe.hes { "HES" } else { "software timestamps" };
    say(
        "device",
        format!("{} · {} SMs · L2 {} MiB · {hes}{ranks}", probe.device, probe.sm_count, probe.l2_bytes >> 20),
    );
    say(
        "plan",
        format!(
            "{} scenarios · {} dropped · {} tokens of state · {} seqs",
            plan.scenarios.len(),
            plan.dropped.len(),
            plan.tokens,
            plan.seqs
        ),
    );
    for d in &plan.dropped {
        say("drop", format!("{} · {}", d.shape, d.why));
    }

    let mut report = preamble(&m, &json_bytes, &w, &plan, probe, bench.ranks.len())?;
    write(&o.out, &report)?;

    let mut runs = Vec::new();
    for s in &plan.scenarios {
        let (records, priced) = bench.scenario(s)?;
        report["scenarios"].as_array_mut().unwrap().extend(records);
        write(&o.out, &report)?;
        runs.push(priced);
    }
    let (total_us, by_op) = cost(&runs);
    let calls: u64 = w.shapes().iter().map(|s| s.weight).sum();
    let dropped: u64 = plan.dropped.iter().map(|d| d.weight).sum();
    let unpriced = match dropped {
        0 => String::new(),
        n => format!(" · {n} dropped"),
    };
    say("cost", format!("{:.3} s · {} calls{unpriced} · {}", total_us / 1e6, calls - dropped, mix_line(&by_op)));
    report["cost"] = json!({"total_us":total_us,"calls":calls - dropped,"dropped_calls":dropped,
        "by_op":by_op.iter().map(|(op, f)| json!({"op":op,"share":f})).collect::<Vec<_>>()});
    write(&o.out, &report)?;
    say_timed("out", o.out.display().to_string(), started);
    Ok(())
}

fn write(out: &std::path::Path, report: &Value) -> Result<()> {
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Ok(std::fs::write(out, serde_json::to_vec(report)?)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn priced(shape: Shape, us: f64, shares: Vec<(String, f64)>) -> Priced {
        Priced { shape, us, shares }
    }

    #[test]
    fn cost_is_every_shapes_time_at_its_weight() {
        let shape = |rows, weight| Shape { groups: 1, rows, chunk: false, context: vec![0], weight };
        let runs = [
            priced(shape(1, 10), 2., vec![("gemm".into(), 1.)]),
            priced(shape(64, 2), 5., vec![("gemm".into(), 0.5), ("attn".into(), 0.5)]),
        ];
        let (total, by_op) = cost(&runs);
        assert_eq!((total, by_op), (30., vec![("gemm".into(), 25. / 30.), ("attn".into(), 5. / 30.)]));
    }

    #[test]
    fn shares_are_of_the_traced_total_and_sorted() {
        let by = mix(&[("gemm".into(), 30.), ("attn".into(), 50.), ("gemm".into(), 20.)]);
        assert_eq!(by, [("attn".to_string(), 0.5), ("gemm".to_string(), 0.5)]);
        assert_eq!(mix(&[]), []);
    }

    #[test]
    fn the_mix_line_names_a_few_ops_and_counts_the_rest() {
        let shares: Vec<(String, f64)> = (0..SHOW + 2).map(|i| (format!("op{i}"), 0.1)).collect();
        assert!(mix_line(&shares).ends_with("op5 10.0% · +2 more"));
        assert_eq!(mix_line(&shares[..2]), "op0 10.0% · op1 10.0%");
        assert_eq!(mix_line(&[]), "");
    }

    fn step(nodes: u32, spans: &[(Option<u32>, u64, u64)]) -> Step {
        let issue = trace::Issue {
            program: "decode".into(),
            vars: Vec::new(),
            graph: true,
            kv: None,
            kernels: std::sync::Arc::new((0..nodes).map(|n| (n, n)).collect()),
        };
        let spans =
            spans.iter().map(|&(call, start, end)| trace::Span { call, name: "k".into(), start, end }).collect();
        Step { seq: 1, issue, spans }
    }

    #[test]
    fn a_sample_spans_its_kernels_and_charges_each_to_its_call() {
        let s = step(3, &[(Some(0), 1_000, 3_000), (Some(1), 2_500, 4_000), (Some(0), 9_000, 10_000)]);
        assert_eq!(timing(&s, 3).unwrap(), (9., vec![3., 1.5, 0.]));
    }

    #[test]
    fn a_sample_missing_kernels_or_naming_no_call_is_refused() {
        assert!(timing(&step(3, &[(Some(0), 0, 1), (Some(1), 1, 2)]), 2).is_err());
        assert!(timing(&step(1, &[(None, 0, 1)]), 1).is_err());
        assert!(timing(&step(1, &[(Some(4), 0, 1)]), 1).is_err());
        assert!(timing(&step(0, &[]), 1).is_err());
    }

    #[test]
    fn tails_and_order_are_preserved() {
        let s = stats(&[1., 1., 1., 1., 1., 1., 1., 1., 10., 30., 50., 100.]);
        assert_eq!((s.n, s.max, s.p50), (12, 100., 1.));
        assert!(s.tail_ratio > 40. && s.cv > 1.);
        assert!(s.block_medians[3] > s.block_medians[0]);
    }
}
