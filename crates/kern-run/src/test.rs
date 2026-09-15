//! `kern test`: the evidence for a kernel swap, over the loaded runtime.
//!
//! The harness itself is `kern-test` (`docs/test.md`): the static diff,
//! the seeded workload recorded on A, replayed on B, noise floor, fuzz,
//! perf and the verdict, written over [`kern_test::Side`]. This module
//! resolves the flags against `kern.toml`, loads A, records, drops A,
//! loads B, replays; and prints or archives the report. The side is
//! [`Ranks`]: the manifest on one GPU, or on one GPU per rank of its
//! topology, each rank a [`Caller`] over its own runtime.

use std::collections::BTreeMap;
use std::ops::Range;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use crate::config::{Config, Target};
use crate::{Caller, Weights};
use anyhow::{bail, Context, Result};
use clap::Args;
use kern_manifest::types::Provision;
use kern_manifest::{Protocol, Verified};
use kern_runtime::{Capacity, GroupRank, HostWeights, PeerHandle, Runtime, Scratch, Topology};
use kern_test::report::{plural, row, Report, Verdict};
use kern_test::{Options, Side, Vars};
use serde_json::json;

/// Flags of `kern test`; anything not given comes from the target /
/// `[test]` in kern.toml, then from the defaults.
#[derive(Args, Clone)]
pub struct TestOpts {
    /// Reference manifest A (assumed correct)
    #[arg(long)]
    pub reference: Option<PathBuf>,
    /// Candidate manifest B
    #[arg(long)]
    manifest: Option<PathBuf>,
    /// Directory of cubins for both manifests; steps resolve by their pinned
    /// sha256, so one dir holds every version (file names are labels)
    #[arg(long)]
    kernels: Option<PathBuf>,
    /// Checkpoint directories or .safetensors files, tensors bound by name
    /// across all of them
    #[arg(long)]
    weights: Vec<String>,
    /// HF tokenizer.json (only needed with --prompt)
    #[arg(long)]
    tokenizer: Option<PathBuf>,
    /// Real-text prompt for the tap's prefill instead of seeded random
    /// tokens (decode tokens stay seeded)
    #[arg(long)]
    prompt: Option<String>,
    /// Prefill length in tokens; 0 = drawn from the seed: half the time
    /// uniform in [1, capacity − steps], half the time a structural
    /// boundary (a page, a chunk, the `tokens` max, ±1)
    #[arg(long, default_value_t = 0)]
    prefill: u64,
    /// Decode steps; the seed picks a length in [steps/2, steps] (default 32)
    #[arg(long)]
    decode_steps: Option<u64>,
    /// How far the end-to-end logits may move, in ulps at the row's scale
    /// (A's max |logit|), and still PASS (with logit evidence), provided
    /// the argmax agrees except at near-ties (default 4)
    #[arg(long)]
    logit_ulp: Option<u64>,
    /// Fuzz rounds per span (0 disables); rounds cycle through the
    /// perturbations of the tapped inputs (jitter, noise, scale, shuffle,
    /// resample, outliers) (default 6)
    #[arg(long)]
    fuzz: Option<usize>,
    /// CUDA device ordinals, one per rank of the manifest's topology (a
    /// single ordinal starts the ranks there, consecutively); default 0
    #[arg(long, value_delimiter = ',')]
    gpu: Vec<usize>,
    /// State capacity in tokens; rounded down to the manifest's page unit
    /// (default 4096)
    #[arg(long)]
    capacity: Option<u64>,
    /// Prefill chunk; 0 = drawn from the seed among the `tokens` max, 512,
    /// a page and a random size
    #[arg(long, default_value_t = 0)]
    chunk: u64,
    /// Replays for span timing (minimum is reported)
    #[arg(long, default_value_t = 20)]
    iters: usize,
    /// Skip capturing both decode programs as CUDA graphs for the step time
    #[arg(long)]
    no_graph_step: bool,
    /// Skip sweeping prefill over the `tokens` var range
    #[arg(long)]
    no_sweep: bool,
    /// Device peak memory bandwidth in GB/s, for the roofline column
    #[arg(long, default_value_t = 8000.0)]
    peak_bw: f64,
    /// Write the test report as JSON here (a directory when several
    /// targets run: one file per target)
    #[arg(long)]
    out: Option<PathBuf>,
    /// Only print the structural diff
    #[arg(long)]
    diff_only: bool,
    /// Skip the timing section
    #[arg(long)]
    no_perf: bool,
    /// Skip the noise-floor re-runs
    #[arg(long)]
    no_noise: bool,
    /// Seed for the workload and the fuzz generator (default 0x5eed)
    #[arg(long)]
    seed: Option<u64>,
    /// Print the report as one JSON object instead of text lines
    #[arg(long)]
    json: bool,
}

/// Resolved options: flag, else kern.toml, else default.
struct Opts {
    a: PathBuf,
    b: PathBuf,
    kernels: PathBuf,
    weights: Weights,
    tokenizer: Option<PathBuf>,
    prompt: Option<String>,
    gpus: Vec<usize>,
    capacity: u64,
    out: Option<PathBuf>,
    diff_only: bool,
    json: bool,
    harness: Options,
}

impl TestOpts {
    fn resolve(self, cfg: Option<&Config>, t: Option<&Target>) -> Result<Opts> {
        let need = |what: &str| match cfg {
            Some(c) => anyhow::anyhow!("no --{what}, and the target in {} does not give one", c.path.display()),
            None => anyhow::anyhow!("no --{what}, and no {} found at or above the cwd", crate::config::FILE),
        };
        let test = cfg.map(|c| &c.test);
        let entries = if self.weights.is_empty() {
            t.map(|t| t.weights.clone()).filter(|w| !w.is_empty()).ok_or_else(|| need("weights"))?
        } else {
            self.weights
        };
        let weights = Weights::parse(&entries)?;
        let gpus = match self.gpu.is_empty() {
            true => vec![cfg.and_then(|c| c.gpu).unwrap_or(0)],
            false => self.gpu,
        };
        let tokenizer = match self.tokenizer.or_else(|| t.and_then(|t| t.tokenizer.clone())) {
            Some(tk) => Some(tk),
            None => crate::checkpoint(&weights.dirs()).tokenizer,
        };
        let a = self.reference.or_else(|| t.and_then(|t| t.reference.clone())).ok_or_else(|| need("reference"))?;
        let b = self.manifest.or_else(|| t.map(|t| t.manifest.clone())).ok_or_else(|| need("manifest"))?;
        let harness = Options {
            a: a.display().to_string(),
            b: b.display().to_string(),
            prompt: None,
            prefill: self.prefill,
            decode_steps: self.decode_steps.or_else(|| test.and_then(|x| x.decode_steps)).unwrap_or(32),
            logit_ulp: self.logit_ulp.or_else(|| test.and_then(|x| x.logit_ulp)).unwrap_or(4),
            fuzz: self.fuzz.or_else(|| test.and_then(|x| x.fuzz)).unwrap_or(6),
            chunk: self.chunk,
            iters: self.iters,
            graph_step: !self.no_graph_step,
            sweep: !self.no_sweep,
            peak_bw: self.peak_bw,
            perf: !self.no_perf,
            noise: !self.no_noise,
            seed: self.seed.or_else(|| test.and_then(|x| x.seed)).unwrap_or(0x5eed),
        };
        Ok(Opts {
            a,
            b,
            kernels: self.kernels.or_else(|| t.map(|t| t.kernels.clone())).ok_or_else(|| need("kernels"))?,
            weights,
            tokenizer,
            prompt: self.prompt.or_else(|| test.and_then(|x| x.prompt.clone())),
            gpus,
            capacity: self.capacity.or_else(|| cfg.and_then(|c| c.capacity)).unwrap_or(4096),
            out: self.out,
            diff_only: self.diff_only,
            json: self.json,
            harness,
        })
    }
}

/// `kern test`: returns the exit code (0 PASS, 1 FAIL, 2 INCONCLUSIVE).
pub fn run(o: TestOpts, cfg: Option<&Config>, target: Option<&Target>) -> Result<i32> {
    execute(o.resolve(cfg, target)?)
}

/// The side: the manifest loaded once per rank, each on its GPU, every
/// rank fed the same sequence. Bytes move from the calling thread (every
/// runtime entry point rebinds its context); whatever runs fans out to
/// one thread per rank, so a collective inside the range finds its peers
/// issuing, and returns when the slowest rank has.
struct Ranks {
    ranks: Vec<Caller>,
}

/// The `Runtime` holds raw CUDA handles; it is used from one thread at a
/// time and binds its context on every entry, so loading it on one thread
/// and moving it once, or lending it to a thread for one call that is
/// joined before the borrow ends, is sound.
struct Sent(Runtime);
#[allow(unsafe_code)]
unsafe impl Send for Sent {}
struct Lent<'a>(&'a mut Caller);
#[allow(unsafe_code)]
unsafe impl Send for Lent<'_> {}

/// A rank that has not returned in this long is hung: a collective
/// waiting for a peer that failed. Nothing in the process can go on.
const HUNG: Duration = Duration::from_secs(600);

impl Ranks {
    /// `f` on every rank at once; the results in rank order, the first
    /// error if any.
    fn each<T: Send>(&mut self, what: &str, f: impl Fn(&mut Caller) -> Result<T> + Sync) -> Result<Vec<T>> {
        let n = self.ranks.len();
        let (tx, rx) = std::sync::mpsc::channel();
        let mut out: Vec<Option<Result<T>>> = (0..n).map(|_| None).collect();
        std::thread::scope(|s| {
            for (q, r) in self.ranks.iter_mut().enumerate() {
                let (lent, tx, f) = (Lent(r), tx.clone(), &f);
                s.spawn(move || {
                    let lent = lent;
                    let _ = tx.send((q, f(lent.0)));
                });
            }
            drop(tx);
            for _ in 0..n {
                match rx.recv_timeout(HUNG) {
                    Ok((q, r)) => out[q] = Some(r),
                    Err(_) => {
                        let hung: Vec<String> =
                            out.iter().enumerate().filter(|(_, r)| r.is_none()).map(|(q, _)| q.to_string()).collect();
                        eprintln!(
                            "rank {} did not return within {}s at {what}: a collective waiting for a peer that failed",
                            hung.join(", "),
                            HUNG.as_secs()
                        );
                        std::process::exit(2);
                    }
                }
            }
        });
        out.into_iter()
            .enumerate()
            .map(|(q, r)| r.expect("every rank replied").with_context(|| format!("rank {q}: {what}")))
            .collect()
    }

    fn rt(&self, rank: usize) -> &Runtime {
        &self.ranks[rank].rt
    }
    fn rt_mut(&mut self, rank: usize) -> &mut Runtime {
        &mut self.ranks[rank].rt
    }
}

impl Side for Ranks {
    type Buf = Scratch;

    fn manifest(&self) -> &Verified {
        &self.rt(0).manifest
    }
    fn provision(&self) -> Provision {
        self.rt(0).provision()
    }
    fn page(&self) -> u64 {
        self.rt(0).page()
    }
    fn vocab(&self) -> u64 {
        self.ranks[0].vocab()
    }
    fn ranks(&self) -> usize {
        self.ranks.len()
    }
    fn stage(&mut self, ids: &[i64]) -> Result<Vars> {
        let vars = self.ranks.iter_mut().map(|c| c.stage(ids)).collect::<Result<Vec<_>>>()?;
        Ok(vars.into_iter().next().expect("a side has a rank"))
    }
    fn reset(&mut self) {
        self.ranks.iter_mut().for_each(Caller::reset)
    }
    fn advance(&mut self, n: u64) {
        self.ranks.iter_mut().for_each(|c| c.advance(n))
    }
    fn calls(&self, program: &str) -> Result<usize> {
        Ok(self.rt(0).call_count(program)?)
    }
    fn run(&mut self, program: &str, vars: &Vars, calls: Range<usize>) -> Result<()> {
        let what = format!("`{program}` calls {}..{}", calls.start, calls.end);
        self.each(&what, |c| Ok(c.rt.run_range(program, vars, calls.start, calls.end)?)).map(drop)
    }
    fn read(&self, rank: usize, buffer: &str, bytes: usize) -> Result<Vec<u8>> {
        self.rt(rank).read_buffer_prefix(buffer, bytes).with_context(|| format!("rank {rank}: reading `{buffer}`"))
    }
    fn write(&mut self, rank: usize, buffer: &str, bytes: &[u8]) -> Result<()> {
        self.rt_mut(rank).write_buffer(buffer, bytes).with_context(|| format!("rank {rank}: writing `{buffer}`"))
    }
    fn alloc(&self, rank: usize, bytes: usize) -> Result<Scratch> {
        self.rt(rank).scratch(bytes).with_context(|| format!("rank {rank}: scratch of {bytes} bytes"))
    }
    fn save(&self, rank: usize, buffer: &str, bytes: usize, into: &mut Scratch) -> Result<()> {
        self.rt(rank).save_buffer(buffer, bytes, into).with_context(|| format!("rank {rank}: saving `{buffer}`"))
    }
    fn load(&mut self, rank: usize, buffer: &str, bytes: usize, from: &Scratch) -> Result<()> {
        self.rt_mut(rank).load_buffer(buffer, bytes, from).with_context(|| format!("rank {rank}: loading `{buffer}`"))
    }
    fn bytes(&self, rank: usize, from: &Scratch, len: usize) -> Result<Vec<u8>> {
        self.rt(rank).read_scratch(from, len).with_context(|| format!("rank {rank}: reading {len} bytes of scratch"))
    }
    fn state_bytes(&self, state: &str) -> Result<usize> {
        Ok(self.rt(0).state_bytes(state)?)
    }
    fn read_state(&self, rank: usize, state: &str, at: Range<usize>) -> Result<Vec<u8>> {
        self.rt(rank)
            .read_state_at(state, at.start, at.len())
            .with_context(|| format!("rank {rank}: reading state `{state}` at {at:?}"))
    }
    fn write_state(&mut self, rank: usize, state: &str, at: usize, bytes: &[u8]) -> Result<()> {
        self.rt_mut(rank)
            .write_state_at(state, at, bytes)
            .with_context(|| format!("rank {rank}: writing state `{state}` at {at}"))
    }
    fn save_state(&self, rank: usize, state: &str, into: &mut Scratch) -> Result<()> {
        self.rt(rank).save_state(state, into).with_context(|| format!("rank {rank}: saving state `{state}`"))
    }
    fn load_state(&mut self, rank: usize, state: &str, from: &Scratch) -> Result<()> {
        self.rt_mut(rank).load_state(state, from).with_context(|| format!("rank {rank}: loading state `{state}`"))
    }
    fn zero_states(&mut self) -> Result<()> {
        self.ranks
            .iter_mut()
            .enumerate()
            .try_for_each(|(q, c)| c.rt.zero_states().with_context(|| format!("rank {q}: zeroing states")))
    }
    fn time(&mut self, program: &str, vars: &Vars, calls: Range<usize>, iters: usize) -> Result<Vec<f32>> {
        let what = format!("timing `{program}` calls {}..{}", calls.start, calls.end);
        let per_rank = self.each(&what, |c| Ok(c.rt.time_range(program, vars, calls.start, calls.end, iters)?))?;
        // the slowest rank per call: what a step waits for
        Ok((0..calls.len()).map(|i| per_rank.iter().map(|t| t[i]).fold(0.0, f32::max)).collect())
    }
    fn capture(&mut self, program: &str, vars: &Vars) -> Result<()> {
        self.each(&format!("capturing `{program}`"), |c| Ok(c.rt.capture(program, vars)?)).map(drop)
    }
    fn time_captured(&mut self, program: &str, vars: &Vars, iters: usize) -> Result<f32> {
        let t =
            self.each(&format!("timing captured `{program}`"), |c| Ok(c.rt.time_captured(program, vars, iters)?))?;
        Ok(t.into_iter().fold(0.0, f32::max))
    }
}

/// Ranks a manifest runs as: the size its topology groups share; 1
/// without a topology.
fn ranks_of(m: &Verified) -> Result<usize> {
    let sizes: Vec<u64> = m.topology.iter().flat_map(|t| t.groups.values().copied()).collect();
    match sizes.as_slice() {
        [] => Ok(1),
        [n, rest @ ..] if rest.iter().all(|r| r == n) => Ok(*n as usize),
        _ => bail!("the manifest's topology groups differ in size; kern test spans every group with all its ranks"),
    }
}

/// The GPU of each rank: the list as given, or `n` from the one given.
fn gpus_of(given: &[usize], n: usize) -> Result<Vec<usize>> {
    match given {
        [g] => Ok((*g..*g + n).collect()),
        gs if gs.len() == n => Ok(gs.to_vec()),
        gs => bail!("the manifest runs as {n} ranks; --gpu names {} devices", gs.len()),
    }
}

/// Load the manifest on every rank's GPU, bind each rank's weights,
/// connect the peers, run what the manifest runs once.
fn load_side(m: &Verified, o: &Opts) -> Result<Ranks> {
    let n = ranks_of(m)?;
    let gpus = gpus_of(&o.gpus, n)?;
    let topology = |q: usize| Topology {
        groups: m
            .topology
            .iter()
            .flat_map(|t| &t.groups)
            .map(|(g, &size)| (g.clone(), GroupRank { index: q as u64, size }))
            .collect(),
    };
    // The test drives one sequence (`Caller` leases one and writes its
    // lines into every table column), so one slot is all the per-sequence
    // states need. Sizing them for the manifest's whole batch made every
    // whole-state read of a span copy 128 slots: on qwen3.8-27b (154 MB of
    // GDN state per slot) a run grew past 700 GB of host memory.
    let capacity = Capacity { tokens: Some(o.capacity), seqs: 1 };
    let host_weights = HostWeights::new();
    let loaded: Vec<Result<Sent>> = std::thread::scope(|s| {
        let handles: Vec<_> = gpus
            .iter()
            .enumerate()
            .map(|(q, &gpu)| {
                let (topo, host_weights) = (topology(q), &host_weights);
                s.spawn(move || -> Result<Sent> {
                    let mut rt = Runtime::load_with_host_weights(
                        m,
                        &o.kernels,
                        gpu,
                        Some(capacity),
                        m.topology.is_some().then_some(&topo),
                        host_weights,
                    )
                    .with_context(|| format!("rank {q} on gpu {gpu}"))?;
                    o.weights.bind(&mut rt, &topo).with_context(|| format!("rank {q}: binding weights"))?;
                    Ok(Sent(rt))
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("a rank's load thread panicked")).collect()
    });
    let mut rts: Vec<Runtime> = loaded.into_iter().map(|r| r.map(|s| s.0)).collect::<Result<_>>()?;
    if let Some(topo) = &m.topology {
        let handles: Vec<BTreeMap<String, PeerHandle>> =
            rts.iter().map(Runtime::export_handles).collect::<Result<_, _>>()?;
        for g in topo.groups.keys() {
            for (q, rt) in rts.iter_mut().enumerate() {
                rt.import_peers(g, &handles).with_context(|| format!("rank {q}: peers of `{g}`"))?;
            }
        }
        for (q, rt) in rts.iter().enumerate() {
            let pending = rt.pending_peers();
            if !pending.is_empty() {
                bail!("rank {q}: peer buffers {pending:?} still unfilled after every group was imported");
            }
        }
    }
    let protocol = Protocol::check(m)?;
    let vars = protocol.vars(1, 1, 1);
    for p in &protocol.once {
        for (q, rt) in rts.iter().enumerate() {
            rt.run(p, &vars).with_context(|| format!("rank {q}: `{p}`"))?;
        }
    }
    Ok(Ranks { ranks: rts.into_iter().map(Caller::new).collect::<Result<_>>()? })
}

fn tokens_of(tok: &tokenizers::Tokenizer, s: &str) -> Result<Vec<i64>> {
    let ids: Vec<i64> =
        tok.encode(s, false).map_err(|e| anyhow::anyhow!("encode: {e}"))?.get_ids().iter().map(|&u| u as i64).collect();
    if ids.is_empty() {
        bail!("prompt is empty: {s:?}");
    }
    Ok(ids)
}

/// Text streams a section's lines as it finishes; JSON waits for the end.
struct Out {
    json: bool,
}

impl Out {
    fn show(&self, lines: &[String]) {
        if !self.json {
            for l in lines {
                println!("{l}");
            }
        }
    }
}

/// Archive, verdict line, JSON: the end of every path through `execute`.
fn finish(o: &Opts, report: Report) -> Result<i32> {
    let out = Out { json: o.json };
    if let Some(p) = &o.out {
        std::fs::write(p, serde_json::to_string_pretty(&json!({"summary": report.summary, "detail": report.detail}))?)?;
        out.show(&[row("out", p.display().to_string(), None)]);
    }
    // A structural diff alone has no verdict; identical programs do (a
    // no-op swap passes).
    let sum = &report.summary;
    if sum.tap.is_some() || sum.diff.programs.is_empty() {
        out.show(&sum.verdict.lines());
    }
    if o.json {
        println!("{}", serde_json::to_string_pretty(sum)?);
    }
    Ok(report.code())
}

fn execute(mut o: Opts) -> Result<i32> {
    let t_start = Instant::now();
    let ja = std::fs::read_to_string(&o.a).with_context(|| format!("reading {}", o.a.display()))?;
    let jb = std::fs::read_to_string(&o.b).with_context(|| format!("reading {}", o.b.display()))?;
    let ma = Verified::from_json(&ja).with_context(|| format!("A ({}) failed verification", o.a.display()))?;
    let mb = Verified::from_json(&jb).with_context(|| format!("B ({}) failed verification", o.b.display()))?;
    Protocol::check(&ma).with_context(|| format!("A ({}) does not fit the serving protocol", o.a.display()))?;
    Protocol::check(&mb).with_context(|| format!("B ({}) does not fit the serving protocol", o.b.display()))?;
    let out = Out { json: o.json };
    let (a, b) = (o.harness.a.clone(), o.harness.b.clone());
    out.show(&[row("kern test", format!("A {a} → B {b}"), None)]);

    // ---- 1. static diff
    let diff = kern_test::diff::diff(&ma, &mb);
    out.show(&diff.lines());
    let verdict = |code: i32, summary: &str| Verdict::new(code, summary.into(), t_start.elapsed().as_secs_f32());
    if o.diff_only {
        return finish(&o, Report::of_diff(&a, &b, diff, verdict(0, "structural diff only, nothing run")));
    }
    if diff.spans.is_empty() {
        return finish(&o, Report::of_diff(&a, &b, diff, verdict(0, "nothing to test: the programs are identical")));
    }
    o.harness.prompt = match &o.prompt {
        Some(text) => {
            let Some(tk) = &o.tokenizer else {
                bail!("--prompt needs a tokenizer (--tokenizer or the target's `tokenizer`)")
            };
            let tokenizer = tokenizers::Tokenizer::from_file(tk).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
            Some(tokens_of(&tokenizer, text)?)
        }
        None => None,
    };

    // ---- 2. A: load, record, unload; B: load, replay. Never both loaded.
    let load = |m: &Verified, side: &str| -> Result<(Ranks, f32)> {
        let t = Instant::now();
        let r = load_side(m, &o).with_context(|| format!("loading {side}"))?;
        let s = t.elapsed().as_secs_f32();
        let gpus = gpus_of(&o.gpus, r.ranks())?;
        out.show(&[row(
            "load",
            format!(
                "{side}: {} on gpu {}",
                plural(r.ranks(), "rank"),
                gpus.iter().map(|g| g.to_string()).collect::<Vec<_>>().join(",")
            ),
            Some(s),
        )]);
        Ok((r, s))
    };
    let (mut side_a, load_a) = load(&ma, "A")?;
    let mut rec = kern_test::record(&o.harness, diff, &mb, &mut side_a, &mut |lines: &[String]| out.show(lines))?;
    drop(side_a);
    let (mut side_b, load_b) = load(&mb, "B")?;
    rec.load_s = load_a + load_b;
    let report = kern_test::replay(&o.harness, rec, &mut side_b, &mut |lines: &[String]| out.show(lines))?;
    finish(&o, report)
}
