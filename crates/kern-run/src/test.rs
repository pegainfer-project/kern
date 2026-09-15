//! `kern test`: the evidence for a kernel swap, over two loaded runtimes.
//!
//! The harness itself is `kern-test` (`docs/test.md`): the static diff,
//! the seeded workload, the span-local tap, noise floor, fuzz, perf and
//! the verdict, written over [`kern_test::Side`]. This module resolves the
//! flags against `kern.toml`, loads A and B, implements `Side` for the
//! runtime-backed [`Caller`], and prints or archives the report.

use std::ops::Range;
use std::path::PathBuf;
use std::time::Instant;

use crate::config::{Config, Target};
use crate::{Caller, Weights};
use anyhow::{bail, Context, Result};
use clap::Args;
use kern_manifest::types::Provision;
use kern_manifest::{Protocol, Verified};
use kern_runtime::{Capacity, Runtime, Scratch, Topology};
use kern_test::report::{row, Report, Verdict};
use kern_test::{Options, Side, Sides, Vars};
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
    /// CUDA device ordinal (default 0)
    #[arg(long)]
    gpu: Option<usize>,
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
    gpu: usize,
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
        let gpu = self.gpu.or_else(|| cfg.and_then(|c| c.gpu)).unwrap_or(0);
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
            gpu,
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

impl Side for Caller {
    type Buf = Scratch;

    fn manifest(&self) -> &Verified {
        &self.rt.manifest
    }
    fn provision(&self) -> Provision {
        self.rt.provision()
    }
    fn page(&self) -> u64 {
        self.rt.page()
    }
    fn vocab(&self) -> u64 {
        Caller::vocab(self)
    }
    fn stage(&mut self, ids: &[i64]) -> Result<Vars> {
        Caller::stage(self, ids)
    }
    fn reset(&mut self) {
        Caller::reset(self)
    }
    fn advance(&mut self, n: u64) {
        Caller::advance(self, n)
    }
    fn calls(&self, program: &str) -> Result<usize> {
        Ok(self.rt.call_count(program)?)
    }
    fn run(&mut self, program: &str, vars: &Vars, calls: Range<usize>) -> Result<()> {
        Ok(self.rt.run_range(program, vars, calls.start, calls.end)?)
    }
    fn read(&self, buffer: &str, bytes: usize) -> Result<Vec<u8>> {
        Ok(self.rt.read_buffer_prefix(buffer, bytes)?)
    }
    fn write(&mut self, buffer: &str, bytes: &[u8]) -> Result<()> {
        Ok(self.rt.write_buffer(buffer, bytes)?)
    }
    fn alloc(&self, bytes: usize) -> Result<Scratch> {
        Ok(self.rt.scratch(bytes)?)
    }
    fn save(&self, buffer: &str, bytes: usize, into: &mut Scratch) -> Result<()> {
        Ok(self.rt.save_buffer(buffer, bytes, into)?)
    }
    fn load(&mut self, buffer: &str, bytes: usize, from: &Scratch) -> Result<()> {
        Ok(self.rt.load_buffer(buffer, bytes, from)?)
    }
    fn bytes(&self, from: &Scratch, len: usize) -> Result<Vec<u8>> {
        Ok(self.rt.read_scratch(from, len)?)
    }
    fn state_bytes(&self, state: &str) -> Result<usize> {
        Ok(self.rt.state_bytes(state)?)
    }
    fn read_state(&self, state: &str, at: Range<usize>) -> Result<Vec<u8>> {
        Ok(self.rt.read_state_at(state, at.start, at.len())?)
    }
    fn write_state(&mut self, state: &str, at: usize, bytes: &[u8]) -> Result<()> {
        Ok(self.rt.write_state_at(state, at, bytes)?)
    }
    fn save_state(&self, state: &str, into: &mut Scratch) -> Result<()> {
        Ok(self.rt.save_state(state, into)?)
    }
    fn load_state(&mut self, state: &str, from: &Scratch) -> Result<()> {
        Ok(self.rt.load_state(state, from)?)
    }
    fn zero_states(&mut self) -> Result<()> {
        Ok(self.rt.zero_states()?)
    }
    fn time(&mut self, program: &str, vars: &Vars, calls: Range<usize>, iters: usize) -> Result<Vec<f32>> {
        Ok(self.rt.time_range(program, vars, calls.start, calls.end, iters)?)
    }
    fn capture(&mut self, program: &str, vars: &Vars) -> Result<()> {
        Ok(self.rt.capture(program, vars)?)
    }
    fn time_captured(&mut self, program: &str, vars: &Vars, iters: usize) -> Result<f32> {
        Ok(self.rt.time_captured(program, vars, iters)?)
    }
}

fn load_side(m: &Verified, o: &Opts) -> Result<Caller> {
    // The test drives one sequence (`Caller` leases one and writes its
    // lines into every table column), so one slot is all the per-sequence
    // states need. Sizing them for the manifest's whole batch made every
    // whole-state read of a span copy 128 slots: on qwen3.8-27b (154 MB of
    // GDN state per slot) a run grew past 700 GB of host memory.
    let seqs = 1;
    let mut rt = Runtime::load(m, &o.kernels, o.gpu, Some(Capacity { tokens: Some(o.capacity), seqs }), None)?;
    o.weights.bind(&mut rt, &Topology::default())?;
    Caller::new(rt)
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

    // ---- load A + B
    let t = Instant::now();
    let mut sides = Sides { a: load_side(&ma, &o)?, b: load_side(&mb, &o)?, load_s: 0.0 };
    sides.load_s = t.elapsed().as_secs_f32();
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
    let report = kern_test::run(&o.harness, diff, &mut sides, &mut |lines: &[String]| out.show(lines))?;
    finish(&o, report)
}
