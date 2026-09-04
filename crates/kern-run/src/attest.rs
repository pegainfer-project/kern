//! `kern test`: evidence for a kernel swap (the attestation).
//!
//! Given two manifests A (the reference — assumed correct) and B (the
//! candidate), attest:
//!   1. diffs them structurally: which kernels changed (interface / impl /
//!      added / removed) and, per program, the aligned call segments
//!      that differ ("cuts") — everything else is shared;
//!   2. taps a seeded workload once (random tokens; prefill length, chunk
//!      and decode steps drawn from the seed, biased to page/chunk/max
//!      boundaries — or `--prompt` text): A and B run in lockstep; each
//!      program run B starts from A's state, at every cut B gets A's
//!      frontier inputs, so what B writes is the cut's own doing
//!      (cut-local); A's inputs/outputs are snapshotted. Then B free-runs
//!      the same workload with nothing injected and the **logits** of every
//!      step are compared end-to-end: the oracle. A token flip where A's
//!      own top-1/top-2 margin is below the logit delta is a near-tie, not
//!      an error;
//!   3. measures the noise floor per cut: A's cut re-run from its own
//!      snapshot against its own output;
//!   4. fuzzes each cut from the snapshot: float frontier inputs perturbed
//!      around the tapped values (jitter / noise / scale / shuffle /
//!      resample / outliers), integers kept as tapped — the kernel is
//!      tested in the distribution it was built for; both sides run,
//!      outputs compared and checked against declared domains;
//!   5. times the cuts in isolation (plus, opt-in, graph-level step time
//!      and a prefill var sweep) and computes a static bytes-moved
//!      roofline for the changed kernels.
//!
//! Everything after the tap is cut-local: cost scales with the cut, not the
//! model. End-to-end drift is reported (the free run), not adjudicated:
//! differences beyond bit/value identity are INCONCLUSIVE here.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::time::Instant;

use crate::config::{Config, Target};
use crate::{env, Caller, DECODE_LIKE, DRIVEN, TOKENS};
use anyhow::{bail, Context, Result};
use clap::Args;
use kern_manifest::types::{Arg, BufferKind, Call, DType, Dim, Dir, Manifest, ParamType, Provision};
use kern_manifest::verify;
use kern_runtime::{values, Capacity, Runtime};
use serde_json::{json, Value};

/// Flags of `kern test`; anything not given comes from the target /
/// `[test]` in kern.toml, then from the defaults.
#[derive(Args, Clone)]
pub struct TestOpts {
    /// Reference manifest A (assumed correct)
    #[arg(long)]
    pub reference: Option<PathBuf>,
    /// Candidate manifest B
    #[arg(long)]
    pub manifest: Option<PathBuf>,
    /// Directory of cubins for both manifests; steps resolve by their pinned
    /// sha256, so one dir holds every version (file names are labels)
    #[arg(long)]
    pub kernels: Option<PathBuf>,
    /// Safetensors artifact(s), tensors bound by name across all of them
    #[arg(long)]
    pub weights: Vec<PathBuf>,
    /// HF tokenizer.json (only needed with --prompt)
    #[arg(long)]
    pub tokenizer: Option<PathBuf>,
    /// Real-text prompt for the tap's prefill instead of seeded random
    /// tokens (decode tokens stay seeded)
    #[arg(long)]
    pub prompt: Option<String>,
    /// Prefill length in tokens; 0 = drawn from the seed: half the time
    /// uniform in [1, capacity − steps], half the time a structural
    /// boundary (a page, a chunk, the `tokens` max, ±1)
    #[arg(long, default_value_t = 0)]
    pub prefill: u64,
    /// Decode steps; the seed picks a length in [steps/2, steps] (default 32)
    #[arg(long)]
    pub decode_steps: Option<u64>,
    /// How far the end-to-end logits may move, in ulps at the row's scale
    /// (A's max |logit|), and still PASS (with logit evidence), provided
    /// the argmax agrees except at near-ties (default 4)
    #[arg(long)]
    pub logit_ulp: Option<u64>,
    /// Fuzz rounds per cut (0 disables); rounds cycle through the
    /// perturbations of the tapped inputs (jitter, noise, scale, shuffle,
    /// resample, outliers) (default 6)
    #[arg(long)]
    pub fuzz: Option<usize>,
    /// CUDA device ordinal (default 0)
    #[arg(long)]
    pub gpu: Option<usize>,
    /// State capacity in tokens; rounded down to the manifest's page unit
    /// (default 4096)
    #[arg(long)]
    pub capacity: Option<u64>,
    /// Prefill chunk; 0 = drawn from the seed among the `tokens` max, 512,
    /// a page and a random size
    #[arg(long, default_value_t = 0)]
    pub chunk: u64,
    /// Replays for cut timing (minimum is reported)
    #[arg(long, default_value_t = 20)]
    pub iters: usize,
    /// Skip capturing both decode programs as CUDA graphs for the step time
    #[arg(long)]
    pub no_graph_step: bool,
    /// Skip sweeping prefill over the `tokens` var range
    #[arg(long)]
    pub no_sweep: bool,
    /// Device peak memory bandwidth in GB/s, for the roofline column
    #[arg(long, default_value_t = 8000.0)]
    pub peak_bw: f64,
    /// Write the attestation as JSON here (a directory when several
    /// targets run: one file per target)
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Only print the structural diff
    #[arg(long)]
    pub diff_only: bool,
    /// Skip the timing section
    #[arg(long)]
    pub no_perf: bool,
    /// Skip the noise-floor re-runs
    #[arg(long)]
    pub no_noise: bool,
    /// Seed for the workload and the fuzz generator (default 0x5eed)
    #[arg(long)]
    pub seed: Option<u64>,
    /// Report format on stdout
    #[arg(long, value_enum, default_value_t = Format::Text)]
    pub format: Format,
    /// ANSI color (text format only)
    #[arg(long, value_enum, default_value_t = Color::Auto)]
    pub color: Color,
}

/// Resolved options: flag, else kern.toml, else default.
struct Opts {
    a: PathBuf,
    b: PathBuf,
    kernels: PathBuf,
    weights: Vec<PathBuf>,
    tokenizer: Option<PathBuf>,
    prompt: Option<String>,
    prefill: u64,
    decode_steps: u64,
    logit_ulp: u64,
    fuzz: usize,
    gpu: usize,
    capacity: u64,
    chunk: u64,
    iters: usize,
    no_graph_step: bool,
    no_sweep: bool,
    peak_bw: f64,
    out: Option<PathBuf>,
    diff_only: bool,
    no_perf: bool,
    no_noise: bool,
    seed: u64,
    format: Format,
    color: Color,
}

impl TestOpts {
    fn resolve(self, cfg: Option<&Config>, t: Option<&Target>) -> Result<Opts> {
        let file = cfg.map_or(crate::config::FILE.to_string(), |c| c.path.display().to_string());
        let need = |what: &str| anyhow::anyhow!("no --{what} and no target in {file} to take it from");
        let test = cfg.map(|c| &c.test);
        Ok(Opts {
            a: self.reference.or_else(|| t.and_then(|t| t.reference.clone())).ok_or_else(|| need("reference"))?,
            b: self.manifest.or_else(|| t.map(|t| t.manifest.clone())).ok_or_else(|| need("manifest"))?,
            kernels: self.kernels.or_else(|| t.map(|t| t.kernels.clone())).ok_or_else(|| need("kernels"))?,
            weights: if self.weights.is_empty() {
                t.map(|t| t.weights.clone()).filter(|w| !w.is_empty()).ok_or_else(|| need("weights"))?
            } else {
                self.weights
            },
            tokenizer: self.tokenizer.or_else(|| t.and_then(|t| t.tokenizer.clone())),
            prompt: self.prompt.or_else(|| test.and_then(|x| x.prompt.clone())),
            prefill: self.prefill,
            decode_steps: self.decode_steps.or_else(|| test.and_then(|x| x.decode_steps)).unwrap_or(32),
            logit_ulp: self.logit_ulp.or_else(|| test.and_then(|x| x.logit_ulp)).unwrap_or(4),
            fuzz: self.fuzz.or_else(|| test.and_then(|x| x.fuzz)).unwrap_or(6),
            gpu: self.gpu.or_else(|| cfg.and_then(|c| c.gpu)).unwrap_or(0),
            capacity: self.capacity.or_else(|| cfg.and_then(|c| c.capacity)).unwrap_or(4096),
            chunk: self.chunk,
            iters: self.iters,
            no_graph_step: self.no_graph_step,
            no_sweep: self.no_sweep,
            peak_bw: self.peak_bw,
            out: self.out,
            diff_only: self.diff_only,
            no_perf: self.no_perf,
            no_noise: self.no_noise,
            seed: self.seed.or_else(|| test.and_then(|x| x.seed)).unwrap_or(0x5eed),
            format: self.format,
            color: self.color,
        })
    }
}

/// `kern test`: returns the exit code (0 PASS, 1 FAIL, 2 INCONCLUSIVE).
pub fn run(o: TestOpts, cfg: Option<&Config>, target: Option<&Target>) -> Result<i32> {
    execute(o.resolve(cfg, target)?)
}

/// The tap's workload, fully determined by `(seed, manifest, options)`:
/// anyone can re-run it, and it does not depend on A's numerics (decode
/// tokens are drawn, not A's argmax).
struct Workload {
    prefill: Vec<i64>,
    chunk: usize,
    decode: Vec<i64>,
    vocab: u64,
    how: &'static str,
}

fn sample_workload(o: &Opts, m: &Manifest, p: Provision, page: u64, prompt: Option<Vec<i64>>) -> Result<Workload> {
    let capacity = p.tokens;
    let mut rng = Rng(o.seed ^ 0x776f_726b_6c6f_6164);
    let tmax = m.vars[TOKENS].max.max(1);
    let vocab = m
        .buffers
        .get("token_ids")
        .and_then(|b| b.domain.as_ref())
        .map(|d| d.resolve(m, &env(1), &p))
        .transpose()?
        .and_then(|r| r.hi)
        .map(|hi| hi as u64 + 1)
        .ok_or_else(|| anyhow::anyhow!("`token_ids` has no domain to take the vocabulary from"))?;
    let steps =
        if o.decode_steps <= 1 { 1 } else { o.decode_steps / 2 + rng.below(o.decode_steps - o.decode_steps / 2 + 1) };
    let hi = capacity.saturating_sub(steps).max(1);
    let chunk = if o.chunk > 0 {
        o.chunk
    } else {
        [tmax, 512.min(tmax), page.min(tmax), 1 + rng.below(tmax)][rng.below(4) as usize]
    }
    .clamp(1, tmax);
    let (n_pre, how) = match &prompt {
        Some(p) => {
            anyhow::ensure!(
                p.len() as u64 <= hi,
                "prompt is {} tokens; capacity {capacity} leaves room for {hi} before {steps} decode steps",
                p.len()
            );
            (p.len() as u64, "prompt")
        }
        None if o.prefill > 0 => (o.prefill.min(hi), "given"),
        None if rng.below(2) == 0 => (1 + rng.below(hi), "uniform"),
        None => {
            // where kernels break: the last partial chunk, the first token
            // of a new page, exactly the var max
            let mut c: Vec<u64> = vec![
                1,
                page - 1,
                page,
                page + 1,
                2 * page + 1,
                chunk - 1,
                chunk,
                chunk + 1,
                2 * chunk + 1,
                3 * chunk - 1,
                tmax,
                tmax + 1,
                hi,
            ];
            c.retain(|&x| (1..=hi).contains(&x));
            c.sort_unstable();
            c.dedup();
            (c[rng.below(c.len() as u64) as usize], "boundary")
        }
    };
    let prefill = prompt.unwrap_or_else(|| (0..n_pre).map(|_| rng.below(vocab) as i64).collect());
    let decode = (0..steps).map(|_| rng.below(vocab) as i64).collect();
    Ok(Workload { prefill, chunk: chunk as usize, decode, vocab, how })
}

/// One end-to-end logits comparison: A (lockstep, uninjected) vs B (free
/// run) on one row of a `logits*` buffer after one program run.
struct LogitRow {
    label: String,
    cmp: Cmp,
    max_abs: f64,
    /// `max_abs` in ulps of the row's scale (A's max |logit|): the
    /// granularity the bf16 row is stored at. Element-wise ulps are
    /// meaningless here — a 1-ulp change of the hidden state moves every
    /// logit by about the same absolute amount, which is thousands of
    /// ulps for a logit near zero, and decides nothing.
    scale_ulps: f64,
    scale: f64,
    argmax_a: usize,
    argmax_b: usize,
    /// A's top-1 − top-2: how far the argmax was from flipping on its own
    margin_a: f64,
    kl: f64,
}

impl LogitRow {
    fn flip(&self) -> bool {
        self.argmax_a != self.argmax_b
    }
    /// The delta could have flipped A's own argmax: not B's doing alone.
    fn near_tie(&self) -> bool {
        self.flip() && self.margin_a <= self.max_abs
    }
}

/// Spacing of `dt` at magnitude `x`.
fn ulp_at(dt: DType, x: f64) -> f64 {
    let mant = match dt {
        DType::Bf16 => 7,
        DType::F16 => 10,
        DType::F32 => 23,
        DType::Fp8E4m3 => 3,
        _ => 0,
    };
    let e = if x.abs() > 0.0 && x.is_finite() { x.abs().log2().floor() } else { 0.0 };
    2f64.powf(e - mant as f64)
}

fn logit_row(label: String, dt: DType, a: &[u8], b: &[u8]) -> LogitRow {
    let (va, vb) = (values::to_f64(dt, a), values::to_f64(dt, b));
    let cmp = compare(dt, a, b);
    let max_abs = va.iter().zip(&vb).map(|(x, y)| (x - y).abs()).filter(|d| d.is_finite()).fold(0.0, f64::max);
    let argmax = |v: &[f64]| {
        v.iter().enumerate().fold((0usize, f64::NEG_INFINITY), |m, (i, &x)| if x > m.1 { (i, x) } else { m })
    };
    let (argmax_a, top1) = argmax(&va);
    let top2 = va.iter().enumerate().filter(|(i, _)| *i != argmax_a).map(|(_, &x)| x).fold(f64::NEG_INFINITY, f64::max);
    let (argmax_b, _) = argmax(&vb);
    let lse = |v: &[f64], m: f64| m + v.iter().map(|x| (x - m).exp()).sum::<f64>().ln();
    let (ma_, mb_) = (top1, vb.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    let (la, lb) = (lse(&va, ma_), lse(&vb, mb_));
    let kl = va.iter().zip(&vb).map(|(x, y)| (x - la).exp() * ((x - la) - (y - lb))).sum::<f64>();
    let scale = va.iter().filter(|x| x.is_finite()).fold(0.0f64, |m, x| m.max(x.abs()));
    let scale_ulps = if max_abs == 0.0 {
        0.0
    } else if max_abs.is_finite() {
        max_abs / ulp_at(dt, scale)
    } else {
        f64::INFINITY
    };
    LogitRow {
        label,
        cmp,
        max_abs,
        scale_ulps,
        scale,
        argmax_a,
        argmax_b,
        margin_a: top1 - top2,
        kl: if kl.is_finite() { kl } else { f64::INFINITY },
    }
}

// ---------------------------------------------------------------- static diff

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    Same,
    Changed,
}

#[derive(Clone, Debug)]
struct Segment {
    kind: Kind,
    a: (usize, usize),
    b: (usize, usize),
}

fn op_changes(ma: &Manifest, mb: &Manifest) -> BTreeMap<String, &'static str> {
    let mut out = BTreeMap::new();
    let names: BTreeSet<&String> = ma.ops.keys().chain(mb.ops.keys()).collect();
    for n in names {
        let kind = match (ma.ops.get(n), mb.ops.get(n)) {
            (Some(_), None) => "removed",
            (None, Some(_)) => "added",
            (Some(x), Some(y)) => {
                if json!(x.params) != json!(y.params) {
                    "interface"
                } else if json!(x.imp) != json!(y.imp) {
                    "impl"
                } else {
                    continue;
                }
            }
            (None, None) => unreachable!(),
        };
        out.insert(n.clone(), kind);
    }
    out
}

/// Align two call lists (LCS over canonical call keys; a call of a changed
/// op never matches across sides) into Same/Changed segments.
fn align(pa: &[Call], pb: &[Call], changed: &BTreeMap<String, &str>) -> Vec<Segment> {
    let key = |c: &Call, side: &str| {
        if changed.contains_key(&c.op) {
            format!("{}@{side}#{}", c.op, json!(c.args))
        } else {
            format!("{}#{}", c.op, json!(c.args))
        }
    };
    let ka: Vec<String> = pa.iter().map(|c| key(c, "A")).collect();
    let kb: Vec<String> = pb.iter().map(|c| key(c, "B")).collect();
    let (n, m) = (ka.len(), kb.len());
    let mut lcs = vec![vec![0u32; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            lcs[i][j] = if ka[i] == kb[j] { lcs[i + 1][j + 1] + 1 } else { lcs[i + 1][j].max(lcs[i][j + 1]) };
        }
    }
    let mut pairs = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < n && j < m {
        if ka[i] == kb[j] {
            pairs.push((i, j));
            i += 1;
            j += 1;
        } else if lcs[i + 1][j] >= lcs[i][j + 1] {
            i += 1;
        } else {
            j += 1;
        }
    }
    let mut segs: Vec<Segment> = Vec::new();
    let (mut ia, mut ib) = (0, 0);
    for (i, j) in pairs {
        if i > ia || j > ib {
            segs.push(Segment { kind: Kind::Changed, a: (ia, i), b: (ib, j) });
        }
        match segs.last_mut() {
            Some(s) if s.kind == Kind::Same && s.a.1 == i && s.b.1 == j => {
                s.a.1 = i + 1;
                s.b.1 = j + 1;
            }
            _ => segs.push(Segment { kind: Kind::Same, a: (i, i + 1), b: (j, j + 1) }),
        }
        ia = i + 1;
        ib = j + 1;
    }
    if ia < n || ib < m {
        segs.push(Segment { kind: Kind::Changed, a: (ia, n), b: (ib, m) });
    }
    segs
}

// ------------------------------------------------------------- dataflow view

#[derive(Default, Clone)]
struct Access {
    reads: BTreeSet<String>,
    writes: BTreeSet<String>,
    state_reads: BTreeSet<String>,
    state_writes: BTreeSet<String>,
}

fn access(m: &Manifest, prog: &str, lo: usize, hi: usize) -> Access {
    let mut acc = Access::default();
    for c in &m.programs[prog][lo..hi] {
        let op = &m.ops[&c.op];
        for (arg, p) in c.args.iter().zip(&op.params) {
            match (arg, p) {
                (Arg::Buf { buf, .. }, ParamType::Buf { dir, .. }) => {
                    if matches!(dir, Dir::In | Dir::InOut) {
                        acc.reads.insert(buf.clone());
                    }
                    if matches!(dir, Dir::Out | Dir::InOut) {
                        acc.writes.insert(buf.clone());
                    }
                }
                (Arg::State { state, .. }, ParamType::State { dir }) => {
                    if matches!(dir, Dir::In | Dir::InOut) {
                        acc.state_reads.insert(state.clone());
                    }
                    if matches!(dir, Dir::Out | Dir::InOut) {
                        acc.state_writes.insert(state.clone());
                    }
                }
                _ => {}
            }
        }
    }
    acc
}

/// Buffers a range reads before it writes them: what the cut consumes from
/// outside.
fn frontier_inputs(m: &Manifest, prog: &str, lo: usize, hi: usize) -> BTreeSet<String> {
    let mut written = BTreeSet::new();
    let mut inputs = BTreeSet::new();
    for c in &m.programs[prog][lo..hi] {
        let op = &m.ops[&c.op];
        for (arg, p) in c.args.iter().zip(&op.params) {
            if let (Arg::Buf { buf, .. }, ParamType::Buf { dir, .. }) = (arg, p) {
                if matches!(dir, Dir::In | Dir::InOut) && !written.contains(buf) {
                    inputs.insert(buf.clone());
                }
                if matches!(dir, Dir::Out | Dir::InOut) {
                    written.insert(buf.clone());
                }
            }
        }
    }
    inputs
}

fn live_elems(m: &Manifest, name: &str, e: &BTreeMap<String, u64>) -> usize {
    m.buffers[name]
        .shape
        .iter()
        .map(|d| match d {
            Dim::Const(c) => *c as usize,
            Dim::Var(s) => e[s] as usize,
        })
        .product()
}

fn live_bytes(m: &Manifest, name: &str, e: &BTreeMap<String, u64>) -> usize {
    live_elems(m, name, e) * m.buffers[name].dtype.bytes() as usize
}

fn is_float(dt: DType) -> bool {
    matches!(dt, DType::Bf16 | DType::F16 | DType::F32 | DType::Fp8E4m3)
}

// ---------------------------------------------------------------- comparison

#[derive(Clone, Debug, Default)]
struct Cmp {
    n: usize,
    n_diff: usize,
    max_ulp: Option<u64>,
    max_abs: f64,
    nan_only_one_side: usize,
    /// Bit-different but value-equal: +0 vs -0.
    signed_zero: usize,
}

fn compare(dt: DType, a: &[u8], b: &[u8]) -> Cmp {
    let w = dt.bytes() as usize;
    let mut c = Cmp { n: a.len() / w, ..Default::default() };
    for (x, y) in a.chunks_exact(w).zip(b.chunks_exact(w)) {
        if x == y {
            continue;
        }
        c.n_diff += 1;
        let (fx, fy) = (values::to_f64(dt, x)[0], values::to_f64(dt, y)[0]);
        if fx.is_nan() != fy.is_nan() {
            c.nan_only_one_side += 1;
            continue;
        }
        if fx == fy {
            c.signed_zero += 1; // +0 vs -0: bit-different, value-equal
            continue;
        }
        c.max_abs = c.max_abs.max((fx - fy).abs());
        if is_float(dt) {
            if let Some(u) = values::ulp_distance(dt, x, y) {
                c.max_ulp = Some(c.max_ulp.unwrap_or(0).max(u));
            }
        }
    }
    c
}

impl Cmp {
    fn identical(&self) -> bool {
        self.n_diff == 0
    }
    /// Every difference is a signed zero.
    fn value_identical(&self) -> bool {
        self.n_diff == self.signed_zero
    }
    fn to_json(&self) -> Value {
        json!({"n": self.n, "n_diff": self.n_diff, "max_ulp": self.max_ulp, "max_abs": self.max_abs,
               "nan_mismatch": self.nan_only_one_side, "signed_zero": self.signed_zero})
    }
}

/// Compare the buffers a segment wrote on both sides (live prefix at `e`),
/// plus any state it wrote. Buffers written by one side only are internal
/// to that side's implementation.
fn compare_written(
    a: &Caller,
    b: &Caller,
    acc_a: &Access,
    acc_b: &Access,
    e: &BTreeMap<String, u64>,
    with_states: bool,
) -> Result<(BTreeMap<String, Cmp>, BTreeMap<String, usize>, Vec<String>)> {
    let mut bufs = BTreeMap::new();
    let mut states = BTreeMap::new();
    let mut one_sided = Vec::new();
    for name in acc_a.writes.union(&acc_b.writes) {
        if !(acc_a.writes.contains(name) && acc_b.writes.contains(name)) {
            one_sided.push(name.clone());
            continue;
        }
        let bytes = live_bytes(&a.rt.manifest, name, e);
        let x = a.rt.read_buffer_prefix(name, bytes)?;
        let y = b.rt.read_buffer_prefix(name, bytes)?;
        bufs.insert(name.clone(), compare(a.rt.manifest.buffers[name].dtype, &x, &y));
    }
    if with_states {
        for name in acc_a.state_writes.union(&acc_b.state_writes) {
            let x = a.rt.read_state(name)?;
            let y = b.rt.read_state(name)?;
            let n = x.iter().zip(&y).filter(|(p, q)| p != q).count();
            states.insert(name.clone(), n);
        }
    }
    Ok((bufs, states, one_sided))
}

// ------------------------------------------------------------------- fuzzing

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.unit().max(1e-300), self.unit());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
}

const MODES: [&str; 6] = ["jitter", "noise", "scale", "shuffle", "resample", "outliers"];

/// Perturb a tapped float tensor, staying in the distribution the kernel
/// was built for. Synthetic values were the old way (uniform, N(0,1), …)
/// and produced noise, not signal: a fused GDN kernel looked "6143/6144
/// wrong" under sequence layouts no caller produces. `row` is the trailing
/// extent (elements per leading-dim row) so `shuffle` permutes rows, not
/// elements.
fn perturb(rng: &mut Rng, mode: usize, x: &[f64], row: usize, dt: DType) -> Vec<f64> {
    let max = match dt {
        DType::Bf16 | DType::F32 => 3.0e38,
        DType::F16 => 65504.0,
        DType::Fp8E4m3 => 448.0,
        _ => unreachable!(),
    };
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let mut v: Vec<f64> = match MODES[mode % MODES.len()] {
        // a few low mantissa bits: the same tensor, different rounding paths
        "jitter" => x.iter().map(|&a| a * (1.0 + rng.normal() / 64.0)).collect(),
        // additive noise at 10% of the tensor's own rms
        "noise" => {
            let rms = (x.iter().filter(|a| a.is_finite()).map(|a| a * a).sum::<f64>() / n as f64).sqrt();
            x.iter().map(|&a| a + rng.normal() * 0.1 * rms).collect()
        }
        // dynamic range: the whole tensor ×¼ … ×4
        "scale" => {
            let f = [0.25, 0.5, 2.0, 4.0][rng.below(4) as usize];
            x.iter().map(|&a| a * f).collect()
        }
        // rows in another order: positions change, values don't
        "shuffle" => {
            let row = row.clamp(1, n);
            let rows = n / row;
            let mut perm: Vec<usize> = (0..rows).collect();
            for i in (1..rows).rev() {
                perm.swap(i, rng.below(i as u64 + 1) as usize);
            }
            let mut v = x.to_vec();
            for (dst, &src) in perm.iter().enumerate() {
                v[dst * row..(dst + 1) * row].copy_from_slice(&x[src * row..(src + 1) * row]);
            }
            v
        }
        // bootstrap: the tensor's own marginal, structure destroyed
        "resample" => (0..n).map(|_| x[rng.below(n as u64) as usize]).collect(),
        // 1% of the elements ×16
        _ => x.iter().map(|&a| if rng.below(100) == 0 { a * 16.0 } else { a }).collect(),
    };
    for a in &mut v {
        if a.is_finite() {
            *a = a.clamp(-max, max);
        }
    }
    v
}

/// Elements per row of the leading dimension (1 for a vector).
fn row_elems(m: &Manifest, name: &str, e: &BTreeMap<String, u64>) -> usize {
    m.buffers[name].shape[1..]
        .iter()
        .map(|d| match d {
            Dim::Const(c) => *c as usize,
            Dim::Var(s) => e[s] as usize,
        })
        .product::<usize>()
        .max(1)
}

// ------------------------------------------------------------------- driver

struct Sides {
    a: Caller,
    b: Caller,
}

/// One cut with a real-workload snapshot: what it consumed (frontier inputs)
/// and what A produced from them.
struct Snap {
    program: String,
    seg: Segment,
    env: BTreeMap<String, u64>,
    inputs: Vec<(String, Vec<u8>)>,
    ref_out: BTreeMap<String, Vec<u8>>,
    ref_states: BTreeMap<String, Vec<u8>>,
    /// Pre-image of every state byte the cut changed (A, before the cut):
    /// `state -> [(offset, bytes)]`. A cut with inout state is not
    /// idempotent — replaying it on its own output shifts the conv window
    /// again, advances the SSM again — so every replay first writes these
    /// back, and writes the post-image back when it is done. States are
    /// opaque; this is byte-level, no model knowledge.
    pre_states: BTreeMap<String, Vec<(usize, Vec<u8>)>>,
    /// Which run's state image the replay starts from: 0 = zeros (prefill
    /// chunk 0), 1 = A's state after prefill (decode step 0). The write-set
    /// alone is not enough once later runs have moved bytes the cut reads
    /// but did not change.
    image: usize,
}

/// How B's write to a state compares with A's, on this cut: bytes of A's
/// write-set (`set`), how many of them B wrote differently (`n_diff`), and
/// how many bytes B changed outside A's write-set (`outside`).
#[derive(Clone, Default)]
struct StateCmp {
    set: usize,
    n_diff: usize,
    outside: usize,
}

/// Runs of `[offset, offset+len)` where `pre` and `post` differ (gaps under
/// 64 bytes are bridged so a sparse update is a few runs, not thousands).
fn diff_runs(pre: &[u8], post: &[u8]) -> Vec<(usize, Vec<u8>)> {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < pre.len() {
        if pre[i] == post[i] {
            i += 1;
            continue;
        }
        let start = i;
        while i < pre.len() && pre[i] != post[i] {
            i += 1;
        }
        match runs.last_mut() {
            Some((_, end)) if start - *end < 64 => *end = i,
            _ => runs.push((start, i)),
        }
    }
    runs.into_iter().map(|(a, b)| (a, pre[a..b].to_vec())).collect()
}

/// Write a state pre-image (or post-image at the same offsets) back.
fn restore_state(
    c: &mut Caller,
    runs: &BTreeMap<String, Vec<(usize, Vec<u8>)>>,
    image: Option<&BTreeMap<String, Vec<u8>>>,
) -> Result<()> {
    for (name, rs) in runs {
        for (off, bytes) in rs {
            match image {
                None => c.rt.write_state_at(name, *off, bytes)?,
                Some(img) => c.rt.write_state_at(name, *off, &img[name][*off..*off + bytes.len()])?,
            }
        }
    }
    Ok(())
}

fn load_side(json: &str, o: &Opts, blobs: &[&[u8]]) -> Result<Caller> {
    // The attestation drives the manifest's whole batch of sequences.
    let seqs = Manifest::from_json(json)?.seq_slots() - 2;
    let mut rt = Runtime::load(json, &o.kernels, o.gpu, Some(Capacity { tokens: Some(o.capacity), seqs }), None)?;
    rt.load_weights(blobs)?;
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

fn seg_label(s: &Segment) -> String {
    format!("A[{}..{}) B[{}..{})", s.a.0, s.a.1, s.b.0, s.b.1)
}

// ------------------------------------------------------------------ report
//
// Sections of aligned tables, rendered as colored text (tty) or GitHub
// markdown. Everything goes to stdout; the JSON report is `--out`.

#[derive(Clone, Copy, PartialEq, Debug, Default)]
enum Style {
    #[default]
    Plain,
    Bold,
    Dim,
    Good,
    Warn,
    Bad,
}

#[derive(Clone, Debug, Default)]
struct Cell {
    s: String,
    st: Style,
}

impl Cell {
    fn new(s: impl Into<String>, st: Style) -> Cell {
        Cell { s: s.into(), st }
    }
    fn good(s: impl Into<String>) -> Cell {
        Cell::new(s, Style::Good)
    }
    fn warn(s: impl Into<String>) -> Cell {
        Cell::new(s, Style::Warn)
    }
    fn bad(s: impl Into<String>) -> Cell {
        Cell::new(s, Style::Bad)
    }
    fn dim(s: impl Into<String>) -> Cell {
        Cell::new(s, Style::Dim)
    }
    fn bold(s: impl Into<String>) -> Cell {
        Cell::new(s, Style::Bold)
    }
}

impl From<String> for Cell {
    fn from(s: String) -> Cell {
        Cell { s, st: Style::Plain }
    }
}
impl From<&str> for Cell {
    fn from(s: &str) -> Cell {
        Cell { s: s.into(), st: Style::Plain }
    }
}

macro_rules! row {
    ($($x:expr),* $(,)?) => { vec![$(Cell::from($x)),*] };
}

enum Block {
    Table { header: bool, rows: Vec<Vec<Cell>> },
    Note(Cell),
}

struct Section {
    title: String,
    subtitle: String,
    blocks: Vec<Block>,
    started: Instant,
    timed: bool,
}

impl Section {
    fn new(title: &str, subtitle: &str) -> Section {
        Section {
            title: title.into(),
            subtitle: subtitle.into(),
            blocks: Vec::new(),
            started: Instant::now(),
            timed: true,
        }
    }
    fn untimed(mut self) -> Section {
        self.timed = false;
        self
    }
    fn table(&mut self, rows: Vec<Vec<Cell>>) {
        self.blocks.push(Block::Table { header: false, rows });
    }
    /// First row is a header.
    fn table_h(&mut self, rows: Vec<Vec<Cell>>) {
        self.blocks.push(Block::Table { header: true, rows });
    }
    fn note(&mut self, c: Cell) {
        self.blocks.push(Block::Note(c));
    }
}

#[derive(Clone, Copy, PartialEq, clap::ValueEnum)]
pub enum Format {
    Text,
    Md,
}

#[derive(Clone, Copy, PartialEq, clap::ValueEnum)]
pub enum Color {
    Auto,
    Always,
    Never,
}

struct Renderer {
    format: Format,
    color: bool,
}

impl Renderer {
    fn new(format: Format, color: Color) -> Renderer {
        use std::io::IsTerminal;
        let color = match color {
            Color::Always => true,
            Color::Never => false,
            Color::Auto => std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none(),
        } && format == Format::Text;
        Renderer { format, color }
    }

    fn paint(&self, c: &Cell) -> String {
        if !self.color || c.st == Style::Plain || c.s.is_empty() {
            return c.s.clone();
        }
        let code = match c.st {
            Style::Bold => "1",
            Style::Dim => "2",
            Style::Good => "32",
            Style::Warn => "33",
            Style::Bad => "1;31",
            Style::Plain => unreachable!(),
        };
        format!("\x1b[{code}m{}\x1b[0m", c.s)
    }

    fn md_cell(c: &Cell) -> String {
        let esc = c.s.replace('|', "\\|");
        match c.st {
            Style::Bad | Style::Bold => format!("**{esc}**"),
            Style::Warn => format!("*{esc}*"),
            Style::Dim => format!("<sub>{esc}</sub>"),
            _ => esc,
        }
    }

    fn header(&self, a: &str, b: &str) {
        match self.format {
            Format::Text => println!(
                "{}   {} {}   {} {}",
                self.paint(&Cell::bold("kern test")),
                self.paint(&Cell::dim("A")),
                a,
                self.paint(&Cell::dim("B")),
                b
            ),
            Format::Md => println!("# kern test\n\n`A` {a} → `B` {b}"),
        }
    }

    fn section(&self, sec: &Section) {
        println!();
        match self.format {
            Format::Text => {
                let t = self.paint(&Cell::bold(&sec.title));
                let took = if sec.timed { format!("   {:.1?}", sec.started.elapsed()) } else { String::new() };
                if sec.subtitle.is_empty() {
                    println!("{t}{}", self.paint(&Cell::dim(&took)));
                } else {
                    // Pad on the raw title so escape codes don't skew it.
                    let pad = " ".repeat(15usize.saturating_sub(sec.title.chars().count()));
                    println!("{t}{pad}{}", self.paint(&Cell::dim(format!("{}{took}", sec.subtitle))));
                }
                for (bi, b) in sec.blocks.iter().enumerate() {
                    match b {
                        Block::Table { header, rows } => {
                            if bi > 0 {
                                println!();
                            }
                            self.text_table(*header, rows)
                        }
                        Block::Note(c) => println!("  {}", self.paint(c)),
                    }
                }
            }
            Format::Md => {
                println!("## {}", sec.title);
                let took = if sec.timed { format!(" ({:.1?})", sec.started.elapsed()) } else { String::new() };
                if !sec.subtitle.is_empty() || sec.timed {
                    println!("\n*{}{took}*", sec.subtitle);
                }
                for b in &sec.blocks {
                    match b {
                        Block::Table { header, rows } => self.md_table(*header, rows),
                        Block::Note(c) => println!("\n{}", Self::md_cell(c)),
                    }
                }
            }
        }
    }

    fn text_table(&self, header: bool, rows: &[Vec<Cell>]) {
        let ncol = rows.iter().map(|r| r.len()).max().unwrap_or(0);
        let width = |s: &str| s.chars().count();
        let mut w = vec![0usize; ncol];
        for r in rows {
            for (i, c) in r.iter().enumerate() {
                w[i] = w[i].max(width(&c.s));
            }
        }
        for (ri, r) in rows.iter().enumerate() {
            let mut line = String::from("  ");
            for (i, c) in r.iter().enumerate() {
                let c = if header && ri == 0 && c.st == Style::Plain { Cell::dim(&c.s) } else { c.clone() };
                line.push_str(&self.paint(&c));
                if i + 1 < r.len() {
                    line.push_str(&" ".repeat(w[i] - width(&c.s) + 3));
                }
            }
            println!("{}", line.trim_end());
        }
    }

    fn md_table(&self, header: bool, rows: &[Vec<Cell>]) {
        let ncol = rows.iter().map(|r| r.len()).max().unwrap_or(0);
        let line = |r: &[Cell]| {
            let mut cells: Vec<String> = r.iter().map(Self::md_cell).collect();
            cells.resize(ncol, String::new());
            format!("| {} |", cells.join(" | "))
        };
        println!();
        let (head, body) =
            if header { (line(&rows[0]), &rows[1..]) } else { (line(&vec![Cell::default(); ncol]), rows) };
        println!("{head}");
        println!("|{}", " --- |".repeat(ncol));
        for r in body {
            println!("{}", line(r));
        }
    }

    fn verdict(&self, code: i32, summary: &str, elapsed: std::time::Duration, out: Option<&std::path::Path>) {
        println!();
        let tag = match code {
            0 => Cell::good("PASS"),
            1 => Cell::bad("FAIL"),
            _ => Cell::warn("INCONCLUSIVE"),
        };
        match self.format {
            Format::Text => {
                println!(
                    "{}   {}   {}   {}",
                    self.paint(&Cell::bold("VERDICT")),
                    self.paint(&tag),
                    summary,
                    self.paint(&Cell::dim(format!("{elapsed:.1?}")))
                );
                if let Some(p) = out {
                    println!("{}", self.paint(&Cell::dim(format!("          attestation written to {}", p.display()))));
                }
            }
            Format::Md => {
                println!("## Verdict\n\n**{}** — {summary} ({elapsed:.1?})", tag.s);
                if let Some(p) = out {
                    println!("\n<sub>attestation: `{}`</sub>", p.display());
                }
            }
        }
    }
}

fn us(ms: f32) -> String {
    if ms >= 1.0 {
        format!("{ms:.3} ms")
    } else {
        format!("{:.1} µs", ms * 1e3)
    }
}

/// B − A as a styled cell: faster is good.
fn delta(a: f32, b: f32) -> Cell {
    let d = (b - a) * 1e3;
    let pct = (b - a) / a.max(1e-9) * 100.0;
    let s = format!(
        "{}{:.1} µs  {}{:.1}%",
        if d < 0.0 { "−" } else { "+" },
        d.abs(),
        if pct < 0.0 { "−" } else { "+" },
        pct.abs()
    );
    if pct <= -1.0 {
        Cell::good(s)
    } else if pct >= 1.0 {
        Cell::bad(s)
    } else {
        Cell::from(s)
    }
}

fn pct_cell(a: f32, b: f32) -> Cell {
    let pct = (b - a) / a.max(1e-9) * 100.0;
    let s = format!("{}{:.1}%", if pct < 0.0 { "−" } else { "+" }, pct.abs());
    if pct <= -1.0 {
        Cell::good(s)
    } else if pct >= 1.0 {
        Cell::bad(s)
    } else {
        Cell::from(s)
    }
}

fn kb(bytes: usize) -> String {
    match bytes {
        b if b >= 1 << 20 => format!("{:.1} MB", b as f64 / 1e6),
        b if b >= 1 << 10 => format!("{:.0} KB", b as f64 / 1e3),
        b => format!("{b} B"),
    }
}

/// Short styled cell for one comparison.
fn cell(c: &Cmp) -> Cell {
    if c.identical() {
        Cell::good("bit-identical")
    } else if c.value_identical() {
        Cell::warn(format!("±0 only ({})", c.signed_zero))
    } else {
        let mut s = format!("{}/{} differ", c.n_diff - c.signed_zero, c.n);
        if let Some(u) = c.max_ulp {
            s += &format!(" · max {u} ulp");
        } else {
            s += &format!(" · max |Δ| {:.2e}", c.max_abs);
        }
        if c.nan_only_one_side > 0 {
            s += &format!(" · {} nan", c.nan_only_one_side);
        }
        Cell::bad(s)
    }
}

/// Summarize one buffer's comparisons across many cuts.
fn summarize(results: &[(String, Cmp)]) -> Cell {
    let n = results.len();
    let bit = results.iter().filter(|(_, c)| c.identical()).count();
    let val = results.iter().filter(|(_, c)| !c.identical() && c.value_identical()).count();
    if bit == n {
        return Cell::good("bit-identical");
    }
    if bit + val == n {
        let z: usize = results.iter().map(|(_, c)| c.signed_zero).sum();
        return Cell::warn(format!("value-identical · ±0 only ({z} elements)"));
    }
    let worst = results
        .iter()
        .filter(|(_, c)| !c.value_identical())
        .max_by_key(|(_, c)| (c.max_ulp, c.n_diff - c.signed_zero))
        .unwrap();
    Cell::bad(format!("{}/{n} cuts differ · worst {} at {}", n - bit - val, cell(&worst.1).s, worst.0))
}

fn execute(o: Opts) -> Result<i32> {
    let t_start = Instant::now();
    let ja = std::fs::read_to_string(&o.a).with_context(|| format!("reading {}", o.a.display()))?;
    let jb = std::fs::read_to_string(&o.b).with_context(|| format!("reading {}", o.b.display()))?;
    let ma = Manifest::from_json(&ja)?;
    let mb = Manifest::from_json(&jb)?;
    verify(&ma).with_context(|| format!("A ({}) failed verification", o.a.display()))?;
    verify(&mb).with_context(|| format!("B ({}) failed verification", o.b.display()))?;
    let mut report = json!({
        "a": o.a.display().to_string(), "b": o.b.display().to_string(),
    });
    let r = Renderer::new(o.format, o.color);
    r.header(&o.a.display().to_string(), &o.b.display().to_string());

    // ---- 1. static diff
    let mut sec = Section::new("DIFF", "").untimed();
    let changed = op_changes(&ma, &mb);
    let mut rows: Vec<Vec<Cell>> = Vec::new();
    if changed.is_empty() {
        rows.push(row!["ops", "no interface or implementation differs"]);
    }
    let launch_desc = |m: &Manifest, op: &kern_manifest::types::Op| -> String {
        let ls = &op.imp.launches;
        let mut names: Vec<String> = ls
            .iter()
            .map(|l| match l.module().and_then(|n| m.modules.get(n)) {
                Some(md) => md.source.rsplit('/').next().unwrap_or(&md.source).to_string(),
                None => l.entry().to_string(),
            })
            .collect();
        names.dedup();
        let n = names.join(" + ");
        if ls.len() > 1 {
            format!("{} launches: {n}", ls.len())
        } else {
            n
        }
    };
    for (k, kind) in &changed {
        let detail = match *kind {
            "interface" => format!("{} → {} params", ma.ops[k].params.len(), mb.ops[k].params.len()),
            "impl" => format!("{}  →  {}", launch_desc(&ma, &ma.ops[k]), launch_desc(&mb, &mb.ops[k])),
            "added" => launch_desc(&mb, &mb.ops[k]),
            _ => launch_desc(&ma, &ma.ops[k]),
        };
        rows.push(row![Cell::bold(k), Cell::warn(*kind), detail]);
    }
    for name in ma.buffers.keys().chain(mb.buffers.keys()).collect::<BTreeSet<_>>() {
        let what = match (ma.buffers.get(name), mb.buffers.get(name)) {
            (Some(x), Some(y)) if json!(x) != json!(y) => "changed",
            (Some(_), None) => "removed",
            (None, Some(_)) => "added",
            _ => continue,
        };
        rows.push(row![format!("buffer {name}"), Cell::warn(what), ""]);
    }
    let mut segments: BTreeMap<String, Vec<Segment>> = BTreeMap::new();
    let mut frontier_warn = false;
    for (pname, pa) in &ma.programs {
        let Some(pb) = mb.programs.get(pname) else {
            rows.push(row![pname.clone(), Cell::warn("only in A"), "skipped"]);
            continue;
        };
        let segs = align(pa, pb, &changed);
        let cuts: Vec<&Segment> = segs.iter().filter(|s| s.kind == Kind::Changed).collect();
        if cuts.is_empty() {
            rows.push(row![pname.clone(), Cell::dim("identical"), format!("{} calls", pa.len())]);
            continue;
        }
        // Group cuts by shape: (A ops, B ops, reads, writes).
        let mut groups: BTreeMap<(String, String, String, String), usize> = BTreeMap::new();
        for s in &cuts {
            let ka = pa[s.a.0..s.a.1].iter().map(|c| c.op.as_str()).collect::<Vec<_>>().join("+");
            let kb = pb[s.b.0..s.b.1].iter().map(|c| c.op.as_str()).collect::<Vec<_>>().join("+");
            let ia = frontier_inputs(&ma, pname, s.a.0, s.a.1);
            let ib = frontier_inputs(&mb, pname, s.b.0, s.b.1);
            let wa = access(&ma, pname, s.a.0, s.a.1).writes;
            let wb = access(&mb, pname, s.b.0, s.b.1).writes;
            if ia != ib || wa != wb {
                frontier_warn = true;
            }
            let reads = ia.iter().cloned().collect::<Vec<_>>().join(", ");
            let writes = wa.iter().cloned().collect::<Vec<_>>().join(", ");
            *groups.entry((ka, kb, reads, writes)).or_default() += 1;
        }
        let shared = pa.len() - cuts.iter().map(|s| s.a.1 - s.a.0).sum::<usize>();
        for (gi, ((ka, kb, reads, writes), n)) in groups.iter().enumerate() {
            let what = if ka == kb {
                ka.clone()
            } else if ka.is_empty() {
                format!("∅ → {kb}")
            } else if kb.is_empty() {
                format!("{ka} → ∅")
            } else {
                format!("{ka} → {kb}")
            };
            rows.push(row![
                if gi == 0 { pname.clone() } else { String::new() },
                Cell::bold(format!("{n} cut{}", if *n == 1 { "" } else { "s" })),
                format!("{what}   {reads} → {writes}"),
                Cell::dim(if gi == 0 { format!("{} calls, {shared} shared", pa.len()) } else { String::new() }),
            ]);
        }
        segments.insert(pname.clone(), segs);
    }
    for pname in mb.programs.keys() {
        if !ma.programs.contains_key(pname) {
            rows.push(row![pname.clone(), Cell::warn("only in B"), "skipped"]);
        }
    }
    sec.table(rows);
    if frontier_warn {
        sec.note(Cell::warn(
            "⚠ some cuts read or write different buffers on the two sides — not a cut-internal replacement",
        ));
    }
    r.section(&sec);
    report["diff"] = json!({
        "kernels": changed,
        "programs": segments.iter().map(|(p, segs)| (p.clone(), json!(segs.iter()
            .filter(|s| s.kind == Kind::Changed)
            .map(|s| json!({"a": [s.a.0, s.a.1], "b": [s.b.0, s.b.1]})).collect::<Vec<_>>())))
            .collect::<serde_json::Map<_, _>>(),
    });
    if o.diff_only {
        return Ok(0);
    }
    if segments.is_empty() {
        println!("\nnothing to test: the programs are identical.");
        return Ok(0);
    }

    // ---- load A + B
    let blobs = o
        .weights
        .iter()
        .map(|p| std::fs::read(p).with_context(|| format!("reading {}", p.display())))
        .collect::<Result<Vec<_>>>()?;
    let refs: Vec<&[u8]> = blobs.iter().map(Vec::as_slice).collect();
    let t = Instant::now();
    let mut s = Sides { a: load_side(&ja, &o, &refs)?, b: load_side(&jb, &o, &refs)? };
    drop(blobs);
    let load_t = t.elapsed();
    let prompt_ids = match &o.prompt {
        Some(text) => {
            let Some(tk) = &o.tokenizer else {
                bail!("--prompt needs a tokenizer (--tokenizer or the target's `tokenizer`)")
            };
            let tokenizer = tokenizers::Tokenizer::from_file(tk).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
            Some(tokens_of(&tokenizer, text)?)
        }
        None => None,
    };
    let wl = sample_workload(&o, &ma, s.a.rt.provision(), s.a.rt.page(), prompt_ids)?;
    let e1 = env(1);
    let cuts_of = |p: &str| -> Vec<Segment> {
        segments.get(p).map(|sg| sg.iter().filter(|s| s.kind == Kind::Changed).cloned().collect()).unwrap_or_default()
    };

    // ---- 2. tap: A and B in lockstep on the prompt, snapshot at every cut
    let mut sec = Section::new("TAP", &format!("workload seed {:#x}: prefill {} tokens ({}) in chunks of {} · {} decode steps · vocab {} · lockstep: each run B starts from A's state, each cut from A's inputs (cut-local) · runtimes loaded in {:.1?}", o.seed, wl.prefill.len(), wl.how, wl.chunk, wl.decode.len(), wl.vocab, load_t));
    let mut snaps: Vec<Snap> = Vec::new();
    let mut snap_state_bytes = 0usize;
    // program -> buffer -> [(cut label, cmp)]
    let mut local_res: BTreeMap<String, BTreeMap<String, Vec<(String, Cmp)>>> = BTreeMap::new();
    let mut local_states: BTreeMap<String, BTreeMap<String, Vec<(String, StateCmp)>>> = BTreeMap::new();
    let mut one_sided_all: BTreeSet<String> = BTreeSet::new();
    let mut local_json = Vec::new();
    let mut n_chunks = 0usize;
    s.a.reset();
    s.b.reset();
    // Run one program in lockstep; at each cut snapshot A's frontier inputs
    // (from A, before the cut), A's outputs after, and compare B's outputs.
    let mut lockstep = |s: &mut Sides,
                        pname: &str,
                        e: &BTreeMap<String, u64>,
                        label_prefix: &str,
                        keep: bool,
                        image: usize|
     -> Result<()> {
        let Some(segs) = segments.get(pname) else {
            s.a.rt.run(pname, e)?;
            s.b.rt.run(pname, e)?;
            return Ok(());
        };
        for seg in segs {
            if seg.kind != Kind::Changed {
                s.a.rt.run_range(pname, e, seg.a.0, seg.a.1)?;
                s.b.rt.run_range(pname, e, seg.b.0, seg.b.1)?;
                continue;
            }
            let label = format!("{label_prefix}{}", seg_label(seg));
            let input_names: BTreeSet<String> = frontier_inputs(&ma, pname, seg.a.0, seg.a.1)
                .union(&frontier_inputs(&mb, pname, seg.b.0, seg.b.1))
                .cloned()
                .collect();
            let mut inputs = Vec::new();
            for n in &input_names {
                if ma.buffers[n].kind != BufferKind::Weight {
                    inputs.push((n.clone(), s.a.rt.read_buffer_prefix(n, live_bytes(&ma, n, e))?));
                }
            }
            // B runs the cut from A's inputs: the columns below are the
            // cut's own doing, not what drifted in from B's earlier cuts.
            // B's end-to-end drift is the free run after the tap.
            for (n, bytes) in &inputs {
                s.b.rt.write_buffer(n, bytes)?;
            }
            let (aa, ab) = (access(&ma, pname, seg.a.0, seg.a.1), access(&mb, pname, seg.b.0, seg.b.1));
            let touched: BTreeSet<String> = aa.state_writes.union(&ab.state_writes).cloned().collect();
            // State is compared on the cut's write-set — the bytes A's run
            // changed — not on the whole allocation (the rest is other
            // layers' history, and it moves between now and any replay).
            // B runs the cut from A's pre-image of that write-set, so what
            // B writes there is this cut's doing, not its own history's.
            // Per-cut state work (whole-state reads) only on the snapshotted
            // run; every other run B starts from a copy of A's state, which
            // gives the same pre-image per cut (layers' write-sets are
            // disjoint) at one copy per run instead of four reads per cut.
            let mut a_pre = BTreeMap::new();
            if keep {
                for st in &touched {
                    a_pre.insert(st.clone(), s.a.rt.read_state(st)?);
                }
            }
            s.a.rt.run_range(pname, e, seg.a.0, seg.a.1)?;
            let mut a_post = BTreeMap::new();
            let mut pre_states = BTreeMap::new();
            if keep {
                for st in &touched {
                    let post = s.a.rt.read_state(st)?;
                    let runs = diff_runs(&a_pre[st], &post);
                    for (off, bytes) in &runs {
                        s.b.rt.write_state_at(st, *off, bytes)?;
                    }
                    pre_states.insert(st.clone(), runs);
                    a_post.insert(st.clone(), post);
                }
            }
            let mut b_pre = BTreeMap::new();
            if keep {
                for st in &touched {
                    b_pre.insert(st.clone(), s.b.rt.read_state(st)?);
                }
            }
            s.b.rt.run_range(pname, e, seg.b.0, seg.b.1)?;
            let (bufs, _, one_sided) = compare_written(&s.a, &s.b, &aa, &ab, e, false)?;
            let mut states = BTreeMap::new();
            for st in touched.iter().filter(|_| keep) {
                let b_post = s.b.rt.read_state(st)?;
                let ap = &a_post[st];
                let runs = &pre_states[st];
                let set: usize = runs.iter().map(|(_, b)| b.len()).sum();
                let n_diff: usize =
                    runs.iter().map(|(off, b)| (0..b.len()).filter(|i| ap[off + i] != b_post[off + i]).count()).sum();
                // bytes B changed that lie outside A's write-set
                let outside: usize = diff_runs(&b_pre[st], &b_post)
                    .iter()
                    .map(|(boff, bb)| {
                        (0..bb.len())
                            .filter(|i| {
                                !runs.iter().any(|(aoff, ab)| (aoff..&(aoff + ab.len())).contains(&&(boff + i)))
                            })
                            .count()
                    })
                    .sum();
                states.insert(st.clone(), StateCmp { set, n_diff, outside });
            }
            for (n, cmp) in &bufs {
                local_res
                    .entry(pname.into())
                    .or_default()
                    .entry(n.clone())
                    .or_default()
                    .push((label.clone(), cmp.clone()));
            }
            for (st, c) in &states {
                local_states
                    .entry(pname.into())
                    .or_default()
                    .entry(st.clone())
                    .or_default()
                    .push((label.clone(), c.clone()));
            }
            one_sided_all.extend(one_sided.iter().cloned());
            local_json.push(json!({"program": pname, "cut": label,
                "buffers": bufs.iter().map(|(n, c)| (n.clone(), c.to_json())).collect::<serde_json::Map<_, _>>(),
                "states": states.iter().map(|(n, c)| (n.clone(), json!({"write_set": c.set, "differ": c.n_diff, "outside": c.outside}))).collect::<serde_json::Map<_, _>>(),
                "one_sided": one_sided}));
            if keep {
                let mut ref_out = BTreeMap::new();
                for n in aa.writes.intersection(&ab.writes) {
                    ref_out.insert(n.clone(), s.a.rt.read_buffer_prefix(n, live_bytes(&ma, n, e))?);
                }
                snap_state_bytes += pre_states.values().flatten().map(|(_, b)| b.len()).sum::<usize>();
                snaps.push(Snap {
                    program: pname.into(),
                    seg: seg.clone(),
                    env: e.clone(),
                    inputs,
                    ref_out,
                    ref_states: a_post,
                    pre_states,
                    image,
                });
            }
        }
        Ok(())
    };
    let shared_states: Vec<String> = ma.states.keys().filter(|n| mb.states.contains_key(*n)).cloned().collect();
    // B starts every program run from A's state: with per-layer write-sets
    // this is the per-cut pre-image for every cut of the run.
    let sync_b = |s: &mut Sides| -> Result<()> {
        for name in &shared_states {
            let img = s.a.rt.read_state(name)?;
            s.b.rt.write_state_at(name, 0, &img)?;
        }
        Ok(())
    };
    // `logits*` buffers a program writes: the end-to-end oracle.
    let logits_of = |prog: &str| -> Vec<String> {
        access(&ma, prog, 0, ma.programs[prog].len())
            .writes
            .into_iter()
            .filter(|n| n.starts_with("logits") && mb.buffers.contains_key(n))
            .collect()
    };
    // (run label, buffer, env, bytes)
    let read_logits = |c: &Caller,
                       prog: &str,
                       e: &BTreeMap<String, u64>,
                       label: &str|
     -> Result<Vec<(String, String, BTreeMap<String, u64>, Vec<u8>)>> {
        logits_of(prog)
            .into_iter()
            .map(|n| {
                Ok((label.to_string(), n.clone(), e.clone(), c.rt.read_buffer_prefix(&n, live_bytes(&ma, &n, e))?))
            })
            .collect()
    };
    let pre = &wl.prefill;
    let chunk = wl.chunk;
    let mut a_logits = Vec::new();
    let mut i = 0;
    while i < pre.len() {
        let c = (pre.len() - i).min(chunk);
        let e = s.a.stage_prefill(&pre[i..i + c])?;
        s.b.stage_prefill(&pre[i..i + c])?;
        sync_b(&mut s)?;
        lockstep(&mut s, "prefill", &e, &format!("chunk {n_chunks} "), n_chunks == 0, 0)?;
        a_logits.extend(read_logits(&s.a, "prefill", &e, &format!("prefill chunk {n_chunks}"))?);
        s.a.advance(c as u64);
        s.b.advance(c as u64);
        i += c;
        n_chunks += 1;
    }
    let n_steps = wl.decode.len();
    // Decode-step programs this manifest declares; the workload rotates
    // through them step by step (a caller may switch between them at any
    // step: same inputs at seqs=1, same state contract).
    let decode_progs: Vec<&str> = DECODE_LIKE.into_iter().filter(|p| ma.programs.contains_key(*p)).collect();
    let prog_of = |k: usize| decode_progs[k % decode_progs.len()];
    // A's state after prefill: the image every decode-step-0 replay starts from.
    let s0: BTreeMap<String, Vec<u8>> =
        shared_states.iter().map(|n| Ok((n.clone(), s.a.rt.read_state(n)?))).collect::<Result<_>>()?;
    for (k, &tok) in wl.decode.iter().enumerate() {
        let p = prog_of(k);
        s.a.stage_decode(tok)?;
        s.b.stage_decode(tok)?;
        sync_b(&mut s)?;
        lockstep(&mut s, p, &e1, &format!("step {k} "), k < decode_progs.len(), 1)?;
        a_logits.extend(read_logits(&s.a, p, &e1, &format!("step {k}"))?);
        s.a.advance(1);
        s.b.advance(1);
    }
    // B free-runs the same workload from zero state, nothing injected: what
    // a caller would get. A's state is untouched by the lockstep (only B
    // received writes), so A is already the end-to-end reference.
    let t_free = Instant::now();
    s.b.rt.zero_states()?;
    s.b.reset();
    let mut b_logits = Vec::new();
    let mut i = 0;
    let mut nc = 0;
    while i < pre.len() {
        let c = (pre.len() - i).min(chunk);
        let e = s.b.stage_prefill(&pre[i..i + c])?;
        s.b.rt.run("prefill", &e)?;
        b_logits.extend(read_logits(&s.b, "prefill", &e, &format!("prefill chunk {nc}"))?);
        s.b.advance(c as u64);
        i += c;
        nc += 1;
    }
    for (k, &tok) in wl.decode.iter().enumerate() {
        let p = prog_of(k);
        s.b.stage_decode(tok)?;
        s.b.rt.run(p, &e1)?;
        b_logits.extend(read_logits(&s.b, p, &e1, &format!("step {k}"))?);
        s.b.advance(1);
    }
    let free_t = t_free.elapsed();
    // ---- end-to-end logits, row by row
    let mut logit_rows: Vec<LogitRow> = Vec::new();
    for ((label, name, e, a), (_, _, _, b)) in a_logits.iter().zip(&b_logits) {
        let dt = ma.buffers[name].dtype;
        let row = row_elems(&ma, name, e) * dt.bytes() as usize;
        let rows = a.len().checked_div(row).unwrap_or(0);
        for r in 0..rows.max(1) {
            let (lo, hi) = if rows > 1 { (r * row, (r + 1) * row) } else { (0, a.len()) };
            let lbl = if a_logits.iter().filter(|x| x.0 == *label).count() > 1 || rows > 1 {
                format!("{label} {name}{}", if rows > 1 { format!("[{r}]") } else { String::new() })
            } else {
                label.clone()
            };
            logit_rows.push(logit_row(lbl, dt, &a[lo..hi], &b[lo..hi]));
        }
    }
    let mut local_identical = true;
    let mut local_bit = true;
    let mut rows = Vec::new();
    let undriven: Vec<String> = segments.keys().filter(|p| !DRIVEN.contains(&p.as_str())).cloned().collect();
    for pname in ma.programs.keys().map(String::as_str) {
        let n_cuts = cuts_of(pname).len();
        if n_cuts == 0 || undriven.iter().any(|u| u == pname) {
            continue;
        }
        let count = if pname == "prefill" && n_chunks > 1 {
            format!("{n_cuts} cuts × {n_chunks} chunks")
        } else if decode_progs.contains(&pname) && n_steps > 1 {
            let n = (0..n_steps).filter(|&k| prog_of(k) == pname).count();
            format!("{n_cuts} cuts × {n} steps")
        } else {
            format!("{n_cuts} cuts")
        };
        let mut first = true;
        if let Some(bufs) = local_res.get(pname) {
            for (buf, res) in bufs {
                local_identical &= res.iter().all(|(_, c)| c.value_identical());
                local_bit &= res.iter().all(|(_, c)| c.identical());
                rows.push(row![
                    if first { pname.to_string() } else { String::new() },
                    count.clone(),
                    buf.clone(),
                    summarize(res)
                ]);
                first = false;
            }
        }
        if let Some(sts) = local_states.get(pname) {
            for (st, res) in sts {
                let bad: Vec<&(String, StateCmp)> = res.iter().filter(|(_, c)| c.n_diff > 0 || c.outside > 0).collect();
                local_identical &= bad.is_empty();
                local_bit &= bad.is_empty();
                let set: usize = res.iter().map(|(_, c)| c.set).max().unwrap_or(0);
                let txt = if bad.is_empty() {
                    Cell::good(format!("bit-identical on the cut's write-set ({})", kb(set)))
                } else {
                    let (label, c) = bad[0];
                    let mut t = format!(
                        "{}/{} cuts differ · {}/{} bytes of the write-set at {label}",
                        bad.len(),
                        res.len(),
                        c.n_diff,
                        c.set
                    );
                    if c.outside > 0 {
                        t += &format!(" · B wrote {} outside A's write-set", kb(c.outside));
                    }
                    Cell::bad(t)
                };
                rows.push(row![
                    if first { pname.to_string() } else { String::new() },
                    format!("{n_cuts} cuts (first run)"),
                    format!("state {st}"),
                    txt
                ]);
                first = false;
            }
        }
    }
    let mut out_cmp = BTreeMap::new();
    let mut e2e_outputs_identical = true;
    let mut e2e_differs: Vec<String> = Vec::new();
    let mut first = true;
    for (name, b) in &ma.buffers {
        if b.kind == BufferKind::Output && mb.buffers.contains_key(name) {
            let bytes = live_bytes(&ma, name, &e1);
            let c =
                compare(b.dtype, &s.a.rt.read_buffer_prefix(name, bytes)?, &s.b.rt.read_buffer_prefix(name, bytes)?);
            if !c.value_identical() {
                e2e_outputs_identical = false;
                e2e_differs.push(name.clone());
            }
            rows.push(row![if first { "end-to-end" } else { "" }, "", format!("output {name}"), cell(&c)]);
            first = false;
            out_cmp.insert(name.clone(), c.to_json());
        }
    }
    let mut e2e_states = serde_json::Map::new();
    for name in ma.states.keys().filter(|n| mb.states.contains_key(*n)) {
        let (a, b) = (s.a.rt.read_state(name)?, s.b.rt.read_state(name)?);
        let d = a.iter().zip(&b).filter(|(p, q)| p != q).count() + a.len().abs_diff(b.len());
        let txt = if d == 0 {
            Cell::good(format!("bit-identical ({})", kb(a.len())))
        } else {
            Cell::warn(format!("{d} of {} bytes differ ({:.2}%)", a.len(), d as f64 * 100.0 / a.len().max(1) as f64))
        };
        rows.push(row![if first { "end-to-end" } else { "" }, "", format!("state {name}"), txt]);
        first = false;
        e2e_states.insert(name.clone(), json!({"differ": d, "bytes": a.len()}));
    }
    sec.table(rows);
    sec.note(Cell::dim(format!("end-to-end: B re-ran the whole workload from zero state with nothing injected ({:.1?}); the cut rows above are B on A's inputs and A's state", free_t)));
    let snap_bytes: usize = snaps
        .iter()
        .map(|sn| {
            sn.inputs.iter().map(|(_, b)| b.len()).sum::<usize>() + sn.ref_out.values().map(|b| b.len()).sum::<usize>()
        })
        .sum();
    sec.note(Cell::dim(format!(
        "{} cuts snapshotted ({} of frontier inputs + reference outputs)",
        snaps.len(),
        kb(snap_bytes)
    )));
    if snap_state_bytes > 0 {
        sec.note(Cell::dim(format!("inout state: B ran every cut from A's pre-image of the cut's write-set; {} of pre-image kept — every replay below starts from it and puts A's post-image back", kb(snap_state_bytes))));
    }
    if !one_sided_all.is_empty() {
        sec.note(Cell::dim(format!(
            "written on one side only (implementation-internal, not compared): {:?}",
            one_sided_all
        )));
    }
    for p in &undriven {
        sec.note(Cell::bad(format!("{p}: changed but not tapped — the workload driver only stages {DRIVEN:?}")));
    }
    r.section(&sec);
    report["local"] = json!({"value_identical": local_identical, "bit_identical": local_bit, "cuts": local_json,
        "end_to_end": {"outputs": out_cmp, "states": e2e_states, "outputs_identical": e2e_outputs_identical}});

    // ---- 2b. logits: the end-to-end oracle
    let have_logits = !logit_rows.is_empty();
    let logits_bit = have_logits && logit_rows.iter().all(|r| r.cmp.identical());
    let logits_max_ulp = logit_rows.iter().map(|r| r.scale_ulps).fold(0.0, f64::max);
    let n_flips = logit_rows.iter().filter(|r| r.flip()).count();
    let n_near = logit_rows.iter().filter(|r| r.near_tie()).count();
    let wide_flip = logit_rows.iter().find(|r| r.flip() && !r.near_tie());
    let logits_within = have_logits && logits_max_ulp <= o.logit_ulp as f64 && wide_flip.is_none();
    {
        let mut sec = Section::new(
            "LOGITS",
            &format!(
                "end-to-end oracle · A (lockstep, nothing injected into A) vs B (free run) · {} rows over {} runs",
                logit_rows.len(),
                a_logits.len()
            ),
        );
        let mut rows = Vec::new();
        if !have_logits {
            rows.push(row![
                Cell::warn("no logits"),
                "no driven program writes a `logits*` buffer — the verdict falls back to cut identity"
            ]);
        } else {
            let worst = logit_rows
                .iter()
                .max_by(|x, y| x.scale_ulps.total_cmp(&y.scale_ulps).then(x.cmp.n_diff.cmp(&y.cmp.n_diff)))
                .unwrap();
            let worst_kl = logit_rows.iter().max_by(|x, y| x.kl.total_cmp(&y.kl)).unwrap();
            rows.push(row![
                "argmax",
                if n_flips == 0 {
                    Cell::good(format!("agrees on all {} rows", logit_rows.len()))
                } else if wide_flip.is_none() {
                    Cell::warn(format!(
                        "{n_flips} flip{} on {} rows, all near-ties (A's own margin ≤ Δ)",
                        if n_flips == 1 { "" } else { "s" },
                        logit_rows.len()
                    ))
                } else {
                    Cell::bad(format!(
                        "{} wide-margin flip{} ({n_near} near-tie) on {} rows",
                        n_flips - n_near,
                        if n_flips - n_near == 1 { "" } else { "s" },
                        logit_rows.len()
                    ))
                }
            ]);
            rows.push(row![
                "max Δ",
                if logits_bit {
                    Cell::good("bit-identical everywhere")
                } else {
                    Cell::from(format!(
                        "{:.4} = {:.2} ulp at the row's scale (max |logit| {:.1}) at {} · {}/{} elements differ",
                        worst.max_abs, worst.scale_ulps, worst.scale, worst.label, worst.cmp.n_diff, worst.cmp.n
                    ))
                }
            ]);
            rows.push(row!["KL(A‖B)", format!("max {:.3e} at {}", worst_kl.kl, worst_kl.label)]);
            for r in logit_rows.iter().filter(|r| r.flip()) {
                let t = format!(
                    "A {} → B {} · A's margin {:.4} · Δ max {:.4}",
                    r.argmax_a, r.argmax_b, r.margin_a, r.max_abs
                );
                rows.push(row![
                    r.label.clone(),
                    if r.near_tie() { Cell::warn(format!("near-tie · {t}")) } else { Cell::bad(format!("✗ {t}")) }
                ]);
            }
        }
        sec.table(rows);
        r.section(&sec);
    }
    report["logits"] = json!({
        "bit_identical": logits_bit, "max_ulp": logits_max_ulp, "flips": n_flips, "near_ties": n_near, "within_limit": logits_within, "limit_ulp": o.logit_ulp,
        "rows": logit_rows.iter().map(|r| json!({"label": r.label, "cmp": r.cmp.to_json(), "max_abs": r.max_abs, "scale_ulps": r.scale_ulps, "scale": r.scale, "argmax_a": r.argmax_a, "argmax_b": r.argmax_b, "margin_a": r.margin_a, "kl": r.kl, "near_tie": r.near_tie()})).collect::<Vec<_>>()});

    // Replay one snapshot's cut on a side from its inputs; returns the
    // written buffers (live prefix) and states.
    // The state a cut writes is put back to A's pre-image first (so the cut
    // sees what it saw in the tap, on either side) and to A's post-image
    // after (so the next cut's reads see the reference, not this replay's
    // output — under fuzz, garbage).
    let replay = |c: &mut Caller,
                  m: &Manifest,
                  sn: &Snap,
                  side_b: bool,
                  inputs: &[(String, Vec<u8>)]|
     -> Result<(BTreeMap<String, Vec<u8>>, BTreeMap<String, Vec<u8>>)> {
        restore_state(c, &sn.pre_states, None)?;
        for (n, bytes) in inputs {
            c.rt.write_buffer(n, bytes)?;
        }
        let r = if side_b { sn.seg.b } else { sn.seg.a };
        let run = c.rt.run_range(&sn.program, &sn.env, r.0, r.1);
        let mut out = BTreeMap::new();
        let mut st = BTreeMap::new();
        if run.is_ok() {
            for n in sn.ref_out.keys() {
                out.insert(n.clone(), c.rt.read_buffer_prefix(n, live_bytes(m, n, &sn.env))?);
            }
            for n in sn.ref_states.keys() {
                st.insert(n.clone(), c.rt.read_state(n)?);
            }
            restore_state(c, &sn.pre_states, Some(&sn.ref_states))?;
        }
        run?;
        Ok((out, st))
    };
    let cmp_out = |m: &Manifest, sn: &Snap, out: &BTreeMap<String, Vec<u8>>| -> BTreeMap<String, Cmp> {
        out.iter().map(|(n, b)| (n.clone(), compare(m.buffers[n].dtype, &sn.ref_out[n], b))).collect()
    };

    // Replays start from the image of the state before the snapshotted run
    // — zeros for prefill chunk 0, A's post-prefill state for decode step 0
    // — on both sides. A cut reads its own layer's slice, which the run's
    // earlier cuts do not touch, so that image is every cut's pre-state;
    // the per-cut pre-image then only undoes the previous replay of the
    // same cut. Re-imaged whenever the replay sequence switches runs.
    let image = |s: &mut Sides, img: usize| -> Result<()> {
        for c in [&mut s.a, &mut s.b] {
            if img == 0 {
                c.rt.zero_states()?;
            } else {
                for (n, b) in &s0 {
                    c.rt.write_state_at(n, 0, b)?;
                }
            }
        }
        Ok(())
    };

    // ---- 3. noise floor: A's cut re-run from its own snapshot
    let mut noise_res: BTreeMap<String, BTreeMap<String, Vec<(String, Cmp)>>> = BTreeMap::new();
    let mut noise_clean = true;
    let mut noisy_states: BTreeSet<(String, String)> = BTreeSet::new();
    if !o.no_noise {
        let mut sec = Section::new(
            "NOISE FLOOR",
            &format!("A re-run from each snapshot (inout state restored) vs A's own output · {} cuts", snaps.len()),
        );
        let mut state_noise = Vec::new();
        let mut cur = None;
        for sn in &snaps {
            if cur != Some(sn.image) {
                image(&mut s, sn.image)?;
                cur = Some(sn.image);
            }
            let (out, st) = replay(&mut s.a, &ma, sn, false, &sn.inputs)?;
            for (n, c) in cmp_out(&ma, sn, &out) {
                noise_clean &= c.identical();
                noise_res.entry(sn.program.clone()).or_default().entry(n).or_default().push((seg_label(&sn.seg), c));
            }
            for (name, bytes) in st {
                let refp = &sn.ref_states[&name];
                let d: usize = sn.pre_states[&name]
                    .iter()
                    .map(|(off, b)| (0..b.len()).filter(|i| bytes[off + i] != refp[off + i]).count())
                    .sum();
                if d > 0 {
                    noise_clean = false;
                    noisy_states.insert((sn.program.clone(), name.clone()));
                    state_noise.push(format!("state {name}: {d} bytes at {} {}", sn.program, seg_label(&sn.seg)));
                }
            }
        }
        let mut rows = Vec::new();
        if noise_clean {
            rows.push(row![Cell::good("clean"), "every cut reproduces its output bit for bit"]);
        } else {
            for (pname, bufs) in &noise_res {
                for (buf, res) in bufs {
                    rows.push(row![pname.clone(), format!("{} cuts", res.len()), buf.clone(), summarize(res)]);
                }
            }
            for t in &state_noise {
                rows.push(row![Cell::warn("⚠"), t.clone()]);
            }
            rows.push(row!["", Cell::warn("A is not deterministic at these cuts; B is judged against this band")]);
        }
        sec.table(rows);
        r.section(&sec);
    }
    report["noise_floor"] = json!({"clean": noise_clean,
        "cuts": noise_res.iter().map(|(p, bufs)| (p.clone(), json!(bufs.iter().map(|(b, res)| (b.clone(), json!(res.iter().map(|(l, c)| json!({"cut": l, "cmp": c.to_json()})).collect::<Vec<_>>()))).collect::<serde_json::Map<_, _>>()))).collect::<serde_json::Map<_, _>>()});

    // ---- 4. fuzz the cuts from their snapshots
    let mut fuzz_ok = true;
    let mut fuzz_identical = true; // value-identical under every distribution
    let mut fuzz_bit = true;
    if o.fuzz > 0 {
        let mut sec = Section::new(
            "FUZZ",
            &format!(
                "{} rounds per cut · {} cuts · float inputs perturbed around the tap ({}) · integers kept as tapped",
                o.fuzz,
                snaps.len(),
                MODES.join(" / ")
            ),
        );
        let mut rng = Rng(o.seed);
        let progs: Vec<String> = segments.keys().cloned().collect();
        // round -> program -> worst cell
        let mut grid: BTreeMap<usize, BTreeMap<String, Cell>> = BTreeMap::new();
        let mut ints_kept: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        let mut rounds_json = Vec::new();
        for round in 0..o.fuzz {
            let mode = round % MODES.len();
            let mut worst: BTreeMap<String, Cmp> = BTreeMap::new();
            let mut violations: BTreeMap<String, Vec<String>> = BTreeMap::new();
            let mut state_diffs: BTreeMap<String, Vec<String>> = BTreeMap::new();
            let mut cur = None;
            for sn in &snaps {
                if cur != Some(sn.image) {
                    image(&mut s, sn.image)?;
                    cur = Some(sn.image);
                }
                let mut inputs = Vec::new();
                for (name, tapped) in &sn.inputs {
                    let decl = &ma.buffers[name];
                    if !is_float(decl.dtype) {
                        // sequence layout, indices, page tables: structure,
                        // not values — a random one is a workload no caller
                        // produces, and the manifest says nothing about rows
                        // outside it
                        ints_kept.entry(sn.program.clone()).or_default().insert(name.clone());
                        inputs.push((name.clone(), tapped.clone()));
                        continue;
                    }
                    let base = values::to_f64(decl.dtype, tapped);
                    let vals = perturb(&mut rng, mode, &base, row_elems(&ma, name, &sn.env), decl.dtype);
                    inputs.push((name.clone(), values::from_f64(decl.dtype, &vals)));
                }
                let (out_a, st_a) = match replay(&mut s.a, &ma, sn, false, &inputs) {
                    Ok(x) => x,
                    Err(err) => bail!(
                        "A crashed under fuzz ({}) at {} cut {}: {err}",
                        MODES[mode],
                        sn.program,
                        seg_label(&sn.seg)
                    ),
                };
                let (out_b, st_b) = match replay(&mut s.b, &mb, sn, true, &inputs) {
                    Ok(x) => x,
                    Err(err) => {
                        println!(
                            "  ✗ B crashed under `{}` at {} cut {}: {err}",
                            MODES[mode],
                            sn.program,
                            seg_label(&sn.seg)
                        );
                        bail!("B crashed under fuzz; the CUDA context is unusable past this point");
                    }
                };
                for (name, b) in &st_b {
                    // on the cut's write-set, like the tap
                    let a = &st_a[name];
                    let d: usize = sn.pre_states[name]
                        .iter()
                        .map(|(off, r)| (0..r.len()).filter(|i| a[off + i] != b[off + i]).count())
                        .sum();
                    if d > 0 {
                        state_diffs
                            .entry(sn.program.clone())
                            .or_default()
                            .push(format!("state {name}: {d} bytes at {}", seg_label(&sn.seg)));
                    }
                }
                for (name, b) in &out_b {
                    let c = compare(ma.buffers[name].dtype, &out_a[name], b);
                    let w = worst.entry(sn.program.clone()).or_default();
                    if (c.n_diff - c.signed_zero, c.max_ulp, c.signed_zero)
                        > (w.n_diff - w.signed_zero, w.max_ulp, w.signed_zero)
                    {
                        *w = c;
                    }
                    // Post-condition: produced values must lie in the
                    // buffer's declared domain (A is checked too — a
                    // violation there is the reference misbehaving).
                    if let Some(d) = &mb.buffers[name].domain {
                        let r = d.resolve(&mb, &sn.env, &s.b.rt.provision())?;
                        for (side, bytes) in [("A", &out_a[name]), ("B", b)] {
                            let v = values::to_f64(mb.buffers[name].dtype, bytes);
                            if let Some(i) = v.iter().position(|x| !r.contains(*x)) {
                                violations
                                    .entry(sn.program.clone())
                                    .or_default()
                                    .push(format!("{side} {name}[{i}] = {} outside domain", v[i]));
                            }
                        }
                    }
                }
            }
            for p in &progs {
                if !snaps.iter().any(|sn| &sn.program == p) {
                    grid.entry(round).or_default().insert(p.clone(), Cell::warn("not tapped"));
                    continue;
                }
                let w = worst.get(p).cloned().unwrap_or_default();
                fuzz_identical &= w.value_identical();
                fuzz_bit &= w.identical();
                let sd = state_diffs.get(p);
                if sd.is_some() {
                    fuzz_identical = false;
                    fuzz_bit = false;
                }
                let txt = match (violations.get(p), sd) {
                    (Some(v), _) => {
                        fuzz_ok = false;
                        Cell::bad(format!("✗ {}", v.join("; ")))
                    }
                    (None, Some(d)) => Cell::bad(format!(
                        "{} · {}{}",
                        cell(&w).s,
                        d[0],
                        if d.len() > 1 { format!(" (+{} cuts)", d.len() - 1) } else { String::new() }
                    )),
                    (None, None) => cell(&w),
                };
                grid.entry(round).or_default().insert(p.clone(), txt);
                rounds_json.push(json!({"mode": MODES[mode], "program": p, "worst": w.to_json(), "violations": violations.get(p), "state_diffs": sd}));
            }
        }
        let mut rows = vec![std::iter::once(Cell::default())
            .chain(progs.iter().map(|p| Cell::from(p.as_str())))
            .collect::<Vec<_>>()];
        for (round, cells) in &grid {
            let name = if o.fuzz > MODES.len() {
                format!("{} #{}", MODES[round % MODES.len()], round / MODES.len())
            } else {
                MODES[round % MODES.len()].to_string()
            };
            rows.push(
                std::iter::once(Cell::from(name))
                    .chain(progs.iter().map(|p| cells.get(p).cloned().unwrap_or_default()))
                    .collect(),
            );
        }
        sec.table_h(rows);
        for (p, u) in &ints_kept {
            sec.note(Cell::dim(format!(
                "{p}: integer inputs kept as tapped: {}",
                u.iter().cloned().collect::<Vec<_>>().join(", ")
            )));
        }
        r.section(&sec);
        report["fuzz"] = json!({"ok": fuzz_ok, "value_identical": fuzz_identical, "bit_identical": fuzz_bit, "rounds": rounds_json,
            "integers_kept": ints_kept});
    }

    // ---- 5. perf: eager step attribution, graph step, sweep, roofline
    if !o.no_perf {
        let sweep_iters = o.iters.min(10);
        let mut sec = Section::new(
            "PERF",
            &format!(
                "eager per-call timing, min of {} (sweep: {sweep_iters}){}",
                o.iters,
                if o.no_graph_step { "" } else { " · graph step median of 100" }
            ),
        );
        let mut rows = vec![row!["", "", "A", "B measured", "B derived", "Δ measured (B − A)"]];
        let mut per_kernel: BTreeMap<String, [(usize, f32, usize); 2]> = BTreeMap::new(); // kernel -> per side (bytes, ms, count)
        let mut state_traffic = false;
        let mut perf_json = serde_json::Map::new();
        // Time a whole program on both sides at `e`; returns (step, Σ cuts)
        // per side and feeds the roofline accumulator.
        let mut step = |s: &mut Sides,
                        pname: &str,
                        e: &BTreeMap<String, u64>,
                        iters: usize,
                        roof: bool|
         -> Result<[(f32, f32); 2]> {
            let ta = s.a.rt.time_range(pname, e, 0, s.a.rt.call_count(pname)?, iters)?;
            let tb = s.b.rt.time_range(pname, e, 0, s.b.rt.call_count(pname)?, iters)?;
            let mut out = [(ta.iter().sum::<f32>(), 0f32), (tb.iter().sum::<f32>(), 0f32)];
            for sg in cuts_of(pname) {
                for (si, m, r, t) in [(0usize, &ma, sg.a, &ta), (1, &mb, sg.b, &tb)] {
                    for i in r.0..r.1 {
                        out[si].1 += t[i];
                        if !roof {
                            continue;
                        }
                        let acc = access(m, pname, i, i + 1);
                        let bytes: usize = acc.reads.iter().map(|n| live_bytes(m, n, e)).sum::<usize>()
                            + acc.writes.iter().map(|n| live_bytes(m, n, e)).sum::<usize>();
                        state_traffic |= !(acc.state_reads.is_empty() && acc.state_writes.is_empty());
                        let ent = per_kernel.entry(format!("{pname} · {}", m.programs[pname][i].op)).or_default();
                        ent[si].0 += bytes;
                        ent[si].1 += t[i];
                        ent[si].2 += 1;
                    }
                }
            }
            Ok(out)
        };
        // derived = A's step with A's cuts swapped for B's cuts (both timed
        // eager); the gap to the measurement is launch-gap / L2 interaction.
        let derived = |a_step: f32, st: [(f32, f32); 2]| a_step - st[0].1 + st[1].1;
        let push_step =
            |rows: &mut Vec<Vec<Cell>>, label: &str, n_cuts: usize, st: [(f32, f32); 2], graph: Option<(f32, f32)>| {
                let d = derived(st[0].0, st);
                rows.push(row![
                    Cell::bold(label),
                    "step, eager",
                    us(st[0].0),
                    us(st[1].0),
                    Cell::bold(us(d)),
                    delta(st[0].0, st[1].0),
                    Cell::dim(format!("measured − derived {:+.1} µs", (st[1].0 - d) * 1e3))
                ]);
                if let Some((ga, gb)) = graph {
                    let d = derived(ga, st);
                    rows.push(row![
                        "",
                        "step, graph (TPOT)",
                        us(ga),
                        us(gb),
                        Cell::bold(us(d)),
                        delta(ga, gb),
                        Cell::dim(format!("{:.0} → {:.0} tok/s", 1e3 / ga, 1e3 / gb))
                    ]);
                }
                rows.push(row![
                    "",
                    format!("Σ {n_cuts} cuts (the swap)"),
                    us(st[0].1),
                    us(st[1].1),
                    "",
                    delta(st[0].1, st[1].1)
                ]);
            };
        let n_cuts = |p: &str| cuts_of(p).len();
        // each decode-step program at the position after the workload
        for &p in &decode_progs {
            if n_cuts(p) == 0 || undriven.iter().any(|u| u == p) {
                continue;
            }
            s.a.stage_decode(*wl.decode.last().unwrap())?;
            s.b.stage_decode(*wl.decode.last().unwrap())?;
            let st = step(&mut s, p, &e1, o.iters, true)?;
            let mut graph = None;
            if !o.no_graph_step {
                s.a.rt.capture(p, &e1)?;
                s.b.rt.capture(p, &e1)?;
                graph = Some((s.a.rt.time_captured(p, &e1, 100)?, s.b.rt.time_captured(p, &e1, 100)?));
            }
            push_step(&mut rows, &format!("{p}  {TOKENS}=1"), n_cuts(p), st, graph);
            let mut j = json!({"tokens": 1, "eager_ms": {"a": st[0].0, "b": st[1].0, "b_derived": derived(st[0].0, st)}, "cut_ms": {"a": st[0].1, "b": st[1].1}});
            if let Some((ga, gb)) = graph {
                j["graph_ms"] = json!({"a": ga, "b": gb, "b_derived": derived(ga, st)});
            }
            perf_json.insert(p.into(), j);
        }
        // prefill: the tapped chunk length plus a sweep over the var range
        if n_cuts("prefill") > 0 && !undriven.iter().any(|u| u == "prefill") {
            let max = ma.vars[TOKENS].max;
            let tap_len = pre.len().min(chunk) as u64;
            let mut points: BTreeSet<u64> = [tap_len].into();
            if !o.no_sweep {
                points.extend([1u64, 16, 128, 512, 2048, 4096, max].into_iter().filter(|&t| t <= max));
            }
            let vocab = s.a.vocab();
            let mut rng = Rng(o.seed);
            let mut sweep = Vec::new();
            let mut sw = vec![
                row!["step A, eager"],
                row!["step B, measured"],
                row![Cell::bold("step B, derived")],
                row!["Δ measured"],
                row![Cell::dim("Σ cuts A")],
                row![Cell::dim("Σ cuts B")],
                row![Cell::dim("Δ")],
            ];
            for &t in &points {
                let tid: Vec<i64> = (0..t).map(|_| rng.below(vocab) as i64).collect();
                s.a.reset();
                s.b.reset();
                let e = s.a.stage_prefill(&tid)?;
                s.b.stage_prefill(&tid)?;
                let st = step(&mut s, "prefill", &e, if t == tap_len { o.iters } else { sweep_iters }, t == tap_len)?;
                if t == tap_len {
                    push_step(&mut rows, &format!("prefill  {TOKENS}={t}"), n_cuts("prefill"), st, None);
                }
                sw[0].push(Cell::from(us(st[0].0)));
                sw[1].push(Cell::from(us(st[1].0)));
                sw[2].push(Cell::bold(us(derived(st[0].0, st))));
                sw[3].push(pct_cell(st[0].0, st[1].0));
                sw[4].push(Cell::dim(us(st[0].1)));
                sw[5].push(Cell::dim(us(st[1].1)));
                sw[6].push(pct_cell(st[0].1, st[1].1));
                sweep.push(json!({"tokens": t, "eager_ms": {"a": st[0].0, "b": st[1].0, "b_derived": derived(st[0].0, st)}, "cut_ms": {"a": st[0].1, "b": st[1].1}}));
            }
            sec.table_h(std::mem::take(&mut rows));
            if points.len() > 1 {
                let mut hdr = row![format!("prefill · {TOKENS} =")];
                hdr.extend(points.iter().map(|t| Cell::from(t.to_string())));
                sw.insert(0, hdr);
                sec.table_h(sw);
            }
            perf_json.insert("prefill".into(), json!(sweep));
        }
        if !rows.is_empty() {
            sec.table_h(rows);
        }
        let mut rows = vec![row!["roofline", "moved / call", "A", "B"]];
        let mut roof = Vec::new();
        for (k, sides) in &per_kernel {
            let fmt = |(bytes, ms, n): (usize, f32, usize)| -> String {
                if n == 0 {
                    return "—".into();
                }
                let gbs = bytes as f64 / 1e9 / (ms as f64 / 1e3);
                format!("{} · {:.1} GB/s · {:.2}% of peak", us(ms / n as f32), gbs, gbs / o.peak_bw * 100.0)
            };
            let n = sides[0].2.max(sides[1].2);
            let per = sides.iter().find(|s| s.2 > 0).map_or(0, |s| s.0 / s.2);
            rows.push(row![
                Cell::bold(format!("{k} ×{n}")),
                format!("{}{}", kb(per), if state_traffic { " + opaque state" } else { "" }),
                fmt(sides[0]),
                fmt(sides[1])
            ]);
            roof.push(json!({"op": k, "bytes_per_call": per, "a": {"ms": sides[0].1, "n": sides[0].2}, "b": {"ms": sides[1].1, "n": sides[1].2}}));
        }
        sec.table_h(rows);
        perf_json.insert("roofline".into(), json!(roof));
        perf_json.insert("peak_bw_gbs".into(), json!(o.peak_bw));
        r.section(&sec);
        report["perf"] = Value::Object(perf_json);
    }

    // Is every local difference inside A's own noise band?
    let states_within = local_states.iter().all(|(p, sts)| {
        sts.iter().all(|(st, res)| {
            res.iter().all(|(_, c)| c.n_diff == 0 && c.outside == 0) || noisy_states.contains(&(p.clone(), st.clone()))
        })
    });
    let within_noise = !local_identical
        && !noise_clean
        && states_within
        && local_res.iter().all(|(p, bufs)| {
            bufs.iter().all(|(b, res)| {
                let worst_local = res
                    .iter()
                    .filter_map(|(_, c)| if c.value_identical() { None } else { c.max_ulp.or(Some(u64::MAX)) })
                    .max();
                let worst_noise = noise_res.get(p).and_then(|nb| nb.get(b)).and_then(|nr| {
                    nr.iter()
                        .filter_map(|(_, c)| if c.value_identical() { None } else { c.max_ulp.or(Some(u64::MAX)) })
                        .max()
                });
                match (worst_local, worst_noise) {
                    (None, _) => true,
                    (Some(l), Some(n)) => l <= n,
                    (Some(_), None) => false,
                }
            })
        });

    // ---- verdict
    let n_rows = logit_rows.len();
    let (code, verdict) = if !fuzz_ok {
        (1, "B violates a declared domain (or crashed) under fuzz".to_string())
    } else if let (Some(f), true) = (wide_flip, noise_clean) {
        (
            1,
            format!(
                "B changes the argmax end-to-end at {}: A {} → B {} with A's margin {:.4} above the logit Δ {:.4}",
                f.label, f.argmax_a, f.argmax_b, f.margin_a, f.max_abs
            ),
        )
    } else if !undriven.is_empty() {
        (2, "a changed program was not tapped — the workload driver can't stage it".to_string())
    } else if local_bit && fuzz_bit {
        (0, "bit-identical at every cut, real and perturbed inputs".to_string())
    } else if local_identical && fuzz_identical {
        (0, "value-identical at every cut (only signed zeros differ)".to_string())
    } else if logits_bit {
        (0, format!("cuts differ, but the end-to-end logits are bit-identical on all {n_rows} rows"))
    } else if logits_within {
        (0, format!("logit evidence: end-to-end logits move ≤ {logits_max_ulp:.2} ulp at their scale (limit {}) on {n_rows} rows, argmax agrees{}", o.logit_ulp, if n_near > 0 { format!(" except {n_near} near-tie{}", if n_near == 1 { "" } else { "s" }) } else { String::new() }))
    } else if within_noise && fuzz_identical {
        (0, "differences at every cut lie within A's own noise floor".to_string())
    } else if have_logits {
        (2, format!("cuts differ; end-to-end logits move up to {logits_max_ulp:.2} ulp at their scale on {n_rows} rows (limit {}), {n_flips} argmax flip{} ({n_near} near-tie){}", o.logit_ulp, if n_flips == 1 { "" } else { "s" }, if !noise_clean { " — A itself is not deterministic at some cuts" } else { "" }))
    } else {
        (2, "cuts differ beyond bit/value identity and no driven program writes logits — no oracle".to_string())
    };
    r.verdict(code, &verdict, t_start.elapsed(), o.out.as_deref());
    report["verdict"] = json!({"code": code, "pass": code == 0, "summary": verdict,
        "noise_clean": noise_clean, "within_noise": within_noise, "end_to_end_outputs_identical": e2e_outputs_identical, "end_to_end_outputs_differ": e2e_differs, "logits_bit_identical": logits_bit, "logits_within_limit": logits_within, "local_value_identical": local_identical, "local_bit_identical": local_bit,
        "fuzz_ok": fuzz_ok, "fuzz_value_identical": fuzz_identical, "fuzz_bit_identical": fuzz_bit});
    if let Some(p) = &o.out {
        std::fs::write(p, serde_json::to_string_pretty(&report)?)?;
    }
    Ok(code)
}
