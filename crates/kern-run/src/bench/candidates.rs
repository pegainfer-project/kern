//! `kern bench --candidate`: other implementations of a manifest's ops,
//! measured beside it, and the cheapest tiling of those that hold.
//!
//! A candidate is a file of `modules` and `ops`
//! ([`kern_manifest::candidate`]). Each is spliced into the manifest and
//! run over the same workload in the same process, the weights left on the
//! device from one to the next; the manifest itself runs first and last,
//! and what its two runs disagree by is the noise everything else is read
//! against.
//!
//! Time alone would pick a fast kernel that computes something else, so a
//! candidate is first held to the manifest's own outputs, scenario by
//! scenario: the tokens and counts the program hands back, and every row
//! of every distribution over tokens it writes (a `logits*` buffer, or any
//! float buffer as wide as the vocabulary), on every rank. It holds where
//! those are identical, or where every row is within the KL `kern test`
//! allows a swap and the outputs moved only where some row changed its
//! argmax, so that a token that moved under it moved on a tie. Anything else
//! is measured and reported and never chosen: whether a different
//! distribution is acceptable is a decision, not a measurement. A scenario
//! that hands nothing back holds a candidate to nothing, and nothing is
//! chosen from it either. A candidate is chosen from at the values of the
//! var where it held in every scenario.
//!
//! The choice is per op ([`pick`]): candidates replace whole ops at once,
//! but the kernel records charge every op separately, so one run prices a
//! candidate's every op, and ops of different candidates combine. The
//! combination is a manifest nobody ran, so it is run: the tiling spliced
//! in, held to the same outputs, its cost beside the sum the choice
//! predicted. Identical implementations combine into an identical
//! manifest; ones merely within the limit need not, each having moved the
//! outputs its own way. When their combination does not hold, the choice
//! is made again from the identical ones alone. What is written is the
//! tiling that held, as a candidate file of its own.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, ensure, Context, Result};
use kern_manifest::candidate::{splice, tile, Candidate, Piece};
use kern_manifest::types::{DType, Dim, Op};
use kern_manifest::{Protocol, Verified};
use kern_runtime::{At, HostWeights, Logit, Resident, Runtime, Scratch};
use kern_test::compare::TOP;
use kern_test::diff::{access, live_bytes, logits_of};
use serde_json::{json, Value};

use super::pick::{gaps, pick, Table, Tiling};
use super::workload::{Scenario, Workload};
use super::{output_fingerprints, preamble, say, say_timed, BenchOpts, Rank};
use crate::{Inputs, Vars};

/// How far a logits row may move and the candidate still stand for the
/// manifest, KL(manifest ‖ candidate) in nats: the default `kern test`
/// holds a swap to.
const LOGIT_KL: f64 = 0.01;

/// A candidate by the name of its file, spliced into the manifest.
struct Named {
    name: String,
    candidate: Candidate,
    manifest: Verified,
}

/// The candidates `paths` name: each a file, or every `.json` of a
/// directory, in name order.
fn read(paths: &[PathBuf], m: &Verified) -> Result<Vec<Named>> {
    let files: Vec<PathBuf> = paths
        .iter()
        .map(|p| match p.is_dir() {
            false => Ok(vec![p.clone()]),
            true => {
                let entries = std::fs::read_dir(p).with_context(|| format!("reading {}", p.display()))?;
                let paths = entries.map(|e| Ok(e?.path())).collect::<Result<Vec<_>>>()?;
                Ok(paths.into_iter().filter(|f| f.extension().is_some_and(|x| x == "json")).collect())
            }
        })
        .collect::<Result<Vec<_>>>()?
        .concat();
    let mut named = files
        .iter()
        .map(|f| {
            let name = f.file_stem().and_then(|s| s.to_str()).context("a candidate file has a name")?.to_string();
            let text = std::fs::read_to_string(f).with_context(|| format!("reading {}", f.display()))?;
            let candidate = Candidate::from_json(&text).with_context(|| format!("candidate {}", f.display()))?;
            let manifest = splice(m, &candidate).with_context(|| format!("candidate `{name}`"))?;
            Ok(Named { name, candidate, manifest })
        })
        .collect::<Result<Vec<_>>>()?;
    named.sort_by(|a, b| a.name.cmp(&b.name));
    ensure!(!named.is_empty(), "--candidate names no candidate file");
    if let Some(w) = named.windows(2).find(|w| w[0].name == w[1].name) {
        bail!("two candidate files are named `{}`", w[0].name);
    }
    Ok(named)
}

/// One scenario of one run: the program's p50 on its slowest rank and
/// what each op was charged of it, in µs a step.
struct Seen {
    id: String,
    weight: u64,
    vars: Vars,
    us: f64,
    ops: BTreeMap<String, f64>,
    /// Against the manifest's first run; `None` on that run itself.
    judged: Option<Judged>,
}

/// What a candidate must do to the manifest's outputs to be chosen from.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Admit {
    /// Leave them identical, or every distribution within the limit.
    Within,
    /// Leave them identical.
    Identical,
}

/// A run's outputs against the manifest's.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Verdict {
    /// Nothing was handed back.
    Unjudged,
    Identical,
    /// Every row of every distribution within the limit.
    Within,
    Differs,
}

impl Verdict {
    fn word(self) -> &'static str {
        match self {
            Verdict::Unjudged => "unjudged",
            Verdict::Identical => "identical",
            Verdict::Within => "within",
            Verdict::Differs => "differs",
        }
    }
}

impl Admit {
    fn word(self) -> &'static str {
        match self {
            Admit::Within => Verdict::Within.word(),
            Admit::Identical => Verdict::Identical.word(),
        }
    }

    fn admits(self, v: Verdict) -> bool {
        v == Verdict::Identical || (self == Admit::Within && v == Verdict::Within)
    }
}

impl Seen {
    /// Whether the run may be chosen from in this scenario.
    fn holds(&self, admit: Admit) -> bool {
        self.judged.is_some_and(|j| admit.admits(j.verdict()))
    }

    fn at(&self, var: Option<&str>) -> u64 {
        var.map_or(0, |v| self.vars.get(v).copied().unwrap_or(0))
    }
}

struct Run {
    seen: Vec<Seen>,
    took_s: f32,
}

impl Run {
    /// What the workload costs on this run, in µs.
    fn cost(&self) -> f64 {
        self.seen.iter().map(|s| s.weight as f64 * s.us).sum()
    }

    fn op_cost(&self, op: &str) -> f64 {
        self.seen.iter().map(|s| s.weight as f64 * s.ops.get(op).copied().unwrap_or(0.)).sum()
    }

    /// Scenarios in which the run does not hold.
    fn misses(&self, admit: Admit) -> usize {
        self.seen.iter().filter(|s| !s.holds(admit)).count()
    }

    /// Every scenario's judgment as one, and in how many the run holds.
    fn judged(&self) -> String {
        let all = self.seen.iter().filter_map(|s| s.judged).fold(Judged::default(), Judged::and);
        let (misses, of) = (self.misses(Admit::Within), self.seen.len());
        let holds = match misses {
            0 => String::new(),
            n if n == of => " · holds nowhere".to_string(),
            n => format!(" · holds in {} of {of} scenarios", of - n),
        };
        format!("{}{holds}", all.line())
    }
}

/// The manifest's logits of one scenario, kept on the device.
struct Rows {
    name: String,
    dtype: DType,
    cols: usize,
    bytes: usize,
    at: Scratch,
}

/// What the manifest handed back on one scenario.
struct Kept {
    outputs: Vec<Value>,
    logits: Vec<Rows>,
}

/// The buffers a program writes that hold a distribution over tokens:
/// those named `logits*`, and any float buffer as wide as the token
/// fill's domain, whatever it is called.
fn distributions(rt: &Runtime, p: &Protocol, program: &str) -> BTreeSet<String> {
    let m = &rt.manifest;
    let vocab = p
        .token_rows()
        .and_then(|t| m.buffers[&t.name].domain.as_ref())
        .and_then(|d| d.resolve(m, &p.vars(1, 1, 1, 1), &rt.provision()).ok())
        .and_then(|r| r.hi)
        .map(|hi| hi as u64 + 1);
    let float = |d: DType| matches!(d, DType::Bf16 | DType::F16 | DType::F32);
    let wide = |name: &String| {
        let b = &m.buffers[name];
        float(b.dtype) && matches!(b.shape.last(), Some(&Dim::Const(c)) if Some(c) == vocab)
    };
    let writes = access(m, program, 0..m.programs[program].calls.len()).writes;
    writes.into_iter().filter(wide).chain(logits_of(m, program)).collect()
}

fn keep(rt: &Runtime, p: &Protocol, s: &Scenario, vars: &Vars) -> Result<Kept> {
    let m = &rt.manifest;
    let logits = distributions(rt, p, &s.program)
        .into_iter()
        .filter_map(|name| {
            let b = &m.buffers[&name];
            let (bytes, cols) = (live_bytes(m, &name, vars), b.shape.last()?);
            let Dim::Const(cols) = *cols else { return None };
            let saved = |at| Rows { at, name: name.clone(), dtype: b.dtype, cols: cols as usize, bytes };
            (bytes > 0).then(|| Ok(saved(rt.save_buffer(&name, 0..bytes)?)))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Kept { outputs: output_fingerprints(rt, p, s.shape.groups)?, logits })
}

/// A run's outputs against the manifest's: one scenario's, or several summed.
#[derive(Default, Clone, Copy)]
struct Judged {
    /// Scenarios that handed anything back.
    scenarios: usize,
    /// Scenarios whose tokens or counts are not the manifest's.
    outputs_differ: usize,
    /// Of those, the ones in which no row changed its argmax: nothing the
    /// distributions show moved the outputs.
    unexplained: usize,
    rows: usize,
    rows_differ: usize,
    kl_sum: f64,
    kl_max: f64,
    flips: usize,
    /// Flips on a row beyond the limit: not ties.
    flips_beyond: usize,
    /// The widest top-1 to top-2 margin the manifest had on a row that flipped.
    flip_margin: f64,
}

/// A row's KL as `kern test` reads it: infinite when it is not a number
/// or one side has a NaN the other has not.
fn kl(l: &Logit) -> f64 {
    match l.cmp.nan_one_side == 0 && l.kl.is_finite() {
        true => l.kl.max(0.),
        false => f64::INFINITY,
    }
}

/// One rank's outputs against what the manifest handed back on it.
fn judge_rank(rt: &Runtime, p: &Protocol, s: &Scenario, kept: &Kept) -> Result<Judged> {
    // The outputs first: reading them binds this rank's context for the rows.
    let moved = output_fingerprints(rt, p, s.shape.groups)? != kept.outputs;
    let rows = kept
        .logits
        .iter()
        .map(|r| {
            Ok(rt.logits(r.dtype, r.cols, TOP, At::Scratch(&r.at, 0..r.bytes), At::Buffer(&r.name, 0..r.bytes))?)
        })
        .collect::<Result<Vec<_>>>()?
        .concat();
    let flips = || rows.iter().filter(|l| l.argmax_a != l.argmax_b);
    Ok(Judged {
        scenarios: (!kept.outputs.is_empty() || !rows.is_empty()) as usize,
        outputs_differ: moved as usize,
        unexplained: (moved && flips().count() == 0) as usize,
        rows: rows.len(),
        rows_differ: rows.iter().filter(|l| l.cmp.n_diff > 0).count(),
        kl_sum: rows.iter().map(kl).sum(),
        kl_max: rows.iter().map(kl).fold(0., f64::max),
        flips: flips().count(),
        flips_beyond: flips().filter(|l| kl(l) > LOGIT_KL).count(),
        flip_margin: flips().map(|l| l.top1 - l.top2).fold(0., f64::max),
    })
}

/// A scenario on every rank against what the manifest handed back on each:
/// one scenario, whatever any rank moved.
fn judge(ranks: &[Rank], p: &Protocol, s: &Scenario, kept: &[Kept]) -> Result<Judged> {
    let each = ranks.iter().zip(kept).map(|(r, k)| judge_rank(&r.rt, p, s, k)).collect::<Result<Vec<_>>>()?;
    let all = each.into_iter().fold(Judged::default(), Judged::and);
    Ok(Judged {
        scenarios: all.scenarios.min(1),
        outputs_differ: all.outputs_differ.min(1),
        unexplained: all.unexplained.min(1),
        ..all
    })
}

impl Judged {
    fn and(self, o: Judged) -> Judged {
        Judged {
            scenarios: self.scenarios + o.scenarios,
            outputs_differ: self.outputs_differ + o.outputs_differ,
            unexplained: self.unexplained + o.unexplained,
            rows: self.rows + o.rows,
            rows_differ: self.rows_differ + o.rows_differ,
            kl_sum: self.kl_sum + o.kl_sum,
            kl_max: self.kl_max.max(o.kl_max),
            flips: self.flips + o.flips,
            flips_beyond: self.flips_beyond + o.flips_beyond,
            flip_margin: self.flip_margin.max(o.flip_margin),
        }
    }

    /// Within the limit means every row is, and tokens or counts moved
    /// only where some row changed its argmax: on a tie. Outputs that moved
    /// under distributions that chose the same tokens differ, however
    /// close the distributions are.
    fn verdict(&self) -> Verdict {
        match self {
            j if j.scenarios == 0 => Verdict::Unjudged,
            j if j.outputs_differ == 0 && j.rows_differ == 0 => Verdict::Identical,
            j if j.rows > 0 && j.kl_max <= LOGIT_KL && j.unexplained == 0 => Verdict::Within,
            _ => Verdict::Differs,
        }
    }

    fn line(&self) -> String {
        match self.verdict() {
            Verdict::Unjudged => "unjudged · the programs hand nothing back".into(),
            Verdict::Identical => format!("identical · {} scenarios · {} logits rows", self.scenarios, self.rows),
            v => format!(
                "{} · KL max {:.1e} mean {:.1e} over {} logits rows · {} flips, {} beyond {LOGIT_KL} · tokens differ in {} of {} scenarios",
                v.word(),
                self.kl_max,
                self.kl_sum / self.rows.max(1) as f64,
                self.rows,
                self.flips,
                self.flips_beyond,
                self.outputs_differ,
                self.scenarios
            ),
        }
    }

    fn json(&self) -> Value {
        json!({"verdict": self.verdict().word(), "rows": self.rows, "rows_differ": self.rows_differ,
            "kl_max": self.kl_max, "kl_mean": self.kl_sum / self.rows.max(1) as f64, "flips": self.flips,
            "flips_beyond": self.flips_beyond, "flip_margin": self.flip_margin,
            "outputs_differ": self.outputs_differ > 0, "outputs_unexplained": self.unexplained > 0})
    }
}

/// The runs of one process: every manifest over the same plan, on the
/// weights the one before left resident, each held to what the first
/// handed back.
struct Session<'a> {
    inputs: &'a Inputs,
    w: &'a Workload,
    corpus: Vec<i64>,
    host_weights: HostWeights,
    resident: Option<Vec<Resident>>,
    /// What the manifest's first run handed back, per scenario per rank.
    kept: Option<Vec<Vec<Kept>>>,
    ids: Vec<String>,
    header: Value,
}

impl Session<'_> {
    fn measure(&mut self, m: &Verified, json_bytes: &[u8]) -> Result<Run> {
        let at = Instant::now();
        let (mut bench, plan) =
            super::open(m, self.inputs, self.w, &self.corpus, &self.host_weights, self.resident.take())?;
        bench.quiet = true;
        bench.prime(plan.source)?;
        let ids: Vec<String> = plan.scenarios.iter().map(|s| s.id.clone()).collect();
        if self.kept.is_none() {
            super::say_setup(&bench, &plan);
            self.header = preamble(m, json_bytes, self.w, &plan, &bench.ranks[0].probe, bench.ranks.len())?;
            self.ids = ids.clone();
        }
        ensure!(ids == self.ids, "a candidate changed which scenarios the manifest takes");
        let mut now = Vec::new();
        let seen = plan
            .scenarios
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let (_, priced) = bench.scenario(s)?;
                let (ranks, p) = (&bench.ranks, &bench.protocol);
                let judged = match &self.kept {
                    None => {
                        let kept = ranks.iter().map(|r| keep(&r.rt, p, s, &priced.vars));
                        now.push(kept.collect::<Result<Vec<_>>>()?);
                        None
                    }
                    Some(kept) => Some(judge(ranks, p, s, &kept[i])?),
                };
                let ops = priced.shares.iter().map(|(op, f)| (op.clone(), priced.us * f)).collect();
                let (id, weight) = (s.id.clone(), s.shape.weight);
                Ok(Seen { id, weight, vars: priced.vars, us: priced.us, ops, judged })
            })
            .collect::<Result<Vec<_>>>();
        // The weights outlive a run that failed: the next manifest binds over them all the same.
        self.resident = Some(bench.into_resident());
        let seen = seen?;
        self.kept.get_or_insert(now);
        Ok(Run { seen, took_s: at.elapsed().as_secs_f32() })
    }
}

/// The var an op's implementations turn on: the one their launches are
/// conditioned on, else the one their grids read; none when that is not
/// one var.
fn axis(impls: &[&Op]) -> Option<String> {
    let launches = || impls.iter().flat_map(|op| &op.imp.launches);
    let whens: BTreeSet<&str> = launches().filter_map(|l| l.when()).map(|w| w.var.as_str()).collect();
    let grids: BTreeSet<&str> =
        launches().filter_map(|l| l.kernel()).flat_map(|k| k.grid.iter().flat_map(|e| e.vars())).collect();
    match (whens.len(), grids.len()) {
        (1, _) => whens.first().map(|v| v.to_string()),
        (0, 1) => grids.first().map(|v| v.to_string()),
        _ => None,
    }
}

/// What `op` cost on each run at each value of `var` the manifest's own
/// run saw it at; infinite where a run after the first does not hold.
fn table(op: &str, var: Option<&str>, runs: &[&Run], admit: Admit) -> Table {
    let runs_op = |i: usize| runs[0].seen[i].ops.contains_key(op);
    let values: BTreeSet<u64> =
        (0..runs[0].seen.len()).filter(|&i| runs_op(i)).map(|i| runs[0].seen[i].at(var)).collect();
    let values: Vec<u64> = values.into_iter().collect();
    let cell = |c: usize, v: u64| {
        let here = || runs[c].seen.iter().enumerate().filter(move |(i, s)| runs_op(*i) && s.at(var) == v).map(|x| x.1);
        match c == 0 || here().all(|s| s.holds(admit)) {
            true => here().map(|s| s.weight as f64 * s.ops.get(op).copied().unwrap_or(0.)).sum(),
            false => f64::INFINITY,
        }
    };
    let costs = (0..runs.len()).map(|c| values.iter().map(|&v| cell(c, v)).collect()).collect();
    Table { values, costs }
}

/// The largest relative difference between two runs of one manifest, over
/// an op's measured values.
fn drift(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).filter(|(a, _)| **a > 0.).map(|(a, b)| (a - b).abs() / a).fold(0., f64::max)
}

fn seconds(us: f64) -> String {
    match us < 1e7 {
        true => format!("{:.3}", us / 1e6),
        false => format!("{:.1}", us / 1e6),
    }
}

fn percent(from: f64, to: f64) -> String {
    format!("{:+.1}%", (to / from - 1.) * 100.)
}

/// A tiling in words: `..32 fast · 33.. baseline`.
fn ranges_line(t: &Tiling, names: &[&str]) -> String {
    t.ranges
        .iter()
        .map(|r| {
            let (lo, hi) =
                (r.min.map_or(String::new(), |v| v.to_string()), r.max.map_or(String::new(), |v| v.to_string()));
            match (r.min, r.max) {
                (None, None) => names[r.candidate].to_string(),
                _ => format!("{lo}..{hi} {}", names[r.candidate]),
            }
        })
        .collect::<Vec<_>>()
        .join(" · ")
}

fn run_json(name: &str, r: &Run) -> Value {
    let scenarios: Vec<Value> = r
        .seen
        .iter()
        .map(|s| {
            json!({"id": s.id, "weight": s.weight, "vars": s.vars, "program_us": s.us, "ops_us": s.ops,
                "gate": s.judged.map(|j| j.json())})
        })
        .collect();
    json!({"name": name, "cost_us": r.cost(), "elapsed_s": r.took_s, "scenarios": scenarios})
}

/// The measured runs a choice is made from.
struct Measured<'a> {
    ops: BTreeSet<&'a str>,
    own: Candidate,
    base: Run,
    /// The candidates that ran, each with its run.
    runs: Vec<(&'a Named, Run)>,
    again: Run,
}

/// A choice: per op its tiling as a candidate, what it saves and its record.
struct Choice {
    picked: Candidate,
    predicted: f64,
    records: Vec<Value>,
}

impl Measured<'_> {
    /// Per op, the cheapest tiling of the manifest's own implementation and
    /// every candidate's that holds under `admit`.
    fn choose(&self, admit: Admit) -> Result<Choice> {
        let (runs, base) = (&self.runs, &self.base);
        let mut picks = Vec::new();
        for &op in &self.ops {
            let with: Vec<&(&Named, Run)> = runs.iter().filter(|(n, _)| n.candidate.ops.contains_key(op)).collect();
            let impls: Vec<&Op> =
                std::iter::once(&self.own.ops[op]).chain(with.iter().map(|(n, _)| &n.candidate.ops[op])).collect();
            let var = axis(&impls);
            let var = var.as_deref();
            let of: Vec<&Run> = std::iter::once(base).chain(with.iter().map(|(_, r)| r)).collect();
            let t = table(op, var, &of, admit);
            let noise = drift(&t.costs[0], &table(op, var, &[base, &self.again], admit).costs[1]);
            let Some(tiling) = pick(&t, noise) else { continue };
            let from: Vec<&Candidate> =
                std::iter::once(&self.own).chain(with.iter().map(|(n, _)| &n.candidate)).collect();
            let names: Vec<&str> =
                std::iter::once("baseline").chain(with.iter().map(|(n, _)| n.name.as_str())).collect();
            let chosen = match (var, tiling.ranges.as_slice()) {
                (Some(var), ranges) if ranges.len() > 1 => {
                    let pieces: Vec<Piece> =
                        ranges.iter().map(|r| Piece { from: from[r.candidate], min: r.min, max: r.max }).collect();
                    tile(op, var, &pieces).with_context(|| format!("tiling `{op}` over `{var}`"))?
                }
                (_, ranges) => from[ranges[0].candidate].only([op]),
            };
            let was: f64 = t.costs[0].iter().sum();
            let over = var.map_or(String::new(), |v| format!("`{v}` "));
            say(
                "pick",
                format!(
                    "{op} · {over}{} · {} → {} s",
                    ranges_line(&tiling, &names),
                    seconds(was),
                    seconds(tiling.cost)
                ),
            );
            let unmeasured = gaps(&t, &tiling);
            for (below, above) in &unmeasured {
                let between = format!("{}..{}", below + 1, above - 1);
                say("gap", format!("{op} · {over}{between} unmeasured · the boundary after {below} is a guess"));
            }
            let ranges: Vec<Value> = tiling
                .ranges
                .iter()
                .map(|r| json!({"candidate": names[r.candidate], "min": r.min, "max": r.max, "cost_us": r.cost}))
                .collect();
            // An infinite cost, where a candidate does not hold, is written as null.
            let record = json!({"op": op, "var": var, "values": t.values, "candidates": names, "costs_us": t.costs,
                "noise": noise, "ranges": ranges, "gaps": unmeasured, "cost_us": tiling.cost, "floor_us": tiling.floor, "baseline_us": was});
            picks.push((chosen, tiling.cost - was, record));
        }
        let picked = Candidate::join(picks.iter().map(|p| &p.0)).context("joining the picks")?;
        ensure!(!picked.ops.is_empty(), "no op of any candidate ran in the workload: nothing to choose");
        let predicted = base.cost() + picks.iter().map(|p| p.1).sum::<f64>();
        Ok(Choice { picked, predicted, records: picks.into_iter().map(|p| p.2).collect() })
    }
}

pub(super) fn run(o: &BenchOpts, inputs: &Inputs, w: &Workload, m: &Verified, json_bytes: &[u8]) -> Result<()> {
    let started = Instant::now();
    let named = read(&o.candidate, m)?;
    let ops: BTreeSet<&str> = named.iter().flat_map(|n| n.candidate.ops.keys().map(String::as_str)).collect();
    let own = Candidate::of(m, ops.iter().copied());
    say("ops", format!("{} with candidates · {}", ops.len(), ops.iter().copied().collect::<Vec<_>>().join(" · ")));

    let mut session = Session {
        inputs,
        w,
        corpus: super::corpus(inputs.tokenizer()?, w.seed)?,
        host_weights: HostWeights::new(),
        resident: None,
        kept: None,
        ids: Vec::new(),
        header: Value::Null,
    };
    let base = session.measure(m, json_bytes)?;
    let calls: u64 = base.seen.iter().map(|s| s.weight).sum();
    say_timed("baseline", format!("{} s · {calls} calls", seconds(base.cost())), started);
    let mut runs = Vec::new();
    let mut failed = Vec::new();
    for n in &named {
        let at = Instant::now();
        // A candidate that cannot be measured, one whose replay does not
        // reproduce its own tokens, is a finding about it, not about the run.
        let r = match session.measure(&n.manifest, n.manifest.to_json().as_bytes()) {
            Ok(r) => r,
            Err(e) => {
                say_timed("candidate", format!("{} · failed · {e:#}", n.name), at);
                failed.push(json!({"name": n.name, "failed": format!("{e:#}")}));
                continue;
            }
        };
        let mine: Vec<&str> = n.candidate.ops.keys().map(String::as_str).collect();
        let (was, is): (f64, f64) =
            (mine.iter().map(|op| base.op_cost(op)).sum(), mine.iter().map(|op| r.op_cost(op)).sum());
        say_timed(
            "candidate",
            format!(
                "{} · {} s · {} · its ops {} → {} s · {}",
                n.name,
                seconds(r.cost()),
                percent(base.cost(), r.cost()),
                seconds(was),
                seconds(is),
                r.judged()
            ),
            at,
        );
        runs.push((n, r));
    }
    let at = Instant::now();
    let again = session.measure(m, json_bytes)?;
    say_timed(
        "baseline",
        format!("{} s · {} · {}", seconds(again.cost()), percent(base.cost(), again.cost()), again.judged()),
        at,
    );

    let measured = Measured { ops, own, base, runs, again };
    let mut tried: Vec<(Admit, Choice, Run)> = Vec::new();
    for admit in [Admit::Within, Admit::Identical] {
        let choice = measured.choose(admit)?;
        if tried.last().is_some_and(|(_, c, _)| c.picked.to_json() == choice.picked.to_json()) {
            break;
        }
        let spliced = splice(m, &choice.picked).context("the picks as one manifest")?;
        let at = Instant::now();
        let confirm = session.measure(&spliced, spliced.to_json().as_bytes())?;
        let predicted = seconds(choice.predicted);
        say_timed(
            "confirm",
            format!("{} s measured · {predicted} s predicted · {}", seconds(confirm.cost()), confirm.judged()),
            at,
        );
        let holds = confirm.misses(admit) == 0;
        tried.push((admit, choice, confirm));
        if holds {
            break;
        }
        if admit == Admit::Within {
            say("again", "the picks together do not hold · choosing from the identical alone".into());
        }
    }
    let Measured { base, runs, again, .. } = measured;
    let held = tried.last().filter(|(admit, _, confirm)| confirm.misses(*admit) == 0);
    if let Some((_, choice, confirm)) = held {
        let picked = &choice.picked;
        let launches: usize = picked.ops.values().map(|op| op.imp.launches.len()).sum();
        say(
            "cost",
            format!(
                "{} → {} s · {} · {launches} launches of {} modules over {} ops",
                seconds(base.cost()),
                seconds(confirm.cost()),
                percent(base.cost(), confirm.cost()),
                picked.modules.len(),
                picked.ops.len()
            ),
        );
    }

    let mut report = std::mem::take(&mut session.header);
    report.as_object_mut().context("the report header is an object")?.remove("scenarios");
    report["logit_kl"] = json!(LOGIT_KL);
    report["baseline"] = run_json("baseline", &base);
    report["candidates"] = runs.iter().map(|(n, r)| run_json(&n.name, r)).chain(failed).collect();
    report["baseline_again"] = run_json("baseline", &again);
    report["choices"] = tried
        .iter()
        .map(|(admit, choice, confirm)| {
            let mut run = run_json("picked", confirm);
            run["predicted_us"] = json!(choice.predicted);
            run["holds"] = json!(confirm.misses(*admit) == 0);
            json!({"admit": admit.word(), "picks": choice.records, "picked": choice.picked, "confirm": run})
        })
        .collect();
    report["picked"] = serde_json::to_value(held.map(|(_, choice, _)| &choice.picked))?;
    super::write(&o.out, &report)?;
    say_timed("out", display(&o.out), started);
    ensure!(
        held.is_some(),
        "the picks do not hold together, and nothing was chosen: {} has every run",
        display(&o.out)
    );
    Ok(())
}

fn display(p: &Path) -> String {
    p.display().to_string()
}
