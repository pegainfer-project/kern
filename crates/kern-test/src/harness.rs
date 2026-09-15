//! Recording A: the seeded workload once, keeping at every span of every
//! program run what the span consumed and what A produced, and an image
//! of A's state at the start of each run; then A's noise floor, A's
//! outputs under fuzz and A's timings. Everything B is later judged
//! against lives in the [`Recording`]; A is not needed after this.
//! Generic over [`Side`]; every decision here is made on numbers the
//! side handed back.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use anyhow::{Context, Result};
use kern_manifest::protocol::{Forward, Rows};
use kern_manifest::types::{BufferKind, Manifest};
use kern_manifest::values;
use kern_manifest::{Protocol, Verified};

use crate::compare::{compare, diff_runs, is_float, perturb, Cmp, MODES};
use crate::diff::{access, constants, frontier_inputs, live_bytes, row_elems, Diff, Span};
use crate::report::{cap, kb, row, Finding, Noise};
use crate::workload::{self, Rng, Workload};
use crate::{Options, Side, Vars};

pub(crate) type Bytes = Vec<u8>;
/// `state -> [(offset, bytes)]`: the runs of a state a span changed.
pub(crate) type Runs = BTreeMap<String, Vec<(usize, Vec<u8>)>>;

/// One span of one program run, as A saw it, per rank.
pub(crate) struct SpanRec<B> {
    pub span: Span,
    /// A's frontier inputs: name, live bytes, the bytes (on the device).
    pub inputs: Vec<Vec<(String, usize, B)>>,
    /// What A wrote that both sides write (live prefix).
    pub ref_out: Vec<BTreeMap<String, Bytes>>,
    /// Kept runs only: pre-image and post-image of every state byte the
    /// span changed. A span with inout state is not idempotent (replaying
    /// it on its own output shifts the conv window again, advances the SSM
    /// again), so every replay first writes the pre-image back, and the
    /// post-image when it is done. States are opaque; this is byte-level.
    pub pre: Vec<Runs>,
    pub post: Vec<Runs>,
}

/// One program run of the workload.
pub(crate) struct Run<B> {
    pub program: String,
    /// `chunk 3` / `step 7`.
    pub label: String,
    pub vars: Vars,
    pub tokens: Vec<i64>,
    /// Positions the cursor moves past this run.
    pub advance: u64,
    /// Per rank: A's shared states at the start of the run. B starts the
    /// run from it; with per-layer write-sets that is the pre-image of
    /// every span of the run at one copy instead of a read per span.
    pub image: Vec<BTreeMap<String, B>>,
    pub spans: Vec<SpanRec<B>>,
}

/// A's `logits*` buffer after one run, per rank.
pub(crate) struct LogitsAt {
    pub label: String,
    pub buffer: String,
    pub vars: Vars,
    pub bytes: Vec<Bytes>,
}

pub(crate) struct NoiseRec {
    pub report: Noise,
    /// program -> buffer -> [(span label, cmp)]
    pub res: BTreeMap<String, BTreeMap<String, Vec<(String, Cmp)>>>,
    pub noisy_states: BTreeSet<(String, String)>,
    pub findings: Vec<Finding>,
}

/// One kept span under one perturbation: the inputs (per rank) and what
/// A made of them.
pub(crate) struct FuzzCase {
    pub at: (usize, usize),
    pub inputs: Vec<Vec<(String, Bytes)>>,
    pub out: Vec<BTreeMap<String, Bytes>>,
    pub states: Vec<Runs>,
}

pub(crate) struct FuzzRec {
    pub rounds: Vec<(&'static str, Vec<FuzzCase>)>,
    pub ints_kept: BTreeMap<String, BTreeSet<String>>,
    pub violations: Vec<String>,
}

pub(crate) struct StepRec {
    pub program: String,
    pub rows: u64,
    pub token: i64,
    pub times: Vec<f32>,
    pub graph_ms: Option<f32>,
}

pub(crate) struct SweepRec {
    pub rows: u64,
    pub tokens: Vec<i64>,
    pub iters: usize,
    pub times: Vec<f32>,
    pub tapped: bool,
}

pub(crate) struct PerfRec {
    pub steps: Vec<StepRec>,
    pub sweep: Vec<SweepRec>,
}

/// Everything B is judged against. Device bytes ([`Side::Buf`]) outlive
/// the side that recorded them.
pub struct Recording<B> {
    /// Seconds spent loading the sides; the caller adds each load.
    pub load_s: f32,
    pub(crate) record_s: f32,
    pub(crate) ma: Manifest,
    pub(crate) mb: Manifest,
    pub(crate) diff: Diff,
    pub(crate) wl: Workload,
    pub(crate) ranks: usize,
    pub(crate) chunk: Option<Forward>,
    steps: Vec<Forward>,
    pub(crate) runs: Vec<Run<B>>,
    /// `(run, span)` of the spans with state images: the first run of each
    /// driven program. Noise and fuzz replay these.
    pub(crate) kept: Vec<(usize, usize)>,
    pub(crate) logits: Vec<LogitsAt>,
    /// Output buffers after the workload, per rank.
    pub(crate) outputs: BTreeMap<String, Vec<Bytes>>,
    /// Per rank: A's shared states after the workload.
    pub(crate) states: Vec<BTreeMap<String, B>>,
    pub(crate) one_sided: BTreeSet<String>,
    pub(crate) noise: Option<NoiseRec>,
    pub(crate) fuzz: Option<FuzzRec>,
    pub(crate) perf: Option<PerfRec>,
}

impl<B> Recording<B> {
    pub(crate) fn spans_of(&self, program: &str) -> Vec<Span> {
        self.diff.spans.get(program).cloned().unwrap_or_default()
    }
    /// Programs with spans the workload does not drive.
    pub(crate) fn undriven(&self) -> Vec<String> {
        let driven: Vec<&str> = self.chunk.iter().chain(&self.steps).map(|f| f.name.as_str()).collect();
        self.diff.spans.keys().filter(|p| !driven.contains(&p.as_str())).cloned().collect()
    }
    pub(crate) fn shared_states(&self) -> Vec<String> {
        self.ma.states.keys().filter(|n| self.mb.states.contains_key(*n)).cloned().collect()
    }
    /// Bytes the kept spans' inputs and reference outputs take.
    pub(crate) fn snapshot_bytes(&self) -> usize {
        self.kept
            .iter()
            .map(|&(r, s)| {
                let sr = &self.runs[r].spans[s];
                sr.inputs.iter().flatten().map(|(_, b, _)| *b).sum::<usize>()
                    + sr.ref_out.iter().flatten().map(|(_, b)| b.len()).sum::<usize>()
            })
            .sum()
    }
    pub(crate) fn pre_image_bytes(&self) -> usize {
        self.kept
            .iter()
            .map(|&(r, s)| {
                self.runs[r].spans[s].pre.iter().flatten().flat_map(|(_, rs)| rs).map(|(_, b)| b.len()).sum::<usize>()
            })
            .sum()
    }
}

/// The workload's programs: the chunk program over the prompt, then every
/// fixed-rows forward in rotation (a caller may switch between them at any
/// step: same state contract).
fn driven(p: &Protocol) -> (Option<Forward>, Vec<Forward>) {
    let chunk = p.chunk().cloned();
    let steps = p.forwards.iter().filter(|f| matches!(f.rows, Rows::Const(_))).cloned().collect();
    (chunk, steps)
}

fn rows_of(f: &Forward) -> u64 {
    match f.rows {
        Rows::Const(r) => r,
        Rows::Var => 1,
    }
}

/// A span label on a rank, when there are several.
pub(crate) fn at_rank(ranks: usize, q: usize, label: &str) -> String {
    if ranks > 1 {
        format!("rank {q} {label}")
    } else {
        label.to_string()
    }
}

fn write_runs<S: Side>(c: &mut S, q: usize, runs: &Runs) -> Result<()> {
    for (name, rs) in runs {
        for (off, bytes) in rs {
            c.write_state(q, name, *off, bytes)?;
        }
    }
    Ok(())
}

/// The bytes now at the offsets of `runs`.
fn read_runs<S: Side>(c: &S, q: usize, runs: &Runs) -> Result<Runs> {
    runs.iter()
        .map(|(name, rs)| {
            let v = rs
                .iter()
                .map(|(off, b)| Ok((*off, c.read_state(q, name, *off..*off + b.len())?)))
                .collect::<Result<_>>()?;
            Ok((name.clone(), v))
        })
        .collect()
}

/// Bytes differing between two run sets over the same offsets.
pub(crate) fn runs_differ(x: &Runs, y: &Runs) -> BTreeMap<String, usize> {
    x.iter()
        .map(|(name, rs)| {
            let d =
                rs.iter().zip(&y[name]).map(|((_, a), (_, b))| a.iter().zip(b).filter(|(p, q)| p != q).count()).sum();
            (name.clone(), d)
        })
        .collect()
}

pub(crate) fn whole<S: Side>(c: &S, q: usize, state: &str) -> Result<Vec<u8>> {
    c.read_state(q, state, 0..c.state_bytes(state)?)
}

/// Put a side's shared states to the image a run started from.
pub(crate) fn image<S: Side>(c: &mut S, run: &Run<S::Buf>) -> Result<()> {
    for (q, img) in run.image.iter().enumerate() {
        for (n, b) in img {
            c.load_state(q, n, b)?;
        }
    }
    Ok(())
}

/// `logits*` buffers a program writes on both sides: the end-to-end oracle.
fn logits_of(ma: &Manifest, mb: &Manifest, prog: &str) -> Vec<String> {
    access(ma, prog, 0..ma.programs[prog].calls.len())
        .writes
        .into_iter()
        .filter(|n| n.starts_with("logits") && mb.buffers.contains_key(n))
        .collect()
}

pub(crate) fn read_logits<S: Side>(
    c: &S,
    ma: &Manifest,
    mb: &Manifest,
    prog: &str,
    e: &Vars,
    label: &str,
) -> Result<Vec<LogitsAt>> {
    logits_of(ma, mb, prog)
        .into_iter()
        .map(|n| {
            let len = live_bytes(ma, &n, e);
            let bytes = (0..c.ranks()).map(|q| c.read(q, &n, len)).collect::<Result<_>>()?;
            Ok(LogitsAt { label: label.to_string(), buffer: n, vars: e.clone(), bytes })
        })
        .collect()
}

/// Replay one recorded span on a side: from the recorded inputs, or from
/// `host_inputs` (perturbed, per rank). Returns the written buffers (live
/// prefix) and the state post-image over the recorded write-set, per rank.
/// The state is put back to A's pre-image first (so the span sees what it
/// saw in the recording, on either side) and to A's post-image after (so
/// the next span's reads see the reference, not this replay's output —
/// under fuzz, garbage).
pub(crate) fn replay_span<S: Side>(
    c: &mut S,
    sr: &SpanRec<S::Buf>,
    run: &Run<S::Buf>,
    side_b: bool,
    host_inputs: Option<&[Vec<(String, Bytes)>]>,
) -> Result<(Vec<BTreeMap<String, Bytes>>, Vec<Runs>)> {
    let ranks = c.ranks();
    for q in 0..ranks {
        write_runs(c, q, &sr.pre[q])?;
        match host_inputs {
            Some(v) => {
                for (n, bytes) in &v[q] {
                    c.write(q, n, bytes)?;
                }
            }
            None => {
                for (n, len, b) in &sr.inputs[q] {
                    c.load(q, n, *len, b)?;
                }
            }
        }
    }
    let r = if side_b { sr.span.b.clone() } else { sr.span.a.clone() };
    c.run(&run.program, &run.vars, r)?;
    let mut out = Vec::new();
    let mut st = Vec::new();
    for q in 0..ranks {
        let mut o = BTreeMap::new();
        for (n, r) in &sr.ref_out[q] {
            o.insert(n.clone(), c.read(q, n, r.len())?);
        }
        out.push(o);
        st.push(read_runs(c, q, &sr.post[q])?);
        write_runs(c, q, &sr.post[q])?;
    }
    Ok((out, st))
}

/// Values a written buffer's declared domain rules out: `side name[i] = v`.
pub(crate) fn domain_violations<S: Side>(
    c: &S,
    m: &Manifest,
    vars: &Vars,
    side: &str,
    out: &[BTreeMap<String, Bytes>],
) -> Result<Vec<String>> {
    let mut v = Vec::new();
    for (q, o) in out.iter().enumerate() {
        for (name, bytes) in o {
            let Some(d) = &m.buffers[name].domain else { continue };
            let r = d.resolve(m, vars, &c.provision())?;
            let vals = values::to_f64(m.buffers[name].dtype, bytes);
            if let Some(i) = vals.iter().position(|x| !r.contains(*x)) {
                v.push(format!("{} {name}[{i}] = {} outside domain", at_rank(c.ranks(), q, side), vals[i]));
            }
        }
    }
    Ok(v)
}

/// Record A on the workload `o` and the diff's spans: what every span
/// consumed and produced, then A's noise floor, fuzz outputs and timings.
/// `mb` is B's manifest: which buffers and states the two sides share,
/// and what B's `once` programs make their own.
pub fn record<S: Side>(
    o: &Options,
    diff: Diff,
    mb: &Verified,
    a: &mut S,
    out: &mut dyn FnMut(&[String]),
) -> Result<Recording<S::Buf>> {
    let t0 = Instant::now();
    let ma: Manifest = (**a.manifest()).clone();
    let pa = Protocol::check(a.manifest()).context("A does not fit the serving protocol")?;
    let (chunk_f, step_fs) = driven(&pa);
    let ranks = a.ranks();
    let wl = workload::sample(o, &ma, &pa, a.provision(), a.page())?;
    let mut rec = Recording {
        load_s: 0.0,
        record_s: 0.0,
        ma: ma.clone(),
        mb: (**mb).clone(),
        diff,
        wl: wl.clone(),
        ranks,
        chunk: chunk_f.clone(),
        steps: step_fs.clone(),
        runs: Vec::new(),
        kept: Vec::new(),
        logits: Vec::new(),
        outputs: BTreeMap::new(),
        states: Vec::new(),
        one_sided: BTreeSet::new(),
        noise: None,
        fuzz: None,
        perf: None,
    };
    let shared = rec.shared_states();
    let pb = Protocol::check(mb).context("B does not fit the serving protocol")?;
    let mut fixed = constants(&ma, &pa.once);
    fixed.extend(constants(mb, &pb.once));
    let mb: &Manifest = mb;
    let chunk_name = chunk_f.as_ref().map_or("", |f| f.name.as_str());

    // ---- the workload: one run at a time, spans recorded as A passes them
    a.reset();
    let mut i = 0;
    let mut n_chunks = 0usize;
    while i < wl.prefill.len() {
        let c = (wl.prefill.len() - i).min(wl.chunk);
        let tokens = wl.prefill[i..i + c].to_vec();
        let keep = n_chunks == 0;
        record_run(a, &mut rec, chunk_name, format!("chunk {n_chunks}"), tokens, c as u64, keep, &shared, &fixed)?;
        a.advance(c as u64);
        i += c;
        n_chunks += 1;
    }
    // Every step stages the drawn token in a forward's rows at the cursor
    // and advances one position: a wide forward's extra rows are
    // overwritten by the next step, on both sides alike.
    for (k, &tok) in wl.decode.iter().enumerate() {
        let f = &step_fs[k % step_fs.len()];
        let tokens = vec![tok; rows_of(f) as usize];
        record_run(a, &mut rec, &f.name, format!("step {k}"), tokens, 1, k < step_fs.len(), &shared, &fixed)?;
        a.advance(1);
    }
    // What a caller would get: the outputs and the states after the workload.
    let e_last = rec.runs.last().map(|r| r.vars.clone()).unwrap_or_default();
    for (name, b) in &ma.buffers {
        if b.kind == BufferKind::Output && mb.buffers.contains_key(name) {
            let len = live_bytes(&ma, name, &e_last);
            rec.outputs.insert(name.clone(), (0..ranks).map(|q| a.read(q, name, len)).collect::<Result<_>>()?);
        }
    }
    for q in 0..ranks {
        let mut img = BTreeMap::new();
        for n in &shared {
            let mut b = a.alloc(q, a.state_bytes(n)?)?;
            a.save_state(q, n, &mut b)?;
            img.insert(n.clone(), b);
        }
        rec.states.push(img);
    }
    let workload_s = t0.elapsed().as_secs_f32();

    // ---- noise floor: A's kept spans replayed from their own recording
    if o.noise {
        let t_n = Instant::now();
        let mut res: BTreeMap<String, BTreeMap<String, Vec<(String, Cmp)>>> = BTreeMap::new();
        let mut noisy_states = BTreeSet::new();
        let mut findings = Vec::new();
        let mut state_noise = Vec::new();
        let (mut compared, mut clean) = (0usize, 0usize);
        let mut cur = None;
        for &(r, s) in &rec.kept {
            let run = &rec.runs[r];
            if cur != Some(r) {
                image(a, run)?;
                cur = Some(r);
            }
            let sr = &run.spans[s];
            let (out_a, st) = replay_span(a, sr, run, false, None)?;
            for q in 0..ranks {
                let label = at_rank(ranks, q, &format!("{} {}", run.label, sr.span.label()));
                for (n, b) in &out_a[q] {
                    let c = compare(ma.buffers[n].dtype, &sr.ref_out[q][n], b);
                    compared += 1;
                    if c.identical() {
                        clean += 1;
                    } else {
                        findings.push(Finding::of(&run.program, &label, n, &c));
                    }
                    res.entry(run.program.clone()).or_default().entry(n.clone()).or_default().push((label.clone(), c));
                }
                for (name, d) in runs_differ(&st[q], &sr.post[q]) {
                    if d > 0 {
                        noisy_states.insert((run.program.clone(), name.clone()));
                        state_noise.push(format!("state {name}: {d} bytes at {} {label}", run.program));
                    }
                }
            }
        }
        let (shown, omitted) = cap(findings.clone(), Finding::severity);
        let report = Noise {
            compared,
            clean,
            findings: shown,
            omitted,
            states: state_noise,
            elapsed_s: t_n.elapsed().as_secs_f32(),
        };
        rec.noise = Some(NoiseRec { report, res, noisy_states, findings });
    }

    // ---- fuzz: the kept spans on perturbed inputs, A's side
    if o.fuzz > 0 {
        let mut rng = Rng(o.seed);
        let mut ints_kept: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        let mut violations = Vec::new();
        let mut rounds = Vec::new();
        for round in 0..o.fuzz {
            let mode = MODES[round % MODES.len()];
            let mut cases = Vec::new();
            let mut cur = None;
            for &(r, s) in &rec.kept {
                let run = &rec.runs[r];
                if cur != Some(r) {
                    image(a, run)?;
                    cur = Some(r);
                }
                let sr = &run.spans[s];
                let mut inputs = Vec::new();
                for q in 0..ranks {
                    let mut v = Vec::new();
                    for (name, len, buf) in &sr.inputs[q] {
                        let decl = &ma.buffers[name];
                        let tapped = a.bytes(buf, *len)?;
                        if !is_float(decl.dtype) {
                            // sequence layout, indices, page tables: structure,
                            // not values — a random one is a workload no caller
                            // produces, and the manifest says nothing about rows
                            // outside it
                            ints_kept.entry(run.program.clone()).or_default().insert(name.clone());
                            v.push((name.clone(), tapped));
                            continue;
                        }
                        let base = values::to_f64(decl.dtype, &tapped);
                        let vals =
                            perturb(&mut rng, round % MODES.len(), &base, row_elems(&ma, name, &run.vars), decl.dtype);
                        v.push((name.clone(), values::from_f64(decl.dtype, &vals)));
                    }
                    inputs.push(v);
                }
                let at = format!("{} span {}", run.program, sr.span.label());
                let (out, states) = replay_span(a, sr, run, false, Some(&inputs))
                    .with_context(|| format!("A crashed under fuzz ({mode}) at {at}"))?;
                // A violation on the reference is the reference misbehaving.
                violations.extend(
                    domain_violations(a, &ma, &run.vars, "A", &out)?.into_iter().map(|v| format!("{mode} {at}: {v}")),
                );
                cases.push(FuzzCase { at: (r, s), inputs, out, states });
            }
            rounds.push((mode, cases));
        }
        rec.fuzz = Some(FuzzRec { rounds, ints_kept, violations });
    }

    // ---- perf: A's side of every timing
    if o.perf {
        let undriven = rec.undriven();
        let mut steps = Vec::new();
        let last_tok = *wl.decode.last().unwrap_or(&0);
        for f in &step_fs {
            let p = f.name.as_str();
            if rec.spans_of(p).is_empty() || undriven.iter().any(|u| u == p) {
                continue;
            }
            let e = a.stage(&vec![last_tok; rows_of(f) as usize])?;
            let times = a.time(p, &e, 0..a.calls(p)?, o.iters)?;
            let graph_ms = if o.graph_step {
                a.capture(p, &e)?;
                Some(a.time_captured(p, &e, 100)?)
            } else {
                None
            };
            steps.push(StepRec { program: p.into(), rows: rows_of(f), token: last_tok, times, graph_ms });
        }
        let mut sweep = Vec::new();
        if chunk_f.is_some() && !rec.spans_of(chunk_name).is_empty() && !undriven.iter().any(|u| u == chunk_name) {
            // The manifest's widest chunk, within what the sequence was provisioned.
            let max = pa.rows.max.min(a.provision().tokens);
            let tap_len = wl.prefill.len().min(wl.chunk) as u64;
            let mut points: BTreeSet<u64> = [tap_len].into();
            if o.sweep {
                points.extend([1u64, 16, 128, 512, 2048, 4096, max].into_iter().filter(|&t| t <= max));
            }
            let vocab = a.vocab();
            let mut rng = Rng(o.seed);
            for &t in &points {
                let tokens: Vec<i64> = (0..t).map(|_| rng.below(vocab) as i64).collect();
                a.reset();
                let e = a.stage(&tokens)?;
                let iters = if t == tap_len { o.iters } else { o.iters.min(10) };
                let times = a.time(chunk_name, &e, 0..a.calls(chunk_name)?, iters)?;
                sweep.push(SweepRec { rows: t, tokens, iters, times, tapped: t == tap_len });
            }
        }
        rec.perf = Some(PerfRec { steps, sweep });
    }
    rec.record_s = t0.elapsed().as_secs_f32();
    let image_bytes = image_bytes(&rec, a)?;
    out(&[row(
        "record",
        format!(
            "A: {} runs · {} spans kept ({}) · state images {} · workload {}",
            rec.runs.len(),
            rec.kept.len(),
            kb(rec.snapshot_bytes()),
            kb(image_bytes),
            crate::report::secs(workload_s)
        ),
        Some(rec.record_s),
    )]);
    Ok(rec)
}

/// Bytes the per-run state images hold.
fn image_bytes<S: Side>(rec: &Recording<S::Buf>, c: &S) -> Result<usize> {
    let mut n = 0;
    for run in &rec.runs {
        for img in &run.image {
            for name in img.keys() {
                n += c.state_bytes(name)?;
            }
        }
    }
    Ok(n)
}

/// Stage and run one program on A, recording every span on the way.
#[allow(clippy::too_many_arguments)]
fn record_run<S: Side>(
    a: &mut S,
    rec: &mut Recording<S::Buf>,
    pname: &str,
    label: String,
    tokens: Vec<i64>,
    advance: u64,
    keep: bool,
    shared: &[String],
    fixed: &BTreeSet<String>,
) -> Result<()> {
    let ranks = rec.ranks;
    let e = a.stage(&tokens)?;
    let mut image = Vec::new();
    for q in 0..ranks {
        let mut img = BTreeMap::new();
        for n in shared {
            let mut b = a.alloc(q, a.state_bytes(n)?)?;
            a.save_state(q, n, &mut b).with_context(|| format!("imaging state `{n}` before {pname} {label}"))?;
            img.insert(n.clone(), b);
        }
        image.push(img);
    }
    let (ma, mb) = (&rec.ma, &rec.mb);
    let segs = rec.spans_of(pname);
    let mut spans = Vec::new();
    let mut ia = 0;
    for span in &segs {
        a.run(pname, &e, ia..span.a.start)?;
        ia = span.a.end;
        // Inputs both sides have and A can hand over: a buffer only B knows
        // (an intermediate of its own), a weight, or what a `once` program
        // made of one (B packs it its own way) is B's to produce.
        let names: Vec<String> = frontier_inputs(ma, pname, span.a.clone())
            .union(&frontier_inputs(mb, pname, span.b.clone()))
            .filter(|n| ma.buffers.contains_key(*n) && mb.buffers.contains_key(*n))
            .filter(|n| ma.buffers[*n].kind != BufferKind::Weight && !fixed.contains(*n))
            .cloned()
            .collect();
        let mut inputs = Vec::new();
        for q in 0..ranks {
            let mut v = Vec::new();
            for n in &names {
                let bytes = live_bytes(ma, n, &e);
                if bytes == 0 {
                    continue;
                }
                let mut buf = a.alloc(q, bytes)?;
                a.save(q, n, bytes, &mut buf)
                    .with_context(|| format!("recording `{n}` ({bytes} B) at {pname} span {}", span.label()))?;
                v.push((n.clone(), bytes, buf));
            }
            inputs.push(v);
        }
        let (aa, ab) = (access(ma, pname, span.a.clone()), access(mb, pname, span.b.clone()));
        // State is compared on the span's write-set — the bytes A's run
        // changed — not on the whole allocation (the rest is other layers'
        // history). Whole-state reads only on the kept run; every other run
        // B starts from A's image of the run.
        let touched: Vec<String> =
            aa.state_writes.union(&ab.state_writes).filter(|s| shared.contains(s)).cloned().collect();
        let mut a_pre: Vec<BTreeMap<String, Vec<u8>>> = Vec::new();
        if keep {
            for q in 0..ranks {
                a_pre.push(touched.iter().map(|st| Ok((st.clone(), whole(a, q, st)?))).collect::<Result<_>>()?);
            }
        }
        a.run(pname, &e, span.a.clone())?;
        let (mut pre, mut post) = (Vec::new(), Vec::new());
        for q in 0..ranks {
            let (mut pq, mut oq) = (Runs::new(), Runs::new());
            if keep {
                for st in &touched {
                    let after = whole(a, q, st)?;
                    let runs = diff_runs(&a_pre[q][st], &after);
                    oq.insert(
                        st.clone(),
                        runs.iter().map(|(off, b)| (*off, after[*off..*off + b.len()].to_vec())).collect(),
                    );
                    pq.insert(st.clone(), runs);
                }
            }
            pre.push(pq);
            post.push(oq);
        }
        let written: Vec<String> = aa.writes.intersection(&ab.writes).cloned().collect();
        rec.one_sided.extend(aa.writes.symmetric_difference(&ab.writes).cloned());
        let mut ref_out = Vec::new();
        for q in 0..ranks {
            ref_out.push(
                written
                    .iter()
                    .map(|n| {
                        let bytes = a
                            .read(q, n, live_bytes(ma, n, &e))
                            .with_context(|| format!("reading `{n}` after {pname} span {}", span.label()))?;
                        Ok((n.clone(), bytes))
                    })
                    .collect::<Result<_>>()?,
            );
        }
        spans.push(SpanRec { span: span.clone(), inputs, ref_out, pre, post });
    }
    a.run(pname, &e, ia..a.calls(pname)?)?;
    let logits_label = if label.starts_with("chunk") { format!("prefill {label}") } else { label.clone() };
    rec.logits.extend(read_logits(a, ma, mb, pname, &e, &logits_label)?);
    let r = rec.runs.len();
    if keep {
        rec.kept.extend((0..spans.len()).map(|s| (r, s)));
    }
    rec.runs.push(Run { program: pname.into(), label, vars: e, tokens, advance, image, spans });
    Ok(())
}
