//! The stages that need the two sides: tap, noise floor, fuzz, perf, and
//! the verdict over what they measured. Generic over [`Side`]; every
//! decision here is made on numbers the sides handed back.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use anyhow::{Context, Result};
use kern_manifest::protocol::{Forward, Rows};
use kern_manifest::types::{BufferKind, Manifest};
use kern_manifest::values;
use kern_manifest::Protocol;
use serde_json::{json, Value};

use crate::compare::{compare, diff_runs, is_float, logit_row, perturb, Cmp, LogitRow, MODES};
use crate::diff::{access, frontier_inputs, live_bytes, row_elems, Access, Diff, Span};
use crate::report::*;
use crate::workload::{self, Rng};
use crate::{Options, Side, Vars};

/// The two sides, and what it cost to bring them up.
pub struct Sides<S> {
    pub a: S,
    pub b: S,
    pub load_s: f32,
}

/// One span with a real-workload snapshot: what it consumed (frontier
/// inputs) and what A produced from them.
struct Snapshot<B> {
    program: String,
    span: Span,
    vars: Vars,
    /// A's frontier inputs: name, live bytes, the bytes (on the device).
    inputs: Vec<(String, usize, B)>,
    /// What A wrote from them, per buffer (live prefix).
    ref_out: BTreeMap<String, Vec<u8>>,
    /// Pre-image and post-image of every state byte the span changed (A,
    /// around the span): `state -> [(offset, bytes)]`. A span with inout
    /// state is not idempotent — replaying it on its own output shifts the
    /// conv window again, advances the SSM again — so every replay first
    /// writes the pre-image back, and the post-image when it is done.
    /// States are opaque; this is byte-level, no model knowledge.
    pre_states: BTreeMap<String, Vec<(usize, Vec<u8>)>>,
    post_states: BTreeMap<String, Vec<(usize, Vec<u8>)>>,
    /// Which run's state image the replay starts from: 0 = zeros (prefill
    /// chunk 0), 1 = A's state after prefill (decode step 0). The write-set
    /// alone is not enough once later runs have moved bytes the span reads
    /// but did not change.
    image: usize,
}

/// How B's write to a state compares with A's, on this span: bytes of A's
/// write-set (`set`), how many of them B wrote differently (`n_diff`), and
/// how many bytes B changed outside A's write-set (`outside`).
#[derive(Clone, Default)]
struct StateCmp {
    set: usize,
    n_diff: usize,
    outside: usize,
}

type Runs = BTreeMap<String, Vec<(usize, Vec<u8>)>>;

fn write_runs<S: Side>(c: &mut S, runs: &Runs) -> Result<()> {
    for (name, rs) in runs {
        for (off, bytes) in rs {
            c.write_state(name, *off, bytes)?;
        }
    }
    Ok(())
}

/// The bytes now at the offsets of `runs`.
fn read_runs<S: Side>(c: &S, runs: &Runs) -> Result<Runs> {
    runs.iter()
        .map(|(name, rs)| {
            let v = rs
                .iter()
                .map(|(off, b)| Ok((*off, c.read_state(name, *off..*off + b.len())?)))
                .collect::<Result<_>>()?;
            Ok((name.clone(), v))
        })
        .collect()
}

/// Bytes differing between two run sets over the same offsets.
fn runs_differ(x: &Runs, y: &Runs) -> BTreeMap<String, usize> {
    x.iter()
        .map(|(name, rs)| {
            let d =
                rs.iter().zip(&y[name]).map(|((_, a), (_, b))| a.iter().zip(b).filter(|(p, q)| p != q).count()).sum();
            (name.clone(), d)
        })
        .collect()
}

/// Compare the buffers a span wrote on both sides (live prefix at
/// `vars`). Buffers written by one side only are internal to that side's
/// implementation.
fn compare_written<S: Side>(
    a: &S,
    b: &S,
    acc_a: &Access,
    acc_b: &Access,
    vars: &Vars,
) -> Result<(BTreeMap<String, Cmp>, Vec<String>)> {
    let mut bufs = BTreeMap::new();
    let mut one_sided = Vec::new();
    for name in acc_a.writes.union(&acc_b.writes) {
        if !(acc_a.writes.contains(name) && acc_b.writes.contains(name)) {
            one_sided.push(name.clone());
            continue;
        }
        let bytes = live_bytes(a.manifest(), name, vars);
        let x = a.read(name, bytes)?;
        let y = b.read(name, bytes)?;
        bufs.insert(name.clone(), compare(a.manifest().buffers[name].dtype, &x, &y));
    }
    Ok((bufs, one_sided))
}

/// Replay one snapshot's span on a side: from the snapshot's inputs, or
/// from `host_inputs` (perturbed). Returns the written buffers (live
/// prefix) and the state post-image over the snapshot's write-set. The
/// state is put back to A's pre-image first (so the span sees what it saw
/// in the tap, on either side) and to A's post-image after (so the next
/// span's reads see the reference, not this replay's output — under fuzz,
/// garbage).
fn replay<S: Side>(
    c: &mut S,
    sn: &Snapshot<S::Buf>,
    side_b: bool,
    host_inputs: Option<&[(String, Vec<u8>)]>,
) -> Result<(BTreeMap<String, Vec<u8>>, Runs)> {
    write_runs(c, &sn.pre_states)?;
    match host_inputs {
        Some(v) => {
            for (n, bytes) in v {
                c.write(n, bytes)?;
            }
        }
        None => {
            for (n, len, b) in &sn.inputs {
                c.load(n, *len, b)?;
            }
        }
    }
    let r = if side_b { sn.span.b.clone() } else { sn.span.a.clone() };
    let run = c.run(&sn.program, &sn.vars, r);
    let mut out = BTreeMap::new();
    let mut st = Runs::new();
    if run.is_ok() {
        for (n, r) in &sn.ref_out {
            out.insert(n.clone(), c.read(n, r.len())?);
        }
        st = read_runs(c, &sn.post_states)?;
        write_runs(c, &sn.post_states)?;
    }
    run?;
    Ok((out, st))
}

fn cmp_out<B>(m: &Manifest, sn: &Snapshot<B>, out: &BTreeMap<String, Vec<u8>>) -> BTreeMap<String, Cmp> {
    out.iter().map(|(n, b)| (n.clone(), compare(m.buffers[n].dtype, &sn.ref_out[n], b))).collect()
}

/// Everything after the static diff: tap, noise, fuzz, perf, verdict.
/// `out` receives each section's lines as it finishes.
pub fn run<S: Side>(o: &Options, diff: Diff, s: &mut Sides<S>, out: &mut dyn FnMut(&[String])) -> Result<Report> {
    let t_start = Instant::now();
    let elapsed = |t: &Instant| t.elapsed().as_secs_f32();
    let ma: Manifest = (**s.a.manifest()).clone();
    let mb: Manifest = (**s.b.manifest()).clone();
    let pa = Protocol::check(s.a.manifest()).context("A does not fit the serving protocol")?;
    // What the workload drives: the chunk program over the prompt, then
    // every fixed-rows forward in rotation (a caller may switch between
    // them at any step: same state contract).
    let chunk_f = pa.chunk().cloned();
    let step_fs: Vec<Forward> = pa.forwards.iter().filter(|f| matches!(f.rows, Rows::Const(_))).cloned().collect();
    let driven: Vec<&str> = chunk_f.iter().chain(&step_fs).map(|f| f.name.as_str()).collect();
    let rows_of = |f: &Forward| match f.rows {
        Rows::Const(r) => r,
        Rows::Var => 1,
    };
    let vars_of = |f: &Forward| pa.vars(1, rows_of(f), rows_of(f));
    let spans: BTreeMap<String, Vec<Span>> = diff.spans.clone();
    let mut sum = Summary { a: o.a.clone(), b: o.b.clone(), diff, ..Default::default() };
    let wl = workload::sample(o, &ma, &pa, s.a.provision(), s.a.page())?;
    let spans_of = |p: &str| -> Vec<Span> { spans.get(p).cloned().unwrap_or_default() };

    // ---- 2. tap: A and B in lockstep on the workload, snapshot at every span
    let t_tap = Instant::now();
    let mut snaps: Vec<Snapshot<S::Buf>> = Vec::new();
    let mut snap_state_bytes = 0usize;
    // program -> buffer -> [(span label, cmp)]
    let mut local_res: BTreeMap<String, BTreeMap<String, Vec<(String, Cmp)>>> = BTreeMap::new();
    let mut local_states: BTreeMap<String, BTreeMap<String, Vec<(String, StateCmp)>>> = BTreeMap::new();
    let mut one_sided_all: BTreeSet<String> = BTreeSet::new();
    let mut n_chunks = 0usize;
    s.a.reset();
    s.b.reset();
    // Run one program in lockstep; at each span snapshot A's frontier inputs
    // (from A, before the span), A's outputs after, and compare B's outputs.
    let mut lockstep =
        |s: &mut Sides<S>, pname: &str, e: &Vars, label_prefix: &str, keep: bool, image: usize| -> Result<()> {
            let Some(segs) = spans.get(pname) else {
                s.a.run(pname, e, 0..s.a.calls(pname)?)?;
                s.b.run(pname, e, 0..s.b.calls(pname)?)?;
                return Ok(());
            };
            let (mut ia, mut ib) = (0, 0);
            for span in segs {
                s.a.run(pname, e, ia..span.a.start)?;
                s.b.run(pname, e, ib..span.b.start)?;
                ia = span.a.end;
                ib = span.b.end;
                let label = format!("{label_prefix}{}", span.label());
                let input_names: BTreeSet<String> = frontier_inputs(&ma, pname, span.a.clone())
                    .union(&frontier_inputs(&mb, pname, span.b.clone()))
                    .cloned()
                    .collect();
                let mut inputs = Vec::new();
                for n in &input_names {
                    if ma.buffers[n].kind != BufferKind::Weight {
                        let bytes = live_bytes(&ma, n, e);
                        let mut buf = s.a.alloc(bytes)?;
                        s.a.save(n, bytes, &mut buf)?;
                        inputs.push((n.clone(), bytes, buf));
                    }
                }
                // B runs the span from A's inputs: the columns below are the
                // span's own doing, not what drifted in from B's earlier spans.
                // B's end-to-end drift is the free run after the tap.
                for (n, bytes, buf) in &inputs {
                    s.b.load(n, *bytes, buf)?;
                }
                let (aa, ab) = (access(&ma, pname, span.a.clone()), access(&mb, pname, span.b.clone()));
                let touched: BTreeSet<String> = aa.state_writes.union(&ab.state_writes).cloned().collect();
                // State is compared on the span's write-set — the bytes A's run
                // changed — not on the whole allocation (the rest is other
                // layers' history, and it moves between now and any replay).
                // B runs the span from A's pre-image of that write-set, so what
                // B writes there is this span's doing, not its own history's.
                // Per-span state work (whole-state reads) only on the snapshotted
                // run; every other run B starts from a copy of A's state, which
                // gives the same pre-image per span (layers' write-sets are
                // disjoint) at one copy per run instead of four reads per span.
                let whole = |c: &S, st: &str| -> Result<Vec<u8>> { c.read_state(st, 0..c.state_bytes(st)?) };
                let mut a_pre = BTreeMap::new();
                if keep {
                    for st in &touched {
                        a_pre.insert(st.clone(), whole(&s.a, st)?);
                    }
                }
                s.a.run(pname, e, span.a.clone())?;
                let mut pre_states = Runs::new();
                let mut post_states = Runs::new();
                if keep {
                    for st in &touched {
                        let post = whole(&s.a, st)?;
                        let runs = diff_runs(&a_pre[st], &post);
                        for (off, bytes) in &runs {
                            s.b.write_state(st, *off, bytes)?;
                        }
                        post_states.insert(
                            st.clone(),
                            runs.iter().map(|(off, b)| (*off, post[*off..*off + b.len()].to_vec())).collect(),
                        );
                        pre_states.insert(st.clone(), runs);
                    }
                }
                let mut b_pre = BTreeMap::new();
                if keep {
                    for st in &touched {
                        b_pre.insert(st.clone(), whole(&s.b, st)?);
                    }
                }
                s.b.run(pname, e, span.b.clone())?;
                let (bufs, one_sided) = compare_written(&s.a, &s.b, &aa, &ab, e)?;
                let mut states = BTreeMap::new();
                for st in touched.iter().filter(|_| keep) {
                    let b_post = whole(&s.b, st)?;
                    let runs = &pre_states[st];
                    let set: usize = runs.iter().map(|(_, b)| b.len()).sum();
                    let n_diff: usize = post_states[st]
                        .iter()
                        .map(|(off, ap)| (0..ap.len()).filter(|i| ap[*i] != b_post[off + i]).count())
                        .sum();
                    // bytes B changed that lie outside A's write-set
                    let outside: usize = diff_runs(&b_pre[st], &b_post)
                        .iter()
                        .map(|(boff, bb)| {
                            (0..bb.len())
                                .filter(|i| {
                                    !runs.iter().any(|(aoff, ab)| (*aoff..aoff + ab.len()).contains(&(boff + i)))
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
                if keep {
                    let mut ref_out = BTreeMap::new();
                    for n in aa.writes.intersection(&ab.writes) {
                        ref_out.insert(n.clone(), s.a.read(n, live_bytes(&ma, n, e))?);
                    }
                    snap_state_bytes += pre_states.values().flatten().map(|(_, b)| b.len()).sum::<usize>();
                    snaps.push(Snapshot {
                        program: pname.into(),
                        span: span.clone(),
                        vars: e.clone(),
                        inputs,
                        ref_out,
                        pre_states,
                        post_states,
                        image,
                    });
                }
            }
            s.a.run(pname, e, ia..s.a.calls(pname)?)?;
            s.b.run(pname, e, ib..s.b.calls(pname)?)?;
            Ok(())
        };
    let shared_states: Vec<String> = ma.states.keys().filter(|n| mb.states.contains_key(*n)).cloned().collect();
    // B starts every program run from A's state: with per-layer write-sets
    // this is the per-span pre-image for every span of the run. One
    // device-side image per state, refreshed each run.
    let mut sync_img: BTreeMap<String, S::Buf> = BTreeMap::new();
    for n in &shared_states {
        sync_img.insert(n.clone(), s.a.alloc(s.a.state_bytes(n)?)?);
    }
    let sync_b = |s: &mut Sides<S>, img: &mut BTreeMap<String, S::Buf>| -> Result<()> {
        for (name, buf) in img.iter_mut() {
            s.a.save_state(name, buf)?;
            s.b.load_state(name, buf)?;
        }
        Ok(())
    };
    // `logits*` buffers a program writes: the end-to-end oracle.
    let logits_of = |prog: &str| -> Vec<String> {
        access(&ma, prog, 0..ma.programs[prog].calls.len())
            .writes
            .into_iter()
            .filter(|n| n.starts_with("logits") && mb.buffers.contains_key(n))
            .collect()
    };
    // (run label, buffer, vars, bytes)
    let read_logits = |c: &S, prog: &str, e: &Vars, label: &str| -> Result<Vec<(String, String, Vars, Vec<u8>)>> {
        logits_of(prog)
            .into_iter()
            .map(|n| Ok((label.to_string(), n.clone(), e.clone(), c.read(&n, live_bytes(&ma, &n, e))?)))
            .collect()
    };
    let pre = &wl.prefill;
    let chunk = wl.chunk;
    let mut a_logits = Vec::new();
    let mut i = 0;
    let chunk_name = chunk_f.as_ref().map_or("", |f| f.name.as_str());
    while i < pre.len() {
        let c = (pre.len() - i).min(chunk);
        let e = s.a.stage(&pre[i..i + c])?;
        s.b.stage(&pre[i..i + c])?;
        sync_b(s, &mut sync_img)?;
        lockstep(s, chunk_name, &e, &format!("chunk {n_chunks} "), n_chunks == 0, 0)?;
        a_logits.extend(read_logits(&s.a, chunk_name, &e, &format!("prefill chunk {n_chunks}"))?);
        s.a.advance(c as u64);
        s.b.advance(c as u64);
        i += c;
        n_chunks += 1;
    }
    let n_steps = wl.decode.len();
    // Every step stages the drawn token in a forward's rows at the cursor
    // and advances one position: a wide forward's extra rows are
    // overwritten by the next step, on both sides alike.
    let prog_of = |k: usize| &step_fs[k % step_fs.len()];
    let stage_step = |c: &mut S, f: &Forward, tok: i64| c.stage(&vec![tok; rows_of(f) as usize]);
    // A's state after prefill: the image every decode-step-0 replay starts from.
    let mut s0: BTreeMap<String, S::Buf> = BTreeMap::new();
    for n in &shared_states {
        let mut buf = s.a.alloc(s.a.state_bytes(n)?)?;
        s.a.save_state(n, &mut buf)?;
        s0.insert(n.clone(), buf);
    }
    for (k, &tok) in wl.decode.iter().enumerate() {
        let f = prog_of(k);
        let e = vars_of(f);
        stage_step(&mut s.a, f, tok)?;
        stage_step(&mut s.b, f, tok)?;
        sync_b(s, &mut sync_img)?;
        lockstep(s, &f.name, &e, &format!("step {k} "), k < step_fs.len(), 1)?;
        a_logits.extend(read_logits(&s.a, &f.name, &e, &format!("step {k}"))?);
        s.a.advance(1);
        s.b.advance(1);
    }
    drop(sync_img);
    // B free-runs the same workload from zero state, nothing injected: what
    // a caller would get. A's state is untouched by the lockstep (only B
    // received writes), so A is already the end-to-end reference.
    let t_free = Instant::now();
    s.b.zero_states()?;
    s.b.reset();
    let mut b_logits = Vec::new();
    let mut i = 0;
    let mut nc = 0;
    while i < pre.len() {
        let c = (pre.len() - i).min(chunk);
        let e = s.b.stage(&pre[i..i + c])?;
        s.b.run(chunk_name, &e, 0..s.b.calls(chunk_name)?)?;
        b_logits.extend(read_logits(&s.b, chunk_name, &e, &format!("prefill chunk {nc}"))?);
        s.b.advance(c as u64);
        i += c;
        nc += 1;
    }
    for (k, &tok) in wl.decode.iter().enumerate() {
        let f = prog_of(k);
        let e = vars_of(f);
        stage_step(&mut s.b, f, tok)?;
        s.b.run(&f.name, &e, 0..s.b.calls(&f.name)?)?;
        b_logits.extend(read_logits(&s.b, &f.name, &e, &format!("step {k}"))?);
        s.b.advance(1);
    }
    let free_t = elapsed(&t_free);
    let snap_bytes: usize = snaps
        .iter()
        .map(|sn| {
            sn.inputs.iter().map(|(_, b, _)| *b).sum::<usize>() + sn.ref_out.values().map(|b| b.len()).sum::<usize>()
        })
        .sum();
    let tap = Tap {
        seed: format!("{:#x}", o.seed),
        prefill: wl.prefill.len(),
        how: wl.how.to_string(),
        chunk: wl.chunk as u64,
        decode: wl.decode.len(),
        vocab: wl.vocab as usize,
        spans: snaps.len(),
        snapshot_bytes: snap_bytes,
        state_pre_image_bytes: snap_state_bytes,
        load_s: s.load_s,
        free_run_ms: free_t * 1e3,
        elapsed_s: elapsed(&t_tap),
    };
    out(&tap.lines());
    sum.tap = Some(tap);

    // ---- 2a. the tap's comparisons, counted; the differing ones named
    let undriven: Vec<String> = spans.keys().filter(|p| !driven.contains(&p.as_str())).cloned().collect();
    let mut all_local: Vec<Finding> = Vec::new();
    let (mut compared, mut n_bit, mut n_val) = (0usize, 0usize, 0usize);
    for (p, bufs) in &local_res {
        for (b, res) in bufs {
            for (l, c) in res {
                compared += 1;
                if c.identical() {
                    n_bit += 1;
                    continue;
                }
                n_val += c.value_identical() as usize;
                all_local.push(Finding::of(p, l, b, c));
            }
        }
    }
    for (p, sts) in &local_states {
        for (st, res) in sts {
            for (l, c) in res {
                compared += 1;
                if c.n_diff == 0 && c.outside == 0 {
                    n_bit += 1;
                    continue;
                }
                let mut t = format!("{}/{} bytes of the write-set differ", c.n_diff, c.set);
                if c.outside > 0 {
                    t += &format!(" · B wrote {} outside A's write-set", kb(c.outside));
                }
                all_local.push(Finding::text(p, l, &format!("state {st}"), t));
            }
        }
    }
    let local_bit = n_bit == compared;
    let local_identical = n_bit + n_val == compared;
    let mut outputs = Vec::new();
    let mut e2e_differs: Vec<String> = Vec::new();
    for (name, b) in &ma.buffers {
        if b.kind == BufferKind::Output && mb.buffers.contains_key(name) {
            // The last step's vars: what its outputs are live over.
            let e_last = vars_of(prog_of(n_steps.saturating_sub(1)));
            let bytes = live_bytes(&ma, name, &e_last);
            let c = compare(b.dtype, &s.a.read(name, bytes)?, &s.b.read(name, bytes)?);
            if !c.value_identical() {
                e2e_differs.push(name.clone());
            }
            outputs.push(Finding::of("", "end-to-end", name, &c));
        }
    }
    let e2e_outputs_identical = e2e_differs.is_empty();
    let mut e2e_states = Vec::new();
    for name in &shared_states {
        let a = s.a.read_state(name, 0..s.a.state_bytes(name)?)?;
        let b = s.b.read_state(name, 0..s.b.state_bytes(name)?)?;
        // KERN_TEST_DUMP=<dir>: both sides' final image of every state, for
        // locating a whole-state difference the spans do not explain.
        if let Ok(dir) = std::env::var("KERN_TEST_DUMP") {
            std::fs::write(format!("{dir}/{name}-a.bin"), &a)?;
            std::fs::write(format!("{dir}/{name}-b.bin"), &b)?;
        }
        let d = a.iter().zip(&b).filter(|(p, q)| p != q).count() + a.len().abs_diff(b.len());
        e2e_states.push(StateE2e { name: name.clone(), bytes: a.len(), differ: d });
    }
    let (findings, omitted) = cap(all_local.clone(), Finding::severity);
    let local = Local {
        compared,
        bit_identical: n_bit,
        value_identical: n_val,
        findings,
        omitted,
        outputs,
        states: e2e_states,
        one_sided: one_sided_all.iter().cloned().collect(),
        undriven: undriven.clone(),
    };
    out(&local.lines());
    sum.local = Some(local);

    // ---- 2b. logits: the end-to-end oracle
    let t_log = Instant::now();
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
    let have_logits = !logit_rows.is_empty();
    let logits_bit = have_logits && logit_rows.iter().all(|r| r.cmp.identical());
    let logits_max_ulp = logit_rows.iter().map(|r| r.scale_ulps).fold(0.0, f64::max);
    let n_flips = logit_rows.iter().filter(|r| r.flip()).count();
    let n_near = logit_rows.iter().filter(|r| r.near_tie()).count();
    let wide_flip = logit_rows.iter().find(|r| r.flip() && !r.near_tie());
    let logits_within = have_logits && logits_max_ulp <= o.logit_ulp as f64 && wide_flip.is_none();
    let worst =
        logit_rows.iter().max_by(|x, y| x.scale_ulps.total_cmp(&y.scale_ulps).then(x.cmp.n_diff.cmp(&y.cmp.n_diff)));
    let worst_kl = logit_rows.iter().max_by(|x, y| x.kl.total_cmp(&y.kl));
    let logits = Logits {
        rows: logit_rows.len(),
        runs: a_logits.len(),
        differ: logit_rows.iter().filter(|r| !r.cmp.identical()).count(),
        flips: n_flips,
        near_ties: n_near,
        max_ulp: logits_max_ulp,
        limit_ulp: o.logit_ulp,
        max_abs: worst.map_or(0.0, |w| w.max_abs),
        scale: worst.map_or(0.0, |w| w.scale),
        worst_at: worst.map_or(String::new(), |w| w.label.clone()),
        kl_max: worst_kl.map_or(0.0, |w| w.kl),
        kl_at: worst_kl.map_or(String::new(), |w| w.label.clone()),
        flipped: logit_rows
            .iter()
            .filter(|r| r.flip())
            .map(|r| Flip {
                row: r.label.clone(),
                argmax_a: r.argmax_a,
                argmax_b: r.argmax_b,
                margin_a: r.margin_a,
                delta: r.max_abs,
                near_tie: r.near_tie(),
            })
            .collect(),
        elapsed_s: elapsed(&t_log),
    };
    out(&logits.lines());
    sum.logits = Some(logits);
    let logit_detail: Vec<Value> = logit_rows
        .iter()
        .filter(|r| !r.cmp.identical())
        .map(|r| json!({"row": r.label, "cmp": r.cmp, "max_abs": r.max_abs, "scale_ulps": r.scale_ulps, "scale": r.scale, "argmax_a": r.argmax_a, "argmax_b": r.argmax_b, "margin_a": r.margin_a, "kl": r.kl, "near_tie": r.near_tie()}))
        .collect();

    // Replays start from the image of the state before the snapshotted run
    // — zeros for prefill chunk 0, A's post-prefill state for decode step 0
    // — on both sides. A span reads its own layer's slice, which the run's
    // earlier spans do not touch, so that image is every span's pre-state;
    // the per-span pre-image then only undoes the previous replay of the
    // same span. Re-imaged whenever the replay sequence switches runs.
    let image = |s: &mut Sides<S>, img: usize| -> Result<()> {
        for c in [&mut s.a, &mut s.b] {
            if img == 0 {
                c.zero_states()?;
            } else {
                for (n, b) in &s0 {
                    c.load_state(n, b)?;
                }
            }
        }
        Ok(())
    };

    // ---- 3. noise floor: A's span re-run from its own snapshot
    let mut noise_res: BTreeMap<String, BTreeMap<String, Vec<(String, Cmp)>>> = BTreeMap::new();
    let mut noise_clean = true;
    let mut noisy_states: BTreeSet<(String, String)> = BTreeSet::new();
    let mut all_noise: Vec<Finding> = Vec::new();
    if o.noise {
        let t_n = Instant::now();
        let mut state_noise = Vec::new();
        let (mut compared, mut clean) = (0usize, 0usize);
        let mut cur = None;
        for sn in &snaps {
            if cur != Some(sn.image) {
                image(s, sn.image)?;
                cur = Some(sn.image);
            }
            let (out_a, st) = replay(&mut s.a, sn, false, None)?;
            for (n, c) in cmp_out(&ma, sn, &out_a) {
                compared += 1;
                if c.identical() {
                    clean += 1;
                } else {
                    noise_clean = false;
                    all_noise.push(Finding::of(&sn.program, &sn.span.label(), &n, &c));
                }
                noise_res.entry(sn.program.clone()).or_default().entry(n).or_default().push((sn.span.label(), c));
            }
            for (name, d) in runs_differ(&st, &sn.post_states) {
                if d > 0 {
                    noise_clean = false;
                    noisy_states.insert((sn.program.clone(), name.clone()));
                    state_noise.push(format!("state {name}: {d} bytes at {} {}", sn.program, sn.span.label()));
                }
            }
        }
        let (findings, omitted) = cap(all_noise.clone(), Finding::severity);
        let noise = Noise { compared, clean, findings, omitted, states: state_noise, elapsed_s: elapsed(&t_n) };
        out(&noise.lines());
        sum.noise = Some(noise);
    }

    // ---- 4. fuzz the spans from their snapshots
    let mut fuzz_ok = true;
    let mut fuzz_identical = true; // value-identical under every distribution
    let mut fuzz_bit = true;
    let mut all_fuzz: Vec<FuzzFinding> = Vec::new();
    if o.fuzz > 0 {
        let t_f = Instant::now();
        let mut rng = Rng(o.seed);
        let mut ints_kept: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        let mut violations = Vec::new();
        let mut state_diffs = Vec::new();
        let (mut compared, mut n_bit, mut n_val) = (0usize, 0usize, 0usize);
        for round in 0..o.fuzz {
            let mode = MODES[round % MODES.len()];
            let mut cur = None;
            for sn in &snaps {
                if cur != Some(sn.image) {
                    image(s, sn.image)?;
                    cur = Some(sn.image);
                }
                let mut inputs = Vec::new();
                for (name, len, buf) in &sn.inputs {
                    let decl = &ma.buffers[name];
                    let tapped = s.a.bytes(buf, *len)?;
                    if !is_float(decl.dtype) {
                        // sequence layout, indices, page tables: structure,
                        // not values — a random one is a workload no caller
                        // produces, and the manifest says nothing about rows
                        // outside it
                        ints_kept.entry(sn.program.clone()).or_default().insert(name.clone());
                        inputs.push((name.clone(), tapped));
                        continue;
                    }
                    let base = values::to_f64(decl.dtype, &tapped);
                    let vals =
                        perturb(&mut rng, round % MODES.len(), &base, row_elems(&ma, name, &sn.vars), decl.dtype);
                    inputs.push((name.clone(), values::from_f64(decl.dtype, &vals)));
                }
                let at = format!("{} span {}", sn.program, sn.span.label());
                let (out_a, st_a) = replay(&mut s.a, sn, false, Some(&inputs))
                    .with_context(|| format!("A crashed under fuzz ({mode}) at {at}"))?;
                let (out_b, st_b) = replay(&mut s.b, sn, true, Some(&inputs)).with_context(|| {
                    format!("B crashed under fuzz ({mode}) at {at}; the CUDA context is unusable past this point")
                })?;
                // on the span's write-set, like the tap
                for (name, d) in runs_differ(&st_a, &st_b) {
                    if d > 0 {
                        state_diffs.push(format!("{mode} {at} state {name}: {d} bytes differ"));
                    }
                }
                for (name, b) in &out_b {
                    let c = compare(ma.buffers[name].dtype, &out_a[name], b);
                    compared += 1;
                    if c.identical() {
                        n_bit += 1;
                    } else {
                        n_val += c.value_identical() as usize;
                        all_fuzz.push(FuzzFinding {
                            mode: mode.into(),
                            at: Finding::of(&sn.program, &sn.span.label(), name, &c),
                        });
                    }
                    // Post-condition: produced values must lie in the
                    // buffer's declared domain (A is checked too — a
                    // violation there is the reference misbehaving).
                    if let Some(d) = &mb.buffers[name].domain {
                        let r = d.resolve(&mb, &sn.vars, &s.b.provision())?;
                        for (side, bytes) in [("A", &out_a[name]), ("B", b)] {
                            let v = values::to_f64(mb.buffers[name].dtype, bytes);
                            if let Some(i) = v.iter().position(|x| !r.contains(*x)) {
                                violations.push(format!("{mode} {at}: {side} {name}[{i}] = {} outside domain", v[i]));
                            }
                        }
                    }
                }
            }
        }
        fuzz_ok = violations.is_empty();
        fuzz_bit = n_bit == compared && state_diffs.is_empty();
        fuzz_identical = n_bit + n_val == compared && state_diffs.is_empty();
        let not_tapped = spans.keys().filter(|p| !snaps.iter().any(|sn| &sn.program == *p)).cloned().collect();
        let (findings, omitted) = cap(all_fuzz.clone(), |f| f.at.severity());
        let fuzz = Fuzz {
            rounds: o.fuzz,
            modes: MODES.iter().take(o.fuzz).map(|m| m.to_string()).collect(),
            compared,
            bit_identical: n_bit,
            value_identical: n_val,
            findings,
            omitted,
            violations,
            state_diffs,
            not_tapped,
            integers_kept: ints_kept,
            elapsed_s: elapsed(&t_f),
        };
        out(&fuzz.lines());
        sum.fuzz = Some(fuzz);
    }

    // ---- 5. perf: eager step attribution, graph step, sweep, roofline
    if o.perf {
        let t_p = Instant::now();
        let sweep_iters = o.iters.min(10);
        // (program, kernel) -> per side (bytes, ms, count)
        let mut per_kernel: BTreeMap<(String, String), [(usize, f32, usize); 2]> = BTreeMap::new();
        let mut state_traffic = false;
        // Time a whole program on both sides at `e`; returns (step, Σ spans)
        // per side and feeds the roofline accumulator.
        let mut step = |s: &mut Sides<S>, pname: &str, e: &Vars, iters: usize, roof: bool| -> Result<[(f32, f32); 2]> {
            let na = s.a.calls(pname)?;
            let nb = s.b.calls(pname)?;
            let ta = s.a.time(pname, e, 0..na, iters)?;
            let tb = s.b.time(pname, e, 0..nb, iters)?;
            let mut out = [(ta.iter().sum::<f32>(), 0f32), (tb.iter().sum::<f32>(), 0f32)];
            for sg in spans_of(pname) {
                for (si, m, r, t) in [(0usize, &ma, sg.a, &ta), (1, &mb, sg.b, &tb)] {
                    for i in r {
                        out[si].1 += t[i];
                        if !roof {
                            continue;
                        }
                        let acc = access(m, pname, i..i + 1);
                        let bytes: usize = acc.reads.iter().map(|n| live_bytes(m, n, e)).sum::<usize>()
                            + acc.writes.iter().map(|n| live_bytes(m, n, e)).sum::<usize>();
                        state_traffic |= !(acc.state_reads.is_empty() && acc.state_writes.is_empty());
                        let ent = per_kernel.entry((pname.into(), m.programs[pname].calls[i].op.clone())).or_default();
                        ent[si].0 += bytes;
                        ent[si].1 += t[i];
                        ent[si].2 += 1;
                    }
                }
            }
            Ok(out)
        };
        // derived = A's step with A's spans swapped for B's spans (both timed
        // eager); the gap to the measurement is launch-gap / L2 interaction.
        let derived = |a_step: f32, st: [(f32, f32); 2]| a_step - st[0].1 + st[1].1;
        let n_spans = |p: &str| spans_of(p).len();
        let mut steps = Vec::new();
        // each fixed-rows forward at the position after the workload
        for f in &step_fs {
            let p = f.name.as_str();
            if n_spans(p) == 0 || undriven.iter().any(|u| u == p) {
                continue;
            }
            let e1 = vars_of(f);
            stage_step(&mut s.a, f, *wl.decode.last().unwrap())?;
            stage_step(&mut s.b, f, *wl.decode.last().unwrap())?;
            let st = step(s, p, &e1, o.iters, true)?;
            let mut graph = None;
            if o.graph_step {
                s.a.capture(p, &e1)?;
                s.b.capture(p, &e1)?;
                graph = Some([s.a.time_captured(p, &e1, 100)?, s.b.time_captured(p, &e1, 100)?]);
            }
            steps.push(StepPerf {
                program: p.into(),
                rows: rows_of(f),
                spans: n_spans(p),
                eager_ms: [st[0].0, st[1].0],
                derived_ms: derived(st[0].0, st),
                graph_ms: graph,
                span_ms: [st[0].1, st[1].1],
            });
        }
        // prefill: the tapped chunk length plus a sweep over the var range
        let mut sweep = Vec::new();
        if chunk_f.is_some() && n_spans(chunk_name) > 0 && !undriven.iter().any(|u| u == chunk_name) {
            // The manifest's widest chunk, within what the sequence was provisioned.
            let max = pa.rows.max.min(s.a.provision().tokens);
            let tap_len = pre.len().min(chunk) as u64;
            let mut points: BTreeSet<u64> = [tap_len].into();
            if o.sweep {
                points.extend([1u64, 16, 128, 512, 2048, 4096, max].into_iter().filter(|&t| t <= max));
            }
            let vocab = s.a.vocab();
            let mut rng = Rng(o.seed);
            for &t in &points {
                let tid: Vec<i64> = (0..t).map(|_| rng.below(vocab) as i64).collect();
                s.a.reset();
                s.b.reset();
                let e = s.a.stage(&tid)?;
                s.b.stage(&tid)?;
                let st = step(s, chunk_name, &e, if t == tap_len { o.iters } else { sweep_iters }, t == tap_len)?;
                if t == tap_len {
                    steps.push(StepPerf {
                        program: chunk_name.into(),
                        rows: t,
                        spans: n_spans(chunk_name),
                        eager_ms: [st[0].0, st[1].0],
                        derived_ms: derived(st[0].0, st),
                        graph_ms: None,
                        span_ms: [st[0].1, st[1].1],
                    });
                }
                sweep.push(SweepPt {
                    rows: t,
                    eager_ms: [st[0].0, st[1].0],
                    derived_ms: derived(st[0].0, st),
                    span_ms: [st[0].1, st[1].1],
                });
            }
        }
        let roofline = per_kernel
            .iter()
            .map(|((program, op), sides)| {
                let per = sides.iter().find(|s| s.2 > 0).map_or(0, |s| s.0 / s.2);
                let side = |(bytes, ms, n): (usize, f32, usize)| -> (f32, f64, f64) {
                    if n == 0 {
                        return (f32::NAN, f64::NAN, f64::NAN);
                    }
                    let gbs = bytes as f64 / 1e9 / (ms as f64 / 1e3);
                    (ms / n as f32 * 1e3, gbs, gbs / o.peak_bw * 100.0)
                };
                let (a, b) = (side(sides[0]), side(sides[1]));
                Roof {
                    program: program.clone(),
                    op: op.clone(),
                    calls: sides[0].2.max(sides[1].2),
                    bytes_per_call: per,
                    us_per_call: [a.0, b.0],
                    gbs: [a.1, b.1],
                    peak_pct: [a.2, b.2],
                }
            })
            .collect();
        let perf = Perf {
            iters: o.iters,
            steps,
            sweep_program: chunk_name.into(),
            sweep,
            roofline,
            peak_bw_gbs: o.peak_bw,
            state_traffic,
            elapsed_s: elapsed(&t_p),
        };
        out(&perf.lines());
        sum.perf = Some(perf);
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
    let (code, text) = if !fuzz_ok {
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
        (0, "bit-identical at every span, real and perturbed inputs".to_string())
    } else if local_identical && fuzz_identical {
        (0, "value-identical at every span (only signed zeros differ)".to_string())
    } else if logits_bit {
        (0, format!("spans differ, but the end-to-end logits are bit-identical on all {n_rows} rows"))
    } else if logits_within {
        (0, format!("logit evidence: end-to-end logits move ≤ {logits_max_ulp:.2} ulp at their scale (limit {}) on {n_rows} rows, argmax agrees{}", o.logit_ulp, if n_near > 0 { format!(" except {n_near} near-tie{}", if n_near == 1 { "" } else { "s" }) } else { String::new() }))
    } else if within_noise && fuzz_identical {
        (0, "differences at every span lie within A's own noise floor".to_string())
    } else if have_logits {
        (2, format!("spans differ; end-to-end logits move up to {logits_max_ulp:.2} ulp at their scale on {n_rows} rows (limit {}), {n_flips} argmax flip{} ({n_near} near-tie){}", o.logit_ulp, if n_flips == 1 { "" } else { "s" }, if !noise_clean { " — A itself is not deterministic at some spans" } else { "" }))
    } else {
        (2, "spans differ beyond bit/value identity and no driven program writes logits — no oracle".to_string())
    };
    sum.verdict = Verdict::new(code, text, elapsed(&t_start) + s.load_s);
    let detail = json!({
        "within_noise": within_noise, "end_to_end_outputs_differ": e2e_differs, "end_to_end_outputs_identical": e2e_outputs_identical,
        "local": all_local, "logit_rows": logit_detail, "noise": all_noise, "fuzz": all_fuzz,
    });
    Ok(Report { summary: sum, detail })
}
