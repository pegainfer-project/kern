//! Replaying B against a [`Recording`] of A: the workload with every run
//! started from A's state image and every span from A's inputs (so what
//! B writes is the span's own doing), B's free run for the end-to-end
//! logits, the kept spans under the perturbations A saw, B's timings, and
//! the verdict over all of it.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use anyhow::{Context, Result};
use serde_json::{json, Value};

use crate::compare::{compare, diff_runs, logit_row, Cmp, LogitRow, MODES};
use crate::diff::{access, live_bytes, row_elems};
use crate::harness::{at_rank, domain_violations, image, read_logits, replay_span, runs_differ, whole, Recording};
use crate::report::*;
use crate::{Options, Side, Vars};

/// How B's write to a state compares with A's, on this span: bytes of A's
/// write-set (`set`), how many of them B wrote differently (`n_diff`), and
/// how many bytes B changed outside A's write-set (`outside`).
#[derive(Clone, Default)]
struct StateCmp {
    set: usize,
    n_diff: usize,
    outside: usize,
}

/// Replay B against `rec` and judge it. `out` receives each section's
/// lines as it finishes.
pub fn replay<S: Side>(
    o: &Options,
    rec: Recording<S::Buf>,
    b: &mut S,
    out: &mut dyn FnMut(&[String]),
) -> Result<Report> {
    let t_start = Instant::now();
    let elapsed = |t: &Instant| t.elapsed().as_secs_f32();
    let ranks = rec.ranks;
    anyhow::ensure!(b.ranks() == ranks, "A recorded {ranks} ranks; B runs as {}", b.ranks());
    let (ma, mb) = (&rec.ma, &rec.mb);
    let mut sum = Summary { a: o.a.clone(), b: o.b.clone(), diff: rec.diff.clone(), ..Default::default() };
    let undriven = rec.undriven();
    let shared = rec.shared_states();

    // ---- 2. the workload on B, run by run from A's image, span by span from A's inputs
    let t_tap = Instant::now();
    // program -> buffer -> [(span label, cmp)]
    let mut local_res: BTreeMap<String, BTreeMap<String, Vec<(String, Cmp)>>> = BTreeMap::new();
    let mut local_states: BTreeMap<String, BTreeMap<String, Vec<(String, StateCmp)>>> = BTreeMap::new();
    b.reset();
    for run in &rec.runs {
        image(b, run)?;
        b.stage(&run.tokens)?;
        let (p, e) = (run.program.as_str(), &run.vars);
        let mut ib = 0;
        for sr in &run.spans {
            b.run(p, e, ib..sr.span.b.start)?;
            ib = sr.span.b.end;
            for q in 0..ranks {
                for (n, len, buf) in &sr.inputs[q] {
                    b.load(q, n, *len, buf)?;
                }
            }
            let kept = sr.pre.iter().any(|r| !r.is_empty());
            let mut b_pre: Vec<BTreeMap<String, Vec<u8>>> = Vec::new();
            if kept {
                for q in 0..ranks {
                    b_pre.push(sr.pre[q].keys().map(|st| Ok((st.clone(), whole(b, q, st)?))).collect::<Result<_>>()?);
                }
            }
            b.run(p, e, sr.span.b.clone())?;
            for q in 0..ranks {
                let label = at_rank(ranks, q, &format!("{} {}", run.label, sr.span.label()));
                for (n, a_bytes) in &sr.ref_out[q] {
                    let c = compare(ma.buffers[n].dtype, a_bytes, &b.read(q, n, a_bytes.len())?);
                    local_res.entry(p.into()).or_default().entry(n.clone()).or_default().push((label.clone(), c));
                }
                for (st, runs) in &sr.pre[q] {
                    let b_post = whole(b, q, st)?;
                    let set: usize = runs.iter().map(|(_, b)| b.len()).sum();
                    let n_diff: usize = sr.post[q][st]
                        .iter()
                        .map(|(off, ap)| (0..ap.len()).filter(|i| ap[*i] != b_post[off + i]).count())
                        .sum();
                    // bytes B changed that lie outside A's write-set
                    let outside: usize = diff_runs(&b_pre[q][st], &b_post)
                        .iter()
                        .map(|(boff, bb)| {
                            (0..bb.len())
                                .filter(|i| {
                                    !runs.iter().any(|(aoff, ab)| (*aoff..aoff + ab.len()).contains(&(boff + i)))
                                })
                                .count()
                        })
                        .sum();
                    local_states
                        .entry(p.into())
                        .or_default()
                        .entry(st.clone())
                        .or_default()
                        .push((label.clone(), StateCmp { set, n_diff, outside }));
                }
            }
        }
        b.run(p, e, ib..b.calls(p)?)?;
        b.advance(run.advance);
    }
    // B free-runs the same workload from zero state, nothing injected: what
    // a caller would get.
    let t_free = Instant::now();
    b.zero_states()?;
    b.reset();
    let mut b_logits = Vec::new();
    for run in &rec.runs {
        let e = b.stage(&run.tokens)?;
        b.run(&run.program, &e, 0..b.calls(&run.program)?)?;
        b_logits.extend(read_logits(b, ma, mb, &run.program, &e, "")?);
        b.advance(run.advance);
    }
    let free_t = elapsed(&t_free);
    let tap = Tap {
        seed: format!("{:#x}", o.seed),
        prefill: rec.wl.prefill.len(),
        how: rec.wl.how.to_string(),
        chunk: rec.wl.chunk as u64,
        decode: rec.wl.decode.len(),
        vocab: rec.wl.vocab as usize,
        ranks,
        runs: rec.runs.len(),
        spans: rec.kept.len(),
        snapshot_bytes: rec.snapshot_bytes(),
        state_pre_image_bytes: rec.pre_image_bytes(),
        load_s: rec.load_s,
        record_s: rec.record_s,
        free_run_ms: free_t * 1e3,
        elapsed_s: elapsed(&t_tap),
    };
    out(&tap.lines());
    sum.tap = Some(tap);

    // ---- 2a. the comparisons, counted; the differing ones named
    let mut all_local: Vec<Finding> = Vec::new();
    let (mut compared, mut n_bit, mut n_val) = (0usize, 0usize, 0usize);
    for (p, bufs) in &local_res {
        for (bn, res) in bufs {
            for (l, c) in res {
                compared += 1;
                if c.identical() {
                    n_bit += 1;
                    continue;
                }
                n_val += c.value_identical() as usize;
                all_local.push(Finding::of(p, l, bn, c));
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
    for (name, per_rank) in &rec.outputs {
        for (q, a_bytes) in per_rank.iter().enumerate() {
            let c = compare(ma.buffers[name].dtype, a_bytes, &b.read(q, name, a_bytes.len())?);
            if !c.value_identical() {
                e2e_differs.push(at_rank(ranks, q, name));
            }
            outputs.push(Finding::of("", "end-to-end", &at_rank(ranks, q, name), &c));
        }
    }
    let e2e_outputs_identical = e2e_differs.is_empty();
    let mut e2e_states = Vec::new();
    for (q, img) in rec.states.iter().enumerate() {
        for name in &shared {
            let len = b.state_bytes(name)?;
            let a_bytes = b.bytes(q, &img[name], len)?;
            let b_bytes = b.read_state(q, name, 0..len)?;
            // KERN_TEST_DUMP=<dir>: both sides' final image of every state, for
            // locating a whole-state difference the spans do not explain.
            if let Ok(dir) = std::env::var("KERN_TEST_DUMP") {
                std::fs::write(format!("{dir}/{name}-r{q}-a.bin"), &a_bytes)?;
                std::fs::write(format!("{dir}/{name}-r{q}-b.bin"), &b_bytes)?;
            }
            let d =
                a_bytes.iter().zip(&b_bytes).filter(|(p, q)| p != q).count() + a_bytes.len().abs_diff(b_bytes.len());
            e2e_states.push(StateE2e { name: at_rank(ranks, q, name), bytes: a_bytes.len(), differ: d });
        }
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
        one_sided: rec.one_sided.iter().cloned().collect(),
        undriven: undriven.clone(),
    };
    out(&local.lines());
    sum.local = Some(local);

    // ---- 2b. logits: the end-to-end oracle
    let t_log = Instant::now();
    let mut logit_rows: Vec<LogitRow> = Vec::new();
    let n_runs_with = |label: &str| rec.logits.iter().filter(|x| x.label == label).count();
    for (la, lb) in rec.logits.iter().zip(&b_logits) {
        let dt = ma.buffers[&la.buffer].dtype;
        let row = row_elems(ma, &la.buffer, &la.vars) * dt.bytes() as usize;
        for q in 0..ranks {
            let (a, bb) = (&la.bytes[q], &lb.bytes[q]);
            let rows = a.len().checked_div(row).unwrap_or(0);
            for r in 0..rows.max(1) {
                let (lo, hi) = if rows > 1 { (r * row, (r + 1) * row) } else { (0, a.len()) };
                let lbl = if n_runs_with(&la.label) > 1 || rows > 1 {
                    format!("{} {}{}", la.label, la.buffer, if rows > 1 { format!("[{r}]") } else { String::new() })
                } else {
                    la.label.clone()
                };
                logit_rows.push(logit_row(at_rank(ranks, q, &lbl), dt, &a[lo..hi], &bb[lo..hi]));
            }
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
        runs: rec.logits.len(),
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

    // ---- 3. noise floor: measured on A while it was loaded
    let (noise_clean, noisy_states, noise_res, all_noise) = match &rec.noise {
        Some(n) => {
            out(&n.report.lines());
            sum.noise = Some(n.report.clone());
            (n.report.compared == n.report.clean, n.noisy_states.clone(), n.res.clone(), n.findings.clone())
        }
        None => (true, BTreeSet::new(), BTreeMap::new(), Vec::new()),
    };

    // ---- 4. fuzz: the kept spans on B, with the inputs A saw
    let mut fuzz_ok = true;
    let mut fuzz_identical = true; // value-identical under every distribution
    let mut fuzz_bit = true;
    let mut all_fuzz: Vec<FuzzFinding> = Vec::new();
    if let Some(fz) = &rec.fuzz {
        let t_f = Instant::now();
        let mut violations = fz.violations.clone();
        let mut state_diffs = Vec::new();
        let (mut compared, mut n_bit, mut n_val) = (0usize, 0usize, 0usize);
        for (mode, cases) in &fz.rounds {
            let mut cur = None;
            for case in cases {
                let (r, s) = case.at;
                let run = &rec.runs[r];
                if cur != Some(r) {
                    image(b, run)?;
                    cur = Some(r);
                }
                let sr = &run.spans[s];
                let at = format!("{} span {}", run.program, sr.span.label());
                let (out_b, st_b) = replay_span(b, sr, run, true, Some(&case.inputs)).with_context(|| {
                    format!("B crashed under fuzz ({mode}) at {at}; the CUDA context is unusable past this point")
                })?;
                for q in 0..ranks {
                    let label = at_rank(ranks, q, &format!("{} {}", run.label, sr.span.label()));
                    // on the span's write-set, like the tap
                    for (name, d) in runs_differ(&case.states[q], &st_b[q]) {
                        if d > 0 {
                            state_diffs
                                .push(format!("{mode} {} state {name}: {d} bytes differ", at_rank(ranks, q, &at)));
                        }
                    }
                    for (name, bb) in &out_b[q] {
                        let c = compare(ma.buffers[name].dtype, &case.out[q][name], bb);
                        compared += 1;
                        if c.identical() {
                            n_bit += 1;
                        } else {
                            n_val += c.value_identical() as usize;
                            all_fuzz.push(FuzzFinding {
                                mode: mode.to_string(),
                                at: Finding::of(&run.program, &label, name, &c),
                            });
                        }
                    }
                }
                // Post-condition: produced values must lie in the buffer's
                // declared domain.
                violations.extend(
                    domain_violations(b, mb, &run.vars, "B", &out_b)?.into_iter().map(|v| format!("{mode} {at}: {v}")),
                );
            }
        }
        fuzz_ok = violations.is_empty();
        fuzz_bit = n_bit == compared && state_diffs.is_empty();
        fuzz_identical = n_bit + n_val == compared && state_diffs.is_empty();
        let not_tapped = rec.not_tapped();
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
            integers_kept: fz.ints_kept.clone(),
            elapsed_s: elapsed(&t_f),
        };
        out(&fuzz.lines());
        sum.fuzz = Some(fuzz);
    }

    // ---- 5. perf: B's side of every timing, then attribution
    if let Some(pf) = &rec.perf {
        let t_p = Instant::now();
        // (program, kernel) -> per side (bytes, ms, count)
        let mut per_kernel: BTreeMap<(String, String), [(usize, f32, usize); 2]> = BTreeMap::new();
        let mut state_traffic = false;
        // Σ over a side's spans of its per-call times; feeds the roofline
        // accumulator when `roof`.
        let mut attribute = |side: usize, pname: &str, times: &[f32], e: &Vars, roof: bool| -> (f32, f32) {
            let m = if side == 0 { ma } else { mb };
            let mut span_ms = 0f32;
            for sg in rec.spans_of(pname) {
                let r = if side == 0 { sg.a } else { sg.b };
                for i in r {
                    span_ms += times[i];
                    if !roof {
                        continue;
                    }
                    let acc = access(m, pname, i..i + 1);
                    let bytes: usize = acc.reads.iter().map(|n| live_bytes(m, n, e)).sum::<usize>()
                        + acc.writes.iter().map(|n| live_bytes(m, n, e)).sum::<usize>();
                    state_traffic |= !(acc.state_reads.is_empty() && acc.state_writes.is_empty());
                    let ent = per_kernel.entry((pname.into(), m.programs[pname].calls[i].op.clone())).or_default();
                    ent[side].0 += bytes;
                    ent[side].1 += times[i];
                    ent[side].2 += 1;
                }
            }
            (times.iter().sum(), span_ms)
        };
        // derived = A's step with A's spans swapped for B's spans (both timed
        // eager); the gap to the measurement is launch-gap / L2 interaction.
        let derived = |a: (f32, f32), bb: (f32, f32)| a.0 - a.1 + bb.1;
        let mut steps = Vec::new();
        for st in &pf.steps {
            let p = st.program.as_str();
            let e = b.stage(&vec![st.token; st.rows as usize])?;
            let tb = b.time(p, &e, 0..b.calls(p)?, o.iters)?;
            let graph_ms = match st.graph_ms {
                Some(ga) => {
                    b.capture(p, &e)?;
                    Some([ga, b.time_captured(p, &e, 100)?])
                }
                None => None,
            };
            let (a, bb) = (attribute(0, p, &st.times, &e, true), attribute(1, p, &tb, &e, true));
            steps.push(StepPerf {
                program: p.into(),
                rows: st.rows,
                spans: rec.spans_of(p).len(),
                eager_ms: [a.0, bb.0],
                derived_ms: derived(a, bb),
                graph_ms,
                span_ms: [a.1, bb.1],
            });
        }
        let chunk_name = rec.chunk.as_ref().map_or("", |f| f.name.as_str());
        let mut sweep = Vec::new();
        for sw in &pf.sweep {
            b.reset();
            let e = b.stage(&sw.tokens)?;
            let tb = b.time(chunk_name, &e, 0..b.calls(chunk_name)?, sw.iters)?;
            let (a, bb) =
                (attribute(0, chunk_name, &sw.times, &e, sw.tapped), attribute(1, chunk_name, &tb, &e, sw.tapped));
            if sw.tapped {
                steps.push(StepPerf {
                    program: chunk_name.into(),
                    rows: sw.rows,
                    spans: rec.spans_of(chunk_name).len(),
                    eager_ms: [a.0, bb.0],
                    derived_ms: derived(a, bb),
                    graph_ms: None,
                    span_ms: [a.1, bb.1],
                });
            }
            sweep.push(SweepPt {
                rows: sw.rows,
                eager_ms: [a.0, bb.0],
                derived_ms: derived(a, bb),
                span_ms: [a.1, bb.1],
            });
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
                let (a, bb) = (side(sides[0]), side(sides[1]));
                Roof {
                    program: program.clone(),
                    op: op.clone(),
                    calls: sides[0].2.max(sides[1].2),
                    bytes_per_call: per,
                    us_per_call: [a.0, bb.0],
                    gbs: [a.1, bb.1],
                    peak_pct: [a.2, bb.2],
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
            bufs.iter().all(|(bn, res)| {
                let worst_local = res
                    .iter()
                    .filter_map(|(_, c)| if c.value_identical() { None } else { c.max_ulp.or(Some(u64::MAX)) })
                    .max();
                let worst_noise = noise_res.get(p).and_then(|nb| nb.get(bn)).and_then(|nr| {
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
    sum.verdict = Verdict::new(code, text, elapsed(&t_start) + rec.record_s + rec.load_s);
    let detail = json!({
        "within_noise": within_noise, "end_to_end_outputs_differ": e2e_differs, "end_to_end_outputs_identical": e2e_outputs_identical,
        "local": all_local, "logit_rows": logit_detail, "noise": all_noise, "fuzz": all_fuzz,
    });
    Ok(Report { summary: sum, detail })
}
