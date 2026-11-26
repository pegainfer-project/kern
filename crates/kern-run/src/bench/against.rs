//! `kern bench --against`: B against A in one process.
//!
//! Both manifests are loaded and primed and stay resident together, B
//! reading A's weights in place. Each scenario is staged on both and
//! replayed in rounds of three blocks, A B A then B A B; a round's middle
//! block against the geometric mean of the two around it sees the same
//! clocks, temperature and power cap, and cancels drift linear over the
//! round ([`paired`]). A round opens with an unscored block of the side
//! it does not start with: the GPU idles while the tracer is drained
//! between rounds, and the replays right after run apart, a few tenths of
//! a percent faster where the power cap binds and slower where it does
//! not. So every scored block follows the other side's, none follows a
//! pause. A scenario stops once its interval is narrow enough, or at
//! `--max-rounds`.
//!
//! What rounds cannot cancel is where a side's buffers landed: the same
//! manifest loaded twice runs a few tenths of a percent apart, the same
//! way on every load in the same order. So the whole is repeated over
//! `--epochs`, both sides loaded afresh each time over the weights the
//! last epoch left resident, A first on even epochs and B first on odd
//! ones, which side loaded first a blocking factor ([`across`]); a
//! scenario's change is Student-t over its epochs. The workload's
//! cost combines an epoch's scenarios by their share of A's, then is
//! Student-t over the epochs: a placement shifts every scenario of its
//! epoch the same way, so scenarios are not independent draws. Ops whose
//! implementation or call count differ between the two are the changed
//! ops; their exclusive time is compared the same way, so a change is
//! read on what it touched, not only on the whole graph's noise.
//!
//! `--out` is rewritten after every scenario, so a run cut short leaves
//! what it measured. Beside it, `<stem>.a.json` and `<stem>.b.json` are
//! ordinary reports with every sample of each side, written per epoch.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{ensure, Context, Result};
use kern_manifest::types::{Call, Manifest};
use kern_manifest::Verified;
use kern_runtime::trace::{self, Step};
use kern_runtime::{HostWeights, Resident};
use serde_json::{json, Value};

use super::paired::{across, combine, median, Paired, Verdict};
use super::workload::{Scenario, Workload};
use super::{
    output_fingerprints, preamble, record, replay_holds, samples, say, say_setup, say_timed, stage, stats, telemetry,
    Bench, BenchOpts, Calls, Rank, Sample, Staged, Timed, WARMUP,
};
use crate::Inputs;

/// Rounds before a scenario may stop on its interval.
const MIN_ROUNDS: usize = 4;

/// One manifest, loaded and primed, and what it is called in the report.
struct Side {
    name: &'static str,
    bench: Bench,
    manifest: Verified,
    header: Value,
}

/// Ops whose implementation differs, or that the programs call a different
/// number of times: what B changed.
fn changed_ops(a: &Manifest, b: &Manifest) -> BTreeSet<String> {
    fn counts(m: &Manifest) -> BTreeMap<&str, usize> {
        m.programs.values().flat_map(|p| &p.calls).fold(BTreeMap::new(), |mut n, c| {
            *n.entry(c.op.as_str()).or_default() += 1;
            n
        })
    }
    let (na, nb) = (counts(a), counts(b));
    let imp = |m: &Manifest, op: &str| m.ops.get(op).map(|o| serde_json::to_value(o).ok());
    na.keys()
        .chain(nb.keys())
        .filter(|op| na.get(*op) != nb.get(*op) || imp(a, op) != imp(b, op))
        .map(|op| op.to_string())
        .collect()
}

/// A sample's exclusive time over the calls of `ops`.
fn of_ops(s: &Sample, calls: &[Call], ops: &BTreeSet<String>) -> f64 {
    calls.iter().zip(&s.exclusive).filter(|(c, _)| ops.contains(&c.op)).map(|(_, us)| us).sum()
}

/// One side's blocks of one scenario: per rank, the blocks' samples.
#[derive(Default)]
struct Blocks {
    per_rank: Vec<Vec<Vec<Sample>>>,
}

impl Blocks {
    /// Appends a block; its index.
    fn push(&mut self, block: Vec<Vec<Sample>>) -> usize {
        if self.per_rank.is_empty() {
            self.per_rank = block.iter().map(|_| Vec::new()).collect();
        }
        self.per_rank.iter_mut().zip(block).for_each(|(r, b)| r.push(b));
        self.per_rank[0].len() - 1
    }

    /// Block `i`'s statistic: the slowest rank's median of `f`, what a step waits for.
    fn median(&self, i: usize, f: impl Fn(&Sample) -> f64) -> f64 {
        self.per_rank.iter().map(|r| median(&r[i].iter().map(&f).collect::<Vec<_>>())).fold(0., f64::max)
    }

    fn all(&self, rank: usize) -> impl Iterator<Item = &Sample> {
        self.per_rank[rank].iter().flatten()
    }
}

/// One round's three scored blocks in the order they ran, as (side, block
/// index): A B A on even rounds, B A B on odd ones.
type Round = [(usize, usize); 3];

/// Per round, A's and B's statistic: the middle block's against the
/// geometric mean of the two around it, which cancels a drift linear over
/// the round. `f` is given the block's side.
fn compared(rounds: &[Round], blocks: &[Blocks; 2], f: impl Fn(usize, &Sample) -> f64) -> Vec<(f64, f64)> {
    rounds
        .iter()
        .map(|r| {
            let [o0, mid, o2] = r.map(|(i, b)| blocks[i].median(b, |x| f(i, x)));
            let outer = (o0 * o2).sqrt();
            match r[0].0 {
                0 => (outer, mid),
                _ => (mid, outer),
            }
        })
        .collect()
}

/// `n` timed replays of every rank's staged scenario; their issue numbers, per rank.
fn visit(side: &mut Side, staged: &[Staged], n: usize, what: &str) -> Result<Vec<Vec<u64>>> {
    let mut both: Vec<(&mut Rank, &Staged)> = side.bench.ranks.iter_mut().zip(staged).collect();
    crate::each(&mut both, what, |(r, st)| {
        (0..n).map(|_| Ok(r.probe.sample(&r.rt, &st.pre, &st.graph)?)).collect::<Result<Vec<u64>>>()
    })
}

fn stage_all(side: &mut Side, s: &Scenario) -> Result<Vec<Staged>> {
    let (p, corpus) = (&side.bench.protocol, &side.bench.corpus);
    let staged = crate::each(&mut side.bench.ranks, &s.id, |r| stage(r, p, s, corpus).map(crate::Sent))?;
    let staged: Vec<Staged> = staged.into_iter().map(|x| x.0).collect();
    visit(side, &staged, WARMUP, &format!("{} warmup", s.id))?;
    Ok(staged)
}

/// Collect until the tracer has returned every issue of `seqs`. A forced
/// flush per block would end every pending record each time; the plain
/// collect leaves the tracer as a scenario of `kern bench` does.
fn gather<'a>(steps: &mut BTreeMap<u64, Step>, seqs: impl Iterator<Item = &'a u64> + Clone) -> Result<()> {
    for _ in 0..COLLECTS {
        steps.extend(trace::collect()?.steps.into_iter().map(|t| (t.seq, t)));
        if seqs.clone().all(|q| steps.contains_key(q)) {
            return Ok(());
        }
    }
    steps.extend(trace::finish()?.steps.into_iter().map(|t| (t.seq, t)));
    Ok(())
}

/// Collects a block waits for its records before the forced one.
const COLLECTS: usize = 8;

/// Everything one scenario measured, both sides.
struct Scored {
    id: String,
    weight: f64,
    a_us: f64,
    b_us: f64,
    graph: Option<Paired>,
    changed: Option<Paired>,
    same_tokens: bool,
}

fn line(p: Option<Paired>) -> String {
    match p {
        Some(p) => {
            let (mid, lo, hi) = p.percent();
            format!("{mid:+.3}% [{lo:+.3}, {hi:+.3}]")
        }
        None => "no interval".into(),
    }
}

fn paired_json(p: Option<Paired>) -> Value {
    match p {
        Some(p) => {
            let (mid, lo, hi) = p.percent();
            json!({"rounds":p.rounds,"ln_ratio":p.mean,"se":p.se,"half":p.half,"percent":mid,"low":lo,"high":hi,
                "verdict":p.verdict()})
        }
        None => Value::Null,
    }
}

struct Run<'a> {
    o: &'a BenchOpts,
    /// A then B; the side at `owner` holds the weights the other may read.
    sides: Vec<Side>,
    owner: usize,
    epoch: usize,
    changed: BTreeSet<String>,
}

impl Drop for Run<'_> {
    fn drop(&mut self) {
        if self.sides.len() == 2 {
            self.sides.remove(1 - self.owner);
        }
    }
}

impl Run<'_> {
    /// The borrower dropped first, the owner's weights kept for the next epoch.
    fn release(mut self) -> Vec<Resident> {
        let mut sides = std::mem::take(&mut self.sides);
        let owner = sides.remove(self.owner);
        drop(sides);
        owner.bench.ranks.into_iter().map(|r| r.rt.into_resident()).collect()
    }

    fn scenario(&mut self, s: &Scenario, records: &mut [Vec<Value>; 2]) -> Result<Scored> {
        let started = Instant::now();
        let (block, o) = (self.o.block, self.o);
        let staged = [stage_all(&mut self.sides[0], s)?, stage_all(&mut self.sides[1], s)?];
        let before: Vec<Value> = self.sides.iter().map(|x| telemetry(x.bench.ranks[0].gpu)).collect();
        let mut steps: BTreeMap<u64, Step> = BTreeMap::new();
        let programs: Vec<Vec<Call>> =
            self.sides.iter().map(|x| x.manifest.programs[&s.program].calls.clone()).collect();
        let mut blocks = [Blocks::default(), Blocks::default()];
        let mut rounds: Vec<Round> = Vec::new();
        let mut graph = None;
        for k in 0..o.max_rounds {
            let order = if k % 2 == 0 { [0, 1, 0] } else { [1, 0, 1] };
            visit(&mut self.sides[order[1]], &staged[order[1]], block, &s.id)?;
            let mut seqs = Vec::new();
            for i in order {
                seqs.push((i, visit(&mut self.sides[i], &staged[i], block, &s.id)?));
            }
            gather(&mut steps, seqs.iter().flat_map(|(_, q)| q.iter().flatten()))?;
            let round = seqs
                .iter()
                .map(|(i, q)| {
                    let per_rank =
                        q.iter().map(|q| samples(&mut steps, q, &programs[*i])).collect::<Result<Vec<_>>>()?;
                    Ok((*i, blocks[*i].push(per_rank)))
                })
                .collect::<Result<Vec<_>>>()?;
            rounds.push([round[0], round[1], round[2]]);
            graph = Paired::of(&compared(&rounds, &blocks, |_, x| x.graph));
            if k + 1 >= MIN_ROUNDS && graph.is_some_and(|p| p.half < o.precision) {
                break;
            }
        }
        let changed = Paired::of(&compared(&rounds, &blocks, |i, x| of_ops(x, &programs[i], &self.changed)));
        let mut tokens = Vec::new();
        for (i, side) in self.sides.iter_mut().enumerate() {
            let p = &side.bench.protocol;
            for (r, st) in side.bench.ranks.iter().zip(&staged[i]) {
                replay_holds(r, p, s, st)?;
            }
            tokens.push(
                side.bench
                    .ranks
                    .iter()
                    .map(|r| output_fingerprints(&r.rt, p, s.shape.groups))
                    .collect::<Result<Vec<_>>>()?,
            );
        }
        trace::finish()?;
        let elapsed_s = started.elapsed().as_secs_f64();
        let mut p50 = [0f64; 2];
        for (i, side) in self.sides.iter().enumerate() {
            let after = telemetry(side.bench.ranks[0].gpu);
            for (q, st) in staged[i].iter().enumerate() {
                let all: Vec<Sample> = blocks[i].all(q).cloned().collect();
                let graph_us: Vec<f64> = all.iter().map(|x| x.graph).collect();
                p50[i] = p50[i].max(stats(&graph_us).p50);
                let telemetry = (if q == 0 { before[i].clone() } else { Value::Null }, after.clone());
                let timed =
                    Timed { vars: st.vars.clone(), graph_us, calls: Calls::of(&all, programs[i].len()), telemetry };
                let mut rec = record(&side.manifest, s, q, timed, elapsed_s);
                rec["epoch"] = json!(self.epoch);
                records[i].push(rec);
            }
        }
        let scored = Scored {
            id: s.id.clone(),
            weight: s.shape.weight as f64,
            a_us: p50[0],
            b_us: p50[1],
            graph,
            changed: (!self.changed.is_empty()).then_some(changed).flatten(),
            same_tokens: tokens[0] == tokens[1],
        };
        let changed = match scored.changed {
            Some(c) => format!(" · changed ops {}", line(Some(c))),
            None => String::new(),
        };
        let tokens = if scored.same_tokens { "" } else { " · tokens differ" };
        say_timed(
            "scenario",
            format!(
                "{} · epoch {} · {:.3} → {:.3} ms · {} · {} rounds{changed}{tokens}",
                s.id,
                self.epoch,
                scored.a_us / 1e3,
                scored.b_us / 1e3,
                line(scored.graph),
                scored.graph.map_or(0, |p| p.rounds)
            ),
            started,
        );
        Ok(scored)
    }
}

fn beside(out: &Path, side: &str) -> PathBuf {
    let stem = out.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "bench".into());
    out.with_file_name(format!("{stem}.{side}.json"))
}

/// A scenario over its epochs: the change Student-t over them, each
/// side's time the median of its epochs' p50s.
fn over_epochs(epochs: &[&Scored]) -> Scored {
    let mid = |f: fn(&Scored) -> f64| median(&epochs.iter().map(|x| f(x)).collect::<Vec<_>>());
    let each = |f: fn(&Scored) -> Option<Paired>| epochs.iter().map(|x| f(x)).collect::<Option<Vec<_>>>();
    Scored {
        id: epochs[0].id.clone(),
        weight: epochs[0].weight,
        a_us: mid(|x| x.a_us),
        b_us: mid(|x| x.b_us),
        graph: each(|x| x.graph).and_then(|p| across(&p)),
        changed: each(|x| x.changed).and_then(|p| across(&p)),
        same_tokens: epochs.iter().all(|x| x.same_tokens),
    }
}

/// The epochs measured so far, each scenario over the epochs that reached it.
struct Summary {
    value: Value,
    verdict: Verdict,
    scenarios: Vec<Scored>,
    total: Option<Paired>,
    changed_total: Option<Paired>,
    cost: (f64, f64),
}

fn summary(o: &BenchOpts, headers: &[Value; 2], changed: &BTreeSet<String>, epochs: &[Vec<Scored>]) -> Summary {
    let ids: Vec<&str> =
        epochs.iter().max_by_key(|e| e.len()).map_or(Vec::new(), |e| e.iter().map(|x| x.id.as_str()).collect());
    let per: Vec<(Scored, Vec<&Scored>)> = ids
        .iter()
        .map(|id| {
            let got: Vec<&Scored> = epochs.iter().filter_map(|e| e.iter().find(|x| x.id == *id)).collect();
            (over_epochs(&got), got)
        })
        .collect();
    let cost = |pick: fn(&Scored) -> f64| per.iter().map(|(x, _)| x.weight * pick(x)).sum::<f64>();
    let share: BTreeMap<&str, f64> = per.iter().map(|(x, _)| (x.id.as_str(), x.weight * x.a_us)).collect();
    let whole: Vec<&Vec<Scored>> = match epochs.iter().filter(|e| e.len() == ids.len()).collect::<Vec<_>>() {
        none if none.is_empty() => epochs.iter().collect(),
        done => done,
    };
    let total_of = |pick: fn(&Scored) -> Option<Paired>| {
        whole
            .iter()
            .map(|e| combine(&e.iter().filter_map(|x| pick(x).map(|p| (share[x.id.as_str()], p))).collect::<Vec<_>>()))
            .collect::<Option<Vec<_>>>()
            .and_then(|p| across(&p))
    };
    let total = total_of(|x| x.graph);
    let changed_total = total_of(|x| x.changed);
    let verdict = total.map_or(Verdict::Unresolved, |p| p.verdict());
    let report = |side: &str| beside(&o.out, side).file_name().map(|f| f.to_string_lossy().into_owned());
    let scenarios = per
        .iter()
        .map(|(x, got)| {
            json!({"id":x.id,"weight":x.weight,"a_p50_us":x.a_us,"b_p50_us":x.b_us,"graph":paired_json(x.graph),
                "changed_ops":paired_json(x.changed),"same_tokens":x.same_tokens,
                "epochs":got.iter().map(|e| json!({"a_p50_us":e.a_us,"b_p50_us":e.b_us,"graph":paired_json(e.graph),
                    "changed_ops":paired_json(e.changed)})).collect::<Vec<_>>()})
        })
        .collect::<Vec<_>>();
    let value = json!({
        "version": super::VERSION,
        "kind": "against",
        "epochs_done": epochs.iter().filter(|e| e.len() == ids.len()).count(),
        "a": {"manifest_sha256": headers[0]["manifest_sha256"], "report": report("a")},
        "b": {"manifest_sha256": headers[1]["manifest_sha256"], "report": report("b")},
        "hardware": headers[0]["hardware"],
        "protocol": headers[0]["protocol"],
        "changed_ops": changed,
        "scenarios": scenarios,
        "cost": {"a_us":cost(|x| x.a_us),"b_us":cost(|x| x.b_us),"change":paired_json(total),
            "changed_ops":paired_json(changed_total)},
        "verdict": verdict,
    });
    let cost = (cost(|x| x.a_us), cost(|x| x.b_us));
    Summary { value, verdict, scenarios: per.into_iter().map(|(x, _)| x).collect(), total, changed_total, cost }
}

pub(super) fn run(
    o: &BenchOpts,
    inputs: &Inputs,
    w: &Workload,
    a: (&Verified, &[u8]),
    b: (&Verified, &[u8]),
) -> Result<Verdict> {
    ensure!(
        o.block >= 1 && o.max_rounds >= 2 && o.epochs >= 1,
        "--block and --epochs must be at least 1 and --max-rounds at least 2"
    );
    let started = Instant::now();
    let corpus = super::corpus(inputs.tokenizer()?, w.seed)?;
    let host_weights = HostWeights::new();
    let manifests = [a, b];
    let load = |side: usize, lent: Option<Vec<Resident>>, first_load: bool| {
        let (name, (m, bytes)) = (["A", "B"][side], manifests[side]);
        let at = Instant::now();
        let (mut bench, plan) = super::open(m, inputs, w, &corpus, &host_weights, lent)
            .with_context(|| format!("loading {name} beside the other (both stay resident)"))?;
        bench.quiet = true;
        bench.prime(plan.source)?;
        say_timed("load", format!("{name} · {} tokens of real prefix per rank", plan.source), at);
        if first_load {
            say_setup(&bench, &plan);
        }
        let mut header = preamble(m, bytes, w, &plan, &bench.ranks[0].probe, bench.ranks.len())?;
        header["protocol"]["against"] = json!({"weights_shared":!o.no_share,"block":o.block,"warmup_per_scenario":WARMUP,
            "order":"(B) A B A, (A) B A B, … per round of blocks, the first unscored","precision":o.precision,"max_rounds":o.max_rounds,"min_rounds":MIN_ROUNDS,
            "epochs":o.epochs,"epoch_order":"A loads first on even epochs, B on odd; both afresh each epoch"});
        anyhow::Ok((Side { name, bench, manifest: m.clone(), header }, plan.scenarios))
    };
    let changed = changed_ops(a.0, b.0);
    say(
        "changed",
        match changed.len() {
            0 => "no op differs".into(),
            _ => changed.iter().cloned().collect::<Vec<_>>().join(" · "),
        },
    );
    let mut resident: Option<Vec<Resident>> = None;
    let mut epochs: Vec<Vec<Scored>> = Vec::new();
    let mut headers: Option<[Value; 2]> = None;
    let mut records = [Vec::new(), Vec::new()];
    for epoch in 0..o.epochs {
        let owner = epoch % 2;
        let (first, scenarios) = load(owner, resident.take(), epoch == 0)?;
        let lent = (!o.no_share).then(|| first.bench.ranks.iter().map(|r| r.rt.lend()).collect());
        let (second, scenarios_other) = load(1 - owner, lent, false)?;
        let ids = |v: &[Scenario]| v.iter().map(|s| (s.id.clone(), s.program.clone())).collect::<Vec<_>>();
        ensure!(ids(&scenarios) == ids(&scenarios_other), "A and B take different scenarios from the workload");
        if epoch == 0 {
            let shared: u64 = second.bench.ranks.iter().map(|r| r.rt.kept_bytes()).sum();
            say("shared", format!("B reads {:.1} GiB of A's weights in place", shared as f64 / (1u64 << 30) as f64));
        }
        let sides = match owner {
            0 => vec![first, second],
            _ => vec![second, first],
        };
        let headers = headers.get_or_insert_with(|| [sides[0].header.clone(), sides[1].header.clone()]);
        let mut run = Run { o, sides, owner, epoch, changed: changed.clone() };
        epochs.push(Vec::new());
        for s in &scenarios {
            let scored = run.scenario(s, &mut records)?;
            epochs.last_mut().expect("pushed above").push(scored);
            super::write(&o.out, &summary(o, headers, &changed, &epochs).value)?;
        }
        for (i, side) in run.sides.iter().enumerate() {
            let mut report = side.header.clone();
            report["scenarios"] = Value::Array(records[i].clone());
            super::write(&beside(&o.out, &side.name.to_lowercase()), &report)?;
        }
        resident = Some(run.release());
    }
    let done = summary(o, headers.as_ref().expect("at least one epoch"), &changed, &epochs);
    for x in &done.scenarios {
        say("over", format!("{} · {:.3} → {:.3} ms · {}", x.id, x.a_us / 1e3, x.b_us / 1e3, line(x.graph)));
    }
    say(
        "cost",
        format!(
            "{:.3} → {:.3} s · {} · {} epochs",
            done.cost.0 / 1e6,
            done.cost.1 / 1e6,
            line(done.total),
            epochs.len()
        ),
    );
    if !changed.is_empty() {
        say("changed", format!("their exclusive time {}", line(done.changed_total)));
    }
    say("verdict", done.verdict.word().into());
    let verdict = done.verdict;
    say_timed("out", o.out.display().to_string(), started);
    Ok(verdict)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL: &str = include_str!("../../../../examples/minimal.json");

    fn manifest(f: impl FnOnce(&mut Value)) -> Manifest {
        let mut m: Value = serde_json::from_str(MINIMAL).unwrap();
        f(&mut m);
        serde_json::from_value(m).unwrap()
    }

    #[test]
    fn the_changed_ops_are_those_implemented_or_called_differently() {
        let base = manifest(|_| {});
        assert!(changed_ops(&base, &base).is_empty());
        let twice = manifest(|m| {
            let calls = m["programs"]["step"]["calls"].as_array_mut().unwrap();
            calls.push(calls[0].clone());
        });
        assert_eq!(changed_ops(&base, &twice), BTreeSet::from(["scale".to_string()]));
        let wider = manifest(|m| m["ops"]["scale"]["impl"]["launches"][0]["block"][0] = json!(128));
        assert_eq!(changed_ops(&base, &wider), BTreeSet::from(["scale".to_string()]));
    }
}
