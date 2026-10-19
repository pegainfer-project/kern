//! e2e_logits on the host: the recording format, the scoring arithmetic,
//! the sampler, the feeding plan against a fake pool, and the judgment.

use std::collections::BTreeMap;

use kern_test::e2e::{self, Judgment, Plan, Position, Recording, Row, Sampling, Step, Top, TOP};

/// splitmix64 over a counter: the tests' only randomness, seeded.
struct Rand(u64);

impl Rand {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let z = (self.0 ^ (self.0 >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        let z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn logits(&mut self, vocab: usize) -> Vec<f32> {
        (0..vocab).map(|_| (self.next() % 2000) as f32 / 100.0 - 10.0).collect()
    }
}

fn scratch(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("kern-test-e2e-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn a_recording_reads_back_as_written() {
    let mut r = Rand(1);
    let rows: Vec<Row> = (0..3)
        .map(|i: i32| {
            let n = i + 1;
            Row {
                source: format!("q{i}"),
                prompt: (0..5 + i).collect(),
                output: (0..n).map(|t| t * 7).collect(),
                top: (0..n).map(|_| e2e::top(&r.logits(64))).collect(),
            }
        })
        .collect();
    let meta = BTreeMap::from([
        ("producer".to_string(), "m.json sha256:00".to_string()),
        ("seed".to_string(), "3".to_string()),
    ]);
    let rec = Recording { meta, rows };
    let path = scratch("roundtrip").join("r.parquet");
    rec.write(&path).unwrap();
    assert_eq!(Recording::read(&path).unwrap(), rec);
}

#[test]
fn a_recording_with_the_wrong_columns_is_named() {
    let rec = Recording { meta: BTreeMap::new(), rows: Vec::new() };
    let path = scratch("columns").join("r.parquet");
    rec.write(&path).unwrap();
    let err = e2e::questions(&path).unwrap_err().to_string();
    assert!(err.contains("no column `messages`"), "{err}");
}

#[test]
fn top_is_the_largest_logits_as_log_probabilities() {
    let row = Rand(2).logits(1000);
    let t = e2e::top(&row);
    let mut ids: Vec<usize> = (0..row.len()).collect();
    ids.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
    let lse = row.iter().map(|&x| (x as f64).exp()).sum::<f64>().ln();
    assert_eq!(t.ids, ids[..TOP].iter().map(|&i| i as i32).collect::<Vec<_>>());
    assert!(t.ids.iter().zip(&t.logprobs).all(|(&i, &lp)| (lp as f64 - (row[i as usize] as f64 - lse)).abs() < 1e-5));
}

#[test]
fn kl_is_zero_against_itself_and_exact_when_the_top_holds_the_mass() {
    let row = Rand(3).logits(500);
    assert!(e2e::kl(&e2e::top(&row), &row) < 1e-9);
    // a vocabulary of TOP: the top is the whole distribution, the bucket empty
    let (p, q) = (Rand(4).logits(TOP), Rand(5).logits(TOP));
    let lse = |r: &[f32]| r.iter().map(|&x| (x as f64).exp()).sum::<f64>().ln();
    let full: f64 = p
        .iter()
        .zip(&q)
        .map(|(&a, &b)| {
            let (la, lb) = (a as f64 - lse(&p), b as f64 - lse(&q));
            la.exp() * (la - lb)
        })
        .sum();
    assert!((e2e::kl(&e2e::top(&p), &q) - full).abs() < 1e-5);
}

#[test]
fn kl_counts_the_rest_as_one_bucket() {
    // base: everything on token 0; the candidate moves half of it to a token
    // outside the base's top
    let base = Top { ids: vec![0], logprobs: vec![0.0] };
    let row = [0.0f32, 0.0];
    assert!((e2e::kl(&base, &row) - 2f64.ln()).abs() < 1e-9);
}

#[test]
fn sampling_draws_the_softmax_at_its_temperature_within_top_k_and_top_p() {
    // probabilities 0.1, 0.3, 0.6 at temperature 1
    let row = [0.0f32, 3f32.ln(), 6f32.ln()];
    let shares = |t: f64, k: usize, p: f64| {
        let s = Sampling::new(t, k, p).unwrap();
        let mut n = [0usize; 3];
        (0..100).for_each(|r| (0..100).for_each(|i| n[e2e::sample(&row, s, 7, r, i) as usize] += 1));
        n.map(|c| c as f64 / 1e4)
    };
    let near = |a: [f64; 3], b: [f64; 3]| a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 0.02);
    let warm = [1.0, 3f64.sqrt(), 6f64.sqrt()].map(|w| w / (1.0 + 3f64.sqrt() + 6f64.sqrt()));
    assert!(near(shares(1.0, 0, 1.0), [0.1, 0.3, 0.6]), "{:?}", shares(1.0, 0, 1.0));
    assert!(near(shares(2.0, 0, 1.0), warm), "{:?}", shares(2.0, 0, 1.0));
    assert!(near(shares(1.0, 2, 1.0), [0.0, 1.0 / 3.0, 2.0 / 3.0]), "{:?}", shares(1.0, 2, 1.0));
    assert_eq!((shares(1.0, 0, 0.5), shares(1.0, 0, 0.7)[0]), ([0.0, 0.0, 1.0], 0.0));
    assert_eq!(shares(0.0, 0, 1.0), [0.0, 0.0, 1.0]);
    assert!(Sampling::new(1.0, 0, 0.0).is_err() && Sampling::new(-1.0, 0, 1.0).is_err());
}

/// The plan carried out against a pool of `budget` token slots that every
/// row fits alone: the shell's whole job, with a step's distribution drawn
/// from what was fed.
struct Run {
    /// Per row, the tokens fed in position order.
    fed: Vec<Vec<i64>>,
    /// Per row, the tokens fed when each distribution was scored.
    scored_at: Vec<Vec<usize>>,
    /// Rows in the order their leases were granted.
    admitted: Vec<usize>,
    plan: Plan,
}

fn carry(
    mut plan: Plan,
    rows: usize,
    budget: usize,
    groups: usize,
    chunk: usize,
    draw: impl Fn(usize, usize) -> Option<i64>,
) -> Run {
    let mut run = Run {
        fed: vec![Vec::new(); rows],
        scored_at: vec![Vec::new(); rows],
        admitted: Vec::new(),
        plan: plan.clone(),
    };
    let (mut used, mut held) = (0, BTreeMap::new());
    while let Some(step) = plan.step() {
        match step {
            Step::Lease { row, tokens } => {
                let ok = used + tokens <= budget;
                if ok {
                    used += tokens;
                    held.insert(row, tokens);
                    run.admitted.push(row);
                }
                plan.leased(ok);
            }
            Step::Prefill { row, at, ids, scores } => {
                assert!(held.contains_key(&row) && ids.len() <= chunk && at == run.fed[row].len());
                run.fed[row].extend(&ids);
                if scores {
                    run.scored_at[row].push(run.fed[row].len());
                    plan.scored(draw(row, plan.position(row)));
                }
            }
            Step::Decode(batch) => {
                assert!(batch.len() <= groups);
                let mut rs: Vec<usize> = batch.iter().map(|b| b.0).collect();
                rs.sort();
                rs.dedup();
                assert_eq!(rs.len(), batch.len(), "a row twice in one call");
                for &(r, pos, t) in &batch {
                    assert!(held.contains_key(&r) && pos == run.fed[r].len());
                    run.fed[r].push(t);
                    assert!(run.fed[r].len() <= held[&r], "row {r} fed past its lease");
                }
                for &(r, ..) in &batch {
                    run.scored_at[r].push(run.fed[r].len());
                    plan.scored(draw(r, plan.position(r)));
                }
            }
            Step::Release(row) => used -= held.remove(&row).expect("released once"),
        }
    }
    assert!(held.is_empty() && used == 0, "leases left at the end");
    run.plan = plan;
    run
}

#[test]
fn a_forced_plan_feeds_every_row_its_baseline_once_longest_first() {
    for seed in 0..300 {
        let mut r = Rand(seed);
        let (groups, chunk, scores) = (1 + r.below(5), 1 + r.below(9), r.below(2) == 1);
        let n = 1 + r.below(9);
        let rows: Vec<(Vec<i64>, Vec<i64>)> = (0..n)
            .map(|_| {
                let (p, o) = (1 + r.below(30), 1 + r.below(30));
                ((0..p).map(|_| r.below(100) as i64).collect(), (0..o).map(|_| r.below(100) as i64).collect())
            })
            .collect();
        let budget = rows.iter().map(|(p, o)| p.len() + o.len() - 1).max().unwrap() + r.below(60);
        let plan = Plan::forced(&rows, groups, chunk, scores).unwrap();
        let run = carry(plan, n, budget, groups, chunk, |_, _| None);
        for (i, (p, o)) in rows.iter().enumerate() {
            let want: Vec<i64> = [p.as_slice(), &o[..o.len() - 1]].concat();
            let at: Vec<usize> = (0..o.len()).map(|k| p.len() + k).collect();
            assert_eq!((&run.fed[i], &run.scored_at[i]), (&want, &at), "seed {seed} row {i}");
        }
        let lens: Vec<usize> = run.admitted.iter().map(|&i| rows[i].1.len()).collect();
        assert!(lens.windows(2).all(|w| w[0] >= w[1]), "seed {seed}: admitted out of order {lens:?}");
    }
}

#[test]
fn a_sampled_plan_stops_at_a_stop_token_or_the_cap() {
    const STOP: i64 = 3;
    for seed in 0..200 {
        let mut r = Rand(1000 + seed);
        let (groups, chunk, scores) = (1 + r.below(4), 1 + r.below(7), r.below(2) == 1);
        let max_new = 1 + r.below(20);
        let n = 1 + r.below(6);
        let prompts: Vec<Vec<i64>> =
            (0..n).map(|_| (0..1 + r.below(20)).map(|_| 10 + r.below(50) as i64).collect()).collect();
        // the draw a sampler would make: a function of row and step only
        let draw = |row: usize, k: usize| Some(((row * 31 + k * 17 + seed as usize) % 23) as i64);
        let plan = Plan::sampled(&prompts, max_new, &[STOP], groups, chunk, scores).unwrap();
        let run = carry(plan, n, 1000, groups, chunk, draw);
        for (i, p) in prompts.iter().enumerate() {
            let want: Vec<i64> = (0..max_new)
                .map(|k| draw(i, k).unwrap())
                .scan(false, |end, t| (!std::mem::replace(end, t == STOP)).then_some(t))
                .collect();
            assert_eq!(run.plan.output(i), want.as_slice(), "seed {seed} row {i}");
            assert_eq!(run.fed[i], [p.as_slice(), &want[..want.len() - 1]].concat());
        }
    }
}

#[test]
fn a_judgment_breaks_the_mean_down_by_depth_and_step() {
    let at = |row, step, depth, kl, cand| Position { row, step, depth, kl, margin: 1.0, base: 7, cand };
    let ps = vec![at(0, 0, 100, 0.1, 7), at(0, 1, 100, 0.3, 8), at(1, 600, 40 << 10, 0.5, 9)];
    let j = Judgment::of(&ps, 0.4);
    let depth: Vec<(usize, usize)> = j.depth.iter().map(|b| (b.from, b.positions)).collect();
    let step: Vec<(usize, usize)> = j.step.iter().map(|b| (b.from, b.positions)).collect();
    assert_eq!((j.positions, (j.mean_kl - 0.3).abs() < 1e-12, j.pass()), (3, true, true));
    assert_eq!(
        (depth, step),
        (vec![(0, 2), (8 << 10, 0), (32 << 10, 1), (128 << 10, 0)], vec![(0, 1), (1, 1), (64, 0), (512, 1)])
    );
    assert_eq!((j.flips, j.worst.iter().map(|p| p.row).collect::<Vec<_>>()), (2, vec![1, 0]));
    assert!(!Judgment::of(&ps, 0.2).pass() && !Judgment::of(&[], 1.0).pass());
}
