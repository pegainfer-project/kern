//! A rollout keeps its queue's turn as items go and finds a budget's rows
//! from a root of the cost model. This is that rollout against one that
//! sorts the queue anew before every item and bisects for the rows: the
//! same bits from the same queue, whatever the model.

use kern_serve::cost::Cost;
use kern_serve::frontier::{
    apply, pack, place, plan_cost, Pipe, Rule, Seq, Start, Terms, Turn, CAPS, PAGE, REPLY, SEQS,
};

const AHEAD: usize = 4;

/// A cost model of stages (ms fixed, µs per row, ms per 1k context, ns per
/// row and context token, ms per sequence).
fn model(stages: &[[f64; 5]]) -> Cost {
    let stage = |(s, [a, b, c, d, e]): (usize, &[f64; 5])| {
        format!(
            r#""{s}": {{"a_ms": {a}, "b_us_per_row": {b}, "c_ms_per_1k_ctx": {c}, "d_ns_per_row_ctx": {d}, "e_ms_per_seq": {e}}}"#
        )
    };
    let fit: Vec<String> = stages.iter().enumerate().map(stage).collect();
    Cost::parse(&format!(r#"{{"fit": {{{}}}}}"#, fit.join(", "))).expect("the model parses")
}

/// Uniform values in [0, 1) from `seed`.
fn uniform(seed: u64) -> impl Iterator<Item = f64> {
    std::iter::successors(Some(seed), |x| Some(x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)))
        .skip(1)
        .map(|x| (x >> 11) as f64 / (1u64 << 53) as f64)
}

/// The most rows within `budget`, by bisection over pages.
fn bisected(seqs: &[Seq], turn: &Turn, budget: f64, m: &Cost, context: i64) -> i64 {
    let hi = CAPS[CAPS.len() - 1];
    let mut parts = Vec::<(i64, i64)>::new();
    let (mut rows, mut ctx) = (0, 0);
    for k in turn.walk() {
        if parts.len() == SEQS || rows == hi {
            break;
        }
        let p = seqs[k].pos;
        let n = turn.left[k].min(hi - rows).min((context - ctx - p).max(0));
        if n <= 0 {
            continue;
        }
        let over = |n: i64| m.slowest(&[&parts[..], &[(p, n)]].concat()) > budget;
        if over(n) {
            let (mut lo, mut top) = (0, n.div_euclid(PAGE));
            while lo < top {
                let mid = (lo + top + 1).div_euclid(2);
                if over(mid * PAGE) {
                    top = mid - 1
                } else {
                    lo = mid
                }
            }
            rows += lo * PAGE;
            break;
        }
        parts.push((p, n));
        rows += n;
        ctx += p + n;
    }
    rows.max(PAGE)
}

/// `plan_cost` with the queue sorted anew before every item.
fn anew(at: Start, rule: Rule, m: &Cost, terms: Terms) -> f64 {
    let (stages, now) = (m.stages(), at.now);
    let (mut seqs, mut pipe, mut answers) = (at.seqs.to_vec(), *at.pipe, at.answers.to_vec());
    answers.sort_by(f64::total_cmp);
    let (mut total, mut ride, mut busy, mut last) = (0.0, 0.0, 0.0, None::<(f64, f64)>);
    while !seqs.is_empty() {
        let turn = Turn::of(&seqs, stages);
        let cap = match rule {
            Rule::Rows(c) => c,
            Rule::Budget(b) => bisected(&seqs, &turn, b, m, at.context),
        };
        let (plan, _) = pack(&seqs, &turn, cap, at.context);
        if plan.is_empty() {
            break;
        }
        let window = stages + at.ahead;
        let go = if answers.len() < window { now } else { now.max(answers[answers.len() - window]) };
        let (parts, fin) = apply(&mut seqs, &plan, &turn.left);
        let ds = m.times(&parts);
        busy += ds[..stages].iter().fold(0.0, |a: f64, &b| a.max(b));
        let (next, end) = place(&pipe, go, &ds);
        pipe = next;
        if let Some((g, e)) = last {
            ride += (go - g) * (e - g) - (go - g) * (go - g) / 2.0;
        }
        last = Some((go, end));
        answers.push(end + REPLY);
        total += (end + REPLY - now) * fin.len() as f64;
    }
    if let Some((g, e)) = last {
        ride += (e - g) * (e - g) / 2.0;
    }
    let base = if terms.lam > 0.0 { total + terms.lam * ride } else { total };
    if terms.price > 0.0 {
        base + terms.price * busy
    } else {
        base
    }
}

/// A queue of up to 24 prompts: cold or behind a kept prefix, some stopping
/// at a checkpoint first, passed over by up to twice the stages.
fn queue(u: &mut impl Iterator<Item = f64>, stages: usize) -> Vec<Seq> {
    let mut next = || u.next().expect("the stream does not end");
    let waiting = 1 + (next() * 24.0) as usize;
    (0..waiting)
        .map(|i| {
            let left = 64 + (next() * next() * 60_000.0) as i64;
            let hit = if next() < 0.4 { 0 } else { (next() * 200_000.0) as i64 };
            let stop = if next() < 0.2 { hit + 1 + (next() * (left - 1) as f64) as i64 } else { hit + left };
            Seq::new(i, hit, hit + left, 0.0).at(hit, stop, (next() * (2 * stages + 2) as f64) as u32)
        })
        .collect()
}

#[test]
fn a_rollout_kept_as_items_go_is_the_one_sorted_anew_before_every_item() {
    let dsv41 = Cost::parse(include_str!("fixtures/pp/dsv41-pp4-cost.json")).expect("the costfit parses");
    let deep: Vec<[f64; 5]> = (0..8).map(|s| [20.0 + s as f64, 2.0 + 0.1 * s as f64, 1e-4, 4e-3, 0.05]).collect();
    let falling = [[5.0, 3.0, 0.0, 2e-3, 0.1], [9.0, 0.5, 1e-3, -2e-3, 0.0], [4.0, -0.2, 0.0, 3e-3, 1.0]];
    for (which, m) in [dsv41, model(&deep), model(&falling)].iter().enumerate() {
        let stages = m.stages();
        for seed in 0..60u64 {
            let mut u = uniform(seed * 3 + which as u64);
            let seqs = queue(&mut u, stages);
            let out = (u.next().expect("endless") * (stages + AHEAD + 3) as f64) as usize;
            let answers: Vec<f64> = u.by_ref().take(out).map(|x| 10_000.0 + x * 80_000.0).collect();
            let (pipe, context) = (Pipe::idle(stages, 9_000.0), [1 << 20, 300_000, 120_000][seed as usize % 3]);
            let at = Start { seqs: &seqs, pipe: &pipe, now: 10_000.0, answers: &answers, ahead: AHEAD, context };
            for rule in CAPS.iter().flat_map(|&c| [Rule::Rows(c), Rule::Budget(m.slowest(&[(0, c)]))]) {
                for terms in [Terms::default(), Terms { lam: 5e-6, price: 0.0 }, Terms { lam: 0.0, price: 3.0 }] {
                    let (kept, sorted) = (plan_cost(at, rule, m, terms), anew(at, rule, m, terms));
                    assert_eq!(kept.to_bits(), sorted.to_bits(), "model {which} seed {seed} {rule:?} {terms:?}");
                }
            }
        }
    }
}
