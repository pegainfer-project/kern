//! The pipeline head's packing policy: what an item carries, in what turn,
//! and when it goes out.

use kern_serve::cost::Cost;
use kern_serve::frontier::{self, goes_out, Seq, Turn, SEQS};
use kern_serve::pack::{pack, passed, turn, Bounds, Waiting};
use kern_serve::policy::{Meter, Policy};

const B: Bounds = Bounds { rows: 8, seqs: 3, context: 20 };

fn w(pos: usize, left: usize) -> Waiting {
    Waiting { pos, left, passed: 0 }
}

fn starved(pos: usize, left: usize, passed: usize) -> Waiting {
    Waiting { pos, left, passed }
}

#[test]
fn fills_the_call_shortest_first() {
    assert_eq!(pack(&[w(0, 5), w(0, 5), w(0, 5)], B, 2), vec![(0, 5), (1, 3)]);
    assert_eq!(pack(&[w(0, 20)], B, 2), vec![(0, 8)]);
    assert_eq!(pack(&[w(0, 1), w(0, 1), w(0, 1), w(0, 1)], B, 2), vec![(0, 1), (1, 1), (2, 1)]);
    assert_eq!(pack(&[w(4, 0), w(0, 2)], B, 2), vec![(1, 2)]);
    assert_eq!(pack(&[w(0, 7), w(10, 2)], B, 2), vec![(1, 2), (0, 6)]);
    assert_eq!(pack(&[], B, 2), vec![]);
}

#[test]
fn a_sequence_the_context_cannot_hold_is_passed_over() {
    assert_eq!(pack(&[w(16, 4), w(0, 8)], B, 2), vec![(0, 4)]);
    assert_eq!(pack(&[w(10, 2), w(0, 8)], B, 2), vec![(0, 2), (1, 6)]);
    assert_eq!(pack(&[w(15, 2), w(12, 1), w(0, 3)], B, 2), vec![(1, 1), (2, 3)]);
}

#[test]
fn the_most_passed_over_goes_first_once_patience_runs_out() {
    let seqs = [w(0, 9), starved(0, 8, 2), starved(0, 7, 3), w(0, 1)];
    assert_eq!((turn(&seqs, 2), turn(&seqs, 4)), (vec![2, 1, 3, 0], vec![3, 2, 1, 0]));
    assert_eq!(pack(&seqs, B, 2), vec![(2, 7), (1, 1)]);
    assert_eq!(passed(&seqs, &pack(&seqs, B, 2)), vec![1, 0, 0, 1]);
    assert_eq!(passed(&[w(3, 0), starved(0, 2, 5)], &[]), vec![0, 6]);
}

/// K3's shape: 8k rows, a 262144 context. A hit arriving behind four cold
/// 200k prompts goes into the next item; cold rows fill what its context
/// leaves, and none once the cold prompt is past what it leaves.
#[test]
fn a_hit_goes_ahead_of_cold_prompts() {
    let k3 = Bounds { rows: 8192, seqs: 16, context: 262_144 };
    let mut seqs = vec![w(0, 200_000); 4];
    seqs.push(w(200_000, 1_600));
    assert_eq!(pack(&seqs, k3, 8), vec![(4, 1_600), (0, 6_592)]);
    seqs[0] = w(100_000, 100_000);
    assert_eq!(pack(&seqs, k3, 8), vec![(4, 1_600), (1, 6_592)]);
    seqs[1..4].fill(w(100_000, 100_000));
    assert_eq!(pack(&seqs, k3, 8), vec![(4, 1_600)]);
}

#[test]
fn a_full_item_runs_ahead_of_the_stages_and_a_partial_one_does_not() {
    let at = |rows, seqs, flying, ahead| goes_out(rows, seqs, 8, flying, 4, ahead);
    assert_eq!((at(8, 1, 4, 1), at(8, 1, 5, 1)), (true, false));
    assert_eq!((at(8, 1, 7, 4), at(8, 1, 8, 4)), (true, false));
    assert_eq!((at(3, SEQS, 4, 1), at(3, SEQS, 5, 1)), (true, false));
    assert_eq!((at(3, 1, 3, 4), at(3, 1, 4, 4)), (true, false));
    assert!(!at(0, 0, 0, 4));
}

/// The cap `depth2048/16384/24` sets for `togo` rows waiting in one prompt.
fn depth(togo: i64) -> i64 {
    let seqs: Vec<Seq> = (togo > 0).then(|| Seq::new(0, 0, togo, 0)).into_iter().collect();
    let cost = Cost::free(4);
    Policy::parse("depth2048/16384/24").unwrap().cap(&Meter::new(4, 4).decision(0.0, &seqs, 0, &cost, 1 << 20))
}

/// K3's shape: a 16k chunk, a 2k floor, spread over 24 items. A lone hit
/// or a few cold prompts go in small items, a backlog in full chunks.
#[test]
fn an_item_takes_the_admitted_rows_spread_over_the_pipeline() {
    assert_eq!((depth(0), depth(1_000), depth(48_000)), (2_048, 2_048, 2_048));
    assert_eq!((depth(100_000), depth(200_000), depth(1_000_000)), (4_224, 8_384, 16_384));
}

/// The pipeline head packs over its own view of a sequence
/// (`frontier::pack`), the single-process scheduler over `Waiting`: over
/// random waiting sets they pack the same item.
#[test]
fn the_head_packs_as_the_scheduler_does() {
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    let mut rand = |n: u64| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x % n
    };
    for _ in 0..20_000 {
        let n = rand(24) as usize;
        let w: Vec<Waiting> = (0..n)
            .map(|_| Waiting {
                pos: rand(64) as usize,
                left: rand(3) as usize * rand(40) as usize,
                passed: rand(6) as usize,
            })
            .collect();
        let (cap, context, patience) = (1 + rand(80) as i64, 1 + rand(400) as i64, 1 + rand(4) as usize);
        let seqs: Vec<Seq> = (w.iter().enumerate())
            .map(|(i, s)| {
                let end = (s.pos + s.left) as i64;
                Seq::at(i, 0, s.pos as i64, end, end, s.passed as u32, 0)
            })
            .collect();
        let (plan, _) = frontier::pack(&seqs, &Turn::of(&seqs, patience), cap, context);
        let ours: Vec<(usize, usize)> = plan.iter().map(|&(k, r)| (k, r as usize)).collect();
        let b = Bounds { rows: cap as usize, seqs: SEQS, context: context as usize };
        assert_eq!(ours, pack(&w, b, patience), "{w:?} cap {cap} context {context} patience {patience}");
    }
}

/// Every small arrival pattern, fed to the end against the rules stated
/// plainly: each item within bounds and nonempty while rows wait, the
/// shortest first while nobody's patience has run out, `passed` the count
/// of items in a row that went without a sequence, no sequence going
/// without for `patience + n` items in a row (`n` the most sequences
/// waiting meanwhile), and every row sent exactly once.
#[test]
fn every_small_arrival_pattern_drains_without_starving() {
    let one: Vec<(usize, usize, usize)> =
        (0..3).flat_map(|at| [0, 9].into_iter().flat_map(move |pos| [1, 5, 11].map(|left| (at, pos, left)))).collect();
    let k = one.len();
    for n in 1..=4u32 {
        for code in 0..k.pow(n) {
            let arrivals: Vec<_> = (0..n).map(|d| one[code / k.pow(d) % k]).collect();
            for patience in [1, 2, 4] {
                drain(&arrivals, patience);
            }
        }
    }
}

fn drain(arrivals: &[(usize, usize, usize)], patience: usize) {
    let total: usize = arrivals.iter().map(|&(.., left)| left).sum();
    let (mut seqs, mut streaks, mut sent) = (Vec::<Waiting>::new(), Vec::<(usize, usize)>::new(), 0);
    for item in 0..=total + 2 {
        for &(_, pos, left) in arrivals.iter().filter(|&&(at, ..)| at == item) {
            seqs.push(w(pos, left));
            streaks.push((0, 0));
        }
        let plan = pack(&seqs, B, patience);
        let waiting = seqs.iter().filter(|s| s.left > 0).count();
        assert_eq!(plan.is_empty(), waiting == 0, "{arrivals:?} p{patience}: {seqs:?}");
        if plan.is_empty() && seqs.len() == arrivals.len() {
            break;
        }
        let rows: usize = plan.iter().map(|&(_, r)| r).sum();
        let context: usize = plan.iter().map(|&(i, r)| seqs[i].pos + r).sum();
        assert!(rows <= B.rows && plan.len() <= B.seqs && context <= B.context, "{seqs:?} {plan:?}");
        assert!(plan.iter().all(|&(i, r)| r > 0 && r <= seqs[i].left), "{seqs:?} {plan:?}");
        if !plan.is_empty() && seqs.iter().all(|s| s.passed < patience) {
            let shortest = (0..seqs.len()).filter(|&i| seqs[i].left > 0).min_by_key(|&i| (seqs[i].left, i));
            assert_eq!(Some(plan[0].0), shortest, "{seqs:?} {plan:?}");
        }
        let counts = passed(&seqs, &plan);
        for (i, s) in seqs.iter_mut().enumerate() {
            let r = plan.iter().find(|&&(j, _)| j == i).map_or(0, |&(_, r)| r);
            let (misses, peak) = streaks[i];
            streaks[i] = match (r, s.left) {
                (0, 0) => (misses, peak),
                (0, _) => (misses + 1, peak.max(waiting)),
                _ => (0, 0),
            };
            let (misses, peak) = streaks[i];
            assert!(misses < patience + peak, "{arrivals:?} p{patience}: seq {i} went without {misses} items");
            assert_eq!(counts[i], misses, "{arrivals:?} p{patience}: seq {i}");
            *s = Waiting { pos: s.pos + r, left: s.left - r, passed: counts[i] };
            sent += r;
        }
    }
    let drained = seqs.iter().all(|s| s.left == 0);
    assert_eq!((sent, seqs.len(), drained), (total, arrivals.len(), true), "{arrivals:?} p{patience}");
}
