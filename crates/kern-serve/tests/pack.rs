//! The pipeline head's packing policy: what an item carries, in what turn,
//! and when it goes out.

use kern_serve::pack::{goes_out, pack, passed, turn, Bounds, Waiting};

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
fn a_full_item_runs_one_ahead_of_the_stages_and_a_partial_one_none() {
    let full = [(0, 8)];
    let partial = [(0, 3)];
    let seqs_full = [(0, 1), (1, 1), (2, 1)];
    let at = |plan: &[(usize, usize)], flying| goes_out(plan, B, flying, 4);
    assert_eq!((at(&full, 4), at(&full, 5)), (true, false));
    assert_eq!((at(&seqs_full, 4), at(&seqs_full, 5)), (true, false));
    assert_eq!((at(&partial, 3), at(&partial, 4)), (true, false));
    assert!(!at(&[], 0));
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
