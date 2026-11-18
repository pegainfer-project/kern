//! The choice between candidates, against every tiling written out: it has
//! the fewest ranges any tiling within the noise has, costs the least of
//! those, and covers every value.

use kern_run::bench::pick::{pick, Range, Table, Tiling};

struct Rand(u64);

impl Rand {
    fn next(&mut self, n: u64) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 33) % n
    }
}

/// Every way to cut `n` values into ranges, each range on its cheapest
/// candidate: (ranges, cost).
fn tilings(t: &Table) -> Vec<(usize, f64)> {
    let n = t.values.len();
    (0..1u32 << (n - 1))
        .map(|cuts| {
            let starts: Vec<usize> = std::iter::once(0).chain((1..n).filter(|i| cuts >> (i - 1) & 1 == 1)).collect();
            let cost = starts
                .iter()
                .enumerate()
                .map(|(k, &i)| {
                    let j = starts.get(k + 1).copied().unwrap_or(n);
                    (0..t.costs.len()).map(|c| t.costs[c][i..j].iter().sum::<f64>()).fold(f64::INFINITY, f64::min)
                })
                .sum();
            (starts.len(), cost)
        })
        .collect()
}

#[test]
fn the_pick_is_the_cheapest_of_the_fewest_ranges_within_the_noise() {
    let mut r = Rand(0x2545_f491_4f6c_dd1d);
    for case in 0..2000 {
        let n = 1 + r.next(7) as usize;
        let c = 1 + r.next(3) as usize;
        let values: Vec<u64> =
            (0..n as u64).scan(0, |v, _| Some(std::mem::replace(v, *v + 1 + r.next(5)) + 1)).collect();
        // The first candidate may run everywhere, as a manifest's own
        // implementation does; the others not at a cell in five.
        let cost = |r: &mut Rand, c: usize| match c > 0 && r.next(5) == 0 {
            true => f64::INFINITY,
            false => (1 + r.next(40)) as f64,
        };
        let costs = (0..c).map(|c| (0..n).map(|_| cost(&mut r, c)).collect()).collect();
        let t = Table { values, costs };
        let noise = [0., 0.02, 0.1, 0.5][r.next(4) as usize];

        let Tiling { ranges, cost, floor } = pick(&t, noise).unwrap();
        let all = tilings(&t);
        let cheapest = all.iter().map(|x| x.1).fold(f64::INFINITY, f64::min);
        let fewest = all.iter().filter(|x| x.1 <= cheapest * (1. + noise)).map(|x| x.0).min().unwrap();
        let of_fewest = all.iter().filter(|x| x.0 == fewest).map(|x| x.1).fold(f64::INFINITY, f64::min);
        assert_eq!((ranges.len(), cost, floor), (fewest, of_fewest, cheapest), "case {case}: {t:?}");

        assert_eq!((ranges[0].min, ranges.last().unwrap().max), (None, None), "case {case}");
        for w in ranges.windows(2) {
            assert_eq!(w[1].min, w[0].max.map(|v| v + 1), "case {case}: {ranges:?}");
        }
        let covered =
            |v: u64| ranges.iter().filter(move |x| x.min.is_none_or(|lo| v >= lo) && x.max.is_none_or(|hi| v <= hi));
        for (i, &v) in t.values.iter().enumerate() {
            let on: Vec<&Range> = covered(v).collect();
            assert_eq!(on.len(), 1, "case {case}: {v} in {ranges:?}");
            assert!(t.costs[on[0].candidate][i].is_finite(), "case {case}: {v} runs what may not");
        }
        assert_eq!(ranges.iter().map(|x| x.cost).sum::<f64>(), cost, "case {case}");
    }
}

#[test]
fn a_boundary_that_buys_less_than_the_noise_is_not_drawn() {
    let t = Table { values: vec![16, 32, 64], costs: vec![vec![100., 100., 100.], vec![99.9, 101., 99.9]] };
    let one = pick(&t, 0.01).unwrap();
    assert_eq!((one.ranges.len(), one.ranges[0].candidate, one.cost), (1, 0, 300.));
    let three = pick(&t, 0.).unwrap();
    assert_eq!(
        three.ranges.iter().map(|x| (x.candidate, x.min, x.max)).collect::<Vec<_>>(),
        [(1, None, Some(16)), (0, Some(17), Some(32)), (1, Some(33), None)]
    );
}

#[test]
fn a_candidate_is_chosen_beside_where_it_may_not_run_and_never_across_it() {
    let inf = f64::INFINITY;
    let t = Table { values: vec![1, 2, 4, 8], costs: vec![vec![10., 10., 10., 10.], vec![1., 1., inf, 1.]] };
    let got = pick(&t, 0.).unwrap();
    assert_eq!(
        got.ranges.iter().map(|x| (x.candidate, x.min, x.max)).collect::<Vec<_>>(),
        [(1, None, Some(2)), (0, Some(3), Some(4)), (1, Some(5), None)]
    );
    assert_eq!((got.cost, got.floor), (13., 13.));
}

#[test]
fn nothing_measured_or_a_value_nothing_may_run_at_is_no_pick() {
    assert_eq!(pick(&Table { values: vec![], costs: vec![vec![]] }, 0.), None);
    assert_eq!(pick(&Table { values: vec![1, 2], costs: vec![vec![1., f64::INFINITY]] }, 0.), None);
    assert_eq!(pick(&Table { values: vec![1], costs: vec![] }, 0.), None);
}
