//! A/B as rounds of blocks of replays, A's beside B's, so the two see the
//! same clocks, temperature and power cap, and a round's log ratio of
//! block medians cancels what drifts slower than a round. The change is
//! the rounds' mean log ratio, with a Student-t interval over them.

use serde::Serialize;

/// Two-sided 95% Student-t quantiles for 1..=30 degrees of freedom.
const T975: [f64; 30] = [
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120,
    2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
];

pub(super) fn t975(df: usize) -> f64 {
    match df {
        0 => f64::INFINITY,
        d if d <= T975.len() => T975[d - 1],
        d => 1.96 + 2.4 / d as f64,
    }
}

pub(super) fn median(v: &[f64]) -> f64 {
    assert!(!v.is_empty());
    let mut v = v.to_vec();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    match n % 2 {
        1 => v[n / 2],
        _ => (v[n / 2 - 1] + v[n / 2]) / 2.,
    }
}

/// B against A over rounds of blocks, as a log ratio: `mean` is ln(B / A),
/// `se` its standard error, `half` the 95% half-width.
#[derive(Serialize, Debug, Clone, Copy, PartialEq)]
pub(super) struct Paired {
    pub rounds: usize,
    pub mean: f64,
    pub se: f64,
    pub half: f64,
}

impl Paired {
    /// Per round, A's and B's statistic; a round with a zero side is left
    /// out. `None` with fewer than two rounds.
    pub(super) fn of(pairs: &[(f64, f64)]) -> Option<Paired> {
        let d: Vec<f64> = pairs.iter().filter(|(a, b)| *a > 0. && *b > 0.).map(|(a, b)| (b / a).ln()).collect();
        let n = d.len();
        (n >= 2).then(|| {
            let mean = d.iter().sum::<f64>() / n as f64;
            let var = d.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
            let se = (var / n as f64).sqrt();
            Paired { rounds: n, mean, se, half: t975(n - 1) * se }
        })
    }

    /// Change in percent with its interval: (mean, low, high).
    pub(super) fn percent(&self) -> (f64, f64, f64) {
        let p = |x: f64| 100. * x.exp_m1();
        (p(self.mean), p(self.mean - self.half), p(self.mean + self.half))
    }

    pub(super) fn verdict(&self) -> Verdict {
        match (self.mean + self.half < 0., self.mean - self.half > 0.) {
            (true, _) => Verdict::Faster,
            (_, true) => Verdict::Slower,
            _ => Verdict::Unresolved,
        }
    }
}

/// Weighted sum of independent per-scenario changes, each weighted by its
/// share of A's cost: ln ratios add as fractions to first order.
pub(super) fn combine(parts: &[(f64, Paired)]) -> Option<Paired> {
    let total: f64 = parts.iter().map(|(w, _)| w).sum();
    (total > 0. && !parts.is_empty()).then(|| {
        let mean = parts.iter().map(|(w, p)| w / total * p.mean).sum();
        let se = parts.iter().map(|(w, p)| (w / total * p.se).powi(2)).sum::<f64>().sqrt();
        let rounds = parts.iter().map(|(_, p)| p.rounds).min().unwrap_or(0);
        Paired { rounds, mean, se, half: 1.96 * se }
    })
}

/// One change measured once per epoch, in epoch order. Each epoch's
/// placement is one draw of a bias its own rounds cannot see, so the
/// interval is Student-t over the epochs, and never narrower than their
/// rounds' noise pooled. Which side loads first alternates with the
/// epoch and moves the bias the same way every time it is the same, so
/// from four epochs on it is a blocking factor: the change is the mean of
/// the even epochs' mean and the odd ones', its error the pooled spread
/// within each on n - 2 degrees of freedom.
pub(super) fn across(epochs: &[Paired]) -> Option<Paired> {
    let n = epochs.len();
    let mean_of = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let spread = |v: &[f64]| v.iter().map(|x| (x - mean_of(v)).powi(2)).sum::<f64>();
    let means: Vec<f64> = epochs.iter().map(|p| p.mean).collect();
    let (weights, between, df): (Vec<f64>, f64, usize) = match n {
        0 => return None,
        1 => return Some(epochs[0]),
        2 | 3 => (vec![1. / n as f64; n], (spread(&means) / ((n - 1) * n) as f64).sqrt(), n - 1),
        _ => {
            let [even, odd]: [Vec<f64>; 2] = [0, 1].map(|r| means.iter().skip(r).step_by(2).copied().collect());
            let (ne, no) = (even.len() as f64, odd.len() as f64);
            let pooled = (spread(&even) + spread(&odd)) / (n - 2) as f64;
            let weights = (0..n).map(|i| if i.is_multiple_of(2) { 0.5 / ne } else { 0.5 / no }).collect();
            (weights, (pooled / 4. * (1. / ne + 1. / no)).sqrt(), n - 2)
        }
    };
    let mean = weights.iter().zip(&means).map(|(w, m)| w * m).sum();
    let within = weights.iter().zip(epochs).map(|(w, p)| (w * p.se).powi(2)).sum::<f64>().sqrt();
    let se = between.max(within);
    Some(Paired { rounds: epochs.iter().map(|p| p.rounds).sum(), mean, se, half: t975(df) * se })
}

#[derive(Serialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum Verdict {
    Faster,
    Slower,
    Unresolved,
}

impl Verdict {
    /// The exit status `kern bench --against` ends with.
    pub(super) fn code(self) -> i32 {
        match self {
            Verdict::Faster => 0,
            Verdict::Slower => 1,
            Verdict::Unresolved => 2,
        }
    }

    pub(super) fn word(self) -> &'static str {
        match self {
            Verdict::Faster => "B faster",
            Verdict::Slower => "B slower",
            Verdict::Unresolved => "unresolved",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_sides_center_on_zero() {
        let p = Paired::of(&[(100., 100.1), (101., 100.9), (99.5, 99.6), (100.2, 100.1)]).unwrap();
        assert!(p.mean.abs() < 1e-3 && p.verdict() == Verdict::Unresolved);
    }

    #[test]
    fn a_steady_one_percent_win_resolves_through_drift() {
        // the GPU slows 3% across the run; B is 1% faster within every round
        let pairs: Vec<(f64, f64)> = (0..8)
            .map(|i| (i, 100. * (1. + 0.004 * i as f64)))
            .map(|(i, a)| (a, a * if i % 2 == 0 { 0.989 } else { 0.991 }))
            .collect();
        let p = Paired::of(&pairs).unwrap();
        let (mid, lo, hi) = p.percent();
        assert!((mid + 1.).abs() < 0.05 && hi < 0. && lo > -1.2, "{mid} [{lo}, {hi}]");
        assert_eq!(p.verdict(), Verdict::Faster);
    }

    #[test]
    fn one_round_is_no_interval() {
        assert_eq!(Paired::of(&[(1., 2.)]), None);
        assert_eq!(Paired::of(&[(0., 2.), (1., 1.)]), None);
    }

    #[test]
    fn combining_weights_means_and_adds_variances() {
        let a = Paired { rounds: 4, mean: -0.02, se: 0.003, half: 0. };
        let b = Paired { rounds: 6, mean: 0.0, se: 0.004, half: 0. };
        let c = combine(&[(3., a), (1., b)]).unwrap();
        assert!((c.mean + 0.015).abs() < 1e-12);
        assert!((c.se - (0.75f64.powi(2) * 9e-6 + 0.25f64.powi(2) * 16e-6).sqrt()).abs() < 1e-12);
        assert_eq!(c.rounds, 4);
    }

    #[test]
    fn epochs_that_disagree_widen_the_interval_past_their_rounds() {
        let e = |mean| Paired { rounds: 4, mean, se: 0.0002, half: 0. };
        assert_eq!(across(&[e(0.003)]).unwrap().mean, 0.003);
        let flipped = across(&[e(0.003), e(-0.003), e(-0.0028)]).unwrap();
        assert!(flipped.half > 0.003 && flipped.verdict() == Verdict::Unresolved);
        let agree = across(&[e(-0.01), e(-0.0101), e(-0.0099), e(-0.01)]).unwrap();
        assert!((agree.mean + 0.01).abs() < 1e-12 && agree.verdict() == Verdict::Faster);
        assert_eq!(agree.rounds, 16);
    }

    #[test]
    fn the_load_order_is_blocked_out() {
        // the side loaded first runs 0.15% fast; B is 0.2% slower
        let e = |i: usize| Paired {
            rounds: 4,
            mean: 0.002 + if i.is_multiple_of(2) { -0.0015 } else { 0.0015 },
            se: 1e-4,
            half: 0.,
        };
        let p = across(&(0..4).map(e).collect::<Vec<_>>()).unwrap();
        assert!((p.mean - 0.002).abs() < 1e-12 && p.verdict() == Verdict::Slower, "{p:?}");
        let noisy = across(&[e(0), e(1), Paired { mean: 0.004, ..e(0) }, e(1)]).unwrap();
        assert!(noisy.half > 0.003 && noisy.verdict() == Verdict::Unresolved, "{noisy:?}");
    }

    #[test]
    fn t_quantiles_fall_toward_the_normal() {
        assert_eq!(t975(1), 12.706);
        assert!(t975(30) > t975(31) && t975(31) > 1.96 && t975(1000) < 1.97);
        assert_eq!(median(&[3., 1., 2., 10.]), 2.5);
    }
}
