//! Choosing between candidates from what was measured of them.
//!
//! One op, several implementations of it, and what each cost at every
//! value of the var the op's launches turn on: the choice is a tiling of
//! that var, one candidate per range, because a `when` is all a manifest
//! has to choose with. Choosing per value is the cheapest tiling and the
//! one with the most ranges. Every range is a launch the manifest carries
//! and a cubin the runtime loads, so the answer is the tiling with the
//! fewest ranges whose cost cannot be told from that cheapest one: within
//! `noise`, the relative difference two runs of the same manifest showed.
//! A boundary that buys less than the measurement can resolve is not
//! drawn.
//!
//! A candidate that may not run at a value costs infinity there, so it is
//! never chosen across it and still chosen beside it.
//!
//! Nothing is interpolated. Where a boundary falls between two measured
//! values more than a doubling apart, the values between them take the
//! larger side's candidate on no evidence, and [`gaps`] names them.
//!
//! A range ends at the last value measured in it and the next begins one
//! above, so a value between two measured ones runs what the larger one
//! runs, and the first and last ranges are open: every value has a
//! candidate.

/// What an op cost under each candidate at each measured value of its var.
#[derive(Debug, Clone)]
pub struct Table {
    /// The measured values, ascending and distinct.
    pub values: Vec<u64>,
    /// `costs[c][i]`: candidate `c` at `values[i]`, infinite where it may not run.
    pub costs: Vec<Vec<f64>>,
}

/// One range of a tiling: candidate `candidate` while the var is in
/// `min..=max`; an end left out is open.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Range {
    pub candidate: usize,
    pub min: Option<u64>,
    pub max: Option<u64>,
    pub cost: f64,
}

/// A tiling, what it costs, and what choosing per value would have.
#[derive(Debug, Clone, PartialEq)]
pub struct Tiling {
    pub ranges: Vec<Range>,
    pub cost: f64,
    pub floor: f64,
}

/// The cheapest candidate over `values[i..j]`, the first of equals.
fn best(t: &Table, i: usize, j: usize) -> Option<(usize, f64)> {
    (0..t.costs.len())
        .map(|c| (c, t.costs[c][i..j].iter().sum::<f64>()))
        .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)))
}

/// The tiling of fewest ranges that costs at most `1 + noise` times what
/// choosing per value does, and the cheapest of that many. `None` when
/// nothing was measured or some value has no candidate that may run.
pub fn pick(t: &Table, noise: f64) -> Option<Tiling> {
    let n = t.values.len();
    if n == 0 || t.costs.is_empty() {
        return None;
    }
    // cheapest[k][j]: the first j values in k + 1 ranges, and where the last range starts.
    let one: Vec<(f64, usize)> = (0..=n).map(|j| (best(t, 0, j).map_or(0., |b| b.1), 0)).collect();
    let cheapest = (1..n).fold(vec![one], |mut rows, k| {
        let prev = &rows[k - 1];
        let row = (0..=n)
            .map(|j| {
                (k..j)
                    .map(|i| (prev[i].0 + best(t, i, j).unwrap().1, i))
                    .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)))
                    .unwrap_or((f64::INFINITY, 0))
            })
            .collect();
        rows.push(row);
        rows
    });
    let floor = cheapest[n - 1][n].0;
    if !floor.is_finite() {
        return None;
    }
    let k = (0..n).find(|&k| cheapest[k][n].0 <= floor * (1. + noise))?;
    let starts = (0..=k).rev().fold(vec![n], |mut cuts, k| {
        cuts.push(cheapest[k][*cuts.last().unwrap()].1);
        cuts
    });
    let ranges: Vec<Range> = starts
        .windows(2)
        .rev()
        .map(|w| (w[1], w[0]))
        .map(|(i, j)| {
            let (candidate, cost) = best(t, i, j).unwrap();
            Range { candidate, min: (i > 0).then(|| t.values[i - 1] + 1), max: (j < n).then(|| t.values[j - 1]), cost }
        })
        .collect();
    Some(Tiling { ranges, cost: cheapest[k][n].0, floor })
}

/// The boundaries of `tiling` drawn across more than a doubling of the
/// var: the measured value each side of one, `(below, above)`.
pub fn gaps(t: &Table, tiling: &Tiling) -> Vec<(u64, u64)> {
    tiling
        .ranges
        .iter()
        .filter_map(|r| r.max)
        .filter_map(|below| Some((below, *t.values.iter().find(|&&v| v > below)?)))
        .filter(|&(below, above)| above > 2 * below)
        .collect()
}
