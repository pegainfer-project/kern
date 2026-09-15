//! Comparing what two sides wrote, and perturbing what a span read. Byte
//! slices in, numbers out; the dtype says how to read them.

use kern_manifest::types::DType;
use kern_manifest::values;
use serde::Serialize;

use crate::workload::Rng;

pub fn is_float(dt: DType) -> bool {
    matches!(dt, DType::Bf16 | DType::F16 | DType::F32 | DType::Fp8E4m3)
}

/// One buffer on two sides, element by element.
#[derive(Serialize, Clone, Default, Debug, PartialEq)]
pub struct Cmp {
    pub n: usize,
    pub n_diff: usize,
    pub max_ulp: Option<u64>,
    pub max_abs: f64,
    pub nan_only_one_side: usize,
    /// Bit-different but value-equal: +0 vs -0.
    pub signed_zero: usize,
}

pub fn compare(dt: DType, a: &[u8], b: &[u8]) -> Cmp {
    let w = dt.bytes() as usize;
    let mut c = Cmp { n: a.len() / w, ..Default::default() };
    for (x, y) in a.chunks_exact(w).zip(b.chunks_exact(w)) {
        if x == y {
            continue;
        }
        c.n_diff += 1;
        let (fx, fy) = (values::to_f64(dt, x)[0], values::to_f64(dt, y)[0]);
        if fx.is_nan() != fy.is_nan() {
            c.nan_only_one_side += 1;
            continue;
        }
        if fx == fy {
            c.signed_zero += 1;
            continue;
        }
        c.max_abs = c.max_abs.max((fx - fy).abs());
        if is_float(dt) {
            if let Some(u) = values::ulp_distance(dt, x, y) {
                c.max_ulp = Some(c.max_ulp.unwrap_or(0).max(u));
            }
        }
    }
    c
}

impl Cmp {
    pub fn identical(&self) -> bool {
        self.n_diff == 0
    }
    /// Every difference is a signed zero.
    pub fn value_identical(&self) -> bool {
        self.n_diff == self.signed_zero
    }
    /// Worst first: differing elements, then ulps.
    pub fn severity(&self) -> (usize, u64) {
        (self.n_diff - self.signed_zero, self.max_ulp.unwrap_or(u64::MAX))
    }
}

/// One end-to-end logits comparison: A (lockstep, uninjected) vs B (free
/// run) on one row of a `logits*` buffer after one program run.
#[derive(Clone, Debug)]
pub struct LogitRow {
    pub label: String,
    pub cmp: Cmp,
    pub max_abs: f64,
    /// `max_abs` in ulps of the row's scale (A's max |logit|): the
    /// granularity the row is stored at. Element-wise ulps are meaningless
    /// here — a 1-ulp change of the hidden state moves every logit by
    /// about the same absolute amount, which is thousands of ulps for a
    /// logit near zero, and decides nothing.
    pub scale_ulps: f64,
    pub scale: f64,
    pub argmax_a: usize,
    pub argmax_b: usize,
    /// A's top-1 − top-2: how far the argmax was from flipping on its own.
    pub margin_a: f64,
    pub kl: f64,
}

impl LogitRow {
    pub fn flip(&self) -> bool {
        self.argmax_a != self.argmax_b
    }
    /// The delta could have flipped A's own argmax: not B's doing alone.
    pub fn near_tie(&self) -> bool {
        self.flip() && self.margin_a <= self.max_abs
    }
}

/// Spacing of `dt` at magnitude `x`.
pub fn ulp_at(dt: DType, x: f64) -> f64 {
    let mant = match dt {
        DType::Bf16 => 7,
        DType::F16 => 10,
        DType::F32 => 23,
        DType::Fp8E4m3 => 3,
        _ => 0,
    };
    let e = if x.abs() > 0.0 && x.is_finite() { x.abs().log2().floor() } else { 0.0 };
    2f64.powf(e - mant as f64)
}

pub fn logit_row(label: String, dt: DType, a: &[u8], b: &[u8]) -> LogitRow {
    let (va, vb) = (values::to_f64(dt, a), values::to_f64(dt, b));
    let cmp = compare(dt, a, b);
    // A NaN on one side is an unbounded delta, not one to skip: a row B
    // poisons must never read as "moved 0 ulp".
    let max_abs = if cmp.nan_only_one_side > 0 {
        f64::INFINITY
    } else {
        va.iter().zip(&vb).map(|(x, y)| (x - y).abs()).filter(|d| d.is_finite()).fold(0.0, f64::max)
    };
    let argmax = |v: &[f64]| {
        v.iter().enumerate().fold((0usize, f64::NEG_INFINITY), |m, (i, &x)| if x > m.1 { (i, x) } else { m })
    };
    let (argmax_a, top1) = argmax(&va);
    let top2 = va.iter().enumerate().filter(|(i, _)| *i != argmax_a).map(|(_, &x)| x).fold(f64::NEG_INFINITY, f64::max);
    let (argmax_b, _) = argmax(&vb);
    let lse = |v: &[f64], m: f64| m + v.iter().map(|x| (x - m).exp()).sum::<f64>().ln();
    let (ma_, mb_) = (top1, vb.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    let (la, lb) = (lse(&va, ma_), lse(&vb, mb_));
    let kl = va.iter().zip(&vb).map(|(x, y)| (x - la).exp() * ((x - la) - (y - lb))).sum::<f64>();
    let scale = va.iter().filter(|x| x.is_finite()).fold(0.0f64, |m, x| m.max(x.abs()));
    let scale_ulps = if max_abs == 0.0 {
        0.0
    } else if max_abs.is_finite() {
        max_abs / ulp_at(dt, scale)
    } else {
        f64::INFINITY
    };
    LogitRow {
        label,
        cmp,
        max_abs,
        scale_ulps,
        scale,
        argmax_a,
        argmax_b,
        margin_a: top1 - top2,
        kl: if kl.is_finite() { kl } else { f64::INFINITY },
    }
}

/// The perturbations a fuzz round cycles through, in order.
pub const MODES: [&str; 6] = ["jitter", "noise", "scale", "shuffle", "resample", "outliers"];

/// Perturb a tapped float tensor, staying in the distribution the kernel
/// was built for: `mode` indexes [`MODES`]. `row` is the trailing extent
/// (elements per leading-dim row) so `shuffle` permutes rows, not
/// elements. Finite values stay within the dtype's range.
pub fn perturb(rng: &mut Rng, mode: usize, x: &[f64], row: usize, dt: DType) -> Vec<f64> {
    let max = match dt {
        DType::Bf16 | DType::F32 => 3.0e38,
        DType::F16 => 65504.0,
        DType::Fp8E4m3 => 448.0,
        _ => unreachable!(),
    };
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let mut v: Vec<f64> = match MODES[mode % MODES.len()] {
        // a few low mantissa bits: the same tensor, different rounding paths
        "jitter" => x.iter().map(|&a| a * (1.0 + rng.normal() / 64.0)).collect(),
        // additive noise at 10% of the tensor's own rms
        "noise" => {
            let rms = (x.iter().filter(|a| a.is_finite()).map(|a| a * a).sum::<f64>() / n as f64).sqrt();
            x.iter().map(|&a| a + rng.normal() * 0.1 * rms).collect()
        }
        // dynamic range: the whole tensor ×¼ … ×4
        "scale" => {
            let f = [0.25, 0.5, 2.0, 4.0][rng.below(4) as usize];
            x.iter().map(|&a| a * f).collect()
        }
        // rows in another order: positions change, values don't
        "shuffle" => {
            let row = row.clamp(1, n);
            let rows = n / row;
            let mut perm: Vec<usize> = (0..rows).collect();
            for i in (1..rows).rev() {
                perm.swap(i, rng.below(i as u64 + 1) as usize);
            }
            let mut v = x.to_vec();
            for (dst, &src) in perm.iter().enumerate() {
                v[dst * row..(dst + 1) * row].copy_from_slice(&x[src * row..(src + 1) * row]);
            }
            v
        }
        // bootstrap: the tensor's own marginal, structure destroyed
        "resample" => (0..n).map(|_| x[rng.below(n as u64) as usize]).collect(),
        // 1% of the elements ×16
        _ => x.iter().map(|&a| if rng.below(100) == 0 { a * 16.0 } else { a }).collect(),
    };
    for a in &mut v {
        if a.is_finite() {
            *a = a.clamp(-max, max);
        }
    }
    v
}

/// Runs of `[offset, offset+len)` where `pre` and `post` differ (gaps under
/// 64 bytes are bridged so a sparse update is a few runs, not thousands),
/// with `pre`'s bytes over each run.
pub fn diff_runs(pre: &[u8], post: &[u8]) -> Vec<(usize, Vec<u8>)> {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < pre.len() {
        if pre[i] == post[i] {
            i += 1;
            continue;
        }
        let start = i;
        while i < pre.len() && pre[i] != post[i] {
            i += 1;
        }
        match runs.last_mut() {
            Some((_, end)) if start - *end < 64 => *end = i,
            _ => runs.push((start, i)),
        }
    }
    runs.into_iter().map(|(a, b)| (a, pre[a..b].to_vec())).collect()
}
