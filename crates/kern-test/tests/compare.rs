//! The comparison and perturbation primitives over enumerated inputs: every
//! pair of interesting values per dtype, every perturbation mode over a
//! few seeds, and sparse byte diffs against a patch-back reference.

use kern_manifest::types::DType;
use kern_manifest::values::{from_f64, to_f64};
use kern_test::compare::{compare, diff_runs, logit_row, perturb, Cmp, MODES, TOP};
use kern_test::workload::Rng;

const FLOATS: [DType; 4] = [DType::Bf16, DType::F16, DType::F32, DType::Fp8E4m3];

/// The values worth pairing: both zeros, neighbours across a binade, the
/// two non-finite kinds.
fn interesting(dt: DType) -> Vec<Vec<u8>> {
    let base = [0.0, -0.0, 1.0, -1.0, 2.0, 0.5, 1.5, 3.0, 1e-3, 100.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
    let w = dt.bytes() as usize;
    let mut v: Vec<Vec<u8>> = base.iter().map(|x| from_f64(dt, &[*x])).collect();
    // the next representable value after 1.0, whatever the dtype's mantissa
    let mut one = from_f64(dt, &[1.0]);
    one[0] = one[0].wrapping_add(1);
    v.push(one);
    v.retain(|b| b.len() == w);
    v.dedup();
    v
}

fn cat(elems: &[&[u8]]) -> Vec<u8> {
    elems.concat()
}

#[test]
fn a_comparison_is_symmetric_and_counts_each_kind_of_difference_once() {
    for dt in FLOATS {
        let vs = interesting(dt);
        for a in &vs {
            for b in &vs {
                let (ab, ba) = (compare(dt, a, b), compare(dt, b, a));
                let (fa, fb) = (to_f64(dt, a)[0], to_f64(dt, b)[0]);
                assert_eq!(
                    (ab.n_diff, ab.nan_only_one_side, ab.signed_zero, ab.max_ulp, ab.max_abs),
                    (ba.n_diff, ba.nan_only_one_side, ba.signed_zero, ba.max_ulp, ba.max_abs),
                    "{dt:?} {fa} vs {fb}"
                );
                assert_eq!(ab.n, 1);
                assert_eq!(ab.n_diff, (a != b) as usize, "{dt:?} {fa} vs {fb}");
                assert_eq!(
                    ab.nan_only_one_side,
                    (a != b && fa.is_nan() != fb.is_nan()) as usize,
                    "{dt:?} {fa} vs {fb}"
                );
                assert_eq!(ab.signed_zero, (a != b && fa == fb) as usize, "{dt:?} {fa} vs {fb}");
                // ulps and abs are measured only on the differing finite-or-inf pairs
                let measured = a != b && fa.is_nan() == fb.is_nan() && !fa.is_nan() && fa != fb;
                assert_eq!(ab.max_ulp.is_some(), measured, "{dt:?} {fa} vs {fb}: {ab:?}");
                assert_eq!(ab.max_abs > 0.0, measured, "{dt:?} {fa} vs {fb}: {ab:?}");
                assert_eq!((ab.identical(), ab.value_identical()), (a == b, a == b || fa == fb), "{dt:?} {fa} vs {fb}");
            }
        }
    }
}

#[test]
fn a_tensor_comparison_is_the_sum_of_its_elements() {
    for dt in FLOATS {
        let vs = interesting(dt);
        let refs: Vec<&[u8]> = vs.iter().map(Vec::as_slice).collect();
        let a = cat(&refs);
        let mut rot = refs.clone();
        rot.rotate_left(3);
        let b = cat(&rot);
        let whole = compare(dt, &a, &b);
        let parts: Vec<Cmp> = refs.iter().zip(&rot).map(|(x, y)| compare(dt, x, y)).collect();
        let sum = |f: fn(&Cmp) -> usize| parts.iter().map(f).sum::<usize>();
        assert_eq!(
            (whole.n, whole.n_diff, whole.nan_only_one_side, whole.signed_zero),
            (refs.len(), sum(|c| c.n_diff), sum(|c| c.nan_only_one_side), sum(|c| c.signed_zero)),
            "{dt:?}"
        );
        assert_eq!(whole.max_ulp, parts.iter().filter_map(|c| c.max_ulp).max(), "{dt:?}");
        assert_eq!(whole.max_abs, parts.iter().map(|c| c.max_abs).fold(0.0, f64::max), "{dt:?}");
    }
}

#[test]
fn severity_orders_identical_below_signed_zeros_below_real_differences() {
    let dt = DType::F32;
    let (z, nz, one, two) = (from_f64(dt, &[0.0]), from_f64(dt, &[-0.0]), from_f64(dt, &[1.0]), from_f64(dt, &[2.0]));
    let same = compare(dt, &one, &one);
    let zeros = compare(dt, &z, &nz);
    let close = compare(dt, &cat(&[&one, &one]), &cat(&[&one, &two]));
    let far = compare(dt, &cat(&[&one, &one]), &cat(&[&two, &two]));
    assert!(same.severity() < zeros.severity() || same.severity() == zeros.severity());
    assert!(zeros.severity() < close.severity() && close.severity() < far.severity());
    // an integer dtype has differing elements but no ulps
    let ints = compare(DType::I32, &7i32.to_le_bytes(), &9i32.to_le_bytes());
    assert_eq!((ints.n_diff, ints.max_ulp, ints.max_abs), (1, None, 2.0));
}

fn row(dt: DType, v: &[f64]) -> Vec<u8> {
    from_f64(dt, v)
}

#[test]
fn a_logit_row_knows_its_argmax_margin_kl_and_top_overlap() {
    let dt = DType::F32;
    let a = [0.5, 2.0, -1.0, 1.75];
    let ra = row(dt, &a);
    let same = logit_row("r".into(), dt, &ra, &ra);
    assert_eq!((same.argmax_a, same.argmax_b, same.flip(), same.rank_in_b), (1, 1, false, 1));
    assert_eq!((same.kl, same.margin_a, same.top), (0.0, 0.25, 4));
    // shifting every logit by a constant leaves argmax, KL and the top set alone
    let shifted = logit_row("r".into(), dt, &ra, &row(dt, &a.map(|x| x + 3.0)));
    assert_eq!((shifted.flip(), shifted.top, shifted.cmp.n_diff), (false, 4, 4));
    assert!(shifted.kl.abs() < 1e-9, "{}", shifted.kl);
    // a near tie broken the other way: a flip, A's token now second in B,
    // almost no mass moved
    let t = [0.5, 2.0, -1.0, 1.99];
    let tie = logit_row("r".into(), dt, &row(dt, &t), &row(dt, &[0.5, 1.99, -1.0, 2.0]));
    assert_eq!((tie.flip(), tie.argmax_b, tie.rank_in_b, tie.top), (true, 3, 2, 4));
    assert!(tie.kl > 0.0 && tie.kl < 1e-3, "{}", tie.kl);
    // a confident token displaced: the same flip, far more mass moved
    let wide = logit_row("r".into(), dt, &row(dt, &[0.0, 5.0, 0.0, 1.0]), &row(dt, &[0.0, 1.0, 0.0, 5.0]));
    assert_eq!((wide.flip(), wide.argmax_b, wide.rank_in_b), (true, 3, 2));
    assert!(wide.kl > 1.0, "{}", wide.kl);
    // the top set is A's TOP most likely tokens found among B's
    let n = TOP + 5;
    let va: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let mut vb = va.clone();
    vb.swap(n - 1, 0); // A's best becomes B's worst, A's worst B's best
    let moved = logit_row("r".into(), dt, &row(dt, &va), &row(dt, &vb));
    assert_eq!((moved.top, moved.rank_in_b, moved.argmax_b), (TOP - 1, n, 0));
    // a NaN on one side is an unbounded move
    let nan = logit_row("r".into(), dt, &ra, &row(dt, &[0.5, f64::NAN, -1.0, 1.75]));
    assert!(nan.kl.is_infinite() && nan.cmp.nan_only_one_side == 1);
}

#[test]
fn every_perturbation_mode_keeps_shape_range_and_its_own_promise() {
    let x: Vec<f64> = (0..24).map(|i| ((i * 7) % 11) as f64 - 5.0).collect();
    for dt in FLOATS {
        let max = match dt {
            DType::F16 => 65504.0,
            DType::Fp8E4m3 => 448.0,
            _ => 3.0e38,
        };
        for mode in 0..MODES.len() {
            for seed in 0..8u64 {
                let y = perturb(&mut Rng(seed), mode, &x, 4, dt);
                assert_eq!(y.len(), x.len(), "{} seed {seed}", MODES[mode]);
                assert!(y.iter().all(|v| !v.is_finite() || v.abs() <= max), "{} seed {seed}: {y:?}", MODES[mode]);
                assert_eq!(
                    perturb(&mut Rng(seed), mode, &x, 4, dt),
                    y,
                    "{} is not a function of the seed",
                    MODES[mode]
                );
                let sorted = |v: &[f64]| {
                    let mut s = v.to_vec();
                    s.sort_by(f64::total_cmp);
                    s
                };
                match MODES[mode] {
                    "jitter" => assert!(x.iter().zip(&y).all(|(a, b)| (a - b).abs() <= a.abs() * 0.1), "{y:?}"),
                    "scale" => {
                        let f = y[1] / x[1];
                        assert!([0.25, 0.5, 2.0, 4.0].contains(&f), "{f}");
                        assert!(x.iter().zip(&y).all(|(a, b)| a * f == *b), "{y:?}");
                    }
                    "shuffle" => {
                        assert_eq!(sorted(&x), sorted(&y));
                        let rows: Vec<&[f64]> = y.chunks(4).collect();
                        assert!(rows.iter().all(|r| x.chunks(4).any(|xr| xr == *r)), "rows torn: {y:?}");
                    }
                    "resample" => assert!(y.iter().all(|v| x.contains(v)), "{y:?}"),
                    "outliers" => assert!(x.iter().zip(&y).all(|(a, b)| *b == *a || *b == a * 16.0), "{y:?}"),
                    _ => assert_eq!(MODES[mode], "noise"),
                }
            }
        }
    }
    assert_eq!(perturb(&mut Rng(1), 0, &[], 1, DType::F32), Vec::<f64>::new());
}

#[test]
fn diff_runs_patch_post_back_into_pre_and_bridge_only_short_gaps() {
    let mut rng = Rng(7);
    for n in [0usize, 1, 63, 64, 65, 200, 1000] {
        for density in [0u64, 1, 3, 50] {
            let pre: Vec<u8> = (0..n).map(|_| rng.draw() as u8).collect();
            let post: Vec<u8> =
                pre.iter().map(|&b| if density > 0 && rng.below(100) < density { b ^ 0x5a } else { b }).collect();
            let runs = diff_runs(&pre, &post);
            let mut patched = post.clone();
            for (at, bytes) in &runs {
                patched[*at..at + bytes.len()].copy_from_slice(bytes);
            }
            assert_eq!(patched, pre, "n {n} density {density}");
            for w in runs.windows(2) {
                let (end, next) = (w[0].0 + w[0].1.len(), w[1].0);
                assert!(next >= end + 64, "runs {end}..{next} should have been one");
            }
            for (at, bytes) in &runs {
                assert!(
                    pre[*at] != post[*at] && pre[at + bytes.len() - 1] != post[at + bytes.len() - 1],
                    "a run has a matching end"
                );
            }
            assert_eq!(runs.is_empty(), pre == post);
        }
    }
}
