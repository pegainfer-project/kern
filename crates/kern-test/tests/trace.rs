//! A recorded reference through the fake side: recorded, written, read
//! back, and every verdict a candidate can get against it.

mod common;

use common::{Fake, Fixture};
use kern_test::trace::{self, against, between, score, Phase, Trace};

fn corpus() -> Trace {
    Trace::corpus(vec![vec![1, 5, 3, 9, 2, 7, 4, 11], vec![6, 6, 2, 13, 8], vec![3, 14]], 3).unwrap()
}

fn quiet() -> impl FnMut(&[String]) {
    |_: &[String]| {}
}

fn recorded(f: &Fixture, t: &mut Trace, name: &str) {
    trace::record(&mut Fake::new(f.manifest()), t, name, &mut quiet()).unwrap();
}

fn judged(f: &Fixture, t: &Trace) -> (i32, String, Vec<String>) {
    let mut lines = Vec::new();
    let r = trace::judge(&mut Fake::new(f.manifest()), t, "ref.parquet", "B", 0.01, &mut |ls: &[String]| {
        lines.extend_from_slice(ls)
    })
    .unwrap();
    (r.code(), r.summary.verdict.summary.clone(), lines)
}

#[test]
fn a_prompt_is_prefilled_to_its_tail_and_stepped_through_it() {
    let t = corpus();
    // 8 tokens: 4 through the chunk, 3 steps, the last is only predicted
    assert_eq!((t.schedule(0).ctx, t.schedule(0).steps), (4, 3));
    // 2 tokens: one in, one predicted, no steps
    assert_eq!((t.schedule(2).ctx, t.schedule(2).steps), (1, 0));
    assert_eq!(t.positions(), 4 + 4 + 1);
    assert_eq!((t.schedule(0).phase(3), t.schedule(0).phase(4)), (Phase::Prefill, Phase::Decode));
}

#[test]
fn a_recording_survives_the_file() {
    let mut t = corpus();
    recorded(&Fixture::default(), &mut t, "a");
    let s = &t.producers["a"];
    assert_eq!(s.len(), t.positions());
    assert_eq!((s[0].prompt, s[0].pos, s[0].ids.len()), (0, 3, common::VOCAB));
    // the kept log probabilities are a distribution's head, most likely first
    assert!(s[0].logprob.windows(2).all(|w| w[0] >= w[1]) && s[0].logprob[0] <= 0.0);
    let path = std::env::temp_dir().join(format!("kern-trace-{}.parquet", std::process::id()));
    t.write(&path).unwrap();
    assert!(Trace::is_parquet(&path));
    let back = Trace::read(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    assert_eq!((back.prompts.clone(), back.decode), (t.prompts.clone(), t.decode));
    let r = &back.producers["a"];
    assert_eq!(
        r.iter().map(|s| (s.prompt, s.pos, s.ids.clone())).collect::<Vec<_>>(),
        s.iter().map(|s| (s.prompt, s.pos, s.ids.clone())).collect::<Vec<_>>()
    );
    // f32 on file
    assert!(r.iter().zip(s).all(|(x, y)| (x.ref_logprob - y.ref_logprob).abs() < 1e-6
        && x.logprob.iter().zip(&y.logprob).all(|(p, q)| (p - q).abs() < 1e-6)));
}

#[test]
fn recording_again_replaces_the_producer_and_keeps_the_others() {
    let mut t = corpus();
    recorded(&Fixture::default(), &mut t, "a");
    recorded(&Fixture::default().scale("scale_round"), &mut t, "b");
    let before = t.producers["a"].clone();
    recorded(&Fixture::default(), &mut t, "a");
    assert_eq!((t.producers.len(), &t.producers["a"]), (2, &before));
    assert_eq!(t.take(1).producers["b"].len(), 4);
}

#[test]
fn the_same_model_is_within_the_limit_on_every_position() {
    let mut t = corpus();
    recorded(&Fixture::default(), &mut t, "a");
    let (code, summary, lines) = judged(&Fixture::default(), &t);
    assert_eq!((code, summary.as_str()), (0, "within KL 1e-2 of `a` on all 9 positions"), "{lines:#?}");
    assert!(lines.iter().any(|l| l.starts_with("prefill") && l.contains("3 rows · 3/3 argmax agree")), "{lines:#?}");
    assert!(lines.iter().any(|l| l.starts_with("decode") && l.contains("6 rows · 6/6 argmax agree")), "{lines:#?}");
}

#[test]
fn a_rounding_change_passes_and_a_wrong_kernel_fails_at_the_first_position() {
    let mut t = corpus();
    recorded(&Fixture::default(), &mut t, "a");
    let (code, _, lines) = judged(&Fixture::default().scale("scale_round"), &t);
    assert_eq!(code, 0, "{lines:#?}");
    let (code, summary, lines) = judged(&Fixture::default().scale("scale_wrong"), &t);
    assert_eq!(code, 1, "{lines:#?}");
    assert!(summary.starts_with("confident argmax flip at prompt 0 pos 4 against a"), "{summary}");
    assert!(summary.ends_with("stopped after 2 positions"), "{summary}");
}

#[test]
fn a_drift_that_moves_mass_but_no_argmax_is_inconclusive() {
    let mut t = corpus();
    recorded(&Fixture::default(), &mut t, "a");
    let (code, summary, _) = judged(&Fixture::default().scale("scale_drift"), &t);
    assert!(code != 0, "{summary}");
    assert!(code == 1 || summary.starts_with("KL up to"), "{summary}");
}

#[test]
fn two_producers_make_a_band_and_the_verdict_reads_against_it() {
    let mut t = corpus();
    recorded(&Fixture::default(), &mut t, "a");
    recorded(&Fixture::default().scale("scale_round"), &mut t, "b");
    let band = trace::band(&t).unwrap();
    assert!(band.kl_p99 > 0.0 && band.kl_p99 < 0.01, "{band:?}");
    let (code, summary, lines) = judged(&Fixture::default(), &t);
    assert_eq!((code, summary.as_str()), (0, "within the band of 2 producers on all 18 positions"), "{lines:#?}");
    assert!(lines.iter().any(|l| l.starts_with("band") && l.contains("between producers")), "{lines:#?}");
    let (code, summary, _) = judged(&Fixture::default().scale("scale_wrong"), &t);
    assert_eq!(code, 1, "{summary}");
}

#[test]
fn producers_that_agree_exactly_still_leave_room_for_the_file_s_f32() {
    let mut t = corpus();
    recorded(&Fixture::default(), &mut t, "a");
    recorded(&Fixture::default().scale("scale_same"), &mut t, "b");
    let band = trace::band(&t).unwrap();
    assert_eq!((band.margin, band.flip_rate, band.kl_p50, band.kl_p99), (0.0, 0.0, trace::KL_FLOOR, trace::KL_FLOOR));
    let (code, summary, _) = judged(&Fixture::default(), &t);
    assert_eq!(code, 0, "{summary}");
    // a rounding change flips a near-tie the producers never flipped: beyond such a band
    let (code, summary, _) = judged(&Fixture::default().scale("scale_round"), &t);
    assert!(code == 0 || summary.contains("beyond the band") || summary.starts_with("confident"), "{summary}");
}

#[test]
fn a_prefill_only_manifest_takes_its_steps_as_one_row_chunks() {
    let mut t = corpus();
    recorded(&Fixture::default().prefill_only(), &mut t, "p");
    assert_eq!(t.producers["p"].len(), t.positions());
    recorded(&Fixture::default(), &mut t, "a");
    // the fake's chunk and step arithmetic agree, so the two producers do
    let (code, _, lines) = judged(&Fixture::default().prefill_only(), &t);
    assert_eq!(code, 0, "{lines:#?}");
}

#[test]
fn a_score_is_the_top_of_a_log_softmax_and_a_comparison_reads_it() {
    let row = [1.0, 3.0, 2.0, 0.0];
    let s = score(0, 0, &row, 2);
    assert_eq!(s.ids, vec![1, 2, 0, 3]);
    let lse = (1f64.exp() + 3f64.exp() + 2f64.exp() + 1.0).ln();
    assert!((s.ref_logprob - (2.0 - lse)).abs() < 1e-12 && (s.logprob[0] - (3.0 - lse)).abs() < 1e-12);
    let same = against(&s, &row);
    assert!((same.kl.abs() < 1e-12) && !same.flip() && same.rank_in_b == 1 && (same.margin_a - 1.0).abs() < 1e-12);
    let flipped = against(&s, &[1.0, 2.0, 3.0, 0.0]);
    assert!(flipped.flip() && flipped.argmax_b == 2 && flipped.rank_in_b == 2 && flipped.kl > 0.1);
    let kept = between(&s, &score(0, 0, &[1.0, 2.0, 3.0, 0.0], 2));
    assert!((kept.kl - flipped.kl).abs() < 1e-9, "{} vs {}", kept.kl, flipped.kl);
}
