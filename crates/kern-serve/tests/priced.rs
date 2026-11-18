//! What a call is priced at off a bench report, and a rank that takes
//! that long.
//!
//! The manifest is kern-manifest's `plain.json`: one rank, a `decode` for
//! up to 4 sequences of one row, a `prefill` for one sequence of up to 8
//! rows, 16-token pages, three a sequence.

use std::time::Instant;

use kern_manifest::protocol::Sampling;
use kern_manifest::types::Manifest;
use kern_manifest::{verify, Protocol};
use kern_serve::pd::Run;
use kern_serve::priced::{pool, Call, Priced, Prices};
use kern_serve::ranks::{Cell, Groups, Ranks};

fn manifest() -> Manifest {
    Manifest::from_json(include_str!("../../kern-manifest/tests/fixtures/plain.json")).unwrap()
}

fn protocol() -> Protocol {
    Protocol::check(&verify(manifest()).unwrap()).unwrap()
}

/// A report of `(program, sequences, rows, chunk, context of each, µs)`.
fn report(scenarios: &[(&str, u64, u64, bool, u64, f64)]) -> String {
    let scenarios: Vec<_> = scenarios
        .iter()
        .map(|&(program, groups, rows, chunk, context, us)| {
            serde_json::json!({
                "scenario": {"program": program, "groups": groups, "rows": rows, "chunk": chunk, "context": vec![context; groups as usize]},
                "rank": 0,
                "graph": {"stats": {"p50": us}},
            })
        })
        .collect();
    serde_json::json!({"manifest_sha256": "abc", "scenarios": scenarios}).to_string()
}

/// Steps of 1, 2 and 4 sequences over 16 and 40 tokens; chunks of 1, 4
/// and 8 rows over an empty sequence, and of 4 and 8 over 40 tokens.
fn prices() -> Prices {
    Prices::read(&report(&[
        ("decode", 1, 1, false, 16, 1000.0),
        ("decode", 1, 1, false, 40, 1240.0),
        ("decode", 2, 1, false, 16, 1100.0),
        ("decode", 2, 1, false, 40, 1580.0),
        ("decode", 4, 1, false, 16, 1300.0),
        ("prefill", 1, 1, true, 0, 2000.0),
        ("prefill", 1, 4, true, 0, 2600.0),
        ("prefill", 1, 8, true, 0, 5000.0),
        ("prefill", 1, 4, true, 40, 3000.0),
        ("prefill", 1, 8, true, 40, 7000.0),
    ]))
    .unwrap()
}

fn step(context: &[u64]) -> Call {
    Call { program: "decode".into(), groups: context.len() as u64, rows: 1, chunk: false, context: context.to_vec() }
}

fn chunk(rows: u64, context: u64) -> Call {
    Call { program: "prefill".into(), groups: 1, rows, chunk: true, context: vec![context] }
}

fn us(c: &Call) -> f64 {
    prices().price(c).unwrap()
}

#[test]
fn a_measured_call_takes_what_was_measured() {
    assert_eq!((us(&step(&[16])), us(&step(&[40])), us(&step(&[40, 40]))), (1000.0, 1240.0, 1580.0));
    assert_eq!((us(&chunk(4, 0)), us(&chunk(8, 40)), us(&chunk(1, 0))), (2600.0, 7000.0, 2000.0));
}

#[test]
fn a_step_pays_for_every_token_its_sequences_hold() {
    // 10 µs a token past the 16 each was measured over, wherever it sits.
    assert_eq!((us(&step(&[16, 40])), us(&step(&[28, 28])), us(&step(&[0, 16]))), (1340.0, 1340.0, 940.0));
    // A size measured at one depth has no slope to pay.
    assert_eq!(us(&step(&[16, 40, 40, 16])), 1300.0);
}

#[test]
fn a_chunk_is_read_between_its_neighbours_and_sloped_by_the_one_above() {
    // Five rows: a quarter of the way from 4 to 8 over an empty sequence,
    // and the 8-row chunk's 50 µs a token.
    assert_eq!((us(&chunk(5, 0)), us(&chunk(5, 20))), (3200.0, 4200.0));
    // Two rows: between 1 and 4, sloped by the 4-row chunk's 10 µs a token.
    assert_eq!((us(&chunk(2, 0)), us(&chunk(2, 30))), (2200.0, 2500.0));
}

#[test]
fn a_call_nobody_measured_is_an_error_that_names_it() {
    let no = |c: &Call| prices().price(c).unwrap_err().to_string();
    assert!(
        no(&step(&[16; 3])).contains("no `decode` step of 3 sequences (it has [1, 2, 4])"),
        "{}",
        no(&step(&[16; 3]))
    );
    assert!(no(&chunk(9, 0)).contains("`prefill` chunks of 1 to 8 rows, and this one has 9"), "{}", no(&chunk(9, 0)));
    assert!(Prices::read("{}").is_err());
}

/// A context adds linearly: whatever the size, the price at the middle of
/// two depths is the middle of their prices.
#[test]
fn every_price_is_linear_in_the_context() {
    let p = prices();
    let calls = (1..=8).map(|rows| chunk(rows, 0)).chain([step(&[0]), step(&[0, 0]), step(&[0; 4])]);
    for c in calls {
        let at = |x: u64| p.price(&Call { context: vec![x; c.context.len()], ..c.clone() }).unwrap();
        for (a, b) in [(0, 40), (8, 24), (16, 48)] {
            assert!((at(a) + at(b) - 2.0 * at((a + b) / 2)).abs() < 1e-6, "{c:?} over {a}..{b}");
        }
    }
}

#[test]
fn a_priced_rank_answers_its_token_once_the_price_has_passed() {
    let m = manifest();
    let mut r = Ranks::new(protocol(), Groups::new(1, 1).unwrap(), vec![pool(&m, 160, 4).unwrap()]).unwrap();
    let mut rank = Priced::new(protocol(), prices(), 77).unwrap();
    let (a, b) = (r.lease(48, |_| Some(0)).unwrap(), r.lease(48, |_| Some(0)).unwrap());

    // A chunk of four rows over an empty sequence: this `prefill` only
    // advances state, so there is nothing to hand back.
    let cells = [Cell { row: &a, ids: vec![5, 6, 7, 8], pos: 0, sampling: Sampling::GREEDY }];
    let mut st = r.stage(&cells, 4, |k| k).unwrap();
    let f = r.protocol().chunk().unwrap().clone();
    let orders = r.orders(Some(st.items(&f)));
    assert_eq!(rank.call(orders[0].item.as_ref().unwrap()).unwrap(), chunk(4, 0));
    let t = Instant::now();
    let answers = rank.run(orders).unwrap();
    assert_eq!((answers[0].tokens.clone(), t.elapsed().as_micros() >= 2600), (vec![], true));

    // A step of two sequences, 16 and 40 tokens deep: the token for each.
    let cells = [
        Cell { row: &a, ids: vec![11], pos: 16, sampling: Sampling::GREEDY },
        Cell { row: &b, ids: vec![12], pos: 40, sampling: Sampling::GREEDY },
    ];
    let mut st = r.stage(&cells, 1, |k| k).unwrap();
    let f = st.forward(r.protocol(), 1).unwrap();
    let orders = r.orders(Some(st.items(&f)));
    assert_eq!(rank.call(orders[0].item.as_ref().unwrap()).unwrap(), step(&[16, 40]));
    let t = Instant::now();
    let answers = rank.run(orders).unwrap();
    assert_eq!((answers[0].tokens.clone(), t.elapsed().as_micros() >= 1340), (vec![77, 77], true));

    // Nothing to run takes no time and answers nothing.
    assert_eq!(rank.run(r.orders(None)).unwrap()[0].tokens, Vec::<i64>::new());
}
