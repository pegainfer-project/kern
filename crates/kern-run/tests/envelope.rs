//! A manifest's envelope as a workload.
//!
//! The fixture is the synthetic paged manifest of `workload.rs`: a `decode`
//! for up to 256 sequences of one row, and a `prefill` for one sequence of
//! rows as fed, up to 2,048. Its pages hold 16 tokens and a sequence 40,960.

use kern_manifest::{Protocol, Verified};
use kern_run::bench::envelope::{sweep, Ask, DEEP, EVERY};
use kern_run::bench::workload::{Plan, Shape, Workload};

fn manifest() -> Verified {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../kern-manifest/tests/fixtures/paged.json");
    Verified::from_json(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn plan(ask: &Ask) -> Plan {
    let m = manifest();
    let p = Protocol::check(&m).unwrap();
    let unit = kern_pool::page_unit(&m);
    let row = kern_pool::row_tokens(&m, unit);
    Plan::check(&Workload::parse(&sweep(&p, unit, row.unwrap(), ask)).unwrap(), &p, unit as usize, row).unwrap()
}

fn ask(chunk: u64, groups: &[u64], tokens: u64) -> Ask {
    Ask { chunk, groups: groups.to_vec(), tokens }
}

fn shapes(p: &Plan) -> Vec<&Shape> {
    p.scenarios.iter().map(|s| &s.shape).collect()
}

/// Every chunk as (rows, context).
fn chunks(p: &Plan) -> Vec<(usize, usize)> {
    shapes(p).into_iter().filter(|s| s.chunk).map(|s| (s.rows, s.context[0])).collect()
}

/// Every step as (sequences, context of each).
fn steps(p: &Plan) -> Vec<(usize, usize)> {
    shapes(p).into_iter().filter(|s| !s.chunk).map(|s| (s.groups, s.context[0])).collect()
}

#[test]
fn a_step_is_swept_at_every_size_over_one_page_and_its_deepest_context() {
    let p = plan(&ask(256, &[1, 8, 256], 81_920));
    // One sequence is bound by the page-table row, eight and 256 by their share of the state.
    assert_eq!(steps(&p), [(1, 16), (1, 40_944), (8, 16), (8, 10_224), (256, 16), (256, 304)]);
}

#[test]
fn chunks_are_dense_over_an_empty_sequence_and_sparse_over_the_deepest() {
    let p = plan(&ask(256, &[1], 81_920));
    let (empty, deep): (Vec<_>, Vec<_>) = chunks(&p).into_iter().partition(|&(_, context)| context == 0);
    assert_eq!(empty.iter().map(|c| c.0).collect::<Vec<_>>(), [1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256]);
    assert_eq!(deep, [1, 2, 4, 8, 16, 32, 64, 128, 256].map(|rows| (rows, 40_704)));
}

#[test]
fn the_largest_chunk_is_measured_whatever_the_spacing() {
    let p = plan(&ask(200, &[1], 81_920));
    let rows = |context: usize| chunks(&p).into_iter().filter(|c| (c.1 == 0) == (context == 0)).map(|c| c.0).max();
    assert_eq!((rows(0), rows(1)), (Some(200), Some(200)));
}

#[test]
fn a_size_the_manifest_has_no_program_for_is_dropped_and_named() {
    let p = plan(&ask(4096, &[1, 512], 1 << 20));
    assert_eq!(steps(&p), [(1, 16), (1, 40_944)]);
    let dropped: Vec<&str> = p.dropped.iter().map(|d| d.shape.as_str()).collect();
    assert!(dropped.contains(&"g512-r1-kv16") && dropped.contains(&"g1-c4096-kv0"), "{dropped:?}");
}

/// Whatever is asked, a scenario fits the state it was given and the row
/// its sequences sit in, no chunk is past the largest, and both spacings
/// hold.
#[test]
fn every_envelope_stays_inside_what_was_asked() {
    for chunk in [1, 31, 32, 33, 100, 128, 500, 2048] {
        for groups in [&[1][..], &[1, 2, 4], &[3, 7, 256]] {
            for tokens in [4096, 40_000, 81_920, 1 << 20] {
                let a = ask(chunk, groups, tokens);
                let p = plan(&a);
                let what = format!("{a:?}");
                assert!(p.dropped.is_empty(), "{what}: {:?}", p.dropped);
                for s in shapes(&p) {
                    let held: usize = s.context.iter().map(|n| n + s.rows).sum();
                    let shallow = s.context.iter().all(|&n| n <= 16);
                    assert!(shallow || held as u64 <= tokens, "{what}: {} holds {held}", s.label());
                    assert!(s.context.iter().all(|n| n + s.rows <= 40_960), "{what}: {}", s.label());
                    assert!(!s.chunk || s.rows as u64 <= chunk, "{what}: {}", s.label());
                }
                let spaced = |deep: bool, every: u64| {
                    chunks(&p).into_iter().filter(|c| (c.1 > 0) == deep).all(|(rows, _)| {
                        let rows = rows as u64;
                        rows == chunk || rows.is_multiple_of(every) || (rows < every && rows.is_power_of_two())
                    })
                };
                assert!(spaced(false, EVERY) && spaced(true, DEEP), "{what}");
                assert_eq!(
                    steps(&p).iter().map(|s| s.0 as u64).collect::<std::collections::BTreeSet<_>>().len(),
                    groups.len()
                );
            }
        }
    }
}
