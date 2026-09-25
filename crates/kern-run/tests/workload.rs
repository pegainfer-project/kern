//! What a sweep expands to, and what a manifest makes of it.
//!
//! The fixture is `examples/qwen3-4b.json`: a `decode` for one sequence of
//! one row, a `decode_batch` for up to 256 of one row, and a `prefill` for
//! one sequence of rows as fed, up to 2,048 rows in all.

use kern_manifest::{Protocol, Verified};
use kern_run::bench::workload::{Plan, Workload};

fn manifest() -> Verified {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples/qwen3-4b.json");
    Verified::from_json(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn plan(sweeps: &str) -> Plan {
    let m = manifest();
    let w = Workload::parse(&format!("samples = 12\nseed = 1\n{sweeps}")).unwrap();
    let unit = kern_pool::page_unit(&m);
    Plan::check(&w, &Protocol::check(&m).unwrap(), unit as usize, kern_pool::row_tokens(&m, unit)).unwrap()
}

fn ids(p: &Plan) -> Vec<&str> {
    p.scenarios.iter().map(|s| s.id.as_str()).collect()
}

#[test]
fn a_sweep_is_the_cross_product_of_its_axes() {
    let w = Workload::parse("samples = 12\nseed = 1\n[[sweep]]\ngroups = [1, 2]\nrows = [1, 4]\ncontext = [8, 16]\n")
        .unwrap();
    let labels: Vec<String> = w.shapes().iter().map(|s| s.label()).collect();
    assert_eq!(labels.len(), 8);
    assert_eq!(labels[0], "g1-r1-kv8");
    assert_eq!(labels[7], "g2-r4-kv16");
}

#[test]
fn a_shape_asked_for_twice_is_one_shape_with_both_weights() {
    let w = Workload::parse(
        "samples = 12\nseed = 1\n\
         [[sweep]]\nrows = [1]\ncontext = [8, 16]\n\
         [[sweep]]\nrows = [1]\ncontext = [16, 32]\n",
    )
    .unwrap();
    let shapes: Vec<(String, u64)> = w.shapes().iter().map(|s| (s.label(), s.weight)).collect();
    assert_eq!(shapes, [("g1-r1-kv8".into(), 1), ("g1-r1-kv16".into(), 2), ("g1-r1-kv32".into(), 1)]);
}

#[test]
fn a_weight_counts_calls_of_every_shape_in_its_sweep() {
    let w = Workload::parse(
        "samples = 12\nseed = 1\n\
         [[sweep]]\ngroups = [1, 2]\nrows = [1]\ncontext = [8]\nweight = 40\n\
         [[sweep]]\ngroups = [2]\nrows = [1]\ncontext = [8]\nweight = 2\n",
    )
    .unwrap();
    let shapes: Vec<(String, u64)> = w.shapes().iter().map(|s| (s.label(), s.weight)).collect();
    assert_eq!(shapes, [("g1-r1-kv8".into(), 40), ("g2-r1-kv8".into(), 42)]);
    assert!(Workload::parse("samples = 12\nseed = 1\n[[sweep]]\nrows = [1]\ncontext = [8]\nweight = 0\n").is_err());
}

#[test]
fn a_per_sequence_context_only_pairs_with_its_own_group_count() {
    let w =
        Workload::parse("samples = 12\nseed = 1\n[[sweep]]\ngroups = [2, 3]\nrows = [1]\ncontext = [[8, 16, 32]]\n")
            .unwrap();
    let labels: Vec<String> = w.shapes().iter().map(|s| s.label()).collect();
    assert_eq!(labels, ["g3-r1-kv8+16+32"]);
}

#[test]
fn a_shape_two_programs_take_becomes_two_scenarios() {
    let p = plan("[[sweep]]\nrows = [1]\ncontext = [128]\n");
    assert_eq!(ids(&p), ["decode-g1-r1-kv128", "prefill-g1-r1-kv128"]);
}

#[test]
fn the_program_follows_from_the_shape_and_nothing_else() {
    let p = plan("[[sweep]]\ngroups = [4]\nrows = [1]\ncontext = [512]\n");
    assert_eq!(ids(&p), ["decode_batch-g4-r1-kv512"]);
    let p = plan("[[sweep]]\nrows = [2048]\ncontext = [0]\n");
    assert_eq!(ids(&p), ["prefill-g1-r2048-kv0"]);
}

#[test]
fn a_shape_no_program_takes_is_dropped_with_its_reason() {
    let p = plan("[[sweep]]\ngroups = [4]\nrows = [1, 2048]\ncontext = [0]\n");
    assert_eq!(ids(&p), ["decode_batch-g4-r1-kv0"]);
    assert_eq!(p.dropped.len(), 1);
    assert_eq!(p.dropped[0].shape, "g4-r2048-kv0");
    assert!(p.dropped[0].why.contains("8192 rows exceeds the manifest's 2048"), "{}", p.dropped[0].why);
}

#[test]
fn a_sequence_longer_than_a_page_table_row_is_dropped() {
    let m = manifest();
    let row = kern_pool::row_tokens(&m, kern_pool::page_unit(&m)).unwrap();
    let p = plan(&format!("[[sweep]]\nrows = [1]\ncontext = [{}, {}]\n", row - 1, row));
    assert_eq!(ids(&p), [format!("decode-g1-r1-kv{}", row - 1), format!("prefill-g1-r1-kv{}", row - 1)]);
    assert_eq!(
        (p.dropped[0].shape.clone(), p.dropped[0].why.contains("page-table row")),
        (format!("g1-r1-kv{row}"), true)
    );
}

#[test]
fn capacity_is_the_largest_scenarios_reach() {
    let p = plan("[[sweep]]\ngroups = [1, 8]\nrows = [1]\ncontext = [128, 2048]\n");
    // Eight sequences of 2,048 + 1, rounded to 16-token pages, plus one page.
    assert_eq!((p.tokens, p.seqs), (8 * 2064 + 16, 8));
}

#[test]
fn a_workload_with_no_sweep_asks_only_for_the_anchors() {
    let p = plan("");
    assert_eq!((p.scenarios.len(), p.dropped.len()), (0, 0));
}

#[test]
fn a_malformed_workload_is_an_error_not_a_dropped_shape() {
    assert!(Workload::parse("samples = 4\nseed = 1\n").is_err());
    assert!(Workload::parse("samples = 12\nseed = 1\n[[sweep]]\nrows = [0]\ncontext = [1]\n").is_err());
    assert!(Workload::parse("samples = 12\nseed = 1\n[[sweep]]\nrows = [1]\n").is_err());
    let unpaired = "samples = 12\nseed = 1\n[[sweep]]\ngroups = [4]\nrows = [1]\ncontext = [[128, 512]]\n";
    assert!(Workload::parse(unpaired).unwrap_err().to_string().contains("pairs with no group count"));
}
