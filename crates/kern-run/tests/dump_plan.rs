//! Dump boundaries are selected before a device is opened. Images preserve
//! allocations rather than guessing the bytes a pointer argument may address.

use std::collections::BTreeMap;

use kern_manifest::types::{DType, Dim};
use kern_manifest::{verify, Manifest, Verified};
use kern_run::dump::plan::{Phase, Plan, Selection, StorageKind};

fn fixture() -> Manifest {
    Manifest::from_json(
        r#"{
          "schema_version": 7, "model": "fixture",
          "vars": {
            "rows": {"max": 4, "axis": "rows"},
            "groups": {"max": 2, "axis": "groups"},
            "columns": {"max": 3}
          },
          "states": {"memory": {"bytes": 64}},
          "buffers": {
            "ids": {"kind": "input", "dtype": "i64", "shape": ["rows"], "fill": "token"},
            "weight": {"kind": "weight", "dtype": "bf16", "shape": [3], "bind": [{"tensor": "w"}]},
            "memory": {"kind": "workspace", "dtype": "bf16", "shape": ["rows", "columns"]},
            "other": {"kind": "workspace", "dtype": "bf16", "shape": [4, 3]},
            "out": {"kind": "output", "dtype": "i64", "shape": ["groups"], "fill": "tokens"}
          },
          "modules": {},
          "ops": {
            "prepare": {
              "params": ["in buffer<bf16>", "out buffer<bf16>", "out buffer<bf16>", "inout state", "in buffer<i64>"],
              "impl": {"launches": [{"entry": "extern:fixture"}]}
            },
            "alias": {
              "params": ["in buffer<bf16>", "out buffer<bf16>", "inout buffer<bf16>", "in state", "out buffer<i64>"],
              "impl": {"launches": [{"entry": "extern:fixture"}]}
            },
            "state": {"params": ["inout state"], "impl": {"launches": [{"entry": "extern:fixture"}]}}
          },
          "programs": {
            "step": {"batch": {"groups": 2, "rows": 1}, "graph": true, "calls": [
              {"label": "prepare", "op": "prepare", "args": [
                {"buf": "weight"}, {"buf": "memory"}, {"buf": "other"}, {"state": "memory"}, {"buf": "ids"}
              ]},
              {"label": "repeated", "op": "alias", "args": [
                {"buf": "memory", "offset": 2}, {"buf": "memory", "offset": 4},
                {"buf": "other"}, {"state": "memory", "offset": 5}, {"buf": "out"}
              ]},
              {"label": "repeated", "op": "state", "args": [{"state": "memory"}]}
            ]}
          }
        }"#,
    )
    .unwrap()
}

fn checked(m: Manifest) -> Verified {
    verify(m).unwrap_or_else(|e| panic!("fixture does not verify: {e}"))
}

fn selection(indices: Vec<usize>) -> Selection {
    Selection { program: "step".into(), invocation: 0, labels: vec![], indices, phase: Phase::Both }
}

fn allocations() -> BTreeMap<String, u64> {
    BTreeMap::from([
        ("buffer:ids".into(), 32),
        ("buffer:weight".into(), 6),
        ("buffer:memory".into(), 24),
        ("buffer:other".into(), 24),
        ("buffer:out".into(), 16),
        ("state:memory".into(), 64),
    ])
}

fn vars(rows: u64, groups: u64, columns: u64) -> BTreeMap<String, u64> {
    BTreeMap::from([("rows".into(), rows), ("groups".into(), groups), ("columns".into(), columns)])
}

#[test]
fn exact_labels_cover_repeated_calls_and_indices_are_a_set() {
    let m = checked(fixture());
    let mut s = selection(vec![]);
    s.labels = vec!["repeated".into(), "prepare".into(), "repeated".into()];
    assert_eq!(Plan::check(&m, s).unwrap().calls, [0, 1, 2]);

    // Enumerate every multiset of three valid indices; the reference is the
    // set of program positions present in the request, in execution order.
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                let asked = vec![a, b, c];
                let expected: Vec<usize> = (0..3).filter(|i| asked.contains(i)).collect();
                assert_eq!(Plan::check(&m, selection(asked)).unwrap().calls, expected);
            }
        }
    }
    let mut s = selection(vec![]);
    s.labels = vec!["peated".into()];
    assert!(Plan::check(&m, s).unwrap_err().to_string().contains("labels match exactly"));
}

#[test]
fn whole_images_include_inputs_weights_outputs_and_distinct_name_spaces() {
    let m = checked(fixture());
    let p = Plan::check(&m, selection(vec![0])).unwrap();
    let images = p.storages(&m, 0, &vars(1, 1, 1), &allocations()).unwrap();
    let names: Vec<(&str, StorageKind, u64)> = images.iter().map(|s| (s.name.as_str(), s.kind, s.bytes)).collect();
    assert_eq!(
        names,
        [
            ("weight", StorageKind::Buffer, 6),
            ("memory", StorageKind::Buffer, 24),
            ("other", StorageKind::Buffer, 24),
            ("memory", StorageKind::State, 64),
            ("ids", StorageKind::Buffer, 32),
        ]
    );
    assert_eq!(
        (&images[1].declared_shape, &images[1].shape, images[1].dtype),
        (&Some(vec![Dim::Var("rows".into()), Dim::Var("columns".into())]), &Some(vec![1, 1]), Some(DType::Bf16))
    );
    assert_eq!((images[3].dtype, &images[3].declared_shape, &images[3].shape), (None, &None, &None));
    assert_eq!(images[0].arguments[0].param.to_string(), "in buffer<bf16>");
    assert_eq!(images[1].arguments[0].param.to_string(), "out buffer<bf16>");
}

#[test]
fn aliases_keep_every_offset_without_shrinking_the_allocation() {
    let m = checked(fixture());
    let p = Plan::check(&m, selection(vec![1, 2])).unwrap();
    // Even where the declared shape at this invocation is smaller than an
    // argument's static offset, its complete backing allocation is preserved.
    for rows in 1..=4 {
        for groups in 1..=2 {
            for columns in 1..=3 {
                let images = p.storages(&m, 1, &vars(rows, groups, columns), &allocations()).unwrap();
                let refs: Vec<(usize, u64)> = images[0].arguments.iter().map(|a| (a.index, a.offset)).collect();
                assert_eq!((images.len(), images[0].bytes, refs), (4, 24, vec![(0, 2), (1, 4)]));
                assert_eq!((&images[0].shape, images[2].arguments[0].offset), (&Some(vec![rows, columns]), 5));
            }
        }
    }
    let images = p.storages(&m, 2, &vars(1, 1, 1), &allocations()).unwrap();
    assert_eq!((images.len(), images[0].kind, images[0].bytes), (1, StorageKind::State, 64));
}
