//! A candidate replaces an op's implementation and nothing else, and a
//! tiling of candidates runs, at every value of its var, exactly what the
//! piece covering that value runs.

use kern_manifest::candidate::{splice, tile, Candidate, Piece};
use kern_manifest::types::Launch;
use kern_manifest::Verified;
use serde_json::{json, Value};

const MAX: u64 = 12;

fn sha(tag: u8) -> String {
    format!("{tag:02x}").repeat(32)
}

fn manifest() -> Verified {
    let mut m: Value = serde_json::from_str(include_str!("../../../examples/minimal.json")).unwrap();
    m["vars"]["tokens"]["max"] = json!(MAX);
    Verified::from_json(&m.to_string()).unwrap()
}

fn launch(module: &str, entry: &str, when: Option<(Option<u64>, Option<u64>)>) -> Value {
    let mut l = json!({"module": module, "entry": entry, "block": [64, 1, 1], "grid": ["tokens", 1, 1]});
    if let Some((min, max)) = when {
        l["when"] = json!({"var": "tokens"});
        if let Some(v) = min {
            l["when"]["min"] = json!(v);
        }
        if let Some(v) = max {
            l["when"]["max"] = json!(v);
        }
    }
    l
}

fn candidate(module: &str, tag: u8, launches: Vec<Value>) -> Candidate {
    let params = ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "inout state", "i32"];
    Candidate::from_json(
        &json!({
            "modules": {module: {"source": format!("{module}.cubin"), "sha256": sha(tag)}},
            "ops": {"scale": {"params": params, "impl": {"launches": launches}}},
        })
        .to_string(),
    )
    .unwrap()
}

/// The entries of `c`'s `scale` that run at `v`, in order.
fn runs(c: &Candidate, v: u64) -> Vec<String> {
    c.ops["scale"]
        .imp
        .launches
        .iter()
        .filter(|l| l.when().is_none_or(|w| w.holds(v)))
        .map(|l| l.entry().to_string())
        .collect()
}

struct Rand(u64);

impl Rand {
    fn next(&mut self, n: u64) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 33) % n
    }
}

#[test]
fn a_manifests_own_ops_splice_back_to_the_manifest() {
    let m = manifest();
    let own = Candidate::of(&m, ["scale"]);
    assert_eq!((own.ops.len(), own.modules.len()), (1, 1));
    assert_eq!(splice(&m, &own).unwrap().to_json(), m.to_json());
}

#[test]
fn a_splice_swaps_the_implementation_and_leaves_the_calls() {
    let m = manifest();
    let c = candidate("fast", 0xab, vec![launch("fast", "scale_tiled", None)]);
    let out = splice(&m, &c).unwrap();
    let entry = out.ops["scale"].imp.launches[0].entry().to_string();
    let modules: Vec<&String> = out.modules.keys().collect();
    assert_eq!((entry.as_str(), modules), ("scale_tiled", vec![&"fast".to_string()]));
    assert_eq!(
        (serde_json::to_value(&out.programs).unwrap(), serde_json::to_value(&out.buffers).unwrap()),
        (serde_json::to_value(&m.programs).unwrap(), serde_json::to_value(&m.buffers).unwrap())
    );
}

#[test]
fn a_module_name_the_manifest_gives_other_bytes_is_renamed_after_its_hash() {
    let m = manifest();
    let mut c = candidate("toy", 0xcd, vec![launch("toy", "scale_other", None)]);
    let out = splice(&m, &c).unwrap();
    assert_eq!(out.modules.keys().collect::<Vec<_>>(), ["toy@cdcdcdcd"]);
    assert_eq!(out.ops["scale"].imp.launches[0].module(), Some("toy@cdcdcdcd"));

    // The manifest's own module needs no declaration in the candidate.
    c.modules.clear();
    let out = splice(&m, &c).unwrap();
    assert_eq!(out.modules["toy"].sha256, m.modules["toy"].sha256);
}

#[test]
fn what_is_not_another_implementation_of_the_op_is_refused_in_full() {
    let m = manifest();
    let c = Candidate::from_json(
        &json!({
            "ops": {
                "scale": {"params": ["in buffer<bf16>", "out buffer<bf16>"],
                          "impl": {"launches": [launch("gone", "k", None)]}},
                "shift": {"params": ["i32"], "impl": {"launches": [launch("toy", "k", None)]}},
            },
        })
        .to_string(),
    )
    .unwrap();
    let errs = splice(&m, &c).unwrap_err().0;
    let has = |s: &str| errs.iter().any(|e| e.contains(s));
    assert_eq!(
        (errs.len(), has("op `scale`: params"), has("module `gone`"), has("op `shift`: the manifest has no such op")),
        (3, true, true, true)
    );
    assert!(splice(&m, &Candidate::default()).is_err());
    assert!(Candidate::from_json(r#"{"ops": {}, "programs": {}}"#).is_err());
}

#[test]
fn a_tiling_runs_at_every_value_what_the_piece_covering_it_runs() {
    let m = manifest();
    let mut r = Rand(0x9e37_79b9_7f4a_7c15);
    for case in 0..400 {
        let n = 1 + r.next(4) as usize;
        let candidates: Vec<Candidate> = (0..n)
            .map(|i| {
                let launches = (0..1 + r.next(3))
                    .map(|j| {
                        let when = match r.next(3) {
                            0 => None,
                            1 => Some((Some(1 + r.next(MAX)), None)),
                            _ => {
                                let lo = 1 + r.next(MAX);
                                Some((Some(lo), Some(lo + r.next(MAX))))
                            }
                        };
                        launch(&format!("m{i}"), &format!("c{i}l{j}"), when)
                    })
                    .collect();
                candidate(&format!("m{i}"), i as u8 + 1, launches)
            })
            .collect();
        // Cut 1..=MAX into contiguous ranges and hand each to a candidate.
        let mut cuts: Vec<u64> = (2..=MAX).filter(|_| r.next(4) == 0).collect();
        cuts.insert(0, 1);
        let pieces: Vec<(usize, u64, u64)> = cuts
            .iter()
            .enumerate()
            .map(|(k, &lo)| (r.next(n as u64) as usize, lo, cuts.get(k + 1).map_or(MAX, |next| next - 1)))
            .collect();
        let last = pieces.len() - 1;
        let asked: Vec<Piece> = pieces
            .iter()
            .enumerate()
            .map(|(k, &(i, lo, hi))| Piece {
                from: &candidates[i],
                min: (k > 0).then_some(lo),
                max: (k < last).then_some(hi),
            })
            .collect();
        let want = |v: u64| {
            let &(i, _, _) = pieces.iter().find(|&&(_, lo, hi)| (lo..=hi).contains(&v)).unwrap();
            runs(&candidates[i], v)
        };
        match tile("scale", "tokens", &asked) {
            Ok(tiled) => {
                for v in 1..=MAX {
                    assert_eq!(runs(&tiled, v), want(v), "case {case} at tokens = {v}");
                }
                let modules: Vec<&str> = tiled.ops["scale"].imp.launches.iter().filter_map(Launch::module).collect();
                assert!(modules.iter().all(|n| tiled.modules.contains_key(*n)), "case {case}");
                assert_eq!(tiled.modules.len(), modules.iter().collect::<std::collections::BTreeSet<_>>().len());
                // A value at which nothing runs is the candidates' own gap, and
                // a manifest may have one; everything else splices.
                splice(&m, &tiled).unwrap_or_else(|e| panic!("case {case}: {e}"));
            }
            Err(_) => assert!((1..=MAX).all(|v| want(v).is_empty()), "case {case} refused a tiling that runs"),
        }
    }
}

#[test]
fn a_piece_conditioned_on_another_var_cannot_be_narrowed() {
    let mut c = candidate("m", 1, vec![launch("m", "k", None)]);
    let Launch::Kernel(k) = &mut c.ops.get_mut("scale").unwrap().imp.launches[0] else { unreachable!() };
    k.when = Some(serde_json::from_value(json!({"var": "seqs", "max": 2})).unwrap());
    let e = tile("scale", "tokens", &[Piece { from: &c, min: None, max: None }]).unwrap_err();
    assert!(e.0[0].contains("runs in a range of `seqs`, not of `tokens`"), "{e}");
}

#[test]
fn pieces_keep_their_own_scratch_and_candidates_join_by_op() {
    let with_scratch = |tag: u8, cols: u64| {
        let mut c = candidate(&format!("m{tag}"), tag, vec![]);
        c.ops.get_mut("scale").unwrap().imp = serde_json::from_value(json!({
            "scratch": {"tmp": {"dtype": "f32", "shape": ["tokens", cols]}},
            "launches": [{"module": format!("m{tag}"), "entry": format!("k{tag}"), "block": [64, 1, 1],
                "grid": ["tokens", 1, 1],
                "params": ["in buffer<bf16>", "out buffer<f32>"],
                "args": [{"param": 0}, {"scratch": "tmp"}]}],
        }))
        .unwrap();
        c
    };
    let (a, b) = (with_scratch(1, 8), with_scratch(2, 16));
    let tiled = tile(
        "scale",
        "tokens",
        &[Piece { from: &a, min: None, max: Some(4) }, Piece { from: &b, min: Some(5), max: None }],
    )
    .unwrap();
    let imp = serde_json::to_value(&tiled.ops["scale"].imp).unwrap();
    assert_eq!(
        (
            imp["scratch"].as_object().unwrap().keys().map(String::as_str).collect::<Vec<_>>(),
            &imp["launches"][0]["args"][1],
            &imp["launches"][1]["args"][1]
        ),
        (vec!["tmp", "tmp@1"], &json!({"scratch": "tmp"}), &json!({"scratch": "tmp@1"}))
    );

    assert!(Candidate::join([&a, &b]).unwrap_err().0[0].contains("op `scale` is implemented twice"));
    let joined = Candidate::join([&tiled]).unwrap();
    assert_eq!(joined.to_json(), tiled.to_json());
}
