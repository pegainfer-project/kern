use kern_manifest::{verify, Manifest};

/// The toy manifest the schema page opens with: every top-level section,
/// one op, one call. Kept valid by this test.
const MINIMAL: &str = include_str!("../../../examples/minimal.json");

#[test]
fn minimal_example_verifies() {
    let m = verify(Manifest::from_json(MINIMAL).expect("parse"))
        .unwrap_or_else(|errs| panic!("examples/minimal.json failed verification:\n{}", errs.join("\n")));
    assert_eq!(m.programs["step"].calls.len(), 1);
    assert_eq!(m.ops["scale"].imp.launches[0].module(), Some("toy"));
}
