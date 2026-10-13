//! `kern verify` reads the serving protocol a manifest's host drives: a
//! hosted manifest's (an engine's caches as host states, the engine
//! sampling) is its self-hosted twin's, with no token handed back.

use std::path::PathBuf;
use std::process::Command;

fn verify(name: &str, m: &serde_json::Value) -> (bool, String) {
    let dir = std::env::temp_dir().join(format!("kern-verify-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path: PathBuf = dir.join("m.json");
    std::fs::write(&path, m.to_string()).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_kern")).arg("verify").arg(&path).output().unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    let text = String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr);
    (out.status.success(), text)
}

fn plain() -> serde_json::Value {
    serde_json::from_str(include_str!("../../kern-manifest/tests/fixtures/plain.json")).unwrap()
}

/// `plain` on an engine's memory: 16-token pages of 2-byte `kv` tokens, one
/// 24-byte line of `gdn` per block. As a protocol over host states it fails:
/// the lines are no runtime state's.
fn hosted() -> serde_json::Value {
    let mut v = plain();
    v["states"] = serde_json::json!({
        "kv": {"host": {"dtype": "u8", "shape": [0, 32], "strides": [32, 1]}},
        "gdn": {"host": {"dtype": "u8", "shape": [0, 24], "strides": [24, 1]}}
    });
    v["buffers"]["line_index"]["domain"]["stride"] = serde_json::json!(24);
    v
}

#[test]
fn a_hosted_manifest_verifies_as_its_host_drives_it() {
    let (ok, text) = verify("plain", &plain());
    assert!(ok, "{text}");
    let (ok, text) = verify("hosted", &hosted());
    assert!(ok, "{text}");
    assert!(text.contains("hosted: the protocol of its self-hosted twin, unsampled"), "{text}");
    assert!(text.contains("lines     `line_index` [3, seqs]"), "{text}");
}
