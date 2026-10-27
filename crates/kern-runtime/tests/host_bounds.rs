//! A runtime running another process's plans (a decode rank, a pipeline
//! stage) holds facts of its own: how many host pages it reserved. A plan
//! that contradicts them is the caller's bug and must come back as an
//! `Api` error before anything is issued, not run into the wrong part of
//! the host block.

use kern_manifest::Verified;

use kern_pool::Copies;
use kern_runtime::{Capacity, Error, Runtime};

/// A paged `kv` of 4 KiB a token and a 24-byte `gdn` slot; the one
/// program that names them is never run.
fn plain(tokens: u64) -> Runtime {
    let m = manifest();
    let dir = std::env::temp_dir().join(format!("kern-host-bounds-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let capacity = Capacity { tokens: Some(tokens), slots: 4, landings: 0 };
    Runtime::load(&m, Some(&dir), 0, Some(capacity), None).unwrap()
}

fn manifest() -> Verified {
    let m = serde_json::json!({
        "schema_version": 7, "model": "host-bounds-test", "vars": {},
        "states": {"kv": {"bytes_per_token": 4096}, "gdn": {"bytes_per_seq": 24}},
        "buffers": {}, "modules": {},
        "ops": {"touch": {
            "params": ["inout state", "inout state", "inout state", "i32", "i32", "i32"],
            "impl": {"launches": [{"entry": "extern:cublaslt_bf16_tn"}]}
        }},
        "programs": {"touch": {"calls": [{"op": "touch", "args": [
            {"state": "kv"}, {"state": "gdn"}, {"state": "kv"}, {"i32": 1}, {"i32": 1}, {"i32": 1}
        ]}]}}
    });
    Verified::from_json(&m.to_string()).unwrap()
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn a_park_past_the_reserved_host_pages_is_refused() {
    let mut rt = plain(4096);
    // Host page 1 of a block of one 64 KiB page and 4096 slots of 24 bytes
    // lies inside the slots: a park there overwrites what they hold.
    rt.reserve_host(1, 4096).unwrap();
    let past = Copies { pages: vec![(0, 1u64)], slot: None };
    let got = rt.host_out(&past).map(|_| ());
    assert!(matches!(got, Err(Error::Api(_))), "a park into host page 1 of 1: {got:?}");
    // The same plan through the checked entry, for contrast.
    assert!(matches!(rt.copy_out(&past), Err(Error::Api(_))));
}
