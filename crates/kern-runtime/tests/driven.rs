//! A runtime running another process's plans (a decode rank, a pipeline
//! stage) holds facts of its own: which chunk backs which position, how
//! many host pages it reserved. A plan that contradicts them is the
//! caller's bug and must come back as an `Api` error before anything is
//! issued, not run into aliased chunks or the wrong part of the host block.

use kern_manifest::Verified;
use std::sync::Arc;

use kern_pool::{Copies, Denied, Remap};
use kern_runtime::{Capacity, Error, Runtime};

/// A paged `kv` of 4 KiB a token and a 24-byte `gdn` slot; the one
/// program that names them is never run.
fn plain(tokens: u64) -> Runtime {
    let m = manifest();
    let dir = std::env::temp_dir().join(format!("kern-driven-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let capacity = Capacity { tokens: Some(tokens), seqs: 4, landings: 0 };
    Runtime::load(&m, Some(&dir), 0, Some(capacity), None).unwrap()
}

fn manifest() -> Verified {
    let m = serde_json::json!({
        "schema_version": 7, "model": "driven-test", "vars": {},
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
fn the_remaps_a_pool_over_the_budget_plans_run() {
    let mut rt = plain(4096);
    let pool = Arc::new(rt.budget().pool(&manifest()).unwrap());
    let mut leases = Vec::new();
    let mut ran = 0;
    // Slots past the first ones are carved from free pages: each a remap.
    while ran < 3 {
        match pool.lease_slot() {
            Ok(l) => leases.push(l),
            Err(Denied::Remapping) => {
                let plan = pool.take_pending().expect("a remap planned");
                rt.remap(plan.clone()).unwrap();
                assert_eq!(rt.remapped(true).unwrap(), 1);
                pool.complete(plan);
                ran += 1;
            }
            Err(d) => panic!("{d:?} after {ran} remaps"),
        }
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn a_remap_that_maps_a_chunk_placed_elsewhere_is_refused() {
    // Enough `kv` for two chunk positions.
    let chunk = plain(16).budget().chunk;
    let mut rt = plain(2 * chunk / 4096);
    let p = rt.placement();
    let mapped: Vec<(usize, u32)> = (0..p.shape()[0]).filter_map(|x| Some((x, p.at(0, x)?))).collect();
    let [(x, c), (y, _), ..] = mapped[..] else { panic!("two positions of `kv` mapped: {mapped:?}") };
    let alias = Remap { unmap: vec![(0, y)], map: vec![(0, y, c)], access: vec![(0, y..y + 1)], ..Remap::default() };
    let got = rt.remap(alias);
    let aliased = got.is_ok() && {
        rt.remapped(true).unwrap();
        let at = |x: usize| x * chunk as usize;
        rt.write_state_at("kv", at(x), &[0xA5; 64]).unwrap();
        rt.read_state_at("kv", at(y), 64).unwrap() == [0xA5; 64]
    };
    assert!(
        matches!(got, Err(Error::Api(_))),
        "chunk {c} of position {x} mapped at {y} too: {got:?}; a write to {x} reads back at {y}: {aliased}"
    );
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
