//! Derived weights end to end: two sources share one staging window, each
//! uploaded only after the call before it has read its predecessor, and
//! the carries they leave behind are what the serving program reads.

use std::collections::BTreeMap;

use half::bf16;
use kern_manifest::types::BufferKind;
use kern_manifest::Verified;
use kern_runtime::{Capacity, Runtime, Safetensors};

const N: usize = 16;

fn tensor(f: impl Fn(usize, usize) -> f32) -> Vec<u8> {
    (0..N * N).flat_map(|i| bf16::from_f32(f(i / N, i % N)).to_le_bytes()).collect()
}

fn checkpoint(tensors: &[(&str, &[u8])]) -> Vec<u8> {
    let (mut meta, mut at) = (serde_json::Map::new(), 0);
    for (name, bytes) in tensors {
        meta.insert(
            name.to_string(),
            serde_json::json!({"dtype": "BF16", "shape": [N, N], "data_offsets": [at, at + bytes.len()]}),
        );
        at += bytes.len();
    }
    let mut header = serde_json::Value::Object(meta).to_string().into_bytes();
    header.resize(header.len().div_ceil(8) * 8, b' ');
    let mut data = (header.len() as u64).to_le_bytes().to_vec();
    data.extend(header);
    tensors.iter().for_each(|(_, b)| data.extend(*b));
    data
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn sources_derive_through_one_window_and_are_gone_after() {
    let gemm = |a: &str, b: &str, c: &str| serde_json::json!({"op": "gemm", "args": [{"buf": a}, {"buf": b}, {"buf": c}, {"i32": N}, {"i32": N}, {"i32": N}]});
    let square = |kind: &str| serde_json::json!({"kind": kind, "dtype": "bf16", "shape": [N, N]});
    let bound = |kind: &str, t: &str| {
        let mut b = square(kind);
        b["bind"] = serde_json::json!([{"tensor": t}]);
        b
    };
    let manifest = serde_json::json!({
        "schema_version": 5, "model": "derive-test", "vars": {}, "states": {},
        "buffers": {
            "eye": bound("weight", "eye"), "s1": bound("source", "a"), "s2": bound("source", "b"),
            "c1": square("carry"), "c2": square("carry"), "o1": square("output"), "o2": square("output")
        },
        "modules": {},
        "ops": {"gemm": {
            "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i32"],
            "impl": {"launches": [{"entry": "extern:cublaslt_bf16_tn"}]}
        }},
        "programs": {
            "load": {"derive": true, "calls": [gemm("eye", "s1", "c1"), gemm("eye", "s2", "c2")]},
            "probe": {"calls": [gemm("eye", "c1", "o1"), gemm("eye", "c2", "o2")]}
        }
    });
    let verified = Verified::from_json(&manifest.to_string()).unwrap();
    let dir = std::env::temp_dir().join(format!("kern-derive-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut rt = Runtime::load(&verified, Some(&dir), 0, Some(Capacity { tokens: Some(1), seqs: 1 }), None).unwrap();
    let sources = |rt: &Runtime| -> Vec<String> {
        rt.buffer_sizes().into_iter().filter(|(_, k, _)| *k == BufferKind::Source).map(|(n, ..)| n.into()).collect()
    };
    assert_eq!(sources(&rt), ["s1", "s2"]);

    let (eye, a, b) = (
        tensor(|r, c| if r == c { 1.0 } else { 0.0 }),
        tensor(|r, c| (r * 3 + c) as f32 % 13.0 - 6.0),
        tensor(|r, c| (r + c * 5) as f32 % 11.0 - 5.0),
    );
    let data = checkpoint(&[("eye", &eye), ("a", &a), ("b", &b)]);
    let tensors = Safetensors::parse(&[&data]).unwrap();
    rt.load_weights(&tensors).unwrap();
    assert_eq!(sources(&rt), Vec::<String>::new());
    assert!(rt.load_weights(&tensors).is_err());

    rt.run("probe", &BTreeMap::new()).unwrap();
    assert_eq!((rt.read_output("o1").unwrap(), rt.read_output("o2").unwrap()), (a, b));
    assert!(rt.run("load", &BTreeMap::new()).is_err());
    drop(rt);
    std::fs::remove_dir_all(dir).unwrap();
}
