//! The fp8 GEMM built-in: e4m3 operands with device-side per-tensor scales
//! land the scaled product, in a run and in a captured graph.

use std::collections::BTreeMap;

use half::bf16;
use kern_manifest::Verified;
use kern_runtime::{Capacity, Runtime};

const M: usize = 32;
const N: usize = 48;
const K: usize = 64;

/// e4m3 for the few exact values the operands use.
fn e4m3(v: f32) -> u8 {
    match v {
        0.0 => 0x00,
        1.0 => 0x38,
        2.0 => 0x40,
        -1.0 => 0xb8,
        -2.0 => 0xc0,
        _ => unreachable!("{v} is not an operand value"),
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn e4m3_operands_land_the_product_scaled_by_both_device_scales() {
    let manifest = serde_json::json!({
        "schema_version": 5, "model": "fp8-gemm-test", "vars": {}, "states": {},
        "buffers": {
            "a": {"kind": "input", "dtype": "fp8e4m3", "shape": [M, K]},
            "w": {"kind": "input", "dtype": "fp8e4m3", "shape": [N, K]},
            "a_scale": {"kind": "input", "dtype": "f32", "shape": [1]},
            "w_scale": {"kind": "input", "dtype": "f32", "shape": [1]},
            "out": {"kind": "output", "dtype": "bf16", "shape": [M, N]}
        },
        "modules": {},
        "ops": {"gemm": {
            "params": ["in buffer<fp8e4m3>", "in buffer<fp8e4m3>", "out buffer<bf16>", "in buffer<f32>", "in buffer<f32>",
                       "i32", "i32", "i32"],
            "impl": {"launches": [{"entry": "extern:cublaslt_fp8_tn"}]}
        }},
        "programs": {"probe": {"calls": [{"op": "gemm", "args": [
            {"buf": "a"}, {"buf": "w"}, {"buf": "out"}, {"buf": "a_scale"}, {"buf": "w_scale"},
            {"i32": M}, {"i32": N}, {"i32": K}
        ]}]}}
    });
    let verified = Verified::from_json(&manifest.to_string()).unwrap();
    let mut rt = Runtime::load(&verified, None, 0, Some(Capacity { tokens: Some(1), seqs: 1 }), None).unwrap();
    // a is a permutation (one 1.0 per row at column 7i mod K), w cycles through -2..=2
    let a: Vec<u8> = (0..M * K).map(|i| e4m3(if i % K == (i / K * 7) % K { 1.0 } else { 0.0 })).collect();
    let wv = |n: usize, k: usize| ((n + k) % 5) as f32 - 2.0;
    let w: Vec<u8> = (0..N * K).map(|i| e4m3(wv(i / K, i % K))).collect();
    rt.write_input("a", &a).unwrap();
    rt.write_input("w", &w).unwrap();
    rt.write_input("a_scale", &0.5f32.to_le_bytes()).unwrap();
    rt.write_input("w_scale", &4.0f32.to_le_bytes()).unwrap();
    let expected: Vec<u8> =
        (0..M * N).flat_map(|i| bf16::from_f32(2.0 * wv(i % N, (i / N * 7) % K)).to_le_bytes()).collect();
    let vars = BTreeMap::new();
    rt.run("probe", &vars).unwrap();
    assert_eq!(rt.read_output("out").unwrap(), expected);
    rt.write_input("w_scale", &2.0f32.to_le_bytes()).unwrap();
    rt.capture("probe", &vars).unwrap();
    rt.run_captured("probe", &vars).unwrap();
    let halved: Vec<u8> = (0..M * N).flat_map(|i| bf16::from_f32(wv(i % N, (i / N * 7) % K)).to_le_bytes()).collect();
    assert_eq!(rt.read_output("out").unwrap(), halved, "the scale is read from the device at run time");
}
