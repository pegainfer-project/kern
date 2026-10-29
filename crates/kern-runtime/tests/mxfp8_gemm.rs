//! The MXFP8 GEMM built-in: e4m3 operands with an e8m0 scale a 32-element block of K, the scales in
//! cuBLAS's 128x4 tiled layout, in a run and in a captured graph.

use std::collections::BTreeMap;

use half::bf16;
use kern_manifest::Verified;
use kern_runtime::{Capacity, Runtime};

// two row tiles of A, a partial second of W, two tiles of K blocks
const M: usize = 256;
const N: usize = 160;
const K: usize = 320;
const KB: usize = K / 32;

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

/// Scale bytes of `rows` rows, entry (r, b) = 2^exp(r, b) at its tiled offset.
fn scales(rows: usize, exp: impl Fn(usize, usize) -> i32) -> Vec<u8> {
    let (row_tiles, kb_tiles) = (rows.div_ceil(128), KB.div_ceil(4));
    let mut sf = vec![127u8; row_tiles * kb_tiles * 512];
    for r in 0..rows {
        for b in 0..KB {
            let at = (r / 128 * kb_tiles + b / 4) * 512 + r % 32 * 16 + r % 128 / 32 * 4 + b % 4;
            sf[at] = (127 + exp(r, b)) as u8;
        }
    }
    sf
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn mxfp8_operands_land_the_product_scaled_by_each_block() {
    let sf_len = |rows: usize| rows.div_ceil(128) * 128 * KB.div_ceil(4) * 4;
    let manifest = serde_json::json!({
        "schema_version": 7, "model": "mxfp8-gemm-test", "vars": {}, "states": {},
        "buffers": {
            "a": {"kind": "input", "dtype": "fp8e4m3", "shape": [M, K]},
            "w": {"kind": "input", "dtype": "fp8e4m3", "shape": [N, K]},
            "a_sf": {"kind": "input", "dtype": "u8", "shape": [sf_len(M)]},
            "w_sf": {"kind": "input", "dtype": "u8", "shape": [sf_len(N)]},
            "out": {"kind": "output", "dtype": "bf16", "shape": [M, N]}
        },
        "modules": {},
        "ops": {"gemm": {
            "params": ["in buffer<fp8e4m3>", "in buffer<fp8e4m3>", "out buffer<bf16>", "in buffer<u8>", "in buffer<u8>",
                       "i32", "i32", "i32"],
            "impl": {"launches": [{"entry": "extern:cublaslt_mxfp8_tn"}]}
        }},
        "programs": {"probe": {"calls": [{"op": "gemm", "args": [
            {"buf": "a"}, {"buf": "w"}, {"buf": "out"}, {"buf": "a_sf"}, {"buf": "w_sf"},
            {"i32": M}, {"i32": N}, {"i32": K}
        ]}]}}
    });
    let verified = Verified::from_json(&manifest.to_string()).unwrap();
    let mut rt =
        Runtime::load(&verified, None, 0, Some(Capacity { tokens: Some(1), slots: 1, landings: 0 }), None).unwrap();
    // row i of a is one 1.0 at column 7i mod K, so out[i, n] = sa(i, b) sw(n, b) w[n, 7i mod K], b its block
    let col = |i: usize| i * 7 % K;
    let a: Vec<u8> = (0..M * K).map(|x| e4m3(if x % K == col(x / K) { 1.0 } else { 0.0 })).collect();
    let wv = |n: usize, k: usize| ((n + k) % 5) as f32 - 2.0;
    let w: Vec<u8> = (0..N * K).map(|x| e4m3(wv(x / K, x % K))).collect();
    let ea = |r: usize, b: usize| ((r + b) % 3) as i32 - 1;
    let ew = |r: usize, b: usize| ((3 * r + 2 * b) % 5) as i32 - 2;
    rt.write_input("a", &a).unwrap();
    rt.write_input("w", &w).unwrap();
    rt.write_input("a_sf", &scales(M, ea)).unwrap();
    rt.write_input("w_sf", &scales(N, ew)).unwrap();
    let expected = |shift: i32| -> Vec<u8> {
        (0..M * N)
            .flat_map(|x| {
                let (i, n) = (x / N, x % N);
                let b = col(i) / 32;
                let v = wv(n, col(i)) * 2f32.powi(ea(i, b) + ew(n, b) + shift);
                bf16::from_f32(v).to_le_bytes()
            })
            .collect()
    };
    let vars = BTreeMap::new();
    rt.run("probe", &vars).unwrap();
    assert_eq!(rt.read_output("out").unwrap(), expected(0));
    rt.write_input("w_sf", &scales(N, |r, b| ew(r, b) - 1)).unwrap();
    rt.capture("probe", &vars).unwrap();
    rt.run_captured("probe", &vars).unwrap();
    assert_eq!(rt.read_output("out").unwrap(), expected(-1), "the scales are read from the device at run time");
}
