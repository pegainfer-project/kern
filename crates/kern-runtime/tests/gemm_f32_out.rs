//! The f32-result GEMM matches an exact CPU product in eager and graph
//! execution. On Linux the runtime also leaves classic cuBLAS unmapped.

use std::collections::BTreeMap;

use half::bf16;
use kern_manifest::Verified;
use kern_runtime::Runtime;
use serde_json::json;

const M: usize = 8;
const N: usize = 16;
const K: usize = 32;

fn av(m: usize, k: usize) -> f32 {
    ((m * 7 + k * 3) % 5) as f32 - 2.0
}

fn wv(n: usize, k: usize) -> f32 {
    ((n * 5 + k * 11) % 7) as f32 - 3.0
}

fn bf16_bytes(values: impl Iterator<Item = f32>) -> Vec<u8> {
    values.flat_map(|v| bf16::from_f32(v).to_le_bytes()).collect()
}

fn product() -> Vec<f32> {
    (0..M).flat_map(|row| (0..N).map(move |col| (0..K).map(|k| av(row, k) * wv(col, k)).sum())).collect()
}

fn manifest(stride: Option<usize>) -> Verified {
    let mut params = vec!["in buffer<bf16>", "in buffer<bf16>", "out buffer<f32>", "i32", "i32", "i32"];
    let mut args = vec![
        json!({"buf": "a"}),
        json!({"buf": "w"}),
        json!({"buf": "out"}),
        json!({"i32": M}),
        json!({"i32": N}),
        json!({"i32": K}),
    ];
    if let Some(stride) = stride {
        params.push("i32");
        args.push(json!({"i32": stride}));
    }
    let m = json!({
        "schema_version": 7, "model": "f32-gemm-test", "vars": {}, "states": {},
        "buffers": {
            "a": {"kind": "input", "dtype": "bf16", "shape": [M, K]},
            "w": {"kind": "input", "dtype": "bf16", "shape": [N, K]},
            "out": {"kind": "output", "dtype": "f32", "shape": [M, stride.unwrap_or(N)]}
        },
        "modules": {},
        "ops": {"gemm": {
            "params": params,
            "impl": {"launches": [{"entry": "extern:cublas_bf16_tn_f32"}]}
        }},
        "programs": {"probe": {"calls": [{"op": "gemm", "args": args}]}}
    });
    Verified::from_json(&m.to_string()).unwrap()
}

fn output(rt: &Runtime, stride: usize) -> Vec<f32> {
    let bytes = rt.read_output("out").unwrap();
    assert_eq!(bytes.len(), M * stride * size_of::<f32>());
    bytes
        .chunks_exact(stride * size_of::<f32>())
        .flat_map(|row| row.as_chunks::<{ size_of::<f32>() }>().0.iter().take(N))
        .map(|&b| f32::from_le_bytes(b))
        .collect()
}

fn check_gemm(stride: Option<usize>) {
    let mut rt = Runtime::load(&manifest(stride), None, 0, None, None).unwrap();
    rt.write_input("a", &bf16_bytes((0..M * K).map(|i| av(i / K, i % K)))).unwrap();
    rt.write_input("w", &bf16_bytes((0..N * K).map(|i| wv(i / K, i % K)))).unwrap();
    let stride = stride.unwrap_or(N);
    let want = product();
    assert!(want.iter().any(|&value| value != 0.0));
    let vars = BTreeMap::new();
    rt.run("probe", &vars).unwrap();
    assert_eq!(output(&rt, stride), want, "eager");

    rt.capture("probe", &vars).unwrap();
    // Replay must overwrite the product from the eager run.
    rt.write_input("a", &bf16_bytes((0..M * K).map(|i| -av(i / K, i % K)))).unwrap();
    rt.run_captured("probe", &vars).unwrap();
    let replay = want.into_iter().map(|value| -value).collect::<Vec<_>>();
    assert_eq!(output(&rt, stride), replay, "graph replay");
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn f32_result_lands_the_product_eager_and_captured() {
    check_gemm(None);
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn f32_result_accepts_a_padded_row_stride() {
    check_gemm(Some(N + 4));
}

#[cfg(target_os = "linux")]
#[test]
#[ignore = "requires a CUDA GPU"]
fn the_runtime_loads_no_libcublas() {
    // Runtime construction used to create a classic handle even without a GEMM.
    let _rt = Runtime::load(&manifest(None), None, 0, None, None).unwrap();
    let maps = std::fs::read_to_string("/proc/self/maps").expect("read Linux process mappings");
    let loaded: Vec<_> = maps.lines().filter(|line| line.contains("/libcublas.so")).collect();
    assert!(loaded.is_empty(), "libcublas is in this process: {loaded:?}");
}
