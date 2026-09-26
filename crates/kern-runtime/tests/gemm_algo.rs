//! A pinned cuBLASLt algorithm on `extern:cublaslt_bf16_tn`: it lands the
//! default op's exact product, per `when` range, in a run and in a captured
//! graph; one the shape cannot run fails the load.

use std::collections::BTreeMap;
use std::os::raw::c_void;

use cudarc::cublaslt::sys as lt;
use half::bf16;
use kern_manifest::Verified;
use kern_runtime::{Capacity, Runtime};
use serde_json::{json, Value};

const MAX: usize = 256;
const N: usize = 512;
const K: usize = 1024;

/// cuBLASLt's first heuristic answer for the bf16 `C[m, N] = A[m, K] · W[N, K]ᵀ`
/// as a manifest `algo`: what a caller that times candidates would write.
fn heuristic_algo(m: usize) -> Value {
    use lt::cublasLtMatmulAlgoConfigAttributes_t as Cfg;
    let ok = |s: lt::cublasStatus_t| assert_eq!(s, lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS);
    let bf = lt::cudaDataType::CUDA_R_16BF;
    unsafe {
        cudarc::driver::result::init().unwrap();
        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
        ctx.bind_to_thread().unwrap();
        let mut handle = std::ptr::null_mut();
        ok(lt::cublasLtCreate(&mut handle));
        let mut desc = std::ptr::null_mut();
        ok(lt::cublasLtMatmulDescCreate(
            &mut desc,
            lt::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            lt::cudaDataType::CUDA_R_32F,
        ));
        let t = cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T;
        ok(lt::cublasLtMatmulDescSetAttribute(
            desc,
            lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
            &t as *const _ as *const c_void,
            4,
        ));
        let mut layouts = [std::ptr::null_mut(); 3];
        for (l, (rows, cols, ld)) in layouts.iter_mut().zip([(K, N, K), (K, m, K), (N, m, N)]) {
            ok(lt::cublasLtMatrixLayoutCreate(l, bf, rows as u64, cols as u64, ld as i64));
        }
        let mut pref = std::ptr::null_mut();
        ok(lt::cublasLtMatmulPreferenceCreate(&mut pref));
        let ws: usize = 32 << 20;
        ok(lt::cublasLtMatmulPreferenceSetAttribute(
            pref,
            lt::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &ws as *const _ as *const c_void,
            8,
        ));
        let mut found = std::mem::zeroed::<lt::cublasLtMatmulHeuristicResult_t>();
        let mut count = 0;
        let [w, a, c] = layouts;
        ok(lt::cublasLtMatmulAlgoGetHeuristic(handle, desc, w, a, c, c, pref, 1, &mut found, &mut count));
        assert_eq!(count, 1);
        let get = |attr: Cfg, size: usize| {
            let mut v = 0u64;
            let mut written = 0;
            ok(lt::cublasLtMatmulAlgoConfigGetAttribute(
                &found.algo,
                attr,
                &mut v as *mut _ as *mut c_void,
                size,
                &mut written,
            ));
            v
        };
        let algo = json!({
            "id": get(Cfg::CUBLASLT_ALGO_CONFIG_ID, 4) as i32,
            "tile": get(Cfg::CUBLASLT_ALGO_CONFIG_TILE_ID, 4),
            "stages": get(Cfg::CUBLASLT_ALGO_CONFIG_STAGES_ID, 4),
            "split_k": get(Cfg::CUBLASLT_ALGO_CONFIG_SPLITK_NUM, 4) as i32,
            "reduction": get(Cfg::CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, 4),
            "swizzle": get(Cfg::CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, 4),
            "custom": get(Cfg::CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, 4),
            "inner_shape": get(Cfg::CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, 2),
            "cluster_shape": get(Cfg::CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, 2),
        });
        lt::cublasLtMatmulPreferenceDestroy(pref);
        for l in layouts {
            lt::cublasLtMatrixLayoutDestroy(l);
        }
        lt::cublasLtMatmulDescDestroy(desc);
        lt::cublasLtDestroy(handle);
        algo
    }
}

/// The default op and a pinned one, one launch per range of `rows`.
fn manifest(small: Value, large: Value) -> Verified {
    let params = json!(["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i32"]);
    let call = |op: &str, out: &str| json!({"op": op, "args": [{"buf": "a"}, {"buf": "w"}, {"buf": out}, {"var": "rows"}, {"i32": N}, {"i32": K}]});
    let m = json!({
        "schema_version": 5, "model": "gemm-algo-test", "vars": {"rows": {"max": MAX}}, "states": {},
        "buffers": {
            "a": {"kind": "input", "dtype": "bf16", "shape": ["rows", K]},
            "w": {"kind": "input", "dtype": "bf16", "shape": [N, K]},
            "default": {"kind": "output", "dtype": "bf16", "shape": ["rows", N]},
            "pinned": {"kind": "output", "dtype": "bf16", "shape": ["rows", N]}
        },
        "modules": {},
        "ops": {
            "gemm": {"params": params, "impl": {"launches": [{"entry": "extern:cublaslt_bf16_tn"}]}},
            "gemm_pinned": {"params": params, "impl": {"launches": [
                {"entry": "extern:cublaslt_bf16_tn", "when": {"var": "rows", "max": 128}, "algo": small},
                {"entry": "extern:cublaslt_bf16_tn", "when": {"var": "rows", "min": 129}, "algo": large}
            ]}}
        },
        "programs": {
            "both": {"batch": {"groups": 1, "rows": "rows"}, "graph": true,
                     "calls": [call("gemm", "default"), call("gemm_pinned", "pinned")]}
        }
    });
    Verified::from_json(&m.to_string()).unwrap()
}

fn load(v: &Verified) -> kern_runtime::Result<Runtime> {
    Runtime::load(v, None, 0, Some(Capacity { tokens: Some(1), seqs: 1 }), None)
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn a_pinned_algorithm_lands_the_default_product_in_each_range() {
    let mut rt = load(&manifest(heuristic_algo(128), heuristic_algo(MAX))).unwrap();
    // a is a permutation (one 1.0 per row at column 7i mod K), w cycles through -2..=2: the product is exact.
    let a: Vec<u8> = (0..MAX * K)
        .flat_map(|i| bf16::from_f32(if i % K == (i / K * 7) % K { 1.0 } else { 0.0 }).to_le_bytes())
        .collect();
    let wv = |n: usize, k: usize| ((n + k) % 5) as f32 - 2.0;
    let w: Vec<u8> = (0..N * K).flat_map(|i| bf16::from_f32(wv(i / K, i % K)).to_le_bytes()).collect();
    for rows in [100, MAX] {
        let vars = BTreeMap::from([("rows".to_string(), rows as u64)]);
        rt.write_input_at("a", &a[..rows * K * 2], &vars).unwrap();
        rt.write_input("w", &w).unwrap();
        let expected: Vec<u8> =
            (0..rows * N).flat_map(|i| bf16::from_f32(wv(i % N, (i / N * 7) % K)).to_le_bytes()).collect();
        rt.run("both", &vars).unwrap();
        assert_eq!(rt.read_output("default").unwrap()[..rows * N * 2], expected[..], "default, {rows} rows");
        assert_eq!(rt.read_output("pinned").unwrap()[..rows * N * 2], expected[..], "pinned, {rows} rows");
        rt.capture("both", &vars).unwrap();
        rt.run_captured("both", &vars).unwrap();
        assert_eq!(rt.read_output("pinned").unwrap()[..rows * N * 2], expected[..], "pinned in a graph, {rows} rows");
    }
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn an_algorithm_the_shape_cannot_run_fails_the_load() {
    let good = heuristic_algo(MAX);
    let mut bad = good.clone();
    bad["tile"] = json!(9999);
    let err = load(&manifest(good, bad)).err().expect("an unknown tile loaded");
    assert!(err.to_string().contains("kernel artifact"), "{err}");
}
