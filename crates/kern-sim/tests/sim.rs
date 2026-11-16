use std::collections::BTreeMap;

use kern_manifest::Verified;
use kern_sim::cluster::{ClusterConfig, EndpointConfig, SimCluster};
use kern_sim::cost::{CostBook, MatchTier, PriceOptions};
use kern_sim::memory::{DeviceMemorySpec, MemoryBudget};
use kern_sim::router::RouterPolicy;
use kern_sim::trace::{
    ArrivalMode, JsonlTraceOptions, RequestTrace, SyntheticSessionConfig,
};
use kern_sim::workload::{price_workload, Workload};
use serde_json::json;

const SHA_A: &str = "1111111111111111111111111111111111111111111111111111111111111111";
const SHA_B: &str = "2222222222222222222222222222222222222222222222222222222222222222";
const SHA_UNIFIED: &str = "3333333333333333333333333333333333333333333333333333333333333333";

/// Build a verified serving manifest with paged KV + optional per-sequence state,
/// parameterized by linear op implementation and module sha256.
fn build_serving_manifest(
    linear_launches: serde_json::Value,
    attn_sha: &str,
    with_recurrent_state: bool,
) -> Verified {
    let mut states = json!({
        "kv": { "bytes_per_token": 1024 }
    });
    let mut buffers = json!({
        "token_ids": { "dtype": "i64", "shape": ["tokens"], "kind": "input", "fill": "token" },
        "positions": { "dtype": "i64", "shape": ["tokens"], "kind": "input", "fill": "position" },
        "slots": { "dtype": "i64", "shape": ["tokens"], "kind": "input", "fill": "slot", "domain": { "index_into": "kv", "stride": 1 } },
        "lengths": { "dtype": "i32", "shape": ["seqs"], "kind": "input", "fill": "seq_len" },
        "pages": { "dtype": "i32", "shape": ["seqs", 64], "kind": "input", "domain": { "index_into": "kv", "stride": 16 } },
        "hidden": { "dtype": "bf16", "shape": ["tokens", 128], "kind": "workspace" },
        "proj": { "dtype": "bf16", "shape": ["tokens", 128], "kind": "workspace" },
        "weight": { "dtype": "bf16", "shape": [128, 128], "kind": "weight", "bind": [{"tensor": "w"}] },
        "next_token": { "dtype": "i64", "shape": ["seqs"], "kind": "output", "fill": "tokens" }
    });

    let mut prefill_calls = vec![
        json!({
            "label": "embed",
            "op": "embed_op",
            "args": [{"buf": "token_ids"}, {"buf": "positions"}, {"buf": "hidden"}, {"var": "tokens"}]
        }),
        json!({
            "label": "attn",
            "op": "attn_op",
            "args": [{"buf": "hidden"}, {"state": "kv"}, {"buf": "slots"}, {"buf": "pages"}, {"buf": "lengths"}, {"buf": "proj"}, {"var": "tokens"}]
        }),
        json!({
            "label": "linear",
            "op": "linear_op",
            "args": [{"buf": "proj"}, {"buf": "weight"}, {"buf": "hidden"}, {"var": "tokens"}]
        }),
    ];

    let mut decode_calls = prefill_calls.clone();

    let linear_json = linear_launches.to_string();
    let uses_tmp_scratch = linear_json.contains("\"scratch\":\"tmp\"");
    let uses_fast_mod = linear_json.contains("\"fast_mod\"");
    let linear_impl = if uses_tmp_scratch {
        json!({
            "scratch": { "tmp": { "dtype": "u8", "shape": ["tokens", 128] } },
            "launches": linear_launches
        })
    } else {
        json!({
            "launches": linear_launches
        })
    };

    let mut ops = json!({
        "embed_op": {
            "params": ["in buffer<i64>", "in buffer<i64>", "out buffer<bf16>", "i32"],
            "impl": {
                "launches": [{
                    "module": "base_mod",
                    "entry": "k_embed",
                    "block": [128, 1, 1],
                    "grid": ["tokens", 1, 1]
                }]
            }
        },
        "attn_op": {
            "params": ["in buffer<bf16>", "inout state", "in buffer<i64>", "in buffer<i32>", "in buffer<i32>", "out buffer<bf16>", "i32"],
            "impl": {
                "launches": [{
                    "module": "attn_mod",
                    "entry": "k_attn",
                    "block": [128, 1, 1],
                    "grid": ["tokens", 1, 1]
                }]
            }
        },
        "linear_op": {
            "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32"],
            "impl": linear_impl
        },
        "sample_op": {
            "params": ["in buffer<bf16>", "out buffer<i64>", "i32"],
            "impl": {
                "launches": [{
                    "module": "base_mod",
                    "entry": "k_sample",
                    "block": [64, 1, 1],
                    "grid": ["seqs", 1, 1],
                    "args": [{"param": 0}, {"param": 1}, {"param": 2}]
                }]
            }
        }
    });

    if with_recurrent_state {
        states["rec"] = json!({ "bytes_per_seq": 65536 });
        buffers["lines"] = json!({
            "dtype": "i32",
            "shape": [1, "seqs"],
            "kind": "input",
            "domain": { "index_into": "rec", "stride": 65536 }
        });
        ops["rec_op"] = json!({
            "params": ["inout buffer<bf16>", "inout state", "in buffer<i32>", "i32"],
            "impl": {
                "launches": [{
                    "module": "base_mod",
                    "entry": "k_recurrent",
                    "block": [64, 1, 1],
                    "grid": ["seqs", 1, 1]
                }]
            }
        });
        let rec_call = json!({
            "label": "recurrent",
            "op": "rec_op",
            "args": [{"buf": "hidden"}, {"state": "rec"}, {"buf": "lines"}, {"var": "seqs"}]
        });
        prefill_calls.push(rec_call.clone());
        decode_calls.push(rec_call);
    }

    let sample_call = json!({
        "label": "sample",
        "op": "sample_op",
        "args": [{"buf": "hidden"}, {"buf": "next_token"}, {"var": "seqs"}]
    });
    prefill_calls.push(sample_call.clone());
    decode_calls.push(sample_call);

    let mut modules = json!({
        "base_mod": { "source": "base.cubin", "sha256": SHA_A },
        "attn_mod": { "source": "attn.cubin", "sha256": attn_sha }
    });
    if uses_fast_mod {
        modules["fast_mod"] = json!({ "source": "fast.cubin", "sha256": SHA_B });
    }

    let doc = json!({
        "schema_version": 7,
        "model": "synthetic-test-model",
        "vars": {
            "tokens": { "max": 256, "axis": "rows" },
            "seqs": { "max": 8, "axis": "groups" }
        },
        "states": states,
        "buffers": buffers,
        "modules": modules,
        "ops": ops,
        "programs": {
            "prefill": {
                "batch": { "groups": 1, "rows": "tokens" },
                "calls": prefill_calls
            },
            "decode_batch": {
                "batch": { "groups": 8, "rows": 1 },
                "graph": true,
                "calls": decode_calls
            }
        }
    });

    Verified::from_json(&doc.to_string()).expect("manifest must pass verification")
}

fn synthetic_bench_json(
    program: &str,
    rows: usize,
    chunk: bool,
    context: Vec<usize>,
    vars: BTreeMap<&str, u64>,
    call_p50s: &[f64],
    graph_p50: f64,
) -> String {
    let calls: Vec<serde_json::Value> = call_p50s
        .iter()
        .map(|&p50| json!({ "in_program": { "stats": { "p50": p50 } } }))
        .collect();
    json!({
        "scenarios": [{
            "scenario": {
                "id": format!("{program}-s0"),
                "groups": context.len(),
                "rows": rows,
                "chunk": chunk,
                "context": context,
                "weight": 1
            },
            "program": program,
            "vars": vars,
            "graph": { "stats": { "p50": graph_p50 } },
            "calls": calls
        }]
    })
    .to_string()
}

#[test]
fn when_range_composition_and_abi_fallback_reprice_accurately() {
    let base_launches = json!([{
        "module": "base_mod",
        "entry": "k_linear_bf16",
        "block": [128, 1, 1],
        "grid": ["tokens", 1, 1],
        "when": { "var": "seqs", "min": 1, "max": 8 },
        "args": [{"param": 0}, {"param": 1}, {"param": 2}, {"param": 3}]
    }]);
    let fast_launches = json!([
        {
            "module": "fast_mod",
            "entry": "k_quant",
            "params": ["in buffer<bf16>", "out buffer<u8>", "i32"],
            "block": [128, 1, 1],
            "grid": ["tokens", 1, 1],
            "args": [{"param": 0}, {"scratch": "tmp"}, {"param": 3}]
        },
        {
            "module": "fast_mod",
            "entry": "k_linear_int8",
            "params": ["in buffer<u8>", "in buffer<bf16>", "out buffer<bf16>", "i32"],
            "block": [128, 1, 1],
            "grid": ["tokens", 1, 1],
            "args": [{"scratch": "tmp"}, {"param": 1}, {"param": 2}, {"param": 3}]
        }
    ]);
    // Composed manifest: uses `base` for tokens <= 1 and `fast` for tokens >= 2,
    // plus repackages `attn_mod` under `SHA_UNIFIED` with the same entry + ABI.
    let composed_launches = json!([
        {
            "module": "base_mod",
            "entry": "k_linear_bf16",
            "block": [128, 1, 1],
            "grid": ["tokens", 1, 1],
            "when": { "var": "tokens", "min": 1, "max": 1 },
            "args": [{"param": 0}, {"param": 1}, {"param": 2}, {"param": 3}]
        },
        {
            "module": "fast_mod",
            "entry": "k_quant",
            "params": ["in buffer<bf16>", "out buffer<u8>", "i32"],
            "block": [128, 1, 1],
            "grid": ["tokens", 1, 1],
            "when": { "var": "tokens", "min": 2 },
            "args": [{"param": 0}, {"scratch": "tmp"}, {"param": 3}]
        },
        {
            "module": "fast_mod",
            "entry": "k_linear_int8",
            "params": ["in buffer<u8>", "in buffer<bf16>", "out buffer<bf16>", "i32"],
            "block": [128, 1, 1],
            "grid": ["tokens", 1, 1],
            "when": { "var": "tokens", "min": 2 },
            "args": [{"scratch": "tmp"}, {"param": 1}, {"param": 2}, {"param": 3}]
        }
    ]);

    let m_base = build_serving_manifest(base_launches, SHA_A, false);
    let m_fast = build_serving_manifest(fast_launches, SHA_A, false);
    let m_comp = build_serving_manifest(composed_launches, SHA_UNIFIED, false);

    let mut book = CostBook::new();
    // At batch=1 (tokens=1): base linear costs 50 us; fast linear costs 70 us (quant overhead).
    // At batch=4 (tokens=4): base linear costs 120 us; fast linear costs 65 us!
    book.ingest_bench_json(
        &m_base,
        &synthetic_bench_json(
            "decode_batch",
            1,
            false,
            vec![127],
            BTreeMap::from([("seqs", 1), ("tokens", 1)]),
            &[5.0, 20.0, 50.0, 3.0],
            78.0,
        ),
    )
    .unwrap();
    book.ingest_bench_json(
        &m_fast,
        &synthetic_bench_json(
            "decode_batch",
            1,
            false,
            vec![127; 4],
            BTreeMap::from([("seqs", 4), ("tokens", 4)]),
            &[8.0, 45.0, 65.0, 4.0],
            122.0,
        ),
    )
    .unwrap();

    let w = Workload::from_toml_str(
        "mix",
        r#"
samples = 12
seed = 1
[[sweep]]
groups = [1]
rows = [1]
context = [127]
weight = 10
[[sweep]]
groups = [4]
rows = [1]
context = [127]
weight = 10
"#,
    )
    .unwrap();

    // With EntryAbi enabled, `attn_op` (repackaged into SHA_UNIFIED) resolves via EntryAbi,
    // while `linear_op` resolves via ExactSha256 from `m_base` at batch=1 (50us) and `m_fast` at batch=4 (65us)!
    let rep = price_workload(&book, &m_comp, &w, PriceOptions::default()).unwrap();
    let expected_us = 10.0 * (5.0 + 20.0 + 50.0 + 3.0) + 10.0 * (8.0 + 45.0 + 65.0 + 4.0);
    assert!(
        (rep.total_us - expected_us).abs() < 1e-6,
        "got {} vs expected {expected_us}",
        rep.total_us
    );
    assert!((rep.calibrated_time_coverage - 1.0).abs() < 1e-9);
    assert!(rep.us_by_tier.contains_key(&MatchTier::EntryAbi));
    assert!(rep.us_by_tier.contains_key(&MatchTier::ExactSha256));

    // Under strict sha256 mode, repackaged `attn_mod` is flagged as Uncalibrated.
    let strict_rep = price_workload(
        &book,
        &m_comp,
        &w,
        PriceOptions {
            allow_abi_fallback: false,
            uncalibrated_launch_us: None,
        },
    )
    .unwrap();
    assert!(strict_rep.calibrated_time_coverage < 1.0);
}

#[test]
fn side_lanes_critical_path_uses_parallel_makespan() {
    let doc = json!({
        "schema_version": 7,
        "model": "parallel-lanes",
        "vars": {
            "tokens": { "max": 64, "axis": "rows" },
            "seqs": { "max": 4, "axis": "groups" }
        },
        "buffers": {
            "token_ids": { "dtype": "i64", "shape": ["tokens"], "kind": "input", "fill": "token" },
            "a": { "dtype": "bf16", "shape": ["tokens", 64], "kind": "workspace" },
            "b0": { "dtype": "bf16", "shape": ["tokens", 64], "kind": "workspace" },
            "b1": { "dtype": "bf16", "shape": ["tokens", 64], "kind": "workspace" },
            "next_token": { "dtype": "i64", "shape": ["seqs"], "kind": "output", "fill": "tokens" }
        },
        "modules": {
            "m": { "source": "m.cubin", "sha256": SHA_A }
        },
        "ops": {
            "prep": {
                "params": ["in buffer<i64>", "out buffer<bf16>"],
                "impl": { "launches": [{ "module": "m", "entry": "k0", "block": [64, 1, 1], "grid": [1, 1, 1] }] }
            },
            "branch_a": {
                "params": ["in buffer<bf16>", "out buffer<bf16>"],
                "impl": { "launches": [{ "module": "m", "entry": "k1", "block": [64, 1, 1], "grid": [1, 1, 1] }] }
            },
            "branch_b": {
                "params": ["in buffer<bf16>", "out buffer<bf16>"],
                "impl": { "launches": [{ "module": "m", "entry": "k2", "block": [64, 1, 1], "grid": [1, 1, 1] }] }
            },
            "join": {
                "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<i64>"],
                "impl": { "launches": [{ "module": "m", "entry": "k3", "block": [64, 1, 1], "grid": [1, 1, 1] }] }
            }
        },
        "programs": {
            "step": {
                "batch": { "groups": 4, "rows": 1 },
                "calls": [
                    { "label": "c0", "op": "prep", "args": [{"buf": "token_ids"}, {"buf": "a"}] },
                    { "label": "c1", "op": "branch_a", "args": [{"buf": "a"}, {"buf": "b0"}] },
                    { "label": "c2", "op": "branch_b", "lane": 1, "after": [0], "args": [{"buf": "a"}, {"buf": "b1"}] },
                    { "label": "c3", "op": "join", "after": [1], "args": [{"buf": "b0"}, {"buf": "b1"}, {"buf": "next_token"}] }
                ]
            }
        }
    });
    let m = Verified::from_json(&doc.to_string()).expect("valid multi-lane manifest");
    // Wait: `c2` is listed AFTER `c1`, and `c2` has `after: [0]` which means after all calls on lane 0 BEFORE `c2` (`c0` and `c1`)!
    // Let's also verify a fork right after `c0`: put `c2` (lane 1, after [0]) BEFORE `c1` (lane 0)!
    let doc_parallel = json!({
        "schema_version": 7,
        "model": "parallel-lanes",
        "vars": {
            "tokens": { "max": 64, "axis": "rows" },
            "seqs": { "max": 4, "axis": "groups" }
        },
        "buffers": doc["buffers"],
        "modules": doc["modules"],
        "ops": doc["ops"],
        "programs": {
            "step": {
                "batch": { "groups": 4, "rows": 1 },
                "calls": [
                    { "label": "c0", "op": "prep", "args": [{"buf": "token_ids"}, {"buf": "a"}] },
                    { "label": "c2", "op": "branch_b", "lane": 1, "after": [0], "args": [{"buf": "a"}, {"buf": "b1"}] },
                    { "label": "c1", "op": "branch_a", "args": [{"buf": "a"}, {"buf": "b0"}] },
                    { "label": "c3", "op": "join", "after": [1], "args": [{"buf": "b0"}, {"buf": "b1"}, {"buf": "next_token"}] }
                ]
            }
        }
    });
    let _ = m;
    let mp = Verified::from_json(&doc_parallel.to_string()).expect("valid parallel manifest");
    let mut book = CostBook::new();
    // c0 = 10us, c2 (lane 1) = 40us, c1 (lane 0) = 25us, c3 (join) = 5us -> critical path = 10 + max(40, 25) + 5 = 55us (sum = 80us).
    book.ingest_bench_json(
        &mp,
        &synthetic_bench_json(
            "step",
            1,
            false,
            vec![0],
            BTreeMap::from([("seqs", 1), ("tokens", 1)]),
            &[10.0, 40.0, 25.0, 5.0],
            55.0,
        ),
    )
    .unwrap();

    let est = book
        .estimate_program(
            &mp,
            "tokens",
            "step",
            &BTreeMap::from([("seqs".into(), 1), ("tokens".into(), 1)]),
            1,
            PriceOptions::default(),
        )
        .unwrap();
    assert!((est.sum_calls_us - 80.0).abs() < 1e-6);
    assert!((est.critical_path_us - 55.0).abs() < 1e-6);
    assert!((est.total_us - 55.0).abs() < 1e-6);
}

#[test]
fn memory_budget_calibration_recovers_exact_pages_and_slot_tradeoff() {
    let launches = json!([{
        "module": "base_mod",
        "entry": "k_linear_bf16",
        "block": [128, 1, 1],
        "grid": ["tokens", 1, 1]
    }]);
    let m = build_serving_manifest(launches, SHA_A, true);
    let total_vram = 4 * (1u64 << 30); // 4 GiB

    let cal_overhead =
        MemoryBudget::calibrate_driver_overhead(&m, total_vram, Some(16), 120_000).unwrap();
    let b16 = MemoryBudget::compute(
        &m,
        DeviceMemorySpec {
            total_vram_bytes: total_vram,
            driver_overhead_bytes: cal_overhead,
            active_slots: Some(16),
            explicit_tokens: None,
        },
    )
    .unwrap();
    assert_eq!(b16.raw_pages, 120_000);
    assert_eq!(b16.scheduler_pages, 119_999);
    assert_eq!(b16.provisioned_slots, 17);

    let b64 = MemoryBudget::compute(
        &m,
        DeviceMemorySpec {
            total_vram_bytes: total_vram,
            driver_overhead_bytes: cal_overhead,
            active_slots: Some(64),
            explicit_tokens: None,
        },
    )
    .unwrap();
    assert!(b64.raw_pages < b16.raw_pages);
    assert_eq!(b64.provisioned_slots, 65);
}

#[test]
fn kv_aware_routing_outperforms_round_robin_on_stateful_multi_turn_sessions() {
    let launches = json!([{
        "module": "base_mod",
        "entry": "k_linear_bf16",
        "block": [128, 1, 1],
        "grid": ["tokens", 1, 1]
    }]);
    let m = build_serving_manifest(launches, SHA_A, true);

    let mut book = CostBook::new();
    // Calibrate prefill to be expensive per token (e.g. 64 tokens -> 6400 us, 256 tokens -> 25600 us)
    // so skipping cached multi-turn history yields a large TTFT and throughput win.
    for &(toks, us) in &[(16usize, 1600.0), (64, 6400.0), (256, 25_600.0)] {
        book.ingest_bench_json(
            &m,
            &synthetic_bench_json(
                "prefill",
                toks,
                true,
                vec![0],
                BTreeMap::from([("seqs", 1), ("tokens", toks as u64)]),
                &[100.0, us * 0.4, us * 0.5, 50.0, 50.0],
                us + 200.0,
            ),
        )
        .unwrap();
    }
    for &(b, us) in &[(1usize, 800.0), (2, 950.0), (4, 1200.0), (8, 1600.0)] {
        book.ingest_bench_json(
            &m,
            &synthetic_bench_json(
                "decode_batch",
                1,
                false,
                vec![128; b],
                BTreeMap::from([("seqs", b as u64), ("tokens", b as u64)]),
                &[50.0, us * 0.4, us * 0.5, 40.0, 40.0],
                us + 130.0,
            ),
        )
        .unwrap();
    }

    let trace = RequestTrace::synthetic_sessions(SyntheticSessionConfig {
        sessions: 12,
        turns_per_session: 4,
        system_prompt_tokens: 64,
        user_tokens_per_turn: 48,
        output_tokens_per_turn: 24,
        session_arrival_gap_ms: 15,
        think_time_ms: 40,
        system_groups: 2,
        seed: 7,
    });

    let mut mem = DeviceMemorySpec::with_gib(4.0);
    mem.active_slots = Some(32);

    let endpoints: Vec<EndpointConfig> = (0..4)
        .map(|i| EndpointConfig::standard(format!("ep-{i}"), m.clone(), mem))
        .collect();

    let rr_cfg = ClusterConfig {
        endpoints: endpoints.clone(),
        router: RouterPolicy::RoundRobin,
        arrival_mode: ArrivalMode::ClosedLoop,
        price_options: PriceOptions::default(),
        shared_pool: None,
    };
    let kv_cfg = ClusterConfig {
        endpoints: endpoints.clone(),
        router: RouterPolicy::kv_aware_default(),
        arrival_mode: ArrivalMode::ClosedLoop,
        price_options: PriceOptions::default(),
        shared_pool: None,
    };

    let rep_rr = SimCluster::run(&book, &rr_cfg, &trace).unwrap();
    let rep_kv = SimCluster::run(&book, &kv_cfg, &trace).unwrap();
    let rep_kv_again = SimCluster::run(&book, &kv_cfg, &trace).unwrap();

    // 1. Deterministic bit-identical replay.
    assert_eq!(rep_kv.wall_time_s.to_bits(), rep_kv_again.wall_time_s.to_bits());
    assert_eq!(rep_kv.total_prefix_hit_tokens, rep_kv_again.total_prefix_hit_tokens);

    // 2. Every request completes without rejection.
    assert_eq!(rep_rr.completed_requests, 48);
    assert_eq!(rep_kv.completed_requests, 48);

    // 3. KV-aware router achieves dramatically higher prefix hit rate and faster TTFT/wall time
    //    on stateful multi-turn sessions than round-robin across 4 endpoints.
    assert!(
        rep_kv.prefix_hit_rate > rep_rr.prefix_hit_rate + 0.25,
        "expected kv_aware hit rate ({:.3}) >> round_robin ({:.3})",
        rep_kv.prefix_hit_rate,
        rep_rr.prefix_hit_rate
    );
    assert!(
        rep_kv.ttft.p50_ms < rep_rr.ttft.p50_ms,
        "expected kv_aware TTFT p50 ({:.2}ms) < round_robin ({:.2}ms)",
        rep_kv.ttft.p50_ms,
        rep_rr.ttft.p50_ms
    );
    assert!(
        rep_kv.wall_time_s < rep_rr.wall_time_s,
        "expected kv_aware wall time ({:.3}s) < round_robin ({:.3}s)",
        rep_kv.wall_time_s,
        rep_rr.wall_time_s
    );
}

#[test]
fn mooncake_jsonl_hash_ids_reuse_pages_across_requests() {
    let launches = json!([{
        "module": "base_mod",
        "entry": "k_linear_bf16",
        "block": [128, 1, 1],
        "grid": ["tokens", 1, 1]
    }]);
    let m = build_serving_manifest(launches, SHA_A, false);
    let mut book = CostBook::new();
    book.ingest_bench_json(
        &m,
        &synthetic_bench_json(
            "prefill",
            64,
            true,
            vec![0],
            BTreeMap::from([("seqs", 1), ("tokens", 64)]),
            &[10.0, 200.0, 200.0, 10.0],
            420.0,
        ),
    )
    .unwrap();
    book.ingest_bench_json(
        &m,
        &synthetic_bench_json(
            "decode_batch",
            1,
            false,
            vec![64],
            BTreeMap::from([("seqs", 1), ("tokens", 1)]),
            &[5.0, 20.0, 20.0, 5.0],
            50.0,
        ),
    )
    .unwrap();

    let jsonl = r#"
{"timestamp": 0, "input_length": 80, "output_length": 16, "hash_ids": [101, 102, 103, 104]}
{"timestamp": 50, "input_length": 80, "output_length": 16, "hash_ids": [101, 102, 103, 999]}
"#;
    let trace = RequestTrace::from_jsonl_str(
        "mooncake-mini",
        jsonl,
        JsonlTraceOptions {
            hash_block_tokens: 16,
            timestamp_unit_ns: 1_000_000,
        },
    )
    .unwrap();

    let cfg = ClusterConfig {
        endpoints: vec![EndpointConfig::standard(
            "ep-0",
            m,
            DeviceMemorySpec::with_gib(4.0),
        )],
        router: RouterPolicy::RoundRobin,
        arrival_mode: ArrivalMode::OpenLoop,
        price_options: PriceOptions::default(),
        shared_pool: None,
    };
    let rep = SimCluster::run(&book, &cfg, &trace).unwrap();
    assert_eq!(rep.completed_requests, 2);
    // Request 2 shares 3 blocks of 16 tokens = 48 tokens with Request 1's stateless KV pages!
    assert_eq!(rep.total_prefix_hit_tokens, 48);
}

#[test]
fn closed_loop_serializes_turns_even_when_gpu_turn_exceeds_nominal_gap() {
    let launches = json!([{
        "module": "base_mod",
        "entry": "k_linear_bf16",
        "block": [128, 1, 1],
        "grid": ["tokens", 1, 1]
    }]);
    let m = build_serving_manifest(launches, SHA_A, true);
    let mut book = CostBook::new();
    // Make each prefill/decode step take 50ms (50,000 µs) so Turn 0 takes >500ms on GPU,
    // far longer than `think_time_ms: 1` + nominal open-loop timestamps.
    book.ingest_bench_json(
        &m,
        &synthetic_bench_json(
            "prefill",
            64,
            true,
            vec![0],
            BTreeMap::from([("seqs", 1), ("tokens", 64)]),
            &[10_000.0, 20_000.0, 15_000.0, 3_000.0, 2_000.0],
            50_000.0,
        ),
    )
    .unwrap();
    book.ingest_bench_json(
        &m,
        &synthetic_bench_json(
            "decode_batch",
            1,
            false,
            vec![64],
            BTreeMap::from([("seqs", 8), ("tokens", 8)]),
            &[5_000.0, 20_000.0, 15_000.0, 5_000.0, 5_000.0],
            50_000.0,
        ),
    )
    .unwrap();

    let trace = RequestTrace::synthetic_sessions(SyntheticSessionConfig {
        sessions: 2,
        turns_per_session: 3,
        system_prompt_tokens: 32,
        user_tokens_per_turn: 32,
        output_tokens_per_turn: 10,
        session_arrival_gap_ms: 5,
        think_time_ms: 1,
        system_groups: 1,
        seed: 99,
    });

    let cfg = ClusterConfig {
        endpoints: vec![EndpointConfig::standard(
            "ep-0",
            m,
            DeviceMemorySpec::with_gib(4.0),
        )],
        router: RouterPolicy::kv_aware_default(),
        arrival_mode: ArrivalMode::ClosedLoop,
        price_options: PriceOptions::default(),
        shared_pool: None,
    };
    let rep = SimCluster::run(&book, &cfg, &trace).unwrap();
    assert_eq!(rep.completed_requests, 6);
    // Because ClosedLoop waits for Turn t to retire before issuing Turn t+1,
    // all 4 follow-up turns must hit their retired stateful checkpoint!
    assert!(
        rep.prefix_hit_rate > 0.50,
        "expected >50% stateful prefix hit rate under ClosedLoop, got {:.3}",
        rep.prefix_hit_rate
    );
}
