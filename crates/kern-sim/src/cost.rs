//! Content-addressed operator cost book and `lane`/`after` critical-path pricer.
//!
//! [`CostBook`] learns operator execution times from `kern bench` JSON reports
//! or traced `steps.parquet` + `kernels.parquet` pairs and prices any verified
//! manifest at concrete call variables `(vars, sum_kv)` without a GPU.
//!
//! Matching is structural rather than by op or module name:
//! 1. At concrete `vars`, `op.imp.launches` is filtered to launches whose
//!    `when` range holds (`when` itself is stripped from the active list, so
//!    rewritten range splits compose cleanly across manifests).
//! 2. Launch geometry (`grid`, `shared_mem`) and `LaunchArg`s (including
//!    forwarded scalar/var/expr call arguments and `pack` fields) are resolved
//!    against `vars`.
//! 3. Two keys are produced per active call:
//!    - [`MatchTier::ExactSha256`]: pins every kernel launch by its module's
//!      `sha256` digest;
//!    - [`MatchTier::EntryAbi`]: matches by kernel `entry`, geometry, ABI, and
//!      resolved arguments when cubins were repackaged into a unified module.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::path::Path;

use anyhow::{bail, ensure, Context, Result};
use arrow_array::{Array, BooleanArray, Int64Array, RecordBatch, StringArray, UInt64Array};
use kern_manifest::types::{
    Arg, BufferKind, Call, DType, Dir, Expr, FieldSrc, Launch, LaunchArg, Manifest, Op, ParamType,
};
use kern_manifest::{Protocol, Verified};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Confidence tier of a call's cost estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchTier {
    /// Matched active launches with identical cubin `sha256`, geometry, and args.
    ExactSha256,
    /// Matched active launches by `(entry, block, grid, shared_mem, params, args)`
    /// across different cubin `sha256` digests.
    EntryAbi,
    /// Estimated from a roofline/analytic fallback constant when uncalibrated.
    AnalyticFallback,
    /// Uncalibrated and no fallback enabled (`0.0 µs` reported alongside coverage).
    Uncalibrated,
}

/// Pricing options controlling fallback tiers.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PriceOptions {
    /// Allow [`MatchTier::EntryAbi`] when [`MatchTier::ExactSha256`] misses.
    pub allow_abi_fallback: bool,
    /// Optional constant per-launch fallback (in µs) when uncalibrated.
    pub uncalibrated_launch_us: Option<f64>,
}

impl Default for PriceOptions {
    fn default() -> Self {
        Self {
            allow_abi_fallback: true,
            uncalibrated_launch_us: None,
        }
    }
}

/// Priced result of one call inside a program.
#[derive(Debug, Clone, Serialize)]
pub struct CallEstimate {
    pub index: usize,
    pub label: Option<String>,
    pub op: String,
    pub us: f64,
    pub active_launches: usize,
    pub kv_dependent: bool,
    pub tier: MatchTier,
}

/// Priced result of one program invocation at concrete `(vars, sum_kv)`.
#[derive(Debug, Clone, Serialize)]
pub struct ProgramEstimate {
    pub program: String,
    /// Whole-program time including inter-launch gap, in microseconds.
    pub total_us: f64,
    /// Critical-path kernel span across `lane`/`after` dependencies, in µs.
    pub critical_path_us: f64,
    /// Sum of all call durations (exceeds `critical_path_us` when side lanes overlap).
    pub sum_calls_us: f64,
    /// Estimated inter-launch / graph gap added to `critical_path_us`, in µs.
    pub gap_us: f64,
    pub active_launches: usize,
    pub calls_by_tier: BTreeMap<MatchTier, usize>,
    pub us_by_tier: BTreeMap<MatchTier, f64>,
    pub calls: Vec<CallEstimate>,
}

impl ProgramEstimate {
    fn known_call_share(&self) -> f64 {
        let total_calls: usize = self.calls_by_tier.values().sum();
        if total_calls == 0 {
            return 1.0;
        }
        let uncal_calls = self
            .calls_by_tier
            .get(&MatchTier::Uncalibrated)
            .copied()
            .unwrap_or(0);
        let uncal_us = self
            .us_by_tier
            .get(&MatchTier::Uncalibrated)
            .copied()
            .unwrap_or(0.0);
        if uncal_calls > 0 && uncal_us <= 0.0 {
            (total_calls - uncal_calls) as f64 / total_calls as f64
        } else {
            1.0
        }
    }

    /// Share of attributed call time matched at [`MatchTier::ExactSha256`].
    pub fn exact_time_coverage(&self) -> f64 {
        let known = self.known_call_share();
        if self.sum_calls_us <= 0.0 {
            return known;
        }
        let exact = self
            .us_by_tier
            .get(&MatchTier::ExactSha256)
            .copied()
            .unwrap_or(0.0);
        (exact / self.sum_calls_us) * known
    }

    /// Share of attributed call time matched at [`MatchTier::ExactSha256`] or [`MatchTier::EntryAbi`].
    pub fn calibrated_time_coverage(&self) -> f64 {
        let known = self.known_call_share();
        if self.sum_calls_us <= 0.0 {
            return known;
        }
        let exact = self
            .us_by_tier
            .get(&MatchTier::ExactSha256)
            .copied()
            .unwrap_or(0.0);
        let abi = self
            .us_by_tier
            .get(&MatchTier::EntryAbi)
            .copied()
            .unwrap_or(0.0);
        ((exact + abi) / self.sum_calls_us) * known
    }
}

/// Accumulator `(sum_us, count)` for unbiased pooling of per-call `p50`s.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
struct Accum {
    sum_us: f64,
    count: u64,
}

impl Accum {
    fn push(&mut self, us: f64) {
        self.sum_us += us;
        self.count += 1;
    }

    fn mean(self) -> Option<f64> {
        (self.count > 0).then_some(self.sum_us / self.count as f64)
    }
}

/// 1D curve over `sum_kv` (or single bucket at `0` for KV-independent calls).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct KvCurve {
    by_kv: BTreeMap<u64, Accum>,
}

impl KvCurve {
    fn insert(&mut self, kv_key: u64, us: f64) {
        self.by_kv.entry(kv_key).or_default().push(us);
    }

    fn eval(&self, sum_kv: u64, kv_dependent: bool) -> Option<f64> {
        if !kv_dependent {
            let mut total_us = 0.0;
            let mut total_n = 0u64;
            for acc in self.by_kv.values() {
                total_us += acc.sum_us;
                total_n += acc.count;
            }
            return (total_n > 0).then_some(total_us / total_n as f64);
        }
        if let Some(acc) = self.by_kv.get(&sum_kv) {
            if let Some(v) = acc.mean() {
                return Some(v);
            }
        }
        let pts: Vec<(u64, f64)> = self
            .by_kv
            .iter()
            .filter_map(|(&kv, acc)| acc.mean().map(|y| (kv, y)))
            .collect();
        interp_1d(&pts, sum_kv)
    }
}

/// 2D surface over `(row_tokens, sum_kv)` for unseen row counts (e.g. arbitrary prefill chunks).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct RowSurface {
    by_rows: BTreeMap<u64, KvCurve>,
}

impl RowSurface {
    fn insert(&mut self, rows: u64, kv_key: u64, us: f64) {
        self.by_rows.entry(rows).or_default().insert(kv_key, us);
    }

    fn eval(&self, rows: u64, sum_kv: u64, kv_dependent: bool) -> Option<f64> {
        if let Some(curve) = self.by_rows.get(&rows) {
            if let Some(v) = curve.eval(sum_kv, kv_dependent) {
                return Some(v);
            }
        }
        let pts: Vec<(u64, f64)> = self
            .by_rows
            .iter()
            .filter_map(|(&r, curve)| curve.eval(sum_kv, kv_dependent).map(|y| (r, y)))
            .collect();
        interp_1d(&pts, rows)
    }
}

/// Summary returned when ingesting a calibration source into [`CostBook`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct IngestStats {
    pub scenarios: usize,
    pub call_observations: usize,
}

/// Content-addressed operator cost table calibrated from `bench.json` and/or Parquet traces.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CostBook {
    exact_eval: BTreeMap<String, KvCurve>,
    abi_eval: BTreeMap<String, KvCurve>,
    exact_tpl: BTreeMap<String, RowSurface>,
    abi_tpl: BTreeMap<String, RowSurface>,
    graph_gap_per_launch: Accum,
    eager_gap_per_launch: Accum,
    sources_ingested: usize,
}

impl CostBook {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn sources_ingested(&self) -> usize {
        self.sources_ingested
    }

    pub fn unique_exact_signatures(&self) -> usize {
        self.exact_eval.len()
    }

    /// Ingest a `kern bench` report JSON alongside the [`Verified`] manifest it measured.
    pub fn ingest_bench_file(&mut self, manifest: &Verified, bench_path: &Path) -> Result<IngestStats> {
        let bytes = std::fs::read(bench_path)
            .with_context(|| format!("reading bench report {}", bench_path.display()))?;
        let report: BenchReport = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing bench report {}", bench_path.display()))?;
        self.ingest_bench_data(manifest, &report)
    }

    /// Ingest parsed `kern bench` JSON text alongside its [`Verified`] manifest.
    pub fn ingest_bench_json(&mut self, manifest: &Verified, bench_json: &str) -> Result<IngestStats> {
        let report: BenchReport = serde_json::from_str(bench_json).context("parsing bench report JSON")?;
        self.ingest_bench_data(manifest, &report)
    }

    fn ingest_bench_data(&mut self, manifest: &Verified, report: &BenchReport) -> Result<IngestStats> {
        let protocol = Protocol::check_unsampled(manifest)?;
        let rows_var = protocol.rows.var.as_str();
        let mut sig_cache = SignatureCache::new();

        // Group multi-rank scenario records by scenario id and pick the slowest rank,
        // matching `kern bench`'s own pricing contract.
        let mut by_scenario: BTreeMap<usize, Vec<&BenchScenarioRecord>> = BTreeMap::new();
        let mut id_order: BTreeMap<String, usize> = BTreeMap::new();
        for rec in &report.scenarios {
            let next_idx = id_order.len();
            let idx = *id_order.entry(rec.scenario.id.clone()).or_insert(next_idx);
            by_scenario.entry(idx).or_default().push(rec);
        }

        let mut scenarios_n = 0usize;
        let mut obs_n = 0usize;

        for recs in by_scenario.values() {
            let slow = recs
                .iter()
                .copied()
                .max_by(|a, b| a.graph.stats.p50.total_cmp(&b.graph.stats.p50))
                .expect("non-empty scenario group");
            let prog = manifest.programs.get(&slow.program).with_context(|| {
                format!("bench report names program `{}`, absent from manifest", slow.program)
            })?;
            ensure!(
                prog.calls.len() == slow.calls.len(),
                "program `{}` has {} calls in manifest, {} in bench report",
                slow.program,
                prog.calls.len(),
                slow.calls.len()
            );

            let (kv_flags, cached_sigs) =
                sig_cache.get_or_build(manifest, rows_var, &slow.program, &slow.vars)?;

            let sum_kv: u64 = slow
                .scenario
                .context
                .iter()
                .map(|&c| c as u64 + slow.scenario.rows as u64)
                .sum();
            let row_val = slow.vars.get(rows_var).copied().unwrap_or(slow.scenario.rows as u64);

            let mut call_durations = Vec::with_capacity(prog.calls.len());
            let mut total_launches = 0usize;

            for (i, (sigs, timed)) in cached_sigs.iter().zip(&slow.calls).enumerate() {
                let us = timed.in_program.stats.p50;
                call_durations.push(us);
                total_launches += sigs.active_launches;
                if sigs.active_launches == 0 {
                    continue;
                }
                let kv_dep = kv_flags[i];
                let kv_key = if kv_dep { sum_kv } else { 0 };
                self.exact_eval.entry(sigs.exact_eval.clone()).or_default().insert(kv_key, us);
                self.abi_eval.entry(sigs.abi_eval.clone()).or_default().insert(kv_key, us);
                self.exact_tpl.entry(sigs.exact_tpl.clone()).or_default().insert(row_val, kv_key, us);
                self.abi_tpl.entry(sigs.abi_tpl.clone()).or_default().insert(row_val, kv_key, us);
                obs_n += 1;
            }

            if total_launches > 0 {
                let crit = critical_path_us(&prog.calls, &call_durations);
                let gap_per_launch = (slow.graph.stats.p50 - crit) / total_launches as f64;
                if prog.graph {
                    self.graph_gap_per_launch.push(gap_per_launch);
                } else {
                    self.eager_gap_per_launch.push(gap_per_launch);
                }
            }
            scenarios_n += 1;
        }

        self.sources_ingested += 1;
        Ok(IngestStats {
            scenarios: scenarios_n,
            call_observations: obs_n,
        })
    }

    /// Ingest a closed `(steps.parquet, kernels.parquet)` trace pair alongside
    /// the [`Verified`] manifest that produced it.
    pub fn ingest_parquet_pair(
        &mut self,
        manifest: &Verified,
        steps_path: &Path,
        kernels_path: &Path,
    ) -> Result<IngestStats> {
        let protocol = Protocol::check_unsampled(manifest)?;
        let rows_var = protocol.rows.var.as_str();

        let steps = read_traced_steps(steps_path)?;
        let kernels_by_seq = read_traced_kernels(kernels_path)?;

        // Map `(program, label)` -> call index (when labels are unique; if an
        // unlabeled call uses `op#idx`, `Sites` matches `format!("{}#{i}", c.op)`).
        let mut label_map: BTreeMap<String, BTreeMap<String, Vec<usize>>> = BTreeMap::new();
        for (pname, prog) in &manifest.programs {
            let mut m: BTreeMap<String, Vec<usize>> = BTreeMap::new();
            for (i, c) in prog.calls.iter().enumerate() {
                let lbl = c.label.clone().unwrap_or_else(|| format!("{}#{i}", c.op));
                m.entry(lbl).or_default().push(i);
            }
            label_map.insert(pname.clone(), m);
        }

        // Group steps by `(program, vars, kv_bucket)` to compute clean p50 per call
        // before inserting into the cost book, rejecting single-step scheduling jitter.
        type GroupKey = (String, BTreeMap<String, u64>, u64);
        let mut grouped_calls: BTreeMap<GroupKey, Vec<Vec<f64>>> = BTreeMap::new();
        let mut grouped_step_us: BTreeMap<GroupKey, Vec<f64>> = BTreeMap::new();
        let mut grouped_sum_kv: BTreeMap<GroupKey, Vec<u64>> = BTreeMap::new();

        for step in &steps {
            let Some(prog) = manifest.programs.get(&step.program) else {
                continue;
            };
            let Some(krows) = kernels_by_seq.get(&step.seq) else {
                continue;
            };
            let Some(lmap) = label_map.get(&step.program) else {
                continue;
            };
            // Skip steps where repeated identical labels prevent unambiguous per-call attribution.
            let mut per_call_ns = vec![0i64; prog.calls.len()];
            let mut cursors: BTreeMap<&str, usize> = BTreeMap::new();
            let mut complete = true;
            for (call_lbl, dur_ns) in krows {
                let Some(indices) = lmap.get(call_lbl.as_str()) else {
                    complete = false;
                    break;
                };
                let idx = if indices.len() == 1 {
                    indices[0]
                } else {
                    let cur = cursors.entry(call_lbl.as_str()).or_insert(0);
                    let chosen = indices.get(*cur).copied().unwrap_or(indices[indices.len() - 1]);
                    *cur += 1;
                    chosen
                };
                per_call_ns[idx] += *dur_ns;
            }
            if !complete {
                continue;
            }
            let raw_kv = step.kv.unwrap_or(0);
            let kv_bucket = if raw_kv == 0 { 0 } else { raw_kv.next_power_of_two() };
            let key: GroupKey = (step.program.clone(), step.vars.clone(), kv_bucket);
            let slots = grouped_calls
                .entry(key.clone())
                .or_insert_with(|| vec![Vec::new(); prog.calls.len()]);
            for (slot, &ns) in slots.iter_mut().zip(&per_call_ns) {
                slot.push(ns as f64 / 1e3);
            }
            grouped_step_us.entry(key.clone()).or_default().push(step.dur_ns as f64 / 1e3);
            grouped_sum_kv.entry(key).or_default().push(raw_kv);
        }

        let mut sig_cache = SignatureCache::new();
        let mut scenarios_n = 0usize;
        let mut obs_n = 0usize;

        for (key, call_series) in &grouped_calls {
            let (pname, vars, _) = key;
            let prog = &manifest.programs[pname];
            let (kv_flags, cached_sigs) =
                sig_cache.get_or_build(manifest, rows_var, pname, vars)?;
            let raw_kvs = &grouped_sum_kv[key];
            let mean_kv = if raw_kvs.is_empty() {
                0
            } else {
                (raw_kvs.iter().sum::<u64>() + raw_kvs.len() as u64 / 2) / raw_kvs.len() as u64
            };
            let row_val = vars.get(rows_var).copied().unwrap_or(1);

            let mut p50_calls = Vec::with_capacity(prog.calls.len());
            let mut total_launches = 0usize;

            for (i, (sigs, samples)) in cached_sigs.iter().zip(call_series).enumerate() {
                let p50 = median_f64(samples);
                p50_calls.push(p50);
                total_launches += sigs.active_launches;
                if sigs.active_launches == 0 {
                    continue;
                }
                let kv_key = if kv_flags[i] { mean_kv } else { 0 };
                self.exact_eval.entry(sigs.exact_eval.clone()).or_default().insert(kv_key, p50);
                self.abi_eval.entry(sigs.abi_eval.clone()).or_default().insert(kv_key, p50);
                self.exact_tpl.entry(sigs.exact_tpl.clone()).or_default().insert(row_val, kv_key, p50);
                self.abi_tpl.entry(sigs.abi_tpl.clone()).or_default().insert(row_val, kv_key, p50);
                obs_n += 1;
            }

            if total_launches > 0 {
                let step_p50 = median_f64(&grouped_step_us[key]);
                let crit = critical_path_us(&prog.calls, &p50_calls);
                let gap_per_launch = (step_p50 - crit) / total_launches as f64;
                if prog.graph {
                    self.graph_gap_per_launch.push(gap_per_launch);
                } else {
                    self.eager_gap_per_launch.push(gap_per_launch);
                }
            }
            scenarios_n += 1;
        }

        self.sources_ingested += 1;
        Ok(IngestStats {
            scenarios: scenarios_n,
            call_observations: obs_n,
        })
    }

    /// Estimate the latency and per-call breakdown of `program_name` at `(vars, sum_kv)`.
    ///
    /// `sum_kv` is the sum of all active sequences' lengths *including* the rows
    /// staged by this call (`sum_i (context_i + rows_i)`), matching `steps.parquet`.
    pub fn estimate_program(
        &self,
        manifest: &Manifest,
        rows_var: &str,
        program_name: &str,
        vars: &BTreeMap<String, u64>,
        sum_kv: u64,
        opts: PriceOptions,
    ) -> Result<ProgramEstimate> {
        let mut cache = SignatureCache::new();
        self.estimate_program_cached(
            manifest,
            rows_var,
            program_name,
            vars,
            sum_kv,
            opts,
            &mut cache,
        )
    }

    /// Like [`Self::estimate_program`], reusing `sig_cache` across repeated queries on `manifest`.
    #[allow(clippy::too_many_arguments)]
    pub fn estimate_program_cached(
        &self,
        manifest: &Manifest,
        rows_var: &str,
        program_name: &str,
        vars: &BTreeMap<String, u64>,
        sum_kv: u64,
        opts: PriceOptions,
        sig_cache: &mut SignatureCache,
    ) -> Result<ProgramEstimate> {
        let prog = manifest
            .programs
            .get(program_name)
            .with_context(|| format!("unknown program `{program_name}`"))?;
        let (kv_flags, cached_sigs) =
            sig_cache.get_or_build(manifest, rows_var, program_name, vars)?;
        let row_val = vars.get(rows_var).copied().unwrap_or(1);

        let mut calls = Vec::with_capacity(prog.calls.len());
        let mut durations = Vec::with_capacity(prog.calls.len());
        let mut active_launches = 0usize;
        let mut calls_by_tier: BTreeMap<MatchTier, usize> = BTreeMap::new();
        let mut us_by_tier: BTreeMap<MatchTier, f64> = BTreeMap::new();

        for (index, (call, sigs)) in prog.calls.iter().zip(cached_sigs).enumerate() {
            let kv_dep = kv_flags[index];
            let (us, tier) = if sigs.active_launches == 0 {
                (0.0, MatchTier::ExactSha256)
            } else if let Some(v) = self
                .exact_eval
                .get(&sigs.exact_eval)
                .and_then(|c| c.eval(sum_kv, kv_dep))
                .or_else(|| {
                    self.exact_tpl
                        .get(&sigs.exact_tpl)
                        .and_then(|s| s.eval(row_val, sum_kv, kv_dep))
                })
            {
                (v, MatchTier::ExactSha256)
            } else if let Some(v) = opts.allow_abi_fallback.then(|| {
                self.abi_eval
                    .get(&sigs.abi_eval)
                    .and_then(|c| c.eval(sum_kv, kv_dep))
                    .or_else(|| {
                        self.abi_tpl
                            .get(&sigs.abi_tpl)
                            .and_then(|s| s.eval(row_val, sum_kv, kv_dep))
                    })
            }).flatten()
            {
                (v, MatchTier::EntryAbi)
            } else if let Some(per_launch) = opts.uncalibrated_launch_us {
                (per_launch * sigs.active_launches as f64, MatchTier::AnalyticFallback)
            } else {
                (0.0, MatchTier::Uncalibrated)
            };

            active_launches += sigs.active_launches;
            durations.push(us);
            if sigs.active_launches > 0 {
                *calls_by_tier.entry(tier).or_default() += 1;
                *us_by_tier.entry(tier).or_default() += us;
            }
            calls.push(CallEstimate {
                index,
                label: call.label.clone(),
                op: call.op.clone(),
                us,
                active_launches: sigs.active_launches,
                kv_dependent: kv_dep,
                tier,
            });
        }

        let critical_path_us = critical_path_us(&prog.calls, &durations);
        let sum_calls_us: f64 = durations.iter().sum();
        let gap_per_launch = if prog.graph {
            self.graph_gap_per_launch
                .mean()
                .or_else(|| self.eager_gap_per_launch.mean())
                .unwrap_or(0.0)
        } else {
            self.eager_gap_per_launch
                .mean()
                .or_else(|| self.graph_gap_per_launch.mean())
                .unwrap_or(0.0)
        };
        let gap_us = gap_per_launch * active_launches as f64;
        let total_us = (critical_path_us + gap_us).max(0.0);

        Ok(ProgramEstimate {
            program: program_name.to_string(),
            total_us,
            critical_path_us,
            sum_calls_us,
            gap_us,
            active_launches,
            calls_by_tier,
            us_by_tier,
            calls,
        })
    }
}

/// Fast reusable handle for pricing a `(program, vars)` across many `sum_kv` values
/// without recomputing active-launch SHA-256 signatures on every step.
pub struct PreparedProgram {
    graph: bool,
    row_val: u64,
    opts: PriceOptions,
    active_launches: usize,
    has_side_lanes: bool,
    calls_meta: Vec<Call>,
    base_durations: Vec<f64>,
    base_sum_us: f64,
    base_calls_by_tier: BTreeMap<MatchTier, usize>,
    base_us_by_tier: BTreeMap<MatchTier, f64>,
    kv_sites: Vec<(usize, CallSignatures)>,
}

/// Lightweight step timing returned by [`PreparedProgram::eval`].
#[derive(Debug, Clone)]
pub struct StepTiming {
    pub total_us: f64,
    pub critical_path_us: f64,
    pub sum_calls_us: f64,
    pub us_by_tier: BTreeMap<MatchTier, f64>,
}

impl CostBook {
    /// Pre-compute active-launch signatures and KV-independent call costs for
    /// `(program_name, vars)` so repeated evaluations across `sum_kv` run in O(#kv_calls).
    pub fn prepare_program(
        &self,
        manifest: &Manifest,
        rows_var: &str,
        program_name: &str,
        vars: &BTreeMap<String, u64>,
        opts: PriceOptions,
    ) -> Result<PreparedProgram> {
        let prog = manifest
            .programs
            .get(program_name)
            .with_context(|| format!("unknown program `{program_name}`"))?;
        let kv_flags = program_kv_dependence(manifest, program_name);
        let row_val = vars.get(rows_var).copied().unwrap_or(1);
        let has_side_lanes = prog.calls.iter().any(|c| c.lane != 0 || !c.after.is_empty());

        let mut active_launches = 0usize;
        let mut base_durations = vec![0.0f64; prog.calls.len()];
        let mut base_sum_us = 0.0f64;
        let mut base_calls_by_tier: BTreeMap<MatchTier, usize> = BTreeMap::new();
        let mut base_us_by_tier: BTreeMap<MatchTier, f64> = BTreeMap::new();
        let mut kv_sites = Vec::new();

        for (i, call) in prog.calls.iter().enumerate() {
            let sigs = call_signatures(manifest, call, vars, rows_var)?;
            active_launches += sigs.active_launches;
            if sigs.active_launches == 0 {
                continue;
            }
            if kv_flags[i] {
                kv_sites.push((i, sigs));
            } else {
                let (us, tier) = self.lookup_call(&sigs, row_val, 0, false, opts);
                base_durations[i] = us;
                base_sum_us += us;
                *base_calls_by_tier.entry(tier).or_default() += 1;
                *base_us_by_tier.entry(tier).or_default() += us;
            }
        }

        Ok(PreparedProgram {
            graph: prog.graph,
            row_val,
            opts,
            active_launches,
            has_side_lanes,
            calls_meta: prog.calls.clone(),
            base_durations,
            base_sum_us,
            base_calls_by_tier,
            base_us_by_tier,
            kv_sites,
        })
    }

    fn lookup_call(
        &self,
        sigs: &CallSignatures,
        row_val: u64,
        sum_kv: u64,
        kv_dep: bool,
        opts: PriceOptions,
    ) -> (f64, MatchTier) {
        if sigs.active_launches == 0 {
            return (0.0, MatchTier::ExactSha256);
        }
        if let Some(v) = self
            .exact_eval
            .get(&sigs.exact_eval)
            .and_then(|c| c.eval(sum_kv, kv_dep))
            .or_else(|| {
                self.exact_tpl
                    .get(&sigs.exact_tpl)
                    .and_then(|s| s.eval(row_val, sum_kv, kv_dep))
            })
        {
            return (v, MatchTier::ExactSha256);
        }
        if opts.allow_abi_fallback {
            if let Some(v) = self
                .abi_eval
                .get(&sigs.abi_eval)
                .and_then(|c| c.eval(sum_kv, kv_dep))
                .or_else(|| {
                    self.abi_tpl
                        .get(&sigs.abi_tpl)
                        .and_then(|s| s.eval(row_val, sum_kv, kv_dep))
                })
            {
                return (v, MatchTier::EntryAbi);
            }
        }
        if let Some(per_launch) = opts.uncalibrated_launch_us {
            (per_launch * sigs.active_launches as f64, MatchTier::AnalyticFallback)
        } else {
            (0.0, MatchTier::Uncalibrated)
        }
    }
}

impl PreparedProgram {
    pub fn eval(&self, book: &CostBook, sum_kv: u64) -> StepTiming {
        let _ = &self.base_calls_by_tier;
        let mut us_by_tier = self.base_us_by_tier.clone();
        let mut sum_calls_us = self.base_sum_us;

        let critical_path_us = if !self.has_side_lanes {
            for (_, sigs) in &self.kv_sites {
                let (us, tier) = book.lookup_call(sigs, self.row_val, sum_kv, true, self.opts);
                sum_calls_us += us;
                *us_by_tier.entry(tier).or_default() += us;
            }
            sum_calls_us
        } else {
            let mut durs = self.base_durations.clone();
            for (idx, sigs) in &self.kv_sites {
                let (us, tier) = book.lookup_call(sigs, self.row_val, sum_kv, true, self.opts);
                durs[*idx] = us;
                sum_calls_us += us;
                *us_by_tier.entry(tier).or_default() += us;
            }
            critical_path_us(&self.calls_meta, &durs)
        };

        let gap_per_launch = if self.graph {
            book.graph_gap_per_launch
                .mean()
                .or_else(|| book.eager_gap_per_launch.mean())
                .unwrap_or(0.0)
        } else {
            book.eager_gap_per_launch
                .mean()
                .or_else(|| book.graph_gap_per_launch.mean())
                .unwrap_or(0.0)
        };
        let gap_us = gap_per_launch * self.active_launches as f64;
        StepTiming {
            total_us: (critical_path_us + gap_us).max(0.0),
            critical_path_us,
            sum_calls_us,
            us_by_tier,
        }
    }
}

/// Critical-path makespan of a program's calls respecting `lane` and `after` barriers.
pub fn critical_path_us(calls: &[Call], durations_us: &[f64]) -> f64 {
    assert_eq!(calls.len(), durations_us.len());
    let max_lane = calls.iter().map(|c| c.lane as usize).max().unwrap_or(0);
    let mut lane_ready = vec![0.0f64; max_lane + 1];
    for (call, &dur) in calls.iter().zip(durations_us) {
        let lane = call.lane as usize;
        let mut start = lane_ready[lane];
        for &dep in &call.after {
            if let Some(&dep_end) = lane_ready.get(dep as usize) {
                start = start.max(dep_end);
            }
        }
        lane_ready[lane] = start + dur.max(0.0);
    }
    lane_ready.into_iter().fold(0.0f64, f64::max)
}

/// Structural classification of which calls in `program_name` depend on KV cache depth.
///
/// Marks calls that bind a paged `State` (`bytes_per_token > 0`) or consume
/// `Fill::SeqLen` / paged-state page-table inputs (directly or through integer
/// workspace/carry plan buffers).
pub fn program_kv_dependence(manifest: &Manifest, program_name: &str) -> Vec<bool> {
    let Some(prog) = manifest.programs.get(program_name) else {
        return Vec::new();
    };
    let mut tainted_bufs: BTreeSet<&str> = BTreeSet::new();
    for (name, b) in &manifest.buffers {
        if b.kind != BufferKind::Input {
            continue;
        }
        let is_seq_len = b.fill == Some(kern_manifest::types::Fill::SeqLen);
        let is_page_table = b
            .domain
            .as_ref()
            .and_then(|d| d.index_into.as_deref())
            .and_then(|s| manifest.states.get(s))
            .is_some_and(|st| st.bytes_per_token > 0);
        if is_seq_len || is_page_table {
            tainted_bufs.insert(name.as_str());
        }
    }

    let mut out = Vec::with_capacity(prog.calls.len());
    for call in &prog.calls {
        let Some(op) = manifest.ops.get(&call.op) else {
            out.push(false);
            continue;
        };
        let mut reads_paged_state = false;
        let mut reads_tainted_buf = false;

        for (idx, arg) in call.args.iter().enumerate() {
            match arg {
                Arg::State { state, .. } => {
                    if manifest.states.get(state).is_some_and(|s| s.bytes_per_token > 0) {
                        reads_paged_state = true;
                    }
                }
                Arg::Buf { buf, .. } => {
                    let dir = op.params.get(idx).and_then(|p| p.dir()).unwrap_or(Dir::In);
                    if matches!(dir, Dir::In | Dir::InOut) && tainted_bufs.contains(buf.as_str()) {
                        reads_tainted_buf = true;
                    }
                }
                _ => {}
            }
        }

        if reads_tainted_buf {
            for (idx, arg) in call.args.iter().enumerate() {
                if let Arg::Buf { buf, .. } = arg {
                    let dir = op.params.get(idx).and_then(|p| p.dir());
                    if matches!(dir, Some(Dir::Out | Dir::InOut)) {
                        if let Some(b) = manifest.buffers.get(buf) {
                            if matches!(b.kind, BufferKind::Workspace | BufferKind::Carry)
                                && matches!(b.dtype, DType::I32 | DType::U32 | DType::I64 | DType::U64)
                            {
                                tainted_bufs.insert(buf.as_str());
                            }
                        }
                    }
                }
            }
        }

        out.push(reads_paged_state || reads_tainted_buf);
    }
    out
}

/// Per-manifest memoization table for `(program, vars)` active-launch signatures
/// and KV-dependence flags across repeated KV-depth queries.
#[derive(Default)]
pub struct SignatureCache {
    kv_flags: BTreeMap<String, Vec<bool>>,
    sigs: BTreeMap<(String, BTreeMap<String, u64>), Vec<CallSignatures>>,
}

impl SignatureCache {
    pub fn new() -> Self {
        Self::default()
    }

    fn get_or_build(
        &mut self,
        manifest: &Manifest,
        rows_var: &str,
        program_name: &str,
        vars: &BTreeMap<String, u64>,
    ) -> Result<(&[bool], &[CallSignatures])> {
        let prog = manifest
            .programs
            .get(program_name)
            .with_context(|| format!("unknown program `{program_name}`"))?;
        if !self.kv_flags.contains_key(program_name) {
            self.kv_flags.insert(
                program_name.to_string(),
                program_kv_dependence(manifest, program_name),
            );
        }
        let key = (program_name.to_string(), vars.clone());
        if let std::collections::btree_map::Entry::Vacant(e) = self.sigs.entry(key.clone()) {
            let mut built = Vec::with_capacity(prog.calls.len());
            for call in &prog.calls {
                built.push(call_signatures(manifest, call, vars, rows_var)?);
            }
            e.insert(built);
        }
        Ok((
            self.kv_flags[program_name].as_slice(),
            self.sigs[&key].as_slice(),
        ))
    }
}

#[derive(Clone)]
struct CallSignatures {
    active_launches: usize,
    exact_eval: String,
    abi_eval: String,
    exact_tpl: String,
    abi_tpl: String,
}

fn call_signatures(
    manifest: &Manifest,
    call: &Call,
    vars: &BTreeMap<String, u64>,
    rows_var: &str,
) -> Result<CallSignatures> {
    let op = manifest
        .ops
        .get(&call.op)
        .with_context(|| format!("unknown op `{}`", call.op))?;

    let active: Vec<&Launch> = op
        .imp
        .launches
        .iter()
        .filter(|l| {
            l.when().is_none_or(|w| {
                vars.get(&w.var)
                    .copied()
                    .unwrap_or(kern_manifest::types::Var::MIN)
                    >= w.min.unwrap_or(0)
                    && w.max
                        .is_none_or(|hi| vars.get(&w.var).copied().unwrap_or(1) <= hi)
            })
        })
        .collect();

    if active.is_empty() {
        return Ok(CallSignatures {
            active_launches: 0,
            exact_eval: String::new(),
            abi_eval: String::new(),
            exact_tpl: String::new(),
            abi_tpl: String::new(),
        });
    }

    let mut exact_eval_h = Sha256::new();
    let mut abi_eval_h = Sha256::new();
    let mut exact_tpl_h = Sha256::new();
    let mut abi_tpl_h = Sha256::new();

    for launch in &active {
        encode_launch(
            manifest,
            op,
            call,
            launch,
            vars,
            None,
            true,
            &mut exact_eval_h,
        )?;
        encode_launch(
            manifest,
            op,
            call,
            launch,
            vars,
            None,
            false,
            &mut abi_eval_h,
        )?;
        encode_launch(
            manifest,
            op,
            call,
            launch,
            vars,
            Some(rows_var),
            true,
            &mut exact_tpl_h,
        )?;
        encode_launch(
            manifest,
            op,
            call,
            launch,
            vars,
            Some(rows_var),
            false,
            &mut abi_tpl_h,
        )?;
    }

    Ok(CallSignatures {
        active_launches: active.len(),
        exact_eval: hex::encode(exact_eval_h.finalize()),
        abi_eval: hex::encode(abi_eval_h.finalize()),
        exact_tpl: hex::encode(exact_tpl_h.finalize()),
        abi_tpl: hex::encode(abi_tpl_h.finalize()),
    })
}

#[allow(clippy::too_many_arguments)]
fn encode_launch(
    manifest: &Manifest,
    op: &Op,
    call: &Call,
    launch: &Launch,
    vars: &BTreeMap<String, u64>,
    symbolic_var: Option<&str>,
    include_sha: bool,
    hasher: &mut Sha256,
) -> Result<()> {
    match launch {
        Launch::Kernel(k) => {
            hasher.update(b"K:");
            if include_sha {
                let m = manifest
                    .modules
                    .get(&k.module)
                    .with_context(|| format!("unknown module `{}`", k.module))?;
                hasher.update(m.sha256.to_ascii_lowercase().as_bytes());
                hasher.update(b":");
            }
            hasher.update(k.entry.as_bytes());
            hasher.update(format!(":b={:?}:c={:?}:pdl={}:g=[", k.block, k.cluster, k.pdl).as_bytes());
            for g in &k.grid {
                hasher.update(expr_repr(g, vars, symbolic_var)?.as_bytes());
                hasher.update(b",");
            }
            hasher.update(b"]:smem=");
            if let Some(sm) = &k.shared_mem {
                hasher.update(expr_repr(sm, vars, symbolic_var)?.as_bytes());
            } else {
                hasher.update(b"0");
            }
        }
        Launch::Extern(e) => {
            hasher.update(b"E:");
            hasher.update(e.entry.as_bytes());
            if let Some(a) = &e.algo {
                hasher.update(
                    format!(
                        ":algo={}/{}/{}/{}/{}/{}/{}/{}/{}",
                        a.id,
                        a.tile,
                        a.stages,
                        a.split_k,
                        a.reduction,
                        a.swizzle,
                        a.custom,
                        a.inner_shape,
                        a.cluster_shape
                    )
                    .as_bytes(),
                );
            }
        }
    }

    hasher.update(b":params=[");
    for p in launch.params_of(op) {
        hasher.update(p.to_string().as_bytes());
        hasher.update(b",");
    }
    hasher.update(b"]:args=[");
    for larg in launch.args_of(op).iter() {
        encode_launch_arg(op, call, larg, vars, symbolic_var, hasher)?;
        hasher.update(b",");
    }
    hasher.update(b"];");
    Ok(())
}

fn encode_launch_arg(
    op: &Op,
    call: &Call,
    larg: &LaunchArg,
    vars: &BTreeMap<String, u64>,
    symbolic_var: Option<&str>,
    hasher: &mut Sha256,
) -> Result<()> {
    match larg {
        LaunchArg::Param { param } => {
            let ptype = op
                .params
                .get(*param)
                .copied()
                .with_context(|| format!("param index {param} out of range"))?;
            encode_forwarded_param(ptype, call.args.get(*param), vars, symbolic_var, hasher)?;
        }
        LaunchArg::Scratch { scratch } => {
            if let Some(sc) = op.imp.scratch.get(scratch) {
                hasher.update(format!("sc:{}:{:?}", sc.dtype, sc.shape).as_bytes());
            } else {
                hasher.update(format!("sc:{scratch}").as_bytes());
            }
        }
        LaunchArg::I32 { i32: v } => hasher.update(format!("i32:{v}").as_bytes()),
        LaunchArg::I64 { i64: v } => hasher.update(format!("i64:{v}").as_bytes()),
        LaunchArg::F32 { f32: v } => hasher.update(format!("f32:{:08x}", v.to_bits()).as_bytes()),
        LaunchArg::U8 { u8: v } => hasher.update(format!("u8:{v}").as_bytes()),
        LaunchArg::Rank { rank } => hasher.update(format!("rank:{rank}").as_bytes()),
        LaunchArg::Pack { pack } => {
            hasher.update(format!("pack:{}:[", pack.size).as_bytes());
            for f in &pack.fields {
                hasher.update(format!("{}@{:?}:", f.at, f.width).as_bytes());
                match &f.src {
                    FieldSrc::Param { param } => {
                        let ptype = op
                            .params
                            .get(*param)
                            .copied()
                            .with_context(|| format!("pack param index {param} out of range"))?;
                        encode_forwarded_param(ptype, call.args.get(*param), vars, symbolic_var, hasher)?;
                    }
                    FieldSrc::Scratch { scratch } => hasher.update(format!("sc:{scratch}").as_bytes()),
                    FieldSrc::I32 { i32: v } => hasher.update(format!("i32:{v}").as_bytes()),
                    FieldSrc::I64 { i64: v } => hasher.update(format!("i64:{v}").as_bytes()),
                    FieldSrc::F32 { f32: v } => hasher.update(format!("f32:{:08x}", v.to_bits()).as_bytes()),
                    FieldSrc::U8 { u8: v } => hasher.update(format!("u8:{v}").as_bytes()),
                    FieldSrc::Var { var } => {
                        hasher.update(var_repr(var, vars, symbolic_var)?.as_bytes());
                    }
                    FieldSrc::Expr { expr } => {
                        hasher.update(expr_repr(expr, vars, symbolic_var)?.as_bytes());
                    }
                    FieldSrc::Rank { rank } => hasher.update(format!("rank:{rank}").as_bytes()),
                    FieldSrc::TensorMap { tensormap: tm } => {
                        hasher.update(
                            format!(
                                "tma:{:?}:{:?}:{:?}:{:?}:{}:{}:{}:{}",
                                tm.dtype,
                                tm.dims,
                                tm.strides,
                                tm.box_,
                                tm.swizzle,
                                tm.l2_promotion,
                                tm.oob_nan,
                                tm.wrap
                            )
                            .as_bytes(),
                        );
                    }
                }
                hasher.update(b",");
            }
            hasher.update(b"]");
        }
    }
    Ok(())
}

fn encode_forwarded_param(
    ptype: ParamType,
    arg: Option<&Arg>,
    vars: &BTreeMap<String, u64>,
    symbolic_var: Option<&str>,
    hasher: &mut Sha256,
) -> Result<()> {
    match ptype {
        ParamType::Buf { dtype, dir } => {
            hasher.update(format!("buf:{dtype}:{dir}").as_bytes());
        }
        ParamType::State { dir } => {
            hasher.update(format!("state:{dir}").as_bytes());
        }
        ParamType::Scalar(_) | ParamType::Bytes(_) => match arg {
            Some(Arg::Var { var }) => hasher.update(var_repr(var, vars, symbolic_var)?.as_bytes()),
            Some(Arg::Expr { expr }) => hasher.update(expr_repr(expr, vars, symbolic_var)?.as_bytes()),
            Some(Arg::I32 { i32: v }) => hasher.update(format!("s:{}", *v as i64).as_bytes()),
            Some(Arg::I64 { i64: v }) => hasher.update(format!("s:{v}").as_bytes()),
            Some(Arg::F32 { f32: v }) => hasher.update(format!("f:{:08x}", v.to_bits()).as_bytes()),
            Some(Arg::U8 { u8: v }) => hasher.update(format!("s:{v}").as_bytes()),
            Some(Arg::Rank { rank }) => hasher.update(format!("rank:{rank}").as_bytes()),
            Some( other ) => bail!("unexpected non-scalar call arg `{other}` bound to `{ptype}`"),
            None => bail!("missing call arg for `{ptype}`"),
        },
    }
    Ok(())
}

fn var_repr(var: &str, vars: &BTreeMap<String, u64>, symbolic_var: Option<&str>) -> Result<String> {
    if symbolic_var == Some(var) {
        return Ok("$rows".to_string());
    }
    let val = vars.get(var).copied().unwrap_or(kern_manifest::types::Var::MIN);
    Ok(format!("s:{val}"))
}

fn expr_repr(expr: &Expr, vars: &BTreeMap<String, u64>, symbolic_var: Option<&str>) -> Result<String> {
    if let Some(sym) = symbolic_var {
        if expr_mentions(expr, sym) {
            return Ok(symbolic_expr(expr, vars, sym));
        }
    }
    let v = expr_eval_default(expr, vars)?;
    Ok(format!("s:{v}"))
}

fn expr_mentions(expr: &Expr, target: &str) -> bool {
    match expr {
        Expr::Const(_) => false,
        Expr::Var(v) => v == target,
        Expr::CeilDiv { ceil_div: (e, _) }
        | Expr::Mul { mul: (e, _) }
        | Expr::Clamp { clamp: (e, _, _) } => expr_mentions(e, target),
        Expr::Add { add: (a, b) } => expr_mentions(a, target) || expr_mentions(b, target),
    }
}

fn symbolic_expr(expr: &Expr, vars: &BTreeMap<String, u64>, sym: &str) -> String {
    match expr {
        Expr::Const(c) => c.to_string(),
        Expr::Var(v) if v == sym => "$rows".to_string(),
        Expr::Var(v) => vars
            .get(v)
            .copied()
            .unwrap_or(kern_manifest::types::Var::MIN)
            .to_string(),
        Expr::CeilDiv { ceil_div: (e, c) } => format!("cd({},{c})", symbolic_expr(e, vars, sym)),
        Expr::Mul { mul: (e, c) } => format!("mul({},{c})", symbolic_expr(e, vars, sym)),
        Expr::Add { add: (a, b) } => {
            format!("add({},{})", symbolic_expr(a, vars, sym), symbolic_expr(b, vars, sym))
        }
        Expr::Clamp { clamp: (e, s, l) } => format!("clamp({},{s},{l})", symbolic_expr(e, vars, sym)),
    }
}

fn expr_eval_default(expr: &Expr, vars: &BTreeMap<String, u64>) -> Result<u64> {
    match expr {
        Expr::Const(c) => Ok(*c),
        Expr::Var(v) => Ok(vars.get(v).copied().unwrap_or(kern_manifest::types::Var::MIN)),
        Expr::CeilDiv { ceil_div: (inner, c) } => {
            ensure!(*c > 0, "division by zero in launch expression");
            let x = expr_eval_default(inner, vars)?;
            Ok(x.saturating_add(c - 1) / c)
        }
        Expr::Mul { mul: (inner, c) } => Ok(expr_eval_default(inner, vars)?.saturating_mul(*c)),
        Expr::Add { add: (a, b) } => {
            Ok(expr_eval_default(a, vars)?.saturating_add(expr_eval_default(b, vars)?))
        }
        Expr::Clamp { clamp: (inner, start, len) } => {
            Ok(expr_eval_default(inner, vars)?.saturating_sub(*start).min(*len))
        }
    }
}

fn interp_1d(pts: &[(u64, f64)], x: u64) -> Option<f64> {
    match pts {
        [] => None,
        [(_, y)] => Some(*y),
        _ => {
            let idx = pts.partition_point(|&(px, _)| px < x);
            if idx < pts.len() && pts[idx].0 == x {
                return Some(pts[idx].1);
            }
            if idx == 0 {
                return Some(pts[0].1);
            }
            if idx < pts.len() {
                let (x0, y0) = pts[idx - 1];
                let (x1, y1) = pts[idx];
                let t = (x - x0) as f64 / (x1 - x0) as f64;
                return Some(y0 + (y1 - y0) * t);
            }
            // Linear extrapolation above the highest calibrated knot with non-negative slope.
            let (x0, y0) = pts[pts.len() - 2];
            let (x1, y1) = pts[pts.len() - 1];
            let slope = ((y1 - y0) / (x1 - x0) as f64).max(0.0);
            Some(y1 + slope * (x - x1) as f64)
        }
    }
}

fn median_f64(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    let mid = v.len() / 2;
    if v.len() % 2 == 1 {
        v[mid]
    } else {
        0.5 * (v[mid - 1] + v[mid])
    }
}

// --- Bench JSON wire structs ---

#[derive(Deserialize)]
struct BenchReport {
    scenarios: Vec<BenchScenarioRecord>,
}

#[derive(Deserialize)]
struct BenchScenarioRecord {
    scenario: BenchScenarioMeta,
    program: String,
    vars: BTreeMap<String, u64>,
    graph: BenchSeries,
    calls: Vec<BenchCallRecord>,
}

#[derive(Deserialize)]
struct BenchScenarioMeta {
    id: String,
    rows: usize,
    context: Vec<usize>,
}

#[derive(Deserialize)]
struct BenchCallRecord {
    in_program: BenchSeries,
}

#[derive(Deserialize)]
struct BenchSeries {
    stats: BenchStats,
}

#[derive(Deserialize)]
struct BenchStats {
    p50: f64,
}

// --- Parquet reader structs ---

struct TracedStep {
    seq: u64,
    program: String,
    vars: BTreeMap<String, u64>,
    kv: Option<u64>,
    dur_ns: i64,
}

fn column<'a, T: Array + 'static>(b: &'a RecordBatch, name: &str) -> Result<&'a T> {
    let c = b
        .column_by_name(name)
        .with_context(|| format!("missing `{name}` column in parquet batch"))?;
    c.as_any()
        .downcast_ref::<T>()
        .with_context(|| format!("column `{name}` has unexpected arrow type {}", c.data_type()))
}

fn parse_vars_csv(text: &str) -> Result<BTreeMap<String, u64>> {
    text.split(',')
        .filter(|kv| !kv.is_empty())
        .map(|kv| {
            let (name, value) = kv.split_once('=').with_context(|| format!("malformed vars `{text}`"))?;
            Ok((
                name.to_string(),
                value.parse().with_context(|| format!("malformed var integer in `{text}`"))?,
            ))
        })
        .collect()
}

fn read_traced_steps(path: &Path) -> Result<Vec<TracedStep>> {
    let reader = ParquetRecordBatchReaderBuilder::try_new(
        File::open(path).with_context(|| format!("opening {}", path.display()))?,
    )?;
    let batches = reader.build()?.collect::<std::result::Result<Vec<RecordBatch>, _>>()?;
    let mut out = Vec::new();
    for b in &batches {
        let seq = column::<UInt64Array>(b, "seq")?;
        let program = column::<StringArray>(b, "program")?;
        let vars_col = column::<StringArray>(b, "vars")?;
        let kv = column::<UInt64Array>(b, "kv")?;
        let _graph = column::<BooleanArray>(b, "graph")?;
        let start = column::<Int64Array>(b, "start_ns")?;
        let end = column::<Int64Array>(b, "end_ns")?;
        for i in 0..b.num_rows() {
            out.push(TracedStep {
                seq: seq.value(i),
                program: program.value(i).to_string(),
                vars: parse_vars_csv(vars_col.value(i))?,
                kv: kv.is_valid(i).then(|| kv.value(i)),
                dur_ns: (end.value(i) - start.value(i)).max(0),
            });
        }
    }
    Ok(out)
}

fn read_traced_kernels(path: &Path) -> Result<BTreeMap<u64, Vec<(String, i64)>>> {
    let reader = ParquetRecordBatchReaderBuilder::try_new(
        File::open(path).with_context(|| format!("opening {}", path.display()))?,
    )?;
    let batches = reader.build()?.collect::<std::result::Result<Vec<RecordBatch>, _>>()?;
    let mut by_seq: BTreeMap<u64, Vec<(String, i64)>> = BTreeMap::new();
    for b in &batches {
        let seq = column::<UInt64Array>(b, "seq")?;
        let dur = column::<Int64Array>(b, "dur_ns")?;
        let call = column::<StringArray>(b, "call")?;
        for i in 0..b.num_rows() {
            if !seq.is_valid(i) || !call.is_valid(i) {
                continue;
            }
            by_seq
                .entry(seq.value(i))
                .or_default()
                .push((call.value(i).to_string(), dur.value(i).max(0)));
        }
    }
    Ok(by_seq)
}
