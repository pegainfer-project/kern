//! Workload representation and multi-manifest static repricing.
//!
//! Mirrors `kern bench`'s `workload.toml` shape grammar and `steps.parquet`
//! trace bucketing so any set of candidate manifests can be priced side by side
//! through a [`CostBook`] on CPU.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context as _, Result};
use arrow_array::{Array, Int64Array, RecordBatch, StringArray, UInt64Array};
use kern_manifest::protocol::Rows;
use kern_manifest::{Protocol, Verified};
use kern_pool::{page_unit, row_tokens};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::{Deserialize, Serialize};

use crate::cost::{CostBook, MatchTier, PriceOptions, SignatureCache};

#[derive(Deserialize, Clone, Debug)]
#[serde(untagged)]
enum SweepContext {
    Every(usize),
    Each(Vec<usize>),
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct Sweep {
    #[serde(default = "default_groups")]
    groups: Vec<usize>,
    #[serde(default)]
    rows: Vec<usize>,
    #[serde(default)]
    chunk: Vec<usize>,
    context: Vec<SweepContext>,
    #[serde(default = "default_weight")]
    weight: u64,
}

fn default_groups() -> Vec<usize> {
    vec![1]
}

fn default_weight() -> u64 {
    1
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct RawWorkload {
    #[allow(dead_code)]
    #[serde(default = "default_samples")]
    samples: usize,
    #[allow(dead_code)]
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    sweep: Vec<Sweep>,
}

fn default_samples() -> usize {
    24
}

/// One call shape: `groups` sequences of `rows` rows each, either a fixed-row
/// step (`chunk == false`) or variable rows as fed (`chunk == true`), over
/// `context` previous tokens per sequence, weighted by call count `weight`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shape {
    pub groups: usize,
    pub rows: usize,
    pub chunk: bool,
    pub context: Vec<usize>,
    pub weight: u64,
}

impl Shape {
    pub fn kv_label(&self) -> String {
        match self.context.split_first() {
            Some((n, rest)) if rest.iter().all(|c| c == n) => n.to_string(),
            _ => self
                .context
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join("+"),
        }
    }

    pub fn label(&self) -> String {
        format!(
            "g{}-{}{}-kv{}",
            self.groups,
            if self.chunk { 'c' } else { 'r' },
            self.rows,
            self.kv_label()
        )
    }
}

/// A portable collection of weighted call shapes to price against manifests.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Workload {
    pub name: String,
    pub shapes: Vec<Shape>,
}

impl Workload {
    /// Parse a `kern bench` `workload.toml` file.
    pub fn from_toml_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading workload {}", path.display()))?;
        let name = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "workload".into());
        Self::from_toml_str(&name, &text)
    }

    /// Parse `workload.toml` text.
    pub fn from_toml_str(name: &str, text: &str) -> Result<Self> {
        let raw: RawWorkload = toml::from_str(text).context("parsing workload TOML")?;
        for s in &raw.sweep {
            ensure!(
                s.rows.is_empty() != s.chunk.is_empty(),
                "a sweep must specify either `rows` or `chunk`, not both or neither"
            );
            ensure!(!s.context.is_empty(), "a sweep needs a non-empty `context`");
            ensure!(s.groups.iter().all(|&g| g > 0), "groups must be positive");
            ensure!(
                s.rows.iter().chain(&s.chunk).all(|&r| r > 0),
                "rows/chunk must be positive"
            );
            ensure!(s.weight > 0, "weight must be positive");
            for c in &s.context {
                if let SweepContext::Each(v) = c {
                    ensure!(
                        s.groups.contains(&v.len()),
                        "per-sequence context of length {} matches no group count in {:?}",
                        v.len(),
                        s.groups
                    );
                }
            }
        }

        let expanded = raw.sweep.iter().flat_map(|s| {
            let rows = s
                .rows
                .iter()
                .map(|&r| (r, false))
                .chain(s.chunk.iter().map(|&r| (r, true)));
            s.groups.iter().flat_map(move |&g| {
                rows.clone().flat_map(move |(r, chunk)| {
                    s.context.iter().filter_map(move |c| match c {
                        SweepContext::Every(n) => Some(Shape {
                            groups: g,
                            rows: r,
                            chunk,
                            context: vec![*n; g],
                            weight: s.weight,
                        }),
                        SweepContext::Each(v) if v.len() == g => Some(Shape {
                            groups: g,
                            rows: r,
                            chunk,
                            context: v.clone(),
                            weight: s.weight,
                        }),
                        SweepContext::Each(_) => None,
                    })
                })
            })
        });

        let mut index_by_label: BTreeMap<String, usize> = BTreeMap::new();
        let mut shapes: Vec<Shape> = Vec::new();
        for s in expanded {
            let lbl = s.label();
            if let Some(&idx) = index_by_label.get(&lbl) {
                shapes[idx].weight += s.weight;
            } else {
                index_by_label.insert(lbl, shapes.len());
                shapes.push(s);
            }
        }

        Ok(Self {
            name: name.to_string(),
            shapes,
        })
    }

    /// Build a [`Workload`] from closed `*.steps.parquet` files (or directories
    /// containing them), interpreted through `reference_manifest`'s protocol.
    pub fn from_steps_parquet(
        name: &str,
        paths: &[PathBuf],
        reference_manifest: &Verified,
    ) -> Result<Self> {
        let protocol = Protocol::check_unsampled(reference_manifest)?;
        let found = collect_steps_files(paths)?;
        ensure!(!found.is_empty(), "no closed `steps.parquet` files found");

        // Bucket by `(program, shape_label, kv_pow2)` and represent each bucket by its
        // median shape and mean context, matching `kern workload` (`served.rs`).
        type BucketKey = (String, String, Option<u64>);
        let mut buckets: BTreeMap<BucketKey, Vec<(bool, usize, usize, usize)>> = BTreeMap::new();

        for path in &found {
            let reader = ParquetRecordBatchReaderBuilder::try_new(
                File::open(path).with_context(|| format!("opening {}", path.display()))?,
            )?;
            let batches = reader.build()?.collect::<std::result::Result<Vec<RecordBatch>, _>>()?;
            for b in &batches {
                let prog_col = col::<StringArray>(b, "program")?;
                let shape_col = col::<StringArray>(b, "shape")?;
                let vars_col = col::<StringArray>(b, "vars")?;
                let kv_col = col::<UInt64Array>(b, "kv")?;
                let _start_col = col::<Int64Array>(b, "start_ns")?;

                for i in 0..b.num_rows() {
                    let pname = prog_col.value(i);
                    let Some(fwd) = protocol.forwards.iter().find(|f| f.name == pname) else {
                        continue;
                    };
                    let vars = parse_vars(vars_col.value(i))?;
                    let groups = vars.get(&protocol.groups.var).copied().unwrap_or(fwd.groups) as usize;
                    if groups == 0 {
                        continue;
                    }
                    let fed = vars.get(&protocol.rows.var).copied().unwrap_or(1) as usize;
                    let (chunk, rows, own) = match fwd.rows {
                        Rows::Const(r) => (false, r as usize, groups * r as usize),
                        Rows::Var => {
                            let each = ((fed + groups / 2) / groups)
                                .min(protocol.rows.max as usize / groups)
                                .max(1);
                            (true, each, fed)
                        }
                    };
                    let raw_kv = kv_col.is_valid(i).then(|| kv_col.value(i));
                    let ctx = raw_kv.map_or(0, |kv| (kv as usize).saturating_sub(own) / groups);
                    let kv_bucket = raw_kv.map(|v| if v == 0 { 0 } else { v.next_power_of_two() });
                    buckets
                        .entry((pname.to_string(), shape_col.value(i).to_string(), kv_bucket))
                        .or_default()
                        .push((chunk, groups, rows, ctx));
                }
            }
        }

        let mut index_by_label: BTreeMap<String, usize> = BTreeMap::new();
        let mut shapes: Vec<Shape> = Vec::new();

        for members in buckets.values() {
            let mut sorted = members.clone();
            sorted.sort_unstable_by_key(|&(chunk, g, r, _)| (chunk, g, r));
            let n = sorted.len();
            let (chunk, groups, rows, _) = sorted[n / 2];
            let mean_ctx = (sorted.iter().map(|m| m.3).sum::<usize>() + n / 2) / n;
            let shape = Shape {
                groups,
                rows,
                chunk,
                context: vec![mean_ctx; groups],
                weight: n as u64,
            };
            let lbl = shape.label();
            if let Some(&idx) = index_by_label.get(&lbl) {
                shapes[idx].weight += shape.weight;
            } else {
                index_by_label.insert(lbl, shapes.len());
                shapes.push(shape);
            }
        }

        Ok(Self {
            name: name.to_string(),
            shapes,
        })
    }

    pub fn total_calls(&self) -> u64 {
        self.shapes.iter().map(|s| s.weight).sum()
    }
}

/// A shape rejected by a manifest's protocol or page-table bounds.
#[derive(Debug, Clone, Serialize)]
pub struct DroppedShape {
    pub shape: String,
    pub weight: u64,
    pub reason: String,
}

/// Priced result for one scenario in a [`Workload`].
#[derive(Debug, Clone, Serialize)]
pub struct ScenarioPrice {
    pub id: String,
    pub program: String,
    pub shape: Shape,
    pub program_us: f64,
    pub weighted_us: f64,
    pub exact_coverage: f64,
    pub calibrated_coverage: f64,
}

/// Aggregate result of pricing a [`Workload`] on a [`Verified`] manifest.
#[derive(Debug, Clone, Serialize)]
pub struct WorkloadPriceReport {
    pub workload: String,
    pub model: String,
    pub total_us: f64,
    pub total_s: f64,
    pub priced_calls: u64,
    pub dropped_calls: u64,
    pub exact_time_coverage: f64,
    pub calibrated_time_coverage: f64,
    pub calls_by_tier: BTreeMap<MatchTier, u64>,
    pub us_by_tier: BTreeMap<MatchTier, f64>,
    /// Share of attributed kernel time by op name, largest first.
    pub by_op: Vec<(String, f64)>,
    pub dropped: Vec<DroppedShape>,
    pub scenarios: Vec<ScenarioPrice>,
}

/// Price every shape of `workload` on `manifest` using `book`.
pub fn price_workload(
    book: &CostBook,
    manifest: &Verified,
    workload: &Workload,
    opts: PriceOptions,
) -> Result<WorkloadPriceReport> {
    let protocol = Protocol::check_unsampled(manifest)?;
    let unit = page_unit(manifest);
    let row_cap = row_tokens(manifest, unit);
    let rank_count = manifest
        .topology
        .as_ref()
        .map_or(1u64, |t| t.groups.values().copied().product::<u64>().max(1));

    let mut scenarios = Vec::new();
    let mut dropped = Vec::new();
    let mut total_us = 0.0f64;
    let mut priced_calls = 0u64;
    let mut dropped_calls = 0u64;
    let mut calls_by_tier: BTreeMap<MatchTier, u64> = BTreeMap::new();
    let mut us_by_tier: BTreeMap<MatchTier, f64> = BTreeMap::new();
    let mut op_weighted_us: BTreeMap<String, f64> = BTreeMap::new();
    let mut sig_cache = SignatureCache::new();

    for shape in &workload.shapes {
        match resolve_shape(&protocol, shape, row_cap) {
            Err(reason) => {
                dropped_calls += shape.weight;
                dropped.push(DroppedShape {
                    shape: shape.label(),
                    weight: shape.weight,
                    reason,
                });
            }
            Ok(program) => {
                let b = shape.groups as u64;
                let rows_total = (shape.groups * shape.rows) as u64;
                let ranks_total = rows_total.saturating_mul(rank_count);
                let sum_kv: u64 = shape
                    .context
                    .iter()
                    .map(|&c| c as u64 + shape.rows as u64)
                    .sum();
                let vars = protocol.vars(b, rows_total, ranks_total, sum_kv);
                let est = book.estimate_program_cached(
                    manifest,
                    &protocol.rows.var,
                    &program,
                    &vars,
                    sum_kv,
                    opts,
                    &mut sig_cache,
                )?;

                let w = shape.weight as f64;
                let weighted_us = w * est.total_us;
                total_us += weighted_us;
                priced_calls += shape.weight;

                for (&tier, &c) in &est.calls_by_tier {
                    *calls_by_tier.entry(tier).or_default() += shape.weight * c as u64;
                }
                for (&tier, &u) in &est.us_by_tier {
                    *us_by_tier.entry(tier).or_default() += w * u;
                }
                if est.sum_calls_us > 0.0 {
                    let scale = weighted_us / est.sum_calls_us;
                    for c in &est.calls {
                        *op_weighted_us.entry(c.op.clone()).or_default() += c.us * scale;
                    }
                }

                scenarios.push(ScenarioPrice {
                    id: format!("{program}-{}", shape.label()),
                    program,
                    shape: shape.clone(),
                    program_us: est.total_us,
                    weighted_us,
                    exact_coverage: est.exact_time_coverage(),
                    calibrated_coverage: est.calibrated_time_coverage(),
                });
            }
        }
    }

    let total_call_sites: u64 = calls_by_tier.values().sum();
    let uncal_calls = calls_by_tier
        .get(&MatchTier::Uncalibrated)
        .copied()
        .unwrap_or(0);
    let uncal_us = us_by_tier
        .get(&MatchTier::Uncalibrated)
        .copied()
        .unwrap_or(0.0);
    let known_call_share = if total_call_sites > 0 && uncal_calls > 0 && uncal_us <= 0.0 {
        (total_call_sites - uncal_calls) as f64 / total_call_sites as f64
    } else {
        1.0
    };

    let sum_tier_us: f64 = us_by_tier.values().sum();
    let exact_us = us_by_tier.get(&MatchTier::ExactSha256).copied().unwrap_or(0.0);
    let abi_us = us_by_tier.get(&MatchTier::EntryAbi).copied().unwrap_or(0.0);
    let exact_time_coverage = if sum_tier_us > 0.0 {
        (exact_us / sum_tier_us) * known_call_share
    } else {
        known_call_share
    };
    let calibrated_time_coverage = if sum_tier_us > 0.0 {
        ((exact_us + abi_us) / sum_tier_us) * known_call_share
    } else {
        known_call_share
    };

    let op_sum: f64 = op_weighted_us.values().sum();
    let mut by_op: Vec<(String, f64)> = op_weighted_us
        .into_iter()
        .map(|(op, u)| (op, if op_sum > 0.0 { u / op_sum } else { 0.0 }))
        .collect();
    by_op.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    Ok(WorkloadPriceReport {
        workload: workload.name.clone(),
        model: manifest.model.clone(),
        total_us,
        total_s: total_us / 1e6,
        priced_calls,
        dropped_calls,
        exact_time_coverage,
        calibrated_time_coverage,
        calls_by_tier,
        us_by_tier,
        by_op,
        dropped,
        scenarios,
    })
}

fn resolve_shape(p: &Protocol, s: &Shape, row_cap: Option<u64>) -> std::result::Result<String, String> {
    if let (Some(r), Some(&mx)) = (row_cap, s.context.iter().max()) {
        if (mx + s.rows) as u64 > r {
            return Err(format!(
                "sequence of {} tokens exceeds page-table row limit ({r})",
                mx + s.rows
            ));
        }
    }
    if s.groups * s.rows > p.rows.max as usize {
        return Err(format!(
            "{} rows exceeds manifest bound {}",
            s.groups * s.rows,
            p.rows.max
        ));
    }
    if s.context.iter().any(|&n| n > 0) && p.chunk().is_none() {
        return Err("non-zero context requires a variable-row chunk program".into());
    }
    let rows_kind = if s.chunk {
        Rows::Var
    } else {
        Rows::Const(s.rows as u64)
    };
    p.forward(s.groups as u64, rows_kind)
        .map(|f| f.name.clone())
        .ok_or_else(|| format!("no forward program accepts {} sequences of {:?}", s.groups, rows_kind))
}

fn collect_steps_files(paths: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut found = Vec::new();
    for p in paths {
        if !p.is_dir() {
            if p.to_string_lossy().ends_with("steps.parquet") {
                found.push(p.clone());
            }
            continue;
        }
        let mut entries: Vec<PathBuf> = std::fs::read_dir(p)
            .with_context(|| format!("reading directory {}", p.display()))?
            .map(|e| Ok(e?.path()))
            .collect::<Result<_>>()?;
        entries.sort();
        let (dirs, files): (Vec<PathBuf>, Vec<PathBuf>) = entries.into_iter().partition(|e| e.is_dir());
        found.extend(
            files
                .into_iter()
                .filter(|f| f.to_string_lossy().ends_with("steps.parquet")),
        );
        found.extend(collect_steps_files(&dirs)?);
    }
    Ok(found)
}

fn col<'a, T: Array + 'static>(b: &'a RecordBatch, name: &str) -> Result<&'a T> {
    let c = b
        .column_by_name(name)
        .with_context(|| format!("missing `{name}` column"))?;
    c.as_any()
        .downcast_ref::<T>()
        .with_context(|| format!("unexpected type for `{name}`"))
}

fn parse_vars(text: &str) -> Result<BTreeMap<String, u64>> {
    text.split(',')
        .filter(|kv| !kv.is_empty())
        .map(|kv| {
            let (k, v) = kv.split_once('=').with_context(|| format!("vars `{text}`"))?;
            Ok((k.to_string(), v.parse()?))
        })
        .collect()
}
