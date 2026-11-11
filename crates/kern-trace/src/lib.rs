//! Every traced kernel on disk, kept for as long as the disk allows.
//!
//! A rank writes what the tracer collects ([`kern_runtime::trace`]) into
//! Parquet files of its own, a pair per quarter hour:
//! `<dir>/date=YYYY-MM-DD/rank=<q>/HHMMSS.{kernels,steps}.parquet`, named
//! by the UTC time the pair was opened. `kernels` is one row per kernel
//! (the step's `seq`, null for a kernel no issue claims; start and
//! duration; the call's label and op; CUPTI's kernel name), `steps` one
//! row per step (program, shape, the exact var values, the exact `seq_len`
//! sum, graph or eager, first start, last end). Timestamps are unix
//! nanoseconds: CUPTI's clock moved by the offset to the wall clock sampled
//! when the pair was opened, so ranks, files, logs and alerts share one
//! time axis.
//! The call is its label, which survives a manifest that renumbers its
//! calls ([`Sites`]). A step is named by its program and shape
//! ([`shape`]): a graph program's var values (its graph key), an eager
//! one's rounded up to a power of two, so a prefill chunk's label set
//! stays small. A file's key-value metadata names the rank, the model,
//! the manifest's sha256, its writer's version ([`meta`]) and the clock
//! offset, and says what a reader cannot see in the rows: whether the GPU
//! stamped the kernels (`kern.hes`), and how many kernel records the
//! process had lost when the file closed (`kern.dropped_records`).
//!
//! A Parquet file is readable once closed, when its footer is written:
//! a pair closes at the next quarter hour, or when `rotate` asks. It is
//! written as `….parquet.part` and renamed when closed, so a reader that
//! globs `*.parquet` only ever finds whole files. Strings go through
//! Parquet's dictionaries and the integer columns through delta encoding
//! under zstd, so a kernel costs a few bytes on disk (docs/serve.md).
//!
//! The crate is the format and nothing else, so that everything that
//! traces writes the same files: kern-serve's ranks, and a manifest hosted
//! in another engine's process.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow_array::{ArrayRef, BooleanArray, Int64Array, RecordBatch, StringArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use kern_manifest::types::Manifest;
use kern_runtime::trace::{self, Collected, Step};
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, Encoding, ZstdLevel};
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;
use parquet::schema::types::ColumnPath;

const QUARTER: u64 = 15 * 60;
const SEC: i64 = 1_000_000_000;

/// The two clocks read together, unix and CUPTI nanoseconds, and whether
/// CUPTI's is the GPU's own (`None`: CUPTI would not say).
#[derive(Debug, Clone, Copy)]
pub struct Now {
    pub unix: i64,
    pub cupti: u64,
    pub hes: Option<bool>,
}

impl Now {
    /// Reads both; the tracer is on.
    pub fn read() -> kern_runtime::Result<Now> {
        let cupti = trace::now()?;
        let unix =
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos() as i64);
        Ok(Now { unix, cupti, hes: trace::hes().ok() })
    }

    fn offset(self) -> i64 {
        self.unix - self.cupti as i64
    }
}

/// What the manifest says each call is: per program, per call, its label
/// (its op and index when it has none) and its op.
pub struct Sites(BTreeMap<String, Vec<(String, String)>>);

impl Sites {
    pub fn new(m: &Manifest) -> Self {
        let named = |(i, c): (usize, &kern_manifest::types::Call)| {
            (c.label.clone().unwrap_or_else(|| format!("{}#{i}", c.op)), c.op.clone())
        };
        Self(
            m.programs
                .iter()
                .map(|(p, prog)| (p.clone(), prog.calls.iter().enumerate().map(named).collect()))
                .collect(),
        )
    }

    /// A call's (label, op); `?` for a kernel whose call is not known.
    fn site(&self, program: &str, call: Option<u32>) -> (String, String) {
        call.and_then(|c| self.0.get(program)?.get(c as usize).cloned()).unwrap_or_else(|| ("?".into(), "?".into()))
    }
}

/// What every file says of its writer: the manifest's model, the sha256 of
/// the manifest as it was read (`json`), and the writer's version.
pub fn meta(m: &Manifest, json: &str, version: &str) -> Vec<(String, String)> {
    use sha2::Digest;
    vec![
        ("kern.model".to_string(), m.model.clone()),
        ("kern.manifest_sha256".to_string(), hex::encode(sha2::Sha256::digest(json.as_bytes()))),
        ("kern.version".to_string(), version.to_string()),
    ]
}

/// An issue's shape label: its graph key, or its vars rounded up to a
/// power of two.
pub fn shape(s: &Step) -> String {
    let vars = s.issue.vars.iter().map(|(n, v)| match s.issue.graph {
        true => format!("{n}={v}"),
        false => format!("{n}<={}", v.next_power_of_two()),
    });
    vars.collect::<Vec<_>>().join(",")
}

struct Open {
    quarter: u64,
    offset: i64,
    kernels: ArrowWriter<File>,
    steps: ArrowWriter<File>,
    paths: [PathBuf; 2],
}

pub struct Recorder {
    root: PathBuf,
    rank: usize,
    /// What every file's metadata says besides its rank and clock offset.
    meta: Vec<(String, String)>,
    open: Option<Open>,
}

fn kernels_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("seq", DataType::UInt64, true),
        Field::new("start_ns", DataType::Int64, false),
        Field::new("dur_ns", DataType::Int64, false),
        Field::new("call", DataType::Utf8, true),
        Field::new("op", DataType::Utf8, true),
        Field::new("kernel", DataType::Utf8, false),
    ]))
}

fn steps_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("seq", DataType::UInt64, false),
        Field::new("program", DataType::Utf8, false),
        Field::new("shape", DataType::Utf8, false),
        Field::new("vars", DataType::Utf8, false),
        Field::new("kv", DataType::UInt64, true),
        Field::new("graph", DataType::Boolean, false),
        Field::new("start_ns", DataType::Int64, false),
        Field::new("end_ns", DataType::Int64, false),
    ]))
}

/// The issue's var values as given, `name=value` joined by commas.
fn exact(s: &Step) -> String {
    s.issue.vars.iter().map(|(n, v)| format!("{n}={v}")).collect::<Vec<_>>().join(",")
}

/// What was collected as the two tables' rows, timestamps moved by `offset`.
fn batches(c: &Collected, sites: &Sites, offset: i64) -> Result<(RecordBatch, RecordBatch)> {
    let at = |ns: u64| ns as i64 + offset;
    let steps: Vec<_> = c.steps.iter().filter(|s| !s.spans.is_empty()).collect();
    let claimed = steps.iter().flat_map(|s| {
        s.spans.iter().map(move |k| {
            let (call, op) = sites.site(&s.issue.program, k.call);
            (Some(s.seq), at(k.start), (k.end.saturating_sub(k.start)) as i64, Some(call), Some(op), k.name.to_string())
        })
    });
    let unclaimed = c
        .unclaimed
        .iter()
        .map(|k| (None, at(k.start), k.end.saturating_sub(k.start) as i64, None, None, k.name.to_string()));
    let mut rows: Vec<_> = claimed.chain(unclaimed).collect();
    rows.sort_by_key(|r| r.1);
    let kernels: Vec<ArrayRef> = vec![
        Arc::new(rows.iter().map(|r| r.0).collect::<UInt64Array>()),
        Arc::new(rows.iter().map(|r| r.1).collect::<Int64Array>()),
        Arc::new(rows.iter().map(|r| r.2).collect::<Int64Array>()),
        Arc::new(rows.iter().map(|r| r.3.as_deref()).collect::<StringArray>()),
        Arc::new(rows.iter().map(|r| r.4.as_deref()).collect::<StringArray>()),
        Arc::new(rows.iter().map(|r| Some(r.5.as_str())).collect::<StringArray>()),
    ];
    let first = |s: &Step| s.spans.iter().map(|k| k.start).min().unwrap_or(0);
    let last = |s: &Step| s.spans.iter().map(|k| k.end).max().unwrap_or(0);
    let step_cols: Vec<ArrayRef> = vec![
        Arc::new(steps.iter().map(|s| Some(s.seq)).collect::<UInt64Array>()),
        Arc::new(steps.iter().map(|s| Some(s.issue.program.as_str())).collect::<StringArray>()),
        Arc::new(steps.iter().map(|s| Some(shape(s))).collect::<StringArray>()),
        Arc::new(steps.iter().map(|s| Some(exact(s))).collect::<StringArray>()),
        Arc::new(steps.iter().map(|s| s.issue.kv).collect::<UInt64Array>()),
        Arc::new(steps.iter().map(|s| Some(s.issue.graph)).collect::<BooleanArray>()),
        Arc::new(steps.iter().map(|s| Some(at(first(s)))).collect::<Int64Array>()),
        Arc::new(steps.iter().map(|s| Some(at(last(s)))).collect::<Int64Array>()),
    ];
    Ok((RecordBatch::try_new(kernels_schema(), kernels)?, RecordBatch::try_new(steps_schema(), step_cols)?))
}

/// `YYYY-MM-DD` and `HHMMSS` of a unix second, in UTC.
fn utc(secs: u64) -> (String, String) {
    // Howard Hinnant's days-to-civil.
    let z = (secs / 86_400) as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    let t = secs % 86_400;
    (format!("{y:04}-{m:02}-{d:02}"), format!("{:02}{:02}{:02}", t / 3600, t / 60 % 60, t % 60))
}

fn part(path: &Path) -> PathBuf {
    path.with_extension("parquet.part")
}

fn writer(path: &Path, schema: SchemaRef, meta: Vec<KeyValue>, deltas: &[&str]) -> Result<ArrowWriter<File>> {
    let props = deltas.iter().fold(
        WriterProperties::builder()
            .set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?))
            .set_max_row_group_row_count(Some(1 << 20))
            .set_key_value_metadata(Some(meta)),
        |b, c| {
            b.set_column_dictionary_enabled(ColumnPath::from(*c), false)
                .set_column_encoding(ColumnPath::from(*c), Encoding::DELTA_BINARY_PACKED)
        },
    );
    let file = File::create(part(path)).with_context(|| format!("creating {}", part(path).display()))?;
    Ok(ArrowWriter::try_new(file, schema, Some(props.build()))?)
}

impl Recorder {
    pub fn new(root: PathBuf, rank: usize, meta: Vec<(String, String)>) -> Self {
        Self { root, rank, meta, open: None }
    }

    /// Appends what was collected to the open pair, first closing it if
    /// `now` is past its quarter hour and opening one if none is.
    pub fn write(&mut self, c: &Collected, sites: &Sites, now: Now) -> Result<()> {
        let secs = (now.unix / SEC) as u64;
        if self.open.as_ref().is_some_and(|o| o.quarter != secs / QUARTER) {
            self.rotate()?;
        }
        if c.steps.is_empty() && c.unclaimed.is_empty() {
            return Ok(());
        }
        let open = match &mut self.open {
            Some(o) => o,
            None => self.open.insert(self.create(now)?),
        };
        let (kernels, steps) = batches(c, sites, open.offset)?;
        open.kernels.write(&kernels)?;
        open.steps.write(&steps)?;
        Ok(())
    }

    fn create(&self, now: Now) -> Result<Open> {
        let secs = (now.unix / SEC) as u64;
        let (date, time) = utc(secs);
        let dir = self.root.join(format!("date={date}")).join(format!("rank={}", self.rank));
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
        let meta: Vec<KeyValue> = [("kern.rank".to_string(), self.rank.to_string())]
            .into_iter()
            .chain(self.meta.iter().cloned())
            .chain([("kern.clock_offset_ns".to_string(), now.offset().to_string())])
            // A pair opens on its first kernels, so a context exists and CUPTI has chosen.
            .chain(now.hes.map(|hes| ("kern.hes".to_string(), hes.to_string())))
            .map(|(k, v)| KeyValue::new(k, v))
            .collect();
        let paths = [dir.join(format!("{time}.kernels.parquet")), dir.join(format!("{time}.steps.parquet"))];
        Ok(Open {
            quarter: secs / QUARTER,
            offset: now.offset(),
            kernels: writer(&paths[0], kernels_schema(), meta.clone(), &["seq", "start_ns", "dur_ns"])?,
            steps: writer(&paths[1], steps_schema(), meta, &["seq", "kv", "start_ns", "end_ns"])?,
            paths,
        })
    }

    /// Closes the open pair, making it readable; returns its paths.
    pub fn rotate(&mut self) -> Result<Vec<PathBuf>> {
        let Some(mut o) = self.open.take() else { return Ok(Vec::new()) };
        for w in [&mut o.kernels, &mut o.steps] {
            w.append_key_value_metadata(KeyValue::new(
                "kern.dropped_records".to_string(),
                trace::dropped().to_string(),
            ));
        }
        o.kernels.close()?;
        o.steps.close()?;
        for p in &o.paths {
            std::fs::rename(part(p), p).with_context(|| format!("renaming {}", part(p).display()))?;
        }
        Ok(o.paths.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utc_names_the_day_and_the_time() {
        assert_eq!(utc(0), ("1970-01-01".into(), "000000".into()));
        assert_eq!(utc(1_793_797_134), ("2026-11-04".into(), "125854".into()));
        assert_eq!(utc(951_825_600), ("2000-02-29".into(), "120000".into()));
    }
}
