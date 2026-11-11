//! The recorder as a rank drives it: what the tracer collected over a
//! small manifest's programs, written and read back once the pair closed.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{Array, Int64Array, RecordBatch, StringArray};
use kern_manifest::types::Manifest;
use kern_runtime::trace::{Collected, Issue, Kernel, Span, Step};
use kern_trace::{meta, Now, Recorder, Sites};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

const SEC: i64 = 1_000_000_000;
const PAGED: &str = include_str!("../../kern-manifest/tests/fixtures/paged.json");

/// One step of `program`: its first two calls, a kernel whose call the
/// tracer could not tell, and after them a kernel no issue claims.
fn collected(seq: u64, program: &str, graph: bool, var: (&str, u64), at: u64) -> Collected {
    let span = |call, start, end| Span { call, name: "k".into(), start, end };
    let issue = Issue {
        program: program.into(),
        vars: vec![(var.0.into(), var.1)],
        graph,
        kv: Some(4096),
        kernels: Arc::new([(0, 0), (1, 1)].into()),
    };
    let spans = vec![span(Some(0), at, at + 10), span(Some(1), at + 10, at + 30), span(None, at + 30, at + 35)];
    let copy = Kernel { correlation: 0, device: 0, start: at + 40, end: at + 45, name: "copy".into(), node: None };
    Collected { steps: vec![Step { seq, issue, spans }], unclaimed: vec![copy] }
}

fn reader(path: &Path) -> ParquetRecordBatchReaderBuilder<File> {
    ParquetRecordBatchReaderBuilder::try_new(File::open(path).unwrap()).unwrap()
}

fn rows(path: &Path) -> Vec<RecordBatch> {
    reader(path).build().unwrap().map(|b| b.unwrap()).collect()
}

fn says(path: &Path, key: &str) -> Option<String> {
    let said = reader(path).metadata().file_metadata().key_value_metadata().cloned().unwrap_or_default();
    said.into_iter().find(|k| k.key == key).and_then(|k| k.value)
}

fn strings(rows: &[RecordBatch], column: &str) -> Vec<Option<String>> {
    let of = |b: &RecordBatch| {
        let c = b.column_by_name(column).unwrap().as_any().downcast_ref::<StringArray>().unwrap().clone();
        (0..c.len()).map(|i| c.is_valid(i).then(|| c.value(i).to_string())).collect::<Vec<_>>()
    };
    rows.iter().flat_map(of).collect()
}

fn ints(rows: &[RecordBatch], column: &str) -> Vec<i64> {
    let of = |b: &RecordBatch| {
        b.column_by_name(column).unwrap().as_any().downcast_ref::<Int64Array>().unwrap().values().to_vec()
    };
    rows.iter().flat_map(of).collect()
}

#[test]
fn a_quarter_hour_closes_into_whole_files_that_name_calls_as_the_manifest_does() {
    let m = Manifest::from_json(PAGED).unwrap();
    let sites = &Sites::new(&m);
    let root = std::env::temp_dir().join(format!("kern-trace-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let mut r = Recorder::new(root.clone(), 3, meta(&m, PAGED, "1.2.3"));
    let t0 = 1_793_797_134 * SEC;
    r.write(&collected(1, "decode", true, ("seqs", 4), 100), sites, Now { unix: t0, cupti: 0 }).unwrap();
    r.write(&collected(2, "prefill", false, ("tokens", 5), 200), sites, Now { unix: t0 + SEC, cupti: SEC as u64 })
        .unwrap();
    let dir = root.join("date=2026-11-04/rank=3");
    let names = || {
        let mut v: Vec<String> =
            std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
        v.sort();
        v
    };
    assert_eq!(names(), ["125854.kernels.parquet.part", "125854.steps.parquet.part"]);
    r.write(&Collected::default(), sites, Now { unix: t0 + 120 * SEC, cupti: 0 }).unwrap();
    assert_eq!(names(), ["125854.kernels.parquet", "125854.steps.parquet"]);
    let (kernels, steps) = (dir.join("125854.kernels.parquet"), dir.join("125854.steps.parquet"));

    // Both writes land in the pair opened by the first, at its offset.
    let k = rows(&kernels);
    assert_eq!(ints(&k, "start_ns"), [100, 110, 130, 140, 200, 210, 230, 240].map(|s| t0 + s));
    // A call is its label, or its op and index where the manifest gives it none.
    let twice = |step: [Option<&str>; 4]| [step, step].concat().into_iter().map(|s| s.map(String::from)).collect();
    let (calls, ops): (Vec<_>, Vec<_>) = (
        twice([Some("embed#0"), Some("layer.0"), Some("?"), None]),
        twice([Some("embed"), Some("layer"), Some("?"), None]),
    );
    assert_eq!((strings(&k, "call"), strings(&k, "op")), (calls, ops));

    // A graph step is named by its var values, an eager one by their power of two.
    let s = rows(&steps);
    let all = |v: [&str; 2]| v.map(|s| Some(s.to_string())).to_vec();
    assert_eq!(
        (strings(&s, "program"), strings(&s, "shape"), strings(&s, "vars")),
        (all(["decode", "prefill"]), all(["seqs=4", "tokens<=8"]), all(["seqs=4", "tokens=5"]))
    );

    let said = (
        says(&kernels, "kern.rank"),
        says(&steps, "kern.model"),
        says(&steps, "kern.version"),
        says(&steps, "kern.clock_offset_ns"),
        says(&kernels, "kern.manifest_sha256").map(|s| s.len()),
    );
    let some = |s: &str| Some(s.to_string());
    assert_eq!(said, (some("3"), some(&m.model), some("1.2.3"), Some(t0.to_string()), Some(64)));
    assert_eq!(r.rotate().unwrap(), Vec::<std::path::PathBuf>::new());
    std::fs::remove_dir_all(&root).unwrap();
}
