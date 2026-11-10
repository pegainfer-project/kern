//! A server's traced steps as a workload.
//!
//! The fixture is the synthetic paged manifest of `workload.rs`: a
//! `decode_batch` for up to 256 sequences of one row, and a `prefill` for
//! one sequence of rows as fed, up to 2,048. Steps are written as the
//! server labels them: a decode step by its exact var values, a chunk by
//! its vars rounded up to powers of two.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray, UInt64Array};
use kern_manifest::protocol::{Positions, Shards};
use kern_manifest::{Protocol, Verified};
use kern_run::bench::served::{read, Served, Step};
use kern_run::bench::workload::{Plan, Workload};
use parquet::arrow::ArrowWriter;
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;

fn manifest() -> Verified {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../kern-manifest/tests/fixtures/paged.json");
    Verified::from_json(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn protocol() -> Protocol {
    Protocol::check(&manifest()).unwrap()
}

fn vars(seqs: u64, tokens: u64) -> BTreeMap<String, u64> {
    BTreeMap::from([("seqs".to_string(), seqs), ("tokens".to_string(), tokens)])
}

/// A decode step of `seqs` sequences, each `context` deep before its row.
fn decode(seqs: u64, context: u64, ns: u64) -> Step {
    Step {
        program: "decode_batch".into(),
        shape: format!("seqs={seqs},tokens={seqs}"),
        vars: vars(seqs, seqs),
        kv: Some(seqs * (context + 1)),
        ns,
    }
}

/// A prefill chunk of `rows` rows on top of `context`.
fn chunk(rows: u64, context: u64, ns: u64) -> Step {
    Step {
        program: "prefill".into(),
        shape: format!("seqs<=1,tokens<={}", rows.next_power_of_two()),
        vars: vars(1, rows),
        kv: Some(context + rows),
        ns,
    }
}

fn served(steps: &[Step]) -> Served {
    Served::new(steps, &protocol()).unwrap()
}

/// Every bucket as (program, chunk, groups, rows, context, steps).
fn calls(s: &Served) -> Vec<(&str, bool, u64, u64, u64, u64)> {
    s.buckets.iter().map(|b| (b.program.as_str(), b.chunk, b.groups, b.rows, b.context, b.steps)).collect()
}

#[test]
fn a_step_is_read_back_into_the_call_its_program_declares() {
    let s = served(&[decode(4, 100, 10), chunk(512, 1000, 30)]);
    assert_eq!(calls(&s), [("prefill", true, 1, 512, 1000, 1), ("decode_batch", false, 4, 1, 100, 1)]);

    // Rows fed unevenly to several sequences: an even split to the nearest row, none of them context.
    let ragged = |rows| Step { shape: "seqs<=4,tokens<=32".into(), vars: vars(4, rows), ..chunk(rows, 400, 5) };
    assert_eq!(calls(&served(&[ragged(31)])), [("prefill", true, 4, 8, 100, 1)]);
    assert_eq!(calls(&served(&[ragged(29)])), [("prefill", true, 4, 7, 100, 1)]);
}

#[test]
fn a_bucket_is_a_label_set_at_its_mean_context() {
    let steps = [decode(4, 100, 10), decode(4, 110, 10), decode(4, 121, 10), decode(4, 1000, 50)];
    let s = served(&steps);
    let buckets: Vec<(Option<u64>, u64, u64, u64)> =
        s.buckets.iter().map(|b| (b.kv, b.steps, b.context, b.ns)).collect();
    assert_eq!(buckets, [(Some(4096), 1, 1000, 50), (Some(512), 3, 110, 30)]);
}

#[test]
fn a_chunk_bucket_is_the_call_in_the_middle_of_its_steps() {
    let full = (0..5).map(|_| chunk(1024, 2000, 100));
    let steps: Vec<Step> = full.chain([chunk(700, 2100, 60), chunk(600, 2200, 50)]).collect();
    assert_eq!(calls(&served(&steps)), [("prefill", true, 1, 1024, 2043, 7)]);
}

#[test]
fn what_no_caller_drives_is_counted_and_another_manifests_trace_is_refused() {
    let load = Step { program: "load".into(), shape: String::new(), vars: BTreeMap::new(), kv: None, ns: 5 };
    let once = Protocol { once: vec!["load".into()], ..protocol() };
    let s = Served::new(&[load.clone(), decode(1, 8, 1)], &once).unwrap();
    assert_eq!((s.other, s.buckets.len(), s.ns()), (1, 1, 1));
    let err = Served::new(&[load, decode(1, 8, 1)], &protocol()).unwrap_err().to_string();
    assert!(err.contains("a `load` step, and the manifest has no program of that name"), "{err}");

    let unfed = Step { vars: BTreeMap::new(), ..chunk(64, 0, 1) };
    let err = Served::new(&[unfed], &protocol()).unwrap_err().to_string();
    assert!(err.contains("carries no `tokens`"), "{err}");
    let wide = Step { vars: vars(4, 64), ..decode(4, 8, 1) };
    let err = Served::new(&[wide], &protocol()).unwrap_err().to_string();
    assert!(err.contains("lays it out for 1 rows each"), "{err}");

    // A member of a group that deals positions traces its share of each length, not the sequence's.
    let positions = Some(Positions { group: "cp".into(), members: 2 });
    let dealt = Protocol { shards: Shards { positions, ..Shards::default() }, ..protocol() };
    let err = Served::new(&[decode(1, 8, 1)], &dealt).unwrap_err().to_string();
    assert!(err.contains("over the 2 members of `cp`"), "{err}");
}

#[test]
fn a_bucket_is_priced_at_what_the_trace_measured_nearest_its_call() {
    // Time linear in context: the mean context prices the bucket exactly.
    let steps: Vec<Step> = (0..11).map(|i| decode(2, 1100 + 10 * i, 2100 + 10 * i)).collect();
    let s = served(&steps);
    assert_eq!((s.buckets.len(), s.buckets[0].context, s.buckets[0].priced_ns), (1, 1150, 11. * 2150.));
    assert_eq!(s.fidelity(), (0., Some(("decode_batch seqs=2,tokens=2".into(), 0.))));

    // Equally near steps all count, however many: not the fastest of them.
    let tied: Vec<Step> = (1..=100).map(|ns| decode(2, 1150, ns)).collect();
    let s = served(&tied);
    assert_eq!((s.buckets[0].priced_ns, s.ns()), (100. * 51., 5050));

    // A slow step far from the others in its bucket is missed by the call at their mean.
    let skewed: Vec<Step> = steps.iter().cloned().chain([decode(2, 2040, 9000)]).collect();
    assert!(served(&skewed).fidelity().0 < 0.);
}

/// splitmix64 over a counter: the tests' only randomness, seeded.
struct Rand(u64);

impl Rand {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let z = (self.0 ^ (self.0 >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        let z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

#[test]
fn any_trace_is_a_workload_the_manifest_takes_whole() {
    let (m, p) = (manifest(), protocol());
    let unit = kern_pool::page_unit(&m);
    let row = kern_pool::row_tokens(&m, unit).unwrap();
    for seed in 0..24 {
        let mut r = Rand(seed);
        let deep = (row - 2048).min(1 << (4 + seed % 12));
        // Every step with the context it was built over, the model the buckets are held against.
        let made: Vec<(Step, u64)> = (0..300)
            .map(|_| {
                let (context, ns) = (r.next() % deep, 1 + r.next() % 1000);
                match r.next() % 3 {
                    0 => (chunk(1 + r.next() % 2048, context, ns), context),
                    _ => (decode(1 << (r.next() % 6), context, ns), context),
                }
            })
            .collect();
        let steps: Vec<Step> = made.iter().map(|(s, _)| s.clone()).collect();
        let s = served(&steps);

        // Every step is in one bucket, with its time.
        let (count, ns) = (s.buckets.iter().map(|b| b.steps).sum::<u64>(), steps.iter().map(|x| x.ns).sum::<u64>());
        assert_eq!((count, s.ns()), (300, ns), "seed {seed}");

        // A bucket's mean context keeps its total, to half a token a step.
        let mut total: BTreeMap<(&str, &str, u64), u64> = BTreeMap::new();
        for (x, context) in &made {
            *total.entry((x.program.as_str(), x.shape.as_str(), x.kv.unwrap().next_power_of_two())).or_default() +=
                context;
        }
        for b in &s.buckets {
            let want = total[&(b.program.as_str(), b.shape.as_str(), b.kv.unwrap())];
            assert!((b.steps * b.context).abs_diff(want) <= b.steps / 2, "seed {seed}: {b:?} against {want}");
        }

        // The order the steps arrive in is not part of the answer.
        let mut reversed = steps.clone();
        reversed.reverse();
        assert_eq!(served(&reversed), s, "seed {seed}");

        // The manifest has a program for every sweep, and they stand for every step.
        let plan = Plan::check(&Workload::parse(&s.toml()).unwrap(), &p, unit as usize, Some(row)).unwrap();
        let weight: u64 = plan.scenarios.iter().map(|x| x.shape.weight).sum();
        assert_eq!((plan.dropped.len(), weight), (0, 300), "seed {seed}");
    }
}

#[test]
fn the_workload_file_says_what_each_sweep_stands_for() {
    let s = served(&[decode(16, 2000, 3_000_000_000), chunk(1024, 0, 500_000_000), decode(1, 5, 1)]);
    assert_eq!(
        s.toml(),
        "# 3 steps, 3.5 s of traced step time in 3 buckets\n\
         # priced at the trace's own nearest steps: +0.00%\n\
         samples = 16\nseed = 0\n\
         \n# decode_batch · seqs=16,tokens=16 · kv<=32k · 1 steps · 3.0 s\n\
         [[sweep]]\ngroups = [16]\nrows = [1]\ncontext = [2000]\nweight = 1\n\
         \n# prefill · seqs<=1,tokens<=1024 · kv<=1k · 1 steps · 0.5 s\n\
         [[sweep]]\ngroups = [1]\nchunk = [1024]\ncontext = [0]\nweight = 1\n\
         \n# decode_batch · seqs=1,tokens=1 · kv<=8 · 1 steps · 0.0 s\n\
         [[sweep]]\ngroups = [1]\nrows = [1]\ncontext = [5]\nweight = 1\n"
    );
}

fn write(path: &std::path::Path, steps: &[Step], sha: &str, rank: u32) {
    let text = |f: fn(&Step) -> String| Arc::new(StringArray::from_iter_values(steps.iter().map(f))) as ArrayRef;
    let named = |s: &Step| s.vars.iter().map(|(n, v)| format!("{n}={v}")).collect::<Vec<_>>().join(",");
    let batch = RecordBatch::try_from_iter([
        ("program", text(|s| s.program.clone())),
        ("shape", text(|s| s.shape.clone())),
        ("vars", text(named)),
        ("kv", Arc::new(steps.iter().map(|s| s.kv).collect::<UInt64Array>()) as ArrayRef),
        ("start_ns", Arc::new(steps.iter().map(|_| 1_000).collect::<Int64Array>()) as ArrayRef),
        ("end_ns", Arc::new(steps.iter().map(|s| 1_000 + s.ns as i64).collect::<Int64Array>()) as ArrayRef),
    ])
    .unwrap();
    let meta = WriterProperties::builder()
        .set_key_value_metadata(Some(vec![
            KeyValue::new("kern.manifest_sha256".to_string(), sha.to_string()),
            KeyValue::new("kern.rank".to_string(), rank.to_string()),
        ]))
        .build();
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut w = ArrowWriter::try_new(std::fs::File::create(path).unwrap(), batch.schema(), Some(meta)).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();
}

#[test]
fn a_trace_directory_is_every_closed_steps_file_under_it() {
    let root = std::env::temp_dir().join(format!("kern-served-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let load = Step { program: "load".into(), shape: String::new(), vars: BTreeMap::new(), kv: None, ns: 5 };
    let (early, late) = (vec![load, decode(4, 100, 10)], vec![chunk(512, 1000, 30)]);
    let day = root.join("date=2026-11-10").join("rank=0");
    write(&day.join("041500.steps.parquet"), &early, "abc", 0);
    write(&day.join("043000.steps.parquet"), &late, "abc", 0);
    write(&day.join("044500.steps.parquet.part"), &late, "open", 0);
    write(&day.join("041500.kernels.parquet"), &late, "not steps", 0);

    let t = read(std::slice::from_ref(&root)).unwrap();
    assert_eq!((t.files, t.manifests.into_iter().collect::<Vec<_>>()), (2, vec!["abc".to_string()]));
    assert_eq!(t.steps, [early.clone(), late.clone()].concat());
    assert_eq!(read(&[day.join("043000.steps.parquet")]).unwrap().steps, late);
    assert!(read(&[root.join("date=2026-11-11")]).is_err());

    // A group's ranks each recorded the same calls: one rank's files are the workload.
    write(&root.join("date=2026-11-10").join("rank=1").join("041500.steps.parquet"), &early, "abc", 1);
    let err = read(std::slice::from_ref(&root)).unwrap_err().to_string();
    assert!(err.contains("are of ranks 0, 1"), "{err}");
    assert_eq!(read(std::slice::from_ref(&day)).unwrap().files, 2);
    std::fs::remove_dir_all(&root).unwrap();
}
