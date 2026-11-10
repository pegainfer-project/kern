//! A server's traced steps as a workload: `kern workload`.
//!
//! `kern-serve --trace-dir` records every step it issued: the program, its
//! var values, the summed sequence lengths and when the step's kernels ran
//! (`steps.parquet`, docs/serve.md). This turns those files into the
//! workload `kern bench` prices, so a manifest is measured on the traffic
//! it served and not on a sweep someone wrote.
//!
//! A step is read back into the call the driver made of it, and the
//! manifest says how: the program's `batch` gives the kind of call and a
//! step's rows, the vars over the groups and rows axes give the rest, and
//! the summed lengths less the call's own rows are its sequences' context.
//! The trace keeps only the sums, so a call's sequences are taken to be
//! equally deep and, where rows are as fed, equally fed. A step of a
//! program the manifest does not have is another manifest's, and refused.
//!
//! Steps are bucketed as the server's own metrics label them: program,
//! shape, context rounded up to a power of two. A scenario is then one
//! `kern_step_compute_seconds_total` series, and what bench measures of it
//! can be held against what that series says. A bucket becomes the call in
//! the middle of its steps, at the bucket's mean context, weighted by
//! their count: the mean keeps the bucket's total context, and a cost
//! linear in context is priced by the total.
//!
//! Every bucket is a sweep, and that is all this does. The file is the
//! whole trace, so it prices to the trace's own total; a smaller workload
//! for a faster loop (the tail dropped, a shape's contexts merged at their
//! weighted mean) is an edit to the file, which says beside each sweep
//! what it stands for. So is a sweep the manifest will not take: the
//! `plan` and `drop` lines name it.
//!
//! The trace also says how much standing a bucket for its steps costs,
//! before any GPU is involved: each bucket is priced at what the
//! server measured for the steps nearest its call, and the sum is held
//! against the time the bucket's own steps took ([`Served::fidelity`]).
//!
//! A trace is one rank's files. Bench issues a call on every rank of the
//! manifest, so one rank's steps are the group's calls, and the ranks'
//! files together would count each call once a rank.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context as _, Result};
use arrow_array::{Array, Int64Array, RecordBatch, StringArray, UInt64Array};
use kern_manifest::protocol::Rows;
use kern_manifest::{Protocol, Verified};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use super::workload::{Plan, Workload};
use super::{digest, say};

/// Measured steps a bucket's price is the median of, before ties.
const NEAR: usize = 25;

#[derive(clap::Args, Debug)]
pub struct ServedOpts {
    /// Steps files a server wrote under --trace-dir, or directories of them
    #[arg(long, required = true, num_args = 1..)]
    trace: Vec<PathBuf>,
    /// The workload to write (TOML)
    #[arg(long)]
    out: PathBuf,
}

/// One traced step, as a steps file records it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Step {
    pub program: String,
    /// The server's shape label: a graph's var values, an eager call's
    /// rounded up to powers of two.
    pub shape: String,
    /// The var values the program read.
    pub vars: BTreeMap<String, u64>,
    /// The call's sequence lengths once its rows are in, summed.
    pub kv: Option<u64>,
    /// First kernel start to last kernel end.
    pub ns: u64,
}

/// The call a step was: `groups` sequences of `rows` rows, a step's or a
/// chunk's, each over `context` tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Call {
    chunk: bool,
    groups: u64,
    rows: u64,
    context: u64,
}

/// A step read back through the protocol; `None` for a program no caller
/// drives (what runs once after load, or inside it). Rows as fed to
/// several sequences are split evenly, to the nearest row the manifest's
/// bound allows.
fn call(p: &Protocol, s: &Step) -> Result<Option<Call>> {
    let Some(f) = p.forwards.iter().find(|f| f.name == s.program) else {
        ensure!(
            p.once.contains(&s.program) || p.derive.contains(&s.program),
            "a `{}` step, and the manifest has no program of that name: the trace is not this manifest's",
            s.program
        );
        return Ok(None);
    };
    let groups = s.vars.get(&p.groups.var).copied().unwrap_or(f.groups);
    let fed = s.vars.get(&p.rows.var).copied();
    ensure!(groups > 0, "a `{}` step of no sequences", s.program);
    let (chunk, rows, own) = match f.rows {
        Rows::Const(r) => {
            ensure!(
                fed.is_none_or(|n| n == groups * r),
                "a `{}` step of {groups} sequences carries `{}` = {}, and the manifest lays it out for {r} rows each: \
                 the trace is not this manifest's",
                s.program,
                p.rows.var,
                fed.unwrap_or(0)
            );
            (false, r, groups * r)
        }
        Rows::Var => {
            let n = fed.with_context(|| {
                format!("a `{}` step carries no `{}`: the trace is not this manifest's", s.program, p.rows.var)
            })?;
            (true, ((n + groups / 2) / groups).min(p.rows.max / groups).max(1), n)
        }
    };
    let context = s.kv.map_or(0, |kv| kv.saturating_sub(own) / groups);
    Ok(Some(Call { chunk, groups, rows, context }))
}

/// Steps of one metric label set and the call that stands for them.
#[derive(Debug, Clone, PartialEq)]
pub struct Bucket {
    pub program: String,
    pub shape: String,
    /// The label's context bound; `None` for a step that wrote no lengths.
    pub kv: Option<u64>,
    pub steps: u64,
    /// The time its steps took.
    pub ns: u64,
    pub chunk: bool,
    pub groups: u64,
    pub rows: u64,
    /// Tokens under each sequence before the call's rows.
    pub context: u64,
    /// Its steps at what the server measured nearest its call.
    pub priced_ns: f64,
}

/// A trace as a workload: a bucket for every label set it holds, most
/// expensive first. Only [`Served::new`] makes one, from steps the
/// protocol could read back, so there is a bucket and every bucket's call
/// is one the manifest declares.
#[derive(Debug, Clone, PartialEq)]
pub struct Served {
    pub buckets: Vec<Bucket>,
    /// Steps of programs no caller drives.
    pub other: u64,
}

/// The call in the middle of a bucket's steps, at its mean context.
fn middle(members: &[(Call, u64)]) -> Call {
    let mut by: Vec<Call> = members.iter().map(|(c, _)| *c).collect();
    by.sort_by_key(|c| (c.chunk, c.groups, c.rows));
    let n = by.len() as u64;
    let context = (by.iter().map(|c| c.context).sum::<u64>() + n / 2) / n;
    Call { context, ..by[by.len() / 2] }
}

/// The median time of the steps nearest `context` in `class`, which is
/// every step of one exact call shape as (context, ns), sorted: the
/// `NEAR` nearest and every step as near as the last of them, so which of
/// several equally near steps count never depends on how long they took.
fn nearest(class: &[(u64, u64)], context: u64) -> f64 {
    let at = class.partition_point(|&(c, _)| c < context);
    let mut off: Vec<u64> = class[at.saturating_sub(NEAR)..(at + NEAR).min(class.len())]
        .iter()
        .map(|&(c, _)| c.abs_diff(context))
        .collect();
    off.sort_unstable();
    let reach = off[(NEAR - 1).min(off.len() - 1)];
    let from = class.partition_point(|&(c, _)| c + reach < context);
    let to = class.partition_point(|&(c, _)| c <= context + reach);
    let mut ns: Vec<u64> = class[from..to].iter().map(|&(_, ns)| ns).collect();
    ns.sort_unstable();
    ns[ns.len() / 2] as f64
}

/// A context bound as the server's `kv` label prints it.
fn bound(kv: Option<u64>) -> String {
    match kv {
        None => String::new(),
        Some(n) if n >= 1 << 20 => format!(" · kv<={}M", n >> 20),
        Some(n) if n >= 1 << 10 => format!(" · kv<={}k", n >> 10),
        Some(n) => format!(" · kv<={n}"),
    }
}

impl Bucket {
    fn label(&self) -> (&str, &str, Option<u64>) {
        (&self.program, &self.shape, self.kv)
    }
}

impl Served {
    /// Bucket `steps` and stand a call for each bucket.
    pub fn new(steps: &[Step], p: &Protocol) -> Result<Served> {
        let read = steps.iter().map(|s| Ok((s, call(p, s)?))).collect::<Result<Vec<_>>>()?;
        let calls: Vec<(&Step, Call)> = read.iter().filter_map(|(s, c)| c.map(|c| (*s, c))).collect();
        ensure!(!calls.is_empty(), "none of the {} steps is of a program a caller drives", steps.len());
        let mut classes: BTreeMap<(&str, bool, u64, u64), Vec<(u64, u64)>> = BTreeMap::new();
        let mut buckets: BTreeMap<(&str, &str, Option<u64>), Vec<(Call, u64)>> = BTreeMap::new();
        for (s, c) in &calls {
            classes.entry((s.program.as_str(), c.chunk, c.groups, c.rows)).or_default().push((c.context, s.ns));
            let kv = s.kv.map(u64::next_power_of_two);
            buckets.entry((s.program.as_str(), s.shape.as_str(), kv)).or_default().push((*c, s.ns));
        }
        classes.values_mut().for_each(|v| v.sort_unstable());
        let mut all: Vec<Bucket> = buckets
            .into_iter()
            .map(|((program, shape, kv), members)| {
                let c = middle(&members);
                let steps = members.len() as u64;
                Bucket {
                    program: program.into(),
                    shape: shape.into(),
                    kv,
                    steps,
                    ns: members.iter().map(|(_, ns)| ns).sum(),
                    chunk: c.chunk,
                    groups: c.groups,
                    rows: c.rows,
                    context: c.context,
                    priced_ns: steps as f64 * nearest(&classes[&(program, c.chunk, c.groups, c.rows)], c.context),
                }
            })
            .collect();
        all.sort_by(|a, b| b.ns.cmp(&a.ns).then_with(|| a.label().cmp(&b.label())));
        Ok(Served { buckets: all, other: (read.len() - calls.len()) as u64 })
    }

    /// The time the bucketed steps took, in ns.
    pub fn ns(&self) -> u64 {
        self.buckets.iter().map(|b| b.ns).sum()
    }

    /// The buckets as their calls price them against the time their steps
    /// took: the relative error overall, and that of the program and shape
    /// furthest off in time.
    pub fn fidelity(&self) -> (f64, Option<(String, f64)>) {
        let mut by: BTreeMap<(&str, &str), (f64, f64)> = BTreeMap::new();
        for b in &self.buckets {
            let e = by.entry((b.program.as_str(), b.shape.as_str())).or_default();
            *e = (e.0 + b.priced_ns, e.1 + b.ns as f64);
        }
        let err = |priced: f64, ns: f64| if ns > 0. { priced / ns - 1. } else { 0. };
        let (priced, ns) = by.values().fold((0., 0.), |a, e| (a.0 + e.0, a.1 + e.1));
        let off = |e: &(f64, f64)| (e.0 - e.1).abs();
        let worst = by.iter().max_by(|a, b| off(a.1).total_cmp(&off(b.1)));
        (err(priced, ns), worst.map(|((program, shape), e)| (format!("{program} {shape}"), err(e.0, e.1))))
    }

    /// The workload file: one sweep a bucket, most expensive first.
    pub fn toml(&self) -> String {
        let mut out = format!(
            "# {} steps, {:.1} s of traced step time in {} buckets\n\
             # priced at the trace's own nearest steps: {:+.2}%\n\
             samples = 16\nseed = 0\n",
            self.buckets.iter().map(|b| b.steps).sum::<u64>(),
            self.ns() as f64 / 1e9,
            self.buckets.len(),
            100. * self.fidelity().0,
        );
        for b in &self.buckets {
            let rows = if b.chunk { "chunk" } else { "rows" };
            write!(
                out,
                "\n# {} · {}{} · {} steps · {:.1} s\n[[sweep]]\ngroups = [{}]\n{rows} = [{}]\ncontext = [{}]\nweight = {}\n",
                b.program,
                b.shape,
                bound(b.kv),
                b.steps,
                b.ns as f64 / 1e9,
                b.groups,
                b.rows,
                b.context,
                b.steps,
            )
            .expect("writing to a string");
        }
        out
    }
}

/// What a set of steps files holds: one rank's steps.
#[derive(Debug, Default)]
pub struct Trace {
    pub steps: Vec<Step>,
    pub files: usize,
    /// Every `kern.manifest_sha256` the files carry.
    pub manifests: BTreeSet<String>,
}

/// Every steps file among `paths`: a file is read as given, a directory
/// is searched to any depth for closed ones.
fn files(paths: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut found = Vec::new();
    for p in paths {
        if !p.is_dir() {
            found.push(p.clone());
            continue;
        }
        let mut inside: Vec<PathBuf> = std::fs::read_dir(p)
            .with_context(|| format!("reading {}", p.display()))?
            .map(|e| Ok(e?.path()))
            .collect::<Result<_>>()?;
        inside.sort();
        let (dirs, here): (Vec<PathBuf>, Vec<PathBuf>) = inside.into_iter().partition(|f| f.is_dir());
        found.extend(here.into_iter().filter(|f| f.to_string_lossy().ends_with("steps.parquet")));
        found.extend(files(&dirs)?);
    }
    Ok(found)
}

fn column<'a, T: Array + 'static>(b: &'a RecordBatch, name: &str) -> Result<&'a T> {
    let c = b.column_by_name(name).with_context(|| format!("no `{name}` column: not a steps file"))?;
    c.as_any().downcast_ref::<T>().with_context(|| format!("column `{name}` is {}", c.data_type()))
}

fn vars(text: &str) -> Result<BTreeMap<String, u64>> {
    text.split(',')
        .filter(|kv| !kv.is_empty())
        .map(|kv| {
            let (name, value) = kv.split_once('=').with_context(|| format!("vars `{text}`"))?;
            Ok((name.to_string(), value.parse().with_context(|| format!("vars `{text}`"))?))
        })
        .collect()
}

fn steps_of(b: &RecordBatch) -> Result<Vec<Step>> {
    let (program, shape, named) =
        (column::<StringArray>(b, "program")?, column::<StringArray>(b, "shape")?, column::<StringArray>(b, "vars")?);
    let (kv, start, end) =
        (column::<UInt64Array>(b, "kv")?, column::<Int64Array>(b, "start_ns")?, column::<Int64Array>(b, "end_ns")?);
    (0..b.num_rows())
        .map(|i| {
            Ok(Step {
                program: program.value(i).to_string(),
                shape: shape.value(i).to_string(),
                vars: vars(named.value(i))?,
                kv: kv.is_valid(i).then(|| kv.value(i)),
                ns: u64::try_from(end.value(i) - start.value(i)).context("a step that ends before it starts")?,
            })
        })
        .collect()
}

/// Read every steps file among `paths`. They are one rank's: a workload
/// is the calls one rank took, and bench issues each on every rank of the
/// manifest, so the files of a group's ranks together would count every
/// call once a rank.
pub fn read(paths: &[PathBuf]) -> Result<Trace> {
    let found = files(paths)?;
    let given = || paths.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ");
    ensure!(!found.is_empty(), "no steps.parquet among {}", given());
    let (trace, ranks) = found.iter().try_fold((Trace::default(), BTreeSet::new()), |(mut t, mut ranks), path| {
        let one = |path: &Path| -> Result<(Vec<Step>, [Option<String>; 2])> {
            let reader = ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(path)?)?;
            let meta = reader.metadata().file_metadata().key_value_metadata();
            let named = |key| meta.and_then(|kv| kv.iter().find(|k| k.key == key).and_then(|k| k.value.clone()));
            let said = [named("kern.manifest_sha256"), named("kern.rank")];
            let batches = reader.build()?.collect::<std::result::Result<Vec<RecordBatch>, _>>()?;
            let steps = batches.iter().map(steps_of).collect::<Result<Vec<_>>>()?;
            Ok((steps.into_iter().flatten().collect(), said))
        };
        let (steps, [sha, rank]) = one(path).with_context(|| path.display().to_string())?;
        t.steps.extend(steps);
        t.manifests.extend(sha);
        t.files += 1;
        ranks.extend(rank);
        anyhow::Ok((t, ranks))
    })?;
    ensure!(
        ranks.len() <= 1,
        "the steps files among {} are of ranks {}, and a workload is one rank's calls: give one rank's files, \
         as in `--trace <dir>/rank=0`",
        given(),
        ranks.iter().map(String::as_str).collect::<Vec<_>>().join(", ")
    );
    Ok(trace)
}

/// Writes the workload the traces in `o` are on the manifest at `path`,
/// saying what it read and how well the workload stands for it.
pub fn run(o: ServedOpts, path: &Path) -> Result<()> {
    let json = std::fs::read(path).with_context(|| format!("manifest {}", path.display()))?;
    // As bench loads it, so the plan below is the plan bench will make.
    let m = Verified::from_json(std::str::from_utf8(&json)?)?.self_hosted()?;
    let protocol = Protocol::check_unsampled(&m)?;
    let trace = read(&o.trace)?;
    let sha = digest(&json);
    ensure!(
        trace.manifests.iter().all(|s| *s == sha),
        "the trace was served on manifest {}, and {} is {sha}",
        trace.manifests.iter().map(String::as_str).collect::<Vec<_>>().join(", "),
        path.display()
    );
    let served = Served::new(&trace.steps, &protocol)?;
    say(
        "trace",
        format!(
            "{} steps · {} file{} · {:.1} s of step time · {} from programs no caller drives{}",
            trace.steps.len(),
            trace.files,
            if trace.files == 1 { "" } else { "s" },
            served.ns() as f64 / 1e9,
            served.other,
            if trace.manifests.is_empty() { " · no manifest sha256 in the files to check" } else { "" },
        ),
    );
    let (err, worst) = served.fidelity();
    let worst = worst.map(|(what, e)| format!(" · worst {what} {:+.2}%", 100. * e)).unwrap_or_default();
    say("fidelity", format!("{:+.2}% priced at the trace's own nearest steps{worst}", 100. * err));
    let text = served.toml();
    let unit = kern_pool::page_unit(&m);
    let plan = Plan::check(&Workload::parse(&text)?, &protocol, unit as usize, kern_pool::row_tokens(&m, unit))?;
    say(
        "plan",
        format!(
            "{} scenarios · {} dropped · {} tokens of state · {} seqs",
            plan.scenarios.len(),
            plan.dropped.len(),
            plan.tokens,
            plan.seqs
        ),
    );
    for d in &plan.dropped {
        say("drop", format!("{} · {}", d.shape, d.why));
    }
    if let Some(parent) = o.out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&o.out, text).with_context(|| o.out.display().to_string())?;
    say("out", o.out.display().to_string());
    Ok(())
}
