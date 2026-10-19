//! `kern test e2e-logits` (`docs/test.md`): [`kern_test::e2e::Plan`] run
//! over the manifest's ranks.
//!
//! The plan decides every call; this module only carries them out. A
//! manifest with a topology runs as `kern test` runs it: a runtime per
//! rank on its own GPU, every rank fed the same batch. A lease is taken
//! on every rank or on none (the ranks' pools are the same pure function
//! of the same asks, so they agree), a call is staged and run on every
//! rank at once so a collective finds its peers issuing, and the
//! distribution is read back from rank 0, whose logits every rank holds
//! whole. `record` renders the question
//! set the way kern-serve does ([`crate::chat`]) and samples on the host;
//! the test feeds the baseline's tokens and scores the candidate's rows
//! against the baseline's. Neither captures a graph or pads a batch: the
//! numbers are those of the programs the manifest declares, eager.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, ensure, Context, Result};
use clap::{Args, Subcommand};
use kern_manifest::protocol::{Forward, Rows};
use kern_manifest::types::{DType, Dim};
use kern_manifest::{Protocol, Verified};
use kern_pool::{Denied, Lease};
use kern_runtime::{Capacity, HostWeights, Runtime};
use kern_test::diff::{live_bytes, logits_of};
use kern_test::e2e::{self, Judgment, Plan, Position, Recording, Row, Sampling, Step};
use kern_test::report::{plural, row, Verdict};
use sha2::{Digest, Sha256};

use crate::chat::Renderer;
use crate::{Given, Inputs, Vars};

/// Flags of `kern test e2e-logits`: a candidate against a baseline, or
/// `record` to sample the baseline.
#[derive(Args)]
#[command(args_conflicts_with_subcommands = true, subcommand_negates_reqs = true)]
pub struct E2eOpts {
    #[command(subcommand)]
    record: Option<Sub>,
    #[command(flatten)]
    load: Load,
    /// The baseline recording to score against
    #[arg(long, required = true)]
    reference: Option<PathBuf>,
    /// The largest mean KL(baseline ‖ candidate), in nats, that still PASSes
    #[arg(long, required = true)]
    kl: Option<f64>,
    /// Write the candidate's recording here (the baseline's schema)
    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Subcommand)]
enum Sub {
    /// Sample a baseline recording over a question set
    Record(RecordOpts),
}

#[derive(Args)]
struct RecordOpts {
    #[command(flatten)]
    load: Load,
    /// The question set: parquet with `source`, `messages`, `tools`
    #[arg(long)]
    questions: PathBuf,
    /// Sampling temperature, 0 greedy (default: the checkpoint's generation_config.json)
    #[arg(long)]
    temperature: Option<f64>,
    /// Draw among the k most likely tokens, 0 all of them (default: generation_config.json, else 0)
    #[arg(long)]
    top_k: Option<usize>,
    /// Draw within the smallest set holding this much mass (default: generation_config.json, else 1)
    #[arg(long)]
    top_p: Option<f64>,
    /// Sampler seed; a draw depends on the seed, the row and the step alone
    #[arg(long)]
    seed: u64,
    /// Output tokens a row samples at most; it stops earlier at a stop token
    #[arg(long, default_value_t = 16 << 10)]
    max_new: usize,
    /// Where the recording is written
    #[arg(long)]
    out: PathBuf,
}

#[derive(Args)]
struct Load {
    /// The manifest to drive
    #[arg(long, required = true)]
    manifest: Option<PathBuf>,
    /// Directory of the manifest's cubins
    #[arg(long)]
    kernels: Option<PathBuf>,
    /// Checkpoint directories or .safetensors files
    #[arg(long, required = true)]
    weights: Vec<String>,
    /// CUDA device ordinals, one per rank (a single one starts the ranks there)
    #[arg(long, value_delimiter = ',', default_value = "0")]
    gpu: Vec<usize>,
}

/// `kern test e2e-logits`: the exit code, 0 PASS and 1 FAIL; an error is
/// the caller's to report (`kern` exits 2).
pub fn run(o: E2eOpts) -> Result<i32> {
    match o.record {
        Some(Sub::Record(r)) => record(r).map(|()| 0),
        None => test(
            &o.load,
            o.reference.as_deref().expect("clap requires --reference"),
            o.kl.expect("clap requires --kl"),
            o.out.as_deref(),
        ),
    }
}

fn record(o: RecordOpts) -> Result<()> {
    let t0 = Instant::now();
    let inputs = o.load.inputs()?;
    let questions = e2e::questions(&o.questions)?;
    let dir =
        inputs.weights.dirs().into_iter().find(|d| d.join("config.json").is_file()).with_context(|| {
            "no --weights directory has a config.json: the chat renderer is picked by its `model_type`"
        })?;
    let renderer = Renderer::load(&dir)?;
    let sampling = sampling_of(&dir, &o)?;
    let prompts: Vec<Vec<i64>> = questions
        .iter()
        .map(|q| renderer.prompt(&q.messages, q.tools.as_deref()).with_context(|| format!("rendering `{}`", q.source)))
        .collect::<Result<_>>()?;
    let mut d = Device::load(&o.load, &inputs)?;
    println!("{}", row("record", format!("{} over {}", d.producer, plural(questions.len(), "question")), None));
    let mut plan = Plan::sampled(&prompts, o.max_new, &inputs.stop_tokens, d.groups, d.chunk_rows, d.chunk_scores)?;
    let names: Vec<String> = questions.iter().map(|q| q.source.clone()).collect();
    let seed = o.seed;
    let scored = d.drive(
        &mut plan,
        &names,
        |r, k, logits| (e2e::top(logits), e2e::sample(logits, sampling, seed, r, k)),
        |s| Some(s.1),
    )?;
    let rows: Vec<Row> = scored
        .into_iter()
        .enumerate()
        .map(|(r, s)| Row {
            source: names[r].clone(),
            prompt: narrow(&prompts[r]),
            output: narrow(plan.output(r)),
            top: s.into_iter().map(|(t, _)| t).collect(),
        })
        .collect();
    let meta = BTreeMap::from([
        ("producer".to_string(), d.producer.clone()),
        ("questions".to_string(), format!("sha256:{}", sha256(&o.questions)?)),
        ("temperature".to_string(), sampling.temperature.to_string()),
        ("top_k".to_string(), sampling.top_k.to_string()),
        ("top_p".to_string(), sampling.top_p.to_string()),
        ("seed".to_string(), o.seed.to_string()),
        ("max_new".to_string(), o.max_new.to_string()),
    ]);
    let positions: usize = rows.iter().map(|r| r.output.len()).sum();
    let rec = Recording { meta, rows };
    rec.write(&o.out)?;
    println!("{}", d.coverage());
    println!("{}", row("out", o.out.display().to_string(), None));
    let summary = format!("{} · {}", plural(rec.rows.len(), "row"), plural(positions, "scored position"));
    println!("{}", row("recorded", summary, Some(t0.elapsed().as_secs_f32())));
    Ok(())
}

fn test(load: &Load, reference: &Path, limit: f64, out: Option<&Path>) -> Result<i32> {
    let t0 = Instant::now();
    let base = Recording::read(reference)?;
    let inputs = load.inputs()?;
    let mut d = Device::load(load, &inputs)?;
    let vocab = d.vocab;
    for r in &base.rows {
        let ids = r.prompt.iter().chain(&r.output).chain(r.top.iter().flat_map(|t| &t.ids));
        if let Some(id) = ids.copied().find(|&id| !(0..vocab as i32).contains(&id)) {
            bail!("{}: row `{}` holds token {id}; `{}` has {vocab} columns", reference.display(), r.source, d.logits);
        }
    }
    println!("{}", row("e2e", format!("{} against {}", d.producer, reference.display()), None));
    let rows: Vec<(Vec<i64>, Vec<i64>)> = base.rows.iter().map(|r| (widen(&r.prompt), widen(&r.output))).collect();
    let mut plan = Plan::forced(&rows, d.groups, d.chunk_rows, d.chunk_scores)?;
    let names: Vec<String> = base.rows.iter().map(|r| r.source.clone()).collect();
    let keep = out.is_some();
    let scored = d.drive(
        &mut plan,
        &names,
        |r, k, logits| {
            let p = Position::of(r, k, rows[r].0.len(), &base.rows[r].top[k], logits);
            (keep.then(|| e2e::top(logits)), p)
        },
        |_| None,
    )?;
    let positions: Vec<Position> = scored.iter().flatten().map(|(_, p)| p.clone()).collect();
    let j = Judgment::of(&positions, limit);
    let mut lines = [vec![d.coverage()], j.lines(&names)].concat();
    if let Some(out) = out {
        let rows = scored
            .into_iter()
            .zip(&base.rows)
            .map(|(s, b)| Row { top: s.into_iter().filter_map(|(t, _)| t).collect(), ..b.clone() })
            .collect();
        let meta = base.meta.clone().into_iter().chain([("producer".to_string(), d.producer.clone())]).collect();
        Recording { meta, rows }.write(out)?;
        lines.push(row("out", out.display().to_string(), None));
    }
    lines.iter().for_each(|l| println!("{l}"));
    let code = i32::from(!j.pass());
    Verdict::new(code, j.summary(), t0.elapsed().as_secs_f32()).lines().iter().for_each(|l| println!("{l}"));
    Ok(code)
}

/// The flags over the checkpoint's `generation_config.json`; `do_sample:
/// false` there is the argmax.
fn sampling_of(dir: &Path, o: &RecordOpts) -> Result<Sampling> {
    let path = dir.join("generation_config.json");
    let g: serde_json::Value = match std::fs::read_to_string(&path) {
        Ok(t) => serde_json::from_str(&t).with_context(|| format!("{} is not JSON", path.display()))?,
        Err(_) => serde_json::Value::Null,
    };
    let greedy = g.get("do_sample").and_then(serde_json::Value::as_bool) == Some(false);
    let temperature = o
        .temperature
        .or(greedy.then_some(0.0))
        .or_else(|| g.get("temperature").and_then(serde_json::Value::as_f64))
        .with_context(|| format!("no --temperature and no `temperature` in {}", path.display()))?;
    let top_k = o.top_k.or_else(|| g.get("top_k").and_then(serde_json::Value::as_u64).map(|k| k as usize)).unwrap_or(0);
    let top_p = o.top_p.or_else(|| g.get("top_p").and_then(serde_json::Value::as_f64)).unwrap_or(1.0);
    Sampling::new(temperature, top_k, top_p)
}

impl Load {
    fn inputs(&self) -> Result<Inputs> {
        let given = Given {
            manifest: self.manifest.clone(),
            kernels: self.kernels.clone(),
            weights: self.weights.clone(),
            ..Default::default()
        };
        Inputs::resolve(given, None, None)
    }
}

/// One rank and the leases of the rows it holds.
struct Rank {
    rt: Runtime,
    leases: BTreeMap<usize, Lease>,
}

/// The manifest on its ranks and what the plan needs to know about it.
struct Device {
    ranks: Vec<Rank>,
    p: Protocol,
    /// The program a prompt chunk goes through, and its rows at most.
    chunk: Forward,
    chunk_rows: usize,
    /// Whether the chunk program leaves the next distribution behind.
    chunk_scores: bool,
    /// Sequences a decode call takes at most.
    groups: usize,
    /// The buffer a distribution is read from: one row per sequence a decode
    /// call runs, the last row a chunk's.
    logits: String,
    dtype: DType,
    /// The logits row's width: the ids a distribution can hold.
    vocab: usize,
    /// The manifest's file name and sha256.
    producer: String,
    /// Calls run, by program.
    calls: BTreeMap<String, usize>,
}

impl Device {
    fn load(load: &Load, inputs: &Inputs) -> Result<Device> {
        let path = &inputs.manifest;
        let json = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let m = Verified::from_json(&json)
            .and_then(|m| m.self_hosted())
            .with_context(|| format!("{} failed verification", path.display()))?;
        let p = Protocol::check(&m).with_context(|| format!("{} does not fit the serving protocol", path.display()))?;
        let chunk =
            p.chunk().cloned().context("no program takes one sequence's rows as fed: nothing feeds a prompt")?;
        let decodes: Vec<&Forward> = p.forwards.iter().filter(|f| f.rows == Rows::Const(1)).collect();
        ensure!(!decodes.is_empty(), "no program takes sequences of one row: nothing decodes");
        let writes = |f: &Forward| logits_of(&m, &f.name);
        let logits = writes(decodes[0])
            .into_iter()
            .min_by_key(|b| b.len())
            .with_context(|| format!("`{}` writes no `logits*` buffer: nothing to score", decodes[0].name))?;
        for f in &decodes {
            ensure!(
                writes(f).contains(&logits),
                "`{}` writes no `{logits}`, `{}` does: one buffer is read after every decode",
                f.name,
                decodes[0].name
            );
        }
        let chunk_scores = writes(&chunk).contains(&logits);
        let vocab = match m.buffers[&logits].shape[..] {
            [_, Dim::Const(v)] => v as usize,
            _ => bail!("`{logits}` is not [rows, vocab] with a constant vocab"),
        };
        let groups = p.max_groups(Rows::Const(1)) as usize;
        let capacity = Capacity { tokens: None, seqs: groups as u64, landings: 0 };
        let mut rts = crate::load_ranks(&m, inputs, &load.gpu, capacity, &HostWeights::new(), None)?;
        crate::each(&mut rts, "running what the manifest runs once", |rt| crate::run_once(rt, &p))?;
        let name = path.file_name().map_or_else(|| path.display().to_string(), |n| n.to_string_lossy().into_owned());
        Ok(Device {
            ranks: rts.into_iter().map(|rt| Rank { rt, leases: BTreeMap::new() }).collect(),
            chunk_rows: p.rows.max as usize,
            dtype: m.buffers[&logits].dtype,
            vocab,
            producer: format!("{name} sha256:{}", hex::encode(Sha256::digest(json.as_bytes()))),
            p,
            chunk,
            chunk_scores,
            groups,
            logits,
            calls: BTreeMap::new(),
        })
    }

    /// Carry out `plan` to the end. `score` turns the distribution that
    /// predicts output position `k` of row `r` into what the caller keeps;
    /// `drawn` reads back the token a sampled row goes on with. What was
    /// kept, per row in output order.
    fn drive<S: Send>(
        &mut self,
        plan: &mut Plan,
        names: &[String],
        score: impl Fn(usize, usize, &[f32]) -> S + Sync,
        drawn: impl Fn(&S) -> Option<i64>,
    ) -> Result<Vec<Vec<S>>> {
        let t0 = Instant::now();
        let mut kept: Vec<Vec<S>> = names.iter().map(|_| Vec::new()).collect();
        let mut done = 0;
        while let Some(step) = plan.step() {
            match step {
                Step::Lease { row, tokens } => plan.leased(self.lease(&names[row], row, tokens)?),
                Step::Prefill { row, at, ids, scores } => {
                    let chunk = self.chunk.name.clone();
                    let vars = self.call(&chunk, &[(row, at)], ids.len(), &ids)?;
                    if scores {
                        let rows = self.read(&vars)?;
                        let s = score(row, plan.position(row), &rows[rows.len() - 1]);
                        plan.scored(drawn(&s));
                        kept[row].push(s);
                    }
                }
                Step::Decode(batch) => {
                    let f = self
                        .p
                        .forward(batch.len() as u64, Rows::Const(1))
                        .cloned()
                        .with_context(|| format!("no program takes {} sequences of one row", batch.len()))?;
                    let at: Vec<(usize, usize)> = batch.iter().map(|&(r, pos, _)| (r, pos)).collect();
                    let ids: Vec<i64> = batch.iter().map(|b| b.2).collect();
                    let vars = self.call(&f.name, &at, 1, &ids)?;
                    let rows = self.read(&vars)?;
                    ensure!(
                        rows.len() >= batch.len(),
                        "`{}` holds {} rows after {} sequences",
                        self.logits,
                        rows.len(),
                        batch.len()
                    );
                    let ss = par_map(batch.len(), |i| score(batch[i].0, plan.position(batch[i].0), &rows[i]));
                    for (s, &(r, ..)) in ss.into_iter().zip(&batch) {
                        plan.scored(drawn(&s));
                        kept[r].push(s);
                    }
                }
                Step::Release(row) => {
                    self.ranks.iter_mut().for_each(|k| drop(k.leases.remove(&row)));
                    done += 1;
                    eprintln!(
                        "{done}/{} · `{}` · {} · {:.0}s",
                        names.len(),
                        names[row],
                        plural(kept[row].len(), "scored position"),
                        t0.elapsed().as_secs_f32()
                    );
                }
            }
        }
        Ok(kept)
    }

    /// `tokens` slots for `row` on every rank, or on none: `false` when the
    /// pool is busy with the rows running. A remap in flight is waited
    /// out, so whether a row fits does not depend on the clock; a row that
    /// can never fit is an error, never a row left out of the score.
    fn lease(&mut self, name: &str, row: usize, tokens: usize) -> Result<bool> {
        let idle = self.ranks[0].leases.is_empty();
        let leases: Vec<Option<Lease>> = self
            .ranks
            .iter_mut()
            .map(|k| match lease(&mut k.rt, tokens) {
                Ok(l) => Ok(Some(l)),
                Err(kern_runtime::Error::Denied(Denied::Busy)) if !idle => Ok(None),
                Err(e) => Err(anyhow::Error::new(e).context(format!("row `{name}` needs {tokens} token slots"))),
            })
            .collect::<Result<_>>()?;
        let Some(leases) = leases.into_iter().collect::<Option<Vec<Lease>>>() else { return Ok(false) };
        self.ranks.iter_mut().zip(leases).for_each(|(k, l)| drop(k.leases.insert(row, l)));
        Ok(true)
    }

    /// `prog` over the sequences `at` (row, position), `rows` rows each.
    fn call(&mut self, prog: &str, at: &[(usize, usize)], rows: usize, ids: &[i64]) -> Result<Vars> {
        *self.calls.entry(prog.to_string()).or_default() += 1;
        let p = &self.p;
        let positions: Vec<usize> = at.iter().map(|a| a.1).collect();
        let vars = crate::each(&mut self.ranks, &format!("`{prog}`"), |k| {
            let leases: Vec<&Lease> = at.iter().map(|(r, _)| &k.leases[r]).collect();
            let vars = crate::stage_batch(&mut k.rt, p, &leases, &positions, rows, ids)?;
            k.rt.run(prog, &vars)?;
            Ok(vars)
        })?;
        Ok(vars.into_iter().next().expect("a manifest has a rank"))
    }

    /// The live rows of the logits buffer after a call at `vars`, rank 0's.
    fn read(&self, vars: &Vars) -> Result<Vec<Vec<f32>>> {
        let rt = &self.ranks[0].rt;
        let len = live_bytes(&rt.manifest, &self.logits, vars);
        let bytes = rt.read_buffer_prefix(&self.logits, len).with_context(|| format!("reading `{}`", self.logits))?;
        let (dtype, w) = (self.dtype, self.vocab * self.dtype.bytes() as usize);
        Ok(par_map(bytes.len() / w, |i| floats(dtype, &bytes[i * w..(i + 1) * w])))
    }

    /// The programs the run drove, a call count each.
    fn coverage(&self) -> String {
        let calls: Vec<String> = self.calls.iter().map(|(p, n)| format!("`{p}` ×{n}")).collect();
        row("programs", calls.join(" · "), None)
    }
}

/// A lease, a remap in flight settled and asked again.
fn lease(rt: &mut Runtime, tokens: usize) -> kern_runtime::Result<Lease> {
    loop {
        match rt.lease(tokens) {
            Err(kern_runtime::Error::Denied(Denied::Remapping)) => rt.settle()?,
            r => return r,
        }
    }
}

/// `f` over `0..n` on every core, in order.
fn par_map<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let threads = std::thread::available_parallelism().map_or(1, |t| t.get()).min(n.max(1));
    let per = n.div_ceil(threads);
    std::thread::scope(|s| {
        let f = &f;
        let hs: Vec<_> = (0..threads)
            .map(|t| s.spawn(move || (t * per..((t + 1) * per).min(n)).map(f).collect::<Vec<T>>()))
            .collect();
        hs.into_iter().flat_map(|h| h.join().expect("a scoring thread panicked")).collect()
    })
}

fn floats(dtype: DType, b: &[u8]) -> Vec<f32> {
    match dtype {
        DType::Bf16 => {
            b.as_chunks::<2>().0.iter().map(|&c| f32::from_bits((u16::from_le_bytes(c) as u32) << 16)).collect()
        }
        DType::F32 => b.as_chunks::<4>().0.iter().map(|&c| f32::from_le_bytes(c)).collect(),
        _ => kern_manifest::values::to_f64(dtype, b).into_iter().map(|x| x as f32).collect(),
    }
}

fn sha256(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(hex::encode(Sha256::digest(&bytes)))
}

fn widen(v: &[i32]) -> Vec<i64> {
    v.iter().map(|&x| x as i64).collect()
}

fn narrow(v: &[i64]) -> Vec<i32> {
    v.iter().map(|&x| x as i32).collect()
}
