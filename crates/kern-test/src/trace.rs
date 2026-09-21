//! A recorded reference: what a manifest predicts over a fixed corpus,
//! kept as one parquet table, so a candidate is judged against a file
//! and nothing else has to be loaded.
//!
//! The corpus is prompts of token ids and one number, `decode`: of each
//! prompt all but its last `decode + 1` tokens go through the chunk
//! program at once, then the next `decode` tokens are fed one at a time,
//! and the distribution over the next token is kept at every one of those
//! `decode + 1` positions: the top-[`TOP`] ids with their log
//! probabilities and the log probability of the token the corpus
//! actually has there. So the prefill path is scored once per prompt and
//! the decode path `decode` times, on state the prefill built; a
//! prefill-only manifest takes its steps as chunks of one row.
//!
//! The table has a row per token position: `prompt`, `pos`, `token`, and
//! `producer` null on that row. A producer that scored the position adds
//! a row with its name and `ref_logprob`, `top_ids`, `top_logprob`; a
//! corpus is a table nobody has scored yet. `decode` is file metadata.
//! Several producers in one file are the band any candidate is read
//! against: no farther from any of them than they are from each other.
//! One producer alone falls back to a fixed KL limit.
//!
//! Everything here but the two functions that drive a [`Side`] is
//! `fn(data) -> data`; the crate's tests reach every verdict through the
//! fake side.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{ensure, Context, Result};
use arrow_array::builder::{Float32Builder, Int32Builder, ListBuilder, StringBuilder};
use arrow_array::{Array, ArrayRef, Float32Array, Int32Array, ListArray, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use kern_manifest::protocol::{Forward, Rows};
use kern_manifest::types::{DType, Manifest};
use kern_manifest::{values, Protocol, Verified};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use serde::Serialize;

use crate::compare::TOP;
use crate::diff::{access, live_bytes, row_elems};
use crate::report::{row, Verdict};
use crate::Side;

/// One scored position: the distribution a producer predicted for the
/// token after `pos` of `prompt`, truncated to its top-[`TOP`].
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Score {
    pub prompt: usize,
    pub pos: usize,
    /// Log probability of the corpus token at `pos + 1`.
    pub ref_logprob: f64,
    /// Most likely first.
    pub ids: Vec<i64>,
    pub logprob: Vec<f64>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Trace {
    pub prompts: Vec<Vec<i64>>,
    pub decode: usize,
    pub producers: BTreeMap<String, Vec<Score>>,
}

/// Where a prompt's scored positions are: `ctx` tokens go through the
/// chunk program, then `steps` more one at a time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Schedule {
    pub ctx: usize,
    pub steps: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Phase {
    Prefill,
    Decode,
}

impl Schedule {
    pub fn phase(&self, pos: usize) -> Phase {
        if pos + 1 == self.ctx {
            Phase::Prefill
        } else {
            Phase::Decode
        }
    }
}

impl Trace {
    pub fn corpus(prompts: Vec<Vec<i64>>, decode: usize) -> Result<Trace> {
        ensure!(!prompts.is_empty(), "the corpus has no prompts");
        for (i, p) in prompts.iter().enumerate() {
            ensure!(p.len() >= 2, "prompt {i} has {} token(s); a scored position needs two", p.len());
        }
        Ok(Trace { prompts, decode, producers: BTreeMap::new() })
    }

    /// A prompt keeps at least one token for the prefill and one to
    /// predict; the decode tail shrinks to fit a short prompt.
    pub fn schedule(&self, prompt: usize) -> Schedule {
        let n = self.prompts[prompt].len();
        let steps = self.decode.min(n - 2);
        Schedule { ctx: n - 1 - steps, steps }
    }

    /// Scored positions per prompt, summed.
    pub fn positions(&self) -> usize {
        (0..self.prompts.len()).map(|i| self.schedule(i).steps + 1).sum()
    }

    /// The first `n` prompts, with their scores.
    pub fn take(&self, n: usize) -> Trace {
        let n = n.min(self.prompts.len());
        Trace {
            prompts: self.prompts[..n].to_vec(),
            decode: self.decode,
            producers: self
                .producers
                .iter()
                .map(|(p, s)| (p.clone(), s.iter().filter(|s| s.prompt < n).cloned().collect()))
                .collect(),
        }
    }

    /// The file starts with parquet's magic.
    pub fn is_parquet(path: &Path) -> bool {
        std::fs::read(path).is_ok_and(|b| b.starts_with(b"PAR1"))
    }

    pub fn read(path: &Path) -> Result<Trace> {
        let f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let b =
            ParquetRecordBatchReaderBuilder::try_new(f).with_context(|| format!("{} as parquet", path.display()))?;
        let meta = b.schema().metadata().clone();
        let decode: usize = meta
            .get("decode")
            .ok_or_else(|| anyhow::anyhow!("{}: no `decode` in the file metadata", path.display()))?
            .parse()
            .with_context(|| format!("{}: `decode` metadata", path.display()))?;
        let expect: Vec<(&str, DataType)> = vec![
            ("prompt", DataType::Int32),
            ("pos", DataType::Int32),
            ("token", DataType::Int32),
            ("producer", DataType::Utf8),
            ("ref_logprob", DataType::Float32),
            ("top_ids", list_of(DataType::Int32)),
            ("top_logprob", list_of(DataType::Float32)),
        ];
        for (name, dt) in &expect {
            let f = b
                .schema()
                .field_with_name(name)
                .map_err(|_| anyhow::anyhow!("{}: no column `{name}`", path.display()))?;
            ensure!(f.data_type() == dt, "{}: column `{name}` is {} , expected {dt}", path.display(), f.data_type());
        }
        let mut prompts: BTreeMap<usize, BTreeMap<usize, i64>> = BTreeMap::new();
        let mut producers: BTreeMap<String, Vec<Score>> = BTreeMap::new();
        for batch in b.build()? {
            let batch = batch?;
            let col = |n: &str| batch.column_by_name(n).expect("checked above");
            let prompt = col("prompt").as_any().downcast_ref::<Int32Array>().expect("i32");
            let pos = col("pos").as_any().downcast_ref::<Int32Array>().expect("i32");
            let token = col("token").as_any().downcast_ref::<Int32Array>().expect("i32");
            let producer = col("producer").as_any().downcast_ref::<StringArray>().expect("utf8");
            let lp = col("ref_logprob").as_any().downcast_ref::<Float32Array>().expect("f32");
            let ids = col("top_ids").as_any().downcast_ref::<ListArray>().expect("list");
            let lps = col("top_logprob").as_any().downcast_ref::<ListArray>().expect("list");
            for r in 0..batch.num_rows() {
                let (p, q) = (prompt.value(r) as usize, pos.value(r) as usize);
                if producer.is_null(r) {
                    prompts.entry(p).or_default().insert(q, token.value(r) as i64);
                    continue;
                }
                let v = ids.value(r);
                let w = lps.value(r);
                producers.entry(producer.value(r).to_string()).or_default().push(Score {
                    prompt: p,
                    pos: q,
                    ref_logprob: lp.value(r) as f64,
                    ids: v
                        .as_any()
                        .downcast_ref::<Int32Array>()
                        .expect("i32")
                        .values()
                        .iter()
                        .map(|&x| x as i64)
                        .collect(),
                    logprob: w
                        .as_any()
                        .downcast_ref::<Float32Array>()
                        .expect("f32")
                        .values()
                        .iter()
                        .map(|&x| x as f64)
                        .collect(),
                });
            }
        }
        let prompts: Vec<Vec<i64>> = prompts
            .into_iter()
            .enumerate()
            .map(|(i, (p, toks))| {
                ensure!(i == p, "{}: prompts are not numbered 0..: {p} at {i}", path.display());
                ensure!(
                    toks.keys().enumerate().all(|(k, &q)| k == q),
                    "{}: prompt {p} has gaps in its positions",
                    path.display()
                );
                Ok(toks.into_values().collect())
            })
            .collect::<Result<_>>()?;
        for s in producers.values_mut() {
            s.sort_by_key(|s| (s.prompt, s.pos));
        }
        let t = Trace::corpus(prompts, decode)?;
        Ok(Trace { producers, ..t })
    }

    pub fn write(&self, path: &Path) -> Result<()> {
        let mut prompt = Int32Builder::new();
        let mut pos = Int32Builder::new();
        let mut token = Int32Builder::new();
        let mut producer = StringBuilder::new();
        let mut lp = Float32Builder::new();
        let mut ids = ListBuilder::new(Int32Builder::new());
        let mut lps = ListBuilder::new(Float32Builder::new());
        let mut by_pos: BTreeMap<(usize, usize), Vec<(&str, &Score)>> = BTreeMap::new();
        for (name, scores) in &self.producers {
            for s in scores {
                by_pos.entry((s.prompt, s.pos)).or_default().push((name, s));
            }
        }
        for (p, toks) in self.prompts.iter().enumerate() {
            for (q, &t) in toks.iter().enumerate() {
                prompt.append_value(p as i32);
                pos.append_value(q as i32);
                token.append_value(t as i32);
                producer.append_null();
                lp.append_null();
                ids.append_null();
                lps.append_null();
                for (name, s) in by_pos.get(&(p, q)).into_iter().flatten() {
                    prompt.append_value(p as i32);
                    pos.append_value(q as i32);
                    token.append_value(t as i32);
                    producer.append_value(name);
                    lp.append_value(s.ref_logprob as f32);
                    ids.values().append_slice(&s.ids.iter().map(|&x| x as i32).collect::<Vec<_>>());
                    ids.append(true);
                    lps.values().append_slice(&s.logprob.iter().map(|&x| x as f32).collect::<Vec<_>>());
                    lps.append(true);
                }
            }
        }
        let schema = Arc::new(
            Schema::new(vec![
                Field::new("prompt", DataType::Int32, false),
                Field::new("pos", DataType::Int32, false),
                Field::new("token", DataType::Int32, false),
                Field::new("producer", DataType::Utf8, true),
                Field::new("ref_logprob", DataType::Float32, true),
                Field::new("top_ids", list_of(DataType::Int32), true),
                Field::new("top_logprob", list_of(DataType::Float32), true),
            ])
            .with_metadata(BTreeMap::from([("decode".to_string(), self.decode.to_string())])),
        );
        let cols: Vec<ArrayRef> = vec![
            Arc::new(prompt.finish()),
            Arc::new(pos.finish()),
            Arc::new(token.finish()),
            Arc::new(producer.finish()),
            Arc::new(lp.finish()),
            Arc::new(ids.finish()),
            Arc::new(lps.finish()),
        ];
        let batch = RecordBatch::try_new(schema.clone(), cols)?;
        let f = std::fs::File::create(path).with_context(|| format!("creating {}", path.display()))?;
        let mut w = ArrowWriter::try_new(f, schema, None)?;
        w.write(&batch)?;
        w.close()?;
        Ok(())
    }
}

fn list_of(dt: DataType) -> DataType {
    DataType::List(Arc::new(Field::new("item", dt, true)))
}

/// One position's distribution from a logits row: log-softmax, the
/// top-[`TOP`] and the corpus token's log probability.
pub fn score(prompt: usize, pos: usize, row: &[f64], next: i64) -> Score {
    let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let lse = max + row.iter().map(|x| (x - max).exp()).sum::<f64>().ln();
    let mut idx: Vec<usize> = (0..row.len()).collect();
    idx.sort_by(|&i, &j| row[j].total_cmp(&row[i]).then(i.cmp(&j)));
    idx.truncate(TOP);
    Score {
        prompt,
        pos,
        ref_logprob: row.get(next as usize).map_or(f64::NEG_INFINITY, |x| x - lse),
        ids: idx.iter().map(|&i| i as i64).collect(),
        logprob: idx.iter().map(|&i| row[i] - lse).collect(),
    }
}

/// One position, a reference `a` against a candidate: did the argmax
/// move, how sure was `a`, how much probability mass moved (KL(a‖b) over
/// `a`'s kept tokens with the rest as one bucket), and where `a`'s
/// argmax ranks in `b`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RowCmp {
    pub argmax_a: i64,
    pub argmax_b: i64,
    pub margin_a: f64,
    pub kl: f64,
    /// 1-based; 0 when beyond what `b` kept.
    pub rank_in_b: usize,
}

impl RowCmp {
    pub fn flip(&self) -> bool {
        self.argmax_a != self.argmax_b
    }
}

/// `a` against a full logits row.
pub fn against(a: &Score, b: &[f64]) -> RowCmp {
    let max = b.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let lse = max + b.iter().map(|x| (x - max).exp()).sum::<f64>().ln();
    let argmax_b = (0..b.len()).max_by(|&i, &j| b[i].total_cmp(&b[j]).then(j.cmp(&i))).unwrap_or(0) as i64;
    let a0 = a.ids[0] as usize;
    let rank = b.get(a0).map_or(0, |&x| 1 + b.iter().filter(|&&y| y > x).count());
    let q = |id: i64| b.get(id as usize).map(|x| x - lse);
    cmp(a, argmax_b, rank, q)
}

/// `a` against another kept score: a token `b` did not keep is taken at
/// the least it did keep, so the KL is what the table can say.
pub fn between(a: &Score, b: &Score) -> RowCmp {
    let floor = b.logprob.last().copied().unwrap_or(f64::NEG_INFINITY);
    let q = |id: i64| Some(b.ids.iter().position(|&x| x == id).map_or(floor, |i| b.logprob[i]));
    let rank = b.ids.iter().position(|&x| x == a.ids[0]).map_or(0, |i| i + 1);
    cmp(a, b.ids[0], rank, q)
}

fn cmp(a: &Score, argmax_b: i64, rank_in_b: usize, q: impl Fn(i64) -> Option<f64>) -> RowCmp {
    let (mut kl, mut p_kept, mut q_kept) = (0.0, 0.0, 0.0);
    for (&id, &lp) in a.ids.iter().zip(&a.logprob) {
        let p = lp.exp();
        match q(id) {
            Some(lq) if lq.is_finite() => {
                kl += p * (lp - lq);
                q_kept += lq.exp();
            }
            _ => kl = f64::INFINITY,
        }
        p_kept += p;
    }
    let (p_rest, q_rest) = ((1.0 - p_kept).max(1e-12), (1.0 - q_kept).max(1e-12));
    kl += p_rest * (p_rest / q_rest).ln();
    RowCmp {
        argmax_a: a.ids[0],
        argmax_b,
        margin_a: a.logprob.first().copied().unwrap_or(0.0) - a.logprob.get(1).copied().unwrap_or(f64::NEG_INFINITY),
        kl: if kl.is_nan() { f64::INFINITY } else { kl.max(0.0) },
        rank_in_b,
    }
}

/// What a manifest runs the corpus through and where it leaves the
/// distribution.
struct Plan {
    chunk: String,
    chunk_max: usize,
    /// A fixed-rows forward for the steps; without one they are chunks of one row.
    step: Option<Forward>,
    logits: String,
    dtype: DType,
}

fn plan(v: &Verified) -> Result<Plan> {
    let m: &Manifest = v;
    let pr = Protocol::check(v).context("the manifest does not fit the serving protocol")?;
    let chunk = pr.chunk().ok_or_else(|| anyhow::anyhow!("no chunk program: the corpus cannot be fed"))?;
    let step = pr.forwards.iter().find(|f| matches!(f.rows, Rows::Const(_)) && !f.span).cloned();
    let n = m.programs[&chunk.name].calls.len();
    let mut logits: Vec<String> = access(m, &chunk.name, 0..n)
        .writes
        .into_iter()
        .filter(|b| b.rsplit('.').next().is_some_and(|l| l.starts_with("logits")))
        .collect();
    logits.sort_by_key(|b| b.len());
    let logits = logits
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("`{}` writes no `logits*` buffer: nothing to score", chunk.name))?;
    Ok(Plan {
        chunk: chunk.name.clone(),
        chunk_max: pr.rows.max as usize,
        step,
        logits: logits.clone(),
        dtype: m.buffers[&logits].dtype,
    })
}

/// Feed the corpus to a side and hand every scored position's logits row
/// to `f` (rank 0's), which says whether to go on.
fn drive<S: Side>(c: &mut S, t: &Trace, mut f: impl FnMut(usize, usize, Phase, &[f64]) -> Result<bool>) -> Result<()> {
    let v = c.manifest().clone();
    let m: &Manifest = &v;
    let p = plan(&v)?;
    let w = p.dtype.bytes() as usize;
    let read_row = |c: &S, e: &crate::Vars, last: bool| -> Result<Vec<f64>> {
        let cols = row_elems(m, &p.logits, e);
        let len = live_bytes(m, &p.logits, e);
        let rows = (len / (cols * w)).max(1);
        let bytes = c.read(0, &p.logits, len)?;
        let r = if last { rows - 1 } else { 0 };
        Ok(values::to_f64(p.dtype, &bytes[r * cols * w..(r + 1) * cols * w]))
    };
    for (i, toks) in t.prompts.iter().enumerate() {
        let s = t.schedule(i);
        c.zero_states()?;
        c.reset();
        let mut fed = 0;
        let mut e = None;
        while fed < s.ctx {
            let n = (s.ctx - fed).min(p.chunk_max);
            let vars = c.stage(&toks[fed..fed + n])?;
            c.run(&p.chunk, &vars, 0..c.calls(&p.chunk)?)?;
            c.advance(n as u64);
            fed += n;
            e = Some(vars);
        }
        let e = e.expect("ctx is at least one token");
        if !f(i, s.ctx - 1, Phase::Prefill, &read_row(c, &e, true)?)? {
            return Ok(());
        }
        for k in 0..s.steps {
            let tok = toks[s.ctx + k];
            let (prog, rows) = match &p.step {
                Some(f) => (
                    f.name.as_str(),
                    match f.rows {
                        Rows::Const(r) => r as usize,
                        Rows::Var => 1,
                    },
                ),
                None => (p.chunk.as_str(), 1),
            };
            let vars = c.stage(&vec![tok; rows])?;
            c.run(prog, &vars, 0..c.calls(prog)?)?;
            c.advance(1);
            if !f(i, s.ctx + k, Phase::Decode, &read_row(c, &vars, p.step.is_none())?)? {
                return Ok(());
            }
        }
    }
    Ok(())
}

/// Score the corpus on a side and add (or replace) its rows in the
/// trace under `producer`.
pub fn record<S: Side>(c: &mut S, t: &mut Trace, producer: &str, out: &mut dyn FnMut(&[String])) -> Result<()> {
    let t0 = Instant::now();
    let replaced = t.producers.contains_key(producer);
    let mut scores = Vec::with_capacity(t.positions());
    let prompts = t.prompts.clone();
    drive(c, t, |i, pos, _, row| {
        scores.push(score(i, pos, row, prompts[i][pos + 1]));
        Ok(true)
    })?;
    let n = scores.len();
    t.producers.insert(producer.to_string(), scores);
    out(&[row(
        "record",
        format!(
            "producer `{producer}`{} · {} prompts · {n} positions (top-{TOP})",
            if replaced { " replaced" } else { "" },
            prompts.len()
        ),
        Some(t0.elapsed().as_secs_f32()),
    )]);
    Ok(())
}

/// The widest disagreement between the producers on file, per
/// statistic: what a candidate may not exceed against any of them. The
/// KL statistics have a floor of [`KL_FLOOR`]: producers that agree to
/// the last bit still leave a candidate room for the f32 the file keeps
/// its log probabilities in.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Band {
    /// The largest margin a producer had where another one flipped its
    /// argmax: a flip above it is a confident token that moved.
    pub margin: f64,
    pub flip_rate: f64,
    pub kl_p50: f64,
    pub kl_p99: f64,
}

pub const KL_FLOOR: f64 = 1e-6;

fn quantile(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[((v.len() - 1) as f64 * q).round() as usize]
}

/// Over every ordered pair of producers, on the positions both scored.
pub fn band(t: &Trace) -> Option<Band> {
    let names: Vec<&String> = t.producers.keys().collect();
    if names.len() < 2 {
        return None;
    }
    let mut b = Band { kl_p50: KL_FLOOR, kl_p99: KL_FLOOR, ..Band::default() };
    for a in &names {
        let index: BTreeMap<(usize, usize), &Score> = t.producers[*a].iter().map(|s| ((s.prompt, s.pos), s)).collect();
        for c in names.iter().filter(|c| c != &a) {
            let cmps: Vec<RowCmp> =
                t.producers[*c].iter().filter_map(|s| index.get(&(s.prompt, s.pos)).map(|x| between(x, s))).collect();
            if cmps.is_empty() {
                continue;
            }
            let flips = cmps.iter().filter(|c| c.flip());
            b.margin = b.margin.max(flips.clone().map(|c| c.margin_a).fold(0.0, f64::max));
            b.flip_rate = b.flip_rate.max(flips.count() as f64 / cmps.len() as f64);
            let mut kl: Vec<f64> = cmps.iter().map(|c| c.kl).collect();
            b.kl_p50 = b.kl_p50.max(quantile(&mut kl, 0.5));
            b.kl_p99 = b.kl_p99.max(quantile(&mut kl, 0.99));
        }
    }
    Some(b)
}

/// One position of the candidate against one producer, as the archive keeps it.
#[derive(Clone, Debug, Serialize)]
pub struct Judged {
    pub producer: String,
    pub prompt: usize,
    pub pos: usize,
    pub phase: Phase,
    #[serde(flatten)]
    pub cmp: RowCmp,
    /// The candidate's log probability of the corpus token, next to the producer's.
    pub ref_logprob: f64,
    pub ref_logprob_a: f64,
}

/// A producer's rows of one phase, summed up.
#[derive(Clone, Debug, Serialize)]
pub struct PhaseStats {
    pub producer: String,
    pub phase: Phase,
    pub rows: usize,
    pub flips: usize,
    /// Flips at a margin above the band's (or, with one producer, beyond the KL limit).
    pub confident: usize,
    pub kl_p50: f64,
    pub kl_p99: f64,
    pub kl_max: f64,
    pub kl_at: String,
}

impl PhaseStats {
    fn of(producer: &str, phase: Phase, rows: &[&Judged], confident: impl Fn(&Judged) -> bool) -> PhaseStats {
        let mut kl: Vec<f64> = rows.iter().map(|j| j.cmp.kl).collect();
        let worst = rows.iter().max_by(|x, y| x.cmp.kl.total_cmp(&y.cmp.kl));
        PhaseStats {
            producer: producer.into(),
            phase,
            rows: rows.len(),
            flips: rows.iter().filter(|j| j.cmp.flip()).count(),
            confident: rows.iter().filter(|j| j.cmp.flip() && confident(j)).count(),
            kl_p50: quantile(&mut kl, 0.5),
            kl_p99: quantile(&mut kl, 0.99),
            kl_max: kl.last().copied().unwrap_or(0.0),
            kl_at: worst.map_or(String::new(), |j| format!("prompt {} pos {}", j.prompt, j.pos)),
        }
    }
    pub fn line(&self) -> String {
        let key = match self.phase {
            Phase::Prefill => "prefill",
            Phase::Decode => "decode",
        };
        let mut s =
            format!("{}: {} rows · {}/{} argmax agree", self.producer, self.rows, self.rows - self.flips, self.rows);
        if self.flips > 0 {
            s += &format!(" ({} confident)", self.confident);
        }
        s += &format!(
            " · KL p50 {:.1e} · p99 {:.1e} · max {:.1e} at {}",
            self.kl_p50, self.kl_p99, self.kl_max, self.kl_at
        );
        row(key, s, None)
    }
}

/// The reference section of the report, as `--json` prints it.
#[derive(Clone, Debug, Serialize)]
pub struct Summary {
    pub reference: String,
    pub candidate: String,
    pub producers: Vec<String>,
    pub prompts: usize,
    pub positions: usize,
    pub band: Option<Band>,
    pub limit_kl: f64,
    pub phases: Vec<PhaseStats>,
    pub flips: Vec<Judged>,
    pub elapsed_s: f32,
    pub verdict: Verdict,
}

#[derive(Debug, Serialize)]
pub struct Report {
    pub summary: Summary,
    pub detail: Vec<Judged>,
}

impl Report {
    pub fn code(&self) -> i32 {
        self.summary.verdict.code
    }
}

/// Judge a side against every producer on file. Stops at the first
/// position that fails: a prefill row that is wrong is a wrong model,
/// and the decode steps would only say so again.
pub fn judge<S: Side>(
    c: &mut S,
    t: &Trace,
    reference: &str,
    candidate: &str,
    limit_kl: f64,
    out: &mut dyn FnMut(&[String]),
) -> Result<Report> {
    let t0 = Instant::now();
    ensure!(!t.producers.is_empty(), "{reference} has no producer: nothing recorded to judge against");
    let producers: Vec<String> = t.producers.keys().cloned().collect();
    let band = band(t);
    out(&[row(
        "reference",
        format!(
            "{reference} · producer{} {} · {} prompts · {} positions ({} prefill + {} decode)",
            if producers.len() == 1 { "" } else { "s" },
            producers.join(", "),
            t.prompts.len(),
            t.positions(),
            t.prompts.len(),
            t.positions() - t.prompts.len()
        ),
        None,
    )]);
    out(&[row(
        "band",
        match &band {
            Some(b) => format!(
                "between producers · flips at margin ≤ {:.3} · flip rate {:.2}% · KL p50 {:.1e} · p99 {:.1e}",
                b.margin,
                b.flip_rate * 100.0,
                b.kl_p50,
                b.kl_p99
            ),
            None => format!("single producer · KL limit {limit_kl:.0e} (--logit-kl): a flip beyond it fails"),
        },
        None,
    )]);
    let index: BTreeMap<(String, usize, usize), &Score> =
        t.producers.iter().flat_map(|(p, ss)| ss.iter().map(move |s| ((p.clone(), s.prompt, s.pos), s))).collect();
    let confident = |j: &Judged| match &band {
        Some(b) => j.cmp.margin_a > b.margin,
        None => j.cmp.kl > limit_kl,
    };
    let mut rows: Vec<Judged> = Vec::new();
    let mut failed: Option<Judged> = None;
    drive(c, t, |i, pos, phase, logits| {
        let next = t.prompts[i][pos + 1];
        let mine = score(i, pos, logits, next);
        for p in &producers {
            let Some(a) = index.get(&(p.clone(), i, pos)) else { continue };
            let j = Judged {
                producer: p.clone(),
                prompt: i,
                pos,
                phase,
                cmp: against(a, logits),
                ref_logprob: mine.ref_logprob,
                ref_logprob_a: a.ref_logprob,
            };
            let fail = j.cmp.flip() && confident(&j);
            rows.push(j.clone());
            if fail {
                failed = Some(j);
                return Ok(false);
            }
        }
        Ok(true)
    })?;
    let mut phases = Vec::new();
    for p in &producers {
        for phase in [Phase::Prefill, Phase::Decode] {
            let rs: Vec<&Judged> = rows.iter().filter(|j| &j.producer == p && j.phase == phase).collect();
            if !rs.is_empty() {
                phases.push(PhaseStats::of(p, phase, &rs, confident));
            }
        }
    }
    out(&phases.iter().map(PhaseStats::line).collect::<Vec<_>>());
    let flips: Vec<Judged> = rows.iter().filter(|j| j.cmp.flip()).cloned().collect();
    out(&flips
        .iter()
        .take(crate::report::SHOW)
        .map(|j| {
            row(
                "flip",
                format!(
                    "{}prompt {} pos {} vs {}: {} → {} · margin {:.3} · KL {:.1e} · reference token is #{} · ref logprob {:.3} → {:.3}",
                    if confident(j) { "✗ " } else { "" },
                    j.prompt,
                    j.pos,
                    j.producer,
                    j.cmp.argmax_a,
                    j.cmp.argmax_b,
                    j.cmp.margin_a,
                    j.cmp.kl,
                    j.cmp.rank_in_b,
                    j.ref_logprob_a,
                    j.ref_logprob
                ),
                None,
            )
        })
        .collect::<Vec<_>>());
    let elapsed = t0.elapsed().as_secs_f32();
    let n = rows.len();
    let kl_max = rows.iter().map(|j| j.cmp.kl).fold(0.0, f64::max);
    let (code, summary) = match (&failed, &band) {
        (Some(j), _) => (
            1,
            format!(
                "confident argmax flip at prompt {} pos {} against {} ({} → {}, margin {:.3}, KL {:.1e}); stopped after {n} positions",
                j.prompt, j.pos, j.producer, j.cmp.argmax_a, j.cmp.argmax_b, j.cmp.margin_a, j.cmp.kl
            ),
        ),
        // the band is over every position a pair shares, so the candidate is
        // read over every position too; the phase lines are where to look
        (None, Some(b)) => {
            let over: Vec<String> = producers
                .iter()
                .filter_map(|p| {
                    let rs: Vec<&Judged> = rows.iter().filter(|j| &j.producer == p).collect();
                    let mut kl: Vec<f64> = rs.iter().map(|j| j.cmp.kl).collect();
                    let rate = rs.iter().filter(|j| j.cmp.flip()).count() as f64 / rs.len().max(1) as f64;
                    let (p50, p99) = (quantile(&mut kl, 0.5), quantile(&mut kl, 0.99));
                    let what = [
                        (rate > b.flip_rate, format!("flip rate {:.2}%", rate * 100.0)),
                        (p50 > b.kl_p50, format!("KL p50 {p50:.1e}")),
                        (p99 > b.kl_p99, format!("KL p99 {p99:.1e}")),
                    ]
                    .into_iter()
                    .filter(|(o, _)| *o)
                    .map(|(_, w)| w)
                    .collect::<Vec<_>>();
                    (!what.is_empty()).then(|| format!("against {p}: {}", what.join(", ")))
                })
                .collect();
            match over.is_empty() {
                true => (0, format!("within the band of {} producers on all {n} positions", producers.len())),
                false => (1, format!("beyond the band: {}", over.join("; "))),
            }
        }
        (None, None) => {
            let flips = flips.len();
            match kl_max <= limit_kl {
                true => (
                    0,
                    format!(
                        "within KL {limit_kl:.0e} of `{}` on all {n} positions{}",
                        producers[0],
                        if flips > 0 { format!(", {flips} flip(s) all ties") } else { String::new() }
                    ),
                ),
                false => (
                    2,
                    format!("KL up to {kl_max:.1e} (limit {limit_kl:.0e}) against `{}` with no confident flip on {n} positions", producers[0]),
                ),
            }
        }
    };
    let verdict = Verdict::new(code, summary, elapsed);
    let summary = Summary {
        reference: reference.into(),
        candidate: candidate.into(),
        producers,
        prompts: t.prompts.len(),
        positions: t.positions(),
        band,
        limit_kl,
        phases,
        flips,
        elapsed_s: elapsed,
        verdict,
    };
    Ok(Report { summary, detail: rows })
}
