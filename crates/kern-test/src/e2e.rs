//! e2e_logits: a candidate scored along a baseline's own tokens, judged by
//! how far its next-token distributions moved (`docs/test.md`).
//!
//! Three tables. A question set is chat conversations (`source`,
//! `messages`, `tools`). A recording is what one producer predicted for
//! them: per conversation the rendered `prompt`, the `output` that was fed
//! after it, and at every output position the top-[`TOP`] of the
//! distribution that predicted it. A baseline recording samples its own
//! output; a candidate's copies the baseline's, so two recordings line up
//! token for token and position for position.
//!
//! Feeding is one loop for both ([`Plan`]): rows longest output first
//! (sampling: in question order), a row admitted when the pool leases it
//! all its tokens at once, its prompt prefilled whole in chunks, then one
//! decode step over every admitted row, each fed its own next token, until
//! its output is fed. The loop never mixes prefill and decode in a call and
//! never pads a batch: it runs the programs the manifest declares, at the
//! batch sizes the rows leave it. The plan is the decisions, the shell runs
//! them: [`Plan::step`] hands out a [`Step`], the shell executes it and
//! answers what only the device knows (did the lease fit, which token was
//! drawn).
//!
//! Scoring is [`kl`]: KL(baseline ‖ candidate) on the baseline's
//! top-[`TOP`] with the rest as one bucket, against the candidate's full
//! row. The verdict is one number, the mean over every position
//! ([`Judgment`]), held to a [`Bar`] nobody has to name: the recording's
//! own noise. A baseline replayed along its own tokens under other batches
//! is not itself, because every batch rounds its own way; `record` measures
//! that at every position and the recording carries it. A candidate that
//! only rounds differently lands on the noise, one that adds error stands
//! above it. A mean does not say what it averaged, so the report reads the
//! positions one by one beside it: where the KL sits (its median and 99th
//! percentile, by depth, by decode step) and, for every top-1 the
//! candidate flipped, the [`Position::regret`], how much less the baseline
//! thought of the candidate's token than of its own. A tie that flips
//! costs nothing; a token the baseline was sure of does.
//!
//! Nothing here touches a device or a clock. A draw is a function of
//! `(seed, row, step)` and the logits alone, so batching can move a
//! recording only through the logits it computes.

use std::collections::{BTreeMap, VecDeque};
use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, ensure, Context, Result};
use arrow_array::builder::{Float32Builder, Float64Builder, Int32Builder, ListBuilder, StringBuilder};
use arrow_array::{Array, ArrayRef, Float32Array, Float64Array, Int32Array, ListArray, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;

pub use crate::compare::TOP;
use crate::report::{plural, row};

/// One conversation of a question set.
#[derive(Clone, Debug, PartialEq)]
pub struct Question {
    pub source: String,
    /// OpenAI chat messages, JSON.
    pub messages: String,
    /// OpenAI tool schemas, JSON.
    pub tools: Option<String>,
}

/// The most likely tokens of one distribution, most likely first, with
/// their log probabilities before any temperature.
#[derive(Clone, Debug, PartialEq)]
pub struct Top {
    pub ids: Vec<i32>,
    pub logprobs: Vec<f32>,
}

/// One conversation as a producer saw it: `top[k]` predicted `output[k]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub source: String,
    pub prompt: Vec<i32>,
    pub output: Vec<i32>,
    pub top: Vec<Top>,
}

/// One producer over a question set. `meta` is the file's key-value
/// metadata: who produced it and how it was sampled. `noise` is what a
/// baseline declares about itself: per row, the KL at every output position
/// when it is replayed along its own tokens under other batches.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Recording {
    pub meta: BTreeMap<String, String>,
    pub rows: Vec<Row>,
    pub noise: Option<Vec<Vec<f64>>>,
}

pub fn questions(path: &Path) -> Result<Vec<Question>> {
    let batches = batches(path, &[("source", DataType::Utf8), ("messages", DataType::Utf8)])?;
    let mut out = Vec::new();
    for b in &batches {
        let source = strings(b, "source");
        let messages = strings(b, "messages");
        let tools = b.column_by_name("tools").map(|c| c.as_any().downcast_ref::<StringArray>().cloned());
        let tools = match tools {
            Some(None) => bail!(
                "{}: column `tools` is {}, expected string",
                path.display(),
                b.schema().field_with_name("tools")?.data_type()
            ),
            Some(Some(t)) => Some(t),
            None => None,
        };
        for r in 0..b.num_rows() {
            out.push(Question {
                source: source.value(r).to_string(),
                messages: messages.value(r).to_string(),
                tools: tools.as_ref().filter(|t| !t.is_null(r)).map(|t| t.value(r).to_string()),
            });
        }
    }
    Ok(out)
}

impl Recording {
    pub fn read(path: &Path) -> Result<Recording> {
        let f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let schema = ParquetRecordBatchReaderBuilder::try_new(f)
            .with_context(|| format!("{} as parquet", path.display()))?
            .schema()
            .clone();
        let meta = schema.metadata().iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        let batches = batches(
            path,
            &[
                ("source", DataType::Utf8),
                ("prompt", list_of(DataType::Int32)),
                ("output", list_of(DataType::Int32)),
                ("top_ids", list_of(list_of(DataType::Int32))),
                ("top_logprobs", list_of(list_of(DataType::Float32))),
            ],
        )?;
        let mut rows = Vec::new();
        let mut noise = Vec::new();
        for b in &batches {
            let declared = match b.column_by_name("noise") {
                None => None,
                Some(c) if same(c.data_type(), &list_of(DataType::Float64)) => {
                    Some(c.as_any().downcast_ref::<ListArray>().expect("list").clone())
                }
                Some(c) => bail!("{}: column `noise` is {}, expected list<float64>", path.display(), c.data_type()),
            };
            let source = strings(b, "source");
            let list = |n: &str| {
                b.column_by_name(n).expect("checked").as_any().downcast_ref::<ListArray>().expect("list").clone()
            };
            let (prompt, output, ids, lps) = (list("prompt"), list("output"), list("top_ids"), list("top_logprobs"));
            for r in 0..b.num_rows() {
                let (ids, lps) = (nested(&ids, r), nested(&lps, r));
                let top: Vec<Top> =
                    (0..ids.len()).map(|k| Top { ids: i32s(&ids.value(k)), logprobs: f32s(&lps.value(k)) }).collect();
                let row = Row {
                    source: source.value(r).to_string(),
                    prompt: i32s(&prompt.value(r)),
                    output: i32s(&output.value(r)),
                    top,
                };
                ensure!(
                    row.top.len() == row.output.len() && row.top.iter().all(|t| t.ids.len() == t.logprobs.len()),
                    "{}: row `{}` has {} output tokens and {} distributions",
                    path.display(),
                    row.source,
                    row.output.len(),
                    row.top.len()
                );
                if let Some(d) = &declared {
                    let kls = d.value(r).as_any().downcast_ref::<Float64Array>().expect("f64").values().to_vec();
                    ensure!(
                        kls.len() == row.output.len(),
                        "{}: row `{}` has {} output tokens and {} noise values",
                        path.display(),
                        row.source,
                        row.output.len(),
                        kls.len()
                    );
                    noise.push(kls);
                }
                rows.push(row);
            }
        }
        Ok(Recording { meta, rows, noise: schema.field_with_name("noise").is_ok().then_some(noise) })
    }

    pub fn write(&self, path: &Path) -> Result<()> {
        let mut source = StringBuilder::new();
        let mut prompt = ListBuilder::new(Int32Builder::new());
        let mut output = ListBuilder::new(Int32Builder::new());
        let mut ids = ListBuilder::new(ListBuilder::new(Int32Builder::new()));
        let mut lps = ListBuilder::new(ListBuilder::new(Float32Builder::new()));
        for r in &self.rows {
            source.append_value(&r.source);
            prompt.values().append_slice(&r.prompt);
            prompt.append(true);
            output.values().append_slice(&r.output);
            output.append(true);
            for t in &r.top {
                ids.values().values().append_slice(&t.ids);
                ids.values().append(true);
                lps.values().values().append_slice(&t.logprobs);
                lps.values().append(true);
            }
            ids.append(true);
            lps.append(true);
        }
        let mut cols: Vec<ArrayRef> = vec![
            Arc::new(source.finish()),
            Arc::new(prompt.finish()),
            Arc::new(output.finish()),
            Arc::new(ids.finish()),
            Arc::new(lps.finish()),
        ];
        if let Some(noise) = &self.noise {
            let mut kls = ListBuilder::new(Float64Builder::new());
            for row in noise {
                kls.values().append_slice(row);
                kls.append(true);
            }
            cols.push(Arc::new(kls.finish()));
        }
        let fields: Vec<Field> = cols
            .iter()
            .zip(["source", "prompt", "output", "top_ids", "top_logprobs", "noise"])
            .map(|(c, n)| Field::new(n, c.data_type().clone(), false))
            .collect();
        let schema = Arc::new(Schema::new(fields).with_metadata(self.meta.clone()));
        let batch = RecordBatch::try_new(schema.clone(), cols)?;
        let f = std::fs::File::create(path).with_context(|| format!("creating {}", path.display()))?;
        let mut w = ArrowWriter::try_new(f, schema, None)?;
        w.write(&batch)?;
        w.close()?;
        Ok(())
    }
}

/// Every batch of the file, after checking it has `cols` with those types.
fn batches(path: &Path, cols: &[(&str, DataType)]) -> Result<Vec<RecordBatch>> {
    let f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let b = ParquetRecordBatchReaderBuilder::try_new(f).with_context(|| format!("{} as parquet", path.display()))?;
    for (name, dt) in cols {
        let f =
            b.schema().field_with_name(name).map_err(|_| anyhow::anyhow!("{}: no column `{name}`", path.display()))?;
        ensure!(same(f.data_type(), dt), "{}: column `{name}` is {}, expected {dt}", path.display(), f.data_type());
    }
    b.build()?.map(|r| r.map_err(Into::into)).collect()
}

fn strings(b: &RecordBatch, name: &str) -> StringArray {
    b.column_by_name(name).expect("checked").as_any().downcast_ref::<StringArray>().expect("utf8").clone()
}

fn nested(l: &ListArray, r: usize) -> ListArray {
    l.value(r).as_any().downcast_ref::<ListArray>().expect("list of lists").clone()
}

fn i32s(a: &ArrayRef) -> Vec<i32> {
    a.as_any().downcast_ref::<Int32Array>().expect("i32").values().to_vec()
}

fn f32s(a: &ArrayRef) -> Vec<f32> {
    a.as_any().downcast_ref::<Float32Array>().expect("f32").values().to_vec()
}

/// A list column's child may be called `item` or `element`, by writer.
fn same(a: &DataType, b: &DataType) -> bool {
    match (a, b) {
        (DataType::List(x), DataType::List(y)) => same(x.data_type(), y.data_type()),
        _ => a == b,
    }
}

fn list_of(dt: DataType) -> DataType {
    DataType::List(Arc::new(Field::new("item", dt, true)))
}

fn lse(row: &[f32]) -> f64 {
    let max = row.iter().map(|&x| x as f64).fold(f64::NEG_INFINITY, f64::max);
    max + row.iter().map(|&x| (x as f64 - max).exp()).sum::<f64>().ln()
}

/// The highest logit, the lowest id among equals.
pub fn argmax(row: &[f32]) -> usize {
    (0..row.len()).fold(0, |best, i| if row[i] > row[best] { i } else { best })
}

/// The top-[`TOP`] of a logits row, as log probabilities.
pub fn top(row: &[f32]) -> Top {
    let lse = lse(row);
    let by = |i: &usize, j: &usize| row[*j].total_cmp(&row[*i]).then(i.cmp(j));
    let mut idx: Vec<usize> = (0..row.len()).collect();
    let k = TOP.min(idx.len());
    if k < idx.len() {
        idx.select_nth_unstable_by(k, by);
        idx.truncate(k);
    }
    idx.sort_by(by);
    Top {
        ids: idx.iter().map(|&i| i as i32).collect(),
        logprobs: idx.iter().map(|&i| (row[i] as f64 - lse) as f32).collect(),
    }
}

/// KL(base ‖ row) over `base`'s kept tokens, everything else one bucket on
/// both sides: how much of the mass `base` puts where it can be seen
/// moved. A row that gives a kept token no mass at all is infinitely far.
pub fn kl(base: &Top, row: &[f32]) -> f64 {
    let lse = lse(row);
    let (mut kl, mut p_kept, mut q_kept) = (0.0f64, 0.0f64, 0.0f64);
    for (&id, &lp) in base.ids.iter().zip(&base.logprobs) {
        let (lp, lq) = (lp as f64, row[id as usize] as f64 - lse);
        let p = lp.exp();
        kl += p * (lp - lq);
        p_kept += p;
        q_kept += lq.exp();
    }
    let (p_rest, q_rest) = ((1.0 - p_kept).max(1e-12), (1.0 - q_kept).max(1e-12));
    let kl = kl + p_rest * (p_rest / q_rest).ln();
    if kl.is_nan() {
        f64::INFINITY
    } else {
        kl.max(0.0)
    }
}

/// splitmix64: one well-mixed word from another.
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// How a baseline draws its tokens: the checkpoint's `generation_config`
/// sampler. `top_k` 0 keeps every token; `temperature` 0 is the argmax.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    pub temperature: f64,
    pub top_k: usize,
    pub top_p: f64,
}

impl Sampling {
    pub fn new(temperature: f64, top_k: usize, top_p: f64) -> Result<Sampling> {
        ensure!(temperature >= 0.0, "temperature {temperature} is negative");
        ensure!(top_p > 0.0 && top_p <= 1.0, "top_p {top_p} is outside (0, 1]");
        Ok(Sampling { temperature, top_k, top_p })
    }
}

/// The token drawn at output position `step` of `row` from a logits row:
/// temperature, then the `top_k` most likely, then the smallest prefix of
/// those holding `top_p` of their mass. The draw is a function of `(seed,
/// row, step)` and the logits alone.
pub fn sample(logits: &[f32], s: Sampling, seed: u64, row: usize, step: usize) -> i64 {
    if s.temperature == 0.0 {
        return argmax(logits) as i64;
    }
    let by = |i: &usize, j: &usize| logits[*j].total_cmp(&logits[*i]).then(i.cmp(j));
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    let k = if s.top_k == 0 { idx.len() } else { s.top_k.min(idx.len()) };
    if k < idx.len() {
        idx.select_nth_unstable_by(k, by);
        idx.truncate(k);
    }
    idx.sort_by(by);
    let max = logits[idx[0]] as f64;
    let w: Vec<f64> = idx.iter().map(|&i| ((logits[i] as f64 - max) / s.temperature).exp()).collect();
    let total: f64 = w.iter().sum();
    let kept = w
        .iter()
        .scan(0.0, |acc, &x| {
            let before = *acc;
            *acc += x;
            Some(before)
        })
        .take_while(|&before| before < s.top_p * total)
        .count();
    let w = &w[..kept];
    let u = (mix(mix(mix(seed) ^ row as u64) ^ step as u64) >> 11) as f64 / (1u64 << 53) as f64;
    let mut left = u * w.iter().sum::<f64>();
    for (&i, &x) in idx.iter().zip(w) {
        if left < x {
            return i as i64;
        }
        left -= x;
    }
    idx[kept - 1] as i64
}

/// What the shell does next; [`Plan::step`] hands them out in order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Step {
    /// Lease `tokens` slots for `row`, then answer [`Plan::leased`].
    Lease { row: usize, tokens: usize },
    /// Run the chunk program over `ids`, `row`'s tokens from position `at`.
    /// When `scores`, its last row's distribution is the row's next one:
    /// answer [`Plan::scored`].
    Prefill { row: usize, at: usize, ids: Vec<i64>, scores: bool },
    /// One decode call: per sequence its row, position and token, in batch
    /// order; every one of them is scored, answered in this order.
    Decode(Vec<(usize, usize, i64)>),
    /// `row` is done: drop its lease.
    Release(usize),
}

#[derive(Clone, Debug)]
struct Seq {
    /// The prompt, then the output known so far.
    tokens: Vec<i64>,
    prompt: usize,
    /// Distributions it is owed: its output's length, or the sampling cap.
    goal: usize,
    scored: usize,
    /// Tokens already in its state.
    fed: usize,
}

/// The feeding loop's decisions over one set of rows (see the module doc).
#[derive(Clone, Debug)]
pub struct Plan {
    seqs: Vec<Seq>,
    waiting: VecDeque<usize>,
    running: Vec<usize>,
    /// Sequences one decode call takes at most.
    groups: usize,
    /// Rows one prefill call takes at most.
    chunk: usize,
    /// The chunk program hands back a distribution for its last row;
    /// without one the last prompt token goes through decode.
    chunk_scores: bool,
    /// Sampling: tokens that end a row's output. Empty when forced.
    stop: Vec<i64>,
    queue: VecDeque<Step>,
    /// A step waiting for its answer: a lease, or the rows of a call not yet scored.
    lease: Option<usize>,
    unscored: VecDeque<usize>,
    admitting: bool,
}

impl Plan {
    /// Rows fed each its baseline's output, longest output first.
    pub fn forced(rows: &[(Vec<i64>, Vec<i64>)], groups: usize, chunk: usize, chunk_scores: bool) -> Result<Plan> {
        for (i, (p, o)) in rows.iter().enumerate() {
            ensure!(
                !p.is_empty() && !o.is_empty(),
                "row {i}: a prompt of {} and an output of {} tokens: nothing to score",
                p.len(),
                o.len()
            );
        }
        let seqs = rows
            .iter()
            .map(|(p, o)| Seq { tokens: [p.as_slice(), o].concat(), prompt: p.len(), goal: o.len(), scored: 0, fed: 0 })
            .collect();
        let mut order: Vec<usize> = (0..rows.len()).collect();
        order.sort_by_key(|&i| (std::cmp::Reverse(rows[i].1.len()), i));
        Plan::new(seqs, order, groups, chunk, chunk_scores, Vec::new())
    }

    /// Rows that draw their own output, at most `max_new` tokens ending at a
    /// `stop` token, in question order.
    pub fn sampled(
        prompts: &[Vec<i64>],
        max_new: usize,
        stop: &[i64],
        groups: usize,
        chunk: usize,
        chunk_scores: bool,
    ) -> Result<Plan> {
        ensure!(max_new > 0, "--max-new 0 samples nothing");
        for (i, p) in prompts.iter().enumerate() {
            ensure!(!p.is_empty(), "row {i}: an empty prompt");
        }
        let seqs = prompts
            .iter()
            .map(|p| Seq { tokens: p.clone(), prompt: p.len(), goal: max_new, scored: 0, fed: 0 })
            .collect();
        Plan::new(seqs, (0..prompts.len()).collect(), groups, chunk, chunk_scores, stop.to_vec())
    }

    fn new(
        seqs: Vec<Seq>,
        order: Vec<usize>,
        groups: usize,
        chunk: usize,
        chunk_scores: bool,
        stop: Vec<i64>,
    ) -> Result<Plan> {
        ensure!(groups > 0 && chunk > 0, "a decode call of {groups} sequences, a prefill call of {chunk} rows");
        Ok(Plan {
            seqs,
            waiting: order.into(),
            running: Vec::new(),
            groups,
            chunk,
            chunk_scores,
            stop,
            queue: VecDeque::new(),
            lease: None,
            unscored: VecDeque::new(),
            admitting: true,
        })
    }

    /// The next step, `None` when every row is done. Panics if
    /// the last step still waits for its answer: the shell answers every
    /// lease and every scored distribution before asking again.
    pub fn step(&mut self) -> Option<Step> {
        assert!(self.lease.is_none() && self.unscored.is_empty(), "the last step has not been answered");
        if let Some(s) = self.queue.pop_front() {
            if let Step::Prefill { row, scores: true, .. } = s {
                self.unscored.push_back(row);
            }
            return Some(s);
        }
        if self.admitting && self.running.len() < self.groups {
            if let Some(&row) = self.waiting.front() {
                let s = &self.seqs[row];
                self.lease = Some(row);
                // the last token fed is the one before the last scored
                return Some(Step::Lease { row, tokens: s.prompt + s.goal - 1 });
            }
        }
        self.admitting = true;
        let batch: Vec<(usize, usize, i64)> =
            self.running.iter().map(|&r| (r, self.seqs[r].fed, self.seqs[r].tokens[self.seqs[r].fed])).collect();
        if batch.is_empty() {
            return None;
        }
        for &(r, ..) in &batch {
            self.seqs[r].fed += 1;
            self.unscored.push_back(r);
        }
        Some(Step::Decode(batch))
    }

    /// The answer to a [`Step::Lease`]. A row that does not fit waits for
    /// the running ones to make room; one that does not fit an empty pool
    /// never will, and the shell says so instead of answering.
    pub fn leased(&mut self, ok: bool) {
        let row = self.lease.take().expect("a lease was asked for");
        if !ok {
            assert!(!self.running.is_empty(), "row {row} does not fit an empty pool");
            self.admitting = false;
            return;
        }
        self.waiting.pop_front();
        let (chunk_scores, chunk) = (self.chunk_scores, self.chunk);
        let s = &mut self.seqs[row];
        let pre = s.prompt - usize::from(!chunk_scores);
        let starts: Vec<usize> = (0..pre).step_by(chunk).collect();
        for &at in &starts {
            let end = (at + chunk).min(pre);
            self.queue.push_back(Step::Prefill {
                row,
                at,
                ids: s.tokens[at..end].to_vec(),
                scores: chunk_scores && end == pre,
            });
        }
        s.fed = pre;
        self.running.push(row);
    }

    /// The answer for the next unscored distribution: when sampling, the
    /// token drawn from it. A row owed nothing more is released.
    pub fn scored(&mut self, drawn: Option<i64>) {
        let row = self.unscored.pop_front().expect("a distribution was owed");
        let s = &mut self.seqs[row];
        s.scored += 1;
        s.tokens.extend(drawn);
        if drawn.is_some_and(|t| self.stop.contains(&t)) || s.scored == s.goal {
            self.running.retain(|&r| r != row);
            self.queue.push_back(Step::Release(row));
        }
    }

    /// Distributions scored so far for `row`: the output position the next one predicts.
    pub fn position(&self, row: usize) -> usize {
        self.seqs[row].scored
    }

    /// A sampled row's output so far.
    pub fn output(&self, row: usize) -> &[i64] {
        let s = &self.seqs[row];
        &s.tokens[s.prompt..]
    }
}

/// One scored position of a candidate against its baseline.
#[derive(Clone, Debug, PartialEq)]
pub struct Position {
    pub row: usize,
    /// Output position: 0 predicted from the prompt, by prefill when the
    /// chunk program writes logits and by decode otherwise.
    pub step: usize,
    /// The row's prompt length.
    pub depth: usize,
    pub kl: f64,
    /// Baseline top-1 minus top-2, in nats.
    pub margin: f64,
    /// The baseline's log probability of its own top-1 minus that of the
    /// candidate's, in nats: 0 where they agree, and a lower bound where
    /// the candidate's token is outside the baseline's kept ones.
    pub regret: f64,
    /// What the recording declares its own producer scores here under
    /// other batches; 0 when it declares nothing.
    pub noise: f64,
    pub base: i64,
    pub cand: i64,
}

impl Position {
    pub fn of(row: usize, step: usize, depth: usize, base: &Top, noise: f64, logits: &[f32]) -> Position {
        let cand = argmax(logits) as i64;
        let first = base.logprobs.first().copied().unwrap_or(0.0) as f64;
        let seen = base.ids.iter().position(|&id| id as i64 == cand).or(base.ids.len().checked_sub(1));
        Position {
            row,
            step,
            depth,
            kl: kl(base, logits),
            margin: first - base.logprobs.get(1).copied().unwrap_or(f32::NEG_INFINITY) as f64,
            regret: seen.and_then(|k| base.logprobs.get(k)).map_or(0.0, |&l| first - l as f64),
            noise,
            base: base.ids.first().copied().unwrap_or(-1) as i64,
            cand,
        }
    }
    pub fn flip(&self) -> bool {
        self.base != self.cand
    }
}

/// Prompt lengths the depth breakdown starts a bucket at.
const DEPTHS: [usize; 4] = [0, 8 << 10, 32 << 10, 128 << 10];
/// Output positions the step breakdown starts a bucket at.
const STEPS: [usize; 4] = [0, 1, 64, 512];
/// Flips the report names.
const WORST: usize = 8;
/// Regrets, in nats, the report counts the flips at or above.
const REGRETS: [f64; 3] = [0.5, 1.0, 2.0];
/// Standard errors a candidate's mean may stand above the recording's noise.
const SIGMAS: f64 = 3.0;

/// What a candidate's mean KL is held to.
#[derive(Clone, Debug, PartialEq)]
pub enum Bar {
    /// A mean the caller named, in nats.
    Nats(f64),
    /// The recording's noise: its mean over the scored positions, how far
    /// above it the candidate's mean stands, and the standard error of that
    /// difference. The error is taken over rows, since the positions of one
    /// conversation move together and conversations do not; a single row
    /// has none.
    Noise { mean_kl: f64, excess: f64, se: f64 },
}

/// Mean KL over the positions of one bucket.
#[derive(Clone, Debug, PartialEq)]
pub struct Bucket {
    pub from: usize,
    pub positions: usize,
    pub mean_kl: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Judgment {
    pub positions: usize,
    pub mean_kl: f64,
    pub bar: Bar,
    /// The KL half the positions stay under, and the KL 99 in 100 do.
    pub median_kl: f64,
    pub p99_kl: f64,
    pub depth: Vec<Bucket>,
    pub step: Vec<Bucket>,
    pub flips: usize,
    /// A regret, and the flips at or above it.
    pub regrets: Vec<(f64, usize)>,
    /// The flips with the most regret, most first.
    pub worst: Vec<Position>,
}

fn buckets(ps: &[Position], starts: &[usize], key: impl Fn(&Position) -> usize) -> Vec<Bucket> {
    starts
        .iter()
        .enumerate()
        .map(|(i, &from)| {
            let end = starts.get(i + 1).copied().unwrap_or(usize::MAX);
            let kls: Vec<f64> = ps.iter().filter(|p| (from..end).contains(&key(p))).map(|p| p.kl).collect();
            Bucket { from, positions: kls.len(), mean_kl: mean(&kls) }
        })
        .collect()
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

/// The value `q` of the way through `sorted`.
fn quantile(sorted: &[f64], q: f64) -> f64 {
    sorted.get((sorted.len().saturating_sub(1) as f64 * q).round() as usize).copied().unwrap_or(0.0)
}

/// The candidate's KL beside the noise the positions carry, row by row.
fn beside(ps: &[Position]) -> Bar {
    let mut rows: BTreeMap<usize, (f64, f64)> = BTreeMap::new();
    for p in ps {
        let (over, n) = rows.entry(p.row).or_default();
        *over += p.kl - p.noise;
        *n += 1.0;
    }
    let n = ps.len().max(1) as f64;
    let excess = rows.values().map(|r| r.0).sum::<f64>() / n;
    let k = rows.len() as f64;
    let spread: f64 = rows.values().map(|&(over, m)| (over - m * excess).powi(2)).sum();
    let se = if rows.len() > 1 { (k / (k - 1.0) * spread).sqrt() / n } else { 0.0 };
    Bar::Noise { mean_kl: mean(&ps.iter().map(|p| p.noise).collect::<Vec<_>>()), excess, se }
}

impl Judgment {
    /// Against a mean the caller names.
    pub fn of(ps: &[Position], limit: f64) -> Judgment {
        Judgment::read(ps, Bar::Nats(limit))
    }

    /// Against the noise the recording declared at the same positions.
    pub fn beside(ps: &[Position]) -> Judgment {
        Judgment::read(ps, beside(ps))
    }

    fn read(ps: &[Position], bar: Bar) -> Judgment {
        let mut flips: Vec<Position> = ps.iter().filter(|p| p.flip()).cloned().collect();
        flips.sort_by(|a, b| {
            b.regret.total_cmp(&a.regret).then(b.kl.total_cmp(&a.kl)).then((a.row, a.step).cmp(&(b.row, b.step)))
        });
        let mut kls: Vec<f64> = ps.iter().map(|p| p.kl).collect();
        let mean_kl = mean(&kls);
        kls.sort_by(f64::total_cmp);
        Judgment {
            positions: ps.len(),
            mean_kl,
            bar,
            median_kl: quantile(&kls, 0.5),
            p99_kl: quantile(&kls, 0.99),
            depth: buckets(ps, &DEPTHS, |p| p.depth),
            step: buckets(ps, &STEPS, |p| p.step),
            flips: flips.len(),
            regrets: REGRETS.iter().map(|&r| (r, flips.iter().filter(|p| p.regret >= r).count())).collect(),
            worst: flips.into_iter().take(WORST).collect(),
        }
    }

    pub fn pass(&self) -> bool {
        self.positions > 0
            && match self.bar {
                Bar::Nats(limit) => self.mean_kl <= limit,
                Bar::Noise { excess, se, .. } => excess <= SIGMAS * se,
            }
    }

    /// The breakdown, a line per fact; `sources` names a row.
    pub fn lines(&self, sources: &[String]) -> Vec<String> {
        let cells = |bs: &[Bucket], name: &dyn Fn(usize, Option<usize>) -> String| -> String {
            bs.iter()
                .enumerate()
                .filter(|(_, b)| b.positions > 0)
                .map(|(i, b)| {
                    format!("{} {:.2e} ({})", name(b.from, bs.get(i + 1).map(|n| n.from)), b.mean_kl, b.positions)
                })
                .collect::<Vec<_>>()
                .join(" · ")
        };
        let depth = |from: usize, to: Option<usize>| match to {
            Some(t) => format!("{}–{}k", from >> 10, t >> 10),
            None => format!("≥{}k", from >> 10),
        };
        let step = |from: usize, to: Option<usize>| match to {
            Some(t) if t == from + 1 => from.to_string(),
            Some(t) => format!("{from}–{}", t - 1),
            None => format!("≥{from}"),
        };
        let mut v = vec![
            row("depth", cells(&self.depth, &depth), None),
            row("step", cells(&self.step, &step), None),
            row("kl", format!("median {:.2e} · p99 {:.2e}", self.median_kl, self.p99_kl), None),
        ];
        let mut f = plural(self.flips, "flip");
        if let Some(most) = self.worst.first() {
            let at: Vec<String> = self.regrets.iter().map(|(r, n)| format!("{n} ≥ {r}")).collect();
            f += &format!(" · by regret in nats: {} · most {:.3} · most regret first:", at.join(" · "), most.regret);
        }
        v.push(row("flips", f, None));
        v.extend(self.worst.iter().map(|p| {
            row(
                "flip",
                format!(
                    "{} step {} (depth {}) · {} → {} · margin {:.3} · regret {:.3} · KL {:.3}",
                    sources.get(p.row).map_or("?", |s| s.as_str()),
                    p.step,
                    p.depth,
                    p.base,
                    p.cand,
                    p.margin,
                    p.regret,
                    p.kl
                ),
                None,
            )
        }));
        v
    }

    pub fn summary(&self) -> String {
        let cmp = if self.pass() { "≤" } else { ">" };
        let bar = match self.bar {
            Bar::Nats(limit) => format!("{limit:.2e}"),
            Bar::Noise { mean_kl, se, .. } => {
                format!("noise {mean_kl:.2e} + {SIGMAS} × {se:.2e} ({:.2} × the noise)", self.mean_kl / mean_kl)
            }
        };
        format!("mean KL {:.2e} {cmp} {bar} over {}", self.mean_kl, plural(self.positions, "scored position"))
    }
}
