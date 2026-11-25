//! `kern run`: end-to-end bs=1 greedy decode over a kern manifest.
//!
//! The prompt goes through the manifest's chunk program; every step after
//! that is one call of the program taking one sequence of `--rows` rows,
//! read back as the tokens it hands the sequence. A plain decode step and
//! a speculative round differ only in the rows the manifest declares for
//! them, so this loop does not know which one it is running.
//!
//! Logging goes to stderr via `tracing` (filter with `RUST_LOG`, default
//! `info`); stdout carries only the generated text.

use std::collections::BTreeMap;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{ensure, Context, Result};
use clap::Args;

use crate::config::{Config, Target};
use crate::{Caller, Given, Inputs, Vars};
use kern_manifest::protocol::{Forward, Rows, Sampling};
use kern_manifest::types::BufferKind;
use kern_manifest::Verified;
use kern_runtime::{Capacity, Runtime};
use tracing::info;

/// Flags of `kern run`; anything not given comes from the target in
/// kern.toml, then from the defaults.
#[derive(Args, Debug, Clone)]
pub struct RunOpts {
    /// Manifest JSON (must pass verification)
    #[arg(long)]
    manifest: Option<PathBuf>,

    /// Directory of cubins; steps resolve by their pinned sha256, so one dir
    /// holds every version (file names are labels)
    #[arg(long)]
    kernels: Option<PathBuf>,

    /// Checkpoint directories or .safetensors files, tensors bound by name
    /// across all of them
    #[arg(long)]
    weights: Vec<String>,

    /// HF tokenizer.json
    #[arg(long)]
    tokenizer: Option<PathBuf>,

    /// Raw (template-free) prompt
    #[arg(long)]
    prompt: Option<String>,

    /// The prompt as token ids, bypassing the tokenizer (a served request's
    /// ids, replayed as they were)
    #[arg(long, value_delimiter = ',', conflicts_with = "prompt")]
    prompt_ids: Vec<i64>,

    /// Max new tokens to generate
    #[arg(long)]
    steps: Option<usize>,

    /// CUDA device ordinal
    #[arg(long)]
    gpu: Option<usize>,

    /// State capacity in tokens (KV pages etc.); rounded down to the
    /// manifest's page unit. Default: what one sequence can reach (the
    /// manifest's page-table row)
    #[arg(long)]
    capacity: Option<u64>,

    /// Prefill chunk in tokens: the manifest's `tokens` bound unless a
    /// smaller one is asked for
    #[arg(long)]
    chunk: Option<u64>,

    /// Debug: launch every program eagerly, ignoring the manifest's `graph`
    #[arg(long)]
    eager: bool,

    /// Rows per sequence of a decode step: a shape some program of the
    /// manifest declares (1 for a plain step, its block for a speculative
    /// round). Default: the widest declared
    #[arg(long)]
    rows: Option<u64>,

    /// Extra token ids that end generation (comma-separated); the eos ids
    /// the checkpoint declares in generation_config.json always apply
    #[arg(long, value_delimiter = ',')]
    stop_tokens: Vec<i64>,

    /// Sampling temperature; 0 takes the most likely token every step
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,
    /// Top-k of a sampled step; under 1 keeps every token
    #[arg(long, default_value_t = 0)]
    top_k: i32,
    /// Top-p of a sampled step; 1 keeps every token
    #[arg(long, default_value_t = 1.0)]
    top_p: f32,
    /// Min-p of a sampled step; 0 keeps every token
    #[arg(long, default_value_t = 0.0)]
    min_p: f32,
    /// The draws' seed: the same seed and prompt replay the same tokens
    #[arg(long, default_value_t = 0)]
    seed: i64,

    /// Debug: dump activations of the first prefill chunk and `--probe-steps`
    /// decode steps into this directory, then exit: after every call whose
    /// label matches `--probe-labels`, the buffer it writes (live rows),
    /// plus the logits the step's tokens are taken from and the tokens.
    /// Programs run call-range by call-range so nothing executes twice.
    #[arg(long)]
    probe_dir: Option<PathBuf>,
    /// Call labels `--probe-dir` dumps after: comma-separated, a label
    /// matches one it equals or ends with (the file is named by the label
    /// minus its last `.part`)
    #[arg(long, default_value = "embed,.down_proj")]
    probe_labels: String,
    /// Decode steps `--probe-dir` dumps
    #[arg(long, default_value_t = 2)]
    probe_steps: usize,
}

/// Resolved options: the shared inputs, plus what only `kern run` has an
/// opinion about.
struct Opts {
    inputs: Inputs,
    tokenizer: PathBuf,
    prompt: String,
    prompt_ids: Vec<i64>,
    steps: usize,
    capacity: Option<u64>,
    chunk: Option<u64>,
    eager: bool,
    rows: Option<u64>,
    sampling: Sampling,
    probe_dir: Option<PathBuf>,
    probe_labels: String,
    probe_steps: usize,
}

impl RunOpts {
    fn resolve(self, cfg: Option<&Config>, t: Option<&Target>) -> Result<Opts> {
        let given = Given {
            manifest: self.manifest,
            kernels: self.kernels,
            weights: self.weights,
            tokenizer: self.tokenizer,
            stop_tokens: self.stop_tokens,
            gpu: self.gpu,
            ..Default::default()
        };
        let inputs = Inputs::resolve(given, cfg, t)?;
        ensure!(
            !inputs.stop_tokens.is_empty(),
            "no stop tokens: no eos_token_id in the weights dir(s)' generation_config.json / config.json and no --stop-tokens"
        );
        Ok(Opts {
            tokenizer: inputs.tokenizer()?.to_path_buf(),
            prompt: self
                .prompt
                .or_else(|| cfg.and_then(|c| c.run.prompt.clone()))
                .unwrap_or_else(|| "The capital of France is".into()),
            prompt_ids: self.prompt_ids,
            steps: self.steps.or_else(|| cfg.and_then(|c| c.run.steps)).unwrap_or(32),
            capacity: self.capacity.or_else(|| cfg.and_then(|c| c.capacity)),
            chunk: self.chunk.or_else(|| cfg.and_then(|c| c.run.chunk)),
            eager: self.eager,
            rows: self.rows,
            sampling: Sampling {
                temperature: self.temperature,
                top_k: self.top_k,
                top_p: self.top_p,
                min_p: self.min_p,
                seed: self.seed,
            },
            probe_dir: self.probe_dir,
            probe_labels: self.probe_labels,
            probe_steps: self.probe_steps,
            inputs,
        })
    }
}

/// `kern run`.
pub fn run(o: RunOpts, cfg: Option<&Config>, target: Option<&Target>) -> Result<()> {
    let mut ready = prepare(o.resolve(cfg, target)?, |_, _| Ok(()))?;
    if let Some(dir) = &ready.probe_dir {
        return crate::probe::probe(
            &mut ready.caller,
            &ready.prompt_ids,
            dir,
            &ready.probe_labels,
            ready.chunk,
            &ready.step,
            ready.probe_steps,
        );
    }
    ready.drive(|rt, f, vars, _, _| {
        rt.issue(&f.name, vars)?;
        rt.synchronize()?;
        Ok(ControlFlow::Continue(()))
    })
}

fn human(bytes: u64) -> String {
    match bytes {
        b if b >= 1 << 30 => format!("{:.1} GiB", b as f64 / (1u64 << 30) as f64),
        b if b >= 1 << 20 => format!("{:.1} MiB", b as f64 / (1u64 << 20) as f64),
        b if b >= 1 << 10 => format!("{:.1} KiB", b as f64 / 1024.0),
        b => format!("{b} B"),
    }
}

fn ellipsize(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(n - 1).collect::<String>())
    }
}

/// What was loaded, before anything runs: the sizes the manifest asked
/// for and the module every op resolved to. A run that is wrong in a
/// boring way (the wrong kernels dir, a state far bigger than expected)
/// is visible here rather than in the output.
fn banner(rt: &Runtime, manifest: &Path, kernels: Option<&Path>, load: Duration) {
    let m = &rt.manifest;
    info!("manifest `{}` (schema v{}, {}): verified", m.model, m.schema_version, manifest.display());
    for (name, v) in &m.vars {
        info!("  var      {name} ∈ [{}, {}] (caller-provided per call)", kern_manifest::types::Var::MIN, v.max);
    }
    for (name, st, alloc) in rt.state_sizes() {
        if st.bytes_per_token > 0 {
            info!(
                "  state    {name}: opaque, {} B/token × capacity {} = {}",
                st.bytes_per_token,
                rt.capacity(),
                human(alloc)
            );
        } else if st.is_per_seq() {
            info!(
                "  state    {name}: opaque, {} per sequence × {} slots = {}",
                human(st.bytes_per_seq),
                rt.seq_slots(),
                human(alloc)
            );
        } else {
            info!("  state    {name}: opaque, fixed {}", human(alloc));
        }
    }
    let mut by_kind: BTreeMap<String, (usize, u64)> = BTreeMap::new();
    for (_, kind, bytes) in rt.buffer_sizes() {
        let e = by_kind.entry(kind.to_string()).or_default();
        e.0 += 1;
        e.1 += bytes;
    }
    let kinds = ["weight", "workspace", "carry", "input", "output"]
        .iter()
        .filter_map(|c| by_kind.get(*c).map(|(n, b)| format!("{c} {n} ({})", human(*b))))
        .collect::<Vec<_>>()
        .join(" | ");
    info!("  buffers  {kinds}");
    for (name, p) in &m.programs {
        let shape = match &p.batch {
            Some(b) => format!(", {} × {:?} per call", b.groups, b.rows),
            None if p.once => ", once after load".into(),
            None if p.derive => ", inside weight loading".into(),
            None => String::new(),
        };
        info!("  program  `{name}`: {} calls{shape}", p.calls.len());
    }
    info!(
        "op resolution: {} of the {} modules the manifest pins loaded from {}, entries matched by \
         cuFuncGetParamInfo layout vs declared params ({load:?}):",
        rt.module_count(),
        m.modules.len(),
        kernels.map(|k| k.display().to_string()).unwrap_or_else(|| "the registry cache".into()),
    );
    let ones = BTreeMap::from_iter(m.vars.keys().map(|v| (v.clone(), 1)));
    for (name, modules) in rt.op_resolution() {
        let op = &m.ops[&name];
        for (li, (l, module)) in op.imp.launches.iter().zip(&modules).enumerate() {
            let label = if li == 0 { name.clone() } else { format!("  ·launch{li}") };
            let sm = match l.kernel().and_then(|k| k.shared_mem.as_ref()) {
                Some(e) => format!(", shmem {:?}", e.eval(&ones).unwrap_or(0)),
                None => String::new(),
            };
            let block = l.kernel().map_or(String::new(), |k| format!(", block {:?}", k.block));
            info!(
                "  {label:<18} {:<44} {:>2} params{block}{sm} <- {module}",
                ellipsize(l.entry(), 44),
                l.params_of(op).len(),
            );
        }
    }
}

/// A single sequence ready to generate, shared by normal execution and
/// diagnostics that execute selected programs call by call.
struct Prepared {
    caller: Caller,
    prompt_ids: Vec<i64>,
    tokenizer: tokenizers::Tokenizer,
    prompt: String,
    steps: usize,
    chunk: u64,
    step: Forward,
    chunk_forward: Forward,
    stop_tokens: Vec<i64>,
    probe_dir: Option<PathBuf>,
    probe_labels: String,
    probe_steps: usize,
}

/// Check the manifest before allocating a device, then load and stage the
/// same sequence that `kern run` would drive.
fn prepare(o: Opts, check: impl FnOnce(&Verified, &str) -> Result<()>) -> Result<Prepared> {
    let manifest_json = std::fs::read_to_string(&o.inputs.manifest)
        .with_context(|| format!("reading manifest {}", o.inputs.manifest.display()))?;
    let t0 = Instant::now();
    let verified = Verified::from_json(&manifest_json)?;
    check(&verified, &manifest_json)?;
    // One sequence: its reach, unless told otherwise (a manifest without
    // paged state takes the runtime's fit).
    let capacity = o.capacity.or_else(|| kern_runtime::seq_capacity(&verified)).map(|tokens| Capacity {
        tokens: Some(tokens),
        slots: 1,
        landings: 0,
    });
    let mut rt = Runtime::load(&verified, o.inputs.kernels.as_deref(), o.inputs.gpu, capacity, None)?;
    rt.set_eager(o.eager);
    banner(&rt, &o.inputs.manifest, o.inputs.kernels.as_deref(), t0.elapsed());

    let t0 = Instant::now();
    o.inputs.weights.bind(&mut rt)?;
    let n_weights = rt.buffer_sizes().iter().filter(|(_, k, _)| *k == BufferKind::Weight).count();
    info!("weights: {n_weights} buffers assembled from {} in {:?}", o.inputs.weights, t0.elapsed());

    let tokenizer = tokenizers::Tokenizer::from_file(&o.tokenizer).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    info!("tokenizer {} · stop tokens {:?}", o.tokenizer.display(), o.inputs.stop_tokens);
    let prompt_ids: Vec<i64> = if o.prompt_ids.is_empty() {
        tokenizer
            .encode(o.prompt.as_str(), false)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?
            .get_ids()
            .iter()
            .map(|&u| u as i64)
            .collect()
    } else {
        o.prompt_ids.clone()
    };
    ensure!(!prompt_ids.is_empty(), "empty prompt");
    info!("prompt: {} tokens {prompt_ids:?}", prompt_ids.len());

    let mut caller = Caller::new(rt)?;
    ensure!(
        o.sampling.temperature <= 0.0 || caller.protocol.samples(),
        "a temperature of {} asks for a draw, but the manifest takes no sampling fills",
        o.sampling.temperature
    );
    caller.sampling = o.sampling;
    let rows = o.rows.or_else(|| caller.protocol.row_shapes().last().copied()).unwrap_or(1);
    let step = caller.forward(Rows::Const(rows))?;
    let chunk_f = caller.chunk_forward()?;
    info!(
        "protocol: prompt through `{}` (rows as fed{}), steps through `{}` ({rows} rows{}), fills {:?}",
        chunk_f.name,
        if chunk_f.emits.is_some() { ", emits a token" } else { "" },
        step.name,
        if step.count.is_some() { ", counted" } else { "" },
        caller.protocol.fills.iter().map(|f| format!("{}={}", f.fill, f.name)).collect::<Vec<_>>()
    );
    ensure!(
        prompt_ids.len() + rows as usize <= caller.limit(),
        "prompt of {} tokens plus a {rows}-row step does not fit the sequence's {} token slots (raise --capacity)",
        prompt_ids.len(),
        caller.limit()
    );

    let chunk = chunk_size(o.chunk, caller.protocol.rows.max)?;
    Ok(Prepared {
        caller,
        prompt_ids,
        tokenizer,
        prompt: o.prompt,
        steps: o.steps,
        chunk,
        step,
        chunk_forward: chunk_f,
        stop_tokens: o.inputs.stop_tokens,
        probe_dir: o.probe_dir,
        probe_labels: o.probe_labels,
        probe_steps: o.probe_steps,
    })
}

impl Prepared {
    /// Drive the real prompt and generated tokens, letting `execute` run
    /// each staged program exactly once. A break ends generation after
    /// that program; the callback must finish its launches and synchronize.
    fn drive(
        &mut self,
        mut execute: impl FnMut(&mut Runtime, &Forward, &Vars, u64, i64) -> Result<ControlFlow<()>>,
    ) -> Result<()> {
        let caller = &mut self.caller;
        let rows = match self.step.rows {
            Rows::Const(rows) => rows,
            Rows::Var => unreachable!("prepare chooses a fixed-rows step"),
        };
        let mut invocations = BTreeMap::new();
        let n_pre = caller.prefill_len(self.prompt_ids.len())?;
        let mut generated = Vec::new();
        if n_pre > 0 {
            let t = Instant::now();
            let mut first = None;
            let mut i = 0;
            while i < n_pre {
                let count = ((n_pre - i) as u64).min(self.chunk) as usize;
                let vars = caller.stage(&self.prompt_ids[i..i + count])?;
                let invocation = invocation(&mut invocations, &self.chunk_forward.name);
                let control = execute(&mut caller.rt, &self.chunk_forward, &vars, invocation, caller.pos)?;
                caller.advance(count as u64);
                first = caller.emitted(&self.chunk_forward)?.first().copied();
                i += count;
                if control.is_break() {
                    return Ok(());
                }
            }
            let dt = t.elapsed();
            let pos = caller.pos;
            let n_chunks = (pos as u64).div_ceil(self.chunk);
            info!(
                "prefill: {pos} tokens in {n_chunks} chunk(s) of <= {} ({dt:?}, {:.0} tok/s{})",
                self.chunk,
                pos as f64 / dt.as_secs_f64(),
                if first.is_some() { ", emits the first token" } else { "" }
            );
            if let Some(first) = first {
                if self.stop_tokens.contains(&first) {
                    info!("stop token {first} at pos {pos}");
                    info!("generated ids: []");
                    println!("{}", self.prompt);
                    return Ok(());
                }
                generated.push(first);
            }
        }

        let mut decode_ns = 0;
        let mut steps = 0u32;
        let mut taken = 0;
        while generated.len() < self.steps {
            let pos = caller.pos as usize;
            let tok = if pos < self.prompt_ids.len() { self.prompt_ids[pos] } else { *generated.last().unwrap() };
            let vars = caller.stage_rows(tok, rows)?;
            let t = Instant::now();
            let invocation = invocation(&mut invocations, &self.step.name);
            let control = execute(&mut caller.rt, &self.step, &vars, invocation, caller.pos)?;
            let out = caller.emitted(&self.step)?;
            ensure!(
                !out.is_empty(),
                "`{}` handed back no token; kern run steps through a forward that samples",
                self.step.name
            );
            decode_ns += t.elapsed().as_nanos();
            steps += 1;
            caller.advance(out.len() as u64);
            taken += out.len() - 1;
            let stopped = accept(&mut generated, &out, &self.stop_tokens, self.steps);
            if let ControlFlow::Break(Some(next)) = stopped {
                info!("stop token {next} at pos {}", caller.pos);
            }
            if control.is_break() {
                return Ok(());
            }
            if stopped.is_break() || caller.pos as usize + rows as usize > caller.limit() {
                break;
            }
        }

        let gen_u32: Vec<u32> = generated.iter().map(|&t| t as u32).collect();
        let text = self.tokenizer.decode(&gen_u32, false).map_err(|e| anyhow::anyhow!("decode: {e}"))?;
        info!("generated ids: {generated:?}");
        info!(
            "{} tokens generated in {steps} steps of {rows} rows, {:.2} ms/step ({:.1} tok/s{})",
            generated.len(),
            decode_ns as f64 / 1e6 / steps.max(1) as f64,
            generated.len() as f64 * 1e9 / decode_ns.max(1) as f64,
            if rows > 1 {
                format!(
                    ", {:.2} tokens/step, {:.1}% of the {} drafted rows taken",
                    generated.len() as f64 / steps.max(1) as f64,
                    taken as f64 * 100.0 / (steps.max(1) as u64 * (rows - 1)) as f64,
                    rows - 1
                )
            } else {
                String::new()
            },
        );
        println!("{}{}", self.prompt, text);
        Ok(())
    }
}

fn invocation(counts: &mut BTreeMap<String, u64>, program: &str) -> u64 {
    let next = counts.entry(program.to_string()).or_default();
    let current = *next;
    *next += 1;
    current
}

/// Stop ids are not generated text; a speculative round may hand back
/// more tokens than the remaining budget allows.
fn accept(generated: &mut Vec<i64>, out: &[i64], stop: &[i64], max: usize) -> ControlFlow<Option<i64>> {
    for &next in out {
        if stop.contains(&next) {
            return ControlFlow::Break(Some(next));
        }
        generated.push(next);
        if generated.len() >= max {
            return ControlFlow::Break(None);
        }
    }
    ControlFlow::Continue(())
}

/// The prefill chunk: the manifest's `tokens` bound, or the smaller one
/// asked for; a bigger one is the caller's mistake, not clamped.
fn chunk_size(asked: Option<u64>, max: u64) -> Result<u64> {
    match asked {
        None => Ok(max),
        Some(c) if (1..=max).contains(&c) => Ok(c),
        Some(c) => anyhow::bail!("--chunk {c}: the manifest's `tokens` bound is {max}; a chunk can only be smaller"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speculative_tokens_stop_at_the_first_eos_or_token_budget() {
        let mut generated = vec![3];
        let first = accept(&mut generated, &[4, 5], &[9], 8);
        let second = accept(&mut generated, &[6, 9, 7], &[9], 8);
        assert_eq!(
            (generated, first, second),
            (vec![3, 4, 5, 6], ControlFlow::Continue(()), ControlFlow::Break(Some(9)))
        );

        let mut generated = vec![3];
        let stopped = accept(&mut generated, &[4, 5, 9], &[9], 3);
        assert_eq!((generated, stopped), (vec![3, 4, 5], ControlFlow::Break(None)));
    }
}
