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
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, ensure, Context, Result};
use clap::Args;

use crate::config::{Config, Target};
use crate::{Caller, Env, STOP_TOKENS};
use kern_manifest::protocol::{Forward, Rows};
use kern_manifest::types::{Arg, Dim, Dir};
use kern_manifest::Verified;
use kern_runtime::{Capacity, Runtime};
use tracing::info;

/// Flags of `kern run`; anything not given comes from the target in
/// kern.toml, then from the defaults.
#[derive(Args, Debug, Clone)]
pub struct RunOpts {
    /// Manifest JSON (must pass verification)
    #[arg(long)]
    pub manifest: Option<PathBuf>,

    /// Directory of cubins; steps resolve by their pinned sha256, so one dir
    /// holds every version (file names are labels)
    #[arg(long)]
    pub kernels: Option<PathBuf>,

    /// Safetensors artifact(s), tensors bound by name across all of them
    #[arg(long)]
    pub weights: Vec<PathBuf>,

    /// HF tokenizer.json
    #[arg(long)]
    pub tokenizer: Option<PathBuf>,

    /// Raw (template-free) prompt
    #[arg(long)]
    pub prompt: Option<String>,

    /// Max new tokens to generate
    #[arg(long)]
    pub steps: Option<usize>,

    /// CUDA device ordinal
    #[arg(long)]
    pub gpu: Option<usize>,

    /// State capacity in tokens (KV pages etc.); rounded down to the
    /// manifest's page unit. Default: what one sequence can reach (the
    /// manifest's page-table row)
    #[arg(long)]
    pub capacity: Option<u64>,

    /// Chunked-prefill chunk size (clamped to the manifest's tokens bound)
    #[arg(long)]
    pub chunk: Option<u64>,

    /// Skip CUDA graph capture, launch every call eagerly
    #[arg(long)]
    pub eager: bool,

    /// Rows per sequence of a decode step: a shape some program of the
    /// manifest declares (1 for a plain step, its block for a speculative
    /// round). Default: the widest declared
    #[arg(long)]
    pub rows: Option<u64>,

    /// Token ids that end generation (comma-separated; default Qwen3's)
    #[arg(long, value_delimiter = ',', default_values_t = STOP_TOKENS)]
    pub stop_tokens: Vec<i64>,

    /// Debug: dump activations of the first prefill chunk and `--probe-steps`
    /// decode steps into this directory, then exit: after every call whose
    /// label matches `--probe-labels`, the buffer it writes (live rows),
    /// plus the logits the step's tokens are taken from and the tokens.
    /// Programs run call-range by call-range so nothing executes twice.
    #[arg(long)]
    pub probe_dir: Option<PathBuf>,
    /// Call labels `--probe-dir` dumps after: comma-separated, a label
    /// matches one it equals or ends with (the file is named by the label
    /// minus its last `.part`)
    #[arg(long, default_value = "embed,.down_proj")]
    pub probe_labels: String,
    /// Decode steps `--probe-dir` dumps
    #[arg(long, default_value_t = 2)]
    pub probe_steps: usize,
}

/// Resolved options: flag, else kern.toml, else default.
struct Opts {
    manifest: PathBuf,
    kernels: PathBuf,
    weights: Vec<PathBuf>,
    tokenizer: PathBuf,
    prompt: String,
    steps: usize,
    gpu: usize,
    capacity: Option<u64>,
    chunk: u64,
    eager: bool,
    rows: Option<u64>,
    stop_tokens: Vec<i64>,
    probe_dir: Option<PathBuf>,
    probe_labels: String,
    probe_steps: usize,
}

impl RunOpts {
    fn resolve(self, cfg: Option<&Config>, t: Option<&Target>) -> Result<Opts> {
        let need = |what: &str| {
            anyhow::anyhow!(
                "no --{what} and no target in {} to take it from",
                cfg.map_or(crate::config::FILE.to_string(), |c| c.path.display().to_string())
            )
        };
        Ok(Opts {
            manifest: self.manifest.or_else(|| t.map(|t| t.manifest.clone())).ok_or_else(|| need("manifest"))?,
            kernels: self.kernels.or_else(|| t.map(|t| t.kernels.clone())).ok_or_else(|| need("kernels"))?,
            weights: if self.weights.is_empty() {
                t.map(|t| t.weights.clone()).filter(|w| !w.is_empty()).ok_or_else(|| need("weights"))?
            } else {
                self.weights
            },
            tokenizer: self
                .tokenizer
                .or_else(|| t.and_then(|t| t.tokenizer.clone()))
                .ok_or_else(|| need("tokenizer"))?,
            prompt: self
                .prompt
                .or_else(|| cfg.and_then(|c| c.run.prompt.clone()))
                .unwrap_or_else(|| "The capital of France is".into()),
            steps: self.steps.or_else(|| cfg.and_then(|c| c.run.steps)).unwrap_or(32),
            gpu: self.gpu.or_else(|| cfg.and_then(|c| c.gpu)).unwrap_or(0),
            capacity: self.capacity.or_else(|| cfg.and_then(|c| c.capacity)),
            chunk: self.chunk.or_else(|| cfg.and_then(|c| c.run.chunk)).unwrap_or(512),
            eager: self.eager,
            rows: self.rows,
            stop_tokens: self.stop_tokens,
            probe_dir: self.probe_dir,
            probe_labels: self.probe_labels,
            probe_steps: self.probe_steps,
        })
    }
}

/// `kern run`.
pub fn run(o: RunOpts, cfg: Option<&Config>, target: Option<&Target>) -> Result<()> {
    execute(o.resolve(cfg, target)?)
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

fn execute(o: Opts) -> Result<()> {
    let manifest_json =
        std::fs::read_to_string(&o.manifest).with_context(|| format!("reading manifest {}", o.manifest.display()))?;
    let t0 = Instant::now();
    let verified = Verified::from_json(&manifest_json)?;
    // One sequence: its reach, unless told otherwise (a manifest without
    // paged state takes the runtime's fit).
    let capacity = o
        .capacity
        .or_else(|| kern_runtime::seq_capacity(&verified))
        .map(|tokens| Capacity { tokens: Some(tokens), seqs: 1 });
    let mut rt = Runtime::load(&verified, &o.kernels, o.gpu, capacity, None)?;
    let load_t = t0.elapsed();

    let m = &rt.manifest;
    info!("manifest `{}` (schema v{}, {}): verified", m.model, m.schema_version, o.manifest.display());
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
            None => String::new(),
        };
        info!("  program  `{name}`: {} calls{shape}", p.calls.len());
    }

    info!(
        "op resolution: {} of the {} modules the manifest pins loaded from {}, entries matched by \
         cuFuncGetParamInfo layout vs declared params ({:?}):",
        rt.module_count(),
        m.modules.len(),
        o.kernels.display(),
        load_t
    );
    let e1 = BTreeMap::from_iter(m.vars.keys().map(|v| (v.clone(), 1)));
    for (name, modules) in rt.op_resolution() {
        let op = &rt.manifest.ops[&name];
        for (li, (l, module)) in op.imp.launches.iter().zip(&modules).enumerate() {
            let label = if li == 0 { name.clone() } else { format!("  ·launch{li}") };
            let sm = match l.kernel().and_then(|k| k.shared_mem.as_ref()) {
                Some(e) => format!(", shmem {:?}", e.eval(&e1).unwrap_or(0)),
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

    let t0 = Instant::now();
    let blobs = o
        .weights
        .iter()
        .map(|p| std::fs::read(p).with_context(|| format!("reading weights {}", p.display())))
        .collect::<Result<Vec<_>>>()?;
    let blob_len: usize = blobs.iter().map(Vec::len).sum();
    rt.load_weights(&blobs.iter().map(Vec::as_slice).collect::<Vec<_>>())?;
    drop(blobs);
    let n_weights = by_kind.get("weight").map_or(0, |e| e.0);
    info!(
        "weights: {n_weights} tensors bound by name from {} ({}) in {:?}",
        o.weights.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(" + "),
        human(blob_len as u64),
        t0.elapsed()
    );

    let tokenizer = tokenizers::Tokenizer::from_file(&o.tokenizer).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    let prompt_ids: Vec<i64> = tokenizer
        .encode(o.prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?
        .get_ids()
        .iter()
        .map(|&u| u as i64)
        .collect();
    ensure!(!prompt_ids.is_empty(), "empty prompt");
    info!("prompt: {} tokens {prompt_ids:?}", prompt_ids.len());

    let mut caller = Caller::new(rt)?;
    for p in caller.protocol.once.clone() {
        caller.rt.run(&p, &e1)?;
    }
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

    if let Some(dir) = &o.probe_dir {
        return probe(&mut caller, &prompt_ids, dir, &o.probe_labels, o.chunk, &step, o.probe_steps);
    }

    // Chunked prefill. A chunk program that hands a token back takes every
    // prompt token and yields the first generated one (a hybrid GDN model:
    // its chunked prefill kernels are a different arithmetic from the
    // decode kernel, and the reference runs the last prompt token through
    // the former). One that only writes state takes the first n-1 prompt
    // tokens, and the last one goes through the step program.
    let chunk = o.chunk.min(caller.protocol.rows.max).max(1);
    let n_prompt = prompt_ids.len();
    let prefill_all = chunk_f.emits.is_some();
    let n_pre = if prefill_all { n_prompt } else { n_prompt - 1 };
    let mut generated: Vec<i64> = Vec::new();
    if n_pre > 0 {
        let t = Instant::now();
        let (first, captured) = caller.prefill(&prompt_ids[..n_pre], chunk, o.eager)?;
        let dt = t.elapsed();
        let pos = caller.pos;
        let n_chunks = (pos as u64).div_ceil(chunk);
        info!(
            "prefill: {pos} tokens in {n_chunks} chunk(s) of <= {chunk} \
             ({dt:?}, {:.0} tok/s{}{})",
            pos as f64 / dt.as_secs_f64(),
            if captured { ", graph-captured" } else { ", eager" },
            if prefill_all { ", emits the first token" } else { "" }
        );
        if let Some(first) = first {
            if o.stop_tokens.contains(&first) {
                info!("stop token {first} at pos {pos}");
                println!("{}", o.prompt);
                return Ok(());
            }
            generated.push(first);
        }
    }

    let env = caller.protocol.env(1, rows, 1);
    if !o.eager {
        let t = Instant::now();
        caller.rt.capture(&step.name, &env)?;
        info!(
            "CUDA graph: `{}` stream-captured at {rows} rows, {} calls -> 1 graph launch/step ({:?})",
            step.name,
            caller.rt.manifest.programs[&step.name].calls.len(),
            t.elapsed()
        );
    }
    let mut decode_ns: u128 = 0;
    let mut steps = 0u32;
    let mut taken = 0usize;

    'steps: while generated.len() < o.steps {
        let pos = caller.pos as usize;
        let tok = if pos < prompt_ids.len() { prompt_ids[pos] } else { *generated.last().unwrap() };
        caller.stage_rows(tok, rows)?;
        let t = Instant::now();
        if o.eager {
            caller.rt.run(&step.name, &env)?;
        } else {
            caller.rt.run_captured(&step.name, &env)?;
        }
        let out = caller.emitted(&step)?.0;
        decode_ns += t.elapsed().as_nanos();
        steps += 1;
        caller.advance(out.len() as u64);
        taken += out.len() - 1;
        for next in out {
            if o.stop_tokens.contains(&next) {
                info!("stop token {next} at pos {}", caller.pos);
                break 'steps;
            }
            generated.push(next);
            if generated.len() >= o.steps {
                break 'steps;
            }
        }
        if caller.pos as usize + rows as usize > caller.limit() {
            break;
        }
    }

    let gen_u32: Vec<u32> = generated.iter().map(|&t| t as u32).collect();
    let text = tokenizer.decode(&gen_u32, false).map_err(|e| anyhow::anyhow!("decode: {e}"))?;
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
    println!("{}{}", o.prompt, text);
    Ok(())
}

/// Activation probe for reference comparison (`--probe-dir`): the first
/// prefill chunk and `steps` decode steps, each run as consecutive call
/// ranges cut after every call whose label equals or ends with one of
/// `labels`, dumping the
/// buffer that call writes (live rows) as `<tag>.<point>.bin` where
/// `point` is the label minus its last `.part`, then the step's logits
/// (the buffer its `tokens` output is taken from) and the tokens.
fn probe(
    caller: &mut Caller,
    prompt_ids: &[i64],
    dir: &std::path::Path,
    labels: &str,
    chunk: u64,
    step: &Forward,
    steps: usize,
) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let labels: Vec<&str> = labels.split(',').filter(|p| !p.is_empty()).collect();
    let matches = |l: &str| labels.iter().any(|p| l == *p || l.ends_with(p));
    let row_bytes = |rt: &Runtime, name: &str| -> usize {
        let b = &rt.manifest.buffers[name];
        b.shape[1..]
            .iter()
            .map(|d| match d {
                Dim::Const(c) => *c as usize,
                _ => 1,
            })
            .product::<usize>()
            * b.dtype.bytes() as usize
    };
    // The buffer a call reads or writes through its first param of `dir`.
    let param_buf = |rt: &Runtime, c: &kern_manifest::types::Call, want: &[Dir]| -> Option<String> {
        let op = &rt.manifest.ops[&c.op];
        c.args.iter().zip(&op.params).find_map(|(a, p)| match (a, p.dir()) {
            (Arg::Buf { buf, .. }, Some(d)) if want.contains(&d) => Some(buf.clone()),
            _ => None,
        })
    };
    let run_probed = |caller: &Caller, f: &Forward, env: &Env, rows: usize, tag: &str| -> Result<()> {
        let rt = &caller.rt;
        let calls = &rt.manifest.programs[&f.name].calls;
        let mut lo = 0;
        for (i, c) in calls.iter().enumerate() {
            let l = c.label.clone().unwrap_or_default();
            if !matches(&l) {
                continue;
            }
            let Some(bufname) = param_buf(rt, c, &[Dir::Out, Dir::InOut]) else { continue };
            rt.run_range(&f.name, env, lo, i + 1)?;
            lo = i + 1;
            let n = match rt.manifest.buffers[&bufname].shape[0] {
                Dim::Const(c) => c as usize,
                _ => rows,
            };
            let point = l.rsplit_once('.').map_or(l.as_str(), |(head, _)| head);
            let data = rt.read_buffer_prefix(&bufname, n * row_bytes(rt, &bufname))?;
            std::fs::write(dir.join(format!("{tag}.{point}.bin")), data)?;
        }
        rt.run_range(&f.name, env, lo, calls.len())?;
        if let Some(i) = f.emits {
            let tokens = &caller.protocol.fills[i];
            if let Some(logits) = calls
                .iter()
                .rev()
                .find(|c| param_buf(rt, c, &[Dir::Out, Dir::InOut]).as_deref() == Some(&tokens.name))
                .and_then(|c| param_buf(rt, c, &[Dir::In]))
            {
                std::fs::write(dir.join(format!("{tag}.logits.bin")), rt.read_buffer(&logits)?)?;
            }
            std::fs::write(dir.join(format!("{tag}.tokens.bin")), rt.read_output(&tokens.name)?)?;
        }
        Ok(())
    };
    let chunk_f = caller.chunk_forward()?;
    let chunk = chunk.min(caller.protocol.rows.max).max(1) as usize;
    let prefill_all = chunk_f.emits.is_some();
    let n_pre = if prefill_all { prompt_ids.len() } else { prompt_ids.len() - 1 };
    let c = n_pre.min(chunk);
    let e = caller.stage(&prompt_ids[..c])?;
    run_probed(caller, &chunk_f, &e, c, "chunk")?;
    caller.advance(c as u64);
    let mut first = caller.emitted(&chunk_f)?.0.first().copied();
    if c < n_pre {
        first = caller.prefill(&prompt_ids[c..n_pre], chunk as u64, true)?.0;
    }
    let mut tok = match first {
        Some(t) => t,
        None => prompt_ids[n_pre],
    };
    let rows = match step.rows {
        Rows::Const(r) => r,
        Rows::Var => bail!("the step program takes rows as fed; probe steps need a fixed-rows program"),
    };
    for s in 0..steps {
        let e = caller.stage_rows(tok, rows)?;
        run_probed(caller, step, &e, rows as usize, &format!("decode{s}"))?;
        let out = caller.emitted(step)?.0;
        caller.advance(out.len() as u64);
        tok = *out.last().unwrap();
    }
    info!("probe: wrote activations to {}", dir.display());
    Ok(())
}
