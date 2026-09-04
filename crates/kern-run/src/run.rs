//! `kern run`: end-to-end bs=1 greedy decode over a kern manifest.
//!
//! The runtime library is model-agnostic; this is the caller-side
//! contract for the qwen3-4b-decode manifest: which input buffers exist and
//! what to put in them each step (token_ids/positions/slot_mapping/seq_lens/
//! cu_seqlens_q/block_table), prefill expressed as repeated tokens=1 decode.
//!
//! Logging goes to stderr via `tracing` (filter with `RUST_LOG`, default
//! `info`); stdout carries only the generated text.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, ensure, Context, Result};
use clap::Args;

use crate::config::{Config, Target};
use crate::{env, i64_from_le, le_bytes_i32, le_bytes_i64, prefill_emits_next_token, Caller, STOP_TOKENS};
use kern_manifest::types::{Dim, Manifest};
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

    /// DSpark speculative decoding (needs the dspark manifest programs)
    #[arg(long)]
    pub spec: bool,

    /// Token ids that end generation (comma-separated; default Qwen3's)
    #[arg(long, value_delimiter = ',', default_values_t = STOP_TOKENS)]
    pub stop_tokens: Vec<i64>,

    /// Debug: dump per-layer activations (`y` after every `*.down_proj`,
    /// embedding, logits) of the first prefill chunk and `--probe-steps`
    /// decode steps into this directory, then exit. Programs run call-range
    /// by call-range so nothing executes twice. With `--spec`: the run goes
    /// on, and every round's verify rows (ids, predictions, logits) land
    /// here instead — the path-vs-path oracle for a state layout change.
    #[arg(long)]
    pub probe_dir: Option<PathBuf>,
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
    spec: bool,
    stop_tokens: Vec<i64>,
    probe_dir: Option<PathBuf>,
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
            spec: self.spec,
            stop_tokens: self.stop_tokens,
            probe_dir: self.probe_dir,
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
    // One sequence: its reach, unless told otherwise (a manifest without
    // paged state takes the runtime's fit).
    let capacity = o
        .capacity
        .or_else(|| kern_runtime::seq_capacity(&Manifest::from_json(&manifest_json).ok()?))
        .map(|tokens| Capacity { tokens: Some(tokens), seqs: 1 });
    let mut rt = Runtime::load(&manifest_json, &o.kernels, o.gpu, capacity, None)?;
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
    for (name, calls) in &m.programs {
        info!("  program  `{name}`: {} calls", calls.len());
    }

    info!(
        "op resolution: {} of the {} modules the manifest pins loaded from {}, entries matched by \
         cuFuncGetParamInfo layout vs declared params ({:?}):",
        rt.module_count(),
        m.modules.len(),
        o.kernels.display(),
        load_t
    );
    for (name, modules) in rt.op_resolution() {
        let op = &rt.manifest.ops[&name];
        for (li, (l, module)) in op.imp.launches.iter().zip(&modules).enumerate() {
            let label = if li == 0 { name.clone() } else { format!("  ·launch{li}") };
            let sm = match l.kernel().and_then(|k| k.shared_mem.as_ref()) {
                Some(e) => format!(", shmem {:?}", e.eval(&env(1)).unwrap_or(0)),
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
    ensure!(
        prompt_ids.len() < caller.limit(),
        "prompt of {} tokens does not fit the sequence's {} token slots (raise --capacity)",
        prompt_ids.len(),
        caller.limit()
    );

    if let (Some(dir), false) = (&o.probe_dir, o.spec) {
        return probe(&mut caller, &prompt_ids, dir, o.chunk, o.probe_steps);
    }

    // Chunked prefill: repeated `prefill` calls. Two caller contracts:
    // - prefill writes state only (qwen3-4b): the first n-1 prompt tokens go
    //   through it and the final prompt token through `decode`, which
    //   produces the first logits — decode doubles as "prefill of the last
    //   token";
    // - prefill emits `next_token` (qwen3.8): every prompt token goes through
    //   it and the last chunk yields the first generated token. Hybrid GDN
    //   models need this — their chunked prefill kernels are a different
    //   arithmetic from the decode kernel, and the reference runs the last
    //   prompt token through the former.
    let chunk = o.chunk.min(caller.rt.manifest.vars["tokens"].max).max(1);
    let n_prompt = prompt_ids.len();
    let prefill_all = prefill_emits_next_token(&caller.rt.manifest);
    let n_pre = if prefill_all { n_prompt } else { n_prompt - 1 };
    let mut generated: Vec<i64> = Vec::new();
    if n_pre > 0 {
        let t = Instant::now();
        let captured = if o.spec {
            // Each chunk's fc taps must be projected into the draft's context
            // KV while positions/slot_mapping still hold this chunk's rows.
            let mut captured = false;
            let mut i = 0;
            while i < n_pre {
                let c = (n_pre - i).min(chunk as usize);
                let e = caller.stage_prefill(&prompt_ids[i..i + c])?;
                if !o.eager && c == chunk as usize {
                    if !captured {
                        caller.rt.capture("prefill", &e)?;
                        captured = true;
                    }
                    caller.rt.run_captured("prefill", &e)?;
                } else {
                    caller.rt.run("prefill", &e)?;
                }
                caller.rt.run("draft_precompute", &e)?;
                caller.advance(c as u64);
                i += c;
            }
            captured
        } else {
            caller.prefill(&prompt_ids[..n_pre], chunk, o.eager)?
        };
        let dt = t.elapsed();
        let pos = caller.pos;
        let n_chunks = (pos as u64).div_ceil(chunk);
        info!(
            "prefill: {pos} tokens in {n_chunks} chunk(s) of <= {chunk} \
             ({dt:?}, {:.0} tok/s{}{})",
            pos as f64 / dt.as_secs_f64(),
            if captured { ", graph-captured" } else { ", eager" },
            if prefill_all { ", emits next_token" } else { "" }
        );
        if prefill_all {
            let first = caller.next_token()?;
            if o.stop_tokens.contains(&first) {
                info!("stop token {first} at pos {pos}");
                println!("{}", o.prompt);
                return Ok(());
            }
            generated.push(first);
        }
    }
    if o.spec {
        let generated = spec_decode(&mut caller, &o, &prompt_ids, generated)?;
        info!("generated ids: {generated:?}");
        let gen_u32: Vec<u32> = generated.iter().map(|&t| t as u32).collect();
        let text = tokenizer.decode(&gen_u32, false).map_err(|e| anyhow::anyhow!("decode: {e}"))?;
        println!("{}{}", o.prompt, text);
        return Ok(());
    }
    let env = env(1);
    if !o.eager {
        let t = Instant::now();
        caller.rt.capture("decode", &env)?;
        info!(
            "CUDA graph: `decode` stream-captured at tokens=1, {} calls -> \
             1 graph launch/step ({:?})",
            caller.rt.manifest.programs["decode"].len(),
            t.elapsed()
        );
    }
    let mut decode_ns: u128 = 0;
    let mut decode_steps = 0u32;

    while generated.len() < o.steps {
        let pos = caller.pos as usize;
        let tok = if pos < prompt_ids.len() { prompt_ids[pos] } else { *generated.last().unwrap() };
        caller.stage_decode(tok)?;

        let t = Instant::now();
        if o.eager {
            caller.rt.run("decode", &env)?;
        } else {
            caller.rt.run_captured("decode", &env)?;
        }
        caller.advance(1);
        let pos = caller.pos;

        if (pos as usize) < prompt_ids.len() {
            continue; // prefill-as-decode: logits unused until the last prompt token
        }
        let next = caller.next_token()?;
        decode_ns += t.elapsed().as_nanos();
        decode_steps += 1;
        if o.stop_tokens.contains(&next) {
            info!("stop token {next} at pos {pos}");
            break;
        }
        generated.push(next);
        if pos as usize + 1 >= caller.limit() {
            break;
        }
    }

    let gen_u32: Vec<u32> = generated.iter().map(|&t| t as u32).collect();
    let text = tokenizer.decode(&gen_u32, false).map_err(|e| anyhow::anyhow!("decode: {e}"))?;
    info!("generated ids: {generated:?}");
    info!(
        "{} tokens generated, {:.1} ms/step ({:.1} tok/s)",
        generated.len(),
        decode_ns as f64 / 1e6 / decode_steps.max(1) as f64,
        decode_steps as f64 * 1e9 / decode_ns.max(1) as f64,
    );
    println!("{}{}", o.prompt, text);
    Ok(())
}

/// Activation probe for reference comparison (`--probe-dir`): the first
/// prefill chunk and two decode steps, each run as consecutive call
/// ranges cut after `embed` and every `l<i>.down_proj`, dumping `residual`
/// / `y` (live `tokens` rows) and the final logits as raw little-endian
/// files `<tag>.<point>.bin`, for the first prefill chunk and `steps`
/// decode steps.
fn probe(caller: &mut Caller, prompt_ids: &[i64], dir: &std::path::Path, chunk: u64, steps: usize) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    match caller.rt.manifest.buffers["y"].shape.as_slice() {
        [Dim::Var(_), Dim::Const(_)] => {}
        s => bail!("unexpected `y` shape {s:?}"),
    }
    // KERN_PROBE_LAYER=<i>: additionally dump, after every call of layer
    // `l<i>.`, the buffer its first `out` param writes (live `tokens` rows).
    let fine: Option<String> = std::env::var("KERN_PROBE_LAYER").ok().map(|l| format!("l{l}."));
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
    let run_probed =
        |rt: &Runtime, program: &str, env: &BTreeMap<String, u64>, tokens: usize, tag: &str| -> Result<()> {
            let calls = &rt.manifest.programs[program];
            let labels: Vec<String> = calls.iter().map(|c| c.label.clone().unwrap_or_default()).collect();
            let mut lo = 0;
            for (i, l) in labels.iter().enumerate() {
                let mut dumps: Vec<(String, String)> = Vec::new();
                if l == "embed" {
                    dumps.push(("embed".into(), "residual".into()));
                } else if let Some(layer) = l.strip_suffix(".down_proj") {
                    dumps.push((layer.to_string(), "y".into()));
                }
                if fine.as_deref().is_some_and(|p| l.starts_with(p)) {
                    let c = &calls[i];
                    let op = &rt.manifest.ops[&c.op];
                    for (arg, p) in c.args.iter().zip(&op.params) {
                        if let (
                            kern_manifest::types::Arg::Buf { buf, .. },
                            kern_manifest::types::ParamType::Buf { dir, .. },
                        ) = (arg, p)
                        {
                            if matches!(dir, kern_manifest::types::Dir::Out | kern_manifest::types::Dir::InOut) {
                                dumps.push((format!("{l}.{buf}"), buf.clone()));
                                break;
                            }
                        }
                    }
                }
                if dumps.is_empty() {
                    continue;
                }
                rt.run_range(program, env, lo, i + 1)?;
                lo = i + 1;
                for (point, bufname) in dumps {
                    let rows = match rt.manifest.buffers[&bufname].shape[0] {
                        Dim::Const(c) => c as usize,
                        _ => tokens,
                    };
                    let data = rt.read_buffer_prefix(&bufname, rows * row_bytes(rt, &bufname))?;
                    std::fs::write(dir.join(format!("{tag}.{point}.bin")), data)?;
                }
            }
            rt.run_range(program, env, lo, labels.len())?;
            std::fs::write(dir.join(format!("{tag}.logits.bin")), rt.read_buffer("logits")?)?;
            std::fs::write(dir.join(format!("{tag}.next_token.bin")), rt.read_output("next_token")?)?;
            Ok(())
        };
    let chunk = chunk.min(caller.rt.manifest.vars["tokens"].max).max(1) as usize;
    let prefill_all = prefill_emits_next_token(&caller.rt.manifest);
    let n_pre = if prefill_all { prompt_ids.len() } else { prompt_ids.len() - 1 };
    let c = n_pre.min(chunk);
    let e = caller.stage_prefill(&prompt_ids[..c])?;
    run_probed(&caller.rt, "prefill", &e, c, "prefill")?;
    caller.advance(c as u64);
    if c < n_pre {
        caller.prefill(&prompt_ids[c..n_pre], chunk as u64, true)?;
    }
    let mut tok = if prefill_all { caller.next_token()? } else { prompt_ids[n_pre] };
    for s in 0..steps {
        let e = caller.stage_decode(tok)?;
        run_probed(&caller.rt, "decode", &e, 1, &format!("decode{s}"))?;
        caller.advance(1);
        tok = caller.next_token()?;
    }
    info!("probe: wrote activations to {}", dir.display());
    Ok(())
}

/// DSpark speculative decoding, caller side. Per round: `draft` proposes 7
/// tokens (anchor + 6 mask queries, markov chain unrolled in-manifest),
/// `verify` runs the target once over [anchor, d0..d6] producing 8 greedy
/// predictions, the accept rule is plain prefix match (greedy spec decode is
/// lossless — output must byte-match plain decode), and `draft_precompute`
/// projects the accepted rows' target hidden states (fc taps in `fc_out`)
/// into the draft's context KV. Rollback is free: rejected slots are simply
/// overwritten by the next round (paged KV: the caller's lease, addressed
/// by position).
///
/// The manifest's `spec` block is the caller contract: `draft` runs over
/// `block` rows per round ([anchor, mask x block-1], `mask_token` filling
/// the undrafted rows), `verify` over the anchor and every drafted token
/// (`draft_tokens`' row width + 1). DSpark's block is its draft count (7
/// rows, 8 to verify); DFlash2's is one more (8 and 8). The last prompt
/// token goes through `decode_spec` unless the manifest's prefill emits the
/// first token. If the manifest takes a `num_accepted_tokens` input (the
/// target's recurrent state resumes from the checkpoint of the last
/// accepted row), it is 1 + the drafts accepted in the previous round.
fn spec_decode(caller: &mut Caller, o: &Opts, prompt_ids: &[i64], mut generated: Vec<i64>) -> Result<Vec<i64>> {
    let rt = &mut caller.rt;
    for p in ["verify", "draft", "draft_precompute"] {
        if !rt.manifest.programs.contains_key(p) {
            bail!("--spec needs program `{p}` (not in this manifest)");
        }
    }
    let n_drafts = match rt.manifest.buffers["draft_tokens"].shape.as_slice() {
        [Dim::Const(n)] | [Dim::Var(_), Dim::Const(n)] => *n as usize,
        s => bail!("unexpected draft_tokens shape {s:?}"),
    };
    let verify_n = n_drafts + 1;
    let Some(spec) = &rt.manifest.spec else {
        bail!("--spec needs the manifest's `spec` block (draft rows and the mask token); this one has none");
    };
    let (draft_rows, mask_token) = (spec.block as usize, spec.mask_token);
    ensure!(draft_rows == n_drafts || draft_rows == verify_n, "draft rows {draft_rows} vs {n_drafts} drafts");
    let has_nacc = rt.manifest.buffers.contains_key("num_accepted_tokens");
    // The recurrent state is committed by an `advance` pass after the
    // accept step (verify always resumes from the committed state, 1);
    // without one, verify itself resumes from the checkpoint of the last
    // accepted row.
    let has_advance = rt.manifest.programs.contains_key("advance");

    if generated.is_empty() {
        // Last prompt token through decode_spec: first logits + its aux tap.
        if !rt.manifest.programs.contains_key("decode_spec") {
            bail!("--spec needs program `decode_spec` (prefill does not emit next_token)");
        }
        caller.stage(&[prompt_ids[caller.pos as usize]])?;
        caller.rt.run("decode_spec", &env(1))?;
        caller.rt.run("draft_precompute", &env(1))?;
        caller.advance(1);
        let first = caller.next_token()?;
        if o.stop_tokens.contains(&first) {
            info!("stop token {first} at pos {}", caller.pos);
            return Ok(Vec::new());
        }
        generated.push(first);
    }

    if !o.eager {
        let t = Instant::now();
        caller.rt.capture("draft", &env(draft_rows as u64))?;
        caller.rt.capture("verify", &env(verify_n as u64))?;
        if has_advance {
            caller.rt.capture("advance", &env(verify_n as u64))?;
        }
        info!(
            "CUDA graphs: `draft` (tokens={draft_rows}) + `verify` (tokens={verify_n}){} captured ({:?})",
            if has_advance { " + `advance`" } else { "" },
            t.elapsed()
        );
    }

    let t0 = Instant::now();
    let mut rounds = 0u32;
    let mut accepted = 0usize;
    let mut nacc = 1i32;
    let mut per_pos = vec![0u32; n_drafts];
    'rounds: while generated.len() < o.steps && caller.pos as usize + verify_n <= caller.limit() {
        let anchor = *generated.last().unwrap();
        // Draft: [anchor, mask x (rows-1)] at pos.., non-causal.
        let mut ids = vec![anchor];
        ids.resize(draft_rows, mask_token);
        caller.stage(&ids)?;
        let rt = &mut caller.rt;
        rt.write_input("anchor_token", &le_bytes_i64(&[anchor]))?;
        if o.eager {
            rt.run("draft", &env(draft_rows as u64))?;
        } else {
            rt.run_captured("draft", &env(draft_rows as u64))?;
        }
        // Row 0 of `draft_tokens` (the buffer may be [seqs, n]).
        let mut drafts = i64_from_le(&rt.read_output("draft_tokens")?);
        drafts.truncate(n_drafts);

        // Verify: one causal target pass over [anchor, d0..] -> verify_n
        // greedy predictions; row i answers "what follows position pos+i".
        let mut vids = vec![anchor];
        vids.extend_from_slice(&drafts);
        caller.stage(&vids)?;
        let rt = &mut caller.rt;
        if has_nacc {
            rt.write_input("num_accepted_tokens", &le_bytes_i32(&[if has_advance { 1 } else { nacc }]))?;
        }
        if o.eager {
            rt.run("verify", &env(verify_n as u64))?;
        } else {
            rt.run_captured("verify", &env(verify_n as u64))?;
        }
        let vt = i64_from_le(&rt.read_output("verify_tokens")?);
        // --probe-dir: this round's verify rows — ids, predictions and
        // logits (`logits_blk`, first verify_n rows).
        if let Some(dir) = &o.probe_dir {
            std::fs::create_dir_all(dir)?;
            let vocab = rt.manifest.buffers["logits_blk"].shape[1..]
                .iter()
                .map(|d| if let Dim::Const(c) = d { *c as usize } else { 1 })
                .product::<usize>()
                * rt.manifest.buffers["logits_blk"].dtype.bytes() as usize;
            std::fs::write(
                dir.join(format!("round{rounds}.logits.bin")),
                rt.read_buffer_prefix("logits_blk", verify_n * vocab)?,
            )?;
            std::fs::write(dir.join(format!("round{rounds}.vids.bin")), le_bytes_i64(&vids))?;
            std::fs::write(dir.join(format!("round{rounds}.vt.bin")), le_bytes_i64(&vt[..verify_n]))?;
        }

        // Accept the longest matching prefix; vt[a] is the correction (or the
        // bonus token when everything matched).
        let mut a = 0;
        while a < n_drafts && drafts[a] == vt[a] {
            per_pos[a] += 1;
            a += 1;
        }
        // Project the accepted rows' aux states into the draft context KV
        // (rows 0..=a of fc_out; positions/slot_mapping still hold them).
        rt.run("draft_precompute", &env(a as u64 + 1))?;
        nacc = a as i32 + 1;
        if has_advance {
            // Commit rows 1..=a into the recurrent state: the line moves to
            // entry `a` of its line-table cell, the pass loads the state
            // after the anchor from there and stores after row a.
            caller.rt.write_input("num_accepted_tokens", &le_bytes_i32(&[nacc]))?;
            caller.set_line_column(a)?;
            if o.eager {
                caller.rt.run("advance", &env(verify_n as u64))?;
            } else {
                caller.rt.run_captured("advance", &env(verify_n as u64))?;
            }
            caller.set_line_column(0)?;
        }
        caller.advance(a as u64 + 1);
        rounds += 1;
        accepted += a;
        for &tok in &vt[..=a] {
            if o.stop_tokens.contains(&tok) {
                info!("stop token {tok} at pos {}", caller.pos);
                break 'rounds;
            }
            generated.push(tok);
            if generated.len() >= o.steps {
                break 'rounds;
            }
        }
    }
    let dt = t0.elapsed();
    let in_rounds = generated.len() - 1; // first token came from prefill / decode_spec
    info!(
        "spec: {in_rounds} tokens in {rounds} rounds ({:.2} tokens/round, \
         {:.1}% drafts accepted, per position {per_pos:?}), {:.2} ms/round ({:.1} tok/s)",
        in_rounds as f64 / rounds.max(1) as f64,
        accepted as f64 * 100.0 / (rounds.max(1) as usize * n_drafts) as f64,
        dt.as_millis() as f64 / rounds.max(1) as f64,
        in_rounds as f64 / dt.as_secs_f64().max(1e-9),
    );
    Ok(generated)
}
