//! The cut gate: the stage manifests of `kern cut`, run in sequence in
//! one process, compute what the unsplit manifest computes, bit for bit.
//!
//!   pp_chain --manifest whole.json --stages s0.json,s1.json --kernels dir
//!            --weights dir --gpus 0,1,2 --prompts ids.txt [--chunk 512]
//!            [--steps 8] [--capacity 65536]
//!
//! The first GPU runs the whole manifest, the rest one stage each, the
//! stages wired through their mailboxes as a pipeline is (here with
//! in-process handles). Each line of `--prompts` is one prompt's token
//! ids. A prompt but its last token goes through the chunk program in
//! `--chunk` pieces, then `--steps` one-row steps feed back what they
//! pick. After every step the input of the call writing the tokens (the
//! logits) and the tokens themselves must be equal on the whole and on the
//! last stage. All stages are issued before any is waited on: a stage's
//! device waits for its predecessor's mailbox on its own.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{bail, ensure, Context, Result};
use kern_manifest::protocol::{Forward, Rows};
use kern_manifest::types::Arg;
use kern_manifest::{Protocol, Verified};
use kern_pool::Lease;
use kern_run::{fills, run_once, write_named, Vars};
use kern_runtime::{Capacity, GroupRank, PeerHandle, Runtime, Topology};

struct Opts {
    manifest: PathBuf,
    stages: Vec<PathBuf>,
    kernels: PathBuf,
    weights: Vec<String>,
    gpus: Vec<usize>,
    prompts: PathBuf,
    chunk: usize,
    steps: usize,
    capacity: u64,
}

fn opts() -> Result<Opts> {
    let mut o = Opts {
        manifest: PathBuf::new(),
        stages: Vec::new(),
        kernels: PathBuf::new(),
        weights: Vec::new(),
        gpus: Vec::new(),
        prompts: PathBuf::new(),
        chunk: 512,
        steps: 8,
        capacity: 65536,
    };
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let v = args.next().with_context(|| format!("{a}: a value"))?;
        match a.as_str() {
            "--manifest" => o.manifest = v.into(),
            "--stages" => o.stages = v.split(',').map(PathBuf::from).collect(),
            "--kernels" => o.kernels = v.into(),
            "--weights" => o.weights.push(v),
            "--gpus" => o.gpus = v.split(',').map(str::parse).collect::<Result<_, _>>()?,
            "--prompts" => o.prompts = v.into(),
            "--chunk" => o.chunk = v.parse()?,
            "--steps" => o.steps = v.parse()?,
            "--capacity" => o.capacity = v.parse()?,
            _ => bail!("unknown argument {a}"),
        }
    }
    ensure!(o.gpus.len() == o.stages.len() + 1, "one GPU for the whole manifest and one per stage");
    Ok(o)
}

fn load(path: &PathBuf, o: &Opts, gpu: usize, weights: &kern_run::Weights) -> Result<(Runtime, Protocol)> {
    let m = Verified::from_json(&std::fs::read_to_string(path)?).with_context(|| format!("{}", path.display()))?;
    let topology = m.topology.as_ref().map(|t| Topology {
        groups: t.groups.iter().map(|(g, &size)| (g.clone(), GroupRank { index: 0, size })).collect(),
    });
    let capacity = Capacity { tokens: Some(o.capacity), seqs: 2 };
    let mut rt = Runtime::load(&m, Some(&o.kernels), gpu, Some(capacity), topology.as_ref())?;
    weights.bind(&mut rt)?;
    let p = Protocol::check_unsampled(&rt.manifest)?;
    Ok((rt, p))
}

/// The buffer the call writing `tokens` reads: the logits it picks from.
fn logits(m: &Verified, program: &str, tokens: &str) -> Result<String> {
    let call = m.programs[program]
        .calls
        .iter()
        .rev()
        .find(|c| c.args.iter().any(|a| matches!(a, Arg::Buf { buf, .. } if buf == tokens)))
        .with_context(|| format!("`{program}` writes no `{tokens}`"))?;
    call.args
        .iter()
        .find_map(|a| match a {
            Arg::Buf { buf, .. } if buf != tokens => Some(buf.clone()),
            _ => None,
        })
        .with_context(|| format!("the call writing `{tokens}` reads no buffer"))
}

/// Stage one call of `ids` at `pos` into `rt` from the values `p` and
/// `lease` give.
fn item(p: &Protocol, lease: &Lease, pos: usize, ids: &[i64]) -> Result<(Vars, BTreeMap<String, Vec<i64>>)> {
    let (vars, values) = fills(p, std::slice::from_ref(lease), &[pos], &[ids.len()], ids);
    let mut rows: BTreeMap<String, Vec<i64>> = values.into_iter().map(|(f, v)| (f.name.clone(), v)).collect();
    for t in &p.page_tables {
        let mut v = Vec::new();
        lease.extend_row(&t.name, &mut v)?;
        rows.insert(t.name.clone(), v.into_iter().map(i64::from).collect());
    }
    Ok((vars, rows))
}

fn issue(rt: &mut Runtime, p: &Protocol, program: &str, vars: &Vars, rows: &BTreeMap<String, Vec<i64>>) -> Result<()> {
    let vars: Vars =
        vars.iter().filter(|(k, _)| rt.manifest.vars.contains_key(*k)).map(|(k, v)| (k.clone(), *v)).collect();
    for (name, v) in rows {
        write_named(rt, p, name, v, &vars)?;
    }
    Ok(rt.issue(program, &vars)?)
}

fn forwards(p: &Protocol) -> Result<(Forward, Forward)> {
    Ok((
        p.chunk().cloned().context("no chunk program")?,
        p.forward(1, Rows::Const(1)).cloned().context("no one-row program")?,
    ))
}

fn main() -> Result<()> {
    tracing_subscriber::fmt().with_env_filter(tracing_subscriber::EnvFilter::from_default_env()).init();
    let o = opts()?;
    let weights = kern_run::Weights::parse(&o.weights)?;
    let (mut whole, pw) = load(&o.manifest, &o, o.gpus[0], &weights)?;
    let mut stages: Vec<(Runtime, Protocol)> =
        o.stages.iter().zip(&o.gpus[1..]).map(|(s, &g)| load(s, &o, g, &weights)).collect::<Result<_>>()?;
    let exports: Vec<BTreeMap<String, PeerHandle>> =
        stages.iter().map(|(rt, _)| rt.export_handles()).collect::<Result<_, _>>()?;
    for (s, (rt, _)) in stages.iter_mut().enumerate() {
        let groups: Vec<String> = rt.manifest.topology.iter().flat_map(|t| t.groups.keys().cloned()).collect();
        for g in groups {
            rt.import_peers(&g, &[exports[s].clone(), exports[s + 1].clone()])?;
        }
    }
    run_once(&whole, &pw)?;
    for (rt, p) in &stages {
        run_once(rt, p)?;
    }

    let (wchunk, wstep) = forwards(&pw)?;
    let (_, sstep) = forwards(&stages[0].1)?;
    let tokens = &pw.fills[wstep.emits.context("the whole manifest's step hands back no tokens")?];
    let (last, lp) = stages.last().unwrap();
    let ltokens = lp.fills[lp
        .forwards
        .iter()
        .find(|f| f.name == sstep.name)
        .and_then(|f| f.emits)
        .context("the last stage's step hands back no tokens")?]
    .clone();
    let wlogits = logits(&whole.manifest, &wstep.name, &tokens.name)?;
    let llogits = logits(&last.manifest, &sstep.name, &ltokens.name)?;
    ensure!(wlogits == llogits, "logits are `{wlogits}` whole, `{llogits}` on the last stage");

    let wlease = whole.lease(o.capacity as usize)?;
    let slease = stages[0].0.lease(o.capacity as usize)?;
    let prompts: Vec<Vec<i64>> = std::fs::read_to_string(&o.prompts)?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split_whitespace().map(str::parse).collect::<Result<_, _>>())
        .collect::<Result<_, _>>()?;
    let mut failed = 0;
    for (n, prompt) in prompts.iter().enumerate() {
        let mut pos = 0;
        let mut calls: Vec<(String, usize, Vec<i64>)> = Vec::new();
        while pos < prompt.len() - 1 {
            let c = (prompt.len() - 1 - pos).min(o.chunk);
            calls.push((wchunk.name.clone(), pos, prompt[pos..pos + c].to_vec()));
            pos += c;
        }
        for (program, pos, ids) in &calls {
            let (vars, rows) = item(&pw, &wlease, *pos, ids)?;
            issue(&mut whole, &pw, program, &vars, &rows)?;
            let (vars, rows) = item(&stages[0].1, &slease, *pos, ids)?;
            for (rt, p) in &mut stages {
                issue(rt, p, program, &vars, &rows)?;
            }
        }
        let mut tok = *prompt.last().unwrap();
        let mut picked = Vec::new();
        let mut same = true;
        for step in 0..o.steps {
            let at = prompt.len() - 1 + step;
            let (vars, rows) = item(&pw, &wlease, at, &[tok])?;
            issue(&mut whole, &pw, &wstep.name, &vars, &rows)?;
            let (vars, rows) = item(&stages[0].1, &slease, at, &[tok])?;
            for (rt, p) in &mut stages {
                issue(rt, p, &sstep.name, &vars, &rows)?;
            }
            for (rt, _) in &stages {
                rt.synchronize()?;
            }
            let (last, _) = stages.last().unwrap();
            let wt = tokens.decode(&whole.read_output(&tokens.name)?)[0];
            let st = ltokens.decode(&last.read_output(&ltokens.name)?)[0];
            let wl = whole.read_buffer(&wlogits)?;
            let sl = last.read_buffer(&llogits)?;
            if wt != st || wl != sl {
                let differ = wl.iter().zip(&sl).filter(|(a, b)| a != b).count();
                println!(
                    "prompt {n} step {step}: token {wt} whole, {st} staged; logits bytes differing {differ}/{}",
                    wl.len()
                );
                same = false;
            }
            picked.push(wt);
            tok = wt;
        }
        println!(
            "prompt {n}: {} tokens, {} chunk calls, {} steps: {} {:?}",
            prompt.len(),
            calls.len(),
            o.steps,
            if same { "IDENTICAL" } else { "DIFFERENT" },
            picked
        );
        failed += usize::from(!same);
    }
    ensure!(failed == 0, "{failed} of {} prompts differ", prompts.len());
    println!("PASS: {} prompts bitwise identical", prompts.len());
    Ok(())
}
