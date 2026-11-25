//! One real sequence driven up to a selected program invocation, with all
//! of its selected call inputs and state exported before the process exits.

use std::collections::BTreeMap;
use std::ops::{ControlFlow, Range};
use std::path::PathBuf;
use std::str::FromStr;

use anyhow::{ensure, Context, Result};
use clap::Args;
use serde_json::json;

use super::artifact::Writer;
use super::capture;
use super::plan::{Phase, Plan, Selection};
use crate::config::{Config, Target};
use crate::run::{self, RunOpts};

#[derive(Args, Debug)]
pub struct DumpOpts {
    #[command(flatten)]
    run: RunOpts,
    /// The protocol forward whose real invocation is observed
    #[arg(long)]
    program: String,
    /// Zero-based execution of the selected program, independent of token count
    #[arg(long, default_value_t = 0)]
    invocation: u64,
    /// Exact call labels; every repeated label is selected
    #[arg(long, value_delimiter = ',', required_unless_present = "call_index", conflicts_with = "call_index")]
    call_label: Vec<String>,
    /// Zero-based call indices in manifest order
    #[arg(long, value_delimiter = ',', required_unless_present = "call_label", conflicts_with = "call_label")]
    call_index: Vec<usize>,
    /// Capture before, after, or both; after alone lacks replay inputs
    #[arg(long, value_enum, default_value = "both")]
    phase: Phase,
    /// Optional partial storage image: buffer:<name>=<start>:<end> or state:<name>=<start>:<end> (bytes)
    #[arg(long)]
    range: Vec<Region>,
    /// Maximum raw payload bytes; weights and state are included (default 1 GiB)
    #[arg(long, default_value_t = 1 << 30, value_parser = clap::value_parser!(u64).range(1..))]
    max_bytes: u64,
    /// An absent or empty directory; dump.json appears only after success
    #[arg(long)]
    out: PathBuf,
}

#[derive(Debug, Clone)]
struct Region {
    name: String,
    bytes: Range<u64>,
}

impl FromStr for Region {
    type Err = anyhow::Error;

    fn from_str(text: &str) -> Result<Self> {
        let (name, bounds) = text.split_once('=').context("range needs storage=start:end, in bytes")?;
        let (kind, object) = name.split_once(':').context("range storage needs buffer:<name> or state:<name>")?;
        ensure!(
            ["buffer", "state"].contains(&kind) && !object.is_empty(),
            "range storage needs buffer:<name> or state:<name>"
        );
        let (start, end) = bounds.split_once(':').context("range needs start:end, in bytes")?;
        let (start, end) = (start.parse::<u64>()?, end.parse::<u64>()?);
        ensure!(start < end, "range start must be less than end");
        Ok(Self { name: name.to_string(), bytes: start..end })
    }
}

/// Capture the selected calls on their real inputs. Every directly bound
/// weight and state is included automatically; partial ranges stay explicit.
pub fn run(o: DumpOpts, cfg: Option<&Config>, target: Option<&Target>) -> Result<()> {
    let selection = Selection {
        program: o.program,
        invocation: o.invocation,
        labels: o.call_label,
        indices: o.call_index,
        phase: o.phase,
    };
    let mut ranges = BTreeMap::new();
    for region in o.range {
        ensure!(ranges.insert(region.name.clone(), region.bytes).is_none(), "duplicate dump range `{}`", region.name);
    }
    if o.out.try_exists()? {
        ensure!(
            std::fs::read_dir(&o.out)?.next().transpose()?.is_none(),
            "dump directory {} is not empty",
            o.out.display()
        );
    }
    let opts = o.run.resolve_dump(cfg, target, &selection.program)?;
    let mut checked = None;
    let mut ready = run::prepare(opts, |manifest, _| {
        checked = Some(Plan::check(manifest, selection)?);
        Ok(())
    })?;
    let plan = checked.context("dump selection was not checked")?;
    let mut writer = Writer::create(&o.out, &ready.manifest_json, o.max_bytes)?;
    let protocol = ready.caller.protocol.clone();
    let mut observation = None;
    let mut emitted = Vec::new();
    ready.drive(|rt, forward, vars, invocation, position| {
        let selected = forward.name == plan.selection.program && invocation == plan.selection.invocation;
        if selected {
            observation = Some(capture::record(rt, &plan, vars, position, &ranges, &mut writer)?);
        } else {
            rt.issue(&forward.name, vars)?;
            rt.synchronize()?;
        }
        if let Some(index) = forward.emits {
            let fill = &protocol.fills[index];
            let tokens = fill.decode(&rt.read_output(&fill.name)?);
            let count = match forward.count {
                Some(index) => {
                    let count = &protocol.fills[index];
                    let values = count.decode(&rt.read_output(&count.name)?);
                    let n = *values.first().context("count output is empty")?;
                    ensure!(n >= 1 && n as usize <= tokens.len(), "invalid emitted token count {n}");
                    n as usize
                }
                None => 1,
            };
            emitted.push(json!({"program":forward.name,"invocation":invocation,"tokens":&tokens[..count]}));
        }
        Ok(if selected { ControlFlow::Break(()) } else { ControlFlow::Continue(()) })
    })?;
    let observation = observation.with_context(|| {
        format!(
            "dump program `{}` invocation {} was not reached before generation stopped",
            plan.selection.program, plan.selection.invocation
        )
    })?;
    let metadata = json!({
        "kern_version":crate::VERSION.as_str(),
        "prompt_ids":ready.prompt_ids,
        "sampling":ready.sampling,
        "capacity":ready.caller.rt.capacity(),
        "sequence_slots":ready.caller.rt.seq_slots(),
        "selection":plan.selection,
        "observation":observation,
        "emitted":emitted
    });
    let artifact = writer.finish(metadata)?;
    tracing::info!("dump: {} blobs written to {}", artifact.blobs.len(), o.out.display());
    Ok(())
}
