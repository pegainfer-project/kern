//! Synchronous call-boundary images from one real invocation. Prefixes and
//! suffixes execute once in list order; no image is reconstructed from a
//! different execution's state.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use anyhow::{ensure, Context, Result};
use kern_manifest::types::{Arg, BufferKind};
use kern_runtime::Runtime;
use serde_json::{json, Value};

use super::artifact::{Blob, Writer};
use super::plan::{Phase, Plan, Storage, StorageKind};
use crate::Vars;

/// Preserve selected calls of one invocation and finish the whole program.
/// Ranges are byte intervals keyed by `buffer:<name>` or `state:<name>`;
/// omitted intervals preserve the whole allocation, including argument aliases.
pub fn record(
    rt: &Runtime,
    plan: &Plan,
    vars: &Vars,
    position: i64,
    ranges: &BTreeMap<String, Range<u64>>,
    writer: &mut Writer,
) -> Result<Value> {
    let allocations = rt
        .buffer_sizes()
        .into_iter()
        .map(|(name, _, bytes)| (format!("buffer:{name}"), bytes))
        .chain(rt.state_sizes().into_iter().map(|(name, _, bytes)| (format!("state:{name}"), bytes)))
        .collect();
    let prepared = plan
        .calls
        .iter()
        .map(|&index| Ok((index, plan.storages(&rt.manifest, index, vars, &allocations)?)))
        .collect::<Result<Vec<_>>>()?;
    let keys: BTreeSet<_> = prepared.iter().flat_map(|(_, stores)| stores.iter().map(key)).collect();
    for (name, range) in ranges {
        ensure!(keys.contains(name), "dump range `{name}` is not bound by a selected call");
        ensure!(range.start < range.end, "dump range `{name}` must have start < end");
    }
    for (_, storages) in &prepared {
        for storage in storages {
            let range = region(storage, ranges);
            ensure!(
                range.end <= storage.bytes,
                "dump range `{}` [{}, {}) exceeds allocation {}",
                key(storage),
                range.start,
                range.end,
                storage.bytes
            );
        }
    }

    let program = &plan.selection.program;
    let calls = &rt.manifest.programs[program].calls;
    let mut immutable = BTreeMap::new();
    let mut observed = Vec::new();
    let mut lo = 0;
    for (index, storages) in prepared {
        rt.run_range(program, vars, lo, index)?;
        let before = if plan.selection.phase != Phase::After {
            images(rt, &storages, ranges, writer, &mut immutable)?
        } else {
            BTreeMap::new()
        };
        rt.run_range(program, vars, index, index + 1)?;
        let mut after = BTreeMap::new();
        if plan.selection.phase != Phase::Before {
            for storage in &storages {
                let name = key(storage);
                let blob = image(rt, storage, region(storage, ranges), writer, &mut immutable)?;
                after.insert(name, blob);
            }
        }
        let call = &calls[index];
        let op = &rt.manifest.ops[&call.op];
        let scalars = call
            .args
            .iter()
            .enumerate()
            .filter_map(|(i, a)| {
                scalar(a, vars)
                    .transpose()
                    .map(|v| v.map(|value| json!({"index":i,"param":op.params[i],"value":value})))
            })
            .collect::<Result<Vec<_>>>()?;
        let launches = op
            .imp
            .launches
            .iter()
            .map(|launch| {
                let active = match launch.when() {
                    Some(when) => when.holds(
                        *vars.get(&when.var).with_context(|| format!("no value for launch var `{}`", when.var))?,
                    ),
                    None => true,
                };
                Ok(json!({"launch":launch,"condition_holds":active}))
            })
            .collect::<Result<Vec<_>>>()?;
        let storage_metadata = storages.iter().map(|s| {
            let range = region(s, ranges);
            let declaration = match s.kind {
                StorageKind::Buffer => json!(rt.manifest.buffers[&s.name]),
                StorageKind::State => json!(rt.manifest.states[&s.name]),
            };
            json!({"id":key(s),"storage":s,"declaration":declaration,"range":[range.start,range.end],"extent":"allocation; logical argument extent is not declared"})
        }).collect::<Vec<_>>();
        observed.push(json!({"index":index,"label":call.label,"call":call,"op":op,"scalars":scalars,"launches":launches,"storages":storage_metadata,"before":before,"after":after}));
        lo = index + 1;
    }
    rt.run_range(program, vars, lo, calls.len())?;
    let mut replay_requirements = Vec::new();
    if plan.selection.phase == Phase::After {
        replay_requirements.push("before images were not captured".to_string());
    }
    for (name, range) in ranges {
        if range.start != 0 || range.end != allocations[name] {
            replay_requirements.push(format!("`{name}` is partial; its complete argument access extent is unknown"));
        }
    }
    Ok(
        json!({"program":program,"invocation":plan.selection.invocation,"cursor":position,"vars":vars,"execution":"eager_serial","rank":0,"phase":plan.selection.phase,"replay_requirements":replay_requirements,"calls":observed}),
    )
}

fn key(storage: &Storage) -> String {
    match storage.kind {
        StorageKind::Buffer => format!("buffer:{}", storage.name),
        StorageKind::State => format!("state:{}", storage.name),
    }
}

fn region(storage: &Storage, ranges: &BTreeMap<String, Range<u64>>) -> Range<u64> {
    ranges.get(&key(storage)).cloned().unwrap_or(0..storage.bytes)
}

fn images(
    rt: &Runtime,
    storages: &[Storage],
    ranges: &BTreeMap<String, Range<u64>>,
    writer: &mut Writer,
    immutable: &mut BTreeMap<String, Blob>,
) -> Result<BTreeMap<String, Blob>> {
    storages.iter().map(|s| Ok((key(s), image(rt, s, region(s, ranges), writer, immutable)?))).collect()
}

fn image(
    rt: &Runtime,
    storage: &Storage,
    range: Range<u64>,
    writer: &mut Writer,
    immutable: &mut BTreeMap<String, Blob>,
) -> Result<Blob> {
    let name = key(storage);
    let weight = storage.kind == StorageKind::Buffer && rt.manifest.buffers[&storage.name].kind == BufferKind::Weight;
    if weight {
        if let Some(blob) = immutable.get(&name) {
            return Ok(blob.clone());
        }
    }
    let blob = writer.blob(range.end - range.start, |offset, len| {
        let offset = usize::try_from(range.start.checked_add(offset).context("dump byte offset overflow")?)?;
        Ok(match storage.kind {
            StorageKind::Buffer => rt.read_buffer_at(&storage.name, offset, len)?,
            StorageKind::State => rt.read_state_at(&storage.name, offset, len)?,
        })
    })?;
    if weight {
        immutable.insert(name, blob.clone());
    }
    Ok(blob)
}

fn scalar(arg: &Arg, vars: &Vars) -> Result<Option<Value>> {
    Ok(match arg {
        Arg::Buf { .. } | Arg::State { .. } => None,
        Arg::Var { var } => Some(json!(*vars.get(var).with_context(|| format!("no value for var `{var}`"))?)),
        Arg::Expr { expr } => Some(json!(expr.eval(vars)?)),
        Arg::I32 { i32: v } => Some(json!(v)),
        Arg::I64 { i64: v } => Some(json!(v)),
        Arg::U8 { u8: v } => Some(json!(v)),
        Arg::F32 { f32: v } => Some(json!({"bits":v.to_bits()})),
        Arg::Rank { rank } => Some(json!({"group":rank,"value":0})),
    })
}
