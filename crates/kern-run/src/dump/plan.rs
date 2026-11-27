//! Call selection and the storage images a dump must preserve. Shapes describe
//! the current invocation; the image always covers the whole allocation, since
//! an interface pointer and its offset do not declare a kernel's access extent.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{ensure, Context, Result};
use clap::ValueEnum;
use kern_manifest::types::{Arg, BufferKind, DType, Dim, ParamType};
use kern_manifest::{Manifest, Protocol, Verified};
use serde::{Deserialize, Serialize};

/// Which side of a selected call to preserve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Before,
    After,
    Both,
}

/// A zero-based invocation of one program and the calls observed within it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Selection {
    pub program: String,
    pub invocation: u64,
    pub labels: Vec<String>,
    pub indices: Vec<usize>,
    pub phase: Phase,
}

/// A checked selection, with each call index once in execution order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Plan {
    pub selection: Selection,
    pub calls: Vec<usize>,
}

/// Buffers and states have separate name spaces and allocation keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StorageKind {
    Buffer,
    State,
}

/// An interface reference into a preserved allocation; its extent is unknown.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Argument {
    pub index: usize,
    pub param: ParamType,
    pub offset: u64,
}

/// One complete allocation, shared by every argument naming it. A state has no
/// element type or shape: those bytes belong to the provider's private layout.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Storage {
    pub name: String,
    pub kind: StorageKind,
    pub bytes: u64,
    pub dtype: Option<DType>,
    pub declared_shape: Option<Vec<Dim>>,
    pub shape: Option<Vec<u64>>,
    pub arguments: Vec<Argument>,
}

impl Plan {
    /// Check the execution boundary before allocating or running anything.
    /// Labels match exactly, including every call with a repeated label.
    pub fn check(m: &Verified, selection: Selection) -> Result<Self> {
        ensure!(
            selection.labels.is_empty() || selection.indices.is_empty(),
            "dump call labels and call indices are mutually exclusive"
        );
        ensure!(
            !selection.labels.is_empty() || !selection.indices.is_empty(),
            "dump needs at least one call label or call index"
        );
        for (name, b) in &m.buffers {
            ensure!(b.kind != BufferKind::Peer, "dump does not support peer buffer `{name}`");
        }
        ensure!(
            m.topology.is_none(),
            "dump supports a standalone single rank without topology; the manifest declares topology"
        );
        for (name, s) in &m.states {
            ensure!(s.host.is_none(), "dump does not support host state `{name}`");
        }
        let program =
            m.programs.get(&selection.program).with_context(|| format!("no program `{}`", selection.program))?;
        ensure!(
            !program.once && !program.derive,
            "dump program `{}` must be a forward, not a once or derive program",
            selection.program
        );
        let protocol = Protocol::check(m)?;
        ensure!(
            protocol.forwards.iter().any(|f| f.name == selection.program),
            "dump program `{}` is not a protocol forward",
            selection.program
        );

        let mut calls = BTreeSet::new();
        for &index in &selection.indices {
            ensure!(
                index < program.calls.len(),
                "dump program `{}`: call index {index} outside 0..{}",
                selection.program,
                program.calls.len()
            );
            calls.insert(index);
        }
        for label in &selection.labels {
            let found: Vec<usize> = program
                .calls
                .iter()
                .enumerate()
                .filter_map(|(i, c)| (c.label.as_deref() == Some(label.as_str())).then_some(i))
                .collect();
            ensure!(
                !found.is_empty(),
                "dump program `{}`: no call has label `{label}` (labels match exactly)",
                selection.program
            );
            calls.extend(found);
        }
        Ok(Self { selection, calls: calls.into_iter().collect() })
    }

    /// Every buffer and state passed to a selected call, in first-argument
    /// order. Allocation keys are `buffer:<name>` and `state:<name>`; repeated
    /// arguments share one image while keeping their distinct offsets.
    pub fn storages(
        &self,
        m: &Manifest,
        call_index: usize,
        vars: &BTreeMap<String, u64>,
        allocations: &BTreeMap<String, u64>,
    ) -> Result<Vec<Storage>> {
        ensure!(self.calls.contains(&call_index), "dump call index {call_index} is not selected");
        let program = m.programs.get(&self.selection.program).context("dump program is no longer in the manifest")?;
        let call = program.calls.get(call_index).context("dump call is no longer in the program")?;
        let op = m.ops.get(&call.op).with_context(|| format!("no op `{}`", call.op))?;
        ensure!(call.args.len() == op.params.len(), "op `{}`: argument and parameter counts differ", call.op);
        let mut storages: Vec<Storage> = Vec::new();
        let mut seen: BTreeMap<(StorageKind, String), usize> = BTreeMap::new();
        for (index, (arg, &param)) in call.args.iter().zip(&op.params).enumerate() {
            let (kind, name, offset) = match arg {
                Arg::Buf { buf, offset } => (StorageKind::Buffer, buf, *offset),
                Arg::State { state, offset } => (StorageKind::State, state, *offset),
                _ => continue,
            };
            let key = match kind {
                StorageKind::Buffer => format!("buffer:{name}"),
                StorageKind::State => format!("state:{name}"),
            };
            let bytes = *allocations.get(&key).with_context(|| format!("dump has no allocation size for `{key}`"))?;
            ensure!(
                offset < bytes,
                "dump op `{}` argument #{index}: offset {offset} outside `{key}` allocation of {bytes} bytes",
                call.op
            );
            let reference = Argument { index, param, offset };
            if let Some(&at) = seen.get(&(kind, name.clone())) {
                let storage: &mut Storage = &mut storages[at];
                storage.arguments.push(reference);
                continue;
            }
            let (dtype, declared_shape, shape) = match kind {
                StorageKind::Buffer => {
                    let buffer = m.buffers.get(name).with_context(|| format!("no buffer `{name}`"))?;
                    let shape: Vec<u64> = buffer
                        .shape
                        .iter()
                        .map(|dim| match dim {
                            Dim::Const(n) => Ok(*n),
                            Dim::Var(var) => {
                                let bound = m.vars.get(var).with_context(|| format!("no var `{var}`"))?;
                                let value = *vars
                                    .get(var)
                                    .with_context(|| format!("dump buffer `{name}`: no value for var `{var}`"))?;
                                ensure!(
                                    (1..=bound.max).contains(&value),
                                    "dump buffer `{name}`: var `{var}` = {value} outside 1..={}",
                                    bound.max
                                );
                                Ok(value)
                            }
                        })
                        .collect::<Result<_>>()?;
                    let shape_bytes = shape.iter().try_fold(buffer.dtype.bytes(), |size, &n| {
                        size.checked_mul(n).context("dump buffer shape overflows its byte count")
                    })?;
                    ensure!(
                        shape_bytes <= bytes,
                        "dump buffer `{name}`: shape needs {shape_bytes} bytes, allocation has {bytes}"
                    );
                    (Some(buffer.dtype), Some(buffer.shape.clone()), Some(shape))
                }
                StorageKind::State => {
                    ensure!(m.states.contains_key(name), "no state `{name}`");
                    (None, None, None)
                }
            };
            seen.insert((kind, name.clone()), storages.len());
            storages.push(Storage {
                name: name.clone(),
                kind,
                bytes,
                dtype,
                declared_shape,
                shape,
                arguments: vec![reference],
            });
        }
        Ok(storages)
    }
}
