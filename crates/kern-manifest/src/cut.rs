//! Cutting a manifest into a pipeline: stage `s` runs the calls between
//! two cut points of every program a serving loop drives, and what it
//! needs from the stage before it, or leaves for the stage after it, is
//! an input or an output of its own like any other.
//!
//! A forward is a pure function of its inputs and its states, so a cut
//! changes no arithmetic: the stages run the same calls in the same order
//! on the same values, and the pipeline's outputs are the unsplit
//! manifest's byte for byte. What a cut adds is a signature per stage. At
//! a cut point the buffers live across it are the ones the calls after it
//! read before they write them whole (an `out` at offset 0 is taken to
//! write the whole buffer, as every generator writes a view at its base)
//! and that the calls before it wrote; each must be a workspace, since an
//! output is read from the last stage and a carry would cross programs.
//! On the stage before the cut such a buffer is an `output`; on the stage
//! after it an `input` when that stage only reads it (on its way to a
//! later stage or not), an `inout` when the stage rewrites it in place.
//! A buffer that arrives is exported, so the stage before can map it and
//! copy into it. How the value travels (a memcpy over NVLink, RDMA, a
//! copy within a process) and when (the stage before has finished the
//! item, the stage after is done with the one before it) is the serving
//! shell's, which wires the stages by name: every arriving buffer of a
//! stage is one its predecessor holds under the same name. A value a
//! stage neither reads nor writes on its way to a later stage is
//! refused: a stage carries only what it uses.
//!
//! A state never travels. Every region of a state (a state and a call's
//! offset into it) belongs to the one stage whose calls touch it, which is
//! what makes the stages' page tables interchangeable: a sequence's pages
//! hold the first stage's layers on one GPU and the next stage's on
//! another, under the same page ids. A region two stages touch is refused.
//! A program run once after load, and one deriving weights while they
//! load, is sliced per stage to the calls the stage's own calls need, so
//! every stage derives its own tables and weights.
//!
//! The prompt reaches its first token the way it does unsplit, from the
//! step or from a chunk program that hands one back; either way the last
//! stage writes it, since only the last stage writes an output. Everything
//! a stage does not use is dropped, and every stage verifies. A state is
//! kept whole or not at all, so a state laid out over the whole model (one
//! KV state for every layer) is carried whole by every stage that touches
//! a piece of it; [`state_use`] says how much of each one a stage uses,
//! and a generator that gives each piece its own state (one per layer)
//! lets a stage carry only its own. A table whose state a stage dropped
//! indexes a kept state of the same kind: a pool hands every paged state
//! the same ids.

use std::collections::{BTreeMap, BTreeSet};

use crate::types::*;
use crate::verify::{verify, Verified};

/// Every reason a manifest cannot be cut where asked, reported together.
#[derive(Debug)]
pub struct CutErrors(pub Vec<String>);

impl std::fmt::Display for CutErrors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("manifest cannot be cut:")?;
        for e in &self.0 {
            write!(f, "\n  - {e}")?;
        }
        Ok(())
    }
}

impl std::error::Error for CutErrors {}

/// Cut `m` before each call index in `at` of every program with a
/// `batch` (all of them must be longer than the last point). Returns the
/// stages in order, each verified.
pub fn cut(m: &Verified, at: &[usize]) -> Result<Vec<Verified>, CutErrors> {
    let mut errs = Vec::new();
    if m.topology.is_some() {
        errs.push("the manifest has a topology; a cut takes a single-rank manifest".to_string());
    }
    if m.cut.is_some() {
        errs.push("the manifest is already a stage of a cut".to_string());
    }
    if at.is_empty() || at[0] == 0 || at.windows(2).any(|w| w[0] >= w[1]) {
        errs.push(format!("cut points {at:?}: at least one, increasing, none at 0"));
    }
    let driven: Vec<(&String, &Program)> = m.programs.iter().filter(|(_, p)| p.batch.is_some()).collect();
    if driven.is_empty() {
        errs.push("no program declares a `batch`: nothing to cut".to_string());
    }
    for (name, p) in &driven {
        if let Some(&last) = at.last() {
            if last >= p.calls.len() {
                errs.push(format!("program `{name}` has {} calls; it cannot be cut before call {last}", p.calls.len()));
            }
        }
    }
    if !errs.is_empty() {
        return Err(CutErrors(errs));
    }

    let stages = at.len() + 1;
    let bounds: Vec<usize> = std::iter::once(0).chain(at.iter().copied()).collect();
    let span = |p: &Program, s: usize| bounds[s]..bounds.get(s + 1).copied().unwrap_or(p.calls.len());

    // What crosses each edge: live after it, written before it.
    let mut crossing: Vec<BTreeSet<&str>> = vec![BTreeSet::new(); at.len()];
    for (name, p) in &driven {
        for (e, &a) in at.iter().enumerate() {
            let before = writes(m, &p.calls[..a]);
            for b in live_in(m, &p.calls[a..]).into_iter().filter(|b| before.contains(b)) {
                if m.buffers[b].kind != BufferKind::Workspace {
                    errs.push(format!(
                        "program `{name}`: {} buffer `{b}` is live across the cut before call {a}; only a workspace travels",
                        m.buffers[b].kind
                    ));
                }
                crossing[e].insert(b);
            }
        }
        for (i, c) in p.calls.iter().enumerate().take(bounds[stages - 1]) {
            for (b, _, dir) in accesses(m, c) {
                if dir != Dir::In && matches!(m.buffers[b].kind, BufferKind::Output | BufferKind::Carry) {
                    errs.push(format!(
                        "program `{name}` call {i} writes {} buffer `{b}` before the last stage; a pipeline's outputs come from its last stage",
                        m.buffers[b].kind
                    ));
                }
            }
        }
    }
    // Each stage's signature: what arrives, what it writes, what leaves.
    let mut kinds: Vec<BTreeMap<&str, BufferKind>> = Vec::with_capacity(stages);
    for s in 0..stages {
        let arriving = if s > 0 { crossing[s - 1].clone() } else { BTreeSet::new() };
        let leaving = crossing.get(s).cloned().unwrap_or_default();
        let (mut touched, mut written) = (BTreeSet::new(), BTreeSet::new());
        for (_, p) in &driven {
            let calls = &p.calls[span(p, s)];
            touched.extend(calls.iter().flat_map(|c| accesses(m, c)).map(|(b, ..)| b));
            written.extend(writes(m, calls));
        }
        let mut k = BTreeMap::new();
        for &b in arriving.union(&leaving) {
            k.insert(
                b,
                match (arriving.contains(b), written.contains(b)) {
                    (true, true) => BufferKind::Inout,
                    (true, false) => BufferKind::Input,
                    (false, _) => BufferKind::Output,
                },
            );
            if !touched.contains(b) {
                errs.push(format!("stage {s}: `{b}` passes through untouched; a stage carries only what it uses"));
            }
        }
        kinds.push(k);
    }
    let mut owner: BTreeMap<(String, u64), usize> = BTreeMap::new();
    for (name, p) in &driven {
        for s in 0..stages {
            for c in &p.calls[span(p, s)] {
                for (st, off) in state_args(c) {
                    match owner.insert((st.to_string(), off), s) {
                        Some(o) if o != s => errs.push(format!(
                            "program `{name}`: state `{st}`+{off} is touched by stages {o} and {s}; a state region belongs to one stage"
                        )),
                        _ => {}
                    }
                }
            }
        }
    }
    if !errs.is_empty() {
        return Err(CutErrors(errs));
    }

    let id = id_of(m, at);
    (0..stages)
        .map(|s| {
            let programs = driven
                .iter()
                .map(|(name, p)| ((*name).clone(), Program { calls: p.calls[span(p, s)].to_vec(), ..(*p).clone() }))
                .collect();
            let st = stage(m, programs, &kinds[s], s, stages, &id);
            verify(st).map_err(|e| CutErrors(e.0.into_iter().map(|d| format!("stage {s}: {d}")).collect()))
        })
        .collect()
}

/// Stage `s` of `stages`: `programs` (already cut), the once and derive
/// programs sliced to what they need, and every declaration they use,
/// the buffers crossing its edges re-declared as `kinds` says.
fn stage(
    m: &Manifest,
    mut programs: BTreeMap<String, Program>,
    kinds: &BTreeMap<&str, BufferKind>,
    s: usize,
    stages: usize,
    id: &str,
) -> Manifest {
    let mut needed: BTreeSet<String> =
        programs.values().flat_map(|p| &p.calls).flat_map(|c| reads(m, c)).map(str::to_string).collect();
    for (name, p) in m.programs.iter().filter(|(_, p)| p.once || p.derive) {
        let mut kept = Vec::new();
        for c in p.calls.iter().rev() {
            if accesses(m, c).iter().any(|(b, _, d)| *d != Dir::In && needed.contains(*b)) {
                needed.extend(reads(m, c).map(str::to_string));
                kept.push(c.clone());
            }
        }
        if !kept.is_empty() {
            kept.reverse();
            programs.insert(name.clone(), Program { calls: kept, ..p.clone() });
        }
    }

    let calls: Vec<&Call> = programs.values().flat_map(|p| &p.calls).collect();
    let mut buffers: BTreeMap<String, Buffer> = BTreeMap::new();
    for c in &calls {
        for a in &c.args {
            if let Arg::Buf { buf, .. } = a {
                if let Some(b) = m.buffers.get(buf) {
                    buffers.insert(buf.clone(), b.clone());
                }
            }
        }
    }
    for (name, &kind) in kinds {
        let b = buffers.get_mut(*name).expect("a crossing buffer is one the stage's calls touch");
        b.kind = kind;
        b.export = kind != BufferKind::Output;
    }
    let states: BTreeMap<String, State> =
        calls.iter().flat_map(|c| state_args(c)).map(|(st, _)| (st.to_string(), m.states[st].clone())).collect();
    let names: BTreeSet<String> = buffers.keys().chain(states.keys()).cloned().collect();
    // A table hands the same ids to every state of its kind: one naming a
    // state this stage dropped names a kept state of that kind instead.
    let kind = |s: &State| (s.bytes_per_token > 0, s.bytes_per_seq > 0, s.is_owned());
    for b in buffers.values_mut() {
        let Some(t) = b.domain.as_ref().and_then(|d| d.index_into.clone()).filter(|t| !names.contains(t)) else {
            continue;
        };
        let to = m.states.get(&t).and_then(|old| states.iter().find(|(_, s)| kind(s) == kind(old))).map(|(n, _)| n);
        match (to, b.domain.as_mut()) {
            (Some(n), Some(d)) => d.index_into = Some(n.clone()),
            _ => b.domain = None,
        }
    }
    let ops: BTreeMap<String, Op> = calls.iter().map(|c| (c.op.clone(), m.ops[&c.op].clone())).collect();
    let modules: BTreeMap<String, Module> = ops
        .values()
        .flat_map(|o| &o.imp.launches)
        .filter_map(Launch::module)
        .map(|n| (n.to_string(), m.modules[n].clone()))
        .collect();
    // A var stays when some kept declaration names it: shapes, grids,
    // call args, batches and domains all spell a var as a bare string.
    let mut strings = BTreeSet::new();
    for v in [serde_json::to_value(&buffers), serde_json::to_value(&ops), serde_json::to_value(&programs)] {
        collect_strings(&v.expect("serializable"), &mut strings);
    }
    let vars = m.vars.iter().filter(|(k, _)| strings.contains(*k)).map(|(k, v)| (k.clone(), v.clone())).collect();
    Manifest {
        schema_version: m.schema_version,
        model: m.model.clone(),
        topology: None,
        cut: Some(Cut { id: id.to_string(), stage: s as u64, stages: stages as u64 }),
        vars,
        states,
        buffers,
        modules,
        ops,
        programs,
    }
}

fn collect_strings(v: &serde_json::Value, out: &mut BTreeSet<String>) {
    match v {
        serde_json::Value::String(s) => {
            out.insert(s.clone());
        }
        serde_json::Value::Array(a) => a.iter().for_each(|x| collect_strings(x, out)),
        serde_json::Value::Object(o) => o.values().for_each(|x| collect_strings(x, out)),
        _ => {}
    }
}

/// Every buffer argument of `c`: name, offset and the direction its op
/// declares.
fn accesses<'a>(m: &'a Manifest, c: &'a Call) -> Vec<(&'a str, u64, Dir)> {
    let params = m.ops.get(&c.op).map(|o| o.params.as_slice()).unwrap_or(&[]);
    c.args
        .iter()
        .zip(params)
        .filter_map(|(a, p)| match a {
            Arg::Buf { buf, offset } => Some((buf.as_str(), *offset, p.dir().unwrap_or(Dir::In))),
            _ => None,
        })
        .collect()
}

/// The buffers `c` reads, by name.
fn reads<'a>(m: &'a Manifest, c: &'a Call) -> impl Iterator<Item = &'a str> {
    accesses(m, c).into_iter().filter(|(_, _, d)| *d != Dir::Out).map(|(b, _, _)| b)
}

/// The buffers `calls` write.
fn writes<'a>(m: &'a Manifest, calls: &'a [Call]) -> BTreeSet<&'a str> {
    calls.iter().flat_map(|c| accesses(m, c)).filter(|(_, _, d)| *d != Dir::In).map(|(b, _, _)| b).collect()
}

/// The buffers `calls` read before writing them whole: what must already
/// hold its value when they start.
fn live_in<'a>(m: &'a Manifest, calls: &'a [Call]) -> BTreeSet<&'a str> {
    let mut killed = BTreeSet::new();
    let mut live = BTreeSet::new();
    for c in calls {
        let acc = accesses(m, c);
        for &(b, _, d) in &acc {
            if d != Dir::Out && !killed.contains(b) {
                live.insert(b);
            }
        }
        for &(b, off, d) in &acc {
            if d == Dir::Out && off == 0 {
                killed.insert(b);
            }
        }
    }
    live
}

/// How much of each state a stage keeps it uses: per state, the regions
/// (distinct call offsets) the stage's calls touch and those the whole
/// manifest's calls touch. A stage allocates a state whole, so a share
/// below one is memory it holds for another stage's regions: a state laid
/// out over the whole model rather than one per piece.
pub fn state_use(whole: &Manifest, stage: &Manifest) -> Vec<(String, usize, usize)> {
    let regions = |m: &Manifest, st: &str| -> BTreeSet<u64> {
        m.programs
            .values()
            .flat_map(|p| &p.calls)
            .flat_map(state_args)
            .filter(|(s, _)| *s == st)
            .map(|(_, o)| o)
            .collect()
    };
    stage.states.keys().map(|st| (st.clone(), regions(stage, st).len(), regions(whole, st).len())).collect()
}

/// Every state region `c` touches.
fn state_args(c: &Call) -> impl Iterator<Item = (&str, u64)> {
    c.args.iter().filter_map(|a| match a {
        Arg::State { state, offset } => Some((state.as_str(), *offset)),
        _ => None,
    })
}

/// The cut's name: FNV-1a over the manifest and the cut points, the same
/// bytes for the same cut.
fn id_of(m: &Manifest, at: &[usize]) -> String {
    let bytes = m.to_json().into_bytes().into_iter().chain(at.iter().flat_map(|a| (*a as u64).to_le_bytes()));
    let h = bytes.fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3));
    format!("{h:016x}")
}
