//! Cutting a manifest into a pipeline: stage `s` runs the calls between
//! two cut points of every program a serving loop drives, and the
//! activations live across a cut travel through a mailbox.
//!
//! A forward is a pure function of its inputs and its states, so a cut
//! changes no arithmetic: the stages run the same calls in the same order
//! on the same values, and the pipeline's outputs are the unsplit
//! manifest's byte for byte. What a cut adds is transport. At a cut point
//! the buffers live across it are the ones the calls after it read before
//! they write them whole (an `out` at offset 0 is taken to write the whole
//! buffer, as every generator writes a view at its base) and that the
//! calls before it wrote; each must be a workspace, since an output is read
//! from the last stage and a carry would cross programs. The upstream stage
//! ends every program by putting them into a box on the downstream GPU
//! (`tools/kernels-src/pp_mailbox.cu`): wait until the box is empty, put,
//! post. The downstream stage starts by waiting for the post, taking them
//! out and freeing the box. One box per edge, laid out per program at the
//! bounds of its vars; the box sits on the downstream GPU and the upstream
//! reaches it through a `peer` buffer of a two-member topology group
//! `pp.<edge>`, its own copy of the box unused.
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
//! stage writes it, since only the last stage writes an output. Everything a stage
//! does not use is dropped, and every stage verifies. A state is kept whole
//! or not at all, so a state laid out over the whole model (one KV state
//! for every layer) is carried whole by every stage that touches a piece of
//! it; [`state_use`] says how much of each one a stage uses, and a
//! generator that gives each piece its own state (one per layer) lets a
//! stage carry only its own. A table whose state a stage dropped indexes a
//! kept state of the same kind: a pool hands every paged state the same ids.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::json;

use crate::types::*;
use crate::verify::{verify, Verified};

/// How long a mailbox wait spins before the stage gives up and traps.
pub const TIMEOUT_NS: i64 = 60_000_000_000;
/// Bytes of a box before its data: the two counters, a cache line each.
const HEADER: u64 = 256;
/// Each carried buffer starts at a multiple of this in the box.
const ALIGN: u64 = 256;
/// Entries of a stage's clock.
const RING: u64 = 256;
/// The module name the pipeline kernels go under in every stage.
const MODULE: &str = "pp";
/// The stage's clock (fill `clock`).
const CLOCK: &str = "pp.clock";

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

/// A buffer an edge carries in one program: where in the box, and how
/// many bytes per call.
#[derive(Debug, Clone)]
struct Carried {
    buffer: String,
    dtype: DType,
    offset: u64,
    bytes: Expr,
}

/// Cut `m` before each call index in `at` of every program with a
/// `batch` (all of them must be longer than the last point), wiring in
/// the mailbox kernels of `pp`. Returns the stages in order, each
/// verified.
pub fn cut(m: &Verified, at: &[usize], pp: &Module) -> Result<Vec<Verified>, CutErrors> {
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
    let max: BTreeMap<String, u64> = m.vars.iter().map(|(k, v)| (k.clone(), v.max)).collect();

    // What each edge carries in each program, and each box's size.
    let mut carried: BTreeMap<(&str, usize), Vec<Carried>> = BTreeMap::new();
    for (name, p) in &driven {
        for (e, &a) in at.iter().enumerate() {
            let before = writes(m, &p.calls[..a]);
            let mut offset = 0;
            let mut list = Vec::new();
            for b in live_in(m, &p.calls[a..]).into_iter().filter(|b| before.contains(b)) {
                let buf = &m.buffers[b];
                if buf.kind != BufferKind::Workspace {
                    errs.push(format!(
                        "program `{name}`: {} buffer `{b}` is live across the cut before call {a}; only a workspace travels",
                        buf.kind
                    ));
                    continue;
                }
                match bytes_of(buf) {
                    Some(bytes) => {
                        let at_max = bytes.eval(&max).unwrap_or(u64::MAX);
                        list.push(Carried { buffer: b.to_string(), dtype: buf.dtype, offset, bytes });
                        offset += at_max.div_ceil(ALIGN) * ALIGN;
                    }
                    None => errs.push(format!(
                        "program `{name}`: buffer `{b}` is shaped {:?}; a carried buffer has at most one var dim",
                        buf.shape
                    )),
                }
            }
            carried.insert((name.as_str(), e), list);
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
    let boxes: Vec<u64> = (0..at.len())
        .map(|e| {
            let data = carried
                .iter()
                .filter(|((_, ee), _)| *ee == e)
                .map(|(_, list)| list.last().map_or(0, |c| c.offset + c.bytes.eval(&max).unwrap_or(0)))
                .max()
                .unwrap_or(0);
            HEADER + data.max(1).div_ceil(ALIGN) * ALIGN
        })
        .collect();

    let id = id_of(m, at);
    let built: Vec<Manifest> = (0..stages)
        .map(|s| {
            let programs: BTreeMap<String, Program> = driven
                .iter()
                .map(|(name, p)| {
                    let into = (s > 0).then(|| &carried[&(name.as_str(), s - 1)][..]);
                    let out = (s + 1 < stages).then(|| &carried[&(name.as_str(), s)][..]);
                    let calls =
                        receive(s, into).into_iter().chain(p.calls[span(p, s)].iter().cloned()).chain(send(s, out));
                    ((*name).clone(), Program { calls: calls.collect(), ..(*p).clone() })
                })
                .collect();
            stage(m, programs, s, stages, &boxes, &id, pp)
        })
        .collect();
    built
        .into_iter()
        .enumerate()
        .map(|(s, st)| verify(st).map_err(|e| CutErrors(e.0.into_iter().map(|d| format!("stage {s}: {d}")).collect())))
        .collect()
}

fn box_name(e: usize) -> String {
    format!("pp.{e}.box")
}

fn peer_name(e: usize) -> String {
    format!("pp.{e}.peer")
}

fn call(op: &str, args: serde_json::Value) -> Call {
    serde_json::from_value(json!({"label": op, "op": op, "args": args})).expect("a well-formed call")
}

/// The calls that start stage `s`: take what the edge into it carries, or
/// stamp the start of a first stage.
fn receive(s: usize, into: Option<&[Carried]>) -> Vec<Call> {
    let Some(list) = into else {
        return vec![call("pp.stamp_first", json!([{"buf": peer_name(s)}, {"buf": CLOCK}]))];
    };
    let b = box_name(s - 1);
    std::iter::once(call("pp.wait_full", json!([{"buf": b}, {"buf": CLOCK}])))
        .chain(list.iter().map(|c| {
            call(
                &format!("pp.take.{}", c.dtype),
                json!([{"buf": b}, {"buf": c.buffer}, {"expr": c.bytes}, {"i64": c.offset}]),
            )
        }))
        .chain(std::iter::once(call("pp.post_empty", json!([{"buf": b}]))))
        .collect()
}

/// The calls that end stage `s`: put what the edge out of it carries, or
/// stamp the end of a last stage.
fn send(s: usize, out: Option<&[Carried]>) -> Vec<Call> {
    let Some(list) = out else {
        return vec![call("pp.stamp_last", json!([{"buf": box_name(s - 1)}, {"buf": CLOCK}]))];
    };
    let p = peer_name(s);
    std::iter::once(call("pp.wait_empty", json!([{"buf": p}, {"buf": CLOCK}])))
        .chain(list.iter().map(|c| {
            call(
                &format!("pp.put.{}", c.dtype),
                json!([{"buf": p}, {"buf": c.buffer}, {"expr": c.bytes}, {"i64": c.offset}]),
            )
        }))
        .chain(std::iter::once(call("pp.post_full", json!([{"buf": p}]))))
        .collect()
}

/// The pipeline ops a stage's calls name: `pp.put.<dtype>` and
/// `pp.take.<dtype>` per carried dtype, the waits, posts and stamps.
fn pp_op(name: &str) -> Op {
    let single = |entry: &str, params: serde_json::Value| {
        json!({"params": params, "impl": {"launches": [
            {"module": MODULE, "entry": entry, "block": [1, 1, 1], "grid": [1, 1, 1]}]}})
    };
    let wait = |entry: &str, first: &str| {
        json!({"params": [first, "out buffer<i64>"], "impl": {"launches": [
            {"module": MODULE, "entry": entry, "block": [1, 1, 1], "grid": [1, 1, 1],
             "params": [first, "out buffer<i64>", "i64"],
             "args": [{"param": 0}, {"param": 1}, {"i64": TIMEOUT_NS}]}]}})
    };
    let copy = |entry: &str, params: serde_json::Value| {
        json!({"params": params, "impl": {"launches": [
            {"module": MODULE, "entry": entry, "block": [256, 1, 1], "grid": [264, 1, 1]}]}})
    };
    let v = match name.split('.').collect::<Vec<_>>().as_slice() {
        ["pp", "wait_empty"] => wait("kern_pp_wait_empty", "in buffer<u64>"),
        ["pp", "wait_full"] => wait("kern_pp_wait_full", "in buffer<u8>"),
        ["pp", "post_full"] => single("kern_pp_post_full", json!(["in buffer<u64>"])),
        ["pp", "post_empty"] => single("kern_pp_post_empty", json!(["inout buffer<u8>"])),
        ["pp", "stamp_first"] => single("kern_pp_stamp_first", json!(["in buffer<u64>", "out buffer<i64>"])),
        ["pp", "stamp_last"] => single("kern_pp_stamp_last", json!(["in buffer<u8>", "out buffer<i64>"])),
        ["pp", "put", dt] => copy("kern_pp_put", json!(["in buffer<u64>", format!("in buffer<{dt}>"), "i64", "i64"])),
        ["pp", "take", dt] => copy("kern_pp_take", json!(["in buffer<u8>", format!("out buffer<{dt}>"), "i64", "i64"])),
        _ => unreachable!("the cut names only the pipeline ops above"),
    };
    serde_json::from_value(v).expect("a well-formed pipeline op")
}

/// Stage `s` of `stages`: `programs` (already cut and wired), the once
/// and derive programs sliced to what they need, and every declaration
/// they use.
fn stage(
    m: &Manifest,
    mut programs: BTreeMap<String, Program>,
    s: usize,
    stages: usize,
    boxes: &[u64],
    id: &str,
    pp: &Module,
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

    let mut buffers: BTreeMap<String, Buffer> = BTreeMap::new();
    let u8_box = |bytes: u64, kind: &str| -> Buffer {
        serde_json::from_value(json!({"dtype": "u8", "shape": [bytes], "kind": kind, "export": true})).expect("a box")
    };
    buffers.insert(
        CLOCK.to_string(),
        serde_json::from_value(json!({"dtype": "i64", "shape": [RING * 5], "kind": "output", "fill": "clock"}))
            .expect("a clock"),
    );
    if s > 0 {
        buffers.insert(box_name(s - 1), u8_box(boxes[s - 1], "carry"));
    }
    let mut topology = None;
    if s + 1 < stages {
        buffers.insert(box_name(s), u8_box(boxes[s], "workspace"));
        buffers.insert(
            peer_name(s),
            serde_json::from_value(
                json!({"dtype": "u64", "shape": [2], "kind": "peer", "of": box_name(s), "group": format!("pp.{s}")}),
            )
            .expect("a peer"),
        );
        topology = Some(Topology { groups: BTreeMap::from([(format!("pp.{s}"), 2)]) });
    }
    let calls: Vec<&Call> = programs.values().flat_map(|p| &p.calls).collect();
    for c in &calls {
        for a in &c.args {
            if let Arg::Buf { buf, .. } = a {
                if let Some(b) = m.buffers.get(buf) {
                    buffers.insert(buf.clone(), b.clone());
                }
            }
        }
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
    let ops: BTreeMap<String, Op> =
        calls.iter().map(|c| (c.op.clone(), m.ops.get(&c.op).cloned().unwrap_or_else(|| pp_op(&c.op)))).collect();
    let mut modules: BTreeMap<String, Module> = ops
        .values()
        .flat_map(|o| &o.imp.launches)
        .filter_map(Launch::module)
        .filter(|n| *n != MODULE)
        .map(|n| (n.to_string(), m.modules[n].clone()))
        .collect();
    modules.insert(MODULE.to_string(), pp.clone());
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
        topology,
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

/// Bytes one call moves of `b`: its size at the call's vars, when at most
/// one dim is a var.
fn bytes_of(b: &Buffer) -> Option<Expr> {
    let constant: u64 = b.shape.iter().filter_map(|d| if let Dim::Const(c) = d { Some(*c) } else { None }).product();
    let vars: Vec<&String> = b.shape.iter().filter_map(|d| if let Dim::Var(v) = d { Some(v) } else { None }).collect();
    let c = constant * b.dtype.bytes();
    match vars.as_slice() {
        [] => Some(Expr::Const(c)),
        [v] => Some(Expr::Mul { mul: (Box::new(Expr::Var((*v).clone())), c) }),
        _ => None,
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

/// The buffers `c` reads, by name; a pipeline call reads its buffers too.
fn reads<'a>(m: &'a Manifest, c: &'a Call) -> impl Iterator<Item = &'a str> {
    let known = m.ops.contains_key(&c.op);
    let acc = accesses(m, c);
    let pipeline: Vec<&str> = if known {
        Vec::new()
    } else {
        c.args.iter().filter_map(|a| if let Arg::Buf { buf, .. } = a { Some(buf.as_str()) } else { None }).collect()
    };
    acc.into_iter().filter(|(_, _, d)| *d != Dir::Out).map(|(b, _, _)| b).chain(pipeline)
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
