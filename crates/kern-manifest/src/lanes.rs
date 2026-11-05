//! The order of a program's calls over its lanes, and the rules that make
//! running them side by side the same as running them in list order.
//!
//! A call runs on a lane (`lane`, default 0) after the call before it on
//! that lane and after every call listed before it on each lane it names
//! in `after`. Every edge points backwards, so the list order is one serial
//! order of the program; the rules make it equal to every other:
//!
//! - two calls neither of which runs after the other touch no buffer,
//!   state or op scratch in common unless both only read it;
//! - a side lane (any lane but 0) starts after something, and the
//!   program's last call is on lane 0 after every call, so a program
//!   begins and ends on lane 0 like a program without lanes;
//! - a side lane holds no cross-rank work (collectives, peer buffers, the
//!   buffers and states a peer buffer maps), so every rank of an SPMD
//!   manifest meets the others in one order, lane 0's. Another process
//!   writing an exported buffer between programs (a pipeline's arriving
//!   activations) is ordered by the host, not by lanes.
//!
//! Lanes are numbered densely from 0 over the manifest, so a runtime makes
//! exactly the streams the manifest uses. Order is computed with vector
//! clocks: a call's clock counts, per lane, the calls that finish before it
//! starts, itself included on its own lane.

use std::collections::{BTreeMap, BTreeSet};

use crate::types::*;

/// What a call touches beside its scalars: a buffer, a state, or the
/// scratch every call of one op shares.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Resource<'a> {
    Buffer(&'a str),
    State(&'a str),
    Scratch(&'a str),
}

impl std::fmt::Display for Resource<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Resource::Buffer(b) => write!(f, "buffer `{b}`"),
            Resource::State(s) => write!(f, "state `{s}`"),
            Resource::Scratch(o) => write!(f, "the scratch of op `{o}`"),
        }
    }
}

/// Every resource `c` touches and whether it writes it; empty for a call
/// whose op does not resolve (reported elsewhere).
fn touches<'a>(m: &'a Manifest, c: &'a Call) -> Vec<(Resource<'a>, bool)> {
    let Some(op) = m.ops.get(&c.op) else { return Vec::new() };
    let args = c.args.iter().zip(&op.params).filter_map(|(a, p)| {
        let writes = p.dir().is_some_and(|d| d != Dir::In);
        match a {
            Arg::Buf { buf, .. } => Some((Resource::Buffer(buf.as_str()), writes)),
            Arg::State { state, .. } => Some((Resource::State(state.as_str()), writes)),
            _ => None,
        }
    });
    let scratch = (!op.imp.scratch.is_empty()).then_some((Resource::Scratch(c.op.as_str()), true));
    let merged = args.chain(scratch).fold(BTreeMap::new(), |mut acc, (r, w)| {
        *acc.entry(r).or_insert(false) |= w;
        acc
    });
    merged.into_iter().collect()
}

/// Why `c`, on a side lane, reaches another rank; `None` when it does not.
fn cross_rank(m: &Manifest, c: &Call) -> Option<String> {
    let op = m.ops.get(&c.op)?;
    if op.imp.launches.iter().any(|l| l.entry().starts_with("extern:nccl")) {
        return Some(format!("op `{}` is a collective", c.op));
    }
    let mapped: BTreeSet<&str> = m.buffers.values().filter_map(|b| b.of.as_deref()).collect();
    c.args.iter().find_map(|a| match a {
        Arg::Buf { buf, .. } if m.buffers.get(buf).is_some_and(|b| b.kind == BufferKind::Peer) => {
            Some(format!("takes peer buffer `{buf}`"))
        }
        Arg::Buf { buf: name, .. } | Arg::State { state: name, .. } if mapped.contains(name.as_str()) => {
            Some(format!("takes `{name}`, which a peer buffer maps"))
        }
        _ => None,
    })
}

/// The lane rules over every program of `m`, each violation one message.
pub(crate) fn check(m: &Manifest, errs: &mut Vec<String>) {
    let used: BTreeSet<u8> = m.programs.values().flat_map(|p| &p.calls).map(|c| c.lane).chain([0]).collect();
    if let Some(gap) = (0..=*used.last().expect("lane 0 is in")).find(|l| !used.contains(l)) {
        errs.push(format!(
            "lanes are numbered densely from 0: lane {gap} runs no call, but lane {} does",
            used.last().unwrap()
        ));
    }
    for (pname, p) in &m.programs {
        check_program(m, pname, p, errs);
    }
}

fn check_program(m: &Manifest, pname: &str, p: &Program, errs: &mut Vec<String>) {
    if p.calls.iter().all(|c| c.lane == 0 && c.after.is_empty()) {
        return;
    }
    let width = p.calls.iter().map(|c| c.lane as usize + 1).max().unwrap_or(1);
    let ctx = |i: usize| {
        let c = &p.calls[i];
        match &c.label {
            Some(l) => format!("program `{pname}` call #{i} ({l}, lane {})", c.lane),
            None => format!("program `{pname}` call #{i} (lane {})", c.lane),
        }
    };

    // Clocks, and the structural rules as they come.
    let mut clocks: Vec<Vec<u32>> = Vec::with_capacity(p.calls.len());
    let mut last: Vec<Option<usize>> = vec![None; width];
    for (i, c) in p.calls.iter().enumerate() {
        let lane = c.lane as usize;
        let mut seen = BTreeSet::new();
        for &l in &c.after {
            if !seen.insert(l) {
                errs.push(format!("{}: `after` names lane {l} twice", ctx(i)));
            } else if l == c.lane {
                errs.push(format!("{}: `after` names its own lane; calls on one lane already run in order", ctx(i)));
            } else if l != 0 && last.get(l as usize).copied().flatten().is_none() {
                errs.push(format!("{}: `after` names lane {l}, which runs no call before it", ctx(i)));
            }
        }
        if lane != 0 && last[lane].is_none() && c.after.is_empty() {
            errs.push(format!("{}: a side lane's first call names the lanes it runs `after`", ctx(i)));
        }
        let joined = std::iter::once(c.lane)
            .chain(c.after.iter().copied())
            .filter_map(|l| last.get(l as usize).copied().flatten())
            .fold(vec![0u32; width], |acc, j| acc.iter().zip(&clocks[j]).map(|(a, b)| *a.max(b)).collect());
        let mut clock = joined;
        clock[lane] += 1;
        clocks.push(clock);
        last[lane] = Some(i);
        if lane != 0 {
            if let Some(why) = cross_rank(m, c) {
                errs.push(format!("{}: {why}; cross-rank work stays on lane 0", ctx(i)));
            }
        }
    }
    let counts: Vec<u32> = (0..width).map(|l| p.calls.iter().filter(|c| c.lane as usize == l).count() as u32).collect();
    match p.calls.last() {
        Some(c) if c.lane != 0 => {
            errs.push(format!("{}: a program ends on lane 0, after every lane", ctx(p.calls.len() - 1)))
        }
        Some(_) => {
            let end = clocks.last().expect("a last call has a clock");
            for (l, (&n, &seen)) in counts.iter().zip(end).enumerate().filter(|(_, (n, s))| n > s) {
                let open = p.calls.iter().rposition(|c| c.lane as usize == l).expect("the lane has calls");
                errs.push(format!(
                    "program `{pname}`: lane {l} is never joined: {} of its {n} calls, up to {}, finish after the program's last call",
                    n - seen,
                    ctx(open)
                ));
            }
        }
        None => {}
    }

    // Races: for each access, the earlier accesses on other lanes that do
    // not finish before it, a suffix of each lane's list.
    let pos = |i: usize| clocks[i][p.calls[i].lane as usize];
    let mut seen: BTreeMap<Resource, Vec<Vec<(usize, bool)>>> = BTreeMap::new();
    for (i, c) in p.calls.iter().enumerate() {
        for (r, writes) in touches(m, c) {
            let lanes = seen.entry(r).or_insert_with(|| vec![Vec::new(); width]);
            for (l, accesses) in lanes.iter().enumerate().filter(|(l, _)| *l != c.lane as usize) {
                let concurrent = accesses.iter().rev().take_while(|(a, _)| pos(*a) > clocks[i][l]);
                for &(a, w) in concurrent.filter(|(_, w)| *w || writes) {
                    let writer = if w { ctx(a) } else { ctx(i) };
                    errs.push(format!(
                        "{} and {} both touch {r} and neither runs after the other; {writer} writes it",
                        ctx(a),
                        ctx(i)
                    ));
                }
            }
            lanes[c.lane as usize].push((i, writes));
        }
    }
}
