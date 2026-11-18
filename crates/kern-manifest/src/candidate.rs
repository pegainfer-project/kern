//! Candidates: other implementations of ops a manifest already has.
//!
//! An op is an interface and an implementation, and a call binds only the
//! interface, so an implementation can be replaced without touching a
//! call. A candidate is that replacement as a file: the `modules` and
//! `ops` sections of a manifest and nothing else, each op with the
//! `params` of the op it stands in for. A kernel written in another
//! language, the same kernel at another launch geometry and the same
//! geometry over another cubin are all the same thing here.
//!
//! [`splice`] puts a candidate into a manifest. The interface is what is
//! checked: an op the manifest lacks, or one whose params differ, is not
//! another implementation of it. Modules travel with the launches that
//! name them; a module the manifest already has under the same name with
//! other bytes is renamed after its hash, and modules nothing launches any
//! more are dropped, so the result verifies as a manifest of its own.
//!
//! [`tile`] is the other direction of the same idea: one implementation
//! assembled from several, each running in its own range of a var. A
//! manifest has no control flow, so "this kernel up to 32 rows, that one
//! above" is every launch of the first under `when` up to 32 and every
//! launch of the second under `when` from 33. A launch that already runs
//! only in a range of that var keeps the part of it inside its piece; one
//! conditioned on another var cannot be narrowed and is refused. What
//! `tile` returns is a candidate again, so a choice between candidates is
//! itself one.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::types::*;
use crate::verify::{verify, Verified};

/// Implementations for some of a manifest's ops, with the modules they launch.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Candidate {
    /// The modules these implementations launch; a launch may also name one the manifest pins.
    #[serde(default)]
    pub modules: BTreeMap<String, Module>,
    /// The ops, each under the name and with the params of the op it replaces.
    pub ops: BTreeMap<String, Op>,
}

/// Every reason a candidate does not fit, reported together.
#[derive(Debug)]
pub struct CandidateErrors(pub Vec<String>);

impl std::fmt::Display for CandidateErrors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("candidate does not fit:")?;
        for e in &self.0 {
            write!(f, "\n  - {e}")?;
        }
        Ok(())
    }
}

impl std::error::Error for CandidateErrors {}

/// One piece of a tiled implementation: `from`'s implementation of the op,
/// running while the var is in `min..=max`; an end left out is open.
#[derive(Debug, Clone, Copy)]
pub struct Piece<'a> {
    pub from: &'a Candidate,
    pub min: Option<u64>,
    pub max: Option<u64>,
}

impl Candidate {
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("a candidate serializes")
    }

    /// The manifest's own implementation of `ops`, as a candidate. An op
    /// the manifest lacks is left out.
    pub fn of<'a>(m: &Manifest, ops: impl IntoIterator<Item = &'a str>) -> Candidate {
        Candidate { modules: m.modules.clone(), ops: m.ops.clone() }.only(ops)
    }

    /// The candidate's implementation of `ops` alone, with the modules
    /// those launch. An op it lacks is left out.
    pub fn only<'a>(&self, ops: impl IntoIterator<Item = &'a str>) -> Candidate {
        let ops: BTreeMap<String, Op> =
            ops.into_iter().filter_map(|n| Some((n.to_string(), self.ops.get(n)?.clone()))).collect();
        let modules =
            launched(ops.values()).into_iter().filter_map(|n| Some((n.clone(), self.modules.get(&n)?.clone())));
        Candidate { modules: modules.collect(), ops }
    }

    /// The candidates as one. An op two of them implement is refused; a
    /// module name two of them give to different bytes stays with the
    /// first and is renamed after its hash in the others.
    pub fn join<'a>(parts: impl IntoIterator<Item = &'a Candidate>) -> Result<Candidate, CandidateErrors> {
        let mut all = Candidate::default();
        let mut errs = Vec::new();
        for c in parts {
            let names = adopt(&mut all.modules, &c.modules);
            for (name, op) in &c.ops {
                match all.ops.contains_key(name) {
                    true => errs.push(format!("op `{name}` is implemented twice")),
                    false => drop(all.ops.insert(name.clone(), renamed(op, &names))),
                }
            }
        }
        if errs.is_empty() {
            Ok(all)
        } else {
            Err(CandidateErrors(errs))
        }
    }
}

/// The module names `ops` launch.
fn launched<'a>(ops: impl IntoIterator<Item = &'a Op>) -> BTreeSet<String> {
    ops.into_iter().flat_map(|op| &op.imp.launches).filter_map(|l| l.module().map(str::to_string)).collect()
}

/// `more` taken into `into`: each module under its own name, or under
/// `<name>@<hash prefix>` when `into` has that name for other bytes. The
/// map says what each of `more`'s names became.
fn adopt(into: &mut BTreeMap<String, Module>, more: &BTreeMap<String, Module>) -> BTreeMap<String, String> {
    more.iter()
        .map(|(name, module)| {
            let taken = into.get(name).is_some_and(|m| m.sha256 != module.sha256);
            let hash: String = module.sha256.chars().take(8).collect();
            let as_name = if taken { format!("{name}@{hash}") } else { name.clone() };
            into.entry(as_name.clone()).or_insert_with(|| module.clone());
            (name.clone(), as_name)
        })
        .collect()
}

/// `op` with its launches' modules renamed as `names` says.
fn renamed(op: &Op, names: &BTreeMap<String, String>) -> Op {
    let mut op = op.clone();
    for l in &mut op.imp.launches {
        if let Launch::Kernel(k) = l {
            if let Some(n) = names.get(&k.module) {
                k.module = n.clone();
            }
        }
    }
    op
}

/// `m` with every op of `c` running `c`'s implementation, verified.
pub fn splice(m: &Verified, c: &Candidate) -> Result<Verified, CandidateErrors> {
    let mut errs = Vec::new();
    if c.ops.is_empty() {
        errs.push("no ops".to_string());
    }
    for (name, op) in &c.ops {
        match m.ops.get(name) {
            None => errs.push(format!("op `{name}`: the manifest has no such op")),
            Some(have) if have.params != op.params => errs.push(format!(
                "op `{name}`: params [{}] are not the manifest's [{}]",
                list(&op.params),
                list(&have.params)
            )),
            Some(_) => {}
        }
        for module in launched([op]) {
            if !c.modules.contains_key(&module) && !m.modules.contains_key(&module) {
                errs.push(format!("op `{name}`: module `{module}` is in neither the candidate nor the manifest"));
            }
        }
    }
    if !errs.is_empty() {
        return Err(CandidateErrors(errs));
    }
    let mut out: Manifest = (**m).clone();
    let names = adopt(&mut out.modules, &c.modules);
    for (name, op) in &c.ops {
        out.ops.insert(name.clone(), renamed(op, &names));
    }
    let used = launched(out.ops.values());
    out.modules.retain(|name, _| used.contains(name));
    verify(out).map_err(|e| CandidateErrors(e.0))
}

fn list(params: &[ParamType]) -> String {
    params.iter().map(|p| p.to_string()).collect::<Vec<_>>().join(", ")
}

/// `l` narrowed to `min..=max` of `var`: `None` when it never runs there,
/// an error when it is conditioned on another var.
fn narrow(l: &Launch, var: &str, min: Option<u64>, max: Option<u64>) -> Result<Option<Launch>, String> {
    let (lo, hi) = match l.when() {
        Some(w) if w.var != var => {
            return Err(format!("launch `{}` runs in a range of `{}`, not of `{var}`", l.entry(), w.var))
        }
        Some(w) => (w.min.max(min), [w.max, max].into_iter().flatten().min()),
        None => (min, max),
    };
    if matches!((lo, hi), (Some(lo), Some(hi)) if lo > hi) {
        return Ok(None);
    }
    // A lower end of 1 is no bound: a var is at least 1.
    let lo = lo.filter(|&v| v > 1);
    let when = (lo.is_some() || hi.is_some()).then(|| When { var: var.to_string(), min: lo, max: hi });
    let mut l = l.clone();
    match &mut l {
        Launch::Kernel(k) => k.when = when,
        Launch::Extern(e) => e.when = when,
    }
    Ok(Some(l))
}

/// The scratch name in a launch argument or one field of its pack.
fn scratch_names(a: &mut LaunchArg) -> Vec<&mut String> {
    match a {
        LaunchArg::Scratch { scratch } => vec![scratch],
        LaunchArg::Pack { pack } => pack
            .fields
            .iter_mut()
            .filter_map(|f| match &mut f.src {
                FieldSrc::Scratch { scratch } => Some(scratch),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

/// `l` with its scratch arguments renamed as `names` says; the names it is
/// wired to afterwards.
fn rescratch(l: &mut Launch, names: &BTreeMap<String, String>) -> Vec<String> {
    l.own_args_mut()
        .into_iter()
        .flatten()
        .flat_map(scratch_names)
        .map(|s| {
            if let Some(n) = names.get(s) {
                *s = n.clone();
            }
            s.clone()
        })
        .collect()
}

/// One implementation of `op` from `pieces`, each running in its own
/// range of `var`, as a candidate. The ranges are the caller's: pieces
/// that overlap both run there, and a value no piece covers runs nothing.
pub fn tile(op: &str, var: &str, pieces: &[Piece]) -> Result<Candidate, CandidateErrors> {
    let mut errs = Vec::new();
    let mut out = Candidate::default();
    let mut imp = Impl { scratch: BTreeMap::new(), launches: Vec::new() };
    let mut params = None;
    let mut wired = BTreeSet::new();
    for (i, p) in pieces.iter().enumerate() {
        let Some(have) = p.from.ops.get(op) else {
            errs.push(format!("piece {i}: no op `{op}`"));
            continue;
        };
        if params.get_or_insert(&have.params) != &&have.params {
            errs.push(format!("piece {i}: op `{op}` has other params than piece 0"));
        }
        let modules = adopt(&mut out.modules, &p.from.modules);
        let have = renamed(have, &modules);
        let scratch: BTreeMap<String, String> = have
            .imp
            .scratch
            .iter()
            .map(|(name, s)| {
                let clash = imp.scratch.get(name).is_some_and(|t| t.dtype != s.dtype || t.shape != s.shape);
                let as_name = if clash { format!("{name}@{i}") } else { name.clone() };
                imp.scratch.entry(as_name.clone()).or_insert_with(|| s.clone());
                (name.clone(), as_name)
            })
            .collect();
        for l in &have.imp.launches {
            match narrow(l, var, p.min, p.max) {
                Ok(Some(mut l)) => {
                    wired.extend(rescratch(&mut l, &scratch));
                    imp.launches.push(l)
                }
                Ok(None) => {}
                Err(e) => errs.push(format!("piece {i}: op `{op}`: {e}")),
            }
        }
    }
    if errs.is_empty() && imp.launches.is_empty() {
        errs.push(format!("op `{op}`: no launch of any piece runs in its range of `{var}`"));
    }
    let Some(params) = params.filter(|_| errs.is_empty()) else {
        return Err(CandidateErrors(errs));
    };
    let used: BTreeSet<&str> = imp.launches.iter().filter_map(|l| l.module()).collect();
    out.modules.retain(|name, _| used.contains(name.as_str()));
    imp.scratch.retain(|name, _| wired.contains(name));
    out.ops.insert(op.to_string(), Op { params: params.clone(), imp });
    Ok(out)
}
