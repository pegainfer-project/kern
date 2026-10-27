//! The pooled tier's decisions: what a prefix is called in the pool, what
//! to ask the pool about a prompt, and when a put is whole.
//!
//! The pool (Mooncake Store, driven by kern-serve) holds three kinds of
//! object, each named by the tokens it holds and nothing else: a *page*,
//! every paged state's bytes for one whole page; a *state*, every
//! per-sequence state's slot after a snapshot's tokens plus the page the
//! snapshot ends inside; and a *mark*, one byte saying every object of a
//! snapshot landed. A page is written by the one member holding it, so
//! its key has no member; a state is written by every member of a group,
//! so its key has one; a mark is written once per snapshot by whoever
//! decided it. A pipeline stage holds its own layers of each, so page and
//! state keys carry the stage.
//!
//! Every key is a [`Keys`] namespace (the manifest, the weights, the cut)
//! over a [`Chain`] digest of the whole pages before the object and the
//! tokens of the page it ends inside. The same tokens under the same
//! namespace are the same key on every instance; a prompt that differs by
//! one token differs in every key from that page on.
//!
//! A [`Probe`] is what to ask about a prompt in one batch: the marks the
//! frontend says the prompt may continue from, newest first, and every
//! page between the device hit and the newest of them, on every stage.
//! The pool is asked whether each exists; [`Probe::chosen`] reads the
//! answer. A [`Landing`] counts a put over a group: every member reports
//! the pages it stored, and the put is whole when all have, at the fewest.
//!
//! Pure: no client, no clock; the same tokens give the same keys and the
//! same answers give the same choice.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use sha2::{Digest, Sha256};

use crate::error::{bail, Result};

pub type Hash = [u8; 32];

fn sha(parts: &[&[u8]]) -> Hash {
    let mut h = Sha256::new();
    for p in parts {
        h.update(p);
    }
    h.finalize().into()
}

fn le(tokens: &[i64]) -> Vec<u8> {
    tokens.iter().flat_map(|t| t.to_le_bytes()).collect()
}

/// The digest of every whole page of a sequence so far, each over the
/// one before it: `at(k)` names the first `k` pages, `at(0)` nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chain {
    unit: usize,
    hashes: Vec<Hash>,
    open: Vec<i64>,
}

impl Chain {
    pub fn new(unit: usize) -> Chain {
        assert!(unit > 0, "a page holds at least one token");
        Chain { unit, hashes: vec![[0; 32]], open: Vec::new() }
    }

    /// The chain over `tokens` from the start.
    pub fn of(unit: usize, tokens: &[i64]) -> Chain {
        let mut c = Chain::new(unit);
        c.extend(tokens);
        c
    }

    /// Append tokens, sealing every page they complete.
    pub fn extend(&mut self, tokens: &[i64]) {
        for &t in tokens {
            self.open.push(t);
            if self.open.len() == self.unit {
                let last = self.hashes[self.hashes.len() - 1];
                self.hashes.push(sha(&[&last, &le(&self.open)]));
                self.open.clear();
            }
        }
    }

    pub fn unit(&self) -> usize {
        self.unit
    }

    /// Whole pages sealed so far.
    pub fn pages(&self) -> usize {
        self.hashes.len() - 1
    }

    /// The digest of the first `k` pages; `k` at most [`Chain::pages`].
    pub fn at(&self, k: usize) -> Hash {
        self.hashes[k]
    }
}

/// A namespace to name objects in: the keys of one manifest over one set
/// of weights, cut one way. Keys of different namespaces never collide,
/// so a pool shared by every deployment only answers for the same model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Keys {
    ns: Hash,
}

impl Keys {
    /// A namespace over `parts`: whatever identifies the manifest, the
    /// weights and the cut, in a fixed order.
    pub fn new(parts: &[&[u8]]) -> Keys {
        let mut all: Vec<&[u8]> = vec![b"kern-pool/1"];
        all.extend_from_slice(parts);
        Keys { ns: sha(&all) }
    }

    fn key(&self, kind: &[u8], stage: Option<u8>, member: Option<u8>, chain: Hash, tail: &[i64]) -> String {
        let stage = stage.map(|s| vec![s]).unwrap_or_default();
        let member = member.map(|m| vec![m]).unwrap_or_default();
        hex::encode(sha(&[&self.ns, kind, &stage, &member, &chain, &le(tail)]))
    }

    /// The page whose whole pages up to and including it digest to `chain`.
    pub fn page(&self, stage: Option<u8>, chain: Hash) -> String {
        self.key(b"page", stage, None, chain, &[])
    }

    /// Member `member`'s state after the pages in `chain` and `tail`.
    pub fn state(&self, stage: Option<u8>, member: u8, chain: Hash, tail: &[i64]) -> String {
        self.key(b"state", stage, Some(member), chain, tail)
    }

    /// The mark of the snapshot after the pages in `chain` and `tail`.
    pub fn mark(&self, chain: Hash, tail: &[i64]) -> String {
        self.key(b"mark", None, None, chain, tail)
    }
}

/// What to ask the pool about one prompt, in one batch: the marks it may
/// continue from, newest first, and the whole pages between the device
/// hit and the newest of them, on every stage. The answer, one flag per
/// key in [`Probe::keys`] order, is read by [`Probe::chosen`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Probe {
    /// Mark lengths, newest first; empty for a manifest without a state.
    pub marks: Vec<usize>,
    /// Page numbers (1-based, as [`Chain::at`] counts) to ask about.
    pub pages: Range<usize>,
    /// Stages to ask each page about: `None` asks once without a stage.
    pub stages: Option<u8>,
    unit: usize,
}

impl Probe {
    /// What to ask for a prompt of `prompt` tokens with `device_hit` of
    /// them on the device. With a state, the marks are `marks` filtered
    /// to those past the hit and short of the last token (which always
    /// goes through a program), and the pages run up to the newest of
    /// them; without one there are no marks and the pages run up to the
    /// last whole page short of the last token. `None` when nothing is
    /// worth asking.
    pub fn plan(
        stateful: bool,
        unit: usize,
        device_hit: usize,
        marks: &[usize],
        prompt: usize,
        stages: Option<u8>,
    ) -> Option<Probe> {
        let first = device_hit / unit + 1;
        let (marks, last) = match stateful {
            true => {
                let marks: Vec<usize> = marks.iter().copied().filter(|&m| m > device_hit && m < prompt).collect();
                let last = marks.iter().copied().max()? / unit;
                (marks, last)
            }
            false => (Vec::new(), prompt.saturating_sub(1) / unit),
        };
        let pages = first..last + 1;
        (!marks.is_empty() || !pages.is_empty()).then_some(Probe { marks, pages, stages, unit })
    }

    /// The keys to ask about, marks first, then every page on every stage.
    pub fn keys(&self, keys: &Keys, chain: &Chain, tokens: &[i64]) -> Vec<String> {
        let unit = chain.unit();
        let marks = self.marks.iter().map(|&m| keys.mark(chain.at(m / unit), &tokens[m / unit * unit..m]));
        let pages = self.pages.clone().flat_map(|k| {
            let stages: Vec<Option<u8>> = match self.stages {
                Some(n) => (0..n).map(Some).collect(),
                None => vec![None],
            };
            stages.into_iter().map(move |s| (k, s))
        });
        marks.chain(pages.map(|(k, s)| keys.page(s, chain.at(k)))).collect()
    }

    /// The length a prompt can continue from, given one flag per key of
    /// [`Probe::keys`]: with marks, the newest mark every whole page
    /// below which exists on every stage; without, the longest run of
    /// such pages from the first asked. `None` when nothing is usable.
    pub fn chosen(&self, exists: &[bool]) -> Option<usize> {
        let unit = self.unit;
        let per = self.stages.map_or(1, usize::from);
        let (marks, pages) = exists.split_at(self.marks.len());
        let page_ok = |k: usize| -> bool {
            let i = (k - self.pages.start) * per;
            pages.get(i..i + per).is_some_and(|s| s.iter().all(|&b| b))
        };
        let run = self.pages.clone().take_while(|&k| page_ok(k)).count();
        let whole = self.pages.start + run;
        match self.marks.is_empty() {
            true => (run > 0).then_some((whole - 1) * unit),
            false => self.marks.iter().zip(marks).find(|&(&m, &e)| e && m / unit < whole).map(|(&m, _)| m),
        }
    }
}

/// One put across a group: every member names the whole pages it
/// stored, and the put is whole once all have answered, at the fewest
/// pages any of them stored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Landing {
    expect: BTreeSet<usize>,
    heard: BTreeMap<usize, usize>,
}

impl Landing {
    pub fn new(members: impl IntoIterator<Item = usize>) -> Landing {
        Landing { expect: members.into_iter().collect(), heard: BTreeMap::new() }
    }

    /// Member `member` stored `pages` whole pages (0: it failed).
    pub fn heard(mut self, member: usize, pages: usize) -> Result<Landing> {
        if !self.expect.contains(&member) {
            bail!(Api, "landing: member {member} is not among {:?}", self.expect);
        }
        if self.heard.insert(member, pages).is_some() {
            bail!(Api, "landing: member {member} answered twice");
        }
        Ok(self)
    }

    /// Once every member answered: the fewest pages any stored.
    pub fn done(&self) -> Option<usize> {
        (self.heard.len() == self.expect.len()).then(|| self.heard.values().copied().min().unwrap_or(0))
    }
}
