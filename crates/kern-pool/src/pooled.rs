//! The pooled tier's decisions: what a prefix is called in the pool, what
//! to ask the pool about a prompt, and when a put is whole.
//!
//! The pool (Mooncake Store, driven by kern-serve) holds three kinds of
//! object, each named by the tokens it holds and nothing else: a *page*,
//! every paged state's bytes for one whole page; a *state*, every
//! per-sequence state's slot after a snapshot's tokens; and a *mark*, one
//! byte saying every object of a snapshot landed. A page is written by the one member holding it, so
//! its key has no member; a state is written by every member of a group,
//! so its key has one; a mark is written once per snapshot by whoever
//! decided it. A pipeline stage holds its own layers of each, so page and
//! state keys carry the stage.
//!
//! Every key is a [`Keys`] namespace (the manifest, the weights, the cut)
//! over a [`Chain`] digest of the whole pages up to the object. A state
//! sits on a page boundary, so its key, like a page's, is the chain at
//! that boundary: the same tokens under the same namespace are the same
//! key on every instance, and a prompt that differs by one token differs
//! in every key from that page on.
//!
//! A [`Probe`] is what to ask about a prompt in one batch: every page
//! between the device hit and the prompt's last token on every stage and,
//! with a state, the mark at every one of those page boundaries. Nobody
//! has to say where a snapshot was taken: whatever any instance left on
//! a boundary the prompt passes is asked for. The pool is asked whether
//! each exists; [`Probe::chosen`] reads the answer.
//!
//! Where a prompt leaves states for the pool alone is [`snapshots`]: on a
//! grid spaced by what a state costs against the pages it covers
//! ([`grid`]), and at the last boundary it reaches, so a cold prompt
//! leaves somewhere to resume from at every stretch of its length and a
//! turn the next one edits leaves its own boundary. [`objects`] names what one member moves for a prefix, each an
//! [`Object`]: a key over the pages and slot holding its bytes on that
//! member; [`landed`] reads a member's flags back as tokens. A [`Landing`]
//! counts a transfer over a group: every member reports the tokens it
//! landed, and it is whole when all have, at the fewest.
//!
//! Pure: no client, no clock; the same tokens give the same keys and the
//! same answers give the same choice.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use serde::{Deserialize, Serialize};
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
        let mut all: Vec<&[u8]> = vec![b"kern-pool/2"];
        all.extend_from_slice(parts);
        Keys { ns: sha(&all) }
    }

    fn key(&self, kind: &[u8], stage: Option<u8>, member: Option<u8>, chain: Hash) -> String {
        let stage = stage.map(|s| vec![s]).unwrap_or_default();
        let member = member.map(|m| vec![m]).unwrap_or_default();
        hex::encode(sha(&[&self.ns, kind, &stage, &member, &chain]))
    }

    /// The page whose whole pages up to and including it digest to `chain`.
    pub fn page(&self, stage: Option<u8>, chain: Hash) -> String {
        self.key(b"page", stage, None, chain)
    }

    /// Member `member`'s state after the pages in `chain`.
    pub fn state(&self, stage: Option<u8>, member: u8, chain: Hash) -> String {
        self.key(b"state", stage, Some(member), chain)
    }

    /// The mark of the snapshot after the pages in `chain`.
    pub fn mark(&self, chain: Hash) -> String {
        self.key(b"mark", None, None, chain)
    }
}

/// What to ask the pool about one prompt, in one batch: the whole pages
/// between the device hit and the last token, on every stage, and with a
/// state the mark at each of their boundaries. The answer, one flag per
/// key in [`Probe::keys`] order, is read by [`Probe::chosen`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Probe {
    /// Page numbers (1-based, as [`Chain::at`] counts) to ask about.
    pub pages: Range<usize>,
    /// Stages to ask each page about: `None` asks once without a stage.
    pub stages: Option<u8>,
    /// Whether a mark is asked at the end of every page.
    pub stateful: bool,
    unit: usize,
}

impl Probe {
    /// What to ask for a prompt of `prompt` tokens with `device_hit` of
    /// them on the device: every page past the hit's whole pages and short
    /// of the last token, which always goes through a program. `None`
    /// when no such page exists.
    pub fn plan(stateful: bool, unit: usize, device_hit: usize, prompt: usize, stages: Option<u8>) -> Option<Probe> {
        let pages = device_hit / unit + 1..prompt.saturating_sub(1) / unit + 1;
        (!pages.is_empty()).then_some(Probe { pages, stages, stateful, unit })
    }

    /// The keys to ask about: with a state the marks, newest first, then
    /// every page on every stage.
    pub fn keys(&self, keys: &Keys, chain: &Chain) -> Vec<String> {
        let marks = self.marks().map(|k| keys.mark(chain.at(k)));
        let per: Vec<Option<u8>> = match self.stages {
            Some(n) => (0..n).map(Some).collect(),
            None => vec![None],
        };
        let pages = self.pages.clone().flat_map(|k| per.iter().map(move |&s| keys.page(s, chain.at(k))));
        marks.chain(pages).collect()
    }

    /// The page boundaries a mark is asked at, newest first.
    fn marks(&self) -> impl Iterator<Item = usize> {
        let asked = if self.stateful { self.pages.clone() } else { 0..0 };
        asked.rev()
    }

    /// The length a prompt can continue from, given one flag per key of
    /// [`Probe::keys`]: with a state, the newest mark every whole page
    /// below which exists on every stage; without, the longest run of
    /// such pages from the first asked. `None` when nothing is usable.
    pub fn chosen(&self, exists: &[bool]) -> Option<usize> {
        let per = self.stages.map_or(1, usize::from);
        let (marks, pages) = exists.split_at(self.marks().count());
        let page_ok = |k: usize| -> bool {
            let i = (k - self.pages.start) * per;
            pages.get(i..i + per).is_some_and(|s| s.iter().all(|&b| b))
        };
        let whole = self.pages.start + self.pages.clone().take_while(|&k| page_ok(k)).count();
        let last = match self.stateful {
            true => self.marks().zip(marks).find(|&(k, &e)| e && k < whole).map(|(k, _)| k),
            false => (whole > self.pages.start).then(|| whole - 1),
        };
        last.map(|k| k * self.unit)
    }
}

/// The spacing of a prompt's pooled snapshots, in tokens: whole pages of
/// `unit` tokens covering twice a state's `slot_bytes` in `token_bytes`
/// per token, so the states a cold prompt leaves cost at most half its
/// pages. One page when a token weighs nothing.
pub fn grid(unit: usize, slot_bytes: u64, token_bytes: u64) -> usize {
    let tokens = (2 * slot_bytes).div_ceil(token_bytes.max(1)) as usize;
    tokens.div_ceil(unit).max(1) * unit
}

/// Where a prompt fed from `start` to `end` snapshots its state for the
/// pool, ascending: every multiple of `grid` past `start` and short of
/// `end`, then the last page boundary at or before `end` when it is past
/// `start`. `grid` is whole pages of `unit` tokens.
pub fn snapshots(unit: usize, grid: usize, start: usize, end: usize) -> Vec<usize> {
    let tail = end / unit * unit;
    let on_grid = (start / grid + 1..).map(|k| k * grid).take_while(|&p| p < tail);
    on_grid.chain(Some(tail).filter(|&t| t > start)).collect()
}

/// A pooled object as one member holds its bytes: every paged state's
/// `pages`, then every per-sequence state's `slot`, under `key`. A page
/// object names one page and no slot; a state object a slot alone.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Object {
    pub key: String,
    pub pages: Vec<i32>,
    pub slot: Option<i32>,
}

/// Member `m`'s objects for tokens `[0, len)` of a group whose owner is
/// member `me`, on `stage`: whole pages `first..=len / unit` on the owner,
/// and the state object when the member holds a slot, which only a
/// prefix ending on a page boundary does. `page_ids` and `slot` are the
/// member's own.
#[allow(clippy::too_many_arguments)]
pub fn objects(
    keys: &Keys,
    stage: Option<u8>,
    chain: &Chain,
    tokens: &[i64],
    first: usize,
    m: usize,
    me: usize,
    page_ids: &[i32],
    slot: Option<i32>,
) -> Vec<Object> {
    let (unit, len) = (chain.unit(), tokens.len());
    let whole = len / unit;
    let pages = (m == me)
        .then(|| {
            (first..=whole).map(|k| Object {
                key: keys.page(stage, chain.at(k)),
                pages: vec![page_ids[k - 1]],
                slot: None,
            })
        })
        .into_iter()
        .flatten();
    assert!(slot.is_none() || len % unit == 0, "a pooled state of {len} tokens is not on a page of {unit}");
    let state =
        slot.map(|s| Object { key: keys.state(stage, m as u8, chain.at(whole)), pages: Vec::new(), slot: Some(s) });
    pages.chain(state).collect()
}

/// Tokens one member's transfer of [`objects`] landed, from one flag per
/// object: with a state, all of `len` or nothing; without, the whole
/// pages from `first` that landed.
pub fn landed(ok: &[bool], stateful: bool, len: usize, first: usize, unit: usize) -> usize {
    match stateful {
        true => {
            if ok.iter().all(|&b| b) {
                len
            } else {
                0
            }
        }
        false => (first - 1 + ok.iter().take_while(|&&b| b).count()) * unit,
    }
}

/// One transfer across a group: every member names the tokens it landed,
/// and the transfer is whole once all have answered, at the fewest tokens
/// any of them landed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Landing {
    expect: BTreeSet<usize>,
    heard: BTreeMap<usize, usize>,
}

impl Landing {
    pub fn new(members: impl IntoIterator<Item = usize>) -> Landing {
        Landing { expect: members.into_iter().collect(), heard: BTreeMap::new() }
    }

    /// Member `member` landed `tokens` tokens (0: it failed).
    pub fn heard(mut self, member: usize, tokens: usize) -> Result<Landing> {
        if !self.expect.contains(&member) {
            bail!(Api, "landing: member {member} is not among {:?}", self.expect);
        }
        if self.heard.insert(member, tokens).is_some() {
            bail!(Api, "landing: member {member} answered twice");
        }
        Ok(self)
    }

    /// Once every member answered: the fewest tokens any landed.
    pub fn done(&self) -> Option<usize> {
        (self.heard.len() == self.expect.len()).then(|| self.heard.values().copied().min().unwrap_or(0))
    }
}
