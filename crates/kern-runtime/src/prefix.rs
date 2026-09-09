//! The checkpoint table: which prefixes of past sequences are kept, on the
//! device or parked on the host, and the longest one a new prompt can
//! start from.
//!
//! A [`Checkpoint`] is bytes; what makes it findable is the tokens it
//! holds. The table keys every entry by a hash chain over its tokens in
//! blocks of the page unit — the chain at depth `d` covers the first `d`
//! pages, a tail hash covers what ends inside the next one — so a lookup
//! hashes the prompt once and probes the depths from the deepest down;
//! the first depth with a usable entry gives the longest prefix. A prompt
//! never uses its last token: that token must still go through a program
//! to produce the next one.
//!
//! An entry without a recurrent state is a chain of pages, usable at any
//! whole page of it: it is registered at every depth, and a checkpoint one
//! page deeper along the same chain extends it instead of adding a second
//! entry, so a sequence checkpointing every page keeps one entry that
//! grows. An entry with a state slot is usable at its own length only
//! (the state is the state after exactly those tokens).
//!
//! A sequence carries its own [`Chain`] and grows it as tokens enter the
//! state, so checkpointing every page hashes each token once; the table
//! reads the chain's key at the checkpoint's length instead of rehashing.
//!
//! Room is the caller's answer to a `Busy` lease: [`Prefix::coldest`]
//! names the entry hit least recently in a tier, [`Prefix::park`] moves a
//! resident one to the host, [`Prefix::remove`] drops one; dropping is all
//! that frees anything (pages shared with a live sequence or a deeper
//! entry stay). Recency is a counter, not a clock. A hit touches every
//! entry found on the prompt's chain, deepest first, so a chain ages
//! together and its leaf goes first.
//!
//! Same tokens, same hashes, same choices: the table has no clock and no
//! hash map, so a replay makes the same decisions in the same order.

use std::collections::{BTreeMap, BTreeSet};

use crate::host::Parked;
use crate::pages::Checkpoint;

/// Where an entry's bytes are.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    Resident,
    Parked,
}

/// An entry a prompt can continue from: the first `len` prompt tokens are
/// already in it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hit {
    pub id: u64,
    pub len: usize,
    pub tier: Tier,
}

/// What the table needs to know of what it keeps: a resident checkpoint,
/// a parked one, or a caller's bundle of several (a tensor-parallel
/// tray's per-rank checkpoints of one sequence).
pub trait Kept {
    /// Tokens held; never 0.
    fn tokens(&self) -> usize;
    /// Whether a recurrent state is held, which pins the entry to its
    /// exact length.
    fn has_slot(&self) -> bool;
}

impl Kept for Checkpoint {
    fn tokens(&self) -> usize {
        Checkpoint::tokens(self)
    }

    fn has_slot(&self) -> bool {
        self.seq_slot().is_some()
    }
}

impl Kept for Parked {
    fn tokens(&self) -> usize {
        Parked::tokens(self)
    }

    fn has_slot(&self) -> bool {
        Parked::has_slot(self)
    }
}

enum Held<R, P> {
    Resident(R),
    Parked(P),
}

impl<R: Kept, P: Kept> Held<R, P> {
    fn tier(&self) -> Tier {
        match self {
            Held::Resident(_) => Tier::Resident,
            Held::Parked(_) => Tier::Parked,
        }
    }

    fn has_slot(&self) -> bool {
        match self {
            Held::Resident(c) => c.has_slot(),
            Held::Parked(p) => p.has_slot(),
        }
    }
}

struct Entry<R, P> {
    held: Held<R, P>,
    key: Key,
    /// `heads[d]` covers the first `d` pages, for every depth a pages-only
    /// entry is registered at; empty for one with a slot.
    heads: Vec<u64>,
    used: u64,
}

impl<R, P> Entry<R, P> {
    /// The (depth, chain) buckets this entry sits in.
    fn buckets(&self) -> Vec<(usize, u64)> {
        if self.heads.is_empty() {
            return vec![(self.key.depth, self.key.chain)];
        }
        let from = if self.key.depth == 0 { 0 } else { 1 };
        (from..=self.key.depth).map(|d| (d, self.heads[d])).collect()
    }
}

pub struct Prefix<R = Checkpoint, P = Parked> {
    unit: usize,
    entries: BTreeMap<u64, Entry<R, P>>,
    /// (depth, chain through it) → entries usable there.
    at_depth: BTreeMap<(usize, u64), Vec<u64>>,
    /// Recency stamp → entry, the eviction order.
    lru: BTreeMap<u64, u64>,
    next_id: u64,
    clock: u64,
}

const SEED: u64 = 0x243F_6A88_85A3_08D3;

/// splitmix64's finalizer: a bijection, so a chain never collides with a
/// shorter one by absorbing a zero.
fn mix(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn fold(h: u64, token: i64) -> u64 {
    mix(h ^ (token as u64).wrapping_add(0x9E37_79B9_7F4A_7C15))
}

fn hash(h: u64, tokens: &[i64]) -> u64 {
    tokens.iter().fold(h, |h, &t| fold(h, t))
}

/// What identifies an entry's tokens: whole pages and the chain through
/// them, then the tokens past them and the chain continued.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Key {
    depth: usize,
    chain: u64,
    tail_len: usize,
    tail: u64,
}

/// The hash chain of one sequence, grown a token at a time: one hash per
/// whole page, one over the tokens past the last whole page. Pure data;
/// the same tokens in any grouping give the same chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chain {
    unit: usize,
    /// `heads[d]` covers the first `d` pages; `heads[0]` is the seed.
    heads: Vec<u64>,
    tail: u64,
    len: usize,
}

impl Chain {
    fn new(unit: usize) -> Chain {
        assert!(unit >= 1);
        Chain { unit, heads: vec![SEED], tail: SEED, len: 0 }
    }

    /// The chain of `tokens`.
    pub fn over(unit: usize, tokens: &[i64]) -> Chain {
        let mut c = Chain::new(unit);
        c.extend(tokens.iter().copied());
        c
    }

    pub fn push(&mut self, token: i64) {
        self.tail = fold(self.tail, token);
        self.len += 1;
        if self.len.is_multiple_of(self.unit) {
            self.heads.push(self.tail);
        }
    }

    pub fn extend(&mut self, tokens: impl IntoIterator<Item = i64>) {
        tokens.into_iter().for_each(|t| self.push(t));
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The key of the first `len` tokens: known at every whole page and
    /// at the chain's own length, nowhere else.
    fn key(&self, len: usize) -> Option<Key> {
        let depth = len / self.unit;
        let chain = *self.heads.get(depth)?;
        let tail_len = len % self.unit;
        match tail_len {
            0 => Some(Key { depth, chain, tail_len, tail: chain }),
            _ if len == self.len => Some(Key { depth, chain, tail_len, tail: self.tail }),
            _ => None,
        }
    }
}

impl<R: Kept, P: Kept> Prefix<R, P> {
    /// A table over sequences paged in `unit` tokens.
    pub fn new(unit: usize) -> Prefix<R, P> {
        assert!(unit >= 1);
        Prefix { unit, entries: BTreeMap::new(), at_depth: BTreeMap::new(), lru: BTreeMap::new(), next_id: 0, clock: 0 }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Entries in a tier.
    pub fn count(&self, tier: Tier) -> usize {
        self.entries.values().filter(|e| e.held.tier() == tier).count()
    }

    fn touch(&mut self, id: u64) {
        let e = self.entries.get_mut(&id).expect("entry");
        self.lru.remove(&e.used);
        self.clock += 1;
        e.used = self.clock;
        self.lru.insert(self.clock, id);
    }

    fn register(&mut self, id: u64, bucket: (usize, u64)) {
        self.at_depth.entry(bucket).or_default().push(id);
    }

    /// The entry in `bucket` whose key is exactly `key`.
    fn exact(&self, bucket: (usize, u64), key: Key) -> Option<u64> {
        self.at_depth.get(&bucket)?.iter().copied().find(|id| self.entries[id].key == key)
    }

    /// Keep `checkpoint`, whose tokens are the first `checkpoint.tokens()`
    /// of `chain`. The same tokens are here already: the new one is dropped
    /// and the old one counts as used. One page past a resident pages-only
    /// entry on the same chain: that entry grows.
    pub fn insert(&mut self, chain: &Chain, checkpoint: R) -> u64 {
        assert_eq!(chain.unit, self.unit, "chain unit and table unit differ");
        let key = chain.key(checkpoint.tokens()).expect("checkpoint length is on the chain");
        if let Some(id) = self.exact((key.depth, key.chain), key) {
            self.touch(id);
            return id;
        }
        let slot = checkpoint.has_slot();
        if !slot && key.tail_len == 0 && key.depth >= 1 {
            let below = (key.depth - 1, chain.heads[key.depth - 1]);
            let grows = self.at_depth.get(&below).and_then(|ids| {
                ids.iter().copied().find(|id| {
                    let e = &self.entries[id];
                    e.key == Key { depth: key.depth - 1, chain: below.1, tail_len: 0, tail: below.1 }
                        && !e.heads.is_empty()
                        && e.held.tier() == Tier::Resident
                })
            });
            if let Some(id) = grows {
                let e = self.entries.get_mut(&id).expect("entry");
                e.held = Held::Resident(checkpoint);
                e.key = key;
                e.heads.push(key.chain);
                self.register(id, (key.depth, key.chain));
                self.touch(id);
                return id;
            }
        }
        let id = self.next_id;
        self.next_id += 1;
        self.clock += 1;
        let heads = if slot { Vec::new() } else { chain.heads[..=key.depth].to_vec() };
        let e = Entry { held: Held::Resident(checkpoint), key, heads, used: self.clock };
        for b in e.buckets() {
            self.register(id, b);
        }
        self.entries.insert(id, e);
        self.lru.insert(self.clock, id);
        id
    }

    /// The longest entry holding a proper prefix of `tokens` (the last
    /// token is never covered): a resident one before a parked one of the
    /// same length. A hit touches it and every entry found on the chain
    /// above it.
    pub fn lookup(&mut self, tokens: &[i64]) -> Option<Hit> {
        let usable = tokens.len().checked_sub(1)?;
        let heads = Chain::over(self.unit, &tokens[..usable]).heads;
        let mut best: Option<(usize, std::cmp::Reverse<Tier>, std::cmp::Reverse<u64>)> = None;
        let mut touched: BTreeSet<u64> = BTreeSet::new();
        for (d, &head) in heads.iter().enumerate().rev() {
            let Some(ids) = self.at_depth.get(&(d, head)) else { continue };
            let full = d * self.unit;
            let room = usable - full;
            for &id in ids {
                let e = &self.entries[&id];
                let len = if e.key.depth == d {
                    let k = e.key;
                    let tail_ok = k.tail_len <= room && k.tail == hash(head, &tokens[full..full + k.tail_len]);
                    match (tail_ok, e.held.has_slot()) {
                        (true, _) => full + k.tail_len,
                        (false, false) => full,
                        (false, true) => continue,
                    }
                } else {
                    full
                };
                if len == 0 {
                    continue;
                }
                touched.insert(id);
                let cand = (len, std::cmp::Reverse(e.held.tier()), std::cmp::Reverse(id));
                if best.is_none_or(|b| cand > b) {
                    best = Some(cand);
                }
            }
        }
        let (len, _, std::cmp::Reverse(id)) = best?;
        self.touch(id);
        for other in touched {
            if other != id {
                self.touch(other);
            }
        }
        Some(Hit { id, len, tier: self.entries[&id].held.tier() })
    }

    pub fn resident(&self, id: u64) -> Option<&R> {
        match &self.entries.get(&id)?.held {
            Held::Resident(c) => Some(c),
            Held::Parked(_) => None,
        }
    }

    pub fn parked(&self, id: u64) -> Option<&P> {
        match &self.entries.get(&id)?.held {
            Held::Parked(p) => Some(p),
            Held::Resident(_) => None,
        }
    }

    /// The entry in `tier` used least recently.
    pub fn coldest(&self, tier: Tier) -> Option<u64> {
        self.lru.values().copied().find(|id| self.entries[id].held.tier() == tier)
    }

    /// Entry `id`, resident, is on the host now as `parked`: its
    /// checkpoint drops.
    /// Move resident entry `id` to the host through `park` (the runtime's
    /// copy): `Ok(true)` when it is parked, `Ok(false)` when `park` handed
    /// the checkpoint back (no room) and the entry stays resident; on an
    /// error the entry is gone with the checkpoint.
    pub fn park<E>(
        &mut self,
        id: u64,
        park: impl FnOnce(R) -> std::result::Result<std::result::Result<P, R>, E>,
    ) -> std::result::Result<bool, E> {
        let e = self.entries.remove(&id).expect("entry");
        let buckets = e.buckets();
        let Entry { held, key, heads, used } = e;
        let Held::Resident(cp) = held else { panic!("entry {id} is parked already") };
        let tokens = cp.tokens();
        let (held, parked) = match park(cp) {
            Ok(Ok(p)) => {
                assert_eq!(p.tokens(), tokens, "parking entry {id}");
                (Held::Parked(p), true)
            }
            Ok(Err(cp)) => (Held::Resident(cp), false),
            Err(e) => {
                self.unregister(id, &buckets, used);
                return Err(e);
            }
        };
        self.entries.insert(id, Entry { held, key, heads, used });
        Ok(parked)
    }

    /// Drop entry `id`; `false` when there is none.
    pub fn remove(&mut self, id: u64) -> bool {
        let Some(e) = self.entries.remove(&id) else { return false };
        self.unregister(id, &e.buckets(), e.used);
        true
    }

    /// Forget an entry already taken out of `entries`.
    fn unregister(&mut self, id: u64, buckets: &[(usize, u64)], used: u64) {
        self.lru.remove(&used);
        for b in buckets {
            let ids = self.at_depth.get_mut(b).expect("bucket");
            ids.retain(|&i| i != id);
            if ids.is_empty() {
                self.at_depth.remove(b);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! `Chain::key` is private: the lookup tests in `tests/prefix.rs` see
    //! it only through what a lookup finds.

    use super::*;

    /// A chain grown a token at a time is the chain over the tokens, its
    /// key at every whole page is the shorter chain's, and it has no key
    /// inside a page it has grown past.
    #[test]
    fn a_chain_is_the_same_however_it_grows() {
        let mut x = 0x9E37_79B9u64;
        let mut rand = |n: usize| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % n as u64) as usize
        };
        for _ in 0..200 {
            let unit = 1 + rand(6);
            let n = rand(40);
            let tokens: Vec<i64> = (0..n).map(|_| rand(3) as i64).collect();
            let mut grown = Chain::new(unit);
            for (i, &t) in tokens.iter().enumerate() {
                assert_eq!(grown, Chain::over(unit, &tokens[..i]));
                grown.push(t);
            }
            assert_eq!((grown.len, &grown), (n, &Chain::over(unit, &tokens)));
            for len in 0..=n {
                let short = Chain::over(unit, &tokens[..len]);
                let expected = (len.is_multiple_of(unit) || len == n).then(|| short.key(len).unwrap());
                assert_eq!(grown.key(len), expected, "unit {unit} len {len} of {n}");
            }
            assert_eq!(grown.key(n + 1), None);
        }
    }
}
