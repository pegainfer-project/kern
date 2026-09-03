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
    pub fn new(unit: usize) -> Chain {
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

    pub fn len(&self) -> usize {
        self.len
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
    use std::sync::Arc;

    use super::*;
    use crate::host::Host;
    use crate::pages::Pool;
    use kern_manifest::types::Manifest;

    /// kv paged in 4 tokens, 8 pages.
    fn pool() -> Arc<Pool> {
        let m = Manifest::from_json(
            r#"{
            "schema_version": 4, "model": "t", "vars": {"tokens": {"max": 8}, "seqs": {"max": 2}},
            "states": {"kv": {"bytes_per_token": 1}},
            "buffers": {
                "block_table": {"kind": "input", "dtype": "i32", "shape": ["seqs", 8], "domain": {"index_into": "kv", "stride": 4}}
            },
            "modules": {}, "ops": {}, "programs": {}
        }"#,
        )
        .unwrap()
        ;
        Arc::new(Pool::new(&m, 4, 8, 0).unwrap().0)
    }

    /// kv paged in 4 tokens plus a recurrent state: 2 slots, 6 pages.
    fn hybrid_pool() -> Arc<Pool> {
        let m = Manifest::from_json(
            r#"{
            "schema_version": 4, "model": "t", "vars": {"tokens": {"max": 8}, "seqs": {"max": 1}},
            "states": {"kv": {"bytes_per_token": 1}, "rec": {"bytes_per_seq": 8}},
            "buffers": {
                "block_table": {"kind": "input", "dtype": "i32", "shape": ["seqs", 8], "domain": {"index_into": "kv", "stride": 4}},
                "line_index": {"kind": "input", "dtype": "i32", "shape": [1, "seqs"], "domain": {"index_into": "rec", "stride": 8}}
            },
            "modules": {}, "ops": {}, "programs": {}
        }"#,
        )
        .unwrap();
        Arc::new(Pool::new(&m, 4, 18, 3).unwrap().0)
    }

    fn table() -> Prefix<Checkpoint, Parked> {
        Prefix::new(4)
    }

    fn toks(n: usize) -> Vec<i64> {
        (0..n as i64).map(|i| i * 7 + 3).collect()
    }

    /// Park entry `id` of `t` into `h`: its checkpoint's nodes, no copies run.
    fn park(t: &mut Prefix, h: &Arc<Host>, id: u64) {
        let parked = t
            .park(id, |cp| {
                let slot = cp.seq_slot().map(|s| (s, 8));
                let (p, _) = h.park(&cp.nodes(), 4, slot, cp.tokens()).unwrap();
                Ok::<_, ()>(Ok(p))
            })
            .unwrap();
        assert!(parked);
    }

    #[test]
    fn one_entry_grows_a_page_at_a_time() {
        let p = pool();
        let mut t = table();
        let mut l = p.lease(12).unwrap();
        let a = t.insert(&Chain::over(4, &toks(4)), p.checkpoint(&mut l, 4).unwrap().0);
        let b = t.insert(&Chain::over(4, &toks(8)), p.checkpoint(&mut l, 8).unwrap().0);
        assert_eq!((a, b, t.len()), (a, a, 1));
        // Ten tokens end inside the third page: a second entry with a tail.
        let c = t.insert(&Chain::over(4, &toks(10)), p.checkpoint(&mut l, 10).unwrap().0);
        assert_ne!(c, a);
        assert_eq!(t.len(), 2);
        // A 12-token prompt of the same tokens may use 11: the 10-token entry.
        assert_eq!(t.lookup(&toks(12)), Some(Hit { id: c, len: 10, tier: Tier::Resident }));
        // A 10-token prompt may use 9: 8 whole pages, from either entry (the lower id).
        assert_eq!(t.lookup(&toks(10)), Some(Hit { id: a, len: 8, tier: Tier::Resident }));
        // Diverging inside the third page: still 8.
        let mut d = toks(12);
        d[9] = -1;
        assert_eq!(t.lookup(&d).map(|h| h.len), Some(8));
        // Diverging inside the second page: the entry is usable at its first page.
        d[5] = -1;
        assert_eq!(t.lookup(&d), Some(Hit { id: a, len: 4, tier: Tier::Resident }));
        d[0] = -1;
        assert_eq!(t.lookup(&d), None);
        assert_eq!(t.lookup(&toks(1)), None);
        assert_eq!(t.lookup(&[]), None);
        // Growing past the tailed entry: page 3 extends `a`, not `c`.
        let e = t.insert(&Chain::over(4, &toks(12)), p.checkpoint(&mut l, 12).unwrap().0);
        assert_eq!((e, t.len()), (a, 2));
        assert_eq!(t.lookup(&toks(13)).map(|h| (h.id, h.len)), Some((a, 12)));
    }

    #[test]
    fn same_tokens_share_one_entry() {
        let p = pool();
        let mut t = table();
        let mut l = p.lease(8).unwrap();
        let a = t.insert(&Chain::over(4, &toks(8)), p.checkpoint(&mut l, 8).unwrap().0);
        assert_eq!(p.used(), 2);
        let b = t.insert(&Chain::over(4, &toks(8)), p.checkpoint(&mut l, 8).unwrap().0);
        assert_eq!((a, b, t.len(), p.used()), (a, a, 1, 2));
        // Same pages but different tokens is a different entry.
        let mut other = toks(8);
        other[7] = -1;
        let c = t.insert(&Chain::over(4, &other), p.checkpoint(&mut l, 8).unwrap().0);
        assert_ne!(c, a);
        assert_eq!(t.len(), 2);
    }

    #[test]
    fn coldest_is_least_recent_and_a_hit_touches_the_chain() {
        let p = pool();
        let mut t = table();
        let mut l = p.lease(12).unwrap();
        let a = t.insert(&Chain::over(4, &toks(8)), p.checkpoint(&mut l, 8).unwrap().0);
        let mut other = toks(8);
        other[6] = -1;
        let c = t.insert(&Chain::over(4, &other), p.checkpoint(&mut l, 8).unwrap().0);
        drop(l);
        // Both are usable at page 1; a hit at 9 tokens of `toks` is a's and touches c's shared page too.
        assert_eq!(t.lookup(&toks(9)).map(|h| h.id), Some(a));
        assert_eq!(t.coldest(Tier::Resident), Some(a));
        // Six tokens of `other` are usable at page 1 through either entry: the lower id.
        assert_eq!(t.lookup(&other[..7]).map(|h| (h.id, h.len)), Some((a, 4)));
        // A hit on c touches a (found at page 1) after it: c, the leaf, is the colder.
        assert_eq!(t.lookup(&[&other[..], &[0]].concat()).map(|h| (h.id, h.len)), Some((c, 8)));
        assert_eq!(t.coldest(Tier::Resident), Some(c));
        assert!(t.remove(c));
        assert_eq!((t.len(), p.used()), (1, 2));
        assert!(t.remove(a));
        assert_eq!((t.is_empty(), p.used(), t.coldest(Tier::Resident)), (true, 0, None));
        assert!(!t.remove(c));
    }

    #[test]
    fn a_page_the_lease_still_holds_survives_removal() {
        let p = pool();
        let mut t = table();
        let mut l = p.lease(8).unwrap();
        let a = t.insert(&Chain::over(4, &toks(8)), p.checkpoint(&mut l, 8).unwrap().0);
        assert!(t.remove(a));
        assert_eq!(p.used(), 2);
        drop(l);
        assert_eq!(p.used(), 0);
    }

    #[test]
    fn restore_then_checkpoint_deeper() {
        let p = pool();
        let mut t = table();
        let mut l = p.lease(8).unwrap();
        let a = t.insert(&Chain::over(4, &toks(8)), p.checkpoint(&mut l, 8).unwrap().0);
        drop(l);
        let hit = t.lookup(&toks(16)).unwrap();
        let (mut l2, _) = p.restore(t.resident(hit.id).unwrap(), hit.len, 16).unwrap();
        assert_eq!(l2.prefix(), 8);
        // The sequence grows one chain and its checkpoints grow the entry.
        let mut c = Chain::over(4, &toks(12));
        assert_eq!(t.insert(&c, p.checkpoint(&mut l2, 12).unwrap().0), a);
        c.extend(toks(16)[12..].iter().copied());
        assert_eq!(t.insert(&c, p.checkpoint(&mut l2, 16).unwrap().0), a);
        drop(l2);
        assert_eq!(t.lookup(&toks(17)).map(|h| h.len), Some(16));
        assert_eq!(t.lookup(&toks(13)).map(|h| h.len), Some(12));
        assert_eq!((t.len(), p.used()), (1, 4));
    }

    #[test]
    fn a_branch_is_its_own_entry() {
        let p = pool();
        let mut t = table();
        let mut l = p.lease(8).unwrap();
        let a = t.insert(&Chain::over(4, &toks(8)), p.checkpoint(&mut l, 8).unwrap().0);
        drop(l);
        // Two sequences continue from a; the first extends it, the second branches.
        let hit = t.lookup(&toks(13)).unwrap();
        let (mut x, _) = p.restore(t.resident(hit.id).unwrap(), hit.len, 12).unwrap();
        let (mut y, _) = p.restore(t.resident(hit.id).unwrap(), hit.len, 12).unwrap();
        let mut ty = toks(12);
        ty[9] = -1;
        assert_eq!(t.insert(&Chain::over(4, &toks(12)), p.checkpoint(&mut x, 12).unwrap().0), a);
        let b = t.insert(&Chain::over(4, &ty), p.checkpoint(&mut y, 12).unwrap().0);
        assert_ne!(b, a);
        assert_eq!(t.lookup(&toks(13)).map(|h| h.id), Some(a));
        assert_eq!(t.lookup(&[&ty[..], &[0]].concat()).map(|h| (h.id, h.len)), Some((b, 12)));
        // Both usable at 8; the lower id wins so the choice is stable.
        assert_eq!(t.lookup(&toks(9)).map(|h| h.id), Some(a));
    }

    #[test]
    fn a_stateful_entry_is_usable_at_its_length_only() {
        let p = hybrid_pool();
        let mut t = table();
        let l = p.lease(12).unwrap();
        let a = t.insert(&Chain::over(4, &toks(10)), p.retire(l, 10));
        assert_eq!(t.lookup(&toks(12)), Some(Hit { id: a, len: 10, tier: Tier::Resident }));
        assert_eq!(t.lookup(&toks(10)), None);
        let mut d = toks(12);
        d[9] = -1;
        assert_eq!(t.lookup(&d), None);
        // A longer retirement on the same chain is a second entry, not a growth.
        let l = p.lease(16).unwrap();
        let b = t.insert(&Chain::over(4, &toks(12)), p.retire(l, 12));
        assert_ne!(b, a);
        assert_eq!(t.lookup(&toks(13)).map(|h| h.id), Some(b));
        assert_eq!(t.lookup(&toks(11)).map(|h| h.id), Some(a));
    }

    #[test]
    fn parked_entries_are_found_after_resident_ones() {
        let p = pool();
        let h = Arc::new(Host::new(64, 4));
        let mut t = table();
        let mut l = p.lease(12).unwrap();
        let a = t.insert(&Chain::over(4, &toks(12)), p.checkpoint(&mut l, 12).unwrap().0);
        drop(l);
        assert_eq!(p.used(), 3);
        park(&mut t, &h, a);
        assert_eq!((p.used(), t.count(Tier::Parked), t.coldest(Tier::Resident)), (0, 1, None));
        assert_eq!(t.lookup(&toks(13)), Some(Hit { id: a, len: 12, tier: Tier::Parked }));
        assert_eq!(t.lookup(&toks(6)), Some(Hit { id: a, len: 4, tier: Tier::Parked }));
        assert_eq!(t.parked(a).map(|q| q.pages(2)), Some(vec![0, 4]));
        // The same tokens resident again: found first.
        let mut l = p.lease(8).unwrap();
        let b = t.insert(&Chain::over(4, &toks(8)), p.checkpoint(&mut l, 8).unwrap().0);
        assert_ne!(b, a);
        assert_eq!(t.lookup(&toks(9)), Some(Hit { id: b, len: 8, tier: Tier::Resident }));
        assert_eq!(t.lookup(&toks(13)).map(|h| h.tier), Some(Tier::Parked));
        assert_eq!((t.coldest(Tier::Parked), t.coldest(Tier::Resident)), (Some(a), Some(b)));
        assert!(t.remove(a));
        assert_eq!((h.used(), t.len()), (0, 1));
    }

    #[test]
    fn a_stateful_park_keeps_its_slot_and_its_length() {
        let p = hybrid_pool();
        let h = Arc::new(Host::new(64, 4));
        let mut t = table();
        let l = p.lease(12).unwrap();
        let a = t.insert(&Chain::over(4, &toks(10)), p.retire(l, 10));
        assert_eq!((p.used(), p.slots_used()), (3, 1));
        park(&mut t, &h, a);
        assert_eq!((p.used(), p.slots_used(), h.used()), (0, 0, 20));
        assert_eq!(t.lookup(&toks(12)), Some(Hit { id: a, len: 10, tier: Tier::Parked }));
        assert_eq!(t.lookup(&toks(9)), None);
        assert!(t.parked(a).unwrap().has_slot());
    }

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
            assert_eq!((grown.len(), &grown), (n, &Chain::over(unit, &tokens)));
            for len in 0..=n {
                let short = Chain::over(unit, &tokens[..len]);
                let expected = (len.is_multiple_of(unit) || len == n).then(|| short.key(len).unwrap());
                assert_eq!(grown.key(len), expected, "unit {unit} len {len} of {n}");
            }
            assert_eq!(grown.key(n + 1), None);
        }
    }

    /// Random inserts, lookups and removals against a brute-force model
    /// that keeps every entry's tokens: a lookup's length is the longest
    /// usable prefix any entry holds, and every entry is findable at its
    /// usable lengths and nowhere else.
    #[test]
    fn lookup_matches_the_brute_force_model() {
        let p = pool();
        let mut t = table();
        let mut model: BTreeMap<u64, Vec<i64>> = BTreeMap::new();
        let mut x = 0x1234_5678_9ABC_DEF1u64;
        let mut rand = move |n: usize| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 33) as usize % n
        };
        let mut lease = p.lease(32).unwrap();
        for _ in 0..600 {
            match rand(3) {
                0 => {
                    // A random token sequence of up to 12 out of a 2-symbol alphabet: prefixes collide often.
                    let n = 1 + rand(12);
                    let tokens: Vec<i64> = (0..n).map(|_| rand(2) as i64).collect();
                    if let Ok((cp, _)) = p.checkpoint(&mut lease, n) {
                        let id = t.insert(&Chain::over(4, &tokens), cp);
                        // A grown entry covers its old tokens too; only the longer sequence is kept.
                        model.retain(|&i, v| i != id || v.len() > tokens.len());
                        model.entry(id).or_insert(tokens);
                    }
                }
                1 => {
                    let n = 1 + rand(14);
                    let tokens: Vec<i64> = (0..n).map(|_| rand(2) as i64).collect();
                    let usable = &tokens[..n - 1];
                    let want = model
                        .values()
                        .map(|v| {
                            // Usable: whole pages of v that prefix the prompt, or all of v when it does.
                            let common = v.iter().zip(usable).take_while(|(a, b)| a == b).count();
                            if common == v.len() {
                                v.len()
                            } else {
                                common / 4 * 4
                            }
                        })
                        .max()
                        .unwrap_or(0);
                    let got = t.lookup(&tokens).map_or(0, |h| h.len);
                    assert_eq!(got, want, "prompt {tokens:?}");
                }
                _ if !model.is_empty() => {
                    let id = *model.keys().nth(rand(model.len())).unwrap();
                    assert!(t.remove(id));
                    model.remove(&id);
                }
                _ => {}
            }
            assert_eq!(t.len(), model.len());
        }
    }
}
