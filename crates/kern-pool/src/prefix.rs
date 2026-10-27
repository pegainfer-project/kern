//! The prefix index: which prefixes of past sequences are kept, and the
//! longest one a new prompt can start from.
//!
//! The index is a radix tree over tokens. An edge is a run of tokens, a
//! node is a prefix length, and an entry sits on the node whose path
//! spells the entry's tokens. The tree only finds; the bytes and what
//! they share are the entries' own business ([`crate::Checkpoint`]), so
//! the tree may branch at any token while no page is ever split. A lookup
//! walks the prompt (never its last token: that one must still go through
//! a program) and takes the longest usable entry: one with a recurrent
//! state is usable at its own length only (the state is the state after
//! exactly those tokens); one without is usable at its own length when the
//! prompt covers it, and at any whole page of the tokens the prompt shares
//! with it otherwise. Same tokens, same path: a prefix two sessions both
//! typed is one entry.
//!
//! Room is the caller's answer to a `Busy` lease: [`Prefix::evict`] drops
//! the entry worth least (below), which is all that frees anything (pages
//! shared with a live sequence or another entry stay). Recency is a
//! counter, not a clock. A hit touches every entry on the prompt's path,
//! the deepest coldest, so a stateless chain ages together and its leaf
//! goes first. A stateless entry made one page past another on the same
//! path replaces it: a sequence checkpointing every page keeps one entry
//! that grows.
//!
//! Entries with a recurrent state are a trie of conversations, since one
//! serves only prompts that continue it exactly. By the stateful entries
//! nearest below it, a stateful entry is a session's newest (none), a turn
//! its session has moved past (one: superseded), or a prefix several
//! conversations share (more). Room takes the superseded first, least
//! recently used first; they stay while there is room, so a client that
//! rewrites its conversation back to an older turn still hits it, and give
//! back only their slot and the page they end inside (their other pages
//! are their successor's prefix). Then the session idle longest gives up
//! its newest, never while anything superseded is left. A shared prefix
//! belongs to no one session: it goes only once at most one conversation
//! is left below it, as that one's.
//!
//! No clock, no hash, no hash map: a replay makes the same decisions in
//! the same order.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::store::Checkpoint;

/// What the index needs to know of what it keeps: a checkpoint, or a
/// caller's bundle of several (a tensor-parallel group's per-rank
/// checkpoints of one sequence). A clone is another holder of the same
/// bytes.
pub trait Kept: Clone {
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
        Checkpoint::has_slot(self)
    }
}

/// An entry a prompt can continue from: its first `len` tokens are
/// already in `found`, a holder of the entry, so it stays usable whatever
/// the index does to the entry afterwards.
#[derive(Debug, Clone)]
pub struct Hit<R> {
    pub len: usize,
    pub found: R,
}

struct Entry<R> {
    key: Arc<[i64]>,
    kept: R,
    slot: bool,
    used: u64,
}

struct Node<R> {
    /// The edge from the parent: never empty except at the root.
    tokens: Vec<i64>,
    /// By the first token of the child's edge.
    children: BTreeMap<i64, Node<R>>,
    entry: Option<Entry<R>>,
    /// Stateless entries in this subtree, this node's included: a prompt
    /// diverging inside an entry is usable at the whole pages it shares
    /// with any of them.
    paged_below: usize,
}

impl<R> Node<R> {
    fn new(tokens: Vec<i64>) -> Node<R> {
        Node { tokens, children: BTreeMap::new(), entry: None, paged_below: 0 }
    }
}

fn lcp(a: &[i64], b: &[i64]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// The node at `key`, made if absent (splitting the edge it falls inside).
fn at_mut<'a, R>(root: &'a mut Node<R>, key: &[i64]) -> &'a mut Node<R> {
    let mut node = root;
    let mut pos = 0;
    loop {
        if pos == key.len() {
            return node;
        }
        let first = key[pos];
        let fresh = !node.children.contains_key(&first);
        let child = node.children.entry(first).or_insert_with(|| Node::new(key[pos..].to_vec()));
        if fresh {
            return child;
        }
        let common = lcp(&child.tokens, &key[pos..]);
        if common < child.tokens.len() {
            let rest = child.tokens.split_off(common);
            let tail = Node {
                tokens: rest,
                children: std::mem::take(&mut child.children),
                entry: child.entry.take(),
                paged_below: child.paged_below,
            };
            child.children.insert(tail.tokens[0], tail);
        }
        pos += common;
        node = child;
    }
}

/// The node `key` tokens down a path the tree spells exactly.
fn path_mut<'a, R>(root: &'a mut Node<R>, key: &[i64]) -> &'a mut Node<R> {
    let mut node = root;
    let mut pos = 0;
    while pos < key.len() {
        let child = node.children.get_mut(&key[pos]).expect("on the path");
        pos += child.tokens.len();
        node = child;
    }
    node
}

/// The node at `key`, if the tree has one.
fn at<'a, R>(root: &'a Node<R>, key: &[i64]) -> Option<&'a Node<R>> {
    let mut node = root;
    let mut pos = 0;
    while pos < key.len() {
        let child = node.children.get(&key[pos])?;
        let n = child.tokens.len();
        if key.len() - pos < n || child.tokens[..] != key[pos..pos + n] {
            return None;
        }
        pos += n;
        node = child;
    }
    Some(node)
}

/// Add `paged` to the subtree counts of every node from the root to `key`.
fn bump<R>(root: &mut Node<R>, key: &[i64], paged: isize) {
    let mut node = root;
    let mut pos = 0;
    loop {
        node.paged_below = (node.paged_below as isize + paged) as usize;
        if pos == key.len() {
            return;
        }
        let child = node.children.get_mut(&key[pos]).expect("path exists");
        pos += child.tokens.len();
        node = child;
    }
}

/// Take the entry at `key` out, pruning the node it leaves empty and
/// merging a node left with one child into it.
fn take_entry<R>(node: &mut Node<R>, key: &[i64]) -> Option<Entry<R>> {
    let removed = if key.is_empty() {
        node.entry.take()?
    } else {
        let first = key[0];
        let child = node.children.get_mut(&first)?;
        let n = child.tokens.len();
        if key.len() < n || child.tokens[..] != key[..n] {
            return None;
        }
        let removed = take_entry(child, &key[n..])?;
        if child.entry.is_none() {
            if child.children.is_empty() {
                node.children.remove(&first);
            } else if child.children.len() == 1 {
                let (_, grand) = child.children.pop_first().expect("one child");
                child.tokens.extend(grand.tokens);
                child.children = grand.children;
                child.entry = grand.entry;
            }
        }
        removed
    };
    if !removed.slot {
        node.paged_below -= 1;
    }
    Some(removed)
}

/// A stateless entry somewhere in `node`'s subtree, the node's own first,
/// then the first child's in token order; `None` when there is none.
fn paged_in_mut<R>(node: &mut Node<R>) -> Option<&mut Node<R>> {
    let mut cur = node;
    loop {
        if cur.entry.as_ref().is_some_and(|e| !e.slot) {
            return Some(cur);
        }
        cur = cur.children.values_mut().find(|c| c.paged_below > 0)?;
    }
}

/// What a stateful entry is to the conversations through it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    Newest,
    Superseded,
    Shared,
}

/// Each stateful entry under `node` (its own included) as its role and
/// recency stamp; returns how many stateful entries are nearest at or
/// below `node`.
fn roles<R>(node: &Node<R>, out: &mut Vec<(Role, u64)>) -> usize {
    let below: usize = node.children.values().map(|c| roles(c, out)).sum();
    match node.entry.as_ref().filter(|e| e.slot) {
        Some(e) => {
            let role = match below {
                0 => Role::Newest,
                1 => Role::Superseded,
                _ => Role::Shared,
            };
            out.push((role, e.used));
            1
        }
        None => below,
    }
}

pub struct Prefix<R = Checkpoint> {
    unit: usize,
    root: Node<R>,
    /// Recency stamp → entry key: the eviction order.
    lru: BTreeMap<u64, Arc<[i64]>>,
    count: usize,
    /// Entries holding a recurrent state.
    states: usize,
    clock: u64,
}

impl<R: Kept> Prefix<R> {
    /// An index over sequences paged in `unit` tokens.
    pub fn new(unit: usize) -> Prefix<R> {
        assert!(unit >= 1);
        Prefix { unit, root: Node::new(Vec::new()), lru: BTreeMap::new(), count: 0, states: 0, clock: 0 }
    }

    /// Stateful entries a later turn of their session has moved past.
    pub fn superseded(&self) -> usize {
        let mut out = Vec::new();
        roles(&self.root, &mut out);
        out.iter().filter(|(r, _)| *r == Role::Superseded).count()
    }

    pub fn entries(&self) -> usize {
        self.count
    }

    /// Restamp `e` with `tick`.
    fn restamp(lru: &mut BTreeMap<u64, Arc<[i64]>>, e: &mut Entry<R>, tick: u64) {
        lru.remove(&e.used);
        e.used = tick;
        lru.insert(tick, Arc::clone(&e.key));
    }

    /// Keep `r`, whose tokens are `key`. The same tokens are here already:
    /// the entry counts as used and the new one is dropped. A stateless
    /// entry one page past a stateless one on the same path replaces it.
    pub fn insert(&mut self, key: &[i64], r: R) {
        assert_eq!(key.len(), r.tokens(), "an entry's key is its tokens");
        assert!(!key.is_empty(), "an entry holds at least one token");
        let slot = r.has_slot();
        self.clock += 1;
        let tick = self.clock;
        let node = at_mut(&mut self.root, key);
        match node.entry.as_mut() {
            Some(e) => {
                assert_eq!(e.slot, slot, "an entry's state is its tokens'");
                Self::restamp(&mut self.lru, e, tick);
            }
            None => {
                let key: Arc<[i64]> = Arc::from(key);
                self.lru.insert(tick, Arc::clone(&key));
                self.count += 1;
                self.states += slot as usize;
                node.entry = Some(Entry { key: Arc::clone(&key), kept: r, slot, used: tick });
                if !slot {
                    bump(&mut self.root, &key, 1);
                }
            }
        }
        if !slot && key.len() > self.unit && key.len().is_multiple_of(self.unit) {
            let shallow = &key[..key.len() - self.unit];
            let redundant = at(&self.root, shallow).and_then(|n| n.entry.as_ref()).is_some_and(|e| !e.slot);
            if redundant {
                self.remove(shallow);
            }
        }
    }

    /// The longest entry usable for `tokens` (the last token is never
    /// covered). A hit touches it and every entry on the prompt's path
    /// above it.
    pub fn lookup(&mut self, tokens: &[i64]) -> Option<Hit<R>> {
        let usable = tokens.len().checked_sub(1)?;
        let q = &tokens[..usable];
        let unit = self.unit;
        // How far the tree spells the prompt: `pos` is the deepest node on
        // its path, `matched` how many tokens it shares (into the edge
        // below `pos` when more than `pos`).
        let (pos, matched) = {
            let mut node = &self.root;
            let mut pos = 0;
            loop {
                if pos == usable {
                    break (pos, pos);
                }
                let Some(child) = node.children.get(&q[pos]) else { break (pos, pos) };
                let common = lcp(&child.tokens, &q[pos..]);
                if common < child.tokens.len() {
                    break (pos, pos + common);
                }
                pos += common;
                node = child;
            }
        };
        // Ticks for the path, root highest, so a chain ages together and
        // its leaf goes first; the clock skips the ticks not handed out.
        let base = self.clock + usable as u64 + 2;
        self.clock = base;
        let mut i = 0u64;
        let Prefix { root, lru, .. } = self;
        let mut best: Option<(usize, R)> = None;
        let mut offer = |len: usize, e: &Entry<R>| {
            if len > 0 && best.as_ref().is_none_or(|(l, _)| len > *l) {
                best = Some((len, e.kept.clone()));
            }
        };
        // A stateless entry off the path is usable at the whole pages the
        // prompt shares with it: `at` tokens through a sibling of the
        // path at depth `at`, `matched` through the edge the prompt
        // continues into below `pos`. The longest wins; one sharing no
        // whole page serves nothing and is not touched.
        let mut cands: Vec<(usize, usize, i64)> = Vec::new();
        let mut node = &mut *root;
        let mut at = 0;
        loop {
            if let Some(e) = node.entry.as_mut() {
                Self::restamp(lru, e, base - i);
                i += 1;
                offer(at, e);
            }
            let next = if at < pos { Some(q[at]) } else { (matched > pos).then(|| q[pos]) };
            if let Some((&k, _)) = node.children.iter().find(|(&k, c)| c.paged_below > 0 && next != Some(k)) {
                cands.push((at, at, k));
            }
            if at == pos {
                if let Some((k, c)) = next.and_then(|k| node.children.get(&k).map(|c| (k, c))) {
                    if c.paged_below > 0 {
                        cands.push((matched, at, k));
                    }
                }
                break;
            }
            let child = node.children.get_mut(&q[at]).expect("on the path");
            at += child.tokens.len();
            node = child;
        }
        let usable = cands.iter().filter(|&&(s, ..)| s >= unit);
        if let Some(&(shared, depth, k)) = usable.max_by_key(|&&(s, ..)| s / unit * unit) {
            let parent = path_mut(root, &q[..depth]);
            let n = paged_in_mut(parent.children.get_mut(&k).expect("a child")).expect("counted");
            let e = n.entry.as_mut().expect("a stateless entry");
            Self::restamp(lru, e, base - i);
            offer(shared / unit * unit, e);
        }
        let (len, found) = best?;
        Some(Hit { len, found })
    }

    /// The entry room takes first: the superseded one used least
    /// recently, else the coldest that is no shared prefix.
    fn victim(&self) -> Option<Arc<[i64]>> {
        if self.states == 0 {
            return self.lru.values().next().cloned();
        }
        let mut out = Vec::new();
        roles(&self.root, &mut out);
        let of = |want: Role| out.iter().filter(move |&&(r, _)| r == want);
        if let Some(&(_, used)) = of(Role::Superseded).min_by_key(|&&(_, u)| u) {
            return self.lru.get(&used).cloned();
        }
        let shared: Vec<u64> = of(Role::Shared).map(|&(_, u)| u).collect();
        let order = &self.lru;
        order.iter().find(|(u, _)| !shared.contains(u)).or(order.iter().next()).map(|(_, k)| Arc::clone(k))
    }

    /// Room for a `Busy` lease: the entry room takes first (the module
    /// doc's order; without states, the one hit least recently) is
    /// dropped, and its tokens are named. `None` when nothing is kept.
    pub fn evict(&mut self) -> Option<Arc<[i64]>> {
        let key = self.victim()?;
        self.remove(&key);
        Some(key)
    }

    /// Drop the entry at `key`, if there is one.
    fn remove(&mut self, key: &[i64]) {
        let Some(e) = take_entry(&mut self.root, key) else { return };
        self.lru.remove(&e.used);
        self.count -= 1;
        self.states -= e.slot as usize;
    }
}
