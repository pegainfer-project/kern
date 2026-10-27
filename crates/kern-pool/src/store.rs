//! What a prefix's bytes are: a chain of frozen pages and a state slot,
//! held by value.
//!
//! A [`Checkpoint`] is the first `len` tokens of a sequence in the pool:
//! the pages holding those tokens, shared through a chain, and the slot
//! holding the recurrent state after them. A checkpoint is a value:
//! cloning it is another holder, dropping the last holder returns what
//! nobody else holds.
//!
//! A [`Node`] is one page and the chain before it, reference-counted, so
//! holding a node holds every page up to it and a page returns when its
//! last holder lets go. **A node's page is never written after the node
//! exists**: nodes are made from a lease's whole pages, from a partial page
//! copied for a new holder, or from a retiring lease's last page, and the
//! only writer of any page is a live lease into its own pages.

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use crate::pages::Pool;

pub(crate) fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A frozen page and the chain before it.
pub(crate) struct Node {
    pub(crate) page: i32,
    parent: Option<Arc<Node>>,
    pool: Arc<Pool>,
}

impl Node {
    pub(crate) fn new(page: i32, parent: Option<Arc<Node>>, pool: &Arc<Pool>) -> Arc<Node> {
        Arc::new(Node { page, parent, pool: Arc::clone(pool) })
    }
}

impl Drop for Node {
    /// Unwind the chain in a loop: a recursive drop of a 65k-page chain
    /// would overflow the stack.
    fn drop(&mut self) {
        self.pool.give_page(self.page);
        let mut next = self.parent.take();
        while let Some(n) = next {
            match Arc::try_unwrap(n) {
                Ok(mut node) => next = node.parent.take(),
                Err(_) => break,
            }
        }
    }
}

/// A slot handed back when its last holder drops.
pub(crate) struct SlotOwn {
    pub(crate) id: i32,
    pool: Arc<Pool>,
}

impl SlotOwn {
    pub(crate) fn new(id: i32, pool: &Arc<Pool>) -> Arc<SlotOwn> {
        Arc::new(SlotOwn { id, pool: Arc::clone(pool) })
    }
}

impl Drop for SlotOwn {
    fn drop(&mut self) {
        self.pool.give_slot(self.id);
    }
}

/// The nodes of a chain, root first.
fn chain_nodes(chain: &Option<Arc<Node>>) -> Vec<Arc<Node>> {
    let mut out = Vec::new();
    let mut cur = chain.clone();
    while let Some(n) = cur {
        cur = n.parent.clone();
        out.push(n);
    }
    out.reverse();
    out
}

/// The pages of a chain, root first.
pub(crate) fn chain_pages(chain: &Option<Arc<Node>>) -> Vec<i32> {
    chain_nodes(chain).iter().map(|n| n.page).collect()
}

/// The node `depth` pages up the chain from `chain` (0: `chain` itself).
pub(crate) fn ancestor(chain: &Option<Arc<Node>>, depth: usize) -> Option<Arc<Node>> {
    let mut cur = chain.clone();
    for _ in 0..depth {
        cur = cur.and_then(|n| n.parent.clone());
    }
    cur
}

/// The first `len` tokens of a sequence that is gone: the pages holding
/// them (shared with whoever else holds them) and, when the manifest has
/// per-sequence states, a slot with the state after those tokens. A
/// checkpoint with a slot is usable at its own length only; one without
/// at any whole page of it. Holding one holds its bytes; dropping the
/// last holder releases what it alone held. Made by [`Pool::checkpoint`]
/// and [`Pool::retire`], spent by [`Pool::restore`].
pub struct Checkpoint {
    pub(crate) len: usize,
    pub(crate) pages: usize,
    /// The chain through the pages; `None` for a slot-only checkpoint.
    pub(crate) chain: Option<Arc<Node>>,
    pub(crate) slot: Option<Arc<SlotOwn>>,
}

impl Clone for Checkpoint {
    fn clone(&self) -> Checkpoint {
        Checkpoint { len: self.len, pages: self.pages, chain: self.chain.clone(), slot: self.slot.clone() }
    }
}

impl Checkpoint {
    /// Tokens held; never 0.
    pub fn tokens(&self) -> usize {
        self.len
    }

    /// Whether a state slot is held.
    pub fn has_slot(&self) -> bool {
        self.slot.is_some()
    }

    pub(crate) fn slot_id(&self) -> Option<i32> {
        self.slot.as_ref().map(|s| s.id)
    }
}

impl fmt::Debug for Checkpoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Checkpoint({} tokens, {} pages", self.len, self.pages)?;
        if let Some(s) = &self.slot {
            write!(f, ", slot {:?}", s.id)?;
        }
        write!(f, ")")
    }
}

/// The byte moves that realize a pool decision, as (from, to) page and
/// slot numbers: `pages` for every paged state, `slot` for every
/// per-sequence state. Empty when nothing moves. A plan is executed by
/// the shell that asked for it, right away, while the handles it was made
/// with are alive.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Copies {
    pub pages: Vec<(i32, i32)>,
    pub slot: Option<(i32, i32)>,
}
