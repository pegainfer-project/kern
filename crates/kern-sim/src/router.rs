//! Cluster routing policies and delayed KV-index shadow view.
//!
//! Mirrors external serving gateways (such as agentgateway / Dynamo's KV-aware
//! router): the router sees each endpoint's queue load and a shadow prefix
//! index updated asynchronously via `Insert` / `Evict` events after a
//! configurable `kv_event_lag_ns`, never reaching into an endpoint's internal
//! `Pool` or mutating its `Prefix` LRU state.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Cluster-level request routing policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "policy", rename_all = "snake_case")]
pub enum RouterPolicy {
    /// Deterministic round-robin across endpoints.
    RoundRobin,
    /// Route to the endpoint with the lowest estimated pending token work.
    LeastWork {
        /// Equivalent prefill-token weight assigned per active running sequence.
        #[serde(default = "default_active_seq_weight")]
        active_seq_weight: f64,
    },
    /// KV-cache and queue-load aware routing over a (possibly lagged) shadow index.
    KvAware {
        /// Weight on uncached prompt tokens (`prompt_len - matched_prefix_tokens`).
        #[serde(default = "default_prefill_weight")]
        prefill_weight: f64,
        /// Weight on already-queued prefill tokens at the endpoint.
        #[serde(default = "default_queue_weight")]
        queue_weight: f64,
        /// Equivalent token penalty per active running/queued sequence on the endpoint.
        #[serde(default = "default_active_seq_weight")]
        active_seq_weight: f64,
        /// Propagation delay from endpoint insert/evict to router shadow view, in ns.
        #[serde(default)]
        kv_event_lag_ns: u64,
    },
    /// Route prompts `<= short_max_tokens` to the first half of endpoints and
    /// longer prompts to the second half, using `LeastWork` within each pool.
    LengthSplit {
        short_max_tokens: usize,
    },
}

fn default_prefill_weight() -> f64 {
    2.0
}

fn default_queue_weight() -> f64 {
    0.5
}

fn default_active_seq_weight() -> f64 {
    128.0
}

impl RouterPolicy {
    pub fn kv_aware_default() -> Self {
        Self::KvAware {
            prefill_weight: default_prefill_weight(),
            queue_weight: default_queue_weight(),
            active_seq_weight: default_active_seq_weight(),
            kv_event_lag_ns: 0,
        }
    }

    pub fn kv_event_lag_ns(&self) -> u64 {
        match self {
            Self::KvAware { kv_event_lag_ns, .. } => *kv_event_lag_ns,
            _ => 0,
        }
    }
}

/// Live load snapshot of one endpoint visible to the router at routing time.
#[derive(Debug, Clone, Copy)]
pub struct EndpointLoad {
    pub endpoint_id: usize,
    pub page_unit: usize,
    pub max_seq_tokens: usize,
    pub queued_requests: usize,
    pub queued_prefill_tokens: usize,
    pub running_sequences: usize,
    pub max_sequences: usize,
}

/// Stateful router instance driving request placement across `n_endpoints`.
pub struct Router {
    policy: RouterPolicy,
    cursor: usize,
    shadows: Vec<ShadowTrie>,
}

impl Router {
    pub fn new(policy: RouterPolicy, n_endpoints: usize) -> Self {
        Self {
            policy,
            cursor: 0,
            shadows: (0..n_endpoints).map(|_| ShadowTrie::default()).collect(),
        }
    }

    pub fn policy(&self) -> &RouterPolicy {
        &self.policy
    }

    /// Apply an asynchronous prefix insert event to `endpoint_id`'s shadow tree.
    pub fn on_insert(&mut self, endpoint_id: usize, key: &[i64], has_slot: bool) {
        if let Some(tree) = self.shadows.get_mut(endpoint_id) {
            tree.insert(key, has_slot);
        }
    }

    /// Apply an asynchronous prefix eviction event to `endpoint_id`'s shadow tree.
    pub fn on_evict(&mut self, endpoint_id: usize, key: &[i64]) {
        if let Some(tree) = self.shadows.get_mut(endpoint_id) {
            tree.remove(key);
        }
    }

    /// Choose an endpoint for a request with `prompt_tokens` and `output_tokens`.
    pub fn route(
        &mut self,
        prompt_tokens: &[i64],
        output_tokens: usize,
        loads: &[EndpointLoad],
    ) -> usize {
        assert!(!loads.is_empty(), "cluster must have at least one endpoint");
        let need_total = prompt_tokens.len() + output_tokens;

        // Prefer endpoints whose page-table row limit can fit this request.
        let feasible: Vec<&EndpointLoad> = loads
            .iter()
            .filter(|l| need_total <= l.max_seq_tokens)
            .collect();
        let candidates: Vec<&EndpointLoad> = if feasible.is_empty() {
            loads.iter().collect()
        } else {
            feasible
        };

        let chosen = match &self.policy {
            RouterPolicy::RoundRobin => {
                let pick = candidates[self.cursor % candidates.len()].endpoint_id;
                self.cursor = self.cursor.wrapping_add(1);
                pick
            }
            RouterPolicy::LeastWork { active_seq_weight } => {
                let start = self.cursor;
                self.cursor = self.cursor.wrapping_add(1);
                Self::best_by(&candidates, start, |l| {
                    l.queued_prefill_tokens as f64
                        + (l.running_sequences + l.queued_requests) as f64 * *active_seq_weight
                })
            }
            RouterPolicy::KvAware {
                prefill_weight,
                queue_weight,
                active_seq_weight,
                ..
            } => {
                let start = self.cursor;
                self.cursor = self.cursor.wrapping_add(1);
                let shadows = &self.shadows;
                Self::best_by(&candidates, start, |l| {
                    let hit = shadows[l.endpoint_id].longest_hit(prompt_tokens, l.page_unit.max(1));
                    let uncached = prompt_tokens.len().saturating_sub(hit) as f64;
                    let active = (l.running_sequences + l.queued_requests) as f64;
                    let saturation = if l.running_sequences >= l.max_sequences {
                        active * *active_seq_weight * 2.0
                    } else {
                        active * *active_seq_weight
                    };
                    *prefill_weight * uncached
                        + *queue_weight * (l.queued_prefill_tokens as f64)
                        + saturation
                })
            }
            RouterPolicy::LengthSplit { short_max_tokens } => {
                let mid = (loads.len() / 2).max(1);
                let pool: Vec<&EndpointLoad> = if loads.len() <= 1 {
                    candidates
                } else if prompt_tokens.len() <= *short_max_tokens {
                    let sub: Vec<&EndpointLoad> = candidates
                        .iter()
                        .copied()
                        .filter(|l| l.endpoint_id < mid)
                        .collect();
                    if sub.is_empty() { candidates } else { sub }
                } else {
                    let sub: Vec<&EndpointLoad> = candidates
                        .iter()
                        .copied()
                        .filter(|l| l.endpoint_id >= mid)
                        .collect();
                    if sub.is_empty() { candidates } else { sub }
                };
                let start = self.cursor;
                self.cursor = self.cursor.wrapping_add(1);
                Self::best_by(&pool, start, |l| {
                    l.queued_prefill_tokens as f64
                        + (l.running_sequences + l.queued_requests) as f64 * 256.0
                })
            }
        };
        chosen
    }

    fn best_by(
        candidates: &[&EndpointLoad],
        tie_offset: usize,
        mut score: impl FnMut(&EndpointLoad) -> f64,
    ) -> usize {
        let n = candidates.len();
        let mut best_ep = candidates[0].endpoint_id;
        let mut best_val = f64::INFINITY;
        for step in 0..n {
            let load = candidates[(tie_offset + step) % n];
            let s = score(load);
            if s < best_val {
                best_val = s;
                best_ep = load.endpoint_id;
            }
        }
        best_ep
    }
}

/// Compact read-only-queryable radix trie mirroring `kern_pool::Prefix` hit rules.
#[derive(Default)]
pub(crate) struct ShadowTrie {
    root: ShadowNode,
}

#[derive(Default)]
struct ShadowNode {
    tokens: Vec<i64>,
    children: BTreeMap<i64, ShadowNode>,
    /// `Some(has_slot)` when an entry sits at this node.
    entry: Option<bool>,
    paged_below: usize,
}

impl ShadowTrie {
    pub(crate) fn insert(&mut self, key: &[i64], has_slot: bool) {
        if key.is_empty() {
            return;
        }
        let added_paged = insert_at(&mut self.root, key, has_slot);
        if added_paged {
            bump_paged(&mut self.root, key, 1);
        }
    }

    pub(crate) fn remove(&mut self, key: &[i64]) {
        if key.is_empty() {
            return;
        }
        let _ = remove_at(&mut self.root, key);
    }

    /// Longest prefix of `prompt` (excluding its final token) reusable under
    /// `kern_pool::Prefix`'s stateful and stateless rules.
    pub(crate) fn longest_hit(&self, prompt: &[i64], unit: usize) -> usize {
        let Some(usable) = prompt.len().checked_sub(1) else {
            return 0;
        };
        if usable == 0 {
            return 0;
        }
        let q = &prompt[..usable];
        let mut node = &self.root;
        let mut depth = 0usize;
        let mut best = 0usize;

        loop {
            if node.entry.is_some() && depth > best {
                best = depth;
            }
            if depth == usable {
                if node.children.values().any(|c| c.paged_below > 0) {
                    best = best.max((depth / unit) * unit);
                }
                break;
            }
            let next_tok = q[depth];
            if node
                .children
                .iter()
                .any(|(&k, c)| k != next_tok && c.paged_below > 0)
            {
                best = best.max((depth / unit) * unit);
            }
            let Some(child) = node.children.get(&next_tok) else {
                break;
            };
            let common = lcp(&child.tokens, &q[depth..]);
            if common < child.tokens.len() {
                if child.paged_below > 0 {
                    best = best.max(((depth + common) / unit) * unit);
                }
                break;
            }
            depth += common;
            node = child;
        }
        best
    }
}

fn lcp(a: &[i64], b: &[i64]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

fn insert_at(root: &mut ShadowNode, key: &[i64], has_slot: bool) -> bool {
    let mut node = root;
    let mut pos = 0;
    loop {
        if pos == key.len() {
            let was_stateless = node.entry == Some(false);
            node.entry = Some(has_slot);
            return !has_slot && !was_stateless;
        }
        let first = key[pos];
        let child = match node.children.entry(first) {
            std::collections::btree_map::Entry::Vacant(e) => {
                e.insert(ShadowNode {
                    tokens: key[pos..].to_vec(),
                    children: BTreeMap::new(),
                    entry: Some(has_slot),
                    paged_below: 0,
                });
                return !has_slot;
            }
            std::collections::btree_map::Entry::Occupied(e) => e.into_mut(),
        };
        let common = lcp(&child.tokens, &key[pos..]);
        if common < child.tokens.len() {
            let rest = child.tokens.split_off(common);
            let tail = ShadowNode {
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

fn bump_paged(root: &mut ShadowNode, key: &[i64], delta: isize) {
    let mut node = root;
    let mut pos = 0;
    loop {
        node.paged_below = (node.paged_below as isize + delta) as usize;
        if pos == key.len() {
            return;
        }
        let Some(child) = node.children.get_mut(&key[pos]) else {
            return;
        };
        pos += child.tokens.len();
        node = child;
    }
}

fn remove_at(node: &mut ShadowNode, key: &[i64]) -> Option<bool> {
    let removed_slot = if key.is_empty() {
        node.entry.take()?
    } else {
        let first = key[0];
        let child = node.children.get_mut(&first)?;
        let n = child.tokens.len();
        if key.len() < n || child.tokens[..] != key[..n] {
            return None;
        }
        let removed = remove_at(child, &key[n..])?;
        if child.entry.is_none() {
            if child.children.is_empty() {
                node.children.remove(&first);
            } else if child.children.len() == 1 {
                let (_, grand) = child.children.pop_first().expect("single child");
                child.tokens.extend(grand.tokens);
                child.children = grand.children;
                child.entry = grand.entry;
            }
        }
        removed
    };
    if !removed_slot {
        node.paged_below = node.paged_below.saturating_sub(1);
    }
    Some(removed_slot)
}
