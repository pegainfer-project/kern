//! The prefix index through its public API: a lookup finds the longest
//! usable prefix, entries hold their checkpoints, and making room drops
//! the entry worth least.

mod common;

use std::sync::Arc;

use kern_pool::{Kept, Prefix};

use common::{hybrid4, pool4, pool_of, Rand};

fn table() -> Prefix {
    Prefix::new(4)
}

fn toks(n: usize) -> Vec<i64> {
    (0..n as i64).map(|i| i * 7 + 3).collect()
}

/// The length of what `t` finds for `tokens`.
fn find(t: &mut Prefix, tokens: &[i64]) -> Option<usize> {
    t.lookup(tokens).map(|h| h.len)
}

#[test]
fn one_entry_grows_a_page_at_a_time() {
    let p = pool4();
    let mut t = table();
    let mut l = p.lease(12).unwrap();
    t.insert(&toks(4), p.checkpoint(&mut l, 4).unwrap().0);
    t.insert(&toks(8), p.checkpoint(&mut l, 8).unwrap().0);
    assert_eq!((t.entries(), find(&mut t, &toks(9))), (1, Some(8)));
    // Ten tokens end inside the third page: a second entry with a tail.
    t.insert(&toks(10), p.checkpoint(&mut l, 10).unwrap().0);
    assert_eq!(t.entries(), 2);
    // A 12-token prompt of the same tokens may use 11: the 10-token entry.
    let h = t.lookup(&toks(12)).unwrap();
    assert_eq!((h.len, h.found.tokens()), (10, 10));
    // A 10-token prompt may use 9: 8 whole pages, the 8-token entry itself.
    let h = t.lookup(&toks(10)).unwrap();
    assert_eq!((h.len, h.found.tokens()), (8, 8));
    // Diverging inside the third page: still 8.
    let mut d = toks(12);
    d[9] = -1;
    assert_eq!(find(&mut t, &d), Some(8));
    // Diverging inside the second page: the entry is usable at its first page.
    d[5] = -1;
    let h = t.lookup(&d).unwrap();
    assert_eq!((h.len, h.found.tokens()), (4, 8));
    d[0] = -1;
    assert_eq!(find(&mut t, &d), None);
    assert!(t.lookup(&toks(1)).is_none());
    assert!(t.lookup(&[]).is_none());
    // Growing past the tailed entry: page 3 replaces the 8-token entry, not the 10.
    t.insert(&toks(12), p.checkpoint(&mut l, 12).unwrap().0);
    assert_eq!(t.entries(), 2);
    assert_eq!(find(&mut t, &toks(13)), Some(12));
    assert_eq!(find(&mut t, &toks(11)), Some(10));
}

#[test]
fn same_tokens_share_one_entry() {
    let p = pool4();
    let mut t = table();
    let mut l = p.lease(8).unwrap();
    t.insert(&toks(8), p.checkpoint(&mut l, 8).unwrap().0);
    assert_eq!(p.used(), 2);
    t.insert(&toks(8), p.checkpoint(&mut l, 8).unwrap().0);
    assert_eq!((t.entries(), p.used()), (1, 2));
    // Same pages but different tokens is a different entry.
    let mut other = toks(8);
    other[7] = -1;
    t.insert(&other, p.checkpoint(&mut l, 8).unwrap().0);
    assert_eq!(t.entries(), 2);
}

#[test]
fn the_coldest_goes_first_and_a_hit_touches_its_path() {
    let p = pool4();
    let mut t = table();
    let mut l = p.lease(12).unwrap();
    let a = toks(8);
    t.insert(&a, p.checkpoint(&mut l, 8).unwrap().0);
    let mut c = toks(8);
    c[6] = -1;
    t.insert(&c, p.checkpoint(&mut l, 8).unwrap().0);
    // A hit at 9 tokens of `a` touches a alone: c is the colder.
    assert_eq!(find(&mut t, &toks(9)), Some(8));
    // Six tokens of c are usable at page 1 through either entry; the
    // one found is touched, so a is the colder again.
    assert_eq!(find(&mut t, &c[..7]), Some(4));
    assert_eq!(find(&mut t, &[&c[..], &[0]].concat()), Some(8));
    // Dropping the coldest (a) frees only what no lease still holds.
    assert_eq!(t.evict(), Some(Arc::from(&a[..])));
    assert_eq!((t.entries(), p.used(), find(&mut t, &toks(9))), (1, 3, Some(4)));
    assert_eq!(find(&mut t, &[&c[..], &[0]].concat()), Some(8));
    assert_eq!(t.evict(), Some(Arc::from(&c[..])));
    assert_eq!((t.entries(), p.used(), t.evict()), (0, 3, None));
    drop(l);
    assert_eq!(p.used(), 0);
}

#[test]
fn restore_then_checkpoint_deeper() {
    let p = pool4();
    let mut t = table();
    let mut l = p.lease(8).unwrap();
    t.insert(&toks(8), p.checkpoint(&mut l, 8).unwrap().0);
    drop(l);
    let hit = t.lookup(&toks(16)).unwrap();
    let (mut l2, _) = p.restore(&hit.found, hit.len, 16).unwrap();
    assert_eq!(l2.prefix(), 8);
    // The sequence's checkpoints grow the entry a page at a time.
    t.insert(&toks(12), p.checkpoint(&mut l2, 12).unwrap().0);
    t.insert(&toks(16), p.checkpoint(&mut l2, 16).unwrap().0);
    drop(l2);
    assert_eq!(find(&mut t, &toks(17)), Some(16));
    assert_eq!(find(&mut t, &toks(13)), Some(12));
    assert_eq!((t.entries(), p.used()), (1, 4));
}

#[test]
fn a_branch_is_its_own_entry() {
    let p = pool4();
    let mut t = table();
    let mut l = p.lease(8).unwrap();
    t.insert(&toks(8), p.checkpoint(&mut l, 8).unwrap().0);
    drop(l);
    // Two sequences continue from the entry; the first extends it, the second branches.
    let hit = t.lookup(&toks(13)).unwrap();
    let (mut x, _) = p.restore(&hit.found, hit.len, 12).unwrap();
    let (mut y, _) = p.restore(&hit.found, hit.len, 12).unwrap();
    let mut ty = toks(12);
    ty[9] = -1;
    t.insert(&toks(12), p.checkpoint(&mut x, 12).unwrap().0);
    t.insert(&ty, p.checkpoint(&mut y, 12).unwrap().0);
    assert_eq!(t.entries(), 2);
    assert_eq!(find(&mut t, &toks(13)), Some(12));
    assert_eq!(find(&mut t, &[&ty[..], &[0]].concat()), Some(12));
    // Both usable at 8 whole tokens; the tree's first branch wins so the choice is stable.
    let h = t.lookup(&toks(9)).unwrap();
    assert_eq!((h.len, h.found.tokens()), (8, 12));
}

#[test]
fn a_stateful_entry_is_usable_at_its_length_only() {
    // Room for two retired entries beside slot 0, and their 7 pages.
    let p = pool_of(&hybrid4(), 7, 3);
    let mut t = table();
    let l = p.lease(12).unwrap();
    t.insert(&toks(10), p.retire(l, 10));
    assert_eq!(find(&mut t, &toks(12)), Some(10));
    assert!(t.lookup(&toks(10)).is_none());
    let mut d = toks(12);
    d[9] = -1;
    assert!(t.lookup(&d).is_none());
    // A longer retirement on the same path is a second entry, not a growth.
    let l = p.lease(16).unwrap();
    t.insert(&toks(12), p.retire(l, 12));
    assert_eq!(t.entries(), 2);
    assert_eq!(find(&mut t, &toks(13)), Some(12));
}

#[test]
fn a_stateful_checkpoint_at_the_history_end_serves_a_turn_that_replaced_the_reply() {
    // A turn: 6 prompt tokens checkpointed mid-page before the reply
    // moves the state, then retired with 4 generated tokens after them.
    // Slot 0, the lease's, the checkpoint's and the next turn's.
    let p = pool_of(&hybrid4(), 6, 4);
    let mut t = table();
    let mut l = p.lease(12).unwrap();
    let (cp, copies) = p.checkpoint(&mut l, 6).unwrap();
    assert_eq!((copies.pages.len(), copies.slot.is_some()), (1, true));
    t.insert(&toks(6), cp);
    t.insert(&toks(10), p.retire(l, 10));
    // The next prompt carries a recorded reply, not ours: only the
    // history-end entry is on its path.
    let next: Vec<i64> = toks(6).into_iter().chain([-5, -6, -7, -8]).collect();
    let h = t.lookup(&next).unwrap();
    assert_eq!((h.len, t.entries()), (6, 2));
    let (lease, copies) = p.restore(&h.found, h.len, 12).unwrap();
    assert_eq!((lease.prefix(), copies.pages.len(), copies.slot.is_some()), (6, 1, true));
    assert_eq!(find(&mut t, &toks(11)), Some(10));
}

/// What the brute-force model keeps of an entry.
#[derive(Clone)]
struct K(usize, bool);

impl Kept for K {
    fn tokens(&self) -> usize {
        self.0
    }

    fn has_slot(&self) -> bool {
        self.1
    }
}

/// Each entry's role among the stateful entries, by how many are nearest
/// below it: 0 newest (or stateless), 1 superseded, 2 for more (shared).
fn roles(entries: &[(&[i64], bool)]) -> Vec<usize> {
    let below = |a: &[i64], b: &[i64]| b.len() > a.len() && b.starts_with(a);
    entries
        .iter()
        .map(|&(e, _)| {
            let nearest = entries.iter().filter(|&&(d, slot)| {
                slot && below(e, d) && !entries.iter().any(|&(f, fs)| fs && below(e, f) && below(f, d))
            });
            nearest.count().min(2)
        })
        .collect()
}

/// Random inserts, lookups and evictions against a brute-force model that
/// keeps every entry's tokens: a lookup's length is the longest usable
/// prefix any entry holds; making room drops one entry and names it; and
/// the counts agree. Which entry is the coldest is the index's business.
/// Run twice, the index makes the same decisions in the same order.
#[test]
fn lookup_matches_the_brute_force_model() {
    let trace = |seed: u64| {
        let mut t: Prefix<K> = Prefix::new(4);
        // Each entry's tokens and whether a state pins it.
        let mut model: Vec<(Vec<i64>, bool)> = Vec::new();
        let mut rand = Rand::new(seed);
        let mut out: Vec<usize> = Vec::new();
        for _ in 0..3000 {
            match rand.next(6) {
                0 | 1 => {
                    // Up to 12 tokens out of a 2-symbol alphabet: prefixes collide often.
                    let n = 1 + rand.next(12);
                    let tokens: Vec<i64> = (0..n).map(|_| rand.next(2) as i64).collect();
                    // Whether a state is held is the sequence's, not the insert's.
                    let slot = tokens.iter().sum::<i64>() % 3 == 0;
                    t.insert(&tokens, K(n, slot));
                    if !model.iter().any(|(k, _)| *k == tokens) {
                        model.push((tokens.clone(), slot));
                    }
                    // One page past a stateless entry replaces it.
                    if !slot && n > 4 && n.is_multiple_of(4) {
                        model.retain(|(k, s)| !(*k == tokens[..n - 4] && !s));
                    }
                }
                2 | 3 => {
                    let n = 1 + rand.next(14);
                    let tokens: Vec<i64> = (0..n).map(|_| rand.next(2) as i64).collect();
                    let usable = &tokens[..n - 1];
                    let want = model
                        .iter()
                        .map(|(k, slot)| {
                            // Usable: all of it when the prompt covers it, else its whole pages when stateless.
                            let common = k.iter().zip(usable).take_while(|(a, b)| a == b).count();
                            if common == k.len() {
                                k.len()
                            } else if *slot {
                                0
                            } else {
                                common / 4 * 4
                            }
                        })
                        .filter(|len| *len > 0)
                        .max();
                    let got = t.lookup(&tokens).map(|h| h.len);
                    assert_eq!(got, want, "prompt {tokens:?}");
                    out.extend(got);
                }
                4 | 5 => {
                    let roles = roles(&model.iter().map(|(k, s)| (&k[..], *s)).collect::<Vec<_>>());
                    let of = |want: usize| model.iter().zip(&roles).filter(move |((_, s), r)| *s && **r == want);
                    let superseded: Vec<Vec<i64>> = of(1).map(|((k, _), _)| k.clone()).collect();
                    let shared: Vec<Vec<i64>> = of(2).map(|((k, _), _)| k.clone()).collect();
                    let movable = model.len() > shared.len();
                    let got = t.evict();
                    // Room takes a superseded entry while one is kept, and a
                    // shared prefix only when nothing else is.
                    match &got {
                        None => assert!(model.is_empty()),
                        Some(key) => {
                            let key = key.to_vec();
                            assert!(
                                superseded.is_empty() || superseded.contains(&key),
                                "took {key:?} over {superseded:?}"
                            );
                            assert!(!movable || !shared.contains(&key), "took shared {key:?}");
                            assert!(model.iter().any(|(k, _)| *k == key), "dropped {key:?}");
                            model.retain(|(k, _)| *k != key);
                        }
                    }
                    out.push(got.map_or(0, |k| k.len() + 100));
                }
                _ => {}
            }
            assert_eq!(t.entries(), model.len());
        }
        out
    };
    assert_eq!(trace(0x1234_5678_9ABC_DEF1), trace(0x1234_5678_9ABC_DEF1));
}

#[test]
fn a_lookup_that_uses_nothing_touches_nothing() {
    let p = pool4();
    let mut t = table();
    let mut l = p.lease(4).unwrap();
    let mut l2 = p.lease(4).unwrap();
    let (a, mut b) = (toks(4), toks(4));
    b[0] = -1;
    t.insert(&b, p.checkpoint(&mut l2, 4).unwrap().0);
    t.insert(&a, p.checkpoint(&mut l, 4).unwrap().0);
    // Nothing shared, then one token of b: under a page, so no hit, and
    // an entry that served nothing is not made warm by it.
    assert_eq!(find(&mut t, &[999, 998]), None);
    assert_eq!(find(&mut t, &[b[0], 55, 56]), None);
    assert_eq!(t.evict(), Some(Arc::from(&b[..])));
}

/// Conversations taking turns, some forked from another's checkpoint,
/// against a reference trie of their checkpoints that stamps recency as
/// the index does (a lookup touches every entry on the prompt's path, the
/// deeper colder; an insert touches its own). A turn hits the deepest
/// checkpoint on its path and is kept at its history end; room is made
/// whenever more than `ROOM` are kept, and now and then besides. Room takes
/// the least recent superseded checkpoint while any is kept, else the
/// least recent newest one, never a prefix two conversations went on
/// from; so a conversation loses its newest only once nothing superseded
/// is left, and a shared prefix outlives the conversations evicted below
/// it until at most one is left.
#[test]
fn room_takes_turns_moved_past_then_the_idlest_conversation_never_a_shared_prefix() {
    const CONVS: usize = 6;
    const ROOM: usize = 10;
    let mut rand = Rand::new(0x5E55_1015_0000_0001);
    let mut t: Prefix<K> = Prefix::new(4);
    // Each checkpoint's tokens and recency; each conversation's history.
    let mut model: Vec<(Vec<i64>, (u64, usize))> = Vec::new();
    let mut convs: Vec<Vec<i64>> = vec![vec![0, 1, 2]; CONVS];
    let mut fresh = 100i64;
    let (mut newest_taken, mut shared_seen) = (0, 0);
    for now in 1..20000u64 {
        match rand.next(8) {
            0 => {
                // A new conversation forks another at one of its checkpoints,
                // or at its whole history, abandoning the one it replaces.
                let (from, into) = (rand.next(CONVS), rand.next(CONVS));
                let on: Vec<&Vec<i64>> = model.iter().map(|(k, _)| k).filter(|k| convs[from].starts_with(k)).collect();
                convs[into] = match rand.next(on.len() + 1) {
                    i if i < on.len() => on[i].clone(),
                    _ => convs[from].clone(),
                };
            }
            r => {
                let s = rand.next(CONVS);
                let ext = 1 + rand.next(5);
                let tail = 1 + rand.next(2);
                let prompt: Vec<i64> =
                    convs[s].iter().copied().chain((0..ext + tail).map(|i| fresh + i as i64)).collect();
                fresh += 1000;
                let usable = &prompt[..prompt.len() - 1];
                let mut path: Vec<usize> = (0..model.len()).filter(|&i| usable.starts_with(&model[i].0)).collect();
                path.sort_by_key(|&i| model[i].0.len());
                let want = path.last().map(|&i| model[i].0.len());
                assert_eq!(t.lookup(&prompt).map(|h| h.len), want, "conversation {s} at {now}");
                for (depth, &i) in path.iter().enumerate() {
                    model[i].1 = (now, usize::MAX - 1 - depth);
                }
                let end = prompt.len() - tail;
                if end > want.unwrap_or(0) {
                    t.insert(&prompt[..end], K(end, true));
                    model.push((prompt[..end].to_vec(), (now, usize::MAX)));
                }
                convs[s] = prompt[..end].to_vec();
                while model.len() > ROOM || (r == 1 && !model.is_empty() && model.len() == ROOM) {
                    let roles = roles(&model.iter().map(|(k, _)| (&k[..], true)).collect::<Vec<_>>());
                    shared_seen += roles.iter().filter(|&&r| r == 2).count();
                    let coldest = |want: usize| {
                        let of = model.iter().zip(&roles).filter(|&(_, &r)| r == want);
                        of.min_by_key(|((_, at), _)| *at).map(|((k, _), _)| k.clone())
                    };
                    let want = coldest(1).or_else(|| coldest(0));
                    newest_taken += (coldest(1).is_none()) as usize;
                    let got = t.evict().map(|k| k.to_vec());
                    assert_eq!(got, want, "room at {now}");
                    model.retain(|(k, _)| Some(k) != got.as_ref());
                }
            }
        }
        let roles = roles(&model.iter().map(|(k, _)| (&k[..], true)).collect::<Vec<_>>());
        let superseded = roles.iter().filter(|&&r| r == 1).count();
        assert_eq!((t.entries(), t.superseded()), (model.len(), superseded), "at {now}");
    }
    // The run went through every case: a newest taken, shared prefixes passed over.
    assert!(newest_taken > 0 && shared_seen > 0, "newest taken {newest_taken}, shared seen {shared_seen}");
}
