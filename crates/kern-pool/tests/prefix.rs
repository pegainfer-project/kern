//! The prefix index through its public API: a lookup finds the longest
//! usable prefix, entries hold their checkpoints, and parking moves an
//! entry to the host without losing it.

mod common;

use std::sync::Arc;

use kern_pool::{Checkpoint, Host, Kept, Parked, Prefix, Tier};

use common::{hybrid_pool4, paged4, pool4, pool_of, Rand};

fn table() -> Prefix<Checkpoint, Parked> {
    Prefix::new(4)
}

fn toks(n: usize) -> Vec<i64> {
    (0..n as i64).map(|i| i * 7 + 3).collect()
}

fn host() -> Arc<Host> {
    Arc::new(Host::new(64, 4, 4, 8))
}

/// Park the entry at `key` into `h`, no copies run.
fn park(t: &mut Prefix, h: &Arc<Host>, key: &[i64]) {
    let parked = t.park(key, |cp| h.park(&cp).map(|(p, _)| Ok(p))).unwrap();
    assert!(parked);
}

#[test]
fn one_entry_grows_a_page_at_a_time() {
    let p = pool4();
    let mut t = table();
    let mut l = p.lease(12).unwrap();
    t.insert(&toks(4), p.checkpoint(&mut l, 4).unwrap().0);
    t.insert(&toks(8), p.checkpoint(&mut l, 8).unwrap().0);
    assert_eq!((t.len(), t.resident(&toks(4)).is_none()), (1, true));
    // Ten tokens end inside the third page: a second entry with a tail.
    t.insert(&toks(10), p.checkpoint(&mut l, 10).unwrap().0);
    assert_eq!(t.len(), 2);
    // A 12-token prompt of the same tokens may use 11: the 10-token entry.
    let h = t.lookup(&toks(12)).unwrap();
    assert_eq!((h.len, h.tier(), h.resident().map(|c| c.tokens())), (10, Tier::Resident, Some(10)));
    // A 10-token prompt may use 9: 8 whole pages, the 8-token entry itself.
    let h = t.lookup(&toks(10)).unwrap();
    assert_eq!((h.len, h.resident().map(|c| c.tokens())), (8, Some(8)));
    // Diverging inside the third page: still 8.
    let mut d = toks(12);
    d[9] = -1;
    assert_eq!(t.lookup(&d).map(|h| h.len), Some(8));
    // Diverging inside the second page: the entry is usable at its first page.
    d[5] = -1;
    let h = t.lookup(&d).unwrap();
    assert_eq!((h.len, h.resident().map(|c| c.tokens())), (4, Some(8)));
    d[0] = -1;
    assert_eq!(t.lookup(&d).map(|h| h.len), None);
    assert!(t.lookup(&toks(1)).is_none());
    assert!(t.lookup(&[]).is_none());
    // Growing past the tailed entry: page 3 replaces the 8-token entry, not the 10.
    t.insert(&toks(12), p.checkpoint(&mut l, 12).unwrap().0);
    assert_eq!((t.len(), t.resident(&toks(8)).is_none(), t.resident(&toks(10)).is_some()), (2, true, true));
    assert_eq!(t.lookup(&toks(13)).map(|h| h.len), Some(12));
}

#[test]
fn same_tokens_share_one_entry() {
    let p = pool4();
    let mut t = table();
    let mut l = p.lease(8).unwrap();
    t.insert(&toks(8), p.checkpoint(&mut l, 8).unwrap().0);
    assert_eq!(p.used(), 2);
    t.insert(&toks(8), p.checkpoint(&mut l, 8).unwrap().0);
    assert_eq!((t.len(), p.used()), (1, 2));
    // Same pages but different tokens is a different entry.
    let mut other = toks(8);
    other[7] = -1;
    t.insert(&other, p.checkpoint(&mut l, 8).unwrap().0);
    assert_eq!(t.len(), 2);
}

#[test]
fn coldest_is_least_recent_and_a_hit_touches_its_path() {
    let p = pool4();
    let mut t = table();
    let mut l = p.lease(12).unwrap();
    let a = toks(8);
    t.insert(&a, p.checkpoint(&mut l, 8).unwrap().0);
    let mut c = toks(8);
    c[6] = -1;
    t.insert(&c, p.checkpoint(&mut l, 8).unwrap().0);
    assert_eq!(t.coldest(Tier::Resident).as_deref(), Some(&a[..]));
    // A hit at 9 tokens of `a` touches a alone: c is the colder.
    assert_eq!(t.lookup(&toks(9)).map(|h| h.len), Some(8));
    assert_eq!(t.coldest(Tier::Resident).as_deref(), Some(&c[..]));
    // Six tokens of c are usable at page 1 through either entry; the
    // one found is touched.
    assert_eq!(t.lookup(&c[..7]).map(|h| h.len), Some(4));
    assert_eq!(t.coldest(Tier::Resident).as_deref(), Some(&a[..]));
    assert_eq!(t.lookup(&[&c[..], &[0]].concat()).map(|h| h.len), Some(8));
    assert_eq!(t.coldest(Tier::Resident).as_deref(), Some(&a[..]));
    // Removing an entry frees only what no lease still holds.
    assert!(t.remove(&c));
    assert_eq!((t.len(), p.used()), (1, 3));
    assert!(t.remove(&a));
    assert_eq!((t.is_empty(), p.used(), t.coldest(Tier::Resident)), (true, 3, None));
    assert!(!t.remove(&c));
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
    let (mut l2, _) = p.restore(hit.resident().unwrap(), hit.len, 16).unwrap();
    assert_eq!(l2.prefix(), 8);
    // The sequence's checkpoints grow the entry a page at a time.
    t.insert(&toks(12), p.checkpoint(&mut l2, 12).unwrap().0);
    t.insert(&toks(16), p.checkpoint(&mut l2, 16).unwrap().0);
    drop(l2);
    assert_eq!(t.lookup(&toks(17)).map(|h| h.len), Some(16));
    assert_eq!(t.lookup(&toks(13)).map(|h| h.len), Some(12));
    assert_eq!((t.len(), p.used()), (1, 4));
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
    let (mut x, _) = p.restore(hit.resident().unwrap(), hit.len, 12).unwrap();
    let (mut y, _) = p.restore(hit.resident().unwrap(), hit.len, 12).unwrap();
    let mut ty = toks(12);
    ty[9] = -1;
    t.insert(&toks(12), p.checkpoint(&mut x, 12).unwrap().0);
    t.insert(&ty, p.checkpoint(&mut y, 12).unwrap().0);
    assert_eq!((t.len(), t.resident(&toks(8)).is_none()), (2, true));
    assert_eq!(t.lookup(&toks(13)).map(|h| h.len), Some(12));
    assert_eq!(t.lookup(&[&ty[..], &[0]].concat()).map(|h| h.len), Some(12));
    // Both usable at 8 whole tokens; the tree's first branch wins so the choice is stable.
    let h = t.lookup(&toks(9)).unwrap();
    assert_eq!((h.len, h.resident().map(|c| c.tokens())), (8, Some(12)));
}

#[test]
fn a_stateful_entry_is_usable_at_its_length_only() {
    let p = hybrid_pool4();
    let mut t = table();
    let l = p.lease(12).unwrap();
    t.insert(&toks(10), p.retire(l, 10));
    let h = t.lookup(&toks(12)).unwrap();
    assert_eq!((h.len, h.tier()), (10, Tier::Resident));
    assert!(t.lookup(&toks(10)).is_none());
    let mut d = toks(12);
    d[9] = -1;
    assert!(t.lookup(&d).is_none());
    // A longer retirement on the same path is a second entry, not a growth.
    let l = p.lease(16).unwrap();
    t.insert(&toks(12), p.retire(l, 12));
    assert_eq!(t.len(), 2);
    assert_eq!(t.lookup(&toks(13)).map(|h| h.len), Some(12));
    assert_eq!(t.lookup(&toks(11)).map(|h| h.len), Some(10));
}

#[test]
fn parked_entries_are_found_after_resident_ones() {
    let p = pool4();
    let h = host();
    let mut t = table();
    let mut l = p.lease(12).unwrap();
    let key = toks(12);
    t.insert(&key, p.checkpoint(&mut l, 12).unwrap().0);
    drop(l);
    assert_eq!(p.used(), 3);
    park(&mut t, &h, &key);
    assert_eq!((p.used(), t.count(Tier::Parked), t.coldest(Tier::Resident)), (0, 1, None));
    let hit = t.lookup(&toks(13)).unwrap();
    assert_eq!((hit.len, hit.tier(), hit.parked().map(|q| q.tokens())), (12, Tier::Parked, Some(12)));
    assert_eq!(t.lookup(&toks(6)).map(|h| (h.len, h.tier())), Some((4, Tier::Parked)));
    assert_eq!(t.parked(&key).map(|q| q.offsets()), Some(vec![0, 4, 8]));
    // A hit outlives what the index does to the entry.
    assert!(t.remove(&key));
    assert_eq!((t.len(), h.pages()), (0, 3));
    drop(hit);
    assert_eq!(h.pages(), 0);
    // Resident and parked: the resident one is found first.
    let mut l = p.lease(12).unwrap();
    t.insert(&key, p.checkpoint(&mut l, 12).unwrap().0);
    park(&mut t, &h, &key);
    t.insert(&toks(8), p.checkpoint(&mut l, 8).unwrap().0);
    assert_eq!(t.lookup(&toks(9)).map(|h| (h.len, h.tier())), Some((8, Tier::Resident)));
    assert_eq!(t.lookup(&toks(13)).map(|h| (h.len, h.tier())), Some((12, Tier::Parked)));
    assert_eq!(
        (t.coldest(Tier::Parked).as_deref(), t.coldest(Tier::Resident).as_deref()),
        (Some(&key[..]), Some(&toks(8)[..]))
    );
    // The same tokens resident again: one entry in both tiers, found
    // resident, and one page past the 8-token entry, which it replaces.
    t.insert(&key, p.checkpoint(&mut l, 12).unwrap().0);
    assert_eq!((t.len(), t.count(Tier::Resident), t.count(Tier::Parked)), (1, 1, 0));
    assert_eq!(t.lookup(&toks(13)).map(|h| (h.len, h.tier())), Some((12, Tier::Resident)));
    assert_eq!(t.lookup(&toks(9)).map(|h| (h.len, h.tier())), Some((8, Tier::Resident)));
    // Parking it again copies nothing and keeps the host copy.
    let mut copied = None;
    assert!(t
        .park(&key, |cp| {
            copied = Some(cp.tokens());
            Ok::<_, ()>(Err(cp))
        })
        .unwrap());
    assert_eq!((copied, t.count(Tier::Parked), h.pages()), (None, 1, 3));
    assert!(t.remove(&key));
    assert_eq!((h.used(), t.len()), (0, 0));
}

#[test]
fn a_stateful_park_keeps_its_slot_and_its_length() {
    let p = hybrid_pool4();
    let h = host();
    let mut t = table();
    let l = p.lease(12).unwrap();
    t.insert(&toks(10), p.retire(l, 10));
    assert_eq!((p.used(), p.slots_used()), (3, 1));
    park(&mut t, &h, &toks(10));
    assert_eq!((p.used(), p.slots_used(), h.used()), (0, 0, 20));
    assert_eq!(t.lookup(&toks(12)).map(|h| (h.len, h.tier())), Some((10, Tier::Parked)));
    assert!(t.lookup(&toks(9)).is_none());
    assert!(t.parked(&toks(10)).unwrap().has_slot());
}

#[test]
fn a_park_that_fails_drops_the_entry_and_one_refused_keeps_it() {
    let p = pool4();
    let h = Arc::new(Host::new(4, 4, 4, 0));
    let mut t = table();
    let mut l = p.lease(8).unwrap();
    t.insert(&toks(8), p.checkpoint(&mut l, 8).unwrap().0);
    // No room: the checkpoint is handed back and stays resident.
    let kept = t.park(&toks(8), |cp| match h.park(&cp) {
        Ok((q, _)) => Ok::<_, ()>(Ok(q)),
        Err(_) => Ok(Err(cp)),
    });
    assert_eq!((kept, t.count(Tier::Resident)), (Ok(false), 1));
    // The copy failed: the entry is gone.
    assert_eq!(t.park(&toks(8), |_| Err("copy failed")), Err("copy failed"));
    assert!(t.is_empty());
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

/// Random inserts, lookups, parks and removals against a brute-force
/// model that keeps every entry's tokens: a lookup's length is the
/// longest usable prefix any entry holds, a resident entry before a
/// parked one, and the counts agree. Run twice, the index makes the
/// same decisions in the same order.
#[test]
fn lookup_matches_the_brute_force_model() {
    let trace = |seed: u64| {
        let mut t: Prefix<K, K> = Prefix::new(4);
        let mut model: Vec<(Vec<i64>, bool, Tier)> = Vec::new();
        let mut rand = Rand::new(seed);
        let mut out: Vec<(usize, Tier)> = Vec::new();
        for _ in 0..3000 {
            match rand.next(5) {
                0 | 1 => {
                    // Up to 12 tokens out of a 2-symbol alphabet: prefixes collide often.
                    let n = 1 + rand.next(12);
                    let tokens: Vec<i64> = (0..n).map(|_| rand.next(2) as i64).collect();
                    // Whether a state is held is the sequence's, not the insert's.
                    let slot = tokens.iter().sum::<i64>() % 3 == 0;
                    t.insert(&tokens, K(n, slot));
                    match model.iter_mut().find(|(v, _, _)| *v == tokens) {
                        Some(e) => e.2 = Tier::Resident,
                        None => {
                            model.push((tokens.clone(), slot, Tier::Resident));
                            // One page past a stateless resident-only entry replaces it.
                            if !slot && n > 4 && n.is_multiple_of(4) {
                                model.retain(|(v, s, tier)| {
                                    !(*v == tokens[..n - 4] && !s && *tier == Tier::Resident) || t.parked(v).is_some()
                                });
                            }
                        }
                    }
                }
                2 => {
                    let n = 1 + rand.next(14);
                    let tokens: Vec<i64> = (0..n).map(|_| rand.next(2) as i64).collect();
                    let usable = &tokens[..n - 1];
                    let want = model
                        .iter()
                        .map(|(v, slot, tier)| {
                            // Usable: all of v when the prompt covers it, else its whole pages when stateless.
                            let common = v.iter().zip(usable).take_while(|(a, b)| a == b).count();
                            let len = if common == v.len() {
                                v.len()
                            } else if *slot {
                                0
                            } else {
                                common / 4 * 4
                            };
                            (len, std::cmp::Reverse(*tier))
                        })
                        .filter(|(len, _)| *len > 0)
                        .max();
                    let got = t.lookup(&tokens).map(|h| (h.len, std::cmp::Reverse(h.tier())));
                    assert_eq!(got, want, "prompt {tokens:?}");
                    if let Some((len, std::cmp::Reverse(tier))) = got {
                        out.push((len, tier));
                    }
                }
                3 if !model.is_empty() => {
                    let i = rand.next(model.len());
                    if model[i].2 == Tier::Resident {
                        let (v, slot) = (model[i].0.clone(), model[i].1);
                        assert_eq!(t.park(&v, |r| Ok::<_, ()>(Ok(K(r.0, slot)))), Ok(true));
                        model[i].2 = Tier::Parked;
                    }
                }
                4 if !model.is_empty() => {
                    let (v, _, _) = model.swap_remove(rand.next(model.len()));
                    assert!(t.remove(&v));
                }
                _ => {}
            }
            let parked = model.iter().filter(|(_, _, tier)| *tier == Tier::Parked).count();
            assert_eq!((t.len(), t.count(Tier::Parked)), (model.len(), parked));
            out.push((t.coldest(Tier::Resident).map_or(0, |k| k.len()), Tier::Resident));
        }
        out
    };
    assert_eq!(trace(0x1234_5678_9ABC_DEF1), trace(0x1234_5678_9ABC_DEF1));
    let _ = pool_of(&paged4(), 4, 8, 0);
}
