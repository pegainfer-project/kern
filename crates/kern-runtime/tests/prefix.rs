//! The prefix table through its public API: entries grow along a chain,
//! a lookup finds the longest usable prefix, and parking moves an entry
//! to the host without losing it.

mod common;

use std::collections::BTreeMap;
use std::sync::Arc;

use kern_runtime::{Chain, Checkpoint, Hit, Host, Parked, Prefix, Tier};

use common::{hybrid_pool4, pool4, Rand};

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
    let p = pool4();
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
    let p = pool4();
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
    let p = pool4();
    let mut t = table();
    let mut l = p.lease(12).unwrap();
    let a = t.insert(&Chain::over(4, &toks(8)), p.checkpoint(&mut l, 8).unwrap().0);
    let mut other = toks(8);
    other[6] = -1;
    let c = t.insert(&Chain::over(4, &other), p.checkpoint(&mut l, 8).unwrap().0);
    // Both are usable at page 1; a hit at 9 tokens of `toks` is a's and touches c's shared page too.
    assert_eq!(t.lookup(&toks(9)).map(|h| h.id), Some(a));
    assert_eq!(t.coldest(Tier::Resident), Some(a));
    // Six tokens of `other` are usable at page 1 through either entry: the lower id.
    assert_eq!(t.lookup(&other[..7]).map(|h| (h.id, h.len)), Some((a, 4)));
    // A hit on c touches a (found at page 1) after it: c, the leaf, is the colder.
    assert_eq!(t.lookup(&[&other[..], &[0]].concat()).map(|h| (h.id, h.len)), Some((c, 8)));
    assert_eq!(t.coldest(Tier::Resident), Some(c));
    // Removing an entry frees only what no lease still holds.
    assert!(t.remove(c));
    assert_eq!((t.len(), p.used()), (1, 3));
    assert!(t.remove(a));
    assert_eq!((t.is_empty(), p.used(), t.coldest(Tier::Resident)), (true, 3, None));
    assert!(!t.remove(c));
    drop(l);
    assert_eq!(p.used(), 0);
}

#[test]
fn restore_then_checkpoint_deeper() {
    let p = pool4();
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
    let p = pool4();
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
    let p = hybrid_pool4();
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
    let p = pool4();
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
    let p = hybrid_pool4();
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

/// Random inserts, lookups and removals against a brute-force model
/// that keeps every entry's tokens: a lookup's length is the longest
/// usable prefix any entry holds, and every entry is findable at its
/// usable lengths and nowhere else.
#[test]
fn lookup_matches_the_brute_force_model() {
    let p = pool4();
    let mut t = table();
    let mut model: BTreeMap<u64, Vec<i64>> = BTreeMap::new();
    let mut rand = Rand::new(0x1234_5678_9ABC_DEF1);
    let mut lease = p.lease(32).unwrap();
    for _ in 0..600 {
        match rand.next(3) {
            0 => {
                // A random token sequence of up to 12 out of a 2-symbol alphabet: prefixes collide often.
                let n = 1 + rand.next(12);
                let tokens: Vec<i64> = (0..n).map(|_| rand.next(2) as i64).collect();
                if let Ok((cp, _)) = p.checkpoint(&mut lease, n) {
                    let id = t.insert(&Chain::over(4, &tokens), cp);
                    // A grown entry covers its old tokens too; only the longer sequence is kept.
                    model.retain(|&i, v| i != id || v.len() > tokens.len());
                    model.entry(id).or_insert(tokens);
                }
            }
            1 => {
                let n = 1 + rand.next(14);
                let tokens: Vec<i64> = (0..n).map(|_| rand.next(2) as i64).collect();
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
                let id = *model.keys().nth(rand.next(model.len())).unwrap();
                assert!(t.remove(id));
                model.remove(&id);
            }
            _ => {}
        }
        assert_eq!(t.len(), model.len());
    }
}
