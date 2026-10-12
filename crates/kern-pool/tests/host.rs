//! The host tier through its public API: parking takes the lowest free
//! host page and host slot, shares a page parked already, restores a
//! parked checkpoint into one lease on fresh device pages, and gives
//! everything back when the last holder drops.

mod common;

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use kern_pool::{runs, Checkpoint, Copies, Denied, Host, Lease, Parked, Pool};

use common::{hybrid4, hybrid_pool4, land, paged4, pool4, pool_of, Rand};

/// A block of `pages` host pages and two host slots.
fn host(pages: u64) -> Arc<Host> {
    Arc::new(Host::new(pages, 2))
}

#[test]
fn pages_and_slots_are_counted_apart() {
    let p = hybrid_pool4();
    let h = host(16);
    let cp = p.retire(p.lease(12).unwrap(), 12);
    let [d0, d1, d2] = cp.page_ids()[..] else { panic!() };
    let (q, plan) = h.park(&cp).unwrap();
    assert_eq!(plan, Copies { pages: vec![(d0, 0), (d1, 1), (d2, 2)], slot: Some((cp.seq_slot().unwrap(), 0)) });
    assert_eq!((q.tokens(), q.page_ids(), q.has_slot()), (12, vec![0, 1, 2], true));
    assert_eq!((h.used(), h.slots_used()), (3, 1));
    drop(q);
    assert_eq!((h.used(), h.slots_used()), (0, 0));
}

#[test]
fn a_page_parked_already_is_shared_not_copied() {
    let p = pool4();
    let h = host(16);
    let mut l = p.lease(16).unwrap();
    let (a, _) = p.checkpoint(&mut l, 8).unwrap();
    let (qa, plan) = h.park(&a).unwrap();
    assert_eq!(plan.pages.len(), 2);
    // The next turn of the session: two pages more, the first two on the host already.
    let (b, _) = p.checkpoint(&mut l, 16).unwrap();
    let (qb, plan) = h.park(&b).unwrap();
    assert_eq!(plan.pages, [(l.page_ids()[2], 2), (l.page_ids()[3], 3)]);
    assert_eq!((h.used(), qb.page_ids(), qa.page_ids()), (4, vec![0, 1, 2, 3], vec![0, 1]));
    // Parked twice: the same host pages, nothing copied.
    let (qb2, plan) = h.park(&b).unwrap();
    assert_eq!((plan, qb2.page_ids()), (Copies::default(), qb.page_ids()));
    drop((qa, qb2));
    // qb holds every page; nothing came back.
    assert_eq!(h.used(), 4);
    drop(qb);
    assert_eq!(h.used(), 0);
    // Gone from the host: parking a's pages again copies again.
    let (_, plan) = h.park(&a).unwrap();
    assert_eq!(plan.pages.len(), 2);
}

#[test]
fn full_keeps_nothing_and_the_lowest_free_unit_goes_first() {
    let p = pool_of(&hybrid4(), 4, 16, 4); // 4 slots, 8 pages
    let h = Arc::new(Host::new(3, 1));
    let mut l = p.lease(12).unwrap();
    let (a, _) = p.checkpoint(&mut l, 8).unwrap();
    let (qa, _) = h.park(&a).unwrap(); // 2 of 3 pages, the one slot
                                       // b's third page fits, a second slot does not.
    let (b, _) = p.checkpoint(&mut l, 12).unwrap();
    assert_eq!(h.park(&b).unwrap_err(), Denied::HostFull);
    // The third page went back; qa and its pages are untouched.
    assert_eq!((h.used(), h.slots_used(), qa.page_ids()), (2, 1, vec![0, 1]));
    drop((qa, l, a, b));
    assert_eq!((h.used(), h.slots_used()), (0, 0));
    // A slot-only checkpoint takes a slot and no page.
    let so = p.checkpoint(&mut p.lease_slot().unwrap(), 1).unwrap().0;
    let (s, plan) = h.park(&so).unwrap();
    assert_eq!(
        (plan.slot, plan.pages.len(), s.page_ids(), h.slots_used()),
        (Some((so.seq_slot().unwrap(), 0)), 0, vec![], 1)
    );
    drop(s);
    // Pages handed back in any order: the lowest is taken again first.
    let q = pool4();
    let mut leases: Vec<Lease> = (0..4).map(|_| q.lease(4).unwrap()).collect();
    let cps: Vec<Checkpoint> = leases.iter_mut().map(|l| q.checkpoint(l, 4).unwrap().0).collect();
    drop(leases);
    let h = Arc::new(Host::new(3, 0));
    let mut parked: Vec<Option<Parked>> = cps[..3].iter().map(|c| Some(h.park(c).unwrap().0)).collect();
    assert_eq!(h.park(&cps[3]).unwrap_err(), Denied::HostFull);
    parked[2] = None;
    parked[0] = None;
    let at = |c: &Checkpoint| h.park(c).unwrap().0.page_ids();
    assert_eq!((at(&cps[3]), at(&cps[3]), h.used()), (vec![0], vec![0], 1));
    let kept = [h.park(&cps[3]).unwrap().0, h.park(&cps[0]).unwrap().0];
    assert_eq!((kept[0].page_ids(), kept[1].page_ids(), h.used()), (vec![0], vec![2], 3));
}

#[test]
fn a_parked_checkpoint_restores_into_one_lease() {
    let p = pool4();
    let h = host(16);
    let mut l = p.lease(10).unwrap();
    let (cp, _) = p.checkpoint(&mut l, 10).unwrap(); // 3 pages, the third a copy
    let (q, _) = h.park(&cp).unwrap();
    drop((l, cp));
    assert_eq!(p.used(), 0);
    // One allocation: the pages the continuation needs, the first three copied in.
    let (mut w, plan) = h.restore(&q, &p, 10, 20).unwrap();
    assert_eq!((w.prefix(), w.pages(), w.tokens(), p.used()), (10, 5, 20, 5));
    assert_eq!(
        plan,
        Copies { pages: q.page_ids().into_iter().zip(w.page_ids().iter().copied()).collect(), slot: None }
    );
    // Its whole pages are sealed, twinned with the host's: the next
    // checkpoint parks them for free and copies only the half page.
    let (cp2, copies) = p.checkpoint(&mut w, 10).unwrap();
    assert_eq!((copies.pages.len(), &cp2.page_ids()[..2]), (1, &w.page_ids()[..2]));
    let (q2, plan) = h.park(&cp2).unwrap();
    assert_eq!((plan.pages.len(), &q2.page_ids()[..2], h.used()), (1, &q.page_ids()[..2], 4));
    // A stateless parked checkpoint restores at any whole page of it.
    let (w2, plan) = h.restore(&q, &p, 4, 8).unwrap();
    assert_eq!((w2.prefix(), w2.pages(), plan.pages.len()), (4, 2, 1));
    drop((q, q2, w, w2, cp2));
    assert_eq!((p.used(), h.used()), (0, 0));
}

#[test]
#[should_panic(expected = "restoring 4 tokens of a parked checkpoint of 10 (with a slot)")]
fn a_stateful_parked_checkpoint_restores_at_its_length_only() {
    let p = hybrid_pool4();
    let h = host(16);
    let cp = p.retire(p.lease(10).unwrap(), 10);
    let (q, plan) = h.park(&cp).unwrap();
    assert!(plan.slot.is_some());
    drop(cp);
    let (w, plan) = h.restore(&q, &p, 10, 11).unwrap();
    assert_eq!((w.seq_slot().is_some(), plan.slot.map(|(o, _)| o)), (true, Some(0)));
    let _ = h.restore(&q, &p, 4, 11);
}

#[test]
fn a_slot_only_parked_checkpoint_restores_to_a_slot_at_any_length() {
    let p = hybrid_pool4();
    let h = host(16);
    let cp = p.checkpoint(&mut p.lease_slot().unwrap(), 10).unwrap().0;
    let (q, _) = h.park(&cp).unwrap();
    drop(cp);
    let (w, plan) = h.restore(&q, &p, 10, 11).unwrap();
    assert_eq!(
        (w.pages(), w.prefix(), w.seq_slot().is_some(), plan.pages.len(), plan.slot.is_some()),
        (0, 10, true, 0, true)
    );
}

/// Bytes on the device and on the host, by position: a page is 4 tokens.
#[derive(Default)]
struct Memory {
    device: BTreeMap<i64, u32>,
    host: BTreeMap<u64, u32>,
}

impl Memory {
    fn park(&mut self, c: &Copies<i32, u64>) {
        for &(page, at) in &c.pages {
            for k in 0..4 {
                let v = self.device[&(page as i64 * 4 + k)];
                self.host.insert(at * 4 + k as u64, v);
            }
        }
    }

    fn wake(&mut self, c: &Copies<u64, i32>) {
        for &(at, page) in &c.pages {
            for k in 0..4 {
                let v = self.host[&(at * 4 + k as u64)];
                self.device.insert(page as i64 * 4 + k, v);
            }
        }
    }

    fn restore(&mut self, c: &Copies) {
        for &(s, d) in &c.pages {
            for k in 0..4 {
                if let Some(&v) = self.device.get(&(s as i64 * 4 + k)) {
                    self.device.insert(d as i64 * 4 + k, v);
                }
            }
        }
    }
}

/// Random checkpoints parked, restored from either tier and dropped with
/// every byte read back through every handle: a parked page is what the
/// checkpoint held when it was parked and what wakes is the same; the
/// host pages the parked handles name are exactly the ones held, and
/// every parked state holds a host slot of its own. Returns the most
/// states parked at once.
fn parked_bytes_come_back_and_name_the_units_held(p: Arc<Pool>, h: Arc<Host>) -> u64 {
    let mut most = 0;
    let mut mem = Memory::default();
    let mut seqs: Vec<(Lease, Vec<u32>)> = Vec::new();
    let mut cps: Vec<(Checkpoint, Vec<u32>)> = Vec::new();
    let mut parked: Vec<(Parked, Vec<u32>)> = Vec::new();
    let mut rand = Rand::new(0x1234_5678_9ABC_DEF1);
    let mut stamp = 0;
    for _ in 0..4000 {
        match rand.next(9) {
            0 => match p.lease(1 + rand.next(32)) {
                Ok(l) => {
                    let content: Vec<u32> = (0..l.tokens())
                        .map(|pos| {
                            stamp += 1;
                            mem.device.insert(l.slot(pos), stamp);
                            stamp
                        })
                        .collect();
                    seqs.push((l, content));
                }
                Err(Denied::Remapping) => land(&p),
                Err(_) => {}
            },
            1 | 2 if !seqs.is_empty() => {
                let i = rand.next(seqs.len());
                let (l, content) = &mut seqs[i];
                let len = 1 + rand.next(content.len());
                match p.checkpoint(l, len) {
                    Ok((cp, copies)) => {
                        mem.restore(&copies);
                        cps.push((cp, content[..len].to_vec()));
                    }
                    Err(Denied::Remapping) => land(&p),
                    Err(_) => {}
                }
            }
            3 | 4 if !cps.is_empty() => {
                let (cp, content) = &cps[rand.next(cps.len())];
                if let Ok((q, plan)) = h.park(cp) {
                    assert!(plan.pages.len() <= cp.page_ids().len());
                    assert_eq!(plan.slot.is_some(), cp.seq_slot().is_some());
                    mem.park(&plan);
                    parked.push((q, content.clone()));
                }
            }
            5 if !parked.is_empty() => {
                let (q, content) = &parked[rand.next(parked.len())];
                let whole = (q.tokens() - 1) / 4;
                let len = if q.has_slot() || whole == 0 || rand.next(2) == 0 {
                    q.tokens()
                } else {
                    (1 + rand.next(whole)) * 4
                };
                if len < 32 {
                    match h.restore(q, &p, len, len + 1 + rand.next(32 - len)) {
                        Ok((l, plan)) => {
                            mem.wake(&plan);
                            let mut content = content[..len].to_vec();
                            for pos in len..l.tokens() {
                                stamp += 1;
                                mem.device.insert(l.slot(pos), stamp);
                                content.push(stamp);
                            }
                            seqs.push((l, content));
                        }
                        Err(Denied::Remapping) => land(&p),
                        Err(_) => {}
                    }
                }
            }
            6 if !cps.is_empty() => {
                let (cp, content) = &cps[rand.next(cps.len())];
                let len = cp.tokens();
                if len < 32 {
                    match p.restore(cp, len, len + 1 + rand.next(32 - len)) {
                        Ok((l, copies)) => {
                            mem.restore(&copies);
                            let mut content = content.clone();
                            for pos in len..l.tokens() {
                                stamp += 1;
                                mem.device.insert(l.slot(pos), stamp);
                                content.push(stamp);
                            }
                            seqs.push((l, content));
                        }
                        Err(Denied::Remapping) => land(&p),
                        Err(_) => {}
                    }
                }
            }
            7 => match rand.next(3) {
                0 if !seqs.is_empty() => {
                    seqs.swap_remove(rand.next(seqs.len()));
                }
                1 if !cps.is_empty() => {
                    cps.swap_remove(rand.next(cps.len()));
                }
                2 if !parked.is_empty() => {
                    parked.swap_remove(rand.next(parked.len()));
                }
                _ => {}
            },
            _ => {}
        }
        let mut pages = BTreeSet::new();
        for (l, content) in &seqs {
            for (pos, &v) in content.iter().enumerate() {
                assert_eq!(mem.device[&(l.page_ids()[pos / 4] as i64 * 4 + (pos % 4) as i64)], v);
            }
            pages.extend(l.page_ids().iter().copied());
        }
        for (cp, content) in &cps {
            let ids = cp.page_ids();
            for (pos, &v) in content.iter().enumerate() {
                assert_eq!(mem.device[&(ids[pos / 4] as i64 * 4 + (pos % 4) as i64)], v);
            }
            pages.extend(ids);
        }
        let mut hosts = BTreeSet::new();
        for (q, content) in &parked {
            let ids = q.page_ids();
            for (pos, &v) in content.iter().enumerate() {
                assert_eq!(mem.host[&(ids[pos / 4] * 4 + (pos % 4) as u64)], v);
            }
            hosts.extend(ids);
        }
        let states = parked.iter().filter(|(q, _)| q.has_slot()).count() as u64;
        assert_eq!((p.used(), h.used(), h.slots_used()), (pages.len(), hosts.len() as u64, states));
        most = most.max(states);
    }
    most
}

#[test]
fn parked_kv_comes_back_and_names_the_pages_held() {
    assert_eq!(parked_bytes_come_back_and_name_the_units_held(pool_of(&paged4(), 4, 24, 0), host(12)), 0);
}

#[test]
fn parked_states_come_back_and_name_the_units_held() {
    let p = pool_of(&hybrid4(), 4, 48, 4);
    assert_eq!(parked_bytes_come_back_and_name_the_units_held(p, Arc::new(Host::new(12, 4))), 4);
}

#[test]
fn runs_are_the_pairs_folded_maximally_in_order() {
    let mut rand = Rand::new(0xC0FF_EE00_D15E_A5E5);
    for _ in 0..2000 {
        let mut pairs: Vec<(i32, u64)> = Vec::new();
        for _ in 0..rand.next(24) {
            let (p, o) = pairs.last().map_or((rand.next(8) as i32, rand.next(8) as u64), |&(p, o)| (p, o));
            // Mostly the next page at the next host page, sometimes a jump on either side.
            let next = match rand.next(4) {
                0 => (p + 1 + rand.next(3) as i32, o + 1),
                1 => (p + 1, o + 2 + rand.next(3) as u64),
                _ => (p + 1, o + 1),
            };
            pairs.push(if pairs.is_empty() { (p, o) } else { next });
        }
        let r = runs(&pairs);
        let expanded: Vec<(i32, u64)> =
            r.iter().flat_map(|&(p, o, n)| (0..n).map(move |i| (p + i as i32, o + i as u64))).collect();
        assert_eq!(expanded, pairs);
        assert!(r.iter().all(|&(_, _, n)| n >= 1));
        assert!(r.windows(2).all(|w| w[0].0 + w[0].2 as i32 != w[1].0 || w[0].1 + w[0].2 as u64 != w[1].1));
    }
}

#[test]
fn consecutive_pages_fold_into_one_copy() {
    assert_eq!(runs(&[(10, 0), (11, 1), (12, 2)]), [(10, 0, 3)]);
    assert_eq!(runs(&[(10, 0), (11, 1), (13, 2), (14, 4)]), [(10, 0, 2), (13, 2, 1), (14, 4, 1)]);
}
