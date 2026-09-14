//! The host tier through its public API: parking places pages at the low
//! end and slots at the high end, shares a page parked already, wakes a
//! checkpoint back onto fresh device pages, and gives everything back
//! when the last holder drops.

mod common;

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use kern_pool::{runs, Checkpoint, Copies, Denied, Host, Lease, Parked, Pool};

use common::{hybrid4, hybrid_pool4, land, paged4, pool4, pool_of, Rand};

/// A block of `pages` 4-byte pages and a slot of 8 bytes, in 4-byte grains.
fn host(pages: u64) -> Arc<Host> {
    Arc::new(Host::new(pages * 4, 4, 4, 8))
}

#[test]
fn pages_go_low_and_slots_high() {
    let p = hybrid_pool4();
    let h = host(16);
    let cp = p.retire(p.lease(12).unwrap(), 12);
    let [d0, d1, d2] = cp.page_ids()[..] else { panic!() };
    let (q, plan) = h.park(&cp).unwrap();
    assert_eq!(plan, Copies { pages: vec![(d0, 0), (d1, 4), (d2, 8)], slot: Some((cp.seq_slot().unwrap(), 56)) });
    assert_eq!((q.tokens(), q.offsets(), q.has_slot()), (12, vec![0, 4, 8], true));
    assert_eq!(h.used(), 20);
    drop(q);
    assert_eq!(h.used(), 0);
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
    assert_eq!(plan.pages, [(l.page_ids()[2], 8), (l.page_ids()[3], 12)]);
    assert_eq!((h.used(), qb.offsets(), qa.offsets()), (16, vec![0, 4, 8, 12], vec![0, 4]));
    // Parked twice: the same host pages, nothing copied.
    let (qb2, plan) = h.park(&b).unwrap();
    assert_eq!((plan, qb2.offsets()), (Copies::default(), qb.offsets()));
    drop((qa, qb2));
    // qb holds every page; nothing came back.
    assert_eq!(h.used(), 16);
    drop(qb);
    assert_eq!(h.used(), 0);
    // Gone from the host: parking a's pages again copies again.
    let (_, plan) = h.park(&a).unwrap();
    assert_eq!(plan.pages.len(), 2);
}

#[test]
fn full_keeps_nothing_and_frees_make_the_block_whole_again() {
    let p = pool_of(&hybrid4(), 4, 16, 4); // 4 slots, 8 pages
    let h = host(6);
    let mut l = p.lease(12).unwrap();
    let (a, _) = p.checkpoint(&mut l, 8).unwrap();
    let (qa, _) = h.park(&a).unwrap(); // 2 pages and a slot: 16 of 24 bytes
                                       // b's third page fits, a second slot does not.
    let (b, _) = p.checkpoint(&mut l, 12).unwrap();
    assert_eq!(h.park(&b).unwrap_err(), Denied::HostFull);
    // The third page went back; qa and its pages are untouched.
    assert_eq!(h.used(), 16);
    drop(qa);
    assert_eq!(h.used(), 0);
    drop((l, a, b));
    // Three pages dropped in any order leave one run: a slot the size of the block fits.
    let q = pool4();
    let mut leases: Vec<Lease> = (0..3).map(|_| q.lease(4).unwrap()).collect();
    let cps: Vec<Checkpoint> = leases.iter_mut().map(|l| q.checkpoint(l, 4).unwrap().0).collect();
    drop(leases);
    let h = Arc::new(Host::new(12, 4, 4, 12));
    let mut parked: Vec<Parked> = cps.iter().map(|c| h.park(c).unwrap().0).collect();
    assert_eq!(h.used(), 12);
    parked.swap_remove(0);
    parked.swap_remove(1);
    parked.swap_remove(0);
    let so = p.checkpoint(&mut p.lease_slot().unwrap(), 1).unwrap().0;
    let (s, plan) = h.park(&so).unwrap();
    assert_eq!(
        (plan.slot, plan.pages.len(), h.used(), s.offsets()),
        (Some((so.seq_slot().unwrap(), 0)), 0, 12, vec![])
    );
    drop(s);
    // Bytes are rounded up to the grain: a 5-byte page takes two.
    let h = Arc::new(Host::new(16, 4, 5, 0));
    let (s, _) = h.park(&cps[0]).unwrap();
    assert_eq!(h.used(), 8);
    drop(s);
}

#[test]
fn wake_is_a_resident_checkpoint_again() {
    let p = pool4();
    let h = host(16);
    let mut l = p.lease(10).unwrap();
    let (cp, _) = p.checkpoint(&mut l, 10).unwrap(); // 3 pages, the third a copy
    let (q, _) = h.park(&cp).unwrap();
    drop((l, cp));
    assert_eq!(p.used(), 0);
    let (w, plan) = h.wake(&q, &p, 10).unwrap();
    assert_eq!((w.tokens(), w.page_ids().len(), w.has_slot(), p.used()), (10, 3, false, 3));
    assert_eq!(plan, Copies { pages: q.offsets().into_iter().zip(w.page_ids()).collect(), slot: None });
    // A woken checkpoint restores like any other.
    let (l2, copies) = p.restore(&w, 10, 20).unwrap();
    assert_eq!((l2.prefix(), &l2.page_ids()[..2], copies.pages.len()), (10, &w.page_ids()[..2], 1));
    // And parks for free: its pages are the host's already.
    let (q2, plan) = h.park(&w).unwrap();
    assert_eq!((plan, q2.offsets(), h.used()), (Copies::default(), q.offsets(), 12));
    // A stateless parked checkpoint wakes at any whole page of it.
    let (w2, plan) = h.wake(&q, &p, 4).unwrap();
    assert_eq!((w2.tokens(), w2.page_ids().len(), plan.pages.len()), (4, 1, 1));
    drop((q, q2, w, w2, l2));
    assert_eq!((p.used(), h.used()), (0, 0));
}

#[test]
#[should_panic(expected = "waking 4 tokens of a parked checkpoint of 10 (with a slot)")]
fn a_stateful_parked_checkpoint_wakes_at_its_length_only() {
    let p = hybrid_pool4();
    let h = host(16);
    let cp = p.retire(p.lease(10).unwrap(), 10);
    let (q, plan) = h.park(&cp).unwrap();
    assert!(plan.slot.is_some());
    drop(cp);
    let (w, plan) = h.wake(&q, &p, 10).unwrap();
    assert_eq!((w.has_slot(), plan.slot.map(|(o, _)| o)), (true, Some(56)));
    let _ = h.wake(&q, &p, 4);
}

/// Bytes on the device and on the host, by position and by offset.
#[derive(Default)]
struct Memory {
    device: BTreeMap<i64, u32>,
    host: BTreeMap<u64, u32>,
}

impl Memory {
    fn park(&mut self, c: &Copies<i32, u64>) {
        for &(page, off) in &c.pages {
            for k in 0..4 {
                let v = self.device[&(page as i64 * 4 + k)];
                self.host.insert(off + k as u64, v);
            }
        }
    }

    fn wake(&mut self, c: &Copies<u64, i32>) {
        for &(off, page) in &c.pages {
            for k in 0..4 {
                let v = self.host[&(off + k as u64)];
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

/// Random checkpoints parked, woken, restored and dropped with every
/// byte read back through every handle: a parked page is what the
/// checkpoint held when it was parked and what wakes is the same; the
/// block holds exactly the pages the parked handles name, disjoint.
#[test]
fn parked_bytes_come_back_and_partition_the_block() {
    let p: Arc<Pool> = pool_of(&paged4(), 4, 24, 0);
    let h = host(12);
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
                    mem.park(&plan);
                    parked.push((q, content.clone()));
                }
            }
            5 if !parked.is_empty() => {
                let (q, content) = &parked[rand.next(parked.len())];
                let whole = (q.tokens() - 1) / 4;
                let len = if whole == 0 || rand.next(2) == 0 { q.tokens() } else { (1 + rand.next(whole)) * 4 };
                match h.wake(q, &p, len) {
                    Ok((cp, plan)) => {
                        mem.wake(&plan);
                        cps.push((cp, content[..len].to_vec()));
                    }
                    Err(Denied::Remapping) => land(&p),
                    Err(_) => {}
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
        let mut offsets = BTreeSet::new();
        for (q, content) in &parked {
            let offs = q.offsets();
            for (pos, &v) in content.iter().enumerate() {
                assert_eq!(mem.host[&(offs[pos / 4] + (pos % 4) as u64)], v);
            }
            offsets.extend(offs);
        }
        assert_eq!((p.used(), h.used()), (pages.len(), offsets.len() as u64 * 4));
    }
}

#[test]
fn consecutive_pages_fold_into_one_copy() {
    assert_eq!(runs(&[(10, 0), (11, 8), (12, 16)], 8), [(10, 0, 3)]);
    assert_eq!(runs(&[(10, 0), (11, 8), (13, 16), (14, 32)], 8), [(10, 0, 2), (13, 16, 1), (14, 32, 1)]);
}
