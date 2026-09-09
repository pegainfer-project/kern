//! The host tier through its public API: parking places pages at the low
//! end and slots at the high end, shares a page parked already, and gives
//! everything back when the parked checkpoint drops.

mod common;

use std::sync::Arc;

use kern_runtime::{Denied, Host, Park, Parked};

use common::Rand;

fn host(units: u64) -> Arc<Host> {
    Arc::new(Host::new(units * 8, 8))
}

#[test]
fn pages_go_low_and_slots_high() {
    let h = host(16);
    // 3 pages of 8 bytes at the low end, one slot of 20 (3 units) at the high end.
    let (p, plan) = h.park(&[(1, 10), (2, 11), (3, 12)], 8, Some((5, 20)), 40).unwrap();
    assert_eq!(plan, Park { pages: vec![(10, 0), (11, 8), (12, 16)], slot: Some((5, 104)) });
    assert_eq!((p.tokens(), p.has_slot(), p.pages(3), p.slot()), (40, true, vec![0, 8, 16], Some(104)));
    assert_eq!((h.used(), h.pages()), (48, 3));
    drop(p);
    assert_eq!((h.used(), h.pages()), (0, 0));
}

#[test]
fn a_page_parked_already_is_shared_not_copied() {
    let h = host(16);
    let (a, _) = h.park(&[(1, 10), (2, 11)], 8, None, 32).unwrap();
    // The next turn of the session: two pages more, the first two on the host already.
    let (b, plan) = h.park(&[(1, 10), (2, 11), (3, 12), (4, 13)], 8, None, 64).unwrap();
    assert_eq!(plan.pages, [(12, 16), (13, 24)]);
    assert_eq!((h.pages(), b.pages(4), a.pages(2)), (4, vec![0, 8, 16, 24], vec![0, 8]));
    drop(a);
    // b holds every page; nothing came back.
    assert_eq!((h.used(), h.pages()), (32, 4));
    drop(b);
    assert_eq!((h.used(), h.pages()), (0, 0));
    // Gone from the registry: parking node 1 again copies again.
    let (_, plan) = h.park(&[(1, 10)], 8, None, 16).unwrap();
    assert_eq!(plan.pages, [(10, 0)]);
}

#[test]
fn full_keeps_nothing_and_frees_make_the_block_whole_again() {
    let h = host(4);
    let (a, _) = h.park(&[(1, 0), (2, 1)], 8, None, 32).unwrap();
    // Node 3's page fits, a two-unit slot does not.
    assert_eq!(h.park(&[(1, 0), (2, 1), (3, 2)], 8, Some((1, 16)), 48).unwrap_err(), Denied::HostFull);
    // Node 3's page went back; a and its two pages are untouched.
    assert_eq!((h.used(), h.pages()), (16, 2));
    drop(a);
    assert_eq!(h.park(&[(7, 0)], 40, None, 16).unwrap_err(), Denied::HostFull);
    assert_eq!(h.used(), 0);
    // Three pages dropped in any order leave one run: a slot the size of the block fits.
    let h = host(3);
    let (a, _) = h.park(&[(1, 0)], 8, None, 16).unwrap();
    let (b, _) = h.park(&[(2, 1)], 8, None, 16).unwrap();
    let (c, _) = h.park(&[(3, 2)], 8, None, 16).unwrap();
    drop(a);
    drop(c);
    drop(b);
    let (s, plan) = h.park(&[], 8, Some((0, 24)), 1).unwrap();
    assert_eq!((plan.slot, h.used()), (Some((0, 0)), 24));
    drop(s);
    // Bytes are rounded up to the grain.
    let (d, plan) = h.park(&[(4, 0)], 9, None, 16).unwrap();
    assert_eq!((plan.pages, h.used()), (vec![(0, 0)], 16));
    drop(d);
}

/// Random parks and drops against a model that only tracks which
/// checkpoints are alive: used bytes are exactly the live pages and
/// slots, and every live range is disjoint from every other.
#[test]
fn accounting_partitions_the_block() {
    let h = host(64);
    let mut live: Vec<(Parked, usize, u64)> = Vec::new();
    let mut rand = Rand::new(0x2545_F491_4F6C_DD1D);
    for _ in 0..3000 {
        if rand.next(3) > 0 || live.is_empty() {
            // A chain of up to 6 nodes, two candidate ids a depth (shared prefixes), maybe a slot.
            let n = 1 + rand.next(6);
            let nodes: Vec<(u64, i32)> = (0..n).map(|i| (2 * i as u64 + 1 + rand.next(2) as u64, i as i32)).collect();
            let slot = (rand.next(2) == 0).then(|| (0, 8 + rand.next(3) as u64 * 8));
            if let Ok((p, plan)) = h.park(&nodes, 8, slot, n * 4) {
                assert!(plan.pages.len() <= n);
                live.push((p, n, slot.map_or(0, |(_, b)| b)));
            }
        } else {
            live.swap_remove(rand.next(live.len()));
        }
        let mut ranges: Vec<(u64, u64)> = Vec::new();
        for (p, n, slot) in &live {
            ranges.extend(p.pages(*n).into_iter().map(|o| (o, 8)));
            ranges.extend(p.slot().map(|o| (o, *slot)));
        }
        ranges.sort();
        ranges.dedup();
        for w in ranges.windows(2) {
            assert!(w[0].0 + w[0].1 <= w[1].0, "overlap {:?} {:?}", w[0], w[1]);
        }
        let held: u64 = ranges.iter().map(|r| r.1).sum();
        let pages = ranges.len() - live.iter().filter(|l| l.0.has_slot()).count();
        assert_eq!((h.used(), h.pages()), (held, pages));
    }
}
