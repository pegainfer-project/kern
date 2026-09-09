//! The pool through its public API: leases, checkpoints, restores, forks,
//! retirements and the remaps that move chunks between pages and slots.

mod common;

use std::sync::Arc;

use kern_manifest::types::{Dim, Manifest};
use kern_runtime::{Checkpoint, Copies, Denied, Kind, Lease, Pool};

use common::{hybrid, hybrid_pool, land, pool, pool_of, two_paged};

/// The pages a checkpoint holds, root first.
fn pages_of(cp: &Checkpoint) -> Vec<i32> {
    cp.nodes().into_iter().map(|(_, page)| page).collect()
}

#[test]
fn a_new_pool_maps_every_chunk_in_its_first_plan() {
    let p = pool();
    assert_eq!((p.total(), p.unit(), p.max_seq_tokens(), p.chunk()), (4, 16, 48, 8));
    assert_eq!(p.tables().collect::<Vec<_>>(), ["block_table", "draft_block_table"]);
    assert_eq!((p.has_slots(), p.slots()), (false, 0));
    assert_eq!(p.lease(1).unwrap().seq_slot(), None);
    // Two chunks over what four pages take are spare, not a torn page.
    let (p2, plan) = Pool::new(&two_paged(), 8, 18, 0).unwrap();
    assert_eq!((p2.total(), p2.pages_max(), plan.map.len(), plan.unmap.len()), (4, 4, 16, 0));
    assert_eq!(plan.made, [(Kind::Page, 0), (Kind::Page, 1), (Kind::Page, 2), (Kind::Page, 3)]);
    let names: Vec<(&str, Kind, u64, usize, usize)> =
        p2.pooled().iter().map(|a| (a.state.as_str(), a.kind, a.object, a.objects, a.positions)).collect();
    assert_eq!(names, [("draft_kv", Kind::Page, 16, 4, 8), ("kv", Kind::Page, 16, 4, 8)]);
    // With slots: every chunk mapped, one access grant per object, nothing unmapped.
    let (p, plan) = Pool::new(&hybrid(), 8, 20, 4).unwrap();
    assert_eq!((plan.map.len(), plan.access.len(), plan.unmap.len(), plan.unmade.len()), (20, 8, 0, 0));
    assert_eq!(plan.made.len(), 8);
    assert_eq!((p.total(), p.slots(), p.pages_max(), p.slots_max()), (4, 4, 10, 6));
    let arenas: Vec<(&str, Kind, usize)> = p.pooled().iter().map(|a| (a.state.as_str(), a.kind, a.positions)).collect();
    assert_eq!(arenas, [("kv", Kind::Page, 20), ("gdn", Kind::Slot, 18)]);
}

#[test]
fn pool_rejects_a_manifest_it_cannot_lay_out() {
    let err = |m: &Manifest, chunks: u32, first_slots: usize| match Pool::new(m, 8, chunks, first_slots) {
        Ok(_) => panic!("laid out"),
        Err(e) => e.to_string(),
    };
    // A state is paged or per sequence, not both.
    let mut m = hybrid();
    m.states.get_mut("gdn").unwrap().bytes = 8;
    assert!(err(&m, 20, 4).contains("one layout"));
    // The line table's shape is [lines, seqs, w] with the state's line count.
    let mut m = hybrid();
    let b = m.buffers.get_mut("line_index").unwrap();
    b.shape = vec![Dim::Const(4), Dim::Var("seqs".into())];
    assert!(err(&m, 20, 4).contains("state holds 3"));
    let b = m.buffers.get_mut("line_index").unwrap();
    b.shape = vec![Dim::Var("seqs".into())];
    assert!(err(&m, 20, 4).contains("[lines, seqs, w]"));
    let b = m.buffers.get_mut("line_index").unwrap();
    b.shape = vec![Dim::Const(3), Dim::Var("seqs".into()), Dim::Const(8)];
    let p = pool_of(&m, 8, 20, 4);
    let a = p.lease(16).unwrap();
    assert_eq!((a.seq_width("line_index").unwrap(), a.seq_lines("line_index").unwrap()), (8, 3));
    // The budget must hold the first slots.
    assert!(err(&hybrid(), 5, 4).contains("hold 1 sequence slots"));
}

#[test]
fn drop_returns_pages() {
    let p = pool();
    let a = p.lease(17).unwrap(); // 2 pages
    let b = p.lease(1).unwrap();
    assert_eq!((a.pages(), a.tokens(), b.pages(), p.used()), (2, 32, 1, 3));
    drop(a);
    assert_eq!(p.used(), 1);
    drop(b);
    assert_eq!(p.used(), 0);
    let all: Vec<Lease> = (0..4).map(|_| p.lease(1).unwrap()).collect();
    let mut ids: Vec<i32> = all.iter().flat_map(|l| l.page_ids().to_vec()).collect();
    ids.sort();
    assert_eq!((ids, p.used()), (vec![0, 1, 2, 3], 4));
    drop(all);
    assert_eq!(p.used(), 0);
}

#[test]
fn denials() {
    let p = pool();
    assert_eq!(p.lease(49).unwrap_err(), Denied::ExceedsRow { limit: 48 });
    let a = p.lease(48).unwrap();
    // One page free, no other kind to take chunks from: busy.
    assert_eq!(p.lease(17).unwrap_err(), Denied::Busy);
    let b = p.lease(16).unwrap();
    assert_eq!(p.lease(1).unwrap_err(), Denied::Busy);
    drop(a);
    assert!(p.lease(33).is_ok());
    drop(b);
    // Two pages of budget cap the row at two pages.
    let small = pool_of(&two_paged(), 8, 8, 0);
    assert_eq!((small.pages_max(), small.lease(48).unwrap_err()), (2, Denied::ExceedsRow { limit: 32 }));
    // A restore is denied the same way: by the row, or by the fresh pages it needs.
    let mut a = p.lease(16).unwrap();
    let (cp, _) = p.checkpoint(&mut a, 16).unwrap();
    assert_eq!(p.restore(&cp, cp.tokens(), 49).unwrap_err(), Denied::ExceedsRow { limit: 48 });
    let _b = p.lease(48).unwrap();
    assert_eq!(p.restore(&cp, cp.tokens(), 17).unwrap_err(), Denied::Busy);
    drop(a);
    assert_eq!(p.restore(&cp, cp.tokens(), 17).unwrap_err(), Denied::Busy);
}

#[test]
fn hybrid_denials() {
    let p = hybrid_pool();
    let mut a = p.lease(32).unwrap();
    let _b = p.lease(16).unwrap();
    let _c = p.lease(16).unwrap();
    // Every page and slot held: only an eviction helps.
    assert_eq!(p.lease(16).unwrap_err(), Denied::Busy);
    assert!(p.take_pending().is_none());
    // A checkpoint copies the slot, so it needs one too; retiring never does.
    assert_eq!(p.checkpoint(&mut a, 16).unwrap_err(), Denied::Busy);
    let cp = p.retire(a, 16);
    assert!(cp.seq_slot().is_some());
    // Five chunks: slot 0 and one page, room for no other slot ever.
    let small = pool_of(&hybrid(), 8, 5, 1);
    assert_eq!((small.slots_max(), small.total()), (1, 1));
    assert_eq!(small.lease(16).unwrap_err(), Denied::ExceedsPool);
}

#[test]
fn slots_and_rows() {
    let p = pool();
    let l = p.lease(20).unwrap(); // 2 pages
    let [p0, p1] = l.page_ids()[..] else { panic!() };
    assert_eq!(l.slot(0), p0 as i64 * 16);
    assert_eq!(l.slot(15), p0 as i64 * 16 + 15);
    assert_eq!(l.slot(16), p1 as i64 * 16);
    assert_eq!(l.slots(15..17), [p0 as i64 * 16 + 15, p1 as i64 * 16]);
    let mut t = Vec::new();
    l.extend_row("block_table", &mut t).unwrap();
    assert_eq!(t, [p0, p1, p0]);
    let mut d = Vec::new();
    l.extend_row("draft_block_table", &mut d).unwrap();
    let mut want: Vec<i32> = (0..4).map(|k| p0 * 4 + k).chain((0..4).map(|k| p1 * 4 + k)).collect();
    want.resize(16, p0 * 4);
    assert_eq!(d, want);
    // Position 17's draft entry (row index 17/4) names the draft page its slot falls in.
    assert_eq!(d[17 / 4], (l.slot(17) / 4) as i32);
    assert!(l.extend_row("slot_mapping", &mut t).is_err());
    assert!(l.seq_line("block_table", 0).is_err());
}

#[test]
#[should_panic(expected = "past the lease")]
fn slot_past_lease() {
    let p = pool();
    p.lease(16).unwrap().slot(16);
}

#[test]
fn seq_slots_and_lines() {
    let m = hybrid();
    // seqs 2 + pad + null: the manifest's first slots are 0..4, three of them leasable.
    assert_eq!(m.seq_slots(), 4);
    let p = hybrid_pool();
    assert_eq!((p.has_slots(), p.slots(), p.seq_tables().collect::<Vec<_>>()), (true, 4, vec!["line_index"]));
    let a = p.lease(16).unwrap();
    let b = p.lease(16).unwrap();
    let c = p.lease(16).unwrap();
    let (sa, sb, sc) = (a.seq_slot().unwrap(), b.seq_slot().unwrap(), c.seq_slot().unwrap());
    let mut got = vec![sa, sb, sc];
    got.sort();
    assert_eq!((got, p.slots_used()), (vec![1, 2, 3], 3));
    // A fourth page is free but no slot is, and one free page's two
    // chunks are short of a slot's three: busy, not exhausted.
    assert_eq!(p.lease(16).unwrap_err(), Denied::Busy);
    // Line r of a slot: slot × 3 + r; the null line 0 belongs to no lease.
    assert_eq!(a.seq_line("line_index", 2).unwrap(), sa * 3 + 2);
    assert_eq!(a.seq_lines("line_index").unwrap(), 3);
    assert!(a.seq_line("line_index", 3).is_err());
    assert_ne!(a.seq_line("line_index", 0).unwrap(), 0);
    drop(b);
    let d = p.lease(16).unwrap();
    assert_eq!(d.seq_slot().unwrap(), sb);
}

#[test]
fn pages_come_from_free_slots() {
    let p = hybrid_pool();
    let a = p.lease(48).unwrap(); // pages 0..3, slot 1
    assert_eq!((p.used(), p.slots_used()), (3, 1));
    // Two pages wanted, one free: the highest free slot's three chunks make page 4.
    assert_eq!(p.lease(32).unwrap_err(), Denied::Remapping);
    let plan = p.take_pending().unwrap();
    assert_eq!(
        (plan.unmap.len(), plan.map.len(), &plan.made[..], &plan.unmade[..]),
        (3, 2, &[(Kind::Page, 4)][..], &[(Kind::Slot, 3)][..])
    );
    assert_eq!(plan.access, [(0, 8..10)]);
    // Until it lands, a caller sees neither the page arriving nor the slot leaving.
    assert_eq!(p.lease(32).unwrap_err(), Denied::Remapping);
    assert!(p.take_pending().is_none());
    assert_eq!((p.total(), p.slots()), (4, 3));
    p.complete(plan);
    let b = p.lease(32).unwrap();
    assert_eq!(
        (b.page_ids().to_vec(), b.seq_slot(), p.total(), p.slots(), p.slots_used()),
        (vec![3, 4], Some(2), 5, 3, 2)
    );
    drop((a, b));
    assert_eq!((p.used(), p.slots_used()), (0, 0));
}

#[test]
fn a_slot_comes_from_free_pages() {
    // 24 chunks: 4 slots and 6 pages.
    let p = pool_of(&hybrid(), 8, 24, 4);
    let held: Vec<Lease> = (0..3).map(|_| p.lease(16).unwrap()).collect();
    assert_eq!((p.total(), p.used(), p.slots_used()), (6, 3, 3));
    // A page is free but every slot is held: two free pages give slot 4.
    assert_eq!(p.lease(16).unwrap_err(), Denied::Remapping);
    let plan = p.take_pending().unwrap();
    assert_eq!(
        (plan.unmade.clone(), plan.made.clone()),
        (vec![(Kind::Page, 5), (Kind::Page, 4)], vec![(Kind::Slot, 4)])
    );
    assert_eq!((plan.unmap.len(), plan.map.len()), (4, 3));
    p.complete(plan);
    let d = p.lease(16).unwrap();
    assert_eq!((d.seq_slot(), p.total(), p.slots(), p.slots_used()), (Some(4), 4, 5, 4));
    drop(held);
    // Chunks stay where they were: pages 4 and 5 are gone, slots 1..3 free.
    assert_eq!((p.total(), p.slots(), p.used()), (4, 5, 1));
}

#[test]
fn checkpoint_shares_pages_and_outlives_the_lease() {
    let p = pool();
    let mut a = p.lease(40).unwrap(); // 3 pages
    let (cp, copies) = p.checkpoint(&mut a, 32).unwrap(); // the first 2
    assert_eq!((cp.tokens(), cp.pages(), cp.seq_slot(), copies), (32, 2, None, Copies::default()));
    assert_eq!((pages_of(&cp), a.pages(), p.used()), (a.page_ids()[..2].to_vec(), 3, 3));
    drop(a);
    // The checkpoint keeps its 2 pages; the lease's third came back.
    assert_eq!(p.used(), 2);
    assert_eq!(p.lease(32).unwrap().pages(), 2);
    drop(cp);
    assert_eq!(p.used(), 0);
    // Retiring keeps the pages up to `len` and returns the rest.
    let a = p.lease(48).unwrap();
    let cp = p.retire(a, 17);
    assert_eq!((cp.tokens(), cp.pages(), p.used()), (17, 2, 2));
}

#[test]
fn checkpoints_along_one_lease_share_one_chain() {
    let p = pool();
    let mut a = p.lease(48).unwrap();
    let c1 = p.checkpoint(&mut a, 16).unwrap().0;
    let c2 = p.checkpoint(&mut a, 32).unwrap().0;
    let c3 = p.checkpoint(&mut a, 48).unwrap().0;
    // A shallower checkpoint taken after a deeper one finds its node up the chain.
    let c2b = p.checkpoint(&mut a, 20).unwrap().0;
    let (n1, n2, n2b) = (c1.nodes(), c2.nodes(), c2b.nodes());
    assert_eq!((n2b[1], n2[0]), (n2[1], n1[0]));
    assert_eq!(p.used(), 3);
    drop(a);
    drop(c3);
    assert_eq!(p.used(), 2);
    drop(c2);
    assert_eq!(p.used(), 2); // c2b still holds page 2's node
    drop(c2b);
    assert_eq!(p.used(), 1);
    drop(c1);
    assert_eq!(p.used(), 0);
}

#[test]
fn restore_shares_whole_pages_and_copies_the_partial_one() {
    let p = pool();
    let mut a = p.lease(40).unwrap();
    let [a0, a1, _] = a.page_ids()[..] else { panic!() };
    // 20 tokens: page a0 whole, a1 holds positions 16..20.
    let (cp, _) = p.checkpoint(&mut a, 20).unwrap();
    drop(a);
    let (b, copies) = p.restore(&cp, cp.tokens(), 40).unwrap();
    let [b0, b1, b2] = b.page_ids()[..] else { panic!() };
    // Shares a0, gets a fresh copy of a1 (a1 itself stays the checkpoint's), one more fresh page.
    assert_eq!((b0, b.prefix(), b.tokens()), (a0, 20, 48));
    assert_ne!(b1, a1);
    assert_eq!(copies, Copies { pages: vec![(a1, b1)], slot: None });
    assert_eq!(p.used(), 4);
    // Positions from 20 on are the lease's to write, into its own copy.
    assert_eq!(b.slot(20), b1 as i64 * 16 + 4);
    assert_eq!(b.slots(31..33), [b1 as i64 * 16 + 15, b2 as i64 * 16]);
    drop(cp);
    // a1 is only the checkpoint's: freed with it; a0 is still b's.
    assert_eq!(p.used(), 3);
    drop(b);
    assert_eq!(p.used(), 0);
    // At a page boundary nothing is copied: the whole pages are the checkpoint's own.
    let mut a = p.lease(32).unwrap();
    let (cp, _) = p.checkpoint(&mut a, 32).unwrap();
    drop(a);
    let (b, copies) = p.restore(&cp, cp.tokens(), 33).unwrap();
    assert_eq!((b.prefix(), &b.page_ids()[..2], b.pages(), copies), (32, &pages_of(&cp)[..], 3, Copies::default()));
    assert_eq!(p.used(), 3);
    drop((b, cp));
    // Restoring shallower than the checkpoint shares its whole pages up to there.
    let mut a = p.lease(40).unwrap();
    let [a0, a1, _] = a.page_ids()[..] else { panic!() };
    let (cp, _) = p.checkpoint(&mut a, 36).unwrap(); // 3 pages, 4 tokens into the third
    drop(a);
    // 32 tokens: a0 and a1 shared, one fresh page.
    let (c, copies) = p.restore(&cp, 32, 33).unwrap();
    assert_eq!((&c.page_ids()[..2], c.prefix(), c.pages(), copies), (&[a0, a1][..], 32, 3, Copies::default()));
    drop(c);
    // 16 tokens: page a0 shared, nothing copied.
    let (b, copies) = p.restore(&cp, 16, 20).unwrap();
    assert_eq!((b.page_ids()[0], b.prefix(), b.pages(), copies), (a0, 16, 2, Copies::default()));
    assert_ne!(b.page_ids()[1], a1);
}

#[test]
#[should_panic(expected = "restoring 4 tokens of a checkpoint of 10 (with a slot)")]
fn a_stateful_checkpoint_restores_at_its_length_only() {
    let p = pool_of(&hybrid(), 8, 26, 4);
    let mut a = p.lease(16).unwrap();
    let (cp, _) = p.checkpoint(&mut a, 10).unwrap();
    let _ = p.restore(&cp, 4, 20);
}

#[test]
#[should_panic(expected = "inside the shared prefix")]
fn restored_lease_refuses_its_prefix() {
    let p = pool();
    let mut a = p.lease(32).unwrap();
    let (cp, _) = p.checkpoint(&mut a, 20).unwrap();
    let (b, _) = p.restore(&cp, cp.tokens(), 40).unwrap();
    b.slot(19);
}

#[test]
fn fork_shares_whole_pages_and_copies_the_partial_one_and_the_slot() {
    let p = pool_of(&hybrid(), 8, 26, 4); // 4 slots, 7 pages
    let mut a = p.lease(40).unwrap(); // 3 pages
    let [a0, a1, _] = a.page_ids()[..] else { panic!() };
    let sa = a.seq_slot().unwrap();
    // The parent is 20 tokens in: page a0 whole, a1 holds 16..20.
    let (b, copies) = p.fork(&mut a, 20, 40).unwrap();
    let [b0, b1, _] = b.page_ids()[..] else { panic!() };
    assert_eq!((b0, b.prefix()), (a0, 20));
    assert_ne!(b1, a1);
    assert_eq!(copies, Copies { pages: vec![(a1, b1)], slot: Some((sa, b.seq_slot().unwrap())) });
    // Both keep writing their own copy of the page; a0 stays until both are gone.
    assert_eq!((a.slot(20), b.slot(20)), (a1 as i64 * 16 + 4, b1 as i64 * 16 + 4));
    assert_eq!((p.used(), p.slots_used()), (5, 2));
    drop(a);
    assert_eq!(p.used(), 3);
    drop(b);
    assert_eq!((p.used(), p.slots_used()), (0, 0));
    // At a page boundary nothing is copied; before the first boundary nothing is shared.
    let mut a = p.lease(32).unwrap();
    let (b, copies) = p.fork(&mut a, 16, 32).unwrap();
    assert_eq!((copies.pages.len(), b.page_ids()[0], b.prefix()), (0, a.page_ids()[0], 16));
    let (c, copies) = p.fork(&mut a, 3, 32).unwrap();
    assert_eq!((c.prefix(), copies.pages), (3, vec![(a.page_ids()[0], c.page_ids()[0])]));
    assert_eq!(p.fork(&mut a, 3, 32).unwrap_err(), Denied::Busy);
}

#[test]
fn wake_is_a_fresh_lease_with_a_prefix() {
    let p = pool();
    let l = p.wake(20, 40).unwrap();
    assert_eq!((l.pages(), l.page_ids().len(), l.prefix(), p.used()), (3, 3, 20, 3));
    assert_eq!(l.slot(20), l.page_ids()[1] as i64 * 16 + 4);
    drop(l);
    assert_eq!(p.used(), 0);
}

#[test]
fn hybrid_checkpoint_copies_the_slot_and_retire_moves_it() {
    // 26 chunks: 4 slots and 7 pages.
    let p = pool_of(&hybrid(), 8, 26, 4);
    let mut a = p.lease(16).unwrap();
    let sa = a.seq_slot().unwrap();
    let (cp, copies) = p.checkpoint(&mut a, 10).unwrap();
    let sc = cp.seq_slot().unwrap();
    assert_ne!(sc, sa);
    assert_eq!((copies, p.slots_used()), (Copies { pages: vec![], slot: Some((sa, sc)) }, 2));
    // A slot each for a and cp: one left; a restore takes it and copies the state in.
    let (b, copies) = p.restore(&cp, cp.tokens(), 17).unwrap();
    let sb = b.seq_slot().unwrap();
    assert_eq!((copies.slot, b.prefix(), p.slots_used()), (Some((sc, sb)), 10, 3));
    // No slot left; of the four free pages the restore needs two, the
    // other two hold a slot's worth of chunks.
    assert_eq!(p.restore(&cp, cp.tokens(), 17).unwrap_err(), Denied::Remapping);
    land(&p);
    let (c, copies) = p.restore(&cp, cp.tokens(), 17).unwrap();
    assert_eq!((copies.slot, p.slots(), p.slots_used(), p.total()), (Some((sc, 4)), 5, 4, 5));
    drop((b, c));
    // Retiring a moves its slot to the checkpoint: no copy; its one page is the same one cp shares.
    let a2 = p.retire(a, 10);
    assert_eq!((a2.seq_slot(), a2.pages(), p.slots_used(), p.used()), (Some(sa), 1, 2, 1));
    drop(cp);
    drop(a2);
    assert_eq!((p.slots_used(), p.used()), (0, 0));
}

#[test]
fn slot_only_leases_move_the_slot_alone() {
    let p = hybrid_pool();
    let mut l = p.lease_slot().unwrap();
    assert_eq!((l.pages(), l.tokens(), l.paged(), l.seq_slot().is_some()), (0, 0, false, true));
    assert_eq!((p.used(), p.slots_used()), (0, 1));
    // A checkpoint copies the slot and holds no page, at any length.
    let (cp, c) = p.checkpoint(&mut l, 5).unwrap();
    assert_eq!((cp.tokens(), cp.pages(), cp.paged(), cp.nodes().len()), (5, 0, false, 0));
    assert_eq!((c.pages.len(), c.slot.map(|(a, _)| a)), (0, l.seq_slot()));
    // Restoring is at its own length and gives a slot-only lease with that prefix.
    let (r, c) = p.restore(&cp, 5, 100).unwrap();
    assert_eq!((r.pages(), r.prefix(), c.pages.len(), c.slot.map(|(a, _)| a)), (0, 5, 0, cp.seq_slot()));
    assert_eq!(p.slots_used(), 3);
    drop(r);
    drop(cp);
    // A fork the same way.
    let (f, c) = p.fork(&mut l, 9, 100).unwrap();
    assert_eq!((f.pages(), f.prefix(), c.slot.map(|(a, _)| a)), (0, 9, l.seq_slot()));
    drop(f);
    // Retiring hands the slot over as it is.
    let slot = l.seq_slot();
    let cp = p.retire(l, 7);
    assert_eq!((cp.tokens(), cp.pages(), cp.seq_slot()), (7, 0, slot));
    drop(cp);
    assert_eq!((p.used(), p.slots_used()), (0, 0));
    // A woken slot-only lease: a slot and the prefix, no page.
    let w = p.wake_slot(11).unwrap();
    assert_eq!((w.pages(), w.prefix(), w.seq_slot().is_some()), (0, 11, true));
}

#[test]
fn a_long_chain_drops_without_recursion() {
    let m = Manifest::from_json(
        r#"{
        "schema_version": 4, "model": "t", "vars": {"tokens": {"max": 8}, "seqs": {"max": 2}},
        "states": {"kv": {"bytes_per_token": 1}},
        "buffers": {
            "block_table": {"kind": "input", "dtype": "i32", "shape": ["seqs", 200000], "domain": {"index_into": "kv", "stride": 1}}
        },
        "modules": {}, "ops": {}, "programs": {}
    }"#,
    )
    .unwrap();
    let p: Arc<Pool> = pool_of(&m, 1, 200_000, 0);
    let mut a = p.lease(200_000).unwrap();
    let cp = p.checkpoint(&mut a, 200_000).unwrap().0;
    drop(a);
    assert_eq!(p.used(), 200_000);
    drop(cp);
    assert_eq!(p.used(), 0);
}
