//! The pool through its public API: leases, checkpoints, restores, forks
//! and retirements.

mod common;

use std::sync::Arc;

use kern_manifest::types::{Dim, Manifest};
use std::collections::{BTreeMap, BTreeSet};

use kern_pool::{Checkpoint, Copies, Denied, Kind, Lease, Pool};

use common::{hybrid, hybrid4, hybrid_pool, paged4, pool, pool_of, two_paged, Rand};

#[test]
fn a_new_pool_holds_its_pages_and_slots() {
    let p = pool();
    assert_eq!((p.pages(), p.unit(), p.max_seq_tokens()), (4, 16, 48));
    assert_eq!(p.tables().collect::<Vec<_>>(), ["block_table", "draft_block_table"]);
    assert_eq!((p.has_slots(), p.slots()), (false, 0));
    assert_eq!(p.lease(1).unwrap().seq_slot(), None);
    let states: Vec<(&str, Kind, u64, usize)> =
        p.pooled().iter().map(|a| (a.state.as_str(), a.kind, a.object, a.objects)).collect();
    assert_eq!(states, [("draft_kv", Kind::Page, 16, 4), ("kv", Kind::Page, 16, 4)]);
    // Slots asked of a manifest without per-sequence states are none.
    assert_eq!(Pool::new(&two_paged(), 4, 3).unwrap().slots(), 0);
    let p = hybrid_pool();
    assert_eq!((p.pages(), p.slots(), p.slots_used()), (4, 4, 0));
    let states: Vec<(&str, Kind, u64, usize)> =
        p.pooled().iter().map(|a| (a.state.as_str(), a.kind, a.object, a.objects)).collect();
    assert_eq!(states, [("kv", Kind::Page, 16, 4), ("gdn", Kind::Slot, 24, 4)]);
}

#[test]
fn pool_rejects_a_manifest_it_cannot_lay_out() {
    let err = |m: &Manifest| match Pool::new(m, 4, 4) {
        Ok(_) => panic!("laid out"),
        Err(e) => e.to_string(),
    };
    // A state is paged or per sequence, not both.
    let mut m = hybrid();
    m.states.get_mut("gdn").unwrap().bytes = 8;
    assert!(err(&m).contains("one layout"));
    // The line table's shape is [lines, seqs, w] with the state's line count.
    let mut m = hybrid();
    let b = m.buffers.get_mut("line_index").unwrap();
    b.shape = vec![Dim::Const(4), Dim::Var("seqs".into())];
    assert!(err(&m).contains("state holds 3"));
    let b = m.buffers.get_mut("line_index").unwrap();
    b.shape = vec![Dim::Var("seqs".into())];
    assert!(err(&m).contains("[lines, seqs, w]"));
    let b = m.buffers.get_mut("line_index").unwrap();
    b.shape = vec![Dim::Const(3), Dim::Var("seqs".into()), Dim::Const(8)];
    let p = pool_of(&m, 4, 4);
    let a = p.lease(16).unwrap();
    assert_eq!((a.seq_width("line_index").unwrap(), a.seq_lines("line_index").unwrap()), (8, 3));
}

#[test]
fn a_line_table_over_one_line_per_slot_holds_the_slot_in_every_row() {
    // Lines of 24 bytes: one per slot, so the table's 3 rows are 3 states
    // of that layout (a host's per-layer states), each at the lease's slot.
    let mut m = hybrid();
    m.buffers.get_mut("line_index").unwrap().domain.as_mut().unwrap().stride = 24;
    let p = pool_of(&m, 4, 4);
    let a = p.lease(16).unwrap();
    let slot = a.seq_slot().unwrap();
    let rows: Vec<i32> = (0..3).map(|r| a.seq_line("line_index", r).unwrap()).collect();
    assert_eq!((rows, a.seq_lines("line_index").unwrap()), (vec![slot; 3], 3));
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
    // One page free: busy.
    assert_eq!(p.lease(17).unwrap_err(), Denied::Busy);
    let b = p.lease(16).unwrap();
    assert_eq!(p.lease(1).unwrap_err(), Denied::Busy);
    drop(a);
    assert!(p.lease(33).is_ok());
    drop(b);
    // Two pages cap the row at two pages.
    let small = pool_of(&two_paged(), 2, 0);
    assert_eq!((small.pages(), small.lease(48).unwrap_err()), (2, Denied::ExceedsRow { limit: 32 }));
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
    // A checkpoint copies the slot, so it needs one too; retiring never does.
    assert_eq!(p.checkpoint(&mut a, 16).unwrap_err(), Denied::Busy);
    let cp = p.retire(a, 16);
    assert!(cp.seq_slot().is_some());
    // Slot 0 and one page: no slot to lease, ever.
    let small = pool_of(&hybrid(), 1, 1);
    assert_eq!((small.slots(), small.pages()), (1, 1));
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
    // A fourth page is free but no slot is: busy, not exhausted.
    assert_eq!(p.lease(16).unwrap_err(), Denied::Busy);
    // Line r of a slot: slot × 3 + r; the null line 0 belongs to no lease.
    assert_eq!(a.seq_line("line_index", 2).unwrap(), sa * 3 + 2);
    assert_eq!(a.seq_line_row("line_index").unwrap(), [sa * 3, sa * 3 + 1, sa * 3 + 2]);
    assert_eq!(a.seq_lines("line_index").unwrap(), 3);
    assert!(a.seq_line("line_index", 3).is_err());
    assert_ne!(a.seq_line("line_index", 0).unwrap(), 0);
    drop(b);
    let d = p.lease(16).unwrap();
    assert_eq!(d.seq_slot().unwrap(), sb);
}

#[test]
fn checkpoint_shares_pages_and_outlives_the_lease() {
    let p = pool();
    let mut a = p.lease(40).unwrap(); // 3 pages
    let (cp, copies) = p.checkpoint(&mut a, 32).unwrap(); // the first 2
    assert_eq!((cp.tokens(), cp.page_ids().len(), cp.seq_slot(), copies), (32, 2, None, Copies::default()));
    assert_eq!((cp.page_ids(), a.pages(), p.used()), (a.page_ids()[..2].to_vec(), 3, 3));
    drop(a);
    // The checkpoint keeps its 2 pages; the lease's third came back.
    assert_eq!(p.used(), 2);
    assert_eq!(p.lease(32).unwrap().pages(), 2);
    drop(cp);
    assert_eq!(p.used(), 0);
    // Retiring keeps the pages up to `len` and returns the rest.
    let a = p.lease(48).unwrap();
    let cp = p.retire(a, 17);
    assert_eq!((cp.tokens(), cp.page_ids().len(), p.used()), (17, 2, 2));
}

#[test]
fn checkpoints_along_one_lease_share_one_chain() {
    let p = pool();
    let mut a = p.lease(48).unwrap();
    let c1 = p.checkpoint(&mut a, 16).unwrap().0;
    let c2 = p.checkpoint(&mut a, 32).unwrap().0;
    let c3 = p.checkpoint(&mut a, 48).unwrap().0;
    // A shallower checkpoint taken after a deeper one shares the whole
    // pages up the chain and copies the page it ends inside.
    let (c2b, copies) = p.checkpoint(&mut a, 20).unwrap();
    assert_eq!((&c2b.page_ids()[..1], c2b.page_ids().len(), copies.pages.len()), (&c1.page_ids()[..], 2, 1));
    assert_ne!(c2b.page_ids()[1], c2.page_ids()[1]);
    assert_eq!(p.used(), 4);
    drop(a);
    drop(c3);
    assert_eq!(p.used(), 3);
    drop(c2);
    assert_eq!(p.used(), 2); // c2b holds page 1's node and its own copy of page 2
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
    // 20 tokens: page a0 whole, a1 holds positions 16..20, copied into a
    // page of the checkpoint's own so the lease can go on writing a1.
    let (cp, copies) = p.checkpoint(&mut a, 20).unwrap();
    let [c0, c1] = cp.page_ids()[..] else { panic!() };
    assert_eq!((c0, copies), (a0, Copies { pages: vec![(a1, c1)], slot: None }));
    assert_ne!(c1, a1);
    drop(a);
    let (b, copies) = p.restore(&cp, cp.tokens(), 40).unwrap();
    let [b0, b1, b2] = b.page_ids()[..] else { panic!() };
    // Shares a0, gets a fresh copy of c1 (c1 itself stays the checkpoint's), one more fresh page.
    assert_eq!((b0, b.prefix(), b.tokens()), (a0, 20, 48));
    assert_ne!(b1, c1);
    assert_eq!(copies, Copies { pages: vec![(c1, b1)], slot: None });
    assert_eq!(p.used(), 4);
    // Positions from 20 on are the lease's to write, into its own copy.
    assert_eq!(b.slot(20), b1 as i64 * 16 + 4);
    assert_eq!(b.slots(31..33), [b1 as i64 * 16 + 15, b2 as i64 * 16]);
    drop(cp);
    // c1 is only the checkpoint's: freed with it; a0 is still b's.
    assert_eq!(p.used(), 3);
    drop(b);
    assert_eq!(p.used(), 0);
    // At a page boundary nothing is copied: the whole pages are the checkpoint's own.
    let mut a = p.lease(32).unwrap();
    let (cp, _) = p.checkpoint(&mut a, 32).unwrap();
    drop(a);
    let (b, copies) = p.restore(&cp, cp.tokens(), 33).unwrap();
    assert_eq!((b.prefix(), &b.page_ids()[..2], b.pages(), copies), (32, &cp.page_ids()[..], 3, Copies::default()));
    assert_eq!(p.used(), 3);
    drop((b, cp));
    // Restoring shallower than the checkpoint shares its whole pages up to there.
    let mut a = p.lease(40).unwrap();
    let [a0, a1, _] = a.page_ids()[..] else { panic!() };
    let (cp, _) = p.checkpoint(&mut a, 36).unwrap(); // 3 pages, the third a copy
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
    let p = pool_of(&hybrid(), 7, 4);
    let mut a = p.lease(16).unwrap();
    let (cp, _) = p.checkpoint(&mut a, 10).unwrap();
    let _ = p.restore(&cp, 4, 20);
}

#[test]
#[should_panic(expected = "inside a sealed page")]
fn a_lease_refuses_the_pages_it_sealed() {
    let p = pool();
    let mut a = p.lease(32).unwrap();
    let (_cp, _) = p.checkpoint(&mut a, 20).unwrap();
    // Page 0 went whole into the checkpoint's chain; page 1 was copied and
    // stays the lease's own.
    assert_eq!(a.slot(20), a.page_ids()[1] as i64 * 16 + 4);
    a.slot(15);
}

#[test]
#[should_panic(expected = "inside the shared prefix")]
fn restored_lease_refuses_its_prefix() {
    let p = pool();
    let mut a = p.lease(32).unwrap();
    let (cp, _) = p.checkpoint(&mut a, 20).unwrap();
    drop(a);
    let (b, _) = p.restore(&cp, cp.tokens(), 40).unwrap();
    b.slot(19);
}

#[test]
fn fork_shares_whole_pages_and_copies_the_partial_one_and_the_slot() {
    let p = pool_of(&hybrid(), 7, 4); // 4 slots, 7 pages
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
fn hybrid_checkpoint_copies_the_slot_and_retire_moves_it() {
    let p = pool_of(&hybrid(), 8, 4);
    let mut a = p.lease(16).unwrap();
    let (a0, sa) = (a.page_ids()[0], a.seq_slot().unwrap());
    let (cp, copies) = p.checkpoint(&mut a, 10).unwrap();
    let (c0, sc) = (cp.page_ids()[0], cp.seq_slot().unwrap());
    assert_ne!((c0, sc), (a0, sa));
    assert_eq!((copies, p.used(), p.slots_used()), (Copies { pages: vec![(a0, c0)], slot: Some((sa, sc)) }, 2, 2));
    // A slot each for a and cp: one left; a restore takes it and copies the state in.
    let (b, copies) = p.restore(&cp, cp.tokens(), 17).unwrap();
    let sb = b.seq_slot().unwrap();
    assert_eq!((copies.slot, b.prefix(), p.slots_used()), (Some((sc, sb)), 10, 3));
    // No slot left: busy, however many pages are free.
    assert_eq!(p.restore(&cp, cp.tokens(), 17).unwrap_err(), Denied::Busy);
    drop(b);
    // Retiring a moves its slot and its page to the checkpoint: no copy.
    let a2 = p.retire(a, 10);
    assert_eq!((a2.seq_slot(), a2.page_ids(), p.slots_used(), p.used()), (Some(sa), vec![a0], 2, 2));
    drop(cp);
    assert_eq!((p.slots_used(), p.used()), (1, 1));
    drop(a2);
    assert_eq!((p.slots_used(), p.used()), (0, 0));
}

#[test]
fn slot_only_leases_move_the_slot_alone() {
    let p = hybrid_pool();
    let mut l = p.lease_slot().unwrap();
    assert_eq!((l.pages(), l.tokens(), l.seq_slot().is_some()), (0, 0, true));
    assert_eq!((p.used(), p.slots_used()), (0, 1));
    // A checkpoint copies the slot and holds no page, at any length.
    let (cp, c) = p.checkpoint(&mut l, 5).unwrap();
    assert_eq!((cp.tokens(), cp.page_ids().len()), (5, 0));
    assert_eq!((c.pages.len(), c.slot.map(|(a, _)| a)), (0, l.seq_slot()));
    // Restoring is at its own length and gives a slot-only lease with that prefix.
    let (r, c) = p.restore(&cp, 5, 100).unwrap();
    assert_eq!((r.pages(), r.prefix(), c.pages.len(), c.slot.map(|(a, _)| a)), (0, 5, 0, cp.seq_slot()));
    // Its copies again, for a fetch that did not land: the slot alone.
    let again = r.copies_from(&cp);
    assert_eq!((again.pages.len(), again.slot), (0, c.slot));
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
    assert_eq!((cp.tokens(), cp.page_ids().len(), cp.seq_slot()), (7, 0, slot));
    drop(cp);
    assert_eq!((p.used(), p.slots_used()), (0, 0));
}

#[test]
fn a_long_chain_drops_without_recursion() {
    let m = Manifest::from_json(
        r#"{
        "schema_version": 7, "model": "t", "vars": {"tokens": {"max": 8}, "seqs": {"max": 2}},
        "states": {"kv": {"bytes_per_token": 1}},
        "buffers": {
            "block_table": {"kind": "input", "dtype": "i32", "shape": ["seqs", 200000], "domain": {"index_into": "kv", "stride": 1}}
        },
        "modules": {}, "ops": {}, "programs": {}
    }"#,
    )
    .unwrap();
    let p: Arc<Pool> = pool_of(&m, 200_000, 0);
    let mut a = p.lease(200_000).unwrap();
    let cp = p.checkpoint(&mut a, 200_000).unwrap().0;
    drop(a);
    assert_eq!(p.used(), 200_000);
    drop(cp);
    assert_eq!(p.used(), 0);
}

/// The positions and slots of a device, each holding the stamp last
/// written there.
#[derive(Default)]
struct Device {
    positions: BTreeMap<i64, u32>,
    slots: BTreeMap<i32, u32>,
}

impl Device {
    fn copy(&mut self, unit: usize, c: &Copies) {
        for &(s, d) in &c.pages {
            for k in 0..unit as i64 {
                match self.positions.get(&(s as i64 * unit as i64 + k)).copied() {
                    Some(v) => self.positions.insert(d as i64 * unit as i64 + k, v),
                    None => self.positions.remove(&(d as i64 * unit as i64 + k)),
                };
            }
        }
        if let Some((s, d)) = c.slot {
            match self.slots.get(&s).copied() {
                Some(v) => self.slots.insert(d, v),
                None => self.slots.remove(&d),
            };
        }
    }

    fn at(&self, pages: &[i32], unit: usize, pos: usize) -> Option<u32> {
        self.positions.get(&(pages[pos / unit] as i64 * unit as i64 + (pos % unit) as i64)).copied()
    }
}

fn state_of(content: &[u32]) -> u32 {
    content.iter().fold(7, |s, &v| s.wrapping_mul(31).wrapping_add(v))
}

struct Seq {
    lease: Lease,
    content: Vec<u32>,
}

struct Held {
    cp: Checkpoint,
    content: Vec<u32>,
}

/// Write positions `content.len()..upto` of `s`, then its state.
fn fill(dev: &mut Device, s: &mut Seq, upto: usize, stamp: &mut u32) {
    for pos in s.content.len()..upto {
        *stamp += 1;
        dev.positions.insert(s.lease.slot(pos), *stamp);
        s.content.push(*stamp);
    }
    if let Some(slot) = s.lease.seq_slot() {
        dev.slots.insert(slot, state_of(&s.content));
    }
}

/// Every position and slot a live handle names reads what its writer
/// put there, and the pool holds exactly what the handles name.
fn check(dev: &Device, p: &Pool, seqs: &[Seq], cps: &[Held]) {
    let unit = p.unit() as usize;
    let (mut pages, mut slots) = (BTreeSet::new(), BTreeSet::new());
    for s in seqs {
        let ids = s.lease.page_ids();
        for (pos, &v) in s.content.iter().enumerate() {
            assert_eq!(dev.at(ids, unit, pos), Some(v), "lease position {pos}");
        }
        if let Some(slot) = s.lease.seq_slot() {
            assert_eq!(dev.slots.get(&slot), Some(&state_of(&s.content)));
            slots.insert(slot);
        }
        pages.extend(ids.iter().copied());
    }
    for c in cps {
        let ids = c.cp.page_ids();
        for (pos, &v) in c.content.iter().enumerate() {
            assert_eq!(dev.at(&ids, unit, pos), Some(v), "checkpoint position {pos}");
        }
        if let Some(slot) = c.cp.seq_slot() {
            assert_eq!(dev.slots.get(&slot), Some(&state_of(&c.content)));
            slots.insert(slot);
        }
        pages.extend(ids);
    }
    assert_eq!((p.used(), p.slots_used()), (pages.len(), slots.len()));
}

/// Random leases, fills, checkpoints, restores, forks, retirements and
/// drops, every position written by its lease and read back through
/// every handle that names it: a page is written by the lease it
/// belongs to and nobody else, so what a checkpoint held stays what
/// it held however far its sequence runs on. (A half-page checkpoint
/// that shared the page its lease kept writing was the bug behind
/// this rewrite.) With a recurrent state a checkpoint is the sequence
/// as of now and restores at its own length only.
fn handles_read_what_their_writer_wrote(m: &Manifest, pages: usize, slots: usize, seed: u64) {
    let p = Arc::new(Pool::new(m, pages, slots).unwrap());
    let unit = p.unit() as usize;
    let max = p.max_seq_tokens();
    let stateful = p.has_slots();
    let mut dev = Device::default();
    let mut seqs: Vec<Seq> = Vec::new();
    let mut cps: Vec<Held> = Vec::new();
    let mut rand = Rand::new(seed);
    let mut stamp = 0;
    // A checkpoint length: the whole sequence when a state pins it, else anywhere in it.
    let cut = |rand: &mut Rand, filled: usize| if stateful { filled } else { 1 + rand.next(filled) };
    for _ in 0..4000 {
        match rand.next(9) {
            0 | 1 => {
                if let Ok(lease) = p.lease(1 + rand.next(max)) {
                    let mut s = Seq { lease, content: Vec::new() };
                    let upto = rand.next(s.lease.tokens() + 1);
                    fill(&mut dev, &mut s, upto, &mut stamp);
                    seqs.push(s);
                }
            }
            2 if !seqs.is_empty() => {
                let i = rand.next(seqs.len());
                let s = &mut seqs[i];
                let upto = s.content.len() + rand.next(s.lease.tokens() - s.content.len() + 1);
                fill(&mut dev, s, upto, &mut stamp);
            }
            3 if !seqs.is_empty() => {
                let i = rand.next(seqs.len());
                let s = &mut seqs[i];
                if s.content.is_empty() {
                    continue;
                }
                let len = cut(&mut rand, s.content.len());
                if let Ok((cp, copies)) = p.checkpoint(&mut s.lease, len) {
                    dev.copy(unit, &copies);
                    cps.push(Held { cp, content: s.content[..len].to_vec() });
                }
            }
            4 if !cps.is_empty() => {
                let c = &cps[rand.next(cps.len())];
                let whole = (c.cp.tokens() - 1) / unit;
                let len = if c.cp.has_slot() || whole == 0 || rand.next(2) == 0 {
                    c.cp.tokens()
                } else {
                    (1 + rand.next(whole)) * unit
                };
                if len >= max {
                    continue;
                }
                if let Ok((lease, copies)) = p.restore(&c.cp, len, len + 1 + rand.next(max - len)) {
                    dev.copy(unit, &copies);
                    let mut s = Seq { lease, content: c.content[..len].to_vec() };
                    let upto = len + rand.next(s.lease.tokens() - len + 1);
                    fill(&mut dev, &mut s, upto, &mut stamp);
                    seqs.push(s);
                }
            }
            5 if !seqs.is_empty() => {
                let i = rand.next(seqs.len());
                let filled = seqs[i].content.len();
                if filled == 0 || filled >= max {
                    continue;
                }
                let len = cut(&mut rand, filled);
                let tokens = len + 1 + rand.next(max - len);
                if let Ok((lease, copies)) = p.fork(&mut seqs[i].lease, len, tokens) {
                    dev.copy(unit, &copies);
                    let mut s = Seq { lease, content: seqs[i].content[..len].to_vec() };
                    let upto = len + rand.next(s.lease.tokens() - len + 1);
                    fill(&mut dev, &mut s, upto, &mut stamp);
                    seqs.push(s);
                }
            }
            6 if !seqs.is_empty() => {
                let s = seqs.swap_remove(rand.next(seqs.len()));
                if s.content.is_empty() {
                    continue;
                }
                let len = cut(&mut rand, s.content.len());
                let content = s.content[..len].to_vec();
                cps.push(Held { cp: p.retire(s.lease, len), content });
            }
            7 if !seqs.is_empty() => {
                seqs.swap_remove(rand.next(seqs.len()));
            }
            8 if !cps.is_empty() => {
                cps.swap_remove(rand.next(cps.len()));
            }
            _ => {}
        }
        check(&dev, &p, &seqs, &cps);
    }
    drop((seqs, cps));
    assert_eq!((p.used(), p.slots_used()), (0, 0));
}

#[test]
fn paged_handles_read_what_their_writer_wrote() {
    // 24 pages of 4 tokens, rows of 8.
    handles_read_what_their_writer_wrote(&paged4(), 24, 0, 0x9E37_79B9_7F4A_7C15);
}

#[test]
fn stateful_handles_read_what_their_writer_wrote() {
    // 3 slots and 6 pages: both run short.
    handles_read_what_their_writer_wrote(&hybrid4(), 6, 3, 0x2545_F491_4F6C_DD1D);
}

#[test]
fn hybrid_handles_read_what_their_writer_wrote() {
    // 3 slots of 24 bytes, 8 pages of 16.
    handles_read_what_their_writer_wrote(&hybrid(), 8, 3, 0x1234_5678_9ABC_DEF1);
}
