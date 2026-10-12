//! The head's accounting through its public API, over real pools of a
//! tiny group manifest (kern-manifest's `ranks.json`: four ranks in one
//! `tp` group, 16-token pages of 16 bytes, a 24-byte recurrent slot) and a
//! fake set of ranks that lands whatever it is told the moment it hears it.

use kern_manifest::types::Manifest;
use kern_manifest::{verify, Protocol};
use kern_pool::{Denied, Pool};
use kern_serve::ranks::{Act, Answer, Cell, Groups, Order, Ranks, Row};

mod common;
use common::{dealt, replicated, Rand};

fn manifest() -> Manifest {
    Manifest::from_json(include_str!("../../kern-manifest/tests/fixtures/ranks.json")).unwrap()
}

/// 20 chunks of 8 bytes, `slots` sequence slots made at the start (slot 0
/// the null one, slot 1 the pad's), the rest pages: 5 pages beside 3 slots.
fn pool(slots: usize) -> Pool {
    Pool::new(&manifest(), 8, 20, slots, None).unwrap().0
}

fn ranks(slots: [usize; 4], host: (u64, u64)) -> Ranks {
    let p = Protocol::check(&verify(manifest()).unwrap()).unwrap();
    Ranks::new(p, Groups::new(4, 4).unwrap(), slots.map(pool).into(), host).unwrap()
}

/// Ranks that land every remap and transfer as soon as they hear of it.
fn answer(orders: &[Order], landed: &mut [u64; 4]) -> Vec<Answer> {
    let count = |o: &Order, f: fn(&Act) -> bool| o.acts.iter().filter(|a| f(a)).count() as u64;
    orders
        .iter()
        .zip(landed.iter_mut())
        .map(|(o, l)| {
            *l += count(o, |a| matches!(a, Act::Park(_) | Act::Wake(_)));
            Answer { remaps: count(o, |a| matches!(a, Act::Remap(_))), landed: *l, ..Answer::default() }
        })
        .collect()
}

fn acts(orders: &[Order]) -> Vec<Vec<Act>> {
    orders.iter().map(|o| o.acts.clone()).collect()
}

#[test]
fn a_row_is_every_members_or_none_and_waits_for_the_slot_being_carved() {
    let mut r = ranks([3, 3, 2, 3], (0, 0));
    let before = (r.pages_used(), r.seq_slots());
    // Rank 2 has no slot free: the owner's pages and rank 1's slot go back.
    assert_eq!(r.lease(20, |_| Some(0)).err(), Some(Denied::Remapping));
    assert_eq!((r.pages_used(), r.seq_slots(), r.busy()), (before.0, before.1, true));
    let o = r.orders(None, true);
    let pads = vec![Act::Zero(1)];
    assert_eq!((acts(&o)[0].clone(), acts(&o)[1].clone(), acts(&o)[3].clone()), (pads.clone(), pads.clone(), pads));
    assert_eq!((o[2].acts.len(), matches!(o[2].acts[1], Act::Remap(_))), (2, true));
    assert_eq!(o.iter().map(|o| o.wait).collect::<Vec<_>>(), [false, false, true, false]);
    r.heard(&answer(&o, &mut [0; 4])).unwrap();
    // Landed: every member leases, the owner pages, its peers a slot alone.
    let row = r.lease(20, |_| Some(0)).unwrap();
    assert_eq!((row.owner().index(), row.prefix(), r.pages_used(), r.busy()), (0, 0, 2, false));
    assert_eq!(acts(&r.orders(None, false)), vec![vec![Act::Zero(2)]; 4]);
    // No rank open is no lease.
    assert_eq!(r.lease(20, |_| None).err(), Some(Denied::Busy));
}

#[test]
fn a_park_holds_its_pages_until_its_copies_land_and_a_wake_rises_once_they_have() {
    let mut r = ranks([3; 4], (16, 4));
    let row = r.lease(40, |_| Some(0)).unwrap();
    let snap = r.retire(row, 32).unwrap();
    r.orders(None, false);
    assert_eq!((r.pages_used(), r.seq_slots().1), (2, 4));
    let sleeping = r.park(snap).ok().unwrap();
    // Parked, the device copies are still held until the ranks report them landed.
    assert_eq!((r.pages_used(), r.parking(), r.host_tier().map(|(u, _)| u > 0)), (2, true, Some(true)));
    let o = r.orders(None, false);
    assert_eq!(o.iter().map(|o| matches!(o.acts[..], [Act::Park(_)])).collect::<Vec<_>>(), [true; 4]);
    let mut landed = [0; 4];
    r.heard(&answer(&o, &mut landed)).unwrap();
    assert_eq!((r.pages_used(), r.seq_slots().1, r.parking(), r.busy()), (0, 0, false, false));
    // A wake takes its pages at once and hands the row out only after its copies.
    let rising = r.wake(&sleeping, 32, 40).unwrap();
    assert_eq!(r.pages_used(), 3);
    let rising = r.awake(rising).err().unwrap();
    let o = r.orders(None, false);
    assert_eq!(o.iter().map(|o| matches!(o.acts[..], [Act::Wake(_)])).collect::<Vec<_>>(), [true; 4]);
    r.heard(&answer(&o, &mut landed)).unwrap();
    let row = r.awake(rising).ok().unwrap();
    assert_eq!((row.owner().index(), row.prefix(), r.pages_used()), (0, 32, 3));
}

#[test]
fn a_park_short_of_host_room_on_any_member_hands_the_snapshot_back() {
    // No host tier at all is the shortest room there is.
    let mut r = ranks([3; 4], (0, 0));
    let row = r.lease(40, |_| Some(0)).unwrap();
    let snap = r.retire(row, 32).unwrap();
    let snap = r.park(snap).err().unwrap();
    assert_eq!((r.pages_used(), r.parking(), r.orders(None, false)[0].acts.len()), (2, false, 2));
    drop(snap);
    assert_eq!(r.pages_used(), 0);
}

#[test]
fn a_step_gives_every_rank_the_same_program_and_vars_over_its_own_rows() {
    let mut r = ranks([4; 4], (0, 0));
    let a = r.lease(20, |q| Some(q.index())).unwrap();
    let b = r.lease(20, |q| Some(if q.index() == 0 { 9 } else { q.index() })).unwrap();
    assert_eq!((a.owner().index(), b.owner().index()), (0, 1));
    let cells = [Cell { row: &a, ids: vec![7], pos: 3 }, Cell { row: &b, ids: vec![8], pos: 5 }];
    let mut st = r.stage(&cells, 1, |k| k).unwrap();
    let f = st.forward(r.protocol(), 1).unwrap();
    let items = st.items(&f);
    assert_eq!(
        items.iter().map(|i| (i.program.as_str(), i.vars.clone())).collect::<std::collections::BTreeSet<_>>().len(),
        1
    );
    // The batch is every member's block, this rank's own first.
    let tokens: Vec<Vec<i64>> = items.iter().map(|i| i.rows["token_ids"].clone()).collect();
    assert_eq!(tokens, [vec![7, 8, 0, 0], vec![8, 0, 0, 7], vec![0, 0, 7, 8], vec![0, 7, 8, 0]]);
    let lens: Vec<Vec<i64>> = items.iter().map(|i| i.rows["seq_lens"].clone()).collect();
    assert_eq!(lens, [vec![4], vec![6], vec![1], vec![1]]);
    assert_eq!(items[2].rows["tp_blocks"], [0, 1, 2, 3, 4]);
    // Each cell's token is read from its owner's answer.
    let answers: Vec<Answer> = (0..4).map(|q| Answer { tokens: vec![100 + q, 0, 0, 0], ..Answer::default() }).collect();
    assert_eq!(st.emitted(r.protocol(), &f, &answers).unwrap(), [vec![100], vec![101]]);
    let o = r.orders(Some(items), false);
    assert_eq!(o.iter().map(|o| o.item.is_some()).collect::<Vec<_>>(), [true; 4]);
}

#[test]
fn a_replicated_row_is_paged_on_every_member_and_keeps_nothing() {
    let mut r = replicated(4, 1);
    r.orders(None, false);
    // Four members of 48 local positions hold 192; past that is past a row.
    assert_eq!(r.max_seq_tokens(), 192);
    assert_eq!(r.lease(193, |_| Some(0)).err(), Some(Denied::ExceedsRow { limit: 192 }));
    // ⌈37 / 4⌉ = 10 local positions: a page on every member, every slot zeroed.
    let mut row = r.lease(37, |_| Some(0)).unwrap();
    let pages: Vec<usize> = row.leases().iter().map(|l| l.pages()).collect();
    assert_eq!((row.owner().index(), pages, r.pages_used()), (0, vec![1; 4], 4));
    let zeros: Vec<Vec<Act>> = row.leases().iter().map(|l| vec![Act::Zero(l.seq_slot().unwrap())]).collect();
    assert_eq!(acts(&r.orders(None, false)), zeros);
    // An adopted row's slots are the push's: nothing is zeroed.
    let pushed = r.adopt(37, |_| Some(1)).unwrap();
    assert_eq!((acts(&r.orders(None, false)), r.pages_used()), (vec![vec![]; 4], 8));
    // Nothing is kept: a checkpoint makes no snapshot, a retire drops the row.
    assert_eq!((r.checkpoint(&mut row, 16).unwrap().is_none(), r.retire(row, 37).is_none()), (true, true));
    drop(pushed);
    assert_eq!((r.pages_used(), r.seq_slots().1, r.orders(None, false)[0].acts.len()), (0, 0, 0));
    // The batch runs over the group the positions are dealt over, or not at all.
    let p = Protocol::check(&verify(dealt(4)).unwrap()).unwrap();
    let pools = (0..4).map(|_| Pool::new(&dealt(4), 256, 128, 10, None).unwrap().0).collect();
    assert!(Ranks::new(p, Groups::new(4, 2).unwrap(), pools, (0, 0)).is_err());
}

/// Random replicated steps over (w, groups, rows, lengths): every member of
/// a group stages the same rows in the same order; position `L - 1` of a
/// row is written by member `(L - 1) % w` alone, into its own lease's slot,
/// and by every other member into its pad page; the members' sequence
/// lengths add up to `L`; each member's page and line tables are its own
/// lease's; each cell's token is read from its group's first member, and
/// a member whose answer differs from it is an error.
#[test]
fn a_replicated_step_deals_every_position_to_exactly_one_member() {
    let mut rand = Rand(0x9E37_79B9_7F4A_7C15);
    for _ in 0..400 {
        let (w, groups) = ([1, 2, 4, 8][rand.next(4)], 1 + rand.next(2));
        let mut r = replicated(w, groups);
        let mut held = vec![0usize; w * groups];
        let mut rows = Vec::new();
        let mut lens = Vec::new();
        for _ in 0..1 + rand.next(8 * groups) {
            let len = 1 + rand.next(48 * w);
            let row = r.lease(len, |q| Some(held[q.index()]).filter(|&k| k < 8)).unwrap();
            held[row.owner().index()] += 1;
            rows.push(row);
            lens.push(len);
        }
        let cells: Vec<Cell> = rows
            .iter()
            .zip(&lens)
            .enumerate()
            .map(|(i, (row, &l))| Cell { row, ids: vec![100 + i as i64], pos: l - 1 })
            .collect();
        let mut st = r.stage(&cells, 1, |k| k.next_power_of_two()).unwrap();
        let f = st.forward(r.protocol(), 1).unwrap();
        let items = st.items(&f);
        let shapes: std::collections::BTreeSet<_> = items.iter().map(|i| (i.program.clone(), i.vars.clone())).collect();
        assert_eq!(shapes.len(), 1);
        let mut answers = Vec::new();
        for g in 0..groups {
            let mine: Vec<usize> = (0..rows.len()).filter(|&i| rows[i].owner().index() == g * w).collect();
            let b = items[g * w].rows["seq_lens"].len();
            let tokens: Vec<i64> = mine.iter().map(|&i| 100 + i as i64).chain(std::iter::repeat(0)).take(b).collect();
            let mut sums = vec![0i64; mine.len()];
            for m in 0..w {
                let it = &items[g * w + m].rows;
                assert_eq!(it["token_ids"], tokens, "w {w}, member {m}");
                let held: Vec<i32> = mine.iter().flat_map(|&i| rows[i].leases()[m].page_ids().to_vec()).collect();
                let mut pads = Vec::new();
                for (k, slot) in it["slot_mapping"].iter().enumerate() {
                    let Some(&i) = mine.get(k) else {
                        pads.push(*slot);
                        continue;
                    };
                    let (pos, lease) = (lens[i] - 1, &rows[i].leases()[m]);
                    match pos % w == m {
                        true => assert_eq!(*slot, lease.slot(pos / w)),
                        false => pads.push(*slot),
                    }
                    sums[k] += it["seq_lens"][k];
                    let table = &it["block_table"][k * 3..k * 3 + lease.pages()];
                    assert_eq!(table, lease.page_ids().iter().map(|&p| p as i64).collect::<Vec<_>>());
                    // The line table spans the `seqs` bound, 8 columns.
                    let lines: Vec<i64> = (0..3).map(|l| it["kda.line_index"][l * 8 + k]).collect();
                    let want: Vec<i64> = (0..3).map(|l| lease.seq_line("kda.line_index", l).unwrap() as i64).collect();
                    assert_eq!(lines, want);
                }
                // The pad page: one page, held by no row.
                let page = pads.first().map(|s| (s / 16) as i32);
                assert!(pads.iter().all(|s| Some((s / 16) as i32) == page), "pads {pads:?}");
                assert!(page.is_none_or(|p| !held.contains(&p)));
            }
            assert_eq!(sums, mine.iter().map(|&i| lens[i] as i64).collect::<Vec<_>>(), "w {w}");
            for _ in 0..w {
                let q = (g * w) as i64;
                answers.push(Answer { tokens: (0..b as i64).map(|k| 1000 * q + k).collect(), ..Answer::default() });
            }
        }
        // Row `k` of its group, read from the group's first member.
        let owner = |i: usize| rows[i].owner().index();
        let want: Vec<Vec<i64>> = (0..rows.len())
            .map(|i| vec![(1000 * owner(i) + (0..i).filter(|&j| owner(j) == owner(i)).count()) as i64])
            .collect();
        assert_eq!(st.emitted(r.protocol(), &f, &answers).unwrap(), want);
        // A member that disagrees with its group, even on a padding row.
        let q = rand.next(w * groups);
        let at = rand.next(answers[q].tokens.len());
        answers[q].tokens[at] += 1;
        let staged = rows.iter().any(|row| row.owner().index() == q / w * w);
        assert_eq!(st.emitted(r.protocol(), &f, &answers).is_err(), w > 1 && staged, "w {w}, rank {q}");
    }
}

#[test]
fn a_rank_sent_only_the_tables_that_changed_holds_every_steps_tables() {
    let mut rand = Rand(0xD1B5_4A32_D192_ED03);
    let (mut sent, mut staged) = (0usize, 0usize);
    for _ in 0..60 {
        let (w, groups) = ([1, 2, 4][rand.next(3)], 1 + rand.next(2));
        let mut r = replicated(w, groups);
        let tables: Vec<String> = {
            let p = r.protocol();
            p.page_tables.iter().map(|t| t.name.clone()).chain(p.line_tables.iter().map(|t| t.name.clone())).collect()
        };
        // Each rank's buffers as its writes leave them.
        let mut bufs = vec![std::collections::BTreeMap::<String, Vec<i64>>::new(); w * groups];
        let mut rows: Vec<(Row, usize)> = Vec::new();
        for _ in 0..40 {
            match rand.next(5) {
                0 => {
                    let held: Vec<usize> = (0..w * groups)
                        .map(|q| rows.iter().filter(|(row, _)| row.owner().index() == q).count())
                        .collect();
                    let pos = rand.next(30 * w);
                    if let Ok(row) = r.lease(pos + 41, |q| Some(held[q.index()]).filter(|&k| k < 8)) {
                        rows.push((row, pos));
                    }
                }
                1 if !rows.is_empty() => drop(rows.swap_remove(rand.next(rows.len()))),
                _ => {}
            }
            if rows.is_empty() {
                continue;
            }
            let cells: Vec<Cell> = rows.iter().map(|(row, pos)| Cell { row, ids: vec![1], pos: *pos }).collect();
            let mut st = r.stage(&cells, 1, |k| k.next_power_of_two()).unwrap();
            let f = st.forward(r.protocol(), 1).unwrap();
            let items = st.items(&f);
            let want: Vec<Vec<(String, Vec<i64>)>> = items
                .iter()
                .map(|i| tables.iter().filter_map(|n| Some((n.clone(), i.rows.get(n)?.clone()))).collect())
                .collect();
            drop(cells);
            for (q, o) in r.orders(Some(items), false).iter().enumerate() {
                let item = o.item.as_ref().unwrap();
                for (n, v) in &item.rows {
                    bufs[q].insert(n.clone(), v.clone());
                }
                for (n, v) in &want[q] {
                    assert_eq!(bufs[q].get(n), Some(v), "rank {q}, `{n}`");
                }
                (sent, staged) =
                    (sent + tables.iter().filter(|n| item.rows.contains_key(*n)).count(), staged + want[q].len());
            }
            rows.iter_mut().for_each(|(_, pos)| *pos += 1);
        }
    }
    assert!(sent * 2 < staged, "{sent} of {staged} tables sent");
}
