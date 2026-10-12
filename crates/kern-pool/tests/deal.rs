//! Dealing a sequence over a group, against a per-row model: position `p`
//! of the source's pages lands on member `p % w` at local position
//! `p / w` of that member's pages, and the runs say exactly that, in as
//! few runs as the pages allow.

mod common;

use kern_pool::{deal, Interleave, Pages, RowRun};

use common::Rand;

/// Page ids for `pages` pages: runs of physically adjacent pages at
/// scattered bases, so some boundaries fold and some do not.
fn page_ids(r: &mut Rand, pages: usize) -> Vec<i32> {
    let mut ids = Vec::with_capacity(pages);
    while ids.len() < pages {
        let base = r.next(1000) as i32;
        let run = 1 + r.next(4);
        ids.extend((0..run).map(|i| base + i as i32).take(pages - ids.len()));
    }
    ids
}

/// The model: (source row, destination row) of every position member `m`
/// holds, in local order.
fn model(src: Pages, dst: Pages, w: u64, m: u64, len: u64) -> Vec<(u64, u64)> {
    let row = |p: Pages, i: u64| p.ids[(i / p.unit) as usize] as u64 * p.unit + i % p.unit;
    (0..len).filter(|p| p % w == m).map(|p| (row(src, p), row(dst, p / w))).collect()
}

fn expand(runs: &[RowRun]) -> Vec<(u64, u64)> {
    runs.iter().flat_map(|r| (0..r.rows).map(move |i| (r.src_row + i * r.src_stride, r.dst_row + i))).collect()
}

#[test]
fn every_row_lands_on_its_owner_at_its_local_position() {
    let mut r = Rand::new(0x9e37_79b9_7f4a_7c15);
    for case in 0..3000 {
        let w = 1 + r.next(9) as u64;
        let src_unit = [1, 2, 3, 4, 8, 16, 64][r.next(7)];
        let dst_unit = [1, 2, 4, 5, 8, 64][r.next(6)];
        let len = r.next(300) as u64;
        let il = Interleave::new(w).unwrap();
        let (src_spare, dst_spare) = (r.next(2), r.next(2));
        let src_ids = page_ids(&mut r, len.div_ceil(src_unit) as usize + src_spare);
        let dst_ids = page_ids(&mut r, il.len(len, 0).div_ceil(dst_unit) as usize + dst_spare);
        let (src, dst) = (Pages { ids: &src_ids, unit: src_unit }, Pages { ids: &dst_ids, unit: dst_unit });
        let mut total = 0;
        for m in 0..w {
            let d = deal(src, dst, il, m, len).unwrap();
            let ctx = format!("case {case}: w {w} units {src_unit}/{dst_unit} len {len} member {m}");
            assert_eq!(expand(d.runs()), model(src, dst, w, m, len), "{ctx}");
            assert_eq!(d.rows(), il.len(len, m), "{ctx}");
            assert!(d.runs().iter().all(|x| x.rows > 0 && x.src_stride == w), "{ctx}: {:?}", d.runs());
            let mergeable =
                |a: &RowRun, b: &RowRun| a.src_row + a.rows * w == b.src_row && a.dst_row + a.rows == b.dst_row;
            assert!(!d.runs().windows(2).any(|p| mergeable(&p[0], &p[1])), "{ctx}: not maximal {:?}", d.runs());
            total += d.rows();
        }
        assert_eq!(total, len, "case {case}");
    }
}

#[test]
fn interleave_counts_every_position_once() {
    for w in 1..=9u64 {
        let il = Interleave::new(w).unwrap();
        for n in 0..100u64 {
            let held: Vec<u64> = (0..w).map(|m| (0..n).filter(|&p| il.owner(p) == m).count() as u64).collect();
            assert_eq!(held, (0..w).map(|m| il.len(n, m)).collect::<Vec<_>>(), "w {w} n {n}");
            assert_eq!((il.len(n, 0), held.iter().sum::<u64>()), (n.div_ceil(w), n), "w {w} n {n}");
        }
        for p in 0..100u64 {
            assert_eq!((il.owner(p) < w, il.local(p) * w + il.owner(p)), (true, p), "w {w} p {p}");
            assert_eq!(il.len(p + 1, il.owner(p)), il.local(p) + 1, "w {w} p {p}");
        }
    }
}

#[test]
fn a_prefill_page_is_an_eighth_of_a_local_page() {
    // 64-row pages over eight members: source page k gives member m its
    // rows m, m + 8, ..., m + 56, into local page k / 8 at rows
    // 8 (k % 8) to 8 (k % 8) + 7. Scattered pages fold nothing; adjacent
    // ones on both sides fold into one run.
    let il = Interleave::new(8).unwrap();
    let (src, dst) = (vec![40, 7, 12, 3, 90, 33, 5, 61, 18], vec![4, 9]);
    let d = deal(Pages { ids: &src, unit: 64 }, Pages { ids: &dst, unit: 64 }, il, 3, 9 * 64).unwrap();
    assert_eq!(d.runs().len(), 9);
    for (k, run) in d.runs().iter().enumerate() {
        let local = dst[k / 8] as u64 * 64 + 8 * (k as u64 % 8);
        assert_eq!(
            *run,
            RowRun { src_row: src[k] as u64 * 64 + 3, src_stride: 8, dst_row: local, rows: 8 },
            "page {k}"
        );
    }
    let (src, dst) = ((100..109).collect::<Vec<i32>>(), vec![20, 21]);
    let d = deal(Pages { ids: &src, unit: 64 }, Pages { ids: &dst, unit: 64 }, il, 5, 9 * 64 - 2).unwrap();
    assert_eq!(d.runs(), [RowRun { src_row: 100 * 64 + 5, src_stride: 8, dst_row: 20 * 64, rows: 72 }]);
}

#[test]
fn a_deal_names_only_rows_its_pages_hold() {
    let il = Interleave::new(4).unwrap();
    let pages = |ids: &'static [i32], unit| Pages { ids, unit };
    let refused = |src, dst, m, len| deal(src, dst, il, m, len).expect_err("dealt").to_string();
    let cases = [
        (refused(pages(&[0, 1], 4), pages(&[0], 4), 4, 8), "member 4 of a group of 4"),
        (refused(pages(&[0, 1], 0), pages(&[0], 4), 0, 8), "pages of 0 source and 4 destination rows"),
        (refused(pages(&[0], 4), pages(&[0], 4), 0, 8), "source holds 1 pages of 4, expected 2 for 8 rows"),
        (refused(pages(&[0, 1, 2], 4), pages(&[0], 2), 0, 12), "destination holds 1 pages of 2, expected 2 for 3 rows"),
        (refused(pages(&[0, -1], 4), pages(&[0], 4), 1, 8), "source page id -1 among the 2 pages dealt"),
    ];
    for (got, want) in cases {
        assert!(got.contains(want), "`{got}` lacks `{want}`");
    }
    assert!(Interleave::new(0).is_err());
    let d = deal(pages(&[], 4), pages(&[], 4), il, 2, 0).unwrap();
    assert_eq!((d.runs().len(), d.rows()), (0, 0));
    let d = deal(pages(&[7, -1], 4), pages(&[3], 4), il, 2, 3).unwrap();
    assert_eq!(d.runs(), [RowRun { src_row: 30, src_stride: 4, dst_row: 12, rows: 1 }]);
}
