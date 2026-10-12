//! What a push decides without a device: how a slot splits over a
//! tensor-parallel group, how a member's dealt rows cut into the gather's
//! pieces, and the pool handle's wire form.

use kern_pool::{deal, Interleave, Pages};
use kern_runtime::{PeerHandle, PoolHandle, Rows, Split};

/// Byte ranges sorted by start tile `[0, total)` exactly: no gap, no overlap.
fn tiles(mut r: Vec<(u64, u64)>, total: u64) -> bool {
    r.sort_unstable();
    r.iter().try_fold(0, |at, &(lo, len)| (lo == at && len > 0).then_some(at + len)) == Some(total)
}

/// Every list of up to three segments of 1–3 rows, each row 1 or 2 bytes a member.
fn segment_lists(of: u64) -> Vec<Vec<(u64, u64)>> {
    let one: Vec<(u64, u64)> = (1..=3).flat_map(|r| (1..=2).map(move |w| (r, w * of))).collect();
    let longer = |lists: &[Vec<(u64, u64)>]| -> Vec<Vec<(u64, u64)>> {
        lists.iter().flat_map(|l| one.iter().map(move |&s| [l.as_slice(), &[s]].concat())).collect()
    };
    let ones: Vec<Vec<(u64, u64)>> = one.iter().map(|&s| vec![s]).collect();
    let twos = longer(&ones);
    let threes = longer(&twos);
    [ones, twos, threes].concat()
}

#[test]
fn the_parts_of_a_slot_tile_it_and_each_part_tiles_its_shard() {
    let mut cases = 0;
    for of in 1..=4 {
        for segments in segment_lists(of) {
            let s = Split::new(&segments, of).unwrap();
            let rows: u64 = segments.iter().map(|(r, _)| r).sum();
            for lines in 1..=3 {
                let slot = lines * s.line();
                let mut sources = Vec::new();
                for part in 0..of {
                    let copies = s.copies(lines, part);
                    assert_eq!(copies.len() as u64, rows, "one copy per row of a segment");
                    assert!(copies.iter().all(|c| c.rows == lines && c.src_pitch >= c.width && c.dst_pitch >= c.width));
                    let r = s.ranges(lines, part);
                    assert!(
                        tiles(r.iter().map(|&(_, d, n)| (d, n)).collect(), slot / of),
                        "{segments:?}/{of} part {part}"
                    );
                    sources.extend(r.iter().map(|&(src, _, n)| (src, n)));
                }
                assert!(tiles(sources, slot), "{segments:?} over {of}, {lines} lines");
                cases += 1;
            }
        }
    }
    assert_eq!(cases, 4 * 3 * (6 + 36 + 216));
}

#[test]
fn a_heads_major_line_splits_into_one_copy_per_row() {
    // A recurrent block of 96 heads × 128 × 128 f32, then three conv
    // windows of three taps over 96 × 128 bf16: nine rows.
    let s = Split::new(&[(1, 96 * 128 * 128 * 4), (9, 96 * 128 * 2)], 4).unwrap();
    let c = s.copies(69, 3);
    assert_eq!((c.len(), s.line() / 4), (10, 24 * 128 * 128 * 4 + 9 * 24 * 128 * 2));
    assert_eq!((c[0].src, c[0].dst, c[0].width, c[0].rows), (3 * 24 * 65536, 0, 24 * 65536, 69));
    assert_eq!(
        (c[1].src, c[1].dst, c[1].src_pitch, c[1].dst_pitch),
        (96 * 65536 + 3 * 6144, 24 * 65536, s.line(), s.line() / 4)
    );
}

#[test]
fn a_split_its_members_cannot_divide_is_refused() {
    let bad = |segments: &[(u64, u64)], of| Split::new(segments, of).is_err();
    assert_eq!(
        (bad(&[(1, 6)], 4), bad(&[], 1), bad(&[(0, 8)], 2), bad(&[(1, 8)], 0), bad(&[(2, 8), (1, 12)], 4)),
        (true, true, true, true, false)
    );
}

#[test]
fn a_pool_handle_crosses_the_wire_unchanged() {
    let chunk = 1u64 << 26;
    let mut b = chunk.to_le_bytes().to_vec();
    b.extend(64u64.to_le_bytes());
    b.extend(2u32.to_le_bytes());
    for (name, kind, object, positions) in [("kv", 0u8, 1u64 << 16, 3u64), ("mem", 1, 1 << 20, 5)] {
        b.extend((name.len() as u32).to_le_bytes());
        b.extend(name.as_bytes());
        b.push(kind);
        b.extend(object.to_le_bytes());
        b.extend(positions.to_le_bytes());
    }
    b.extend(2u32.to_le_bytes());
    for i in 0..2u8 {
        let mut h = [i; PeerHandle::BYTES];
        h[64..].copy_from_slice(&chunk.to_le_bytes());
        b.extend(PeerHandle::from_bytes(&h).unwrap().to_bytes().unwrap());
    }
    let h = PoolHandle::from_bytes(&b).expect("a handle");
    assert_eq!(h.to_bytes().as_deref(), Some(&b[..]));
    assert_eq!(
        (PoolHandle::from_bytes(&b[..b.len() - 1]), PoolHandle::from_bytes(&[b.as_slice(), &[0]].concat())),
        (None, None)
    );
}

/// `n` page ids in runs of three adjacent pages at scattered bases, so
/// some page boundaries fold and some do not.
fn scattered(n: usize) -> Vec<i32> {
    (0..n).map(|k| ((k / 3 * 37) % 101 * 4 + k % 3) as i32).collect()
}

#[test]
fn a_member_s_pieces_move_exactly_its_dealt_rows() {
    let mut cases = 0;
    for w in 1..=9 {
        for (su, du) in [(1, 1), (8, 64), (64, 64), (64, 5), (3, 16)] {
            for len in [0u64, 1, 7, 63, 64, 65, 200, 1001] {
                let il = Interleave::new(w).unwrap();
                let (s, d) = (scattered(len.div_ceil(su) as usize), scattered(il.len(len, 0).div_ceil(du) as usize));
                let (src, dst) = (Pages { ids: &s, unit: su }, Pages { ids: &d, unit: du });
                for m in 0..w {
                    let dl = deal(src, dst, il, m, len).unwrap();
                    for row in [16, 1152, Rows::PIECE + 16] {
                        let want: Vec<(u64, u64, u64)> = dl
                            .runs()
                            .iter()
                            .flat_map(|r| {
                                (0..r.rows).map(move |i| ((r.src_row + i * w) * row, (r.dst_row + i) * row, row))
                            })
                            .collect();
                        let pieces = Rows::dealt(&dl, row);
                        let got: Vec<(u64, u64, u64)> = pieces.iter().flat_map(Rows::ranges).collect();
                        let ctx = format!("w {w} units {su}/{du} len {len} member {m} row {row}");
                        assert_eq!(got, want, "{ctx}");
                        assert!(
                            pieces.iter().all(|p| p.rows > 0 && (p.rows == 1 || p.rows * row <= Rows::PIECE)),
                            "{ctx}: {pieces:?}"
                        );
                        cases += 1;
                    }
                }
            }
        }
    }
    assert_eq!(cases, 45 * 5 * 8 * 3);
}

#[test]
fn a_k3_prompt_cuts_into_a_block_per_page_or_per_64_kib() {
    let (len, row, il) = (219_000, 1152, Interleave::new(8).unwrap());
    let (s, d): (Vec<i32>, Vec<i32>) = ((0..3422).collect(), (0..428).collect());
    let adjacent = deal(Pages { ids: &s, unit: 64 }, Pages { ids: &d, unit: 64 }, il, 3, len).unwrap();
    let p = Rows::dealt(&adjacent, row);
    assert_eq!((adjacent.runs().len(), p.len(), p[0].rows, p[488].rows), (1, 489, 56, 27375 - 488 * 56));
    assert_eq!((p[1].src, p[1].src_pitch, p[1].dst, p[1].width), ((56 * 8 + 3) * row, 8 * row, 56 * row, row));
    let s: Vec<i32> = (0..3422).map(|k| k * 2).collect();
    let apart = deal(Pages { ids: &s, unit: 64 }, Pages { ids: &d, unit: 64 }, il, 3, len).unwrap();
    let p = Rows::dealt(&apart, row);
    assert_eq!((p.len(), p[0].rows, p[1].src, p[1].dst), (3422, 8, (128 + 3) * row, 8 * row));
}
