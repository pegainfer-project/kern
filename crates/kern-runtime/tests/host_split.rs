//! The host tier's byte budget split into host pages and host slots: it
//! fits, it holds one slot per `turn` pages as nearly as whole units allow,
//! and what is left over is less than a unit the budget could still have held.

use kern_runtime::host_split;

#[test]
fn a_budget_splits_into_whole_units_one_slot_per_turn() {
    let sizes = [0u64, 1, 3, 4096, 221_184, 49 << 20, 147 << 20, (1 << 30) + 7];
    let budgets = [0u64, 1, 5, 4095, 1 << 20, 128 << 30, 512 << 30, u64::MAX / 4];
    let turns = [0u64, 1, 11, 128, 1 << 40];
    for &bytes in &budgets {
        for &page in &sizes {
            for &slot in &sizes {
                for &turn in &turns {
                    let (p, s) = host_split(bytes, page, slot, turn);
                    let held = p as u128 * page as u128 + s as u128 * slot as u128;
                    let left = bytes as u128 - held;
                    let at = format!("{bytes} bytes, page {page}, slot {slot}, turn {turn}: {p} pages {s} slots");
                    assert!(held <= bytes as u128, "{at}");
                    assert_eq!((page == 0 && p > 0, slot == 0 && s > 0), (false, false), "{at}");
                    match (page, slot) {
                        (0, 0) => assert_eq!((p, s), (0, 0), "{at}"),
                        (0, slot) => assert!(left < slot as u128, "{at}"),
                        (page, 0) => assert!(left < page as u128, "{at}"),
                        (page, slot) => {
                            let entry = turn as u128 * page as u128 + slot as u128;
                            assert_eq!(s as u128, bytes as u128 / entry, "{at}");
                            assert!(left < page as u128, "{at}");
                            assert!(p as u128 >= s as u128 * turn as u128, "{at}");
                        }
                    }
                }
            }
        }
    }
}
