//! The pooled tier's decisions through their public API: keys name tokens
//! and nothing else, a probe asks one batch and reads the answer, a
//! prompt's snapshots fall on its page boundaries, a member's objects hold
//! its own pages and slot and read back as tokens, a landing is whole when
//! every member answered.

mod common;

use std::collections::BTreeSet;

use common::Rand;
use kern_pool::{grid, landed, objects, snapshots, Chain, Keys, Landing, Probe};

fn toks(n: usize) -> Vec<i64> {
    (0..n as i64).map(|i| i * 7 + 3).collect()
}

fn ns() -> Keys {
    Keys::new(&[b"manifest", b"weights"])
}

#[test]
fn a_chain_names_every_whole_page_by_the_tokens_before_it() {
    let t = toks(20);
    let whole = Chain::of(4, &t);
    assert_eq!(whole.pages(), 5);
    // Built token by token it is the same chain.
    let mut grown = Chain::new(4);
    for tok in &t {
        grown.extend(std::slice::from_ref(tok));
    }
    assert_eq!(grown, whole);
    // A prefix shares every page up to where it ends.
    let short = Chain::of(4, &t[..10]);
    assert_eq!((short.pages(), short.at(2), short.at(0)), (2, whole.at(2), whole.at(0)));
    // One token changed in page 2 changes page 2 and everything after, not before.
    let mut u = t.clone();
    u[5] += 1;
    let other = Chain::of(4, &u);
    assert_eq!(
        (other.at(1) == whole.at(1), other.at(2) == whole.at(2), other.at(5) == whole.at(5)),
        (true, false, false)
    );
}

#[test]
fn keys_differ_by_kind_stage_member_and_namespace() {
    let k = ns();
    let c = Chain::of(4, &toks(8));
    let keys = [
        k.page(None, c.at(2)),
        k.page(Some(0), c.at(2)),
        k.page(Some(1), c.at(2)),
        k.page(None, c.at(1)),
        k.state(None, 0, c.at(2)),
        k.state(None, 1, c.at(2)),
        k.state(Some(0), 0, c.at(2)),
        k.state(None, 0, c.at(1)),
        k.mark(c.at(2)),
        k.mark(c.at(1)),
        Keys::new(&[b"manifest", b"other weights"]).page(None, c.at(2)),
    ];
    let distinct: BTreeSet<&String> = keys.iter().collect();
    assert_eq!((distinct.len(), keys.iter().all(|k| k.len() == 64)), (keys.len(), true));
    // The same tokens anywhere are the same key.
    assert_eq!(k.page(None, Chain::of(4, &toks(12)).at(2)), keys[0]);
}

#[test]
fn a_stateful_probe_asks_a_mark_on_every_page_boundary_and_takes_the_newest_with_its_pages() {
    // Unit 4, 3 tokens on the device, a 20-token prompt: pages 1..=4 (the 20th token runs through a program).
    let p = Probe::plan(true, 4, 3, 20, None).unwrap();
    assert_eq!((p.pages.clone(), p.stages), (1..5, None));
    let keys = p.keys(&ns(), &Chain::of(4, &toks(20)));
    // Marks at 16, 12, 8, 4, then pages 1..=4.
    let c = Chain::of(4, &toks(20));
    assert_eq!((keys.len(), keys[0].clone(), keys[4].clone()), (8, ns().mark(c.at(4)), ns().page(None, c.at(1))));
    // Everything there: the newest mark.
    assert_eq!(p.chosen(&[true, true, true, true, true, true, true, true]), Some(16));
    // Page 3 gone: 16 and 12 need it, 8 does not.
    assert_eq!(p.chosen(&[true, true, true, true, true, true, false, true]), Some(8));
    // Page 1 gone: no mark has its pages.
    assert_eq!(p.chosen(&[true, true, true, true, false, true, true, true]), None);
    // Only an old mark left, with its pages.
    assert_eq!(p.chosen(&[false, false, false, true, true, true, true, true]), Some(4));
    // A mark inside the device hit's own page: none asked there.
    let p = Probe::plan(true, 4, 5, 9, None).unwrap();
    assert_eq!((p.pages.clone(), p.chosen(&[true, true])), (2..3, Some(8)));
    // Nothing past the hit's page: nothing to ask.
    assert_eq!(Probe::plan(true, 4, 15, 16, None), None);
}

#[test]
fn a_paged_probe_asks_every_page_on_every_stage_and_takes_the_longest_run() {
    // 19-token prompt, 4 on the device: pages 2..=4 (the 19th token runs through a program), 2 stages.
    let p = Probe::plan(false, 4, 4, 19, Some(2)).unwrap();
    assert_eq!((p.pages.clone(), p.stages), (2..5, Some(2)));
    assert_eq!(p.keys(&ns(), &Chain::of(4, &toks(19))).len(), 6);
    assert_eq!(p.chosen(&[true, true, true, true, true, true]), Some(16));
    // Stage 1 lost page 3: the run stops at page 2.
    assert_eq!(p.chosen(&[true, true, true, false, true, true]), Some(8));
    assert_eq!(p.chosen(&[false, true, true, true, true, true]), None);
    // The last token is the only one past the hit: nothing to ask.
    assert_eq!(Probe::plan(false, 4, 8, 9, None), None);
}

#[test]
fn a_probe_agrees_with_the_brute_force_model() {
    let mut r = Rand::new(11);
    for _ in 0..2000 {
        let unit = 1 + r.next(4);
        let prompt = 2 + r.next(40);
        let hit = r.next(prompt);
        let stateful = r.next(2) == 0;
        let stages = (r.next(2) == 0).then(|| 1 + r.next(3) as u8);
        // Every page past the hit's whole pages and short of the last token.
        let asked: Vec<usize> = (1..=(prompt - 1) / unit).filter(|&k| k * unit > hit / unit * unit).collect();
        let Some(p) = Probe::plan(stateful, unit, hit, prompt, stages) else {
            assert!(asked.is_empty(), "nothing planned for unit {unit} hit {hit} prompt {prompt}");
            continue;
        };
        let n = p.keys(&ns(), &Chain::of(unit, &toks(prompt))).len();
        let per = stages.map_or(1, usize::from);
        let marks = if stateful { asked.len() } else { 0 };
        assert_eq!(n, marks + asked.len() * per);
        let exists: Vec<bool> = (0..n).map(|_| r.next(3) > 0).collect();
        let mark_ok = |k: usize| exists[asked.iter().rev().position(|&a| a == k).unwrap()];
        let page_ok = |k: usize| (0..per).all(|s| exists[marks + (k - asked[0]) * per + s]);
        let held = |k: usize| asked.iter().filter(|&&a| a <= k).all(|&a| page_ok(a));
        let want = match stateful {
            true => asked.iter().rev().find(|&&k| mark_ok(k) && held(k)),
            false => asked.iter().rev().find(|&&k| held(k)),
        };
        assert_eq!(
            p.chosen(&exists),
            want.map(|k| k * unit),
            "unit {unit} hit {hit} prompt {prompt} exists {exists:?}"
        );
    }
}

#[test]
fn snapshots_fall_on_the_grid_and_the_last_boundary_past_the_start() {
    // Unit 4, a grid of 12, fed from 5 to 30: 12 and 24 on the grid, 28 the last boundary.
    assert_eq!(snapshots(4, 12, 5, 30), [12, 24, 28]);
    // The end on the grid is the last boundary, once.
    assert_eq!(snapshots(4, 12, 0, 24), [12, 24]);
    // Nothing past the start.
    assert_eq!(snapshots(4, 12, 25, 27), Vec::<usize>::new());
    // The spacing covers twice a state's bytes in whole pages: a 100-byte
    // state over 3-byte tokens wants 67 tokens, 17 pages of 4.
    assert_eq!((grid(4, 100, 3), grid(4, 0, 3), grid(64, 1, 0)), (68, 4, 64));
    let mut r = Rand::new(5);
    for _ in 0..2000 {
        let unit = 1 + r.next(8);
        let g = unit * (1 + r.next(5));
        let start = r.next(100);
        let end = start + r.next(100);
        let s = snapshots(unit, g, start, end);
        let want: Vec<usize> = (start + 1..=end).filter(|&p| p % unit == 0 && (p % g == 0 || p + unit > end)).collect();
        assert_eq!(s, want, "unit {unit} grid {g} start {start} end {end}");
    }
}

#[test]
fn a_landing_is_whole_once_every_member_answered_at_the_fewest_tokens() {
    let l = Landing::new([0, 2, 3]);
    assert_eq!(l.done(), None);
    let l = l.heard(2, 5).unwrap().heard(0, 7).unwrap();
    assert_eq!(l.done(), None);
    let l = l.heard(3, 6).unwrap();
    assert_eq!(l.done(), Some(5));
    // A member not expected, or answering twice, is a caller bug.
    assert!(Landing::new([0]).heard(1, 1).is_err());
    assert!(Landing::new([0]).heard(0, 1).unwrap().heard(0, 1).is_err());
    // A failed member lands nothing.
    assert_eq!(Landing::new([0, 1]).heard(0, 0).unwrap().heard(1, 4).unwrap().done(), Some(0));
    // Nobody to wait for is whole at once.
    assert_eq!(Landing::new([]).done(), Some(0));
    // Any arrival order gives the same answer.
    let mut r = Rand::new(3);
    for _ in 0..200 {
        let n = 1 + r.next(5);
        let tokens: Vec<usize> = (0..n).map(|_| r.next(9)).collect();
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            order.swap(i, r.next(i + 1));
        }
        let mut l = Landing::new(0..n);
        for (i, &m) in order.iter().enumerate() {
            assert_eq!(l.done().is_some(), i == n, "whole before everyone answered");
            l = l.heard(m, tokens[m]).unwrap();
        }
        assert_eq!(l.done(), tokens.iter().copied().min());
    }
}

#[test]
fn a_members_objects_are_its_own_pages_and_slot_under_the_prefix_keys() {
    let (k, t, unit) = (ns(), toks(10), 4);
    let chain = Chain::of(unit, &t);
    let pages = [40, 41, 42];
    let (t8, c8) = (&t[..8], Chain::of(unit, &t[..8]));
    // The owner (member 1 of 1) moves pages 1..=2 and its state.
    let own = objects(&k, None, &c8, t8, 1, 1, 1, &pages, Some(9));
    let keys: Vec<&str> = own.iter().map(|o| o.key.as_str()).collect();
    assert_eq!(keys, [k.page(None, c8.at(1)), k.page(None, c8.at(2)), k.state(None, 1, c8.at(2))]);
    assert_eq!(
        own.iter().map(|o| (o.pages.clone(), o.slot)).collect::<Vec<_>>(),
        [(vec![40], None), (vec![41], None), (vec![], Some(9))]
    );
    // Another member of the group moves its state alone, under its own key.
    let other = objects(&k, None, &c8, t8, 1, 0, 1, &[], Some(3));
    assert_eq!(
        other.iter().map(|o| (o.key.as_str(), o.pages.clone(), o.slot)).collect::<Vec<_>>(),
        [(k.state(None, 0, c8.at(2)).as_str(), vec![], Some(3))]
    );
    // From page 2, without a state, on a stage: the pages alone, keyed by it.
    let paged = objects(&k, Some(3), &chain, &t, 2, 0, 0, &pages, None);
    assert_eq!(
        paged.iter().map(|o| (o.key.as_str(), o.pages.clone(), o.slot)).collect::<Vec<_>>(),
        [(k.page(Some(3), chain.at(2)).as_str(), vec![41], None)]
    );
    // Without a state a prefix may end inside a page, which is not moved.
    assert_eq!(objects(&k, None, &chain, &t, 1, 0, 0, &pages, None).len(), 2);
}

#[test]
#[should_panic(expected = "not on a page")]
fn a_state_off_a_page_boundary_is_not_pooled() {
    let t = toks(10);
    objects(&ns(), None, &Chain::of(4, &t), &t, 1, 0, 0, &[40, 41, 42], Some(9));
}

#[test]
fn a_transfer_reads_back_as_tokens_all_or_nothing_with_a_state_by_the_page_without() {
    assert_eq!(landed(&[true, true, true], true, 10, 1, 4), 10);
    assert_eq!(landed(&[true, false, true], true, 10, 1, 4), 0);
    assert_eq!(landed(&[true, true, false], false, 12, 1, 4), 8);
    assert_eq!(landed(&[true], false, 12, 3, 4), 12);
    assert_eq!(landed(&[false, true], false, 12, 2, 4), 4);
    assert_eq!(landed(&[], false, 4, 2, 4), 4);
    // Every member's flags over a random group land what the
    // brute-force model says, and the landing takes the fewest.
    let mut r = Rand::new(11);
    for _ in 0..300 {
        let (unit, first) = (1 + r.next(4), 1 + r.next(3));
        let len = first * unit + r.next(3 * unit);
        let stateful = r.next(2) == 0;
        let members = 1 + r.next(3);
        let flags: Vec<Vec<bool>> = (0..members)
            .map(|_| (0..(len / unit - first + 1) + usize::from(stateful)).map(|_| r.next(4) != 0).collect())
            .collect();
        let want = |ok: &[bool]| match stateful {
            true => len * usize::from(ok.iter().all(|&b| b)),
            false => (first - 1 + ok.iter().position(|&b| !b).unwrap_or(ok.len())) * unit,
        };
        let mut l = Landing::new(0..members);
        for (m, ok) in flags.iter().enumerate() {
            assert_eq!(landed(ok, stateful, len, first, unit), want(ok), "{ok:?} stateful {stateful}");
            l = l.heard(m, landed(ok, stateful, len, first, unit)).unwrap();
        }
        assert_eq!(l.done(), flags.iter().map(|ok| want(ok)).min());
    }
}
