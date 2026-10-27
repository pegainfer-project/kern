//! The pooled tier's decisions through their public API: keys name tokens
//! and nothing else, a probe asks one batch and reads the answer, a
//! member's objects hold its own pages and slot and read back as tokens,
//! a landing is whole when every member answered.

mod common;

use std::collections::BTreeSet;

use common::Rand;
use kern_pool::{landed, objects, Chain, Keys, Landing, Probe};

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
fn keys_differ_by_kind_stage_member_tail_and_namespace() {
    let k = ns();
    let c = Chain::of(4, &toks(8));
    let keys = [
        k.page(None, c.at(2)),
        k.page(Some(0), c.at(2)),
        k.page(Some(1), c.at(2)),
        k.page(None, c.at(1)),
        k.state(None, 0, c.at(2), &[]),
        k.state(None, 1, c.at(2), &[]),
        k.state(Some(0), 0, c.at(2), &[]),
        k.state(None, 0, c.at(2), &[9]),
        k.state(None, 0, c.at(2), &[9, 9]),
        k.mark(c.at(2), &[]),
        k.mark(c.at(2), &[9]),
        Keys::new(&[b"manifest", b"other weights"]).page(None, c.at(2)),
    ];
    let distinct: BTreeSet<&String> = keys.iter().collect();
    assert_eq!((distinct.len(), keys.iter().all(|k| k.len() == 64)), (keys.len(), true));
    // The same tokens anywhere are the same key.
    assert_eq!(k.page(None, Chain::of(4, &toks(12)).at(2)), keys[0]);
}

#[test]
fn a_stateful_probe_asks_the_marks_and_the_pages_up_to_the_newest() {
    // Unit 4, 3 tokens on the device, marks at 14 and 9 (and 2, behind the hit; and 20, the last token).
    let p = Probe::plan(true, 4, 3, &[20, 14, 9, 2], 20, None).unwrap();
    assert_eq!((p.marks.clone(), p.pages.clone(), p.stages), (vec![14, 9], 1..4, None));
    let keys = p.keys(&ns(), &Chain::of(4, &toks(20)), &toks(20));
    assert_eq!(keys.len(), 2 + 3);
    // Everything there: the newest mark.
    assert_eq!(p.chosen(&[true, true, true, true, true]), Some(14));
    // Page 3 gone: 14 needs pages 1..=3, 9 needs 1..=2.
    assert_eq!(p.chosen(&[true, true, true, true, false]), Some(9));
    // Page 1 gone: no mark has its pages.
    assert_eq!(p.chosen(&[true, true, false, true, true]), None);
    // The newest mark missing, the older there with its pages.
    assert_eq!(p.chosen(&[false, true, true, true, true]), Some(9));
    // A mark inside the device hit's own page needs no page at all.
    let p = Probe::plan(true, 4, 5, &[7], 20, None).unwrap();
    assert_eq!((p.pages.clone(), p.chosen(&[true])), (2..2, Some(7)));
    // Nothing past the hit: nothing to ask.
    assert_eq!(Probe::plan(true, 4, 15, &[14, 9], 20, None), None);
}

#[test]
fn a_paged_probe_asks_every_page_on_every_stage_and_takes_the_longest_run() {
    // 19-token prompt, 4 on the device: pages 2..=4 (the 19th token runs through a program), 2 stages.
    let p = Probe::plan(false, 4, 4, &[], 19, Some(2)).unwrap();
    assert_eq!((p.marks.len(), p.pages.clone(), p.stages), (0, 2..5, Some(2)));
    assert_eq!(p.keys(&ns(), &Chain::of(4, &toks(19)), &toks(19)).len(), 6);
    assert_eq!(p.chosen(&[true, true, true, true, true, true]), Some(16));
    // Stage 1 lost page 3: the run stops at page 2.
    assert_eq!(p.chosen(&[true, true, true, false, true, true]), Some(8));
    assert_eq!(p.chosen(&[false, true, true, true, true, true]), None);
    // The last token is the only one past the hit: nothing to ask.
    assert_eq!(Probe::plan(false, 4, 8, &[], 9, None), None);
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
        let marks: Vec<usize> = (0..r.next(4)).map(|_| r.next(prompt + 2)).collect();
        let Some(p) = Probe::plan(stateful, unit, hit, &marks, prompt, stages) else {
            let any_mark = stateful && marks.iter().any(|&m| m > hit && m < prompt);
            let any_page = !stateful && hit / unit < (prompt - 1) / unit;
            assert!(
                !any_mark && !any_page,
                "nothing planned for unit {unit} hit {hit} prompt {prompt} marks {marks:?}"
            );
            continue;
        };
        let n = p.keys(&ns(), &Chain::of(unit, &toks(prompt)), &toks(prompt)).len();
        let exists: Vec<bool> = (0..n).map(|_| r.next(3) > 0).collect();
        let per = stages.map_or(1, usize::from);
        let page_ok = |k: usize| (0..per).all(|s| exists[p.marks.len() + (k - p.pages.start) * per + s]);
        let whole: BTreeSet<usize> = p.pages.clone().take_while(|&k| page_ok(k)).collect();
        let want = match stateful {
            true => p
                .marks
                .iter()
                .zip(&exists)
                .filter(|&(&m, &e)| e && (hit / unit + 1..=m / unit).all(|k| whole.contains(&k)))
                .map(|(&m, _)| m)
                .next(),
            false => whole.iter().max().map(|&k| k * unit),
        };
        assert_eq!(p.chosen(&exists), want, "unit {unit} hit {hit} prompt {prompt} marks {marks:?} exists {exists:?}");
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
    // The owner (member 1 of 1) moves pages 1..=2 and its state with the
    // page the prefix ends inside.
    let own = objects(&k, None, &chain, &t, 1, 1, 1, &pages, Some(9));
    let keys: Vec<&str> = own.iter().map(|o| o.key.as_str()).collect();
    assert_eq!(keys, [k.page(None, chain.at(1)), k.page(None, chain.at(2)), k.state(None, 1, chain.at(2), &t[8..])]);
    assert_eq!(
        own.iter().map(|o| (o.pages.clone(), o.slot)).collect::<Vec<_>>(),
        [(vec![40], None), (vec![41], None), (vec![42], Some(9))]
    );
    // Another member of the group moves its state alone, under its own key.
    let other = objects(&k, None, &chain, &t, 1, 0, 1, &[], Some(3));
    assert_eq!(
        other.iter().map(|o| (o.key.as_str(), o.pages.clone(), o.slot)).collect::<Vec<_>>(),
        [(k.state(None, 0, chain.at(2), &t[8..]).as_str(), vec![], Some(3))]
    );
    // From page 2, without a state, on a stage: the pages alone, keyed by it.
    let paged = objects(&k, Some(3), &chain, &t, 2, 0, 0, &pages, None);
    assert_eq!(
        paged.iter().map(|o| (o.key.as_str(), o.pages.clone(), o.slot)).collect::<Vec<_>>(),
        [(k.page(Some(3), chain.at(2)).as_str(), vec![41], None)]
    );
    // A prefix ending on a page boundary holds its state in the slot alone.
    let even = objects(&k, None, &Chain::of(unit, &t[..8]), &t[..8], 1, 0, 0, &pages, Some(9));
    assert_eq!(even.last().map(|o| (o.pages.clone(), o.slot)), Some((vec![], Some(9))));
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
