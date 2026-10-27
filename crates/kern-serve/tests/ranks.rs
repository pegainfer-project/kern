//! The head's accounting through its public API, over real pools of a
//! tiny group manifest (kern-manifest's `ranks.json`: four ranks in one
//! `tp` group, 16-token pages of 16 bytes, a 24-byte recurrent slot).

use std::collections::BTreeMap;

use infront_engine::SamplingParams;
use kern_manifest::protocol::{Sampling, Values};
use kern_manifest::types::Manifest;
use kern_manifest::{verify, Protocol};
use kern_pool::{Denied, Pool};
use kern_serve::ranks::{sampling, Act, Answer, Cell, Groups, Order, Ranks, Row};

mod common;
use common::{dealt, replicated, Rand};

fn manifest() -> Manifest {
    Manifest::from_json(include_str!("../../kern-manifest/tests/fixtures/ranks.json")).unwrap()
}

/// 5 pages and `slots` sequence slots (slot 0 the null one, slot 1 the pad's).
fn pool(slots: usize) -> Pool {
    Pool::new(&manifest(), 5, slots).unwrap()
}

fn ranks(slots: [usize; 4]) -> Ranks {
    ranks_of(manifest(), slots)
}

fn ranks_of(m: Manifest, slots: [usize; 4]) -> Ranks {
    let p = Protocol::check(&verify(m).unwrap()).unwrap();
    Ranks::new(p, Groups::new(4, 4).unwrap(), slots.map(pool).into()).unwrap()
}

/// The same manifest drawing each sequence's token over the sampling fills.
fn sampled() -> Manifest {
    let mut m = serde_json::to_value(manifest()).unwrap();
    let fills = [("temperature", "f32"), ("top_k", "i32"), ("top_p", "f32"), ("min_p", "f32"), ("seed", "i64")];
    for (name, dtype) in fills {
        m["buffers"][name] = serde_json::json!({"kind": "input", "dtype": dtype, "shape": ["seqs"], "fill": name});
        let param = serde_json::json!(format!("in buffer<{dtype}>"));
        m["ops"]["step"]["params"].as_array_mut().unwrap().push(param);
        m["programs"]["decode"]["calls"][0]["args"].as_array_mut().unwrap().push(serde_json::json!({"buf": name}));
    }
    Manifest::from_json(&m.to_string()).unwrap()
}

fn acts(orders: &[Order]) -> Vec<Vec<Act>> {
    orders.iter().map(|o| o.acts.clone()).collect()
}

#[test]
fn a_row_is_every_members_or_none() {
    let mut r = ranks([3, 3, 2, 3]);
    let before = (r.pages_used(), r.seq_slots());
    // Rank 2 has no slot free: the owner's pages and rank 1's slot go back.
    assert_eq!(r.lease(20, |_| Some(0)).err(), Some(Denied::Busy));
    assert_eq!((r.pages_used(), r.seq_slots()), (before.0, before.1));
    assert_eq!(acts(&r.orders(None)), vec![vec![Act::Zero(1)]; 4]);
    // With a slot on every member: the owner pages, its peers a slot alone.
    let mut r = ranks([3; 4]);
    r.orders(None);
    let row = r.lease(20, |_| Some(0)).unwrap();
    assert_eq!((row.owner().index(), row.prefix(), r.pages_used()), (0, 0, 2));
    assert_eq!(acts(&r.orders(None)), vec![vec![Act::Zero(2)]; 4]);
    // No rank open is no lease.
    assert_eq!(r.lease(20, |_| None).err(), Some(Denied::Busy));
}

#[test]
fn a_step_gives_every_rank_the_same_program_and_vars_over_its_own_rows() {
    let mut r = ranks([4; 4]);
    let a = r.lease(20, |q| Some(q.index())).unwrap();
    let b = r.lease(20, |q| Some(if q.index() == 0 { 9 } else { q.index() })).unwrap();
    assert_eq!((a.owner().index(), b.owner().index()), (0, 1));
    let cells = [
        Cell { row: &a, ids: vec![7], pos: 3, sampling: Sampling::GREEDY },
        Cell { row: &b, ids: vec![8], pos: 5, sampling: Sampling::GREEDY },
    ];
    let mut st = r.stage(&cells, 1, |k| k).unwrap();
    let f = st.forward(r.protocol(), 1).unwrap();
    let items = st.items(&f);
    assert_eq!(
        items.iter().map(|i| (i.program.as_str(), i.vars.clone())).collect::<std::collections::BTreeSet<_>>().len(),
        1
    );
    // The batch is every member's block, this rank's own first.
    let tokens: Vec<Vec<i64>> = items.iter().map(|i| i.rows["token_ids"].ints().unwrap().to_vec()).collect();
    assert_eq!(tokens, [vec![7, 8, 0, 0], vec![8, 0, 0, 7], vec![0, 0, 7, 8], vec![0, 7, 8, 0]]);
    let lens: Vec<Vec<i64>> = items.iter().map(|i| i.rows["seq_lens"].ints().unwrap().to_vec()).collect();
    assert_eq!(lens, [vec![4], vec![6], vec![1], vec![1]]);
    assert_eq!(items[2].rows["tp_blocks"].ints().unwrap(), [0, 1, 2, 3, 4]);
    // Each cell's token is read from its owner's answer.
    let answers: Vec<Answer> = (0..4).map(|q| Answer { tokens: vec![100 + q, 0, 0, 0], ..Answer::default() }).collect();
    assert_eq!(st.emitted(r.protocol(), &f, &answers).unwrap(), [vec![100], vec![101]]);
    let o = r.orders(Some(items));
    assert_eq!(o.iter().map(|o| o.item.is_some()).collect::<Vec<_>>(), [true; 4]);
}

#[test]
fn a_replicated_row_is_paged_on_every_member_and_keeps_nothing() {
    let mut r = replicated(4, 1);
    r.orders(None);
    // Four members of 48 local positions hold 192; past that is past a row.
    assert_eq!(r.max_seq_tokens(), 192);
    assert_eq!(r.lease(193, |_| Some(0)).err(), Some(Denied::ExceedsRow { limit: 192 }));
    // ⌈37 / 4⌉ = 10 local positions: a page on every member, every slot zeroed.
    let mut row = r.lease(37, |_| Some(0)).unwrap();
    let pages: Vec<usize> = row.parts().iter().map(|l| l.pages()).collect();
    assert_eq!((row.owner().index(), pages, r.pages_used()), (0, vec![1; 4], 4));
    let zeros: Vec<Vec<Act>> = row.parts().iter().map(|l| vec![Act::Zero(l.seq_slot().unwrap())]).collect();
    assert_eq!(acts(&r.orders(None)), zeros);
    // An adopted row's slots are the push's: nothing is zeroed.
    let pushed = r.adopt(37, |_| Some(1)).unwrap();
    assert_eq!((acts(&r.orders(None)), r.pages_used()), (vec![vec![]; 4], 8));
    // Nothing is kept: a checkpoint makes no snapshot, a retire drops the row.
    assert_eq!((r.checkpoint(&mut row, 16).unwrap().is_none(), r.retire(row, 37).is_none()), (true, true));
    drop(pushed);
    assert_eq!((r.pages_used(), r.seq_slots().1, r.orders(None)[0].acts.len()), (0, 0, 0));
    // The batch runs over the group the positions are dealt over, or not at all.
    let p = Protocol::check(&verify(dealt(4)).unwrap()).unwrap();
    let pools = (0..4).map(|_| Pool::new(&dealt(4), 128, 10).unwrap()).collect();
    assert!(Ranks::new(p, Groups::new(4, 2).unwrap(), pools).is_err());
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
            .map(|(i, (row, &l))| Cell { row, ids: vec![100 + i as i64], pos: l - 1, sampling: Sampling::GREEDY })
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
                let it: BTreeMap<&str, &[i64]> =
                    items[g * w + m].rows.iter().map(|(n, v)| (n.as_str(), v.ints().unwrap())).collect();
                assert_eq!(it["token_ids"], tokens, "w {w}, member {m}");
                let held: Vec<i32> = mine.iter().flat_map(|&i| rows[i].parts()[m].page_ids().to_vec()).collect();
                let mut pads = Vec::new();
                for (k, slot) in it["slot_mapping"].iter().enumerate() {
                    let Some(&i) = mine.get(k) else {
                        pads.push(*slot);
                        continue;
                    };
                    let (pos, lease) = (lens[i] - 1, &rows[i].parts()[m]);
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
        let mut bufs = vec![std::collections::BTreeMap::<String, Values>::new(); w * groups];
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
            let cells: Vec<Cell> = rows
                .iter()
                .map(|(row, pos)| Cell { row, ids: vec![1], pos: *pos, sampling: Sampling::GREEDY })
                .collect();
            let mut st = r.stage(&cells, 1, |k| k.next_power_of_two()).unwrap();
            let f = st.forward(r.protocol(), 1).unwrap();
            let items = st.items(&f);
            let want: Vec<Vec<(String, Values)>> = items
                .iter()
                .map(|i| tables.iter().filter_map(|n| Some((n.clone(), i.rows.get(n)?.clone()))).collect())
                .collect();
            drop(cells);
            for (q, o) in r.orders(Some(items)).iter().enumerate() {
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

#[test]
fn a_request_draws_as_it_asks_or_takes_the_most_likely_token() {
    assert_eq!(sampling(&SamplingParams::default()), Sampling::GREEDY);
    let asked = SamplingParams { temperature: 0.7, top_k: 40, top_p: 0.9, min_p: 0.1, seed: Some(3) };
    assert_eq!(sampling(&asked), Sampling { temperature: 0.7, top_k: 40, top_p: 0.9, min_p: 0.1, seed: 3 });
    // One survivor is the most likely token, whatever the temperature.
    assert_eq!(sampling(&SamplingParams { top_k: 1, ..asked }), Sampling::GREEDY);
    // Without a seed, two requests draw apart.
    let unseeded = SamplingParams { seed: None, ..asked };
    let (x, y) = (sampling(&unseeded), sampling(&unseeded));
    assert_eq!((x.temperature, x.top_k, x.seed == y.seed), (0.7, 40, false));
}

#[test]
fn a_cell_is_staged_with_its_own_draw_and_a_padding_row_with_the_most_likely_token() {
    let mut r = ranks_of(sampled(), [4; 4]);
    let a = r.lease(20, |q| Some(q.index())).unwrap();
    let b = r.lease(20, |q| Some(if q.index() == 0 { 9 } else { q.index() })).unwrap();
    let hot = Sampling { temperature: 0.8, top_k: 50, top_p: 0.95, min_p: 0.05, seed: 7 };
    let cells = [
        Cell { row: &a, ids: vec![7], pos: 3, sampling: hot },
        Cell { row: &b, ids: vec![8], pos: 5, sampling: Sampling::GREEDY },
    ];
    // A bucket of two: each owner's row and a padding row on every rank.
    let mut st = r.stage(&cells, 1, |_| 2).unwrap();
    let f = st.forward(r.protocol(), 1).unwrap();
    let items = st.items(&f);
    let column = |name: &str| -> Vec<Values> { items.iter().map(|i| i.rows[name].clone()).collect() };
    let floats = |own: f32, rest: f32| {
        vec![
            Values::Floats(vec![own, rest]),
            Values::Floats(vec![rest; 2]),
            Values::Floats(vec![rest; 2]),
            Values::Floats(vec![rest; 2]),
        ]
    };
    let ints = |own: i64| {
        vec![Values::Ints(vec![own, 0]), Values::Ints(vec![0; 2]), Values::Ints(vec![0; 2]), Values::Ints(vec![0; 2])]
    };
    assert_eq!(column("temperature"), floats(0.8, 0.0));
    assert_eq!(column("top_p"), floats(0.95, 1.0));
    assert_eq!(column("min_p"), floats(0.05, 0.0));
    assert_eq!(column("top_k"), ints(50));
    assert_eq!(column("seed"), ints(7));
}
