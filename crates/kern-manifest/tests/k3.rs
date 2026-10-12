//! Kimi-K3 lives in kern-k3-pp-prefill; these are a few of its layers
//! (fixtures/k3/trim.py) so a change here that breaks its manifests fails
//! here first. The first stage of its pipeline hands back hidden states,
//! the last one tokens; both are one chunk of one sequence, as long as fed,
//! sized by the context it attends.

use kern_manifest::protocol::{Axis, Bound, Forward, LineTable, PageTable, Rows};
use kern_manifest::types::Fill;
use kern_manifest::{Protocol, Verified};

fn stage(json: &str) -> Verified {
    Verified::from_json(json).unwrap_or_else(|e| panic!("{e}"))
}

fn bound(var: &str, max: u64) -> Bound {
    Bound { var: var.into(), max }
}

fn fills(p: &Protocol) -> Vec<(&str, Fill, Axis)> {
    p.fills.iter().map(|f| (f.name.as_str(), f.fill, f.axis)).collect()
}

fn chunk(emits: Option<usize>) -> Vec<Forward> {
    vec![Forward { name: "prefill".into(), groups: 1, rows: Rows::Var, emits, count: None }]
}

#[test]
fn first_stage_feeds_a_chunk_and_hands_back_no_token() {
    let m = stage(include_str!("fixtures/k3/head.json"));
    let refused = Protocol::check(&m).unwrap_err();
    assert_eq!(refused.0, ["no program with a `batch` writes a `tokens` output: no call hands a token back"]);
    let p = Protocol::check_unsampled(&m).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        (&p.rows, &p.groups, &p.ranks, &p.context),
        (&bound("tokens", 8192), &bound("seqs", 1), &Some(bound("rows", 8192)), &Some(bound("ctx", 16384)))
    );
    assert_eq!(p.forwards, chunk(None));
    assert_eq!(
        fills(&p),
        [
            ("seq_lens", Fill::SeqLen, Axis::Groups),
            ("slot_mapping", Fill::Slot, Axis::Rows),
            ("token_ids", Fill::Token, Axis::Ranks)
        ]
    );
    assert_eq!(p.page_tables, [PageTable { name: "block_table".into(), width: 256 }]);
    assert_eq!(p.line_tables, [LineTable { name: "kda.line_index".into(), lines: 18, width: 1, axis: Axis::Ranks }]);
    assert_eq!((p.once, p.derive, m.group_size("ep")), (vec!["load".into()], vec!["derive".into()], Some(1)));
}

#[test]
fn last_stage_takes_hidden_states_and_hands_back_a_token() {
    let m = stage(include_str!("fixtures/k3/tail.json"));
    let p = Protocol::check(&m).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(p.forwards, chunk(p.fills.iter().position(|f| f.name == "next_token")));
    assert_eq!(
        fills(&p),
        [
            ("next_token", Fill::Tokens, Axis::Groups),
            ("seq_lens", Fill::SeqLen, Axis::Groups),
            ("slot_mapping", Fill::Slot, Axis::Rows)
        ]
    );
    assert_eq!(p.context, Some(bound("ctx", 16384)));
    assert_eq!(p.line_tables, [LineTable { name: "kda.line_index".into(), lines: 17, width: 1, axis: Axis::Ranks }]);
}
