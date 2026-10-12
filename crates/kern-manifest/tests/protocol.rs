//! The serving protocol as a driver reads it: what a manifest's `fill`,
//! `batch` and `once` declarations project to, and what they refuse.

use std::collections::BTreeMap;

use kern_manifest::protocol::{Axis, Bound, Heads, LineTable, PageTable, Positions, ProtocolErrors, Rows, Shards};
use kern_manifest::types::{Batch, Buffer, Dim, Fill, Op};
use kern_manifest::{verify, Manifest, Protocol};
/// A single-rank contract: a prefill chunk that only fills state, a
/// bs=1 decode and a batched one, a page table and a line table. The
/// fixture verifies, so every buffer and state is consumed by a call.
fn plain() -> Manifest {
    Manifest::from_json(include_str!("fixtures/plain.json")).unwrap()
}

/// The same plus a speculative round: 4 rows per sequence, an anchor,
/// per-row tokens and a count.
fn speculative() -> Manifest {
    let mut m = plain();
    let buf = |s: &str| serde_json::from_str::<Buffer>(s).unwrap();
    m.buffers
        .insert("anchor_token".into(), buf(r#"{"kind": "input", "dtype": "i64", "shape": ["seqs"], "fill": "token"}"#));
    m.buffers.insert(
        "verify_tokens".into(),
        buf(r#"{"kind": "output", "dtype": "i64", "shape": ["seqs", 4], "fill": "tokens"}"#),
    );
    m.buffers.insert("nacc".into(), buf(r#"{"kind": "output", "dtype": "i32", "shape": ["seqs"], "fill": "count"}"#));
    m.ops.insert("read".into(), op(&["in buffer<i64>"]));
    m.ops.insert("accept".into(), op(&["out buffer<i64>", "out buffer<i32>"]));
    m.programs.insert(
        "round".into(),
        serde_json::from_str(
            r#"{"batch": {"groups": 2, "rows": 4}, "calls": [
            {"op": "read", "args": [{"buf": "anchor_token"}]},
            {"op": "accept", "args": [{"buf": "verify_tokens"}, {"buf": "nacc"}]}]}"#,
        )
        .unwrap(),
    );
    m
}

/// A manifest whose batch spans its ranks: the tokens, the line table
/// and the output over every rank's rows, the rest this rank's own.
fn ranks() -> Manifest {
    Manifest::from_json(include_str!("fixtures/ranks.json")).unwrap()
}

/// Decode context parallelism: a `tp` group of 4 deals the paged states
/// by position and splits the recurrent one by heads.
fn dealt() -> Manifest {
    Manifest::from_json(include_str!("fixtures/dealt.json")).unwrap()
}

/// An op with an extern launch, for a fixture that only needs the ABI.
fn op(params: &[&str]) -> Op {
    let params: Vec<String> = params.iter().map(|p| format!("\"{p}\"")).collect();
    serde_json::from_str(&format!(
        r#"{{"params": [{}], "impl": {{"launches": [{{"entry": "extern:x"}}]}}}}"#,
        params.join(", ")
    ))
    .unwrap()
}

/// The protocol of a fixture, which has to verify first: a fixture that
/// does not is a broken fixture, not a protocol finding.
fn protocol(m: &Manifest) -> Result<Protocol, ProtocolErrors> {
    Protocol::check(&verify(m.clone()).unwrap_or_else(|e| panic!("fixture does not verify: {e}")))
}

fn rejects(m: &Manifest, what: &str) {
    let Err(e) = protocol(m) else { panic!("accepted, expected `{what}`") };
    assert!(e.iter().any(|x| x.contains(what)), "no `{what}` in {e:#?}");
}

#[test]
fn plain_contract() {
    let p = protocol(&plain()).unwrap();
    assert_eq!((p.rows.var.as_str(), p.rows.max, p.groups.var.as_str(), p.groups.max), ("tokens", 8, "seqs", 4));
    assert_eq!(p.ranks, None);
    assert_eq!(
        (
            p.token_rows().unwrap().name.as_str(),
            p.any(Fill::Slot).unwrap().name.as_str(),
            p.any(Fill::SeqLen).unwrap().name.as_str()
        ),
        ("token_ids", "slot_mapping", "seq_lens")
    );
    assert_eq!(p.any(Fill::CuSeqlens).map(|f| f.axis), Some(Axis::Fixed(5)));
    assert_eq!(p.page_tables, vec![PageTable { name: "block_table".into(), width: 3 }]);
    assert_eq!(p.line_tables, vec![LineTable { name: "line_index".into(), lines: 3, width: 1, axis: Axis::Groups }]);
    let names: Vec<(&str, u64, Rows, bool)> =
        p.forwards.iter().map(|f| (f.name.as_str(), f.groups, f.rows, f.emits.is_some())).collect();
    assert_eq!(
        names,
        [
            ("decode", 1, Rows::Const(1), true),
            ("decode_batch", 4, Rows::Const(1), true),
            ("prefill", 1, Rows::Var, false)
        ]
    );
    // The tightest bound wins; a prefill chunk is the var-rows call.
    assert_eq!(p.forward(1, Rows::Const(1)).map(|f| f.name.as_str()), Some("decode"));
    assert_eq!(p.forward(3, Rows::Const(1)).map(|f| f.name.as_str()), Some("decode_batch"));
    assert_eq!(p.forward(5, Rows::Const(1)), None);
    assert_eq!(p.chunk().map(|f| f.name.as_str()), Some("prefill"));
    assert_eq!((p.row_shapes(), p.max_groups(Rows::Const(1))), (vec![1], 4));
    assert_eq!(p.vars(3, 6, 6, 6), BTreeMap::from([("tokens".into(), 6), ("seqs".into(), 3)]));
}

#[test]
fn speculative_contract() {
    let p = protocol(&speculative()).unwrap();
    let round = p.forwards.iter().find(|f| f.name == "round").unwrap();
    assert_eq!((round.groups, round.rows), (2, Rows::Const(4)));
    assert_eq!(round.emits.map(|i| (p.fills[i].name.as_str(), p.fills[i].width)), Some(("verify_tokens", 4)));
    assert_eq!(round.count.map(|i| p.fills[i].name.as_str()), Some("nacc"));
    assert_eq!(p.filled(Fill::Token, Axis::Groups).map(|f| f.name.as_str()), Some("anchor_token"));
    assert_eq!((p.row_shapes(), p.max_groups(Rows::Const(4))), (vec![1, 4], 2));
    // The plain decode in the same manifest hands back one per sequence.
    let decode = p.forwards.iter().find(|f| f.name == "decode").unwrap();
    assert_eq!((decode.emits.map(|i| p.fills[i].width), decode.count), (Some(1), None));
}

/// The plain contract plus a program sized by its context: the prefill
/// chunk's `batch` names `ctx`, a var nothing else reads.
fn contextual() -> Manifest {
    let mut m = plain();
    m.vars.insert("ctx".into(), serde_json::from_str(r#"{"max": 64}"#).unwrap());
    m.programs.get_mut("prefill").unwrap().batch.as_mut().unwrap().context = Some("ctx".into());
    m
}

#[test]
fn context_contract() {
    let p = protocol(&contextual()).unwrap();
    assert_eq!(p.context, Some(Bound { var: "ctx".into(), max: 64 }));
    assert_eq!(p.vars(1, 5, 5, 37).get("ctx"), Some(&37));
    let plain = protocol(&plain()).unwrap();
    assert_eq!((plain.context.clone(), plain.vars(1, 5, 5, 37).get("ctx")), (None, None));
    let mut m = contextual();
    m.programs.get_mut("prefill").unwrap().batch.as_mut().unwrap().context = Some("tokens".into());
    m.vars.remove("ctx");
    rejects(&m, "batch.context is `tokens`, which sizes the call itself");
    let mut m = contextual();
    m.vars.insert("other".into(), serde_json::from_str(r#"{"max": 2}"#).unwrap());
    m.programs.get_mut("decode").unwrap().batch.as_mut().unwrap().context = Some("other".into());
    rejects(&m, "one var bounds every context");
}

#[test]
fn ranks_contract() {
    let p = protocol(&ranks()).unwrap();
    assert_eq!(p.ranks, Some(Bound { var: "rows".into(), max: 32 }));
    assert_eq!(p.token_rows().unwrap().axis, Axis::Ranks);
    assert_eq!(p.line_tables[0].axis, Axis::Ranks);
    assert_eq!(p.any(Fill::Error).map(|f| f.name.as_str()), Some("tp_err"));
    assert_eq!(p.once, vec!["tp_init".to_string()]);
    assert_eq!(p.vars(2, 2, 8, 9), BTreeMap::from([("tokens".into(), 2), ("seqs".into(), 2), ("rows".into(), 8)]));
    assert_eq!(
        (p.shards.positions.is_none(), p.shards.heads.get("kda").map(|h| (h.members, h.segments.clone()))),
        (true, Some((4, vec![(3, 32)])))
    );
}

#[test]
fn a_group_dealing_positions_is_a_replicated_batch() {
    let p = protocol(&dealt()).unwrap();
    assert_eq!(
        p.shards,
        Shards {
            positions: Some(Positions { group: "tp".into(), members: 4 }),
            heads: BTreeMap::from([(
                "kda".into(),
                Heads { group: "tp".into(), members: 4, segments: vec![(1, 64), (2, 16)] }
            )]),
        }
    );
    assert_eq!((p.ranks.is_none(), p.token_rows().map(|f| f.axis)), (true, Some(Axis::Rows)));
    assert_eq!((p.line_tables[0].axis, p.page_tables[0].width), (Axis::Groups, 3));
    assert_eq!(protocol(&plain()).unwrap().shards, Shards::default());
}

#[test]
fn encodes_by_dtype() {
    let p = protocol(&plain()).unwrap();
    let (slots, seq_lens) = (p.any(Fill::Slot).unwrap(), p.any(Fill::SeqLen).unwrap());
    assert_eq!(seq_lens.encode(&[3, -1]), vec![3, 0, 0, 0, 255, 255, 255, 255]);
    assert_eq!(slots.encode(&[2]), 2i64.to_le_bytes());
    assert_eq!(seq_lens.decode(&[7, 0, 0, 0]), vec![7]);
}

/// The axes are the vars', not the roles': a manifest may read no slot and
/// no length (a pipeline stage past the first) and still have both axes.
#[test]
fn axes_are_declared_on_the_vars() {
    let mut m = plain();
    m.buffers.get_mut("seq_lens").unwrap().fill = None;
    m.buffers.get_mut("token_ids").unwrap().fill = None;
    let p = protocol(&m).unwrap();
    assert_eq!((p.rows.var.as_str(), p.groups.var.as_str(), p.token_rows()), ("tokens", "seqs", None));
    let mut m = plain();
    m.vars.get_mut("tokens").unwrap().axis = None;
    m.vars.get_mut("seqs").unwrap().axis = None;
    let Err(e) = protocol(&m) else { panic!() };
    assert_eq!(e.len(), 2);
    rejects(&m, "no var has `axis: rows`");
    rejects(&m, "no var has `axis: groups`");
    let mut m = plain();
    m.vars.insert("other".into(), serde_json::from_str(r#"{"max": 8}"#).unwrap());
    m.buffers.get_mut("positions").unwrap().shape = vec![Dim::Var("other".into())];
    rejects(&m, "fill `position` `positions` is over var `other`, which counts no axis");
}

#[test]
fn shape_rules() {
    let mut m = plain();
    m.buffers.get_mut("cu_seqlens_q").unwrap().shape = vec![Dim::Const(4)];
    rejects(&m, "expected [n] with n >= groups + 1");
    let mut m = plain();
    m.buffers.get_mut("block_table").unwrap().shape = vec![Dim::Var("seqs".into())];
    rejects(&m, "page table `block_table` is shaped");
    let mut m = plain();
    m.buffers.get_mut("line_index").unwrap().shape = vec![Dim::Const(3), Dim::Var("tokens".into())];
    rejects(&m, "column per row");
    let mut m = plain();
    m.buffers.get_mut("seq_lens").unwrap().shape = vec![Dim::Var("tokens".into())];
    rejects(&m, "expected [groups]");
}

#[test]
fn batch_rules() {
    let mut m = plain();
    m.programs.get_mut("decode_batch").unwrap().batch = Some(Batch { groups: 5, rows: Dim::Const(1), context: None });
    rejects(&m, "5 groups exceed the 4");
    let mut m = plain();
    m.programs.get_mut("decode_batch").unwrap().batch = Some(Batch { groups: 4, rows: Dim::Const(3), context: None });
    rejects(&m, "4 sequences of 3 rows exceed the 8");
    let mut m = plain();
    m.programs.get_mut("prefill").unwrap().batch =
        Some(Batch { groups: 2, rows: Dim::Var("tokens".into()), context: None });
    let p = protocol(&m).unwrap();
    let prefill = p.forwards.iter().find(|f| f.name == "prefill").unwrap();
    assert_eq!((prefill.groups, prefill.rows), (2, Rows::Var));
    m.buffers.get_mut("cu_seqlens_q").unwrap().fill = None;
    rejects(&m, "needs a `cu_seqlens` fill to split them");
    let mut m = plain();
    m.programs.get_mut("prefill").unwrap().batch =
        Some(Batch { groups: 1, rows: Dim::Var("seqs".into()), context: None });
    rejects(&m, "the rows of a call go in `tokens`");
    let mut m = plain();
    m.programs.get_mut("decode_batch").unwrap().batch = Some(Batch { groups: 1, rows: Dim::Const(1), context: None });
    rejects(&m, "accept the same call shape");
    let mut m = plain();
    for p in m.programs.values_mut() {
        p.batch = None;
    }
    rejects(&m, "no program declares a `batch`");
    let mut m = plain();
    m.buffers.remove("next_token");
    m.ops.remove("head");
    let write = m.programs["prefill"].calls[0].clone();
    for p in ["decode", "decode_batch"] {
        m.programs.get_mut(p).unwrap().calls = vec![write.clone()];
    }
    rejects(&m, "no call hands a token back");
}

#[test]
fn what_a_forward_hands_back_is_dataflow() {
    // A round handing back 4 per sequence must be a 4-row call.
    let mut m = speculative();
    m.programs.get_mut("round").unwrap().batch = Some(Batch { groups: 2, rows: Dim::Const(3), context: None });
    rejects(&m, "hands back `verify_tokens` of 4 per sequence, but a call has 3 rows");
    // A count needs several tokens per sequence to count.
    let mut m = speculative();
    m.programs.get_mut("decode").unwrap().calls.push(
        serde_json::from_str(r#"{"op": "accept", "args": [{"buf": "verify_tokens"}, {"buf": "nacc"}]}"#).unwrap(),
    );
    rejects(&m, "writes 2 `tokens` outputs");
    let mut m = speculative();
    m.programs.get_mut("decode").unwrap().calls =
        vec![serde_json::from_str(r#"{"op": "accept", "args": [{"buf": "next_token"}, {"buf": "nacc"}]}"#).unwrap()];
    rejects(&m, "no `tokens` output of several per sequence to count");
    // Two counts written by one call are the manifest's error, not a panic.
    let mut m = speculative();
    m.buffers.insert(
        "nacc_too".into(),
        serde_json::from_str(r#"{"kind": "output", "dtype": "i32", "shape": ["seqs"], "fill": "count"}"#).unwrap(),
    );
    m.ops.insert("count".into(), op(&["out buffer<i32>"]));
    m.programs
        .get_mut("round")
        .unwrap()
        .calls
        .push(serde_json::from_str(r#"{"op": "count", "args": [{"buf": "nacc_too"}]}"#).unwrap());
    rejects(&m, "fill `count` is on more than one buffer");
}
