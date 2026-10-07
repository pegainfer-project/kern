//! A cut changes transport, never arithmetic: run symbolically, the stages
//! of any cut that verifies compute what the whole manifest computes.

use std::collections::BTreeMap;

use kern_manifest::cut::{cut, state_use};
use kern_manifest::types::{Arg, Dir, Manifest, Module, ParamType};
use kern_manifest::{verify, Protocol, Verified};
use serde_json::{json, Value};

const QWEN3: &str = include_str!("../../../examples/qwen3-4b.json");

fn pp() -> Module {
    Module { source: "pp_mailbox.cubin".into(), sha256: "ab".repeat(32) }
}

struct Rand(u64);

impl Rand {
    fn next(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 33) as usize % n
    }
}

fn hash(parts: &[u64]) -> u64 {
    parts
        .iter()
        .flat_map(|p| p.to_le_bytes())
        .fold(0xcbf2_9ce4_8422_2325, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3))
}

fn name_hash(s: &str) -> u64 {
    hash(&s.bytes().map(u64::from).collect::<Vec<_>>())
}

/// A random straight-line program over four workspaces, an input, a carry
/// the once program derives and one paged state in per-layer regions, its
/// calls reading only what is written. Ops are opaque functions of what
/// they read.
fn random_manifest(r: &mut Rand) -> Manifest {
    let ops = json!({
        "f1": ["in buffer<bf16>", "out buffer<bf16>"],
        "f2": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>"],
        "g": ["inout buffer<bf16>"],
        "sw": ["in buffer<bf16>", "inout state"],
        "sr": ["in state", "in buffer<bf16>", "out buffer<bf16>"],
        "z": ["out buffer<bf16>"]
    });
    let ws = ["w0", "w1", "w2", "w3"];
    let mut readable = vec!["x", "c"];
    let mut calls: Vec<Value> = vec![json!({"op": "f1", "args": [{"buf": "x"}, {"buf": ws[r.next(4)]}]})];
    let pick = |r: &mut Rand, from: &[&'static str]| from[r.next(from.len())];
    readable.push(match &calls[0]["args"][1]["buf"] {
        Value::String(s) => ws.into_iter().find(|w| w == s).unwrap(),
        _ => unreachable!(),
    });
    for i in 1..4 + r.next(9) {
        // A layer is three calls and owns one region of the state.
        let region = json!({"state": "s", "offset": 4 * (i / 3)});
        let out = ws[r.next(4)];
        let c = match r.next(5) {
            0 => json!({"op": "f1", "args": [{"buf": pick(r, &readable)}, {"buf": out}]}),
            1 => json!({"op": "f2", "args": [{"buf": pick(r, &readable)}, {"buf": pick(r, &readable)}, {"buf": out}]}),
            2 => {
                let b = pick(r, &readable[2..]);
                json!({"op": "g", "args": [{"buf": b}]})
            }
            3 => json!({"op": "sw", "args": [{"buf": pick(r, &readable)}, region]}),
            _ => json!({"op": "sr", "args": [region, {"buf": pick(r, &readable)}, {"buf": out}]}),
        };
        if c["op"] != "g" && c["op"] != "sw" && !readable.contains(&out) {
            readable.push(out);
        }
        calls.push(c);
    }
    calls.push(json!({"op": "f1", "args": [{"buf": pick(r, &readable)}, {"buf": "y"}]}));
    for (i, c) in calls.iter_mut().enumerate() {
        c["label"] = json!(format!("c{i}"));
    }
    let used: Vec<String> = calls.iter().map(|c| c["op"].as_str().unwrap().to_string()).chain(["z".into()]).collect();
    let buf = |kind: &str, shape: Value| json!({"dtype": "bf16", "shape": shape, "kind": kind});
    let mut buffers = json!({
        "x": buf("input", json!(["tokens", 8])), "y": buf("output", json!(["tokens", 8])),
        "c": buf("carry", json!([8])), "d": buf("carry", json!([8]))
    });
    let named = serde_json::to_string(&calls).unwrap();
    for w in ws.iter().filter(|w| named.contains(&format!("\"{w}\""))) {
        buffers[*w] = buf("workspace", json!(["tokens", 8]));
    }
    let ops: serde_json::Map<String, Value> = ops
        .as_object()
        .unwrap()
        .iter()
        .filter(|(k, _)| used.contains(k))
        .map(|(k, p)| (k.clone(), json!({"params": p, "impl": {"launches": [{"entry": "extern:x"}]}})))
        .collect();
    let uses_state = named.contains("\"state\"");
    Manifest::from_json(
        &json!({
            "schema_version": 5, "model": "t", "vars": {"tokens": {"max": 4}},
            "states": if uses_state { json!({"s": {"bytes_per_token": 16}}) } else { json!({}) },
            "buffers": buffers, "modules": {}, "ops": ops,
            "programs": {
                "init": {"once": true, "calls": [
                    {"label": "i0", "op": "z", "args": [{"buf": "c"}]},
                    {"label": "i1", "op": "z", "args": [{"buf": "d"}]}]},
                "p": {"batch": {"groups": 1, "rows": 1}, "calls": calls}
            }
        })
        .to_string(),
    )
    .unwrap()
}

/// What one stage (or the whole manifest) leaves behind: its buffers and
/// the state regions it wrote.
#[derive(Default)]
struct Machine {
    bufs: BTreeMap<String, u64>,
    states: BTreeMap<(String, u64), u64>,
}

/// The edge a pipeline buffer belongs to: `pp.<e>.box` / `pp.<e>.peer`.
fn edge(name: &str) -> u64 {
    name.split('.').nth(1).unwrap().parse().unwrap()
}

impl Machine {
    /// Run `program` of `m`: every op a hash of what it reads into what it
    /// writes; the pipeline ops move values through `mail`.
    fn run(&mut self, m: &Manifest, program: &str, mail: &mut BTreeMap<(u64, i64), u64>) {
        for c in &m.programs[program].calls {
            let label = c.label.clone().unwrap();
            let buf = |a: &Arg| match a {
                Arg::Buf { buf, .. } => buf.clone(),
                _ => unreachable!(),
            };
            let off = |a: &Arg| match a {
                Arg::I64 { i64: v } => *v,
                _ => unreachable!(),
            };
            if c.op.starts_with("pp.put.") {
                mail.insert((edge(&buf(&c.args[0])), off(&c.args[3])), self.bufs[&buf(&c.args[1])]);
                continue;
            }
            if c.op.starts_with("pp.take.") {
                self.bufs.insert(buf(&c.args[1]), mail[&(edge(&buf(&c.args[0])), off(&c.args[3]))]);
                continue;
            }
            if c.op.starts_with("pp.") {
                continue;
            }
            let params = &m.ops[&c.op].params;
            let mut read = vec![name_hash(&c.op), name_hash(&label)];
            for (a, p) in c.args.iter().zip(params) {
                match (a, p) {
                    (Arg::Buf { buf, .. }, ParamType::Buf { dir: Dir::In | Dir::InOut, .. }) => {
                        read.push(*self.bufs.get(buf).unwrap_or_else(|| panic!("{label} reads unset `{buf}`")))
                    }
                    (Arg::State { state, offset }, ParamType::State { dir: Dir::In | Dir::InOut }) => {
                        read.push(*self.states.get(&(state.clone(), *offset)).unwrap_or(&0))
                    }
                    _ => {}
                }
            }
            let h = hash(&read);
            for (j, (a, p)) in c.args.iter().zip(params).enumerate() {
                let v = hash(&[h, j as u64]);
                match (a, p) {
                    (Arg::Buf { buf, .. }, ParamType::Buf { dir: Dir::Out | Dir::InOut, .. }) => {
                        self.bufs.insert(buf.clone(), v);
                    }
                    (Arg::State { state, offset }, ParamType::State { dir: Dir::Out | Dir::InOut }) => {
                        self.states.insert((state.clone(), *offset), v);
                    }
                    _ => {}
                }
            }
        }
    }

    fn start(m: &Manifest) -> Machine {
        let mut me = Machine::default();
        me.bufs.insert("x".into(), name_hash("x"));
        if m.programs.contains_key("init") {
            me.run(m, "init", &mut BTreeMap::new());
        }
        me
    }
}

#[test]
fn every_cut_that_verifies_computes_the_whole() {
    let (mut cuts, mut refused) = (0, 0);
    for seed in 1..=400u64 {
        let mut r = Rand(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let m = verify(random_manifest(&mut r)).unwrap_or_else(|e| panic!("seed {seed}: {e}"));
        let n = m.programs["p"].calls.len();
        let a = 1 + r.next(n - 1);
        let at = if a + 1 < n && r.next(2) == 0 { vec![a, a + 1 + r.next(n - a - 1)] } else { vec![a] };

        let mut whole = Machine::start(&m);
        whole.run(&m, "p", &mut BTreeMap::new());

        let stages = match cut(&m, &at, &pp()) {
            Ok(s) => s,
            Err(e) => {
                assert!(e.0.iter().all(|d| d.contains("touched by stages")), "seed {seed} at {at:?}: {e}");
                refused += 1;
                continue;
            }
        };
        cuts += 1;
        assert_eq!(stages.len(), at.len() + 1);
        let mut mail = BTreeMap::new();
        let mut states = BTreeMap::new();
        let mut y = None;
        for (s, st) in stages.iter().enumerate() {
            assert_eq!(st.cut.as_ref().map(|c| (c.stage, c.stages)), Some((s as u64, stages.len() as u64)));
            let mut me = Machine::start(st);
            me.run(st, "p", &mut mail);
            for (k, v) in me.states {
                assert!(states.insert(k, v).is_none(), "seed {seed}: a state region written by two stages");
            }
            y = me.bufs.get("y").copied();
        }
        assert_eq!((y, states), (whole.bufs.get("y").copied(), whole.states), "seed {seed} at {at:?}");
    }
    assert!(cuts > 200 && refused > 0, "{cuts} cuts, {refused} refused");
}

fn qwen3() -> Verified {
    verify(Manifest::from_json(QWEN3).unwrap()).unwrap()
}

/// The buffers stage `s`'s program puts into its outgoing box.
fn carried(st: &Manifest, program: &str) -> Vec<String> {
    st.programs[program]
        .calls
        .iter()
        .filter(|c| c.op.starts_with("pp.put."))
        .map(|c| match &c.args[1] {
            Arg::Buf { buf, .. } => buf.clone(),
            _ => unreachable!(),
        })
        .collect()
}

#[test]
fn qwen3_cuts_at_a_layer_into_stages_that_each_serve() {
    // l18.qkv_proj: embed, l0 norm, 18 layers of 12 calls.
    let at = 2 + 18 * 12;
    assert_eq!(qwen3().programs["prefill"].calls[at].label.as_deref(), Some("l18.qkv_proj"));
    let stages = cut(&qwen3(), &[at], &pp()).unwrap_or_else(|e| panic!("{e}"));
    let (head, tail) = (&stages[0], &stages[1]);
    // The residual stream and the normed input are all that crosses, in
    // every program.
    for p in ["prefill", "decode", "decode_batch"] {
        assert_eq!(carried(head, p), ["residual", "x"]);
    }
    // Each stage holds its own weights and its own calls of the KV.
    assert!(
        head.buffers.contains_key("model.embed_tokens.weight")
            && !tail.buffers.contains_key("model.embed_tokens.weight")
    );
    assert!(tail.buffers.contains_key("lm_head.weight") && !head.buffers.contains_key("lm_head.weight"));
    assert!(!head.buffers.contains_key("model.layers.18.self_attn.qkv_proj.weight"));
    // A KV state per layer: each stage keeps its 18 and uses them whole,
    // and its tables index one it kept.
    for (st, first) in [(head, "kv.0"), (tail, "kv.18")] {
        let used = state_use(&qwen3(), st);
        assert_eq!((used.len(), used.iter().all(|(_, a, b)| a == b)), (18, true));
        let table = |b: &str| st.buffers[b].domain.as_ref().and_then(|d| d.index_into.clone());
        assert_eq!((table("block_table"), table("slot_mapping")), (Some(first.into()), Some(first.into())));
    }
    // The head stages tokens and hands nothing back; the tail is fed
    // activations and samples.
    let ph = Protocol::check_unsampled(head).unwrap_or_else(|e| panic!("{e}"));
    assert!(ph.forwards.iter().all(|f| f.emits.is_none()));
    let pt = Protocol::check(tail).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(pt.forward(1, kern_manifest::protocol::Rows::Const(1)).map(|f| f.emits.is_some()), Some(true));
    assert_eq!(head.cut.as_ref().map(|c| &c.id), tail.cut.as_ref().map(|c| &c.id));
    // The same cut is the same bytes.
    let again = cut(&qwen3(), &[at], &pp()).unwrap();
    assert_eq!(again[1].to_json(), tail.to_json());

    let three = cut(&qwen3(), &[2 + 12 * 12, 2 + 24 * 12], &pp()).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(three.len(), 3);
    Protocol::check_unsampled(&three[1]).unwrap_or_else(|e| panic!("{e}"));
    assert_ne!(three[0].cut.as_ref().map(|c| &c.id), head.cut.as_ref().map(|c| &c.id));
}

#[test]
fn a_cut_says_why_it_cannot_be_made() {
    let n = qwen3().programs["prefill"].calls.len();
    for at in [vec![], vec![0], vec![5, 5], vec![n]] {
        assert!(cut(&qwen3(), &at, &pp()).is_err(), "{at:?}");
    }
}
