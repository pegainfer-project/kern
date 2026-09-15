//! The fake side: a host interpreter over a tiny manifest, and the fixture
//! family it runs. The model is two layers of `scale` (elementwise, the op
//! under swap) and `mix` (a recurrence over a per-token state: not
//! idempotent, reads the previous token's slot), an `embed` in front and a
//! `head` behind that writes bf16 logits and the argmax token. Every
//! behaviour is keyed by the launch entry the manifest names, so a B
//! variant is a manifest edit and nothing else — the way a real swap is.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::ops::Range;

use anyhow::{anyhow, bail, Result};
use kern_manifest::protocol::Axis;
use kern_manifest::types::{Arg, DType, Dim, Fill, Provision};
use kern_manifest::values;
use kern_manifest::{verify, Manifest, Protocol, Verified};
use kern_test::{Options, Side, Vars};

pub const D: usize = 4;
pub const VOCAB: usize = 16;
pub const LAYERS: usize = 2;
pub const CAPACITY: u64 = 16;

/// One manifest of the family: which entry each op launches, and the
/// structural variations.
#[derive(Clone, Debug)]
pub struct Fixture {
    pub scale: &'static str,
    pub mix: &'static str,
    pub head: &'static str,
    /// At this layer, the `scale` call goes to a second op `scale_alt`
    /// launching this entry: a change at one layer only.
    pub alt: Option<(usize, &'static str)>,
    /// Name of the logits buffer; anything not starting with `logits` is
    /// no oracle.
    pub logits: &'static str,
    /// An extra program without a `batch` (a harness-only layer): changed
    /// but undriven.
    pub probe: bool,
}

impl Default for Fixture {
    fn default() -> Self {
        Fixture { scale: "scale", mix: "mix", head: "head", alt: None, logits: "logits", probe: false }
    }
}

impl Fixture {
    pub fn scale(self, e: &'static str) -> Self {
        Fixture { scale: e, ..self }
    }
    pub fn mix(self, e: &'static str) -> Self {
        Fixture { mix: e, ..self }
    }
    pub fn head(self, e: &'static str) -> Self {
        Fixture { head: e, ..self }
    }
    pub fn alt(self, layer: usize, e: &'static str) -> Self {
        Fixture { alt: Some((layer, e)), ..self }
    }
    pub fn logits(self, name: &'static str) -> Self {
        Fixture { logits: name, ..self }
    }
    pub fn probe(self) -> Self {
        Fixture { probe: true, ..self }
    }

    pub fn manifest(&self) -> Verified {
        let op = |params: &[&str], entry: &str| serde_json::json!({"params": params, "impl": {"launches": [{"entry": format!("extern:{entry}")}]}});
        let mut ops = serde_json::Map::new();
        ops.insert("embed".into(), op(&["in buffer<i64>", "out buffer<f32>"], "embed"));
        ops.insert("scale".into(), op(&["in buffer<f32>", "out buffer<f32>"], self.scale));
        if let Some((_, e)) = self.alt {
            ops.insert("scale_alt".into(), op(&["in buffer<f32>", "out buffer<f32>"], e));
        }
        ops.insert(
            "mix".into(),
            op(
                &["in buffer<f32>", "in buffer<i64>", "in buffer<i32>", "inout state", "i32", "out buffer<f32>"],
                self.mix,
            ),
        );
        ops.insert("head".into(), op(&["in buffer<f32>", "out buffer<bf16>", "out buffer<i64>"], self.head));
        let mut calls = vec![serde_json::json!({"op": "embed", "args": [{"buf": "token_ids"}, {"buf": "hidden"}]})];
        for l in 0..LAYERS {
            let sc = match self.alt {
                Some((k, _)) if k == l => "scale_alt",
                _ => "scale",
            };
            calls.push(serde_json::json!({"op": sc, "args": [{"buf": "hidden"}, {"buf": "act"}]}));
            calls.push(serde_json::json!({"op": "mix", "args": [
                {"buf": "act"}, {"buf": "slot_mapping"}, {"buf": "seq_lens"}, {"state": "kv"}, {"i32": l},
                {"buf": "hidden"}]}));
        }
        calls.push(serde_json::json!({"op": "head", "args": [
            {"buf": "hidden"}, {"buf": self.logits}, {"buf": "next_token"}]}));
        let mut programs = serde_json::json!({
            "prefill": {"batch": {"groups": 1, "rows": "tokens"}, "calls": calls},
            "decode": {"batch": {"groups": 1, "rows": 1}, "calls": calls},
        });
        if self.probe {
            programs["probe"] = serde_json::json!({"calls": calls});
        }
        let m = serde_json::json!({
            "schema_version": 5, "model": "fake",
            "vars": {"tokens": {"max": 8}, "seqs": {"max": 1}},
            "states": {"kv": {"bytes_per_token": LAYERS * D * 4}},
            "buffers": {
                "token_ids": {"kind": "input", "dtype": "i64", "shape": ["tokens"], "fill": "token",
                              "domain": {"min": 0, "max": VOCAB - 1}},
                "slot_mapping": {"kind": "input", "dtype": "i64", "shape": ["tokens"], "fill": "slot",
                                 "domain": {"index_into": "kv"}},
                "seq_lens": {"kind": "input", "dtype": "i32", "shape": ["seqs"], "fill": "seq_len"},
                "hidden": {"kind": "workspace", "dtype": "f32", "shape": ["tokens", D]},
                "act": {"kind": "workspace", "dtype": "f32", "shape": ["tokens", D]},
                self.logits: {"kind": "output", "dtype": "bf16", "shape": ["tokens", VOCAB]},
                "next_token": {"kind": "output", "dtype": "i64", "shape": ["seqs"], "fill": "tokens",
                               "domain": {"min": 0, "max": VOCAB - 1}},
            },
            "modules": {},
            "ops": ops,
            "programs": programs,
        });
        let m: Manifest = serde_json::from_value(m).expect("fixture parses");
        verify(m).unwrap_or_else(|e| panic!("fixture does not verify: {e}"))
    }
}

/// The interpreter over one manifest of the family.
pub struct Fake {
    m: Verified,
    protocol: Protocol,
    bufs: BTreeMap<String, Vec<u8>>,
    states: BTreeMap<String, Vec<u8>>,
    pos: i64,
    /// How often `scale_noisy` has seen each input: it flips sign on every
    /// repeat, so no replay reproduces the original.
    seen: BTreeMap<Vec<u8>, u64>,
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
}

fn bytes_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn i64s(b: &[u8]) -> Vec<i64> {
    b.as_chunks::<8>().0.iter().map(|c| i64::from_le_bytes(*c)).collect()
}

impl Fake {
    pub fn new(m: Verified) -> Fake {
        let protocol = Protocol::check(&m).expect("fixture has a protocol");
        let max: Vars = m.vars.iter().map(|(k, v)| (k.clone(), v.max)).collect();
        let bufs = m.buffers.keys().map(|n| (n.clone(), vec![0u8; kern_test::diff::live_bytes(&m, n, &max)])).collect();
        let states =
            m.states.iter().map(|(n, s)| (n.clone(), vec![0u8; (s.bytes_per_token * CAPACITY) as usize])).collect();
        Fake { m, protocol, bufs, states, pos: 0, seen: BTreeMap::new() }
    }

    fn rows(&self, vars: &Vars) -> usize {
        vars[&self.protocol.rows.var] as usize
    }

    fn call(&mut self, op: &str, args: &[Arg], vars: &Vars) -> Result<()> {
        let t = self.rows(vars);
        let entry = self.m.ops[op].imp.launches[0].entry().to_string();
        let buf = |k: usize| match &args[k] {
            Arg::Buf { buf, .. } => buf.clone(),
            _ => panic!("expected a buffer arg"),
        };
        match entry.as_str() {
            "extern:embed" => {
                let ids = i64s(&self.bufs[&buf(0)][..t * 8]);
                let h: Vec<f32> = ids
                    .iter()
                    .flat_map(|&tok| (0..D).map(move |d| (((tok + 1) * (d as i64 + 1)) % 7 - 3) as f32))
                    .collect();
                self.bufs.get_mut(&buf(1)).unwrap()[..t * D * 4].copy_from_slice(&bytes_f32(&h));
            }
            e if e.starts_with("extern:scale") => {
                let raw = self.bufs[&buf(0)][..t * D * 4].to_vec();
                let x = f32s(&raw);
                // The noisy variant is ±3%, the sign flipping each time the
                // same input comes back: A's own band (6%) always contains
                // its distance to the exact op (3%).
                let seen = self.seen.entry(raw).or_insert(0);
                *seen += 1;
                let noisy = if *seen % 2 == 1 { 1.0 + 2f32.powi(-5) } else { 1.0 - 2f32.powi(-5) };
                let y: Vec<f32> = x
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| match e {
                        "extern:scale" | "extern:scale_same" => v * 0.5,
                        "extern:scale_negzero" if v == 0.0 => -0.0,
                        "extern:scale_negzero" => v * 0.5,
                        "extern:scale_round" => v * 0.5 * (1.0 + 2f32.powi(-9)),
                        "extern:scale_drift" => v * 0.5 * 1.05,
                        "extern:scale_wrong" => v * 0.5 + 1.0,
                        "extern:scale_nan" if i == 0 => f32::NAN,
                        "extern:scale_nan" => v * 0.5,
                        "extern:scale_dim3" if i % D == 3 => v * 0.5 + 1.0,
                        "extern:scale_dim3" => v * 0.5,
                        "extern:scale_noisy" => v * 0.5 * noisy,
                        "extern:scale_crash" => v * 0.5,
                        _ => panic!("no such scale: {e}"),
                    })
                    .collect();
                // A kernel with a baked-in assumption: the exact model only
                // ever produces multiples of 1/16, and the first input off
                // that grid (fuzz's jitter) kills it.
                if e == "extern:scale_crash" && x.iter().any(|v| (v * 16.0).fract() != 0.0) {
                    bail!("scale_crash: illegal memory access");
                }
                self.bufs.get_mut(&buf(1)).unwrap()[..t * D * 4].copy_from_slice(&bytes_f32(&y));
            }
            e if e.starts_with("extern:mix") => {
                let act = f32s(&self.bufs[&buf(0)][..t * D * 4]);
                let slots = i64s(&self.bufs[&buf(1)][..t * 8]);
                let Arg::State { state, .. } = &args[3] else { panic!("expected a state arg") };
                let Arg::I32 { i32: layer } = args[4] else { panic!("expected the layer") };
                let cell = |slot: i64| (slot as usize * LAYERS + layer as usize) * D * 4;
                let st = self.states.get_mut(state).unwrap();
                let mut h = vec![0f32; t * D];
                for (r, &slot) in slots.iter().enumerate() {
                    let c = cell(slot);
                    let mut cur = f32s(&st[c..c + D * 4]);
                    for d in 0..D {
                        cur[d] += act[r * D + d];
                    }
                    st[c..c + D * 4].copy_from_slice(&bytes_f32(&cur));
                    if e == "extern:mix_leak" && ((slot + 1) as u64) < CAPACITY {
                        let n = cell(slot + 1);
                        st[n..n + D * 4].copy_from_slice(&bytes_f32(&act[r * D..(r + 1) * D]));
                    }
                    let prev = if slot > 0 { f32s(&st[cell(slot - 1)..cell(slot - 1) + D * 4]) } else { vec![0.0; D] };
                    for d in 0..D {
                        h[r * D + d] = cur[d] + 0.5 * prev[d];
                    }
                }
                self.bufs.get_mut(&buf(5)).unwrap()[..t * D * 4].copy_from_slice(&bytes_f32(&h));
            }
            e if e.starts_with("extern:head") => {
                let h = f32s(&self.bufs[&buf(0)][..t * D * 4]);
                let w = |d: usize, v: usize| (((3 * d + 5 * v + d * v) % 17) % 5) as f32 - 2.0;
                let mut logits = vec![0f32; t * VOCAB];
                for r in 0..t {
                    for v in 0..VOCAB {
                        logits[r * VOCAB + v] = (0..3).map(|d| h[r * D + d] * w(d, v)).sum();
                    }
                    if e == "extern:head_swap" {
                        let row = &mut logits[r * VOCAB..(r + 1) * VOCAB];
                        let (top1, m1) =
                            row.iter().enumerate().fold((0, f32::MIN), |m, (i, &x)| if x > m.1 { (i, x) } else { m });
                        let (top2, m2) = row
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| *i != top1)
                            .fold((0, f32::MIN), |m, (i, &x)| if x > m.1 { (i, x) } else { m });
                        let margin = m1 - m2;
                        row[top1] -= 0.6 * margin;
                        row[top2] += 0.6 * margin;
                    }
                }
                let last = &logits[(t - 1) * VOCAB..t * VOCAB];
                let argmax =
                    last.iter().enumerate().fold((0, f32::MIN), |m, (i, &x)| if x > m.1 { (i, x) } else { m }).0;
                let tok = if e == "extern:head_bad" { 99 } else { argmax as i64 };
                let lg = values::from_f64(DType::Bf16, &logits.iter().map(|&x| x as f64).collect::<Vec<_>>());
                self.bufs.get_mut(&buf(1)).unwrap()[..lg.len()].copy_from_slice(&lg);
                self.bufs.get_mut(&buf(2)).unwrap()[..8].copy_from_slice(&tok.to_le_bytes());
            }
            e => bail!("no such op: {e}"),
        }
        Ok(())
    }
}

impl Side for Fake {
    type Buf = Vec<u8>;

    fn manifest(&self) -> &Verified {
        &self.m
    }
    fn provision(&self) -> Provision {
        Provision { tokens: CAPACITY, seq_slots: 0 }
    }
    fn page(&self) -> u64 {
        4
    }
    fn vocab(&self) -> u64 {
        VOCAB as u64
    }
    fn stage(&mut self, ids: &[i64]) -> Result<Vars> {
        let c = ids.len();
        let pos = self.pos;
        let vars = self.protocol.vars(1, c as u64, c as u64);
        let p = self.protocol.clone();
        let mut put = |f: &kern_manifest::protocol::Filled, v: &[i64]| {
            let b = f.encode(v);
            self.bufs.get_mut(&f.name).unwrap()[..b.len()].copy_from_slice(&b);
        };
        put(p.token_rows(), ids);
        put(p.slots(), &(pos..pos + c as i64).collect::<Vec<_>>());
        put(p.seq_lens(), &[pos + c as i64]);
        if let Some(f) = p.filled(Fill::Position, Axis::Rows) {
            put(f, &(pos..pos + c as i64).collect::<Vec<_>>());
        }
        Ok(vars)
    }
    fn reset(&mut self) {
        self.pos = 0;
    }
    fn advance(&mut self, n: u64) {
        self.pos += n as i64;
    }
    fn calls(&self, program: &str) -> Result<usize> {
        Ok(self.m.programs[program].calls.len())
    }
    fn run(&mut self, program: &str, vars: &Vars, calls: Range<usize>) -> Result<()> {
        let cs = self.m.programs[program].calls[calls].to_vec();
        for c in cs {
            self.call(&c.op, &c.args, vars)?;
        }
        Ok(())
    }
    fn read(&self, buffer: &str, bytes: usize) -> Result<Vec<u8>> {
        Ok(self.bufs[buffer][..bytes].to_vec())
    }
    fn write(&mut self, buffer: &str, bytes: &[u8]) -> Result<()> {
        self.bufs.get_mut(buffer).ok_or_else(|| anyhow!("no buffer `{buffer}`"))?[..bytes.len()].copy_from_slice(bytes);
        Ok(())
    }
    fn alloc(&self, bytes: usize) -> Result<Vec<u8>> {
        Ok(vec![0; bytes])
    }
    fn save(&self, buffer: &str, bytes: usize, into: &mut Vec<u8>) -> Result<()> {
        into[..bytes].copy_from_slice(&self.bufs[buffer][..bytes]);
        Ok(())
    }
    fn load(&mut self, buffer: &str, bytes: usize, from: &Vec<u8>) -> Result<()> {
        self.bufs.get_mut(buffer).unwrap()[..bytes].copy_from_slice(&from[..bytes]);
        Ok(())
    }
    fn bytes(&self, from: &Vec<u8>, len: usize) -> Result<Vec<u8>> {
        Ok(from[..len].to_vec())
    }
    fn state_bytes(&self, state: &str) -> Result<usize> {
        Ok(self.states[state].len())
    }
    fn read_state(&self, state: &str, at: Range<usize>) -> Result<Vec<u8>> {
        Ok(self.states[state][at].to_vec())
    }
    fn write_state(&mut self, state: &str, at: usize, bytes: &[u8]) -> Result<()> {
        self.states.get_mut(state).unwrap()[at..at + bytes.len()].copy_from_slice(bytes);
        Ok(())
    }
    fn save_state(&self, state: &str, into: &mut Vec<u8>) -> Result<()> {
        into.copy_from_slice(&self.states[state]);
        Ok(())
    }
    fn load_state(&mut self, state: &str, from: &Vec<u8>) -> Result<()> {
        self.states.get_mut(state).unwrap().copy_from_slice(from);
        Ok(())
    }
    fn zero_states(&mut self) -> Result<()> {
        for s in self.states.values_mut() {
            s.fill(0);
        }
        Ok(())
    }
    fn time(&mut self, _program: &str, _vars: &Vars, calls: Range<usize>, _iters: usize) -> Result<Vec<f32>> {
        Ok(vec![0.01; calls.len()])
    }
    fn capture(&mut self, _program: &str, _vars: &Vars) -> Result<()> {
        Ok(())
    }
    fn time_captured(&mut self, _program: &str, _vars: &Vars, _iters: usize) -> Result<f32> {
        Ok(0.5)
    }
}

/// The options every fixture runs with unless a test says otherwise: a
/// short, fixed workload; no timing.
pub fn options() -> Options {
    Options {
        a: "A".into(),
        b: "B".into(),
        prompt: None,
        prefill: 5,
        decode_steps: 4,
        logit_ulp: 4,
        fuzz: 6,
        chunk: 3,
        iters: 1,
        graph_step: false,
        sweep: false,
        peak_bw: 1.0,
        perf: false,
        noise: true,
        seed: 0x5eed,
    }
}

/// Run A against B with `o`; the report and the lines it printed.
pub fn test(a: &Fixture, b: &Fixture, o: &Options) -> Result<(kern_test::Report, Vec<String>)> {
    let (ma, mb) = (a.manifest(), b.manifest());
    let diff = kern_test::diff::diff(&ma, &mb);
    let mut lines = diff.lines();
    let mut sides = kern_test::Sides { a: Fake::new(ma), b: Fake::new(mb), load_s: 0.0 };
    let report = kern_test::run(o, diff, &mut sides, &mut |ls: &[String]| lines.extend_from_slice(ls))?;
    lines.extend(report.summary.verdict.lines());
    Ok((report, lines))
}

/// The verdict's code and its first words.
pub fn verdict(r: &kern_test::Report) -> (i32, String) {
    let v = &r.summary.verdict;
    (v.code, v.summary.split(':').next().unwrap_or("").to_string())
}

pub fn dim(d: &Dim) -> u64 {
    match d {
        Dim::Const(c) => *c,
        Dim::Var(_) => 0,
    }
}
