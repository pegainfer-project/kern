//! What the AgentX replays share: the trace (HF
//! `semianalysisai/cc-traces-weka-062126`, one session per line) flattened
//! to timed requests, tokens synthesized from its 64-token block hashes,
//! and the page-only manifest a pool is built from. Each replay uses its
//! own subset.
#![allow(dead_code)]

use std::io::{BufRead, BufReader};
use std::path::Path;

use kern_manifest::types::Manifest;
use serde::Deserialize;

#[derive(Deserialize)]
pub struct Session {
    requests: Vec<Record>,
}

#[derive(Deserialize)]
pub struct Record {
    t: f64,
    #[serde(rename = "type")]
    kind: String,
    #[serde(rename = "in", default)]
    input: u64,
    #[serde(default)]
    out: u64,
    #[serde(default)]
    hash_ids: Vec<u32>,
    api_time: Option<f64>,
    #[serde(default)]
    requests: Vec<Record>,
}

/// One model request: `t` seconds into its session, `input` prompt tokens
/// whose 64-token blocks hash to `blocks`, `out` generated, live `api`
/// seconds.
pub struct Req {
    pub session: usize,
    pub t: f64,
    pub input: usize,
    pub out: usize,
    pub blocks: Vec<u32>,
    pub api: f64,
    pub subagent: bool,
}

pub const BLOCK: usize = 64;

fn flatten(session: usize, s: Session) -> Vec<Req> {
    let req = |q: Record, subagent: bool| Req {
        session,
        t: q.t,
        input: q.input as usize,
        out: q.out as usize,
        blocks: q.hash_ids,
        api: q.api_time.unwrap_or(1.0),
        subagent,
    };
    let mut out: Vec<Req> = s
        .requests
        .into_iter()
        .flat_map(|r| match r.kind.as_str() {
            "subagent" => r.requests.into_iter().map(|q| req(q, true)).collect(),
            _ => vec![req(r, false)],
        })
        .collect();
    out.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap());
    out
}

/// The trace's requests in admission order: sessions start `concurrency`
/// at a time, in file order, each when the one that many before it ends.
pub struct Trace {
    pub reqs: Vec<Req>,
    pub sessions: usize,
    offset: Vec<f64>,
    /// Request indices by admission time, ties by index.
    pub order: Vec<usize>,
}

impl Trace {
    pub fn load(path: &Path, sessions_cap: usize, concurrency: usize) -> Trace {
        let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let mut reqs: Vec<Req> = Vec::new();
        let mut spans: Vec<(f64, f64)> = Vec::new();
        for (i, line) in BufReader::new(file).lines().enumerate().take(sessions_cap) {
            let s: Session = serde_json::from_str(&line.unwrap()).expect("trace line");
            let flat = flatten(i, s);
            let end = flat.iter().map(|r| r.t + r.api).fold(0.0, f64::max);
            spans.push((flat.first().map_or(0.0, |r| r.t), end));
            reqs.extend(flat);
        }
        let sessions = spans.len();
        let mut offset = vec![0.0f64; sessions];
        for k in concurrency..sessions {
            offset[k] = offset[k - concurrency] + spans[k - concurrency].1;
        }
        let at = |r: &Req| offset[r.session] + r.t;
        let mut order: Vec<usize> = (0..reqs.len()).collect();
        order.sort_by(|&a, &b| at(&reqs[a]).partial_cmp(&at(&reqs[b])).unwrap().then(a.cmp(&b)));
        Trace { reqs, sessions, offset, order }
    }

    /// When request `ri` is admitted.
    pub fn at(&self, ri: usize) -> f64 {
        let r = &self.reqs[ri];
        self.offset[r.session] + r.t
    }

    /// The tokens of request `i`: its prompt, then its output as the next
    /// request of its session carries it when one extends it, else tokens
    /// nothing else shares.
    pub fn tokens(&self, i: usize) -> Vec<i64> {
        let reqs = &self.reqs;
        let r = &reqs[i];
        let mut out: Vec<i64> = (0..r.input).map(|p| token(r.session, i, &r.blocks, p)).collect();
        let successor = reqs[i + 1..].iter().take(16).find(|q| {
            q.session == r.session && q.blocks.len() > r.blocks.len() && q.blocks[..r.blocks.len()] == r.blocks[..]
        });
        out.extend((r.input..r.input + r.out).map(|p| match successor {
            Some(q) if p / BLOCK < q.blocks.len() => token(q.session, i, &q.blocks, p),
            _ => token(r.session, i, &[], p),
        }));
        out
    }
}

/// The token at position `p` of a request whose blocks are `blocks`:
/// from the block hash when one covers it, else unique to (`session`,
/// `req`, `p`).
fn token(session: usize, req: usize, blocks: &[u32], p: usize) -> i64 {
    match blocks.get(p / BLOCK) {
        Some(&h) => ((session as i64) << 40) | ((h as i64) << 7) | (p % BLOCK) as i64,
        None => (1 << 62) | ((session as i64) << 40) | ((req as i64) << 20) | p as i64,
    }
}

/// A manifest with one paged state of `kv_bytes` per token in pages of
/// `unit`, a recurrent state of `state` bytes per sequence when nonzero,
/// and block tables of `row` pages.
pub fn manifest(unit: usize, kv_bytes: u64, state: u64, slots: usize, row: usize) -> Manifest {
    let state_json = if state > 0 { format!(r#", "rec": {{"bytes_per_seq": {state}}}"#) } else { String::new() };
    let line_json = if state > 0 {
        r#", "line_index": {"kind": "input", "dtype": "i32", "shape": [1, "seqs"], "domain": {"index_into": "rec", "stride": 1}}"#
    } else {
        ""
    };
    Manifest::from_json(&format!(
        r#"{{
        "schema_version": 7, "model": "replay", "vars": {{"tokens": {{"max": 1, "axis": "rows"}}, "seqs": {{"max": {seqs}, "axis": "groups"}}}},
        "states": {{"kv": {{"bytes_per_token": {kv_bytes}}}{state_json}}},
        "buffers": {{
            "block_table": {{"kind": "input", "dtype": "i32", "shape": ["seqs", {row}], "domain": {{"index_into": "kv", "stride": {unit}}}}}{line_json}
        }},
        "modules": {{}}, "ops": {{}}, "programs": {{}}
    }}"#,
        seqs = slots.saturating_sub(2).max(1)
    ))
    .expect("synthetic manifest")
}

/// The `p` quantile of `xs`, sorting them; none when empty.
pub fn pct<T: Copy + Ord>(xs: &mut [T], p: f64) -> Option<T> {
    xs.sort_unstable();
    xs.get(((xs.len() as f64 * p) as usize).min(xs.len().checked_sub(1)?)).copied()
}
