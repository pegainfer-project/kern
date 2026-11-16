//! Request-level trace loading, Mooncake block-hash synthesis, and multi-turn
//! agentic session generation.
//!
//! Supports both open-loop replay (requests arrive at fixed trace timestamps)
//! and closed-loop multi-turn session replay (turn `k` of a session arrives
//! `think_time_ns` after turn `k - 1` finishes, so endpoint speedups shift
//! downstream arrival times).

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};

/// How request arrivals are scheduled during cluster simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArrivalMode {
    /// Every request arrives at its fixed `arrival_ns`.
    OpenLoop,
    /// Turn 0 of each session arrives at `arrival_ns`; subsequent turns in the
    /// same `session_id` arrive `think_time_ns` after the previous turn completes.
    ClosedLoop,
}

/// One request in a replay trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceRequest {
    pub id: u64,
    pub session_id: Option<String>,
    pub turn_index: u32,
    pub arrival_ns: u64,
    pub think_time_ns: u64,
    pub prompt_tokens: Vec<i64>,
    pub output_tokens: usize,
}

/// Ordered request workload ready for [`crate::cluster::SimCluster`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RequestTrace {
    pub name: String,
    pub requests: Vec<TraceRequest>,
}

/// Options for parsing Mooncake / JSONL traces.
#[derive(Debug, Clone, Copy)]
pub struct JsonlTraceOptions {
    /// Tokens per `hash_ids` entry when expanding Mooncake block hashes into
    /// synthetic token sequences (`64` matches a standard 64-token KV page).
    pub hash_block_tokens: usize,
    /// Timestamp unit in the JSONL file (`1_000_000` when `timestamp` is in ms).
    pub timestamp_unit_ns: u64,
}

impl Default for JsonlTraceOptions {
    fn default() -> Self {
        Self {
            hash_block_tokens: 64,
            timestamp_unit_ns: 1_000_000,
        }
    }
}

/// Parameters for generating a deterministic multi-turn agentic session workload.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SyntheticSessionConfig {
    pub sessions: usize,
    pub turns_per_session: usize,
    /// Shared system/tool prompt length across sessions (tests cross-session prefix reuse).
    pub system_prompt_tokens: usize,
    /// Mean new user/tool tokens appended per turn.
    pub user_tokens_per_turn: usize,
    /// Mean assistant output tokens generated per turn (also appended to the
    /// next turn's prompt so multi-turn prefix + recurrent slot reuse applies).
    pub output_tokens_per_turn: usize,
    /// Mean inter-arrival gap between new sessions, in milliseconds.
    pub session_arrival_gap_ms: u64,
    /// Client/tool think time between turn completion and next turn arrival, in ms.
    pub think_time_ms: u64,
    /// Number of distinct system prompt templates (`1` means every session shares one).
    pub system_groups: usize,
    pub seed: u64,
}

impl Default for SyntheticSessionConfig {
    fn default() -> Self {
        Self {
            sessions: 16,
            turns_per_session: 4,
            system_prompt_tokens: 512,
            user_tokens_per_turn: 192,
            output_tokens_per_turn: 128,
            session_arrival_gap_ms: 40,
            think_time_ms: 150,
            system_groups: 2,
            seed: 42,
        }
    }
}

#[derive(Deserialize)]
struct RawJsonlLine {
    #[serde(default)]
    session_id: Option<String>,
    #[serde(default)]
    timestamp: Option<f64>,
    #[serde(default)]
    arrival_ms: Option<f64>,
    #[serde(default)]
    think_ms: Option<f64>,
    #[serde(default)]
    input_length: Option<usize>,
    #[serde(default)]
    output_length: Option<usize>,
    #[serde(default)]
    hash_ids: Option<Vec<u64>>,
    #[serde(default)]
    prompt_tokens: Option<Vec<i64>>,
}

impl RequestTrace {
    /// Read a Mooncake or native JSONL request trace from `path`.
    pub fn from_jsonl_file(path: &Path, opts: JsonlTraceOptions) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading trace {}", path.display()))?;
        let name = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "trace".into());
        Self::from_jsonl_str(&name, &text, opts)
    }

    /// Parse Mooncake (`timestamp`, `input_length`, `output_length`, `hash_ids`)
    /// or native (`session_id`, `arrival_ms`, `think_ms`, `prompt_tokens`, `output_length`) JSONL.
    pub fn from_jsonl_str(name: &str, text: &str, opts: JsonlTraceOptions) -> Result<Self> {
        ensure!(opts.hash_block_tokens > 0, "hash_block_tokens must be > 0");
        let mut requests = Vec::new();
        let mut session_turns: BTreeMap<String, (u32, u64)> = BTreeMap::new();
        let mut base_ns: Option<u64> = None;

        for (line_no, raw_line) in text.lines().enumerate() {
            let trimmed = raw_line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let row: RawJsonlLine = serde_json::from_str(trimmed)
                .with_context(|| format!("JSONL line {}", line_no + 1))?;

            let raw_arrival_ns = if let Some(ms) = row.arrival_ms {
                (ms.max(0.0) * 1e6) as u64
            } else if let Some(ts) = row.timestamp {
                (ts.max(0.0) * opts.timestamp_unit_ns as f64) as u64
            } else {
                0
            };
            let origin = *base_ns.get_or_insert(raw_arrival_ns);
            let arrival_ns = raw_arrival_ns.saturating_sub(origin);

            let id = requests.len() as u64;
            let prompt_tokens = if let Some(toks) = row.prompt_tokens {
                ensure!(!toks.is_empty(), "JSONL line {}: empty prompt_tokens", line_no + 1);
                toks
            } else {
                let input_len = row.input_length.unwrap_or(64).max(1);
                tokens_from_hashes(row.hash_ids.as_deref().unwrap_or(&[]), input_len, opts.hash_block_tokens, id)
            };

            let output_tokens = row.output_length.unwrap_or(64).max(1);
            let explicit_think = row.think_ms.map(|ms| (ms.max(0.0) * 1e6) as u64);

            let (turn_index, think_time_ns) = if let Some(sid) = &row.session_id {
                let entry = session_turns.entry(sid.clone()).or_insert((0, arrival_ns));
                let t_idx = entry.0;
                let inferred_gap = arrival_ns.saturating_sub(entry.1);
                entry.0 += 1;
                entry.1 = arrival_ns;
                (t_idx, explicit_think.unwrap_or(inferred_gap))
            } else {
                (0, explicit_think.unwrap_or(0))
            };

            requests.push(TraceRequest {
                id,
                session_id: row.session_id,
                turn_index,
                arrival_ns,
                think_time_ns,
                prompt_tokens,
                output_tokens,
            });
        }

        requests.sort_by_key(|r| (r.arrival_ns, r.id));
        for (idx, req) in requests.iter_mut().enumerate() {
            req.id = idx as u64;
        }

        Ok(Self {
            name: name.to_string(),
            requests,
        })
    }

    /// Deterministically synthesize a multi-turn agentic session workload where
    /// follow-up turns carry the exact assistant tokens emitted by prior turns.
    pub fn synthetic_sessions(cfg: SyntheticSessionConfig) -> Self {
        let groups = cfg.system_groups.max(1);
        let sys_prompts: Vec<Vec<i64>> = (0..groups)
            .map(|g| {
                (0..cfg.system_prompt_tokens)
                    .map(|pos| token_from_key(0x5157_0000_0000_0000 ^ (g as u64), pos as u64))
                    .collect()
            })
            .collect();

        let mut rng = SplitMix64::new(cfg.seed);
        let mut raw_reqs = Vec::with_capacity(cfg.sessions * cfg.turns_per_session);
        let mut arrival_cursor_ns = 0u64;

        for s in 0..cfg.sessions {
            if s > 0 {
                let jitter_pct = 50 + (rng.next_u64() % 101); // 0.5x .. 1.5x mean gap
                arrival_cursor_ns += (cfg.session_arrival_gap_ms * 1_000_000 * jitter_pct) / 100;
            }
            let sid = format!("sess-{s:04}");
            let mut history = sys_prompts[s % groups].clone();
            let mut open_loop_cursor_ns = arrival_cursor_ns;

            for t in 0..cfg.turns_per_session {
                let user_len = jitter_len(&mut rng, cfg.user_tokens_per_turn);
                for pos in 0..user_len {
                    let key = ((s as u64) << 32) ^ ((t as u64) << 16) ^ 0xA1;
                    history.push(token_from_key(key, pos as u64));
                }

                let out_len = jitter_len(&mut rng, cfg.output_tokens_per_turn);
                let think_ns = if t == 0 {
                    0
                } else {
                    let jitter_pct = 75 + (rng.next_u64() % 51);
                    (cfg.think_time_ms * 1_000_000 * jitter_pct) / 100
                };
                if t > 0 {
                    open_loop_cursor_ns += think_ns + (out_len as u64) * 5_000_000;
                }

                let prompt_tokens = history.clone();
                // Append the deterministic assistant reply that `SimCluster` emits for this
                // request so turn `t + 1` matches `Prefix` both on paged KV and recurrent slots.
                for pos in 0..out_len {
                    history.push(generated_token_id(&sid, t as u32, pos));
                }

                raw_reqs.push(TraceRequest {
                    id: 0,
                    session_id: Some(sid.clone()),
                    turn_index: t as u32,
                    arrival_ns: open_loop_cursor_ns,
                    think_time_ns: think_ns,
                    prompt_tokens,
                    output_tokens: out_len,
                });
            }
        }

        raw_reqs.sort_by_key(|r| (r.arrival_ns, r.session_id.clone(), r.turn_index));
        for (idx, r) in raw_reqs.iter_mut().enumerate() {
            r.id = idx as u64;
        }

        Self {
            name: format!("synthetic-{}x{}", cfg.sessions, cfg.turns_per_session),
            requests: raw_reqs,
        }
    }
}

/// Deterministic token emitted at output offset `pos` for `(session_or_req, turn)`.
/// Shared between [`RequestTrace::synthetic_sessions`] and [`crate::cluster::SimCluster`]
/// so multi-turn prompts continue the exact token sequence retired into [`kern_pool::Prefix`].
pub fn generated_token_id(session_key: &str, turn_index: u32, pos: usize) -> i64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in session_key.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h ^= (turn_index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    token_from_key(h, pos as u64)
}

fn tokens_from_hashes(hashes: &[u64], input_len: usize, block_tokens: usize, req_salt: u64) -> Vec<i64> {
    let mut out = Vec::with_capacity(input_len);
    for (blk_idx, &h) in hashes.iter().enumerate() {
        for off in 0..block_tokens {
            if out.len() == input_len {
                return out;
            }
            out.push(token_from_key(h ^ 0x424c_4b00_0000_0000, off as u64));
        }
        let _ = blk_idx;
    }
    let mut tail_pos = 0u64;
    while out.len() < input_len {
        out.push(token_from_key(0x554e_4951_0000_0000 ^ req_salt, tail_pos));
        tail_pos += 1;
    }
    out
}

fn token_from_key(seed: u64, offset: u64) -> i64 {
    let mut z = seed.wrapping_add(offset.wrapping_mul(0x9e37_79b9_7f4a_7c15));
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    // Keep synthetic token IDs in a clean positive vocabulary range `1..131072`.
    1 + (z % 131_071) as i64
}

fn jitter_len(rng: &mut SplitMix64, base: usize) -> usize {
    let base = base.max(1) as u64;
    let pct = 80 + (rng.next_u64() % 41); // 0.8x .. 1.2x
    ((base * pct) / 100).max(1) as usize
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}
