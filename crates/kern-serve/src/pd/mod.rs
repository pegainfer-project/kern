//! Prefill/decode disaggregation: a pipeline head hands each prompt's
//! states to a decode tier and relays what it generates.
//!
//! Two kinds of process meet here. The prefill side is a pipeline (`pp`):
//! its head fronts the client, leases from its one pool and sends items to
//! its stages. The decode side is a set of ranks (`executor`, `ranks`) whose
//! head runs a [`Decoder`] instead of an HTTP scheduler: no prefix cache, no
//! prompt, only rows that arrive whole and decode. The decode tier usually
//! deals its paged states by position over a group (`Ranks` in replicated
//! mode) and splits its per-sequence states by heads; it may as well be one
//! GPU owning its rows.
//!
//! The P head connects to the D head, one TCP stream of frames (a `u32`
//! length, then JSON), and the D head opens with [`Pools`]: every D rank's
//! pool handle and placement, its states' shapes and shards, the longest
//! sequence it holds. The P head checks that its stages, itself among them,
//! hold every D state exactly once with the same shape ([`cover`]), hands
//! the pools to the stages in their tables, and each stage mirrors every D
//! rank's pool (`map_pool`, `place`). From then on, per request ([`Ask`]
//! from P, [`Tell`] from D):
//!
//! - admission asks D for seats first: `Reserve { tokens }` (prompt plus
//!   `max_tokens`) is answered `Grant` (a lease on every member, adopted
//!   without zeroing), `Busy` (no room now; P asks again once D says
//!   `Room`) or `Refuse` (it will never fit). A request of one token never
//!   asks: P finishes it.
//! - where a D rank's chunks are is the connection's, not a request's: D
//!   tells `Placed` with the rank's next epoch whenever its placement moved,
//!   ahead of the grant that needs it, and each seat names the epoch it
//!   lives in. An object held stays where every later placement says, so a
//!   mirror at that epoch or past it writes the seat right. The P head
//!   places its own mirror and forwards the placement to every stage on
//!   its job stream, where it lands ahead of any push behind it; a stage
//!   refuses a push whose seats' epochs its mirrors have not reached.
//! - right after the item that writes the prompt's last rows, the P head
//!   sends every stage a [`Push`]; each stage moves its states into the
//!   seats on its compute stream ([`moves`]: rows dealt by position, a
//!   slot's heads to each member) and, once a mark after them lands, says
//!   `Pushed`. The P lease goes to the prefix cache only after the push is
//!   issued; stream order keeps any later reuse of its pages behind it.
//! - with every stage's `Pushed` and the first token in, P sends `Start`
//!   and D adds the row to its running batch at the next step; every step
//!   answers `Tokens`. P's front streams them to the client as they come,
//!   never behind the head's wait on its pipeline ([`Relays`]), judging stops
//!   with the request's full policy; D stops only at `max_tokens`, so a stop
//!   P sees first ends in `Abort`.
//! - `Abort` frees the seats, and P sends it only while no push is in
//!   flight: before the push, or after every stage said `Pushed`.
//!
//! [`Handoffs`] (until `Start`, one a tier, [`Tiers`] over them all) and
//! [`Relays`] (after) are the P side's half as pure state machines,
//! [`Decoder`] the D head's; the processes around them only move frames.

mod decoder;
mod handoff;
pub(crate) mod mirror;
mod relay;
pub(crate) mod tier;
mod tiers;

use std::collections::BTreeMap;

use anyhow::{bail, Context, Result};
use kern_manifest::types::Manifest;
use kern_pool::{deal, Deal, Interleave, Pages};
use kern_runtime::Split;
use serde::{Deserialize, Serialize};

pub use decoder::{Decoder, Run};
pub use handoff::{Handoffs, Out, Standing};
pub use relay::{Relay, Relays};
pub use tiers::{Gate, Tiers};

/// A request on the P/D connection: the P head's request id.
pub type Req = u64;

/// A pooled state as either tier provisions it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Shape {
    /// Bytes per token row.
    Paged { row: u64 },
    /// Bytes per sequence slot.
    Seq { bytes: u64 },
}

/// The pooled states of `m` by name; fixed and host states are no
/// sequence's and never move.
pub fn shapes(m: &Manifest) -> BTreeMap<String, Shape> {
    m.states
        .iter()
        .filter(|(_, s)| s.host.is_none())
        .filter_map(|(n, s)| match (s.bytes_per_token, s.bytes_per_seq) {
            (0, 0) => None,
            (0, bytes) => Some((n.clone(), Shape::Seq { bytes })),
            (row, _) => Some((n.clone(), Shape::Paged { row })),
        })
        .collect()
}

/// What the decode tier tells a prefill head that connects: everything a
/// stage needs to mirror it and to push into it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pools {
    /// Every D rank's `PoolHandle`, hex, in rank order.
    pub handles: Vec<String>,
    /// Every D rank's `Placement` at epoch 0, hex, in rank order.
    pub placements: Vec<String>,
    pub states: BTreeMap<String, Shape>,
    /// Members of a D group, the ranks one row spans.
    pub group: usize,
    /// Whether the paged states deal their positions over the group;
    /// otherwise the group's first member that holds pages owns them.
    pub dealt: bool,
    /// The per-sequence states split by heads over the group, by name:
    /// their segments.
    pub heads: BTreeMap<String, Vec<(u64, u64)>>,
    /// Tokens per D page.
    pub page: u64,
    /// The longest sequence (prompt and completion) a D row holds.
    pub max_seq_tokens: usize,
}

/// Check that the stages, in stage order, hold every D state exactly once
/// with its shape: a paged state's row bytes the same on both sides, a
/// per-sequence state's slot `group` times a member's when D splits it by
/// heads (and one line of its segments), the same otherwise and then only
/// on a group of one. Every mismatch is reported, not the first.
pub fn cover(stages: &[BTreeMap<String, Shape>], d: &Pools) -> Result<()> {
    let mut why = Vec::new();
    for (name, want) in &d.states {
        let on: Vec<usize> = (0..stages.len()).filter(|&s| stages[s].contains_key(name)).collect();
        let s = match on.as_slice() {
            [s] => *s,
            [] => {
                why.push(format!("`{name}`: the decode tier holds it, no prefill stage does"));
                continue;
            }
            many => {
                why.push(format!("`{name}`: stages {many:?} all hold it; expected exactly one"));
                continue;
            }
        };
        let got = stages[s][name];
        let w = d.group as u64;
        match (got, *want, d.heads.get(name)) {
            (Shape::Paged { row: p }, Shape::Paged { row: q }, _) if p != q => {
                why.push(format!("`{name}`: {p} bytes per token on stage {s}, {q} on the decode tier"))
            }
            (Shape::Paged { .. }, Shape::Paged { .. }, _) => {}
            (Shape::Seq { bytes: p }, Shape::Seq { bytes: q }, Some(segs)) => {
                let line: u64 = segs.iter().map(|(r, x)| r * x).sum();
                if p != q * w || line != p {
                    why.push(format!(
                        "`{name}`: {p} bytes per slot on stage {s}; expected {} ({q} on each of {w} members, one line of its heads' segments, {line})",
                        q * w
                    ));
                }
            }
            (Shape::Seq { bytes: p }, Shape::Seq { bytes: q }, None) if p != q || w > 1 => why.push(format!(
                "`{name}`: {p} bytes per slot on stage {s}, {q} on each of {w} decode members and not split by heads; expected the same bytes on a group of one"
            )),
            (Shape::Seq { .. }, Shape::Seq { .. }, None) => {}
            (got, want, _) => why.push(format!("`{name}`: {got:?} on stage {s}, {want:?} on the decode tier")),
        }
    }
    for (s, states) in stages.iter().enumerate() {
        for name in states.keys().filter(|n| !d.states.contains_key(*n)) {
            why.push(format!("`{name}`: stage {s} holds it, the decode tier does not"));
        }
    }
    match why.is_empty() {
        true => Ok(()),
        false => bail!("the prefill stages do not cover the decode tier's states: {}", why.join("; ")),
    }
}

/// The P head's words to the D head.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Ask {
    /// Seats for a sequence of up to `tokens` (prompt and completion).
    Reserve { req: Req, tokens: usize },
    /// The seats hold the prompt's `len` positions: decode from `first`,
    /// at most `max_tokens` tokens more.
    Start { req: Req, first: u32, len: usize, max_tokens: usize },
    /// Free the seats; nothing is pushing into them.
    Abort { req: Req },
}

/// The D head's words to the P head.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Tell {
    Pools(Pools),
    /// D rank `rank`'s placement (hex) moved: this is its `epoch`th, every
    /// epoch one past the last.
    Placed {
        rank: usize,
        epoch: u64,
        placement: String,
    },
    /// The seats, one per member of a group in member order, and the
    /// tier's load with them held.
    Grant {
        req: Req,
        seats: Vec<Seat>,
        load: Load,
    },
    /// No room now; a `Room` follows once there may be.
    Busy {
        req: Req,
    },
    Room,
    Refuse {
        req: Req,
        why: Refusal,
    },
    /// One decode step: each running request's token, and whether it was
    /// the last D will make; the tier's load once the step's last rows left.
    Tokens {
        step: Vec<(Req, u32, bool)>,
        load: Load,
    },
}

/// How full a decode tier is: its seats held (reserved and running) and its
/// pages held by rows of the pages rows can hold, every rank together.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Load {
    pub seats: usize,
    pub pages: usize,
    pub pages_total: usize,
}

/// The decode tier a request reserves on, of `seats` (each tier's seats
/// held, `None` for one that cannot be asked now): power of two choices,
/// the two candidates drawn among those that can be asked by a hash of
/// `req` rather than a random draw, so the same requests go the same way;
/// the one holding fewer seats wins, the first drawn on a tie.
pub fn pick(req: Req, seats: &[Option<usize>]) -> Option<usize> {
    let ready: Vec<(usize, usize)> = seats.iter().enumerate().filter_map(|(i, s)| s.map(|s| (i, s))).collect();
    let n = ready.len() as u64;
    let h = mix(req);
    let a = (h % n.max(1)) as usize;
    let b = match n {
        0 | 1 => a,
        _ => ((a as u64 + 1 + (h / n) % (n - 1)) % n) as usize,
    };
    let (&(i, s), &(j, t)) = (ready.get(a)?, ready.get(b)?);
    Some(if t < s { j } else { i })
}

/// SplitMix64's finalizer: a request id spread over every bit.
fn mix(x: u64) -> u64 {
    let x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    let x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Why a reserve can never be seated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Refusal {
    ContextLength { limit: usize },
    KvBudget,
}

/// One D member's part of a row: its rank, its pages and its slot, and the
/// epoch of the rank's placement it was granted in.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Seat {
    pub rank: usize,
    pub pages: Vec<i32>,
    pub slot: Option<i32>,
    pub epoch: u64,
}

/// What every stage moves for one request: the prompt's `len` positions in
/// the P pages and slot, into the D seats of decode tier `tier`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Push {
    pub req: Req,
    pub tier: usize,
    pub pages: Vec<i32>,
    pub slot: Option<i32>,
    pub len: usize,
    pub seats: Vec<Seat>,
}

/// One copy a stage runs for a push, into D rank `rank`'s mirror.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Move {
    /// The member's rows of a paged state.
    Rows { state: String, rank: usize, deal: Deal },
    /// Member `part`'s heads of a per-sequence state.
    Part { state: String, rank: usize, from: i32, into: i32, split: Split, part: u64 },
    /// A per-sequence state whole.
    Slot { state: String, rank: usize, from: i32, into: i32 },
}

/// The copies a stage holding `mine` runs for `push`, its pages `unit`
/// tokens each: member by member, state by state, none for a stage that
/// holds no D state. The rows of a sequence
/// are dealt once per member and the same deal serves every paged state,
/// which share page ids on either side.
pub fn moves(push: &Push, d: &Pools, mine: &BTreeMap<String, Shape>, unit: u64) -> Result<Vec<Move>> {
    let (w, len) = (d.group, push.len as u64);
    if push.seats.len() != w {
        bail!("{} seats for a decode group of {w}", push.seats.len());
    }
    let src = Pages { ids: &push.pages, unit };
    let paged = mine.values().any(|s| matches!(s, Shape::Paged { .. }));
    let mut out = Vec::new();
    for (m, seat) in push.seats.iter().enumerate() {
        let dst = Pages { ids: &seat.pages, unit: d.page };
        let rows = match (paged, d.dealt, seat.pages.is_empty()) {
            (false, ..) => None,
            (true, true, _) => Some(deal(src, dst, Interleave::new(w as u64)?, m as u64, len)?),
            (true, false, false) => Some(deal(src, dst, Interleave::new(1)?, 0, len)?),
            (true, false, true) => None,
        };
        for (state, shape) in mine {
            let rank = seat.rank;
            let slots = || -> Result<(i32, i32)> {
                let from = push.slot.with_context(|| format!("`{state}` is per sequence; the push names no slot"))?;
                let into =
                    seat.slot.with_context(|| format!("`{state}` is per sequence; rank {rank}'s seat has no slot"))?;
                Ok((from, into))
            };
            match (shape, &rows, d.heads.get(state)) {
                (Shape::Paged { .. }, Some(deal), _) => {
                    out.push(Move::Rows { state: state.clone(), rank, deal: deal.clone() })
                }
                (Shape::Paged { .. }, None, _) => {}
                (Shape::Seq { .. }, _, Some(segs)) => {
                    let (from, into) = slots()?;
                    let split = Split::new(segs, w as u64)?;
                    out.push(Move::Part { state: state.clone(), rank, from, into, split, part: m as u64 });
                }
                (Shape::Seq { .. }, _, None) => {
                    let (from, into) = slots()?;
                    out.push(Move::Slot { state: state.clone(), rank, from, into });
                }
            }
        }
    }
    Ok(out)
}
