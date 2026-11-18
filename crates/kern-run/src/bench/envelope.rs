//! A manifest's envelope as a workload: `kern workload` without a trace.
//!
//! A server's trace says which shapes its traffic made ([`super::served`]).
//! Before there is a trace the manifest still says which shapes can be
//! made at all: its protocol bounds a chunk's rows and a step's sequences,
//! and its page tables bound a context. This writes the sweep that covers
//! those bounds with the fewest scenarios that still price a call inside
//! them, for someone who needs a call's time as a function of its shape (a
//! capacity estimate, a simulated server) and has no traffic to read.
//!
//! What is swept is what measurement said a price needs (docs/bench.md,
//! "A workload from the manifest alone"):
//!
//! - **Two contexts.** What a context adds to a call was linear in it, so
//!   a call is measured over its shallowest context and its deepest, and
//!   never in between.
//! - **A step at every batch size.** The sizes are the ones a server pads
//!   its batches to. They are the server's policy and no manifest says
//!   them, so they are given.
//! - **A chunk's rows densely over an empty sequence, sparsely over a deep
//!   one.** A chunk's time over no context is not smooth in its rows (one
//!   row past a tile cost seven percent), and no formula in rows and
//!   context came within a tenth of it, so it is tabulated every
//!   [`EVERY`] rows. What a context adds moves with the rows only in
//!   steps, so it is measured every [`DEEP`] rows and read from the
//!   measured chunk above.
//!
//! A price is then two numbers a call size: its time over the shallowest
//! context, and the slope to the deepest. The deepest context is what the
//! page-table row and the state a scenario may hold leave a sequence; the
//! state is given, since bench sizes one allocation for its largest
//! scenario and the GPU it runs on is not in the manifest.
//!
//! Nothing is validated here that the plan does not already say: a batch
//! size or a chunk the manifest has no program for is a `drop` line, as in
//! any workload.

use std::fmt::Write as _;
use std::path::Path;

use anyhow::{Context as _, Result};
use kern_manifest::{Protocol, Verified};

use super::say;

/// Rows between the chunks measured over an empty sequence.
pub const EVERY: u64 = 32;
/// Rows between the chunks measured over the deepest context.
pub const DEEP: u64 = 4 * EVERY;

#[derive(clap::Args, Debug)]
pub struct EnvelopeOpts {
    /// Without a trace: rows of the largest chunk to price (the server's prefill chunk)
    #[arg(long, required_unless_present = "trace", conflicts_with = "trace")]
    chunk: Option<u64>,
    /// Without a trace: sequences a step is priced at (the batch sizes the server pads to)
    #[arg(long, value_delimiter = ',', required_unless_present = "trace", conflicts_with = "trace")]
    groups: Vec<u64>,
    /// Without a trace: GiB of paged state a scenario may hold on a rank
    #[arg(long, required_unless_present = "trace", conflicts_with = "trace")]
    state_gib: Option<u64>,
}

/// What an envelope is asked to cover.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ask {
    /// Rows of the largest chunk.
    pub chunk: u64,
    /// Sequences of every step.
    pub groups: Vec<u64>,
    /// Tokens of state a scenario may hold.
    pub tokens: u64,
}

/// The deepest context, in whole pages, that `seqs` sequences of `rows`
/// rows can each sit on: what the page-table row holds, and their share of
/// the state.
fn deepest(ask: &Ask, unit: u64, row: u64, seqs: u64, rows: u64) -> u64 {
    row.min(ask.tokens / seqs.max(1)).saturating_sub(rows) / unit * unit
}

/// The chunks measured `every` rows apart up to the largest, which is
/// always one of them, after the powers of two below the first.
fn chunks(ask: &Ask, every: u64) -> Vec<u64> {
    let small = (0..).map(|k| 1 << k).take_while(|&c| c < every);
    let spaced = (1..).map(|i| i * every).take_while(|&c| c < ask.chunk);
    small.chain(spaced).filter(|&c| c < ask.chunk).chain([ask.chunk]).collect()
}

fn list(v: &[u64]) -> String {
    v.iter().map(u64::to_string).collect::<Vec<_>>().join(", ")
}

/// The workload file that covers `ask` on a manifest whose pages hold
/// `unit` tokens and whose sequences hold `row`: every step size over one
/// page and over its deepest context, then the chunks.
pub fn sweep(p: &Protocol, unit: u64, row: u64, ask: &Ask) -> String {
    let mut out = format!(
        "# the envelope: chunks of up to {} rows, steps of {} sequences, {} tokens of state\n\
         samples = 12\nseed = 0\n",
        ask.chunk,
        list(&ask.groups),
        ask.tokens,
    );
    for rows in p.row_shapes() {
        for &g in &ask.groups {
            let mut context = vec![unit, deepest(ask, unit, row, g, rows).max(unit)];
            context.dedup();
            write!(
                out,
                "\n# a step of {rows} row{} a sequence · over one page and the deepest context\n\
                 [[sweep]]\ngroups = [{g}]\nrows = [{rows}]\ncontext = [{}]\n",
                if rows == 1 { "" } else { "s" },
                list(&context),
            )
            .expect("writing to a string");
        }
    }
    if p.chunk().is_some() && ask.chunk > 0 {
        let deep = deepest(ask, unit, row, 1, ask.chunk);
        write!(
            out,
            "\n# chunks over an empty sequence · every {EVERY} rows\n[[sweep]]\nchunk = [{}]\ncontext = [0]\n",
            list(&chunks(ask, EVERY)),
        )
        .expect("writing to a string");
        if deep > 0 {
            write!(
                out,
                "\n# chunks over the deepest context · every {DEEP} rows · what a context adds\n\
                 [[sweep]]\nchunk = [{}]\ncontext = [{deep}]\n",
                list(&chunks(ask, DEEP)),
            )
            .expect("writing to a string");
        }
    }
    out
}

/// Writes the envelope of the manifest at `path` as a workload, saying what
/// it covers and what `kern bench` will make of it.
pub fn run(o: EnvelopeOpts, out: &Path, path: &Path) -> Result<()> {
    let json = std::fs::read(path).with_context(|| format!("manifest {}", path.display()))?;
    // As bench loads it, so the plan below is the plan bench will make.
    let m = Verified::from_json(std::str::from_utf8(&json)?)?.self_hosted()?;
    let protocol = Protocol::check_unsampled(&m)?;
    let unit = kern_pool::page_unit(&m);
    let row = kern_pool::row_tokens(&m, unit)
        .with_context(|| format!("{} pages no state, so a context has no depth to sweep", path.display()))?;
    let per_token: u64 = m.states.values().map(|s| s.bytes_per_token).sum();
    let gib = o.state_gib.expect("clap requires --state-gib without a trace");
    let ask = Ask {
        chunk: o.chunk.expect("clap requires --chunk without a trace"),
        groups: o.groups,
        tokens: (gib << 30) / per_token,
    };
    say(
        "envelope",
        format!(
            "chunks of up to {} rows · steps of {} sequences · {} tokens of state in {gib} GiB · a sequence holds {row}",
            ask.chunk,
            list(&ask.groups),
            ask.tokens,
        ),
    );
    super::save(&sweep(&protocol, unit, row, &ask), out, &m, &protocol)
}
