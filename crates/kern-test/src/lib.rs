//! kern-test: the evidence for a kernel swap, as a library.
//!
//! Given two manifests A (the reference, assumed correct) and B (the
//! candidate) and a [`Side`] driving each, the harness:
//!   1. diffs them structurally ([`diff`]): which ops changed (interface /
//!      impl / added / removed) and, per program, the aligned runs of
//!      calls that differ, the **spans**; everything between spans is
//!      shared;
//!   2. records A ([`record`]): a seeded workload runs once on A and, at
//!      every span of every program run, A's frontier inputs and outputs
//!      are kept, with an image of A's state at the start of each run.
//!      The first run of each program also keeps the state bytes its
//!      spans changed, so the span can be replayed on its own. A's noise
//!      floor (each kept span replayed against itself), A's outputs under
//!      fuzz and A's timings are recorded too: after this A is not needed;
//!   3. replays B ([`replay`]): B runs the same workload starting every
//!      program run from A's state image and every span from A's frontier
//!      inputs, so what B writes is the span's own doing; then B free-runs
//!      the workload and the logits of every step are compared end to
//!      end against A's: the oracle. Fuzz replays the kept spans on B with
//!      the perturbed inputs A saw; perf times B where A was timed. The
//!      verdict is a pure function of what was measured (`FAIL` / `PASS`
//!      / `INCONCLUSIVE`, exit codes 1 / 0 / 2), stated last, after one
//!      line per fact.
//!
//! Recording first and replaying second means the two sides never have
//! to be loaded at once: a model that fills its GPUs leaves no room for a
//! second copy. What the recording keeps on the device ([`Side::Buf`])
//! outlives the side that made it, so B loads A's images device to
//! device; only what a verdict reads goes through the host.
//!
//! A side may be several ranks: an SPMD manifest with a `topology` runs
//! as one runtime per rank, every rank fed the same sequence, every call
//! range run on all ranks at once (a collective inside a span needs its
//! peers). Recording and comparison are rank-local: rank q of B against
//! rank q of A, findings named by rank.
//!
//! The crate knows no CUDA. Logic is `fn(data) -> data` over the manifest
//! ([`diff`], [`workload`], [`compare`], [`report`]); every effect goes
//! through [`Side`]. `kern` implements it over the runtime; the crate's
//! own tests implement it as a host interpreter over a tiny manifest, so
//! every verdict row has a fixture that reaches it without a GPU.

#![forbid(unsafe_code)]

pub mod compare;
pub mod diff;
mod harness;
mod replay;
pub mod report;
pub mod workload;

use std::collections::BTreeMap;
use std::ops::Range;

use anyhow::Result;
use kern_manifest::types::Provision;
use kern_manifest::Verified;

pub use harness::{record, Recording};
pub use replay::replay;
pub use report::{Report, Summary};

/// The var values one call runs at.
pub type Vars = BTreeMap<String, u64>;

/// One driven side: a loaded manifest, as one rank or several, plus the
/// single sequence the workload feeds every rank as. Every method is
/// synchronous; nothing here branches on what it moves. Whatever runs
/// (`run`, `time`, `capture`) runs on every rank at once; whatever moves
/// bytes names the rank.
pub trait Side {
    /// Bytes this side set aside on its device. A handle outlives the side
    /// that allocated it: a side loaded later on the same device (the same
    /// rank) can load from it.
    type Buf;

    fn manifest(&self) -> &Verified;
    fn provision(&self) -> Provision;
    /// The page unit the paged states are laid out in, in tokens.
    fn page(&self) -> u64;
    /// Vocabulary size as the token fill's domain declares it.
    fn vocab(&self) -> u64;
    /// Ranks this side runs as; 1 without a topology.
    fn ranks(&self) -> usize;

    /// Stage `ids` as the sequence's next rows on every rank, in every
    /// fill the manifest declares; returns the call's vars. Does not
    /// advance.
    fn stage(&mut self, ids: &[i64]) -> Result<Vars>;
    /// Back to position 0 (a new prompt over the same slots).
    fn reset(&mut self);
    fn advance(&mut self, n: u64);

    fn calls(&self, program: &str) -> Result<usize>;
    /// Run calls `calls` of a program eagerly on every rank; returns when
    /// the last rank is done. A rank that fails or never returns is an
    /// error naming it.
    fn run(&mut self, program: &str, vars: &Vars, calls: Range<usize>) -> Result<()>;

    /// The first `bytes` of a buffer, on the host.
    fn read(&self, rank: usize, buffer: &str, bytes: usize) -> Result<Vec<u8>>;
    /// Overwrite the first `bytes.len()` bytes of a buffer from the host.
    fn write(&mut self, rank: usize, buffer: &str, bytes: &[u8]) -> Result<()>;
    fn alloc(&self, rank: usize, bytes: usize) -> Result<Self::Buf>;
    /// The first `bytes` of a buffer into `into`.
    fn save(&self, rank: usize, buffer: &str, bytes: usize, into: &mut Self::Buf) -> Result<()>;
    /// The first `bytes` of `from` into a buffer.
    fn load(&mut self, rank: usize, buffer: &str, bytes: usize, from: &Self::Buf) -> Result<()>;
    /// The first `len` bytes of `from`, on the host.
    fn bytes(&self, from: &Self::Buf, len: usize) -> Result<Vec<u8>>;

    fn state_bytes(&self, state: &str) -> Result<usize>;
    fn read_state(&self, rank: usize, state: &str, at: Range<usize>) -> Result<Vec<u8>>;
    fn write_state(&mut self, rank: usize, state: &str, at: usize, bytes: &[u8]) -> Result<()>;
    /// A state's whole allocation into `into`.
    fn save_state(&self, rank: usize, state: &str, into: &mut Self::Buf) -> Result<()>;
    fn load_state(&mut self, rank: usize, state: &str, from: &Self::Buf) -> Result<()>;
    /// Every state on every rank zeroed: a fresh sequence from position 0.
    fn zero_states(&mut self) -> Result<()>;

    /// Per-call time in ms for calls `calls`, minimum over `iters` replays;
    /// with several ranks, the slowest rank's.
    fn time(&mut self, program: &str, vars: &Vars, calls: Range<usize>, iters: usize) -> Result<Vec<f32>>;
    /// Capture a program as a graph at `vars`, for [`Side::time_captured`].
    fn capture(&mut self, program: &str, vars: &Vars) -> Result<()>;
    /// Median wall time per replay of the captured program, in ms.
    fn time_captured(&mut self, program: &str, vars: &Vars, iters: usize) -> Result<f32>;
}

/// What the harness is told beyond the two manifests; the knobs of
/// `kern test`, resolved.
#[derive(Debug, Clone)]
pub struct Options {
    /// Labels for A and B in the report (their paths).
    pub a: String,
    pub b: String,
    /// A real prompt's token ids for the prefill instead of seeded tokens.
    pub prompt: Option<Vec<i64>>,
    /// Prefill length in tokens; 0 = drawn from the seed.
    pub prefill: u64,
    /// Decode steps; the seed picks a length in `[steps/2, steps]`.
    pub decode_steps: u64,
    /// How far the end-to-end logits may move, in ulps at the row's scale,
    /// and still PASS with logit evidence.
    pub logit_ulp: u64,
    /// Fuzz rounds per span (0 disables).
    pub fuzz: usize,
    /// Prefill chunk; 0 = drawn from the seed.
    pub chunk: u64,
    /// Replays for span timing (the minimum is reported).
    pub iters: usize,
    /// Capture both decode programs as graphs for the step time.
    pub graph_step: bool,
    /// Sweep prefill over the rows var's range.
    pub sweep: bool,
    /// Device peak memory bandwidth in GB/s, for the roofline column.
    pub peak_bw: f64,
    pub perf: bool,
    pub noise: bool,
    /// Seed of the workload and the fuzz generator.
    pub seed: u64,
}
