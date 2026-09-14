//! The states' accounting, on the host and nowhere else.
//!
//! A manifest's paged and per-sequence states are backed by one budget of
//! physical chunks. This crate decides everything about that budget
//! without touching a device: which chunk backs which page or slot
//! (`chunks`), which sequence holds which pages and slot and what a
//! checkpoint of it shares (`pages`: [`Pool`], [`Lease`], [`Checkpoint`]),
//! which checkpoints are kept and which prefix a new prompt can start from
//! (`prefix`: [`Prefix`]), and which of them are parked in host memory
//! (`host`: [`Host`]). Every decision comes back as data — a [`Remap`], a
//! [`Park`], [`Copies`] — that the runtime executes on its streams.
//!
//! Pure by construction: no clock, no randomness, no iteration over an
//! unordered map, so the same calls in the same order lease the same
//! slots. The tests in `tests/` drive it against reference models and
//! need no GPU.

mod chunks;
mod error;
mod host;
mod pages;
mod prefix;

pub use chunks::{Kind, Remap};
pub use error::{Error, Result};
pub use host::{runs, Host, Park, Parked};
pub use pages::{chunks_for, page_unit, row_tokens, Checkpoint, Copies, Denied, Lease, Pool, Pooled};
pub use prefix::{Chain, Hit, Kept, Prefix, Tier};
