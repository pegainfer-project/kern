//! The states' accounting, on the host and nowhere else.
//!
//! A manifest's paged and per-sequence states are so many pages and
//! sequence slots, sized at load. This crate decides everything about
//! them without touching a device: which sequence holds which pages and
//! slot and what a checkpoint of it shares (`pages`: [`Pool`], [`Lease`],
//! [`Checkpoint`]), what a checkpoint's bytes are in a tier and how they
//! are held
//! (`store`: [`Store`], [`Copies`]), which of them are parked in host
//! memory (`host`: [`Host`], [`Parked`]), which prefix a new prompt can
//! start from (`prefix`: [`Prefix`]), and which rows of a sequence go to
//! which member of a group that deals positions (`deal`: [`Interleave`],
//! [`deal`]). Every decision comes back as data — a [`Copies`], a
//! [`Deal`] — that the runtime executes on its streams;
//! every handle owns what it names and returns it when dropped.
//!
//! Pure by construction: no clock, no randomness, no hash, no iteration
//! over an unordered map, so the same calls in the same order lease the
//! same slots. The tests in `tests/` drive it against reference models
//! and need no GPU.

mod deal;
mod error;
mod host;
mod pages;
mod prefix;
mod store;

pub use deal::{deal, Deal, Interleave, Pages, RowRun};
pub use error::{Error, Result};
pub use host::{runs, Host, Parked};
pub use pages::{page_unit, row_tokens, Checkpoint, Denied, Kind, Lease, Pool, Pooled};
pub use prefix::{Evicted, Found, Hit, Kept, Prefix, Tier};
pub use store::{Copies, Storage, Store};
