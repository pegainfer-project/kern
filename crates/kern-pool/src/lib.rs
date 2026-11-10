//! The states' accounting, on the host and nowhere else.
//!
//! A manifest's paged and per-sequence states are so many pages and
//! sequence slots, sized at load. This crate decides everything about
//! them without touching a device: which sequence holds which pages and
//! slot and what a checkpoint of it shares (`pages`: [`Pool`], [`Lease`];
//! `store`: [`Checkpoint`], [`Copies`]), which prefix a new prompt can
//! start from (`prefix`: [`Prefix`]), and which rows of a sequence go to
//! which member of a group that deals positions (`deal`: [`Interleave`],
//! [`deal`]), and what the pooled tier is asked and told (`pooled`:
//! [`Keys`], [`Probe`], [`Landing`], [`snapshots`]), and where that tier
//! keeps its objects when it is one local disk (`disk`: [`Disk`], over the
//! file as a [`Ring`] and the pinned block its bytes pass through as an
//! [`Arena`]). Every decision comes
//! back as data — a [`Copies`], a [`Deal`] — that the runtime executes on
//! its streams; every handle owns what it names and returns it when
//! dropped.
//!
//! Pure by construction: no clock, no randomness, no iteration over an
//! unordered map, so the same calls in the same order lease the same
//! slots and name the same keys. The tests in `tests/` drive it against reference models
//! and need no GPU.

mod arena;
mod deal;
mod disk;
mod error;
mod pages;
mod pooled;
mod prefix;
mod ring;
mod store;

pub use arena::{Arena, Room};
pub use deal::{deal, runs, Deal, Interleave, Pages, RowRun};
pub use disk::{Disk, Move, Plan};
pub use error::{Error, Result};
pub use pages::{page_unit, row_tokens, Denied, Kind, Lease, Pool, Pooled};
pub use pooled::{grid, landed, objects, snapshots, Chain, Hash, Keys, Landing, Object, Probe};
pub use prefix::{Hit, Kept, Prefix};
pub use ring::{Extent, Placed, Ring};
pub use store::{Checkpoint, Copies};
