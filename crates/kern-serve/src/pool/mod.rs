//! The pooled tier's client, one per rank process, over one of two
//! stores: Mooncake Store (`mooncake`), which every instance of the
//! manifest shares over RDMA or NVLink, or this rank's own local disk
//! (`ssd`), written through on every snapshot.
//!
//! Both answer the same four acts the scheduler decides (`Act::Probe`,
//! `Put`, `Get`, `Mark`, `ranks.rs`) with the same flags, so nothing above
//! this module knows which one it has. They differ in who moves the bytes
//! to the device: Mooncake's NIC reads and writes the pooled states
//! directly, after the act's [`kern_runtime::Mark`]; the disk cannot, so
//! the rank's own copies through a pinned block carry them, and those are
//! issued on the thread that drives the device ([`Client::issue`]).

mod mooncake;
mod ssd;

use std::collections::BTreeMap;

use anyhow::{bail, Result};
use kern_pool::{Kind, Object};
use kern_runtime::{Runtime, Span};

use crate::ranks::Landed;

pub use mooncake::Transport;

/// Where a pooled object's bytes live on this rank: one range per state,
/// in the states' name order, which is also the order of its bytes in a
/// store.
type Buffers = Vec<(u64, usize)>;

/// `o`'s bytes among `spans`, the rank's pooled states by name.
fn buffers(spans: &BTreeMap<String, Span>, o: &Object) -> Buffers {
    spans
        .values()
        .flat_map(|s| {
            let at = |i: i32| (s.ptr + i as u64 * s.object, s.object as usize);
            match s.kind {
                Kind::Page => o.pages.iter().map(|&p| at(p)).collect::<Vec<_>>(),
                Kind::Slot => o.slot.map(at).into_iter().collect(),
            }
        })
        .collect()
}

/// Which store the tier is and how to reach it.
#[derive(Debug, Clone)]
pub enum Config {
    Mooncake(mooncake::Config),
    Ssd(ssd::Config),
}

pub use mooncake::Config as MooncakeConfig;
pub use ssd::Config as SsdConfig;

/// This rank's client of the tier.
pub enum Client {
    Mooncake(mooncake::Client),
    Ssd(ssd::Client),
}

impl Client {
    /// Join the tier with every pooled state of `rt`. Called on the thread
    /// that drives the device, once it is loaded.
    pub fn open(cfg: &Config, rt: &Runtime) -> Result<Client> {
        Ok(match cfg {
            Config::Mooncake(c) => Client::Mooncake(mooncake::Client::open(c, rt.pooled_spans())?),
            Config::Ssd(c) => Client::Ssd(ssd::Client::open(c, rt)?),
        })
    }

    pub fn probe(&self, id: u64, keys: Vec<String>) {
        match self {
            Client::Mooncake(c) => c.probe(id, keys),
            Client::Ssd(c) => c.probe(id, &keys),
        }
    }

    /// Store `objects` once everything issued on `rt` so far has run.
    pub fn put(&self, rt: &Runtime, id: u64, objects: &[Object]) -> Result<()> {
        match self {
            Client::Mooncake(c) => {
                c.put(id, rt.mark()?, objects);
                Ok(())
            }
            Client::Ssd(c) => c.put(rt, id, objects),
        }
    }

    /// Fetch `objects` into their pages and slot, behind everything issued
    /// on `rt` so far.
    pub fn get(&self, rt: &Runtime, id: u64, objects: &[Object]) -> Result<()> {
        match self {
            Client::Mooncake(c) => {
                c.get(id, rt.mark()?, objects);
                Ok(())
            }
            Client::Ssd(c) => c.get(id, objects),
        }
    }

    pub fn mark(&self, id: u64, key: String) {
        match self {
            Client::Mooncake(c) => c.mark(id, key),
            Client::Ssd(c) => c.mark(id, &key),
        }
    }

    /// Issue on `rt` whatever the tier's own threads have readied for the
    /// device since the last call: before every order's acts.
    pub fn issue(&self, rt: &Runtime) -> Result<()> {
        match self {
            Client::Mooncake(_) => Ok(()),
            Client::Ssd(c) => c.issue(rt),
        }
    }

    /// Every act answered since the last call.
    pub fn landed(&self) -> Vec<Landed> {
        match self {
            Client::Mooncake(c) => c.landed(),
            Client::Ssd(c) => c.landed(),
        }
    }

    /// Hand every answer to `f` on a thread of its own instead, as it
    /// comes. A pipeline stage answers so, between items nobody drives its
    /// device, and the disk tier needs the device driven to land a fetch.
    pub fn forward(&mut self, f: impl FnMut(Landed) -> bool + Send + 'static) -> Result<()> {
        match self {
            Client::Mooncake(c) => c.forward(f),
            Client::Ssd(_) => bail!("the disk tier (`--pool-ssd`) serves a scheduler's ranks, not a pipeline stage"),
        }
    }
}
