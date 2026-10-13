//! A prefill stage's view of the decode tier: every D rank's pool mapped
//! into this runtime, and the pushes run into them.
//!
//! Mapping and placing are synchronous and happen at start (epoch 0), or
//! on the issuing thread when a rank's next placement arrives (`place`
//! waits for this runtime's stream, so a remap on D costs a stage a
//! drained stream once). Epochs come one at a time, in order; a push
//! whose seat names an epoch a mirror has not reached is refused, never
//! written through an older one. A push itself is copies on the compute
//! stream behind the item that wrote the prompt's last rows; whoever
//! issued it marks after it.

use std::collections::BTreeMap;

use anyhow::{ensure, Context, Result};
use kern_pool::Placement;
use kern_runtime::{MappedPool, PoolHandle, Runtime};

use super::{moves, Move, Pools, Push, Shape};

pub(crate) struct Mirrors {
    pools: Pools,
    /// This runtime's states the decode tier holds.
    mine: BTreeMap<String, Shape>,
    /// Every D rank's pool and the epoch of its placement, rank order.
    ranks: Vec<(MappedPool, u64)>,
}

fn placement(hex: &str, q: usize) -> Result<Placement> {
    let b = hex::decode(hex).with_context(|| format!("placement of decode rank {q}"))?;
    Placement::from_bytes(&b).with_context(|| format!("placement of decode rank {q} does not parse"))
}

impl Mirrors {
    /// Map every rank of `pools` into `rt` where its placement says; `mine`
    /// is this runtime's states, of which those the decode tier holds are
    /// pushed.
    pub fn map(rt: &Runtime, pools: Pools, mine: &BTreeMap<String, Shape>) -> Result<Mirrors> {
        let mut ranks = Vec::with_capacity(pools.handles.len());
        for (q, (h, p)) in pools.handles.iter().zip(&pools.placements).enumerate() {
            let b = hex::decode(h).with_context(|| format!("pool handle of decode rank {q}"))?;
            let h =
                PoolHandle::from_bytes(&b).with_context(|| format!("pool handle of decode rank {q} does not parse"))?;
            let mut m = rt.map_pool(&h, &format!("decode rank {q}'s pool"))?;
            rt.place(&mut m, &placement(p, q)?).with_context(|| format!("placing decode rank {q}'s pool"))?;
            ranks.push((m, 0));
        }
        let mine = mine.iter().filter(|(n, _)| pools.states.contains_key(*n)).map(|(n, s)| (n.clone(), *s)).collect();
        Ok(Mirrors { pools, mine, ranks })
    }

    /// Place rank `q`'s mirror at its next placement, `epoch`.
    pub fn place(&mut self, rt: &Runtime, q: usize, epoch: u64, hex: &str) -> Result<()> {
        let (m, at) = self.ranks.get_mut(q).with_context(|| format!("a placement for decode rank {q}, of none"))?;
        ensure!(epoch == *at + 1, "decode rank {q}'s placement {epoch} after {at}; epochs come one at a time");
        rt.place(m, &placement(hex, q)?).with_context(|| format!("placing decode rank {q}'s pool"))?;
        *at = epoch;
        Ok(())
    }

    /// Issue `push` on `rt`'s compute stream, every copy, once every seat's
    /// mirror has reached the seat's epoch.
    pub fn push(&mut self, rt: &Runtime, push: &Push) -> Result<()> {
        let req = push.req;
        for s in &push.seats {
            let at = self.ranks.get(s.rank).map(|&(_, at)| at);
            ensure!(
                at.is_some_and(|at| at >= s.epoch),
                "request {req}: a seat on decode rank {} at placement {}; this mirror is at {at:?}",
                s.rank,
                s.epoch
            );
        }
        for mv in moves(push, &self.pools, &self.mine, rt.page())? {
            let (Move::Rows { state, rank, .. } | Move::Part { state, rank, .. } | Move::Slot { state, rank, .. }) =
                &mv;
            let what = format!("request {req}: `{state}` to decode rank {rank}");
            let to = |q: usize| {
                self.ranks
                    .get(q)
                    .map(|(m, _)| m)
                    .with_context(|| format!("request {req}: a seat on decode rank {q}, of none"))
            };
            match &mv {
                Move::Rows { state, rank, deal } => rt.push_rows(state, deal, to(*rank)?),
                Move::Part { state, rank, from, into, split, part } => {
                    rt.push_slot_part(state, *from, *into, split, *part, to(*rank)?)
                }
                Move::Slot { state, rank, from, into } => rt.push_slot(state, *from, *into, to(*rank)?),
            }
            .with_context(|| what)?;
        }
        Ok(())
    }
}
