//! A prefill stage's view of the decode tier: every D rank's pool mapped
//! into this runtime, and the pushes run into them.
//!
//! Mapping and placing are synchronous and happen at start, or on the
//! issuing thread when a grant says a rank's pool moved (`place` waits for
//! this runtime's stream, so a remap on D costs a stage a drained stream
//! once). A push itself is copies on the compute stream behind the item
//! that wrote the prompt's last rows; whoever issued it marks after it.

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use kern_pool::Placement;
use kern_runtime::{MappedPool, PoolHandle, Runtime};

use super::{moves, Move, Pools, Push, Shape};

pub(crate) struct Mirrors {
    pools: Pools,
    /// This runtime's states the decode tier holds.
    mine: BTreeMap<String, Shape>,
    /// Every D rank's pool, rank order.
    ranks: Vec<MappedPool>,
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
            ranks.push(m);
        }
        let mine = mine.iter().filter(|(n, _)| pools.states.contains_key(*n)).map(|(n, s)| (n.clone(), *s)).collect();
        Ok(Mirrors { pools, mine, ranks })
    }

    /// Issue `push` on `rt`'s compute stream: the moved placements first,
    /// then every copy.
    pub fn push(&mut self, rt: &Runtime, push: &Push) -> Result<()> {
        for (q, p) in &push.placements {
            let m = self.ranks.get_mut(*q).with_context(|| format!("a placement for decode rank {q}, of none"))?;
            rt.place(m, &placement(p, *q)?).with_context(|| format!("placing decode rank {q}'s pool"))?;
        }
        let req = push.req;
        for mv in moves(push, &self.pools, &self.mine, rt.page())? {
            let (Move::Rows { state, rank, .. } | Move::Part { state, rank, .. } | Move::Slot { state, rank, .. }) =
                &mv;
            let what = format!("request {req}: `{state}` to decode rank {rank}");
            let to = |q: usize| {
                self.ranks.get(q).with_context(|| format!("request {req}: a seat on decode rank {q}, of none"))
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
