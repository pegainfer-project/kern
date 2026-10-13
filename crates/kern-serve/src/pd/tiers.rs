//! The P head's side of every decode tier at once: which tier a request is
//! on and what follows from each thing the head hears or does, as a pure
//! state machine over one [`Handoffs`] a tier.
//!
//! A request reserves on the tier [`pick`] takes among those that can be
//! asked now, each tier's seats as it last said ([`Load`], on every grant
//! and every step); it stays on that tier until it starts decoding, ends
//! or is released. Everything comes back as `(tier, Out)`, in order, for
//! the head to carry out without deciding anything.

use anyhow::{bail, Result};
use kern_pool::Lease;
use pegainfer_frontend::engine::StopPolicy;

use super::{pick, Ask, Handoffs, Load, Out, Relay, Req, Standing, Tell};

/// What admission does with the request at the front of its queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gate {
    /// Lease it now: handed off to that tier once prefilled, or, with
    /// none, finished on P.
    Lease(Option<usize>),
    /// Leave it at the front: its seats are being asked for, or no tier
    /// can be asked now.
    Wait,
}

pub struct Tiers {
    tiers: Vec<(Handoffs, Load)>,
}

impl Tiers {
    /// `count` decode tiers, each pushed into by `stages` stages.
    pub fn new(count: usize, stages: usize) -> Tiers {
        Tiers { tiers: (0..count).map(|t| (Handoffs::new(stages, t), Load::default())).collect() }
    }

    /// Each tier's load as it last said.
    pub fn loads(&self) -> Vec<Load> {
        self.tiers.iter().map(|(_, l)| *l).collect()
    }

    /// The tier `req` is reserved or underway on, and where it stands there.
    fn of(&self, req: Req) -> Option<(usize, Standing)> {
        self.tiers.iter().map(|(h, _)| h.standing(req)).enumerate().find(|&(_, s)| s != Standing::Unasked)
    }

    /// Requests admitted on P whose clients the tiers answer, every tier's.
    pub fn open(&self) -> Vec<Req> {
        self.tiers.iter().flat_map(|(h, _)| h.open()).collect()
    }

    /// Requests some tier holds seats for or is asked about.
    pub fn live(&self) -> usize {
        self.tiers.iter().map(|(h, _)| h.live()).sum()
    }

    /// Where the request at the front of the queue goes, `tokens` its
    /// prompt and `max_tokens` together. One of more than one token is
    /// handed off when there are tiers, and waits until one grants it seats.
    pub fn gate(
        &mut self,
        req: Req,
        tokens: usize,
        max_tokens: usize,
        stop: &StopPolicy,
    ) -> Result<(Gate, Vec<(usize, Out)>)> {
        let handoff = !self.tiers.is_empty() && Relay::hands_off(max_tokens);
        Ok(match self.of(req).filter(|_| handoff) {
            None if handoff => {
                let seats: Vec<Option<usize>> = self.tiers.iter().map(|(h, l)| h.ready().then_some(l.seats)).collect();
                let outs = match pick(req, &seats) {
                    Some(t) => {
                        let a = self.tiers[t].0.reserve(req, tokens, Relay::new(max_tokens, stop.clone()));
                        vec![(t, Out::Ask(a.expect("a ready tier takes a reserve")))]
                    }
                    None => Vec::new(),
                };
                (Gate::Wait, outs)
            }
            None => (Gate::Lease(None), Vec::new()),
            Some((_, Standing::Reserving)) => (Gate::Wait, Vec::new()),
            Some((t, Standing::Granted)) => (Gate::Lease(Some(t)), Vec::new()),
            Some((t, _)) => bail!("request {req} is underway on decode tier {t} while waiting"),
        })
    }

    /// The granted `req` is admitted on P on `tier`.
    pub fn admitted(&mut self, tier: usize, req: Req) {
        self.tiers[tier].0.admitted(req);
    }

    /// P will never seat the granted `req`: its seats go back.
    pub fn release(&mut self, tier: usize, req: Req) -> Vec<(usize, Out)> {
        on(tier, self.tiers[tier].0.release(req))
    }

    /// The client of `req` left: whatever tier it is on lets it go.
    pub fn abort(&mut self, req: Req) -> Vec<(usize, Out)> {
        match self.of(req) {
            Some((t, _)) => on(t, self.tiers[t].0.abort(req)),
            None => Vec::new(),
        }
    }

    /// `req`'s last rows are on their way through `lease`: its push, if it
    /// is handed off.
    pub fn prefilled(&mut self, req: Req, lease: &Lease, len: usize) -> Vec<(usize, Out)> {
        match self.of(req) {
            Some((t, _)) => on(t, self.tiers[t].0.prefilled(req, lease, len)),
            None => Vec::new(),
        }
    }

    /// Stage `stage`'s push of `req` landed.
    pub fn pushed(&mut self, req: Req, stage: usize) -> Result<Vec<(usize, Out)>> {
        let Some((t, _)) = self.of(req) else { bail!("stage {stage} pushed request {req}, on no decode tier") };
        Ok(on(t, self.tiers[t].0.pushed(req, stage)?))
    }

    /// The first token of a handed-off `req`; nothing once it left.
    pub fn first(&mut self, req: Req, tok: u32) -> Result<Vec<(usize, Out)>> {
        match self.of(req) {
            Some((t, _)) => Ok(on(t, self.tiers[t].0.first(req, tok)?)),
            None => Ok(Vec::new()),
        }
    }

    /// What tier `tier` said, but its tokens.
    pub fn hear(&mut self, tier: usize, t: Tell) -> Result<Vec<(usize, Out)>> {
        let Some((h, l)) = self.tiers.get_mut(tier) else { bail!("decode tier {tier} of {} spoke", self.tiers.len()) };
        if let Tell::Grant { load, .. } = &t {
            *l = *load;
        }
        Ok(on(tier, h.hear(t)?))
    }

    /// Tier `tier`'s load after a step.
    pub fn load(&mut self, tier: usize, load: Load) {
        if let Some((_, l)) = self.tiers.get_mut(tier) {
            *l = load;
        }
    }

    /// The front ended `req`, decoding on `tier`: its seats go back.
    pub fn stop(&self, tier: usize, req: Req) -> Result<(usize, Out)> {
        if tier >= self.tiers.len() {
            bail!("an abort of request {req} on decode tier {tier}, of {}", self.tiers.len());
        }
        Ok((tier, Out::Ask(Ask::Abort { req })))
    }
}

fn on(tier: usize, outs: Vec<Out>) -> Vec<(usize, Out)> {
    outs.into_iter().map(|o| (tier, o)).collect()
}
