//! The requests decoding on D, as the front relays their tokens.
//!
//! A request comes here from the [`Handoffs`](super::Handoffs) the moment
//! it starts on D and stays until its client has its last token or is gone.
//! Its tokens are judged against the request's full stop policy as they
//! come: D stops only at `max_tokens`, so a stop seen here first ends the
//! request and frees its seats with an `Abort`. A token for a request no
//! longer here was on its way when the request ended, and is dropped.

use std::collections::BTreeMap;

use anyhow::{bail, Result};
use pegainfer_frontend::engine::{FinishReason, StopCause, StopPolicy};

use super::{Ask, Out, Req};

/// A request's tokens as the client sees them: each one counts toward
/// `max_tokens` and may be its stop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Relay {
    pub(super) max_tokens: usize,
    stop: StopPolicy,
    pub(super) generated: usize,
}

impl Relay {
    pub fn new(max_tokens: usize, stop: StopPolicy) -> Relay {
        Relay { max_tokens, stop, generated: 0 }
    }

    /// Whether a request of `max_tokens` is handed off at all: one token
    /// is the prefill's own, and P finishes it.
    pub fn hands_off(max_tokens: usize) -> bool {
        max_tokens > 1
    }

    /// `tok` generated: how it ends the request, if it does.
    pub(super) fn take(&mut self, tok: u32) -> Option<(FinishReason, Option<StopCause>)> {
        self.generated += 1;
        match self.stop.classify(tok, |_| false) {
            Some(c) => Some((FinishReason::Stop, Some(c))),
            None if self.generated >= self.max_tokens => Some((FinishReason::Length, None)),
            None => None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Relays {
    live: BTreeMap<Req, Relay>,
}

impl Relays {
    pub fn new() -> Relays {
        Relays::default()
    }

    /// `req` started on D with `relay` as its tokens so far.
    pub fn start(&mut self, req: Req, relay: Relay) {
        self.live.insert(req, relay);
    }

    pub fn holds(&self, req: Req) -> bool {
        self.live.contains_key(&req)
    }

    pub fn live(&self) -> usize {
        self.live.len()
    }

    /// The client of `req` went away: its seats are freed.
    pub fn abort(&mut self, req: Req) -> Vec<Out> {
        match self.live.remove(&req) {
            Some(_) => vec![Out::Ask(Ask::Abort { req })],
            None => Vec::new(),
        }
    }

    /// One D step: each running request's token, its last when D says so.
    pub fn hear(&mut self, step: Vec<(Req, u32, bool)>) -> Result<Vec<Out>> {
        let mut out = Vec::new();
        for (req, tok, done) in step {
            let Some(r) = self.live.get_mut(&req) else { continue };
            out.push(Out::Emit { req, tok });
            match (r.take(tok), done) {
                (Some((why, cause)), true) => {
                    self.live.remove(&req);
                    out.push(Out::Finish { req, why, cause });
                }
                (Some((why, cause)), false) => {
                    self.live.remove(&req);
                    out.push(Out::Finish { req, why, cause });
                    out.push(Out::Ask(Ask::Abort { req }));
                }
                (None, true) => {
                    bail!("the decode tier ended request {req} at {} of {} tokens", r.generated, r.max_tokens)
                }
                (None, false) => {}
            }
        }
        Ok(out)
    }
}
