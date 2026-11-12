//! When a generated token ends a request.
//!
//! The frontend lowers the request's EOS, `ignore_eos` and explicit stop
//! ids into a [`StopPolicy`] and carries `min_tokens` beside it. vLLM masks
//! stop tokens out of the draw until `min_tokens` tokens are out; a
//! manifest hands back token ids, never logits, so nothing is masked here,
//! and a stop token drawn before then is an ordinary token. The length
//! comes out as vLLM's; the tokens can differ.

use infront_engine::{Request, StopCause, StopPolicy};

/// A request's stop policy, and how many tokens come out before it applies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stop {
    policy: StopPolicy,
    min_tokens: usize,
}

impl Stop {
    pub fn new(policy: StopPolicy, min_tokens: usize) -> Stop {
        Stop { policy, min_tokens }
    }

    pub fn of(r: &Request) -> Stop {
        Stop::new(r.stop_policy.clone(), r.min_tokens)
    }

    /// Why `tok`, the request's generated token `at` (from 0), ends it, if
    /// it does.
    pub fn cause(&self, at: usize, tok: u32) -> Option<StopCause> {
        (at >= self.min_tokens).then(|| self.policy.classify(tok, |_| false)).flatten()
    }
}
