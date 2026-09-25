//! A hosted manifest without its host: the runtime allocates the host
//! states itself.
//!
//! A host state is memory the host allocates in a declared layout, and the
//! inputs indexing it say how: a page table over it (`stride` tokens per
//! block) or a line table (`stride` bytes per line, one line per block).
//! Every host state of one layout shares the ids the tables hand out, as a
//! serving engine's layers share its block ids. [`Verified::self_hosted`]
//! turns each host state into the owned state of the same layout, a paged
//! one of block bytes / tokens per block per token or a per-sequence one of
//! a block per sequence, so the runtime's pool provisions it and the
//! programs run unchanged over the same strides. A tool that is its own
//! host (`kern bench`) runs the manifest a serving engine runs this way.

use std::collections::BTreeSet;

use crate::types::{BufferKind, HostTensor, State};
use crate::verify::{verify, Verified, VerifyErrors};

/// How the tables index one layout of host state.
enum Indexed {
    Tokens(u64),
    Lines,
}

impl Verified {
    /// This manifest with every host state owned by the runtime in the
    /// same layout; itself when it has none. A host state no table indexes,
    /// or one indexed both per token and per line, has no owned
    /// equivalent.
    pub fn self_hosted(&self) -> Result<Verified, VerifyErrors> {
        let layouts: Vec<&HostTensor> = self.states.values().filter_map(|s| s.host.as_ref()).collect();
        if layouts.is_empty() {
            return Ok(self.clone());
        }
        let strides = |h: &HostTensor| -> BTreeSet<u64> {
            self.buffers
                .values()
                .filter(|b| b.kind == BufferKind::Input)
                .filter_map(|b| b.domain.as_ref())
                .filter(|d| {
                    d.index_into.as_deref().and_then(|t| self.states.get(t)).and_then(|s| s.host.as_ref()) == Some(h)
                })
                .map(|d| d.stride)
                .collect()
        };
        let indexed = |h: &HostTensor| -> Result<Indexed, String> {
            let block = h.strides.first().copied().unwrap_or(0) * h.dtype.bytes();
            let s = strides(h);
            match (s.iter().max(), s.contains(&block)) {
                (None, _) => Err(format!("no table indexes host layout {}", h.describe())),
                (Some(_), true) if s.len() == 1 => Ok(Indexed::Lines),
                (Some(_), true) => Err(format!("host layout {} is indexed both per line and per token", h.describe())),
                (Some(&k), false) if block.is_multiple_of(k) && s.iter().all(|&x| k.is_multiple_of(x)) => {
                    Ok(Indexed::Tokens(k))
                }
                (Some(_), false) => Err(format!(
                    "host layout {}: tables of {s:?} tokens do not tile its {block}-byte blocks",
                    h.describe()
                )),
            }
        };
        let mut m = (**self).clone();
        let mut errs = Vec::new();
        for (name, st) in m.states.iter_mut() {
            let Some(h) = st.host.take() else { continue };
            let block = h.strides[0] * h.dtype.bytes();
            *st = match indexed(&h) {
                Ok(Indexed::Tokens(k)) => State { bytes_per_token: block / k, ..State::default() },
                Ok(Indexed::Lines) => State { bytes_per_seq: block, ..State::default() },
                Err(e) => {
                    errs.push(format!("state `{name}`: {e}"));
                    continue;
                }
            };
        }
        match errs.is_empty() {
            true => verify(m),
            false => Err(VerifyErrors(errs)),
        }
    }
}
