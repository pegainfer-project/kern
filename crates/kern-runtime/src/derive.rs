//! Derivation: weights that are a function of checkpoint tensors rather
//! than the tensors themselves (an expert stack shuffled into a GEMM's
//! layout), computed while the weights load.
//!
//! A `source` buffer is checkpoint bytes no serving program reads; a
//! `derive` program turns sources into the carries the other programs
//! read. Device memory holds a source only while derivation needs it: the
//! derive programs run call by call, in program-name order, and each
//! source occupies a window of one shared staging allocation from the
//! call that first reads it to the call that last does. Two sources
//! alive at the same call never share bytes; a source whose last reader
//! has run gives its bytes to the next one. The peak is the derived
//! weights plus the widest set of sources one call needs at once, never
//! the whole checkpoint twice.
//!
//! [`layout`] decides the windows and is pure; the runtime allocates the
//! staging, compiles the derive programs against it, and in
//! [`Runtime::load_weights`] uploads each source just before its first
//! reader. After the last call the staging is freed and the derive
//! programs are dropped with it: nothing can run them again.

use std::collections::BTreeMap;

use kern_manifest::types::{Arg, BufferKind, Manifest};

use crate::compile::{CompiledProgram, Dense};
use crate::device::DeviceBuf;
use crate::error::Error;
use crate::load::Lanes;
use crate::weights::Tensors;
use crate::{Result, Runtime};

/// Alignment of every source in the staging allocation.
const ALIGN: u64 = 256;

/// Where each source sits in the staging allocation and when it arrives.
pub(crate) struct Plan {
    /// Byte offset of every source.
    pub(crate) at: BTreeMap<String, u64>,
    /// Bytes of the staging allocation.
    pub(crate) bytes: u64,
    /// Per derive call, in run order: the sources it reads first.
    pub(crate) arrivals: Vec<Vec<String>>,
}

/// What is left to derive once the weights are in: the derive programs
/// lowered against the staging, and the staging itself.
pub(crate) struct Derivation {
    pub(crate) programs: Vec<CompiledProgram>,
    pub(crate) arrivals: Vec<Vec<String>>,
    pub(crate) stage: DeviceBuf,
}

impl Runtime {
    /// Run the derive programs call by call, each source uploaded just
    /// before its first reader, then free the staging and forget the
    /// sources. The stream is drained before a source lands, so no call
    /// still reads the bytes it takes over.
    pub(crate) fn derive(&mut self, d: Derivation, tensors: &dyn Tensors, stage: &Lanes) -> Result<u64> {
        let vars = Dense::check(&self.manifest, &BTreeMap::new(), &vec![false; self.manifest.vars.len()])?;
        let calls = d.programs.iter().flat_map(|p| p.call_ranges.iter().map(move |&(lo, hi)| &p.launches[lo..hi]));
        let mut uploaded = 0;
        for (launches, arriving) in calls.zip(&d.arrivals) {
            if !arriving.is_empty() {
                self.stream.synchronize()?;
                uploaded += self.fill_sources(arriving, tensors, stage)?;
            }
            for l in launches {
                self.launch(l, &vars).map_err(|e| Error::Call { context: l.ctx.clone(), source: Box::new(e) })?;
            }
        }
        self.stream.synchronize()?;
        for s in d.arrivals.iter().flatten() {
            self.buffers.remove(s);
        }
        drop(d.stage);
        Ok(uploaded)
    }
}

/// The derive programs' names, in the order they run.
pub(crate) fn programs(m: &Manifest) -> Vec<&str> {
    m.programs.iter().filter(|(_, p)| p.derive).map(|(n, _)| n.as_str()).collect()
}

/// The staging plan for a manifest whose sources are `bytes` long.
pub(crate) fn plan(m: &Manifest, bytes: &BTreeMap<String, u64>) -> Plan {
    let calls: Vec<_> = programs(m).into_iter().flat_map(|p| &m.programs[p].calls).collect();
    let mut live: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    for (k, c) in calls.iter().enumerate() {
        for a in &c.args {
            let Arg::Buf { buf, .. } = a else { continue };
            if m.buffers.get(buf).is_some_and(|b| b.kind == BufferKind::Source) {
                live.entry(buf.as_str()).and_modify(|(_, last)| *last = k).or_insert((k, k));
            }
        }
    }
    let windows: Vec<Window> =
        live.iter().map(|(n, &(first, last))| Window { first, last, bytes: bytes[*n] }).collect();
    let (at, total) = layout(&windows);
    let mut arrivals = vec![Vec::new(); calls.len()];
    live.iter().for_each(|(n, (first, _))| arrivals[*first].push(n.to_string()));
    Plan { at: live.keys().map(|n| n.to_string()).zip(at).collect(), bytes: total, arrivals }
}

/// A source's lifetime in derive calls, `first..=last`, and its size.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Window {
    first: usize,
    last: usize,
    bytes: u64,
}

/// First fit in arrival order: each window takes the lowest aligned gap
/// among the windows still alive when it arrives. Returns every window's
/// offset, in input order, and the bytes the layout spans.
pub(crate) fn layout(windows: &[Window]) -> (Vec<u64>, u64) {
    let mut order: Vec<usize> = (0..windows.len()).collect();
    order.sort_by_key(|&i| (windows[i].first, i));
    let mut at = vec![0u64; windows.len()];
    let mut placed: Vec<usize> = Vec::new();
    for i in order {
        let w = windows[i];
        let mut taken: Vec<(u64, u64)> = placed
            .iter()
            .filter(|&&j| windows[j].last >= w.first)
            .map(|&j| (at[j], at[j] + windows[j].bytes))
            .collect();
        taken.sort();
        let mut lo = 0;
        for (a, b) in taken {
            if lo + w.bytes <= a {
                break;
            }
            lo = lo.max(b.next_multiple_of(ALIGN));
        }
        at[i] = lo;
        placed.push(i);
    }
    let total = windows.iter().zip(&at).map(|(w, a)| a + w.bytes).max().unwrap_or(0);
    (at, total)
}

#[cfg(test)]
mod tests {
    use super::{layout, Window, ALIGN};

    /// Windows alive at one call never overlap, every offset is aligned,
    /// and the span is the sum of sizes at most, over random lifetimes.
    #[test]
    fn live_windows_are_disjoint() {
        let mut s = 0x2545_f491_4f6c_dd1du64;
        let mut next = |n: u64| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s % n
        };
        for _ in 0..2000 {
            let windows: Vec<Window> = (0..next(12))
                .map(|_| {
                    let first = next(10) as usize;
                    Window { first, last: first + next(4) as usize, bytes: 1 + next(5000) }
                })
                .collect();
            let (at, total) = layout(&windows);
            for (i, (a, w)) in at.iter().zip(&windows).enumerate() {
                assert_eq!((a % ALIGN, a + w.bytes <= total), (0, true));
                for (b, v) in at.iter().zip(&windows).skip(i + 1) {
                    let alive_together = w.first <= v.last && v.first <= w.last;
                    let apart = a + w.bytes <= *b || b + v.bytes <= *a;
                    assert!(!alive_together || apart, "{w:?} at {a} and {v:?} at {b}");
                }
            }
            let sum: u64 = windows.iter().map(|w| w.bytes.next_multiple_of(ALIGN)).sum();
            assert!(total <= sum);
        }
    }

    /// A chain of sources each read by one call reuses one window.
    #[test]
    fn a_chain_reuses_one_window() {
        let windows: Vec<Window> = (0..5).map(|k| Window { first: k, last: k, bytes: 1000 + k as u64 }).collect();
        assert_eq!(layout(&windows), (vec![0; 5], 1004));
    }
}
