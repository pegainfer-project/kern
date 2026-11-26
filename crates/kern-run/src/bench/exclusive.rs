//! What a call adds to its program: from the later of its first kernel's
//! start and its predecessors' end, to its last kernel's end.
//!
//! A call's kernels can start before the call may run: a PDL launch
//! reaches its SMs early and waits in `griddepcontrol.wait`. Summing kernel
//! durations charges that wait to the call, so the more calls an op is
//! split into, the more the per-call times sum past the graph's. The
//! predecessors are the program's own edges: the call before on the same
//! lane, and the last call before on every lane it is `after`. A call that
//! launched nothing passes its predecessors' end on.

use std::collections::BTreeMap;

use kern_manifest::types::Call;

/// Every call's first kernel start and last kernel end in one replay, ns;
/// `None` for a call that launched nothing.
pub(super) type Extent = Option<(u64, u64)>;

/// Per call, in µs, of a replay that began at `start` (ns).
pub(super) fn exclusive(calls: &[Call], extents: &[Extent], start: u64) -> Vec<f64> {
    assert_eq!(calls.len(), extents.len());
    let mut done: Vec<u64> = Vec::with_capacity(calls.len());
    let mut last_on: BTreeMap<u8, usize> = BTreeMap::new();
    calls
        .iter()
        .zip(extents)
        .enumerate()
        .map(|(i, (c, extent))| {
            let ready = std::iter::once(c.lane)
                .chain(c.after.iter().copied())
                .filter_map(|lane| last_on.get(&lane).map(|&j| done[j]))
                .fold(start, u64::max);
            let (us, end) = match *extent {
                Some((first, last)) => (last.saturating_sub(first.max(ready)) as f64 / 1e3, last),
                None => (0., ready),
            };
            done.push(end);
            last_on.insert(c.lane, i);
            us
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(lane: u8, after: &[u8]) -> Call {
        Call { label: None, op: "op".into(), args: Vec::new(), lane, after: after.to_vec() }
    }

    #[test]
    fn a_pdl_wait_is_charged_to_the_call_before() {
        let calls = [call(0, &[]), call(0, &[])];
        let ex = exclusive(&calls, &[Some((0, 10_000)), Some((2_000, 13_000))], 0);
        assert_eq!(ex, [10., 3.]);
        assert_eq!(ex.iter().sum::<f64>(), 13.);
    }

    #[test]
    fn a_gap_between_calls_belongs_to_neither() {
        let calls = [call(0, &[]), call(0, &[])];
        assert_eq!(exclusive(&calls, &[Some((0, 4_000)), Some((6_000, 9_000))], 0), [4., 3.]);
    }

    #[test]
    fn a_side_lane_waits_only_for_what_it_is_after() {
        // fork after lane 0's first call, side work overlapping lane 0's second, join back
        let calls = [call(0, &[]), call(1, &[0]), call(0, &[]), call(0, &[1])];
        let ex =
            exclusive(&calls, &[Some((0, 2_000)), Some((1_000, 7_000)), Some((2_500, 5_000)), Some((3_000, 8_000))], 0);
        assert_eq!(ex, [2., 5., 2.5, 1.]);
    }

    #[test]
    fn a_call_that_launched_nothing_passes_its_predecessor_on() {
        let calls = [call(0, &[]), call(0, &[]), call(0, &[])];
        assert_eq!(exclusive(&calls, &[Some((0, 5_000)), None, Some((1_000, 6_000))], 0), [5., 0., 1.]);
    }

    #[test]
    fn a_kernel_ending_before_it_may_start_adds_nothing() {
        let calls = [call(0, &[]), call(1, &[0])];
        assert_eq!(exclusive(&calls, &[Some((0, 5_000)), Some((100, 900))], 0), [5., 0.]);
    }
}
