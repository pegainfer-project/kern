//! What the next item of a packed prefill carries: the oldest sequences'
//! next rows, back to back, until the call is full.
//!
//! An item reads a stage's weights once whatever its rows, so a pipeline
//! fed one prompt per item pays a whole weight pass for a prompt's last
//! few hundred rows. Packing fills each item instead: sequences in
//! admission order each give their next rows, the last one a partial chunk
//! if that is what fits, until the item has its rows, its sequences or its
//! context (the sum of its sequences' lengths after the call, what a
//! program sized by its context expands). A sequence never skips ahead of
//! an older one, so no prompt starves behind shorter ones arriving later.

/// One sequence waiting to be fed: `pos` rows already sent, `left` still to send.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Waiting {
    pub pos: usize,
    pub left: usize,
}

/// What one call may hold.
#[derive(Debug, Clone, Copy)]
pub struct Bounds {
    pub rows: usize,
    pub seqs: usize,
    pub context: usize,
}

/// The next item as `(index into seqs, rows)`, oldest first; empty when
/// nothing is waiting or the oldest cannot advance at all.
pub fn pack(seqs: &[Waiting], b: Bounds) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let (mut rows, mut context) = (0, 0);
    for (i, s) in seqs.iter().enumerate().filter(|(_, s)| s.left > 0) {
        let room = b.context.saturating_sub(context + s.pos);
        let n = s.left.min(b.rows - rows).min(room);
        if n == 0 || out.len() == b.seqs {
            break;
        }
        out.push((i, n));
        rows += n;
        context += s.pos + n;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const B: Bounds = Bounds { rows: 8, seqs: 3, context: 20 };

    fn w(pos: usize, left: usize) -> Waiting {
        Waiting { pos, left }
    }

    #[test]
    fn fills_the_call_oldest_first() {
        assert_eq!(pack(&[w(0, 5), w(0, 5), w(0, 5)], B), vec![(0, 5), (1, 3)]);
        assert_eq!(pack(&[w(0, 20)], B), vec![(0, 8)]);
        assert_eq!(pack(&[w(0, 1), w(0, 1), w(0, 1), w(0, 1)], B), vec![(0, 1), (1, 1), (2, 1)]);
        assert_eq!(pack(&[w(4, 0), w(0, 2)], B), vec![(1, 2)]);
        assert_eq!(pack(&[], B), vec![]);
    }

    #[test]
    fn a_long_context_takes_the_room() {
        assert_eq!(pack(&[w(16, 8), w(0, 8)], B), vec![(0, 4)]);
        assert_eq!(pack(&[w(10, 2), w(0, 8)], B), vec![(0, 2), (1, 6)]);
        assert_eq!(pack(&[w(20, 2)], B), vec![]);
    }

    /// Every small input against the rules restated plainly: within bounds,
    /// in order, no sequence skipped while it could still take a row, and
    /// feeding the plan repeatedly sends every row exactly once.
    #[test]
    fn every_small_input_drains_within_bounds() {
        let shapes = (0..4).flat_map(|a| (0..4).flat_map(move |b| (0..10).map(move |c| [a, b, c])));
        for lefts in shapes.flat_map(|l| (0..3).map(move |p| (l, p))) {
            let (lefts, p0) = lefts;
            let mut seqs: Vec<Waiting> =
                lefts.iter().enumerate().map(|(i, &l)| w(if i == 0 { p0 } else { 0 }, l)).collect();
            let total: usize = seqs.iter().map(|s| s.left).sum();
            let mut sent = 0;
            for _ in 0..=total {
                let plan = pack(&seqs, B);
                if plan.is_empty() {
                    break;
                }
                let rows: usize = plan.iter().map(|&(_, n)| n).sum();
                let context: usize = plan.iter().map(|&(i, n)| seqs[i].pos + n).sum();
                assert!(rows <= B.rows && plan.len() <= B.seqs && context <= B.context, "{seqs:?} {plan:?}");
                assert!(plan.windows(2).all(|x| x[0].0 < x[1].0), "{plan:?}");
                let first = seqs.iter().position(|s| s.left > 0).unwrap();
                assert_eq!(plan[0].0, first, "{seqs:?} {plan:?}");
                for &(i, n) in &plan {
                    seqs[i] = w(seqs[i].pos + n, seqs[i].left - n);
                    sent += n;
                }
            }
            assert_eq!((sent, seqs.iter().all(|s| s.left == 0)), (total, true), "{lefts:?} from {p0}");
        }
    }
}
