//! What the next item of a pipeline's prefill carries, and when it goes.
//!
//! An item reads a stage's weights once whatever its rows, so a pipeline
//! fed one prompt per item pays a whole weight pass for a prompt's last
//! few hundred rows. Packing fills each item instead: sequences in turn
//! each give their next rows, the last one a partial chunk if that is what
//! fits, until the item has its rows, its sequences or its context (the
//! sum of its sequences' lengths after the call, what a program sized by
//! its context expands). A sequence the context has no room for is passed
//! over, not waited on.
//!
//! The turn is shortest remaining first. A prompt continuing from a long
//! kept prefix has a few thousand rows left where a cold one has its whole
//! length, and a cold 200k prompt holds a pipeline for seconds: oldest
//! first, every hit admitted behind cold prompts waits all of them out.
//! Nothing starves: a sequence passed over by `patience` items goes ahead
//! of every sequence passed over fewer times, and the one passed over most
//! always fits (its own context is below the bound), so a sequence with
//! rows left is in at least one of every `patience + n` items, `n` the
//! sequences waiting.
//!
//! Answers are collected and prompts admitted only between items, and each
//! send waits on the head's device, so how far the head runs ahead of the
//! stages is how late it decides. A full item goes out while at most as
//! many items as stages are in flight (one queued ahead of stage 0), a
//! partial one only while fewer: packing waits for rows only when the
//! stages have work, and a burst of long prompts neither stops the head
//! answering and admitting nor puts the hit arriving behind it after every
//! item the burst could fill.

use std::cmp::Reverse;

/// One sequence waiting to be fed: `pos` rows already sent, `left` still
/// to send, `passed` items gone out without it since it last had rows in one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Waiting {
    pub pos: usize,
    pub left: usize,
    pub passed: usize,
}

/// What one call may hold.
#[derive(Debug, Clone, Copy)]
pub struct Bounds {
    pub rows: usize,
    pub seqs: usize,
    pub context: usize,
}

/// The indices of the sequences with rows left, in their turn: those
/// passed over at least `patience` times, most passed first, then the rest
/// by rows left; ties go to the older (lower index).
pub fn turn(seqs: &[Waiting], patience: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..seqs.len()).filter(|&i| seqs[i].left > 0).collect();
    order.sort_by_key(|&i| match seqs[i] {
        s if s.passed >= patience => (false, Reverse(s.passed), 0, i),
        s => (true, Reverse(0), s.left, i),
    });
    order
}

/// The next item as `(index into seqs, rows)`, in turn; empty when nothing
/// is waiting.
pub fn pack(seqs: &[Waiting], b: Bounds, patience: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let (mut rows, mut context) = (0, 0);
    for i in turn(seqs, patience) {
        if out.len() == b.seqs || rows == b.rows {
            break;
        }
        let s = seqs[i];
        let n = s.left.min(b.rows - rows).min(b.context.saturating_sub(context + s.pos));
        if n > 0 {
            out.push((i, n));
            rows += n;
            context += s.pos + n;
        }
    }
    out
}

/// Each sequence's `passed` once `plan` has gone out: none for the
/// sequences in it, one more for those with rows left it went without.
pub fn passed(seqs: &[Waiting], plan: &[(usize, usize)]) -> Vec<usize> {
    seqs.iter()
        .enumerate()
        .map(|(i, s)| match plan.iter().any(|&(j, _)| j == i) {
            true => 0,
            false if s.left > 0 => s.passed + 1,
            false => s.passed,
        })
        .collect()
}

/// Whether `plan` goes out with `flying` items in a pipeline of `depth`
/// stages: a full item (all its rows or all its sequences) while at most
/// `depth` are in flight, a partial one while fewer.
pub fn goes_out(plan: &[(usize, usize)], b: Bounds, flying: usize, depth: usize) -> bool {
    let rows: usize = plan.iter().map(|&(_, n)| n).sum();
    let full = rows == b.rows || plan.len() == b.seqs;
    !plan.is_empty() && flying < depth + usize::from(full)
}
