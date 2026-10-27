//! The head's prefix cache: where a prompt starts and what it leaves.
//!
//! The head's pool is the pipeline's one accounting, so its index of kept
//! prefixes is the pipeline's too: a [`Prefix`] over the head's
//! checkpoints keyed by their tokens, as the single-process scheduler
//! keeps one (`scheduler.rs`, `docs/pool.md`). A prompt continues from the
//! longest entry holding a proper prefix of it — sharing the entry's whole
//! pages and copying the page it ends inside and its state slot — and is
//! fed from there. A prompt whose last rows are out leaves all of itself:
//! its lease retires into the index, its pages and slot becoming the
//! entry's, nothing copied. A recurrent state cannot roll back, so a
//! manifest with one keeps a prompt instead where the frontend says the
//! next turn parts from it (`Request::history_tokens`): the head stops
//! feeding it there for a checkpoint, which copies the page it ends inside
//! and the slot; its end is kept nowhere, so a turn leaves one entry,
//! where every next turn passes whatever reply it renders. A client that
//! retries or edits its last request parts from the turn shortly before
//! that point instead, where no state was kept, so the head also stops a
//! little short of it for a snapshot ([`Cache::snapshot`]), room made for
//! it as for the turn: the turn supersedes it at once, so it is among the
//! first taken when room is wanted.
//!
//! A `Busy` pool makes room and asks again, dropping the entry worth least
//! ([`Prefix`]'s order): a turn its conversation has moved past, else the
//! newest of the conversation idle longest.
//!
//! Every act the pool decides runs on the head's runtime as it is decided,
//! on its compute stream, and is queued for the stages, whose states hold
//! the same page and slot ids for their own layers: the next item carries
//! the queue and each stage runs it before the item's launches, so every
//! stage's stream sees the acts where the head's did. A stage keeps no
//! accounting: its pages and slots are the head's, all of them there from
//! load, and a state slot is taken only while one is free, room made for
//! it otherwise.

use kern_pool::{Checkpoint, Copies, Denied, Lease, Prefix};
use kern_runtime::{Error, Runtime};
use pegainfer_frontend::engine::PrefixCacheCounters;

use crate::ranks::Act;

/// The acts the cache asks of the pool's shell: the head's runtime, which
/// runs their copies on its stream as it decides them, or a bare pool.
pub(super) trait Lessor {
    fn lease(&mut self, tokens: usize) -> Result<Lease, Error>;
    fn lease_from(&mut self, cp: &Checkpoint, len: usize, tokens: usize) -> Result<(Lease, Copies), Error>;
    fn checkpoint(&mut self, lease: &mut Lease, len: usize) -> Result<(Checkpoint, Copies), Error>;
    fn retire(&mut self, lease: Lease, len: usize) -> Checkpoint;
}

impl Lessor for Runtime {
    fn lease(&mut self, tokens: usize) -> Result<Lease, Error> {
        Runtime::lease(self, tokens)
    }

    fn lease_from(&mut self, cp: &Checkpoint, len: usize, tokens: usize) -> Result<(Lease, Copies), Error> {
        Runtime::lease_from(self, cp, len, tokens)
    }

    fn checkpoint(&mut self, lease: &mut Lease, len: usize) -> Result<(Checkpoint, Copies), Error> {
        Runtime::checkpoint(self, lease, len)
    }

    fn retire(&mut self, lease: Lease, len: usize) -> Checkpoint {
        Runtime::retire(self, lease, len)
    }
}

/// What the cache did since the counts were last taken.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(super) struct Counts {
    /// Prompts that continued from an entry, and the tokens they skipped.
    pub hits: u64,
    pub hit_tokens: u64,
    /// Prompts kept where their history ends, and those skipped for want
    /// of room.
    pub kept: u64,
    pub skipped: u64,
    /// Snapshots kept short of that point, and those skipped.
    pub snapshots: u64,
    pub snapshot_skips: u64,
    /// Entries dropped for room.
    pub evictions: u64,
}

pub(super) struct Cache {
    prefix: Prefix,
    /// The acts run on the head and not yet on the stages, in order.
    acts: Vec<Act>,
    counts: Counts,
    /// What the frontend's `/metrics` says of the entries: cumulative,
    /// where `counts` is the log line's window.
    prefix_cache: PrefixCacheCounters,
}

impl Cache {
    /// An empty cache over pages of `unit` tokens.
    pub fn new(unit: usize) -> Cache {
        Cache {
            prefix: Prefix::new(unit),
            acts: Vec::new(),
            counts: Counts::default(),
            prefix_cache: PrefixCacheCounters::default(),
        }
    }

    /// Prompt tokens looked up and found since the cache was made.
    pub fn prefix_cache(&self) -> PrefixCacheCounters {
        self.prefix_cache
    }

    pub fn entries(&self) -> usize {
        self.prefix.entries()
    }

    /// Entries kept that a later turn of their conversation moved past.
    pub fn superseded(&self) -> usize {
        self.prefix.superseded()
    }

    /// The counts since the last call.
    pub fn take_counts(&mut self) -> Counts {
        std::mem::take(&mut self.counts)
    }

    /// The acts decided since the last call, in the order the head ran
    /// them: what the next item carries.
    pub fn take_acts(&mut self) -> Vec<Act> {
        std::mem::take(&mut self.acts)
    }

    /// A lease with room for all of `ids`, continuing from the longest
    /// entry or fresh, and the tokens it starts past. `Busy` makes room
    /// and asks again, until the lease fits or nothing is left to make
    /// room with; every other denial is the caller's.
    pub fn lease(&mut self, rt: &mut impl Lessor, ids: &[i64]) -> Result<(Lease, usize), Error> {
        loop {
            // Room made last time may have dropped the entry hit then.
            let got = match self.prefix.lookup(ids) {
                Some(h) => rt.lease_from(&h.found, h.len, ids.len()).map(|(l, c)| (l, Some(Act::Copy(c.into())))),
                None => rt.lease(ids.len()).map(|l| {
                    let zero = l.seq_slot().map(Act::Zero);
                    (l, zero)
                }),
            };
            match got {
                Ok((lease, act)) => {
                    self.acts.extend(act.filter(|a| !idle(a)));
                    let start = lease.prefix();
                    let c = &mut self.counts;
                    c.hits += u64::from(start > 0);
                    c.hit_tokens += start as u64;
                    let p = &mut self.prefix_cache;
                    p.queries += ids.len() as u64;
                    p.hits += start as u64;
                    return Ok((lease, start));
                }
                Err(Error::Denied(Denied::Busy)) if self.evict() => {}
                Err(e) => return Err(e),
            }
        }
    }

    /// Keep the first `key.len()` tokens of `lease`, a prompt still being
    /// fed, which `key` names. `Busy` makes room as a lease does; with
    /// nothing left to make room with, or any other denial, the prompt goes
    /// on without it.
    pub fn keep(&mut self, rt: &mut impl Lessor, lease: &mut Lease, key: &[i64]) -> Result<(), Error> {
        let kept = self.checkpoint(rt, lease, key)?;
        *match kept {
            true => &mut self.counts.kept,
            false => &mut self.counts.skipped,
        } += 1;
        Ok(())
    }

    /// [`Cache::keep`] short of where the prompt is kept, counted apart.
    pub fn snapshot(&mut self, rt: &mut impl Lessor, lease: &mut Lease, key: &[i64]) -> Result<(), Error> {
        let kept = self.checkpoint(rt, lease, key)?;
        *match kept {
            true => &mut self.counts.snapshots,
            false => &mut self.counts.snapshot_skips,
        } += 1;
        Ok(())
    }

    fn checkpoint(&mut self, rt: &mut impl Lessor, lease: &mut Lease, key: &[i64]) -> Result<bool, Error> {
        loop {
            match rt.checkpoint(lease, key.len()) {
                Ok((cp, copies)) => {
                    self.acts.extend(Some(Act::Copy(copies.into())).filter(|a| !idle(a)));
                    self.prefix.insert(key, cp);
                    return Ok(true);
                }
                Err(Error::Denied(Denied::Busy)) if self.evict() => {}
                Err(Error::Denied(_)) => return Ok(false),
                Err(e) => return Err(e),
            }
        }
    }

    /// A prompt whose last rows are out, all of `key` in its state, as
    /// an entry: its pages and slot become the entry's.
    pub fn retire(&mut self, rt: &mut impl Lessor, lease: Lease, key: &[i64]) {
        let cp = rt.retire(lease, key.len());
        self.prefix.insert(key, cp);
    }

    /// Make room: the entry worth least is dropped; `false` when nothing
    /// is kept.
    fn evict(&mut self) -> bool {
        let dropped = self.prefix.evict().is_some();
        self.counts.evictions += dropped as u64;
        dropped
    }
}

/// An act with nothing to do.
fn idle(a: &Act) -> bool {
    matches!(a, Act::Copy(m) if m.pages.is_empty() && m.slot.is_none())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use kern_manifest::types::Manifest;
    use kern_pool::Pool;

    use super::*;

    /// The bare pool as the head's shell: the same decisions, no device.
    struct Bare {
        pool: Arc<Pool>,
    }

    impl Lessor for Bare {
        fn lease(&mut self, tokens: usize) -> Result<Lease, Error> {
            self.pool.lease(tokens).map_err(Error::Denied)
        }

        fn lease_from(&mut self, cp: &Checkpoint, len: usize, tokens: usize) -> Result<(Lease, Copies), Error> {
            self.pool.restore(cp, len, tokens).map_err(Error::Denied)
        }

        fn checkpoint(&mut self, lease: &mut Lease, len: usize) -> Result<(Checkpoint, Copies), Error> {
            self.pool.checkpoint(lease, len).map_err(Error::Denied)
        }

        fn retire(&mut self, lease: Lease, len: usize) -> Checkpoint {
            self.pool.retire(lease, len)
        }
    }

    /// The device copies among the acts since the last call, pages and
    /// slots, in order.
    fn copies(c: &mut Cache) -> (Vec<(i32, i32)>, Vec<(i32, i32)>) {
        c.take_acts().into_iter().fold((vec![], vec![]), |(mut p, mut s), a| {
            if let Act::Copy(m) = a {
                p.extend(m.pages);
                s.extend(m.slot);
            }
            (p, s)
        })
    }

    /// kv paged in 4 tokens, a row of 16 pages; with `rec`, also a
    /// recurrent state of one 8-byte line per sequence.
    fn manifest(rec: bool) -> Manifest {
        let (state, table) = match rec {
            true => (
                r#", "rec": {"bytes_per_seq": 8}"#,
                r#", "line_index": {"kind": "input", "dtype": "i32", "shape": [1, "seqs"], "domain": {"index_into": "rec", "stride": 8}}"#,
            ),
            false => ("", ""),
        };
        Manifest::from_json(&format!(
            r#"{{
            "schema_version": 5, "model": "t", "vars": {{"tokens": {{"max": 8}}, "seqs": {{"max": 2}}}},
            "states": {{"kv": {{"bytes_per_token": 1}}{state}}},
            "buffers": {{
                "block_table": {{"kind": "input", "dtype": "i32", "shape": ["seqs", 16], "domain": {{"index_into": "kv", "stride": 4}}}}{table}
            }},
            "modules": {{}}, "ops": {{}}, "programs": {{}}
        }}"#
        ))
        .expect("fixture parses")
    }

    /// `pages` pages of 4 tokens and, with a recurrent state, `slots`
    /// slots (slot 0 among them).
    fn pool(pages: usize, slots: usize) -> Bare {
        Bare { pool: Arc::new(Pool::new(&manifest(slots > 0), pages, slots).unwrap()) }
    }

    fn prompt(seed: i64, n: usize) -> Vec<i64> {
        (0..n as i64).map(|i| seed * 1000 + i).collect()
    }

    #[test]
    fn a_prompt_continues_from_what_the_one_before_left() {
        let mut rt = pool(16, 0);
        let mut c = Cache::new(4);
        let p1 = prompt(1, 10);
        let (l1, start) = c.lease(&mut rt, &p1).unwrap();
        assert_eq!((start, l1.page_ids(), copies(&mut c)), (0, &[0, 1, 2][..], (vec![], vec![])));
        c.retire(&mut rt, l1, &p1);
        // The same prompt again: its last token is never covered, so it
        // continues at the whole pages it shares, nothing copied.
        let (again, start) = c.lease(&mut rt, &p1).unwrap();
        assert_eq!((start, again.page_ids(), copies(&mut c)), (8, &[0, 1, 3][..], (vec![], vec![])));
        drop(again);
        // A second turn covers the whole of the first: the page the first
        // ends inside is copied for the new lease, which writes on past it.
        let p2: Vec<i64> = p1.iter().copied().chain(prompt(2, 6)).collect();
        let (l2, start) = c.lease(&mut rt, &p2).unwrap();
        assert_eq!((start, l2.page_ids(), copies(&mut c)), (10, &[0, 1, 3, 4][..], (vec![(2, 3)], vec![])));
        assert_eq!(c.take_counts(), Counts { hits: 2, hit_tokens: 18, ..Counts::default() });
    }

    #[test]
    fn a_full_pool_drops_the_coldest_entry() {
        // Four pages: a 6-token prompt takes two.
        let mut rt = pool(4, 0);
        let mut c = Cache::new(4);
        for seed in 1..=2 {
            let p = prompt(seed, 6);
            let (l, _) = c.lease(&mut rt, &p).unwrap();
            c.retire(&mut rt, l, &p);
        }
        assert_eq!((c.entries(), rt.pool.used()), (2, 4));
        // A third has no room until the coldest, the first, goes.
        let p3 = prompt(3, 6);
        let (l3, start) = c.lease(&mut rt, &p3).unwrap();
        assert_eq!((start, l3.page_ids(), c.entries()), (0, &[0, 1][..], 1));
        c.retire(&mut rt, l3, &p3);
        // The second again: the lookup makes it the warmest, so the third
        // goes for the page its lease needs past the shared one.
        let (l2, start) = c.lease(&mut rt, &prompt(2, 6)).unwrap();
        assert_eq!((start, l2.page_ids(), c.entries()), (4, &[2, 0][..], 1));
        // A lease held by a prompt is not the cache's to drop: with the
        // entries gone, a prompt that does not fit waits.
        assert!(matches!(c.lease(&mut rt, &prompt(4, 16)), Err(Error::Denied(Denied::Busy))));
        assert_eq!((c.entries(), c.take_counts().evictions), (0, 3));
    }

    #[test]
    fn a_recurrent_state_is_kept_where_the_history_ends_and_a_full_pool_waits() {
        // Slot 0 and two slots to lease or keep.
        let mut rt = pool(16, 3);
        let mut c = Cache::new(4);
        let p1 = prompt(1, 10);
        let (mut l1, _) = c.lease(&mut rt, &p1).unwrap();
        // The history ends at 6: the page it ends inside and the slot are
        // copied into the entry's own.
        c.keep(&mut rt, &mut l1, &p1[..6]).unwrap();
        assert_eq!((l1.seq_slot(), copies(&mut c)), (Some(1), (vec![(1, 3)], vec![(1, 2)])));
        // Its end is not kept: l1 goes back to the pool. Another prompt
        // left whole takes the freed slot, so both slots are held.
        drop(l1);
        let p0 = prompt(3, 5);
        let (l0, _) = c.lease(&mut rt, &p0).unwrap();
        c.retire(&mut rt, l0, &p0);
        // The next turn parts from p1 where its history ends and continues
        // from that entry, which the lookup makes the warmer: its slot comes
        // from dropping the other conversation.
        let p2: Vec<i64> = p1[..6].iter().copied().chain(prompt(2, 5)).collect();
        let (l2, start) = c.lease(&mut rt, &p2).unwrap();
        assert_eq!((start, l2.seq_slot(), copies(&mut c).1), (6, Some(1), vec![(2, 1)]));
        assert_eq!(c.take_counts(), Counts { hits: 1, hit_tokens: 6, kept: 1, evictions: 1, ..Counts::default() });
        // Both slots held: the pool alone has nothing to lease with.
        assert_eq!(Pool::lease(&rt.pool, 4).err(), Some(Denied::Busy));
    }

    #[test]
    fn room_takes_the_turns_a_conversation_moved_past_before_any_conversation() {
        // Five slots besides slot 0: a lease and four entries.
        let mut rt = pool(32, 6);
        let mut c = Cache::new(4);
        let mut history: Vec<Vec<i64>> = vec![Vec::new(); 2];
        let mut turn = |c: &mut Cache, rt: &mut Bare, s: usize, n: i64| {
            // The history so far, the turn's message, and a tail the next
            // turn renders otherwise: kept where the history ends, the
            // lease then back to the pool.
            let p: Vec<i64> = history[s].iter().copied().chain(prompt(n, 4)).chain([-1, -2]).collect();
            let (mut l, start) = c.lease(rt, &p).unwrap();
            c.keep(rt, &mut l, &p[..p.len() - 2]).unwrap();
            drop(l);
            let hit = start == history[s].len();
            history[s] = p[..p.len() - 2].to_vec();
            hit
        };
        // Conversation 0 takes three turns, conversation 1 one: the slots
        // hold 0's first two turns, superseded, and the newest of each.
        let hits: Vec<bool> = [(0, 1), (0, 2), (0, 3), (1, 4)].map(|(s, n)| turn(&mut c, &mut rt, s, n)).into();
        assert_eq!((hits, c.entries(), c.superseded()), (vec![true, true, true, true], 4, 2));
        // Two more turns each: every one hits, and room came from turns
        // moved past, never from a conversation's newest.
        let hits: Vec<bool> = [(1, 5), (0, 6), (1, 7), (0, 8)].map(|(s, n)| turn(&mut c, &mut rt, s, n)).into();
        assert_eq!((hits, c.entries() - c.superseded()), (vec![true; 4], 2));
        assert!(c.take_counts().evictions > 0);
    }

    #[test]
    fn a_snapshot_short_of_the_history_serves_a_rewind_and_goes_first_for_room() {
        // Slot 0 and three slots: the lease's, the snapshot's, the history's.
        let mut rt = pool(16, 4);
        let mut c = Cache::new(4);
        let p1 = prompt(1, 12);
        let (mut l1, _) = c.lease(&mut rt, &p1).unwrap();
        c.snapshot(&mut rt, &mut l1, &p1[..4]).unwrap();
        c.keep(&mut rt, &mut l1, &p1[..10]).unwrap();
        drop(l1);
        let n = c.take_counts();
        assert_eq!((n.snapshots, n.kept, n.evictions, c.entries(), c.superseded()), (1, 1, 0, 2, 1));
        // A retry that rewrites the turn from 7 on parts from the history
        // before its end: the snapshot is where it continues.
        let retry: Vec<i64> = p1[..7].iter().copied().chain(prompt(2, 5)).collect();
        let (mut l2, start) = c.lease(&mut rt, &retry).unwrap();
        assert_eq!(start, 4);
        // Its lease, the snapshot and the history hold every slot: the
        // retry's snapshot takes the superseded one's, the history stays.
        c.snapshot(&mut rt, &mut l2, &retry[..8]).unwrap();
        let n = c.take_counts();
        assert_eq!((n.snapshots, n.snapshot_skips, n.evictions, c.entries(), c.superseded()), (1, 0, 1, 2, 0));
        drop(l2);
        assert_eq!(c.lease(&mut rt, &p1).unwrap().1, 10);
    }

    /// The bytes a stage holds, as the acts and the items leave them: each
    /// page a token per position, each slot the tokens its recurrent state
    /// has taken in.
    struct Stage {
        unit: usize,
        pages: Vec<Vec<i64>>,
        slots: Vec<Vec<i64>>,
    }

    impl Stage {
        fn new(unit: usize, pages: usize, slots: usize) -> Stage {
            Stage { unit, pages: vec![vec![-9; unit]; pages], slots: vec![vec![-9]; slots] }
        }

        fn run(&mut self, acts: Vec<Act>) {
            for a in acts {
                match a {
                    Act::Zero(s) => self.slots[s as usize].clear(),
                    Act::Copy(m) => {
                        m.pages.iter().for_each(|&(a, b)| self.pages[b as usize] = self.pages[a as usize].clone());
                        m.slot.iter().for_each(|&(a, b)| self.slots[b as usize] = self.slots[a as usize].clone());
                    }
                }
            }
        }

        /// An item feeding `rows` of `lease` from `pos`.
        fn feed(&mut self, lease: &Lease, pos: usize, rows: &[i64]) {
            for (i, &t) in rows.iter().enumerate() {
                let p = pos + i;
                self.pages[lease.page_ids()[p / self.unit] as usize][p % self.unit] = t;
            }
            if let Some(s) = lease.seq_slot() {
                self.slots[s as usize].extend(rows);
            }
        }

        /// What `lease` reads for its first `n` positions, and its state.
        fn read(&self, lease: &Lease, n: usize) -> (Vec<i64>, Option<Vec<i64>>) {
            let ids = lease.page_ids();
            let toks = (0..n).map(|p| self.pages[ids[p / self.unit] as usize][p % self.unit]).collect();
            (toks, lease.seq_slot().map(|s| self.slots[s as usize].clone()))
        }
    }

    struct Rng(u64);

    impl Rng {
        fn below(&mut self, n: usize) -> usize {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 % n as u64) as usize
        }
    }

    /// Conversations taking turns, a few prompts fed at once in random
    /// chunks, through a pool small enough that room is made all the time:
    /// a stage that runs only the acts and the items, in order, holds for
    /// every prompt that continues from an entry exactly its tokens before
    /// the start, and its prompt's tokens before its start in its state.
    #[test]
    fn a_stage_running_the_acts_in_order_reads_every_prompt_it_continues() {
        for (rec, seed) in [(true, 1), (true, 3), (false, 4)] {
            let slots = if rec { 6 } else { 0 };
            let mut rt = pool(28, slots);
            let mut st = Stage::new(4, rt.pool.pages(), slots);
            let mut c = Cache::new(4);
            let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ seed);
            // Every turn so far of each conversation, the newest last.
            let mut turns: Vec<Vec<Vec<i64>>> = vec![vec![vec![]]; 6];
            struct Fed {
                conv: usize,
                lease: Lease,
                ids: Vec<i64>,
                pos: usize,
                mark: usize,
            }
            let mut active: Vec<Fed> = Vec::new();
            let mut total = Counts::default();
            let mut word = 0i64;
            for _ in 0..20000 {
                let idle: Vec<usize> = (0..turns.len()).filter(|&v| active.iter().all(|f| f.conv != v)).collect();
                if active.len() < 3 && !idle.is_empty() && rng.below(3) == 0 {
                    let conv = idle[rng.below(idle.len())];
                    // Mostly the newest turn, now and then an older one.
                    let t = &turns[conv];
                    let from = if rng.below(8) == 0 { rng.below(t.len()) } else { t.len() - 1 };
                    let mut ids = t[from].clone();
                    let mark = (ids.len() + 1 + rng.below(9)).min(60);
                    while ids.len() < mark {
                        word += 1;
                        ids.push(word);
                    }
                    ids.extend([-1, -2]);
                    match c.lease(&mut rt, &ids) {
                        Ok((lease, start)) => {
                            st.run(c.take_acts());
                            let (toks, state) = st.read(&lease, start);
                            assert_eq!(toks, ids[..start], "pages of a prompt starting at {start}");
                            if rec {
                                assert_eq!(state.as_deref(), Some(&ids[..start]), "state at {start}");
                            }
                            active.push(Fed { conv, lease, ids, pos: start, mark });
                        }
                        Err(Error::Denied(Denied::Busy)) => {}
                        Err(e) => panic!("{e}"),
                    }
                } else if !active.is_empty() {
                    let i = rng.below(active.len());
                    let f = &mut active[i];
                    let upto = if f.pos < f.mark { f.mark } else { f.ids.len() };
                    let n = (1 + rng.below(6)).min(upto - f.pos);
                    st.feed(&f.lease, f.pos, &f.ids[f.pos..f.pos + n]);
                    f.pos += n;
                    if f.pos == f.mark && rec {
                        c.keep(&mut rt, &mut f.lease, &f.ids[..f.mark]).unwrap();
                        st.run(c.take_acts());
                    }
                    if f.pos == f.ids.len() {
                        let f = active.swap_remove(i);
                        turns[f.conv].push(f.ids[..f.mark].to_vec());
                        match rec {
                            true => drop(f.lease),
                            false => c.retire(&mut rt, f.lease, &f.ids),
                        }
                    }
                }
                let n = c.take_counts();
                total = Counts { hits: total.hits + n.hits, evictions: total.evictions + n.evictions, ..total };
            }
            assert!(total.hits > 20 && total.evictions > 0, "{seed}: {total:?}");
        }
    }
}
