//! Physical chunks and where they are mapped.
//!
//! A pooled state — every paged state and every per-sequence state — is an
//! address range reserved once and never moved; what backs it is a pool of
//! physical chunks all pooled states share. A page is an interval of chunk
//! positions in every paged state's range, a slot one in every
//! per-sequence state's; an object exists (can be handed out) only while
//! every position it covers is mapped. Chunks stay where they were last
//! used — a freed page keeps its chunks as a page — and only when one kind
//! runs dry does the pool take chunks from free objects of the other kind.
//! That move is a [`Remap`]: unmaps and maps in chunk numbers that the
//! runtime executes off the serving thread; until it lands, the objects it
//! makes are not free and the ones it unmakes are gone.
//!
//! Positions are shared where an object boundary falls inside a chunk: a
//! position counts the objects that exist over it, is mapped when the
//! first arrives and unmapped when the last leaves.

use std::ops::Range;

/// What an arena's objects are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Page,
    Slot,
}

/// A pooled state's reserved range, in chunk positions.
#[derive(Debug, Clone)]
struct Arena {
    kind: Kind,
    /// Bytes per object.
    object: u64,
    /// The chunk mapped at each position.
    chunk: Vec<Option<u32>>,
    /// Objects existing over each position.
    users: Vec<u16>,
}

/// Unmaps, then maps, then access grants, in chunk positions; what the
/// plan makes and unmakes.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Remap {
    /// (arena, position): the chunk there comes off.
    pub unmap: Vec<(usize, usize)>,
    /// (arena, position, chunk): the chunk goes on there.
    pub map: Vec<(usize, usize, u32)>,
    /// (arena, positions): grant access over a whole object once mapped.
    pub access: Vec<(usize, Range<usize>)>,
    pub made: Vec<(Kind, i32)>,
    pub unmade: Vec<(Kind, i32)>,
}

impl Remap {
    pub fn is_empty(&self) -> bool {
        self.unmap.is_empty() && self.map.is_empty()
    }
}

/// The chunk pool: `total` chunks of `chunk` bytes, each mapped at one
/// position of one arena or free.
#[derive(Debug, Clone)]
pub struct Chunks {
    chunk: u64,
    arenas: Vec<Arena>,
    free: Vec<u32>,
}

impl Chunks {
    /// `total` chunks over arenas of `(kind, bytes per object, objects)`;
    /// every chunk starts free.
    pub fn new(chunk: u64, arenas: &[(Kind, u64, usize)], total: u32) -> Chunks {
        assert!(chunk >= 1);
        let arenas = arenas
            .iter()
            .map(|&(kind, object, objects)| {
                let positions = (object * objects as u64).div_ceil(chunk) as usize;
                Arena { kind, object, chunk: vec![None; positions], users: vec![0; positions] }
            })
            .collect();
        Chunks { chunk, arenas, free: (0..total).rev().collect() }
    }

    pub fn chunk(&self) -> u64 {
        self.chunk
    }

    pub fn free(&self) -> usize {
        self.free.len()
    }

    /// Positions of each arena.
    #[cfg(test)]
    fn positions(&self) -> Vec<usize> {
        self.arenas.iter().map(|a| a.chunk.len()).collect()
    }

    /// Positions `object` of arena `a` covers.
    pub fn interval(&self, a: usize, object: usize) -> Range<usize> {
        let ar = &self.arenas[a];
        let lo = ar.object * object as u64;
        (lo / self.chunk) as usize..((lo + ar.object).div_ceil(self.chunk)) as usize
    }

    fn arenas_of(&self, kind: Kind) -> impl Iterator<Item = usize> + '_ {
        self.arenas.iter().enumerate().filter(move |(_, a)| a.kind == kind).map(|(i, _)| i)
    }

    /// Chunks making `object` of `kind` takes: its positions nobody
    /// exists over yet.
    pub fn cost(&self, kind: Kind, object: usize) -> usize {
        self.arenas_of(kind).map(|a| self.interval(a, object).filter(|&p| self.arenas[a].users[p] == 0).count()).sum()
    }

    /// Bring `object` of `kind` into existence, taking chunks from the
    /// free ones; the caller has checked [`Chunks::cost`] against
    /// [`Chunks::free`].
    pub fn make(&mut self, kind: Kind, object: usize, plan: &mut Remap) {
        for a in self.arenas_of(kind).collect::<Vec<_>>() {
            let range = self.interval(a, object);
            for p in range.clone() {
                let ar = &mut self.arenas[a];
                if ar.users[p] == 0 {
                    let c = self.free.pop().expect("a free chunk for every uncovered position");
                    ar.chunk[p] = Some(c);
                    plan.map.push((a, p, c));
                }
                ar.users[p] += 1;
            }
            plan.access.push((a, range));
        }
        plan.made.push((kind, object as i32));
    }

    /// Take `object` of `kind` out of existence; positions it alone
    /// covered come off and their chunks are free again (the plan unmaps
    /// before it maps, so a later `make` in the same plan may reuse them).
    pub fn unmake(&mut self, kind: Kind, object: usize, plan: &mut Remap) {
        for a in self.arenas_of(kind).collect::<Vec<_>>() {
            for p in self.interval(a, object) {
                let ar = &mut self.arenas[a];
                ar.users[p] -= 1;
                if ar.users[p] == 0 {
                    let c = ar.chunk[p].take().expect("an existing object is mapped");
                    self.free.push(c);
                    plan.unmap.push((a, p));
                }
            }
        }
        plan.unmade.push((kind, object as i32));
    }

    /// Every mapped (arena, position, chunk).
    #[cfg(test)]
    pub(crate) fn mapped(&self) -> Vec<(usize, usize, u32)> {
        self.arenas
            .iter()
            .enumerate()
            .flat_map(|(a, ar)| ar.chunk.iter().enumerate().filter_map(move |(p, c)| c.map(|c| (a, p, c))))
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn users(&self, a: usize) -> &[u16] {
        &self.arenas[a].users
    }

    #[cfg(test)]
    pub(crate) fn free_ids(&self) -> &[u32] {
        &self.free
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pages of 16 bytes and slots of 24 over 12-byte chunks: pages share
    /// every other boundary chunk, slots every one.
    fn chunks() -> Chunks {
        Chunks::new(12, &[(Kind::Page, 16, 6), (Kind::Slot, 24, 3)], 20)
    }

    #[test]
    fn intervals_share_boundary_chunks() {
        let c = chunks();
        assert_eq!(c.positions(), [8, 6]);
        assert_eq!((c.interval(0, 0), c.interval(0, 1), c.interval(0, 2)), (0..2, 1..3, 2..4));
        assert_eq!((c.interval(1, 0), c.interval(1, 1)), (0..2, 2..4));
        assert_eq!((c.cost(Kind::Page, 0), c.cost(Kind::Slot, 0)), (2, 2));
    }

    #[test]
    fn make_and_unmake_count_users() {
        let mut c = chunks();
        let mut plan = Remap::default();
        c.make(Kind::Page, 0, &mut plan);
        c.make(Kind::Page, 1, &mut plan);
        // Position 1 is shared: mapped once, used twice.
        assert_eq!((plan.map.len(), &c.users(0)[..4], c.free()), (3, &[1, 2, 1, 0][..], 17));
        assert_eq!(plan.access, [(0, 0..2), (0, 1..3)]);
        assert_eq!(c.cost(Kind::Page, 2), 1);
        let mut plan = Remap::default();
        c.unmake(Kind::Page, 0, &mut plan);
        // Only position 0 comes off; page 1 still exists over position 1.
        assert_eq!((plan.unmap, &c.users(0)[..3], c.free()), (vec![(0, 0)], &[0, 1, 1][..], 18));
        assert_eq!(plan.unmade, [(Kind::Page, 0)]);
        let mut plan = Remap::default();
        c.unmake(Kind::Page, 1, &mut plan);
        assert_eq!((plan.unmap.len(), c.free(), c.mapped().len()), (2, 20, 0));
    }

    #[test]
    fn a_plan_reuses_the_chunks_it_frees() {
        let mut c = Chunks::new(12, &[(Kind::Page, 24, 2), (Kind::Slot, 24, 2)], 2);
        let mut plan = Remap::default();
        c.make(Kind::Slot, 0, &mut plan);
        assert_eq!((c.free(), c.cost(Kind::Page, 0)), (0, 2));
        let mut plan = Remap::default();
        c.unmake(Kind::Slot, 0, &mut plan);
        c.make(Kind::Page, 0, &mut plan);
        let freed: Vec<u32> = plan.map.iter().map(|&(_, _, ch)| ch).collect();
        assert_eq!((plan.unmap.len(), freed.len(), c.free()), (2, 2, 0));
        assert_eq!(plan.made, [(Kind::Page, 0)]);
        assert_eq!(plan.unmade, [(Kind::Slot, 0)]);
    }
}
