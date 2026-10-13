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
//!
//! Another runtime reaches these states through a [`Placement`]: the
//! chunk at every position, which it mirrors over the same chunks mapped
//! into its own address space. A remap moves chunks only off objects
//! nobody holds, so an object held when a placement is taken stays where
//! that placement says for as long as it is held.

use std::collections::BTreeSet;
use std::ops::Range;

use crate::error::{bail, Result};

/// What an arena's objects are.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
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
    /// The access grants as maximal contiguous spans per arena: one grant
    /// per span costs the driver the same per chunk and nothing per object.
    pub fn access_spans(&self) -> Vec<(usize, Range<usize>)> {
        let mut grants: Vec<_> = self.access.iter().map(|(a, r)| (*a, r.start, r.end)).collect();
        grants.sort_unstable();
        grants.into_iter().fold(Vec::new(), |mut spans: Vec<(usize, Range<usize>)>, (a, start, end)| {
            match spans.last_mut() {
                Some((la, lr)) if *la == a && start <= lr.end => lr.end = lr.end.max(end),
                _ => spans.push((a, start..end)),
            }
            spans
        })
    }
}

/// The chunk at every position of every arena, in arena order: where a
/// mirror of the arenas maps what. Each chunk is at one position at most.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Placement(Vec<Vec<Option<u32>>>);

impl Placement {
    /// Arenas of `positions` positions each, nothing mapped.
    pub fn unmapped(positions: &[usize]) -> Placement {
        Placement(positions.iter().map(|&n| vec![None; n]).collect())
    }

    /// Positions per arena.
    pub fn shape(&self) -> Vec<usize> {
        self.0.iter().map(Vec::len).collect()
    }

    /// The chunk at position `p` of arena `a`.
    pub fn at(&self, a: usize, p: usize) -> Option<u32> {
        self.0[a][p]
    }

    /// One past the highest chunk placed (0 when nothing is).
    pub fn chunks(&self) -> u32 {
        self.0.iter().flatten().flatten().map(|&c| c + 1).max().unwrap_or(0)
    }

    /// The remap that turns this placement into `next`, of the same shape:
    /// every position whose chunk changes comes off, then gets its new
    /// chunk, then is granted access.
    pub fn toward(&self, next: &Placement) -> Remap {
        assert_eq!(self.shape(), next.shape(), "placements of different arenas");
        let mut plan = Remap::default();
        for (a, (was, now)) in self.0.iter().zip(&next.0).enumerate() {
            for (p, (&w, &n)) in was.iter().zip(now).enumerate() {
                if w == n {
                    continue;
                }
                if w.is_some() {
                    plan.unmap.push((a, p));
                }
                if let Some(c) = n {
                    plan.map.push((a, p, c));
                    plan.access.push((a, p..p + 1));
                }
            }
        }
        plan
    }

    /// This placement once `plan` has run over `chunks` chunks: its unmaps
    /// off mapped positions, then its maps onto empty ones, every chunk at
    /// one position at most, then access over mapped positions only. A plan
    /// that breaks any of these is its planner's bug, refused whole.
    pub fn apply(&self, plan: &Remap, chunks: u32) -> Result<Placement> {
        let shape = self.shape();
        let inside = |a: usize, p: usize| shape.get(a).is_some_and(|&n| p < n);
        let mut next = self.clone();
        for &(a, p) in &plan.unmap {
            if !inside(a, p) || next.0[a][p].take().is_none() {
                bail!(Api, "a remap unmaps position {p} of arena {a}, which holds no chunk");
            }
        }
        let mut placed: BTreeSet<u32> = next.0.iter().flatten().flatten().copied().collect();
        for &(a, p, c) in &plan.map {
            if c >= chunks || !inside(a, p) {
                bail!(Api, "a remap maps chunk {c} at position {p} of arena {a}: {chunks} chunks, positions {shape:?}");
            }
            if let Some(was) = next.0[a][p] {
                bail!(Api, "a remap maps chunk {c} at position {p} of arena {a}, which holds chunk {was}");
            }
            if !placed.insert(c) {
                bail!(Api, "a remap maps chunk {c} at position {p} of arena {a} while it is placed elsewhere");
            }
            next.0[a][p] = Some(c);
        }
        if let Some((a, p)) = plan
            .access
            .iter()
            .flat_map(|(a, r)| r.clone().map(|p| (*a, p)))
            .find(|&(a, p)| !inside(a, p) || next.0[a][p].is_none())
        {
            bail!(Api, "a remap grants access at position {p} of arena {a}, which holds no chunk");
        }
        Ok(next)
    }

    /// Wire form: the arena count, then per arena its position count and
    /// a chunk per position (`u32::MAX` for none), little-endian `u32`s.
    pub fn to_bytes(&self) -> Vec<u8> {
        let words = std::iter::once(self.0.len() as u32).chain(
            self.0.iter().flat_map(|a| std::iter::once(a.len() as u32).chain(a.iter().map(|c| c.unwrap_or(u32::MAX)))),
        );
        words.flat_map(u32::to_le_bytes).collect()
    }

    /// The placement [`Placement::to_bytes`] wrote; `None` for bytes that
    /// are not one, or that place a chunk twice.
    pub fn from_bytes(b: &[u8]) -> Option<Placement> {
        if !b.len().is_multiple_of(4) {
            return None;
        }
        let mut words = b.as_chunks::<4>().0.iter().map(|&w| u32::from_le_bytes(w));
        let arenas = (0..words.next()?)
            .map(|_| {
                let n = words.next()?;
                (0..n).map(|_| words.next().map(|c| (c != u32::MAX).then_some(c))).collect::<Option<Vec<_>>>()
            })
            .collect::<Option<Vec<_>>>()?;
        let placed: Vec<u32> = arenas.iter().flatten().flatten().copied().collect();
        let distinct: BTreeSet<u32> = placed.iter().copied().collect();
        (words.next().is_none() && distinct.len() == placed.len()).then_some(Placement(arenas))
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

    pub fn placement(&self) -> Placement {
        Placement(self.arenas.iter().map(|a| a.chunk.clone()).collect())
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
}
