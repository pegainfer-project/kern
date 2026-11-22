//! The disk tier's decisions, whole: for one act, which objects move,
//! through which room of the pinned block ([`Arena`]) and which extent of
//! the file ([`Ring`]), and what the act answers for each object. The
//! shell moves the bytes the [`Plan`] names and hands back what landed;
//! everything else is here.
//!
//! Puts and gets stage through halves of the block of their own, so a
//! backlog of writes never leaves a fetch without room. A put takes a room
//! before it places, so a block too full to stage the object never evicts
//! a live one from the file. A key the file holds or another put is
//! writing moves nothing and is answered as landed (every snapshot puts
//! its whole prefix). A put is answered once it is staged, not once it is
//! written: the tier is a hint (pool.md §9, rule 1), and a write that later
//! fails is only an object nobody finds. A get moves the act's objects in
//! order up to the first the file does not hold at exactly its size, or
//! the block has no room for: a fetch lands a leading run or nothing, so
//! reading past a gap buys nothing. An object with no bytes on this rank
//! is landed at once.

use crate::arena::{Arena, Room};
use crate::ring::{Extent, Placed, Ring};

/// The file and the block, as accounting.
#[derive(Debug)]
pub struct Disk {
    ring: Ring,
    puts: Arena,
    gets: Arena,
}

/// Object `at` of an act, moving between `extent` of the file and `room`
/// of the block.
#[derive(Debug, PartialEq, Eq)]
pub struct Move {
    pub at: usize,
    pub key: String,
    pub extent: Extent,
    pub room: Room,
}

/// What one act does: a flag per object, and the objects that move.
#[derive(Debug)]
pub struct Plan {
    pub ok: Vec<bool>,
    pub moves: Vec<Move>,
}

/// One wave of a multi-wave put or get over a remaining slice of objects:
/// the moves staged into the pinned block, how many objects of the slice
/// were consumed (`next`), and whether the act is finished (`done`) or
/// paused waiting for staging room.
#[derive(Debug)]
pub struct Wave {
    pub plan: Plan,
    pub next: usize,
    pub done: bool,
}

impl Disk {
    /// A file of `capacity` bytes and a block of `staging`, both placed on
    /// `align`; the block's first half stages puts, the rest gets.
    pub fn new(capacity: u64, staging: u64, align: u64) -> Disk {
        let half = staging / 2 / align * align;
        Disk {
            ring: Ring::new(capacity, align),
            puts: Arena::new(0, half, align),
            gets: Arena::new(half, staging - half, align),
        }
    }

    pub fn capacity(&self) -> u64 {
        self.ring.capacity()
    }

    pub fn staging(&self) -> u64 {
        self.puts.size() + self.gets.size()
    }

    pub fn put_staging(&self) -> u64 {
        self.puts.size()
    }

    pub fn get_staging(&self) -> u64 {
        self.gets.size()
    }

    /// Whether every non-zero object is currently on disk at its exact byte
    /// size and fits inside the gets staging half.
    pub fn holds_all(&self, objects: &[(&str, u64)]) -> bool {
        objects.iter().all(|&(key, bytes)| {
            bytes == 0 || (bytes <= self.gets.size() && self.ring.find(key).is_some_and(|e| e.bytes() == bytes))
        })
    }

    /// Whether each key is written.
    pub fn probe(&self, keys: &[String]) -> Vec<bool> {
        keys.iter().map(|k| self.ring.find(k).is_some()).collect()
    }

    /// A mark: an object of no bytes, written as soon as it is placed.
    pub fn mark(&mut self, key: &str) -> bool {
        match self.ring.place(key, 0) {
            Placed::New(e) => self.ring.commit(key, e, true),
            Placed::Held => true,
            Placed::Refused => false,
        }
    }

    /// A put of `objects`, each its key and its bytes on this rank: `ok`
    /// is its answer once the moves are staged.
    pub fn put(&mut self, objects: &[(&str, u64)]) -> Plan {
        let mut plan = Plan { ok: vec![true; objects.len()], moves: Vec::new() };
        for (at, &(key, bytes)) in objects.iter().enumerate() {
            if bytes == 0 || self.ring.find(key).is_some() {
                continue;
            }
            let Some(room) = self.puts.take(bytes) else {
                plan.ok[at] = false;
                continue;
            };
            match self.ring.place(key, bytes) {
                Placed::New(extent) => plan.moves.push(Move { at, key: key.to_string(), extent, room }),
                Placed::Held => self.puts.give(room),
                Placed::Refused => {
                    self.puts.give(room);
                    plan.ok[at] = false;
                }
            }
        }
        plan
    }

    /// Settle a put whose writes ended, `landed` one flag per move: each
    /// committed or forgotten and its room back. How many were kept.
    pub fn wrote(&mut self, plan: Plan, landed: &[bool]) -> usize {
        assert_eq!(landed.len(), plan.moves.len(), "a flag per move");
        let mut kept = 0;
        for (m, &l) in plan.moves.into_iter().zip(landed) {
            kept += usize::from(self.ring.commit(&m.key, m.extent, l));
            self.puts.give(m.room);
        }
        kept
    }

    /// Give up on a put whose rooms may still be written by a copy that
    /// never ran: nothing is committed, and its rooms are never handed out
    /// again.
    pub fn abandon(&mut self, plan: Plan) {
        for m in &plan.moves {
            self.ring.commit(&m.key, m.extent, false);
        }
    }

    /// A get of `objects`, each its key and its bytes on this rank.
    pub fn get(&mut self, objects: &[(&str, u64)]) -> Plan {
        let mut plan = Plan { ok: vec![false; objects.len()], moves: Vec::new() };
        for (at, &(key, bytes)) in objects.iter().enumerate() {
            if bytes == 0 {
                plan.ok[at] = true;
                continue;
            }
            let Some(extent) = self.ring.find(key).filter(|e| e.bytes() == bytes) else { break };
            let Some(room) = self.gets.take(bytes) else { break };
            plan.moves.push(Move { at, key: key.to_string(), extent, room });
        }
        plan
    }

    /// A get's reads ended, `got` one flag per move: each believed when it
    /// landed whole and the file still holds its extent. The rooms stay
    /// taken until [`Disk::give`].
    pub fn read(&self, plan: &mut Plan, got: &[bool]) {
        assert_eq!(got.len(), plan.moves.len(), "a flag per move");
        for (m, &g) in plan.moves.iter().zip(got) {
            plan.ok[m.at] = g && self.ring.holds(m.extent);
        }
    }

    /// Plan one wave of a put over `objects`, staging up to `budget` bytes
    /// (at least one object if it fits) so callers can stream puts larger
    /// than the puts staging half. `plan.ok` covers `0..next`, and `done`
    /// is true once `next == objects.len()`.
    pub fn put_wave(&mut self, objects: &[(&str, u64)], budget: u64) -> Wave {
        let mut ok = Vec::new();
        let mut moves = Vec::new();
        let mut staged = 0u64;
        let mut next = 0usize;
        for (at, &(key, bytes)) in objects.iter().enumerate() {
            if bytes == 0 || self.ring.find(key).is_some() {
                ok.push(true);
                next = at + 1;
                continue;
            }
            if bytes > self.puts.size() {
                ok.push(false);
                next = at + 1;
                continue;
            }
            if !moves.is_empty() && staged.saturating_add(bytes) > budget {
                break;
            }
            let Some(room) = self.puts.take(bytes) else {
                break;
            };
            match self.ring.place(key, bytes) {
                Placed::New(extent) => {
                    moves.push(Move { at, key: key.to_string(), extent, room });
                    ok.push(true);
                    staged += bytes;
                }
                Placed::Held => {
                    self.puts.give(room);
                    ok.push(true);
                }
                Placed::Refused => {
                    self.puts.give(room);
                    ok.push(false);
                }
            }
            next = at + 1;
        }
        Wave { plan: Plan { ok, moves }, next, done: next == objects.len() }
    }

    /// Plan one wave of a get over `objects`, staging up to `budget` bytes
    /// (at least one object if it fits) so callers can stream fetches larger
    /// than the gets staging half. `plan.ok` covers `0..next`. `done` is
    /// true once `next == objects.len()`, a key is missing/wrong-sized on
    /// disk, or an object exceeds the gets staging half.
    pub fn get_wave(&mut self, objects: &[(&str, u64)], budget: u64) -> Wave {
        let mut ok = Vec::new();
        let mut moves = Vec::new();
        let mut staged = 0u64;
        let mut next = 0usize;
        let mut done = false;
        for (at, &(key, bytes)) in objects.iter().enumerate() {
            if bytes == 0 {
                ok.push(true);
                next = at + 1;
                continue;
            }
            if bytes > self.gets.size() {
                done = true;
                break;
            }
            let Some(extent) = self.ring.find(key).filter(|e| e.bytes() == bytes) else {
                done = true;
                break;
            };
            if !moves.is_empty() && staged.saturating_add(bytes) > budget {
                break;
            }
            let Some(room) = self.gets.take(bytes) else {
                break;
            };
            moves.push(Move { at, key: key.to_string(), extent, room });
            ok.push(false);
            staged += bytes;
            next = at + 1;
        }
        if next == objects.len() {
            done = true;
        }
        Wave { plan: Plan { ok, moves }, next, done }
    }

    /// A get's room back.
    pub fn give(&mut self, room: Room) {
        self.gets.give(room);
    }
}
