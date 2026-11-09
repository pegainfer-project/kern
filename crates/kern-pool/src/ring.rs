//! A pooled tier on one local disk, as accounting: where each object's
//! bytes go in one file used as a ring, and which objects it still holds.
//!
//! Positions are logical and only grow. A position's byte in the file is
//! the position modulo the capacity, and an object never straddles the
//! end: the head skips to the next lap instead. Everything placed more
//! than one capacity behind the head is gone. The ring evicts first in,
//! first out, by moving its tail, and never anything else.
//!
//! A placement is two-phase. [`Ring::place`] reserves the bytes for a
//! write; [`Ring::commit`] makes the object findable once the write
//! landed, or forgets it when the write failed. A placement that would lap
//! a write still running is refused ([`Placed::Busy`]), so two writes
//! never share a byte. A read is checked against the tail after it lands
//! ([`Ring::holds`]): whatever overwrote it was placed first, and placing
//! moved the tail past it.
//!
//! A key written or being written is not placed again ([`Placed::Held`]):
//! the tier puts a snapshot's whole prefix every time, and the ring keeps
//! the first copy. An object of no bytes (a mark) takes no room and lives
//! until the tail passes where it was placed.

use std::collections::{BTreeMap, VecDeque};

/// Where one object's bytes are: `len` bytes at `offset` in the file,
/// placed at logical position `begin`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Extent {
    begin: u64,
    offset: u64,
    len: u64,
}

impl Extent {
    /// The byte in the file it starts at, a multiple of the ring's alignment.
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// Its bytes, the object's rounded up to the ring's alignment.
    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// What [`Ring::place`] decided.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placed {
    /// Write the object here, then [`Ring::commit`].
    New(Extent),
    /// The key is written or being written already.
    Held,
    /// Room for it would lap a write still running.
    Busy,
    /// The object is larger than the ring.
    TooLarge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Writing,
    Written,
}

/// One file's objects: the head, what lives between the tail and it, and
/// in what order it was placed.
#[derive(Debug)]
pub struct Ring {
    capacity: u64,
    align: u64,
    head: u64,
    entries: BTreeMap<String, (Extent, State)>,
    order: VecDeque<(u64, String)>,
}

impl Ring {
    /// A ring over `capacity` bytes, rounded down to `align`; every extent
    /// starts on an `align` boundary and is a whole number of them.
    pub fn new(capacity: u64, align: u64) -> Ring {
        assert!(align > 0, "an alignment of at least one byte");
        Ring { capacity: capacity / align * align, align, head: 0, entries: BTreeMap::new(), order: VecDeque::new() }
    }

    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    /// The oldest logical position still held.
    fn tail(&self) -> u64 {
        self.head.saturating_sub(self.capacity)
    }

    /// Room for `key`'s `bytes` at the head, evicting what that laps.
    pub fn place(&mut self, key: &str, bytes: u64) -> Placed {
        if self.entries.contains_key(key) {
            return Placed::Held;
        }
        let len = bytes.div_ceil(self.align) * self.align;
        if len > self.capacity {
            return Placed::TooLarge;
        }
        let at = self.head % self.capacity.max(1);
        let begin = match len > self.capacity - at {
            true => self.head + (self.capacity - at),
            false => self.head,
        };
        let head = begin + len;
        let tail = head.saturating_sub(self.capacity);
        let laps_a_write = self
            .order
            .iter()
            .take_while(|(b, _)| *b < tail)
            .any(|(b, k)| matches!(self.entries.get(k), Some((e, State::Writing)) if e.begin == *b));
        if laps_a_write {
            return Placed::Busy;
        }
        self.head = head;
        self.evict();
        let e = Extent { begin, offset: begin % self.capacity.max(1), len };
        self.entries.insert(key.to_string(), (e, State::Writing));
        self.order.push_back((begin, key.to_string()));
        Placed::New(e)
    }

    /// Settle `key`'s write to `e`: findable from now on when it landed,
    /// forgotten when it did not. False when `key` is not being written
    /// to `e`.
    pub fn commit(&mut self, key: &str, e: Extent, landed: bool) -> bool {
        match self.entries.get_mut(key) {
            Some((x, s @ State::Writing)) if *x == e => {
                if landed {
                    *s = State::Written;
                } else {
                    self.entries.remove(key);
                }
                landed
            }
            _ => false,
        }
    }

    /// Where `key`'s bytes are, once written.
    pub fn find(&self, key: &str) -> Option<Extent> {
        match self.entries.get(key) {
            Some(&(e, State::Written)) => Some(e),
            _ => None,
        }
    }

    /// Whether `e`'s bytes are still the ones written there: asked after a
    /// read of them landed.
    pub fn holds(&self, e: Extent) -> bool {
        e.begin >= self.tail()
    }

    /// Objects written and being written.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn evict(&mut self) {
        let tail = self.tail();
        while let Some((begin, _)) = self.order.front() {
            if *begin >= tail {
                return;
            }
            let (begin, key) = self.order.pop_front().expect("a front");
            if self.entries.get(&key).is_some_and(|(e, _)| e.begin == begin) {
                self.entries.remove(&key);
            }
        }
    }
}
