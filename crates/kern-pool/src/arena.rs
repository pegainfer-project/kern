//! Room in one fixed block, first fit: what a tier that cannot reach the
//! device directly stages its transfers through. A [`Room`] is handed out
//! by [`Arena::take`] and only [`Arena::give`] takes it back, so the same
//! bytes are never handed out twice.

use std::collections::BTreeMap;

/// `len` bytes at `offset` of the block, until given back.
#[derive(Debug, PartialEq, Eq)]
pub struct Room {
    offset: u64,
    len: u64,
}

impl Room {
    /// A multiple of the arena's alignment.
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// The bytes asked for, rounded up to the arena's alignment.
    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// A range's free ranges, by offset; neighbours are always merged.
#[derive(Debug)]
pub struct Arena {
    size: u64,
    align: u64,
    free: BTreeMap<u64, u64>,
}

impl Arena {
    /// The `size` bytes of a block from `begin`, both on `align` (the
    /// size rounded down).
    pub fn new(begin: u64, size: u64, align: u64) -> Arena {
        assert!(align > 0 && begin.is_multiple_of(align), "a range that starts on its alignment");
        let size = size / align * align;
        let free = if size > 0 { BTreeMap::from([(begin, size)]) } else { BTreeMap::new() };
        Arena { size, align, free }
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    /// Free bytes.
    pub fn free(&self) -> u64 {
        self.free.values().sum()
    }

    /// The lowest room of `bytes`, rounded up to the alignment; none when
    /// no free range is that long.
    pub fn take(&mut self, bytes: u64) -> Option<Room> {
        let len = bytes.div_ceil(self.align) * self.align;
        if len == 0 {
            return Some(Room { offset: 0, len: 0 });
        }
        let (&at, &n) = self.free.iter().find(|(_, &n)| n >= len)?;
        self.free.remove(&at);
        if n > len {
            self.free.insert(at + len, n - len);
        }
        Some(Room { offset: at, len })
    }

    pub fn give(&mut self, r: Room) {
        if r.len == 0 {
            return;
        }
        let (mut at, mut len) = (r.offset, r.len);
        if let Some((&p, &n)) = self.free.range(..at).next_back() {
            if p + n == at {
                self.free.remove(&p);
                (at, len) = (p, len + n);
            }
        }
        if let Some(n) = self.free.remove(&(at + len)) {
            len += n;
        }
        self.free.insert(at, len);
    }
}
