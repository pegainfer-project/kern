//! Fixtures the integration tests share.

#![allow(dead_code)]

use kern_manifest::types::Manifest;
use kern_manifest::{verify, Protocol};
use kern_pool::Pool;
use kern_serve::ranks::{Groups, Ranks};

/// kern-manifest's `dealt.json` over a group of `w`: two paged states of
/// 16-token pages dealt by position, a page table of 3 pages (48 local
/// positions a member), and a 24-byte recurrent slot a member, its heads'
/// segments scaled to the group.
pub fn dealt(w: usize) -> Manifest {
    dealt_with(w, 24)
}

/// [`dealt`] with a recurrent slot of `slot` bytes a member, a multiple of
/// 6: its heads' segments one row of two thirds of it, two of a sixth.
pub fn dealt_with(w: usize, slot: u64) -> Manifest {
    let json = include_str!("../../../kern-manifest/tests/fixtures/dealt.json")
        .replace("\"tp\": 4", &format!("\"tp\": {w}"))
        .replace("\"bytes_per_seq\": 24", &format!("\"bytes_per_seq\": {slot}"))
        .replace("[[1, 64], [2, 16]]", &format!("[[1, {}], [2, {}]]", slot * 2 / 3 * w as u64, slot / 6 * w as u64));
    Manifest::from_json(&json).unwrap()
}

/// `groups` replicated groups of `w`, every member's pool `slots` slots
/// (the null one and the pad's among them) and the rest of 128 chunks of
/// 256 bytes as pages.
pub fn replicated_with(w: usize, groups: usize, slots: usize) -> Ranks {
    replicated_of(dealt(w), w, groups, slots)
}

/// `groups` replicated groups of `w` running `m`, every member's pool as
/// [`replicated_with`]'s.
pub fn replicated_of(m: Manifest, w: usize, groups: usize, slots: usize) -> Ranks {
    let p = Protocol::check(&verify(m.clone()).unwrap()).unwrap();
    let pools = (0..w * groups).map(|_| Pool::new(&m, 256, 128, slots, None).unwrap().0).collect();
    Ranks::new(p, Groups::new(w * groups, w).unwrap(), pools, (0, 0)).unwrap()
}

/// `groups` replicated groups of `w`, ten slots a member and pages to spare.
pub fn replicated(w: usize, groups: usize) -> Ranks {
    replicated_with(w, groups, 10)
}

/// xorshift64: deterministic cases without a dependency.
pub struct Rand(pub u64);

impl Rand {
    pub fn next(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 33) as usize % n
    }
}
