//! The local-disk tier's accounting through its public API: a ring that
//! never hands out bytes a running write or a findable object still owns,
//! keeps what was placed less than a lap ago; an arena whose rooms never
//! overlap and merge back into one block; and the plans an act gets, one
//! flag per object and every byte it moves staged in a room of its own.

mod common;

use std::collections::BTreeMap;

use common::Rand;
use kern_pool::{Arena, Disk, Extent, Placed, Plan, Ring, Room};

fn new(p: Placed) -> Extent {
    match p {
        Placed::New(e) => e,
        other => panic!("placed {other:?}"),
    }
}

#[test]
fn a_ring_places_at_the_head_skips_the_end_and_evicts_the_oldest() {
    let mut r = Ring::new(100, 8);
    assert_eq!(r.capacity(), 96);
    let a = new(r.place("a", 30));
    assert_eq!((a.offset(), a.len()), (0, 32));
    // Not findable until its write lands; placed or written, never twice.
    assert_eq!((r.find("a"), r.place("a", 30)), (None, Placed::Held));
    assert!(r.commit("a", a, true));
    assert_eq!((r.find("a"), r.place("a", 30)), (Some(a), Placed::Held));
    let b = new(r.place("b", 40));
    assert_eq!((b.offset(), b.len()), (32, 40));
    assert!(r.commit("b", b, true));
    // 24 bytes left before the end: a 32-byte object starts the next lap
    // and laps `a`, nothing else.
    let c = new(r.place("c", 32));
    assert_eq!(c.offset(), 0);
    assert_eq!((r.find("a"), r.find("b"), r.holds(a), r.holds(b)), (None, Some(b), false, true));
    // A failed write is forgotten; its key can be placed again.
    assert!(!r.commit("c", c, false));
    assert_eq!(r.find("c"), None);
    assert!(matches!(r.place("c", 32), Placed::New(_)));
    assert_eq!(r.place("huge", 97), Placed::Refused);
}

#[test]
fn a_ring_refuses_to_lap_a_running_write() {
    let mut r = Ring::new(64, 8);
    let a = new(r.place("a", 48));
    // `b` would overwrite `a`'s bytes while its write runs.
    assert_eq!(r.place("b", 32), Placed::Refused);
    assert!(r.commit("a", a, true));
    let b = new(r.place("b", 32));
    assert_eq!((b.offset(), r.find("a")), (0, None));
    // A commit of a key to an extent it was not placed at changes nothing.
    assert!(!r.commit("b", a, true));
    assert_eq!(r.find("b"), None);
}

#[test]
fn a_mark_takes_no_room_and_lives_a_lap() {
    let mut r = Ring::new(64, 8);
    let m = new(r.place("m", 0));
    assert!(m.is_empty());
    assert!(r.commit("m", m, true));
    let a = new(r.place("a", 32));
    assert_eq!(a.offset(), 0);
    assert!(r.commit("a", a, true));
    assert!(r.find("m").is_some());
    let b = new(r.place("b", 40));
    assert!(r.commit("b", b, true));
    assert_eq!((r.find("m"), r.find("a")), (None, None));
}

/// A write placed but not landed, or a read issued but not landed.
struct Write {
    key: String,
    e: Extent,
    id: u32,
}

struct Read {
    e: Extent,
    id: u32,
    seen: Vec<u32>,
}

/// The file as which write last put each byte.
fn bytes(disk: &[u32], e: Extent) -> Vec<u32> {
    disk[e.offset() as usize..(e.offset() + e.len()) as usize].to_vec()
}

fn overlap(a: Extent, b: Extent) -> bool {
    a.offset() < b.offset() + b.len() && b.offset() < a.offset() + a.len()
}

#[test]
fn a_ring_never_shows_bytes_but_the_ones_written_for_the_key() {
    const ALIGN: u64 = 4;
    const MAX: u64 = 40;
    for seed in 1..=40u64 {
        let mut rand = Rand::new((seed * 0x9e37_79b9) | 1);
        let cap = 64 * ALIGN;
        let mut r = Ring::new(cap, ALIGN);
        let mut disk = vec![0u32; cap as usize];
        let mut writing: Vec<Write> = Vec::new();
        let mut reading: Vec<Read> = Vec::new();
        // The write id each findable key's bytes came from, and the
        // logical bytes placed since each key was.
        let mut written: BTreeMap<String, u32> = BTreeMap::new();
        let mut since: BTreeMap<String, (u64, u64)> = BTreeMap::new();
        let mut next_id = 1u32;
        for _ in 0..3000 {
            match rand.next(10) {
                0..=3 => {
                    let key = format!("k{}", rand.next(60));
                    let len = match rand.next(8) {
                        0 => 0,
                        _ => rand.next(MAX as usize + 1) as u64,
                    };
                    let known = r.find(&key).is_some() || writing.iter().any(|w| w.key == key);
                    match r.place(&key, len) {
                        Placed::New(e) => {
                            assert!(!known, "seed {seed}: `{key}` placed again");
                            assert!(
                                writing.iter().all(|w| !overlap(w.e, e)),
                                "seed {seed}: `{key}` placed over a running write"
                            );
                            for (_, after) in since.values_mut() {
                                *after += e.len();
                            }
                            since.insert(key.clone(), (e.len(), 0));
                            writing.push(Write { key, e, id: next_id });
                            next_id += 1;
                        }
                        Placed::Held => assert!(known, "seed {seed}: `{key}` held but never placed"),
                        // Never larger than the ring here: only a running write refuses.
                        Placed::Refused => assert!(!writing.is_empty(), "seed {seed}: refused with nothing written"),
                    }
                }
                4..=6 if !writing.is_empty() => {
                    let w = writing.swap_remove(rand.next(writing.len()));
                    for b in &mut disk[w.e.offset() as usize..(w.e.offset() + w.e.len()) as usize] {
                        *b = w.id;
                    }
                    let landed = rand.next(10) > 0;
                    assert_eq!(r.commit(&w.key, w.e, landed), landed, "seed {seed}");
                    if landed {
                        written.insert(w.key, w.id);
                    }
                }
                7 => {
                    let key = format!("k{}", rand.next(60));
                    if let Some(e) = r.find(&key) {
                        let id = written[&key];
                        assert!(bytes(&disk, e).iter().all(|&b| b == id), "seed {seed}: `{key}` reads another write");
                        reading.push(Read { e, id, seen: bytes(&disk, e) });
                    }
                }
                8 if !reading.is_empty() => {
                    let rd = reading.swap_remove(rand.next(reading.len()));
                    if r.holds(rd.e) {
                        assert!(
                            rd.seen.iter().chain(&bytes(&disk, rd.e)).all(|&b| b == rd.id),
                            "seed {seed}: a read the ring holds saw another write"
                        );
                    }
                }
                _ => {}
            }
            // Less than a lap behind the head (one skip of at most MAX
            // over the end), a written key is still findable.
            for (key, &(len, after)) in &since {
                if written.contains_key(key) && len + after + MAX <= cap {
                    assert!(r.find(key).is_some(), "seed {seed}: `{key}` evicted {after} bytes later");
                }
            }
            written.retain(|k, _| r.find(k).is_some());
            since.retain(|k, _| r.find(k).is_some() || writing.iter().any(|w| &w.key == k));
        }
    }
}

#[test]
fn an_arena_hands_out_aligned_rooms_first_fit() {
    let mut a = Arena::new(0, 100, 16);
    assert_eq!((a.size(), a.free()), (96, 96));
    let x = a.take(20).unwrap();
    let y = a.take(16).unwrap();
    assert_eq!((x.offset(), x.len(), y.offset(), y.len()), (0, 32, 32, 16));
    assert_eq!(a.take(64), None);
    a.give(x);
    // The freed head fits 32; the 48 after `y` fits what it does not.
    assert_eq!(a.take(48).map(|r| r.offset()), Some(48));
    assert_eq!(a.take(32).map(|r| r.offset()), Some(0));
    assert_eq!((a.free(), a.take(0).map(|r| r.len())), (0, Some(0)));
}

#[test]
fn an_arena_never_overlaps_and_merges_back_whole() {
    for seed in 1..=40u64 {
        let mut rand = Rand::new((seed * 0x2545_f491) | 1);
        let mut a = Arena::new(0, 1024, 8);
        let mut held: Vec<Room> = Vec::new();
        for _ in 0..2000 {
            if rand.next(2) == 0 || held.is_empty() {
                let bytes = rand.next(200) as u64;
                if let Some(r) = a.take(bytes) {
                    assert_eq!(
                        (r.offset() % 8, r.len(), r.offset() + r.len() <= 1024),
                        (0, bytes.div_ceil(8) * 8, true)
                    );
                    if !r.is_empty() {
                        assert!(
                            held.iter()
                                .all(|h| r.offset() + r.len() <= h.offset() || h.offset() + h.len() <= r.offset()),
                            "seed {seed}: {r:?} overlaps a room held"
                        );
                    }
                    held.push(r);
                }
            } else {
                a.give(held.swap_remove(rand.next(held.len())));
            }
            assert_eq!(a.free(), 1024 - held.iter().map(Room::len).sum::<u64>(), "seed {seed}");
        }
        for r in held.drain(..) {
            a.give(r);
        }
        assert_eq!(a.take(1024).map(|r| r.offset()), Some(0), "seed {seed}");
    }
}

fn put(d: &mut Disk, objects: &[(&str, u64)], landed: bool) -> Vec<bool> {
    let p = d.put(objects);
    let ok = p.ok.clone();
    let n = p.moves.len();
    let kept = d.wrote(p, &vec![landed; n]);
    assert_eq!(kept, if landed { n } else { 0 });
    ok
}

#[test]
fn a_put_stages_before_it_places_and_skips_what_the_disk_holds() {
    // 64 bytes of block: 32 stage puts, 32 gets.
    let mut d = Disk::new(64, 64, 8);
    assert_eq!(put(&mut d, &[("a", 30), ("none", 0)], true), [true, true]);
    assert_eq!(d.probe(&["a".into(), "none".into()]), [true, false]);
    // On disk already: landed, nothing moves.
    let p = d.put(&[("a", 30)]);
    assert_eq!((p.ok.as_slice(), p.moves.len()), (&[true][..], 0));
    // No room in the puts' half: refused, and `a` was not evicted for it.
    assert_eq!(put(&mut d, &[("b", 40)], true), [false]);
    assert_eq!(d.probe(&["a".into(), "b".into()]), [true, false]);
    // A write that did not land is forgotten, though its put was answered.
    assert_eq!(put(&mut d, &[("c", 8)], false), [true]);
    assert_eq!(d.probe(&["c".into()]), [false]);
}

#[test]
fn a_key_another_put_is_writing_moves_nothing_and_is_found_once_written() {
    let mut d = Disk::new(256, 128, 8);
    let first = d.put(&[("k", 16)]);
    let second = d.put(&[("k", 16), ("j", 8)]);
    assert_eq!((first.moves.len(), second.moves.len(), second.ok.as_slice()), (1, 1, &[true, true][..]));
    assert_eq!(d.probe(&["k".into()]), [false]);
    assert_eq!(d.wrote(first, &[true]), 1);
    assert_eq!(d.probe(&["k".into()]), [true]);
}

#[test]
fn a_backlog_of_puts_leaves_the_gets_their_room() {
    let mut d = Disk::new(1024, 64, 8);
    put(&mut d, &[("g", 32)], true);
    let backlog = d.put(&[("a", 32)]);
    assert_eq!((backlog.moves.len(), d.put(&[("b", 8)]).ok.as_slice()), (1, &[false][..]));
    assert_eq!(d.get(&[("g", 32)]).moves.len(), 1);
}

#[test]
fn a_get_moves_the_leading_run_the_disk_holds_at_its_size() {
    let mut d = Disk::new(256, 128, 8);
    put(&mut d, &[("p0", 16), ("p1", 16), ("p3", 16)], true);
    let at = |p: &Plan| p.moves.iter().map(|m| (m.at, m.key.clone())).collect::<Vec<_>>();
    // `p2` is missing: nothing past it is read.
    let p = d.get(&[("p0", 16), ("p1", 16), ("p2", 16), ("p3", 16)]);
    assert_eq!(at(&p), [(0, "p0".to_string()), (1, "p1".to_string())]);
    // Held at another size is not held.
    assert!(d.get(&[("p0", 24)]).moves.is_empty());
    // The gets' half holds 64: four rooms of 16, then none.
    let q = d.get(&[("p3", 16), ("p3", 16), ("p3", 16)]);
    assert_eq!(q.moves.len(), 2);
    let mut p = p;
    d.read(&mut p, &[true, false]);
    assert_eq!(p.ok, [true, false, false, false]);
}

#[test]
fn an_abandoned_put_commits_nothing_and_keeps_its_rooms() {
    let mut d = Disk::new(256, 64, 8);
    let p = d.put(&[("a", 32)]);
    d.abandon(p);
    assert_eq!(d.probe(&["a".into()]), [false]);
    // Its room may still be written by a copy that never ran.
    assert_eq!(d.put(&[("b", 8)]).ok, [false]);
}

#[test]
fn every_plan_flags_each_object_and_stages_each_move_alone() {
    for seed in 1..=40u64 {
        let mut rand = Rand::new((seed * 0x5851_f42d) | 1);
        let mut d = Disk::new(4096, 512, 8);
        let mut open: Vec<(bool, Plan)> = Vec::new();
        for _ in 0..1500 {
            if rand.next(3) > 0 || open.is_empty() {
                let keys: Vec<String> = (0..1 + rand.next(6)).map(|_| format!("k{}", rand.next(40))).collect();
                let objects: Vec<(&str, u64)> = keys.iter().map(|k| (k.as_str(), (rand.next(5) * 8) as u64)).collect();
                let get = rand.next(2) == 0;
                let p = if get { d.get(&objects) } else { d.put(&objects) };
                assert_eq!(p.ok.len(), objects.len(), "seed {seed}");
                for m in &p.moves {
                    let (key, bytes) = objects[m.at];
                    assert_eq!((m.key.as_str(), m.extent.bytes(), m.room.len()), (key, bytes, bytes.div_ceil(8) * 8));
                }
                if get {
                    // The moves are the objects in order, the empty ones skipped.
                    let ats: Vec<usize> = p.moves.iter().map(|m| m.at).collect();
                    let lead: Vec<usize> = (0..objects.len()).filter(|&i| objects[i].1 > 0).take(ats.len()).collect();
                    assert_eq!(ats, lead, "seed {seed}");
                }
                let rooms = || open.iter().flat_map(|(_, p)| &p.moves).map(|m| &m.room);
                for m in &p.moves {
                    assert!(
                        rooms()
                            .all(|r| m.room.offset() + m.room.len() <= r.offset()
                                || r.offset() + r.len() <= m.room.offset()),
                        "seed {seed}: a room staged twice"
                    );
                }
                open.push((get, p));
            } else {
                let (get, mut p) = open.remove(0);
                let flags: Vec<bool> = p.moves.iter().map(|_| rand.next(8) > 0).collect();
                if get {
                    d.read(&mut p, &flags);
                    // A get lands what moved and landed, and nothing else.
                    let moved: Vec<usize> = p.moves.iter().zip(&flags).filter(|(_, &f)| f).map(|(m, _)| m.at).collect();
                    assert!(
                        p.ok.iter()
                            .enumerate()
                            .all(|(i, &ok)| !ok || moved.contains(&i) || !p.moves.iter().any(|m| m.at == i)),
                        "seed {seed}"
                    );
                    for m in p.moves {
                        d.give(m.room);
                    }
                } else {
                    // A write is found once it landed, never when it did not.
                    let keys: Vec<(String, bool)> =
                        p.moves.iter().zip(&flags).map(|(m, &f)| (m.key.clone(), f)).collect();
                    let landed = flags.iter().filter(|&&f| f).count();
                    assert_eq!(d.wrote(p, &flags), landed, "seed {seed}");
                    for (k, f) in keys {
                        assert_eq!(d.probe(std::slice::from_ref(&k)), [f], "seed {seed}: `{k}`");
                    }
                }
            }
        }
    }
}
