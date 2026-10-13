//! A placement through the public API: a mirror that follows the pool's
//! placements by the remaps between them holds exactly what the pool
//! placed, and an object held when a placement is taken keeps the chunks
//! that placement names for as long as it is held, whatever the pool
//! remaps in between. An executor that checks each remap against the
//! placement the earlier ones left takes exactly the plans a mapper could
//! run, and every plan the pool makes is one of them.

mod common;

use std::collections::BTreeSet;

use kern_manifest::types::Manifest;
use kern_pool::{Denied, Kind, Lease, Placement, Pool, Remap};

use common::{hybrid, hybrid4, Rand};

/// A mirror's positions, as plain data the test applies remaps to.
fn model(p: &Placement) -> Vec<Vec<Option<u32>>> {
    p.shape().iter().enumerate().map(|(a, &n)| (0..n).map(|q| p.at(a, q)).collect()).collect()
}

/// `m` after a mapper ran `plan` over `chunks` chunks: unmaps off mapped
/// positions, then maps of chunks that exist onto empty ones, then access
/// over mapped ones only, and no chunk at two positions; `None` when one
/// of these fails.
fn run(m: &[Vec<Option<u32>>], plan: &Remap, chunks: u32) -> Option<Vec<Vec<Option<u32>>>> {
    let mut m = m.to_vec();
    for &(a, q) in &plan.unmap {
        m.get_mut(a)?.get_mut(q)?.take()?;
    }
    for &(a, q, c) in &plan.map {
        let at = m.get_mut(a)?.get_mut(q)?;
        if at.is_some() || c >= chunks {
            return None;
        }
        *at = Some(c);
    }
    for (a, r) in plan.access_spans() {
        if !r.clone().all(|q| m.get(a).and_then(|v| v.get(q)).is_some_and(Option::is_some)) {
            return None;
        }
    }
    let placed: Vec<u32> = m.iter().flatten().flatten().copied().collect();
    (placed.iter().collect::<BTreeSet<_>>().len() == placed.len()).then_some(m)
}

/// (arena, position, chunk) of every position `l`'s pages and slot cover.
fn held(p: &Pool, at: &Placement, l: &Lease) -> Vec<(usize, usize, Option<u32>)> {
    let chunk = p.chunk();
    let objects = |kind: Kind| -> Vec<u64> {
        match kind {
            Kind::Page => l.page_ids().iter().map(|&g| g as u64).collect(),
            Kind::Slot => l.seq_slot().map(|s| s as u64).into_iter().collect(),
        }
    };
    p.pooled()
        .iter()
        .enumerate()
        .flat_map(|(a, ar)| {
            objects(ar.kind)
                .into_iter()
                .flat_map(move |o| {
                    let lo = o * ar.object;
                    (lo / chunk) as usize..(lo + ar.object).div_ceil(chunk) as usize
                })
                .map(move |q| (a, q, at.at(a, q)))
        })
        .collect()
}

fn mirror_follows_the_pool(m: &Manifest, chunk: u64, chunks: u32, first_slots: usize, seed: u64) {
    let p = std::sync::Arc::new(Pool::new(m, chunk, chunks, first_slots, None).unwrap().0);
    let mut mirror = model(&Placement::unmapped(&p.placement().shape()));
    let mut driven = p.placement();
    let mut leases: Vec<(Lease, Vec<(usize, usize, Option<u32>)>)> = Vec::new();
    let mut rand = Rand::new(seed);
    let mut remaps = 0;
    for _ in 0..3000 {
        let got = match rand.next(5) {
            0 | 1 => Some(p.lease(1 + rand.next(p.max_seq_tokens()))),
            2 if p.has_slots() => Some(p.lease_slot()),
            3 if !leases.is_empty() => {
                leases.swap_remove(rand.next(leases.len()));
                None
            }
            _ => None,
        };
        match got {
            Some(Ok(l)) => {
                let at = p.placement();
                let cover = held(&p, &at, &l);
                assert!(cover.iter().all(|&(_, _, c)| c.is_some()), "a leased object over an empty position");
                leases.push((l, cover));
            }
            Some(Err(Denied::Remapping)) => {
                let plan = p.take_pending().expect("a remap planned");
                driven = driven.apply(&plan, chunks).expect("the pool's own remap follows its placement");
                p.complete(plan);
                assert_eq!(driven, p.placement());
                remaps += 1;
            }
            _ => {}
        }
        let now = p.placement();
        let plan = Placement::from_bytes(&model_bytes(&mirror)).expect("a mirror is a placement").toward(&now);
        mirror = run(&mirror, &plan, chunks).expect("a mirror runs the remap toward the pool");
        assert_eq!(mirror, model(&now));
        assert_eq!(Placement::from_bytes(&now.to_bytes()), Some(now.clone()));
        for (l, cover) in &leases {
            assert_eq!(&held(&p, &now, l), cover, "a held object moved");
        }
    }
    assert!(remaps > 0, "pages and slots never traded a chunk");
}

/// The wire form of a model, so the mirror is rebuilt from bytes each step.
fn model_bytes(m: &[Vec<Option<u32>>]) -> Vec<u8> {
    std::iter::once(m.len() as u32)
        .chain(m.iter().flat_map(|a| std::iter::once(a.len() as u32).chain(a.iter().map(|c| c.unwrap_or(u32::MAX)))))
        .flat_map(u32::to_le_bytes)
        .collect()
}

#[test]
fn a_mirror_follows_pages_and_slots_trading_chunks() {
    // 3 slots and 6 pages over 18 one-object chunks.
    mirror_follows_the_pool(&hybrid4(), 4, 18, 3, 0x2545_F491_4F6C_DD1D);
}

#[test]
fn a_mirror_follows_objects_sharing_a_chunk() {
    // 16-byte chunks: a 24-byte slot ends halfway into one, so two slots share it.
    mirror_follows_the_pool(&hybrid(), 16, 12, 3, 0x9E37_79B9_7F4A_7C15);
}

#[test]
fn a_placement_naming_a_chunk_twice_is_refused() {
    let twice = model_bytes(&[vec![Some(3), None], vec![Some(3)]]);
    let once = model_bytes(&[vec![Some(3), None], vec![Some(4)]]);
    assert_eq!((Placement::from_bytes(&twice), Placement::from_bytes(&once).map(|p| p.chunks())), (None, Some(5)));
    assert_eq!(Placement::from_bytes(&once[..once.len() - 1]), None);
}

/// Small random placements and plans, valid and not: an unmap, a map or
/// an access anywhere in or just past the arenas, chunks up to one past.
#[test]
fn a_remap_is_taken_exactly_when_a_mapper_could_run_it() {
    let (shape, chunks) = ([3usize, 2], 6u32);
    let mut rand = Rand::new(0xD1B5_4A32_D192_ED03);
    let (mut taken, mut refused) = (0, 0);
    for _ in 0..20_000 {
        let mut free: Vec<u32> = (0..chunks).collect();
        let m: Vec<Vec<Option<u32>>> = shape
            .iter()
            .map(|&n| (0..n).map(|_| (rand.next(2) == 0).then(|| free.swap_remove(rand.next(free.len())))).collect())
            .collect();
        // One in eight past its arenas, one in eight past its arena's positions.
        let at = |rand: &mut Rand| {
            let a = if rand.next(8) == 0 { shape.len() } else { rand.next(shape.len()) };
            let n = shape.get(a).copied().unwrap_or(0);
            (a, if rand.next(8) == 0 { n } else { rand.next(n.max(1)) })
        };
        let mut plan = Remap::default();
        for _ in 0..rand.next(3) {
            plan.unmap.push(at(&mut rand));
        }
        for _ in 0..rand.next(3) {
            let (a, q) = at(&mut rand);
            plan.map.push((a, q, rand.next(chunks as usize + 1) as u32));
        }
        for _ in 0..rand.next(2) {
            let (a, q) = at(&mut rand);
            plan.access.push((a, q..q + 1 + rand.next(2)));
        }
        let p = Placement::from_bytes(&model_bytes(&m)).expect("distinct chunks");
        let got = p.apply(&plan, chunks).map(|p| model(&p)).ok();
        assert_eq!(got, run(&m, &plan, chunks), "{m:?} under {plan:?}");
        match got {
            Some(_) => taken += 1,
            None => refused += 1,
        }
    }
    assert!(taken > 1000 && refused > 1000, "taken {taken}, refused {refused}");
}
